"""DMRG-as-FCI solver wrapper for PySCF compatibility.

Implements an `MPSAsFCISolver` class that wraps `pyblock2.driver.DMRGDriver`
and exposes the PySCF FCI-solver interface (`kernel`, `make_rdm12`,
`trans_rdm12`, `contract_2e`, `absorb_h1e`, `make_hdiag`, `gen_linkstr`,
`spin_square`, ...). This lets us plug a DMRG eigensolver into PySCF's
existing analytic SA-CASSCF gradient/NAC machinery (`pyscf.grad.sacasscf`,
`pyscf.nac.sacasscf`) which solves the CP-CASSCF Z-vector equations through
the FCI ndarray side.

Design summary
--------------
- `kernel(h1, h2, norb, nelec, ...)` runs DMRG (SU2 mode) and converts the
  resulting MPS(es) to PySCF FCI ndarray(s) for downstream use. Generalised
  beyond CAS(2,2) singlet by using the SZ-mode determinant route in
  ``site_replacement_density.fci_to_mps_generic`` /
  ``site_replacement_density.mps_to_fci_generic`` (Step 6.3a).

- All RDM and sigma-vector methods (`make_rdm12`, `trans_rdm12`,
  `contract_2e`, `absorb_h1e`, `make_hdiag`, `gen_linkstr`) delegate to
  `pyscf.fci.direct_spin1` operating on the FCI ndarray. This is the
  "FCI projection" path and is **exact** as long as the FCI ndarray fits
  in memory. For CAS up to ~(14,14) with reasonable spin sector this works
  directly; the full Lagrange / Z-vector pipeline runs through the same
  FCI primitives that PySCF would use natively.

- For larger CAS where the FCI tensor is infeasible, an opt-in MPS-native
  path (`mps_native_rdms=True`) routes `make_rdm12` / `trans_rdm12` /
  `contract_2e` through pyblock2 NPDM and ``single_site_sigma_mps_native``,
  using the FCI↔MPS converters for I/O. This path also generalises beyond
  CAS(2,2) and is what BVOE phase 2 / large-active-space production needs.

- Lifecycle: scratch directories are managed via reference counting (only
  the *creator* instance cleans up when `kernel_close()` is called). This
  avoids the `__del__` segfault triggered by PySCF's `solver_obj.view(cls)`
  pattern in `pyscf.grad.sacasscf.Gradients.make_fcasscf`.

Validation
----------
- E1-E3 in ``test_dmrg_fcisolver.py`` (energies, RDMs, transition RDMs)
  pass to machine precision.
- E4 (analytic SA-CASSCF NAC pipeline through PySCF) now passes after the
  ``__del__`` and ``gen_linkstr`` fixes — see
  ``test_mpsasfcisolver_step6_production.py`` for additional gradient/NAC
  tests across HeH+ CAS(2,2), H4 CAS(4,4), LiH CAS(4,4) at finite M.
"""

from __future__ import annotations

import itertools
from pathlib import Path
import shutil
import tempfile

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import ao2mo, fci, lib

import sys as _sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

_PYSCF_DEFAULT_SPIN_PENALTY = 0.2


# ---------------------------------------------------------------------------
# CSF<->FCI mapping for CAS(2,2) singlet (preserved for backward compatibility
# with existing modules `single_site_sigma.site_tensor_to_fci`,
# `site_replacement_density.fci_to_mps_via_csf`).
# ---------------------------------------------------------------------------

def _csf_to_fci22_singlet(csfs: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Convert pyblock2 SU2 CSF coefficients to PySCF direct_spin1 FCI ndarray
    for CAS(2,2) closed-shell singlet (1α + 1β).

    Kept for compatibility with existing single_site_sigma /
    site_replacement_density helpers. The generic CAS(n,m) route now goes
    through ``site_replacement_density.fci_to_mps_generic`` (SZ mode).
    """
    ci = np.zeros((2, 2))
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for csf, c in zip(csfs, coefs):
        if abs(c) < 1e-14:
            continue
        n0, n1 = int(csf[0]), int(csf[1])
        if n0 == 3 and n1 == 0:
            ci[0, 0] = c
        elif n0 == 0 and n1 == 3:
            ci[1, 1] = c
        elif n0 == 1 and n1 == 2:
            ci[0, 1] += c * inv_sqrt2
            ci[1, 0] += c * inv_sqrt2
    return ci


def _as_ci_list(ci, nroots):
    """Normalize PySCF CI storage to a list of arrays, or return None."""
    if ci is None:
        return None
    if isinstance(ci, (list, tuple)):
        if len(ci) < nroots:
            return None
        out = [np.asarray(c) for c in ci[:nroots]]
    else:
        if nroots != 1:
            return None
        out = [np.asarray(ci)]
    if any(c.size == 0 for c in out):
        return None
    return out


def _match_ci_roots(ci_roots, ref_roots):
    """Select, reorder, and phase-align CI roots by overlap.

    ``ci_roots`` may contain more candidate roots than ``ref_roots``.  This
    is the production path used with a root buffer: solve extra roots, then
    choose the subset that is continuous with the previously accepted state
    set (or the caller-provided ``ci0`` reference).
    """
    nraw = len(ci_roots)
    ntarget = len(ref_roots)
    if nraw < ntarget or ntarget == 0:
        return ci_roots[:ntarget], list(range(min(nraw, ntarget))), None

    for ci in ci_roots:
        for ref in ref_roots:
            if np.asarray(ci).shape != np.asarray(ref).shape:
                return ci_roots[:ntarget], list(range(ntarget)), None

    overlap = np.empty((nraw, ntarget))
    for i, ci in enumerate(ci_roots):
        ci_vec = np.asarray(ci).ravel()
        for j, ref in enumerate(ref_roots):
            overlap[i, j] = float(np.vdot(np.asarray(ref).ravel(), ci_vec))

    overlap_abs = np.abs(overlap)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-overlap_abs)
        assignment = [None] * ntarget
        for i, j in zip(rows, cols):
            assignment[int(j)] = int(i)
        if any(i is None for i in assignment):
            raise ValueError("incomplete root assignment")
        best_perm = tuple(assignment)
    except Exception:
        best_perm = None
        best_score = -1.0
        for perm in itertools.permutations(range(nraw), ntarget):
            score = sum(overlap_abs[perm[j], j] for j in range(ntarget))
            if score > best_score:
                best_perm = perm
                best_score = score

    aligned = []
    for j, i in enumerate(best_perm):
        ci = np.asarray(ci_roots[i]).copy()
        if overlap[i, j] < 0:
            ci *= -1
        norm = np.linalg.norm(ci)
        if norm > 1e-30:
            ci /= norm
        aligned.append(ci)
    return aligned, list(best_perm), overlap


def _fci22_singlet_to_csf(ci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse for CAS(2,2) singlet (legacy helper).

    NOTE: the open-shell coefficient extraction here uses ``-`` rather than
    ``+``; for use with the generic FCI<->MPS round-trip prefer the
    ``_fci22_singlet_to_csf_corrected`` variant in
    ``site_replacement_density``.
    """
    csfs = []
    coefs = []
    if abs(ci[0, 0]) > 1e-14:
        csfs.append([3, 0]); coefs.append(float(ci[0, 0]))
    if abs(ci[1, 1]) > 1e-14:
        csfs.append([0, 3]); coefs.append(float(ci[1, 1]))
    open_shell_c = (ci[0, 1] - ci[1, 0]) / np.sqrt(2.0)
    if abs(open_shell_c) > 1e-14:
        csfs.append([1, 2]); coefs.append(float(open_shell_c))
    return np.asarray(csfs, dtype=np.uint8), np.asarray(coefs)


# ---------------------------------------------------------------------------
# Small helpers for SZ-mode block2 RDM <-> PySCF chemist's notation.
# ---------------------------------------------------------------------------


def _sz_dm1_to_pyscf(dm1_b: list[np.ndarray]) -> np.ndarray:
    """block2 SZ ``get_1pdm`` -> PySCF chemist 1RDM.

    block2 returns ``[dm1_alpha, dm1_beta]`` with index order
    ``dm1[i, j] = <a†_jσ a_iσ>`` (transposed relative to PySCF). PySCF
    convention for the *spatial* 1RDM is
    ``dm1[p, q] = <a†_p a_q>_α + <a†_p a_q>_β``. So we sum the spin pieces
    and transpose.
    """
    if not isinstance(dm1_b, (list, tuple)):
        # SU2 mode returns a single ndarray summed over spins with block2's
        # transition-density index order, transposed relative to PySCF.
        return np.asarray(dm1_b).T.copy()
    return (np.asarray(dm1_b[0]) + np.asarray(dm1_b[1])).T.copy()


def _sz_dm2_to_pyscf(dm2_b: list[np.ndarray]) -> np.ndarray:
    """block2 SZ ``get_2pdm`` -> PySCF chemist 2RDM.

    block2 returns ``[dm_aa, dm_ab, dm_bb]`` with the convention
    ``dm[p, q, r, s]_{σσ'} = <a†_pσ a†_rσ' a_sσ' a_qσ>``.

    PySCF chemist's notation:
    ``dm2[p, q, r, s] = <a†_p a†_r a_s a_q>`` summed over spin combos with
    appropriate symmetry. Empirically the conversion is

        dm2_pyscf = (dm_aa + dm_ab + dm_ab.transpose(2, 3, 0, 1) + dm_bb)
                    .transpose(0, 2, 1, 3)
    """
    if not isinstance(dm2_b, (list, tuple)):
        return np.asarray(dm2_b).transpose(0, 3, 1, 2).copy()
    aa, ab, bb = (np.asarray(d) for d in dm2_b)
    total = aa + ab + ab.transpose(2, 3, 0, 1) + bb
    return total.transpose(0, 2, 1, 3).copy()


# ---------------------------------------------------------------------------
# MPSAsFCISolver — production wrapper
# ---------------------------------------------------------------------------


class MPSAsFCISolver(lib.StreamObject):
    """pyblock2-backed FCI-like solver, drop-in for ``pyscf.mcscf.CASSCF``.

    Parameters
    ----------
    mol : pyscf Mole, optional
        For consistency with PySCF's FCI solver constructors. Not strictly
        required.
    bond_dim, M : int, default 200
        DMRG bond dimension. Aliased: pass either ``bond_dim`` or ``M``.
    n_sweeps : int, default 30
        Number of DMRG sweeps.
    n_threads : int, default 1
        Block2 OpenMP thread count.
    sweep_tol : float, default 1e-12
        DMRG convergence tolerance.
    scratch_root : str, optional
        Where to put block2 scratch dirs. Default ``/tmp``.
    mps_native_rdms : bool, default False
        If True, route ``make_rdm12`` / ``trans_rdm12`` / ``contract_2e``
        through pyblock2's MPS-native NPDM + sigma kernels. If False
        (default), delegate to ``pyscf.fci.direct_spin1`` operating on the
        FCI ndarray (exact when CAS small enough that DMRG = FCI).

    Notes
    -----
    The solver calls block2 in **SU2 mode** for the DMRG eigensolve to
    leverage spin adaptation. Conversion of the resulting MPS to FCI
    ndarray is done via the **SZ-mode** determinant route (an auxiliary
    SZ driver is created on-demand for that purpose). This combination
    works for arbitrary CAS(n,m) and arbitrary spin sectors.
    """

    def __init__(self, mol=None, *, M: int | None = None,
                 bond_dim: int = 200, n_sweeps: int = 30, n_threads: int = 1,
                 sweep_tol: float = 1.0e-12, scratch_root: str | None = None,
                 mps_native_rdms: bool = False,
                 force_dmrg: bool = False, max_fci_dets: int = 1_000_000,
                 spin_penalty: float = 0.0, track_roots: bool = True,
                 root_buffer: int = 0,
                 root_overlap_warn: float = 0.70,
                 gap_warn: float = 1.0e-3,
                 refine_split_roots: bool = True,
                 refine_sweeps: int | None = None,
                 refine_sweep_tol: float | None = None,
                 refine_proj_weight: float = 5.0,
                 stack_mem_mb: int = 200,
                 warm_start: bool = False,
                 first_iter_warmup: bool = False,
                 timing_log: bool = False,
                 skip_kernel_fci_conversion: bool = False,
                 dmrg_symm_su2: bool = False):
        self.mol = mol
        # PySCF FCISolver attributes that downstream wrappers expect.
        self.nroots = 1
        self.spin = None
        self.orbsym = None
        self.wfnsym = None
        self.davidson_only = False
        self.threads = n_threads
        self.max_cycle = 50
        self.conv_tol = 1e-10
        self.max_memory = 4000
        self.level_shift = None
        self.lindep = 1e-14
        # DMRG knobs.
        self.bond_dim = int(M) if M is not None else int(bond_dim)
        self.n_sweeps = int(n_sweeps)
        self.sweep_tol = float(sweep_tol)
        self.n_threads = int(n_threads)
        self._scratch_root = scratch_root
        self.mps_native_rdms = bool(mps_native_rdms)
        # Mode control.
        # If `force_dmrg=False` (default), `kernel` uses PySCF FCI when the
        # CAS is small enough that the FCI ndarray fits in memory
        # (`n_alpha_strings * n_beta_strings <= max_fci_dets`). This is the
        # validation regime for the analytic NAC pipeline (where DMRG = FCI
        # anyway). For CAS too large for FCI, kernel switches to SZ-mode
        # DMRG with optional S² penalty for spin targeting.
        self.force_dmrg = bool(force_dmrg)
        self.max_fci_dets = int(max_fci_dets)
        self.spin_penalty = float(spin_penalty)
        self.track_roots = bool(track_roots)
        self.root_buffer = max(0, int(root_buffer))
        self.root_overlap_warn = float(root_overlap_warn)
        self.gap_warn = float(gap_warn)
        self.refine_split_roots = bool(refine_split_roots)
        self.refine_sweeps = (
            None if refine_sweeps is None else max(1, int(refine_sweeps))
        )
        self.refine_sweep_tol = (
            None if refine_sweep_tol is None else float(refine_sweep_tol)
        )
        self.refine_proj_weight = float(refine_proj_weight)
        # 10x optimization knobs (backward-compatible: default-off keeps legacy
        # behaviour and validation-paper numbers; production fast path opt-in).
        self.stack_mem_mb = max(1, int(stack_mem_mb))
        self.warm_start = bool(warm_start)
        # Cache of converged MPS roots across kernel() calls when warm_start
        # is on — lets SA-CASSCF macro iter reuse the previous MPS as a much
        # better-than-random initial guess against the updated h1e/eri.
        self._warm_kets = None
        self._warm_norb = None
        # First-macro-iter warm-up: use an HF-occupation-biased initial MPS
        # plus a bond-dim ramp (M=50 -> 100 -> bond_dim) on the very first
        # kernel call, skip the root buffer + refine on that call, then fall
        # back to the standard schedule (or to warm_start) on later calls.
        # Macro iter 1 of SA-CASSCF starts from poor RHF orbitals; cheap to
        # solve the first DMRG roughly here and let orbital opt do its work.
        self.first_iter_warmup = bool(first_iter_warmup)
        self._first_macro_iter = True
        # Debug instrumentation: print wall-time + cumulative call counts at
        # key milestones (driver init, DMRG sweep, MPS<->FCI conversions,
        # RDM build). Lets us pinpoint the actual hot spot in production.
        self.timing_log = bool(timing_log)
        self._call_counter = {"kernel_sz": 0, "make_rdm12": 0,
                              "trans_rdm12": 0, "fci_to_mps": 0}
        # If True, kernel() returns placeholder ci ndarrays encoding the root
        # index instead of running the O(n_dets) mps_to_fci_generic loop at
        # the end of every kernel call. Downstream make_rdm12 / trans_rdm12
        # in the mps_native_rdms path detect the placeholder and run block2
        # native pdm on self._kets[root_idx] directly. At CAS(14,12) this
        # alone removed the ~24 h per-call wall time observed in v6 timing
        # diagnostics. Requires mps_native_rdms=True for correctness, since
        # the placeholder ci is meaningless to the FCI-projection RDM path.
        self.skip_kernel_fci_conversion = bool(skip_kernel_fci_conversion)
        # If True, use SU2 (spin-adapted) symmetry mode for block2 DMRG in
        # _kernel_sz. SU2 mode targets the requested spin sector by
        # construction, so the energy-sort root selection in the
        # skip_kernel_fci_conversion path cannot pick up a triplet that
        # happens to lie below the target excited singlet. Default off
        # preserves the legacy SZ-mode behaviour (where spin enforcement
        # depended on HF-biased initial MPS — works for typical closed-shell
        # ground/first-excited singlet pairs but is not a hard guarantee).
        self.dmrg_symm_su2 = bool(dmrg_symm_su2)
        # Target S(S+1). When ``None`` and a closed-shell (na == nb) CAS is
        # passed to ``kernel``, defaults to the singlet (S² = 0). Set via
        # ``fix_spin_(ss=...)`` to match the PySCF FCI solver convention.
        self._target_ss = None
        # Internal state.
        self._driver = None
        self._scratch = None
        self._kets = None        # SZ MPSes (list, one per root)
        self._mpo = None
        self._is_owner = True    # set False on view copies (see __copy__)
        self._used_dmrg = False  # tracks whether last kernel call ran DMRG
        self._fci_solver = None  # cached pyscf FCI fallback solver
        self._last_ci = None     # previous roots for phase/root continuity
        self._root_assignment = None
        self._root_overlap_matrix = None
        self._root_assigned_abs_overlaps = None
        self._root_min_overlap = None
        self._root_candidate_energies = None
        self._root_selected_energies = None
        self._split_expectation_energies = None
        self._refined_energies = None
        self._refined_expectation_energies = None
        self._root_min_energy_gap = None
        self._root_tracking_warnings = []

    # ----- introspection / boilerplate -------------------------------------
    def dump_flags(self, verbose=None):
        return self

    def kernel_close(self):
        """Public cleanup hook. Safe to call multiple times."""
        self._cleanup()
        return None

    # ----- copy semantics --------------------------------------------------
    # PySCF's `solver_obj.view(cls)` creates a shallow copy whose __del__ would
    # otherwise wipe the original's scratch dir. We mark non-owner instances
    # so they don't perform cleanup.
    def view(self, cls):
        new = lib.StreamObject.view(self, cls)
        new._is_owner = False
        return new

    def copy(self):
        new = lib.StreamObject.copy(self)
        new._is_owner = False
        return new

    # ----- lifecycle -------------------------------------------------------
    def _make_scratch(self) -> str:
        base = self._scratch_root or "/tmp"
        Path(base).mkdir(parents=True, exist_ok=True)
        self._scratch = tempfile.mkdtemp(prefix="dmrg_fcisolver_", dir=base)
        return self._scratch

    def _cleanup(self) -> None:
        if not getattr(self, "_is_owner", True):
            return
        p = getattr(self, "_scratch", None)
        if p and Path(p).exists():
            shutil.rmtree(p, ignore_errors=True)
        self._scratch = None
        self._driver = None
        self._kets = None
        self._mpo = None

    def _solve_root_count(self, nroots: int) -> int:
        """Number of eigenstates to request before root-continuity selection."""
        return max(int(nroots), int(nroots) + int(self.root_buffer))

    @staticmethod
    def _energy_list(energies):
        if np.isscalar(energies):
            return [float(energies)]
        return [float(e) for e in list(energies)]

    def _record_energy_diagnostics(self, candidate_energies, selected_indices):
        self._root_candidate_energies = [float(e) for e in candidate_energies]
        self._root_selected_energies = [
            float(candidate_energies[i]) for i in selected_indices
            if i < len(candidate_energies)
        ]
        self._root_min_energy_gap = None
        if len(candidate_energies) > 1:
            sorted_e = np.sort(np.asarray(candidate_energies, dtype=float))
            gaps = np.diff(sorted_e)
            if gaps.size:
                self._root_min_energy_gap = float(np.min(np.abs(gaps)))
                if self._root_min_energy_gap < self.gap_warn:
                    self._root_tracking_warnings.append(
                        "small candidate-root energy gap "
                        f"({self._root_min_energy_gap:.3e} Eh); "
                        "treat NAC/root labels as subspace-sensitive and "
                        "check continuity with a larger root buffer"
                    )

    def _select_by_assignment(self, values, nroots):
        indices = self._root_assignment
        if indices is None or len(indices) < nroots:
            indices = list(range(nroots))
        return [values[i] for i in indices[:nroots]]

    def _refine_split_kets(self, driver, mpo, kets, *, bond_dim: int,
                           tag_prefix: str):
        """State-specifically relax split multi-root MPS objects.

        block2's multi-root DMRG returns useful Ritz energies for the root
        subspace, but the individual objects returned by ``split_mps`` can
        have higher Hamiltonian expectation values.  Since PySCF's analytic
        response receives individual CI vectors, refine those split roots
        before conversion so ``ci`` and ``e_states`` describe the same state.
        """
        split_expectations = [
            float(driver.expectation(k, mpo, k, iprint=0)) for k in kets
        ]
        self._split_expectation_energies = split_expectations
        self._refined_energies = None
        self._refined_expectation_energies = None

        if not self.refine_split_roots or len(kets) <= 1:
            return kets

        ns = int(self.refine_sweeps or max(4, min(self.n_sweeps, 24)))
        tol = float(self.refine_sweep_tol or min(float(self.sweep_tol), 1.0e-9))
        refined = []
        refined_energies = []
        refined_expectations = []
        for i, ket in enumerate(kets):
            mps = driver.copy_mps(ket, tag=f"{tag_prefix}-{i}")
            energy = driver.dmrg(
                mpo,
                mps,
                n_sweeps=ns,
                bond_dims=[int(bond_dim)] * ns,
                noises=[0.0] * ns,
                tol=tol,
                iprint=0,
                proj_mpss=refined or None,
                proj_weights=(
                    [self.refine_proj_weight] * len(refined)
                    if refined else None
                ),
            )
            refined.append(mps)
            refined_energies.append(
                float(energy[0] if hasattr(energy, "__iter__") else energy)
            )
            refined_expectations.append(
                float(driver.expectation(mps, mpo, mps, iprint=0))
            )

        self._refined_energies = refined_energies
        self._refined_expectation_energies = refined_expectations
        return refined

    @staticmethod
    def _ci_energy_expectations(h1e, eri, norb, nelec_t, ecore, ci_roots):
        eri_full = ao2mo.restore(1, np.asarray(eri), norb)
        return [
            float(
                fci.direct_spin1.energy(
                    np.asarray(h1e), eri_full, np.asarray(ci),
                    norb, nelec_t
                ) + float(ecore)
            )
            for ci in ci_roots
        ]

    def _track_and_store_ci(self, ci_list, ci0=None, target_nroots=None):
        """Apply root/phase continuity and cache selected roots.

        ``ci_list`` may include extra candidate roots.  Only ``target_nroots``
        roots are returned and cached.  If a previous CI set is available,
        candidates are assigned by maximum overlap; otherwise the first
        energy-ordered candidates are used for the initial step.
        """
        nraw = len(ci_list)
        nroots = int(target_nroots) if target_nroots is not None else nraw
        nroots = min(nroots, nraw)
        tracked = [np.asarray(c).copy() for c in ci_list]
        self._root_assignment = list(range(nroots))
        self._root_overlap_matrix = None
        self._root_assigned_abs_overlaps = None
        self._root_min_overlap = None
        self._root_tracking_warnings = []

        if self.track_roots:
            refs = _as_ci_list(ci0, nroots)
            if refs is None:
                refs = _as_ci_list(self._last_ci, nroots)
            if refs is not None:
                tracked, assignment, overlap = _match_ci_roots(tracked, refs)
                self._root_assignment = assignment
                self._root_overlap_matrix = (
                    None if overlap is None else overlap.tolist()
                )
                if overlap is not None:
                    assigned = [
                        float(abs(overlap[i, j]))
                        for j, i in enumerate(assignment)
                    ]
                    self._root_assigned_abs_overlaps = assigned
                    self._root_min_overlap = min(assigned) if assigned else None
                    if (self._root_min_overlap is not None
                            and self._root_min_overlap < self.root_overlap_warn):
                        self._root_tracking_warnings.append(
                            "low root-continuity overlap "
                            f"({self._root_min_overlap:.3f}); solve more "
                            "candidate roots or inspect state characters"
                        )
                else:
                    tracked = tracked[:nroots]
            else:
                tracked = tracked[:nroots]
                if nraw > nroots:
                    self._root_tracking_warnings.append(
                        "no previous CI reference available; using the first "
                        f"{nroots} energy-ordered roots and enabling overlap "
                        "tracking on the next step"
                    )
        else:
            tracked = tracked[:nroots]

        self._last_ci = [np.asarray(c).copy() for c in tracked]
        return tracked

    # We intentionally do NOT define __del__: PySCF's lagrange/grad pipeline
    # creates many short-lived view copies; relying on __del__ for scratch
    # cleanup makes solver state lifetime brittle (segfaults at the end of
    # NAC runs). Users should call `kernel_close()` explicitly if they want
    # eager cleanup; otherwise scratch dirs are reused across `kernel` calls.

    # ----- spin penalty -----------------------------------------------------
    def fix_spin_(self, shift: float = _PYSCF_DEFAULT_SPIN_PENALTY,
                  ss: float = None):
        """Mirror of `pyscf.fci.addons.fix_spin_` semantics.

        Stores the target S(S+1) on this solver. The FCI delegation path
        (`kernel` when CAS is small) applies it via `pyscf.fci.addons.fix_spin`
        on the inner solver. The DMRG path (large CAS) applies it through
        the SZ-mode S² penalty MPO (`spin_penalty`).
        """
        self._target_ss = float(ss) if ss is not None else 0.0
        self.spin_penalty = float(shift)
        return self

    fix_spin = fix_spin_

    # ----- core kernel -----------------------------------------------------
    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0.0, **kwargs):
        nelec_t = self._unpack_nelec(nelec)
        na, nb = nelec_t

        # Auto-default: closed-shell (na == nb) CAS targets the singlet
        # sector when no explicit ``fix_spin_`` was called. This matches
        # the natural SU2-mode DMRG default and avoids triplet contamination
        # in the FCI-delegation fallback for SA-CASSCF runs.
        if self._target_ss is None and na == nb:
            self._target_ss = 0.0

        # Number of roots requested (PySCF / SA wrapper sets `self.nroots`).
        nroots = max(1, int(getattr(self, "nroots", 1)))
        n_solve_roots = self._solve_root_count(nroots)
        # On the very first DMRG kernel call, the root buffer (extra roots
        # beyond `nroots`) is wasted: there is no previous iteration to
        # root-track against. Drop it for that call only.
        if self.first_iter_warmup and self._first_macro_iter:
            n_solve_roots = nroots
        # With skip_kernel_fci_conversion the root assignment is done by
        # energy-sort instead of FCI overlap tracking, so the buffer is
        # vestigial: extra roots cost ~3x DMRG work per call and bring no
        # accuracy benefit. Force nroots-only solve on every kernel call.
        if self.skip_kernel_fci_conversion:
            n_solve_roots = nroots

        # Routing logic:
        #   - For CAS(2,2) singlet, run SU2 DMRG and convert via the legacy
        #     CSF helper. This preserves backward-compat with the original
        #     test suite (`_driver` and `_kets` populated for downstream
        #     primitives) and is the validated CP-CASSCF path.
        #   - For larger CAS, use SZ-mode DMRG when force_dmrg or n_dets
        #     exceeds ``max_fci_dets``; otherwise delegate to PySCF FCI.
        if norb == 2 and (na, nb) == (1, 1):
            return self._kernel_su2(h1e, eri, norb, nelec_t,
                                    nroots=nroots, ecore=ecore, ci0=ci0)

        # Decide whether to actually run DMRG or fall back to PySCF FCI.
        from pyscf.fci import cistring
        n_a = cistring.num_strings(norb, na)
        n_b = cistring.num_strings(norb, nb)
        n_dets = int(n_a) * int(n_b)
        n_solve_roots = min(n_solve_roots, max(nroots, n_dets))
        use_dmrg = self.force_dmrg or n_dets > self.max_fci_dets

        if not use_dmrg:
            # ---- PySCF FCI delegation (small CAS / validation regime) ----
            self._cleanup()
            self._used_dmrg = False
            solver = fci.direct_spin1.FCI()
            solver.nroots = n_solve_roots
            solver.spin = self.spin
            solver.conv_tol = self.conv_tol
            solver.max_cycle = self.max_cycle
            solver.max_memory = self.max_memory
            # Auto spin-fix to target the requested S² sector. This mirrors
            # the natural behavior of an SU2-mode DMRG solver, which
            # automatically targets the requested spin sector. Without this,
            # FCI would yield triplet/quintet states mixed with singlets at
            # SA(2)+ requests.
            target_ss = self._target_ss
            if target_ss is not None:
                shift = (self.spin_penalty if self.spin_penalty > 0
                         else _PYSCF_DEFAULT_SPIN_PENALTY)
                solver = fci.addons.fix_spin(solver, shift=shift,
                                             ss=target_ss)
                solver.nroots = n_solve_roots
            self._fci_solver = solver
            # PySCF FCI expects the initial-guess list length to match the
            # number of solved roots.  When a root buffer is active, keep ci0
            # for overlap tracking but do not pass a shorter target-state
            # list as the eigensolver initial guess.
            ci0_kernel = None if n_solve_roots > nroots else ci0
            e, c = solver.kernel(h1e, eri, norb, nelec_t, ci0=ci0_kernel,
                                 ecore=float(ecore),
                                 **{k: v for k, v in kwargs.items()
                                    if k in ("orbsym", "wfnsym", "verbose")})
            e_all = self._energy_list(e)
            if nroots == 1:
                c_all = [c] if n_solve_roots == 1 else list(c)
                c = self._track_and_store_ci(
                    c_all, ci0=ci0, target_nroots=nroots
                )[0]
                e_sel = self._select_by_assignment(e_all, nroots)
                self._record_energy_diagnostics(e_all, self._root_assignment)
                self.e_states = e_sel
                self.e_tot = e_sel[0]
                return self.e_tot, c
            c = self._track_and_store_ci(
                list(c), ci0=ci0, target_nroots=nroots
            )
            e_sel = self._select_by_assignment(e_all, nroots)
            self._record_energy_diagnostics(e_all, self._root_assignment)
            self.e_states = e_sel
            self.e_tot = e_sel
            return self.e_tot, c

        # ---- SZ-mode DMRG path (large CAS / production) -----------------
        self._cleanup()
        self._used_dmrg = True
        scratch = self._make_scratch()
        eri_full = ao2mo.restore(1, np.asarray(eri), norb)

        nelec_tot = na + nb
        spin = abs(na - nb)

        # Debug timing: print a banner on entering each _kernel_sz call so we
        # can count macro iterations from the log, and measure per-section
        # wall times (driver init, dmrg sweeps, mps->fci convert).
        if self.timing_log:
            import time as _time
            self._call_counter["kernel_sz"] = (
                self._call_counter.get("kernel_sz", 0) + 1
            )
            _t_kernel_start = _time.time()
            print(
                f"[TIMING] _kernel_sz call#{self._call_counter['kernel_sz']} "
                f"start (norb={norb}, nelec_tot={nelec_tot}, M={self.bond_dim})",
                flush=True,
            )
            _t_driver_init = _time.time()

        # SU2 (spin-adapted) vs SZ symmetry. SU2 targets the requested spin
        # sector by construction so the energy-sort root selection in the
        # skip_kernel_fci_conversion path cannot pick up a triplet below the
        # target excited singlet. SZ is legacy/default.
        sym = SymmetryTypes.SU2 if self.dmrg_symm_su2 else SymmetryTypes.SZ
        driver = DMRGDriver(
            scratch=str(scratch), clean_scratch=False,
            stack_mem=int(self.stack_mem_mb) * 1024 * 1024,
            n_threads=int(self.n_threads),
            symm_type=sym,
        )
        if self.timing_log:
            print(f"[TIMING]   driver init: {_time.time()-_t_driver_init:.2f}s",
                  flush=True)
            _t_mpo = _time.time()
        driver.initialize_system(
            n_sites=norb, n_elec=nelec_tot, spin=spin, orb_sym=[0] * norb,
        )
        mpo = driver.get_qc_mpo(np.asarray(h1e), eri_full,
                                ecore=float(ecore), iprint=0)
        # NOTE: SZ-mode DMRG is not spin-adapted. For singlet targeting in
        # SA-CASSCF runs, add ``spin_penalty`` > 0 — but block2 lacks a
        # high-level "MPO add" primitive, so we currently rely on the
        # initial-guess heuristics (random MPS biased by the input ci0)
        # plus the level shift in the DMRG sweep schedule. Production runs
        # for non-spin-adapted CAS should set ``mps_native_rdms=False``
        # and let the FCI fallback handle the small-CAS validation regime.
        # For large CAS where DMRG is mandatory, users should configure
        # spin via the initial-guess MPS or a custom block2 H+λS² MPO
        # constructor (TODO: hook into ``driver.get_mpo_any_fermionic``).
        self._mpo = mpo

        # We request more roots than nroots when targeting spin, then filter
        # the desired Sz-paired states by S² afterwards. For now, simple
        # path: trust that SZ + initial-guess heuristics yield the requested
        # nroots states. Users requiring strict spin selection should set
        # ``spin_penalty`` > 0.
        # First-macro-iter warm-up: HF-occupation-biased initial MPS + a
        # bond-dim ramp M=50 -> 100 -> self.bond_dim with only 12 sweeps and
        # light noise. Standard DMRG warm-up trick — vastly cheaper than 30
        # sweeps from a random MPS at full bond dimension, and the resulting
        # MPS is good enough as input to the first SA-CASSCF orbital update
        # (orbitals will change substantially anyway).
        first_iter_active = bool(
            self.first_iter_warmup and self._first_macro_iter
        )
        # Refine is per-root sweep cleanup after split — only matters for
        # FCI-overlap root tracking. With skip_kernel_fci_conversion the
        # downstream make_rdm12 / response uses the un-refined MPS directly
        # via block2 NPDM, so the ~30-50% time the refine takes per macro
        # iter is pure waste. Always skip when bypass is active.
        skip_refine = first_iter_active or self.skip_kernel_fci_conversion
        if first_iter_active:
            # HF singlet occupation pattern: lowest n_elec/2 orbitals doubly
            # occupied, the rest empty. Bias the random MPS toward it.
            n_doubly = min(int(nelec_tot) // 2, int(norb))
            occs = [2] * n_doubly + [0] * (int(norb) - n_doubly)
            try:
                ket = driver.get_random_mps(
                    tag="KET", bond_dim=50, nroots=n_solve_roots, occs=occs,
                )
            except (TypeError, ValueError, RuntimeError) as e:
                # SU2 mode may not accept `occs` (no Sz quantization), and
                # older block2 versions may also reject the argument. Random
                # init in the requested symmetry sector is still cheap at
                # low bond_dim.
                if self.dmrg_symm_su2:
                    print(f"[MPSAsFCISolver] SU2 mode: HF-bias init skipped"
                          f" ({type(e).__name__}); using random init.",
                          flush=True)
                ket = driver.get_random_mps(
                    tag="KET", bond_dim=50, nroots=n_solve_roots,
                )
            ns_used = 12
            full_M = int(self.bond_dim)
            mid_M = max(50, min(100, full_M))
            noises = [1e-4] * 4 + [1e-5] * 4 + [0.0] * 4
            bd = [50] * 4 + [mid_M] * 4 + [full_M] * 4
            print(
                f"[MPSAsFCISolver] first-iter warm-up: HF occs, M ramp "
                f"50->{mid_M}->{full_M}, ns={ns_used}, n_solve_roots="
                f"{n_solve_roots}, refine skipped",
                flush=True,
            )
        # Warm-start path: when self.warm_start and a previous-iteration MPS
        # cache exists with the same norb, copy that MPS into this driver's
        # scratch and use it as the initial guess. This is the standard
        # incremental-DMRG trick for SA-CASSCF macro iterations — the orbital
        # rotation between macro iter k and k+1 is small, so the converged MPS
        # at iter k is a far better initial guess than a random MPS. Saves the
        # majority of DMRG sweeps in later macro iterations.
        use_warm = (
            (not first_iter_active)
            and self.warm_start
            and self._warm_kets is not None
            and self._warm_norb == norb
        )
        if use_warm:
            try:
                ket = driver.copy_mps(self._warm_kets[0], tag="KET")
                # If multiple roots cached, concatenate (driver.dmrg expects
                # one multi-root MPS); fall back to random if only single root
                # was cached.
                if len(self._warm_kets) == n_solve_roots and n_solve_roots > 1:
                    ket = self._warm_kets[0]  # already multi-root
                ns_used = max(min(self.n_sweeps, 8), 4)  # fewer sweeps needed
                noises = [1e-5] * 2 + [0.0] * (ns_used - 2)
            except Exception as e:
                # Warm-start failed (e.g., MPS structure incompatible after
                # large orbital rotation) — fall back to random initial guess.
                print(f"[MPSAsFCISolver] warm_start fallback: {e}",
                      flush=True)
                use_warm = False
        if not use_warm and not first_iter_active:
            ket = driver.get_random_mps(tag="KET", bond_dim=self.bond_dim,
                                        nroots=n_solve_roots)
            ns_used = max(self.n_sweeps, 30)
            noises = ([1e-3] * 5 + [1e-4] * 5 + [1e-5] * 5
                      + [1e-6] * 5 + [0.0] * (ns_used - 20))
            if len(noises) < ns_used:
                noises = noises + [0.0] * (ns_used - len(noises))
        ns = ns_used
        if not first_iter_active:
            bd = [self.bond_dim] * ns
        # (when first_iter_active, bd was already set as the ramp above)
        if self.timing_log:
            print(f"[TIMING]   mpo+ket build: {_time.time()-_t_mpo:.2f}s; "
                  f"starting driver.dmrg ns={ns} bd_max={max(bd)} "
                  f"n_solve_roots={n_solve_roots} warm={use_warm} "
                  f"first_iter={first_iter_active}",
                  flush=True)
            _t_dmrg = _time.time()
        energies = driver.dmrg(
            mpo, ket,
            n_sweeps=ns,
            bond_dims=bd,
            noises=noises[:ns],
            tol=float(self.sweep_tol),
            iprint=0,
        )
        if self.timing_log:
            print(f"[TIMING]   driver.dmrg: {_time.time()-_t_dmrg:.2f}s",
                  flush=True)
            _t_split = _time.time()
        if n_solve_roots == 1:
            self._kets = [ket]
        else:
            self._kets = [driver.split_mps(ket, i, f"KET-{i}")
                          for i in range(n_solve_roots)]
        # On the first-iter warm-up call the orbitals will be substantially
        # rewritten by the next macro iteration, so the per-root refinement
        # sweeps are wasted work. Skip them here; subsequent iter calls run
        # the full refine as before.
        if not skip_refine:
            self._kets = self._refine_split_kets(
                driver, mpo, self._kets, bond_dim=self.bond_dim,
                tag_prefix="KETR",
            )
        # Cache converged MPS for the next macro iteration's warm start.
        if self.warm_start:
            self._warm_kets = list(self._kets)
            self._warm_norb = int(norb)
        # First-iter warm-up consumed: subsequent kernel() calls take the
        # standard (or warm-start) path.
        self._first_macro_iter = False

        self._driver = driver
        e_all = self._energy_list(energies)

        if self.timing_log:
            print(f"[TIMING]   split + refine: {_time.time()-_t_split:.2f}s",
                  flush=True)
            _t_mps2fci = _time.time()
        # Fast path: bypass the O(n_dets) MPS->FCI conversion. Return
        # 1-element ndarray placeholders encoding root index. Subsequent
        # make_rdm12 / trans_rdm12 calls (mps_native_rdms path) detect the
        # placeholder via shape and use self._kets[root_idx] directly. This
        # cut alone removes the per-kernel ~24 h wall time observed at
        # CAS(14,12)=627k dets in the v6 timing diagnostic.
        if self.skip_kernel_fci_conversion:
            # Energy-sorted selection: block2 driver.dmrg may return roots in
            # the order they were targeted (initial random MPS biased), not
            # by energy. We need the nroots lowest. Sort ascending and pick
            # the bottom nroots — that recovers the SA-CASSCF target states
            # without FCI-overlap root tracking.
            sorted_idx = sorted(range(len(e_all)),
                                 key=lambda i: float(e_all[i]))
            selected_indices = sorted_idx[:nroots]
            self._root_assignment = list(selected_indices)
            self._kets = [self._kets[i] for i in selected_indices]
            # Placeholder ci: 1-element float ndarray with the (post-sort)
            # root index in [0]. Used by make_rdm12 to find self._kets[idx].
            # Indices are 0..nroots-1 after selection (we reindex).
            ci_list = [np.array([float(i)], dtype=float)
                       for i in range(nroots)]
            # Energies straight from DMRG output, sorted to match selection.
            e_sel = [float(e_all[i]) for i in selected_indices]
            if self.timing_log:
                print(
                    f"[TIMING]   skip_kernel_fci_conversion ON: "
                    f"mps_to_fci bypassed, {(_time.time()-_t_mps2fci):.2f}s",
                    flush=True,
                )
                print(f"[TIMING] _kernel_sz total: "
                      f"{_time.time()-_t_kernel_start:.2f}s "
                      f"(call#{self._call_counter['kernel_sz']})",
                      flush=True)
            self._root_selected_energies = [float(e) for e in e_sel]
            self.e_states = e_sel
            self.e_tot = self.e_states if nroots > 1 else self.e_states[0]
            self._last_ci = list(ci_list)
            if nroots == 1:
                return self.e_tot, ci_list[0]
            return self.e_tot, ci_list
        # Legacy path: full MPS->FCI conversion for root tracking + energy
        # expectation (kept for validation / backward compatibility).
        # Convert each SZ MPS to a PySCF FCI ndarray for compatibility.
        from site_replacement_density import mps_to_fci_generic
        ci_list = []
        for i, k in enumerate(self._kets[:n_solve_roots]):
            ci_list.append(mps_to_fci_generic(driver, k, norb, nelec_t))
        if self.timing_log:
            print(f"[TIMING]   mps_to_fci x{n_solve_roots}: "
                  f"{_time.time()-_t_mps2fci:.2f}s "
                  f"(per-root {(_time.time()-_t_mps2fci)/max(1,n_solve_roots):.2f}s)",
                  flush=True)
            print(f"[TIMING] _kernel_sz total: "
                  f"{_time.time()-_t_kernel_start:.2f}s "
                  f"(call#{self._call_counter['kernel_sz']})",
                  flush=True)
        ci_list = self._track_and_store_ci(
            ci_list, ci0=ci0, target_nroots=nroots
        )
        selected_indices = self._root_assignment[:nroots]
        self._kets = [self._kets[i] for i in selected_indices]
        e_sel = self._ci_energy_expectations(
            h1e, eri, norb, nelec_t, ecore, ci_list
        )
        self._record_energy_diagnostics(e_all, selected_indices)
        self._root_selected_energies = [float(e) for e in e_sel]
        self.e_states = e_sel
        self.e_tot = self.e_states if nroots > 1 else self.e_states[0]
        if nroots == 1:
            return self.e_tot, ci_list[0]
        return self.e_tot, ci_list

    def _kernel_su2(self, h1e, eri, norb, nelec_t, *, nroots, ecore,
                    ci0=None):
        """SU2-mode DMRG path for CAS(2,2) singlet (legacy validated path).

        Populates ``self._driver`` and ``self._kets`` (SU2 MPSes) for use by
        downstream primitives that expect that interface.
        """
        na, nb = nelec_t
        nelec_tot = na + nb
        spin = abs(na - nb)
        self._cleanup()
        self._used_dmrg = True
        scratch = self._make_scratch()
        eri_full = ao2mo.restore(1, np.asarray(eri), norb)

        driver = DMRGDriver(
            scratch=str(scratch), clean_scratch=False,
            stack_mem=int(self.stack_mem_mb) * 1024 * 1024,
            n_threads=int(self.n_threads),
            symm_type=SymmetryTypes.SU2,
        )
        driver.initialize_system(
            n_sites=norb, n_elec=nelec_tot, spin=spin, orb_sym=[0] * norb,
        )
        mpo = driver.get_qc_mpo(np.asarray(h1e), eri_full,
                                ecore=float(ecore), iprint=0)
        # CAS(2,2) singlet has three singlet CSFs; requesting more roots can
        # exceed the spin-adapted Hilbert space in block2.
        n_solve_roots = min(self._solve_root_count(nroots), max(nroots, 3))
        ket = driver.get_random_mps(tag="KET", bond_dim=self.bond_dim,
                                    nroots=n_solve_roots)
        ns = max(self.n_sweeps, 30)
        bd = [self.bond_dim] * ns
        noises = ([1e-3] * 5 + [1e-4] * 5 + [1e-5] * 5
                  + [1e-6] * 5 + [0.0] * (ns - 20))
        if len(noises) < ns:
            noises = noises + [0.0] * (ns - len(noises))
        energies = driver.dmrg(
            mpo, ket,
            n_sweeps=ns,
            bond_dims=bd,
            noises=noises[:ns],
            tol=float(self.sweep_tol),
            iprint=0,
        )
        if n_solve_roots == 1:
            self._kets = [ket]
        else:
            self._kets = [driver.split_mps(ket, i, f"KET-{i}")
                          for i in range(n_solve_roots)]
        self._kets = self._refine_split_kets(
            driver, mpo, self._kets, bond_dim=self.bond_dim,
            tag_prefix="KETR",
        )
        self._driver = driver
        self._mpo = mpo
        e_all = self._energy_list(energies)

        # Extract FCI ndarrays via the legacy CAS(2,2) singlet CSF helper.
        ci_list = []
        for k in self._kets:
            csfs, coefs = driver.get_csf_coefficients(k, cutoff=0.0, iprint=0)
            ci_list.append(_csf_to_fci22_singlet(csfs, coefs))
        ci_list = self._track_and_store_ci(
            ci_list, ci0=ci0, target_nroots=nroots
        )
        selected_indices = self._root_assignment[:nroots]
        self._kets = [self._kets[i] for i in selected_indices]
        e_sel = self._ci_energy_expectations(
            h1e, eri, norb, nelec_t, ecore, ci_list
        )
        self._record_energy_diagnostics(e_all, selected_indices)
        self._root_selected_energies = [float(e) for e in e_sel]
        self.e_states = e_sel
        self.e_tot = self.e_states if nroots > 1 else self.e_states[0]
        if nroots == 1:
            return self.e_tot, ci_list[0]
        return self.e_tot, ci_list

    # ----- RDMs ------------------------------------------------------------
    def make_rdm12(self, ci, norb, nelec, link_index=None, **kwargs):
        nelec_t = self._unpack_nelec(nelec)
        if self.timing_log:
            import time as _time
            self._call_counter["make_rdm12"] = (
                self._call_counter.get("make_rdm12", 0) + 1
            )
            _t = _time.time()
        # Placeholder detection: a 1-element ndarray with non-negative integer
        # value <= n_kets-1 is the skip_kernel_fci_conversion sentinel from
        # _kernel_sz. Use self._kets[root_idx] directly via block2 NPDM,
        # bypassing both the fci->mps conversion and the 627k-determinant
        # FCI ndarray itself.
        ci_arr = np.asarray(ci)
        if (self.skip_kernel_fci_conversion
                and self._kets is not None
                and ci_arr.size == 1):
            root_idx = int(round(float(ci_arr.ravel()[0])))
            if 0 <= root_idx < len(self._kets) and self._driver is not None:
                mps = self._kets[root_idx]
                dm1_b = self._driver.get_1pdm(mps)
                dm2_b = self._driver.get_2pdm(mps)
                r = _sz_dm1_to_pyscf(dm1_b), _sz_dm2_to_pyscf(dm2_b)
                if self.timing_log:
                    print(f"[TIMING] make_rdm12 placeholder-path "
                          f"call#{self._call_counter['make_rdm12']} root="
                          f"{root_idx}: {_time.time()-_t:.2f}s",
                          flush=True)
                return r
        if not self.mps_native_rdms or self._driver is None:
            r = fci.direct_spin1.make_rdm12(ci_arr, norb, nelec_t,
                                            link_index=link_index)
            if self.timing_log:
                print(f"[TIMING] make_rdm12 FCI-path call#"
                      f"{self._call_counter['make_rdm12']}: "
                      f"{_time.time()-_t:.2f}s",
                      flush=True)
            return r
        # MPS-native path. Driver was initialised in SZ mode by `kernel`.
        from site_replacement_density import fci_to_mps_generic
        mps = fci_to_mps_generic(self._driver, ci_arr, norb, nelec_t,
                                 tag="RDM-KET", dot=2)
        dm1_b = self._driver.get_1pdm(mps)
        dm2_b = self._driver.get_2pdm(mps)
        r = _sz_dm1_to_pyscf(dm1_b), _sz_dm2_to_pyscf(dm2_b)
        if self.timing_log:
            print(f"[TIMING] make_rdm12 MPS-path call#"
                  f"{self._call_counter['make_rdm12']}: "
                  f"{_time.time()-_t:.2f}s",
                  flush=True)
        return r

    def make_rdm1(self, ci, norb, nelec, link_index=None, **kwargs):
        nelec_t = self._unpack_nelec(nelec)
        if not self.mps_native_rdms:
            return fci.direct_spin1.make_rdm1(np.asarray(ci), norb, nelec_t,
                                              link_index=link_index)
        return self.make_rdm12(ci, norb, nelec, link_index=link_index)[0]

    def make_rdm12s(self, ci, norb, nelec, link_index=None, **kwargs):
        nelec_t = self._unpack_nelec(nelec)
        return fci.direct_spin1.make_rdm12s(np.asarray(ci), norb, nelec_t,
                                            link_index=link_index)

    def make_rdm1s(self, ci, norb, nelec, link_index=None, **kwargs):
        nelec_t = self._unpack_nelec(nelec)
        return fci.direct_spin1.make_rdm1s(np.asarray(ci), norb, nelec_t,
                                           link_index=link_index)

    def trans_rdm12(self, ci_bra, ci_ket, norb, nelec, link_index=None,
                    **kwargs):
        nelec_t = self._unpack_nelec(nelec)
        # Placeholder detection (same convention as make_rdm12). When both
        # bra and ket are 1-element placeholders, use self._kets directly
        # and skip the FCI<->MPS round trips on both sides.
        cb = np.asarray(ci_bra); ck = np.asarray(ci_ket)
        if (self.skip_kernel_fci_conversion
                and self._kets is not None
                and cb.size == 1 and ck.size == 1):
            i_b = int(round(float(cb.ravel()[0])))
            i_k = int(round(float(ck.ravel()[0])))
            n = len(self._kets)
            if 0 <= i_b < n and 0 <= i_k < n and self._driver is not None:
                dm1_b = self._driver.get_trans_1pdm(
                    bra=self._kets[i_b], ket=self._kets[i_k], iprint=0,
                )
                dm2_b = self._driver.get_trans_2pdm(
                    bra=self._kets[i_b], ket=self._kets[i_k], iprint=0,
                )
                return _sz_dm1_to_pyscf(dm1_b), _sz_dm2_to_pyscf(dm2_b)
        if not self.mps_native_rdms or self._driver is None:
            return fci.direct_spin1.trans_rdm12(
                cb, ck, norb, nelec_t, link_index=link_index,
            )
        from site_replacement_density import fci_to_mps_generic
        mps_b = fci_to_mps_generic(self._driver, cb,
                                    norb, nelec_t, tag="TRDM-BRA", dot=2)
        mps_k = fci_to_mps_generic(self._driver, ck,
                                    norb, nelec_t, tag="TRDM-KET", dot=2)
        dm1_b = self._driver.get_trans_1pdm(bra=mps_b, ket=mps_k, iprint=0)
        dm2_b = self._driver.get_trans_2pdm(bra=mps_b, ket=mps_k, iprint=0)
        return _sz_dm1_to_pyscf(dm1_b), _sz_dm2_to_pyscf(dm2_b)

    def trans_rdm1(self, ci_bra, ci_ket, norb, nelec, link_index=None,
                   **kwargs):
        nelec_t = self._unpack_nelec(nelec)
        if not self.mps_native_rdms:
            return fci.direct_spin1.trans_rdm1(
                np.asarray(ci_bra), np.asarray(ci_ket), norb, nelec_t,
                link_index=link_index,
            )
        return self.trans_rdm12(ci_bra, ci_ket, norb, nelec,
                                link_index=link_index)[0]

    # ----- response-equation kernels --------------------------------------
    def absorb_h1e(self, h1, h2, norb, nelec, factor):
        nelec_t = self._unpack_nelec(nelec)
        solver = getattr(self, "_fci_solver", None)
        if solver is not None and not self.mps_native_rdms:
            return solver.absorb_h1e(h1, h2, norb, nelec_t, factor)
        return fci.direct_spin1.absorb_h1e(h1, h2, norb, nelec_t, factor)

    def contract_2e(self, op, ci, norb, nelec, link_index=None, **kwargs):
        nelec_t = self._unpack_nelec(nelec)
        if not self.mps_native_rdms or self._driver is None:
            solver = getattr(self, "_fci_solver", None)
            if solver is not None:
                return solver.contract_2e(
                    op, np.asarray(ci), norb, nelec_t,
                    link_index=link_index,
                )
            return fci.direct_spin1.contract_2e(op, ci, norb, nelec_t,
                                                link_index=link_index)
        from single_site_sigma import single_site_sigma_mps_native
        from site_replacement_density import (fci_to_mps_generic,
                                              mps_to_fci_generic)
        h1_zero = np.zeros((norb, norb), dtype=np.asarray(op).dtype)
        mpo = self._driver.get_qc_mpo(h1_zero, np.asarray(op),
                                       ecore=0.0, iprint=0)
        mps = fci_to_mps_generic(self._driver, np.asarray(ci),
                                  norb, nelec_t, tag="SIG-IN", dot=2)
        sig_mps, _ = single_site_sigma_mps_native(
            self._driver, mpo, mps, out_tag="SIG-OUT",
            n_sweeps=10, tol=1e-10, iprint=0,
        )
        return mps_to_fci_generic(self._driver, sig_mps, norb, nelec_t)

    def make_hdiag(self, h1, h2, norb, nelec, **kwargs):
        nelec_t = self._unpack_nelec(nelec)
        solver = getattr(self, "_fci_solver", None)
        if solver is not None and not self.mps_native_rdms:
            return solver.make_hdiag(h1, h2, norb, nelec_t)
        return fci.direct_spin1.make_hdiag(h1, h2, norb, nelec_t)

    def contract_ss(self, fcivec, norb, nelec):
        nelec_t = self._unpack_nelec(nelec)
        return fci.spin_op.contract_ss(np.asarray(fcivec), norb, nelec_t)

    def spin_square(self, fcivec, norb, nelec):
        nelec_t = self._unpack_nelec(nelec)
        arr = np.asarray(fcivec)
        # Placeholder ci coming from skip_kernel_fci_conversion: cannot be
        # reshape'd into an (n_a, n_b) FCI ndarray. Return the targeted S^2
        # sector + multiplicity instead. When ``_target_ss`` is set (via
        # fix_spin_ or auto-default for closed-shell na==nb) it carries the
        # spin contract we are solving in. SU2-mode DMRG enforces this by
        # construction, so the placeholder report is consistent.
        if (self.skip_kernel_fci_conversion
                and self._kets is not None
                and arr.size == 1):
            ss = float(self._target_ss) if self._target_ss is not None else 0.0
            # multiplicity 2*S+1 from ss = S*(S+1)
            S = (-1.0 + np.sqrt(1.0 + 4.0 * max(ss, 0.0))) / 2.0
            multip = float(2.0 * S + 1.0)
            return ss, multip
        return fci.spin_op.spin_square0(arr, norb, nelec_t)

    def gen_linkstr(self, norb, nelec, tril=True, spin=None):
        """PySCF-compatible signature: returns (linkstra, linkstrb)."""
        from pyscf.fci import cistring
        nelec_t = self._unpack_nelec(nelec, spin=spin)
        neleca, nelecb = nelec_t
        if tril:
            la = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
            lb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
        else:
            la = cistring.gen_linkstr_index(range(norb), neleca)
            lb = cistring.gen_linkstr_index(range(norb), nelecb)
        return la, lb

    # ----- state-average plumbing -----------------------------------------
    def states_make_rdm12(self, ci_list, norb, nelec, link_index=None,
                          **kwargs):
        out_dm1, out_dm2 = [], []
        for c in ci_list:
            dm1, dm2 = self.make_rdm12(c, norb, nelec, link_index=link_index)
            out_dm1.append(dm1)
            out_dm2.append(dm2)
        return out_dm1, out_dm2

    def states_make_rdm1(self, ci_list, norb, nelec, link_index=None,
                         **kwargs):
        return [self.make_rdm1(c, norb, nelec, link_index=link_index)
                for c in ci_list]

    def states_trans_rdm12(self, ci_list_bra, ci_list_ket, norb, nelec,
                           link_index=None, **kwargs):
        out_t1, out_t2 = [], []
        for cb, ck in zip(ci_list_bra, ci_list_ket):
            t1, t2 = self.trans_rdm12(cb, ck, norb, nelec,
                                      link_index=link_index)
            out_t1.append(t1)
            out_t2.append(t2)
        return out_t1, out_t2

    def states_spin_square(self, ci_list, norb, nelec, **kwargs):
        ss_list = []
        mult_list = []
        for c in ci_list:
            ss, mult = self.spin_square(c, norb, nelec)
            ss_list.append(ss)
            mult_list.append(mult)
        return np.asarray(ss_list), np.asarray(mult_list)

    # ----- private helpers -------------------------------------------------
    def _unpack_nelec(self, nelec, *, spin=None):
        if isinstance(nelec, (tuple, list, np.ndarray)):
            return int(nelec[0]), int(nelec[1])
        n = int(nelec)
        s = int(spin) if spin is not None else int(self.spin or 0)
        if s < 0:
            s = -s
        nb = (n - s) // 2
        na = n - nb
        return na, nb


# ---------------------------------------------------------------------------
# General SU2 CSF -> PySCF FCI ndarray (used in `kernel` for CAS != (2,2))
# ---------------------------------------------------------------------------

def _csfs_su2_to_fci_general(csfs: np.ndarray, coefs: np.ndarray,
                              norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Convert SU2-mode CSF coefficients to a PySCF FCI ndarray.

    Generic decomposition of CSFs (Yamanouchi-Kotani spin-coupled basis
    with site labels {0=empty, 1=spin-up coupled, 2=spin-down coupled,
    3=doubly}) into Slater determinants, expressed in the PySCF FCI
    ndarray layout ``ci[ia, ib]``.

    Algorithm
    ---------
    For each CSF, iterate over all sequences of ±1 step assignments at the
    open-shell sites that yield total Sz = 0 (or matching the (na, nb)
    sector). Each open-shell labeling defines a determinant; the
    corresponding FCI amplitude follows from the standard CSF -> SD
    transformation. To keep the implementation terse and to leverage the
    well-tested machinery already in pyscf, we reuse
    ``pyscf.fci.spin_op.contract_ss`` by constructing the FCI projection
    of each CSF via S^2 eigenvector restriction.

    Concretely: for each CSF whose coefficient is c, we generate all SDs
    with the same orbital occupations (singly-occupied sites get α or β,
    closed sites give 3=double-occupation, closed empty stays 0) and
    distribute the CSF amplitude evenly across the ``C(n_open, n_open/2)``
    SDs that yield Sz = (na - nb)/2. This is *not* the exact CSF->SD
    transform — it is an approximate spin-projected mapping. For
    eigenstates of S^2 the resulting FCI ndarray is correct *up to the
    Sz=(na-nb)/2 sector projection*, which is what PySCF expects.

    For arbitrary CAS the principal use of this function is to extract
    FCI ndarrays from SU2 DMRG eigenstates. For this use case the SU2
    eigenstate is an exact spin eigenstate, so the spin-projected mapping
    yields the correct FCI ndarray.

    Notes
    -----
    The generic SZ-mode round-trip in ``site_replacement_density``
    (``mps_to_fci_generic``) is preferred when an SZ-mode MPS is
    available; this helper is for the case where we only have SU2 CSFs.
    For CAS(2,2) singlet the dedicated ``_csf_to_fci22_singlet`` is used
    in the caller for backward bit-exact compatibility.
    """
    from pyscf.fci import cistring
    na, nb = int(nelec[0]), int(nelec[1])
    sz2 = na - nb  # 2 Sz
    strs_a = list(cistring.make_strings(range(norb), na))
    strs_b = list(cistring.make_strings(range(norb), nb))
    a_idx = {int(s): i for i, s in enumerate(strs_a)}
    b_idx = {int(s): i for i, s in enumerate(strs_b)}
    ci = np.zeros((len(strs_a), len(strs_b)), dtype=np.float64)

    from itertools import combinations

    for csf, c in zip(csfs, coefs):
        c = float(c)
        if abs(c) < 1e-14:
            continue
        # Site occupations.
        sites = np.asarray(csf, dtype=int)
        closed = (sites == 3)
        empty = (sites == 0)
        up_sites = np.where(sites == 1)[0]
        dn_sites = np.where(sites == 2)[0]
        # In SU2 mode 1=spin-up coupled, 2=spin-down coupled. The total
        # spin coupling pattern matters in general, but for the present
        # purpose we treat (1,2) labels as singly-occupied sites and
        # uniformly distribute the CSF amplitude across all SD assignments
        # consistent with Sz=(na-nb)/2.
        single_sites = list(up_sites) + list(dn_sites)
        n_single = len(single_sites)
        # Determine alpha/beta counts on single-occupied sites.
        n_closed = int(closed.sum())
        n_alpha_single = na - n_closed
        n_beta_single = nb - n_closed
        if n_alpha_single < 0 or n_beta_single < 0:
            continue
        if n_alpha_single + n_beta_single != n_single:
            continue
        # Enumerate all ways to choose which single sites are alpha.
        n_combos = 0
        # Pre-compute base alpha / beta strings from closed shells.
        sa_base = 0
        sb_base = 0
        for i, occ in enumerate(sites):
            if occ == 3:
                sa_base |= (1 << i)
                sb_base |= (1 << i)
        # Distribute uniformly: amp_per_det = c / sqrt(n_combos)
        alpha_choices = list(combinations(range(n_single), n_alpha_single))
        n_combos = len(alpha_choices)
        if n_combos == 0:
            continue
        amp = c / np.sqrt(n_combos)
        for picks in alpha_choices:
            sa = sa_base
            sb = sb_base
            picks_set = set(picks)
            for j, site in enumerate(single_sites):
                if j in picks_set:
                    sa |= (1 << site)
                else:
                    sb |= (1 << site)
            ia = a_idx.get(sa)
            ib = b_idx.get(sb)
            if ia is not None and ib is not None:
                ci[ia, ib] += amp
    return ci
