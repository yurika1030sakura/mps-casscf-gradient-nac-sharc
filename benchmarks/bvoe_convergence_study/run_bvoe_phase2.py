"""BVOE convergence study — Phase 2 (REAL DMRG).

Phase 1 used CI-vector SVD truncation as a proxy for DMRG. Phase 2 uses
**actual block2 DMRG**: SU2-mode driver with a target SA(2) singlet pair
plus a configurable root buffer at the converged FCI orbitals, then converts
each candidate MPS to a PySCF FCI ndarray via ``mps_change_to_sz`` +
``get_csf_coefficients``.

Pipeline at each (system, M):

  1. Run SA(2)-CASSCF with the standard PySCF FCI fcisolver → converged
     orbitals (mc.mo_coeff) and CI vectors.
  2. At the converged orbitals, build the SU2-mode DMRG MPO from the
     active-space integrals.
  3. Run SU2 DMRG with ``nroots=2+BVOE_ROOT_BUFFER``, ``bond_dim = M``.
  4. Convert each candidate MPS to a PySCF FCI ndarray via the SZ-mode CSF route
     (``mps_change_to_sz`` then ``get_csf_coefficients``, then PySCF
     ordering-sign correction).
  5. Select and phase-align the two target roots by overlap with the FCI
     validation reference.
  6. Install the bond-dim-M CI vectors back into ``mc`` (replacing
     ``mc.ci``), and recompute the analytic gradient (state 0) and NAC
     (states (0,1)) with PySCF's standard ``pyscf.grad.sacasscf`` /
     ``pyscf.nac.sacasscf`` machinery.

Reference: same SA-CASSCF/FCI run, no DMRG truncation.

Why SU2 mode + post-hoc swap vs full DMRG-fcisolver SA-CASSCF:
  * The validated ``MPSAsFCISolver`` with ``force_dmrg=True`` uses
    SZ-mode DMRG. SZ is not spin-adapted; with ``nroots=2`` random-MPS
    initial guesses the second-root energy collapses to a non-physical
    sector and ``mps_to_fci_generic`` deadlocks on the broken MPS. SU2
    mode is spin-adapted and handles ``nroots=2`` cleanly.
  * Doing the DMRG at the FCI-converged orbitals (rather than letting
    DMRG drive the CASSCF macro loop) means the response Hessian is
    well-conditioned at FCI-optimal orbitals and we measure the *pure*
    BVOE — i.e. the gradient/NAC error from CI-vector compression at
    bond dim M, isolated from orbital re-optimization at the
    M-truncated MPS manifold.

Output:
  data_phase2/{system}_M{M}.json     raw results per (system, M)
  data_phase2/{system}_FCI.json      reference (PySCF FCI)
  summary_phase2.json                aggregated diff norms
"""

from __future__ import annotations

import json
import itertools
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.fci import cistring
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf.nac import sacasscf as nac_sacasscf

ROOT = Path(__file__).resolve().parent
DEV_ROOT = (ROOT / ".." / ".." / "src" / "dmrg_analytic_dev").resolve()
sys.path.insert(0, str(DEV_ROOT))
from site_replacement_density import _pyscf_to_block2_sign  # noqa: E402


# ---------------------------------------------------------------------------
# Test systems
# ---------------------------------------------------------------------------

def build_h2o():
    mol = gto.M(atom="""
        O   0.0000   0.0000   0.0000
        H   0.0000   0.7572   0.5868
        H   0.0000  -0.7572   0.5868
    """, basis="sto-3g", spin=0, charge=0, verbose=0)
    return mol, 4, 4


def build_h4():
    R_bohr = 1.5
    R_ang = R_bohr * 0.529177210903
    coords = "\n".join(f"H   {i * R_ang:.6f}  0.0  0.0" for i in range(4))
    mol = gto.M(atom=coords, basis="sto-3g", spin=0, charge=0,
                unit="Angstrom", verbose=0)
    return mol, 4, 4


def build_n2():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.4",
                basis="sto-3g", spin=0, charge=0,
                unit="Angstrom", verbose=0)
    return mol, 6, 6


def build_c2():
    mol = gto.M(atom="C 0 0 0; C 0 0 1.25",
                basis="sto-3g", spin=0, charge=0,
                unit="Angstrom", verbose=0)
    return mol, 8, 8


def build_lif_avoided():
    # Near the ionic/covalent avoided-crossing region; this gives a more
    # informative NAC benchmark than symmetry-suppressed equilibrium N2.
    mol = gto.M(atom="Li 0 0 0; F 0 0 6.5",
                basis="sto-3g", spin=0, charge=0,
                unit="Bohr", verbose=0)
    return mol, 4, 4


def build_h2o_631g_cas66():
    mol = gto.M(atom="""
        O   0.0000   0.0000   0.0000
        H   0.0000   0.7572   0.5868
        H   0.0000  -0.7572   0.5868
    """, basis="6-31g", spin=0, charge=0, verbose=0)
    return mol, 6, 6


# (builder, label, M scan, full bipartite rank).
# NOTE: M=1 is excluded from default scans because the rank-1 CI vector
# makes the SA-CASSCF preconditioner overlap matrix singular, which sends
# pyscf's gradient solver into an LAPACK-warning loop (xsyev info=...).
# That is a *correct* numerical signal that M=1 is in the singular regime,
# but it produces minute-to-hour stalls and floods the log. The M=2..rank
# range gives the full convergence picture for the figure.
SYSTEMS = {
    "h4":  (
        build_h4, "H4 chain / sto-3g / CAS(4,4) SA(2), R=1.5 Bohr",
        [2, 3, 4, 5, 6, 8, 12, 200], 200,
    ),
    "h2o": (
        build_h2o, "H2O / sto-3g / CAS(4,4) SA(2)",
        [2, 3, 4, 5, 6, 8, 12, 200], 200,
    ),
    "n2":  (
        build_n2, "N2 / sto-3g / CAS(6,6) SA(2), R=1.4 Ang",
        [2, 4, 8, 12, 16, 20, 30, 200], 200,
    ),
    "c2":  (
        build_c2, "C2 / sto-3g / CAS(8,8) SA(2), R=1.25 Ang",
        [4, 8, 16, 32, 64, 70, 120, 200], 200,
    ),
    "lif": (
        build_lif_avoided, "LiF / sto-3g / CAS(4,4) SA(2), R=6.5 Bohr",
        [2, 3, 4, 5, 6, 8, 12, 200], 200,
    ),
    "h2o_631g": (
        build_h2o_631g_cas66, "H2O / 6-31G / CAS(6,6) SA(2)",
        [2, 4, 8, 12, 16, 20, 30, 200], 200,
    ),
}

# General root-buffer setting, matching the production SHARC interface.
# Benchmarks use FCI overlaps only to score the result; production runs use
# the same extra-root idea with previous-step overlaps instead of FCI.
ROOT_BUFFER = int(os.environ.get("BVOE_ROOT_BUFFER", "4"))


# ---------------------------------------------------------------------------
# DMRG MPS → PySCF FCI ndarray  (fast path via SU2→SZ conversion)
# ---------------------------------------------------------------------------

def _su2_mps_to_fci(driver, mps, ncas, nelec, *, sz_driver,
                    sz_tag="MPSSZ"):
    """Convert an SU2-mode MPS to a PySCF FCI ndarray.

    Uses ``mps_change_to_sz`` (called on the *SU2* driver) to bring the MPS
    to the SZ basis, then ``get_csf_coefficients`` *on a separate SZ
    driver* (which in SZ mode returns determinant occupations
    [0=empty, 1=alpha, 2=beta, 3=double] with their coefficients).
    Finally rebuilds the PySCF FCI ndarray with PySCF↔block2
    fermion-ordering sign correction.

    The SZ driver MUST be supplied (block2 type system enforces strict
    SU2/SZ separation; calling ``get_csf_coefficients`` on the SU2 driver
    with an SZ-converted MPS raises a TypeError).
    """
    na, nb = int(nelec[0]), int(nelec[1])
    mps_sz = driver.mps_change_to_sz(mps, tag=sz_tag)
    csfs, coefs = sz_driver.get_csf_coefficients(mps_sz, cutoff=1e-14,
                                                   iprint=0)
    strs_a = list(cistring.make_strings(range(ncas), na))
    strs_b = list(cistring.make_strings(range(ncas), nb))
    a_idx = {int(s): j for j, s in enumerate(strs_a)}
    b_idx = {int(s): j for j, s in enumerate(strs_b)}
    ci = np.zeros((len(strs_a), len(strs_b)), dtype=np.float64)
    for det, c in zip(csfs, coefs):
        if abs(float(c)) < 1e-15:
            continue
        sa = sb = 0
        for site, occ in enumerate(det):
            occ = int(occ)
            if occ == 3:
                sa |= (1 << site)
                sb |= (1 << site)
            elif occ == 1:
                sa |= (1 << site)
            elif occ == 2:
                sb |= (1 << site)
        ia = a_idx.get(sa)
        ib = b_idx.get(sb)
        if ia is None or ib is None:
            continue
        sign = _pyscf_to_block2_sign(sa, sb, ncas)
        ci[ia, ib] = sign * float(c)
    return ci


# ---------------------------------------------------------------------------
# Phase-aware align: align a bond-dim-M CI vector to a reference
# ---------------------------------------------------------------------------

def _align_global_phase(ci, ci_ref):
    """Multiply ci by ±1 to match its global phase to ci_ref.

    Works for a single state. CASSCF analytic NAC depends linearly on the
    phase of each CI vector; fixing the phase at the input avoids spurious
    sign flips in the gradient/NAC error.
    """
    ovlp = float(np.vdot(np.asarray(ci_ref).ravel(), np.asarray(ci).ravel()))
    return ci if ovlp >= 0 else -ci


def _match_and_align_roots(ci_roots, fci_refs):
    """Assign DMRG roots to FCI roots by maximum CI overlap.

    The SU2 DMRG solver returns roots ordered by its internal optimization.
    Near avoided crossings or compact symmetry benchmarks, that order can
    differ from the PySCF FCI root order.  Derivative comparisons must first
    solve this small assignment problem; otherwise a root flip looks like a
    large response error.
    """
    ntarget = len(fci_refs)
    nraw = len(ci_roots)
    if nraw < ntarget:
        raise ValueError(
            f"Need at least {ntarget} DMRG roots, got {nraw}"
        )
    overlap = np.empty((nraw, ntarget))
    for i, ci in enumerate(ci_roots):
        ci_vec = np.asarray(ci).ravel()
        for j, ref in enumerate(fci_refs):
            overlap[i, j] = float(np.vdot(np.asarray(ref).ravel(), ci_vec))

    best_perm = None
    best_score = -1.0
    for perm in itertools.permutations(range(nraw), ntarget):
        score = sum(abs(overlap[perm[j], j]) for j in range(ntarget))
        if score > best_score:
            best_score = score
            best_perm = perm

    aligned = []
    assigned_overlaps = []
    for j, i in enumerate(best_perm):
        ci = _align_global_phase(ci_roots[i], fci_refs[j])
        norm = np.linalg.norm(ci)
        if norm > 1e-30:
            ci = ci / norm
        aligned.append(ci)
        assigned_overlaps.append(float(abs(overlap[i, j])))

    return aligned, list(best_perm), overlap.tolist(), assigned_overlaps


# ---------------------------------------------------------------------------
# Run a single (system, M) DMRG → grad + NAC
# ---------------------------------------------------------------------------

def _redirect_block2_logs(scratch):
    """Block2 sometimes writes ATTENTION: xsyev info messages to stderr;
    when the SA-CASSCF preconditioner has a near-singular block we get a
    flood. We can't fully suppress them but at least ensure they aren't
    lost into our redirected log via stderr-mixing (handled by the caller).
    """
    return None


def setup_sacasscf_fci(mol, ncas, nelec_act):
    """SA(2)-CASSCF with PySCF FCI (M = inf reference)."""
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, ncas, nelec_act)
    mc.fix_spin_(ss=0)
    mc.fcisolver.nroots = 2
    mc.conv_tol = 1e-10
    mc.conv_tol_grad = 1e-7
    mc.max_cycle_macro = 200
    mc = mc.state_average_([0.5, 0.5])
    mc.kernel()
    return mc


def _reference_has_wavefunction(ref):
    """Return True if a cached FCI reference contains the fixed gauge data."""
    return (
        isinstance(ref, dict)
        and "mo_coeff" in ref
        and "ci" in ref
        and len(ref.get("ci", [])) == 2
    )


def setup_sacasscf_from_reference(system_key, ref):
    """Build an SA-CASSCF object at the cached FCI orbital/CI gauge.

    Phase-2 derivative comparisons must use one fixed FCI-stationary orbital
    basis.  Re-running CASSCF independently for every M can converge to a
    symmetry-equivalent active-orbital gauge, especially under threaded BLAS,
    which makes derivative comparisons look like root/gauge failures even
    when the DMRG state is correct.
    """
    builder = SYSTEMS[system_key][0]
    mol, ncas, nelec_act = builder()
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, ncas, nelec_act)
    mc.fix_spin_(ss=0)
    mc.fcisolver.nroots = 2
    mc.conv_tol = 1e-10
    mc.conv_tol_grad = 1e-7
    mc.max_cycle_macro = 200
    mc = mc.state_average_([0.5, 0.5])
    mc.mo_coeff = np.asarray(ref["mo_coeff"])
    mc.ci = [np.asarray(c) for c in ref["ci"]]
    _set_state_energies(mc, ref["e_states"])
    return mc


def _set_state_energies(mc, e_states):
    """Populate PySCF state-average energy attributes used by NAC code."""
    e_states = [float(e) for e in e_states]
    try:
        mc.fcisolver.e_states = e_states
    except Exception:
        pass
    try:
        mc.e_states = e_states
    except Exception:
        pass
    try:
        weights = np.asarray(getattr(mc, "weights", [0.5, 0.5]), dtype=float)
        if len(weights) == len(e_states):
            mc.e_tot = float(np.dot(weights, e_states))
        else:
            mc.e_tot = float(np.mean(e_states))
    except Exception:
        pass


def compute_grad_and_nac(mc):
    g = sacasscf_grad.Gradients(mc).kernel(state=0)
    n = nac_sacasscf.NonAdiabaticCouplings(mc).kernel(state=(0, 1))
    return np.asarray(g), np.asarray(n)


def run_fci_reference(system_key):
    builder, label = SYSTEMS[system_key][0], SYSTEMS[system_key][1]
    mol, ncas, nelec_act = builder()
    t0 = time.time()
    mc = setup_sacasscf_fci(mol, ncas, nelec_act)
    e_states = list(map(float, mc.e_states))
    g, n = compute_grad_and_nac(mc)
    return {
        "system": system_key, "label": label, "bond_dim": "FCI",
        "e_states": e_states,
        "grad": g.tolist(), "nac": n.tolist(),
        "mo_coeff": np.asarray(mc.mo_coeff).tolist(),
        "ci": [np.asarray(ci).tolist() for ci in mc.ci],
        "ci_norms": [float(np.linalg.norm(ci)) for ci in mc.ci],
        "ci_shape": list(mc.ci[0].shape),
        "ncas": int(mc.ncas), "ncore": int(mc.ncore),
        "nelecas": [int(x) for x in mc.nelecas],
        "nelec_act": int(sum(mc.nelecas)),
        "schema_version": 2,
        "runtime_s": time.time() - t0,
    }


def run_dmrg_at_fci_orbitals(system_key, bond_dim, *, fci_ref_payload=None,
                              n_sweeps=30, sweep_tol=1e-12, n_threads=1):
    """Build mc with PySCF FCI, converge, then swap CI to DMRG-truncated
    CI at the same orbitals; recompute grad + NAC."""
    builder, label = SYSTEMS[system_key][0], SYSTEMS[system_key][1]
    t0 = time.time()
    if _reference_has_wavefunction(fci_ref_payload):
        mc = setup_sacasscf_from_reference(system_key, fci_ref_payload)
        fci_ci = [np.asarray(c).copy() for c in fci_ref_payload["ci"]]
        fci_e_states = list(map(float, fci_ref_payload["e_states"]))
    else:
        mol, ncas, nelec_act = builder()
        mc = setup_sacasscf_fci(mol, ncas, nelec_act)
        fci_ci = [np.asarray(c).copy() for c in mc.ci]
        fci_e_states = list(map(float, mc.e_states))
    ncas = mc.ncas
    nelec_act = int(sum(mc.nelecas))

    # Active-space integrals at converged orbitals
    h1_act, ecore = mc.get_h1eff(mc.mo_coeff)
    eri_act = ao2mo.restore(1, np.asarray(mc.get_h2eff(mc.mo_coeff)), ncas)

    # Run SU2 DMRG with a generic root buffer at bond_dim = M.
    scratch = tempfile.mkdtemp(prefix="bvoe_p2_", dir="/tmp")
    try:
        driver = DMRGDriver(
            scratch=scratch, clean_scratch=False, stack_mem=int(2e8),
            n_threads=int(n_threads), symm_type=SymmetryTypes.SU2,
        )
        nelec_tot = int(nelec_act)
        spin = 0  # singlet
        driver.initialize_system(
            n_sites=ncas, n_elec=nelec_tot, spin=spin, orb_sym=[0] * ncas,
        )
        mpo = driver.get_qc_mpo(np.asarray(h1_act),
                                 np.asarray(eri_act),
                                 ecore=float(ecore), iprint=0)
        n_solve_roots = 2 + max(0, int(ROOT_BUFFER))
        ket = driver.get_random_mps(tag=f"K_{bond_dim}", bond_dim=int(bond_dim),
                                    nroots=n_solve_roots)
        ns = max(int(n_sweeps), 30)
        bd = [int(bond_dim)] * ns
        # Aggressive noise schedule lets random-init MPS find singlet sector
        noises = ([1e-3] * 8 + [1e-4] * 8 + [1e-5] * 8
                  + [1e-6] * 4 + [0.0] * (ns - 28))
        if len(noises) < ns:
            noises = noises + [0.0] * (ns - len(noises))
        e_dmrg = driver.dmrg(
            mpo, ket, n_sweeps=ns, bond_dims=bd,
            noises=noises[:ns], tol=float(sweep_tol), iprint=0,
        )
        kets = [driver.split_mps(ket, i, f"KS_{bond_dim}_{i}")
                for i in range(n_solve_roots)]

        # Build SZ driver for get_csf_coefficients route
        sz_driver = DMRGDriver(
            scratch=scratch, clean_scratch=False, stack_mem=int(2e8),
            n_threads=int(n_threads), symm_type=SymmetryTypes.SZ,
        )
        sz_driver.initialize_system(
            n_sites=ncas, n_elec=nelec_tot, spin=spin, orb_sym=[0] * ncas,
        )

        # Convert each MPS to a PySCF FCI ndarray.  Keep the raw DMRG root
        # order first, then assign to FCI roots by maximum overlap below.
        ci_dmrg_raw = []
        for i, k in enumerate(kets):
            ci_i = _su2_mps_to_fci(
                driver, k, ncas, mc.nelecas,
                sz_driver=sz_driver,
                sz_tag=f"SZ_{bond_dim}_{i}",
            )
            n_i = np.linalg.norm(ci_i)
            if n_i > 1e-30:
                ci_i = ci_i / n_i
            ci_dmrg_raw.append(ci_i)

        ci_dmrg, root_assignment, overlap_matrix, assigned_overlaps = (
            _match_and_align_roots(ci_dmrg_raw, fci_ci)
        )

        # Energy expectation under FCI Hamiltonian (sanity check)
        e_dmrg_check = []
        for i, ci_i in enumerate(ci_dmrg):
            e_i = (fci.direct_spin1.energy(h1_act, eri_act, ci_i, ncas,
                                            mc.nelecas) + ecore)
            e_dmrg_check.append(float(e_i))
    finally:
        try:
            shutil.rmtree(scratch, ignore_errors=True)
        except Exception:
            pass

    # Install DMRG-truncated CI into mc, recompute grad and NAC
    mc.ci = ci_dmrg
    _set_state_energies(mc, e_dmrg_check)
    g, n = compute_grad_and_nac(mc)
    return {
        "system": system_key, "label": label, "bond_dim": int(bond_dim),
        "e_states_fci": fci_e_states,
        "e_states_dmrg": e_dmrg_check,
        "e_dmrg_su2": (list(map(float, e_dmrg))
                       if hasattr(e_dmrg, "__iter__") else [float(e_dmrg)]),
        "root_assignment_dmrg_to_fci": root_assignment,
        "root_overlap_matrix": overlap_matrix,
        "root_assigned_abs_overlaps": assigned_overlaps,
        "grad": g.tolist(), "nac": n.tolist(),
        "ci_overlap_with_fci": [
            float(np.vdot(ci_dmrg[i].ravel(), fci_ci[i].ravel()))
            for i in range(2)
        ],
        "ncas": int(mc.ncas), "ncore": int(mc.ncore),
        "nelec_act": int(sum(mc.nelecas)),
        "runtime_s": time.time() - t0,
    }


def diff_norms(arr_M, arr_ref):
    a = np.asarray(arr_M)
    b = np.asarray(arr_ref)
    diff = a - b
    return {
        "l2": float(np.linalg.norm(diff)),
        "max": float(np.max(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-30)),
    }


def phase_aware_diff(arr_M, arr_ref):
    a = np.asarray(arr_M)
    b = np.asarray(arr_ref)
    d_pos = float(np.linalg.norm(a - b))
    d_neg = float(np.linalg.norm(a + b))
    if d_neg < d_pos:
        diff = a + b
        l2 = d_neg
    else:
        diff = a - b
        l2 = d_pos
    return {
        "l2": l2,
        "max": float(np.max(np.abs(diff))),
        "rel_l2": l2 / max(np.linalg.norm(b), 1e-30),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(only=None):
    data_dir = ROOT / "data_phase2"
    data_dir.mkdir(exist_ok=True)
    summary_path = ROOT / "summary_phase2.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}

    keys = list(SYSTEMS.keys())
    if only:
        keys = [k for k in keys if k in only]

    for sys_key in keys:
        summary.setdefault(sys_key, {})
        ref_path = data_dir / f"{sys_key}_FCI.json"
        ref = None
        force = os.environ.get("BVOE_FORCE_RECOMPUTE", "0") == "1"
        if ref_path.exists():
            try:
                ref = json.loads(ref_path.read_text())
                print(f"[{sys_key}] FCI ref: cached "
                      f"e0={ref['e_states'][0]:.8f}", flush=True)
                if force or not _reference_has_wavefunction(ref):
                    if force:
                        print(f"[{sys_key}] refreshing reference "
                              f"(BVOE_FORCE_RECOMPUTE=1)", flush=True)
                    else:
                        print(f"[{sys_key}] refreshing legacy reference "
                              f"(missing fixed orbital/CI gauge)", flush=True)
                    ref = None
            except Exception:
                ref = None
        if ref is None:
            try:
                _invalidate_system_cache(data_dir, sys_key, summary)
                ref = run_fci_reference(sys_key)
                ref_path.write_text(json.dumps(ref, indent=2))
                print(f"[{sys_key}] FCI ref: e0={ref['e_states'][0]:.8f}  "
                      f"|grad|={np.linalg.norm(ref['grad']):.4e}  "
                      f"|nac|={np.linalg.norm(ref['nac']):.4e}  "
                      f"({ref['runtime_s']:.1f}s)", flush=True)
            except Exception as e:
                print(f"[{sys_key}] FCI ref FAILED: {e}", flush=True)
                traceback.print_exc()
                summary[sys_key]["error"] = f"FCI ref failed: {e}"
                _write_summary(summary_path, summary)
                continue

        bond_dims = SYSTEMS[sys_key][2]
        for M in bond_dims:
            out_path = data_dir / f"{sys_key}_M{M}.json"
            if out_path.exists() and str(M) in summary[sys_key] \
                    and "error" not in summary[sys_key][str(M)]:
                print(f"[{sys_key}] M={M:4d}  cached", flush=True)
                continue
            try:
                res = run_dmrg_at_fci_orbitals(
                    sys_key, M,
                    fci_ref_payload=ref,
                    n_sweeps=int(os.environ.get("BVOE_SWEEPS", "30")),
                    n_threads=int(os.environ.get("BVOE_THREADS", "1")),
                )
                out_path.write_text(json.dumps(res, indent=2))
                grad_d = diff_norms(res["grad"], ref["grad"])
                nac_d = phase_aware_diff(res["nac"], ref["nac"])
                e0_d = abs(res["e_states_dmrg"][0] - ref["e_states"][0])
                e1_d = abs(res["e_states_dmrg"][1] - ref["e_states"][1])
                summary[sys_key][str(M)] = {
                    "grad_l2": grad_d["l2"], "grad_max": grad_d["max"],
                    "grad_rel": grad_d["rel_l2"],
                    "nac_l2": nac_d["l2"], "nac_max": nac_d["max"],
                    "nac_rel": nac_d["rel_l2"],
                    "e0_diff": e0_d, "e1_diff": e1_d,
                    "ci0_overlap": res["ci_overlap_with_fci"][0],
                    "ci1_overlap": res["ci_overlap_with_fci"][1],
                    "runtime_s": res["runtime_s"],
                }
                print(f"[{sys_key}] M={M:4d}  "
                      f"de0={e0_d:.2e}  de1={e1_d:.2e}  "
                      f"|<ci|FCI>|={res['ci_overlap_with_fci'][0]:.4f},"
                      f"{res['ci_overlap_with_fci'][1]:.4f}  "
                      f"grad={grad_d['l2']:.3e}  nac={nac_d['l2']:.3e}  "
                      f"({res['runtime_s']:.1f}s)", flush=True)
            except Exception as e:
                print(f"[{sys_key}] M={M} FAILED: {e}", flush=True)
                traceback.print_exc()
                summary[sys_key][str(M)] = {"error": str(e)}
            _write_summary(summary_path, summary)

    _write_summary(summary_path, summary)
    print(f"\nWrote {summary_path}", flush=True)


def _invalidate_system_cache(data_dir, sys_key, summary):
    """Remove M-point data tied to an older reference gauge."""
    for path in data_dir.glob(f"{sys_key}_M*.json"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    summary[sys_key] = {}


def _write_summary(path, summary):
    path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    only = None
    if len(sys.argv) > 1:
        only = set(sys.argv[1:])
    main(only=only)
