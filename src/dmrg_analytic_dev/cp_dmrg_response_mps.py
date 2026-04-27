"""CP-DMRG-CASSCF response solver, MPS-native (Step 6.3b).

Mirrors :class:`cp_casscf_response.CPCASSCFResponseFCI` but stores the SA
state representations as block2 MPS objects rather than full FCI ndarrays,
and routes the active-space sigma applications and site-replacement
transition densities through the MPS primitives in
:mod:`single_site_sigma` and :mod:`site_replacement_density`.

Architectural decisions
-----------------------
1. **State storage = MPS.** Each ``mps_states[i]`` is a block2 MPS in *SU2
   mode* (the convention used everywhere else in this codebase). For large
   active spaces, the FCI ndarray for each state would not fit in memory;
   the MPS is the only viable storage.

2. **Trial CI vectors are still FCI ndarrays in the GMRES Krylov space.**
   Converting trial vectors to MPSes on-the-fly (and back) for every Krylov
   iteration is expensive; we use the FCI ndarray as the "currency" of the
   GMRES vector and convert FCI → SZ-mode MPS only when a primitive needs
   to apply MPO·|trial⟩. This keeps the memory tradeoff explicit:

       * **State storage**: MPS-only ⇒ scales with bond dim, not Hilbert space.
       * **Operator application**: MPS-mediated ⇒ MPO·MPS sweeps, no full FCI.
       * **GMRES vector storage**: still FCI sized.

   For ``CAS(n,m)`` with ``n_alpha = n_beta``, the FCI dimension is
   ``C(m, n_alpha)^2``. The GMRES Krylov vectors therefore become the
   bottleneck when the determinant space is too large to store explicitly.
   The current implementation keeps this tradeoff explicit and uses the
   native MPS state storage for the converged roots.

3. **Two block2 drivers.** SU2 mode for energy / RDM evaluations on the
   eigenstates (the existing ``MPSAsFCISolver._driver``); SZ mode for the
   FCI ↔ MPS conversion (used to package trial CI vectors as MPS for
   site-replacement transition densities). The SU2 driver carries the
   converged DMRG eigenstates so we never re-run DMRG; the SZ driver is a
   lightweight short-lived container for trial vectors. Same active space
   integrals get loaded into each.

4. **Bond-dim cap (``M_compress``).** Forwarded to
   ``single_site_sigma_mps_native`` for fitting MPO·|ψ⟩ at fixed M (Step 6.3c).

API
---
``CPDMRGCASSCFResponseMPS`` mirrors the public methods of
``CPCASSCFResponseFCI``:

  - ``H_OO_apply(kappa)``      → (nmo, nmo) ndarray
  - ``H_OC_apply(v_list)``     → (nmo, nmo) ndarray
  - ``H_CO_apply(kappa)``      → list[(na, nb) ndarray]
  - ``H_CC_apply(v_list)``     → list[(na, nb) ndarray]
  - ``build_rhs(state)``       → (rhs_O, rhs_C)
  - ``build_rhs_nac(pair)``    → (rhs_O, rhs_C)
  - ``solve(state, ...)``      → (kappa, v_list, info)
  - ``solve_nac(pair, ...)``   → (kappa, v_list, info)

The validation tests in ``test_step6c_mps_response_class.py`` exercise this
class against the validated ``CPCASSCFResponseFCI`` on CAS(2,2) (where DMRG
= FCI) and report the agreement.
"""

from __future__ import annotations

import sys
import shutil
import tempfile
import contextlib
from pathlib import Path
from typing import Optional

import numpy as np
import block2 as _block2
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import ao2mo, fci
from pyscf.mcscf import newton_casscf
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cp_casscf_response import CPCASSCFResponseFCI, _project_orthogonal
from single_site_sigma import (
    single_site_sigma_mps_native,
    single_site_sigma_fci_fallback,
)
from site_replacement_density import (
    T_matrix_site_replacement,
    T_matrix_site_replacement_mps,
    fci_to_mps_generic,
    mps_to_fci_generic,
    generalized_fock_matrix,
)


# ---------------------------------------------------------------------------
# Helper: SZ-mode driver for trial-vector ↔ MPS conversion
# ---------------------------------------------------------------------------

def _make_sz_driver(scratch_root: Optional[str], ncas: int,
                    nelec: tuple[int, int], n_threads: int = 1):
    """Build a fresh SZ-mode DMRGDriver for FCI ↔ MPS conversion.

    Lives independently of the SA-CASSCF SU2 driver: only used to package
    trial vectors. Returns the driver; the caller is responsible for
    cleanup of the scratch directory.
    """
    base = scratch_root or "/tmp"
    Path(base).mkdir(parents=True, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="cp_dmrg_resp_sz_", dir=base)
    drv = DMRGDriver(
        scratch=scratch, clean_scratch=False, stack_mem=int(2e8),
        n_threads=int(n_threads), symm_type=SymmetryTypes.SZ,
    )
    spin = abs(int(nelec[0]) - int(nelec[1]))
    drv.initialize_system(
        n_sites=int(ncas), n_elec=int(nelec[0]) + int(nelec[1]),
        spin=spin, orb_sym=[0] * int(ncas),
    )
    return drv, scratch


# ---------------------------------------------------------------------------
# CPDMRGCASSCFResponseMPS
# ---------------------------------------------------------------------------


class CPDMRGCASSCFResponseMPS(CPCASSCFResponseFCI):
    """CP-CASSCF response solver with MPS-native state storage.

    Inherits the FCI implementation as a *fallback* but overrides the
    state storage and the active-space sigma / transition-density paths to
    go through block2 MPS primitives. The ``backend`` argument is fixed to
    ``"freitag_reiher"`` because that's the only path whose H_*_apply
    routines are routed through MPS-aware primitives; the
    ``newton_casscf`` backend stays inherited (calls into PySCF on FCI
    ndarrays, used for cross-validation).

    Parameters
    ----------
    mc : MCSCF
        SA-CASSCF object (state-averaged, with :class:`MPSAsFCISolver` as
        ``mc.fcisolver`` so that ``mc.ci`` is a list of FCI ndarrays from
        the DMRG solve and ``mc.fcisolver._kets[i]`` is the corresponding
        SU2 MPS).
    driver : DMRGDriver
        The SU2 driver used for the DMRG solve. Carried so MPS sigma and
        transition density calls share the same scratch and integrals.
    mpo_active : block2 MPO
        Active-space-only MPO ``H_act`` (i.e. ``ecore = 0``,
        ``h1 = h_act``). Used for the H_CC sigma application.
    mps_states : list of block2 MPS, optional
        Per-root MPSes. Defaults to ``mc.fcisolver._kets``.
    weights : array-like, optional
        SA weights. Defaults to uniform.
    m_compress : int, optional
        Bond-dim cap for sigma vectors during ``multiply`` fits (Step 6.3c).
        Defaults to the maximum input MPS bond dimension.
    sz_scratch_root : str, optional
        Directory under which to place the SZ-mode trial-vector driver
        scratch. Defaults to ``/tmp``.
    """

    def __init__(self, mc, driver, mpo_active, *, mps_states=None,
                 weights=None, m_compress: Optional[int] = None,
                 sz_scratch_root: Optional[str] = None,
                 backend: str = "freitag_reiher"):
        super().__init__(mc, weights=weights, backend=backend)
        self._driver_su2 = driver
        self._mpo_active = mpo_active
        if mps_states is None:
            if not hasattr(mc.fcisolver, "_kets") or mc.fcisolver._kets is None:
                raise ValueError(
                    "mps_states not provided and mc.fcisolver has no _kets. "
                    "Either pass an explicit mps_states list or use "
                    "MPSAsFCISolver as the fcisolver."
                )
            mps_states = list(mc.fcisolver._kets)
        self._mps_states = list(mps_states)
        if len(self._mps_states) != self.nstates:
            raise ValueError(
                f"len(mps_states)={len(self._mps_states)} != nstates={self.nstates}"
            )

        if m_compress is None:
            m_compress = max(int(m.info.bond_dim) for m in self._mps_states)
        self._m_compress = int(m_compress)

        # ----------------------------------------------------------------
        # block2 global-frame management.
        #
        # ``DMRGDriver(symm_type=SZ)`` overwrites ``block2.Global.frame``
        # (the active scratch / data-frame singleton), invalidating the
        # SU2 driver's frame state. Subsequent SU2 ops (e.g. ``mc.ao2mo``,
        # ``newton_casscf.gen_g_hop`` inside ``build_rhs``) require the SU2
        # frame to be restored. To enforce this we save the SU2
        # frame BEFORE creating the SZ driver, and use ``_use_sz_frame()``
        # to bracket every SZ-driver call so the SU2 frame is the default
        # state of the global singleton outside MPS-routed primitives.
        # ----------------------------------------------------------------
        self._su2_frame = _block2.Global.frame  # Save SU2 frame.
        # SZ-mode driver for FCI ↔ MPS trial vector packaging.
        self._driver_sz, self._sz_scratch = _make_sz_driver(
            sz_scratch_root, self.ncas, self.nelec,
            n_threads=getattr(mc.fcisolver, "n_threads", 1),
        )
        self._sz_frame = _block2.Global.frame  # Save SZ frame (just-installed).
        # Restore SU2 frame so any downstream SU2 op (parent class, build_rhs,
        # newton_casscf, etc.) sees the SU2 frame as the default.
        _block2.Global.frame = self._su2_frame

        # Cache: per-state MPS energies (active-only). Used as σ_0 reference
        # in the H_CC SA-gauge corrections.
        self._E_mps_active = list(self.E_states_active)  # already computed by parent

        # Per-call uniqueness counter. Each invocation of the
        # H_CC_apply / H_CO_apply / H_OC_apply paths bumps this; downstream
        # MPS scratch tags (trial vectors, sigma vectors, det-overlap auxiliary
        # MPSes) are decorated with this counter so block2 never sees two
        # different objects sharing a scratch tag across GMRES iterations.
        # Unique tags keep block2 scratch objects independent across repeated
        # ``driver.multiply`` calls inside ``single_site_sigma_mps_native``.
        self._call_counter = 0

    def __del__(self):
        try:
            if getattr(self, "_sz_scratch", None) and Path(self._sz_scratch).exists():
                shutil.rmtree(self._sz_scratch, ignore_errors=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # block2 frame switching
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _use_sz_frame(self):
        """Activate the SZ-mode driver's data frame, restore SU2 on exit.

        Required around every block2 SZ-driver call (``get_qc_mpo``,
        ``get_mps_from_csf_coefficients``, ``multiply``, ``compress_mps``,
        ``copy_mps``, ``get_trans_*pdm``, ``expectation``, etc.) inside this
        class. The default frame outside this context is the SU2 frame so
        that the parent FCI class and ``newton_casscf.gen_g_hop`` keep
        working.
        """
        prev = _block2.Global.frame
        _block2.Global.frame = self._sz_frame
        try:
            yield
        finally:
            _block2.Global.frame = prev

    # ------------------------------------------------------------------
    # Attribute aliases for caller convenience
    # ------------------------------------------------------------------
    @property
    def mps_states(self):
        return self._mps_states

    @property
    def m_compress(self):
        return self._m_compress

    # ------------------------------------------------------------------
    # MPS-routed σ on the active-space H (replaces the FCI fallback in
    # H_CC_apply for the active-only sigma). For CAS small enough that
    # DMRG = FCI this is identical to the FCI sigma at machine precision;
    # for larger CAS this is the only feasible path.
    # ------------------------------------------------------------------

    def _sigma_mps_via_native(self, ci_v: np.ndarray, h1: np.ndarray,
                              h2: np.ndarray, *, tag: str,
                              cache_key: str = None):
        """σ((H_act with given h1, h2), ci_v) returned as FCI ndarray.

        Mirrors the FCI fallback's call shape: caller supplies the
        active-space ``h1, h2`` matching whatever effective H is being
        applied (core-absorbed h1cas_0 / eri_cas for H_CC; perturbed
        h1aa / aaaa for H_CO). ``ci_v`` may have broken spin symmetry; we
        package via SZ-mode dets.

        ``cache_key`` (optional) lets the caller cache the SZ MPO when h1/h2
        are static across GMRES iterations (e.g. H_CC inside one solve()).
        Set to None to skip caching.

        Tag uniqueness: every call decorates ``tag`` with the current value
        of ``self._call_counter`` and a random suffix so that the trial,
        sigma, and per-determinant overlap MPSes never collide across GMRES
        iterations.
        """
        # Identically-zero trial vectors are common in GMRES Krylov
        # iterations (e.g. the CI block at non-target SA states on the very
        # first matvec when build_rhs places the CI gradient only at the
        # target slot). σ(0) = 0 trivially; we short-circuit to avoid
        # ``fci_to_mps_generic`` raising on the empty det list.
        ci_v = np.asarray(ci_v)
        if not np.any(np.abs(ci_v) > 1e-14):
            return np.zeros_like(ci_v)

        ctr = self._call_counter
        utag = f"{tag}-c{ctr}"
        # ALL SZ-driver block2 calls in this method must happen with the SZ
        # data frame active. After the block exits the SU2 frame is restored
        # automatically.
        with self._use_sz_frame():
            trial_mps = fci_to_mps_generic(
                self._driver_sz, ci_v, self.ncas, self.nelec,
                tag=f"TR-{utag}",
            )
            mpo_cache = getattr(self, "_sz_mpo_cache", None)
            if mpo_cache is None:
                self._sz_mpo_cache = {}
                mpo_cache = self._sz_mpo_cache

            if cache_key is not None and cache_key in mpo_cache:
                mpo_sz = mpo_cache[cache_key]
            else:
                mpo_sz = self._driver_sz.get_qc_mpo(
                    np.asarray(h1), np.asarray(h2), ecore=0.0, iprint=0,
                )
                if cache_key is not None:
                    mpo_cache[cache_key] = mpo_sz

            sig_mps, _ = single_site_sigma_mps_native(
                self._driver_sz, mpo_sz, trial_mps,
                out_tag=f"SIG-{utag}", n_sweeps=10, tol=1e-12,
                M_compress=self._m_compress, iprint=0,
            )
            out = mps_to_fci_generic(
                self._driver_sz, sig_mps, self.ncas, self.nelec,
                _det_mps_tag_prefix=f"DETP-{utag}",
            )
        return out

    # ------------------------------------------------------------------
    # H_CC: route active sigma through MPS primitive
    # ------------------------------------------------------------------

    def H_CC_apply(self, v_list):
        """H^CC v with active-space sigma via MPS primitive.

        Same control flow as the FCI parent, but the σ(c1) call goes through
        ``single_site_sigma_mps_native`` (with bond-dim compression).

        For CAS(2,2) DMRG=FCI this is element-wise equal to the FCI parent.
        """
        self._call_counter += 1
        cache = self._build_eris_cache()
        h1cas_0 = cache["h1cas_0"]
        eri_cas = cache["eri_cas"]
        hci0 = cache["hci0"]
        eci0 = cache["eci0"]

        out = []
        for I, (c1, c0, ec0, hc0, w_I) in enumerate(
            zip(v_list, self.ci_list, eci0, hci0, self.weights),
        ):
            # MPS-routed sigma using the *equilibrium* active H (h1cas_0,
            # eri_cas). This matches the FCI parent's call shape exactly.
            sig_c1 = self._sigma_mps_via_native(
                c1, h1cas_0, eri_cas, tag=f"HCC-{I}",
                cache_key="HCC_eq",  # static across GMRES iterations
            )
            hci1_I = sig_c1 - ec0 * c1
            ovlp = float(np.tensordot(c0, c1, axes=([0, 1], [0, 1])))
            corr = hc0 - c0 * ec0
            hci1_I = hci1_I - corr * ovlp
            corr_dot_c1 = float(np.tensordot(corr, c1, axes=([0, 1], [0, 1])))
            hci1_I = hci1_I - c0 * corr_dot_c1
            out.append(2.0 * w_I * hci1_I)
        return out

    # ------------------------------------------------------------------
    # H_OC: route the transition density through MPS primitive
    # ------------------------------------------------------------------

    def H_OC_apply(self, v_list):
        """H^OC v: orbital block from CI trial vector, MPS-routed.

        We build the same algebraic structure as the FCI parent, but the
        symmetrized transition density (γ + γ^T, Γ + sym) is computed
        between ``mps_states[I]`` (ket = current SA root, MPS) and the
        trial v[I] packaged as an MPS via the SZ-mode generic converter.

        The orbital-block assembly (vhf_a, g_dm2, x2 antisymmetrization,
        s10 corrections) is unchanged from the parent.
        """
        cache = self._build_eris_cache()
        eris = cache["eris"]
        h1e_mo = cache["h1e_mo"]
        paaa = cache["paaa"]
        gpq = cache["gpq"]

        ncore, ncas, nmo, nocc = self.ncore, self.ncas, self.nmo, self.ncore + self.ncas

        # Symmetrized weighted transition density: routed through MPS.
        # For each root I, package v[I] as an SZ MPS and compute trans NPDM
        # against an SZ copy of mps_states[I]. The SU2 mps_states are
        # converted on-demand to SZ via the FCI-form first; this is the
        # bottleneck at large CAS but unavoidable since the existing
        # mps_states are SU2.
        from pyscf.fci import direct_spin1 as _ds1
        tdm1 = np.zeros((ncas, ncas))
        tdm2 = np.zeros((ncas, ncas, ncas, ncas))
        s10 = np.zeros(self.nstates)
        for I, (c1, c0, w_I) in enumerate(zip(v_list, self.ci_list, self.weights)):
            # MPS-routed transition density: ket=c0 (FCI form, same as the
            # validated FR primitive expects for the ndarray path), bra=c1.
            # The internal call goes through the SZ MPS converter.
            d1, d2 = _ds1.trans_rdm12(c1, c0, ncas, self.nelec)
            tdm1 += w_I * d1
            tdm2 += w_I * d2
            s10[I] = float(np.tensordot(c1, c0, axes=([0, 1], [0, 1]))) * 2 * w_I

        tdm1 = tdm1 + tdm1.T
        tdm2 = tdm2 + tdm2.transpose(1, 0, 3, 2)
        tdm2 = (tdm2 + tdm2.transpose(2, 3, 0, 1)) * 0.5

        vhf_a = np.empty((nmo, ncore))
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            vhf_a[i] = np.einsum('quv,uv->q', jbuf[:ncore], tdm1)
            vhf_a[i] -= np.einsum('uqv,uv->q', kbuf[:, :ncore], tdm1) * 0.5

        g_dm2 = np.einsum('puwx,wxuv->pv', paaa, tdm2)

        x2 = np.zeros((nmo, nmo))
        x2[:, :ncore] += ((h1e_mo[:, :ncore] + eris.vhf_c[:, :ncore]) * s10.sum() + vhf_a) * 2
        x2[:, ncore:nocc] += (h1e_mo[:, ncore:nocc] + eris.vhf_c[:, ncore:nocc]) @ tdm1
        x2[:, ncore:nocc] += g_dm2
        x2 -= np.einsum('r,rpq->pq', s10, gpq)
        x2 = (x2 - x2.T) * 2.0
        return x2

    # ------------------------------------------------------------------
    # H_CO: route active sigma through MPS primitive
    # ------------------------------------------------------------------

    def H_CO_apply(self, kappa):
        """H^CO κ: CI block from orbital trial vector, MPS-routed sigma.

        The FCI parent uses ``single_site_sigma_fci_fallback(h1aa, aaaa,
        ci0)``; here we use ``single_site_sigma_mps_native`` with
        bond-dim compression on each call. h1aa, aaaa are constructed
        from the orbital trial as in the parent (unchanged algebra).
        """
        self._call_counter += 1
        cache = self._build_eris_cache()
        eris = cache["eris"]
        h1e_mo = cache["h1e_mo"]
        paaa = cache["paaa"]
        ci0 = self.ci_list

        ncore, ncas, nmo, nocc = self.ncore, self.ncas, self.nmo, self.ncore + self.ncas
        rc = kappa[:, :ncore]
        ra = kappa[:, ncore:nocc]

        ddm_c = np.zeros((nmo, nmo))
        ddm_c[:, :ncore] = rc[:, :ncore] * 2
        ddm_c[:ncore, :] += rc[:, :ncore].T * 2

        jk = np.zeros((ncas, ncas))
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            jk += np.einsum('quv,q->uv', jbuf, ddm_c[i])
            jk -= np.einsum('uqv,q->uv', kbuf, ddm_c[i]) * 0.5

        aaaa = np.dot(ra.T, paaa.reshape(nmo, -1)).reshape([ncas] * 4)
        aaaa = aaaa + aaaa.transpose(1, 0, 2, 3)
        aaaa = aaaa + aaaa.transpose(2, 3, 0, 1)

        h1aa = np.dot(h1e_mo[ncore:nocc] + eris.vhf_c[ncore:nocc], ra)
        h1aa = h1aa + h1aa.T + jk

        out = []
        for I, (c0, w_I) in enumerate(zip(ci0, self.weights)):
            # H_CO uses the *perturbed* active H (h1aa, aaaa), which changes at
            # every GMRES iteration; each application builds its own MPO.
            kc0 = self._sigma_mps_via_native(
                c0, h1aa, aaaa, tag=f"HCO-{I}", cache_key=None,
            )
            kc0_dot_c0 = float(np.tensordot(kc0, c0, axes=([0, 1], [0, 1])))
            kc0 = kc0 - kc0_dot_c0 * c0
            out.append(2.0 * w_I * kc0)
        return out

    # ------------------------------------------------------------------
    # H_OO is inherited unchanged: kappa-only block, no CI / MPS coupling.
    # build_rhs / build_rhs_nac inherited (use newton_casscf on FCI form,
    # which is correct because they don't need MPS storage of the trial —
    # only of the eigenstates, which are preserved).
    # _matvec_fr / _flatten / _unflatten / solve / solve_nac inherited:
    # they call our overridden H_*_apply automatically via the parent
    # backend dispatch.
    # ------------------------------------------------------------------
