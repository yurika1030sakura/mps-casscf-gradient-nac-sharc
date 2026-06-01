"""MPS-Krylov backend for CP-DMRG-CASSCF response.

This module keeps the CI-response vectors in block2 MPS form during the
Krylov iteration:

* orbital variables are still dense NumPy arrays, because their dimension is
  small and independent of the determinant Hilbert space;
* CI Krylov vectors are MPS objects;
* Arnoldi orthogonalization uses MPS overlaps;
* linear combinations use ``DMRGDriver.addition`` and are compressed/fitted by
  block2;
* small-CAS validation helpers can still convert FCI vectors to MPS so the
  implementation can be compared against the established FCI backend.

The response RHS, Krylov CI vectors, Hessian-vector products, and post-solve
Lagrange nuclear derivative assembly are represented in MPS form, so the
active-space response vector is not stored as a determinant ndarray.  Exact
small-active-space tests compare this backend against the FCI response path;
large-active-space runs use the same MPS operations without an FCI reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import time

import numpy as np

from cp_dmrg_response_mps import CPDMRGCASSCFResponseMPS
from mps_lagrange_assembly import (
    Lci_dot_dgci_dx_from_tdm,
    Lorb_dot_dgorb_dx_from_rdms,
)
from single_site_sigma import single_site_sigma_mps_native, site_tensor_to_fci
from site_replacement_density import (
    _block2_trans_rdm12_to_pyscf,
    _prepare_trans_npdm_mps,
    fci_to_mps_via_csf,
    transition_rdm_site_replacement_mps,
)


@dataclass
class MPSKrylovVector:
    """Mixed orbital/MPS vector used by the Arnoldi solver."""

    owner: "CPDMRGCASSCFResponseMPSKrylov"
    kappa: np.ndarray
    ci_mps: list
    label: str = "V"

    def inner(self, other: "MPSKrylovVector") -> float:
        return self.owner.vector_inner(self, other)

    def norm(self) -> float:
        return float(np.sqrt(max(self.inner(self), 0.0)))

    def scaled(self, scalar: float, label: str = "S") -> "MPSKrylovVector":
        return self.owner.vector_linear_combination([(float(scalar), self)], label=label)

    def add_scaled(
        self,
        other: "MPSKrylovVector",
        scalar: float,
        label: str = "A",
    ) -> "MPSKrylovVector":
        return self.owner.vector_linear_combination(
            [(1.0, self), (float(scalar), other)], label=label,
        )


class CPDMRGCASSCFResponseMPSKrylov(CPDMRGCASSCFResponseMPS):
    """CP response class with MPS-valued CI Krylov vectors.

    It does not call SciPy GMRES, because SciPy requires a flat dense vector.
    The local Arnoldi implementation stores each basis vector as
    :class:`MPSKrylovVector` and uses block2 MPS operations for the active-space
    component.
    """

    def __init__(self, mc, driver, mpo_active, *, mps_states=None,
                 weights=None, m_compress=None, sz_scratch_root=None,
                 backend: str = "freitag_reiher", mps_fit_sweeps: int = 10,
                 mps_fit_tol: float = 1.0e-10, mps_only: bool = False,
                 phase_ci_list=None, initial_guess: str = "zero",
                 initial_guess_sweeps: int = 4,
                 initial_guess_tol: float = 1.0e-6,
                 initial_guess_proj_weight: float = 20.0,
                 linear_solver: str = "gmres"):
        if mps_only:
            self.backend = backend
            self.mc = mc
            self.mol = mc.mol
            self.mf = mc._scf
            self.mo_coeff = mc.mo_coeff
            self.ncore = mc.ncore
            self.ncas = mc.ncas
            self.nmo = mc.mo_coeff.shape[1]
            nelec = mc.nelecas
            if isinstance(nelec, (tuple, list)):
                self.nelec = (int(nelec[0]), int(nelec[1]))
            else:
                self.nelec = (
                    int(nelec) // 2 + int(nelec) % 2,
                    int(nelec) // 2,
                )
            if mps_states is None:
                if not hasattr(mc.fcisolver, "_kets") or mc.fcisolver._kets is None:
                    raise ValueError(
                        "mps_only=True requires explicit mps_states or a "
                        "fcisolver with populated _kets."
                    )
                mps_states = list(mc.fcisolver._kets)
            self.ci_list = [None] * len(mps_states)
            self.nstates = len(mps_states)
            if weights is None:
                weights = np.ones(self.nstates) / self.nstates
            self.weights = np.asarray(weights)
            self._driver_su2 = driver
            self._mpo_active = mpo_active
            self._mps_states = list(mps_states)
            self._m_compress = (
                int(m_compress) if m_compress is not None
                else max(int(m.info.bond_dim) for m in self._mps_states)
            )
            import block2 as _block2
            self._su2_frame = _block2.Global.frame
            self._sz_frame = self._su2_frame
            self._driver_sz = None
            self._sz_scratch = None
            self._h_op_cache = None
            self._nrot_cache = None
            self._eris_cache = None
        else:
            super().__init__(
                mc, driver, mpo_active, mps_states=mps_states,
                weights=weights, m_compress=m_compress,
                sz_scratch_root=sz_scratch_root, backend=backend,
            )
        self._mps_fit_sweeps = int(mps_fit_sweeps)
        self._mps_fit_tol = float(mps_fit_tol)
        self._initial_guess = str(initial_guess).strip().lower()
        self._initial_guess_sweeps = int(initial_guess_sweeps)
        self._initial_guess_tol = float(initial_guess_tol)
        self._initial_guess_proj_weight = float(initial_guess_proj_weight)
        self._linear_solver = str(linear_solver).strip().lower()
        self._tag_counter = count()
        self._identity_mpo = None
        self._zero_state_mps_cache = {}
        self._mps_hcc_mpo = None
        self._phase_ci_list = phase_ci_list
        self._state_mps = self._build_phase_aligned_state_mps()
        self._hci0_mps_cache = None
        self._corr_mps_cache = None
        self._eci0_mps_cache = None
        self._state_transition_rdm_cache = {}
        self._timings = {}
        self._hcc_shifted_mpo_cache = {}
        self._gmres_recycle_cache = None

    def _is_cached_zero_mps(self, state: int, mps) -> bool:
        return self._zero_state_mps_cache.get(int(state)) is mps

    def _add_timing(self, key: str, seconds: float) -> None:
        self._timings[key] = self._timings.get(key, 0.0) + float(seconds)

    def _reset_timing(self) -> None:
        self._timings = {}

    # ------------------------------------------------------------------
    # Basic MPS helpers
    # ------------------------------------------------------------------

    def _new_tag(self, prefix: str) -> str:
        return f"KRY-{prefix}-{next(self._tag_counter)}"

    def _identity(self):
        if self._identity_mpo is None:
            with self._use_su2_frame():
                self._identity_mpo = self._driver_su2.get_identity_mpo()
        return self._identity_mpo

    def _canonical_kappa(self, kappa: np.ndarray) -> np.ndarray:
        """Project a full antisymmetric orbital matrix to PySCF's uniq-var space."""
        packed = self.mc.pack_uniq_var(np.asarray(kappa))
        if packed.size == 0:
            return np.zeros((self.nmo, self.nmo))
        return self.mc.unpack_uniq_var(packed)

    def _copy_mps(self, mps, tag: str):
        with self._use_su2_frame():
            return self._driver_su2.copy_mps(mps, tag=tag)

    def _mps_overlap(self, bra, ket) -> float:
        with self._use_su2_frame():
            return float(
                self._driver_su2.expectation(
                    bra, self._identity(), ket, iprint=0,
                )
            )

    def _scale_mps(self, mps, scalar: float, tag: str):
        return self._combine_mps([(float(scalar), mps)], tag=tag)

    def _zero_like(self, ref_mps, tag: str):
        return self._scale_mps(ref_mps, 0.0, tag=tag)

    def _combine_mps(self, terms, tag: str):
        """Fit a linear combination of MPS objects with block2 addition."""
        filtered = [(float(c), m) for c, m in terms if abs(float(c)) > 1.0e-15]
        if not filtered:
            ref = terms[0][1]
            filtered = [(0.0, ref)]

        def add_two(ca, ma, cb, mb, out_tag):
            bra = self._copy_mps(ma, out_tag)
            bd = max(int(ma.info.bond_dim), int(mb.info.bond_dim), int(bra.info.bond_dim))
            with self._use_su2_frame():
                self._driver_su2.addition(
                    bra, ma, mb,
                    mpo_a=float(ca), mpo_b=float(cb),
                    n_sweeps=self._mps_fit_sweeps,
                    tol=self._mps_fit_tol,
                    bra_bond_dims=[bd],
                    iprint=0,
                )
            return bra

        if len(filtered) == 1:
            c, m = filtered[0]
            if abs(c - 1.0) <= 1.0e-15:
                return self._copy_mps(m, tag)
            return add_two(c, m, 0.0, m, tag)

        c0, m0 = filtered[0]
        c1, m1 = filtered[1]
        acc = add_two(c0, m0, c1, m1, tag)
        for i, (c, m) in enumerate(filtered[2:], start=2):
            acc = add_two(1.0, acc, c, m, self._new_tag(f"{tag}-SUM{i}"))
        return acc

    def _build_phase_aligned_state_mps(self):
        """Return state MPSes in the same global phase as ``self.ci_list``.

        This uses the small-CAS CSF round-trip only for validation.  In a
        production trajectory, any continuous MPS phase convention can be used
        as long as it is applied consistently.
        """
        phase_refs = (
            self._phase_ci_list
            if self._phase_ci_list is not None
            else self.ci_list
        )
        if (
            self.ncas != 2
            or tuple(self.nelec) != (1, 1)
            or any(ci is None for ci in phase_refs)
        ):
            return list(self._mps_states)

        out = []
        with self._use_su2_frame():
            for i, (mps, ci) in enumerate(zip(self._mps_states, phase_refs)):
                ref = fci_to_mps_via_csf(
                    self._driver_su2, ci, self.ncas, self.nelec,
                    tag=self._new_tag(f"PHASE-REF-{i}"),
                )
                phase = float(self._driver_su2.expectation(
                    ref, self._identity(), mps, iprint=0,
                ))
                sign = float(np.sign(phase) or 1.0)
                out.append(self._scale_mps(
                    mps, sign, tag=self._new_tag(f"STATE-{i}"),
                ))
        return out

    def _state_rdm12_mps(self, mps, tag: str):
        with self._use_su2_frame():
            safe = _prepare_trans_npdm_mps(
                self._driver_su2, mps, tag=self._new_tag(tag),
            )
            dm1_b = self._driver_su2.get_1pdm(safe, iprint=0)
            dm2_b = self._driver_su2.get_2pdm(safe, iprint=0)
        return _block2_trans_rdm12_to_pyscf(np.asarray(dm1_b), np.asarray(dm2_b))

    def _build_eris_cache(self):
        """Build response integral/RDM cache using state MPS RDMs.

        This mirrors the parent cache algebra but replaces the per-state
        ``direct_spin1.make_rdm12`` calls with block2 MPS RDM contractions.
        """
        if getattr(self, "_eris_cache", None) is not None:
            return self._eris_cache

        ncore, ncas, nmo, nocc = (
            self.ncore, self.ncas, self.nmo, self.ncore + self.ncas
        )
        eris = self.mc.ao2mo(self.mo_coeff)

        casdm1_per = []
        casdm2_per = []
        for i, state in enumerate(self._state_mps):
            d1, d2 = self._state_rdm12_mps(state, tag=f"STATE-RDM-{i}")
            casdm1_per.append(d1)
            casdm2_per.append(d2)
        casdm1_per = np.asarray(casdm1_per)
        casdm2_per = np.asarray(casdm2_per)
        casdm1_avg = np.einsum("r,rpq->pq", self.weights, casdm1_per)
        casdm2_avg = np.einsum("r,rpqst->pqst", self.weights, casdm2_per)

        nroots = self.nstates
        paaa = np.empty((nmo, ncas, ncas, ncas))
        vhf_a_state = np.empty((nroots, nmo, nmo))
        g_dm2_state = np.empty((nroots, nmo, ncas))
        hdm2_state = np.empty((nroots, nmo, ncas, nmo, ncas))
        eri_cas = np.empty((ncas, ncas, ncas, ncas))
        dm2tmp = (
            casdm2_per.transpose(0, 2, 3, 1, 4)
            + casdm2_per.transpose(0, 1, 3, 2, 4)
        ).reshape(nroots, ncas**2, -1)
        jtmp = np.empty((nroots, nmo, ncas, ncas))
        ktmp = np.empty((nroots, nmo, ncas, ncas))
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            paaa[i] = jbuf[ncore:nocc]
            for r in range(nroots):
                vhf_a_state[r, i] = (
                    np.einsum("quv,uv->q", jbuf, casdm1_per[r])
                    - np.einsum("uqv,uv->q", kbuf, casdm1_per[r]) * 0.5
                )
                jtmp[r] = (
                    jbuf.reshape(nmo, -1)
                    @ casdm2_per[r].reshape(ncas * ncas, -1)
                ).reshape(nmo, ncas, ncas)
                ktmp[r] = (
                    kbuf.transpose(1, 0, 2).reshape(nmo, -1)
                    @ dm2tmp[r]
                ).reshape(nmo, ncas, ncas)
            if ncore <= i < nocc:
                eri_cas[i - ncore] = jbuf[ncore:nocc]
            hdm2_state[:, i] = (ktmp + jtmp).transpose(0, 2, 1, 3)

        for r in range(nroots):
            g_dm2_state[r] = np.einsum(
                "puwx,wxuv->pv", paaa, casdm2_per[r],
            )

        vhf_ca = eris.vhf_c[None] + vhf_a_state
        h1e_mo = self.mo_coeff.T @ self.mc.get_hcore() @ self.mo_coeff

        gpq = np.zeros((nroots, nmo, nmo))
        gpq[:, :, :ncore] = (
            h1e_mo[None, :, :ncore] + vhf_ca[:, :, :ncore]
        ) * 2
        gpq[:, :, ncore:nocc] = np.einsum(
            "pa,rab->rpb",
            h1e_mo[:, ncore:nocc] + eris.vhf_c[:, ncore:nocc],
            casdm1_per,
        )
        gpq[:, :, ncore:nocc] += g_dm2_state

        vhf_ca_avg = np.einsum("r,rpq->pq", self.weights, vhf_ca)
        hdm2_avg = np.einsum("r,rpqst->pqst", self.weights, hdm2_state)
        eri_cas = np.ascontiguousarray(eri_cas)
        h1cas_0 = (
            h1e_mo[ncore:nocc, ncore:nocc]
            + eris.vhf_c[ncore:nocc, ncore:nocc]
        )

        self._eris_cache = dict(
            eris=eris,
            h1e_mo=h1e_mo,
            paaa=paaa,
            eri_cas=eri_cas,
            vhf_a_state=vhf_a_state,
            vhf_ca_avg=vhf_ca_avg,
            casdm1_per=casdm1_per,
            casdm2_per=casdm2_per,
            casdm1_avg=casdm1_avg,
            casdm2_avg=casdm2_avg,
            gpq=gpq,
            hdm2_avg=hdm2_avg,
            h1cas_0=h1cas_0,
            hci0=None,
            eci0=None,
        )
        return self._eris_cache

    def _gpq_from_active_rdms(self, casdm1, casdm2, *, vhf_c=None):
        """Orbital-gradient matrix ``g_pq`` for one active-space density.

        This mirrors the ``gpq`` construction in
        ``pyscf.mcscf.newton_casscf.gen_g_hop`` but accepts the active-space
        1/2-RDMs directly.  It is used to build response RHS vectors without
        calling PySCF's dense-CI ``gen_g_hop`` path.
        """
        cache = self._build_eris_cache()
        eris = cache["eris"]
        h1e_mo = cache["h1e_mo"]
        paaa = cache["paaa"]
        if vhf_c is None:
            vhf_c = eris.vhf_c

        casdm1 = np.asarray(casdm1)
        casdm2 = np.asarray(casdm2)
        ncore, ncas, nmo, nocc = (
            self.ncore, self.ncas, self.nmo, self.ncore + self.ncas
        )
        vhf_a = np.empty((nmo, nmo))
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            vhf_a[i] = (
                np.einsum("quv,uv->q", jbuf, casdm1)
                - np.einsum("uqv,uv->q", kbuf, casdm1) * 0.5
            )

        gpq = np.zeros((nmo, nmo))
        vhf_ca = np.asarray(vhf_c) + vhf_a
        gpq[:, :ncore] = (
            h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
        ) * 2
        gpq[:, ncore:nocc] = (
            h1e_mo[:, ncore:nocc] + np.asarray(vhf_c)[:, ncore:nocc]
        ) @ casdm1
        gpq[:, ncore:nocc] += np.einsum("puwx,wxuv->pv", paaa, casdm2)
        return gpq

    # ------------------------------------------------------------------
    # Conversion helpers for validation and tests
    # ------------------------------------------------------------------

    def ci_mps_from_fci_list(self, ci_list, label: str):
        """Build MPS CI blocks from FCI arrays for validation inputs/RHS."""
        if self.ncas != 2 or tuple(self.nelec) != (1, 1):
            raise NotImplementedError(
                "FCI to SU2 MPS validation conversion is implemented only for "
                "CAS(2,2) singlet."
            )
        out = []
        with self._use_su2_frame():
            for i, ci in enumerate(ci_list):
                ci = np.asarray(ci)
                if np.any(np.abs(ci) > 1.0e-14):
                    out.append(fci_to_mps_via_csf(
                        self._driver_su2, ci, self.ncas, self.nelec,
                        tag=self._new_tag(f"{label}-CI{i}"),
                    ))
                else:
                    out.append(self._zero_like(
                        self._state_mps[i], tag=self._new_tag(f"{label}-ZERO{i}"),
                    ))
        return out

    def vector_from_fci(
        self,
        kappa: np.ndarray,
        ci_list,
        label: str = "FROMFCI",
    ) -> MPSKrylovVector:
        return MPSKrylovVector(
            owner=self,
            kappa=self._canonical_kappa(kappa),
            ci_mps=self.ci_mps_from_fci_list(ci_list, label=label),
            label=label,
        )

    def vector_to_fci_list(self, vec: MPSKrylovVector):
        if self.ncas != 2 or tuple(self.nelec) != (1, 1):
            raise NotImplementedError(
                "MPS to FCI validation conversion is implemented only for "
                "CAS(2,2) singlet."
            )
        out = []
        with self._use_su2_frame():
            for mps in vec.ci_mps:
                out.append(site_tensor_to_fci(
                    self._driver_su2, mps, self.ncas, self.nelec,
                ))
        return out

    def flatten_for_validation(self, vec: MPSKrylovVector) -> np.ndarray:
        return self._flatten(vec.kappa, self.vector_to_fci_list(vec))

    # ------------------------------------------------------------------
    # MPS vector-space operations
    # ------------------------------------------------------------------

    def vector_inner(self, a: MPSKrylovVector, b: MPSKrylovVector) -> float:
        orb = float(np.dot(
            self.mc.pack_uniq_var(a.kappa),
            self.mc.pack_uniq_var(b.kappa),
        ))
        ci = 0.0
        for istate, (ma, mb) in enumerate(zip(a.ci_mps, b.ci_mps)):
            if (
                self._is_cached_zero_mps(istate, ma)
                or self._is_cached_zero_mps(istate, mb)
            ):
                continue
            ci += self._mps_overlap(ma, mb)
        return float(orb + ci)

    def vector_linear_combination(self, terms, label: str) -> MPSKrylovVector:
        """Return ``sum_i coeff_i * vector_i`` in mixed orbital/MPS form."""
        if not terms:
            raise ValueError("empty vector linear combination")
        kappa = np.zeros_like(terms[0][1].kappa)
        for coeff, vec in terms:
            kappa += float(coeff) * vec.kappa
        kappa = self._canonical_kappa(kappa)

        ci_out = []
        for istate in range(self.nstates):
            ci_terms = [
                (float(coeff), vec.ci_mps[istate])
                for coeff, vec in terms
                if (
                    abs(float(coeff)) > 1.0e-15
                    and not self._is_cached_zero_mps(istate, vec.ci_mps[istate])
                )
            ]
            if not ci_terms:
                ci_out.append(self._zero_state_mps(istate))
                continue
            ci_out.append(self._combine_mps(
                ci_terms, tag=self._new_tag(f"{label}-CI{istate}"),
            ))
        return MPSKrylovVector(self, kappa, ci_out, label=label)

    # ------------------------------------------------------------------
    # MPS Hessian blocks
    # ------------------------------------------------------------------

    def _hcc_mpo(self):
        if self._mps_hcc_mpo is None:
            cache = self._build_eris_cache()
            with self._use_su2_frame():
                self._mps_hcc_mpo = self._driver_su2.get_qc_mpo(
                    np.asarray(cache["h1cas_0"]),
                    np.asarray(cache["eri_cas"]),
                    ecore=0.0,
                    iprint=0,
                )
        return self._mps_hcc_mpo

    def _hcc_shifted_mpo(self, state: int):
        state = int(state)
        if state not in self._hcc_shifted_mpo_cache:
            cache = self._build_eris_cache()
            self._build_hcc_state_cache()
            with self._use_su2_frame():
                self._hcc_shifted_mpo_cache[state] = self._driver_su2.get_qc_mpo(
                    np.asarray(cache["h1cas_0"]),
                    np.asarray(cache["eri_cas"]),
                    ecore=-float(self._eci0_mps_cache[state]),
                    iprint=0,
                )
        return self._hcc_shifted_mpo_cache[state]

    def _sigma_mps(self, mpo, mps, tag: str):
        with self._use_su2_frame():
            sig, _ = single_site_sigma_mps_native(
                self._driver_su2, mpo, mps,
                out_tag=tag,
                n_sweeps=self._mps_fit_sweeps,
                tol=self._mps_fit_tol,
                M_compress=self._m_compress,
                iprint=0,
            )
        return sig

    def _build_hcc_state_cache(self):
        if self._hci0_mps_cache is not None:
            return
        mpo = self._hcc_mpo()
        hci0 = []
        corr = []
        eci0 = []
        for i, state in enumerate(self._state_mps):
            sig = self._sigma_mps(mpo, state, tag=self._new_tag(f"HCI0-{i}"))
            e = self._mps_overlap(state, sig)
            c = self._combine_mps(
                [(1.0, sig), (-e, state)],
                tag=self._new_tag(f"CORR-{i}"),
            )
            hci0.append(sig)
            corr.append(c)
            eci0.append(float(e))
        self._hci0_mps_cache = hci0
        self._corr_mps_cache = corr
        self._eci0_mps_cache = eci0

    def H_CC_apply_mps(self, ci_mps_list) -> list:
        self._build_hcc_state_cache()
        out = []
        for i, (trial, state, corr, e, w) in enumerate(zip(
            ci_mps_list,
            self._state_mps,
            self._corr_mps_cache,
            self._eci0_mps_cache,
            self.weights,
        )):
            if self._is_cached_zero_mps(i, trial):
                out.append(self._zero_state_mps(i))
                continue
            # Use (H - E_i) as the MPO so the output combination has one fewer
            # fitted MPS term in every Krylov matvec.
            sig = self._sigma_mps(
                self._hcc_shifted_mpo(i), trial,
                tag=self._new_tag(f"HCC-SIG-{i}"),
            )
            ovlp = self._mps_overlap(state, trial)
            corr_dot = self._mps_overlap(state, sig)
            scale = 2.0 * float(w)
            out_i = self._combine_mps(
                [
                    (scale, sig),
                    (-scale * ovlp, corr),
                    (-scale * corr_dot, state),
                ],
                tag=self._new_tag(f"HCC-OUT-{i}"),
            )
            out.append(out_i)
        return out

    def H_CO_apply_mps(self, kappa: np.ndarray) -> list:
        if not np.any(np.abs(kappa) > 1.0e-14):
            return self.zero_ci_mps_list(label="HCO-ZERO")
        cache = self._build_eris_cache()
        eris = cache["eris"]
        h1e_mo = cache["h1e_mo"]
        paaa = cache["paaa"]
        ncore, ncas, nmo, nocc = (
            self.ncore, self.ncas, self.nmo, self.ncore + self.ncas
        )
        rc = kappa[:, :ncore]
        ra = kappa[:, ncore:nocc]

        ddm_c = np.zeros((nmo, nmo))
        ddm_c[:, :ncore] = rc[:, :ncore] * 2
        ddm_c[:ncore, :] += rc[:, :ncore].T * 2

        jk = np.zeros((ncas, ncas))
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            jk += np.einsum("quv,q->uv", jbuf, ddm_c[i])
            jk -= np.einsum("uqv,q->uv", kbuf, ddm_c[i]) * 0.5

        aaaa = np.dot(ra.T, paaa.reshape(nmo, -1)).reshape([ncas] * 4)
        aaaa = aaaa + aaaa.transpose(1, 0, 2, 3)
        aaaa = aaaa + aaaa.transpose(2, 3, 0, 1)

        h1aa = np.dot(h1e_mo[ncore:nocc] + eris.vhf_c[ncore:nocc], ra)
        h1aa = h1aa + h1aa.T + jk

        with self._use_su2_frame():
            mpo = self._driver_su2.get_qc_mpo(
                np.asarray(h1aa), np.asarray(aaaa), ecore=0.0, iprint=0,
            )

        out = []
        for i, (state, w) in enumerate(zip(self._state_mps, self.weights)):
            sig = self._sigma_mps(mpo, state, tag=self._new_tag(f"HCO-SIG-{i}"))
            proj = self._mps_overlap(state, sig)
            scale = 2.0 * float(w)
            projected = self._combine_mps(
                [(scale, sig), (-scale * proj, state)],
                tag=self._new_tag(f"HCO-PROJ-{i}"),
            )
            out.append(projected)
        return out

    def H_OC_apply_mps(self, ci_mps_list) -> np.ndarray:
        if all(
            self._is_cached_zero_mps(i, trial)
            for i, trial in enumerate(ci_mps_list)
        ):
            return np.zeros((self.nmo, self.nmo))

        cache = self._build_eris_cache()
        eris = cache["eris"]
        h1e_mo = cache["h1e_mo"]
        paaa = cache["paaa"]
        gpq = cache["gpq"]

        ncore, ncas, nmo, nocc = (
            self.ncore, self.ncas, self.nmo, self.ncore + self.ncas
        )
        tdm1 = np.zeros((ncas, ncas))
        tdm2 = np.zeros((ncas, ncas, ncas, ncas))
        s10 = np.zeros(self.nstates)
        for i, (trial, state, w) in enumerate(zip(
            ci_mps_list, self._state_mps, self.weights,
        )):
            if self._is_cached_zero_mps(i, trial):
                continue
            with self._use_su2_frame():
                d1, d2 = transition_rdm_site_replacement_mps(
                    self._driver_su2, trial, state, ncas, self.nelec,
                    trial_tag=self._new_tag(f"HOC-{i}"),
                )
            tdm1 += float(w) * d1
            tdm2 += float(w) * d2
            s10[i] = self._mps_overlap(trial, state) * 2.0 * float(w)

        tdm1 = tdm1 + tdm1.T
        tdm2 = tdm2 + tdm2.transpose(1, 0, 3, 2)
        tdm2 = (tdm2 + tdm2.transpose(2, 3, 0, 1)) * 0.5

        vhf_a = np.empty((nmo, ncore))
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            vhf_a[i] = np.einsum("quv,uv->q", jbuf[:ncore], tdm1)
            vhf_a[i] -= np.einsum("uqv,uv->q", kbuf[:, :ncore], tdm1) * 0.5

        g_dm2 = np.einsum("puwx,wxuv->pv", paaa, tdm2)
        x2 = np.zeros((nmo, nmo))
        x2[:, :ncore] += (
            (h1e_mo[:, :ncore] + eris.vhf_c[:, :ncore]) * s10.sum()
            + vhf_a
        ) * 2
        x2[:, ncore:nocc] += (
            h1e_mo[:, ncore:nocc] + eris.vhf_c[:, ncore:nocc]
        ) @ tdm1
        x2[:, ncore:nocc] += g_dm2
        x2 -= np.einsum("r,rpq->pq", s10, gpq)
        return (x2 - x2.T) * 2.0

    def H_OO_apply(self, kappa: np.ndarray) -> np.ndarray:
        """Orbital-orbital Hessian block from MPS-derived SA RDMs.

        This replaces the parent implementation that calls PySCF
        ``newton_casscf.gen_g_hop`` with dense CI vectors.  The algebra follows
        the H_oo block in PySCF's Newton CASSCF code, using cached state RDMs
        obtained from block2 MPS contractions.
        """
        kappa = self._canonical_kappa(kappa)
        kappa_packed = self.mc.pack_uniq_var(kappa)
        if kappa_packed.size == 0:
            return np.zeros_like(kappa)

        cache = self._build_eris_cache()
        eris = cache["eris"]
        h1e_mo = cache["h1e_mo"]
        casdm1 = cache["casdm1_avg"]
        gpq = cache["gpq"]
        vhf_ca = cache["vhf_ca_avg"]
        hdm2 = cache["hdm2_avg"]
        ncore, ncas, nmo, nocc = (
            self.ncore, self.ncas, self.nmo, self.ncore + self.ncas
        )

        dm1 = np.zeros((nmo, nmo))
        idx = np.arange(ncore)
        dm1[idx, idx] = 2.0
        dm1[ncore:nocc, ncore:nocc] = casdm1

        x1 = kappa
        x2 = h1e_mo @ x1 @ dm1
        g_orb = np.einsum("r,rpq->pq", self.weights, gpq)
        x2 -= (g_orb + g_orb.T) @ x1 * 0.5
        x2[:ncore] += x1[:ncore, ncore:] @ vhf_ca[ncore:] * 2.0
        x2[ncore:nocc] += casdm1 @ x1[ncore:nocc] @ eris.vhf_c
        x2[:, ncore:nocc] += np.einsum(
            "purv,rv->pu", hdm2, x1[:, ncore:nocc],
        )

        if ncore > 0:
            va, vc = self.mc.update_jk_in_ah(
                self.mo_coeff, x1, casdm1, eris,
            )
            x2[ncore:nocc] += va
            x2[:ncore, ncore:] += vc

        return self._canonical_kappa((x2 - x2.T) * 2.0)

    def weighted_lci_transition_rdm_mps(self, ci_mps_list):
        """Weighted SA transition RDM ``sum_I w_I <L_I|E|I>`` from MPS Lci."""
        tdm1 = np.zeros((self.ncas, self.ncas))
        tdm2 = np.zeros((self.ncas, self.ncas, self.ncas, self.ncas))
        for i, (trial, state, w) in enumerate(zip(
            ci_mps_list, self._state_mps, self.weights,
        )):
            if self._is_cached_zero_mps(i, trial):
                continue
            with self._use_su2_frame():
                d1, d2 = transition_rdm_site_replacement_mps(
                    self._driver_su2, trial, state, self.ncas, self.nelec,
                    trial_tag=self._new_tag(f"LCI-TDM-{i}"),
                )
            tdm1 += float(w) * d1
            tdm2 += float(w) * d2
        return tdm1, tdm2

    def Lci_dot_dgci_dx_mps(
        self,
        lci_mps_list,
        *,
        mo_coeff=None,
        atmlst=None,
        mf_grad=None,
        eris=None,
        verbose=None,
    ):
        """Assemble the CI-Lagrange nuclear derivative from MPS Lci vectors."""
        tdm1, tdm2 = self.weighted_lci_transition_rdm_mps(lci_mps_list)
        return Lci_dot_dgci_dx_from_tdm(
            tdm1, tdm2, self.mc,
            mo_coeff=self.mo_coeff if mo_coeff is None else mo_coeff,
            atmlst=atmlst,
            mf_grad=mf_grad,
            eris=eris,
            verbose=verbose,
            symmetrize=True,
        )

    def LdotJnuc_mps(
        self,
        kappa: np.ndarray,
        lci_mps_list,
        *,
        mo_coeff=None,
        ci=None,
        atmlst=None,
        mf_grad=None,
        eris=None,
        verbose=None,
    ):
        """Full Lagrange nuclear derivative from dense orbital and MPS CI parts."""
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci is None:
            ci = self.ci_list
        de_lci = self.Lci_dot_dgci_dx_mps(
            lci_mps_list,
            mo_coeff=mo_coeff,
            atmlst=atmlst,
            mf_grad=mf_grad,
            eris=eris,
            verbose=verbose,
        )
        cache = self._build_eris_cache()
        de_lorb = Lorb_dot_dgorb_dx_from_rdms(
            np.asarray(kappa),
            cache["casdm1_avg"],
            cache["casdm2_avg"],
            self.mc,
            mo_coeff=mo_coeff,
            atmlst=atmlst,
            mf_grad=mf_grad,
            eris=eris,
            verbose=verbose,
        )
        return de_lci + de_lorb

    def matvec_mps(self, vec: MPSKrylovVector) -> MPSKrylovVector:
        kappa_in = self._canonical_kappa(vec.kappa)
        t0 = time.perf_counter()
        kappa_hoc = self.H_OC_apply_mps(vec.ci_mps)
        self._add_timing("H_OC_apply_mps", time.perf_counter() - t0)
        t0 = time.perf_counter()
        ci_hco = self.H_CO_apply_mps(kappa_in)
        self._add_timing("H_CO_apply_mps", time.perf_counter() - t0)
        t0 = time.perf_counter()
        ci_hcc = self.H_CC_apply_mps(vec.ci_mps)
        self._add_timing("H_CC_apply_mps", time.perf_counter() - t0)
        t0 = time.perf_counter()
        kappa_hoo = (
            self.H_OO_apply(kappa_in)
            if kappa_in.size else np.zeros_like(kappa_in)
        )
        self._add_timing("H_OO_apply", time.perf_counter() - t0)
        kappa_out = self._canonical_kappa(kappa_hoo + kappa_hoc)

        ci_out = []
        t0 = time.perf_counter()
        for i, (a, b) in enumerate(zip(ci_hco, ci_hcc)):
            merged = self._combine_mps(
                [(1.0, a), (1.0, b)],
                tag=self._new_tag(f"MV-CI{i}"),
            )
            for j, state in enumerate(self._state_mps):
                ovlp = self._mps_overlap(merged, state)
                if abs(ovlp) > 1.0e-14:
                    merged = self._combine_mps(
                        [(1.0, merged), (-ovlp, state)],
                        tag=self._new_tag(f"MV-PROJ{i}-{j}"),
                    )
            ci_out.append(merged)
        self._add_timing("matvec_ci_merge_project", time.perf_counter() - t0)

        return MPSKrylovVector(self, kappa_out, ci_out, label="A" + vec.label)

    # ------------------------------------------------------------------
    # Pure-MPS Arnoldi/GMRES
    # ------------------------------------------------------------------

    def zero_vector_mps(self, label: str = "ZERO") -> MPSKrylovVector:
        return MPSKrylovVector(
            self,
            np.zeros((self.nmo, self.nmo)),
            self.zero_ci_mps_list(label=label),
            label=label,
        )

    def _zero_state_mps(self, state: int):
        state = int(state)
        if state not in self._zero_state_mps_cache:
            self._zero_state_mps_cache[state] = self._zero_like(
                self._state_mps[state],
                tag=self._new_tag(f"ZERO-STATE-{state}"),
            )
        return self._zero_state_mps_cache[state]

    def zero_ci_mps_list(self, label: str = "ZERO-CI") -> list:
        return [self._zero_state_mps(i) for i in range(self.nstates)]

    def project_ci_mps_list(
        self,
        ci_mps_list,
        label: str = "PROJ-CI",
        *,
        mode: str = "slot",
    ):
        """Project CI-MPS slots out of the SA reference-root subspace.

        PySCF's dense response solver projects each CI Lagrange multiplier
        against its corresponding reference root after solving.  MPS linear
        combinations/compressions can reintroduce small root components, so
        the production MPS path enforces the same final gauge immediately
        before the Lagrange nuclear derivative is assembled.
        """
        mode = str(mode).strip().lower()
        if mode not in {"slot", "all"}:
            raise ValueError(f"unknown CI-MPS projection mode {mode!r}")
        projected = []
        max_before = 0.0
        max_after = 0.0
        for i, mps in enumerate(ci_mps_list):
            if self._is_cached_zero_mps(i, mps):
                projected.append(self._zero_state_mps(i))
                continue
            terms = [(1.0, mps)]
            refs = (
                [(i, self._state_mps[i])]
                if mode == "slot"
                else list(enumerate(self._state_mps))
            )
            for j, state in refs:
                ovlp = self._mps_overlap(state, mps)
                max_before = max(max_before, abs(float(ovlp)))
                if abs(ovlp) > 1.0e-14:
                    terms.append((-float(ovlp), state))
            if len(terms) == 1:
                out = self._copy_mps(mps, tag=self._new_tag(f"{label}-{i}"))
            else:
                out = self._combine_mps(
                    terms, tag=self._new_tag(f"{label}-{i}"),
                )
            for _, state in refs:
                max_after = max(max_after, abs(float(self._mps_overlap(state, out))))
            projected.append(out)
        return projected, {
            "max_root_overlap_before_projection": float(max_before),
            "max_root_overlap_after_projection": float(max_after),
        }

    def project_vector_ci_mps(
        self,
        vec: MPSKrylovVector,
        label: str,
        *,
        mode: str = "all",
    ):
        ci_mps, meta = self.project_ci_mps_list(
            vec.ci_mps, label=label, mode=mode,
        )
        return MPSKrylovVector(self, vec.kappa, ci_mps, label=vec.label), meta

    def _hcc_inverse_initial_guess(self, rhs: MPSKrylovVector):
        """Approximate CI initial guess from block2 inverse-MPO fitting.

        This is a correction-vector style preconditioner for the active-space
        CI slots only.  The subsequent MPS-GMRES still solves the full coupled
        response equation and corrects this initial guess, so this optional
        path should only affect convergence speed.
        """
        ci_guess = []
        t0 = time.perf_counter()
        for i, rhs_i in enumerate(rhs.ci_mps):
            if self._is_cached_zero_mps(i, rhs_i):
                ci_guess.append(self._zero_state_mps(i))
                continue
            bra = self._copy_mps(
                rhs_i,
                tag=self._new_tag(f"X0-HCC-BRA-{i}"),
            )
            with self._use_su2_frame():
                self._driver_su2.multiply(
                    bra,
                    self._identity(),
                    rhs_i,
                    left_mpo=self._hcc_shifted_mpo(i),
                    n_sweeps=max(self._initial_guess_sweeps, 1),
                    tol=self._initial_guess_tol,
                    bra_bond_dims=[self._m_compress],
                    proj_mpss=[self._state_mps[i]],
                    proj_weights=[self._initial_guess_proj_weight],
                    linear_max_iter=4000,
                    iprint=0,
                )
            ovlp = self._mps_overlap(self._state_mps[i], bra)
            if abs(ovlp) > 1.0e-14:
                bra = self._combine_mps(
                    [(1.0, bra), (-ovlp, self._state_mps[i])],
                    tag=self._new_tag(f"X0-HCC-PROJ-{i}"),
                )
            ci_guess.append(bra)
        self._add_timing("initial_guess_hcc_inverse", time.perf_counter() - t0)
        return MPSKrylovVector(
            self,
            np.zeros_like(rhs.kappa),
            ci_guess,
            label="X0-HCC",
        )

    def _gmres_recycle_initial_guess(self, rhs: MPSKrylovVector):
        """Build an initial guess from the previous GMRES Arnoldi subspace.

        The SA-CASSCF response operator is identical for the gradient and NAC
        RHSs at a fixed geometry.  After solving one RHS, the Arnoldi relation
        ``A Q_n = Q_{n+1} H`` gives a cheap least-squares approximation for a
        subsequent RHS without extra MPS matvecs:

            min_y || P_Q b - Q_{n+1} H y ||,  x0 = Q_n y.

        This is a recycled-subspace initial guess only; the following GMRES
        solve still forms the explicit residual and converges to the requested
        tolerance.
        """
        cache = self._gmres_recycle_cache
        if not cache:
            return None
        q = list(cache.get("q", ()))
        h = np.asarray(cache.get("h", np.zeros((0, 0))), dtype=float)
        if h.ndim != 2 or h.shape[1] == 0 or len(q) < h.shape[0]:
            return None
        t0 = time.perf_counter()
        coeff_rhs = np.array([q[i].inner(rhs) for i in range(h.shape[0])])
        y, *_ = np.linalg.lstsq(h, coeff_rhs, rcond=None)
        terms = [
            (float(c), q[i])
            for i, c in enumerate(y[:h.shape[1]])
            if abs(float(c)) > 1.0e-14
        ]
        self._add_timing("initial_guess_gmres_recycle_project",
                         time.perf_counter() - t0)
        if not terms:
            return None
        t0 = time.perf_counter()
        x0 = self.vector_linear_combination(terms, label="X0-RECYCLE")
        self._add_timing("initial_guess_gmres_recycle_build",
                         time.perf_counter() - t0)
        return x0

    def initial_guess_vector_mps(self, rhs: MPSKrylovVector):
        if self._initial_guess in {"zero", "none", ""}:
            return None
        if self._initial_guess in {
            "gmres-recycle", "gmres_recycle", "recycle", "recycled-gmres",
        }:
            return self._gmres_recycle_initial_guess(rhs)
        if self._initial_guess in {
            "hcc-inverse", "hcc_inverse", "ci-hcc-inverse",
        }:
            return self._hcc_inverse_initial_guess(rhs)
        raise ValueError(f"Unknown MPS response initial_guess={self._initial_guess!r}")

    def build_rhs_mps(self, state: int) -> MPSKrylovVector:
        """Build a state-gradient response RHS without dense CI vectors.

        PySCF's dense-CI RHS is the state-specific orbital gradient plus the
        CI residual for the target state.  The orbital part is computed from
        the target state's MPS RDMs; the CI part is the active-space residual
        ``(H-E)|Psi_state>`` represented directly as an MPS.  For a fully
        converged DMRG eigenstate this CI residual should be numerically small,
        but retaining it preserves the PySCF response convention.
        """
        state = int(state)
        cache = self._build_eris_cache()
        gpq_state = cache["gpq"][state]
        rhs_kappa = -self._canonical_kappa((gpq_state - gpq_state.T) * 2.0)

        self._build_hcc_state_cache()
        rhs_ci = self.zero_ci_mps_list(label=f"RHS-G{state}")
        rhs_ci[state] = self._scale_mps(
            self._corr_mps_cache[state],
            -2.0,
            tag=self._new_tag(f"RHS-G{state}-CI"),
        )
        return MPSKrylovVector(self, rhs_kappa, rhs_ci, label=f"RHS-G{state}")

    def _state_transition_rdm_mps(self, bra: int, ket: int):
        key = (int(bra), int(ket))
        if key in self._state_transition_rdm_cache:
            d1, d2 = self._state_transition_rdm_cache[key]
            return d1.copy(), d2.copy()
        with self._use_su2_frame():
            d1, d2 = transition_rdm_site_replacement_mps(
                self._driver_su2,
                self._state_mps[int(bra)],
                self._state_mps[int(ket)],
                self.ncas,
                self.nelec,
                trial_tag=self._new_tag(f"STATE-TDM-{bra}-{ket}"),
            )
        self._state_transition_rdm_cache[key] = (np.asarray(d1), np.asarray(d2))
        return d1.copy(), d2.copy()

    def build_rhs_nac_mps(self, state_pair: tuple[int, int]) -> MPSKrylovVector:
        """Build a NAC response RHS without dense CI vectors.

        The convention follows ``pyscf.nac.sacasscf``: ``state_pair`` is
        ``(ket, bra)``.  The orbital RHS is built from the symmetrized
        transition density ``<bra|E|ket>`` with the active-space core
        correction used by PySCF's NAC implementation.  The CI slots are MPS
        residuals crossed between the two roots, matching PySCF's dense-CI
        assignment.
        """
        ket, bra = int(state_pair[0]), int(state_pair[1])
        d1, d2 = self._state_transition_rdm_mps(bra, ket)
        castm1 = 0.5 * (d1 + d1.T)
        castm2 = 0.5 * (d2 + d2.transpose(1, 0, 3, 2))

        cache = self._build_eris_cache()
        eris = cache["eris"]
        moH = self.mo_coeff.conj().T
        vnocore = np.asarray(eris.vhf_c).copy()
        vnocore[:, :self.ncore] = (
            -moH @ self.mc.get_hcore() @ self.mo_coeff[:, :self.ncore]
        )
        gpq = self._gpq_from_active_rdms(castm1, castm2, vhf_c=vnocore)
        rhs_kappa = -self._canonical_kappa((gpq - gpq.T) * 2.0)

        self._build_hcc_state_cache()
        rhs_ci = self.zero_ci_mps_list(label=f"RHS-NAC{ket}-{bra}")

        # PySCF assigns 0.5*g_all_ket to the bra slot and 0.5*g_all_bra to
        # the ket slot.  Since g_all_ci = 2*(H-E)|Psi>, the assigned vector is
        # just the active-space residual MPS.
        corr_bra = self._copy_mps(
            self._corr_mps_cache[bra],
            tag=self._new_tag(f"RHS-NAC-CORR-BRA{bra}"),
        )
        ov_bra = self._mps_overlap(self._state_mps[bra], corr_bra)
        rhs_ci[ket] = self._combine_mps(
            [(1.0, corr_bra), (-ov_bra, self._state_mps[bra])],
            tag=self._new_tag(f"RHS-NAC-KET{ket}"),
        )

        corr_ket = self._copy_mps(
            self._corr_mps_cache[ket],
            tag=self._new_tag(f"RHS-NAC-CORR-KET{ket}"),
        )
        ov_ket = self._mps_overlap(self._state_mps[ket], corr_ket)
        rhs_ci[bra] = self._combine_mps(
            [(1.0, corr_ket), (-ov_ket, self._state_mps[ket])],
            tag=self._new_tag(f"RHS-NAC-BRA{bra}"),
        )

        rhs_ci = [
            self._scale_mps(v, -1.0, tag=self._new_tag(f"RHS-NAC-SIGN{i}"))
            for i, v in enumerate(rhs_ci)
        ]
        return MPSKrylovVector(self, rhs_kappa, rhs_ci, label=f"RHS-NAC{ket}-{bra}")

    def gmres_mps(
        self,
        rhs: MPSKrylovVector,
        *,
        tol: float = 1.0e-8,
        max_iter: int = 30,
        verbose: bool = False,
    ):
        """Solve ``A x = rhs`` with Arnoldi basis vectors stored as MPS."""
        self._reset_timing()
        x0 = self.initial_guess_vector_mps(rhs)
        if x0 is not None:
            t0 = time.perf_counter()
            ax0 = self.matvec_mps(x0)
            rhs = rhs.add_scaled(ax0, -1.0, label="RHS-X0")
            self._add_timing("initial_guess_residual_build", time.perf_counter() - t0)

        beta = rhs.norm()
        if beta < 1.0e-14:
            sol = self.zero_vector_mps("SOL-ZERO") if x0 is None else x0
            return sol, 0, {
                "residual": 0.0,
                "niter": 0,
                "timings_s": {},
                "initial_guess": self._initial_guess,
            }
        conv_abs = float(tol)
        conv_rel = float(tol) * beta

        q = [rhs.scaled(1.0 / beta, label="Q0")]
        h = np.zeros((max_iter + 1, max_iter), dtype=float)
        y_best = None
        n_best = 0
        residual = beta
        solve_t0 = time.perf_counter()

        for j in range(max_iter):
            t0 = time.perf_counter()
            w = self.matvec_mps(q[j])
            self._add_timing("gmres_matvec_total", time.perf_counter() - t0)
            t0 = time.perf_counter()
            for i in range(j + 1):
                h[i, j] = q[i].inner(w)
                w = w.add_scaled(q[i], -h[i, j], label=f"ORTH{j}-{i}")
            h[j + 1, j] = w.norm()
            self._add_timing("gmres_orthogonalization", time.perf_counter() - t0)
            if h[j + 1, j] > 1.0e-13 and j + 1 == len(q):
                t0 = time.perf_counter()
                q.append(w.scaled(1.0 / h[j + 1, j], label=f"Q{j + 1}"))
                self._add_timing("gmres_basis_normalization", time.perf_counter() - t0)

            t0 = time.perf_counter()
            e1 = np.zeros(j + 2)
            e1[0] = beta
            y, *_ = np.linalg.lstsq(h[:j + 2, :j + 1], e1, rcond=None)
            residual = float(np.linalg.norm(e1 - h[:j + 2, :j + 1] @ y))
            self._add_timing("gmres_small_lstsq", time.perf_counter() - t0)
            y_best = y
            n_best = j + 1
            if verbose:
                print(f"  MPS-GMRES iter {j + 1}: residual={residual:.3e}")
            if residual <= conv_abs or residual <= conv_rel:
                break
            if h[j + 1, j] <= 1.0e-13:
                break

        assert y_best is not None
        t0 = time.perf_counter()
        correction = self.vector_linear_combination(
            [(float(y_best[i]), q[i]) for i in range(n_best)],
            label="SOL",
        )
        self._add_timing("gmres_final_combination", time.perf_counter() - t0)
        if x0 is None:
            solution = correction
        else:
            t0 = time.perf_counter()
            solution = self.vector_linear_combination(
                [(1.0, x0), (1.0, correction)],
                label="SOL-X0",
            )
            self._add_timing("initial_guess_solution_merge", time.perf_counter() - t0)
        converged = residual <= conv_abs or residual <= conv_rel
        info = 0 if converged else max_iter
        if len(q) >= n_best + 1:
            self._gmres_recycle_cache = {
                "q": list(q[:n_best + 1]),
                "h": np.array(h[:n_best + 1, :n_best], copy=True),
            }
        return solution, info, {
            "residual": residual,
            "relative_residual": residual / beta,
            "niter": n_best,
            "linear_solver": "gmres",
            "initial_guess": self._initial_guess,
            "timings_s": {
                **{k: float(v) for k, v in self._timings.items()},
                "gmres_total": float(time.perf_counter() - solve_t0),
            },
        }

    def bicgstab_mps(
        self,
        rhs: MPSKrylovVector,
        *,
        tol: float = 1.0e-8,
        max_iter: int = 30,
        verbose: bool = False,
    ):
        """Solve ``A x = rhs`` with a short-recurrence MPS BiCGSTAB solver.

        This optional path avoids the growing Arnoldi basis and the
        O(iteration^2) MPS orthogonalization cost of GMRES.  It is intended for
        large active spaces where MPS addition/overlap dominates wall time.
        """
        self._reset_timing()
        x0 = self.initial_guess_vector_mps(rhs)
        bnorm = rhs.norm()
        if bnorm < 1.0e-14:
            sol = self.zero_vector_mps("SOL-ZERO") if x0 is None else x0
            return sol, 0, {
                "residual": 0.0,
                "relative_residual": 0.0,
                "niter": 0,
                "linear_solver": "bicgstab",
                "timings_s": {},
                "initial_guess": self._initial_guess,
            }

        if x0 is None:
            x = self.zero_vector_mps("BICG-X0")
            r = rhs
        else:
            t0 = time.perf_counter()
            ax0 = self.matvec_mps(x0)
            r = rhs.add_scaled(ax0, -1.0, label="BICG-R0")
            x = x0
            self._add_timing("initial_guess_residual_build", time.perf_counter() - t0)

        residual = r.norm()
        conv_abs = float(tol)
        conv_rel = float(tol) * bnorm
        if residual <= conv_abs or residual <= conv_rel:
            return x, 0, {
                "residual": residual,
                "relative_residual": residual / bnorm,
                "niter": 0,
                "linear_solver": "bicgstab",
                "timings_s": {k: float(v) for k, v in self._timings.items()},
                "initial_guess": self._initial_guess,
            }

        r_hat = r.scaled(1.0, label="BICG-RHAT")
        p = None
        v = self.zero_vector_mps("BICG-V0")
        rho_old = 1.0
        alpha = 1.0
        omega = 1.0
        n_done = 0
        solve_t0 = time.perf_counter()

        for it in range(1, max_iter + 1):
            rho_new = r_hat.inner(r)
            if abs(rho_new) < 1.0e-30:
                break
            if it == 1 or p is None:
                p = r
            else:
                beta_coeff = (rho_new / rho_old) * (alpha / omega)
                p = self.vector_linear_combination(
                    [
                        (1.0, r),
                        (beta_coeff, p),
                        (-beta_coeff * omega, v),
                    ],
                    label=f"BICG-P{it}",
                )

            t0 = time.perf_counter()
            v = self.matvec_mps(p)
            self._add_timing("bicgstab_matvec_total", time.perf_counter() - t0)
            denom = r_hat.inner(v)
            if abs(denom) < 1.0e-30:
                break
            alpha = rho_new / denom
            s = r.add_scaled(v, -alpha, label=f"BICG-S{it}")
            s_norm = s.norm()
            if s_norm <= conv_abs or s_norm <= conv_rel:
                x = x.add_scaled(p, alpha, label=f"BICG-X{it}")
                residual = s_norm
                n_done = it
                break

            t0 = time.perf_counter()
            t_vec = self.matvec_mps(s)
            self._add_timing("bicgstab_matvec_total", time.perf_counter() - t0)
            tt = t_vec.inner(t_vec)
            if abs(tt) < 1.0e-30:
                break
            omega = t_vec.inner(s) / tt
            if abs(omega) < 1.0e-30:
                break
            x = self.vector_linear_combination(
                [(1.0, x), (alpha, p), (omega, s)],
                label=f"BICG-X{it}",
            )
            r = s.add_scaled(t_vec, -omega, label=f"BICG-R{it}")
            residual = r.norm()
            n_done = it
            if verbose:
                print(f"  MPS-BiCGSTAB iter {it}: residual={residual:.3e}")
            if residual <= conv_abs or residual <= conv_rel:
                break
            rho_old = rho_new

        converged = residual <= conv_abs or residual <= conv_rel
        info = 0 if converged else max_iter
        return x, info, {
            "residual": residual,
            "relative_residual": residual / bnorm,
            "niter": n_done,
            "linear_solver": "bicgstab",
            "initial_guess": self._initial_guess,
            "timings_s": {
                **{k: float(v) for k, v in self._timings.items()},
                "bicgstab_total": float(time.perf_counter() - solve_t0),
            },
        }

    def cr_mps(
        self,
        rhs: MPSKrylovVector,
        *,
        tol: float = 1.0e-8,
        max_iter: int = 30,
        verbose: bool = False,
    ):
        """Solve ``A x = rhs`` with a short-recurrence conjugate-residual solver.

        The projected SA-CASSCF response operator is symmetric.  CR uses that
        symmetry to minimize the residual without building a full Arnoldi
        basis, which removes the O(iteration**2) MPS overlap cost that dominates
        large active-space GMRES solves.  The solver is conservative: it still
        reports convergence only from the explicit mixed orbital/MPS residual
        norm, so accuracy is controlled by the same ``tol`` criterion as GMRES.
        """
        self._reset_timing()
        x0 = self.initial_guess_vector_mps(rhs)
        bnorm = rhs.norm()
        if bnorm < 1.0e-14:
            sol = self.zero_vector_mps("SOL-ZERO") if x0 is None else x0
            return sol, 0, {
                "residual": 0.0,
                "relative_residual": 0.0,
                "niter": 0,
                "linear_solver": "cr",
                "timings_s": {},
                "initial_guess": self._initial_guess,
            }

        if x0 is None:
            x = self.zero_vector_mps("CR-X0")
            r = rhs
        else:
            t0 = time.perf_counter()
            ax0 = self.matvec_mps(x0)
            r = rhs.add_scaled(ax0, -1.0, label="CR-R0")
            x = x0
            self._add_timing("initial_guess_residual_build", time.perf_counter() - t0)

        residual = r.norm()
        conv_abs = float(tol)
        conv_rel = float(tol) * bnorm
        if residual <= conv_abs or residual <= conv_rel:
            return x, 0, {
                "residual": residual,
                "relative_residual": residual / bnorm,
                "niter": 0,
                "linear_solver": "cr",
                "timings_s": {k: float(v) for k, v in self._timings.items()},
                "initial_guess": self._initial_guess,
            }

        p = r
        t0 = time.perf_counter()
        ap = self.matvec_mps(p)
        self._add_timing("cr_matvec_total", time.perf_counter() - t0)
        ap_ap = ap.inner(ap)
        n_done = 0
        solve_t0 = time.perf_counter()

        for it in range(1, max_iter + 1):
            if abs(ap_ap) < 1.0e-30:
                break
            alpha = r.inner(ap) / ap_ap
            x = x.add_scaled(p, alpha, label=f"CR-X{it}")
            r_next = r.add_scaled(ap, -alpha, label=f"CR-R{it}")
            residual = r_next.norm()
            n_done = it
            if verbose:
                print(f"  MPS-CR iter {it}: residual={residual:.3e}")
            if residual <= conv_abs or residual <= conv_rel:
                r = r_next
                break

            t0 = time.perf_counter()
            ar_next = self.matvec_mps(r_next)
            self._add_timing("cr_matvec_total", time.perf_counter() - t0)
            beta = ar_next.inner(ap) / ap_ap
            p = self.vector_linear_combination(
                [(1.0, r_next), (-beta, p)],
                label=f"CR-P{it}",
            )
            ap = self.vector_linear_combination(
                [(1.0, ar_next), (-beta, ap)],
                label=f"CR-AP{it}",
            )
            ap_ap = ap.inner(ap)
            r = r_next

        converged = residual <= conv_abs or residual <= conv_rel
        info = 0 if converged else max_iter
        return x, info, {
            "residual": residual,
            "relative_residual": residual / bnorm,
            "niter": n_done,
            "linear_solver": "cr",
            "initial_guess": self._initial_guess,
            "timings_s": {
                **{k: float(v) for k, v in self._timings.items()},
                "cr_total": float(time.perf_counter() - solve_t0),
            },
        }

    def solve_linear_mps(
        self,
        rhs: MPSKrylovVector,
        *,
        tol: float = 1.0e-8,
        max_iter: int = 30,
        verbose: bool = False,
    ):
        if self._linear_solver in {"gmres", "arnoldi"}:
            return self.gmres_mps(rhs, tol=tol, max_iter=max_iter, verbose=verbose)
        if self._linear_solver in {"bicgstab", "bi-cgstab", "bcgs"}:
            return self.bicgstab_mps(
                rhs, tol=tol, max_iter=max_iter, verbose=verbose,
            )
        if self._linear_solver in {"cr", "conjugate-residual", "conjugate_residual"}:
            return self.cr_mps(rhs, tol=tol, max_iter=max_iter, verbose=verbose)
        raise ValueError(f"Unknown MPS response linear_solver={self._linear_solver!r}")

    def solve_mps(self, state: int, tol: float = 1.0e-8,
                  max_iter: int = 30, verbose: bool = False):
        """Solve a state-gradient response equation with MPS Krylov vectors.

        The RHS and the GMRES CI Krylov vectors are represented as MPS objects.
        """
        rhs = self.build_rhs_mps(state)
        rhs, rhs_proj_meta = self.project_vector_ci_mps(
            rhs, label=f"RHS-G{state}-GAUGE", mode="all",
        )
        sol, info, meta = self.solve_linear_mps(
            rhs, tol=tol, max_iter=max_iter, verbose=verbose,
        )
        ci_mps, proj_meta = self.project_ci_mps_list(
            sol.ci_mps, label=f"SOL-G{state}-PROJ", mode="all",
        )
        meta = {
            **dict(meta),
            "rhs_max_root_overlap_before_projection": rhs_proj_meta[
                "max_root_overlap_before_projection"
            ],
            "rhs_max_root_overlap_after_projection": rhs_proj_meta[
                "max_root_overlap_after_projection"
            ],
            **proj_meta,
        }
        return sol.kappa, ci_mps, info, meta

    def solve_nac_mps(self, state_pair: tuple[int, int], tol: float = 1.0e-8,
                      max_iter: int = 30, verbose: bool = False):
        """Solve a NAC response equation with MPS Krylov vectors."""
        rhs = self.build_rhs_nac_mps(state_pair)
        rhs, rhs_proj_meta = self.project_vector_ci_mps(
            rhs,
            label=f"RHS-NAC{int(state_pair[0])}-{int(state_pair[1])}-GAUGE",
            mode="all",
        )
        sol, info, meta = self.solve_linear_mps(
            rhs, tol=tol, max_iter=max_iter, verbose=verbose,
        )
        ci_mps, proj_meta = self.project_ci_mps_list(
            sol.ci_mps,
            label=f"SOL-NAC{int(state_pair[0])}-{int(state_pair[1])}-PROJ",
            mode="all",
        )
        meta = {
            **dict(meta),
            "rhs_max_root_overlap_before_projection": rhs_proj_meta[
                "max_root_overlap_before_projection"
            ],
            "rhs_max_root_overlap_after_projection": rhs_proj_meta[
                "max_root_overlap_after_projection"
            ],
            **proj_meta,
        }
        return sol.kappa, ci_mps, info, meta
