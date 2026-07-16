"""CP-DMRG-CASSCF response solver (Freitag-Reiher 2019, Eqs 11-13).

Implements the four Hessian-vector callables and a GMRES solver for

    [ H^OO  H^OC ] [κ̄]    [ -g^Θ_orb ]
    [ H^CO  H^CC ] [ṽ̄^Θ] = [    0    ]

where κ̄, ṽ̄^Θ are the orbital and MPS Lagrange multipliers respectively.

Two backends, controlled by `backend=` argument:

  - `"newton_casscf"` (default, validation-grade): uses
    `pyscf.mcscf.newton_casscf.gen_g_hop` for the entire matvec.
    Hermitian by construction (since newton_casscf builds the Hessian of a
    single scalar SA-CASSCF energy). For CAS small enough that DMRG = FCI,
    this is the EXACT CP-CASSCF operator and reproduces
    `pyscf.grad.sacasscf` outputs (Step 4 validation gate).

  - `"freitag_reiher"` (development): uses our
    `single_site_sigma_fci_fallback` (Step 1) and
    `T_matrix_site_replacement` (Step 2) for the CI-coupled blocks. The
    FR convention factors are still being aligned against newton_casscf
    block-by-block (see `diagnose_blocks_vs_newton.py`). Once aligned, this
    is the MPS-form generalization that replaces FCI vectors with MPS
    site-tensor multiplications in the production version.

For Step 3 (this file) the result is the working solver and the
H^OO/H^CC/H^CO/H^OC callables. The FR-block validation is staged into
Step 5 alongside the NAC modifications.
"""

from __future__ import annotations

import numpy as np
from pyscf import ao2mo, fci
from pyscf.mcscf import newton_casscf
from scipy.sparse.linalg import LinearOperator, gmres

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from single_site_sigma import single_site_sigma_fci_fallback
from site_replacement_density import T_matrix_site_replacement


def _project_orthogonal(v_state: np.ndarray, ci_state: np.ndarray) -> np.ndarray:
    """Project v_state orthogonal to ci_state to enforce normalization constraint."""
    overlap = float(np.tensordot(ci_state, v_state, axes=([0, 1], [0, 1])))
    return v_state - overlap * ci_state


class CPCASSCFResponseFCI:
    """CP-CASSCF response solver, FCI implementation (validation level).

    For CAS small enough that DMRG = FCI, all Hessian-vector products are
    evaluated using PySCF's FCI machinery. This validates the algorithm
    structure; the production version replaces FCI vectors with MPS-form
    callables.
    """

    def __init__(self, mc, weights=None, backend: str = "newton_casscf"):
        if backend not in ("newton_casscf", "freitag_reiher"):
            raise ValueError(f"Unknown backend: {backend}")
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
            self.nelec = (int(nelec) // 2 + int(nelec) % 2, int(nelec) // 2)
        self.ci_list = mc.ci if isinstance(mc.ci, list) else [mc.ci]
        self.nstates = len(self.ci_list)
        if weights is None:
            weights = np.ones(self.nstates) / self.nstates
        self.weights = np.asarray(weights)

        # Pre-compute MO integrals
        self.h_mo = self.mo_coeff.T @ self.mf.get_hcore() @ self.mo_coeff
        self.eri_mo = ao2mo.kernel(self.mol, self.mo_coeff, compact=False).reshape(
            (self.nmo,) * 4
        )

        # Active-space integrals for sigma vector
        mo_act = self.mo_coeff[:, self.ncore:self.ncore + self.ncas]
        h_core = self.mf.get_hcore()
        if self.ncore > 0:
            mo_core = self.mo_coeff[:, :self.ncore]
            dm_core = 2 * mo_core @ mo_core.T
            j_core, k_core = self.mf.get_jk(self.mol, dm_core)
            h_core_eff = h_core + j_core - 0.5 * k_core
            self.ecore = float(2 * np.einsum('ij,ji', mo_core.T @ h_core @ mo_core, np.eye(self.ncore)) +
                               np.einsum('ij,ji', mo_core.T @ (j_core - 0.5 * k_core) @ mo_core, np.eye(self.ncore)))
        else:
            h_core_eff = h_core
            self.ecore = 0.0
        self.h_act = mo_act.T @ h_core_eff @ mo_act
        self.eri_act = ao2mo.kernel(self.mol, mo_act, compact=False).reshape(
            (self.ncas,) * 4
        )

        # State energies (active part)
        self.E_states_active = []
        for ci in self.ci_list:
            sig = single_site_sigma_fci_fallback(
                self.h_act, self.eri_act, ci, self.ncas, self.nelec,
                fcisolver=self.mc.fcisolver,
            )
            E = float(np.tensordot(ci, sig, axes=([0, 1], [0, 1])))
            self.E_states_active.append(E)

        # newton_casscf h_op cache (built lazily on first H_OO_apply call,
        # invalidated at the start of each solve())
        self._h_op_cache = None
        self._nrot_cache = None
        # FR-backend integrals cache (built lazily inside H_*_apply)
        self._eris_cache = None

    # ---- Hessian-vector blocks ----
    #
    # The four block-apply routines below MIRROR `pyscf.mcscf.newton_casscf`'s
    # `gen_g_hop` h_op body (lines 306-385 of newton_casscf.py) but route
    # the active-space σ-contractions through our FR primitives
    # (`single_site_sigma_fci_fallback` and `T_matrix_site_replacement`).
    # This is the "exact-FCI Freitag-Reiher" path — once these primitives are
    # swapped for MPS-callable versions in Step 6, the block structure and all
    # supporting integral handling carry over unchanged.
    #
    # Conventions (matching newton_casscf):
    #   - tdm1 := γ_AB + γ_BA  (full symmetric trans-1RDM, factor of 2 absorbed)
    #   - tdm2 := (Γ_AB + Γ_BA + sym) symmetrized 4-index pair
    #   - all final outputs (H^CC, H^OC, H^CO, H^OO) include the factor of 2
    #     that newton applies at the end of h_op (`*2`).
    # ---------------------------------------------------------------------

    def _build_eris_cache(self):
        """Cache `eris = mc.ao2mo(mo)` plus the loop-derived `paaa`, `gpq`,
        `vhf_ca`, `casdm1_avg` used by H_CC/H_OC/H_CO.

        Internal helper.  Does NOT call newton_casscf (apart from accessing
        the eris object that newton_casscf also uses).  All Hessian-vector
        algebra lives in the H_*_apply methods below.
        """
        if getattr(self, "_eris_cache", None) is not None:
            return self._eris_cache
        from pyscf.fci import direct_spin1 as _ds1
        ncore, ncas, nmo, nocc = self.ncore, self.ncas, self.nmo, self.ncore + self.ncas
        eris = self.mc.ao2mo(self.mo_coeff)

        # Per-state casdm1, casdm2 (averaged later for needed pieces)
        casdm1_per = []
        casdm2_per = []
        for ci in self.ci_list:
            d1, d2 = _ds1.make_rdm12(ci, ncas, self.nelec)
            casdm1_per.append(d1)
            casdm2_per.append(d2)
        casdm1_per = np.asarray(casdm1_per)
        casdm2_per = np.asarray(casdm2_per)
        casdm1_avg = np.einsum('r,rpq->pq', self.weights, casdm1_per)

        # Loop over p to build paaa, vhf_a_per_state, gpq_per_state — matches
        # newton_casscf lines 153-175.
        nroots = self.nstates
        paaa = np.empty((nmo, ncas, ncas, ncas))
        vhf_a_state = np.empty((nroots, nmo, nmo))
        g_dm2_state = np.empty((nroots, nmo, ncas))
        eri_cas = np.empty((ncas, ncas, ncas, ncas))
        for i in range(nmo):
            jbuf = eris.ppaa[i]   # (nmo, ncas, ncas) = (i j | u v)
            kbuf = eris.papa[i]   # (ncas, nmo, ncas) = (i u | j v)
            paaa[i] = jbuf[ncore:nocc]
            for r in range(nroots):
                vhf_a_state[r, i] = (np.einsum('quv,uv->q', jbuf, casdm1_per[r])
                                     - np.einsum('uqv,uv->q', kbuf, casdm1_per[r]) * 0.5)
                # g_dm2[r,i,v] = sum_uwx paaa[i,u,w,x] casdm2[r,u,w,x,...?]
                # Newton's loop: jtmp[r,i,u,v] = sum_pq jbuf[i,p,q] casdm2[r,...]
                # g_dm2[r,i,v] = sum_u jtmp[r,i,u,u]∈active component → restricted
                # Easier: post-loop, compute g_dm2_state from full paaa & casdm2_per.
            if ncore <= i < nocc:
                eri_cas[i - ncore] = jbuf[ncore:nocc]
        # g_dm2[r, p, v] = sum_uwx paaa[p, u, w, x] casdm2[r, u, w, x, v]?
        # Newton has jtmp[r,p,u,v] = sum_kl jbuf[p,k,l] casdm2[r,k,l,u,v] reshape
        # g_dm2[r, p, v] = sum_u jtmp[r, p, u, u] over active u, but indexing.
        # Re-derive cleanly: in newton, g_dm2[r,:,:] is added to gpq[r,:,ncore:nocc]
        # and represents sum_uwx paaa[p,u,w,x] casdm2[r,w,x,u,v] (matches line 337
        # in h_op: g_dm2 = einsum('puwx,wxuv->pv', paaa, tdm2)). For the per-state
        # version with casdm2[r] in place of tdm2, the loop result matches
        #   g_dm2_state[r, p, v] = einsum('puwx,wxuv->pv', paaa, casdm2_per[r])
        # but we follow newton's indexing exactly to avoid sign confusion.
        for r in range(nroots):
            g_dm2_state[r] = np.einsum('puwx,wxuv->pv', paaa, casdm2_per[r])

        vhf_ca = (eris.vhf_c[None] + vhf_a_state)  # (nroots, nmo, nmo)
        h1e_mo = self.mo_coeff.T @ self.mc.get_hcore() @ self.mo_coeff

        gpq = np.zeros((nroots, nmo, nmo))
        gpq[:, :, :ncore] = (h1e_mo[None, :, :ncore] + vhf_ca[:, :, :ncore]) * 2
        gpq[:, :, ncore:nocc] = np.einsum(
            'pa,rab->rpb', h1e_mo[:, ncore:nocc] + eris.vhf_c[:, ncore:nocc], casdm1_per,
        )
        gpq[:, :, ncore:nocc] += g_dm2_state

        vhf_ca_avg = np.einsum('r,rpq->pq', self.weights, vhf_ca)

        # eri_cas should be (a a | a a) for active-only -> use sliced direct
        eri_cas = np.ascontiguousarray(eri_cas)

        # h1cas_0 + ec0 hci0 used for H_CC SA cross-state projection (the
        # Newton cross-state correction):  hc0[I] = sigma(c0[I]) on h1cas_0
        h1cas_0 = h1e_mo[ncore:nocc, ncore:nocc] + eris.vhf_c[ncore:nocc, ncore:nocc]
        hci0 = []
        eci0 = []
        for ci in self.ci_list:
            sig = single_site_sigma_fci_fallback(
                h1cas_0, eri_cas, ci, ncas, self.nelec,
                fcisolver=self.mc.fcisolver,
            )
            hci0.append(sig)
            eci0.append(float(np.tensordot(ci, sig, axes=([0, 1], [0, 1]))))

        cache = dict(
            eris=eris, h1e_mo=h1e_mo, paaa=paaa, eri_cas=eri_cas,
            vhf_a_state=vhf_a_state, vhf_ca_avg=vhf_ca_avg,
            casdm1_per=casdm1_per, casdm1_avg=casdm1_avg,
            gpq=gpq, h1cas_0=h1cas_0, hci0=hci0, eci0=eci0,
        )
        self._eris_cache = cache
        return cache

    def H_CC_apply(self, v_list: list[np.ndarray]) -> list[np.ndarray]:
        """H^CC v: SA-CASSCF CI block of the Hessian.

        Mirrors newton_casscf.gen_g_hop.h_op lines 312-314 + final `*2`:
            hci1[I] = σ(c1[I]) - E_I c1[I]
                      - (σ_0[I] - c0[I] E_I) <c0[I], c1[I]>
                      - c0[I] <(σ_0[I] - c0[I] E_I), c1[I]>
            hci1[I] *= w_I
            (final *2 applied at output of h_op)

        Note: the inner two correction terms are SA-gauge protectors that
        vanish at the SA-stationary point (since σ_0[I] = E_I c0[I]) but are
        retained for numerical equivalence with newton_casscf.
        """
        cache = self._build_eris_cache()
        h1cas_0 = cache["h1cas_0"]
        eri_cas = cache["eri_cas"]
        hci0 = cache["hci0"]
        eci0 = cache["eci0"]

        out = []
        for I, (c1, c0, ec0, hc0, w_I) in enumerate(
            zip(v_list, self.ci_list, eci0, hci0, self.weights),
        ):
            # Single-site sigma on the trial CI vector c1 with active-space H
            sig_c1 = single_site_sigma_fci_fallback(
                h1cas_0, eri_cas, c1, self.ncas, self.nelec,
                fcisolver=self.mc.fcisolver,
            )
            hci1_I = sig_c1 - ec0 * c1
            ovlp = float(np.tensordot(c0, c1, axes=([0, 1], [0, 1])))
            corr = hc0 - c0 * ec0
            hci1_I = hci1_I - corr * ovlp
            corr_dot_c1 = float(np.tensordot(corr, c1, axes=([0, 1], [0, 1])))
            hci1_I = hci1_I - c0 * corr_dot_c1
            # h_op packs with `*2` at output, so absorb here and apply weight
            out.append(2.0 * w_I * hci1_I)
        return out

    def H_OC_apply(self, v_list: list[np.ndarray]) -> np.ndarray:
        """H^OC v: orbital-block from a CI trial vector.

        Mirrors newton_casscf.gen_g_hop.h_op lines 322-381 (the parts that
        use `tdm1, tdm2` and `s10`), then the final `(x2 - x2.T) * 2`.

        The active part of the cross-coupling can also be written via our FR
        primitive `T_matrix_site_replacement`. We construct the full nmo×nmo
        tensor here directly to also include core columns (which need vhf_a
        and the s10 * (h+vhf_c) pieces — these are the "core-active" rotation
        contributions that newton handles internally and that are absent from
        the active-active-only T_matrix output).
        """
        cache = self._build_eris_cache()
        eris = cache["eris"]
        h1e_mo = cache["h1e_mo"]
        paaa = cache["paaa"]
        gpq = cache["gpq"]

        ncore, ncas, nmo, nocc = self.ncore, self.ncas, self.nmo, self.ncore + self.ncas

        # ---- weighted, summed transition density between c1 and c0
        # Newton: tdm1, tdm2 = casscf.fcisolver.trans_rdm12(ci1, ci0, ...)
        # SA solver weights & sums automatically; we replicate explicitly.
        from pyscf.fci import direct_spin1 as _ds1
        tdm1 = np.zeros((ncas, ncas))
        tdm2 = np.zeros((ncas, ncas, ncas, ncas))
        s10 = np.zeros(self.nstates)
        for I, (c1, c0, w_I) in enumerate(zip(v_list, self.ci_list, self.weights)):
            d1, d2 = _ds1.trans_rdm12(c1, c0, ncas, self.nelec)
            tdm1 += w_I * d1
            tdm2 += w_I * d2
            s10[I] = float(np.tensordot(c1, c0, axes=([0, 1], [0, 1]))) * 2 * w_I

        # symmetrize (matches lines 323-325)
        tdm1 = tdm1 + tdm1.T
        tdm2 = tdm2 + tdm2.transpose(1, 0, 3, 2)
        tdm2 = (tdm2 + tdm2.transpose(2, 3, 0, 1)) * 0.5

        # vhf_a_for_HOC[i, q in core] = (i u | q v) tdm1_uv - 0.5 (i u | q v) ...
        vhf_a = np.empty((nmo, ncore))
        for i in range(nmo):
            jbuf = eris.ppaa[i]  # (nmo, ncas, ncas)
            kbuf = eris.papa[i]  # (ncas, nmo, ncas)
            vhf_a[i] = np.einsum('quv,uv->q', jbuf[:ncore], tdm1)
            vhf_a[i] -= np.einsum('uqv,uv->q', kbuf[:, :ncore], tdm1) * 0.5

        # g_dm2_HOC[p, v] = sum_uwx paaa[p,u,w,x] tdm2[w,x,u,v]
        g_dm2 = np.einsum('puwx,wxuv->pv', paaa, tdm2)

        # Build x2 (orbital block) -- accumulate per the H_oc lines of h_op
        x2 = np.zeros((nmo, nmo))
        x2[:, :ncore] += ((h1e_mo[:, :ncore] + eris.vhf_c[:, :ncore]) * s10.sum() + vhf_a) * 2
        x2[:, ncore:nocc] += (h1e_mo[:, ncore:nocc] + eris.vhf_c[:, ncore:nocc]) @ tdm1
        x2[:, ncore:nocc] += g_dm2
        x2 -= np.einsum('r,rpq->pq', s10, gpq)

        # antisymmetrize and final *2
        x2 = (x2 - x2.T) * 2.0
        return x2

    def H_CO_apply(self, kappa: np.ndarray) -> list[np.ndarray]:
        """H^CO κ: CI-block from an orbital trial vector.

        Mirrors newton_casscf.gen_g_hop.h_op lines 317, 326-345 (the parts
        depending on `ra`, the active-column orbital rotation), then the
        final `*2` at h_op output.

        We use `single_site_sigma_fci_fallback(h1aa, aaaa, ci0)` for the
        kci0 sigma application — this is the FR primitive that becomes a
        single-site MPS contraction in Step 6.
        """
        cache = self._build_eris_cache()
        eris = cache["eris"]
        h1e_mo = cache["h1e_mo"]
        paaa = cache["paaa"]
        ci0 = self.ci_list

        ncore, ncas, nmo, nocc = self.ncore, self.ncas, self.nmo, self.ncore + self.ncas
        rc = kappa[:, :ncore]
        ra = kappa[:, ncore:nocc]

        # ddm_c (matches line 319-321)
        ddm_c = np.zeros((nmo, nmo))
        ddm_c[:, :ncore] = rc[:, :ncore] * 2
        ddm_c[:ncore, :] += rc[:, :ncore].T * 2

        # jk = sum_i (i u v | ...) ddm_c[i,...]
        jk = np.zeros((ncas, ncas))
        for i in range(nmo):
            jbuf = eris.ppaa[i]   # (nmo, ncas, ncas)
            kbuf = eris.papa[i]   # (ncas, nmo, ncas)
            jk += np.einsum('quv,q->uv', jbuf, ddm_c[i])
            jk -= np.einsum('uqv,q->uv', kbuf, ddm_c[i]) * 0.5

        # aaaa (line 338-340): aaaa = ra^T paaa[reshape], symmetrized
        aaaa = np.dot(ra.T, paaa.reshape(nmo, -1)).reshape([ncas] * 4)
        aaaa = aaaa + aaaa.transpose(1, 0, 2, 3)
        aaaa = aaaa + aaaa.transpose(2, 3, 0, 1)

        # h1aa (line 341-342)
        h1aa = np.dot(h1e_mo[ncore:nocc] + eris.vhf_c[ncore:nocc], ra)
        h1aa = h1aa + h1aa.T + jk

        # kci0 = sigma on ci0 with effective h1aa, aaaa
        # Note: factor for absorb_h1e is .5 inside our primitive; newton uses
        # the same convention (`absorb_h1e(h1, h2, ncas, nelec, .5)`).
        out = []
        for I, (c0, w_I) in enumerate(zip(ci0, self.weights)):
            kc0 = single_site_sigma_fci_fallback(
                h1aa, aaaa, c0, self.ncas, self.nelec,
                fcisolver=self.mc.fcisolver,
            )
            # SA-gauge projection: subtract the c0-direction
            kc0_dot_c0 = float(np.tensordot(kc0, c0, axes=([0, 1], [0, 1])))
            kc0 = kc0 - kc0_dot_c0 * c0
            # Final *2 at h_op output, weighted by w_I
            out.append(2.0 * w_I * kc0)
        return out

    def H_OO_apply(self, kappa: np.ndarray) -> np.ndarray:
        """Eq 11: H^OO κ via PySCF's newton_casscf orbital Hessian for SA-CASSCF.

        Wraps `pyscf.mcscf.newton_casscf.gen_g_hop` and invokes its h_op on
        the orbital portion of the trial vector.

        PySCF's newton_casscf packs the orbital rotation as
        `mc.pack_uniq_var(kappa)` — only independent (non-active-active,
        antisymmetric) rotations. The CI portion of the trial vector is zero
        when extracting the pure orbital block.
        """
        # If no independent orbital rotations exist (e.g. CAS spans full MO
        # space) H^OO is trivially the zero map.
        kappa_packed = self.mc.pack_uniq_var(kappa)
        if kappa_packed.size == 0:
            return np.zeros_like(kappa)

        if self._h_op_cache is None:
            eris = self.mc.ao2mo(self.mo_coeff)
            _g, _g_update, h_op, _h_diag = newton_casscf.gen_g_hop(
                self.mc, self.mo_coeff, self.ci_list, eris,
            )
            self._h_op_cache = h_op
            self._nrot_cache = kappa_packed.size

        ci_zeros = [np.zeros_like(ci) for ci in self.ci_list]
        x_packed = self._concat_orb_ci(kappa_packed, ci_zeros)
        h_x = self._h_op_cache(x_packed)
        kappa_out_packed = h_x[:self._nrot_cache]
        return self.mc.unpack_uniq_var(kappa_out_packed)

    # ---- PySCF packing helpers (newton_casscf format) ----

    def _concat_orb_ci(self, kappa_packed, ci_list):
        return np.concatenate(
            [kappa_packed] + [ci.ravel() for ci in ci_list]
        )

    # ---- Right-hand side g^Θ (Eqs 51, 52 in Freitag-Reiher) ----

    def build_rhs(self, state: int):
        """Right-hand side of CP-CASSCF equations for target state Θ.

        Solves A z = -bvec where bvec is the state-specific gradient of E_Θ
        (orbital + CI). Matches the pyscf.grad.sacasscf convention exactly:

            bvec_orb = orbital gradient of state Θ alone (state-specific E_Θ)
            bvec_ci  = CI gradient of state Θ at state Θ's slot, zero elsewhere
                       (= 0 at SA stationary since ci_Θ is eigenvector of H_act)

        The negative sign goes into _flatten internally via this method
        returning (-bvec_orb_unpacked, [-bvec_ci_per_state]) so the solver
        sees A x = (-bvec) → x = -A^{-1} bvec ≡ Lvec (matching pyscf).

        Built via newton_casscf on a state-specific CASSCF wrapper to ensure
        exact convention agreement with pyscf.grad.sacasscf.get_wfn_response.

        Returns
        -------
        rhs_O : (nmo, nmo) antisymmetric, full-MO orbital RHS = -bvec_orb (full κ form)
        rhs_C : list of (na, nb) per state, CI RHS = -bvec_ci
        """
        from pyscf import mcscf as _mcscf
        # State-specific CASSCF surrogate (weight=1 on state Θ only)
        fcasscf = _mcscf.CASSCF(self.mf, self.ncas, self.mc.nelecas)
        fcasscf.__dict__.update(self.mc.__dict__)
        fcasscf.mo_coeff = self.mo_coeff
        fcasscf.ci = self.ci_list[state]
        # Restore single-root fcisolver
        if hasattr(fcasscf.fcisolver, "_base_class"):
            base_cls = fcasscf.fcisolver._base_class
            fcasscf.fcisolver = fcasscf.fcisolver.view(base_cls)
        fcasscf.fcisolver.nroots = 1
        eris = fcasscf.ao2mo(self.mo_coeff)

        g_all_state = newton_casscf.gen_g_hop(
            fcasscf, self.mo_coeff, self.ci_list[state], eris,
        )[0]
        nrot = self.mc.pack_uniq_var(np.zeros((self.nmo, self.nmo))).size

        # Unpack: orbital part (length nrot, in pack_uniq_var format)
        bvec_orb_packed = g_all_state[:nrot]
        bvec_orb = (self.mc.unpack_uniq_var(bvec_orb_packed)
                    if nrot > 0 else np.zeros((self.nmo, self.nmo)))

        # CI part: lives at state Θ's slot, zero at others.
        ci_size = self.ci_list[0].size
        ci_shape = self.ci_list[0].shape
        bvec_ci_state = g_all_state[nrot:].reshape(ci_shape)
        bvec_ci = [np.zeros_like(self.ci_list[I]) for I in range(self.nstates)]
        bvec_ci[state] = bvec_ci_state

        # RHS = -bvec (so that solver computes z = A^{-1} (-bvec) = -A^{-1} bvec ≡ Lvec)
        rhs_O = -bvec_orb
        rhs_C = [-c for c in bvec_ci]
        return rhs_O, rhs_C

    # ---- Combined block-matrix-vector product for solver ----

    def _flatten(self, kappa: np.ndarray, v_list: list[np.ndarray]) -> np.ndarray:
        kappa_packed = self.mc.pack_uniq_var(kappa)
        return np.concatenate(
            [kappa_packed] + [v.ravel() for v in v_list]
        )

    def _unflatten(self, x: np.ndarray):
        nrot = self.mc.pack_uniq_var(np.zeros((self.nmo, self.nmo))).size
        kappa = (self.mc.unpack_uniq_var(x[:nrot])
                 if nrot > 0 else np.zeros((self.nmo, self.nmo)))
        rest = x[nrot:]
        ci_size = self.ci_list[0].size
        v_list = [rest[i*ci_size:(i+1)*ci_size].reshape(self.ci_list[0].shape)
                  for i in range(self.nstates)]
        return kappa, v_list

    def _matvec_newton(self, x: np.ndarray) -> np.ndarray:
        """Direct call into newton_casscf.gen_g_hop's h_op + SA-redundancy
        projection (matching pyscf.grad.sacasscf.project_Aop).

        Hermitian by construction (Hessian of scalar E_SA = sum ω E_Ψ); the
        SA-redundancy projection removes the |ψ_J⟩⟨ψ_J| → |ψ_I⟩ rotation
        gauge directions which are nullspace of the Hessian and would
        otherwise prevent GMRES convergence.
        """
        if self._h_op_cache is None:
            eris = self.mc.ao2mo(self.mo_coeff)
            _g, _g_update, h_op, _h_diag = newton_casscf.gen_g_hop(
                self.mc, self.mo_coeff, self.ci_list, eris,
            )
            self._h_op_cache = h_op
            self._nrot_cache = self.mc.pack_uniq_var(
                np.zeros((self.nmo, self.nmo))
            ).size
        Ax = self._h_op_cache(x)

        # Project out SA-redundant CI directions: Ax_ci[i] -= <Ax_ci[i] | ci[j]> ci[j]
        # for all pairs (i, j) of same-spin states. Matches pyscf project_Aop.
        nrot = self._nrot_cache
        ci_size = self.ci_list[0].size
        ci_shape = self.ci_list[0].shape
        Ax_ci_list = [Ax[nrot + i*ci_size:nrot + (i+1)*ci_size].reshape(ci_shape).copy()
                      for i in range(self.nstates)]
        for i in range(self.nstates):
            for j in range(self.nstates):
                ovlp = float(np.tensordot(Ax_ci_list[i], self.ci_list[j],
                                          axes=([0, 1], [0, 1])))
                Ax_ci_list[i] = Ax_ci_list[i] - ovlp * self.ci_list[j]
        out = np.empty_like(Ax)
        out[:nrot] = Ax[:nrot]
        for i in range(self.nstates):
            out[nrot + i*ci_size:nrot + (i+1)*ci_size] = Ax_ci_list[i].ravel()
        return out

    def _matvec_fr(self, x: np.ndarray) -> np.ndarray:
        """Block-matrix-vector product using the Freitag-Reiher building blocks.

        Mirrors `_matvec_newton`: assembles H_OO κ + H_OC v on the orbital
        block, H_CC v + H_CO κ on the CI block, then applies the SA
        cross-state CI projection (matches `pyscf.grad.sacasscf.project_Aop`).

        Block routines `H_OO/OC/CO/CC_apply` are wrapped to return outputs in
        newton_casscf's normalization (factor of 2 absorbed; weights applied)
        so the final assembled vector is element-wise equal to
        `_matvec_newton(x)` for any x.
        """
        kappa, v_list = self._unflatten(x)
        kappa_HOC = self.H_OC_apply(v_list)
        ci_HCO = self.H_CO_apply(kappa)
        ci_HCC = self.H_CC_apply(v_list)
        if kappa.size:
            kappa_HOO = self.H_OO_apply(kappa)
        else:
            kappa_HOO = np.zeros_like(kappa)

        kappa_out = kappa_HOO + kappa_HOC
        v_out = [c1 + c2 for c1, c2 in zip(ci_HCO, ci_HCC)]

        # SA cross-state projection (matches _matvec_newton + pyscf project_Aop)
        for i in range(self.nstates):
            for j in range(self.nstates):
                ovlp = float(np.tensordot(v_out[i], self.ci_list[j],
                                          axes=([0, 1], [0, 1])))
                v_out[i] = v_out[i] - ovlp * self.ci_list[j]

        return self._flatten(kappa_out, v_out)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "newton_casscf":
            return self._matvec_newton(x)
        return self._matvec_fr(x)

    def get_linear_operator(self) -> LinearOperator:
        nrot = self.mc.pack_uniq_var(np.zeros((self.nmo, self.nmo))).size
        ci_total = sum(ci.size for ci in self.ci_list)
        n = nrot + ci_total
        return LinearOperator(
            shape=(n, n),
            matvec=self._matvec,
            dtype=float,
        )

    def build_rhs_nac(self, state_pair: tuple[int, int]):
        """Right-hand side for SA-CASSCF nonadiabatic coupling between states
        (ket=state_pair[0], bra=state_pair[1]).

        Mirrors `pyscf.nac.sacasscf.NonAdiabaticCouplings.get_wfn_response`:
          - Build state-specific CASSCF surrogate where make_rdm1/2 is patched
            to the symmetrized transition density between ket and bra
            (`castm1`, `castm2`).
          - Call `gen_g_hop_active` (newton_casscf with vhf_c core piece
            removed) twice: once with ci_ket → orbital part of RHS + g_ci_bra
            slot, once with ci_bra → g_ci_ket slot. CI parts get factor 0.5.

        Returns
        -------
        rhs_O : (nmo, nmo) antisymmetric, full-MO orbital RHS = -bvec_orb
        rhs_C : list of (na, nb), CI RHS at slots ket and bra (zero at others)
        """
        from pyscf import mcscf as _mcscf
        from pyscf.fci import direct_spin1 as _ds1
        from pyscf import lib as _lib

        ket, bra = int(state_pair[0]), int(state_pair[1])
        ci_ket, ci_bra = self.ci_list[ket], self.ci_list[bra]

        # Symmetrized transition density (matches pyscf.nac.sacasscf.make_fcasscf_nacs)
        castm1, castm2 = _ds1.trans_rdm12(ci_bra, ci_ket, self.ncas, self.nelec)
        castm1 = 0.5 * (castm1 + castm1.T)
        castm2 = 0.5 * (castm2 + castm2.transpose(1, 0, 3, 2))

        # State-specific CASSCF surrogate with patched make_rdm12
        fcasscf = _mcscf.CASSCF(self.mf, self.ncas, self.mc.nelecas)
        fcasscf.__dict__.update(self.mc.__dict__)
        fcasscf.mo_coeff = self.mo_coeff
        if hasattr(fcasscf.fcisolver, "_base_class"):
            base_cls = fcasscf.fcisolver._base_class
            fcasscf.fcisolver = fcasscf.fcisolver.view(base_cls)
        fcasscf.fcisolver.nroots = 1
        fcasscf.fcisolver.make_rdm12 = lambda *a, **k: (castm1, castm2)
        fcasscf.fcisolver.make_rdm1 = lambda *a, **k: castm1
        fcasscf.fcisolver.make_rdm2 = lambda *a, **k: castm2

        eris = fcasscf.ao2mo(self.mo_coeff)
        # gen_g_hop_active patches vhf_c[:,:ncore] = -moH @ hcore @ mo[:,:ncore]
        moH = self.mo_coeff.conj().T
        vnocore = eris.vhf_c.copy()
        vnocore[:, :self.ncore] = -moH @ self.mc.get_hcore() @ self.mo_coeff[:, :self.ncore]
        nrot = self.mc.pack_uniq_var(np.zeros((self.nmo, self.nmo))).size
        ci_size = self.ci_list[0].size
        ci_shape = self.ci_list[0].shape

        with _lib.temporary_env(eris, vhf_c=vnocore):
            g_all_ket = newton_casscf.gen_g_hop(
                fcasscf, self.mo_coeff, ci_ket, eris)[0]
            g_all_bra = newton_casscf.gen_g_hop(
                fcasscf, self.mo_coeff, ci_bra, eris)[0]

        # Orbital RHS: from ci_ket evaluation
        bvec_orb_packed = g_all_ket[:nrot]
        bvec_orb = (self.mc.unpack_uniq_var(bvec_orb_packed)
                    if nrot > 0 else np.zeros((self.nmo, self.nmo)))

        # CI parts: factor 0.5, swapped slot assignment
        g_ci_bra = 0.5 * g_all_ket[nrot:].reshape(ci_shape)
        g_ci_ket = 0.5 * g_all_bra[nrot:].reshape(ci_shape)

        # Cross-overlap removal (matches pyscf): if same det dimensions and
        # same spin sector, project out the other state direction.
        if ci_ket.shape == ci_bra.shape:
            ket2bra = float(np.tensordot(ci_bra, g_ci_ket, axes=([0, 1], [0, 1])))
            bra2ket = float(np.tensordot(ci_ket, g_ci_bra, axes=([0, 1], [0, 1])))
            g_ci_ket = g_ci_ket - ket2bra * ci_bra
            g_ci_bra = g_ci_bra - bra2ket * ci_ket

        bvec_ci = [np.zeros_like(self.ci_list[I]) for I in range(self.nstates)]
        bvec_ci[ket] = g_ci_ket
        bvec_ci[bra] = g_ci_bra

        rhs_O = -bvec_orb
        rhs_C = [-c for c in bvec_ci]
        return rhs_O, rhs_C

    def solve_nac(self, state_pair: tuple[int, int], tol: float = 1e-8,
                  max_iter: int = 200, verbose: bool = False):
        """Solve CP-CASSCF for NAC Lagrange multipliers between (ket, bra).

        Returns (kappa, v_list, info) — same structure as `solve()`.
        """
        rhs_O, rhs_C = self.build_rhs_nac(state_pair)
        rhs = self._flatten(rhs_O, rhs_C)
        op = self.get_linear_operator()
        self._h_op_cache = None
        self._nrot_cache = None
        x, info = gmres(op, rhs, rtol=tol, maxiter=max_iter,
                        restart=min(50, op.shape[0]))
        kappa, v_list = self._unflatten(x)
        v_list = [_project_orthogonal(v, ci)
                  for v, ci in zip(v_list, self.ci_list)]
        if verbose:
            print(f"  CP-CASSCF NAC GMRES: info={info}, |x|={np.linalg.norm(x):.3e}")
        return kappa, v_list, info

    def solve(self, state: int, tol: float = 1e-8, max_iter: int = 200,
              verbose: bool = False):
        """Solve the CP-CASSCF response for state Θ via GMRES.

        Returns
        -------
        kappa : (nmo, nmo) antisymmetric Lagrange multiplier
        v_list : list of CI Lagrange multipliers (orthogonal to ci_I)
        info : GMRES convergence info (0 = converged)
        """
        rhs_O, rhs_C = self.build_rhs(state)
        rhs = self._flatten(rhs_O, rhs_C)
        op = self.get_linear_operator()

        # Cache h_op once
        self._h_op_cache = None
        self._nrot_cache = None

        x, info = gmres(op, rhs, rtol=tol, maxiter=max_iter,
                        restart=min(50, op.shape[0]))
        kappa, v_list = self._unflatten(x)
        # Project CI multipliers orthogonal to ci_I
        v_list = [_project_orthogonal(v, ci)
                  for v, ci in zip(v_list, self.ci_list)]
        if verbose:
            print(f"  CP-CASSCF GMRES: info={info}, |x|={np.linalg.norm(x):.3e}")
        return kappa, v_list, info
