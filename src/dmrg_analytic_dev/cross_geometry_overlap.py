"""MPS-native cross-geometry wavefunction overlap (FCI-free).

The finite-difference derivative coupling needs <Psi_I(R) | Psi_J(R')>, an
overlap of two CASSCF states whose active orbitals differ (the MOs at R vs R').
The active-orbital cross overlap is

    s[p, q] = <phi_p^A | phi_q^B> = C_a_act^T @ S_AO(R, R') @ C_b_act        (ncas x ncas)

which is near-identity for a small displacement.  Expressing the ket in the bra
basis is the Thouless orbital rotation that sends the creation operators
a^+_q(B) = sum_p s[p, q] a^+_p(A); the corresponding many-body operator is

    R(s) = exp( sum_pq [log s]_pq  E_pq ),   E_pq = sum_sigma a^+_{p sigma} a_{q sigma}.

We build the one-body generator as a (deliberately non-symmetrized) quantum-
chemistry MPO and apply exp(generator) with block2's time-evolution driver, then
take an ordinary same-basis MPS overlap.  No determinant expansion is formed, so
the cost is independent of the determinant-space size -- the route that lets the
NAC finite differences reach active spaces where an FCI vector cannot be built.

The orbital-rotation convention is calibrated against the determinant-level
``overlap_fci`` reference on small active spaces (see
``test_cross_geometry_overlap.py``); ``ROTATION_CONVENTION`` records the verified
choice.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import logm, polar

# Calibrated against overlap_fci (see test_cross_geometry_overlap.py /
# diag_rotation_parts.py).  The single-particle rotation by s uses its polar
# factors s = U P.  The unitary factor U is applied with the anti-Hermitian
# generator log(U) and reproduces overlap_fci to ~1e-6 at finite-difference
# magnitude (exact in the small-h limit); ``u_sign = -1`` (apply exp(-X-hat)) is
# the verified choice.
#
# The symmetric factor P is the *non-unitary* stretch.  For a central-difference
# derivative coupling it is negligible: at a finite-difference step the active
# cross overlap is unitary to O(h) (the non-unitary part is O(h^2) active-space
# leakage), and that part is even in the displacement, so it cancels to O(h^2)
# in (overlap(+h) - overlap(-h)) / 2h.  The default rotation is therefore
# U-only; the stretch is available (``include_stretch=True``) for completeness
# but is not used in the NAC path.
ROTATION_CONVENTION = {"u_sign": -1.0, "p_sign": -1.0, "order": "P_then_U"}


def _real_log(M):
    Z = logm(np.asarray(M, dtype=float))
    if np.iscomplexobj(Z):
        if np.max(np.abs(Z.imag)) > 1.0e-7:
            raise ValueError(f"orbital-rotation log has imaginary part "
                             f"{np.max(np.abs(Z.imag)):.2e}; s not near-identity?")
        Z = Z.real
    return np.ascontiguousarray(Z)


def _apply_onebody_exp(driver, mps, Gmat, *, ncas, tag, sign, hermitian,
                       n_steps, bond_dim):
    """Apply exp(sign * sum_ij Gmat[i,j] E_ij) to ``mps`` via time evolution.

    td_dmrg computes exp(-t * mpo); with mpo = G-hat and target_t = -sign we get
    exp(sign * G-hat).  ``hermitian`` selects the (anti-)Hermitian integrator.
    """
    g2e = np.zeros((ncas, ncas, ncas, ncas))
    mpo = driver.get_qc_mpo(h1e=np.ascontiguousarray(Gmat), g2e=g2e,
                            symmetrize=False, add_ident=False, iprint=0)
    bd = int(bond_dim) if bond_dim is not None else int(mps.info.bond_dim)
    target_t = -float(sign)
    return driver.td_dmrg(
        mpo, mps, delta_t=target_t / n_steps, target_t=target_t,
        n_steps=n_steps, te_type="rk4", n_sub_sweeps=2, normalize_mps=False,
        hermitian=bool(hermitian), bond_dims=[bd], iprint=0,
    )


def rotate_mps_orbitals(driver, mps, s, *, ncas, tag, n_steps=24,
                        bond_dim=None, convention=None, include_stretch=False):
    """Return the MPS with its single-particle basis rotated by ``s``.

    By default only the unitary (polar) factor of ``s`` is applied -- the
    rotation that carries the finite-difference derivative coupling.  Set
    ``include_stretch=True`` to also apply the symmetric non-unitary factor.
    ``s`` is the ncas x ncas orbital transformation; for a finite-difference
    step it is near-identity, so the generator is small and the evolution is
    accurate.
    """
    conv = dict(ROTATION_CONVENTION if convention is None else convention)
    U, P = polar(np.asarray(s, dtype=float))     # s = U @ P
    Xg = _real_log(U)                            # antisymmetric -> anti-Hermitian

    ket = driver.copy_mps(mps, f"{tag}-0")
    if include_stretch:
        Yg = _real_log(P)                        # symmetric -> Hermitian
        ket = _apply_onebody_exp(driver, ket, Yg, ncas=ncas, tag=f"{tag}-P",
                                 sign=conv["p_sign"], hermitian=True,
                                 n_steps=n_steps, bond_dim=bond_dim)
    return _apply_onebody_exp(driver, ket, Xg, ncas=ncas, tag=f"{tag}-U",
                              sign=conv["u_sign"], hermitian=False,
                              n_steps=n_steps, bond_dim=bond_dim)


def cross_geometry_active_overlap(mol_a, mol_b, mo_a, mo_b, ncore, ncas):
    """s[p, q] = <phi_p^A | phi_q^B> over the active orbitals (ncas x ncas)."""
    from pyscf import gto
    s_ao = gto.intor_cross("int1e_ovlp", mol_a, mol_b)
    ca = mo_a[:, ncore:ncore + ncas]
    cb = mo_b[:, ncore:ncore + ncas]
    return ca.T @ s_ao @ cb
