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


def cross_geometry_overlap_matrix(obj_ref, obj_disp, mol_ref, mol_disp,
                                  mo_ref, mo_disp, ncore, ncas, nst, *,
                                  host_frame, tag):
    """<Psi_I(ref) | Psi_J(disp)> for all I, J -- FCI-free.

    Transports each displaced state into the reference driver, rotates it into
    the reference orbital basis (by the transpose of the active cross overlap),
    and takes the same-basis MPS overlap; each ket column is phase-aligned to the
    reference root via its own diagonal sign.  Returns ``(O, s)`` with ``s`` the
    active cross overlap.
    """
    import block2

    s = cross_geometry_active_overlap(mol_ref, mol_disp, mo_ref, mo_disp,
                                      ncore, ncas)
    host_drv = obj_ref._driver_su2
    O = np.zeros((nst, nst))
    for j in range(nst):
        loaded = load_foreign_mps(host_drv, host_frame, obj_disp._driver_su2,
                                  obj_disp._su2_frame, obj_disp._state_mps[j],
                                  tag=f"{tag}{j}")
        block2.Global.frame = host_frame
        rot = rotate_mps_orbitals(host_drv, loaded, s.T, ncas=ncas,
                                  tag=f"{tag}rot{j}", include_stretch=False)
        col = np.array([obj_ref._mps_overlap(obj_ref._state_mps[i], rot)
                        for i in range(nst)])
        sgn = 1.0 if col[j] >= 0 else -1.0
        O[:, j] = sgn * col
    return O, s


def load_foreign_mps(host_driver, host_frame, foreign_driver, foreign_frame,
                     foreign_mps, tag):
    """Load an MPS built by a *different* driver into ``host_driver``.

    A finite-difference-displaced state is the DMRG state of a different
    geometry, so it lives in a different block2 driver/frame and cannot be
    overlapped against the reference state directly.  Reconstructing it from a
    CI read-out is too lossy (it drops the O(1e-5) displacement components); the
    state must be transported intact.  This makes a uniquely-tagged on-disk copy
    in the foreign frame, moves its block2 files into the host scratch, and
    re-loads it in the host frame.  Both drivers must share the active-space
    size and symmetry (true for two geometries of the same active space).
    """
    import glob
    import os
    import shutil

    import block2

    prev = block2.Global.frame
    block2.Global.frame = foreign_frame
    try:
        # copy_mps writes the MPS info but not the per-site tensor files; flush
        # them explicitly so the on-disk copy is complete before transport.
        cp = foreign_driver.copy_mps(foreign_mps, tag)
        cp.save_data()
    finally:
        block2.Global.frame = prev

    # block2 names an MPS's files with the tag in the MIDDLE
    # (``{tag}-mps_info.bin``, ``F.MPS.{tag}.<site>`` tensors,
    # ``F.MPS.INFO.{tag}.<L/R>.<site>`` state info), so match the tag anywhere.
    src_dir = foreign_driver.frame.mps_dir
    dst_dir = host_driver.frame.mps_dir
    moved = []
    for f in glob.glob(os.path.join(src_dir, "*" + tag + "*")):
        if os.path.isfile(f):
            shutil.copy(f, os.path.join(dst_dir, os.path.basename(f)))
            moved.append(os.path.basename(f))
    import os as _os
    if _os.environ.get("XFER_DEBUG"):
        print(f"[xfer] foreign_tag={getattr(foreign_mps.info,'tag',None)} new_tag={tag}")
        print(f"[xfer] src_dir={src_dir}")
        print(f"[xfer] dst_dir={dst_dir}  same={src_dir==dst_dir}")
        print(f"[xfer] moved {len(moved)} files: {moved}")
    if not moved:
        raise RuntimeError(f"no MPS files found for tag {tag!r} in {src_dir}")

    block2.Global.frame = host_frame
    try:
        return host_driver.load_mps(tag)
    finally:
        block2.Global.frame = prev
