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
from scipy.linalg import logm, polar, schur

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


def discrete_gauge(s):
    """Factor an active cross-overlap ``s`` as ``s = G @ s_res``.

    ``G`` is the signed permutation closest to the polar unitary ``U`` of ``s``
    (round each row of ``U`` to its largest-magnitude column with that entry's
    sign); ``s_res = G.T @ s`` is then near the SPD stretch (near-identity,
    det>0), so ``logm(polar(s_res)[0])`` is real even when ``s`` has a sign flip
    / column swap / reflection (det U ~ -1) that makes ``logm(polar(s)[0])``
    complex.  ``G`` is exactly orthogonal (det +-1); it is the discrete gauge of
    the displaced active orbitals (relabel + sign) that must be applied to the
    displaced state for the overlap to stay invariant.

    Returns ``(G, s_res)``.
    """
    s = np.asarray(s, dtype=float)
    n = s.shape[0]
    U = polar(s)[0]
    G = np.zeros((n, n))
    used = set()
    for p in sorted(range(n), key=lambda r: -np.max(np.abs(U[r]))):
        for q in np.argsort(-np.abs(U[p])):
            q = int(q)
            if q not in used:
                used.add(q)
                G[p, q] = 1.0 if U[p, q] >= 0 else -1.0
                break
    s_res = G.T @ s
    return G, s_res


def real_so_log(U):
    """Real antisymmetric log of a special-orthogonal ``U`` (det +1).

    Valid even when ``U`` has eigenvalues near -1 (rotation angle ~pi -- e.g. a
    180-deg rotation of a near-degenerate active-orbital pair, the polyene-C20
    case), where ``scipy.linalg.logm`` returns the complex principal branch.  Uses
    the real Schur form: 2x2 rotation blocks map to ``theta*[[0,-1],[1,0]]``, and
    pairs of -1 eigenvalues (each pair = a 180-deg rotation, det +1) map to a
    ``pi`` rotation block.  An odd number of -1 eigenvalues is a genuine reflection
    (det -1) with no real log -> raised (needs discrete sign surgery).
    """
    U = np.asarray(U, dtype=float)
    n = U.shape[0]
    T, Z = schur(U, output="real")
    L = np.zeros((n, n))
    neg = []
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > 1.0e-9:
            theta = np.arctan2(T[i + 1, i], T[i, i])
            L[i, i + 1] = -theta
            L[i + 1, i] = theta
            i += 2
        else:
            if T[i, i] < 0:
                neg.append(i)
            i += 1
    for a in range(0, len(neg) - 1, 2):
        p, q = neg[a], neg[a + 1]
        L[p, q] += -np.pi
        L[q, p] += np.pi
    if len(neg) % 2 == 1:
        raise ValueError("orbital-rotation matrix is a det=-1 reflection (odd "
                         "sign flips); no real antisymmetric log -- discrete sign "
                         "surgery on the MPS is required")
    return Z @ L @ Z.T


def _real_log(M):
    M = np.asarray(M, dtype=float)
    Z = logm(M)
    if np.iscomplexobj(Z):
        if np.max(np.abs(Z.imag)) > 1.0e-7:
            # Near-(-1) eigenvalues: scipy returns the complex principal branch.
            # A special-orthogonal matrix (det +1) still has a REAL antisymmetric
            # log (e.g. a 180-deg rotation of a near-degenerate pair, the C20
            # overlap case); build it from the real Schur form.  A genuine
            # reflection (det -1) has no real log and raises a clear message.
            if (abs(float(np.linalg.det(M)) - 1.0) < 1.0e-6
                    and np.max(np.abs(M @ M.T - np.eye(M.shape[0]))) < 1.0e-6):
                return np.ascontiguousarray(real_so_log(M))
            raise ValueError(f"orbital-rotation log has imaginary part "
                             f"{np.max(np.abs(Z.imag)):.2e}; det={float(np.linalg.det(M)):.3f} "
                             f"(det -1 reflection needs discrete sign surgery)")
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
    O_raw = np.zeros((nst, nst))
    for j in range(nst):
        loaded = load_foreign_mps(host_drv, host_frame, obj_disp._driver_su2,
                                  obj_disp._su2_frame, obj_disp._state_mps[j],
                                  tag=f"{tag}{j}")
        block2.Global.frame = host_frame
        rot = rotate_mps_orbitals(host_drv, loaded, s.T, ncas=ncas,
                                  tag=f"{tag}rot{j}", include_stretch=False)
        O_raw[:, j] = np.array([obj_ref._mps_overlap(obj_ref._state_mps[i], rot)
                                for i in range(nst)])
    O, diag_min, swapped = _assign_states_by_overlap(O_raw)
    if swapped or diag_min < 0.9:
        _kind = ("GENUINE state-set instability (displaced set lacks the smooth "
                 "continuation -> FD NAC inadmissible, certificate-carried)"
                 if diag_min < 0.9 else "relabeling repaired by max-overlap assignment")
        print(f"[cross_overlap {tag}] state assignment: swapped={swapped} "
              f"min_matched_overlap={diag_min:.3f} -- {_kind}", flush=True)
    return O, s


def _assign_states_by_overlap(O_raw):
    """Maximum-overlap state assignment + phase tracking for a cross-geometry
    wavefunction-overlap matrix (the standard overlap-based-coupling remedy:
    Plasser et al. 2016; Granucci/Persico local diabatization; SHARC).

    ``O_raw[i, j] = <ref_i | disp_j>`` uses the displaced states in ENERGY order,
    which mislabels the off-diagonal coupling when a displaced excited state
    swaps/mixes character relative to the reference -- e.g. the polyene 1Bu/2Ag
    near-degeneracy, where the 1--2 gap is small even though the 0--1 gap is wide.
    We assign each reference state to the displaced state it most overlaps (greedy
    max-|overlap|), then sign-fix and reorder so column k tracks reference state k.
    The off-diagonal coupling overlap is then taken between continuously tracked
    states, not energy-ordered (possibly swapped) ones.  The smallest matched
    |overlap| ``diag_min`` is a continuity diagnostic: ``diag_min`` well below 1
    means the displaced state set does not contain the smooth continuation of a
    reference state (a genuine state-set instability the assignment cannot repair,
    e.g. SA(2) averaging a different second state) -- a finite-difference NAC is
    then genuinely inadmissible and the analytic coupling is certificate-carried.
    The antisymmetric (Hammes-Schiffer--Tully) finite-difference form is preserved:
    we do NOT symmetrize the off-diagonal, so the derivative coupling survives.
    """
    n = O_raw.shape[0]
    A = np.abs(O_raw)
    assign = [-1] * n          # assign[k] = displaced state tracked to reference k
    used = set()
    for _ in range(n):
        bi = bj = -1
        bv = -1.0
        for i in range(n):
            if assign[i] >= 0:
                continue
            for j in range(n):
                if j in used:
                    continue
                if A[i, j] > bv:
                    bv, bi, bj = A[i, j], i, j
        assign[bi] = bj
        used.add(bj)
    swapped = any(assign[k] != k for k in range(n))
    O = np.zeros((n, n))
    diag = []
    for k in range(n):
        j = assign[k]
        sgn = 1.0 if O_raw[k, j] >= 0 else -1.0
        O[:, k] = sgn * O_raw[:, j]
        diag.append(abs(O_raw[k, j]))
    return O, (min(diag) if diag else 1.0), swapped


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
            dst = os.path.join(dst_dir, os.path.basename(f))
            # When basin-pinning shares one persistent_dir across R/P/M, the
            # displaced MPS already lives in dst_dir; copying it onto itself
            # raises shutil.SameFileError. Skip the copy in that case (the file
            # is already in place) but still record it as present.
            if os.path.abspath(f) != os.path.abspath(dst):
                shutil.copy(f, dst)
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
