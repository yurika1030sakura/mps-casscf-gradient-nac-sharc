"""Sweep-localized coupled CP-DMRG-SA-CASSCF response solver (Schur form).

The production response solves the coupled orbital/CI linear system

    [ H_OO  H_OC ] [ z_kappa ]   [ b_kappa ]
    [ H_CO  H_CC ] [ z_C     ] = [ b_C     ]

with a single global MPS-Krylov (GMRES) iteration over mixed orbital+CI
vectors, which stores and orthogonalizes a growing Arnoldi basis of full MPS
objects.  This module solves the same system by block elimination:

    (H_OO - H_OC H_CC^{-1} H_CO) z_kappa = b_kappa - H_OC H_CC^{-1} b_C
    z_C = H_CC^{-1} (b_C - H_CO z_kappa)

The small dense orbital Schur system is solved with a standard dense GMRES,
while every H_CC^{-1} application is a per-site sweep solve done by block2's
own linear solver (DMRGDriver.multiply with left_mpo = H_CC - E_i), with the
state root projected out.  The CI work is therefore handled by block2's
optimized sweeps instead of a global Krylov pile of mixed MPS vectors.

Correctness is verified independently of the block algebra: the assembled
solution is checked against the true residual of the *global* operator
(obj.matvec_mps), so a returned solution always satisfies the same equation
the global solver targets.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla


def _norm(obj, v):
    return float(np.sqrt(max(obj._mps_overlap(v, v), 0.0)))


def _proj_out_roots(obj, v, tag):
    """Remove every state-root component from an MPS (the matvec projector P)."""
    out = v
    for j, sm in enumerate(obj._state_mps):
        ov = obj._mps_overlap(sm, out)
        if abs(ov) > 1.0e-14:
            out = obj._combine_mps(
                [(1.0, out), (-ov, sm)], tag=obj._new_tag(f"{tag}-P{j}"),
            )
    return out


def _ci_block_inverse(obj, w_list, *, n_sweeps, tol, solver_type, proj_weight):
    """Apply H_CC^{-1} per state: solve (H_CC - E_i) v_i = w_i, v_i orthogonal
    to the roots.  Zero (or near-zero) slots map to the cached zero MPS.

    The true correction vector (H_CC-E)^{-1} w is intrinsically higher-rank than
    the wavefunction, so a single fixed ``bra_bond_dims=[m_compress]`` floors the
    residual (representability ceiling, not an iteration-count failure).  If the
    response object carries ``_ci_bra_schedule`` (a GROWING bond-dim schedule) and
    optionally ``_ci_noises`` (ReducedPerturbative noise), pass them through so
    block2 grows |x> to its own rank -- the standard dynamical-DMRG correction
    vector.  Defaults reproduce the old fixed-m behaviour exactly.
    """
    schedule = list(getattr(obj, "_ci_bra_schedule", None) or [int(obj._m_compress)])
    noises = getattr(obj, "_ci_noises", None)
    out = []
    for i, w in enumerate(w_list):
        wp = _proj_out_roots(obj, w, f"CIINV-RHS{i}")
        if _norm(obj, wp) < 1.0e-13:
            out.append(obj._zero_state_mps(i))
            continue
        sm = obj._state_mps[i]
        bra = obj._copy_mps(wp, tag=obj._new_tag(f"CIINV-BRA{i}"))
        kw = dict(
            left_mpo=obj._hcc_shifted_mpo(i),
            n_sweeps=int(n_sweeps), tol=float(tol),
            bra_bond_dims=schedule,
            proj_mpss=[sm], proj_weights=[float(proj_weight)],
            linear_max_iter=4000, solver_type=solver_type, iprint=0,
        )
        if noises is not None:
            kw["noises"] = list(noises)
        with obj._use_su2_frame():
            obj._driver_su2.multiply(bra, obj._identity(), wp, **kw)
        out.append(_proj_out_roots(obj, bra, f"CIINV-SOL{i}"))
    return out


def solve_state_sweep_schur(
    obj,
    state: int,
    *,
    rhs=None,
    orb_tol: float = 1.0e-9,
    orb_max_iter: int = 200,
    ci_sweeps: int = 50,
    ci_tol: float = 1.0e-11,
    solver_type: str = "MinRes",
    proj_weight: float = 1.0e3,
    residual_tol: float | None = None,
):
    """Solve the CP response for ``state`` by the sweep-localized Schur method.

    Returns ``(kappa, ci_mps_list, info, meta)`` matching ``obj.solve_mps``.
    """
    state = int(state)
    obj._build_eris_cache()
    obj._build_hcc_state_cache()
    if rhs is None:
        rhs = obj.build_rhs_mps(state)

    b_kappa = obj._canonical_kappa(rhs.kappa)
    b_ci = rhs.ci_mps
    pack = obj.mc.pack_uniq_var
    unpack = obj.mc.unpack_uniq_var
    n_orb = pack(b_kappa).size

    def Hcc_inv(w_list):
        return _ci_block_inverse(
            obj, w_list, n_sweeps=ci_sweeps, tol=ci_tol,
            solver_type=solver_type, proj_weight=proj_weight,
        )

    # --- build dense H_OO once by probing (small relative to the determinant
    #     space; absolute size depends on the orbital partition) ---
    M_OO = np.zeros((n_orb, n_orb))
    for k in range(n_orb):
        e = np.zeros(n_orb); e[k] = 1.0
        kap = obj._canonical_kappa(unpack(e))
        M_OO[:, k] = pack(obj._canonical_kappa(obj.H_OO_apply(kap)))

    # symmetric eigendecomposition for a regularized (pseudo-inverse)
    # orbital preconditioner -- robust to a near-singular orbital Hessian
    A_sym = 0.5 * (M_OO + M_OO.T)
    hoo_w, hoo_V = np.linalg.eigh(A_sym)
    hoo_scale = max(1.0, float(np.max(np.abs(hoo_w))))
    hoo_keep = np.abs(hoo_w) > 1.0e-8 * hoo_scale

    def hoo_precond(y):
        c = hoo_V.T @ np.asarray(y)
        c = np.where(hoo_keep, c / np.where(hoo_keep, hoo_w, 1.0), 0.0)
        return hoo_V @ c

    kept = hoo_w[hoo_keep]
    hoo_diag = {
        "HOO_eig_min_abs": float(np.min(np.abs(hoo_w))),
        "HOO_eig_max_abs": float(np.max(np.abs(hoo_w))),
        "HOO_rank_eff": int(np.count_nonzero(hoo_keep)),
        "HOO_cond_eff": (float(np.max(np.abs(kept)) / np.min(np.abs(kept)))
                         if kept.size else None),
    }

    # --- Schur RHS:  b_kappa - H_OC H_CC^{-1} b_C ---
    binv = Hcc_inv(b_ci)
    rhs_schur = pack(b_kappa) - pack(
        obj._canonical_kappa(obj.H_OC_apply_mps(binv))
    )

    n_schur_applies = {"count": 0}

    def schur_matvec(x):
        n_schur_applies["count"] += 1
        kap = obj._canonical_kappa(unpack(np.asarray(x)))
        hco = obj.H_CO_apply_mps(kap)                 # orbital -> CI
        z = Hcc_inv(hco)                              # H_CC^{-1} H_CO x
        hoc = obj._canonical_kappa(obj.H_OC_apply_mps(z))   # CI -> orbital
        hoo = obj._canonical_kappa(obj.H_OO_apply(kap))
        return pack(hoo - hoc)

    S = spla.LinearOperator((n_orb, n_orb), matvec=schur_matvec)
    z_packed, orb_info = spla.gmres(
        S, rhs_schur, rtol=orb_tol, atol=0.0, maxiter=orb_max_iter,
        M=spla.LinearOperator((n_orb, n_orb), matvec=hoo_precond),
    )
    z_kappa = obj._canonical_kappa(unpack(z_packed))

    # --- recover z_C = H_CC^{-1}(b_C - H_CO z_kappa) ---
    hco_zk = obj.H_CO_apply_mps(z_kappa)
    rhs_C = [
        obj._combine_mps([(1.0, b_ci[i]), (-1.0, hco_zk[i])],
                         tag=obj._new_tag(f"SCHUR-RHSC{i}"))
        for i in range(len(b_ci))
    ]
    z_C = Hcc_inv(rhs_C)

    from cp_dmrg_response_mps_krylov import MPSKrylovVector
    z = MPSKrylovVector(obj, z_kappa, z_C, label=f"SCHUR-Z{state}")

    # --- true residual against the GLOBAL operator (the correctness arbiter) ---
    az = obj.matvec_mps(z)
    r_kappa = obj._canonical_kappa(b_kappa - obj._canonical_kappa(az.kappa))
    r_ci = [
        obj._combine_mps([(1.0, b_ci[i]), (-1.0, az.ci_mps[i])],
                         tag=obj._new_tag(f"SCHUR-RES{i}"))
        for i in range(len(b_ci))
    ]
    res_norm = float(np.sqrt(
        np.sum(pack(r_kappa) ** 2)
        + sum(max(obj._mps_overlap(r, r), 0.0) for r in r_ci)
    ))
    b_norm = float(np.sqrt(
        np.sum(pack(b_kappa) ** 2)
        + sum(max(obj._mps_overlap(c, c), 0.0) for c in b_ci)
    ))
    rel_res = res_norm / max(b_norm, 1.0e-30)
    resid_tol = (float(residual_tol) if residual_tol is not None
                 else max(10.0 * orb_tol, 1.0e-8))
    info = 0 if rel_res < resid_tol else 1
    meta = {
        "method": "sweep_schur",
        "orb_dim": int(n_orb),
        "orb_gmres_info": int(orb_info),
        "schur_applies": int(n_schur_applies["count"]),
        "true_residual_rel": rel_res,
        "residual_tol_used": resid_tol,
    }
    meta.update(hoo_diag)
    return z_kappa, z_C, info, meta
