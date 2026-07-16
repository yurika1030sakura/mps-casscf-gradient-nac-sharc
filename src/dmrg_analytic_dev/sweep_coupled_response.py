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


def _ci_block_inverse(obj, w_list, *, n_sweeps, tol, solver_type, proj_weight,
                      bra_schedule=None, noises=None):
    """Apply H_CC^{-1} per state: solve (H_CC - E_i) v_i = w_i, v_i orthogonal
    to the roots.  Zero (or near-zero) slots map to the cached zero MPS.

    The true correction vector (H_CC-E)^{-1} w is intrinsically higher-rank than
    the wavefunction, so a single fixed ``bra_bond_dims=[m_compress]`` floors the
    residual (representability ceiling, not an iteration-count failure).  Passing
    a GROWING ``bra_schedule`` (+ optional ReducedPerturbative ``noises``) lets
    block2 grow |x> to its own rank -- the standard dynamical-DMRG correction
    vector.  ``bra_schedule``/``noises`` args override the obj-level
    ``_ci_bra_schedule``/``_ci_noises``; defaults reproduce the old fixed-m
    behaviour.  This is used to run the orbital-Schur loop at a CHEAP moderate m
    (the Schur complement is insensitive to the high-rank CI tail) while the final
    response vector z_C is solved once at a high adaptive schedule.
    """
    if bra_schedule is not None:
        schedule = list(bra_schedule)
    else:
        schedule = list(getattr(obj, "_ci_bra_schedule", None) or [int(obj._m_compress)])
    if noises is None:
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
    ci_m_loop: int | None = None,
    ci_schedule_final: list | None = None,
    ci_noises_final: list | None = None,
    verbose: bool = False,
    kappa_only: bool = False,
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

    import time as _time
    _loop_sched = [int(ci_m_loop)] if ci_m_loop is not None else None

    def Hcc_inv(w_list):
        # Orbital-Schur loop: cheap moderate-m CI-solves (the Schur complement is
        # insensitive to the high-rank CI tail, so the orbital answer converges at
        # moderate m -- this is what makes the orbital GMRES affordable).
        return _ci_block_inverse(
            obj, w_list, n_sweeps=ci_sweeps, tol=ci_tol,
            solver_type=solver_type, proj_weight=proj_weight,
            bra_schedule=_loop_sched,
        )

    def Hcc_inv_final(w_list):
        # The actual response vector z_C: solve once at the high ADAPTIVE schedule
        # (grow to the correction vector's own rank).
        return _ci_block_inverse(
            obj, w_list, n_sweeps=max(int(ci_sweeps),
                                      len(ci_schedule_final) if ci_schedule_final else 1),
            tol=ci_tol, solver_type=solver_type, proj_weight=proj_weight,
            bra_schedule=ci_schedule_final, noises=ci_noises_final,
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
    _gm = {"it": 0, "t0": _time.perf_counter()}

    def _gmres_cb(rk):
        _gm["it"] += 1
        if verbose:
            rv = float(rk) if np.isscalar(rk) else float(np.linalg.norm(rk))
            print(f"  [schur] orbital-GMRES iter {_gm['it']} resid={rv:.3e} "
                  f"n_schur_applies={n_schur_applies['count']} "
                  f"wall={_time.perf_counter() - _gm['t0']:.0f}s", flush=True)

    # ONE Arnoldi cycle, no restarts: the moderate-m loop CI-solve floors the TRUE
    # residual (the metric scipy's rtol checks) at the m-noise level, and scipy's
    # restart cycles re-trigger residual spikes around that floor (observed
    # oscillation iter12=3e-10 -> iter13=4.5e-6).  GMRES is monotone WITHIN a
    # single Arnoldi cycle, so restart=orb_max_iter+maxiter=1 converges
    # monotonically to the noise floor and returns the best iterate -- no
    # oscillation, and size-robust (no per-size tol tuning needed).
    z_packed, orb_info = spla.gmres(
        S, rhs_schur, rtol=orb_tol, atol=0.0,
        restart=int(orb_max_iter), maxiter=1,
        M=spla.LinearOperator((n_orb, n_orb), matvec=hoo_precond),
        callback=_gmres_cb, callback_type="pr_norm",
    )
    z_kappa = obj._canonical_kappa(unpack(z_packed))
    if verbose:
        print(f"  [schur] orbital-GMRES done: info={orb_info} iters={_gm['it']} "
              f"total_schur_applies={n_schur_applies['count']} "
              f"wall={_time.perf_counter() - _gm['t0']:.0f}s", flush=True)

    if kappa_only:
        # Cheap orbital-only solve for the per-size m-convergence audit: the
        # moderate-m loop's adequacy is size-dependent (at larger CAS the same m
        # covers less of the higher-rank CI space), so the caller compares z_kappa
        # across ci_m_loop values to verify moderate-m suffices AT THIS size
        # rather than assuming a tuning calibrated on a smaller system.
        return z_kappa, None, int(orb_info), {
            "kappa_only": True, "orb_iters": int(_gm["it"]),
            "n_schur_applies": int(n_schur_applies["count"]), **hoo_diag}

    if verbose:
        print("  [schur] now final z_C solve (adaptive high-m correction vector)",
              flush=True)
    # --- recover z_C = H_CC^{-1}(b_C - H_CO z_kappa) ---
    hco_zk = obj.H_CO_apply_mps(z_kappa)
    rhs_C = [
        obj._combine_mps([(1.0, b_ci[i]), (-1.0, hco_zk[i])],
                         tag=obj._new_tag(f"SCHUR-RHSC{i}"))
        for i in range(len(b_ci))
    ]
    if verbose:
        print("  [schur] final z_C solve (adaptive high-m correction vector)...",
              flush=True)
    z_C = Hcc_inv_final(rhs_C)

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
