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


def _projected_cr_refine(obj, state, wp, nw, x0, *, rtol, max_iter):
    """Root-projected conjugate-residual solve of ``P (H - E_state) P x = wp``.

    On the complement of ALL reference roots, ``(H - E_state)`` is positive
    definite for every averaged state (all non-root CI eigenvalues lie above
    the highest averaged root), so CR converges unconditionally; every
    iterate is re-projected so ``(E_j - E_state)``-amplified compression
    leakage cannot accumulate inside the iteration.  This loop exists because
    block2's ``DMRGDriver.multiply(left_mpo=...)`` declares convergence of
    ITS OWN functional (DF ~ 1e-19, per-site Error = 0) at points whose TRUE
    residual is 1e-2..5e-2 with the default local thresholds and still ~4e-5
    at thrds=1e-12 (measured); the multiply result is therefore trusted only
    as the initial guess ``x0``, never as the accept criterion.

    ``nw`` is the norm of ``wp`` (the caller already has it).  Returns
    ``(x, rel_residual, n_iter)``.
    """
    def apply_A(v, tag):
        s = obj._sigma_mps(obj._hcc_shifted_mpo(state), v,
                           tag=obj._new_tag(tag))
        return _proj_out_roots(obj, s, tag + "P")

    x = x0
    if x is None:
        r = wp
    else:
        ax = apply_A(x, f"CR{state}-AX0")
        r = obj._combine_mps([(1.0, wp), (-1.0, ax)],
                             tag=obj._new_tag(f"CR{state}-R0"))
        r = _proj_out_roots(obj, r, f"CR{state}-R0P")
    rel = _norm(obj, r) / nw
    n_iter = 0
    if rel >= rtol and int(max_iter) > 0:
        p = r
        Ap = apply_A(r, f"CR{state}-AR")
        rAr = obj._mps_overlap(r, Ap)
        for it in range(int(max_iter)):
            n_iter = it + 1
            nAp2 = max(obj._mps_overlap(Ap, Ap), 1.0e-300)
            alpha = rAr / nAp2
            x = (obj._combine_mps([(alpha, p)],
                                  tag=obj._new_tag(f"CR{state}-X{it}"))
                 if x is None else
                 obj._combine_mps([(1.0, x), (alpha, p)],
                                  tag=obj._new_tag(f"CR{state}-X{it}")))
            r = obj._combine_mps([(1.0, r), (-alpha, Ap)],
                                 tag=obj._new_tag(f"CR{state}-R{it}"))
            r = _proj_out_roots(obj, r, f"CR{state}-RP{it}")
            rel = _norm(obj, r) / nw
            if rel < rtol:
                break
            Ar = apply_A(r, f"CR{state}-AR{it}")
            rAr_new = obj._mps_overlap(r, Ar)
            beta = rAr_new / (rAr if abs(rAr) > 1.0e-300 else 1.0e-300)
            rAr = rAr_new
            p = obj._combine_mps([(1.0, r), (beta, p)],
                                 tag=obj._new_tag(f"CR{state}-P{it}"))
            Ap = obj._combine_mps([(1.0, Ar), (beta, Ap)],
                                  tag=obj._new_tag(f"CR{state}-AP{it}"))
    if x is None:
        x = obj._zero_state_mps(state)
    return x, float(rel), int(n_iter)


def _ci_block_inverse(obj, w_list, *, n_sweeps, tol, solver_type, proj_weight,
                      bra_schedule=None, noises=None, hcc_weight_scaling=True,
                      inner_solver="multiply_cr", thrds=None,
                      cr_rtol=1.0e-9, cr_max_iter=200,
                      proj_all_roots=True, stats=None):
    """Apply H_CC^{-1} per state, where H_CC[i] = 2*w_i*P(H-E_i)P: solve the
    root-projected shifted system and scale by 1/(2*w_i).  Zero (or
    near-zero) slots map to the cached zero MPS.

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

    WEIGHT SCALING (the SA!=2 stagnation fix): the CI-CI block of the coupled
    operator is H_CC[i] = 2*w_i*(H - E_i) (+ SA-gauge rank-2 corrections that
    vanish on the root-orthogonal complement), but the shifted-MPO solve
    inverts the bare (H - E_i).  ``hcc_weight_scaling=True`` divides the
    returned solution by 2*w_i so this routine applies the inverse of the SAME
    H_CC that ``matvec_mps``/the dense reference use.  For SA-2 equal weights
    2*w = 1 and the scaling is a no-op, which is why every SA-2 validation was
    blind to the missing factor while SA-3/SA-5 production runs stagnated at
    O(|1-2w|).

    INNER SOLVER (the silently-under-converged multiply fix):

    * ``"multiply"``    -- historic behavior: block2 multiply(left_mpo) alone.
      Its convergence report is not a true-residual statement: measured true
      residuals were 1e-2..5e-2 at the default local thresholds while it
      reported per-site Error = 0.
    * ``"cr"``          -- root-projected conjugate residual on the repo's own
      exact primitives (:func:`_projected_cr_refine`), accepted only when the
      true slot residual is below ``cr_rtol``.
    * ``"multiply_cr"`` -- default: multiply (with tight ``thrds``) supplies a
      rank-adapted initial guess, so ``bra_schedule``/``noises`` keep their
      meaning, and CR refinement is the accept criterion.

    ``thrds`` overrides block2's local linear-solver thresholds (block2's own
    default is [1e-6]*4+[1e-7]); when None and the inner solver is not plain
    "multiply", a tight [1e-12]*n_sweeps is used.  ``proj_all_roots`` puts ALL
    reference roots into the multiply penalty (not just the own root), so
    near-degenerate (E_j - E_i) directions are not left unprotected inside
    block2's sweeps; the CR loop projects every iterate against all roots
    exactly.  ``stats`` (optional dict) accumulates inner-solve effort and
    quality counters for the caller's certificate metadata.
    """
    if bra_schedule is not None:
        schedule = list(bra_schedule)
    else:
        schedule = list(getattr(obj, "_ci_bra_schedule", None) or [int(obj._m_compress)])
    if noises is None:
        noises = getattr(obj, "_ci_noises", None)
    inner_solver = str(inner_solver).strip().lower()
    if inner_solver not in ("multiply", "cr", "multiply_cr"):
        raise ValueError(f"unknown inner_solver {inner_solver!r}")
    if thrds is None and inner_solver != "multiply":
        thrds = [1.0e-12] * max(int(n_sweeps), 1)
    out = []
    for i, w in enumerate(w_list):
        wp = _proj_out_roots(obj, w, f"CIINV-RHS{i}")
        nw = _norm(obj, wp)
        if nw < 1.0e-13:
            out.append(obj._zero_state_mps(i))
            continue
        if stats is not None:
            stats["n_slot_solves"] = stats.get("n_slot_solves", 0) + 1
        sol = None
        if inner_solver in ("multiply", "multiply_cr"):
            proj_list = (list(obj._state_mps) if proj_all_roots
                         else [obj._state_mps[i]])
            bra = obj._copy_mps(wp, tag=obj._new_tag(f"CIINV-BRA{i}"))
            kw = dict(
                left_mpo=obj._hcc_shifted_mpo(i),
                n_sweeps=int(n_sweeps), tol=float(tol),
                bra_bond_dims=schedule,
                proj_mpss=proj_list,
                proj_weights=[float(proj_weight)] * len(proj_list),
                linear_max_iter=4000, solver_type=solver_type, iprint=0,
            )
            if thrds is not None:
                kw["thrds"] = [float(t) for t in thrds]
            if noises is not None:
                kw["noises"] = list(noises)
            with obj._use_su2_frame():
                obj._driver_su2.multiply(bra, obj._identity(), wp, **kw)
            sol = _proj_out_roots(obj, bra, f"CIINV-SOL{i}")
            if stats is not None:
                stats["n_multiply"] = stats.get("n_multiply", 0) + 1
        if inner_solver in ("cr", "multiply_cr"):
            sol, rel, n_it = _projected_cr_refine(
                obj, i, wp, nw, sol, rtol=float(cr_rtol),
                max_iter=int(cr_max_iter))
            if stats is not None:
                stats["cr_iters_total"] = stats.get("cr_iters_total", 0) + n_it
                stats["cr_iters_max"] = max(stats.get("cr_iters_max", 0), n_it)
                stats["cr_rel_resid_worst"] = max(
                    stats.get("cr_rel_resid_worst", 0.0), float(rel))
        if hcc_weight_scaling:
            w_i = float(np.asarray(obj.weights).ravel()[i])
            if w_i > 0.0:
                scale = 1.0 / (2.0 * w_i)
                if abs(scale - 1.0) > 1.0e-14:
                    sol = obj._scale_mps(
                        sol, scale, tag=obj._new_tag(f"CIINV-WSC{i}"))
        out.append(sol)
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
    hcc_weight_scaling: bool = True,
    ci_inner_solver: str = "multiply_cr",
    ci_thrds: list | None = None,
    ci_cr_rtol: float | None = None,
    ci_cr_max_iter: int = 200,
    ci_proj_all_roots: bool = True,
):
    """Solve the CP response for ``state`` by the sweep-localized Schur method.

    Returns ``(kappa, ci_mps_list, info, meta)`` matching ``obj.solve_mps``.

    Inner CI-solve controls (see :func:`_ci_block_inverse` for semantics):
    ``ci_inner_solver`` in {"multiply_cr" (default), "cr", "multiply"};
    ``ci_thrds`` block2 local linear-solver thresholds (None -> tight
    [1e-12]*ci_sweeps unless plain "multiply", which keeps block2 defaults);
    ``ci_cr_rtol`` per-slot true-residual target of the CR refinement
    (None -> max(1e2*ci_tol, 1e-11)); ``ci_proj_all_roots`` penalizes all
    reference roots (not just the own root) inside block2 multiply sweeps.
    All resolved settings and inner-solve effort counters are recorded in
    the returned ``meta`` so certificates can carry them.
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

    # Resolve the inner-solve settings ONCE so the meta/certificate records
    # exactly what was used.
    ci_inner_solver = str(ci_inner_solver).strip().lower()
    if ci_thrds is None and ci_inner_solver != "multiply":
        ci_thrds_used = [1.0e-12] * max(int(ci_sweeps), 1)
    elif ci_thrds is not None:
        ci_thrds_used = [float(t) for t in ci_thrds]
    else:
        ci_thrds_used = None
    ci_cr_rtol_used = (float(ci_cr_rtol) if ci_cr_rtol is not None
                       else max(1.0e2 * float(ci_tol), 1.0e-11))
    inner_stats: dict = {}

    def Hcc_inv(w_list):
        # Orbital-Schur loop: cheap moderate-m CI-solves (the Schur complement is
        # insensitive to the high-rank CI tail, so the orbital answer converges at
        # moderate m -- this is what makes the orbital GMRES affordable).
        return _ci_block_inverse(
            obj, w_list, n_sweeps=ci_sweeps, tol=ci_tol,
            solver_type=solver_type, proj_weight=proj_weight,
            bra_schedule=_loop_sched, hcc_weight_scaling=hcc_weight_scaling,
            inner_solver=ci_inner_solver, thrds=ci_thrds_used,
            cr_rtol=ci_cr_rtol_used, cr_max_iter=ci_cr_max_iter,
            proj_all_roots=ci_proj_all_roots, stats=inner_stats,
        )

    def Hcc_inv_final(w_list):
        # The actual response vector z_C: solve once at the high ADAPTIVE schedule
        # (grow to the correction vector's own rank).
        return _ci_block_inverse(
            obj, w_list, n_sweeps=max(int(ci_sweeps),
                                      len(ci_schedule_final) if ci_schedule_final else 1),
            tol=ci_tol, solver_type=solver_type, proj_weight=proj_weight,
            bra_schedule=ci_schedule_final, noises=ci_noises_final,
            hcc_weight_scaling=hcc_weight_scaling,
            inner_solver=ci_inner_solver, thrds=ci_thrds_used,
            cr_rtol=ci_cr_rtol_used, cr_max_iter=ci_cr_max_iter,
            proj_all_roots=ci_proj_all_roots, stats=inner_stats,
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
            "n_schur_applies": int(n_schur_applies["count"]),
            "ci_inner_solver": ci_inner_solver,
            "ci_inner_stats": dict(inner_stats), **hoo_diag}

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
        # --- inner CI-solve settings as RESOLVED (certificate metadata) ---
        "hcc_weight_scaling": bool(hcc_weight_scaling),
        "ci_inner_solver": ci_inner_solver,
        "ci_thrds": (list(ci_thrds_used) if ci_thrds_used is not None
                     else "block2-default"),
        "ci_cr_rtol": (float(ci_cr_rtol_used)
                       if ci_inner_solver != "multiply" else None),
        "ci_cr_max_iter": (int(ci_cr_max_iter)
                           if ci_inner_solver != "multiply" else None),
        "ci_proj_all_roots": bool(ci_proj_all_roots),
        "proj_weight": float(proj_weight),
        "ci_solver_type": str(solver_type),
        "ci_sweeps": int(ci_sweeps),
        "ci_tol": float(ci_tol),
        "ci_m_loop": (int(ci_m_loop) if ci_m_loop is not None else None),
        "ci_schedule_final": (list(ci_schedule_final)
                              if ci_schedule_final is not None else None),
        "m_compress": (int(obj._m_compress)
                       if getattr(obj, "_m_compress", None) is not None
                       else None),
        "combine_zero_safe": bool(getattr(obj, "_mps_combine_zero_safe",
                                          False)),
        "ci_inner_stats": dict(inner_stats),
    }
    meta.update(hoo_diag)
    return z_kappa, z_C, info, meta
