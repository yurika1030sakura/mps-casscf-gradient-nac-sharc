"""Certified auto-solver for CP-DMRG-CASSCF gradient/NAC response.

Picks a solver, accepts its result only if the a posteriori certificate
(:mod:`certified_response`) confirms the true residual against the full coupled
operator, and otherwise falls back to the robust global MPS-Krylov solver.  The
returned solution is always the one that passed the certificate.

For a state gradient the sweep-localized Schur solver is tried first (it routes
the active-space work through block2 sweeps); if its certified residual is not
below tolerance, the global MPS-Krylov solver is used as a fallback.  Interstate
NAC right-hand sides are solved directly with the global solver, which is the
route that builds the NAC RHS.  Either way the accepted solution carries a
certificate, and the per-attempt records are kept in ``certificate.extra``.
"""

from __future__ import annotations

from typing import Optional

from cp_dmrg_response_mps_krylov import MPSKrylovVector
from certified_response import certify_response


def _attempt_record(label, cert):
    return {
        "solver": label,
        "true_residual_relative": cert.true_residual_relative,
        "root_projector_leakage": cert.root_projector_leakage,
        "converged": cert.converged,
        "wall_s": cert.wall_s,
    }


def solve_response_auto(
    obj,
    *,
    state: Optional[int] = None,
    nac_pair=None,
    tol: float = 1.0e-7,
    max_iter: int = 60,
    cert_tol: float = 1.0e-6,
    prefer_schur: bool = True,
):
    """Solve and certify one response RHS.

    Exactly one of ``state`` (gradient) or ``nac_pair`` (interstate coupling)
    must be given.  Returns ``(vector, certificate)``; the certificate's
    ``extra["attempts"]`` lists every solver tried and its certified residual,
    and ``extra["accepted_solver"]`` names the one whose solution is returned.
    """
    import time

    if (state is None) == (nac_pair is None):
        raise ValueError("pass exactly one of state= (gradient) or nac_pair= (NAC)")

    attempts = []

    # --- interstate NAC: global solver only (it builds the NAC RHS) ---
    if nac_pair is not None:
        pair = tuple(int(s) for s in nac_pair)
        t0 = time.perf_counter()
        kappa, ci, _info, _meta = obj.solve_nac_mps(pair, tol=tol, max_iter=max_iter)
        wall = time.perf_counter() - t0
        z = MPSKrylovVector(obj, kappa, ci, label=f"AUTO-NAC{pair}")
        cert = certify_response(
            obj, z, state=pair[0], rhs_kind="nac", nac_pair=pair,
            tol=cert_tol, solver="global_mps_krylov", wall_s=wall,
        )
        attempts.append(_attempt_record("global_mps_krylov", cert))
        cert.extra["attempts"] = attempts
        cert.extra["accepted_solver"] = "global_mps_krylov"
        return z, cert

    # --- state gradient: Schur first, global fallback ---
    state = int(state)
    if prefer_schur:
        from sweep_coupled_response import solve_state_sweep_schur
        try:
            t0 = time.perf_counter()
            kappa, ci, _info, _meta = solve_state_sweep_schur(
                obj, state, orb_tol=tol, ci_tol=max(1.0e-2 * tol, 1.0e-11),
            )
            wall = time.perf_counter() - t0
            z = MPSKrylovVector(obj, kappa, ci, label=f"AUTO-SCHUR-G{state}")
            cert = certify_response(
                obj, z, state=state, rhs_kind="grad",
                tol=cert_tol, solver="sweep_schur", wall_s=wall,
            )
            attempts.append(_attempt_record("sweep_schur", cert))
            if cert.converged:
                cert.extra["attempts"] = attempts
                cert.extra["accepted_solver"] = "sweep_schur"
                return z, cert
        except Exception as exc:  # noqa: BLE001 - fall back on any Schur failure
            attempts.append({"solver": "sweep_schur", "error": repr(exc)})

    t0 = time.perf_counter()
    kappa, ci, _info, _meta = obj.solve_mps(state, tol=tol, max_iter=max_iter)
    wall = time.perf_counter() - t0
    z = MPSKrylovVector(obj, kappa, ci, label=f"AUTO-GLOBAL-G{state}")
    cert = certify_response(
        obj, z, state=state, rhs_kind="grad",
        tol=cert_tol, solver="global_mps_krylov", wall_s=wall,
    )
    attempts.append(_attempt_record("global_mps_krylov", cert))
    cert.extra["attempts"] = attempts
    cert.extra["accepted_solver"] = "global_mps_krylov"
    return z, cert


def compute_all_responses_certified(
    obj,
    *,
    gradient_states=(),
    nac_pairs=(),
    tol: float = 1.0e-7,
    max_iter: int = 60,
    cert_tol: float = 1.0e-6,
    recycle: bool = True,
):
    """Solve and certify every response RHS at one geometry on a single object.

    The coupled operator is identical for all RHS at a fixed geometry, so the
    geometry/RDM/ERI caches are built once and (when ``recycle=True``) each solve
    is warm-started from the previous solve's Arnoldi subspace.  Recycling only
    seeds the initial guess; each accepted solution still passes the true-residual
    certificate, so the answers are independent of the warm start.

    Returns an ordered ``dict`` keyed by ``("grad", I)`` / ``("nac", (I, J))``
    with values ``(vector, certificate)``.  Each certificate's
    ``extra["niter"]`` records the iteration count of the accepted solve.
    """
    import time

    if recycle:
        obj._initial_guess = "gmres-recycle"

    results = {}
    for s in gradient_states:
        s = int(s)
        t0 = time.perf_counter()
        kappa, ci, _info, meta = obj.solve_mps(s, tol=tol, max_iter=max_iter)
        wall = time.perf_counter() - t0
        z = MPSKrylovVector(obj, kappa, ci, label=f"BATCH-G{s}")
        cert = certify_response(obj, z, state=s, rhs_kind="grad",
                                tol=cert_tol, solver="global_mps_krylov",
                                wall_s=wall)
        cert.extra["niter"] = meta.get("niter")
        results[("grad", s)] = (z, cert)

    for pair in nac_pairs:
        p = tuple(int(x) for x in pair)
        t0 = time.perf_counter()
        kappa, ci, _info, meta = obj.solve_nac_mps(p, tol=tol, max_iter=max_iter)
        wall = time.perf_counter() - t0
        z = MPSKrylovVector(obj, kappa, ci, label=f"BATCH-NAC{p}")
        cert = certify_response(obj, z, state=p[0], rhs_kind="nac", nac_pair=p,
                                tol=cert_tol, solver="global_mps_krylov",
                                wall_s=wall)
        cert.extra["niter"] = meta.get("niter")
        results[("nac", p)] = (z, cert)

    return results
