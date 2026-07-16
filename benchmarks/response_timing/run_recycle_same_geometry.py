"""Same-geometry warm-start: does recycling the previous response's Krylov
subspace cut the iteration count for the next RHS, without changing the answer?

At a fixed geometry the SA-DMRG-CASSCF response operator A is identical for the
state gradients and the interstate NACs; only the RHS changes.  The solver can
therefore seed each solve with a least-squares projection of the new RHS onto
the previous solve's Arnoldi subspace (initial_guess="gmres-recycle").  This is
a warm start only: the GMRES solve still forms the explicit residual and is
accepted under the same true-residual certificate, so the accepted solution is
unchanged.

This driver solves the sequence (grad 0, grad 1, NAC 0-1) twice -- once from a
zero guess, once chained with recycling -- on the same converged reference.  It
records the per-RHS iteration count and wall time for both, certifies every
accepted solution, and verifies that the orbital response is invariant to the
warm start (so any iteration saving is genuine, not a changed answer).

Usage:  python run_recycle_same_geometry.py [--only heh_cas22]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DEV = _HERE.parents[1] / "src" / "dmrg_analytic_dev"
SHARC = _HERE.parents[1] / "sharc_interface"
for p in (str(DEV), str(SHARC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response
from cp_dmrg_response_mps_krylov import MPSKrylovVector
from certified_response import certify_response

ANG = 1.8897261246257702

SYSTEMS = {
    "heh_cas22": dict(
        atoms=["He", "H"],
        coords_ang=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        basis="3-21G", charge=1, spin=0, ncas=2, nelecas=2, bond_dim=200,
    ),
    "h2o_cas44": dict(
        atoms=["O", "H", "H"],
        coords_ang=np.array([
            [0.000, 0.000, 0.117],
            [0.000, 0.757, -0.469],
            [0.000, -0.757, -0.469],
        ]),
        basis="6-31G", charge=0, spin=0, ncas=4, nelecas=4, bond_dim=400,
    ),
}

# the same-geometry RHS sequence: two state gradients then their NAC
RHS_SEQUENCE = [("grad", 0), ("grad", 1), ("nac", (0, 1))]


def _solve_one(obj, kind, key, *, tol, max_iter):
    t0 = time.perf_counter()
    if kind == "grad":
        kappa, ci, info, meta = obj.solve_mps(int(key), tol=tol, max_iter=max_iter)
    else:
        kappa, ci, info, meta = obj.solve_nac_mps(tuple(key), tol=tol,
                                                  max_iter=max_iter)
    wall = time.perf_counter() - t0
    z = MPSKrylovVector(obj, kappa, ci, label=f"{kind}-{key}")
    state_for_rhs = key if kind == "grad" else key[0]
    return z, info, meta, wall, int(state_for_rhs)


def run_one(key, *, tol=1.0e-8, max_iter=60, n_threads=1, stack_mem_mb=None):
    s = SYSTEMS[key]
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=s["bond_dim"], n_sweeps=30, sweep_tol=1.0e-10,
               n_threads=int(n_threads))
    if stack_mem_mb is not None:
        cfg["stack_mem_mb"] = int(stack_mem_mb)
    coords = s["coords_ang"] * ANG
    _mol, _mf, mc, _solver = fdv.build_sa_dmrg_casscf(
        s["atoms"], coords, basis=s["basis"], charge=s["charge"],
        spin=s["spin"], ncas=s["ncas"], nelecas=s["nelecas"], nroots=2,
        weights=[0.5, 0.5], solver_cfg=cfg,
    )

    modes = {}
    kappas = {}
    for mode in ("zero", "gmres-recycle"):
        obj = _make_mps_krylov_response(mc)
        obj._initial_guess = mode
        rows = []
        for kind, rkey in RHS_SEQUENCE:
            z, info, meta, wall, state = _solve_one(
                obj, kind, rkey, tol=tol, max_iter=max_iter)
            if kind == "nac":
                cert = certify_response(
                    obj, z, state=state, tol=1.0e-6, rhs_kind="nac",
                    nac_pair=rkey, solver=f"global_mps_krylov[{mode}]",
                    wall_s=wall)
            else:
                cert = certify_response(
                    obj, z, state=state, tol=1.0e-6, rhs_kind="grad",
                    solver=f"global_mps_krylov[{mode}]", wall_s=wall)
            rows.append({
                "rhs": f"{kind}:{rkey}",
                "niter": meta.get("niter"),
                "wall_s": wall,
                "true_residual_relative": cert.true_residual_relative,
                "converged": cert.converged,
            })
            kappas[(mode, f"{kind}:{rkey}")] = obj.mc.pack_uniq_var(
                obj._canonical_kappa(z.kappa))
        modes[mode] = rows

    # the warm start must not change the accepted solution: compare the orbital
    # response (a basis-independent vector in the uniq-rotation parameterization)
    max_kappa_drift = 0.0
    for kind, rkey in RHS_SEQUENCE:
        tag = f"{kind}:{rkey}"
        d = float(np.linalg.norm(kappas[("zero", tag)] - kappas[("gmres-recycle", tag)]))
        max_kappa_drift = max(max_kappa_drift, d)

    zero_iters = [r["niter"] for r in modes["zero"]]
    rec_iters = [r["niter"] for r in modes["gmres-recycle"]]
    total_zero = sum(i for i in zero_iters if i is not None)
    total_rec = sum(i for i in rec_iters if i is not None)
    all_conv = all(r["converged"] for rows in modes.values() for r in rows)

    return {
        "system": key, "ncas": s["ncas"], "nelecas": s["nelecas"],
        "basis": s["basis"], "bond_dim": s["bond_dim"],
        "rhs_sequence": [f"{k}:{v}" for k, v in RHS_SEQUENCE],
        "zero_guess": modes["zero"],
        "recycled": modes["gmres-recycle"],
        "total_iters_zero": total_zero,
        "total_iters_recycled": total_rec,
        "iter_reduction": total_zero - total_rec,
        "max_kappa_drift_zero_vs_recycled": max_kappa_drift,
        "solution_invariant": bool(max_kappa_drift < 1.0e-6),
        "all_converged": bool(all_conv),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="heh_cas22")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stack-mem-mb", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    keys = [args.only] if args.only != "all" else list(SYSTEMS)
    results = []
    for key in keys:
        print(f"=== {key} ===", flush=True)
        try:
            r = run_one(key, n_threads=args.threads, stack_mem_mb=args.stack_mem_mb)
            print(json.dumps(r, indent=2), flush=True)
        except Exception as exc:
            r = {"system": key, "status": "error",
                 "exception": type(exc).__name__, "message": str(exc),
                 "traceback_tail": traceback.format_exc()[-2500:]}
            print(f"  ERROR {key}: {exc}", flush=True)
        results.append(r)
    out = (Path(args.out) if args.out
           else _HERE / "data" / "recycle_same_geometry.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": "same_geometry_warm_start_recycling",
        "results": results,
    }, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    ok = all(r.get("solution_invariant") and r.get("all_converged")
             for r in results if "status" not in r)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
