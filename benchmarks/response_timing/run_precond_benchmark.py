"""Debug: which existing initial-guess config minimizes GMRES iterations?

The trajectory ran through auto_response with initial_guess="gmres-recycle" and
needed ~150 GMRES iterations/step. The CI-block preconditioner "hcc-inverse"
exists but was never enabled. This benchmark solves the same RHS sequence
(grad0, grad1, NAC01) under three configs on identical references and reports
iteration count + certified true residual, so we can see whether a zero-risk
config change fixes the cost before touching the certified GMRES itself.

Usage: python run_precond_benchmark.py --only h2o_cas44
"""
from __future__ import annotations
import argparse, json, sys, time, traceback
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
    "heh_cas22": dict(atoms=["He", "H"], coords_ang=np.array([[0., 0., 0.], [0., 0., 0.74]]),
                      basis="3-21G", charge=1, spin=0, ncas=2, nelecas=2, bond_dim=200),
    "h2o_cas44": dict(atoms=["O", "H", "H"], coords_ang=np.array([
        [0., 0., 0.117], [0., 0.757, -0.469], [0., -0.757, -0.469]]),
        basis="6-31G", charge=0, spin=0, ncas=4, nelecas=4, bond_dim=400),
    # near-degenerate: stretched LiF (closing ionic/covalent gap) -> ill-conditioned response
    "lif_cas66_stretch": dict(atoms=["Li", "F"], coords_ang=np.array([[0., 0., 0.], [0., 0., 3.2]]),
                              basis="6-31G", charge=0, spin=0, ncas=6, nelecas=6, bond_dim=400),
}
RHS_SEQUENCE = [("grad", 0), ("grad", 1), ("nac", (0, 1))]
MODES = ["zero", "gmres-recycle", "hcc-inverse"]


def solve_one(obj, kind, key, *, tol, max_iter):
    t0 = time.perf_counter()
    if kind == "grad":
        kappa, ci, info, meta = obj.solve_mps(int(key), tol=tol, max_iter=max_iter)
    else:
        kappa, ci, info, meta = obj.solve_nac_mps(tuple(key), tol=tol, max_iter=max_iter)
    wall = time.perf_counter() - t0
    z = MPSKrylovVector(obj, kappa, ci, label=f"{kind}-{key}")
    return z, meta, wall, (key if kind == "grad" else key[0])


def run_one(key, *, tol=1e-8, max_iter=300, n_threads=4):
    s = SYSTEMS[key]
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=s["bond_dim"], n_sweeps=30, sweep_tol=1e-10, n_threads=n_threads)
    coords = s["coords_ang"] * ANG
    _mol, _mf, mc, _solver = fdv.build_sa_dmrg_casscf(
        s["atoms"], coords, basis=s["basis"], charge=s["charge"], spin=s["spin"],
        ncas=s["ncas"], nelecas=s["nelecas"], nroots=2, weights=[0.5, 0.5], solver_cfg=cfg)
    out = {"system": key, "ncas": s["ncas"], "bond_dim": s["bond_dim"], "modes": {}}
    for mode in MODES:
        try:
            obj = _make_mps_krylov_response(mc)
            obj._initial_guess = mode
            rows, tot = [], 0
            for kind, rkey in RHS_SEQUENCE:
                z, meta, wall, st = solve_one(obj, kind, rkey, tol=tol, max_iter=max_iter)
                cert = certify_response(obj, z, state=int(st), tol=1e-6,
                                        rhs_kind=("nac" if kind == "nac" else "grad"),
                                        nac_pair=(rkey if kind == "nac" else None),
                                        solver=f"global[{mode}]", wall_s=wall)
                ni = meta.get("niter")
                tot += (ni or 0)
                rows.append({"rhs": f"{kind}:{rkey}", "niter": ni, "wall_s": round(wall, 1),
                             "true_resid": cert.true_residual_relative, "conv": cert.converged})
            out["modes"][mode] = {"rows": rows, "total_iters": tot,
                                  "all_conv": all(r["conv"] for r in rows)}
            print(f"[{key}] {mode:14s}: total_iters={tot}  per-rhs={[r['niter'] for r in rows]}  "
                  f"conv={out['modes'][mode]['all_conv']}", flush=True)
        except Exception as e:
            out["modes"][mode] = {"error": type(e).__name__, "msg": str(e)[:200]}
            print(f"[{key}] {mode}: ERROR {e}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(_HERE / "data" / "precond_benchmark.json"))
    args = ap.parse_args()
    keys = [args.only] if args.only else list(SYSTEMS)
    results = []
    for k in keys:
        print(f"\n===== {k} =====", flush=True)
        try:
            results.append(run_one(k, n_threads=args.threads))
        except Exception as e:
            results.append({"system": k, "error": str(e), "tb": traceback.format_exc()[-1500:]})
            print(f"{k} FAILED: {e}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"benchmark": "precond_config", "results": results}, indent=1))
    print("\nWrote", args.out, flush=True)
