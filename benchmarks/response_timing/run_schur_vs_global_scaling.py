"""Scaling study: sweep-localized coupled Schur response vs global MPS-Krylov.

For a sequence of growing active spaces, build SA(2)-DMRG-CASSCF and solve the
state-0 CP response two ways:
  * global  -- the existing mixed-vector MPS-Krylov GMRES (solve_mps);
  * schur   -- the sweep-localized coupled Schur solver (CI block by block2
               per-site sweeps, orbital-CI coupling closed in a dense Schur
               space).
Records wall time, CI-operation counts (global GMRES iterations vs Schur
applications), and the agreement between the two solutions, so the cost
structure can be compared as the active space grows.

Usage:  python run_schur_vs_global_scaling.py [--only h2o_cas44]
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
from sweep_coupled_response import solve_state_sweep_schur

ANG = 1.8897261246257702  # Bohr per Angstrom

SYSTEMS = {
    "heh_cas22": dict(
        atoms=["He", "H"],
        coords_ang=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        basis="3-21G", charge=1, spin=0, ncas=2, nelecas=2,
    ),
    "h2o_cas44": dict(
        atoms=["O", "H", "H"],
        coords_ang=np.array([
            [0.000, 0.000, 0.117],
            [0.000, 0.757, -0.469],
            [0.000, -0.757, -0.469],
        ]),
        basis="6-31G", charge=0, spin=0, ncas=4, nelecas=4,
    ),
    "n2_cas66": dict(
        atoms=["N", "N"],
        coords_ang=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]]),
        basis="6-31G", charge=0, spin=0, ncas=6, nelecas=6,
    ),
    "c2_cas88": dict(
        atoms=["C", "C"],
        coords_ang=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.25]]),
        basis="6-31G", charge=0, spin=0, ncas=8, nelecas=8,
    ),
}


def run_one(key, *, bond_dim=400, tol=1.0e-8):
    s = SYSTEMS[key]
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=bond_dim, n_sweeps=30, sweep_tol=1.0e-10)
    coords_bohr = s["coords_ang"] * ANG
    _mol, _mf, mc, _solver = fdv.build_sa_dmrg_casscf(
        s["atoms"], coords_bohr, basis=s["basis"], charge=s["charge"],
        spin=s["spin"], ncas=s["ncas"], nelecas=s["nelecas"], nroots=2,
        weights=[0.5, 0.5], solver_cfg=cfg,
    )

    # global MPS-Krylov
    obj_g = _make_mps_krylov_response(mc)
    t0 = time.perf_counter()
    k_g, ci_g, info_g, meta_g = obj_g.solve_mps(0, tol=tol, max_iter=60)
    wall_g = time.perf_counter() - t0

    # sweep-Schur
    obj_s = _make_mps_krylov_response(mc)
    t0 = time.perf_counter()
    k_s, ci_s, info_s, meta_s = solve_state_sweep_schur(
        obj_s, 0, orb_tol=tol, ci_tol=1.0e-10,
    )
    wall_s = time.perf_counter() - t0

    diff_k = float(np.linalg.norm(
        obj_g.mc.pack_uniq_var(obj_g._canonical_kappa(k_g))
        - obj_g.mc.pack_uniq_var(obj_g._canonical_kappa(k_s))
    ))
    # CI agreement: recompute global CI on obj_s for same-object overlap
    _kg2, ci_g2, _i2, _m2 = obj_s.solve_mps(0, tol=tol, max_iter=60)
    diff_ci = 0.0
    for a, b in zip(ci_s, ci_g2):
        d = obj_s._combine_mps([(1.0, a), (-1.0, b)], tag=obj_s._new_tag("CMP"))
        diff_ci = max(diff_ci, float(np.sqrt(max(obj_s._mps_overlap(d, d), 0.0))))

    return {
        "system": key,
        "ncas": s["ncas"], "nelecas": s["nelecas"], "basis": s["basis"],
        "bond_dim": bond_dim,
        "global_wall_s": wall_g,
        "schur_wall_s": wall_s,
        "speedup_global_over_schur": wall_g / max(wall_s, 1e-30),
        "global_niter": meta_g.get("niter"),
        "schur_applies": meta_s.get("schur_applies"),
        "schur_orb_dim": meta_s.get("orb_dim"),
        "global_residual": meta_g.get("residual"),
        "schur_true_residual_rel": meta_s.get("true_residual_rel"),
        "diff_kappa_global_vs_schur": diff_k,
        "diff_ci_global_vs_schur": diff_ci,
        "agree": bool(diff_k < 1e-5 and diff_ci < 1e-5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single system key")
    ap.add_argument("--bond-dim", type=int, default=400)
    args = ap.parse_args()
    keys = [args.only] if args.only else list(SYSTEMS)
    results = []
    for key in keys:
        print(f"=== {key} ===", flush=True)
        try:
            r = run_one(key, bond_dim=args.bond_dim)
            print(json.dumps(r, indent=2), flush=True)
        except Exception as exc:
            r = {"system": key, "status": "error",
                 "exception": type(exc).__name__, "message": str(exc),
                 "traceback_tail": traceback.format_exc()[-2500:]}
            print(f"  ERROR {key}: {exc}", flush=True)
        results.append(r)
    out = _HERE / "data" / "schur_vs_global_scaling.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": "sweep_schur_vs_global_mps_krylov_scaling",
        "results": results,
    }, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
