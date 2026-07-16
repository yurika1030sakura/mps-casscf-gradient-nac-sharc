"""Repeated-call benchmark: ethylene CAS(2,2) torsion path.

Walks ethylene through the C=C twist (the classic S0/S1 approach), and at each
geometry computes the two state gradients and their NAC with the certified
response solver -- once from a zero guess and once with same-geometry recycling.
For every step it records the response wall time and iteration count for both
modes (the warm-start speedup), the certified true residual, the state energies
and gap, and the root continuity to the previous step (the cross-geometry active
overlap singular values and the overlap matrix).  This is a repeated-call /
path study, not a propagated trajectory; no long-time dynamics is claimed.

Usage:  python run_repeated_call_trajectory.py --n-steps 46 --theta-max 90
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
from auto_response import compute_all_responses_certified
from overlap_fci_reference import overlap_fci
from cross_geometry_overlap import cross_geometry_active_overlap

ANG = 1.8897261246257702


def ethylene_geometry(theta_deg, *, rcc=1.33, rch=1.08, accd=121.5):
    """Twisted ethylene: one CH2 rotated by theta about the C=C (x) axis."""
    th = np.deg2rad(theta_deg)
    a = np.deg2rad(accd)
    cx = rcc / 2.0
    hy, hx = rch * np.sin(a), rch * np.cos(a)
    c1 = np.array([-cx, 0.0, 0.0])
    c2 = np.array([cx, 0.0, 0.0])
    # C1 hydrogens fixed in the xy-plane
    h1a = c1 + np.array([hx, hy, 0.0])
    h1b = c1 + np.array([hx, -hy, 0.0])
    # C2 hydrogens twisted by theta about x
    h2a = c2 + np.array([-hx, hy * np.cos(th), hy * np.sin(th)])
    h2b = c2 + np.array([-hx, -hy * np.cos(th), -hy * np.sin(th)])
    atoms = ["C", "C", "H", "H", "H", "H"]
    coords_ang = np.array([c1, c2, h1a, h1b, h2a, h2b])
    return atoms, coords_ang


def _norm(arr):
    return float(np.linalg.norm(np.asarray(arr)))


def run(n_steps=46, theta_max=90.0, basis="6-31G", bond_dim=200,
        n_threads=1, stack_mem_mb=None):
    thetas = np.linspace(0.0, theta_max, int(n_steps))
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=bond_dim, n_sweeps=30, sweep_tol=1.0e-10,
               n_threads=int(n_threads))
    if stack_mem_mb is not None:
        cfg["stack_mem_mb"] = int(stack_mem_mb)

    steps = []
    prev = None
    tot_zero = tot_rec = 0.0
    for th in thetas:
        atoms, coords_ang = ethylene_geometry(th)
        coords_bohr = coords_ang * ANG
        try:
            _mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
                atoms, coords_bohr, basis=basis, charge=0, spin=0,
                ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5], solver_cfg=cfg,
            )
            ncas, ncore = mc.ncas, mc.ncore
            nelec = (1, 1)
            e = list(np.asarray(solver.e_states, dtype=float).ravel())

            # zero-guess vs recycled response over the same-geometry RHS set
            t0 = time.perf_counter()
            res0 = compute_all_responses_certified(
                _make_mps_krylov_response(mc), gradient_states=[0, 1],
                nac_pairs=[(0, 1)], tol=1.0e-8, recycle=False)
            wall_zero = time.perf_counter() - t0
            t0 = time.perf_counter()
            res1 = compute_all_responses_certified(
                _make_mps_krylov_response(mc), gradient_states=[0, 1],
                nac_pairs=[(0, 1)], tol=1.0e-8, recycle=True)
            wall_rec = time.perf_counter() - t0
            tot_zero += wall_zero
            tot_rec += wall_rec

            iters_zero = {f"{k[0]}:{k[1]}": c.extra.get("niter")
                          for k, (_z, c) in res0.items()}
            iters_rec = {f"{k[0]}:{k[1]}": c.extra.get("niter")
                         for k, (_z, c) in res1.items()}
            max_resid = max(c.true_residual_relative for _z, c in res1.values())
            converged = all(c.converged for _z, c in res1.values())

            ci = fdv.mps_ci_list(solver, ncas, nelec, 2)
            mo = mc.mo_coeff

            cont = None
            if prev is not None:
                s = cross_geometry_active_overlap(prev["mol"], _mol, prev["mo"],
                                                  mo, ncore, ncas)
                svals = np.linalg.svd(s, compute_uv=False)
                O = np.array([[overlap_fci(prev["ci"][i], ci[j], s, ncas, nelec)
                               for j in range(2)] for i in range(2)])
                cont = {
                    "active_subspace_sigma_min": float(np.min(svals)),
                    "root_overlap_diag": [abs(float(O[0, 0])), abs(float(O[1, 1]))],
                    "root_overlap_offdiag": [abs(float(O[0, 1])), abs(float(O[1, 0]))],
                }

            steps.append({
                "theta_deg": float(th), "e_states": e,
                "gap_Eh": float(abs(e[1] - e[0])),
                "grad0_norm": _norm(res1[("grad", 0)][0].kappa),
                "nac01_certified_residual": max_resid,
                "all_converged": bool(converged),
                "wall_zero_s": wall_zero, "wall_recycled_s": wall_rec,
                "iters_zero": iters_zero, "iters_recycled": iters_rec,
                "continuity": cont,
            })
            prev = {"mol": _mol, "mo": mo, "ci": ci}
            print(f"theta={th:5.1f}  gap={abs(e[1]-e[0]):.4f}  "
                  f"t_zero={wall_zero:.1f}s t_rec={wall_rec:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            steps.append({"theta_deg": float(th), "status": "error",
                          "message": str(exc),
                          "traceback_tail": traceback.format_exc()[-1500:]})
            print(f"  ERROR theta={th}: {exc}", flush=True)

    return {
        "benchmark": "ethylene_cas22_torsion_repeated_call",
        "basis": basis, "ncas": 2, "nelecas": 2, "bond_dim": bond_dim,
        "n_steps": int(n_steps), "theta_max": float(theta_max),
        "total_response_wall_zero_s": tot_zero,
        "total_response_wall_recycled_s": tot_rec,
        "recycled_speedup": tot_zero / max(tot_rec, 1e-30),
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=46)
    ap.add_argument("--theta-max", type=float, default=90.0)
    ap.add_argument("--basis", default="6-31G")
    ap.add_argument("--bond-dim", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stack-mem-mb", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = run(n_steps=args.n_steps, theta_max=args.theta_max, basis=args.basis,
            bond_dim=args.bond_dim, n_threads=args.threads,
            stack_mem_mb=args.stack_mem_mb)
    out = Path(args.out) if args.out else _HERE / "data" / "repeated_call_trajectory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"Wrote {out}  (recycled speedup {r['recycled_speedup']:.2f}x)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
