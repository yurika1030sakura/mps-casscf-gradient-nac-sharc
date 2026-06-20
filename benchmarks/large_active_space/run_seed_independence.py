"""Seed-independence test for SA-DMRG-CASSCF references.

A valid, converged calculation must give the SAME energy for ANY random initial
MPS.  Run-to-run scatter is a symptom of under-convergence, not something to be
hidden by fixing a seed.  This script runs the reference SA-DMRG-CASSCF of a
polyene with several independent random seeds at one or more convergence levels
and reports the energy spread (max - min over seeds).  The goal is to find the
convergence settings (bond-dimension schedule, sweep tolerance, CASSCF gradient
tolerance) for which the spread is below a target (e.g. 1e-6 Eh), i.e. the
result is genuinely seed-independent.

The seed is controlled globally via block2.Random.rand_seed BEFORE each build;
the solver itself keeps dmrg_seed=None so it does not re-seed.

Usage:
  python run_seed_independence.py --ncarbon 16 --schedule current --seeds 1 2 3
  python run_seed_independence.py --ncarbon 16 --schedule tight   --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DEV = _HERE.parents[1] / "src" / "dmrg_analytic_dev"
SHARC = _HERE.parents[1] / "sharc_interface"
for p in (str(DEV), str(SHARC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import block2
from run_cas_directional_fd import build_progressive
from run_polyene_beyond_fci import polyene_geometry, det_dim

ANG = 1.8897261246257702

# (M, sweep_tol, conv_tol_grad, max_macro)
SCHEDULES = {
    "current": [
        (256, 1.0e-7, 1.0e-4, 30),
        (512, 1.0e-8, 3.0e-5, 40),
        (800, 1.0e-9, 1.0e-5, 60),
    ],
    "tight": [
        (256, 1.0e-8, 1.0e-4, 40),
        (512, 1.0e-10, 1.0e-5, 60),
        (800, 1.0e-12, 1.0e-6, 150),
    ],
}


def run(ncarbon, seeds, schedule_label, threads, stack_mem_mb, out_path):
    sched = SCHEDULES[schedule_label]
    atoms = polyene_geometry(ncarbon)
    symbols = [a[0] for a in atoms]
    coords = np.array([a[1] for a in atoms]) * ANG  # bohr
    ddim = det_dim(ncarbon, (ncarbon // 2, ncarbon - ncarbon // 2))

    out = {"system": f"polyene_C{ncarbon}", "ncas": ncarbon, "nelecas": ncarbon,
           "det_dim": ddim, "schedule": schedule_label, "m_schedule": sched,
           "seeds": list(seeds), "per_seed": []}

    def flush():
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
            f.flush(); os.fsync(f.fileno())

    for seed in seeds:
        block2.Random.rand_seed(int(seed))   # control the random initial MPS
        t0 = time.perf_counter()
        mol, mc, solver, blog = build_progressive(
            symbols, coords, "sto-3g", ncarbon, ncarbon, m_schedule=sched,
            threads=threads, stack_mem_mb=stack_mem_mb)
        e = [float(x) for x in mc.e_states]
        out["per_seed"].append({
            "seed": int(seed), "converged": bool(mc.converged),
            "e_states": e, "e0": e[0], "gap_Eh": float(e[1] - e[0]),
            "wall_s": time.perf_counter() - t0,
            "stage_e0": [s["e_states"][0] for s in blog["stages"]],
        })
        flush()
        print(f"  seed={seed} conv={mc.converged} E0={e[0]:.8f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    e0s = [r["e0"] for r in out["per_seed"]]
    out["e0_min"] = min(e0s)
    out["e0_max"] = max(e0s)
    out["e0_spread_Eh"] = max(e0s) - min(e0s)
    out["seed_independent_1e6"] = bool((max(e0s) - min(e0s)) < 1.0e-6)
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, default=16)
    ap.add_argument("--schedule", choices=list(SCHEDULES), default="current")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--stack-mem-mb", type=int, default=8000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(_HERE / "data" / f"seedindep_c{args.ncarbon}_{args.schedule}.json")
    print(f"=== seed-independence C{args.ncarbon} schedule={args.schedule} "
          f"seeds={args.seeds} ===", flush=True)
    r = run(args.ncarbon, args.seeds, args.schedule, args.threads,
            args.stack_mem_mb, out)
    print(f"E0 spread over seeds = {r['e0_spread_Eh']:.2e} Eh "
          f"(seed-independent<1e-6: {r['seed_independent_1e6']})", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
