"""Generality stress test for the system-general certified derivative engine.

Runs ``certified_engine.compute_certified_derivatives`` on a deliberately diverse
set of systems -- different spin sectors (singlet / doublet / triplet), different
elements, different active-space sizes -- with ZERO per-system tuning, and
collects each system's PASS/WARN/FAIL verdict, certificate, spin purity, and
wall time.  The point is to show that the single general entry point either
returns a certified result or clearly flags the system, without hand-holding.

Usage:  python run_engine_stress_test.py --out data/engine_stress.json
"""
from __future__ import annotations

import argparse
import json
import os
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

import certified_engine as ce

ANG = 1.8897261246257702

# (label, atoms, coords_ang, basis, charge, spin, ncas, nelecas, grad_states, nac_pairs)
SYSTEMS = [
    ("HeH+_s",   ["He", "H"], [(0, 0, 0), (0, 0, 0.90)], "3-21G", 1, 0, 2, 2, [0], [(0, 1)]),
    ("LiH_s",    ["Li", "H"], [(0, 0, 0), (0, 0, 1.60)], "6-31G", 0, 0, 2, 2, [0], [(0, 1)]),
    ("H2O_s",    ["O", "H", "H"],
     [(0, 0, 0.117), (0, 0.757, -0.469), (0, -0.757, -0.469)], "6-31G", 0, 0, 4, 4, [0], [(0, 1)]),
    ("CH3_doublet", ["C", "H", "H", "H"],
     [(0, 0, 0), (0, 1.079, 0), (0.934, -0.539, 0), (-0.934, -0.539, 0)],
     "6-31G", 0, 1, 3, 3, [0], []),
    ("CH2_triplet", ["C", "H", "H"],
     [(0, 0, 0.110), (0, 0.867, -0.655), (0, -0.867, -0.655)], "6-31G", 0, 2, 2, 2, [0], []),
    ("ethylene_s", ["C", "C", "H", "H", "H", "H"],
     [(0, 0, 0.667), (0, 0, -0.667), (0, 0.923, 1.238), (0, -0.923, 1.238),
      (0, 0.923, -1.238), (0, -0.923, -1.238)], "6-31G", 0, 0, 2, 2, [0], [(0, 1)]),
]


def run_one(spec):
    (label, atoms, geom, basis, charge, spin, ncas, nelecas, gstates, npairs) = spec
    coords = np.array(geom) * ANG
    t0 = time.perf_counter()
    try:
        out = ce.compute_certified_derivatives(
            atoms, coords, basis=basis, charge=charge, spin=spin, ncas=ncas,
            nelecas=nelecas, nroots=max(2, 1 + max([0] + [b for _a, b in npairs] + gstates)),
            gradient_states=gstates, nac_pairs=npairs, max_bond_dim=128)
        rec = {"label": label, "spin": spin, "ncas": ncas, "nelecas": nelecas,
               "overall_health": out["overall_health"],
               "build_converged": out["build"]["converged"],
               "build_health": out["build"]["build_health"]["overall"],
               "s2_per_state": out["build"].get("s2_per_state"),
               "escalated": out["build"]["escalated"],
               "level_shift_warning": out["build"]["level_shift_warning"],
               "grad_certs": {str(s): out["gradients"][s]["certificate"].get("true_residual_relative")
                              for s in out.get("gradients", {})},
               "grad_health": {str(s): out["gradients"][s]["health"] for s in out.get("gradients", {})},
               "nac_certs": {k: v["certificate"].get("true_residual_relative")
                             for k, v in out.get("nacs", {}).items()},
               "fci_free_ok": (not out["build"]["beyond_fci"]) or (not out["fci_free"]["dense_bridge_used"]),
               "wall_s": time.perf_counter() - t0}
    except Exception as exc:  # noqa: BLE001
        rec = {"label": label, "status": "engine_raised",
               "exception": type(exc).__name__, "message": str(exc)[:300],
               "traceback_tail": traceback.format_exc()[-1500:],
               "wall_s": time.perf_counter() - t0}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of labels to run")
    ap.add_argument("--out", default=str(_HERE / "data" / "engine_stress.json"))
    args = ap.parse_args()
    systems = SYSTEMS
    if args.only:
        keep = set(args.only.split(","))
        systems = [s for s in SYSTEMS if s[0] in keep]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in systems:
        print(f"=== engine: {spec[0]} (spin={spec[5]}, CAS({spec[6]},{spec[7]})) ===", flush=True)
        rec = run_one(spec)
        records.append(rec)
        with open(out_path, "w") as f:
            json.dump({"records": records}, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        if "overall_health" in rec:
            print(f"  -> {rec['overall_health']}  build={rec['build_health']} "
                  f"s2={rec.get('s2_per_state')} {rec['wall_s']:.0f}s", flush=True)
        else:
            print(f"  -> RAISED {rec.get('exception')}: {rec.get('message')}", flush=True)
    npass = sum(1 for r in records if r.get("overall_health") == "PASS")
    print(f"\nSUMMARY: {npass}/{len(records)} PASS; "
          f"verdicts={[r.get('overall_health', 'RAISED') for r in records]}", flush=True)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
