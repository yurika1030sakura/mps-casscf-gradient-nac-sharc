"""Test: high-accuracy FD validation of the determinant cross-geometry NAC.

Pushes the 2-point determinant NAC FD (~1e-4) toward <1e-5 via 5-point /
Richardson FD + closed-shell core-overlap factor, and reports the gap-weighted
coupling error.  Runs on HeH+ CAS(2,2)/3-21G and ethylene CAS(2,2)/6-31G.

Target:  analytic-vs-FD < 1e-5  OR  gap-weighted < 1e-6 (for the best estimator).
Emits a golden JSON next to this file.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import nac_validation as nv
from analytic_cp_sharc import compute_grad_nac_analytic_cp


def _pick_component(case, bra=0, ket=1):
    """Pick the (atom, axis) with the largest analytic |d| so we test a
    component that actually couples (a near-zero component is uninformative)."""
    coords0 = np.asarray(case["coords_bohr"], dtype=float)
    mol0, _mf0, mc0, _s0 = nv.build_sa_dmrg_casscf(
        case["atoms"], coords0, basis=case["basis"], charge=case["charge"],
        spin=case["spin"], ncas=case["ncas"], nelecas=case["nelecas"],
        nroots=case["nroots"], weights=case["weights"],
        solver_cfg=case["solver_cfg"],
    )
    res = compute_grad_nac_analytic_cp(
        mc0, gradient_states=[], nac_pairs=[(bra, ket)],
        backend="mps-krylov", tol=1.0e-8, max_iter=200,
    )
    d = np.abs(np.asarray(res["nac"][(bra, ket)], dtype=float))
    atom, axis = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(atom), int(axis), float(d[atom, axis])


def _run(label, case, h_bohr):
    atom, axis, dmag = _pick_component(case)
    res = nv.run_case(case, bra=0, ket=1, atom=atom, axis=axis,
                      h_bohr=h_bohr, label=label)
    res["picked_d_magnitude"] = dmag
    # target check: best estimator < 1e-5  OR  its gap-weighted err < 1e-6
    best = res["best_estimator"]
    best_err = res["errors"][best]
    best_gap_err = res["gap_weighted_errors"][best]
    res["target_met"] = bool(best_err < 1.0e-5 or best_gap_err < 1.0e-6)
    res["status"] = "pass" if res["target_met"] else "fail"
    return res


def main():
    cases = [
        ("HeH+_CAS22_321G", nv.heh_case(), 1.0e-3),
        ("ethylene_CAS22_631G", nv.ethylene_case(), 1.0e-3),
    ]
    results = []
    for label, case, h in cases:
        try:
            r = _run(label, case, h)
        except Exception as exc:
            r = {
                "label": label, "status": "fail",
                "exception": type(exc).__name__, "message": str(exc),
                "traceback_tail": traceback.format_exc()[-3000:],
            }
        results.append(r)
        # readable table
        print(f"\n=== {label} ===", flush=True)
        if r.get("status") == "fail" and "errors" not in r:
            print(f"  FAILED: {r.get('exception')}: {r.get('message')}",
                  flush=True)
            continue
        print(f"  component (atom,axis)=({r['atom']},{r['axis']})  "
              f"h={r['h_bohr']:.1e}  ncore={r['ncore']}", flush=True)
        print(f"  analytic d_01      = {r['analytic_d']:+.8e}", flush=True)
        print(f"  gap (E1-E0)        = {r['gap_ref']:+.8e}", flush=True)
        print(f"  analytic h_01(gapW)= {r['analytic_gapweighted']:+.8e}",
              flush=True)
        print("  {:<18s} {:>16s} {:>14s} {:>14s}".format(
            "estimator", "value", "|err|", "|gapW err|"), flush=True)
        for name in ("two_point", "five_point", "richardson",
                     "core_two_point", "core_five_point", "core_richardson"):
            print("  {:<18s} {:>16.8e} {:>14.3e} {:>14.3e}".format(
                name, r["estimators"][name], r["errors"][name],
                r["gap_weighted_errors"][name]), flush=True)
        print(f"  best estimator     = {r['best_estimator']}  "
              f"(|err|={r['best_error']:.3e})", flush=True)
        print(f"  core-overlap det   = "
              f"{ {k: round(v,8) for k,v in r['det_core'].items()} }",
              flush=True)
        print(f"  det_core max dev from 1 = {r['det_core_max_dev_from_1']:.3e}",
              flush=True)
        print(f"  block-leakage max  = {r['block_leakage_max']:.3e}", flush=True)
        print(f"  -> target_met = {r['target_met']}  ({r['status']})",
              flush=True)

    # strip nothing (already JSON-safe); write golden
    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps({
        "milestone": "NAC_FD_high_accuracy",
        "purpose": (
            "Push determinant cross-geometry NAC finite difference from ~1e-4 "
            "toward <1e-5 via 5-point/Richardson FD + closed-shell core-overlap "
            "factor + gap-weighting, vs analytic CP-CASSCF derivative coupling."
        ),
        "systems": ["HeH+ /3-21G/CAS(2,2)/SA(2)",
                    "ethylene /6-31G/CAS(2,2) pi-pi*/SA(2)"],
        "target": "best-estimator |analytic-FD| < 1e-5  OR  gap-weighted < 1e-6",
        "results": results,
    }, indent=2) + "\n")
    print(f"\nWrote {out_path}", flush=True)
    return 0 if all(r.get("status") == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
