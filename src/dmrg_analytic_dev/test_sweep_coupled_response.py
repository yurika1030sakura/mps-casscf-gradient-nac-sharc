"""Validate the sweep-localized coupled Schur response against the global solver.

On HeH+ CAS(2,2) (DMRG = FCI) the sweep-Schur solver must reproduce the global
MPS-Krylov solution for both the orbital response (kappa) and the CI response
(per-state MPS), and its assembled solution must satisfy the true residual of
the global operator.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "sharc_interface"))

from analytic_cp_sharc import _make_mps_krylov_response
from test_step6c_mps_response_class import _setup_heh
from sweep_coupled_response import solve_state_sweep_schur


def _ci_diff(obj, a_list, b_list):
    """max_i || a_i - b_i ||."""
    d = 0.0
    for a, b in zip(a_list, b_list):
        diff = obj._combine_mps([(1.0, a), (-1.0, b)], tag=obj._new_tag("CMP"))
        d = max(d, float(np.sqrt(max(obj._mps_overlap(diff, diff), 0.0))))
    return d


def test_schur_matches_global(state=0):
    mc, _solver = _setup_heh()
    obj = _make_mps_krylov_response(mc)

    # global reference
    k_glob, ci_glob, info_glob, meta_glob = obj.solve_mps(
        state, tol=1.0e-9, max_iter=40,
    )
    # sweep-Schur solver (fresh object so caches are independent)
    obj2 = _make_mps_krylov_response(mc)
    k_schur, ci_schur, info_schur, meta = solve_state_sweep_schur(
        obj2, state, orb_tol=1.0e-9, ci_tol=1.0e-11,
    )

    diff_k = float(np.linalg.norm(
        obj.mc.pack_uniq_var(obj._canonical_kappa(k_glob))
        - obj.mc.pack_uniq_var(obj._canonical_kappa(k_schur))
    ))
    diff_ci = _ci_diff(obj2, ci_schur, [
        obj2._copy_mps(c, tag=obj2._new_tag(f"GLOB{i}"))
        for i, c in enumerate(ci_glob)
    ]) if False else None
    # CI comparison across the two objects: bring global CI into obj2 by overlap
    # (same active basis / same geometry), simplest is to recompute on obj2.
    k_glob2, ci_glob2, _, _ = obj2.solve_mps(state, tol=1.0e-9, max_iter=40)
    diff_ci = _ci_diff(obj2, ci_schur, ci_glob2)

    ok = (
        info_schur == 0
        and meta["true_residual_rel"] < 1.0e-6
        and diff_k < 1.0e-5
        and diff_ci < 1.0e-5
    )
    return {
        "name": "SCHUR_matches_global",
        "diff_kappa_vs_global": diff_k,
        "diff_ci_vs_global": diff_ci,
        "true_residual_rel": meta["true_residual_rel"],
        "orb_dim": meta["orb_dim"],
        "schur_applies": meta["schur_applies"],
        "info_schur": info_schur,
        "tol": 1.0e-5,
        "status": "pass" if ok else "fail",
    }


def main():
    try:
        result = test_schur_matches_global()
    except Exception as exc:
        result = {"name": "SCHUR_matches_global", "status": "fail",
                  "exception": type(exc).__name__, "message": str(exc),
                  "traceback_tail": traceback.format_exc()[-3000:]}
    print(json.dumps(result, indent=2))
    Path(__file__).with_suffix(".json").write_text(json.dumps({
        "milestone": "sweep_coupled_schur_vs_global",
        "system": "HeH+ / 3-21G / CAS(2,2) / SA(2)",
        "result": result,
    }, indent=2) + "\n")
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
