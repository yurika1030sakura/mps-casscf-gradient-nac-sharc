"""Validate the SHARC-facing MPS-Krylov analytic CP path.

This checks that ``analytic_cp_sharc.compute_grad_nac_analytic_cp`` can use
``dmrg-response-mode=mps-krylov`` semantics and reproduce the established
projected-CI path for a small exact active space.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SHARC_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SHARC_ROOT))
PUBLIC_SHARC_ROOT = ROOT.parents[1] / "sharc_interface"
if PUBLIC_SHARC_ROOT.exists():
    sys.path.insert(0, str(PUBLIC_SHARC_ROOT))

from analytic_cp_sharc import compute_grad_nac_analytic_cp  # noqa: E402
from test_step6c_mps_response_class import _setup_heh  # noqa: E402


def test_s1_mps_krylov_grad_nac_matches_projected_path():
    mc, _solver = _setup_heh()
    ref = compute_grad_nac_analytic_cp(
        mc, gradient_states=[0], nac_pairs=[(0, 1)],
        backend="projected-ci", tol=1.0e-8, max_iter=100,
    )
    got = compute_grad_nac_analytic_cp(
        mc, gradient_states=[0], nac_pairs=[(0, 1)],
        backend="mps-krylov", tol=1.0e-8, max_iter=20,
    )
    grad_diff = float(np.linalg.norm(ref["grad"][0] - got["grad"][0]))
    nac_diff = float(np.linalg.norm(ref["nac"][(0, 1)] - got["nac"][(0, 1)]))
    ok = grad_diff < 1.0e-8 and nac_diff < 1.0e-8
    return {
        "name": "S1_mps_krylov_sharc_grad_nac_matches_projected_path",
        "grad_diff": grad_diff,
        "nac_diff": nac_diff,
        "tol": 1.0e-8,
        "status": "pass" if ok else "fail",
    }


def main():
    cases = [test_s1_mps_krylov_grad_nac_matches_projected_path]
    results = []
    for case in cases:
        try:
            result = case()
        except Exception as exc:
            result = {
                "name": case.__name__,
                "status": "fail",
                "exception": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc()[-2000:],
            }
        results.append(result)
        print(f"  {result['name']}: {result['status']}")

    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps({
        "milestone": "MPS_Krylov_SHARC_interface",
        "purpose": (
            "Validate SHARC-facing gradient/NAC assembly through the MPS-Krylov "
            "response backend against the projected-CI path."
        ),
        "results": results,
    }, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
