"""Timing benchmark for projected-CI and MPS-Krylov response paths.

This public benchmark uses a small exact active space so both response paths
can be compared to the same numerical answer.  It records wall times for the
gradient and NAC assembly through the SHARC-facing analytic CP helper.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
SHARC = REPO / "sharc_interface"
for path in (SRC, SRC / "dmrg_analytic_dev", SHARC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analytic_cp_sharc import compute_grad_nac_analytic_cp  # noqa: E402
from test_step6c_mps_response_class import _setup_heh  # noqa: E402


def timed_call(label, fn):
    t0 = time.perf_counter()
    out = fn()
    return label, out, time.perf_counter() - t0


def main():
    mc, _solver = _setup_heh()
    cases = []
    for mode in ("projected-ci", "mps-krylov"):
        mc.fcisolver.response_initial_guess = "zero"
        cases.append(timed_call(
            f"{mode}:grad0+nac01",
            lambda mode=mode: compute_grad_nac_analytic_cp(
                mc,
                gradient_states=[0],
                nac_pairs=[(0, 1)],
                backend=mode,
                tol=1.0e-8,
                max_iter=100 if mode == "projected-ci" else 20,
            ),
        ))
    mc.fcisolver.response_initial_guess = "hcc-inverse"
    mc.fcisolver.response_initial_guess_sweeps = 6
    mc.fcisolver.response_initial_guess_tol = 1.0e-10
    cases.append(timed_call(
        "mps-krylov+hcc-inverse:grad0+nac01",
        lambda: compute_grad_nac_analytic_cp(
            mc,
            gradient_states=[0],
            nac_pairs=[(0, 1)],
            backend="mps-krylov",
            tol=1.0e-8,
            max_iter=20,
        ),
    ))

    ref = cases[0][1]
    rows = []
    for label, out, seconds in cases:
        rows.append({
            "label": label,
            "wall_time_s": seconds,
            "grad0_norm": float(np.linalg.norm(out["grad"][0])),
            "nac01_norm": float(np.linalg.norm(out["nac"][(0, 1)])),
            "grad0_diff_vs_projected": float(np.linalg.norm(
                out["grad"][0] - ref["grad"][0],
            )),
            "nac01_diff_vs_projected": float(np.linalg.norm(
                out["nac"][(0, 1)] - ref["nac"][(0, 1)],
            )),
        })

    result = {
        "benchmark": "mps_krylov_response_timing",
        "system": "HeH+",
        "basis": "3-21G",
        "active_space": "CAS(2,2)",
        "purpose": (
            "Compare SHARC-facing projected-CI and MPS-Krylov analytic CP "
            "response wall times on an exact small active-space reference."
        ),
        "rows": rows,
    }
    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
