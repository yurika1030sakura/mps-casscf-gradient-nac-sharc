"""Batch certified response on HeH+ CAS(2,2): all RHS at one geometry.

Exercises ``compute_all_responses_certified`` for the gradients of both states
and their NAC on a single object with recycling, and checks that every accepted
solution is certified and that the recycled later RHS need no extra iterations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SHARC = _HERE.parents[1] / "sharc_interface"
if str(_SHARC) not in sys.path:
    sys.path.insert(0, str(_SHARC))

import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response
from auto_response import compute_all_responses_certified

ANG = 1.8897261246257702


def main():
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=200, n_sweeps=30, sweep_tol=1.0e-10, n_threads=1)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]) * ANG
    _mol, _mf, mc, _solver = fdv.build_sa_dmrg_casscf(
        ["He", "H"], coords, basis="3-21G", charge=1, spin=0,
        ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5], solver_cfg=cfg,
    )
    obj = _make_mps_krylov_response(mc)
    res = compute_all_responses_certified(
        obj, gradient_states=[0, 1], nac_pairs=[(0, 1)], tol=1.0e-8, recycle=True,
    )

    rows = []
    for key, (_z, cert) in res.items():
        rows.append({
            "rhs": f"{key[0]}:{key[1]}",
            "niter": cert.extra.get("niter"),
            "true_residual_relative": cert.true_residual_relative,
            "converged": cert.converged,
        })

    out = {"name": "batch_response_heh_cas22", "rhs": rows}
    # all certified, and the later (recycled) RHS converge without extra iters
    later_iters = [r["niter"] for r in rows[1:] if r["niter"] is not None]
    out["status"] = "pass" if (
        all(r["converged"] for r in rows)
        and all(n == 0 for n in later_iters)
    ) else "fail"

    (_HERE / "test_batch_response.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
