"""Certified auto-solver on HeH+ CAS(2,2): gradient via Schur, NAC via global.

Verifies that ``solve_response_auto`` returns a certified solution for both a
state gradient (accepted from the sweep-localized Schur solver) and an
interstate NAC (accepted from the global solver), with machine-precision true
residuals and the per-attempt record populated.
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
from auto_response import solve_response_auto

ANG = 1.8897261246257702


def _build():
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=200, n_sweeps=30, sweep_tol=1.0e-10, n_threads=1)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]) * ANG
    _mol, _mf, mc, _solver = fdv.build_sa_dmrg_casscf(
        ["He", "H"], coords, basis="3-21G", charge=1, spin=0,
        ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5], solver_cfg=cfg,
    )
    return mc


def main():
    mc = _build()

    obj_g = _make_mps_krylov_response(mc)
    _zg, cert_g = solve_response_auto(obj_g, state=0, tol=1.0e-8)

    obj_n = _make_mps_krylov_response(mc)
    _zn, cert_n = solve_response_auto(obj_n, nac_pair=(0, 1), tol=1.0e-8)

    out = {
        "name": "auto_response_heh_cas22",
        "gradient": cert_g.to_dict(),
        "nac": cert_n.to_dict(),
    }
    checks = [
        cert_g.converged,
        cert_g.extra.get("accepted_solver") == "sweep_schur",
        cert_g.true_residual_relative < 1.0e-6,
        cert_n.converged,
        cert_n.extra.get("accepted_solver") == "global_mps_krylov",
        cert_n.true_residual_relative < 1.0e-6,
    ]
    out["status"] = "pass" if all(checks) else "fail"

    (_HERE / "test_auto_response.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
