"""Certify both response routes on HeH+ CAS(2,2) and serialize the certificate.

Checks that ``certify_response`` reproduces the solvers' own accept decision:
the recomputed true residual is below tolerance, the active-space block does not
leak into the reference roots, and the reported bond dimension / orbital
dimension are sane.  Exercised for the global MPS-Krylov solver and the
sweep-localized Schur solver, which solve the same system.
"""

from __future__ import annotations

import json
import sys
import time
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
from sweep_coupled_response import solve_state_sweep_schur
from cp_dmrg_response_mps_krylov import MPSKrylovVector
from certified_response import certify_response

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
    tol = 1.0e-6

    # global MPS-Krylov
    obj_g = _make_mps_krylov_response(mc)
    t0 = time.perf_counter()
    k_g, ci_g, info_g, _meta_g = obj_g.solve_mps(0, tol=1.0e-8, max_iter=60)
    wall_g = time.perf_counter() - t0
    z_g = MPSKrylovVector(obj_g, k_g, ci_g, label="CERT-GLOBAL")
    cert_g = certify_response(obj_g, z_g, state=0, tol=tol, solver="global_mps_krylov",
                              wall_s=wall_g)

    # sweep-localized Schur
    obj_s = _make_mps_krylov_response(mc)
    t0 = time.perf_counter()
    k_s, c_s, info_s, _meta_s = solve_state_sweep_schur(
        obj_s, 0, orb_tol=1.0e-8, ci_tol=1.0e-10,
    )
    wall_s = time.perf_counter() - t0
    z_s = MPSKrylovVector(obj_s, k_s, c_s, label="CERT-SCHUR")
    cert_s = certify_response(obj_s, z_s, state=0, tol=tol, solver="sweep_schur",
                              wall_s=wall_s)

    out = {
        "name": "response_certificate_heh_cas22",
        "tol": tol,
        "global": cert_g.to_dict(),
        "schur": cert_s.to_dict(),
    }

    checks = []
    for tag, cert in (("global", cert_g), ("schur", cert_s)):
        checks.append(cert.converged)
        checks.append(cert.true_residual_relative < tol)
        checks.append(cert.response_bond_dim >= 1)
        checks.append(cert.orbital_dim >= 1)
        checks.append(cert.root_projector_leakage is None
                      or cert.root_projector_leakage < 1.0e-6)
    out["status"] = "pass" if all(checks) else "fail"

    (_HERE / "test_response_certificate.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
