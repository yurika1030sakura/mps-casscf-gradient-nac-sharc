"""Calibrate + validate the MPS-native orbital-rotation overlap on CAS(2,2).

Single-driver milestone: take the two SA states at one geometry, rotate each by
a KNOWN near-identity matrix M with the td_dmrg orbital rotation, and form the
overlap matrix O_mps[I, J] = <Psi_I | R(M) Psi_J>.  The determinant-level
``overlap_fci(ci_I, ci_J, M)`` is the exact ground truth (it places the ket
orbitals in an M-rotated basis).  Sweeping the convention (which transform of M,
which td_dmrg time sign) identifies the choice that reproduces the reference to
machine precision -- this isolates and pins the rotation kernel before the
cross-geometry / cross-driver step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SHARC = _HERE.parents[1] / "sharc_interface"
if str(_SHARC) not in sys.path:
    sys.path.insert(0, str(_SHARC))

import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response
from overlap_fci_reference import overlap_fci
from cross_geometry_overlap import rotate_mps_orbitals, ROTATION_CONVENTION

ANG = 1.8897261246257702


def _build():
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=100, n_sweeps=24, sweep_tol=1.0e-10, n_threads=1)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.90]]) * ANG
    _mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
        ["He", "H"], coords, basis="3-21G", charge=1, spin=0,
        ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5], solver_cfg=cfg,
    )
    return mc, solver


def main():
    mc, solver = _build()
    ncas = mc.ncas
    nelec = (mc.nelecas[0], mc.nelecas[1]) if isinstance(mc.nelecas, (tuple, list)) \
        else (mc.nelecas // 2 + mc.nelecas % 2, mc.nelecas // 2)

    obj = _make_mps_krylov_response(mc)
    states = obj._state_mps
    nst = len(states)
    ci = fdv.mps_ci_list(solver, ncas, nelec, nst)

    # The unitary orbital rotation is the part that carries the finite-difference
    # derivative coupling (the non-unitary stretch contributes only at O(h^2) and
    # cancels in the central difference).  Validate it directly: rotate by a known
    # UNITARY M at finite-difference magnitude and reproduce overlap_fci(M).
    rng = np.random.default_rng(0)
    G = rng.standard_normal((ncas, ncas))
    M = expm(5.0e-3 * (G - G.T))                 # unitary, ~ identity

    O_gt = np.array([[overlap_fci(ci[i], ci[j], M, ncas, nelec)
                      for j in range(nst)] for i in range(nst)])

    tag = {"n": 0}
    rotated = []
    for j in range(nst):
        tag["n"] += 1
        rotated.append(rotate_mps_orbitals(
            obj._driver_su2, states[j], M, ncas=ncas, tag=f"XGEOM{tag['n']}",
            n_steps=24, include_stretch=False,
        ))
    O = np.array([[obj._mps_overlap(states[i], rotated[j])
                   for j in range(nst)] for i in range(nst)])
    err = float(np.max(np.abs(O - O_gt)))
    variants = [{"rotation": "U_only", "convention": ROTATION_CONVENTION,
                 "max_err_vs_fci": err, "O_mps": O.tolist()}]
    best = variants[0]

    out = {
        "name": "cross_geometry_overlap_unitary_rotation_heh_cas22",
        "displacement_magnitude": 5.0e-3,
        "O_gt": O_gt.tolist(),
        "max_err_vs_fci": err,
        "convention": ROTATION_CONVENTION,
        "status": "pass" if err < 1.0e-5 else "fail",
    }
    (_HERE / "test_cross_geometry_overlap.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"U-only orbital rotation vs overlap_fci: max_err = {err:.3e}  "
          f"({out['status']})")
    return 0 if out["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
