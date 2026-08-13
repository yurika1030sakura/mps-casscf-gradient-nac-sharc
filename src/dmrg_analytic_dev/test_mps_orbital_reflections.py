"""Validate det=-1 MPS orbital gauges beyond the two-orbital toy case."""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "sharc_interface"))

import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response
from cross_geometry_overlap import rotate_mps_orbitals
from overlap_fci_reference import overlap_fci

ANG = 1.8897261246257702


def _phase_gauge_error(actual, reference):
    nr, nc = actual.shape
    best = np.inf
    for ltail in product((-1.0, 1.0), repeat=nr - 1):
        left = np.array((1.0,) + ltail)
        for rt in product((-1.0, 1.0), repeat=nc):
            right = np.array(rt)
            best = min(best, float(np.max(np.abs(
                left[:, None] * actual * right[None, :] - reference
            ))))
    return best


def main():
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=100, n_sweeps=24, sweep_tol=1.0e-10, n_threads=1,
               mps_native_rdms=True, skip_kernel_fci_conversion=True)
    coords = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.9],
        [0.0, 0.0, 1.8], [0.0, 0.0, 2.7],
    ]) * ANG
    _mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
        ["H"] * 4, coords, basis="sto-3g", charge=0, spin=0,
        ncas=4, nelecas=4, nroots=2, weights=[0.5, 0.5], solver_cfg=cfg,
    )
    obj = _make_mps_krylov_response(mc)
    states = obj._state_mps
    ci = fdv.mps_ci_list(solver, 4, (2, 2), 2)

    transforms = []
    for p in range(4):
        D = np.eye(4)
        D[p, p] = -1.0
        transforms.append((f"sign_{p}", D))
    swap = np.eye(4)
    swap[[0, 1]] = swap[[1, 0]]
    transforms.append(("odd_swap_01", swap))

    errors = {}
    for icase, (name, matrix) in enumerate(transforms):
        gt = np.array([
            [overlap_fci(ci[i], ci[j], matrix, 4, (2, 2))
             for j in range(2)] for i in range(2)
        ])
        rotated = [
            rotate_mps_orbitals(
                obj._driver_su2, states[j], matrix, ncas=4,
                tag=f"REFL-{icase}-{j}", n_steps=48,
            ) for j in range(2)
        ]
        got = np.array([
            [obj._mps_overlap(states[i], rotated[j]) for j in range(2)]
            for i in range(2)
        ])
        errors[name] = _phase_gauge_error(got, gt)

    worst = max(errors.values())
    print("reflection phase-gauge errors:", errors)
    print("MPS ORBITAL REFLECTION TEST: PASS" if worst < 2.0e-5
          else "MPS ORBITAL REFLECTION TEST: FAIL")
    return 0 if worst < 2.0e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
