"""Validate cross-driver MPS transport: the displacement survives intact.

A finite-difference-displaced state is built by a separate driver/frame.  The
diagnostic ``diag_cross_geometry_transport`` showed that rebuilding it from a CI
read-out drops the O(1e-5) displacement components (the cross-geometry overlap
off-diagonal collapses to zero).  Here the displaced state is instead
transported intact with ``load_foreign_mps`` and overlapped, in the reference
basis (s = I), against the reference states.  Success means the off-diagonal
matches the determinant-level ``overlap_fci`` reference (~2e-5) rather than
collapsing to zero -- i.e. the displacement is preserved.
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
from overlap_fci_reference import overlap_fci
from cross_geometry_overlap import load_foreign_mps

ANG = 1.8897261246257702


def _build(z_bohr):
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=100, n_sweeps=24, sweep_tol=1.0e-10, n_threads=1)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, z_bohr]])
    mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
        ["He", "H"], coords, basis="3-21G", charge=1, spin=0,
        ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5], solver_cfg=cfg,
    )
    return mol, mc, solver


def main():
    h = 1.0e-3
    z0 = 0.90 * ANG
    _mol_R, mc_R, solver_R = _build(z0)
    _mol_P, mc_P, solver_P = _build(z0 + h)

    ncas = mc_R.ncas
    nelec = (1, 1)
    nst = 2
    ci_R = fdv.mps_ci_list(solver_R, ncas, nelec, nst)
    ci_P = fdv.mps_ci_list(solver_P, ncas, nelec, nst)

    # The response object captures the *current* block2 global frame at
    # construction.  With two live solvers, pin the frame to each solver's own
    # driver frame first, so each response uses its own scratch (otherwise the
    # second solver's frame leaks into the first response object).
    import block2
    block2.Global.frame = mc_R.fcisolver._driver.frame
    obj_R = _make_mps_krylov_response(mc_R)     # host
    block2.Global.frame = mc_P.fcisolver._driver.frame
    obj_P = _make_mps_krylov_response(mc_P)     # foreign

    loaded = [
        load_foreign_mps(obj_R._driver_su2, obj_R._su2_frame,
                         obj_P._driver_su2, obj_P._su2_frame,
                         obj_P._state_mps[j], tag=f"XFER{j}")
        for j in range(nst)
    ]

    self_ovl = [float(obj_R._mps_overlap(loaded[j], loaded[j])) for j in range(nst)]
    O_mps = np.array([[obj_R._mps_overlap(obj_R._state_mps[i], loaded[j])
                       for j in range(nst)] for i in range(nst)])
    O_fci = np.array([[overlap_fci(ci_R[i], ci_P[j], np.eye(ncas), ncas, nelec)
                       for j in range(nst)] for i in range(nst)])

    # discriminating quantity: the off-diagonal magnitude (the displacement)
    off_mps = abs(float(O_mps[0, 1]))
    off_fci = abs(float(O_fci[0, 1]))
    rel_off_err = abs(off_mps - off_fci) / max(off_fci, 1e-30)

    out = {
        "name": "cross_driver_transport_heh_cas22",
        "h_bohr": h,
        "self_overlaps": self_ovl,
        "offdiag_mps": off_mps,
        "offdiag_fci": off_fci,
        "rel_offdiag_err": rel_off_err,
        "O_mps_abs": np.abs(O_mps).tolist(),
        "O_fci_abs": np.abs(O_fci).tolist(),
        "status": "pass" if (all(abs(s - 1.0) < 1e-6 for s in self_ovl)
                             and rel_off_err < 0.05) else "fail",
    }
    (_HERE / "test_cross_driver_transport.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"off-diagonal: transported {off_mps:.3e} vs exact {off_fci:.3e} "
          f"(rel err {rel_off_err:.2e}) -> {out['status']}")
    return 0 if out["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
