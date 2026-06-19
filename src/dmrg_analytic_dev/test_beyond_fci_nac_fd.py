"""End-to-end MPS-native (FCI-free) NAC finite difference, validated at CAS(2,2).

Assembles the bra-fixed derivative coupling

    d^x_{IJ} = ( <Psi_I(R) | Psi_J(R+h)> - <Psi_I(R) | Psi_J(R-h)> ) / (2h)

entirely from MPS operations: the displaced states are transported into the
reference driver, their orbitals rotated into the reference basis, and the
overlaps taken as same-basis MPS expectations -- no determinant expansion.  At
CAS(2,2) (where DMRG is exact) the result is checked against the determinant-
level finite-difference NAC (``fd_validation.fd_nac``) and the analytic NAC.
The same construction is FCI-free, so it carries over to active spaces where no
FCI vector can be built.
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

import block2
import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response, compute_grad_nac_analytic_cp
from cross_geometry_overlap import (cross_geometry_active_overlap,
                                    load_foreign_mps, rotate_mps_orbitals)

ANG = 1.8897261246257702


def _build(z_bohr):
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=100, n_sweeps=24, sweep_tol=1.0e-10, n_threads=1)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, z_bohr]])
    mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
        ["He", "H"], coords, basis="3-21G", charge=1, spin=0,
        ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5], solver_cfg=cfg)
    return mol, mc, solver


def _response_pinned(mc):
    block2.Global.frame = mc.fcisolver._driver.frame
    return _make_mps_krylov_response(mc)


def main():
    h = 1.0e-3
    z0 = 0.90 * ANG
    mol_R, mc_R, _sR = _build(z0)
    mol_P, mc_P, _sP = _build(z0 + h)
    mol_M, mc_M, _sM = _build(z0 - h)

    ncas, ncore = mc_R.ncas, mc_R.ncore
    nst = 2
    obj_R = _response_pinned(mc_R)
    obj_P = _response_pinned(mc_P)
    obj_M = _response_pinned(mc_M)
    host_drv, host_frame = obj_R._driver_su2, mc_R.fcisolver._driver.frame

    s_p = cross_geometry_active_overlap(mol_R, mol_P, mc_R.mo_coeff, mc_P.mo_coeff,
                                        ncore, ncas)
    s_m = cross_geometry_active_overlap(mol_R, mol_M, mc_R.mo_coeff, mc_M.mo_coeff,
                                        ncore, ncas)

    from overlap_fci_reference import overlap_fci
    ci_R = fdv.mps_ci_list(_sR, ncas, (1, 1), nst)
    ci_P = fdv.mps_ci_list(_sP, ncas, (1, 1), nst)
    ci_M = fdv.mps_ci_list(_sM, ncas, (1, 1), nst)

    def displaced_overlap_row(obj_disp, s, sgn_tag, transpose=False):
        """<Psi_I(R) | Psi_J(disp)> for all I, J, phase-aligned per ket J."""
        M = s.T if transpose else s
        O = np.zeros((nst, nst))
        for j in range(nst):
            loaded = load_foreign_mps(host_drv, host_frame,
                                      obj_disp._driver_su2,
                                      obj_disp._su2_frame,
                                      obj_disp._state_mps[j],
                                      tag=f"NAC{sgn_tag}{j}")
            block2.Global.frame = host_frame
            rot = rotate_mps_orbitals(host_drv, loaded, M, ncas=ncas,
                                      tag=f"NACrot{sgn_tag}{j}", include_stretch=False)
            col = np.array([obj_R._mps_overlap(obj_R._state_mps[i], rot)
                            for i in range(nst)])
            sgn = 1.0 if col[j] >= 0 else -1.0
            O[:, j] = sgn * col
        return O

    # The cross-geometry orbital rotation is applied with the transpose of the
    # active cross overlap (calibrated against overlap_fci): rotating the
    # displaced ket by s^T reproduces <Psi_I(R)|Psi_J(disp)>.
    Op = displaced_overlap_row(obj_P, s_p, "P", transpose=True)
    Om = displaced_overlap_row(obj_M, s_m, "M", transpose=True)
    d_mps = (Op - Om) / (2.0 * h)            # d[I, J] along the H z-displacement
    d_mps_01 = float(d_mps[0, 1])

    # determinant-level FD NAC reference (same h, atom 1, z)
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=100, n_sweeps=24, sweep_tol=1.0e-10, n_threads=1)
    coords_R = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, z0]])
    d_fci = fdv.fd_nac(["He", "H"], coords_R, bra=0, ket=1, basis="3-21G",
                       charge=1, spin=0, ncas=2, nelecas=2, nroots=2,
                       weights=[0.5, 0.5], solver_cfg=cfg, h_bohr=h,
                       atmlst=[1], components=[2])
    d_fci_01 = float(d_fci[1, 2])

    # analytic NAC
    res = compute_grad_nac_analytic_cp(mc_R, gradient_states=[], nac_pairs=[(0, 1)],
                                       backend="mps-krylov", tol=1.0e-8, max_iter=200)
    d_an_01 = float(np.asarray(res["nac"][(0, 1)])[1, 2])

    err_vs_fci = abs(abs(d_mps_01) - abs(d_fci_01))
    err_vs_an = abs(abs(d_mps_01) - abs(d_an_01))
    out = {
        "name": "beyond_fci_nac_fd_heh_cas22",
        "h_bohr": h,
        "d_mps_native_01": d_mps_01,
        "d_fci_fd_01": d_fci_01,
        "d_analytic_01": d_an_01,
        "abs_err_mps_vs_fci_fd": err_vs_fci,
        "abs_err_mps_vs_analytic": err_vs_an,
        "status": "pass" if (err_vs_fci < 1.0e-4 and err_vs_an < 1.0e-4) else "fail",
    }
    (_HERE / "test_beyond_fci_nac_fd.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"MPS-native NAC FD {d_mps_01:.6e} | FCI FD {d_fci_01:.6e} | "
          f"analytic {d_an_01:.6e} -> {out['status']}")
    return 0 if out["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
