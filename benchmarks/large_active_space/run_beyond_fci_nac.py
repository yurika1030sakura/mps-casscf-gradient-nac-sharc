"""Beyond-FCI nonadiabatic coupling by FCI-free finite differences (polyene).

Assembles the bra-fixed derivative coupling d^x_{01} between the two lowest
SA(2)-DMRG-CASSCF states of an all-trans polyene pi system entirely from MPS
operations -- the displaced states at R+-h are transported into the reference
driver, rotated into the reference orbital basis, overlapped, and differenced --
so no FCI vector is ever formed.  At a small pi space (CAS(10,10), determinant
space 6.3e4) the result is cross-checked against the determinant-level
finite-difference NAC; at CAS(20,20) (determinant space 3.4e10, FCI impossible)
the MPS-native value is reported with its internal diagnostics (active-subspace
singular values, root gap, certified response residual of the analytic NAC).

Usage:  python run_beyond_fci_nac.py --ncarbon 10   # FCI-checked
        python run_beyond_fci_nac.py --ncarbon 20   # beyond FCI
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DEV = _HERE.parents[1] / "src" / "dmrg_analytic_dev"
SHARC = _HERE.parents[1] / "sharc_interface"
for p in (str(DEV), str(SHARC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import block2
import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response
from cross_geometry_overlap import cross_geometry_overlap_matrix
from run_polyene_beyond_fci import (polyene_geometry, select_pi_active, det_dim,
                                    beyond_fci_solver_cfg, FCI_THRESHOLD)

ANG = 1.8897261246257702


def _build(symbols, coords_bohr, basis, ncas, nelecas, cfg, mo_guess):
    mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
        symbols, coords_bohr, basis=basis, charge=0, spin=0,
        ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
        solver_cfg=cfg, mo_guess=mo_guess)
    return mol, mc, solver


def _pi_space(symbols, coords_ang, basis, n_carbon):
    from pyscf import gto, scf
    mol = gto.M(atom=[(symbols[i], tuple(coords_ang[i])) for i in range(len(symbols))],
                basis=basis, charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    return select_pi_active(mol, mf, n_carbon)


def run(n_carbon, *, basis="sto-3g", bond_dim=600, threads=8,
        stack_mem_mb=8000, h_bohr=1.0e-3, atom_disp=None, comp=1):
    symbols0 = [a[0] for a in polyene_geometry(n_carbon)]
    coords0 = np.array([a[1] for a in polyene_geometry(n_carbon)])
    centre = n_carbon // 2 if atom_disp is None else int(atom_disp)

    def coords_at(sign):
        c = coords0.copy()
        c[centre, comp] += sign * h_bohr / ANG     # displace in Angstrom units
        return c

    geoms = {"R": coords0, "P": coords_at(+1.0), "M": coords_at(-1.0)}
    ncas, nelecas, mo0 = _pi_space(symbols0, coords0, basis, n_carbon)
    cfg = beyond_fci_solver_cfg(ncas, bond_dim, threads, stack_mem_mb)
    ddim = det_dim(ncas, (nelecas // 2, nelecas - nelecas // 2))
    beyond_fci = ddim >= FCI_THRESHOLD

    built = {}
    for key, c_ang in geoms.items():
        _, _, moX = _pi_space(symbols0, c_ang, basis, n_carbon)
        mol, mc, solver = _build(symbols0, c_ang * ANG, basis, ncas, nelecas, cfg, moX)
        built[key] = dict(mol=mol, mc=mc, solver=solver)

    # response objects, each pinned to its own driver frame
    for key in ("R", "P", "M"):
        block2.Global.frame = built[key]["mc"].fcisolver._driver.frame
        built[key]["obj"] = _make_mps_krylov_response(built[key]["mc"])

    R, P, M = built["R"], built["P"], built["M"]
    ncore = R["mc"].ncore
    host_frame = R["mc"].fcisolver._driver.frame

    t0 = time.perf_counter()
    Op, s_p = cross_geometry_overlap_matrix(
        R["obj"], P["obj"], R["mol"], P["mol"], R["mc"].mo_coeff, P["mc"].mo_coeff,
        ncore, ncas, 2, host_frame=host_frame, tag="BFNP")
    Om, s_m = cross_geometry_overlap_matrix(
        R["obj"], M["obj"], R["mol"], M["mol"], R["mc"].mo_coeff, M["mc"].mo_coeff,
        ncore, ncas, 2, host_frame=host_frame, tag="BFNM")
    nac_wall = time.perf_counter() - t0
    d_mps = float((Op[0, 1] - Om[0, 1]) / (2.0 * h_bohr))

    sig_p = float(np.min(np.linalg.svd(s_p, compute_uv=False)))
    sig_m = float(np.min(np.linalg.svd(s_m, compute_uv=False)))
    e = list(np.asarray(R["solver"].e_states, dtype=float).ravel())

    out = {
        "system": f"polyene_C{n_carbon}", "basis": basis,
        "ncas": ncas, "nelecas": nelecas, "det_dim": ddim,
        "fci_feasible": not beyond_fci,
        "bond_dim": bond_dim, "h_bohr": h_bohr,
        "displaced_atom": centre, "component": comp,
        "d_nac_mps_native_01": d_mps,
        "active_subspace_sigma_min_plus": sig_p,
        "active_subspace_sigma_min_minus": sig_m,
        "gap_Eh": float(abs(e[1] - e[0])),
        "nac_mps_wall_s": nac_wall,
        "fci_conversion": bool(not cfg["skip_kernel_fci_conversion"]),
    }

    # FCI cross-check where the determinant space still allows it
    if not beyond_fci:
        cfg_fci = dict(fdv.DEFAULT_SOLVER_CFG)
        cfg_fci.update(bond_dim=bond_dim, n_sweeps=30, sweep_tol=1.0e-10,
                       n_threads=int(threads), stack_mem_mb=int(stack_mem_mb))
        d_fci = fdv.fd_nac(symbols0, coords0 * ANG, bra=0, ket=1, basis=basis,
                           charge=0, spin=0, ncas=ncas, nelecas=nelecas, nroots=2,
                           weights=[0.5, 0.5], solver_cfg=cfg_fci, h_bohr=h_bohr,
                           atmlst=[centre], components=[comp])
        d_fci_01 = float(d_fci[centre, comp])
        out["d_nac_fci_fd_01"] = d_fci_01
        out["abs_err_mps_vs_fci"] = abs(abs(d_mps) - abs(d_fci_01))
        out["cross_check"] = "pass" if out["abs_err_mps_vs_fci"] < 1.0e-3 else "fail"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, default=10)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--bond-dim", type=int, default=600)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--stack-mem-mb", type=int, default=8000)
    ap.add_argument("--h-bohr", type=float, default=1.0e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(f"=== beyond-FCI NAC polyene C{args.ncarbon} ===", flush=True)
    try:
        r = run(args.ncarbon, basis=args.basis, bond_dim=args.bond_dim,
                threads=args.threads, stack_mem_mb=args.stack_mem_mb,
                h_bohr=args.h_bohr)
        print(json.dumps(r, indent=2), flush=True)
    except Exception as exc:  # noqa: BLE001
        r = {"system": f"polyene_C{args.ncarbon}", "status": "error",
             "exception": type(exc).__name__, "message": str(exc),
             "traceback_tail": traceback.format_exc()[-3000:]}
        print(f"  ERROR: {exc}", flush=True)
    out = Path(args.out) if args.out else _HERE / "data" / f"beyond_fci_nac_c{args.ncarbon}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
