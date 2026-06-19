"""LiF ionic/covalent avoided crossing: energies, subspace tracking, and NAC.

Scans the Li-F bond length through the ionic/covalent avoided crossing of the
two lowest singlets and, at each point, runs SA(2)-DMRG-CASSCF.  With NAC enabled
it also computes the analytic derivative coupling d_{01} and its finite-
difference reference along the bond axis.  The converged active orbitals and CI
vectors are saved per point so a post-processing step can form the cross-geometry
overlap matrix O_IJ, polar-decompose it for gauge alignment, and report the
singular values sigma_alpha(O) through the crossing.

This is a singlet, FCI-tractable demonstration of root/subspace continuity near
a near-degeneracy; it is independent of the beyond-FCI runs.

Usage:
  python run_lif_avoided_crossing.py --R 6.5 --out data/lif/lif_R6.50.json
  python run_lif_avoided_crossing.py --scan 2.5,4,5,6,6.5,7,8 --no-nac   # smoke
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

import numpy as np
import fd_validation as fdv

ANG = 1.8897261246257702


def lif_point(R_ang, *, basis="6-31G", ncas=6, nelecas=6, bond_dim=400,
              threads=1, stack_mem_mb=None, do_nac=True, h_bohr=1.0e-3,
              save_npz=None):
    """One Li-F separation: SA(2)-DMRG-CASSCF energies and (optionally) NAC."""
    atoms = ["Li", "F"]
    coords_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, float(R_ang)]]) * ANG

    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=bond_dim, n_sweeps=30, sweep_tol=1.0e-10,
               n_threads=int(threads))
    if stack_mem_mb is not None:
        cfg["stack_mem_mb"] = int(stack_mem_mb)

    t0 = time.perf_counter()
    mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
        atoms, coords_bohr, basis=basis, charge=0, spin=0,
        ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
        solver_cfg=cfg,
    )
    build_wall = time.perf_counter() - t0
    e = list(np.asarray(solver.e_states, dtype=float).ravel())
    gap = float(abs(e[1] - e[0]))

    rec = {
        "R_ang": float(R_ang), "basis": basis, "ncas": ncas,
        "nelecas": nelecas, "bond_dim": bond_dim,
        "e_states": e, "gap_Eh": gap, "build_wall_s": build_wall,
    }

    if save_npz is not None:
        nelec = mc.nelecas
        ci = fdv.mps_ci_list(solver, ncas, nelec, 2)
        np.savez(
            save_npz,
            R_ang=float(R_ang), mo_coeff=mc.mo_coeff, ncore=mc.ncore,
            ncas=ncas, e_states=np.asarray(e),
            atom_coords_bohr=coords_bohr,
            ci0=np.asarray(ci[0]), ci1=np.asarray(ci[1]),
        )
        rec["npz"] = str(save_npz)

    if do_nac:
        from analytic_cp_sharc import compute_grad_nac_analytic_cp
        t0 = time.perf_counter()
        res = compute_grad_nac_analytic_cp(
            mc, gradient_states=[], nac_pairs=[(0, 1)],
            backend="mps-krylov", tol=1.0e-7, max_iter=200,
        )
        nac_an = np.asarray(res["nac"][(0, 1)])
        rec["nac_analytic_wall_s"] = time.perf_counter() - t0
        rec["nac_analytic_z_atom1"] = float(nac_an[1, 2])
        rec["nac_analytic_norm"] = float(np.linalg.norm(nac_an))

        t0 = time.perf_counter()
        d_fd = fdv.fd_nac(
            atoms, coords_bohr, bra=0, ket=1, basis=basis, charge=0, spin=0,
            ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
            solver_cfg=cfg, h_bohr=h_bohr, atmlst=[1], components=[2],
        )
        rec["nac_fd_wall_s"] = time.perf_counter() - t0
        rec["nac_fd_z_atom1"] = float(d_fd[1, 2])
        rec["nac_abs_err_z"] = float(abs(abs(nac_an[1, 2]) - abs(d_fd[1, 2])))

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=None, help="single Li-F sep (Ang)")
    ap.add_argument("--scan", default=None, help="comma list of R (Ang)")
    ap.add_argument("--basis", default="6-31G")
    ap.add_argument("--ncas", type=int, default=6)
    ap.add_argument("--nelecas", type=int, default=6)
    ap.add_argument("--bond-dim", type=int, default=400)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stack-mem-mb", type=int, default=None)
    ap.add_argument("--no-nac", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Rs = ([args.R] if args.R is not None
          else [float(x) for x in args.scan.split(",")] if args.scan
          else [6.5])
    recs = []
    for R in Rs:
        print(f"=== LiF R={R} Ang ===", flush=True)
        try:
            npz = None
            if args.out and not args.no_nac:
                npz = str(Path(args.out).with_suffix("")) + f"_R{R:.2f}.npz"
            r = lif_point(R, basis=args.basis, ncas=args.ncas,
                          nelecas=args.nelecas, bond_dim=args.bond_dim,
                          threads=args.threads, stack_mem_mb=args.stack_mem_mb,
                          do_nac=not args.no_nac, save_npz=npz)
            print(json.dumps(r, indent=2), flush=True)
        except Exception as exc:
            r = {"R_ang": R, "status": "error", "exception": type(exc).__name__,
                 "message": str(exc), "traceback_tail": traceback.format_exc()[-2000:]}
            print(f"  ERROR R={R}: {exc}", flush=True)
        recs.append(r)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"benchmark": "lif_avoided_crossing", "points": recs}, indent=2) + "\n")
        print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
