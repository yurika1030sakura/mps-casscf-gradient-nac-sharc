"""Finite-difference validation suite: analytic vs FD gradients and NACs.

Reviewer-2 internal consistency, made systematic.  For each FCI-accessible
small system this compares:

  * the analytic SA-DMRG-CASSCF gradient with central finite differences of the
    *same* DMRG state-energy surface, and
  * the analytic derivative coupling with the determinant cross-geometry overlap
    finite difference,

across an h-scan (h = 2e-3, 1e-3, 5e-4, 2e-4 bohr by default).  Each record is a
single (system, quantity, h) entry carrying the analytic value, the FD value,
the absolute error, and a self-diagnosing health verdict, so the table can be
read straight from JSONL and a customer can see at a glance whether a point is
trustworthy.  The large-CAS counterparts (CAS(10,10)/(14,14)/(20,20)) live in
the beyond-FCI drivers; this suite covers the FCI-accessible regression rung.

Usage:
  python run_fd_validation_suite.py --system heh --out data/fd_heh.jsonl
  python run_fd_validation_suite.py --system all --out data/fd_suite.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
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

import fd_validation as fdv
from analytic_cp_sharc import compute_grad_nac_analytic_cp
from system_diagnostics import assess_point

ANG = 1.8897261246257702

# Each system: FCI-accessible, with the bond-axis component used for the scan.
SYSTEMS = {
    "heh": dict(
        atoms=["He", "H"], z_ang=0.90, basis="3-21G", charge=1, spin=0,
        ncas=2, nelecas=2, bond_dim=100, atom=1, axis=2,
        grad_tol=1e-6, nac_tol=1e-5),
    "ethylene_pi": dict(
        # C=C pi/pi* CAS(2,2); displace a carbon along the C=C (z) axis.
        atoms=["C", "C", "H", "H", "H", "H"],
        geom_ang=[(0, 0, 0.667), (0, 0, -0.667),
                  (0, 0.923, 1.238), (0, -0.923, 1.238),
                  (0, 0.923, -1.238), (0, -0.923, -1.238)],
        basis="6-31G", charge=0, spin=0, ncas=2, nelecas=2, bond_dim=200,
        atom=0, axis=2, grad_tol=1e-5, nac_tol=1e-4),
}

DEFAULT_H_SCAN = (2.0e-3, 1.0e-3, 5.0e-4, 2.0e-4)


def _coords_bohr(cfg):
    if "z_ang" in cfg:
        return np.array([[0.0, 0.0, 0.0],
                         [0.0, 0.0, cfg["z_ang"] * ANG]])
    return np.array(cfg["geom_ang"]) * ANG


def run_system(name, cfg, h_scan):
    coords = _coords_bohr(cfg)
    atoms = cfg["atoms"]
    a, x = cfg["atom"], cfg["axis"]
    scfg = dict(fdv.DEFAULT_SOLVER_CFG)
    scfg.update(bond_dim=cfg["bond_dim"], n_sweeps=24, sweep_tol=1e-10, n_threads=1)

    # analytic reference (gradient of S0 and NAC 0-1) at R
    _mol, _mf, mc, _solver = fdv.build_sa_dmrg_casscf(
        atoms, coords, basis=cfg["basis"], charge=cfg["charge"], spin=cfg["spin"],
        ncas=cfg["ncas"], nelecas=cfg["nelecas"], nroots=2, weights=[0.5, 0.5],
        solver_cfg=scfg)
    res = compute_grad_nac_analytic_cp(
        mc, gradient_states=[0], nac_pairs=[(0, 1)], backend="mps-krylov",
        tol=1e-8, max_iter=300)
    g_an = float(np.asarray(res["grad"][0])[a, x])
    d_an = float(np.asarray(res["nac"][(0, 1)])[a, x])

    records = []
    for h in h_scan:
        rec = {"kind": "fd", "system": name, "active_space": f"CAS({cfg['ncas']},{cfg['nelecas']})",
               "basis": cfg["basis"], "atom": a, "axis": x, "h_bohr": float(h)}
        try:
            g_fd_arr, gdiag = fdv.fd_gradient(
                atoms, coords, state=0, basis=cfg["basis"], charge=cfg["charge"],
                spin=cfg["spin"], ncas=cfg["ncas"], nelecas=cfg["nelecas"], nroots=2,
                weights=[0.5, 0.5], solver_cfg=scfg, h_bohr=h, atmlst=[a],
                components=[x], track_roots="fci_overlap", return_diagnostics=True)
            g_fd = float(g_fd_arr[a, x])
            sig = min((c["active_subspace_sigma_min"] for c in gdiag["components"]),
                      default=None)
            d_fd_arr = fdv.fd_nac(
                atoms, coords, bra=0, ket=1, basis=cfg["basis"], charge=cfg["charge"],
                spin=cfg["spin"], ncas=cfg["ncas"], nelecas=cfg["nelecas"], nroots=2,
                weights=[0.5, 0.5], solver_cfg=scfg, h_bohr=h, atmlst=[a],
                components=[x])
            d_fd = float(d_fd_arr[a, x])
            rec.update(
                g_analytic=g_an, g_fd=g_fd, grad_abs_err=abs(g_an - g_fd),
                d_analytic=d_an, d_fd=d_fd, nac_abs_err=abs(abs(d_an) - abs(d_fd)),
                active_subspace_sigma_min=sig,
                grad_health=assess_point(
                    active_subspace_sigma_min=sig).to_dict()["overall"],
                grad_ok=bool(abs(g_an - g_fd) < cfg["grad_tol"]),
                nac_ok=bool(abs(abs(d_an) - abs(d_fd)) < cfg["nac_tol"]))
        except Exception as exc:  # noqa: BLE001
            rec.update(status="error", exception=type(exc).__name__,
                       message=str(exc)[:300])
        records.append(rec)
    return records


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="all", choices=["all"] + list(SYSTEMS))
    ap.add_argument("--h-scan", type=float, nargs="*", default=list(DEFAULT_H_SCAN))
    ap.add_argument("--out", default=str(_HERE / "data" / "fd_validation_suite.jsonl"))
    args = ap.parse_args()

    names = list(SYSTEMS) if args.system == "all" else [args.system]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    for name in names:
        print(f"=== FD validation: {name} ===", flush=True)
        try:
            for rec in run_system(name, SYSTEMS[name], args.h_scan):
                append_jsonl(out, rec)
                msg = (f"  h={rec['h_bohr']:.0e} grad_err={rec.get('grad_abs_err','?'):} "
                       f"nac_err={rec.get('nac_abs_err','?')}"
                       if "grad_abs_err" in rec else f"  {rec.get('message','error')}")
                print(msg, flush=True)
        except Exception as exc:  # noqa: BLE001
            append_jsonl(out, {"kind": "error", "system": name,
                               "message": str(exc), "tb": traceback.format_exc()[-1500:]})
            print(f"  ERROR {name}: {exc}", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
