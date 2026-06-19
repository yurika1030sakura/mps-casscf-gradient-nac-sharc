"""Beyond-FCI active-space benchmark: all-trans polyene pi spaces.

For C(n)H(n+2) all-trans polyenes the pi space is CAS(n,n).  The ladder
  C10 -> CAS(10,10)  det dim C(10,5)^2 = 6.35e4   (FCI-easy)
  C14 -> CAS(14,14)  det dim C(14,7)^2 = 1.18e7   (FCI-borderline)
  C20 -> CAS(20,20)  det dim C(20,10)^2 = 3.41e10 (FCI-impossible)
spans from the FCI-accessible regime into the regime where a conventional
FCI-CASSCF derivative is impossible and DMRG is required.

Validation where FCI is impossible uses the analytic SA-DMRG-CASSCF gradient
against a central finite difference of the solver's own state energy
(needs no FCI reference and no cross-geometry overlap), on a curated set of
Cartesian components.  The active-space determinant dimension is reported so
the beyond-FCI claim is explicit.
"""

from __future__ import annotations

import argparse
import json
import math
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

from pyscf import gto, scf
from pyscf.mcscf import avas
import fd_validation as fdv

ANG = 1.8897261246257702


def polyene_geometry(n_carbon):
    """All-trans C(n)H(n+2) in Angstrom (planar zig-zag backbone + H's)."""
    # alternating double/single bonds; standard sp2 geometry
    r_cc = 1.40            # mean C-C (Angstrom)
    ang = math.radians(120.0)
    dx = r_cc * math.sin(ang / 2.0)
    dy = r_cc * math.cos(ang / 2.0)
    atoms = []
    cx = 0.0
    carbons = []
    for i in range(n_carbon):
        cy = 0.0 + (dy if i % 2 else 0.0)
        carbons.append((cx, cy))
        atoms.append(("C", (cx, cy, 0.0)))
        cx += dx
    # backbone H's (one per carbon, plus terminal extra H's) -- approximate,
    # geometry is relaxed implicitly only through the response test's FD which
    # uses the same Hamiltonian on both sides, so exactness is not required.
    r_ch = 1.09
    for i, (x, y) in enumerate(carbons):
        hy = y + (r_ch if (i % 2 == 0) else -r_ch)
        atoms.append(("H", (x, hy, 0.0)))
    # two terminal H's to satisfy C(n)H(n+2)
    x0, y0 = carbons[0]
    atoms.append(("H", (x0 - r_ch, y0, 0.0)))
    xN, yN = carbons[-1]
    atoms.append(("H", (xN + r_ch, yN, 0.0)))
    return atoms


def det_dim(ncas, nelec):
    na, nb = nelec
    return math.comb(ncas, na) * math.comb(ncas, nb)


def run_one(n_carbon, *, basis="sto-3g", bond_dim=800, threads=8,
            stack_mem_mb=8000, h_bohr=1.0e-3, fd_components=2):
    atoms = polyene_geometry(n_carbon)
    coords_ang = np.array([a[1] for a in atoms])
    symbols = [a[0] for a in atoms]
    coords_bohr = coords_ang * ANG

    # RHF + AVAS pi-space selection
    mol = gto.M(atom=[(symbols[i], tuple(coords_ang[i])) for i in range(len(symbols))],
                basis=basis, charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    ncas, nelecas, mo_init = avas.avas(mf, ["C 2pz"], canonicalize=True,
                                       with_iao=False)
    ncas = int(ncas)
    nelecas = int(nelecas)
    na = nelecas // 2 + nelecas % 2
    nb = nelecas // 2
    ddim = det_dim(ncas, (na, nb))

    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=bond_dim, n_sweeps=30, sweep_tol=1.0e-9,
               n_threads=int(threads), stack_mem_mb=int(stack_mem_mb))

    t0 = time.perf_counter()
    _mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
        symbols, coords_bohr, basis=basis, charge=0, spin=0,
        ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
        solver_cfg=cfg, mo_guess=mo_init,
    )
    build_wall = time.perf_counter() - t0
    e_states = list(np.asarray(solver.e_states, dtype=float).ravel())

    # analytic gradient (state 0)
    t0 = time.perf_counter()
    g_an = fdv.analytic_gradient(mc, 0, backend="mps-krylov", tol=1e-7,
                                 max_iter=80)
    analytic_wall = time.perf_counter() - t0

    # FD gradient validation on a few curated components (heavy atoms, in-plane)
    # pick the two carbons nearest the chain centre, x-component
    nC = n_carbon
    centre = nC // 2
    comp_atoms = [centre, max(0, centre - 1)][:fd_components]
    fd_results = []
    t0 = time.perf_counter()
    for a in comp_atoms:
        g_fd = fdv.fd_gradient(
            symbols, coords_bohr, state=0, basis=basis, charge=0, spin=0,
            ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
            solver_cfg=cfg, h_bohr=h_bohr, atmlst=[a], components=[0],
            track_roots=True,
        )
        fd_results.append({
            "atom": int(a), "comp": 0,
            "g_analytic": float(g_an[a, 0]),
            "g_fd": float(g_fd[a, 0]),
            "abs_err": float(abs(g_an[a, 0] - g_fd[a, 0])),
        })
    fd_wall = time.perf_counter() - t0
    max_err = max((r["abs_err"] for r in fd_results), default=0.0)

    return {
        "n_carbon": n_carbon, "basis": basis,
        "ncas": ncas, "nelecas": nelecas, "det_dim": ddim,
        "fci_feasible": ddim < 5.0e7,
        "bond_dim": bond_dim,
        "e_states": e_states,
        "build_wall_s": build_wall,
        "analytic_grad_wall_s": analytic_wall,
        "fd_grad_wall_s": fd_wall,
        "fd_components": fd_results,
        "fd_grad_max_abs_err": max_err,
        "validated": bool(max_err < 5.0e-4),
        "h_bohr": h_bohr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, required=True)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--bond-dim", type=int, default=800)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--stack-mem-mb", type=int, default=8000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        r = run_one(args.ncarbon, basis=args.basis, bond_dim=args.bond_dim,
                    threads=args.threads, stack_mem_mb=args.stack_mem_mb)
        print(json.dumps(r, indent=2), flush=True)
    except Exception as exc:
        r = {"n_carbon": args.ncarbon, "status": "error",
             "exception": type(exc).__name__, "message": str(exc),
             "traceback_tail": traceback.format_exc()[-3000:]}
        print(f"ERROR: {exc}", flush=True)
    out = Path(args.out) if args.out else _HERE / "data" / f"polyene_c{args.ncarbon}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"benchmark": "polyene_beyond_fci", "result": r},
                              indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
