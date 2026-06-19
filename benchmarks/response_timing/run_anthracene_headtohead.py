"""Decisive cost test: global MPS-Krylov vs sweep-localized Schur response on
anthracene CAS(14,14).

The smaller systems (HeH+ ... C2) fix the agreement and the cost *trend*; this
run puts both solvers on a realistic pi active space with a non-trivial orbital
rotation manifold (STO-3G anthracene: 80 orbitals, 40 core, 14 active), where
the orbital block of the response is large enough that the orbital/CI coupling
cost actually matters.  Both solvers are run on the same converged
SA(2)-DMRG-CASSCF reference; we record wall time, CI-operation counts, the
true (global-operator) residual of each solution, and their mutual agreement.

Usage:  python run_anthracene_headtohead.py --bond-dim 600 --threads 8
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
LARGE = _HERE.parents[1] / "benchmarks" / "large_active_space"
for p in (str(DEV), str(SHARC), str(LARGE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response
from sweep_coupled_response import solve_state_sweep_schur
from run_polyene_beyond_fci import select_pi_active, det_dim

ANG = 1.8897261246257702  # Bohr per Angstrom


def anthracene_geometry(a=1.40, ch=1.09):
    """Idealized planar (D2h) anthracene in the xy-plane, Angstrom.

    Three linearly fused regular hexagons (C-C = a), built from the honeycomb
    so the pi axis is z.  Hydrogens are placed radially outward from each ring
    centre through the CH carbons at distance ch.  Returns (atoms, coords_ang).
    """
    h = a * np.sqrt(3.0) / 2.0          # horizontal half-width of a hexagon
    hy = a / 2.0                        # y of the vertical-edge vertices
    # ring centres on the long (x) axis
    centres = {"M": 0.0, "R": 2.0 * h, "L": -2.0 * h}
    # six vertices of a regular hexagon (vertical left/right edges) about a centre
    def hexagon(cx):
        return {
            1: (cx + h, hy), 2: (cx, a), 3: (cx - h, hy),
            4: (cx - h, -hy), 5: (cx, -a), 6: (cx + h, -hy),
        }
    M = hexagon(centres["M"]); R = hexagon(centres["R"]); L = hexagon(centres["L"])
    # carbons: middle ring full, outer rings contribute only their non-shared 4
    carbons = [M[1], M[2], M[3], M[4], M[5], M[6],
               R[1], R[2], R[5], R[6],          # R[3]=M? no: R shares M1,M6
               L[2], L[3], L[4], L[5]]
    # the shares: R left edge = M right edge (M1,M6); L right edge = M left edge (M3,M4)
    # so R's new carbons are R1,R2,R5,R6 and L's new carbons are L2,L3,L4,L5
    # CH carbons (2 ring-C neighbours) and which ring centre they belong to
    ch_carbons = [
        (M[2], centres["M"]), (M[5], centres["M"]),
        (R[1], centres["R"]), (R[2], centres["R"]),
        (R[5], centres["R"]), (R[6], centres["R"]),
        (L[2], centres["L"]), (L[3], centres["L"]),
        (L[4], centres["L"]), (L[5], centres["L"]),
    ]
    hydrogens = []
    for (cx_c, cy_c), cx_ring in ch_carbons:
        dx, dy = cx_c - cx_ring, cy_c - 0.0
        n = np.hypot(dx, dy)
        hydrogens.append((cx_c + ch * dx / n, cy_c + ch * dy / n))

    atoms = ["C"] * len(carbons) + ["H"] * len(hydrogens)
    xy = np.array(carbons + hydrogens, dtype=float)
    coords = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])

    # sanity: composition and bonded distances
    assert atoms.count("C") == 14 and atoms.count("H") == 10, "not C14H10"
    cc = []
    for i in range(14):
        for j in range(i + 1, 14):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < 1.6:
                cc.append(d)
    assert len(cc) == 16, f"expected 16 C-C bonds, got {len(cc)}"  # anthracene
    assert max(abs(d - a) for d in cc) < 1e-9, "C-C bonds not all = a"
    for i in range(14, 24):
        dmin = min(np.linalg.norm(coords[i] - coords[k]) for k in range(14))
        assert abs(dmin - ch) < 1e-9, "C-H bond != ch"
    return atoms, coords


def run(bond_dim=600, tol=1.0e-8, n_threads=8, stack_mem_mb=8000,
        basis="sto-3g", max_iter=60):
    atoms, coords_ang = anthracene_geometry()
    coords_bohr = coords_ang * ANG

    # pi active space: build a throwaway RHF to rank C-pz character, then feed
    # the reordered MOs to build_sa_dmrg_casscf as the CASSCF guess.
    from pyscf import gto, scf
    mol0 = gto.M(atom=[(atoms[i], tuple(coords_bohr[i])) for i in range(len(atoms))],
                 basis=basis, unit="Bohr", symmetry=False, verbose=0)
    mf0 = scf.RHF(mol0).run(conv_tol=1.0e-12)
    ncas, nelecas, mo_init = select_pi_active(mol0, mf0, 14)
    assert (ncas, nelecas) == (14, 14), f"not CAS(14,14): {(ncas, nelecas)}"

    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=bond_dim, n_sweeps=30, sweep_tol=1.0e-10,
               n_threads=int(n_threads), stack_mem_mb=int(stack_mem_mb))

    t_build = time.perf_counter()
    _mol, _mf, mc, _solver = fdv.build_sa_dmrg_casscf(
        atoms, coords_bohr, basis=basis, charge=0, spin=0,
        ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
        solver_cfg=cfg, mo_guess=mo_init,
    )
    wall_build = time.perf_counter() - t_build

    # global MPS-Krylov
    obj_g = _make_mps_krylov_response(mc)
    t0 = time.perf_counter()
    k_g, ci_g, info_g, meta_g = obj_g.solve_mps(0, tol=tol, max_iter=max_iter)
    wall_g = time.perf_counter() - t0

    # sweep-Schur
    obj_s = _make_mps_krylov_response(mc)
    t0 = time.perf_counter()
    k_s, ci_s, info_s, meta_s = solve_state_sweep_schur(
        obj_s, 0, orb_tol=tol, ci_tol=1.0e-10,
    )
    wall_s = time.perf_counter() - t0

    # Orbital response is a plain vector in the uniq-rotation parameterization,
    # so it compares across the two response objects directly.  The active-space
    # (CI) parts live as MPS on different scratch objects; rather than pay for a
    # second global solve just to overlap them, we lean on the Schur solver's
    # own true residual against the *global* coupled operator (meta_s) as the
    # correctness arbiter -- once kappa agrees and that residual is tight, the
    # CI block is fixed by the recovery equation z_C = H_CC^{-1}(b_C-H_Ck z_k).
    diff_k = float(np.linalg.norm(
        obj_g.mc.pack_uniq_var(obj_g._canonical_kappa(k_g))
        - obj_g.mc.pack_uniq_var(obj_g._canonical_kappa(k_s))
    ))

    return {
        "system": "anthracene_cas1414",
        "basis": basis, "ncas": ncas, "nelecas": nelecas,
        "n_orbitals": int(mc.mo_coeff.shape[1]),
        "ncore": int(mc.ncore),
        "det_dim": float(det_dim(ncas, (nelecas // 2, nelecas - nelecas // 2))),
        "bond_dim": bond_dim,
        "casscf_build_wall_s": wall_build,
        "global_wall_s": wall_g,
        "schur_wall_s": wall_s,
        "speedup_global_over_schur": wall_g / max(wall_s, 1e-30),
        "global_niter": meta_g.get("niter"),
        "schur_applies": meta_s.get("schur_applies"),
        "schur_orb_dim": meta_s.get("orb_dim"),
        "global_residual": meta_g.get("residual"),
        "global_info": int(info_g),
        "schur_true_residual_rel": meta_s.get("true_residual_rel"),
        "schur_info": int(info_s),
        "schur_HOO_cond_eff": meta_s.get("HOO_cond_eff"),
        "schur_HOO_rank_eff": meta_s.get("HOO_rank_eff"),
        "schur_residual_tol_used": meta_s.get("residual_tol_used"),
        "diff_kappa_global_vs_schur": diff_k,
        "agree": bool(diff_k < 1e-5
                      and (meta_s.get("true_residual_rel") or 1.0) < 1e-6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bond-dim", type=int, default=600)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--stack-mem-mb", type=int, default=8000)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print("=== anthracene CAS(14,14) head-to-head ===", flush=True)
    try:
        r = run(bond_dim=args.bond_dim, n_threads=args.threads,
                stack_mem_mb=args.stack_mem_mb, basis=args.basis,
                max_iter=args.max_iter)
        print(json.dumps(r, indent=2), flush=True)
    except Exception as exc:
        r = {"system": "anthracene_cas1414", "status": "error",
             "exception": type(exc).__name__, "message": str(exc),
             "traceback_tail": traceback.format_exc()[-3000:]}
        print(f"  ERROR: {exc}", flush=True)
    out = Path(args.out) if args.out else _HERE / "data" / "anthracene_headtohead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": "anthracene_cas1414_global_vs_schur",
        "result": r,
    }, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
