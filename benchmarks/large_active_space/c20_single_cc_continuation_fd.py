"""Branch-following (continuation) central FD of C20 single_cc.

The full-step warm-start (c20_single_cc_basin_tracked_fd.py) recovered basin A at
+h (e=-755.28166105, EXACTLY the analytic-predicted smooth-branch value) but the
-h full-step build fell to a higher branch (-755.27979 vs basin-A-expected
-755.28133): a pure optimization/branch-tracking failure on the soft direction.
The forward difference on basin A already gives g=(e(+h)-e_ref)/h=-0.1665 vs
analytic -0.1646 (1.9e-3).

This script gets the CLEAN central difference by CONTINUATION: a ladder of K small
steps from the reference to +-h, each SA-CASSCF build warm-started (orbitals) from
the PREVIOUS step's converged solution.  Tiny steps keep the optimizer inside
basin A, so both +-h land on the smooth branch and g_central matches the analytic.
Each step's e0 and its deviation from the linear basin-A branch are logged so a
basin escape is visible immediately.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DEV = _HERE.parents[1] / "src" / "dmrg_analytic_dev"
SH = _HERE.parents[1] / "sharc_interface"
for _p in (str(_HERE), str(DEV), str(SH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ANG = 1.8897261246257702
_T0 = time.perf_counter()


def log(m):
    print(f"[{time.perf_counter() - _T0:8.1f}s] {m}", flush=True)


def main():
    import block2
    from pyscf import gto
    from run_cas_directional_fd import build_progressive, named_directions
    from run_polyene_beyond_fci import polyene_geometry
    import fd_validation as fdv

    n = 20
    h = 1.0e-3
    K = 4                       # ladder steps from ref to +-h
    ref_seed = 2                # seed 2 reaches basin A at the reference (verified)
    m_schedule = [(256, 1e-8, 1e-4, 40), (512, 1e-10, 1e-5, 60), (800, 1e-12, 1e-6, 100)]
    threads, stack = 16, 24000

    symbols = [a[0] for a in polyene_geometry(n)]
    coords0 = np.array([a[1] for a in polyene_geometry(n)]) * ANG
    ncas = nelecas = n
    q = named_directions(symbols, coords0)["single_cc"]
    out = {"system": "polyene_C20", "direction": "single_cc", "h_bohr": h, "K": K,
           "analytic_single_cc_lowbasin": -0.16458174441719800}
    out_path = _HERE / "data" / "c20_single_cc_continuation.json"

    def build_at(coords_bohr, mo_guess, seed):
        block2.Random.rand_seed(int(seed))
        mol = gto.M(atom=[(symbols[i], tuple(coords_bohr[i])) for i in range(len(symbols))],
                    basis="sto-3g", unit="Bohr", verbose=0)
        mg = mo_guess
        if mg is not None and mo_guess_mol[0] is not None:
            mg = fdv.project_mo_to_new_geometry(mo_guess_mol[0], mol, mo_guess)[0]
        _mol, mc, _sol, _blog = build_progressive(
            symbols, coords_bohr, "sto-3g", ncas, nelecas, m_schedule=m_schedule,
            mo_guess=mg, threads=threads, stack_mem_mb=stack)
        return mol, mc

    mo_guess_mol = [None]  # holds the mol of the current warm-start orbitals

    # reference at basin A
    log(f"REFERENCE build seed={ref_seed} ...")
    mol_ref, mc_ref = build_at(coords0, None, ref_seed)
    e_ref = float(mc_ref.e_states[0])
    mo_ref = np.array(mc_ref.mo_coeff)
    out["e_ref"] = e_ref
    log(f"REFERENCE e0={e_ref:.8f} (basin A ~ -755.2815 or lower is fine)")
    # Anchor the branch prediction on the ACTUAL converged reference energy (basin A
    # has a ~1.5e-4 convergence spread; -755.28149 .. -755.28165 are all basin A).
    # Only abort if the reference clearly landed in a HIGHER basin (>= -755.2810).
    if e_ref > -755.2810:
        log("WARNING: reference landed in a higher basin; aborting")
        out["status"] = "ref_higher_basin"; json.dump(out, open(out_path, "w"), indent=2, default=str)
        return 1

    g_analytic = out["analytic_single_cc_lowbasin"]
    ends = {}
    for sgn in (+1.0, -1.0):
        log(f"=== CONTINUATION ladder sgn={sgn:+.0f}, K={K} ===")
        mo_cur = mo_ref
        mol_cur = mol_ref
        ladder = []
        for k in range(1, K + 1):
            frac = sgn * (k / K) * h
            coords_k = coords0 + frac * q
            mo_guess_mol[0] = mol_cur
            mol_k, mc_k = build_at(coords_k, mo_cur, ref_seed)
            e_k = float(mc_k.e_states[0])
            disp = frac  # signed displacement magnitude along q
            e_branch = e_ref + g_analytic * disp  # linear basin-A prediction
            dev = e_k - e_branch
            ladder.append({"k": k, "frac_bohr": disp, "e0": e_k,
                           "e_branch_pred": e_branch, "dev_from_branch": dev,
                           "on_branch": bool(abs(dev) < 5.0e-4)})
            log(f"   k={k}/{K} disp={disp:+.2e} e0={e_k:.8f} "
                f"branch_pred={e_branch:.8f} dev={dev:+.2e} "
                f"on_branch={abs(dev)<5e-4}")
            mo_cur = np.array(mc_k.mo_coeff)
            mol_cur = mol_k
        ends[sgn] = ladder[-1]["e0"]
        out[f"ladder_{'plus' if sgn>0 else 'minus'}"] = ladder

    g_central = (ends[+1.0] - ends[-1.0]) / (2.0 * h)
    out["e_plus"], out["e_minus"] = ends[+1.0], ends[-1.0]
    out["g_fd_central_branch_followed"] = g_central
    out["g_fd_forward"] = (ends[+1.0] - e_ref) / h
    out["abs_err_central_vs_analytic"] = abs(g_central - g_analytic)
    log(f"BRANCH-FOLLOWED central FD single_cc = {g_central:.6f}  "
        f"vs analytic {g_analytic:.6f}  abs_err = {out['abs_err_central_vs_analytic']:.3e}")
    out["verdict"] = ("ANALYTIC CONFIRMED -- naive FD single_cc was a branch-jump artifact"
                      if out["abs_err_central_vs_analytic"] < 5.0e-3 else
                      "still off -- investigate")
    log(f"VERDICT: {out['verdict']}")
    json.dump(out, open(out_path, "w"), indent=2, default=str)
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
