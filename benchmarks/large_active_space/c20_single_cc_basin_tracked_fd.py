"""Basin-tracked FD of the C20 single_cc directional gradient.

DIAGNOSIS (proven from dirfd_c20.json displaced energies): the naive FD single_cc
is corrupted because BOTH +-h displaced SA-CASSCF builds drifted to the HIGHER
stationary point (e_disp ~ +2.7/+1.6 mEh above the low reference e0=-755.281495,
i.e. the ~3 mEh basin gap), even though the analytic gradient (-0.165) implies the
LOW basin continues to e(+h) ~ -755.28166 -- LOWER than the build's -755.27882.
So the displaced builds failed to find the global-min (low) basin along this soft
bond-alternation direction; the analytic value is the correct low-basin derivative.

This script re-runs the single_cc +-h displaced builds with BASIN TRACKING:
  - reference: multi-seed (1,2,3) cold build, take the LOWEST e0 -> basin A, save mo_A.
  - each displacement: warm-start orbitals from basin-A mo, AND try multiple seeds,
    take the LOWEST converged e0 (the global min at that geometry).
  - report whether a basin-A solution (e ~ eref +- g*h, ~-755.2817) was recovered,
    the active-space overlap to mo_A, and the resulting basin-tracked g_fd_dir.
Compare g_fd_dir to the analytic single_cc = -0.16458 (beyond_fci_schur_c20.json,
which itself built at the LOW basin e0=-755.2814945237).

If basin-tracked FD single_cc ~ -0.165  -> analytic CONFIRMED; naive FD was a
branch-jump artifact; the C20 beyond-FCI gradient承重 holds in all 3 directions.
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
    seeds = (1, 2, 3)
    # tight schedule (lowest-energy, most converged) -- matches run_seed_independence "tight"
    m_schedule = [(256, 1e-8, 1e-4, 40), (512, 1e-10, 1e-5, 60), (800, 1e-12, 1e-6, 150)]
    threads, stack = 16, 24000

    symbols = [a[0] for a in polyene_geometry(n)]
    coords0 = np.array([a[1] for a in polyene_geometry(n)]) * ANG  # bohr
    ncas = nelecas = n
    q = named_directions(symbols, coords0)["single_cc"]
    out = {"system": "polyene_C20", "direction": "single_cc", "h_bohr": h,
           "seeds": list(seeds), "m_schedule": m_schedule,
           "analytic_single_cc_lowbasin": -0.16458174441719800}
    out_path = _HERE / "data" / "c20_single_cc_basin_tracked.json"

    def multiseed_build(coords_bohr, mo_guess, label):
        """Build SA-CASSCF over seeds, return (best_e0, best_mo, per_seed, sigma)."""
        best = None
        per = []
        mol_b = gto.M(atom=[(symbols[i], tuple(coords_bohr[i])) for i in range(len(symbols))],
                      basis="sto-3g", unit="Bohr", verbose=0)
        for s in seeds:
            block2.Random.rand_seed(int(s))
            t = time.perf_counter()
            mol_s, mc_s, _sol, blog = build_progressive(
                symbols, coords_bohr, "sto-3g", ncas, nelecas, m_schedule=m_schedule,
                mo_guess=mo_guess, threads=threads, stack_mem_mb=stack)
            e0 = float(mc_s.e_states[0])
            conv = bool(mc_s.converged)
            per.append({"seed": s, "e0": e0, "converged": conv,
                        "wall_s": time.perf_counter() - t})
            log(f"   {label} seed={s}: e0={e0:.8f} conv={conv} "
                f"wall={per[-1]['wall_s']:.0f}s")
            if conv and (best is None or e0 < best[0]):
                best = (e0, np.array(mc_s.mo_coeff), mol_s, mc_s)
        return best, per

    # 1) reference at basin A (lowest of seeds), cold
    log("REFERENCE multi-seed cold build...")
    ref_best, ref_per = multiseed_build(coords0, None, "ref")
    e_ref = ref_best[0]
    mo_A = ref_best[1]
    mol_A = ref_best[2]
    out["e_ref_lowbasin"] = e_ref
    out["ref_per_seed"] = ref_per
    log(f"REFERENCE basin-A e0={e_ref:.8f} (expect ~ -755.2814945)")

    # 2) displaced builds, warm-started from basin-A orbitals, multi-seed -> lowest
    e_disp = {}
    for sgn in (+1.0, -1.0):
        disp = coords0 + sgn * h * q
        mol_d = gto.M(atom=[(symbols[i], tuple(disp[i])) for i in range(len(symbols))],
                      basis="sto-3g", unit="Bohr", verbose=0)
        mo_guess = fdv.project_mo_to_new_geometry(mol_A, mol_d, mo_A)[0]
        log(f"DISPLACED sgn={sgn:+.0f} multi-seed warm-started from basin-A...")
        d_best, d_per = multiseed_build(disp, mo_guess, f"disp{sgn:+.0f}")
        e0 = d_best[0]
        sig = fdv.active_subspace_overlap(mol_A, mo_A, d_best[2], d_best[3].mo_coeff,
                                          ncas, d_best[3].ncore)
        e_disp[sgn] = e0
        dev = e0 - e_ref
        # smooth basin-A expectation: e(+-h) ~ e_ref +- g*h ; |dev| should be ~ g*h ~ 1.6e-4
        out[f"disp_{'plus' if sgn>0 else 'minus'}"] = {
            "e0": e0, "e0_minus_eref": dev, "active_overlap": float(sig),
            "per_seed": d_per,
            "in_basin_A": bool(abs(dev) < 5.0e-4)}  # 3x g*h tolerance
        log(f"DISPLACED sgn={sgn:+.0f}: e0={e0:.8f} dev_from_eref={dev:+.2e} "
            f"overlap={sig:.4f} in_basinA={abs(dev)<5e-4}")

    g_fd = (e_disp[+1.0] - e_disp[-1.0]) / (2.0 * h)
    out["g_fd_dir_basin_tracked"] = g_fd
    out["abs_err_vs_analytic"] = abs(g_fd - out["analytic_single_cc_lowbasin"])
    log(f"BASIN-TRACKED FD single_cc = {g_fd:.6f}  vs analytic -0.16458  "
        f"abs_err = {out['abs_err_vs_analytic']:.3e}")
    verdict = ("ANALYTIC CONFIRMED (naive FD was a branch-jump artifact)"
               if out["abs_err_vs_analytic"] < 5.0e-3 else
               "STILL MISMATCHED -- single_cc may be a true stationary-point fold")
    out["verdict"] = verdict
    log(f"VERDICT: {verdict}")
    json.dump(out, open(out_path, "w"), indent=2, default=str)
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
