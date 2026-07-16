"""Clean beyond-FCI gradient validation anchored to ONE shared reference.

Both the analytic CP-DMRG-CASSCF directional gradient AND the finite-difference
directional gradient are evaluated from the IDENTICAL persisted lowest-basin
reference orbitals (build_reference_lowest), so they sit at the same SA-CASSCF
stationary point on the multi-valued polyene landscape -- the only way to make the
comparison airtight.

Analytic: ce.build_robust(mo_guess=ref_mo, seed pinned) -> Schur response -> g0.q.
FD: BRANCH-FOLLOWING continuation -- a ladder of K small warm-started steps from the
reference to +-h, each SA-CASSCF build warm-started (orbitals) from the previous
step, so the optimizer cannot escape the reference basin along soft directions
(the failure mode that made the naive single_cc FD branch-jump to a higher basin).
Each step logs its deviation from the linear basin branch so an escape is visible.
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
SH = _HERE.parents[1] / "sharc_interface"
for _p in (str(_HERE), str(DEV), str(SH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ANG = 1.8897261246257702
_T0 = time.perf_counter()


def log(m):
    print(f"[{time.perf_counter() - _T0:8.1f}s] {m}", flush=True)


def build_schedule(build_m, ceiling):
    levels = []
    m = int(build_m)
    while m < int(ceiling):
        levels.append(m)
        m = int(m * 1.5)
    levels.append(int(ceiling))
    sched = []
    for L in levels:
        sched += [L, L]
    return sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, default=20)
    ap.add_argument("--bond-dim", type=int, default=800)
    ap.add_argument("--m-ceiling", type=int, default=1600)
    ap.add_argument("--ci-m-loop", type=int, default=256)
    ap.add_argument("--orb-max-iter", type=int, default=20)
    ap.add_argument("--ci-sweeps", type=int, default=16)
    ap.add_argument("--ci-tol", type=float, default=1e-6)
    ap.add_argument("--orb-tol", type=float, default=1e-4)
    ap.add_argument("--response-tol", type=float, default=1e-4)
    ap.add_argument("--h-bohr", type=float, default=1e-3)
    ap.add_argument("--ladder-K", type=int, default=4, help="continuation steps to +-h")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--stack-mem-mb", type=int, default=24000)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--skip-analytic", action="store_true")
    ap.add_argument("--skip-fd", action="store_true")
    args = ap.parse_args()

    import block2
    from pyscf import gto
    import certified_engine as ce
    from analytic_cp_sharc import (_make_mps_krylov_response,
                                   _gradient_one_state_mps_krylov)
    from sweep_coupled_response import solve_state_sweep_schur
    from certified_response import certify_response
    from cp_dmrg_response_mps_krylov import MPSKrylovVector
    from run_cas_directional_fd import build_progressive, named_directions
    from run_polyene_beyond_fci import det_dim
    import fd_validation as fdv
    from build_reference_lowest import build_or_load_reference, TIGHT_SCHEDULE

    n = args.ncarbon
    out_path = _HERE / "data" / f"beyond_fci_clean_c{n}.json"
    result = {"system": f"polyene_C{n}", "h_bohr": args.h_bohr, "ladder_K": args.ladder_K}

    # --- shared reference (lowest basin, persisted) ---
    ref = build_or_load_reference(n, basis="sto-3g", seeds=args.seeds,
                                  threads=args.threads, stack_mem_mb=args.stack_mem_mb)
    symbols, coords0 = ref["symbols"], ref["coords_bohr"]
    ncas, nelecas, mo_ref, e_ref, ref_seed = (ref["ncas"], ref["nelecas"],
                                              ref["mo_coeff"], ref["e0"], ref["seed"])
    result["reference"] = {"e0": e_ref, "seed": ref_seed, "cached": ref["from_cache"]}
    dd = det_dim(ncas, (nelecas // 2, nelecas - nelecas // 2))
    result["det_dim"] = dd
    log(f"C{n} CAS({ncas},{nelecas}) det={dd:.3e}; shared reference e0={e_ref} seed={ref_seed}")
    dirs = named_directions(symbols, coords0)

    # ------------------------------------------------ ANALYTIC g0.q at the reference
    if not args.skip_analytic:
        try:
            if ref_seed is not None:
                block2.Random.rand_seed(int(ref_seed))
            mol, mc, solver, info = ce.build_robust(
                symbols, coords0, basis="sto-3g", charge=0, spin=0,
                ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
                mo_guess=mo_ref, max_bond_dim=args.bond_dim, threads=args.threads,
                stack_mem_mb=args.stack_mem_mb)
            e0_an = info["e_states"][0]
            drift = abs(e0_an - e_ref) if e_ref is not None else None
            log(f"ANALYTIC build e0={e0_an:.9f} drift_from_ref={drift}")
            result["analytic_build_e0"] = e0_an
            result["analytic_build_drift"] = drift
            if drift is not None and drift > 1e-5:
                log(f"WARNING: analytic build drifted {drift:.2e} from reference basin")
            block2.Global.frame = mc.fcisolver._driver.frame
            obj = _make_mps_krylov_response(mc)
            sched = build_schedule(args.bond_dim, args.m_ceiling)
            noises = [1e-4] * (len(sched) - 4) + [1e-5] * 2 + [0.0] * 2
            obj._m_compress = int(args.m_ceiling)
            obj._ci_bra_schedule = sched
            obj._ci_noises = noises
            kappa, ci, sinfo, meta = solve_state_sweep_schur(
                obj, 0, orb_tol=args.orb_tol, orb_max_iter=args.orb_max_iter,
                ci_sweeps=args.ci_sweeps, ci_tol=args.ci_tol, solver_type="MinRes",
                proj_weight=1e3, residual_tol=args.response_tol,
                ci_m_loop=args.ci_m_loop, ci_schedule_final=sched,
                ci_noises_final=noises, verbose=True)
            z = MPSKrylovVector(obj, kappa, ci, label="clean0")
            cert = certify_response(obj, z, state=0, tol=args.response_tol,
                                    solver="sweep_schur")
            result["true_residual_relative"] = float(cert.true_residual_relative)
            g0 = _gradient_one_state_mps_krylov(mc, obj, 0, tol=args.response_tol,
                                                max_iter=1, precomputed_z=(kappa, ci))
            result["analytic_dir"] = {nm: float(np.tensordot(g0, q))
                                      for nm, q in dirs.items()}
            log(f"ANALYTIC true_resid={cert.true_residual_relative:.3e} "
                f"g.q={json.dumps(result['analytic_dir'])}")
        except Exception as exc:  # noqa: BLE001
            result["analytic_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            result["analytic_tb"] = traceback.format_exc()[-1500:]
            log(f"ANALYTIC ERROR: {result['analytic_error']}")
        json.dump(result, open(out_path, "w"), indent=2, default=str)

    # --------------------------------------------- BRANCH-FOLLOWING continuation FD
    if not args.skip_fd:
        h, K = args.h_bohr, args.ladder_K
        g_an = result.get("analytic_dir", {})
        result["fd_dir"] = {}
        mo_g_mol = [None]

        def build_at(coords_bohr, mo_guess, seed):
            if seed is not None:
                block2.Random.rand_seed(int(seed))
            mol = gto.M(atom=[(symbols[i], tuple(coords_bohr[i]))
                              for i in range(len(symbols))],
                        basis="sto-3g", unit="Bohr", verbose=0)
            mg = mo_guess
            if mg is not None and mo_g_mol[0] is not None:
                mg = fdv.project_mo_to_new_geometry(mo_g_mol[0], mol, mo_guess)[0]
            _mol, mc_s, _sol, _bl = build_progressive(
                symbols, coords_bohr, "sto-3g", ncas, nelecas,
                m_schedule=TIGHT_SCHEDULE, mo_guess=mg,
                threads=args.threads, stack_mem_mb=args.stack_mem_mb)
            return mol, mc_s

        # reference build (from persisted ref mo, pinned seed) -> the FD anchor
        mol_ref, mc_ref = build_at(coords0, mo_ref, ref_seed)
        e_ref_fd = float(mc_ref.e_states[0])
        mo_ref_fd = np.array(mc_ref.mo_coeff)
        result["fd_reference_e0"] = e_ref_fd
        log(f"FD reference e0={e_ref_fd:.9f}")

        for nm, q in dirs.items():
            try:
                ga = g_an.get(nm)  # analytic slope for the branch prediction
                ends = {}
                ladders = {}
                for sgn in (+1.0, -1.0):
                    mo_cur, mol_cur = mo_ref_fd, mol_ref
                    lad = []
                    for k in range(1, K + 1):
                        frac = sgn * (k / K) * h
                        coords_k = coords0 + frac * q
                        mo_g_mol[0] = mol_cur
                        mol_k, mc_k = build_at(coords_k, mo_cur, ref_seed)
                        e_k = float(mc_k.e_states[0])
                        pred = (e_ref_fd + ga * frac) if ga is not None else None
                        dev = (e_k - pred) if pred is not None else None
                        lad.append({"k": k, "frac": frac, "e0": e_k, "dev": dev})
                        mo_cur, mol_cur = np.array(mc_k.mo_coeff), mol_k
                    ends[sgn] = lad[-1]["e0"]
                    ladders[sgn] = lad
                g_fd = (ends[+1.0] - ends[-1.0]) / (2.0 * h)
                rec = {"g_fd_central": g_fd,
                       "g_fd_forward": (ends[+1.0] - e_ref_fd) / h,
                       "ladder_plus": ladders[+1.0], "ladder_minus": ladders[-1.0]}
                if ga is not None:
                    rec["analytic"] = ga
                    rec["abs_err"] = abs(g_fd - ga)
                result["fd_dir"][nm] = rec
                log(f"FD {nm}: central={g_fd:.6f} forward={rec['g_fd_forward']:.6f}"
                    + (f" analytic={ga:.6f} abs_err={rec['abs_err']:.3e}"
                       if ga is not None else ""))
                json.dump(result, open(out_path, "w"), indent=2, default=str)
            except Exception as exc:  # noqa: BLE001
                result["fd_dir"][nm] = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
                log(f"FD {nm} ERROR: {result['fd_dir'][nm]['error']}")

    json.dump(result, open(out_path, "w"), indent=2, default=str)
    log(f"wrote {out_path}")
    # summary
    log("=== SUMMARY analytic vs branch-followed FD ===")
    for nm in dirs:
        a = result.get("analytic_dir", {}).get(nm)
        f = result.get("fd_dir", {}).get(nm, {})
        log(f"  {nm:12s} analytic={a} fd_central={f.get('g_fd_central')} "
            f"abs_err={f.get('abs_err')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
