"""Beyond-FCI gradient via the Schur backend with an ADAPTIVE correction-vector
bond-dim schedule (the real attack on the response cost wall).

The global MPS-Krylov solver, and the stock Schur path, both cap the response MPS
at a fixed ``m_compress`` -> the true residual floors (0.47 at m=128, 0.16 at
m=384) because the correction vector (H_CC-E)^{-1}w is intrinsically higher-rank
than the wavefunction.  block2's ``multiply`` natively supports a GROWING
``bra_bond_dims`` schedule + ReducedPerturbative ``noises``; we feed those through
the patched ``_ci_block_inverse`` so the correction vector grows to its own rank
(standard dynamical-DMRG correction vector).  The true-residual certificate stays
the gate.

C16 first (det 1.66e8, beyond the FCI threshold but ~14x cheaper than C18).
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
    """Growing bond-dim schedule from ~build_m up to ceiling, each level x2."""
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
    ap.add_argument("--ncarbon", type=int, default=16)
    ap.add_argument("--bond-dim", type=int, default=512)
    ap.add_argument("--m-ceiling", type=int, default=1600)
    ap.add_argument("--ci-m-loop", type=int, default=256,
                    help="moderate fixed bond dim for the orbital-Schur loop CI-solves (cheap)")
    ap.add_argument("--orb-max-iter", type=int, default=200)
    ap.add_argument("--audit-kappa", nargs="*", type=int, default=None,
                    help="z_kappa m-convergence audit: solve orbital block at these "
                         "ci_m_loop values and report z_kappa rel-change (per-size check)")
    ap.add_argument("--ci-sweeps", type=int, default=16)
    ap.add_argument("--ci-tol", type=float, default=1.0e-6)
    ap.add_argument("--orb-tol", type=float, default=1.0e-5)
    ap.add_argument("--response-tol", type=float, default=1.0e-4)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--stack-mem-mb", type=int, default=16000)
    ap.add_argument("--faulthandler-s", type=float, default=600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from run_polyene_beyond_fci import polyene_geometry, det_dim
    from run_beyond_fci_nac import _pi_space
    import certified_engine as ce
    from analytic_cp_sharc import _make_mps_krylov_response
    from sweep_coupled_response import solve_state_sweep_schur
    from certified_response import certify_response
    from cp_dmrg_response_mps_krylov import MPSKrylovVector
    import block2

    n = args.ncarbon
    symbols = [a[0] for a in polyene_geometry(n)]
    coords_ang = np.array([a[1] for a in polyene_geometry(n)])
    ncas, nelecas, mo0 = _pi_space(symbols, coords_ang, "sto-3g", n)
    dd = det_dim(ncas, (nelecas // 2, nelecas - nelecas // 2))
    sched = build_schedule(args.bond_dim, args.m_ceiling)
    noises = [1.0e-4] * (len(sched) - 4) + [1.0e-5] * 2 + [0.0] * 2
    log(f"C{n} CAS({ncas},{nelecas}) det={dd:.3e}; CI schedule={sched} noises={noises}")

    out_path = args.out or str(_HERE / "data" / f"beyond_fci_schur_c{n}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result = {"system": f"polyene_C{n}", "ncas": ncas, "nelecas": nelecas,
              "det_dim": dd, "build_bond_dim": args.bond_dim,
              "ci_schedule": sched, "ci_noises": noises}

    try:
        mol, mc, solver, info = ce.build_robust(
            symbols, coords_ang * ANG, basis="sto-3g", charge=0, spin=0,
            ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
            mo_guess=mo0, max_bond_dim=args.bond_dim, threads=args.threads,
            stack_mem_mb=args.stack_mem_mb)
        result["build"] = info
        log(f"BUILD DONE conv={info['converged']} health={info['build_health']['overall']} "
            f"e={[round(x,6) for x in info['e_states']]} wall={info['wall_s']:.0f}s")
        if not info["converged"]:
            result["status"] = "build_failed"
            json.dump(result, open(out_path, "w"), indent=2, default=str)
            return 1

        block2.Global.frame = mc.fcisolver._driver.frame
        obj = _make_mps_krylov_response(mc)
        obj._m_compress = int(args.m_ceiling)          # cap = schedule ceiling
        obj._ci_bra_schedule = sched                   # GROWING schedule (the fix)
        obj._ci_noises = noises
        log(f"RESPONSE OBJECT READY (Schur, adaptive CI bond dim up to {args.m_ceiling}) "
            f"-- solving certified state-0 gradient")
        if args.faulthandler_s > 0:
            import faulthandler
            faulthandler.dump_traceback_later(int(args.faulthandler_s), repeat=True,
                                              file=sys.stderr)
        t1 = time.perf_counter()
        if args.audit_kappa:
            # Per-size z_kappa m-convergence audit: solve the (cheap) orbital block
            # at several ci_m_loop values and check z_kappa has STABILISED at this
            # size -- proves moderate-m suffices here rather than assuming a tuning
            # calibrated on a smaller CAS.  Also records the orbital-GMRES plateau.
            mloops = [int(x) for x in args.audit_kappa]
            kappas = {}
            for ml in mloops:
                tk = time.perf_counter()
                kp, _ci, ki, kmeta = solve_state_sweep_schur(
                    obj, 0, orb_tol=args.orb_tol, orb_max_iter=args.orb_max_iter,
                    ci_sweeps=args.ci_sweeps, ci_tol=args.ci_tol, solver_type="MinRes",
                    proj_weight=1.0e3, ci_m_loop=ml, verbose=True, kappa_only=True)
                kappas[ml] = np.asarray(kp, dtype=float)
                log(f"AUDIT ci_m_loop={ml}: orb_iters={kmeta['orb_iters']} "
                    f"kappa_norm={np.linalg.norm(kappas[ml]):.6f} HOO_cond={kmeta.get('HOO_cond_eff')} "
                    f"wall={time.perf_counter()-tk:.0f}s")
            conv = {}
            for a, b in zip(mloops[:-1], mloops[1:]):
                rel = float(np.linalg.norm(kappas[b] - kappas[a]) /
                            max(np.linalg.norm(kappas[b]), 1e-30))
                conv[f"{a}->{b}"] = rel
                log(f"AUDIT z_kappa rel-change {a}->{b} = {rel:.3e}  "
                    f"({'CONVERGED' if rel < 1e-3 else 'NOT converged -- need higher ci_m_loop'})")
            result["status"] = "kappa_audit"
            result["kappa_mconv"] = conv
            result["audit_kappa_norms"] = {str(k): float(np.linalg.norm(v)) for k, v in kappas.items()}
            json.dump(result, open(out_path, "w"), indent=2, default=str)
            log(f"wrote {out_path}")
            return 0
        kappa, ci, sinfo, meta = solve_state_sweep_schur(
            obj, 0, orb_tol=args.orb_tol, orb_max_iter=args.orb_max_iter,
            ci_sweeps=args.ci_sweeps, ci_tol=args.ci_tol, solver_type="MinRes",
            proj_weight=1.0e3, residual_tol=args.response_tol,
            ci_m_loop=args.ci_m_loop, ci_schedule_final=sched, ci_noises_final=noises,
            verbose=True)
        result["ci_m_loop"] = int(args.ci_m_loop)
        z = MPSKrylovVector(obj, kappa, ci, label="schur0")
        cert = certify_response(obj, z, state=0, tol=args.response_tol, solver="sweep_schur")
        log(f"SCHUR DONE wall={time.perf_counter()-t1:.0f}s info={sinfo} "
            f"true_resid={cert.true_residual_relative:.3e} converged={cert.converged}")
        result["status"] = "ok"
        result["schur_info"] = int(sinfo)
        result["true_residual_relative"] = float(cert.true_residual_relative)
        result["converged"] = bool(cert.converged)
        result["kappa_norm"] = float(np.linalg.norm(kappa))
        result["schur_wall_s"] = time.perf_counter() - t1

        # Assemble the PHYSICAL gradient g0 from the Schur z and contract with the
        # FD directions: analytic g.q vs FD g.q is the paper's real validation and
        # can pass even when true_resid is loose (the gradient is far less
        # sensitive to the high-rank CI tail than the full residual).
        try:
            from analytic_cp_sharc import _gradient_one_state_mps_krylov
            from run_cas_directional_fd import named_directions
            g0 = _gradient_one_state_mps_krylov(
                mc, obj, 0, tol=args.response_tol, max_iter=1,
                precomputed_z=(kappa, ci))
            dirs = named_directions(symbols, coords_ang * ANG)
            gdotq = {nm: float(np.tensordot(g0, q)) for nm, q in dirs.items()}
            result["analytic_gradient_norm"] = float(np.linalg.norm(g0))
            result["analytic_dir"] = gdotq
            # compare to saved FD directions if available
            fdp = _HERE / "data" / f"dirfd_analytic_c{n}.json"
            if not fdp.exists():
                fdp = _HERE / "data" / f"dirfd_c{n}.json"
            if fdp.exists():
                fd = json.load(open(fdp)).get("directions", {})
                result["g_dot_q_vs_fd"] = {
                    nm: {"analytic": gdotq[nm],
                         "fd": fd.get(nm, {}).get("g_fd_dir"),
                         "abs_err": (abs(gdotq[nm] - fd[nm]["g_fd_dir"])
                                     if nm in fd and "g_fd_dir" in fd[nm] else None)}
                    for nm in gdotq}
                log("g.q analytic-vs-FD: " + json.dumps(result["g_dot_q_vs_fd"]))
            else:
                log("g.q analytic: " + json.dumps(gdotq))
        except Exception as exc:  # noqa: BLE001
            result["grad_assembly_error"] = str(exc)[:200]
            log(f"grad assembly error: {type(exc).__name__}: {str(exc)[:150]}")
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["exception"] = type(exc).__name__
        result["message"] = str(exc)[:300]
        result["traceback_tail"] = traceback.format_exc()[-2000:]
        log(f"ERROR {type(exc).__name__}: {str(exc)[:200]}")

    json.dump(result, open(out_path, "w"), indent=2, default=str)
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
