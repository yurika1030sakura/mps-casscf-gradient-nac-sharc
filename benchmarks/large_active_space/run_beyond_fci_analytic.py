"""Critical experiment: does the ANALYTIC MPS response z-solve complete at a
genuinely beyond-FCI active space?

The paper's central claim is a *certified analytic* SA-DMRG-CASSCF response past
the FCI wall.  The directional-FD runs left the analytic comparison unfinished at
C20 (analytic_done=False), so this driver isolates the analytic response: it
builds one reference-geometry polyene CAS(n,n) pi space, then solves the certified
analytic gradient[0] and NAC(0,1) with the MPS-Krylov backend and reports each
true-residual certificate -- with stage timestamps so a stall is pinned to the
build vs the response, and with NO cross-geometry overlap (the FD-validation path
whose orbital-reflection crash is a separate issue and must not mask this result).

C18 CAS(18,18): det = C(18,9)^2 = 2.36e9 (dense FCI vector ~19 GB, infeasible).
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


def log(msg):
    print(f"[{time.perf_counter() - _T0:8.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, default=18)
    ap.add_argument("--bond-dim", type=int, default=800)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--stack-mem-mb", type=int, default=16000)
    ap.add_argument("--out", default=None)
    # Phase-2 response-solver levers (diagnosis: GMRES O(iter^2) + no preconditioner
    # + m_compress=build-M is the beyond-FCI stall).  Defaults = the fixed config.
    ap.add_argument("--linear-solver", default="cr",
                    help="cr (conjugate residual, short-recurrence) | bicgstab | gmres")
    ap.add_argument("--initial-guess", default="hcc-inverse",
                    help="hcc-inverse (CI-block preconditioned) | gmres-recycle | zero")
    ap.add_argument("--m-compress", type=int, default=256,
                    help="response bond dim (below the build M to cut per-matvec cost)")
    ap.add_argument("--response-tol", type=float, default=1.0e-4,
                    help="response tolerance (FD is only h=1e-3 accurate)")
    ap.add_argument("--faulthandler-s", type=float, default=180.0,
                    help="dump all-thread stacks to stderr every N s during the solve (0=off)")
    ap.add_argument("--fit-sweeps", type=int, default=4,
                    help="MPS addition/sigma fit sweeps in the response (was 10; 4 is plenty)")
    ap.add_argument("--max-iter", type=int, default=80,
                    help="max Krylov iterations for the response solve")
    args = ap.parse_args()

    from run_polyene_beyond_fci import polyene_geometry, det_dim
    from run_beyond_fci_nac import _pi_space
    import certified_engine as ce
    from analytic_cp_sharc import _make_mps_krylov_response
    from auto_response import compute_all_responses_certified
    import block2

    n = args.ncarbon
    symbols = [at[0] for at in polyene_geometry(n)]
    coords_ang = np.array([at[1] for at in polyene_geometry(n)])
    ncas, nelecas, mo0 = _pi_space(symbols, coords_ang, "sto-3g", n)
    dd = det_dim(ncas, (nelecas // 2, nelecas - nelecas // 2))
    log(f"C{n} CAS({ncas},{nelecas}) det={dd:.3e} (dense FCI vector ~{dd * 8 / 1e9:.0f} GB) -- building")

    out_path = args.out or str(_HERE / "data" / f"beyond_fci_analytic_c{n}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result = {"system": f"polyene_C{n}", "ncas": ncas, "nelecas": nelecas,
              "det_dim": dd, "bond_dim": args.bond_dim}

    try:
        mol, mc, solver, info = ce.build_robust(
            symbols, coords_ang * ANG, basis="sto-3g", charge=0, spin=0,
            ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
            mo_guess=mo0, max_bond_dim=args.bond_dim, threads=args.threads,
            stack_mem_mb=args.stack_mem_mb)
        result["build"] = info
        log(f"BUILD DONE converged={info['converged']} health={info['build_health']['overall']} "
            f"e={[round(x, 6) for x in info['e_states']]} wall={info['wall_s']:.0f}s")
        if not info["converged"]:
            result["status"] = "build_failed"
            json.dump(result, open(out_path, "w"), indent=2, default=str)
            log(f"build not converged; wrote {out_path}")
            return 1

        # Phase-2 response config: the factory reads these off the solver.
        mc.fcisolver.response_linear_solver = args.linear_solver
        mc.fcisolver.response_initial_guess = args.initial_guess
        mc.fcisolver.response_m_compress = int(args.m_compress)
        mc.fcisolver.mps_fit_sweeps = int(args.fit_sweeps)  # response fit sweeps (not the build)
        result["response_config"] = {
            "linear_solver": args.linear_solver, "initial_guess": args.initial_guess,
            "m_compress": int(args.m_compress), "response_tol": args.response_tol,
            "fit_sweeps": int(args.fit_sweeps), "max_iter": int(args.max_iter)}
        block2.Global.frame = mc.fcisolver._driver.frame
        obj = _make_mps_krylov_response(mc)
        log(f"RESPONSE OBJECT READY (solver={args.linear_solver} guess={args.initial_guess} "
            f"m_compress={args.m_compress} tol={args.response_tol:.1e}) -- solving grad[0]+NAC(0,1)")
        # Locate any stall: dump the live Python stack of every thread to stderr
        # on a timer, so a frozen-but-compute-bound operation reveals its exact
        # call site instead of silently grinding.
        if args.faulthandler_s > 0:
            import faulthandler
            faulthandler.dump_traceback_later(int(args.faulthandler_s),
                                              repeat=True, file=sys.stderr)
        t1 = time.perf_counter()
        certs = compute_all_responses_certified(
            obj, gradient_states=[0], nac_pairs=[(0, 1)],
            tol=args.response_tol, cert_tol=args.response_tol * 10.0,
            max_iter=args.max_iter, verbose=True)
        log(f"RESPONSE DONE wall={time.perf_counter() - t1:.0f}s")

        result["certs"] = {}
        for key, pair in certs.items():
            _z, cert = pair
            cd = cert.to_dict()
            hv = cert.health().overall
            tr = cd.get("true_residual_relative")
            log(f"  {key}: health={hv} true_residual_relative={tr}")
            result["certs"][str(key)] = {"health": hv, "certificate": cd}
        result["status"] = "ok"
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
