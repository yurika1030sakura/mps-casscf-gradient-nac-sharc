"""Airtight beyond-FCI 承重 at C22/C24: analytic gradient vs BRANCH-FOLLOWED FD,
both at ONE shared lowest-basin reference.

C22/C24 SA-CASSCF surfaces are strongly multi-valued (the C20 single_cc lesson at
larger scale): the naive directional FD displaced builds jump stationary points, so
beyond_fci_schur_c22/c24.json's g.q-vs-saved-FD are garbage (errors 0.4-1.7) even
though the analytic gradient is fine.  This driver removes the confound:
  1. build ONE reference (build_progressive, multi-seed -> lowest basin), mo_A, e_ref;
  2. analytic g.q via the split-m Schur response on that reference;
  3. branch-followed continuation FD for each direction: a ladder of K small warm-
     started steps to +-h that keeps the optimizer in basin A (small steps + orbital
     warm-start), with per-step deviation from the analytic-predicted branch logged
     so any escape is visible;
  4. compare g_central(branch-followed) to the analytic g.q -- at the SAME basin.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
DEV = _HERE.parents[1] / "src" / "dmrg_analytic_dev"
SH = _HERE.parents[1] / "sharc_interface"
for _p in (str(_HERE), str(DEV), str(SH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import faulthandler
faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
ANG = 1.8897261246257702
_T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-_T0:8.1f}s] {m}", flush=True)

def build_schedule(build_m, ceiling):
    levels, m = [], int(build_m)
    while m < int(ceiling):
        levels.append(m); m = int(m * 1.5)
    levels.append(int(ceiling))
    return [L for L in levels for _ in (0, 1)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, default=22)
    ap.add_argument("--directions", nargs="+", default=["central_cc", "bla"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--K", type=int, default=4, help="continuation ladder steps to +-h")
    ap.add_argument("--h-bohr", type=float, default=1.0e-3)
    ap.add_argument("--bond-dim", type=int, default=800)
    ap.add_argument("--m-ceiling", type=int, default=800)
    ap.add_argument("--ci-m-loop", type=int, default=256)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--stack-mem-mb", type=int, default=24000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import block2
    from pyscf import gto
    from run_polyene_beyond_fci import polyene_geometry, select_pi_active, det_dim
    from run_cas_directional_fd import build_progressive, named_directions
    from analytic_cp_sharc import _make_mps_krylov_response, _gradient_one_state_mps_krylov
    from sweep_coupled_response import solve_state_sweep_schur
    import fd_validation as fdv

    n = args.ncarbon
    symbols = [a[0] for a in polyene_geometry(n)]
    coords0 = np.array([a[1] for a in polyene_geometry(n)]) * ANG     # bohr
    ncas = nelecas = n
    dd = det_dim(ncas, (ncas // 2, ncas - ncas // 2))
    h = args.h_bohr
    m_schedule = [(256, 1e-8, 1e-4, 40), (512, 1e-10, 1e-5, 60),
                  (int(args.bond_dim), 1e-12, 1e-6, 100)]
    sched = build_schedule(args.bond_dim, args.m_ceiling)
    noises = [1.0e-4] * max(0, len(sched) - 4) + [1.0e-5] * 2 + [0.0] * 2
    out_path = args.out or str(_HERE / "data" / f"polyene_continuation_c{n}.json")
    result = {"system": f"polyene_C{n}", "ncas": ncas, "det_dim": dd,
              "beyond_fci": bool(dd >= 5.0e7), "h_bohr": h, "K": args.K,
              "directions": args.directions}
    log(f"C{n} CAS({ncas},{ncas}) det={dd:.3e} dirs={args.directions}")

    def pi_mo(c_bohr):
        mol = gto.M(atom=[(symbols[i], tuple(c_bohr[i])) for i in range(len(symbols))],
                    basis="sto-3g", unit="Bohr", verbose=0)
        from pyscf import scf
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        return select_pi_active(mol, mf, n)[2]

    # 1) reference: multi-seed -> lowest basin
    best = None
    for s in args.seeds:
        block2.Random.rand_seed(int(s))
        t = time.perf_counter()
        molX, mcX, solX, blogX = build_progressive(
            symbols, coords0, "sto-3g", ncas, nelecas, m_schedule=m_schedule,
            mo_guess=pi_mo(coords0), threads=args.threads, stack_mem_mb=args.stack_mem_mb)
        e0 = float(blogX["stages"][-1]["e_states"][0])
        log(f"  seed {s}: e0={e0:.8f} wall={time.perf_counter()-t:.0f}s")
        if best is None or e0 < best[0]:
            best = (e0, mcX, molX, np.array(mcX.mo_coeff))
    e_ref, mc, mol0, mo_A = best
    result["e_ref"] = e_ref
    log(f"REFERENCE basin e0={e_ref:.8f}")

    # 2) analytic g.q via split-m Schur on this reference
    block2.Global.frame = mc.fcisolver._driver.frame
    obj = _make_mps_krylov_response(mc)
    obj._m_compress = int(args.m_ceiling); obj._ci_bra_schedule = sched; obj._ci_noises = noises
    t1 = time.perf_counter()
    kappa, ci, sinfo, meta = solve_state_sweep_schur(
        obj, 0, orb_tol=1e-4, orb_max_iter=20, ci_sweeps=16, ci_tol=1e-6,
        solver_type="MinRes", proj_weight=1e3, residual_tol=1e-4,
        ci_m_loop=args.ci_m_loop, ci_schedule_final=sched, ci_noises_final=noises, verbose=True)
    g0 = _gradient_one_state_mps_krylov(mc, obj, 0, tol=1e-4, max_iter=1, precomputed_z=(kappa, ci))
    dirs = named_directions(symbols, coords0)
    g_analytic = {k: float(np.tensordot(g0, dirs[k])) for k in args.directions}
    result["analytic_dir"] = g_analytic
    log(f"analytic g.q: {{{', '.join(f'{k}:{v:.6f}' for k,v in g_analytic.items())}}} (Schur wall {time.perf_counter()-t1:.0f}s)")

    # 3) branch-followed continuation FD per direction
    result["fd"] = {}
    for dname in args.directions:
        q = dirs[dname]; ga = g_analytic[dname]
        ends = {}
        for sgn in (+1.0, -1.0):
            mo_cur, mol_cur = mo_A, mol0
            ladder = []
            for k in range(1, args.K + 1):
                frac = sgn * (k / args.K) * h
                ck = coords0 + frac * q
                mol_k = gto.M(atom=[(symbols[i], tuple(ck[i])) for i in range(len(symbols))],
                              basis="sto-3g", unit="Bohr", verbose=0)
                mg = fdv.project_mo_to_new_geometry(mol_cur, mol_k, mo_cur)[0]
                _m, mck, _s, blk = build_progressive(
                    symbols, ck, "sto-3g", ncas, nelecas, m_schedule=m_schedule,
                    mo_guess=mg, threads=args.threads, stack_mem_mb=args.stack_mem_mb)
                e_k = float(blk["stages"][-1]["e_states"][0])
                dev = e_k - (e_ref + ga * frac)
                ladder.append({"k": k, "frac": frac, "e0": e_k, "dev_from_branch": dev,
                               "on_branch": bool(abs(dev) < 5e-4)})
                log(f"  {dname} sgn={sgn:+.0f} k={k}/{args.K} disp={frac:+.2e} e0={e_k:.8f} dev={dev:+.2e} on_branch={abs(dev)<5e-4}")
                mo_cur, mol_cur = np.array(mck.mo_coeff), mol_k
            ends[sgn] = ladder[-1]["e0"]
            result["fd"].setdefault(dname, {})[f"ladder_{'p' if sgn>0 else 'm'}"] = ladder
        g_central = (ends[+1.0] - ends[-1.0]) / (2.0 * h)
        result["fd"][dname]["g_fd_central"] = g_central
        result["fd"][dname]["abs_err_vs_analytic"] = abs(g_central - ga)
        log(f"{dname}: branch-followed central FD={g_central:+.6f} vs analytic {ga:+.6f} abs_err={abs(g_central-ga):.3e}")
        json.dump(result, open(out_path, "w"), indent=2, default=str)
    result["status"] = "ok"
    json.dump(result, open(out_path, "w"), indent=2, default=str)
    log(f"wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
