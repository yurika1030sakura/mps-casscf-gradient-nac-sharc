"""Beyond-FCI analytic SA-DMRG-CASSCF gradient = FD on linear [n]acenes.

Generalizes the polyene 承重 (run_beyond_fci_schur.py) to a chemically distinct,
strongly-correlated 2D fused-ring pi family.  Reuses the generic Schur response and
the generic directional_fd; only geometry / pi-space / FD directions are acene-specific.

  --nrings 2  naphthalene CAS(10,10) det 6.3e4  -> also exact-FCI cross-check
  --nrings 3  anthracene  CAS(14,14) det 1.2e7
  --nrings 4  tetracene   CAS(18,18) det 2.4e9  (beyond FCI)
  --nrings 5  pentacene   CAS(22,22) det 5.0e11 (beyond FCI)

State-0 SA(2) gradient (the 承重 quantity).  FD displaced builds warm-start from the
reference orbitals; an energy in-basin check flags any branch jump (C20 lesson).
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
    ap.add_argument("--nrings", type=int, default=2)
    ap.add_argument("--bond-dim", type=int, default=800)
    ap.add_argument("--m-ceiling", type=int, default=800)
    ap.add_argument("--ci-m-loop", type=int, default=256)
    ap.add_argument("--orb-max-iter", type=int, default=20)
    ap.add_argument("--orb-tol", type=float, default=1.0e-4)
    ap.add_argument("--ci-sweeps", type=int, default=16)
    ap.add_argument("--ci-tol", type=float, default=1.0e-6)
    ap.add_argument("--response-tol", type=float, default=1.0e-4)
    ap.add_argument("--h-bohr", type=float, default=1.0e-3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--stack-mem-mb", type=int, default=24000)
    ap.add_argument("--fd", action="store_true", help="also run directional FD")
    ap.add_argument("--fci-check", action="store_true",
                    help="exact pyscf-FCI ground-truth gradient (naphthalene only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from pyscf import gto, scf, mcscf
    import certified_engine as ce
    from acene_systems import acene_geometry, acene_named_directions
    from run_polyene_beyond_fci import select_pi_active, det_dim
    from analytic_cp_sharc import _make_mps_krylov_response, _gradient_one_state_mps_krylov
    from sweep_coupled_response import solve_state_sweep_schur
    from certified_response import certify_response
    from cp_dmrg_response_mps_krylov import MPSKrylovVector

    n = args.nrings
    name = {2: "naphthalene", 3: "anthracene", 4: "tetracene", 5: "pentacene"}.get(n, f"[{n}]acene")
    atoms, coords_ang = acene_geometry(n)
    symbols = atoms
    coords_bohr = coords_ang * ANG
    nC = atoms.count("C")
    sched = build_schedule(args.bond_dim, args.m_ceiling)
    noises = [1.0e-4] * max(0, len(sched) - 4) + [1.0e-5] * 2 + [0.0] * 2
    dd = det_dim(nC, (nC // 2, nC - nC // 2))
    log(f"{name} [{n}]acene C{nC}H{atoms.count('H')} CAS({nC},{nC}) det={dd:.3e} sched={sched}")

    out_path = args.out or str(_HERE / "data" / f"acene_beyond_fci_{name}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result = {"system": name, "n_rings": n, "ncas": nC, "nelecas": nC, "det_dim": dd,
              "beyond_fci": bool(dd >= 5.0e7), "h_bohr": args.h_bohr}

    # mo guess: pi active space
    mol = gto.M(atom=[(symbols[i], tuple(coords_bohr[i])) for i in range(len(symbols))],
                basis="sto-3g", unit="Bohr", symmetry=False, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    ncas, nelecas, mo0 = select_pi_active(mol, mf, nC)
    log(f"RHF E={mf.e_tot:.6f}; pi-space CAS({ncas},{nelecas})")

    # ---- build SA(2) DMRG-CASSCF via build_progressive (MPS-native at EVERY size
    #      -> avoids the build_robust det>=5e7 native/skip gate that hangs in
    #      mps_to_fci_generic for small/medium CAS; AND it is the SAME path the FD
    #      uses, so analytic and FD share one consistent reference -- the C20
    #      cross-build-inconsistency lesson). ----
    from run_cas_directional_fd import build_progressive, DEFAULT_M_SCHEDULE
    m_schedule = DEFAULT_M_SCHEDULE
    bt = time.perf_counter()
    mol0, mc, solver, blog = build_progressive(
        symbols, coords_bohr, "sto-3g", ncas, nelecas, m_schedule=m_schedule,
        mo_guess=mo0, threads=args.threads, stack_mem_mb=args.stack_mem_mb)
    last = blog["stages"][-1]
    conv = bool(last.get("converged")); e_states = last.get("e_states")
    result["build"] = {"converged": conv, "e_states": e_states,
                       "wall_s": time.perf_counter() - bt, "stages": blog["stages"]}
    log(f"BUILD conv={conv} e={[round(x,6) for x in e_states]} wall={time.perf_counter()-bt:.0f}s")
    if not conv:
        result["status"] = "build_failed"; json.dump(result, open(out_path, "w"), indent=2, default=str); return 1
    e_ref0 = float(e_states[0])

    import block2
    block2.Global.frame = mc.fcisolver._driver.frame
    obj = _make_mps_krylov_response(mc)
    obj._m_compress = int(args.m_ceiling)
    obj._ci_bra_schedule = sched
    obj._ci_noises = noises

    # ---- analytic state-0 gradient via split-m Schur ----
    t1 = time.perf_counter()
    kappa, ci, sinfo, meta = solve_state_sweep_schur(
        obj, 0, orb_tol=args.orb_tol, orb_max_iter=args.orb_max_iter,
        ci_sweeps=args.ci_sweeps, ci_tol=args.ci_tol, solver_type="MinRes",
        proj_weight=1.0e3, residual_tol=args.response_tol,
        ci_m_loop=args.ci_m_loop, ci_schedule_final=sched, ci_noises_final=noises, verbose=True)
    z = MPSKrylovVector(obj, kappa, ci, label="schur0")
    cert = certify_response(obj, z, state=0, tol=args.response_tol, solver="sweep_schur")
    log(f"SCHUR DONE wall={time.perf_counter()-t1:.0f}s true_resid={cert.true_residual_relative:.3e}")
    g0 = _gradient_one_state_mps_krylov(mc, obj, 0, tol=args.response_tol, max_iter=1,
                                        precomputed_z=(kappa, ci))
    result["true_residual_relative"] = float(cert.true_residual_relative)
    result["analytic_gradient_norm"] = float(np.linalg.norm(g0))

    dirs = acene_named_directions(symbols, coords_bohr)
    gdotq = {k: float(np.tensordot(g0, q)) for k, q in dirs.items()}
    result["analytic_dir"] = gdotq
    log("analytic g.q: " + json.dumps({k: round(v, 6) for k, v in gdotq.items()}))

    # ---- exact pyscf-FCI ground truth (naphthalene / FCI-feasible) ----
    if args.fci_check and dd < 5.0e6:
        mcf = mcscf.CASSCF(mf, ncas, nelecas).state_average_([0.5, 0.5])
        mcf.conv_tol = 1e-9
        mcf.kernel(mo0)
        gfci = mcf.nuc_grad_method().kernel(state=0)
        fci_dir = {k: float(np.tensordot(gfci, q)) for k, q in dirs.items()}
        result["fci_states"] = [float(x) for x in mcf.e_states]
        result["fci_dir"] = fci_dir
        result["analytic_vs_fci"] = {k: {"analytic": gdotq[k], "fci": fci_dir[k],
                                         "abs_err": abs(gdotq[k] - fci_dir[k])} for k in gdotq}
        log("analytic vs EXACT-FCI: " + json.dumps(result["analytic_vs_fci"]))

    # ---- directional FD (warm-started from the SAME reference, basin-checked) ----
    if args.fd:
        from run_cas_directional_fd import directional_fd
        moA = mc.mo_coeff; e_ref = e_ref0          # reuse the single consistent build
        result["fd_reference_e0"] = e_ref
        fd = {}
        for k, q in dirs.items():
            d = directional_fd(symbols, coords_bohr, q, basis="sto-3g", ncas=ncas,
                               nelecas=nelecas, m_schedule=m_schedule, mo0=moA, mol0=mol0,
                               h_bohr=args.h_bohr, threads=args.threads, stack_mem_mb=args.stack_mem_mb,
                               g_analytic=g0)
            d["in_basin"] = bool(abs(d["e_plus"] - e_ref) < 5e-3 and abs(d["e_minus"] - e_ref) < 5e-3)
            fd[k] = d
            log(f"FD {k}: analytic={d['analytic_dir']:+.6f} fd={d['g_fd_dir']:+.6f} "
                f"abs_err={d['abs_err']:.2e} in_basin={d['in_basin']}")
        result["fd"] = fd

    result["status"] = "ok"
    json.dump(result, open(out_path, "w"), indent=2, default=str)
    log(f"wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
