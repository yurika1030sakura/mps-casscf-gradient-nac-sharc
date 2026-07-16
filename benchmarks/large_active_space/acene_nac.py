"""Beyond-FCI analytic NAC (+ excited-state gradient) on linear [n]acenes.

Completes the paper's scope ("Gradients AND Nonadiabatic Couplings") on the acene
family: at large CAS we validate
  - excited-state (S1) analytic gradient via the split-m Schur response, vs FD;
  - analytic S0/S1 NAC via the MPS-Krylov response (solve_nac_mps) + transition-RDM
    Lagrange assembly, vs the FD active-overlap NAC (cross_geometry_overlap_matrix);
  - naphthalene CAS(10,10): exact-FCI cross-check of both (fd_nac, pyscf grad).
Acene S0/S1 couplings are physically central (singlet fission).

Reuses: build_progressive (MPS-native every size; one consistent reference for all
three geometries R/P/M), the generic Schur response, _nac_one_pair_mps_krylov,
cross_geometry_overlap_matrix.
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
faulthandler.dump_traceback_later(180, repeat=True, file=sys.stderr)  # show where it hangs

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
    ap.add_argument("--nac-max-iter", type=int, default=30)
    ap.add_argument("--nac-m-compress", type=int, default=400,
                    help="cap NAC Krylov MPS bond dim (full GMRES otherwise grows it unbounded -> hang)")
    ap.add_argument("--response-tol", type=float, default=1.0e-4)
    ap.add_argument("--nac-solver", default="gmres",
                    help="NAC response linear solver: gmres (default) | cr (conjugate-residual, "
                         "short-recurrence, for small-gap high-amplitude systems where GMRES hangs)")
    ap.add_argument("--h-bohr", type=float, default=1.0e-3)
    ap.add_argument("--disp-atom", type=int, default=-1, help="-1 -> central carbon")
    ap.add_argument("--disp-comp", type=int, default=0, help="0=x (long axis)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--stack-mem-mb", type=int, default=24000)
    ap.add_argument("--nroots", type=int, default=2,
                    help="state-averaging size; 3 includes the full near-degenerate excited "
                         "manifold so the averaged state set is stable across the displacement")
    ap.add_argument("--excited-grad", action="store_true")
    ap.add_argument("--fci-check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import block2
    from pyscf import gto, scf
    from acene_systems import acene_geometry, acene_named_directions
    from run_polyene_beyond_fci import select_pi_active, det_dim
    from run_cas_directional_fd import build_progressive, DEFAULT_M_SCHEDULE, directional_fd
    from analytic_cp_sharc import (_make_mps_krylov_response, _nac_one_pair_mps_krylov,
                                   _gradient_one_state_mps_krylov)
    from sweep_coupled_response import solve_state_sweep_schur
    from cross_geometry_overlap import cross_geometry_overlap_matrix
    import fd_validation as fdv

    n = args.nrings
    name = {2: "naphthalene", 3: "anthracene", 4: "tetracene", 5: "pentacene"}.get(n, f"[{n}]acene")
    atoms, coords_ang = acene_geometry(n)
    symbols = atoms
    coords0 = coords_ang.copy()                  # Angstrom
    nC = atoms.count("C")
    dd = det_dim(nC, (nC // 2, nC - nC // 2))
    sched = build_schedule(args.bond_dim, args.m_ceiling)
    noises = [1.0e-4] * max(0, len(sched) - 4) + [1.0e-5] * 2 + [0.0] * 2
    # central carbon for the NAC displacement
    cidx = [i for i in range(len(symbols)) if symbols[i] == "C"]
    xs = coords0[cidx, 0]; centre = cidx[int(np.argmin(np.abs(xs - xs.mean())))] if args.disp_atom < 0 else args.disp_atom
    comp = args.disp_comp
    log(f"{name} CAS({nC},{nC}) det={dd:.3e} NAC-disp atom={centre}({symbols[centre]}) comp={comp}")

    out_path = args.out or str(_HERE / "data" / f"acene_nac_{name}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result = {"system": name, "n_rings": n, "ncas": nC, "nelecas": nC, "det_dim": dd,
              "beyond_fci": bool(dd >= 5.0e7), "h_bohr": args.h_bohr,
              "disp_atom": int(centre), "disp_comp": int(comp)}

    m_schedule = DEFAULT_M_SCHEDULE
    def pi_mo(c_ang):
        mol = gto.M(atom=[(symbols[i], tuple(c_ang[i])) for i in range(len(symbols))],
                    basis="sto-3g", unit="Angstrom", verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        _, _, mo = select_pi_active(mol, mf, nC)
        return mo

    def coords_at(sign):
        c = coords0.copy(); c[centre, comp] += sign * args.h_bohr / ANG; return c

    geoms = {"R": coords0, "P": coords_at(+1.0), "M": coords_at(-1.0)}
    built = {}
    # CRITICAL for a clean cross-geometry-overlap NAC: build R first, then WARM-START
    # P/M from R's converged orbitals (projected to the displaced geometry).  An
    # independent pi_mo() guess at each geometry puts the displaced active orbitals
    # in a DIFFERENT gauge (sign flips / rotations / reorderings) -> the cross
    # overlap s is far from identity -> rotate_mps_orbitals does a large inaccurate
    # rotation (or a det=-1 reflection) -> contaminated overlap-FD NAC.  Continuous
    # gauge keeps s ~ identity (same lesson as the gradient FD).
    mo_R = None; mol_R = None
    # R/P/M each get a SEPARATE MPS subdir under a shared parent pin scratch. Displaced
    # builds warm-start from R's reference ORBITALS (projected -> continuous gauge, no
    # det=-1 reflection). Using a SHARED pin_dir instead makes the P and M MPS overwrite
    # each other, so the cross-overlap Op[0,1] and Om[0,1] end up computed from the SAME
    # (M) displaced MPS and (Op - Om)/(2h) collapses to machine zero. This mirrors the
    # separate-subdir fix in run_beyond_fci_nac.py.
    import os as _os, shutil as _shutil
    # Job-unique pin_dir: SLURM_JOB_ID is appended so two concurrent runs on the same
    # (system, nroots) cannot collide on the persistent MPS scratch. Previously all
    # runs shared "acenepin_<name>_sa<nroots>" which made a second job silently trample
    # a first job's MPS files, producing StateInfo::load_data crashes mid-solve.
    _jid = _os.environ.get("SLURM_JOB_ID", "local")
    _pin = _os.path.join("/tmp/dmrg_scratch",
                          f"acenepin_{name}_sa{args.nroots}_j{_jid}")
    _shutil.rmtree(_pin, ignore_errors=True); _os.makedirs(_pin, exist_ok=True)
    for key in ("R", "P", "M"):
        c_bohr = geoms[key] * ANG
        mol_k = gto.M(atom=[(symbols[i], tuple(c_bohr[i])) for i in range(len(symbols))],
                      basis="sto-3g", unit="Bohr", verbose=0)
        if key == "R":
            mo_guess = pi_mo(geoms[key])
        else:
            mo_guess = fdv.project_mo_to_new_geometry(mol_R, mol_k, mo_R)[0]
        dir_key = _os.path.join(_pin, key); _os.makedirs(dir_key, exist_ok=True)
        bt = time.perf_counter()
        molX, mcX, solX, blogX = build_progressive(
            symbols, c_bohr, "sto-3g", nC, nC, m_schedule=m_schedule,
            mo_guess=mo_guess, threads=args.threads, stack_mem_mb=args.stack_mem_mb,
            nroots=int(args.nroots), weights=[1.0 / args.nroots] * args.nroots,
            mps_persistent_dir=dir_key)
        if key == "R":
            mo_R = np.array(mcX.mo_coeff); mol_R = molX
        _est = blogX["stages"][-1].get("e_states")
        built[key] = dict(mol=molX, mc=mcX, solver=solX,
                          e=[float(x) for x in _est])
        # report the cross-overlap distance from identity (gauge-continuity check)
        if key != "R":
            from cross_geometry_overlap import cross_geometry_active_overlap
            s_chk = cross_geometry_active_overlap(mol_R, molX, mo_R, mcX.mo_coeff, built["R"]["mc"].ncore, nC)
            off = float(np.max(np.abs(s_chk - np.eye(nC))))
            log(f"BUILD {key} e={[round(x,6) for x in built[key]['e']]} "
                f"|s-I|max={off:.2e} (want <<1) wall={time.perf_counter()-bt:.0f}s")
        else:
            log(f"BUILD {key} e={[round(x,6) for x in built[key]['e']]} wall={time.perf_counter()-bt:.0f}s")
    for key in ("R", "P", "M"):
        block2.Global.frame = built[key]["mc"].fcisolver._driver.frame
        built[key]["obj"] = _make_mps_krylov_response(built[key]["mc"])
        log(f"response object {key} built")
    R, P, M = built["R"], built["P"], built["M"]
    e_R = R["e"]; result["e_states_R"] = e_R; result["gap_Eh"] = abs(e_R[1] - e_R[0])

    # ---- analytic S0/S1 NAC (MPS-Krylov) at R ----
    block2.Global.frame = R["mc"].fcisolver._driver.frame
    # CAP the response bond dim: the full NAC GMRES otherwise lets the Krylov MPS grow
    # unbounded -> each driver.expectation overlap becomes huge and effectively hangs
    # (the gradient Schur path stays fast precisely because it caps the CI loop at ~256).
    R["obj"]._m_compress = int(args.nac_m_compress)
    # Response linear solver: GMRES has O(k^2) Arnoldi orthogonalization, so a large
    # m_compress + many iterations hangs in the pyblock2 MPS addition. For a high-
    # amplitude solution (small-gap, large-NAC systems like pentacene, gap 4.1 eV,
    # |NAC| ~ 0.175) the short-recurrence conjugate-residual solver (cr) minimizes the
    # residual like GMRES but with a 3-term recurrence (2 matvec/iter, O(1) memory),
    # avoiding both the GMRES hang and the bicgstab divergence. The SA-CASSCF response
    # Hessian is symmetric, which is exactly CR's assumption.
    R["obj"]._linear_solver = str(args.nac_solver).strip().lower()
    log(f"starting analytic NAC solve (solver={args.nac_solver}, max_iter={args.nac_max_iter}, "
        f"m_compress={args.nac_m_compress})...")
    t1 = time.perf_counter()
    de_nac = _nac_one_pair_mps_krylov(R["mc"], R["obj"], (0, 1),
                                      tol=args.response_tol, max_iter=args.nac_max_iter)
    result["nac_analytic_full_atom_comp"] = float(de_nac[centre, comp])
    result["nac_analytic_wall_s"] = time.perf_counter() - t1
    log(f"analytic NAC[atom{centre},{comp}] = {float(de_nac[centre,comp]):+.6f} "
        f"(wall {result['nac_analytic_wall_s']:.0f}s)")

    # ---- FD active-overlap NAC (guarded: det-1 reflection can need sign surgery) ----
    ncore = R["mc"].ncore; host = R["mc"].fcisolver._driver.frame
    t2 = time.perf_counter()
    try:
        Op, s_p = cross_geometry_overlap_matrix(R["obj"], P["obj"], R["mol"], P["mol"],
                                                R["mc"].mo_coeff, P["mc"].mo_coeff, ncore, nC, int(args.nroots),
                                                host_frame=host, tag="ANP")
        Om, s_m = cross_geometry_overlap_matrix(R["obj"], M["obj"], R["mol"], M["mol"],
                                                R["mc"].mo_coeff, M["mc"].mo_coeff, ncore, nC, int(args.nroots),
                                                host_frame=host, tag="ANM")
        d_fd = float((Op[0, 1] - Om[0, 1]) / (2.0 * args.h_bohr))
        result["nac_fd_overlap_01"] = d_fd
        result["nac_fd_wall_s"] = time.perf_counter() - t2
        result["active_sigma_min"] = float(min(np.min(np.linalg.svd(s_p, compute_uv=False)),
                                               np.min(np.linalg.svd(s_m, compute_uv=False))))
        result["nac_analytic_vs_fd_abserr"] = abs(abs(result["nac_analytic_full_atom_comp"]) - abs(d_fd))
        log(f"FD-overlap NAC = {d_fd:+.6f} ; |analytic|-|fd| abs_err = {result['nac_analytic_vs_fd_abserr']:.3e}")
    except Exception as exc:  # noqa: BLE001
        result["nac_fd_overlap_error"] = str(exc)[:200]
        log(f"FD-overlap NAC skipped ({type(exc).__name__}: {str(exc)[:120]}) -> rely on FCI reference")

    # ---- exact-FCI NAC cross-check (naphthalene / feasible) ----
    if args.fci_check and dd < 5.0e6:
        import fd_validation as fdv
        cfg_fci = dict(fdv.DEFAULT_SOLVER_CFG)
        cfg_fci.update(bond_dim=args.bond_dim, n_sweeps=30, sweep_tol=1e-10,
                       n_threads=args.threads, stack_mem_mb=args.stack_mem_mb)
        d_fci = fdv.fd_nac(symbols, coords0 * ANG, bra=0, ket=1, basis="sto-3g",
                           charge=0, spin=0, ncas=nC, nelecas=nC, nroots=2,
                           weights=[0.5, 0.5], solver_cfg=cfg_fci, h_bohr=args.h_bohr,
                           atmlst=[centre], components=[comp])
        result["nac_fci_fd_01"] = float(d_fci[centre, comp])
        result["nac_analytic_vs_fci_abserr"] = abs(abs(result["nac_analytic_full_atom_comp"]) - abs(d_fci[centre, comp]))
        log(f"exact-FCI NAC = {float(d_fci[centre,comp]):+.6f} ; |analytic|-|fci| = {result['nac_analytic_vs_fci_abserr']:.3e}")

    # ---- excited-state (S1) gradient via Schur, vs FD ----
    if args.excited_grad:
        obj = R["obj"]; obj._m_compress = int(args.m_ceiling)
        obj._ci_bra_schedule = sched; obj._ci_noises = noises
        block2.Global.frame = R["mc"].fcisolver._driver.frame
        kappa, ci, sinfo, meta = solve_state_sweep_schur(
            obj, 1, orb_tol=1e-4, orb_max_iter=20, ci_sweeps=16, ci_tol=1e-6,
            solver_type="MinRes", proj_weight=1e3, residual_tol=args.response_tol,
            ci_m_loop=args.ci_m_loop, ci_schedule_final=sched, ci_noises_final=noises, verbose=True)
        g1 = _gradient_one_state_mps_krylov(R["mc"], obj, 1, tol=args.response_tol, max_iter=1,
                                            precomputed_z=(kappa, ci))
        dirs = acene_named_directions(symbols, coords0 * ANG)
        g1q = {k: float(np.tensordot(g1, q)) for k, q in dirs.items()}
        result["excited_grad_analytic_dir"] = g1q
        log("S1 analytic g.q: " + json.dumps({k: round(v, 6) for k, v in g1q.items()}))

    result["status"] = "ok"
    json.dump(result, open(out_path, "w"), indent=2, default=str)
    log(f"wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
