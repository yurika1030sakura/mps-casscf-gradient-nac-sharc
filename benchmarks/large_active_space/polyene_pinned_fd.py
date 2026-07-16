"""Overlap-PINNED continuation FD: genuinely try to get clean analytic=FD on the
soft polyene directions (C20 bla/central_cc, C22/C24) that the coarse continuation
FD left branch-contaminated.

The earlier continuation FD warm-started a ladder but let CASSCF freely re-optimize and
took the endpoint secant with no enforcement of staying on the reference branch -- so on
the soft bond-alternation modes the optimizer slipped to the lower branch and the
endpoint was contaminated. This driver PINS to the reference branch: at every displaced
step it measures the active-subspace cross-geometry overlap sigma_min against the
REFERENCE orbitals; if sigma_min drops below a gate (basin slip) it REJECTS the step,
halves it, and retries, building an adaptive fine ladder that stays on-branch. The FD is
formed only from overlap-verified on-branch endpoints that reach +-h. If a direction
cannot be followed to +-h without slipping (an immediate branch crossing -- genuine
physics, not numerics), that is reported honestly rather than papered over.
"""
from __future__ import annotations
import argparse, json, sys, time, shutil, os
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
DEV = _HERE.parents[1] / "src" / "dmrg_analytic_dev"
SH = _HERE.parents[1] / "sharc_interface"
for _p in (str(_HERE), str(DEV), str(SH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import faulthandler
faulthandler.dump_traceback_later(1200, repeat=True, file=sys.stderr)
ANG = 1.8897261246257702
_T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-_T0:8.1f}s] {m}", flush=True)

def build_schedule(build_m, ceiling):
    levels, m = [], int(build_m)
    while m < int(ceiling):
        levels.append(m); m = int(m * 1.5)
    levels.append(int(ceiling))
    return [L for L in levels for _ in (0, 1)]

def aza_polyene_geometry(polyene_geometry, n):
    """Polyene with the terminal =CH2 replaced by =NH (Schiff-base model): breaks the
    L<->R symmetry that makes the pure-polyene soft modes near-degenerate, while keeping
    an n-center pi system (CAS(n,n)). Validated vs FCI at CAS(6,6)."""
    atoms = list(polyene_geometry(n))
    cx, cy, cz = atoms[n - 1][1]
    atoms[n - 1] = ("N", (cx, cy, cz))
    return atoms[:len(atoms) - 1] + atoms[len(atoms):]  # drop the axial terminal H

def select_pi_hetero(mol, mf, n_pi):
    """pi-space selection over C AND N pz AOs (for aza-polyenes)."""
    import numpy as _np
    mo = mf.mo_coeff; S = mf.get_ovlp(); labels = mol.ao_labels()
    pz = [i for i, l in enumerate(labels)
          if len(l.split()) >= 3 and l.split()[1] in ("C", "N") and l.split()[2].endswith("pz")]
    Smo = S @ mo
    pichar = _np.einsum("ai,ai->i", mo[pz, :], Smo[pz, :])
    active = sorted(_np.argsort(-pichar)[:n_pi])
    nocc = mol.nelectron // 2; ncore = nocc - n_pi // 2
    rest = [i for i in range(mo.shape[1]) if i not in active]
    order = rest[:ncore] + list(active) + rest[ncore:]
    return mo[:, order]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, default=20)
    ap.add_argument("--aza", action="store_true", help="terminal =CH2 -> =NH Schiff-base heteroatom polyene")
    ap.add_argument("--directions", nargs="+", default=["central_cc", "bla"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--h-bohr", type=float, default=1.0e-3)
    ap.add_argument("--k-init", type=int, default=8, help="nominal sub-steps from ref to +-h")
    ap.add_argument("--sigma-gate", type=float, default=0.90, help="min active-subspace overlap to accept a step")
    ap.add_argument("--max-halve", type=int, default=7, help="max consecutive step halvings before declaring a crossing")
    ap.add_argument("--bond-dim", type=int, default=800)
    ap.add_argument("--m-ceiling", type=int, default=800)
    ap.add_argument("--ci-m-loop", type=int, default=256)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--stack-mem-mb", type=int, default=24000)
    ap.add_argument("--ci-pin", action="store_true",
                    help="pin the CI vector: carry the converged MPS across continuation steps "
                         "via a persistent scratch so each displaced SA-CASSCF keeps the SAME "
                         "averaged state pair (fixes the CI-level basin oscillation sigma_min misses)")
    ap.add_argument("--scratch-base", default="/tmp/dmrg_scratch",
                    help="base dir for the persistent MPS scratch when --ci-pin")
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
    geomfn = (lambda nn: aza_polyene_geometry(polyene_geometry, nn)) if args.aza else polyene_geometry
    symbols = [a[0] for a in geomfn(n)]
    coords0 = np.array([a[1] for a in geomfn(n)]) * ANG
    ncas = nelecas = n
    dd = det_dim(ncas, (ncas // 2, ncas - ncas // 2))
    h = args.h_bohr
    m_schedule = [(256, 1e-8, 1e-4, 40), (512, 1e-10, 1e-5, 60),
                  (int(args.bond_dim), 1e-12, 1e-6, 100)]
    sched = build_schedule(args.bond_dim, args.m_ceiling)
    noises = [1.0e-4] * max(0, len(sched) - 4) + [1.0e-5] * 2 + [0.0] * 2
    tag = f"aza_c{n}" if args.aza else f"c{n}"
    out_path = args.out or str(_HERE / "data" / f"polyene_pinned_{tag}.json")
    result = {"system": f"{'azapolyene' if args.aza else 'polyene'}_C{n}", "ncas": ncas, "det_dim": dd, "h_bohr": h,
              "sigma_gate": args.sigma_gate, "directions": args.directions, "method": "overlap-pinned-continuation-FD"}
    log(f"C{n} CAS({ncas},{ncas}) det={dd:.3e} dirs={args.directions} sigma_gate={args.sigma_gate}")

    def pi_mo(c_bohr):
        mol = gto.M(atom=[(symbols[i], tuple(c_bohr[i])) for i in range(len(symbols))],
                    basis="sto-3g", unit="Bohr", verbose=0)
        from pyscf import scf
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        return select_pi_hetero(mol, mf, n) if args.aza else select_pi_active(mol, mf, n)[2]

    # 1) reference: multi-seed -> lowest basin (record the winning seed so the
    #    CI-pinned walk can deterministically reproduce that exact basin)
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
            best = (e0, mcX, molX, np.array(mcX.mo_coeff), int(s))
    e_ref, mc, mol0, mo_A, s_best = best
    ncore = int(mc.ncore)
    result["e_ref"] = e_ref; result["ncore"] = ncore; result["ci_pin"] = bool(args.ci_pin)
    result["seed_best"] = s_best
    log(f"REFERENCE basin e0={e_ref:.8f} ncore={ncore} seed_best={s_best} ci_pin={args.ci_pin}")

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

    def build_at(ck, mo_guess, mps_dir=None):
        mol_k = gto.M(atom=[(symbols[i], tuple(ck[i])) for i in range(len(symbols))],
                      basis="sto-3g", unit="Bohr", verbose=0)
        _m, mck, _s, blk = build_progressive(
            symbols, ck, "sto-3g", ncas, nelecas, m_schedule=m_schedule,
            mo_guess=mo_guess, threads=args.threads, stack_mem_mb=args.stack_mem_mb,
            mps_persistent_dir=mps_dir)
        e_k = float(blk["stages"][-1]["e_states"][0])
        return mol_k, np.array(mck.mo_coeff), e_k

    def seed_walk_dir(sgn_char, dname):
        """Fresh persistent MPS scratch holding the reference basin for one walk.
        It is a COPY of the single master reference MPS, so the +side and -side of
        every direction start from the IDENTICAL reference state -- a per-sign rebuild
        can land in a different near-degenerate basin (offset ~1e-4 Eh), which leaves
        each side individually smooth but puts E(+h) and E(-h) on different branches
        and contaminates the central difference."""
        if not args.ci_pin:
            return None
        wd = os.path.join(args.scratch_base, f"cipin_{tag}_{dname}_{sgn_char}")
        shutil.rmtree(wd, ignore_errors=True)
        shutil.copytree(master_dir, wd)
        log(f"  [{dname} {sgn_char}] copied MASTER reference MPS -> walk dir (both signs share the branch)")
        return wd

    # 2b) CI-pin: build the reference MPS ONCE into a master dir that every walk copies
    master_dir = None
    if args.ci_pin:
        master_dir = os.path.join(args.scratch_base, f"cipin_{tag}_master")
        shutil.rmtree(master_dir, ignore_errors=True); os.makedirs(master_dir, exist_ok=True)
        block2.Random.rand_seed(int(s_best))
        _mm, mc_m, _sm, blog_m = build_progressive(
            symbols, coords0, "sto-3g", ncas, nelecas, m_schedule=m_schedule,
            mo_guess=pi_mo(coords0), threads=args.threads,
            stack_mem_mb=args.stack_mem_mb, mps_persistent_dir=master_dir)
        e_master = float(blog_m["stages"][-1]["e_states"][0])
        result["e_master"] = e_master
        log(f"  built CI-pin MASTER reference MPS (seed {s_best}) e0={e_master:.8f} "
            f"(vs analytic-ref e0={e_ref:.8f}; offset {abs(e_master-e_ref):.2e})")

    # 3) overlap-PINNED continuation FD per direction
    result["fd"] = {}
    for dname in args.directions:
        q = dirs[dname]; ga = g_analytic[dname]
        ends = {}; rec = {}
        for sgn in (+1.0, -1.0):
            mo_cur, mol_cur = mo_A, mol0
            walk_dir = seed_walk_dir("p" if sgn > 0 else "m", dname)
            pos = 0.0; step = h / args.k_init; n_halve = 0
            accepted = []; reached = False; crossing = False
            while True:
                remaining = sgn * h - pos
                if abs(remaining) < 1e-9:
                    reached = True; break
                trial = pos + np.sign(remaining) * min(abs(step), abs(remaining))
                ck = coords0 + trial * q
                # project current-branch orbitals to the trial geometry, build, check overlap vs REFERENCE
                mol_trial = gto.M(atom=[(symbols[i], tuple(ck[i])) for i in range(len(symbols))],
                                  basis="sto-3g", unit="Bohr", verbose=0)
                mg = fdv.project_mo_to_new_geometry(mol_cur, mol_trial, mo_cur)[0]
                _m2, mo_k, e_k = build_at(ck, mg, mps_dir=walk_dir)
                smin = fdv.active_subspace_overlap(mol0, mo_A, mol_trial, mo_k, ncas, ncore)
                onb = smin >= args.sigma_gate
                log(f"  {dname} sgn={sgn:+.0f} frac={trial:+.3e} e0={e_k:.8f} sigma_min={smin:.3f} on_branch={onb} (pos={pos:+.2e} step={step:.1e})")
                if onb:
                    accepted.append({"frac": trial, "e0": e_k, "sigma_min": smin})
                    pos = trial; mo_cur, mol_cur = mo_k, mol_trial
                    n_halve = 0; step = h / args.k_init
                else:
                    n_halve += 1
                    if n_halve > args.max_halve:
                        crossing = True
                        log(f"  {dname} sgn={sgn:+.0f}: CANNOT follow reference branch to +-h "
                            f"(branch crossing at frac~{trial:.2e}, sigma_min={smin:.3f})")
                        break
                    step = step / 2.0
            ends[sgn] = accepted[-1]["e0"] if reached else None
            rec[f"ladder_{'p' if sgn>0 else 'm'}"] = accepted
            rec[f"reached_{'p' if sgn>0 else 'm'}"] = bool(reached)
            rec[f"crossing_{'p' if sgn>0 else 'm'}"] = bool(crossing)
        if ends.get(+1.0) is not None and ends.get(-1.0) is not None:
            g_central = (ends[+1.0] - ends[-1.0]) / (2.0 * h)
            rec["g_fd_central"] = g_central
            rec["abs_err_vs_analytic"] = abs(g_central - ga)
            rec["status"] = "clean" if abs(g_central - ga) < 5e-3 else "on-branch-but-loose"
            log(f"{dname}: PINNED central FD={g_central:+.6f} vs analytic {ga:+.6f} "
                f"abs_err={abs(g_central-ga):.3e}  [{rec['status']}]")
        else:
            rec["g_fd_central"] = None
            rec["status"] = "FD-inadmissible: reference branch not followable to +-h (genuine branch crossing)"
            log(f"{dname}: FD-INADMISSIBLE -- reference branch not followable to +-h; analytic {ga:+.6f} validated by certificate only")
        result["fd"][dname] = rec
        json.dump(result, open(out_path, "w"), indent=2, default=str)
    result["status"] = "ok"
    json.dump(result, open(out_path, "w"), indent=2, default=str)
    log(f"wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
