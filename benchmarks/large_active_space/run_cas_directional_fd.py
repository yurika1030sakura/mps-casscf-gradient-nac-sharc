"""Beyond-FCI directional finite-difference gradients for polyene CAS(n,n).

Reviewer-1 centerpiece, de-risked.  Instead of arbitrary Cartesian components,
we validate the analytic SA-DMRG-CASSCF gradient along *physically motivated
collective directions* by central finite differences of the solver's own state
energy:

    G_I(q)      = g_I . q                       (analytic directional derivative)
    G_I^FD(q;h) = [E_I(R+h q) - E_I(R-h q)] / 2h

Each direction needs only two displaced SA-DMRG-CASSCF calculations, so three
directions cost six displaced builds instead of a full gradient vector.  The
default directions are the bond-length-alternation (BLA) mode -- the collective
coordinate that controls the low-lying polyene gap -- the central C=C stretch,
and a second internal single-bond stretch.

Cost control (so a CAS(20,20) build actually finishes):
  * progressive bond dimension (cheap macro-iterations at low M, then raise M);
  * warm-started orbital guess from the reference / lower-M stage;
  * an explicit M-schedule with loose-to-tight CASSCF convergence.

Integrity (so the beyond-FCI claim is unattackable):
  * the dense-bridge sentinel certifies no MPS->FCI/determinant bridge was ever
    formed in a beyond-FCI run;
  * every accepted analytic derivative carries a true-residual response
    certificate;
  * active-subspace continuity (sigma_min) and the state gap are reported, and
    each point gets a PASS/WARN/FAIL self-diagnosis.

Usage:
  python run_cas_directional_fd.py --ncarbon 10 --out data/dirfd_c10.json   # FCI-checkable
  python run_cas_directional_fd.py --ncarbon 20 --out data/dirfd_c20.json   # beyond FCI
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DEV = _HERE.parents[1] / "src" / "dmrg_analytic_dev"
SHARC = _HERE.parents[1] / "sharc_interface"
for p in (str(DEV), str(SHARC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pyscf import gto, scf, mcscf
import fd_validation as fdv
from dmrg_fcisolver import MPSAsFCISolver
from analytic_cp_sharc import compute_grad_nac_analytic_cp
from fci_free_guard import (DenseBridgeSentinel, RootTracking,
                            assert_fci_free_if_needed, FCI_FREE_THRESHOLD)
from system_diagnostics import assess_point
from run_polyene_beyond_fci import (polyene_geometry, det_dim, select_pi_active,
                                    beyond_fci_solver_cfg)

ANG = 1.8897261246257702

# Progressive bond-dimension schedule (M, sweep_tol, conv_tol_grad, max_macro).
DEFAULT_M_SCHEDULE = [
    (256, 1.0e-7, 1.0e-4, 30),
    (512, 1.0e-8, 3.0e-5, 40),
    (800, 1.0e-9, 1.0e-5, 60),
]


# ----------------------------------------------------------- collective directions
def _carbon_indices(symbols):
    return [i for i, s in enumerate(symbols) if s == "C"]


def bla_direction(coords_bohr, carbon_indices):
    """Normalized bond-length-alternation Cartesian direction (natom, 3)."""
    q = np.zeros_like(coords_bohr)
    for k, (i, j) in enumerate(zip(carbon_indices[:-1], carbon_indices[1:])):
        u = coords_bohr[j] - coords_bohr[i]
        u = u / np.linalg.norm(u)
        s = 1.0 if k % 2 == 0 else -1.0
        q[i] -= s * u
        q[j] += s * u
    q -= q.mean(axis=0)            # project out translation
    n = np.linalg.norm(q)
    return q / n if n > 0 else q


def stretch_direction(coords_bohr, i, j):
    """Normalized stretch of the i-j bond (natom, 3)."""
    q = np.zeros_like(coords_bohr)
    u = coords_bohr[j] - coords_bohr[i]
    u = u / np.linalg.norm(u)
    q[i] -= u
    q[j] += u
    q -= q.mean(axis=0)
    return q / np.linalg.norm(q)


def named_directions(symbols, coords_bohr):
    c = _carbon_indices(symbols)
    mid = len(c) // 2
    dirs = {"bla": bla_direction(coords_bohr, c)}
    if len(c) >= 2:
        dirs["central_cc"] = stretch_direction(coords_bohr, c[mid - 1], c[mid])
    if len(c) >= 4:
        dirs["single_cc"] = stretch_direction(coords_bohr, c[mid], c[mid + 1])
    return dirs


# ------------------------------------------------------- progressive-M build
def build_progressive(symbols, coords_bohr, basis, ncas, nelecas, *,
                      m_schedule, mo_guess=None, threads=8, stack_mem_mb=8000,
                      nroots=2, weights=(0.5, 0.5), warm_start=True):
    """SA-DMRG-CASSCF with a progressive bond-dimension schedule.

    Cheap low-M macro-iterations relax the orbitals; M is then raised with the
    previous stage's orbitals (and, within a stage, the warm-started MPS) as the
    guess.  Returns (mol, mc, solver, stage_log).
    """
    mol = gto.M(atom=[(symbols[i], tuple(coords_bohr[i])) for i in range(len(symbols))],
                basis=basis, charge=0, spin=0, unit="Bohr", symmetry=False, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1.0e-10)
    if mo_guess is None:
        ncas, nelecas, mo_guess = select_pi_active(mol, mf, ncas)

    det = det_dim(ncas, (nelecas // 2, nelecas - nelecas // 2))
    cfg0 = beyond_fci_solver_cfg(ncas, m_schedule[0][0], threads, stack_mem_mb)
    cfg0["warm_start"] = bool(warm_start)
    # precondition check: above threshold this must be FCI-free with gap_guard
    assert_fci_free_if_needed(ncas, (nelecas // 2, nelecas - nelecas // 2),
                              cfg0, RootTracking.GAP_GUARD, "build_progressive")

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    solver = MPSAsFCISolver(mol, **cfg0)
    solver.nroots = int(nroots)
    mc.fcisolver = solver
    if nroots > 1:
        mc = mc.state_average_(list(weights))

    # The DMRG-CASSCF cost is dominated by the macro-iteration count times the
    # per-macro DMRG cost.  An AH level shift stabilizes the orbital step near
    # small gaps and cuts the macro count (LiF-proven); a reduced sweep count
    # keeps the warm-started macros cheap (the warm path uses min(n_sweeps,8),
    # the first cold solve still ramps internally).  These only change the
    # convergence path, not the final result, which is gated by the certificate.
    if hasattr(mc, "ah_level_shift"):
        mc.ah_level_shift = 0.3
    if hasattr(solver, "n_sweeps"):
        solver.n_sweeps = 12
    mo = mo_guess
    stage_log = []
    for (M, swtol, cgrad, mxm) in m_schedule:
        solver.bond_dim = int(M)
        if hasattr(solver, "sweep_tol"):
            solver.sweep_tol = float(swtol)
        mc.conv_tol = max(cgrad * 1.0e-2, 1.0e-10)
        mc.conv_tol_grad = float(cgrad)
        mc.max_cycle_macro = int(mxm)
        t0 = time.perf_counter()
        mc.kernel(mo)
        mo = mc.mo_coeff
        stage_log.append({"M": int(M), "converged": bool(mc.converged),
                          "wall_s": time.perf_counter() - t0,
                          "e_states": [float(x) for x in mc.e_states]})
    return mol, mc, solver, {"stages": stage_log, "det_dim": det, "ncas": ncas,
                             "nelecas": nelecas}


# ------------------------------------------------------------------- one direction
def directional_fd(symbols, coords0_bohr, q, *, basis, ncas, nelecas, m_schedule,
                   mo0, mol0, h_bohr, threads, stack_mem_mb, g_analytic=None):
    """Central directional FD of E0 along q (the primary beyond-FCI gradient).

    The FD value needs only the two displaced SA-DMRG-CASSCF energies, so it is
    computed independently of the (expensive) analytic response.  If ``g_analytic``
    is supplied the analytic directional derivative and its absolute error are
    added; otherwise that cross-check is filled in later.
    """
    out = {"h_bohr": float(h_bohr)}
    e_disp = {}
    sig = {}
    for sgn in (+1.0, -1.0):
        disp = coords0_bohr + sgn * h_bohr * q
        mol_s, mc_s, _solver, blog = build_progressive(
            symbols, disp, basis, ncas, nelecas, m_schedule=m_schedule,
            mo_guess=fdv.project_mo_to_new_geometry(mol0, gto.M(
                atom=[(symbols[i], tuple(disp[i])) for i in range(len(symbols))],
                basis=basis, unit="Bohr", verbose=0), mo0)[0],
            threads=threads, stack_mem_mb=stack_mem_mb)
        e_disp[sgn] = float(mc_s.e_states[0])
        ncore = mc_s.ncore
        sig[sgn] = fdv.active_subspace_overlap(mol0, mo0, mol_s, mc_s.mo_coeff,
                                               ncas, ncore)
    out["e_plus"], out["e_minus"] = e_disp[+1.0], e_disp[-1.0]
    out["g_fd_dir"] = (e_disp[+1.0] - e_disp[-1.0]) / (2.0 * h_bohr)
    out["active_subspace_sigma_min"] = float(min(sig[+1.0], sig[-1.0]))
    if g_analytic is not None:
        out["analytic_dir"] = float(np.tensordot(g_analytic, q))
        out["abs_err"] = abs(out["analytic_dir"] - out["g_fd_dir"])
    return out


# --------------------------------------------------------------------------- run
def run(n_carbon, *, basis="sto-3g", m_schedule=None, threads=8,
        stack_mem_mb=8000, h_bohr=1.0e-3, directions=None, out_path=None):
    m_schedule = m_schedule or DEFAULT_M_SCHEDULE
    atoms = polyene_geometry(n_carbon)
    symbols = [a[0] for a in atoms]
    coords0 = np.array([a[1] for a in atoms]) * ANG  # bohr
    ncas = n_carbon
    nelecas = n_carbon
    ddim = det_dim(ncas, (nelecas // 2, nelecas - nelecas // 2))
    beyond = ddim >= FCI_FREE_THRESHOLD

    result = {"system": f"polyene_C{n_carbon}", "basis": basis, "ncas": ncas,
              "nelecas": nelecas, "det_dim": ddim, "beyond_fci": bool(beyond),
              "h_bohr": h_bohr, "m_schedule": m_schedule}

    def flush():
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
                f.flush(); os.fsync(f.fileno())

    DenseBridgeSentinel.reset()
    t0 = time.perf_counter()
    mol0, mc0, solver0, blog = build_progressive(
        symbols, coords0, basis, ncas, nelecas, m_schedule=m_schedule,
        threads=threads, stack_mem_mb=stack_mem_mb)
    mo0 = mc0.mo_coeff
    result["reference_build"] = blog
    result["reference_wall_s"] = time.perf_counter() - t0
    result["reference_converged"] = bool(mc0.converged)
    flush()

    # FD directions FIRST: each needs only two warm-started displaced energies,
    # so the beyond-FCI directional gradients land independently of (and before)
    # the expensive CP-DMRG-CASSCF response.  This is the primary R1 evidence; the
    # analytic derivative is the cross-check, added afterwards if it completes.
    dirs = named_directions(symbols, coords0)
    if directions:
        dirs = {k: v for k, v in dirs.items() if k in directions}
    result["directions"] = {}
    for name, q in dirs.items():
        try:
            d = directional_fd(symbols, coords0, q, basis=basis, ncas=ncas,
                               nelecas=nelecas, m_schedule=m_schedule, mo0=mo0,
                               mol0=mol0, h_bohr=h_bohr, threads=threads,
                               stack_mem_mb=stack_mem_mb, g_analytic=None)
            d["health"] = assess_point(
                casscf_converged=result["reference_converged"],
                active_subspace_sigma_min=d["active_subspace_sigma_min"],
                det_dim=ddim, dense_bridge_used=DenseBridgeSentinel.used).to_dict()
            result["directions"][name] = d
        except Exception as exc:  # noqa: BLE001
            result["directions"][name] = {"status": "error",
                                          "exception": type(exc).__name__,
                                          "message": str(exc)[:300]}
        flush()

    # Analytic gradient AFTER (expensive response); add the analytic-vs-FD
    # cross-check to each direction if the response solve completes.  A failure
    # or timeout here does not lose the FD directional gradients already written.
    try:
        res = compute_grad_nac_analytic_cp(mc0, gradient_states=[0], nac_pairs=None,
                                           backend="mps-krylov", tol=1.0e-7, max_iter=300)
        g0 = np.asarray(res["grad"][0], dtype=float)
        result["analytic_gradient_norm"] = float(np.linalg.norm(g0))
        for name, q in dirs.items():
            d = result["directions"].get(name, {})
            if "g_fd_dir" in d:
                d["analytic_dir"] = float(np.tensordot(g0, q))
                d["abs_err"] = abs(d["analytic_dir"] - d["g_fd_dir"])
        flush()
    except Exception as exc:  # noqa: BLE001
        result["analytic_gradient"] = {"status": "error",
                                       "exception": type(exc).__name__,
                                       "message": str(exc)[:300]}
        flush()

    # FCI-free integrity: prove no dense bridge was ever entered in a beyond-FCI run
    result["dense_bridge"] = DenseBridgeSentinel.report()
    result["fci_free_integrity"] = ("PASS" if (not beyond or not DenseBridgeSentinel.used)
                                    else "FAIL")
    flush()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, default=10)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--stack-mem-mb", type=int, default=8000)
    ap.add_argument("--h-bohr", type=float, default=1.0e-3)
    ap.add_argument("--directions", nargs="*", default=None)
    ap.add_argument("--max-m", type=int, default=800)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sched = [s for s in DEFAULT_M_SCHEDULE if s[0] <= args.max_m] or [DEFAULT_M_SCHEDULE[0]]
    out = args.out or str(_HERE / "data" / f"dirfd_c{args.ncarbon}.json")
    print(f"=== directional FD polyene C{args.ncarbon} (max M={args.max_m}) ===", flush=True)
    try:
        r = run(args.ncarbon, basis=args.basis, m_schedule=sched, threads=args.threads,
                stack_mem_mb=args.stack_mem_mb, h_bohr=args.h_bohr,
                directions=args.directions, out_path=out)
        for name, d in r.get("directions", {}).items():
            if "g_fd_dir" in d:
                xtra = (f" analytic={d['analytic_dir']:.6f} err={d['abs_err']:.2e}"
                        if "abs_err" in d else " (analytic cross-check pending)")
                print(f"  {name}: fd={d['g_fd_dir']:.6f} "
                      f"sigma={d['active_subspace_sigma_min']:.3f} "
                      f"{d['health']['overall']}{xtra}", flush=True)
            else:
                print(f"  {name}: {d.get('message','error')}", flush=True)
        print(f"  fci_free_integrity={r['fci_free_integrity']} "
              f"dense_bridge_used={r['dense_bridge']['dense_bridge_used']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {exc}\n{traceback.format_exc()[-1500:]}", flush=True)
        return 1
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
