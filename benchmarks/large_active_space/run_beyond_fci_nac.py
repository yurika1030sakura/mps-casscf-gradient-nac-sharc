"""Beyond-FCI nonadiabatic coupling by FCI-free finite differences (polyene).

Assembles the bra-fixed derivative coupling d^x_{01} between the two lowest
SA(2)-DMRG-CASSCF states of an all-trans polyene pi system entirely from MPS
operations -- the displaced states at R+-h are transported into the reference
driver, rotated into the reference orbital basis, overlapped, and differenced --
so no FCI vector is ever formed.  At a small pi space (CAS(10,10), determinant
space 6.3e4) the result is cross-checked against the determinant-level
finite-difference NAC; at CAS(20,20) (determinant space 3.4e10, FCI impossible)
the MPS-native value is reported with its internal diagnostics (active-subspace
singular values, root gap, certified response residual of the analytic NAC).

Usage:  python run_beyond_fci_nac.py --ncarbon 10   # FCI-checked
        python run_beyond_fci_nac.py --ncarbon 20   # beyond FCI
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
SHARC = _HERE.parents[1] / "sharc_interface"
for p in (str(DEV), str(SHARC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import block2
import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response, _nac_one_pair_mps_krylov
from cross_geometry_overlap import cross_geometry_overlap_matrix
from run_polyene_beyond_fci import (polyene_geometry, select_pi_active, det_dim,
                                    beyond_fci_solver_cfg, FCI_THRESHOLD)

ANG = 1.8897261246257702


def aza_polyene_geometry(n):
    """Polyene with the terminal =CH2 replaced by =NH (Schiff-base model): breaks
    the L<->R symmetry that makes the pure-polyene soft modes near-degenerate,
    giving a healthy S0/S1 gap and an admissible NAC, while keeping an n-center pi
    system (CAS(n,n)).  Mirrors the aza gradient datapoint."""
    atoms = list(polyene_geometry(n))
    cx, cy, cz = atoms[n - 1][1]
    atoms[n - 1] = ("N", (cx, cy, cz))
    return atoms[:len(atoms) - 1] + atoms[len(atoms):]  # drop the axial terminal H


def select_pi_hetero(mol, mf, n_pi):
    """pi-space selection over C AND N pz AOs (for aza-polyenes)."""
    mo = mf.mo_coeff; S = mf.get_ovlp(); labels = mol.ao_labels()
    pz = [i for i, l in enumerate(labels)
          if len(l.split()) >= 3 and l.split()[1] in ("C", "N") and l.split()[2].endswith("pz")]
    Smo = S @ mo
    pichar = np.einsum("ai,ai->i", mo[pz, :], Smo[pz, :])
    active = sorted(np.argsort(-pichar)[:n_pi])
    nocc = mol.nelectron // 2; ncore = nocc - n_pi // 2
    rest = [i for i in range(mo.shape[1]) if i not in active]
    order = rest[:ncore] + list(active) + rest[ncore:]
    ncas = n_pi; nelecas = n_pi
    return ncas, nelecas, mo[:, order]


def _build(symbols, coords_bohr, basis, ncas, nelecas, cfg, mo_guess, nroots=2, mps_dir=None):
    # nroots>2 spans the near-degenerate excited manifold (the polyene 1Bu/2Ag pair)
    # so the averaged state set is stable across the displacement.  When mps_dir is
    # given the build reloads the MPS already stored there as its initial guess (basin
    # pinning): the reference build saves its MPS, and the displaced builds reload it,
    # so the displaced SA-CASSCF inherits the reference basin rather than re-converging
    # to a possibly different stationary point on the non-convex landscape (Sec. branch).
    from run_cas_directional_fd import build_progressive, DEFAULT_M_SCHEDULE
    w = [1.0 / nroots] * nroots
    mol, mc, solver, _blog = build_progressive(
        symbols, coords_bohr, basis, ncas, nelecas,
        m_schedule=DEFAULT_M_SCHEDULE, mo_guess=mo_guess,
        threads=int(cfg.get("n_threads", 8)),
        stack_mem_mb=int(cfg.get("stack_mem_mb", 8000)),
        nroots=int(nroots), weights=w, mps_persistent_dir=mps_dir)
    return mol, mc, solver


def _pi_space(symbols, coords_ang, basis, n_carbon, aza=False):
    from pyscf import gto, scf
    mol = gto.M(atom=[(symbols[i], tuple(coords_ang[i])) for i in range(len(symbols))],
                basis=basis, charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    return select_pi_hetero(mol, mf, n_carbon) if aza else select_pi_active(mol, mf, n_carbon)


def run(n_carbon, *, basis="sto-3g", bond_dim=600, threads=8,
        stack_mem_mb=8000, h_bohr=1.0e-3, atom_disp=None, comp=1, aza=False, nroots=2,
        nac_max_iter=30, nac_response_tol=1.0e-3, nac_solver="gmres",
        nac_m_compress=None):
    geomfn = (lambda nn: aza_polyene_geometry(nn)) if aza else polyene_geometry
    symbols0 = [a[0] for a in geomfn(n_carbon)]
    coords0 = np.array([a[1] for a in geomfn(n_carbon)])
    centre = n_carbon // 2 if atom_disp is None else int(atom_disp)

    def coords_at(sign):
        c = coords0.copy()
        c[centre, comp] += sign * h_bohr / ANG     # displace in Angstrom units
        return c

    geoms = {"R": coords0, "P": coords_at(+1.0), "M": coords_at(-1.0)}
    ncas, nelecas, mo0 = _pi_space(symbols0, coords0, basis, n_carbon, aza=aza)
    cfg = beyond_fci_solver_cfg(ncas, bond_dim, threads, stack_mem_mb)
    ddim = det_dim(ncas, (nelecas // 2, nelecas - nelecas // 2))
    beyond_fci = ddim >= FCI_THRESHOLD

    from pyscf import gto as _gto
    import os, shutil
    # Single persistent MPS scratch shared by R -> P -> M (basin pinning).
    # Job-unique pin_dir: SLURM_JOB_ID is appended so two concurrent runs on the same
    # (system, nroots) cannot collide on the persistent MPS scratch. Previously all
    # runs shared "nacpin_..._sa<nroots>" which made a second job silently trample
    # a first job's MPS files.
    _jid = os.environ.get("SLURM_JOB_ID", "local")
    pin_dir = os.path.join("/tmp/dmrg_scratch",
                           f"nacpin_{'aza_' if aza else ''}c{n_carbon}_sa{nroots}_j{_jid}")
    shutil.rmtree(pin_dir, ignore_errors=True); os.makedirs(pin_dir, exist_ok=True)
    built = {}
    # R/P/M each get a SEPARATE MPS subdir. The displaced builds warm-start from the
    # reference ORBITALS (projected -> continuous gauge, no det=-1 reflection), which
    # keeps a symmetry-broken system on the reference basin, while keeping the +/-
    # displaced MPS DISTINCT. Sharing one dir made the + and - MPS overwrite each other
    # and collapsed the cross-geometry finite difference to machine zero.
    dir_R = os.path.join(pin_dir, "R"); os.makedirs(dir_R, exist_ok=True)
    _, _, mo_R0 = _pi_space(symbols0, coords0, basis, n_carbon, aza=aza)
    molR, mcR, solR = _build(symbols0, coords0 * ANG, basis, ncas, nelecas, cfg, mo_R0,
                             nroots=nroots, mps_dir=dir_R)
    built["R"] = dict(mol=molR, mc=mcR, solver=solR)
    mo_ref = np.array(mcR.mo_coeff)
    # Displaced builds warm-start from the reference ORBITALS (projected, continuous
    # sign gauge -> no det=-1 reflection) AND from the reference MPS in pin_dir (basin
    # pinning -> same SA-CASSCF branch as R).  The orbital warm-start alone keeps the
    # gauge continuous but does not pin the basin on the non-convex landscape; the MPS
    # reload does.  R -> P -> M share pin_dir sequentially, so both displaced geometries
    # inherit the reference basin without copying MPS files between directories.
    for key in ("P", "M"):
        c_ang = geoms[key]
        mol_disp = _gto.M(atom=[(symbols0[i], tuple(c_ang[i])) for i in range(len(symbols0))],
                          basis=basis, charge=0, spin=0, verbose=0)
        mo_guess = fdv.project_mo_to_new_geometry(molR, mol_disp, mo_ref)[0]
        dir_key = os.path.join(pin_dir, key); os.makedirs(dir_key, exist_ok=True)
        mol, mc, solver = _build(symbols0, c_ang * ANG, basis, ncas, nelecas, cfg, mo_guess,
                                 nroots=nroots, mps_dir=dir_key)
        built[key] = dict(mol=mol, mc=mc, solver=solver)

    # ---- SOLUTION-CONSISTENCY GATE on the three reference energies -------------
    # A displaced build can be perfectly gauge-continuous (active-orbital overlap
    # s ~ identity, sigma_min ~ 1) and STILL have converged to a different, lower
    # state-averaged CASSCF solution.  The orbital diagnostics cannot see this; the
    # mutual consistency of the three SA reference energies can.  A displaced solution
    # that sits ~1e-4 Eh below the other two corrupts the finite difference by orders
    # of magnitude (observed at pentacene CAS(22,22)).  Report the spread and flag it.
    _e_ref = {k: float(np.asarray(built[k]["mc"].e_states, dtype=float).ravel()[0])
              for k in ("R", "P", "M")}
    _spread = max(_e_ref.values()) - min(_e_ref.values())
    _consistent = bool(_spread <= 1.0e-5)
    _verdict = ("PASS" if _consistent else
                "FAIL (a displaced build landed on a DIFFERENT SA-CASSCF solution; "
                "the finite difference is not admissible)")
    print(f"[build-consistency] E_SA(R/P/M) = "
          f"{_e_ref['R']:.8f} / {_e_ref['P']:.8f} / {_e_ref['M']:.8f}  "
          f"spread={_spread:.2e} Eh  -> {_verdict}", flush=True)

    # response objects, each pinned to its own driver frame
    for key in ("R", "P", "M"):
        block2.Global.frame = built[key]["mc"].fcisolver._driver.frame
        built[key]["obj"] = _make_mps_krylov_response(built[key]["mc"])

    R, P, M = built["R"], built["P"], built["M"]
    ncore = R["mc"].ncore
    host_frame = R["mc"].fcisolver._driver.frame

    t0 = time.perf_counter()
    Op, s_p = cross_geometry_overlap_matrix(
        R["obj"], P["obj"], R["mol"], P["mol"], R["mc"].mo_coeff, P["mc"].mo_coeff,
        ncore, ncas, int(nroots), host_frame=host_frame, tag="BFNP")
    Om, s_m = cross_geometry_overlap_matrix(
        R["obj"], M["obj"], R["mol"], M["mol"], R["mc"].mo_coeff, M["mc"].mo_coeff,
        ncore, ncas, int(nroots), host_frame=host_frame, tag="BFNM")
    nac_wall = time.perf_counter() - t0
    d_mps = float((Op[0, 1] - Om[0, 1]) / (2.0 * h_bohr))

    sig_p = float(np.min(np.linalg.svd(s_p, compute_uv=False)))
    sig_m = float(np.min(np.linalg.svd(s_m, compute_uv=False)))
    e = list(np.asarray(R["mc"].e_states, dtype=float).ravel())

    # analytic S0/S1 NAC at R (MPS-Krylov) -- the symmetric analytic-vs-FD validation
    # mirroring the gradient.  Cap the response bond dim so the GMRES Krylov MPS
    # does not grow unbounded (the acene path uses the same cap).
    block2.Global.frame = R["mc"].fcisolver._driver.frame
    # Response-MPS compression cap. The default (400) suffices for a well-conditioned,
    # low-amplitude coupling, but a small-gap system has a high-amplitude, higher-rank
    # response solution that a 400-state compression cannot represent -- at pentacene
    # the fix required BOTH the conjugate-residual solver AND raising this cap to 600.
    R["obj"]._m_compress = (int(nac_m_compress) if nac_m_compress
                            else min(int(bond_dim), 400))
    # cr (conjugate-residual) avoids GMRES's O(k^2) Arnoldi hang for high-amplitude
    # (small-gap, large-NAC) solutions; gmres is the default for well-conditioned cases.
    R["obj"]._linear_solver = str(nac_solver).strip().lower()
    # For an FCI-ACCESSIBLE polyene (det < FCI_THRESHOLD) the analytic MPS-Krylov solve
    # is both redundant (naphthalene CAS(10,10) already anchors analytic-vs-FCI NAC) and
    # pathological (the linear-polyene 1Bu/2Ag near-degeneracy makes the CP-CASSCF
    # response singular so GMRES stagnates ~2-4e-3 and the downstream Lagrangian assembly
    # hangs). Skip it and report only the cross-geometry MPS-native FD for such systems.
    if beyond_fci:
        t1 = time.perf_counter()
        de_nac = _nac_one_pair_mps_krylov(R["mc"], R["obj"], (0, 1),
                                          tol=float(nac_response_tol), max_iter=int(nac_max_iter))
        d_analytic = float(de_nac[centre, comp])
        nac_analytic_wall = time.perf_counter() - t1
    else:
        d_analytic = None
        nac_analytic_wall = 0.0

    out = {
        "system": f"{'azapolyene' if aza else 'polyene'}_C{n_carbon}", "basis": basis,
        "aza": bool(aza),
        "ncas": ncas, "nelecas": nelecas, "det_dim": ddim,
        "fci_feasible": not beyond_fci,
        "bond_dim": bond_dim, "h_bohr": h_bohr,
        "displaced_atom": centre, "component": comp,
        "d_nac_mps_native_01": d_mps,
        "d_nac_analytic_01": d_analytic,
        "abs_err_analytic_vs_fd": (abs(abs(d_analytic) - abs(d_mps))
                                   if d_analytic is not None else None),
        "nac_analytic_wall_s": nac_analytic_wall,
        "active_subspace_sigma_min_plus": sig_p,
        "active_subspace_sigma_min_minus": sig_m,
        "nroots": int(nroots),
        "gap_Eh": float(abs(e[1] - e[0])),
        "gap_12_Eh": float(abs(e[2] - e[1])) if len(e) > 2 else None,
        "nac_mps_wall_s": nac_wall,
        "fci_conversion": bool(not cfg["skip_kernel_fci_conversion"]),
        "e_sa_ref_RPM": _e_ref,
        "build_consistency_spread_Eh": _spread,
        "build_consistency_pass": _consistent,
    }

    # FCI cross-check where the determinant space still allows it
    if not beyond_fci:
        cfg_fci = dict(fdv.DEFAULT_SOLVER_CFG)
        cfg_fci.update(bond_dim=bond_dim, n_sweeps=30, sweep_tol=1.0e-10,
                       n_threads=int(threads), stack_mem_mb=int(stack_mem_mb))
        d_fci = fdv.fd_nac(symbols0, coords0 * ANG, bra=0, ket=1, basis=basis,
                           charge=0, spin=0, ncas=ncas, nelecas=nelecas, nroots=2,
                           weights=[0.5, 0.5], solver_cfg=cfg_fci, h_bohr=h_bohr,
                           atmlst=[centre], components=[comp])
        d_fci_01 = float(d_fci[centre, comp])
        out["d_nac_fci_fd_01"] = d_fci_01
        out["abs_err_mps_vs_fci"] = abs(abs(d_mps) - abs(d_fci_01))
        out["cross_check"] = "pass" if out["abs_err_mps_vs_fci"] < 1.0e-3 else "fail"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, default=10)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--bond-dim", type=int, default=600)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--stack-mem-mb", type=int, default=8000)
    ap.add_argument("--h-bohr", type=float, default=1.0e-3)
    ap.add_argument("--aza", action="store_true",
                    help="terminal =CH2 -> =NH Schiff-base heteroatom polyene (healthy gap, clean NAC)")
    ap.add_argument("--nroots", type=int, default=2,
                    help="state-averaging size; 3 includes the full 1Bu/2Ag near-degenerate "
                         "manifold so the averaged state set is stable across the displacement")
    ap.add_argument("--nac-max-iter", type=int, default=30,
                    help="MPS-Krylov NAC response max iterations (higher = tighter residual, longer wall)")
    ap.add_argument("--nac-response-tol", type=float, default=1.0e-3,
                    help="MPS-Krylov NAC response tolerance target")
    ap.add_argument("--nac-solver", default="gmres",
                    help="NAC response linear solver: gmres | cr (conjugate-residual)")
    ap.add_argument("--nac-m-compress", type=int, default=None,
                    help="cap on the response Krylov MPS bond dim (default min(bond_dim,400)); raise for small-gap, high-amplitude couplings")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = f"{'azapolyene' if args.aza else 'polyene'} C{args.ncarbon} SA({args.nroots})"
    print(f"=== beyond-FCI NAC {tag} ===", flush=True)
    try:
        r = run(args.ncarbon, basis=args.basis, bond_dim=args.bond_dim,
                threads=args.threads, stack_mem_mb=args.stack_mem_mb,
                h_bohr=args.h_bohr, aza=args.aza, nroots=args.nroots,
                nac_max_iter=args.nac_max_iter, nac_response_tol=args.nac_response_tol,
                nac_solver=args.nac_solver, nac_m_compress=args.nac_m_compress)
        print(json.dumps(r, indent=2), flush=True)
    except Exception as exc:  # noqa: BLE001
        r = {"system": tag, "status": "error",
             "exception": type(exc).__name__, "message": str(exc),
             "traceback_tail": traceback.format_exc()[-3000:]}
        print(f"  ERROR: {exc}", flush=True)
    _sa = "" if args.nroots == 2 else f"_sa{args.nroots}"
    _fn = f"beyond_fci_nac_{'aza_c' if args.aza else 'c'}{args.ncarbon}{_sa}.json"
    out = Path(args.out) if args.out else _HERE / "data" / _fn
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
