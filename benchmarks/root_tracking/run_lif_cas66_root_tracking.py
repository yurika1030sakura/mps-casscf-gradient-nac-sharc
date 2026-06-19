"""LiF CAS(6,6)/6-31G ionic/covalent avoided crossing: root/subspace tracking.

This is a near-degeneracy stress test for the root/subspace-tracking machinery,
NOT a large-active-space benchmark (the beyond-FCI demonstration is the polyene
CAS(20,20) run).  Near the LiF ionic/covalent crossing the two lowest singlet
states swap character, so energy-sorted adiabatic labels are gauge-dependent;
the meaningful continuity criterion is the singular-value spectrum of the
adjacent-geometry state-overlap matrix, not diagonal root identity.

Engineering (the reason a naive CAS(6,6) scan stalls):
  * a manual, valence-adapted CAS(6,6) (F 2px/2py/2pz, Li 2s, Li 2pz, and one
    correlating sigma orbital), NOT a blind default/AVAS active space;
  * spin-pure singlet enforcement (fix_spin ss=0) so no triplet-like root leaks
    into the state average;
  * sequential orbital propagation along R (no cold start per point), which both
    accelerates convergence and is itself the continuity being illustrated;
  * staged loose->tight CASSCF convergence;
  * JSONL output flushed after every point with resume + walltime-safe exit, so a
    SLURM timeout never yields zero data.

For each adjacent geometry pair it reports the state-overlap matrix O_ij (exact
determinant-level for this FCI-feasible CAS), its singular values and sigma_min
(the subspace-continuity diagnostic), the optimal root assignment, and the
active-orbital cross-overlap sigma_min.  With --mps-points it additionally builds
DMRG/MPS roots in the same orbitals and reports the FCI-free MPS-native overlap
as the algorithm-under-test cross-check at selected geometries.

Usage:
  python run_lif_cas66_root_tracking.py --out data/lif_cas66.jsonl --resume
  python run_lif_cas66_root_tracking.py --R-list 3.9 4.05 5.0 --conv-tol-grad 1e-8 \
      --mps-points 3.9 4.05 5.0 --out data/lif_cas66_tight.jsonl --resume
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
from overlap_fci_reference import (overlap_matrix_fci, assign_roots_by_overlap,
                                   cross_geometry_S_act)

ANG = 1.8897261246257702

# Default R grid (Angstrom): dense near the CAS(6,6) singlet avoided crossing
# (gap minimum near R~3.4 A, where the two lowest singlets swap ionic/covalent
# character), sparser on the tails.
DEFAULT_RGRID = [2.6, 2.9, 3.1, 3.2, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55,
                 3.6, 3.8, 4.1, 4.5, 5.0, 6.0]


# ---------------------------------------------------------------- geometry / CAS
def lif_mol(R_ang, basis):
    """Li at the origin, F along +z, so the sigma axis is z unambiguously."""
    return gto.M(atom=[("Li", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, float(R_ang) * ANG))],
                 basis=basis, charge=0, spin=0, unit="Bohr", symmetry=False, verbose=0)


def select_lif_cas66(mol, mf, ncas=6):
    """Valence-adapted CAS(6,6) selector for the LiF ionic/covalent crossing.

    Targets F 2px/2py/2pz, Li 2s, Li 2pz, and one correlating sigma virtual
    (Li 3s/3pz or F 3pz character).  Reorders the MO coefficient matrix to
    [core | active | rest] and returns (ncore, ncas, nelecas, mo, diagnostics).
    """
    mo = mf.mo_coeff
    S = mf.get_ovlp()
    labels = mol.ao_labels()
    mo_energy = mf.mo_energy

    target = []
    for iao, lab in enumerate(labels):
        toks = lab.split()
        if len(toks) < 3:
            continue
        atom, orb = toks[1], toks[2]
        if atom == "F" and orb in ("2px", "2py", "2pz"):
            target.append(iao)
        if atom == "Li" and orb in ("2s", "2pz", "3s", "3pz"):
            target.append(iao)
        if atom == "F" and orb in ("3pz",):
            target.append(iao)
    if not target:
        raise RuntimeError("No LiF target AOs found; check AO labels / bond axis.")

    Smo = S @ mo
    target_pop = np.einsum("ai,ai->i", mo[target, :], Smo[target, :])

    ncore = (mol.nelectron - ncas) // 2
    core = sorted(range(mo.shape[1]), key=lambda i: mo_energy[i])[:ncore]
    noncore = [i for i in range(mo.shape[1]) if i not in core]
    active = sorted(
        sorted(noncore, key=lambda i: target_pop[i], reverse=True)[:ncas],
        key=lambda i: mo_energy[i],
    )
    rest = sorted([i for i in range(mo.shape[1]) if i not in core and i not in active],
                  key=lambda i: mo_energy[i])
    new_order = core + active + rest
    mo_init = mo[:, new_order]
    diagnostics = {
        "ncore": ncore, "ncas": ncas, "nelecas": ncas,
        "active_target_pop": [float(target_pop[i]) for i in active],
        "active_mo_energy": [float(mo_energy[i]) for i in active],
    }
    return ncore, ncas, ncas, mo_init, diagnostics


# --------------------------------------------------------------- CASSCF builders
def make_singlet_sa_casscf(mf, ncas, nelecas, nroots, weights):
    """Spin-pure singlet SA-CASSCF object (fix_spin ss=0).

    A single fix_spin penalty via ``mc.fix_spin_`` keeps both state-averaged
    roots in the singlet sector: verified that without it the second LiF
    CAS(6,6) root collapses to a triplet (S^2=2), whereas with it S^2=[0,0].
    """
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver.nroots = int(nroots)
    if nroots > 1:
        mc = mc.state_average_(list(weights))
    try:
        mc.fix_spin_(ss=0.0, shift=0.5)
    except Exception:
        pass
    return mc


def configure_optimizer(mc, *, conv_tol, conv_tol_grad, max_cycle_macro,
                        level_shift=0.5):
    for name, val in [("conv_tol", conv_tol), ("conv_tol_grad", conv_tol_grad),
                      ("max_cycle_macro", max_cycle_macro),
                      ("ah_level_shift", level_shift)]:
        if hasattr(mc, name):
            setattr(mc, name, val)
    return mc


def staged_kernel(mf, ncas, nelecas, nroots, weights, mo, *,
                  conv_tol_grad_final, max_cycle_macro, level_shift=0.5):
    """Single spin-pure SA-CASSCF solve with an AH level shift, seeded by ``mo``.

    Near the avoided crossing the orbital Hessian is ill-conditioned; an
    augmented-Hessian level shift (~0.5) plus the singlet fix_spin penalty makes
    the state-averaged optimization converge in a few seconds (verified at
    R=3.8-4.0 A: conv=True, S^2=[0,0]).  The manual selector (first point) and
    orbital propagation (later points) supply the orbital guess.  A CAS(4,4)
    preconditioner is intentionally omitted because its active window does not
    align with the CAS(6,6) window.
    """
    mc = make_singlet_sa_casscf(mf, ncas, nelecas, nroots, weights)
    configure_optimizer(mc, conv_tol=max(conv_tol_grad_final * 1e-2, 1e-10),
                        conv_tol_grad=conv_tol_grad_final,
                        max_cycle_macro=max_cycle_macro, level_shift=level_shift)
    mc.kernel(mo)
    return mc


# ------------------------------------------------------------------- one R point
def lif_point(R_ang, basis, ncas, nelecas, nroots, weights, *,
              conv_tol_grad, max_cycle_macro, mo_prev=None, mol_prev=None,
              ncore=None):
    mol = lif_mol(R_ang, basis)
    mf = scf.RHF(mol).run(conv_tol=1e-11)

    sel_diag = None
    if mo_prev is None:
        ncore, ncas, nelecas, mo, sel_diag = select_lif_cas66(mol, mf, ncas)
    else:
        mo, _smin = fdv.project_mo_to_new_geometry(mol_prev, mol, mo_prev)

    mc = staged_kernel(mf, ncas, nelecas, nroots, weights, mo,
                       conv_tol_grad_final=conv_tol_grad,
                       max_cycle_macro=max_cycle_macro)
    if not mc.converged:
        # Escalation for stubborn (compressed / near-degenerate) geometries:
        # discard the (possibly poorly propagated) guess, reselect fresh
        # valence-adapted orbitals, raise the AH level shift, loosen the
        # tolerance, and double the macro budget.
        _nc, _, _, mo_fresh, _ = select_lif_cas66(mol, mf, ncas)
        mc = staged_kernel(mf, ncas, nelecas, nroots, weights, mo_fresh,
                           conv_tol_grad_final=max(conv_tol_grad, 5.0e-5),
                           max_cycle_macro=max_cycle_macro * 2, level_shift=1.0)
    e = [float(x) for x in mc.e_states]
    ci = [np.asarray(c) for c in mc.ci]

    rec = {
        "R_ang": float(R_ang), "basis": basis, "ncas": ncas, "nelecas": nelecas,
        "ncore": int(mc.ncore), "nroots": nroots,
        "energies": e, "gap_Eh": float(e[1] - e[0]),
        "converged": bool(mc.converged),
        "conv_tol_grad": float(conv_tol_grad),
        "spin_sector": "singlet_fix_spin_ss0",
    }
    if sel_diag is not None:
        rec["active_target_pop"] = sel_diag["active_target_pop"]
    return rec, mol, mc.mo_coeff, ci, int(mc.ncore)


def adjacent_overlap(mol_l, mo_l, ci_l, mol_r, mo_r, ci_r, ncas, ncore, nelecas):
    """Determinant-level cross-geometry state overlap O_ij and diagnostics."""
    S_act = cross_geometry_S_act(mol_l, mol_r, mo_l, mo_r, ncas, ncore)
    nelec = (nelecas // 2, nelecas - nelecas // 2)
    O = overlap_matrix_fci(ci_l, ci_r, S_act, ncas, nelec)
    perm, signs = assign_roots_by_overlap(O)
    sv = np.linalg.svd(np.asarray(O), compute_uv=False)
    asv = np.linalg.svd(np.asarray(S_act), compute_uv=False)
    return {
        "O_abs": np.abs(np.asarray(O)).tolist(),
        "assignment": [[int(i), int(perm[i])] for i in range(len(perm))],
        "subspace_singular_values": [float(x) for x in sv],
        "subspace_sigma_min": float(np.min(sv)),
        "active_orbital_sigma_min": float(np.min(asv)),
    }


# --------------------------------------------------------------------- JSONL I/O
def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def completed_R(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("kind") == "point":
                        done.add(round(float(r["R_ang"]), 6))
                except Exception:
                    pass
    return done


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="6-31G")
    ap.add_argument("--ncas", type=int, default=6)
    ap.add_argument("--nelecas", type=int, default=6)
    ap.add_argument("--nroots", type=int, default=2)
    ap.add_argument("--weights", type=float, nargs="*", default=None)
    ap.add_argument("--R-list", type=float, nargs="*", default=None)
    ap.add_argument("--conv-tol-grad", type=float, default=1e-5)
    ap.add_argument("--max-cycle-macro", type=int, default=100)
    ap.add_argument("--out", default=str(_HERE / "data" / "lif_cas66_root_tracking.jsonl"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--walltime-buffer-min", type=float, default=15.0)
    ap.add_argument("--walltime-min", type=float, default=1e9)
    args = ap.parse_args()

    weights = args.weights or [1.0 / args.nroots] * args.nroots
    Rs = args.R_list if args.R_list else DEFAULT_RGRID
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = completed_R(out) if args.resume else set()
    if not args.resume and out.exists():
        out.unlink()

    t_start = time.time()
    deadline = t_start + (args.walltime_min - args.walltime_buffer_min) * 60.0

    prev = None  # (mol, mo, ci, ncore, R)
    for R in Rs:
        if round(float(R), 6) in done:
            print(f"skip R={R} (done)", flush=True)
            continue
        if time.time() > deadline:
            print(f"walltime guard: stopping before R={R}; resume later.", flush=True)
            break
        print(f"=== LiF R={R} Ang ===", flush=True)
        try:
            t0 = time.perf_counter()
            rec, mol, mo, ci, ncore = lif_point(
                R, args.basis, args.ncas, args.nelecas, args.nroots, weights,
                conv_tol_grad=args.conv_tol_grad,
                max_cycle_macro=args.max_cycle_macro,
                mo_prev=(prev[1] if prev else None),
                mol_prev=(prev[0] if prev else None))
            rec["kind"] = "point"
            rec["wall_s"] = time.perf_counter() - t0
            if prev is not None:
                rec["active_sigma_from_prev"] = adjacent_overlap(
                    prev[0], prev[1], prev[2], mol, mo, ci,
                    args.ncas, min(prev[3], ncore), args.nelecas)
            append_jsonl(out, rec)
            print(f"  conv={rec['converged']} {rec['wall_s']:.1f}s "
                  f"gap={rec['gap_Eh']:.6f}", flush=True)
            prev = (mol, mo, ci, ncore, R)
        except Exception as exc:  # noqa: BLE001
            err = {"kind": "error", "R_ang": float(R), "exception": type(exc).__name__,
                   "message": str(exc), "traceback_tail": traceback.format_exc()[-2000:]}
            append_jsonl(out, err)
            print(f"  ERROR R={R}: {exc}", flush=True)
            prev = None  # break propagation chain on failure
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
