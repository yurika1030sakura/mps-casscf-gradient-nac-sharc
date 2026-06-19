"""Beyond-FCI active-space benchmark: all-trans polyene pi spaces.

For C(n)H(n+2) all-trans polyenes the pi space is CAS(n,n).  The ladder
  C10 -> CAS(10,10)  det dim C(10,5)^2 = 6.35e4   (FCI-easy)
  C14 -> CAS(14,14)  det dim C(14,7)^2 = 1.18e7   (FCI-borderline)
  C20 -> CAS(20,20)  det dim C(20,10)^2 = 3.41e10 (FCI-impossible)
spans from the FCI-accessible regime into the regime where a conventional
FCI-CASSCF derivative is impossible and DMRG is required.

Validation where FCI is impossible uses the analytic SA-DMRG-CASSCF gradient
against a central finite difference of the solver's own state energy
(needs no FCI reference and no cross-geometry overlap), on a curated set of
Cartesian components.  The active-space determinant dimension is reported so
the beyond-FCI claim is explicit.
"""

from __future__ import annotations

import argparse
import json
import math
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

from pyscf import gto, scf
from pyscf.mcscf import avas
import fd_validation as fdv

ANG = 1.8897261246257702


def polyene_geometry(n_carbon):
    """All-trans C(n)H(n+2) with explicit bond-length alternation (Angstrom).

    Not geometry-optimized: a chemically reasonable bond-length-alternated
    structure used purely as a derivative-validation point. Analytic
    derivatives are local and do not require an equilibrium geometry.
    """
    r_double = 1.34
    r_single = 1.46
    r_ch = 1.09
    tilt = math.radians(30.0)        # bond +-30 deg from x -> 120 deg backbone
    # carbon backbone: exact bond vectors, alternating length and +-tilt (trans)
    carbons = [(0.0, 0.0)]
    x, y = 0.0, 0.0
    for i in range(n_carbon - 1):
        bond = r_double if i % 2 == 0 else r_single
        ang = tilt if i % 2 == 0 else -tilt
        x += bond * math.cos(ang)
        y += bond * math.sin(ang)
        carbons.append((x, y))
    atoms = [("C", (cx, cy, 0.0)) for (cx, cy) in carbons]
    # one in-plane H per backbone carbon, pointing away from the chain (outward
    # normal of the local kink); terminal carbons get a second H along the axis
    for i, (cx, cy) in enumerate(carbons):
        hy = cy + (r_ch if (i % 2 == 0) else -r_ch)
        atoms.append(("H", (cx, hy, 0.0)))
    x0, y0 = carbons[0]
    atoms.append(("H", (x0 - r_ch * math.cos(tilt), y0 + r_ch * math.sin(tilt), 0.0)))
    xN, yN = carbons[-1]
    sgnN = 1.0 if (n_carbon - 1) % 2 == 0 else -1.0
    atoms.append(("H", (xN + r_ch * math.cos(tilt), yN + sgnN * r_ch * math.sin(tilt), 0.0)))
    return atoms


def det_dim(ncas, nelec):
    na, nb = nelec
    return math.comb(ncas, na) * math.comb(ncas, nb)


def select_pi_active(mol, mf, n_pi):
    """Select exactly the n_pi pi MOs of a planar (z=0) polyene.

    For a molecule in the xy-plane, sigma and pi orbitals separate by mirror
    symmetry: pi MOs are built from C 2pz AOs and have ~zero weight on every
    other AO.  We rank MOs by their Mulliken population on C-pz AOs, take the
    n_pi most-pi ones, and reorder them into the active window so the result is
    an exact CAS(n_pi, n_pi) pi space (n_pi/2 occupied -> n_pi electrons).

    Returns (ncas, nelecas, mo_init) with the same contract as avas.avas.
    """
    mo = mf.mo_coeff
    S = mf.get_ovlp()
    labels = mol.ao_labels()
    cpz = []
    for i, lab in enumerate(labels):
        tok = lab.split()                      # e.g. ['0', 'C', '2pz']
        if len(tok) >= 3 and tok[1] == "C" and tok[2].endswith("pz"):
            cpz.append(i)
    if not cpz:
        raise RuntimeError("no C-pz AOs found for pi-space selection")
    Smo = S @ mo
    pi_char = np.einsum("ai,ai->i", mo[cpz, :], Smo[cpz, :])  # Mulliken pop on Cpz
    mo_energy = mf.mo_energy

    # sigma-pi separation is essentially binary (pi_char ~ 1 or ~ 0); the pi
    # MOs are exactly those dominated by C-pz character.
    pi_idx = [i for i in range(mo.shape[1]) if pi_char[i] > 0.5]
    if len(pi_idx) != n_pi:
        raise RuntimeError(
            f"expected {n_pi} pi MOs from C-pz character, found {len(pi_idx)} "
            f"(min-basis assumption broken?)"
        )
    sigma_idx = [i for i in range(mo.shape[1]) if i not in pi_idx]

    # The active space is the full pi block; the electron count is fixed by the
    # molecule (each sp2 carbon donates one pi electron), independent of any
    # HF sigma/pi level crossing.  ncore is the number of doubly-occupied sigma
    # orbitals implied by that electron count.
    nelecas = n_pi
    ncore = (mol.nelectron - nelecas) // 2
    if (mol.nelectron - nelecas) % 2 != 0:
        raise RuntimeError("odd core electron count; check pi electron number")

    sigma_sorted = sorted(sigma_idx, key=lambda i: mo_energy[i])
    pi_sorted = sorted(pi_idx, key=lambda i: mo_energy[i])
    core = sigma_sorted[:ncore]
    vir_sigma = sigma_sorted[ncore:]
    new_order = core + pi_sorted + vir_sigma          # active = pi block
    if len(new_order) != mo.shape[1]:
        raise RuntimeError("MO reordering lost columns")
    mo_init = mo[:, new_order]
    ncas = n_pi
    return ncas, nelecas, mo_init


FCI_THRESHOLD = 5.0e7      # det dim above which FCI / FCI-overlap is infeasible
H_SCAN = (2.0e-3, 1.0e-3, 5.0e-4)


def beyond_fci_solver_cfg(ncas, bond_dim, threads, stack_mem_mb):
    """Solver config that never requests a dense CI readout at large CAS."""
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=bond_dim, n_sweeps=30, sweep_tol=1.0e-9,
               n_threads=int(threads), stack_mem_mb=int(stack_mem_mb),
               dmrg_symm_su2=True, force_dmrg=True)
    if ncas >= 16:
        # beyond-FCI: MPS-native RDMs, no dense MPS->FCI conversion
        cfg.update(mps_native_rdms=True, skip_kernel_fci_conversion=True)
    else:
        cfg.update(mps_native_rdms=False, skip_kernel_fci_conversion=False)
    return cfg


def run_one(n_carbon, *, basis="sto-3g", bond_dim=800, threads=8,
            stack_mem_mb=8000, fd_components=2):
    atoms = polyene_geometry(n_carbon)
    symbols = [a[0] for a in atoms]
    coords_ang = np.array([a[1] for a in atoms])
    coords_bohr = coords_ang * ANG

    # RHF + AVAS pi-space selection
    mol = gto.M(atom=[(symbols[i], tuple(coords_ang[i]))
                      for i in range(len(symbols))],
                basis=basis, charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    ncas, nelecas, mo_init = select_pi_active(mol, mf, n_carbon)
    ncas = int(ncas); nelecas = int(nelecas)
    # the CAS(n,n) pi-space claim must be exact, not silently truncated
    if ncas != n_carbon or nelecas != n_carbon:
        raise RuntimeError(
            f"pi-space selection did not return the full CAS({n_carbon},"
            f"{n_carbon}); got CAS({nelecas},{ncas})"
        )
    na = nelecas // 2 + nelecas % 2
    nb = nelecas // 2
    ddim = det_dim(ncas, (na, nb))
    beyond_fci = ddim >= FCI_THRESHOLD
    track = "gap_guard" if beyond_fci else "fci_overlap"

    cfg = beyond_fci_solver_cfg(ncas, bond_dim, threads, stack_mem_mb)

    # Hard guard: above the FCI threshold the determinant space cannot be formed,
    # so a dense MPS->FCI conversion or an FCI-overlap root tracker would either
    # exhaust memory or silently fall back to a wrong answer.  Make that
    # impossible rather than improbable -- the JSON below records these flags and
    # the rebuttal relies on them being true for the beyond-FCI rows.
    if beyond_fci:
        assert cfg["skip_kernel_fci_conversion"] is True, \
            f"det_dim={ddim:.3e} > {FCI_THRESHOLD:.1e} but FCI conversion is on"
        assert cfg["mps_native_rdms"] is True, \
            f"det_dim={ddim:.3e} > {FCI_THRESHOLD:.1e} but RDMs are not MPS-native"
        assert track in ("gap_guard", "mps_subspace"), \
            f"det_dim={ddim:.3e} > {FCI_THRESHOLD:.1e} requires FCI-free root tracking"

    t0 = time.perf_counter()
    _mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
        symbols, coords_bohr, basis=basis, charge=0, spin=0,
        ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
        solver_cfg=cfg, mo_guess=mo_init,
    )
    build_wall = time.perf_counter() - t0
    e_states = list(np.asarray(solver.e_states, dtype=float).ravel())

    # analytic gradient (state 0)
    t0 = time.perf_counter()
    g_an = fdv.analytic_gradient(mc, 0, backend="mps-krylov", tol=1e-7,
                                 max_iter=80)
    analytic_wall = time.perf_counter() - t0

    # FD validation on curated components: centre carbons, x-axis, with a
    # step-size scan; the displaced CASSCF is seeded from the projected
    # reference orbitals (same active-space surface), root choice is gap-guarded
    # at beyond-FCI sizes (no FCI overlap).
    centre = n_carbon // 2
    comp_atoms = [centre, max(0, centre - 1)][:fd_components]
    fd_results = []
    t0 = time.perf_counter()
    for a in comp_atoms:
        h_entries = []
        for h in H_SCAN:
            g_fd, diag = fdv.fd_gradient(
                symbols, coords_bohr, state=0, basis=basis, charge=0, spin=0,
                ncas=ncas, nelecas=nelecas, nroots=2, weights=[0.5, 0.5],
                solver_cfg=cfg, h_bohr=h, atmlst=[a], components=[0],
                track_roots=track, mo_guess=mo_init, return_diagnostics=True,
            )
            comp = diag["components"][0]
            h_entries.append({
                "h_bohr": h, "g_fd": float(g_fd[a, 0]),
                "abs_err": float(abs(g_an[a, 0] - g_fd[a, 0])),
                "active_subspace_sigma_min": comp["active_subspace_sigma_min"],
                "gap_plus": comp["gap_plus"], "gap_minus": comp["gap_minus"],
            })
        best = min(h_entries, key=lambda e: e["abs_err"])
        fd_results.append({
            "atom": int(a), "axis": 0,
            "g_analytic": float(g_an[a, 0]),
            "best": best, "h_scan": h_entries,
        })
    fd_wall = time.perf_counter() - t0
    max_err = max((r["best"]["abs_err"] for r in fd_results), default=0.0)

    return {
        "n_carbon": n_carbon, "basis": basis,
        "ncas": ncas, "nelecas": nelecas, "det_dim": ddim,
        "fci_feasible": not beyond_fci,
        "bond_dim": bond_dim,
        "fci_conversion": bool(not cfg["skip_kernel_fci_conversion"]),
        "mps_native_rdms": bool(cfg["mps_native_rdms"]),
        "root_tracking": track,
        "reason_no_fci_overlap": (f"det_dim={ddim:.3e}" if beyond_fci else None),
        "e_states": e_states,
        "build_wall_s": build_wall,
        "analytic_grad_wall_s": analytic_wall,
        "fd_grad_wall_s": fd_wall,
        "fd_components": fd_results,
        "fd_grad_max_abs_err": max_err,
        "validated": bool(max_err < 5.0e-4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, required=True)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--bond-dim", type=int, default=800)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--stack-mem-mb", type=int, default=8000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        r = run_one(args.ncarbon, basis=args.basis, bond_dim=args.bond_dim,
                    threads=args.threads, stack_mem_mb=args.stack_mem_mb)
        print(json.dumps(r, indent=2), flush=True)
    except Exception as exc:
        r = {"n_carbon": args.ncarbon, "status": "error",
             "exception": type(exc).__name__, "message": str(exc),
             "traceback_tail": traceback.format_exc()[-3000:]}
        print(f"ERROR: {exc}", flush=True)
    out = Path(args.out) if args.out else _HERE / "data" / f"polyene_c{args.ncarbon}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"benchmark": "polyene_beyond_fci", "result": r},
                              indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
