#!/usr/bin/env python3
"""Anthracene pi-space CAS(14,14) DMRG convergence benchmark.

It uses RHF + AVAS to define the 14 pi orbitals of anthracene, evaluates a
fixed-orbital CASCI/FCI reference when requested, runs SU2 DMRG-CASCI for the
lowest singlet roots at a sequence of bond dimensions, and reports convergence
against the FCI energies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.fci import cistring
from pyscf.mcscf import avas


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def build_anthracene_geometry() -> str:
    """Return a planar idealized anthracene geometry in Angstrom."""
    r_cc = 1.397
    r_ch = 1.09
    centers = [(0.0, 0.0), (np.sqrt(3.0) * r_cc, 0.0),
               (2.0 * np.sqrt(3.0) * r_cc, 0.0)]
    carbons = []
    for cx, cy in centers:
        for k in range(6):
            theta = np.deg2rad(30.0 + 60.0 * k)
            pos = np.array([cx + r_cc * np.cos(theta),
                            cy + r_cc * np.sin(theta), 0.0])
            if not any(np.linalg.norm(pos - old) < 1e-5 for old in carbons):
                carbons.append(pos)
    carbons = np.asarray(carbons)
    carbons[:, 0] -= np.mean(carbons[:, 0])
    carbons[:, 1] -= np.mean(carbons[:, 1])

    neighbors = {i: [] for i in range(len(carbons))}
    for i, pi in enumerate(carbons):
        for j, pj in enumerate(carbons[:i]):
            if abs(np.linalg.norm(pi - pj) - r_cc) < 1e-3:
                neighbors[i].append(j)
                neighbors[j].append(i)

    atoms = [("C", xyz) for xyz in carbons]
    for i, carbon in enumerate(carbons):
        if len(neighbors[i]) != 2:
            continue
        inward = np.zeros(3)
        for j in neighbors[i]:
            vec = carbons[j] - carbon
            inward += vec / np.linalg.norm(vec)
        hydrogen = carbon - r_ch * inward / np.linalg.norm(inward)
        atoms.append(("H", hydrogen))

    return "\n".join(
        f"{sym:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}"
        for sym, xyz in atoms
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--m-list", default="64,128,256,512,1024")
    parser.add_argument("--nroots", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=80)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-8)
    parser.add_argument("--skip-fci", action="store_true")
    parser.add_argument("--fci-conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--fci-max-cycle", type=int, default=120)
    parser.add_argument("--spin-shift", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=120000)
    parser.add_argument("--stack-mem", type=float, default=2.0e9)
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--out", default=str(DATA_DIR / "anthracene_pi14.json"))
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def run_dmrg_casci(h1, eri, ecore, ncas, nelec, *, bond_dim, nroots,
                   sweeps, sweep_tol, threads, stack_mem, scratch_root):
    scratch = tempfile.mkdtemp(
        prefix=f"anth_pi14_M{bond_dim}_",
        dir=scratch_root,
    )
    t0 = time.time()
    try:
        driver = DMRGDriver(
            scratch=scratch,
            clean_scratch=False,
            stack_mem=int(stack_mem),
            n_threads=int(threads),
            symm_type=SymmetryTypes.SU2,
        )
        driver.initialize_system(
            n_sites=int(ncas),
            n_elec=int(nelec),
            spin=0,
            orb_sym=[0] * int(ncas),
        )
        mpo = driver.get_qc_mpo(
            np.asarray(h1),
            np.asarray(eri),
            ecore=float(ecore),
            iprint=0,
        )
        ket = driver.get_random_mps(
            tag=f"K_M{bond_dim}",
            bond_dim=int(bond_dim),
            nroots=int(nroots),
        )
        nsweep = max(int(sweeps), 30)
        noises = ([1e-3] * 8 + [1e-4] * 8 + [1e-5] * 8
                  + [1e-6] * 4 + [0.0] * max(0, nsweep - 28))
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=nsweep,
            bond_dims=[int(bond_dim)] * nsweep,
            noises=noises[:nsweep],
            tol=float(sweep_tol),
            iprint=0,
        )
        if hasattr(energies, "__iter__"):
            e_list = [float(x) for x in energies]
        else:
            e_list = [float(energies)]
        return {"M": int(bond_dim), "energies_hartree": e_list,
                "runtime_s": time.time() - t0}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_fci_reference(h1, eri, ecore, ncas, nelec, *, nroots, conv_tol,
                      max_cycle, spin_shift):
    """Run direct fixed-orbital CASCI/FCI for CAS(14,14)."""
    neleca = int(nelec) // 2
    nelecb = int(nelec) - neleca
    solver = fci.direct_spin1.FCI()
    solver.nroots = int(nroots)
    solver.conv_tol = float(conv_tol)
    solver.max_cycle = int(max_cycle)
    solver = fci.addons.fix_spin(solver, shift=float(spin_shift), ss=0)
    solver.nroots = int(nroots)
    t0 = time.time()
    energies, _ = solver.kernel(
        np.asarray(h1),
        np.asarray(eri),
        int(ncas),
        (neleca, nelecb),
        ecore=float(ecore),
    )
    if hasattr(energies, "__iter__"):
        e_list = [float(x) for x in energies]
    else:
        e_list = [float(energies)]
    return {"energies_hartree": e_list, "runtime_s": time.time() - t0}


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = args.scratch_root
    if scratch_root is not None:
        Path(scratch_root).mkdir(parents=True, exist_ok=True)

    mol = gto.M(
        atom=build_anthracene_geometry(),
        basis=args.basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        symmetry=False,
        verbose=0,
        max_memory=args.memory_mb,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.kernel()
    ncas, nelecas, mo_coeff = avas.avas(
        mf,
        ["C 2pz"],
        threshold=0.20,
        canonicalize=True,
    )
    ncas = int(ncas)
    nelecas = int(nelecas)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    h1, ecore = mc.get_h1eff(mo_coeff)
    eri = ao2mo.restore(1, np.asarray(mc.get_h2eff(mo_coeff)), ncas)

    output = {
        "system": "anthracene",
        "basis": args.basis,
        "active_space": [nelecas, ncas],
        "avas_labels": ["C 2pz"],
        "nroots": int(args.nroots),
        "m_list": [int(x) for x in args.m_list.split(",") if x.strip()],
        "sweeps": int(args.sweeps),
        "sweep_tol": float(args.sweep_tol),
        "threads": int(args.threads),
        "rhf_energy_hartree": float(mf.e_tot),
        "preview_only": bool(args.preview_only),
        "fci_reference": None,
        "fci_dimension_det": int(
            cistring.num_strings(ncas, nelecas // 2)
            * cistring.num_strings(ncas, nelecas - nelecas // 2)
        ),
        "results": [],
    }
    out_path.write_text(json.dumps(output, indent=2))

    print("Anthracene pi-space benchmark")
    print(f"basis={args.basis} active=CAS({nelecas},{ncas}) RHF={mf.e_tot:.12f}")
    print(f"output={out_path}")

    if args.preview_only:
        return

    if not args.skip_fci:
        print("[anthracene] FCI reference start", flush=True)
        output["fci_reference"] = run_fci_reference(
            h1,
            eri,
            ecore,
            ncas,
            nelecas,
            nroots=args.nroots,
            conv_tol=args.fci_conv_tol,
            max_cycle=args.fci_max_cycle,
            spin_shift=args.spin_shift,
        )
        out_path.write_text(json.dumps(output, indent=2))
        e_msg = " ".join(
            f"E{i}={e:.10f}"
            for i, e in enumerate(output["fci_reference"]["energies_hartree"])
        )
        print(
            f"[anthracene] FCI {e_msg} "
            f"runtime={output['fci_reference']['runtime_s']:.1f}s",
            flush=True,
        )

    for bond_dim in output["m_list"]:
        print(f"[anthracene] M={bond_dim} start", flush=True)
        result = run_dmrg_casci(
            h1,
            eri,
            ecore,
            ncas,
            nelecas,
            bond_dim=bond_dim,
            nroots=args.nroots,
            sweeps=args.sweeps,
            sweep_tol=args.sweep_tol,
            threads=args.threads,
            stack_mem=args.stack_mem,
            scratch_root=scratch_root,
        )
        output["results"].append(result)
        if output["fci_reference"] is not None:
            ref = output["fci_reference"]["energies_hartree"]
            result["delta_vs_fci_mEh"] = [
                1000.0 * (result["energies_hartree"][i] - ref[i])
                for i in range(min(len(result["energies_hartree"]), len(ref)))
            ]
        ref = output["results"][-1]["energies_hartree"]
        for row in output["results"]:
            row["delta_vs_largest_completed_M_mEh"] = [
                1000.0 * (row["energies_hartree"][i] - ref[i])
                for i in range(min(len(row["energies_hartree"]), len(ref)))
            ]
        out_path.write_text(json.dumps(output, indent=2))
        e_msg = " ".join(f"E{i}={e:.10f}" for i, e in enumerate(result["energies_hartree"]))
        if "delta_vs_fci_mEh" in result:
            d_msg = " ".join(
                f"dE{i}={d:.3e} mEh"
                for i, d in enumerate(result["delta_vs_fci_mEh"])
            )
        else:
            d_msg = ""
        print(
            f"[anthracene] M={bond_dim} {e_msg} {d_msg} "
            f"runtime={result['runtime_s']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
