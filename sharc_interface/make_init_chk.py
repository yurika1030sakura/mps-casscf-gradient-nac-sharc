#!/usr/bin/env python3
"""Build the initial PySCF checkpoint consumed by SHARC_PYSCF.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from pyscf import gto, mcscf, scf

DEFAULTS = {
    "basis": "cc-pVDZ",
    "charge": 0,
    "spin": 0,
    "ncas": 10,
    "nelecas": 12,
    "roots": 3,
}


def read_xyz(path: Path) -> str:
    lines = path.read_text().splitlines()
    natom = int(lines[0].strip())
    atoms = []
    for line in lines[2 : 2 + natom]:
        sym, x, y, z = line.split()[:4]
        atoms.append(f"{sym} {x} {y} {z}")
    return "; ".join(atoms)


def load_template_settings(path: Path) -> dict[str, int | str]:
    settings: dict[str, int | str] = {}
    for raw_line in path.read_text().splitlines():
        line = re.sub(r"#.*$", "", raw_line).strip()
        if not line:
            continue
        fields = line.split()
        key = fields[0].lower()
        values = fields[1:]
        if key == "basis" and values:
            settings["basis"] = values[0]
        elif key == "charge" and values:
            settings["charge"] = int(values[0])
        elif key == "ncas" and values:
            settings["ncas"] = int(values[0])
        elif key == "nelecas" and values:
            settings["nelecas"] = int(values[0])
        elif key == "roots" and values:
            roots = [int(value) for value in values]
            settings["roots"] = next((root for root in roots if root > 0), roots[0])
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz", help="Input XYZ geometry in Angstrom.")
    parser.add_argument("--chkfile", default="QM/pyscf.init.chk")
    parser.add_argument(
        "--template",
        default=None,
        help="Optional PYSCF.template file used to fill in basis/charge/ncas/nelecas/roots.",
    )
    parser.add_argument("--basis", default=None)
    parser.add_argument("--charge", type=int, default=None)
    parser.add_argument("--spin", type=int, default=DEFAULTS["spin"])
    parser.add_argument("--ncas", type=int, default=None)
    parser.add_argument("--nelecas", type=int, default=None)
    parser.add_argument("--roots", type=int, default=None)
    parser.add_argument("--fix-spin-shift", type=float, default=0.2)
    parser.add_argument("--max-cycle-macro", type=int, default=50)
    parser.add_argument("--output", default="QM/pyscf_init.log")
    args = parser.parse_args()

    template_settings = {}
    if args.template is not None:
        template_settings = load_template_settings(Path(args.template))

    basis = args.basis if args.basis is not None else str(template_settings.get("basis", DEFAULTS["basis"]))
    charge = args.charge if args.charge is not None else int(template_settings.get("charge", DEFAULTS["charge"]))
    ncas = args.ncas if args.ncas is not None else int(template_settings.get("ncas", DEFAULTS["ncas"]))
    nelecas = args.nelecas if args.nelecas is not None else int(template_settings.get("nelecas", DEFAULTS["nelecas"]))
    roots = args.roots if args.roots is not None else int(template_settings.get("roots", DEFAULTS["roots"]))

    chkfile = Path(args.chkfile)
    chkfile.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    mol = gto.M(
        atom=read_xyz(Path(args.xyz)),
        basis=basis,
        charge=charge,
        spin=args.spin,
        symmetry=False,
        unit="Angstrom",
        output=args.output,
        verbose=4,
    )

    if args.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.conv_tol = 1.0e-9
    mf.kernel()

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    try:
        from mrh.my_pyscf.fci import csf_solver

        mc.fcisolver = csf_solver(mol, smult=args.spin + 1)
    except ImportError:
        if args.spin == 0:
            mc.fix_spin_(ss=0, shift=args.fix_spin_shift)
    if roots > 1:
        weights = [1.0 / roots] * roots
        mc.state_average_(weights)
    mc.max_cycle_macro = args.max_cycle_macro
    mc.chkfile = str(chkfile)
    mc.chk_ci = True
    mc.kernel()

    print(f"Wrote initial checkpoint to {chkfile}")
    if getattr(mc, "e_states", None) is not None:
        print("State energies:")
        for idx, energy in enumerate(mc.e_states, start=1):
            print(f"  root {idx}: {energy:.10f} Ha")
    else:
        print(f"CASSCF energy: {mc.e_tot:.10f} Ha")


if __name__ == "__main__":
    main()
