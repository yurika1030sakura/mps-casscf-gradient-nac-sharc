#!/usr/bin/env python3
"""Finite-difference DMRG gap-gradient diagnostic for SHARC/PySCF inputs.

This is a method-development/validation tool. It computes
grad(E_upper - E_lower) from DMRG overlay energies by finite differences.
That vector is the DMRG-level analogue of the MECI branching-space g vector.

It does not compute a DMRG derivative coupling vector and therefore is not a
complete DMRG MECI optimizer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from SHARC_PYSCF_ext import build_mol, gen_solver, readqmin, setup_workdir

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
HARTREE_TO_EV = 27.211386245988


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute finite-difference DMRG gap gradients for a SHARC QM.in."
    )
    parser.add_argument(
        "qmin",
        help="Path to QM.in. PYSCF.template and PYSCF.resources are read from the same directory.",
    )
    parser.add_argument(
        "--state-pair",
        nargs=2,
        type=int,
        default=(0, 1),
        metavar=("LOWER", "UPPER"),
        help="Zero-based DMRG root indices for E_upper - E_lower.",
    )
    parser.add_argument("--fd-step", type=float, default=None, help="Finite-difference step in Bohr.")
    parser.add_argument("--startm", type=int, default=None, help="Override DMRG startM.")
    parser.add_argument("--maxm", type=int, default=None, help="Override DMRG maxM.")
    parser.add_argument("--nsteps", type=int, default=None, help="Override DMRG nsteps.")
    parser.add_argument("--sweep-tol", type=float, default=None, help="Override DMRG sweep tolerance.")
    parser.add_argument(
        "--components",
        default="all",
        help=(
            "Comma-separated 1-based coordinate list such as '1:z,2:x,2:y'. "
            "Use 'all' for all 3N components."
        ),
    )
    parser.add_argument(
        "--output",
        default="dmrg_gap_gradient.json",
        help="JSON output path, relative to the QM.in directory unless absolute.",
    )
    return parser.parse_args()


def parse_components(spec: str, natom: int) -> list[tuple[int, int]]:
    if spec.strip().lower() == "all":
        return [(iatom, idir) for iatom in range(natom) for idir in range(3)]

    out: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        atom_text, axis_text = token.split(":", 1)
        iatom = int(atom_text) - 1
        axis = axis_text.strip().lower()
        if iatom < 0 or iatom >= natom:
            raise ValueError(f"Atom index out of range in component {token!r}")
        if axis not in AXIS_INDEX:
            raise ValueError(f"Unknown axis in component {token!r}; use x, y, or z")
        out.append((iatom, AXIS_INDEX[axis]))
    if not out:
        raise ValueError("No components selected")
    return out


def finite_or_none(array: np.ndarray):
    out = []
    for value in array.reshape(-1):
        out.append(None if np.isnan(value) else float(value))
    return np.array(out, dtype=object).reshape(array.shape).tolist()


def main() -> int:
    args = parse_args()
    qmin_path = Path(args.qmin).resolve()
    workdir = qmin_path.parent
    os.chdir(workdir)
    os.environ.setdefault("SHARC_DMRG_EXPERIMENTAL", "1")

    qmin = readqmin(qmin_path.name)
    qmin["template"]["dmrg-grad-mode"] = "finite-diff"
    if args.fd_step is not None:
        qmin["template"]["dmrg-fd-step"] = float(args.fd_step)
    if args.startm is not None:
        qmin["template"]["dmrg-startm"] = int(args.startm)
    if args.maxm is not None:
        qmin["template"]["dmrg-maxm"] = int(args.maxm)
    if args.nsteps is not None:
        qmin["template"]["dmrg-nsteps"] = int(args.nsteps)
    if args.sweep_tol is not None:
        qmin["template"]["dmrg-sweep-tol"] = float(args.sweep_tol)

    setup_workdir(qmin)
    mol = build_mol(qmin)
    solver = gen_solver(mol, qmin)
    grad_driver = solver.nuc_grad_method()

    lower, upper = args.state_pair
    nroots = int(qmin["template"]["roots"][0])
    if lower < 0 or upper < 0 or lower >= nroots or upper >= nroots or lower == upper:
        raise ValueError(f"Invalid state pair {args.state_pair}; nroots={nroots}")

    coords = np.array(solver.mol.atom_coords(unit="Bohr"), dtype=float)
    components = parse_components(args.components, coords.shape[0])
    step = float(qmin["template"].get("dmrg-fd-step", 1.0e-3))
    state_grads = np.full((nroots,) + coords.shape, np.nan)
    gap_grad = np.full(coords.shape, np.nan)

    for iatom, idir in components:
        plus = coords.copy()
        minus = coords.copy()
        plus[iatom, idir] += step
        minus[iatom, idir] -= step
        e_plus = grad_driver._dmrg_energies_at(plus)
        e_minus = grad_driver._dmrg_energies_at(minus)
        deriv = (e_plus - e_minus) / (2.0 * step)
        state_grads[:, iatom, idir] = deriv
        gap_grad[iatom, idir] = deriv[upper] - deriv[lower]

    finite_gap = gap_grad[np.isfinite(gap_grad)]
    result = {
        "qmin": str(qmin_path),
        "workdir": str(workdir),
        "state_pair_zero_based": [lower, upper],
        "fd_step_bohr": step,
        "nroots": nroots,
        "components": [
            {"atom_1based": iatom + 1, "axis": "xyz"[idir]} for iatom, idir in components
        ],
        "reference_dmrg_energies_hartree": [float(x) for x in np.asarray(solver.e_states)],
        "reference_gap_hartree": float(np.asarray(solver.e_states)[upper] - np.asarray(solver.e_states)[lower]),
        "reference_gap_ev": float((np.asarray(solver.e_states)[upper] - np.asarray(solver.e_states)[lower]) * HARTREE_TO_EV),
        "state_gradients_hartree_per_bohr": finite_or_none(state_grads),
        "gap_gradient_hartree_per_bohr": finite_or_none(gap_grad),
        "computed_gap_gradient_norm": float(np.linalg.norm(finite_gap)) if finite_gap.size else None,
        "all_components_computed": len(components) == coords.shape[0] * 3,
    }
    if result["all_components_computed"]:
        result["gap_gradient_translation_residual"] = [
            float(x) for x in np.nansum(gap_grad, axis=0)
        ]

    output = Path(args.output)
    if not output.is_absolute():
        output = workdir / output
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output}")
    print(
        "DMRG gap gradient norm over computed components:",
        result["computed_gap_gradient_norm"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

