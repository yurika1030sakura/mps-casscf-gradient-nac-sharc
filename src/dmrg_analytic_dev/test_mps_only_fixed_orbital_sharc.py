"""Smoke-test the fixed-orbital MPS-only SHARC facade.

The test constructs the same object shape that ``SHARC_PYSCF_ext.py`` uses
when ``method dmrg-casscf``, ``dmrg-fixed-orbitals true``, and
``dmrg-response-mode mps-krylov`` are requested.  The active-space roots are
block2 MPS objects; ``mc.ci`` contains only small placeholders and is not a
determinant-space root store.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from pyscf import gto

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for sharc_root in (ROOT.parent, ROOT.parents[1] / "sharc_interface"):
    if sharc_root.exists():
        sys.path.insert(0, str(sharc_root))

from SHARC_PYSCF_ext import gen_solver, get_dipole_elements  # noqa: E402
from analytic_cp_sharc import compute_grad_nac_analytic_cp  # noqa: E402


def _qmin():
    return {
        "method": 5,
        "states": [2],
        "scratchdir": tempfile.mkdtemp(prefix="fixed_mps_sharc_"),
        "memory": 2000,
        "ncpu": 1,
        "template": {
            "ncas": 2,
            "nelecas": 2,
            "roots": [2],
            "conv-tol": 1.0e-10,
            "dmrg-fixed-orbitals": True,
            "dmrg-response-mode": "mps-krylov",
            "dmrg-maxm": 40,
            "dmrg-nsteps": 8,
            "dmrg-sweep-tol": 1.0e-8,
            "dmrg-refine-split-roots": 0,
            "fix-spin-shift": 0.2,
        },
    }


def test_f1_fixed_orbital_mps_only_sharc_quantities():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        unit="Bohr",
        basis="sto-3g",
        verbose=0,
    )
    mc = gen_solver(mol, _qmin())
    dipole = get_dipole_elements(mc)
    out = compute_grad_nac_analytic_cp(
        mc,
        gradient_states=[0],
        nac_pairs=[(0, 1)],
        backend="mps-krylov",
        tol=1.0e-6,
        max_iter=20,
    )
    ci_sizes = [int(np.asarray(ci).size) for ci in mc.ci]
    ok = (
        bool(mc.converged)
        and hasattr(mc.fcisolver, "_kets")
        and all(size == 1 for size in ci_sizes)
        and np.asarray(mc.e_states).shape == (2,)
        and dipole.shape == (3, 2, 2)
        and np.isfinite(dipole).all()
        and np.isfinite(out["grad"][0]).all()
        and np.isfinite(out["nac"][(0, 1)]).all()
    )
    return {
        "name": "F1_fixed_orbital_mps_only_sharc_quantities",
        "ci_placeholder_sizes": ci_sizes,
        "energies_hartree": np.asarray(mc.e_states, dtype=float).tolist(),
        "dipole_norm": float(np.linalg.norm(dipole)),
        "grad0_norm": float(np.linalg.norm(out["grad"][0])),
        "nac01_norm": float(np.linalg.norm(out["nac"][(0, 1)])),
        "status": "pass" if ok else "fail",
    }


def main():
    cases = [test_f1_fixed_orbital_mps_only_sharc_quantities]
    results = []
    for case in cases:
        try:
            result = case()
        except Exception as exc:
            result = {
                "name": case.__name__,
                "status": "fail",
                "exception": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc()[-2000:],
            }
        results.append(result)
        print(f"  {result['name']}: {result['status']}")

    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps({
        "milestone": "MPS_only_fixed_orbital_SHARC_facade",
        "purpose": (
            "Verify that a fixed-orbital MPS-only SHARC-facing object can "
            "produce energies, dipoles, gradients, and NACs without dense "
            "active-space CI roots."
        ),
        "results": results,
    }, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
