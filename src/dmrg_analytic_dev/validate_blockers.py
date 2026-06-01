#!/usr/bin/env python3
"""Archive validation for the original raw-MPS DMRG gradient/NAC blockers.

This script is intentionally historical. It demonstrates why directly handing
raw block2 MPS objects to PySCF's stock response machinery was blocked. The
current production paths are `MPSAsFCISolver` and `CPDMRGCASSCFResponseMPS`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, scf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmrg_sharc_bridge import DriverMultiRootDMRGCI, HybridDMRGSharcSolver


def scratch_dir(name: str) -> Path:
    root = Path(os.getenv("DMRG_VALIDATE_SCRATCH", "/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{name}_", dir=root))


def h2_mol():
    return gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        spin=0,
        charge=0,
        symmetry=False,
        verbose=0,
    )


def run_dmrg_sa_matches_fci() -> dict[str, object]:
    mol = h2_mol()
    mf = scf.RHF(mol).run(conv_tol=1.0e-10)
    cas = mcscf.CASCI(mf, 2, 2)
    h1e, ecore = cas.get_h1cas()
    g2e = cas.get_h2cas()
    fci_e, _ = fci.direct_spin0.FCI().kernel(h1e, g2e, 2, (1, 1), ecore=ecore, nroots=2)

    mc = mcscf.CASSCF(mf, 2, 2)
    mc.max_cycle_macro = 3
    mc.chkfile = None
    mc.chk_ci = False
    mc.dump_chk = lambda *args, **kwargs: None
    solver = DriverMultiRootDMRGCI(mf)
    solver.dmrg_args.update(
        {
            "startM": 20,
            "maxM": 50,
            "sweep_tol": 1.0e-7,
            "nsteps": 8,
            "memory": int(2e8),
            "scratch_root": str(scratch_dir("dmrg_sa")),
            "dav_max_iter": 200,
        }
    )
    mc.fcisolver = solver
    mc.state_average_([0.5, 0.5])
    mc.kernel()
    err = float(np.max(np.abs(np.asarray(mc.fcisolver.e_states) - np.asarray(fci_e))))
    return {
        "name": "dmrg_sa_matches_fci",
        "status": "pass" if err < 1.0e-10 else "fail",
        "energy_error_hartree": err,
        "dmrg_energies_hartree": np.asarray(mc.fcisolver.e_states).tolist(),
        "fci_energies_hartree": np.asarray(fci_e).tolist(),
    }


def run_pyscf_gradient_expected_failure() -> dict[str, object]:
    mol = h2_mol()
    mf = scf.RHF(mol).run(conv_tol=1.0e-10)
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.max_cycle_macro = 3
    mc.chkfile = None
    mc.chk_ci = False
    mc.dump_chk = lambda *args, **kwargs: None
    solver = DriverMultiRootDMRGCI(mf)
    solver.dmrg_args.update(
        {
            "startM": 20,
            "maxM": 50,
            "sweep_tol": 1.0e-7,
            "nsteps": 8,
            "memory": int(2e8),
            "scratch_root": str(scratch_dir("dmrg_grad_blocker")),
            "dav_max_iter": 200,
        }
    )
    mc.fcisolver = solver
    mc.state_average_([0.5, 0.5])
    mc.kernel()
    try:
        mc.nuc_grad_method().kernel(state=0)
    except Exception as exc:
        tb = traceback.format_exc()
        expected = "MPS" in tb and "ravel" in tb
        return {
            "name": "pyscf_gradient_expected_failure",
            "status": "expected_fail" if expected else "fail",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": tb[-1200:],
        }
    return {
        "name": "pyscf_gradient_expected_failure",
        "status": "fail",
        "message": "PySCF analytic gradient unexpectedly succeeded with raw MPS states.",
    }


def run_fd_gap_gradient() -> dict[str, object]:
    mol = gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="sto-3g",
        spin=0,
        charge=0,
        symmetry=False,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1.0e-10)
    base = mcscf.CASSCF(mf, 2, 2).state_average([0.5, 0.5])
    base.max_cycle_macro = 3
    base.chkfile = None
    base.chk_ci = False
    base.dump_chk = lambda *args, **kwargs: None
    base.kernel()
    template = {
        "roots": [2],
        "dmrg-ncas": 2,
        "dmrg-nelecas": 2,
        "dmrg-startm": 20,
        "dmrg-maxm": 50,
        "dmrg-sweep-tol": 1.0e-7,
        "dmrg-memory-mb": 500,
        "dmrg-nsteps": 8,
        "dmrg-grad-mode": "finite-diff",
        "dmrg-fd-step": 2.0e-3,
        "grad-max-cycle": 20,
        "verbose": 0,
        "fix-spin-shift": 0.2,
        "conv-tol": 1.0e-8,
        "conv-tol-grad": 1.0e-5,
        "max-stepsize": 0.02,
        "max-cycle-macro": 10,
        "max-cycle-micro": 4,
        "ah-level-shift": 1.0e-8,
        "ah-conv-tol": 1.0e-12,
        "ah-max-cycle": 20,
        "ah-lindep": 1.0e-14,
        "ah-start-tol": 2.5,
        "ah-start-cycle": 3,
    }
    hybrid = HybridDMRGSharcSolver(
        base,
        {"template": template, "scratchdir": str(scratch_dir("dmrg_gap_grad")), "memory": 1000},
    )
    grad_obj = hybrid.nuc_grad_method()
    grads = grad_obj.kernel_all()
    gap_grad = grads[1] - grads[0]
    residual = float(np.linalg.norm(gap_grad.sum(axis=0)))
    return {
        "name": "fd_gap_gradient",
        "status": "pass" if residual < 1.0e-8 else "fail",
        "all_gradient_shape": list(grads.shape),
        "gap_gradient_shape": list(gap_grad.shape),
        "translation_residual": residual,
    }


def main() -> int:
    cases = [
        run_dmrg_sa_matches_fci,
        run_pyscf_gradient_expected_failure,
        run_fd_gap_gradient,
    ]
    results = []
    for case in cases:
        try:
            results.append(case())
        except Exception as exc:
            results.append(
                {
                    "name": case.__name__,
                    "status": "fail",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback_tail": traceback.format_exc()[-1200:],
                }
            )

    out = {
        "purpose": "Historical raw-MPS analytic DMRG gradient/NAC blocker validation",
        "archival_note": (
            "Superseded by MPSAsFCISolver and CPDMRGCASSCFResponseMPS. "
            "The raw-MPS PySCF gradient failure is retained as an audit trail, "
            "not as a current production blocker."
        ),
        "superseded_by": [
            "test_dmrg_fcisolver.py",
            "test_mpsasfcisolver_step6_production.py",
            "test_step6c_mps_response_class.py",
            "step7b_damn_smoke/run_damn_smoke.py",
        ],
        "results": results,
    }
    output_path = Path(__file__).with_suffix(".json")
    output_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"Wrote {output_path}")

    accepted = {"pass", "expected_fail"}
    return 0 if all(item["status"] in accepted for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
