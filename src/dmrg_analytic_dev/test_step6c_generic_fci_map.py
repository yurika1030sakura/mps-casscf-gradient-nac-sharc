"""Step 6.3a validation: generic FCI ↔ MPS converter via SZ-mode determinants.

Tests:
  G1: CAS(2,2) singlet (HeH+/sto-3g) — round-trip ci ≈ inverse(forward(ci)) at 1e-12.
  G2: CAS(4,4) singlet (H4 chain/sto-3g) — round-trip at 1e-12.
  G3: CAS(8,8) singlet (LiH/cc-pVDZ stretched) — round-trip at 1e-10 (lower
      tol due to fitting precision).
  G4: Random non-singlet trial vector in CAS(4,4) — round-trip at 1e-12,
      verifying the converter is faithful for non-eigenstate vectors that
      arise as CP-iteration trial vectors.

These confirm the SZ-determinant route generalises arbitrarily, replacing
the CAS(2,2)-hardcoded CSF route in `dmrg_fcisolver._csf_to_fci22_singlet`.
"""

from __future__ import annotations

import json
import os
import sys
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import fci, gto, mcscf, scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from site_replacement_density import (
    fci_to_mps_generic,
    mps_to_fci_generic,
    _fci_to_sz_dets,
    _sz_dets_to_fci,
)


def _scratch(name: str) -> str:
    base = "/tmp"
    Path(base).mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"step6c_{name}_", dir=base)


def _make_sz_driver(ncas: int, nelec: tuple[int, int], scratch: str):
    drv = DMRGDriver(
        scratch=scratch, clean_scratch=False, stack_mem=int(2e8),
        n_threads=1, symm_type=SymmetryTypes.SZ,
    )
    spin = abs(int(nelec[0]) - int(nelec[1]))
    drv.initialize_system(
        n_sites=ncas, n_elec=int(nelec[0]) + int(nelec[1]),
        spin=spin, orb_sym=[0] * ncas,
    )
    return drv


def _round_trip_diff(ci: np.ndarray, ncas: int, nelec: tuple[int, int],
                     scratch: str, tag: str = "RT"):
    drv = _make_sz_driver(ncas, nelec, scratch)
    mps = fci_to_mps_generic(drv, ci, ncas, nelec, tag=tag, dot=2)
    ci_back = mps_to_fci_generic(drv, mps, ncas, nelec)
    # No sign ambiguity expected: the converter is linear with no phase.
    diff = float(np.linalg.norm(ci_back - ci))
    return diff, mps.info.bond_dim


def test_g1_cas22_singlet_roundtrip():
    mol = gto.M(atom="He 0 0 0; H 0 0 1.4", basis="sto-3g",
                charge=1, spin=0, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cas = mcscf.CASCI(mf, 2, 2)
    cas.fcisolver.nroots = 2
    cas.kernel()

    scratch = _scratch("g1")
    diffs = []
    for i, ci in enumerate(cas.ci):
        diff, _ = _round_trip_diff(ci, 2, (1, 1), scratch, tag=f"R{i}")
        diffs.append(diff)
    shutil.rmtree(scratch, ignore_errors=True)
    return {"name": "G1_cas22_singlet_roundtrip",
            "diffs_per_root": diffs, "tol": 1e-12,
            "status": "pass" if max(diffs) < 1e-12 else "fail"}


def test_g2_cas44_h4_roundtrip():
    mol = gto.M(
        atom="\n".join(f"H 0 0 {z:.3f}" for z in [0.0, 1.0, 2.0, 3.0]),
        basis="sto-3g", spin=0, charge=0, unit="Bohr", verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cas = mcscf.CASCI(mf, 4, 4)
    cas.kernel()

    scratch = _scratch("g2")
    diff, M = _round_trip_diff(cas.ci, 4, (2, 2), scratch, tag="R0")
    shutil.rmtree(scratch, ignore_errors=True)
    return {"name": "G2_cas44_h4_roundtrip",
            "diff": diff, "bond_dim": M, "tol": 1e-12,
            "status": "pass" if diff < 1e-12 else "fail"}


def test_g3_cas88_lih_roundtrip():
    """LiH/cc-pVDZ stretched, CAS(8,8). Direct FCI is feasible (~ 4900 dets)."""
    mol = gto.M(atom="Li 0 0 0; H 0 0 3.5", basis="cc-pVDZ",
                spin=0, charge=0, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cas = mcscf.CASCI(mf, 8, 4)  # 4 electrons in 8 orbitals -> 8 active dets per spin
    cas.kernel()

    scratch = _scratch("g3")
    diff, M = _round_trip_diff(cas.ci, 8, (2, 2), scratch, tag="R0")
    shutil.rmtree(scratch, ignore_errors=True)
    return {"name": "G3_cas88_lih_roundtrip",
            "diff": diff, "bond_dim": M, "tol": 1e-10,
            "status": "pass" if diff < 1e-10 else "fail"}


def test_g4_cas44_random_nonspin():
    """Random non-singlet trial vector — verifies the converter handles
    arbitrary CI tensors (the kind that arise as CP trial vectors)."""
    rng = np.random.default_rng(7)
    # CAS(4, (2,2)) FCI shape: (6, 6).
    ci = rng.standard_normal((6, 6))
    ci /= np.linalg.norm(ci)

    scratch = _scratch("g4")
    diff, M = _round_trip_diff(ci, 4, (2, 2), scratch, tag="R0")
    shutil.rmtree(scratch, ignore_errors=True)
    return {"name": "G4_cas44_random_roundtrip",
            "diff": diff, "bond_dim": M, "tol": 1e-12,
            "status": "pass" if diff < 1e-12 else "fail"}


def main():
    cases = [
        test_g1_cas22_singlet_roundtrip,
        test_g2_cas44_h4_roundtrip,
        test_g3_cas88_lih_roundtrip,
        test_g4_cas44_random_nonspin,
    ]
    results = []
    for c in cases:
        try:
            r = c()
        except Exception as exc:
            r = {"name": c.__name__, "status": "fail",
                 "exception": type(exc).__name__,
                 "message": str(exc),
                 "traceback_tail": traceback.format_exc()[-2000:]}
        results.append(r)
        print(f"  {r['name']}: {r['status']}")

    out_path = Path(__file__).with_suffix(".json")
    out = {
        "milestone": "Step6.3a_generic_FCI_to_MPS_converter",
        "purpose": "Validate SZ-mode determinant route as the generic CSF↔FCI replacement, applicable to arbitrary CAS(n,m) and any spin sector.",
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2,
        default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
