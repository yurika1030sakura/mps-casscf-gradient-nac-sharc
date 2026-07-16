"""FCI-free MPS-Krylov response benchmark on a hydrogen chain.

This benchmark is deliberately independent of an FCI reference.  It starts
from spin-adapted block2 MPS roots, constructs the MPS-only response backend,
and evaluates one gradient and one NAC through the SHARC-facing MPS-Krylov
assembly.  It records wall-time diagnostics for production-style large active
spaces where dense active-space CI vectors are not part of the runtime path.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import ao2mo, fci, gto, mcscf, scf

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
SHARC = REPO / "sharc_interface"
for path in (SRC, SRC / "dmrg_analytic_dev", SHARC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analytic_cp_sharc import (  # noqa: E402
    _gradient_one_state_mps_krylov,
    _nac_one_pair_mps_krylov,
)
from cp_dmrg_response_mps_krylov import CPDMRGCASSCFResponseMPSKrylov  # noqa: E402


def build_hchain(natom: int, spacing_bohr: float):
    atom = ";".join(f"H 0 0 {i * spacing_bohr:.10f}" for i in range(natom))
    mol = gto.M(atom=atom, basis="sto-3g", unit="Bohr", spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1.0e-12, verbose=0)
    mc = mcscf.CASSCF(mf, natom, natom)
    mc.fcisolver = fci.direct_spin1.FCI()
    mc.fcisolver.nroots = 2
    mc = mc.state_average([0.5, 0.5])
    mc.mo_coeff = mf.mo_coeff
    mc.ci = [np.zeros((1, 1)), np.zeros((1, 1))]
    return mol, mf, mc


def run_dmrg(mol, mf, ncas, nelec, bond_dim, sweeps, tol, scratch):
    mo = mf.mo_coeff
    h1 = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.kernel(mol, mo, compact=False).reshape((ncas,) * 4)
    driver = DMRGDriver(
        scratch=str(scratch),
        clean_scratch=True,
        stack_mem=int(2e8),
        n_threads=1,
        symm_type=SymmetryTypes.SU2,
    )
    driver.initialize_system(
        n_sites=ncas, n_elec=nelec, spin=0, orb_sym=[0] * ncas,
    )
    mpo = driver.get_qc_mpo(h1, eri, ecore=mol.energy_nuc(), iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=bond_dim, nroots=2)
    noises = [1.0e-4] * min(4, sweeps) + [0.0] * max(0, sweeps - 4)
    t0 = time.perf_counter()
    energies = driver.dmrg(
        mpo, ket,
        n_sweeps=sweeps,
        bond_dims=[bond_dim] * sweeps,
        noises=noises,
        tol=tol,
        iprint=0,
    )
    dmrg_time = time.perf_counter() - t0
    kets = [driver.split_mps(ket, i, f"KET-{i}") for i in range(2)]
    return driver, mpo, kets, np.asarray(energies, dtype=float), dmrg_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--natom", type=int, default=4)
    parser.add_argument("--spacing-bohr", type=float, default=1.4)
    parser.add_argument("--bond-dim", type=int, default=50)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-8)
    parser.add_argument("--response-tol", type=float, default=1.0e-6)
    parser.add_argument("--response-max-iter", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    mol, mf, mc = build_hchain(args.natom, args.spacing_bohr)
    scratch = Path(tempfile.mkdtemp(prefix="mps_only_hchain_"))
    driver, mpo, kets, energies, dmrg_time = run_dmrg(
        mol, mf, args.natom, args.natom,
        args.bond_dim, args.sweeps, args.sweep_tol, scratch,
    )
    mc.e_tot = energies

    cp = CPDMRGCASSCFResponseMPSKrylov(
        mc, driver, mpo,
        mps_states=kets,
        weights=np.array([0.5, 0.5]),
        m_compress=args.bond_dim,
        mps_fit_sweeps=max(4, min(args.sweeps, 10)),
        mps_fit_tol=args.response_tol,
        mps_only=True,
    )

    t0 = time.perf_counter()
    grad0 = _gradient_one_state_mps_krylov(
        mc, cp, 0, tol=args.response_tol, max_iter=args.response_max_iter,
    )
    grad_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    nac01 = _nac_one_pair_mps_krylov(
        mc, cp, (0, 1), tol=args.response_tol,
        max_iter=args.response_max_iter,
    )
    nac_time = time.perf_counter() - t0

    result = {
        "benchmark": "mps_only_hchain_response",
        "system": f"H{args.natom} chain",
        "basis": "sto-3g",
        "active_space": f"CAS({args.natom},{args.natom})",
        "uses_fci_reference": False,
        "uses_dense_ci_response_vector": False,
        "bond_dim": args.bond_dim,
        "sweeps": args.sweeps,
        "energies_hartree": energies.tolist(),
        "timings_s": {
            "dmrg": dmrg_time,
            "gradient_state0": grad_time,
            "nac_01": nac_time,
        },
        "gradient_state0_norm": float(np.linalg.norm(grad0)),
        "nac_01_norm": float(np.linalg.norm(nac01)),
    }
    out_path = Path(args.output) if args.output else Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
