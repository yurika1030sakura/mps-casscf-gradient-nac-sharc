"""
Validate that the FCI-free SU2-mode solver (skip_kernel_fci_conversion +
mps_native_rdms) correctly targets a doublet (2S=1) spin sector.

Companion to ``test_su2_mode_triplet.py``: that test covers the triplet
(2S=2) sector, this one covers the doublet (2S=1) sector, so the backend is
checked across the singlet/doublet/triplet manifold rather than singlet only.

Approach: linear H3 chain at d=2.0 Bohr (stretched, near-degenerate), an
odd-electron system whose ground state is a doublet. We target the doublet
sector via mol.spin=1 (Sz=1/2 representative: 2 alpha + 1 beta). FCI in that
sector gives the reference energies and density matrices; DMRG SU2 with
spin=1 in initialize_system, FCI conversion disabled and MPS-native RDMs,
must match.

This exercises:
  - SU2 mode targeting the correct doublet (2S=1) sector;
  - energy-sort selection in the skip_kernel_fci_conversion path picking the
    doublet roots without dense CI readout;
  - make_rdm12 on a doublet MPS without an FCI bridge.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from pyscf import gto, scf, mcscf, fci

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from dmrg_fcisolver import MPSAsFCISolver


def make_h3_doublet(d_bohr: float = 2.0):
    """Linear H3 chain, stretched, configured as a doublet (mol.spin=1).
    Sz=1/2 representative: 2 alpha + 1 beta electron."""
    return gto.M(atom=f"H 0 0 0; H 0 0 {d_bohr}; H 0 0 {2*d_bohr}",
                 basis="sto-3g", charge=0, spin=1, unit="Bohr",
                 symmetry=False, verbose=0)


def compare(name, ref, got, atol=1e-6):
    r = np.asarray(ref).ravel(); g = np.asarray(got).ravel()
    diff = float(np.max(np.abs(r - g)))
    status = "PASS" if diff < atol else "FAIL"
    print(f"  {name:18s}  max|diff|={diff:.3e}  [{status}]")
    return diff < atol


def main():
    mol = make_h3_doublet()
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    print(f"Mol: {mol.nelec} electrons (na, nb), spin=1 (doublet sector)")
    print()

    # ------------- FCI reference (CASCI in fixed orbitals) -------------
    print("=== FCI reference: H3 CAS(3,3) doublet (Sz=1/2, 2S=1) ===")
    mc_fci = mcscf.CASCI(mf, 3, 3)
    mc_fci.fcisolver.nroots = 2
    mc_fci.kernel()
    e_fci = list(map(float, mc_fci.e_tot))
    ci0 = mc_fci.ci[0]
    dm1_0_fci, dm2_0_fci = fci.direct_spin1.make_rdm12(ci0, 3, mol.nelec)
    print(f"  e_states = {e_fci}")
    print(f"  Sz from FCI: na={mol.nelec[0]}, nb={mol.nelec[1]}, "
          f"2Sz={mol.nelec[0]-mol.nelec[1]}")
    print()

    # ------------- FCI-free DMRG (SU2 mode targets 2S=1 doublet) -------
    print("=== DMRG: SU2 mode, spin=1, CAS(3,3), FCI conversion disabled ===")
    mc_dmrg = mcscf.CASCI(mf, 3, 3)
    mc_dmrg.fcisolver = MPSAsFCISolver(
        mol,
        bond_dim=120, n_sweeps=30, n_threads=1, sweep_tol=1e-12,
        force_dmrg=True, max_fci_dets=10_000,
        mps_native_rdms=True,
        skip_kernel_fci_conversion=True,
        first_iter_warmup=False,    # HF singlet bias not appropriate here
        stack_mem_mb=2000,
        warm_start=False,
        root_buffer=4,
        dmrg_symm_su2=True,         # SU2 mode (uses spin param as 2S)
    )
    mc_dmrg.fcisolver.nroots = 2
    # Tell the solver the target spin (S=1/2 → S^2 = S(S+1) = 0.75)
    mc_dmrg.fcisolver._target_ss = 0.75
    mc_dmrg.kernel()
    e_dmrg = list(map(float, mc_dmrg.e_tot))
    fsv = mc_dmrg.fcisolver
    dm1_0_dmrg, dm2_0_dmrg = fsv.make_rdm12(np.array([0.0]), 3, mol.nelec)
    print(f"  e_states = {e_dmrg}")
    print()

    print("=== diffs (atol = 1e-6) ===")
    ok = True
    ok &= compare("e_state0",  e_fci[0],     e_dmrg[0])
    ok &= compare("e_state1",  e_fci[1],     e_dmrg[1])
    ok &= compare("dm1_root0", dm1_0_fci,    dm1_0_dmrg)
    ok &= compare("dm2_root0", dm2_0_fci,    dm2_0_dmrg)
    print()
    if ok:
        print("DOUBLET ACCURACY: PASS — SU2 mode correctly targets the "
              "2S=1 sector without an FCI bridge.")
    else:
        print("DOUBLET ACCURACY: FAIL — investigate.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
