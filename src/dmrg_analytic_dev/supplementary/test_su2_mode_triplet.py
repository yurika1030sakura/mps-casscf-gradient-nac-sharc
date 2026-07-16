"""
Validate that v10 (skip_fci + SU2 mode) correctly targets a non-singlet spin
sector on a system whose ground state is a triplet.

Approach: H4 chain at d=2.0 Bohr (near dissociation), where the singlet and
triplet manifolds are nearly degenerate. We explicitly target the triplet
sector via mol.spin=2 (Sz=1 representative). FCI in that sector gives the
ground triplet energy; DMRG SU2 with spin=2 in initialize_system should
match.

This exercises:
  - SU2 mode targeting the correct (non-singlet) spin sector
  - Energy-sort selection in skip_kernel_fci_conversion path picking the
    triplet roots (rather than crossing into a lower singlet — which would
    actually be the BUG case the SU2 mode prevents)
  - make_rdm12 placeholder path on triplet MPS
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from pyscf import gto, scf, mcscf, fci

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from dmrg_fcisolver import MPSAsFCISolver


def make_h4_triplet(d_bohr: float = 2.0):
    """H4 chain at stretched geometry, configured as a triplet (mol.spin=2).
    Sz=1 representative: 3 alpha + 1 beta electron."""
    return gto.M(atom=f"H 0 0 0; H 0 0 {d_bohr}; H 0 0 {2*d_bohr}; "
                       f"H 0 0 {3*d_bohr}", basis="sto-3g",
                 charge=0, spin=2, unit="Bohr",
                 symmetry=False, verbose=0)


def compare(name, ref, got, atol=1e-6):
    r = np.asarray(ref).ravel(); g = np.asarray(got).ravel()
    diff = float(np.max(np.abs(r - g)))
    status = "PASS" if diff < atol else "FAIL"
    print(f"  {name:18s}  max|diff|={diff:.3e}  [{status}]")
    return diff < atol


def main():
    mol = make_h4_triplet()
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    print(f"Mol: {mol.nelec} electrons (na, nb), spin=2 (triplet sector)")
    print()

    # ------------- FCI reference (CASCI in fixed orbitals) -------------
    print("=== FCI reference: H4 CAS(4,4) triplet (Sz=1, 2S=2) ===")
    mc_fci = mcscf.CASCI(mf, 4, 4)
    mc_fci.fcisolver.nroots = 2
    mc_fci.kernel()
    e_fci = list(map(float, mc_fci.e_tot))
    ci0 = mc_fci.ci[0]
    dm1_0_fci, dm2_0_fci = fci.direct_spin1.make_rdm12(ci0, 4, mol.nelec)
    print(f"  e_states = {e_fci}")
    print(f"  Sz from FCI: na={mol.nelec[0]}, nb={mol.nelec[1]}, "
          f"2Sz={mol.nelec[0]-mol.nelec[1]}")
    print()

    # ------------- v10 DMRG (SU2 mode targets 2S=2 triplet) -----------
    print("=== v10 DMRG: SU2 mode, spin=2 (auto from na-nb), CAS(4,4) ===")
    mc_v10 = mcscf.CASCI(mf, 4, 4)
    mc_v10.fcisolver = MPSAsFCISolver(
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
    mc_v10.fcisolver.nroots = 2
    # Tell the solver the target spin (S=1 → S^2 = S(S+1) = 2)
    mc_v10.fcisolver._target_ss = 2.0
    mc_v10.kernel()
    e_v10 = list(map(float, mc_v10.e_tot))
    fsv = mc_v10.fcisolver
    dm1_0_v10, dm2_0_v10 = fsv.make_rdm12(np.array([0.0]), 4, mol.nelec)
    print(f"  e_states = {e_v10}")
    print()

    print("=== diffs (atol = 1e-6) ===")
    ok = True
    ok &= compare("e_state0",  e_fci[0],     e_v10[0])
    ok &= compare("e_state1",  e_fci[1],     e_v10[1])
    ok &= compare("dm1_root0", dm1_0_fci,    dm1_0_v10)
    ok &= compare("dm2_root0", dm2_0_fci,    dm2_0_v10)
    print()
    if ok:
        print("TRIPLET ACCURACY: PASS — SU2 mode correctly targets the "
              "2S=2 sector.")
    else:
        print("TRIPLET ACCURACY: FAIL — investigate.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
