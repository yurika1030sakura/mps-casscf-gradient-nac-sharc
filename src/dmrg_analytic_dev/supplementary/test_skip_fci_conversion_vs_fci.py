"""
Validate v9 fast-path eigensolver + RDM accuracy vs FCI on H4 chain CAS(4,4).

Scope of this test:
  - SA-CASSCF converged energies (the v9 sort + placeholder energy selection)
  - 1- and 2-RDMs via make_rdm12 placeholder path (block2 NPDM on cached MPS)
  - transition RDMs (block2 NPDM on bra/ket cached MPS pair)

These are the quantities v9 actually changes. The full gradient/NAC pipeline
is handled by the MPS-Krylov response (CPDMRGCASSCFResponseMPSKrylov), which
is validated separately by test_mps_krylov_response.py and test_mps_krylov_
sharc_interface.py — that path does not call PySCF contract_2e and is
unaffected by the skip_kernel_fci_conversion changes.

Run:
    python3 test_skip_fci_conversion_vs_fci.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from pyscf import gto, scf, mcscf, fci

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from dmrg_fcisolver import MPSAsFCISolver


def make_h4(d_bohr: float = 1.5):
    return gto.M(atom=f"H 0 0 0; H 0 0 {d_bohr}; H 0 0 {2*d_bohr}; "
                       f"H 0 0 {3*d_bohr}", basis="sto-3g",
                 charge=0, spin=0, unit="Bohr",
                 symmetry=False, verbose=0)


def compare(name, ref, got, atol=1e-6):
    r = np.asarray(ref); g = np.asarray(got)
    diff = np.max(np.abs(r - g))
    diff_sign = np.max(np.abs(r + g))
    best = min(diff, diff_sign)
    status = "PASS" if best < atol else "FAIL"
    print(f"  {name:18s}  max|diff|={diff:.3e}  max|sum|={diff_sign:.3e}"
          f"  best={best:.3e}  [{status}]")
    return best < atol


def main():
    mol = make_h4()
    mf = scf.RHF(mol).run(conv_tol=1e-12)

    # ----------------- FCI reference (SA-CASSCF + RDMs) -----------------
    print("=== FCI reference: H4 chain CAS(4,4)/sto-3g SA(2 singlets) ===")
    mc_fci = mcscf.CASSCF(mf, 4, 4)
    mc_fci.fix_spin_(ss=0)
    mc_fci.fcisolver.nroots = 2
    mc_fci = mc_fci.state_average_([0.5, 0.5])
    mc_fci.kernel()
    e_fci = list(map(float, mc_fci.e_states))
    # mc.ci is the list of FCI vectors per root after state-averaging
    ci0_fci, ci1_fci = mc_fci.ci[0], mc_fci.ci[1]
    dm1_0_fci, dm2_0_fci = fci.direct_spin1.make_rdm12(ci0_fci, 4, (2, 2))
    dm1_1_fci, dm2_1_fci = fci.direct_spin1.make_rdm12(ci1_fci, 4, (2, 2))
    t1_fci, t2_fci = fci.direct_spin1.trans_rdm12(ci0_fci, ci1_fci, 4, (2, 2))
    print(f"  e_states = {e_fci}")
    print()

    # ----------------- v9 DMRG (all fast-path knobs ON) -----------------
    print("=== v9 DMRG (skip_fci ON, mps_native_rdms ON, energy-sort, ")
    print("    no buffer, no refine, first_iter_warmup ON) ===")
    mc_v9 = mcscf.CASSCF(mf, 4, 4)
    mc_v9.fcisolver = MPSAsFCISolver(
        mol,
        bond_dim=120, n_sweeps=30, n_threads=1, sweep_tol=1e-12,
        force_dmrg=True, max_fci_dets=10_000,
        # v9+v10 fast path knobs
        mps_native_rdms=True,
        skip_kernel_fci_conversion=True,
        first_iter_warmup=True,
        stack_mem_mb=2000,
        warm_start=True,
        root_buffer=4,         # static API; kernel forces nroots
        dmrg_symm_su2=True,    # NEW: spin-adapted block2 DMRG so the energy-
                               # sort selection cannot cross-pick a triplet
    )
    mc_v9.fcisolver.nroots = 2
    mc_v9 = mc_v9.state_average_([0.5, 0.5])
    mc_v9.kernel()
    e_v9 = list(map(float, mc_v9.e_states))
    # In v9 path, mc.ci entries are 1-element placeholders. We invoke
    # MPSAsFCISolver.make_rdm12 directly with placeholders — the same code
    # path PySCF's macro iter uses internally.
    fsv = mc_v9.fcisolver
    dm1_0_v9, dm2_0_v9 = fsv.make_rdm12(np.array([0.0]), 4, (2, 2))
    dm1_1_v9, dm2_1_v9 = fsv.make_rdm12(np.array([1.0]), 4, (2, 2))
    t1_v9, t2_v9 = fsv.trans_rdm12(np.array([0.0]), np.array([1.0]), 4, (2, 2))
    print(f"  e_states = {e_v9}")
    print()

    # ----------------- compare -----------------
    print("=== diffs (atol = 1e-6) ===")
    ok = True
    ok &= compare("e_state0",     e_fci[0],     e_v9[0])
    ok &= compare("e_state1",     e_fci[1],     e_v9[1])
    ok &= compare("dm1_root0",    dm1_0_fci,    dm1_0_v9)
    ok &= compare("dm1_root1",    dm1_1_fci,    dm1_1_v9)
    ok &= compare("dm2_root0",    dm2_0_fci,    dm2_0_v9)
    ok &= compare("dm2_root1",    dm2_1_fci,    dm2_1_v9)
    ok &= compare("trans_dm1",    t1_fci,       t1_v9)
    ok &= compare("trans_dm2",    t2_fci,       t2_v9)
    print()
    if ok:
        print("ACCURACY: PASS — v9 fast path matches FCI on H4 CAS(4,4).")
        print("Energies, RDMs, and transition RDMs all agree to numerical "
              "precision.")
    else:
        print("ACCURACY: FAIL — investigate.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
