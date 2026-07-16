"""
Spin-sector GRADIENT finite-difference validation for SA-DMRG-CASSCF.

Upgrades the doublet/triplet (non-singlet) validation from energy/RDM-only
(supplementary/test_su2_mode_{doublet,triplet}.py) to the DERIVATIVE level:
for each non-singlet system we compute

  (a) the analytic gradient of state 0 via the response backend
      [fd_validation.analytic_gradient(mc, 0) -> compute_grad_nac_analytic_cp];
  (b) the central finite-difference gradient of state 0 via
      fd_validation.fd_gradient(..., track_roots='gap_guard');

and compare the same (atom, axis) component.  Target |analytic - FD| < 1e-5
Eh/Bohr.

IMPORTANT: the analytic gradient via the response (CP) backend may not support
non-singlet sectors.  If it raises, we capture the EXACT traceback (a real
finding) and still report the FD gradient plus the energy/RDM agreement vs
PySCF FCI, so the integrator knows whether the FD machinery and the spin
sector itself are sound even if the analytic path is not yet wired for
non-singlet states.

Systems:
  - H3 linear chain, spacing 1.8 Bohr, spin=1 (doublet, 2S=1), CAS(3,3).
  - H4 linear chain, spacing 2.0 Bohr, spin=2 (triplet, 2S=2), CAS(4,4).
Basis STO-3G, nroots=2, weights [.5, .5].
"""
from __future__ import annotations
import sys
import os
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import fd_validation as fdv
from pyscf import gto, scf, mcscf, fci


def make_chain(nH, spacing):
    atoms = ["H"] * nH
    coords = np.array([[0.0, 0.0, i * spacing] for i in range(nH)], dtype=float)
    return atoms, coords


def fci_reference(atoms, coords, *, spin, ncas, nelecas, nroots):
    """CASCI/FCI reference energies + state-0 RDMs in the targeted spin sector."""
    atom_list = [(atoms[i], tuple(coords[i])) for i in range(len(atoms))]
    mol = gto.M(atom=atom_list, basis="sto-3g", charge=0, spin=spin,
                unit="Bohr", symmetry=False, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver.nroots = nroots
    mc.kernel()
    e = list(map(float, mc.e_tot))
    dm1, dm2 = fci.direct_spin1.make_rdm12(mc.ci[0], ncas, mol.nelec)
    return mol, e, dm1, dm2


def fci_free_cfg():
    """FCI-free SU2 solver config.

    The DEFAULT_SOLVER_CFG keeps the legacy SU2-MPS->FCI bridge
    (skip_kernel_fci_conversion=False), and that bridge produces a
    zero/garbage CI vector for NON-singlet sectors -- the DMRG energies then
    collapse to the bare nuclear-repulsion value (active energy ~0).  The
    working doublet/triplet supplementary tests avoid this by running fully
    FCI-free (mps_native_rdms + skip_kernel_fci_conversion), which targets the
    requested 2S sector correctly.  We mirror that here.
    """
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg["skip_kernel_fci_conversion"] = True
    cfg["mps_native_rdms"] = True
    return cfg


def run_system(name, *, nH, spacing, spin, ncas, nelecas):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    nroots = 2
    weights = [0.5, 0.5]
    cfg = fci_free_cfg()
    atoms, coords = make_chain(nH, spacing)

    twos = spin                      # SU2 2S
    S = spin / 2.0
    target_ss = S * (S + 1.0)
    print(f"  SU2 twos (=spin) = {twos}; target S^2 = (spin/2)(spin/2+1) "
          f"= {target_ss:.4f}")

    result = {"name": name, "twos": twos, "target_ss": target_ss}

    # ----- FCI/CASCI reference (energy + RDM agreement) -----
    mol_fci, e_fci, dm1_fci, dm2_fci = fci_reference(
        atoms, coords, spin=spin, ncas=ncas, nelecas=nelecas, nroots=nroots)
    print(f"  FCI e_states = {e_fci}")
    result["e_fci"] = e_fci

    # ----- Build SA-DMRG-CASSCF -----
    mol, mf, mc, solver = fdv.build_sa_dmrg_casscf(
        atoms, coords, basis="sto-3g", charge=0, spin=spin,
        ncas=ncas, nelecas=nelecas, nroots=nroots, weights=weights,
        solver_cfg=cfg)
    e_dmrg = list(fdv.state_energies(solver))
    print(f"  DMRG e_states = {e_dmrg}")
    result["e_dmrg"] = e_dmrg
    e_err = float(np.max(np.abs(np.array(e_fci) - np.array(e_dmrg))))
    print(f"  |E_DMRG - E_FCI| max = {e_err:.3e}")
    result["energy_err_vs_fci"] = e_err

    # RDM agreement state 0 (via the SU2->FCI bridge used by mps_ci_list)
    try:
        dm1_dmrg, dm2_dmrg = solver.make_rdm12(np.array([0.0]), ncas, mol.nelec)
        d1 = float(np.max(np.abs(np.asarray(dm1_dmrg) - dm1_fci)))
        d2 = float(np.max(np.abs(np.asarray(dm2_dmrg) - dm2_fci)))
        print(f"  |dm1_DMRG - dm1_FCI| max = {d1:.3e}")
        print(f"  |dm2_DMRG - dm2_FCI| max = {d2:.3e}")
        result["dm1_err_vs_fci"] = d1
        result["dm2_err_vs_fci"] = d2
    except Exception as exc:
        print(f"  RDM comparison failed: {exc!r}")
        result["rdm_error"] = repr(exc)

    # ----- (a) Analytic gradient of state 0 -----
    analytic_ok = False
    g_analytic_comp = None
    try:
        g_analytic = fdv.analytic_gradient(mc, 0)
        g_analytic = np.asarray(g_analytic, dtype=float)
        g_analytic_comp = float(g_analytic[1, 2])   # atom 1, axis z
        analytic_ok = True
        print(f"  ANALYTIC grad[atom1, z] = {g_analytic_comp:.8e}")
        result["analytic_grad"] = g_analytic_comp
    except Exception as exc:
        tb = traceback.format_exc()
        print("  ANALYTIC gradient RAISED (non-singlet not supported?):")
        print(tb)
        result["analytic_error"] = repr(exc)
        result["analytic_traceback"] = tb

    # ----- (b) Finite-difference gradient of state 0 -----
    g_fd = fdv.fd_gradient(
        atoms, coords, state=0, basis="sto-3g", charge=0, spin=spin,
        ncas=ncas, nelecas=nelecas, nroots=nroots, weights=weights,
        solver_cfg=cfg,
        track_roots="gap_guard", atmlst=[1], components=[2],
        return_diagnostics=False)
    g_fd_comp = float(np.asarray(g_fd)[1, 2])
    print(f"  FD       grad[atom1, z] = {g_fd_comp:.8e}")
    result["fd_grad"] = g_fd_comp

    if analytic_ok:
        err = abs(g_analytic_comp - g_fd_comp)
        result["grad_abs_err"] = err
        result["target_met"] = err < 1e-5
        print(f"  |analytic - FD| = {err:.3e}   "
              f"(target < 1e-5: {'PASS' if err < 1e-5 else 'FAIL'})")
    else:
        result["grad_abs_err"] = None
        result["target_met"] = None
        print("  |analytic - FD| = N/A (analytic backend did not run)")

    return result


def main():
    results = []
    results.append(run_system(
        "H3 doublet (spin=1, 2S=1), spacing 1.8 Bohr, CAS(3,3)",
        nH=3, spacing=1.8, spin=1, ncas=3, nelecas=3))
    results.append(run_system(
        "H4 triplet (spin=2, 2S=2), spacing 2.0 Bohr, CAS(4,4)",
        nH=4, spacing=2.0, spin=2, ncas=4, nelecas=4))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in results:
        print(f"\n{r['name']}")
        print(f"  twos={r['twos']}  target S^2={r['target_ss']:.4f}")
        print(f"  energy_err_vs_fci = {r.get('energy_err_vs_fci')}")
        print(f"  dm1_err_vs_fci    = {r.get('dm1_err_vs_fci')}")
        print(f"  dm2_err_vs_fci    = {r.get('dm2_err_vs_fci')}")
        if "analytic_grad" in r:
            print(f"  analytic_grad     = {r['analytic_grad']:.8e}")
        else:
            print(f"  analytic_grad     = ERROR: {r.get('analytic_error')}")
        print(f"  fd_grad           = {r['fd_grad']:.8e}")
        print(f"  grad_abs_err      = {r.get('grad_abs_err')}")
        print(f"  target_met        = {r.get('target_met')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
