"""Fixed-orbital DMRG-CASCI derivative coupling numerator.

**Milestone 3 (CASCI partial)** of the analytic-DMRG-NAC plan.

Status (this session)
---------------------
- One-electron part `<Psi_I | (dh/dR) | Psi_J>`: analytic, validated against FD
  to numerical precision (T1_only_analytic vs T1_only_FD agree).
- Two-electron part `<Psi_I | (dg/dR) | Psi_J>`: implemented as FD only here.
  The fully analytic ERI-derivative contraction with a *transition* 2-RDM
  needs explicit handling of all four AO-index contributions to int2e_ip1,
  not the symmetrized state-density shortcut used in `pyscf.grad.casci`.
  That refactor is left for a follow-up session.
- Total derivative coupling numerator therefore exposed in two flavors:
    (a) `casci_dh_dR_numerator(...)` — semi-analytic (analytic T1 + FD T2).
    (b) `casci_dh_dR_numerator_full_fd(...)` — pure finite difference.
  Both should agree to FD-step precision, and both are independently
  validatable against the overlap-based time-derivative coupling tau_IJ
  multiplied by the energy gap.

Math
----
Full derivative coupling numerator at fixed MO coefficients C:

    N^x_{IJ} = sum_{pq} (dh_{pq}/dx)|_C T1_{pq}^IJ
             + (1/2) sum_{pqrs} (d(pq|rs)/dx)|_C T2_{pqrs}^IJ

The one-electron part transforms cleanly via PySCF's `hcore_generator`:

    dh_AO(R)/dR_A = pyscf.grad.rhf.Gradients.hcore_generator(mol)(A)

contracted with `T1_AO = C T1 C^T`.

The two-electron part requires AO-derivative integrals contracted with a
4-index density `T2_AO = sum C C C C T2`. PySCF's `int2e_ip1` differentiates
only AO index 1; obtaining the full atom derivative requires using the 8-fold
ERI symmetry to fold contributions from indices 2, 3, 4 into the same int2e_ip1
output, which works cleanly *only when the 4-index density itself has the
matching symmetry* (true for state densities, generally false for transition
densities). Until that's done properly, this module ships a finite-difference
implementation of the 2e contribution that uses the same T2 contracted with
displaced AO integrals at fixed C.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.grad import rhf as rhf_grad

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from overlap_fci_reference import cross_geometry_S_act, overlap_fci  # noqa: E402


def transition_density_in_ao(
    t1_act: np.ndarray,
    t2_act: np.ndarray,
    mo_coeff: np.ndarray,
    ncore: int,
    ncas: int,
):
    """Lift active-space transition RDMs to full AO basis.

    Returns
    -------
    T1_ao : (nao, nao)
    T2_ao : (nao, nao, nao, nao) in Mulliken (12|34) ordering.
    """
    nao, nmo = mo_coeff.shape
    mo_act = mo_coeff[:, ncore:ncore + ncas]
    T1_ao = mo_act @ t1_act @ mo_act.T
    # Use distinct letter cases for AO and MO indices to avoid einsum collisions
    # when nao == nmo (e.g. CAS(2,2)/sto-3g H2 or HeH+).
    T2_ao = np.einsum(
        "pqrt,Mp,Nq,Lr,St->MNLS",
        t2_act, mo_act, mo_act, mo_act, mo_act, optimize=True,
    )
    return T1_ao, T2_ao


def _t1_contribution(mol, atm_idx: int, T1_ao: np.ndarray, mf=None) -> np.ndarray:
    """sum_x <I| dh/dR_A,x |J> for one atom: (3,) array."""
    if mf is None:
        mf = scf.RHF(mol).run(conv_tol=1e-12)
    g = rhf_grad.Gradients(mf)
    h_atm = g.hcore_generator(mol)(atm_idx)
    return np.einsum("xij,ij->x", h_atm, T1_ao)


def _t2_contribution_fd(
    mol_factory,
    coords_ref: np.ndarray,
    atm_idx: int,
    t2_act: np.ndarray,
    mo_coeff: np.ndarray,
    ncore: int,
    ncas: int,
    dx_bohr: float = 1.0e-3,
) -> np.ndarray:
    """sum_x <I| (1/2) dg/dR_A,x |J> for one atom via finite difference.

    Uses fixed reference C and t2_act, displacing only the molecular geometry.
    """
    mo_act = mo_coeff[:, ncore:ncore + ncas]

    def e2_at(mol_disp):
        eri = mol_disp.intor("int2e", aosym="s1").reshape(
            (mol_disp.nao_nr(),) * 4
        )
        eri_act = np.einsum(
            "MNLS,Mp,Nq,Lr,St->pqrt",
            eri, mo_act, mo_act, mo_act, mo_act, optimize=True,
        )
        return 0.5 * float(np.einsum("pqrs,pqrs", eri_act, t2_act))

    out = np.zeros(3)
    for ax in range(3):
        disp = np.zeros_like(coords_ref)
        disp[atm_idx, ax] = dx_bohr
        mol_p = mol_factory(coords_ref + disp)
        mol_m = mol_factory(coords_ref - disp)
        out[ax] = (e2_at(mol_p) - e2_at(mol_m)) / (2 * dx_bohr)
    return out


def _make_factory_from_mol(mol):
    """Return a `mol_factory(coords_bohr)` that rebuilds `mol` at new coords."""
    atom_symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    basis = mol.basis
    charge = mol.charge
    spin = mol.spin

    def factory(coords_bohr):
        if coords_bohr is None:
            return mol
        return gto.M(
            atom=[(s, tuple(coords_bohr[i])) for i, s in enumerate(atom_symbols)],
            basis=basis, charge=charge, spin=spin, unit="Bohr",
            symmetry=False, verbose=0,
        )
    return factory


def casci_dh_dR_numerator(
    mol,
    mo_coeff: np.ndarray,
    t1_act: np.ndarray,
    t2_act: np.ndarray,
    ncore: int,
    ncas: int,
    atmlst: list[int] | None = None,
    mf=None,
    fd_step_bohr: float = 1.0e-3,
):
    """<Psi_I | dH/dR | Psi_J> at FIXED orbitals (semi-analytic).

    1e part: analytic via PySCF hcore_generator.
    2e part: finite difference over displaced AO integrals at fixed C.

    Returns
    -------
    nac_num : (natm, 3) array in Hartree/Bohr.
    """
    if atmlst is None:
        atmlst = list(range(mol.natm))
    if mf is None:
        mf = scf.RHF(mol).run(conv_tol=1e-12)

    T1_ao, _ = transition_density_in_ao(t1_act, t2_act, mo_coeff, ncore, ncas)

    factory = _make_factory_from_mol(mol)
    coords_ref = mol.atom_coords()

    nac = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        nac[k] += _t1_contribution(mol, ia, T1_ao, mf=mf)
        nac[k] += _t2_contribution_fd(
            factory, coords_ref, ia, t2_act, mo_coeff, ncore, ncas,
            dx_bohr=fd_step_bohr,
        )
    return nac


def casci_dh_dR_numerator_full_fd(
    mol,
    mo_coeff: np.ndarray,
    t1_act: np.ndarray,
    t2_act: np.ndarray,
    ncore: int,
    ncas: int,
    atmlst: list[int] | None = None,
    fd_step_bohr: float = 1.0e-3,
):
    """Pure-FD reference for `casci_dh_dR_numerator` cross-check."""
    if atmlst is None:
        atmlst = list(range(mol.natm))

    factory = _make_factory_from_mol(mol)
    coords_ref = mol.atom_coords()
    mo_act = mo_coeff[:, ncore:ncore + ncas]

    def e_at(mol_disp):
        h_ao = mol_disp.intor("int1e_kin") + mol_disp.intor("int1e_nuc")
        eri = mol_disp.intor("int2e", aosym="s1").reshape(
            (mol_disp.nao_nr(),) * 4
        )
        h_act = mo_act.T @ h_ao @ mo_act
        eri_act = np.einsum(
            "MNLS,Mp,Nq,Lr,St->pqrt",
            eri, mo_act, mo_act, mo_act, mo_act, optimize=True,
        )
        e1 = float(np.einsum("pq,pq", h_act, t1_act))
        e2 = 0.5 * float(np.einsum("pqrs,pqrs", eri_act, t2_act))
        return e1 + e2

    nac = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        for ax in range(3):
            disp = np.zeros_like(coords_ref)
            disp[ia, ax] = fd_step_bohr
            nac[k, ax] = (e_at(factory(coords_ref + disp))
                          - e_at(factory(coords_ref - disp))) / (2 * fd_step_bohr)
    return nac


def overlap_based_time_derivative_coupling(
    mol_factory,
    cas_factory,
    bra_idx: int,
    ket_idx: int,
    atmlst: list[int] | None = None,
    dx_bohr: float = 1.0e-3,
):
    """tau^x_{IJ} ≈ ( <I(R-dx)|J(R+dx)> - <I(R+dx)|J(R-dx)> ) / (2 dx).

    Includes wavefunction response (orbital relaxation + CI relaxation), unlike
    the fixed-orbital `casci_dh_dR_numerator`. Useful for cross-checks.
    """
    mol_ref = mol_factory(None)
    if atmlst is None:
        atmlst = list(range(mol_ref.natm))
    coords_ref = mol_ref.atom_coords()

    tau = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        for ax in range(3):
            disp = np.zeros_like(coords_ref)
            disp[ia, ax] = dx_bohr
            mol_p = mol_factory(coords_ref + disp)
            mol_m = mol_factory(coords_ref - disp)
            mc_p, ci_p_list = cas_factory(mol_p)
            mc_m, ci_m_list = cas_factory(mol_m)
            S_pm = cross_geometry_S_act(
                mol_p, mol_m, mc_p.mo_coeff, mc_m.mo_coeff,
                mc_p.ncas, mc_p.ncore,
            )
            S_mp = cross_geometry_S_act(
                mol_m, mol_p, mc_m.mo_coeff, mc_p.mo_coeff,
                mc_m.ncas, mc_m.ncore,
            )
            o_pm = overlap_fci(
                ci_p_list[bra_idx], ci_m_list[ket_idx],
                S_pm, mc_p.ncas, mc_p.nelecas,
            )
            o_mp = overlap_fci(
                ci_m_list[bra_idx], ci_p_list[ket_idx],
                S_mp, mc_m.ncas, mc_m.nelecas,
            )
            tau[k, ax] = (o_pm - o_mp) / (2.0 * dx_bohr)
    return tau
