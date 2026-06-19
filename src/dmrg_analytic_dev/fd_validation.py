"""Finite-difference validation of analytic SA-DMRG-CASSCF gradients and NACs.

Two checks are provided, both against the *same* SA-DMRG-CASSCF quantities the
analytic code produces (not against a higher-level or FCI reference):

* gradient -- central difference of the DMRG state energy,
      g_I[a,x] = (E_I(R + h e_ax) - E_I(R - h e_ax)) / (2 h),
  with E_I read from the solver's own ``e_states``;

* derivative coupling -- central difference of the cross-geometry wavefunction
  overlap,
      tau^x_IJ = ( <I(R-h)|J(R+h)> - <I(R+h)|J(R-h)> ) / (2 h),
  evaluated with full orbital + CI relaxation.

The gradient check needs no wavefunction overlap and therefore remains
available where FCI is impossible (CAS(20,20)+); it is the primary validation
at the largest active spaces.  The overlap-based coupling check runs at the
active spaces where the MPS->FCI bridge is tractable.

Reused machinery (nothing re-derived here):
  * cross-geometry active overlap  -> overlap_fci_reference.cross_geometry_S_act
  * non-orthogonal CI overlap      -> overlap_fci_reference.overlap_fci
  * root matching / phase signs    -> overlap_fci_reference.assign_roots_by_overlap
  * MPS -> FCI coefficient bridge  -> site_replacement_density.mps_to_fci_generic
  * analytic gradient / NAC entry  -> analytic_cp_sharc.compute_grad_nac_analytic_cp
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pyscf import gto, scf, mcscf

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parents[1] / "sharc_interface"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dmrg_fcisolver import MPSAsFCISolver
from overlap_fci_reference import (
    cross_geometry_S_act,
    overlap_fci,
    overlap_matrix_fci,
    assign_roots_by_overlap,
)
from site_replacement_density import _pyscf_to_block2_sign
from analytic_cp_sharc import compute_grad_nac_analytic_cp


# Solver configuration used for finite differences. force_dmrg=True so the
# energies and wavefunctions are the genuine DMRG ones; dmrg_symm_su2=True so
# the stored roots are spin-pure (an SZ-mode open-shell excited state comes
# out spin-contaminated, which corrupts the overlap-based derivative coupling
# even though its energy is correct); warm_start / persistent cache OFF so a
# displaced geometry never inherits a converged guess from the reference
# point (that would bias E_I(R+-h)).
DEFAULT_SOLVER_CFG = dict(
    bond_dim=200,
    n_sweeps=30,
    sweep_tol=1.0e-12,
    n_threads=1,
    mps_native_rdms=False,
    skip_kernel_fci_conversion=False,
    dmrg_symm_su2=True,
    force_dmrg=True,
    warm_start=False,
    mps_persistent_dir=None,
)

DEFAULT_H_SCAN = (2.0e-3, 1.0e-3, 5.0e-4, 2.0e-4, 1.0e-4)


def _assert_fd_safe(cfg: dict) -> None:
    if cfg.get("warm_start", False):
        raise ValueError("finite differences require warm_start=False")
    if cfg.get("mps_persistent_dir") is not None:
        raise ValueError("finite differences require mps_persistent_dir=None")


def build_sa_dmrg_casscf(
    atoms,
    coords_bohr,
    *,
    basis,
    charge=0,
    spin=0,
    ncas,
    nelecas,
    nroots,
    weights=None,
    solver_cfg=None,
    mo_guess=None,
):
    """Build and converge SA-DMRG-CASSCF at one geometry (Bohr coordinates).

    Returns ``(mol, mf, mc, solver)``; ``solver`` is the live ``mc.fcisolver``
    holding the SZ driver and converged MPS roots.
    """
    cfg = dict(DEFAULT_SOLVER_CFG if solver_cfg is None else solver_cfg)
    _assert_fd_safe(cfg)
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    atom_list = [(atoms[i], tuple(coords_bohr[i])) for i in range(len(atoms))]
    mol = gto.M(
        atom=atom_list, basis=basis, charge=charge, spin=spin,
        unit="Bohr", symmetry=False, verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1.0e-12)
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    # Tight orbital convergence is essential for finite-difference NAC: the
    # cross-geometry overlap is first order in the orbital-rotation error, so
    # loose CASSCF convergence (pyscf default ~1e-7) injects orbital-gauge
    # noise into the derivative coupling even though the variational energy
    # (hence the FD gradient) is only second order in that error.
    mc.conv_tol = 1.0e-10
    mc.conv_tol_grad = 1.0e-8
    mc.max_cycle_macro = 200
    solver = MPSAsFCISolver(mol, **cfg)
    solver.nroots = int(nroots)
    mc.fcisolver = solver
    if weights is None:
        weights = [1.0 / nroots] * nroots
    if nroots > 1:
        mc = mc.state_average_(list(weights))
    mc.kernel(mo_guess)
    return mol, mf, mc, mc.fcisolver


def state_energies(solver) -> np.ndarray:
    return np.asarray(solver.e_states, dtype=float).ravel()


def _su2_ket_to_fci(su2_driver, sz_driver, mps, ncas, nelec, *, tag,
                    cutoff=1.0e-12):
    """Convert one spin-adapted (SU2) DMRG root to a PySCF FCI ndarray.

    The SU2 MPS is first re-expressed in the SZ basis, its CSF coefficients
    are read out, and they are scattered into the (na_str, nb_str) FCI array
    with the PySCF<->block2 determinant-ordering sign.  Mirrors the converter
    used by the anthracene CAS(14,14) benchmark.
    """
    from pyscf.fci import cistring

    na, nb = int(nelec[0]), int(nelec[1])
    mps_sz = su2_driver.mps_change_to_sz(mps, tag=tag)
    dets, coefs = sz_driver.get_csf_coefficients(
        mps_sz, cutoff=float(cutoff), iprint=0,
    )
    strs_a = list(cistring.make_strings(range(ncas), na))
    strs_b = list(cistring.make_strings(range(ncas), nb))
    a_idx = {int(s): j for j, s in enumerate(strs_a)}
    b_idx = {int(s): j for j, s in enumerate(strs_b)}
    ci = np.zeros((len(strs_a), len(strs_b)), dtype=np.float64)
    for det, c in zip(dets, coefs):
        c = float(c)
        if abs(c) < cutoff:
            continue
        sa = sb = 0
        for site, occ in enumerate(det):
            occ = int(occ)
            if occ == 3:
                sa |= (1 << site); sb |= (1 << site)
            elif occ == 1:
                sa |= (1 << site)
            elif occ == 2:
                sb |= (1 << site)
        ia = a_idx.get(sa); ib = b_idx.get(sb)
        if ia is None or ib is None:
            continue
        ci[ia, ib] = _pyscf_to_block2_sign(sa, sb, ncas) * c
    norm = float(np.linalg.norm(ci))
    if norm > 1.0e-30:
        ci /= norm
    return ci


def mps_ci_list(solver, ncas, nelec, nroots) -> list:
    """Convert the converged spin-adapted DMRG roots to FCI coefficient arrays.

    The solver runs in SU2 mode, so ``solver._driver`` is the SU2 driver and
    the kets are spin-pure; a throwaway SZ driver provides the CSF readout.
    """
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    nelec = (int(nelec[0]), int(nelec[1]))
    n_elec_tot = nelec[0] + nelec[1]
    su2_driver = solver._driver
    # The SZ driver MUST share the SU2 solver's scratch: mps_change_to_sz reads
    # the SU2 MPS files written there, and a fresh scratch would not see them.
    scratch = solver._scratch
    sz_driver = DMRGDriver(
        scratch=scratch, clean_scratch=False,
        stack_mem=int(solver.stack_mem_mb) * 1024 * 1024,
        n_threads=int(solver.n_threads),
        symm_type=SymmetryTypes.SZ,
    )
    sz_driver.initialize_system(
        n_sites=int(ncas), n_elec=int(n_elec_tot), spin=0,
        orb_sym=[0] * int(ncas),
    )
    return [
        _su2_ket_to_fci(
            su2_driver, sz_driver, solver._kets[i], int(ncas), nelec,
            tag=f"FDSZ-{i}",
        )
        for i in range(int(nroots))
    ]


def _atom_components(natom, atmlst, components):
    if atmlst is None:
        atmlst = list(range(natom))
    if components is None:
        components = (0, 1, 2)
    return [(a, x) for a in atmlst for x in components]


def fd_gradient(
    atoms,
    coords_bohr,
    *,
    state,
    basis,
    charge=0,
    spin=0,
    ncas,
    nelecas,
    nroots,
    weights=None,
    solver_cfg=None,
    h_bohr=1.0e-3,
    atmlst=None,
    components=None,
    track_roots=True,
):
    """Central-difference gradient of E_state from the DMRG state energies.

    Returns an (natom, 3) array (unevaluated components left as 0) in
    E_h/Bohr.  When ``track_roots`` is set, the displaced state energies are
    re-ordered to follow the reference root by maximum overlap, guarding
    against energy-sorted root swaps at avoided crossings.
    """
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    natom = len(atoms)
    grad = np.zeros((natom, 3))
    ncore = None

    # reference roots, for optional root tracking across displacements
    if track_roots:
        mol0, _mf0, mc0, solver0 = build_sa_dmrg_casscf(
            atoms, coords_bohr, basis=basis, charge=charge, spin=spin,
            ncas=ncas, nelecas=nelecas, nroots=nroots, weights=weights,
            solver_cfg=solver_cfg,
        )
        ncore = mc0.ncore
        ci0 = mps_ci_list(solver0, ncas, mc0.nelecas, nroots)
        mo0 = mc0.mo_coeff

    for (a, x) in _atom_components(natom, atmlst, components):
        e_disp = {}
        ci_disp = {}
        mo_disp = {}
        mol_disp = {}
        for sgn in (+1, -1):
            cc = coords_bohr.copy()
            cc[a, x] += sgn * h_bohr
            mol_s, _mf_s, mc_s, solver_s = build_sa_dmrg_casscf(
                atoms, cc, basis=basis, charge=charge, spin=spin,
                ncas=ncas, nelecas=nelecas, nroots=nroots, weights=weights,
                solver_cfg=solver_cfg,
            )
            e_disp[sgn] = state_energies(solver_s)
            if track_roots:
                ci_disp[sgn] = mps_ci_list(solver_s, ncas, mc_s.nelecas, nroots)
                mo_disp[sgn] = mc_s.mo_coeff
                mol_disp[sgn] = mol_s

        idx_p, idx_m = state, state
        if track_roots:
            # match displaced roots back to the reference root `state`
            S_p = cross_geometry_S_act(
                mol0, mol_disp[+1], mo0, mo_disp[+1], ncas, ncore,
            )
            O_p = overlap_matrix_fci(ci0, ci_disp[+1], S_p, ncas, mc0.nelecas)
            perm_p, _ = assign_roots_by_overlap(O_p)
            idx_p = int(perm_p[state])
            S_m = cross_geometry_S_act(
                mol0, mol_disp[-1], mo0, mo_disp[-1], ncas, ncore,
            )
            O_m = overlap_matrix_fci(ci0, ci_disp[-1], S_m, ncas, mc0.nelecas)
            perm_m, _ = assign_roots_by_overlap(O_m)
            idx_m = int(perm_m[state])

        grad[a, x] = (e_disp[+1][idx_p] - e_disp[-1][idx_m]) / (2.0 * h_bohr)
    return grad


def analytic_gradient(mc, state, *, backend="mps-krylov", tol=1.0e-8, max_iter=500):
    """Analytic SA-DMRG-CASSCF gradient of `state` via the response backend."""
    res = compute_grad_nac_analytic_cp(
        mc, gradient_states=[int(state)], nac_pairs=None,
        backend=backend, tol=tol, max_iter=max_iter,
    )
    return np.asarray(res["grad"][int(state)], dtype=float)


def fd_nac(
    atoms,
    coords_bohr,
    *,
    bra,
    ket,
    basis,
    charge=0,
    spin=0,
    ncas,
    nelecas,
    nroots,
    weights=None,
    solver_cfg=None,
    h_bohr=1.0e-3,
    atmlst=None,
    components=None,
):
    """Overlap finite-difference derivative coupling d^x_{bra,ket}.

    Matches the manuscript definition d^A_IJ = <Psi_I | dPsi_J/dR_A> directly:

        d^x = ( <I(R) | J(R+h)> - <I(R) | J(R-h)> ) / (2 h),

    with the bra fixed at the reference geometry R and the displaced ket
    phase-aligned to the reference root so the arbitrary DMRG sign at R+-h
    does not leak into the difference.  This bra-fixed central difference
    reproduces the analytic derivative coupling with unit factor (no 2x from a
    doubly-displaced bra/ket form).
    """
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    natom = len(atoms)
    dcoup = np.zeros((natom, 3))

    mol0, _mf0, mc0, solver0 = build_sa_dmrg_casscf(
        atoms, coords_bohr, basis=basis, charge=charge, spin=spin,
        ncas=ncas, nelecas=nelecas, nroots=nroots, weights=weights,
        solver_cfg=solver_cfg,
    )
    ncore = mc0.ncore
    nelec = mc0.nelecas
    ci0 = mps_ci_list(solver0, ncas, nelec, nroots)
    mo0 = mc0.mo_coeff

    def _aligned(coords):
        """Build at displaced geometry; reorder+rephase roots onto R labels."""
        mol_s, _mf_s, mc_s, solver_s = build_sa_dmrg_casscf(
            atoms, coords, basis=basis, charge=charge, spin=spin,
            ncas=ncas, nelecas=nelecas, nroots=nroots, weights=weights,
            solver_cfg=solver_cfg,
        )
        ci_s = mps_ci_list(solver_s, ncas, nelec, nroots)
        S = cross_geometry_S_act(mol0, mol_s, mo0, mc_s.mo_coeff, ncas, ncore)
        O = overlap_matrix_fci(ci0, ci_s, S, ncas, nelec)
        perm, signs = assign_roots_by_overlap(O)
        ci_aligned = [None] * nroots
        for I in range(nroots):
            j = int(perm[I])
            ci_aligned[I] = float(signs[I]) * ci_s[j]
        return mol_s, mc_s.mo_coeff, ci_aligned

    for (a, x) in _atom_components(natom, atmlst, components):
        cp = coords_bohr.copy(); cp[a, x] += h_bohr
        cm = coords_bohr.copy(); cm[a, x] -= h_bohr
        mol_p, mo_p, ci_p = _aligned(cp)
        mol_m, mo_m, ci_m = _aligned(cm)
        # <I(R) | J(R+h)> and <I(R) | J(R-h)>, bra fixed at reference
        S_0p = cross_geometry_S_act(mol0, mol_p, mo0, mo_p, ncas, ncore)
        S_0m = cross_geometry_S_act(mol0, mol_m, mo0, mo_m, ncas, ncore)
        o_p = overlap_fci(ci0[bra], ci_p[ket], S_0p, ncas, nelec)
        o_m = overlap_fci(ci0[bra], ci_m[ket], S_0m, ncas, nelec)
        dcoup[a, x] = (o_p - o_m) / (2.0 * h_bohr)
    return dcoup


def phase_aware_error(a: np.ndarray, b: np.ndarray) -> float:
    """min(||a-b||, ||a+b||) -- robust to an overall sign convention."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(min(
        np.linalg.norm(a - b), np.linalg.norm(a + b),
    ))


def grad_error(analytic: np.ndarray, fd: np.ndarray, mask=None):
    a = np.asarray(analytic, dtype=float)
    f = np.asarray(fd, dtype=float)
    if mask is not None:
        a = a[mask]; f = f[mask]
    diff = (a - f).ravel()
    return {
        "max_abs_err": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms_err": float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0,
    }
