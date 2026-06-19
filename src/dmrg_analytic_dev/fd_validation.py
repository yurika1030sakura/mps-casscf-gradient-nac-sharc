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

from pyscf import gto as _gto
from dmrg_fcisolver import MPSAsFCISolver
from overlap_fci_reference import (
    cross_geometry_S_act,
    overlap_fci,
    overlap_matrix_fci,
    assign_roots_by_overlap,
)
from site_replacement_density import _pyscf_to_block2_sign
from analytic_cp_sharc import compute_grad_nac_analytic_cp


def project_mo_to_new_geometry(mol_ref, mol_new, mo_ref):
    """Project reference MOs into the AO basis of a displaced geometry.

    Returns an ``S_new``-orthonormal MO guess with the same column ordering, so
    a displaced-geometry CASSCF starts from (and stays on) the same orbital /
    active-space surface as the reference -- the prerequisite for a meaningful
    finite-difference derivative.  Also returns the minimum singular value of
    the cross-geometry MO overlap as an orbital-continuity diagnostic.
    """
    S_new = mol_new.intor_symmetric("int1e_ovlp")
    S_cross = _gto.intor_cross("int1e_ovlp", mol_new, mol_ref)
    C = np.linalg.solve(S_new, S_cross @ mo_ref)         # AO least-squares map
    M = C.T @ S_new @ C
    evals, evecs = np.linalg.eigh(0.5 * (M + M.T))
    sigma_min = float(np.sqrt(max(np.min(evals), 0.0)))
    if np.min(evals) < 1.0e-8:
        raise RuntimeError(
            f"MO projection nearly singular (min eval {np.min(evals):.3e})"
        )
    X = evecs @ np.diag(evals ** -0.5) @ evecs.T          # symmetric orthonorm.
    return C @ X, sigma_min


def active_subspace_overlap(mol_a, mo_a, mol_b, mo_b, ncas, ncore):
    """Minimum singular value of the active-active cross-geometry overlap.

    sigma_min ~ 1 means the active space is continuous between the two
    geometries (the finite-difference point is on the same surface); a small
    value flags a discontinuity that invalidates the FD component."""
    S_act = cross_geometry_S_act(mol_a, mol_b, mo_a, mo_b, ncas, ncore)
    s = np.linalg.svd(S_act, compute_uv=False)
    return float(np.min(s))


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

# Determinant-space size above which a dense FCI vector cannot be formed.  The
# default solver config carries the FCI-convertible settings (so small cases
# keep a cheap dense readout for scoring); past this threshold the build path
# forces the FCI-free settings regardless of what the caller passed, so a large
# active space can never silently trigger a dense MPS->FCI conversion.
FCI_FREE_THRESHOLD = 5.0e7


def _determinant_dimension(ncas, nelecas) -> float:
    from math import comb
    if isinstance(nelecas, (tuple, list)):
        na, nb = int(nelecas[0]), int(nelecas[1])
    else:
        na = int(nelecas) // 2 + int(nelecas) % 2
        nb = int(nelecas) // 2
    return float(comb(int(ncas), na) * comb(int(ncas), nb))


def _enforce_fci_free(cfg: dict, ncas, nelecas) -> dict:
    """Force FCI-free solver settings when the determinant space is too large.

    Returns ``cfg`` unchanged below the threshold; above it, sets
    ``skip_kernel_fci_conversion`` and ``mps_native_rdms`` to ``True`` so the
    solver never attempts a dense FCI readout it cannot hold.
    """
    if _determinant_dimension(ncas, nelecas) > FCI_FREE_THRESHOLD:
        cfg = dict(cfg)
        cfg["skip_kernel_fci_conversion"] = True
        cfg["mps_native_rdms"] = True
    return cfg


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
    cfg = _enforce_fci_free(cfg, ncas, nelecas)
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


def _build_mol(atoms, coords_bohr, basis, charge, spin):
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    atom_list = [(atoms[i], tuple(coords_bohr[i])) for i in range(len(atoms))]
    return _gto.M(atom=atom_list, basis=basis, charge=charge, spin=spin,
                  unit="Bohr", symmetry=False, verbose=0)


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
    track_roots="fci_overlap",
    gap_min=1.0e-3,
    subspace_min=0.90,
    return_diagnostics=False,
    mo_guess=None,
):
    """Central-difference gradient of E_state from the DMRG state energies.

    Displaced-geometry CASSCF runs are seeded with the reference orbitals
    projected into the displaced AO basis, so both sides of the difference stay
    on the same active-space surface.

    ``track_roots`` selects how the displaced state index is chosen:
      * ``"fci_overlap"`` -- assign the displaced root to the reference root by
        cross-geometry FCI wavefunction overlap (FCI/MPS-bridge must be
        tractable; use only when the determinant space is small enough);
      * ``"gap_guard"`` -- keep the energy-ordered index but require the state
        gap to stay above ``gap_min`` at R and R+-h (the FCI-free choice for
        beyond-FCI active spaces);
      * ``False`` -- plain energy-ordered index, no guard.

    In every mode the active-active cross-geometry overlap singular value is
    recorded as an orbital-continuity diagnostic; this needs only the
    (ncas x ncas) MO overlap and is available at any active-space size.

    Returns the (natom, 3) gradient array, or ``(grad, diagnostics)`` when
    ``return_diagnostics`` is set.
    """
    if track_roots is True:
        track_roots = "fci_overlap"
    ddim = _determinant_dimension(ncas, nelecas)
    if ddim > FCI_FREE_THRESHOLD and track_roots == "fci_overlap":
        raise ValueError(
            f"fd_gradient: determinant space det_dim={ddim:.3e} exceeds the "
            f"FCI-free threshold {FCI_FREE_THRESHOLD:.1e}; 'fci_overlap' root "
            f"tracking forms dense CI vectors via mps_ci_list and is forbidden "
            f"for beyond-FCI active spaces. Use track_roots='gap_guard' "
            f"(or 'mps_subspace')."
        )
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    natom = len(atoms)
    grad = np.zeros((natom, 3))

    # reference build (needed for the projected MO guess in every mode)
    mol0, _mf0, mc0, solver0 = build_sa_dmrg_casscf(
        atoms, coords_bohr, basis=basis, charge=charge, spin=spin,
        ncas=ncas, nelecas=nelecas, nroots=nroots, weights=weights,
        solver_cfg=solver_cfg, mo_guess=mo_guess,
    )
    ncore = mc0.ncore
    mo0 = mc0.mo_coeff
    e0 = state_energies(solver0)
    ci0 = (mps_ci_list(solver0, ncas, mc0.nelecas, nroots)
           if track_roots == "fci_overlap" else None)

    diags = {"track_roots": track_roots, "components": [],
             "ref_gap": float(e0[1] - e0[0]) if len(e0) > 1 else None}

    for (a, x) in _atom_components(natom, atmlst, components):
        e_disp, mo_disp, mol_disp, sig_disp, idx = {}, {}, {}, {}, {}
        for sgn in (+1, -1):
            cc = coords_bohr.copy()
            cc[a, x] += sgn * h_bohr
            mol_s = _build_mol(atoms, cc, basis, charge, spin)
            mo_guess, _smin_full = project_mo_to_new_geometry(mol0, mol_s, mo0)
            _mol, _mf, mc_s, solver_s = build_sa_dmrg_casscf(
                atoms, cc, basis=basis, charge=charge, spin=spin,
                ncas=ncas, nelecas=nelecas, nroots=nroots, weights=weights,
                solver_cfg=solver_cfg, mo_guess=mo_guess,
            )
            e_disp[sgn] = state_energies(solver_s)
            mo_disp[sgn] = mc_s.mo_coeff
            mol_disp[sgn] = mol_s
            # active-space continuity (cheap ncas x ncas overlap; no FCI)
            sig_disp[sgn] = active_subspace_overlap(
                mol0, mo0, mol_s, mc_s.mo_coeff, ncas, ncore)
            idx[sgn] = state
            if track_roots == "fci_overlap":
                ci_s = mps_ci_list(solver_s, ncas, mc_s.nelecas, nroots)
                S = cross_geometry_S_act(mol0, mol_s, mo0, mc_s.mo_coeff,
                                         ncas, ncore)
                O = overlap_matrix_fci(ci0, ci_s, S, ncas, mc0.nelecas)
                perm, _ = assign_roots_by_overlap(O)
                idx[sgn] = int(perm[state])

        # guards
        sig_min = min(sig_disp[+1], sig_disp[-1])
        gp = e_disp[+1]; gm = e_disp[-1]
        gap_p = float(gp[1] - gp[0]) if len(gp) > 1 else None
        gap_m = float(gm[1] - gm[0]) if len(gm) > 1 else None
        if track_roots == "gap_guard":
            gaps = [g for g in (diags["ref_gap"], gap_p, gap_m) if g is not None]
            if gaps and min(gaps) < gap_min:
                raise RuntimeError(
                    f"FD component (atom {a}, axis {x}) too close to a root "
                    f"crossing (min gap {min(gaps):.2e} < {gap_min}); use "
                    f"subspace tracking or skip this component."
                )
        if sig_min < subspace_min:
            raise RuntimeError(
                f"active-space discontinuity at FD component (atom {a}, axis "
                f"{x}): sigma_min {sig_min:.3f} < {subspace_min}"
            )

        grad[a, x] = (e_disp[+1][idx[+1]] - e_disp[-1][idx[-1]]) / (2.0 * h_bohr)
        diags["components"].append({
            "atom": int(a), "axis": int(x), "h_bohr": float(h_bohr),
            "g_fd": float(grad[a, x]),
            "active_subspace_sigma_min": sig_min,
            "gap_plus": gap_p, "gap_minus": gap_m,
            "idx_plus": int(idx[+1]), "idx_minus": int(idx[-1]),
        })

    return (grad, diags) if return_diagnostics else grad


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

    ddim = _determinant_dimension(ncas, nelecas)
    if ddim > FCI_FREE_THRESHOLD:
        raise ValueError(
            f"fd_nac: determinant space det_dim={ddim:.3e} exceeds the FCI-free "
            f"threshold {FCI_FREE_THRESHOLD:.1e}; this determinant-overlap NAC "
            f"finite difference forms dense CI vectors via mps_ci_list. For "
            f"beyond-FCI active spaces use the MPS-native cross-geometry overlap "
            f"path (cross_geometry_overlap.cross_geometry_overlap_matrix / "
            f"benchmarks/large_active_space/run_beyond_fci_nac.py)."
        )

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
