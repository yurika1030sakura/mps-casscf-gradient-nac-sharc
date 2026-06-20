"""High-accuracy finite-difference validation of the determinant cross-geometry
derivative coupling (NAC), pushing the 2-point FD error from ~1e-4 toward <1e-5.

Background
----------
``fd_validation.fd_nac`` forms the 2-point central difference of the *active*
CAS-determinant cross-geometry overlap,

    f_ij(s) = <Psi_i(R0) | Psi_j(R0 + s e_k)>_active ,   d_ij = (f(h)-f(-h))/(2h)

with the bra fixed at the reference geometry and the displaced ket phase/sign
aligned to the reference root.  At a single step h this carries an O(h^2)
truncation error (~1e-4 at h~1e-3) plus two systematic biases that the analytic
derivative coupling does *not* drop:

  (a) the closed-shell **core** orbitals also move with R, so the true total
      wavefunction overlap is

          <Phi_i(R0)|Phi_j(R0+s)>_total
              = det[(C_core0)^T S_cross C_core(s)]^2  *  f_ij(s)

      (the ^2 from alpha+beta closed-shell spin); the determinant-active path
      omits the core factor, biasing the slope;

  (b) finite-step truncation, removable by higher-order / Richardson FD.

This module rebuilds the *same* SA-CASSCF (small, FCI-exact CAS) and compares
several FD estimators of the off-diagonal d_01 against the analytic CP response.

Estimators implemented (off-diagonal d_01, one chosen atom/axis):
  1. 2-point        : (f(h) - f(-h)) / (2h)
  2. 5-point        : (-f(2h) + 8 f(h) - 8 f(-h) + f(-2h)) / (12 h)
  3. Richardson D4  : (4 D2(h) - D2(2h)) / 3
  4. core-corrected : items 1-3 applied to F(s) = det[...]^2 * f(s) instead of f
  5. block-leakage  : ||(C_core0)^T S_cross C_act(s)||_F and
                      ||(C_act0)^T S_cross C_virt(s)||_F  (small = clean blocks)
  6. gap-weighted   : h_01 = (E1-E0) d_01 , its error too

Nothing from fd_validation / overlap_fci_reference is re-derived; their helpers
are imported.  This file is import-only against those modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pyscf import gto as _gto

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parents[1] / "sharc_interface"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fd_validation as fdv
from fd_validation import (
    build_sa_dmrg_casscf,
    mps_ci_list,
    project_mo_to_new_geometry,
    _build_mol,
)
from overlap_fci_reference import (
    cross_geometry_S_act,
    overlap_fci,
    overlap_matrix_fci,
    assign_roots_by_overlap,
)
from analytic_cp_sharc import compute_grad_nac_analytic_cp


# ----------------------------------------------------------------------------
# tight solver config (FCI-exact small CAS): SCF 1e-12 / CASSCF conv 1e-10 are
# already baked into build_sa_dmrg_casscf; here we keep the DMRG bond dim modest
# and sweeps tight so the MPS == FCI vector to machine precision.
# ----------------------------------------------------------------------------
TIGHT_SOLVER_CFG = dict(fdv.DEFAULT_SOLVER_CFG)
TIGHT_SOLVER_CFG.update(bond_dim=64, n_sweeps=24, sweep_tol=1.0e-12)


def core_overlap_factor(mol0, mol_s, C0, Cs, ncore):
    """Closed-shell core cross-geometry overlap factor and its determinant.

    Returns ``(factor, det_core)`` where

        det_core = det[(C0_core)^T S_cross(R0,Rs) Cs_core]
        factor   = det_core ** 2          (alpha + beta closed shell)

    ``det_core`` -> 1 as Rs -> R0; its deviation from 1 is the core
    contribution that the active-only determinant overlap omits.
    """
    if ncore == 0:
        return 1.0, 1.0
    S_AO = _gto.intor_cross("int1e_ovlp", mol0, mol_s)
    Cc0 = C0[:, :ncore]
    Ccs = Cs[:, :ncore]
    S_core = Cc0.T @ S_AO @ Ccs
    det_core = float(np.linalg.det(S_core))
    return det_core * det_core, det_core


def block_leakage(mol0, mol_s, C0, Cs, ncore, ncas):
    """Frobenius norms of off-block cross-geometry MO overlaps (continuity).

    Returns dict with:
      core_act  = ||(C0_core)^T S_cross Cs_act||_F   (core leaking into active)
      act_virt  = ||(C0_act)^T  S_cross Cs_virt||_F  (active leaking into virt)
    Both should be small (the displaced active/core/virt blocks stay block-
    diagonal under the cross-geometry overlap); large values flag a rotation
    that would corrupt the determinant-only overlap derivative.
    """
    S_AO = _gto.intor_cross("int1e_ovlp", mol0, mol_s)
    SC = S_AO @ Cs                       # AO_0 x MO_s
    core0 = C0[:, :ncore]
    act0 = C0[:, ncore:ncore + ncas]
    acts = slice(ncore, ncore + ncas)
    virts = slice(ncore + ncas, Cs.shape[1])
    M_core_act = core0.T @ SC[:, acts]
    M_act_virt = act0.T @ SC[:, virts]
    return {
        "core_act": float(np.linalg.norm(M_core_act)),
        "act_virt": float(np.linalg.norm(M_act_virt)),
    }


def _aligned_build(case, coords, mol0, mo0, ci0, ncore):
    """Build SA-CASSCF at displaced ``coords``; return overlap-aligned data.

    Returns (mol_s, mo_s, ci_aligned, e_states).  The displaced roots are
    permuted+rephased onto the reference root labels by maximum FCI overlap, and
    the CASSCF is seeded by the projected reference MOs so it stays on the same
    orbital surface (prerequisite for a meaningful derivative).
    """
    ncas = case["ncas"]
    nelecas = case["nelecas"]
    nroots = case["nroots"]
    mol_guess = _build_mol(case["atoms"], coords, case["basis"],
                           case["charge"], case["spin"])
    mo_guess, _smin = project_mo_to_new_geometry(mol0, mol_guess, mo0)
    mol_s, _mf_s, mc_s, solver_s = build_sa_dmrg_casscf(
        case["atoms"], coords, basis=case["basis"], charge=case["charge"],
        spin=case["spin"], ncas=ncas, nelecas=nelecas, nroots=nroots,
        weights=case["weights"], solver_cfg=case["solver_cfg"],
        mo_guess=mo_guess,
    )
    ci_s = mps_ci_list(solver_s, ncas, mc_s.nelecas, nroots)
    S = cross_geometry_S_act(mol0, mol_s, mo0, mc_s.mo_coeff, ncas, ncore)
    O = overlap_matrix_fci(ci0, ci_s, S, ncas, mc_s.nelecas)
    perm, signs = assign_roots_by_overlap(O)
    ci_aligned = [None] * nroots
    for I in range(nroots):
        j = int(perm[I])
        ci_aligned[I] = float(signs[I]) * ci_s[j]
    e_states = np.asarray(solver_s.e_states, dtype=float).ravel()
    # reorder energies onto reference labels too
    e_aligned = np.array([e_states[int(perm[I])] for I in range(nroots)])
    return mol_s, mc_s.mo_coeff, ci_aligned, e_aligned


def nac_fd_estimators(case, *, bra=0, ket=1, atom, axis, h_bohr=1.0e-3):
    """Compute all FD estimators of d_{bra,ket} along one (atom, axis).

    Builds f_ij(s) and core-corrected F_ij(s) = det[...]^2 f_ij(s) at the four
    displacements s in {-2h, -h, +h, +2h}, then forms 2-point / 5-point /
    Richardson estimators of d_01 from both, plus block-leakage diagnostics and
    the gap-weighted coupling.

    Returns a dict (JSON-serializable scalars).
    """
    ncas = case["ncas"]
    nelecas = case["nelecas"]
    nroots = case["nroots"]

    coords0 = np.asarray(case["coords_bohr"], dtype=float)

    # reference build
    mol0, _mf0, mc0, solver0 = build_sa_dmrg_casscf(
        case["atoms"], coords0, basis=case["basis"], charge=case["charge"],
        spin=case["spin"], ncas=ncas, nelecas=nelecas, nroots=nroots,
        weights=case["weights"], solver_cfg=case["solver_cfg"],
    )
    ncore = mc0.ncore
    mo0 = mc0.mo_coeff
    ci0 = mps_ci_list(solver0, ncas, mc0.nelecas, nroots)
    e0 = np.asarray(solver0.e_states, dtype=float).ravel()
    gap0 = float(e0[1] - e0[0])

    # displacements: s in {+2h,+h,-h,-2h}
    steps = {"+2h": +2.0 * h_bohr, "+h": +1.0 * h_bohr,
             "-h": -1.0 * h_bohr, "-2h": -2.0 * h_bohr}

    f = {}        # active-only determinant overlap f_ij(s)
    F = {}        # core-corrected total overlap F_ij(s)
    det_core = {} # core overlap determinant at each step
    leak = {}     # block-leakage diagnostics
    gaps = {}

    for key, s in steps.items():
        cc = coords0.copy()
        cc[atom, axis] += s
        mol_s, mo_s, ci_s, e_s = _aligned_build(case, cc, mol0, mo0, ci0, ncore)
        # active determinant overlap <Psi_bra(R0)|Psi_ket(Rs)>
        S_act = cross_geometry_S_act(mol0, mol_s, mo0, mo_s, ncas, ncore)
        o_act = overlap_fci(ci0[bra], ci_s[ket], S_act, ncas, mc0.nelecas)
        factor, dc = core_overlap_factor(mol0, mol_s, mo0, mo_s, ncore)
        f[key] = float(o_act)
        F[key] = float(factor * o_act)
        det_core[key] = float(dc)
        leak[key] = block_leakage(mol0, mol_s, mo0, mo_s, ncore, ncas)
        gaps[key] = float(e_s[1] - e_s[0])

    h = h_bohr

    def d2(g):  # 2-point central diff of dict g (uses +h,-h)
        return (g["+h"] - g["-h"]) / (2.0 * h)

    def d2_2h(g):  # 2-point central diff at step 2h
        return (g["+2h"] - g["-2h"]) / (4.0 * h)

    def d5(g):  # 5-point central diff
        return (-g["+2h"] + 8.0 * g["+h"] - 8.0 * g["-h"] + g["-2h"]) / (12.0 * h)

    def richardson(g):  # D4 = (4 D2(h) - D2(2h)) / 3
        return (4.0 * d2(g) - d2_2h(g)) / 3.0

    est = {
        "two_point": d2(f),
        "five_point": d5(f),
        "richardson": richardson(f),
        "core_two_point": d2(F),
        "core_five_point": d5(F),
        "core_richardson": richardson(F),
    }

    return {
        "f": f, "F": F, "det_core": det_core, "leak": leak,
        "gap_ref": gap0, "gaps": gaps,
        "estimators": est,
        "ncore": int(ncore),
        "mc0": mc0,  # live object for the analytic reference
    }


def analytic_nac_z(mc0, *, bra=0, ket=1, atom, axis, tol=1.0e-8, max_iter=200):
    """Analytic CP-CASSCF derivative coupling component d_{bra,ket}[atom,axis].

    Uses backend='mps-krylov'.  With pyscf's default mult_ediff=False the
    returned NAC is divided by (E_bra - E_ket), i.e. it IS the derivative
    coupling d (not the gap-weighted h), so it compares directly to the FD
    overlap slope.
    """
    res = compute_grad_nac_analytic_cp(
        mc0, gradient_states=[], nac_pairs=[(bra, ket)],
        backend="mps-krylov", tol=tol, max_iter=max_iter,
    )
    d = np.asarray(res["nac"][(bra, ket)], dtype=float)
    return float(d[atom, axis]), d


def _phase_abs_err(a, b):
    """min(|a-b|, |a+b|): robust to an overall sign convention."""
    return float(min(abs(a - b), abs(a + b)))


def run_case(case, *, bra=0, ket=1, atom, axis, h_bohr=1.0e-3,
             label="case"):
    """Full comparison for one molecule: FD estimators vs analytic NAC.

    Returns a JSON-serializable result dict (no live pyscf objects).
    """
    fdres = nac_fd_estimators(case, bra=bra, ket=ket, atom=atom, axis=axis,
                              h_bohr=h_bohr)
    mc0 = fdres.pop("mc0")
    d_an, _full = analytic_nac_z(mc0, bra=bra, ket=ket, atom=atom, axis=axis)

    est = fdres["estimators"]
    gap = fdres["gap_ref"]

    errors = {name: _phase_abs_err(d_an, val) for name, val in est.items()}

    # gap-weighted coupling h_01 = (E1-E0) * d_01 for each estimator + analytic
    h_an = gap * d_an
    gap_errors = {name: _phase_abs_err(h_an, gap * val)
                  for name, val in est.items()}

    # pick best (smallest |err|) FD estimator
    best_name = min(errors, key=errors.get)

    leak_max = max(
        max(v["core_act"], v["act_virt"]) for v in fdres["leak"].values()
    )
    det_core_dev = max(abs(v - 1.0) for v in fdres["det_core"].values())

    return {
        "label": label,
        "bra": bra, "ket": ket, "atom": atom, "axis": axis, "h_bohr": h_bohr,
        "analytic_d": d_an,
        "gap_ref": gap,
        "analytic_gapweighted": h_an,
        "estimators": est,
        "errors": errors,
        "gap_weighted_errors": gap_errors,
        "best_estimator": best_name,
        "best_error": errors[best_name],
        "det_core": fdres["det_core"],
        "det_core_max_dev_from_1": det_core_dev,
        "block_leakage": fdres["leak"],
        "block_leakage_max": leak_max,
        "gaps_displaced": fdres["gaps"],
        "ncore": fdres["ncore"],
    }


# ----------------------------------------------------------------------------
# system definitions
# ----------------------------------------------------------------------------
def heh_case():
    return dict(
        atoms=["He", "H"],
        coords_bohr=np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.90 / 0.52917721067]]),
        basis="3-21G", charge=1, spin=0,
        ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5],
        solver_cfg=TIGHT_SOLVER_CFG,
    )


def ethylene_case():
    """C2H4 / 6-31G CAS(2,2) pi/pi*.  Planar D2h equilibrium-ish geometry."""
    ang = 1.0 / 0.52917721067
    # planar ethylene: C=C along z, H in the xz plane
    coords_ang = np.array([
        [0.0,  0.0,  0.6695],   # C
        [0.0,  0.0, -0.6695],   # C
        [0.0,  0.9289,  1.2321],  # H
        [0.0, -0.9289,  1.2321],  # H
        [0.0,  0.9289, -1.2321],  # H
        [0.0, -0.9289, -1.2321],  # H
    ])
    return dict(
        atoms=["C", "C", "H", "H", "H", "H"],
        coords_bohr=coords_ang * ang,
        basis="6-31G", charge=0, spin=0,
        ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5],
        solver_cfg=TIGHT_SOLVER_CFG,
    )
