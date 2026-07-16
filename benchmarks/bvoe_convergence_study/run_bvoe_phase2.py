"""BVOE convergence study — Phase 2 (REAL DMRG).

Phase 1 used CI-vector SVD truncation as a proxy for DMRG. Phase 2 uses
actual spin-adapted block2 DMRG at fixed SA-CASSCF orbitals.  A configurable
candidate-root buffer is included so that the target roots can be selected by
overlap with the fixed-orbital FCI reference.

Pipeline at each (system, M):

  1. Run SA(2)-CASSCF with the standard PySCF FCI fcisolver to get converged
     orbitals (mc.mo_coeff), then polish the CI roots by fixed-orbital FCI.
  2. At the converged orbitals, build the SU2-mode DMRG MPO from the
     active-space integrals.
  3. Run SU2 DMRG with ``nroots=2+BVOE_ROOT_BUFFER``, ``bond_dim = M``.
  4. Convert each candidate MPS to a PySCF FCI ndarray via the SZ-mode CSF route
     (``mps_change_to_sz`` then ``get_csf_coefficients``, then PySCF
     ordering-sign correction).
  5. Select and phase-align the two target roots by overlap with the polished
     FCI validation reference.
  6. Install the bond-dim-M CI vectors back into ``mc`` (replacing
     ``mc.ci``), and recompute the analytic gradient (state 0) and NAC
     (states (0,1)) with PySCF's standard ``pyscf.grad.sacasscf`` /
     ``pyscf.nac.sacasscf`` machinery.

Reference: the same SA-CASSCF orbitals followed by fixed-orbital
rediagonalization of the singlet active-space Hamiltonian with PySCF
``direct_spin0`` FCI.  This benchmark isolates the bond-dimension response
error of the active-space wavefunction at identical orbitals.

Output:
  data_phase2/{system}_M{M}.json     raw results per (system, M)
  data_phase2/{system}_FCI.json      reference (PySCF FCI)
  summary_phase2.json                aggregated diff norms
"""

from __future__ import annotations

import json
import itertools
import multiprocessing as mp
import os
import queue
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.fci import cistring, spin_op
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf.nac import sacasscf as nac_sacasscf

ROOT = Path(__file__).resolve().parent
DEV_ROOT = ROOT.parent
for candidate in (
    DEV_ROOT,
    ROOT.parents[2] / "dmrg_sacasscf_response_public" / "src" / "dmrg_analytic_dev",
    ROOT.parents[1] / "src" / "dmrg_analytic_dev",
):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
from site_replacement_density import _pyscf_to_block2_sign  # noqa: E402


# ---------------------------------------------------------------------------
# Test systems
# ---------------------------------------------------------------------------

def build_h2o_basis(basis="sto-3g", active=(4, 4)):
    mol = gto.M(atom="""
        O   0.0000   0.0000   0.0000
        H   0.0000   0.7572   0.5868
        H   0.0000  -0.7572   0.5868
    """, basis=basis, spin=0, charge=0, verbose=0)
    return mol, int(active[0]), int(active[1])


def build_h2o():
    return build_h2o_basis("sto-3g", active=(4, 4))


def build_h4_basis(basis="sto-3g"):
    R_bohr = 1.5
    R_ang = R_bohr * 0.529177210903
    coords = "\n".join(f"H   {i * R_ang:.6f}  0.0  0.0" for i in range(4))
    mol = gto.M(atom=coords, basis=basis, spin=0, charge=0,
                unit="Angstrom", verbose=0)
    return mol, 4, 4


def build_h4():
    return build_h4_basis("sto-3g")


def build_n2_basis(basis="sto-3g"):
    mol = gto.M(atom="N 0 0 0; N 0 0 1.4",
                basis=basis, spin=0, charge=0,
                unit="Angstrom", verbose=0)
    return mol, 6, 6


def build_n2():
    return build_n2_basis("sto-3g")


def build_c2_basis(basis="sto-3g"):
    mol = gto.M(atom="C 0 0 0; C 0 0 1.25",
                basis=basis, spin=0, charge=0,
                unit="Angstrom", verbose=0)
    return mol, 8, 8


def build_c2():
    return build_c2_basis("sto-3g")


def build_lif_basis(basis="sto-3g"):
    # Near the ionic/covalent avoided-crossing region; this gives a more
    # informative NAC benchmark than symmetry-suppressed equilibrium N2.
    mol = gto.M(atom="Li 0 0 0; F 0 0 6.5",
                basis=basis, spin=0, charge=0,
                unit="Bohr", verbose=0)
    return mol, 4, 4


def build_lif_avoided():
    return build_lif_basis("sto-3g")


def build_h2o_631g_cas66():
    return build_h2o_basis("6-31g", active=(6, 6))


def build_ethylene_basis(basis="sto-3g"):
    # Planar ethylene is a canonical two-state pi/pi* active-space benchmark.
    mol = gto.M(atom="""
        C   0.000000   0.000000   0.669500
        C   0.000000   0.000000  -0.669500
        H   0.000000   0.928900   1.232100
        H   0.000000  -0.928900   1.232100
        H   0.000000   0.928900  -1.232100
        H   0.000000  -0.928900  -1.232100
    """, basis=basis, spin=0, charge=0, unit="Angstrom",
        symmetry=False, verbose=0)
    return mol, 2, 2


def build_butadiene_basis(basis="sto-3g"):
    # Approximate trans-1,3-butadiene; CAS(4,4) targets the pi manifold.
    mol = gto.M(atom="""
        C  -2.0580   0.0000   0.0000
        C  -0.7110   0.0000   0.0000
        C   0.7110   0.0000   0.0000
        C   2.0580   0.0000   0.0000
        H  -2.6400   0.9300   0.0000
        H  -2.6400  -0.9300   0.0000
        H  -0.1690   0.9450   0.0000
        H   0.1690  -0.9450   0.0000
        H   2.6400   0.9300   0.0000
        H   2.6400  -0.9300   0.0000
    """, basis=basis, spin=0, charge=0, unit="Angstrom",
        symmetry=False, verbose=0)
    return mol, 4, 4


def build_formaldehyde_basis(basis="sto-3g"):
    # Standard carbonyl n/pi* photochemistry test case.  The compact CAS(4,4)
    # active space keeps the benchmark in an FCI-reference regime.
    mol = gto.M(atom="""
        C   0.0000   0.0000   0.0000
        O   0.0000   0.0000   1.2080
        H   0.0000   0.9360  -0.5870
        H   0.0000  -0.9360  -0.5870
    """, basis=basis, spin=0, charge=0, unit="Angstrom",
        symmetry=False, verbose=0)
    return mol, 4, 4


def build_benzene_basis(basis="sto-3g"):
    # Regular hexagon benzene; CAS(6,6) is the classic pi-space benchmark.
    r_c = 1.397
    r_h = 2.479
    lines = []
    for i in range(6):
        theta = 2.0 * np.pi * i / 6.0
        lines.append(f"C  {r_c*np.cos(theta): .8f}  {r_c*np.sin(theta): .8f}  0.0")
    for i in range(6):
        theta = 2.0 * np.pi * i / 6.0
        lines.append(f"H  {r_h*np.cos(theta): .8f}  {r_h*np.sin(theta): .8f}  0.0")
    mol = gto.M(atom="\n".join(lines), basis=basis, spin=0, charge=0,
                unit="Angstrom", symmetry=False, verbose=0)
    return mol, 6, 6


def _basis_builder(kind, basis):
    if kind == "h4":
        return lambda basis=basis: build_h4_basis(basis)
    if kind == "h2o":
        active = (4, 4) if basis == "sto-3g" else (6, 6)
        return lambda basis=basis, active=active: build_h2o_basis(
            basis, active=active
        )
    if kind == "n2":
        return lambda basis=basis: build_n2_basis(basis)
    if kind == "c2":
        return lambda basis=basis: build_c2_basis(basis)
    if kind == "lif":
        return lambda basis=basis: build_lif_basis(basis)
    if kind == "ethylene":
        return lambda basis=basis: build_ethylene_basis(basis)
    if kind == "butadiene":
        return lambda basis=basis: build_butadiene_basis(basis)
    if kind == "formaldehyde":
        return lambda basis=basis: build_formaldehyde_basis(basis)
    if kind == "benzene":
        return lambda basis=basis: build_benzene_basis(basis)
    raise KeyError(kind)


def _basis_key(kind, basis):
    suffix = {
        "sto-3g": "",
        "3-21g": "_321g",
        "6-31g": "_631g",
    }[basis]
    return f"{kind}{suffix}"


def _basis_label(kind, basis):
    labels = {
        "h4": f"H4 chain / {basis} / CAS(4,4) SA(2), R=1.5 Bohr",
        "h2o": (
            f"H2O / {basis} / "
            f"{'CAS(4,4)' if basis == 'sto-3g' else 'CAS(6,6)'} SA(2)"
        ),
        "n2": f"N2 / {basis} / CAS(6,6) SA(2), R=1.4 Ang",
        "c2": f"C2 / {basis} / CAS(8,8) SA(2), R=1.25 Ang",
        "lif": f"LiF / {basis} / CAS(4,4) SA(2), R=6.5 Bohr",
        "ethylene": f"ethylene / {basis} / CAS(2,2) SA(2)",
        "butadiene": f"trans-butadiene / {basis} / CAS(4,4) SA(2)",
        "formaldehyde": f"formaldehyde / {basis} / CAS(4,4) SA(2)",
        "benzene": f"benzene / {basis} / CAS(6,6) SA(2)",
    }
    return labels[kind]


# (builder, label, M scan, full bipartite rank).
# NOTE: M=1 is excluded from default scans because the rank-1 CI vector
# makes the SA-CASSCF preconditioner overlap matrix singular, which sends
# pyscf's gradient solver into an LAPACK-warning loop (xsyev info=...).
# That is a *correct* numerical signal that M=1 is in the singular regime,
# but it produces minute-to-hour stalls and floods the log. The M=2..rank
# range gives the full convergence picture for the figure.
BASIS_SET_MATRIX = ("sto-3g", "3-21g", "6-31g")
M_SCANS = {
    "h4": [2, 3, 4, 5, 6, 8, 12, 200],
    "h2o": [2, 3, 4, 5, 6, 8, 12, 20, 30, 200],
    "n2": [2, 4, 8, 12, 16, 20, 30, 200],
    # CAS(8,8) is the only present small-FCI benchmark where M=200 is still a
    # truncation rather than an effectively full-rank calculation; include
    # higher-M points under the same fixed-reference protocol.
    "c2": [4, 8, 16, 32, 64, 70, 120, 200, 400, 800],
    "lif": [2, 3, 4, 5, 6, 8, 12, 20, 30, 200],
    "ethylene": [2, 3, 4, 6, 8, 12, 200],
    "butadiene": [2, 3, 4, 5, 6, 8, 12, 20, 200],
    "formaldehyde": [2, 3, 4, 5, 6, 8, 12, 20, 200],
    "benzene": [4, 8, 12, 16, 24, 32, 64, 120, 200],
}


def _build_systems():
    systems = {}
    for kind in (
        "h4", "h2o", "n2", "c2", "lif",
        "ethylene", "butadiene", "formaldehyde", "benzene",
    ):
        for basis in BASIS_SET_MATRIX:
            key = _basis_key(kind, basis)
            systems[key] = (
                _basis_builder(kind, basis),
                _basis_label(kind, basis),
                M_SCANS[kind],
                200,
            )
    return systems


SYSTEMS = _build_systems()

# General root-buffer setting, matching the production SHARC interface.
# Benchmarks use FCI overlaps only to score the result; production runs use
# the same extra-root idea with previous-step overlaps instead of FCI.
ROOT_BUFFER = int(os.environ.get("BVOE_ROOT_BUFFER", "4"))
DMRG_DAV_THRD = float(os.environ.get("BVOE_DAV_THRD", "1e-14"))
DMRG_DAV_MAX_ITER = int(os.environ.get("BVOE_DAV_MAX_ITER", "8000"))
DMRG_DAV_DEF_MAX_SIZE = int(os.environ.get("BVOE_DAV_DEF_MAX_SIZE", "80"))
DMRG_SWEEP_TOL = float(os.environ.get("BVOE_SWEEP_TOL", "1e-14"))
REFINE_SPLIT_ROOTS = os.environ.get("BVOE_REFINE_SPLIT_ROOTS", "1") != "0"
REFINE_SWEEPS = int(os.environ.get("BVOE_REFINE_SWEEPS", "20"))
REFINE_SWEEP_TOL = float(os.environ.get("BVOE_REFINE_SWEEP_TOL", "1e-12"))
REFINE_PROJ_WEIGHT = float(os.environ.get("BVOE_REFINE_PROJ_WEIGHT", "5.0"))
MPS_COEFF_CUTOFF = float(os.environ.get("BVOE_MPS_COEFF_CUTOFF", "1e-16"))
SA_CASSCF_CONV_TOL = float(os.environ.get("BVOE_CASSCF_CONV_TOL", "1e-12"))
SA_CASSCF_CONV_TOL_GRAD = float(os.environ.get(
    "BVOE_CASSCF_CONV_TOL_GRAD", "1e-9"
))
SA_CASSCF_MAX_CYCLE = int(os.environ.get("BVOE_CASSCF_MAX_CYCLE", "300"))
FCI_POLISH_CONV_TOL = float(os.environ.get("BVOE_FCI_POLISH_CONV_TOL",
                                           "1e-14"))
FCI_POLISH_MAX_CYCLE = int(os.environ.get("BVOE_FCI_POLISH_MAX_CYCLE",
                                          "300"))
FCI_POLISH_MAX_SPACE = int(os.environ.get("BVOE_FCI_POLISH_MAX_SPACE", "80"))
FCI_POLISH_PSPACE_SIZE = int(os.environ.get(
    "BVOE_FCI_POLISH_PSPACE_SIZE", "2000"
))
FCI_POLISH_SCAN_ROOTS = int(os.environ.get(
    "BVOE_FCI_POLISH_SCAN_ROOTS", "20"
))
FCI_POLISH_SPIN_SHIFT = float(os.environ.get("BVOE_FCI_POLISH_SPIN_SHIFT",
                                             "0.5"))
FCI_POLISH_SPIN_TOL = float(os.environ.get("BVOE_FCI_POLISH_SPIN_TOL",
                                           "1e-6"))
FCI_DEGENERACY_TOL = float(os.environ.get("BVOE_FCI_DEGENERACY_TOL", "1e-4"))
ROOT_LOCK_OVERLAP = float(os.environ.get("BVOE_ROOT_LOCK_OVERLAP", "0.90"))
ROOT_LOCK_MARGIN = float(os.environ.get("BVOE_ROOT_LOCK_MARGIN", "0.10"))
MIN_DERIVATIVE_ROOT_OVERLAP = float(
    os.environ.get("BVOE_MIN_DERIVATIVE_ROOT_OVERLAP", "0.50")
)
POINT_TIMEOUT_S = float(os.environ.get("BVOE_POINT_TIMEOUT_S", "1800"))
SCHEMA_VERSION = 5


def _singlet_csf_dim(norb, nelec):
    """Spin-adapted singlet CSF count for closed-shell CAS(n_e, n_orb)."""
    if isinstance(nelec, (tuple, list, np.ndarray)):
        neleca, nelecb = int(nelec[0]), int(nelec[1])
        nelec_tot = neleca + nelecb
        if neleca != nelecb:
            return int(cistring.num_strings(norb, neleca)
                       * cistring.num_strings(norb, nelecb))
    else:
        nelec_tot = int(nelec)
    if nelec_tot % 2:
        return int(cistring.num_strings(norb, nelec_tot // 2 + 1)
                   * cistring.num_strings(norb, nelec_tot // 2))
    half = nelec_tot // 2
    if half < 0 or half > norb:
        return 0
    dim = int(cistring.num_strings(norb, half) ** 2)
    if 0 <= half - 1 and half + 1 <= norb:
        dim -= int(cistring.num_strings(norb, half - 1)
                   * cistring.num_strings(norb, half + 1))
    return max(1, dim)


def _ci_residual_norms(mc):
    """Return fixed-orbital FCI residual norms for the current CI roots."""
    h1, ecore = mc.get_h1eff(mc.mo_coeff)
    eri = ao2mo.restore(1, np.asarray(mc.get_h2eff(mc.mo_coeff)), mc.ncas)
    h2e = fci.direct_spin1.absorb_h1e(
        h1, eri, mc.ncas, mc.nelecas, 0.5
    )
    out = []
    for ci in mc.ci:
        ci = np.asarray(ci)
        hci = fci.direct_spin1.contract_2e(
            h2e, ci, mc.ncas, mc.nelecas
        ) + float(ecore) * ci
        energy = float(np.vdot(ci, hci))
        resid = hci - energy * ci
        out.append({
            "energy_expectation": energy,
            "residual_l2": float(np.linalg.norm(resid)),
        })
    return out


def _configure_fci_polish_solver(solver, nroots):
    solver.nroots = int(nroots)
    solver.conv_tol = FCI_POLISH_CONV_TOL
    solver.max_cycle = FCI_POLISH_MAX_CYCLE
    solver.max_space = FCI_POLISH_MAX_SPACE
    solver.pspace_size = FCI_POLISH_PSPACE_SIZE
    return solver


def _spin_square(ci, ncas, nelec):
    ss, mult = spin_op.spin_square(np.asarray(ci), ncas, nelec)
    return float(ss), float(mult)


def _select_lowest_singlets(energies_all, ci_all, scan, nroots=2):
    selected = []
    for energy, ci, row in zip(energies_all, ci_all, scan):
        if float(row["spin_square"]) <= FCI_POLISH_SPIN_TOL:
            selected.append((float(energy), np.asarray(ci), row))
            if len(selected) == int(nroots):
                break
    if len(selected) < int(nroots):
        raise RuntimeError(
            "FCI root scan did not contain enough singlet roots: "
            f"needed {int(nroots)}, found {len(selected)}, scan={scan}"
        )
    return selected


def _selected_root_clusters(selected, scan, tol):
    """Return same-energy singlet root clusters for each selected target root."""
    singlet_rows = [
        row for row in scan
        if float(row.get("spin_square", 999.0)) <= FCI_POLISH_SPIN_TOL
    ]
    clusters = []
    for target_index, (_, _, selected_row) in enumerate(selected[:2]):
        energy = float(selected_row["energy"])
        members = [
            row for row in singlet_rows
            if abs(float(row["energy"]) - energy) <= float(tol)
        ]
        clusters.append({
            "target_index": int(target_index),
            "selected_root": int(selected_row["root"]),
            "selected_energy": energy,
            "cluster_roots": [int(row["root"]) for row in members],
            "cluster_energies": [float(row["energy"]) for row in members],
            "cluster_size": int(len(members)),
            "isolated": bool(len(members) <= 1),
        })
    return clusters


def _polish_fci_roots_at_fixed_orbitals(mc):
    """Rediagonalize target-spin FCI at fixed CASSCF orbitals.

    All public singlet benchmarks use the same reference construction:
    final SA-CASSCF orbitals are frozen, the singlet active-space Hamiltonian
    is diagonalized with PySCF direct_spin0 FCI, and the lowest target roots
    define energies, CI vectors, gradients, and NACs.
    """
    before = _ci_residual_norms(mc)
    h1, ecore = mc.get_h1eff(mc.mo_coeff)
    eri = ao2mo.restore(1, np.asarray(mc.get_h2eff(mc.mo_coeff)), mc.ncas)
    solver = _configure_fci_polish_solver(
        fci.direct_spin0.FCI(), max(2, FCI_POLISH_SCAN_ROOTS)
    )
    energies_all, ci_all = solver.kernel(
        h1, eri, mc.ncas, mc.nelecas, ecore=float(ecore)
    )
    scan = []
    for i, (energy, ci) in enumerate(zip(energies_all, ci_all)):
        ss, mult = _spin_square(ci, mc.ncas, mc.nelecas)
        row = {"root": int(i), "energy": float(energy),
               "spin_square": ss, "multiplicity": mult}
        scan.append(row)
    selected = _select_lowest_singlets(energies_all, ci_all, scan, nroots=2)

    energies = [row[0] for row in selected[:2]]
    mc.ci = [row[1] for row in selected[:2]]
    _set_state_energies(mc, energies)
    mc.converged = True
    after = _ci_residual_norms(mc)
    selected_clusters = _selected_root_clusters(
        selected, scan, FCI_DEGENERACY_TOL
    )
    return {
        "mode": "spin_adapted_singlet_fci",
        "conv_tol": FCI_POLISH_CONV_TOL,
        "max_cycle": FCI_POLISH_MAX_CYCLE,
        "max_space": FCI_POLISH_MAX_SPACE,
        "pspace_size": FCI_POLISH_PSPACE_SIZE,
        "scan_roots": FCI_POLISH_SCAN_ROOTS,
        "spin_tol": FCI_POLISH_SPIN_TOL,
        "degeneracy_tol": FCI_DEGENERACY_TOL,
        "selection": "lowest_singlets_from_root_scan",
        "root_scan": scan,
        "selected_roots": [row[2] for row in selected[:2]],
        "selected_root_clusters": selected_clusters,
        "before": before,
        "after": after,
    }


# ---------------------------------------------------------------------------
# DMRG MPS → PySCF FCI ndarray  (fast path via SU2→SZ conversion)
# ---------------------------------------------------------------------------

def _su2_mps_to_fci(driver, mps, ncas, nelec, *, sz_driver,
                    sz_tag="MPSSZ"):
    """Convert an SU2-mode MPS to a PySCF FCI ndarray.

    Uses ``mps_change_to_sz`` (called on the *SU2* driver) to bring the MPS
    to the SZ basis, then ``get_csf_coefficients`` *on a separate SZ
    driver* (which in SZ mode returns determinant occupations
    [0=empty, 1=alpha, 2=beta, 3=double] with their coefficients).
    Finally rebuilds the PySCF FCI ndarray with PySCF↔block2
    fermion-ordering sign correction.

    The SZ driver MUST be supplied (block2 type system enforces strict
    SU2/SZ separation; calling ``get_csf_coefficients`` on the SU2 driver
    with an SZ-converted MPS raises a TypeError).
    """
    na, nb = int(nelec[0]), int(nelec[1])
    mps_sz = driver.mps_change_to_sz(mps, tag=sz_tag)
    csfs, coefs = sz_driver.get_csf_coefficients(
        mps_sz, cutoff=MPS_COEFF_CUTOFF, iprint=0
    )
    strs_a = list(cistring.make_strings(range(ncas), na))
    strs_b = list(cistring.make_strings(range(ncas), nb))
    a_idx = {int(s): j for j, s in enumerate(strs_a)}
    b_idx = {int(s): j for j, s in enumerate(strs_b)}
    ci = np.zeros((len(strs_a), len(strs_b)), dtype=np.float64)
    for det, c in zip(csfs, coefs):
        if abs(float(c)) < 1e-15:
            continue
        sa = sb = 0
        for site, occ in enumerate(det):
            occ = int(occ)
            if occ == 3:
                sa |= (1 << site)
                sb |= (1 << site)
            elif occ == 1:
                sa |= (1 << site)
            elif occ == 2:
                sb |= (1 << site)
        ia = a_idx.get(sa)
        ib = b_idx.get(sb)
        if ia is None or ib is None:
            continue
        sign = _pyscf_to_block2_sign(sa, sb, ncas)
        ci[ia, ib] = sign * float(c)
    return ci


# ---------------------------------------------------------------------------
# Phase-aware align: align a bond-dim-M CI vector to a reference
# ---------------------------------------------------------------------------

def _align_global_phase(ci, ci_ref):
    """Multiply ci by ±1 to match its global phase to ci_ref.

    Works for a single state. CASSCF analytic NAC depends linearly on the
    phase of each CI vector; fixing the phase at the input avoids spurious
    sign flips in the gradient/NAC error.
    """
    ovlp = float(np.vdot(np.asarray(ci_ref).ravel(), np.asarray(ci).ravel()))
    return ci if ovlp >= 0 else -ci


def _orthonormal_ci_basis(ci_roots):
    """Modified Gram-Schmidt basis for CI vectors flattened to one dimension."""
    if not ci_roots:
        return np.zeros((0, 0))
    dim = np.asarray(ci_roots[0]).size
    basis = []
    for ci in ci_roots:
        vec = np.asarray(ci, dtype=float).ravel().copy()
        for q in basis:
            vec -= q * float(np.vdot(q, vec))
        norm = float(np.linalg.norm(vec))
        if norm > 1e-10:
            basis.append(vec / norm)
    if not basis:
        return np.zeros((dim, 0))
    return np.column_stack(basis)


def _target_cluster_sets(selected_root_clusters, ntarget):
    """Normalize selected-root cluster metadata to one set per target root."""
    cluster_sets = []
    for target in range(ntarget):
        if selected_root_clusters and target < len(selected_root_clusters):
            roots = selected_root_clusters[target].get("cluster_roots", [])
            if roots:
                cluster_sets.append(set(map(int, roots)))
                continue
        cluster_sets.append({target})
    return cluster_sets


def _target_groups(selected_root_clusters, ntarget):
    """Group selected target roots that share one FCI degeneracy cluster."""
    cluster_sets = _target_cluster_sets(selected_root_clusters, ntarget)
    groups = []
    assigned = set()
    for i in range(ntarget):
        if i in assigned:
            continue
        group = {i}
        changed = True
        while changed:
            changed = False
            union = set().union(*(cluster_sets[j] for j in group))
            for j in range(ntarget):
                if j in group:
                    continue
                if union & cluster_sets[j]:
                    group.add(j)
                    changed = True
        groups.append(sorted(group))
        assigned.update(group)
    return groups, cluster_sets


def _project_group_to_candidate_subspace(ci_roots, fci_refs, group, pool):
    """Align selected FCI references to a DMRG candidate-root subspace.

    For isolated roots this reduces to ordinary root assignment.  For an exact
    or near-degenerate FCI target root, any unitary rotation within the
    degenerate subspace is a valid eigenbasis, so single raw-root overlaps are
    not a well-defined error metric.  The polar alignment below picks the
    orthonormal vectors inside the selected DMRG candidate subspace that are
    closest to the fixed FCI gauge.
    """
    q = _orthonormal_ci_basis([ci_roots[i] for i in pool])
    if q.shape[1] < len(group):
        raise RuntimeError(
            f"DMRG candidate subspace rank {q.shape[1]} is smaller than "
            f"target group size {len(group)}"
        )
    c = np.column_stack([np.asarray(fci_refs[j]).ravel() for j in group])
    projected = q @ (q.T @ c)
    subspace_overlaps = [
        float(np.linalg.norm(projected[:, col]))
        for col in range(projected.shape[1])
    ]

    # Orthogonal Procrustes/polar alignment: closest orthonormal columns in the
    # DMRG candidate subspace to the fixed FCI target gauge.
    s = q.T @ c
    u, sigma, vh = np.linalg.svd(s, full_matrices=False)
    aligned_flat = q @ (u[:, :len(group)] @ vh)

    aligned = []
    target_overlaps = []
    for col, target in enumerate(group):
        ci = aligned_flat[:, col].reshape(np.asarray(fci_refs[target]).shape)
        ci = _align_global_phase(ci, fci_refs[target])
        norm = np.linalg.norm(ci)
        if norm > 1e-30:
            ci = ci / norm
        aligned.append(ci)
        target_overlaps.append(
            float(abs(np.vdot(np.asarray(fci_refs[target]).ravel(),
                              np.asarray(ci).ravel())))
        )
    return aligned, target_overlaps, subspace_overlaps, list(map(float, sigma))


def _match_and_align_roots(ci_roots, fci_refs, selected_root_clusters=None):
    """Assign or subspace-align DMRG roots to FCI target roots.

    The SU2 DMRG solver returns roots ordered by its internal optimization.
    Near avoided crossings or compact symmetry benchmarks, that order can
    differ from the PySCF FCI root order.  If the FCI scan identifies an exact
    or near-degenerate target root, the comparison is made after projecting the
    DMRG candidate-root subspace onto the fixed FCI gauge; this is the same
    rule for every benchmark system.
    """
    ntarget = len(fci_refs)
    nraw = len(ci_roots)
    if nraw < ntarget:
        raise ValueError(
            f"Need at least {ntarget} DMRG roots, got {nraw}"
        )
    overlap = np.empty((nraw, ntarget))
    for i, ci in enumerate(ci_roots):
        ci_vec = np.asarray(ci).ravel()
        for j, ref in enumerate(fci_refs):
            overlap[i, j] = float(np.vdot(np.asarray(ref).ravel(), ci_vec))

    aligned = [None] * ntarget
    root_assignment = [None] * ntarget
    assigned_overlaps = [None] * ntarget
    diagnostics = [None] * ntarget
    available = set(range(nraw))

    groups, cluster_sets = _target_groups(selected_root_clusters, ntarget)
    degenerate_groups = []
    for group in groups:
        cluster_union = set().union(*(cluster_sets[j] for j in group))
        if len(cluster_union) > len(group) or any(
            len(cluster_sets[j]) > 1 for j in group
        ):
            degenerate_groups.append(group)
    degenerate_groups.sort(
        key=lambda g: len(set().union(*(cluster_sets[j] for j in g))),
        reverse=True,
    )
    degenerate_targets = set().union(*map(set, degenerate_groups)) if degenerate_groups else set()

    # Protect well-isolated roots before any near-degenerate subspace match.
    # Without this, a one-target degenerate group can choose a larger pool
    # that accidentally includes an already obvious isolated root.  The
    # subsequent isolated assignment then has to choose from leftover roots and
    # can report zero overlap even though the correct raw root was present.
    isolated_candidates = []
    for target in range(ntarget):
        if target in degenerate_targets or len(cluster_sets[target]) != 1:
            continue
        scores = np.sort(np.abs(overlap[:, target]))
        best_score = float(scores[-1]) if scores.size else 0.0
        second_score = float(scores[-2]) if scores.size > 1 else 0.0
        if (best_score >= ROOT_LOCK_OVERLAP
                and best_score - second_score >= ROOT_LOCK_MARGIN):
            isolated_candidates.append(target)

    if isolated_candidates:
        best_perm = None
        best_score = -1.0
        for perm in itertools.permutations(
            sorted(available), len(isolated_candidates)
        ):
            score = sum(
                abs(overlap[root, target])
                for root, target in zip(perm, isolated_candidates)
            )
            if score > best_score:
                best_score = score
                best_perm = perm
        if best_perm is not None:
            for root, target in zip(best_perm, isolated_candidates):
                target_overlap = float(abs(overlap[root, target]))
                if target_overlap < ROOT_LOCK_OVERLAP:
                    continue
                ci = _align_global_phase(ci_roots[root], fci_refs[target])
                norm = np.linalg.norm(ci)
                if norm > 1e-30:
                    ci = ci / norm
                aligned[target] = ci
                root_assignment[target] = int(root)
                assigned_overlaps[target] = target_overlap
                diagnostics[target] = {
                    "target_index": int(target),
                    "mode": "confident_isolated_root_lock",
                    "raw_roots": [int(root)],
                    "fci_cluster_roots": sorted(map(int, cluster_sets[target])),
                    "target_overlap": target_overlap,
                    "subspace_overlap": target_overlap,
                    "group_singular_values": [target_overlap],
                    "lock_overlap_threshold": float(ROOT_LOCK_OVERLAP),
                    "lock_margin_threshold": float(ROOT_LOCK_MARGIN),
                }
                available.remove(root)

    for group in degenerate_groups:
        group = [target for target in group if aligned[target] is None]
        if not group:
            continue
        cluster_union = set().union(*(cluster_sets[j] for j in group))
        cluster_size = max(len(group), len(cluster_union))
        pool_size = min(max(cluster_size, len(group)), len(available))
        if pool_size < len(group):
            raise RuntimeError(
                f"Need {len(group)} available DMRG roots for target group "
                f"{group}, got {pool_size}"
            )

        if pool_size == len(group):
            best_pool = None
            best_score = -1.0
            for pool in itertools.permutations(sorted(available), len(group)):
                score = sum(abs(overlap[root, target])
                            for root, target in zip(pool, group))
                if score > best_score:
                    best_pool = list(pool)
                    best_score = score
            pool = best_pool
        else:
            best_pool = None
            best_score = -1.0
            for pool in itertools.combinations(sorted(available), pool_size):
                q = _orthonormal_ci_basis([ci_roots[i] for i in pool])
                if q.shape[1] < len(group):
                    continue
                score = 0.0
                for target in group:
                    ref_vec = np.asarray(fci_refs[target]).ravel()
                    score += float(np.linalg.norm(q.T @ ref_vec))
                if score > best_score:
                    best_pool = list(pool)
                    best_score = score
            if best_pool is None:
                raise RuntimeError(
                    f"Could not construct DMRG subspace for target group {group}"
                )
            pool = best_pool

        aligned_group, target_ovlps, subspace_ovlps, singular_values = (
            _project_group_to_candidate_subspace(ci_roots, fci_refs, group, pool)
        )
        mode = (
            "single_root_assignment"
            if len(group) == 1 and len(pool) == 1
            else "degenerate_subspace_projection"
        )
        for local_col, target in enumerate(group):
            aligned[target] = aligned_group[local_col]
            root_assignment[target] = (
                int(pool[local_col]) if len(pool) == len(group)
                else [int(i) for i in pool]
            )
            assigned_overlaps[target] = float(target_ovlps[local_col])
            diagnostics[target] = {
                "target_index": int(target),
                "mode": mode,
                "raw_roots": (
                    [int(pool[local_col])] if len(pool) == len(group)
                    else [int(i) for i in pool]
                ),
                "fci_cluster_roots": sorted(map(int, cluster_sets[target])),
                "target_overlap": float(target_ovlps[local_col]),
                "subspace_overlap": float(subspace_ovlps[local_col]),
                "group_singular_values": singular_values,
            }
        available.difference_update(pool)

    remaining_targets = [i for i in range(ntarget) if aligned[i] is None]
    if remaining_targets:
        best_perm = None
        best_score = -1.0
        for perm in itertools.permutations(
            sorted(available), len(remaining_targets)
        ):
            score = sum(
                abs(overlap[root, target])
                for root, target in zip(perm, remaining_targets)
            )
            if score > best_score:
                best_score = score
                best_perm = perm
        if best_perm is None:
            raise RuntimeError(
                f"Could not assign isolated targets {remaining_targets}"
            )
        for root, target in zip(best_perm, remaining_targets):
            ci = _align_global_phase(ci_roots[root], fci_refs[target])
            norm = np.linalg.norm(ci)
            if norm > 1e-30:
                ci = ci / norm
            target_overlap = float(abs(overlap[root, target]))
            aligned[target] = ci
            root_assignment[target] = int(root)
            assigned_overlaps[target] = target_overlap
            diagnostics[target] = {
                "target_index": int(target),
                "mode": "single_root_assignment",
                "raw_roots": [int(root)],
                "fci_cluster_roots": sorted(map(int, cluster_sets[target])),
                "target_overlap": target_overlap,
                "subspace_overlap": target_overlap,
                "group_singular_values": [target_overlap],
            }

    if any(ci is None for ci in aligned):
        raise RuntimeError("Root matching failed to assign all target roots")

    return (
        aligned,
        root_assignment,
        overlap.tolist(),
        assigned_overlaps,
        diagnostics,
    )


# ---------------------------------------------------------------------------
# Run a single (system, M) DMRG → grad + NAC
# ---------------------------------------------------------------------------

def _redirect_block2_logs(scratch):
    """Block2 sometimes writes ATTENTION: xsyev info messages to stderr;
    when the SA-CASSCF preconditioner has a near-singular block we get a
    flood. We can't fully suppress them but at least ensure they aren't
    lost into our redirected log via stderr-mixing (handled by the caller).
    """
    return None


def setup_sacasscf_fci(mol, ncas, nelec_act):
    """SA(2)-CASSCF with PySCF FCI (M = inf reference)."""
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, ncas, nelec_act)
    mc.fix_spin_(ss=0)
    mc.fcisolver.nroots = 2
    mc.conv_tol = SA_CASSCF_CONV_TOL
    mc.conv_tol_grad = SA_CASSCF_CONV_TOL_GRAD
    mc.max_cycle_macro = SA_CASSCF_MAX_CYCLE
    mc = mc.state_average_([0.5, 0.5])
    mc.kernel()
    return mc


def _reference_has_wavefunction(ref):
    """Return True if a cached FCI reference contains the fixed gauge data."""
    return (
        isinstance(ref, dict)
        and "mo_coeff" in ref
        and "ci" in ref
        and len(ref.get("ci", [])) == 2
        and int(ref.get("schema_version", 0)) >= SCHEMA_VERSION
    )


def setup_sacasscf_from_reference(system_key, ref):
    """Build an SA-CASSCF object at the cached FCI orbital/CI gauge.

    Phase-2 derivative comparisons must use one fixed FCI-stationary orbital
    basis.  Re-running CASSCF independently for every M can converge to a
    symmetry-equivalent active-orbital gauge, especially under threaded BLAS,
    which makes derivative comparisons look like root/gauge failures even
    when the DMRG state is correct.
    """
    builder = SYSTEMS[system_key][0]
    mol, ncas, nelec_act = builder()
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, ncas, nelec_act)
    mc.fix_spin_(ss=0)
    mc.fcisolver.nroots = 2
    mc.conv_tol = SA_CASSCF_CONV_TOL
    mc.conv_tol_grad = SA_CASSCF_CONV_TOL_GRAD
    mc.max_cycle_macro = SA_CASSCF_MAX_CYCLE
    mc = mc.state_average_([0.5, 0.5])
    mc.mo_coeff = np.asarray(ref["mo_coeff"])
    mc.ci = [np.asarray(c) for c in ref["ci"]]
    _set_state_energies(mc, ref["e_states"])
    mc.converged = True
    return mc


def _set_state_energies(mc, e_states):
    """Populate PySCF state-average energy attributes used by NAC code."""
    e_states = [float(e) for e in e_states]
    try:
        mc.fcisolver.e_states = e_states
    except Exception:
        pass
    try:
        mc.e_states = e_states
    except Exception:
        pass
    try:
        weights = np.asarray(getattr(mc, "weights", [0.5, 0.5]), dtype=float)
        if len(weights) == len(e_states):
            mc.e_tot = float(np.dot(weights, e_states))
        else:
            mc.e_tot = float(np.mean(e_states))
    except Exception:
        pass


def _configure_lagrange_solver(obj):
    """Use a stricter Lagrange solve for root/gauge-sensitive NAC benchmarks."""
    obj.max_cycle = int(os.environ.get("BVOE_LAGRANGE_MAX_CYCLE", "500"))
    obj.conv_atol = float(os.environ.get("BVOE_LAGRANGE_CONV_ATOL", "1e-12"))
    obj.conv_rtol = float(os.environ.get("BVOE_LAGRANGE_CONV_RTOL", "1e-7"))
    return obj


def _lagrange_diag(obj):
    return {
        "converged": bool(getattr(obj, "converged", False)),
        "internal_converged": bool(getattr(obj, "_conv", False)),
        "max_cycle": int(getattr(obj, "max_cycle", -1)),
        "conv_atol": float(getattr(obj, "conv_atol", np.nan)),
        "conv_rtol": float(getattr(obj, "conv_rtol", np.nan)),
    }


def compute_grad_and_nac(mc, *, with_diagnostics=False):
    mc.converged = True
    grad_obj = _configure_lagrange_solver(sacasscf_grad.Gradients(mc))
    g = grad_obj.kernel(state=0)
    nac_obj = _configure_lagrange_solver(
        nac_sacasscf.NonAdiabaticCouplings(mc)
    )
    n = nac_obj.kernel(state=(0, 1))
    if with_diagnostics:
        return (
            np.asarray(g),
            np.asarray(n),
            {
                "gradient_lagrange": _lagrange_diag(grad_obj),
                "nac_lagrange": _lagrange_diag(nac_obj),
            },
        )
    return np.asarray(g), np.asarray(n)


def run_fci_reference(system_key):
    builder, label = SYSTEMS[system_key][0], SYSTEMS[system_key][1]
    mol, ncas, nelec_act = builder()
    t0 = time.time()
    mc = setup_sacasscf_fci(mol, ncas, nelec_act)
    fci_polish = _polish_fci_roots_at_fixed_orbitals(mc)
    e_states = list(map(float, mc.e_states))
    g, n, derivative_diag = compute_grad_and_nac(mc, with_diagnostics=True)
    return {
        "system": system_key, "label": label, "bond_dim": "FCI",
        "e_states": e_states,
        "grad": g.tolist(), "nac": n.tolist(),
        "derivative_diagnostics": derivative_diag,
        "fci_polish_diagnostics": fci_polish,
        "mo_coeff": np.asarray(mc.mo_coeff).tolist(),
        "ci": [np.asarray(ci).tolist() for ci in mc.ci],
        "ci_norms": [float(np.linalg.norm(ci)) for ci in mc.ci],
        "ci_shape": list(mc.ci[0].shape),
        "ncas": int(mc.ncas), "ncore": int(mc.ncore),
        "nelecas": [int(x) for x in mc.nelecas],
        "nelec_act": int(sum(mc.nelecas)),
        "schema_version": SCHEMA_VERSION,
        "runtime_s": time.time() - t0,
    }


def run_dmrg_at_fci_orbitals(system_key, bond_dim, *, fci_ref_payload=None,
                              n_sweeps=30, sweep_tol=1e-12, n_threads=1):
    """Build mc with PySCF FCI, converge, then swap CI to DMRG-truncated
    CI at the same orbitals; recompute grad + NAC."""
    builder, label = SYSTEMS[system_key][0], SYSTEMS[system_key][1]
    t0 = time.time()
    if _reference_has_wavefunction(fci_ref_payload):
        mc = setup_sacasscf_from_reference(system_key, fci_ref_payload)
        fci_ci = [np.asarray(c).copy() for c in fci_ref_payload["ci"]]
        fci_e_states = list(map(float, fci_ref_payload["e_states"]))
    else:
        mol, ncas, nelec_act = builder()
        mc = setup_sacasscf_fci(mol, ncas, nelec_act)
        fci_ci = [np.asarray(c).copy() for c in mc.ci]
        fci_e_states = list(map(float, mc.e_states))
    ncas = mc.ncas
    nelec_act = int(sum(mc.nelecas))

    # Active-space integrals at converged orbitals
    h1_act, ecore = mc.get_h1eff(mc.mo_coeff)
    eri_act = ao2mo.restore(1, np.asarray(mc.get_h2eff(mc.mo_coeff)), ncas)

    # Run SU2 DMRG with a generic root buffer at bond_dim = M.
    scratch = tempfile.mkdtemp(prefix="bvoe_p2_", dir="/tmp")
    try:
        driver = DMRGDriver(
            scratch=scratch, clean_scratch=False, stack_mem=int(2e8),
            n_threads=int(n_threads), symm_type=SymmetryTypes.SU2,
        )
        nelec_tot = int(nelec_act)
        spin = 0  # singlet
        driver.initialize_system(
            n_sites=ncas, n_elec=nelec_tot, spin=spin, orb_sym=[0] * ncas,
        )
        mpo = driver.get_qc_mpo(np.asarray(h1_act),
                                 np.asarray(eri_act),
                                 ecore=float(ecore), iprint=0)
        n_solve_roots = 2 + max(0, int(ROOT_BUFFER))
        n_solve_roots = min(n_solve_roots,
                            max(2, _singlet_csf_dim(ncas, mc.nelecas)))
        ket = driver.get_random_mps(tag=f"K_{bond_dim}", bond_dim=int(bond_dim),
                                    nroots=n_solve_roots)
        ns = max(int(n_sweeps), 30)
        bd = [int(bond_dim)] * ns
        # Aggressive noise schedule lets random-init MPS find singlet sector
        noises = ([1e-3] * 8 + [1e-4] * 8 + [1e-5] * 8
                  + [1e-6] * 4 + [0.0] * (ns - 28))
        if len(noises) < ns:
            noises = noises + [0.0] * (ns - len(noises))
        e_dmrg = driver.dmrg(
            mpo, ket, n_sweeps=ns, bond_dims=bd,
            noises=noises[:ns], thrds=[DMRG_DAV_THRD] * ns,
            tol=float(sweep_tol), iprint=0,
            dav_max_iter=DMRG_DAV_MAX_ITER,
            dav_def_max_size=DMRG_DAV_DEF_MAX_SIZE,
        )
        kets = [driver.split_mps(ket, i, f"KS_{bond_dim}_{i}")
                for i in range(n_solve_roots)]
        split_expectations = [
            float(driver.expectation(k, mpo, k, iprint=0))
            for k in kets
        ]
        refined_energies = None
        refined_expectations = None
        if REFINE_SPLIT_ROOTS and n_solve_roots > 1:
            refined_kets = []
            refined_energies = []
            refined_expectations = []
            ns_ref = max(int(REFINE_SWEEPS), 1)
            for i, k in enumerate(kets):
                mps = driver.copy_mps(k, tag=f"KSR_{bond_dim}_{i}")
                e_ref = driver.dmrg(
                    mpo,
                    mps,
                    n_sweeps=ns_ref,
                    bond_dims=[int(bond_dim)] * ns_ref,
                    noises=[0.0] * ns_ref,
                    thrds=[DMRG_DAV_THRD] * ns_ref,
                    tol=float(REFINE_SWEEP_TOL),
                    iprint=0,
                    dav_max_iter=DMRG_DAV_MAX_ITER,
                    dav_def_max_size=DMRG_DAV_DEF_MAX_SIZE,
                    proj_mpss=refined_kets or None,
                    proj_weights=(
                        [REFINE_PROJ_WEIGHT] * len(refined_kets)
                        if refined_kets else None
                    ),
                )
                refined_kets.append(mps)
                refined_energies.append(
                    float(e_ref[0] if hasattr(e_ref, "__iter__") else e_ref)
                )
                refined_expectations.append(
                    float(driver.expectation(mps, mpo, mps, iprint=0))
                )
            kets = refined_kets

        # Build SZ driver for get_csf_coefficients route
        sz_driver = DMRGDriver(
            scratch=scratch, clean_scratch=False, stack_mem=int(2e8),
            n_threads=int(n_threads), symm_type=SymmetryTypes.SZ,
        )
        sz_driver.initialize_system(
            n_sites=ncas, n_elec=nelec_tot, spin=spin, orb_sym=[0] * ncas,
        )

        # Convert each MPS to a PySCF FCI ndarray.  Keep the raw DMRG root
        # order first, then assign to FCI roots by maximum overlap below.
        ci_dmrg_raw = []
        for i, k in enumerate(kets):
            ci_i = _su2_mps_to_fci(
                driver, k, ncas, mc.nelecas,
                sz_driver=sz_driver,
                sz_tag=f"SZ_{bond_dim}_{i}",
            )
            n_i = np.linalg.norm(ci_i)
            if n_i > 1e-30:
                ci_i = ci_i / n_i
            ci_dmrg_raw.append(ci_i)

        (
            ci_dmrg,
            root_assignment,
            overlap_matrix,
            assigned_overlaps,
            alignment_diagnostics,
        ) = _match_and_align_roots(
            ci_dmrg_raw,
            fci_ci,
            selected_root_clusters=(
                fci_ref_payload
                .get("fci_polish_diagnostics", {})
                .get("selected_root_clusters", [])
                if isinstance(fci_ref_payload, dict) else []
            ),
        )

        # Energy expectation under FCI Hamiltonian (sanity check)
        e_dmrg_check = []
        for i, ci_i in enumerate(ci_dmrg):
            e_i = (fci.direct_spin1.energy(h1_act, eri_act, ci_i, ncas,
                                            mc.nelecas) + ecore)
            e_dmrg_check.append(float(e_i))
        expectation_reference = (
            refined_expectations
            if refined_expectations is not None
            else split_expectations
        )
        projection_energy_defect_mEh = []
        for i, source in enumerate(root_assignment[:len(e_dmrg_check)]):
            if isinstance(source, (int, np.integer)):
                projection_energy_defect_mEh.append(
                    float(1000.0 * (
                        e_dmrg_check[i] - expectation_reference[int(source)]
                    ))
                )
            else:
                projection_energy_defect_mEh.append(None)
    finally:
        try:
            shutil.rmtree(scratch, ignore_errors=True)
        except Exception:
            pass

    # Install DMRG-truncated CI into mc, recompute grad and NAC
    min_overlap = min(abs(float(x)) for x in assigned_overlaps[:2])
    if min_overlap < MIN_DERIVATIVE_ROOT_OVERLAP:
        raise RuntimeError(
            "root-overlap QC failed before derivative solve: "
            f"min assigned overlap {min_overlap:.6f} < "
            f"{MIN_DERIVATIVE_ROOT_OVERLAP:.6f}"
        )
    mc.ci = ci_dmrg
    _set_state_energies(mc, e_dmrg_check)
    mc.converged = True
    g, n, derivative_diag = compute_grad_and_nac(mc, with_diagnostics=True)
    return {
        "system": system_key, "label": label, "bond_dim": int(bond_dim),
        "e_states_fci": fci_e_states,
        "e_states_dmrg": e_dmrg_check,
        "e_dmrg_su2": (list(map(float, e_dmrg))
                       if hasattr(e_dmrg, "__iter__") else [float(e_dmrg)]),
        "split_expectation_energies_hartree": split_expectations,
        "refined_energies_hartree": refined_energies,
        "refined_expectation_energies_hartree": refined_expectations,
        "projection_energy_defect_mEh": projection_energy_defect_mEh,
        "root_assignment_dmrg_to_fci": root_assignment,
        "root_overlap_matrix": overlap_matrix,
        "root_assigned_abs_overlaps": assigned_overlaps,
        "root_alignment_diagnostics": alignment_diagnostics,
        "grad": g.tolist(), "nac": n.tolist(),
        "derivative_diagnostics": derivative_diag,
        "ci_residual_diagnostics": _ci_residual_norms(mc),
        "ci_overlap_with_fci": [
            float(np.vdot(ci_dmrg[i].ravel(), fci_ci[i].ravel()))
            for i in range(2)
        ],
        "ncas": int(mc.ncas), "ncore": int(mc.ncore),
        "nelec_act": int(sum(mc.nelecas)),
        "runtime_s": time.time() - t0,
    }


def _dmrg_point_worker(result_queue, system_key, bond_dim, fci_ref_payload,
                       n_sweeps, sweep_tol, n_threads):
    try:
        result_queue.put((
            "ok",
            run_dmrg_at_fci_orbitals(
                system_key,
                bond_dim,
                fci_ref_payload=fci_ref_payload,
                n_sweeps=n_sweeps,
                sweep_tol=sweep_tol,
                n_threads=n_threads,
            ),
        ))
    except Exception as exc:
        result_queue.put(("error", str(exc), traceback.format_exc()))


def run_dmrg_point_with_timeout(system_key, bond_dim, *, fci_ref_payload=None,
                                n_sweeps=30, sweep_tol=1e-12, n_threads=1):
    """Run one DMRG point in a child process so a bad low-M solve cannot
    block the whole benchmark set."""
    if POINT_TIMEOUT_S <= 0:
        return run_dmrg_at_fci_orbitals(
            system_key,
            bond_dim,
            fci_ref_payload=fci_ref_payload,
            n_sweeps=n_sweeps,
            sweep_tol=sweep_tol,
            n_threads=n_threads,
        )

    # Do not use fork here.  The parent process may already have run PySCF,
    # BLAS/OpenMP, and block2 code while building the FCI reference; forking
    # after that can inherit locked native-thread state and leave every child
    # process hanging until the timeout.  Spawn is slower but robust and keeps
    # the per-point timeout useful for general systems.
    ctx_name = os.environ.get("BVOE_MP_CONTEXT", "spawn")
    ctx = mp.get_context(ctx_name)
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_dmrg_point_worker,
        args=(
            result_queue,
            system_key,
            bond_dim,
            fci_ref_payload,
            n_sweeps,
            sweep_tol,
            n_threads,
        ),
    )
    proc.start()
    proc.join(float(POINT_TIMEOUT_S))
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(10)
        raise TimeoutError(
            f"DMRG point timed out after {POINT_TIMEOUT_S:.0f}s"
        )

    try:
        status, *payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(
            f"DMRG point exited with code {proc.exitcode} without a result"
        ) from exc
    if status == "ok":
        return payload[0]
    raise RuntimeError(payload[0] + "\n" + payload[1])


def diff_norms(arr_M, arr_ref):
    a = np.asarray(arr_M)
    b = np.asarray(arr_ref)
    diff = a - b
    return {
        "l2": float(np.linalg.norm(diff)),
        "max": float(np.max(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-30)),
    }


def phase_aware_diff(arr_M, arr_ref):
    a = np.asarray(arr_M)
    b = np.asarray(arr_ref)
    d_pos = float(np.linalg.norm(a - b))
    d_neg = float(np.linalg.norm(a + b))
    if d_neg < d_pos:
        diff = a + b
        l2 = d_neg
    else:
        diff = a - b
        l2 = d_pos
    return {
        "l2": l2,
        "max": float(np.max(np.abs(diff))),
        "rel_l2": l2 / max(np.linalg.norm(b), 1e-30),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(only=None):
    data_dir = Path(os.environ.get(
        "BVOE_DATA_DIR", str(ROOT / "data_phase2")
    )).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(os.environ.get(
        "BVOE_SUMMARY_PATH", str(ROOT / "summary_phase2.json")
    )).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}

    keys = list(SYSTEMS.keys())
    if only:
        keys = [k for k in keys if k in only]

    for sys_key in keys:
        summary.setdefault(sys_key, {})
        ref_path = data_dir / f"{sys_key}_FCI.json"
        ref = None
        force = os.environ.get("BVOE_FORCE_RECOMPUTE", "0") == "1"
        if ref_path.exists():
            try:
                ref = json.loads(ref_path.read_text())
                print(f"[{sys_key}] FCI ref: cached "
                      f"e0={ref['e_states'][0]:.8f}", flush=True)
                if force or not _reference_has_wavefunction(ref):
                    if force:
                        print(f"[{sys_key}] refreshing reference "
                              f"(BVOE_FORCE_RECOMPUTE=1)", flush=True)
                    else:
                        print(f"[{sys_key}] refreshing legacy reference "
                              f"(missing fixed orbital/CI gauge)", flush=True)
                    ref = None
            except Exception:
                ref = None
        if ref is None:
            try:
                _invalidate_system_cache(data_dir, sys_key, summary)
                ref = run_fci_reference(sys_key)
                ref_path.write_text(json.dumps(ref, indent=2))
                print(f"[{sys_key}] FCI ref: e0={ref['e_states'][0]:.8f}  "
                      f"|grad|={np.linalg.norm(ref['grad']):.4e}  "
                      f"|nac|={np.linalg.norm(ref['nac']):.4e}  "
                      f"({ref['runtime_s']:.1f}s)", flush=True)
            except Exception as e:
                print(f"[{sys_key}] FCI ref FAILED: {e}", flush=True)
                traceback.print_exc()
                summary[sys_key]["error"] = f"FCI ref failed: {e}"
                _write_summary(summary_path, summary)
                continue

        bond_dims = SYSTEMS[sys_key][2]
        for M in bond_dims:
            out_path = data_dir / f"{sys_key}_M{M}.json"
            if out_path.exists() and str(M) in summary[sys_key] \
                    and "error" not in summary[sys_key][str(M)]:
                print(f"[{sys_key}] M={M:4d}  cached", flush=True)
                continue
            try:
                res = run_dmrg_point_with_timeout(
                    sys_key, M,
                    fci_ref_payload=ref,
                    n_sweeps=int(os.environ.get("BVOE_SWEEPS", "30")),
                    sweep_tol=DMRG_SWEEP_TOL,
                    n_threads=int(os.environ.get("BVOE_THREADS", "1")),
                )
                out_path.write_text(json.dumps(res, indent=2))
                grad_d = diff_norms(res["grad"], ref["grad"])
                nac_d = phase_aware_diff(res["nac"], ref["nac"])
                e0_d = abs(res["e_states_dmrg"][0] - ref["e_states"][0])
                e1_d = abs(res["e_states_dmrg"][1] - ref["e_states"][1])
                alignment = res.get("root_alignment_diagnostics", [])
                cluster_sizes = [
                    len(item.get("fci_cluster_roots", []))
                    for item in alignment if isinstance(item, dict)
                ]
                degenerate_targets = [
                    int(item["target_index"])
                    for item in alignment
                    if isinstance(item, dict)
                    and len(item.get("fci_cluster_roots", [])) > 1
                ]
                summary[sys_key][str(M)] = {
                    "grad_l2": grad_d["l2"], "grad_max": grad_d["max"],
                    "grad_rel": grad_d["rel_l2"],
                    "nac_l2": nac_d["l2"], "nac_max": nac_d["max"],
                    "nac_rel": nac_d["rel_l2"],
                    "e0_diff": e0_d, "e1_diff": e1_d,
                    "ci0_overlap": res["ci_overlap_with_fci"][0],
                    "ci1_overlap": res["ci_overlap_with_fci"][1],
                    "root_alignment_modes": [
                        item.get("mode", "missing")
                        for item in alignment if isinstance(item, dict)
                    ],
                    "root_assignment_dmrg_to_fci": (
                        res.get("root_assignment_dmrg_to_fci")
                    ),
                    "root_assigned_abs_overlaps": (
                        res.get("root_assigned_abs_overlaps")
                    ),
                    "root_alignment_diagnostics": alignment,
                    "max_fci_cluster_size": (
                        int(max(cluster_sizes)) if cluster_sizes else 1
                    ),
                    "degenerate_target_roots": degenerate_targets,
                    "min_target_overlap": float(min(
                        abs(float(x)) for x in res["ci_overlap_with_fci"]
                    )),
                    "grad_lagrange_converged": bool(
                        res.get("derivative_diagnostics", {})
                        .get("gradient_lagrange", {})
                        .get("converged", False)
                    ),
                    "nac_lagrange_converged": bool(
                        res.get("derivative_diagnostics", {})
                        .get("nac_lagrange", {})
                        .get("converged", False)
                    ),
                    "runtime_s": res["runtime_s"],
                }
                print(f"[{sys_key}] M={M:4d}  "
                      f"de0={e0_d:.2e}  de1={e1_d:.2e}  "
                      f"|<ci|FCI>|={res['ci_overlap_with_fci'][0]:.4f},"
                      f"{res['ci_overlap_with_fci'][1]:.4f}  "
                      f"grad={grad_d['l2']:.3e}  nac={nac_d['l2']:.3e}  "
                      f"({res['runtime_s']:.1f}s)", flush=True)
            except Exception as e:
                print(f"[{sys_key}] M={M} FAILED: {e}", flush=True)
                traceback.print_exc()
                summary[sys_key][str(M)] = {"error": str(e)}
            _write_summary(summary_path, summary)

    _write_summary(summary_path, summary)
    print(f"\nWrote {summary_path}", flush=True)


def _invalidate_system_cache(data_dir, sys_key, summary):
    """Remove M-point data tied to an older reference gauge."""
    for path in data_dir.glob(f"{sys_key}_M*.json"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    summary[sys_key] = {}


def _write_summary(path, summary):
    path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    only = None
    if len(sys.argv) > 1:
        only = set(sys.argv[1:])
    main(only=only)
