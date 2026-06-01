#!/usr/bin/env python3
"""Anthracene pi-space CAS(14,14) DMRG gradient/NAC benchmark.

This is the large-active-space validation that matters for the methods paper:
run an SA(2)-DMRG-CASSCF orbital optimization, then evaluate analytic
SA-CASSCF gradients and the S0/S1 derivative coupling for fixed-orbital
SU2-DMRG roots at several bond dimensions.  By default the large-CAS reference
is the largest completed M value.  With ``--fci-reference``, the same fixed
orbitals are also rediagonalized with PySCF direct_spin0 FCI and the DMRG
energy/gradient/NAC errors are reported against that FCI response reference.

The benchmark deliberately reports response diagnostics and root tracking
metadata.  FCI is optional because production DMRG-SHARC calculations use
previous-step overlap, while this validation mode can use FCI when the active
space is still barely tractable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf.mcscf import avas
from pyscf.nac import sacasscf as nac_sacasscf
from pyscf.fci import cistring, spin_op


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SHARC_ROOT = PROJECT_ROOT / "sharc_pyscf_casscf"
DEV_ROOT = SHARC_ROOT / "dmrg_analytic_dev"
for _path in (SHARC_ROOT, DEV_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dmrg_sharc_bridge import DriverMultiRootDMRGCI
from site_replacement_density import _pyscf_to_block2_sign


DATA_DIR = ROOT / "data"


class CasscfNonConvergenceError(RuntimeError):
    """Carry CASSCF diagnostics back to the JSON writer on failure."""

    def __init__(self, diagnostics: dict):
        super().__init__("SA-DMRG-CASSCF did not converge")
        self.diagnostics = diagnostics


def build_anthracene_geometry() -> str:
    """Return a planar idealized anthracene geometry in Angstrom."""
    r_cc = 1.397
    r_ch = 1.09
    centers = [
        (0.0, 0.0),
        (np.sqrt(3.0) * r_cc, 0.0),
        (2.0 * np.sqrt(3.0) * r_cc, 0.0),
    ]
    carbons = []
    for cx, cy in centers:
        for k in range(6):
            theta = np.deg2rad(30.0 + 60.0 * k)
            pos = np.array([
                cx + r_cc * np.cos(theta),
                cy + r_cc * np.sin(theta),
                0.0,
            ])
            if not any(np.linalg.norm(pos - old) < 1e-5 for old in carbons):
                carbons.append(pos)
    carbons = np.asarray(carbons)
    carbons[:, 0] -= np.mean(carbons[:, 0])
    carbons[:, 1] -= np.mean(carbons[:, 1])

    neighbors = {i: [] for i in range(len(carbons))}
    for i, pi in enumerate(carbons):
        for j, pj in enumerate(carbons[:i]):
            if abs(np.linalg.norm(pi - pj) - r_cc) < 1e-3:
                neighbors[i].append(j)
                neighbors[j].append(i)

    atoms = [("C", xyz) for xyz in carbons]
    for i, carbon in enumerate(carbons):
        if len(neighbors[i]) != 2:
            continue
        inward = np.zeros(3)
        for j in neighbors[i]:
            vec = carbons[j] - carbon
            inward += vec / np.linalg.norm(vec)
        hydrogen = carbon - r_ch * inward / np.linalg.norm(inward)
        atoms.append(("H", hydrogen))

    return "\n".join(
        f"{sym:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}"
        for sym, xyz in atoms
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--m-list", default="256,512")
    parser.add_argument("--orbital-m", type=int, default=None)
    parser.add_argument("--nroots", type=int, default=2)
    parser.add_argument("--root-buffer", type=int, default=2)
    parser.add_argument("--casscf-max-cycle", type=int, default=20)
    parser.add_argument("--casscf-conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--casscf-conv-tol-grad", type=float, default=3.0e-5)
    parser.add_argument("--casscf-max-stepsize", type=float, default=None)
    parser.add_argument("--casscf-max-cycle-micro", type=int, default=None)
    parser.add_argument("--casscf-ah-level-shift", type=float, default=None)
    parser.add_argument("--casscf-ah-conv-tol", type=float, default=None)
    parser.add_argument("--casscf-ah-start-tol", type=float, default=None)
    parser.add_argument("--casscf-ah-start-cycle", type=int, default=None)
    parser.add_argument("--casscf-kf-trust-region", type=float, default=None)
    parser.add_argument("--casscf-nsteps", type=int, default=24)
    parser.add_argument("--casscf-sweep-tol", type=float, default=1.0e-7)
    parser.add_argument("--casscf-verbose", type=int, default=4)
    parser.add_argument("--allow-nonconverged-casscf", action="store_true")
    parser.add_argument("--dmrg-random-seed", type=int, default=123456)
    parser.add_argument("--eval-sweeps", type=int, default=80)
    parser.add_argument("--eval-sweep-tol", type=float, default=1.0e-8)
    parser.add_argument("--refine-split-roots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-sweeps", type=int, default=24)
    parser.add_argument("--refine-sweep-tol", type=float, default=1.0e-9)
    parser.add_argument("--refine-proj-weight", type=float, default=5.0)
    parser.add_argument("--dav-thrd", type=float, default=1.0e-12)
    parser.add_argument("--dav-max-iter", type=int, default=4000)
    parser.add_argument("--dav-def-max-size", type=int, default=80)
    parser.add_argument("--mps-coeff-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--lagrange-max-cycle", type=int, default=500)
    parser.add_argument("--lagrange-conv-atol", type=float, default=1.0e-10)
    parser.add_argument("--lagrange-conv-rtol", type=float, default=1.0e-6)
    parser.add_argument("--fci-reference", action="store_true")
    parser.add_argument("--fci-solver-roots", type=int, default=6)
    parser.add_argument("--fci-conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--fci-max-cycle", type=int, default=300)
    parser.add_argument("--fci-max-space", type=int, default=80)
    parser.add_argument("--fci-pspace-size", type=int, default=2000)
    parser.add_argument("--fci-spin-tol", type=float, default=1.0e-6)
    parser.add_argument("--reference-npz", default=None)
    parser.add_argument("--save-reference-npz", action="store_true")
    parser.add_argument("--reuse-reference-npz", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=120000)
    parser.add_argument("--stack-mem", type=float, default=2.0e9)
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--out", default=str(DATA_DIR / "anthracene_pi14_gradnac.json"))
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def parse_m_list(text: str) -> list[int]:
    out = [int(x) for x in text.replace(",", " ").split() if x.strip()]
    if not out:
        raise ValueError("m-list is empty")
    return sorted(dict.fromkeys(out))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_molecule(args: argparse.Namespace):
    mol = gto.M(
        atom=build_anthracene_geometry(),
        basis=args.basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        symmetry=False,
        verbose=0,
        max_memory=args.memory_mb,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.max_memory = args.memory_mb
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    ncas, nelecas, mo0 = avas.avas(
        mf,
        ["C 2pz"],
        threshold=0.20,
        canonicalize=True,
    )
    return mol, mf, int(ncas), int(nelecas), mo0


def configure_reference_mc(mf, ncas: int, nelecas: int | tuple[int, int],
                           args: argparse.Namespace, mo_coeff: np.ndarray,
                           casscf_diag: dict | None = None):
    """Construct a fixed-orbital SA-CASSCF object from cached orbitals."""
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = float(args.casscf_conv_tol)
    mc.conv_tol_grad = float(args.casscf_conv_tol_grad)
    mc.max_cycle_macro = int(args.casscf_max_cycle)
    mc.verbose = int(args.casscf_verbose)
    mc.chkfile = None
    mc.chk_ci = False
    mc.dump_chk = lambda *a, **k: None
    mc = mc.state_average_([1.0 / args.nroots] * args.nroots)
    mc.mo_coeff = np.asarray(mo_coeff)
    mc.converged = True
    if casscf_diag:
        e_states = casscf_diag.get("e_states") or []
        if e_states:
            _set_state_energies(mc, [float(x) for x in e_states])
        elif casscf_diag.get("e_tot") is not None:
            mc.e_tot = float(casscf_diag["e_tot"])
    return mc


def _reference_npz_path(args: argparse.Namespace) -> Path:
    if args.reference_npz:
        return Path(args.reference_npz).resolve()
    return (DATA_DIR / "anthracene_pi14_reference_cache.npz").resolve()


def save_reference_npz(path: Path, mc_ref, casscf_diag: dict,
                       fci_ci: list[np.ndarray], fci_record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mo_coeff=np.asarray(mc_ref.mo_coeff),
        fci_ci=np.stack([np.asarray(ci) for ci in fci_ci], axis=0),
        casscf_diag_json=np.asarray(json.dumps(casscf_diag)),
        fci_record_json=np.asarray(json.dumps(fci_record)),
    )


def load_reference_npz(path: Path, mf, ncas: int, nelecas: int,
                       args: argparse.Namespace):
    data = np.load(path, allow_pickle=False)
    casscf_diag = json.loads(str(data["casscf_diag_json"]))
    fci_record = json.loads(str(data["fci_record_json"]))
    mc_ref = configure_reference_mc(
        mf, ncas, nelecas, args, np.asarray(data["mo_coeff"]), casscf_diag
    )
    fci_ci = [np.asarray(ci) for ci in np.asarray(data["fci_ci"])]
    return mc_ref, casscf_diag, {"ci": fci_ci, "record": fci_record}


def run_sa_dmrg_casscf(mf, ncas: int, nelecas: int, mo0: np.ndarray,
                       args: argparse.Namespace, orbital_m: int):
    """Optimize SA(2)-DMRG-CASSCF orbitals with the MPS-native RDM solver."""
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = float(args.casscf_conv_tol)
    mc.conv_tol_grad = float(args.casscf_conv_tol_grad)
    mc.max_cycle_macro = int(args.casscf_max_cycle)
    mc.verbose = int(args.casscf_verbose)
    if args.casscf_max_stepsize is not None:
        mc.max_stepsize = float(args.casscf_max_stepsize)
    if args.casscf_max_cycle_micro is not None:
        mc.max_cycle_micro = int(args.casscf_max_cycle_micro)
    if args.casscf_ah_level_shift is not None:
        mc.ah_level_shift = float(args.casscf_ah_level_shift)
    if args.casscf_ah_conv_tol is not None:
        mc.ah_conv_tol = float(args.casscf_ah_conv_tol)
    if args.casscf_ah_start_tol is not None:
        mc.ah_start_tol = float(args.casscf_ah_start_tol)
    if args.casscf_ah_start_cycle is not None:
        mc.ah_start_cycle = int(args.casscf_ah_start_cycle)
    if args.casscf_kf_trust_region is not None:
        mc.kf_trust_region = float(args.casscf_kf_trust_region)
    mc.chkfile = None
    mc.chk_ci = False
    mc.dump_chk = lambda *a, **k: None

    solver = DriverMultiRootDMRGCI(mf)
    solver.verbose = int(args.casscf_verbose)
    start_m = max(64, min(int(orbital_m), int(round(orbital_m / 2))))
    scratch_root = Path(args.scratch_root or tempfile.gettempdir()).resolve()
    solver.dmrg_args.update({
        "startM": start_m,
        "maxM": int(orbital_m),
        "sweep_tol": float(args.casscf_sweep_tol),
        "nsteps": int(args.casscf_nsteps),
        "memory": int(args.stack_mem),
        "scratch_root": str(scratch_root / "anthracene_pi14_casscf"),
        "dav_max_iter": int(args.dav_max_iter),
        "n_threads": int(args.threads),
        "random_seed": int(args.dmrg_random_seed),
    })
    mc.fcisolver = solver
    mc = mc.state_average_([1.0 / args.nroots] * args.nroots)

    macro_history = []
    t0 = time.time()

    def _json_scalar(value):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, np.ndarray) and value.shape == ():
            return float(value)
        return None

    def _callback(envs):
        rec = {"elapsed_s": time.time() - t0}
        for key, value in envs.items():
            scalar = _json_scalar(value)
            if scalar is not None and np.isfinite(scalar):
                rec[str(key)] = scalar
        macro_history.append(rec)
        imacro = rec.get("imacro", len(macro_history))
        pieces = [f"[anth-gradnac] CASSCF macro {int(imacro)}"]
        for key in ("e_tot", "de", "norm_gorb", "norm_ddm"):
            if key in rec:
                pieces.append(f"{key}={rec[key]:.3e}")
        pieces.append(f"elapsed={rec['elapsed_s']:.1f}s")
        print(" ".join(pieces), flush=True)

    mc.callback = _callback

    mc.kernel(mo0)
    runtime = time.time() - t0
    diagnostics = {
        "orbital_m": int(orbital_m),
        "start_m": int(start_m),
        "runtime_s": runtime,
        "converged": bool(mc.converged),
        "e_tot": float(getattr(mc, "e_tot", np.nan)),
        "e_states": [
            float(x) for x in np.asarray(getattr(mc, "e_states", [])).ravel()
        ],
        "conv_tol": float(args.casscf_conv_tol),
        "conv_tol_grad": float(args.casscf_conv_tol_grad),
        "max_cycle_macro": int(args.casscf_max_cycle),
        "max_stepsize": float(mc.max_stepsize),
        "max_cycle_micro": int(mc.max_cycle_micro),
        "ah_level_shift": float(mc.ah_level_shift),
        "ah_conv_tol": float(mc.ah_conv_tol),
        "ah_start_tol": float(mc.ah_start_tol),
        "ah_start_cycle": int(mc.ah_start_cycle),
        "kf_trust_region": float(mc.kf_trust_region),
        "dmrg_random_seed": int(args.dmrg_random_seed),
        "macro_history": macro_history,
    }
    if not mc.converged:
        diagnostics["error"] = "SA-DMRG-CASSCF did not converge"
        if not bool(args.allow_nonconverged_casscf):
            raise CasscfNonConvergenceError(diagnostics)
    return mc, diagnostics


def _su2_mps_to_fci(driver, mps, ncas: int, nelec: tuple[int, int], *,
                    sz_driver, sz_tag: str, cutoff: float):
    """Convert a SU2 DMRG MPS root to a PySCF FCI ndarray."""
    na, nb = int(nelec[0]), int(nelec[1])
    mps_sz = driver.mps_change_to_sz(mps, tag=sz_tag)
    dets, coefs = sz_driver.get_csf_coefficients(
        mps_sz,
        cutoff=float(cutoff),
        iprint=0,
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
        ci[ia, ib] = _pyscf_to_block2_sign(sa, sb, ncas) * c
    norm = float(np.linalg.norm(ci))
    if norm > 1.0e-30:
        ci /= norm
    return ci, {
        "n_coefficients": int(len(coefs)),
        "norm_after_projection": norm,
        "cutoff": float(cutoff),
    }


def match_roots(ci_raw: list[np.ndarray], ref_ci: list[np.ndarray] | None,
                nroots: int):
    if ref_ci is None:
        return ci_raw[:nroots], list(range(nroots)), None, None
    nraw = len(ci_raw)
    overlap = np.empty((nraw, nroots))
    for i, ci in enumerate(ci_raw):
        for j, ref in enumerate(ref_ci[:nroots]):
            overlap[i, j] = float(np.vdot(ref.ravel(), ci.ravel()))
    best = None
    best_score = -1.0
    for perm in itertools.permutations(range(nraw), nroots):
        score = sum(abs(overlap[perm[j], j]) for j in range(nroots))
        if score > best_score:
            best = perm
            best_score = score
    aligned = []
    assigned = []
    for j, i in enumerate(best):
        ci = ci_raw[i].copy()
        if overlap[i, j] < 0:
            ci *= -1.0
        aligned.append(ci)
        assigned.append(float(abs(overlap[i, j])))
    return aligned, list(best), overlap.tolist(), assigned


def run_fixed_orbital_su2_dmrg(mc_ref, bond_dim: int, args: argparse.Namespace,
                               reference_ci: list[np.ndarray] | None,
                               reference_label: str):
    """Run SU2 DMRG at the optimized orbitals and return FCI-projected roots."""
    ncas = int(mc_ref.ncas)
    nelec = tuple(int(x) for x in mc_ref.nelecas)
    nelec_tot = int(sum(nelec))
    h1_act, ecore = mc_ref.get_h1eff(mc_ref.mo_coeff)
    eri_act = ao2mo.restore(1, np.asarray(mc_ref.get_h2eff(mc_ref.mo_coeff)), ncas)
    scratch_root = Path(args.scratch_root or tempfile.gettempdir()).resolve()
    scratch = tempfile.mkdtemp(prefix=f"anth_pi14_gradnac_M{bond_dim}_", dir=scratch_root)
    t0 = time.time()
    try:
        driver = DMRGDriver(
            scratch=scratch,
            clean_scratch=False,
            stack_mem=int(args.stack_mem),
            n_threads=int(args.threads),
            symm_type=SymmetryTypes.SU2,
        )
        driver.initialize_system(
            n_sites=ncas,
            n_elec=nelec_tot,
            spin=0,
            orb_sym=[0] * ncas,
        )
        mpo = driver.get_qc_mpo(np.asarray(h1_act), np.asarray(eri_act),
                                ecore=float(ecore), iprint=0)
        n_solve_roots = int(args.nroots) + max(0, int(args.root_buffer))
        ket = driver.get_random_mps(
            tag=f"K_M{bond_dim}",
            bond_dim=int(bond_dim),
            nroots=n_solve_roots,
        )
        nsweep = max(int(args.eval_sweeps), 30)
        noises = ([1e-3] * 8 + [1e-4] * 8 + [1e-5] * 8
                  + [1e-6] * 4 + [0.0] * max(0, nsweep - 28))
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=nsweep,
            bond_dims=[int(bond_dim)] * nsweep,
            noises=noises[:nsweep],
            thrds=[float(args.dav_thrd)] * nsweep,
            tol=float(args.eval_sweep_tol),
            dav_max_iter=int(args.dav_max_iter),
            dav_def_max_size=int(args.dav_def_max_size),
            iprint=0,
        )
        raw_energies = [float(x) for x in (
            list(energies) if hasattr(energies, "__iter__") else [energies]
        )]
        kets = [
            driver.split_mps(ket, i, f"KS_M{bond_dim}_{i}")
            for i in range(n_solve_roots)
        ]
        split_expectations = [
            float(driver.expectation(mps, mpo, mps, iprint=0))
            for mps in kets
        ]
        sz_driver = DMRGDriver(
            scratch=scratch,
            clean_scratch=False,
            stack_mem=int(args.stack_mem),
            n_threads=int(args.threads),
            symm_type=SymmetryTypes.SZ,
        )
        sz_driver.initialize_system(
            n_sites=ncas,
            n_elec=nelec_tot,
            spin=0,
            orb_sym=[0] * ncas,
        )
        ci_raw = []
        projection_diag = []
        for i, mps in enumerate(kets):
            ci_i, diag_i = _su2_mps_to_fci(
                driver,
                mps,
                ncas,
                nelec,
                sz_driver=sz_driver,
                sz_tag=f"SZ_M{bond_dim}_{i}",
                cutoff=float(args.mps_coeff_cutoff),
            )
            ci_raw.append(ci_i)
            projection_diag.append(diag_i)
        ci_sel, assignment, overlap, assigned = match_roots(
            ci_raw, reference_ci, int(args.nroots)
        )

        refined_energies = None
        refined_expectations = None
        refined_projection_diag = None
        initial_assignment = [int(x) for x in assignment]
        refined_assignment = None
        refined_overlap = None
        if bool(args.refine_split_roots):
            refined_kets = []
            refined_energies_all = []
            refined_expectations_all = []
            refined_projection_diag_all = []
            refined_ci = []
            for target, source_root in enumerate(assignment):
                mps = driver.copy_mps(
                    kets[int(source_root)],
                    tag=f"KSR_M{bond_dim}_{target}",
                )
                nsweep_ref = max(int(args.refine_sweeps), 1)
                e_refine = driver.dmrg(
                    mpo,
                    mps,
                    n_sweeps=nsweep_ref,
                    bond_dims=[int(bond_dim)] * nsweep_ref,
                    noises=[0.0] * nsweep_ref,
                    thrds=[float(args.dav_thrd)] * nsweep_ref,
                    tol=float(args.refine_sweep_tol),
                    dav_max_iter=int(args.dav_max_iter),
                    dav_def_max_size=int(args.dav_def_max_size),
                    proj_mpss=refined_kets or None,
                    proj_weights=(
                        [float(args.refine_proj_weight)] * len(refined_kets)
                        if refined_kets else None
                    ),
                    iprint=0,
                )
                refined_energies_all.append(
                    float(e_refine[0] if hasattr(e_refine, "__iter__") else e_refine)
                )
                refined_expectations_all.append(
                    float(driver.expectation(mps, mpo, mps, iprint=0))
                )
                ci_i, diag_i = _su2_mps_to_fci(
                    driver,
                    mps,
                    ncas,
                    nelec,
                    sz_driver=sz_driver,
                    sz_tag=f"SZR_M{bond_dim}_{target}",
                    cutoff=float(args.mps_coeff_cutoff),
                )
                if reference_ci is not None:
                    ov = float(np.vdot(reference_ci[target].ravel(), ci_i.ravel()))
                    if ov < 0:
                        ci_i *= -1.0
                refined_ci.append(ci_i)
                refined_projection_diag_all.append(diag_i)
                refined_kets.append(mps)

            if reference_ci is not None:
                ci_sel, refined_assignment, refined_overlap, assigned = match_roots(
                    refined_ci,
                    reference_ci,
                    int(args.nroots),
                )
                refined_order = [int(i) for i in refined_assignment]
            else:
                ci_sel = refined_ci[:int(args.nroots)]
                refined_order = list(range(int(args.nroots)))
            assignment = [initial_assignment[i] for i in refined_order]
            refined_energies = [refined_energies_all[i] for i in refined_order]
            refined_expectations = [refined_expectations_all[i] for i in refined_order]
            refined_projection_diag = [
                refined_projection_diag_all[i] for i in refined_order
            ]
            if refined_overlap is not None:
                overlap = refined_overlap

        e_check = [
            float(fci.direct_spin1.energy(h1_act, eri_act, ci, ncas, nelec) + ecore)
            for ci in ci_sel
        ]
        expectation_reference = (
            refined_expectations
            if refined_expectations is not None
            else [split_expectations[int(i)] for i in assignment]
        )
        projection_energy_defect_mEh = [
            float(1000.0 * (e_check[i] - expectation_reference[i]))
            for i in range(min(len(e_check), len(expectation_reference)))
        ]
        return {
            "ci": ci_sel,
            "record": {
                "M": int(bond_dim),
                "raw_energies_hartree": raw_energies,
                "split_expectation_energies_hartree": split_expectations,
                "refined_energies_hartree": refined_energies,
                "refined_expectation_energies_hartree": refined_expectations,
                "projected_energies_hartree": e_check,
                "projection_energy_defect_mEh": projection_energy_defect_mEh,
                "root_tracking_reference": reference_label,
                "initial_root_assignment_before_refinement": initial_assignment,
                "refined_root_assignment_within_selected": refined_assignment,
                "root_assignment": assignment,
                "root_overlap_matrix": overlap,
                "root_assigned_abs_overlaps": assigned,
                "root_overlap_matrix_vs_previous": overlap,
                "root_assigned_abs_overlaps_vs_previous": assigned,
                "mps_projection": projection_diag,
                "refined_mps_projection": refined_projection_diag,
                "runtime_dmrg_projection_s": time.time() - t0,
            },
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _set_state_energies(mc, e_states: list[float]) -> None:
    try:
        mc.e_states = [float(x) for x in e_states]
    except Exception:
        pass
    try:
        mc.fcisolver.e_states = [float(x) for x in e_states]
    except Exception:
        pass
    weights = np.asarray(getattr(mc, "weights", [0.5, 0.5]), dtype=float)
    mc.e_tot = float(np.dot(weights, np.asarray(e_states, dtype=float)))


def build_eval_mc(mc_ref, ci_roots: list[np.ndarray], e_states: list[float]):
    mf = mc_ref._scf
    mc = mcscf.CASSCF(mf, mc_ref.ncas, mc_ref.nelecas)
    mc.fix_spin_(ss=0)
    mc.fcisolver.nroots = len(ci_roots)
    mc.conv_tol = mc_ref.conv_tol
    mc.conv_tol_grad = mc_ref.conv_tol_grad
    mc.max_cycle_macro = mc_ref.max_cycle_macro
    mc = mc.state_average_([1.0 / len(ci_roots)] * len(ci_roots))
    mc.mo_coeff = np.asarray(mc_ref.mo_coeff)
    mc.ci = [np.asarray(ci) for ci in ci_roots]
    _set_state_energies(mc, e_states)
    mc.converged = True
    return mc


def _configure_lagrange(obj, args: argparse.Namespace):
    obj.max_cycle = int(args.lagrange_max_cycle)
    obj.conv_atol = float(args.lagrange_conv_atol)
    obj.conv_rtol = float(args.lagrange_conv_rtol)
    return obj


def _lagrange_diag(obj) -> dict:
    return {
        "converged": bool(getattr(obj, "converged", False)),
        "internal_converged": bool(getattr(obj, "_conv", False)),
        "max_cycle": int(getattr(obj, "max_cycle", -1)),
        "conv_atol": float(getattr(obj, "conv_atol", np.nan)),
        "conv_rtol": float(getattr(obj, "conv_rtol", np.nan)),
    }


def compute_gradients_and_nac(mc, args: argparse.Namespace):
    gradients = []
    grad_diag = []
    for state in range(len(mc.ci)):
        grad_obj = _configure_lagrange(sacasscf_grad.Gradients(mc), args)
        grad = np.asarray(grad_obj.kernel(state=state))
        gradients.append(grad)
        grad_diag.append(_lagrange_diag(grad_obj))
    nac_obj = _configure_lagrange(nac_sacasscf.NonAdiabaticCouplings(mc), args)
    nac = np.asarray(nac_obj.kernel(state=(0, 1)))
    return gradients, nac, {
        "gradient_lagrange": grad_diag,
        "nac_lagrange": _lagrange_diag(nac_obj),
    }


def _as_list(value, expected_len: int | None = None):
    if isinstance(value, (list, tuple)):
        return list(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [arr.item()]
    if expected_len is not None and arr.shape[0] == int(expected_len):
        return [arr[i] for i in range(int(expected_len))]
    if arr.ndim == 1:
        return [arr[i] for i in range(arr.shape[0])]
    return [value]


def _spin_square(ci, ncas: int, nelec: tuple[int, int]) -> tuple[float, float]:
    ss, mult = spin_op.spin_square(np.asarray(ci), int(ncas), nelec)
    return float(ss), float(mult)


def _ci_residual_norms(h1, eri, ecore, ncas, nelec, ci_roots):
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, ncas, nelec, 0.5)
    out = []
    for ci in ci_roots:
        ci = np.asarray(ci)
        hci = fci.direct_spin1.contract_2e(h2e, ci, ncas, nelec)
        hci = hci + float(ecore) * ci
        energy = float(np.vdot(ci, hci))
        resid = hci - energy * ci
        out.append({
            "energy_expectation": energy,
            "residual_l2": float(np.linalg.norm(resid)),
        })
    return out


def run_fixed_orbital_fci_response(mc_ref, args: argparse.Namespace):
    """Optional validation-only FCI response reference at fixed DMRG orbitals."""
    ncas = int(mc_ref.ncas)
    nelec = tuple(int(x) for x in mc_ref.nelecas)
    h1_act, ecore = mc_ref.get_h1eff(mc_ref.mo_coeff)
    eri_act = ao2mo.restore(
        1, np.asarray(mc_ref.get_h2eff(mc_ref.mo_coeff)), ncas
    )
    solver = fci.direct_spin0.FCI()
    solver.nroots = max(int(args.nroots), int(args.fci_solver_roots))
    solver.conv_tol = float(args.fci_conv_tol)
    solver.max_cycle = int(args.fci_max_cycle)
    solver.max_space = int(args.fci_max_space)
    solver.pspace_size = int(args.fci_pspace_size)
    solver.max_memory = int(args.memory_mb)

    t0 = time.time()
    energies, ci_roots = solver.kernel(
        np.asarray(h1_act),
        np.asarray(eri_act),
        ncas,
        nelec,
        ecore=float(ecore),
    )
    e_all = [float(x) for x in _as_list(energies, solver.nroots)]
    ci_all = [np.asarray(ci) for ci in _as_list(ci_roots, solver.nroots)]
    root_scan = []
    for i, (energy, ci) in enumerate(zip(e_all, ci_all)):
        ss, mult = _spin_square(ci, ncas, nelec)
        root_scan.append({
            "root": int(i),
            "energy": float(energy),
            "spin_square": ss,
            "multiplicity": mult,
        })

    selected = []
    for energy, ci, row in zip(e_all, ci_all, root_scan):
        if float(row["spin_square"]) <= float(args.fci_spin_tol):
            selected.append((float(energy), np.asarray(ci), row))
            if len(selected) == int(args.nroots):
                break
    if len(selected) < int(args.nroots):
        raise RuntimeError(
            "FCI root scan did not contain enough singlet roots: "
            f"needed {int(args.nroots)}, found {len(selected)}, "
            f"scan={root_scan}"
        )

    e_sel = [row[0] for row in selected]
    ci_sel = [row[1] for row in selected]
    mc_fci = build_eval_mc(mc_ref, ci_sel, e_sel)
    gradients, nac, derivative_diag = compute_gradients_and_nac(mc_fci, args)
    return {
        "ci": ci_sel,
        "record": {
            "mode": "validation_only_fixed_orbital_direct_spin0_fci",
            "energies_hartree": e_sel,
            "root_scan": root_scan,
            "selected_roots": [row[2] for row in selected],
            "residual_diagnostics": _ci_residual_norms(
                np.asarray(h1_act), np.asarray(eri_act), float(ecore),
                ncas, nelec, ci_sel
            ),
            "gradient_norms_hartree_per_bohr": [
                float(np.linalg.norm(g)) for g in gradients
            ],
            "gradients_hartree_per_bohr": [g.tolist() for g in gradients],
            "nac_01_au": nac.tolist(),
            "nac_norm_au": float(np.linalg.norm(nac)),
            "derivative_diagnostics": derivative_diag,
            "settings": {
                "nroots": int(args.nroots),
                "solver_roots": int(args.fci_solver_roots),
                "conv_tol": float(args.fci_conv_tol),
                "max_cycle": int(args.fci_max_cycle),
                "max_space": int(args.fci_max_space),
                "pspace_size": int(args.fci_pspace_size),
                "spin_tol": float(args.fci_spin_tol),
            },
            "runtime_s": time.time() - t0,
        },
    }


def phase_aware_l2(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def add_largest_m_errors(results: list[dict]) -> None:
    if not results:
        return
    ref = results[-1]
    ref_g = [np.asarray(g) for g in ref["gradients_hartree_per_bohr"]]
    ref_n = np.asarray(ref["nac_01_au"])
    ref_e = np.asarray(ref["projected_energies_hartree"])
    for rec in results:
        g = [np.asarray(x) for x in rec["gradients_hartree_per_bohr"]]
        n = np.asarray(rec["nac_01_au"])
        e = np.asarray(rec["projected_energies_hartree"])
        rec["delta_vs_largest_completed_M"] = {
            "reference_M": int(ref["M"]),
            "energy_mEh": [
                float(1000.0 * (e[i] - ref_e[i]))
                for i in range(min(len(e), len(ref_e)))
            ],
            "gradient_l2_per_state_mEh_per_bohr": [
                float(np.linalg.norm(g[i] - ref_g[i]) * 1000.0)
                for i in range(min(len(g), len(ref_g)))
            ],
            "nac_l2_phase_aware_au": phase_aware_l2(n, ref_n),
        }


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(args.scratch_root or tempfile.gettempdir()).resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)

    m_list = parse_m_list(args.m_list)
    orbital_m = int(args.orbital_m or max(m_list))

    mol, mf, ncas, nelecas, mo0 = build_molecule(args)
    fci_dim = int(
        cistring.num_strings(ncas, nelecas // 2)
        * cistring.num_strings(ncas, nelecas - nelecas // 2)
    )
    output = {
        "schema_version": 1,
        "benchmark": "anthracene_pi14_dmrg_sacasscf_gradient_nac",
        "system": "anthracene",
        "basis": args.basis,
        "active_space": [int(nelecas), int(ncas)],
        "fci_dimension_det": fci_dim,
        "nroots": int(args.nroots),
        "m_list": m_list,
        "orbital_m": orbital_m,
        "workflow": (
            "SA(2)-DMRG-CASSCF orbitals followed by fixed-orbital SU2-DMRG "
            "root projection and PySCF analytic SA-CASSCF gradient/NAC."
        ),
        "settings": vars(args),
        "rhf_energy_hartree": float(mf.e_tot),
        "preview_only": bool(args.preview_only),
        "casscf_reference": None,
        "fci_reference": None,
        "results": [],
    }
    write_json(out_path, output)
    print("Anthracene CAS(14,14) DMRG gradient/NAC benchmark")
    print(f"basis={args.basis} active=CAS({nelecas},{ncas}) fci_dim={fci_dim}")
    print(f"m_list={m_list} orbital_m={orbital_m} output={out_path}", flush=True)

    if args.preview_only:
        return 0

    fci_reference_ci = None
    fci_payload = None
    reference_npz = _reference_npz_path(args)
    if args.reuse_reference_npz:
        print(
            f"[anth-gradnac] loading cached CASSCF/FCI reference {reference_npz}",
            flush=True,
        )
        mc_ref, casscf_diag, fci_payload = load_reference_npz(
            reference_npz, mf, ncas, nelecas, args
        )
        output["casscf_reference"] = casscf_diag
        output["fci_reference"] = fci_payload["record"]
        fci_reference_ci = [ci.copy() for ci in fci_payload["ci"]]
        write_json(out_path, output)
        print(
            "[anth-gradnac] cached reference loaded "
            f"E={casscf_diag.get('e_tot', float('nan')):.10f}",
            flush=True,
        )
    else:
        print("[anth-gradnac] SA-DMRG-CASSCF orbital optimization start", flush=True)
        try:
            mc_ref, casscf_diag = run_sa_dmrg_casscf(
                mf, ncas, nelecas, mo0, args, orbital_m
            )
        except CasscfNonConvergenceError as err:
            output["casscf_reference"] = err.diagnostics
            output["error"] = {
                "stage": "SA-DMRG-CASSCF orbital optimization",
                "message": str(err),
            }
            write_json(out_path, output)
            raise
        output["casscf_reference"] = casscf_diag
        write_json(out_path, output)
        print(
            "[anth-gradnac] SA-DMRG-CASSCF done "
            f"E={casscf_diag['e_tot']:.10f} runtime={casscf_diag['runtime_s']:.1f}s",
            flush=True,
        )

        if args.fci_reference:
            print("[anth-gradnac] validation-only FCI response start", flush=True)
            fci_payload = run_fixed_orbital_fci_response(mc_ref, args)
            fci_reference_ci = [ci.copy() for ci in fci_payload["ci"]]
            output["fci_reference"] = fci_payload["record"]
            write_json(out_path, output)
            print(
                "[anth-gradnac] validation-only FCI response done "
                f"|g0|={output['fci_reference']['gradient_norms_hartree_per_bohr'][0]:.3e} "
                f"|nac01|={output['fci_reference']['nac_norm_au']:.3e} "
                f"runtime={output['fci_reference']['runtime_s']:.1f}s",
                flush=True,
            )
            if args.save_reference_npz:
                save_reference_npz(
                    reference_npz,
                    mc_ref,
                    casscf_diag,
                    fci_reference_ci,
                    output["fci_reference"],
                )
                print(
                    f"[anth-gradnac] saved cached CASSCF/FCI reference {reference_npz}",
                    flush=True,
                )

    previous_ci = None
    for bond_dim in m_list:
        print(f"[anth-gradnac] M={bond_dim} fixed-orbital DMRG start", flush=True)
        if fci_reference_ci is not None:
            reference_ci = fci_reference_ci
            reference_label = "validation_fci"
        else:
            reference_ci = previous_ci
            reference_label = "energy_order" if previous_ci is None else "previous_M"
        payload = run_fixed_orbital_su2_dmrg(
            mc_ref, bond_dim, args, reference_ci, reference_label
        )
        previous_ci = [ci.copy() for ci in payload["ci"]]
        rec = dict(payload["record"])
        rec["stage"] = "dmrg_projection_complete"
        output["results"].append(rec)
        add_largest_m_errors(output["results"])
        write_json(out_path, output)
        print(
            f"[anth-gradnac] M={bond_dim} DMRG projection done "
            f"min_overlap={min(rec['root_assigned_abs_overlaps']):.6f} "
            f"max_proj_defect_mEh="
            f"{max(abs(x) for x in rec['projection_energy_defect_mEh']):.3e}",
            flush=True,
        )
        mc_eval = build_eval_mc(
            mc_ref,
            payload["ci"],
            payload["record"]["projected_energies_hartree"],
        )
        print(f"[anth-gradnac] M={bond_dim} analytic grad/NAC start", flush=True)
        t0 = time.time()
        gradients, nac, derivative_diag = compute_gradients_and_nac(mc_eval, args)
        rec["stage"] = "derivatives_complete"
        rec.update({
            "gradients_hartree_per_bohr": [g.tolist() for g in gradients],
            "gradient_norms_hartree_per_bohr": [
                float(np.linalg.norm(g)) for g in gradients
            ],
            "nac_01_au": nac.tolist(),
            "nac_norm_au": float(np.linalg.norm(nac)),
            "derivative_diagnostics": derivative_diag,
            "runtime_derivatives_s": time.time() - t0,
        })
        if output["fci_reference"] is not None:
            fci_ref = output["fci_reference"]
            e = np.asarray(rec["projected_energies_hartree"])
            e_ref = np.asarray(fci_ref["energies_hartree"])
            g_ref = [
                np.asarray(x) for x in fci_ref["gradients_hartree_per_bohr"]
            ]
            n_ref = np.asarray(fci_ref["nac_01_au"])
            rec["delta_vs_fci_reference"] = {
                "energy_mEh": [
                    float(1000.0 * (e[i] - e_ref[i]))
                    for i in range(min(len(e), len(e_ref)))
                ],
                "gradient_l2_per_state_mEh_per_bohr": [
                    float(np.linalg.norm(gradients[i] - g_ref[i]) * 1000.0)
                    for i in range(min(len(gradients), len(g_ref)))
                ],
                "nac_l2_phase_aware_au": phase_aware_l2(
                    nac, n_ref
                ),
                "min_root_overlap_vs_fci": (
                    float(min(payload["record"]["root_assigned_abs_overlaps"]))
                    if payload["record"]["root_assigned_abs_overlaps"] else None
                ),
            }
        output["results"][-1] = rec
        add_largest_m_errors(output["results"])
        write_json(out_path, output)
        print(
            f"[anth-gradnac] M={bond_dim} "
            f"|g0|={rec['gradient_norms_hartree_per_bohr'][0]:.3e} "
            f"|nac01|={rec['nac_norm_au']:.3e} "
            f"runtime_deriv={rec['runtime_derivatives_s']:.1f}s",
            flush=True,
        )
    print("[anth-gradnac] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
