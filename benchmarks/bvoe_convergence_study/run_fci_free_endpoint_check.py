#!/usr/bin/env python3
"""FCI-free root-selection endpoint check for public benchmark systems.

This script tests a separate claim from the FCI-reference convergence curves:
the DMRG calculation can select roots and evaluate gradients/NACs without
using FCI CI vectors at runtime.  FCI is loaded only as an offline validation
reference after the DMRG endpoint result exists.

The orbitals are the fixed validation orbitals from ``data_phase2/*_FCI.json``;
that keeps the endpoint directly comparable to the BVOE convergence curves.
The root-selection policy, however, is FCI-free: energy order for the first
endpoint, or previous accepted DMRG roots for a sequence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import ao2mo, fci

from run_bvoe_phase2 import (
    DMRG_DAV_DEF_MAX_SIZE,
    DMRG_DAV_MAX_ITER,
    DMRG_DAV_THRD,
    MPS_COEFF_CUTOFF,
    REFINE_PROJ_WEIGHT,
    REFINE_SPLIT_ROOTS,
    REFINE_SWEEP_TOL,
    REFINE_SWEEPS,
    ROOT,
    SYSTEMS,
    _match_and_align_roots,
    _set_state_energies,
    _singlet_csf_dim,
    _su2_mps_to_fci,
    compute_grad_and_nac,
    diff_norms,
    phase_aware_diff,
    setup_sacasscf_from_reference,
)


DEFAULT_SYSTEMS = [
    "c2",
    "c2_321g",
    "c2_631g",
    "lif",
    "n2",
    "ethylene",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        default=",".join(DEFAULT_SYSTEMS),
        help="Comma/space-separated benchmark system keys.",
    )
    parser.add_argument(
        "--default-m",
        type=int,
        default=200,
        help="Default endpoint bond dimension.",
    )
    parser.add_argument(
        "--c2-m",
        type=int,
        default=400,
        help="Endpoint bond dimension for C2 variants.",
    )
    parser.add_argument("--root-buffer", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-14)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stack-mem", type=float, default=2.0e8)
    parser.add_argument(
        "--out",
        default=str(ROOT / "data_fci_free_endpoint" / "endpoint_check.json"),
    )
    return parser.parse_args()


def parse_systems(text: str) -> list[str]:
    systems = [item for item in text.replace(",", " ").split() if item]
    unknown = [item for item in systems if item not in SYSTEMS]
    if unknown:
        raise ValueError(f"Unknown systems: {unknown}")
    return systems


def endpoint_m(system_key: str, args: argparse.Namespace) -> int:
    return int(args.c2_m if system_key.startswith("c2") else args.default_m)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_reference(system_key: str) -> dict:
    path = ROOT / "data_phase2" / f"{system_key}_FCI.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing validation reference {path}; run BVOE refresh first."
        )
    return json.loads(path.read_text())


def select_roots_without_fci(
    ci_raw: list[np.ndarray],
    *,
    nroots: int,
    previous_ci: list[np.ndarray] | None,
) -> tuple[list[np.ndarray], list[int], list[list[float]] | None,
           list[float] | None, str]:
    """Select roots by energy order or previous-DMRG overlap, never FCI."""
    if previous_ci is None:
        selected = []
        for ci in ci_raw[:nroots]:
            ci = np.asarray(ci).copy()
            norm = np.linalg.norm(ci)
            if norm > 1.0e-30:
                ci /= norm
            selected.append(ci)
        return selected, list(range(nroots)), None, None, "energy_order"

    overlap = np.empty((len(ci_raw), nroots))
    for i, ci in enumerate(ci_raw):
        ci_vec = np.asarray(ci).ravel()
        for j, ref in enumerate(previous_ci[:nroots]):
            overlap[i, j] = float(np.vdot(np.asarray(ref).ravel(), ci_vec))
    best = None
    best_score = -1.0
    import itertools

    for perm in itertools.permutations(range(len(ci_raw)), nroots):
        score = sum(abs(overlap[perm[j], j]) for j in range(nroots))
        if score > best_score:
            best = perm
            best_score = score
    if best is None:
        raise RuntimeError("Could not assign roots by previous-DMRG overlap")

    selected = []
    assigned = []
    for j, i in enumerate(best):
        ci = np.asarray(ci_raw[i]).copy()
        if overlap[i, j] < 0:
            ci *= -1.0
        norm = np.linalg.norm(ci)
        if norm > 1.0e-30:
            ci /= norm
        selected.append(ci)
        assigned.append(float(abs(overlap[i, j])))
    return selected, list(best), overlap.tolist(), assigned, "previous_dmrg_overlap"


def run_dmrg_candidates(mc, bond_dim: int, args: argparse.Namespace):
    ncas = int(mc.ncas)
    nelec = tuple(int(x) for x in mc.nelecas)
    nelec_tot = int(sum(nelec))
    h1_act, ecore = mc.get_h1eff(mc.mo_coeff)
    eri_act = ao2mo.restore(1, np.asarray(mc.get_h2eff(mc.mo_coeff)), ncas)
    n_solve_roots = 2 + max(0, int(args.root_buffer))
    n_solve_roots = min(n_solve_roots, max(2, _singlet_csf_dim(ncas, nelec)))

    scratch = tempfile.mkdtemp(prefix="fci_free_ep_", dir="/tmp")
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
        mpo = driver.get_qc_mpo(
            np.asarray(h1_act), np.asarray(eri_act),
            ecore=float(ecore), iprint=0,
        )
        ket = driver.get_random_mps(
            tag=f"K_EP_{bond_dim}",
            bond_dim=int(bond_dim),
            nroots=n_solve_roots,
        )
        nsweep = max(int(args.sweeps), 30)
        noises = ([1.0e-3] * 8 + [1.0e-4] * 8 + [1.0e-5] * 8
                  + [1.0e-6] * 4 + [0.0] * max(0, nsweep - 28))
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=nsweep,
            bond_dims=[int(bond_dim)] * nsweep,
            noises=noises[:nsweep],
            thrds=[DMRG_DAV_THRD] * nsweep,
            tol=float(args.sweep_tol),
            iprint=0,
            dav_max_iter=DMRG_DAV_MAX_ITER,
            dav_def_max_size=DMRG_DAV_DEF_MAX_SIZE,
        )
        raw_energies = (
            list(map(float, energies))
            if hasattr(energies, "__iter__") else [float(energies)]
        )
        kets = [
            driver.split_mps(ket, i, f"KS_EP_{bond_dim}_{i}")
            for i in range(n_solve_roots)
        ]
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
                mps = driver.copy_mps(k, tag=f"KSR_EP_{bond_dim}_{i}")
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
        for i, ket_i in enumerate(kets):
            ci = _su2_mps_to_fci(
                driver,
                ket_i,
                ncas,
                nelec,
                sz_driver=sz_driver,
                sz_tag=f"SZ_EP_{bond_dim}_{i}",
            )
            norm = np.linalg.norm(ci)
            if norm > 1.0e-30:
                ci = ci / norm
            ci_raw.append(ci)
        return {
            "ci_raw": ci_raw,
            "raw_energies_hartree": raw_energies,
            "split_expectation_energies_hartree": split_expectations,
            "refined_energies_hartree": refined_energies,
            "refined_expectation_energies_hartree": refined_expectations,
            "runtime_s": time.time() - t0,
            "n_solve_roots": int(n_solve_roots),
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def evaluate_ci(mc_template, ci_roots: list[np.ndarray]) -> tuple[list[float], np.ndarray, np.ndarray, dict]:
    h1_act, ecore = mc_template.get_h1eff(mc_template.mo_coeff)
    eri_act = ao2mo.restore(
        1, np.asarray(mc_template.get_h2eff(mc_template.mo_coeff)),
        mc_template.ncas,
    )
    e_states = [
        float(fci.direct_spin1.energy(
            h1_act, eri_act, ci, mc_template.ncas, mc_template.nelecas
        ) + ecore)
        for ci in ci_roots
    ]
    mc_template.ci = [np.asarray(ci) for ci in ci_roots]
    _set_state_energies(mc_template, e_states)
    mc_template.converged = True
    grad, nac, diag = compute_grad_and_nac(mc_template, with_diagnostics=True)
    return e_states, grad, nac, diag


def compare_to_fci(e_states, grad, nac, fci_ref: dict) -> dict:
    return {
        "energy_mEh": [
            float(1000.0 * (float(e_states[i]) - float(fci_ref["e_states"][i])))
            for i in range(min(len(e_states), len(fci_ref["e_states"])))
        ],
        "gradient_l2_mEh_per_bohr": float(
            diff_norms(grad, fci_ref["grad"])["l2"] * 1000.0
        ),
        "nac_l2_phase_aware_au": float(
            phase_aware_diff(nac, fci_ref["nac"])["l2"]
        ),
    }


def run_system(system_key: str, args: argparse.Namespace) -> dict:
    fci_ref = load_reference(system_key)
    bond_dim = endpoint_m(system_key, args)
    mc = setup_sacasscf_from_reference(system_key, fci_ref)
    candidates = run_dmrg_candidates(mc, bond_dim, args)
    ci_runtime, assignment, prev_overlap, prev_assigned, policy = (
        select_roots_without_fci(
            candidates["ci_raw"],
            nroots=2,
            previous_ci=None,
        )
    )
    e_runtime, grad_runtime, nac_runtime, diag_runtime = evaluate_ci(
        mc, ci_runtime
    )

    fci_ci = [np.asarray(ci) for ci in fci_ref["ci"]]
    (
        ci_aligned,
        offline_assignment,
        offline_overlap,
        offline_assigned,
        offline_diag,
    ) = _match_and_align_roots(
        candidates["ci_raw"],
        fci_ci,
        selected_root_clusters=(
            fci_ref.get("fci_polish_diagnostics", {})
            .get("selected_root_clusters", [])
        ),
    )
    mc_aligned = setup_sacasscf_from_reference(system_key, fci_ref)
    e_aligned, grad_aligned, nac_aligned, diag_aligned = evaluate_ci(
        mc_aligned, ci_aligned
    )

    return {
        "system": system_key,
        "label": SYSTEMS[system_key][1],
        "bond_dim": int(bond_dim),
        "n_solve_roots": candidates["n_solve_roots"],
        "root_selection_policy": policy,
        "fci_ci_used_for_runtime_root_selection": False,
        "validation_orbitals": "fixed_orbitals_from_fci_reference_benchmark",
        "raw_energies_hartree": candidates["raw_energies_hartree"],
        "split_expectation_energies_hartree": (
            candidates["split_expectation_energies_hartree"]
        ),
        "refined_energies_hartree": candidates["refined_energies_hartree"],
        "refined_expectation_energies_hartree": (
            candidates["refined_expectation_energies_hartree"]
        ),
        "runtime_root_assignment": assignment,
        "runtime_overlap_matrix_vs_previous": prev_overlap,
        "runtime_assigned_abs_overlaps_vs_previous": prev_assigned,
        "runtime_e_states_hartree": e_runtime,
        "runtime_derivative_diagnostics": diag_runtime,
        "runtime_error_vs_fci_by_energy_order": compare_to_fci(
            e_runtime, grad_runtime, nac_runtime, fci_ref
        ),
        "offline_validation": {
            "description": (
                "FCI loaded after the DMRG endpoint to diagnose the best "
                "FCI-gauge comparison; not used by runtime root selection."
            ),
            "root_assignment_dmrg_to_fci": offline_assignment,
            "root_overlap_matrix": offline_overlap,
            "root_assigned_abs_overlaps": offline_assigned,
            "root_alignment_diagnostics": offline_diag,
            "e_states_hartree": e_aligned,
            "derivative_diagnostics": diag_aligned,
            "error_vs_fci_after_offline_alignment": compare_to_fci(
                e_aligned, grad_aligned, nac_aligned, fci_ref
            ),
        },
        "dmrg_candidate_runtime_s": candidates["runtime_s"],
    }


def main() -> int:
    args = parse_args()
    systems = parse_systems(args.systems)
    out_path = Path(args.out).resolve()
    output = {
        "schema_version": 1,
        "purpose": (
            "Endpoint check that DMRG root selection and derivative "
            "evaluation do not use FCI CI vectors at runtime. FCI is an "
            "offline validation reference only."
        ),
        "systems": systems,
        "settings": vars(args),
        "results": [],
    }
    write_json(out_path, output)
    print("FCI-free endpoint check")
    print(f"systems={systems}")
    print(f"output={out_path}", flush=True)

    for system in systems:
        print(f"[fci-free] {system} start", flush=True)
        try:
            record = run_system(system, args)
            output["results"].append(record)
            write_json(out_path, output)
            rt_err = record["runtime_error_vs_fci_by_energy_order"]
            off_err = record["offline_validation"][
                "error_vs_fci_after_offline_alignment"
            ]
            print(
                f"[fci-free] {system} M={record['bond_dim']} "
                f"runtime_grad={rt_err['gradient_l2_mEh_per_bohr']:.3e} mEh/Bohr "
                f"offline_grad={off_err['gradient_l2_mEh_per_bohr']:.3e} mEh/Bohr "
                f"offline_nac={off_err['nac_l2_phase_aware_au']:.3e}",
                flush=True,
            )
        except Exception as exc:
            output["results"].append({
                "system": system,
                "error": str(exc),
            })
            write_json(out_path, output)
            print(f"[fci-free] {system} FAILED: {exc}", flush=True)
    print("[fci-free] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
