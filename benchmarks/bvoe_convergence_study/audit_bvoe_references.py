"""Audit fixed-orbital FCI references and DMRG response pathologies.

Run from ``bvoe_convergence_study`` after ``run_bvoe_phase2.py``.  The report
is intended for manuscript/supporting-information QC: it checks that every
reference uses the same spin-adapted singlet FCI protocol, summarizes the
largest-M error against that reference, and classifies molecule-independent
failure modes that require an adaptive rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def largest_m_record(summary_entry: dict) -> tuple[int | None, dict | None]:
    points = [
        (int(key), value)
        for key, value in summary_entry.items()
        if key.isdigit() and isinstance(value, dict) and "grad_l2" in value
    ]
    if not points:
        return None, None
    return max(points, key=lambda item: item[0])


def min_target_overlap(record: dict) -> float:
    return abs(float(record.get(
        "min_target_overlap",
        min(abs(float(record.get("ci0_overlap", 0.0))),
            abs(float(record.get("ci1_overlap", 0.0)))),
    )))


def root_modes(record: dict) -> list[str]:
    modes = record.get("root_alignment_modes", [])
    if isinstance(modes, list):
        return [str(mode) for mode in modes]
    return []


def classify_point(
    record: dict,
    *,
    overlap_tol: float,
    grad_tol: float,
) -> list[str]:
    """Classify a benchmark point without using molecule-specific rules."""
    if "error" in record:
        return ["runtime/error"]
    if "grad_l2" not in record:
        return ["missing derivative data"]

    labels = []
    overlap = min_target_overlap(record)
    if overlap < overlap_tol:
        labels.append("root/subspace overlap")
    if not bool(record.get("grad_lagrange_converged", False)):
        labels.append("gradient response")
    if not bool(record.get("nac_lagrange_converged", False)):
        labels.append("NAC response")
    if float(record.get("grad_l2", 0.0)) > grad_tol:
        labels.append("gradient target")
    if int(record.get("max_fci_cluster_size", 1)) > 1:
        labels.append("near-degenerate FCI cluster")
    if "degenerate_subspace_projection" in root_modes(record):
        labels.append("subspace-aligned root")
    return labels or ["OK"]


def recommended_action(labels: list[str]) -> str:
    label_set = set(labels)
    if "runtime/error" in label_set:
        return "inspect error, then rerun with stricter convergence"
    if "root/subspace overlap" in label_set:
        return "increase root buffer/M/sweeps; inspect state characters"
    if "gradient response" in label_set or "NAC response" in label_set:
        return "tighten CASSCF/DMRG convergence and response tolerances"
    if "gradient target" in label_set:
        return "increase M and verify monotonic high-M convergence"
    if "near-degenerate FCI cluster" in label_set:
        return "report subspace alignment; avoid raw-root label claims"
    return "no action"


def qc_bad_points(
    summary_entry: dict,
    *,
    overlap_tol: float,
    grad_tol: float,
) -> list[str]:
    bad = []
    for key, value in sorted(
        summary_entry.items(),
        key=lambda item: int(item[0]) if item[0].isdigit() else -1,
    ):
        if not key.isdigit() or not isinstance(value, dict):
            continue
        if "error" in value:
            bad.append(f"M={key}: error")
            continue
        if "grad_l2" not in value:
            continue
        labels = classify_point(
            value, overlap_tol=overlap_tol, grad_tol=grad_tol
        )
        reasons = [label for label in labels if label not in {
            "OK", "near-degenerate FCI cluster", "subspace-aligned root"
        }]
        if reasons:
            bad.append(f"M={key}: " + ", ".join(reasons))
    return bad


def fmt_float(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}e}"


def make_report(root: Path, *, overlap_tol: float, grad_tol: float) -> str:
    data_dir = root / "data_phase2"
    summary_path = root / "summary_phase2.json"
    summary = load_json(summary_path) if summary_path.exists() else {}

    lines = [
        "# BVOE Reference Protocol Audit",
        "",
        "Reference protocol used for validation:",
        "",
        "1. RHF is converged first.",
        "2. Equal-weight SA(2)-CASSCF is converged with the PySCF FCI solver.",
        "3. The converged orbitals are held fixed.",
        "4. The singlet active-space Hamiltonian is rediagonalized with PySCF direct_spin0 FCI.",
        "5. The two lowest singlet roots are used; S^2 and residuals are recorded as QC diagnostics.",
        "6. DMRG roots are converted to PySCF CI vectors. High-confidence isolated roots are locked first; roots in a same-energy FCI cluster are aligned by a general candidate-subspace projection before derivative errors are evaluated.",
        "",
        "The table reports the largest-M DMRG response error against that fixed-orbital FCI reference.",
        "",
        "| system | schema | FCI mode | FCI root clusters | max S^2 | max FCI residual | largest M | grad error (mEh/Bohr) | NAC error | min overlap | QC |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    fci_files = sorted(data_dir.glob("*_FCI.json"))
    for path in fci_files:
        system = path.stem.removesuffix("_FCI")
        ref = load_json(path)
        polish = ref.get("fci_polish_diagnostics", {})
        selected = polish.get("selected_roots", [])
        clusters = polish.get("selected_root_clusters", [])
        cluster_label = ", ".join(
            "[" + ",".join(str(x) for x in row.get("cluster_roots", [])) + "]"
            for row in clusters
        ) or "missing"
        max_s2 = max((float(row.get("spin_square", 999.0)) for row in selected), default=None)
        residuals = [
            float(row.get("residual_l2", 999.0))
            for row in polish.get("after", [])
            if isinstance(row, dict)
        ]
        max_resid = max(residuals, default=None)
        schema = int(ref.get("schema_version", 0))
        mode = str(polish.get("mode", "missing"))

        m, rec = largest_m_record(summary.get(system, {}))
        if rec is None:
            grad_meh = nac = min_overlap = None
            qc = "missing M scan"
        else:
            grad_meh = float(rec["grad_l2"]) * 1e3
            nac = float(rec["nac_l2"])
            min_overlap = min_target_overlap(rec)
            qc_flags = []
            if schema < 5:
                qc_flags.append("legacy schema")
            if mode != "spin_adapted_singlet_fci":
                qc_flags.append("reference mode")
            if max_s2 is None or max_s2 > 1e-6:
                qc_flags.append("S^2")
            if max_resid is None or max_resid > 1e-8:
                qc_flags.append("FCI residual")
            if min_overlap < overlap_tol:
                qc_flags.append("largest-M overlap")
            if float(rec.get("grad_l2", 0.0)) > grad_tol:
                qc_flags.append("gradient target")
            if not bool(rec.get("grad_lagrange_converged", False)):
                qc_flags.append("grad Lagrange")
            if not bool(rec.get("nac_lagrange_converged", False)):
                qc_flags.append("NAC Lagrange")
            qc = "OK" if not qc_flags else ", ".join(qc_flags)

        lines.append(
            "| {system} | {schema} | {mode} | {clusters} | {max_s2} | {max_resid} | {m} | {grad} | {nac} | {overlap} | {qc} |".format(
                system=system,
                schema=schema,
                mode=mode,
                clusters=cluster_label,
                max_s2=fmt_float(max_s2),
                max_resid=fmt_float(max_resid),
                m=m if m is not None else "n/a",
                grad=fmt_float(grad_meh),
                nac=fmt_float(nac),
                overlap=fmt_float(min_overlap, precision=4),
                qc=qc,
            )
        )

    lines.extend([
        "",
        "Low-M points excluded from accuracy claims by QC:",
        "",
    ])
    for system in sorted(summary):
        bad = qc_bad_points(
            summary[system], overlap_tol=overlap_tol, grad_tol=grad_tol
        )
        if bad:
            lines.append(f"- `{system}`: " + "; ".join(bad))
    if lines[-1] == "":
        lines.append("- none")

    lines.extend([
        "",
        "Molecule-independent pathology classifier:",
        "",
        "| system | M | labels | recommended action |",
        "|---|---:|---|---|",
    ])
    any_pathology = False
    for system in sorted(summary):
        for key, value in sorted(
            summary[system].items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else -1,
        ):
            if not key.isdigit() or not isinstance(value, dict):
                continue
            labels = classify_point(
                value, overlap_tol=overlap_tol, grad_tol=grad_tol
            )
            actionable = [
                label for label in labels
                if label not in {"OK", "near-degenerate FCI cluster",
                                 "subspace-aligned root"}
            ]
            if not actionable:
                continue
            any_pathology = True
            lines.append(
                f"| {system} | {key} | {', '.join(labels)} | "
                f"{recommended_action(labels)} |"
            )
    if not any_pathology:
        lines.append("| none | n/a | OK | no action |")

    lines.extend([
        "",
        "Adaptive rerun ladder:",
        "",
        "1. If root/subspace overlap fails, increase the candidate-root buffer before changing molecule-specific settings.",
        "2. If response convergence fails, tighten CASSCF/DMRG/response tolerances and rerun the same benchmark point.",
        "3. If only the gradient target fails while overlap and response are clean, increase bond dimension and verify high-M convergence.",
        "4. If a near-degenerate cluster is present, compare subspaces and avoid interpreting raw root labels as chemically unique.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overlap-tol", type=float, default=0.98)
    parser.add_argument("--grad-tol", type=float, default=1e-4)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output or (root / "REFERENCE_PROTOCOL_AUDIT.md")
    report = make_report(
        root, overlap_tol=args.overlap_tol, grad_tol=args.grad_tol
    )
    output.write_text(report)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
