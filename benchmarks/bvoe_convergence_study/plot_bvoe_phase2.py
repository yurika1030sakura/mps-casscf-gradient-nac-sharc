"""Publication-style plots for the fixed-orbital DMRG response benchmarks.

The script reads ``summary_phase2.json`` and writes vector PDFs plus high-DPI
PNGs under ``figures/``.  The layout is tuned for ACS two-column figures:
compact lettering, embedded TrueType fonts, colorblind-safe colors, and
minimal decoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary_phase2.json"
SUMMARY = json.loads(SUMMARY_PATH.read_text())
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "legend.fontsize": 6.6,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)


# Okabe-Ito palette with one neutral gray.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
    "gray": "#6E6E6E",
}

CHEM_ACC_MEH = 0.1
OVERLAP_TOL = 0.98
BASIS_LABELS = ("STO-3G", "3-21G", "6-31G")
BASIS_KEYS = {
    "H$_4$": ("h4", "h4_321g", "h4_631g"),
    "H$_2$O": ("h2o", "h2o_321g", "h2o_631g"),
    "N$_2$": ("n2", "n2_321g", "n2_631g"),
    "C$_2$": ("c2", "c2_321g", "c2_631g"),
    "LiF": ("lif", "lif_321g", "lif_631g"),
    "ethylene": ("ethylene", "ethylene_321g", "ethylene_631g"),
    "butadiene": ("butadiene", "butadiene_321g", "butadiene_631g"),
    "formaldehyde": ("formaldehyde", "formaldehyde_321g", "formaldehyde_631g"),
    "benzene": ("benzene", "benzene_321g", "benzene_631g"),
}


def point_passes_qc(record: dict, field: str) -> bool:
    """Return whether a point is suitable for convergence plots.

    Drops points whose root-overlap quality is below ``OVERLAP_TOL``.
    The Lagrange-response convergence flags are honored if present; if
    absent (as in the schema actually written by ``run_bvoe_phase2.py``)
    the point is assumed converged — the response solve would have
    raised before the record was written otherwise.
    """
    min_overlap = abs(float(record.get(
        "min_target_overlap",
        min(abs(float(record.get("ci0_overlap", 0.0))),
            abs(float(record.get("ci1_overlap", 0.0)))),
    )))
    if min_overlap < OVERLAP_TOL:
        return False
    if field == "grad_l2":
        return bool(record.get("grad_lagrange_converged", True))
    if field == "nac_l2":
        return bool(record.get("nac_lagrange_converged", True))
    return True


def get_curve(system: str, field: str, *, qc_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted M values and the requested field for one system."""
    points = []
    for key, value in SUMMARY[system].items():
        if key.isdigit() and isinstance(value, dict) and field in value:
            if qc_only and not point_passes_qc(value, field):
                continue
            points.append((int(key), float(value[field])))
    if not points:
        return np.array([]), np.array([])
    points.sort(key=lambda item: item[0])
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def largest_m_record(system: str, *, qc_only: bool = True) -> tuple[int, dict]:
    """Return the largest-M record for a system."""
    points = [
        (int(key), value)
        for key, value in SUMMARY[system].items()
        if key.isdigit() and isinstance(value, dict) and "grad_l2" in value
        and (not qc_only or (
            point_passes_qc(value, "grad_l2")
            and point_passes_qc(value, "nac_l2")
        ))
    ]
    if not points:
        raise KeyError(system)
    return max(points, key=lambda item: item[0])


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        ".pdf": {},
        ".png": {"dpi": 600},
    }.items():
        path = FIG_DIR / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        print(f"Wrote {path}")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        weight="bold",
    )


def style_log_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color="#D8D8D8", linewidth=0.45)
    ax.grid(True, which="minor", color="#EFEFEF", linewidth=0.28)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_convergence() -> None:
    systems = [
        ("h4", "H$_4$ / STO-3G", PALETTE["black"], "o"),
        ("h2o_631g", "H$_2$O / 6-31G", PALETTE["blue"], "s"),
        ("n2_631g", "N$_2$ / 6-31G", PALETTE["green"], "^"),
        ("c2_631g", "C$_2$ / 6-31G", PALETTE["vermillion"], "D"),
        ("lif_631g", "LiF / 6-31G", PALETTE["purple"], "P"),
        ("benzene_631g", "benzene / 6-31G", PALETTE["orange"], "X"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.78), constrained_layout=False)
    fig.subplots_adjust(left=0.088, right=0.992, top=0.73, bottom=0.19, wspace=0.30)
    ax_g, ax_n = axes
    handles = []
    labels = []

    for system, label, color, marker in systems:
        if system not in SUMMARY:
            continue
        ms, grad = get_curve(system, "grad_l2")
        if ms.size:
            line = ax_g.plot(
                ms,
                np.maximum(grad * 1e3, 1e-13),
                color=color,
                marker=marker,
                markersize=4.0,
                linewidth=1.15,
                markeredgewidth=0.35,
                markeredgecolor="white",
                label=label,
            )[0]
            handles.append(line)
            labels.append(label)
        ms, nac = get_curve(system, "nac_l2")
        if ms.size:
            ax_n.plot(
                ms,
                np.maximum(nac, 1e-15),
                color=color,
                marker=marker,
                markersize=4.0,
                linewidth=1.15,
                markeredgewidth=0.35,
                markeredgecolor="white",
            )

    for ax in axes:
        style_log_axis(ax)
        ax.set_xlabel(r"DMRG bond dimension $M$")

    ax_g.set_ylabel(r"$||\nabla E_0(M)-\nabla E_0^{\rm FCI}||_2$ (mE$_h$/Bohr)")
    ax_n.set_ylabel(r"$||\mathbf{d}_{01}(M)-\mathbf{d}_{01}^{\rm FCI}||_2$ (a.u.)")
    ax_g.set_title("Gradient")
    ax_n.set_title("Derivative coupling")
    panel_label(ax_g, "A")
    panel_label(ax_n, "B")

    ax_g.axhspan(1e-13, CHEM_ACC_MEH, color="#D9EAD3", alpha=0.42, zorder=0)
    ax_g.axhline(CHEM_ACC_MEH, color="#4D7F3A", linewidth=0.75, linestyle=(0, (4, 2)))
    ax_g.text(
        0.98,
        0.10,
        r"$0.1$ mE$_h$/Bohr",
        transform=ax_g.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#3B642C",
    )
    ax_g.set_ylim(bottom=8e-13)
    ax_n.set_ylim(bottom=1e-13)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.1,
        labelspacing=0.45,
    )
    save_figure(fig, "bvoe_phase2")


def plot_root_sensitive_diagnostics() -> None:
    systems = [
        ("c2", "C$_2$ / STO-3G", PALETTE["black"], "o"),
        ("c2_321g", "C$_2$ / 3-21G", PALETTE["blue"], "s"),
        ("c2_631g", "C$_2$ / 6-31G", PALETTE["sky"], "^"),
        ("lif", "LiF / STO-3G", PALETTE["vermillion"], "D"),
        ("lif_321g", "LiF / 3-21G", PALETTE["purple"], "P"),
        ("lif_631g", "LiF / 6-31G", PALETTE["orange"], "X"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.78), constrained_layout=False)
    fig.subplots_adjust(left=0.088, right=0.992, top=0.73, bottom=0.19, wspace=0.30)
    ax_g, ax_n = axes
    handles = []
    labels = []

    for system, label, color, marker in systems:
        if system not in SUMMARY:
            continue
        ms, grad = get_curve(system, "grad_l2")
        if ms.size:
            line = ax_g.plot(
                ms,
                np.maximum(grad * 1e3, 1e-13),
                color=color,
                marker=marker,
                markersize=3.8,
                linewidth=1.05,
                markeredgewidth=0.35,
                markeredgecolor="white",
                label=label,
            )[0]
            handles.append(line)
            labels.append(label)
        ms, nac = get_curve(system, "nac_l2")
        if ms.size:
            ax_n.plot(
                ms,
                np.maximum(nac, 1e-15),
                color=color,
                marker=marker,
                markersize=3.8,
                linewidth=1.05,
                markeredgewidth=0.35,
                markeredgecolor="white",
            )

    for ax in axes:
        style_log_axis(ax)
        ax.set_xlabel(r"DMRG bond dimension $M$")
    ax_g.set_ylabel(r"gradient error (mE$_h$/Bohr)")
    ax_n.set_ylabel("NAC error (a.u.)")
    ax_g.set_title("Root-tracking benchmarks")
    ax_n.set_title("Gauge-sensitive NAC")
    panel_label(ax_g, "A")
    panel_label(ax_n, "B")
    ax_g.axhspan(1e-13, CHEM_ACC_MEH, color="#D9EAD3", alpha=0.35, zorder=0)
    ax_g.axhline(CHEM_ACC_MEH, color="#4D7F3A", linewidth=0.75, linestyle=(0, (4, 2)))

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.1,
        labelspacing=0.45,
    )
    save_figure(fig, "bvoe_phase2_diagnostics")


def high_m_matrices() -> tuple[np.ndarray, np.ndarray, list[str]]:
    grad = np.full((len(BASIS_KEYS), len(BASIS_LABELS)), np.nan)
    nac = np.full_like(grad, np.nan)
    labels = list(BASIS_KEYS)
    for row, label in enumerate(labels):
        for col, key in enumerate(BASIS_KEYS[label]):
            if key not in SUMMARY:
                continue
            try:
                _, rec = largest_m_record(key, qc_only=True)
            except KeyError:
                continue
            grad[row, col] = float(rec["grad_l2"]) * 1e3
            nac[row, col] = float(rec["nac_l2"])
    return grad, nac, labels


def log_matrix(values: np.ndarray, floor: float) -> np.ndarray:
    return np.log10(np.maximum(values, floor))


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, threshold: float | None = None) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isfinite(values[i, j]):
                ax.text(j, i, "QC", ha="center", va="center",
                        color="#777777", fontsize=5.5)
                continue
            text = f"{values[i, j]:.0e}"
            color = "white" if np.log10(max(values[i, j], 1e-15)) > -3.0 else "#222222"
            weight = "bold" if threshold is not None and values[i, j] <= threshold else "normal"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=5.7, weight=weight)


def plot_highm_heatmap() -> None:
    grad, nac, row_labels = high_m_matrices()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.1), constrained_layout=True)

    specs = [
        (
            axes[0],
            grad,
            1e-10,
            1e-1,
            r"largest-$M$ gradient error",
            r"log$_{10}$(mE$_h$/Bohr)",
            CHEM_ACC_MEH,
        ),
        (
            axes[1],
            nac,
            1e-7,
            1e-1,
            r"largest-$M$ NAC error",
            r"log$_{10}$(a.u.)",
            1e-4,
        ),
    ]

    cmap = colors.LinearSegmentedColormap.from_list(
        "acs_blue_orange",
        ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B", "#D55E00"],
    )
    cmap.set_bad("#F2F2F2")

    for idx, (ax, matrix, vmin, vmax, title, cbar_label, threshold) in enumerate(specs):
        image = ax.imshow(
            log_matrix(matrix, vmin),
            cmap=cmap,
            vmin=np.log10(vmin),
            vmax=np.log10(vmax),
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xticks(range(len(BASIS_LABELS)), BASIS_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)), row_labels if idx == 0 else [""] * len(row_labels))
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(BASIS_LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        annotate_heatmap(ax, matrix, threshold)
        panel_label(ax, "A" if idx == 0 else "B")
        cbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.015)
        cbar.set_label(cbar_label)
        cbar.outline.set_linewidth(0.4)

    save_figure(fig, "bvoe_highm_heatmap")


def plot_basis_matrix() -> None:
    """Compact supporting figure: high-M errors grouped by molecule and basis."""
    grad, nac, row_labels = high_m_matrices()
    x = np.arange(len(row_labels))
    width = 0.22
    basis_colors = [PALETTE["gray"], PALETTE["sky"], PALETTE["orange"]]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True, constrained_layout=True)
    for col, (basis, color) in enumerate(zip(BASIS_LABELS, basis_colors)):
        offset = (col - 1) * width
        axes[0].scatter(x + offset, grad[:, col], s=22, color=color, edgecolor="white", linewidth=0.35, label=basis)
        axes[1].scatter(x + offset, nac[:, col], s=22, color=color, edgecolor="white", linewidth=0.35)

    axes[0].axhspan(1e-10, CHEM_ACC_MEH, color="#D9EAD3", alpha=0.38, zorder=0)
    axes[0].axhline(CHEM_ACC_MEH, color="#4D7F3A", linewidth=0.75, linestyle=(0, (4, 2)))
    axes[0].set_ylabel(r"gradient error (mE$_h$/Bohr)")
    axes[1].set_ylabel("NAC error (a.u.)")
    axes[1].set_xticks(x, row_labels, rotation=32, ha="right")
    axes[1].set_xlabel("benchmark system")

    for ax, matrix in zip(axes, (grad, nac)):
        positives = matrix[np.isfinite(matrix) & (matrix > 0)]
        if positives.size:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.0, 1.0)
            ax.text(
                0.5,
                0.5,
                "pending",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#777777",
                fontsize=7.0,
            )
        ax.grid(True, axis="y", which="major", color="#D8D8D8", linewidth=0.45)
        ax.grid(True, axis="y", which="minor", color="#EFEFEF", linewidth=0.28)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    axes[0].legend(frameon=False, ncol=3, loc="upper right", handletextpad=0.3, columnspacing=0.9)
    save_figure(fig, "bvoe_basis_matrix")


if __name__ == "__main__":
    plot_convergence()
    plot_root_sensitive_diagnostics()
    plot_highm_heatmap()
    plot_basis_matrix()
