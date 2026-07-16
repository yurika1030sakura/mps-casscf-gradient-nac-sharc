#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "figures" / "beyond_fci_validation.pdf"
GUARD = 5.0e7

grad = [
    ("Naph.", 6.3504e4, 4.0e-4, (7, 6)),
    ("C16", 1.656369e8, 1.0e-3, (6, 6)),
    ("aza-C20", 3.4134779536e10, 3.5e-3, (6, 6)),
]

nac = [
    ("Naph.", 6.3504e4, 1.6e-3, (7, -11)),
    ("C16", 1.656369e8, 5.1e-3, (6, 6)),
    ("tetracene", 2.3639044e9, 1.0e-2, (6, 6)),
    ("aza-C20", 3.4134779536e10, 4.5e-3, (6, 6)),
    ("C22", 4.97634306624e11, 2.8e-3, (-35, -13)),
    ("aza-C22", 4.97634306624e11, 7.3e-3, (-42, 7)),
    ("pentacene", 4.97634306624e11, 1.1e-3, (-44, -16)),
]

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), constrained_layout=True)

specs = [
    (axes[0], grad, r"gradient discrepancy ($E_h$/Bohr)", "A"),
    (axes[1], nac, r"coupling discrepancy ($a_0^{-1}$)", "B"),
]

for ax, data, ylabel, panel in specs:
    xs = [x for _, x, _, _ in data]
    ys = [y for _, _, y, _ in data]
    ax.scatter(xs, ys, s=38, zorder=3)
    for label, x, y, offset in data:
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=7.5)
    ax.axvline(GUARD, linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_S=0$ determinant-space dimension")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linewidth=0.45, alpha=0.45)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.20)
    ax.text(0.02, 0.97, panel, transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    ax.text(
        GUARD * 1.16, 0.97, r"dense-array guard $5\times10^7$",
        transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=7.2,
    )
    ax.set_xlim(3e4, 1.5e12)

axes[0].set_ylim(2.5e-4, 7e-3)
axes[1].set_ylim(7e-4, 1.8e-2)
axes[0].set_title("Gradient finite differences")
axes[1].set_title("MPS-overlap coupling finite differences")

fig.savefig(OUT)
print(OUT)
