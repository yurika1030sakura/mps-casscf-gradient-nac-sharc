#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures" / "anthracene_convergence_revised.pdf"
M = np.array([64, 128, 256, 512])
grad = np.array([3.39, 3.68e-1, 3.56e-2, 1.57e-3])  # mEh/Bohr
nac = np.array([2.22e-2, 2.52e-3, 1.68e-4, 9.11e-6])  # a0^-1

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)

for ax, y, ylabel, panel in [
    (axes[0], grad, r"absolute gradient error (m$E_h$/Bohr)", "A"),
    (axes[1], nac, r"absolute NAC error ($a_0^{-1}$)", "B"),
]:
    ax.plot(M, y, marker="o", linewidth=1.5, markersize=5)
    ax.set_yscale("log")
    ax.set_xticks(M)
    ax.set_xlabel("DMRG bond dimension $M$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linewidth=0.45, alpha=0.45)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.20)
    ax.text(0.02, 0.97, panel, transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    ax.annotate(
        f"{y[-1]:.2e}", xy=(M[-1], y[-1]), xytext=(-6, 8),
        textcoords="offset points", ha="right", va="bottom", fontsize=7.5,
    )

axes[0].set_title("Strict MPS-Krylov response")
axes[1].set_title("Strict MPS-Krylov response")
fig.savefig(OUT, bbox_inches="tight")
print(OUT)
