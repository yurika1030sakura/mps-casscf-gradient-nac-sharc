from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def box(ax, xy, wh, text, fc, ec="#333333"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color="#111111",
        linespacing=1.2,
    )


def arrow(ax, start, end, text=None, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color="#333333",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.035, text, ha="center", va="center", fontsize=7.5)


fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

box(ax, (0.04, 0.63), (0.18, 0.20), "PySCF\nSA-CASSCF\nreference", "#D8E7F5")
box(ax, (0.04, 0.18), (0.18, 0.20), "block2\nDMRG roots\nat bond dimension M", "#DDEFD8")
box(ax, (0.31, 0.40), (0.22, 0.24), "MPSAsFCISolver\nroot tracking\nRDMs / transition RDMs\nresponse operations", "#FFF1CC")
box(ax, (0.62, 0.58), (0.18, 0.20), "PySCF\nSA-CASSCF\ngradients / NACs", "#E8DDF5")
box(ax, (0.62, 0.18), (0.18, 0.20), "Validation\nFCI reference\nfixed orbitals", "#F6D7D7")
box(ax, (0.84, 0.40), (0.12, 0.20), "SHARC\nQM.out", "#D7F1EF")

arrow(ax, (0.22, 0.73), (0.31, 0.56), "orbitals,\nHamiltonian")
arrow(ax, (0.22, 0.28), (0.31, 0.48), "MPS roots")
arrow(ax, (0.53, 0.55), (0.62, 0.68), "solver API")
arrow(ax, (0.53, 0.45), (0.62, 0.30), "benchmark")
arrow(ax, (0.80, 0.68), (0.84, 0.52), "H, DM,\ngrad, NAC")

ax.text(
    0.50,
    0.92,
    "DMRG-active-space response bridge for PySCF and SHARC",
    ha="center",
    va="center",
    fontsize=11,
    weight="bold",
)

fig.savefig(OUT / "workflow_architecture.pdf", bbox_inches="tight")
fig.savefig(OUT / "workflow_architecture.png", dpi=300, bbox_inches="tight")
