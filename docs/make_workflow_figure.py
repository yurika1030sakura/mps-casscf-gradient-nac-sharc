from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

INK = "#18212B"
MUTED = "#56636E"
BLUE = "#DCECF7"
GREEN = "#DFF0D8"
GOLD = "#FFF0BE"
TEAL = "#DDF3F1"
ROSE = "#F5D7D7"
LAV = "#E9E3F4"
BG1 = "#F6F8FA"
BG2 = "#FAFAF7"


def text(ax, x, y, s, size=8, weight="normal", color=INK, ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, weight=weight,
            color=color, linespacing=1.15)


def box(ax, x, y, w, h, label, fc, size=7.4, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor="#5D6972",
        linewidth=0.9,
    )
    ax.add_patch(patch)
    text(ax, x + w / 2, y + h / 2, label, size=size, weight=weight)
    return patch


def arrow(ax, x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.1,
        color="#2F4052",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    ))


fig, ax = plt.subplots(figsize=(7.1, 3.45))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Lane backgrounds.
ax.add_patch(Rectangle((0.035, 0.58), 0.93, 0.30, color=BG1, zorder=0))
ax.add_patch(Rectangle((0.035, 0.15), 0.93, 0.28, color=BG2, zorder=0))
text(ax, 0.06, 0.85, "validation", size=7.6, weight="bold", color=MUTED,
     ha="left")
text(ax, 0.06, 0.40, "production", size=7.6, weight="bold", color=MUTED,
     ha="left")

# Inputs.
box(ax, 0.08, 0.66, 0.17, 0.12, "fixed-orbital\nFCI roots", ROSE)
box(ax, 0.08, 0.23, 0.17, 0.12, "previous-step\nroots", BLUE)

# DMRG roots.
box(ax, 0.34, 0.44, 0.20, 0.22, "block2 DMRG\nMPS roots\nbond dimension M",
    GREEN, size=7.2, weight="bold")
for i in range(5):
    x = 0.375 + 0.032 * i
    ax.plot([x, x], [0.455, 0.485], color="#3E6B48", lw=1.0)
    ax.scatter([x], [0.47], s=26, color="#7BB661", edgecolor="#2F5638",
               linewidth=0.55, zorder=3)
    if i:
        ax.plot([x - 0.032, x], [0.47, 0.47], color="#3E6B48", lw=0.9)

# Solver layer.
box(ax, 0.63, 0.38, 0.20, 0.34,
    "PySCF response\nwrapper\n\nroot / subspace\nmatch\nRDM / TDM",
    GOLD, size=7.0, weight="bold")

# Outputs.
box(ax, 0.64, 0.74, 0.18, 0.09, "BVOE errors", LAV, size=7.1)
box(ax, 0.64, 0.19, 0.18, 0.10, "gradients\nNACs", LAV, size=7.1)
box(ax, 0.84, 0.19, 0.11, 0.10, "SHARC\nQM.out", TEAL, size=6.7)

# Arrows.
arrow(ax, 0.25, 0.72, 0.34, 0.58)
arrow(ax, 0.25, 0.29, 0.34, 0.51)
arrow(ax, 0.54, 0.55, 0.63, 0.55)
arrow(ax, 0.73, 0.72, 0.73, 0.74)
arrow(ax, 0.73, 0.38, 0.73, 0.29)
arrow(ax, 0.82, 0.24, 0.84, 0.24)

# Minimal annotations.
text(ax, 0.50, 0.93, "DMRG-SA-CASSCF response workflow", size=10.0,
     weight="bold")
text(ax, 0.865, 0.12, "optimization / dynamics", size=6.2, color=MUTED)

fig.savefig(OUT / "workflow_architecture.pdf", bbox_inches="tight")
fig.savefig(OUT / "workflow_architecture.png", dpi=360, bbox_inches="tight")
