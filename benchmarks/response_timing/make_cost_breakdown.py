#!/usr/bin/env python
"""Absolute SA-DMRG-CASSCF build and analytic-response wall times for the
linear-polyene series (native MPS-Krylov, no determinant conversion).

The point of the figure: from C16 to C24 the M_S=0 determinant-space dimension
grows by more than four orders of magnitude, while the measured wall times grow
by only about a factor of three to six, because the MPS cost is set by the bond
dimension and sweep schedule rather than by the determinant count. The
determinant dimension is therefore printed under each system so the decoupling
is visible on the plot itself, not only in the caption. No cross-system speedup
factor is inferred (the settings differ per system; see the SI).

Reproducible from data/cost_breakdown.json (values match the SI wall-time table).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data", "cost_breakdown.json")) as fh:
    rows = json.load(fh)["rows"]

systems = [r["system"] for r in rows]
det = np.array([r["det_dim"] for r in rows])
build = np.array([r["build_s"] for r in rows], float)
resp = np.array([r["response_s"] for r in rows], float)

x = np.arange(len(systems))
w = 0.38

fig, ax = plt.subplots(figsize=(7.2, 4.5))
b1 = ax.bar(x - w / 2 - 0.01, build, w, color="#1f77b4", label="SA-DMRG-CASSCF build", zorder=3)
b2 = ax.bar(x + w / 2 + 0.01, resp, w, color="#ff7f0e", label="analytic response", zorder=3)

ax.set_yscale("log")
ax.set_ylabel("wall time / s (log scale)")
ax.set_ylim(200, 1.6e4)
ax.set_axisbelow(True)
ax.grid(axis="y", color="0.85", lw=0.6, zorder=0)

# value labels above each bar
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color="0.15")


def sci(v):
    e = int(np.floor(np.log10(v)))
    m = v / 10 ** e
    return rf"${m:.2f}\times10^{{{e}}}$"


# two-line x tick labels: system name + M_S=0 determinant dimension
ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n{sci(d)}" for s, d in zip(systems, det)], fontsize=9.5)
ax.set_xlabel(r"linear-polyene active space  (with $M_S{=}0$ determinant-space dimension)",
              labelpad=8)

# headline: the decoupling, stated as ratios across the series
det_ratio = det[-1] / det[0]
r_lo = min(build[-1] / build[0], resp[-1] / resp[0])
r_hi = max(build[-1] / build[0], resp[-1] / resp[0])
de = int(np.floor(np.log10(det_ratio)))
dm = det_ratio / 10 ** de
headline = (rf"determinant space $\times{dm:.1f}\times10^{{{de}}}$ (C16$\to$C24)" "\n"
            rf"wall time only $\times${r_lo:.0f}–$\times${r_hi:.0f}")
ax.text(0.015, 0.97, headline,
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", lw=0.7))

ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.015, 0.80))

fig.tight_layout()
out = os.path.normpath(os.path.join(HERE, "..", "..", "methods_manuscript", "figures", "cost_breakdown.pdf"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
print("wrote", out)
print(f"det ratio C16->C24 = {det_ratio:.3e}; build ratio = {build[-1]/build[0]:.2f}; resp ratio = {resp[-1]/resp[0]:.2f}")
