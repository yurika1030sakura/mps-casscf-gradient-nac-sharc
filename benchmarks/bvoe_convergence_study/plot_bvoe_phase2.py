"""Plot BVOE convergence — Phase 2 (REAL DMRG).

Two-panel publication figure:
  Left:  ||grad(M) − grad_FCI||_2 vs M
  Right: ||NAC(M)  − NAC_FCI ||_2 vs M  (phase-aware diff)

Reads `summary_phase2.json`. Produces `figures/bvoe_phase2.png` and `.pdf`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary_phase2.json"
SUMMARY = json.loads(SUMMARY_PATH.read_text())

# Display order, labels, color, marker
SYSTEM_PLOT = {
    "h2o":      ("H$_2$O / STO-3G CAS(4,4)",       "tab:orange", "s"),
    "h4":       ("H$_4$ / STO-3G CAS(4,4) R=1.5 a$_0$",
                 "tab:green",  "^"),
    "n2":       ("N$_2$ / STO-3G CAS(6,6) R=1.4 Å",
                 "tab:red",    "D"),
    "c2":       ("C$_2$ / STO-3G CAS(8,8) R=1.25 Å",
                 "tab:blue",   "o"),
    "lif":      ("LiF / STO-3G CAS(4,4) R=6.5 a$_0$",
                 "tab:brown",  "P"),
    "h2o_631g": ("H$_2$O / 6-31G CAS(6,6)",        "tab:purple", "v"),
}


def get_curve(d, key):
    Ms, vals = [], []
    for k, v in d.items():
        if not k.isdigit() or "error" in v:
            continue
        Ms.append(int(k))
        vals.append(v[key])
    if not Ms:
        return np.array([]), np.array([])
    order = np.argsort(Ms)
    return np.array(Ms)[order], np.array(vals)[order]


fig, (ax_g, ax_n) = plt.subplots(1, 2, figsize=(10.5, 4.8))

CHEM_ACC_THR = 0.1  # mE_h/Bohr  (≈1.6e-4 a.u.)
legend_handles = []
legend_labels = []

for sys_key, (label, color, marker) in SYSTEM_PLOT.items():
    if sys_key not in SUMMARY:
        continue
    d = SUMMARY[sys_key]
    if not any(k.isdigit() for k in d):
        continue

    Ms, gvals = get_curve(d, "grad_l2")
    if Ms.size:
        gvals_mEh = np.maximum(gvals * 1e3, 1e-12)
        line, = ax_g.plot(Ms, gvals_mEh, color=color, marker=marker,
                          label=label, linewidth=1.7, markersize=6)
        legend_handles.append(line)
        legend_labels.append(label)

    Ms, nvals = get_curve(d, "nac_l2")
    if Ms.size:
        ax_n.plot(Ms, np.maximum(nvals, 1e-15), color=color, marker=marker,
                  label=label, linewidth=1.7, markersize=6)

for ax, title, ylabel in [
    (ax_g, "(a) Analytic gradient error",
     r"$\|\nabla E_0(M) - \nabla E_0^{\rm FCI}\|_2$  (mE$_h$/Bohr)"),
    (ax_n, r"(b) Analytic NAC error  (states 0,1)",
     r"$\|\mathbf{d}_{01}(M) - \mathbf{d}_{01}^{\rm FCI}\|_2$  (a.u.)"),
]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"DMRG bond dimension $M$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

# Chemical accuracy reference
ax_g.axhline(CHEM_ACC_THR, color="gray", linestyle="--", linewidth=0.8,
             alpha=0.7)
xmin, xmax = ax_g.get_xlim()
ax_g.text(xmax * 0.95, CHEM_ACC_THR * 1.2,
          "0.1 mE$_h$/Bohr (chem. accuracy)",
          fontsize=7, color="gray", ha="right", va="bottom")

fig.suptitle(
    r"BVOE phase 2: real DMRG (block2 SU2) at FCI-converged orbitals",
    fontsize=11, y=0.96,
)

fig.legend(
    legend_handles, legend_labels,
    loc="lower center", ncol=3, fontsize=7.5, frameon=True,
    bbox_to_anchor=(0.5, 0.02),
)
fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.28,
                    wspace=0.30)

fig_dir = ROOT / "figures"
fig_dir.mkdir(exist_ok=True)
out_png = fig_dir / "bvoe_phase2.png"
out_pdf = fig_dir / "bvoe_phase2.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")
