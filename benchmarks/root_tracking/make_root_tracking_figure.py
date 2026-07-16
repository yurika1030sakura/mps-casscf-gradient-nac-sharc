#!/usr/bin/env python
"""Subspace-aware MPS root tracking along the LiF ionic/covalent avoided crossing.

Four panels probing whether the overlap machinery fails near
symmetry breaking / conical intersections:
  (a) SA-CASSCF state energies vs R -- the avoided crossing itself
  (b) the S0/S1 gap closing 1.91 -> 0.13 eV
  (c) the cross-geometry active-space overlap matrix elements: the DIAGONAL
      (matched) overlaps and the OFF-DIAGONAL (mixing) overlaps.  The off-diagonal
      grows as the gap closes and finally overtakes the diagonal.
  (d) the assignment margin min(diag)/max(offdiag) -- the quantity that decides
      whether a root LABEL is unambiguous -- together with the subspace sigma_min.
      The margin falls below 1 at R=5.0 A: past that point an individual adiabatic
      root label is genuinely gauge-dependent, and the health diagnostic FLAGS it
      rather than silently returning a spurious assignment.

Honest reading (this is the point of the figure): the assignment itself never
inverts (it stays diagonal at every converged point), but its MARGIN collapses as
the degeneracy is approached, and the code reports that collapse.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HARTREE_EV = 27.211386

def load(fn):
    recs = [json.loads(l) for l in open(os.path.join(HERE, "data", fn)) if l.strip()]
    pts = sorted([r for r in recs if r.get("kind") == "point"], key=lambda x: x["R_ang"])
    errs = [r for r in recs if r.get("kind") == "error"]
    return pts, errs

pts, errs = load("lif_cas66_root_tracking.jsonl")

R = np.array([p["R_ang"] for p in pts])
E0 = np.array([p["energies"][0] for p in pts])
E1 = np.array([p["energies"][1] for p in pts])
gap = np.array([p["gap_Eh"] for p in pts]) * HARTREE_EV
conv = np.array([bool(p["converged"]) for p in pts])

# overlap data (absent at the reference point)
Rm, dmin, offmax, smin, health = [], [], [], [], []
for p in pts:
    a = p.get("active_sigma_from_prev")
    if not a:
        continue
    O = np.asarray(a["O_abs"], dtype=float)
    n = O.shape[0]
    d = np.array([O[i, i] for i in range(n)])
    off = O - np.diag(np.diag(O))
    Rm.append(p["R_ang"])
    dmin.append(d.min())
    offmax.append(np.abs(off).max())
    smin.append(float(a["subspace_sigma_min"]))
    health.append(a.get("health", {}).get("overall", "?"))
Rm = np.array(Rm); dmin = np.array(dmin); offmax = np.array(offmax); smin = np.array(smin)
margin = dmin / np.maximum(offmax, 1e-12)

R_nonconv = R[~conv]
R_singular = [e.get("R_ang") for e in errs if e.get("R_ang") is not None]

fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.4))
C0, C1, CO, CM = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"

def mark_special(a, ytxt=None):
    for r in R_nonconv:
        a.axvline(r, color="#ff7f0e", ls=":", lw=1.4, zorder=0)
    for r in R_singular:
        a.axvline(r, color="k", ls="-.", lw=1.2, alpha=0.6, zorder=0)

# (a) energies
a = ax[0, 0]
a.plot(R, E0, "o-", color=C0, ms=4, label=r"$S_0$")
a.plot(R, E1, "s-", color=C1, ms=4, label=r"$S_1$")
mark_special(a)
a.set_xlabel(r"$R_{\mathrm{Li-F}}$ / $\AA$"); a.set_ylabel(r"$E$ / $E_{\mathrm{h}}$")
a.set_title("(a) SA(2)-CASSCF(6,6) avoided crossing", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=9)

# (b) gap
a = ax[0, 1]
a.semilogy(R, gap, "o-", color="#9467bd", ms=4)
mark_special(a)
a.axhline(gap.min(), color=CO, ls="--", lw=0.8)
a.annotate(f"{gap.min():.2f} eV", xy=(R[-1], gap.min()), xytext=(8, 5),
           textcoords="offset points", ha="left", fontsize=8, color=CO)
a.set_xlabel(r"$R_{\mathrm{Li-F}}$ / $\AA$"); a.set_ylabel(r"$E_1-E_0$ / eV")
a.set_title(r"(b) gap closes $1.91\rightarrow0.13$ eV", fontsize=10, loc="left")

# (c) overlap matrix elements
a = ax[1, 0]
a.semilogy(Rm, dmin, "o-", color=C0, ms=4, label=r"min diagonal $|O_{ii}|$ (matched)")
a.semilogy(Rm, offmax, "^-", color=C1, ms=4, label=r"max off-diagonal $|O_{ij}|$ (mixing)")
mark_special(a)
a.set_xlabel(r"$R_{\mathrm{Li-F}}$ / $\AA$"); a.set_ylabel("cross-geometry overlap")
a.set_title("(c) matched vs mixing overlap", fontsize=10, loc="left")
a.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="none")

# (d) assignment margin + subspace sigma_min
a = ax[1, 1]
a.semilogy(Rm, margin, "o-", color=CM, ms=4, label=r"assignment margin  $\min|O_{ii}|/\max|O_{ij}|$")
a.semilogy(Rm, smin, "s--", color="#8c564b", ms=4, label=r"subspace $\sigma_{\min}$")
a.axhline(1.0, color="k", lw=1.0)
_bbox = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9)
_tr = a.get_yaxis_transform()  # x in axes fraction, y in data coordinates
a.text(0.015, 1.0, "label unambiguous", transform=_tr, ha="left", va="bottom",
       fontsize=7, color="0.35", bbox=_bbox, zorder=6)
a.text(0.015, 1.0, "label gauge-dependent", transform=_tr, ha="left", va="top",
       fontsize=7, color="0.35", bbox=_bbox, zorder=6)
bad = margin < 1.0
if bad.any():
    a.plot(Rm[bad], margin[bad], "o", mfc="none", mec="r", ms=11, mew=1.6, zorder=5)
mark_special(a)
a.set_xlabel(r"$R_{\mathrm{Li-F}}$ / $\AA$"); a.set_ylabel("margin  /  $\\sigma_{\\min}$")
a.set_title("(d) where a root LABEL stops being meaningful", fontsize=10, loc="left")
a.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="none")

# shared legend for the vertical markers
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color="#ff7f0e", ls=":", lw=1.4,
           label=r"CASSCF non-convergence (recovered by the escalation ladder)"),
    Line2D([0], [0], color="k", ls="-.", lw=1.2, alpha=0.6,
           label=r"MO projection singular $\rightarrow$ no unique root label exists"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.045, 1, 1))
out = os.path.join(HERE, "..", "..", "methods_manuscript", "figures",
                   "lif_root_tracking.pdf")
out = os.path.normpath(out)
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
print("wrote", out)

print("\n=== numbers for the caption ===")
for r, d, o, m, s, h in zip(Rm, dmin, offmax, margin, smin, health):
    print(f"  R={r:.2f} A  diag={d:.3f} off={o:.5f} margin={m:8.2f} sub_smin={s:.3f} health={h}")
print(f"  gap: {gap.max():.2f} -> {gap.min():.2f} eV")
print(f"  CASSCF non-converged at R = {list(R_nonconv)} A (escalation ladder recovers)")
print(f"  MO projection singular at R = {R_singular} A")
print(f"  margin < 1 (label gauge-dependent) at R = {list(Rm[margin < 1.0])} A")
