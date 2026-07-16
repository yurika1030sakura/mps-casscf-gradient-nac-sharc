"""Diagnose the conical-intersection failure of the analytic SA-CASSCF gradient/NAC.

Builds twisted ethylene (rotate one CH2 about the C=C axis) to approach the S1/S0
twisted CI, scans the S1-S0 gap, and at small-gap geometries computes the state
gradients + NAC via our analytic CP path WITH diagnostics, catching exactly what
fails: CASSCF non-convergence? the CP-response near-null rotation mode (small
singular value ~ ΔE)? NAC magnitude overflow?  Goal: confirm the mechanism before
fixing, and read off the gap + offending singular value.
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parents[1] / "src" / "dmrg_analytic_dev"),
           str(_HERE.parents[1] / "sharc_interface")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import faulthandler
faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)
_T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-_T0:7.1f}s] {m}", flush=True)

import numpy as np
from pyscf import gto, scf, mcscf, fci

def twisted_ethylene(theta_deg, pyr_ang=0.0):
    """Ethylene with one CH2 rotated by theta about the C-C (x) axis, plus
    pyramidalization of C2 by pyr_ang (deg): both C2 H's tilt toward C1 along x,
    driving C2 toward sp3 (the twisted-pyramidalized S1/S0 CI coordinate)."""
    th = np.deg2rad(theta_deg); pr = np.deg2rad(pyr_ang)
    dCC = 0.6695; dx_ch = 0.5680; dy_ch = 0.9289; rCH = np.hypot(dx_ch, dy_ch)
    atoms = [["C", (dCC, 0.0, 0.0)], ["C", (-dCC, 0.0, 0.0)]]
    atoms += [["H", (dCC + dx_ch, +dy_ch, 0.0)], ["H", (dCC + dx_ch, -dy_ch, 0.0)]]
    # C2 H's: base in-plane vector (-dx_ch, +-dy_ch, 0) rel to C2, twist about x,
    # then pyramidalize by tilting toward +x (toward C1) by pr.
    for sy in (+1.0, -1.0):
        vx, vy, vz = -dx_ch, sy * dy_ch, 0.0
        # twist about x
        y = vy * np.cos(th) - vz * np.sin(th)
        z = vy * np.sin(th) + vz * np.cos(th)
        vx2, vy2, vz2 = vx, y, z
        # pyramidalize: rotate the C2-H vector toward +x by pr (mix vx with the
        # in-(y,z)-plane radial component), keeping bond length
        r_perp = np.hypot(vy2, vz2)
        vx_new = vx2 * np.cos(pr) - (-r_perp) * np.sin(pr)   # tilt toward +x
        scale = (r_perp * np.cos(pr) + vx2 * np.sin(pr)) / max(r_perp, 1e-12)
        vy_new, vz_new = vy2 * scale, vz2 * scale
        atoms.append(["H", (-dCC + vx_new, vy_new, vz_new)])
    return atoms

def sa_casscf(atoms, fix_singlet=True):
    mol = gto.M(atom=atoms, basis="6-31G*", spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    mc = mcscf.CASSCF(mf, 2, 2)
    if fix_singlet:
        mc.fcisolver = fci.addons.fix_spin_(fci.direct_spin1.FCI(), ss=0)
    mc = mc.state_average_([0.5, 0.5])
    if fix_singlet:
        mc.fcisolver = fci.addons.fix_spin_(mc.fcisolver, ss=0)
    mc.conv_tol = 1e-9
    mc.kernel()
    return mol, mf, mc

log("=== scan twist=90 + pyramidalization, S1-S0 gap (reach the CI) ===")
small_gap_geoms = []
for pr in (0, 15, 30, 45, 60, 75):
    try:
        _, _, mc = sa_casscf(twisted_ethylene(90, pr))
        gap = (mc.e_states[1] - mc.e_states[0]) * 27.2114
        log(f"  twist=90 pyr={pr:3d} deg  E0={mc.e_states[0]:.6f} E1={mc.e_states[1]:.6f}  gap={gap:.4f} eV  conv={mc.converged}")
        small_gap_geoms.append((pr, gap))
    except Exception as exc:  # noqa: BLE001
        log(f"  twist=90 pyr={pr:3d} deg  CASSCF FAILED: {type(exc).__name__}: {str(exc)[:100]}")

# test the analytic gradient + NAC at the smallest-gap geometries, catching the failure
from analytic_cp_sharc import compute_grad_nac_analytic_cp
test_prs = [pr for pr, g in sorted(small_gap_geoms, key=lambda x: x[1])[:4]]
for pr in test_prs:
    log(f"=== analytic grad+NAC at twist=90 pyr={pr} deg (near CI) ===")
    try:
        mol, mf, mc = sa_casscf(twisted_ethylene(90, pr))
        gap = (mc.e_states[1] - mc.e_states[0]) * 27.2114
        log(f"  gap={gap:.4f} eV; computing analytic grad[0],grad[1],NAC(0,1)...")
        with np.errstate(all="raise"):
            res = compute_grad_nac_analytic_cp(mc, gradient_states=[0, 1], nac_pairs=[(0, 1)],
                                               backend="newton_casscf", tol=1e-9, max_iter=500)
        g0 = np.linalg.norm(res["grad"][0]); g1 = np.linalg.norm(res["grad"][1])
        nac = np.linalg.norm(res["nac"][(0, 1)])
        log(f"  OK: |grad0|={g0:.4f} |grad1|={g1:.4f} |NAC|={nac:.4f}  (NAC ~ 1/gap expected to grow)")
    except Exception as exc:  # noqa: BLE001
        log(f"  FAILED at gap={gap:.4f} eV: {type(exc).__name__}: {str(exc)[:160]}")
        log("  traceback (where the overflow/singularity arises):")
        for line in traceback.format_exc().splitlines()[-12:]:
            log("    " + line)
log("DONE")
