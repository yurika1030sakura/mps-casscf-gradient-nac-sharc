"""Reproduce the in-trajectory ethylene crash STANDALONE via the exact warm-start.

The SHARC trajectory crashed at step 71 ('Calculator failed to converge'), but the
step-71 geometry converges fine cold/standalone in every config. The only in-traj
difference is the WARM-START orbitals from step 70 projected to step 71. This script
builds SA-dmrg-CASSCF at step 70, then warm-starts step 71 from those orbitals (the
SHARC MPSAsFCISolver settings: SZ, root_buffer=4) to reproduce the failure -- then
tests fixes (SU2, level shift, more macros) so we can iterate fast.
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, "src/dmrg_analytic_dev")
import faulthandler; faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
_T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-_T0:7.1f}s] {m}", flush=True)

from pyscf import gto, scf, mcscf
from dmrg_fcisolver import MPSAsFCISolver
import fd_validation as fdv

# step 70 (last success) and step 71 (crash) geometries, Angstrom
GEOM70 = """C 0.737737377 0 0
C -0.732766280 0 0
H 1.189861605 0.951246210 0.353316003
H 1.189861510 -0.951246298 -0.353316012
H -1.219456797 0.974853239 -0.162251086
H -1.219456612 -0.974853108 0.162250982"""
GEOM71 = """C 0.7366076 0 0
C -0.7351035 0 0
H 1.1882159 0.9479537 0.3265192
H 1.1882158 -0.9479538 -0.3265192
H -1.1971703 0.9559051 -0.1341430
H -1.1971701 -0.9559050 0.1341429"""

def build(geom, mo_guess=None, su2=False, root_buffer=4, macro=50, lshift=0.0, label=""):
    mol = gto.M(atom=geom, basis="6-31G*", spin=0, verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    cfg = dict(bond_dim=100, n_sweeps=12, n_threads=4, stack_mem_mb=2000,
               force_dmrg=True, dmrg_symm_su2=su2)
    sol = MPSAsFCISolver(mol, **cfg); sol.nroots = 2
    try: sol.root_buffer = root_buffer
    except Exception: pass
    mc = mcscf.CASSCF(mf, 2, 2); mc.fcisolver = sol
    mc = mc.state_average_([0.5, 0.5])
    mc.max_cycle_macro = macro; mc.conv_tol = 1e-8
    if lshift: mc.ah_level_shift = lshift
    t = time.perf_counter()
    mc.kernel(mo_guess)
    es = np.asarray(getattr(sol, "e_states", mc.e_states))
    log(f"  {label:38s} conv={mc.converged} e={[round(float(x),5) for x in es[:2]]} wall={time.perf_counter()-t:.0f}s")
    return mol, mc

log("build step 70 (warm-start source)...")
mol70, mc70 = build(GEOM70, su2=False, label="step70 (SZ rb4)")
mo70 = np.array(mc70.mo_coeff)

log("=== reproduce: step 71 WARM-started from step 70 orbitals (SHARC: SZ, rb4, macro=50) ===")
mol71 = gto.M(atom=GEOM71, basis="6-31G*", spin=0, verbose=0)
mo70_proj = fdv.project_mo_to_new_geometry(mol70, mol71, mo70)[0]
build(GEOM71, mo_guess=mo70_proj, su2=False, root_buffer=4, macro=50, label="step71 WARM (SZ rb4 macro50)")

log("=== candidate fixes ===")
build(GEOM71, mo_guess=mo70_proj, su2=False, root_buffer=4, macro=300, lshift=0.5, label="WARM SZ macro300 lshift0.5")
build(GEOM71, mo_guess=mo70_proj, su2=True,  root_buffer=4, macro=50,  label="WARM SU2 rb4 macro50")
build(GEOM71, mo_guess=mo70_proj, su2=True,  root_buffer=0, macro=50,  label="WARM SU2 rb0 macro50")
build(GEOM71, mo_guess=mo70_proj, su2=True,  root_buffer=0, macro=300, lshift=0.5, label="WARM SU2 rb0 macro300 lshift0.5")
log("DONE")
