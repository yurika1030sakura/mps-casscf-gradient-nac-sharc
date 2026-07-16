"""N2 excited-state gradient reference check: is the analytic state-1 gradient a BUG or an
FD-reference artifact?

Computes the N2/6-31G CAS(6,6) SA(2) state-1 nuclear gradient three ways:
  (a) analytic via the MPS-Krylov response (our implementation)
  (b) the EXACT pyscf SA-CASSCF state-specific analytic gradient (FCI solver) -- the
      independent reference
  (c) (reference already known) FD of the state-1 energy = -0.5023

If (a) == (b) but != (c): our analytic is correct; the FD reference swapped roots.
If (a) != (b): a real bug in the excited-state analytic gradient.

Hardened: faulthandler dumps the stack if any step stalls; every step flushes.
"""
import faulthandler
import sys
import time

import numpy as np

sys.path.insert(0, "src/dmrg_analytic_dev")
sys.path.insert(0, "sharc_interface")
faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
_T0 = time.perf_counter()


def log(m):
    print(f"[{time.perf_counter()-_T0:7.1f}s] {m}", flush=True)


from pyscf import gto, scf, mcscf
import fd_validation as fdv

BOHR = 1.0 / 0.52917721067
atoms = ["N", "N"]
coords = np.array([[0, 0, 0], [0, 0, 1.098 * BOHR]])

log("building N2 CAS(6,6) SA(2) DMRG (analytic-MPS path)...")
cfg = dict(fdv.DEFAULT_SOLVER_CFG)
cfg.update(bond_dim=200, n_sweeps=30, sweep_tol=1e-12)
_m, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
    atoms, coords, basis="6-31G", charge=0, spin=0, ncas=6, nelecas=6,
    nroots=2, weights=[0.5, 0.5], solver_cfg=cfg)
log(f"build done; e_states={list(np.round(np.asarray(solver.e_states),6))}")

log("(a) analytic-MPS state-1 gradient...")
g_mps = fdv.analytic_gradient(mc, 1, backend="mps-krylov", tol=1e-8, max_iter=300)
log(f"(a) analytic-MPS state1 g[1,2] = {float(g_mps[1,2]):.6f}")

log("(b) exact pyscf-FCI SA-CASSCF state-1 gradient (reference)...")
mf = scf.RHF(gto.M(atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.098)]], basis="6-31G", verbose=0))
mf.kernel()
mcf = mcscf.CASSCF(mf, 6, 6).state_average_([0.5, 0.5])
mcf.conv_tol = 1e-10
mcf.kernel()
log(f"pyscf SA-CASSCF converged={mcf.converged} e={list(np.round(mcf.e_states,6))}")
try:
    g_fci = mcf.nuc_grad_method().kernel(state=1)
    log(f"(b) pyscf-FCI state1 g[1,2] = {float(g_fci[1,2]):.6f}")
    log(f"VERDICT: analytic-MPS={float(g_mps[1,2]):.6f} pyscf-FCI={float(g_fci[1,2]):.6f} "
        f"|diff|={abs(float(g_mps[1,2])-float(g_fci[1,2])):.2e} (FD ref was -0.5023)")
except Exception as exc:  # noqa: BLE001
    log(f"(b) pyscf FCI grad FAILED: {type(exc).__name__}: {str(exc)[:200]}")
log("done")
