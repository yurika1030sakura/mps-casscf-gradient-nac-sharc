"""Non-singlet (DOUBLET) analytic gradient vs exact FCI: completes the spin-generality
demonstration alongside the triplet CH2 test (open-shell spin-generality check). Methyl
radical CH3 (2A2'', planar D3h), full-valence CAS(7,7)/6-31G*, SA(2) over the two lowest
doublets, DMRG(SU2, S=1/2) analytic state-0 gradient vs FCI(direct_spin1, spin=1)."""
import sys, numpy as np, time, tempfile
sys.path.insert(0, "sharc_interface")
sys.path.insert(0, "src/dmrg_analytic_dev")
from pyscf import gto, scf, mcscf, fci
from dmrg_fcisolver import MPSAsFCISolver
from analytic_cp_sharc import compute_grad_nac_analytic_cp
t0 = time.time(); L = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
# methyl radical CH3, planar D3h, C-H 1.079 A, doublet (spin=1 -> 2S=1, S=1/2)
mol = gto.M(atom="C 0 0 0; H 1.079 0 0; H -0.5395 0.9345 0; H -0.5395 -0.9345 0",
            basis="6-31G*", spin=1, verbose=0)
mf = scf.ROHF(mol).run(conv_tol=1e-10)
# FCI ground truth (direct_spin1, doublet), SA(2), full-valence CAS(7,7)
mcf = mcscf.CASSCF(mf, 7, 7); mcf.fcisolver = fci.direct_spin1.FCI(); mcf.fcisolver.spin = 1
mcf = mcf.state_average([0.5, 0.5]); mcf.conv_tol = 1e-9; mcf.kernel()
gf = mcf.nuc_grad_method().kernel(state=0)
L(f"FCI doublet CAS(7,7): e_states={[round(float(x),6) for x in mcf.e_states[:2]]} |g0|={np.linalg.norm(gf):.6f}")
# DMRG (SU2, doublet sector) warm-started from FCI orbitals (same basin) for a fair check
scr = tempfile.mkdtemp(prefix="nsd_", dir="/tmp/dmrg_scratch")
sol = MPSAsFCISolver(mol, bond_dim=200, n_sweeps=16, n_threads=8, sweep_tol=1e-10, scratch_root=scr,
                     force_dmrg=True, max_fci_dets=100000, root_buffer=4, refine_split_roots=True,
                     dmrg_symm_su2=True, skip_kernel_fci_conversion=True, mps_native_rdms=True)
sol.fix_spin_(ss=0.75)   # S(S+1)=0.75 for S=1/2 doublet
mc = mcscf.CASSCF(mf, 7, 7); mc.fcisolver = sol; mc = mc.state_average([0.5, 0.5]); mc.conv_tol = 1e-8
mc.kernel(mcf.mo_coeff)
L(f"DMRG doublet: e_states={[round(float(x),6) for x in np.asarray(mc.e_states)[:2]]} (FCI {[round(float(x),6) for x in mcf.e_states[:2]]})")
r = compute_grad_nac_analytic_cp(mc, gradient_states=[0], nac_pairs=[], backend="mps-krylov", tol=1e-4, max_iter=50)
gd = np.asarray(r["grad"][0])
L(f"DMRG doublet analytic grad0: |g|={np.linalg.norm(gd):.6f}  vs FCI max|dg|={np.abs(gd-gf).max():.2e}")
print(">>> VERDICT:", "SU2 doublet gradient = exact FCI -> doublet validated"
      if np.abs(gd - gf).max() < 3e-3 else f"diff {np.abs(gd-gf).max():.1e}", flush=True)
import json
json.dump({"system": "CH3_doublet", "ncas": 7, "nelecas": 7, "spin": 1,
           "g_fci_norm": float(np.linalg.norm(gf)), "g_dmrg_norm": float(np.linalg.norm(gd)),
           "max_abs_grad_err": float(np.abs(gd - gf).max())},
          open("data/nonsinglet_doublet_ch3.json", "w"), indent=2)
print("DONE")
