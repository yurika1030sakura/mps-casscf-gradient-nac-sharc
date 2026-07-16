"""Non-singlet (triplet) analytic gradient vs exact FCI: validates SU2 spin-sector
targeting for open-shell derivatives (spin-generality check: converts spin generality from
assertion to demonstration). Triplet CH2 (3B1), CAS(6,6)/6-31G*, SA(2) over the two
lowest triplets, DMRG(SU2, S=1) analytic state-0 gradient vs FCI(direct_spin1) gradient."""
import sys, numpy as np, time
sys.path.insert(0,"sharc_interface")
sys.path.insert(0,"src/dmrg_analytic_dev")
from pyscf import gto, scf, mcscf, fci
from dmrg_fcisolver import MPSAsFCISolver
from analytic_cp_sharc import compute_grad_nac_analytic_cp
import tempfile
t0=time.time(); L=lambda m:print(f"[{time.time()-t0:.0f}s] {m}",flush=True)
# triplet methylene 3B1: bent ~134 deg, C-H 1.078 A
mol=gto.M(atom="C 0 0 0.1; H 0 0.99 -0.31; H 0 -0.99 -0.31", basis="6-31G*", spin=2, verbose=0)
mf=scf.ROHF(mol).run(conv_tol=1e-10)
# FCI ground truth (direct_spin1, triplet), SA(2)
mcf=mcscf.CASSCF(mf,6,6); mcf.fcisolver=fci.direct_spin1.FCI(); mcf.fcisolver.spin=2
mcf=mcf.state_average([0.5,0.5]); mcf.conv_tol=1e-9; mcf.kernel()
gf=mcf.nuc_grad_method().kernel(state=0)
L(f"FCI triplet CAS(6,6): e_states={[round(float(x),6) for x in mcf.e_states[:2]]} |g0|={np.linalg.norm(gf):.6f}")
# DMRG (SU2, triplet sector) warm-started from FCI orbitals (same basin) for fair check
scr=tempfile.mkdtemp(prefix="ns_",dir="/tmp/dmrg_scratch")
sol=MPSAsFCISolver(mol,bond_dim=200,n_sweeps=16,n_threads=8,sweep_tol=1e-10,scratch_root=scr,force_dmrg=True,
   max_fci_dets=100000,root_buffer=4,refine_split_roots=True,dmrg_symm_su2=True,skip_kernel_fci_conversion=True,mps_native_rdms=True)
sol.fix_spin_(ss=2.0)   # S(S+1)=2 for S=1 triplet
mc=mcscf.CASSCF(mf,6,6); mc.fcisolver=sol; mc=mc.state_average([0.5,0.5]); mc.conv_tol=1e-8
mc.kernel(mcf.mo_coeff)
L(f"DMRG triplet: e_states={[round(float(x),6) for x in np.asarray(mc.e_states)[:2]]} (FCI {[round(float(x),6) for x in mcf.e_states[:2]]})")
r=compute_grad_nac_analytic_cp(mc,gradient_states=[0],nac_pairs=[],backend="mps-krylov",tol=1e-4,max_iter=50)
gd=np.asarray(r["grad"][0])
L(f"DMRG triplet analytic grad0: |g|={np.linalg.norm(gd):.6f}  vs FCI max|dg|={np.abs(gd-gf).max():.2e}")
print(">>> VERDICT:", "SU2 triplet gradient = exact FCI -> non-singlet validated" if np.abs(gd-gf).max()<3e-3 else f"diff {np.abs(gd-gf).max():.1e}",flush=True)
print("DONE")
