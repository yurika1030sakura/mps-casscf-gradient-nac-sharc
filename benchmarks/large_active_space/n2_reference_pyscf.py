"""Pure-pyscf N2 state-1 gradient reference check (no DMRG, no block2 -> can't hang there).
Compares the EXACT pyscf SA-CASSCF state-1 analytic gradient to the known
analytic-MPS value (-0.365) and the FD reference (-0.5023)."""
import time, numpy as np
from pyscf import gto, scf, mcscf
t0=time.time()
mol=gto.M(atom=[['N',(0,0,0)],['N',(0,0,1.098)]],basis='6-31G',verbose=0)
mf=scf.RHF(mol); mf.kernel()
mc=mcscf.CASSCF(mf,6,6).state_average_([0.5,0.5]); mc.conv_tol=1e-10; mc.kernel()
print('pyscf SA-CASSCF conv=%s e=%s (%.0fs)'%(mc.converged,list(np.round(mc.e_states,6)),time.time()-t0),flush=True)
g1=mc.nuc_grad_method().kernel(state=1)
print('pyscf-FCI state1 g[1,2]=%.6f'%float(g1[1,2]),flush=True)
print('VERDICT: analytic-MPS=-0.36538  pyscf-FCI=%.6f  FD=-0.50230'%float(g1[1,2]),flush=True)
print('  -> if pyscf-FCI ~= -0.365 : analytic correct, FD artifact;  if ~= -0.502 : analytic BUG',flush=True)
