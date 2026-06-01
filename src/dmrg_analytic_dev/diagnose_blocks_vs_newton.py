"""Diagnose factor mismatches by comparing FR blocks to newton_casscf h_op.

newton_casscf.gen_g_hop returns (g, g_update, h_op, h_diag) where h_op acts
on x = [X_packed, c_1, c_2, ...]:
  - feed [X, 0_ci]   → get [A_OO X, A_CO X]
  - feed [0_orb, c]  → get [A_OC c, A_CC c]

These outputs are guaranteed Hermitian-paired (since newton_casscf builds the
Hessian of a single scalar SA-CASSCF energy). So we use newton_casscf as the
ground truth and compare each block from cp_casscf_response to it.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.mcscf import newton_casscf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cp_casscf_response import CPCASSCFResponseFCI, _project_orthogonal


def setup_heh_321g():
    mol = gto.M(atom="He 0 0 0; H 0 0 1.4", basis="3-21G",
                charge=1, spin=0, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fix_spin_(ss=0)
    mc.fcisolver.nroots = 2
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-10
    mc.max_cycle_macro = 200
    mc = mc.state_average_([0.5, 0.5])
    mc.kernel()
    return mc


def main():
    mc = setup_heh_321g()
    cp = CPCASSCFResponseFCI(mc)
    eris = mc.ao2mo(mc.mo_coeff)
    g_all, _g_upd, h_op, _h_diag = newton_casscf.gen_g_hop(
        mc, mc.mo_coeff, cp.ci_list, eris,
    )
    nrot = mc.pack_uniq_var(np.zeros((cp.nmo, cp.nmo))).size
    ci_size = cp.ci_list[0].size
    print(f"nrot = {nrot}, ci_size = {ci_size}, nstates = {cp.nstates}")
    print(f"newton_casscf returned vector size: total = {nrot + ci_size*cp.nstates}")
    print(f"|g_all| = {np.linalg.norm(g_all):.3e}")

    rng = np.random.default_rng(7)

    # === Test 1: feed unit X (packed orbital), zero CI -> get [H_OO X, H_CO X]
    print("\n--- H^OO and H^CO probe (unit X) ---")
    X_packed = rng.standard_normal(nrot)
    c_zero = [np.zeros_like(ci) for ci in cp.ci_list]
    x_in = np.concatenate([X_packed] + [ci.ravel() for ci in c_zero])
    y = h_op(x_in)
    HOO_X_packed = y[:nrot]
    HCO_X_per_state = [y[nrot + i*ci_size:nrot + (i+1)*ci_size].reshape(cp.ci_list[0].shape)
                       for i in range(cp.nstates)]
    print(f"newton_casscf H_OO X norm: {np.linalg.norm(HOO_X_packed):.3e}")
    print(f"newton_casscf H_CO X[state 0] norm: {np.linalg.norm(HCO_X_per_state[0]):.3e}")

    # Compare to my implementations
    kappa = mc.unpack_uniq_var(X_packed)
    HOO_my = cp.H_OO_apply(kappa)
    HOO_my_packed = mc.pack_uniq_var(HOO_my)
    HCO_my_per_state = cp.H_CO_apply(kappa)
    print(f"my       H_OO X norm: {np.linalg.norm(HOO_my_packed):.3e}")
    print(f"my       H_CO X[state 0] norm: {np.linalg.norm(HCO_my_per_state[0]):.3e}")
    print(f"diff H_OO: {np.linalg.norm(HOO_my_packed - HOO_X_packed):.3e}")
    print(f"diff H_CO[0]: {np.linalg.norm(HCO_my_per_state[0] - HCO_X_per_state[0]):.3e}")
    if np.linalg.norm(HCO_X_per_state[0]) > 1e-10:
        print(f"ratio my/newton H_CO[0]:  "
              f"max = {np.max(np.abs(HCO_my_per_state[0]) / np.maximum(np.abs(HCO_X_per_state[0]),1e-12)):.4f}")

    # === Test 2: feed zero X, unit CI -> get [H_OC c, H_CC c]
    print("\n--- H^OC and H^CC probe (unit c) ---")
    c_in = [rng.standard_normal(ci.shape) for ci in cp.ci_list]
    x_in = np.concatenate([np.zeros(nrot)] + [c.ravel() for c in c_in])
    y = h_op(x_in)
    HOC_c_packed = y[:nrot]
    HCC_c_per_state = [y[nrot + i*ci_size:nrot + (i+1)*ci_size].reshape(cp.ci_list[0].shape)
                       for i in range(cp.nstates)]
    print(f"newton_casscf H_OC c norm: {np.linalg.norm(HOC_c_packed):.3e}")
    print(f"newton_casscf H_CC c[state 0] norm: {np.linalg.norm(HCC_c_per_state[0]):.3e}")

    HOC_my = cp.H_OC_apply(c_in)
    HOC_my_packed = mc.pack_uniq_var(HOC_my)
    HCC_my_per_state = cp.H_CC_apply(c_in)
    print(f"my       H_OC c norm: {np.linalg.norm(HOC_my_packed):.3e}")
    print(f"my       H_CC c[state 0] norm: {np.linalg.norm(HCC_my_per_state[0]):.3e}")
    print(f"diff H_OC: {np.linalg.norm(HOC_my_packed - HOC_c_packed):.3e}")
    print(f"diff H_CC[0]: {np.linalg.norm(HCC_my_per_state[0] - HCC_c_per_state[0]):.3e}")
    if np.linalg.norm(HOC_c_packed) > 1e-10:
        ratio = HOC_my_packed / np.where(np.abs(HOC_c_packed) > 1e-12, HOC_c_packed, 1.0)
        print(f"per-element ratio my/newton H_OC: \n  {ratio}")

    if np.linalg.norm(HCC_c_per_state[0]) > 1e-10:
        idx = np.unravel_index(np.argmax(np.abs(HCC_c_per_state[0])), HCC_c_per_state[0].shape)
        my_v = HCC_my_per_state[0][idx]
        ne_v = HCC_c_per_state[0][idx]
        print(f"  max-elem H_CC[0]: my={my_v:.6e}, newton={ne_v:.6e}, ratio={my_v/ne_v:.4f}")


if __name__ == "__main__":
    main()
