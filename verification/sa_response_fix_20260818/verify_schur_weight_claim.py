"""Adversarial verification of the claimed 1/(2w) bug in _ci_block_inverse.

Dense emulation of solve_state_sweep_schur's EXACT algebra using the repo's
own dense SA-CASSCF Hessian blocks (cp_casscf_response.CPCASSCFResponseFCI,
which mirrors pyscf newton_casscf.gen_g_hop including the 2*w_I factors).

For each SA setup (SA-2, SA-3, SA-4 on LiH CAS(2,2); SA-5 on LiH CAS(4,4)):
  1. build dense H_OO, H_OC, H_CO, H_CC by probing the block applies;
  2. emulate the Schur solver with the BARE inverse (H-E_i)^{-1} exactly as
     _ci_block_inverse does (root-projected, no weight factor);
  3. emulate the FIXED inverse [2 w_i (H-E_i)]^{-1};
  4. compute the TRUE relative residual against the full projected operator
     (the same arbiter solve_state_sweep_schur uses, matvec_mps convention);
  5. compare both to the exact lstsq solution of the full system;
  6. verify the predicted residual identity r_c = (1-2w) * (b_c - H_CO z_k)
     and that r_kappa ~ 0 for the buggy solver;
  7. test the 'absorb the scale downstream' alternative: is buggy z a
     constant multiple of exact z? (if not, no downstream convention can fix)
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "dmrg_analytic_dev"))
from pyscf import gto, scf, mcscf
from cp_casscf_response import CPCASSCFResponseFCI

np.set_printoptions(precision=3, suppress=False)


def run_case(label, mol, ncas, nelecas, nroots, state=0):
    mf = scf.RHF(mol).run(verbose=0)
    w = np.ones(nroots) / nroots
    mc = mcscf.CASSCF(mf, ncas, nelecas).state_average_(list(w))
    mc.conv_tol = 1e-12
    mc.verbose = 0
    mc.kernel()
    # rebuild a plain SA-CASSCF object view for the response class
    obj = CPCASSCFResponseFCI(mc, weights=w, backend="freitag_reiher")

    nmo = obj.nmo
    nrot = mc.pack_uniq_var(np.zeros((nmo, nmo))).size
    ci_shape = obj.ci_list[0].shape
    nci = obj.ci_list[0].size
    ns = obj.nstates
    C = np.stack([c.ravel() for c in obj.ci_list])          # (ns, nci) roots

    # projector onto complement of ALL roots (matches _proj_out_roots / matvec proj)
    P = np.eye(nci) - C.T @ C

    def pack(k):  return mc.pack_uniq_var(k)
    def unpack(x): return mc.unpack_uniq_var(x)

    # ---- dense blocks by probing the repo's own applies ----
    HOO = np.zeros((nrot, nrot))
    for k in range(nrot):
        e = np.zeros(nrot); e[k] = 1.0
        HOO[:, k] = pack(obj.H_OO_apply(unpack(e)))

    # H_CO: orbital -> CI (per state), with weights 2*w_I inside apply
    HCO = np.zeros((ns * nci, nrot))
    for k in range(nrot):
        e = np.zeros(nrot); e[k] = 1.0
        out = obj.H_CO_apply(unpack(e))
        HCO[:, k] = np.concatenate([o.ravel() for o in out])

    # H_OC: CI -> orbital, weighted inside apply
    HOC = np.zeros((nrot, ns * nci))
    for i in range(ns):
        for a in range(nci):
            v = [np.zeros(ci_shape) for _ in range(ns)]
            v[i].flat[a] = 1.0
            HOC[:, i * nci + a] = pack(obj.H_OC_apply(v))

    # H_CC: slot-diagonal, weighted inside apply
    HCC = np.zeros((ns * nci, ns * nci))
    for i in range(ns):
        for a in range(nci):
            v = [np.zeros(ci_shape) for _ in range(ns)]
            v[i].flat[a] = 1.0
            out = obj.H_CC_apply(v)
            HCC[:, i * nci + a] = np.concatenate([o.ravel() for o in out])

    # bare active-space CI Hamiltonian (what _hcc_shifted_mpo encodes)
    from cp_casscf_response import single_site_sigma_fci_fallback
    cache = obj._build_eris_cache()
    h1cas0, eri_cas = cache["h1cas_0"], cache["eri_cas"]
    Hci = np.zeros((nci, nci))
    for a in range(nci):
        v = np.zeros(nci); v[a] = 1.0
        Hci[:, a] = single_site_sigma_fci_fallback(
            h1cas0, eri_cas, v.reshape(ci_shape), obj.ncas, obj.nelec,
            fcisolver=mc.fcisolver).ravel()
    eci0 = cache["eci0"]

    # sanity: P HCC_ii P == 2 w_i P (Hci - E_i) P ?
    for i in range(ns):
        blk = HCC[i*nci:(i+1)*nci, i*nci:(i+1)*nci]
        lhs = P @ blk @ P
        rhs = 2.0 * w[i] * (P @ (Hci - eci0[i]*np.eye(nci)) @ P)
        assert np.max(np.abs(lhs - rhs)) < 1e-9, f"HCC block identity fails st {i}"

    # ---- RHS (state gradient), pyscf convention ----
    rhs_O, rhs_C = obj.build_rhs(state)
    b_k = pack(rhs_O)
    b_c = np.concatenate([c.ravel() for c in rhs_C])

    # ---- projected full operator (matvec_mps convention) ----
    Pbig = np.kron(np.eye(ns), P)
    def A_apply(zk, zc):
        rk = HOO @ zk + HOC @ zc
        rc = Pbig @ (HCO @ zk + HCC @ zc)
        return rk, rc

    A_full = np.zeros((nrot + ns*nci, nrot + ns*nci))
    for k in range(nrot + ns*nci):
        e = np.zeros(nrot + ns*nci); e[k] = 1.0
        rk, rc = A_apply(e[:nrot], e[nrot:])
        A_full[:, k] = np.concatenate([rk, rc])
    b_full = np.concatenate([b_k, Pbig @ b_c])
    z_exact, *_ = np.linalg.lstsq(A_full, b_full, rcond=1e-12)

    # ---- per-state bare projected inverse (what _ci_block_inverse computes) ----
    def make_Dinv(scale_by_2w: bool):
        Dinv = np.zeros((ns*nci, ns*nci))
        for i in range(ns):
            M = P @ (Hci - eci0[i]*np.eye(nci)) @ P
            Minv = np.linalg.pinv(M, rcond=1e-11)          # inverse on P-space
            Minv = P @ Minv @ P
            if scale_by_2w:
                Minv = Minv / (2.0 * w[i])
            Dinv[i*nci:(i+1)*nci, i*nci:(i+1)*nci] = Minv
        return Dinv

    def schur_solve(Dinv):
        binv = Dinv @ (Pbig @ b_c)                          # RHS projected first
        rhs_schur = b_k - HOC @ binv
        S = HOO - HOC @ Dinv @ HCO
        zk, *_ = np.linalg.lstsq(S, rhs_schur, rcond=1e-12)
        zc = Dinv @ (Pbig @ (b_c - HCO @ zk))
        rk, rc = A_apply(zk, zc)
        r = np.concatenate([b_k - rk, Pbig @ b_c - rc])
        rel = np.linalg.norm(r) / max(np.linalg.norm(b_full), 1e-30)
        return zk, zc, rel, np.linalg.norm(b_k - rk), np.linalg.norm(Pbig@b_c - rc)

    zk_bug, zc_bug, rel_bug, rk_bug, rc_bug = schur_solve(make_Dinv(False))
    zk_fix, zc_fix, rel_fix, rk_fix, rc_fix = schur_solve(make_Dinv(True))

    # predicted identity for the buggy path: r_c = (1-2w)*(b_c - HCO zk_bug) in P-space
    pred_rc = np.concatenate([
        (1.0 - 2.0*w[i]) * (Pbig @ (b_c - HCO @ zk_bug))[i*nci:(i+1)*nci]
        for i in range(ns)])
    rk_a, rc_a = A_apply(zk_bug, zc_bug)
    act_rc = Pbig @ b_c - rc_a
    id_err = np.linalg.norm(pred_rc - act_rc)

    # can a constant rescale of buggy z reproduce exact z? (downstream absorption test)
    z_bug = np.concatenate([zk_bug, zc_bug])
    denom = float(z_bug @ z_bug)
    alpha = float(z_bug @ z_exact) / denom if denom > 0 else 0.0
    absorb_err = np.linalg.norm(alpha * z_bug - z_exact) / max(np.linalg.norm(z_exact), 1e-30)
    fix_vs_exact = np.linalg.norm(np.concatenate([zk_fix, zc_fix]) - z_exact) / max(np.linalg.norm(z_exact), 1e-30)

    print(f"\n=== {label}: SA-{ns} equal weights, 2w = {2*w[0]:.4f}, nrot={nrot}, nci={nci} ===")
    print(f"  E_states = {[f'{e:.6f}' for e in mc.e_states]}")
    print(f"  buggy (bare (H-E)^-1):  true rel resid = {rel_bug:.3e}"
          f"   [|r_k|={rk_bug:.2e} |r_c|={rc_bug:.2e}]")
    print(f"  fixed (/(2w)):          true rel resid = {rel_fix:.3e}"
          f"   [|r_k|={rk_fix:.2e} |r_c|={rc_fix:.2e}]")
    print(f"  identity check r_c=(1-2w)(b_c-HCO zk): err = {id_err:.3e}")
    print(f"  fixed z vs exact z (lstsq of full system): rel diff = {fix_vs_exact:.3e}")
    print(f"  best constant-rescale of buggy z vs exact: rel err = {absorb_err:.3e}"
          f"  (alpha={alpha:.4f})")
    return rel_bug, rel_fix


mol_lih = gto.M(atom="Li 0 0 0; H 0 0 1.8", basis="sto-3g", verbose=0)
run_case("LiH CAS(2,2)", mol_lih, 2, 2, 2)
run_case("LiH CAS(2,2)", mol_lih, 2, 2, 3)
run_case("LiH CAS(2,2)", mol_lih, 2, 2, 4)
run_case("LiH CAS(4,4) SA-5", mol_lih, 4, 4, 5)
