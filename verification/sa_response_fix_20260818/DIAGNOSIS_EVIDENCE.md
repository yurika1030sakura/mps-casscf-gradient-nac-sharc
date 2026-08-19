# MPS-level reproducer: SA-CASSCF response stagnation (sweep-Schur + global)

Repo under test: this repository at commit 776c42c (pre-fix), imported READ-ONLY.
All candidate fixes live in generated scratch copies
(sweep_coupled_response_fixed.py / sweep_coupled_response_fixed_all.py).
Env: python with pyscf + pyblock2 (see requirements.txt).

DELIVERABLE: test_sa_weight_ci_block_inverse.py  (pytest; see its docstring
for the full defect list D1-D4).  Helpers: common.py, dense_ref.py.

## Evidence trail (all logs preserved)

| stage | file | result |
|---|---|---|
| 1 | stage1_h4_old.py / stage1.log | H4/3-21G/CAS(4,4) SA-3 EQUAL, UNMODIFIED Schur: S0 6.07e-2, S1 4.44e-2 (production: 9.1e-2/4.4e-2) |
| 2 | stage2_h4_fixed.py / stage2.log | weight-fix alone: S0 5.05e-2, S1 4.26e-2 (insufficient); dense solve() DIVERGES on equal weights (info=400, dense z 1e11) |
| 3 | stage3_lih_sa2.py / stage3.log | LiH/CAS(2,2) SA-2 (0.7,0.3): OLD floors 4.06e-2 both states; weight-fix ALONE certifies 6e-16/2e-15 and matches dense to 3e-15/4e-15 -> D1 proven, blind spot = equal weights |
| 4 | stage4_h4_dissect.py / stage4.log | H4 SA-3 UNEQUAL floors too (4.3e-2 fixed) -> not equal-weight-specific; GLOBAL solve_mps floors (3.1e-2) -> not Schur-specific; residual lives in CI slots of the non-target states |
| 5 | stage5_dense_projected.py / stage5.log | dense projected lstsq machinery; exposed mc.ci-based reference as garbage |
| 6 | stage6_readout_check.py / stage6.log | kets<->mc.ci overlap up to 0.7071 mismatch; MPS rhs slot != dense rhs slot |
| 7 | stage7_state_rotation.py / stage7.log | mc.ci NOT eigenvectors of dense H_act (\|Hc-Ec\| ~ 0.1-0.2, both weight patterns); MPS kets ARE (1e-11) -> kernel MPS->FCI conversion broken at ncas>2 (D4) |
| 8 | stage8_hccinv_fidelity.py / stage8.log | inner inverse forward-fidelity 1.2e-2/1.7e-2/5.1e-2 per slot on physical coupling RHS |
| 9 | stage9_inner_solver_levers.py / stage9.log | penalty own-root vs ALL roots, W 1e3 vs 1e5, CG vs MinRes, growing schedule+noises: ALL INERT (identical floors); iterative refinement converges slowly (~0.5-0.7x/iter) |
| 10 | stage10_sigma_truncation.py / stage10.log | _sigma_mps and _combine_mps EXACT vs dense (1e-15) -> arithmetic not guilty; kets-based dense residual of MPS z (5.6e-2) == MPS certificate (5.0e-2) -> arbiter honest, solver wrong (D3) |
| 11 | stage11_inner_krylov.py / stage11.log | block2 multiply(left_mpo) verbose: DF=-2e-19, per-site Error=0, Nmult=2 -- SILENT false convergence; thrds=1e-12 -> 3.6e-5 (default thrds=[1e-6]*4+[1e-7] never overridden = D2); root-projected CR on exact primitives: 1e-11 in 16-17 iters, all slots |
| 12 | stage12_full_fix.py / stage12.log | 1/(2w)+CR alone did NOT rescue H4 (4.9e-2) -- pointed past the inner solver |
| 13 | stage13_verify_cr_engaged.py / stage13.log | CR engaged; arbiter CI slot-1 resid 3.94e-3 is z_C-INDEPENDENT; in-function meta resid (0.0089) under-reports vs certificate (0.049) |
| 14 | stage14_cancelling_addition.py / stage14.log | ROOT CAUSE (D5): combine(a,b) exact when first term large (2.6e-18) but combine(zero-ish, a) returns ~0 (3.9e-14) -- ZERO-INITIALIZED ADDITION FITS ARE ALS FIXED POINTS for n_sites>2; H_CC z_C[1] = 1.4e-12 proves z_C[1] was annihilated via rhs_C = combine([(1, b_ci[i]=cached ZERO), (-1, hco_zk[i])]) at sweep_coupled_response.py:242-246 (same trap in the in-function residual :258-261); vector_linear_combination skips cached-zero slots, which is why certify stayed honest |
| 15 | stage15_block_vs_dense.py / stage15.log | ALL MPS blocks (H_CO/H_OC/H_OO/H_CC/rhs) match kets-based dense to 1e-13; dense residual decomposition: ci1 = 3.94e-3 = \|P H_CO z_k\|_1 exactly (the annihilated slot) |
| 16 | stage16_final.py / stage16.log | THE LADDER (H4 SA-3 equal, cert/dense): none 6.1e-2/6.3e-2 -> F1 5.0e-2/5.6e-2 -> F1+F2 7.3e-3/7.3e-3 (meta now honest) -> F1+F2+F3 2.5e-11/4.8e-11, \|z-z_dense\| = 6.9e-12 (s0); s1: 4.4e-2 -> 4.3e-2 -> 6.5e-3 -> 2.6e-11; LiH 0.7/0.3 complete fix: 4.9e-15/1.2e-15 |

## The fixes (scratch modules; repo-ready translation)
- F1 (D1): scale the _ci_block_inverse solution by 1/(2*obj.weights[i]).
- F2 (D5): never let _combine_mps initialize its addition fit from a
  zero-norm MPS -- order terms largest-first, drop zero terms (patched at
  sweep_coupled_response.py:242-246 rhs_C and :258-261 r_ci; the ROBUST repo
  fix is inside _combine_mps itself so every call site is safe).
- F3 (D2): replace the driver.multiply(left_mpo) inner solve with a
  root-projected conjugate-residual loop on _sigma_mps/_combine_mps (multiply
  is silently ~1e-2 wrong at default thrds, ~4e-5 at thrds=1e-12; on the root
  complement (H-E_i) is PD for every slot so CR converges in <= 20 iters).
  At production scale, multiply(tight thrds) can serve as a preconditioner
  with CR refinement on top.

Final validation: final_test_run.log (all tests of the deliverable).

## Failed/refuted hypotheses (preserved as evidence)
- stage2/4: equal-weight SA redundancy as the PRIMARY cause -- refuted
  (unequal SA-3 floors identically; projections in matvec/certify are fine).
- stage9 B/C: penalty->exact-projection or heavier penalty as the fix --
  inert (proj_mpss extension changed nothing measurable).
- stage10: sigma/combine bond-dim truncation (S2 at small scale) -- refuted
  at this scale (exact to 1e-15); S2 remains a separate production-scale
  (m=800) concern, not reproducible here.
