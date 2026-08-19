# SA!=2 response-solver fix — verification record (2026-08-18)

Three defects, three fixes (F1 1/(2w) weight, F2 zero-safe combine, F3 CR acceptor),
branch response-exact-projection @d91109b. Full evidence chain in this directory.

## Independent adversarial run (SA-3 UNEQUAL weights 0.5/0.3/0.2, H4/3-21G/CAS(4,4))
- s0 (w=0.5): cert 4.387e-11, |z-z_dense| 1.66e-11  OK
- s1 (w=0.3): cert 6.402e-11, |z-z_dense| 1.43e-11  OK
- s2 (w=0.2): cert 6.314e-08 — misses the 1e-8 academic bar, 16x BELOW the 1e-6
  production certificate bar; 6x wall-time of s0 (227s vs 37s). Top-averaged-state
  CR convergence degrades with the small PD margin — COST finding, not correctness.
- Mutations (each fix disabled individually, unequal weights): all CAUGHT —
  M1 no-weight 2.59e-2, M2 no-zero-safe 2.39e-2, M3 multiply-only 7.54e-3.
  Every fix is individually load-bearing and the test suite detects its loss.

## Production-scale A/B in flight
A re-measurement on the production CAS(22,18)/SA-3-equal/def2-SVP case is running with
external solver settings identical to the pre-fix run (m=800/MinRes/proj_weight=1e3).
Baseline to beat: S0 9.10e-2, S1 4.35e-2. The post-fix residual measures the genuine
m=800 truncation floor of the response vector.
