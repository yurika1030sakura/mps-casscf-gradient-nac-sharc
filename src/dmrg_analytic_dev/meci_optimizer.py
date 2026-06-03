"""MECI optimization driver for SA-DMRG-CASSCF (paper-tier method).

Implements Bearpark/Robb projected-gradient MECI search using:
    * SA-CASSCF energies from MPSAsFCISolver + DMRG
    * Analytic gradient + NAC from paper's `compute_grad_nac_analytic_cp`
      (MPS-Krylov response, no FCI projection)
    * Cross-step MPS warm-start via `mps_persistent_dir` so per-step
      DMRG cost stays at ~minutes instead of hours

Workflow per opt step:
    1. SA-CASSCF orbital opt (warm-started from previous step's MPS)
    2. Compute gradient(state0), gradient(state1), NAC(0,1) via paper's
       analytic CP-CASSCF response solver
    3. Project gradient onto MECI seam using Bearpark formula
    4. Take quasi-Newton / steepest-descent step
    5. Convergence check on |∇seam| + |E1-E0|

References
----------
- Bearpark, M. J.; Robb, M. A.; Schlegel, H. B. Chem. Phys. Lett. 1994,
  223, 269-274.  (Original Bearpark projected-gradient method.)
- Levine, B. G.; Coe, J. D.; Martinez, T. J. J. Phys. Chem. B 2008, 112,
  405-413.  (Modern penalty-function variant; for reference only.)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Make sibling modules importable when run from any cwd
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def bearpark_meci_gradient(
    grad0: np.ndarray,
    grad1: np.ndarray,
    nac01: np.ndarray,
    e0: float,
    e1: float,
    *,
    sigma: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Bearpark projected MECI gradient.

    Decomposes the average gradient into in-seam and orthogonal pieces:
      - g_avg = 0.5 (∇E0 + ∇E1)              [average force on the seam]
      - g_dif = ∇E1 - ∇E0                    [difference vector]
      - h     = NAC(0,1) vector              [derivative-coupling vector]

    Defines:
      - g_seam = g_avg projected OUT of the {ĝ_dif, ĥ} plane
                 (minimizes average energy along the seam)
      - g_gap  = 2 σ (E1 - E0) ĝ_dif
                 (drives the gap toward zero along the difference vector)

    Returns:
      g_meci = g_seam + g_gap, plus diagnostics dict.

    Parameters
    ----------
    grad0, grad1
        Nuclear gradients (natom, 3) of the two states.
    nac01
        Derivative coupling (natom, 3) between states 0 and 1.
    e0, e1
        State energies (used for the gap-driving term).
    sigma
        Penalty weight for the gap term; 5 is a robust default.

    Returns
    -------
    g_meci : ndarray (natom, 3)
        Step direction for MECI optimization.
    info : dict
        Diagnostics: in_seam norm, gap, projection magnitudes.
    """
    grad0 = np.asarray(grad0, dtype=float)
    grad1 = np.asarray(grad1, dtype=float)
    nac01 = np.asarray(nac01, dtype=float)
    assert grad0.shape == grad1.shape == nac01.shape, (
        f"shape mismatch: g0={grad0.shape} g1={grad1.shape} "
        f"nac={nac01.shape}"
    )

    g_avg = 0.5 * (grad0 + grad1)
    g_dif = grad1 - grad0

    # Build orthonormal {ĝ_dif, ĥ} basis spanning the branching plane.
    g_dif_norm = float(np.linalg.norm(g_dif))
    nac_norm = float(np.linalg.norm(nac01))
    if g_dif_norm < 1e-12 or nac_norm < 1e-12:
        # Degenerate cases — just return the average gradient.
        return g_avg, {
            "warning": "branching plane vector(s) near zero",
            "g_dif_norm": g_dif_norm,
            "nac_norm": nac_norm,
        }

    e_dif = g_dif.ravel() / g_dif_norm
    # Gram-Schmidt orthogonalize the NAC vector against g_dif direction.
    h_proj = nac01.ravel() - (nac01.ravel() @ e_dif) * e_dif
    h_norm = float(np.linalg.norm(h_proj))
    if h_norm < 1e-12:
        # NAC parallel to g_dif: branching plane degenerate; only kill g_dif.
        e_h = None
        g_seam_flat = g_avg.ravel() - (g_avg.ravel() @ e_dif) * e_dif
    else:
        e_h = h_proj / h_norm
        g_avg_flat = g_avg.ravel()
        g_seam_flat = (
            g_avg_flat
            - (g_avg_flat @ e_dif) * e_dif
            - (g_avg_flat @ e_h) * e_h
        )

    g_seam = g_seam_flat.reshape(g_avg.shape)

    # Gap-driving term: 2σ(E1-E0) along the difference direction.
    gap = float(e1 - e0)
    g_gap = (2.0 * sigma * gap) * (g_dif / g_dif_norm)

    g_meci = g_seam + g_gap
    return g_meci, {
        "gap": gap,
        "g_dif_norm": g_dif_norm,
        "nac_norm": nac_norm,
        "g_seam_norm": float(np.linalg.norm(g_seam)),
        "g_gap_norm": float(np.linalg.norm(g_gap)),
        "g_meci_norm": float(np.linalg.norm(g_meci)),
    }


class MECIOptimizer:
    """Simple BFGS-driven MECI optimizer over Cartesian geometry.

    Initialize with a callable ``compute(geom_au_natomx3)`` that returns
    a dict with keys ``e0``, ``e1``, ``grad0``, ``grad1``, ``nac01``
    (all in atomic units, gradients/NAC shape (natom, 3)).

    Run with ``.optimize(initial_geom, max_steps=..., ...)``.

    The optimizer uses Cartesian Hessian initialized as a scaled identity
    and updated with BFGS. For most photochemistry MECI optimizations
    this converges in 20-50 steps from a reasonable initial guess.
    """

    def __init__(
        self,
        compute_fn,
        *,
        sigma: float = 5.0,
        trust_radius: float = 0.3,
        bfgs_min_step: float = 1e-3,
        conv_grad: float = 5e-4,   # max |g_meci|
        conv_egap: float = 1e-4,   # |E1 - E0|
        log_path: str | None = None,
    ):
        self.compute_fn = compute_fn
        self.sigma = float(sigma)
        self.trust_radius = float(trust_radius)
        self.bfgs_min_step = float(bfgs_min_step)
        self.conv_grad = float(conv_grad)
        self.conv_egap = float(conv_egap)
        self.log_path = log_path
        self.history = []

    def _log(self, msg: str):
        print(msg, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(msg + "\n")

    def optimize(self, geom0: np.ndarray, max_steps: int = 50):
        geom = np.asarray(geom0, dtype=float).copy()
        natom = geom.shape[0]
        ndof = natom * 3

        # Initial inverse Hessian guess (identity / 0.5)
        H_inv = np.eye(ndof) * 2.0
        prev_g = None
        prev_x = None

        for step in range(max_steps):
            t0 = time.time()
            self._log(f"\n=== MECI step {step}/{max_steps} ===")
            out = self.compute_fn(geom)
            e0 = float(out["e0"])
            e1 = float(out["e1"])
            grad0 = np.asarray(out["grad0"])
            grad1 = np.asarray(out["grad1"])
            nac01 = np.asarray(out["nac01"])

            g_meci, diag = bearpark_meci_gradient(
                grad0, grad1, nac01, e0, e1, sigma=self.sigma,
            )
            g_flat = g_meci.ravel()
            g_norm = float(np.linalg.norm(g_flat))
            gap = e1 - e0

            wall = time.time() - t0
            self._log(
                f"  E0={e0:.10f}  E1={e1:.10f}  gap={gap:+.3e}  "
                f"|g_meci|={g_norm:.3e}  wall={wall:.1f}s"
            )
            self._log(
                f"  diag: g_dif|={diag['g_dif_norm']:.3e}  "
                f"|nac|={diag['nac_norm']:.3e}  "
                f"|g_seam|={diag['g_seam_norm']:.3e}  "
                f"|g_gap|={diag['g_gap_norm']:.3e}"
            )

            self.history.append(dict(
                step=step, e0=e0, e1=e1, gap=gap,
                g_norm=g_norm, geom=geom.copy(), wall=wall,
            ))

            # Convergence
            if abs(gap) < self.conv_egap and g_norm < self.conv_grad:
                self._log(
                    f"\nCONVERGED at step {step}: |gap|={abs(gap):.3e} "
                    f"< {self.conv_egap}  |g|={g_norm:.3e} < {self.conv_grad}"
                )
                return geom, e0, e1, True

            # BFGS update on inverse Hessian (skip on step 0)
            if prev_g is not None and prev_x is not None:
                s = (geom - prev_x).ravel()
                y = g_flat - prev_g
                sy = float(s @ y)
                if sy > 1e-8:  # ensure positive curvature
                    rho = 1.0 / sy
                    I = np.eye(ndof)
                    V = I - rho * np.outer(s, y)
                    H_inv = V @ H_inv @ V.T + rho * np.outer(s, s)

            # Quasi-Newton step
            step_vec = -(H_inv @ g_flat)
            step_norm = float(np.linalg.norm(step_vec))
            if step_norm > self.trust_radius:
                step_vec *= self.trust_radius / step_norm
            if step_norm < self.bfgs_min_step:
                # Fall back to steepest descent if BFGS gives ~zero step
                step_vec = -0.1 * g_flat
                self._log(f"  BFGS step tiny ({step_norm:.2e}), using SD")

            prev_g = g_flat.copy()
            prev_x = geom.copy()
            geom = geom + step_vec.reshape(natom, 3)

        self._log(f"\nMAX_STEPS reached without convergence ({max_steps})")
        return geom, e0, e1, False
