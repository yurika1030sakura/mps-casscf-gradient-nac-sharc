"""Diagnostic: why a real cross-geometry overlap needs the displaced MPS intact.

The orbital-rotation kernel is validated separately (see
``test_cross_geometry_overlap.py``: it reproduces ``overlap_fci`` to ~8e-6 at
finite-difference magnitude).  To assemble <Psi_I(R) | Psi_J(R+h)> the displaced
state ``Psi_J(R+h)`` must be available in the reference driver.

A tempting shortcut -- read the displaced state out to an FCI vector and rebuild
it in the reference driver via the CSF round-trip -- is shown here to be too
lossy: at a finite-difference step the displaced state differs from the
reference by O(1e-5) components, and the CSF round-trip discards them, collapsing
the rebuilt state back onto the reference.  The no-rotation control below makes
this explicit: the rebuilt overlap matrix is the identity (the displacement
information is gone), whereas the exact ``overlap_fci`` with s = I retains the
O(1e-5) off-diagonal.

Conclusion: the production cross-geometry overlap must transport the displaced
MPS itself (copy its block2 files into the reference driver's scratch and
``load_mps`` it), not reconstruct it from a truncated CI read-out.  This script
records that requirement; it is a diagnostic, not a pass/fail regression.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SHARC = _HERE.parents[1] / "sharc_interface"
if str(_SHARC) not in sys.path:
    sys.path.insert(0, str(_SHARC))

import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response
from overlap_fci_reference import overlap_fci
from site_replacement_density import fci_to_mps_via_csf

ANG = 1.8897261246257702


def _build(z_bohr):
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=100, n_sweeps=24, sweep_tol=1.0e-10, n_threads=1)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, z_bohr]])
    mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
        ["He", "H"], coords, basis="3-21G", charge=1, spin=0,
        ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5], solver_cfg=cfg,
    )
    return mol, mc, solver


def main():
    h = 1.0e-3
    z0 = 0.90 * ANG
    _mol_R, mc_R, solver_R = _build(z0)
    _mol_P, _mc_P, solver_P = _build(z0 + h)

    ncas = mc_R.ncas
    nelec = (mc_R.nelecas[0], mc_R.nelecas[1]) if isinstance(mc_R.nelecas, (tuple, list)) \
        else (mc_R.nelecas // 2 + mc_R.nelecas % 2, mc_R.nelecas // 2)
    nst = 2

    ci_R = fdv.mps_ci_list(solver_R, ncas, nelec, nst)
    ci_P = fdv.mps_ci_list(solver_P, ncas, nelec, nst)

    obj = _make_mps_krylov_response(mc_R)
    host = obj._driver_su2
    states_R = obj._state_mps

    rebuilt_P = [fci_to_mps_via_csf(host, np.asarray(ci_P[j]), ncas, nelec,
                                    tag=f"DIAGP{j}") for j in range(nst)]
    O_rebuilt = np.array([[obj._mps_overlap(states_R[i], rebuilt_P[j])
                           for j in range(nst)] for i in range(nst)])
    O_exact = np.array([[overlap_fci(ci_R[i], ci_P[j], np.eye(ncas), ncas, nelec)
                         for j in range(nst)] for i in range(nst)])

    offdiag_rebuilt = float(np.max(np.abs(O_rebuilt - np.diag(np.diag(O_rebuilt)))))
    offdiag_exact = float(np.max(np.abs(O_exact - np.diag(np.diag(O_exact)))))

    out = {
        "name": "cross_geometry_transport_requirement_heh_cas22",
        "h_bohr": h,
        "rebuilt_offdiagonal": offdiag_rebuilt,
        "exact_offdiagonal": offdiag_exact,
        "finding": ("fci->mps round-trip drops the O(1e-5) finite-difference "
                    "components; production needs cross-driver MPS transport"),
        "O_rebuilt": O_rebuilt.tolist(),
        "O_exact": O_exact.tolist(),
    }
    (_HERE / "diag_cross_geometry_transport.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"rebuilt off-diagonal {offdiag_rebuilt:.2e} vs exact {offdiag_exact:.2e}"
          f" -> round-trip loses the displacement; cross-driver transport required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
