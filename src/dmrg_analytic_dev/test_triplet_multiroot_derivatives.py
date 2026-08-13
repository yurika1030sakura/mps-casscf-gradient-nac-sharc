"""Independent multiroot triplet validation of the certified MPS backend.

This closes a production gap left by the historical non-singlet tests: they
checked triplet energies/RDMs and one ground-root gradient, but not both roots
of a state average together with an interstate NAC.  Here an asymmetric H4
CAS(4,4), SA2 calculation is constrained to S=1 and evaluated two ways:

1. dense-FCI SA-CASSCF plus PySCF's analytic gradient/NAC implementation;
2. :func:`certified_engine.compute_certified_derivatives`, which solves and
   certifies MPS-valued response vectors and assembles derivatives from those
   exact same accepted vectors.

Gradients are phase invariant.  The NAC is compared modulo its unavoidable
global electronic-state sign.  The 3-21G basis is load-bearing: unlike an
H4/STO-3G/CAS(4,4) full-CI-in-the-entire-MO-space calculation, it leaves four
external orbitals and therefore exercises a non-zero physical orbital-response
block.  A full-active STO-3G test would normalize the response residual by a
tiny DMRG eigensolver-noise RHS rather than test CP-CASSCF response.  This case
is still small enough for an independent dense reference; production target
spaces remain FCI-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf.nac import sacasscf as sacasscf_nac

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parents[1] / "sharc_interface"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from certified_engine import compute_certified_derivatives  # noqa: E402


ATOMS = ["H", "H", "H", "H"]
COORDS_BOHR = np.array(
    [
        [0.00, 0.00, 0.00],
        [0.15, 0.00, 1.35],
        [-0.20, 0.10, 2.90],
        [0.25, -0.15, 4.70],
    ],
    dtype=float,
)
BASIS = "3-21g"


def _dense_triplet_reference():
    mol = gto.M(
        atom=[(a, tuple(r)) for a, r in zip(ATOMS, COORDS_BOHR)],
        basis=BASIS, charge=0, spin=2, unit="Bohr", symmetry=False,
        verbose=0,
    )
    # Explicit RHF is an orbital reference only.  Unlike the scf.RHF factory
    # (which dispatches to ROHF for mol.spin != 0), it keeps PySCF's analytic
    # SA-CASSCF NAC core-density path rank-compatible.  The active solver below
    # still targets the physical (3 alpha, 1 beta), S=1 sector exactly.
    mf = scf.hf.RHF(mol).run(conv_tol=1.0e-12)
    mc = mcscf.CASSCF(mf, 4, (3, 1))
    mc.fcisolver.nroots = 2
    mc.fix_spin_(ss=2.0, shift=0.5)
    mc = mc.state_average_([0.5, 0.5])
    mc.conv_tol = 1.0e-10
    mc.conv_tol_grad = 1.0e-6
    mc.max_cycle_macro = 200
    mc.kernel()
    if not mc.converged:
        raise RuntimeError("dense triplet SA-CASSCF reference did not converge")
    orbital_dim = mc.pack_uniq_var(
        np.zeros((mc.mo_coeff.shape[1], mc.mo_coeff.shape[1]))
    ).size
    if orbital_dim <= 0:
        raise RuntimeError(
            "triplet validation must exercise a non-zero orbital-response block"
        )

    s2 = [
        float(fci.spin_square(np.asarray(ci), 4, mc.nelecas)[0])
        for ci in mc.ci
    ]
    gradients = {
        state: np.asarray(
            sacasscf_grad.Gradients(mc).kernel(state=state), dtype=float,
        )
        for state in (0, 1)
    }
    nac = np.asarray(
        sacasscf_nac.NonAdiabaticCouplings(mc).kernel(state=(0, 1)),
        dtype=float,
    )
    return mc, gradients, nac, s2


def _phase_invariant_error(value, reference):
    return min(
        float(np.linalg.norm(value - reference)),
        float(np.linalg.norm(value + reference)),
    )


def main():
    ref, g_ref, nac_ref, s2_ref = _dense_triplet_reference()
    out = compute_certified_derivatives(
        ATOMS, COORDS_BOHR, basis=BASIS, charge=0, spin=2,
        ncas=4, nelecas=4, nroots=2, weights=[0.5, 0.5],
        mo_guess=ref.mo_coeff, gradient_states=[0, 1], nac_pairs=[(0, 1)],
        grad_tol=1.0e-6, nac_tol=1.0e-6, max_bond_dim=64,
        threads=4, stack_mem_mb=1000, force_fci_free=True,
        dmrg_sweep_tol=1.0e-14, refine_sweeps=40,
    )

    print("triplet reference energies:", list(map(float, ref.e_states)))
    print("triplet reference <S^2>:", s2_ref)
    print("certified build energies:", out["build"]["e_states"])
    print("certified build <S^2>:", out["build"].get("s2_per_state"))
    print("overall health:", out["overall_health"])

    ok = out["overall_health"] == "PASS"
    ok &= out["fci_free"]["required"] is True
    ok &= out["fci_free"]["dense_bridge_used"] is False
    ok &= tuple(ref.nelecas) == (3, 1)
    ok &= max(abs(x - 2.0) for x in s2_ref) < 1.0e-7
    build_s2 = out["build"].get("s2_per_state")
    ok &= build_s2 is not None
    if build_s2 is not None:
        ok &= max(abs(float(x) - 2.0) for x in build_s2) < 1.0e-12

    energy_error = float(np.max(np.abs(
        np.asarray(out["build"]["e_states"]) - np.asarray(ref.e_states)
    )))
    print(f"max energy error: {energy_error:.3e} Eh")
    ok &= energy_error < 1.0e-6

    for state in (0, 1):
        rec = out["gradients"][state]
        cert = rec.get("certificate", {})
        value = rec.get("grad")
        error = np.inf if value is None else float(np.max(np.abs(
            np.asarray(value, dtype=float) - g_ref[state]
        )))
        print(
            f"G{state}: health={rec['health']} max|MPS-FCI|={error:.3e} "
            f"true_rel_res={cert.get('true_residual_relative')}"
        )
        ok &= rec["health"] == "PASS"
        ok &= error < 5.0e-6

    nrec = out["nacs"]["(0, 1)"]
    ncert = nrec.get("certificate", {})
    nvalue = nrec.get("nac")
    nac_error = np.inf if nvalue is None else _phase_invariant_error(
        np.asarray(nvalue, dtype=float), nac_ref,
    )
    print(
        f"NAC(0,1): health={nrec['health']} phase-invariant error="
        f"{nac_error:.3e} true_rel_res="
        f"{ncert.get('true_residual_relative')}"
    )
    ok &= nrec["health"] == "PASS"
    ok &= float(np.linalg.norm(nac_ref)) > 1.0e-6
    ok &= nac_error < 5.0e-6

    print("TRIPLET MULTIROOT DERIVATIVE TEST: PASS" if ok else
          "TRIPLET MULTIROOT DERIVATIVE TEST: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
