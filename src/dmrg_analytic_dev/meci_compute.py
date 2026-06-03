"""compute_fn factory for MECIOptimizer: SA-DMRG-CASSCF + paper's
analytic CP-CASSCF response. Per-step pipeline:

    1. Build pyscf.gto.M from current geom
    2. RHF (chkfile-warm-started if available)
    3. SA-CASSCF orbital opt with MPSAsFCISolver (DMRG)
       - mps_persistent_dir + previous-step MPS → cross-step warm start
    4. Compute grad(state0), grad(state1), NAC(0,1) via paper's
       compute_grad_nac_analytic_cp (MPS-Krylov backend)

Returns the dict the optimizer expects:
    {'e0': ..., 'e1': ..., 'grad0': ..., 'grad1': ..., 'nac01': ...}

Usage::

    from meci_compute import make_compute_fn
    compute_fn = make_compute_fn(
        atoms=['N', 'C', 'C', 'N', 'C', 'N', 'C', 'N',
               'H', 'H', 'H', 'H'],
        basis='cc-pVDZ', charge=0,
        ncas=12, nelecas=14,
        avas_labels=['C 2pz', 'N 2pz', 'N 2pn'],
        bond_dim=512,
        mps_persistent_dir='./damn_meci_xstep',
        sa_weights=[0.5, 0.5],
    )
    from meci_optimizer import MECIOptimizer
    opt = MECIOptimizer(compute_fn, log_path='meci.log')
    geom_au, e0, e1, ok = opt.optimize(geom_init_au, max_steps=50)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make sibling modules importable when run from any cwd.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pyscf import gto, scf, mcscf, lib
from pyscf.mcscf import avas


def make_compute_fn(
    *,
    atoms,
    basis: str,
    charge: int = 0,
    spin: int = 0,
    ncas: int,
    nelecas: int,
    avas_labels=None,
    sa_weights=None,
    bond_dim: int = 512,
    n_sweeps: int = 24,
    sweep_tol: float = 1e-7,
    n_threads: int = 8,
    stack_mem_mb: int = 12000,
    casscf_conv_tol: float = 1e-5,
    casscf_max_cycle: int = 30,
    response_tol: float = 1e-5,
    response_max_iter: int = 50,
    mps_persistent_dir: str | None = None,
    chkfile_path: str | None = None,
    state_pair=(0, 1),
):
    """Build a compute_fn(geom_au) -> dict for the MECI optimizer.

    The returned compute_fn:
      - accepts geom in Bohr, shape (natom, 3)
      - returns dict with e0, e1, grad0, grad1, nac01 in atomic units

    The DMRG solver + chkfile use cross-step persistence so subsequent
    calls (from the optimizer's iteration loop) warm-start orbitals AND
    the MPS — typically reducing per-step cost from ~20h to ~1-2h at
    CAS(24,18) M=1024.
    """
    from dmrg_fcisolver import MPSAsFCISolver
    # The analytic CP-CASSCF response (paper's MPS-Krylov backend).
    sys.path.insert(0, str(_HERE.parent))  # sharc_interface neighbor
    from analytic_cp_sharc import compute_grad_nac_analytic_cp

    if sa_weights is None:
        sa_weights = [0.5, 0.5]
    n_states = len(sa_weights)
    ket, bra = int(state_pair[0]), int(state_pair[1])

    def _build_mol(geom_au: np.ndarray) -> gto.Mole:
        # geom in Bohr (atomic units)
        atom_list = [
            (atoms[i], tuple(geom_au[i].tolist())) for i in range(len(atoms))
        ]
        return gto.M(
            atom=atom_list, basis=basis, charge=charge, spin=spin,
            unit="Bohr", symmetry=False, verbose=0,
        )

    def compute(geom_au: np.ndarray):
        mol = _build_mol(np.asarray(geom_au, dtype=float))

        # ---- RHF (chkfile warm-start if available) -------------------
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        if chkfile_path:
            mf.chkfile = chkfile_path
            if Path(chkfile_path).exists():
                # Use chkfile MO as initial guess
                mf.init_guess = "chkfile"
        mf.run()

        # ---- Active orbital selection (AVAS) -------------------------
        if avas_labels:
            ncas_avas, nelecas_avas, mo_init = avas.avas(
                mf, list(avas_labels), canonicalize=True, with_iao=False,
            )
            assert ncas_avas == ncas, (
                f"AVAS picked {ncas_avas} active orbs, expected {ncas}; "
                f"check avas_labels"
            )
        else:
            mo_init = mf.mo_coeff

        # ---- SA-CASSCF with DMRG fcisolver ---------------------------
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.conv_tol = float(casscf_conv_tol)
        mc.max_cycle_macro = int(casscf_max_cycle)
        mc.fcisolver = MPSAsFCISolver(
            mol,
            bond_dim=bond_dim, n_sweeps=n_sweeps, sweep_tol=sweep_tol,
            n_threads=n_threads, stack_mem_mb=stack_mem_mb,
            force_dmrg=True, max_fci_dets=2_000_000_000,
            mps_native_rdms=True,
            skip_kernel_fci_conversion=True,
            warm_start=True,
            first_iter_warmup=True,
            dmrg_symm_su2=True,
            timing_log=True,
            mps_persistent_dir=mps_persistent_dir,
        )
        mc.fcisolver.nroots = n_states
        mc = mc.state_average_(list(sa_weights))
        mc.kernel(mo_init)

        # ---- Analytic gradient + NAC via paper's MPS-Krylov ---------
        cp_results = compute_grad_nac_analytic_cp(
            mc,
            gradient_states=[ket, bra],
            nac_pairs=[(ket, bra)],
            backend="mps-krylov",
            tol=float(response_tol),
            max_iter=int(response_max_iter),
        )
        grad0 = np.asarray(cp_results["grad"][ket])
        grad1 = np.asarray(cp_results["grad"][bra])
        nac01 = np.asarray(cp_results["nac"][(ket, bra)])

        return dict(
            e0=float(mc.e_states[ket]),
            e1=float(mc.e_states[bra]),
            grad0=grad0,
            grad1=grad1,
            nac01=nac01,
        )

    return compute
