#!/usr/bin/env python3
"""Validate the local multistate DMRG backend in subprocesses.

The local pyblock2 environment crashes hard enough that we do not want the
driver process itself to segfault while checking feasibility. This script runs
small child processes and reports whether they exit cleanly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

PYTHON = "/n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11"
ROOT = Path("/n/home04/yulili/daisuan/prebiotic_sutherland/sharc_pyscf_casscf")


def run_case(name: str, code: str) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    proc = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "name": name,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def main() -> int:
    scratch_root = Path(os.getenv("DMRG_VALIDATE_SCRATCH", "/tmp"))
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="dmrg_multistate_check_", dir=scratch_root))

    single_root = textwrap.dedent(
        f"""
        from pyscf import gto, scf, mcscf
        from pyblock2.dmrgscf import DMRGCI
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', spin=0, charge=0, symmetry=False, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.max_cycle_macro = 3
        solver = DMRGCI(mf)
        solver.dmrg_args['startM'] = 20
        solver.dmrg_args['maxM'] = 50
        solver.dmrg_args['sweep_tol'] = 1e-6
        mc.fcisolver = solver
        mc.kernel()
        print('single_root_ok', mc.e_tot)
        """
    )

    experimental_multiroot = textwrap.dedent(
        f"""
        from pyscf import gto, scf, mcscf
        from dmrg_sharc_bridge import ExperimentalMultiRootDMRGCI
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', spin=0, charge=0, symmetry=False, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.max_cycle_macro = 3
        mc.state_average_([0.5, 0.5])
        solver = ExperimentalMultiRootDMRGCI(mf)
        solver.dmrg_args['startM'] = 20
        solver.dmrg_args['maxM'] = 50
        solver.dmrg_args['sweep_tol'] = 1e-6
        solver.dmrg_args['nsteps'] = 8
        solver.dmrg_args['memory'] = int(2e8)
        solver.dmrg_args['scratch_root'] = {str(scratch / 'experimental_sa')!r}
        mc.fcisolver = solver
        mc.kernel()
        print('experimental_sa_ok', mc.e_tot, mc.e_states)
        """
    )

    raw_sidmrg = textwrap.dedent(
        f"""
        from pyscf import gto, scf, mcscf, ao2mo
        from pyblock2.sidmrg import SIDMRG, SpinLabel
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', spin=0, charge=0, symmetry=False, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        ff = mcscf.CASCI(mf, 2, 2)
        h1e, e_core = ff.get_h1cas()
        g2e = ao2mo.restore(1, ff.get_h2cas(), 2)
        dmrg = SIDMRG(scratch={str(scratch / 'raw_sidmrg')!r}, memory=2e8, omp_threads=1, verbose=1)
        dmrg.init_hamiltonian('c1', n_sites=2, n_elec=2, twos=0, isym=1, orb_sym=[1,1], e_core=e_core, h1e=h1e, g2e=g2e)
        energies = dmrg.dmrg(target=SpinLabel(2, 0, 1), nroots=2, tag='SA', weights=[0.5, 0.5], bond_dims=[50, 50, 50, 100, 100, 100], noises=[1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 0], dav_thrds=[1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-9], n_steps=10, conv_tol=1e-8)
        print('raw_sidmrg_ok', energies)
        """
    )

    driver_multiroot = textwrap.dedent(
        f"""
        import numpy as np
        from pyscf import ao2mo, fci, gto, mcscf, scf
        from pyblock2.driver.core import DMRGDriver, MPOAlgorithmTypes, SymmetryTypes

        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', spin=0, charge=0, symmetry=False, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        cas = mcscf.CASCI(mf, 2, 2)
        h1e, ecore = cas.get_h1cas()
        g2e = ao2mo.restore(1, cas.get_h2cas(), 2)

        cisolver = fci.direct_spin0.FCI()
        fci_e, _ = cisolver.kernel(h1e, g2e, 2, (1, 1), ecore=ecore, nroots=2)

        driver = DMRGDriver(
            scratch={str(scratch / 'driver_multiroot')!r},
            clean_scratch=True,
            stack_mem=200000000,
            n_threads=1,
            symm_type=SymmetryTypes.SU2,
        )
        try:
            driver.initialize_system(n_sites=2, n_elec=2, spin=0, orb_sym=[0, 0])
            mpo = driver.get_qc_mpo(
                np.asarray(h1e),
                np.asarray(g2e),
                ecore=float(ecore),
                algo_type=MPOAlgorithmTypes.NC,
                symmetrize=False,
                iprint=0,
            )
            ket = driver.get_random_mps('KET', bond_dim=50, nroots=2, dot=2)
            energies = driver.dmrg(
                mpo,
                ket,
                n_sweeps=8,
                tol=1e-8,
                bond_dims=[20, 30, 50, 50],
                noises=[1e-5, 1e-6, 0.0, 0.0],
                thrds=[1e-6, 1e-7, 1e-8, 1e-8],
                iprint=0,
                dav_max_iter=200,
            )
            root0 = driver.split_mps(ket, 0, 'ROOT0')
            root1 = driver.split_mps(ket, 1, 'ROOT1')
            dm0 = np.asarray(driver.get_1pdm(root0))
            dm1 = np.asarray(driver.get_1pdm(root1))
            tdm01 = np.asarray(driver.get_trans_1pdm(root0, root1))
            print('driver_multiroot_ok', np.asarray(energies).tolist(), np.asarray(fci_e).tolist())
            print('driver_pdm_shapes', dm0.shape, dm1.shape, tdm01.shape)
            print('driver_fci_error', float(np.max(np.abs(np.asarray(energies) - np.asarray(fci_e)))))
            print('driver_overlap_matrix', [[float(driver.expectation(x, driver.get_identity_mpo(), y)) for y in (root0, root1)] for x in (root0, root1)])
        finally:
            driver.finalize()
        """
    )

    driver_solver_sa = textwrap.dedent(
        f"""
        import numpy as np
        from pyscf import fci, gto, mcscf, scf
        from dmrg_sharc_bridge import DriverMultiRootDMRGCI

        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', spin=0, charge=0, symmetry=False, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.max_cycle_macro = 3
        mc.chkfile = None
        mc.chk_ci = False
        mc.dump_chk = lambda *args, **kwargs: None
        solver = DriverMultiRootDMRGCI(mf)
        solver.dmrg_args['startM'] = 20
        solver.dmrg_args['maxM'] = 50
        solver.dmrg_args['sweep_tol'] = 1e-7
        solver.dmrg_args['nsteps'] = 8
        solver.dmrg_args['memory'] = int(2e8)
        solver.dmrg_args['scratch_root'] = {str(scratch / 'driver_solver_sa')!r}
        solver.dmrg_args['dav_max_iter'] = 200
        mc.fcisolver = solver
        mc.state_average_([0.5, 0.5])
        mc.kernel()
        e_states = np.asarray(mc.fcisolver.e_states)
        print('driver_solver_sa_ok', float(mc.e_tot), e_states.tolist())
        print('driver_solver_rdm_trace', float(np.trace(mc.fcisolver.make_rdm1(mc.ci, 2, (1, 1)))))
        """
    )

    ethylene_driver_fci = textwrap.dedent(
        f"""
        import numpy as np
        from pyscf import fci, gto, mcscf, scf
        from dmrg_sharc_bridge import DriverMultiRootDMRGCI

        mol = gto.M(
            atom='''
            C  0.000000  0.000000  0.669500
            C  0.000000  0.000000 -0.669500
            H  0.000000  0.928900  1.232100
            H  0.000000 -0.928900  1.232100
            H  0.000000  0.928900 -1.232100
            H  0.000000 -0.928900 -1.232100
            ''',
            basis='sto-3g',
            spin=0,
            charge=0,
            symmetry=False,
            verbose=0,
        )
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        cas = mcscf.CASCI(mf, 2, 2)
        h1e, ecore = cas.get_h1cas()
        g2e = cas.get_h2cas()
        fcisolver = fci.direct_spin0.FCI()
        fci_e, fci_ci = fcisolver.kernel(h1e, g2e, 2, (1, 1), ecore=ecore, nroots=2)

        solver = DriverMultiRootDMRGCI(mf)
        solver.dmrg_args['startM'] = 20
        solver.dmrg_args['maxM'] = 80
        solver.dmrg_args['sweep_tol'] = 1e-8
        solver.dmrg_args['nsteps'] = 10
        solver.dmrg_args['memory'] = int(3e8)
        solver.dmrg_args['scratch_root'] = {str(scratch / 'ethylene_driver_fci')!r}
        solver.dmrg_args['dav_max_iter'] = 300
        dmrg_e, dmrg_states = solver.kernel(h1e, g2e, 2, (1, 1), ecore=ecore, nroots=2)

        dmrg_dm0 = solver.make_rdm1(dmrg_states[0], 2, (1, 1))
        dmrg_dm1 = solver.make_rdm1(dmrg_states[1], 2, (1, 1))
        dmrg_tdm = solver.trans_rdm1(dmrg_states[0], dmrg_states[1], 2, (1, 1))
        dmrg_ovlp = np.array([[solver.overlap(x, y) for y in dmrg_states] for x in dmrg_states], dtype=float)

        fci_dm0 = fcisolver.make_rdm1(fci_ci[0], 2, (1, 1))
        fci_dm1 = fcisolver.make_rdm1(fci_ci[1], 2, (1, 1))
        fci_tdm = fcisolver.trans_rdm1(fci_ci[0], fci_ci[1], 2, (1, 1))

        mo_cas = cas.mo_coeff[:, cas.ncore:cas.ncore + cas.ncas]
        with mol.with_common_origin((0, 0, 0)):
            ao_r = mol.intor('int1e_r')
        cas_r = np.einsum('xuv,up,vq->xpq', ao_r, mo_cas, mo_cas)
        dmrg_tdip = -np.einsum('xpq,qp->x', cas_r, dmrg_tdm)
        fci_tdip = -np.einsum('xpq,qp->x', cas_r, fci_tdm)
        tdip_err = min(
            np.linalg.norm(dmrg_tdip - fci_tdip),
            np.linalg.norm(dmrg_tdip + fci_tdip),
        )

        print('ethylene_driver_fci_ok', np.asarray(dmrg_e).tolist(), np.asarray(fci_e).tolist())
        print('ethylene_energy_error', float(np.max(np.abs(np.asarray(dmrg_e) - np.asarray(fci_e)))))
        print('ethylene_rdm_errors', float(np.linalg.norm(dmrg_dm0 - fci_dm0)), float(np.linalg.norm(dmrg_dm1 - fci_dm1)))
        print('ethylene_transition_dipole_error_phase_free', float(tdip_err))
        print('ethylene_overlap_error', float(np.linalg.norm(dmrg_ovlp - np.eye(2))))
        """
    )

    fd_gradient_h2 = textwrap.dedent(
        f"""
        import numpy as np
        from pyscf import gto, mcscf, scf
        from dmrg_sharc_bridge import HybridDMRGSharcSolver

        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', spin=0, charge=0, symmetry=False, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        base = mcscf.CASSCF(mf, 2, 2).state_average([0.5, 0.5])
        base.max_cycle_macro = 3
        base.chkfile = None
        base.chk_ci = False
        base.dump_chk = lambda *args, **kwargs: None
        base.kernel()
        template = {{
            'roots': [2],
            'dmrg-ncas': 2,
            'dmrg-nelecas': 2,
            'dmrg-startm': 20,
            'dmrg-maxm': 50,
            'dmrg-sweep-tol': 1e-7,
            'dmrg-memory-mb': 500,
            'dmrg-nsteps': 8,
            'dmrg-grad-mode': 'finite-diff',
            'dmrg-fd-step': 1e-3,
            'grad-max-cycle': 20,
            'verbose': 0,
            'fix-spin-shift': 0.2,
            'conv-tol': 1e-8,
            'conv-tol-grad': 1e-5,
            'max-stepsize': 0.02,
            'max-cycle-macro': 10,
            'max-cycle-micro': 4,
            'ah-level-shift': 1e-8,
            'ah-conv-tol': 1e-12,
            'ah-max-cycle': 20,
            'ah-lindep': 1e-14,
            'ah-start-tol': 2.5,
            'ah-start-cycle': 3,
        }}
        hybrid = HybridDMRGSharcSolver(
            base,
            {{'template': template, 'scratchdir': {str(scratch / 'fd_gradient_h2')!r}, 'memory': 1000}},
        )
        grad = hybrid.nuc_grad_method().kernel(state=0)
        print('fd_gradient_h2_ok', grad.shape, float(np.linalg.norm(grad.sum(axis=0))))
        """
    )

    fd_gap_gradient_h2 = textwrap.dedent(
        f"""
        import numpy as np
        from pyscf import gto, mcscf, scf
        from dmrg_sharc_bridge import HybridDMRGSharcSolver

        mol = gto.M(atom='H 0 0 -0.37; H 0 0 0.37', basis='sto-3g', spin=0, charge=0, symmetry=False, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        base = mcscf.CASSCF(mf, 2, 2).state_average([0.5, 0.5])
        base.max_cycle_macro = 3
        base.chkfile = None
        base.chk_ci = False
        base.dump_chk = lambda *args, **kwargs: None
        base.kernel()
        template = {{
            'roots': [2],
            'dmrg-ncas': 2,
            'dmrg-nelecas': 2,
            'dmrg-startm': 20,
            'dmrg-maxm': 50,
            'dmrg-sweep-tol': 1e-7,
            'dmrg-memory-mb': 500,
            'dmrg-nsteps': 8,
            'dmrg-grad-mode': 'finite-diff',
            'dmrg-fd-step': 2e-3,
            'grad-max-cycle': 20,
            'verbose': 0,
            'fix-spin-shift': 0.2,
            'conv-tol': 1e-8,
            'conv-tol-grad': 1e-5,
            'max-stepsize': 0.02,
            'max-cycle-macro': 10,
            'max-cycle-micro': 4,
            'ah-level-shift': 1e-8,
            'ah-conv-tol': 1e-12,
            'ah-max-cycle': 20,
            'ah-lindep': 1e-14,
            'ah-start-tol': 2.5,
            'ah-start-cycle': 3,
        }}
        hybrid = HybridDMRGSharcSolver(
            base,
            {{'template': template, 'scratchdir': {str(scratch / 'fd_gap_gradient_h2')!r}, 'memory': 1000}},
        )
        grad_obj = hybrid.nuc_grad_method()
        all_grads = grad_obj.kernel_all()
        gap_grad = all_grads[1] - all_grads[0]
        gap_grad_direct = grad_obj.gap_gradient((0, 1))
        print('fd_gap_gradient_h2_ok', all_grads.shape, gap_grad.shape)
        print('fd_gap_gradient_consistency', float(np.linalg.norm(gap_grad - gap_grad_direct)))
        print('fd_gap_gradient_translation', float(np.linalg.norm(gap_grad.sum(axis=0))))
        """
    )

    results = [
        run_case("single_root_dmrgscf", single_root),
        run_case("experimental_sa_dmrg", experimental_multiroot),
        run_case("raw_sidmrg_multiroot", raw_sidmrg),
        run_case("driver_multiroot_dmrg", driver_multiroot),
        run_case("driver_solver_sa_casscf", driver_solver_sa),
        run_case("ethylene_driver_fci", ethylene_driver_fci),
        run_case("fd_gradient_h2", fd_gradient_h2),
        run_case("fd_gap_gradient_h2", fd_gap_gradient_h2),
    ]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
