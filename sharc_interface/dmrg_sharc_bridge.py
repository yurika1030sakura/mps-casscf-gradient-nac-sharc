#!/usr/bin/env python3
"""Local bridge code for experimental DMRG/PySCF/SHARC integration.

This module intentionally separates two use cases:

1. ``ExperimentalMultiRootDMRGCI``:
   An experimental multi-root DMRG FCI solver built on ``pyblock2.sidmrg``.
   It is intended to unblock exploratory SA-DMRG-CASSCF energy/orbital tests
   in the local environment where the stock ``pyblock2.dmrgscf.DMRGCI``
   supports only single-root MPS objects.

2. ``DriverMultiRootDMRGCI``:
   A second experimental multi-root solver built on ``pyblock2.driver``.
   This is the preferred route in the local environment because the lower-level
   ``sidmrg`` multi-root path currently segfaults on small subprocess tests.

3. ``HybridDMRGSharcSolver``:
   A pragmatic SHARC-facing wrapper. It keeps a conventional SA-CASSCF
   reference solver for gradients/NACs, and overlays multi-root DMRG-CASCI
   energies (and, when available, transition 1-RDM dipoles) on top of those
   orbitals. This is not a fully self-consistent DMRG-SHARC implementation,
   but it creates a usable bridge for mechanism screening while the true
   multi-state DMRG gradient/NAC machinery remains a research project.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from block2 import OpNames, OpNamesSet, Random
from block2.su2 import Expect, MovingEnvironment, PDM2MPOQC, RuleQC, SimplifiedMPO
from pyscf import ao2mo, lib, mcscf, scf
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.mcscf import avas
from pyblock2.driver.core import DMRGDriver, MPOAlgorithmTypes, SymmetryTypes
from pyblock2.sidmrg import SIDMRG, SpinLabel


def _nelecas_tuple(nelecas) -> tuple[int, int]:
    if isinstance(nelecas, tuple):
        return int(nelecas[0]), int(nelecas[1])
    return _unpack_nelec(int(nelecas), None)


def _total_active_electrons(nelecas) -> int:
    na, nb = _nelecas_tuple(nelecas)
    return na + nb


def _ncore_for_space(nelectron: int, nelecas) -> int:
    return (nelectron - _total_active_electrons(nelecas)) // 2


def _parse_csv_ints(text: str | None) -> list[int]:
    if not text:
        return []
    out = []
    for token in text.replace(",", " ").split():
        out.append(int(token))
    return out


def _parse_csv_labels(text: str | None) -> list[str]:
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def _build_dmrg_schedule(dmrg_args: dict) -> dict[str, object]:
    start_m = int(dmrg_args.get("startM", 250))
    max_m = int(dmrg_args.get("maxM", 800))
    sweep_tol = float(dmrg_args.get("sweep_tol", 1.0e-6))
    nsteps = int(dmrg_args.get("nsteps", 24))

    if max_m < start_m:
        max_m = start_m

    if start_m == max_m:
        bond_dims = [start_m, max_m]
    else:
        mid_m = max(start_m, int(round(math.sqrt(start_m * max_m))))
        bond_dims = [start_m, mid_m, max_m, max_m]

    noises = [1.0e-5, 1.0e-6, 0.0, 0.0][: len(bond_dims)]
    dav_thrds = [1.0e-6, 1.0e-7, min(1.0e-8, sweep_tol), min(1.0e-8, sweep_tol)][: len(bond_dims)]

    return {
        "bond_dims": bond_dims,
        "noises": noises,
        "dav_thrds": dav_thrds,
        "n_steps": max(nsteps, len(bond_dims)),
        "conv_tol": sweep_tol,
    }


class ExperimentalMultiRootDMRGCI(lib.StreamObject):
    """Experimental multi-root DMRG solver compatible with PySCF state averaging.

    This class is deliberately scoped to the pieces required for SA-CASSCF
    orbital optimization: multi-root energies plus diagonal 1-/2-RDMs.
    It does not yet implement the explicit CI-response machinery needed for a
    full analytic SA-DMRG-CASSCF gradient/NAC stack.
    """

    def __init__(self, mf):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.converged = False
        self.e_tot = None
        self.e_states = None
        self.nroots = 1
        self.weights = None
        self.wfnsym = None
        self.dmrg_args = {
            "startM": 250,
            "maxM": 800,
            "sweep_tol": 1.0e-6,
            "nsteps": 24,
            "memory": int(lib.param.MAX_MEMORY * 1e6),
            "scratch_root": None,
        }
        self._sidmrg = None
        self._mpss = None
        self._kernel_count = 0
        self._active_scratch = None

    def copy(self):
        obj = self.__class__(self._scf)
        obj.verbose = self.verbose
        obj.dmrg_args = dict(self.dmrg_args)
        return obj

    def dump_flags(self, verbose=None):
        return self

    def _cleanup(self) -> None:
        if self._sidmrg is not None:
            try:
                self._sidmrg.__del__()
            except Exception:
                pass
        self._sidmrg = None
        self._mpss = None

    def _scratch_dir(self) -> Path:
        root = self.dmrg_args.get("scratch_root")
        if root:
            base = Path(root).resolve()
        else:
            base = Path.cwd() / "dmrg_sa_scratch"
        base.mkdir(parents=True, exist_ok=True)
        scratch = base / f"kernel_{self._kernel_count:04d}"
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        self._active_scratch = scratch
        return scratch

    def kernel(self, h1e, g2e, norb, nelec, ecore=0, ci0=None, **kwargs):
        del ci0
        kwargs.pop("wfnsym", None)

        self._cleanup()
        self._kernel_count += 1

        na, nb = _nelecas_tuple(nelec)
        nroots = int(kwargs.get("nroots", getattr(self, "nroots", 1)))
        weights = getattr(self, "weights", None)
        if weights is not None:
            weights = list(weights)
        elif nroots > 1:
            weights = [1.0 / nroots] * nroots

        schedule = _build_dmrg_schedule(self.dmrg_args)
        scratch = self._scratch_dir()

        sidmrg = SIDMRG(
            scratch=str(scratch),
            memory=float(self.dmrg_args["memory"]),
            omp_threads=lib.num_threads(),
            verbose=max(0, min(int(self.verbose) - 4, 2)),
        )
        g2e = ao2mo.restore(1, g2e, norb)
        sidmrg.init_hamiltonian(
            "c1",
            n_sites=norb,
            n_elec=na + nb,
            twos=na - nb,
            isym=1,
            orb_sym=[1] * norb,
            e_core=ecore,
            h1e=np.asarray(h1e),
            g2e=np.asarray(g2e),
            save_fcidump=str(scratch / "FCIDUMP"),
        )
        energies = sidmrg.dmrg(
            target=SpinLabel(na + nb, na - nb, 1),
            nroots=nroots,
            weights=weights,
            tag="SA",
            **schedule,
        )
        mpss = list(sidmrg.prepare_mps(tags=["SA"]))

        self._sidmrg = sidmrg
        self._mpss = mpss
        self.nroots = nroots
        self.e_states = np.asarray(energies, dtype=float)
        self.e_tot = float(np.dot(self.e_states, weights)) if weights is not None else float(self.e_states[0])
        self.converged = True

        if nroots == 1:
            return float(self.e_states[0]), mpss[0]
        return self.e_states.copy(), mpss

    def _state(self, state=None):
        if state is None:
            if not self._mpss:
                raise RuntimeError("No DMRG state available. Run kernel() first.")
            return self._mpss[0]
        return state

    def _make_diag_rdm1_raw(self, state) -> np.ndarray:
        dm = self._sidmrg.onepdm([state], soc=False, has_tran=False)[0]
        return np.asarray(dm).T.copy()

    def _make_diag_rdm2_raw(self, state) -> np.ndarray:
        pmpo = PDM2MPOQC(self._sidmrg.hamil, 0)
        pmpo = SimplifiedMPO(pmpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        pme = MovingEnvironment(pmpo, state, state, "2PDM")
        pme.delayed_contraction = OpNamesSet.normal_ops()
        pme.cached_contraction = True
        pme.save_partition_info = True
        pme.init_environments(False)
        expect = Expect(pme, state.info.bond_dim + 100, state.info.bond_dim + 100)
        expect.iprint = max(0, min(int(self.verbose) - 4, 2))
        expect.solve(True, state.center == 0)
        dmr = expect.get_2pdm_spatial(self._sidmrg.n_sites)
        dm = np.array(dmr, copy=True)
        dmr.deallocate()
        pmpo.deallocate()
        return dm.transpose((0, 3, 1, 2))

    def make_rdm1(self, state=None, norb=None, nelec=None):
        del norb, nelec
        return self._make_diag_rdm1_raw(self._state(state))

    def make_rdm2(self, state=None, norb=None, nelec=None):
        del norb, nelec
        return self._make_diag_rdm2_raw(self._state(state))

    def make_rdm12(self, state=None, norb=None, nelec=None):
        del norb, nelec
        state = self._state(state)
        return self._make_diag_rdm1_raw(state), self._make_diag_rdm2_raw(state)

    def trans_rdm1(self, state_bra, state_ket, norb=None, nelec=None):
        del norb, nelec
        pdm = self._sidmrg.trans_onepdm([state_bra, state_ket], soc=False, has_tran=True)
        return np.asarray(pdm[0, 1]).T.copy()

    def __del__(self):
        self._cleanup()


class DriverMultiRootDMRGCI(lib.StreamObject):
    """Experimental multi-root DMRG solver using the higher-level driver API."""

    def __init__(self, mf):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.converged = False
        self.e_tot = None
        self.e_states = None
        self.nroots = 1
        self.weights = None
        self.wfnsym = None
        self.dmrg_args = {
            "startM": 250,
            "maxM": 800,
            "sweep_tol": 1.0e-6,
            "nsteps": 24,
            "memory": int(lib.param.MAX_MEMORY * 1e6),
            "scratch_root": None,
            "mpo_algo": "NC",
            "dav_max_iter": 800,
            "n_threads": 1,
            "random_seed": 123456,
            "refine_split_roots": True,
            "refine_sweeps": 20,
            "refine_sweep_tol": 1.0e-8,
            "refine_proj_weight": 5.0,
        }
        self._driver = None
        self._ket = None
        self._states = None
        self._kernel_count = 0
        self._active_scratch = None

    def copy(self):
        obj = self.__class__(self._scf)
        obj.verbose = self.verbose
        obj.dmrg_args = dict(self.dmrg_args)
        return obj

    def dump_flags(self, verbose=None):
        return self

    def _cleanup(self) -> None:
        if self._driver is not None:
            try:
                self._driver.finalize()
            except Exception:
                pass
        self._driver = None
        self._ket = None
        self._states = None

    def _scratch_dir(self) -> Path:
        root = self.dmrg_args.get("scratch_root")
        if root:
            base = Path(root).resolve()
        else:
            base = Path.cwd() / "dmrg_driver_scratch"
        base.mkdir(parents=True, exist_ok=True)
        scratch = base / f"kernel_{self._kernel_count:04d}"
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        self._active_scratch = scratch
        return scratch

    def _mpo_algo(self):
        name = str(self.dmrg_args.get("mpo_algo", "NC")).strip()
        if not name or name.lower() == "default":
            return None
        return getattr(MPOAlgorithmTypes, name)

    def kernel(self, h1e, g2e, norb, nelec, ecore=0, ci0=None, **kwargs):
        del ci0
        kwargs.pop("wfnsym", None)

        self._cleanup()
        self._kernel_count += 1

        na, nb = _nelecas_tuple(nelec)
        nroots = int(kwargs.get("nroots", getattr(self, "nroots", 1)))
        weights = getattr(self, "weights", None)
        if weights is not None:
            weights = list(weights)
        elif nroots > 1:
            weights = [1.0 / nroots] * nroots

        schedule = _build_dmrg_schedule(self.dmrg_args)
        scratch = self._scratch_dir()
        g2e = ao2mo.restore(1, g2e, norb)

        driver = DMRGDriver(
            scratch=str(scratch),
            clean_scratch=True,
            stack_mem=int(self.dmrg_args["memory"]),
            n_threads=int(self.dmrg_args.get("n_threads", 1)),
            symm_type=SymmetryTypes.SU2,
        )
        driver.initialize_system(
            n_sites=norb,
            n_elec=na + nb,
            spin=abs(na - nb),
            orb_sym=[0] * norb,
        )
        mpo = driver.get_qc_mpo(
            np.asarray(h1e),
            np.asarray(g2e),
            ecore=float(ecore),
            algo_type=self._mpo_algo(),
            symmetrize=False,
            iprint=max(0, min(int(self.verbose) - 4, 2)),
        )
        random_seed = self.dmrg_args.get("random_seed")
        if random_seed is not None:
            Random.rand_seed(int(random_seed))
        ket = driver.get_random_mps(
            "SA" if nroots > 1 else "ROOT0",
            bond_dim=int(self.dmrg_args.get("startM", 250)),
            nroots=nroots,
            dot=2,
        )
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=int(schedule["n_steps"]),
            tol=float(schedule["conv_tol"]),
            bond_dims=list(schedule["bond_dims"]),
            noises=list(schedule["noises"]),
            thrds=list(schedule["dav_thrds"]),
            iprint=max(0, min(int(self.verbose) - 4, 2)),
            dav_max_iter=int(self.dmrg_args.get("dav_max_iter", 800)),
        )

        if nroots == 1:
            states = [ket]
            energies = np.asarray([energies], dtype=float)
        else:
            states = [driver.split_mps(ket, iroot, f"ROOT{iroot}") for iroot in range(nroots)]
            energies = np.asarray(energies, dtype=float)
            if bool(self.dmrg_args.get("refine_split_roots", True)):
                refined = []
                refined_expectations = []
                ns_ref = max(1, int(self.dmrg_args.get("refine_sweeps", 20)))
                tol_ref = float(self.dmrg_args.get("refine_sweep_tol", 1.0e-8))
                for iroot, state in enumerate(states):
                    mps = driver.copy_mps(state, tag=f"ROOTR{iroot}")
                    driver.dmrg(
                        mpo,
                        mps,
                        n_sweeps=ns_ref,
                        tol=tol_ref,
                        bond_dims=[int(self.dmrg_args.get("maxM", 800))] * ns_ref,
                        noises=[0.0] * ns_ref,
                        thrds=[float(schedule["dav_thrds"][-1])] * ns_ref,
                        iprint=max(0, min(int(self.verbose) - 4, 2)),
                        dav_max_iter=int(self.dmrg_args.get("dav_max_iter", 800)),
                        proj_mpss=refined or None,
                        proj_weights=(
                            [float(self.dmrg_args.get("refine_proj_weight", 5.0))]
                            * len(refined)
                            if refined else None
                        ),
                    )
                    refined.append(mps)
                    refined_expectations.append(
                        float(driver.expectation(mps, mpo, mps, iprint=0))
                    )
                states = refined
                energies = np.asarray(refined_expectations, dtype=float)

        self._driver = driver
        self._ket = ket
        self._states = states
        self.nroots = nroots
        self.e_states = energies
        self.e_tot = float(np.dot(self.e_states, weights)) if weights is not None else float(self.e_states[0])
        self.converged = True

        if nroots == 1:
            return float(self.e_states[0]), states[0]
        return self.e_states.copy(), states

    def _state(self, state=None):
        if state is None:
            if not self._states:
                raise RuntimeError("No DMRG state available. Run kernel() first.")
            return self._states[0]
        return state

    def make_rdm1(self, state=None, norb=None, nelec=None):
        del norb, nelec
        dm = self._driver.get_1pdm(self._state(state))
        return np.asarray(dm).T.copy()

    def make_rdm2(self, state=None, norb=None, nelec=None):
        del norb, nelec
        dm = self._driver.get_2pdm(self._state(state))
        return np.asarray(dm).transpose((0, 3, 1, 2)).copy()

    def make_rdm12(self, state=None, norb=None, nelec=None):
        del norb, nelec
        state = self._state(state)
        dm1 = np.asarray(self._driver.get_1pdm(state)).T.copy()
        dm2 = np.asarray(self._driver.get_2pdm(state)).transpose((0, 3, 1, 2)).copy()
        return dm1, dm2

    def trans_rdm1(self, state_bra, state_ket, norb=None, nelec=None):
        del norb, nelec
        pdm = self._driver.get_trans_1pdm(state_bra, state_ket)
        return np.asarray(pdm).T.copy()

    def trans_rdm12(self, state_bra, state_ket, norb=None, nelec=None):
        """Return (transition 1-RDM, transition 2-RDM) between roots.

        Index conventions match pyscf.fci.direct_spin1.trans_rdm12:
        - dm1[p, q] = <bra| E_pq |ket>            (Mulliken 1-particle)
        - dm2[p, q, r, s] = <bra| E_pq E_rs - delta_qr E_ps |ket>
                          = <bra| a_p^+ a_r^+ a_s a_q |ket>   (Mulliken 12,34)

        pyblock2 returns 2-RDM in physicist (1,2,3,4) ordering when sliced as
        (i,j,k,l) -> <i^+ j^+ l k>; we transpose to chemist Mulliken order to
        match PySCF's internal convention.
        """
        del norb, nelec
        dm1 = np.asarray(self._driver.get_trans_1pdm(state_bra, state_ket)).T.copy()
        dm2_pyblock = np.asarray(self._driver.get_trans_2pdm(state_bra, state_ket))
        # pyblock2 default 2pdm layout: (p, q, r, s) ~ <p^+ q^+ s r> physicist
        # PySCF Mulliken (12|34): dm2_chem[p,q,r,s] = <p^+ r^+ s q>
        # transpose pattern (0,3,1,2) sends physicist -> chemist Mulliken
        dm2 = dm2_pyblock.transpose((0, 3, 1, 2)).copy()
        return dm1, dm2

    def overlap(self, state_bra, state_ket):
        impo = self._driver.get_identity_mpo()
        return self._driver.expectation(state_bra, impo, state_ket)

    def __del__(self):
        self._cleanup()


class FiniteDifferenceDMRGGradient:
    """Numerical gradient of the DMRG overlay energy.

    This is intentionally a validation fallback, not a production dynamics
    engine. It recomputes RHF, SA-CASSCF orbitals, and DMRG overlay energies at
    each displaced geometry.
    """

    def __init__(self, hybrid_solver):
        self.hybrid_solver = hybrid_solver
        self.converged = False
        self.max_cycle = int(hybrid_solver.template.get("grad-max-cycle", 50))
        self._counter = 0

    @property
    def parent(self):
        return self.hybrid_solver

    def _build_mol(self, coords_bohr):
        mol = self.parent.mol.copy()
        mol.set_geom_(coords_bohr, unit="Bohr")
        mol.build(False, False)
        return mol

    def _build_base_solver(self, mol):
        template = self.parent.template
        mf = scf.RHF(mol)
        mf.verbose = max(0, min(int(template.get("verbose", 0)) - 3, 2))
        mf.max_memory = int(self.parent.qmin.get("memory", 4000))
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF failed to converge in finite-difference DMRG gradient")

        base = mcscf.CASSCF(mf, self.parent.base_solver.ncas, self.parent.base_solver.nelecas)
        try:
            base.fix_spin_(ss=0, shift=template.get("fix-spin-shift", 0.2))
        except Exception:
            pass

        nroots = int(template["roots"][0])
        weights = [1.0 / nroots] * nroots
        base = base.state_average(weights)

        base.conv_tol = template["conv-tol"]
        base.conv_tol_grad = template["conv-tol-grad"]
        base.max_stepsize = template["max-stepsize"]
        base.max_cycle_macro = template["max-cycle-macro"]
        base.max_cycle_micro = template["max-cycle-micro"]
        base.ah_level_shift = template["ah-level-shift"]
        base.ah_conv_tol = template["ah-conv-tol"]
        base.ah_max_cycle = template["ah-max-cycle"]
        base.ah_lindep = template["ah-lindep"]
        base.ah_start_tol = template["ah-start-tol"]
        base.ah_start_cycle = template["ah-start-cycle"]
        base.chkfile = None
        base.chk_ci = False
        base.dump_chk = lambda *args, **kwargs: None

        init_mo = None
        if getattr(self.parent.base_solver, "mo_coeff", None) is not None:
            try:
                init_mo = mcscf.project_init_guess(base, self.parent.base_solver.mo_coeff)
            except Exception:
                init_mo = None
        base.kernel(init_mo)
        if not base.converged:
            raise RuntimeError("SA-CASSCF failed to converge in finite-difference DMRG gradient")
        return base

    def _dmrg_energies_at(self, coords_bohr):
        self._counter += 1
        cfg = self.parent._overlay_config()
        mol = self._build_mol(coords_bohr)
        base = self._build_base_solver(mol)
        overlay_ncas, overlay_nelecas, overlay_mo = self.parent._select_overlay_mos(cfg, base)

        casci = mcscf.CASCI(base._scf, overlay_ncas, overlay_nelecas)
        casci.mo_coeff = overlay_mo
        h1e, e_core = casci.get_h1cas()
        g2e = casci.get_h2cas()

        overlay = DriverMultiRootDMRGCI(base._scf)
        overlay.verbose = max(0, min(int(self.parent.verbose), 4))
        overlay.dmrg_args.update(
            {
                "startM": cfg["startM"],
                "maxM": cfg["maxM"],
                "sweep_tol": cfg["sweep_tol"],
                "nsteps": cfg["nsteps"],
                "memory": cfg["memory"],
                "scratch_root": str(Path(self.parent.qmin["scratchdir"]) / "dmrg_fd_grad" / f"point_{self._counter:05d}"),
            }
        )
        nroots = int(self.parent.template["roots"][0])
        energies, _ = overlay.kernel(
            h1e,
            g2e,
            overlay_ncas,
            overlay_nelecas,
            ecore=e_core,
            nroots=nroots,
        )
        if nroots == 1:
            energies = np.asarray([energies], dtype=float)
        return np.asarray(energies, dtype=float)

    def kernel(self, state=0):
        step = float(self.parent.template.get("dmrg-fd-step", 1.0e-3))
        coords = np.array(self.parent.mol.atom_coords(unit="Bohr"), dtype=float)
        grad = np.zeros_like(coords)

        for iatom in range(coords.shape[0]):
            for idir in range(3):
                plus = coords.copy()
                minus = coords.copy()
                plus[iatom, idir] += step
                minus[iatom, idir] -= step
                e_plus = self._dmrg_energies_at(plus)[state]
                e_minus = self._dmrg_energies_at(minus)[state]
                grad[iatom, idir] = (e_plus - e_minus) / (2.0 * step)

        self.converged = True
        return grad

    def kernel_all(self):
        """Return finite-difference DMRG gradients for all overlay roots.

        This shares the expensive displaced DMRG calculations across states.
        It is intended for MECI/seam diagnostics, especially the gradient of
        the DMRG energy gap. It is still a numerical validation tool, not a
        production dynamics gradient implementation.
        """
        step = float(self.parent.template.get("dmrg-fd-step", 1.0e-3))
        nroots = int(self.parent.template["roots"][0])
        coords = np.array(self.parent.mol.atom_coords(unit="Bohr"), dtype=float)
        grads = np.zeros((nroots,) + coords.shape)

        for iatom in range(coords.shape[0]):
            for idir in range(coords.shape[1]):
                plus = coords.copy()
                minus = coords.copy()
                plus[iatom, idir] += step
                minus[iatom, idir] -= step
                e_plus = self._dmrg_energies_at(plus)
                e_minus = self._dmrg_energies_at(minus)
                grads[:, iatom, idir] = (e_plus - e_minus) / (2.0 * step)

        self.converged = True
        return grads

    def gap_gradient(self, state_pair=(0, 1)):
        """Return finite-difference gradient of a DMRG energy gap.

        ``state_pair`` is zero-based and ordered as ``(lower, upper)``.
        The returned vector is ``grad(E_upper - E_lower)`` in Hartree/Bohr.
        This is the DMRG analogue of the MECI branching-space g vector.
        """
        lower, upper = state_pair
        grads = self.kernel_all()
        return grads[upper] - grads[lower]


class HybridDMRGSharcSolver:
    """SHARC-facing hybrid solver with SA-CASSCF gradients/NACs and DMRG energies."""

    def __init__(self, base_solver, qmin):
        self.base_solver = base_solver
        self.qmin = qmin
        self.template = qmin["template"]
        self.verbose = base_solver.verbose
        self.stdout = base_solver.stdout
        self.mol = base_solver.mol
        self._scf = base_solver._scf
        self._overlay_solver = None
        self._overlay_mo_coeff = None
        self._overlay_ncore = None
        self._overlay_ncas = None
        self._overlay_nelecas = None
        self._overlay_states = None
        self._overlay_energies = None
        self.ci = None
        self.e_states = None
        self.e_tot = None
        self.converged = False
        self.chkfile = getattr(base_solver, "chkfile", None)
        self.chk_ci = getattr(base_solver, "chk_ci", False)

    def __getattr__(self, name):
        return getattr(self.base_solver, name)

    def update(self, chkfile):
        return self.base_solver.update(chkfile)

    def _overlay_config(self) -> dict[str, object]:
        return {
            "ncas": int(self.template.get("dmrg-ncas", self.base_solver.ncas)),
            "nelecas": int(self.template.get("dmrg-nelecas", _total_active_electrons(self.base_solver.nelecas))),
            "cas_list": _parse_csv_ints(self.template.get("dmrg-cas-list")),
            "avas_labels": _parse_csv_labels(self.template.get("dmrg-avas-labels")),
            "avas_threshold": float(self.template.get("dmrg-avas-threshold", 0.20)),
            "startM": int(self.template.get("dmrg-startm", 250)),
            "maxM": int(self.template.get("dmrg-maxm", 800)),
            "sweep_tol": float(self.template.get("dmrg-sweep-tol", 1.0e-6)),
            "nsteps": int(self.template.get("dmrg-nsteps", 24)),
            "memory": int(self.template.get("dmrg-memory-mb", 24000)) * 1000000,
            "scratch_root": str(Path(self.qmin["scratchdir"]) / "dmrg_overlay"),
        }

    def _select_overlay_mos(self, cfg: dict[str, object], base_solver=None):
        if base_solver is None:
            base_solver = self.base_solver
        overlay_ncas = int(cfg["ncas"])
        overlay_nelecas = int(cfg["nelecas"])
        cas_list = cfg["cas_list"]
        avas_labels = cfg["avas_labels"]

        if cas_list:
            if overlay_ncas != len(cas_list):
                overlay_ncas = len(cas_list)
            mo_coeff = mcscf.sort_mo(
                base_solver,
                base_solver.mo_coeff,
                cas_list,
                base=1,
            )
            return overlay_ncas, overlay_nelecas, mo_coeff

        if avas_labels:
            with lib.temporary_env(base_solver._scf, mo_coeff=base_solver.mo_coeff):
                overlay_ncas, overlay_nelecas, mo_coeff = avas.avas(
                    base_solver._scf,
                    avas_labels,
                    threshold=float(cfg["avas_threshold"]),
                    canonicalize=True,
                )
            return int(overlay_ncas), int(overlay_nelecas), mo_coeff

        return overlay_ncas, overlay_nelecas, base_solver.mo_coeff

    def _run_overlay_dmrg(self):
        cfg = self._overlay_config()
        nroots = int(self.qmin["template"]["roots"][0])
        overlay_ncas, overlay_nelecas, overlay_mo = self._select_overlay_mos(cfg)
        overlay_ncore = _ncore_for_space(self.mol.nelectron, overlay_nelecas)

        casci = mcscf.CASCI(self.base_solver._scf, overlay_ncas, overlay_nelecas)
        casci.mo_coeff = overlay_mo
        h1e, e_core = casci.get_h1cas()
        g2e = casci.get_h2cas()

        overlay = DriverMultiRootDMRGCI(self.base_solver._scf)
        overlay.verbose = self.verbose
        overlay.dmrg_args.update(
            {
                "startM": cfg["startM"],
                "maxM": cfg["maxM"],
                "sweep_tol": cfg["sweep_tol"],
                "nsteps": cfg["nsteps"],
                "memory": cfg["memory"],
                "scratch_root": cfg["scratch_root"],
            }
        )
        energies, states = overlay.kernel(
            h1e,
            g2e,
            overlay_ncas,
            overlay_nelecas,
            ecore=e_core,
            nroots=nroots,
        )
        if nroots == 1:
            energies = np.asarray([energies], dtype=float)
            states = [states]

        self._overlay_solver = overlay
        self._overlay_mo_coeff = overlay_mo
        self._overlay_ncore = overlay_ncore
        self._overlay_ncas = overlay_ncas
        self._overlay_nelecas = overlay_nelecas
        self._overlay_states = list(states)
        self._overlay_energies = np.asarray(energies, dtype=float)

    def _sync_reference(self):
        self.mo_coeff = self.base_solver.mo_coeff
        self.ncore = self.base_solver.ncore
        self.ncas = self.base_solver.ncas
        self.nelecas = self.base_solver.nelecas
        self.ci = self.base_solver.ci
        self.e_tot = self.base_solver.e_tot
        self.chkfile = getattr(self.base_solver, "chkfile", None)
        self.chk_ci = getattr(self.base_solver, "chk_ci", False)

    def _needs_overlay(self) -> bool:
        return "h" in self.qmin or "dm" in self.qmin

    def kernel(self, mo_coeff=None):
        self.base_solver.kernel(mo_coeff)
        self._sync_reference()
        if self._needs_overlay():
            self._run_overlay_dmrg()
            self.e_states = self._overlay_energies.copy()
            self.converged = bool(self.base_solver.converged and self._overlay_solver.converged)
        else:
            self._overlay_solver = None
            self.e_states = np.asarray(getattr(self.base_solver, "e_states", [self.base_solver.e_tot]), dtype=float)
            self.converged = bool(self.base_solver.converged)
        return self.e_states

    def nuc_grad_method(self):
        mode = str(self.template.get("dmrg-grad-mode", "base")).lower()
        if mode in {"fd", "finite", "finite-diff", "finite_difference", "dmrg-fd"}:
            return FiniteDifferenceDMRGGradient(self)
        return self.base_solver.nuc_grad_method()

    def nac_method(self):
        return self.base_solver.nac_method()

    def get_dipole_elements(self):
        if self._overlay_solver is None:
            raise RuntimeError("DMRG overlay not available. Run kernel() first.")

        nroots = len(self._overlay_states)
        dip_matrix = np.zeros((3, nroots, nroots))

        mo_core = self._overlay_mo_coeff[:, : self._overlay_ncore]
        mo_cas = self._overlay_mo_coeff[
            :,
            self._overlay_ncore : self._overlay_ncore + self._overlay_ncas,
        ]
        dm_core = 2.0 * mo_core @ mo_core.conj().T

        gauge_center = (0, 0, 0)
        charges = self.mol.atom_charges()
        coords = self.mol.atom_coords() - gauge_center
        nucl_term = charges.dot(coords)

        with self.mol.with_common_origin(gauge_center):
            dipole_ints = self.mol.intor("int1e_r")

        for state in range(nroots):
            casdm1 = self._overlay_solver.make_rdm1(self._overlay_states[state])
            dm1 = dm_core + mo_cas @ casdm1 @ mo_cas.conj().T
            dip = nucl_term - np.einsum("xij,ji->x", dipole_ints, dm1)
            dip_matrix[:, state, state] = dip

        for bra in range(nroots):
            for ket in range(bra + 1, nroots):
                t_dm = self._overlay_solver.trans_rdm1(
                    self._overlay_states[bra],
                    self._overlay_states[ket],
                )
                t_dm = mo_cas @ t_dm @ mo_cas.conj().T
                t_dip = -np.einsum("xij,ji->x", dipole_ints, t_dm)
                dip_matrix[:, bra, ket] = t_dip
                dip_matrix[:, ket, bra] = t_dip

        return dip_matrix
