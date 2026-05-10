"""Analytic CP-CASSCF gradient + NAC backends for the SHARC PySCF interface.

This module provides two response routes used by the SHARC-facing PySCF
interface.  The projected-CI route wires the validated
``CPCASSCFResponseFCI`` solver into PySCF's SA-CASSCF gradient/NAC assembly.
The MPS-Krylov route uses ``CPDMRGCASSCFResponseMPSKrylov`` so the
active-space response RHS, Krylov vectors, transition densities, and
Lagrange assembly are represented by block2 MPS objects rather than dense
determinant arrays.

Why route through PySCF's `Gradients` / `NonAdiabaticCouplings` objects
instead of re-implementing the assembly?
  - PySCF already has battle-tested code for `get_ham_response`,
    `get_LdotJnuc`, etc.  We only swap the Lagrange solver step.
  - This is exactly the validation pattern used in
    `test_step4_baseline_gradient.py::test_G3_full_gradient_matches` and
    `test_step5b_nac_baseline.py::test_N3_full_nac_matches`, both of which
    verified bit-equivalent agreement with the pyscf reference.
  - The SHARC interface is unchanged between projected-CI validation and
    MPS-Krylov response modes: only the inner matvec/RHS and Lagrange
    assembly routines change.

Entry point:

    compute_grad_nac_analytic_cp(mc, gradient_states, nac_pairs, backend=...)
        -> {"grad": {state_idx: np.ndarray (natom, 3)},
            "nac":  {(I, J): np.ndarray (natom, 3)}}

Notes for the SHARC writer:
  - `gradient_states` is a list of 0-based state indices Θ.
  - `nac_pairs` is a list of 0-based (I, J) state pairs (ket, bra). The
    returned NAC follows pyscf's sign convention: the antisymmetric pair
    (J, I) = -(I, J) is NOT filled in here -- the caller does that when
    building the SHARC NAC matrix.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
from functools import reduce

_HERE = Path(__file__).resolve().parent
for _DEV_DIR in (
    _HERE / "dmrg_analytic_dev",
    _HERE.parent / "src" / "dmrg_analytic_dev",
):
    if _DEV_DIR.exists() and str(_DEV_DIR) not in sys.path:
        sys.path.insert(0, str(_DEV_DIR))

from cp_casscf_response import CPCASSCFResponseFCI  # noqa: E402
from cp_dmrg_response_mps_krylov import CPDMRGCASSCFResponseMPSKrylov  # noqa: E402


def _make_response(mc, backend: str = "newton_casscf") -> CPCASSCFResponseFCI:
    """Construct the response solver for the given converged SA-CASSCF object.

    Honours ``mc.weights`` if present; otherwise falls back to equal weights.
    """
    weights = getattr(mc, "weights", None)
    if weights is None:
        # Older pyscf versions store weights under a different attribute on
        # state-averaged solvers; fall back to equal weights.
        nstates = len(mc.ci) if isinstance(mc.ci, list) else 1
        weights = np.full(nstates, 1.0 / nstates)
    return CPCASSCFResponseFCI(mc, weights=np.asarray(weights), backend=backend)


def _make_mps_krylov_response(mc) -> CPDMRGCASSCFResponseMPSKrylov:
    """Construct the MPS-Krylov response solver from an MPSAsFCISolver run."""
    fcisolver = getattr(mc, "fcisolver", None)
    driver = getattr(fcisolver, "_driver", None)
    mpo = getattr(fcisolver, "_mpo", None)
    if driver is None or mpo is None or getattr(fcisolver, "_kets", None) is None:
        raise RuntimeError(
            "dmrg-response-mode=mps-krylov requires a converged "
            "MPSAsFCISolver with _driver, _mpo, and _kets populated."
        )
    weights = getattr(mc, "weights", None)
    if weights is None:
        nstates = len(mc.ci) if isinstance(mc.ci, list) else 1
        weights = np.full(nstates, 1.0 / nstates)
    phase_ci_list = None
    ci = getattr(mc, "ci", None)
    if isinstance(ci, (list, tuple)) and all(
        hasattr(x, "shape") and np.asarray(x).size > 1 for x in ci
    ):
        phase_ci_list = [np.asarray(x) for x in ci]
    return CPDMRGCASSCFResponseMPSKrylov(
        mc,
        driver,
        mpo,
        mps_states=list(fcisolver._kets),
        weights=np.asarray(weights),
        m_compress=int(
            getattr(
                fcisolver,
                "response_m_compress",
                getattr(fcisolver, "bond_dim", 500),
            )
        ),
        mps_fit_sweeps=int(getattr(fcisolver, "mps_fit_sweeps", 10)),
        mps_fit_tol=float(getattr(fcisolver, "mps_fit_tol", 1.0e-10)),
        initial_guess=str(getattr(fcisolver, "response_initial_guess", "zero")),
        initial_guess_sweeps=int(
            getattr(fcisolver, "response_initial_guess_sweeps", 4)
        ),
        initial_guess_tol=float(
            getattr(fcisolver, "response_initial_guess_tol", 1.0e-6)
        ),
        initial_guess_proj_weight=float(
            getattr(fcisolver, "response_initial_guess_proj_weight", 20.0)
        ),
        linear_solver=str(getattr(fcisolver, "response_linear_solver", "gmres")),
        mps_only=True,
        phase_ci_list=phase_ci_list,
    )


def _state_energies(mc, cp=None, min_size: int = 1):
    for attr in ("e_states", "e_tot"):
        try:
            val = getattr(mc, attr)
        except Exception:
            continue
        if val is not None:
            arr = np.asarray(val, dtype=float).ravel()
            if arr.size >= min_size:
                return arr
    if cp is not None:
        cp._build_hcc_state_cache()
        arr = np.asarray(cp._eci0_mps_cache, dtype=float)
        if arr.size >= min_size:
            return arr
    raise RuntimeError("Could not determine state energies for NAC scaling.")


def _gradient_one_state(mc, cp: CPCASSCFResponseFCI, state: int,
                        tol: float, max_iter: int) -> np.ndarray:
    """Compute analytic SA-CASSCF nuclear gradient for `state` using our CP solver.

    Mirrors `test_step4_baseline_gradient.py::test_G3_full_gradient_matches`.
    """
    from pyscf.grad import sacasscf as sacasscf_grad

    grad_obj = sacasscf_grad.Gradients(mc)

    # Solve the CP-CASSCF response with our solver
    kappa, v_list, info = cp.solve(state=state, tol=tol, max_iter=max_iter)
    if info != 0:
        # GMRES did not declare convergence; we still proceed -- pyscf would
        # have warned similarly. The caller can decide.
        print(f"[analytic_cp_sharc] WARNING: CP-CASSCF gradient GMRES "
              f"info={info} for state {state}", flush=True)
    Lvec_mine = grad_obj.pack_uniq_var(kappa, v_list)

    # Monkey-patch solve_lagrange so pyscf's kernel uses our Lvec
    def _fake_solve(*args, **kwargs):
        bvec = grad_obj.get_wfn_response(state=state)
        Aop, Adiag = grad_obj.get_Aop_Adiag(state=state)
        return True, Lvec_mine, bvec, Aop, Adiag
    grad_obj.solve_lagrange = _fake_solve

    de = grad_obj.kernel(state=state)
    return np.asarray(de)


def _gradient_one_state_mps_krylov(
    mc,
    cp: CPDMRGCASSCFResponseMPSKrylov,
    state: int,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """Compute a gradient using MPS-Krylov response and MPS Lagrange assembly."""
    from pyscf.grad import sacasscf as sacasscf_grad
    from pyscf.grad import casscf as casscf_grad
    from pyscf import lib

    grad_obj = sacasscf_grad.Gradients(mc)
    mf_grad = mc._scf.nuc_grad_method()
    kappa, lci_mps, info, meta = cp.solve_mps(
        state=state, tol=tol, max_iter=max_iter,
    )
    if info != 0:
        print(
            "[analytic_cp_sharc] WARNING: MPS-Krylov gradient response "
            f"info={info}, residual={meta.get('residual')} for state {state}",
            flush=True,
        )
    cp._last_response_info = {
        "kind": "gradient",
        "state": int(state),
        "info": int(info),
        **dict(meta),
    }
    cache = cp._build_eris_cache()
    eris = cache["eris"]
    casdm1 = cache["casdm1_per"][state]
    casdm2 = cache["casdm2_per"][state]
    fcasscf = grad_obj.make_fcasscf(state)
    fcasscf_grad = casscf_grad.Gradients(fcasscf)
    fcasscf_grad._finalize = lambda: None

    def _make_rdm12(*args, **kwargs):
        return casdm1, casdm2

    with lib.temporary_env(
        fcasscf.fcisolver,
        make_rdm12=_make_rdm12,
        make_rdm1=lambda *args, **kwargs: casdm1,
        make_rdm2=lambda *args, **kwargs: casdm2,
    ):
        ham = fcasscf_grad.kernel(
            mo_coeff=mc.mo_coeff, ci=None, atmlst=None, verbose=0,
        )
    ldot = cp.LdotJnuc_mps(
        kappa, lci_mps, mo_coeff=mc.mo_coeff, ci=mc.ci,
        eris=eris, mf_grad=mf_grad, verbose=0,
    )
    return np.asarray(ham + ldot)


def _nac_one_pair(mc, cp: CPCASSCFResponseFCI, state_pair: tuple,
                  tol: float, max_iter: int) -> np.ndarray:
    """Compute analytic SA-CASSCF NAC for state pair (ket, bra) using our CP solver.

    Mirrors `test_step5b_nac_baseline.py::test_N3_full_nac_matches`.
    """
    from pyscf.nac import sacasscf as nac_sacasscf

    nac_obj = nac_sacasscf.NonAdiabaticCouplings(mc)
    ket, bra = int(state_pair[0]), int(state_pair[1])

    kappa, v_list, info = cp.solve_nac((ket, bra), tol=tol, max_iter=max_iter)
    if info != 0:
        print(f"[analytic_cp_sharc] WARNING: CP-CASSCF NAC GMRES info={info} "
              f"for pair ({ket},{bra})", flush=True)
    Lvec_mine = nac_obj.pack_uniq_var(kappa, v_list)

    def _fake_solve(*args, **kwargs):
        bvec = nac_obj.get_wfn_response(state=(ket, bra))
        Aop, Adiag = nac_obj.get_Aop_Adiag(state=(ket, bra))
        return True, Lvec_mine, bvec, Aop, Adiag
    nac_obj.solve_lagrange = _fake_solve

    de = nac_obj.kernel(state=(ket, bra))
    return np.asarray(de)


def _nac_one_pair_mps_krylov(
    mc,
    cp: CPDMRGCASSCFResponseMPSKrylov,
    state_pair: tuple,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """Compute a NAC using MPS-Krylov response and MPS Lagrange assembly."""
    from pyscf.nac import sacasscf as nac_sacasscf
    from pyscf.grad import casscf as casscf_grad
    from pyscf import lib

    nac_obj = nac_sacasscf.NonAdiabaticCouplings(mc)
    ket, bra = int(state_pair[0]), int(state_pair[1])
    mf_grad = mc._scf.nuc_grad_method()
    kappa, lci_mps, info, meta = cp.solve_nac_mps(
        (ket, bra), tol=tol, max_iter=max_iter,
    )
    if info != 0:
        print(
            "[analytic_cp_sharc] WARNING: MPS-Krylov NAC response "
            f"info={info}, residual={meta.get('residual')} for pair "
            f"({ket},{bra})",
            flush=True,
        )
    cp._last_response_info = {
        "kind": "nac",
        "pair": [int(ket), int(bra)],
        "info": int(info),
        **dict(meta),
    }
    cache = cp._build_eris_cache()
    eris = cache["eris"]
    tdm1, tdm2 = cp._state_transition_rdm_mps(bra, ket)
    castm1 = 0.5 * (tdm1 + tdm1.T)
    castm2 = 0.5 * (tdm2 + tdm2.transpose(1, 0, 3, 2))

    fcisolver_attr = {
        "make_rdm12": lambda *args, **kwargs: (castm1, castm2),
        "make_rdm1": lambda *args, **kwargs: castm1,
        "make_rdm2": lambda *args, **kwargs: castm2,
    }
    fcasscf = nac_obj.make_fcasscf(
        state=ket, fcisolver_attr=fcisolver_attr,
    )
    fcasscf_grad = casscf_grad.Gradients(fcasscf)
    ham = nac_sacasscf.grad_elec_active(
        fcasscf_grad, mo_coeff=mc.mo_coeff, ci=None,
        eris=eris, mf_grad=mf_grad, verbose=0,
    )
    if not getattr(nac_obj, "use_etfs", False):
        # Same CSF term as pyscf.nac.sacasscf.nac_csf, using the MPS
        # transition 1-RDM rather than direct_spin1.trans_rdm1.
        ncore, ncas = mc.ncore, mc.ncas
        mo_cas = mc.mo_coeff[:, ncore:][:, :ncas]
        tm1 = reduce(np.dot, (mo_cas, tdm1.conj().T - tdm1, mo_cas.conj().T))
        e_states = _state_energies(mc, cp, min_size=max(ket, bra) + 1)
        ham += nac_sacasscf._nac_csf(
            mc.mol, mf_grad, tm1, getattr(nac_obj, "atmlst", None),
        ) * (e_states[bra] - e_states[ket])
    ldot = cp.LdotJnuc_mps(
        kappa, lci_mps, mo_coeff=mc.mo_coeff, ci=mc.ci,
        eris=eris, mf_grad=mf_grad, verbose=0,
    )
    de = np.asarray(ham + ldot)
    if not getattr(nac_obj, "mult_ediff", False):
        e_states = _state_energies(mc, cp, min_size=max(ket, bra) + 1)
        de = de / (e_states[bra] - e_states[ket])
    return de


def compute_grad_nac_analytic_cp(mc, gradient_states=None, nac_pairs=None,
                                 backend: str = "newton_casscf",
                                 tol: float = 1e-8, max_iter: int = 500):
    """Compute SA-CASSCF nuclear gradients and NAC vectors via analytic CP-CASSCF.

    Parameters
    ----------
    mc :
        Converged state-averaged CASSCF object (`pyscf.mcscf.CASSCF`).
    gradient_states : iterable[int] | None
        0-based state indices for which to return ∇E_Θ.
    nac_pairs : iterable[(int, int)] | None
        0-based (ket, bra) pairs for which to return ⟨Λ|∂/∂R|Θ⟩.
    backend : str
        Backend for `CPCASSCFResponseFCI`. Default "newton_casscf" (FCI fallback,
        bit-equivalent to pyscf for sub-DMRG CAS sizes).
    tol : float
        GMRES residual tolerance for the response equations.
    max_iter : int
        Maximum GMRES iterations.

    Returns
    -------
    dict with keys "grad" and "nac":
        grad : {state_idx: np.ndarray(natom, 3)}
        nac  : {(I, J):     np.ndarray(natom, 3)}
    """
    gradient_states = list(gradient_states or [])
    nac_pairs = [tuple(p) for p in (nac_pairs or [])]

    response_mode = str(backend).strip().lower()
    use_mps_krylov = response_mode in {
        "mps-krylov", "mps_krylov", "mps-native", "mps_native",
    }
    if response_mode in {"projected-ci", "projected_ci", "fci-projected",
                         "fci_projected"}:
        backend = "newton_casscf"
    cp = _make_mps_krylov_response(mc) if use_mps_krylov else _make_response(
        mc, backend=backend,
    )

    out = {"grad": {}, "nac": {}, "diagnostics": {"grad": {}, "nac": {}}}

    for state in gradient_states:
        if use_mps_krylov:
            out["grad"][int(state)] = _gradient_one_state_mps_krylov(
                mc, cp, int(state), tol=tol, max_iter=max_iter,
            )
            out["diagnostics"]["grad"][int(state)] = dict(
                getattr(cp, "_last_response_info", {})
            )
        else:
            out["grad"][int(state)] = _gradient_one_state(
                mc, cp, int(state), tol=tol, max_iter=max_iter,
            )

    for pair in nac_pairs:
        if use_mps_krylov:
            out["nac"][(int(pair[0]), int(pair[1]))] = _nac_one_pair_mps_krylov(
                mc, cp, (int(pair[0]), int(pair[1])),
                tol=tol, max_iter=max_iter,
            )
            out["diagnostics"]["nac"][
                (int(pair[0]), int(pair[1]))
            ] = dict(getattr(cp, "_last_response_info", {}))
        else:
            out["nac"][(int(pair[0]), int(pair[1]))] = _nac_one_pair(
                mc, cp, (int(pair[0]), int(pair[1])),
                tol=tol, max_iter=max_iter)

    return out


# --------------------------------------------------------------------------
# SHARC integration helpers (consume qmin / fill SHARC-shaped result arrays)
# --------------------------------------------------------------------------

def _statemap_to_zero_based_state(qmin, sharc_state_index: int) -> int:
    """Map SHARC's 1-based statemap index → 0-based PySCF FCI root index."""
    mult, state, _ms = qmin["statemap"][sharc_state_index]
    # PySCF FCI root index is (state - 1) within that multiplicity
    return state - 1


def fill_grad_nac_arrays(solver, qmin, results: dict):
    """Fill in qmin-shaped grad / nacdr arrays from analytic CP results.

    Parameters
    ----------
    solver : converged SA-CASSCF solver (the `solver` returned by gen_solver)
    qmin   : the SHARC qmin dict (statemap, gradmap, nacmap, natom, nmstates)
    results : output dict of `compute_grad_nac_analytic_cp`

    Returns
    -------
    grad_list, nac_array, err
        Same shapes that `get_grad` / `get_nac` produce in SHARC_PYSCF_ext.py.
    """
    nmstates = qmin["nmstates"]
    natom = qmin["natom"]
    zerograd = np.zeros((natom, 3))

    # ---- gradient list ordered by SHARC statemap ----
    grad_list = []
    for i in sorted(qmin["statemap"]):
        mult, state, _ms = tuple(qmin["statemap"][i])
        if (mult, state) in qmin["gradmap"]:
            zb = state - 1
            de = results["grad"].get(zb)
            if de is None:
                grad_list.append(zerograd)
            else:
                grad_list.append(np.asarray(de))
        else:
            grad_list.append(zerograd)

    # ---- nac array (nmstates x nmstates x natom x 3) ----
    nac = np.zeros((nmstates, nmstates, natom, 3))
    for i in sorted(qmin["statemap"]):
        for j in sorted(qmin["statemap"]):
            m1, s1, ms1 = tuple(qmin["statemap"][i])
            m2, s2, ms2 = tuple(qmin["statemap"][j])
            if m1 != m2 or ms1 != ms2 or s1 == s2:
                continue
            if (m1, s1, m2, s2) not in qmin["nacmap"]:
                continue
            bra_zb = i - 1
            ket_zb = j - 1
            # We may have computed (bra_zb, ket_zb) or (ket_zb, bra_zb)
            if (bra_zb, ket_zb) in results["nac"]:
                de = results["nac"][(bra_zb, ket_zb)]
            elif (ket_zb, bra_zb) in results["nac"]:
                de = -results["nac"][(ket_zb, bra_zb)]
            else:
                continue
            nac[bra_zb, ket_zb] = de
            nac[ket_zb, bra_zb] = -de

    return grad_list, nac, 0
