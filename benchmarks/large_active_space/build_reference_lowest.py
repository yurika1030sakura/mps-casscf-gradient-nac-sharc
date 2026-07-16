"""Canonical lowest-basin SA-CASSCF reference for the beyond-FCI validation.

The polyene SA-CASSCF(2N,2N) orbital landscape is multi-valued (C20: >=3 stationary
points within ~3 mEh).  Both build_robust and build_progressive start from a RANDOM
initial MPS (solver.dmrg_seed=None), so each run lands in a random basin -- which is
why the analytic-g.q and FD drivers, run independently, ended up at DIFFERENT basins
and the soft single_cc directional gradient mismatched (a basin/branch artifact, NOT
a response error: the +h continuation lands exactly on the analytic-predicted branch).

This module builds ONE canonical reference: multi-seed (block2.Random.rand_seed) cold
builds at a TIGHT schedule, take the LOWEST converged e0, and PERSIST mo_coeff to a
.npy keyed on (n, basis).  BOTH the analytic g.q driver and the FD driver then LOAD
this exact mo_coeff as their initial guess, so they are evaluated at the IDENTICAL
SA-CASSCF stationary point and the comparison is airtight.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DEV = _HERE.parents[1] / "src" / "dmrg_analytic_dev"
SH = _HERE.parents[1] / "sharc_interface"
for _p in (str(_HERE), str(DEV), str(SH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ANG = 1.8897261246257702

# Tight schedule: many macro-iterations + tight conv so a given basin is reproducible
# to ~1e-9 Eh (kills the run-to-run convergence noise that the FD amplifies by 1/2h).
TIGHT_SCHEDULE = [(256, 1e-8, 1e-4, 60), (512, 1e-10, 1e-5, 80), (800, 1e-12, 1e-6, 200)]


def _log(m):
    print(f"[refbuild] {m}", flush=True)


def reference_paths(n, basis="sto-3g"):
    base = _HERE / "data" / f"reference_mo_c{n}_{basis}"
    return base.with_suffix(".npy"), base.with_suffix(".meta.json")


def build_or_load_reference(n, *, basis="sto-3g", seeds=(1, 2, 3, 4, 5),
                            m_schedule=None, threads=16, stack_mem_mb=24000,
                            force=False):
    """Return dict(symbols, coords_bohr, ncas, nelecas, mo_coeff, e0, seed).

    Persists the lowest-basin mo_coeff so repeated calls (and the other driver)
    reuse the identical reference.
    """
    from run_cas_directional_fd import build_progressive
    from run_polyene_beyond_fci import polyene_geometry
    import block2

    m_schedule = m_schedule or TIGHT_SCHEDULE
    symbols = [a[0] for a in polyene_geometry(n)]
    coords = np.array([a[1] for a in polyene_geometry(n)]) * ANG
    ncas = nelecas = n
    mo_path, meta_path = reference_paths(n, basis)

    if mo_path.exists() and not force:
        mo = np.load(mo_path)
        meta = json.load(open(meta_path)) if meta_path.exists() else {}
        _log(f"loaded persisted reference {mo_path.name} e0={meta.get('e0')}")
        return {"symbols": symbols, "coords_bohr": coords, "ncas": ncas,
                "nelecas": nelecas, "mo_coeff": mo, "e0": meta.get("e0"),
                "seed": meta.get("seed"), "from_cache": True, "meta": meta}

    best = None  # (e0, mo, seed)
    per_seed = []
    for s in seeds:
        block2.Random.rand_seed(int(s))
        t = time.perf_counter()
        _mol, mc, _sol, _blog = build_progressive(
            symbols, coords, basis, ncas, nelecas, m_schedule=m_schedule,
            threads=threads, stack_mem_mb=stack_mem_mb)
        e0 = float(mc.e_states[0])
        conv = bool(mc.converged)
        per_seed.append({"seed": s, "e0": e0, "converged": conv,
                         "wall_s": time.perf_counter() - t})
        _log(f"seed={s}: e0={e0:.9f} conv={conv} ({per_seed[-1]['wall_s']:.0f}s)")
        if conv and (best is None or e0 < best[0]):
            best = (e0, np.array(mc.mo_coeff), s)
    if best is None:
        raise RuntimeError(f"no converged seed for C{n} reference")

    e0, mo, seed = best
    np.save(mo_path, mo)
    meta = {"n": n, "basis": basis, "e0": e0, "seed": seed,
            "seeds_tried": list(seeds), "per_seed": per_seed,
            "m_schedule": m_schedule, "e0_spread": (max(p["e0"] for p in per_seed)
                                                    - min(p["e0"] for p in per_seed))}
    json.dump(meta, open(meta_path, "w"), indent=2, default=str)
    _log(f"PERSISTED reference {mo_path.name}: e0={e0:.9f} seed={seed} "
         f"spread={meta['e0_spread']:.2e}")
    return {"symbols": symbols, "coords_bohr": coords, "ncas": ncas,
            "nelecas": nelecas, "mo_coeff": mo, "e0": e0, "seed": seed,
            "from_cache": False, "meta": meta}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncarbon", type=int, required=True)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--stack-mem-mb", type=int, default=24000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    r = build_or_load_reference(a.ncarbon, basis=a.basis, seeds=a.seeds,
                                threads=a.threads, stack_mem_mb=a.stack_mem_mb,
                                force=a.force)
    _log(f"C{a.ncarbon} reference e0={r['e0']} seed={r['seed']} "
         f"cached={r['from_cache']}")
