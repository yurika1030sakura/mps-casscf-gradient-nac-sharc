"""
Anthracene CAS(14,14)/STO-3G fast-path DMRG-CASCI M-scan vs cached FCI.

Fixed-orbital validation of the seven opt-in fast-path flags listed in
the top-level README. Loads the manuscript's reference FCI energies
(state-averaged singlet pair on AVAS pi14 orbitals) and runs the v11
MPSAsFCISolver fast path at multiple bond dimensions M, reporting per-M
energy error and wall time.

Expected behaviour: monotonic decrease of max|dE| vs M, sub-mHa accuracy
by M ~ 1024 on 4 CPU cores. The numbers in the table below were produced
by this script and are quoted in the README Validation section.

Reproduce:
    PYTHONPATH=../../src python run_anthracene_pi14_fastpath_mscan.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas

# Allow `python run_anthracene_pi14_fastpath_mscan.py` from this directory.
HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src" / "dmrg_analytic_dev"
sys.path.insert(0, str(SRC))
from dmrg_fcisolver import MPSAsFCISolver  # noqa: E402

# Cached FCI reference for planar anthracene CAS(14,14)/STO-3G SA(2 singlets).
# These are the manuscript's published reference energies.
E_FCI = np.array([-529.7030437226188, -529.5556316123168])


def build_anthracene():
    """Idealized planar anthracene geometry (D2h, three fused hexagons)."""
    r_cc, r_ch = 1.397, 1.09
    centers = [(0.0, 0.0), (np.sqrt(3.0) * r_cc, 0.0),
               (2.0 * np.sqrt(3.0) * r_cc, 0.0)]
    carbons = []
    for cx, cy in centers:
        for k in range(6):
            theta = np.deg2rad(30.0 + 60.0 * k)
            pos = np.array([cx + r_cc * np.cos(theta),
                            cy + r_cc * np.sin(theta), 0.0])
            if not any(np.linalg.norm(pos - old) < 1e-5 for old in carbons):
                carbons.append(pos)
    carbons = np.asarray(carbons)
    carbons[:, 0] -= carbons[:, 0].mean()
    carbons[:, 1] -= carbons[:, 1].mean()
    neighbors = {i: [] for i in range(len(carbons))}
    for i, pi in enumerate(carbons):
        for j, pj in enumerate(carbons[:i]):
            if abs(np.linalg.norm(pi - pj) - r_cc) < 1e-3:
                neighbors[i].append(j); neighbors[j].append(i)
    atoms = []
    for i, p in enumerate(carbons):
        atoms.append(("C", tuple(p)))
        if len(neighbors[i]) == 2:
            v_in = np.zeros(3)
            for j in neighbors[i]:
                v_in += carbons[j] - p
            v_out = -v_in / np.linalg.norm(v_in)
            atoms.append(("H", tuple(p + v_out * r_ch)))
    return gto.M(atom=atoms, basis="sto-3g", charge=0, spin=0, unit="Angstrom",
                 symmetry=False, verbose=0)


def time_section(fn, *args, **kwargs):
    t0 = time.time(); out = fn(*args, **kwargs); return out, time.time() - t0


def run_fastpath_casci(mol, mf, mo_avas, ncas, nelecas, M):
    """Fixed-orbital DMRG-CASCI on AVAS pi orbitals with every fast-path
    flag enabled. Single kernel call, no orbital optimization."""
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.mo_coeff = mo_avas
    mc.fcisolver = MPSAsFCISolver(
        mol,
        bond_dim=M, n_sweeps=30, n_threads=4,
        sweep_tol=1e-10,
        force_dmrg=True, max_fci_dets=20_000_000,
        mps_native_rdms=True,
        skip_kernel_fci_conversion=True,
        first_iter_warmup=True,
        warm_start=True,
        stack_mem_mb=8000,
        root_buffer=4,
        dmrg_symm_su2=True,
    )
    mc.fcisolver.nroots = 2
    mc.kernel()
    return list(map(float, mc.e_tot))


def main():
    print("=" * 78)
    print("Anthracene CAS(14,14)/STO-3G fast-path DMRG-CASCI M-scan vs cached FCI")
    print("=" * 78)
    mol = build_anthracene()
    print(f"  norb = {mol.nao}, nelec = {mol.nelectron}")
    mf, t_rhf = time_section(lambda: scf.RHF(mol).run(conv_tol=1e-10))
    print(f"  RHF E = {mf.e_tot:.10f}  ({t_rhf:.1f}s)")

    print("\nAVAS pi14 active orbital selection...")
    ncas, nelecas, mo_avas = avas.avas(
        mf, ["C 2pz"], canonicalize=False, with_iao=False)
    print(f"  AVAS picked CAS({nelecas},{ncas})")
    assert (ncas, nelecas) == (14, 14), "expected pi14 active space"

    print(f"\nCached FCI reference: E0={E_FCI[0]:.10f}  E1={E_FCI[1]:.10f}")
    print(f"                       gap = {(E_FCI[1]-E_FCI[0])*27.2114:.3f} eV")

    print("\nFast-path DMRG-CASCI M-scan:")
    print(f"  {'M':>5}  {'E0 (Ha)':>16}  {'E1 (Ha)':>16}  "
          f"{'|dE0| mHa':>10}  {'|dE1| mHa':>10}  {'wall (s)':>9}")
    print("  " + "-" * 76)
    for M in [64, 128, 256, 512, 1024]:
        try:
            e, t = time_section(run_fastpath_casci, mol, mf, mo_avas, 14, 14, M)
            dE0 = 1000.0 * (e[0] - E_FCI[0])
            dE1 = 1000.0 * (e[1] - E_FCI[1])
            print(f"  {M:>5}  {e[0]:>16.10f}  {e[1]:>16.10f}  "
                  f"{abs(dE0):>10.3e}  {abs(dE1):>10.3e}  {t:>9.1f}")
        except Exception as ex:
            print(f"  {M:>5}  ERROR: {type(ex).__name__}: {ex}")
    print()
    print("Expected: |dE| -> 0 monotonically as M -> FCI saturation.")


if __name__ == "__main__":
    main()
