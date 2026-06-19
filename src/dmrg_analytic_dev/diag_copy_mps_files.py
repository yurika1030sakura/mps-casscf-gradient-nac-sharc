"""Decisive diagnostic for the MPS copy/transport: what files exist, and does
copy_mps + load_mps round-trip within a single driver?
"""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "sharc_interface"))

import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response

ANG = 1.8897261246257702


def main():
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=100, n_sweeps=24, sweep_tol=1.0e-10, n_threads=1)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.90]]) * ANG
    _mol, _mf, mc, _solver = fdv.build_sa_dmrg_casscf(
        ["He", "H"], coords, basis="3-21G", charge=1, spin=0,
        ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5], solver_cfg=cfg)
    obj = _make_mps_krylov_response(mc)
    drv = obj._driver_su2
    st0 = obj._state_mps[0]
    st1 = obj._state_mps[1]
    print("state0 tag:", st0.info.tag, " state1 tag:", st1.info.tag)
    print("scratch:", drv.frame.mps_dir)

    import block2
    block2.Global.frame = obj._su2_frame
    cp = drv.copy_mps(st0, "XFERTEST")
    cp.save_data()

    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(drv.frame.mps_dir, "*")))
    print("--- files matching XFERTEST* ---")
    for f in files:
        if f.startswith("XFERTEST"):
            print("  ", f)
    print("--- files matching original state0 tag", repr(st0.info.tag), "---")
    for f in files:
        if f.startswith(str(st0.info.tag)):
            print("  ", f)

    print("--- files matching *XFERTEST* (tag anywhere) ---")
    for f in files:
        if "XFERTEST" in f:
            print("  ", f)
    print("--- files matching *KRY-STATE-0* (tag anywhere) ---")
    for f in files:
        if "KRY-STATE-0" in f:
            print("  ", f)

    # within-driver round trip
    loaded = drv.load_mps("XFERTEST")
    ov_self = float(obj._mps_overlap(loaded, loaded))
    ov_st0 = float(obj._mps_overlap(st0, loaded))
    ov_st1 = float(obj._mps_overlap(st1, loaded))
    print(f"within-driver: <loaded|loaded>={ov_self:.6f}  "
          f"<st0|loaded>={ov_st0:.6f}  <st1|loaded>={ov_st1:.6f}")
    print("(expect <st0|loaded>=1, <st1|loaded>=0 if copy/load preserves state0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
