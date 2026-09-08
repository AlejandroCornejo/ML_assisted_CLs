#!/usr/bin/env python3
"""Small acceptance test for the one-solve periodic RVE tangent used by FE2."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon_fe2_mpl")

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(ROOT / "00_rve"), str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
_kratos_candidates = (
    Path("/home/sares/Kratos_Eigen_Check/bin/Release"),
    Path("/home/kratos/Kratos_Eigen_Check/bin/Release"),
)
KRATOS_PATH = next((p for p in _kratos_candidates if p.is_dir()), _kratos_candidates[0])
if str(KRATOS_PATH) not in sys.path:
    sys.path.append(str(KRATOS_PATH))

import config as cfg  # noqa: E402
from periodic_fom import PeriodicRVE  # noqa: E402


def main():
    # The deployed Stage-03 mesh generated the labels.  Reuse it exactly;
    # this test is about the tangent, not a new meshing operation.
    base = ROOT / "03_data" / "rve_mesh"
    rve = PeriodicRVE(base, cell_area=cfg.CELL_AREA)
    # States are inside the coupon training envelope, including the coupled
    # shear that distinguishes this RVE from the old one.
    states = np.array(((0.015, -0.007, -0.006),
                       (0.100, -0.045, -0.032)))
    worst = 0.0
    for E in states:
        S, C = rve.stress_and_tangent_consistent(E)
        Sf, Cf = rve.stress_and_tangent(E, h=1.0e-6)
        err_s = np.linalg.norm(S - Sf) / max(np.linalg.norm(Sf), 1.0)
        err_c = np.linalg.norm(C - Cf) / max(np.linalg.norm(Cf), 1.0)
        worst = max(worst, err_s, err_c)
        print(f"E={E}: stress rel={err_s:.3e}, tangent rel={err_c:.3e}")
    print(f"PERIODIC_CONSISTENT_TANGENT_{'PASS' if worst < 2e-5 else 'FAIL'} worst={worst:.3e}")
    return 0 if worst < 2e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
