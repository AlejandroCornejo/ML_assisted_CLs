#!/usr/bin/env python3
"""Runs all 4 already-trained PANN tiers (certified/ICNN, free, ICKAN,
regression) as Cook's-membrane's material law at nx=8, ny=8 -- the same
mesh used for the fe2_extension D-HPROM-ANN/HPROM-ANN speedup comparison
-- so the two families can be compared on equal footing at one shared
mesh resolution.

Imports run_newton read-only from Cook.gid/run_cook_pann_claude.py (the
paper's own, already-validated driver, which only ever runs nx=16 in its
own __main__ block); nothing there is modified. Saves under
fe2_extension/ with an explicit "_nx8" suffix so nothing here can ever
collide with or overwrite the paper-cited nx=16 result files.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import os

os.chdir(str(HERE))  # so run_newton's relative np.savez(...) lands under fe2_extension/, not Cook.gid/

from run_cook_pann_claude import run_newton  # noqa: E402  (read-only reuse)

if __name__ == "__main__":
    results = {}
    for which in ("certified", "free", "ickan", "regression"):
        print(f"=== {which} (nx=8) ===")
        results[which] = run_newton(
            which, nx=8, ny=8, diagnose=True, use_line_search=True, save_npz=True, name_suffix="_nx8",
        )
        print()

    print("=== summary (nx=8) ===")
    for which, res in results.items():
        print(f"{which:10s}: fully_converged={res['fully_converged']}, "
              f"E11={res['e11_range']}, gamma12={res['gamma12_range']}, "
              f"tip_uy={res['tip_uy_range']}")
