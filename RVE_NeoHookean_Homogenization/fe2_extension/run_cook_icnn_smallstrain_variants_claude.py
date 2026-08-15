#!/usr/bin/env python3
"""Exploratory, non-paper run: full 20-step Cook (nx=8) with the two new
zero-strain-tangent-loss ICNN checkpoints (tangent0_weight=1.0 and 0.1) as
the macroscopic material law, exactly like the original certified PANN is
already used in run_cook_hprom_ann_claude.py's MATERIAL_FUNCS -- just
pointed at the new checkpoints instead. Reuses run_newton_fe2 unchanged
(same mesh, load ramp, tolerance, line search, stall handling as every
other Cook comparison this session).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
PANN_DIR = ROOT / "pann" / "anisotropic"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import run_cook_hprom_ann_claude as cook_driver  # noqa: E402
import pann_constitutive_law_claude as pann_law  # noqa: E402


def _make_checkpoint_material_func(checkpoint_name: str):
    law = pann_law.PannLaw(checkpoint_name, "polyconvex")

    def _f(e_voigt, young=None, poisson=None):
        return law.pk2_and_tangent(e_voigt)
    return _f


cook_driver.MATERIAL_FUNCS["certified_smallstrain_w1"] = _make_checkpoint_material_func(
    "PANN_anisotropic_polyconvex_smallstrain_claude.pt"
)
cook_driver.MATERIAL_FUNCS["certified_smallstrain_w0p1"] = _make_checkpoint_material_func(
    "PANN_anisotropic_polyconvex_smallstrain_w0p1_claude.pt"
)

if __name__ == "__main__":
    results = {}
    for which in ("certified_smallstrain_w1", "certified_smallstrain_w0p1"):
        print(f"\n=== running Cook nx=8, 20 steps, which={which} ===")
        res = cook_driver.run_newton_fe2(which, nx=8, ny=8, verbose=False, save_npz=True)
        results[which] = res
        print(f"[{which}] fully_converged={res['fully_converged']}  ever_diverged={res['ever_diverged']}")
        for s in res["step_log"]:
            print(f"  step {s['step']:2d} iters={s['iters']:2d} status={s['status']:10s} "
                  f"tip_uy=({s['tip_uy_min']:.4f},{s['tip_uy_max']:.4f})")

    print("\n\n=== SUMMARY ===")
    for which, res in results.items():
        print(f"{which:28s} fully_converged={res['fully_converged']}  "
              f"final tip_uy={res['tip_uy_range']}  wall={res['wall_time']:.1f}s")
