#!/usr/bin/env python3
"""Re-run Cook (nx=8) for the ICNN/ICKAN checkpoints actually reported in
the paper's Table 7 (w_tan0=5 small-strain-corrected variants), matching
run_cook_icnn_smallstrain_variants_claude.py's own pattern. Needed to get
e_gp/U states for a reaction-force-consistent S-err recomputation --
pann_constitutive_law_claude.py's own default "certified"/"ickan" keys
point at the UNCORRECTED (w_tan0=0) checkpoints instead, confirmed by a
direct re-run: those give tip_uy=3.4705-3.6070 (ICNN)/3.5222-3.6593
(ICKAN), matching tab:cook-fe2-corrected's own "original" column exactly,
not tab:cook-fe2's reported 3.0979-3.2330/3.1678-3.3019.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import run_cook_hprom_ann_claude as cook_driver  # noqa: E402
import pann_constitutive_law_claude as pann_law  # noqa: E402


def _make_checkpoint_material_func(checkpoint_name: str, kind: str):
    law = pann_law.PannLaw(checkpoint_name, kind)

    def _f(e_voigt, young=None, poisson=None):
        return law.pk2_and_tangent(e_voigt)
    return _f


cook_driver.MATERIAL_FUNCS["pann_certified_w5"] = _make_checkpoint_material_func(
    "PANN_anisotropic_polyconvex_smallstrain_w5p0_claude.pt", "polyconvex"
)
cook_driver.MATERIAL_FUNCS["pann_ickan_w5"] = _make_checkpoint_material_func(
    "PANN_anisotropic_polyconvex_ickan_smallstrain_w5p0_claude.pt", "ickan"
)

EXPECTED_TIP_UY = {
    "pann_certified_w5": (3.0979, 3.2330),
    "pann_ickan_w5": (3.1678, 3.3019),
}

if __name__ == "__main__":
    for which in ("pann_certified_w5", "pann_ickan_w5"):
        print(f"\n=== running Cook nx=8, 20 steps, which={which} ===")
        res = cook_driver.run_newton_fe2(which, nx=8, ny=8, verbose=False, save_npz=True)
        d = np.load(HERE / f"cook_results_{which}_claude.npz")
        tip_lo, tip_hi = float(d["tip_uy_min_per_step"][-1]), float(d["tip_uy_max_per_step"][-1])
        exp_lo, exp_hi = EXPECTED_TIP_UY[which]
        ok = abs(tip_lo - exp_lo) < 1.0e-2 and abs(tip_hi - exp_hi) < 1.0e-2
        print(f"[{which}] tip_uy=({tip_lo:.4f},{tip_hi:.4f}) expected=({exp_lo:.4f},{exp_hi:.4f}) "
              f"MATCH={ok}")
        assert ok, f"{which}: tip_uy mismatch vs paper's own reported Table-7 value -- wrong checkpoint?"
