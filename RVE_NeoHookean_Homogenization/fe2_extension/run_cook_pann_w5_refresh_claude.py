#!/usr/bin/env python3
"""Redo the ICNN/ICKAN (tier 3a/3b) rows of Cook's nx=8 suite using the
CORRECT checkpoint Table 7 actually reports (the w_tan0=5 small-strain-
corrected variant) -- run_cook_nx8_full_suite_claude.py's own
"pann_certified"/"pann_ickan" keys point at the UNCORRECTED checkpoint
instead (confirmed: gave tip_uy=3.4705-3.6070/3.5222-3.6593, matching
the known "original" column, not Table 7's own 3.0979-3.2330/3.1678-
3.3019), exactly the mismatch run_cook_pann_smallstrain_w5_claude.py
already documented and fixed once before. Mirrors that script's own
checkpoint registration verbatim, just with this session's own naming
(pann_certified_w5/pann_ickan_w5) and full verbose/save settings."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)
import os
os.chdir(str(HERE))

import run_cook_hprom_ann_claude as cook_module  # noqa: E402
import pann_constitutive_law_claude as pann_law  # noqa: E402


def _make_checkpoint_material_func(checkpoint_name, kind):
    law = pann_law.PannLaw(checkpoint_name, kind)

    def _f(e_voigt, young=None, poisson=None):
        return law.pk2_and_tangent(e_voigt)
    return _f


cook_module.MATERIAL_FUNCS["pann_certified_w5"] = _make_checkpoint_material_func(
    "PANN_anisotropic_polyconvex_smallstrain_w5p0_claude.pt", "polyconvex"
)
cook_module.MATERIAL_FUNCS["pann_ickan_w5"] = _make_checkpoint_material_func(
    "PANN_anisotropic_polyconvex_ickan_smallstrain_w5p0_claude.pt", "ickan"
)

EXPECTED_TIP_UY = {
    "pann_certified_w5": (3.0979, 3.2330),
    "pann_ickan_w5": (3.1678, 3.3019),
}

if __name__ == "__main__":
    t0_all = time.time()
    for which in ("pann_certified_w5", "pann_ickan_w5"):
        t0 = time.time()
        res = cook_module.run_newton_fe2(which, nx=8, ny=8, n_steps=20, verbose=False, save_npz=True)
        d = np.load(HERE / f"cook_results_{which}_claude.npz")
        tip_lo, tip_hi = float(d["tip_uy_min_per_step"][-1]), float(d["tip_uy_max_per_step"][-1])
        exp_lo, exp_hi = EXPECTED_TIP_UY[which]
        ok = abs(tip_lo - exp_lo) < 1.0e-2 and abs(tip_hi - exp_hi) < 1.0e-2
        print(f"[{which}] fully_converged={res['fully_converged']}, tip_uy=({tip_lo:.4f},{tip_hi:.4f}) "
              f"expected=({exp_lo:.4f},{exp_hi:.4f}) MATCH={ok} wall={res['wall_time']:.1f}s "
              f"(elapsed {time.time() - t0:.1f}s)", flush=True)
        assert ok, f"{which}: tip_uy mismatch vs paper's own reported Table-7 value -- wrong checkpoint?"
    print(f"TOTAL elapsed: {time.time() - t0_all:.1f}s", flush=True)
    print("PANN_W5_REFRESH_DONE_MARKER", flush=True)
