#!/usr/bin/env python3
"""Track B, Stage 6: re-run Cook's membrane through run_cook_hprom_ann_claude
.py's own run_newton_fe2, but with the online law singletons pre-seeded to
point at THIS session's new, genuinely hyper-reduced meshes + reaction-
force stress rule (maw_dynamic_reaction_force_dhpromann/,
.../_hpromann/), instead of the default hprom/ann/maw_dynamic/ (old
Z_union mesh, naive-average sig rule).

Uses the SAME nx=8, ny=8 resolution as every already-reported Table 7 row
(confirmed this session: cook_results_dhprom_f64_consistent_claude.npz's
own coords array is (289,2), matching nx=ny=8's quadratic-mesh node count
exactly) -- so the new numbers are directly comparable, not a different
problem size. Uses the "_consistent" (analytic-tangent) variant per this
session's explicit choice, now that reaction_force_hom_tangent_claude.py's
analytic d(hom_sig)/dE has been verified against finite differences
(<=0.0003% relative error across 5 test states, both classes).

get_law_float64() is a module-level singleton in both
dhprom_ann_direct_law_float64_claude.py / hprom_ann_iterative_law_float64_
claude.py; MATERIAL_FUNCS' own wrapper functions call it with no
arguments, so it must be seeded with the correct hprom_ann_dir BEFORE the
first material call of a run -- done here explicitly, once, per class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"
NX, NY = 8, 8


def main():
    import dhprom_ann_direct_law_float64_claude as dhprom_f64_module
    import hprom_ann_iterative_law_float64_claude as hprom_iter_f64_module
    from run_cook_hprom_ann_claude import run_newton_fe2

    assert dhprom_f64_module._DEFAULT_LAW_F64 is None, "singleton already seeded -- refusing to silently ignore hprom_ann_dir"
    assert hprom_iter_f64_module._DEFAULT_LAW_F64 is None, "singleton already seeded -- refusing to silently ignore hprom_ann_dir"

    law_d = dhprom_f64_module.get_law_float64(hprom_ann_dir=str(DHPROMANN_DIR))
    print(f"[run-cook-rf] D-HPROM-ANN law seeded with hprom_ann_dir={DHPROMANN_DIR}")
    law_h = hprom_iter_f64_module.get_law_float64(hprom_ann_dir=str(HPROMANN_DIR))
    print(f"[run-cook-rf] HPROM-ANN law seeded with hprom_ann_dir={HPROMANN_DIR}")

    print(f"\n=== D-HPROM-ANN-FE2 (dhprom_f64_consistent), nx={NX}, ny={NY} ===")
    res_d = run_newton_fe2("dhprom_f64_consistent", nx=NX, ny=NY, verbose=True, use_line_search=True, save_npz=True)
    print(f"[run-cook-rf] D-HPROM-ANN-FE2 done: fully_converged={res_d['fully_converged']}, "
          f"ever_diverged={res_d['ever_diverged']}, wall_time={res_d['wall_time']:.1f}s")

    print(f"\n=== HPROM-ANN-FE2 (hprom_iterative_f64_consistent), nx={NX}, ny={NY} ===")
    res_h = run_newton_fe2("hprom_iterative_f64_consistent", nx=NX, ny=NY, verbose=True, use_line_search=True, save_npz=True)
    print(f"[run-cook-rf] HPROM-ANN-FE2 done: fully_converged={res_h['fully_converged']}, "
          f"ever_diverged={res_h['ever_diverged']}, wall_time={res_h['wall_time']:.1f}s")

    print("\n=== SUMMARY ===")
    for label, res in (("D-HPROM-ANN-FE2", res_d), ("HPROM-ANN-FE2", res_h)):
        print(f"  {label}: fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
              f"wall_time={res['wall_time']:.1f}s, n_material_calls={res['n_material_calls']}, "
              f"tip_uy={res['tip_uy_range']}")


if __name__ == "__main__":
    main()
