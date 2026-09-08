#!/usr/bin/env python3
"""Same as run_cook_reaction_force_claude.py, but pointing at the
37-point, FIXED (classic, non-adaptive) ECM rule's meshes
(maw_dynamic_reaction_force_classic37_{dhpromann,hpromann}/) instead of
the 10-point ANN-fitted one -- mitigation attempt for the tip_uy
degradation diagnosed with the 10-point rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_classic37_dhpromann"
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_classic37_hpromann"
NX, NY = 8, 8


def main():
    import dhprom_ann_direct_law_float64_claude as dhprom_f64_module
    import hprom_ann_iterative_law_float64_claude as hprom_iter_f64_module
    from run_cook_hprom_ann_claude import run_newton_fe2

    assert dhprom_f64_module._DEFAULT_LAW_F64 is None, "singleton already seeded"
    assert hprom_iter_f64_module._DEFAULT_LAW_F64 is None, "singleton already seeded"

    dhprom_f64_module.get_law_float64(hprom_ann_dir=str(DHPROMANN_DIR))
    print(f"[run-cook-classic37] D-HPROM-ANN law seeded with hprom_ann_dir={DHPROMANN_DIR}")
    hprom_iter_f64_module.get_law_float64(hprom_ann_dir=str(HPROMANN_DIR))
    print(f"[run-cook-classic37] HPROM-ANN law seeded with hprom_ann_dir={HPROMANN_DIR}")

    print(f"\n=== D-HPROM-ANN-FE2 (dhprom_f64_consistent), nx={NX}, ny={NY} ===")
    res_d = run_newton_fe2("dhprom_f64_consistent", nx=NX, ny=NY, verbose=True, use_line_search=True, save_npz=True)
    print(f"[run-cook-classic37] D-HPROM-ANN-FE2 done: fully_converged={res_d['fully_converged']}, "
          f"ever_diverged={res_d['ever_diverged']}, wall_time={res_d['wall_time']:.1f}s")

    print(f"\n=== HPROM-ANN-FE2 (hprom_iterative_f64_consistent), nx={NX}, ny={NY} ===")
    res_h = run_newton_fe2("hprom_iterative_f64_consistent", nx=NX, ny=NY, verbose=True, use_line_search=True, save_npz=True)
    print(f"[run-cook-classic37] HPROM-ANN-FE2 done: fully_converged={res_h['fully_converged']}, "
          f"ever_diverged={res_h['ever_diverged']}, wall_time={res_h['wall_time']:.1f}s")

    print("\n=== SUMMARY ===")
    for label, res in (("D-HPROM-ANN-FE2", res_d), ("HPROM-ANN-FE2", res_h)):
        print(f"  {label}: fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
              f"wall_time={res['wall_time']:.1f}s, n_material_calls={res['n_material_calls']}, "
              f"tip_uy={res['tip_uy_range']}")


if __name__ == "__main__":
    main()
