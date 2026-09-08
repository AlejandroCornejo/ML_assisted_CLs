import sys
sys.path.insert(0, ".")
from pathlib import Path

import dhprom_ann_direct_law_float64_claude as dhprom_f64_module
from run_cook_hprom_ann_claude import run_newton_fe2

DHPROMANN_DIR = Path("maw_dynamic_reaction_force_classic82_dhpromann").resolve()

dhprom_f64_module.get_law_float64(hprom_ann_dir=str(DHPROMANN_DIR))
print(f"[run-cook-82] D-HPROM-ANN law seeded with hprom_ann_dir={DHPROMANN_DIR}", flush=True)

res = run_newton_fe2("dhprom_f64_consistent", nx=8, ny=8, verbose=True, use_line_search=True, save_npz=True)
print(f"[run-cook-82] D-HPROM-ANN-FE2 (82pt) done: fully_converged={res['fully_converged']}, "
      f"ever_diverged={res['ever_diverged']}, wall_time={res['wall_time']:.1f}s, tip_uy={res['tip_uy_range']}", flush=True)
print("COOK_82PT_DONE_MARKER", flush=True)
