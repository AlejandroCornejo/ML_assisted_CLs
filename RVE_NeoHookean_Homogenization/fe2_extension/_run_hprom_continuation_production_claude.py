import sys
sys.path.insert(0, ".")
from pathlib import Path
import time

import numpy as np

HERE = Path(".").resolve()
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

import run_cruciform_fe2_claude as rc

wrapper = rc.make_hprom_continuation_material_func(HPROMANN_DIR)
rc.MATERIAL_FUNCS["hprom_iterative_f64_continuation"] = wrapper

print(f"CHECKPOINT: law.qp_init_mode = {wrapper.law.qp_init_mode!r}", flush=True)

t0 = time.time()
res = rc.run_newton_fe2_cruciform(
    "hprom_iterative_f64_continuation", n_body=12, n_arm_len=8, n_steps=20,
    delta_x_final=1.2, delta_y_final=1.2, verbose=True, use_line_search=True, save_npz=True,
)
d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
g12 = np.abs(d["e_gp"][:, 2])
print(f"[hprom_iterative_f64_continuation] PRODUCTION fully_converged={res['fully_converged']}, "
      f"ever_diverged={res['ever_diverged']}, E11={res['e11_range']}, E22={res['e22_range']}, "
      f"g12={res['g12_range']}, median|g12|={np.median(g12):.4f}, frac>0.1={np.mean(g12 > 0.1):.3f}, "
      f"wall={res['wall_time']:.1f}s (elapsed {time.time()-t0:.1f}s)", flush=True)
print("per-step status:", [s["status"] for s in res["step_log"]], flush=True)
print("PRODUCTION_DONE_MARKER", flush=True)
