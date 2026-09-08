import sys
sys.path.insert(0, ".")
from pathlib import Path
import time

import numpy as np

HERE = Path(".").resolve()
DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"
N_BODY, N_ARM_LEN = 9, 6
DELTA = 1.2
N_STEPS = 20

import dhprom_ann_direct_law_float64_claude as dhprom_f64_module
import run_cruciform_fe2_claude as rc

results = {}

print(f"=== Remaining PANN tiers (n_body={N_BODY}) === (ickan already done via convergence study)", flush=True)
for which in ("pann_regression", "pann_free", "pann_certified"):
    t0 = time.time()
    res = rc.run_newton_fe2_cruciform(
        which, n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
        delta_x_final=DELTA, delta_y_final=DELTA, verbose=False, use_line_search=True, save_npz=True,
    )
    results[which] = res
    d = np.load(HERE / f"cruciform_results_{which}_claude.npz")
    g12 = np.abs(d["e_gp"][:, 2])
    print(f"[{which}] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"wall={res['wall_time']:.1f}s (elapsed {time.time()-t0:.1f}s)", flush=True)

print(f"\n=== D-HPROM-ANN-FE2 (n_body={N_BODY}) ===", flush=True)
dhprom_f64_module.get_law_float64(hprom_ann_dir=str(DHPROMANN_DIR))
t0 = time.time()
res = rc.run_newton_fe2_cruciform(
    "dhprom_f64_consistent", n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
    delta_x_final=DELTA, delta_y_final=DELTA, verbose=True, use_line_search=True, save_npz=True,
)
results["dhprom_f64_consistent"] = res
print(f"[dhprom_f64_consistent] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
      f"wall={res['wall_time']:.1f}s (elapsed {time.time()-t0:.1f}s)", flush=True)

print(f"\n=== HPROM-ANN-FE2 with continuation fix (n_body={N_BODY}) ===", flush=True)
wrapper = rc.make_hprom_continuation_material_func(HPROMANN_DIR)
rc.MATERIAL_FUNCS["hprom_iterative_f64_continuation"] = wrapper
t0 = time.time()
res = rc.run_newton_fe2_cruciform(
    "hprom_iterative_f64_continuation", n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
    delta_x_final=DELTA, delta_y_final=DELTA, verbose=True, use_line_search=True, save_npz=True,
)
results["hprom_iterative_f64_continuation"] = res
print(f"[hprom_iterative_f64_continuation] fully_converged={res['fully_converged']}, "
      f"ever_diverged={res['ever_diverged']}, wall={res['wall_time']:.1f}s (elapsed {time.time()-t0:.1f}s)", flush=True)
print("per-step status:", [s["status"] for s in res["step_log"]], flush=True)

print("\n=== PRODUCTION n_body=9 SUMMARY ===", flush=True)
for which, res in results.items():
    print(f"  {which:32s}: fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"wall={res['wall_time']:.1f}s", flush=True)
print("PRODUCTION_NBODY9_DONE_MARKER", flush=True)
