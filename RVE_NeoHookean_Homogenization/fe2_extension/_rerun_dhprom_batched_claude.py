#!/usr/bin/env python3
"""Re-run just the D-HPROM-ANN cruciform row (n_body=6, same config as
the just-completed batch) with the newly-batched decoder, to confirm
end-to-end: (a) real wall-clock drop, (b) identical convergence
behavior, (c) final results match the pre-optimization saved npz to
floating-point precision. Does not touch the 4 PANN tiers (unaffected
by this change) or HPROM-ANN/FOM (both on hold)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
N_BODY, N_ARM_LEN = 6, 4
DELTA = 1.2
N_STEPS = 20

old = np.load(HERE / "cruciform_results_dhprom_f64_consistent_claude.npz")
old_u_nodal = old["u_nodal"].copy()
old_e_gp = old["e_gp"].copy()
old_s_gp = old["s_gp"].copy()
old_wall = None  # not stored in npz; compared via the earlier reported 639.1s only

import dhprom_ann_direct_law_float64_claude as dhprom_f64_module  # noqa: E402
from run_cruciform_fe2_claude import run_newton_fe2_cruciform  # noqa: E402

assert dhprom_f64_module._DEFAULT_LAW_F64 is None
dhprom_f64_module.get_law_float64(hprom_ann_dir=str(DHPROMANN_DIR))

print("=== D-HPROM-ANN-FE2 RERUN with batched decoder ===", flush=True)
t0 = time.time()
res = run_newton_fe2_cruciform(
    "dhprom_f64_consistent", n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
    delta_x_final=DELTA, delta_y_final=DELTA, verbose=True, use_line_search=True, save_npz=True,
)
wall = time.time() - t0
print(f"[rerun] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
      f"wall={res['wall_time']:.1f}s (elapsed {wall:.1f}s)", flush=True)

new = np.load(HERE / "cruciform_results_dhprom_f64_consistent_claude.npz")
u_err = np.max(np.abs(new["u_nodal"] - old_u_nodal))
e_err = np.max(np.abs(new["e_gp"] - old_e_gp))
s_err = np.max(np.abs(new["s_gp"] - old_s_gp))
s_rel = s_err / max(np.max(np.abs(old_s_gp)), 1e-300)
print(f"[rerun] max abs diff vs pre-optimization run: u_nodal={u_err:.3e}, e_gp={e_err:.3e}, "
      f"s_gp={s_err:.3e} (rel {s_rel:.3e})", flush=True)
print(f"[rerun] previous wall time: 639.1s -> new wall time: {res['wall_time']:.1f}s "
      f"({639.1/max(res['wall_time'],1e-9):.2f}x)", flush=True)
print("RERUN_DONE_MARKER", flush=True)
