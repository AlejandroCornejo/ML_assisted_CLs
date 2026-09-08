#!/usr/bin/env python3
"""First-ever run of Linear-HPROM (pure POD, no ANN) through the cruciform
FE2 driver -- this combination has never been run before (Linear-HPROM was
only previously exercised on Cook's membrane). Short (5-step) run at
n_body=6 purely to confirm it converges and gives sane physics before
trusting it for profiling or a full 20-step run."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cruciform_fe2_claude as rc  # noqa: E402

wrapper = rc.make_linear_hprom_continuation_material_func()
rc.MATERIAL_FUNCS["linear_hprom_continuation"] = wrapper

print("=== Linear-HPROM-FE2 (cruciform) FIRST TEST: n_body=6, 5 steps ===", flush=True)
t0 = time.time()
res = rc.run_newton_fe2_cruciform(
    "linear_hprom_continuation", n_body=6, n_arm_len=4, n_steps=5,
    delta_x_final=1.2, delta_y_final=1.2, verbose=True, use_line_search=True, save_npz=False,
)
wall = time.time() - t0
print(f"[first_test] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
      f"wall={res['wall_time']:.1f}s (elapsed {wall:.1f}s), n_material_calls={res['n_material_calls']}",
      flush=True)
print(f"[first_test] e11_range={res['e11_range']}, e22_range={res['e22_range']}, "
      f"g12_range={res['g12_range']}", flush=True)
for step in res["step_log"]:
    print(f"[first_test] step_log: {step}", flush=True)
print("FIRST_TEST_DONE_MARKER", flush=True)
