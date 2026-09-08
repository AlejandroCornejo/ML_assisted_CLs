#!/usr/bin/env python3
"""First-ever full cruciform FE2 run of Linear-HPROM (pure POD, no ANN),
via the parallel wrapper (LinearHpromParallelContinuationWrapper),
n_body=6/n_arm_len=4/20 steps -- same mesh/loading as every other model's
own cruciform row this session.

Ordering follows the same fork-after-Kratos-threading discipline already
established and verified this session for the FOM-nested parallel law:
set thread-limiting env vars, create the persistent process pool, THEN
import run_cruciform_fe2_claude (which imports fom_solver_rve/
KratosMultiphysics and touches Kratos as soon as run_newton_fe2_cruciform
actually runs) -- never the other way around.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import linear_hprom_law_parallel_claude as par_module  # noqa: E402
par_module.ensure_persistent_executor(n_workers=16)

import run_cruciform_fe2_claude as rc  # noqa: E402

wrapper = rc.make_linear_hprom_parallel_continuation_material_func(n_workers=16)
rc.MATERIAL_FUNCS["linear_hprom_parallel_continuation"] = wrapper

N_BODY, N_ARM_LEN = 6, 4
DELTA = 1.2
N_STEPS = 20

print(f"=== Linear-HPROM-FE2 (cruciform, PARALLEL) n_body={N_BODY}, {N_STEPS} steps ===", flush=True)
t0 = time.time()
res = rc.run_newton_fe2_cruciform(
    "linear_hprom_parallel_continuation", n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
    delta_x_final=DELTA, delta_y_final=DELTA, verbose=True, use_line_search=True, save_npz=True,
)
wall = time.time() - t0
print(f"[run] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
      f"wall={res['wall_time']:.1f}s (elapsed {wall:.1f}s), n_material_calls={res['n_material_calls']}",
      flush=True)
print(f"[run] e11_range={res['e11_range']}, e22_range={res['e22_range']}, "
      f"g12_range={res['g12_range']}", flush=True)
print("RUN_DONE_MARKER", flush=True)
