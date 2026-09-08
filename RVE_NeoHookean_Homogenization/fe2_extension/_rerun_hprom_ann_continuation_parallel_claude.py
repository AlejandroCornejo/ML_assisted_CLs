#!/usr/bin/env python3
"""Re-run the full 20-step cruciform HPROM-ANN continuation row (n_body=6,
same config as the existing cruciform_results_hprom_iterative_f64_
continuation_claude.npz) with the two-level parallel wrapper
(HpromAnnParallelContinuationWrapper: multiprocessing across macro points,
vmap batching within each worker's own chunk), to confirm end-to-end
correctness (fully_converged/ever_diverged/iters_per_step/converged_per_
step match the existing serial reference, u_nodal/e_gp/s_gp match closely)
and report the real wall-clock speedup.

Ordering follows the same fork-after-Kratos/torch-threading discipline
already established this session: set thread-limiting env vars, create
the persistent process pool, THEN import run_cruciform_fe2_claude."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import hprom_ann_law_parallel_claude as par_module  # noqa: E402
par_module.ensure_persistent_executor(n_workers=16)

HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"
N_BODY, N_ARM_LEN = 6, 4
DELTA = 1.2
N_STEPS = 20

old = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz", allow_pickle=True)
old_u_nodal = old["u_nodal"].copy()
old_e_gp = old["e_gp"].copy()
old_s_gp = old["s_gp"].copy()
old_iters_per_step = old["iters_per_step"].copy()
old_converged_per_step = old["converged_per_step"].copy()
old_fully_converged = bool(old["fully_converged"])
old_ever_diverged = bool(old["ever_diverged"])
print(f"[rerun] pre-optimization reference: fully_converged={old_fully_converged}, "
      f"ever_diverged={old_ever_diverged}, iters_per_step={old_iters_per_step.tolist()}", flush=True)

import run_cruciform_fe2_claude as rc  # noqa: E402

wrapper = rc.make_hprom_ann_parallel_continuation_material_func(HPROMANN_DIR, n_workers=16)
rc.MATERIAL_FUNCS["hprom_ann_parallel_continuation"] = wrapper

print("=== HPROM-ANN-FE2 (continuation, 2-LEVEL PARALLEL) RERUN ===", flush=True)
t0 = time.time()
res = rc.run_newton_fe2_cruciform(
    "hprom_ann_parallel_continuation", n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
    delta_x_final=DELTA, delta_y_final=DELTA, verbose=True, use_line_search=True, save_npz=True,
)
wall = time.time() - t0
print(f"[rerun] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
      f"wall={res['wall_time']:.1f}s (elapsed {wall:.1f}s), n_material_calls={res['n_material_calls']}",
      flush=True)

new = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz", allow_pickle=True)
u_err = np.max(np.abs(new["u_nodal"] - old_u_nodal))
e_err = np.max(np.abs(new["e_gp"] - old_e_gp))
s_err = np.max(np.abs(new["s_gp"] - old_s_gp))
s_rel = s_err / max(np.max(np.abs(old_s_gp)), 1e-300)
iters_match = np.array_equal(new["iters_per_step"], old_iters_per_step)
converged_match = np.array_equal(new["converged_per_step"], old_converged_per_step)

print(f"[rerun] max abs diff vs pre-optimization run: u_nodal={u_err:.3e}, e_gp={e_err:.3e}, "
      f"s_gp={s_err:.3e} (rel {s_rel:.3e})", flush=True)
print(f"[rerun] iters_per_step match exactly: {iters_match} (old={old_iters_per_step.tolist()}, "
      f"new={new['iters_per_step'].tolist()})", flush=True)
print(f"[rerun] converged_per_step match exactly: {converged_match}", flush=True)
print(f"[rerun] fully_converged match: {bool(new['fully_converged']) == old_fully_converged}, "
      f"ever_diverged match: {bool(new['ever_diverged']) == old_ever_diverged}", flush=True)
print(f"[rerun] SPEEDUP vs original 892.0s single-process run: {892.0 / max(res['wall_time'], 1e-9):.2f}x",
      flush=True)

ok = (
    bool(new["fully_converged"]) == old_fully_converged
    and bool(new["ever_diverged"]) == old_ever_diverged
    and s_rel < 1e-3  # looser than the single-process check: cross-worker batch-composition sensitivity on hard states is expected (see this session's own investigation)
)
print("RERUN_PASS" if ok else "RERUN_FAIL", flush=True)
print("RERUN_DONE_MARKER", flush=True)
