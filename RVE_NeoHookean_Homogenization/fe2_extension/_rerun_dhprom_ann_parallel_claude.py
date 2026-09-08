#!/usr/bin/env python3
"""Re-run the full 20-step cruciform D-HPROM-ANN row (n_body=6, same
config as the existing cruciform_results_dhprom_f64_consistent_claude.npz,
itself already decoder-batched + reaction-force-batched, serial) with the
process-parallel material function (dhprom_ann_law_parallel_claude.py),
to confirm end-to-end correctness and report the real wall-clock speedup.

Same fork-after-Kratos/torch-threading discipline as every other parallel
rerun this session: set thread-limiting env vars, create the persistent
pool, THEN import run_cruciform_fe2_claude."""
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

import dhprom_ann_law_parallel_claude as par_module  # noqa: E402
par_module.ensure_persistent_executor(n_workers=16)

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
N_BODY, N_ARM_LEN = 6, 4
DELTA = 1.2
N_STEPS = 20

old = np.load(HERE / "cruciform_results_dhprom_f64_consistent_claude.npz", allow_pickle=True)
old_u_nodal = old["u_nodal"].copy()
old_e_gp = old["e_gp"].copy()
old_s_gp = old["s_gp"].copy()
old_iters_per_step = old["iters_per_step"].copy()
old_converged_per_step = old["converged_per_step"].copy()
old_fully_converged = bool(old["fully_converged"])
old_ever_diverged = bool(old["ever_diverged"])
print(f"[rerun] pre-parallel reference: fully_converged={old_fully_converged}, "
      f"ever_diverged={old_ever_diverged}, iters_per_step={old_iters_per_step.tolist()}", flush=True)

import run_cruciform_fe2_claude as rc  # noqa: E402

material_func = rc.make_dhprom_ann_parallel_material_func(DHPROMANN_DIR, n_workers=16)
rc.MATERIAL_FUNCS["dhprom_ann_parallel"] = material_func

print("=== D-HPROM-ANN-FE2 PARALLEL RERUN ===", flush=True)
t0 = time.time()
res = rc.run_newton_fe2_cruciform(
    "dhprom_ann_parallel", n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
    delta_x_final=DELTA, delta_y_final=DELTA, verbose=True, use_line_search=True, save_npz=True,
)
wall = time.time() - t0
print(f"[rerun] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
      f"wall={res['wall_time']:.1f}s (elapsed {wall:.1f}s), n_material_calls={res['n_material_calls']}",
      flush=True)

new = np.load(HERE / "cruciform_results_dhprom_ann_parallel_claude.npz", allow_pickle=True)
u_err = np.max(np.abs(new["u_nodal"] - old_u_nodal))
e_err = np.max(np.abs(new["e_gp"] - old_e_gp))
s_err = np.max(np.abs(new["s_gp"] - old_s_gp))
s_rel = s_err / max(np.max(np.abs(old_s_gp)), 1e-300)
iters_match = np.array_equal(new["iters_per_step"], old_iters_per_step)
converged_match = np.array_equal(new["converged_per_step"], old_converged_per_step)

print(f"[rerun] max abs diff vs pre-parallel run: u_nodal={u_err:.3e}, e_gp={e_err:.3e}, "
      f"s_gp={s_err:.3e} (rel {s_rel:.3e})", flush=True)
print(f"[rerun] iters_per_step match exactly: {iters_match} (old={old_iters_per_step.tolist()}, "
      f"new={new['iters_per_step'].tolist()})", flush=True)
print(f"[rerun] converged_per_step match exactly: {converged_match}", flush=True)
print(f"[rerun] fully_converged match: {bool(new['fully_converged']) == old_fully_converged}, "
      f"ever_diverged match: {bool(new['ever_diverged']) == old_ever_diverged}", flush=True)
print(f"[rerun] SPEEDUP vs prior serial (decoder+reaction-force-batched) 217.5s run: "
      f"{217.5 / max(res['wall_time'], 1e-9):.2f}x", flush=True)
print(f"[rerun] SPEEDUP vs ORIGINAL fully-unoptimized 639.1s run: "
      f"{639.1 / max(res['wall_time'], 1e-9):.2f}x", flush=True)

ok = (
    bool(new["fully_converged"]) == old_fully_converged
    and bool(new["ever_diverged"]) == old_ever_diverged
    and iters_match and converged_match
    and s_rel < 1e-6
)
print("RERUN_PASS" if ok else "RERUN_FAIL", flush=True)
print("RERUN_DONE_MARKER", flush=True)
