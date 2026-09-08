#!/usr/bin/env python3
"""Verify LinearHpromParallelContinuationWrapper reproduces the serial
LinearHpromContinuationWrapper's output (and internal q_prev_by_point
bookkeeping) across a sequence of calls with real strains, on a machine
that has NOT touched Kratos in this process before the pool is created
(fork-after-threading discipline) -- and reports the real wall-clock
speedup."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# IMPORTANT: create the pool before any Kratos-touching import in THIS
# (parent) process.
import linear_hprom_law_parallel_claude as par_module  # noqa: E402
par_module.ensure_persistent_executor(n_workers=16)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from linear_hprom_iterative_law_float64_claude import LinearHpromIterativeLawFloat64  # noqa: E402
from run_cook_linear_hprom_claude import LinearHpromContinuationWrapper  # noqa: E402

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

N = 120
rng = np.random.default_rng(3)
base_idx = rng.choice(e_gp.shape[0], size=N, replace=False)
E_base = e_gp[base_idx]

serial_law = LinearHpromIterativeLawFloat64()
serial_wrapper = LinearHpromContinuationWrapper(serial_law)
parallel_wrapper = par_module.LinearHpromParallelContinuationWrapper(n_workers=16)

N_CALLS = 4
ok = True
t_serial_total, t_parallel_total = 0.0, 0.0
for call_idx in range(N_CALLS):
    E_call = E_base * (1.0 + 0.03 * call_idx) + 0.01 * call_idx * np.array([1.0, -0.5, 0.3])

    t0 = time.perf_counter()
    S_ser, CC_ser = serial_wrapper(E_call)
    t_serial = time.perf_counter() - t0
    t_serial_total += t_serial

    t0 = time.perf_counter()
    S_par, CC_par = parallel_wrapper(E_call)
    t_parallel = time.perf_counter() - t0
    t_parallel_total += t_parallel

    s_err = np.max(np.abs(S_ser - S_par))
    s_rel = s_err / max(np.max(np.abs(S_ser)), 1e-300)
    cc_err = np.max(np.abs(CC_ser - CC_par))
    cc_rel = cc_err / max(np.max(np.abs(CC_ser)), 1e-300)

    n_prev_ser, n_prev_par = len(serial_wrapper.q_prev_by_point), len(parallel_wrapper.q_prev_by_point)
    q_prev_err = 0.0
    for i in serial_wrapper.q_prev_by_point:
        if i in parallel_wrapper.q_prev_by_point:
            q_prev_err = max(q_prev_err, np.max(np.abs(
                serial_wrapper.q_prev_by_point[i] - parallel_wrapper.q_prev_by_point[i]
            )))

    print(f"[verify] call {call_idx}: serial={t_serial:.3f}s, parallel={t_parallel:.3f}s "
          f"({t_serial / max(t_parallel, 1e-9):.2f}x), S rel err={s_rel:.3e}, CC rel err={cc_rel:.3e}, "
          f"n_prev(ser/par)={n_prev_ser}/{n_prev_par}, q_prev max abs diff={q_prev_err:.3e}", flush=True)

    if s_rel > 1e-8 or cc_rel > 1e-8 or n_prev_ser != n_prev_par or q_prev_err > 1e-8:
        ok = False

print(f"\n[verify] TOTAL: serial={t_serial_total:.3f}s, parallel={t_parallel_total:.3f}s "
      f"({t_serial_total / max(t_parallel_total, 1e-9):.2f}x)", flush=True)
print("VERIFY_PASS" if ok else "VERIFY_FAIL", flush=True)
