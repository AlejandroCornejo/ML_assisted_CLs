#!/usr/bin/env python3
"""Correctness gate for the warm-start support just added to
fom_nested_consistent_law_claude.evaluate_with_tangent.

The claim being tested is NOT "warm-starting is a good approximation" -- it
is that warm-starting changes NOTHING about the converged answer, because
this RVE is path-independent (Neo-Hookean hyperelastic, no history), so the
solution is a pure function of E and the from-zero ramp is only a Newton-
convergence aid. If that claim holds, S and CC from a warm start must match
a cold start to solver tolerance, not merely "closely".

Nothing downstream should be built on warm-starting unless this passes.
"""
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

from fom_nested_consistent_law_claude import evaluate_with_tangent  # noqa: E402

# Representative of the final dogbone state (force-controlled, F=1e9):
# E11 near its max, E22 near its Poisson-driven min, modest shear.
E_TARGET = np.array([0.2578, -0.0968, 0.0825])
# A "previous macro Newton iterate" a few percent away, i.e. exactly the
# kind of small increment warm-starting is meant to exploit.
E_PREV = 0.95 * E_TARGET


def rel(a, b):
    den = max(np.linalg.norm(b), 1.0e-30)
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / den)


if __name__ == "__main__":
    print(f"E_target = {E_TARGET}", flush=True)
    print(f"E_prev   = {E_PREV}", flush=True)

    t0 = time.perf_counter()
    S_cold, CC_cold = evaluate_with_tangent(E_TARGET, out_dir=str(HERE / "_warmcheck_cold"))
    t_cold = time.perf_counter() - t0
    print(f"\n[cold]  {t_cold:.1f}s   S={S_cold}", flush=True)

    # Build the warm-start state: one cold solve at E_prev, keeping its u.
    t0 = time.perf_counter()
    _S_prev, _CC_prev, u_prev = evaluate_with_tangent(
        E_PREV, out_dir=str(HERE / "_warmcheck_prev"), return_u=True)
    t_prev = time.perf_counter() - t0
    print(f"[prep]  {t_prev:.1f}s   (cold solve at E_prev, to obtain u_prev)", flush=True)

    t0 = time.perf_counter()
    S_warm, CC_warm, _u = evaluate_with_tangent(
        E_TARGET, u_init=u_prev, E_start=E_PREV,
        out_dir=str(HERE / "_warmcheck_warm"), return_u=True)
    t_warm = time.perf_counter() - t0
    print(f"[warm]  {t_warm:.1f}s   S={S_warm}", flush=True)

    err_S = rel(S_warm, S_cold)
    err_CC = rel(CC_warm, CC_cold)
    print(f"\nrelative difference warm vs cold:  S {err_S:.3e}   CC {err_CC:.3e}", flush=True)
    print(f"speedup on the target solve:       {t_cold / max(t_warm, 1e-9):.1f}x "
          f"({t_cold:.1f}s -> {t_warm:.1f}s)", flush=True)

    # Solver tolerances in this project's own FOM runs are 1e-6 relative, so
    # anything at/below ~1e-5 is agreement to solver tolerance; 1e-3+ would
    # mean the two paths genuinely disagree and the premise is wrong.
    ok = (err_S < 1.0e-5) and (err_CC < 1.0e-5)
    print("WARM_START_VERIFY_PASS" if ok else "WARM_START_VERIFY_FAIL", flush=True)
