#!/usr/bin/env python3
"""Produces the SERIAL HpromIterativeContinuationWrapper reference results
on the EXACT SAME (E_call, E_call*1.02) two-call sequence that
_test_hprom_ann_parallel_alone_claude.py used for the parallel path -- run
as a fully separate process so the comparison is apples-to-apples with no
shared-process fork hazard."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import run_cruciform_fe2_claude as rc  # noqa: E402

HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"
wrapper = rc.make_hprom_continuation_material_func(HPROMANN_DIR)

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

N = 600
rng = np.random.default_rng(4)
idx = rng.choice(e_gp.shape[0], size=min(N, e_gp.shape[0]), replace=True)
E_call = e_gp[idx]

S, CC = wrapper(E_call)
S2, CC2 = wrapper(E_call * 1.02)

np.savez(HERE / "_serial_reference_for_hprom_ann_parallel_verify_result_claude.npz",
         S=S, CC=CC, S2=S2, CC2=CC2, E=E_call,
         q_prev_keys=np.array(list(wrapper.q_prev_by_point.keys())),
         q_prev_vals=np.array([wrapper.q_prev_by_point[k] for k in wrapper.q_prev_by_point]))
print(f"[serial_ref] done, {len(wrapper.q_prev_by_point)}/{len(E_call)} points tracked as converged", flush=True)
print("SERIAL_REF_DONE_MARKER", flush=True)
