#!/usr/bin/env python3
"""Produces the SERIAL dhprom_ann_pk2_2d_vectorized_consistent_float64
reference results on the EXACT SAME (E_call, E_call*1.02) two-call
sequence that _test_dhprom_ann_parallel_alone_claude.py used for the
parallel path -- run as a fully separate process for a clean comparison."""
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

import dhprom_ann_direct_law_float64_claude as m  # noqa: E402

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
assert m._DEFAULT_LAW_F64 is None
m.get_law_float64(hprom_ann_dir=str(DHPROMANN_DIR))

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

N = 600
rng = np.random.default_rng(5)
idx = rng.choice(e_gp.shape[0], size=min(N, e_gp.shape[0]), replace=True)
E_call = e_gp[idx]

S, CC = m.dhprom_ann_pk2_2d_vectorized_consistent_float64(E_call)
S2, CC2 = m.dhprom_ann_pk2_2d_vectorized_consistent_float64(E_call * 1.02)

np.savez(HERE / "_serial_reference_for_dhprom_ann_parallel_verify_result_claude.npz",
         S=S, CC=CC, S2=S2, CC2=CC2, E=E_call)
print("SERIAL_REF_DONE_MARKER", flush=True)
