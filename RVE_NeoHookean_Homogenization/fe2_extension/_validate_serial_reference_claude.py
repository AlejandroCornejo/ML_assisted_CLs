#!/usr/bin/env python3
"""Standalone (deliberately separate-process) serial reference for
fom_nested_law_parallel_claude.py's own validation -- run as its own
`python3` invocation, never in the same process as the pool-creating
script, so there is no shared Kratos/OpenMP parent state either way."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fom_nested_law_claude import fom_nested_pk2_2d_vectorized  # noqa: E402

point_b = np.array([[0.0, 0.0, 0.0], [-0.008, 0.012, 0.002]])
S, CC = fom_nested_pk2_2d_vectorized(point_b, verbose=False)
np.savez(HERE / "_validate_serial_result_claude.npz", S=S, CC=CC, E=point_b)
print(f"[serial] S={S.tolist()}")
print("SERIAL_DONE_MARKER")
