#!/usr/bin/env python3
"""Genuine (non-reduced) FE^2 material law for Cook: no surrogate, no
reduced-order model at all. At every Gauss point, solves the full
990-element RVE FOM from a cold start (reusing the already-validated
time_fom_single_query_claude.solve_at_strain), and finite-differences a
tangent from 6 extra perturbed solves (central differences, 3 independent
Voigt strain components). This is the "true" multiscale reference the
ROM/HPROM and PANN routes are compared against.

Not meant for a full 20-step ramp (see run_cook_hprom_ann_claude.py's own
cost discussion) -- one call over all 384 Gauss points of the nx=8 Cook
mesh costs on the order of 384*7*~0.7s, and a single macro Newton
iteration needs at least one such call. Exploratory use only
(run_cook_fom_nested_step1_claude.py), not wired into any paper result.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from time_fom_single_query_claude import solve_at_strain  # noqa: E402
from _material_law_guard_claude import true_neo_hookean_active  # noqa: E402

_N_CALLS = 0


def fom_nested_pk2_2d_vectorized(E_flat, young=None, poisson=None, h=1.0e-4, verbose=True, out_dir=None):
    """(E_flat (N,3), young, poisson) -> (S (N,3), CC (N,3,3)), the exact
    contract Cook's VectorizedAssembler expects from
    fom._neo_hookean_pk2_2d_vectorized. young/poisson accepted but unused
    (signature compatibility only -- the true microscale RVE ignores the
    macro-law's own nominal properties).

    Wrapped in true_neo_hookean_active(): fom_solver_rve._neo_hookean_pk2_2d_vectorized
    is a single module-global name. Whoever calls this function has already
    monkeypatched it to point HERE (that is how this function gets invoked at
    all), but solve_at_strain's own internal RVE solve uses that exact same
    global for its own, unrelated VectorizedAssembler -- without restoring the
    true law first, this function would recursively call itself inside its own
    inner solve (an RVE inside an RVE inside an RVE...) until Python's
    recursion limit kills the process. Confirmed the hard way: the first
    attempt at this script did exactly that."""
    global _N_CALLS
    _N_CALLS += 1
    E_flat = np.asarray(E_flat, dtype=float).reshape(-1, 3)
    n = E_flat.shape[0]
    S = np.zeros((n, 3), dtype=float)
    CC = np.zeros((n, 3, 3), dtype=float)
    t0 = time.perf_counter()
    for i in range(n):
        e0 = E_flat[i]
        with true_neo_hookean_active():
            _eps0, sig0 = solve_at_strain(e0, out_dir=out_dir)
            S[i] = sig0
            for k in range(3):
                ep = e0.copy()
                ep[k] += h
                em = e0.copy()
                em[k] -= h
                _epsp, sigp = solve_at_strain(ep, out_dir=out_dir)
                _epsm, sigm = solve_at_strain(em, out_dir=out_dir)
                CC[i, :, k] = (sigp - sigm) / (2.0 * h)
        if verbose and (i + 1) % 16 == 0:
            dt = time.perf_counter() - t0
            print(f"    [fom_nested call #{_N_CALLS}] {i + 1}/{n} Gauss points done "
                  f"({dt:.1f}s elapsed, {dt / (i + 1):.3f}s/GP)", flush=True)
    if verbose:
        dt = time.perf_counter() - t0
        print(f"    [fom_nested call #{_N_CALLS}] DONE: {n} Gauss points, {dt:.1f}s total "
              f"({dt / max(n, 1):.3f}s/GP)", flush=True)
    return S, CC
