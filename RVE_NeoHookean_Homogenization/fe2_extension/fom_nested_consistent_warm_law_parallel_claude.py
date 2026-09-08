#!/usr/bin/env python3
"""Parallel wrapper for fom_nested_consistent_warm_law_claude.

Warm-start state is held by the PARENT, not by the workers, and the relevant
slice is shipped to each chunk and returned updated. That is deliberate: an
earlier version of this file kept state in worker module globals and relied
on chunk i always reaching worker i. ProcessPoolExecutor.map gives no such
affinity -- tasks come off a shared queue and go to whichever worker is idle
-- so a point's state frequently sat in a worker that did not receive that
point, which silently cold-started instead. That produced perfectly correct
values (cold and warm agree by construction) with almost no speedup, and it
is exactly the kind of failure a value-only check cannot see, which is why
the verification here counts warm starts instead of only comparing numbers.

IPC cost of shipping state: ~34 KB per point (4244 free dofs, float64), so
~56 MB each way for this project's 1656-Gauss-point dogbone -- under a second
of pickling against a saving of tens of nonlinear substeps per point.

Same fork-after-Kratos/torch-threading discipline as every other
*_law_parallel_claude.py here: call ensure_persistent_executor BEFORE the
parent process touches Kratos.
"""
from __future__ import annotations

import multiprocessing
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_FORK_CTX = multiprocessing.get_context("fork")
_PERSISTENT_EXECUTOR = None

# Parent-held authoritative state: {global_gp_index: (E_prev (3,), u_prev (n_dof,))}
_STATE = {}
_LAST_WARM_FRACTION = None


def ensure_persistent_executor(n_workers=16):
    """Call FIRST, before the parent imports/uses Kratos. Idempotent."""
    global _PERSISTENT_EXECUTOR
    if _PERSISTENT_EXECUTOR is None:
        _PERSISTENT_EXECUTOR = ProcessPoolExecutor(max_workers=n_workers, mp_context=_FORK_CTX)
    return _PERSISTENT_EXECUTOR


def reset_state():
    """Drop warm-start state (e.g. between independent macro problems)."""
    _STATE.clear()


def last_warm_fraction():
    """Fraction of Gauss points warm-started on the most recent call --
    the direct observable that the mechanism is actually engaged. 0.0 on
    the first call of a run (nothing to continue from yet), ~1.0 after."""
    return _LAST_WARM_FRACTION


def _worker(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    E_chunk, gp_chunk, state_chunk, worker_id, silence = args
    from fom_nested_consistent_warm_law_claude import (
        fom_nested_consistent_warm_pk2_2d_vectorized,
    )
    out_dir = str(HERE / f"fom_nested_warm_scratch_worker{worker_id}")
    return fom_nested_consistent_warm_pk2_2d_vectorized(
        E_chunk, gp_indices=gp_chunk, out_dir=out_dir, verbose=False, silence=silence,
        state_in=state_chunk, return_state=True)


def fom_nested_consistent_warm_pk2_2d_vectorized_parallel(E_flat, young=None, poisson=None,
                                                            verbose=True, n_workers=None,
                                                            silence=True):
    global _LAST_WARM_FRACTION
    E_flat = np.asarray(E_flat, dtype=float).reshape(-1, 3)
    n = E_flat.shape[0]
    if n_workers is None:
        n_workers = min(16, os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, n))

    gp_all = np.arange(n, dtype=np.int64)
    E_chunks = [c for c in np.array_split(E_flat, n_workers, axis=0) if len(c) > 0]
    gp_chunks = [c for c in np.array_split(gp_all, n_workers, axis=0) if len(c) > 0]
    args = []
    for i, (E_c, gp_c) in enumerate(zip(E_chunks, gp_chunks)):
        state_c = {int(g): _STATE[int(g)] for g in gp_c if int(g) in _STATE}
        args.append((E_c, gp_c, state_c, i, silence))

    t0 = time.perf_counter()
    if _PERSISTENT_EXECUTOR is not None:
        results = list(_PERSISTENT_EXECUTOR.map(_worker, args))
    else:
        with ProcessPoolExecutor(max_workers=len(args), mp_context=_FORK_CTX) as ex:
            results = list(ex.map(_worker, args))

    S = np.concatenate([r[0] for r in results], axis=0)
    CC = np.concatenate([r[1] for r in results], axis=0)
    n_warm = int(sum(r[3] for r in results))
    for r in results:
        _STATE.update(r[2])
    _LAST_WARM_FRACTION = n_warm / max(n, 1)

    if verbose:
        dt = time.perf_counter() - t0
        print(f"    [fom_nested_consistent_warm_parallel] {n} Gauss points across {len(args)} "
              f"workers, {dt:.1f}s wall ({dt / max(n, 1):.3f}s/GP), "
              f"{n_warm}/{n} warm-started, {len(_STATE)} points in state", flush=True)
    return S, CC
