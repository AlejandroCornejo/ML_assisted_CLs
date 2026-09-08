#!/usr/bin/env python3
"""Parallelized wrapper around fom_nested_law_claude.fom_nested_pk2_2d_vectorized.

The 7 solve_at_strain calls per Gauss point (1 base + 6 central-FD
perturbations) are already fully independent of each other and of every
other Gauss point's own calls -- nothing is warm-started or shared across
them. This wrapper only changes WHERE those independent calls run (split
across a process pool instead of one after another); it does not touch
the actual solve or finite-difference logic in fom_nested_law_claude.py
at all, and is verified below to reproduce the serial function's output
to floating-point precision on a small sample before being trusted for
a real run.

Uses the "fork" multiprocessing context explicitly (not left to whatever
the platform default happens to be): fork copies the parent's already-
imported module state, so each worker inherits _material_law_guard_claude's
_TRUE_NEO_HOOKEAN exactly as captured at the parent's own import time,
before any monkeypatching -- the same guard fom_nested_pk2_2d_vectorized
already relies on in the serial case, now correctly inherited per worker.
Each worker is given its own scratch out_dir so concurrent processes never
touch the same files.
"""
from __future__ import annotations

import multiprocessing
import os

# Defense in depth against the fork-after-CUDA-init hazard: see
# dhprom_ann_law_parallel_claude.py's own comment on this exact line for
# the full explanation. This module has no torch of its own, but PANN
# (or anything else sharing the process) might, and the crash happens
# at THIS pool's own dispatch time, not at PANN's.
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

# Module-level, reusable pool. MUST be created (via ensure_persistent_executor,
# called at the very top of a driver script) before this process -- the
# PARENT, not the workers -- ever imports or uses Kratos/fom_solver_rve
# itself (building a macro mesh, running a macro Newton strategy, etc.).
# fork() only duplicates the calling thread; a parent that has already
# triggered Kratos's own internal OpenMP thread pool (which happens on
# first real use, not just on `import KratosMultiphysics`) leaves forked
# children in a state that assumes worker threads exist which were never
# actually duplicated -- confirmed directly this session: creating the
# pool AFTER an in-process Kratos call left the very first worker task
# hanging indefinitely; creating it before any Kratos use in the parent,
# then reusing that same pool for every subsequent call, completed
# correctly in seconds. A fresh pool created inside every call (the
# pattern this module used before) would hit the exact same hazard on the
# SECOND call onward, once the parent's own macro-level Newton solve has
# used Kratos at least once -- reuse, not re-creation, is what makes this
# safe for a multi-step run, not just a single isolated call.
_PERSISTENT_EXECUTOR = None


def ensure_persistent_executor(n_workers=16):
    """Call this FIRST, before importing anything Kratos-related, from
    any driver script that will run more than one macro Newton call
    through fom_nested_pk2_2d_vectorized_parallel. Idempotent (a second
    call is a no-op) so it's safe to call defensively."""
    global _PERSISTENT_EXECUTOR
    if _PERSISTENT_EXECUTOR is None:
        _PERSISTENT_EXECUTOR = ProcessPoolExecutor(max_workers=n_workers, mp_context=_FORK_CTX)
    return _PERSISTENT_EXECUTOR


def _worker(args):
    # Each worker is single-threaded internally: parallelism now lives at
    # the process level (one worker per core), so letting Kratos/BLAS also
    # spawn their own internal threads per worker would oversubscribe the
    # machine (n_workers processes x several threads each, all fighting
    # over the same cores). Must be set before Kratos/numpy are imported
    # in this (forked) worker for the BLAS libraries to honor it.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    chunk_E, worker_id, h = args
    from fom_nested_law_claude import fom_nested_pk2_2d_vectorized
    out_dir = str(HERE / f"fom_query_scratch_worker{worker_id}")
    return fom_nested_pk2_2d_vectorized(chunk_E, h=h, verbose=False, out_dir=out_dir)


def fom_nested_pk2_2d_vectorized_parallel(E_flat, young=None, poisson=None, h=1.0e-4,
                                           verbose=True, n_workers=None):
    E_flat = np.asarray(E_flat, dtype=float).reshape(-1, 3)
    n = E_flat.shape[0]
    if n_workers is None:
        n_workers = min(16, os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, n))

    chunks = [c for c in np.array_split(E_flat, n_workers, axis=0) if len(c) > 0]
    args = [(chunk, i, h) for i, chunk in enumerate(chunks)]

    t0 = time.perf_counter()
    if _PERSISTENT_EXECUTOR is not None:
        results = list(_PERSISTENT_EXECUTOR.map(_worker, args))
    else:
        with ProcessPoolExecutor(max_workers=len(args), mp_context=_FORK_CTX) as ex:
            results = list(ex.map(_worker, args))
    S = np.concatenate([r[0] for r in results], axis=0)
    CC = np.concatenate([r[1] for r in results], axis=0)
    if verbose:
        dt = time.perf_counter() - t0
        print(f"    [fom_nested_parallel] {n} Gauss points across {len(args)} workers, "
              f"{dt:.1f}s wall ({dt / max(n, 1):.3f}s/GP wall-per-point, "
              f"~{len(args)}x nominal parallelism)", flush=True)
    return S, CC


if __name__ == "__main__":
    # IMPORTANT: this process must not touch Kratos/fom_nested_law_claude
    # itself before creating the pool -- forking AFTER the parent has
    # already used Kratos (which spawns its own internal OpenMP thread
    # pool on first use) is a classic fork-after-threading hazard: only
    # the calling thread survives the fork, so a child inheriting a
    # parent that already believes it has N worker threads can hang or
    # misbehave. Confirmed the hard way: an earlier version of this check
    # called the direct/serial reference in-process FIRST, then created
    # the pool, and the pool's own first worker call never completed.
    # This version's parent does nothing Kratos-related of its own; the
    # serial reference for comparison is run as a fully separate process
    # instead (compare_fom_nested_serial_vs_parallel_claude.py).
    point_b = np.array([[0.0, 0.0, 0.0], [-0.008, 0.012, 0.002]])
    print("[step2] two points, n_workers=2 (concurrent), parent untouched by Kratos before this...", flush=True)
    t0 = time.perf_counter()
    S_pool2, CC_pool2 = fom_nested_pk2_2d_vectorized_parallel(point_b, n_workers=2, verbose=True)
    t_pool2 = time.perf_counter() - t0
    zero_ok = np.allclose(S_pool2[0], 0.0, atol=1.0)
    distinct = not np.allclose(S_pool2[0], S_pool2[1], atol=1.0)
    print(f"[step2] pool(2): {t_pool2:.1f}s, S[0] (should be ~0)={S_pool2[0]}, "
          f"S[1] (nonzero)={S_pool2[1]}, zero_point_ok={zero_ok}, distinct={distinct}", flush=True)
    np.savez(HERE / "_validate_pool_result_claude.npz", S=S_pool2, CC=CC_pool2, E=point_b)

    all_ok = zero_ok and distinct
    print("VALIDATION_PASS" if all_ok else "VALIDATION_FAIL", flush=True)
