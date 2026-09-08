#!/usr/bin/env python3
"""Parallelized fom_nested_consistent_law_claude.py.

Same simple case as dhprom_ann_law_parallel_claude.py: no continuation/
warm-start state at all (fom_nested_consistent_pk2_2d_vectorized cold-
starts every solve from E=0, exactly matching the original fom_nested_
law_claude.py's own always-cold-start behavior -- this correction changes
HOW the stress/tangent are computed, not the solve's own warm-start
policy), so this is a pure function of E_flat: split into chunks, dispatch
each to a persistent worker, concatenate. Each worker calls the EXISTING,
already-verified fom_nested_consistent_pk2_2d_vectorized on its own chunk
unmodified.

Same fork-after-Kratos/torch-threading discipline as every other
*_law_parallel_claude.py this session.
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_FORK_CTX = multiprocessing.get_context("fork")
_PERSISTENT_EXECUTOR = None


def ensure_persistent_executor(n_workers=16):
    """Call this FIRST, before importing anything Kratos-related, from
    any driver script that will run more than one macro Newton call
    through fom_nested_consistent_pk2_2d_vectorized_parallel. Idempotent."""
    global _PERSISTENT_EXECUTOR
    if _PERSISTENT_EXECUTOR is None:
        _PERSISTENT_EXECUTOR = ProcessPoolExecutor(max_workers=n_workers, mp_context=_FORK_CTX)
    return _PERSISTENT_EXECUTOR


def _worker(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    E_chunk, worker_id = args
    from fom_nested_consistent_law_claude import fom_nested_consistent_pk2_2d_vectorized
    out_dir = str(HERE / f"fom_nested_consistent_scratch_worker{worker_id}")
    return fom_nested_consistent_pk2_2d_vectorized(E_chunk, out_dir=out_dir, verbose=False)


def fom_nested_consistent_pk2_2d_vectorized_parallel(E_flat, young=None, poisson=None,
                                                        verbose=True, n_workers=None):
    E_flat = np.asarray(E_flat, dtype=float).reshape(-1, 3)
    n = E_flat.shape[0]
    if n_workers is None:
        n_workers = min(16, os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, n))

    chunks = [c for c in np.array_split(E_flat, n_workers, axis=0) if len(c) > 0]
    args = [(chunk, i) for i, chunk in enumerate(chunks)]

    import time
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
        print(f"    [fom_nested_consistent_parallel] {n} Gauss points across {len(args)} workers, "
              f"{dt:.1f}s wall ({dt / max(n, 1):.3f}s/GP wall-per-point)", flush=True)
    return S, CC
