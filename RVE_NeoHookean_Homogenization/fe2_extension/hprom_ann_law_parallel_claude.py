#!/usr/bin/env python3
"""Parallelized HpromIterativeContinuationWrapper: two-level parallelism.

Unlike LinearHpromIterativeLawFloat64 (no torch at all, purely Kratos-CPU-
bound), HpromAnnIterativeLawFloat64's dominant cost (74.5% measured this
session) IS torch/decoder-Jacobian+Hessian dispatch overhead, already
batched across many macro Gauss points at once via evaluate_with_tangent_
batch (vmap(jacfwd)/vmap(torch.func.hessian)) within a single process.
That batching's own speedup (verified this session: ~3.3x at a 40-point
batch) still holds at the ~600/16=37-point chunk size each worker gets
here, so this module does NOT abandon vmap batching for a plain per-point
multiprocessing loop (that would throw away the bigger win) -- instead
each worker calls evaluate_with_tangent_batch ONCE on its own chunk,
composing vmap-within-worker with process-level parallelism across
workers. The remaining (smaller, Kratos-native) per-point costs inside
that batch call get the SAME cross-core speedup LinearHpromIterativeLaw
Float64's own parallel wrapper gets.

q_prev/step_index are threaded EXPLICITLY per point in the arguments sent
to each worker (never held as implicit worker-side state), exactly as in
linear_hprom_law_parallel_claude.py -- which worker processes which point
on a given call cannot affect correctness; the main process's own
q_prev_by_point dict (identical bookkeeping and "commit only if converged"
rule to HpromIterativeContinuationWrapper's serial version) is the single
source of truth for warm-starting.

Same fork-after-Kratos/torch-threading discipline as fom_nested_law_
parallel_claude.py and linear_hprom_law_parallel_claude.py: ensure_
persistent_executor() must be called before ANY Kratos- or torch-
touching code runs in the parent process (this module's own top level
imports neither -- the law import is lazy, inside _worker, so it only
ever happens post-fork, in the child), and the SAME pool is reused for
every subsequent call.
"""
from __future__ import annotations

import multiprocessing
import os

# Defense in depth against the fork-after-CUDA-init hazard: see
# dhprom_ann_law_parallel_claude.py's own comment on this exact line for
# the full explanation (confirmed the hard way this session, running
# PANN -- which does not force CPU-only torch -- before this pool's own
# first, lazily-spawned dispatch).
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
    """Call this FIRST, before importing anything Kratos- or torch-
    related, from any driver script that will run more than one macro
    Newton call through HpromAnnParallelContinuationWrapper. Idempotent."""
    global _PERSISTENT_EXECUTOR
    if _PERSISTENT_EXECUTOR is None:
        _PERSISTENT_EXECUTOR = ProcessPoolExecutor(max_workers=n_workers, mp_context=_FORK_CTX)
    return _PERSISTENT_EXECUTOR


_WORKER_LAW = None  # persists across .map() calls WITHIN one forked worker process's lifetime


def _worker(args):
    global _WORKER_LAW
    # One worker per core provides the parallelism; letting torch/Kratos/
    # BLAS also spawn their own internal threads per worker would
    # oversubscribe the machine. Must be set before torch/numpy import
    # inside this (forked) worker to be honored (torch's own
    # set_num_threads(1) inside the law's __init__ additionally caps its
    # own pool once built, per-worker).
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    E_chunk, q_prev_chunk, step_index_chunk, hprom_ann_dir = args
    if _WORKER_LAW is None:
        # hprom_ann_iterative_law_float64_claude.py itself imports
        # KratosMultiphysics BEFORE appending KRATOS_PATH to sys.path
        # (relies on its caller having done so already, same as every
        # other script this session that imports it directly) -- this
        # forked worker's sys.path is whatever the parent had at fork
        # time, which by design (see this module's own docstring) never
        # touches Kratos, so it must be added here explicitly.
        kratos_path = "/home/kratos/Kratos_Eigen_Check/bin/Release"
        if kratos_path not in sys.path:
            sys.path.append(kratos_path)
        import hprom_ann_iterative_law_float64_claude as law_module
        _WORKER_LAW = law_module.HpromAnnIterativeLawFloat64(
            hprom_ann_dir=str(hprom_ann_dir), qp_init_mode="continuation",
        )
    law = _WORKER_LAW

    n = E_chunk.shape[0]
    q_prev_batch = np.zeros((n, law.n_primary), dtype=float)
    for i in range(n):
        if q_prev_chunk[i] is not None:
            q_prev_batch[i] = q_prev_chunk[i]
    step_index_batch = np.asarray(step_index_chunk, dtype=int)

    _, S, q_p_new, _n_it, converged, _, CC, _, _ = law.evaluate_with_tangent_batch(
        E_chunk, q_prev_batch=q_prev_batch, step_index_batch=step_index_batch,
    )
    return S, CC, list(q_p_new), list(converged)


class HpromAnnParallelContinuationWrapper:
    """Same per-point warm-start / commit-only-if-converged discipline as
    HpromIterativeContinuationWrapper (run_cruciform_fe2_claude.py), but
    splits the n macro Gauss points across a persistent process pool, with
    each worker batching its own chunk internally via evaluate_with_
    tangent_batch instead of looping one point at a time."""

    def __init__(self, hprom_ann_dir, n_workers=16):
        self.hprom_ann_dir = str(hprom_ann_dir)
        self.n_workers = int(n_workers)
        self.q_prev_by_point = {}

    def __call__(self, e_voigt, young=None, poisson=None):
        e_voigt = np.asarray(e_voigt, dtype=float)
        n = e_voigt.shape[0]

        q_prev_list = [self.q_prev_by_point.get(i) for i in range(n)]
        step_index_list = [1 if q is None else 2 for q in q_prev_list]

        n_workers = max(1, min(self.n_workers, n))
        idx_chunks = [c for c in np.array_split(np.arange(n), n_workers) if c.size > 0]
        chunks = [
            (e_voigt[c], [q_prev_list[i] for i in c], [step_index_list[i] for i in c], self.hprom_ann_dir)
            for c in idx_chunks
        ]

        executor = ensure_persistent_executor(self.n_workers)
        results = list(executor.map(_worker, chunks))

        S = np.concatenate([r[0] for r in results], axis=0)
        CC = np.concatenate([r[1] for r in results], axis=0)
        q_p_new_flat, converged_flat = [], []
        for r in results:
            q_p_new_flat.extend(r[2])
            converged_flat.extend(r[3])

        idx_flat = np.concatenate(idx_chunks)
        n_nonconverged = 0
        for local_j, i in enumerate(idx_flat):
            if converged_flat[local_j]:
                self.q_prev_by_point[int(i)] = q_p_new_flat[local_j]
            else:
                n_nonconverged += 1
        if n_nonconverged:
            print(f"    [hprom-ann-parallel-continuation] {n_nonconverged}/{n} points "
                  "did not converge internally this call")
        return S, CC
