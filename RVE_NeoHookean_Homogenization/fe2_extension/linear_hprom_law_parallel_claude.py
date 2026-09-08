#!/usr/bin/env python3
"""Parallelized LinearHpromContinuationWrapper.

LinearHpromIterativeLawFloat64's dominant per-point costs (profiled this
session: native Kratos res_assembler.Assemble ~26%, SetDisplacementFrom
EquationVector/UpdateCurrentCoordinatesFromDisplacement ~35% combined,
vec_assembler.ComputeLocalArrays ~5%) are all compiled Kratos C++ calls,
not torch/ANN dispatch overhead -- so unlike D-HPROM-ANN/HPROM-ANN, there
is no vmap-style batching lever here. The ~600 macro Gauss points ARE
independent of each other within one macro-Newton material-function call,
so the lever is process-level parallelism instead: split them across a
persistent pool of worker processes, each holding its own
LinearHpromIterativeLawFloat64 instance built ONCE (on that worker's first
task) and reused for the rest of the run.

q_prev/step_index are threaded EXPLICITLY per point in the arguments sent
to each worker (never held as implicit worker-side state) -- which worker
happens to process which point on a given call therefore cannot affect
correctness. The main process's own q_prev_by_point dict (identical
bookkeeping to LinearHpromContinuationWrapper's serial version, including
its "only commit a CONVERGED q_p as the next warm start" rule) remains the
single source of truth for warm-starting.

Same fork-after-threading discipline as fom_nested_law_parallel_claude.py
(verified this session for that module): ensure_persistent_executor() must
be called before ANY Kratos-touching code runs in the parent process, and
the SAME pool is reused for every subsequent call (recreating it per call
would hit the fork-after-Kratos-threading hazard on the second call
onward, once the parent's own macro Newton solve has used Kratos at
least once).
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
    """Call this FIRST, before importing anything Kratos-related, from any
    driver script that will run more than one macro Newton call through
    LinearHpromParallelContinuationWrapper. Idempotent."""
    global _PERSISTENT_EXECUTOR
    if _PERSISTENT_EXECUTOR is None:
        _PERSISTENT_EXECUTOR = ProcessPoolExecutor(max_workers=n_workers, mp_context=_FORK_CTX)
    return _PERSISTENT_EXECUTOR


_WORKER_LAW = None  # persists across .map() calls WITHIN one forked worker process's lifetime


def _worker(args):
    global _WORKER_LAW
    # One worker per core now provides the parallelism; letting Kratos/BLAS
    # also spawn their own internal threads per worker would oversubscribe
    # the machine. Must be set before Kratos/numpy import inside this
    # (forked) worker for the BLAS libraries to honor it.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    E_chunk, q_prev_chunk, step_index_chunk = args
    if _WORKER_LAW is None:
        import linear_hprom_iterative_law_float64_claude as law_module
        _WORKER_LAW = law_module.LinearHpromIterativeLawFloat64()
    law = _WORKER_LAW

    n = E_chunk.shape[0]
    S = np.zeros((n, 3), dtype=float)
    CC = np.zeros((n, 3, 3), dtype=float)
    q_p_new = [None] * n
    converged = [False] * n
    for i in range(n):
        _, S[i], q_p_new[i], _n_it, converged[i], _, CC[i], _, _ = law.evaluate_with_tangent(
            E_chunk[i], q_prev=q_prev_chunk[i], step_index=step_index_chunk[i],
        )
    return S, CC, q_p_new, converged


class LinearHpromParallelContinuationWrapper:
    """Same per-point warm-start / commit-only-if-converged discipline as
    LinearHpromContinuationWrapper (run_cook_linear_hprom_claude.py), but
    splits the n macro Gauss points across a persistent process pool
    instead of looping serially in this process."""

    def __init__(self, n_workers=16):
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
            (e_voigt[c], [q_prev_list[i] for i in c], [step_index_list[i] for i in c])
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
                # Only commit a CONVERGED q_p as the next warm start -- same
                # rule and same reason as LinearHpromContinuationWrapper's
                # serial version (a rejected macro line-search trial's own
                # possibly-garbage q_p must never poison the warm start for
                # the next call).
                self.q_prev_by_point[int(i)] = q_p_new_flat[local_j]
            else:
                n_nonconverged += 1
        if n_nonconverged:
            print(f"    [linear-hprom-parallel-continuation] {n_nonconverged}/{n} points "
                  "did not converge internally this call")
        return S, CC
