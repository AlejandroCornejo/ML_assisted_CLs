#!/usr/bin/env python3
"""Same two-level-parallel HPROM-ANN wrapper as hprom_ann_law_parallel_
claude.py, EXCEPT the underlying HpromAnnIterativeLawFloat64 is constructed
with qp_init_mode="mu_affine" instead of "continuation" -- every call's
inner-Newton-correction initial guess comes from the precomputed affine map
mu -> q_p (offline-fit alongside the rest of the HPROM-ANN model, the same
"qm_init_mu_affine.npz" data continuation mode itself falls back on for a
point's very first-ever call), from the CURRENT macro strain alone, with
NO cross-call/cross-step memory at all -- unlike "continuation" mode, which
warm-starts every call after the first from whatever q_p the SAME point
converged to last time.

Purpose: isolate whether HPROM-ANN's crash (same "Invalid Green-Lagrange
strain state" seen under n_body=12, force control, and the square-panel
trajectory, always with the continuation wrapper) comes from bad warm-
starting carrying the inner correction onto a wrong branch, or persists
even with every call starting fresh from the same affine guess a truly
memoryless evaluation would use -- i.e., whether the fragility lives in the
warm-start bookkeeping or in the correction network's own response at the
difficult state, independent of any call history.

A separate module (not a modified copy in place) so this has its own
independent persistent executor pool / worker-side law instance, with zero
risk of stale state bleeding in from the already-validated continuation
wrapper's own pool."""
from __future__ import annotations

import multiprocessing
import os

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
    related. Idempotent. Independent of hprom_ann_law_parallel_claude.py's
    own pool/global state."""
    global _PERSISTENT_EXECUTOR
    if _PERSISTENT_EXECUTOR is None:
        _PERSISTENT_EXECUTOR = ProcessPoolExecutor(max_workers=n_workers, mp_context=_FORK_CTX)
    return _PERSISTENT_EXECUTOR


_WORKER_LAW = None


def _worker(args):
    global _WORKER_LAW
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    E_chunk, hprom_ann_dir = args
    if _WORKER_LAW is None:
        kratos_path = "/home/kratos/Kratos_Eigen_Check/bin/Release"
        if kratos_path not in sys.path:
            sys.path.append(kratos_path)
        import hprom_ann_iterative_law_float64_claude as law_module
        _WORKER_LAW = law_module.HpromAnnIterativeLawFloat64(
            hprom_ann_dir=str(hprom_ann_dir), qp_init_mode="mu_affine",
        )
    law = _WORKER_LAW

    n = E_chunk.shape[0]
    # q_prev/step_index are ignored by _initial_qp_guess whenever
    # qp_init_mode=="mu_affine" (confirmed by reading the actual branch
    # logic), but the batch call still requires arrays of the right shape.
    q_prev_batch = np.zeros((n, law.n_primary), dtype=float)
    step_index_batch = np.ones(n, dtype=int)

    _, S, q_p_new, _n_it, converged, _, CC, _, _ = law.evaluate_with_tangent_batch(
        E_chunk, q_prev_batch=q_prev_batch, step_index_batch=step_index_batch,
    )
    return S, CC, list(converged)


class HpromAnnParallelAffineWrapper:
    """No cross-call state at all (unlike HpromAnnParallelContinuationWrapper)
    -- every call is independent, splitting the n macro Gauss points across
    a persistent process pool, each worker batching its own chunk via
    evaluate_with_tangent_batch, always seeded from the mu-affine initial
    guess."""

    def __init__(self, hprom_ann_dir, n_workers=16):
        self.hprom_ann_dir = str(hprom_ann_dir)
        self.n_workers = int(n_workers)

    def __call__(self, e_voigt, young=None, poisson=None):
        e_voigt = np.asarray(e_voigt, dtype=float)
        n = e_voigt.shape[0]

        n_workers = max(1, min(self.n_workers, n))
        idx_chunks = [c for c in np.array_split(np.arange(n), n_workers) if c.size > 0]
        chunks = [(e_voigt[c], self.hprom_ann_dir) for c in idx_chunks]

        executor = ensure_persistent_executor(self.n_workers)
        results = list(executor.map(_worker, chunks))

        S = np.concatenate([r[0] for r in results], axis=0)
        CC = np.concatenate([r[1] for r in results], axis=0)
        converged_flat = []
        for r in results:
            converged_flat.extend(r[2])
        n_nonconverged = sum(1 for c in converged_flat if not c)
        if n_nonconverged:
            print(f"    [hprom-ann-parallel-affine] {n_nonconverged}/{n} points "
                  "did not converge internally this call")
        return S, CC
