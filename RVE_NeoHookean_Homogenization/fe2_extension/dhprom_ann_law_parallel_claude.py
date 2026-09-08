#!/usr/bin/env python3
"""Parallelized D-HPROM-ANN vectorized material function.

Unlike HpromAnnIterativeLawFloat64/LinearHpromIterativeLawFloat64,
DHpromAnnDirectLawFloat64 has no inner Newton correction loop and no
continuation/warm-starting state at all -- q_p comes directly from a
closed-form affine map of E (see dhprom_ann_direct_law_float64_claude.py's
own evaluate_with_tangent). Its module-level dhprom_ann_pk2_2d_vectorized_
consistent_float64 is therefore already a pure function of E_flat (no
state threaded between calls), so this parallel wrapper is simpler than
hprom_ann_law_parallel_claude.py's/linear_hprom_law_parallel_claude.py's
own continuation-aware ones: no per-point q_prev bookkeeping needed at
all, just split E_flat into chunks and dispatch.

Each worker calls the EXISTING, already-verified dhprom_ann_pk2_2d_
vectorized_consistent_float64 on its own chunk unmodified -- that function
already batches the decoder Jacobian (vmap(jacfwd), verified this session
at a 40-point scale to hold its ~60x per-row speedup; the ~37-point chunk
size each of 16 workers gets here is the same regime) and the reaction-
force tangent across whatever batch it's given, so this composes with,
rather than replaces, that batching -- same two-level design as HPROM-ANN's
own parallel wrapper. Unlike HPROM-ANN, there is no inner-iteration
convergence check here at all (q_p is a direct, deterministic affine map,
not an iterative solve), so there is no borderline-convergence-threshold
sensitivity to worry about across different chunk sizes -- only ordinary
floating-point reordering, already verified negligible.

Same fork-after-Kratos/torch-threading discipline as every other
*_law_parallel_claude.py this session: ensure_persistent_executor() must
be called before ANY Kratos- or torch-touching code runs in the parent
process (this module's own top level imports neither).
"""
from __future__ import annotations

import multiprocessing
import os

# Defense in depth: a fork-context ProcessPoolExecutor's workers are
# spawned LAZILY, at first actual task dispatch, not eagerly at pool
# construction (confirmed the hard way this session -- a driver script
# that ran PANN, which unlike every law in this file does not force
# CPU-only torch, before this pool's own first dispatch crashed with
# "CUDA error: initialization error": the fork happened only once the
# parent had already initialized a real CUDA context, which a forked
# child cannot validly inherit). Disabling CUDA visibility for this
# whole process at import time keeps it (and everything it forks) CPU-
# only regardless of what any OTHER code in the same process does or
# doesn't force internally, and regardless of when the actual fork
# happens to occur.
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
    Newton call through dhprom_ann_pk2_2d_vectorized_consistent_float64_
    parallel / make_dhprom_ann_parallel_material_func. Idempotent."""
    global _PERSISTENT_EXECUTOR
    if _PERSISTENT_EXECUTOR is None:
        _PERSISTENT_EXECUTOR = ProcessPoolExecutor(max_workers=n_workers, mp_context=_FORK_CTX)
    return _PERSISTENT_EXECUTOR


_WORKER_SEEDED = False


def _worker(args):
    global _WORKER_SEEDED
    # One worker per core provides the parallelism; letting torch/Kratos/
    # BLAS also spawn their own internal threads per worker would
    # oversubscribe the machine.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    E_chunk, dhpromann_dir = args
    # dhprom_ann_direct_law_float64_claude.py itself adds KRATOS_PATH to
    # sys.path before importing KratosMultiphysics (unlike hprom_ann_
    # iterative_law_float64_claude.py, which relies on its caller having
    # done so), so no explicit sys.path fix is needed here -- but adding
    # it defensively costs nothing and protects against that ordering
    # ever changing.
    kratos_path = "/home/kratos/Kratos_Eigen_Check/bin/Release"
    if kratos_path not in sys.path:
        sys.path.append(kratos_path)
    import dhprom_ann_direct_law_float64_claude as law_module
    if not _WORKER_SEEDED:
        # Seeds this worker's own get_law_float64() singleton with the
        # reaction-force-convention weights dir, mirroring exactly what
        # every serial driver script (e.g. run_cruciform_overnight_
        # claude.py) does in the main process before its own first call --
        # the module-level vectorized function takes no hprom_ann_dir
        # argument itself, only the singleton factory does.
        law_module.get_law_float64(hprom_ann_dir=str(dhpromann_dir))
        _WORKER_SEEDED = True
    S, CC = law_module.dhprom_ann_pk2_2d_vectorized_consistent_float64(E_chunk)
    return S, CC


def dhprom_ann_pk2_2d_vectorized_consistent_float64_parallel(
    E_flat, dhpromann_dir, young=None, poisson=None, n_workers=16,
):
    E_flat = np.asarray(E_flat, dtype=float).reshape(-1, 3)
    n = E_flat.shape[0]
    n_workers = max(1, min(n_workers, n))
    idx_chunks = [c for c in np.array_split(np.arange(n), n_workers) if c.size > 0]
    chunks = [(E_flat[c], dhpromann_dir) for c in idx_chunks]

    executor = ensure_persistent_executor(n_workers)
    results = list(executor.map(_worker, chunks))

    S = np.concatenate([r[0] for r in results], axis=0)
    CC = np.concatenate([r[1] for r in results], axis=0)
    return S, CC


def make_dhprom_ann_parallel_material_func(dhpromann_dir, n_workers=16):
    """Returns a MATERIAL_FUNCS-compatible callable. No continuation
    wrapper needed (unlike HPROM-ANN/Linear-HPROM) since D-HPROM-ANN
    carries no per-point state across calls at all."""
    def _f(e_voigt, young=None, poisson=None):
        return dhprom_ann_pk2_2d_vectorized_consistent_float64_parallel(
            e_voigt, dhpromann_dir=dhpromann_dir, n_workers=n_workers,
        )
    return _f
