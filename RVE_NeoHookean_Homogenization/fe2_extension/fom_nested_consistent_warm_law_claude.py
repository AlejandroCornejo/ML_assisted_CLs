#!/usr/bin/env python3
"""Warm-started FOM-FE2 material law: same converged answer as
fom_nested_consistent_law_claude.py, an order of magnitude cheaper.

Why this is not an approximation. This RVE is Neo-Hookean hyperelastic with
no history variables, so its converged solution is a pure function of the
macro strain E. Ramping from E=0 in ~200*|E| substeps is purely a Newton-
convergence aid, not physics; continuing instead from the same Gauss point's
own previously converged state needs 1 substep and lands on the SAME state.
Verified directly (_verify_fom_warm_vs_cold_claude.py) at a representative
final-load dogbone strain: S and CC agree with the cold solve to 1.1e-15 and
6.9e-15 relative -- double-precision roundoff, not merely solver tolerance --
while that one solve went from 5.1s to 0.4s (12.3x), a ratio that GROWS with
|E| since cold cost scales with it and warm cost does not.

Caveat, stated rather than assumed: path-independence fails where the RVE
admits several stable branches at one E (pore buckling under compression),
since a warm start could follow a different branch than a from-zero ramp.
Tension-dominated use (this project's dogbone) is safe; near pore closure
prefer cold starts.

State is held per Gauss point, keyed by the caller's own global Gauss-point
index, so each point continues from its own history rather than a neighbour's.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fom_nested_consistent_law_claude import evaluate_with_tangent  # noqa: E402

# {global_gauss_point_index: (E_prev (3,), u_prev (n_dof,))}
_STATE = {}


def reset_state():
    """Drop all warm-start state (e.g. between independent macro problems
    in one process). Callers that never call this simply keep continuing."""
    _STATE.clear()


def state_size():
    return len(_STATE)


@contextlib.contextmanager
def _silenced(active=True):
    """Suppress the solver's own per-substep chatter at the FILE DESCRIPTOR
    level, not just sys.stdout: Kratos prints from C++, which bypasses
    Python-level redirection entirely. Without this, one 20-step macro run
    over 1656 Gauss points wrote a 4.2 GB log (measured), which is pure
    overhead -- the per-substep trace is of no diagnostic value once the
    solve is known to converge."""
    if not active:
        yield
        return
    fd = sys.stdout.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(devnull, fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def evaluate_warm(E, gp_index, out_dir=None, silence=True):
    """(S (3,), CC (3,3)) at macro strain E for Gauss point `gp_index`,
    continuing from that point's own last converged state when available."""
    E = np.asarray(E, dtype=float).reshape(3)
    prev = _STATE.get(int(gp_index))
    E_start, u_init = (None, None) if prev is None else prev
    with _silenced(silence):
        S, CC, u = evaluate_with_tangent(
            E, out_dir=out_dir, u_init=u_init, E_start=E_start, return_u=True)
    _STATE[int(gp_index)] = (E.copy(), u)
    return S, CC


def fom_nested_consistent_warm_pk2_2d_vectorized(E_flat, gp_indices=None, out_dir=None,
                                                   verbose=False, silence=True,
                                                   state_in=None, return_state=False):
    """Same (E_flat (N,3)) -> (S (N,3), CC (N,3,3)) contract as
    fom_nested_consistent_pk2_2d_vectorized, warm-started per Gauss point.

    gp_indices: global index per row, so state follows the right point.

    state_in/return_state make this drivable STATELESSLY, which is what the
    parallel wrapper needs: ProcessPoolExecutor.map gives no task-to-worker
    affinity (a shared queue hands each chunk to whichever worker is idle),
    so module-level state in a worker is not reliably the state belonging to
    the points that worker just received. Passing the relevant slice in and
    the updated slice back out makes correctness independent of scheduling.
    With return_state=True the return is (S, CC, state_out, n_warm)."""
    E_flat = np.asarray(E_flat, dtype=float).reshape(-1, 3)
    n = E_flat.shape[0]
    if gp_indices is None:
        gp_indices = np.arange(n, dtype=np.int64)
    gp_indices = np.asarray(gp_indices, dtype=np.int64).reshape(-1)
    if gp_indices.size != n:
        raise ValueError(f"gp_indices size {gp_indices.size} != number of strains {n}")

    S = np.zeros((n, 3), dtype=float)
    CC = np.zeros((n, 3, 3), dtype=float)
    state_out = {}
    n_warm = 0
    t0 = time.perf_counter()
    for i in range(n):
        gp = int(gp_indices[i])
        E_i = E_flat[i]
        prev = (state_in or {}).get(gp) if state_in is not None else _STATE.get(gp)
        if prev is None:
            E_start, u_init = None, None
        else:
            E_start, u_init = prev
            n_warm += 1
        with _silenced(silence):
            S[i], CC[i], u = evaluate_with_tangent(
                E_i, out_dir=out_dir, u_init=u_init, E_start=E_start, return_u=True)
        if state_in is None:
            _STATE[gp] = (E_i.copy(), u)
        else:
            state_out[gp] = (E_i.copy(), u)
    if verbose:
        dt = time.perf_counter() - t0
        print(f"    [fom_nested_consistent_warm] {n} Gauss points, {dt:.1f}s "
              f"({dt / max(n, 1):.3f}s/GP), {n_warm}/{n} warm-started", flush=True)
    if return_state:
        return S, CC, state_out, n_warm
    return S, CC
