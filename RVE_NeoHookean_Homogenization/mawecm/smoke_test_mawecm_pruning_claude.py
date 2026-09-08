#!/usr/bin/env python3
"""Stage 0 verification gate (see plan): confirm the ported run_mawecm_pruning
returns a sane, non-negative, constraint-satisfying weight vector on a tiny,
hand-computable synthetic problem. Not a physics check -- that comes once
real reaction-force data feeds it (Stage 1-3).

Toy problem: 6 candidate "points" with known per-candidate values a_i, at 4
training "states" (q_train rows), where each state's target is simply
sum_i(a_i * w_i) = sum(a_i) (i.e. w_ini = all-ones reproduces the exact sum
trivially, and pruning should be able to find a SPARSER w that still
reproduces sum(a_i), since with only 1 scalar constraint per state, keeping a
single well-chosen candidate can satisfy it exactly).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mawecm_pruning_claude import run_mawecm_pruning  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    n_cand = 6
    n_states = 4

    a = rng.uniform(1.0, 2.0, size=n_cand)  # per-candidate value, same at every state (toy)
    target = float(np.sum(a))  # the constraint every state must match exactly

    z_ini = np.arange(n_cand, dtype=np.int64)
    w_ini = np.ones(n_cand, dtype=float)  # w_ini @ a == sum(a) == target, exact by construction

    A_blocks = [a.reshape(1, n_cand) for _ in range(n_states)]
    b_blocks = [np.array([target]) for _ in range(n_states)]
    q_train = rng.normal(size=(n_states, 2))  # arbitrary coordinates, unused by phase-1 pruning

    print(f"[smoke-test] n_cand={n_cand}, target={target:.6f}, w_ini @ a = {float(w_ini @ a):.6f}")

    result = run_mawecm_pruning(
        A_blocks=A_blocks, b_blocks=b_blocks, z_ini=z_ini, w_ini=w_ini, q_train=q_train,
        options={"verbose": True, "n_stop": 1},
    )

    W_support = result["W_support"]
    Z_support = result["Z_support"]
    print(f"[smoke-test] final support size={Z_support.size} (candidates kept: {Z_support})")
    print(f"[smoke-test] final weights:\n{W_support}")

    assert np.all(W_support >= -1.0e-9), "found a negative weight -- nonnegativity violated"

    a_support = a[Z_support]
    reproduced = a_support @ W_support  # (n_states,)
    err = np.max(np.abs(reproduced - target))
    print(f"[smoke-test] max constraint error across states = {err:.3e}")
    assert err < 1.0e-8, "pruned support does not reproduce the target constraint"

    print("[smoke-test] PASSED: ported run_mawecm_pruning returns a sane, "
          "non-negative, constraint-satisfying sparse weight vector.")


if __name__ == "__main__":
    main()
