#!/usr/bin/env python3
"""Fit the adaptive weight FIELDS w(mu) from the pruned per-state weights.

WHY THIS STEP IS WHERE THE METHOD IS ACTUALLY JUDGED. The pruning returns
weights at the 495 training states only, and it reproduces the targets there
to ~1e-14. That number is not accuracy: with 10 weights against 4-5
constraints per state the system is UNDERDETERMINED, so it is satisfied
exactly by construction. Deployment needs w(mu) at arbitrary mu, and the
regression is what has to generalize.

THE SOFTMAX PARAMETRIZATION IS THE POINT.

    w(mu) = target_sum * softmax(logits(mu))

gives non-negativity and an exact weight sum BY CONSTRUCTION, for any mu,
however wrong the logits are. Those are exactly the two properties that made
the classic fixed-weight rule degrade only 1.2x out of the training envelope,
against 16x for the unconstrained network closure. Hernandez notes generic
weight regressions guarantee neither, "particularly outside the convex hull of
the training data"; this implementation prevents that failure structurally
rather than hoping the fit is good.

For the sum to be representable at all it must be CONSTANT across states, which
is why both rules carry a volume row -- verified at 1546.0000 to 1e-15 on both.

Three things are measured, in increasing relevance:

  1. weight-field regression error on held-out states -- how well the fit
     reproduces the pruned weights;
  2. CONSTRAINT reproduction using the REGRESSED weights, which is what the
     hyperreduction actually needs and which the 1e-14 above does not measure;
  3. (next script) the deployed stress error in and out of the envelope.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(HERE), str(PROJ / "mawecm")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mawecm_ann_weights import eval_mawecm_ann, fit_mawecm_ann  # noqa: E402

TARGET_SUM = 1546.0
VAL_FRAC = 0.15


def main():
    ds = np.load(HERE / "maw_dataset.npz")
    rl = np.load(HERE / "maw_rules.npz")
    sub = rl["sub"]
    q_all = ds["q_train"][sub]

    out = {}
    for nm, nrows in (("res", 4), ("sig", 5)):
        Z = np.asarray(rl[f"Z_{nm}"], dtype=np.int64)
        W = np.asarray(rl[f"W_{nm}"], dtype=float)          # (10, n_states)
        A_full = ds[f"A_{nm}"][sub]
        b_full = ds[f"b_{nm}"][sub]
        z_cand = ds[f"z_{nm}"]
        loc = np.searchsorted(z_cand, Z)
        assert np.array_equal(z_cand[loc], Z), "support index mapping mismatch"
        A = A_full[:, :, loc]                                # (n_states, m, 10)

        # Held-out split of STATES, so the regression is judged on mu it has
        # not seen. The pruning itself used all of them, so this measures the
        # regression only.
        rng = np.random.default_rng(5)
        perm = rng.permutation(q_all.shape[0])
        n_val = int(VAL_FRAC * q_all.shape[0])
        vi, ti = perm[:n_val], perm[n_val:]

        print(f"\n=== {nm}: {Z.size} points, {q_all.shape[0]} states "
              f"({len(ti)} fit / {len(vi)} held out) ===", flush=True)
        t0 = time.perf_counter()
        # Configuration copied from the previous project's own validated
        # driver (fit_final_10point_rule_claude.py) rather than chosen here.
        # The decisive setting is physics_weight = 1.0 with the constraint
        # blocks passed: it trains the field against CONSTRAINT VIOLATION, not
        # against the pruned weight VALUES. Training on the values (the
        # library default, physics_weight = 0.0) is optimizing an intermediate
        # instead of the objective, and measured the difference plainly -- a
        # 5-10% weight-field error turned into an 84% constraint error on the
        # residual rule, because with only 10 points the constraints are
        # near-cancellations and badly conditioned in the weights.
        A_blocks = [A[k] for k in ti]
        b_blocks = [b_full[k] for k in ti]
        model = fit_mawecm_ann(
            q_train=q_all[ti], W_train=W[:, ti], target_sum=TARGET_SUM,
            constraint_A_blocks=A_blocks, constraint_b_blocks=b_blocks,
            physics_q_train=q_all[ti],
            physics_constraint_A_blocks=A_blocks,
            physics_constraint_b_blocks=b_blocks,
            hidden_dims=(256, 256, 256), activation="gelu",
            epochs=40000, patience=3000, physics_weight=1.0,
            verbose=False, label=f"MAW-{nm}")
        dt = time.perf_counter() - t0

        W_hat_all = eval_mawecm_ann(q_all, model)            # (10, n_states)
        e_w = np.linalg.norm(W_hat_all[:, vi] - W[:, vi]) / np.linalg.norm(W[:, vi])
        e_w_fit = np.linalg.norm(W_hat_all[:, ti] - W[:, ti]) / np.linalg.norm(W[:, ti])

        def crel(Wx, ids):
            r = [np.linalg.norm(A[k] @ Wx[:, k] - b_full[k])
                 / max(np.linalg.norm(b_full[k]), 1e-300) for k in ids]
            return np.array(r)

        c_pruned = crel(W, vi)
        c_fitted = crel(W_hat_all, vi)
        s = W_hat_all.sum(axis=0)

        print(f"  fitted in {dt:.1f}s")
        print(f"  weight field   rel err   fit {e_w_fit:.4e}   held out {e_w:.4e}")
        print(f"  constraints on held-out states:")
        print(f"     pruned weights  median {np.median(c_pruned):.4e}")
        print(f"     FITTED weights  median {np.median(c_fitted):.4e}   "
              f"p90 {np.percentile(c_fitted, 90):.4e}")
        print(f"  structural: weights >= 0 {bool(np.all(W_hat_all >= 0))},  "
              f"sum {s.min():.4f}..{s.max():.4f} (target {TARGET_SUM})")
        out[nm] = dict(model=model, Z=Z)
        np.savez_compressed(HERE / f"maw_field_{nm}.npz", Z=Z, **model)

    print("\nMAW_FIELD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
