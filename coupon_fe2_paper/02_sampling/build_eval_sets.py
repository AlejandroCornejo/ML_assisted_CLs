#!/usr/bin/env python3
"""Stage 02: the two evaluation sets, declared BEFORE any training runs.

Both are fixed here with a recorded seed, and neither is ever used to select a
model, a hyperparameter or a mesh. That is the point of writing them down now
rather than later.

TEST SET (in-box). Uniform in the box, nudged off the training grid nodes: a
held-out set placed on grid nodes would measure interpolation at points the
model has already seen.

PROBE SET (outside the box), for problem 1b. Where the certificates have to
earn their place. Once training is designed for the problem every tier becomes
accurate INSIDE the envelope, including the uncertified ones, so in-domain
accuracy cannot discriminate; As\'ad\'s failures come from outside the training
subdomain and from stability. Hernandez independently predicts a failure mode
here, noting the weight regression guarantees neither positivity nor volume
preservation "particularly outside the convex hull of the training data".

The probe is structured as RINGS of increasing overshoot along one direction
at a time, so a failure can be ATTRIBUTED rather than merely observed --
reportable as "tier X holds to 1.5x in load but fails at 1.2x in shear":

  * load       -- E11 beyond the box, others inside
  * shear      -- g12 beyond the box, others inside
  * transverse -- E22 beyond the box, others inside

Overshoot is measured in units of the box HALF-SPAN about the box centre, so
"1.5x" means 1.5 half-spans from centre in that component. E11 stays >= 0
throughout: the coupon is loaded in tension, and macro compression would put
the cell near pore buckling where the single-branch assumption fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

N_TEST = 400
RINGS = (1.05, 1.1, 1.25, 1.5, 2.0)
N_PER_PATTERN = 10
# Which components are pushed OUT. All 7 non-empty subsets of {E11, E22, g12},
# so faces, edges AND corners of the enlarged box are covered -- As'ad probes
# a cube, and single-axis rings alone leave the diagonals untested, which are
# the states furthest from the training data and so the likeliest to break a
# surrogate. Keeping the subsets labelled preserves attribution.
PATTERNS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
COMP = ("load", "transverse", "shear")


def load_grid():
    d = np.load(HERE / "train_grid.npz")
    return {k: d[k] for k in d.files}


def build_test(g, rng):
    blo, bhi = g["blo"], g["bhi"]
    steps = g["steps"]
    X = rng.uniform(blo, bhi, size=(N_TEST, 3))
    # Push at least a quarter cell off the nearest node in every coordinate.
    for j in range(3):
        h = float(steps[j])
        frac = (X[:, j] - blo[j]) / h
        off = frac - np.round(frac)
        near = np.abs(off) < 0.25
        X[near, j] += np.sign(off[near] + 1e-12) * 0.25 * h
    return np.clip(X, blo, bhi)


def build_probe(g, rng):
    blo, bhi = np.asarray(g["blo"]), np.asarray(g["bhi"])
    ctr, half = 0.5 * (blo + bhi), 0.5 * (bhi - blo)
    out, labels = [], []
    for f in RINGS:
        for pat in PATTERNS:
            X = rng.uniform(blo, bhi, size=(N_PER_PATTERN, 3))
            for j in pat:
                # E11 is pushed outward only; it is clipped at zero because
                # macro compression would put the cell near pore buckling.
                sgn = np.ones(N_PER_PATTERN) if j == 0 else \
                    rng.choice((-1.0, 1.0), N_PER_PATTERN)
                X[:, j] = ctr[j] + sgn * f * half[j]
            X[:, 0] = np.maximum(X[:, 0], 0.0)
            out.append(X)
            labels += ["+".join(COMP[j] for j in pat) + f"_{f:g}x"] * N_PER_PATTERN
    return np.concatenate(out, axis=0), np.array(labels)


def inside_box(X, g, tol=1e-9):
    return np.all((X >= np.asarray(g["blo"]) - tol)
                  & (X <= np.asarray(g["bhi"]) + tol), axis=1)


def main():
    g = load_grid()
    rng = np.random.default_rng(cfg.SAMPLING_SEED)
    train, test = g["E"], build_test(g, rng)
    probe, labels = build_probe(g, rng)

    print(f"train {train.shape[0]}   test {test.shape[0]}   "
          f"probe {probe.shape[0]} = {len(RINGS)} rings x {len(PATTERNS)} "
          f"patterns x {N_PER_PATTERN}")
    for nm, arr in (("train", train), ("test", test), ("probe", probe)):
        print(f"  {nm:6s} E11 [{arr[:, 0].min():+.5f},{arr[:, 0].max():+.5f}]  "
              f"E22 [{arr[:, 1].min():+.5f},{arr[:, 1].max():+.5f}]  "
              f"g12 [{arr[:, 2].min():+.5f},{arr[:, 2].max():+.5f}]")

    in_test, in_probe = inside_box(test, g), inside_box(probe, g)
    d = np.linalg.norm(test[:, None, :] - train[None, :, :], axis=2).min(axis=1)
    hmin = float(np.min(g["steps"]))

    checks = [
        ("test set entirely inside the box", bool(np.all(in_test)),
         f"{int(np.sum(in_test))}/{test.shape[0]}"),
        ("probe set entirely OUTSIDE the box", bool(not np.any(in_probe)),
         f"{int(np.sum(in_probe))} of {probe.shape[0]} leaked inside"),
        ("test set off the training nodes", bool(np.min(d) > 0.02 * hmin),
         f"min distance {np.min(d):.3e} vs step {hmin:.3e}"),
        ("all 7 out-patterns present (faces, edges, corner)",
         len({s.rsplit(chr(95), 1)[0] for s in labels}) == len(PATTERNS),
         f"{len({s.rsplit(chr(95), 1)[0] for s in labels})} patterns"),
        ("corner states present (all three out)",
         any(s.count("+") == 2 for s in labels),
         str(sorted({s for s in labels if s.count("+") == 2})[:1])),
        ("probe stays in tension", float(probe[:, 0].min()) >= 0.0,
         f"min E11 = {probe[:, 0].min():+.5f}"),
    ]
    print("\nacceptance:")
    for name, good, detail in checks:
        print(f"  [{'OK  ' if good else 'FAIL'}] {name}  ({detail})")
    ok = all(x for _, x, _ in checks)

    np.savez(HERE / "eval_sets.npz", test=test, probe=probe, labels=labels,
             seed=cfg.SAMPLING_SEED)
    print("\nEVAL_SETS_PASS" if ok else "\nEVAL_SETS_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
