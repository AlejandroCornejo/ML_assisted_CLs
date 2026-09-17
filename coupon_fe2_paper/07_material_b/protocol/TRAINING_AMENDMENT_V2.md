# Material B: prospective training-stopping amendment (v2)

The [machine-readable amendment](training_recipe_v2.json) applies to **all 15
official model–seed runs**, started from their original seed and common feature
table. The eight completed v1 runs are pilots, not v2 results. This amendment
was written after inspecting their **validation** curves, which often still
improved near the fixed LBFGS limit, and before opening any reserved test or
held-out path prediction labels. It does not claim to have been prespecified
before those pilot curves.

Only the optimization **stopping budgets and rules** change. The v1 split,
labels, selected 32 features, architectures, objective, initialization, seeds,
optimizer parameters, stress-validation metric, and best-checkpoint rule stay
fixed. In particular, Adam already used `ReduceLROnPlateau`: every ten Adam
steps, validation stress is measured; after 40 validation checks without a
`1e-4` relative improvement, the scheduler halves the learning rate, down to
`1e-5`. A reduced learning rate does not itself end training.

| Phase | Earliest stop | Meseta criterion | Safety cap |
|---|---:|---|---:|
| Adam | 2,600 steps | No **cumulative** 0.1% improvement in best validation stress over 80 validation checks (800 steps), **and** learning rate has reached `1e-5` | 7,800 steps |
| LBFGS | 40 outer calls | No **cumulative** 0.1% improvement in best validation stress over 20 outer calls | 120 outer calls |

The minima preserve the original v1 work, while the caps allow up to three
times its budget without making runtime unbounded. The 0.1% threshold and
patiences are **operational choices**, fixed now for all models and seeds; they
were not calibrated against test accuracy and are not a physical tolerance.

For example, several small improvements can add up to 0.1%; then the patience
clock restarts. A poorer current score cannot erase an earlier best score. The
Adam clock starts at the initial validation score; the LBFGS clock starts from
the best Adam score. The same rule is reconstructed from saved validation
history after an interruption, so a resumed run has the same stopping decision.
The `strong_wolfe` search still chooses steps **inside** each LBFGS call; there
is no separate LBFGS scheduler. Each call still has up to ten internal
iterations, and actual iterations/evaluations must be reported.

If the cap arrives first, the report says `safety_cap`, not “converged.” A
`validation_plateau` is also just a finite-data, finite-patience stopping rule,
not a proof of optimizer convergence or physical fidelity. Every final model
still restores the strictly lowest validation stress checkpoint across
initialization, Adam and LBFGS. All 15 reports and checkpoints must be
hash-locked before the reserved prediction labels are evaluated.

The v1 runner and recipe are retained unchanged. Use
`python -m protocol.train_material_b_v2` and new paths under
`results/neural_training_v2/`; the v2 runner refuses the v1 pilot directory.
Its checkpoint identity includes both recipe hashes and runner source hashes.
