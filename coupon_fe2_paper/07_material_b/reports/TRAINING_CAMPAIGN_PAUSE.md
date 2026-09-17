# Material B training campaign — paused for stopping-rule review

Historical pilot record. Its checkpoint directories were subsequently deleted
at the user's request; compact run reports survive in
`../archive/pilot_training_reports.tar.gz`. Current training uses the single
[`official recipe`](../protocol/training_recipe.json).

Date: 16 September 2026. The user questioned whether the declared fixed
optimization budget establishes convergence. It does not. The campaign was
paused before launching further runs; no optimization recipe was changed.

## Recoverable state

| State | Runs |
|---|---|
| Complete under recipe v1 | Free 16/29; ICNN-fixed 16/29; ICNN-learned 16/29; ICKAN-fixed 16; ICKAN-learned 16 |
| Interrupted, resumable | Free 47 at saved Adam step 800; ICKAN-fixed 29 at step 1000; ICKAN-learned 29 at step 200 |
| Not started | ICNN-fixed 47; ICNN-learned 47; ICKAN-fixed 47; ICKAN-learned 47 |

All eight completed `model.pt` hashes match their `run_report.json` records.
The three interrupted runs have `run_state.pt`, no `model.pt` and no failure
record. Their most recent optimizer work beyond the saved step was intentionally
discarded on interruption; the saved state is resumable without changing the
declared seed or model. A complete-run report does **not** imply that the
statistical test set has been evaluated.

The completed reports contain validation stress scores only. Six of the eight
selected their best checkpoint at LBFGS call 40, one at call 39 and one at call
29. This is a warning that the finite budget may be active for some models,
not proof that additional optimization will help test generalization. The
current recipe has no early stopping: Adam runs 2,600 steps, with a
validation-driven learning-rate scheduler, followed by 40 LBFGS outer calls.
The best validation checkpoint is retained, but a run ends at the budget even
if its validation score is still decreasing.

No test or held-out path label was read by the runner. The final evaluation
gate remains closed: all 15 final checkpoints must first be completed and
hash-locked. Any change to stopping rules now requires an explicit versioned
protocol amendment, applied uniformly and documented as occurring after
inspection of intermediate **validation** curves but before inspection of
reserved test/path outcomes. Existing recipe-v1 checkpoints must be preserved
and must not silently be relabeled as having followed a new rule.

## Subsequent amendment, without retraining yet

The prospective [v2 stopping amendment](../protocol/TRAINING_AMENDMENT_V2.md)
now specifies validation-plateau stopping and safety caps for all 15 fresh
official runs. The v1 runner, recipe, eight completed pilots, three partial
states and four unstarted slots remain unchanged. On 16 September 2026 the
v2 campaign was launched with three concurrent workers; its live status is
`../results/neural_training_v2/campaign_status.json`. The reserved evaluation
gate remains closed.
