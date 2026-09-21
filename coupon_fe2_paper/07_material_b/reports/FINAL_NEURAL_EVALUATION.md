# Internal Material-B neural evaluation — 18 September 2026

**For our working record, not manuscript text.** The reserved 512-state test
and ten previously declared 40-state paths were opened once, after the exact
15-checkpoint manifest was verified. The eligibility exception for the three
Adam-budget-limited Free runs, the definitions of all metrics, and the
no-retraining rule were recorded beforehand in
[`FINAL_EVALUATION_INTERNAL.md`](../protocol/FINAL_EVALUATION_INTERNAL.md).
No test or path error selected a checkpoint or a representative seed.

## Primary test results

Aggregate relative norm errors, percent; mean ± sample standard deviation
over seeds 16, 29 and 47. Energy, work-conjugate second-Piola stress and
engineering-strain tangent are distinct metrics.

| Model | Energy | Stress | Tangent |
|---|---:|---:|---:|
| Free | 0.0015 ± 0.0006 | 0.0078 ± 0.0006 | 0.0900 ± 0.0079 |
| ICNN-fixed | 0.0355 ± 0.0008 | 0.1421 ± 0.0016 | 0.8896 ± 0.0075 |
| ICNN-learned | 0.0062 ± 0.0002 | 0.0350 ± 0.0008 | 0.2859 ± 0.0037 |
| ICKAN-fixed | 0.0319 ± 0.0012 | 0.1346 ± 0.0031 | 1.3045 ± 0.1560 |
| ICKAN-learned | 0.0060 ± 0.0002 | 0.0372 ± 0.0015 | 0.6667 ± 0.1729 |

Within each constrained core, learned features improve all three **test**
aggregates relative to fixed features. This is the controlled feature
comparison. Free obtains smaller errors, but differs in architecture and
optimization history; it is not a parameter-matched ablation. Its Adam stage
reached 200,000 steps without a plateau, while L-BFGS subsequently plateaued
without improving its selected Adam checkpoints. Do not call Free converged.

## Held-out paths: important qualification

The aggregate path-tangent errors are 0.9549 ± 0.1038% for ICNN-fixed versus
1.0534 ± 0.1770% for ICNN-learned, and 1.8471 ± 0.6298% for ICKAN-fixed
versus 2.8567 ± 0.9521% for ICKAN-learned. On nine of the ten individual
paths, however, each learned variant has a lower mean tangent error than its
fixed counterpart. The reversal is concentrated in **biaxial compression**:
mean tangent errors there are 1.486% fixed versus 2.482% learned for ICNN,
and 3.463% fixed versus 6.777% learned for ICKAN. This is an observed
path-specific limitation, not an excuse to remove that path or retune models.
Stress and energy aggregates on all paths remain lower for the learned
variants than their fixed counterparts. Full per-seed and per-path results,
including median/95th-percentile/maximum statewise errors, are retained in
[`per_run.json`](../results/neural_evaluation/per_run.json).

## Numerical quality checks, not proofs

All 15 saved models reproduce their recorded validation scores before the
reserved-label gate opens. The final test-tangent matrices have maximum
asymmetry below `8.35e-7 Pa`. An independent central-difference check on
16 deterministic test-input indices and all ten path endpoints differs from
the automatic-differentiation tangent by at most `4.31e-10`, relative to the
largest tangent entry of each model. Reference energy is exactly zero at the
reported precision and the largest reference-stress norm is below
`2.90e-7 Pa`; the reference tangent is fitted, not imposed exactly, with
the worst relative Frobenius error about `0.495%`.

A deterministic rank-one screen used 26 input states times 16 unit-direction
pairs per model. No negative curvature was observed; the smallest sampled
value was `2.34e8 Pa`. This is a **finite numerical screen**, not a proof of
polyconvexity or RVE stability, and a positive screen for Free gives no global
guarantee. The independent constitutive construction argument for the
constrained models remains the relevant analytical claim.

## Audit trail and next decision

- Exact model hashes: [`final_checkpoint_manifest.json`](../results/neural_training/final_checkpoint_manifest.json).
- One-time gate, evaluator/code and source hashes: [`gate_decision.json`](../results/neural_evaluation/gate_decision.json).
- All seed summaries: [`summary.json`](../results/neural_evaluation/summary.json).
- Stored test/path/reference predictions: [`predictions.npz`](../results/neural_evaluation/predictions.npz).
- Derivative and sampled-curvature checks: [`mechanical_audit.json`](../results/neural_evaluation/mechanical_audit.json).

No further training is justified by these reserved results. Next, inspect
plots chosen by the **predeclared median-validation seeds** (Free 29,
ICNN-fixed 29, ICNN-learned 29, ICKAN-fixed 29, ICKAN-learned 47), with the
biaxial-compression caveat visible. Any paper-facing figure or claim should
be based on this locked record, not on a fresh model selection.
