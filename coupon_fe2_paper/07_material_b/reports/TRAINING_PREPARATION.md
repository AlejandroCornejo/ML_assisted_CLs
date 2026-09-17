# Material B: shared features and exact training recipe

Status: preparation complete; **zero neural optimizer steps**. The original
FOM data protocol, labels, material A, shared solver/model sources and manuscript
LaTeX were not changed. Numerical recipe details were completed after data
assembly, before feature selection and before any B neural fitting; do not
describe them all as predeclared before FOM generation.

## Frozen artifacts

- [Exact recipe](../protocol/training_recipe_v1.json), with
  [readable rationale and reproduction](../protocol/TRAINING_RECIPE.md).
- [Selection manifest](../results/feature_selection_v1/manifest.json): allowed
  label accesses, source hashes, scales and fit/reference-only diagnostics.
- `../results/feature_selection_v1/feature_table.npz`: full-precision shared
  32-row table, complete candidate bank, selected indices, NNLS coefficients,
  QR order, feature scales and reference tangent.
- [Initialization checks](../results/feature_selection_v1/initialization_checks.json):
  all five models and three seeds, six fixed/learned pairs, imported source hashes.

| Artifact | SHA-256 |
|---|---|
| Original data protocol | `6e252d9bea52d1e71ee11d5103c582ce20654adbfdb599e4e86a15e7374def8f` |
| Approved labels | `c2edcb3cc663c75f342f66927335c700d30b2867b0a16b9ff668905366dc32e2` |
| Training recipe | `7eb803beeb7a897d3fc5a64454db3627a2ff36a7825e9e4c643ca8933780b397` |
| Shared feature payload | `627a4078a1732e38911c78e93afa4afc762a63cdd7866ce400e606230d5f5405` |

## Selection and limits

Exactly 32 features were selected from 2,028 admissible candidates: 12 from
the fit/reference NNLS support union and 20 from fit-only pivoted-QR diversity.
The prewritten overflow rule was not needed. Endpoint candidates were projected
to a small strict interior margin before selection, identically for all four
constrained models; this avoids a hidden learned-constructor shift.

Only `E_fit`, `S_fit`, `W_fit`, `E_reference`, `S_reference`, `W_reference` and
`D_reference` were requested. Whole-file hashing does not evaluate predictions
or load reserved arrays. Validation/test/path labels were not used for selection,
normalization, calibration or initialization inspection.

The selected affine NNLS scaffold has fit stress error 4.1190%, fit energy error
2.5252%, and reference-tangent error 9.5223%. These are diagnostics of the
nonnegative affine scaffold, **not trained ICNN/ICKAN results or generalization
metrics**. Its coefficients were refitted with the declared 0.02 tangent weight;
the neural initialization additionally applies positive coefficient floors and
nonlinear calibration. The full candidate bank can fit the reference tangent
to numerical precision, but that does not make the restricted, jointly fitted
scaffold's tangent exact.

The fixed baseline is informed by fit labels, not deliberately arbitrary.
Finite candidate selection and greedy QR do not prove an optimal embedding.
The table preserves the paired-feature admissibility conditions; it does not
prove the effective FOM is globally polyconvex or accurately approximable by
this model family.

## Implementation checks

All 15 initializations have finite full-fit losses and diagnostic parameter
gradients. All six fixed/learned pairs have matching initial core states,
full-fit energy/stress responses and reference tangents to numerical tolerance.
Maximum normalized stress difference is 5.56e-16. Maximum reference stress norm
is below 3.62e-7 Pa; reference energy is zero numerically. The equality is
floating-point equality within reported tolerances, not a bitwise identity
claim for inverse-softplus/sigmoid round trips.

| Model | Trainable parameters |
|---|---:|
| Free | 25,473 |
| ICNN-fixed | 2,219 |
| ICNN-learned | 2,379 |
| ICKAN-fixed | 3,839 |
| ICKAN-learned | 3,999 |

Each learned embedding adds 160 parameters (five per feature). Counts include
trainable additive biases even where reference subtraction cancels their energy
offset. Only within-core fixed/learned comparisons are controlled; Free versus
constrained models and ICNN versus ICKAN are not parameter-matched ablations.

The implementation suite passes **81/81 tests** (65 existing, 16 added), with
no skipped neural-setup tests in the existing torch environment. New checks cover
reserved-array access guards, strict initialization bounds, deterministic
selection/overflow, physical energy gradients, reference Hessians, matching
pair starts, exact float64 Free scales and reference-tangent backpropagation
through learned feature parameters. Central differences and agreement with
the historical NumPy formulas are numerical implementation checks, not proofs.

An independent repeat into a new temporary directory reproduced the feature
payload, selection manifest and saved recipe **byte for byte**. Original label,
data-protocol and imported-model source hashes also match. This checks the
deterministic preparation on this host, not a repeat of FOM solves or neural
optimization. The [verification receipt](../results/feature_selection_v1/reproducibility_checks.json)
records the result.

## Subsequent execution status

The B-specific resumable runner has since been implemented and checked with a
bounded training/resume smoke test; see [the execution receipt](TRAINING_RUNNER_SMOKE.md).
That smoke run is deliberately incomplete and produced no final model. A
subsequent recipe-v1 campaign was
[paused for stopping-rule review](TRAINING_CAMPAIGN_PAUSE.md) after eight
complete runs; no recipe change may be hidden as a retry. Failures and any
justified protocol amendment must remain explicit.

Test/path prediction targets remain reserved until every final checkpoint is
locked. Initialization checks establish a fair starting comparison and finite
derivatives, not approximation accuracy, nonnegative energy everywhere,
structural stability or uniqueness.
