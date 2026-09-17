# Material B training runner: bounded smoke check

Date: 16 September 2026. The [B-specific runner](../protocol/train_material_b.py)
implements the already frozen [recipe](../protocol/training_recipe_v1.json).
This report is an **implementation check**, not a training or accuracy result.

## What was checked

- Each of Free, ICNN-fixed, ICNN-learned, ICKAN-fixed and ICKAN-learned completed
  one real, full-4,200-state Adam step (seed 16) and stopped in the declared
  incomplete state. None wrote a final `model.pt`.
- ICNN-fixed, seed 16 was run to step 10 two ways: continuously, and as steps
  1–5 followed by a new process resuming to step 10. All 12 compared fields
  matched **exactly**: identity, phase/counters, current and best weights,
  best score/origin, validation history, Adam state, scheduler state and Python,
  NumPy and torch RNG states. The initial and step-10 validation scores were
  `0.0046441060061106` and `0.0033953402362515493`, respectively. These are
  early validation scores, not final model performance.
- The initial score and the step-10 score both advanced the scheduler, as the
  frozen validation-call schedule requires.
- Three additional synthetic unit tests check exact allowed-array access,
  Adam state restoration, and LBFGS history restoration. The full local suite
  passes **84/84**. The LBFGS check is synthetic; no full 2,600-step/40-call
  scientific run has yet been completed.

All smoke checkpoints are outside the project, under
`/tmp/material_b_runner_verified_yj1LIp/`; they are disposable and are not
paper evidence. No statistical test/path label, test prediction, or model
generalization score was read. The allowed label keys are fit strain/stress/
energy, reference strain/stress/energy/tangent, and validation strain/stress.

Runner source SHA-256:
`d513ac7783894d38af2c3233b865feec2cb64e798031daded952ab6849a455b1`.
The frozen recipe, approved label file and feature table retained their
recorded SHA-256 values. The smoke pause flags do not change those inputs or
produce a final checkpoint.

## Next gate

Run all 15 declared model/seed combinations to completion. Keep failures
visible and do not change the frozen recipe silently. Only after all 15 final
checkpoints are hash-locked may final test and held-out path predictions be
opened. This smoke check does not establish learned accuracy, polyconvexity of
the FOM, or structural equilibrium stability.
