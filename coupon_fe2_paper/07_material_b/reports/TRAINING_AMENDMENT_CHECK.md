# Material B stopping amendment: implementation check

Historical check of the second validation-only pilot. Its heavy checkpoints
were later deleted; see the [pilot summary](TRAINING_PILOT_SUMMARY.md). The
current campaign uses the single
[`official recipe`](../protocol/training_recipe.json), not this amendment.

Date: 16 September 2026. The [v2 stopping amendment](../protocol/TRAINING_AMENDMENT_V2.md)
was implemented **before** any v2 official retraining. This is a verification
record, not a neural-accuracy result.

- The frozen v1 recipe SHA-256 is
  `7eb803beeb7a897d3fc5a64454db3627a2ff36a7825e9e4c643ca8933780b397`.
  The v1 runner SHA-256 remains
  `d513ac7783894d38af2c3233b865feec2cb64e798031daded952ab6849a455b1`.
  Both files were left unchanged; the v2 runner is separate.
- The v2 recipe changes only the declared Adam/LBFGS budgets and stopping
  policy. A test compares all other effective recipe fields, including the
  existing Adam `ReduceLROnPlateau` settings, against v1.
- The Material B suite passed: **91 tests**. New tests cover cumulative small
  improvements, minimum work, learning-rate gating, plateau versus safety cap,
  parent-recipe integrity, reconstructed stopping state after resume, and the
  completed report's two stopping reasons.
- A bounded v2 smoke run used `Free`, seed 16, with no final checkpoint: one
  Adam step, interruption, resume to step two. Its model tensors, Adam and
  scheduler states, validation history, identity and torch RNG matched a
  continuous two-step run exactly. The smoke output was isolated in `/tmp`.
- The eight completed v1 pilot `model.pt` SHA-256 hashes still match their
  corresponding reports. No pilot result was overwritten or promoted to v2.
- The runner reads only fit/reference and validation arrays. Reserved test
  and held-out path prediction labels have not been opened for v2 decisions.

The next step is 15 **fresh** v2 runs, not resumption of v1 pilots. A run
reaching the safety cap must be reported as budget-limited. Even a validation
plateau is an operational stopping condition, not proof of convergence.
