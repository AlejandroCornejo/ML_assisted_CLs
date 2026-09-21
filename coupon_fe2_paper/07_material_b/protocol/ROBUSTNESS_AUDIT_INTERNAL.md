# Bounded post-test robustness audit (internal only)

This is exploratory, specified after opening the final test results. It does
not change training, checkpoint selection, test metrics, or the manuscript.
All 15 models retain the frozen final-checkpoint manifest.

## Fixed questions and budget

1. Volume collapse: evaluate all models at 81 logarithmically spaced values
   of J from 1 to 1e-8, with F=sqrt(J) I. Calculate the finite Free limit
   directly from its network at limiting features, without evaluating a
   singular deformation. Compare with the constrained construction's growth
   barrier. This is an architecture/growth test, NOT a FOM validation, and
   its advantage must not be attributed to polyconvexity alone.
2. Rank-one curvature: use engineering strains (E11,E22,2E12). Search the
   approved box [-.04,.20]^2 x [-.08,.08] and a fixed double-size box
   [-.08,.40]^2 x [-.16,.16]. The latter is unvalidated extrapolation;
   positive det(F) does not establish absence of microscopic contact.
   Each box uses the same 512 scrambled Sobol points (seed 20260918) plus
   its eight corners for every model, and 32 equally spaced unit b
   directions. Minimize over unit a exactly via the acoustic matrix.
   Refine from the two lowest state minima with bounded Powell optimization
   in three normalized strain coordinates and the b angle, at most 120
   function evaluations per start. No domain or budget extension in response
   to results. Retain failed/nonconverged optimization statuses too.
3. Verify each model/domain minimum using direct differentiation of
   W(F+t a tensor b), and centered energy differences at h=1e-3,3e-4,1e-4.
   The acoustic matrix includes the current-stress geometric contribution.
   Positive search results are not a global certificate. A negative value
   needs magnitude and finite-difference agreement, not merely its sign.
4. If Free has a verified negative minimum, retain at most one witness per
   seed (prefer the approved box). Attempt the corresponding FOM state on
   working/check meshes with existing continuation and physical screens.
   A solver or physical-screen failure is inconclusive, not proof that Free
   introduced an artificial instability. No other FOM campaign is authorized
   by this protocol. If no negative Free witness is found, skip this step.

Save the rule and executable hashes before evaluation. Save all numerical
results, the actual evaluation counts and a reader-facing plot together in
`results/robustness_audit`; add a short internal report under `reports/`.
Do not overwrite an existing audit or alter frozen source files.
