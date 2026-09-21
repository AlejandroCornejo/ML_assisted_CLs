# Internal note: Material-B final neural evaluation

This is an internal execution note, **not manuscript text**. It is fixed before
reading reserved test or held-out-path prediction targets.

## Gate amendment

The original training rule admitted only 15 checkpoints whose Adam and L-BFGS
phases both stopped by validation plateau. The locked campaign instead has 12
such checkpoints and three Free checkpoints whose Adam phase was stopped at
200,000 steps while still improving. Their L-BFGS phase did satisfy its
original plateau rule, but did not improve the best Adam validation score.
`FREE_LBFGS_CLOSEOUT.md`, the review audit, each final report and
`../results/neural_training/final_checkpoint_manifest.json` preserve this
distinction. This note amends **evaluation eligibility only**: accept this
exact mixed set after checking all hashes. It does not relabel Free as
Adam-converged, authorize additional training, or change any checkpoint.
The exact 15-checkpoint manifest accepted here has SHA-256
`d42387b9c6b84cf810b0fc0c5872380da90ffa2af3cca1bb4d891ecd58ec2282`.

The evaluator must verify the 15-entry manifest, all model and report hashes,
the approved FOM-label hash, frozen data protocol, feature-preparation hash,
and the source hashes carried by the saved model identities **before** opening
reserved target arrays. Its dry run uses validation labels only. An output
directory may be created once; it must never overwrite a previous final
evaluation. Any implementation correction after opening reserved targets must
be documented separately, with the original result retained. Test or path
errors may not select checkpoints, models, seeds, hyperparameters or plots.

## Fixed metrics and presentation

For a target array $Y$ and prediction $\widehat Y$, the aggregate percentage
error is $100\lVert\widehat Y-Y\rVert/\lVert Y\rVert$, with Euclidean or
Frobenius norm over all states and components. Apply this separately to energy
$W$, work-conjugate second-Piola stress $S=(S_{11},S_{22},S_{12})$, and the
$3\times3$ engineering-strain tangent $D=\partial S/\partial
(E_{11},E_{22},2E_{12})$. Reject a zero aggregate denominator rather than
inventing a test-dependent scale.

For each state, divide its absolute energy error or stress/tangent norm error
by the larger of the reference magnitude and the corresponding **fit-derived**
floor in the feature-preparation manifest. Report the median, 95th percentile
and maximum of these percentages. Stress-component RMSE is divided by its
fit-derived component scale. This follows the frozen data protocol; it does
not alter the training or checkpoint-selection objective.

The independent test has 512 states. Ten predeclared paths each have 40
nonreference states; the shared zero-strain reference is scored separately,
never multiplied into path errors. Report every seed, then mean and sample
standard deviation (ddof=1). A curve may represent only the median-**validation**
seed within each model, with seed-number tie-break; never the best test seed.
The controlled feature comparison is fixed versus learned **within** ICNN and
within ICKAN, not a parameter-matched comparison across architectures.

Numerical approximation errors are not proofs of objectivity, polyconvexity,
global nonnegative energy, equilibrium uniqueness or RVE stability. Reference
normalization and tangent symmetry are separate mechanical checks; a sampled
rank-one screen, if shown later, is numerical evidence only. The Free Adam cap
and subsequent validation-guided procedural amendments must remain visible
when these results are reported.
