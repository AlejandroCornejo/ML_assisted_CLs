# Frozen material-B data and comparison protocol — v1

This protocol fixes the strain coordinates, data roles, mesh audit, model
comparison and reporting rules **before** generating the full FOM labels. Its
machine-readable source is [data_protocol_v1.json](data_protocol_v1.json).
The domain is the separately screened asymmetric box
E11,E22 in [-0.04,0.20] and 2E12 in [-0.08,0.08]; see the
[preflight report](../reports/ASYMMETRIC_BOX_REPORT.md). Passing that finite
screen motivates the limits but does not certify every point in the box.

## Sampling and separation of roles

The fit set contains 4,200 states: 4,096 Owen-scrambled Sobol points in the
open volume, 16 independent two-dimensional Sobol points on each of the six
faces, and all eight corners. The explicit boundary samples prevent the
closed-box boundary from becoming an accidental extrapolation. Independent
scrambles produce 512 validation and 512 test states in the open volume.
The reference state is stored separately.

Validation selects checkpoints; it does not select the domain, sampling rule,
architectures or test metric. Test labels remain unopened until every model
and seed is locked. Ten predeclared straight FOM paths contain 41 points each
and are never used for fitting or selection. Their shared reference point is
not scored. No out-of-domain probe is added for B: the separate A probe remains
supplementary extrapolation evidence, whereas B tests a second microstructure
and the fixed-versus-learned feature question.

This design deliberately does not copy A's Cartesian grid. A required that
grid for graph-coupled MAW-ECM, the reduced decoder and an affine reduced-map
fit. B has no POD, HPROM, MAW-ECM or FE² deployment. A low-discrepancy design
therefore covers the three-dimensional constitutive box without paying for
unused neighborhood/indexing structure.

## FOM and mesh policy

The 4,621-element mesh supplies working labels. A predeclared 64-state subset
is repeated on the 8,961-element mesh: 16 strain-only maximin states from each
of fit volume, validation and test, plus all eight corners and eight maximin
face states. Twenty-four states are independently reached from zero to check
ordering/branch agreement. Selection uses strain coordinates only, before any
label is evaluated.

Continuation limits the Euclidean norm of each engineering-strain-vector
increment, $(\Delta E_{11},\Delta E_{22},\Delta\gamma_{12})$, to 0.01,
restores the last converged state and halves failed increments down to the
frozen minimum. A failed requested state is retained as a failure;
it is not replaced by an easier sample. Training does not begin unless every
fit, validation, test and path endpoint is available under this policy.
Energy and stress are stored everywhere. Consistent tangents are stored for
the reference, test, path and mesh-audit states, where they provide evidence;
they are not computed at thousands of fit points merely to remain unused.
The 24 selected cold-start states are the only exception: their warm and cold
tangents and independent displacement coordinates are retained so branch
agreement is checked on the complete constitutive state, not on stress alone.
Every sampled state also records the reduced residual, quadrature-point
microscopic determinant, exact periodic-jump error and deformed cavity-polygon
gap/self-intersection screen. These finite polygon and quadrature checks still
do not prove injectivity or absence of contact between sampled states.

## Models and the controlled claim

B compares Free, ICNN-fixed, ICNN-learned, ICKAN-fixed and ICKAN-learned.
Regression remains the direct-stress/integrability witness on A; repeating it
on B would not answer the feature question. One admissible 32-feature table is
selected once from fit labels and the reference tangent by the frozen
NNLS/fit-only pivoted-QR rule. It is shared by both cores. Fixed variants keep
it unchanged; learned variants start from precisely the same table and update
directions and admissible exponents.

Thus fixed versus learned is a controlled comparison within each core. It is
not a claim that ICNN, ICKAN and Free have equal parameter counts. Widths,
parameter counts, wall times and all three initialization seeds are reported.
Each seed is selected only by validation stress. Tables report every seed and
mean plus standard deviation; figures use the median-validation seed, never
the best test seed.

The common objective uses energy, stress and the same reference-tangent
anchor. Normalizations and all per-state metric floors come only from fit.
Aggregate energy, stress and tangent errors are accompanied by componentwise
errors and median/95th-percentile/maximum distributions. Mechanical audits
remain distinct from approximation metrics and from architectural proofs.

## Current state

`prepare_design.py` freezes coordinates and hashes without importing the FOM
or a neural model. Adding geometry/contour provenance reproduced the original
coordinate NPZ bit for bit (SHA-256
`fca1626aab3d4dee9538a57cf02f102b44eb2fc500bb055f36d8f093536449a8`).
Bounded throughput, pause/resume, working/check-mesh and cold-start smoke tests
have completed. Full FOM generation and assembly are complete; the
[campaign report](../reports/CAMPAIGN_REPORT.md) records 20/20 passing checks.
Raw records remain in `../results/data_campaign_v1/`; approved labels and
provenance are in `../results/data_labels_v1.npz` and
`../results/data_labels_v1.json`. All neural training remains pending.

The campaign comprises 103 restartable chunks: 4,200 fit, 512 validation,
512 test, 400 nonreference path states and one reference, plus 64 repeated
working-mesh audit states, 64 check-mesh states and 24 independent cold starts
(5,777 solves in total). Training is gated on the final assembler result,
including every mesh and cold-start comparison; that gate now passes.
The maximum budgets above alone are not a complete optimization recipe.
Exact optimizer/learning-rate details are now fixed by the separate
[training-recipe addendum](TRAINING_RECIPE.md),
without rewriting the frozen data JSON. Its numerical details were completed
after assembly, before feature selection and before any neural optimization.
The shared 32-feature table and all 15 initialization checks are complete;
see [TRAINING_PREPARATION.md](../reports/TRAINING_PREPARATION.md). A resumable
B-specific training runner and trained checkpoints remain pending.
