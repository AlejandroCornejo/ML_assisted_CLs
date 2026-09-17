# Material-B reference/domain preflight — 15 September 2026

**Outcome: the complete finite preflight passes its declared thresholds.**
The unchanged 4,621-element mesh passes comparison with a new 8,961-element
mesh at all 32 original pilot targets and all 18 added domain states.
Recommendation: use 4,621 elements as the working reference when defining the
data protocol, with additional checks of difficult campaign states against
the denser mesh. This is not an exact-error bound or approval everywhere in
the strain box. The worst tangent difference, 0.0890%, is close to the 0.1%
screen and must not be dismissed when setting the surrogate-error budget.

No full data campaign, training, PROM/HPROM or FE² was run. Material A and the
manuscript LaTeX were not changed. The earlier failed screens remain in the
[original pilot report](PILOT_REPORT.md) and their original JSON records.
This new result does not turn those historical failures into passes.

## Test set and mesh decision

[preflight_spec.json](../preflight_spec.json) was fixed before the new solves.
Geometry, matrix law, periodic constraints and physical targets are unchanged.
The reference/check meshes have 9,582 / 18,394 nodes and 4,621 / 8,961
quadratic triangles, each integrated with three Gauss points.

The strain convention is e = (E11, E22, 2 E12), where E is Green–Lagrange
strain. The candidate box remains E11, E22 in [-0.04, 0.10] and
2 E12 in [-0.08, 0.08]. The 18 added targets comprise all eight corners,
biaxial extension/compression, two mixed-normal states, two shear-face centers
and four normal-face centers. Each added target is reached from zero strain.
The original 32 targets retain their eight rays and four fractions.
These are pilot checks, not a frozen final surrogate-test set.

| Worst relative mesh difference, all 50 targets | Result | Declared threshold |
|---|---:|---:|
| Effective stress-vector norm | 0.042337% | 0.1% |
| Strain-tangent matrix norm | 0.088997% | 0.1% |
| Energy | 0.029030% | 0.1% |
| Weighted microscopic PK1 norm statistic | 0.036393% | 2% |
| Sampled microscopic PK1 maximum | 4.064541% | 5% |

Here s = (S11, S22, S12) is the work-conjugate second-Piola stress vector,
D = partial s / partial e is its strain tangent, and PK1 denotes first
Piola–Kirchhoff stress. Energy and effective stress use the full cell area.
For each quantity the difference is divided by the denser-mesh norm, with
the existing dimensional numerical floor of 1 in the quantity's solver units.
The nonzero-state norms exceed that floor. Vector/matrix comparisons do not
bound relative errors in individual near-zero components.

The microscopic norm statistic is sqrt(sum(w ||P||_F^2) / sum(w)), where
P is microscopic PK1 and w are solid integration weights. The maximum is
max ||P||_F over the integration points. Neither is a pointwise field-error
estimate: no cross-mesh field interpolation was performed.

The first four maxima occur at e = (-0.04, -0.04, -0.08); the sampled-maximum
difference is largest at e = (-0.04, -0.04, +0.08). Thus the added corner
checks are materially more demanding than the original rays. The 0.1%
output thresholds were declared before these solves; the original 2% / 5%
microscopic-statistic thresholds were not relaxed.

## Continuation and numerical implementation

The B-only predictor updates the affine displacement throughout the mesh.
The independent unknown q is **total displacement**, with full displacement
u = T q + g(e); T applies periodic identifications and g carries the imposed
lift. Between two strains, the predictor adds [(F_new - F_old) X] at the
independent coordinates X. Its compatibility identity is

    T delta_q + delta_g = [(F_new - F_old) X] at every full displacement DOF.

This changes the Newton initial guess, not the equilibrium equations,
material parameters or stopping tolerances. Failed increments are logged
and halved, keeping the last successful state. The maximum engineering-strain
increment norm is 0.01 and the minimum is 1e-6.

Before using it for new states, six previously successful endpoints were
recomputed from zero with maximum increments 0.01 and 0.005. All 12 tests
passed the 1e-7 agreement screen. Worst relative differences against the
earlier dense ramps were 1.47e-14 in stress, 3.55e-14 in tangent, 3.85e-16
in energy and 1.41e-15 in microscopic node displacements. Agreement supports
the computed branch; it does not demonstrate uniqueness.

The new reference/check stages reached all their targets with **zero rejected
increments**. Recorded affine-identity errors were below 2.1e-17. Across their
68 target states, the worst reduced equilibrium residual was 4.54e-13;
the minimum microscopic determinant at quadrature points was 0.803635.
Positive sampled determinants were also checked at each accepted increment.

Three declared corners were checked on both meshes with central differences
at steps 1e-4 and 5e-5: all 12 derivative checks passed. Worst discrepancies
were 1.73e-7 for stress versus energy gradient, 4.84e-8 for the tangent,
and 1.84e-8 for asymmetry of the unsymmetrized finite-difference tangent,
against thresholds of 1e-4. The solver's internally symmetrized tangent is
not counted as an independent symmetry test.

Endpoint polygons through cavity boundary nodes showed no overlap or
self-intersection in the tested neighboring periodic images; minimum gap
was 0.283264 in cell coordinates. Periodic displacement-jump checks passed.
These straight polygons do not certify the curved boundaries or intermediate
states. Stress-dependent acoustic-matrix screening at 72 unit directions
per target found positive sampled rank-one curvature, minimum 238.634 MPa.
This screen is reported separately, not used as a polyconvexity certificate.

The geometry, shared periodic solver, configuration, core FOM solver and
material-file hashes still match the original pilot. The pilot driver's
only intervening changes are plot styling. Helper/specification hashes are
locked between validation and the new stages; driver hashes are retained.
The consolidation also records its own source hash and the historical
y-compression retry report hash. Synthetic tests check failure retention and
record integrity; they are not physical demonstrations.

## Next move and scope

Define the data protocol before generating the full campaign: domain,
sampling counts, seeds, disjoint partitions, independent test paths,
near-zero metric normalization, checkpoint selection and optimization budgets.
Include a logged failure/retry policy and a denser-mesh audit of difficult
states, especially combined compression and shear. Pilot wall times with
overlapping processes are not a speedup benchmark or a calibrated campaign
cost estimate. Keep B's artifacts separate from A.

The candidate box is kinematically admissible by the C-positive-definiteness
bound recorded in the original pilot. Neither that analytical bound nor
these finite numerical checks establish full-domain microscopic stability,
contact absence, uniqueness, exact discretization accuracy or representability
by a globally polyconvex homogenized law. No learned-feature benefit or
cross-geometry transfer has yet been measured.

Raw evidence:

- [Predictor validation](../preflight_validation_v1/report.json).
- [Added states on the working reference](../preflight_reference_v1/report.json).
- [All 50 states on the denser check mesh](../preflight_check_v1/report.json).
- [Consolidated decision and per-state comparisons](../results/preflight_decision_v1.json).
- [Reproduction instructions](../README.md#reference-domain-preflight).
