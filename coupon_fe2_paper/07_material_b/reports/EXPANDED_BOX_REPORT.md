# B: expanded-box preflight — 15 September 2026

**Result: do not adopt the complete enlarged box.** All 20 targets were
reached on both cached meshes, but two cross-mesh comparisons failed the
predeclared thresholds. A denser-mesh cavity-contour screen also failed.
The original box and all historical records remain unchanged. No training
domain has been frozen, and no B data campaign or neural training has begun.

[Frozen specification](../expanded_box_spec.json) ·
[Decision and checks](../results/expanded_decision_v1.json) ·
[Reference stage](../expanded_reference_v1/report.json) ·
[Denser stage](../expanded_check_v1/report.json).

## Test and results

The candidate is E11, E22 in [-0.04, 0.20], with gamma = 2 E12 in
[-0.16, 0.16]. The engineering Green-strain vector is e = (E11, E22, gamma).
Geometry, Neo-Hookean matrix, quadrature and 4,621 / 8,961-element periodic
meshes were unchanged. Corners, face centers, biaxial/mixed normal loading
and both pure Green-shear signs give 20 comparable targets. Two exact saved
endpoints per mesh were reused with provenance; 18 new targets per mesh
were solved from corresponding saved smaller-domain states.

These maxima all occur at e = (-0.04, -0.04, -0.16):

| Quantity | Mesh difference | Declared limit | Outcome |
|---|---:|---:|---|
| Homogenized material stress vector | 1.0944% | 0.1% | Fail |
| Engineering-strain tangent matrix | 2.2531% | 0.1% | Fail |
| Homogenized energy | 0.1366% | 0.1% | Fail |
| Weighted microscopic PK1 RMS statistic | 0.0646% | 2% | Pass |
| Sampled microscopic PK1 maximum | 24.8497% | 5% | Fail |

Differences use the denser-mesh norm, with floor 1 in recorded units.
The stress vector is s = (S11, S22, S12), and D = partial s / partial e.
Microscopic entries compare scalar statistics, not pointwise fields: RMS
is sqrt(sum(w ||P||²) / sum(w)) and the peak is max ||P|| over quadrature
points. P is microscopic first-Piola stress; w are reference integration
weights including thickness.

At e = (-0.04, -0.04, 0.16), tangent and sampled-peak differences are
0.3042% and 7.9155%; both fail. The other 18 pairs pass all mesh thresholds.
This does not approve intermediate states or a domain obtained by removing
only two failed sample points. No thresholds or failure flags were relaxed.

All 12 selected FD checks passed: worst energy-gradient discrepancy
8.42e-8, tangent discrepancy 1.54e-7, and unsymmetrized FD tangent asymmetry
7.33e-8, against 1e-4. Four independent zero-start checks agreed within
1e-7; largest discrepancy 7.37e-14 in D. These cover two selected targets
per mesh, not every state or the failed negative corner. One denser-mesh
Newton increment was rejected and successfully halved; its record is retained.

## Contour diagnostic and limitations

The denser negative-corner solution has two proper node-polygon crossings
in the second cavity. Angular node ordering matches the actual quadratic
FE boundary connectivity. Sampling the quadratic edges retains two crossings.
The [saved-field diagnostic](../results/expanded_field_audit_v1/audit.json) and
[contour figure](../results/expanded_field_audit_v1/compression_shear_contours.png)
preserve this additional inspection without revising declared flags.

Quadrature determinants are positive (minimum 0.57556). A separate
66-point-per-element check on all 40 saved targets also found positive
determinants (minimum 0.36874). Finite local checks do not guarantee global
injectivity or absence of self-contact. Worst reduced residual is 3.21e-13.
Sampled stress-dependent rank-one curvature remains positive, minimum
219.50 MPa; this is not a polyconvexity or microscopic-stability proof.
The contour concern does not establish physical buckling, plasticity or
exact contact. No contact model was introduced. Mesh sensitivity is not an
exact-solution error bound.

The entire candidate is macroscopically kinematically admissible:
C = [[1+2 E11, gamma], [gamma, 1+2 E22]] has diagonals at least 0.92 and
|gamma| <= 0.16. Hence vᵀ C v >= 0.92 ||v||² - 0.32 |v1 v2|
>= 0.76 ||v||² for every vector v. This algebraic bound does not rescue
the failed microscopic/mesh screens.

Source, mesh and seed hashes and exact used-driver snapshots were checked
during consolidation. Thirty-six synthetic/exact unit tests passed;
these do not establish physical validity. Denser stage and consolidator
return code 1 for their recorded screen failure, not an unhandled error.
A, its artifacts, the shared FOM and manuscript LaTeX were not edited.

## Next decision and the reviewers' nonlinearity question

The plan remains two separately trained materials, a fixed/learned-feature
comparison, and FE² only for A. Domain design precedes protocol freezing.
The recommendation is to test more asymmetric bounds: increase normal
tensile limits to 0.20 while retaining compression -0.04 and original
Green shear ±0.08. This needs a new declared combined/boundary screen;
it is **not yet approved**. No strain-dependent envelope is defined yet.

The earlier project's
[stage-0 generator](../../../RVE_NeoHookean_Homogenization/trajectories/stage0_training_trajectory.py)
specifies six directional limits independently. Its
[saved bundle](../../../RVE_NeoHookean_Homogenization/trajectories/stage_0_trajectory/stage_0_trajectories.npz)
declares normal bounds [-0.1, 2.0] and
engineering shear ±0.1. This supports asymmetric sampling design, not
copying its amplitudes to B or newly validating its historical FOM campaign.

Make nonlinear response visible using FOM curves, the fixed reference
prediction D0 e, and tangent changes. The
[completed larger-ray exploration](NONLINEAR_EXPLORATION.md) measures
50–52% stress-vector departure at axial Eii = 0.20 and axial tangent ratios
about 0.40 relative to rest. These are finite hyperelastic responses,
not approval of every combination of their amplitudes.

A read-only [saved-reference audit](../results/reference_linearity_v1.json) gives
42.64% aggregate stress discrepancy on A's 400 independent test states.
Its saved C0 uses independent six-point linear assembly versus three-point
finite-strain FOM quadrature; it is not a bitwise-identical assembly claim.
A more demanding [fit-only constant-tangent diagnostic](../results/constant_tangent_v1.json)
still gives 11.25% on the same independent test. It uses the existing
4208/742 split, fits only 4208 rows, and enforces neither symmetry nor
stability. It is not the neural Regression baseline or a new energy model.
Aggregate discrepancy is ||s - D e|| over all states divided by ||s|| over
all states. No nonfinite pairs were excluded from these A diagnostics.

A large reference error alone does not prove that every fitted linear model
is inaccurate; the second diagnostic addresses that weakness empirically.
Neither statistic is a universal approximation lower bound. These audits
inform how to present nonlinearity, not neural model/checkpoint selection
or B domain choice using learned-model test errors.
