# Material-B pilot — 15 September 2026

Recommendation: retain this geometry as the candidate second material.
The FOM pilot provides strong numerical support for the homogenized responses
on the tested paths, but **does not pass the complete declared mesh screen**.
Do not start full data generation or training yet. No material-A source,
mesh, data or trained model was changed; these results are not inserted into
the manuscript as final second-material evidence.

## Geometry and loads

The square cell has side L = 2 at the same arbitrary cell scale as material A.
Four elliptical, traction-free cavities occupy exactly 20% of its analytic area.
The matrix remains compressible Neo-Hookean, E = 1.628 GPa and nu = 0.4,
under plane strain. Effective quantities use the full reference-cell area.

| Cavity | Center / L | Area / cell area | Aspect a/b | Angle |
|---|---|---:|---:|---:|
| 1 | (-0.24, -0.21) | 0.055 | 1.6 | 18 degrees |
| 2 | (0.23, -0.24) | 0.045 | 2.1 | -37 degrees |
| 3 | (-0.20, 0.24) | 0.060 | 1.4 | 63 degrees |
| 4 | (0.25, 0.21) | 0.040 | 1.8 | 107 degrees |

Here a and b are major/minor semiaxes and k = a/b is the aspect ratio.
For each area fraction f, b = L sqrt(f/(pi k)) and a = k b.
The cavities are distinct, not a repeated 2 x 2 cell. Their enclosing circles
give a conservative periodic separation bound of 0.24178, or 0.12089 L.
This is an analytic sufficient-separation argument evaluated numerically,
not an exact minimum-ligament calculation. Solid connectivity and periodic
node correspondence passed the mesh checks.

The eight rays and four fractions per ray were specified before the FOM runs.
They exercise axial extension/compression in both directions, both Green-shear
signs and two mixed states. All 32 nonzero targets are retained.
The strain vector is e = (E11, E22, 2 E12): axial strain is not uniaxial
stress, and an off-diagonal Green-strain ray is not constant-area simple shear.
The conjugate stress vector is s = (S11, S22, S12); the strain tangent is
D = partial s / partial e.

[Geometry](pilot_v2/geometry.png) · [FOM response curves](pilot_v2/pilot_response.png)
· [Example deformed field](pilot_v2/fine_combined_field.png).
The field visualization uses triangle corners and element means of Gauss-point
first-Piola stress norms; the saved fields retain the actual quadratic-mesh data.

## Numerical evidence

The three meshes have 1,088 / 2,340 / 4,621 quadratic triangles
(3,264 / 7,020 / 13,863 integration points).
Opposite boundary nodes match after coordinate rounding. Reference-element
Jacobians were positive at 66 sampled points per triangle.
That sampling is not an everywhere-positivity proof.

| Worst relative difference across the 32 targets | Coarse / fine | Fine / finer | Declared screen |
|---|---:|---:|---:|
| Effective stress, vector norm | 0.2214% | 0.0774% | 1% |
| Strain tangent, matrix norm | 0.1427% | 0.0514% | 2% |
| Volume-weighted microscopic PK1 L2 norm | 0.1529% | 0.0486% | 2% |
| Sampled microscopic PK1 maximum | **6.2133%** | **5.0764%** | **5%** |

PK1 means first Piola–Kirchhoff stress. These are mesh-sensitivity differences,
not errors against an exact solution and not pointwise field-error estimates.
The worst fine/finer maximum difference occurs at the negative-Green-shear
endpoint. It is not rounded into a pass.

On both coarse/fine meshes, three endpoints were checked with two finite-
difference steps. Worst relative discrepancies were 6.87e-7 for stress versus
energy gradient, 5.17e-8 for the strain tangent, and 3.26e-9 for symmetry of the
unsymmetrized finite-difference tangent. The existing consistent tangent is
symmetrized internally; its symmetry alone is not counted as a test.
Worst reduced equilibrium residuals were below 3.4e-13.
Reference energy and stress were zero in these computations.

Independent three-point linear assembly agreed with the reference tangents
to below 3e-14 relative difference. Six-point quadrature changed them by
0.0136% (coarse) and 0.0032% (fine).
Native Kratos element forces/stiffness agreed with vectorized assembly at a
mixed endpoint to below 1.4e-12 / 7.4e-15 respectively.
These are code/derivative/quadrature checks, not independent physical experiments.

Saved endpoint displacement jumps matched the imposed periodic jumps to below
1e-17. Microscopic determinants remained positive at quadrature points;
the minimum across the combined three-mesh results was about 0.93285.
Straight polygons through deformed cavity nodes showed no endpoint overlap
or self-intersection; their smallest tested separation was 0.29441.
This does not certify the curved boundary or intermediate states.
Stress-dependent rank-one screening at the coarse/fine states and 72 sampled
directions found no negative curvature (minimum about 266.87 MPa).
It does not establish rank-one convexity, polyconvexity or uniqueness.

## Failures and continuation

The continuation density d sets the substep count to
max(1, ceil(d ||e_target - e_start||_2)). It changes numerical increments,
not the geometry, matrix law or target state.

The first run failed in coarse x compression and fine y compression at
E11 = -0.01 and E22 = -0.01, respectively, with d = 200.
Those failures remain in [pilot_v1](pilot_v1/report.json).
The repeated coarse/fine pilot at d = 400 reached all targets; independent
zero-start ramps at d = 800 agreed in stress to below 7e-15 relative difference.
Its overall screening flag remains false because of the peak-stress criterion.

The finer mesh at d = 400 failed partway to the E22 = -0.03 target.
The unchanged y-compression path was retried at d = 800 and 1600:
both reached all four targets and agreed to below 4e-15 in stress and
3.7e-14 in tangent. Both retries used the identical mesh hash.
No failed physical target was removed.
The fine/finer table uses the original finer run for the other seven paths
and the complete d = 1600 retry for all four y-compression samples.
This predetermined more-refined continuation is not selection of the
lowest-error result. There are 32 distinct successful comparison targets.
Original failure flags are preserved.

Agreement of these continuation routes supports the computed branch;
it does not prove that no other microscopic equilibrium exists.
Wall times are pilot timings with overlapping diagnostic processes,
not speedup benchmarks or full-data-generation cost estimates.

Raw evidence:

- [Repeated pilot](pilot_v2/report.json).
- [Original finer-mesh attempt](refinement_v1/report.json).
- [Targeted retry, d = 800](refinement_retry_800/report.json) and
  [d = 1600](refinement_retry_1600/report.json).
- [Saved-field audit](saved_field_audit_v2.json).
- [Complete reference cross-check](reference_verification_v2_complete.json).

The partial reference-check file records an invocation before the fine
endpoint existed; it is marked partial, not used as a complete verification.
The principal source files used for pilot_v2 are preserved in source_snapshot;
their hashes match its report. Subsequent plot-styling changes did not rerun
or modify the numerical results.

## Candidate domain and next move

A candidate for the constitutive-data protocol is
E11, E22 in [-0.04, 0.10] and 2 E12 in [-0.08, 0.08].
It exposes both normal directions, shear signs and mixed couplings without
claiming a new structural application. Its bounds follow the pilot extents,
not learned-model test errors. **This box is not frozen or FOM-validated.**

The box is kinematically admissible: C = [[1+2E11, 2E12], [2E12, 1+2E22]]
has diagonal entries at least 0.92 and off-diagonal magnitude at most 0.08.
Thus v^T C v >= 0.84 ||v||^2 for every planar vector v, using
2 |v1 v2| <= v1^2 + v2^2. This elementary bound proves C is
positive definite throughout the box; it says nothing about microscopic
equilibrium, contact, stability or discretization accuracy.

Before freezing the campaign:

- Check biaxial/volumetric and box-boundary states not covered by the rays.
- Resolve reference-mesh approval. The declared peak screen still fails;
  the 4,621-element mesh is a prospective reference, not certified accurate.
  Any decision to limit accuracy claims to homogenized outputs must be explicit,
  not a retrospective change to the recorded field-screen threshold.
- Set a reference-error budget appropriate to the planned surrogate errors,
  and a logged continuation/retry policy. Avoid silently excluding hard states.
- Then freeze sampling, splits, independent test paths, metrics, optimization
  budgets and initialization seeds. No model has yet been trained for B.

This pilot does not establish contact absence, global finite-strain stability,
global nonnegative energy, uniqueness, random-material representativity,
cross-geometry transfer, or benefits of learned features.
