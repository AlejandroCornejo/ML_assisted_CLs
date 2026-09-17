# Material B — completed constitutive FOM campaign

Decision: **PASS under the frozen numerical screens**, completed on
15 September 2026 local time (16 September UTC). All 103 jobs and 5,777
requested solutions are available; no failed target or rejected continuation
increment is recorded. No sample was replaced, no tolerance was relaxed,
and no neural model was trained.

## Data and provenance

The [protocol](../protocol/DATA_PROTOCOL.md) uses
e = (E11, E22, 2E12), with E11/E22 in [-0.04, 0.20] and
2E12 in [-0.08, 0.08]. The geometry is the separately specified four-cavity
periodic cell. The working/check meshes contain 4,621/8,961 quadratic triangles.

| Role | States | Stored constitutive labels |
|---|---:|---|
| Fit | 4,200 | Energy and second-Piola stress; tangents at eight warm/cold audit states |
| Validation | 512 | Energy and stress; tangents at eight warm/cold audit states |
| Test | 512 | Energy, stress and tangent |
| Ten paths | 400 nonreference states | Energy, stress and tangent |
| Reference | 1 | Energy, stress and tangent |
| Working/check mesh audit | 64 + 64 repeats | Energy, stress, tangent and field statistics |
| Independent cold starts | 24 repeats | Energy, stress, tangent and independent displacements |

The [assembled labels](../results/data_labels_v1.npz) contain 5,625 main
constitutive states and the separate audit labels. Tangents not requested at
fit/validation states remain `NaN`, not zero. The origin is stored separately;
each path has 40 nonzero states, with its 41-point parameter array retained.
Residual, minimum quadrature J, polygon gap and periodic-jump diagnostics are
retained for the main sets; complete state diagnostics and audit displacements
remain in the raw chunk files.

The [machine-readable approval](../results/data_labels_v1.json) records all
20 passing checks, 64 mesh comparisons, 24 cold comparisons, source/chunk hashes
and the sampled curvature results. The
[campaign manifest](../results/data_campaign_v1/manifest.json) records the
frozen 103-job plan. A terminal interruption was recovered from atomic state
checkpoints without changing sources or coordinates; it is not a mechanical
failure. The observed start-to-finish elapsed time was 69 min 28 s, including
that interruption, with four workers. Resumed job wall times alone are not
a total computational-cost measurement.

Independent reassembly reproduced the NPZ **bit for bit**:

`c2edcb3cc663c75f342f66927335c700d30b2867b0a16b9ff668905366dc32e2`

This checks deterministic assembly of saved chunks, not bitwise reproducibility
of a fresh FOM solve on another machine.

## Numerical acceptance

| Cross-mesh quantity | Worst relative difference | Frozen threshold |
|---|---:|---:|
| Stress vector | 0.0423372% | 0.1% |
| Engineering-strain tangent | 0.0889966% | 0.1% |
| Energy | 0.0290297% | 0.1% |
| Microscopic first-Piola weighted RMS norm | 0.0363935% | 2% |
| Sampled microscopic first-Piola maximum norm | 4.06454% | 5% |

Relative differences use the check-mesh value in the denominator, with the
existing numerical floor. These compare homogenized outputs and microscopic
statistics, not pointwise correspondence between different meshes. The tangent
still uses roughly 89% of its threshold: the close margin is not hidden by
rounding or interpreted as an exact-solution error bound.

The 24 warm/cold comparisons have worst relative differences of
8.68e-15 in stress, 6.04e-14 in tangent, 7.41e-16 in energy and
2.11e-15 in independent displacements, versus the unchanged 1e-7 tolerance.
This supports ordering/branch agreement at those states, not global uniqueness.

Across all generated states, minimum microscopic quadrature J is 0.803635,
minimum deformed-cavity polygon gap is 0.245537, maximum periodic-jump error
is 3.93e-17, and maximum reduced residual is 2.84e-12 (limit 1e-7).
No cavity polygon self-intersects. Reference stress and energy are zero;
all main effective energy labels are nonnegative.

The stress-dependent rank-one screen covers 1,065 tangent-labelled audit/test/
path/reference states, using acoustic-matrix eigenvalues for all vectors a at
72 sampled unit directions b. The minimum observed curvature is 238.634 MPa.
It is a finite numerical screen, **not proof of rank-one convexity,
polyconvexity, absence of buckling or complete stability**. Consistent tangents
are symmetrized internally, so their symmetry alone is not an independent test.
Energy-gradient and unsymmetrized FD-tangent checks were performed in the
preceding preflights, not at every campaign state.

Polygon chords and sampled quadrature determinants do not certify the curved
boundary, positivity everywhere, injectivity or absence of contact along every
intermediate continuation state. Two meshes establish finite sensitivity,
not asymptotic convergence. No contact, damage or plasticity model is introduced.
The rejected ±0.16 shear candidate remains rejected.

## Next gate

The recipe and shared-feature preparation are now complete; see the separate
[preparation report](TRAINING_PREPARATION.md). This does not introduce trained
results into the FOM campaign report.

1. Implement and smoke-test the resumable B-specific optimization runner using
   the exact recipe, shared table and frozen normalizations.
2. Run the predeclared five models and three seeds, including the controlled
   fixed/learned comparisons for each core.
3. Lock all 15 checkpoints before consulting prediction metrics on test/paths.
   [The prepared path plotting script](../protocol/plot_campaign_response.py)
   enforces a complete checkpoint lock and approved data hashes. It has not
   opened the reserved path curves. Existing preflight response figures remain
   the pretraining evidence of nonlinearity.

The campaign implementation suite passed 65/65 tests, including synthetic
integrity rejection; preparation adds 16 tests (81/81 passing). Material A,
shared FOM code and manuscript LaTeX were not
modified by this campaign. No second FE²/POD/HPROM campaign was launched.
