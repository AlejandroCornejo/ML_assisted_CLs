# B: asymmetric-box preflight — 15 September 2026

**Result: the declared finite screen passes.** The candidate uses
E11, E22 in [-0.04, 0.20] and engineering Green shear
gamma = 2 E12 in [-0.08, 0.08]. It extends only the positive normal limits
of the original screened box. All 18 targets were reached on both unchanged
meshes, and every predeclared mesh, field, derivative and independent-start
check passed without relaxing a threshold.

[Frozen specification](../asymmetric_box_spec.json) ·
[Decision and checks](../results/asymmetric_decision_v1.json) ·
[Reference stage](../asymmetric_reference_v1/report.json) ·
[Denser stage](../asymmetric_check_v1/report.json).

## What was checked

The 18 targets comprise all eight box corners, biaxial compression and
extension, two mixed-normal states, two shear-face centers and four
normal-face centers. Three exact original-box endpoints per mesh were reused
with recorded provenance; the other 15 targets per mesh were newly solved.
Thus the comparison contains 18 unique cross-mesh pairs, 30 new target solves
and six exact endpoint reuses. The reference and check meshes contain 4,621
and 8,961 quadratic triangles, respectively.

| Quantity | Worst mesh difference | Declared limit | Worst state |
|---|---:|---:|---|
| Homogenized material stress vector | 0.04234% | 0.1% | old low-low, negative-shear corner |
| Engineering-strain tangent matrix | 0.08900% | 0.1% | old low-low, negative-shear corner |
| Homogenized energy | 0.02903% | 0.1% | old low-low, negative-shear corner |
| Weighted microscopic PK1 RMS statistic | 0.03639% | 2% | old low-low, negative-shear corner |
| Sampled microscopic PK1 maximum | 4.06454% | 5% | old low-low, positive-shear corner |

The worst values are unchanged from the original-box preflight. Among only
the 15 newly solved states per mesh, the maxima are 0.00457% in stress,
0.00879% in tangent, 0.00530% in energy, 0.00489% in the microscopic RMS
statistic and 1.4098% in the sampled microscopic maximum. These smaller
values support the positive-normal extension, but they do not increase the
margin of the old tangent or peak-stress checks.

All 12 selected finite-difference checks passed. Their worst discrepancies
were 8.48e-8 for stress as the energy gradient, 6.21e-8 for the tangent and
4.33e-9 for symmetry of the unsymmetrized finite-difference tangent, against
the common tolerance 1e-4. Four selected states were also reached
independently from zero. The largest discrepancy from the continued solution
was 6.53e-14 in the tangent, against 1e-7. No continuation increment was
rejected in either stage.

## Physical screens and interpretation

All saved endpoint screens passed. The minimum quadrature-point microscopic
determinant was 0.80364, the maximum reduced residual was 1.98e-13, the
minimum sampled deformed-cavity polygon gap was 0.24554, and the maximum
periodic-jump error was 2.78e-17. The minimum sampled stress-dependent
rank-one curvature was 238.63 MPa. These are numerical endpoint screens;
they are not proofs between targets.

The full macroscopic candidate is kinematically admissible. With
C = [[1+2 E11, gamma], [gamma, 1+2 E22]], both diagonal entries are at least
0.92 and |gamma| is at most 0.08. Hence the smallest eigenvalue is bounded
below by 0.84. This algebraic fact does not establish microscopic injectivity,
absence of contact or equilibrium uniqueness.

The result supports adopting these asymmetric limits for the B data protocol.
It is a finite preflight, not a validation of every point in the continuous
box, an exact-solution error bound, a convergence theorem or a proof of
stability, contact avoidance or branch uniqueness. The tangent maximum remains
close to its 0.1% limit, so the campaign protocol must retain denser-mesh
audits of difficult states and preserve all failures. The previously failed
candidate with shear extended to ±0.16 remains rejected; this pass does not
rehabilitate it.

The consolidated record verifies frozen specifications, source and helper
hashes, used-driver snapshots, field-file hashes and agreement between saved
NPZ arrays and reports. Material A, the shared FOM, the constitutive law and
the manuscript LaTeX were not changed. No training data campaign or neural
training was started.
