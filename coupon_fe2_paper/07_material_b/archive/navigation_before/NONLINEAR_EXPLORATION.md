# B: bounded larger-strain exploration — 15 September 2026

**Result: B reaches the six additional targets on both meshes, with a more
pronounced departure from its reference linearization.** This exploration
does not change the training box or establish reliability in a larger box.

[Physical comparison figure](nonlinear_response_v1/nonlinear_response.png)
· [PDF](nonlinear_response_v1/nonlinear_response.pdf)
· [Decision, thresholds and numerical values](nonlinear_response_v1/decision.json).

## What was tested

The same geometry, matrix law and 4,621 / 8,961-element periodic meshes were
used. [The exploration specification](nonlinear_exploration_spec.json) was
fixed before new solves: axial Green strains of 0.15 and 0.20 in each axis,
and positive Green-shear components 2 E12 of 0.12 and 0.16. Each path starts
from its already computed original endpoint and uses the validated affine
predictor with logged step-halving. No compression extension, further load
levels, data campaign, training or FE² was performed. A and manuscript LaTeX
were left unchanged.

For each mesh, the zero-strain tangent D0 was computed separately. The
reference linearization is s_linear = D0 e, where
e = (E11, E22, 2 E12), s = (S11, S22, S12), and D = partial s / partial e.
The plotted deviation is ||s - D0 e|| / ||s||. This uses the full stress
vector and the same mesh; it is not a deviation from a fitted secant line.

| Path | Original endpoint | New endpoint | Original linear deviation | New linear deviation |
|---|---:|---:|---:|---:|
| Axial X | E11 = 0.10 | E11 = 0.20 | 24.4% | 50.1% |
| Axial Y | E22 = 0.10 | E22 = 0.20 | 25.3% | 51.8% |
| Positive Green shear | 2 E12 = 0.08 | 2 E12 = 0.16 | 16.0% | 29.8% |

These percentages are normalized by FOM stress: they do not mean that each
stress component is the same percentage below its linear prediction.
At the new axial endpoints, D11/D0,11 and D22/D0,22 are about 0.402 and
0.392. The slope of second-Piola stress against Green strain decreases;
this is not plasticity, a yield event or a demonstrated instability.
In the shear case, induced normal stresses contribute substantially to the
vector deviation; a shear-component-only plot hides that contribution.

The previous plots looked nearly linear on their chosen scale, but the
original range already has appreciable nonlinearity relative to D0.
The new plots explicitly show that reference and all three stress components.
They use saved states with straight connecting guides, not a fitted smooth
model. The shaded region is the old range of each individual ray, not a
projection proving coverage of a three-dimensional box.

The axial values are Green strains, not engineering elongations:
Eii = 0.20 corresponds to axial stretch sqrt(1.4), about 18.3% elongation.
Transverse normal strain and shear are held at zero in axial loading;
this is not uniaxial stress. Green shear is not constant-area simple shear.

## Numerical support and limits

All six cross-mesh comparisons passed the unchanged prospective-reference
screens. Maximum relative differences, using the denser-mesh norm, were:

- Stress 0.02360%, tangent 0.02112%, energy 0.01544%: each below 0.1%.
- Weighted microscopic PK1 norm statistic 0.02085%: below 2%.
- Sampled microscopic PK1 maximum 1.36082%: below 5%.

No continuation increments were rejected. Minimum sampled microscopic J
was 0.90055, minimum endpoint cavity-polygon gap was 0.26643, and the worst
reduced residual was 3.18e-13. Endpoint periodicity and polygon screens
passed. Sampled stress-dependent rank-one curvature stayed positive,
minimum about 284.57 MPa over 72 directions per target.
These are finite numerical screens, not proofs of full-domain stability,
exact field accuracy, absence of curved-boundary contact or uniqueness.

At the new X-extension endpoint, two finite-difference steps on both meshes
gave four derivative checks. Worst stress/energy-gradient discrepancy was
5.32e-8, tangent discrepancy 5.60e-8 and unsymmetrized FD tangent asymmetry
4.28e-9, each below the original 1e-4 screen.

This does not demonstrate an advantage of learned features, material isotropy,
cross-geometry transfer or a particular model's extrapolation accuracy.
Larger amplitudes alone do not establish that B is a harder learning problem.

## Records and next decision

[Reference repeat](nonlinear_reference_v2/report.json) and
[denser check](nonlinear_check_v1/report.json) retain states, attempts, fields
and source hashes. The first reference invocation had a baseline-record
construction TypeError before any extended state was recorded. Its
[incomplete record](nonlinear_reference_v1/INCOMPLETE.md) is retained and
excluded; this was an orchestration bug, not mechanical nonconvergence.
Used driver snapshots preserve the exact versions. The correction changes
only baseline metadata construction, not the FOM or continuation helper.

Before including larger limits in training, decide the intended domain and
check its new combined/boundary loads, including negative shear. Success on
these three rays does not approve combinations of their enlarged amplitudes.
Alternatively, the original box already contains measurable nonlinearity.
No final data split, sampling count or optimization budget has been frozen.
