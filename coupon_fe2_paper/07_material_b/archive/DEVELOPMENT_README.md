# Archived development record: Material B

This was the evolving root README during geometry, mesh and data development.
It is retained for provenance, not as the current entry point. Relative links
below were written from the former root location and may therefore require
prefixing `../` when followed from this archive.

This directory is independent of material A's meshes, data and trained models.
The frozen constitutive FOM campaign is complete and passes its declared
numerical screens; neural training has not started. No PROM/HPROM or second
FE² deployment is performed here. See the
[campaign report and next gate](reports/CAMPAIGN_REPORT.md).
The [exact training recipe and shared 32-feature table](reports/TRAINING_PREPARATION.md)
are now fixed. All 15 initializations and six fixed/learned pairs pass their
implementation checks; the B-specific resumable optimization runner is next.

[Latest asymmetric-domain preflight](reports/ASYMMETRIC_BOX_REPORT.md): all 18
predeclared comparisons pass on 4,621 / 8,961 quadratic triangles when only
the positive normal limits are increased to 0.20. This finite result supports
using E11,E22 in [-0.04,0.20] and 2E12 in [-0.08,0.08] in the B data protocol.
The former mesh remains the working reference with campaign-level checks,
not an exact-error bound or full-box theorem. The data protocol is now frozen.
The [earlier pilot](reports/PILOT_REPORT.md) and its failed fine/finer sampled-peak
screen (5.0764% against 5%) remain unchanged.

The [subsequent expanded-box screen](reports/EXPANDED_BOX_REPORT.md) **fails**:
two compressed/sheared corners exceed mesh thresholds and one contour
screen fails. It remains rejected and is not superseded by silently removing
its difficult corners. The smaller asymmetric candidate above is a separate
predeclared test.

## Directory guide

- `reports/`: readable scientific reports, including unfavorable outcomes.
- `results/`: completed decisions, diagnostic JSONs and current figures.
- `tests/`: synthetic/exact implementation tests.
- `protocol/`: frozen design, restartable campaign writer and final assembler.
- `work/`: active scratch space; completed bounded smoke tests are archived.
- `archive/`: inactive attempts, superseded figures and diagnostics; see its
  [index and relocation record](archive/README.md).

Raw operational runs, scripts and frozen specifications remain at the root
because later stages and source hashes depend on their exact paths. No
scientific payload was deleted. The archive records original-to-current
paths and SHA-256 hashes; frozen JSON path strings were not rewritten.

## Candidate and purpose

[pilot_spec.json](pilot_spec.json) specifies a square periodic cell with four
distinct elliptical cavities. The matrix properties and plane-strain model are
inherited from the existing reference solver and checked at runtime:
E = 1.628 GPa, nu = 0.4.
Total void area is 20% of the full reference cell area. Effective stresses and
energies are normalized by that full area, including the cavities.

The center coordinates are divided by cell side L; each cavity's area fraction
is relative to the entire cell. Given area fraction f and aspect ratio k = a/b,
the semiaxes are b = L sqrt(f/(pi k)) and a = k b.
The differing areas, aspect ratios, angles and centers do not form a 2 x 2
repetition of material A. Parameters are specified directly, not optimized
against any learned model's errors. Four cavities are not inherently a harder
constitutive problem. This is a periodic material, not a statistical claim
about random porous media.

Geometry checks bound separation using enclosing circles for each ellipse and
nearby periodic images. This is a conservative geometric bound, not a sampled
estimate of the exact minimum ligament. Enclosed, disjoint cavities leave the
solid connected; OCC topology and the mesh graph are also checked.
Periodic boundary-node coordinates are compared after MDPA coordinate rounding.
Curved Tri6 Jacobians are sampled at 66 reference points per element;
this sampling does not prove positivity everywhere.

## What the pilot checks

Eight prescribed Green-strain rays, with four nonzero samples each, exercise
extension/compression in both axes, both shear signs and mixed states.
Engineering components are e = (E11, E22, 2 E12).
Axial strain rays do not impose zero transverse stress. Green-strain shear
does not describe a constant-area simple-shear loading.

Two periodic meshes are compared in effective stress, strain tangent and
microscopic first-Piola stress statistics. Three endpoints are checked using
two finite-difference steps for stress = energy gradient, the consistent
tangent, and symmetry of the unsymmetrized finite-difference tangent.
The existing consistent tangent is symmetrized internally; its symmetry is
therefore not treated as an independent numerical test.
Each endpoint is also reached independently from zero using finer continuation.
Residuals, microscopic determinants and endpoint fields are retained.
The thresholds are engineering screening criteria declared before the solves,
not mathematical certificates or final accuracy requirements.

The folder archive/pilot_v1 preserves the first run and its original specification
(density 200 for the paths, 400 for comparison). Coarse-mesh x compression
failed at E11 = -0.01. The file archive/diagnostics/compression_diagnostic_v1.json records cold-start
attempts: density 200 reproduced the failure, while 400 and 800 reached -0.04.
The current specification changes only continuation density to 400/800;
loads, geometry, meshes and tolerances remain unchanged. The folder pilot_v2
is the full repeat with that numerical refinement, not a replacement of the
failed run.

The original fine-mesh run also failed in y compression at E22 = -0.01.
The coarse/fine peak-stress difference at the first y-extension sample is
5.07%, just above the unchanged 5% threshold. This result is not rounded into
a pass. An additional fine/finer check repeats all 32 targets on a third mesh;
it does not replace the original coarse/fine screening.
The complete repeated pilot gives a worst coarse/fine peak difference of 6.21%.
On the finer mesh, y compression also required a continuation retry:
the unchanged path at densities 800 and 1600 reached all targets.
Both original failed runs and both targeted retries are retained.

The separate saved-field audit checks exact periodic displacement jumps
numerically and screens deformed cavity polygons for overlap/self-intersection.
These polygons connect boundary nodes by straight segments; they do not
certify the curved boundary or intermediate continuation states.
It also computes stress-dependent rank-one curvatures at 72 sampled directions
per state, not a polyconvexity certificate.
The reference verification compares the zero-strain tangent with an independent
linear assembly, and finite-strain forces/stiffness with native Kratos elements.
Matching-quadrature agreement and six-point quadrature sensitivity are reported
separately. The partial reference-verification file records a premature audit
invocation before the fine endpoint existed; use the complete verification.

## Reproduction

The current host uses Python 3.12, the project .pydeps directory, Gmsh 4.15.2
installed in .venv_fe2, and the existing Kratos release found by periodic_fom.
From the repository root, use a **new** output directory:

    PYTHONPATH=coupon_fe2_paper/.pydeps coupon_fe2_paper/.venv_fe2/bin/python \
      coupon_fe2_paper/07_material_b/run_pilot.py \
      --out coupon_fe2_paper/07_material_b/pilot_new

Add --mesh-only to generate and check meshes without solving.
Reports include the effective specification, SHA-256 hashes of the mesh and
principal source/material files, failure records and numerical diagnostics.
Every output directory also contains a frozen copy of its input specification.
Plots are FOM pilot evidence, not learned-model comparisons.
The repeated pilot returns exit code 1 because its declared mesh screen fails;
this is a recorded outcome, not an unhandled execution error.
Pinned package versions are in requirements.txt; the principal files used for
pilot_v2 are preserved in its source_snapshot folder.

## Limits and next decision

The rays do not establish reliability throughout their bounding strain box.
Successful Newton solves, positive sampled microscopic determinants and
positive strain-tangent eigenvalues do not prove rank-one stability, absence
of contact, uniqueness of microscopic equilibria or validity beyond these
states. No microscopic contact model is introduced.
These mesh comparisons assess sensitivity; they are not an asymptotic convergence study.
These checks informed the frozen data-domain choice. Model roles, seeds,
test partitions and maximum optimization budgets are predeclared; the exact
optimizer/learning-rate recipe has now been completed before neural training
in [TRAINING_RECIPE.md](protocol/TRAINING_RECIPE.md). These added numerical
details were not all frozen before FOM generation.
The subsequent preflight extends these checks to a fourth mesh and box-boundary
states; see its separate report above for the current working-reference decision.

## Reference/domain preflight

The subsequent test specification is frozen in
[preflight_spec.json](preflight_spec.json). It adds a fourth mesh and 18
corner, face, biaxial and mixed-normal states to the original 32 targets.
The homogenized-output comparison targets are 0.1% in stress-vector norm,
strain-tangent matrix norm and energy. The original microscopic-statistic
thresholds remain 2% for the weighted norm and 5% for the sampled maximum.
These are finite mesh-sensitivity screens, not exact-solution error bounds.

[adaptive_continuation.py](adaptive_continuation.py) changes only B's Newton
initial guesses and continuation increments. The independent unknowns are
total displacements, so the predictor adds the affine displacement change at
every independent coordinate, rather than updating the boundary lift alone.
Failed increments are retained and halved; the last successful state is kept.
Neither the shared solver's equilibrium equations nor its material law,
constraints or convergence tolerances are changed.

Run the stages into **new** directories, validating the predictor first:

    PYTHONPATH=coupon_fe2_paper/.pydeps coupon_fe2_paper/.venv_fe2/bin/python \
      coupon_fe2_paper/07_material_b/run_preflight.py --stage validation \
      --out coupon_fe2_paper/07_material_b/work/preflight_validation_new

    PYTHONPATH=coupon_fe2_paper/.pydeps coupon_fe2_paper/.venv_fe2/bin/python \
      coupon_fe2_paper/07_material_b/run_preflight.py --stage reference \
      --validation coupon_fe2_paper/07_material_b/work/preflight_validation_new \
      --out coupon_fe2_paper/07_material_b/work/preflight_reference_new

    PYTHONPATH=coupon_fe2_paper/.pydeps coupon_fe2_paper/.venv_fe2/bin/python \
      coupon_fe2_paper/07_material_b/run_preflight.py --stage check \
      --validation coupon_fe2_paper/07_material_b/work/preflight_validation_new \
      --out coupon_fe2_paper/07_material_b/work/preflight_check_new

The last two stages can run independently after validation. Wait for both to
complete before consolidating; a successful stage exit means its numerical
targets were reached, not that the mesh-comparison screen passed:

    PYTHONPATH=coupon_fe2_paper/.pydeps coupon_fe2_paper/.venv_fe2/bin/python \
      coupon_fe2_paper/07_material_b/summarize_preflight.py \
      --reference coupon_fe2_paper/07_material_b/work/preflight_reference_new \
      --check coupon_fe2_paper/07_material_b/work/preflight_check_new \
      --out coupon_fe2_paper/07_material_b/results/preflight_decision_new.json

The continuation unit tests use a mock solver to test step-halving and failure
retention only. Physical verification is the separate FOM comparison against
the earlier dense-ramp solutions, including microscopic node displacements.
The synthetic summary tests reject incomplete stages, missing/duplicated
targets and mismatched strains, and check the mesh-difference denominator.

Run all implementation tests from the repository root:

    PYTHONPATH=coupon_fe2_paper/.pydeps coupon_fe2_paper/.venv_fe2/bin/python \
      -m unittest discover -s coupon_fe2_paper/07_material_b/tests -p 'test_*.py' -v

## Physical response plots

[Saved-field response figure and definitions](results/physical_response_v2/README.md)
show signed axial stress, the equivalent von Mises of the homogenized Cauchy
tensor, and strain energy on the original eight rays. Only saved FOM states
are used; connecting lines are guides and the origin is analytic. The
plane-strain out-of-plane stress is recovered using the underlying matrix's
3D Neo-Hookean law, not a 3D extension of the learned in-plane energy.
The script plot_physical_response.py does not import or run the FOM solver.

[Larger-strain exploration and linearization comparison](reports/NONLINEAR_EXPLORATION.md)
add six predeclared targets on the same two meshes. Both the original and
extended rays show measurable nonlinear stress response relative to the
same-mesh zero-strain tangent. This is not approval of an enlarged training
box. The subsequent combined/boundary screen is reported below.

## Enlarged-box decision and reference-linearity diagnostics

[EXPANDED_BOX_REPORT.md](reports/EXPANDED_BOX_REPORT.md) records 20 cross-mesh pairs,
36 new targets, four reused targets, 12 FD checks and four independent
zero-start checks. Targets were reached but the complete enlarged box fails
its unchanged screens. No failures were discarded or rounded into passes.
The additional contour/FE-map audit does not prove contact or instability.

The driver uses the same cached meshes and verified saved-state provenance:

    PYTHONPATH=coupon_fe2_paper/.pydeps coupon_fe2_paper/.venv_fe2/bin/python \
      coupon_fe2_paper/07_material_b/run_box_extension.py --mesh reference \
      --out coupon_fe2_paper/07_material_b/work/expanded_reference_new

Use --mesh check with a separate new output for the denser stage; --plan-only
checks its inputs without solving. Wait for both stages before running
summarize_box_extension.py with --reference, --check and a new --out JSON.
audit_expanded_fields.py accepts the same stage paths and a new output folder.

audit_reference_linearity.py and audit_constant_tangent.py read saved A/B
data only. They write new diagnostic JSONs, never A artifacts or neural
checkpoints. The latter uses only A's existing fit partition for coefficients;
validation/test rows are evaluated without selection. It is an unconstrained
linear stress map, not the neural Regression baseline or an energy model.
Together with FOM curves and tangent changes, these diagnostics support a
precise account of nonlinearity without forcing unsupported load combinations.

## Asymmetric-domain decision and next operation

[ASYMMETRIC_BOX_REPORT.md](reports/ASYMMETRIC_BOX_REPORT.md) records the
successful normal-extension-only candidate: E11/E22 in [-0.04, 0.20] and
2 E12 in [-0.08, 0.08]. Eighteen cross-mesh pairs, 12 selected FD checks and
four independent-start checks pass unchanged tolerances. This is finite
evidence for choosing the sampling limits, not validation of the continuous
box. The failed ±0.16 shear candidate remains rejected.

The B data and comparison protocol is now frozen in
[protocol/DATA_PROTOCOL.md](protocol/DATA_PROTOCOL.md), with its machine-readable
[specification](protocol/data_protocol_v1.json). The deterministic
[coordinate design](results/data_protocol_design_v1.json) contains 4,200 fit,
512 validation and 512 untouched test states, plus ten independent paths and
predeclared mesh/cold-start audits. These files contain strain coordinates and
hashes only. Full-campaign raw labels are complete under
`results/data_campaign_v1/`; the approved deterministic assembly is
[data_labels_v1.npz](results/data_labels_v1.npz), with its
[20-check approval record](results/data_labels_v1.json). No neural results exist yet.

The bounded writer tests passed for fit (without tangent), test (with an
intentional pause/resume), both audit meshes and an independent cold start.
The fit smoke identified an absent-tangent JSON serialization error; its
failed record is retained in `archive/smoke_writer_failures/`. The corrected
writer stores absent tangents as `NaN` in NPZ and `null` in JSON, not as zero.
All requested states also pass residual, microscopic-determinant,
periodic-jump and deformed-cavity-polygon checks before acceptance.

The campaign used four independent processes and 103 restartable chunks.
All 5,777 requested solutions and all 20 assembly checks passed, including
the frozen 64-state mesh audit and 24-state warm/cold agreement checks.
Independent reassembly reproduced the assembled NPZ bit for bit; this is
not a fresh-solve reproducibility claim. Freeze the exact training recipe and
fit-only feature selection next; do not open reserved prediction curves for
model selection or start a second FE² deployment.
