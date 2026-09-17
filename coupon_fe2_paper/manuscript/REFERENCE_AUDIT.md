# Reference audit — working record, not a completed literature review

Updated: 2026-09-10, expanded manuscript v0.2.

## Definitions before the model comparison — 10 September 2026

- Moved Table 2 from the baseline introduction to Section 4.7, after the
  feature and core definitions, complete-energy proof, and separate
  nonnegative-energy bound. It now summarizes established material
  instead of introducing the 32 inputs and saved-parameter certificate.
  Float placement is constrained to keep it after the preceding material.
- Section 4.2 now introduces the direct-stress and energy baselines.
  Its prose explicitly defines Free's four objective inputs before using
  the composite energy map and explaining its reference correction.
  Removed a redundant explanation of the input dimension.
- Removed the premature 32-feature count from the motivation paragraph.
  The first such count in Section 4 now follows the feature equation and
  its constraints. The invariant-to-directional bridge remains intact.
- Kept all subsection numbers, C1–C6 identifiers, table separators and
  constitutive claims. Added source-order checks to the manuscript
  validator for feature/count/table order, bound/table order and
  Free-input/energy-map order.
- The PDF compiles and validation passes without warnings: 36 pages,
  43 references, nine figures and seven tables. Table 2 is now on p. 23,
  after the bound on p. 22. No numerical model or timing was changed.

## Criterion order, Table 2 and invariant-based feature lineage — 10 September 2026

- Added thin horizontal separators between every Table 2 row. Included
  the previously implicit C3 stress-symmetry row and labeled C1–C6 in
  order; the caption distinguishes permitted anisotropy from enforcement
  of a named material symmetry group.
- Preserved all criterion numbers. Reordered their explanatory paragraphs
  and the complete-energy proof to follow C1, C2–C3, C4, C5, C6. Removed
  an undefined forward use of the convex representation symbol introduced
  by this move. Equations and cross-references use automatic numbering.
- Added a bridge in Section 4.3 from the three classical 3-D principal
  invariants to the two independent in-plane principal invariants, then
  to the directional stretch invariant I4(d). Rewrote the paired-feature
  equation in equivalent invariant and norm forms; no model changed.
- Checked the prior `PANN_anisotropic_claude.tex` feature definition:
  its 15 entries used the trace, directional-power aggregates and their
  cofactor counterparts, J and J squared. The text describes continuity
  of the underlying measures, not equivalence of the two embeddings or
  a claim of representational completeness.
- Checked supplied Klein (2022), the invariant-based composition in
  Section 3.1, and Gasser–Ogden–Holzapfel (2006), Section 4's directional
  invariant discussion. Material directions in the current RVE model are
  descriptors, not assumed physical fibers. No unavailable Boehler source
  was added or claimed as read.
- Verified the 2-D trace/cofactor identities, determinant invariant and
  the equality of the two paired-feature expressions on admissible random
  deformations (seed 20260910, 1000 candidates). The maximum relative
  feature difference was 1.10e-15. This is an algebra/implementation check,
  not a new constitutive accuracy experiment.
- Visually inspected Table 2 (p. 19) and the invariant-to-feature bridge
  (pp. 19–20). The reading copy has 36 pages, 43 references, nine figures
  and seven tables. No training, meshes, result inputs or timings changed.

## Explicit separation of closure and weight ANNs — 10 September 2026

- Section 3.4.2 now introduces three distinct networks with separate
  parameters: the existing secondary-coordinate closure, a residual-weight
  ANN, and a stress-weight ANN. The two additional networks predict logits;
  a scaled softmax converts these to positive element multipliers with
  prescribed sums. Their output dimensions are the respective pruned
  support sizes, not the displacement-space dimension.
- The exposition now follows the offline order: closure training, support
  selection/pruning, then separate weight-network fitting. It identifies
  which parameters are optimized in the integration loss and which data
  remain fixed. The online paragraph distinguishes evaluation of the
  trained maps from retraining or pruning. Inputs are the current reduced
  coordinates; identifying them with the imposed strain is reserved for
  the direct operating mode.
- Figure 2 explicitly shows the post-pruning residual-weight ANN and
  identifies the separate stress-weight ANN, retaining monochrome styling.
  Equations (38)–(39), on p. 13, give the network roles and output maps.
- Checked `05_validation/maw_lab.py::fit_field`, the separate field fits
  in `sweep_phase2.py`, and separate residual/stress model loading in
  `deploy_maw_full.py`. Also checked the supplied Hernández et al.
  MAW–ECM discussion of post-pruning continuous weight representations.
  Pruning details remain attributed to that source; the constraint-
  preserving neural output map is identified as our implementation.
- Compilation and manuscript validation pass without warnings; no model,
  support, simulation, or timing artifact was changed.

## Monochrome stress example and Section 4 narrative — 10 September 2026

- Removed all color coding from the cubature diagram, including fills and
  colored constraints. Figure 2 remains native LaTeX/TikZ, with black
  text, rules and arrows on white. Added panel (c): the explicit five-row
  stress system, comprising first Piola components (11, 12, 21, 22) and
  element-count normalization. Matrix labels sit below the factors so
  that the displayed matrix-vector products are dimensionally unambiguous.
- Expanded Section 3.4 to define each element's integrated first Piola
  contribution and explain the four physical rows, separate stress
  multipliers, full-mesh target, and the corresponding fixed-ECM snapshot
  matrix. First Piola stress is not generally symmetric; conversion to
  the symmetric three-component second Piola output remains downstream.
- Rechecked `05_validation/hprom_full.py::build_stress_integrand`,
  `05_validation/build_full_integrand.py`, and `maw_lab.py::blocks/targets`.
  A read-only synthetic check (seed 41; 7 elements, 3 states, four tensor
  components; support [5, 1, 3]) verified the five-row system against the
  actual helpers, including nonsymmetric off-diagonal components, the
  full-mesh target, and statewise blocks of the fixed-ECM snapshot matrix.
- Rebuilt the opening of Section 4 around successive modeling decisions:
  direct stress prediction and its missing potential constraint; learning
  an energy to enforce constitutive integrability; then selecting energy
  shape and growth restrictions. Regression and Free are now defined
  before the admissibility checklist uses their names. Free means no
  convexity constraint, not absence of all mechanical structure.
- Re-read the supplied As'ad et al. (2022) text, Section 3.1 (p. 2743),
  Section 3.3 (strain convexity and its stress-dependent implications),
  and Section 4 (p. 2748, the three ANN variants). The source explicitly
  distinguishes standard regression, hyperelastic ANN, and convex
  hyperelastic ANN. Our Free baseline follows the potential-based
  principle; neither its name nor its exact implementation is attributed
  to that paper. Its convex-in-strain variant is not called polyconvex.
- Visually checked the revised diagram and Section 4 text. The manuscript
  compiles and validates without warnings: 35 pages, 43 references,
  nine figures and seven tables. Figure 2 is on p. 14; Section 4 begins
  on p. 16. No training, model, simulation result or timing was changed.

## Cubature matrix assembly made explicit — 10 September 2026

- Expanded Section 3.4 and added native-LaTeX Figure 2 (p. 14) to show
  element contributions, the element-by-state/component snapshot matrix,
  its integration-basis compression, and the fixed ECM fitting system.
  The second panel shows the separate per-state adaptive-weight system
  on a common selected support. Dimensions and row/column meanings are
  explicit. The target always comes from the full mesh, not a sum over
  the selected columns alone.
- Checked `05_validation/attic/ecm_residual.py`,
  `05_validation/attic/build_ecm_supports.py`, and
  `05_validation/hprom_full.py`: residual integrands are projected element
  forces; stress integrands are element contributions to the average first
  Piola tensor, ordered (11, 12, 21, 22). Physical quadrature measures are
  inside these contributions. Integration modes are distinct from
  displacement POD modes; the SVD description also covers their
  computation through the snapshot Gram matrix.
- Checked `RVE_NeoHookean_Homogenization/core/empirical_cubature_method.py`,
  `EmpiricalCubatureMethod.SetUp`: the last row of `G_cub` is the normalized
  component of the constant vector orthogonal to the retained integration
  basis, not a raw row of ones. The manuscript now states this construction
  and its nonzero-remainder condition. Exact reproduction implies the
  element-count normalization; this is not claimed to be exact physical
  volume integration on an unequal-volume mesh.
- Checked `05_validation/build_full_integrand.py` and
  `05_validation/maw_lab.py`, including `blocks` and `targets`: adaptive
  neural weight fitting uses uncompressed statewise integrands. Its last
  row is ones, and its last target entry is the full element count.
  Residual and stress supports and training systems remain separate.
- A read-only numerical assembly check called the actual `blocks`,
  `targets`, and `EmpiricalCubatureMethod.SetUp` helpers on synthetic
  contributions (seed 12; 9 elements, 5 states, 3 components; unsorted
  support [7, 1, 4]; 4 retained modes). It verified state/component
  flattening, selected columns, full-mesh targets, and the orthogonalized
  `G_cub` construction to floating-point precision. This checks the
  documented assembly conventions, not surrogate accuracy.
- Figure 2 uses the manuscript's own text and mathematics fonts; its
  vector PDF and PNG previews are generated by `build_method_figures.py`.
  Visually inspected both the preview and compiled manuscript page.
  The current reading copy has 35 pages, 43 references, nine figures and
  seven tables; compilation and manuscript validation pass without
  warnings. The cycle plot is now Figure 6; its table remains Table 4.
  Earlier dated entries below retain their historical numbering.
- No model, support, mesh, training result, simulation, or timing record
  was changed. This is an explanatory revision checked against the code.

## Introduction rebuilt around the modeling argument — 10 September 2026

- Rewrote the whole introduction in response to the user's criticism of a
  reference-by-reference narrative. Preserved all 43 citation keys and
  their bibliographic entries; reordered entries by first appearance.
  References support successive modeling decisions rather than each
  receiving a standalone summary.
- The PROM thread now follows two distinct costs: representing the solution
  set and evaluating its projected operators. The nonlinear closure reduces
  independent coordinates; hyperreduction is still needed for local assembly.
  The discussion then explains why a fixed integration rule may retain a
  global constraint even when the displacement representation is local or
  nonlinear, motivating SAW–ECM and MAW–ECM.
- Re-read the introduction of the supplied 2 September MAW–ECM source:
  `/home/sares/maw-ecm-paper/hernandez2026hyperreduction_2_sept_2026_JAHO/hernandez2026hyperreduction.tex`.
  Its motivating fixed-rule/global-span argument and its local-space to
  continuous-manifold connection informed the organization, not copied
  prose. Checked the RBF closure and weight-field sections of that source.
  The manuscript distinguishes its RBF realization from our ANN closure
  and neural weight predictors; adaptive cubature is not ANN-specific.
- Checked Grimberg et al. (2021), introductory pp. 1847–1848: the two
  hyperreduction families and the scope of the cited ECSW stability results.
  Did not adopt a blanket claim that DEIM/approximate-then-project methods
  are unstable or that ECM/adaptive cubature automatically preserves energy.
  Our ECM choice is motivated by direct approximation of reduced integrals
  and evaluation of selected FE contributions using the existing law.
- Checked the original ECM article (journal year 2017; online 2016), CECM
  (journal year 2024; online 2023), and SAW–ECM (2024) abstracts and relevant
  introductory passages. CECM relaxes spatial placement; SAW–ECM and
  MAW–ECM adapt weights to discrete subspaces and continuous latent states.
  They are not presented as successive versions of one linear chronology.
- The PANN thread now connects constitutive input information, potential
  structure, appropriate curvature variables, feature expressiveness, and
  admissible reference normalization. The core alternatives serve the
  question of where to place scalar approximation flexibility. The paired
  features emerge as the response to the identified design requirements.
- Reworked the assessment around the distinct purposes of the material
  example and structural deployment, retaining the same reference coverage,
  scope limitations and timing qualifications. No new scientific results,
  training, model changes, or timing measurements were introduced.

## Cross-section consistency after the Section 4 revision — 10 September 2026

- Aligned the abstract, introductory contribution/assessment paragraphs,
  Section 4 opening, and conclusions with the complete-energy proof.
  Core convexity alone is not presented as sufficient: componentwise
  monotonicity and convexity of the features in independent minors are
  explicit. Reference corrections and growth terms belong to the proof.
- The abstract now identifies growth as architectural and global
  nonnegative energy as a separate trained-parameter bound. The introduction
  distinguishes that bound from both the construction proof and sampled
  tests; polyconvexity is not presumed for every homogenized response.
- Preserved the main contribution as RVE constitutive PANN development,
  with PROMs as complementary alternatives and FE² as a deployment example.
  The transition into Section 4 contrasts microscopic state reconstruction
  with direct effective constitutive learning.
- Sections 2 and 3 already distinguish a common stress–tangent interface
  from constitutive guarantees; no further mathematical changes were needed.
  Numerical results, model parameters, and timing data are unchanged.

## Section 4: admissibility, construction proof and implementation audit — 10 September 2026

### Exposition and mathematical review

- Restored the C1–C6 organizing device from the older manuscript, without
  transferring its 15-feature formulas or unrestricted claims to this RVE.
  The criteria distinguish potential structure, objectivity, stress symmetry
  and directional dependence, reference normalization, 2-D polyconvexity,
  and three explicit growth limits. This is our presentation convention,
  not a universally prescribed six-point standard.
- Separated symmetric stress, material symmetry, and major tangent symmetry.
  Polyconvexity is a sufficient modeling restriction, not a necessary law
  for every homogenized material. The 2-D guarantee does not establish a
  general 3-D extension, exact material symmetry identification, arbitrary
  extrapolation accuracy, or structural uniqueness/Newton convergence.
- Reordered the argument: criteria and baselines; feature motivation and
  joint-convexity proposition; monotone ICNN and integrated-hat ICKAN;
  complete-energy proof; saved-parameter nonnegative-energy bound; tests.
- Re-derived the directional Hessian determinant in Appendix A:
  `a b (a-b-1) t^(2a-2) J^(-2b-2)`. Checked the endpoint case
  `a=1, b=0` and extension at zero; smoothness is only asserted on GL+(2),
  where a nonzero material direction cannot have zero deformed length.
- Re-derived the paired reference derivative, including the engineering
  shear convention: `rho_i (1,1,0)`. The `1/p_i,1/q_i` factors explain
  why unequal powers still give an isotropic reference derivative.
- The complete-energy proof now constructs a convex function of independent
  `(G,j)` explicitly. Its two composition inequalities identify the separate
  roles of feature convexity, core monotonicity, and core convexity.
  The affine determinant normalization preserves this proof for either sign
  of the learned reference pressure. Each analytic growth term has zero
  reference value and gradient.
- Rechecked the supporting-plane lower bound and Appendix B's per-feature
  determinant minimization. Global nonnegative energy remains a separate
  sufficient saved-parameter check; it is not inferred from zero energy and
  stress at the reference state.

### Source and implementation checks

- Read the previous manuscript's admissibility checklist and compared its
  organization with the current formulas. Rechecked primary-source passages
  in Amos et al. (2017), Section 3.1/Proposition 1; Klein et al. (2022),
  invariant-network composition; Linden et al. (2023), Sections 2.2 and
  3.2–3.3; and Thakolkaran et al. (2025), Appendix A.3, Eqs. A.15–A.18.
  These are claim-level checks, not a new claim of full-corpus reading.
- Compared the text with `FlexibleEnergy`, `PositiveICNN`,
  `PositiveSplineKAN`, `IntegratedHatSplineLayer`, and the independent
  lower-bound audit. Added the ICNN recurrence and the ICKAN's positive
  input skip to match the implementation. Feature centering/scaling are
  deformation-independent at frozen parameters, even with dynamic centering
  during training.
- Five existing tests in `test_flexible_pann.py` passed under
  `/home/sares/.venv-rom/bin/python`. They cover feature constraints and
  reference normalization, derivatives, analytic stress, spline joins/tails,
  and physical scaling/tangent symmetry. The system Python lacked PyTorch;
  the existing project environment was used without installing packages.
- Re-ran `audit_flexible.py` on the frozen selected ICNN and ICKAN, without
  revealing held-out labels or retraining. The recorded artifact is
  `audit/section4_selected_models_20260910.json`, with source/checkpoint hashes.
  Both first- and second-derivative checks pass. Both 6000-state energy
  samples are finite and contain no negative energy; these samples are not
  the nonnegativity proof. The sufficient interval-bound margins remain
  17.254466681464738 (ICNN) and 16.818882758222216 (ICKAN), in normalized units.
  Analytic-tail margins are also positive for both models.
- An additional read-only spot check compared the independent-minor feature
  formula with the deployed features, checked its 96 sampled feature
  Hessians per model, and verified energy objectivity and the first-Piola
  rotation rule. The maximum reference-feature derivative discrepancies
  were 6.66e-16 and 5.00e-16. This is a numerical consistency check, not an
  independent proof of global convexity.
- No model, training box, production constitutive evaluator, saved accuracy,
  or timing result was changed. The manuscript remains a draft with the
  outstanding literature/provenance items below.

## Completed HPROM closed-cycle comparison — 8 September 2026

- Added `06_fe2/audit_other_hprom_closed_cycles.py` and its frozen output
  `other_hprom_closed_cycle_audit.json`. They complement, without replacing,
  the preceding HPROM–ANN audit. Both use the original regression-selected
  rectangle, unchanged model checkpoints and 8, 16, 32, 64, 128 Gauss points
  per edge. Input/source hashes are checked by `build_evidence.py`.
- Affine HPROM: signed work at 128 points/edge is
  −1.2341570641146973 J/m³; reverse traversal gives +1.234157063998282.
  Tightening the displacement-increment tolerance from 1e-10 to 1e-12 gives
  −1.2341570637654513. Variation over quadrature orders is below 2e-9 J/m³.
- D-HPROM–ANN: signed work is +54.08150299359113 J/m³; reverse traversal
  gives −54.08150299359113. Variation over orders is below 2e-9 J/m³.
  No microscopic equilibrium is solved, so a solver-tolerance check would
  not be meaningful for this variant.
- Figure 5 and Table 4 now include all seven surrogates. Table 4 places
  models in rows and quadrature orders in columns to retain readable type.
  The separately archived FOM cycle remains the eight-point reference:
  no new FOM refinement study is implied.
- All three deployed reduced constitutive output maps have nonzero work
  on this path. This does not contradict the potential structure of the
  affine fixed-weight reduced residual: the effective stress has its own
  independently fitted cubature and is not defined by differentiating that
  residual's energy. This test alone does not isolate the cause of the
  defects, establish their importance on arbitrary paths, or establish a
  limitation of every HPROM/MAW–ECM construction.
- Production constitutive laws, supports, training and structural timing
  results are unchanged. These diagnostics are not new speed-up runs.

## Section 3 feedback: corrections and new cycle experiment — 8 September 2026

This entry supersedes the support-provenance statement in the initial
Section 3 audit below. The earlier review inspected a preliminary
fixed-support training workflow rather than following the deployed
checkpoints through their actual generation chain.

### Adaptive pruning versus graph coupling

- The stress field is loaded from `maw_phase2_sig.npz`; the residual field
  from `maw_res_long10.npz`. The latter's `res_10_z` array is exactly equal
  to `05_validation/attic/maw_phase2_res.npz`'s `res_10_z`. The long residual
  fit keeps the support and refits weights; it does not select a new support.
- `05_validation/sweep_phase2.py` calls `run_mawecm_pruning` on initial
  fixed-ECM candidates before neural field fitting. Thus saying that the
  deployed ten-element supports came directly from fixed ECM was wrong.
- The script sets `smooth_laplacian_all_iterations=True` and
  `alpha_smooth=1e4` and calls a graph builder, but discards its return. It
  passes neither `K_graph` nor `use_global_graph_2ndstage=True`.
- In the available `mawecm_pruning_claude.py`, the latter option defaults
  to false. Lines defining `use_global_graph`, `K_graph` and `phase2_mode`
  distinguish local active-set regularization from the global graph branch.
  `smooth_laplacian_all_iterations` forces phase 2 from the first removal;
  it does NOT override the global-graph option.
- Consequently the current callable pipeline uses adaptive pruning with
  local positivity enforcement, not global graph coupling. Its comments
  describe a graph-regularized intention that the call does not implement.
  An intermediate commentary repeating those comments was corrected after
  inspecting the callee.
- The checkpoints do not store historical stage/graph diagnostics or the
  source version used to generate them. Therefore historical graph coupling
  is not certified by this inspection. Do not silently enable graph coupling:
  that would require new supports, weight fitting and numerical assessments.
  This provenance issue must be resolved before submission if claims depend
  on specifically graph-regularized MAW–ECM.

### Representation and tolerances

- Removed the premature coupon reference from the POD-error discussion.
- Total independent displacement was retained to preserve load-correlated
  information for input-informed identification, not because fluctuation
  decoders would be incompatible with periodic lifting or the direct mode.
  The saved preliminary modal-fit diagnostic has maximum relative fit
  residuals across the first three modes of 0.126119 (fluctuation) and
  0.0663654 (total). This is not the later T_m inverse-identification metric.
  Its old Gram-matrix POD ranks are not imported as a fresh high-accuracy
  rank comparison.
- Even though Fbar(e)-I is nonlinear in e, it is symmetric in the adopted
  representative. The associated affine nodal fields span at most three
  fixed spatial fields. Nonlinearity in loading does not by itself imply
  many affine-field POD modes. Snapshot normalization and representation
  affect the reported rank; 39 is not an intrinsic material dimension.
- GPR/RBF closures are explicit alternatives, as described in the supplied
  aresdeparga2026 paper; ANN is the chosen common parametrization.
  Input-informed decoder construction is introduced in the MAW paper
  alongside, rather than as an obligatory part of, the cubature algorithm.
- `ecm_supports.npz` records residual/stress integrand-SVD tolerances
  1e-3/1e-5, ranks 134/72 and supports 135/73. The selection wrapper calls
  ECM with relative compressed-constraint tolerance 1e-6.
  MAW candidate generation uses integrand-SVD tolerance 1e-4 and the same
  ECM tolerance; its ten-element final support is a prescribed pruning target.
  These distinct controls now have symbols and values in the manuscript.

### Newton comparison and restricted online operations

- Affine fixed-weight HPROM has zero decoder-curvature and weight-derivative
  terms: its projected stiffness is the exact reduced residual Jacobian.
  HPROM–ANN uses modified Newton, omitting both terms from its iteration
  matrix but not from the differentiated constitutive residual.
- This asymmetry is now explicit in Sections 3 and 6. Archived FE2 records
  give macro iterations but not micro-Newton totals. The 2.23x ratio is an
  implemented-workflow comparison, not an isolated architecture benefit.
  The externally quoted 143-to-176 iteration change is not an experiment
  from these frozen timing records and is not inserted as our result.
- Analytic/automatic derivatives of the neural decoder and weights are
  possible. The code's use of finite differences does not justify claiming
  that no HPROM–ANN derivative exists in closed form. Affine state
  sensitivities require linear solves, not an explicit nonlinear solution.
- The optimized `_g_at`, stored support-local moment arms and
  `ReducedStressBatch.lifting_jacobian` evaluate lifting and derivatives
  on selected rows. No full displacement-sized lifting evaluation is
  charged per online query in these optimized paths.
- Table 1 now shows only the HDM reference and three deployed reduced
  evaluators; every state entry references its defining equation.

### New HPROM–ANN closed-cycle witness

Run `python3 -B coupon_fe2_paper/06_fe2/audit_maw_closed_cycle.py`.
The script reuses `rectangle_points` and the exact cycle geometry from the
frozen neural/FOM audit. It does not search for a more adverse HPROM path.
No production solver or model was edited. Continuation re-solves reduced
equilibrium at each point; the instrumented assembly count is one call per
modified-Newton iteration. Temporary reduced meshes are isolated in a
temporary directory and cleaned up after the diagnostic.

Signed work (J/m³), 8/16/32/64/128 Gauss points per edge:
247.774103567, 247.774103580, 247.774103517, 247.774103423,
247.774103299.
The reversed 128-point traversal gives -247.774103844.
At microscopic increment tolerance 1e-12, the forward result is
247.774103572. These controls support a nonconservativity witness for this
deployed map, not a claim that all MAW–ECM variants lack potentials.
Nor does the cycle isolate the contributions of decoder, residual weights
and independently fitted output weights.

At the default tolerance and 128 points per edge, the counter records 2416
iterations, including 202 for the first query's continuation from zero;
subsequent queries require 4–5 iterations (median 4). These counts are
specific to this instrumented material-path test, not the archived FE2 run.
No speedup or wall-clock benchmark is derived from this diagnostic.

`06_fe2/maw_closed_cycle_audit.json` records settings, source/checkpoint hashes,
work and counts. The evidence generator verifies those hashes and includes
the new curve and signed table column. Abstract, introduction scope and
conclusions acknowledge this result as well as Regression's cycle defect.
Existing timing and accuracy inputs are unchanged.

## Section 3: PROMs, hyperreduction and direct evaluation — 8 September 2026

Initial revision record: the support-provenance statement and table layout
below are superseded by the feedback audit above.

The section now distinguishes displacement-space reduction, integration
reduction, and bypassing reduced equilibrium. Primary/secondary closure and
Galerkin projection precede fixed/adaptive ECM; stress reconstruction,
constitutive differentiation and the direct mode have separate explanations.
The summary table includes full-mesh PROM–ANN, making clear that a nonlinear
decoder does not itself imply hyperreduction. Its last column now says
“Neural prediction”: “None” does not imply that POD/ECM require no data.

Targeted primary-source checks (not new full-reading claims):

- `aresdeparga2026nonlinear`, Section 2, pp. 3–5, Eqs. (1)–(7):
  HDM/PROM nomenclature, total/primary/secondary dimensions, latent closure,
  and retention of secondary information without secondary equilibrium
  unknowns. Its introduction explicitly allows the framework to use
  Galerkin projection; we do not identify that with its CFD LSPG equations.
- Current MAW–ECM source:
  `/home/sares/maw-ecm-paper/hernandez2026hyperreduction_2_sept_2026_JAHO/hernandez2026hyperreduction.tex`,
  input-informed-construction subsection and hyperelastic-example equations
  `eq:least_squares_master_coordinates` through
  `eq:input_aligned_coordinates`, plus `remark:noiterations`.
  The strain-informed rotation and direct evaluation option are now cited
  there explicitly, rather than left without attribution. The source's
  two-input identification error is not imported into our three-input case.
- Hernández et al. (2017), Section 4.4, pp. 700–701, Eqs. (52)–(54):
  compression of integration snapshots and discrete point/weight fitting.
  Our description distinguishes element multipliers from physical Gauss
  weights; a multiplier-sum constraint is not claimed to enforce exact
  physical volume on an unequal-volume mesh.
- Farhat–Chapman–Avery (2015), Section 4.1, pp. 1089–1091, particularly
  Eqs. (36)–(38): the fixed-weight reduced potential. The state-dependent
  product-rule distinction is derived explicitly in our section; no ECSW
  energy-conservation theorem is transferred to independent output cubature.

Implementation checks:

- `04_training/build_decoder_basis.py` and `train_nslave.py`: retained
  affine displacement convention, input-informed rotation and scaling,
  secondary complement within the original POD span, raw secondary MSE,
  and smooth tanh closure.
- `05_validation/maw_lab.py`, `train_maw_fields.py` and the support-fitting
  workflow: compressed-integrand ECM supports followed by neural weight
  fitting. This is explicitly distinguished from the source paper's
  graph-coupled pruning procedure. Softmax multiplies by full element count;
  residual and stress targets include an appended multiplier-sum row.
- `06_fe2/linear_hprom_fast.py`, `maw_hprom_ann_fast.py`,
  `direct_hprom_ann_fast.py` and `reduced_stress_batch.py`: different
  tangent evaluation paths, modified versus complete residual Jacobians,
  and symmetric extraction of the actual three-component stress output.
  The former blanket statement that all partial derivatives are evaluated
  by central differences has been corrected.
- `05_validation/reduced_mesh.py`: retained elements/nodes and row-
  restricted full-model operators, not independent remeshing or rebuilding
  periodic face constraints on the disconnected selected mesh.

Read-only numerical check against `decoder_basis_B_r39.npz`:
N = 6320, n = 3, secondary dimension = 36;
total-block orthonormality defect = 9.708e-15;
retained-span defect = 7.751e-15;
coordinate-transform identity defect = 1.624e-15.
The relative difference between fitted coordinates and prescribed strain is
3.605e-5, confirming that their equality is approximate, not kinematic.
These are basis/coordinate checks, not new surrogate-accuracy experiments.

The PDF was rebuilt and Section 3's pages visually inspected. No solver,
network weights, supports, meshes, numerical-result inputs or timing records
were modified. The full-reference-reading submission gate remains open.

## Section 2: reference RVE formulation — 8 September 2026

The section now explains the constitutive input, periodic equilibrium,
effective energy/stress, and equilibrium-dependent tangent before introducing
the structural interface. FE2 remains a deployment example, not the central
contribution. Section 1's DECM/CECM sentence now describes the formulation
directly, retaining the citation without the distancing possessive “their”.

Targeted checks for this revision, not new full-paper reading claims:

- Miehe–Schröder–Schotte (1999), pp. 393–394, Eqs. (19)–(29):
  affine-plus-periodic kinematics, periodic fluctuation conditions,
  work consistency, translation removal, and the weak equilibrium form.
  This supports the homogenization framework, not an attribution of our
  porous plane-strain example to that paper.
- `00_rve/periodic_fom.py`, `_g`: displacement jumps use `(Fbar-I)` and
  the anchor fixes the fluctuation, not the entire boundary displacement.
  `core/fom_solver_rve.py` in `RVE_NeoHookean_Homogenization`:
  `DeformationGradientFromGreenLagrange2D` uses the positive-definite square
  root of `I+2E`, rejecting nonpositive eigenvalues.
- `PeriodicRVE.homogenized_stress` and `homogenized_energy`: first Piola
  averaging precedes conversion to effective second Piola stress; the
  normalization uses total cell reference volume, including the pore.
- `PeriodicRVE.stress_and_tangent_consistent`: the converged-state
  sensitivity includes equilibrium relaxation. The implementation uses
  affine-retaining independent coordinates rather than Section 2's
  fluctuation coordinates; the change of coordinates remains explicit in
  Section 3. The lifting's second derivative is finite-differenced
  algebraically, without extra perturbed nonlinear RVE solves in that
  tangent routine.
- `06_fe2/test_periodic_tangent.py` provides the separate equilibrium-
  resolving finite-difference check. Its source was inspected; the
  numerical test was not rerun for this editorial revision.

No solver, model weights, training domain, numerical result, or timing
record was changed. Newton stationarity is not presented as global
minimization, nor is homogenized polyconvexity presumed. The PDF was rebuilt
and Section 2's layout inspected; source/citation/label checks pass.
This revision does not close the outstanding full-reading queue.

## Argument-led literature revision and figure typography — 8 September 2026

The abstract now introduces the appeal of the non-intrusive and intrusive
routes, explicitly identifies ECM hyperreduction, and distinguishes the
affine HPROM from the nonlinear-manifold HPROM–ANN with MAW–ECM.
The literature comparison table has been removed from the active manuscript;
its relevant distinctions are integrated into the argument in Sections
1.1–1.3. The former table and pre-revision sources remain in
`archive_pre_prose_revision/`.

Targeted source checks used for this revision (not additional full-paper
reading claims):

- Barnett et al. (2023), introduction, pp. 1–3: the Kolmogorov n-width concerns
  best linear approximation; its slow decay motivates local and nonlinear
  representations. No such slow-decay result is asserted for this RVE.
- Grimberg et al. (2021), introduction, pp. 1847–1848, and Ares De Parga et al.
  (2023), introduction, pp. 2–3: approximate-then-project versus
  project-then-approximate, including DEIM versus sampling-and-weighting.
- Hernández et al. (2024), abstract and Section 1, particularly p. 4: DECM
  denotes discrete candidate-point selection; CECM adjusts positions and
  weights. It is not a synonym for state-continuous weights.
- Bravo et al. (2024), abstract: SAW–ECM shares locations across subspaces
  while adapting their weights. The current MAW–ECM source, introductory
  subsections on adaptive weights and manifold-adaptive ECM, extends this
  idea to continuous latent coordinates. CECM and SAW–ECM are distinct
  directions, not a claimed chronological derivation from one another.
- Liu et al. (2022), description of history-to-history learning, and Ghaderi
  et al. (2020), history descriptors and maximum-stretch variables: the prose
  now explains how past loading enters the model. As'ad–Farhat (2026) is used
  separately to motivate the cost of generating data across several scales.
- Fritzen et al. (2016), summary, and Hernández et al. (2020), abstract:
  hardware acceleration complements reduction; domain decomposition is a
  distinct multiscale setting, not an interchangeable homogenization scheme.

One reference is added: `Chaturantabut2010`, S. Chaturantabut and D. C.
Sorensen, *Nonlinear model reduction via discrete empirical interpolation*,
SIAM J. Sci. Comput. 32(5), 2737–2764 (2010),
[publisher record and abstract](https://epubs.siam.org/doi/10.1137/090766498).
The publisher abstract and metadata were checked, not the full article.
The description of its place in the two hyperreduction families is supported
by the supplied Grimberg/Ares De Parga primary texts inspected above. No DEIM
error theorem or quantitative performance claim is imported. Full reading
of this newly cited article remains pending. The bibliography now has 43
cited works; existing entry contents are preserved and citation order updated.

Figure 1 is now a TikZ source compiled in the manuscript, inheriting its
LaTeX text/math fonts. The preview generator compiles the same figure source
with the same font packages. Numerical data and timing evidence are unchanged.

## Constitutive contribution and illustrative FE2 scope — 8 September 2026

Following the scope discussion with the user, the introduction now leads with
representing the anisotropic RVE response rather than accelerating FE2.
The paired-feature polyconvex PANN construction is the main contribution;
PROM-based surrogates provide complementary assessment, and the tensile
coupon illustrates structural deployment. The abstract is aligned with this
hierarchy. Title, authors, affiliations and all 42 references are retained.

The scope distinguishes errors relative to FOM–FE2 within the adopted model
from independent physical validation of first-order homogenization against a
cell-resolved structure. No such comparison, general robustness guarantee,
or applicability claim for microscopic buckling/localization is added.
The supplied conversation with Joaquín informed the editorial scope; it is
not treated as a published source or evidence for a new mathematical claim.
Methods, numerical results, figures and timing evidence are unchanged.
The preceding text is preserved in checkpoint `f933e32d`.

## PANN/PROM framing and author metadata — 7 September 2026

The title and author order are restored from
`RVE_NeoHookean_Homogenization/pann/anisotropic/PANN_anisotropic_claude.tex`:
*Physics-augmented neural networks and projection-based reduced-order models
for anisotropic hyperelasticity*; S. Ares de Parga, A. Cornejo,
J. A. Hernández, and R. Rossi.

Affiliation wording and the CIMNE note for S. Ares de Parga come from
`/home/sares/maw-ecm-paper/hernandez2026hyperreduction_2_sept_2026_JAHO/hernandez2026hyperreduction.tex`.
Both local MAW–ECM versions have the same author/affiliation block.
S. Ares de Parga is listed with CIMNE and Stanford; J. A. Hernández with
CIMNE and UPC–ESEIAAT; R. Rossi with CIMNE and UPC–DECA. A. Cornejo is
listed with CIMNE and UPC–DECA, combining the user's explicit CIMNE/UPC
instruction with his departmental association in the original PANN source.
No new corresponding-author designation or email is inferred.

The introduction now motivates surrogates through FE2 cost and distinguishes
non-intrusive constitutive approximation from intrusive PROM construction.
PANN denotes physics augmentation, not a uniform guarantee shared by all
architectures. ANN-enhanced PROMs remain intrusive, and the direct decoder
mode retains its intrusive construction even though it bypasses online
equilibrium iterations. This restores the original framing without restoring
its unsupported exclusivity/universality claims. All 42 references and their
entry contents are retained; the bibliography is reordered by first citation.
This revision does not add a full-corpus reading claim or a measured
memory-saving claim.

## Introduction integration — 7 September 2026

The former Section 7's literature positioning and limitations now appear in
the introduction (Sections 1.3 and 1.5); its comparison is Table 1. There is
no separate positioning section after the examples. All 42 references are
retained, with bibliography entries reordered by first citation.

The requested introductory models were revisited: Ares De Parga et al.
(`aresdeparga2026nonlinear`), Faisal As'ad et al. (`asad2022`), and Hernández
et al.'s current MAW–ECM manuscript (`hernandez2026`). Their argumentative
structure informs this revision; neither their results nor their prose are
transferred into the present work. The primary/secondary PROM terminology
is preserved. Scope restrictions, including the unresolved extrapolative
Free-model FOM reference and the actual timing protocol, remain explicit.
This editorial revision adds no claim of further full-corpus reading.

## What changed in this revision

The nine newly supplied references have been identified and renamed to their
exact manuscript keys. PDF bytes were preserved, verified with SHA-256:

| Key / new PDF stem | Supplied text | Reading completed in v0.2 |
|---|---|---|
| GaoNeffRoventaThiel2017 | 6-page article | Complete extracted text |
| Xu2021 | 25-page article | Complete extracted text |
| Tac2022 | 18-page main article | Complete extracted text; separate supplement not supplied |
| Abdolazizi2025 | 33-page article | Complete extracted text |
| Ghaderi2020 | 20-page article | Complete extracted text |
| Gasser2006 | 21-page article | Complete extracted text |
| Chmiel2024 | 28-page conference paper | Complete extracted text |
| Miehe1999 | 32-page article | Complete extracted text |
| Ciarlet1988 | Chapters 4 and 7, 125 scanned pages | Complete supplied chapter text through OCR; key definition/theorem pages checked visually |

The complete supplied texts of aresdeparga2026nonlinear (23 pages),
barnett2023 (20 pages), asad2022 (22 pages), thakolkaran2025 (30 pages)
and klein2022 (25 pages)
were also read, including their supplied numerical sections and appendices.
The nomenclature of aresdeparga2026nonlinear is now used in the expanded PROM section; the precise
mapping to our strain-informed coordinates is in
[NOMENCLATURE.md](NOMENCLATURE.md).

Detailed findings, locations, limitations and source-version corrections:
[READING_NOTES_v02.md](READING_NOTES_v02.md). Original-to-new filenames,
unchanged hashes and PDF metadata:
[rename manifest](audit/new_sources/rename_manifest.json).

The current manuscript has 43 cited references. Boehler is not among them:
the user has already reported that the complete chapter is unavailable,
and the present construction does not depend on its missing text.

## Important status distinction

The initial v0.1 inventory contained 47 PDFs, 1,657 pages and approximately
847,121 extracted words. That is a historical snapshot, not the current
directory count: there are now 55 PDFs. The old inventory and extracted
texts are retained for provenance, while the new-source manifest records
this revision's additions and renames.

Extraction is not reading. The newly supplied article/chapter texts listed
above have now been read, but the earlier request to read every reference
in the entire corpus remains unfinished. No abstract-only inspection is
recorded as a full-paper review. OCR reading does not mean independent
verification of every mathematical symbol. The manuscript is explicitly
labelled v0.2, not submission-ready.

The corpus inventory records original paths and SHA-256 hashes in
`audit/corpus_inventory.json`. Additional primary-source downloads, including
Ball, Amos and Linden, are separately recorded in
`audit/additional_sources/provenance.json`.

The original manuscript and the current Hernández MAW–ECM manuscript were
consulted for organization and scope. The former's bibliography is a search
inventory, not a source of verified claims. The latter's example structure
motivates separating general methodology from application-specific questions.
Its numerical results are not transferred into our tables.

## Claim-level checks used in v0.1

These locations identify passages actually inspected, not a claim that the
entire corresponding document has been read. Page numbers below are the
printed page numbers where given, otherwise PDF/preprint page numbers.

| Source | Inspected supporting location | Permitted statement | Excluded overstatement |
|---|---|---|---|
| As'ad, Avery & Farhat (2022), corpus 10 | §§3.1–3.3, pp.2742–2746; Theorems 1–2 and Eq.(13); introductory regression setup | Direct stress regression is not necessarily conservative; energy differentiation imposes a potential; Green-strain convexity has the stated tensile-state stability implication | Their model is globally polyconvex; strain convexity is strictly stronger than polyconvexity |
| Amos, Xu & Kolter (2017), downloaded primary paper | §3.1, Proposition 1 and Eq.(2) | Nonnegative hidden weights and convex nondecreasing activations give input convexity; direct-input weights may be unrestricted | Every ICNN is monotone in all inputs; training its parameters is a convex problem |
| Klein et al. (2022), corpus 30 | Abstract and introduction, distinguishing the two formulations | Invariant formulation is objective and polyconvex; minor/gradient formulation uses approximate objectivity by augmentation | Both variants enforce exact objectivity; failure of one invariant basis disproves polyconvexity |
| Linden et al. (2023), downloaded arXiv v2 | §2.2; §3.3, pp.11–14; Eqs.(30)–(49) | Affine-J isotropic normalization; compatible TI corrections; independent growth; nonnegative energy is a separate issue | Reference normalization alone guarantees global W≥0; affine-J correction originates in our work |
| Klein et al. (2026), corpus 31, arXiv v1 | §§2–3.3.5, pp.4–14; §§4.2.2–4.3.2, pp.19–22, particularly Eqs.(74)–(78) and final paragraph of §4.3.2 | Triclinic objective polyconvex feature formulation, optional finite-group symmetrization, possible use without known material symmetry | Method restricted to the cubic examples; our work is the first group-unspecified anisotropic polyconvex network |
| Thakolkaran et al. (2025), corpus 44 | §§2.2–2.3 pp.3–6; Appendix A.1–A.3 pp.21–23, Eqs.(A.15)–(A.18); full article subsequently read in v0.2 | Published monotonic ICKAN approach for polyconvex isotropic compressible hyperelasticity; interior spline constraints; finite-secant tail formulas as printed | Their exact spline construction is our integrated-hat implementation; finite secants give exact derivative-matched convex tails for every strictly convex spline; the tested unconstrained comparison proves universal failure |
| Kalina et al. (2023), corpus 39 | pp.827–828, abstract and introduction | FEANN combines physics-constrained invariant energy networks with automated homogenization data mining | This is a different paper solely because its filename is a DOI; its authors/date follow the filename rather than the paper |
| Yvonnet & He (2007), corpus 46 | pp.341–342, abstract and introduction | Reduced finite-strain multiscale method using POD and microscopic solves | Equivalent to the separate Yvonnet–Monteiro–He (2013) energy-database paper |
| Hernández, Caicedo & Ferrer (2017), corpus 25 | pp.687–688, abstract and opening formulation | Positive empirical cubature for projected nonlinear FE integrands; volume constraint | Original 2014 reduced homogenization paper and 2017 ECM are identical algorithms |
| Barnett, Farhat & Maday (2023), corpus 14 | p.1, abstract | PROM–ANN uses learned reduced-space closure and supports hyperreduction | Present direct constitutive closure is the identical online equilibrium algorithm |
| Farhat, Chapman & Avery (2015), corpus 20 | pp.1077–1078, summary and scope | ECSW preserves Lagrangian structure for the stated nonlinear dynamic setting | This theorem automatically covers arbitrary state-dependent weights or separately integrated homogenized stress |
| Hernández, Ares de Parga & Rossi (2026), supplied local manuscript | Section organization, formulation scope, nonlinear-manifold/adaptive-cubature discussion, benchmark rationale and conclusions | Established MAW–ECM machinery, distinct example purposes, need to differentiate adaptive weights | New ECM algorithm introduced by the present PANN manuscript; global constitutive guarantees inherited merely by naming MAW–ECM |
| Ball (1977), downloaded scan | Title/contents and foundational attribution checked; full proof-level reading pending | Historical attribution of polyconvexity framework; our 2-D construction is proved explicitly in the draft | Full existence theorem applies without checking coercivity, admissible spaces and boundary data |

Do not broaden a claim beyond its checked passage merely because another paper
cites the same source. Corrections to an existing source's wording or equations
must come from independent reasoning, not uncritical transcription. In
particular, the manuscript writes the full geometric term in the finite-strain
rank-one test and distinguishes semi-definite from strict conditions.

### ICKAN tail qualification: independent reasoning, not a quotation

The published Appendix A.3 uses a forward secant for the left tail and a
backward secant for the right tail, with finite epsilon. It describes C1
matching, but the printed formulas do not imply exact C1 or global convexity
across the joins for a general convex interior spline. For example, on an
interior interval beginning at x=0, f(x)=x² gives left-tail secant slope epsilon,
whereas the interior derivative at the join is zero: the derivative jumps
downward. An exact endpoint derivative, or an appropriate one-sided bound,
would avoid a convexity violation. This is a qualification of the finite-
secant extrapolation formula, not a claim that the interior spline proof is
invalid or that the paper's numerical examples necessarily visit those joins.
Our integrated-hat tails are analytical and have matching first and second
derivatives. The literature table now records this scope explicitly.

## Corpus anomalies and source availability

1. **Boehler (1987): incomplete.** The supplied file contains the actual first
   two chapter pages, pp.13–14, not the complete pp.13–30 chapter. Both available
   pages were inspected in v0.1. The user cannot obtain the complete chapter
   *Introduction to the invariant formulation of anisotropic constitutive
   equations* in *Applications of Tensor Functions in Solid Mechanics*.
   It is not cited in v0.2; no result depends on its missing pages.
2. `ICNN_ICKAN_and_KAN_for_Constitutive_Modelling.pdf` is an AI-generated
   secondary report. It is not an admissible primary source for a mechanical
   theorem, an attribution, or a literature-comparison checkbox.
3. `spencer1984constitutive.pdf` contains an approximately 290-page book, not
   just the constitutive-theory chapter. It must not be marked fully read after
   checking the opening chapter. `spencer1971theory.pdf` is a 115-page chapter.
4. `s00466-022-02260-0.pdf` is Kalina et al.'s FEANN paper, Computational
   Mechanics 71 (2023) 827–851. It is not an unidentified or separate reference.
5. `tac2024benchmarking.pdf` contains the January 2023 arXiv v1 of *Benchmarks
   for physics-informed data-driven hyperelasticity*. Version-specific claims
   need that version recorded; the filename does not establish its contents.
6. `Yvonnet.pdf` is the 2013 Yvonnet–Monteiro–He nonconcurrent energy-database
   work. The 2007 Yvonnet–He R3M paper is present in a different file.
7. The nine references previously requested as missing are now supplied,
   renamed and read as recorded above. In particular, balls.pdf contained
   Ciarlet's chapters 4 and 7, not Ball's 1977 article.

The bibliography distinguishes published metadata from the consulted
preprint for Linden, Tac's benchmark and Kalina's generalized-structure-tensor
paper. Klein (2026) remains the inspected arXiv v1. The local MAW–ECM
publication status is marked for author confirmation. No guessed DOI is added.

## Full-reading completion queue

The inventory is the exhaustive queue. Most documents still require full-text
reading; the claim-level table above does not close their entries. Prioritize:

1. Finish all methods, proofs, numerical sections and appendices of the papers
   directly compared: As'ad 2023/2026; Klein 2026; Linden; Tac's benchmark;
   Kalina; the actual current MAW–ECM version. As'ad 2022, Thakolkaran 2025
   Tac 2022 and Klein 2022 are now fully read as supplied.
2. Finish the projection/hyperreduction lineage and distinguish formulation
   hypotheses: Yvonnet, Hernández 2014/2017/2020/2024, Farhat 2014/2015,
   Barnett, Lee, Amsallem, Fritzen, Zahr, He, Patera/Yano, Grimberg, An,
   Bravo, Ares de Parga (2023), Romor, Guo, Koike, Rossi, Cornejo.
   The 2026 Ares De Parga latent-closure paper and Barnett 2023 are now
   fully read; Barnett's other paper remains open.
3. Complete foundational invariant/convexity references and broad-context
   sources: Ball, Spencer, Schröder–Neff, Bažant–Oh, Linka, Liu, Aldakheel,
   Boehler is excluded unless a future claim makes its complete text essential.
4. For each retained citation, verify author spelling, exact version, journal
   year versus online year, title, locator and DOI. Then check the final claim
   wording a second time against the primary source.

## Corrections to earlier interpretation of our own results

- A prior mechanics graphic called the entire in-box sample FOM-validated.
  Additional uniform in-box audit states do not have independently solved FOM
  labels. The new table labels them **unlabelled**.
- A prior cycle graphic retained an obsolete “16 points/edge” annotation.
  The new figure reads the complete 8/16/32/64/128 convergence record.
- The spline cycle residual is a quadrature error; it is not evidence of
  constitutive dissipation or loss of the potential structure.
- The endpoint-energy field in the original audit should not be treated as an
  independent check when generated by subtracting the same scalar. The paper
  uses the separately integrated stress work instead.
- FOM Newton nonconvergence near an extrapolative Free curvature candidate is
  a numerical outcome, not proof that the physical branch ends or is unstable.
- Optimized HPROM–ANN versus optimized HPROM is approximately 2.23×, not the
  older 3.88× ratio against an unoptimized linear implementation.
- “Polyconvex”, “exact potential”, “accurate”, and “Newton converges” are
  separate claims. The new manuscript does not collapse them into one column.

## Submission gate

Do not remove the working-draft label until the requested full reading is
complete, missing essential sources are resolved, all table claims are checked
at formulation level, author metadata are confirmed, and publication-final
experiment/timing choices are agreed. A compiled PDF is not this gate.
