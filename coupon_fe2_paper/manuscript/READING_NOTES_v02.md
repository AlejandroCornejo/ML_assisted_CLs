# Primary-source reading notes — v0.2

7 September 2026. These notes record what was read and what it changed.
They are not a certification of every proof or of the entire earlier corpus.

## Newly supplied texts

The complete extracted text of the eight articles below was read, including
their numerical sections, conclusions and supplied appendices. Ciarlet was
read through the OCR text of both supplied chapters, including the exercise
statements; the exercises were not solved. OCR is particularly unreliable for
subscripts, inequalities and matrix formulas. Key Ciarlet pages were also
inspected visually. No missing supplement is counted as read.

The rename manifest records original and new names and unchanged SHA-256
hashes. All nine new names equal the keys used in references.tex.

### GaoNeffRoventaThiel2017 — 6 pages

On the Convexity of Nonlinear Elastic Energies in the Right Cauchy–Green
Tensor, Journal of Elasticity 127 (2017), 303–308.
DOI: 10.1007/s10659-016-9601-6.

- Convexity in C is distinct from rank-one convexity and polyconvexity in F.
  This distinction is explicit in the paper, not a criticism invented for
  the present manuscript.
- Proposition 2.1 combines convexity in C with positive-definite second
  Piola stress at a smooth solution to obtain the stated global-minimization
  implication. The stress condition is essential.
- The manuscript therefore includes the full second variation:
  the strain-Hessian term plus S:(H-transpose H). It does not infer a
  globally positive F-tangent from a positive E-Hessian.
- The result is not an unconditional guarantee for compressive states,
  nor does it identify polyconvexity with strain convexity.

### Xu2021 — 25 pages

Learning constitutive relations using symmetric positive definite neural
networks, Journal of Computational Physics 428 (2021), 110072.
DOI: 10.1016/j.jcp.2020.110072.

- The construction is an incremental stress update with a learned
  lower-triangular factor and a product L L-transpose multiplying the strain
  increment. The applications include behavior outside pure hyperelasticity.
- This product is positive semidefinite; strict definiteness additionally
  requires nonsingularity of L.
- The product is not automatically a Hessian of a scalar potential.
  Differentiating the complete update when L depends on its state adds
  derivative terms. Its algebraic factorization must not be confused with
  the complete algorithmic constitutive tangent.
- The literature table describes the represented update and does not award
  it hyperelastic integrability or polyconvexity by inference.

### Tac2022 — 18 pages, main article

Data-driven tissue mechanics with polyconvex neural ordinary differential
equations, CMAME 398 (2022), 115248.
DOI: 10.1016/j.cma.2022.115248.

- Scalar ODE flows supply monotone functions used as invariant-energy
  derivatives. The invariant construction and sign conditions are necessary
  parts of the polyconvex argument.
- The worked tissue models use selected fiber invariants and incompressible
  kinematics. The paper also discusses compressible/generalized extensions;
  it would be incorrect to call the entire framework incompressible-only.
- Additive invariant terms and selected mixed terms supply integrable energy
  constructions, rather than arbitrary independently learned stress outputs.
- The separate supplementary material was not supplied or read. The
  manuscript does not rely on an equation available only there.

### Abdolazizi2025 — 33 pages

Constitutive Kolmogorov–Arnold Networks (CKANs): Combining accuracy and
interpretability in data-driven material modeling, JMPS 203 (2025), 106212.
DOI: 10.1016/j.jmps.2025.106212.

- This is the published JMPS article, not merely an unidentified preprint.
  It combines scalar-energy modeling, spline edges, reference correction,
  partial monotonicity, pruning and symbolic simplification.
- Its monotone Hermite construction in selected inputs does not impose
  input convexity of the whole energy core.
- Section 6 explicitly identifies polyconvexity as a possible further
  development. The paper must not be listed as an ICKAN/polyconvex model.
- Symbolic simplification is fitted with attention to constitutively
  relevant derivatives; a small value error alone is not the whole criterion.
- The new introduction and comparison table now distinguish CKAN from
  Thakolkaran's ICKAN and from our integrated-hat implementation.

### Ghaderi2020 — 20 pages

A Physics-Informed Assembly of Feed-Forward Neural Network Engines to
Predict Inelasticity in Cross-Linked Polymers, Polymers 12 (2020), 2628.
DOI: 10.3390/polym12112628.

- Directional microsphere integration and repeated learning-agent types
  reduce a tensorial constitutive problem to structured scalar components.
  The directional state includes maximum-history information.
- Consequently this is not the same history-independent input–output
  problem as our four constitutive tiers.
- The paper discusses weight-sign constraints and convexity in Appendix B.
  A sign condition by itself is insufficient for a generic activation:
  allowed sine/tanh choices require additional restrictions for a global
  monotone-convex composition argument.
- That last sentence is an independent mathematical qualification, not a
  quotation or a numerical falsification of all their examples. The table
  uses the safe description of their physical decomposition and does not
  adopt a general convexity theorem from weight signs alone.

### Gasser2006 — 21 pages

Hyperelastic modelling of arterial layers with distributed collagen fibre
orientations, Journal of the Royal Society Interface 3 (2006), 15–35.
DOI: 10.1098/rsif.2005.0073.

- Distributed fiber orientations lead to a generalized structural tensor,
  H = kappa I + (1 - 3 kappa) a tensor-product a in the stated axisymmetric
  setting. Its parameter range interpolates aligned and isotropic
  orientation distributions.
- Two fiber families and their dispersion change the deformation
  mechanisms predicted in strip loading. This supports the introduction's
  argument that the chosen directional representation matters.
- Appendix A's convexity discussion has specific incompressible,
  principal-stretch hypotheses. It is not a general polyconvexity theorem
  for arbitrary anisotropic energy networks.
- No arterial numerical results or fitted dispersion constants are
  transferred to the porous-cell example.

### Chmiel2024 — 28 pages

Assessment of Projection-Based Model Order Reduction for a Benchmark
Hypersonic Flow Problem, AIAA SciTech 2024, paper 2024-0250.
DOI: 10.2514/6.2024-0250.

- The paper compares affine, quadratic and ANN-augmented approximation
  spaces with LSPG and ECSW in a hypersonic double-cone benchmark.
- Its assessment demonstrates why a favorable global error alone can miss
  important local flow structures. This motivates complementing aggregate
  coupon errors with interpretable fields, not transferring a CFD
  robustness theorem into elasticity.
- The local examples and the choice of quantities of interest have a
  distinct purpose; they are not additional plots merely for breadth.
- Its timings and Mach-number results are not our evidence and are not
  copied into the coupon comparison.

### Miehe1999 — 32 pages

Computational homogenization analysis in finite plasticity simulation of
texture development in polycrystalline materials, CMAME 171 (1999), 387–418.
DOI: 10.1016/S0045-7825(98)00218-7.

- The paper develops a finite-deformation micro-to-macro setting with
  microscopic constitutive evolution, equilibrium, effective stress and
  effective tangent. It distinguishes uniform-deformation/Taylor,
  uniform-boundary-deformation and periodic conditions.
- Microfluctuations, averaged first Piola stress and the energetic
  micro–macro consistency condition support the general background used
  in the new homogenization section.
- The constitutive setting is finite crystal plasticity, not our
  Neo-Hookean porous material. The displayed structural applications use
  the Taylor option; they must not all be called nested periodic FE2
  simulations.
- Our specific periodic implementation and tangent are documented from
  the current code rather than attributed verbatim to this paper.

### Ciarlet1988 — supplied chapters 4 and 7, 125 scanned pages

Mathematical Elasticity, Volume I: Three-Dimensional Elasticity,
North-Holland, 1988.

- The file originally named balls.pdf is not Ball's 1977 article. It
  contains Chapter 4, pp.137–198 (62 pages), followed by Chapter 7,
  pp.345–407 (63 pages).
- Chapter 4 was read in full as supplied: hyperelasticity and virtual work;
  frame indifference and isotropy; natural-state expansion; growth;
  convexity; Ball's polyconvex construction; Ogden examples and exercises.
- Chapter 7 was read in full as supplied: weak convergence and lower
  semicontinuity; the direct method; weak limits of cofactors and
  determinants; existence; unilateral constraints; injectivity; the
  historical open-problem discussion and exercise statements.
- Key attribution locations: Section 4.9, pp.174–182, for the convex
  representation in deformation minors; Section 7.4, pp.359–361, for the
  outline of the direct method; Theorem 7.7-1, pp.371–377, for the existence
  argument and hypotheses.
- Visually checked PDF pages 1, 39 and 90, including printed pp.175 and
  372. The latter explicitly requires coercive growth, admissible boundary
  data, a nonempty admissible set, continuous loading functional and a
  finite-energy competitor. Polyconvexity is one hypothesis, not the whole
  theorem. The discussion permits nonuniqueness.
- Section 7.9 adds conditions for almost-everywhere injectivity; orientation
  preservation alone must not be advertised as global noninterpenetration.
  Section 7.10's historical open problems are not asserted to remain open
  in 2026.
- The complete book was not supplied or read. OCR text reading is not an
  independent line-by-line verification of every proof and scanned formula.
  Our two-dimensional feature proof is written explicitly in our manuscript;
  no three-dimensional existence theorem is silently applied to it.

## Nomenclature source read in full

The complete supplied 23-page text of aresdeparga2026nonlinear was read:
S. Ares De Parga, R. Tezaur, C.G. Hernández and C. Farhat, CMAME 448 (2026),
118443, DOI 10.1016/j.cma.2025.118443.

- Adopt HDM, PROM, HPROM, ROB; primary and secondary generalized coordinates;
  V, barred V, q, barred q, and the learned closure map.
- The general framework need not use the same total ROB dimension as a
  conventional affine PROM. Equality of the 39-dimensional spans is a
  property of this RVE deployment, not a universal definition.
- The source considers neural, Gaussian-process and radial-basis closure
  alternatives. Only the ANN alternative is deployed here.
- Its CFD LSPG equations are not substituted for our Galerkin virtual-work
  equations. The direct strain substitution is our additional operating
  mode, not an assertion about the source's online algorithm.
- Our independent coordinates are affine-retaining. The stored primary
  coordinates also have an invertible strain-informed scaling/rotation;
  they are not simply the first three POD coefficients.
- See NOMENCLATURE.md for the exact mapping to the deployed arrays.

## Further core papers read in full during the final check

### barnett2023 — complete supplied 20-page article

J. Barnett, C. Farhat and Y. Maday, Neural-network-augmented projection-based
model order reduction for mitigating the Kolmogorov barrier to reducibility,
Journal of Computational Physics 492 (2023), 112420.
DOI: 10.1016/j.jcp.2023.112420.

- Sections 3–4 give the primary/secondary decomposition, the learned
  reduced-coordinate closure, its tangent and the resulting LSPG equations.
  The network predicts secondary coordinates, not the full nodal state.
- The paper's nonlinear projection used to construct ECSW training data
  must not be confused with simply projecting a snapshot onto the primary
  basis. Our coordinate-target loss and deployed cubature training are
  described from our implementation, not presented as identical to theirs.
- The LSPG test space and residual objective are not imported into our
  Galerkin microscopic virtual-work formulation.
- Section 5.5 explicitly distinguishes mesh-size reduction from actual
  speedup. Section 5.7 studies the tradeoff between primary and secondary
  dimensions. Fewer solved coordinates or sampled elements are useful
  diagnostics, not substitutes for measured complete online wall time.
- Full numerical sections, conclusions and reference list were read;
  the reported Burgers speedups are not transferred to our coupon.

### asad2022 — complete supplied 22-page article

F. As'ad, P. Avery and C. Farhat, A mechanics-informed artificial neural
network approach in data-driven constitutive modeling, International Journal
for Numerical Methods in Engineering 123 (2022), 2738–2759.
DOI: 10.1002/nme.6957.

- Sections 2–3 distinguish componentwise stress regression, an energy
  potential, objectivity and input convexity in Green–Lagrange strain.
  Their reference correction is incorporated into training. Our Free model
  is a separate four-input implementation, not a literal replication of
  their architecture or training protocol.
- The energy argument concerns the stated hyperelastic and loading
  assumptions. Convexity in strain and the tensile-state result are not
  a global polyconvexity theorem in the deformation gradient.
- Sections 4.3–4.5 assign distinct jobs to rigid translation, a pressure-
  loaded cylinder and a prestressed dynamic membrane. The numerical
  failures support the individual demonstrations, not a theorem that
  every unconstrained network fails on every structural example.
- Section 4.6 deploys the learned woven-fabric response in supersonic
  parachute inflation. The flight comparison supports inflation and drag;
  the authors explicitly state that the available flight data do not
  validate their computed von Mises stress. This distinction matters when
  choosing visually impressive examples without overclaiming validation.
- The microscopic model is a plane-stress woven-fabric setting; its
  synthetic-data noise, splits, budgets and FSI machinery are not our
  porous-cell protocol. All supplied numerical sections and references
  were read, not only the mechanical-constraint discussion.

### thakolkaran2025 — complete supplied 30-page article

P. Thakolkaran et al., Can KAN CANs? Input-convex Kolmogorov-Arnold Networks
(KANs) as hyperelastic constitutive artificial neural networks (CANs),
CMAME 443 (2025), 118089. DOI: 10.1016/j.cma.2025.118089.

- Sections 2.2–2.3 construct isotropic compressible energy models from
  polyconvex invariant inputs and nondecreasing convex uniform B-splines.
  The derivative conditions at the reference follow from their chosen
  invariants. They are not our paired anisotropic features.
- Section 3 uses NN-EUCLID equilibrium residuals with displacement fields
  and aggregate reaction forces, without stress labels. Our constitutive
  fitting instead uses supervised microscopic stress labels.
- Section 4 distinguishes a training specimen with a hole, six homogeneous
  deformation paths and an unseen two-hole validation geometry. Training
  data include artificial displacement noise and prior spatial denoising.
  The lowest-loss member of ten initializations is used. These particulars
  matter when discussing generalization, not only the displayed curves.
- Appendix D's unconstrained comparison adds a SiLU bias term and does no
  additional hyperparameter tuning. It demonstrates failures of those
  tested models; it does not establish that unsupervised learning without
  polyconvexity is mathematically impossible for every model and dataset.
- Appendices A–F, training parameters, symbolic fitting and references were
  all read. The finite-secant tail qualification in REFERENCE_AUDIT.md
  remains an independent mathematical check of the printed formulas,
  not a claim about the trajectories visited in the published simulations.
- Appendix F preserves monotonicity and convexity during symbolic fitting
  with nonnegative input/output scales and an admissible function library.
  The fitting objective shown there uses activation values. This must not
  be confused with the derivative-aware CKAN symbolification procedure.

### klein2022 — complete supplied 25-page article

D.K. Klein, M. Fernández, R.J. Martin, P. Neff and O. Weeger, Polyconvex
anisotropic hyperelasticity with neural networks, JMPS 159 (2022), 104703.
DOI: 10.1016/j.jmps.2021.104703.

- Sections 3.1–3.2 distinguish an invariant-input monotone ICNN with exact
  objectivity from an ICNN in deformation minors with rotation-augmented
  training for approximate objectivity. The latter's tested inputs are
  (F, J); cofactors are allowed by the general construction but did not
  improve the reported lattice fits.
- Group averaging imposes the chosen finite symmetry group exactly.
  The transverse-isotropy example instead averages six rotations, so full
  continuous axial symmetry, like objectivity, is only approximated in
  that example of the deformation-gradient model.
- Section 4.3 and Figure 7 explicitly show weak shear approximation by
  the invariant model for the BCC cell. Figure 8 shows the improved
  deformation-gradient model. Section 5 then gives a successful small
  invariant model for a different, analytical transverse-isotropy target.
  This contrast is now connected explicitly to our feature-design question.
  It is not evidence that invariant inputs always fail.
- Reference stress is fitted approximately in these models, not enforced
  by the later affine-J normalization used here. The volumetric growth
  term is additional; Section 2 excludes a general coercivity analysis.
- The cubic-cell examples use energy and stress labels; the transverse-
  isotropy example uses stress labels alone. Training strategies, random
  observer tests and multiple initializations are distinguished in the text.
- Section 6 explicitly discusses interpretability, initialization dependence
  and extrapolation limitations. The appendix supplies sufficient convex
  composition conditions, not a converse proving that every architecture
  outside those sufficient conditions is nonpolyconvex.
- All supplied sections, appendix and references were read. No existence,
  universal-generalization or FE-deployment guarantee is inferred merely
  from their successful constitutive fits.

## Additional claim-level checks, not complete reading

The introduction's broader PMOR lineage was checked against the supplied
abstracts, introductions and relevant formulation passages of Amsallem2012,
Barnett2022, Lee2020, Farhat2014, Grimberg2021, Hernandez2014, An2008,
Hernandez2020, Hernandez2024, Bravo2024, AresDeParga2023 and Fritzen2016.
The contextual constitutive statements were likewise checked in
AsadFarhat2026, Aldakheel2023, Liu2022, Linka2023, Tac2024,
SchroderNeff2003 and Kalina2024. These entries remain open for complete
reading. The v0.1 claim-level table covers the other core citations.

The local Hernández–Ares de Parga–Rossi MAW–ECM manuscript was consulted
for the actual method, organization and distinct benchmark purposes.
Its complete-text reading is not marked finished by those inspections.
It is cited as a supplied manuscript pending author confirmation.

## Version corrections

- Linden2023: published JMPS 179 (2023), 105363; the inspected full text is
  arXiv:2302.02403v2. The bibliography records that distinction.
- Tac2024: the local PDF is January 2023 arXiv v1, with a different title.
  Published metadata were checked against the
  [author-hosted journal PDF](https://biomechanics.stanford.edu/paper/CMECH24.pdf):
  Computational Mechanics 73 (2024), 49–65. The complete published version
  has not been substituted for a reading of the local preprint.
- Kalina2024: the key is retained for compatibility with the local 2024
  preprint; the published article is CMAME **437**, not 433, (2025), 117725.
  Checked against the
  [publisher record](https://www.sciencedirect.com/science/article/pii/S0045782524009812)
  and the authors' TU Dresden record.
- Boehler1987 remains unavailable in complete form. The user has already
  reported that they cannot obtain it. It is not cited in v0.2 and does not
  block the explicit feature proof. Do not repeatedly request it unless a
  new, indispensable attribution is identified.
