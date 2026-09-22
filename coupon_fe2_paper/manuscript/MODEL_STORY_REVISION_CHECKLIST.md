# Model-story revision checklist

**Status:** active editorial plan
**Scope:** introduction, Sections 2--6, conclusions, supplement, and paper-facing figures/tables
**Working rule:** revise and approve one block at a time; do not perform a global rewrite.

## 1. Editorial objective

The paper will compare **learned hyperelastic energies**, rather than present a
ladder of increasingly constrained neural models. Its central message is:

> We construct learnable anisotropic paired features that retain polyconvexity,
> exact reference normalization, and growth; we assess whether learning these
> features improves the constrained models on a multidirectional microstructure,
> whether the resulting construction accurately represents an independent
> deployment microstructure, and how the corresponding energies perform in FE2.

The final evidence must answer five questions, in this order:

- [ ] Do learned paired features improve the guaranteed constructions over
      fixed admissible features on the multicavity microstructure?
- [ ] How sensitive is that conclusion to the number `m` of paired features?
- [ ] How accurately do the selected learned energies reproduce independent FOM
      responses for the single-cavity deployment microstructure?
- [ ] What accuracy is traded for the requested mechanical guarantees?
- [ ] How do those energies compare with the intrusive references in structural
      accuracy and online cost?

## 2. Decisions already closed

- [x] Remove **Pure Regression** as a model from the study.
- [x] Do not move Pure Regression to the supplement.
- [x] Remove its numerical results, table rows, plotted points, training details,
      cycle results, and conclusions from all paper-facing material.
- [x] Retain only the general theoretical motivation that independently fitted
      stress components need not derive from a scalar potential.
- [x] Keep one flexible energy-based reference to quantify the effect of the
      convexity constraints.
- [x] Keep intrusive reduced models as a distinct FE2 accuracy--cost route, not
      as members of the constitutive neural-model hierarchy.
- [x] Treat intrusive reduced models as secondary deployment references, not as
      a contribution with equal narrative weight to the paired-feature energies.
- [x] Add a controlled multicavity sensitivity study in the number of paired
      features, without using its test results to reselect the locked `m=32`
      models.
- [x] Remove the paper-facing identifiers **Material A** and **Material B**.
      Use **single-cavity microstructure** for the FE2 deployment case and
      **multicavity microstructure** for the controlled representation study.
      Internal filenames, scripts, and provenance may retain A/B identifiers.
- [x] Present the multicavity representation study before the single-cavity
      deployment qualification, so that the value of learned features is tested
      before the paper uses them in the structural case.
- [x] Preserve internal Regression code and archived results for provenance;
      removing the model from the paper does not authorize deleting evidence.
- [x] Replace the current model-hierarchy Figure 1 with an original visual
      summary centered on the paired-feature constitutive construction.
- [x] Prepare a simplified graphical-abstract export only after the retained
      numerical evidence is frozen; do not burden it with PROM or baseline
      taxonomy.

## 3. Final model vocabulary

Paper-facing names:

- **FOM:** mechanical reference.
- **Unconstrained energy:** flexible hyperelastic reference. The word
  *unconstrained* refers only to the convexity restrictions; this model remains
  energy-based, objective, and reference-normalized.
- **ICNN-fixed / ICNN-learned:** controlled feature comparison for the ICNN core.
- **ICKAN-fixed / ICKAN-learned:** controlled feature comparison for the ICKAN core.
- **HPROM and variants:** intrusive FE2 references, introduced only in their own
  methodological and structural-deployment context.

Paper-facing benchmark names:

- **Multicavity microstructure:** four-cavity, multidirectional representation
  benchmark used for fixed/learned ablation and feature-count sensitivity.
- **Single-cavity microstructure:** rotated single-cavity benchmark used for
  constitutive qualification and subsequent FE2 deployment.

Internal directories and checkpoint slugs may continue to use `Free`, `A`, and
`B`; only the paper-facing terminology changes.

## 4. Execution order

The final manuscript order will be:

1. Introduction.
2. Finite-strain homogenization and reference problem.
3. Learnable paired-feature polyconvex energies.
4. Compact projection-based structural reference.
5. Constitutive evidence: multicavity representation study followed by
   single-cavity deployment qualification.
6. Structural deployment of the single-cavity microstructure.
7. Conclusions.

The PROM/HPROM material remains a self-contained methods section because the
structural results compare several intrusive variants whose differences must be
understandable. Its position and length, however, will make its secondary role
unambiguous: the proposed constitutive construction comes first, and the PROM
section retains only what is needed to interpret the deployment study.

Editorial work will not follow this reader order mechanically. We first close
the constitutive construction already under review, then enact the section
swap, compress the PROM section, and finally return to the opening and evidence
sections. Each block ends with a compiled-PDF review and explicit approval
before the next block begins.

### Block 1 -- Section 3: constitutive construction

- [x] Remove the definition and implementation of Pure Regression.
- [x] Open the non-intrusive route with the decision to learn a scalar energy.
- [x] Introduce **Unconstrained energy** once, briefly and precisely.
- [x] Present ICNN and ICKAN as the two guaranteed constructions.
- [x] After the core-architecture tutorial review, add a compact paper-facing
      integrated-hat visualization in the established manuscript style. It
      should show the chain from localized nonnegative curvature to accumulated
      slope and the resulting convex connection, and may also show how several
      weighted hats combine. Decide in context whether this is clearer as one
      composite figure or two small figures; do not include both by default.
- [x] After closing the complete-energy subsection, design the method-centered replacement for
      Figure 1: periodic RVE response -> learnable paired directional features
      -> ICNN/ICKAN core -> normalized energy -> stress, tangent, and guarantees.
      The paired features must be the visual focus; PROM and comparison
      baselines do not belong in this overview.
- [x] Retain integrability as a mechanical requirement, not as an experimental
      contest against a stress-regression baseline.
- [x] Remove Pure Regression from the model-hierarchy figure.
- [x] Keep Unconstrained energy as a short textual reference rather than part of
      the proposed-method overview.
- [x] Check every forward reference from the constitutive construction to Sections 5 and 6.
- [x] Compile and inspect the affected pages.

**Approval criterion:** a reader can identify the proposed models, the sole
flexible reference, and the purpose of every comparison without reading the
results section.

### Block 2 -- Section 1: introduction

- [ ] Rebuild the introduction around one primary contribution: learnable
      paired directional features for guaranteed anisotropic energies.
- [ ] Motivate scalar energy learning without defining Pure Regression as a
      paper-facing model.
- [ ] Present ICNN and ICKAN as two cores implementing the same constitutive
      design, rather than as unrelated methods.
- [ ] Preview the distinct roles of the multicavity representation benchmark and
      the single-cavity deployment benchmark, in that logical order.
- [ ] Introduce the intrusive route only as a secondary FE2 accuracy--cost
      reference.
- [ ] Remove claims inherited from the previous co-equal PANN/PROM narrative.
- [ ] Compile and inspect the complete introduction in context.

**Approval criterion:** the introduction promises the paired-feature
constitutive contribution first and assigns every later numerical study a clear
role. It will receive a final consistency pass after Sections 5--6 are closed.

### Block 3 -- Section 2: reference problem and homogenization

- [ ] Verify that Section 2 remains focused on kinematics, homogenization, FOM
      quantities, and work-conjugate notation.
- [ ] Make no change where the current explanation already supports the revised
      story.
- [ ] Remove only obsolete taxonomy or forward references.
- [ ] Verify notation and cross-references after the constitutive-section changes.
- [ ] Compile and inspect Section 2 in context.

**Approval criterion:** Section 2 supplies the common mechanical reference
without anticipating either surrogate route unnecessarily.

### Block 4 -- Section 4: reduced microscopic reference

- [x] Recast projection-based models as secondary intrusive references for the
      structural deployment, not as a co-equal paper contribution.
- [x] Open the section with its limited purpose: define the intrusive references
      required later for the structural accuracy--cost comparison.
- [x] Explain, in one common framework, what each retained model reconstructs,
      which variables it solves for, and what stress and tangent it returns.
- [x] Keep the distinction between fixed and adaptive integration, and between
      iterative and direct deployment, because those distinctions are used in
      Section 6.
- [x] End with one compact comparison table for the retained intrusive models.
- [x] Retain in the body only the concepts and equations required to understand
      the deployed reduced models and their accuracy--cost comparison.
- [x] Move detailed PROM/HPROM, closure, cubature, and implementation derivations
      to the main appendices when they are not needed for the central argument.
- [x] In particular, consider moving the full POD/SVD derivation, detailed
      strain-coordinate rotation, manifold-Hessian terms, expanded cubature
      matrices, fitting losses, and implementation audits to the supplement.
- [x] Preserve enough information and citations for reproducibility after the
      reduction.
- [x] Remove artificial contrasts with Pure Regression.
- [x] Decide which intrusive models are genuinely needed in the final body.
- [x] Compile and inspect the shortened section together with its appendices.

**Approval criterion:** a reader understands the intrusive reference route, but
Section 4 no longer competes with the constitutive contribution for the center
of the paper.

### Block 5 -- Section 5: constitutive evidence

- [ ] Retitle and organize the section as a progression from representation
      evidence to deployment qualification, rather than by model count or the
      historical A/B identifiers.
- [ ] Remove **Material A** and **Material B** from headings, captions, tables,
      legends, and paper-facing prose; use **single-cavity microstructure** and
      **multicavity microstructure** consistently.
- [ ] Open with one compact benchmark-design subsection: state the common matrix
      law and porosity, show both geometries, document the two mesh checks, and
      explain that the result order follows scientific role rather than geometric
      simplicity.
- [ ] State the questions before giving training details or numerical results:
      first whether features should be learned, then whether the learned
      construction is accurate enough for the structural case.
- [ ] Remove every Pure Regression result and reference.
- [ ] Show FOM, ICNN, and ICKAN in the primary response plots; distinguish fixed
      and learned variants only where the controlled ablation requires it.
- [ ] Keep Unconstrained energy compact: preferably one separated reference row
      or a short numerical statement, not a dominant curve in every figure.

#### Block 5A -- Multicavity representation study

- [ ] Present the multicavity microstructure first and define its sampling domain,
      fitting/validation/test roles, and reserved loading paths before reporting
      any model comparison.
- [ ] State explicitly that the primary ablation is fixed versus learned features
      within the same core, under common data, objectives, widths, optimization
      rules, and evaluation states.
- [ ] Compare ICNN-fixed with ICNN-learned and ICKAN-fixed with ICKAN-learned at
      the locked `m=32` using seeds 16, 29, and 47.
- [ ] Use independent aggregate errors and predeclared axial, shear, and combined
      paths to determine whether any fixed/learned difference is reproducible and
      mechanically interpretable.
- [x] Before new training, freeze a multicavity feature-count protocol using only
      fitting and reference data to construct deterministic count-specific feature
      sets at `m = 8, 16, 24, 32, 40`; validation, test, and path labels remain
      closed during this step.
- [x] Verify that the `m=32` member inherits the existing locked feature table
      byte-for-byte; resolve any mismatch before launching the sensitivity campaign.
- [ ] Run ICNN-fixed, ICNN-learned, ICKAN-fixed, and ICKAN-learned for seeds
      16, 29, and 47 at every feature count: 4 cores/feature treatments x 5
      counts x 3 seeds = 60 fresh constrained fits. Do not include Free, whose
      architecture has no paired-feature count.
- [ ] Apply one prospective validation-plateau rule to all 60 fits, with 200000
      Adam steps as a safety cap rather than a required training length; retain
      the stop reason and final checkpoint hash for every fit.
- [ ] Keep data, core widths, objective, scheduler, stopping rule, and evaluation
      protocol fixed across `m`; report trainable parameter counts because the
      sweep is not parameter-matched.
- [ ] Report all declared feature counts as a sensitivity study; do not choose a
      new primary `m` from test outcomes.
- [ ] Summarize the sweep in one compact plot of error versus `m`, showing
      variation across seeds for fixed and learned features.
- [ ] If it remains legible and genuinely explanatory, add one compact view of
      learned orientations relative to the cavity geometry; do not treat visual
      alignment alone as evidence of physical identification.
- [ ] Conclude this study without using test results to select a different `m` or
      to select the independently trained single-cavity checkpoints.

#### Block 5B -- Single-cavity deployment qualification

- [ ] Introduce the single-cavity microstructure only after the representation
      study has established why feature learning is being examined.
- [ ] Explain its coupon-derived strain domain, Cartesian sampling, and
      fit/validation/independent-test split as preparation for FE2 rather than as
      a second ablation campaign.
- [ ] Evaluate the locked ICNN-learned and ICKAN-learned checkpoints against FOM
      on independent states and predeclared axial, shear, and combined paths.
- [ ] Use Unconstrained energy only to contextualize the cost of the guarantees;
      do not make beating it the objective of this study.
- [ ] Report the implementation and saved-checkpoint audits required for the
      models passed to Section 6: reference normalization, derivatives, the
      nonnegative-energy sufficient bound, and any retained finite diagnostic.
- [ ] End with an explicit handoff: the single-cavity models proceed to the
      structural deployment because their accuracy and domain coverage have now
      been assessed, not because the multicavity test selected them.

#### Block 5C -- Scope and evidence map

- [ ] State that the two microstructures are trained independently and do not
      demonstrate transfer between geometries.
- [ ] Separate analytical guarantees from finite implementation audits and from
      predictive accuracy.
- [ ] State in one closing synthesis what the multicavity study establishes, what
      the single-cavity study establishes, and what remains for Section 6.
- [ ] Do not claim architectural superiority from unmatched historical budgets.
- [ ] Check that every figure answers a stated question.
- [ ] Compile and inspect the complete Section 5.

**Approval criterion:** the reader encounters the evidence in causal order---the
value of learning the features is tested before learned features are deployed---
and can state the distinct purpose of each microstructure without relying on A/B
labels. No result suggests that the objective is to beat the unconstrained
reference in in-domain fitting error.

### Block 6 -- Section 6 and supplement: structural deployment

- [ ] Replace every paper-facing **Material A** label by **single-cavity
      microstructure** or a natural shortened reference to that benchmark.
- [ ] Remove Pure Regression from FE2 tables, plots, captions, and discussion.
- [ ] Retain Unconstrained energy as a visually secondary non-intrusive reference.
- [ ] Compare the guaranteed energies with FOM and only the intrusive reduced
      references needed to contextualize structural accuracy and online cost.
- [ ] Make clear that Section 6 evaluates deployment, not neural architecture.
- [ ] Compress PROM configuration and timing discussion to what is necessary for
      interpreting the retained comparison; move secondary diagnostics to the
      supplement.
- [ ] Remove the Regression row and related interpretation from Supplementary S3.
- [ ] Keep the closed-cycle note only if it remains useful as a brief diagnostic
      of the effective intrusive outputs.
- [ ] Describe the origin of that diagnostic honestly without reintroducing the
      deleted model into the paper's narrative.
- [ ] Regenerate every affected paper-facing table and figure from its source.
- [ ] Compile and inspect the complete Section 6 and supplement.

**Approval criterion:** Section 6 tells one story about structural accuracy and
cost; the intrusive and constitutive routes are not mixed conceptually.

### Block 7 -- Abstract, conclusions, and title

- [ ] Revisit the introduction after the evidence sections are frozen.
- [ ] Align the abstract with the evidence actually retained.
- [ ] Make every conclusion traceable to a table, figure, or construction proof.
- [ ] Remove claims inherited from the old progression-of-models narrative.
- [ ] Remove PROM from the title and reduce it in the abstract and keywords if
      the final body confirms its secondary role.
- [ ] Select a title only after the final methodological balance is visible.
- [ ] Derive the final separate graphical abstract from the approved method
      overview, simplify it to a left-to-right thumbnail-readable story, and
      export it at Elsevier's required aspect ratio and resolution.
- [ ] Compile and inspect the opening and closing pages together.

**Approval criterion:** the first page promises exactly the paper delivered by
Sections 3--6, and the conclusions claim no more than those sections establish.

### Block 8 -- Global audit

- [ ] Search paper-facing text, captions, tables, and legends for residual
      **Material A** and **Material B** identifiers; retain A/B only in internal
      paths, scripts, checkpoint metadata, and provenance where renaming would
      obscure reproducibility.
- [ ] Search manuscript, supplement, captions, tables, and figure labels for
      residual `Regression`, `Pure Regression`, and equivalent model names.
- [ ] Verify that any remaining discussion of componentwise stress fitting is
      general background, not a hidden evaluated model.
- [ ] Use **Unconstrained energy** consistently in all paper-facing text.
- [ ] Check that `Free` survives only in internal filenames, code, and provenance.
- [ ] Check all equation, section, table, figure, and supplementary references.
- [ ] Regenerate all paper-facing assets affected by the removal.
- [ ] Compile manuscript and supplement from a clean auxiliary state.
- [ ] Review the PDFs page by page for spacing, empty regions, captions, and
      orphaned explanations.
- [ ] Record the final editorial decision and evidence scope in `REVISION_PLAN.md`.

**Completion criterion:** the paper contains one coherent constitutive story,
no paper-facing Pure Regression model, and no dangling evidence or references.

## 5. Review protocol

For every block:

1. Inspect the current source and list only the necessary changes.
2. Apply the smallest coherent edit.
3. Show the user what changed and what was intentionally preserved.
4. Compile and inspect the affected pages.
5. Mark checklist items only after verification.
6. Wait for approval before starting the next block.

## 6. Current status

- [x] Editorial direction agreed.
- [x] Incremental execution agreed.
- [x] Checklist created.
- [x] Final manuscript order agreed in principle: constitutive construction
      before the compact intrusive reference.
- [x] New Figure 1 and the constitutive block approved in the compiled PDF.
- [x] Sections 3 and 4 swapped so that the constitutive contribution precedes
      the reduced microscopic reference; roadmap and cross-references updated.
- [x] Reordered Sections 3 and 4 approved in the compiled PDF.
- [x] Section 4 opening and constrained microscopic reference approved; HDM/FOM
      terminology and the placement of Figures 2--3 corrected.
- [x] Section 4.2 approved after its narrative revision.
- [x] Compact Section 4.3 approved in the main text; detailed
      strain-informed rotation, reconstruction split, and manifold tangent moved
      to Appendix D of the main manuscript.
- [x] Sections 4.4--4.7 compressed and approved; detailed cubature conventions
      and implementation derivatives retained in the appendix.
- [x] Section 4 now closes with a common constitutive-output framework and a
      compact comparison table for HDM/FOM, HPROM, HPROM--ANN, and D-HPROM--ANN.
- [x] New Figure 4 approved in the compiled PDF and referenced from the Section 4
      roadmap, the nonlinear-reduction discussion, and the constitutive-output
      subsection.
- [x] Block 4 approved: the intrusive route is complete but visually and
      narratively secondary to the paired-feature constitutive contribution.
- [x] Section 5 evidence order agreed in principle: multicavity representation
      study first, single-cavity deployment qualification second.
- [x] Paper-facing naming decision approved: replace A/B by descriptive
      microstructure names during Blocks 5--6; internal provenance may retain A/B.
