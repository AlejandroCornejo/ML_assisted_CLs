# Reference audit — working record, not a completed literature review

Updated: 2026-09-07, expanded manuscript v0.2.

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

The new manuscript has 42 cited references. Boehler is not among them:
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
