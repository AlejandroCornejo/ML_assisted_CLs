# New manuscript — current periodic RVE and coupon

Expanded working draft v0.2, updated 2026-09-15. Written in English for a computational-mechanics
journal. This directory is separate from, and does not overwrite, the previous
Claude manuscript or any result/checkpoint.

Sections 5--6 now contain the two-material assessment skeleton. All reported
results still concern material A; blue draft tasks identify the pending
multicavity material B, data/response plots and fixed-versus-learned-feature
comparison. Material A's reduced-state representation is now in Section 6.
The existing probe and detailed curvature audits are preserved in the
separately compiled supplement, not discarded. See `REVISION_PLAN.md` for
the experimental sequence; no new simulations or training accompanied this
editorial integration.

## Deliverables

- `manuscript.tex`: master source; the substantial text is in `sections/`.
  Original title and author order restored; affiliation wording follows the
  current MAW–ECM manuscript, with its CIMNE note for S. Ares de Parga.
- `manuscript.pdf`: compiled 40-page working reading copy, with 43 cited references,
  nine figures and six tables, including visible pending-evidence notes.
- `supplementary.tex` / `supplementary.pdf`: standalone two-page supplement
  with the material-A probe figure, archived test/probe table and detailed
  rank-one curvature audit. Sections S1--S2 are referenced from the main text.
- `sections/introduction.tex`: motivation, PMOR/hyperreduction lineage,
  constitutive learning, prose-based literature positioning, precise
  contribution, and assessment scope. The former standalone Section 7 has
  been integrated here, not retained after the numerical examples.
- `sections/homogenization.tex`: revised Section 2, organized from the strain
  input through periodic equilibrium, effective outputs, and consistent
  differentiation to the common structural constitutive interface.
- `sections/reduced_micromechanics.tex`: revised Section 3, separating
  affine/nonlinear displacement reduction from fixed/adaptive integration,
  using `aresdeparga2026nonlinear` nomenclature. Input-informed coordinates
  and direct evaluation are attributed to MAW–ECM; implementation-specific
  training, stress extraction and tangent procedures are distinguished.
- `figures/`: nine reproducible figures, PDF vectors plus PNG previews.
  Figures 1 and 2 are native TikZ (`model_hierarchy.tex` and
  `cubature_matrices.tex`), compiled directly in the manuscript to inherit
  its text and math fonts; their PDF/PNG files are previews. Figure 2
  is monochrome and illustrates the raw integrand matrix, compressed
  fixed-ECM constraints, and statewise adaptive-weight systems, including
  their full-mesh targets. Its third panel expands the four first Piola
  components and normalization row for stress cubature.
- `sections/learned_laws.tex` and `sections/mechanical_requirements.tex`:
  Section 4 introduces direct stress regression, the unconstrained-energy
  Free baseline, and the motivation for a polyconvex energy before its
  explicit C1–C6 admissibility checklist, with the
  feature proof and both monotone cores preceding the complete-energy
  proposition. Nonnegative energy is a separate saved-parameter bound;
  numerical checks are distinguished from architectural guarantees.
- `audit/section4_selected_models_20260910.json`: rerun derivative,
  energy-bound and sampling checks for both frozen selected PANNs.
  This is an implementation audit, not a retraining or timing experiment.
- `tables/`: five generated numerical tables; reduction and architecture
  tables are in the section sources. The former literature table is archived.
- `evidence_manifest.json`: result inputs, hashes, exact measured times and
  recomputed field errors. No new timing experiment is implied.
- `../06_fe2/audit_maw_closed_cycle.py` and `maw_closed_cycle_audit.json`:
  new frozen-model cycle diagnostic. It re-solves HPROM–ANN at each Gauss
  point and checks quadrature refinement, reversal and a tighter equilibrium
  tolerance. Its nonzero net work is included in Figure 6 and Table 4;
  these diagnostic iteration counts are not structural timing-run counts.
- `../06_fe2/audit_other_hprom_closed_cycles.py` and
  `other_hprom_closed_cycle_audit.json`: complementary frozen-model tests
  for affine HPROM and D-HPROM–ANN on the same cycle. Both are included in
  Figure 6 and Table 4, with reversal checks and, for the iterative affine
  model, a tighter equilibrium tolerance. No timing runs are repeated.
- `REFERENCE_AUDIT.md`: checked claims, corrections, missing sources and the
  unfinished full-reading queue. **Not all supplied references have been read.**
- `READING_NOTES_v02.md`: completed reading of the eight new articles,
  Ciarlet's two supplied chapters, the 2026 latent-closure paper, Barnett 2023,
  As'ad 2022, Thakolkaran 2025 and Klein 2022; distinct claim-level checks
  also recorded.
- `NOMENCLATURE.md`: exact mapping from primary/secondary notation to the
  strain-informed coordinates and saved arrays.
- `audit/new_sources/rename_manifest.json`: nine PDF renames, original
  names, unchanged hashes and source metadata.
- `method_figures_manifest.json`: basis and coordinate diagnostics.
- `validation_report.json`: source consistency and artifact checks.
- `archive_v01/`: the immediately preceding manuscript source/PDF and its
  README/audit, retained for comparison. The older Claude manuscript is
  also untouched by this revision.
- `archive_pre_intro_integration/`: section sources and bibliography before
  moving the final positioning section into the introduction.
- `archive_pre_pann_prom_revision/`: sources before restoring the original
  title, author block, and PANN/PROM framing.
- `archive_pre_prose_revision/`: introduction, master source and comparison
  table before the argument-led literature rewrite and native LaTeX diagram.

## Narrative

1. Representing the anisotropic RVE response and the cost of repeated RVE
   evaluations; the polyconvex PANN construction as the central contribution;
   PROM-based models as complementary comparisons; structural deployment as
   an illustration. The literature positioning remains in the introduction.
2. General periodic homogenization and work-conjugate variables.
3. Affine and nonlinear PROMs; primary/secondary coordinates and closure
   fitting; fixed/adaptive ECM; actual reduced meshes; consistent derivatives
   and the direct operating mode.
4. Mechanical requirements, learned constitutive construction, proofs and
   distinction between the four neural model tiers.
5. Example I: the current RVE tests approximability, integrability and sampled
   rank-one curvature. Regression and all three deployed HPROM variants have nonzero cycle
   witnesses; the Free OOD
   witness is not described as an FOM-confirmed artificial instability.
6. Example II: the tensile coupon illustrates structural deployment and tests
   error propagation and online time for the same RVE. FOM–FE2 provides a
   computational reference within the adopted homogenization model, not an
   independent validation against a cell-resolved structure. This is neither
   a second material example nor a new FE2 method.
7. Conclusions. Literature positioning and assessment scope now precede the
   methodology, within Section 1.

The organization follows the principle in Joaquín's manuscript that each
example has a specific role. It does not copy his prose or import his results.

The introduction revision draws on the problem-to-gap progression in
`aresdeparga2026nonlinear`, the distinction between fitting and constitutive
structure in `asad2022` (Faisal As'ad), and the mechanism-led explanation and
purpose-specific examples of the MAW–ECM manuscript. It preserves the
primary/secondary coordinate terminology. This editorial change does not
alter the methods, numerical evidence, or measured timings.

## Reproduce numerical assets

From the project root:

Building the method-figure previews additionally requires Tectonic and
`pdftoppm`; set `TECTONIC` if the executable is not on PATH. The existing
temporary Tectonic installation is detected as a fallback. The manuscript
itself composes Figures 1 and 2 directly from TikZ and does not depend on
their previews.

```bash
MPLCONFIGDIR=/tmp/coupon-manuscript-mpl OPENBLAS_NUM_THREADS=1 \
  python3 coupon_fe2_paper/manuscript/build_evidence.py
MPLCONFIGDIR=/tmp/coupon-manuscript-mpl OPENBLAS_NUM_THREADS=1 \
  python3 coupon_fe2_paper/manuscript/build_method_figures.py
```

The scripts load NumPy/Matplotlib from the existing local `.pydeps`, leave
source results untouched, and overwrite only generated assets in this new
manuscript directory. The evidence script verifies common mesh hashes, load and convergence
metadata. Error norms are the declared unweighted stored-array norms, not
quadrature-weighted tensor norms. The POD script sums the discarded singular
values directly, avoiding cancellation from subtracting a cumulative sum
from one.

If citation order changes, reorder the existing bibliography entries without
changing their keys:

```bash
python3 coupon_fe2_paper/manuscript/order_bibliography.py
```

Compile from this directory with a standard LaTeX installation, e.g.:

```bash
latexmk -pdf manuscript.tex supplementary.tex
```

A portable Tectonic executable was placed outside the repository at
`/tmp/coupon-manuscript-tools/tectonic`; it may not survive system cleanup:

```bash
XDG_CACHE_HOME=/tmp/coupon-tectonic-cache \
  /tmp/coupon-manuscript-tools/tectonic --keep-logs manuscript.tex
XDG_CACHE_HOME=/tmp/coupon-tectonic-cache \
  /tmp/coupon-manuscript-tools/tectonic --keep-logs supplementary.tex
```

Initial compilation downloads TeX packages. The bibliography is included
from `references.tex`, so no BibTeX database is required. After compilation:

```bash
python3 coupon_fe2_paper/manuscript/validate_manuscript.py
```

The validator checks both documents. If PDFs and logs were compiled into
a temporary output directory, pass `--build-dir /absolute/path/to/build`.
`build_evidence.py` generates separate `constitutive_errors.tex` (main-paper
test results) and `constitutive_probe_errors.tex` (supplement); regenerating
assets will not restore probe columns to the main table.

## Remaining work before submission

The newly supplied texts have now been read, as precisely documented in the
audit. This expanded revision is **not the completion of the earlier request
to read the entire corpus**, nor a submission-ready manuscript.

- Finish the still-open primary-source readings and formulation checks.
  Boehler is unavailable and not cited; no present theorem depends on it.
- Confirm authors, affiliations and the citation status of the current MAW–ECM
  manuscript with the authors.
- Decide the second microstructure's scientific question before launching it.
  No second-material results have been invented.
- Complete matched-budget feature/normalization ablations and repeated seeds
  if making comparative architecture claims.
- Repeat FOM and all learned tiers under a common timing protocol; report
  identical-resource and tuned-resource comparisons separately. The draft
  retains the actual unequal repetition counts and thread configurations.
- Report offline costs and break-even counts if making offline-inclusive
  efficiency claims. Add weighted/invariant error metrics as supplementary
  quantities instead of silently changing the old ones.
- Resolve the Free extrapolative FOM reference before claiming false loss of
  stability. No forced failure is required to write an honest comparison.
- Strengthen scale-separation and mesh-convergence evidence if positioning
  the coupon as a physical validation rather than a fixed-discretization
  computational comparison.

No expensive simulation was started by the manuscript-generation script.

## Verification performed for this draft

- Numerical assets regenerated successfully from saved arrays and metadata;
  support unions checked as 183, 19, and 10 elements.
- All 43 bibliography entries are cited, keys and labels are unique, all
  referenced labels and figure files exist, and all nine renamed PDFs retain
  their original hashes.
- TeX compiled with no unresolved citations/references, overfull or underfull
  boxes. DOI links can break across lines; the bibliography uses one format.
- POD basis orthogonality and the coordinate-transform identity checked
  numerically. The truncation norm and coordinate-fit error are explicitly
  not presented as stress or structural error bounds.
- Representative PDF pages and figure previews inspected; reduced-support
  title overlap corrected in the figure generator.
- Introduction layout inspected after the prose revision: positioning is
  explained in paragraphs rather than a comparison table; Figure 1 is on
  page 5 and the methodology starts on page 6. The validator checks the
  positioning location, absence of the superseded table, native TikZ figure,
  PANN/PROM terminology, front matter, and final Conclusions section.
- A fresh attempt to run `test_flexible_pann.py` could not import PyTorch in
  the v0.1 environment. No new network unit-test pass is claimed for v0.2.
  The manuscript's derivative/energy checks refer to the archived independent
  checkpoint audit, whose hash is recorded, not that failed rerun attempt.
