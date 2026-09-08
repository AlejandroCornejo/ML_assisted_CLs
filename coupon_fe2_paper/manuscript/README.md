# New manuscript — current periodic RVE and coupon

Expanded working draft v0.2, 2026-09-07. Written in English for a computational-mechanics
journal. This directory is separate from, and does not overwrite, the previous
Claude manuscript or any result/checkpoint.

## Deliverables

- `manuscript.tex`: master source; the substantial text is in `sections/`.
  Original title and author order restored; affiliation wording follows the
  current MAW–ECM manuscript, with its CIMNE note for S. Ares de Parga.
- `manuscript.pdf`: compiled 28-page reading copy, with 42 cited references,
  eight figures and eight tables.
- `sections/introduction.tex`: motivation, PMOR/hyperreduction lineage,
  constitutive learning, literature positioning (including Table 1), precise
  contribution, and assessment scope. The former standalone Section 7 has
  been integrated here, not retained after the numerical examples.
- `sections/reduced_micromechanics.tex`: expanded derivation using
  `aresdeparga2026nonlinear` nomenclature, checked against the deployed code.
- `figures/`: eight reproducible figures, PDF vectors plus PNG previews.
- `tables/`: five generated numerical tables; reduction, architecture and
  literature-comparison tables are in the section sources.
- `evidence_manifest.json`: result inputs, hashes, exact measured times and
  recomputed field errors. No new timing experiment is implied.
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

## Narrative

1. FE2 cost and the need for surrogates; non-intrusive PANNs and intrusive
   projection-based reduced-order models; what each learns and retains;
   the feature-design question, contribution, and purpose and scope of the
   numerical examples. Memory savings are not reported as measured results.
2. General periodic homogenization and work-conjugate variables.
3. Affine and nonlinear PROMs; primary/secondary coordinates and closure
   fitting; fixed/adaptive ECM; actual reduced meshes; consistent derivatives
   and the direct operating mode.
4. Mechanical requirements, learned constitutive construction, proofs and
   distinction between the four neural model tiers.
5. Example I: the current RVE tests approximability, integrability and sampled
   rank-one curvature. Regression has a valid cycle witness; the Free OOD
   witness is not described as an FOM-confirmed artificial instability.
6. Example II: the coupon tests error propagation and online time for the same
   RVE. This is not represented as a second material example.
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
latexmk -pdf manuscript.tex
```

A portable Tectonic executable was placed outside the repository at
`/tmp/coupon-manuscript-tools/tectonic`; it may not survive system cleanup:

```bash
XDG_CACHE_HOME=/tmp/coupon-tectonic-cache \
  /tmp/coupon-manuscript-tools/tectonic --keep-logs manuscript.tex
```

Initial compilation downloads TeX packages. The bibliography is included
from `references.tex`, so no BibTeX database is required. After compilation:

```bash
python3 coupon_fe2_paper/manuscript/validate_manuscript.py
```

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
- All 42 bibliography entries are cited, keys and labels are unique, all
  referenced labels and figure files exist, and all nine renamed PDFs retain
  their original hashes.
- TeX compiled with no unresolved citations/references, overfull or underfull
  boxes. DOI links can break across lines; the bibliography uses one format.
- POD basis orthogonality and the coordinate-transform identity checked
  numerically. The truncation norm and coordinate-fit error are explicitly
  not presented as stress or structural error bounds.
- Representative PDF pages and figure previews inspected; reduced-support
  title overlap corrected in the figure generator.
- Introduction layout inspected after integration: the literature comparison
  is Table 1 on pages 4–5, the model hierarchy is on page 6 alongside the
  assessment scope, and the methodology starts on page 6. The validator checks that
  literature positioning remains inside the introduction and that the final
  main section is Conclusions, as well as the requested PANN/PROM terminology
  and restored front matter.
- A fresh attempt to run `test_flexible_pann.py` could not import PyTorch in
  the v0.1 environment. No new network unit-test pass is claimed for v0.2.
  The manuscript's derivative/energy checks refer to the archived independent
  checkpoint audit, whose hash is recorded, not that failed rerun attempt.
