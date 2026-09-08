# Revision v0.2 — 7 September 2026

Requested scope: identify and read the newly supplied primary references;
rename their PDFs to bibliographic keys; substantially rebuild the manuscript,
especially its introduction and projection-based methodology, using the
nomenclature of Ares De Parga et al. (2026).

## Work sequence

1. Identify each new PDF by its contents, record its version and hash, and
   rename without overwriting another file. Preserve an explicit undo map.
2. Read the new primary texts; distinguish complete reading, targeted reading,
   and any OCR/access limitation. Boehler remains incomplete and is not needed
   for an unsupported theorem attribution.
3. Recover the 2026 latent-closure notation and check the actual periodic
   implementation: HDM, PROM, HPROM; N, N_s, V_tot, V, barred V, q, barred q,
   n, barred n, n_tot, n_tra, and the closure map N. Explain deviations forced
   by the strain-driven RVE lift explicitly rather than asserting an orthogonal
   POD decomposition that the implementation does not use.
4. Rewrite the introduction as a connected argument through homogenization,
   affine and nonlinear PMOR, empirical cubature, constitutive learning, and
   the precise contribution. Restore relevant primary literature, not a target
   citation count. Keep application geometry in the numerical examples.
5. Develop the projection and hyperreduction equations, consistent derivatives,
   direct versus equilibrated closure, and online complexity. Strengthen the
   constitutive comparison and separate properties from sampled evidence.
6. Preserve result/checkpoint files and the former manuscript. Compile the new
   reading copy, check references and layout, and document unresolved tasks.

No new FE2 timing or training campaign is part of this editorial revision.
No second-RVE evidence will be invented. The manuscript remains a working
draft until the full-source and experimental submission gates are satisfied.

## Delivery record

- Nine newly supplied PDFs renamed and hash-verified. Eight article texts
  and both supplied Ciarlet chapters read; the full 2026 nomenclature source
  read as well. Barnett 2023, As'ad 2022, Thakolkaran 2025 and Klein 2022
  subsequently read in full, with further formulation-specific refinements.
  The full-book and full-earlier-corpus limitations are explicit.
- Master source split into maintained section files. Introduction rebuilt,
  PMOR formulation expanded, actual closure/weight training described, and
  mechanical requirements separated from numerical observations.
- Eight figures and eight tables, including two new method/POD figures.
  Bibliography expanded to 42 relevant cited works and ordered by first
  appearance. Boehler excluded; source versions recorded.
- Preceding v0.1 archived; no training, FE2 result, timing or checkpoint
  modified. Validation and reading notes accompany the 27-page PDF.

## Introduction integration — completed 7 September 2026

The user requested that the former Section 7, "Position relative to prior
work and limitations", be part of the introduction. That standalone section
has been removed; its formulation-specific comparison is now Table 1 in
Section 1.3, and its scope and limitations are incorporated into Section 1.5.
Section 1.4 states the contribution after the literature positioning.
Conclusions is now Section 7. This is a structural integration, not merely a
rewrite that leaves the original final positioning section in place.

The introductions of Ares De Parga et al. (2026), As'ad et al. (2022), and the
current MAW–ECM manuscript informed the problem-to-gap progression, the
distinction between fitting and mechanical constraints, and the explanation
of each numerical example's purpose. The feature-order implication supplies
a concrete explanation of the representation issue before the construction
is introduced. The primary/secondary nomenclature is retained.

All 42 bibliography entries are retained and reordered by first citation.
The pre-integration sources are in `archive_pre_intro_integration/`.
Methods, numerical assets and timing evidence are unchanged. Source checks,
compilation and visual inspection of the integrated introduction pass;
these checks do not close the outstanding full-corpus reading or experimental
submission requirements.

## PANN/PROM framing and front matter — completed 7 September 2026

- Restored the exact original PANN manuscript title and its author order.
  Used the official affiliation wording in the supplied MAW–ECM source and
  its CIMNE note for S. Ares de Parga; A. Cornejo has CIMNE and UPC–DECA.
- Rebuilt the introductory progression around FE2 cost, offline/online
  surrogates, non-intrusive PANNs, and intrusive PROMs. Explained why physics
  augmentation is compatible with non-intrusiveness and ANN enrichment with
  intrusive projection. Kept the primary/secondary nomenclature and the
  direct decoder's distinct status.
- Updated the abstract, section heading and method diagram consistently.
  Literature positioning remains inside Section 1. Memory demand motivates
  acceleration, but no unmeasured memory saving is claimed.
- Preserved the method equations, numerical results, timing evidence and all
  42 bibliography entries. Pre-revision sources are archived in
  `archive_pre_pann_prom_revision/`; the original Claude source is untouched.
  The updated reading copy has 28 pages and passes compilation and source
  validation without warnings.
