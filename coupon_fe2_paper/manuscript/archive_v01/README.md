# New manuscript — current periodic RVE and coupon

Working draft v0.1, 2026-09-06. Written in English for a computational-mechanics
journal. This directory is separate from, and does not overwrite, the previous
Claude manuscript or any result/checkpoint.

## Deliverables

- `manuscript.tex`: new argument, general theory first, then two numerical
  questions with their own purposes. Author list is deliberately unassigned.
- `manuscript.pdf`: compiled reading copy, when compilation is available.
- `figures/`: six reproducible figures, PDF vectors plus PNG previews.
- `tables/`: five generated numerical tables; architecture and literature
  comparison tables are in the TeX source.
- `evidence_manifest.json`: result inputs, hashes, exact measured times and
  recomputed field errors. No new timing experiment is implied.
- `REFERENCE_AUDIT.md`: checked claims, corrections, missing sources and the
  unfinished full-reading queue. **Not all supplied references have been read.**

## Narrative

1. Why constitutive approximation and reduced micromechanics solve related
   but different problems.
2. General periodic homogenization and work-conjugate variables.
3. Learned constitutive construction, proofs and distinction between the four
   neural model tiers.
4. Linear ECM, nonlinear-manifold adaptive cubature, and direct closure.
5. Example I: the current RVE tests approximability, integrability and sampled
   rank-one curvature. Regression has a valid cycle witness; the Free OOD
   witness is not described as an FOM-confirmed artificial instability.
6. Example II: the coupon tests error propagation and online time for the same
   RVE. This is not represented as a second material example.
7. Formulation-specific literature comparison, limitations and conclusions.

The organization follows the principle in Joaquín's manuscript that each
example has a specific role. It does not copy his prose or import his results.

## Reproduce numerical assets

From the project root:

```bash
MPLCONFIGDIR=/tmp/coupon-manuscript-mpl OPENBLAS_NUM_THREADS=1 \
  python3 coupon_fe2_paper/manuscript/build_evidence.py
```

The script loads NumPy/Matplotlib from the existing local `.pydeps`, leaves
source results untouched, and overwrites only generated assets in this new
manuscript directory. It verifies common mesh hashes, load and convergence
metadata. Error norms are the declared unweighted stored-array norms, not
quadrature-weighted tensor norms.

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

Initial compilation downloads TeX packages. The source contains its
bibliography, so it does not require a separate BibTeX database.

## Remaining work before submission

This is a substantial first draft, **not the completion of the user's full
reference-reading request**. See the audit for the exact distinction.

- Finish every supplied reference and the necessary additional primary
  sources; obtain the complete Boehler chapter (currently pp.13–14 only).
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
- TeX compiled with no unresolved citations/references or overfull boxes.
  A few nonfatal underfull text-box warnings remain.
- Representative PDF pages and figure previews inspected; reduced-support
  title overlap corrected in the figure generator.
- A fresh attempt to run `test_flexible_pann.py` could not import PyTorch in
  the available Python environment. No new network unit-test pass is claimed.
  The manuscript's derivative/energy checks refer to the archived independent
  checkpoint audit, whose hash is recorded, not that failed rerun attempt.
