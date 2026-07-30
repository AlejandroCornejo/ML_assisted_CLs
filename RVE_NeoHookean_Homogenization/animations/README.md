# WCCM2026 animations

Presentation-generation scripts and final rendered assets (GIFs, PNGs, EPS)
for the WCCM2026 talk. These are not part of the core FOM/PROM-ANN/HPROM-ANN/
PANN pipeline and are kept separate for that reason.

## Known limitation of this migration

Only `make_hprom_ann_three_panel_animation.py` (and
`make_hprom_ann_rve_displacement_gif.py`, which imports its `DEFAULT_*`
constants) had its default paths updated for the new directory layout — this
also fixed a real bug: `DEFAULT_ANN_DIR` previously pointed at the superseded
plain `stage_7_ann_model_ls` checkpoint (validation error 3.28e-4) instead of
the canonical one (2.61e-4, the numbers actually reported in
`prom/ann/README.md`-equivalent documentation). The rename performed during
migration means `DEFAULT_ANN_DIR` now resolves correctly.

`DEFAULT_RESULT_DIR` still points at a Stage-10 run that needs to be
regenerated: the original directory it referenced
(`stage_10_hprom_ann_ls_results_mawecm_..._ann_hrom`) was found to hold a
different physical trajectory than the one used everywhere else labeled
"Stage-10" (see `../DATA_DICTIONARY.md`), so it was not migrated. Re-run
`hprom/ann/stage10_test_hprom_ann_ls.py` with the `maw_dynamic` mode to
produce a fresh, correctly-labeled `stage_10_results_maw_dynamic/` directory
before re-rendering.

The remaining ~13 scripts in this directory (manim workflow diagrams, POD
coordinate figures, training-trajectory animations, etc.) still contain
bare/relative paths written for the old flat repository layout and were
**not** individually re-pathed as part of this migration -- re-rendering any
of them will likely require updating their path constants first. The final
rendered outputs already produced (GIFs, PNGs) were migrated as-is and do not
need re-rendering to be viewed.

Two large Manim cache directories (`media/`, `manim_media/`, ~28MB combined)
were deliberately **not** migrated -- they are fully regenerable build
byproducts of the scripts here, not source content.
