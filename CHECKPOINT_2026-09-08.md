# Work-in-progress checkpoint — 2026-09-08

This checkpoint preserves the current scientific code, saved results, reduced
models, figures, manuscript sources/PDFs, and reference-audit records. It is
not a validated release or a complete backup of the local workspace.

The preceding commit `c5048d1d` fixes the Windows-invalid accidental filename
and adds repository portability checks and targeted ignore rules.

## Existing deletions captured

The workspace already lacked 15 tracked configuration/mesh symlinks under
`RVE_NeoHookean_Homogenization/{hprom/ann,pod,prom/ann,prom/pod,trajectories}`,
65 generated Manim assets under `manim_slide47_media`, and four files under
`softmax_analytical_edge`. The checkpoint records these existing deletions;
it does not restore them or imply that affected legacy entry points work.
Earlier versions remain available in the parent commit.

## Deliberately outside the checkpoint

- Local Python installations, virtual environments, bytecode and build caches.
- LaTeX build intermediates; already tracked modified intermediates are left
  out of this checkpoint rather than rewriting their tracking history.
- The large local datasets and FE2 checkpoints explicitly listed in
  `.gitignore`, including `03_data/data.npz`, `05_validation/full_integrand.npz`
  and `linear_residual_ecm_dataset_claude.npz`.
- Other previously ignored datasets/output and files outside this repository,
  including the external reference-PDF collection.

Ignored data remain on disk and need separate backup/distribution. A clone
alone will not reproduce every training or FE2 workflow. Absolute local paths
in scripts or audit metadata may need adjustment on another computer.

The checkpoint is checked for indexed path compatibility and blob sizes;
this does not certify Windows runtime compatibility, numerical correctness,
or reproducibility. No new training or FE2 timing campaign is part of creating
this checkpoint. Publishing it to the shared remote is a separate operation.

## Checks at checkpoint creation

- Indexed path/case-collision and 100 MiB blob-limit checks passed.
- All 278 added/modified Python sources parsed without syntax errors.
- Git whitespace checks report existing formatting in generated files; these
  assets were preserved without a bulk reformat.
- No training, FE2 run, or full software regression suite was executed.
