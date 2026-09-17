# Archived material-B evidence

Nothing here was discarded for giving an unfavorable result. Numerical
JSON/NPZ/MDPA/source files retain their original bytes. Historical specifications
and absolute paths inside their records are unchanged, not silently rebased.

- `pilot_v1/`: first pilot, including failed compression targets.
- `refinement_retry_800/`: successful targeted retry; the operational chain
  uses the separately retained density-1600 record at the root.
- `nonlinear_reference_v1/`: incomplete metadata-construction attempt,
  explicitly not mechanical nonconvergence.
- `mesh_check_v1/`, `expanded_plan_*`: mesh-only and plan-only records.
- `physical_response_v1/`: earlier figure layout; current layout is in
  `../results/physical_response_v2/`.
- `diagnostics/`: original compression diagnostic and partial reference audit.
- `navigation_before/`: exact pre-cleanup reports for recovering their original
  paths/links; current navigation lives in `../reports/`.
- `data_protocol_pre_physical_screen/`: original frozen coordinate design and
  metadata before geometry/contour provenance was added; the regenerated
  coordinate NPZ has exactly the same bytes.
- `smoke_writer_v1/`, `smoke_writer_v2/`: earlier bounded writer/resume tests,
  retained before later source revisions.
- `smoke_writer_failures/`: fit smoke that exposed absent-tangent JSON
  serialization; a writer failure, not a mechanical failure or discarded load.
- `smoke_writer_physical/`: successful residual/J/polygon/jump smoke tests
  moved intact from `../work/` after completion; reports and copied drivers
  retain their original bytes. See its [index](smoke_writer_physical/README.md).
- `DEVELOPMENT_README.md`: the former 286-line root README. It records how the
  project evolved but is no longer the entry point.
- `legacy_code/`: nine completed diagnostics/plotters removed from active code;
  their bytes and recorded hashes are preserved. The current figure generator
  is `../paper_evidence/build_figures.py`.
- `pilot_training_reports.tar.gz`: 23 validation-only neural pilot reports and
  the second pilot's campaign status. Pilot model/optimizer checkpoints and
  logs were deleted at the user's request; this compact archive does not
  restore them. See the [summary](../reports/TRAINING_PILOT_SUMMARY.md).

[relocation_manifest.json](relocation_manifest.json) gives every original and
current file path, byte count and SHA-256 hash. To reverse the reorganization,
use its explicit `moves` mapping in reverse, without overwriting occupied
destinations; restore the four navigation originals from `navigation_before/`.
No reversal has been performed. To verify preserved payloads and unmoved
sources from the repository root:

    coupon_fe2_paper/.venv_fe2/bin/python \
      coupon_fe2_paper/07_material_b/maintenance/organize_stage1.py verify

[`cleanup_v2.json`](cleanup_v2.json) records the later reader-facing cleanup,
including the nine archived scripts and the 103 deleted byte-identical driver
snapshots with their canonical recovery source.

The remaining root run folders are still operational inputs. Moving them
requires a separate path/provenance migration, not another blind cleanup.
