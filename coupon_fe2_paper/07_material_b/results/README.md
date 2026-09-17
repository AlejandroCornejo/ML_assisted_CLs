# Results map

These are machine outputs. For a human introduction, go to
[`../paper_evidence/`](../paper_evidence/README.md).

## Current artifacts

| Artifact | Meaning |
|---|---|
| `data_protocol_design_v1.*` | Coordinates and roles only; no FOM labels |
| `data_labels_v1.npz` | Approved assembled FOM labels |
| `data_labels_v1.json` | Acceptance checks and provenance for those labels |
| `feature_selection_v1/` | Shared 32-feature table, scales and initialization checks |
| `data_campaign_v1/` | Raw restartable campaign records; auditor-facing, not explanatory |

`neural_training/` is the **only official** 15-run campaign. Its status file
distinguishes `complete`, `needs_review` and `failed`. The former pilot neural
checkpoint directories were deleted at the user's request; their compact
reports are in `../archive/pilot_training_reports.tar.gz`, with limitations
spelled out in [the pilot summary](../reports/TRAINING_PILOT_SUMMARY.md).
Temporary smoke checkpoints were kept outside this directory and are not
evidence of learned accuracy.

The raw campaign contains 103 chunk folders. Each keeps `labels.npz`, its
`report.json` and its solver log. The former 103 identical copies of
`run_data_stage.py` were removed after byte comparison with the single frozen
source in `../protocol/`; chunk reports already store that source hash.

## Pre-campaign diagnostics

- `physical_response_v2/` and `nonlinear_response_v1/`: saved pilot/larger-
  tension responses used to explain pretraining nonlinearity.
- `preflight_decision_v1.json` and `asymmetric_decision_v1.json`: mesh/domain
  decisions that led to the final protocol.
- `expanded_decision_v1.json` and `expanded_field_audit_v1/`: the rejected
  larger-shear domain. They are retained because the unfavorable result matters.
- `reference_*`, `constant_tangent_v1.json`, `saved_field_audit_v2.json`:
  targeted mechanical/numerical checks, not model-accuracy results.

Old PNG/PDF files inside diagnostic folders are not paper figures. The only
current reader-facing figures live in `../paper_evidence/`.
