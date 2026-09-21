# Report map

Start with these three:

1. [`CAMPAIGN_REPORT.md`](CAMPAIGN_REPORT.md): are the FOM labels numerically
   acceptable, and what are the limitations?
2. [`TRAINING_PREPARATION.md`](TRAINING_PREPARATION.md): what was fixed before
   neural optimization? Its pending-work statements are historical.
3. [`FINAL_NEURAL_EVALUATION.md`](FINAL_NEURAL_EVALUATION.md): what did the locked
   15-model test and held-out paths show, including unfavorable evidence?

The other reports document how those decisions were reached:

| Report | Role today |
|---|---|
| `PILOT_REPORT.md` | Initial geometry/loading/mesh investigation |
| `PREFLIGHT_REPORT.md` | Reference-mesh and candidate-domain checks |
| `NONLINEAR_EXPLORATION.md` | Evidence that axial response becomes strongly nonlinear |
| `EXPANDED_BOX_REPORT.md` | Rejected larger-shear domain; retained unfavorable evidence |
| `ASYMMETRIC_BOX_REPORT.md` | Accepted final asymmetric domain |
| `TRAINING_CAMPAIGN_PAUSE.md` | Why the eight v1 runs remain pilots |
| `TRAINING_AMENDMENT_CHECK.md` | Historical check of the second validation-only pilot |
| `TRAINING_PILOT_SUMMARY.md` | Why pilot checkpoints were removed and the single official rule was adopted |

These are audit records, not chapters that a new reader must read sequentially.
The short scientific story and current figures are in `../paper_evidence/`.
# Latest bounded audit

The subsequent user-requested [directed search](DIRECTED_SEARCH.md) expands
the domain and optimization budget and finds verified negative Free curvature
in all three seeds. Its FOM comparison and extrapolation limits are reported there.

[`ROBUSTNESS_AUDIT.md`](ROBUSTNESS_AUDIT.md) closes the post-test volume-collapse
and rank-one-curvature audit. No negative Free witness was found; the finite
Free collapse limit is documented separately from stability and accuracy.
