# Report map

Start with only these two:

1. [`CAMPAIGN_REPORT.md`](CAMPAIGN_REPORT.md): are the FOM labels numerically
   acceptable, and what are the limitations?
2. [`TRAINING_PREPARATION.md`](TRAINING_PREPARATION.md): what is fixed before
   neural optimization, and what remains undone?

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
