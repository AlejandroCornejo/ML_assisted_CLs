# Material B: Free-model L-BFGS closeout amendment

This amendment is recorded **before** resuming the three Free runs from their
Adam-step-200,000 review boundary. It uses only fit/reference and validation
labels; reserved test and held-out-path labels remain unopened.

The original official rule requires an Adam validation plateau before entering
L-BFGS. At 200,000 steps, Free seeds 16, 29, and 47 had not met that rule and
their best validation scores were still improving. The user chose to stop Adam
at this safety ceiling but to run the already specified L-BFGS phase, rather
than omit L-BFGS for Free. This is a **budget-limited Adam exception**, not an
Adam plateau or a claim of convergence.

For these three runs only:

1. Preserve the complete step-200,000 run states and campaign status in a
   review-audit folder, with SHA-256 hashes, before changing a checkpoint.
2. Start L-BFGS from each run's strict best-validation Adam checkpoint. Retain
   the existing training objective, data split, model, L-BFGS settings,
   validation metric, best-checkpoint selection, and L-BFGS plateau rule.
3. Stop L-BFGS only under its original validation-plateau rule. If its first
   300-call review boundary arrives first, pause for review; do not turn the
   boundary into a success criterion or silently authorize more calls.
4. Save the final strict best-validation checkpoint across Adam and L-BFGS.
   Label its report `adam_stop_reason=budget_cap_200000_without_plateau` and
   `lbfgs_stop_reason=validation_plateau`, distinct from the 12 runs satisfying
   both plateau rules. Record this amendment and the closeout runner hashes.

The official rule file and original training code are not rewritten, so their
hashes and the historical run identities remain auditable. The three Free
reports must never be relabeled as fully plateau-complete. Even if L-BFGS
plateaus, stopping Adam while validation was improving is a limitation of the
model comparison and must be disclosed. No reserved test or path evaluation is
authorized until all 15 final checkpoints and reports are verified and frozen
under an explicitly amended evaluation gate.
