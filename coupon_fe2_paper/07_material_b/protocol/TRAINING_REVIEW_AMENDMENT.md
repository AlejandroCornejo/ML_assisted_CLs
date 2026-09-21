# Material B: collective Adam review amendments

The official rule was frozen before these 15 runs. It says that a documented
review may authorize one additional 5,000-step Adam block. At the first common
review boundary, all 15 runs reached Adam step 20,000 without meeting the
validation-plateau rule. None failed or entered L-BFGS. Each run's minimum
validation score was attained between steps 19,910 and 20,000; some individual
scores fluctuate, so this is evidence of recent useful optimization, not a
guarantee that every next step will improve.

The user approved a **single collective review every 10,000 Adam steps** after
all 15 runs have paused, instead of reviewing at every 5,000-step boundary.
This changes the *practical review cadence*; it does not change the validation
metric, stopping test, scheduler, model, data split, optimizer, or final-model
selection. It is recorded transparently as two contiguous 5,000-step approval
events carrying one batch identifier and decision time. A model may still stop
before step 30,000 if the unchanged validation-plateau rule triggers.

The 15 original checkpoints and campaign status are copied under
`../results/neural_training/reviews/adam_20000_to_30000/` before approval.
Its `approval.json` records checkpoint hashes, the reason, and the batch size.
No reserved test or held-out path responses were consulted. The next collective
review at step 30,000 found that all 15 runs improved their best validation
scores by 8.1--49.4% (in normalized mean-square stress error) over step 20,000.
All 15 then received a synchronized 10,000-step extension to step 40,000, with
backups and approval hashes under `reviews/adam_30000_to_40000/`.

At step 40,000, all 15 remained in Adam without plateau or failure. Their best
validation mean-square scores improved by 4.7--31.3% relative to step 30,000;
the best steps were 39,790--40,000. Twelve of the 15 had not yet triggered
even one scheduler learning-rate reduction. The user then approved a new
**100,000-step Adam safety ceiling** to permit overnight training, not a fixed
finalization step. The exact current weights, optimizer, scheduler, and RNG
states are resumed from step 40,000; no run is restarted. This supersedes the
earlier plan to stop every 10,000 steps for a manual collective review. The
single approval is transparently recorded as twelve contiguous 5,000-step
bookkeeping events per run with one shared review batch identifier.

The unchanged validation-plateau rule remains active at every scheduled check.
Runs that satisfy it may enter L-BFGS before step 100,000. A run reaching step
100,000 without it must remain `needs_review`, with no final `model.pt` from a
budget cutoff. L-BFGS safety reviews require a separate decision. These changes
were made after inspecting validation behavior and must be disclosed as such
when reporting the final campaign; no reserved test or held-out path responses
were consulted.

At the 100,000-step ceiling, seven of the 15 runs satisfied both validation
plateau rules and completed (three ICNN-fixed, three ICKAN-fixed, and one
ICKAN-learned). The remaining eight stopped at the Adam safety ceiling: three
Free, three ICNN-learned, and two ICKAN-learned. Their best validation MSEs
improved by 25.9--71.4% relative to step 40,000, and each best checkpoint was
at step 99,910--100,000. The user therefore approved lifting the safety
ceiling **for these eight only** to 200,000 Adam steps. The seven completed
model checkpoints remain untouched. The eight resume from exact step-100,000
states; the unchanged validation-plateau test may stop any of them sooner.
Reaching 200,000 without plateau remains a review pause, not completion.
This further data-dependent procedural amendment must be disclosed in the
paper. Reserved test and held-out path responses remain unopened.
