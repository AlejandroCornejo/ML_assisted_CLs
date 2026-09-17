# Why the training pilots were superseded

The original fixed-budget pilot had eight completed model–seed runs. Its
2,600-step Adam and 40-call LBFGS budgets frequently selected a checkpoint
at the final LBFGS call. A second, validation-aware pilot completed all 15
runs but still ended Adam at its 7,800-step safety cap in every case. Across
those 15 runs, the best normalized validation-stress MSE improved by 9.1–29.5%
over the final 800 Adam steps; no Adam learning rate had been reduced. Ten
LBFGS runs met the pilot plateau rule; five reached its 120-call cap. Those
five improved by 0.28–5.09% over their final 20 LBFGS calls.

These observations use validation labels only. They show that the pilot caps
were active optimization limits, not evidence of a settled response. The
reserved test and held-out path prediction labels were not opened. The
[single official recipe](../protocol/training_recipe.json) therefore makes a
review boundary a **pause**, never a completed fit, and applies the same
validation-stop rules to all 15 fresh runs.

At the user's request, the two pilot checkpoint directories were deleted to
reduce clutter. Before deletion, the 23 run reports and the completed pilot
campaign status were checked byte-for-byte against
[`../archive/pilot_training_reports.tar.gz`](../archive/pilot_training_reports.tar.gz).
That 240-kB archive retains their validation histories and provenance hashes,
but **not** their model weights, optimizer states or logs. Deleted checkpoints
are not recoverable from this archive. The frozen FOM labels, feature table,
source recipes and FOM audit directories remain in place.
