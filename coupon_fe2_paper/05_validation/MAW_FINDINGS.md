# MAW-ECM on this problem: what works, what does not, and why

## Summary

MAW-ECM works for the **homogenized stress** rule and does not work at small
support for the **manifold-projected residual** rule. The dividing line is not
a fitting difficulty; it is a structural property of the integrand that can be
measured before any network is trained.

The earlier conclusion that "MAW-ECM is a negative result on this problem" was
wrong for two independent reasons, both of which were mine.

## Error 1: the baseline was 4x and 7x more expensive than the method

MAW-ECM at 10 points was compared against the classic fixed-weight ECM at 40
points (residual) and 73 points (stress). At **equal point count** the picture
inverts. Classic fixed-weight ECM, median relative constraint error on 742
held-out states:

| points | residual rule | stress rule |
|-------:|--------------:|------------:|
| 5   | 7.44e+00 | 4.43e-02 |
| 10  | 2.38e+00 | 1.08e-02 |
| 15  | 4.96e-01 | 4.01e-03 |
| 20  | 3.26e-01 | 1.00e-03 |
| 30  | 4.64e-02 | 5.46e-04 |
| 40  | 2.08e-02 | 2.00e-04 |
| 73  | 3.26e-03 | 2.09e-05 |
| 121 | 3.71e-04 | 3.33e-06 |

Against the 10-point row, the original MAW result (1.02e-01 residual,
3.10e-03 stress) was already **23x** and **3.5x** better, not 35x and 13x
worse. The whole negative result was an artifact of the comparison.

## Error 2: the weight field was trained on the wrong objective

`fit_mawecm_ann` minimizes

    loss_kl  +  mse_weight * loss_mse  +  physics_weight * loss_phys

with `mse_weight = 10.0` and `physics_weight = 0.0` by default. The first two
terms match the **pruned weight values**; only the third is constraint
satisfaction. The first run here used the defaults, so the objective we care
about carried none of the loss. Restoring the previous project's own
`physics_weight = 1.0` improved the residual rule 8.2x (8.39e-01 -> 1.02e-01)
and the stress rule 2.1x. Note that at `physics_weight = 1.0` and
`mse_weight = 10.0` the objective still carries only ~1% of the loss.

The intermediate is not needed at all. Constraint satisfaction needs only
`A(q)` and `b(q)`, and `b` is the full-mesh integral, now stored at every state
in `full_integrand.npz`. So the field is trained **directly on constraint
satisfaction**, with no target weights in the loss and no weighting to tune.
This also removed a data starvation problem: the pipeline fit on 495 states
because that is what the pruning subsampled to; there are 4950, and for a
3-input regression asked to generalize, 421 samples is ~7.5 per dimension.

## The actual finding: non-negative cubature versus cancellation

A cubature constrained to **non-negative** weights cannot cheaply reproduce an
integral whose value is a near-cancellation of its terms. It has no opposing
signs available, so it must place enormous weights on a few elements. Measured
over 4950 states:

| integrand | \|sum_e c_e\| / sum_e \|c_e\| |
|---|---|
| homogenized stress | 9.51e-01 (5% cancellation) |
| projected residual | 2.05e-02 (**98% cancellation**) |

That ratio predicts everything that follows, and it costs no training to
measure. At 10 points, solving per state for the weights closest to uniform
that satisfy the constraints exactly subject to `w >= 0`:

| | stress | residual |
|---|---|---|
| oracle constraint error | 2.12e-16 | 1.45e-09 |
| states where `w >= 0` is active | **0.0%** | **98.8%** |
| peak weight / uniform | 2.5 | 8.6 |
| weight change between nearest neighbours in q | 4.68e-02 | 1.26e-01 |
| warm-start KL reached | 2.69e-05 | 9.08e-03 |

For the stress rule the feasible set is roomy: the bound is never active, the
exact solution is unique and smooth in q, and a network reaches 2.9e-03 within
1000 epochs against the classic rule's 1.08e-02 at the same cost.

For the residual rule at 10 points the exact solution sits on a **face** of the
feasible set at 98.8% of states, with one element carrying 1330 of the total
1546 -- the rule is really using fewer points than it has. Since which face is
active changes with q, the optimal weight field is piecewise rather than
smooth, and a continuous network cannot represent it. Training directly on
constraints does converge, but from 8.2e+01 to 8.5e-01 over 21000 epochs and
still falling: the objective is right and the landscape is bad.

Caveat on strength of evidence: the correlation between the cancellation ratio
and every downstream difficulty measure is strong and mechanistically sensible,
but that cancellation *causes* the difficulty is an inference, not a controlled
result.

## The trained fields, as cubature rules

Median relative constraint error on 742 held-out states, field trained directly
on constraints over 4208 states:

| rule | points | classic fixed w | MAW field w(q) | gain |
|---|---:|---:|---:|---:|
| stress | 10 | 1.048e-02 | **1.611e-04** | **65x** |
| stress | 15 | 3.959e-03 | **9.322e-05** | **42x** |
| residual | 20 | 4.324e-01 | **1.658e-02** | **26x** |
| residual | 30 | 4.997e-02 | 2.910e-02 | 1.7x |

Two things to read here. The stress rule at 10 points beats the classic rule at
**40** points, a 4x reduction in support. And the residual rule does work at 20
points -- the failure is specific to the small-support regime where
non-negativity binds, so "MAW-ECM does not work for the residual" was too
strong; it does not work for the residual **at 10 points**. Its collapse to
1.7x at 30 points, where the bound is active at 99.2%, is consistent with that
reading.

## The support was never pruned: an error that cost most of the result

Everything in the two sections below used the **classic ECM's own support** with
an adaptive weight field laid on top. That is not MAW-ECM, it is ECM with
variable weights: the support was never chosen by adaptive-weight pruning, which
is the part of the method that reaches point counts a fixed-weight rule cannot.
Running the actual recipe -- classic ECM at tol 1e-4 giving 75 candidates, phase
1 unregularized to 50, phase 2 graph-regularized at alpha = 1e4 to the target --
changes the deployed stress error by **5x**.

Median relative constraint error on 742 held-out states, stress rule:

| points | classic support, fixed w | MAW support, best fixed w | MAW support, oracle w(q) | MAW support, fitted w(q) | gain |
|---:|---:|---:|---:|---:|---:|
| 30 | 5.459e-04 | 4.102e-01 | 2.32e-16 | 7.270e-05 | 7.5x |
| 25 | 1.276e-03 | 3.618e-01 | 2.07e-16 | 7.616e-05 | 16.8x |
| 20 | 9.823e-04 | 8.011e-02 | 2.14e-16 | 7.640e-05 | 12.9x |
| 15 | 3.959e-03 | 1.583e-03 | 1.69e-16 | 6.335e-05 | 62.5x |
| **10** | 1.048e-02 | 2.370e-03 | 1.53e-16 | **9.011e-05** | **116x** |

Three things this settles.

**The fitted field is flat at ~7e-05 from 30 points down to 10.** Accuracy is
limited by the regression, not by the support size, so 10 points performs as
well as 30. Non-negativity is inactive (0.0%) and the oracle exact to 2e-16 at
every count, consistent with the stress integrand's low cancellation.

**The pruned support is bad for fixed weights** -- 4.10e-01 at 30 points, 750x
worse than the classic support there. This answers a question that had been
conflated throughout: the value is **entirely in the adaptive weights**, and
the pruning selects points that only make sense if the weights vary. It is not
a better support, it is a different one.

**The pruning still helps the field**: 9.01e-05 against the 1.45e-04 the same
kind of field reached on the classic support, 1.6x.

## Deployment: MAW at 10 points matches the classic rule at 75

Residual inside Newton is the classic fixed-weight ECM at tol 1e-4, 88 elements,
identical in every row. Alignment against the full-mesh stress verified at
1.4e-16. 40/40 solved on both sets.

| stress rule | points | in-envelope | out-of-envelope | degradation |
|---|---:|---:|---:|---:|
| classic ECM, fixed w | 75 | 9.3205e-04 | 6.1438e-02 | 65.9x |
| classic ECM, fixed w | 15 | 3.5798e-03 | 1.0654e-01 | 29.8x |
| classic ECM, fixed w | 10 | 8.2241e-03 | 5.5213e-01 | 67.1x |
| MAW-ECM, field w(q) | 15 | 1.0880e-03 | 2.0525e-01 | 188.7x |
| **MAW-ECM, field w(q)** | **10** | **1.1332e-03** | **6.0967e-02** | 53.8x |

**The 10-point adaptive rule reproduces the 75-point classic rule** -- 1.22x
away in-envelope, marginally better out -- at **7.5x fewer points**. Against the
classic rule at equal cost it is **7.26x** better in-envelope and **9.06x** out.
Structural guarantees hold at every deployed query including all 40
out-of-envelope ones: minimum weight 6.666e+00, weight sum 1546.000000 to
1546.000000.

Two things not explained.

1. **MAW-10 is saturated at the floor.** 1.1332e-03 against the 9.3205e-04 that
   the classic 75-point rule reaches, which is set by the manifold closure, not
   by quadrature -- refining the residual rule from 40 to 88 elements moved it
   from 9.2952e-04 to 9.3205e-04, i.e. not at all. So no further cubature
   accuracy buys anything here; the decoder would have to be retrained.
2. **MAW-15 is 3.4x WORSE than MAW-10 out of envelope** (2.05e-01 against
   6.10e-02) and degrades 188.7x against 53.8x, despite a *better* in-envelope
   constraint error (6.34e-05 against 9.01e-05). More points generalizing worse
   is counterintuitive and has no mechanism here. It could be an accident of one
   fit; it needs repeating across seeds before it goes in the paper.

## Superseded: the same fields on the classic ECM support

Homogenized stress against the FOM, on the SAME converged q from the SAME
HPROM-ANN Newton solve, so the cubature is the only thing that varies.
Alignment against the full-mesh stress verified at 2.8e-16.

| stress rule | points | in-envelope | out-of-envelope | degradation |
|---|---:|---:|---:|---:|
| classic ECM, fixed w | 73 | 9.295e-04 | 4.972e-02 | 53.5x |
| classic ECM, fixed w | 10 | 8.259e-03 | 1.432e-01 | 17.3x |
| MAW-ECM, field w(q) | 10 | **5.668e-03** | 2.347e-01 | 41.4x |

A **65x** advantage in constraint reproduction becomes **1.46x** in deployed
stress error in-envelope, and **0.61x** -- a loss -- out of envelope. Both
numbers matter more than the 65x, because they are the quantity the paper
reports.

Two separate causes, and they should not be conflated:

1. **In-envelope, the cubature is not the dominant error.** The classic 73-point
   rule reaches 9.3e-04, which is the floor set by the HPROM-ANN's own solution
   error, not by quadrature. Driving the cubature's constraint error from
   1.0e-02 to 1.6e-04 cannot buy more than that floor allows.
2. **Out of envelope, the weight field is a trained component queried
   off-distribution**, and it degrades the way trained closures degrade here:
   41.4x against the fixed rule's 17.3x. This project has already measured the
   same split -- the reduced basis generalizes out of envelope while the trained
   closure does not.

**This bears on the paper's thesis and sharpens it.** Both structural guarantees
hold exactly at every deployed state, in-envelope and out: minimum weight
2.971e+00, weight sum 1546.000000 to 1546.000000 at every single query. The
adaptive rule is therefore a data-fitted component that *does* carry structural
guarantees valid everywhere -- and it still loses accuracy out of envelope. So
the claim cannot be that structural guarantees are what separates the methods
that generalize from those that do not. It has to be narrower: the guarantees
must be ones that **control the error being reported**. Non-negativity and exact
volume conservation bound the volume, and they prevent the catastrophic failure
modes Hernandez warns about outside the convex hull; they do not bound the
stress error, and here they did not.

## A reproducibility defect found along the way, now fixed

The classic ECM support depends on `OMP_NUM_THREADS`. Same input, same rank:

| threads | first support entries | 20-point residual baseline |
|---|---|---|
| 8 | 19 61 98 167 208 268 | 3.1997e-01 |
| 6 | 19 29 45 61 63 183 | 4.3241e-01 |

The first explanation recorded here -- that threaded LAPACK returns a different
basis for a near-degenerate singular subspace -- was wrong, and measuring it
says so plainly. Nothing in the pipeline uses a randomized SVD; it is
`np.linalg.svd`, LAPACK `gesdd`. Across 8 and 6 threads:

- left singular vectors agree to **2.1e-13** (sign-corrected, first 25);
- the relative gaps are wide, min `sv[i+1]/sv[i] = 0.350` for `i < 20`, so there
  is no near-degeneracy to exploit;
- `EmpiricalCubatureMethod` contains no `random`, `seed`, `shuffle` or
  `permutation` at all, and given a bit-identical basis returns an identical
  support at 8, 6 and 1 threads.

The real cause is that the greedy ECM **amplifies** the last-bit differences.
Perturbing the input basis by a relative epsilon at fixed thread count:

| relative perturbation of U | support | baseline error |
|---|---|---|
| 0 | 19 61 98 167 208 | 3.1997e-01 |
| 1e-15 | 19 61 98 167 208 | 3.1997e-01 |
| **1e-13** | **19 29 45 61 63** | **4.3241e-01** |
| 1e-11 | 19 29 45 61 63 | 4.3241e-01 |

So `np.argmax(ObjFun)` is resolving a near-tie whose outcome flips somewhere
between 1e-15 and 1e-13, and threaded LAPACK's reduction-order differences are
comfortably past that threshold.

Pinning the integrand SVD to one thread is still the right fix and is
sufficient, since the ECM is deterministic given a fixed basis -- verified
identical across 8, 6 and 2 threads. But it fixes reproducibility, not the
underlying fact, which matters for how baselines are reported: **the classic ECM
support at a given rank is not a stable function of the data.** At rank 19 there
are at least two reachable supports whose accuracy differs by 35%. Any
comparison against "the classic ECM" carries that ambiguity, and the MAW gains
below should be read against it -- the residual rule's 26x at 20 points becomes
19x against the better of the two supports, while the stress rule's 65x is far
outside the ambiguity either way. A careful baseline would take the best of
several perturbed restarts rather than one run.

The trained-field tables above predate the pinning, so they are being
regenerated for consistency; the deployment comparison in particular had the MAW
support built from a 6-thread basis and the classic baselines from an 8-thread
one.

## Still owed

- A **serial wall-clock run**. No timings anywhere above; speed-up is not
  measured on a shared machine.
- The deployed-q diagnostic, to confirm cause 2 above rather than infer it:
  whether the HPROM-ANN's converged q falls outside the q box the field was fit
  on, and by how much, separately in and out of envelope.

## Consequences for the design

- **Stress rule: use MAW-ECM.** This is also, independently, exactly where the
  previous project deployed it -- on the homogenization target via
  `_evaluate_maw_hom_weights_current`, not inside the Newton loop.
- **Residual rule: MAW-ECM is viable at 20 points, not at 10.** At 20 points it
  gives 1.66e-02 against the classic rule's 4.32e-01, and beats the classic
  40-point rule -- a 2x support reduction. At 10 points it cannot be regressed,
  for the reason above. The relevant diagnostic is peak weight and bound
  activity (3.4 and 82.2% at 20 points, against 8.6 and 98.8% at 10), both
  measurable before training.
- Whether the residual's 2x support reduction is worth an adaptive rule inside
  the Newton loop is a separate question this has not answered, since a
  q-dependent weight makes the consistent tangent require dw/dq. The previous
  project wrote that Jacobian (`mawecm_ann_jacobian_claude.py`) for exactly
  this reason, so it is available rather than new work.
- Both structural guarantees hold by construction at every deployed state,
  in-envelope and out: `w = n_elements * softmax(logits(q))` gives `w >= 0` and
  `sum(w) = n_elements` exactly, and that sum IS the volume row of the
  constraint system, so that row is satisfied identically and only the 3
  (residual) or 4 (stress) physical rows have to be learned.

## Seed robustness, and why it was worth running

Deployed stress error, same pruned support, only the field's initialization and
training seed changed:

| seed | 10 pts, in | 10 pts, out | 15 pts, in | 15 pts, out |
|---|---:|---:|---:|---:|
| 11 | 1.1332e-03 | 6.0967e-02 | 1.0880e-03 | **2.0525e-01** |
| 23 | 1.0900e-03 | 6.6915e-02 | 1.1070e-03 | 1.2041e-01 |
| 37 | 1.1239e-03 | 6.8710e-02 | 1.1244e-03 | 8.4173e-02 |

The 10-point rule is stable to **±2% in-envelope and ±6% out**, so the headline
holds. The 15-point rule is stable in-envelope but swings by a **factor 2.4**
out of envelope, and seed 11 -- the one the "MAW-15 is 3.4x worse than MAW-10"
anomaly was reported from -- is its worst of three. That anomaly was largely
seed variance. What survives is weaker and different: 10 points is somewhat
better than 15 out of envelope and *much* more stable across seeds.

I had argued this check was unnecessary because the constraint error varied only
+/-6% across seeds. That was wrong: the number the paper quotes is the DEPLOYED
one, which passes through the Newton solve and is queried at different q. A
stable fit does not imply a stable deployed result, and without the repeat a
3.4x effect with an invented mechanism would have gone into the paper.

## The residual: three explanations tried, two of them wrong

**Phase 2 from the first iteration** (`smooth_laplacian_all_iterations`, no
phase 1) does change something real: non-negativity activity drops from
94.9-98.8% on the classic support to **5-26%** on the pruned one. And MAW beats
the classic rule at equal point count everywhere, 2.7x to 31x. But the fitted
field sits at ~1e-02 at every support size:

| points | classic, fixed w | MAW, fitted w(q) |
|---:|---:|---:|
| 10 | 2.3039e+00 | 7.44e-02 |
| 15 | 4.9772e-01 | 2.58e-02 |
| 20 | 3.1997e-01 | 6.04e-02 |
| 25 | 1.1781e-01 | 1.42e-02 |
| 30 | 4.5483e-02 | 1.66e-02 |
| 80 | 2.3767e-03 | 9.78e-03 |

Eight times the points buys 7.6x, while the classic rule improves by a factor of
a thousand over the same range -- and at 80 points MAW **loses**, 0.24x. The
88-point classic reference is 1.5805e-03; nothing adaptive came within 6x of it.

**Explanation 1, non-smoothness of the optimal weight field: too weak.** The
nearest-neighbour jump in the oracle weights is 4.9-6.7e-02 for the residual
against 3.1-3.9e-02 for the stress -- 1.6x, which cannot account for 200x.

**Explanation 2, amplification: measured and real, but I connected it to the
wrong quantity.** Perturbing the oracle weights by a relative epsilon:

| rule | \|b\| / sum\|Aw\| | eps=1e-4 | eps=1e-3 | eps=1e-2 |
|---|---:|---:|---:|---:|
| stress, 10 pts | 0.987 | 1.75e-05 | 1.73e-04 | 1.75e-03 |
| residual, 10 pts | 0.036 | 1.00e-03 | 1.01e-02 | 9.71e-02 |

A relative weight error eps gives ~0.17*eps of constraint error for the stress
(attenuated) and ~10*eps for the residual (amplified) -- a 57x difference,
matching the cancellation ratio. But I then inferred a weight error of ~1.4e-03
from the fitted constraint error, and that is wrong: the ACTUAL weight error
against the oracle is **0.23 to 1.95** for the residual and 0.07 to 0.22 for the
stress. The field does not approach the oracle at all; the system is
underdetermined and it finds a different vector in the solution set. The
amplification factor applies to random perturbations, not to a field that is
optimizing constraint satisfaction directly.

**Explanation 3, optimization -- what the evidence actually supports.** Fit
error versus held-out error, both rules, every support:

| rule | pts | constraint, FIT states | constraint, HELD OUT | gap |
|---|---:|---:|---:|---:|
| stress | 10 | 6.65e-05 | 9.01e-05 | 1.35x |
| stress | 30 | 6.04e-05 | 7.27e-05 | 1.20x |
| residual | 10 | 5.93e-02 | 7.44e-02 | 1.25x |
| residual | 30 | 1.23e-02 | 1.66e-02 | 1.35x |
| residual | 80 | 6.94e-03 | 9.78e-03 | 1.41x |

**There is no generalization gap anywhere: 1.18x to 1.41x.** The residual field
performs the same where it has data as where it does not, so it is not failing
to generalize -- it is failing to fit. And the 80-point run's best epoch was
20000 of 20000: it never plateaued, it ran out of budget. The same was true at
the very first attempt, where the loss fell from 8.2e+01 to 8.5e-01 over 21000
epochs and was still falling.

So "the residual cannot be represented by an adaptive rule" is NOT established.
What is established is that Adam on this loss converges very slowly. A longer
budget, a different architecture, or an RBF regressor -- which replaces the
optimization with a linear solve and so sidesteps the pathology rather than
fighting it -- are all live options, and the existing solver already supports
`regressor_type = "rbf"` with `clip_nonnegative` and `renorm_target`, which
deliver the same two structural guarantees exactly, though by post-processing
rather than by construction.

## Two defects in the fitting code, found by watching a curve

Both were mine, both were reported as properties of the problem before being
recognized as bugs, and both are one-line fixes.

**1. Disabling early stopping silently disabled LR annealing.** The scheduler's
patience was `patience // 4`, sharing the early-stopping parameter, so passing
`patience = 100000` to see a full 100k-epoch run gave the scheduler a patience
of 25000 epochs -- longer than the worst stall in the run (17449 epochs). The
learning rate therefore stayed at 1.5e-3 for all 100000 epochs and was never
reduced once. What that produced, and what I described as difficulty of the
problem: a loss oscillating at **32x its own best**, sudden spikes at epochs
40000/45000/55000/95000, and three apparent plateaus I called convergence, all
of which were a fixed step size bouncing around the optimum. Fixed with an
independent `sched_patience` (default 1000). Measured effect on the 10-point
residual rule: **7.44e-02 without annealing, 2.30e-02 with it, 3.2x**, from the
same support, seed and budget.

Scope: only the two 100k runs were affected. Every sweep used
`PATIENCE = 4000`, hence a scheduler patience of 1000 -- the same value now
made the default -- so the stress results are unaffected.

**2. The model is selected by a criterion different from the one reported.**
`best_state` is kept by validation LOSS, which is the mean of SQUARED relative
constraint errors, while every table in this document quotes the MEDIAN relative
error. The two diverge badly here, because the residual's error distribution has
a long tail: at 10 points the loss froze at epoch 16160 and never improved for
the remaining 84000 epochs, while the median kept falling from ~5e-02 to
2.3e-02. So the run selects a model whose median is roughly twice as bad as one
it visited and discarded. This is the same mistake as fitting the pruned weight
values instead of the constraints -- optimizing an intermediate rather than the
objective -- displaced from the loss into the model-selection rule.

## The full MAW-HPROM-ANN, and what the yardstick error cost

Two more corrections, and the second one reversed the conclusion.

**The residual was judged against the wrong target.** Every residual number was
compared to the classic 88-point rule's constraint error of 1.5805e-03, which
MAW never approached, so I concluded the residual could not be hyperreduced
adaptively. But deployment does not need that accuracy, and this was already
measured: refining the residual cubature from 40 to 88 elements -- a 13x
improvement in constraint error, 2.08e-02 to 1.58e-03 -- moved the deployed
stress error from 9.2952e-04 to 9.3205e-04, i.e. by 0.3%. The floor is the
manifold closure, not the quadrature. Judging the residual rule by its
constraint error was measuring an intermediate for the third time in this work.

With the adaptive rule actually placed inside the Newton loop (weights frozen
within each iteration, an inconsistent tangent, so the fixed point is unchanged
and only the rate suffers):

| residual | stress | in-envelope | out-of-envelope | fails (probe) | its/solve |
|---|---|---:|---:|---:|---:|
| classic 88 | classic 75 | 9.3205e-04 | 6.1438e-02 | 0/40 | 143 |
| classic 88 | MAW 10 | 1.1332e-03 | 6.0967e-02 | 0/40 | 143 |
| MAW 10 | classic 75 | 2.2388e-03 | **2.1966e-02** | 5/40 | 176 |
| **MAW 10** | **MAW 10** | **1.8761e-03** | 4.6821e-02 | 5/40 | 176 |

**20 reduced elements against 163 reach a factor 2 of the floor.** And two
effects that pull in opposite directions, which must not be conflated: the
convergence failures track the RESIDUAL rule exactly (0 with classic, 5 with
MAW, independent of the stress rule), while out-of-envelope ACCURACY is
*improved* 2.8x by the MAW residual and *worsened* 2.1x by the MAW stress.

Improving the residual field from 6.06e-02 to 3.41e-02 constraint error changed
the deployed result from 1.8684e-03 to 1.8761e-03 -- nothing. Which also settles
the model-selection defect above: it cost 2.2x in constraint error and zero in
the quantity the paper reports.

## The hierarchy, timed serially on an idle machine

No timing was reported anywhere until here, deliberately: accuracy is unaffected
by CPU contention, speed-up is not. Same 40 in-envelope states, same 1e-10
tolerance, same 200/unit-strain ramp, asserted at run time.

| model | elements | ms/state | speed-up | stress error |
|---|---:|---:|---:|---:|
| FOM | 1546 | 8449.45 | 1.0x | 4.1407e-15 |
| PROM (39 modes) | 1546 | 1633.15 | 5.2x | 4.7724e-07 |
| HPROM (39 modes) | 208 | 177.90 | 47.5x | 1.5339e-04 |
| HPROM-ANN | 163 | 109.89 | 76.9x | 9.3205e-04 |
| MAW-HPROM-ANN | 20 | 78.74 | 107.3x | 1.8761e-03 |
| D-HPROM-ANN | 75 | 0.92 | 9179x | 8.1019e-04 |
| MAW-D-HPROM-ANN | 10 | 0.70 | 12129x | 8.3409e-04 |

Two runs of this table agree to 1% on every timing.

**Where MAW-ECM pays, and where it barely does.** In the D-HPROM-ANN it is a
free win: 1.32x faster at the same accuracy (8.34e-04 against 8.10e-04), because
with no Newton loop there is no iteration penalty -- just 10 elements instead of
75. In the MAW-HPROM-ANN it is a genuine trade: 1.40x faster for 2x the error.
The element counts suggested far more (8x fewer elements), and the gap is the
23% extra Newton iterations from the inconsistent tangent plus a weight-field
evaluation and a w_detJ rescaling on every iteration. This is exactly the number
that no amount of constraint-error reporting could have supplied.

Note also that the D-HPROM-ANN is more accurate in-envelope than the HPROM-ANN
(8.10e-04 against 9.32e-04): the model that solves nothing beats the model that
solves a reduced equilibrium, because the direct regression is fitted on FOM
snapshots while the HPROM-ANN carries its own reduced-solve error. Consistent
with the earlier out-of-envelope measurement, where the ordering reverses (16x
degradation against 4.9x).

A defect worth recording because it was caught only by an implausible number:
the first version of this table evaluated the D-HPROM-ANN on the FULL mesh, and
then, once moved to the reduced one, applied `A_M` twice -- `PhiMA` already
contains it, and E is itself the latent coordinate in this model. That was worth
a relative stress error of 1.5e+02.

## Scripts

| script | what it does |
|---|---|
| `build_full_integrand.py` | per-element contributions on the full mesh, all 4950 states, 48 s. Makes every support/weight question pure numpy. |
| `maw_lab.py` | shared library: constraint blocks, cached integrand SVD, classic ECM at any rank, per-state oracle weights, the two-stage field fit, numpy field evaluation (verified against torch to 5.9e-16). |
| `sweep_baseline.py` | the classic accuracy-vs-cost curve above. |
| `sweep_feasibility.py` | the training-free feasibility test above. |
| `train_maw_fields.py` | the deployable fields. |
| `deploy_maw.py` | homogenized stress error vs FOM in and out of envelope, with an alignment check against the full-mesh stress. Reports no timings by design -- speed-up must be measured in a serial run. |
