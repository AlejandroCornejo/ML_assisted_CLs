# Material B: bounded robustness audit — closed

**Bottom line:** this audit did not find a loss of rank-one convexity in Free.
It did establish that the current Free architecture lacks a divergent
volume-collapse barrier. These are different conclusions. Neither changes
Free's superior measured interpolation accuracy in the final evaluation.

This is a post-test exploratory audit, not a preregistered primary result.
The rule was fixed before this audit's numerical execution in
[`ROBUSTNESS_AUDIT_INTERNAL.md`](../protocol/ROBUSTNESS_AUDIT_INTERNAL.md).
No training, model selection, manuscript changes, or new FOM runs occurred.
All 15 frozen model hashes and their source/data gate were verified again.

## 1. Volume collapse: a demonstrated architectural limitation

For F=sqrt(J) I, Free's raw input tends to (-1,-1,0,-1) as J tends to zero
from above. Its finite-weight softplus network and affine reference
correction therefore have a finite limit. Direct evaluation of these limiting
network inputs gives:

| Free seed | Limiting energy density [MPa] |
|---|---:|
| 16 | 439.6131 |
| 29 | 419.8097 |
| 47 | 397.4787 |

The 81-point sweep from J=1 to 1e-8 approaches these limits. In contrast,
the constrained construction has a positive logarithmic barrier and the
feature/core contribution is bounded below on this path; its energy diverges.
The plotted finite samples illustrate that analytical growth property; they
do not prove a limit by themselves. At J=1 all energies are zero, so that
reference point is omitted only from the logarithmic plot.

**Scope:** this contrasts the present full constructions, not polyconvexity
in isolation. A Free architecture could also receive an analytic barrier.
Enormous extrapolated energies are not evidence of quantitative accuracy;
no FOM/contact-validity claim is made near complete volume collapse.

## 2. Rank-one search: no negative witness found

Each model used the identical 520 states and 32 b directions in each box,
minimizing over a through the acoustic-matrix eigenvalue. Two bounded local
refinements followed in each model/domain. Engineering shear is 2E12.

| Model | Approved-box minimum across seeds [MPa] | Double-size-box minimum across seeds [MPa] |
|---|---:|---:|
| Free | 236.54–241.08 | 135.31–149.42 |
| ICNN-fixed | 239.43–240.44 | 150.08–163.31 |
| ICNN-learned | 241.91–242.58 | 227.19–229.34 |
| ICKAN-fixed | 233.33–235.97 | 174.16–196.18 |
| ICKAN-learned | 243.06–245.87 | 230.81–234.59 |

Entries are the range of the three lowest-found values, NOT certified global
minima. There were 7,014 local objective evaluations: 5 of 60 refinements
reported convergence, and 55 exhausted the fixed 120-evaluation budget.
No search expansion or budget extension followed these positive results.
The double-size box is constitutively admissible at the macroscopic level,
but its FOM equilibrium/contact validity has not been established.

At all 30 retained minima, direct automatic differentiation of
W(F+t a tensor b) agrees with the acoustic calculation to relative error
2.73e-15 or less. Centered energy differences at three step sizes have a
worst relative discrepancy of 1.09e-4 (0.0109%). Two analytic unit tests
also passed, including the geometric current-stress contribution and
engineering-shear convention.

No Free witness triggered the conditional FOM comparison. Positive sampled
curvatures do not establish global stability. Conversely, larger curvatures
in the constrained models do not establish better accuracy or solver behavior.

## Decision for the paper

Keep the accuracy results unchanged. Material B supports the learned-versus-
fixed feature comparison within each constrained core; it does not currently
demonstrate that Free produces artificial instability or inferior FE solves.
The collapse result can illustrate the scope of the growth guarantee, with
the qualifications above. Do not replace the material or keep broadening
this audit merely to obtain a negative Free result.

## Files

- [Summary figure](../results/robustness_audit/audit.png) / [vector PDF](../results/robustness_audit/audit.pdf).
- [Numerical results](../results/robustness_audit/summary.json): all models,
  states, directions, optimization statuses and verification values.
- [Closure](../results/robustness_audit/closure.json): integrity checks,
  numerical agreement and the untriggered FOM branch.
- `../results/robustness_audit/executed_source.py` preserves the exact hashed
  executable used. Subsequently only the plot scale/legend were improved;
  both source hashes are retained. Numerical results were not rerun.
