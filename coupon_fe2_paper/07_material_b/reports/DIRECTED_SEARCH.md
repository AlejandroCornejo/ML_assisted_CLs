# Material B: directed search for Free loss of rank-one convexity

The stronger, user-requested follow-up found negative rank-one curvature in
**all three frozen Free models**. Direct F-path automatic differentiation and
energy finite differences independently confirm the negative values. This
establishes that these learned energies are not globally rank-one convex and
therefore not globally polyconvex. These witnesses are outside the approved
training/test strain box; the earlier positive-box audit remains valid.

This is an exploratory post-test result. The expanded domain and budget were
specified in [the internal rule](../protocol/DIRECTED_SEARCH_INTERNAL.md)
before this experiment ran. No training or checkpoint selection changed.

## Search and verified witnesses

Principal stretches range from 0.55 to 1.50 with arbitrary in-plane material
orientation. Each seed used 4,096 shared Sobol states and 64 directions,
followed by eight bounded local curvature minimizations. All 24 reported
convergence within their 2,000-evaluation budgets. The cloud already contained
negative states for every seed (1,514 / 1,443 / 1,643 respectively).

The strongest found negative curvatures were -336.23, -355.84 and -359.96 MPa
for seeds 16, 29 and 47. All three passed independent derivative checks.
These are found values, not certified global minima.

We then minimized ||E||_F subject to curvature <= -1 MPa, using eight starts
per seed. All 24 constrained optimizations reported convergence. We retained
the closest feasible evaluated state (constraint tolerance 1 Pa). Thus the
approximately -1 MPa values below are an intentional margin, not an inferred
natural instability magnitude or the most negative result.

| Free seed | E11 | E22 | 2E12 | J | Verified curvature [MPa] |
|---|---:|---:|---:|---:|---:|
| 16 | -0.083884 | -0.172163 | -0.027525 | 0.738183 | -1.000 |
| 29 | -0.080447 | -0.177566 | 0.009424 | 0.735543 | -1.000 |
| 47 | -0.162486 | -0.095323 | 0.008718 | 0.739094 | -1.000 |

Directions a and b are normalized; full precision states, directions and F
are retained in `results/directed_search/summary.json`. Finite differences
at h=1e-3, 3e-4 and 1e-4 agree with the acoustic calculation within 0.069%
at these three witnesses. The calculation includes the current-stress term.
The complete local search/refinement used 11,959 curvature evaluations.

## Matched constrained models

Every one of the 12 constrained models was evaluated at each selected state
and in its exact same rank-one direction. Their curvatures are positive:
across all 36 comparisons they range from approximately 148.88 to 258.57 MPa.
Different Free seeds need not fail at the same state: the table deliberately
reports a separate witness for each one.

## FOM interpretation

FOM verification was stopped at the user's request. No remeshing was performed:
the existing 4,621-element working mesh and 8,961-element check mesh were reused.
The working-mesh continuation failed before reaching all three selected targets,
exhausting step halving at the prescribed minimum increment. The seed-47 check-mesh
run also finished with that failure; the seed-16 and seed-29 check-mesh runs were
interrupted and are explicitly marked `stopped_by_user`. No FOM job remains active.
The verified neural counterexamples remain valid. These attempts do not establish
an artificial instability relative to a stable physical RVE, or a failure of a
structural solve. Failed FOM continuation does not prove physical instability.

## Figure and reproducibility

[Figure](../results/directed_search/witnesses.png) /
[vector PDF](../results/directed_search/witnesses.pdf).
Left: a straight engineering-strain path toward the seed-29 witness, with
the same fixed rank-one direction for every model. Lines use the previously
selected median-validation seed of each model; bands span all three seeds.
The shaded strip identifies the portion inside the approved strain box.
Right: each Free seed's energy along its own rank-one witness after subtracting
the tangent line. Its downward curvature is visible without the much larger
constant and linear energy terms. This plotting subtraction changes no curvature.
No FOM response is represented by either panel.

Numerical search: `protocol/directed_search.py`; FOM comparison:
`protocol/check_directed_fom.py`; figure: `protocol/plot_directed_search.py`.
Rules, source and frozen-checkpoint hashes are in
`results/directed_search/specification.json`. The three cloud arrays, all
optimization termination statuses, matched comparisons and plot data are saved.
