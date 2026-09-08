# Coupon PANN: corrected diagnosis and first certified extension

Update: the subsequent learned-feature/independent-volume experiment reaches
approximately **0.65% test stress error for both ICNN and spline-ICKAN**.
See [FLEXIBLE_TRAINING.md](FLEXIBLE_TRAINING.md). The first-stage measurements
and diagnosis below are retained as the experimental record.

Date: 2026-09-05. This note supersedes the diagnosis/recommendations in
sections 7–9 of `PANN_MEMO.md`, not its recorded baseline measurements.
The shared manuscript, shared model implementations, dataset, and original
`pann_*.pt` checkpoints have not been changed.

## What the double-check changed

The original feature bank has a stronger restriction than the raw feature-order
test suggested. Every same-power balanced system satisfies

    sum_i w_i d_i d_i^T = I.

For a feature `Q = sum_i w_i (d_i^T C d_i)^p`, at `C = c I`,

    dQ/dC = p c^(p-1) I.

The original cofactor features have the same property in 2D, as do trace, J,
and J². Therefore *any differentiable core* of these features produces
isotropic stress on the entire equal-biaxial-stretch path. The isotropic
volumetric correction does not change this conclusion.

Differentiating that statement along the path, and using the symmetric
energy Hessian, gives the following reference tangent identities in the
engineering convention `[E11,E22,gamma12]`:

    D11 = D22,    D13 + D23 = 0.

These are **not requirements of polyconvexity**. They are restrictions of this
particular feature construction. They do not imply full material isotropy:
the old model can still have anisotropic deviatoric response. But they rule
out many otherwise ordinary anisotropic elastic tangents.

The coupon's stored periodic linear tangent is, in GPa,

    [[1.562010, 0.626422, 0.178517],
     [0.626422, 1.215682, 0.169428],
     [0.178517, 0.169428, 0.316296]].

Thus `D11-D22 = 0.346329 GPa` and `D13+D23 = 0.347945 GPa`, not zero.
This stored tangent is from the separate Stage-00 calculation. As an
independent check using the actual production-grid labels, quadratic local
stress regression on 93 states with `||E|| < .04` gives respectively
approximately `0.341108 GPa` and `0.349323 GPa`. That regression is an
estimate, not an exact tangent, but confirms the mismatch is not a small
mesh discrepancy. `audit_enrichment.py` reproduces both comparisons.

Rotating or adding any number of the old same-power balanced systems
preserves the identities. Purely volumetric features cannot remove them.
The earlier recommendation to fix this just by adding directions was
therefore insufficient.

## What we implemented

`enriched_pann.py` introduces paired perpendicular directions with *different*
powers, optionally with determinant-dependent factors:

    z(F,J) = |F d|^(2p)/(p J^b) + |F d_perp|^(2q)/(q J^c).

The admissibility conditions are

    p,q >= 1/2;
    0 <= b <= 2p-1;  0 <= c <= 2q-1;
    rho = 2-b/p-c/q >= 0.

Why this helps, in plain language: perpendicular terms have matching first
derivatives at the undeformed state, so the simple reference-pressure
correction still works. Their different powers let their stiffnesses evolve
differently. We retain the cancellation needed at the reference state without
forcing isotropic stress all along the equal-biaxial path.

The model uses seven features selected by nonnegative least squares from a
fixed, predeclared dictionary. Selection uses the fitting subset only.
Centering and positive scaling also use only that subset. Constant centering
does not expand ICNN expressivity; it is a numerical parametrization choice.

### Certificate for the new features and normalization

For independent variables `(v,J)`, `|v|^a/J^b` is convex when `a >= 1` and
`0 <= b <= a-1`. For `|v|>0`, the radial/J Hessian determinant is

    a b (a-b-1) |v|^(2a-2) J^(-2b-2) >= 0,

and its diagonal and tangential terms are nonnegative. Continuity gives the
convex extension at `v=0`; the boundary case `a=1,b=0` is the norm itself.
Taking `a=2p` or `2q` proves each paired feature convex in independent `(F,J)`.
A convex coordinatewise nondecreasing core H therefore gives a convex
function of `(F,J)`, hence a 2D polyconvex energy on `J=det F>0`.

At the reference state, in physical engineering strain coordinates,

    dz/dE = rho [1,1,0].

If `h_i` are the core's nonnegative reference derivatives with respect to
the unscaled features, the reference pressure is `r = sum_i h_i rho_i >= 0`.
Consequently

    W = H(z)-H(z0)-r log J + beta/2 (J-1)^2
        + epsilon (tr C/2 - 1 - log J)

is polyconvex, objective, energy-consistent, and has `W(I)=0, S(I)=0`.
Here `beta>0` and `epsilon=1e-10>0` in normalized energy units. The independent
epsilon term ensures a compression barrier even for dictionaries with zero
reference pressure. Beta supplies quadratic volumetric growth. The epsilon
term also grows with `||F||` at bounded J. These are constitutive statements,
not a complete existence theorem for an arbitrary boundary-value problem.

There is also a nonnegativity argument for the paired-only construction.
Let `t=|Fd|²`, `u=|Fd_perp|²`. Using `a-1 >= log(a)` separately in the two
terms gives

    z-z0-rho log J >= log(t u / J²) >= 0.

The last inequality is Hadamard's inequality for C in the perpendicular
basis. The supporting-hyperplane inequality for convex H and `h_i>=0`
then gives nonnegative corrected structural energy. The explicit volumetric
terms are nonnegative as well. No claim of universality follows from this
certificate.

### Why the KAN experiment has a separate name

The new `core='ickan'` is a **local input-convex spline-KAN variant**, not
the external B-spline implementation used in the old checkpoint. Its edges
are C² cubic splines in truncated-power form:

    phi(x) = bias + softplus(a) x + sum_k softplus(c_k) (x-knot_k)_+^3/6.

Their first and second derivatives are nonnegative everywhere; the second
derivative is continuous at the knots. Positive compositions/sums and a
positive direct input skip preserve the certificate. The right tail is
cubic, not the old linear extrapolation, so this is a genuine model variant
and should be identified as such in comparisons.

The separate audit found that `/home/kratos/ICKANs/ickan/spline.py` uses
`eps=1e-3` *secants* for its linear tails. For a curved convex interior, the
left secant exceeds the interior endpoint derivative and the right secant
is smaller. This makes downward derivative jumps, inconsistent with global
convexity. A concrete admissible coefficient example gives jumps about
`0.00449991` at both joins. `audit_enrichment.py` reproduces it directly from
the installed source. This is a separate implementation issue, not an
explanation for the entire coupon fitting error. It does mean that the old
ICKAN's global tail certificate needs attention before being asserted as
an implementation-level guarantee. No external package was patched.

## Measurements

All stress errors below are relative L2 norms in physical units. Test has
400 valid states; probe uses the same 345 finite-label states as the old
trainer, excluding the five failed FOM labels. No compression states were
discarded merely because `J<1`.

| Model | Test stress error | Probe stress error |
|---|---:|---:|
| Saved original ICNN | 22.781% | 28.868% |
| Saved original ICKAN | 18.326% | 22.885% |
| Paired ICNN, seeds 5/6/7 | about 6.459% | about 15.357% |
| Paired C² spline-KAN, seeds 5/6/7 | 6.455–6.457% | 15.335–15.346% |
| Saved free energy MLP | 1.264% | 5.502% |

New test energy errors are approximately 1.64–1.66%; reference energy and
stress vanish to floating-point precision. These are trained warm-started
models, with 1500 full-batch Adam steps and up to 300 requested L-BFGS
iterations. L-BFGS mostly terminates its inner solves early near this fit;
the history records outer requests, not a promise of 300 effective updates.
Three seeds check repeatability of this *warm-started* procedure, not global
optimizer robustness or random-start performance. Original saved models used
their original training budgets, so the table is not a compute-matched ablation.

For a feature-only control, fitting positive linear sums with exactly the
same fit/validation split and loss gives:

| Feature bank | Feature count | Validation stress error |
|---|---:|---:|
| Original same-power balanced systems | 15 | 18.295% |
| Twelve rotated copies of those systems | 147 | 16.563% |
| New paired dictionary, NNLS-selected terms | 7 | 7.111% |

The nonlinear cores barely improve on the new linear sum. Thus the main
measured gain is **feature expressivity**, not a demonstrated nonlinear
advantage of ICNN versus KAN. NNLS can select different equivalent sparse
representations on different numerical platforms because the dictionary
contains dependent columns; predictions/losses matter more than unique angles.

Seven automated tests pass: old identities including the 147-feature control;
new nonzero coupling; analytic stress vs autograd; sampled feature Hessians
in independent `(F,J)`; spline monotonicity/convexity and C² joins; reference
normalization, gradcheck/gradgradcheck, nonnegativity and input domain; invalid
specification rejection. The analytical proof is the certificate; finite
sampling tests check its implementation and do not replace the proof.

## Corrections to the interpretation and manuscript scope

1. Ordering the *raw* 15 features against total W is not an impossibility
   proof: the volume/reference correction must be included. The earlier LP
   audit permits energy-order feasibility with sufficiently large free
   pressure. Ordering feasibility itself does not prove representability by
   a convex core or compatibility with the model's reference-derived pressure.
2. The demonstrated limitation is architectural, not evidence that this
   RVE's effective energy is non-polyconvex. Nor do positive sampled acoustic
   tensors prove global polyconvexity of the RVE.
3. `J<1` alone is not a pore-closure or instability criterion. Regenerating
   the dataset after rejecting every such state is not justified by this
   diagnosis. Blindly adding `1/J` also changes the sign of its reference
   derivative and needs a revised pressure argument.
4. The old ICNN polyconvexity argument does not imply that its feature bank
   represents arbitrary anisotropic materials. “No prescribed symmetry group”
   and “unrestricted anisotropic representability” are different claims.
   Wording in the manuscript's introduction/table caption and the
   `sec:related-gap` discussion should be narrowed accordingly. The old
   numerical results have not been disproved; broad representational claims
   are what this counterexample challenges. The separate legacy spline-tail
   issue also needs a code/proof reconciliation for ICKAN.

For context, the separation between invariant selection and reference-state
normalization is already explicit in [Linden et al., Neural networks meet
hyperelasticity](https://arxiv.org/abs/2302.02403), whose normalization results
are stated for isotropic and transversely isotropic settings. The paired
construction above is derived here for this experiment; no novelty claim or
comprehensive literature-priority claim is made.

## Reproduce and next decision

From `coupon_fe2_paper/06_pann/`, use a new output directory for each run:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 run_enrichment.py --core icnn --seed 5 --output enrichment_results/new_icnn_seed5
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 run_enrichment.py --core ickan --seed 5 --output enrichment_results/new_ickan_seed5
python3 -m unittest -v test_enriched_pann.py
OPENBLAS_NUM_THREADS=2 python3 audit_enrichment.py enrichment_results/icnn_seed5 enrichment_results/ickan_seed5 --controls --output enrichment_results/replay.json
```

Each run saves `model.pt`, configuration and dataset hash in `manifest.json`,
fit/validation indices, history, and results. The fixed split has **4208 fit
and 742 validation states**, correcting the memo's earlier off-by-one counts.
`load_enriched(path)` restores a model with the existing normalized
`energy_and_stress` interface; physical stress is its output times
`energy_scale/strain_scale`. Tangent requires `energy_scale/strain_scale²`.
Calls need autograd enabled, even for energy-only evaluation.

`enrichment_results/audit_all.json` is the authoritative replayed evaluation.
The initial seed-5 checkpoint metadata predates the finite-probe-label mask
and contains NaN probe metrics; its weights are sound and unchanged. The
sidecar reports have been refreshed from the saved weights using that mask.
Seeds 6/7 used the corrected evaluator from the start.

This is a completed **first extension and validation experiment**, not a
production FE² deployment or a complete solution of the accuracy problem.
The remaining ~6.46% interpolation and ~15.35% extrapolation error are material.
Before deploying, improve or explicitly accept those constitutive errors,
check tangents over the intended loading paths, and run a small coupon
comparison against the FOM. Further representation changes should be screened
on fitting/validation data; do not repeatedly tune against the test/probe set.
