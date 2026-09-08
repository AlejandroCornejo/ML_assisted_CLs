# Second-stage training: learned polyconvex features and independent volume terms

2026-09-05. This extends `ENRICHMENT_AUDIT.md`; it does not alter the shared
manuscript/model code or overwrite any earlier checkpoint. Both selected
models now achieve approximately 0.65% held-out test stress error.

## Final measured results

Selection was locked by validation stress error, not test/probe performance.
The selected artifacts are:

- ICNN: `enrichment_results/flex_icnn_dynamic32_seed16/model.pt`.
- C2 spline-ICKAN: `enrichment_results/flex_ickan_hat32/model.pt`.
- Selection, all candidate scores, hashes, and audits:
  `enrichment_results/selected_models.json`.
- Full independent audit: `enrichment_results/selected_models_audit.json`.

Relative L2 errors, reported as percentages:

| Model | Validation stress | Test stress | Test energy | Probe stress |
|---|---:|---:|---:|---:|
| New ICNN, 32 learned features, widths 24/24 | 0.7503% | 0.6491% | 0.2169% | 7.0582% |
| New spline-ICKAN, 32 learned features, widths 12/12 | 0.7612% | 0.6551% | 0.2148% | 7.0227% |
| Saved original ICNN | — | 22.7807% | — | 28.8681% |
| Saved original ICKAN | — | 18.3262% | — | 22.8851% |
| Saved free energy MLP | — | 1.2643% | — | 5.5024% |

The new models use different features, normalization, training objectives,
and budgets; this is **not a compute-matched architecture ranking**. The two
winning candidates used seed 16, 2600 full-batch Adam steps and up to 400
L-BFGS iterations. A second 32-feature ICNN run reached 0.7750% validation
stress error; the 16-feature ICNN and spline-ICKAN variants reached 0.9348%
and 0.9380%. This provides supporting repeatability evidence across variants,
not a completed multi-seed statistical comparison of one frozen recipe.

Selected reference tangent errors against the Stage-00 C0 are 1.735% and
1.773%. Reference energy is zero and residual reference stress is about
5e-8 Pa (roundoff). Test component-wise stress errors `[S11,S22,S12]` are
`[0.409%,0.785%,1.851%]` for ICNN and `[0.422%,0.797%,1.827%]` for ICKAN.
The 95th-percentile per-state stress errors are 1.97% and 2.09%; worst test
states are 5.53% and 5.78%. Global relative L2 accuracy is not a uniform
pointwise error bound.

Both selected weights pass the global nonnegative-energy sufficient bound
described below, with conservative finite-interval margins 17.25 and 16.82
in normalized energy units. Both pass energy gradcheck/gradgradcheck and
6000 broad admissible energy samples (principal stretches 0.25 to 3), with
finite nonnegative energies. Their 2400 sampled rank-one directional
curvatures are positive, with minima approximately 238.21 and 237.80 MPa.
The source unit suite has **12 passing tests**, including physical-unit
tangents and the optional analytic stress implementation. Sampling is an
implementation audit, not the source of the polyconvexity certificate.

### Extrapolation remains a limitation

The outer probe uses 345 finite FOM labels; five failed FOM labels are excluded
by the same finite-label mask used for the saved baselines. Probe error by
overshoot ring is:

| Ring factor | ICNN stress error | ICKAN stress error |
|---|---:|---:|
| 1.05 | 1.554% | 1.495% |
| 1.10 | 1.644% | 1.575% |
| 1.25 | 2.872% | 2.805% |
| 1.50 | 5.087% | 4.963% |
| 2.00 | 12.258% | 12.244% |

Ring factors scale distances from the sampling-box center in selected
components; they are not literal coupon load multipliers. Worst individual
probe states reach about 44–46% stress error. Neither model is validated for
unrestricted extrapolation. No FE2 coupon simulation has yet been run with
these checkpoints. The natural next step is a small FOM-vs-surrogate coupon
comparison, including actual strain-domain coverage and Newton behaviour.

The constitutive-training objective is therefore achieved for this dataset;
production FE2 readiness is a separate validation step.

## What changed, and what did not

We retain a potential, objective material features, convex nondecreasing
ICNN/ICKAN cores, an exactly normalized reference state, polyconvexity, and
volumetric growth. Three restrictions are relaxed:

1. The correction is affine in J, `-r(J-1)`, instead of `-r log J`.
   Affine functions preserve convexity for either sign of r. An independent
   positive coefficient alpha multiplies `J-1-log J`. Reference normalization
   no longer fixes the logarithmic volumetric stiffness.
2. Feature directions, powers, and determinant exponents can be trained.
   Constraints are enforced by parametrization, not by a penalty.
3. The final spline candidate uses integrated triangular curvature functions.
   The previous positive cubic-hinge basis forced nondecreasing curvature;
   convexity requires only nonnegative curvature. The new basis allows it to
   rise and fall and uses exact linear tails, without secant joins.

These are **new constitutive variants**, not a learning-rate fix for the
unchanged original fifteen features. The KAN variant is local C2 spline code,
not the external legacy B-spline implementation. No claim that either old or
new feature family is universal is made.

The unchanged fitting split has 4208 samples, validation 742 (split seed 5).
Candidate selection/scaling use fitting data plus the separately available
Stage-00 reference stiffness. Test and probe labels are withheld during
second-stage training and selection. The candidate dictionary has no fitted
angles initially; positive linear NNLS selects useful features, reference
tangent NNLS adds locally useful features, and fit-only pivoted QR adds
response diversity. Width, initialization, and feature count are then tested
using validation scores.

The objective is relative stress MSE + 0.2 relative energy MSE + 0.02 relative
reference-tangent MSE in the learned-feature runs. Checkpoint selection uses
validation stress error. C0 comes from the separate coarse periodic
calculation, not an exact production-mesh tangent.

## Constitutive construction

For perpendicular unit directions d and e, each feature is

    z_i = |Fd|^(2p)/(p J^b) + |Fe|^(2q)/(q J^c).

The trainable parametrization enforces

    p,q = 1/2 + softplus(raw_power),
    b = (2p-1) sigmoid(raw_b),  c = (2q-1) sigmoid(raw_c).

Each term is convex in independent `(F,J)` because its norm exponent a and
determinant exponent b satisfy `a>=1`, `0<=b<=a-1`. Constant centering and
positive scaling preserve this fact. If centering is dynamic, the current
reference feature value `1/p+1/q` is subtracted; this is still constant with
respect to strain for every frozen parameter set.

Write H for the convex nondecreasing core in unscaled features, and h_i for
its nonnegative derivative at the reference. The reference feature gradient
is `rho_i [1,1,0]`, where `rho_i=2-b/p-c/q`, now allowed to have either sign.
Set `r=sum_i h_i rho_i`. The normalized energy is

    W = H(z)-H(z0)-r(J-1)
        + alpha (J-1-log J) + beta/2 (J-1)^2
        + epsilon (tr C/2-1-log J).

Alpha, beta, and epsilon are strictly positive (positive parametrization and
a small floor). The reference terms give `W(I)=0` and `S(I)=0`. Convexity in
independent `(F,J)`, and therefore 2D polyconvexity, follows term by term:
convex monotone composition, affine J, and convex volume/growth terms.
Objectivity follows from dependence on C and J. The gradient of W is the
stress, and its Hessian gives a symmetric tangent. Growth at `J->0+` and
`J->infinity` comes from the independent logarithmic/quadratic terms;
epsilon also supplies growth in norm(F) at bounded J. This is not a full
existence theorem or a guarantee of a nonlinear FE solve converging.

### A posteriori global nonnegative-energy certificate

Nonnegative W does **not** follow from affine reference normalization alone.
`audit_flexible.py` checks an additional sufficient condition for each saved
candidate, not for every possible optimizer iterate.

Define `k_i=1/p_i+1/q_i` and `eta_i=rho_i/k_i`. Since
`t=|Fd|²`, `u=|Fe|²` obey `t*u>=J²`, minimizing each feature at fixed J gives

    z_i >= k_i J^eta_i.

Using the supporting plane of H, and `tr C/2>=J`, bounds W below by

    g(J) = sum_i h_i k_i [J^eta_i-1-eta_i(J-1)]
           + (alpha+epsilon)(J-1-log J) + beta/2 (J-1)^2.

Here `g(1)=g'(1)=0`, and

    J² g''(J) = alpha+epsilon+beta J² + sum_i c_i J^eta_i,
    c_i = h_i k_i eta_i(eta_i-1).

Negative c_i can only occur for `0<eta_i<1`. Let N be their total absolute
coefficient. Two simple sufficient conditions for `g''>=0` on all J>0 are

    alpha+epsilon + sum_{eta_i<0} c_i - N >= 0,
    beta + sum_{eta_i>=1} c_i - N >= 0.

The first covers `0<J<=1`, the second `J>=1`. If these simple conditions fail,
the audit uses conservative power bounds on finite J intervals and analytic
bounds on both infinite tails. Unlike random sampling, passing those
sufficient inequalities establishes the lower-bound argument for all J>0.
The inequalities are evaluated in floating point, with their positive margins
reported; this is not machine-checked interval arithmetic. A failed bound
does not by itself show W is negative.

### The spline core

The final spline basis is the double integral of a nonnegative triangular
hat of width 2h centered on knot k. With `s=(x-k)/h`, its value divided by h²
is zero for `s<=-1`, `(s+1)^3/6` for `-1<s<=0`,
`(-s^3+3s^2+3s+1)/6` for `0<s<1`, and s for `s>=1`.
Each edge is a positive linear slope plus a positive sum of these functions
and an unrestricted bias. First and second derivatives are nonnegative,
second derivatives are continuous, and curvature can decrease back to zero.
Both tail slopes match analytically. Positive layer compositions/sums preserve
convexity and monotonicity.

## Reproduction and artifact conventions

From this directory, each run requires a new output directory. Example:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 train_flexible.py --core icnn --features 32 --learn-features --init calibrated --epochs 2600 --tangent-weight 0.02 --seed 16 --output enrichment_results/new_icnn
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 train_flexible.py --core ickan --features 32 --widths 12,12 --spline-basis integrated_hat --learn-features --init calibrated --epochs 2600 --tangent-weight 0.02 --seed 16 --output enrichment_results/new_ickan
python3 -m unittest -v test_enriched_pann.py test_flexible_pann.py
```

Runs save the dataset hash, arguments, configuration, exact split, best
checkpoint, final selected checkpoint, history and validation results.
`--evaluate` is off by default. Only after selecting candidates should
`audit_flexible.py --evaluate` reveal their test/probe performance.
`--resume model.pt` continues a candidate into a new output directory.

The trainer evolved during these experiments. Early manifests omit newer
options; loader defaults preserve their meaning (`dynamic_center=False`,
`spline_basis='cubic_hinge'`, `analytic_stress=False`). Later runs record these
options explicitly. Source defaults now select the integrated-hat basis and
dynamic centering. The analytic stress path is the exact chain rule, verified
against energy autograd including parameter gradients; it reduces training
overhead but does not change the constitutive ansatz. No exact bitwise
reproduction across software versions/platforms is promised.

Use `load_flexible(path)` to restore a candidate. The model accepts normalized
engineering Green strain `[E11,E22,gamma12]/strain_scale` and emits normalized
energy/stress. Multiply energy by `energy_scale`, stress by
`energy_scale/strain_scale`, and the normalized stress Jacobian by
`energy_scale/strain_scale²`. Keep autograd enabled. Original checkpoint
loaders do not understand this new model family.

For a physical-units query:

```python
from flexible_pann import load_flexible, physical_response

model, checkpoint = load_flexible(
    'enrichment_results/flex_icnn_dynamic32_seed16/model.pt'
)
response = physical_response(
    model, checkpoint, [[0.03, -0.01, 0.02]], tangent=True
)
# response['energy']: (n,), response['stress']: (n,3),
# response['tangent']: (n,3,3), all in Pa; input strain is dimensionless.
```

`finalize_flexible.py` selects the completed flex_* candidates using validation
alone and then reruns their audit/evaluation. An interface-resume smoke test
in `enrichment_results/resume_interface_smoke` reproduces the selected ICNN
metrics with the faster analytic chain rule and no optimization steps.
