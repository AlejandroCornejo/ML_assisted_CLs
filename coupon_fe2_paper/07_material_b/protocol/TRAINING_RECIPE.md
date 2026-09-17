# Material B: frozen v1 training preparation

This page documents the frozen feature/model preparation and the historical
fixed-budget pilot. The **single official** stopping rule is in
[`training_recipe.json`](training_recipe.json); do not interpret the pilot
budgets below as its stopping rule.

The [machine-readable recipe](training_recipe_v1.json) completes the numerical
details **after FOM assembly, before shared-feature selection and before any B
neural optimization**. It does not claim those details were all fixed before
FOM generation. The original data protocol and approved labels are unchanged.

The [preparation report](../reports/TRAINING_PREPARATION.md) records the selected
table, initialization checks, hashes, limitations and current next step.

## What is shared, and why

The four constrained variants use one table of 32 paired features,

    z = t^p/(p J^b) + u^q/(q J^c),
    t = d^T C d, u = d_perp^T C d_perp, C = I+2E, J = sqrt(det C).

Each row stores (theta,p,q,b,c); d=(cos theta,sin theta) and d_perp is its
perpendicular unit direction. Bounds p,q>1/2 and 0<b<2p-1, 0<c<2q-1 make
finite learned-parameter initialization possible and imply the non-strict
convexity bounds. These are feature-construction conditions, not empirical
stability findings for the homogenized FOM.

The candidate bank follows the historical 12-angle, five-power grid. Boundary
entries are projected **before selection** to p,q>=0.5001 and coupling ratios
in [0.001,0.999]. Without this step, the existing learned constructor would
shift endpoint entries while the fixed model kept them, spoiling an exactly
shared start. Neither validation nor reserved targets determine this margin.

Selection uses only energy/stress fit labels and the separate reference tangent:

- Nonnegative least squares (NNLS) first fits the weighted energy/stress
  responses of every candidate. Another NNLS fits the reference tangent.
- Retain the union of their feature supports (coefficient threshold 1e-6).
  The explicit overflow rule ranks contributions if there are more than 32.
- Fit-only pivoted QR adds distinct response shapes until exactly 32 entries
  are retained. Its 600 rows are deterministic; analytic volume columns are
  excluded from feature selection.
- Refit nonnegative coefficients on the selected table with the same energy,
  stress and reference-tangent weights as the future neural objective.

The affine response columns are z-z0-rho*(J-1), with z0=1/p+1/q and
rho=2-b/p-c/q, plus (J-1)^2/2 and J-1-log J. Their first derivatives at zero
vanish. The closed-form reference Hessians and energy gradients are checked
against central differences; the NumPy port was also compared with the old
implementation. These numerical checks are not the polyconvexity proof.

This is an informed fixed baseline, not an arbitrary table deliberately left
weak. The finite dictionary and greedy QR heuristic do not establish optimal
features. Both fixed cores use the same selected table; each learned core
starts there and adds five trainable parameters per feature. All core weights
and initialization seeds match within each fixed/learned pair to numerical
precision. Free uses its existing four material-C features, not this embedding.

## Units, scales and reference anchor

Energy W and second Piola stress S are in Pa. The physical strain vector is
e=(E11,E22,2E12), conjugate to s=(S11,S22,S12). Write

    a = max_fit |e|, b = max_fit |W|,
    x=e/a, w=W/b, s=S*a/b, d0=D_reference*a^2/b.

Here D_reference is the derivative of physical stress with respect to physical
engineering strain at zero. Derivatives of normalized energy with respect to x
then have exactly the units/scaling of s and d0.

The objective is

    mean((s_pred-s)^2)/mean_fit(s^2)
    + 0.2*mean((w_pred-w)^2)/mean_fit(w^2)
    + 0.02*mean((d_pred(0)-d0)^2)/mean(d0^2).

All samples/components contribute to each mean. The reference term is one
separate anchor shared by every model; the eight incidental fit tangents are
not training labels. Validation energy/tangents do not select checkpoints.
Metric floors are fit-derived and specified separately in JSON, rather than
using test values to stabilize per-state errors near zero.

Feature scales are the fit maxima of |z-z0| with floor 1e-4, held fixed throughout
training. The centering value follows the current exact reference value when
powers are learned. Reference sensitivities retain their parameter derivatives.
Free's legacy float32 buffer casts are corrected locally to the exact saved
float64 scales; the shared class is not edited.

## Optimization and selection

| Item | Free | ICNN fixed/learned | ICKAN fixed/learned |
|---|---:|---:|---:|
| Hidden widths | 128,128,64 | 24,24 | 12,12 |
| Initial Adam learning rate | 4e-4 | 5e-3 | 5e-3 |
| Initialization seeds | 16,29,47 | 16,29,47 | 16,29,47 |
| Maximum Adam steps | 2600 | 2600 | 2600 |
| Maximum LBFGS internal iterations | 400 | 400 | 400 |

The learning rates preserve the existing Free and flexible-core trainer choices;
they were not selected using B validation/test prediction errors. Different
core widths and initializations mean that cross-core performance is not a
parameter-matched comparison.

Execution is deterministic CPU float64 with two torch/BLAS threads and full
4,200-state batches. Adam uses betas=(0.9,0.999), eps=1e-8, zero weight decay
and gradient-norm clipping at 50. Validation stress is checked initially and
every ten steps. ReduceLROnPlateau halves the rate after 40 unimproved
validation calls, down to 1e-5; this patience counts checks, not optimizer steps.
There is no early-stopping shortcut.

LBFGS starts from the best Adam checkpoint and uses 40 outer calls, at most ten
internal iterations each, retaining history across calls. Settings: lr=1,
history=50, tolerance_grad=1e-10, tolerance_change=1e-12, strong-Wolfe search,
nominal max_eval=12 per call, no gradient clipping. Actual iterations and
closure evaluations must be reported; 400 is an iteration upper bound, not a
guarantee of 400 completed iterations or a bound on line-search evaluations.

Select the strictly lowest **validation stress** objective across initialization
and both phases, using the frozen fit stress denominator. Ties retain the
earliest checkpoint. Save resumable current state separately from best state;
retain failures rather than changing seeds or silently tuning. Hash-lock all
15 final checkpoints before any test/path predictive evaluation. Report every
seed, mean and sample standard deviation; curve seeds are the median-validation
ones, not the best test runs.

ICNN uses fixed softplus hidden activations and nonnegative linear connections.
ICKAN uses nonnegative integrated-hat connection coefficients, fixed six knots
from -1.2 to 1.2 (spacing 0.48), convex/nondecreasing edge functions and linear
tails. No adaptive knot/grid updates are allowed. The shared initialization
uses the NNLS direct skip and fit-only nonlinear-amplitude calibration. This
calibration is a heuristic relative to the affine baseline, not a curvature
bound; analytic coefficient floors can also affect the initial response.

## Reproduce preparation, not training

The existing `/home/sares/.venv-rom/bin/python` supplies torch 2.14.0+cpu,
NumPy 2.5.3 and SciPy 1.18.1 on Python 3.12. The FOM-only `.venv_fe2` does not
contain torch. No package installation was needed. From the repository root,
use new output paths:

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PYTHONPATH=coupon_fe2_paper/.pydeps:coupon_fe2_paper/07_material_b \
    /home/sares/.venv-rom/bin/python -B -m protocol.select_features \
      --out coupon_fe2_paper/07_material_b/work/features_reproduce

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PYTHONPATH=coupon_fe2_paper/.pydeps:coupon_fe2_paper/07_material_b \
    /home/sares/.venv-rom/bin/python -B -m protocol.training_setup \
      --features coupon_fe2_paper/07_material_b/work/features_reproduce \
      --out coupon_fe2_paper/07_material_b/work/initialization_reproduce.json

`select_features` has no torch/FOM import and requests exactly seven allowed NPZ
arrays through lazy access. `training_setup` constructs the declared models,
evaluates fit/reference losses and backpropagates diagnostic gradients, but
never creates or steps an optimizer and never selects a checkpoint.

The B-specific historical pilot runner is `train_material_b.py`; current
official runs use `train_material_b_official.py`. The historical A trainers
must not be called: their
data paths, splits and scaling do not implement this B protocol.
