# Stage 02 results: the sampling design

## Why structured, and not Sobol

An earlier draft called for Sobol sampling, on the correct but insufficient
grounds that the RVE is path-independent so a point cloud suffices for the
constitutive surrogates. The hyperreduction machinery imposes structure that
Sobol would break. Three independent requirements, all pointing the same way:

1. **Graph-Laplacian regularization of the adaptive weights.** MAW-ECM solves
   for the weight fields jointly over all sampled states, with a Laplacian
   term coupling NEIGHBOURING states; Hernandez demonstrates the accuracy
   degradation without it (alpha_G = 0 against 10^4). Uniformly spaced
   neighbourhood relations are therefore needed, and a grid also permits the
   direct graphical inspection of weight fields he cites as his second reason
   for restricting to a rectangle.
2. **Identification of the decoder's latent coordinates.** The factorized
   decoder is `d_red = Phi_M A_M q_M + Phi_S N_slave(q_M)` with
   `q_M = T_m Phi_ROM^T d_red`, `T_m` identified input-informed. That needs a
   manifold covered densely and uniformly INDEXED BY mu. Latent dimension
   follows the parameter dimension: **r_D = 3** here, against 2 in Hernandez's
   benchmark.
3. **Conditioning of the affine mu -> q_p least-squares map**, the object that
   distinguishes the two ANN variants (see below).

## A box, after trying a cone

The pre-pass cloud is a thin bundle about a dominant ray whose deviations
scale linearly with load, so a load-normalized "cone" fits it much more
tightly: measured 20^3-histogram occupancy is 9.1% for the raw box against
17.8% normalized. A cone grid was built, and then rejected, because it lost
more than it gained:

| | box | cone |
|---|---|---|
| conditioning of the affine mu -> q_p fit | **23.4** | 57.1 |
| cell size | uniform | grows with the load |
| axis | none needed | a ray fitted to the too-stiff pre-pass material |
| "outside the domain" | unambiguous | needed three separate directions |
| samples in the visited region | ~19% | 100% |

The first three lines are each one of the requirements that motivated
structure in the first place. Conditioning is exactly what requirement (c) is
about; non-uniform cells make requirement (a)'s graph Laplacian anisotropic,
which is precisely what Hernandez's uniform Cartesian grid avoids; and the
cone's axis depends on a fit to a material whose stress is over-predicted by
19.7% to 109.6%. The cone's only advantage was density, and density is buyable
here -- samples are cheap warm-started RVE solves.

## The grid

Bounds: cloud bounds widened so each component's total SPAN grows by
(1 + margin), i.e. HALF the margin on each side. Applying the full margin to
both sides widens the span by 1 + 2*margin = 1.8 for a stated margin of 0.4,
which is not what "40% margin" means; that error was made first and caught by
looking at the figure.

E11 is clipped at zero. The coupon is loaded in tension, and admitting macro
compression would put the cell near pore buckling, where its converged state
is no longer a single branch and the warm-start exactness argument fails.

Resolution from a uniform ABSOLUTE step, so the graph Laplacian is isotropic.
Each span is snapped to an exact multiple of the step, so the spacing is
equal by construction rather than to within rounding -- integer node counts
otherwise leave an 8% spread.

| | domain | measured cloud | step |
|---|---|---|---|
| E11 | [+0.00000, +0.19600] | [+0.00566, +0.16212] | 0.01150 |
| E22 | [-0.09865, +0.01635] | [-0.08031, -0.00200] | 0.01150 |
| g12 | [-0.16807, +0.10793] | [-0.12639, +0.06625] | 0.01150 |

**18 x 11 x 25 = 4950 states plus the zero state = 4951**, comparable to the
4851 of Hernandez's 2D benchmark.

Acceptance: contains 100% of the cloud; affine mu-fit design matrix conditions
at 27.7; spacing isotropic to 1.0000; no duplicates; zero state present;
tension only.

## The affine mu -> q_p map, and what it demands

In the code the object is `qp_init_mu_affine`, and `q_p = [mu, 1] @ b_aff`.
The same object plays two different roles, and that IS the distinction between
the variants:

| | role of the affine mu -> q_p map |
|---|---|
| HPROM-ANN (iterative) | INITIALIZER; Newton then solves the projected equilibrium for q_p |
| D-HPROM-ANN (direct) | the ANSWER; there is no equilibrium solve at all |

So D-HPROM-ANN's accuracy is bounded by how affine `q_p` really is in mu --
a structural assumption of the method, not a data-quantity issue. If `q_p` is
not near-affine, no amount of data removes the error floor, and that is a
reportable finding rather than a bug.

Why it tends to hold: at moderate strain the micro fluctuation responds almost
linearly to the macro strain, so `q_p ~ A mu` with a small correction, and
`N_ANN(q_p)` absorbs the nonlinearity in the slave coordinates. That is an
empirical property of the moderate regime, not a guarantee.

Stage 04 therefore measures the affine fit's residual BEFORE any tier
comparison, under BOTH snapshot conventions -- POD on the fluctuation (this
project's `hprom_solver_rve`, `u_aff_free + phi_f q`) and POD with the affine
part retained (Hernandez's, where `q_p` is dominated by a term exactly linear
in mu). The convention is chosen from the measured residual.

## Evaluation sets, declared before any training

Seed `20260903`, recorded. Neither set is ever used to select a model, a
hyperparameter or a mesh.

**Test set**: 400 states uniform in the box, nudged at least a quarter cell
off every training node -- a held-out set placed on grid nodes would measure
interpolation at points already seen. Verified 400/400 inside the box.
Reported check:   [OK  ] test set off the training nodes  (min distance 5.365e-03 vs step 1.150e-02)

**Probe set (problem 1b)**: 300 states, 4 overshoot rings (1.1x, 1.25x, 1.5x,
2.0x) x 3 directions x 25, overshoot measured in box half-spans from the box
centre. Verified 0 of 300 leaked inside the box.

One direction overshoots at a time, so a failure can be ATTRIBUTED rather than
merely observed -- reportable as "tier X holds to 1.5x in load but fails at
1.2x in shear": `load` (E11 out), `shear` (g12 out), `transverse` (E22 out).
E11 stays >= 0 throughout, for the same pore-buckling reason as the box.

Note that in 2D PROJECTIONS some probe points appear inside the box: a point
overshooting in E22 has E11 and g12 inside, so it lies inside the E11-g12
projection while being outside in the third dimension. The containment test is
done in 3D.

**Stage 03 must confirm every train, test and probe state actually solves** --
that becomes one of its acceptance tests. All are far inside the RVE's
measured usable range (E11 >= 1.70), but the probe's shear levels exceed
anything tested on the cell so far.

Why the probe matters, restated because it is easy to lose: once training is
designed for the problem, every tier becomes accurate INSIDE the envelope,
including the uncertified ones, so in-domain accuracy cannot discriminate.
As'ad's failures come from outside the training subdomain and from stability.
Hernandez independently predicts a failure mode in this region, noting the
weight regression guarantees neither positivity nor volume preservation
"particularly outside the convex hull of the training data".
