# Best ICKAN vs ICNN Predictions

This folder collects the best prediction folders obtained so far for the trajectory-3 reproduction test:

- `ICKAN_best_prediction/`: best ICKAN result.
- `ICNN_best_prediction/`: best ICNN result.

The original Alejandro scripts in the repository root were not modified. All changes and experiments were done inside `Sebastian_ICKAN_Tests`.

## Problem Tested

We used the stage-1 FOM data from:

```text
stage_1_training_set_fom
```

The reduced diagnostic test was not full generalization over all trajectories. It was first a controlled reproduction test:

```text
train on trajectory 3
predict on trajectory 3
```

This was intentional. Before asking the model to generalize to new loading paths, we wanted to check whether the constitutive surrogate can reproduce one known path.

## Model Input

The best experiments used:

```text
strain source: applied_strain
input mode: principal
```

The raw available macro input is:

```text
[E_xx, E_yy, gamma_xy]
```

From that, the code builds the right Cauchy-Green tensor:

```text
C = I + 2E
```

Then it computes the principal stretches from the eigenvalues of `C`:

```text
lambda_i = sqrt(eigenvalue_i(C))
```

For `order_stretches = 1`, the actual neural-network input is essentially:

```text
[lambda_bar_1, lambda_bar_2, log(J)]
```

where:

```text
J = sqrt(det(C))
lambda_bar_i = J^(-1/3) lambda_i
```

So the network does not receive the raw `E_xx, E_yy, gamma_xy` directly in the best case. It receives deformation features derived from principal stretches and volume change.

## Best ICKAN Setup

The best ICKAN prediction copied here came from:

```text
Sebastian_ICKAN_Tests/ICKAN_prediction_traj3_3000_appliedstrain_principal_grid30_width541_blended010_W001_adam600_poststep
```

Main settings:

- Model: ICKAN.
- Input: principal features from `applied_strain`.
- Trajectory: 3.
- Samples used for training: 3000.
- Width: `5,4,1`.
- Grid size: `30`.
- Stress loss: blended.
- Component-balanced stress weight: `0.10`.
- Energy loss weight: `0.01`.
- Optimization: Adam warmup was the important stabilizing change; previous pure LBFGS runs were much more fragile.

Best ICKAN metrics:

| Metric | Value |
|---|---:|
| Global relative L2 | `6.2709e-02` |
| Sxx relative L2 | `5.9046e-02` |
| Syy relative L2 | `6.6090e-02` |
| Sxy relative L2 | `4.4380e-01` |

## Best ICNN Setup

The best ICNN prediction copied here came from:

```text
Sebastian_ICKAN_Tests/ICNN_prediction_traj3_full_from1000resume_appliedstrain_principal_width323216_blended010_W001_lowLR
```

Main settings:

- Model: ICNN.
- Input: principal features from `applied_strain`.
- Trajectory: 3.
- Samples used for training: 1000.
- Width: `32,32,16`.
- Activation: smooth `softplus`.
- Convexity: imposed with non-negative hidden-to-hidden weights.
- Stress loss: blended.
- Component-balanced stress weight: `0.10`.
- Energy loss weight: `0.01`.
- Optimization:
  - Adam warmup to get into a good basin.
  - LBFGS refinement.
  - Resume at lower learning rate for final polishing.

Best ICNN metrics:

| Metric | Value |
|---|---:|
| Global relative L2 | `2.6145e-02` |
| Sxx relative L2 | `2.7186e-02` |
| Syy relative L2 | `2.5043e-02` |
| Sxy relative L2 | `1.6228e-01` |

## What Changed and Why It Helped

The first ICKAN trials were unstable because the spline grid and LBFGS optimization were sensitive. Grid updates could produce jumps or NaNs, and pure stress fitting often plateaued early.

The important changes were:

- Use `applied_strain` consistently as the model input, because it is the controlled macro loading parameter.
- Plot both `applied_strain` and homogenized `strain`, because the model input and the postprocessed RVE response are not identical.
- Use principal-stretch features instead of raw strain for the best models.
- Use Adam warmup before LBFGS. This made the optimization much less brittle.
- Use a blended stress loss:

```text
L_stress = (1 - alpha) L_global + alpha L_component
```

with:

```text
alpha = 0.10
```

This keeps Sxx/Syy dominant enough, but prevents Sxy from being completely ignored.

- Add a small energy loss:

```text
L_total = L_stress + beta L_energy
beta = 0.01
```

This helps keep the learned energy closer to the reference energy, without letting energy dominate the stress fit.

## Main Conclusion

For this trajectory-3 reproduction test, the ICNN is clearly better than the ICKAN:

```text
ICKAN global L2: 6.27e-02
ICNN global L2 : 2.61e-02
```

The ICNN also improves all stress components. Sxy remains the hardest component because its physical scale is much smaller than Sxx and Syy, so small absolute errors become large relative errors.

Important theoretical note: the ICNN guarantees convexity with respect to the inputs given to the network. Since the inputs are principal-stretch features, this is not automatically a full proof of physical polyconvexity in F. It is, however, a controlled convex energy model and worked better in this numerical reproduction test.
