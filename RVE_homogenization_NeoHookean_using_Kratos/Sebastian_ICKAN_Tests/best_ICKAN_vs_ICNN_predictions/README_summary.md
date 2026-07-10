# Best ICKAN vs ICNN Predictions

This folder keeps only the two prediction sets we want to present:

- `ICKAN_best_prediction/`: best ICKAN result.
- `ICNN_best_prediction/`: best ICNN result.
- `checkpoints/`: checkpoints for those two models.

Extra ICNN diagnostic cases were moved to:

```text
_archived_extra_icnn_cases_20260709/
```

The original Alejandro scripts in the repository root were not modified.

## Test

The diagnostic test is:

```text
train on trajectory 3
predict on trajectory 3
```

This is a reproduction test, not yet a full generalization test.

## Model Input

Both selected models use:

```text
strain source: applied_strain
input mode: principal
```

The raw macro input is:

```text
[E_xx, E_yy, gamma_xy]
```

From this we build:

```text
E = [[E_xx, gamma_xy/2],
     [gamma_xy/2, E_yy]]
```

```text
C = I + 2E
```

Then:

```text
lambda_i = sqrt(eigenvalue_i(C))
J = sqrt(det(C))
lambda_bar_i = J^(-1/3) lambda_i
```

The selected ICKAN receives:

```text
[lambda_bar_1, lambda_bar_2, log(J)]
```

The selected ICNN receives order-2 principal features:

```text
[lambda_bar_1, lambda_bar_2, lambda_bar_1^2, lambda_bar_2^2, log(J)]
```

These are not exactly the original `K1,K2,K3` ICKAN invariant inputs. Those invariants are used in diagnostic plots, but not as the input of the selected best models.

## ICKAN

Main settings:

- Model: ICKAN.
- Input: principal features from `applied_strain`.
- Trajectory: 3.
- Training samples: 3000.
- Width: `5,4,1`.
- Grid size: `30`.
- Stress loss: blended.
- Component loss weight: `0.10`.
- Energy loss weight: `0.01`.
- Optimizer strategy: Adam warmup; this was key to remove the strange jumps.

Metrics:

| Metric | Value |
|---|---:|
| Global relative L2 | `6.2709e-02` |
| Sxx relative L2 | `5.9046e-02` |
| Syy relative L2 | `6.6090e-02` |
| Sxy relative L2 | `4.4380e-01` |

## ICNN

Main settings:

- Model: ICNN.
- Input: principal features from `applied_strain`.
- Trajectory: 3.
- Training samples: 1000.
- Evaluation: full trajectory.
- Width: `32,32,16`.
- Principal-stretch order: `2`.
- Activation: `softplus`.
- Convexity: non-negative hidden-to-hidden weights.
- Final refinement: global stress loss.
- Final energy loss weight: `0.00`.
- Optimizer strategy: Adam warmup, LBFGS refinement, and low-learning-rate polishing.

Metrics:

| Metric | Value |
|---|---:|
| Global relative L2 | `1.6363e-02` |
| Sxx relative L2 | `1.5782e-02` |
| Syy relative L2 | `1.6793e-02` |
| Sxy relative L2 | `3.1687e-01` |

## Plot Colors

```text
Reference : black
ICKAN     : red
ICNN      : blue
```

Future additional diagnostic cases should use dark green, then orange.

PNG files use transparency to make overlap easier to read. EPS files are also saved, but PostScript does not support transparency.

## Invariant Plots

Invariant diagnostic plots are stored in:

```text
invariant_plots_applied_strain/
invariant_plots_homogenized_strain/
```

The plots are scatter plots to avoid artificial jumps from connecting different branches of the trajectory.

The figures use LaTeX and include the full definitions:

```text
C = I + 2E
I1 = tr(C)
I3 = det(C)
J = sqrt(I3)
I1_bar = I1 I3^(-1/3)
I2_bar = (I1 + I3 - 1) I3^(-2/3)
K1 = I1_bar - 3 = I1 I3^(-1/3) - 3
K2 = I2_bar^(3/2) - 3 sqrt(3)
K3 = (J - 1)^2
```

Important: `K1` is not simply `I1 - 3`; it is `I1_bar - 3`.

## Main Conclusion

```text
ICKAN global L2: 6.27e-02
ICNN global L2 : 1.64e-02
```

The selected ICNN is the best current model for this trajectory-3 reproduction test. Its convexity is with respect to the input features, so this is not by itself a full proof of physical polyconvexity in `F`.
