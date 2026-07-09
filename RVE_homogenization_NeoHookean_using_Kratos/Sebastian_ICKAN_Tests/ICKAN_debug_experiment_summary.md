# ICKAN trajectory 3 debug summary

Date: 2026-07-08

Scope:
- FOM data: `stage_1_training_set_fom/trajectory_3`
- Model input strain source: `applied_strain`
- Samples: 3000
- Base architecture unless stated otherwise: `input-mode principal`, `hidden-widths 5,4,1`, `grid-size 30`, `spline-degree 3`, `base_fun silu`
- Optimizer schedule: Adam warmup 600 epochs with cyclic LR; LBFGS omitted when it was observed to be flat.

Main code fixes:
- `Sebastian_ICKAN_Tests/train_ICKAN_surrogate.py`
  - Added `--stress-loss blended`.
  - Added `--component-loss-weight`.
  - Re-evaluates the loss after each optimizer step so the printed/best loss and saved state describe the same parameters.
- `/home/kratos/ICKANs/ickan/spline.py`
  - Avoids NaNs in `curve2coef` when endpoint sample spacing is zero. This matters for `applied_strain`, where the shear input has only a few repeated levels.

Prediction metrics on trajectory 3:

| case | global L2 | Sxx L2 | Syy L2 | Sxy L2 | Sxy RMSE |
|---|---:|---:|---:|---:|---:|
| principal, component, W=1e-2 | 0.07828 | 0.07059 | 0.08521 | 0.36290 | 0.001654 |
| principal, blended alpha=0.10, W=1e-2 | 0.06271 | 0.05905 | 0.06609 | 0.44380 | 0.002023 |
| principal, blended alpha=0.10, W=0 | 0.06263 | 0.05886 | 0.06612 | 0.44603 | 0.002033 |
| principal, blended alpha=0.20, W=1e-2 | 0.06685 | 0.06332 | 0.07014 | 0.40585 | 0.001850 |
| principal, blended alpha=0.25, W=1e-2 | 0.06656 | 0.06179 | 0.07094 | 0.40810 | 0.001860 |
| principal, blended alpha=0.50, W=1e-2 | 0.06641 | 0.06340 | 0.06923 | 0.41987 | 0.001914 |
| direct strain, blended alpha=0.25, W=1e-2 | 0.45111 | 0.45578 | 0.44641 | 0.69901 | 0.003186 |
| hybrid, blended alpha=0.25, W=1e-2 | 0.25236 | 0.26633 | 0.23764 | 0.40573 | 0.001849 |
| orthotropic signed invariants, blended alpha=0.25, W=1e-2 | 0.12063 | 0.13029 | 0.11012 | 0.53571 | 0.002442 |
| principal, grid=60, blended alpha=0.10, W=1e-2, Adam 400 | 0.06308 | 0.05979 | 0.06612 | 0.46280 | 0.002110 |
| principal, resume best + Adam 300 LR=5e-4 | 0.06462 | 0.06118 | 0.06782 | 0.39865 | 0.001817 |

Additional experiments on 2026-07-09:
- Upstream ICKAN invariant inputs were implemented as `ickan_invariants` and `ickan_invariants_linear`, but direct trajectory-3 tests trained much worse than the principal-stretch input. Keep them only as diagnostics for now.
- `principal` with `order-stretches=2` trained much worse (`best_loss=1.965e-01` after 400 Adam epochs), so adding raw squared stretch inputs is not currently helpful.
- `grid=60` improved the 400-epoch training loss compared with grid 30 at the same epoch budget, but did not improve prediction metrics over the 600-epoch grid-30 baseline.
- Constant Adam LR `5e-3` from scratch was stable but worse (`best_loss=1.151e-02`) than the cyclic 600-epoch baseline.
- `--resume-checkpoint` was added to `train_ICKAN_surrogate.py`. Fine-tuning the best checkpoint with LR `5e-4` reduced the training loss (`1.037e-02 -> 9.341e-03`), but shifted the metric tradeoff: Sxy improved while Sxx/Syy and global L2 became slightly worse.

Recommendation:
- Use `principal + blended alpha=0.10 + W=1e-2` if the priority is overall Sxx/Syy/global accuracy.
- Use `principal + component + W=1e-2` if the priority is relative Sxy accuracy.
- Use the resumed LR=5e-4 checkpoint only if improving Sxy is worth a small loss in Sxx/Syy/global L2.
- Do not use `direct_strain`, `hybrid`, or `orthotropic_invariants_signed` for this current trajectory-3 reproduction test without further architecture work.

Suggested current checkpoint:

```text
Sebastian_ICKAN_Tests/ICKAN_training_traj3_3000_appliedstrain_principal_grid30_width541_blended010_W001_adam600_poststep/ICKAN_model_checkpoint.pth
```

Alternative checkpoint if Sxy is prioritized:

```text
Sebastian_ICKAN_Tests/ICKAN_training_traj3_3000_appliedstrain_principal_grid30_width541_blended010_W001_resume_lr5e4_adam300/ICKAN_model_checkpoint.pth
```

Suggested current prediction directory:

```text
Sebastian_ICKAN_Tests/ICKAN_prediction_traj3_3000_appliedstrain_principal_grid30_width541_blended010_W001_adam600_poststep
```

Alternative prediction directory if Sxy is prioritized:

```text
Sebastian_ICKAN_Tests/ICKAN_prediction_traj3_3000_appliedstrain_principal_grid30_width541_blended010_W001_resume_lr5e4_adam300
```
