# Sebastian ICKAN Tests

This folder collects notes and local dependencies for testing Alejandro's ICKAN/KAN
surrogate idea on the Stage 1 FOM RVE database.

## Current dependency status

- The generated FOM data already exists in `../stage_1_training_set_fom`.
- Core Python dependencies are available in the current environment:
  - Python 3.12
  - PyTorch
  - NumPy
  - Matplotlib
- Alejandro's ICKAN repository is available here:

```text
/home/kratos/ICKANs
```

whose remote is:

```text
https://github.com/mmc-group/ICKANs.git
```

- A baseline KAN implementation is also available as `kan`.
- The local baseline KAN source is available here:

```text
/home/kratos/pykan
```

whose remote is:

```text
https://github.com/KindXiaoming/pykan.git
```

## Important distinction

Alejandro's current script imports:

```python
import ickan as KAN
```

The current environment does not install `ickan` globally, but the Sebastian
copy inserts `/home/kratos/ICKANs` into `sys.path` before importing:

```python
import ickan as KAN
```

This means the Sebastian tests now use the real ICKAN source by default.
If `/home/kratos/ICKANs` is unavailable, the copied surrogate falls back to
baseline `pykan` from `/home/kratos/pykan`.

Therefore:

- The default Sebastian workflow uses ICKAN.
- The fallback non-IC KAN path is still useful for comparison, but should not be
  treated as identical to ICKAN.
- The training and prediction scripts print/save the actual backend.

## Do we need to compile anything?

No full repository compilation is needed for the KAN/ICKAN side. The KAN code is
Python. For this part we mainly need:

1. the FOM `.npy` data,
2. PyTorch/NumPy/Matplotlib,
3. `/home/kratos/ICKANs` or baseline `kan`,
4. a script that imports the intended package cleanly.

The Kratos/FOM side would need Kratos installed if we wanted to regenerate the
FOM database, but it is not required just to train a KAN surrogate from the
already-generated Stage 1 arrays.

## Suggested next steps

1. Use `/home/kratos/ICKANs` for ICKAN runs.
2. Keep `/home/kratos/pykan` as the baseline KAN reference implementation.
3. Do not overwrite the original Alejandro script.
4. Add a train/validation/test split before trusting any plotted results.

## Current Sebastian test scripts

The original Alejandro scripts have been copied into this folder:

```text
Sebastian_ICKAN_Tests/ICKAN_surrogate.py
Sebastian_ICKAN_Tests/run_ICKAN_surrogate.py
```

Only these copies should be modified for local tests. The originals in the repo
root should remain untouched.

The copied `run_ICKAN_surrogate.py` is now considered a legacy combined
experiment. It trains, predicts and plots in one pass, so it is too easy to
confuse training fit with actual prediction.

Use the separated workflow instead.

Train a reusable checkpoint:

```bash
python3 Sebastian_ICKAN_Tests/train_ICKAN_surrogate.py \
  --strain-source strain \
  --train-trajectories 1-8 \
  --samples-per-trajectory 500 \
  --input-mode principal \
  --order-stretches 3 \
  --hidden-widths 16,16,8 \
  --optimizer lbfgs \
  --out-dir Sebastian_ICKAN_Tests/ICKAN_training
```

Predict held-out trajectories from that checkpoint:

```bash
python3 Sebastian_ICKAN_Tests/predict_ICKAN_surrogate.py \
  --checkpoint Sebastian_ICKAN_Tests/ICKAN_training/ICKAN_model_checkpoint.pth \
  --trajectories 9-10 \
  --out-dir Sebastian_ICKAN_Tests/ICKAN_prediction
```

For comparison with the Stage 12 HPROM-MAWECM-GPR benchmark, train with all
Stage 1 trajectories and predict on the same Stage 12 FOM path:

```bash
python3 -B Sebastian_ICKAN_Tests/train_ICKAN_surrogate.py \
  --strain-source strain \
  --train-trajectories all \
  --samples-per-trajectory 500 \
  --input-mode principal \
  --order-stretches 3 \
  --hidden-widths 16,16,8 \
  --optimizer lbfgs \
  --epochs 200 \
  --out-dir Sebastian_ICKAN_Tests/ICKAN_training_all_stage1

python3 -B Sebastian_ICKAN_Tests/predict_ICKAN_surrogate.py \
  --checkpoint Sebastian_ICKAN_Tests/ICKAN_training_all_stage1/ICKAN_model_checkpoint.pth \
  --stage12-results-dir bakckup_21_modes/stage_12_hprom_mawecm_gpr_ls_results_phase1_100_sum990_fullhom \
  --out-dir Sebastian_ICKAN_Tests/ICKAN_prediction_stage12
```

The command above uses all Stage 1 trajectories but only 500 samples per
trajectory. For the largest common Stage 1 sample count currently available,
use `--samples-per-trajectory 5042`; this is much heavier because it trains on
50,420 samples.

The `prediction_metrics.json` file reports `relative_l2_global`, which is the
metric directly comparable to Stage 12 entries such as
`stress_rel_hprom_vs_fom`.

The prediction script plots every trajectory as a separate curve. This fixes the
old plotting artifact where the last point of one trajectory was connected to
the first point of the next trajectory, producing artificial return lines.

The training script performs one initial KAN grid update before optimization,
using the actual KAN inputs computed from the training strains. It does **not**
move the spline grid during LBFGS training by default. Repeated grid updates can
destabilize the optimization because they change the spline basis after the
optimizer has already adapted the coefficients. To experiment with the old
behavior, pass `--update-grid-during-training` explicitly.

LBFGS reports one external epoch, but each epoch can run many internal closure
evaluations. The default is `--lbfgs-max-iter 20`, which can look very slow
because stress training requires second-order autograd. For quick diagnostics,
use a smaller value such as `--lbfgs-max-iter 5`.

Input modes:

- `principal`: original principal-stretch/logJ features. This imposes a strong
  invariant/isotropic structure on the learned energy.
- `direct_strain`: raw normalized `[E_xx,E_yy,G_xy]` only.
- `hybrid`: principal-stretch/logJ features plus raw normalized
  `[E_xx,E_yy,G_xy]`. This is mainly a diagnostic mode: it relaxes the symmetry
  of `principal` so we can test whether `S_xx` and `S_yy` are being flattened by
  an over-restrictive input representation.

Direct stress MLP sanity check:

```bash
python3 -u -B Sebastian_ICKAN_Tests/train_direct_stress_mlp_baseline.py \
  --train-trajectories 3 \
  --samples-per-trajectory 1000 \
  --hidden-widths 64,64,64 \
  --activation silu \
  --epochs 5000 \
  --out-dir Sebastian_ICKAN_Tests/MLP_stress_baseline_traj3_overfit1000
```

This baseline is not an energy model and does not enforce hyperelastic
consistency. It is only a data sanity check for whether normalized
`[E_xx,E_yy,G_xy] -> [S_xx,S_yy,S_xy]` is learnable.

Fast smoke test:

```bash
python3 Sebastian_ICKAN_Tests/train_ICKAN_surrogate.py \
  --train-trajectories 1-2 \
  --samples-per-trajectory 3 \
  --epochs 0 \
  --out-dir Sebastian_ICKAN_Tests/ICKAN_training_smoke

python3 Sebastian_ICKAN_Tests/predict_ICKAN_surrogate.py \
  --checkpoint Sebastian_ICKAN_Tests/ICKAN_training_smoke/ICKAN_model_checkpoint.pth \
  --trajectories 3 \
  --samples-per-trajectory 3 \
  --out-dir Sebastian_ICKAN_Tests/ICKAN_prediction_smoke
```

The old copied runner defaults to:

```text
input : ../stage_1_training_set_fom
output: Sebastian_ICKAN_Tests/ICKAN_predictions
```

Relative `--out-dir` values are interpreted inside `Sebastian_ICKAN_Tests`.

Legacy smoke test:

```bash
python3 Sebastian_ICKAN_Tests/run_ICKAN_surrogate.py \
  --min-steps 3 \
  --epochs 0 \
  --skip-kan-plot \
  --out-dir ICKAN_predictions_smoke
```

Legacy ICKAN/KAN run, using computed homogenized strain:

```bash
python3 Sebastian_ICKAN_Tests/run_ICKAN_surrogate.py \
  --strain-source strain \
  --min-steps 500 \
  --out-dir ICKAN_predictions
```

Legacy ICKAN/KAN run, using imposed applied strain:

```bash
python3 Sebastian_ICKAN_Tests/run_ICKAN_surrogate.py \
  --strain-source applied_strain \
  --min-steps 500 \
  --out-dir ICKAN_predictions_applied_strain
```

The copied surrogate prepends `/home/kratos/ICKANs` to `sys.path`, then tries
`import ickan` first. If that module is unavailable, it falls back to baseline
`kan` from `/home/kratos/pykan` and prints:

```text
[SEBASTIAN-ICKAN] backend module       : kan
```

That fallback is useful for baseline KAN testing, but it is not proof that we
are reproducing Alejandro's exact ICKAN implementation.

## Note on Alejandro's screenshot of `MultKAN.py`

Alejandro showed changes inside an `ickan/MultKAN.py` file. The local ICKAN
repo now lives at:

```text
/home/kratos/ICKANs
```

In that repo, `MultKAN.__init__()` already calls `self.adjust_init_grid()`.
We patched `/home/kratos/ICKANs/ickan/MultKAN.py` so `adjust_init_grid()` accepts
both global ranges such as `[-1, 1]` and per-input ranges such as
`[[-1, 1], [-1, 1], ...]`.

We did **not** patch `/home/kratos/pykan/kan/MultKAN.py`; `pykan` remains the
baseline fallback.

We verified this with smoke tests:

```bash
python3 Sebastian_ICKAN_Tests/run_ICKAN_surrogate.py \
  --min-steps 3 \
  --epochs 0 \
  --skip-kan-plot \
  --out-dir ICKAN_predictions_smoke

python3 Sebastian_ICKAN_Tests/run_ICKAN_surrogate.py \
  --min-steps 3 \
  --epochs 0 \
  --out-dir ICKAN_predictions_smoke_plot
```

Both passed using backend `kan`. The second command also exercised KAN plotting.

Conclusion: those visible Alejandro edits are not required for the current
Sebastian baseline KAN tests. If Alejandro provides the exact `adjust_init_grid`
function or the full `ickan` repository, then we can apply/reproduce it directly.
