2D MAW-ECM clean base (minimal files)
=====================================

This folder keeps only the minimum needed to start from:
1) structured parameter mesh in (Gx_macro, Gxy_macro),
2) two trajectories that cover that mesh,
3) FOM runs using those trajectories.

Quick start
-----------

1) Generate stage-0 trajectories (structured mesh + 2 paths):

python3 stage0_training_trajectory.py \
  --out-dir stage_0_trajectory \
  --gx-min -0.10 --gx-max 0.42 \
  --gxy-min -0.05 --gxy-max 0.05 \
  --n-gx 41 --n-gxy 11 \
  --mapping green_lagrange_upper \
  --include-origin 1

2) Run FOM training over both trajectories:

python3 stage1_training_fom_solver_rve.py \
  --stage0-file stage_0_trajectory/stage_0_trajectories.npz \
  --which both \
  --out-dir stage_1_training_set_fom

Notes
-----
- Stage-0 bundle keeps compatibility with stage1/fom by exporting:
  trajectory_1, trajectory_2 as [Exx, Eyy, Gxy].
- Parametric metadata and mesh connectivity are also saved in the same NPZ.
- Compression is intentionally limited by default (`gx-min=-0.10`) to avoid
  non-convergent heavily-compressed states observed with symmetric ranges.
