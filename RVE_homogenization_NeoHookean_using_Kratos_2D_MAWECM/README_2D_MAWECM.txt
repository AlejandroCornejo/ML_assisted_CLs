2D MAW-ECM / PROM-GPR / HPROM-MAWECM-GPR
==========================================

This directory contains the 2D RVE workflow used to compare:

1. FOM: full-order Kratos RVE simulation.
2. PROM-GPR: master/slave PROM where q_s(q_m) is predicted with sparse GPR.
3. HPROM-GPR: classical ECM hyperreduction.
4. HPROM-MAWECM-GPR: adaptive residual ECM weights over the q_m manifold.

Current recommended Stage8 setup
--------------------------------

The current working configuration is:

- q_m -> q_s: sparse anisotropic GPR trained in Stage4.
- residual: MAW-ECM adaptive weights w_res(q_m).
- homogenization: separate classical ECM rules for eps and sig.
- online residual weights: RBF in q_m, evaluated every Newton iteration.
- online tangent: include analytic d(w_res)/d(q_m), with the corrected solver sign.
- graph regularization: optional Stage8b phase 2. Recommended default is phase 1 first
  (`--smooth-laplacian-all-iterations 0`); advanced runs may force phase 2 from
  the first elimination with `--smooth-laplacian-all-iterations 1`.

Use this pair for the recommended MAW + ECM-hom case:

```bash
--maw-mode res_only
--hom-source ecm_separate
```

This means:

- Stage8b first computes a classical residual ECM bootstrap zINI/wINI.
- Stage8b runs MAW-ECM only on the residual channel.
- Stage8b also computes separate classical ECM rules for eps and sig.
- Stage8 online uses MAW for residual and fixed ECM for homogenized eps/sig.

Important implementation notes
------------------------------

MAW-RBF weights are global RBF functions in q_m:

```text
w_e(q_m) = scale_e * (sum_i alpha_{i,e} phi_i(q_m) + p(q_m)^T beta_e)
```

The derivative d(w_res)/d(q_m) is analytic, not finite difference. It is computed in:

```text
mawecm_rbf_weights.py::eval_mawecm_rbf_with_jacobian
```

The kNN option is not used for online RBF interpolation. It is only used to build a graph Laplacian for Stage8b phase-2 regularization when:

```bash
--use-global-graph-2ndstage 1 --graph-mode knn
```

The corrected MAW tangent sign is important. The solver convention is:

```text
R = rhs = -f_int
K_alg * dq = R
K_alg = -dR/dq
```

Therefore the adaptive-weight tangent contribution must enter as:

```text
K_red <- J_u^T K_ff J_u - J_u^T (sum_e R_e tensor dw_e/dq_m)
```

This is implemented in:

```text
hprom_mawecm_gpr_solver_rve.py
hprom_mawecm_rbf_solver_rve.py
```

With the corrected sign, use:

```bash
--hprom-update-maw-each-iter 1
--hprom-include-weight-tangent 1
```

No-fallback policy
------------------

For Stage8, prefer explicit modes. Avoid silent conceptual mixing:

- Use `--hom-source full_mesh` only when you intentionally want full-mesh homogenization.
- Use `--hom-source ecm_separate` when you want eps/sig classical ECM in Stage8b.
- Use `--hom-source fixed_ecm` only when you intentionally provide a previous ECM file.
- Use `--hprom-homogenization-mode full_fom` only for residual-only MAW diagnostics.
- Use `--hprom-homogenization-mode ecm_separate` when eps/sig should use fixed ECM weights.

Full pipeline commands
----------------------

Run from this directory:

```bash
cd /home/kratos/ML_assisted_CLs_clean/RVE_homogenization_NeoHookean_using_Kratos_2D_MAWECM
```

Stage0: structured parameter trajectories
-----------------------------------------

```bash
python3 stage0_training_trajectory.py \
  --out-dir stage_0_trajectory \
  --gx-min -0.10 --gx-max 0.42 \
  --gxy-min -0.05 --gxy-max 0.05 \
  --n-gx 41 --n-gxy 11 \
  --mapping green_lagrange_upper \
  --include-origin 1
```

Stage1: FOM training set
------------------------

```bash
python3 stage1_training_fom_solver_rve.py \
  --stage0-file stage_0_trajectory/stage_0_trajectories.npz \
  --which both \
  --out-dir stage_1_training_set_fom
```

Stage2a: POD basis
------------------

```bash
python3 stage2a_build_pod_from_fom.py \
  --fom-dir stage_1_training_set_fom \
  --out-dir stage_2a_pod_data \
  --mesh rve_geometry \
  --pod-energy-loss 1e-10 \
  --pod-rank 0 \
  --save-w-free 1 \
  --save-plots 1
```

Stage2b: master/slave split
---------------------------

```bash
python3 stage2b_build_ls_master_from_pod.py \
  --pod-dir stage_2a_pod_data \
  --stage0-file stage_0_trajectory/stage_0_trajectories.npz \
  --out-dir stage_2b_ls_master \
  --mu-parametrization gx_gxy \
  --master-dim 2 \
  --save-plots 1
```

Stage3: PROM regression dataset
-------------------------------

```bash
python3 stage3_build_prom_regression_dataset.py \
  --stage2a-dir stage_2a_pod_data \
  --stage2b-dir stage_2b_ls_master \
  --stage0-file stage_0_trajectory/stage_0_trajectories.npz \
  --out-dir stage_3_prom_regression_dataset \
  --target-space both_ms \
  --mu-space gx_gxy \
  --train-frac 0.80 \
  --val-frac 0.10 \
  --seed 42 \
  --save-plots 1
```

Stage4: sparse PROM-GPR q_m -> q_s
----------------------------------

```bash
python3 stage4_train_prom_sparse_gpr.py \
  --dataset-file stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz \
  --out-dir stage_4_prom_gpr_sparse \
  --gpr-input-space q_m \
  --num-inducing 451 \
  --inducing-selection kmeans \
  --kmeans-fit-samples 40000 \
  --epochs 120 \
  --batch-size 2048 \
  --lr 0.05 \
  --min-noise 1e-6 \
  --seed 42
```

Stage5: PROM-GPR online sanity test
-----------------------------------

```bash
python3 stage5_test_prom_gpr_online.py \
  --run-fom \
  --run-prom-gpr \
  --prom-fail-on-nonconvergence 1 \
  --save-plots 1
```

Stage8a: residual and homogenization dataset for MAW-ECM
--------------------------------------------------------

Use `--include-homogenization 1` if Stage8b must compute eps/sig ECM rules.

```bash
python3 stage8a_build_mawecm_res_dataset.py \
  --fom-dir stage_1_training_set_fom \
  --pod-dir stage_2a_pod_data \
  --stage3-dataset-file stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz \
  --stage0-file stage_0_trajectory/stage_0_trajectories.npz \
  --mesh rve_geometry \
  --include-homogenization 1 \
  --out-dir stage_8a_mawecm_res_dataset
```

Stage8b recommended: residual MAW-ECM + eps/sig classical ECM
-------------------------------------------------------------

This is the recommended current target.

```bash
python3 stage8b_build_mawecm_res_model_rbf.py \
  --dataset-dir stage_8a_mawecm_res_dataset \
  --maw-mode res_only \
  --hom-source ecm_separate \
  --maw-min-support-size-res 0 \
  --use-global-graph-2ndstage 1 \
  --smooth-laplacian-all-iterations 0 \
  --alpha-smooth 1e4 \
  --graph-mode structured_grid \
  --max-number-zeros-active-set-loop-maw-ecm 1 \
  --res-bootstrap-constrain-sum-weights 1 \
  --enforce-sum-weights 1 \
  --sum-weights-target 990.0 \
  --tol-rank-rel 1e-14 \
  --save-weight-field-plots 1 \
  --show-weight-field-plots 0 \
  --out-dir stage_8b_hprom_mawecm_res_only_auto_ecmhom_sum990_graph
```

Expected Stage8b log markers:

```text
[Stage8b] Hom bootstrap ECM (eps): ...
[Stage8b] Hom bootstrap ECM (sig): ...
[MAW-ECM][Phase1] start (no regularization): ...
[MAW-ECM][Phase2] starting regularization (...): ...
[MAW] mode=res_only ...
```

Stage8 online recommended test
------------------------------

```bash
python3 stage8_test_hprom_mawecm_gpr_online.py \
  --run-fom \
  --run-prom-gpr \
  --run-hprom-mawecm-gpr \
  --mawecm-file stage_8b_hprom_mawecm_res_only_auto_ecmhom_sum990_graph/ecm_weights_all.npz \
  --hprom-homogenization-mode ecm_separate \
  --hprom-use-hrom-mdpa 0 \
  --hprom-update-maw-each-iter 1 \
  --hprom-include-weight-tangent 1 \
  --hprom-fail-on-nonconvergence 1 \
  --save-plots 1 \
  --out-dir stage_8_online_hprom_mawecm_gpr_res_only_auto_ecmhom_sum990_graph
```

Useful Stage8 variants
----------------------

Residual MAW only, full-mesh homogenization. Use this to isolate residual hyperreduction error:

```bash
python3 stage8b_build_mawecm_res_model_rbf.py \
  --dataset-dir stage_8a_mawecm_res_dataset \
  --maw-mode res_only \
  --hom-source full_mesh \
  --maw-min-support-size-res 0 \
  --use-global-graph-2ndstage 1 \
  --smooth-laplacian-all-iterations 0 \
  --alpha-smooth 1e4 \
  --graph-mode structured_grid \
  --max-number-zeros-active-set-loop-maw-ecm 1 \
  --res-bootstrap-constrain-sum-weights 1 \
  --enforce-sum-weights 1 \
  --sum-weights-target 990.0 \
  --tol-rank-rel 1e-14 \
  --save-weight-field-plots 1 \
  --show-weight-field-plots 0 \
  --out-dir stage_8b_hprom_mawecm_res_only_auto_fullhom_sum990_graph

python3 stage8_test_hprom_mawecm_gpr_online.py \
  --run-fom \
  --run-prom-gpr \
  --run-hprom-mawecm-gpr \
  --mawecm-file stage_8b_hprom_mawecm_res_only_auto_fullhom_sum990_graph/ecm_weights_all.npz \
  --hprom-homogenization-mode full_fom \
  --hprom-use-hrom-mdpa 0 \
  --hprom-update-maw-each-iter 1 \
  --hprom-include-weight-tangent 1 \
  --hprom-fail-on-nonconvergence 1 \
  --save-plots 1 \
  --out-dir stage_8_online_hprom_mawecm_gpr_res_only_auto_fullhom_sum990_graph
```

Phase 1 only, no graph regularization:

```bash
python3 stage8b_build_mawecm_res_model_rbf.py \
  --dataset-dir stage_8a_mawecm_res_dataset \
  --maw-mode res_only \
  --hom-source ecm_separate \
  --maw-min-support-size-res 0 \
  --use-global-graph-2ndstage 0 \
  --smooth-laplacian-all-iterations 0 \
  --max-number-zeros-active-set-loop-maw-ecm 0 \
  --res-bootstrap-constrain-sum-weights 1 \
  --enforce-sum-weights 1 \
  --sum-weights-target 990.0 \
  --tol-rank-rel 1e-14 \
  --save-weight-field-plots 1 \
  --show-weight-field-plots 0 \
  --out-dir stage_8b_hprom_mawecm_res_only_auto_ecmhom_sum990_phase1
```

Fixed residual support size, for accuracy/speed sweeps:

```bash
python3 stage8b_build_mawecm_res_model_rbf.py \
  --dataset-dir stage_8a_mawecm_res_dataset \
  --maw-mode res_only \
  --hom-source ecm_separate \
  --maw-min-support-size-res 100 \
  --use-global-graph-2ndstage 1 \
  --smooth-laplacian-all-iterations 0 \
  --alpha-smooth 1e4 \
  --graph-mode structured_grid \
  --max-number-zeros-active-set-loop-maw-ecm 1 \
  --res-bootstrap-constrain-sum-weights 1 \
  --enforce-sum-weights 1 \
  --sum-weights-target 990.0 \
  --tol-rank-rel 1e-14 \
  --save-weight-field-plots 1 \
  --show-weight-field-plots 0 \
  --out-dir stage_8b_hprom_mawecm_res_only_100_ecmhom_sum990_graph
```

HROM mdpa mode
--------------

The recommended debugging mode is currently:

```bash
--hprom-use-hrom-mdpa 0
```

If you want true reduced mdpa execution, build the HROM mdpa after Stage8b. For `ecm_separate`, the mesh must contain all elements used by residual and homogenization; use `Z_union`.

```bash
python3 stage6c_create_hrom_mdpa.py \
  --base-mesh rve_geometry \
  --ecm-file stage_8b_hprom_mawecm_res_only_auto_ecmhom_sum990_graph/ecm_weights_all.npz \
  --selection-key Z_union \
  --condition-mode all \
  --output-mesh rve_geometry_stage8b_maw_res_only_auto_ecmhom_sum990_graph_hrom \
  --inplace-ecm

python3 stage8_test_hprom_mawecm_gpr_online.py \
  --run-fom \
  --run-prom-gpr \
  --run-hprom-mawecm-gpr \
  --mawecm-file stage_8b_hprom_mawecm_res_only_auto_ecmhom_sum990_graph/ecm_weights_all.npz \
  --hprom-homogenization-mode ecm_separate \
  --hprom-use-hrom-mdpa 1 \
  --hprom-hrom-strict 1 \
  --hprom-update-maw-each-iter 1 \
  --hprom-include-weight-tangent 1 \
  --hprom-fail-on-nonconvergence 1 \
  --save-plots 1 \
  --out-dir stage_8_online_hprom_mawecm_gpr_res_only_auto_ecmhom_sum990_graph_hrom
```

Troubleshooting
---------------

If Stage8b says the homogenization blocks are missing, rebuild Stage8a with:

```bash
--include-homogenization 1
```

If `--hprom-include-weight-tangent 1` converges worse than `0`, check that the corrected tangent sign is present:

```text
k_red = k_red - (du_dqm.T @ drw_dqm)
```

If repeated runs give slightly different Newton counts, isolate randomness:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

For deterministic residual bootstrap, use:

```bash
--res-bootstrap-rsvd-randomized 0
```

If HROM mdpa fails with missing `Z_eps` or `Z_sig`, the reduced mesh was built with the wrong selection key. For `ecm_separate`, rebuild with:

```bash
--selection-key Z_union
```

Generated outputs
-----------------

Typical generated directories are:

```text
stage_0_trajectory/
stage_1_training_set_fom/
stage_2a_pod_data/
stage_2b_ls_master/
stage_3_prom_regression_dataset/
stage_4_prom_gpr_sparse/
stage_8a_mawecm_res_dataset/
stage_8b_hprom_mawecm_*/
stage_8_online_hprom_mawecm_*/
```

These are run products. Keep code/config/docs separate from generated results when committing.
