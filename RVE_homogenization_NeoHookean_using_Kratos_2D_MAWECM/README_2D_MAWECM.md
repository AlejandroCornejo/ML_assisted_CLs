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
- homogenization strain: separate MAW-ECM adaptive weights w_eps(q_m).
- homogenization stress: separate MAW-ECM adaptive weights w_sig(q_m).
- online residual weights: RBF in q_m, evaluated every Newton iteration.
- online homogenization weights: independent RBFs in q_m for eps and sig.
- online tangent: include analytic d(w_res)/d(q_m), with the corrected solver sign.
- graph regularization: channel-dependent Stage8b policy.

Use this pair for the current validated all-adaptive case:

```bash
--maw-mode res_eps_sig
--hom-source ecm_separate
```

This means:

- Stage8b first computes a classical residual ECM bootstrap zINI/wINI.
- Stage8b also computes separate classical ECM bootstraps for eps and sig.
- Stage8b runs three independent MAW-ECM reductions: residual, eps, and sig.
- Residual MAW uses the staged policy: phase 1 first, then graph phase 2.
- Eps/sig MAW use graph phase 2 from the first elimination attempt.
- Stage8 online evaluates three independent RBF weight fields.

The key policy difference is intentional:

```bash
# residual: phase 1 first, graph phase 2 only after phase 1 stalls
--use-global-graph-2ndstage 1
--smooth-laplacian-all-iterations 0

# epsilon/sigma: graph phase 2 from the beginning
--maw-hom-use-global-graph-2ndstage 1
--maw-hom-smooth-laplacian-all-iterations 1
```

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
- Use `--hprom-homogenization-mode maw_separate` only with Stage8b files built with
  `--maw-mode res_eps_sig`; this evaluates separate adaptive MAW-RBF weights for eps
  and sig online.
- Use `--hprom-homogenization-mode sig_maw_eps_ecm` only with Stage8b files built with
  `--maw-mode res_sig`; this keeps eps on fixed classical ECM and evaluates adaptive
  MAW-RBF weights only for sig.

For hom MAW (`eps/sig`), Stage8b enforces:

```text
A_j w_j = b_j,
1^T w_j = 1^T w_ini,
w_j >= 0.
```

This is strict. If hom MAW is active, `--maw-hom-enforce-nonnegativity 0` is rejected.

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

Stage8b recommended: residual/eps/sig MAW-ECM, all separate
------------------------------------------------------------

This is the current validated target. The residual channel first performs phase 1
without regularization and then activates graph phase 2. The homogenization
channels (`eps` and `sig`) use graph phase 2 from the beginning because this was
the robust configuration observed in the RVE tests.

```bash
python3 stage8b_build_mawecm_res_model_rbf.py \
  --dataset-dir stage_8a_mawecm_res_dataset \
  --maw-mode res_eps_sig \
  --hom-source ecm_separate \
  --maw-min-support-size-res 0 \
  --maw-min-support-size-eps 0 \
  --maw-min-support-size-sig 0 \
  --use-global-graph-2ndstage 1 \
  --smooth-laplacian-all-iterations 0 \
  --maw-hom-use-global-graph-2ndstage 1 \
  --maw-hom-smooth-laplacian-all-iterations 1 \
  --alpha-smooth 1e4 \
  --maw-hom-alpha-smooth 1e4 \
  --graph-mode structured_grid \
  --max-number-zeros-active-set-loop-maw-ecm 1 \
  --res-bootstrap-rsvd-randomized 0 \
  --res-bootstrap-constrain-sum-weights 1 \
  --enforce-sum-weights 1 \
  --sum-weights-target 990.0 \
  --tol-rank-rel 1e-14 \
  --maw-hom-enforce-nonnegativity 1 \
  --maw-hom-conservative 1 \
  --maw-hom-cv-max-rel 0 \
  --save-weight-field-plots 1 \
  --show-weight-field-plots 0 \
  --out-dir stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph
```

Expected Stage8b log markers:

```text
[Stage8b] Hom bootstrap ECM (eps): ...
[Stage8b] Hom bootstrap ECM (sig): ...
[MAW-ECM][Phase1] start (no regularization): ...
[MAW-ECM][Phase1] skipped (smooth_laplacian_all_iterations=1): ...
[MAW-ECM][Phase2] starting regularization (...): ...
[MAW] mode=res_eps_sig ...
```

Stage8 online recommended test
------------------------------

```bash
python3 stage8_test_hprom_mawecm_gpr_online.py \
  --run-fom \
  --run-prom-gpr \
  --run-hprom-mawecm-gpr \
  --mawecm-file stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/ecm_weights_all.npz \
  --hprom-homogenization-mode maw_separate \
  --hprom-use-hrom-mdpa 0 \
  --hprom-update-maw-each-iter 1 \
  --hprom-include-weight-tangent 1 \
  --hprom-fail-on-nonconvergence 1 \
  --save-plots 1 \
  --out-dir stage_8_online_hprom_mawecm_gpr_res_eps_sig_auto_ecmhom_sum990_graph
```

Stage8b variant: residual MAW + eps/sig MAW separately
-------------------------------------------------------

This builds three independent MAW rules:

- `Z_res,w_res(q_m)` for the residual.
- `Z_eps,w_eps(q_m)` for homogenized strain.
- `Z_sig,w_sig(q_m)` for homogenized stress.

The hom channels always keep nonnegative weights and preserve their classical ECM
sum of weights.

```bash
python3 stage8b_build_mawecm_res_model_rbf.py \
  --dataset-dir stage_8a_mawecm_res_dataset \
  --maw-mode res_eps_sig \
  --hom-source ecm_separate \
  --maw-min-support-size-res 0 \
  --maw-min-support-size-eps 0 \
  --maw-min-support-size-sig 0 \
  --use-global-graph-2ndstage 1 \
  --smooth-laplacian-all-iterations 0 \
  --maw-hom-use-global-graph-2ndstage 1 \
  --maw-hom-smooth-laplacian-all-iterations 1 \
  --alpha-smooth 1e4 \
  --maw-hom-alpha-smooth 1e4 \
  --graph-mode structured_grid \
  --max-number-zeros-active-set-loop-maw-ecm 1 \
  --res-bootstrap-rsvd-randomized 0 \
  --res-bootstrap-constrain-sum-weights 1 \
  --enforce-sum-weights 1 \
  --sum-weights-target 990.0 \
  --tol-rank-rel 1e-14 \
  --maw-hom-enforce-nonnegativity 1 \
  --maw-hom-conservative 1 \
  --maw-hom-cv-max-rel 0 \
  --save-weight-field-plots 1 \
  --show-weight-field-plots 0 \
  --out-dir stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph
```

Online test with adaptive hom MAW:

```bash
python3 stage8_test_hprom_mawecm_gpr_online.py \
  --run-fom \
  --run-prom-gpr \
  --run-hprom-mawecm-gpr \
  --mawecm-file stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/ecm_weights_all.npz \
  --hprom-homogenization-mode maw_separate \
  --hprom-use-hrom-mdpa 0 \
  --hprom-update-maw-each-iter 1 \
  --hprom-include-weight-tangent 1 \
  --hprom-fail-on-nonconvergence 1 \
  --save-plots 1 \
  --out-dir stage_8_online_hprom_mawecm_gpr_res_eps_sig_auto_ecmhom_sum990_graph
```

Stage8b variant: residual MAW + sig MAW, eps fixed ECM
-------------------------------------------------------

This builds two adaptive MAW rules and one fixed hom rule:

- `Z_res,w_res(q_m)` for the residual.
- `Z_sig,w_sig(q_m)` for homogenized stress.
- `Z_eps,w_eps` from classical ECM, fixed online.

The stress MAW channel keeps nonnegative weights and preserves its classical ECM
sum of weights. The validated stress policy is the same as the full hom-MAW
policy: use graph phase 2 from the first elimination attempt.

```bash
--maw-hom-use-global-graph-2ndstage 1
--maw-hom-smooth-laplacian-all-iterations 1
```

The previous apparent optimum with `--maw-phase1-stop-size-sig 54` was just an
implicit way of entering phase 2 immediately, because 54 was the initial sigma
ECM support size.

By default, hom MAW can follow the residual graph settings:

```bash
--maw-hom-use-global-graph-2ndstage -1
--maw-hom-smooth-laplacian-all-iterations -1
--maw-hom-alpha-smooth -1
```

The remaining difference from residual MAW is the optional hom CV guard. It is useful
to avoid overly aggressive sigma/epsilon RBF fields, but it is not part of the
residual workflow. Disable it for residual-like pruning behavior:

```bash
--maw-hom-cv-max-rel 0
```

```bash
python3 stage8b_build_mawecm_res_model_rbf.py \
  --dataset-dir stage_8a_mawecm_res_dataset \
  --maw-mode res_sig \
  --hom-source ecm_separate \
  --maw-min-support-size-res 0 \
  --maw-min-support-size-sig 0 \
  --use-global-graph-2ndstage 1 \
  --smooth-laplacian-all-iterations 0 \
  --maw-hom-use-global-graph-2ndstage 1 \
  --maw-hom-smooth-laplacian-all-iterations 1 \
  --alpha-smooth 1e4 \
  --maw-hom-alpha-smooth 1e4 \
  --graph-mode structured_grid \
  --max-number-zeros-active-set-loop-maw-ecm 1 \
  --res-bootstrap-rsvd-randomized 0 \
  --res-bootstrap-constrain-sum-weights 1 \
  --enforce-sum-weights 1 \
  --sum-weights-target 990.0 \
  --tol-rank-rel 1e-14 \
  --maw-hom-enforce-nonnegativity 1 \
  --maw-hom-conservative 1 \
  --maw-hom-cv-max-rel 0 \
  --save-weight-field-plots 1 \
  --show-weight-field-plots 0 \
  --out-dir stage_8b_hprom_mawecm_res_sig_auto_ecmhom_sum990_graph
```

Online test with adaptive sig MAW and fixed eps ECM:

```bash
python3 stage8_test_hprom_mawecm_gpr_online.py \
  --run-fom \
  --run-prom-gpr \
  --run-hprom-mawecm-gpr \
  --mawecm-file stage_8b_hprom_mawecm_res_sig_auto_ecmhom_sum990_graph/ecm_weights_all.npz \
  --hprom-homogenization-mode sig_maw_eps_ecm \
  --hprom-use-hrom-mdpa 0 \
  --hprom-update-maw-each-iter 1 \
  --hprom-include-weight-tangent 1 \
  --hprom-fail-on-nonconvergence 1 \
  --save-plots 1 \
  --out-dir stage_8_online_hprom_mawecm_gpr_res_sig_auto_ecmhom_sum990_graph
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

Phase 1 until a controlled support size, then graph phase 2:

```bash
python3 stage8b_build_mawecm_res_model_rbf.py \
  --dataset-dir stage_8a_mawecm_res_dataset \
  --maw-mode res_only \
  --hom-source ecm_separate \
  --maw-min-support-size-res 0 \
  --maw-phase1-stop-size-res 40 \
  --use-global-graph-2ndstage 1 \
  --smooth-laplacian-all-iterations 0 \
  --alpha-smooth 1e6 \
  --graph-mode structured_grid \
  --max-number-zeros-active-set-loop-maw-ecm 1 \
  --res-bootstrap-constrain-sum-weights 1 \
  --enforce-sum-weights 1 \
  --sum-weights-target 990.0 \
  --tol-rank-rel 1e-14 \
  --save-weight-field-plots 1 \
  --show-weight-field-plots 0 \
  --out-dir stage_8b_hprom_mawecm_res_only_auto_ecmhom_sum990_phase1to40_graph

python3 stage8_test_hprom_mawecm_gpr_online.py \
  --run-fom \
  --run-prom-gpr \
  --run-hprom-mawecm-gpr \
  --mawecm-file stage_8b_hprom_mawecm_res_only_auto_ecmhom_sum990_phase1to40_graph/ecm_weights_all.npz \
  --hprom-homogenization-mode ecm_separate \
  --hprom-use-hrom-mdpa 0 \
  --hprom-update-maw-each-iter 1 \
  --hprom-include-weight-tangent 1 \
  --hprom-fail-on-nonconvergence 1 \
  --save-plots 1 \
  --out-dir stage_8_online_hprom_mawecm_gpr_res_only_auto_ecmhom_sum990_phase1to40_graph
```

Stage8b defaults to deterministic residual/hom bootstrap:

```bash
--res-bootstrap-rsvd-randomized 0
```

Use `--res-bootstrap-rsvd-randomized 1` only for randomized SVD experiments. It can
produce slightly different initial ECM supports and therefore slightly different
MAW pruning paths.

Stage9: direct GPR predictor baseline
-------------------------------------

Stage9 compares the corrected models against a no-solve predictor:

```text
strain/parameter -> q_m predictor -> sparse GPR q_s(q_m) -> u -> homogenization
```

This path does not run Newton, does not assemble residual elements, and does not
use residual MAW weights. It is useful to quantify how much the residual
corrector actually improves over the direct parameter-to-solution map.

Important interpretation:

- This is not "HPROM with zero Newton iterations".
- It predicts `q_m` independently from the strain/parameter at every step.
- It still uses the selected homogenization backend, for example `maw_separate`,
  so eps/sig MAW weights can be evaluated for postprocessing.
- Use the same output directory as a previous Stage8 run if you want FOM/PROM/HPROM
  read from cache and only the direct predictor computed.

Example using the validated HROM mdpa + adaptive eps/sig MAW setup:

```bash
python3 stage9_compare_direct_gpr_predictor.py \
  --run-direct-gpr \
  --mawecm-file stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/ecm_weights_all.npz \
  --hprom-homogenization-mode maw_separate \
  --hprom-use-hrom-mdpa 1 \
  --hprom-hrom-strict 1 \
  --hprom-update-maw-each-iter 1 \
  --hprom-include-weight-tangent 1 \
  --hprom-profile-timers 0 \
  --hprom-verbose-newton 0 \
  --hprom-log-every 100 \
  --save-plots 1 \
  --out-dir stage_8_online_hprom_mawecm_gpr_res_eps_sig_auto_ecmhom_sum990_graph_hrom_profiled
```

Stage9 writes:

```text
stage9_online_summary.json
stage9_online_compare_arrays.npz
trajectory_direct_gpr_predictor_*.npy
stage9_compare_*.png
stage9_timing_comparison.png
```

Representative validated result on the unseen trajectory:

```text
rel stress error HPROM-MAW  vs FOM    : 8.179e-04
rel stress error DIRECT-GPR vs FOM    : 1.020e-03
rel strain error HPROM-MAW  vs FOM    : 7.265e-04
rel strain error DIRECT-GPR vs FOM    : 5.100e-04
runtime HPROM-MAW [s]                 : 2.485
runtime DIRECT-GPR [s]                : 0.831
speedup FOM/DIRECT-GPR                : 186.60x
```

HROM mdpa mode
--------------

The simplest debugging mode is still full mdpa with reduced assembly loops:

```bash
--hprom-use-hrom-mdpa 0
```

The validated true reduced-mdpa mode is also supported. Build the HROM mdpa after
Stage8b. For separate residual/eps/sig supports, the reduced mesh must contain all
channel supports; use `Z_union`. The same command also saves the selection plots
for `Z_res`, `Z_eps`, `Z_sig`, and `Z_union`.

```bash
python3 stage6c_create_hrom_mdpa.py \
  --base-mesh rve_geometry \
  --ecm-file stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/ecm_weights_all.npz \
  --selection-key Z_union \
  --condition-mode all \
  --output-mesh rve_geometry_stage8b_maw_res_eps_sig_auto_ecmhom_sum990_graph_hrom \
  --inplace-ecm \
  --save-selection-image stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/Z_union_selected_elements.png \
  --save-extra-selection-images 1 \
  --extra-selection-keys Z_res,Z_eps,Z_sig,Z_union \
  --model-label HPROM-MAWECM-GPR

python3 stage8_test_hprom_mawecm_gpr_online.py \
  --run-hprom-mawecm-gpr \
  --mawecm-file stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/ecm_weights_all.npz \
  --hprom-homogenization-mode maw_separate \
  --hprom-use-hrom-mdpa 1 \
  --hprom-hrom-strict 1 \
  --hprom-update-maw-each-iter 1 \
  --hprom-include-weight-tangent 1 \
  --hprom-fail-on-nonconvergence 1 \
  --save-plots 1 \
  --out-dir stage_8_online_hprom_mawecm_gpr_res_eps_sig_auto_ecmhom_sum990_graph_hrom
```

Expected `stage6c` outputs:

```text
rve_geometry_stage8b_maw_res_eps_sig_auto_ecmhom_sum990_graph_hrom.mdpa
stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/Z_union_selected_elements.png
stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/Z_union_selected_elements_Z_res.png
stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/Z_union_selected_elements_Z_eps.png
stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/Z_union_selected_elements_Z_sig.png
stage_8b_hprom_mawecm_res_eps_sig_auto_ecmhom_sum990_graph/Z_union_selected_elements_Z_union.png
```

`stage6c` necessarily reads the full `rve_geometry.mdpa` once because it creates
the reduced mdpa and stores full-to-HROM maps in the ECM npz. The online HROM run
should not read `rve_geometry.mdpa`; use cached FOM/PROM data and run only
`--run-hprom-mawecm-gpr` when checking HROM-only behavior.

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

Stage8b already defaults to deterministic residual/hom bootstrap. To force it
explicitly, use:

```bash
--res-bootstrap-rsvd-randomized 0
```

If HROM mdpa fails with missing `Z_eps` or `Z_sig`, the reduced mesh was built with the wrong selection key. For separate hom supports, rebuild with:

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
