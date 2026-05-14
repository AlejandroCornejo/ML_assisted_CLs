# Neo-Hookean RVE Workflow: Stages 0 to 10

This folder contains a complete Reduced Order Modeling workflow for a 2D Neo-Hookean RVE with a hole:

- FOM data generation
- POD-based PROM
- ECM-based HPROM
- ANN-manifold PROM, called PROM-ANN
- RBF-manifold PROM, called PROM-RBF
- POD-DL / POD-AE manifold PROM, called PROM-POD-DL
- ECM-based HPROM for the RBF manifold, called HPROM-RBF

The goal of this README is handoff: someone new should be able to run the pipeline in order and understand what each stage does.

## Plot Style (LaTeX by Default)

All plotting scripts in this repository now apply a shared LaTeX-friendly Matplotlib setup via:

- `plot_style_utils.apply_latex_plot_style()`

Default behavior is `text.usetex=True` (paper-style plots).  
If you need a fallback without TeX on a machine that does not have a LaTeX installation, run with:

```bash
RVE_PLOT_USE_TEX=0 python3 <script>.py
```

## 1) Problem Setup and Data Formats

This section is important for ML integration, for example KANs, because it defines the input/output pairs used in the workflow.

### 1.1 Mechanics Formulation

- **Element type**: `TotalLagrangianElement2D6N`, 2D plane strain.
- **Constitutive law**: `HyperElasticPlaneStrain2DLaw`, Neo-Hookean.
- **Material parameters**:
  - `YOUNG_MODULUS = 1628 MPa`
  - `POISSON_RATIO = 0.4`
- **Strain measure**: Green-Lagrange strain tensor.

```math
\mathbf{E} = \frac{1}{2}(\mathbf{C} - \mathbf{I})
```

- **Stress measure**: Second Piola-Kirchhoff stress tensor.

```math
\mathbf{S}
```

### 1.2 Note for Alejandro: KAN Data Extraction

For training the KAN, please locate the database inside the Kratos training output directory:

```bash
stage_1_training_set_fom/trajectory_1/
```

Inside this folder, you will find several `.npy` arrays containing the data over `N` time steps.

- **`trajectory_1_U.npy`**: Shape `(16802, 4244)`. This is the full nodal displacement field, affine plus fluctuation, over time.

```math
\mathbf{u} = \mathbf{u}_{\mathrm{aff}} + \mathbf{w}
```

Here, `4244` is the total number of free degrees of freedom in the RVE mesh. You can ignore this file for the KAN if the KAN only needs homogenized strain/stress pairs.

- **Homogenized tensors**: Shape `(16802, 3)`. The three columns correspond to 2D Voigt notation:

```math
[XX,\ YY,\ XY]
```

Keep in mind that the shear component is stored as engineering shear:

```math
G_{xy} = \gamma_{xy} = 2E_{xy}
```

The main homogenized files are:

- **`trajectory_1_applied_strain.npy`**: The macroscopic Green-Lagrange strain imposed on the RVE boundaries.

```math
\bar{\mathbf{E}}
```

- **`trajectory_1_strain.npy`**: The macroscopic Green-Lagrange strain computed from the RVE. This is the computed volume average, denoted here as $\hat{\mathbf{E}}$, which may drift slightly from the applied strain $\bar{\mathbf{E}}$ as discussed in the meetings.

```math
\hat{\mathbf{E}}
```

- **`trajectory_1_stress.npy`**: The macroscopic, volume-averaged Second Piola-Kirchhoff stress obtained from the RVE homogenization.

```math
\bar{\mathbf{S}}
```

### 1.3 Boundary Lifting Setup

Macro strain is prescribed in Voigt form as:

```math
\mathbf{E}_{\mathrm{voigt}}
=
\begin{bmatrix}
E_{xx} \\
E_{yy} \\
G_{xy}
\end{bmatrix},
\qquad
G_{xy}=2E_{xy}
```

The corresponding tensor form is:

```math
\mathbf{E}
=
\begin{bmatrix}
E_{xx} & G_{xy}/2 \\
G_{xy}/2 & E_{yy}
\end{bmatrix}
```

To apply this as a boundary displacement condition on the RVE nodes, we compute the right Cauchy-Green tensor:

```math
\mathbf{C}=2\mathbf{E}+\mathbf{I}
```

Then we choose the rotation-free deformation gradient:

```math
\mathbf{F}=\mathbf{U}=\mathbf{C}^{1/2}
```

where `\mathbf{C}^{1/2}` denotes the symmetric positive-definite matrix square root of `\mathbf{C}`. This choice is valid for admissible strain states for which `\mathbf{C}` is positive definite.

The affine boundary displacement is:

```math
\mathbf{u}_{\mathrm{aff}}
=
(\mathbf{F}-\mathbf{I})(\mathbf{X}-\mathbf{X}_c)
```

where `\mathbf{X}` is the reference position and `\mathbf{X}_c` is the domain center.

**ML note:** Although the ML datasets track `\mathbf{E}`, the finite-element solver applies spatial displacements through `\mathbf{F}`. By setting:

```math
\mathbf{F}=\mathbf{U}=\mathbf{C}^{1/2}
```

we remove arbitrary rigid-body rotations from the boundary lifting. This gives a unique rotation-free representative of the deformation for each admissible strain state. This is useful for ML because rigid-body rotations do not contribute to strain energy or stress, but they can contaminate the input/output relationship if not controlled.

The total displacement is decomposed as:

```math
\mathbf{u} = \mathbf{u}_{\mathrm{aff}} + \mathbf{w}
```

with homogeneous fluctuation displacement on the Dirichlet boundary:

```math
\mathbf{w}_D = \mathbf{0}
```

The fluctuation field `\mathbf{w}` is the field used for reduction.

## 2) Stage Dependency Graph

Core chain:

```text
Stage0 -> Stage1(FOM) -> Stage2 -> Stage3/4 -> Stage5a -> Stage5b -> Stage6
```

ANN chain:

```text
Stage1(FOM) + Stage2 -> Stage7a -> Stage7b -> Stage7c -> Stage8
```

RBF chain:

```text
Stage1(FOM) + Stage2 -> Stage7a -> Stage7b-RBF -> Stage7c -> Stage8-RBF -> Stage9a -> Stage9b -> Stage10
```

POD-DL chain:

```text
Stage1(FOM) + Stage2 -> Stage7a-POD-DL -> Stage7b-POD-DL -> Stage7c -> Stage8-POD-DL -> Stage9a-POD-DL -> Stage9b-POD-DL -> Stage10-POD-DL
```

Final benchmark entrypoints:

- `Stage6`: FOM vs PROM vs HPROM
- `Stage8`: PROM-ANN benchmark
- `Stage8-RBF`: PROM-RBF benchmark
- `Stage8-POD-DL`: PROM-POD-DL benchmark
- `Stage10`: FOM vs PROM-RBF vs HPROM-RBF
- `Stage10-POD-DL`: FOM vs PROM-POD-DL vs HPROM-POD-DL

## 3) Main Reduced Models

### 3.1 Linear PROM: Galerkin

With POD basis:

```math
\Phi_f \in \mathbb{R}^{n_f \times r}
```

the free fluctuation field is approximated as:

```math
\mathbf{w}_f \approx \Phi_f \mathbf{q}
```

The reduced residual and tangent stiffness are:

```math
\mathbf{r}_r = \Phi_f^T \mathbf{r}_f
```

```math
\mathbf{K}_r = \Phi_f^T \mathbf{K}_{ff}\Phi_f
```

The reduced Newton system is written in this codebase as:

```math
\mathbf{K}_r \Delta\mathbf{q} = \mathbf{r}_r
```

Note: in this codebase, the assembled `rhs` sign convention is such that the Newton update is written with `+rhs`. This is equivalent to the usual `-R` form after accounting for the internal sign convention.

### 3.2 HPROM: ECM Hyperreduction

Element contributions are weighted by ECM cubature weights:

```math
\sum_{e\in\mathcal{Z}} \omega_e \mathbf{r}_e
```

```math
\sum_{e\in\mathcal{Z}} \omega_e \mathbf{K}_e
```

where:

- `\mathcal{Z}` is the selected ECM element set.
- `\omega_e` is the positive cubature weight of element `e`.
- `\mathbf{r}_e` is the element residual contribution.
- `\mathbf{K}_e` is the element tangent stiffness contribution.

The weighted residual and tangent are then assembled and projected to reduced coordinates as in the PROM.

### 3.3 PROM-ANN: Galerkin on a Nonlinear Manifold

The POD basis is split into primary and secondary modes:

```math
\Phi_f =
\begin{bmatrix}
\Phi_p & \Phi_s
\end{bmatrix}
```

with:

```math
\dim(\mathbf{q}_p)=4,
\qquad
\dim(\mathbf{q}_s)=17
```

(for a total of 21 POD modes). An ANN is trained to approximate the secondary coordinates from the primary coordinates:

```math
\mathbf{q}_s = \mathcal{N}(\mathbf{q}_p)
```

In the online/offline manifold-consistent implementation, we use the corrected map:

```math
\bar{\mathcal{N}}(\mathbf{q}_p)
=
\mathcal{N}(\mathbf{q}_p)
-
\mathcal{N}(\mathbf{0})
-
\mathbf{J}_{\mathcal{N}}(\mathbf{0})\,\mathbf{q}_p
```

with:

```math
\mathbf{J}_{\mathcal{N}}(\mathbf{q}_p)
=
\frac{\partial \mathcal{N}}{\partial \mathbf{q}_p},
\qquad
\bar{\mathbf{J}}_{\mathcal{N}}(\mathbf{q}_p)
=
\mathbf{J}_{\mathcal{N}}(\mathbf{q}_p)-\mathbf{J}_{\mathcal{N}}(\mathbf{0})
```

The decoder is:

```math
\mathbf{w}_f(\mathbf{q}_p)
=
\Phi_p\mathbf{q}_p
+
\Phi_s\bar{\mathcal{N}}(\mathbf{q}_p)
```

The manifold Jacobian is:

```math
\mathbf{J}_m
=
\frac{\partial \mathbf{w}_f}{\partial \mathbf{q}_p}
=
\Phi_p
+
\Phi_s
\bar{\mathbf{J}}_{\mathcal{N}}(\mathbf{q}_p)
```

The reduced residual is:

```math
\mathbf{r}_r
=
\mathbf{J}_m^T\mathbf{r}_f
```

The reduced tangent stiffness is approximated as:

```math
\mathbf{K}_r
\approx
\mathbf{J}_m^T
\mathbf{K}_{ff}
\mathbf{J}_m
```

The reduced Newton system is:

```math
\mathbf{K}_r \Delta\mathbf{q}_p
=
\mathbf{r}_r
```

Here:

- `\mathbf{r}_r` is the reduced residual vector.
- `\mathbf{K}_r` is the reduced tangent stiffness matrix.
- `\mathbf{J}_m` is the manifold Jacobian.
- `\Delta\mathbf{q}_p` is the Newton-Raphson increment in the reduced primary coordinate space.

Important: the tangent formula above is a practical Gauss-Newton-type approximation on the nonlinear manifold. It ignores second-order manifold curvature terms involving the Hessian of the decoder. This is usually acceptable when the residual is small near convergence, but it is not the fully exact Newton tangent on a nonlinear manifold.

### 3.4 PROM-RBF: Galerkin on a Nonlinear RBF Manifold

The same POD split is used:

```math
\Phi_f =
\begin{bmatrix}
\Phi_p & \Phi_s
\end{bmatrix}
```

The RBF map is trained with primary coordinates only:

```math
\mathbf{q}_s
=
\mathcal{R}(\mathbf{q}_p)
```

where `\mathcal{R}` is the compact-center RBF map trained in Stage 7b-RBF.

In the online/offline manifold-consistent implementation, we use the corrected map:

```math
\bar{\mathcal{R}}(\mathbf{q}_p)
=
\mathcal{R}(\mathbf{q}_p)
-
\mathcal{R}(\mathbf{0})
-
\mathbf{J}_{\mathcal{R}}(\mathbf{0})\,\mathbf{q}_p
```

with:

```math
\mathbf{J}_{\mathcal{R}}(\mathbf{q}_p)
=
\frac{\partial \mathcal{R}}{\partial \mathbf{q}_p},
\qquad
\bar{\mathbf{J}}_{\mathcal{R}}(\mathbf{q}_p)
=
\mathbf{J}_{\mathcal{R}}(\mathbf{q}_p)-\mathbf{J}_{\mathcal{R}}(\mathbf{0})
```

The decoder is:

```math
\mathbf{w}_f(\mathbf{q}_p)
=
\Phi_p\mathbf{q}_p
+
\Phi_s\bar{\mathcal{R}}(\mathbf{q}_p)
```

The manifold Jacobian with respect to the reduced unknowns is:

```math
\mathbf{J}_m
=
\Phi_p
+
\Phi_s
\bar{\mathbf{J}}_{\mathcal{R}}(\mathbf{q}_p)
```

The reduced residual is:

```math
\mathbf{r}_r
=
\mathbf{J}_m^T\mathbf{r}_f
```

The reduced tangent stiffness is approximated as:

```math
\mathbf{K}_r
\approx
\mathbf{J}_m^T
\mathbf{K}_{ff}
\mathbf{J}_m
```

The reduced Newton system is:

```math
\mathbf{K}_r \Delta\mathbf{q}_p
=
\mathbf{r}_r
```

As in PROM-ANN, this tangent is an approximate tangent on the nonlinear manifold and ignores second-order decoder curvature terms.

### 3.5 PROM-POD-DL / POD-AE: Galerkin on a Latent Manifold

An autoencoder is trained in POD coordinate space:

```math
\hat{\mathbf{q}}
=
\mathcal{D}(\mathcal{E}(\mathbf{q}))
```

where:

- `\mathcal{E}` is the encoder.
- `\mathcal{D}` is the decoder.
- Scaling is embedded inside the network layers.

The online manifold is parameterized by a latent variable:

```math
\mathbf{z}
```

The decoded POD coordinate is shifted to enforce origin anchoring:

```math
\mathbf{q}(\mathbf{z})
=
\mathcal{D}(\mathbf{z})
-
\mathbf{q}_{\mathrm{ref}}
```

with:

```math
\mathbf{q}_{\mathrm{ref}}
=
\mathcal{D}(\mathcal{E}(\mathbf{0}))
```

The physical fluctuation field is reconstructed as:

```math
\mathbf{w}_f(\mathbf{z})
=
\Phi_q \mathbf{q}(\mathbf{z})
```

The manifold Jacobian is:

```math
\mathbf{J}_m
=
\frac{\partial \mathbf{w}_f}{\partial \mathbf{z}}
=
\Phi_q
\frac{\partial \mathbf{q}}{\partial \mathbf{z}}
```

The reduced residual is:

```math
\mathbf{r}_r
=
\mathbf{J}_m^T\mathbf{r}_f
```

The reduced tangent stiffness is approximated as:

```math
\mathbf{K}_r
\approx
\mathbf{J}_m^T
\mathbf{K}_{ff}
\mathbf{J}_m
```

The reduced Newton system is:

```math
\mathbf{K}_r \Delta\mathbf{z}
=
\mathbf{r}_r
```

Again, this is an approximate tangent on the nonlinear manifold unless second-order decoder curvature terms are explicitly included.

## 4) How to Run

Always run from:

```bash
cd /home/kratos/ML_assisted_CLs/homogenization_NeoHookean_using_Kratos_Hole_Tension
```

### Stage 0: Generate Training Trajectories

Default setup is the box domain, matching the 200 percent tension, 10 percent compression, and 10 percent shear setup:

```bash
python3 stage0_training_trajectory.py \
  --design box \
  --emax 2.0 \
  --relative-boundary 1.0,0.05,1.0,0.05,0.05,0.05 \
  --rows-per-layer 9 \
  --n-layers 3 \
  --ref-steps 400 \
  --reference-amplitude 0.5 \
  --movie-frames 300 \
  --movie-fps 24 \
  --out-dir stage_0_trajectory
```

You can also run:

```bash
python3 stage0_training_trajectory.py
```

because these are now the defaults.

Primary output:

```bash
stage_0_trajectory/stage_0_trajectories.npz
```

### Stage 1: Generate FOM Snapshots

```bash
python3 stage1_training_fom_solver_rve.py \
  --stage0-file stage_0_trajectory/stage_0_trajectories.npz \
  --which all \
  --out-dir stage_1_training_set_fom
```

Convergence safety: Stage 1 immediately aborts if the Newton-Raphson solver fails to converge on any trajectory. If this happens, re-run Stage 0 with a higher `--ref-steps`, for example `400`, to reduce strain increments.

Per-trajectory outputs include:

```bash
trajectory_i_U.npy
trajectory_i_applied_strain.npy
trajectory_i_strain.npy
trajectory_i_stress.npy
```

### Stage 2: Build POD Basis for Free Fluctuations

```bash
python3 stage2_pod_rve.py
```

Output folder:

```bash
stage_2_pod_rve/
```

Key files:

```bash
pod_basis_free.npy
free_dofs.npy
dirichlet_dofs.npy
eq_map.npy
domain_center.npy
```

### Stage 3: PROM Verification on a Seen Trajectory

```bash
python3 stage3_verification_rve.py --index 9
```

### Stage 4: PROM Generalization Test on an Unseen Path

```bash
python3 stage4_test_rve.py --run-fom
```

If the FOM solution is already cached, you can run:

```bash
python3 stage4_test_rve.py
```

### Stage 5a: Build ECM Dataset

```bash
python3 stage5_build_ecm_dataset.py
```

New options (independent sampling for residual vs homogenization):

```bash
python3 stage5_build_ecm_dataset.py \
  --snapshot-percent-res 5.0 \
  --snapshot-percent-hom 5.0
```

Useful flags:

- `--snapshot-percent-res`: percentage used for `Q_ecm`/`b_full` (residual ECM)
- `--snapshot-percent-hom`: percentage used for `C_hom`/`b_hom` (strain/stress homogenization ECM)
- `--sampling-mode {param_aware,stratified}`
- `--seed`
- `--param-aware-time-weight`

Output folder:

```bash
stage_5_ecm_dataset/
```

Key files:

```bash
Q_ecm.dat
b_full.dat
C_hom.dat
b_hom.dat
meta.npz
```

`meta.npz` now stores separate counters:

- `N_s_res`: residual snapshot count
- `N_s_hom`: homogenization snapshot count

### Stage 5b: Compute ECM Weights

```bash
python3 stage5_compute_ecm_weights.py
```

Coupling modes (`--ecm-coupling-mode`):

- `independent`: three separate ECM solves (RES, EPS, SIG)
- `cascade`: EPS initialized from RES and SIG initialized from EPS (legacy `coupled`)
- `single`: one ECM solve shared by RES, EPS, and SIG, built from one aggregated matrix and one RSVD

For `single`, block scaling before the aggregated RSVD is configurable with:

- `--single-block-normalization fro` (default): equal Frobenius-energy weighting of RES/EPS/SIG
- `--single-block-normalization row`: scales by `1/sqrt(n_rows)` per block
- `--single-block-normalization none`: no scaling

Optional: `--rsvd-tol-single` (if omitted or `<=0`, it uses `min(rsvd_tol_res, rsvd_tol_eps, rsvd_tol_sig)`).

Example:

```bash
python3 stage5_compute_ecm_weights.py --ecm-coupling-mode single
```

Output:

```bash
stage_5_hprom_data/ecm_weights_all.npz
```

### Stage 6: FOM vs PROM vs HPROM Benchmark

```bash
python3 stage6_test_hprom.py --run-fom --run-prom
```

Notes:

- Stage 6 is self-contained and does not depend on Stage 4 outputs.
- To reuse cached outputs, run:

```bash
python3 stage6_test_hprom.py
```

### Stage 7a: Build ANN/RBF Manifold Dataset

```bash
python3 stage7a_prepare_ann_rbf_dataset.py
```

This stage projects fluctuation snapshots, after subtracting affine lifting, to:

- inputs: `q_p`, using 4 modes
- targets: `q_s`, using 17 modes (for a total of 21 modes)

### Stage 7a-LS-RBF: Build Least-Squares RBF Dataset + Jacobian Diagnostics

```bash
python3 stage7a_prepare_rbf_dataset_ls.py \
  --n-primary 3 \
  --out-dir stage_7_ann_data_ls \
  --plot-max-samples 110000 \
  --jacobian-mesh-type delaunay \
  --jacobian-mesh-max-points 6000 \
  --jacobian-mesh-seed 42
```

Notes:

- This command uses **no strain input** for RBF by default.
- It computes and saves:
  - domain comparison plots (`mu`, first-3 POD, first-3 LS-primary),
  - Delaunay mesh figures in parameter and mapped LS-primary domains,
  - parameter-mesh Jacobian diagnostics,
  - macro deformation Jacobian check (`det(F)` from applied strain),
  - a LaTeX summary note:
    - `stage_7_ann_data_ls/ls_rbf_parameter_mesh_jacobian_note.tex`

### Stage 7a-POD-DL: Build POD-DL Dataset

```bash
python3 stage7a_prepare_pod_dl_dataset.py \
  --fom-dir stage_1_training_set_fom \
  --basis-dir stage_2_pod_rve \
  --out-dir stage_7_pod_dl_data \
  --q-dim 21
```

Outputs in:

```bash
stage_7_pod_dl_data/
```

Key files:

```bash
q_dataset.npy
phi_q.npy
pod_dl_dataset_metadata.npz
```

### Stage 7b: Train Manifold ANN

```bash
python3 stage7b_train_ann_manifold.py
```

Current ANN activation is `ELU` (hidden stack: `Linear + BatchNorm1d + ELU`).
The trainer supports architecture/hyperparameter tuning through CLI (e.g. `--hidden-layers`,
`--dropout`, `--loss smoothl1`, `--lr`, `--batch-size`, `--epochs`, `--grad-clip-norm`).

Outputs in:

```bash
stage_7_ann_data/
```

Key files:

```bash
manifold_ann.pt
manifold_ann_metadata.npz
```

### Stage 7b-RBF: Train Compact-Center RBF Manifold

Policy:
- grid search is always enabled (training without grid search is blocked),
- the number of centers is user-configurable via `--max-centers` (default `4000`),
- grid search uses all available training samples.

```bash
python3 stage7b_train_rbf_manifold.py \
  --data-dir stage_7_ann_data \
  --out-dir stage_7_rbf_data \
  --max-centers 8000 \
  --center-selection kmeans \
  --kmeans-max-iters 30 \
  --kmeans-batch-size 4096 \
  --kmeans-fit-samples 60000 \
  --sparse-prune-centers 4000 \
  --kernel gaussian \
  --ridge 1e-10 \
  --grid-kernels gaussian \
  --grid-eps-values 0.2,0.5,1.0,2.0,3.0,4.0,5.0 \
  --grid-ridges 1e-10,1e-8 \
  --grid-folds 5
```

Notes:
- `--center-selection random|kmeans` controls how center positions are chosen.
- `--sparse-prune-centers K` keeps the top-`K` centers by `||weights||_2` after an initial fit and then refits the model (sparse RBF).
- `--rbf-metric anisotropic` enables per-dimension epsilon scaling (analytic derivatives are preserved).
- If `--anisotropic-scales` is omitted, Stage 7b estimates anisotropy from linear sensitivity and normalizes it.
- For stable bash parsing, keep comma-separated lists without spaces.

Outputs in:

```bash
stage_7_rbf_data/
```

Key files:

```bash
rbf_model.npz
phi_p.npy
phi_s.npy
training_summary.txt
```

Important: keep comma-separated CLI values without spaces in bash.

### Stage 7b-Sparse-GPR (optional): True Sparse-GP Manifold

This branch trains a pure sparse variational GP manifold `q_p -> q_s` and exports
`sparse_gp_model.npz` with analytic kernel coefficients for online PROM/HPROM-GPR.
No RBF-hyperfit export is used in this branch.

```bash
python3 stage7b_train_sparse_gpr_manifold.py \
  --data-dir stage_7_ann_data \
  --out-dir stage_7_gpr_data \
  --num-inducing 800 \
  --inducing-selection kmeans \
  --kmeans-max-iters 40 \
  --kmeans-batch-size 4096 \
  --kmeans-fit-samples 40000 \
  --epochs 120 \
  --batch-size 2048 \
  --lr 0.05 \
  --val-fraction 0.1 \
  --device auto
```

Outputs:
- `sparse_gp_model.npz` (analytic sparse-GP online model),
- `phi_p.npy`, `phi_s.npy`,
- `training_summary.txt`.

If the input dataset contains `ls_targets_train.npy` (LS branch), Stage 7b-Sparse-GPR
also exports:

- `qp_init_mu_affine.npz`

This file enables the online initializer mode:

- `--qp-init-mode mu_affine`

in `stage8_test_prom_gpr.py` and `stage10_test_hprom_gpr.py`.

### Recommended Model (Current Best): Sparse-GPR

Current recommended branch for production runs:
- Stage 7 sparse-GPR manifold (`sparse_gp_model.npz`)
- Stage 9 GPR ECM
- Stage 10 HPROM-GPR benchmark

Complete cascade:

```bash
cd /home/kratos/ML_assisted_CLs_clean/RVE_homogenization_NeoHookean_using_Kratos

python3 stage7a_prepare_ann_rbf_dataset.py --n-primary 4

python3 stage7b_train_sparse_gpr_manifold.py \
  --data-dir stage_7_ann_data \
  --out-dir stage_7_gpr_data \
  --num-inducing 800 \
  --inducing-selection kmeans \
  --kmeans-max-iters 40 \
  --kmeans-batch-size 4096 \
  --kmeans-fit-samples 110000 \
  --train-samples 0 \
  --val-fraction 0.1 \
  --epochs 120 \
  --batch-size 2048 \
  --lr 0.05 \
  --device auto \
  --seed 42

python3 stage9_build_ecm_dataset_gpr.py \
  --gpr-dir stage_7_gpr_data \
  --out-dir stage_9_ecm_dataset_gpr \
  --snapshot-percent-res 5 \
  --snapshot-percent-hom 5 \
  --residual-fit-mode gauss_newton

python3 stage9_compute_ecm_weights_gpr.py \
  --data-dir stage_9_ecm_dataset_gpr \
  --out-dir stage_9_hprom_gpr_data \
  --ecm-coupling-mode cascade \
  --ecm-tol-res 0 \
  --ecm-tol-eps 0 \
  --ecm-tol-sig 0

python3 build_hrom_mesh_from_ecm.py \
  --base-mesh rve_geometry \
  --ecm-file stage_9_hprom_gpr_data/ecm_weights_all.npz \
  --selection-key Z_union \
  --condition-mode all \
  --output-mesh rve_geometry_stage_9_hprom_gpr_data_z_union_hrom \
  --inplace-ecm

python3 stage10_test_hprom_gpr.py \
  --run-prom-gpr \
  --run-hprom-gpr \
  --gpr-data-dir stage_7_gpr_data \
  --hprom-gpr-dir stage_9_hprom_gpr_data \
  --out-dir stage_10_hprom_gpr_results
```

### Stage 7b-POD-DL: Train POD-DL / POD-AE Manifold

```bash
python3 stage7b_train_pod_dl_manifold.py \
  --data-dir stage_7_pod_dl_data \
  --fom-dir stage_1_training_set_fom \
  --basis-dir stage_2_pod_rve \
  --out-dir stage_7_pod_dl_data \
  --q-dim 21 \
  --latent-dim 4 \
  --latent-sweep 3,6,9,12
```

Outputs in:

```bash
stage_7_pod_dl_data/
```

Key files:

```bash
pod_dl_autoencoder.pt
phi_q.npy
training_summary.txt
```

### Stage 7c: Pure Reconstruction Check on One Training Trajectory

RBF check:

```bash
python3 stage7c_test_rbf_rom.py --trajectory-index 1
```

ANN check:

```bash
python3 stage7c_test_ann_rom.py --trajectory-index 1
```

POD-DL check:

```bash
python3 stage7c_test_pod_dl_rom.py --trajectory-index 1
```

Outputs are stored in:

```bash
stage_7c_reconstruction_results/<model>_traj_<idx>/
```

Bounds comparison in one run, including POD-4, POD-21, ANN, RBF, and POD-DL:

```bash
python3 stage7c_compare_bounds.py \
  --trajectory-index 1 \
  --pod-dl-data-dir stage_7_pod_dl_data
```

Outputs are stored in:

```bash
stage_7c_reconstruction_results/bounds_traj_<idx>/
```

### Stage 8: PROM-ANN Benchmark

Stage 8 uses the same benchmark trajectory logic as Stages 4 and 6:

- multi-view trajectory plot
- dynamic segment stepping
- reference step level `400`

Generate only the trajectory plot and step report:

```bash
python3 stage8_test_prom_ann.py --plot-only
```

PROM-ANN only:

```bash
python3 stage8_test_prom_ann.py --no-compare
```

PROM-ANN plus local FOM/HPROM comparison:

```bash
python3 stage8_test_prom_ann.py
```

Force recompute Stage 8 local baselines:

```bash
python3 stage8_test_prom_ann.py --run-fom --run-hprom
```

Default Stage 8 trajectory mode is already the Stage-6 dense waypoint path.

Results are stored in:

```bash
stage_8_prom_ann_results/
```

### Stage 8-RBF: PROM-RBF Benchmark

PROM-RBF only:

```bash
python3 stage8_test_prom_rbf.py --no-compare
```

PROM-RBF plus local FOM/HPROM comparison:

```bash
python3 stage8_test_prom_rbf.py
```

Force recompute Stage 8-RBF local baselines:

```bash
python3 stage8_test_prom_rbf.py --run-fom --run-hprom
```

Default Stage 8-RBF trajectory mode is already the Stage-6 dense waypoint path.

### Stage 8-POD-DL: PROM-POD-DL Benchmark

PROM-POD-DL only:

```bash
python3 stage8_test_prom_dl.py --no-compare
```

PROM-POD-DL plus local FOM/HPROM comparison:

```bash
python3 stage8_test_prom_dl.py
```

Force recompute Stage 8-POD-DL local baselines:

```bash
python3 stage8_test_prom_dl.py --run-fom --run-hprom
```

Default Stage 8-POD-DL trajectory mode is already the Stage-6 dense waypoint path.

### Stage 8-GPR: PROM-GPR Benchmark

PROM-GPR only:

```bash
python3 stage8_test_prom_gpr.py --no-compare
```

PROM-GPR only with affine initialization from macro strain (`mu -> q_p`):

```bash
python3 stage8_test_prom_gpr.py --no-compare --qp-init-mode mu_affine
```

PROM-GPR plus local FOM/HPROM comparison:

```bash
python3 stage8_test_prom_gpr.py
```

Force recompute Stage 8-GPR local baselines:

```bash
python3 stage8_test_prom_gpr.py --run-fom --run-hprom
```

### Stage 9a: Build ECM Dataset for HPROM-RBF

This stage uses the same stratified snapshot sampling logic as Stage 5a, but projects element residuals with the RBF-manifold tangent:

```math
\mathbf{J}_m
```

instead of the linear POD basis.

```bash
python3 stage9_build_ecm_dataset_rbf.py
```

Optional nonlinear manifold-consistency fit for residual snapshots (before building `Q_ecm`):

- `--residual-fit-mode none` (default): use projected `q_p` directly
- `--residual-fit-mode gauss_newton`: local Gauss-Newton fit of `q_p`
- tuning flags: `--fit-max-iters`, `--fit-rel-tol`, `--fit-l2-reg`, `--fit-step-tol`

Example:

```bash
python3 stage9_build_ecm_dataset_rbf.py \
  --residual-fit-mode gauss_newton \
  --fit-max-iters 8 \
  --fit-rel-tol 1e-6
```

The ANN dataset builder (`stage9_build_ecm_dataset_ann.py`) supports the same residual-fit flags.
The POD-DL dataset builder (`stage9_build_ecm_dataset_pod_dl.py`) supports the same residual-fit flags.

Output folder:

```bash
stage_9_ecm_dataset_rbf/
```

Key files:

```bash
Q_ecm.dat
b_full.dat
C_hom.dat
b_hom.dat
meta.npz
```

### Stage 9b: Compute ECM Weights for HPROM-RBF

This stage uses the same RSVD plus ECM workflow as Stage 5b, applied to Stage 9a data.

```bash
python3 stage9_compute_ecm_weights_rbf.py
```

Stage 9 ECM scripts (`stage9_compute_ecm_weights_rbf.py` and
`stage9_compute_ecm_weights_ann.py` and `stage9_compute_ecm_weights_pod_dl.py`) support the same
`--ecm-coupling-mode` values as Stage 5b:

- `independent`
- `cascade` (legacy alias: `coupled`)
- `single`

For `single`, they also support:

- `--single-block-normalization {fro,row,none}`
- `--rsvd-tol-single`

Examples:

```bash
python3 stage9_compute_ecm_weights_rbf.py --ecm-coupling-mode single
python3 stage9_compute_ecm_weights_ann.py --ecm-coupling-mode single
python3 stage9_compute_ecm_weights_pod_dl.py --ecm-coupling-mode single
```

Output:

```bash
stage_9_hprom_rbf_data/ecm_weights_all.npz
```

### Stage 9c: Build HROM Mesh from ECM + Paper-Style Selection Image

```bash
python3 build_hrom_mesh_from_ecm.py \
  --base-mesh rve_geometry \
  --ecm-file stage_9_hprom_rbf_data/ecm_weights_all.npz \
  --selection-key Z_union \
  --condition-mode all \
  --output-mesh rve_geometry_stage_9_hprom_rbf_data_z_union_hrom \
  --inplace-ecm \
  --save-selection-image stage_9_hprom_rbf_data/Z_union_selected_elements_paper.png \
  --model-label "HPROM-RBF"
```

Notes:

- LaTeX text rendering for the selection image is **enabled by default**.
- If your machine has no TeX installation, disable it with:

```bash
--no-use-tex
```

### Stage 10: FOM vs PROM-RBF vs HPROM-RBF Benchmark

```bash
python3 stage10_test_hprom_rbf.py
```

Optional forced recompute:

```bash
python3 stage10_test_hprom_rbf.py --run-fom --run-prom-rbf --run-hprom-rbf
```

### Stage 9a/9b + Stage 10 for HPROM-GPR

Build GPR ECM dataset:

```bash
python3 stage9_build_ecm_dataset_gpr.py \
  --gpr-dir stage_7_gpr_data \
  --snapshot-percent-res 5 \
  --snapshot-percent-hom 5 \
  --residual-fit-mode gauss_newton
```

Compute GPR ECM weights:

```bash
python3 stage9_compute_ecm_weights_gpr.py \
  --ecm-coupling-mode cascade
```

Build HROM mesh from GPR ECM selection:

```bash
python3 build_hrom_mesh_from_ecm.py \
  --base-mesh rve_geometry \
  --ecm-file stage_9_hprom_gpr_data/ecm_weights_all.npz \
  --selection-key Z_union \
  --condition-mode all \
  --output-mesh rve_geometry_stage_9_hprom_gpr_data_z_union_hrom \
  --inplace-ecm \
  --save-selection-image stage_9_hprom_gpr_data/Z_union_selected_elements_paper.png \
  --model-label "HPROM-GPR (ECM)"
```

Run FOM vs PROM-GPR vs HPROM-GPR:

```bash
python3 stage10_test_hprom_gpr.py
```

Run with affine `mu -> q_p` initialization:

```bash
python3 stage10_test_hprom_gpr.py \
  --run-prom-gpr --run-hprom-gpr \
  --gpr-data-dir stage_7_gpr_data_ls \
  --hprom-gpr-dir stage_9_hprom_gpr_data_ls \
  --out-dir stage_10_hprom_gpr_ls_results \
  --qp-init-mode mu_affine
```

Important:
- `--qp-init-mode mu_affine` requires `qp_init_mu_affine.npz` inside `--gpr-data-dir`.
- If that file is missing, the script aborts by design.

Optional forced recompute:

```bash
python3 stage10_test_hprom_gpr.py --run-fom --run-prom-gpr --run-hprom-gpr
```

### Stage 10 (Sparse-Point Variant): HPROM-GPR at N Points Only

This variant keeps the full Stage-10 trajectory definition but solves HPROM-GPR only
at a sparse subset of dynamic points (for example 20 points), then compares against
FOM sampled at the same points.

```bash
python3 stage10_test_hprom_gpr_sparse_points.py \
  --n-points 20 \
  --run-hprom-gpr \
  --gpr-data-dir stage_7_gpr_data_ls \
  --hprom-gpr-dir stage_9_hprom_gpr_data_ls \
  --out-dir stage_10_hprom_gpr_sparse_points \
  --qp-init-mode mu_affine
```

Plot behavior:
- FOM is shown as the full trajectory (all points).
- HPROM-GPR is shown as scatter only at the sparse evaluated points.

### Stage 9a/9b + Stage 10 for HPROM-POD-DL

Build POD-DL ECM dataset (same 5/5 default sampling style):

```bash
python3 stage9_build_ecm_dataset_pod_dl.py \
  --snapshot-percent-res 5 \
  --snapshot-percent-hom 5
```

Compute POD-DL ECM weights:

```bash
python3 stage9_compute_ecm_weights_pod_dl.py \
  --ecm-coupling-mode cascade
```

Build HROM mesh from POD-DL ECM selection:

```bash
python3 build_hrom_mesh_from_ecm.py \
  --base-mesh rve_geometry \
  --ecm-file stage_9_hprom_pod_dl_data/ecm_weights_all.npz \
  --selection-key Z_union \
  --condition-mode all \
  --output-mesh rve_geometry_stage_9_hprom_pod_dl_data_z_union_hrom \
  --inplace-ecm \
  --save-selection-image stage_9_hprom_pod_dl_data/Z_union_selected_elements_paper.png \
  --model-label "HPROM-POD-DL (ECM)"
```

Run FOM vs PROM-POD-DL vs HPROM-POD-DL:

```bash
python3 stage10_test_hprom_dl.py
```

Optional forced recompute:

```bash
python3 stage10_test_hprom_dl.py --run-fom --run-prom-dl --run-hprom-dl
```

### ANN-LS Variant (Quick Cascade)

ANN-LS uses the LS-coordinate dataset from Stage 7a-LS and trains the ANN on that LS dataset.

1) Build ANN-LS dataset:

```bash
python3 stage7a_prepare_ann_dataset_ls.py \
  --n-primary 3 \
  --out-dir stage_7_ann_data_ls
```

2) Train ANN on LS dataset:

```bash
python3 stage7b_train_ann_manifold.py \
  --data-dir stage_7_ann_data_ls
```

Recommended ANN-LS sweep (20 configurations, auto-select best):

```bash
python3 stage7b_sweep_ann_manifold.py \
  --data-dir stage_7_ann_data_ls \
  --out-dir stage_7_ann_ls_sweep \
  --best-out-dir stage_7_ann_data_ls \
  --n-trials 20 \
  --epochs 2500 \
  --seed 42
```

Progressive-width, fast-Jacobian oriented ANN-LS sweep (example family includes
architectures like `5,10,15,20,23` and related progressive variants):

```bash
python3 stage7b_sweep_ann_manifold.py \
  --data-dir stage_7_ann_data_ls \
  --out-dir stage_7_ann_ls_sweep_progressive_fast \
  --best-out-dir stage_7_ann_data_ls \
  --n-trials 20 \
  --epochs 2500 \
  --seed 42 \
  --config-mode progressive_fast
```

Latest best ANN-LS sweep result (May 10, 2026; `stage_7_ann_ls_sweep/sweep_summary.txt`):

- `trial_20`
- `hidden_layers=128,128,128`
- `activation=gelu`
- `dropout=0.0`
- `use_batchnorm=0`
- `loss=mse`
- `lr=1.2e-3`
- `weight_decay=5e-5`
- `batch_size=3072`
- `patience=80`
- `lr_patience=35`
- `lr_factor=0.5`
- `grad_clip_norm=2.0`
- `best_val_scaled_mse=3.5087393728538923e-07` at epoch `2487`

To train directly with this fixed best configuration (without sweep):

```bash
python3 stage7b_train_ann_manifold.py \
  --data-dir stage_7_ann_data_ls \
  --hidden-layers 128,128,128 \
  --activation gelu \
  --dropout 0.0 \
  --no-batchnorm \
  --loss mse \
  --lr 1.2e-3 \
  --weight-decay 5e-5 \
  --batch-size 3072 \
  --epochs 2500 \
  --patience 80 \
  --lr-patience 35 \
  --lr-factor 0.5 \
  --grad-clip-norm 2.0 \
  --seed 1020
```

3) Build ANN-LS ECM dataset (example with 5/5 sampling):

```bash
python3 stage9_build_ecm_dataset_ann_ls.py \
  --snapshot-percent-res 5 \
  --snapshot-percent-hom 5
```

4) Compute ANN-LS ECM weights:

```bash
python3 stage9_compute_ecm_weights_ann_ls.py \
  --ecm-coupling-mode cascade
```

5) Build ANN-LS HROM mesh and update ECM file in place:

```bash
python3 build_hrom_mesh_from_ecm.py \
  --base-mesh rve_geometry \
  --ecm-file stage_9_hprom_ann_data_ls/ecm_weights_all.npz \
  --selection-key Z_union \
  --condition-mode all \
  --output-mesh rve_geometry_stage_9_hprom_ann_data_ls_z_union_hrom \
  --inplace-ecm \
  --save-selection-image stage_9_hprom_ann_data_ls/Z_union_selected_elements_paper.png \
  --model-label "HPROM-ANN-LS"
```

6) Run PROM-ANN-LS benchmark:

```bash
python3 stage8_test_prom_ann_ls.py --run-fom --run-hprom
```

7) Run HPROM-ANN-LS benchmark:

```bash
python3 stage10_test_hprom_ann_ls.py --run-fom --run-prom-ann --run-hprom-ann
```

The file:

```bash
stage7c_reconstruction_check.py
```

remains the shared implementation for reconstruction checks. Use the dedicated entrypoints:

```bash
stage7c_test_ann_rom.py
stage7c_test_rbf_rom.py
stage7c_test_pod_dl_rom.py
```

for model-specific runs.

## 5) Practical Notes

- Scripts expect to run from this folder because relative paths are used.
- Most scripts add the Kratos binary path internally:

```bash
/home/kratos/Kratos_Eigen_Check/bin/Release
```

- FOM, PROM, HPROM, PROM-ANN, PROM-RBF, and PROM-POD-DL reuse previous-step stiffness in Newton iteration 0 by default.
- You can disable this behavior from the CLI with:

```bash
--no-old-stiffness-first-it
```

This flag is available in:

```bash
fom_solver_rve.py
prom_solver_rve.py
stage8_test_prom_ann.py
stage8_test_prom_rbf.py
stage8_test_prom_dl.py
```

- Stage outputs are cache-friendly. Re-running often reuses existing `.npy` data unless force flags are passed.
- If you update trajectory definitions or material parameters, regenerate downstream stages for consistency.

## 6) Key Files

### Solvers

```bash
fom_solver_rve.py
prom_solver_rve.py
hprom_solver_rve.py
hprom_rbf_solver_rve.py
hprom_gpr_solver_rve.py
prom_ann_solver_rve.py
prom_rbf_solver_rve.py
prom_gpr_solver_rve.py
prom_dl_solver_rve.py
```

### Stage Launchers

```bash
stage0_training_trajectory.py
stage1_training_fom_solver_rve.py
stage2_pod_rve.py
stage3_verification_rve.py
stage4_test_rve.py
stage5_build_ecm_dataset.py
stage5_compute_ecm_weights.py
stage6_test_hprom.py
stage7a_prepare_ann_rbf_dataset.py
stage7a_prepare_pod_dl_dataset.py
stage7b_train_ann_manifold.py
stage7b_train_rbf_manifold.py
stage7b_train_sparse_gpr_manifold.py
stage7b_train_pod_dl_manifold.py
stage7c_test_ann_rom.py
stage7c_test_rbf_rom.py
stage7c_test_pod_dl_rom.py
stage7c_compare_bounds.py
stage8_test_prom_ann.py
stage8_test_prom_rbf.py
stage8_test_prom_gpr.py
stage8_test_prom_dl.py
stage9_build_ecm_dataset_rbf.py
stage9_build_ecm_dataset_gpr.py
stage9_compute_ecm_weights_rbf.py
stage9_compute_ecm_weights_gpr.py
stage10_test_hprom_rbf.py
stage10_test_hprom_gpr.py
```

### Inputs

```bash
ProjectParameters.json
rve_geometry.mdpa
```

## 7) Consistency Checklist Before Running

Before running the full workflow, check that the following are consistent:

- The material parameters in `ProjectParameters.json` match the values documented here.
- The trajectory settings in Stage 0 match the intended strain domain.
- If the mesh changes, regenerate Stage 1 and all downstream stages.
- If the POD dimension changes, regenerate Stage 2 and all downstream reduced models.
- If the ANN, RBF, sparse-GPR, or POD-DL architecture changes, regenerate the corresponding Stage 7 model and downstream benchmarks.
- If ECM weights are regenerated, rerun the corresponding HPROM benchmark.
