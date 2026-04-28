# Neo-Hookean RVE Workflow (Stages 0 to 8)

This folder contains a complete Reduced Order Modeling workflow for a 2D Neo-Hookean RVE with a hole:

- FOM data generation
- POD-based PROM
- ECM-based HPROM
- ANN-manifold PROM (PROM-ANN)
- RBF-manifold PROM (PROM-RBF)
- POD-DL / POD-AE manifold PROM (PROM-POD-DL)

The goal of this README is handoff: someone new should be able to run the pipeline in order and understand what each stage does.

## 1) Problem Setup & Data Formats

This section is vital for ML integration (e.g., KANs) to understand the input/output pairs.

### 1.1 Mechanics Formulation
- **Element type**: `TotalLagrangianElement2D6N` (2D Plane Strain).
- **Constitutive law**: `HyperElasticPlaneStrain2DLaw` (Neo-Hookean).
- **Material parameters**: `YOUNG_MODULUS = 1628 MPa`, `POISSON_RATIO = 0.4`.
- **Strain Measure**: Green-Lagrange Strain Tensor, $\mathbf{E} = \frac{1}{2}(\mathbf{C} - \mathbf{I})$.
- **Stress Measure**: Second Piola-Kirchhoff (PK2) Stress Tensor, $\mathbf{S}$.

### 1.2 Data Array Shapes (`.npy` files)
The FOM Stage 1 generates trajectories (e.g., `trajectory_1`) containing `.npy` arrays over $N$ time steps:
- **`trajectory_i_U.npy`**: Shape `(N_steps, N_DoFs)` (e.g., `16802 x 4244`). Contains the full nodal displacement field (affine + fluctuation, $\mathbf{u} = \mathbf{u}_{\text{aff}} + \mathbf{w}$) over time. The variable $N_{DoFs}$ is the total number of free degrees of freedom in the mesh.
- **Homogenized Tensors**: Shape `(N_steps, 3)`. The `3` columns correspond to the 2D Voigt notation `[XX, YY, XY]`. Keep in mind the shear component is engineering shear ($\gamma_{xy} = 2 E_{xy}$).
  - **`trajectory_i_applied_strain.npy`**: The macroscopic Green-Lagrange strain ($\bar{\mathbf{E}}$) imposed on the RVE boundaries.
  - **`trajectory_i_strain.npy`**: The macroscopic Green-Lagrange strain computed from the RVE (matches applied strain).
  - **`trajectory_i_stress.npy`**: The macroscopic (volume-averaged) Second Piola-Kirchhoff stress ($\bar{\mathbf{S}}$) resulting from the homogenization of the RVE.

### 1.3 Boundary Lifting Setup
Macro strain is prescribed as:
\[
\mathbf{E} = [E_{xx}, E_{yy}, G_{xy}]^T,\quad G_{xy}=2E_{12}
\]

To apply this as a boundary displacement condition on the RVE nodes, we compute the Deformation Gradient ($\mathbf{F}$):
\[
\mathbf{C}=2\mathbf{E}+\mathbf{I},\quad \mathbf{F}=\sqrt{\mathbf{C}},\quad
\mathbf{u}_{\text{aff}}=(\mathbf{F}-\mathbf{I})(\mathbf{X}-\mathbf{X}_c)
\]

**ML Note (Why $\mathbf{F} = \sqrt{\mathbf{C}}$?):** While the ML datasets track $\mathbf{E}$, finite-element solvers apply spatial displacements via $\mathbf{F}$. By defining $\mathbf{F}$ as the square root of the right Cauchy-Green tensor ($\mathbf{C}$), we enforce $\mathbf{F} = \mathbf{U}$ (the pure stretch tensor from the polar decomposition $\mathbf{F}=\mathbf{RU}$). This artificially strips away any rigid body rotations ($\mathbf{R} = \mathbf{I}$). Eliminating rigid body rotations is crucial for ML training because rotations generate zero stress/strain energy. Removing them ensures a strictly unique, one-to-one mapping between the kinematics and the constitutive response.

The total displacement is decomposed as:
\[
\mathbf{u} = \mathbf{u}_{\text{aff}} + \mathbf{w},\quad \mathbf{w}_D = 0
\]
where \(\mathbf{w}\) is the fluctuation field used for reduction.

## 2) Stage Dependency Graph

Core chain:

`Stage0 -> Stage1(FOM) -> Stage2 -> Stage3/4 -> Stage5a -> Stage5b -> Stage6`

ANN chain:

`Stage1(FOM) + Stage2 -> Stage7a -> Stage7b -> Stage8`

RBF chain:

`Stage1(FOM) + Stage2 -> Stage7a -> Stage7b-RBF -> Stage7c -> Stage8-RBF`

POD-DL chain:

`Stage1(FOM) + Stage2 -> Stage7d -> Stage7c -> Stage8c`

Final benchmark:

`Stage8` is the single PROM-ANN benchmark entrypoint.  
It can run:
- PROM-ANN only, or
- PROM-ANN + local FOM/HPROM baselines inside `stage_8_prom_ann_results`.

## 3) Main Reduced Models (Formulas)

### 3.1 Linear PROM (Galerkin)

With POD basis \(\Phi_f \in \mathbb{R}^{n_f \times r}\):
\[
\mathbf{w}_f \approx \Phi_f \mathbf{q}
\]
\[
\mathbf{r}_r = \Phi_f^T \mathbf{r}_f,\quad
\mathbf{K}_r = \Phi_f^T \mathbf{K}_{ff}\Phi_f,\quad
\mathbf{K}_r \Delta\mathbf{q} = \mathbf{r}_r
\]

Note: in this codebase, the assembled `rhs` sign convention is such that the Newton update is written with `+rhs` (equivalent to the usual `-R` form).

### 3.2 HPROM (ECM)

Element contributions are weighted by ECM cubature weights:
\[
\sum_{e\in\mathcal{Z}} \omega_e \, \mathbf{r}_e,\quad
\sum_{e\in\mathcal{Z}} \omega_e \, \mathbf{K}_e
\]
then projected to reduced coordinates as in PROM.

### 3.3 PROM-ANN (Galerkin on nonlinear manifold)

Split POD basis:
\[
\Phi_f = [\Phi_p \ \Phi_s],\quad \dim(\mathbf{q}_p)=3,\ \dim(\mathbf{q}_s)=6
\]
Train ANN:
\[
\mathbf{q}_s = \mathcal{N}(\mathbf{q}_p)
\]
Use shifted manifold to pass through origin:
\[
\hat{\mathcal{N}}(\mathbf{q}_p)=\mathcal{N}(\mathbf{q}_p)-\mathcal{N}(\mathbf{0})
\]
Decoder:
\[
\mathbf{w}_f(\mathbf{q}_p)=\Phi_p\mathbf{q}_p+\Phi_s\hat{\mathcal{N}}(\mathbf{q}_p)
\]
Manifold Jacobian:
\[
\mathbf{J}_m = \frac{\partial \mathbf{w}_f}{\partial \mathbf{q}_p}
= \Phi_p + \Phi_s \frac{\partial \hat{\mathcal{N}}}{\partial \mathbf{q}_p}
\]
Reduced equations:
\[
\mathbf{r}_r = \mathbf{J}_m^T\mathbf{r}_f,\quad
\mathbf{K}_r \approx \mathbf{J}_m^T\mathbf{K}_{ff}\mathbf{J}_m,\quad
\mathbf{K}_r\Delta\mathbf{q}_p=\mathbf{r}_r
\]

### 3.4 PROM-RBF (Galerkin on nonlinear manifold)

Use the same split:
\[
\Phi_f = [\Phi_p \ \Phi_s],\quad \mathbf{q}_s=\mathcal{R}(\mathbf{q}_p[, \mathbf{E}])
\]
where \(\mathcal{R}\) is a compact-center RBF map trained in Stage 7b-RBF.
As with PROM-ANN, the shifted map is used online:
\[
\hat{\mathcal{R}}(\mathbf{q}_p)=\mathcal{R}(\mathbf{q}_p)-\mathcal{R}(\mathbf{0})
\]
or \(\mathcal{R}([\mathbf{q}_p,\mathbf{E}])-\mathcal{R}([\mathbf{0},\mathbf{E}])\) if macro strain is part of the input.
The reduced solve uses:
\[
\mathbf{r}_r = \mathbf{J}_m^T\mathbf{r}_f,\quad
\mathbf{K}_r \approx \mathbf{J}_m^T\mathbf{K}_{ff}\mathbf{J}_m
\]
with \(\mathbf{J}_m=\Phi_p+\Phi_s\frac{\partial \hat{\mathcal{R}}}{\partial \mathbf{q}_p}\).

### 3.5 PROM-POD-DL / POD-AE (Galerkin on latent manifold)

Train an autoencoder in POD space:
\[
\hat{\mathbf{q}} = \mathcal{D}(\mathcal{E}(\mathbf{q}))
\]
with scaling embedded inside the network (part of the model layers).

Online manifold in latent space \(\mathbf{z}\):
\[
\mathbf{q}(\mathbf{z}) = \mathcal{D}(\mathbf{z}) - \mathbf{q}_{\text{ref}}, \qquad
\mathbf{w}_f(\mathbf{z}) = \Phi_q \mathbf{q}(\mathbf{z})
\]
where \(\mathbf{q}_{\text{ref}}=\mathcal{D}(\mathcal{E}(\mathbf{0}))\) enforces origin anchoring.

Manifold Jacobian:
\[
\mathbf{J}_m = \frac{\partial \mathbf{w}_f}{\partial \mathbf{z}}
= \Phi_q \frac{\partial \mathbf{q}}{\partial \mathbf{z}}
\]

Reduced equations:
\[
\mathbf{r}_r = \mathbf{J}_m^T\mathbf{r}_f,\quad
\mathbf{K}_r \approx \mathbf{J}_m^T\mathbf{K}_{ff}\mathbf{J}_m,\quad
\mathbf{K}_r\Delta\mathbf{z}=\mathbf{r}_r
\]

## 4) How to Run (Recommended Order)

Always run from:

```bash
cd /home/kratos/ML_assisted_CLs/homogenization_NeoHookean_using_Kratos_Hole_Tension
```

### Stage 0 - Generate training trajectories

Default setup is the box domain (rectangular limits), matching the 200% tension / 10% compression / 10% shear setup:

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
You can run `python3 stage0_training_trajectory.py` directly; these are now the defaults.

Primary output: `stage_0_trajectory/stage_0_trajectories.npz`

### Stage 1 - Generate FOM snapshots (training set)

```bash
python3 stage1_training_fom_solver_rve.py \
  --stage0-file stage_0_trajectory/stage_0_trajectories.npz \
  --which all \
  --out-dir stage_1_training_set_fom
```

**Convergence safety:** Stage 1 will immediately abort if the Newton-Raphson solver fails to converge on any trajectory. If this happens, re-run Stage 0 with a higher `--ref-steps` (e.g., 400) to reduce strain increments.

Per-trajectory outputs: `trajectory_i_U.npy`, `trajectory_i_applied_strain.npy`, `trajectory_i_stress.npy`, etc.

### Stage 2 - Build POD basis (free fluctuations)

```bash
python3 stage2_pod_rve.py
```

Output folder: `stage_2_pod_rve/`  
Key files: `pod_basis_free.npy`, `free_dofs.npy`, `dirichlet_dofs.npy`, `eq_map.npy`, `domain_center.npy`

### Stage 3 - PROM verification (seen trajectory index)

```bash
python3 stage3_verification_rve.py --index 9
```

### Stage 4 - PROM generalization test (unseen path)

```bash
python3 stage4_test_rve.py --run-fom
```

If FOM is already cached, you can run without `--run-fom`.

### Stage 5a - Build ECM dataset

```bash
python3 stage5_build_ecm_dataset.py
```

Output: `stage_5_ecm_dataset/` (`Q_ecm.dat`, `b_full.dat`, `C_hom.dat`, `b_hom.dat`, `meta.npz`)

### Stage 5b - Compute ECM weights

```bash
python3 stage5_compute_ecm_weights.py
```

Output: `stage_5_hprom_data/ecm_weights_all.npz`

### Stage 6 - FOM vs PROM vs HPROM benchmark

```bash
python3 stage6_test_hprom.py --run-fom --run-prom
```

Notes:
- Stage 6 is now self-contained (not dependent on Stage 4 outputs).
- You can skip forced reruns and use cache:
  ```bash
  python3 stage6_test_hprom.py
  ```

### Stage 7a - Build ANN/RBF manifold dataset

```bash
python3 stage7a_prepare_ann_rbf_dataset.py
```

This stage projects fluctuation snapshots (after subtracting affine lifting) to:
- inputs: `q_p` (3 modes)
- targets: `q_s` (6 modes)

### Stage 7a - Build POD-DL dataset

```bash
python3 stage7a_prepare_pod_dl_dataset.py \
  --fom-dir stage_1_training_set_fom \
  --basis-dir stage_2_pod_rve \
  --out-dir stage_7_pod_dl_data \
  --q-dim 9
```

Outputs in `stage_7_pod_dl_data/`: `q_dataset.npy`, `phi_q.npy`, `pod_dl_dataset_metadata.npz`.

### Stage 7b - Train manifold ANN

```bash
python3 stage7b_train_ann_manifold.py
```

Outputs in `stage_7_ann_data/`: `manifold_ann.pt`, `manifold_ann_metadata.npz`

### Stage 7b-RBF - Train compact-center RBF manifold

```bash
python3 stage7b_train_rbf_manifold.py \
  --data-dir stage_7_ann_data \
  --out-dir stage_7_rbf_data \
  --max-centers 4000 \
  --kernel gaussian \
  --epsilon 0.0 \
  --ridge 1e-10
```

Outputs in `stage_7_rbf_data/`: `rbf_model.npz`, `phi_p.npy`, `phi_s.npy`, `training_summary.txt`

With K-fold grid search (gaussian-only, ridge fixed at `1e-10`):

```bash
python3 stage7b_train_rbf_manifold.py \
  --data-dir stage_7_ann_data \
  --out-dir stage_7_rbf_data \
  --max-centers 4000 \
  --kernel gaussian \
  --epsilon 0.0 \
  --ridge 1e-10 \
  --grid-search \
  --grid-kernels gaussian \
  --grid-eps-values 0.01,0.05,0.1,0.25,0.5,1.0,2.0,5.0 \
  --grid-ridges 1e-10 \
  --grid-folds 5 \
  --grid-max-samples 110000
```
Important: keep comma-separated CLI values without spaces in bash.

### Stage 7b-POD-DL - Train POD-DL / POD-AE manifold

```bash
python3 stage7b_train_pod_dl_manifold.py \
  --data-dir stage_7_pod_dl_data \
  --fom-dir stage_1_training_set_fom \
  --basis-dir stage_2_pod_rve \
  --out-dir stage_7_pod_dl_data \
  --q-dim 9 \
  --latent-dim 4 \
  --latent-sweep 3,4,5,6
```

Outputs in `stage_7_pod_dl_data/`: `pod_dl_autoencoder.pt`, `phi_q.npy`, `training_summary.txt`.

### Stage 7c - Pure reconstruction check on one training trajectory

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

Outputs in `stage_7c_reconstruction_results/<model>_traj_<idx>/`.

Bounds comparison in one run (POD-4, POD-9, ANN, RBF, POD-DL):

```bash
python3 stage7c_compare_bounds.py \
  --trajectory-index 1 \
  --pod-dl-data-dir stage_7_pod_dl_data
```

Outputs in `stage_7c_reconstruction_results/bounds_traj_<idx>/`.

### Stage 8 - PROM-ANN benchmark (single entrypoint)

Stage 8 uses the same benchmark trajectory logic as Stages 4/6:
- multi-view trajectory plot
- dynamic segment stepping
- reference step level `400`

Generate only the trajectory plot + step report:

```bash
python3 stage8_test_prom_ann.py --plot-only
```

PROM-ANN only:

```bash
python3 stage8_test_prom_ann.py --no-compare
```

PROM-ANN plus local FOM/HPROM comparison (stored in `stage_8_prom_ann_results`):

```bash
python3 stage8_test_prom_ann.py
```

Force recompute Stage 8 local baselines:

```bash
python3 stage8_test_prom_ann.py --run-fom --run-hprom
```

Default Stage 8 trajectory mode is already the Stage-6 dense waypoint path.

### Stage 8-RBF - PROM-RBF benchmark (single entrypoint)

PROM-RBF only:

```bash
python3 stage8_test_prom_rbf.py --no-compare
```

PROM-RBF + local FOM/HPROM comparison:

```bash
python3 stage8_test_prom_rbf.py
```

Force recompute Stage 8-RBF local baselines:

```bash
python3 stage8_test_prom_rbf.py --run-fom --run-hprom
```

Default Stage 8-RBF trajectory mode is already the Stage-6 dense waypoint path.

### Stage 8-POD-DL - PROM-POD-DL benchmark (single entrypoint)

PROM-POD-DL only:

```bash
python3 stage8_test_prom_dl.py --no-compare
```

PROM-POD-DL + local FOM/HPROM comparison:

```bash
python3 stage8_test_prom_dl.py
```

Force recompute Stage 8-POD-DL local baselines:

```bash
python3 stage8_test_prom_dl.py --run-fom --run-hprom
```

Default Stage 8-POD-DL trajectory mode is already the Stage-6 dense waypoint path.

### Stage 9a - Build ECM dataset for HPROM-RBF

Uses the same stratified snapshot sampling logic as Stage 5a, but projects
element residuals with the RBF-manifold tangent \(J_m\) instead of linear POD basis.

```bash
python3 stage9_build_ecm_dataset_rbf.py
```

Output folder: `stage_9_ecm_dataset_rbf/` (`Q_ecm.dat`, `b_full.dat`, `C_hom.dat`, `b_hom.dat`, `meta.npz`)

### Stage 9b - Compute ECM weights for HPROM-RBF

Uses the same RSVD + ECM workflow as Stage 5b, applied to Stage 9a data.

```bash
python3 stage9_compute_ecm_weights_rbf.py
```

Output: `stage_9_hprom_rbf_data/ecm_weights_all.npz`

### Stage 10 - FOM vs PROM-RBF vs HPROM-RBF benchmark

```bash
python3 stage10_test_hprom_rbf.py
```

Optional forced recompute:

```bash
python3 stage10_test_hprom_rbf.py --run-fom --run-prom-rbf --run-hprom-rbf
```

`stage7c_reconstruction_check.py` remains the shared implementation.  
Use the dedicated entrypoints `stage7c_test_ann_rom.py`, `stage7c_test_rbf_rom.py`, and `stage7c_test_pod_dl_rom.py` for runs.

## 5) Practical Notes

- Scripts expect to run from this folder (relative paths are used).
- Most scripts add Kratos binary path internally:
  `/home/kratos/Kratos_Eigen_Check/bin/Release`
- FOM/PROM/HPROM/PROM-ANN reuse previous-step stiffness in Newton iteration 0 by default.
  You can disable from CLI with `--no-old-stiffness-first-it` in `fom_solver_rve.py`, `prom_solver_rve.py`, `stage8_test_prom_ann.py`, `stage8_test_prom_rbf.py`, and `stage8_test_prom_dl.py`.
- Stage outputs are cache-friendly. Re-running often reuses existing `.npy` data unless force flags are passed.
- If you update trajectory definitions or material parameters, regenerate downstream stages for consistency.

## 6) Key Files

- Solvers:
  - `fom_solver_rve.py`
  - `prom_solver_rve.py`
  - `hprom_solver_rve.py`
  - `hprom_rbf_solver_rve.py`
  - `prom_ann_solver_rve.py`
  - `prom_rbf_solver_rve.py`
  - `prom_dl_solver_rve.py`
- Stage launchers:
  - `stage0_training_trajectory.py` ... `stage8_test_prom_ann.py`
  - `stage8_test_prom_rbf.py`
  - `stage8_test_prom_dl.py`
  - `stage9_build_ecm_dataset_rbf.py`
  - `stage9_compute_ecm_weights_rbf.py`
  - `stage10_test_hprom_rbf.py`
- Inputs:
  - `ProjectParameters.json`
  - `rve_geometry.mdpa`
