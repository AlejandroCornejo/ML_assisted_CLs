import os
import sys
import numpy as np
import tqdm
import argparse

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    setup_kratos_parameters,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    DeformationGradientFromGreenLagrange2D,
)


def _build_affine_lifting_helpers(mesh_name="rve_geometry"):
    """Build coordinate maps per equation ID for exact finite-deformation lifting."""
    parameters = setup_kratos_parameters(mesh_name)
    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()

    n_total_dofs, eq_map_runtime, _ = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    sim._InitializeDomainCenterIfNeeded(mp)
    x0c, y0c = sim._x0c, sim._y0c

    dof_x = np.zeros(n_total_dofs, dtype=float)
    dof_y = np.zeros(n_total_dofs, dtype=float)
    is_x_dof = np.zeros(n_total_dofs, dtype=bool)

    for i, node in enumerate(mp.Nodes):
        xr = node.X0 - x0c
        yr = node.Y0 - y0c
        ix = int(eq_map_runtime[i, 0])
        iy = int(eq_map_runtime[i, 1])
        if 0 <= ix < n_total_dofs:
            dof_x[ix] = xr
            dof_y[ix] = yr
            is_x_dof[ix] = True
        if 0 <= iy < n_total_dofs:
            dof_x[iy] = xr
            dof_y[iy] = yr
            is_x_dof[iy] = False

    sim.Finalize()
    return n_total_dofs, dof_x, dof_y, is_x_dof, eq_map_runtime


def _compute_affine_free_displacement(E, x_free, y_free, is_x_free):
    """u_aff = (F-I)X restricted to free DOFs."""
    F = DeformationGradientFromGreenLagrange2D(E)
    ux = (F[0, 0] - 1.0) * x_free + F[0, 1] * y_free
    uy = F[1, 0] * x_free + (F[1, 1] - 1.0) * y_free
    return np.where(is_x_free, ux, uy)


def prepare_ann_dataset(n_primary=3, include_macro_strain_input=False):
    basis_dir = "stage_2_pod_rve"
    train_dir = "stage_1_training_set_fom"
    out_dir = "stage_7_ann_data"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading Rank-9 basis and filters from {basis_dir}...")
    phi = np.load(os.path.join(basis_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(basis_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(basis_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(basis_dir, "eq_map.npy"))
    
    n_total = int(phi.shape[1])
    n_total_dofs = int(len(free_dofs) + len(dir_dofs))
    n_p = int(n_primary)
    if n_p < 1 or n_p >= n_total:
        raise ValueError(f"n_primary must satisfy 1 <= n_primary < {n_total}. Got {n_p}.")
    n_s = int(n_total - n_p)
    
    phi_p = phi[:, :n_p]
    phi_s = phi[:, n_p:n_p + n_s]
    
    print(f"Partition: n_p={n_p}, n_s={n_s} (Total={n_p+n_s})")
    print(f"ANN input features: {'q_p + [Exx,Eyy,Gxy]' if include_macro_strain_input else 'q_p only'}")

    # Affine lifting geometry (exact finite-deformation mapping used in Stage 2)
    n_total_runtime, dof_x, dof_y, is_x_dof, eq_map_runtime = _build_affine_lifting_helpers()
    if n_total_runtime != n_total_dofs:
        raise RuntimeError(
            f"DOF mismatch between runtime model ({n_total_runtime}) and POD metadata ({n_total_dofs})."
        )
    if eq_map_runtime.shape == eq_map.shape and not np.array_equal(eq_map_runtime, eq_map):
        raise RuntimeError("eq_map mismatch between runtime model and Stage 2 metadata.")
    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]

    # Collect all projection pairs
    q_p_list = []
    q_s_list = []
    x_ann_list = []

    # Walk through subdirectories
    traj_dirs = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and d.startswith("trajectory_")
    ]
    traj_dirs.sort(key=lambda x: int(x.split('_')[1]))

    print(f"Projecting snapshots from {len(traj_dirs)} trajectory directories...")
    for d in tqdm.tqdm(traj_dirs):
        idx = d.split('_')[1]
        snap_file = os.path.join(train_dir, d, f"trajectory_{idx}_U.npy")
        strain_file = os.path.join(train_dir, d, f"trajectory_{idx}_applied_strain.npy")
        
        if not os.path.exists(snap_file) or not os.path.exists(strain_file):
            continue
            
        U = np.load(snap_file)
        E_hist = np.load(strain_file)
        if U.ndim != 2:
            continue

        # Stage 1 stores U as (n_steps, n_total_dofs). Keep backward compatibility
        # if legacy data is transposed as (n_total_dofs, n_steps).
        if U.shape[1] == n_total_dofs:
            U_free = U[:, free_dofs]
        elif U.shape[0] == n_total_dofs:
            U_free = U[free_dofs, :].T
        else:
            continue

        if E_hist.ndim != 2 or E_hist.shape[1] != 3:
            continue

        n_steps = min(U_free.shape[0], E_hist.shape[0])
        if n_steps <= 0:
            continue
        U_free = U_free[:n_steps, :]
        E_hist = E_hist[:n_steps, :]

        if U_free.shape[1] != phi.shape[0]:
            continue

        # Use fluctuation snapshots (w = u - u_aff), consistent with Stage 2 POD.
        qp = np.empty((n_steps, n_p), dtype=float)
        qs = np.empty((n_steps, n_s), dtype=float)
        for k in range(n_steps):
            u_aff_free = _compute_affine_free_displacement(E_hist[k], x_free, y_free, is_x_free)
            w_free = U_free[k] - u_aff_free
            qp[k, :] = w_free @ phi_p
            qs[k, :] = w_free @ phi_s

        q_p_list.append(qp)
        q_s_list.append(qs)
        if include_macro_strain_input:
            x_ann_list.append(np.hstack([qp, E_hist]))
        else:
            x_ann_list.append(qp)

    if not q_p_list:
        raise RuntimeError("No valid snapshots found to build ANN dataset.")

    # Concatenate all
    X_qp = np.vstack(q_p_list)
    Y = np.vstack(q_s_list)
    X_ann = np.vstack(x_ann_list)

    print(f"Dataset summary:")
    print(f"  Input (q_p): {X_qp.shape}")
    print(f"  Input (ANN): {X_ann.shape}")
    print(f"  Output (q_s): {Y.shape}")

    # Save raw reduced coordinates and ANN input
    np.save(os.path.join(out_dir, "q_p_train.npy"), X_qp)
    np.save(os.path.join(out_dir, "q_s_train.npy"), Y)
    np.save(os.path.join(out_dir, "ann_input_train.npy"), X_ann)
    
    # Also save the basis partition for the solver
    np.save(os.path.join(out_dir, "phi_p.npy"), phi_p)
    np.save(os.path.join(out_dir, "phi_s.npy"), phi_s)
    np.savez(
        os.path.join(out_dir, "ann_dataset_metadata.npz"),
        n_primary=np.array([n_p], dtype=np.int64),
        n_secondary=np.array([n_s], dtype=np.int64),
        input_dim=np.array([X_ann.shape[1]], dtype=np.int64),
        include_macro_strain_input=np.array([1 if include_macro_strain_input else 0], dtype=np.int64),
    )

    print(f"Dataset successfully saved to {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7a (ANN/RBF): prepare manifold dataset")
    parser.add_argument(
        "--n-primary",
        type=int,
        default=4,
        help="Number of primary POD modes (n_p). Secondary modes are n_total - n_p.",
    )
    parser.add_argument(
        "--with-strain-input",
        action="store_true",
        help="Append macro strain [Exx,Eyy,Gxy] to ANN input features.",
    )
    args = parser.parse_args()
    prepare_ann_dataset(n_primary=args.n_primary, include_macro_strain_input=args.with_strain_input)
