import os
import sys
import argparse
import numpy as np
import tqdm
from scipy.spatial import Delaunay, QhullError, cKDTree

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


def _load_training_fluctuations(phi, basis_dir, train_dir, mesh_name="rve_geometry"):
    free_dofs = np.load(os.path.join(basis_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(basis_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(basis_dir, "eq_map.npy"))

    n_total_dofs = int(len(free_dofs) + len(dir_dofs))
    n_total_runtime, dof_x, dof_y, is_x_dof, eq_map_runtime = _build_affine_lifting_helpers(mesh_name)

    if n_total_runtime != n_total_dofs:
        raise RuntimeError(
            f"DOF mismatch between runtime model ({n_total_runtime}) and POD metadata ({n_total_dofs})."
        )
    if eq_map_runtime.shape == eq_map.shape and not np.array_equal(eq_map_runtime, eq_map):
        raise RuntimeError("eq_map mismatch between runtime model and Stage 2 metadata.")

    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]

    traj_dirs = [
        d
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and d.startswith("trajectory_")
    ]
    traj_dirs.sort(key=lambda x: int(x.split("_")[1]))

    w_list = []
    e_list = []

    print(f"Collecting fluctuation snapshots from {len(traj_dirs)} trajectory directories...")
    for d in tqdm.tqdm(traj_dirs):
        idx = d.split("_")[1]
        snap_file = os.path.join(train_dir, d, f"trajectory_{idx}_U.npy")
        strain_file = os.path.join(train_dir, d, f"trajectory_{idx}_applied_strain.npy")

        if not os.path.exists(snap_file) or not os.path.exists(strain_file):
            continue

        U = np.load(snap_file)
        E_hist = np.load(strain_file)
        if U.ndim != 2:
            continue

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

        W = np.empty_like(U_free)
        for k in range(n_steps):
            u_aff_free = _compute_affine_free_displacement(E_hist[k], x_free, y_free, is_x_free)
            W[k, :] = U_free[k, :] - u_aff_free

        w_list.append(W)
        e_list.append(E_hist)

    if not w_list:
        raise RuntimeError("No valid snapshots found to build LS-RBF dataset.")

    return np.vstack(w_list), np.vstack(e_list)


def _build_ls_targets(q_full, e_hist, n_primary):
    n_p = int(n_primary)
    n_from_strain = min(3, n_p)

    blocks = [e_hist[:, :n_from_strain]]
    if n_p > n_from_strain:
        n_extra = n_p - n_from_strain
        blocks.append(q_full[:, :n_extra])

    mu = np.hstack(blocks)
    return mu, n_from_strain


def _fit_ls_primary_basis(phi, q_full, mu, n_primary, sample_mask=None):
    n_p = int(n_primary)
    n_samples = int(q_full.shape[0])
    if sample_mask is None:
        idx = np.arange(n_samples, dtype=np.int64)
    else:
        idx = np.nonzero(np.asarray(sample_mask, dtype=bool))[0].astype(np.int64)
    if idx.size < max(10, 2 * n_p):
        raise RuntimeError(
            f"Not enough samples for LS fit after filtering: {idx.size} samples for n_primary={n_p}."
        )

    q_sel = q_full[idx, :]
    mu_sel = mu[idx, :]

    q_scale = np.std(q_sel, axis=0)
    q_scale = np.where(q_scale < 1e-12, 1.0, q_scale)
    mu_scale = np.std(mu_sel, axis=0)
    mu_scale = np.where(mu_scale < 1e-12, 1.0, mu_scale)

    q_scaled = q_sel / q_scale[None, :]
    mu_scaled = mu_sel / mu_scale[None, :]

    c_scaled, *_ = np.linalg.lstsq(q_scaled, mu_scaled, rcond=None)
    c = c_scaled / q_scale[:, None]

    phi_candidate = phi @ c
    u_p, s_p, _ = np.linalg.svd(phi_candidate, full_matrices=False)

    tol_p = (s_p[0] if s_p.size > 0 else 1.0) * 1e-12
    rank_p = int(np.sum(s_p > tol_p))
    if rank_p < n_p:
        raise RuntimeError(
            f"LS primary basis is rank-deficient: rank={rank_p}, requested n_primary={n_p}."
        )

    phi_p = u_p[:, :n_p]
    return phi_p, c, s_p, idx


def _build_secondary_basis_from_primary(phi, phi_p, n_secondary):
    n_s = int(n_secondary)
    residual_space = phi - phi_p @ (phi_p.T @ phi)
    u_s, s_s, _ = np.linalg.svd(residual_space, full_matrices=False)

    tol_s = (s_s[0] if s_s.size > 0 else 1.0) * 1e-12
    rank_s = int(np.sum(s_s > tol_s))
    if rank_s < n_s:
        raise RuntimeError(
            f"Cannot extract full secondary space: rank={rank_s}, requested n_secondary={n_s}."
        )
    return u_s[:, :n_s]


def _compute_cell_jacobians(mu_nodes, q_nodes, cells, cell_type):
    mu_m = np.asarray(mu_nodes, dtype=float)
    q_m = np.asarray(q_nodes, dtype=float)
    cells_in = np.asarray(cells, dtype=np.int64)
    cells_out = np.array(cells_in, copy=True)

    n_cells = int(cells_out.shape[0])
    det_j_param = np.empty(n_cells, dtype=float)
    det_j_map = np.empty(n_cells, dtype=float)
    cond_j_param = np.empty(n_cells, dtype=float)
    valid = np.ones(n_cells, dtype=bool)

    for e in range(n_cells):
        ids = cells_out[e, :]
        mu_e = mu_m[ids, :]
        q_e = q_m[ids, :]

        if str(cell_type).lower() == "tet":
            ia, ib, ic = 1, 2, 3
        elif str(cell_type).lower() == "hex":
            ia, ib, ic = 1, 3, 4
        else:
            raise ValueError(f"Unsupported cell_type='{cell_type}'.")

        m_e = np.column_stack((mu_e[ia] - mu_e[0], mu_e[ib] - mu_e[0], mu_e[ic] - mu_e[0]))
        det_m = float(np.linalg.det(m_e))

        if det_m < 0.0:
            if str(cell_type).lower() == "tet":
                ids2 = np.array([ids[0], ids[2], ids[1], ids[3]], dtype=np.int64)
            else:
                ids2 = np.array([ids[0], ids[3], ids[2], ids[1], ids[4], ids[7], ids[6], ids[5]], dtype=np.int64)
            mu_e = mu_m[ids2, :]
            q_e = q_m[ids2, :]
            m_e = np.column_stack((mu_e[ia] - mu_e[0], mu_e[ib] - mu_e[0], mu_e[ic] - mu_e[0]))
            det_m = float(np.linalg.det(m_e))
            ids = ids2

        if abs(det_m) < 1e-20 or not np.isfinite(det_m):
            valid[e] = False
            det_j_param[e] = np.nan
            det_j_map[e] = np.nan
            cond_j_param[e] = np.nan
            continue

        q_e_mat = np.column_stack((q_e[ia] - q_e[0], q_e[ib] - q_e[0], q_e[ic] - q_e[0]))
        try:
            a_e = q_e_mat @ np.linalg.inv(m_e)
        except np.linalg.LinAlgError:
            valid[e] = False
            det_j_param[e] = np.nan
            det_j_map[e] = np.nan
            cond_j_param[e] = np.nan
            continue

        det_j_param[e] = det_m
        det_j_map[e] = float(np.linalg.det(a_e))
        cond_j_param[e] = float(np.linalg.cond(m_e))
        cells_out[e, :] = ids

    return det_j_param, det_j_map, cond_j_param, valid, cells_out


def _assign_samples_to_structured_hex_cells(mu3, d_mu, mu_center, grid_shape):
    mu3 = np.asarray(mu3, dtype=float)
    nx, ny, nz = [int(v) for v in grid_shape]
    if nx < 2 or ny < 2 or nz < 2:
        raise RuntimeError(f"Invalid structured grid shape: {grid_shape}")

    xi = (mu3 - mu_center[None, :]) @ d_mu.T
    xi_min = np.min(xi, axis=0)
    xi_max = np.max(xi, axis=0)
    span = np.maximum(xi_max - xi_min, 1e-14)
    t = (xi - xi_min[None, :]) / span[None, :]
    t = np.clip(t, 0.0, 1.0 - 1e-12)

    ix = np.minimum((t[:, 0] * float(nx - 1)).astype(np.int64), nx - 2)
    iy = np.minimum((t[:, 1] * float(ny - 1)).astype(np.int64), ny - 2)
    iz = np.minimum((t[:, 2] * float(nz - 1)).astype(np.int64), nz - 2)

    cid = ix * ((ny - 1) * (nz - 1)) + iy * (nz - 1) + iz
    return cid


def _compute_structured_hex_negative_sample_mask(mu3, qls3, mesh_max_points):
    mu3 = np.asarray(mu3, dtype=float)
    qls3 = np.asarray(qls3, dtype=float)

    xi, d_mu, mu_center, _ = _compute_deformation_factor_matrix(mu3)
    _ = xi  # explicit for readability
    mu_m, _, q_m, cells, grid_shape = _build_structured_hex_mesh_from_mu(
        mu3=mu3,
        qpod3=qls3,
        qls3=qls3,
        d_mu=d_mu,
        mu_center=mu_center,
        mesh_max_points=int(mesh_max_points),
        knn_k=12,
    )
    det_j_param, det_j_map, _, valid, cells = _compute_cell_jacobians(mu_m, q_m, cells, cell_type="hex")

    bad_cell_mask = np.logical_or(~valid, ~np.isfinite(det_j_map))
    bad_cell_mask = np.logical_or(bad_cell_mask, det_j_map <= 0.0)

    sample_cell_ids = _assign_samples_to_structured_hex_cells(mu3, d_mu, mu_center, grid_shape)
    sample_bad = bad_cell_mask[sample_cell_ids]

    return {
        "sample_bad_mask": sample_bad,
        "bad_cell_mask": bad_cell_mask,
        "det_j_map": det_j_map,
        "det_j_param": det_j_param,
        "valid_cells_mask": valid,
        "grid_shape": tuple(int(v) for v in grid_shape),
    }


def _build_primary_secondary_bases(
    phi,
    w_free,
    q_full,
    e_hist,
    n_primary,
    enforce_positive_jacobian_ls=False,
    jacobian_ls_max_iters=4,
    jacobian_ls_mesh_max_points=4096,
    jacobian_ls_drop_fraction=1.0,
    jacobian_ls_seed=42,
    jacobian_ls_min_samples=6000,
):
    n_total = int(phi.shape[1])
    n_p = int(n_primary)
    n_s = n_total - n_p

    mu, n_from_strain = _build_ls_targets(q_full, e_hist, n_p)

    if (not bool(enforce_positive_jacobian_ls)) or n_p < 3 or mu.shape[1] < 3:
        phi_p, c, s_p, fit_idx = _fit_ls_primary_basis(phi, q_full, mu, n_p, sample_mask=None)
        phi_s = _build_secondary_basis_from_primary(phi, phi_p, n_s)
        ls_info = {
            "enabled": bool(enforce_positive_jacobian_ls),
            "iterations": 1,
            "best_negative_samples": 0,
            "best_negative_cells": 0,
            "active_samples": int(fit_idx.size),
            "total_samples": int(q_full.shape[0]),
        }
        return phi_p, phi_s, c, mu, n_from_strain, s_p, np.ones(q_full.shape[0], dtype=bool), ls_info

    rng = np.random.default_rng(int(jacobian_ls_seed))
    total_samples = int(q_full.shape[0])
    min_keep = int(max(10 * n_p, min(int(jacobian_ls_min_samples), total_samples)))

    active = np.ones(total_samples, dtype=bool)
    best = None
    history = []

    n_iter = int(max(1, jacobian_ls_max_iters))
    for it in range(n_iter):
        phi_p, c, s_p, fit_idx = _fit_ls_primary_basis(phi, q_full, mu, n_p, sample_mask=active)
        q_ls_all = w_free @ phi_p

        jac = _compute_structured_hex_negative_sample_mask(
            mu3=mu[:, :3],
            qls3=q_ls_all[:, :3],
            mesh_max_points=int(jacobian_ls_mesh_max_points),
        )
        bad_samples = np.asarray(jac["sample_bad_mask"], dtype=bool)
        bad_active = np.logical_and(active, bad_samples)
        n_bad_active = int(np.sum(bad_active))
        n_bad_cells = int(np.sum(jac["bad_cell_mask"]))

        history.append(
            {
                "iter": int(it + 1),
                "active_samples": int(np.sum(active)),
                "bad_active_samples": n_bad_active,
                "bad_cells": n_bad_cells,
            }
        )
        print(
            f"[LS-JacobianAware] iter {it + 1}/{n_iter}: "
            f"active={int(np.sum(active))}, bad_samples={n_bad_active}, bad_cells={n_bad_cells}"
        )

        score = (n_bad_active, n_bad_cells, -int(np.sum(active)))
        if (best is None) or (score < best["score"]):
            best = {
                "score": score,
                "phi_p": phi_p.copy(),
                "c": c.copy(),
                "s_p": s_p.copy(),
                "active_mask": active.copy(),
                "n_bad_active": n_bad_active,
                "n_bad_cells": n_bad_cells,
                "grid_shape": jac["grid_shape"],
            }

        if n_bad_active == 0:
            break

        bad_idx = np.nonzero(bad_active)[0]
        if bad_idx.size == 0:
            break

        drop_frac = float(np.clip(jacobian_ls_drop_fraction, 0.0, 1.0))
        n_drop = int(np.ceil(drop_frac * bad_idx.size))
        n_drop = max(1, min(n_drop, bad_idx.size))

        remaining_after = int(np.sum(active)) - n_drop
        if remaining_after < min_keep:
            n_drop = max(0, int(np.sum(active)) - min_keep)
        if n_drop <= 0:
            break

        if n_drop < bad_idx.size:
            drop_sel = np.sort(rng.choice(bad_idx, size=n_drop, replace=False))
        else:
            drop_sel = bad_idx
        active[drop_sel] = False

    if best is None:
        raise RuntimeError("Jacobian-aware LS fit failed to produce a valid basis.")

    phi_p = best["phi_p"]
    c = best["c"]
    s_p = best["s_p"]
    active_mask = best["active_mask"]
    phi_s = _build_secondary_basis_from_primary(phi, phi_p, n_s)

    ls_info = {
        "enabled": True,
        "iterations": int(len(history)),
        "history": history,
        "best_negative_samples": int(best["n_bad_active"]),
        "best_negative_cells": int(best["n_bad_cells"]),
        "active_samples": int(np.sum(active_mask)),
        "total_samples": int(total_samples),
        "grid_shape": best["grid_shape"],
    }

    return phi_p, phi_s, c, mu, n_from_strain, s_p, active_mask, ls_info


def _plot_domain_3d(data, labels, title, out_file):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        s=2,
        alpha=0.25,
        c="#0f766e",
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close(fig)


def _save_domain_comparison_plots(
    out_dir,
    mu_targets,
    q_pod,
    q_ls,
    max_samples=120000,
    seed=42,
):
    n = int(mu_targets.shape[0])
    if n == 0:
        return

    n_use = min(int(max_samples), n)
    if n_use < n:
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(n, size=n_use, replace=False))
        mu_targets = mu_targets[idx, :]
        q_pod = q_pod[idx, :]
        q_ls = q_ls[idx, :]

    _plot_domain_3d(
        mu_targets,
        labels=(r"$\mu_1$", r"$\mu_2$", r"$\mu_3$"),
        title="Domain in target coordinates (mu)",
        out_file=os.path.join(out_dir, "domain_mu_targets_3d.png"),
    )
    _plot_domain_3d(
        q_pod,
        labels=(r"$q_1$", r"$q_2$", r"$q_3$"),
        title="Domain in first 3 POD coordinates",
        out_file=os.path.join(out_dir, "domain_qpod_first3_3d.png"),
    )
    _plot_domain_3d(
        q_ls,
        labels=(r"$q^{LS}_1$", r"$q^{LS}_2$", r"$q^{LS}_3$"),
        title="Domain in first 3 LS-primary coordinates",
        out_file=os.path.join(out_dir, "domain_qls_first3_3d.png"),
    )


def _compute_deformation_factor_matrix(mu_points, eps=1e-12):
    """
    Build the deformation-factor matrix used to scale the parameter domain.
    We use a diagonal scaling so each parameter spans approximately [-1, 1].
    """
    mu_points = np.asarray(mu_points, dtype=float)
    mu_min = np.min(mu_points, axis=0)
    mu_max = np.max(mu_points, axis=0)
    mu_center = 0.5 * (mu_min + mu_max)
    mu_span = np.maximum(mu_max - mu_min, float(eps))

    # xi = D_mu (mu - mu_center), with D_mu diagonal.
    d_mu = np.diag(2.0 / mu_span)
    xi = (mu_points - mu_center[None, :]) @ d_mu.T
    return xi, d_mu, mu_center, mu_span


def _grid_subsample_indices(points, max_points=6000, seed=42):
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    if n <= int(max_points):
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    p_min = np.min(pts, axis=0)
    p_max = np.max(pts, axis=0)
    span = np.maximum(p_max - p_min, 1e-12)
    p_unit = (pts - p_min[None, :]) / span[None, :]

    nbin = int(np.ceil(max_points ** (1.0 / 3.0)))
    nbin = max(nbin, 2)
    keys = np.floor(np.clip(p_unit, 0.0, 1.0 - 1e-12) * float(nbin)).astype(np.int64)

    _, idx_unique = np.unique(keys, axis=0, return_index=True)
    idx_unique = np.asarray(idx_unique, dtype=np.int64)

    if idx_unique.size > int(max_points):
        keep = rng.choice(idx_unique.size, size=int(max_points), replace=False)
        return np.sort(idx_unique[keep])

    if idx_unique.size < int(max_points):
        mask = np.ones(n, dtype=bool)
        mask[idx_unique] = False
        rest = np.nonzero(mask)[0]
        n_add = min(int(max_points) - int(idx_unique.size), int(rest.size))
        if n_add > 0:
            add = rng.choice(rest, size=n_add, replace=False)
            return np.sort(np.concatenate([idx_unique, add]))

    return np.sort(idx_unique)


def _compute_macro_deformation_jacobian(e_hist):
    j_vals = np.empty(e_hist.shape[0], dtype=float)
    for i in range(e_hist.shape[0]):
        f = DeformationGradientFromGreenLagrange2D(e_hist[i, :3])
        j_vals[i] = float(np.linalg.det(f))
    return j_vals


def _idw_interpolate(query_pts, sample_pts, sample_vals, k=12, power=2.0, eps=1e-12):
    q = np.asarray(query_pts, dtype=float)
    x = np.asarray(sample_pts, dtype=float)
    y = np.asarray(sample_vals, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("sample_pts and sample_vals must have the same number of rows.")
    if x.shape[0] < 1:
        raise ValueError("At least one sample point is required for interpolation.")

    tree = cKDTree(x)
    k_eff = int(min(max(1, int(k)), x.shape[0]))
    dist, idx = tree.query(q, k=k_eff)

    if k_eff == 1:
        return y[np.asarray(idx, dtype=np.int64), :]

    dist = np.asarray(dist, dtype=float)
    idx = np.asarray(idx, dtype=np.int64)
    weights = 1.0 / np.maximum(dist, float(eps)) ** float(power)
    wsum = np.sum(weights, axis=1, keepdims=True)
    wsum = np.maximum(wsum, float(eps))
    vals = y[idx, :]
    return np.sum(weights[:, :, None] * vals, axis=1) / wsum


def _build_structured_hex_mesh_from_mu(
    mu3,
    qpod3,
    qls3,
    d_mu,
    mu_center,
    mesh_max_points=6000,
    knn_k=12,
):
    mu3 = np.asarray(mu3, dtype=float)
    qpod3 = np.asarray(qpod3, dtype=float)
    qls3 = np.asarray(qls3, dtype=float)
    if mu3.shape[0] < 8:
        raise RuntimeError("Structured hexa mesh needs at least 8 samples.")

    # Normalized coordinates for robust structured sampling.
    xi = (mu3 - mu_center[None, :]) @ d_mu.T
    xi_min = np.min(xi, axis=0)
    xi_max = np.max(xi, axis=0)

    n_axis = int(np.round(float(mesh_max_points) ** (1.0 / 3.0)))
    n_axis = max(3, n_axis)
    while n_axis ** 3 > int(mesh_max_points) and n_axis > 3:
        n_axis -= 1

    nx = ny = nz = int(n_axis)
    gx = np.linspace(xi_min[0], xi_max[0], nx)
    gy = np.linspace(xi_min[1], xi_max[1], ny)
    gz = np.linspace(xi_min[2], xi_max[2], nz)
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    xi_nodes = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    inv_d = np.diag(1.0 / np.maximum(np.diag(d_mu), 1e-14))
    mu_nodes = xi_nodes @ inv_d.T + mu_center[None, :]

    qpod_nodes = _idw_interpolate(
        query_pts=xi_nodes,
        sample_pts=xi,
        sample_vals=qpod3,
        k=int(knn_k),
        power=2.0,
    )
    qls_nodes = _idw_interpolate(
        query_pts=xi_nodes,
        sample_pts=xi,
        sample_vals=qls3,
        k=int(knn_k),
        power=2.0,
    )

    def _nid(i, j, k):
        return i * (ny * nz) + j * nz + k

    hexes = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                n000 = _nid(i, j, k)
                n100 = _nid(i + 1, j, k)
                n110 = _nid(i + 1, j + 1, k)
                n010 = _nid(i, j + 1, k)
                n001 = _nid(i, j, k + 1)
                n101 = _nid(i + 1, j, k + 1)
                n111 = _nid(i + 1, j + 1, k + 1)
                n011 = _nid(i, j + 1, k + 1)
                hexes.append([n000, n100, n110, n010, n001, n101, n111, n011])

    hexes = np.asarray(hexes, dtype=np.int64)
    if hexes.shape[0] == 0:
        raise RuntimeError("Structured hex mesh produced zero cells.")

    return mu_nodes, qpod_nodes, qls_nodes, hexes, (nx, ny, nz)


def _plot_jacobian_histograms(j_param_det, j_map_det, j_phys, out_dir):
    import matplotlib.pyplot as plt

    def _safe_hist(ax, data, color, title, xlabel):
        x = np.asarray(data, dtype=float)
        x = x[np.isfinite(x)]
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.35)
        if x.size == 0:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
            return

        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if np.isclose(x_min, x_max):
            ax.hist(x, bins=1, color=color, alpha=0.85)
            return

        n_bins = min(70, max(20, int(np.sqrt(x.size))))
        ax.hist(x, bins=n_bins, color=color, alpha=0.85)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _safe_hist(
        axes[0],
        j_param_det,
        color="#0f766e",
        title=r"$\det(J_{\mu})$ (parameter mesh)",
        xlabel=r"$\det(J_{\mu})$",
    )
    _safe_hist(
        axes[1],
        j_map_det,
        color="#1d4ed8",
        title=r"$\det(\partial q^{LS}/\partial \mu)$",
        xlabel=r"$\det(\partial q^{LS}/\partial \mu)$",
    )
    _safe_hist(
        axes[2],
        j_phys,
        color="#b45309",
        title=r"$\det(F)$ from applied strain",
        xlabel=r"$\det(F)$",
    )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "jacobian_histograms_ls_rbf.png"), dpi=220)
    plt.close(fig)


def _extract_unique_edges_from_tets(tets):
    t = np.asarray(tets, dtype=np.int64)
    if t.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    pairs = np.vstack(
        [
            t[:, [0, 1]],
            t[:, [0, 2]],
            t[:, [0, 3]],
            t[:, [1, 2]],
            t[:, [1, 3]],
            t[:, [2, 3]],
        ]
    )
    pairs = np.sort(pairs, axis=1)
    edges = np.unique(pairs, axis=0)
    return edges


def _extract_unique_edges_from_hexes(hexes):
    h = np.asarray(hexes, dtype=np.int64)
    if h.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    local = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int64,
    )
    pairs = np.vstack([h[:, e] for e in local])
    pairs = np.sort(pairs, axis=1)
    edges = np.unique(pairs, axis=0)
    return edges


def _plot_cell_mesh_3d(
    points,
    cells,
    cell_type,
    labels,
    title,
    out_file,
    edge_color="#0f766e",
    node_color="#111827",
    max_edges=22000,
    edge_seed=42,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    pts = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)

    if str(cell_type).lower() == "tet":
        edges = _extract_unique_edges_from_tets(cells)
    elif str(cell_type).lower() == "hex":
        edges = _extract_unique_edges_from_hexes(cells)
    else:
        raise ValueError(f"Unsupported cell_type='{cell_type}'.")

    if edges.shape[0] == 0:
        return

    if edges.shape[0] > int(max_edges):
        rng = np.random.default_rng(int(edge_seed))
        keep = rng.choice(edges.shape[0], size=int(max_edges), replace=False)
        edges = edges[np.sort(keep), :]

    seg = np.stack([pts[edges[:, 0], :], pts[edges[:, 1], :]], axis=1)

    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    lc = Line3DCollection(seg, colors=edge_color, linewidths=0.35, alpha=0.30)
    ax.add_collection3d(lc)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.8, c=node_color, alpha=0.35, linewidths=0)

    pmin = np.min(pts, axis=0)
    pmax = np.max(pts, axis=0)
    span = np.maximum(pmax - pmin, 1e-12)
    pad = 0.05 * span
    lo = pmin - pad
    hi = pmax + pad

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def _plot_tetrahedral_mesh_3d(
    points,
    tets,
    labels,
    title,
    out_file,
    edge_color="#0f766e",
    node_color="#111827",
    max_edges=22000,
    edge_seed=42,
):
    _plot_cell_mesh_3d(
        points=points,
        cells=tets,
        cell_type="tet",
        labels=labels,
        title=title,
        out_file=out_file,
        edge_color=edge_color,
        node_color=node_color,
        max_edges=max_edges,
        edge_seed=edge_seed,
    )


def _plot_hexahedral_mesh_3d(
    points,
    hexes,
    labels,
    title,
    out_file,
    edge_color="#0f766e",
    node_color="#111827",
    max_edges=22000,
    edge_seed=42,
):
    _plot_cell_mesh_3d(
        points=points,
        cells=hexes,
        cell_type="hex",
        labels=labels,
        title=title,
        out_file=out_file,
        edge_color=edge_color,
        node_color=node_color,
        max_edges=max_edges,
        edge_seed=edge_seed,
    )


def _write_mesh_jacobian_tex(
    out_dir,
    d_mu,
    mu_center,
    mu_span,
    n_mesh_nodes,
    n_cells,
    cell_type,
    neg_param_count,
    neg_map_count,
    j_param_abs_min,
    j_map_min,
    j_phys_min,
    j_phys_max,
    mesh_type,
    run_command,
):
    tex_path = os.path.join(out_dir, "ls_rbf_parameter_mesh_jacobian_note.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\documentclass[11pt]{article}" + "\n")
        f.write(r"\usepackage{amsmath,amssymb,booktabs,graphicx}" + "\n")
        f.write(r"\usepackage[margin=1in]{geometry}" + "\n")
        f.write(r"\begin{document}" + "\n")
        f.write(r"\section*{LS-RBF Parameter-Mesh Jacobian Diagnostics}" + "\n")
        f.write(
            r"This note documents the Jacobian checks added to the LS primary-mode selection stage "
            r"for the RBF manifold pipeline." + "\n"
        )
        f.write(r"\subsection*{0. Command used}" + "\n")
        f.write(r"\begin{verbatim}" + "\n")
        f.write(run_command + "\n")
        f.write(r"\end{verbatim}" + "\n")

        f.write(r"\subsection*{1. Deformation-factor matrix}" + "\n")
        f.write(
            r"We define a diagonal deformation-factor (scaling) matrix to normalize the parameter "
            r"domain before meshing:" + "\n"
        )
        f.write(r"\begin{equation}" + "\n")
        f.write(r"\boldsymbol{\xi} = \mathbf{D}_{\mu}(\boldsymbol{\mu}-\boldsymbol{\mu}_c),\qquad ")
        f.write(r"\mathbf{D}_{\mu} = \mathrm{diag}\!\left(\frac{2}{\Delta \mu_1},\frac{2}{\Delta \mu_2},\frac{2}{\Delta \mu_3}\right)." + "\n")
        f.write(r"\end{equation}" + "\n")
        f.write(
            r"This reduces anisotropy in the parameter cloud and improves numerical robustness of "
            r"the mesh and local Jacobians." + "\n"
        )

        f.write(r"\subsection*{2. Parameter mesh and map Jacobians}" + "\n")
        if str(mesh_type).lower() == "structured_hex":
            f.write(
                r"A structured 3D hexahedral mesh is built in the normalized parameter space. "
                r"For each hexahedral cell $e$, using local edge vectors from a reference corner "
                r"in $(\mu,q^{LS})$, we compute a local affine map "
                r"$q^{LS}\approx \mathbf{A}_e\mu+\mathbf{b}_e$." + "\n"
            )
        else:
            f.write(
                r"A 3D Delaunay tetrahedralization is built in the normalized parameter space. "
                r"For each tetrahedron $e$, using four corresponding nodes in $(\mu,q^{LS})$, we compute "
                r"a local affine map $q^{LS}\approx \mathbf{A}_e\mu+\mathbf{b}_e$." + "\n"
            )
        f.write(r"\begin{equation}" + "\n")
        f.write(r"\mathbf{A}_e = \mathbf{Q}_e\mathbf{M}_e^{-1},\qquad ")
        f.write(r"\mathbf{M}_e=[\mu_a-\mu_0\ \mu_b-\mu_0\ \mu_c-\mu_0],\quad ")
        f.write(r"\mathbf{Q}_e=[q_a-q_0\ q_b-q_0\ q_c-q_0]." + "\n")
        f.write(r"\end{equation}" + "\n")
        f.write(
            r"We monitor: (i) $\det(\mathbf{M}_e)$ for parameter-mesh degeneracy, and "
            r"(ii) $\det(\mathbf{A}_e)=\det(\partial q^{LS}/\partial\mu)$ for orientation inversion." + "\n"
        )

        f.write(r"\subsection*{3. Physical deformation Jacobian}" + "\n")
        f.write(
            r"For each applied Green-Lagrange strain sample, we compute $F=\sqrt{2E+I}$ "
            r"and monitor $\det(F)$ to check local admissibility of the macro deformation." + "\n"
        )

        f.write(r"\subsection*{4. Run summary}" + "\n")
        f.write(r"\begin{center}" + "\n")
        f.write(r"\begin{tabular}{ll}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Quantity & Value \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(f"$N_\\text{{mesh nodes}}$ & {int(n_mesh_nodes)} \\\\\n")
        f.write(f"$N_\\text{{cells}}$ & {int(n_cells)} \\\\\n")
        f.write(f"mesh type & {mesh_type} \\\\\n")
        f.write(f"mesh cell type & {cell_type} \\\\\n")
        f.write(f"negative $\\det(J_\\mu)$ count & {int(neg_param_count)} \\\\\n")
        f.write(f"negative $\\det(\\partial q^{{LS}}/\\partial\\mu)$ count & {int(neg_map_count)} \\\\\n")
        f.write(f"$\\min |\\det(J_\\mu)|$ & {j_param_abs_min:.6e} \\\\\n")
        f.write(f"$\\min\\det(\\partial q^{{LS}}/\\partial\\mu)$ & {j_map_min:.6e} \\\\\n")
        f.write(f"$\\min\\det(F)$ & {j_phys_min:.6e} \\\\\n")
        f.write(f"$\\max\\det(F)$ & {j_phys_max:.6e} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{center}" + "\n")

        f.write(r"\subsection*{5. Deformation-factor matrix used}" + "\n")
        f.write(r"\begin{equation}" + "\n")
        f.write(
            r"\mathbf{D}_{\mu}=\mathrm{diag}\left("
            + f"{d_mu[0,0]:.6e},"
            + f"{d_mu[1,1]:.6e},"
            + f"{d_mu[2,2]:.6e}"
            + r"\right)"
        )
        f.write(r"\end{equation}" + "\n")
        f.write(r"\begin{align}" + "\n")
        f.write(
            r"\mu_c &= \left("
            + f"{mu_center[0]:.6e},"
            + f"{mu_center[1]:.6e},"
            + f"{mu_center[2]:.6e}"
            + r"\right),\\"
            + "\n"
        )
        f.write(
            r"\Delta\mu &= \left("
            + f"{mu_span[0]:.6e},"
            + f"{mu_span[1]:.6e},"
            + f"{mu_span[2]:.6e}"
            + r"\right)."
            + "\n"
        )
        f.write(r"\end{align}" + "\n")

        f.write(r"\subsection*{6. Figures}" + "\n")
        f.write(r"\begin{figure}[h!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{domain_mu_targets_3d.png}" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{domain_qls_first3_3d.png}" + "\n")
        f.write(r"\caption{Point clouds in target parameter space and LS-primary reduced space.}" + "\n")
        f.write(r"\end{figure}" + "\n")

        f.write(r"\begin{figure}[h!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{domain_qpod_first3_3d.png}" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{domain_qls_first3_3d.png}" + "\n")
        f.write(r"\caption{Point clouds in POD-first-3 and LS-primary-first-3 coordinates.}" + "\n")
        f.write(r"\end{figure}" + "\n")

        f.write(r"\begin{figure}[h!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{parameter_mesh_mu_edges_3d.png}" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{parameter_mesh_q_pod_edges_3d.png}" + "\n")
        f.write(r"\caption{Parameter-mesh edges and mapped POD-first-3 mesh edges.}" + "\n")
        f.write(r"\end{figure}" + "\n")

        f.write(r"\begin{figure}[h!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{parameter_mesh_q_ls_edges_3d.png}" + "\n")
        f.write(r"\caption{Parameter-mesh edges mapped to LS-primary-first-3 space.}" + "\n")
        f.write(r"\end{figure}" + "\n")

        f.write(r"\begin{figure}[h!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\includegraphics[width=0.70\textwidth]{jacobian_histograms_ls_rbf.png}" + "\n")
        f.write(r"\caption{Histograms of parameter-cell, mapping, and physical Jacobians.}" + "\n")
        f.write(r"\end{figure}" + "\n")

        f.write(r"\end{document}" + "\n")
    return tex_path


def _run_parameter_mesh_jacobian_checks(
    mu_targets,
    q_pod,
    q_ls,
    e_hist,
    out_dir,
    mesh_max_points=6000,
    mesh_seed=42,
    mesh_type="structured_hex",
    run_command="python3 stage7a_prepare_rbf_dataset_ls.py --n-primary 3",
):
    if mu_targets.shape[1] < 3 or q_pod.shape[1] < 3 or q_ls.shape[1] < 3:
        raise ValueError(
            "Jacobian checks require at least 3 target coordinates, 3 POD coordinates, and 3 LS coordinates."
        )

    mu3 = np.asarray(mu_targets[:, :3], dtype=float)
    qpod3 = np.asarray(q_pod[:, :3], dtype=float)
    q3 = np.asarray(q_ls[:, :3], dtype=float)

    xi, d_mu, mu_center, mu_span = _compute_deformation_factor_matrix(mu3)

    mesh_kind = str(mesh_type).lower()
    grid_shape = None
    if mesh_kind == "delaunay":
        idx_sub = _grid_subsample_indices(xi, max_points=int(mesh_max_points), seed=int(mesh_seed))
        xi_m = xi[idx_sub, :]
        mu_m = mu3[idx_sub, :]
        qpod_m = qpod3[idx_sub, :]
        q_m = q3[idx_sub, :]

        # Remove possible duplicates after scaling/rounding to avoid QHull failures.
        key = np.round(xi_m, decimals=12)
        _, idx_unique = np.unique(key, axis=0, return_index=True)
        idx_unique = np.sort(idx_unique.astype(np.int64))
        xi_m = xi_m[idx_unique, :]
        mu_m = mu_m[idx_unique, :]
        qpod_m = qpod_m[idx_unique, :]
        q_m = q_m[idx_unique, :]

        if xi_m.shape[0] < 4:
            raise RuntimeError("Not enough distinct mesh nodes for 3D Delaunay Jacobian checks.")

        try:
            tri = Delaunay(xi_m, qhull_options="QJ Qbb Qc")
        except QhullError as exc:
            raise RuntimeError(f"Delaunay mesh construction failed: {exc}") from exc

        cells = np.asarray(tri.simplices, dtype=np.int64)
        cell_type = "tet"
    elif mesh_kind == "structured_hex":
        mu_m, qpod_m, q_m, cells, grid_shape = _build_structured_hex_mesh_from_mu(
            mu3=mu3,
            qpod3=qpod3,
            qls3=q3,
            d_mu=d_mu,
            mu_center=mu_center,
            mesh_max_points=int(mesh_max_points),
            knn_k=12,
        )
        cell_type = "hex"
    else:
        raise ValueError(
            f"Unsupported mesh_type='{mesh_type}'. Available: 'delaunay', 'structured_hex'."
        )

    n_cells = int(cells.shape[0])
    if n_cells <= 0:
        raise RuntimeError("Mesh construction produced zero cells.")

    det_j_param, det_j_map, cond_j_param, valid, cells = _compute_cell_jacobians(
        mu_nodes=mu_m,
        q_nodes=q_m,
        cells=cells,
        cell_type=cell_type,
    )

    det_j_param_valid = det_j_param[valid]
    det_j_map_valid = det_j_map[valid]
    cond_valid = cond_j_param[valid]

    if det_j_param_valid.size == 0:
        raise RuntimeError("All mesh cells were degenerate in Jacobian checks.")

    j_phys = _compute_macro_deformation_jacobian(e_hist)
    neg_phys = int(np.sum(j_phys <= 0.0))
    neg_map = int(np.sum(det_j_map_valid <= 0.0))
    neg_param = int(np.sum(det_j_param_valid <= 0.0))

    np.save(os.path.join(out_dir, "parameter_mesh_nodes_mu.npy"), mu_m)
    np.save(os.path.join(out_dir, "parameter_mesh_nodes_q_pod.npy"), qpod_m)
    np.save(os.path.join(out_dir, "parameter_mesh_nodes_q_ls.npy"), q_m)
    np.save(os.path.join(out_dir, "parameter_mesh_cells.npy"), cells)
    if cell_type == "tet":
        np.save(os.path.join(out_dir, "parameter_mesh_tetrahedra.npy"), cells)
    else:
        np.save(os.path.join(out_dir, "parameter_mesh_hexahedra.npy"), cells)
        np.save(os.path.join(out_dir, "parameter_mesh_structured_grid_shape.npy"), np.array(grid_shape, dtype=np.int64))
    np.save(os.path.join(out_dir, "jacobian_det_parameter_mesh.npy"), det_j_param)
    np.save(os.path.join(out_dir, "jacobian_det_mu_to_q_ls.npy"), det_j_map)
    np.save(os.path.join(out_dir, "jacobian_cond_parameter_mesh.npy"), cond_j_param)
    np.save(os.path.join(out_dir, "macro_detF_from_applied_strain.npy"), j_phys)
    np.save(os.path.join(out_dir, "deformation_factor_matrix_mu.npy"), d_mu)
    np.save(os.path.join(out_dir, "mu_center_for_scaling.npy"), mu_center)
    np.save(os.path.join(out_dir, "mu_span_for_scaling.npy"), mu_span)
    with open(os.path.join(out_dir, "jacobian_mesh_type.txt"), "w", encoding="utf-8") as f:
        f.write(f"{mesh_kind}\n")
    with open(os.path.join(out_dir, "jacobian_mesh_cell_type.txt"), "w", encoding="utf-8") as f:
        f.write(f"{cell_type}\n")

    fig_mu = os.path.join(out_dir, "parameter_mesh_mu_edges_3d.png")
    fig_qpod = os.path.join(out_dir, "parameter_mesh_q_pod_edges_3d.png")
    fig_qls = os.path.join(out_dir, "parameter_mesh_q_ls_edges_3d.png")

    if cell_type == "tet":
        _plot_tetrahedral_mesh_3d(
            points=mu_m,
            tets=cells[valid, :],
            labels=(r"$\mu_1$", r"$\mu_2$", r"$\mu_3$"),
            title="Tetra mesh in parameter space (mu)",
            out_file=fig_mu,
            edge_color="#0f766e",
            node_color="#111827",
            max_edges=22000,
            edge_seed=mesh_seed,
        )
        _plot_tetrahedral_mesh_3d(
            points=qpod_m,
            tets=cells[valid, :],
            labels=(r"$q_1$", r"$q_2$", r"$q_3$"),
            title="Mapped tetra mesh in POD-first-3 space",
            out_file=fig_qpod,
            edge_color="#7c3aed",
            node_color="#111827",
            max_edges=22000,
            edge_seed=mesh_seed,
        )
        _plot_tetrahedral_mesh_3d(
            points=q_m,
            tets=cells[valid, :],
            labels=(r"$q^{LS}_1$", r"$q^{LS}_2$", r"$q^{LS}_3$"),
            title="Mapped tetra mesh in LS-primary space",
            out_file=fig_qls,
            edge_color="#1d4ed8",
            node_color="#111827",
            max_edges=22000,
            edge_seed=mesh_seed,
        )
    else:
        _plot_hexahedral_mesh_3d(
            points=mu_m,
            hexes=cells[valid, :],
            labels=(r"$\mu_1$", r"$\mu_2$", r"$\mu_3$"),
            title="Structured hexa mesh in parameter space (mu)",
            out_file=fig_mu,
            edge_color="#0f766e",
            node_color="#111827",
            max_edges=28000,
            edge_seed=mesh_seed,
        )
        _plot_hexahedral_mesh_3d(
            points=qpod_m,
            hexes=cells[valid, :],
            labels=(r"$q_1$", r"$q_2$", r"$q_3$"),
            title="Mapped structured hexa mesh in POD-first-3 space",
            out_file=fig_qpod,
            edge_color="#7c3aed",
            node_color="#111827",
            max_edges=28000,
            edge_seed=mesh_seed,
        )
        _plot_hexahedral_mesh_3d(
            points=q_m,
            hexes=cells[valid, :],
            labels=(r"$q^{LS}_1$", r"$q^{LS}_2$", r"$q^{LS}_3$"),
            title="Mapped structured hexa mesh in LS-primary space",
            out_file=fig_qls,
            edge_color="#1d4ed8",
            node_color="#111827",
            max_edges=28000,
            edge_seed=mesh_seed,
        )

    _plot_jacobian_histograms(det_j_param_valid, det_j_map_valid, j_phys, out_dir)

    txt_path = os.path.join(out_dir, "ls_rbf_jacobian_checks_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("LS-RBF parameter mesh and Jacobian diagnostics\n")
        f.write(f"mesh_nodes={mu_m.shape[0]}\n")
        f.write(f"mesh_cells_total={n_cells}\n")
        f.write(f"mesh_cells_valid={int(np.sum(valid))}\n")
        f.write(f"mesh_cells_degenerate={int(np.sum(~valid))}\n")
        f.write(f"mesh_type={mesh_kind}\n")
        f.write(f"mesh_cell_type={cell_type}\n")
        if grid_shape is not None:
            f.write(f"structured_grid_shape={grid_shape[0]},{grid_shape[1]},{grid_shape[2]}\n")
        f.write(f"negative_detJ_param_count={neg_param}\n")
        f.write(f"negative_detJ_map_count={neg_map}\n")
        f.write(f"negative_detF_macro_count={neg_phys}\n")
        f.write(f"min_abs_detJ_param={float(np.nanmin(np.abs(det_j_param_valid))):.16e}\n")
        f.write(f"min_detJ_map={float(np.nanmin(det_j_map_valid)):.16e}\n")
        f.write(f"max_detJ_map={float(np.nanmax(det_j_map_valid)):.16e}\n")
        f.write(f"median_detJ_map={float(np.nanmedian(det_j_map_valid)):.16e}\n")
        f.write(f"min_condJ_param={float(np.nanmin(cond_valid)):.16e}\n")
        f.write(f"max_condJ_param={float(np.nanmax(cond_valid)):.16e}\n")
        f.write(f"min_detF_macro={float(np.min(j_phys)):.16e}\n")
        f.write(f"max_detF_macro={float(np.max(j_phys)):.16e}\n")
        f.write("D_mu_diagonal=" + ",".join([f"{float(d_mu[i,i]):.16e}" for i in range(3)]) + "\n")
        f.write("mu_center=" + ",".join([f"{float(v):.16e}" for v in mu_center]) + "\n")
        f.write("mu_span=" + ",".join([f"{float(v):.16e}" for v in mu_span]) + "\n")

    tex_path = _write_mesh_jacobian_tex(
        out_dir=out_dir,
        d_mu=d_mu,
        mu_center=mu_center,
        mu_span=mu_span,
        n_mesh_nodes=mu_m.shape[0],
        n_cells=n_cells,
        cell_type=cell_type,
        neg_param_count=neg_param,
        neg_map_count=neg_map,
        j_param_abs_min=float(np.nanmin(np.abs(det_j_param_valid))),
        j_map_min=float(np.nanmin(det_j_map_valid)),
        j_phys_min=float(np.min(j_phys)),
        j_phys_max=float(np.max(j_phys)),
        mesh_type=mesh_kind,
        run_command=run_command,
    )

    return {
        "summary_txt": txt_path,
        "summary_tex": tex_path,
        "neg_detJ_param_count": neg_param,
        "neg_detJ_map_count": neg_map,
        "neg_detF_macro_count": neg_phys,
        "min_detJ_map": float(np.nanmin(det_j_map_valid)),
        "min_detF_macro": float(np.min(j_phys)),
        "mesh_type": mesh_kind,
        "mesh_cell_type": cell_type,
    }


def prepare_rbf_dataset_least_squares(
    n_primary=3,
    include_macro_strain_input=False,
    basis_dir="stage_2_pod_rve",
    train_dir="stage_1_training_set_fom",
    out_dir="stage_7_ann_data_ls",
    mesh_name="rve_geometry",
    save_domain_plots=True,
    plot_max_samples=120000,
    plot_seed=42,
    run_jacobian_checks=True,
    jacobian_mesh_max_points=6000,
    jacobian_mesh_seed=42,
    jacobian_mesh_type="structured_hex",
    enforce_positive_jacobian_ls=True,
    jacobian_ls_max_iters=4,
    jacobian_ls_mesh_max_points=4096,
    jacobian_ls_drop_fraction=1.0,
    jacobian_ls_seed=42,
    jacobian_ls_min_samples=6000,
    run_command_example=None,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading POD basis from {basis_dir}...")
    phi = np.load(os.path.join(basis_dir, "pod_basis_free.npy"))

    n_total = int(phi.shape[1])
    n_p = int(n_primary)
    if n_p < 1 or n_p >= n_total:
        raise ValueError(f"n_primary must satisfy 1 <= n_primary < {n_total}. Got {n_p}.")

    w_free, e_hist = _load_training_fluctuations(
        phi=phi,
        basis_dir=basis_dir,
        train_dir=train_dir,
        mesh_name=mesh_name,
    )

    q_full = w_free @ phi
    phi_p, phi_s, c_ls, mu_targets, n_from_strain, svals_primary, ls_active_mask, ls_info = _build_primary_secondary_bases(
        phi=phi,
        w_free=w_free,
        q_full=q_full,
        e_hist=e_hist,
        n_primary=n_p,
        enforce_positive_jacobian_ls=bool(enforce_positive_jacobian_ls),
        jacobian_ls_max_iters=int(jacobian_ls_max_iters),
        jacobian_ls_mesh_max_points=int(jacobian_ls_mesh_max_points),
        jacobian_ls_drop_fraction=float(jacobian_ls_drop_fraction),
        jacobian_ls_seed=int(jacobian_ls_seed),
        jacobian_ls_min_samples=int(jacobian_ls_min_samples),
    )

    qp = w_free @ phi_p
    qs = w_free @ phi_s

    if include_macro_strain_input:
        x_rbf = np.hstack([qp, e_hist])
    else:
        x_rbf = qp

    n_s = int(phi_s.shape[1])

    print("Dataset summary:")
    print(f"  Fluctuation snapshots W: {w_free.shape}")
    print(f"  Full reduced coordinates Q: {q_full.shape}")
    print(f"  LS targets Mu: {mu_targets.shape}")
    print(f"  Primary coordinates q_p: {qp.shape}")
    print(f"  Secondary coordinates q_s: {qs.shape}")
    print(f"  RBF input: {x_rbf.shape}")
    print(
        f"  Primary source: {n_from_strain} strain components"
        f" + {max(0, n_p - n_from_strain)} auxiliary POD components"
    )
    if bool(ls_info.get("enabled", False)):
        print(
            f"  Jacobian-aware LS: iterations={int(ls_info.get('iterations', 0))}, "
            f"active_samples={int(ls_info.get('active_samples', w_free.shape[0]))}/"
            f"{int(ls_info.get('total_samples', w_free.shape[0]))}, "
            f"best_bad_samples={int(ls_info.get('best_negative_samples', 0))}, "
            f"best_bad_cells={int(ls_info.get('best_negative_cells', 0))}"
        )

    np.save(os.path.join(out_dir, "q_p_train.npy"), qp)
    np.save(os.path.join(out_dir, "q_s_train.npy"), qs)
    np.save(os.path.join(out_dir, "ann_input_train.npy"), x_rbf)
    np.save(os.path.join(out_dir, "phi_p.npy"), phi_p)
    np.save(os.path.join(out_dir, "phi_s.npy"), phi_s)

    np.save(os.path.join(out_dir, "ls_coefficients_qfull_to_master.npy"), c_ls)
    np.save(os.path.join(out_dir, "ls_targets_train.npy"), mu_targets)
    np.save(os.path.join(out_dir, "ls_jacobian_active_mask.npy"), ls_active_mask.astype(np.uint8))

    np.savez(
        os.path.join(out_dir, "ann_dataset_metadata.npz"),
        n_primary=np.array([n_p], dtype=np.int64),
        n_secondary=np.array([n_s], dtype=np.int64),
        input_dim=np.array([x_rbf.shape[1]], dtype=np.int64),
        include_macro_strain_input=np.array([1 if include_macro_strain_input else 0], dtype=np.int64),
        selection_method=np.array(["least_squares"], dtype="U32"),
        jacobian_aware_ls=np.array([1 if bool(enforce_positive_jacobian_ls) else 0], dtype=np.int64),
    )

    proj_orth = float(np.linalg.norm(phi_p.T @ phi_s))
    summary_path = os.path.join(out_dir, "ls_dataset_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 7a LS-RBF dataset summary\n")
        f.write(f"basis_dir={basis_dir}\n")
        f.write(f"train_dir={train_dir}\n")
        f.write(f"n_primary={n_p}\n")
        f.write(f"n_secondary={n_s}\n")
        f.write(f"include_macro_strain_input={int(include_macro_strain_input)}\n")
        f.write(f"n_samples={w_free.shape[0]}\n")
        f.write(f"n_free_dofs={w_free.shape[1]}\n")
        f.write(f"ls_target_dim={mu_targets.shape[1]}\n")
        f.write(f"ls_targets_from_strain={n_from_strain}\n")
        f.write(f"orthogonality_norm_phi_p_T_phi_s={proj_orth:.16e}\n")
        f.write(f"jacobian_aware_ls_enabled={int(bool(ls_info.get('enabled', False)))}\n")
        f.write(f"jacobian_aware_ls_iterations={int(ls_info.get('iterations', 0))}\n")
        f.write(f"jacobian_aware_ls_active_samples={int(ls_info.get('active_samples', w_free.shape[0]))}\n")
        f.write(f"jacobian_aware_ls_total_samples={int(ls_info.get('total_samples', w_free.shape[0]))}\n")
        f.write(f"jacobian_aware_ls_best_negative_samples={int(ls_info.get('best_negative_samples', 0))}\n")
        f.write(f"jacobian_aware_ls_best_negative_cells={int(ls_info.get('best_negative_cells', 0))}\n")
        if "grid_shape" in ls_info:
            gs = ls_info["grid_shape"]
            f.write(f"jacobian_aware_ls_grid_shape={int(gs[0])},{int(gs[1])},{int(gs[2])}\n")
        if svals_primary.size > 0:
            f.write(f"primary_singular_value_max={float(np.max(svals_primary)):.16e}\n")
            f.write(f"primary_singular_value_min={float(np.min(svals_primary)):.16e}\n")

    if save_domain_plots and n_p >= 3 and mu_targets.shape[1] >= 3:
        _save_domain_comparison_plots(
            out_dir=out_dir,
            mu_targets=mu_targets[:, :3],
            q_pod=q_full[:, :3],
            q_ls=qp[:, :3],
            max_samples=int(plot_max_samples),
            seed=int(plot_seed),
        )
        print(
            "Domain comparison plots saved to: "
            f"{out_dir}/domain_mu_targets_3d.png, "
            f"{out_dir}/domain_qpod_first3_3d.png, "
            f"{out_dir}/domain_qls_first3_3d.png"
        )
    elif save_domain_plots:
        print(
            "[LS-RBF] Domain plots skipped because at least 3 coordinates are required "
            f"(n_primary={n_p}, mu_dim={mu_targets.shape[1]})."
        )

    if run_jacobian_checks and n_p >= 3 and mu_targets.shape[1] >= 3:
        jac_info = _run_parameter_mesh_jacobian_checks(
            mu_targets=mu_targets[:, :3],
            q_pod=q_full[:, :3],
            q_ls=qp[:, :3],
            e_hist=e_hist,
            out_dir=out_dir,
            mesh_max_points=int(jacobian_mesh_max_points),
            mesh_seed=int(jacobian_mesh_seed),
            mesh_type=str(jacobian_mesh_type).lower(),
            run_command=(
                run_command_example
                if run_command_example is not None
                else "python3 stage7a_prepare_rbf_dataset_ls.py --n-primary 3 --out-dir stage_7_ann_data_ls"
            ),
        )
        print("Jacobian diagnostics:")
        print(
            f"  negative det(J_mu): {jac_info['neg_detJ_param_count']} | "
            f"negative det(dq_ls/dmu): {jac_info['neg_detJ_map_count']} | "
            f"negative det(F): {jac_info['neg_detF_macro_count']}"
        )
        print(f"  mesh type: {jac_info['mesh_type']} ({jac_info['mesh_cell_type']})")
        print(f"  min det(dq_ls/dmu): {jac_info['min_detJ_map']:.3e}")
        print(f"  min det(F): {jac_info['min_detF_macro']:.3e}")
        print(f"  Summary txt: {jac_info['summary_txt']}")
        print(f"  Summary tex: {jac_info['summary_tex']}")
    elif run_jacobian_checks:
        print(
            "[LS-RBF] Jacobian checks skipped because at least 3 coordinates are required "
            f"(n_primary={n_p}, mu_dim={mu_targets.shape[1]})."
        )

    print(f"Dataset successfully saved to {out_dir}/")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7a LS-RBF: prepare manifold dataset with least-squares primary basis")
    parser.add_argument(
        "--n-primary",
        type=int,
        default=3,
        help="Number of primary modes to use in the LS primary basis.",
    )
    parser.add_argument(
        "--with-strain-input",
        action="store_true",
        help="Append macro strain [Exx,Eyy,Gxy] to RBF input features.",
    )
    parser.add_argument("--basis-dir", type=str, default="stage_2_pod_rve", help="POD basis directory.")
    parser.add_argument("--train-dir", type=str, default="stage_1_training_set_fom", help="Training snapshots directory.")
    parser.add_argument("--out-dir", type=str, default="stage_7_ann_data_ls", help="Output dataset directory.")
    parser.add_argument("--mesh-name", type=str, default="rve_geometry", help="Kratos mesh name.")
    parser.add_argument(
        "--skip-domain-plots",
        action="store_true",
        help="Disable LS/POD/mu domain comparison plots.",
    )
    parser.add_argument(
        "--plot-max-samples",
        type=int,
        default=120000,
        help="Maximum number of points used in domain comparison plots.",
    )
    parser.add_argument(
        "--plot-seed",
        type=int,
        default=42,
        help="Random seed for optional point subsampling in domain plots.",
    )
    parser.add_argument(
        "--skip-jacobian-checks",
        action="store_true",
        help="Disable parameter-mesh and deformation Jacobian diagnostics.",
    )
    parser.add_argument(
        "--jacobian-mesh-max-points",
        type=int,
        default=6000,
        help="Maximum number of points used to build the parameter mesh.",
    )
    parser.add_argument(
        "--jacobian-mesh-seed",
        type=int,
        default=42,
        help="Random seed for parameter-mesh subsampling.",
    )
    parser.add_argument(
        "--jacobian-mesh-type",
        type=str,
        default="structured_hex",
        choices=["delaunay", "structured_hex"],
        help="Mesh type for parameter-domain Jacobian checks.",
    )
    parser.add_argument(
        "--disable-jacobian-aware-ls",
        action="store_true",
        help="Disable iterative Jacobian-aware filtering during LS primary-mode fit.",
    )
    parser.add_argument(
        "--jacobian-ls-max-iters",
        type=int,
        default=4,
        help="Maximum Jacobian-aware LS refit iterations.",
    )
    parser.add_argument(
        "--jacobian-ls-mesh-max-points",
        type=int,
        default=4096,
        help="Structured mesh point budget used by Jacobian-aware LS filtering.",
    )
    parser.add_argument(
        "--jacobian-ls-drop-fraction",
        type=float,
        default=1.0,
        help="Fraction of bad samples dropped at each Jacobian-aware LS iteration (0..1].",
    )
    parser.add_argument(
        "--jacobian-ls-seed",
        type=int,
        default=42,
        help="Random seed for Jacobian-aware LS sample dropping.",
    )
    parser.add_argument(
        "--jacobian-ls-min-samples",
        type=int,
        default=6000,
        help="Minimum active samples kept during Jacobian-aware LS filtering.",
    )
    args = parser.parse_args()

    cmd_example = "python3 stage7a_prepare_rbf_dataset_ls.py " + " ".join(sys.argv[1:])

    prepare_rbf_dataset_least_squares(
        n_primary=args.n_primary,
        include_macro_strain_input=args.with_strain_input,
        basis_dir=args.basis_dir,
        train_dir=args.train_dir,
        out_dir=args.out_dir,
        mesh_name=args.mesh_name,
        save_domain_plots=not args.skip_domain_plots,
        plot_max_samples=args.plot_max_samples,
        plot_seed=args.plot_seed,
        run_jacobian_checks=not args.skip_jacobian_checks,
        jacobian_mesh_max_points=args.jacobian_mesh_max_points,
        jacobian_mesh_seed=args.jacobian_mesh_seed,
        jacobian_mesh_type=args.jacobian_mesh_type,
        enforce_positive_jacobian_ls=not args.disable_jacobian_aware_ls,
        jacobian_ls_max_iters=args.jacobian_ls_max_iters,
        jacobian_ls_mesh_max_points=args.jacobian_ls_mesh_max_points,
        jacobian_ls_drop_fraction=args.jacobian_ls_drop_fraction,
        jacobian_ls_seed=args.jacobian_ls_seed,
        jacobian_ls_min_samples=args.jacobian_ls_min_samples,
        run_command_example=cmd_example,
    )
