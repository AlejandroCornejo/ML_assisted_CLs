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


def _fit_ls_master_map(q_full, mu, n_primary):
    """Fit the Joaquin/2D-MAWECM POD-to-master map."""
    n_p = int(n_primary)
    if n_p != int(mu.shape[1]):
        raise RuntimeError(
            "The LS master dimension must equal the physical parameter dimension. "
            f"Got n_primary={n_p}, mu_dim={mu.shape[1]}."
        )

    # Solve min ||Q T_m^T - Mu||_F.
    t_m_t, *_ = np.linalg.lstsq(q_full, mu, rcond=None)
    mu_hat = q_full @ t_m_t
    err_rel = float(np.linalg.norm(mu_hat - mu) / max(np.linalg.norm(mu), 1e-30))
    rmse = np.sqrt(np.mean((mu_hat - mu) ** 2, axis=0))
    return {
        "T_m": np.asarray(t_m_t.T, dtype=float),
        "T_m_t": np.asarray(t_m_t, dtype=float),
        "mu_hat": np.asarray(mu_hat, dtype=float),
        "error_rel": err_rel,
        "rmse": np.asarray(rmse, dtype=float),
    }


def _build_consistent_master_slave_split(phi, q_full, t_m_t):
    """Build Phi_m, A_m, Phi_s and coordinates exactly as in the 2D workflow."""
    phi_c = np.asarray(phi @ t_m_t, dtype=float)
    u_m, s_m, v_m_t = np.linalg.svd(phi_c, full_matrices=False)
    tol = (s_m[0] if s_m.size else 1.0) * 1e-12
    rank = int(np.sum(s_m > tol))
    expected_rank = int(t_m_t.shape[1])
    if rank != expected_rank:
        raise RuntimeError(
            f"LS master basis rank mismatch: rank={rank}, expected={expected_rank}."
        )

    phi_m = np.asarray(u_m[:, :rank], dtype=float)
    sigma_m = np.asarray(s_m[:rank], dtype=float)
    v_m_t = np.asarray(v_m_t[:rank, :], dtype=float)
    a_m = np.asarray(np.diag(sigma_m) @ v_m_t, dtype=float)

    c_m = np.asarray(phi.T @ phi_m, dtype=float)
    gram_m = c_m.T @ c_m
    if np.linalg.norm(gram_m - np.eye(rank)) > 1e-10:
        raise RuntimeError("C_m is not orthonormal enough for a stable master/slave split.")

    # Orthogonal complement of C_m inside the retained POD space.
    projector_s = np.eye(phi.shape[1], dtype=float) - c_m @ c_m.T
    evals, evecs = np.linalg.eigh(projector_s)
    order = np.argsort(evals)[::-1]
    c_s = np.asarray(evecs[:, order[: phi.shape[1] - rank]], dtype=float)
    if c_s.size:
        c_s = np.asarray(np.linalg.qr(c_s)[0], dtype=float)

    phi_s = np.asarray(phi @ c_s, dtype=float)
    q_master = np.asarray(q_full @ c_m, dtype=float)
    try:
        q_m = np.linalg.solve(a_m, q_master.T).T
    except np.linalg.LinAlgError:
        q_m = np.linalg.lstsq(a_m, q_master.T, rcond=None)[0].T
    q_s = np.asarray(q_full @ c_s, dtype=float)

    w_ref = np.asarray(phi @ q_full.T, dtype=float).T
    w_rec = (phi_m @ (a_m @ q_m.T)).T + (phi_s @ q_s.T).T
    decoder_rel = float(np.linalg.norm(w_rec - w_ref) / max(np.linalg.norm(w_ref), 1e-30))

    return {
        "phi_c": phi_c,
        "phi_m": phi_m,
        "phi_s": phi_s,
        "A_m": a_m,
        "C_m": c_m,
        "C_s": c_s,
        "sigma_m": sigma_m,
        "v_m_t": v_m_t,
        "q_m": q_m,
        "q_s": q_s,
        "q_master": q_master,
        "decoder_rel_error": decoder_rel,
        "ortho_ms": float(np.linalg.norm(phi_m.T @ phi_s)),
    }


def _fit_affine_mu_to_qm(mu, q_m):
    x_aug = np.hstack([np.asarray(mu, dtype=float), np.ones((mu.shape[0], 1))])
    b_aff, *_ = np.linalg.lstsq(x_aug, q_m, rcond=None)
    q_hat = x_aug @ b_aff
    rel = float(np.linalg.norm(q_hat - q_m) / max(np.linalg.norm(q_m), 1e-30))
    return np.asarray(b_aff, dtype=float), rel


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


def _plot_domain_3d(data, labels, title, out_file):
    import matplotlib.pyplot as plt
    from plot_style_utils import apply_latex_plot_style
    apply_latex_plot_style()

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
    q_m,
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
        q_m = q_m[idx, :]

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
        q_m,
        labels=(r"$q^m_1$", r"$q^m_2$", r"$q^m_3$"),
        title="Domain in consistent LS-master coordinates",
        out_file=os.path.join(out_dir, "domain_qm_first3_3d.png"),
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


def _plot_jacobian_histograms(j_param_det, j_map_det, j_phys, out_dir):
    import matplotlib.pyplot as plt
    from plot_style_utils import apply_latex_plot_style
    apply_latex_plot_style()

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
        title=r"$\det(\partial q_m/\partial \mu)$",
        xlabel=r"$\det(\partial q_m/\partial \mu)$",
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
    from plot_style_utils import apply_latex_plot_style
    apply_latex_plot_style()
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
        f.write(
            r"A 3D Delaunay tetrahedralization is built in the normalized parameter space. "
            r"For each tetrahedron $e$, using four corresponding actual states in $(\mu,q_m)$, we compute "
            r"a local affine map $q_m\approx \mathbf{A}_e\mu+\mathbf{b}_e$." + "\n"
        )
        f.write(r"\begin{equation}" + "\n")
        f.write(r"\mathbf{A}_e = \mathbf{Q}_e\mathbf{M}_e^{-1},\qquad ")
        f.write(r"\mathbf{M}_e=[\mu_a-\mu_0\ \mu_b-\mu_0\ \mu_c-\mu_0],\quad ")
        f.write(r"\mathbf{Q}_e=[q_a-q_0\ q_b-q_0\ q_c-q_0]." + "\n")
        f.write(r"\end{equation}" + "\n")
        f.write(
            r"We monitor: (i) $\det(\mathbf{M}_e)$ for parameter-mesh degeneracy, and "
            r"(ii) $\det(\mathbf{A}_e)=\det(\partial q_m/\partial\mu)$ for local folding." + "\n"
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
        f.write(r"\includegraphics[width=0.48\textwidth]{domain_qm_first3_3d.png}" + "\n")
        f.write(r"\caption{Point clouds in physical parameter space and consistent LS-master space.}" + "\n")
        f.write(r"\end{figure}" + "\n")

        f.write(r"\begin{figure}[h!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{domain_qpod_first3_3d.png}" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{domain_qm_first3_3d.png}" + "\n")
        f.write(r"\caption{Point clouds in POD-first-3 and consistent LS-master coordinates.}" + "\n")
        f.write(r"\end{figure}" + "\n")

        f.write(r"\begin{figure}[h!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{parameter_mesh_mu_edges_3d.png}" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{parameter_mesh_q_pod_edges_3d.png}" + "\n")
        f.write(r"\caption{Parameter-mesh edges and mapped POD-first-3 mesh edges.}" + "\n")
        f.write(r"\end{figure}" + "\n")

        f.write(r"\begin{figure}[h!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\includegraphics[width=0.48\textwidth]{parameter_mesh_q_m_edges_3d.png}" + "\n")
        f.write(r"\caption{Delaunay parameter-mesh edges mapped to consistent LS-master space.}" + "\n")
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
    q_m,
    e_hist,
    out_dir,
    mesh_max_points=6000,
    mesh_seed=42,
    mesh_type="delaunay",
    run_command="python3 stage7a_prepare_rbf_dataset_ls.py --n-primary 3",
):
    if mu_targets.shape[1] < 3 or q_pod.shape[1] < 3 or q_m.shape[1] < 3:
        raise ValueError(
            "Jacobian checks require at least 3 target coordinates, 3 POD coordinates, and 3 LS coordinates."
        )

    mu3 = np.asarray(mu_targets[:, :3], dtype=float)
    qpod3 = np.asarray(q_pod[:, :3], dtype=float)
    q3 = np.asarray(q_m[:, :3], dtype=float)

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
        idx_sub = idx_sub[idx_unique]

        if xi_m.shape[0] < 4:
            raise RuntimeError("Not enough distinct mesh nodes for 3D Delaunay Jacobian checks.")

        try:
            tri = Delaunay(xi_m, qhull_options="QJ Qbb Qc")
        except QhullError as exc:
            raise RuntimeError(f"Delaunay mesh construction failed: {exc}") from exc

        cells = np.asarray(tri.simplices, dtype=np.int64)
        cell_type = "tet"
    else:
        raise ValueError(
            f"Unsupported mesh_type='{mesh_type}'. Only 'delaunay' is allowed: "
            "the former structured IDW box extrapolated synthetic states."
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
    nonzero_map = det_j_map_valid[np.abs(det_j_map_valid) > 1e-14]
    orientation = float(np.sign(np.median(nonzero_map))) if nonzero_map.size else 0.0
    folded_map = int(
        np.sum(det_j_map_valid * orientation <= 1e-14)
        if orientation != 0.0
        else det_j_map_valid.size
    )
    neg_map = int(np.sum(det_j_map_valid <= 0.0))
    neg_param = int(np.sum(det_j_param_valid <= 0.0))

    np.save(os.path.join(out_dir, "parameter_mesh_nodes_mu.npy"), mu_m)
    np.save(os.path.join(out_dir, "parameter_mesh_nodes_q_pod.npy"), qpod_m)
    np.save(os.path.join(out_dir, "parameter_mesh_nodes_q_m.npy"), q_m)
    np.save(os.path.join(out_dir, "parameter_mesh_sample_indices.npy"), idx_sub)
    np.save(os.path.join(out_dir, "parameter_mesh_cells.npy"), cells)
    if cell_type == "tet":
        np.save(os.path.join(out_dir, "parameter_mesh_tetrahedra.npy"), cells)
        for stale_name in (
            "parameter_mesh_hexahedra.npy",
            "parameter_mesh_structured_grid_shape.npy",
            "parameter_mesh_nodes_q_ls.npy",
            "jacobian_det_mu_to_q_ls.npy",
            "parameter_mesh_q_ls_edges_3d.png",
            "domain_qls_first3_3d.png",
        ):
            stale_path = os.path.join(out_dir, stale_name)
            if os.path.exists(stale_path):
                os.remove(stale_path)
    else:
        np.save(os.path.join(out_dir, "parameter_mesh_hexahedra.npy"), cells)
        np.save(os.path.join(out_dir, "parameter_mesh_structured_grid_shape.npy"), np.array(grid_shape, dtype=np.int64))
    np.save(os.path.join(out_dir, "jacobian_det_parameter_mesh.npy"), det_j_param)
    np.save(os.path.join(out_dir, "jacobian_det_mu_to_q_m.npy"), det_j_map)
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
    fig_qls = os.path.join(out_dir, "parameter_mesh_q_m_edges_3d.png")

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
            labels=(r"$q^m_1$", r"$q^m_2$", r"$q^m_3$"),
            title="Mapped tetra mesh in consistent LS-master space",
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
            labels=(r"$q^m_1$", r"$q^m_2$", r"$q^m_3$"),
            title="Mapped structured hexa mesh in consistent LS-master space",
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
        f.write("mesh_nodes_source=actual_training_samples\n")
        f.write("synthetic_idw_interpolation=0\n")
        if grid_shape is not None:
            f.write(f"structured_grid_shape={grid_shape[0]},{grid_shape[1]},{grid_shape[2]}\n")
        f.write(f"negative_detJ_param_count={neg_param}\n")
        f.write(f"negative_detJ_map_count={neg_map}\n")
        f.write(f"dominant_orientation_detJ_map={orientation:.1f}\n")
        f.write(f"folded_or_singular_detJ_map_count={folded_map}\n")
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
        "folded_detJ_map_count": folded_map,
        "neg_detF_macro_count": neg_phys,
        "min_detJ_map": float(np.nanmin(det_j_map_valid)),
        "min_detF_macro": float(np.min(j_phys)),
        "mesh_type": mesh_kind,
        "mesh_cell_type": cell_type,
    }


def prepare_rbf_dataset_least_squares(
    n_primary=3,
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
    jacobian_mesh_type="delaunay",
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
    mu_targets, n_from_strain = _build_ls_targets(q_full, e_hist, n_p)
    ls = _fit_ls_master_map(q_full, mu_targets, n_p)
    split = _build_consistent_master_slave_split(phi, q_full, ls["T_m_t"])
    phi_p = split["phi_m"]
    phi_s = split["phi_s"]
    qp = split["q_m"]
    qs = split["q_s"]
    b_aff, affine_rel = _fit_affine_mu_to_qm(mu_targets, qp)
    x_rbf = qp
    ls_active_mask = np.ones(q_full.shape[0], dtype=np.uint8)

    n_s = int(phi_s.shape[1])

    print("Dataset summary:")
    print(f"  Fluctuation snapshots W: {w_free.shape}")
    print(f"  Full reduced coordinates Q: {q_full.shape}")
    print(f"  LS targets Mu: {mu_targets.shape}")
    print(f"  Master coordinates q_m: {qp.shape}")
    print(f"  Secondary coordinates q_s: {qs.shape}")
    print(f"  RBF input: {x_rbf.shape}")
    print(
        f"  Primary source: {n_from_strain} strain components"
        f" + {max(0, n_p - n_from_strain)} auxiliary POD components"
    )
    print(f"  POD -> mu LS relative error: {ls['error_rel']:.3e}")
    print(f"  Exact master/slave decoder error: {split['decoder_rel_error']:.3e}")
    print(f"  mu -> q_m affine initializer error: {affine_rel:.3e}")

    np.save(os.path.join(out_dir, "q_m_train.npy"), qp)
    np.save(os.path.join(out_dir, "q_p_train.npy"), qp)
    np.save(os.path.join(out_dir, "q_s_train.npy"), qs)
    np.save(os.path.join(out_dir, "ann_input_train.npy"), x_rbf)
    np.save(os.path.join(out_dir, "phi_m.npy"), phi_p)
    np.save(os.path.join(out_dir, "phi_p.npy"), phi_p)
    np.save(os.path.join(out_dir, "phi_s.npy"), phi_s)

    np.save(os.path.join(out_dir, "T_m.npy"), ls["T_m"])
    np.save(os.path.join(out_dir, "ls_coefficients_qfull_to_master.npy"), ls["T_m_t"])
    np.save(os.path.join(out_dir, "phi_c.npy"), split["phi_c"])
    np.save(os.path.join(out_dir, "A_m.npy"), split["A_m"])
    np.save(os.path.join(out_dir, "C_m.npy"), split["C_m"])
    np.save(os.path.join(out_dir, "C_s.npy"), split["C_s"])
    np.save(os.path.join(out_dir, "q_master_train.npy"), split["q_master"])
    np.save(os.path.join(out_dir, "q_m_init_from_mu_A.npy"), b_aff[:-1, :])
    np.save(os.path.join(out_dir, "q_m_init_from_mu_b.npy"), b_aff[-1, :])
    np.savez(
        os.path.join(out_dir, "qm_init_mu_affine.npz"),
        b_aff=b_aff,
        mu_dim=np.array([mu_targets.shape[1]], dtype=np.int64),
        qp_dim=np.array([n_p], dtype=np.int64),
        rel_fit=np.array([affine_rel], dtype=float),
        n_samples=np.array([mu_targets.shape[0]], dtype=np.int64),
    )
    np.save(os.path.join(out_dir, "ls_targets_train.npy"), mu_targets)
    np.save(os.path.join(out_dir, "ls_jacobian_active_mask.npy"), ls_active_mask.astype(np.uint8))

    np.savez(
        os.path.join(out_dir, "ann_dataset_metadata.npz"),
        n_primary=np.array([n_p], dtype=np.int64),
        n_secondary=np.array([n_s], dtype=np.int64),
        input_dim=np.array([x_rbf.shape[1]], dtype=np.int64),
        include_macro_strain_input=np.array([0], dtype=np.int64),
        selection_method=np.array(["joaquin_ls_master"], dtype="U32"),
        coordinate_system=np.array(["q_m_with_A_m"], dtype="U32"),
        jacobian_aware_ls=np.array([0], dtype=np.int64),
    )

    proj_orth = float(np.linalg.norm(phi_p.T @ phi_s))
    summary_path = os.path.join(out_dir, "ls_dataset_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 7a LS-RBF dataset summary\n")
        f.write(f"basis_dir={basis_dir}\n")
        f.write(f"train_dir={train_dir}\n")
        f.write(f"n_primary={n_p}\n")
        f.write(f"n_secondary={n_s}\n")
        f.write("include_macro_strain_input=0\n")
        f.write(f"n_samples={w_free.shape[0]}\n")
        f.write(f"n_free_dofs={w_free.shape[1]}\n")
        f.write(f"ls_target_dim={mu_targets.shape[1]}\n")
        f.write(f"ls_targets_from_strain={n_from_strain}\n")
        f.write(f"orthogonality_norm_phi_p_T_phi_s={proj_orth:.16e}\n")
        f.write("coordinate_system=q_m_with_A_m\n")
        f.write(f"pod_to_mu_ls_rel_error={ls['error_rel']:.16e}\n")
        f.write("pod_to_mu_ls_rmse=" + ",".join(f"{v:.16e}" for v in ls["rmse"]) + "\n")
        f.write(f"decoder_rel_error={split['decoder_rel_error']:.16e}\n")
        f.write(f"mu_to_qm_affine_rel_error={affine_rel:.16e}\n")
        f.write(f"orthogonality_norm_phi_m_T_phi_s={split['ortho_ms']:.16e}\n")

    if save_domain_plots and n_p >= 3 and mu_targets.shape[1] >= 3:
        _save_domain_comparison_plots(
            out_dir=out_dir,
            mu_targets=mu_targets[:, :3],
            q_pod=q_full[:, :3],
            q_m=qp[:, :3],
            max_samples=int(plot_max_samples),
            seed=int(plot_seed),
        )
        print(
            "Domain comparison plots saved to: "
            f"{out_dir}/domain_mu_targets_3d.png, "
            f"{out_dir}/domain_qpod_first3_3d.png, "
            f"{out_dir}/domain_qm_first3_3d.png"
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
            q_m=qp[:, :3],
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
            f"folded/singular det(dq_m/dmu): {jac_info['folded_detJ_map_count']} | "
            f"negative det(F): {jac_info['neg_detF_macro_count']}"
        )
        print(f"  mesh type: {jac_info['mesh_type']} ({jac_info['mesh_cell_type']})")
        print(f"  min det(dq_m/dmu): {jac_info['min_detJ_map']:.3e}")
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
        default="delaunay",
        choices=["delaunay"],
        help="Delaunay tetrahedralization of actual training states; synthetic IDW grids are disabled.",
    )
    args = parser.parse_args()

    cmd_example = "python3 stage7a_prepare_rbf_dataset_ls.py " + " ".join(sys.argv[1:])

    prepare_rbf_dataset_least_squares(
        n_primary=args.n_primary,
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
        run_command_example=cmd_example,
    )
