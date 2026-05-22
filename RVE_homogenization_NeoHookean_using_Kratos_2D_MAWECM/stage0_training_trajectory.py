#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 0 (2D-MAWECM): structured parametric mesh + 2 covering trajectories.

This script builds the parameter mesh first in (Gx_macro, Gxy_macro), then
constructs two equidistant serpentine trajectories that jointly cover all mesh
rows. It exports a Stage-1 compatible bundle with keys trajectory_1, trajectory_2
containing strain waypoints [Exx, Eyy, Gxy].
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def _append_unique(points, p, tol=1e-14):
    q = np.asarray(p, dtype=float)
    if len(points) == 0 or np.linalg.norm(q - points[-1]) > tol:
        points.append(q)


def build_structured_mesh(gx_min, gx_max, n_gx, gxy_min, gxy_max, n_gxy):
    gx_vals = np.linspace(float(gx_min), float(gx_max), int(n_gx))
    gxy_vals = np.linspace(float(gxy_min), float(gxy_max), int(n_gxy))
    # Ensure origin alignment with the structured grid when the range crosses zero.
    if float(gx_min) < 0.0 < float(gx_max):
        gx_vals[int(np.argmin(np.abs(gx_vals)))] = 0.0
    if float(gxy_min) < 0.0 < float(gxy_max):
        gxy_vals[int(np.argmin(np.abs(gxy_vals)))] = 0.0

    xx, yy = np.meshgrid(gx_vals, gxy_vals, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    quads = []
    for j in range(n_gxy - 1):
        for i in range(n_gx - 1):
            n0 = j * n_gx + i
            n1 = n0 + 1
            n2 = n0 + n_gx + 1
            n3 = n0 + n_gx
            quads.append([n0, n1, n2, n3])
    quads = np.asarray(quads, dtype=int)
    return gx_vals, gxy_vals, nodes, quads


def build_grid_graph_edges(n_gx, n_gxy):
    edges = []
    for j in range(n_gxy):
        for i in range(n_gx):
            nid = j * n_gx + i
            if i + 1 < n_gx:
                edges.append([nid, nid + 1])
            if j + 1 < n_gxy:
                edges.append([nid, nid + n_gx])
    return np.asarray(edges, dtype=int)


def build_laplacian(n_nodes, edges):
    L = np.zeros((n_nodes, n_nodes), dtype=float)
    for a, b in edges:
        ia = int(a)
        ib = int(b)
        L[ia, ia] += 1.0
        L[ib, ib] += 1.0
        L[ia, ib] -= 1.0
        L[ib, ia] -= 1.0
    return L


def ordered_rows_from_center(gxy_vals):
    n = len(gxy_vals)
    center = int(np.argmin(np.abs(gxy_vals)))
    rows = [center]
    for d in range(1, n):
        up = center + d
        dn = center - d
        if up < n:
            rows.append(up)
        if dn >= 0:
            rows.append(dn)
    seen = set()
    out = []
    for r in rows:
        if r not in seen:
            out.append(r)
            seen.add(r)
    return out


def build_serpentine_rows(
    gx_vals,
    gxy_vals,
    row_ids,
    start_left_to_right=True,
    include_origin=True,
    entry_side=None,
    include_center_row_nodes=True,
):
    pts = []
    if include_origin:
        _append_unique(pts, [0.0, 0.0])

    left_to_right = bool(start_left_to_right)
    if len(row_ids) > 0 and include_origin:
        # First, walk on the center row (Gxy=0) through mesh nodes so both
        # trajectories jointly cover all structured nodes on that row.
        if include_center_row_nodes:
            i0 = int(np.argmin(np.abs(gx_vals)))
            if entry_side == "left":
                x_walk = gx_vals[:i0][::-1]  # from near 0 to xmin
            elif entry_side == "right":
                x_walk = gx_vals[i0 + 1 :]   # from near 0 to xmax
            else:
                x_walk = gx_vals
            for x in x_walk:
                _append_unique(pts, [float(x), 0.0])

        first_y = float(gxy_vals[row_ids[0]])
        if entry_side == "left":
            first_x = float(gx_vals[0])
        elif entry_side == "right":
            first_x = float(gx_vals[-1])
        else:
            first_x = float(gx_vals[0] if left_to_right else gx_vals[-1])
        # Enter the first row with an orthogonal move (horizontal then vertical),
        # avoiding origin-to-row diagonals.
        _append_unique(pts, [first_x, 0.0])
        _append_unique(pts, [first_x, first_y])

    for rid in row_ids:
        y = float(gxy_vals[rid])
        x_seq = gx_vals if left_to_right else gx_vals[::-1]
        for x in x_seq:
            _append_unique(pts, [float(x), y])
        left_to_right = not left_to_right
    return np.asarray(pts, dtype=float)


def map_param_to_strain(path_param, mapping):
    gx = np.asarray(path_param[:, 0], dtype=float)
    gxy = np.asarray(path_param[:, 1], dtype=float)

    if mapping == "small_strain":
        exx = gx
        eyy = np.zeros_like(gx)
        gamma = gxy
    elif mapping == "green_lagrange_upper":
        # F = I + G, with G = [[Gx, Gxy], [0, 0]]
        exx = gx + 0.5 * gx * gx
        eyy = 0.5 * gxy * gxy
        gamma = gxy * (1.0 + gx)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")

    return np.column_stack([exx, eyy, gamma])


def plot_structured_mesh(gx_vals, gxy_vals, traj1, traj2, out_file):
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for y in gxy_vals:
        ax.plot(gx_vals, np.full_like(gx_vals, y), color="#7f8c8d", lw=0.9, alpha=0.55)
    for x in gx_vals:
        ax.plot(np.full_like(gxy_vals, x), gxy_vals, color="#7f8c8d", lw=0.9, alpha=0.55)

    ax.plot(traj1[:, 0], traj1[:, 1], color="#1f77b4", lw=1.8, label="Trajectory 1")
    ax.plot(traj2[:, 0], traj2[:, 1], color="#d62728", lw=1.8, label="Trajectory 2")
    ax.scatter([traj1[0, 0]], [traj1[0, 1]], color="#1f77b4", s=25)
    ax.scatter([traj2[0, 0]], [traj2[0, 1]], color="#d62728", s=25)

    ax.set_xlabel(r"$G_x^{macro}$")
    ax.set_ylabel(r"$G_{xy}^{macro}$")
    ax.set_title("Structured parameter mesh with 2 covering trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def plot_mesh_topology(gx_vals, gxy_vals, nodes, edges, out_file):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for a, b in edges:
        pa = nodes[int(a)]
        pb = nodes[int(b)]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#7f8c8d", lw=0.8, alpha=0.55)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=10, color="#2c3e50", alpha=0.85, label="Nodes")
    ax.set_xlabel(r"$G_x^{macro}$")
    ax.set_ylabel(r"$G_{xy}^{macro}$")
    ax.set_title("Structured mesh topology (nodes + edges)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def plot_graph_and_laplacian(nodes, edges, laplacian, out_file):
    n = laplacian.shape[0]
    ii, jj = np.where(np.abs(laplacian) > 0.0)

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax0 = axs[0]
    for a, b in edges:
        pa = nodes[int(a)]
        pb = nodes[int(b)]
        ax0.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#3498db", lw=0.8, alpha=0.65)
    ax0.scatter(nodes[:, 0], nodes[:, 1], s=8, color="#1f2d3d", alpha=0.8)
    ax0.set_xlabel(r"$G_x^{macro}$")
    ax0.set_ylabel(r"$G_{xy}^{macro}$")
    ax0.set_title("Graph used for Laplacian")
    ax0.grid(True, alpha=0.25)

    ax1 = axs[1]
    ax1.scatter(jj, ii, s=2, c="#2ecc71", alpha=0.75)
    ax1.set_xlim(-1, n)
    ax1.set_ylim(n, -1)
    ax1.set_xlabel("column index")
    ax1.set_ylabel("row index")
    ax1.set_title("Laplacian sparsity pattern")
    ax1.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build structured Stage-0 trajectories for 2D MAW-ECM workflow."
    )
    # Default range is intentionally asymmetric (safer in compression),
    # aligned with the previous stable envelope philosophy.
    p.add_argument("--gx-min", type=float, default=-0.10)
    p.add_argument("--gx-max", type=float, default=0.42)
    p.add_argument("--gxy-min", type=float, default=-0.05)
    p.add_argument("--gxy-max", type=float, default=0.05)
    p.add_argument("--n-gx", type=int, default=41)
    p.add_argument("--n-gxy", type=int, default=11)
    p.add_argument(
        "--mapping",
        type=str,
        default="green_lagrange_upper",
        choices=["green_lagrange_upper", "small_strain"],
        help="Map from parametric G-space to strain waypoints [Exx,Eyy,Gxy].",
    )
    p.add_argument("--include-origin", type=int, default=1, choices=[0, 1])
    p.add_argument("--ref-steps", type=int, default=400)
    p.add_argument("--reference-amplitude", type=float, default=0.5)
    p.add_argument("--out-dir", type=str, default="stage_0_trajectory")
    p.add_argument("--trajectory-file", type=str, default="stage_0_trajectories.npz")
    p.add_argument("--save-plots", type=int, default=1, choices=[0, 1])
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_gx < 2 or args.n_gxy < 2:
        raise ValueError("n-gx and n-gxy must be >= 2.")
    if args.gx_min >= args.gx_max:
        raise ValueError("gx-min must be < gx-max.")
    if args.gxy_min >= args.gxy_max:
        raise ValueError("gxy-min must be < gxy-max.")

    gx_vals, gxy_vals, nodes, quads = build_structured_mesh(
        args.gx_min, args.gx_max, args.n_gx, args.gxy_min, args.gxy_max, args.n_gxy
    )
    edges = build_grid_graph_edges(args.n_gx, args.n_gxy)
    laplacian = build_laplacian(nodes.shape[0], edges)

    center_row = int(np.argmin(np.abs(gxy_vals)))
    rows_t1 = list(range(center_row - 1, -1, -1))          # below zero: -dy, -2dy, ...
    rows_t2 = list(range(center_row + 1, len(gxy_vals)))    # above zero: +dy, +2dy, ...

    traj1_param = build_serpentine_rows(
        gx_vals=gx_vals,
        gxy_vals=gxy_vals,
        row_ids=rows_t1,
        start_left_to_right=True,
        include_origin=bool(args.include_origin),
        entry_side="left",
    )
    traj2_param = build_serpentine_rows(
        gx_vals=gx_vals,
        gxy_vals=gxy_vals,
        row_ids=rows_t2,
        start_left_to_right=False,
        include_origin=bool(args.include_origin),
        entry_side="right",
    )

    traj1 = map_param_to_strain(traj1_param, args.mapping)
    traj2 = map_param_to_strain(traj2_param, args.mapping)

    os.makedirs(args.out_dir, exist_ok=True)
    bundle_path = os.path.join(args.out_dir, args.trajectory_file)

    payload = {
        "trajectory_count": np.array([2], dtype=int),
        "ref_steps": np.array([int(args.ref_steps)], dtype=int),
        "reference_amplitude": np.array([float(args.reference_amplitude)], dtype=float),
        "param_names": np.array(["Gx_macro", "Gxy_macro"], dtype="<U16"),
        "mapping": np.array([args.mapping], dtype="<U32"),
        "structured_mesh_shape": np.array([int(args.n_gxy), int(args.n_gx)], dtype=int),
        "gx_values": np.asarray(gx_vals, dtype=float),
        "gxy_values": np.asarray(gxy_vals, dtype=float),
        "grid_nodes_param": np.asarray(nodes, dtype=float),
        "grid_cells_quad": np.asarray(quads, dtype=int),
        "grid_graph_edges": np.asarray(edges, dtype=int),
        "grid_graph_laplacian": np.asarray(laplacian, dtype=float),
        "trajectory_labels": np.array([[1.0, +1.0], [2.0, -1.0]], dtype=float),
        "trajectory_param_1": np.asarray(traj1_param, dtype=float),
        "trajectory_param_2": np.asarray(traj2_param, dtype=float),
        "trajectory_1": np.asarray(traj1, dtype=float),
        "trajectory_2": np.asarray(traj2, dtype=float),
    }
    np.savez(bundle_path, **payload)

    if int(args.save_plots) == 1:
        fig_path = os.path.join(args.out_dir, "structured_param_mesh_and_trajectories.png")
        fig_mesh = os.path.join(args.out_dir, "structured_param_mesh_topology.png")
        fig_graph_lap = os.path.join(args.out_dir, "structured_param_graph_and_laplacian.png")
        plot_structured_mesh(gx_vals, gxy_vals, traj1_param, traj2_param, fig_path)
        plot_mesh_topology(gx_vals, gxy_vals, nodes, edges, fig_mesh)
        plot_graph_and_laplacian(nodes, edges, laplacian, fig_graph_lap)
        print(f"[INFO] Saved figure: {fig_path}")
        print(f"[INFO] Saved figure: {fig_mesh}")
        print(f"[INFO] Saved figure: {fig_graph_lap}")

    print("[INFO] Stage0 2D-MAWECM bundle created")
    print(f"[INFO] out_file={bundle_path}")
    print(f"[INFO] grid: n_gx={args.n_gx}, n_gxy={args.n_gxy}, nodes={nodes.shape[0]}, quads={quads.shape[0]}")
    print(f"[INFO] graph: edges={edges.shape[0]}")
    print(f"[INFO] trajectory_1 points={traj1.shape[0]}, trajectory_2 points={traj2.shape[0]}")
    print(f"[INFO] mapping={args.mapping}, include_origin={args.include_origin}")


if __name__ == "__main__":
    main()
