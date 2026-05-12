#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 0: trajectory generation + visualization.

Outputs only:
- training_trajectory_coverage.png
- trajectory_01_02_final_overview_3d.png
- trajectory_01_02_comparison_3d.mp4
- stage_0_trajectories.npz

Logic:
- 2 trajectories per Gxy level (positive scan + negative scan)
- Each trajectory starts from (0,0,0)
- Spacing controlled by a shared target_dy, so upper/lower scans have similar spacing
  even if ellipse center is shifted.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()
from matplotlib import animation


def parse_relative_boundary(text):
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError("relative-boundary must have 6 values: Exx+,Exx-,Eyy+,Eyy-,Gxy+,Gxy-")
    arr = np.array(vals, dtype=float)
    if np.any(arr <= 0.0):
        raise ValueError("All relative-boundary values must be > 0.")
    return arr


def build_boundary(emax, rel6, domain_type="box"):
    exx_pos = float(emax) * float(rel6[0])
    exx_neg = float(emax) * float(rel6[1])
    eyy_pos = float(emax) * float(rel6[2])
    eyy_neg = float(emax) * float(rel6[3])
    gxy_pos = float(emax) * float(rel6[4])
    gxy_neg = float(emax) * float(rel6[5])

    center = np.array(
        [
            0.5 * (exx_pos - exx_neg),
            0.5 * (eyy_pos - eyy_neg),
            0.5 * (gxy_pos - gxy_neg),
        ],
        dtype=float,
    )
    axes = np.array(
        [
            0.5 * (exx_pos + exx_neg),
            0.5 * (eyy_pos + eyy_neg),
            0.5 * (gxy_pos + gxy_neg),
        ],
        dtype=float,
    )
    return {
        "domain_type": domain_type,
        "center": center,
        "axes": axes,
        "limits": {
            "Exx": [-exx_neg, exx_pos],
            "Eyy": [-eyy_neg, eyy_pos],
            "Gxy": [-gxy_neg, gxy_pos],
        },
    }


def ellipse_section_at_gxy(boundary, gxy):
    c = boundary["center"]
    a = boundary["axes"]
    kz = (float(gxy) - c[2]) / a[2]
    gamma = 1.0 - kz * kz
    if gamma < -1e-12:
        return None
    gamma = max(gamma, 0.0)

    cx = c[0]
    cy = c[1]
    ax_eff = a[0] * np.sqrt(gamma)
    ay_eff = a[1] * np.sqrt(gamma)
    return cx, cy, ax_eff, ay_eff, cy - ay_eff, cy + ay_eff


def ellipse_x_bounds(y, cx, cy, ax_eff, ay_eff):
    if abs(float(ax_eff)) <= 1e-14 or abs(float(ay_eff)) <= 1e-14:
        return cx, cx
    yy = (float(y) - cy) / ay_eff
    q = 1.0 - yy * yy
    if q <= 0.0:
        return cx, cx
    dx = ax_eff * np.sqrt(q)
    return cx - dx, cx + dx


def box_section_at_gxy(boundary, gxy):
    """Cross-section of the box domain at a given Gxy. Returns same tuple as ellipse version."""
    lim = boundary["limits"]
    if float(gxy) < lim["Gxy"][0] - 1e-12 or float(gxy) > lim["Gxy"][1] + 1e-12:
        return None
    c = boundary["center"]
    a = boundary["axes"]
    return c[0], c[1], a[0], a[1], c[1] - a[1], c[1] + a[1]


def section_at_gxy(boundary, gxy):
    """Dispatch to ellipsoid or box cross-section."""
    if boundary.get("domain_type") == "box":
        return box_section_at_gxy(boundary, gxy)
    return ellipse_section_at_gxy(boundary, gxy)


def x_bounds_at_y(boundary, y, cx, cy, ax_eff, ay_eff):
    """X-range at a given y: constant for box, elliptical for ellipsoid."""
    if boundary.get("domain_type") == "box":
        return cx - ax_eff, cx + ax_eff
    return ellipse_x_bounds(y, cx, cy, ax_eff, ay_eff)


def safe_gxy_for_origin(boundary):
    """Max absolute Gxy reachable from (0,0) without leaving the domain."""
    if boundary.get("domain_type") == "box":
        return boundary["axes"][2]  # full Gxy range for box
    c, a = boundary["center"], boundary["axes"]
    dx0, dy0 = (0.0 - c[0]) / a[0], (0.0 - c[1]) / a[1]
    q_safe = 1.0 - dx0 * dx0 - dy0 * dy0
    return a[2] * np.sqrt(max(0.0, q_safe))


def append_if_different(points, point, tol=1e-14):
    p = np.array(point, dtype=float)
    if len(points) == 0 or np.linalg.norm(p - points[-1]) > tol:
        points.append(p)


def project_inside_layer(boundary, gxy, x_ref, y_ref):
    sec = section_at_gxy(boundary, gxy)
    if sec is None:
        raise ValueError(f"No section at Gxy={gxy:.6e}")

    cx, cy, ax_eff, ay_eff, y_min, y_max = sec
    if abs(float(ax_eff)) <= 1e-14 or abs(float(ay_eff)) <= 1e-14:
        return np.array([cx, cy, float(gxy)], dtype=float)

    y = float(np.clip(y_ref, y_min, y_max))
    xl, xr = x_bounds_at_y(boundary, y, cx, cy, ax_eff, ay_eff)
    x = float(np.clip(x_ref, xl, xr))
    return np.array([x, y, float(gxy)], dtype=float)


def make_levels(y_start, y_end, target_dy):
    dy = abs(float(y_end) - float(y_start))
    if dy <= 1e-14:
        return np.array([float(y_start)], dtype=float)
    n = int(max(2, np.ceil(dy / float(target_dy)) + 1))
    return np.linspace(float(y_start), float(y_end), n)


def build_serpentine_path(boundary, gxy, y_levels, start_at_origin=True, scan_left_to_right=True):
    points = []
    if start_at_origin:
        # 1. Start at absolute origin
        append_if_different(points, [0.0, 0.0, 0.0])
        
        # 2. Determine the maximum vertical 'safe' shear for the origin (0,0)
        g_safe_limit = safe_gxy_for_origin(boundary)
        
        # Vertical segment up to the safe limit
        g_vert = float(np.clip(abs(gxy), 0.0, g_safe_limit)) * np.sign(gxy)
        append_if_different(points, [0.0, 0.0, g_vert])
    
    # 3. Projected 'center' in the target Gxy plane
    # If Gxy > g_safe_limit, this point will be on the inner surface of the ellipsoid
    plane_origin = project_inside_layer(boundary, gxy, 0.0, 0.0)
    append_if_different(points, plane_origin)

    if len(y_levels) > 0:
        # 4. Axis-aligned transition to the first row's Y-coordinate
        # Ensuring we stay inside by projecting the intermediate point
        yfirst = y_levels[0]
        row_entry = project_inside_layer(boundary, gxy, plane_origin[0], yfirst)
        append_if_different(points, row_entry)

    sec = section_at_gxy(boundary, gxy)
    cx, cy, ax_eff, ay_eff, _, _ = sec

    on_right = not scan_left_to_right
    for y in y_levels:
        xl, xr = x_bounds_at_y(boundary, y, cx, cy, ax_eff, ay_eff)
        # Move along X axis to the start of the row
        if on_right:
            append_if_different(points, [xr, y, float(gxy)])
            append_if_different(points, [xl, y, float(gxy)])
        else:
            append_if_different(points, [xl, y, float(gxy)])
            append_if_different(points, [xr, y, float(gxy)])
        on_right = not on_right

    return np.array(points, dtype=float)


def build_two_trajectories_for_level(boundary, gxy, target_dy):
    sec = section_at_gxy(boundary, gxy)
    if sec is None:
        return None, None

    _, _, ax_eff, ay_eff, y_min, y_max = sec
    if abs(float(ax_eff)) <= 1e-14 or abs(float(ay_eff)) <= 1e-14:
        center = project_inside_layer(boundary, gxy, 0.0, 0.0)
        t = np.array([[0.0, 0.0, 0.0], center], dtype=float)
        return t, t.copy()

    # Projected origin in the Gxy plane
    plane_origin = project_inside_layer(boundary, gxy, 0.0, 0.0)
    y0 = plane_origin[1]
    
    y_up = make_levels(y0, y_max, target_dy)
    y_dn = make_levels(y0, y_min, target_dy)

    # Positive trajectory (upper side sweep)
    # Visualizer used xl -> xr for t_pos (scan_left_to_right=True)
    t_pos = build_serpentine_path(boundary, gxy, y_up, start_at_origin=True, scan_left_to_right=True)

    # Negative trajectory (lower side sweep)
    # Visualizer used xr -> xl for t_neg (scan_left_to_right=False)
    t_neg = build_serpentine_path(boundary, gxy, y_dn, start_at_origin=True, scan_left_to_right=False)

    return t_pos, t_neg


def build_stage0_trajectories(boundary, n_rows_ref, n_layers):
    sec0 = section_at_gxy(boundary, 0.0)
    if sec0 is None:
        raise ValueError("No Exx-Eyy cross-section exists at Gxy=0.")

    _, _, _, _, y_min0, y_max0 = sec0
    span0 = float(y_max0 - y_min0)
    if span0 <= 1e-14:
        raise ValueError("Degenerate Exx-Eyy section at Gxy=0.")

    n_rows_ref = int(max(2, n_rows_ref))
    target_dy = span0 / float(n_rows_ref - 1)

    n_layers = int(max(2, n_layers))
    gxy_pos = float(boundary["limits"]["Gxy"][1])
    gxy_neg = float(boundary["limits"]["Gxy"][0])

    levels = [0.0]
    if gxy_pos > 0.0:
        levels.extend(np.linspace(0.0, gxy_pos, n_layers)[1:].tolist())
    if gxy_neg < 0.0:
        levels.extend(np.linspace(0.0, gxy_neg, n_layers)[1:].tolist())

    trajectories = []
    labels = []
    for g in levels:
        t_pos, t_neg = build_two_trajectories_for_level(boundary, float(g), target_dy)
        if t_pos is None:
            continue
        trajectories.append(t_pos)
        labels.append((float(g), "pos"))
        trajectories.append(t_neg)
        labels.append((float(g), "neg"))

    return trajectories, labels, levels, target_dy


def scaled_path(path_e):
    p = np.array(path_e, dtype=float, copy=True)
    p[:, 2] *= 0.5
    return p


def compute_segment_steps(path_e, ref_steps, reference_amplitude):
    z = scaled_path(path_e)
    seg_len = np.linalg.norm(np.diff(z, axis=0), axis=1)
    steps = np.maximum(1, np.ceil(float(ref_steps) * seg_len / float(reference_amplitude)).astype(int))
    return seg_len, steps


def boundary_scaled_center_axes(boundary):
    c = np.array(boundary["center"], dtype=float).copy()
    a = np.array(boundary["axes"], dtype=float).copy()
    c[2] *= 0.5
    a[2] *= 0.5
    return c, a


def compute_bounds(paths_e, boundary, pad_ratio=0.08):
    pts = np.vstack([scaled_path(p) for p in paths_e])
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)

    c, a = boundary_scaled_center_axes(boundary)
    mins = np.minimum(mins, c - a)
    maxs = np.maximum(maxs, c + a)

    span = np.maximum(maxs - mins, 1e-12)
    pad = pad_ratio * span
    return mins - pad, maxs + pad


def draw_boundary(ax, boundary):
    if boundary.get("domain_type") == "box":
        _draw_box_boundary(ax, boundary)
        return
    c, a = boundary_scaled_center_axes(boundary)
    u = np.linspace(0.0, 2.0 * np.pi, 56)
    v = np.linspace(0.0, np.pi, 28)
    uu, vv = np.meshgrid(u, v)
    x = c[0] + a[0] * np.cos(uu) * np.sin(vv)
    y = c[1] + a[1] * np.sin(uu) * np.sin(vv)
    z = c[2] + a[2] * np.cos(vv)
    ax.plot_surface(x, y, z, color="gray", alpha=0.10, linewidth=0.0)


def _draw_box_boundary(ax, boundary):
    c, a = boundary_scaled_center_axes(boundary)
    xs = [c[0] - a[0], c[0] + a[0]]
    ys = [c[1] - a[1], c[1] + a[1]]
    zs = [c[2] - a[2], c[2] + a[2]]
    for i in range(2):
        for j in range(2):
            ax.plot([xs[0], xs[1]], [ys[i], ys[i]], [zs[j], zs[j]], "k-", lw=0.7, alpha=0.25)
            ax.plot([xs[i], xs[i]], [ys[0], ys[1]], [zs[j], zs[j]], "k-", lw=0.7, alpha=0.25)
            ax.plot([xs[i], xs[i]], [ys[j], ys[j]], [zs[0], zs[1]], "k-", lw=0.7, alpha=0.25)


def set_axes3d(ax, mins, maxs, title):
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect((maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]))
    ax.set_xlabel("Exx")
    ax.set_ylabel("Eyy")
    ax.set_zlabel("Gxy/2")
    ax.set_title(title)


def path_style(i, op1, op2):
    # odd index -> positive trajectory, even index -> negative trajectory
    if (i % 2) == 0:
        return "tab:blue", float(op1)
    return "tab:orange", float(op2)


def save_coverage_image(paths, boundary, out_file, op1, op2):
    mins, maxs = compute_bounds(paths, boundary)

    fig = plt.figure(figsize=(12, 5.8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    draw_boundary(ax3d, boundary)

    for i, p in enumerate(paths):
        ps = scaled_path(p)
        color, alpha = path_style(i, op1, op2)
        label = None
        if i == 0:
            label = "Positive trajectories"
        if i == 1:
            label = "Negative trajectories"
        ax3d.plot(ps[:, 0], ps[:, 1], ps[:, 2], color=color, lw=1.2, alpha=alpha, label=label)

    ax3d.scatter([0.0], [0.0], [0.0], color="k", s=28, label="Start")
    set_axes3d(ax3d, mins, maxs, "Stage 0 Trajectory Coverage (3D)")
    ax3d.view_init(elev=24.0, azim=40.0)
    ax3d.legend(loc="upper left", fontsize=8)

    ax2d = fig.add_subplot(1, 2, 2)
    for i, p in enumerate(paths):
        ps = scaled_path(p)
        color, alpha = path_style(i, op1, op2)
        ax2d.plot(ps[:, 0], ps[:, 1], color=color, lw=1.0, alpha=alpha)

    sec0 = section_at_gxy(boundary, 0.0)
    if sec0 is not None:
        cx, cy, ax_eff, ay_eff, _, _ = sec0
        if boundary.get("domain_type") == "box":
            rect_x = [cx - ax_eff, cx + ax_eff, cx + ax_eff, cx - ax_eff, cx - ax_eff]
            rect_y = [cy - ay_eff, cy - ay_eff, cy + ay_eff, cy + ay_eff, cy - ay_eff]
            ax2d.plot(rect_x, rect_y, "k--", lw=1.0, alpha=0.75)
        else:
            t = np.linspace(0.0, 2.0 * np.pi, 300)
            ax2d.plot(cx + ax_eff * np.cos(t), cy + ay_eff * np.sin(t), "k--", lw=1.0, alpha=0.75)

    ax2d.set_xlabel("Exx")
    ax2d.set_ylabel("Eyy")
    ax2d.set_title("Projection Exx-Eyy")
    ax2d.grid(True, alpha=0.3)
    ax2d.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def save_final_overview(paths, boundary, out_file, op1, op2):
    mins, maxs = compute_bounds(paths, boundary)

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    draw_boundary(ax, boundary)

    for i, p in enumerate(paths):
        ps = scaled_path(p)
        color, alpha = path_style(i, op1, op2)
        ax.plot(ps[:, 0], ps[:, 1], ps[:, 2], color=color, lw=1.3, alpha=alpha)

    ax.scatter([0.0], [0.0], [0.0], color="k", s=30, label="Start")
    set_axes3d(ax, mins, maxs, "Stage 0 Final Overview")
    ax.view_init(elev=24.0, azim=38.0)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def interpolate_path(path_scaled, n_frames):
    p = np.array(path_scaled, dtype=float)
    if p.shape[0] < 2:
        return np.repeat(p, int(max(1, n_frames)), axis=0)

    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if total <= 1e-14:
        return np.repeat(p[:1, :], int(max(1, n_frames)), axis=0)

    s_query = np.linspace(0.0, total, int(max(2, n_frames)))
    out = np.zeros((len(s_query), 3), dtype=float)
    for i, sq in enumerate(s_query):
        k = int(np.searchsorted(cum, sq, side="right") - 1)
        k = max(0, min(k, len(seg) - 1))
        ds = seg[k]
        if ds <= 1e-14:
            out[i, :] = p[k, :]
        else:
            xi = (sq - cum[k]) / ds
            out[i, :] = (1.0 - xi) * p[k, :] + xi * p[k + 1, :]
    return out


def save_movie(paths, boundary, out_file, n_frames, fps, op1, op2):
    mins, maxs = compute_bounds(paths, boundary)

    anim_paths = [interpolate_path(scaled_path(p), n_frames) for p in paths]
    full_paths = [scaled_path(p) for p in paths]

    fig = plt.figure(figsize=(8.8, 6.8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    draw_boundary(ax, boundary)
    set_axes3d(ax, mins, maxs, "Stage 0 Trajectories (Animated)")

    lines = []
    heads = []
    for i, pfull in enumerate(full_paths):
        color, alpha = path_style(i, op1, op2)
        faint = max(0.10, 0.30 * alpha)
        ax.plot(pfull[:, 0], pfull[:, 1], pfull[:, 2], color=color, lw=0.9, alpha=faint)
        line, = ax.plot([], [], [], color=color, lw=1.8, alpha=alpha)
        head = ax.scatter([], [], [], color=color, s=22, alpha=alpha)
        lines.append(line)
        heads.append(head)

    ax.scatter([0.0], [0.0], [0.0], color="k", s=30, label="Start")

    def _update(frame):
        for i, panim in enumerate(anim_paths):
            pp = panim[: frame + 1, :]
            lines[i].set_data(pp[:, 0], pp[:, 1])
            lines[i].set_3d_properties(pp[:, 2])
            heads[i]._offsets3d = ([pp[-1, 0]], [pp[-1, 1]], [pp[-1, 2]])

        az = 35.0 + 0.30 * frame
        ax.view_init(elev=24.0, azim=az)
        return tuple(lines + heads)

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=int(max(2, n_frames)),
        interval=1000.0 / float(max(1, fps)),
        blit=False,
    )

    writer = animation.FFMpegWriter(fps=int(max(1, fps)), bitrate=2500)
    ani.save(out_file, writer=writer)
    plt.close(fig)


def save_individual_plots(paths, labels, boundary, out_dir):
    debug_dir = os.path.join(out_dir, "individual_trajectories")
    os.makedirs(debug_dir, exist_ok=True)
    mins, maxs = compute_bounds(paths, boundary)

    for i, p in enumerate(paths, start=1):
        g, kind = labels[i-1]
        ps = scaled_path(p)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        
        draw_boundary(ax, boundary)
        ax.plot(ps[:, 0], ps[:, 1], ps[:, 2], color="tab:blue" if kind=="pos" else "tab:orange", lw=1.5, label=f"Traj {i} ({kind})")
        ax.scatter([0], [0], [0], color="k", s=40, label="Start (0,0,0)")
        
        set_axes3d(ax, mins, maxs, f"Trajectory {i:02d}: Gxy={g:.3f} ({kind})")
        ax.view_init(elev=22, azim=35)
        ax.legend(loc="upper left", fontsize=8)
        
        fname = os.path.join(debug_dir, f"trajectory_{i:02d}_{g:+.3f}_{kind}.png")
        fig.tight_layout()
        fig.savefig(fname, dpi=120)
        plt.close(fig)


def save_stage0_bundle(paths, labels, levels, out_file, ref_steps, reference_amplitude, rel_boundary, emax, domain_type):
    payload = {
        "trajectory_count": np.array([len(paths)], dtype=int),
        "ref_steps": np.array([int(ref_steps)], dtype=int),
        "reference_amplitude": np.array([float(reference_amplitude)], dtype=float),
        "relative_boundary": np.array(rel_boundary, dtype=float),
        "gxy_levels": np.array(levels, dtype=float),
        "emax": np.array([float(emax)], dtype=float),
        "domain_type": np.array([domain_type], dtype=str),
    }

    # labels as [gxy, sign_flag] with sign_flag: +1 (pos), -1 (neg)
    labels_array = []
    for g, sign in labels:
        labels_array.append([float(g), 1.0 if sign == "pos" else -1.0])
    payload["trajectory_labels"] = np.array(labels_array, dtype=float)

    for i, p in enumerate(paths, start=1):
        payload[f"trajectory_{i}"] = np.array(p, dtype=float)

    np.savez(out_file, **payload)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 0 trajectory generator")
    p.add_argument(
        "--design",
        type=str,
        default="box",
        choices=["box"],
        help="Domain geometry. Ellipsoid mode is disabled; box is always used.",
    )
    p.add_argument("--emax", type=float, default=2.0)
    p.add_argument("--relative-boundary", type=str, default="1.0,0.05,1.0,0.05,0.05,0.05")
    p.add_argument("--rows-per-layer", type=int, default=9)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--traj1-opacity", type=float, default=0.90)
    p.add_argument("--traj2-opacity", type=float, default=0.45)
    p.add_argument("--projection-opacity", type=float, default=0.35)  # kept for compatibility
    p.add_argument("--ref-steps", type=int, default=400)
    p.add_argument("--reference-amplitude", type=float, default=0.5)
    p.add_argument("--movie-frames", type=int, default=300)
    p.add_argument("--movie-fps", type=int, default=24)
    p.add_argument("--out-dir", type=str, default="stage_0_trajectory")
    p.add_argument("--trajectory-file", type=str, default="stage_0_trajectories.npz")
    return p.parse_args()


def main():
    args = parse_args()

    if args.design != "box":
        raise ValueError("Only box design is supported.")
    if args.rows_per_layer < 2:
        raise ValueError("rows-per-layer must be >= 2")
    if args.n_layers < 2:
        raise ValueError("n-layers must be >= 2")

    domain_type = "box"
    rel = parse_relative_boundary(args.relative_boundary)
    boundary = build_boundary(args.emax, rel, domain_type=domain_type)

    paths, labels, levels, target_dy = build_stage0_trajectories(
        boundary=boundary,
        n_rows_ref=args.rows_per_layer,
        n_layers=args.n_layers,
    )

    all_steps = []
    for p in paths:
        _, st = compute_segment_steps(p, args.ref_steps, args.reference_amplitude)
        all_steps.append(int(np.sum(st)))

    os.makedirs(args.out_dir, exist_ok=True)
    img1 = os.path.join(args.out_dir, "training_trajectory_coverage.png")
    img2 = os.path.join(args.out_dir, "trajectory_01_02_final_overview_3d.png")
    mp4 = os.path.join(args.out_dir, "trajectory_01_02_comparison_3d.mp4")
    bundle = os.path.join(args.out_dir, args.trajectory_file)

    save_coverage_image(paths, boundary, img1, args.traj1_opacity, args.traj2_opacity)
    save_final_overview(paths, boundary, img2, args.traj1_opacity, args.traj2_opacity)
    save_movie(paths, boundary, mp4, args.movie_frames, args.movie_fps, args.traj1_opacity, args.traj2_opacity)
    save_stage0_bundle(
        paths=paths,
        labels=labels,
        levels=levels,
        out_file=bundle,
        ref_steps=args.ref_steps,
        reference_amplitude=args.reference_amplitude,
        rel_boundary=rel,
        emax=args.emax,
        domain_type=domain_type,
    )
    save_individual_plots(paths, labels, boundary, args.out_dir)

    print(f"[INFO] Saved image: {img1}")
    print(f"[INFO] Saved image: {img2}")
    print(f"[INFO] Saved movie: {mp4}")
    print(f"[INFO] Saved stage0 trajectories: {bundle}")
    print(f"[INFO] trajectories_generated={len(paths)} (2 per level)")
    print(f"[INFO] levels_used={levels}")
    print(f"[INFO] target_dy={target_dy:.6e}")
    print(f"[INFO] ref_steps={args.ref_steps}, reference_amplitude={args.reference_amplitude}")
    print(f"[INFO] steps_per_trajectory={all_steps}")
    print(f"[INFO] TOTAL estimated steps: {int(np.sum(all_steps))}")


if __name__ == "__main__":
    main()
