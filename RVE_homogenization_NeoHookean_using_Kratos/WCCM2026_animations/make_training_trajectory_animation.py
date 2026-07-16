#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create a side-by-side GIF of the RVE training trajectories.

Left panel: deformed RVE mesh rendered with PyVista from saved FOM displacement
history.

Right panel: current point on the applied strain trajectory in
(E_xx, E_yy, G_xy) parameter space.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_MDPA = PROJECT_DIR / "rve_geometry.mdpa"
DEFAULT_TRAJ_DIR = PROJECT_DIR / "stage_1_training_set_fom"
DEFAULT_STAGE0_FILE = PROJECT_DIR / "stage_0_trajectory" / "stage_0_trajectories.npz"


@dataclass(frozen=True)
class MdpaMesh:
    points: np.ndarray
    cells: np.ndarray
    cell_types: np.ndarray


@dataclass(frozen=True)
class RenderStats:
    xy_min: np.ndarray
    xy_max: np.ndarray
    max_displacement: float


def parse_mdpa_quadratic_triangles(mdpa_file: Path) -> MdpaMesh:
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: list[list[int]] = []
    in_nodes = False
    in_elements = False

    with mdpa_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("Begin Nodes"):
                in_nodes = True
                continue
            if line.startswith("End Nodes"):
                in_nodes = False
                continue

            if line.startswith("Begin Geometries Triangle2D6"):
                in_elements = True
                continue
            if line.startswith("End Geometries"):
                in_elements = False
                continue

            if in_nodes:
                parts = line.split()
                node_id = int(parts[0])
                nodes[node_id] = (float(parts[1]), float(parts[2]), float(parts[3]))
                continue

            if in_elements:
                parts = line.split()
                elements.append([int(v) for v in parts[1:7]])

    if not nodes:
        raise RuntimeError(f"No nodes found in {mdpa_file}")
    if not elements:
        raise RuntimeError(f"No Triangle2D6 geometries found in {mdpa_file}")

    sorted_ids = sorted(nodes)
    if sorted_ids != list(range(1, len(sorted_ids) + 1)):
        raise RuntimeError(
            "This animation assumes consecutive node IDs starting at 1, "
            "matching the saved FOM equation-vector ordering."
        )

    points = np.array([nodes[node_id] for node_id in sorted_ids], dtype=float)
    id_to_idx = {node_id: i for i, node_id in enumerate(sorted_ids)}

    cell_blocks = []
    for elem in elements:
        conn = [id_to_idx[node_id] for node_id in elem]
        cell_blocks.append(np.array([6] + conn, dtype=np.int64))

    cells = np.concatenate(cell_blocks)
    cell_types = np.full(len(elements), pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8)
    return MdpaMesh(points=points, cells=cells, cell_types=cell_types)


def load_fom_history(
    trajectory_dir: Path,
    trajectory_index: int,
    mmap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    directory = trajectory_dir / f"trajectory_{trajectory_index}"
    prefix = f"trajectory_{trajectory_index}"
    u_file = directory / f"{prefix}_U.npy"
    strain_file = directory / f"{prefix}_applied_strain.npy"

    if not u_file.exists():
        raise FileNotFoundError(u_file)
    if not strain_file.exists():
        raise FileNotFoundError(strain_file)

    mmap_mode = "r" if mmap else None
    return np.load(u_file, mmap_mode=mmap_mode), np.load(strain_file, mmap_mode=mmap_mode)


def load_stage0_paths(stage0_file: Path) -> list[np.ndarray]:
    if not stage0_file.exists():
        return []
    data = np.load(stage0_file)
    count = int(np.ravel(data["trajectory_count"])[0])
    return [np.asarray(data[f"trajectory_{i}"], dtype=float) for i in range(1, count + 1)]


def parse_trajectory_indices(text: str, stage0_paths: list[np.ndarray]) -> list[int]:
    raw = str(text).strip().lower()
    if raw in {"all", "1-10", "1:10"}:
        count = len(stage0_paths) if stage0_paths else 10
        return list(range(1, count + 1))

    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(v.strip()) for v in part.split("-", 1)]
            step = 1 if b >= a else -1
            indices.extend(range(a, b + step, step))
        else:
            indices.append(int(part))

    if not indices:
        raise ValueError("No trajectory indices were requested.")
    return indices


def choose_frame_indices(n_steps: int, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or max_frames >= n_steps:
        return np.arange(n_steps, dtype=int)
    return np.unique(np.linspace(0, n_steps - 1, int(max_frames), dtype=int))


def current_deformed_points(
    reference_points: np.ndarray,
    u_frame: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    disp = np.asarray(u_frame).reshape(reference_points.shape[0], 2)
    points = reference_points.copy()
    points[:, :2] += scale * disp
    disp_mag = np.linalg.norm(disp, axis=1)
    return points, disp_mag


def compute_render_stats(
    mesh_data: MdpaMesh,
    trajectory_dir: Path,
    trajectory_indices: list[int],
    frames_per_trajectory: int,
    deformation_scale: float,
) -> RenderStats:
    ref_xy = mesh_data.points[:, :2]
    xy_min = np.full(2, np.inf, dtype=float)
    xy_max = np.full(2, -np.inf, dtype=float)
    max_displacement = 0.0

    for trajectory_index in trajectory_indices:
        u_history, _ = load_fom_history(trajectory_dir, trajectory_index, mmap=True)
        frame_indices = choose_frame_indices(u_history.shape[0], frames_per_trajectory)
        selected_u = np.asarray(u_history[frame_indices]).reshape(
            len(frame_indices),
            mesh_data.points.shape[0],
            2,
        )
        selected_xy = ref_xy[None, :, :] + deformation_scale * selected_u
        xy_min = np.minimum(xy_min, np.min(selected_xy, axis=(0, 1)))
        xy_max = np.maximum(xy_max, np.max(selected_xy, axis=(0, 1)))
        max_displacement = max(
            max_displacement,
            float(np.max(np.linalg.norm(selected_u, axis=2))),
        )

    return RenderStats(
        xy_min=xy_min,
        xy_max=xy_max,
        max_displacement=max(max_displacement, 1.0e-12),
    )


def configure_rve_plotter(
    mesh_data: MdpaMesh,
    initial_u_frame: np.ndarray,
    deformation_scale: float,
    panel_size: tuple[int, int],
    stats: RenderStats,
    camera_margin: float,
) -> tuple[pv.Plotter, pv.UnstructuredGrid]:
    first_points, first_mag = current_deformed_points(
        mesh_data.points,
        initial_u_frame,
        deformation_scale,
    )
    deformed_mesh = pv.UnstructuredGrid(
        mesh_data.cells,
        mesh_data.cell_types,
        first_points,
    )
    deformed_mesh.point_data["|u|"] = first_mag

    reference_mesh = pv.UnstructuredGrid(
        mesh_data.cells,
        mesh_data.cell_types,
        mesh_data.points,
    )

    center = 0.5 * (stats.xy_min + stats.xy_max)
    span = np.maximum(stats.xy_max - stats.xy_min, 1.0e-12)
    aspect = panel_size[0] / max(panel_size[1], 1)
    parallel_scale = float(camera_margin) * max(span[1], span[0] / aspect)

    plotter = pv.Plotter(off_screen=True, window_size=panel_size)
    plotter.set_background("white")
    plotter.add_mesh(
        reference_mesh,
        color="#cfcfcf",
        style="wireframe",
        line_width=0.30,
        opacity=0.28,
    )
    plotter.add_mesh(
        deformed_mesh,
        scalars="|u|",
        cmap="coolwarm",
        clim=(0.0, stats.max_displacement),
        show_edges=True,
        edge_color="#111111",
        line_width=0.18,
        scalar_bar_args={
            "title": "|u|",
            "vertical": False,
            "position_x": 0.28,
            "position_y": 0.018,
            "width": 0.44,
            "height": 0.050,
            "title_font_size": 12,
            "label_font_size": 9,
            "color": "black",
            "fmt": "%.2f",
        },
    )
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera.focal_point = (float(center[0]), float(center[1]), 0.0)
    plotter.camera.position = (float(center[0]), float(center[1]), 10.0)
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.parallel_scale = parallel_scale
    return plotter, deformed_mesh


def render_rve_frame(
    plotter: pv.Plotter,
    mesh: pv.UnstructuredGrid,
    reference_points: np.ndarray,
    u_frame: np.ndarray,
    deformation_scale: float,
) -> np.ndarray:
    points, disp_mag = current_deformed_points(reference_points, u_frame, deformation_scale)
    mesh.points[:, :] = points
    mesh.point_data["|u|"][:] = disp_mag
    mesh.GetPoints().Modified()
    mesh.Modified()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plotter.update_scalars(disp_mag, mesh=mesh, render=False)
    plotter.render()
    return np.asarray(plotter.screenshot(return_img=True))


def setup_parametric_axes(
    ax,
    stage0_paths: list[np.ndarray],
    elev: float,
    azim: float,
) -> None:
    stacked = np.vstack([path for path in stage0_paths if path.size])
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    pad = np.maximum(0.08 * (maxs - mins), 0.03)

    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    ax.set_xlabel(r"$E_{xx}$")
    ax.set_ylabel(r"$E_{yy}$")
    ax.set_zlabel(r"$G_{xy}$")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, color="#d9d9d9", linewidth=0.5)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 1))


def render_parametric_frame(
    stage0_paths: list[np.ndarray],
    full_path: np.ndarray,
    current_step_index: int,
    trajectory_index: int | None,
    sequence_index: int | None,
    selected_count: int,
    design_count: int,
    completed_trajectories: set[int],
    panel_size: tuple[int, int],
    elev: float,
    azim: float,
    final_view: bool = False,
) -> np.ndarray:
    width, height = panel_size
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    setup_parametric_axes(ax, stage0_paths, elev=elev, azim=azim)

    for i, path in enumerate(stage0_paths, start=1):
        if path.shape[0] < 2:
            continue
        is_completed = i in completed_trajectories
        is_current = trajectory_index == i
        if is_current:
            color = "#284b63"
            linewidth = 1.5
            alpha = 0.42
        elif is_completed or final_view:
            color = "#b75d2a"
            linewidth = 1.4
            alpha = 0.70
        else:
            color = "#b8b8b8"
            linewidth = 0.75
            alpha = 0.25

        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

    if full_path.size and trajectory_index is not None:
        current_step_index = int(np.clip(current_step_index, 0, full_path.shape[0] - 1))
        progress = full_path[: current_step_index + 1]
        current = full_path[current_step_index]
        ax.plot(
            full_path[:, 0],
            full_path[:, 1],
            full_path[:, 2],
            color="#284b63",
            linewidth=1.0,
            alpha=0.25,
        )
        ax.plot(
            progress[:, 0],
            progress[:, 1],
            progress[:, 2],
            color="#d1492e",
            linewidth=3.2,
        )
        ax.scatter(
            [current[0]],
            [current[1]],
            [current[2]],
            s=58,
            color="#d1492e",
            edgecolor="black",
            linewidth=0.55,
            depthshade=False,
        )
        title = (
            f"Representative path {sequence_index}/{selected_count} "
            f"(trajectory {trajectory_index}/{design_count})"
        )
    else:
        if selected_count == design_count:
            title = f"All {design_count} training trajectories"
        else:
            title = f"{selected_count} representative paths; full {design_count}-path design"

    ax.set_title(title, pad=12)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_outer_title(frame: np.ndarray, title: str, left_info: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    title_font = load_font(18)
    info_font = load_font(17)
    band_height = 82
    draw.rectangle((0, 0, image.width, band_height), fill=(255, 255, 255))
    draw.text((18, 8), title, fill=(20, 20, 20), font=title_font)
    draw.text((18, 46), left_info, fill=(20, 20, 20), font=info_font)
    return np.asarray(image)


def make_animation(args: argparse.Namespace) -> Path:
    pv.OFF_SCREEN = True

    mesh_data = parse_mdpa_quadratic_triangles(args.mdpa_file)
    stage0_paths = load_stage0_paths(args.stage0_file)
    trajectory_indices = parse_trajectory_indices(args.trajectory_indices, stage0_paths)
    design_count = len(stage0_paths) if stage0_paths else max(trajectory_indices)
    selected_count = len(trajectory_indices)

    n_nodes = mesh_data.points.shape[0]
    panel_size = (args.panel_width, args.panel_height)
    stats = compute_render_stats(
        mesh_data,
        args.trajectory_dir,
        trajectory_indices,
        args.frames_per_trajectory,
        args.deformation_scale,
    )

    first_u_history, _ = load_fom_history(args.trajectory_dir, trajectory_indices[0], mmap=True)
    plotter, rve_mesh = configure_rve_plotter(
        mesh_data,
        first_u_history[0],
        args.deformation_scale,
        panel_size,
        stats,
        args.camera_margin,
    )

    output_file = args.output_dir / args.output_name
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = max(1, int(round(1000.0 / args.fps)))
    total_main_frames = len(trajectory_indices) * int(args.frames_per_trajectory)
    total_frames = total_main_frames + int(args.end_spin_frames)
    completed: set[int] = set()
    frame_number = 0
    last_rve_img: np.ndarray | None = None
    last_u_frame: np.ndarray | None = None

    with imageio.get_writer(output_file, mode="I", duration=frame_duration_ms, loop=0) as writer:
        for sequence_index, trajectory_index in enumerate(trajectory_indices, start=1):
            u_history, applied_history = load_fom_history(
                args.trajectory_dir,
                trajectory_index,
                mmap=True,
            )
            if u_history.ndim != 2 or u_history.shape[1] != 2 * n_nodes:
                raise RuntimeError(
                    f"Unexpected U history shape {u_history.shape}; expected (*, {2 * n_nodes})."
                )
            if applied_history.shape[0] != u_history.shape[0]:
                raise RuntimeError(
                    f"Trajectory {trajectory_index} has mismatched U/strain steps: "
                    f"{u_history.shape[0]} vs {applied_history.shape[0]}."
                )

            frame_indices = choose_frame_indices(
                u_history.shape[0],
                args.frames_per_trajectory,
            )
            full_path = np.asarray(applied_history)

            for local_frame_index, step_index in enumerate(frame_indices):
                current_strain = np.asarray(applied_history[step_index])
                rve_img = render_rve_frame(
                    plotter,
                    rve_mesh,
                    mesh_data.points,
                    u_history[step_index],
                    args.deformation_scale,
                )
                param_img = render_parametric_frame(
                    stage0_paths,
                    full_path,
                    step_index,
                    trajectory_index,
                    sequence_index,
                    selected_count,
                    design_count,
                    completed,
                    panel_size,
                    elev=args.param_elev,
                    azim=args.param_azim,
                )
                combined = np.concatenate([rve_img[..., :3], param_img[..., :3]], axis=1)
                combined = add_outer_title(
                    combined,
                    "RVE training data: FOM deformation and applied strain trajectory",
                    f"Representative path {sequence_index}/{selected_count} "
                    f"(trajectory {trajectory_index}/{design_count})  |  "
                    f"frame {frame_number + 1}/{total_frames}  |  "
                    f"visual scale {args.deformation_scale:.2f}x  |  "
                    f"E=({current_strain[0]:.3f}, {current_strain[1]:.3f}, {current_strain[2]:.3f})",
                )
                writer.append_data(combined)
                frame_number += 1
                last_rve_img = rve_img
                last_u_frame = np.asarray(u_history[step_index])

            completed.add(trajectory_index)

        if args.end_spin_frames > 0 and last_u_frame is not None:
            if last_rve_img is None:
                last_rve_img = render_rve_frame(
                    plotter,
                    rve_mesh,
                    mesh_data.points,
                    last_u_frame,
                    args.deformation_scale,
                )

            for spin_frame in range(args.end_spin_frames):
                if args.end_spin_frames > 1:
                    spin_fraction = spin_frame / float(args.end_spin_frames - 1)
                else:
                    spin_fraction = 0.0
                azim = args.param_azim + args.end_spin_degrees * spin_fraction
                param_img = render_parametric_frame(
                    stage0_paths,
                    np.zeros((0, 3), dtype=float),
                    0,
                    None,
                    None,
                    selected_count,
                    design_count,
                    completed,
                    panel_size,
                    elev=args.param_elev,
                    azim=azim,
                    final_view=True,
                )
                combined = np.concatenate([last_rve_img[..., :3], param_img[..., :3]], axis=1)
                combined = add_outer_title(
                    combined,
                    (
                        f"RVE training data: {selected_count} representative paths shown; "
                        f"{design_count}-path design"
                    ),
                    f"Final view  |  frame {frame_number + 1}/{total_frames}  |  "
                    f"visual scale {args.deformation_scale:.2f}x",
                )
                writer.append_data(combined)
                frame_number += 1

    plotter.close()
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create WCCM-style RVE training trajectory animation."
    )
    parser.add_argument(
        "--trajectory-indices",
        type=str,
        default="all",
        help='Trajectory list, e.g. "all", "1-10", or "1,3,5".',
    )
    parser.add_argument("--mdpa-file", type=Path, default=DEFAULT_MDPA)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--stage0-file", type=Path, default=DEFAULT_STAGE0_FILE)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--output-name",
        type=str,
        default="training_trajectories_1_to_10_rve_animation.gif",
    )
    parser.add_argument(
        "--frames-per-trajectory",
        type=int,
        default=60,
        help="Rendered RVE frames per trajectory; parameter paths are still drawn from full histories.",
    )
    parser.add_argument("--fps", type=float, default=22.5)
    parser.add_argument("--end-spin-frames", type=int, default=35)
    parser.add_argument("--end-spin-degrees", type=float, default=160.0)
    parser.add_argument("--panel-width", type=int, default=600)
    parser.add_argument("--panel-height", type=int, default=533)
    parser.add_argument("--deformation-scale", type=float, default=1.0)
    parser.add_argument(
        "--camera-margin",
        type=float,
        default=0.75,
        help="Larger values leave more whitespace around the deformed RVE.",
    )
    parser.add_argument("--param-elev", type=float, default=23.0)
    parser.add_argument("--param-azim", type=float, default=-58.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = make_animation(args)
    print(f"Saved animation to {output_file}")


if __name__ == "__main__":
    main()
