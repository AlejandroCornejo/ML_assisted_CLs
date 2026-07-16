#!/usr/bin/env python3
"""Render a synchronized HPROM-ANN response animation for the WCCM slides.

Panels
------
Left
    The ANN-decoded, deformed RVE.  The full RVE is displayed, while the
    actual MAW-ECM supports used online are overlaid: residual elements in
    orange and stress elements in blue.
Upper right
    The Stage 10 macro-strain path in ``(E_xx, E_yy, G_xy)``.
Lower right
    Equivalent homogenized stress over the load steps.

The RVE states are reconstructed from the saved HPROM-ANN primary-coordinate
history.  They are not FOM displacement snapshots.
"""

from __future__ import annotations

import argparse
import json
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
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_RESULT_DIR = PROJECT_DIR / (
    "stage_10_hprom_ann_ls_results_mawecm_res_eps_sig_phase1to40_phase2to10_sum990_ann_hrom"
)
DEFAULT_ECM_FILE = PROJECT_DIR / (
    "stage_12_hprom_ann_mawecm_res_eps_sig_phase1to40_phase2to10_sum990_ann/"
    "ecm_weights_all.npz"
)
DEFAULT_ANN_DIR = PROJECT_DIR / "stage_7_ann_model_ls"
DEFAULT_POD_DIR = PROJECT_DIR / "stage_2_pod_rve"
DEFAULT_MDPA_FILE = PROJECT_DIR / "rve_geometry.mdpa"

RESIDUAL_COLOR = "#F59E0B"
DIRECT_COLOR = "#D66522"
STRESS_COLOR = "#1976D2"
INK = "#18212B"
MUTED = "#7D8790"


@dataclass(frozen=True)
class MeshData:
    points: np.ndarray
    eq_map: np.ndarray
    cells: np.ndarray
    cell_types: np.ndarray


@dataclass(frozen=True)
class RveStats:
    xy_min: np.ndarray
    xy_max: np.ndarray
    max_displacement: float


@dataclass(frozen=True)
class PerformanceMetrics:
    fom_time_s: float
    hprom_time_s: float
    direct_time_s: float
    hprom_stress_error: float
    direct_stress_error: float

    @property
    def hprom_speedup(self) -> float:
        return self.fom_time_s / self.hprom_time_s

    @property
    def direct_speedup(self) -> float:
        return self.fom_time_s / self.direct_time_s


def parse_mdpa_quadratic_triangles(path: Path) -> MeshData:
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: list[list[int]] = []
    section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line == "Begin Nodes":
            section = "nodes"
            continue
        if line == "End Nodes":
            section = None
            continue
        if line.startswith("Begin Geometries Triangle2D6"):
            section = "triangles"
            continue
        if line == "End Geometries":
            section = None
            continue
        fields = line.split()
        if section == "nodes":
            nodes[int(fields[0])] = (float(fields[1]), float(fields[2]), float(fields[3]))
        elif section == "triangles":
            elements.append([int(v) for v in fields[1:7]])

    if not nodes or not elements:
        raise RuntimeError(f"Could not parse quadratic-triangle mesh from {path}.")

    node_ids = sorted(nodes)
    if node_ids != list(range(1, len(node_ids) + 1)):
        raise RuntimeError("The animation expects consecutive node IDs starting at 1.")
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    points = np.asarray([nodes[node_id] for node_id in node_ids], dtype=float)

    cell_blocks = [
        np.asarray([6] + [node_to_index[node_id] for node_id in element], dtype=np.int64)
        for element in elements
    ]
    cells = np.concatenate(cell_blocks)
    cell_types = np.full(len(elements), pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8)
    eq_map = np.arange(2 * len(node_ids), dtype=np.int64).reshape(len(node_ids), 2)
    return MeshData(points=points, eq_map=eq_map, cells=cells, cell_types=cell_types)


def load_ann_decoder(ann_dir: Path) -> tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    # The training module is kept as the single source of truth for the ANN
    # architecture, so this decoder matches the one used by Stage 10.
    import sys

    project_text = str(PROJECT_DIR)
    if project_text not in sys.path:
        sys.path.insert(0, project_text)
    from stage7b_train_ann_manifold import ManifoldANN

    metadata = np.load(ann_dir / "manifold_ann_metadata.npz", allow_pickle=True)
    phi_p = np.asarray(np.load(ann_dir / "phi_m.npy"), dtype=float)
    phi_s = np.asarray(np.load(ann_dir / "phi_s.npy"), dtype=float)

    hidden_layers = tuple(int(v) for v in np.asarray(metadata["hidden_layers"]).ravel())
    activation = str(np.ravel(metadata["activation"])[0])
    dropout = float(np.ravel(metadata["dropout"])[0])
    use_batchnorm = bool(int(np.ravel(metadata["use_batchnorm"])[0]))
    origin_anchored = bool(int(np.ravel(metadata["origin_anchored"])[0])) if "origin_anchored" in metadata else False

    model = ManifoldANN(
        metadata["x_mean"],
        metadata["x_std"],
        metadata["y_mean"],
        metadata["y_std"],
        in_dim=int(np.ravel(metadata["input_dim"])[0]),
        out_dim=int(np.ravel(metadata["output_dim"])[0]),
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        origin_anchored=origin_anchored,
    )
    model.load_state_dict(torch.load(ann_dir / "manifold_ann.pt", map_location="cpu", weights_only=True))
    model.eval()
    return phi_p, phi_s, model


def deformation_gradient_from_green_lagrange(strain: np.ndarray) -> np.ndarray:
    exx, eyy, gamma_xy = np.asarray(strain, dtype=float)
    right_cauchy_green = np.array(
        [[1.0 + 2.0 * exx, gamma_xy], [gamma_xy, 1.0 + 2.0 * eyy]],
        dtype=float,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(right_cauchy_green)
    if np.min(eigenvalues) <= 0.0:
        raise RuntimeError(f"Invalid Green-Lagrange state with eigenvalues {eigenvalues}.")
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def make_affine_displacement(
    strain: np.ndarray,
    mesh: MeshData,
    domain_center: np.ndarray,
) -> np.ndarray:
    xy = mesh.points[:, :2] - domain_center.reshape(1, 2)
    deformation_gradient = deformation_gradient_from_green_lagrange(strain)
    displacement_xy = xy @ (deformation_gradient - np.eye(2)).T
    displacement = np.zeros(2 * mesh.points.shape[0], dtype=float)
    displacement[mesh.eq_map[:, 0]] = displacement_xy[:, 0]
    displacement[mesh.eq_map[:, 1]] = displacement_xy[:, 1]
    return displacement


def reconstruct_hprom_displacements(
    mesh: MeshData,
    result_dir: Path,
    ann_dir: Path,
    pod_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the accepted HPROM coordinate history to full nodal states."""
    q_history = np.asarray(np.load(result_dir / "hprom_ann_run_q_p.npy"), dtype=float)
    applied_strain = np.asarray(np.load(result_dir / "single_run_applied_strain.npy"), dtype=float)
    if q_history.shape[0] != applied_strain.shape[0]:
        raise RuntimeError("HPROM q history and applied-strain history have inconsistent lengths.")

    phi_p, phi_s, ann_model = load_ann_decoder(ann_dir)
    free_dofs = np.asarray(np.load(pod_dir / "free_dofs.npy"), dtype=np.int64)
    dirichlet_dofs = np.asarray(np.load(pod_dir / "dirichlet_dofs.npy"), dtype=np.int64)
    domain_center = np.asarray(np.load(pod_dir / "domain_center.npy"), dtype=float)

    if phi_p.shape[0] != free_dofs.size or phi_s.shape[0] != free_dofs.size:
        raise RuntimeError("POD basis and free-DOF data are inconsistent.")
    if phi_p.shape[1] != q_history.shape[1]:
        raise RuntimeError("Saved HPROM q history is incompatible with the ANN primary basis.")

    with torch.no_grad():
        q_secondary = ann_model(torch.as_tensor(q_history, dtype=torch.float32)).cpu().numpy()

    # This is the exact Stage 10 reconstruction after its origin/Jacobian
    # terms cancel algebraically: u_fluc = Phi_m q + Phi_s N_theta(q).
    fluctuation_free = q_history @ phi_p.T + q_secondary @ phi_s.T
    n_steps = q_history.shape[0]
    full_displacement = np.zeros((n_steps, 2 * mesh.points.shape[0]), dtype=float)

    for step, strain in enumerate(applied_strain):
        affine = make_affine_displacement(strain, mesh, domain_center)
        full_displacement[step, dirichlet_dofs] = affine[dirichlet_dofs]
        full_displacement[step, free_dofs] = affine[free_dofs] + fluctuation_free[step]

    stress = np.asarray(np.load(result_dir / "hprom_ann_stress.npy"), dtype=float)
    if stress.shape[0] != n_steps:
        raise RuntimeError("HPROM stress history and state history have inconsistent lengths.")
    return full_displacement, applied_strain, stress


def equivalent_stress(stress: np.ndarray) -> np.ndarray:
    sxx, syy, sxy = np.asarray(stress, dtype=float).T[:3]
    return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))


def load_performance_metrics(
    result_dir: Path,
    fom_stress: np.ndarray,
    hprom_stress: np.ndarray,
    direct_hprom_stress: np.ndarray,
) -> PerformanceMetrics:
    """Load timings from runs performed on the same 1,150-step test path."""
    hprom_timing = np.load(result_dir / "hprom_ann_iter_timing_stats.npz")
    direct_timing = np.load(result_dir / "dhprom_ann_timing_stats.npz")
    hprom_time_s = float(np.ravel(hprom_timing["rve_kernel"])[0])
    direct_time_s = float(np.ravel(direct_timing["rve_kernel"])[0])

    comparison_dir = PROJECT_DIR / "compare_3solvers_stage_path_results"
    comparison_path = np.asarray(
        np.load(comparison_dir / "fom_run" / "trajectory_1_applied_strain.npy"), dtype=float
    )
    stage10_path = np.asarray(np.load(result_dir / "single_run_applied_strain.npy"), dtype=float)
    if comparison_path.shape != stage10_path.shape or not np.allclose(comparison_path, stage10_path):
        raise RuntimeError("The saved FOM timing does not correspond to the Stage 10 test path.")

    fom_time_s: float | None = None
    for raw_line in (comparison_dir / "compare3_summary.txt").read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("time_fom:"):
            fom_time_s = float(raw_line.split(":", 1)[1].strip())
            break
    if fom_time_s is None:
        raise RuntimeError("Could not find time_fom in compare3_summary.txt.")

    hprom_error = float(np.linalg.norm(hprom_stress - fom_stress) / np.linalg.norm(fom_stress))
    direct_error = float(np.linalg.norm(direct_hprom_stress - fom_stress) / np.linalg.norm(fom_stress))
    return PerformanceMetrics(
        fom_time_s=fom_time_s,
        hprom_time_s=hprom_time_s,
        direct_time_s=direct_time_s,
        hprom_stress_error=hprom_error,
        direct_stress_error=direct_error,
    )


def cell_blocks(mesh: MeshData, selected_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = [mesh.cells[7 * int(index) : 7 * (int(index) + 1)] for index in selected_indices]
    return np.concatenate(blocks), mesh.cell_types[np.asarray(selected_indices, dtype=int)]


def full_to_nodal_displacement(displacement: np.ndarray, mesh: MeshData) -> np.ndarray:
    return np.asarray(displacement, dtype=float)[mesh.eq_map].reshape(mesh.points.shape[0], 2)


def compute_rve_stats(mesh: MeshData, displacements: np.ndarray) -> RveStats:
    xy_min = np.full(2, np.inf)
    xy_max = np.full(2, -np.inf)
    max_displacement = 0.0
    for displacement in displacements:
        nodal_displacement = full_to_nodal_displacement(displacement, mesh)
        xy = mesh.points[:, :2] + nodal_displacement
        xy_min = np.minimum(xy_min, np.min(xy, axis=0))
        xy_max = np.maximum(xy_max, np.max(xy, axis=0))
        max_displacement = max(max_displacement, float(np.linalg.norm(nodal_displacement, axis=1).max()))
    return RveStats(xy_min=xy_min, xy_max=xy_max, max_displacement=max(max_displacement, 1.0e-12))


def configure_rve_plotter(
    mesh: MeshData,
    initial_displacement: np.ndarray,
    residual_elements: np.ndarray,
    stress_elements: np.ndarray,
    stats: RveStats,
    window_size: tuple[int, int],
) -> tuple[pv.Plotter, pv.UnstructuredGrid, pv.UnstructuredGrid, pv.UnstructuredGrid]:
    nodal_displacement = full_to_nodal_displacement(initial_displacement, mesh)
    points = mesh.points.copy()
    points[:, :2] += nodal_displacement
    magnitude = np.linalg.norm(nodal_displacement, axis=1)

    deformed_mesh = pv.UnstructuredGrid(mesh.cells, mesh.cell_types, points)
    deformed_mesh.point_data["displacement"] = magnitude
    residual_cells, residual_types = cell_blocks(mesh, residual_elements)
    stress_cells, stress_types = cell_blocks(mesh, stress_elements)
    residual_mesh = pv.UnstructuredGrid(residual_cells, residual_types, points.copy())
    stress_mesh = pv.UnstructuredGrid(stress_cells, stress_types, points.copy())

    width, height = window_size
    span = np.maximum(stats.xy_max - stats.xy_min, 1.0e-12)
    center = 0.5 * (stats.xy_min + stats.xy_max)
    # One camera for the whole animation: global deformed extrema include every
    # reconstructed state, so the RVE never recentres or changes scale.  VTK's
    # parallel scale is the half-height of the viewport.
    parallel_scale = 0.54 * max(span[1], span[0] / (width / height))

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    plotter.add_mesh(
        deformed_mesh,
        scalars="displacement",
        cmap="coolwarm",
        clim=(0.0, stats.max_displacement),
        opacity=0.60,
        show_edges=True,
        edge_color="#35414D",
        line_width=0.22,
        scalar_bar_args={
            "title": "|u|",
            "vertical": False,
            "position_x": 0.27,
            "position_y": 0.016,
            "width": 0.48,
            "height": 0.052,
            "title_font_size": 12,
            "label_font_size": 9,
            "color": "black",
            "fmt": "%.2f",
        },
    )
    plotter.add_mesh(
        residual_mesh,
        color=RESIDUAL_COLOR,
        opacity=0.96,
        show_edges=True,
        edge_color="#FFF3D5",
        line_width=2.5,
    )
    plotter.add_mesh(
        stress_mesh,
        color=STRESS_COLOR,
        opacity=1.0,
        show_edges=True,
        edge_color="#FFFFFF",
        line_width=2.5,
    )
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera.focal_point = (float(center[0]), float(center[1]), 0.0)
    plotter.camera.position = (float(center[0]), float(center[1]), 10.0)
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.parallel_scale = parallel_scale
    return plotter, deformed_mesh, residual_mesh, stress_mesh


def render_rve_frame(
    plotter: pv.Plotter,
    deformed_mesh: pv.UnstructuredGrid,
    residual_mesh: pv.UnstructuredGrid,
    stress_mesh: pv.UnstructuredGrid,
    mesh: MeshData,
    displacement: np.ndarray,
) -> np.ndarray:
    nodal_displacement = full_to_nodal_displacement(displacement, mesh)
    points = mesh.points.copy()
    points[:, :2] += nodal_displacement
    magnitude = np.linalg.norm(nodal_displacement, axis=1)
    for displayed_mesh in (deformed_mesh, residual_mesh, stress_mesh):
        displayed_mesh.points[:, :] = points
        displayed_mesh.GetPoints().Modified()
        displayed_mesh.Modified()
    deformed_mesh.point_data["displacement"][:] = magnitude
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plotter.update_scalars(magnitude, mesh=deformed_mesh, render=False)
    plotter.render()
    return np.asarray(plotter.screenshot(return_img=True))[..., :3]


def set_macro_axes(ax, strain_path: np.ndarray) -> None:
    minimum = np.min(strain_path, axis=0)
    maximum = np.max(strain_path, axis=0)
    pad = np.maximum(0.08 * (maximum - minimum), 0.04)
    ax.set_xlim(minimum[0] - pad[0], maximum[0] + pad[0])
    ax.set_ylim(minimum[1] - pad[1], maximum[1] + pad[1])
    ax.set_zlim(minimum[2] - pad[2], maximum[2] + pad[2])
    ax.set_xlabel(r"$E_{xx}$", labelpad=2)
    ax.set_ylabel(r"$E_{yy}$", labelpad=2)
    ax.set_zlabel(r"$G_{xy}$", labelpad=2)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(True, color="#D8D8D8", linewidth=0.45)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.view_init(elev=22.0, azim=-54.0)
    ax.set_box_aspect((1.25, 1.15, 0.72))


def render_trajectory_frame(
    strain_path: np.ndarray,
    current_index: int,
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    set_macro_axes(ax, strain_path)
    current_index = int(np.clip(current_index, 0, len(strain_path) - 1))
    progress = strain_path[: current_index + 1]
    current = strain_path[current_index]
    ax.plot(*strain_path.T, color="#CCD3D9", linewidth=1.25, alpha=0.9)
    ax.plot(*progress.T, color=RESIDUAL_COLOR, linewidth=2.9, solid_capstyle="round")
    ax.scatter(
        [current[0]], [current[1]], [current[2]],
        s=38,
        color=RESIDUAL_COLOR,
        edgecolor=INK,
        linewidth=0.5,
        depthshade=False,
    )
    ax.set_title("Applied macro-strain trajectory", color=INK, fontsize=12, pad=3)
    fig.subplots_adjust(left=0.00, right=1.00, bottom=0.02, top=0.93)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def render_stress_frame(
    hprom_stress: np.ndarray,
    fom_stress: np.ndarray,
    direct_hprom_stress: np.ndarray,
    current_index: int,
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    sigma_hprom = equivalent_stress(hprom_stress) / 1.0e6
    sigma_fom = equivalent_stress(fom_stress) / 1.0e6
    sigma_direct = equivalent_stress(direct_hprom_stress) / 1.0e6
    step = np.arange(sigma_hprom.size)
    current_index = int(np.clip(current_index, 0, sigma_hprom.size - 1))

    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax.plot(step, sigma_fom, color="#8B949C", linewidth=1.15, linestyle="--", alpha=0.82)
    ax.plot(step, sigma_hprom, color="#BBD7EE", linewidth=1.35)
    ax.plot(step, sigma_direct, color="#F2C4A6", linewidth=1.35)
    ax.plot(
        step[: current_index + 1],
        sigma_fom[: current_index + 1],
        color=INK,
        linewidth=1.65,
        linestyle="--",
        label="FOM",
    )
    ax.plot(
        step[: current_index + 1],
        sigma_hprom[: current_index + 1],
        color=STRESS_COLOR,
        linewidth=2.45,
        label="HPROM-ANN",
    )
    ax.plot(
        step[: current_index + 1],
        sigma_direct[: current_index + 1],
        color=DIRECT_COLOR,
        linewidth=2.15,
        label="D-HPROM-ANN",
    )
    ax.scatter(
        [current_index],
        [sigma_hprom[current_index]],
        s=27,
        color=STRESS_COLOR,
        edgecolor=INK,
        linewidth=0.55,
        zorder=4,
    )
    ax.scatter(
        [current_index],
        [sigma_direct[current_index]],
        s=22,
        color=DIRECT_COLOR,
        edgecolor=INK,
        linewidth=0.5,
        zorder=4,
    )
    ymax = max(
        float(np.max(sigma_hprom)),
        float(np.max(sigma_fom)),
        float(np.max(sigma_direct)),
        1.0,
    )
    ax.set_xlim(0, len(step) - 1)
    ax.set_ylim(0, 1.28 * ymax)
    ax.set_xlabel("load step", fontsize=10.5, color=INK)
    ax.set_ylabel(r"$\overline{\sigma}_{\mathrm{eq}}$ [MPa]", fontsize=10.5, color=INK)
    ax.set_title("Homogenized equivalent stress", color=INK, fontsize=13.5, pad=6)
    ax.tick_params(labelsize=8.5, colors=INK)
    ax.grid(True, color="#D9DDE0", linewidth=0.55)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        fontsize=7.8,
        ncol=3,
        handlelength=2.0,
        columnspacing=1.2,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.65)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ) if bold else (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_and_compose(
    rve: np.ndarray,
    trajectory: np.ndarray,
    stress: np.ndarray,
    performance: PerformanceMetrics,
    current_step: int,
    n_steps: int,
) -> np.ndarray:
    canvas = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24, bold=True)
    heading_font = load_font(17, bold=True)
    detail_font = load_font(15)
    performance_font = load_font(13, bold=True)
    value_font = load_font(21, bold=True)
    small_font = load_font(11)

    draw.text((24, 16), "HPROM-ANN: hyper-reduced RVE response", fill=INK, font=title_font)
    draw.text((27, 62), "HPROM-ANN RVE", fill=INK, font=heading_font)
    draw.text((548, 62), "online cost", fill=INK, font=heading_font)
    draw.text((724, 62), "online load path and homogenized response", fill=INK, font=heading_font)

    rve_image = Image.fromarray(rve).resize((510, 610), Image.Resampling.LANCZOS)
    trajectory_image = Image.fromarray(trajectory).resize((540, 292), Image.Resampling.LANCZOS)
    stress_image = Image.fromarray(stress).resize((540, 292), Image.Resampling.LANCZOS)
    canvas.paste(rve_image, (20, 90))
    canvas.paste(trajectory_image, (718, 82))
    canvas.paste(stress_image, (718, 390))

    draw.rounded_rectangle((40, 95, 296, 133), radius=5, fill=(255, 255, 255), outline=(215, 219, 223), width=1)
    draw.rectangle((53, 108, 65, 120), fill=RESIDUAL_COLOR)
    draw.text((74, 101), "residual support: 10 elements", fill=INK, font=detail_font)
    draw.rectangle((53, 124, 65, 136), fill=STRESS_COLOR)
    draw.text((74, 117), "stress support: 10 elements", fill=INK, font=detail_font)

    # Compact performance/error strip: the RVE remains the dominant left panel.
    draw.line((535, 96, 535, 690), fill=(210, 215, 220), width=1)
    card_x = 549
    draw.text((card_x, 101), "FOM reference", fill=MUTED, font=small_font)
    draw.text((card_x, 116), f"{performance.fom_time_s:.0f} s", fill=INK, font=value_font)

    draw.rectangle((card_x, 166, card_x + 5, 243), fill=STRESS_COLOR)
    draw.text((card_x + 12, 163), "HPROM-ANN", fill=INK, font=performance_font)
    draw.text((card_x + 12, 180), f"{performance.hprom_time_s:.2f} s", fill=STRESS_COLOR, font=value_font)
    draw.text((card_x + 12, 208), f"{performance.hprom_speedup:.0f}x speed-up", fill=INK, font=detail_font)
    draw.text(
        (card_x + 12, 229),
        f"{100.0 * performance.hprom_stress_error:.3f}% stress error",
        fill=MUTED,
        font=small_font,
    )

    draw.rectangle((card_x, 283, card_x + 5, 360), fill=DIRECT_COLOR)
    draw.text((card_x + 12, 280), "D-HPROM-ANN", fill=INK, font=performance_font)
    draw.text((card_x + 12, 297), f"{performance.direct_time_s:.2f} s", fill=DIRECT_COLOR, font=value_font)
    draw.text((card_x + 12, 325), f"{performance.direct_speedup:.0f}x speed-up", fill=INK, font=detail_font)
    draw.text(
        (card_x + 12, 346),
        f"{100.0 * performance.direct_stress_error:.3f}% stress error",
        fill=MUTED,
        font=small_font,
    )
    draw.text((card_x, 392), "RVE online kernel", fill=MUTED, font=small_font)

    step_text = f"step {current_step:04d} / {n_steps - 1:04d}"
    bbox = draw.textbbox((0, 0), step_text, font=detail_font)
    draw.text((1253 - (bbox[2] - bbox[0]), 17), step_text, fill=MUTED, font=detail_font)
    return np.asarray(canvas)


def choose_frame_indices(n_steps: int, frames: int) -> np.ndarray:
    if frames <= 1:
        return np.array([n_steps - 1], dtype=int)
    return np.unique(np.linspace(0, n_steps - 1, frames, dtype=int))


def gif_duration_seconds(path: Path) -> float:
    """Read the encoded GIF duration after palette/frame optimization."""
    total_ms = 0
    with Image.open(path) as image:
        while True:
            total_ms += int(image.info.get("duration", 0))
            try:
                image.seek(image.tell() + 1)
            except EOFError:
                break
    return total_ms / 1000.0


def build_animation(args: argparse.Namespace) -> tuple[Path, Path]:
    pv.OFF_SCREEN = True
    mesh = parse_mdpa_quadratic_triangles(args.mdpa_file)
    ecm = np.load(args.ecm_file, allow_pickle=True)
    residual_elements = np.asarray(ecm["Z_res"], dtype=int)
    stress_elements = np.asarray(ecm["Z_sig"], dtype=int)
    if residual_elements.size != 10 or stress_elements.size != 10:
        raise RuntimeError(
            f"Expected 10 residual and 10 stress elements, got {residual_elements.size} and {stress_elements.size}."
        )

    full_displacement, strain_path, hprom_stress = reconstruct_hprom_displacements(
        mesh, args.result_dir, args.ann_dir, args.pod_dir
    )
    fom_stress = np.asarray(np.load(args.result_dir / "fom_stress.npy"), dtype=float)
    direct_hprom_stress = np.asarray(np.load(args.result_dir / "dhprom_ann_stress.npy"), dtype=float)
    if fom_stress.shape != hprom_stress.shape or direct_hprom_stress.shape != hprom_stress.shape:
        raise RuntimeError("FOM, HPROM-ANN, and D-HPROM-ANN stress histories must have matching shapes.")
    performance = load_performance_metrics(
        args.result_dir,
        fom_stress,
        hprom_stress,
        direct_hprom_stress,
    )

    frame_indices = choose_frame_indices(full_displacement.shape[0], args.main_frames)
    # Use all steps, not just sampled GIF frames, to define a permanent view.
    stats = compute_rve_stats(mesh, full_displacement)
    plotter, rve_mesh, residual_mesh, stress_mesh = configure_rve_plotter(
        mesh,
        full_displacement[frame_indices[0]],
        residual_elements,
        stress_elements,
        stats,
        window_size=(510, 610),
    )

    output_file = args.output_dir / args.output_name
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file = output_file.with_suffix(".json")
    duration_ms = max(1, int(round(1000.0 / args.fps)))
    total_frames = len(frame_indices) + args.end_hold_frames

    try:
        with imageio.get_writer(output_file, mode="I", duration=duration_ms, loop=0) as writer:
            last_composite: np.ndarray | None = None
            for frame_number, step_index in enumerate(frame_indices, start=1):
                rve_frame = render_rve_frame(
                    plotter, rve_mesh, residual_mesh, stress_mesh, mesh, full_displacement[step_index]
                )
                trajectory_frame = render_trajectory_frame(strain_path, step_index, (540, 292))
                stress_frame = render_stress_frame(
                    hprom_stress,
                    fom_stress,
                    direct_hprom_stress,
                    step_index,
                    (540, 292),
                )
                composite = annotate_and_compose(
                    rve_frame,
                    trajectory_frame,
                    stress_frame,
                    performance,
                    int(step_index),
                    full_displacement.shape[0],
                )
                writer.append_data(composite)
                last_composite = composite
                print(f"Rendered frame {frame_number}/{total_frames} (step {step_index}).", flush=True)

            if last_composite is not None:
                for hold_number in range(args.end_hold_frames):
                    writer.append_data(last_composite)
                    print(
                        f"Rendered end hold {hold_number + 1}/{args.end_hold_frames} "
                        f"({len(frame_indices) + hold_number + 1}/{total_frames}).",
                        flush=True,
                    )
    finally:
        plotter.close()

    metadata = {
        "model": "HPROM-ANN",
        "result_dir": str(args.result_dir),
        "ecm_file": str(args.ecm_file),
        "residual_support_size": int(residual_elements.size),
        "stress_support_size": int(stress_elements.size),
        "rve_state_source": "Stage 10 HPROM-ANN q history reconstructed with the Stage 7 ANN decoder",
        "fom_time_s": performance.fom_time_s,
        "hprom_ann_rve_kernel_s": performance.hprom_time_s,
        "direct_hprom_ann_rve_kernel_s": performance.direct_time_s,
        "hprom_ann_speedup_vs_fom": performance.hprom_speedup,
        "direct_hprom_ann_speedup_vs_fom": performance.direct_speedup,
        "hprom_ann_relative_stress_error": performance.hprom_stress_error,
        "direct_hprom_ann_relative_stress_error": performance.direct_stress_error,
        "main_frames": int(len(frame_indices)),
        "end_hold_frames": int(args.end_hold_frames),
        "fps": float(args.fps),
        "duration_seconds": gif_duration_seconds(output_file),
        "output_size": [1280, 720],
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_file, metadata_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the WCCM HPROM-ANN three-panel animation.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--ecm-file", type=Path, default=DEFAULT_ECM_FILE)
    parser.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--pod-dir", type=Path, default=DEFAULT_POD_DIR)
    parser.add_argument("--mdpa-file", type=Path, default=DEFAULT_MDPA_FILE)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-name", default="hprom_ann_three_panel_response.gif")
    parser.add_argument("--main-frames", type=int, default=100)
    parser.add_argument("--end-hold-frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file, metadata_file = build_animation(args)
    print(f"Saved animation: {output_file}")
    print(f"Saved metadata: {metadata_file}")


if __name__ == "__main__":
    main()
