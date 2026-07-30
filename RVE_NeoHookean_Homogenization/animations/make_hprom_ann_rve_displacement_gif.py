#!/usr/bin/env python3
"""Create a compact displacement-only HPROM-ANN RVE animation."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyvista as pv

from make_hprom_ann_three_panel_animation import (
    DEFAULT_ANN_DIR,
    DEFAULT_ECM_FILE,
    DEFAULT_MDPA_FILE,
    DEFAULT_POD_DIR,
    DEFAULT_RESULT_DIR,
    MeshData,
    RveStats,
    cell_blocks,
    compute_rve_stats,
    full_to_nodal_displacement,
    parse_mdpa_quadratic_triangles,
    reconstruct_hprom_displacements,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "hprom_ann_rve_displacement.gif"
FRAME_COUNT = 72
FPS = 12
WINDOW_SIZE = (520, 520)


def configure_plotter(
    mesh: MeshData,
    initial_displacement: np.ndarray,
    residual_elements: np.ndarray,
    stress_elements: np.ndarray,
    stats: RveStats,
) -> tuple[pv.Plotter, pv.UnstructuredGrid, pv.UnstructuredGrid, pv.UnstructuredGrid]:
    nodal_displacement = full_to_nodal_displacement(initial_displacement, mesh)
    points = mesh.points.copy()
    points[:, :2] += nodal_displacement
    magnitude = np.linalg.norm(nodal_displacement, axis=1)

    deformed_mesh = pv.UnstructuredGrid(mesh.cells, mesh.cell_types, points)
    deformed_mesh.point_data["displacement"] = magnitude
    residual_cells, residual_types = cell_blocks(mesh, residual_elements)
    stress_cells, stress_types = cell_blocks(mesh, stress_elements)
    residual_points = points.copy()
    stress_points = points.copy()
    residual_points[:, 2] = 1.0e-4
    stress_points[:, 2] = 2.0e-4
    residual_mesh = pv.UnstructuredGrid(residual_cells, residual_types, residual_points)
    stress_mesh = pv.UnstructuredGrid(stress_cells, stress_types, stress_points)

    span = np.maximum(stats.xy_max - stats.xy_min, 1.0e-12)
    center = 0.5 * (stats.xy_min + stats.xy_max)
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background("white")
    plotter.add_mesh(
        deformed_mesh,
        scalars="displacement",
        cmap="coolwarm",
        clim=(0.0, stats.max_displacement),
        opacity=0.82,
        show_edges=True,
        edge_color="#35414D",
        line_width=0.30,
        show_scalar_bar=False,
    )
    plotter.add_mesh(
        residual_mesh,
        color="#F59E0B",
        opacity=1.0,
        show_edges=False,
        lighting=False,
    )
    plotter.add_mesh(
        stress_mesh,
        color="#1976D2",
        opacity=1.0,
        show_edges=False,
        lighting=False,
    )
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera.focal_point = (float(center[0]), float(center[1]), 0.0)
    plotter.camera.position = (float(center[0]), float(center[1]), 10.0)
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.parallel_scale = 0.56 * max(span[0], span[1])
    return plotter, deformed_mesh, residual_mesh, stress_mesh


def render_frame(
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
    deformed_mesh.points[:, :] = points
    deformed_mesh.GetPoints().Modified()
    deformed_mesh.Modified()
    for displayed_mesh, z_offset in ((residual_mesh, 1.0e-4), (stress_mesh, 2.0e-4)):
        displayed_mesh.points[:, :] = points
        displayed_mesh.points[:, 2] = z_offset
        displayed_mesh.GetPoints().Modified()
        displayed_mesh.Modified()
    magnitude = np.linalg.norm(nodal_displacement, axis=1)
    deformed_mesh.point_data["displacement"][:] = magnitude
    plotter.update_scalars(magnitude, mesh=deformed_mesh, render=False)
    plotter.render()
    return np.asarray(plotter.screenshot(return_img=True))[..., :3]


def main() -> None:
    pv.OFF_SCREEN = True
    mesh = parse_mdpa_quadratic_triangles(DEFAULT_MDPA_FILE)
    displacements, _, _ = reconstruct_hprom_displacements(
        mesh,
        DEFAULT_RESULT_DIR,
        DEFAULT_ANN_DIR,
        DEFAULT_POD_DIR,
    )
    stats = compute_rve_stats(mesh, displacements)
    ecm = np.load(DEFAULT_ECM_FILE, allow_pickle=True)
    residual_elements = np.asarray(ecm["Z_res"], dtype=int)
    stress_elements = np.asarray(ecm["Z_sig"], dtype=int)
    frame_indices = np.unique(np.linspace(0, displacements.shape[0] - 1, FRAME_COUNT, dtype=int))
    plotter, deformed_mesh, residual_mesh, stress_mesh = configure_plotter(
        mesh,
        displacements[frame_indices[0]],
        residual_elements,
        stress_elements,
        stats,
    )
    duration_ms = int(round(1000.0 / FPS))
    try:
        with imageio.get_writer(OUTPUT_FILE, mode="I", duration=duration_ms, loop=0) as writer:
            for step_index in frame_indices:
                writer.append_data(
                    render_frame(
                        plotter,
                        deformed_mesh,
                        residual_mesh,
                        stress_mesh,
                        mesh,
                        displacements[step_index],
                    )
                )
    finally:
        plotter.close()


if __name__ == "__main__":
    main()
