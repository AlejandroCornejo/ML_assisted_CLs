#!/usr/bin/env python3
"""Render the RVE nodes on which the affine macro deformation is prescribed.

The node set is read directly from ``rve_geometry.mdpa``'s ``dirichlet``
submodel part.  In ``ProjectParameters.json`` that submodel part constrains
all DISPLACEMENT components; Stage 1 and Stage 10 then prescribe the
finite-deformation affine displacement on its in-plane DOFs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MDPA_FILE = PROJECT_DIR / "rve_geometry.mdpa"
POD_DIR = PROJECT_DIR / "stage_2_pod_rve"
OUTPUT_FILE = SCRIPT_DIR / "rve_macro_deformation_dirichlet_nodes.png"


@dataclass(frozen=True)
class MdpaMesh:
    points: np.ndarray
    cells: np.ndarray
    cell_types: np.ndarray
    dirichlet_ids: np.ndarray


def parse_rve_mesh(mdpa_file: Path) -> MdpaMesh:
    """Read the quadratic triangles and exact Dirichlet node IDs from MDPA."""
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: list[list[int]] = []
    dirichlet_ids: list[int] = []
    in_nodes = False
    in_elements = False
    in_dirichlet = False
    in_dirichlet_nodes = False

    with mdpa_file.open("r", encoding="utf-8") as mesh_file:
        for raw_line in mesh_file:
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
            if line.startswith("Begin SubModelPart dirichlet"):
                in_dirichlet = True
                continue
            if in_dirichlet and line.startswith("End SubModelPart"):
                in_dirichlet = False
                in_dirichlet_nodes = False
                continue
            if in_dirichlet and line.startswith("Begin SubModelPartNodes"):
                in_dirichlet_nodes = True
                continue
            if in_dirichlet and line.startswith("End SubModelPartNodes"):
                in_dirichlet_nodes = False
                continue

            if in_nodes:
                parts = line.split()
                nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
                continue
            if in_elements:
                parts = line.split()
                elements.append([int(value) for value in parts[1:7]])
                continue
            if in_dirichlet_nodes:
                dirichlet_ids.append(int(line.split()[0]))

    if not nodes or not elements or not dirichlet_ids:
        raise RuntimeError("The mesh, quadratic triangles, or Dirichlet node set was not found.")

    sorted_ids = sorted(nodes)
    if sorted_ids != list(range(1, len(sorted_ids) + 1)):
        raise RuntimeError("Expected consecutive node IDs starting at one.")

    points = np.asarray([nodes[node_id] for node_id in sorted_ids], dtype=float)
    index_by_id = {node_id: index for index, node_id in enumerate(sorted_ids)}
    cells = np.concatenate(
        [np.asarray([6] + [index_by_id[node_id] for node_id in element], dtype=np.int64)
         for element in elements]
    )
    cell_types = np.full(len(elements), pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8)
    return MdpaMesh(
        points=points,
        cells=cells,
        cell_types=cell_types,
        dirichlet_ids=np.asarray(sorted(set(dirichlet_ids)), dtype=np.int64),
    )


def validate_dirichlet_count(dirichlet_ids: np.ndarray) -> None:
    """Check against the saved POD partition built from the same FOM model."""
    dof_file = POD_DIR / "dirichlet_dofs.npy"
    if not dof_file.exists():
        return
    fixed_dofs = np.load(dof_file)
    expected_dofs = 2 * len(dirichlet_ids)
    if fixed_dofs.size != expected_dofs:
        raise RuntimeError(
            f"MDPA gives {len(dirichlet_ids)} boundary nodes ({expected_dofs} in-plane DOFs), "
            f"but {dof_file} contains {fixed_dofs.size} fixed DOFs."
        )


def render(mesh_data: MdpaMesh, output_file: Path) -> None:
    mesh = pv.UnstructuredGrid(mesh_data.cells, mesh_data.cell_types, mesh_data.points)
    boundary_indices = mesh_data.dirichlet_ids - 1
    boundary_points = mesh_data.points[boundary_indices].copy()
    boundary_points[:, 2] = 0.04
    boundary = pv.PolyData(boundary_points)

    xy_min = mesh_data.points[:, :2].min(axis=0)
    xy_max = mesh_data.points[:, :2].max(axis=0)
    center = 0.5 * (xy_min + xy_max)
    span = xy_max - xy_min

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1080))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        color="#edf1f4",
        show_edges=True,
        edge_color="#7b8790",
        line_width=0.65,
        opacity=1.0,
        lighting=False,
    )
    plotter.add_mesh(
        boundary,
        color="#d62828",
        point_size=15,
        render_points_as_spheres=True,
        lighting=False,
    )
    plotter.add_text(
        "RVE macro-deformation boundary",
        position="upper_left",
        font_size=25,
        color="#17212b",
        font="arial",
    )
    plotter.add_text(
        f"red: prescribed affine displacement, {len(mesh_data.dirichlet_ids)} nodes",
        position=(28, 72),
        font_size=15,
        color="#4c5965",
        font="arial",
    )
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera.focal_point = (float(center[0]), float(center[1]), 0.0)
    plotter.camera.position = (float(center[0]), float(center[1]), 10.0)
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.parallel_scale = 1.24 * float(max(span[0], span[1]))
    plotter.screenshot(str(output_file), transparent_background=False)
    plotter.close()


def main() -> None:
    mesh_data = parse_rve_mesh(MDPA_FILE)
    validate_dirichlet_count(mesh_data.dirichlet_ids)
    render(mesh_data, OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")
    print(f"Dirichlet nodes: {len(mesh_data.dirichlet_ids)}")


if __name__ == "__main__":
    main()
