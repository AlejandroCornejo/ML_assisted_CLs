#!/usr/bin/env python3
"""Build the data-backed image used in the constitutive overview.

The overview must show an actual RVE response, not a decorative surrogate for
one.  This script reconstructs the physical nodal displacement from a stored
periodic FOM snapshot and plots the deformed quadratic mesh.  The selected
test state is deliberately fixed so that rebuilding the manuscript is
deterministic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-coupon-overview")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGURES = HERE / "figures"
TEST_STATE = 37


def load_corner_connectivity(path: Path, node_ids: np.ndarray) -> np.ndarray:
    """Read the three corner nodes of each Triangle2D6 geometry."""
    triangles: list[list[int]] = []
    inside = False
    for line in path.read_text().splitlines():
        if line.startswith("Begin Geometries Triangle2D6"):
            inside = True
            continue
        if inside and line.startswith("End Geometries"):
            break
        if inside:
            fields = line.split()
            if len(fields) == 7:
                triangles.append([int(value) for value in fields[1:4]])
    index = {int(node_id): i for i, node_id in enumerate(node_ids)}
    return np.asarray([[index[node_id] for node_id in row] for row in triangles])


def main() -> None:
    sys.path[:0] = [str(PROJECT / "00_rve"), str(PROJECT)]
    import config as cfg  # noqa: E402
    from periodic_fom import PeriodicRVE  # noqa: E402

    rve = PeriodicRVE(PROJECT / "03_data/rve_mesh",
                      cell_area=cfg.CELL_SIDE**2)
    strains = np.load(PROJECT / "02_sampling/eval_sets.npz")["test"]
    snapshots = np.load(PROJECT / "03_data/eval_snapshots.npz")["U_test"]
    strain = strains[TEST_STATE]
    independent_displacement = snapshots[TEST_STATE]
    displacement_dofs = np.asarray(
        rve.T @ independent_displacement + rve._g(strain)
    )

    nodes = list(rve._mp.Nodes)
    node_ids = np.asarray([node.Id for node in nodes], dtype=int)
    reference = np.asarray([[node.X0, node.Y0] for node in nodes])
    displacement = np.column_stack(
        (displacement_dofs[rve._eq_map[:, 0]],
         displacement_dofs[rve._eq_map[:, 1]])
    )
    deformed = reference + displacement
    triangles = load_corner_connectivity(
        PROJECT / "03_data/rve_mesh.mdpa", node_ids
    )
    triangulation = mtri.Triangulation(
        deformed[:, 0], deformed[:, 1], triangles
    )

    magnitude = np.linalg.norm(displacement, axis=1)
    fig, ax = plt.subplots(figsize=(2.2, 2.0))
    # ParaView's familiar Cool-to-Warm convention: low displacement in blue,
    # high displacement in red.  No numerical colorbar is used in the method
    # overview because this state is illustrative rather than a reported test.
    ax.tripcolor(triangulation, magnitude, shading="gouraud", cmap="coolwarm")
    ax.triplot(triangulation, color="#334155", linewidth=0.18, alpha=0.42)
    ax.set_aspect("equal")
    ax.margins(0.025)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "rve_deformed_overview.pdf",
                transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIGURES / "rve_deformed_overview.png", dpi=300,
                transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # The feature panel uses the same discretized cell in its undeformed
    # configuration.  Direction pairs are overlaid in TikZ so that every
    # sensor visibly acts on the RVE rather than on an abstract card.
    reference_triangulation = mtri.Triangulation(
        reference[:, 0], reference[:, 1], triangles
    )
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.tripcolor(reference_triangulation, np.zeros(len(reference)),
                 shading="flat",
                 cmap=LinearSegmentedColormap.from_list(
                     "reference_cell", ("#E8F0F6", "#E8F0F6")
                 ), vmin=0.0, vmax=1.0)
    ax.triplot(reference_triangulation, color="#4F6A80",
               linewidth=0.16, alpha=0.42)
    ax.set_aspect("equal")
    ax.margins(0.025)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(FIGURES / "rve_reference_overview.pdf",
                transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIGURES / "rve_reference_overview.png", dpi=300,
                transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    main()
