#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trajectory gallery -- final (most-deformed) RVE configuration for each of the
10 Stage-1 FOM training trajectories.

For every ``trajectory_N`` under ``trajectories/stage_1_training_set_fom/``,
this script takes the LAST step of the full nodal displacement history
(``trajectory_N_U.npy``, shape ``(n_steps, 4244)``, stored in Kratos's own
internal DOF-equation-ID order) together with the matching macro-strain
(``trajectory_N_applied_strain.npy``), reconstructs the per-node ``(ux, uy)``
displacement field using the SAME DOF machinery the FOM solver itself uses
(``core/fom_solver_rve.py``: ``SetUpDofEquationIdsAndDisplacementAdaptor`` /
``SetDisplacementFromEquationVector``), and renders the deformed mesh colored
by nodal displacement magnitude on a shared color scale.

Before any figure is produced, the reconstructed displacement field is
sanity-checked on the outer ("dirichlet") boundary against the closed-form
affine boundary condition

    u_d(X, Y) = (F - I) (X - Xc, Y - Yc),      F = sqrtm(I + 2E)

where ``F`` is the unique symmetric positive-definite square root of
``C = I + 2E`` (computed here independently via eigendecomposition -- NOT by
calling into ``fom_solver_rve.DeformationGradientFromGreenLagrange2D`` -- so
this is a genuine external check on the eq_map / DOF-reconstruction
pipeline). If any trajectory fails this check, the script raises instead of
silently producing a figure built on a wrong DOF ordering.

The Kratos-loading boilerplate mirrors the proven working pattern used in
``pann/direct_energy/reaction_force_direct_stress.py`` and
``studies/fom_tangent_stability_test/run_fom_energy_hessian_audit.py``.

Usage:
    python3 make_trajectory_gallery_claude.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
CORE_DIR = REPO_ROOT / "core"
TRAJ_DIR = REPO_ROOT / "trajectories" / "stage_1_training_set_fom"
OUT_PNG = HERE / "trajectory_gallery_claude.png"

N_TRAJECTORIES = 10
BC_TOL = 1.0e-8  # tight tolerance for the closed-form Dirichlet sanity check

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from plot_style_utils import apply_latex_plot_style  # noqa: E402

apply_latex_plot_style()

import KratosMultiphysics as KM  # noqa: E402
import fom_solver_rve as fom  # noqa: E402


def make_parameters() -> "KM.Parameters":
    """Build ProjectParameters with absolute mesh/material paths (cwd-independent).

    Mirrors ``pann/direct_energy/reaction_force_direct_stress.py``'s and
    ``studies/fom_tangent_stability_test/run_fom_energy_hessian_audit.py``'s
    ``make_parameters``, so this script can be run from any directory. Does
    NOT modify ``core/fom_solver_rve.py``.
    """
    with open(CORE_DIR / "ProjectParameters.json", encoding="utf-8") as f:
        config = json.load(f)

    config["output_processes"] = {"gid_output": [], "vtk_output": []}
    config["solver_settings"]["echo_level"] = 0
    parameters = KM.Parameters(json.dumps(config))

    mesh_base = str(CORE_DIR / "rve_geometry")
    fom.SetInputMeshFilename(parameters, mesh_base)
    material_parts = fom.DetectMaterialSubModelParts(str(CORE_DIR / "rve_geometry.mdpa"))
    parameters = fom.ConfigureElementModelerForMaterialParts(parameters, material_parts)
    fom.SetMaterialsFilename(parameters, str(CORE_DIR / "StructuralMaterials.json"))
    return parameters


def symmetric_psd_sqrt_2x2(mat: np.ndarray) -> np.ndarray:
    """Eigendecomposition-based symmetric PSD square root of a 2x2 SPD matrix."""
    eigvals, eigvecs = np.linalg.eigh(mat)
    if np.min(eigvals) <= 0.0:
        raise RuntimeError(f"Matrix is not positive definite: eigenvalues={eigvals}")
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def deformation_gradient_from_green_lagrange(e_voigt: np.ndarray) -> np.ndarray:
    """F = sqrtm(C), with C = I + 2E and E = [[E11, g12/2], [g12/2, E22]].

    This is deliberately independent of
    ``fom_solver_rve.DeformationGradientFromGreenLagrange2D``: it exists only
    to provide an external check on the DOF-reconstruction pipeline below,
    per the task's verification requirement.
    """
    e11, e22, g12 = (float(v) for v in np.asarray(e_voigt, dtype=float).reshape(3))
    E = np.array([[e11, 0.5 * g12], [0.5 * g12, e22]], dtype=float)
    C = np.eye(2, dtype=float) + 2.0 * E
    return symmetric_psd_sqrt_2x2(C)


class KratosMesh:
    """Loads ``core/rve_geometry.mdpa`` through KratosMultiphysics and exposes
    the DOF equation map, reference nodal coordinates, element connectivity,
    and Dirichlet-boundary node indices needed to turn a raw (4244,)
    equation-ordered displacement vector into a plottable deformed mesh.
    """

    def __init__(self):
        parameters = make_parameters()
        self.model = KM.Model()
        self.sim = fom.RVEHomogenizationDatasetGenerator(self.model, parameters)
        self.sim.Initialize()
        self.mp = self.sim._GetSolver().GetComputingModelPart()

        self.n_dof, self.eq_map, self.ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(self.mp)

        n_nodes = self.mp.NumberOfNodes()
        node_ids = np.empty(n_nodes, dtype=int)
        X0 = np.empty(n_nodes, dtype=float)
        Y0 = np.empty(n_nodes, dtype=float)
        for i, node in enumerate(self.mp.Nodes):
            node_ids[i] = node.Id
            X0[i] = node.X0
            Y0[i] = node.Y0
        self.X0 = X0
        self.Y0 = Y0
        id_to_local = {int(nid): i for i, nid in enumerate(node_ids)}

        # Element connectivity, mapped from Kratos node IDs to local indices
        # in the same order as mp.Nodes (i.e. matching eq_map's rows).
        # Kratos Triangle2D6N local node ordering is [v0, v1, v2, m(0,1),
        # m(1,2), m(0,2)] (verified empirically below against reference
        # coordinates: mid-side nodes not on the curved hole boundary must
        # coincide with the straight-line midpoint of their two end vertices).
        elements = list(self.mp.Elements)
        connectivity = np.array(
            [[id_to_local[node.Id] for node in elem.GetGeometry()] for elem in elements],
            dtype=int,
        )
        self.connectivity = connectivity
        self._verify_tri6_node_ordering()

        dirichlet_mp = self.mp.GetSubModelPart("dirichlet")
        self.dirichlet_local = np.array(
            [id_to_local[node.Id] for node in dirichlet_mp.Nodes], dtype=int
        )

        self.sim._InitializeDomainCenterIfNeeded(self.mp)
        self.x0c = float(self.sim._x0c)
        self.y0c = float(self.sim._y0c)

        # Sub-triangulation for smooth Gouraud-shaded rendering of the 6-node
        # (quadratic) triangles: split each element into 4 linear
        # sub-triangles using all 6 real mesh nodes (exact node positions and
        # displacements -- no geometric interpolation beyond what Kratos
        # itself represents with a quadratic triangle).
        v0, v1, v2, m01, m12, m02 = connectivity.T
        self.sub_triangles = np.concatenate(
            [
                np.stack([v0, m01, m02], axis=1),
                np.stack([m01, v1, m12], axis=1),
                np.stack([m02, m12, v2], axis=1),
                np.stack([m01, m12, m02], axis=1),
            ],
            axis=0,
        )

    def _verify_tri6_node_ordering(self, tol: float = 1.0e-6) -> None:
        """Confirm the assumed local connectivity ordering [v0,v1,v2,m01,m12,m02]
        by checking that edges NOT on the curved hole boundary have their
        mid-side node exactly at the straight-line midpoint of its vertices.
        """
        v0, v1, v2, m01, m12, m02 = self.connectivity.T
        mid12_straight = 0.5 * np.stack([self.X0[v1] + self.X0[v2], self.Y0[v1] + self.Y0[v2]], axis=1)
        mid02_straight = 0.5 * np.stack([self.X0[v0] + self.X0[v2], self.Y0[v0] + self.Y0[v2]], axis=1)
        p12 = np.stack([self.X0[m12], self.Y0[m12]], axis=1)
        p02 = np.stack([self.X0[m02], self.Y0[m02]], axis=1)
        err12 = float(np.max(np.linalg.norm(p12 - mid12_straight, axis=1)))
        err02 = float(np.max(np.linalg.norm(p02 - mid02_straight, axis=1)))
        if err12 > tol or err02 > tol:
            raise RuntimeError(
                "Assumed Triangle2D6N local node ordering [v0,v1,v2,m(0,1),m(1,2),"
                f"m(0,2)] failed a geometric sanity check (err12={err12:.3e}, "
                f"err02={err02:.3e}); refusing to build a mesh visualization on an "
                "unverified connectivity assumption."
            )

    def displacement_field(self, u_eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (ux, uy) per node, in the same mp.Nodes order as X0/Y0."""
        fom.SetDisplacementFromEquationVector(u_eq, self.eq_map, self.ta)
        data = np.asarray(self.ta.data, dtype=float)
        return data[:, 0].copy(), data[:, 1].copy()

    def close(self) -> None:
        self.sim.Finalize()


def verify_dirichlet_boundary_condition(
    mesh: KratosMesh,
    trajectory_index: int,
    e_last: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
) -> tuple[float, float]:
    """Check the reconstructed nodal displacement on the outer ('dirichlet')
    boundary against the closed-form affine map u_d = (F-I)(X-Xc, Y-Yc).

    Returns (max_abs_error, max_relative_error) and RAISES if the tolerance
    is not met -- this is the guard against shipping a figure built on a
    wrong eq_map / DOF ordering assumption.
    """
    idx = mesh.dirichlet_local
    Xr = mesh.X0[idx] - mesh.x0c
    Yr = mesh.Y0[idx] - mesh.y0c

    F = deformation_gradient_from_green_lagrange(e_last)
    ux_pred = (F[0, 0] - 1.0) * Xr + F[0, 1] * Yr
    uy_pred = F[1, 0] * Xr + (F[1, 1] - 1.0) * Yr

    max_err = float(np.max(np.abs(np.concatenate([ux[idx] - ux_pred, uy[idx] - uy_pred]))))
    scale = float(max(np.max(np.abs(ux_pred)), np.max(np.abs(uy_pred)), 1.0e-300))
    rel_err = max_err / scale

    print(
        f"[verify] trajectory {trajectory_index}: {idx.size} dirichlet nodes, "
        f"e_last=[{e_last[0]:.6g}, {e_last[1]:.6g}, {e_last[2]:.6g}], "
        f"max|error|={max_err:.3e}, relative={rel_err:.3e}"
    )
    if max_err > BC_TOL:
        raise RuntimeError(
            f"Trajectory {trajectory_index}: reconstructed displacement does not "
            f"match the closed-form Dirichlet boundary condition "
            f"(max|error|={max_err:.3e} > tol={BC_TOL:.1e}). The eq_map / DOF "
            f"ordering assumption is likely wrong; refusing to plot."
        )
    return max_err, rel_err


def load_trajectory_final_state(trajectory_index: int) -> tuple[np.ndarray, np.ndarray]:
    root = TRAJ_DIR / f"trajectory_{trajectory_index}"
    e_hist = np.load(root / f"trajectory_{trajectory_index}_applied_strain.npy")
    u_hist = np.load(root / f"trajectory_{trajectory_index}_U.npy")
    return e_hist[-1].copy(), u_hist[-1].copy()


def format_strain_title(trajectory_index: int, e_last: np.ndarray) -> str:
    e11, e22, g12 = (float(v) for v in e_last)
    return (
        rf"Traj. {trajectory_index}: "
        rf"$E_{{11}}{{=}}{e11:.3g},\ E_{{22}}{{=}}{e22:.3g},\ "
        rf"\gamma_{{12}}{{=}}{g12:.3g}$"
    )


def main() -> None:
    mesh = KratosMesh()
    try:
        print(
            f"[mesh] {mesh.mp.NumberOfNodes()} nodes, {len(mesh.connectivity)} elements, "
            f"n_dof={mesh.n_dof}, domain center=({mesh.x0c:.6g}, {mesh.y0c:.6g}), "
            f"{mesh.dirichlet_local.size} dirichlet-boundary nodes"
        )

        states = []
        verification_summary = []
        global_max_mag = 0.0
        for n in range(1, N_TRAJECTORIES + 1):
            e_last, u_last = load_trajectory_final_state(n)
            ux, uy = mesh.displacement_field(u_last)
            max_err, rel_err = verify_dirichlet_boundary_condition(mesh, n, e_last, ux, uy)
            verification_summary.append((n, max_err, rel_err))

            mag = np.sqrt(ux**2 + uy**2)
            global_max_mag = max(global_max_mag, float(np.max(mag)))
            states.append((n, e_last, ux.copy(), uy.copy(), mag))

        print(
            "[verify] SUMMARY: worst max|error| over all trajectories = "
            f"{max(m for _, m, _ in verification_summary):.3e} "
            f"(tolerance {BC_TOL:.1e})"
        )

        vmin, vmax = 0.0, global_max_mag
        print(f"[gallery] shared color scale: |u| in [{vmin:.6g}, {vmax:.6g}]")

        fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
        im = None
        for (n, e_last, ux, uy, mag), ax in zip(states, axes.flat):
            xdef = mesh.X0 + ux
            ydef = mesh.Y0 + uy
            triang = mtri.Triangulation(xdef, ydef, triangles=mesh.sub_triangles)
            im = ax.tripcolor(
                triang, mag, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax
            )
            # adjustable="datalim" keeps every panel's physical box the same
            # uniform grid size/position (only the data window pads to
            # preserve a true 1:1 aspect); the default "box" instead shrinks
            # each panel's box individually, which staggers panels of
            # different shapes and collides their titles.
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            ax.set_title(format_strain_title(n, e_last), fontsize=12, pad=6)

        fig.suptitle(
            "Training trajectories: final deformed configurations",
            fontsize=15,
        )
        fig.subplots_adjust(
            left=0.02, right=0.90, top=0.87, bottom=0.03, wspace=0.08, hspace=0.30
        )
        cbar_ax = fig.add_axes((0.92, 0.12, 0.015, 0.72))
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r"$|\mathbf{u}|$")

        fig.savefig(OUT_PNG)
        print(f"[gallery] Saved figure to {OUT_PNG}")
    finally:
        mesh.close()


if __name__ == "__main__":
    main()
