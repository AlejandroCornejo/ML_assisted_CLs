#!/usr/bin/env python3
"""Force-controlled square panel with a COMBINED tension+shear traction:
a single constant, symmetric target nominal-stress tensor S=[[1,r],[r,1]]
(r=SHEAR_RATIO) applied via t=S.N on each of the 4 edges (consistent
Simpson-weighted nodal loads, reusing consistent_edge_force exactly as-is).
Self-equilibrated automatically (zero net force AND zero net torque, since
it is literally the boundary traction of a single constant symmetric stress
tensor) -- verified numerically below, not just asserted.

Reuses run_newton_fe2_cruciform_force UNCHANGED (all its proven Newton-loop/
line-search/stall-detection logic) via the same monkey-patch-the-builder
trick used for the filleted-cruciform and square-panel displacement tests:
swap build_cruciform_model_part for the square panel, and swap tip_force_unit
for this combined-traction pattern -- zero new solve logic.

Purpose: does adding a modest, trained-range shear component (r=0.1, i.e.
shear traction = 10% of the tension traction) alongside the tension ramp
shift Free's immediate-stall / Regression's limit-point thresholds found
under PURE equibiaxial force control earlier this session?"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.chdir(str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.insert(0, KRATOS_PATH)

import numpy as np

import run_cruciform_fe2_claude as rc  # noqa: E402
import run_cruciform_fe2_force_controlled_claude as rcf  # noqa: E402
from build_square_panel_mesh_claude import build_square_panel_mesh  # noqa: E402
from test_square_panel_shear_claude import build_square_panel_model_part  # noqa: E402

N_BODY = 8
L_BODY = 12.0
REF_FORCE_SQUARE = 5.150081381e9  # ICNN's own |reaction_px| at delta=1.2 on this same panel, verified this session
SHEAR_RATIO = 0.1  # shear traction = 10% of the tension traction; a modest, always-present, clearly-labeled fraction


def square_panel_force_unit(coords, edge_nodes, n_dof, eq_map, shear_ratio=SHEAR_RATIO):
    """t = S.N per edge, S=[[1,r],[r,1]] constant -- self-equilibrated by
    construction (boundary traction of a single constant symmetric stress
    tensor has zero net force and zero net torque)."""
    f = np.zeros((coords.shape[0], 2))
    f += rcf.consistent_edge_force(coords, edge_nodes["px"], direction=(1.0, shear_ratio))     # N=(+1,0)
    f += rcf.consistent_edge_force(coords, edge_nodes["mx"], direction=(-1.0, -shear_ratio))   # N=(-1,0)
    f += rcf.consistent_edge_force(coords, edge_nodes["py"], direction=(shear_ratio, 1.0))     # N=(0,+1)
    f += rcf.consistent_edge_force(coords, edge_nodes["my"], direction=(-shear_ratio, -1.0))   # N=(0,-1)
    f_eq = np.zeros(n_dof)
    np.add.at(f_eq, eq_map[:, 0], f[:, 0])
    np.add.at(f_eq, eq_map[:, 1], f[:, 1])
    return f_eq


def _verify_self_equilibrated():
    coords, tris, edge_nodes, center_node = build_square_panel_mesh(n_body=N_BODY, L_body=L_BODY)
    n_dof = coords.shape[0] * 2
    eq_map = np.arange(n_dof).reshape(-1, 2)
    f_eq = square_panel_force_unit(coords, edge_nodes, n_dof, eq_map)
    fx, fy = f_eq[0::2].sum(), f_eq[1::2].sum()
    torque = np.sum(coords[:, 0] * f_eq[1::2] - coords[:, 1] * f_eq[0::2])
    print(f"[self-equilibration check] net Fx={fx:.3e}  net Fy={fy:.3e}  net torque={torque:.3e} "
          f"(all should be ~0 to float precision)")
    assert abs(fx) < 1e-9 and abs(fy) < 1e-9 and abs(torque) < 1e-6


def build_square_panel_model_part_free(n_body, n_arm_len, L_body=L_BODY, arm_width_fraction=None,
                                        L_arm=None, fix_tips=False):
    return build_square_panel_model_part(n_body, n_arm_len, L_body=L_body, fix_tips=False)


def main():
    _verify_self_equilibrated()

    which_list = ["pann_regression", "pann_free", "pann_certified"]
    multiples = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    original_builder = rc.build_cruciform_model_part
    original_tip_force = rcf.tip_force_unit
    rc.build_cruciform_model_part = build_square_panel_model_part_free
    rcf.build_cruciform_model_part = build_square_panel_model_part_free
    rcf.tip_force_unit = lambda coords, edge_nodes, n_dof, eq_map: square_panel_force_unit(coords, edge_nodes, n_dof, eq_map)
    try:
        for which in which_list:
            print(f"\n=== {which}, square panel, combined tension+shear (r={SHEAR_RATIO}) force control ===", flush=True)
            for mult in multiples:
                t0 = time.time()
                try:
                    res = rcf.run_newton_fe2_cruciform_force(
                        which, n_body=N_BODY, n_arm_len=0, n_steps=20,
                        total_force_final=mult * REF_FORCE_SQUARE,
                        verbose=False, use_line_search=True, save_npz=False,
                    )
                    last = res["step_log"][-1]
                    print(f"  [{which}] mult={mult:5.1f}x  fully_converged={res['fully_converged']}  "
                          f"ever_diverged={res['ever_diverged']}  "
                          f"tip_ux={last['tip_ux_px_mean']:.4f}  tip_uy={last['tip_uy_py_mean']:.4f}  "
                          f"({time.time() - t0:.1f}s)", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{which}] mult={mult:5.1f}x  CRASHED after {time.time() - t0:.1f}s: "
                          f"{type(exc).__name__}: {exc}", flush=True)
    finally:
        rc.build_cruciform_model_part = original_builder
        rcf.build_cruciform_model_part = original_builder
        rcf.tip_force_unit = original_tip_force

    print("SQUARE_PANEL_COMBINED_FORCE_TEST_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
