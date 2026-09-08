#!/usr/bin/env python3
"""Confirm the plain square panel gives exactly-uniform, shear-free macro
strain (E12=0 to numerical precision, not just small) across a delta range
pushing E11=E22 up near the RVE's own trained upper bound (2.0), unlike
every cruciform variant tried this session. Same monkey-patch-the-builder
reuse pattern as test_cruciform_filleted_shear_claude.py -- zero changes to
the proven Newton-loop/line-search code in run_newton_fe2_cruciform itself."""
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
import KratosMultiphysics as KM  # noqa: E402

import run_cruciform_fe2_claude as rc  # noqa: E402
from build_square_panel_mesh_claude import build_square_panel_mesh  # noqa: E402

L_BODY = 12.0
N_BODY = 8


def build_square_panel_model_part(n_body, n_arm_len, L_body=L_BODY, arm_width_fraction=None,
                                   L_arm=None, fix_tips=True):
    """Signature-compatible with build_cruciform_model_part (n_arm_len/
    arm_width_fraction/L_arm accepted but unused -- there are no arms)."""
    coords, tris, tip_nodes, center_node = build_square_panel_mesh(n_body=N_BODY, L_body=L_body)
    model = KM.Model()
    mp = model.CreateModelPart("Structure")
    mp.SetBufferSize(1)
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.AddNodalSolutionStepVariable(KM.REACTION)

    for i, (x, y) in enumerate(coords):
        mp.CreateNewNode(i + 1, float(x), float(y), 0.0)

    prop = mp.GetProperties()[1]
    prop.SetValue(KM.YOUNG_MODULUS, 1.0)
    prop.SetValue(KM.POISSON_RATIO, 0.3)
    prop.SetValue(KM.THICKNESS, 1.0)

    for e, conn in enumerate(tris):
        node_ids = [int(c) + 1 for c in conn]
        mp.CreateNewElement("TotalLagrangianElement2D6N", e + 1, node_ids, prop)

    KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, mp)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, mp)

    mp.GetNode(center_node + 1).Fix(KM.DISPLACEMENT_X)
    mp.GetNode(center_node + 1).Fix(KM.DISPLACEMENT_Y)
    if fix_tips:
        for nid in tip_nodes["px"] + tip_nodes["mx"]:
            mp.GetNode(nid + 1).Fix(KM.DISPLACEMENT_X)
        for nid in tip_nodes["py"] + tip_nodes["my"]:
            mp.GetNode(nid + 1).Fix(KM.DISPLACEMENT_Y)

    return mp, coords, tris, tip_nodes, center_node


def main():
    which = "pann_certified"
    deltas = [1.2, 3.6, 7.2]

    original_builder = rc.build_cruciform_model_part
    rc.build_cruciform_model_part = build_square_panel_model_part
    try:
        for delta in deltas:
            F11_expected = 1.0 + delta / (L_BODY / 2.0)
            E11_expected = 0.5 * (F11_expected ** 2 - 1.0)
            t0 = time.time()
            res = rc.run_newton_fe2_cruciform(
                which, n_body=N_BODY, n_arm_len=0, n_steps=20,
                delta_x_final=delta, delta_y_final=delta,
                verbose=False, use_line_search=True, save_npz=False,
            )
            e11_lo, e11_hi = res["e11_range"]
            e22_lo, e22_hi = res["e22_range"]
            g12_lo, g12_hi = res["g12_range"]
            peak_g12 = max(abs(g12_lo), abs(g12_hi))
            print(f"[SQUARE PANEL] delta={delta:.1f}  E11_expected~={E11_expected:.4f}  "
                  f"E11=[{e11_lo:+.6f},{e11_hi:+.6f}]  E22=[{e22_lo:+.6f},{e22_hi:+.6f}]  "
                  f"g12=[{g12_lo:+.2e},{g12_hi:+.2e}]  peak|g12|={peak_g12:.2e}  "
                  f"fully_converged={res['fully_converged']}  ({time.time() - t0:.1f}s)", flush=True)
    finally:
        rc.build_cruciform_model_part = original_builder

    print("SQUARE_PANEL_SHEAR_TEST_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
