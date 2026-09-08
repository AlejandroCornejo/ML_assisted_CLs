#!/usr/bin/env python3
"""Measure peak corner shear on the filleted cruciform mesh (R=1.8, gmsh
unstructured Tri6, see gen_cruciform_filleted_mesh_claude.py) at the
established delta=1.2 protocol, and compare directly against the sharp-corner
baseline (peak|g12|=0.1999 at the same delta). Reuses run_newton_fe2_cruciform
UNCHANGED (all its proven Newton-loop/line-search logic) by temporarily
monkey-patching which model-part builder it calls -- zero risk of a new bug
in the solve itself, since the solve code is not touched, only which mesh it
receives."""
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
from gen_cruciform_filleted_mesh_claude import generate_filleted_cruciform_mesh  # noqa: E402

R_FILLET = 1.8


def build_cruciform_model_part_filleted(n_body, n_arm_len, L_body=12.0, arm_width_fraction=2.0 / 3.0,
                                         L_arm=8.0, fix_tips=True):
    """Signature-compatible with build_cruciform_model_part (n_body/n_arm_len
    accepted but unused -- the unstructured mesh's density is controlled by
    gen_cruciform_filleted_mesh_claude's own size_far/size_near_fillet)."""
    coords, tris, tip_nodes, center_node, _reentrant = generate_filleted_cruciform_mesh(
        L_body=L_body, arm_width_fraction=arm_width_fraction, L_arm=L_arm, R=R_FILLET,
    )
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
    deltas = [1.2, 2.4, 4.8]

    original_builder = rc.build_cruciform_model_part
    rc.build_cruciform_model_part = build_cruciform_model_part_filleted
    try:
        for delta in deltas:
            t0 = time.time()
            res = rc.run_newton_fe2_cruciform(
                which, n_body=6, n_arm_len=4, n_steps=20,
                delta_x_final=delta, delta_y_final=delta,
                verbose=False, use_line_search=True, save_npz=False,
            )
            e11_lo, e11_hi = res["e11_range"]
            g12_lo, g12_hi = res["g12_range"]
            peak_g12 = max(abs(g12_lo), abs(g12_hi))
            print(f"[FILLETED R={R_FILLET}] delta={delta:.1f}  E11=[{e11_lo:+.4f},{e11_hi:+.4f}]  "
                  f"g12=[{g12_lo:+.4f},{g12_hi:+.4f}]  peak|g12|={peak_g12:.4f}  "
                  f"fully_converged={res['fully_converged']}  ({time.time() - t0:.1f}s)", flush=True)
    finally:
        rc.build_cruciform_model_part = original_builder

    print("FILLETED_SHEAR_TEST_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
