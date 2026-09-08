#!/usr/bin/env python3
"""Build the 237-element reduced HROM mesh for the linear HPROM's Z_res
rule (linear_residual_z_res_rule_claude.npz), reusing build_one_hrom_mesh
from build_hrom_meshes_reaction_force_claude.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "core") not in sys.path:
    sys.path.insert(0, str(ROOT / "core"))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM  # noqa: E402

from fom_solver_rve import (  # noqa: E402
    setup_kratos_parameters,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
)
from build_hrom_meshes_reaction_force_claude import build_one_hrom_mesh  # noqa: E402

RULE_NPZ = HERE / "linear_residual_z_res_rule_claude.npz"
OUT_DIR = HERE / "linear_hprom_zres_hrom_mesh"


def main():
    rule = np.load(RULE_NPZ)
    Z_res = np.asarray(rule["Z_res"], dtype=np.int64).reshape(-1)
    w_res_full = np.asarray(rule["w_res_full"], dtype=float).reshape(-1)
    n_elem = int(rule["n_elem"])
    print(f"[build-linear-hrom-mesh] Z_res size={Z_res.size}")

    parameters = setup_kratos_parameters(str(HERE / "rve_geometry"))
    model_origin = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model_origin, parameters)
    sim.Initialize()
    origin_mp = sim._GetSolver().GetComputingModelPart()
    _, eq_map_full, _ = SetUpDofEquationIdsAndDisplacementAdaptor(origin_mp)
    full_nodeid_to_eqxy = {
        int(node.Id): (int(eq_map_full[i, 0]), int(eq_map_full[i, 1]))
        for i, node in enumerate(origin_mp.Nodes)
    }
    origin_elements = list(origin_mp.Elements)
    print(f"[build-linear-hrom-mesh] origin mesh: {len(origin_elements)} elements, {origin_mp.NumberOfNodes()} nodes")

    meta = build_one_hrom_mesh(origin_mp, origin_elements, full_nodeid_to_eqxy, Z_res, OUT_DIR, "linear-HPROM")

    out = dict(meta)
    out["Z_res"] = Z_res
    out["w_res_full"] = w_res_full
    out["w_res_hrom"] = w_res_full[meta["hrom_element_full_indices"]]
    out["n_elem"] = np.array([n_elem], dtype=np.int64)

    out_npz = OUT_DIR / "ecm_weights_all.npz"
    np.savez(out_npz, **out)
    print(f"[build-linear-hrom-mesh] saved {out_npz}")

    sim.Finalize()


if __name__ == "__main__":
    main()
