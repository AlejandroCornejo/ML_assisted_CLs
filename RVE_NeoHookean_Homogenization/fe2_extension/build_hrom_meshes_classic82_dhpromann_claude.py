#!/usr/bin/env python3
"""Build the D-HPROM-ANN-only HROM mesh + ecm_weights_all.npz for the
82-point classic ECM reaction-force rule (classic (non-randomized) SVD,
RSVD_TOL=1e-6 -> rank 82; near machine-precision held-out accuracy:
Table-6=0.0000%, Cook=0.0004%). D-HPROM-ANN only -- this rule was not
built for HPROM-ANN (which also needs Z_res; not requested here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("core", "prom/ann", "hprom/ann"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

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
from build_hrom_meshes_classic37_claude import build_ecm_npz_fixed_classic  # noqa: E402

CLASSIC_ECM_NPZ = HERE / "classic_ecm_reaction_force_result_n82_claude.npz"
ORIGINAL_ECM_NPZ = ROOT / "hprom" / "ann" / "maw_dynamic" / "ecm_weights_all.npz"
OUT_DIR_DHPROMANN = HERE / "maw_dynamic_reaction_force_classic82_dhpromann"


def main():
    original = dict(np.load(ORIGINAL_ECM_NPZ, allow_pickle=True))
    classic = np.load(CLASSIC_ECM_NPZ)

    Z_support = np.asarray(classic["Z_elements"], dtype=np.int64).reshape(-1)
    w_fixed = np.asarray(classic["w_sel"], dtype=float).reshape(-1)
    print(f"[build-classic82] D-HPROM-ANN mesh needs {Z_support.size} elements")

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
    print(f"[build-classic82] origin mesh: {len(origin_elements)} elements, {origin_mp.NumberOfNodes()} nodes")

    meta_dhpromann = build_one_hrom_mesh(
        origin_mp, origin_elements, full_nodeid_to_eqxy, Z_support, OUT_DIR_DHPROMANN, "D-HPROM-ANN-classic82"
    )
    build_ecm_npz_fixed_classic(original, Z_support, w_fixed, meta_dhpromann, OUT_DIR_DHPROMANN)

    sim.Finalize()
    print("[build-classic82] done.")


if __name__ == "__main__":
    main()
