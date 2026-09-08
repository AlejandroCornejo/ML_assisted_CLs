#!/usr/bin/env python3
"""Build the combined HROM mesh (Z_res union Z_sig) + ecm_weights_all.npz
for the Cook-compatible linear HPROM material law:
  - Z_res (237 points, this session's own linear-residual classic ECM rule,
    linear_residual_z_res_rule_claude.npz) for the hyper-reduced Newton
    residual assembly.
  - Z_sig (37 points, this session's own reaction-force classic ECM rule,
    classic_ecm_reaction_force_result_claude.npz) for the ONLINE, NATIVE
    reaction-force stress output -- not a post-hoc correction, using the
    same "fixed_classic" regressor_type already validated for D-HPROM-ANN/
    HPROM-ANN's own reaction-force rule.

Reuses build_hrom_meshes_reaction_force_claude.py's build_one_hrom_mesh
unmodified. Uses hprom/ann/maw_dynamic/ecm_weights_all.npz purely as a
metadata TEMPLATE (nq, nothing else this law actually reads) -- its own
Z_res/Z_sig/Z_eps/weights are all overwritten below.
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

ORIGINAL_ECM_NPZ = ROOT / "hprom" / "ann" / "maw_dynamic" / "ecm_weights_all.npz"
LINEAR_RES_RULE_NPZ = HERE / "linear_residual_z_res_rule_claude.npz"
CLASSIC37_NPZ = HERE / "classic_ecm_reaction_force_result_claude.npz"
OUT_DIR = HERE / "linear_hprom_law_mesh"


def main():
    original = dict(np.load(ORIGINAL_ECM_NPZ, allow_pickle=True))
    res_rule = np.load(LINEAR_RES_RULE_NPZ)
    sig_rule = np.load(CLASSIC37_NPZ)

    Z_res = np.asarray(res_rule["Z_res"], dtype=np.int64).reshape(-1)
    w_res_full_src = np.asarray(res_rule["w_res_full"], dtype=float).reshape(-1)
    Z_sig = np.asarray(sig_rule["Z_elements"], dtype=np.int64).reshape(-1)
    w_sig_fixed = np.asarray(sig_rule["w_sel"], dtype=float).reshape(-1)

    overlap = sorted(set(Z_res.tolist()) & set(Z_sig.tolist()))
    Z_union = np.unique(np.concatenate([Z_res, Z_sig]))
    print(f"[build-linear-law-mesh] Z_res({Z_res.size}) union Z_sig({Z_sig.size}) "
          f"-> {Z_union.size} elements (overlap={len(overlap)})")

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
    print(f"[build-linear-law-mesh] origin mesh: {len(origin_elements)} elements, {origin_mp.NumberOfNodes()} nodes")

    meta = build_one_hrom_mesh(
        origin_mp, origin_elements, full_nodeid_to_eqxy, Z_union, OUT_DIR, "linear-HPROM-law"
    )

    replaced = dict(original)
    replaced.update(meta)
    n_elem_ref = int(np.ravel(replaced["n_elem"])[0])

    replaced["Z_res"] = Z_res
    w_res_full_arr = np.zeros(n_elem_ref, dtype=float)
    w_res_full_arr[Z_res] = w_res_full_src[Z_res]
    replaced["w_res_full"] = w_res_full_arr

    replaced["Z_sig"] = Z_sig
    replaced["maw_sig_regressor_type"] = np.array(["fixed_classic"])
    replaced["maw_sig_w_fixed"] = w_sig_fixed
    stale = [k for k in list(replaced) if k.startswith("maw_sig_ann_")] + [
        "maw_sig_coord_train", "maw_sig_W_train", "maw_sig_b_train",
    ]
    for k in stale:
        replaced.pop(k, None)

    full_idx = meta["hrom_element_full_indices"]
    for tag in ("res", "eps", "sig"):
        key = f"w_{tag}_full"
        if key in replaced:
            w_full = np.asarray(replaced[key], dtype=float).reshape(-1)
            if w_full.size == n_elem_ref:
                replaced[f"w_{tag}_hrom"] = w_full[full_idx]

    out_npz = OUT_DIR / "ecm_weights_all.npz"
    np.savez(out_npz, **replaced)
    print(f"[build-linear-law-mesh] saved {out_npz}")

    sim.Finalize()
    print("[build-linear-law-mesh] done.")


if __name__ == "__main__":
    main()
