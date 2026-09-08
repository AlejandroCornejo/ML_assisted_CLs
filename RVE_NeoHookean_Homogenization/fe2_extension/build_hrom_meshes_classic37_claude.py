#!/usr/bin/env python3
"""Mitigation attempt: rebuild the HROM meshes + ecm_weights_all.npz using
the 37-point CLASSIC (fixed, non-adaptive) ECM rule
(classic_ecm_reaction_force_result_claude.npz) instead of the 10-point
ANN-fitted one, after diagnosing that the reaction-force formula's own
sensitivity to POD-ANN displacement-reconstruction error (worse at
Cook's shear-dominated, near-origin operating states than at Table 6's
own validation trajectory) is amplified by BOTH (a) a small cubature
support and (b) an ADDITIONAL ANN-extrapolation-uncertainty layer on the
weights themselves. The classic rule removes (b) entirely (w_sel is a
FIXED vector, no ANN evaluation at all -- KratosROM-side, uses the
already-supported regressor_type="fixed_classic" path in
hprom_ann_solver_rve.py's _build_maw_hom_target_model /
_evaluate_maw_hom_weights_current) and, being a much better-conditioned
37-point cubature (oracle accuracy 0.0038%/0.0413% on Table-6/Cook vs the
10-point ANN rule's 0.16%/1.37%), should also reduce (a).

Reuses build_hrom_meshes_reaction_force_claude.py's own mesh-building
helpers unmodified (imported, not copy-pasted).
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

CLASSIC_ECM_NPZ = HERE / "classic_ecm_reaction_force_result_claude.npz"
ORIGINAL_ECM_NPZ = ROOT / "hprom" / "ann" / "maw_dynamic" / "ecm_weights_all.npz"

OUT_DIR_HPROMANN = HERE / "maw_dynamic_reaction_force_classic37_hpromann"
OUT_DIR_DHPROMANN = HERE / "maw_dynamic_reaction_force_classic37_dhpromann"


def build_ecm_npz_fixed_classic(original, Z_support, w_fixed, hrom_meta, out_dir):
    replaced = dict(original)
    replaced.update(hrom_meta)
    replaced["Z_sig"] = np.asarray(Z_support, dtype=np.int64)
    replaced["maw_sig_regressor_type"] = np.array(["fixed_classic"])
    replaced["maw_sig_w_fixed"] = np.asarray(w_fixed, dtype=float).reshape(-1)
    # Stale fields from the OLD (10-element ANN) sig target: coord_train/
    # W_train's own row count is checked against the NEW Z_sig's size
    # regardless of regressor_type (_build_maw_hom_target_model validates
    # this unconditionally), so leaving them would hard-fail; the
    # ann_* weights are similarly stale/unused now but removed too, for
    # clarity, not because leaving them would break anything.
    stale = [k for k in list(replaced) if k.startswith("maw_sig_ann_")] + [
        "maw_sig_coord_train", "maw_sig_W_train", "maw_sig_b_train",
    ]
    for k in stale:
        replaced.pop(k, None)

    n_ref = int(np.ravel(replaced["n_elem"])[0])
    full_idx = hrom_meta["hrom_element_full_indices"]
    for tag in ("res", "eps", "sig"):
        key = f"w_{tag}_full"
        if key in replaced:
            w_full = np.asarray(replaced[key], dtype=float).reshape(-1)
            if w_full.size == n_ref:
                replaced[f"w_{tag}_hrom"] = w_full[full_idx]

    out_npz = out_dir / "ecm_weights_all.npz"
    np.savez(out_npz, **replaced)
    print(f"    saved {out_npz}")
    return out_npz


def main():
    original = dict(np.load(ORIGINAL_ECM_NPZ, allow_pickle=True))
    classic = np.load(CLASSIC_ECM_NPZ)

    Z_res = np.asarray(original["Z_res"], dtype=np.int64).reshape(-1)
    Z_support = np.asarray(classic["Z_elements"], dtype=np.int64).reshape(-1)
    w_fixed = np.asarray(classic["w_sel"], dtype=float).reshape(-1)
    overlap = sorted(set(Z_res.tolist()) & set(Z_support.tolist()))
    print(f"[build-classic37] Z_res     ({Z_res.size})  = {sorted(Z_res.tolist())}")
    print(f"[build-classic37] Z_support ({Z_support.size}) = {sorted(Z_support.tolist())}")
    print(f"[build-classic37] Z_res & Z_support overlap: {overlap}")

    Z_hpromann = np.unique(np.concatenate([Z_res, Z_support]))
    Z_dhpromann = np.unique(Z_support)
    print(f"[build-classic37] HPROM-ANN mesh needs   {Z_hpromann.size} elements")
    print(f"[build-classic37] D-HPROM-ANN mesh needs {Z_dhpromann.size} elements")

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
    print(f"[build-classic37] origin mesh: {len(origin_elements)} elements, {origin_mp.NumberOfNodes()} nodes")

    meta_hpromann = build_one_hrom_mesh(
        origin_mp, origin_elements, full_nodeid_to_eqxy, Z_hpromann, OUT_DIR_HPROMANN, "HPROM-ANN-classic37"
    )
    meta_dhpromann = build_one_hrom_mesh(
        origin_mp, origin_elements, full_nodeid_to_eqxy, Z_dhpromann, OUT_DIR_DHPROMANN, "D-HPROM-ANN-classic37"
    )

    build_ecm_npz_fixed_classic(original, Z_support, w_fixed, meta_hpromann, OUT_DIR_HPROMANN)
    build_ecm_npz_fixed_classic(original, Z_support, w_fixed, meta_dhpromann, OUT_DIR_DHPROMANN)

    sim.Finalize()
    print("[build-classic37] done.")


if __name__ == "__main__":
    main()
