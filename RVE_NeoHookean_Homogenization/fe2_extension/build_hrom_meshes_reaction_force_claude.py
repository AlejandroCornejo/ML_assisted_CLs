#!/usr/bin/env python3
"""Build two genuinely hyper-reduced Kratos meshes for the unified,
reaction-force-targeting stress rule:

  - HPROM-ANN (iterative): needs Z_res (residual rule, for its own Newton
    loop) union Z_support (this session's new reaction-force stress rule,
    replacing Z_sig). Z_eps (strain-homogenization) is DELIBERATELY
    excluded: evaluate()'s first return value (hom_eps) is discarded by
    every caller in this project (dhprom_ann_pk2_2d_vectorized_float64 /
    _consistent_float64 and their hprom_iterative counterparts all assign
    it to `_`), confirmed by grepping every call site -- so nothing
    downstream ever needs it, and keeping it would cost 10 more mesh
    elements for zero benefit. Dropping it from the MESH is safe even
    without a code change: _evaluate_maw_hom_weights_current
    (hprom_ann_solver_rve.py) only copies weights for elements present in
    full_to_local, silently zero-filling missing ones -- no crash, and the
    (now meaningless) hom_eps is discarded anyway.
  - D-HPROM-ANN (direct): needs Z_support alone -- it never runs Newton
    iterations, so it never needs Z_res either.

Both sets are disjoint (verified below: Z_res & Z_support overlap is
empty), so HPROM-ANN's mesh is exactly 20 elements, D-HPROM-ANN's is
exactly 10 -- matching the user's own expectation exactly ("por eso la
nueva deberia ser solo 20... o menos, si hay algun elemento repetido").

Uses the native Kratos utility this project's OLD sibling project already
relied on for this exact purpose
(RVE_homogenization_NeoHookean_using_Kratos/stage6c_create_hrom_mdpa.py):
KratosMultiphysics.RomApplication.RomAuxiliaryUtilities.
SetHRomComputingModelPartWithLists. The submodelpart-copying helper
(_copy_top_level_submodelparts_by_intersection) is ported verbatim from
that script; the Dirichlet-coverage-augmentation helper is NOT ported,
since the currently-deployed mesh never used it either
(hrom_dirichlet_nodes_required=0 in the existing ecm_weights_all.npz) and
every new element is already Dirichlet-boundary-adjacent by construction
(Z_support was drawn from the ~156-element boundary-touching candidate
pool from the start).

hrom_mesh_base/hrom_full_mesh_base are written as ABSOLUTE paths (not the
existing deployment's bare relative names) so the resulting ecm npz is
correct regardless of the caller's CWD -- a latent fragility in the
existing convention (confirmed: hprom/ann/maw_dynamic/ecm_weights_all.npz's
own hrom_mesh_base is a bare name that only resolves because
fe2_extension/ happens to keep its own same-named copy of the mesh
alongside ProjectParameters.json).
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
from KratosMultiphysics import kratos_utilities  # noqa: E402
import KratosMultiphysics.RomApplication as KratosROM  # noqa: E402

from fom_solver_rve import (  # noqa: E402
    setup_kratos_parameters,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
)

ORIGINAL_ECM_NPZ = ROOT / "hprom" / "ann" / "maw_dynamic" / "ecm_weights_all.npz"
NEW_SIG_MODEL_NPZ = HERE / "reaction_force_ecm_ann_model_FINAL_claude.npz"

OUT_DIR_HPROMANN = HERE / "maw_dynamic_reaction_force_hpromann"
OUT_DIR_DHPROMANN = HERE / "maw_dynamic_reaction_force_dhpromann"
MESH_NAME = "rve_geometry_hrom"  # bare name inside each output dir


def _copy_top_level_submodelparts_by_intersection(origin_mp, reduced_mp):
    """Verbatim port of stage6c_create_hrom_mdpa.py's helper of the same
    name -- preserves top-level submodelparts (material, dirichlet) in the
    reduced mesh by ID intersection."""
    reduced_node_ids = set(int(n.Id) for n in reduced_mp.Nodes)
    reduced_elem_ids = set(int(e.Id) for e in reduced_mp.Elements)
    reduced_cond_ids = set(int(c.Id) for c in reduced_mp.Conditions)

    copied = []
    for origin_smp in origin_mp.SubModelParts:
        name = str(origin_smp.Name)
        node_ids = [int(n.Id) for n in origin_smp.Nodes if int(n.Id) in reduced_node_ids]
        elem_ids = [int(e.Id) for e in origin_smp.Elements if int(e.Id) in reduced_elem_ids]
        cond_ids = [int(c.Id) for c in origin_smp.Conditions if int(c.Id) in reduced_cond_ids]

        if not node_ids and not elem_ids and not cond_ids:
            continue

        if reduced_mp.HasSubModelPart(name):
            reduced_smp = reduced_mp.GetSubModelPart(name)
        else:
            reduced_smp = reduced_mp.CreateSubModelPart(name)

        existing_node_ids = set(int(n.Id) for n in reduced_smp.Nodes)
        existing_elem_ids = set(int(e.Id) for e in reduced_smp.Elements)
        existing_cond_ids = set(int(c.Id) for c in reduced_smp.Conditions)

        add_node_ids = [i for i in node_ids if i not in existing_node_ids]
        add_elem_ids = [i for i in elem_ids if i not in existing_elem_ids]
        add_cond_ids = [i for i in cond_ids if i not in existing_cond_ids]

        if add_node_ids:
            reduced_smp.AddNodes(add_node_ids)
        if add_elem_ids:
            reduced_smp.AddElements(add_elem_ids)
        if add_cond_ids:
            reduced_smp.AddConditions(add_cond_ids)

        copied.append((name, len(add_node_ids), len(add_elem_ids), len(add_cond_ids)))

    return copied


def build_one_hrom_mesh(origin_mp, origin_elements, full_nodeid_to_eqxy, selected_idx_full, out_dir, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mesh = str(out_dir / MESH_NAME)

    selected_idx_full = np.unique(np.asarray(selected_idx_full, dtype=np.int64))
    selected_elem_ids_1 = [int(origin_elements[int(i)].Id) for i in selected_idx_full]
    selected_elem_ids_0 = [eid - 1 for eid in selected_elem_ids_1]
    # condition-mode="all", matching the existing deployment's own
    # hrom_condition_mode field exactly.
    selected_cond_ids_0 = [int(cond.Id) - 1 for cond in origin_mp.Conditions]

    model_hrom = KM.Model()
    hrom_mp = model_hrom.CreateModelPart(origin_mp.Name)
    KratosROM.RomAuxiliaryUtilities.SetHRomComputingModelPartWithLists(
        selected_elem_ids_0, selected_cond_ids_0, origin_mp, hrom_mp,
    )
    copied_subparts = _copy_top_level_submodelparts_by_intersection(origin_mp, hrom_mp)

    io_flags = KM.IO.WRITE | KM.IO.MESH_ONLY | KM.IO.SCIENTIFIC_PRECISION
    KM.ModelPartIO(out_mesh, io_flags).WriteModelPart(hrom_mp)
    kratos_utilities.DeleteFileIfExisting(f"{out_mesh}.time")

    full_elem_id_to_index = {int(elem.Id): i for i, elem in enumerate(origin_elements)}
    hrom_elem_ids_1 = np.array([int(elem.Id) for elem in hrom_mp.Elements], dtype=np.int64)
    hrom_elem_full_indices = np.array([full_elem_id_to_index[int(eid)] for eid in hrom_elem_ids_1.tolist()], dtype=np.int64)
    hrom_cond_ids_0 = np.array([int(cond.Id) - 1 for cond in hrom_mp.Conditions], dtype=np.int64)
    hrom_node_ids = np.array([int(node.Id) for node in hrom_mp.Nodes], dtype=np.int64)
    hrom_node_full_eqid_x = np.array(
        [int(full_nodeid_to_eqxy[int(nid)][0]) for nid in hrom_node_ids.tolist()], dtype=np.int64
    )
    hrom_node_full_eqid_y = np.array(
        [int(full_nodeid_to_eqxy[int(nid)][1]) for nid in hrom_node_ids.tolist()], dtype=np.int64
    )

    print(f"[{label}] selected {selected_idx_full.size} elements -> HROM mesh has "
          f"{hrom_elem_ids_1.size} elements / {hrom_node_ids.size} nodes / {hrom_cond_ids_0.size} conditions")
    if copied_subparts:
        for name, nn, ne, nc in copied_subparts:
            print(f"    submodelpart '{name}': +nodes={nn}, +elems={ne}, +conds={nc}")
    print(f"    wrote {out_mesh}.mdpa")

    return {
        "hrom_mesh_base": np.array(out_mesh),
        "hrom_full_mesh_base": np.array(str(HERE / "rve_geometry")),
        "hrom_selection_key": np.array(f"Z_support_custom_{label}"),
        "hrom_condition_mode": np.array("all"),
        "hrom_element_full_indices": hrom_elem_full_indices,
        "hrom_element_ids_0based": hrom_elem_ids_1 - 1,
        "hrom_condition_ids_0based": hrom_cond_ids_0,
        "hrom_node_ids": hrom_node_ids,
        "hrom_node_full_eqid_x": hrom_node_full_eqid_x,
        "hrom_node_full_eqid_y": hrom_node_full_eqid_y,
        "hrom_n_elem": np.array([int(hrom_elem_ids_1.size)], dtype=np.int64),
        "hrom_n_cond": np.array([int(hrom_cond_ids_0.size)], dtype=np.int64),
        "hrom_dirichlet_submodelpart": np.array("dirichlet"),
        "hrom_dirichlet_nodes_total": np.array([0], dtype=np.int64),
        "hrom_dirichlet_nodes_required": np.array([0], dtype=np.int64),
        "hrom_dirichlet_nodes_covered_before": np.array([0], dtype=np.int64),
        "hrom_dirichlet_nodes_covered_after": np.array([0], dtype=np.int64),
        "hrom_dirichlet_added_elements": np.array([0], dtype=np.int64),
    }


def build_ecm_npz(original, new_sig_model, Z_support, hrom_meta, out_dir):
    n_layers = int(new_sig_model["scalar_n_layers"])
    replaced = dict(original)
    replaced.update(hrom_meta)
    replaced["Z_sig"] = np.asarray(Z_support, dtype=np.int64)
    replaced["maw_sig_regressor_type"] = np.array(["ann"])
    replaced["maw_sig_ann_x_mean"] = np.asarray(new_sig_model["x_mean"], dtype=float)
    replaced["maw_sig_ann_x_std"] = np.asarray(new_sig_model["x_std"], dtype=float)
    replaced["maw_sig_ann_activation"] = np.array([str(new_sig_model["scalar_activation"])])
    replaced["maw_sig_ann_hidden_dims"] = np.asarray(new_sig_model["hidden_dims"], dtype=np.int64)
    replaced["maw_sig_ann_n_layers"] = np.array([n_layers], dtype=np.int64)
    replaced["maw_sig_ann_target_sum"] = np.array([float(new_sig_model["scalar_target_sum"])])
    replaced["maw_sig_ann_best_epoch"] = np.array([int(new_sig_model["scalar_best_epoch"])], dtype=np.int64)
    replaced["maw_sig_ann_train_rel_error"] = np.array([float(new_sig_model["scalar_train_rel_error"])])
    replaced["maw_sig_ann_val_rel_error"] = np.array([float(new_sig_model["scalar_val_rel_error"])])
    for i in range(n_layers):
        replaced[f"maw_sig_ann_W_{i}"] = np.asarray(new_sig_model[f"W_{i}"], dtype=float)
        replaced[f"maw_sig_ann_b_{i}"] = np.asarray(new_sig_model[f"b_{i}"], dtype=float)
    stale = [k for k in list(replaced) if k.startswith("maw_sig_ann_W_") or k.startswith("maw_sig_ann_b_")]
    for k in stale:
        idx = int(k.rsplit("_", 1)[1])
        if idx >= n_layers:
            del replaced[k]

    # w_*_hrom projections are informational/diagnostic only (never read by
    # the online classes at runtime -- confirmed by grepping every
    # ecm_data[...] access site) but kept for parity with the existing
    # deployment's own file format.
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
    new_sig_model = np.load(NEW_SIG_MODEL_NPZ)

    Z_res = np.asarray(original["Z_res"], dtype=np.int64).reshape(-1)
    Z_support = np.asarray(new_sig_model["Z_support"], dtype=np.int64).reshape(-1)
    overlap = sorted(set(Z_res.tolist()) & set(Z_support.tolist()))
    print(f"[build] Z_res     ({Z_res.size})  = {sorted(Z_res.tolist())}")
    print(f"[build] Z_support ({Z_support.size}) = {sorted(Z_support.tolist())}")
    print(f"[build] Z_res & Z_support overlap: {overlap}")

    Z_hpromann = np.unique(np.concatenate([Z_res, Z_support]))
    Z_dhpromann = np.unique(Z_support)
    print(f"[build] HPROM-ANN mesh needs   {Z_hpromann.size} elements: {sorted(Z_hpromann.tolist())}")
    print(f"[build] D-HPROM-ANN mesh needs {Z_dhpromann.size} elements: {sorted(Z_dhpromann.tolist())}")

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
    print(f"[build] origin mesh: {len(origin_elements)} elements, {origin_mp.NumberOfNodes()} nodes")

    meta_hpromann = build_one_hrom_mesh(
        origin_mp, origin_elements, full_nodeid_to_eqxy, Z_hpromann, OUT_DIR_HPROMANN, "HPROM-ANN"
    )
    meta_dhpromann = build_one_hrom_mesh(
        origin_mp, origin_elements, full_nodeid_to_eqxy, Z_dhpromann, OUT_DIR_DHPROMANN, "D-HPROM-ANN"
    )

    build_ecm_npz(original, new_sig_model, Z_support, meta_hpromann, OUT_DIR_HPROMANN)
    build_ecm_npz(original, new_sig_model, Z_support, meta_dhpromann, OUT_DIR_DHPROMANN)

    sim.Finalize()
    print("[build] done.")


if __name__ == "__main__":
    main()
