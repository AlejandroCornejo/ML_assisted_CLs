#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build an HROM mesh (.mdpa) from ECM-selected elements and update ECM data
with reduced-mesh-aligned residual weights.
"""

import argparse
import os
import sys
import re
import numpy as np

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from KratosMultiphysics import kratos_utilities
import KratosMultiphysics.RomApplication as KratosROM
from fom_solver_rve import setup_kratos_parameters, RVEHomogenizationDatasetGenerator


def _strip_mdpa_extension(mesh_name):
    s = str(mesh_name)
    return s[:-5] if s.endswith(".mdpa") else s


def _to_numpy_dict(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _select_condition_ids_0based(origin_mp, selected_elem_ids_1based, mode):
    mode = str(mode).strip().lower()
    if mode == "none":
        return []
    if mode == "all":
        return [int(cond.Id) - 1 for cond in origin_mp.Conditions]

    selected_nodes = set()
    elem_id_set = set(int(v) for v in selected_elem_ids_1based)
    for elem in origin_mp.Elements:
        if int(elem.Id) in elem_id_set:
            geom = elem.GetGeometry()
            for i in range(geom.PointsNumber()):
                selected_nodes.add(int(geom[i].Id))

    cond_ids_0 = []
    for cond in origin_mp.Conditions:
        geom = cond.GetGeometry()
        keep = True
        for i in range(geom.PointsNumber()):
            if int(geom[i].Id) not in selected_nodes:
                keep = False
                break
        if keep:
            cond_ids_0.append(int(cond.Id) - 1)
    return cond_ids_0


def _project_weight_to_hrom(data, full_key, full_indices):
    if full_key not in data:
        return None
    w_full = np.asarray(data[full_key], dtype=float).reshape(-1)
    if np.max(full_indices, initial=-1) >= w_full.size:
        raise RuntimeError(
            f"Cannot project {full_key}: max full index {int(np.max(full_indices))} >= len({full_key})={w_full.size}"
        )
    return w_full[full_indices]


def _save_ecm(path, data):
    np.savez(path, **data)


def _copy_top_level_submodelparts_by_intersection(origin_mp, reduced_mp):
    """
    Preserve top-level submodelparts (e.g. material, dirichlet) in the reduced mesh.
    Entities are copied by ID intersection with the reduced model part.
    """
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


def main():
    p = argparse.ArgumentParser(description="Build HROM mesh from ECM selection and update ECM npz.")
    p.add_argument("--base-mesh", default="rve_geometry", help="Base mdpa mesh name (with or without .mdpa).")
    p.add_argument("--ecm-file", required=True, help="Input ECM npz file (e.g. stage_9_hprom_rbf_data/ecm_weights_all.npz).")
    p.add_argument(
        "--selection-key",
        default="Z_union",
        choices=["Z_res", "Z_union", "Z_eps", "Z_sig"],
        help="ECM index set used to build HROM mesh.",
    )
    p.add_argument(
        "--condition-mode",
        default="all",
        choices=["all", "selected_nodes", "none"],
        help="Which conditions to include in the HROM mesh.",
    )
    p.add_argument("--output-mesh", default=None, help="Output mesh base name (without .mdpa).")
    p.add_argument(
        "--inplace-ecm",
        action="store_true",
        help="Overwrite --ecm-file with added HROM keys. If omitted, writes a new npz.",
    )
    p.add_argument(
        "--output-ecm-file",
        default=None,
        help="Output ECM npz path when not using --inplace-ecm.",
    )
    args = p.parse_args()

    base_mesh = _strip_mdpa_extension(args.base_mesh)
    ecm_file = str(args.ecm_file)
    if not os.path.exists(ecm_file):
        raise FileNotFoundError(ecm_file)

    if args.output_mesh:
        out_mesh = _strip_mdpa_extension(args.output_mesh)
    else:
        src_tag = os.path.basename(os.path.dirname(ecm_file)) or "ecm_data"
        src_tag = re.sub(r"[^A-Za-z0-9_]+", "_", src_tag)
        out_mesh = f"{base_mesh}_{src_tag}_{args.selection_key.lower()}_hrom"

    data = _to_numpy_dict(ecm_file)
    if args.selection_key not in data:
        raise KeyError(f"{args.selection_key} not found in {ecm_file}. Available keys: {sorted(data.keys())}")

    selected_idx = np.asarray(data[args.selection_key], dtype=np.int64).reshape(-1)
    selected_idx = np.unique(selected_idx)
    if selected_idx.size == 0:
        raise RuntimeError(f"{args.selection_key} is empty.")
    if np.min(selected_idx) < 0:
        raise RuntimeError(f"{args.selection_key} contains negative indices.")

    parameters = setup_kratos_parameters(base_mesh)
    model_origin = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model_origin, parameters)
    sim.Initialize()
    origin_mp = sim._GetSolver().GetComputingModelPart()

    origin_elements = list(origin_mp.Elements)
    n_elem_full = len(origin_elements)
    if np.max(selected_idx) >= n_elem_full:
        raise RuntimeError(
            f"{args.selection_key} max index {int(np.max(selected_idx))} out of range for mesh elements {n_elem_full}."
        )

    selected_elem_ids_1 = [int(origin_elements[int(i)].Id) for i in selected_idx]
    selected_elem_ids_0 = [eid - 1 for eid in selected_elem_ids_1]
    selected_cond_ids_0 = _select_condition_ids_0based(origin_mp, selected_elem_ids_1, mode=args.condition_mode)

    model_hrom = KM.Model()
    hrom_mp = model_hrom.CreateModelPart(origin_mp.Name)
    KratosROM.RomAuxiliaryUtilities.SetHRomComputingModelPartWithLists(
        selected_elem_ids_0,
        selected_cond_ids_0,
        origin_mp,
        hrom_mp,
    )
    copied_subparts = _copy_top_level_submodelparts_by_intersection(origin_mp, hrom_mp)

    io_flags = KM.IO.WRITE | KM.IO.MESH_ONLY | KM.IO.SCIENTIFIC_PRECISION
    KM.ModelPartIO(out_mesh, io_flags).WriteModelPart(hrom_mp)
    kratos_utilities.DeleteFileIfExisting(f"{out_mesh}.time")

    full_elem_id_to_index = {int(elem.Id): i for i, elem in enumerate(origin_elements)}
    hrom_elem_ids_1 = np.array([int(elem.Id) for elem in hrom_mp.Elements], dtype=np.int64)
    hrom_elem_ids_0 = hrom_elem_ids_1 - 1
    hrom_elem_full_indices = np.array([full_elem_id_to_index[int(eid)] for eid in hrom_elem_ids_1], dtype=np.int64)
    hrom_cond_ids_0 = np.array([int(cond.Id) - 1 for cond in hrom_mp.Conditions], dtype=np.int64)

    data_out = dict(data)
    data_out["hrom_mesh_base"] = np.array(out_mesh)
    data_out["hrom_full_mesh_base"] = np.array(base_mesh)
    data_out["hrom_selection_key"] = np.array(str(args.selection_key))
    data_out["hrom_condition_mode"] = np.array(str(args.condition_mode))
    data_out["hrom_element_full_indices"] = hrom_elem_full_indices
    data_out["hrom_element_ids_0based"] = hrom_elem_ids_0
    data_out["hrom_condition_ids_0based"] = hrom_cond_ids_0
    data_out["hrom_n_elem"] = np.array([int(hrom_elem_ids_0.size)], dtype=np.int64)
    data_out["hrom_n_cond"] = np.array([int(hrom_cond_ids_0.size)], dtype=np.int64)
    data_out["w_res_hrom"] = _project_weight_to_hrom(data, "w_res_full", hrom_elem_full_indices)
    if "w_eps_full" in data:
        data_out["w_eps_hrom"] = _project_weight_to_hrom(data, "w_eps_full", hrom_elem_full_indices)
    if "w_sig_full" in data:
        data_out["w_sig_hrom"] = _project_weight_to_hrom(data, "w_sig_full", hrom_elem_full_indices)

    if args.inplace_ecm:
        out_ecm = ecm_file
    else:
        out_ecm = str(args.output_ecm_file) if args.output_ecm_file else ecm_file.replace(".npz", "_with_hrom_mesh.npz")
    _save_ecm(out_ecm, data_out)

    print("=" * 72)
    print("HROM mesh build complete")
    print("=" * 72)
    print(f"Base mesh           : {base_mesh}.mdpa")
    print(f"ECM file            : {ecm_file}")
    print(f"Selection key       : {args.selection_key}")
    print(f"Selected elements   : {selected_idx.size} / {n_elem_full} ({100.0 * selected_idx.size / max(n_elem_full, 1):.1f}%)")
    print(f"Condition mode      : {args.condition_mode}")
    print(f"HROM mesh           : {out_mesh}.mdpa")
    print(f"HROM elems/conds    : {hrom_elem_ids_0.size} / {hrom_cond_ids_0.size}")
    if copied_subparts:
        print("Copied submodelparts:")
        for name, nn, ne, nc in copied_subparts:
            print(f"  - {name}: nodes={nn}, elems={ne}, conds={nc}")
    else:
        print("Copied submodelparts: none")
    print(f"Updated ECM file    : {out_ecm}")
    sim.Finalize()


if __name__ == "__main__":
    main()
