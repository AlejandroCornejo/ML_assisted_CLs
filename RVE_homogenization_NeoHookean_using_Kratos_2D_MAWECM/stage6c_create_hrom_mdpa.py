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
from typing import Dict, List, Set, Tuple

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


def _split_keys_csv(txt):
    out = []
    for token in str(txt).split(","):
        key = token.strip()
        if key:
            out.append(key)
    return out


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


def _element_node_ids(elem) -> Tuple[int, ...]:
    geom = elem.GetGeometry()
    return tuple(int(geom[i].Id) for i in range(geom.PointsNumber()))


def _augment_selection_with_dirichlet_coverage(
    origin_mp,
    origin_elements,
    selected_idx,
    data,
    min_dirichlet_nodes,
    dirichlet_submodelpart="dirichlet",
):
    """Greedily add boundary-touching elements until a minimum number of
    dirichlet nodes is covered by the selected element set.

    Selection is greedy by:
      1) maximum newly covered dirichlet nodes,
      2) highest absolute ECM weight score (|w_res| + |w_eps| + |w_sig|),
      3) lowest element index.
    """
    info = {
        "enabled": bool(int(min_dirichlet_nodes) > 0),
        "submodelpart": str(dirichlet_submodelpart),
        "n_dirichlet_nodes_total": 0,
        "n_dirichlet_nodes_required": 0,
        "n_dirichlet_nodes_covered_before": 0,
        "n_dirichlet_nodes_covered_after": 0,
        "n_boundary_candidates": 0,
        "n_added_elements": 0,
    }
    selected_idx = np.asarray(selected_idx, dtype=np.int64).reshape(-1)
    if int(min_dirichlet_nodes) <= 0:
        return selected_idx, info

    name = str(dirichlet_submodelpart).strip()
    if not name:
        return selected_idx, info
    if not origin_mp.HasSubModelPart(name):
        print(
            f"[WARN] Requested dirichlet coverage but submodelpart '{name}' does not exist. "
            "Selection is left unchanged."
        )
        return selected_idx, info

    smp = origin_mp.GetSubModelPart(name)
    dir_nodes: Set[int] = set(int(n.Id) for n in smp.Nodes)
    info["n_dirichlet_nodes_total"] = int(len(dir_nodes))
    if len(dir_nodes) == 0:
        print(
            f"[WARN] Submodelpart '{name}' has zero nodes in base mesh. "
            "Selection is left unchanged."
        )
        return selected_idx, info

    n_req = int(min(int(min_dirichlet_nodes), len(dir_nodes)))
    info["n_dirichlet_nodes_required"] = int(n_req)
    if n_req <= 0:
        return selected_idx, info

    n_elem = len(origin_elements)
    elem_nodes: List[Tuple[int, ...]] = [()] * n_elem
    touches_dir = np.zeros(n_elem, dtype=bool)
    for i, elem in enumerate(origin_elements):
        nodes_i = _element_node_ids(elem)
        elem_nodes[i] = nodes_i
        for nid in nodes_i:
            if nid in dir_nodes:
                touches_dir[i] = True
                break

    selected_set: Set[int] = set(int(i) for i in selected_idx.tolist())
    covered_nodes: Set[int] = set()
    for i in selected_set:
        for nid in elem_nodes[i]:
            if nid in dir_nodes:
                covered_nodes.add(nid)
    info["n_dirichlet_nodes_covered_before"] = int(len(covered_nodes))
    if len(covered_nodes) >= n_req:
        info["n_dirichlet_nodes_covered_after"] = int(len(covered_nodes))
        return np.array(sorted(selected_set), dtype=np.int64), info

    # Build a tie-break score from available full-mesh weight magnitudes.
    score = np.zeros(n_elem, dtype=float)
    for key in ("w_res_full", "w_eps_full", "w_sig_full"):
        if key in data:
            vec = np.asarray(data[key], dtype=float).reshape(-1)
            if vec.size == n_elem:
                score += np.abs(vec)

    candidates: Set[int] = set(np.flatnonzero(touches_dir).astype(np.int64).tolist()) - selected_set
    info["n_boundary_candidates"] = int(len(candidates))
    uncovered = set(dir_nodes - covered_nodes)
    added = 0

    while len(covered_nodes) < n_req and candidates:
        best_idx = -1
        best_gain = -1
        best_score = -np.inf
        for i in candidates:
            nodes_i = elem_nodes[i]
            gain = 0
            for nid in nodes_i:
                if nid in uncovered:
                    gain += 1
            sc = float(score[i])
            if gain > best_gain or (gain == best_gain and sc > best_score) or (
                gain == best_gain and sc == best_score and i < best_idx
            ):
                best_idx = int(i)
                best_gain = int(gain)
                best_score = sc

        if best_idx < 0:
            break

        # If no additional dirichlet coverage can be gained, stop.
        if best_gain <= 0:
            break

        selected_set.add(best_idx)
        added += 1
        candidates.remove(best_idx)
        for nid in elem_nodes[best_idx]:
            if nid in uncovered:
                uncovered.remove(nid)
                covered_nodes.add(nid)

    info["n_added_elements"] = int(added)
    info["n_dirichlet_nodes_covered_after"] = int(len(covered_nodes))
    if len(covered_nodes) < n_req:
        print(
            f"[WARN] Requested {n_req} dirichlet nodes in support, reached {len(covered_nodes)}. "
            "Proceeding with best effort."
        )

    return np.array(sorted(selected_set), dtype=np.int64), info


def _compute_selection_weight_summaries(data, selected_idx):
    idx = np.asarray(selected_idx, dtype=np.int64).reshape(-1)
    out = {}
    keys = (
        ("res", "w_res_full"),
        ("eps", "w_eps_full"),
        ("sig", "w_sig_full"),
    )
    for tag, key in keys:
        if key not in data:
            continue
        vec = np.asarray(data[key], dtype=float).reshape(-1)
        if idx.size > 0 and int(np.max(idx)) >= vec.size:
            continue
        s_sel = float(np.sum(vec[idx])) if idx.size > 0 else 0.0
        s_all = float(np.sum(vec))
        out[tag] = {
            "selected": s_sel,
            "total": s_all,
        }
    return out


def _format_weight_summary_for_plot(selection_key, weight_summaries, use_tex=False):
    if not weight_summaries:
        return ""
    sk = str(selection_key).strip()
    key_map = {"Z_res": "res", "Z_eps": "eps", "Z_sig": "sig"}
    if sk in key_map and key_map[sk] in weight_summaries:
        d = weight_summaries[key_map[sk]]
        if bool(use_tex):
            return rf"$\Sigma w_{{{key_map[sk]}}}^{{sel}}={d['selected']:.4f}$ / $\Sigma w_{{{key_map[sk]}}}^{{all}}={d['total']:.4f}$"
        return f"sum(w_{key_map[sk]}) sel/all = {d['selected']:.4f}/{d['total']:.4f}"

    parts = []
    for tag in ("res", "eps", "sig"):
        if tag not in weight_summaries:
            continue
        d = weight_summaries[tag]
        if bool(use_tex):
            parts.append(rf"$\Sigma w_{{{tag}}}^{{sel}}={d['selected']:.3f}$")
        else:
            parts.append(f"sum(w_{tag})={d['selected']:.3f}")
    return " | ".join(parts)


def _build_extra_image_path(base_png, key):
    base_png = str(base_png)
    root, ext = os.path.splitext(base_png)
    ext = ext if ext else ".png"
    return f"{root}_{key}{ext}"


def _save_selected_elements_image_latex_style(
    origin_mp,
    selected_elem_ids_1based,
    out_png,
    model_label="HPROM",
    selection_key="Z_union",
    weight_summary_text="",
    width=2200,
    height=1400,
    use_tex=False,
):
    """
    Save a paper-style PNG showing ECM-selected elements using Matplotlib.
    Uses a LaTeX-like paper style (Computer Modern family) with optional TeX rendering.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style_utils import apply_latex_plot_style
    apply_latex_plot_style()
    from matplotlib.collections import PolyCollection

    fig_w = max(float(width) / 240.0, 7.0)
    fig_h = max(float(height) / 240.0, 5.0)
    # Scale typography with figure area so sizes are not hard-coded.
    base_w, base_h = 12.0, 8.0
    style_scale = np.sqrt((fig_w * fig_h) / (base_w * base_h))

    plt.rcParams.update(
        {
            "text.usetex": bool(use_tex),
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 22.0 * style_scale,
            "axes.labelsize": 20.0 * style_scale,
            "legend.fontsize": 15.0 * style_scale,
            "xtick.labelsize": 16.0 * style_scale,
            "ytick.labelsize": 16.0 * style_scale,
            "lines.linewidth": 2.5,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "figure.figsize": (fig_w, fig_h),
        }
    )

    selected_set = set(int(v) for v in selected_elem_ids_1based)
    n_total = int(origin_mp.NumberOfElements())
    n_sel = int(len(selected_set))
    pct = 100.0 * float(n_sel) / max(float(n_total), 1.0)
    pct_symbol = r"\%" if bool(use_tex) else "%"

    tris_all = []
    tris_sel = []
    for elem in origin_mp.Elements:
        geom = elem.GetGeometry()
        if geom.PointsNumber() < 3:
            continue
        tri = np.array(
            [
                [float(geom[0].X0), float(geom[0].Y0)],
                [float(geom[1].X0), float(geom[1].Y0)],
                [float(geom[2].X0), float(geom[2].Y0)],
            ],
            dtype=float,
        )
        tris_all.append(tri)
        if int(elem.Id) in selected_set:
            tris_sel.append(tri)

    if len(tris_sel) == 0:
        raise RuntimeError("No selected elements found to render.")

    tris_all = np.asarray(tris_all, dtype=float)
    tris_sel = np.asarray(tris_sel, dtype=float)
    out_png = str(out_png)
    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    coll_all = PolyCollection(
        tris_all,
        facecolors=(0.97, 0.97, 0.97, 1.0),
        edgecolors=(0.72, 0.72, 0.72, 0.45),
        linewidths=0.18,
    )
    coll_sel = PolyCollection(
        tris_sel,
        facecolors=(0.83, 0.22, 0.22, 0.92),
        edgecolors=(0.08, 0.08, 0.08, 0.9),
        linewidths=0.22,
    )
    ax.add_collection(coll_all)
    ax.add_collection(coll_sel)

    xmin = float(np.min(tris_all[:, :, 0]))
    xmax = float(np.max(tris_all[:, :, 0]))
    ymin = float(np.min(tris_all[:, :, 1]))
    ymax = float(np.max(tris_all[:, :, 1]))
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - 0.02 * dx, xmax + 0.02 * dx)
    ax.set_ylim(ymin - 0.02 * dy, ymax + 0.02 * dy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(
        f"{model_label} Reduced Mesh (ECM)\n"
        f"{selection_key}: {n_sel}/{n_total} elements ({pct:.1f}{pct_symbol})"
    )
    info_txt = f"Selected: {n_sel}/{n_total} ({pct:.1f}{pct_symbol})"
    extra_txt = str(weight_summary_text).strip()
    if extra_txt:
        info_txt = info_txt + "\n" + extra_txt
    ax.text(
        0.015,
        0.015,
        info_txt,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

        copied.append(
            (
                name,
                len(add_node_ids),
                len(add_elem_ids),
                len(add_cond_ids),
                int(reduced_smp.NumberOfNodes()),
                int(reduced_smp.NumberOfElements()),
                int(reduced_smp.NumberOfConditions()),
            )
        )

    return copied


def main():
    p = argparse.ArgumentParser(description="Build HROM mesh from ECM selection and update ECM npz.")
    p.add_argument("--base-mesh", default="rve_geometry", help="Base mdpa mesh name (with or without .mdpa).")
    p.add_argument("--ecm-file", required=True, help="Input ECM npz file (e.g. stage_6b_hprom_ecm/ecm_weights_all.npz).")
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
    p.add_argument(
        "--save-selection-image",
        default=None,
        help="Optional PNG path to save a paper-style image with selected elements.",
    )
    p.add_argument(
        "--save-extra-selection-images",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "If 1, also save separate images for extra selection keys "
            "(e.g. Z_res, Z_eps, Z_sig, Z_union)."
        ),
    )
    p.add_argument(
        "--extra-selection-keys",
        type=str,
        default="Z_res,Z_eps,Z_sig,Z_union",
        help="Comma-separated list of selection keys for extra images.",
    )
    p.add_argument(
        "--image-width",
        type=int,
        default=2200,
        help="Image width in pixels for --save-selection-image.",
    )
    p.add_argument(
        "--image-height",
        type=int,
        default=1400,
        help="Image height in pixels for --save-selection-image.",
    )
    p.add_argument(
        "--model-label",
        default="HPROM",
        help="Model label to include in the image title (e.g. HPROM-RBF, HPROM-ANN, HPROM-LS-RBF).",
    )
    p.add_argument(
        "--min-dirichlet-nodes",
        type=int,
        default=0,
        help=(
            "If >0, greedily augment selected elements so at least this many nodes "
            "from the dirichlet submodelpart are covered in the reduced mesh."
        ),
    )
    p.add_argument(
        "--dirichlet-submodelpart",
        type=str,
        default="dirichlet",
        help="Name of the submodelpart containing Dirichlet nodes in the base mesh.",
    )
    p.add_argument(
        "--use-tex",
        dest="use_tex",
        action="store_true",
        help="Enable Matplotlib LaTeX rendering in the selection image (default: enabled).",
    )
    p.add_argument(
        "--no-use-tex",
        dest="use_tex",
        action="store_false",
        help="Disable Matplotlib LaTeX rendering in the selection image.",
    )
    p.set_defaults(use_tex=True)
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

    selected_idx, dir_cov_info = _augment_selection_with_dirichlet_coverage(
        origin_mp=origin_mp,
        origin_elements=origin_elements,
        selected_idx=selected_idx,
        data=data,
        min_dirichlet_nodes=int(args.min_dirichlet_nodes),
        dirichlet_submodelpart=str(args.dirichlet_submodelpart),
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
    weight_summaries = _compute_selection_weight_summaries(data, hrom_elem_full_indices)

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
    data_out["hrom_dirichlet_submodelpart"] = np.array(str(dir_cov_info["submodelpart"]))
    data_out["hrom_dirichlet_nodes_total"] = np.array([int(dir_cov_info["n_dirichlet_nodes_total"])], dtype=np.int64)
    data_out["hrom_dirichlet_nodes_required"] = np.array([int(dir_cov_info["n_dirichlet_nodes_required"])], dtype=np.int64)
    data_out["hrom_dirichlet_nodes_covered_before"] = np.array([int(dir_cov_info["n_dirichlet_nodes_covered_before"])], dtype=np.int64)
    data_out["hrom_dirichlet_nodes_covered_after"] = np.array([int(dir_cov_info["n_dirichlet_nodes_covered_after"])], dtype=np.int64)
    data_out["hrom_dirichlet_added_elements"] = np.array([int(dir_cov_info["n_added_elements"])], dtype=np.int64)

    if args.inplace_ecm:
        out_ecm = ecm_file
    else:
        out_ecm = str(args.output_ecm_file) if args.output_ecm_file else ecm_file.replace(".npz", "_with_hrom_mesh.npz")
    _save_ecm(out_ecm, data_out)

    out_img = None
    out_extra_imgs = []
    if args.save_selection_image:
        out_img = str(args.save_selection_image)
        w_plot_txt = _format_weight_summary_for_plot(
            selection_key=str(args.selection_key),
            weight_summaries=weight_summaries,
            use_tex=bool(args.use_tex),
        )
        _save_selected_elements_image_latex_style(
            origin_mp=origin_mp,
            selected_elem_ids_1based=selected_elem_ids_1,
            out_png=out_img,
            model_label=str(args.model_label),
            selection_key=str(args.selection_key),
            weight_summary_text=w_plot_txt,
            width=int(args.image_width),
            height=int(args.image_height),
            use_tex=bool(args.use_tex),
        )
        if bool(int(args.save_extra_selection_images)):
            for key in _split_keys_csv(args.extra_selection_keys):
                if key not in data:
                    continue
                key_idx = np.asarray(data[key], dtype=np.int64).reshape(-1)
                if key_idx.size == 0:
                    continue
                key_idx = np.unique(key_idx)
                if np.min(key_idx) < 0 or np.max(key_idx) >= n_elem_full:
                    continue
                key_elem_ids_1 = [int(origin_elements[int(i)].Id) for i in key_idx]
                key_wsum = _compute_selection_weight_summaries(data, key_idx)
                key_txt = _format_weight_summary_for_plot(
                    selection_key=key,
                    weight_summaries=key_wsum,
                    use_tex=bool(args.use_tex),
                )
                out_key_img = _build_extra_image_path(out_img, key)
                _save_selected_elements_image_latex_style(
                    origin_mp=origin_mp,
                    selected_elem_ids_1based=key_elem_ids_1,
                    out_png=out_key_img,
                    model_label=str(args.model_label),
                    selection_key=str(key),
                    weight_summary_text=key_txt,
                    width=int(args.image_width),
                    height=int(args.image_height),
                    use_tex=bool(args.use_tex),
                )
                out_extra_imgs.append(out_key_img)

    print("=" * 72)
    print("HROM mesh build complete")
    print("=" * 72)
    print(f"Base mesh           : {base_mesh}.mdpa")
    print(f"ECM file            : {ecm_file}")
    print(f"Selection key       : {args.selection_key}")
    print(f"Selected elements   : {selected_idx.size} / {n_elem_full} ({100.0 * selected_idx.size / max(n_elem_full, 1):.1f}%)")
    if weight_summaries:
        for tag in ("res", "eps", "sig"):
            if tag not in weight_summaries:
                continue
            d = weight_summaries[tag]
            print(f"sum(w_{tag}) sel/all : {d['selected']:.6f} / {d['total']:.6f}")
    if int(args.min_dirichlet_nodes) > 0:
        print(
            "Dirichlet coverage  : "
            f"required={dir_cov_info['n_dirichlet_nodes_required']}, "
            f"before={dir_cov_info['n_dirichlet_nodes_covered_before']}, "
            f"after={dir_cov_info['n_dirichlet_nodes_covered_after']}, "
            f"added_elems={dir_cov_info['n_added_elements']}"
        )
    print(f"Condition mode      : {args.condition_mode}")
    print(f"HROM mesh           : {out_mesh}.mdpa")
    print(f"HROM elems/conds    : {hrom_elem_ids_0.size} / {hrom_cond_ids_0.size}")
    if copied_subparts:
        print("Copied submodelparts:")
        for name, nn, ne, nc, tn, te, tc in copied_subparts:
            print(
                f"  - {name}: added(nodes/elems/conds)=({nn}/{ne}/{nc}), "
                f"total=({tn}/{te}/{tc})"
            )
    else:
        print("Copied submodelparts: none")
    print(f"Updated ECM file    : {out_ecm}")
    if out_img is not None:
        print(f"Selection image     : {out_img}")
        print(f"Selection image TeX : {bool(args.use_tex)}")
        if out_extra_imgs:
            print("Extra selection images:")
            for pth in out_extra_imgs:
                print(f"  - {pth}")
    sim.Finalize()


if __name__ == "__main__":
    main()
