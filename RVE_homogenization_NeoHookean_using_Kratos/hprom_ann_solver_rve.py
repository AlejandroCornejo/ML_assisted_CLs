#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPROM-ANN solver:
  - Hyper-reduced assembly on ECM-selected elements
  - Nonlinear ANN manifold projection with tangent Jacobian
"""

import os
import sys
import time
import numpy as np
import torch
try:
    from torch.func import jacfwd as _torch_jacfwd
except Exception:
    _torch_jacfwd = None

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import VectorizedAssembler
from fom_solver_rve import (
    DeformationGradientFromGreenLagrange2D,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    AssembleGlobalSystem,
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    BuildDynamicSegmentSteps,
    CalculateHomogenizedStressAndStrainKratosReference,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
    USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    NEWTON_TOL_ABS,
)
from hprom_solver_rve import (
    AssembleHyperReducedSystem,
    ResetHyperReductionAssemblyStats,
    GetHyperReductionAssemblyStats,
    ResolveResidualHyperReductionSelection,
    ResolveActiveFreeDofsAndBasisRows,
    ResolveHomogenizationWeightSelection,
    CalculateHomogenizedFromAssemblerWithElementWeights,
    GetReferenceIntegrationMeasureFromMesh,
)
from prom_ann_solver_rve import LoadPromAnnModel
from mawecm_rbf_weights import eval_mawecm_rbf
from mawecm_ann_weights import eval_mawecm_ann


def _as_int_scalar(arr, key):
    return int(np.ravel(arr[key])[0])


def _as_float_scalar(arr, key):
    return float(np.ravel(arr[key])[0])


def _as_str_scalar(arr, key, default=""):
    if key not in arr:
        return str(default)
    return str(np.ravel(arr[key])[0])


def _build_maw_hom_target_model(ecm_data, target, prefix=None, z_key=None, target_label=None):
    t = str(target).strip().lower()
    if t not in ("eps", "sig"):
        raise ValueError(f"Unsupported MAW homogenization target '{target}'.")
    z_key = z_key or {"eps": "Z_eps", "sig": "Z_sig"}[t]
    prefix = prefix or f"maw_{t}_"
    label = str(target_label or t)
    regressor_type = _as_str_scalar(ecm_data, prefix + "regressor_type", default="rbf").strip().lower()
    if regressor_type not in ("rbf", "ann", "fixed_classic"):
        raise RuntimeError(
            f"[HPROM-ANN] Unsupported MAW homogenization regressor '{regressor_type}' for {label}."
        )
    if regressor_type == "fixed_classic":
        required = [
            z_key,
            prefix + "w_fixed",
        ]
    elif regressor_type == "ann":
        required = [
            z_key,
            prefix + "ann_x_mean",
            prefix + "ann_x_std",
            prefix + "ann_activation",
            prefix + "ann_target_sum",
            prefix + "ann_n_layers",
        ]
    else:
        required = [
            z_key,
            prefix + "rbf_centers",
            prefix + "rbf_length_scales",
            prefix + "rbf_alpha",
            prefix + "rbf_beta",
            prefix + "rbf_scale",
            prefix + "rbf_poly_mode",
        ]
    missing = [k for k in required if k not in ecm_data]
    if missing:
        raise RuntimeError(f"[HPROM-ANN] Missing MAW homogenization keys for {label}: {missing}")

    z_support_full = np.asarray(ecm_data[z_key], dtype=np.int64).reshape(-1)
    if z_support_full.size == 0:
        raise RuntimeError(f"[HPROM-ANN] Empty MAW homogenization support for {label}.")

    renorm_enabled = (
        bool(_as_int_scalar(ecm_data, "maw_rbf_renorm"))
        if "maw_rbf_renorm" in ecm_data
        else True
    )
    clip_nonnegative = (
        bool(_as_int_scalar(ecm_data, "maw_rbf_clip_nonnegative"))
        if "maw_rbf_clip_nonnegative" in ecm_data
        else True
    )
    renorm_target = None
    if renorm_enabled and (prefix + "renorm_target") in ecm_data:
        renorm_target = _as_float_scalar(ecm_data, prefix + "renorm_target")
    coord_label = _as_str_scalar(ecm_data, prefix + "coord_label", default="q").strip().lower()
    if coord_label not in ("q", "mu"):
        raise RuntimeError(
            f"[HPROM-ANN] Unsupported MAW homogenization coordinate '{coord_label}' for {label}."
        )

    rbf_model = None
    ann_model = None
    w_fixed = None
    coord_train = None
    W_train = None
    if (prefix + "coord_train") in ecm_data and (prefix + "W_train") in ecm_data:
        coord_train = np.asarray(ecm_data[prefix + "coord_train"], dtype=float)
        W_train = np.asarray(ecm_data[prefix + "W_train"], dtype=float)
        if coord_train.ndim != 2:
            raise RuntimeError(
                f"[HPROM-ANN] Invalid MAW {label} coord_train shape {coord_train.shape}; expected 2D."
            )
        if W_train.ndim != 2:
            raise RuntimeError(
                f"[HPROM-ANN] Invalid MAW {label} W_train shape {W_train.shape}; expected 2D."
            )
        if W_train.shape[0] != z_support_full.size:
            raise RuntimeError(
                f"[HPROM-ANN] Invalid MAW {label} W_train/support mismatch: "
                f"{W_train.shape[0]} vs {z_support_full.size}."
            )
        if W_train.shape[1] != coord_train.shape[0]:
            raise RuntimeError(
                f"[HPROM-ANN] Invalid MAW {label} W_train/coord_train mismatch: "
                f"{W_train.shape[1]} vs {coord_train.shape[0]}."
            )
    if regressor_type == "fixed_classic":
        w_fixed = np.asarray(ecm_data[prefix + "w_fixed"], dtype=float).reshape(-1)
        if w_fixed.size != z_support_full.size:
            raise RuntimeError(
                f"[HPROM-ANN] Fixed-classic MAW {label} support/weight mismatch: "
                f"{z_support_full.size} vs {w_fixed.size}."
            )
    elif regressor_type == "ann":
        n_layers = _as_int_scalar(ecm_data, prefix + "ann_n_layers")
        ann_model = {
            "x_mean": np.asarray(ecm_data[prefix + "ann_x_mean"], dtype=float),
            "x_std": np.asarray(ecm_data[prefix + "ann_x_std"], dtype=float),
            "activation": _as_str_scalar(ecm_data, prefix + "ann_activation", default="silu"),
            "target_sum": _as_float_scalar(ecm_data, prefix + "ann_target_sum"),
            "n_layers": int(n_layers),
        }
        for i in range(int(n_layers)):
            w_key = prefix + f"ann_W_{i}"
            b_key = prefix + f"ann_b_{i}"
            if w_key not in ecm_data or b_key not in ecm_data:
                raise RuntimeError(
                    f"[HPROM-ANN] Missing ANN layer arrays for {label}: {w_key}/{b_key}."
                )
            ann_model[f"W_{i}"] = np.asarray(ecm_data[w_key], dtype=float)
            ann_model[f"b_{i}"] = np.asarray(ecm_data[b_key], dtype=float)
    else:
        rbf_model = {
            "center_ids": (
                np.asarray(ecm_data[prefix + "rbf_center_ids"], dtype=np.int64).reshape(-1)
                if (prefix + "rbf_center_ids") in ecm_data
                else np.arange(np.asarray(ecm_data[prefix + "rbf_centers"], dtype=float).shape[0], dtype=np.int64)
            ),
            "centers": np.asarray(ecm_data[prefix + "rbf_centers"], dtype=float),
            "length_scales": np.asarray(ecm_data[prefix + "rbf_length_scales"], dtype=float),
            "Alpha": np.asarray(ecm_data[prefix + "rbf_alpha"], dtype=float),
            "Beta": np.asarray(ecm_data[prefix + "rbf_beta"], dtype=float),
            "scale": np.asarray(ecm_data[prefix + "rbf_scale"], dtype=float),
            "poly_mode": int(np.ravel(ecm_data[prefix + "rbf_poly_mode"])[0]),
            "lambda_reg": (
                _as_float_scalar(ecm_data, prefix + "rbf_lambda")
                if (prefix + "rbf_lambda") in ecm_data
                else 0.0
            ),
            "n_centers": (
                _as_int_scalar(ecm_data, prefix + "rbf_n_centers")
                if (prefix + "rbf_n_centers") in ecm_data
                else int(np.asarray(ecm_data[prefix + "rbf_centers"], dtype=float).shape[0])
            ),
        }
    return {
        "target": label,
        "base_target": t,
        "z_support_full": z_support_full,
        "regressor_type": regressor_type,
        "rbf_model": rbf_model,
        "ann_model": ann_model,
        "w_fixed": w_fixed,
        "coord_train": coord_train,
        "W_train": W_train,
        "renorm_target": renorm_target,
        "clip_nonnegative": clip_nonnegative,
        "coord_label": coord_label,
    }


def _has_maw_hom_component_models(ecm_data):
    required = []
    for base in ("eps", "sig"):
        for comp in range(3):
            prefix = f"maw_{base}_{comp}_"
            required.extend([f"Z_{base}_{comp}", prefix + "regressor_type"])
    return all(k in ecm_data for k in required)


def _build_maw_hom_component_models(ecm_data, target):
    t = str(target).strip().lower()
    if t not in ("eps", "sig"):
        raise ValueError(f"Unsupported MAW homogenization component target '{target}'.")
    models = []
    for comp in range(3):
        models.append(
            _build_maw_hom_target_model(
                ecm_data,
                t,
                prefix=f"maw_{t}_{comp}_",
                z_key=f"Z_{t}_{comp}",
                target_label=f"{t}_{comp}",
            )
        )
    return models


def _build_full_to_local_map(ecm_data, n_elem_reference, n_current_elements):
    if int(n_current_elements) == int(n_elem_reference):
        return None
    if "hrom_element_full_indices" not in ecm_data:
        raise RuntimeError(
            "[HPROM-ANN] HROM mesh is active but hrom_element_full_indices is missing. "
            "Regenerate the HROM mdpa with the matching ECM/MAW file and --inplace-ecm."
        )
    full_ids = np.asarray(ecm_data["hrom_element_full_indices"], dtype=np.int64).reshape(-1)
    if full_ids.size != int(n_current_elements):
        raise RuntimeError(
            f"[HPROM-ANN] hrom_element_full_indices size {full_ids.size} "
            f"!= current element count {n_current_elements}."
        )
    return {int(full_idx): int(local_idx) for local_idx, full_idx in enumerate(full_ids.tolist())}


def _evaluate_nearest_maw_hom_support_weights(q_query, target_model):
    coord_train = target_model.get("coord_train", None)
    W_train = target_model.get("W_train", None)
    if coord_train is None or W_train is None:
        raise RuntimeError(
            f"[HPROM-ANN] MAW nearest/oracle requested for {target_model['target']}, "
            "but coord_train/W_train were not stored in the ECM file. Rebuild Stage12b "
            "with a MAW model that stores training weights."
        )
    coord_train = np.asarray(coord_train, dtype=float)
    W_train = np.asarray(W_train, dtype=float)
    q = np.asarray(q_query, dtype=float).reshape(1, -1)
    if coord_train.ndim != 2 or W_train.ndim != 2:
        raise RuntimeError(
            f"[HPROM-ANN] Invalid MAW nearest data for {target_model['target']}: "
            f"coord_train={coord_train.shape}, W_train={W_train.shape}."
        )
    if q.shape[1] != coord_train.shape[1]:
        raise RuntimeError(
            f"[HPROM-ANN] MAW nearest coordinate mismatch for {target_model['target']}: "
            f"query dim={q.shape[1]}, train dim={coord_train.shape[1]}."
        )
    scale = np.std(coord_train, axis=0)
    scale = np.where(np.abs(scale) > 1.0e-14, scale, 1.0)
    diff = (coord_train - q) / scale
    nearest_id = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))
    return W_train[:, nearest_id].reshape(-1)


def _evaluate_maw_hom_weights_current(
    q_m,
    e_vec,
    target_model,
    n_elem_reference,
    n_current_elements,
    full_to_local,
    eval_mode="model",
):
    coord_label = str(target_model.get("coord_label", "q")).strip().lower()
    if coord_label == "mu":
        q_query = np.asarray(e_vec, dtype=float).reshape(1, -1)
    else:
        q_query = np.asarray(q_m, dtype=float).reshape(1, -1)
    eval_mode = str(eval_mode or "model").strip().lower()
    if eval_mode == "oracle":
        eval_mode = "nearest"
    if eval_mode not in ("model", "nearest"):
        raise RuntimeError(
            f"[HPROM-ANN] Unsupported MAW homogenization eval mode '{eval_mode}'. "
            "Use 'model' or 'nearest'."
        )
    regressor_type = str(target_model.get("regressor_type", "rbf")).strip().lower()
    if regressor_type == "fixed_classic":
        w_support = np.asarray(target_model["w_fixed"], dtype=float).reshape(-1)
    elif eval_mode == "nearest":
        w_support = _evaluate_nearest_maw_hom_support_weights(q_query, target_model)
    elif regressor_type == "ann":
        w_support = eval_mawecm_ann(q_query, target_model["ann_model"]).reshape(-1)
    else:
        w_support = eval_mawecm_rbf(
            q_query=q_query,
            model=target_model["rbf_model"],
            clip_nonnegative=bool(target_model.get("clip_nonnegative", True)),
            renorm_target=target_model.get("renorm_target", None),
        ).reshape(-1)

    z_full = np.asarray(target_model["z_support_full"], dtype=np.int64).reshape(-1)
    if z_full.size != w_support.size:
        raise RuntimeError(
            f"[HPROM-ANN] MAW {target_model['target']} support/weight mismatch: "
            f"{z_full.size} vs {w_support.size}."
        )
    w_full = np.zeros(int(n_elem_reference), dtype=float)
    w_full[z_full] = w_support
    if full_to_local is None:
        if int(n_current_elements) != w_full.size:
            raise RuntimeError(
                f"[HPROM-ANN] Current mesh has {n_current_elements} elements but "
                f"MAW weights are full-mesh length {w_full.size}."
            )
        return w_full

    w_current = np.zeros(int(n_current_elements), dtype=float)
    for full_idx, local_idx in full_to_local.items():
        if 0 <= int(full_idx) < w_full.size:
            w_current[int(local_idx)] = w_full[int(full_idx)]
    return w_current


def LoadHpromAnnModel(
    basis_dir="stage_2_pod_rve",
    ann_data_dir="stage_7_ann_data",
    hprom_ann_dir="stage_9_hprom_ann_data",
):
    phi_p, phi_s, free_dofs, dir_dofs, eq_map, ann_model, device, include_macro = LoadPromAnnModel(
        basis_dir=basis_dir,
        ann_data_dir=ann_data_dir,
    )
    Xc, Yc = np.load(os.path.join(basis_dir, "domain_center.npy"))
    ecm = np.load(os.path.join(hprom_ann_dir, "ecm_weights_all.npz"))
    ecm_data = {k: ecm[k] for k in ecm.files}

    return (
        phi_p,
        phi_s,
        free_dofs,
        dir_dofs,
        eq_map,
        Xc,
        Yc,
        ann_model,
        device,
        ecm_data,
        include_macro,
    )


def RunHpromAnnBatchSimulation(
    parameters,
    phi_p,
    phi_s,
    free_dofs,
    ann_model,
    device,
    ecm_data,
    out_dir="stage_10_hprom_ann_results",
    strain_path=None,
    trajectory_index=None,
    relnorm_cutoff=1e-5,
    max_its=25,
    abs_res_cutoff=NEWTON_TOL_ABS,
    dq_abs_cutoff=1.0e-6,
    normalized_dq_cutoff=1.0e-4,
    max_res_for_rel_convergence=1.0e-1,
    min_rel_drop_stop=1.0e-2,
    stagnation_relnorm_gate=1.0e-4,
    max_dq_norm=0.5,
    old_stiffness_residual_cutoff=1.0e5,
    regularization=1.0e-10,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    use_old_stiffness_in_first_iteration=USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    verbose_iterations=False,
    use_fast_dirichlet_bc=True,
    eq_map_full=None,
    Xc=None,
    Yc=None,
    qp_init_mode="continuation",
    fail_on_nonconvergence=False,
    homogenization_mode="ecm_fixed",
    maw_hom_eval_mode="model",
    return_stats=False,
):
    t_wall_total_start = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)

    free_dofs_ref = np.asarray(free_dofs, dtype=np.int64).reshape(-1)
    phi_p_ref = np.asarray(phi_p, dtype=float)
    phi_s_ref = np.asarray(phi_s, dtype=float)
    if phi_p_ref.shape[0] != phi_s_ref.shape[0]:
        raise ValueError("phi_p and phi_s must have the same number of rows (n_free).")
    if phi_p_ref.shape[0] != free_dofs_ref.size:
        raise RuntimeError(
            f"[HPROM-ANN] Basis/free_dofs mismatch: phi rows={phi_p_ref.shape[0]} "
            f"!= len(free_dofs)={free_dofs_ref.size}."
        )

    n_primary = int(phi_p_ref.shape[1])
    n_secondary = int(phi_s_ref.shape[1])

    dt = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
    E_wp = np.array(strain_path, dtype=float)
    n_seg = len(E_wp) - 1
    seg_steps, _ = BuildDynamicSegmentSteps(
        E_wp,
        reference_steps=reference_steps,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=reference_amplitude,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    end_time = dt * float(n_steps_total)

    model_kratos = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model_kratos, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()

    n_total_dof, eq_id_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    vec_full_assembler = VectorizedAssembler(mp, n_total_dof, eq_id_map, log_label="HPROMANNFullSyncAssembler")
    elements = list(mp.Elements)
    entities = list(mp.Elements) + list(mp.Conditions)
    full_mesh_base = str(np.ravel(ecm_data["hrom_full_mesh_base"])[0]) if "hrom_full_mesh_base" in ecm_data else "rve_geometry"
    free_dofs, dir_dofs, basis_rows = ResolveActiveFreeDofsAndBasisRows(
        mp,
        n_total_dof,
        eq_id_map,
        free_dofs_reference=free_dofs_ref,
        eq_map_reference=eq_map_full,
        full_mesh_base=full_mesh_base,
        solver_label="HPROM-ANN",
    )
    phi_p = phi_p_ref[basis_rows, :]
    phi_s = phi_s_ref[basis_rows, :]
    Z_res, w_res_selected, using_hrom_mesh = ResolveResidualHyperReductionSelection(
        ecm_data,
        n_current_elements=len(elements),
        solver_label="HPROM-ANN",
    )
    Z_union = np.asarray(ecm_data["Z_union"], dtype=np.int64).reshape(-1) if "Z_union" in ecm_data else Z_res
    n_elem_reference = int(np.ravel(ecm_data["n_elem"])[0]) if "n_elem" in ecm_data else len(elements)
    hom_mode = str(homogenization_mode).strip().lower()
    if hom_mode in ("maw", "maw_separate"):
        hom_mode = "maw_dynamic"
    if hom_mode not in ("ecm_fixed", "maw_dynamic"):
        raise ValueError(
            f"Unsupported HPROM-ANN homogenization_mode='{homogenization_mode}'. "
            "Use 'ecm_fixed' or 'maw_dynamic'."
        )
    maw_hom_eval_mode = str(maw_hom_eval_mode or "model").strip().lower()
    if maw_hom_eval_mode == "oracle":
        maw_hom_eval_mode = "nearest"
    if maw_hom_eval_mode not in ("model", "nearest"):
        raise ValueError(
            f"Unsupported maw_hom_eval_mode='{maw_hom_eval_mode}'. Use 'model' or 'nearest'."
        )

    if Xc is None or Yc is None:
        sim._InitializeDomainCenterIfNeeded(mp)
        x0c, y0c = float(sim._x0c), float(sim._y0c)
    else:
        # Keep affine lifting center consistent with basis/training reference mesh.
        x0c, y0c = float(Xc), float(Yc)

    maw_eps_hom = None
    maw_sig_hom = None
    maw_hom_componentwise = False
    full_to_local_hom = None
    if hom_mode == "maw_dynamic":
        maw_hom_componentwise = _has_maw_hom_component_models(ecm_data)
        if maw_hom_componentwise:
            maw_eps_hom = _build_maw_hom_component_models(ecm_data, "eps")
            maw_sig_hom = _build_maw_hom_component_models(ecm_data, "sig")
        else:
            maw_eps_hom = _build_maw_hom_target_model(ecm_data, "eps")
            maw_sig_hom = _build_maw_hom_target_model(ecm_data, "sig")
        full_to_local_hom = _build_full_to_local_map(
            ecm_data,
            n_elem_reference=n_elem_reference,
            n_current_elements=len(elements),
        )
        w_eps_hom = None
        w_sig_hom = None
        using_weighted_hom = True
    else:
        w_eps_hom, w_sig_hom, using_weighted_hom = ResolveHomogenizationWeightSelection(
            ecm_data,
            n_current_elements=len(elements),
            solver_label="HPROM-ANN",
        )
    hom_reference_measure = GetReferenceIntegrationMeasureFromMesh(full_mesh_base)
    print(f"  [HPROM-ANN] Homogenization reference measure A0 (full mesh): {hom_reference_measure:.6e}")
    with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
        f.write(f"{float(hom_reference_measure):.16e}\n")
    if not using_weighted_hom:
        raise RuntimeError(
            "[HPROM-ANN] ECM homogenization weights are required (w_eps/w_sig not available)."
        )
    if hom_mode == "maw_dynamic":
        if maw_hom_componentwise:
            eps_sizes = [int(m["z_support_full"].size) for m in maw_eps_hom]
            sig_sizes = [int(m["z_support_full"].size) for m in maw_sig_hom]
            eps_regs = [str(m["regressor_type"]) for m in maw_eps_hom]
            sig_regs = [str(m["regressor_type"]) for m in maw_sig_hom]
            print(
                "  [HPROM-ANN] Using component-wise dynamic MAW-ECM homogenization "
                f"(eps |Z|={eps_sizes}, sig |Z|={sig_sizes}, "
                f"regressors eps/sig={eps_regs}/{sig_regs}, eval={maw_hom_eval_mode})."
            )
        else:
            print(
                "  [HPROM-ANN] Using dynamic MAW-ECM homogenization weights "
                f"(eps |Z|={maw_eps_hom['z_support_full'].size}, "
                f"sig |Z|={maw_sig_hom['z_support_full'].size}, "
                f"coords eps/sig={maw_eps_hom['coord_label']}/{maw_sig_hom['coord_label']}, "
                f"regressors eps/sig={maw_eps_hom['regressor_type']}/{maw_sig_hom['regressor_type']}, "
                f"eval={maw_hom_eval_mode})."
            )
    else:
        print("  [HPROM-ANN] Using ECM-weighted homogenization (w_eps / w_sig).")
    dof_x = np.zeros(n_total_dof, dtype=float)
    dof_y = np.zeros(n_total_dof, dtype=float)
    is_x_dof = np.zeros(n_total_dof, dtype=bool)
    for i, node in enumerate(mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        idx_x = int(eq_id_map[i, 0])
        idx_y = int(eq_id_map[i, 1])
        if 0 <= idx_x < n_total_dof:
            dof_x[idx_x] = xr
            dof_y[idx_x] = yr
            is_x_dof[idx_x] = True
        if 0 <= idx_y < n_total_dof:
            dof_x[idx_y] = xr
            dof_y[idx_y] = yr
            is_x_dof[idx_y] = False
    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]
    x_dir = dof_x[dir_dofs]
    y_dir = dof_y[dir_dofs]
    is_x_dir = is_x_dof[dir_dofs]

    results_eps = [np.zeros(3, dtype=float)]
    results_sig = [np.zeros(3, dtype=float)]
    q_hist = [np.zeros(n_primary, dtype=float)]

    ann_input_dim = int(ann_model.input_scaler.mean.numel())
    expected_input_dim = int(n_primary)
    if ann_input_dim != expected_input_dim:
        raise ValueError(
            f"ANN input size mismatch: model expects {ann_input_dim}, "
            f"but solver was configured for {expected_input_dim}."
        )

    def _is_finite(arr):
        return bool(np.all(np.isfinite(np.asarray(arr))))

    def _compute_affine_component(e_vec, x_loc, y_loc, is_x_loc):
        F = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (F[0, 0] - 1.0) * x_loc + F[0, 1] * y_loc
        uy = F[1, 0] * x_loc + (F[1, 1] - 1.0) * y_loc
        return np.where(is_x_loc, ux, uy)

    def _compute_affine_free_displacement(e_vec):
        return _compute_affine_component(e_vec, x_free, y_free, is_x_free)

    def _compute_affine_dirichlet_displacement(e_vec):
        return _compute_affine_component(e_vec, x_dir, y_dir, is_x_dir)

    def _capture_current_displacement_vector():
        disp_vec = np.zeros(n_total_dof, dtype=float)
        for i, node in enumerate(mp.Nodes):
            d = node.GetSolutionStepValue(KM.DISPLACEMENT)
            idx_x, idx_y = eq_id_map[i, 0], eq_id_map[i, 1]
            if idx_x < n_total_dof:
                disp_vec[idx_x] = d[0]
            if idx_y < n_total_dof:
                disp_vec[idx_y] = d[1]
        return disp_vec

    def _apply_total_free_displacement(u_total_free, base_disp_vec=None):
        if base_disp_vec is None:
            disp_vec = _capture_current_displacement_vector()
        else:
            disp_vec = np.asarray(base_disp_vec, dtype=float).copy()
        disp_vec[free_dofs] = np.asarray(u_total_free, dtype=float).reshape(-1)
        SetDisplacementFromEquationVector(disp_vec, eq_id_map, ta)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)
        return disp_vec

    def _build_ann_input(q_tensor, e_tensor):
        return q_tensor

    qp_init_mode = str(qp_init_mode).strip().lower()
    if qp_init_mode not in ("continuation", "previous", "zero", "mu_affine"):
        raise ValueError(
            f"Unsupported qp_init_mode='{qp_init_mode}'. "
            "Use one of: continuation, previous, zero, mu_affine."
        )
    qp_aff = getattr(ann_model, "qp_init_mu_affine", None)
    if qp_init_mode in ("continuation", "mu_affine") and qp_aff is None:
        raise RuntimeError(
            f"[HPROM-ANN] qp_init_mode='{qp_init_mode}' requires qm_init_mu_affine.npz "
            "in the ANN model directory."
        )

    def _evaluate_qs_and_jac(qp_vec, e_vec):
        qp_arr = np.asarray(qp_vec, dtype=float).reshape(-1)
        e_arr = np.asarray(e_vec, dtype=float).reshape(3)

        qp_tensor = torch.from_numpy(qp_arr.astype(np.float32)).unsqueeze(0).to(device)
        e_tensor = torch.from_numpy(e_arr.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            q_s_map_tensor = ann_model(_build_ann_input(qp_tensor, e_tensor))

        with torch.enable_grad():
            qp_in_vec = qp_tensor.squeeze(0).clone().detach().requires_grad_(True)

            def ann_from_qvec(q_vec):
                q_local = q_vec.unsqueeze(0)
                out = ann_model(_build_ann_input(q_local, e_tensor))
                return out.squeeze(0)

            if _torch_jacfwd is not None:
                try:
                    jac_ann = _torch_jacfwd(ann_from_qvec)(qp_in_vec)
                except Exception:
                    jac_ann = torch.autograd.functional.jacobian(ann_from_qvec, qp_in_vec)
            else:
                jac_ann = torch.autograd.functional.jacobian(ann_from_qvec, qp_in_vec)

        q_s_map = q_s_map_tensor.detach().cpu().numpy().reshape(-1)
        jac_np = jac_ann.detach().cpu().numpy()
        return q_s_map, jac_np

    def _compute_weighted_decoder_hessian(qp_vec, e_vec, output_weights):
        qp_arr = np.asarray(qp_vec, dtype=float).reshape(-1)
        e_arr = np.asarray(e_vec, dtype=float).reshape(3)
        weights_np = np.asarray(output_weights, dtype=float).reshape(-1)
        if weights_np.size != n_secondary:
            raise RuntimeError(
                "Invalid decoder-Hessian weights: "
                f"got {weights_np.size}, expected {n_secondary}."
            )
        qp_tensor = torch.from_numpy(qp_arr.astype(np.float32)).to(device)
        e_tensor = torch.from_numpy(e_arr.astype(np.float32)).unsqueeze(0).to(device)
        weights = torch.from_numpy(weights_np.astype(np.float32)).to(device)

        with torch.enable_grad():
            qp_in = qp_tensor.clone().detach().requires_grad_(True)

            def weighted_ann_output(q_vec):
                q_local = q_vec.view(1, -1)
                q_s_raw = ann_model(_build_ann_input(q_local, e_tensor)).reshape(-1)
                return torch.dot(weights, q_s_raw)

            hessian = torch.autograd.functional.hessian(
                weighted_ann_output,
                qp_in,
                vectorize=True,
            )
        return hessian.detach().cpu().numpy().reshape(n_primary, n_primary)

    def _solve_reduced_system(K_sys, rhs):
        try:
            dq_loc = np.linalg.solve(K_sys, rhs)
        except np.linalg.LinAlgError:
            dq_loc, *_ = np.linalg.lstsq(K_sys, rhs, rcond=None)
        if _is_finite(dq_loc):
            return dq_loc

        K_reg = K_sys + float(regularization) * np.eye(K_sys.shape[0], dtype=K_sys.dtype)
        try:
            dq_loc = np.linalg.solve(K_reg, rhs)
        except np.linalg.LinAlgError:
            dq_loc, *_ = np.linalg.lstsq(K_reg, rhs, rcond=None)
        return dq_loc

    def _initial_qp_guess(e_vec, q_prev, step_index):
        if qp_init_mode == "continuation" and int(step_index) > 1:
            return np.asarray(q_prev, dtype=float).copy()
        if qp_init_mode == "previous":
            return np.asarray(q_prev, dtype=float).copy()
        if qp_init_mode == "zero":
            return np.zeros(n_primary, dtype=float)
        mu_dim = int(qp_aff["mu_dim"])
        mu = np.asarray(e_vec, dtype=float).reshape(-1)[:mu_dim]
        if mu.size < mu_dim:
            raise RuntimeError(
                f"[HPROM-ANN] q_m initializer expects mu_dim={mu_dim}, got {mu.size}."
            )
        return np.concatenate([mu, np.array([1.0])]) @ np.asarray(
            qp_aff["b_aff"],
            dtype=float,
        )

    q_p = np.zeros(n_primary, dtype=float)
    q_m_scale = ann_model.input_scaler.std.detach().cpu().numpy().reshape(-1).astype(float)
    if q_m_scale.size != n_primary:
        raise RuntimeError(
            f"ANN q_m scale size mismatch: got {q_m_scale.size}, expected {n_primary}."
        )
    q_m_scale = np.maximum(np.abs(q_m_scale), 1.0e-12)
    Kr_old = None
    q0_const, J0_const = _evaluate_qs_and_jac(np.zeros(n_primary, dtype=float), np.zeros(3, dtype=float))
    a_m = np.asarray(getattr(ann_model, "a_m_np", None), dtype=float)
    if a_m.shape != (n_primary, n_primary):
        raise RuntimeError(
            f"[HPROM-ANN] Missing or invalid LS master map A_m: {a_m.shape}."
        )
    phi_master = phi_p @ a_m
    phi_p_eff = phi_master + phi_s @ J0_const
    w0_const = phi_s @ q0_const

    print(f"  [HPROM-ANN] Solving for {n_steps_total} dynamic increments...")
    print(f"  [HPROM-ANN] Active mesh elements: {len(elements)} (reference full mesh: {n_elem_reference})")
    print("  [HPROM-ANN] Manifold correction active: N(0)=0 and J(0)=0.")
    print(
        f"  [HPROM-ANN] ECM residual elements: {len(Z_res)} / {len(elements)} | "
        f"union reference size: {len(Z_union)} / {n_elem_reference}"
    )
    if using_hrom_mesh:
        print("  [HPROM-ANN] Using reduced HROM mesh residual weights (w_res_hrom).")
    print("  [HPROM-ANN] LS decoder active: u = Phi_m A_m q_m + Phi_s q_s(q_m).")
    print(f"  [HPROM-ANN] q_m initializer mode: {qp_init_mode}")
    if qp_init_mode in ("continuation", "mu_affine") and qp_aff is not None:
        print(f"  [HPROM-ANN] Using affine initializer: {qp_aff['path']}")
    t_map = 0.0
    t_assembly = 0.0
    t_projection = 0.0
    t_solve = 0.0
    t_full_sync = 0.0
    step_iters = []
    ResetHyperReductionAssemblyStats()

    for step in range(1, n_steps_total + 1):
        time_val = float(step) * float(dt)
        mp.CloneTimeStep(time_val)
        mp.ProcessInfo[KM.DELTA_TIME] = dt
        mp.ProcessInfo[KM.TIME] = time_val
        mp.ProcessInfo[KM.STEP] = step

        sim.time, sim.step, sim.end_time = time_val, step, end_time
        sim.InitializeSolutionStep()

        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        E = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]
        u_aff_free = _compute_affine_free_displacement(E)
        sim.batch_strain = E.copy()

        if use_fast_dirichlet_bc:
            disp_base_step = np.zeros(n_total_dof, dtype=float)
            disp_base_step[dir_dofs] = _compute_affine_dirichlet_displacement(E)
        else:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()

        if step == 1 or step % 100 == 0 or step == n_steps_total:
            print(f"\n[HPROM-ANN] Step {step:04d}/{n_steps_total} | E={E}")
        verbose_step = bool(verbose_iterations)

        it = 0
        res_norm_0 = None
        q_prev_np = q_p.copy()
        q_p = _initial_qp_guess(E, q_prev_np, step)
        converged = bool(max_its == 0)
        nonfinite_detected = False
        Kr_last = None
        dq_norm_prev = None
        prev_res_norm = None
        prev_q_eval = None
        prev_qs_eval = None
        plateau_count = 0
        q_step_start = q_p.copy()
        best_q = q_step_start.copy()
        best_res = np.inf
        best_rel = np.inf
        it_step_count = 0

        while it < max_its:
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1
            it_step_count += 1
            t0 = time.perf_counter()
            q_s_map, J_ann_raw = _evaluate_qs_and_jac(q_p, E)
            t_map += time.perf_counter() - t0
            q_s = q_s_map - q0_const - J0_const @ q_p
            J_ann = J_ann_raw - J0_const

            if verbose_step:
                print(f"    > q_p norm: {np.linalg.norm(q_p):.3e} | q_s norm: {np.linalg.norm(q_s):.3e}")
            if (not _is_finite(q_p)) or (not _is_finite(q_s)):
                print("  [HPROM-ANN] WARNING: non-finite reduced state detected.")
                nonfinite_detected = True
                break
            q_eval_delta = np.inf
            q_eval_delta_normalized = np.inf
            if prev_q_eval is not None and prev_q_eval.shape == q_p.shape:
                q_eval_change = q_p - prev_q_eval
                q_eval_delta = float(np.linalg.norm(q_eval_change))
                q_eval_delta_normalized = float(
                    np.linalg.norm(q_eval_change / q_m_scale)
                )
            prev_q_eval = q_p.copy()
            q_s_delta = np.inf
            if prev_qs_eval is not None and prev_qs_eval.shape == q_s.shape:
                q_s_delta = float(np.linalg.norm(q_s - prev_qs_eval))
            prev_qs_eval = q_s.copy()

            if J_ann.shape != (n_secondary, n_primary):
                raise RuntimeError(
                    f"Invalid ANN Jacobian shape {J_ann.shape}; expected ({n_secondary}, {n_primary})."
                )
            if not _is_finite(J_ann):
                print("  [HPROM-ANN] WARNING: non-finite ANN Jacobian detected.")
                nonfinite_detected = True
                break

            u_fluc = w0_const + phi_p_eff @ q_p + phi_s @ q_s
            if not _is_finite(u_fluc):
                print("  [HPROM-ANN] WARNING: non-finite reconstructed displacement detected.")
                nonfinite_detected = True
                break
            u_free = u_aff_free + u_fluc

            J_manifold = phi_p_eff + phi_s @ J_ann
            if not _is_finite(J_manifold):
                print("  [HPROM-ANN] WARNING: non-finite manifold Jacobian detected.")
                nonfinite_detected = True
                break

            u_eq_curr = _apply_total_free_displacement(u_free, base_disp_vec=disp_base_step)

            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            t0 = time.perf_counter()
            K_hp, rhs_hp = AssembleHyperReducedSystem(
                mp, n_total_dof, elements, Z_res, w_res_selected, u_eq=u_eq_curr
            )
            t_assembly += time.perf_counter() - t0
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            r_full = rhs_hp[free_dofs]
            K_free_sparse = K_hp[free_dofs][:, free_dofs]
            if (not _is_finite(r_full)) or (not _is_finite(K_free_sparse.data)):
                print("  [HPROM-ANN] WARNING: non-finite full residual/stiffness detected.")
                nonfinite_detected = True
                break

            t0 = time.perf_counter()
            KJ = K_free_sparse @ J_manifold
            r_r = J_manifold.T @ r_full
            K_std = J_manifold.T @ KJ
            # Exact nonlinear-manifold Newton tangent. Since rhs=-f_int,
            # d(J^T rhs)/dq = H_u:rhs - J^T K J, hence the additive
            # Newton matrix is J^T K J - H_u:rhs.
            curvature_weights = phi_s.T @ r_full
            K_curv = _compute_weighted_decoder_hessian(q_p, E, curvature_weights)
            K_curv = 0.5 * (K_curv + K_curv.T)
            K_r = K_std - K_curv
            t_projection += time.perf_counter() - t0
            Kr_last = K_r
            if (not _is_finite(r_r)) or (not _is_finite(K_r)):
                print("  [HPROM-ANN] WARNING: non-finite reduced residual/stiffness detected.")
                nonfinite_detected = True
                break

            res_norm = float(np.linalg.norm(r_r))
            if not np.isfinite(res_norm):
                print("  [HPROM-ANN] WARNING: non-finite reduced residual norm detected.")
                nonfinite_detected = True
                break
            if res_norm_0 is None:
                res_norm_0 = max(res_norm, 1e-30)
            rel_res = res_norm / (res_norm_0 + 1e-12)
            if res_norm < best_res:
                best_res = float(res_norm)
                best_q = q_p.copy()
                best_rel = float(rel_res)
            update_text = ""
            if np.isfinite(q_eval_delta_normalized):
                update_text = f", ||D_q^-1 dq_m|| = {q_eval_delta_normalized:.3e}"
            print(f"  > It {it:02d}: ||R_r|| = {res_norm:.3e}, rel = {rel_res:.3e}{update_text}")

            if res_norm < float(abs_res_cutoff):
                print(f"  > Converged in {it} iterations.")
                converged = True
                break
            if (
                it > 0
                and np.isfinite(q_eval_delta_normalized)
                and q_eval_delta_normalized < float(normalized_dq_cutoff)
            ):
                print(
                    f"  > Converged in {it} iterations "
                    f"(normalized q_m update={q_eval_delta_normalized:.3e} "
                    f"< {float(normalized_dq_cutoff):.3e})."
                )
                converged = True
                break
            if (
                dq_norm_prev is not None
                and dq_norm_prev < float(dq_abs_cutoff)
                and rel_res < float(relnorm_cutoff)
                and res_norm < float(max_res_for_rel_convergence)
            ):
                print(f"  > Converged in {it} iterations (small update + reduced residual).")
                converged = True
                break
            if prev_res_norm is not None:
                rel_drop = abs(prev_res_norm - res_norm) / max(prev_res_norm, 1e-30)
                q_tol = 20.0 * float(dq_abs_cutoff)
                q_is_flat = (
                    np.isfinite(q_eval_delta)
                    and np.isfinite(q_s_delta)
                    and (q_eval_delta < q_tol)
                    and (q_s_delta < q_tol)
                )
                if (
                    it >= 2
                    and q_is_flat
                    and rel_res < max(float(relnorm_cutoff), 0.5 * float(stagnation_relnorm_gate))
                ):
                    print(
                        f"  > Converged in {it} iterations (frozen q-state: "
                        f"q_p_delta={q_eval_delta:.3e}, q_s_delta={q_s_delta:.3e}, rel={rel_res:.3e})."
                    )
                    converged = True
                    break
                if rel_drop < float(min_rel_drop_stop) and q_is_flat:
                    plateau_count += 1
                else:
                    plateau_count = 0
                if (
                    rel_drop < float(min_rel_drop_stop)
                    and rel_res < float(stagnation_relnorm_gate)
                    and res_norm < float(max_res_for_rel_convergence)
                ):
                    print(
                        f"  > Converged in {it} iterations (stagnation criterion: rel_drop={rel_drop:.3e})."
                    )
                    converged = True
                    break
                if (
                    plateau_count >= 3
                    and rel_res < max(3.0 * float(relnorm_cutoff), float(stagnation_relnorm_gate))
                    and (dq_norm_prev is None or dq_norm_prev < 5.0 * float(dq_abs_cutoff))
                ):
                    print(
                        f"  > Converged in {it} iterations (flat residual plateau: "
                        f"rel_drop={rel_drop:.3e}, q_p_delta={q_eval_delta:.3e}, q_s_delta={q_s_delta:.3e})."
                    )
                    converged = True
                    break
            prev_res_norm = float(res_norm)

            K_solve = K_r
            used_old = False
            if (
                it == 0
                and use_old_stiffness_in_first_iteration
                and Kr_old is not None
                and Kr_old.shape == K_r.shape
                and res_norm < float(old_stiffness_residual_cutoff)
            ):
                K_solve = Kr_old
                used_old = True

            t0 = time.perf_counter()
            dq_p = _solve_reduced_system(K_solve, r_r)
            if used_old and not _is_finite(dq_p):
                dq_p = _solve_reduced_system(K_r, r_r)
                used_old = False
            t_solve += time.perf_counter() - t0
            if not _is_finite(dq_p):
                print("  [HPROM-ANN] WARNING: non-finite reduced update detected.")
                nonfinite_detected = True
                break

            dq_norm = float(np.linalg.norm(dq_p))
            if dq_norm > float(max_dq_norm) and dq_norm > 0.0:
                scale = float(max_dq_norm) / dq_norm
                dq_p *= scale
                dq_norm = float(np.linalg.norm(dq_p))
                print(f"    > large reduced update clipped with scale={scale:.3e}")

            q_trial = q_p + dq_p
            if not _is_finite(q_trial):
                print("  [HPROM-ANN] WARNING: non-finite reduced state update detected.")
                nonfinite_detected = True
                break
            q_p = q_trial
            dq_norm_prev = dq_norm
            if used_old:
                print("    > using previous reduced stiffness (K_old) at first iteration")
            it += 1

        if not converged:
            quasi_converged = (
                np.isfinite(best_res)
                and np.isfinite(best_rel)
                and (best_rel < float(relnorm_cutoff))
                and (best_res < float(max_res_for_rel_convergence))
            )
            plateau_quasi = (
                np.isfinite(best_res)
                and np.isfinite(best_rel)
                and plateau_count >= 3
                and (best_rel < max(3.0 * float(relnorm_cutoff), float(stagnation_relnorm_gate)))
            )
            if quasi_converged or plateau_quasi:
                q_p = best_q.copy()
                converged = True
                if quasi_converged:
                    print(
                        "  [HPROM-ANN] Step accepted as quasi-converged: "
                        f"best ||R_r||={best_res:.3e}, rel={best_rel:.3e}."
                    )
                else:
                    print(
                        "  [HPROM-ANN] Step accepted as plateau-converged: "
                        f"best ||R_r||={best_res:.3e}, rel={best_rel:.3e}, "
                        f"plateau_count={plateau_count}."
                    )
                Kr_old = None
            else:
                msg = f"[HPROM-ANN] Step {step} did not converge in {max_its} iterations."
                if bool(fail_on_nonconvergence):
                    raise RuntimeError(
                        msg + f" Best ||R_r||={best_res:.3e}, rel={best_rel:.3e}."
                    )
                print(f"  [HPROM-ANN] WARNING: {msg}")
                if nonfinite_detected:
                    print("  [HPROM-ANN] WARNING: non-finite state encountered; rolling back to best finite iterate.")
                if np.isfinite(best_res):
                    q_p = best_q.copy()
                    print(f"  [HPROM-ANN] Using best finite iterate with ||R_r||={best_res:.3e}.")
                else:
                    q_p = q_step_start.copy()
                    print("  [HPROM-ANN] Reverting to previous-step reduced state.")
                Kr_old = None
        elif Kr_last is not None and _is_finite(Kr_last):
            Kr_old = Kr_last.copy()
        else:
            Kr_old = None

        q_s_final_map, _ = _evaluate_qs_and_jac(q_p, E)
        q_s_final = q_s_final_map - q0_const - J0_const @ q_p
        u_fluc_final = w0_const + phi_p_eff @ q_p + phi_s @ q_s_final
        if not _is_finite(u_fluc_final):
            q_p = q_step_start.copy()
            q_s_final_map, _ = _evaluate_qs_and_jac(q_p, E)
            q_s_final = q_s_final_map - q0_const - J0_const @ q_p
            u_fluc_final = w0_const + phi_p_eff @ q_p + phi_s @ q_s_final
        if not _is_finite(u_fluc_final):
            raise RuntimeError("HPROM-ANN accepted state is non-finite after rollback.")

        if not use_fast_dirichlet_bc:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()
        _apply_total_free_displacement(u_aff_free + u_fluc_final, base_disp_vec=disp_base_step)

        InitializeNonLinearIteration(entities, mp.ProcessInfo)
        u_curr = _capture_current_displacement_vector()
        t0 = time.perf_counter()
        _, _ = vec_full_assembler.Assemble(u_curr)
        t_full_sync += time.perf_counter() - t0
        FinalizeNonLinearIteration(entities, mp.ProcessInfo)

        if hom_mode == "maw_dynamic":
            if maw_hom_componentwise:
                w_eps_step = np.vstack(
                    [
                        _evaluate_maw_hom_weights_current(
                            q_p,
                            E,
                            model,
                            n_elem_reference=n_elem_reference,
                            n_current_elements=len(elements),
                            full_to_local=full_to_local_hom,
                            eval_mode=maw_hom_eval_mode,
                        )
                        for model in maw_eps_hom
                    ]
                )
                w_sig_step = np.vstack(
                    [
                        _evaluate_maw_hom_weights_current(
                            q_p,
                            E,
                            model,
                            n_elem_reference=n_elem_reference,
                            n_current_elements=len(elements),
                            full_to_local=full_to_local_hom,
                            eval_mode=maw_hom_eval_mode,
                        )
                        for model in maw_sig_hom
                    ]
                )
            else:
                w_eps_step = _evaluate_maw_hom_weights_current(
                    q_p,
                    E,
                    maw_eps_hom,
                    n_elem_reference=n_elem_reference,
                    n_current_elements=len(elements),
                    full_to_local=full_to_local_hom,
                    eval_mode=maw_hom_eval_mode,
                )
                w_sig_step = _evaluate_maw_hom_weights_current(
                    q_p,
                    E,
                    maw_sig_hom,
                    n_elem_reference=n_elem_reference,
                    n_current_elements=len(elements),
                    full_to_local=full_to_local_hom,
                    eval_mode=maw_hom_eval_mode,
                )
            hom_eps, hom_sig = CalculateHomogenizedFromAssemblerWithElementWeights(
                vec_full_assembler,
                w_eps=w_eps_step,
                w_sig=w_sig_step,
                reference_measure=hom_reference_measure,
            )
        elif using_weighted_hom:
            hom_eps, hom_sig = CalculateHomogenizedFromAssemblerWithElementWeights(
                vec_full_assembler,
                w_eps=w_eps_hom,
                w_sig=w_sig_hom,
                reference_measure=hom_reference_measure,
            )
        else:
            hom_eps, hom_sig = CalculateHomogenizedStressAndStrainKratosReference(
                mp,
                reference_area_e=np.asarray(vec_full_assembler.area_e, dtype=float),
                reference_measure=hom_reference_measure,
            )
        sim.FinalizeSolutionStep()
        step_iters.append(int(it_step_count))

        q_hist.append(q_p.copy())
        results_eps.append(hom_eps)
        results_sig.append(hom_sig)

    sim.Finalize()
    t_wall_total = time.perf_counter() - t_wall_total_start
    t_accounted = t_map + t_assembly + t_projection + t_solve + t_full_sync
    t_other = max(t_wall_total - t_accounted, 0.0)
    iters = np.asarray(step_iters, dtype=float)
    hr_stats = GetHyperReductionAssemblyStats()
    timing_stats = {
        "n_steps": int(len(step_iters)),
        "newton_iters_total": int(np.sum(iters)) if iters.size else 0,
        "newton_iters_mean_per_step": float(np.mean(iters)) if iters.size else 0.0,
        "newton_iters_max_per_step": int(np.max(iters)) if iters.size else 0,
        "map": float(t_map),
        "assembly": float(t_assembly),
        "projection": float(t_projection),
        "solve": float(t_solve),
        "full_sync": float(t_full_sync),
        "accounted": float(t_accounted),
        "other": float(t_other),
        "total": float(t_wall_total),
    }
    for k, v in hr_stats.items():
        timing_stats[f"hr_{k}"] = float(v)
    if timing_stats["newton_iters_total"] > 0:
        timing_stats["mean_time_per_newton_iter"] = float(
            timing_stats["total"] / float(timing_stats["newton_iters_total"])
        )
        timing_stats["mean_map_time_per_newton_iter"] = float(
            timing_stats["map"] / float(timing_stats["newton_iters_total"])
        )
    else:
        timing_stats["mean_time_per_newton_iter"] = 0.0
        timing_stats["mean_map_time_per_newton_iter"] = 0.0

    print(
        f"\n[HPROM-ANN] Timing: map={t_map:.3f}s, assembly={t_assembly:.3f}s, "
        f"projection={t_projection:.3f}s, solve={t_solve:.3f}s, full_sync={t_full_sync:.3f}s, "
        f"accounted={t_accounted:.3f}s, other={t_other:.3f}s, total={t_wall_total:.3f}s, "
        f"iters(total={timing_stats['newton_iters_total']}, mean/step={timing_stats['newton_iters_mean_per_step']:.2f})"
    )
    print(
        f"  [HPROM-ANN] Hyper-assembly stats: calls={int(hr_stats['calls_total'])}, "
        f"vec={int(hr_stats['vectorized_calls'])}, local={int(hr_stats['local_calls'])}, "
        f"vec_fail={int(hr_stats['vectorized_failures'])}, "
        f"cache(rebuild={int(hr_stats['cache_rebuilds'])}, reuse={int(hr_stats['cache_reuses'])}), "
        f"u_eq_none={int(hr_stats['calls_without_u_eq'])}, "
        f"mean_vec={1e3*hr_stats['mean_vectorized_time_per_call']:.3f}ms, "
        f"mean_local={1e3*hr_stats['mean_local_time_per_call']:.3f}ms"
    )

    np.savez(os.path.join(out_dir, "hprom_ann_timing_stats.npz"), **{
        k: np.array([v], dtype=float) for k, v in timing_stats.items()
    })
    with open(os.path.join(out_dir, "hprom_ann_timing_stats.txt"), "w", encoding="utf-8") as f:
        for k, v in timing_stats.items():
            f.write(f"{k}={v}\n")

    tag = f"trajectory_{trajectory_index}" if trajectory_index is not None else "hprom_ann_run"
    np.save(os.path.join(out_dir, f"{tag}_q_p.npy"), np.stack(q_hist))
    np.save(os.path.join(out_dir, f"{tag}_strain.npy"), np.stack(results_eps))
    np.save(os.path.join(out_dir, f"{tag}_stress.npy"), np.stack(results_sig))

    if return_stats:
        return np.array(results_eps), np.array(results_sig), timing_stats
    return np.array(results_eps), np.array(results_sig)
