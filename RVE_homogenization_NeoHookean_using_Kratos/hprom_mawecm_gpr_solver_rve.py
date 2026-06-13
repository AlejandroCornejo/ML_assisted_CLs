#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPROM-MAWECM-GPR solver:
  - Hyper-reduced assembly with dynamic MAW-ECM residual weights
  - Optional dynamic MAW-ECM homogenization weights (eps/sig)
  - Nonlinear GPR manifold projection with tangent Jacobian
"""

import os
import sys
import numpy as np
import time
from scipy.sparse import coo_matrix

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
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
    USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    NEWTON_TOL_ABS,
)
from hprom_solver_rve import (
    AssembleHyperReducedSystem,
    ResolveActiveFreeDofsAndBasisRows,
    CalculateHomogenizedFromAssemblerWithElementWeights,
    GetReferenceIntegrationMeasureFromMesh,
)
from prom_gpr_solver_rve import LoadPromGprModel
from sparse_gp_manifold_model import evaluate_sparse_gp_map_and_jacobian_qp
from mawecm_rbf_weights import eval_mawecm_rbf, eval_mawecm_rbf_with_jacobian


WEIGHT_ZERO_TOL = 1.0e-14


def _as_int_scalar(arr, key):
    return int(np.ravel(arr[key])[0])


def _as_float_scalar(arr, key):
    return float(np.ravel(arr[key])[0])


def _build_maw_target_model(ecm_data, target, required=True):
    t = str(target).strip().lower()
    z_key = {"res": "Z_res", "eps": "Z_eps", "sig": "Z_sig"}[t]
    prefix = f"maw_{t}_"
    req = [
        prefix + "rbf_centers",
        prefix + "rbf_length_scales",
        prefix + "rbf_alpha",
        prefix + "rbf_beta",
        prefix + "rbf_scale",
        prefix + "rbf_poly_mode",
    ]
    missing = [k for k in req if k not in ecm_data]
    if missing:
        if bool(required):
            raise RuntimeError(f"[HPROM-MAWECM-GPR] Missing MAW keys for target='{t}': {missing}")
        return None

    z_support_full = np.asarray(ecm_data[z_key], dtype=np.int64).reshape(-1)
    if z_support_full.size == 0:
        if bool(required):
            raise RuntimeError(f"[HPROM-MAWECM-GPR] Empty support for target='{t}'.")
        return None

    renorm_target = None
    rk = prefix + "renorm_target"
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
    if rk in ecm_data and renorm_enabled:
        renorm_target = float(np.ravel(ecm_data[rk])[0])

    model = {
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
        "lambda_reg": _as_float_scalar(ecm_data, prefix + "rbf_lambda")
        if (prefix + "rbf_lambda") in ecm_data
        else 0.0,
        "n_centers": _as_int_scalar(ecm_data, prefix + "rbf_n_centers")
        if (prefix + "rbf_n_centers") in ecm_data
        else int(np.asarray(ecm_data[prefix + "rbf_centers"], dtype=float).shape[0]),
    }

    return {
        "target": t,
        "z_support_full": z_support_full,
        "rbf_model": model,
        "renorm_target": renorm_target,
        "clip_nonnegative": clip_nonnegative,
    }


def _build_full_to_local_map(ecm_data, n_elem_ref, n_cur):
    if int(n_cur) == int(n_elem_ref):
        return None
    if "hrom_element_full_indices" not in ecm_data:
        raise RuntimeError(
            "[HPROM-MAWECM-GPR] Current mesh differs from reference full mesh and "
            "hrom_element_full_indices is unavailable."
        )
    full_ids = np.asarray(ecm_data["hrom_element_full_indices"], dtype=np.int64).reshape(-1)
    if full_ids.size != int(n_cur):
        raise RuntimeError(
            f"[HPROM-MAWECM-GPR] hrom_element_full_indices size {full_ids.size} "
            f"!= current mesh elements {n_cur}."
        )
    out = {}
    for loc, full in enumerate(full_ids.tolist()):
        out[int(full)] = int(loc)
    return out


def _map_support_to_current(z_support_full, full_to_local):
    z_full = np.asarray(z_support_full, dtype=np.int64).reshape(-1)
    if full_to_local is None:
        return z_full.copy(), np.arange(z_full.size, dtype=np.int64), np.zeros(0, dtype=np.int64)

    loc = []
    pos = []
    miss = []
    for i, full_idx in enumerate(z_full.tolist()):
        local_idx = full_to_local.get(int(full_idx))
        if local_idx is None:
            miss.append(int(full_idx))
        else:
            loc.append(int(local_idx))
            pos.append(int(i))
    if not loc:
        raise RuntimeError("[HPROM-MAWECM-GPR] Support mapping to current mesh is empty.")
    return (
        np.asarray(loc, dtype=np.int64),
        np.asarray(pos, dtype=np.int64),
        np.asarray(miss, dtype=np.int64),
    )


def _evaluate_support_weights(q_p, target_model):
    q_vec = np.asarray(q_p, dtype=float).reshape(1, -1)
    renorm_target = target_model.get("renorm_target", None)
    w = eval_mawecm_rbf(
        q_query=q_vec,
        model=target_model["rbf_model"],
        clip_nonnegative=bool(target_model.get("clip_nonnegative", True)),
        renorm_target=renorm_target,
    )
    return np.asarray(w, dtype=float).reshape(-1)


def _evaluate_support_weights_and_jacobian(q_p, target_model):
    q_vec = np.asarray(q_p, dtype=float).reshape(1, -1)
    renorm_target = target_model.get("renorm_target", None)
    w, dw = eval_mawecm_rbf_with_jacobian(
        q_query=q_vec,
        model=target_model["rbf_model"],
        clip_nonnegative=bool(target_model.get("clip_nonnegative", True)),
        renorm_target=renorm_target,
    )
    return (
        np.asarray(w, dtype=float).reshape(-1),
        np.asarray(dw[:, 0, :], dtype=float),
    )


def _support_weights_to_full(z_support_full, w_support, n_elem_ref):
    z = np.asarray(z_support_full, dtype=np.int64).reshape(-1)
    ws = np.asarray(w_support, dtype=float).reshape(-1)
    if z.size != ws.size:
        raise RuntimeError(
            f"[HPROM-MAWECM-GPR] Support/weights mismatch: |Z|={z.size}, |w|={ws.size}."
        )
    out = np.zeros(int(n_elem_ref), dtype=float)
    out[z] = ws
    return out


def _project_full_weights_to_current(w_full, n_cur, full_to_local):
    wf = np.asarray(w_full, dtype=float).reshape(-1)
    if full_to_local is None:
        if wf.size != int(n_cur):
            raise RuntimeError(
                f"[HPROM-MAWECM-GPR] Full weight length {wf.size} != current mesh elements {n_cur}."
            )
        return wf.copy()
    out = np.zeros(int(n_cur), dtype=float)
    for full_idx, local_idx in full_to_local.items():
        if 0 <= int(full_idx) < wf.size:
            out[int(local_idx)] = wf[int(full_idx)]
    return out


def _build_free_dof_index_map(n_dof, free_dofs):
    idx = -np.ones(int(n_dof), dtype=np.int64)
    f = np.asarray(free_dofs, dtype=np.int64).reshape(-1)
    idx[f] = np.arange(f.size, dtype=np.int64)
    return idx


def _prepare_unit_rhs_scatter(assembler_unit, free_dof_index_map):
    ne = int(assembler_unit.n_elems)
    ndl = int(assembler_unit.n_local_dof)
    eq_ids = np.asarray(assembler_unit.local_eq_ids, dtype=np.int64).reshape(ne, ndl)
    free_ids = np.asarray(free_dof_index_map, dtype=np.int64)[eq_ids]
    valid = free_ids >= 0
    if not np.any(valid):
        return {
            "ne": ne,
            "ndl": ndl,
            "valid": valid,
            "row_idx": np.zeros(0, dtype=np.int64),
            "col_idx": np.zeros(0, dtype=np.int64),
        }
    row_idx = free_ids[valid]
    col_idx = np.broadcast_to(np.arange(ne, dtype=np.int64)[:, None], (ne, ndl))[valid]
    return {
        "ne": ne,
        "ndl": ndl,
        "valid": valid,
        "row_idx": row_idx,
        "col_idx": col_idx,
    }


def _scatter_rhs_loc_by_element_free(rhs_loc, n_free, scatter_cache):
    ne = int(scatter_cache["ne"])
    ndl = int(scatter_cache["ndl"])
    rhs = np.asarray(rhs_loc, dtype=float).reshape(ne, ndl)
    row_idx = np.asarray(scatter_cache["row_idx"], dtype=np.int64)
    col_idx = np.asarray(scatter_cache["col_idx"], dtype=np.int64)
    if row_idx.size == 0:
        return np.zeros((int(n_free), ne), dtype=float)
    out = np.zeros((int(n_free), ne), dtype=float)
    np.add.at(out, (row_idx, col_idx), rhs[scatter_cache["valid"]])
    return out


class _DynamicWeightedResidualAssembler:
    def __init__(self, mp, n_dof, eq_map, elements, selected_indices):
        self.n_dof = int(n_dof)
        self.elem_idx = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        if self.elem_idx.size == 0:
            raise RuntimeError("[HPROM-MAWECM-GPR] Empty residual support set for dynamic assembly.")

        selected_elements = [elements[int(i)] for i in self.elem_idx.tolist()]
        self._assembler = VectorizedAssembler(
            mp,
            int(n_dof),
            eq_map,
            elements=selected_elements,
            element_scales=np.ones(self.elem_idx.size, dtype=float),
            log_label="HPROMMAWResidualAssembler",
        )
        self.n_sel = int(self.elem_idx.size)
        self._rhs = np.zeros(int(n_dof), dtype=float)

    def assemble(self, u_eq, elem_weights):
        w = np.asarray(elem_weights, dtype=float).reshape(-1)
        if w.size != self.n_sel:
            raise RuntimeError(
                f"[HPROM-MAWECM-GPR] residual weight length {w.size} != support size {self.n_sel}."
            )

        self._assembler.Assemble(np.asarray(u_eq, dtype=float).reshape(-1))

        rhs_elem = -self._assembler._f_int.reshape(self.n_sel, -1)
        self._rhs.fill(0.0)
        np.add.at(
            self._rhs,
            self._assembler.rows_R,
            (w[:, None] * rhs_elem).reshape(-1),
        )

        k_vals = (w[:, None, None] * self._assembler._K_total).reshape(-1)
        K = coo_matrix(
            (k_vals, (self._assembler.rows_K, self._assembler.cols_K)),
            shape=(self.n_dof, self.n_dof),
        ).tocsr()
        return K, self._rhs


def LoadHpromMawEcmGprModel(
    basis_dir="stage_2_pod_rve",
    gpr_data_dir="stage_7_gpr_data_ls",
    hprom_mawecm_gpr_dir="stage_12_hprom_mawecm_gpr_data_ls",
):
    phi_p, phi_s, free_dofs, dir_dofs, eq_map, gpr_model, include_macro = LoadPromGprModel(
        basis_dir=basis_dir, gpr_data_dir=gpr_data_dir
    )
    Xc, Yc = np.load(os.path.join(basis_dir, "domain_center.npy"))

    ecm = np.load(os.path.join(hprom_mawecm_gpr_dir, "ecm_weights_all.npz"))
    ecm_data = {k: ecm[k] for k in ecm.files}

    return (
        phi_p,
        phi_s,
        free_dofs,
        dir_dofs,
        eq_map,
        Xc,
        Yc,
        gpr_model,
        ecm_data,
        include_macro,
    )


def RunHpromMawEcmGprBatchSimulation(
    parameters,
    phi_p,
    phi_s,
    free_dofs,
    gpr_model,
    ecm_data,
    out_dir="stage_12_hprom_mawecm_gpr_ls_results",
    strain_path=None,
    trajectory_index=None,
    relnorm_cutoff=1e-5,
    max_its=25,
    abs_res_cutoff=NEWTON_TOL_ABS,
    dq_abs_cutoff=1.0e-6,
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
    return_stats=False,
    qp_init_mode="continuation",
    dynamic_residual_weights=True,
    include_weight_tangent=True,
    dynamic_homogenization_weights=False,
    evaluate_homogenization=True,
    homogenization_mode="full_fom",
    fail_on_nonconvergence=True,
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
            f"[HPROM-MAWECM-GPR] Basis/free_dofs mismatch: phi rows={phi_p_ref.shape[0]} "
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
    vec_full_assembler = VectorizedAssembler(
        mp, n_total_dof, eq_id_map, log_label="HPROMMAWFullSyncAssembler"
    )
    elements = list(mp.Elements)
    entities = list(mp.Elements) + list(mp.Conditions)
    full_mesh_base = (
        str(np.ravel(ecm_data["hrom_full_mesh_base"])[0])
        if "hrom_full_mesh_base" in ecm_data
        else "rve_geometry"
    )
    free_dofs, dir_dofs, basis_rows = ResolveActiveFreeDofsAndBasisRows(
        mp,
        n_total_dof,
        eq_id_map,
        free_dofs_reference=free_dofs_ref,
        eq_map_reference=eq_map_full,
        full_mesh_base=full_mesh_base,
        solver_label="HPROM-MAWECM-GPR",
    )
    phi_p = phi_p_ref[basis_rows, :]
    phi_s = phi_s_ref[basis_rows, :]

    maw_res = _build_maw_target_model(ecm_data, "res", required=True)
    maw_eps = None
    maw_sig = None

    n_elem_current = int(len(elements))
    n_elem_reference = int(np.ravel(ecm_data["n_elem"])[0]) if "n_elem" in ecm_data else len(elements)
    full_to_local = _build_full_to_local_map(ecm_data, n_elem_reference, n_elem_current)
    using_hrom_mesh = full_to_local is not None

    z_res_local, z_res_pos, miss_res = _map_support_to_current(maw_res["z_support_full"], full_to_local)
    z_eps_pos = None
    z_sig_pos = None

    if miss_res.size:
        print(
            f"[HPROM-MAWECM-GPR] WARNING: residual support lost {miss_res.size} full-mesh indices "
            "when mapping to current mesh."
        )

    dyn_res_assembler = _DynamicWeightedResidualAssembler(
        mp=mp,
        n_dof=n_total_dof,
        eq_map=eq_id_map,
        elements=elements,
        selected_indices=z_res_local,
    )
    use_weight_tangent = bool(dynamic_residual_weights) and bool(include_weight_tangent)
    unit_rhs_scatter = (
        _prepare_unit_rhs_scatter(
            assembler_unit=dyn_res_assembler._assembler,
            free_dof_index_map=_build_free_dof_index_map(n_total_dof, free_dofs),
        )
        if use_weight_tangent
        else None
    )

    w_res_anchor = np.asarray(ecm_data["w_res"], dtype=float).reshape(-1) if "w_res" in ecm_data else None
    if w_res_anchor is None or w_res_anchor.size != maw_res["z_support_full"].size:
        w_res_anchor_local = np.ones(z_res_local.size, dtype=float)
    else:
        w_res_anchor_local = np.asarray(w_res_anchor[z_res_pos], dtype=float)

    if Xc is None or Yc is None:
        sim._InitializeDomainCenterIfNeeded(mp)
        x0c, y0c = float(sim._x0c), float(sim._y0c)
    else:
        # Keep affine lifting center consistent with basis/training reference mesh.
        x0c, y0c = float(Xc), float(Yc)
    use_dynamic_hom_weights = False
    w_eps_anchor_full = None
    w_sig_anchor_full = None
    w_eps_fixed_cur = None
    w_sig_fixed_cur = None
    hom_reference_measure = None
    hom_mode = str(homogenization_mode).strip().lower()
    hom_mode_alias = {
        "full": "full_fom",
        "full_mesh": "full_fom",
        "full_fom": "full_fom",
        "full_reference": "full_fom",
        "ecm": "ecm_fixed",
        "ecm_fixed": "ecm_fixed",
        "fixed": "ecm_fixed",
        "fixed_classic": "ecm_fixed",
        "maw": "maw_dynamic",
        "maw_dynamic": "maw_dynamic",
        "dynamic_maw": "maw_dynamic",
    }
    if hom_mode not in hom_mode_alias:
        raise ValueError(
            f"Unsupported homogenization_mode='{homogenization_mode}'. "
            "Use one of: full_fom, ecm_fixed, maw_dynamic."
        )
    hom_mode = hom_mode_alias[hom_mode]
    if bool(dynamic_homogenization_weights) and hom_mode != "maw_dynamic":
        hom_mode = "maw_dynamic"

    if bool(evaluate_homogenization):
        w_eps_anchor_full = (
            np.asarray(ecm_data["w_eps_full"], dtype=float).reshape(-1)
            if "w_eps_full" in ecm_data
            else None
        )
        w_sig_anchor_full = (
            np.asarray(ecm_data["w_sig_full"], dtype=float).reshape(-1)
            if "w_sig_full" in ecm_data
            else None
        )
        if w_eps_anchor_full is None or w_eps_anchor_full.size != n_elem_reference:
            w_eps_anchor_full = np.zeros(n_elem_reference, dtype=float)
        if w_sig_anchor_full is None or w_sig_anchor_full.size != n_elem_reference:
            w_sig_anchor_full = np.zeros(n_elem_reference, dtype=float)

        hom_reference_measure = GetReferenceIntegrationMeasureFromMesh(full_mesh_base)
        print(
            f"  [HPROM-MAWECM-GPR] Homogenization reference measure A0 (full mesh): "
            f"{hom_reference_measure:.6e}"
        )
        with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
            f.write(f"{float(hom_reference_measure):.16e}\n")

        if hom_mode == "full_fom":
            if using_hrom_mesh:
                raise RuntimeError(
                    "[HPROM-MAWECM-GPR] homogenization_mode='full_fom' requires the full mesh. "
                    "Use ecm_fixed or maw_dynamic for HROM mdpa runs."
                )
            use_dynamic_hom_weights = False
            w_eps_fixed_cur = None
            w_sig_fixed_cur = None
            print(
                "  [HPROM-MAWECM-GPR] Homogenization mode: full_fom "
                "(unweighted full mesh, no homogenization hyper-reduction)."
            )
        elif hom_mode == "ecm_fixed":
            def _resolve_fixed_hom_weight(key_hrom, key_full):
                if key_hrom in ecm_data:
                    w = np.asarray(ecm_data[key_hrom], dtype=float).reshape(-1)
                    if w.size == n_elem_current:
                        return w
                if key_full in ecm_data:
                    w_full = np.asarray(ecm_data[key_full], dtype=float).reshape(-1)
                    if w_full.size == n_elem_reference:
                        return _project_full_weights_to_current(w_full, n_elem_current, full_to_local)
                return None

            w_eps_fixed_cur = _resolve_fixed_hom_weight("w_eps_hrom", "w_eps_full")
            w_sig_fixed_cur = _resolve_fixed_hom_weight("w_sig_hrom", "w_sig_full")
            if w_eps_fixed_cur is None or w_sig_fixed_cur is None:
                raise RuntimeError(
                    "[HPROM-MAWECM-GPR] homogenization_mode='ecm_fixed' requested, "
                    "but fixed eps/sig homogenization weights are unavailable."
                )
            use_dynamic_hom_weights = False
            print(
                "  [HPROM-MAWECM-GPR] Homogenization mode: ecm_fixed "
                "(fixed classical eps/sig weights)."
            )
        else:
            use_dynamic_hom_weights = True
            maw_eps = _build_maw_target_model(ecm_data, "eps", required=True)
            maw_sig = _build_maw_target_model(ecm_data, "sig", required=True)
            z_eps_local, z_eps_pos, miss_eps = _map_support_to_current(
                maw_eps["z_support_full"], full_to_local
            )
            z_sig_local, z_sig_pos, miss_sig = _map_support_to_current(
                maw_sig["z_support_full"], full_to_local
            )
            if miss_eps.size:
                print(
                    f"[HPROM-MAWECM-GPR] WARNING: eps support lost {miss_eps.size} full-mesh indices "
                    "when mapping to current mesh."
                )
            if miss_sig.size:
                print(
                    f"[HPROM-MAWECM-GPR] WARNING: sig support lost {miss_sig.size} full-mesh indices "
                    "when mapping to current mesh."
                )
            print("  [HPROM-MAWECM-GPR] Homogenization mode: dynamic MAW weights.")
    else:
        print("  [HPROM-MAWECM-GPR] Homogenization evaluation disabled (q-only mode).")

    if bool(dynamic_residual_weights):
        print("  [HPROM-MAWECM-GPR] Residual mode: dynamic MAW weights.")
        print(f"  [HPROM-MAWECM-GPR] Include d(w_res)/d(q_m) tangent term: {int(use_weight_tangent)}")
    else:
        print("  [HPROM-MAWECM-GPR] Residual mode: fixed support anchor weights.")
        print("  [HPROM-MAWECM-GPR] Residual assembly backend: cached fixed-weight assembler.")
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

    gpr_input_dim = int(gpr_model["input_dim"])
    expected_input_dim = int(n_primary)
    if gpr_input_dim != expected_input_dim:
        raise ValueError(
            f"GPR input size mismatch: model expects {gpr_input_dim}, "
            f"but solver was configured for {expected_input_dim}."
        )
    qp_init_mode = str(qp_init_mode).strip().lower()
    if qp_init_mode not in ("continuation", "previous", "zero", "mu_affine"):
        raise ValueError(
            f"Unsupported qp_init_mode='{qp_init_mode}'. "
            "Use one of: continuation, previous, zero, mu_affine."
        )
    qp_aff = gpr_model.get("qp_init_mu_affine", None)
    if qp_init_mode in ("continuation", "mu_affine") and qp_aff is None:
        raise RuntimeError(
            f"[HPROM-MAWECM-GPR] qp_init_mode='{qp_init_mode}' requires "
            "qp_init_mu_affine.npz, but it is missing "
            f"in model directory."
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

    def _build_gpr_input(qp_vec, e_vec):
        qp = np.asarray(qp_vec, dtype=float).reshape(-1)
        return qp

    def _evaluate_qs_and_jac(qp_vec, e_vec):
        x_in = _build_gpr_input(qp_vec, e_vec)
        q_s_map, j_qs_qp = evaluate_sparse_gp_map_and_jacobian_qp(x_in, gpr_model, n_primary)
        return np.asarray(q_s_map, dtype=float).reshape(-1), np.asarray(j_qs_qp, dtype=float)

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
                f"[HPROM-MAWECM-GPR] qp_init_mu_affine expects mu_dim={mu_dim} but got only {mu.size} strain components."
            )
        x_aug = np.concatenate([mu, np.array([1.0], dtype=float)])
        q0 = x_aug @ np.asarray(qp_aff["b_aff"], dtype=float)
        return np.asarray(q0, dtype=float).reshape(-1)

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

    def _evaluate_residual_weights_local(qp_vec):
        w_support_all = _evaluate_support_weights(qp_vec, maw_res)
        w_local = np.asarray(w_support_all[z_res_pos], dtype=float)
        nz = np.flatnonzero(np.abs(w_local) > WEIGHT_ZERO_TOL)
        if nz.size == 0:
            return np.asarray(w_res_anchor_local, dtype=float).copy()
        return w_local

    def _evaluate_residual_weights_and_jacobian_local(qp_vec):
        w_support_all, dw_support_all = _evaluate_support_weights_and_jacobian(qp_vec, maw_res)
        w_local = np.asarray(w_support_all[z_res_pos], dtype=float)
        dw_local = np.asarray(dw_support_all[z_res_pos, :], dtype=float)
        if dw_local.shape != (z_res_local.size, n_primary):
            raise RuntimeError(
                "[HPROM-MAWECM-GPR] residual MAW weight Jacobian shape mismatch: "
                f"got {dw_local.shape}, expected ({z_res_local.size}, {n_primary})."
            )
        nz = np.flatnonzero(np.abs(w_local) > WEIGHT_ZERO_TOL)
        if nz.size == 0:
            return (
                np.asarray(w_res_anchor_local, dtype=float).copy(),
                np.zeros((z_res_local.size, n_primary), dtype=float),
            )
        return w_local, dw_local

    def _evaluate_target_weights_current_full(qp_vec, target_model, z_pos, w_anchor_full):
        w_support_all = _evaluate_support_weights(qp_vec, target_model)
        w_support_kept = np.asarray(w_support_all[z_pos], dtype=float)
        z_full_kept = np.asarray(target_model["z_support_full"], dtype=np.int64).reshape(-1)[z_pos]
        w_full = _support_weights_to_full(z_full_kept, w_support_kept, n_elem_reference)
        w_current = _project_full_weights_to_current(w_full, n_elem_current, full_to_local)
        if np.sum(np.abs(w_current)) <= WEIGHT_ZERO_TOL:
            w_current = _project_full_weights_to_current(w_anchor_full, n_elem_current, full_to_local)
        return w_current

    q_p = np.zeros(n_primary, dtype=float)
    predictor_only_mode = int(max_its) <= 0
    Kr_old = None
    J_full = np.zeros((n_total_dof, n_primary), dtype=float)
    q0_const, J0_const = _evaluate_qs_and_jac(np.zeros(n_primary, dtype=float), np.zeros(3, dtype=float))
    a_m = np.asarray(gpr_model["A_m"], dtype=float)
    if a_m.shape != (n_primary, n_primary):
        raise RuntimeError(
            f"[HPROM-MAWECM-GPR] A_m shape mismatch: got {a_m.shape}, "
            f"expected {(n_primary, n_primary)}."
        )
    phi_master = phi_p @ a_m
    phi_p_eff = phi_master + phi_s @ J0_const
    w0_const = phi_s @ q0_const

    print(f"  [HPROM-MAWECM-GPR] Solving for {n_steps_total} dynamic increments...")
    print(
        f"  [HPROM-MAWECM-GPR] Active mesh elements: {len(elements)} "
        f"(reference full mesh: {n_elem_reference})"
    )
    print("  [HPROM-MAWECM-GPR] Manifold correction active: N(0)=0 and J(0)=0.")
    print(f"  [HPROM-MAWECM-GPR] q_m initializer mode: {qp_init_mode}")
    if predictor_only_mode:
        print("  [HPROM-MAWECM-GPR] Predictor-only mode enabled (max_its <= 0): no Newton correction.")
    if qp_init_mode in ("continuation", "mu_affine") and qp_aff is not None:
        print(f"  [HPROM-MAWECM-GPR] Using affine initializer: {qp_aff['path']}")
    print(
        f"  [HPROM-MAWECM-GPR] MAW residual support on current mesh: "
        f"{len(z_res_local)} / {len(elements)}"
    )
    if using_hrom_mesh:
        print("  [HPROM-MAWECM-GPR] Running on reduced mesh with full-to-local MAW support mapping.")
    t_map = 0.0
    t_assembly = 0.0
    t_projection = 0.0
    t_solve = 0.0
    t_full_sync = 0.0
    step_iters = []

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
            print(f"\n[HPROM-MAWECM-GPR] Step {step:04d}/{n_steps_total} | E={E}")
        verbose_step = bool(verbose_iterations)

        q_p = _initial_qp_guess(E, q_p, step)
        it = 0
        res_norm_0 = None
        converged = False
        nonfinite_detected = False
        Kr_last = None
        dq_norm_prev = None
        prev_res_norm = None
        q_step_start = q_p.copy()
        best_q = q_step_start.copy()
        best_res = np.inf
        best_rel = np.inf
        it_step_count = 0

        while it < max_its:
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1
            it_step_count += 1
            t0 = time.perf_counter()
            q_s_map, J_gpr_raw = _evaluate_qs_and_jac(q_p, E)
            t_map += time.perf_counter() - t0
            q_s = q_s_map - q0_const - J0_const @ q_p
            J_gpr = J_gpr_raw - J0_const

            if verbose_step:
                print(f"    > q_m norm: {np.linalg.norm(q_p):.3e} | q_s norm: {np.linalg.norm(q_s):.3e}")
            if (not _is_finite(q_p)) or (not _is_finite(q_s)):
                print("  [HPROM-MAWECM-GPR] WARNING: non-finite reduced state detected.")
                nonfinite_detected = True
                break

            if J_gpr.shape != (n_secondary, n_primary):
                raise RuntimeError(
                    f"Invalid GPR Jacobian shape {J_gpr.shape}; expected ({n_secondary}, {n_primary})."
                )
            if not _is_finite(J_gpr):
                print("  [HPROM-MAWECM-GPR] WARNING: non-finite GPR Jacobian detected.")
                nonfinite_detected = True
                break

            u_fluc = w0_const + phi_p_eff @ q_p + phi_s @ q_s
            if not _is_finite(u_fluc):
                print("  [HPROM-MAWECM-GPR] WARNING: non-finite reconstructed displacement detected.")
                nonfinite_detected = True
                break
            u_free = u_aff_free + u_fluc

            J_manifold = phi_p_eff + phi_s @ J_gpr
            if not _is_finite(J_manifold):
                print("  [HPROM-MAWECM-GPR] WARNING: non-finite manifold Jacobian detected.")
                nonfinite_detected = True
                break

            u_eq_curr = _apply_total_free_displacement(u_free, base_disp_vec=disp_base_step)

            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            t0 = time.perf_counter()
            dw_res_local = None
            if bool(dynamic_residual_weights):
                if use_weight_tangent:
                    w_res_local, dw_res_local = _evaluate_residual_weights_and_jacobian_local(q_p)
                else:
                    w_res_local = _evaluate_residual_weights_local(q_p)
                K_hp, rhs_hp = dyn_res_assembler.assemble(u_eq_curr, w_res_local)
            else:
                w_res_local = np.asarray(w_res_anchor_local, dtype=float)
                K_hp, rhs_hp = AssembleHyperReducedSystem(
                    mp,
                    n_total_dof,
                    elements,
                    z_res_local,
                    w_res_local,
                    u_eq=u_eq_curr,
                )
            t_assembly += time.perf_counter() - t0
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            if (not _is_finite(rhs_hp)) or (not _is_finite(K_hp.data)):
                print("  [HPROM-MAWECM-GPR] WARNING: non-finite full residual/stiffness detected.")
                nonfinite_detected = True
                break

            t0 = time.perf_counter()
            J_full.fill(0.0)
            J_full[free_dofs, :] = J_manifold
            KJ = K_hp @ J_full
            r_r = J_full.T @ rhs_hp
            K_r = J_full.T @ KJ
            if use_weight_tangent and dw_res_local is not None:
                # Same sign convention as the validated 2D MAWECM solver:
                # Newton uses K_alg dq = R with R = rhs = -f_int.
                # For R(q)=sum_e w_e(q) R_e(u(q)), K_alg=-dR/dq, hence
                # the weight-dependency contribution is -sum_e R_e \otimes dw_e/dq.
                rhs_unit_loc = -np.asarray(dyn_res_assembler._assembler._f_int, dtype=float).reshape(
                    dyn_res_assembler.n_sel,
                    int(dyn_res_assembler._assembler.n_local_dof),
                )
                r_free_e = _scatter_rhs_loc_by_element_free(
                    rhs_loc=rhs_unit_loc,
                    n_free=free_dofs.size,
                    scatter_cache=unit_rhs_scatter,
                )
                drw_dqp = r_free_e @ dw_res_local
                K_r = K_r - (J_manifold.T @ drw_dqp)
            t_projection += time.perf_counter() - t0
            Kr_last = K_r
            if (not _is_finite(r_r)) or (not _is_finite(K_r)):
                print("  [HPROM-MAWECM-GPR] WARNING: non-finite reduced residual/stiffness detected.")
                nonfinite_detected = True
                break

            res_norm = float(np.linalg.norm(r_r))
            if not np.isfinite(res_norm):
                print("  [HPROM-MAWECM-GPR] WARNING: non-finite reduced residual norm detected.")
                nonfinite_detected = True
                break
            if res_norm_0 is None:
                res_norm_0 = max(res_norm, 1e-30)
            rel_res = res_norm / (res_norm_0 + 1e-12)
            if res_norm < best_res:
                best_res = float(res_norm)
                best_q = q_p.copy()
                best_rel = float(rel_res)
            print(f"  > It {it:02d}: ||R_r|| = {res_norm:.3e}, rel = {rel_res:.3e}")

            if res_norm < float(abs_res_cutoff):
                print(f"  > Converged in {it} iterations.")
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
                print("  [HPROM-MAWECM-GPR] WARNING: non-finite reduced update detected.")
                nonfinite_detected = True
                break

            dq_norm = float(np.linalg.norm(dq_p))
            if dq_norm > float(max_dq_norm) and dq_norm > 0.0:
                scale = float(max_dq_norm) / dq_norm
                dq_p *= scale
                dq_norm = float(np.linalg.norm(dq_p))
                print(f"    > large reduced update clipped with scale={scale:.3e}")

            alpha = 1.0 if it <= 10 else 0.5
            q_trial = q_p + alpha * dq_p
            if not _is_finite(q_trial):
                print("  [HPROM-MAWECM-GPR] WARNING: non-finite reduced state update detected.")
                nonfinite_detected = True
                break
            q_p = q_trial
            dq_norm_prev = abs(alpha) * dq_norm
            if used_old:
                print("    > using previous reduced stiffness (K_old) at first iteration")
            it += 1

        if not converged:
            if predictor_only_mode:
                converged = True
                Kr_old = None
            else:
                quasi_converged = (
                    np.isfinite(best_res)
                    and np.isfinite(best_rel)
                    and (best_rel < float(relnorm_cutoff))
                    and (best_res < float(max_res_for_rel_convergence))
                )
                if quasi_converged:
                    q_p = best_q.copy()
                    converged = True
                    print(
                        "  [HPROM-MAWECM-GPR] Step accepted as quasi-converged: "
                        f"best ||R_r||={best_res:.3e}, rel={best_rel:.3e}."
                    )
                    Kr_old = None
                else:
                    msg = (
                        f"[HPROM-MAWECM-GPR] Step {step}/{n_steps_total} did not converge in "
                        f"{max_its} iterations. best ||R_r||={best_res:.3e}, rel={best_rel:.3e}."
                    )
                    if bool(fail_on_nonconvergence):
                        raise RuntimeError(msg)
                    print(f"  [HPROM-MAWECM-GPR] WARNING: {msg}")
                    if nonfinite_detected:
                        print("  [HPROM-MAWECM-GPR] WARNING: non-finite state encountered; rolling back to best finite iterate.")
                    if np.isfinite(best_res):
                        q_p = best_q.copy()
                        print(f"  [HPROM-MAWECM-GPR] Using best finite iterate with ||R_r||={best_res:.3e}.")
                    else:
                        q_p = q_step_start.copy()
                        print("  [HPROM-MAWECM-GPR] Reverting to previous-step reduced state.")
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
            raise RuntimeError("HPROM-MAWECM-GPR accepted state is non-finite after rollback.")

        if not use_fast_dirichlet_bc:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()
        u_eq_final = _apply_total_free_displacement(
            u_aff_free + u_fluc_final, base_disp_vec=disp_base_step
        )
        if bool(evaluate_homogenization):
            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            t0 = time.perf_counter()
            _, _ = vec_full_assembler.Assemble(u_eq_final)
            t_full_sync += time.perf_counter() - t0
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            if bool(use_dynamic_hom_weights):
                w_eps_hom = _evaluate_target_weights_current_full(q_p, maw_eps, z_eps_pos, w_eps_anchor_full)
                w_sig_hom = _evaluate_target_weights_current_full(q_p, maw_sig, z_sig_pos, w_sig_anchor_full)
            else:
                w_eps_hom = w_eps_fixed_cur
                w_sig_hom = w_sig_fixed_cur
            hom_eps, hom_sig = CalculateHomogenizedFromAssemblerWithElementWeights(
                vec_full_assembler,
                w_eps=w_eps_hom,
                w_sig=w_sig_hom,
                reference_measure=hom_reference_measure,
            )
        else:
            hom_eps = np.zeros(3, dtype=float)
            hom_sig = np.zeros(3, dtype=float)
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
        f"\n[HPROM-MAWECM-GPR] Timing: map={t_map:.3f}s, assembly={t_assembly:.3f}s, "
        f"projection={t_projection:.3f}s, solve={t_solve:.3f}s, full_sync={t_full_sync:.3f}s, "
        f"accounted={t_accounted:.3f}s, other={t_other:.3f}s, total={t_wall_total:.3f}s, "
        f"iters(total={timing_stats['newton_iters_total']}, "
        f"mean/step={timing_stats['newton_iters_mean_per_step']:.2f})"
    )
    np.savez(os.path.join(out_dir, "hprom_mawecm_gpr_timing_stats.npz"), **{
        k: np.array([v], dtype=float) for k, v in timing_stats.items()
    })
    with open(os.path.join(out_dir, "hprom_mawecm_gpr_timing_stats.txt"), "w", encoding="utf-8") as f:
        for k, v in timing_stats.items():
            f.write(f"{k}={v}\n")

    tag = f"trajectory_{trajectory_index}" if trajectory_index is not None else "hprom_mawecm_gpr_run"
    np.save(os.path.join(out_dir, f"{tag}_q_p.npy"), np.stack(q_hist))
    np.save(os.path.join(out_dir, f"{tag}_strain.npy"), np.stack(results_eps))
    np.save(os.path.join(out_dir, f"{tag}_stress.npy"), np.stack(results_sig))

    if return_stats:
        return np.array(results_eps), np.array(results_sig), timing_stats
    return np.array(results_eps), np.array(results_sig)
