#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPROM RVE Solver (Hyper-Reduced Projected Order Model)
Only assembles over ECM-selected elements with cubature weights.
"""

import os
import sys
import time
import numpy as np

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    DeformationGradientFromGreenLagrange2D,
    setup_kratos_parameters,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    CalculateHomogenizedStressAndStrainKratosReference,
    DetectMaterialSubModelParts,
    ConfigureElementModelerForMaterialParts,
    StripMdpaExtension,
    BuildDynamicSegmentSteps,
    RVEHomogenizationDatasetGenerator,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
    USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    NEWTON_TOL_ABS,
    DISP_TOL_ABS,
)
from fom_solver_rve import VectorizedAssembler
from scipy.sparse import coo_matrix
from homogenization_gappy import (
    extract_sampled_hom_vector_from_assembler,
    evaluate_gappy_homogenization_from_sample,
)

# =============================================================================
# HPROM Assembly: Only selected elements, weighted
# =============================================================================

USE_VECTORIZED_HPROM_ASSEMBLY = True
WEIGHT_ZERO_TOL = 1.0e-14


def _BuildEqMapFromNodes(mp):
    eq_map = np.empty((mp.NumberOfNodes(), 2), dtype=int)
    for i, node in enumerate(mp.Nodes):
        eq_map[i, 0] = node.GetDof(KM.DISPLACEMENT_X).EquationId
        eq_map[i, 1] = node.GetDof(KM.DISPLACEMENT_Y).EquationId
    return eq_map


def _CaptureDisplacementFromNodes(mp, n_dof, eq_map):
    u = np.zeros(n_dof, dtype=float)
    for i, node in enumerate(mp.Nodes):
        d = node.GetSolutionStepValue(KM.DISPLACEMENT)
        idx_x = int(eq_map[i, 0])
        idx_y = int(eq_map[i, 1])
        if 0 <= idx_x < n_dof:
            u[idx_x] = d[0]
        if 0 <= idx_y < n_dof:
            u[idx_y] = d[1]
    return u


def _AssembleHyperReducedSystemKratosLocal(mp, n_dof, elements, elem_indices, elem_weights):
    pi = mp.ProcessInfo
    rhs = np.zeros(n_dof, dtype=float)

    total_entries = 0
    valid_data = []
    for idx, w_e in zip(elem_indices, elem_weights):
        elem = elements[int(idx)]
        ids = np.array(elem.EquationIdVector(pi), dtype=int)
        mask = ids >= 0
        n_m = int(np.sum(mask))
        if n_m == 0:
            continue
        total_entries += n_m * n_m
        valid_data.append((elem, ids, mask, float(w_e)))

    rows = np.zeros(total_entries, dtype=int)
    cols = np.zeros(total_entries, dtype=int)
    vals = np.zeros(total_entries, dtype=float)

    curr = 0
    for elem, ids, mask, w_e in valid_data:
        LHS = KM.Matrix()
        RHS = KM.Vector()
        elem.CalculateLocalSystem(LHS, RHS, pi)

        ids_m = ids[mask]
        rhs[ids_m] += w_e * np.array(RHS, dtype=float)[mask]

        A = w_e * np.array(LHS, dtype=float)[np.ix_(mask, mask)]
        n_m = ids_m.size
        num = n_m * n_m

        rows[curr:curr+num] = np.repeat(ids_m, n_m)
        cols[curr:curr+num] = np.tile(ids_m, n_m)
        vals[curr:curr+num] = A.reshape(-1)
        curr += num

    K = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    return K, rhs

def AssembleHyperReducedSystem(mp, n_dof, elements, elem_indices, elem_weights, u_eq=None):
    """
    Assemble K and rhs using ONLY the ECM-selected elements,
    scaling each element's contribution by its cubature weight.
    """
    if not USE_VECTORIZED_HPROM_ASSEMBLY:
        return _AssembleHyperReducedSystemKratosLocal(mp, n_dof, elements, elem_indices, elem_weights)

    elem_idx_arr = np.asarray(elem_indices, dtype=np.int64).reshape(-1)
    elem_w_arr = np.asarray(elem_weights, dtype=float).reshape(-1)
    cache = getattr(AssembleHyperReducedSystem, "_vectorized_cache", {})
    rec = cache.get(id(mp))

    needs_rebuild = True
    if rec is not None:
        needs_rebuild = (
            rec["n_dof"] != int(n_dof)
            or rec["n_elem"] != int(len(elements))
            or rec["idx"].shape != elem_idx_arr.shape
            or rec["w"].shape != elem_w_arr.shape
            or not np.array_equal(rec["idx"], elem_idx_arr)
            or not np.allclose(rec["w"], elem_w_arr, rtol=0.0, atol=0.0)
        )

    if needs_rebuild:
        eq_map = _BuildEqMapFromNodes(mp)
        selected_elements = [elements[int(i)] for i in elem_idx_arr]
        assembler = VectorizedAssembler(
            mp,
            int(n_dof),
            eq_map,
            elements=selected_elements,
            element_scales=elem_w_arr,
            log_label="HPROMVectorizedAssembler",
        )
        rec = {
            "n_dof": int(n_dof),
            "n_elem": int(len(elements)),
            "idx": elem_idx_arr.copy(),
            "w": elem_w_arr.copy(),
            "eq_map": eq_map,
            "assembler": assembler,
        }
        cache[id(mp)] = rec
        setattr(AssembleHyperReducedSystem, "_vectorized_cache", cache)

    try:
        if u_eq is None:
            u_curr = _CaptureDisplacementFromNodes(mp, int(n_dof), rec["eq_map"])
        else:
            u_curr = np.asarray(u_eq, dtype=float).reshape(-1)
        return rec["assembler"].Assemble(u_curr)
    except Exception as exc:
        print(f"[HPROM] WARNING: vectorized hyper-reduced assembly failed, using Kratos local system. ({exc})")
        return _AssembleHyperReducedSystemKratosLocal(mp, n_dof, elements, elem_indices, elem_weights)


def _LoadNodeIdsFromMeshBase(mesh_base):
    cache = getattr(_LoadNodeIdsFromMeshBase, "_cache", {})
    key = str(mesh_base)
    if key in cache:
        return cache[key]

    mdl = KM.Model()
    mp = mdl.CreateModelPart("tmp")
    KM.ModelPartIO(str(mesh_base)).ReadModelPart(mp)
    node_ids = np.array([int(node.Id) for node in mp.Nodes], dtype=np.int64)
    cache[key] = node_ids
    setattr(_LoadNodeIdsFromMeshBase, "_cache", cache)
    return node_ids


def _BuildFullNodeComponentToBasisRowMap(
    free_dofs_reference,
    eq_map_reference,
    full_mesh_base="rve_geometry",
):
    key = (str(full_mesh_base), int(eq_map_reference.shape[0]), int(len(free_dofs_reference)))
    cache = getattr(_BuildFullNodeComponentToBasisRowMap, "_cache", {})
    if key in cache:
        return cache[key]

    node_ids = _LoadNodeIdsFromMeshBase(full_mesh_base)
    if node_ids.size != int(eq_map_reference.shape[0]):
        # Conservative fallback if mesh-reader order differs from stored basis map
        node_ids = np.arange(1, int(eq_map_reference.shape[0]) + 1, dtype=np.int64)

    eq_to_row = {int(eq): i for i, eq in enumerate(np.asarray(free_dofs_reference, dtype=np.int64).reshape(-1))}
    out = {}
    eq_map_ref = np.asarray(eq_map_reference, dtype=np.int64)
    for i, node_id in enumerate(node_ids):
        eq_x = int(eq_map_ref[i, 0])
        eq_y = int(eq_map_ref[i, 1])
        if eq_x in eq_to_row:
            out[(int(node_id), 0)] = int(eq_to_row[eq_x])
        if eq_y in eq_to_row:
            out[(int(node_id), 1)] = int(eq_to_row[eq_y])

    cache[key] = out
    setattr(_BuildFullNodeComponentToBasisRowMap, "_cache", cache)
    return out


def ResolveResidualHyperReductionSelection(ecm_data, n_current_elements, solver_label="HPROM"):
    """
    Resolve element indices and cubature weights for residual assembly.
    Supports both:
      - full-mesh ECM arrays (Z_res + w_res_full), and
      - reduced HROM mesh arrays (w_res_hrom aligned with current mesh order).
    """
    n_cur = int(n_current_elements)
    if "w_res_hrom" in ecm_data:
        w_res_hrom = np.asarray(ecm_data["w_res_hrom"], dtype=float).reshape(-1)
        if w_res_hrom.size == n_cur:
            nz = np.flatnonzero(np.abs(w_res_hrom) > WEIGHT_ZERO_TOL)
            if nz.size == 0:
                raise RuntimeError(f"[{solver_label}] w_res_hrom has no nonzero entries.")
            return nz.astype(np.int64), w_res_hrom[nz], True
        print(
            f"[{solver_label}] WARNING: w_res_hrom has size {w_res_hrom.size}, "
            f"but current mesh has {n_cur} elements. Falling back to Z_res/w_res_full."
        )

    z_res = np.asarray(ecm_data["Z_res"], dtype=np.int64).reshape(-1)
    w_res_full = np.asarray(ecm_data["w_res_full"], dtype=float).reshape(-1)
    if np.max(z_res, initial=-1) >= n_cur:
        raise RuntimeError(
            f"[{solver_label}] ECM element index out of range: max(Z_res)={int(np.max(z_res))} "
            f"for mesh with {n_cur} elements."
        )
    if w_res_full.size != n_cur:
        raise RuntimeError(
            f"[{solver_label}] w_res_full length {w_res_full.size} does not match mesh elements {n_cur}."
        )
    w_sel = w_res_full[z_res]
    nz = np.flatnonzero(np.abs(w_sel) > WEIGHT_ZERO_TOL)
    if nz.size == 0:
        raise RuntimeError(f"[{solver_label}] w_res_full[Z_res] has no nonzero entries.")
    if nz.size != z_res.size:
        print(
            f"[{solver_label}] WARNING: {z_res.size - nz.size} zero-weight residual elements detected in Z_res; "
            "they will be skipped."
        )
    return z_res[nz], w_sel[nz], False


def GetReferenceIntegrationMeasureFromMesh(full_mesh_base):
    """
    Compute reference full-mesh measure for Kratos-reference homogenization:
      A_ref = sum_e Area_e
    Cached per mesh base.
    """
    key = str(full_mesh_base)
    cache = getattr(GetReferenceIntegrationMeasureFromMesh, "_cache", {})
    if key in cache:
        return float(cache[key])

    params = setup_kratos_parameters(key)
    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, params)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    ref_measure = float(np.sum([float(elem.GetGeometry().Area()) for elem in mp.Elements]))
    sim.Finalize()

    cache[key] = ref_measure
    setattr(GetReferenceIntegrationMeasureFromMesh, "_cache", cache)
    return ref_measure


def ResolveHomogenizationWeightSelection(ecm_data, n_current_elements, solver_label="HPROM"):
    """
    Resolve element weights for fast homogenized output on the current mesh order.
    Priority:
      1) projected reduced-mesh weights: w_eps_hrom / w_sig_hrom
      2) full-mesh weights (only if current mesh == full mesh): w_eps_full / w_sig_full
    Returns:
      (w_eps_or_none, w_sig_or_none, using_weighted_hom)
    """
    n_cur = int(n_current_elements)

    def _pick(key_hrom, key_full):
        if key_hrom in ecm_data:
            w = np.asarray(ecm_data[key_hrom], dtype=float).reshape(-1)
            if w.size == n_cur:
                return w
            print(
                f"[{solver_label}] WARNING: {key_hrom} has size {w.size}, "
                f"but current mesh has {n_cur} elements."
            )
        if key_full in ecm_data:
            w = np.asarray(ecm_data[key_full], dtype=float).reshape(-1)
            if w.size == n_cur:
                return w
        return None

    w_eps = _pick("w_eps_hrom", "w_eps_full")
    w_sig = _pick("w_sig_hrom", "w_sig_full")
    using = (w_eps is not None) and (w_sig is not None)
    return w_eps, w_sig, using


def CalculateHomogenizedFromAssemblerWithElementWeights(assembler, w_eps=None, w_sig=None, reference_measure=None):
    """
    Compute homogenized strain/stress from the most recent VectorizedAssembler state.
    Uses the same Kratos-reference operator as CalculateHomogenizedStressAndStrainKratosReference:
      value_hom = (sum_e A_e * mean_gp(value_gp,e)) / A_ref
    If element weights are provided, they are applied to per-element contributions.
    """
    if getattr(assembler, "n_elems", 0) == 0:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    if hasattr(assembler, "area_e"):
        area_e = np.asarray(assembler.area_e, dtype=float).reshape(-1)
    else:
        # Fallback for safety on legacy assembler objects.
        area_e = np.sum(np.asarray(assembler.w_detJ, dtype=float), axis=1)

    eps_mean_e = np.mean(assembler._E_voigt, axis=1)  # (n_elem, 3)
    sig_mean_e = np.mean(assembler._S_voigt, axis=1)  # (n_elem, 3)

    def _avg(mean_e, w):
        if w is None:
            den = float(np.sum(area_e))
            num = np.einsum("e,ej->j", area_e, mean_e)
        else:
            ww = np.asarray(w, dtype=float).reshape(-1)
            if ww.size != mean_e.shape[0]:
                raise RuntimeError(
                    f"Homogenization weight size {ww.size} does not match active elements {mean_e.shape[0]}."
                )
            nz = np.flatnonzero(np.abs(ww) > WEIGHT_ZERO_TOL)
            if nz.size == 0:
                return np.zeros(mean_e.shape[1], dtype=float)
            ww_nz = ww[nz]
            area_nz = area_e[nz]
            mean_nz = mean_e[nz]
            if reference_measure is None:
                den = float(np.dot(ww_nz, area_nz))
            else:
                den = float(reference_measure)
            num = np.einsum("e,ej->j", ww_nz * area_nz, mean_nz)
        if abs(den) <= 1e-30:
            return np.asarray(num, dtype=float)
        return np.asarray(num, dtype=float) / den

    hom_eps = _avg(eps_mean_e, w_eps)
    hom_sig = _avg(sig_mean_e, w_sig)
    return hom_eps, hom_sig


def ResolveGappyHomogenizationOperator(
    ecm_data,
    gappy_data,
    n_current_elements,
    using_hrom_mesh=False,
    solver_label="HPROM",
):
    """
    Resolve an optional Gappy-POD homogenization operator on the current mesh.

    Returns dict:
      {
        "M": (6, m) matrix,
        "b": (6,) offset,
        "sample_elements": (m/6,) local element indices in current mesh order,
      }
    or None when unavailable/incompatible.
    """
    src = gappy_data if gappy_data is not None else ecm_data
    if src is None:
        return None
    if ("hom_gappy_matrix" not in src) or ("hom_gappy_offset" not in src):
        return None

    M = np.asarray(src["hom_gappy_matrix"], dtype=float)
    b = np.asarray(src["hom_gappy_offset"], dtype=float).reshape(-1)
    if M.ndim != 2 or M.shape[0] != 6 or b.size != 6:
        print(f"[{solver_label}] WARNING: invalid gappy operator shape, ignoring.")
        return None

    if using_hrom_mesh:
        if "hom_gappy_sample_elements_hrom" in src:
            z = np.asarray(src["hom_gappy_sample_elements_hrom"], dtype=np.int64).reshape(-1)
        elif (
            ("hom_gappy_sample_elements" in src)
            and ("hrom_element_full_indices" in ecm_data)
        ):
            z_full = np.asarray(src["hom_gappy_sample_elements"], dtype=np.int64).reshape(-1)
            full_to_local = {
                int(full_idx): i
                for i, full_idx in enumerate(np.asarray(ecm_data["hrom_element_full_indices"], dtype=np.int64))
            }
            z_loc = []
            missing = []
            for full_idx in z_full:
                loc = full_to_local.get(int(full_idx))
                if loc is None:
                    missing.append(int(full_idx))
                else:
                    z_loc.append(int(loc))
            if missing:
                print(
                    f"[{solver_label}] WARNING: gappy operator missing {len(missing)} sampled elements "
                    "in HROM mapping; ignoring gappy operator."
                )
                return None
            z = np.asarray(z_loc, dtype=np.int64)
        else:
            print(
                f"[{solver_label}] WARNING: gappy operator has no HROM sample mapping; "
                "ignoring gappy operator on reduced mesh."
            )
            return None
    else:
        if "hom_gappy_sample_elements" not in src:
            print(f"[{solver_label}] WARNING: gappy operator missing sample element list.")
            return None
        z = np.asarray(src["hom_gappy_sample_elements"], dtype=np.int64).reshape(-1)

    n_cur = int(n_current_elements)
    if z.size == 0:
        print(f"[{solver_label}] WARNING: gappy operator sample set is empty.")
        return None
    if np.min(z) < 0 or np.max(z) >= n_cur:
        print(
            f"[{solver_label}] WARNING: gappy sample index out of range for current mesh "
            f"(min={int(np.min(z))}, max={int(np.max(z))}, n_elem={n_cur})."
        )
        return None

    expected_cols = 6 * int(z.size)
    if M.shape[1] != expected_cols:
        print(
            f"[{solver_label}] WARNING: gappy matrix columns={M.shape[1]} "
            f"do not match 6*|Z|={expected_cols}."
        )
        return None

    output_is_average = False
    if "hom_gappy_output_is_average" in src:
        output_is_average = bool(int(np.ravel(src["hom_gappy_output_is_average"])[0]))

    reference_measure = None
    if "hom_gappy_reference_measure" in src:
        val = float(np.ravel(src["hom_gappy_reference_measure"])[0])
        if np.isfinite(val) and abs(val) > 1.0e-30:
            reference_measure = val

    return {
        "M": M,
        "b": b,
        "sample_elements": z,
        "output_is_average": output_is_average,
        "reference_measure": reference_measure,
    }


def ResolveActiveFreeDofsAndBasisRows(
    mp,
    n_dof,
    eq_id_map_active,
    free_dofs_reference,
    eq_map_reference=None,
    full_mesh_base="rve_geometry",
    solver_label="HPROM",
):
    """
    Build active free/dirichlet DOF partitions from the current model part and
    map each active free DOF to its corresponding row in a basis defined on the
    reference full mesh free DOFs.
    """
    n_dof = int(n_dof)
    eq_map_act = np.asarray(eq_id_map_active, dtype=np.int64)
    free_ref = np.asarray(free_dofs_reference, dtype=np.int64).reshape(-1)

    dof_node_id = np.full(n_dof, -1, dtype=np.int64)
    dof_comp = np.full(n_dof, -1, dtype=np.int8)  # 0 -> X, 1 -> Y
    free_mask = np.ones(n_dof, dtype=bool)

    for i, node in enumerate(mp.Nodes):
        idx_x = int(eq_map_act[i, 0])
        idx_y = int(eq_map_act[i, 1])
        if 0 <= idx_x < n_dof:
            dof_node_id[idx_x] = int(node.Id)
            dof_comp[idx_x] = 0
            if node.GetDof(KM.DISPLACEMENT_X).IsFixed():
                free_mask[idx_x] = False
        if 0 <= idx_y < n_dof:
            dof_node_id[idx_y] = int(node.Id)
            dof_comp[idx_y] = 1
            if node.GetDof(KM.DISPLACEMENT_Y).IsFixed():
                free_mask[idx_y] = False

    free_dofs_active = np.nonzero(free_mask)[0].astype(np.int64)
    dir_dofs_active = np.nonzero(~free_mask)[0].astype(np.int64)

    if eq_map_reference is None:
        eq_to_row = {int(eq): i for i, eq in enumerate(free_ref)}
        missing = [int(eq) for eq in free_dofs_active if int(eq) not in eq_to_row]
        if missing:
            raise RuntimeError(
                f"[{solver_label}] Could not map {len(missing)} active free DOFs to basis rows. "
                "Provide eq_map_reference when using reduced HROM meshes."
            )
        basis_rows = np.array([eq_to_row[int(eq)] for eq in free_dofs_active], dtype=np.int64)
        return free_dofs_active, dir_dofs_active, basis_rows

    nodecomp_to_row = _BuildFullNodeComponentToBasisRowMap(
        free_ref,
        np.asarray(eq_map_reference, dtype=np.int64),
        full_mesh_base=full_mesh_base,
    )
    basis_rows = np.empty(free_dofs_active.size, dtype=np.int64)
    missing = []
    for i, eq in enumerate(free_dofs_active):
        key = (int(dof_node_id[int(eq)]), int(dof_comp[int(eq)]))
        row = nodecomp_to_row.get(key)
        if row is None:
            missing.append(key)
        else:
            basis_rows[i] = int(row)
    if missing:
        preview = ", ".join([f"(node={nid},comp={'X' if c==0 else 'Y'})" for nid, c in missing[:8]])
        raise RuntimeError(
            f"[{solver_label}] Missing basis-row mapping for {len(missing)} active DOFs. "
            f"First missing keys: {preview}"
        )
    return free_dofs_active, dir_dofs_active, basis_rows


# =============================================================================
# HPROM Solver
# =============================================================================

def RunHpromBatchSimulation(
    parameters,
    phi_f,        # (n_free, r)
    free_dofs,
    dir_dofs,
    eq_map,
    Xc, Yc,
    ecm_data,     # dict with Z_res, w_res_full, Z_union, w_sig_full, etc.
    out_dir="hprom_results",
    save_plot=True,
    strain_path=None,
    trajectory_index=None,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    max_newton_it=20,
    use_old_stiffness_in_first_iteration=USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    use_fast_dirichlet_bc=True,
    homogenization_method="ecm_weighted",
    homogenization_gappy_data=None,
):
    os.makedirs(out_dir, exist_ok=True)
    phi_f_ref = np.asarray(phi_f, dtype=float)
    free_dofs_ref = np.asarray(free_dofs, dtype=np.int64).reshape(-1)
    if phi_f_ref.shape[0] != free_dofs_ref.size:
        raise RuntimeError(
            f"[HPROM] Basis/free_dofs mismatch: phi_f rows={phi_f_ref.shape[0]} "
            f"!= len(free_dofs)={free_dofs_ref.size}."
        )

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

    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    elements = list(mp.Elements)
    all_entities = list(mp.Elements) + list(mp.Conditions)
    n_dof, eq_id_map, ta_disp = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    vec_full_assembler = VectorizedAssembler(mp, n_dof, eq_id_map, log_label="HPROMFullSyncAssembler")

    full_mesh_base = str(np.ravel(ecm_data["hrom_full_mesh_base"])[0]) if "hrom_full_mesh_base" in ecm_data else "rve_geometry"
    free_dofs, dir_dofs_local, basis_rows = ResolveActiveFreeDofsAndBasisRows(
        mp,
        n_dof,
        eq_id_map,
        free_dofs_reference=free_dofs_ref,
        eq_map_reference=eq_map,
        full_mesh_base=full_mesh_base,
        solver_label="HPROM",
    )
    phi_f = phi_f_ref[basis_rows, :]
    z_res_local, w_res_selected, using_hrom_mesh = ResolveResidualHyperReductionSelection(
        ecm_data,
        n_current_elements=len(elements),
        solver_label="HPROM",
    )
    z_union_ref = np.asarray(ecm_data["Z_union"], dtype=np.int64).reshape(-1) if "Z_union" in ecm_data else z_res_local
    n_elem_reference = int(np.ravel(ecm_data["n_elem"])[0]) if "n_elem" in ecm_data else len(elements)

    x0c = float(Xc)
    y0c = float(Yc)
    dof_x = np.zeros(n_dof, dtype=float)
    dof_y = np.zeros(n_dof, dtype=float)
    is_x_dof = np.zeros(n_dof, dtype=bool)
    for i, node in enumerate(mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        idx_x = int(eq_id_map[i, 0])
        idx_y = int(eq_id_map[i, 1])
        if 0 <= idx_x < n_dof:
            dof_x[idx_x] = xr
            dof_y[idx_x] = yr
            is_x_dof[idx_x] = True
        if 0 <= idx_y < n_dof:
            dof_x[idx_y] = xr
            dof_y[idx_y] = yr
            is_x_dof[idx_y] = False

    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]
    x_dir = dof_x[dir_dofs_local]
    y_dir = dof_y[dir_dofs_local]
    is_x_dir = is_x_dof[dir_dofs_local]

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
        disp_vec = np.zeros(n_dof, dtype=float)
        for i, node in enumerate(mp.Nodes):
            d = node.GetSolutionStepValue(KM.DISPLACEMENT)
            idx_x, idx_y = eq_id_map[i, 0], eq_id_map[i, 1]
            if idx_x < n_dof:
                disp_vec[idx_x] = d[0]
            if idx_y < n_dof:
                disp_vec[idx_y] = d[1]
        return disp_vec

    def _apply_total_free_displacement(u_total_free, base_disp_vec=None):
        if base_disp_vec is None:
            disp_vec = _capture_current_displacement_vector()
        else:
            disp_vec = np.asarray(base_disp_vec, dtype=float).copy()
        disp_vec[free_dofs] = np.asarray(u_total_free, dtype=float).reshape(-1)
        SetDisplacementFromEquationVector(disp_vec, eq_id_map, ta_disp)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)
        return disp_vec

    n_elem = len(elements)
    print(f"[HPROM] Active mesh elements: {n_elem} (reference full mesh: {n_elem_reference})")
    print(
        f"[HPROM] Residual assembly elements: {z_res_local.size} "
        f"({100. * z_res_local.size / max(n_elem, 1):.1f}% of active mesh)"
    )
    if using_hrom_mesh:
        print(f"[HPROM] Using reduced HROM mesh weights (w_res_hrom).")
    else:
        print(f"[HPROM] Using full-mesh ECM indices (Z_res).")
    print(
        f"[HPROM] ECM union reference size: {z_union_ref.size} "
        f"({100. * z_union_ref.size / max(n_elem_reference, 1):.1f}% of full mesh)"
    )

    w_eps_hom, w_sig_hom, using_weighted_hom = ResolveHomogenizationWeightSelection(
        ecm_data,
        n_current_elements=len(elements),
        solver_label="HPROM",
    )
    hom_reference_measure = GetReferenceIntegrationMeasureFromMesh(full_mesh_base)
    print(f"[HPROM] Homogenization reference measure A0 (full mesh): {hom_reference_measure:.6e}")
    with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
        f.write(f"{float(hom_reference_measure):.16e}\n")

    method = str(homogenization_method).strip().lower()
    if method in ("default", "ecm", "ecm_weighted"):
        method = "ecm_weighted"
    elif method in ("gappy", "gappy_pod", "gappy_pod_residual_sampling"):
        method = "gappy_pod"
    elif method in ("kratos", "kratos_reference", "direct"):
        method = "kratos_reference"
    else:
        raise RuntimeError(f"[HPROM] Unknown homogenization_method='{homogenization_method}'.")

    gappy_op = ResolveGappyHomogenizationOperator(
        ecm_data=ecm_data,
        gappy_data=homogenization_gappy_data,
        n_current_elements=len(elements),
        using_hrom_mesh=using_hrom_mesh,
        solver_label="HPROM",
    )
    use_gappy_hom = False
    gappy_ref_measure = None
    if method == "gappy_pod":
        if gappy_op is not None:
            use_gappy_hom = True
            using_weighted_hom = False
            print(
                f"[HPROM] Using Gappy-POD homogenization on residual sample set "
                f"(|Z|={int(gappy_op['sample_elements'].size)})."
            )
            if gappy_op.get("output_is_average", False):
                print("[HPROM] Gappy operator outputs are already homogenized averages.")
            else:
                gappy_ref_measure = gappy_op.get("reference_measure")
                if gappy_ref_measure is None:
                    gappy_ref_measure = GetReferenceIntegrationMeasureFromMesh(full_mesh_base)
                print(
                    "[HPROM] Gappy operator outputs integrals; normalizing by "
                    f"reference measure A_ref={float(gappy_ref_measure):.6e}."
                )
        else:
            raise RuntimeError("[HPROM] Gappy homogenization requested but operator is unavailable.")
    elif method == "kratos_reference":
        using_weighted_hom = False
        use_gappy_hom = False
        print("[HPROM] Using Kratos-reference homogenization.")

    if method == "ecm_weighted":
        if not using_weighted_hom:
            raise RuntimeError(
                "[HPROM] ECM homogenization selected but weights are unavailable on the current mesh."
            )
        print("[HPROM] Using ECM-weighted homogenization (w_eps / w_sig).")
    elif method == "kratos_reference":
        print("[HPROM] Using unweighted full/reduced numerator with reference-area denominator.")

    # HPROM Initialization
    r = phi_f.shape[1]
    q = np.zeros(r, dtype=float)
    phi_full = np.zeros((n_dof, r), dtype=float)
    phi_full[free_dofs, :] = phi_f
    phi_full_T = phi_full.T

    Q_hist, strain_hist, stress_hist = [], [], []

    # Initial state
    Q_hist.append(q.copy())
    strain_hist.append(np.zeros(3))
    stress_hist.append(np.zeros(3))
    Kr_old = None

    t_assembly = 0.0
    t_solve = 0.0
    t_project = 0.0
    t_full_sync = 0.0

    for step in range(1, n_steps_total + 1):
        time_val = float(step) * float(dt)
        mp.CloneTimeStep(time_val)
        mp.ProcessInfo[KM.DELTA_TIME] = dt
        mp.ProcessInfo[KM.TIME] = time_val
        mp.ProcessInfo[KM.STEP] = step

        sim.time, sim.step, sim.end_time = time_val, step, end_time
        sim.InitializeSolutionStep()

        # Interpolate waypoint
        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        E_t = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]

        sim.batch_strain = E_t.copy()
        u_aff_free = _compute_affine_free_displacement(E_t)
        if use_fast_dirichlet_bc:
            disp_base_step = np.zeros(n_dof, dtype=float)
            disp_base_step[dir_dofs_local] = _compute_affine_dirichlet_displacement(E_t)
        else:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()

        print(f"\n[HPROM] Step {step:03d}/{n_steps_total} | t={time_val:.6f}")

        converged = False
        Kr_last = None
        for it in range(max_newton_it):
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1

            # Reconstruction: u = u_aff + Phi_f * q
            u_free = u_aff_free + phi_f @ q
            t0_proj = time.perf_counter()
            u_eq_curr = _apply_total_free_displacement(u_free, base_disp_vec=disp_base_step)
            t_project += time.perf_counter() - t0_proj

            InitializeNonLinearIteration(all_entities, mp.ProcessInfo)

            # *** HYPER-REDUCED ASSEMBLY ***
            t0 = time.perf_counter()
            K_hp, rhs_hp = AssembleHyperReducedSystem(
                mp, n_dof, elements, z_res_local, w_res_selected, u_eq=u_eq_curr
            )
            t_assembly += time.perf_counter() - t0

            FinalizeNonLinearIteration(all_entities, mp.ProcessInfo)

            # Project to reduced space
            Kf_phi = K_hp @ phi_full
            Kr = phi_full_T @ Kf_phi
            Kr_last = Kr
            rr = phi_full_T @ rhs_hp

            nR = np.linalg.norm(rr)
            if it > 0 and nR < NEWTON_TOL_ABS:
                print(f"  > It {it:02d}: ||r_r|| = {nR:.3e} (CONVERGED)")
                converged = True
                break

            t0 = time.perf_counter()
            K_solve = Kr
            used_old = False
            if (
                it == 0
                and use_old_stiffness_in_first_iteration
                and Kr_old is not None
                and Kr_old.shape == Kr.shape
            ):
                K_solve = Kr_old
                used_old = True

            try:
                dq = np.linalg.solve(K_solve, rr)
            except np.linalg.LinAlgError:
                if used_old:
                    try:
                        dq = np.linalg.solve(Kr, rr)
                        used_old = False
                    except np.linalg.LinAlgError:
                        dq, *_ = np.linalg.lstsq(Kr, rr, rcond=None)
                        used_old = False
                else:
                    dq, *_ = np.linalg.lstsq(Kr, rr, rcond=None)
            t_solve += time.perf_counter() - t0
            nq = np.linalg.norm(dq)

            solve_tag = " (K_old)" if used_old else ""
            print(f"  > It {it:02d}: ||r_r|| = {nR:.3e}, ||dq|| = {nq:.3e}{solve_tag}")

            q += dq
            if nq < DISP_TOL_ABS:
                print(f"  > It {it:02d}: ||dq|| = {nq:.3e} (CONVERGED)")
                converged = True
                break

        if not converged:
            print(f" [WARNING] Step {step} did not converge.")
        if Kr_last is not None:
            Kr_old = Kr_last.copy()

        u_fluc_final = phi_f @ q
        if not use_fast_dirichlet_bc:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()
        u_eq_final = _apply_total_free_displacement(
            u_aff_free + u_fluc_final, base_disp_vec=disp_base_step
        )

        InitializeNonLinearIteration(all_entities, mp.ProcessInfo)
        t0_sync = time.perf_counter()
        _, _ = vec_full_assembler.Assemble(u_eq_final)
        t_full_sync += time.perf_counter() - t0_sync
        FinalizeNonLinearIteration(all_entities, mp.ProcessInfo)

        if use_gappy_hom:
            sampled_c = extract_sampled_hom_vector_from_assembler(
                vec_full_assembler, gappy_op["sample_elements"]
            )
            hom_vec = evaluate_gappy_homogenization_from_sample(
                sampled_c, gappy_op["M"], gappy_op["b"]
            )
            if not gappy_op.get("output_is_average", False):
                hom_vec = np.asarray(hom_vec, dtype=float) / float(gappy_ref_measure)
            hom_eps = np.asarray(hom_vec[0:3], dtype=float)
            hom_sig = np.asarray(hom_vec[3:6], dtype=float)
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

        Q_hist.append(q.copy())
        strain_hist.append(hom_eps)
        stress_hist.append(hom_sig)

    sim.Finalize()
    print(
        f"\n[HPROM] Timing: assembly={t_assembly:.3f}s, solve={t_solve:.3f}s, "
        f"project={t_project:.3f}s, full_sync={t_full_sync:.3f}s"
    )

    # Save Results
    tag = f"trajectory_{trajectory_index}" if trajectory_index else "hprom_run"
    np.save(os.path.join(out_dir, f"{tag}_q.npy"), np.stack(Q_hist))
    np.save(os.path.join(out_dir, f"{tag}_strain.npy"), np.stack(strain_hist))
    np.save(os.path.join(out_dir, f"{tag}_stress.npy"), np.stack(stress_hist))

    return np.stack(strain_hist), np.stack(stress_hist)


if __name__ == "__main__":
    from fom_solver_rve import SetInputMeshFilename, LoadStrainWaypointsFromFile

    model_dir = "pod_rbf_model"
    hprom_dir = "hprom_data"

    phi_f = np.load(os.path.join(model_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(model_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(model_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(model_dir, "eq_map.npy"))
    Xc, Yc = np.load(os.path.join(model_dir, "domain_center.npy"))

    ecm = np.load(os.path.join(hprom_dir, "ecm_weights_all.npz"))
    ecm_data = {k: ecm[k] for k in ecm.files}

    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())
    SetInputMeshFilename(parameters, "rve_geometry")
    mdpa_path = f"{StripMdpaExtension('rve_geometry')}.mdpa"
    material_parts = DetectMaterialSubModelParts(mdpa_path)
    parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
    parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(
        "StructuralMaterials.json"
    )

    strain_path, meta = LoadStrainWaypointsFromFile("stage_0_trajectory/stage_0_trajectories.npz", 1)

    RunHpromBatchSimulation(
        parameters, phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc,
        ecm_data=ecm_data,
        strain_path=strain_path,
        trajectory_index=1,
        reference_amplitude=meta.get("reference_amplitude", 0.10),
    )
