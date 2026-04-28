#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
from KratosMultiphysics.StructuralMechanicsApplication import (
    python_solvers_wrapper_structural as structural_solvers,
)
import KratosMultiphysics.StructuralMechanicsApplication as SMApp  # noqa: F401
import KratosMultiphysics.ConstitutiveLawsApplication as CLA  # noqa: F401

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


USE_OLD_STIFFNESS_IN_FIRST_ITERATION = True
NEWTON_TOL_REL = 1e-6
NEWTON_TOL_ABS = 1e-6
DISP_TOL_REL = 1e-8
DISP_TOL_ABS = 1e-8
CONVERGENCE_CRITERION = "displacement"  # "residual", "displacement", "both"

DEFAULT_OUTPUT_DIR = "stage_1_training_set_fom"

# 0 -> Exx, 1 -> Eyy, 2 -> Exy
UNIAXIAL_COMPONENT = 0

# Kratos-like line-search on directional residual.
# Default is disabled to mirror current ProjectParameters.json.
USE_LINE_SEARCH = False
LS_MAX_IT = 10
LS_FIRST_ALPHA = 0.5
LS_SECOND_ALPHA = 1.0
LS_MIN_ALPHA = 0.1
LS_MAX_ALPHA = 2.0
LS_TOL = 0.5

# Dynamic path-time discretization:
# Reference: a unit-amplitude jump (0 -> 1) uses 400 steps.
REFERENCE_STEPS_FOR_UNIT_AMPLITUDE = 400
MIN_STEPS_PER_SEGMENT = 1


def GreenLagrangeVoigtToTensor2D(strain_voigt):
    exx = float(strain_voigt[0])
    eyy = float(strain_voigt[1])
    gxy = float(strain_voigt[2])  # engineering shear = 2 * E12
    e12 = 0.5 * gxy
    return np.array([[exx, e12], [e12, eyy]], dtype=float)


def DeformationGradientFromGreenLagrange2D(strain_voigt):
    e_tensor = GreenLagrangeVoigtToTensor2D(strain_voigt)
    c_tensor = 2.0 * e_tensor + np.eye(2, dtype=float)  # C = F^T F

    eigvals, eigvecs = np.linalg.eigh(c_tensor)
    if np.min(eigvals) <= 0.0:
        raise RuntimeError(
            "Invalid Green-Lagrange strain state: C=2E+I is not positive definite."
        )

    f_tensor = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return f_tensor


# =============================================================================
# DOFs / vectors
# =============================================================================


def SetUpDofEquationIdsAndDisplacementAdaptor(mp):
    dofs = KM.DofsArrayType()
    KM.DofArrayUtilities.SetUpDofArray(mp, dofs)
    KM.DofArrayUtilities.SetDofEquationIds(dofs)

    eq_map = np.empty((mp.NumberOfNodes(), 2), dtype=int)
    for i, node in enumerate(mp.Nodes):
        eq_map[i, 0] = node.GetDof(KM.DISPLACEMENT_X).EquationId
        eq_map[i, 1] = node.GetDof(KM.DISPLACEMENT_Y).EquationId

    ta = KM.TensorAdaptors.HistoricalVariableTensorAdaptor(mp.Nodes, KM.DISPLACEMENT, [2])
    ta.Check()
    return len(dofs), eq_map, ta


def SetDisplacementFromEquationVector(u_eq, eq_map, ta):
    ta.data[:, 0] = u_eq[eq_map[:, 0]]
    ta.data[:, 1] = u_eq[eq_map[:, 1]]
    ta.StoreData()


def UpdateCurrentCoordinatesFromDisplacement(mp, step=0):
    KM.VariableUtils().UpdateCurrentPosition(mp.Nodes, KM.DISPLACEMENT, int(step))


def ExtractDirichletBoundaryConditions(mp, step_index=0):
    dofs, vals = [], []
    for node in mp.Nodes:
        dx = node.GetDof(KM.DISPLACEMENT_X)
        dy = node.GetDof(KM.DISPLACEMENT_Y)
        if dx.IsFixed():
            dofs.append(dx.EquationId)
            vals.append(node.GetSolutionStepValue(KM.DISPLACEMENT_X, step_index))
        if dy.IsFixed():
            dofs.append(dy.EquationId)
            vals.append(node.GetSolutionStepValue(KM.DISPLACEMENT_Y, step_index))

    if not dofs:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

    dofs = np.array(dofs, dtype=int)
    vals = np.array(vals, dtype=float)
    p = np.argsort(dofs)
    return dofs[p], vals[p]


def PrecomputeDirichletPartitionFromNodes(mp, n_dof, x0c, y0c):
    """
    Precompute constrained/free equation IDs and reference coordinates for fast
    analytical Dirichlet updates without scanning all nodes every step.
    """
    dir_dofs = []
    dir_x = []
    dir_y = []
    dir_is_x = []

    free_mask = np.ones(n_dof, dtype=bool)

    for node in mp.Nodes:
        xr = float(node.X0) - float(x0c)
        yr = float(node.Y0) - float(y0c)

        dx = node.GetDof(KM.DISPLACEMENT_X)
        dy = node.GetDof(KM.DISPLACEMENT_Y)

        if dx.IsFixed():
            eq = int(dx.EquationId)
            if 0 <= eq < n_dof:
                dir_dofs.append(eq)
                dir_x.append(xr)
                dir_y.append(yr)
                dir_is_x.append(True)
                free_mask[eq] = False

        if dy.IsFixed():
            eq = int(dy.EquationId)
            if 0 <= eq < n_dof:
                dir_dofs.append(eq)
                dir_x.append(xr)
                dir_y.append(yr)
                dir_is_x.append(False)
                free_mask[eq] = False

    if len(dir_dofs) == 0:
        dir_dofs = np.zeros(0, dtype=np.int64)
        dir_x = np.zeros(0, dtype=float)
        dir_y = np.zeros(0, dtype=float)
        dir_is_x = np.zeros(0, dtype=bool)
    else:
        dir_dofs = np.asarray(dir_dofs, dtype=np.int64)
        dir_x = np.asarray(dir_x, dtype=float)
        dir_y = np.asarray(dir_y, dtype=float)
        dir_is_x = np.asarray(dir_is_x, dtype=bool)

        p = np.argsort(dir_dofs)
        dir_dofs = dir_dofs[p]
        dir_x = dir_x[p]
        dir_y = dir_y[p]
        dir_is_x = dir_is_x[p]

    free_dofs = np.nonzero(free_mask)[0].astype(np.int64)
    return dir_dofs, dir_x, dir_y, dir_is_x, free_dofs


def ComputeDirichletValuesFromGreenLagrange(strain_voigt, dir_x, dir_y, dir_is_x):
    if dir_x.size == 0:
        return np.zeros(0, dtype=float)

    f_tensor = DeformationGradientFromGreenLagrange2D(strain_voigt)
    ux = (f_tensor[0, 0] - 1.0) * dir_x + f_tensor[0, 1] * dir_y
    uy = f_tensor[1, 0] * dir_x + (f_tensor[1, 1] - 1.0) * dir_y
    return np.where(dir_is_x, ux, uy)


# =============================================================================
# Element-level assembly from Kratos local systems
# =============================================================================


def AssembleGlobalSystem(mp, n_dof, entities=None):
    """Efficiently assembles the global sparse stiffness matrix and RHS vector."""
    if entities is None:
        entities = list(mp.Elements) + list(mp.Conditions)
    pi = mp.ProcessInfo

    rhs = np.zeros(n_dof, dtype=float)
    
    # Pre-calculate total number of sparse entries to avoid list extensions
    total_entries = 0
    valid_entities = []
    for ent in entities:
        ids_raw = ent.EquationIdVector(pi)
        n_local = len(ids_raw)
        if n_local > 0:
            ids = np.array(ids_raw, dtype=int)
            mask = ids >= 0
            if np.any(mask):
                valid_entities.append((ent, ids, mask))
                n_mask = np.sum(mask)
                total_entries += n_mask * n_mask
    
    rows = np.zeros(total_entries, dtype=int)
    cols = np.zeros(total_entries, dtype=int)
    vals = np.zeros(total_entries, dtype=float)
    
    curr = 0
    for ent, ids, mask in valid_entities:
        LHS = KM.Matrix()
        RHS = KM.Vector()
        ent.CalculateLocalSystem(LHS, RHS, pi)
        
        ids_m = ids[mask]
        rhs[ids_m] += np.array(RHS, dtype=float)[mask]
        
        A = np.array(LHS, dtype=float)[np.ix_(mask, mask)]
        n_m = ids_m.size
        num = n_m * n_m
        
        rows[curr:curr+num] = np.repeat(ids_m, n_m)
        cols[curr:curr+num] = np.tile(ids_m, n_m)
        vals[curr:curr+num] = A.reshape(-1)
        curr += num
        
    K = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    return K, rhs


def AssembleGlobalRHS(mp, n_dof, entities=None):
    rhs = np.zeros(n_dof, dtype=float)
    pi = mp.ProcessInfo

    if entities is None:
        entities = list(mp.Elements) + list(mp.Conditions)

    for ent in entities:
        RHS = KM.Vector()
        ids = np.array(ent.EquationIdVector(pi), dtype=int)
        if ids.size == 0:
            continue

        ent.CalculateRightHandSide(RHS, pi)

        mask = ids >= 0
        if not np.any(mask):
            continue

        ids_m = ids[mask]
        rhs_local = np.array(RHS, dtype=float)[mask]
        rhs[ids_m] += rhs_local

    return rhs


def InitializeNonLinearIteration(entities, pi):
    for ent in entities:
        if hasattr(ent, "InitializeNonLinearIteration"):
            ent.InitializeNonLinearIteration(pi)


def FinalizeNonLinearIteration(entities, pi):
    for ent in entities:
        if hasattr(ent, "FinalizeNonLinearIteration"):
            ent.FinalizeNonLinearIteration(pi)


# =============================================================================
# Homogenization and GP extraction
# =============================================================================


def _voigt3(v):
    a = np.array(v, dtype=float).reshape(-1)
    if a.size == 3:
        return a
    if a.size >= 6:
        return a[[0, 1, 3]]
    if a.size == 4:
        return a[[0, 1, 3]]
    out = np.zeros(3, dtype=float)
    out[: min(3, a.size)] = a[: min(3, a.size)]
    return out


def PrecomputeElementIntegrationWeights(elements):
    n_elems = len(elements)
    n_gauss = len(elements[0].GetGeometry().IntegrationPoints())

    w_gp = np.zeros((n_elems, n_gauss), dtype=float)
    area_e = np.zeros(n_elems, dtype=float)

    for e, elem in enumerate(elements):
        geom = elem.GetGeometry()
        area_e[e] = geom.Area()
        props = elem.Properties
        thickness = props[KM.THICKNESS] if props.Has(KM.THICKNESS) else 1.0

        ips = geom.IntegrationPoints()
        for g, ip in enumerate(ips):
            detJ = geom.DeterminantOfJacobian(g)
            w_gp[e, g] = ip[3] * detJ * thickness

    return w_gp, area_e


def EvaluateGaussPointData(elements, mp):
    """Returns strain and stress at Gauss points; third return is a compatibility placeholder."""
    pi = mp.ProcessInfo
    n_elems = len(elements)
    if n_elems == 0:
        return np.zeros((0, 0, 3)), np.zeros((0, 0, 3)), np.zeros((0, 0, 0))

    n_gauss = len(elements[0].GetGeometry().IntegrationPoints())

    eps_gp = np.zeros((n_elems, n_gauss, 3), dtype=float)
    sig_gp = np.zeros((n_elems, n_gauss, 3), dtype=float)
    iv_gp = np.zeros((n_elems, n_gauss, 0), dtype=float)

    for e, elem in enumerate(elements):
        eps_list = elem.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, pi)
        sig_list = elem.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, pi)

        for g in range(n_gauss):
            eps_gp[e, g, :] = _voigt3(eps_list[g])
            sig_gp[e, g, :] = _voigt3(sig_list[g])

    return eps_gp, sig_gp, iv_gp




def CalculateHomogenizedStressAndStrain(w_gp, area_e, eps_gp, sig_gp):
    w = w_gp[..., None]
    hom_eps = np.sum(w * eps_gp, axis=(0, 1))
    hom_sig = np.sum(w * sig_gp, axis=(0, 1))
    total_w = float(np.sum(w_gp))
    if total_w > 0.0:
        return hom_eps / total_w, hom_sig / total_w

    A = float(np.sum(area_e))
    if A > 0.0:
        return hom_eps / A, hom_sig / A

    return hom_eps, hom_sig


def CalculateHomogenizedStressAndStrainKratosLike(area_e, eps_gp, sig_gp):
    if eps_gp.size == 0 or sig_gp.size == 0:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    n_ip = int(eps_gp.shape[1]) if eps_gp.ndim >= 2 else 0
    if n_ip <= 0:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    area = float(np.sum(area_e))
    if area <= 0.0:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    eps_e = np.sum(eps_gp, axis=1) / float(n_ip)
    sig_e = np.sum(sig_gp, axis=1) / float(n_ip)
    eps_h = np.sum(area_e[:, None] * eps_e, axis=0) / area
    sig_h = np.sum(area_e[:, None] * sig_e, axis=0) / area
    return eps_h, sig_h


def CalculateHomogenizedStressAndStrainKratosReference(mp):
    process_info = mp.ProcessInfo

    first = next(iter(mp.Elements), None)
    if first is None:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    dummy = np.array(
        first.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info),
        dtype=float,
    )
    n_ip, voigt = dummy.shape

    eps_h = np.zeros(voigt, dtype=float)
    sig_h = np.zeros(voigt, dtype=float)
    area = 0.0

    for elem in mp.Elements:
        eps = np.array(
            elem.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info),
            dtype=float,
        )
        sig = np.array(
            elem.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info),
            dtype=float,
        )

        A = float(elem.GetGeometry().Area())
        area += A
        eps_h += A * np.sum(eps, axis=0) / float(n_ip)
        sig_h += A * np.sum(sig, axis=0) / float(n_ip)

    if area <= 0.0:
        return eps_h, sig_h
    return eps_h / area, sig_h / area


def ComputeEquivalentStressStrain2D(strain_hist, stress_hist):
    Exx = strain_hist[:, 0]
    Eyy = strain_hist[:, 1]
    Gxy = strain_hist[:, 2]
    E12 = 0.5 * Gxy  # Kratos strain vector uses engineering shear in Voigt form.

    Emean = 0.5 * (Exx + Eyy)
    Edev_xx = Exx - Emean
    Edev_yy = Eyy - Emean
    Edev_xy = E12
    Edev_norm2 = Edev_xx * Edev_xx + Edev_yy * Edev_yy + 2.0 * Edev_xy * Edev_xy
    eps_eq = np.sqrt(np.maximum((2.0 / 3.0) * Edev_norm2, 0.0))

    Sxx = stress_hist[:, 0]
    Syy = stress_hist[:, 1]
    Sxy = stress_hist[:, 2]
    Smean = 0.5 * (Sxx + Syy)
    Sdev_xx = Sxx - Smean
    Sdev_yy = Syy - Smean
    Sdev_xy = Sxy
    Sdev_norm2 = Sdev_xx * Sdev_xx + Sdev_yy * Sdev_yy + 2.0 * Sdev_xy * Sdev_xy
    sig_eq = np.sqrt(np.maximum((3.0 / 2.0) * Sdev_norm2, 0.0))

    return eps_eq, sig_eq


# =============================================================================
# Line search
# =============================================================================


def ComputeLineSearchAlpha(
    mp,
    u_base,
    du_free,
    dir_dofs,
    dir_vals,
    free_dofs,
    eq_map,
    ta_disp,
    n_dof,
    entities,
    first_alpha=LS_FIRST_ALPHA,
    second_alpha=LS_SECOND_ALPHA,
    max_it=LS_MAX_IT,
    min_alpha=LS_MIN_ALPHA,
    max_alpha=LS_MAX_ALPHA,
    tol=LS_TOL,
):
    """
    Secant/interpolation on:
        r(alpha) = (alpha*du)^T * R(u + alpha*du)
    where R is assembled from Kratos element/condition RHS.
    """
    u_trial = u_base.copy()
    pi = mp.ProcessInfo

    def eval_r(alpha):
        u_trial[:] = u_base
        u_trial[free_dofs] = u_base[free_dofs] + alpha * du_free
        u_trial[dir_dofs] = dir_vals

        SetDisplacementFromEquationVector(u_trial, eq_map, ta_disp)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)

        InitializeNonLinearIteration(entities, pi)
        rhs = AssembleGlobalRHS(mp, n_dof, entities)
        FinalizeNonLinearIteration(entities, pi)

        r_f = rhs[free_dofs]
        return float(alpha * np.dot(du_free, r_f))

    x1 = float(first_alpha)
    x2 = float(second_alpha)

    r1 = eval_r(x1)
    r2 = eval_r(x2)

    rmax = max(abs(r1), abs(r2))
    converged = False
    it = 0
    x = x2

    while (not converged) and (it < int(max_it)):
        rmin = min(abs(r1), abs(r2))

        if abs(r1 - r2) > 1e-10:
            x = (r1 * x2 - r2 * x1) / (r1 - r2)
        else:
            x = 1.0

        if x < min_alpha:
            x = min_alpha
        elif x > max_alpha:
            x = max_alpha

        rf = eval_r(x)

        if rmin < tol * rmax:
            converged = True
            break
        if abs(rf) < tol * rmax:
            converged = True
            break

        if abs(r1) > abs(r2):
            r1 = rf
            x1 = x
        else:
            r2 = r1
            x2 = x1
            r1 = rf
            x1 = x

        rmax = max(rmax, abs(rf))
        it += 1

    return float(x)


# =============================================================================
# AnalysisStage wrapper
# =============================================================================


class RVEHomogenizationDatasetGenerator(analysis_stage.AnalysisStage):
    def __init__(self, model, project_parameters):
        super().__init__(model, project_parameters)
        self.batch_strain = np.zeros(3, dtype=float)

        self._center_initialized = False
        self._x0c = 0.0
        self._y0c = 0.0

    def _CreateSolver(self):
        return structural_solvers.CreateSolver(self.model, self.project_parameters)

    def __CreateListOfProcesses(self):
        order = self._GetOrderOfProcessesInitialization()
        self._list_of_processes = self._CreateProcesses("processes", order)
        dep = self._CheckDeprecatedOutputProcesses(self._list_of_processes)

        order_out = self._GetOrderOfOutputProcessesInitialization()
        self._list_of_output_processes = self._CreateProcesses("output_processes", order_out)

        self._list_of_processes.extend(self._list_of_output_processes)
        self._list_of_output_processes.extend(dep)

    def _InitializeDomainCenterIfNeeded(self, mp):
        if self._center_initialized:
            return
        xs = [node.X0 for node in mp.Nodes]
        ys = [node.Y0 for node in mp.Nodes]
        if len(xs) == 0:
            self._x0c = 0.0
            self._y0c = 0.0
        else:
            xmin, xmax = float(min(xs)), float(max(xs))
            ymin, ymax = float(min(ys)), float(max(ys))
            self._x0c = 0.5 * (xmin + xmax)
            self._y0c = 0.5 * (ymin + ymax)
        self._center_initialized = True

    def ApplyBoundaryConditions(self):
        super().ApplyBoundaryConditions()

        # Consistent Finite Deformation mapping: u_macro = (F - I) X
        # Matching the kratos_solver_rve.py implementation for Neo-Hookean consistency.
        f_tensor = DeformationGradientFromGreenLagrange2D(self.batch_strain)
        
        mp = self._GetSolver().GetComputingModelPart()
        self._InitializeDomainCenterIfNeeded(mp)
        x0c, y0c = self._x0c, self._y0c

        for node in mp.Nodes:
            xr = node.X0 - x0c
            yr = node.Y0 - y0c

            ux = (f_tensor[0, 0] - 1.0) * xr + f_tensor[0, 1] * yr
            uy = f_tensor[1, 0] * xr + (f_tensor[1, 1] - 1.0) * yr

            if node.IsFixed(KM.DISPLACEMENT_X):
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, ux)
            if node.IsFixed(KM.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Y, uy)
            if node.IsFixed(KM.DISPLACEMENT_Z):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)


def generate_safe_straight_path(emax, rel6):
    """
    Synchronized safe segmented trajectory from Stage 4.
    Used for benchmarking unseen multi-axial loading.
    """
    exx_pos, exx_neg, eyy_pos, eyy_neg, gxy_pos, gxy_neg = rel6
    
    # Define multi-axial waypoints within the ellipsoid
    pts = [
        [0.0, 0.0, 0.0],
        [0.8*exx_pos, 0.4*eyy_pos, 0.0],
        [0.85*exx_pos, 0.5*eyy_neg, 0.3*gxy_pos],
        [0.4*exx_neg, 0.7*eyy_pos, 0.45*gxy_neg],
        [0.2*exx_pos, 0.8*eyy_pos, 0.1*gxy_pos],
        [0.0, 0.0, 0.0]
    ]
    control_points = np.array(pts) * emax
    
    # Linear interpolation (n_interp = 50 per segment)
    full_path = [control_points[0]]
    for i in range(len(control_points)-1):
        p0, p1 = control_points[i], control_points[i+1]
        n_interp = 50
        for alpha in np.linspace(0, 1, n_interp+1)[1:]:
            full_path.append(p0 + alpha*(p1 - p0))
            
    return control_points, np.array(full_path)


# =============================================================================
# Path utilities
# =============================================================================


def BuildDynamicSegmentSteps(
    path_points,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    min_steps=MIN_STEPS_PER_SEGMENT,
    reference_amplitude=None,
):
    p = np.asarray(path_points, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if p.ndim != 2 or p.shape[0] < 2:
        raise ValueError("path_points must have shape (n_waypoints, n_components), with n_waypoints >= 2.")

    dp = np.diff(p, axis=0)
    # Consistency with Stage 0: use Exy = 0.5 * Gxy for segment length calculation
    dp_scaled = dp.copy()
    if dp_scaled.shape[1] == 3:
        dp_scaled[:, 2] *= 0.5

    seg_len = np.linalg.norm(dp_scaled, axis=1)
    if reference_amplitude is None:
        p_scaled = p.copy()
        if p_scaled.shape[1] == 3:
            p_scaled[:, 2] *= 0.5
        ref_amp = float(np.max(np.linalg.norm(p_scaled, axis=1)))
        if ref_amp <= 0.0:
            ref_amp = float(np.max(seg_len))
    else:
        ref_amp = float(abs(reference_amplitude))
    if (not np.isfinite(ref_amp)) or ref_amp <= 0.0:
        ref_amp = 1.0

    seg_steps = np.maximum(
        int(min_steps),
        np.ceil(float(reference_steps) * seg_len / ref_amp).astype(int),
    )
    return seg_steps, ref_amp


# =============================================================================
# Trajectory Loading (NPZ/JSON/CLI)
# =============================================================================


def _ValidateStrainPathArray(arr, source):
    p = np.array(arr, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] != 3:
        raise ValueError(f"{source} must contain a (n_waypoints,3) array with n_waypoints >= 2.")
    return p


def LoadStrainWaypointsFromFile(path, trajectory_index=1):
    import json
    if int(trajectory_index) < 1:
        raise ValueError("trajectory-index must be >= 1.")

    fpath = str(path)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"strain-waypoints-file not found: {fpath}")

    ext = os.path.splitext(fpath)[1].lower()
    meta = {}

    if ext == ".npy":
        arr = np.load(fpath)
        traj = _ValidateStrainPathArray(arr, "npy file")
        return traj, meta

    if ext == ".npz":
        data = np.load(fpath)
        idx = int(trajectory_index)
        key_main = f"trajectory_{idx}"
        if key_main in data:
            arr = data[key_main]
        elif "trajectory" in data:
            arr = data["trajectory"]
        else:
            available = ", ".join(list(data.keys()))
            raise ValueError(f"npz file does not contain '{key_main}'. Available keys: {available}")

        if "ref_steps" in data:
            meta["ref_steps"] = int(np.ravel(data["ref_steps"])[0])
        if "reference_amplitude" in data:
            meta["reference_amplitude"] = float(np.ravel(data["reference_amplitude"])[0])

        traj = _ValidateStrainPathArray(arr, "npz trajectory")
        return traj, meta

    if ext == ".json":
        with open(fpath, "r") as f:
            obj = json.load(f)
        idx = int(trajectory_index)
        key_main = f"trajectory_{idx}"
        if isinstance(obj, dict):
            if key_main in obj: arr = obj[key_main]
            elif "trajectory" in obj: arr = obj["trajectory"]
            else: raise ValueError("json file missing 'trajectory' key.")
        else:
            arr = obj
        traj = _ValidateStrainPathArray(arr, "json trajectory")
        return traj, meta

    raise ValueError(f"Unsupported extension: '{ext}'")


def StripMdpaExtension(mesh_name):
    s = str(mesh_name)
    if s.endswith(".mdpa"):
        return s[:-5]
    return s


def SetInputMeshFilename(parameters, mesh_name):
    mesh_base = StripMdpaExtension(mesh_name)
    parameters["modelers"][0]["parameters"]["input_filename"].SetString(mesh_base)
    parameters["problem_data"]["problem_name"].SetString(mesh_base)


def FindSubModelPartNamesInMdpa(mdpa_path):
    names = []
    if not os.path.exists(mdpa_path):
        return names
    with open(mdpa_path, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("Begin SubModelPart "):
                parts = s.split()
                if len(parts) >= 3:
                    names.append(parts[2])
    return names


def _MaterialPartSortKey(name):
    if name == "material": return (0, 0, name)
    if name.startswith("material_"):
        suffix = name.split("_", 1)[1]
        if suffix.isdigit(): return (1, int(suffix), name)
    return (2, 0, name)


def DetectMaterialSubModelParts(mdpa_path):
    names = FindSubModelPartNamesInMdpa(mdpa_path)
    mats = sorted([n for n in names if n.startswith("material")], key=_MaterialPartSortKey)
    if not mats:
        raise RuntimeError(f"No material* submodelparts found in mdpa: {mdpa_path}")
    return mats


def ConfigureElementModelerForMaterialParts(parameters, material_parts):
    import json
    # Use PrettyPrintJsonString to get a python-parsable string from Kratos Parameters
    params_dict = json.loads(parameters.PrettyPrintJsonString())
    
    # Usually modelers[1] is the ElementCreationModeler
    if len(params_dict.get("modelers", [])) > 1:
        old_list = params_dict["modelers"][1]["parameters"].get("elements_list", [])
        element_name = old_list[0]["element_name"] if len(old_list) > 0 else "TotalLagrangianElement2D6N"
        
        params_dict["modelers"][1]["parameters"]["elements_list"] = [
            {"model_part_name": f"Structure.{name}", "element_name": element_name}
            for name in material_parts
        ]
    return KM.Parameters(json.dumps(params_dict))


def WriteRuntimeMaterialsFile(
    material_parts,
    young_mpa,
    poisson,
    out_filename="StructuralMaterials.runtime.json",
    density=7850.0,
    thickness=0.05,
):
    import json
    young_pa = float(young_mpa) * 1.0e6
    props = []
    for i, name in enumerate(material_parts, start=1):
        props.append({
            "model_part_name": f"Structure.{name}",
            "properties_id": i,
            "Material": {
                "constitutive_law": {"name": "HyperElasticPlaneStrain2DLaw"},
                "Variables": {
                    "DENSITY": float(density),
                    "YOUNG_MODULUS": float(young_pa),
                    "POISSON_RATIO": float(poisson),
                    "THICKNESS": float(thickness),
                },
                "Tables": None,
            },
        })
    with open(out_filename, "w") as f:
        json.dump({"properties": props}, f, indent=4)
    return out_filename


def setup_kratos_parameters(mesh="rve_geometry"):
    """
    High-level ProjectParameters setup with auto-detected material parts.
    """
    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())

    # Keep solver output quiet for performance and cleaner logs.
    if parameters.Has("problem_data") and parameters["problem_data"].Has("echo_level"):
        parameters["problem_data"]["echo_level"].SetInt(0)
    if parameters.Has("solver_settings") and parameters["solver_settings"].Has("echo_level"):
        parameters["solver_settings"]["echo_level"].SetInt(0)
    
    SetInputMeshFilename(parameters, mesh)
    mdpa_path = f"{StripMdpaExtension(mesh)}.mdpa"
    
    if os.path.exists(mdpa_path):
        material_parts = DetectMaterialSubModelParts(mdpa_path)
        parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
        runtime_materials = WriteRuntimeMaterialsFile(material_parts, 1628.0, 0.4)
        parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(runtime_materials)
    
    return parameters


def ParseStrainWaypoints(path_string):
    rows = []
    for chunk in str(path_string).split(";"):
        chunk = chunk.strip()
        if not chunk: continue
        vals = [float(x.strip()) for x in chunk.split(",")]
        if len(vals) != 3:
            raise ValueError("Each waypoint must have 3 values: Ex,Ey,Gxy")
        rows.append(vals)
    return np.array(rows, dtype=float)


# =============================================================================
# Nonlinear controls
# =============================================================================


def _GetSettingInt(parameters, key, default):
    if parameters.Has("solver_settings") and parameters["solver_settings"].Has(key):
        return int(parameters["solver_settings"][key].GetInt())
    return int(default)


def _GetSettingFloat(parameters, key, default):
    if parameters.Has("solver_settings") and parameters["solver_settings"].Has(key):
        return float(parameters["solver_settings"][key].GetDouble())
    return float(default)


def _GetSettingBool(parameters, key, default):
    if parameters.Has("solver_settings") and parameters["solver_settings"].Has(key):
        return bool(parameters["solver_settings"][key].GetBool())
    return bool(default)


def _GetSettingString(parameters, key, default):
    if parameters.Has("solver_settings") and parameters["solver_settings"].Has(key):
        return str(parameters["solver_settings"][key].GetString())
    return str(default)


def _NormalizeConvergenceCriterion(criterion_raw):
    s = str(criterion_raw).strip().lower()
    if s in ("residual", "residual_criterion"):
        return "residual"
    if s in ("displacement", "displacement_criterion"):
        return "displacement"
    if s in ("both", "and_criterion"):
        return "both"
    if s in ("either", "or_criterion"):
        return "either"

    has_res = "residual" in s
    has_disp = "displacement" in s
    if has_res and has_disp:
        if "or" in s:
            return "either"
        return "both"
    if has_res:
        return "residual"
    if has_disp:
        return "displacement"
    return CONVERGENCE_CRITERION


def ResolveNonlinearControls(
    parameters,
    max_newton_it=None,
    use_old_stiffness_in_first_iteration=None,
    use_line_search=None,
    convergence_criterion=None,
    residual_relative_tolerance=None,
    residual_absolute_tolerance=None,
    displacement_relative_tolerance=None,
    displacement_absolute_tolerance=None,
):
    cfg = {
        "max_newton_it": _GetSettingInt(parameters, "max_iteration", 10),
        "use_old_stiffness_in_first_iteration": _GetSettingBool(
            parameters,
            "use_old_stiffness_in_first_iteration",
            USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
        ),
        "use_line_search": _GetSettingBool(parameters, "line_search", USE_LINE_SEARCH),
        "convergence_criterion": _NormalizeConvergenceCriterion(
            _GetSettingString(parameters, "convergence_criterion", CONVERGENCE_CRITERION)
        ),
        "residual_relative_tolerance": _GetSettingFloat(
            parameters, "residual_relative_tolerance", NEWTON_TOL_REL
        ),
        "residual_absolute_tolerance": _GetSettingFloat(
            parameters, "residual_absolute_tolerance", NEWTON_TOL_ABS
        ),
        "displacement_relative_tolerance": _GetSettingFloat(
            parameters, "displacement_relative_tolerance", DISP_TOL_REL
        ),
        "displacement_absolute_tolerance": _GetSettingFloat(
            parameters, "displacement_absolute_tolerance", DISP_TOL_ABS
        ),
    }

    if max_newton_it is not None:
        cfg["max_newton_it"] = int(max_newton_it)
    if use_old_stiffness_in_first_iteration is not None:
        cfg["use_old_stiffness_in_first_iteration"] = bool(use_old_stiffness_in_first_iteration)
    if use_line_search is not None:
        cfg["use_line_search"] = bool(use_line_search)
    if convergence_criterion is not None:
        cfg["convergence_criterion"] = _NormalizeConvergenceCriterion(convergence_criterion)
    if residual_relative_tolerance is not None:
        cfg["residual_relative_tolerance"] = float(residual_relative_tolerance)
    if residual_absolute_tolerance is not None:
        cfg["residual_absolute_tolerance"] = float(residual_absolute_tolerance)
    if displacement_relative_tolerance is not None:
        cfg["displacement_relative_tolerance"] = float(displacement_relative_tolerance)
    if displacement_absolute_tolerance is not None:
        cfg["displacement_absolute_tolerance"] = float(displacement_absolute_tolerance)

    return cfg


def CheckConvergenceStatus(criterion, res_ok, disp_ok):
    if criterion == "residual":
        return bool(res_ok)
    if criterion == "displacement":
        return bool(disp_ok)
    if criterion == "either":
        return bool(res_ok or disp_ok)
    return bool(res_ok and disp_ok)


# =============================================================================
# Driver
# =============================================================================


def RunFomBatchSimulation(
    parameters,
    out_dir=DEFAULT_OUTPUT_DIR,
    save_plot=True,
    strain_path=None,
    trajectory_index=None,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    max_newton_it=None,
    use_old_stiffness_in_first_iteration=None,
    use_line_search=None,
    convergence_criterion=None,
    residual_relative_tolerance=None,
    residual_absolute_tolerance=None,
    displacement_relative_tolerance=None,
    displacement_absolute_tolerance=None,
    use_fast_dirichlet_bc=True,
):
    """Executes the RVE simulation for a given strain trajectory."""
    os.makedirs(out_dir, exist_ok=True)
    dt = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()

    if strain_path is None:
        raise ValueError("strain_path must be provided to RunFomBatchSimulation.")
    
    E_wp = np.array(strain_path, dtype=float)
    n_wp = E_wp.shape[0]
    n_seg = n_wp - 1

    seg_steps, ref_amp_used = BuildDynamicSegmentSteps(
        E_wp,
        reference_steps=reference_steps,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=reference_amplitude,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    end_time = dt * float(n_steps_total)

    if n_steps_total <= 0:
        raise RuntimeError("Dynamic step allocation produced zero total steps.")

    controls = ResolveNonlinearControls(
        parameters,
        max_newton_it=max_newton_it,
        use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
        use_line_search=use_line_search,
        convergence_criterion=convergence_criterion,
        residual_relative_tolerance=residual_relative_tolerance,
        residual_absolute_tolerance=residual_absolute_tolerance,
        displacement_relative_tolerance=displacement_relative_tolerance,
        displacement_absolute_tolerance=displacement_absolute_tolerance,
    )
    print(f"[FOM] Starting trajectory. Total steps: {n_steps_total}")
    print(
        "[FOM] Nonlinear controls: "
        f"criterion={controls['convergence_criterion']}, "
        f"line_search={controls['use_line_search']}, "
        f"max_it={controls['max_newton_it']}, "
        f"res_tol=(rel {controls['residual_relative_tolerance']:.2e}, abs {controls['residual_absolute_tolerance']:.2e}), "
        f"disp_tol=(rel {controls['displacement_relative_tolerance']:.2e}, abs {controls['displacement_absolute_tolerance']:.2e}), "
        f"use_old_K0={controls['use_old_stiffness_in_first_iteration']}"
    )

    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()

    mp = sim._GetSolver().GetComputingModelPart()
    elements = list(mp.Elements)
    entities = list(mp.Elements) + list(mp.Conditions)
    n_dof, eq_map, ta_disp = SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    # Fast path precomputation: fixed/free partition and node reference coordinates
    # for analytical affine Dirichlet values from Green-Lagrange strain.
    if use_fast_dirichlet_bc:
        sim._InitializeDomainCenterIfNeeded(mp)
        x0c, y0c = float(sim._x0c), float(sim._y0c)
        (
            dir_dofs_fast,
            dir_x_fast,
            dir_y_fast,
            dir_is_x_fast,
            free_dofs_fast,
        ) = PrecomputeDirichletPartitionFromNodes(mp, n_dof, x0c, y0c)
    else:
        dir_dofs_fast = None
        dir_x_fast = None
        dir_y_fast = None
        dir_is_x_fast = None
        free_dofs_fast = None

    u_n = np.zeros(n_dof, dtype=float)
    U_hist, strain_hist, stress_hist, applied_strain_hist = [], [], [], []
    
    # Store initial state (origin)
    U_hist.append(u_n.copy())
    strain_hist.append(np.zeros(3, dtype=float))
    stress_hist.append(np.zeros(3, dtype=float))
    applied_strain_hist.append(np.zeros(3, dtype=float))
    
    K_old = None

    for step in range(1, n_steps_total + 1):
        time_val = float(step) * float(dt)
        mp.CloneTimeStep(time_val)
        mp.ProcessInfo[KM.DELTA_TIME] = dt
        mp.ProcessInfo[KM.TIME] = time_val
        mp.ProcessInfo[KM.STEP] = step

        sim.time, sim.step, sim.end_time = time_val, step, end_time
        sim.InitializeSolutionStep()

        # Interpolate waypoints
        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        E_t = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]

        sim.batch_strain = E_t.copy()
        applied_strain_hist.append(E_t.copy())
        if use_fast_dirichlet_bc:
            dir_dofs = dir_dofs_fast
            free_dofs = free_dofs_fast
            dir_vals = ComputeDirichletValuesFromGreenLagrange(
                E_t, dir_x_fast, dir_y_fast, dir_is_x_fast
            )
        else:
            sim.ApplyBoundaryConditions()
            dir_dofs, dir_vals = ExtractDirichletBoundaryConditions(mp, step_index=0)
            is_free = np.ones(n_dof, dtype=bool)
            is_free[dir_dofs] = False
            free_dofs = np.nonzero(is_free)[0].astype(int)

        # Solve Newton iterations

        u_pred = u_n.copy()
        u_pred[dir_dofs] = dir_vals
        u = u_pred.copy()
        
        print(
            f"\n[FOM] Step {step:03d} | t={time_val:.6f} | "
            f"fixed={dir_dofs.size}, free={free_dofs.size}"
        )

        converged = False
        Kff_last = None
        nR0 = None
        nD0 = None
        nR_last = np.nan
        nD_last = np.nan
        for it in range(int(controls["max_newton_it"])):
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1
            SetDisplacementFromEquationVector(u, eq_map, ta_disp)
            UpdateCurrentCoordinatesFromDisplacement(mp, step=0)

            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            K, rhs = AssembleGlobalSystem(mp, n_dof, entities)
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            r_f = rhs[free_dofs]
            nR = np.linalg.norm(r_f)
            nR_last = nR

            if nR0 is None:
                nR0 = max(float(nR), 1.0e-30)
            r_rel = float(nR) / float(nR0)
            res_ok = (nR <= controls["residual_absolute_tolerance"]) or (
                r_rel <= controls["residual_relative_tolerance"]
            )

            Kff_curr = K[free_dofs][:, free_dofs]
            Kff_last = Kff_curr
            K_solve = Kff_curr
            used_old = False
            if (
                it == 0
                and controls["use_old_stiffness_in_first_iteration"]
                and K_old is not None
                and K_old.shape == Kff_curr.shape
            ):
                K_solve = K_old
                used_old = True

            try:
                du = spsolve(K_solve, r_f)
            except Exception:
                if used_old:
                    du = spsolve(Kff_curr, r_f)
                    used_old = False
                else:
                    raise

            alpha = 1.0
            if controls["use_line_search"]:
                alpha = ComputeLineSearchAlpha(
                    mp=mp,
                    u_base=u,
                    du_free=du,
                    dir_dofs=dir_dofs,
                    dir_vals=dir_vals,
                    free_dofs=free_dofs,
                    eq_map=eq_map,
                    ta_disp=ta_disp,
                    n_dof=n_dof,
                    entities=entities,
                )
            du_eff = alpha * du
            nD = np.linalg.norm(du_eff)
            nD_last = nD
            if nD0 is None:
                nD0 = max(float(nD), 1.0e-30)
            d_rel = float(nD) / float(nD0)
            disp_ok = (nD <= controls["displacement_absolute_tolerance"]) or (
                d_rel <= controls["displacement_relative_tolerance"]
            )

            solve_tag = " (K_old)" if used_old else ""
            alpha_tag = f", alpha={alpha:.3e}" if controls["use_line_search"] else ""
            print(
                f"  > It {it:02d}: ||R||={nR:.3e} (rel {r_rel:.3e}), "
                f"||du||={nD:.3e} (rel {d_rel:.3e}){alpha_tag}{solve_tag}"
            )

            u[free_dofs] += du_eff
            converged = CheckConvergenceStatus(
                controls["convergence_criterion"], res_ok=res_ok, disp_ok=disp_ok
            )
            if converged:
                print(f"  > It {it:02d}: converged by '{controls['convergence_criterion']}' criterion")
                break

        if not converged:
            msg = (
                f"[ERROR] Step {step} failed to converge after {controls['max_newton_it']} iterations. "
                f"criterion={controls['convergence_criterion']}, "
                f"last ||R||={nR_last:.3e}, last ||du||={nD_last:.3e}. "
                f"Consider increasing --ref-steps for smaller strain increments."
            )
            print(msg)
            raise RuntimeError(msg)

        if Kff_last is not None:
            K_old = Kff_last.copy()

        # Post-process step
        # Ensure the converged displacement field is explicitly synchronized
        # to the model part before Gauss-point postprocessing.
        SetDisplacementFromEquationVector(u, eq_map, ta_disp)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)
        # Force a full LocalSystem evaluation at converged state so that
        # constitutive responses cached by the elements are synchronized.
        # NOTE: RHS-only evaluation may not update all stress/tangent-related
        # internal quantities for some elements.
        InitializeNonLinearIteration(entities, mp.ProcessInfo)
        _, _ = AssembleGlobalSystem(mp, n_dof, entities)
        FinalizeNonLinearIteration(entities, mp.ProcessInfo)
        u_n = u.copy()
        eps_h, sig_h = CalculateHomogenizedStressAndStrainKratosReference(mp)

        strain_hist.append(eps_h)
        stress_hist.append(sig_h)
        U_hist.append(u.copy())

        sim.FinalizeSolutionStep()
        
    sim.Finalize()
    
    # Save results with simple naming
    tag = f"trajectory_{trajectory_index}" if trajectory_index else "single_run"
    np.save(os.path.join(out_dir, f"{tag}_strain.npy"), np.stack(strain_hist))
    np.save(os.path.join(out_dir, f"{tag}_stress.npy"), np.stack(stress_hist))
    np.save(os.path.join(out_dir, f"{tag}_U.npy"), np.stack(U_hist))
    np.save(os.path.join(out_dir, f"{tag}_applied_strain.npy"), np.stack(applied_strain_hist))

    if save_plot:
        _save_diagnostic_plots(np.stack(strain_hist), np.stack(stress_hist), out_dir, tag)

    return strain_hist, stress_hist


def _save_diagnostic_plots(strain_hist, stress_hist, out_dir, tag):
    Exx, Eyy, Exy = strain_hist[:, 0], strain_hist[:, 1], strain_hist[:, 2]
    Sxx, Syy, Sxy = stress_hist[:, 0], stress_hist[:, 1], stress_hist[:, 2]

    plt.figure()
    plt.plot(Exx, Sxx, label="sigma_xx"); plt.plot(Eyy, Syy, label="sigma_yy"); plt.plot(Exy, Sxy, label="sigma_xy")
    plt.xlabel("Strain [-]"); plt.ylabel("Stress [Pa]"); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(out_dir, f"{tag}_plots.png")); plt.close()


def main():
    import argparse
    p = argparse.ArgumentParser(description="FOM RVE solver (manual assembly, KRATOS-consistent homogenization)")
    p.add_argument("--mesh", type=str, default=None, help="Mesh base name.")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--emax", type=float, default=0.10)
    p.add_argument("--strain-waypoints", type=str, default=None)
    p.add_argument("--strain-waypoints-file", type=str, default=None)
    p.add_argument("--trajectory-index", type=int, default=1)
    p.add_argument("--ref-steps", type=int, default=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE)
    p.add_argument("--reference-amplitude", type=float, default=None)
    p.add_argument(
        "--max-newton-it",
        type=int,
        default=None,
        help="Override max nonlinear iterations (default: read ProjectParameters).",
    )
    p.add_argument("--young-mpa", type=float, default=1628.0)
    p.add_argument("--poisson", type=float, default=0.4)
    p.add_argument(
        "--convergence-criterion",
        type=str,
        default=None,
        choices=["residual", "displacement", "both", "either"],
        help="Override nonlinear convergence criterion (default: read ProjectParameters).",
    )
    p.add_argument(
        "--residual-relative-tolerance",
        type=float,
        default=None,
        help="Override residual relative tolerance (default: read ProjectParameters).",
    )
    p.add_argument(
        "--residual-absolute-tolerance",
        type=float,
        default=None,
        help="Override residual absolute tolerance (default: read ProjectParameters).",
    )
    p.add_argument(
        "--displacement-relative-tolerance",
        type=float,
        default=None,
        help="Override displacement relative tolerance (default: read ProjectParameters).",
    )
    p.add_argument(
        "--displacement-absolute-tolerance",
        type=float,
        default=None,
        help="Override displacement absolute tolerance (default: read ProjectParameters).",
    )
    p.add_argument(
        "--line-search",
        dest="line_search",
        action="store_true",
        help="Force-enable line search (default: read ProjectParameters).",
    )
    p.add_argument(
        "--no-line-search",
        dest="line_search",
        action="store_false",
        help="Force-disable line search (default: read ProjectParameters).",
    )
    p.add_argument(
        "--old-stiffness-first-it",
        dest="old_stiffness_first_it",
        action="store_true",
        help="Force-enable reuse of previous-step stiffness in iteration 0.",
    )
    p.add_argument(
        "--no-old-stiffness-first-it",
        dest="old_stiffness_first_it",
        action="store_false",
        help="Force-disable reuse of previous-step stiffness in iteration 0.",
    )
    p.set_defaults(line_search=None, old_stiffness_first_it=None)
    args = p.parse_args()

    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())

    # --- Automatic Material Detection (Synchronized with Kratos solver) ---
    if args.mesh is not None:
        SetInputMeshFilename(parameters, args.mesh)
    
    mesh_base = parameters["modelers"][0]["parameters"]["input_filename"].GetString()
    mdpa_path = f"{StripMdpaExtension(mesh_base)}.mdpa"
    
    if os.path.exists(mdpa_path):
        material_parts = DetectMaterialSubModelParts(mdpa_path)
        parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
        
        runtime_materials = WriteRuntimeMaterialsFile(
            material_parts=material_parts,
            young_mpa=args.young_mpa,
            poisson=args.poisson,
        )
        parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(runtime_materials)
        print(f"[INFO] Auto-detected material parts: {material_parts}")
        print(f"[INFO] Generated runtime materials file: {runtime_materials}")
    # -----------------------------------------------------------------------

    strain_path = None
    if args.strain_waypoints_file:
        strain_path, meta = LoadStrainWaypointsFromFile(args.strain_waypoints_file, args.trajectory_index)
        if args.reference_amplitude is None and "reference_amplitude" in meta:
            args.reference_amplitude = meta["reference_amplitude"]
        if args.ref_steps == REFERENCE_STEPS_FOR_UNIT_AMPLITUDE and "ref_steps" in meta:
            args.ref_steps = meta["ref_steps"]
    elif args.strain_waypoints:
        strain_path = ParseStrainWaypoints(args.strain_waypoints)

    if strain_path is not None:
        print(f"[INFO] Running trajectory index {args.trajectory_index}")
        try:
            RunFomBatchSimulation(
                parameters=parameters,
                out_dir=args.out_dir,
                save_plot=True,
                strain_path=strain_path,
                trajectory_index=args.trajectory_index,
                reference_amplitude=args.reference_amplitude if args.reference_amplitude else args.emax,
                reference_steps=args.ref_steps,
                max_newton_it=args.max_newton_it,
                use_old_stiffness_in_first_iteration=args.old_stiffness_first_it,
                use_line_search=args.line_search,
                convergence_criterion=args.convergence_criterion,
                residual_relative_tolerance=args.residual_relative_tolerance,
                residual_absolute_tolerance=args.residual_absolute_tolerance,
                displacement_relative_tolerance=args.displacement_relative_tolerance,
                displacement_absolute_tolerance=args.displacement_absolute_tolerance,
            )
        except RuntimeError as e:
            print(f"\n[FATAL] Simulation failed: {e}")
            sys.exit(1)
    else:
        print("[ERROR] Please provide --strain-waypoints-file or --strain-waypoints.")
        sys.exit(1)


if __name__ == "__main__":
    main()
