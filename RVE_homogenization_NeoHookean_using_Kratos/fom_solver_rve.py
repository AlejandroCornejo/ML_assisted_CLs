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

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# =============================================================================
# Vectorized Constitutive Law (Neo-Hookean Plane Strain 2D)
# Matches Kratos HyperElasticIsotropicNeoHookeanPlaneStrain2D exactly.
# =============================================================================

def _neo_hookean_pk2_2d_vectorized(E_voigt, young, poisson):
    """
    Vectorized PK2 stress and tangent for compressible Neo-Hookean plane strain.
    E_voigt: (N, 3) with [E11, E22, 2*E12]
    Returns: S_voigt (N, 3), C_voigt (N, 3, 3)
    """
    mu = young / (2.0 * (1.0 + poisson))
    lmbda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    E11, E22, G12 = E_voigt[:, 0], E_voigt[:, 1], E_voigt[:, 2]
    E12 = 0.5 * G12

    C11 = 2.0 * E11 + 1.0
    C22 = 2.0 * E22 + 1.0
    C12 = 2.0 * E12

    detC = C11 * C22 - C12 * C12
    J = np.sqrt(np.maximum(detC, 1e-30))
    lnJ = np.log(J)
    invDetC = 1.0 / detC

    # C^{-1} components
    Ci11 = C22 * invDetC
    Ci22 = C11 * invDetC
    Ci12 = -C12 * invDetC

    # S = lambda*ln(J)*C^{-1} + mu*(I - C^{-1})
    N = E_voigt.shape[0]
    S = np.empty((N, 3))
    S[:, 0] = lmbda * lnJ * Ci11 + mu * (1.0 - Ci11)
    S[:, 1] = lmbda * lnJ * Ci22 + mu * (1.0 - Ci22)
    S[:, 2] = lmbda * lnJ * Ci12 + mu * (0.0 - Ci12)

    # Tangent C_ijkl using Voigt map: 0->(0,0), 1->(1,1), 2->(0,1)
    Ci = np.empty((N, 2, 2))
    Ci[:, 0, 0] = Ci11; Ci[:, 1, 1] = Ci22
    Ci[:, 0, 1] = Ci12; Ci[:, 1, 0] = Ci12

    voigt2d = [(0, 0), (1, 1), (0, 1)]
    CC = np.empty((N, 3, 3))
    coeff = mu - lmbda * lnJ  # (N,)
    for iv in range(3):
        i0, i1 = voigt2d[iv]
        for jv in range(3):
            j0, j1 = voigt2d[jv]
            CC[:, iv, jv] = (
                lmbda * Ci[:, i0, i1] * Ci[:, j0, j1]
                + coeff * (Ci[:, i0, j0] * Ci[:, i1, j1] + Ci[:, i0, j1] * Ci[:, i1, j0])
            )

    return S, CC


# =============================================================================
# Triangle2D6 shape function local gradients
# =============================================================================

def _tri6_DN_local(xi, eta):
    DN = np.zeros((6, 2))
    DN[0, 0] = 4*xi + 4*eta - 3;  DN[0, 1] = 4*xi + 4*eta - 3
    DN[1, 0] = 4*xi - 1;          DN[1, 1] = 0.0
    DN[2, 0] = 0.0;               DN[2, 1] = 4*eta - 1
    DN[3, 0] = 4 - 8*xi - 4*eta;  DN[3, 1] = -4*xi
    DN[4, 0] = 4*eta;             DN[4, 1] = 4*xi
    DN[5, 0] = -4*eta;            DN[5, 1] = 4 - 4*xi - 8*eta
    return DN


# =============================================================================
# Vectorized Assembler
# =============================================================================

class VectorizedAssembler:
    """
    Replaces per-element CalculateLocalSystem calls with bulk NumPy operations.
    One-time init extracts connectivity, DN, weights from Kratos geometry objects.
    Per-iteration Assemble() computes K and RHS via tensor contractions.
    """

    def __init__(self, mp, n_dof, eq_map, elements=None, element_scales=None, log_label="VectorizedAssembler"):
        self.n_dof = n_dof
        self.eq_map = eq_map

        if elements is None:
            elements = list(mp.Elements)
        else:
            elements = list(elements)
        self.n_elems = len(elements)
        if self.n_elems == 0:
            return

        self.n_nodes = len(elements[0].GetGeometry())
        self.n_local_dof = self.n_nodes * 2

        # --- Material properties per element ---
        self.young = np.empty(self.n_elems)
        self.poisson = np.empty(self.n_elems)
        self.thickness = np.empty(self.n_elems)
        for e, elem in enumerate(elements):
            props = elem.Properties
            self.young[e] = props[KM.YOUNG_MODULUS]
            self.poisson[e] = props[KM.POISSON_RATIO]
            self.thickness[e] = props[KM.THICKNESS] if props.Has(KM.THICKNESS) else 1.0

        # Check if all elements share same material (common case → faster path)
        self._uniform_material = (
            np.all(self.young == self.young[0]) and
            np.all(self.poisson == self.poisson[0])
        )
        if self._uniform_material:
            self._young_scalar = float(self.young[0])
            self._poisson_scalar = float(self.poisson[0])

        # --- Connectivity: node_id → 0-based index ---
        max_id = max(node.Id for node in mp.Nodes)
        id_to_idx = np.zeros(max_id + 1, dtype=int)
        for i, node in enumerate(mp.Nodes):
            id_to_idx[node.Id] = i

        self.connectivity = np.array(
            [[id_to_idx[node.Id] for node in elem.GetGeometry()] for elem in elements]
        )  # (n_elems, n_nodes)

        # local equation IDs: (n_elems, n_local_dof)
        # eq_map is (n_nodes_total, 2), connectivity indexes into it
        self.local_eq_ids = eq_map[self.connectivity].reshape(self.n_elems, -1)

        if element_scales is None:
            self.element_scales = np.ones(self.n_elems, dtype=float)
        else:
            self.element_scales = np.asarray(element_scales, dtype=float).reshape(-1)
            if self.element_scales.size != self.n_elems:
                raise ValueError(
                    f"element_scales size {self.element_scales.size} does not match "
                    f"number of elements {self.n_elems}."
                )

        # --- Shape function derivatives and integration weights ---
        self.n_gauss = elements[0].GetGeometry().IntegrationPointsNumber()
        self.DN = np.zeros((self.n_elems, self.n_gauss, self.n_nodes, 2))
        self.w_detJ = np.zeros((self.n_elems, self.n_gauss))
        self.area_e = np.zeros(self.n_elems, dtype=float)

        for e, elem in enumerate(elements):
            geom = elem.GetGeometry()
            coords = np.array([[geom[i].X0, geom[i].Y0] for i in range(self.n_nodes)])
            self.area_e[e] = float(geom.Area())
            ips = geom.IntegrationPoints()

            for g in range(self.n_gauss):
                xi, eta = ips[g][0], ips[g][1]
                w = ips[g][3]
                DN_local = _tri6_DN_local(xi, eta)
                J = coords.T @ DN_local  # (2,2) — Kratos convention
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                self.DN[e, g] = DN_local @ invJ
                self.w_detJ[e, g] = w * detJ * self.thickness[e]
        self.w_detJ *= self.element_scales[:, None]

        # --- Precompute COO indices for sparse assembly ---
        self.rows_K = np.repeat(self.local_eq_ids, self.n_local_dof, axis=1).flatten()
        self.cols_K = np.tile(self.local_eq_ids, (1, self.n_local_dof)).flatten()
        self.rows_R = self.local_eq_ids.flatten()

        # --- Workspace buffers reused on each Assemble() call ---
        self._u_node = np.empty((self.n_elems, self.n_nodes, 2), dtype=float)
        self._grad_u = np.empty((self.n_elems, self.n_gauss, 2, 2), dtype=float)
        self._F = np.empty((self.n_elems, self.n_gauss, 2, 2), dtype=float)
        self._FtF = np.empty((self.n_elems, self.n_gauss, 2, 2), dtype=float)
        self._E_voigt = np.empty((self.n_elems, self.n_gauss, 3), dtype=float)
        self._S_voigt = np.empty((self.n_elems, self.n_gauss, 3), dtype=float)
        self._CC = np.empty((self.n_elems, self.n_gauss, 3, 3), dtype=float)
        self._St = np.empty((self.n_elems, self.n_gauss, 2, 2), dtype=float)
        self._P = np.empty((self.n_elems, self.n_gauss, 2, 2), dtype=float)
        self._tmp_ag_ai = np.empty((self.n_elems, self.n_gauss, self.n_nodes, 2), dtype=float)
        self._f_int = np.empty((self.n_elems, self.n_nodes, 2), dtype=float)
        self._B = np.empty((self.n_elems, self.n_gauss, 3, self.n_local_dof), dtype=float)
        self._CB = np.empty((self.n_elems, self.n_gauss, 3, self.n_local_dof), dtype=float)
        self._K_gp = np.empty((self.n_elems, self.n_gauss, self.n_local_dof, self.n_local_dof), dtype=float)
        self._K_mat = np.empty((self.n_elems, self.n_local_dof, self.n_local_dof), dtype=float)
        self._DNSt = np.empty((self.n_elems, self.n_gauss, self.n_nodes, 2), dtype=float)
        self._S_DN_DN_gp = np.empty((self.n_elems, self.n_gauss, self.n_nodes, self.n_nodes), dtype=float)
        self._S_DN_DN = np.empty((self.n_elems, self.n_nodes, self.n_nodes), dtype=float)
        self._K_geo = np.empty((self.n_elems, self.n_local_dof, self.n_local_dof), dtype=float)
        self._K_total = np.empty((self.n_elems, self.n_local_dof, self.n_local_dof), dtype=float)
        self._rhs = np.zeros(self.n_dof, dtype=float)

        print(f"[{log_label}] {self.n_elems} elements, {self.n_gauss} GPs, "
              f"total area={np.sum(self.w_detJ):.6e}")

    def Assemble(self, u_global):
        """Compute global K (sparse CSR) and RHS from current displacement vector."""
        if self.n_elems == 0:
            return csr_matrix((self.n_dof, self.n_dof)), np.zeros(self.n_dof)

        ne, ng, nn = self.n_elems, self.n_gauss, self.n_nodes
        ndl = self.n_local_dof

        # 1. Gather nodal displacements -> (ne, nn, 2)
        self._u_node[..., 0] = u_global[self.eq_map[:, 0]][self.connectivity]
        self._u_node[..., 1] = u_global[self.eq_map[:, 1]][self.connectivity]

        # 2. Kinematics
        # grad_u[e,g,i,j] = sum_a u[e,a,i] * DN[e,g,a,j]
        np.matmul(np.swapaxes(self._u_node[:, None, :, :], 2, 3), self.DN, out=self._grad_u)
        self._F[:] = self._grad_u
        self._F[..., 0, 0] += 1.0
        self._F[..., 1, 1] += 1.0

        # Green-Lagrange: E = 0.5*(F^T F - I)
        np.matmul(np.swapaxes(self._F, 2, 3), self._F, out=self._FtF)
        self._E_voigt[..., 0] = 0.5 * (self._FtF[..., 0, 0] - 1.0)
        self._E_voigt[..., 1] = 0.5 * (self._FtF[..., 1, 1] - 1.0)
        self._E_voigt[..., 2] = self._FtF[..., 0, 1]  # 2*E12
        E_flat = self._E_voigt.reshape(-1, 3)

        # 3. Constitutive law
        if self._uniform_material:
            S_flat, CC_flat = _neo_hookean_pk2_2d_vectorized(
                E_flat, self._young_scalar, self._poisson_scalar)
        else:
            # Per-element material (rare but supported)
            young_gp = np.repeat(self.young, ng)
            poisson_gp = np.repeat(self.poisson, ng)
            S_flat = np.empty_like(E_flat)
            CC_flat = np.empty((E_flat.shape[0], 3, 3))
            # Group by unique (E, nu) pairs for efficiency
            for (y, p) in set(zip(self.young, self.poisson)):
                mask = (young_gp == y) & (poisson_gp == p)
                S_flat[mask], CC_flat[mask] = _neo_hookean_pk2_2d_vectorized(
                    E_flat[mask], y, p)

        self._S_voigt[:] = S_flat.reshape(ne, ng, 3)
        self._CC[:] = CC_flat.reshape(ne, ng, 3, 3)

        # S tensor (ne, ng, 2, 2)
        self._St[..., 0, 0] = self._S_voigt[..., 0]
        self._St[..., 1, 1] = self._S_voigt[..., 1]
        self._St[..., 0, 1] = self._S_voigt[..., 2]
        self._St[..., 1, 0] = self._S_voigt[..., 2]

        # 4. Internal forces: f_int[e,a,i] = sum_g F_ij S_jk DN_ak * w
        np.matmul(self._F, self._St, out=self._P)
        np.matmul(self.DN, np.swapaxes(self._P, 2, 3), out=self._tmp_ag_ai)
        np.multiply(self._tmp_ag_ai, self.w_detJ[..., None, None], out=self._tmp_ag_ai)
        np.sum(self._tmp_ag_ai, axis=1, out=self._f_int)

        # 5. Tangent stiffness
        # B-matrix: B[e,g,v,(a*2+i)] for Voigt v, node a, dof i
        B = self._B
        B.fill(0.0)
        F00 = self._F[..., 0, 0]
        F01 = self._F[..., 0, 1]
        F10 = self._F[..., 1, 0]
        F11 = self._F[..., 1, 1]
        for a in range(nn):
            c0 = 2 * a
            c1 = c0 + 1
            dn0 = self.DN[..., a, 0]
            dn1 = self.DN[..., a, 1]
            B[..., 0, c0] = F00 * dn0
            B[..., 0, c1] = F10 * dn0
            B[..., 1, c0] = F01 * dn1
            B[..., 1, c1] = F11 * dn1
            B[..., 2, c0] = F00 * dn1 + F01 * dn0
            B[..., 2, c1] = F10 * dn1 + F11 * dn0

        # K_mat = sum_g B^T C B * w_detJ
        np.matmul(self._CC, B, out=self._CB)
        np.matmul(np.swapaxes(B, 2, 3), self._CB, out=self._K_gp)
        np.multiply(self._K_gp, self.w_detJ[..., None, None], out=self._K_gp)
        np.sum(self._K_gp, axis=1, out=self._K_mat)

        # K_geo: K_geo[(a,i),(b,i)] += sum_g,k,l S_kl DN_ak DN_bl * w_detJ
        np.matmul(self.DN, self._St, out=self._DNSt)
        np.matmul(self._DNSt, np.swapaxes(self.DN, 2, 3), out=self._S_DN_DN_gp)
        np.multiply(self._S_DN_DN_gp, self.w_detJ[..., None, None], out=self._S_DN_DN_gp)
        np.sum(self._S_DN_DN_gp, axis=1, out=self._S_DN_DN)

        self._K_geo.fill(0.0)
        self._K_geo[:, 0::2, 0::2] = self._S_DN_DN
        self._K_geo[:, 1::2, 1::2] = self._S_DN_DN

        np.add(self._K_mat, self._K_geo, out=self._K_total)

        # 6. Global assembly
        self._rhs.fill(0.0)
        np.add.at(self._rhs, self.rows_R, -self._f_int.reshape(ne, -1).ravel())

        K = coo_matrix((self._K_total.ravel(), (self.rows_K, self.cols_K)),
                        shape=(self.n_dof, self.n_dof)).tocsr()

        return K, self._rhs

    def CalculateHomogenizedStressAndStrainFromLastAssembly(self):
        """Compute homogenized response from the most recent Assemble() state."""
        volume = float(np.sum(self.w_detJ))
        if volume <= 0.0:
            return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
        w = self.w_detJ[..., None]
        eps_h = np.sum(self._E_voigt * w, axis=(0, 1)) / volume
        sig_h = np.sum(self._S_voigt * w, axis=(0, 1)) / volume
        return eps_h, sig_h



USE_OLD_STIFFNESS_IN_FIRST_ITERATION = True
NEWTON_TOL_REL = 1e-6
NEWTON_TOL_ABS = 1e-6
DISP_TOL_REL = 1e-8
DISP_TOL_ABS = 1e-8
CONVERGENCE_CRITERION = "displacement"  # "residual", "displacement", "both"
# Keep Kratos reference post-processing by default to preserve bitwise consistency
# with the baseline FOM solver. Set to True to trade a small accuracy drift for speed.
USE_VECTORIZED_HOMOGENIZATION = False

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
            # Accept arbitrary whitespace (spaces/tabs) after "Begin" and "SubModelPart".
            parts = s.split()
            if len(parts) >= 3 and parts[0] == "Begin" and parts[1] == "SubModelPart":
                names.append(parts[2])
    return names


def DetectMdpaSections(mdpa_path):
    """
    Detect whether an mdpa contains top-level Geometries and/or Elements sections.
    """
    has_geometries = False
    has_elements = False
    if not os.path.exists(mdpa_path):
        return has_geometries, has_elements

    with open(mdpa_path, "r") as f:
        for line in f:
            s = line.strip()
            parts = s.split()
            if len(parts) >= 2 and parts[0] == "Begin":
                if parts[1] == "Geometries":
                    has_geometries = True
                elif parts[1] == "Elements":
                    has_elements = True
            if has_geometries and has_elements:
                break
    return has_geometries, has_elements


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


def SetMaterialsFilename(parameters, materials_filename="StructuralMaterials.json"):
    parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(
        str(materials_filename)
    )
    return parameters


def setup_kratos_parameters(mesh="rve_geometry"):
    """
    High-level ProjectParameters setup with auto-detected material parts.
    """
    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())
    
    SetInputMeshFilename(parameters, mesh)
    mdpa_path = f"{StripMdpaExtension(mesh)}.mdpa"
    has_geometries, has_elements = DetectMdpaSections(mdpa_path)

    if os.path.exists(mdpa_path):
        material_parts = DetectMaterialSubModelParts(mdpa_path)
        parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
        SetMaterialsFilename(parameters, "StructuralMaterials.json")

        # If the mesh already contains Elements (typical reduced/HROM mesh),
        # disable CreateEntitiesFromGeometriesModeler to avoid deleting them.
        if has_elements and not has_geometries:
            import json
            params_dict = json.loads(parameters.PrettyPrintJsonString())
            modelers = params_dict.get("modelers", [])
            filtered = [
                m for m in modelers
                if "CreateEntitiesFromGeometriesModeler" not in str(m.get("name", ""))
            ]
            params_dict["modelers"] = filtered
            parameters = KM.Parameters(json.dumps(params_dict))
    
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

    # Initialize vectorized assembler (one-time cost)
    assembler = VectorizedAssembler(mp, n_dof, eq_map)

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
            K, rhs = assembler.Assemble(u)
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
        _, _ = assembler.Assemble(u)
        FinalizeNonLinearIteration(entities, mp.ProcessInfo)
        u_n = u.copy()
        if USE_VECTORIZED_HOMOGENIZATION:
            eps_h, sig_h = assembler.CalculateHomogenizedStressAndStrainFromLastAssembly()
        else:
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
    p.add_argument(
        "--young-mpa",
        type=float,
        default=None,
        help="Deprecated: runtime material generation is disabled; StructuralMaterials.json is used.",
    )
    p.add_argument(
        "--poisson",
        type=float,
        default=None,
        help="Deprecated: runtime material generation is disabled; StructuralMaterials.json is used.",
    )
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
    if args.young_mpa is not None or args.poisson is not None:
        print(
            "[INFO] --young-mpa / --poisson provided, but runtime material generation is disabled. "
            "Using StructuralMaterials.json."
        )

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
        SetMaterialsFilename(parameters, "StructuralMaterials.json")
        print(f"[INFO] Auto-detected material parts: {material_parts}")
        print("[INFO] Using materials file: StructuralMaterials.json")
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
