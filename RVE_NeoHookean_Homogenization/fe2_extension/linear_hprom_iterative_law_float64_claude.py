#!/usr/bin/env python3
"""Cook-compatible material law for the LINEAR HPROM (pure POD, no ANN):
solves a hyper-reduced Newton iteration using a FIXED classic-ECM Z_res
rule (linear_residual_z_res_rule_claude.npz, 237 elements, this session's
own build), and computes the homogenized stress ONLINE via the SAME
reaction-force formula already used natively by DHpromAnnDirectLawFloat64/
HpromAnnIterativeLawFloat64 -- NOT a post-hoc correction -- using the
existing 37-point classic-ECM reaction-force rule
(classic_ecm_reaction_force_result_claude.npz) via the same
regressor_type="fixed_classic" path already validated for those classes.

Structurally a stripped-down HpromAnnIterativeLawFloat64: no ANN model, no
phi_p/phi_s split (pure phi_f), no manifold-curvature Hessian, no MAW-
adaptive weight regression for Z_res (fixed weights -> zero Jacobian by
construction, simpler than the ANN case's dynamic weight machinery). Reuses
reaction_force_hom_sig_and_jacobian, maw_hom_weight_and_jacobian_single_
model, and _build_maw_hom_target_model/_evaluate_maw_hom_weights_current
unmodified.
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
    VectorizedAssembler,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    DeformationGradientFromGreenLagrange2D,
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    NEWTON_TOL_ABS,
    USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
)
from hprom_solver_rve import (  # noqa: E402
    GetReferenceIntegrationMeasureFromMesh,
    ResolveActiveFreeDofsAndBasisRows,
)
from _material_law_guard_claude import true_neo_hookean_active  # noqa: E402
from hprom_ann_solver_rve import (  # noqa: E402
    _build_full_to_local_map,
    _build_maw_hom_target_model,
    _evaluate_maw_hom_weights_current,
)
from deformation_gradient_jacobian_claude import deformation_gradient_and_jacobian_2d  # noqa: E402
from maw_hom_weight_jacobian_claude import maw_hom_weight_and_jacobian_single_model  # noqa: E402
from reaction_force_hom_tangent_claude import reaction_force_hom_sig_and_jacobian  # noqa: E402

DEFAULT_BASIS_DIR = str(ROOT / "pod" / "stage_2_pod_rve")
DEFAULT_LAW_MESH_DIR = str(HERE / "linear_hprom_law_mesh")


class LinearHpromIterativeLawFloat64:
    """Pure-POD (no ANN) hyper-reduced iterative RVE law -- Newton solve on
    phi_f alone with a fixed, classic-ECM Z_res rule; online reaction-force
    stress via a separate fixed classic-ECM Z_sig rule."""

    def __init__(
        self,
        basis_dir=DEFAULT_BASIS_DIR,
        law_mesh_dir=DEFAULT_LAW_MESH_DIR,
        max_its=25,
        abs_res_cutoff=NEWTON_TOL_ABS,
        dq_abs_cutoff=1.0e-6,
        regularization=1.0e-10,
        use_old_stiffness_in_first_iteration=USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
        max_dq_norm=0.5,
        verbose=False,
    ):
        self.max_its = int(max_its)
        self.abs_res_cutoff = float(abs_res_cutoff)
        self.dq_abs_cutoff = float(dq_abs_cutoff)
        self.regularization = float(regularization)
        self.use_old_stiffness_in_first_iteration = bool(use_old_stiffness_in_first_iteration)
        self.max_dq_norm = float(max_dq_norm)
        self.verbose = bool(verbose)

        basis_dir = Path(basis_dir)
        phi_f_ref = np.load(basis_dir / "pod_basis_free.npy")
        free_dofs_ref = np.load(basis_dir / "free_dofs.npy").astype(np.int64)
        eq_map_ref = np.load(basis_dir / "eq_map.npy").astype(np.int64)
        Xc, Yc = np.load(basis_dir / "domain_center.npy")

        ecm = np.load(Path(law_mesh_dir) / "ecm_weights_all.npz", allow_pickle=True)
        ecm_data = {k: ecm[k] for k in ecm.files}

        full_mesh_base = str(np.ravel(ecm_data["hrom_full_mesh_base"])[0])
        hrom_mesh_base = str(np.ravel(ecm_data["hrom_mesh_base"])[0])

        parameters = setup_kratos_parameters(hrom_mesh_base)
        model_kratos = KM.Model()
        sim = RVEHomogenizationDatasetGenerator(model_kratos, parameters)
        sim.Initialize()
        mp = sim._GetSolver().GetComputingModelPart()
        n_total_dof, eq_id_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
        self.mp, self.eq_id_map, self.ta, self.n_total_dof = mp, eq_id_map, ta, n_total_dof
        self.vec_assembler = VectorizedAssembler(
            mp, n_total_dof, eq_id_map, log_label="LinearHpromIterativeLawFloat64"
        )
        self.elements = list(mp.Elements)
        self.entities = list(mp.Elements) + list(mp.Conditions)
        n_elem_reference = int(np.ravel(ecm_data["n_elem"])[0])
        self.n_elem_reference = n_elem_reference
        self.n_current_elements = len(self.elements)

        free_dofs, dir_dofs, basis_rows = ResolveActiveFreeDofsAndBasisRows(
            mp, n_total_dof, eq_id_map,
            free_dofs_reference=free_dofs_ref, eq_map_reference=eq_map_ref,
            full_mesh_base=full_mesh_base, solver_label="LinearHpromIterativeLawFloat64",
        )
        self.free_dofs = np.asarray(free_dofs, dtype=np.int64)
        self.dir_dofs = np.asarray(dir_dofs, dtype=np.int64)
        self.dof_to_dirpos_local = -np.ones(n_total_dof, dtype=np.int64)
        self.dof_to_dirpos_local[self.dir_dofs] = np.arange(self.dir_dofs.size, dtype=np.int64)
        self.thickness_scalar = float(np.asarray(self.vec_assembler.thickness, dtype=float).reshape(-1)[0])
        self.phi_f = np.asarray(phi_f_ref, dtype=float)[basis_rows, :]
        self.n_primary = self.phi_f.shape[1]

        full_to_local_res = _build_full_to_local_map(
            ecm_data, n_elem_reference=n_elem_reference, n_current_elements=len(self.elements),
        )
        Z_res_full = np.asarray(ecm_data["Z_res"], dtype=np.int64).reshape(-1)
        w_res_full_arr = np.asarray(ecm_data["w_res_full"], dtype=float).reshape(-1)
        z_res_local = np.array(
            [full_to_local_res[int(i)] for i in Z_res_full if int(i) in full_to_local_res], dtype=np.int64
        )
        w_res_local = np.array(
            [w_res_full_arr[int(i)] for i in Z_res_full if int(i) in full_to_local_res], dtype=float
        )
        if z_res_local.size != Z_res_full.size:
            print(f"[LinearHpromIterativeLawFloat64] WARNING: "
                  f"{Z_res_full.size - z_res_local.size} Z_res elements missing from the current mesh.")
        selected_elements = [self.elements[int(i)] for i in z_res_local]
        self.res_assembler = VectorizedAssembler(
            mp, n_total_dof, eq_id_map, elements=selected_elements, element_scales=w_res_local,
            log_label="LinearHpromResAssembler",
        )

        self.maw_sig_hom = _build_maw_hom_target_model(ecm_data, "sig")
        self.full_to_local_hom = _build_full_to_local_map(
            ecm_data, n_elem_reference=n_elem_reference, n_current_elements=len(self.elements),
        )
        self.hom_reference_measure = GetReferenceIntegrationMeasureFromMesh(full_mesh_base)

        if Xc is None or Yc is None:
            sim._InitializeDomainCenterIfNeeded(mp)
            x0c, y0c = float(sim._x0c), float(sim._y0c)
        else:
            x0c, y0c = float(Xc), float(Yc)
        dof_x = np.zeros(n_total_dof, dtype=float)
        dof_y = np.zeros(n_total_dof, dtype=float)
        is_x_dof = np.zeros(n_total_dof, dtype=bool)
        for i, node in enumerate(mp.Nodes):
            xr, yr = float(node.X0) - x0c, float(node.Y0) - y0c
            idx_x, idx_y = int(eq_id_map[i, 0]), int(eq_id_map[i, 1])
            if 0 <= idx_x < n_total_dof:
                dof_x[idx_x], dof_y[idx_x], is_x_dof[idx_x] = xr, yr, True
            if 0 <= idx_y < n_total_dof:
                dof_x[idx_y], dof_y[idx_y], is_x_dof[idx_y] = xr, yr, False
        self.x_free, self.y_free, self.is_x_free = dof_x[self.free_dofs], dof_y[self.free_dofs], is_x_dof[self.free_dofs]
        self.x_dir, self.y_dir, self.is_x_dir = dof_x[self.dir_dofs], dof_y[self.dir_dofs], is_x_dof[self.dir_dofs]

        self._phi_full = np.zeros((n_total_dof, self.n_primary), dtype=float)
        self._phi_full[self.free_dofs, :] = self.phi_f
        self._phi_full_T = self._phi_full.T

    @staticmethod
    def _affine(e_vec, x_loc, y_loc, is_x_loc):
        F = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (F[0, 0] - 1.0) * x_loc + F[0, 1] * y_loc
        uy = F[1, 0] * x_loc + (F[1, 1] - 1.0) * y_loc
        return np.where(is_x_loc, ux, uy)

    def _dirichlet_sensitivity(self, e_vec, heps=1.0e-6):
        e_vec = np.asarray(e_vec, dtype=float).reshape(3)
        sens = np.empty((3, self.dir_dofs.size), dtype=float)
        for k in range(3):
            step = heps if abs(e_vec[k]) < 1.0 else heps * max(1.0, abs(e_vec[k]))
            e_plus, e_minus = e_vec.copy(), e_vec.copy()
            e_plus[k] += step
            e_minus[k] -= step
            u_plus = self._affine(e_plus, self.x_dir, self.y_dir, self.is_x_dir)
            u_minus = self._affine(e_minus, self.x_dir, self.y_dir, self.is_x_dir)
            sens[k] = (u_plus - u_minus) / (2.0 * step)
        return sens

    def _reaction_force_c_e(self, E):
        assembler = self.vec_assembler
        local_dirpos = self.dof_to_dirpos_local[assembler.local_eq_ids]
        valid = local_dirpos >= 0
        f_int_flat = assembler._f_int.reshape(assembler.n_elems, -1)
        sens = self._dirichlet_sensitivity(E)
        denom = self.thickness_scalar * float(self.hom_reference_measure)

        c_e = np.zeros((assembler.n_elems, 3), dtype=float)
        for k in range(3):
            sk = np.zeros_like(f_int_flat)
            sk[valid] = sens[k, local_dirpos[valid]]
            c_e[:, k] = np.sum(sk * f_int_flat, axis=1) / denom
        return c_e

    def _reaction_force_hom_sig(self, E, w_sig):
        c_e = self._reaction_force_c_e(E)
        return c_e.T @ np.asarray(w_sig, dtype=float).reshape(-1)

    @staticmethod
    def _solve_reduced_system(K_sys, rhs, regularization):
        try:
            dq = np.linalg.solve(K_sys, rhs)
        except np.linalg.LinAlgError:
            dq, *_ = np.linalg.lstsq(K_sys, rhs, rcond=None)
        if np.all(np.isfinite(dq)):
            return dq
        K_reg = K_sys + regularization * np.eye(K_sys.shape[0], dtype=K_sys.dtype)
        try:
            dq = np.linalg.solve(K_reg, rhs)
        except np.linalg.LinAlgError:
            dq, *_ = np.linalg.lstsq(K_reg, rhs, rcond=None)
        return dq

    def evaluate(self, E, q_prev=None, step_index=1):
        with true_neo_hookean_active():
            return self._evaluate_impl(E, q_prev=q_prev, step_index=step_index)

    def _evaluate_impl(self, E, q_prev=None, step_index=1, _cache_out=None):
        E = np.asarray(E, dtype=float).reshape(-1)
        q_p = np.zeros(self.n_primary, dtype=float) if q_prev is None else np.asarray(q_prev, dtype=float).copy()

        u_aff_free = self._affine(E, self.x_free, self.y_free, self.is_x_free)
        disp_base = np.zeros(self.n_total_dof, dtype=float)
        disp_base[self.dir_dofs] = self._affine(E, self.x_dir, self.y_dir, self.is_x_dir)

        phi_full, phi_full_T = self._phi_full, self._phi_full_T

        converged = False
        it = 0
        for it in range(self.max_its):
            u_free = u_aff_free + self.phi_f @ q_p
            u_eq_curr = disp_base.copy()
            u_eq_curr[self.free_dofs] = u_free

            InitializeNonLinearIteration(self.entities, self.mp.ProcessInfo)
            K_hp, rhs_hp = self.res_assembler.Assemble(u_eq_curr)
            FinalizeNonLinearIteration(self.entities, self.mp.ProcessInfo)

            K_r = phi_full_T @ (K_hp @ phi_full)
            r_r = phi_full_T @ rhs_hp

            res_norm = float(np.linalg.norm(r_r))
            if self.verbose:
                print(f"    it {it:02d}: ||r_r||={res_norm:.3e}")
            if it > 0 and res_norm < self.abs_res_cutoff:
                converged = True
                # q_p is UNCHANGED this iteration (this break fires before
                # the update below), so K_r/_K_total were just computed
                # FROM the q_p being returned -- safe to hand back to
                # evaluate_with_tangent instead of recomputing via a
                # second res_assembler.Assemble call. NOT valid for the
                # OTHER break below (dq_norm < dq_abs_cutoff), which fires
                # AFTER q_p has already been updated -- K_r there would be
                # stale relative to the returned q_p.
                if _cache_out is not None:
                    _cache_out.update(
                        q_p=q_p.copy(), K_r=K_r,
                        K_total_support=self.res_assembler._K_total.copy(),
                    )
                break

            dq_p = self._solve_reduced_system(K_r, r_r, self.regularization)
            dq_norm = float(np.linalg.norm(dq_p))
            if dq_norm > self.max_dq_norm and dq_norm > 0.0:
                dq_p *= self.max_dq_norm / dq_norm
                dq_norm = float(np.linalg.norm(dq_p))
            q_p = q_p + dq_p
            if dq_norm < self.dq_abs_cutoff:
                converged = True
                break

        u_free = u_aff_free + self.phi_f @ q_p
        u_eq_curr = disp_base.copy()
        u_eq_curr[self.free_dofs] = u_free
        SetDisplacementFromEquationVector(u_eq_curr, self.eq_id_map, self.ta)
        UpdateCurrentCoordinatesFromDisplacement(self.mp, step=0)
        self.vec_assembler.ComputeLocalArrays(u_eq_curr)

        w_sig = _evaluate_maw_hom_weights_current(
            q_p, E, self.maw_sig_hom, n_elem_reference=self.n_elem_reference,
            n_current_elements=self.n_current_elements, full_to_local=self.full_to_local_hom,
        )
        hom_sig = self._reaction_force_hom_sig(E, w_sig)
        hom_eps = np.zeros(3, dtype=float)  # discarded by every caller in this project
        return hom_eps, hom_sig, q_p, it, converged

    def evaluate_with_tangent(self, E, q_prev=None, step_index=1):
        with true_neo_hookean_active():
            E = np.asarray(E, dtype=float).reshape(-1)
            _cache = {}
            hom_eps, hom_sig, q_p, n_iters, converged = self._evaluate_impl(
                E, q_prev=q_prev, step_index=step_index, _cache_out=_cache,
            )

            u_aff_free = self._affine(E, self.x_free, self.y_free, self.is_x_free)
            disp_base = np.zeros(self.n_total_dof, dtype=float)
            disp_base[self.dir_dofs] = self._affine(E, self.x_dir, self.y_dir, self.is_x_dir)
            u_free = u_aff_free + self.phi_f @ q_p
            u_eq_curr = disp_base.copy()
            u_eq_curr[self.free_dofs] = u_free

            phi_full, phi_full_T = self._phi_full, self._phi_full_T

            # _evaluate_impl's own last Newton iteration already computed
            # this exact linearization (at this exact q_p) right before
            # declaring convergence -- reuse it instead of recomputing via
            # a second res_assembler.Assemble call. Falls back to the
            # original recompute whenever the cache wasn't populated (the
            # OTHER break condition fired) or (defensively) doesn't match
            # the returned q_p.
            if _cache and np.array_equal(_cache["q_p"], q_p):
                K_r = _cache["K_r"]
                K_total_support = _cache["K_total_support"]
            else:
                InitializeNonLinearIteration(self.entities, self.mp.ProcessInfo)
                K_hp, _rhs_hp = self.res_assembler.Assemble(u_eq_curr)
                FinalizeNonLinearIteration(self.entities, self.mp.ProcessInfo)
                K_r = phi_full_T @ (K_hp @ phi_full)
                K_total_support = None  # read live below, just refreshed by the Assemble call above

            _F, dF_dE = deformation_gradient_and_jacobian_2d(E)
            du_aff_dE = np.zeros((self.n_total_dof, 3), dtype=float)
            for k in range(3):
                d00, d01, d10, d11 = dF_dE[0, 0, k], dF_dE[0, 1, k], dF_dE[1, 0, k], dF_dE[1, 1, k]
                du_aff_dE[self.free_dofs, k] = np.where(
                    self.is_x_free, d00 * self.x_free + d01 * self.y_free, d10 * self.x_free + d11 * self.y_free,
                )
                du_aff_dE[self.dir_dofs, k] = np.where(
                    self.is_x_dir, d00 * self.x_dir + d01 * self.y_dir, d10 * self.x_dir + d11 * self.y_dir,
                )

            support_assembler = self.res_assembler
            local_eq_ids_support = support_assembler.local_eq_ids
            du_local_dE_support = du_aff_dE[local_eq_ids_support, :]
            if K_total_support is None:
                K_total_support = support_assembler._K_total
            d_neg_fint_dE_support = -np.einsum("eij,ejk->eik", K_total_support, du_local_dE_support)

            dr_full_dE = np.zeros((self.n_total_dof, 3), dtype=float)
            for k in range(3):
                np.add.at(dr_full_dE[:, k], support_assembler.rows_R, d_neg_fint_dE_support[:, :, k].reshape(-1))

            dG_dE = phi_full_T @ dr_full_dE
            dq_p_dE = self._solve_reduced_system(K_r, dG_dE, self.regularization) if dG_dE.ndim == 1 else np.linalg.solve(K_r, dG_dE)

            # NOT self.vec_assembler.ComputeLocalArrays(u_eq_curr) again
            # here: _evaluate_impl's own tail already called it with this
            # EXACT SAME u_eq_curr (same q_p, same formula) right before
            # returning -- unconditionally redundant regardless of which
            # break condition fired, unlike the K_r cache above. Nothing
            # in between touches self.vec_assembler's state (res_assembler
            # is a separate object), so its state is still exactly what
            # that call left it in.

        w_sig, dw_sig_dE = maw_hom_weight_and_jacobian_single_model(
            E, self.maw_sig_hom, n_elem_reference=self.n_elem_reference,
            n_current_elements=self.n_current_elements, full_to_local=self.full_to_local_hom,
        )
        hom_sig_check = self._reaction_force_hom_sig(E, w_sig)

        ddisp_dE = np.zeros((self.n_total_dof, 3), dtype=float)
        _F, dF_dE = deformation_gradient_and_jacobian_2d(E)
        for k in range(3):
            d00, d01, d10, d11 = dF_dE[0, 0, k], dF_dE[0, 1, k], dF_dE[1, 0, k], dF_dE[1, 1, k]
            ddisp_dE[self.free_dofs, k] = np.where(
                self.is_x_free, d00 * self.x_free + d01 * self.y_free, d10 * self.x_free + d11 * self.y_free,
            )
            ddisp_dE[self.dir_dofs, k] = np.where(
                self.is_x_dir, d00 * self.x_dir + d01 * self.y_dir, d10 * self.x_dir + d11 * self.y_dir,
            )
        ddisp_dE[self.free_dofs, :] += self.phi_f @ dq_p_dE

        local_eq_ids = self.vec_assembler.local_eq_ids
        du_local_dE = ddisp_dE[local_eq_ids, :]
        B = self.vec_assembler._B
        CC = self.vec_assembler._CC
        dE_gp_dE = np.einsum("egvc,ecx->egvx", B, du_local_dE)
        dS_gp_dE = np.einsum("egvw,egwx->egvx", CC, dE_gp_dE)

        _hom_sig_at_u_eq_curr, dSig_hom_dE = reaction_force_hom_sig_and_jacobian(
            self, E, w_sig, dw_sig_dE, du_local_dE, dS_gp_dE,
        )

        hom_eps_check = np.zeros(3, dtype=float)
        dEps_hom_dE = np.zeros((3, 3), dtype=float)
        return hom_eps, hom_sig, q_p, n_iters, converged, dEps_hom_dE, dSig_hom_dE, hom_eps_check, hom_sig_check


_DEFAULT_LAW = None


def get_law(**kwargs):
    global _DEFAULT_LAW
    if _DEFAULT_LAW is None:
        _DEFAULT_LAW = LinearHpromIterativeLawFloat64(**kwargs)
    return _DEFAULT_LAW


if __name__ == "__main__":
    law = LinearHpromIterativeLawFloat64(verbose=True)
    eps0, sig0, q0, n_it, conv = law.evaluate(np.array([0.05, 0.0, 0.0]))
    print(f"[linear_hprom_iterative_law] evaluate([0.05,0,0]) -> sig={sig0}, iters={n_it}, converged={conv}")
