#!/usr/bin/env python3
"""Precision-floor experiment, Stage 2: float64 copy of
HpromAnnIterativeLaw (hprom_ann_iterative_law_claude.py), which this file
leaves completely untouched -- same rationale and same discipline as
dhprom_ann_direct_law_float64_claude.py: a full, independent copy (not a
subclass, to avoid ever running the original's float32 setup), casting
the decoder network to double BEFORE its first forward pass, and every
np.float32 cast in the original changed to np.float64.

Every float32 site in the original: self.Vs (__init__), and the three
decoder-forward-pass helpers (_eval_qs_only, _eval_qs_and_jac,
_compute_weighted_decoder_hessian). Everything else in the original is
already plain float64 numpy and needs no change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import KratosMultiphysics as KM

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("core", "prom/ann", "hprom/ann"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from fom_solver_rve import (
    setup_kratos_parameters,
    RVEHomogenizationDatasetGenerator,
    VectorizedAssembler,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    DeformationGradientFromGreenLagrange2D,
    CalculateHomogenizedFromAssemblerWithElementWeights,
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    NEWTON_TOL_ABS,
    USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
)
from hprom_solver_rve import (
    GetReferenceIntegrationMeasureFromMesh,
    ResolveActiveFreeDofsAndBasisRows,
)
from _material_law_guard_claude import true_neo_hookean_active
from hprom_ann_solver_rve import (
    LoadHpromAnnModel,
    _build_full_to_local_map,
    _has_maw_hom_component_models,
    _build_maw_hom_component_models,
    _build_maw_hom_target_model,
    _evaluate_maw_hom_weights_current,
    _build_maw_res_target_model,
    _map_support_to_current,
    _DynamicWeightedResidualAssembler,
    _build_free_dof_index_map,
    _prepare_unit_rhs_scatter,
    _evaluate_maw_res_support_weights_and_jacobian,
    _scatter_rhs_loc_by_element_free,
    WEIGHT_ZERO_TOL,
)
from deformation_gradient_jacobian_claude import deformation_gradient_and_jacobian_2d
from maw_hom_weight_jacobian_claude import maw_hom_weight_and_jacobian_single_model
from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw

DEFAULT_ANN_DATA_DIR = str(ROOT / "prom" / "ann" / "stage_7_ann_model_ls")
DEFAULT_HPROM_ANN_DIR = str(ROOT / "hprom" / "ann" / "maw_dynamic")
DEFAULT_BASIS_DIR = str(ROOT / "pod" / "stage_2_pod_rve")


class HpromAnnIterativeLawFloat64:
    """Line-for-line float64 copy of HpromAnnIterativeLaw. See module docstring."""

    def __init__(
        self,
        ann_data_dir=DEFAULT_ANN_DATA_DIR,
        hprom_ann_dir=DEFAULT_HPROM_ANN_DIR,
        basis_dir=DEFAULT_BASIS_DIR,
        homogenization_mode="maw_dynamic",
        maw_hom_eval_mode="model",
        max_its=25,
        abs_res_cutoff=NEWTON_TOL_ABS,
        dq_abs_cutoff=1.0e-6,
        normalized_dq_cutoff=1.0e-4,
        relnorm_cutoff=1.0e-5,
        max_res_for_rel_convergence=1.0e-1,
        min_rel_drop_stop=1.0e-2,
        stagnation_relnorm_gate=1.0e-4,
        max_dq_norm=0.5,
        old_stiffness_residual_cutoff=1.0e5,
        regularization=1.0e-10,
        use_old_stiffness_in_first_iteration=USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
        include_manifold_curvature=True,
        qp_init_mode="continuation",
        fail_on_nonconvergence=False,
        verbose=False,
    ):
        self.max_its = int(max_its)
        self.abs_res_cutoff = float(abs_res_cutoff)
        self.dq_abs_cutoff = float(dq_abs_cutoff)
        self.normalized_dq_cutoff = float(normalized_dq_cutoff)
        self.relnorm_cutoff = float(relnorm_cutoff)
        self.max_res_for_rel_convergence = float(max_res_for_rel_convergence)
        self.min_rel_drop_stop = float(min_rel_drop_stop)
        self.stagnation_relnorm_gate = float(stagnation_relnorm_gate)
        self.max_dq_norm = float(max_dq_norm)
        self.old_stiffness_residual_cutoff = float(old_stiffness_residual_cutoff)
        self.regularization = float(regularization)
        self.use_old_stiffness_in_first_iteration = bool(use_old_stiffness_in_first_iteration)
        self.include_manifold_curvature = bool(include_manifold_curvature)
        self.qp_init_mode = str(qp_init_mode).strip().lower()
        self.fail_on_nonconvergence = bool(fail_on_nonconvergence)
        self.verbose = bool(verbose)

        (
            phi_p_ref, phi_s_ref, free_dofs_ref, _dir_dofs_ref, eq_map_ref,
            Xc, Yc, ann_model, device, ecm_data, _include_macro,
        ) = LoadHpromAnnModel(basis_dir=basis_dir, ann_data_dir=ann_data_dir, hprom_ann_dir=hprom_ann_dir)
        # Force CPU + cap torch's own thread pool at 1: same verified fix
        # as dhprom_ann_direct_law_float64_claude.py, for the identical
        # reason -- thousands of independent batch-size-1 calls through a
        # tiny MLP, where GPU dispatch and multi-thread overhead dominate
        # the (tiny) actual compute. See that file's own comments for the
        # measured, verified numbers.
        device = torch.device("cpu")
        ann_model = ann_model.to(device)
        ann_model = ann_model.double()  # <-- cast BEFORE any forward pass (this file's whole point)
        self.ann_model, self.device = ann_model, device
        torch.set_num_threads(1)

        full_mesh_base = str(np.ravel(ecm_data["hrom_full_mesh_base"])[0]) if "hrom_full_mesh_base" in ecm_data else "rve_geometry"
        hrom_mesh_base = str(np.ravel(ecm_data["hrom_mesh_base"])[0]) if "hrom_mesh_base" in ecm_data else full_mesh_base

        parameters = setup_kratos_parameters(hrom_mesh_base)
        model_kratos = KM.Model()
        sim = RVEHomogenizationDatasetGenerator(model_kratos, parameters)
        sim.Initialize()
        mp = sim._GetSolver().GetComputingModelPart()
        n_total_dof, eq_id_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
        self.mp, self.eq_id_map, self.ta, self.n_total_dof = mp, eq_id_map, ta, n_total_dof
        self.vec_assembler = VectorizedAssembler(mp, n_total_dof, eq_id_map, log_label="HpromAnnIterativeLawFloat64")
        self.elements = list(mp.Elements)
        self.entities = list(mp.Elements) + list(mp.Conditions)
        n_elem_reference = int(np.ravel(ecm_data["n_elem"])[0]) if "n_elem" in ecm_data else len(self.elements)
        self.n_elem_reference = n_elem_reference
        self.n_current_elements = len(self.elements)

        free_dofs, dir_dofs, basis_rows = ResolveActiveFreeDofsAndBasisRows(
            mp, n_total_dof, eq_id_map,
            free_dofs_reference=free_dofs_ref, eq_map_reference=eq_map_ref,
            full_mesh_base=full_mesh_base, solver_label="HpromAnnIterativeLawFloat64",
        )
        self.free_dofs = np.asarray(free_dofs, dtype=np.int64)
        self.dir_dofs = np.asarray(dir_dofs, dtype=np.int64)
        self.dof_to_dirpos_local = -np.ones(n_total_dof, dtype=np.int64)
        self.dof_to_dirpos_local[self.dir_dofs] = np.arange(self.dir_dofs.size, dtype=np.int64)
        self.thickness_scalar = float(np.asarray(self.vec_assembler.thickness, dtype=float).reshape(-1)[0])
        phi_p = np.asarray(phi_p_ref, dtype=float)[basis_rows, :]
        phi_s = np.asarray(phi_s_ref, dtype=float)[basis_rows, :]
        self.phi_p, self.phi_s = phi_p, phi_s
        self.n_primary, self.n_secondary = phi_p.shape[1], phi_s.shape[1]

        maw_res = _build_maw_res_target_model(ecm_data, required=True)
        full_to_local_res = _build_full_to_local_map(
            ecm_data, n_elem_reference=n_elem_reference, n_current_elements=len(self.elements),
        )
        Z_res, z_res_pos, miss_res = _map_support_to_current(maw_res["z_support_full"], full_to_local_res)
        if miss_res.size:
            print(f"[HpromAnnIterativeLawFloat64] WARNING: residual MAW support lost {miss_res.size} full-mesh indices.")
        w_anchor = np.asarray(ecm_data["w_res"], dtype=float).reshape(-1) if "w_res" in ecm_data else None
        if w_anchor is not None and w_anchor.size == maw_res["z_support_full"].size:
            self.w_res_anchor_local = np.asarray(w_anchor[z_res_pos], dtype=float)
        else:
            self.w_res_anchor_local = np.ones(Z_res.size, dtype=float)
        self.maw_res, self.Z_res, self.z_res_pos = maw_res, Z_res, z_res_pos

        self.dyn_res_assembler = _DynamicWeightedResidualAssembler(
            mp=mp, n_dof=n_total_dof, eq_map=eq_id_map, elements=self.elements, selected_indices=Z_res,
        )
        self.dyn_res_assembler.prepare_reduced_action(self.free_dofs)
        self.unit_rhs_scatter = _prepare_unit_rhs_scatter(
            assembler_unit=self.dyn_res_assembler._assembler,
            free_dof_index_map=_build_free_dof_index_map(n_total_dof, self.free_dofs),
        )

        hom_mode = str(homogenization_mode).strip().lower()
        if hom_mode in ("maw", "maw_separate"):
            hom_mode = "maw_dynamic"
        if hom_mode != "maw_dynamic":
            raise NotImplementedError("This extraction currently only supports homogenization_mode='maw_dynamic'.")
        self.maw_hom_eval_mode = str(maw_hom_eval_mode or "model").strip().lower()
        self.maw_hom_componentwise = _has_maw_hom_component_models(ecm_data)
        if self.maw_hom_componentwise:
            self.maw_eps_hom = _build_maw_hom_component_models(ecm_data, "eps")
            self.maw_sig_hom = _build_maw_hom_component_models(ecm_data, "sig")
        else:
            self.maw_eps_hom = _build_maw_hom_target_model(ecm_data, "eps")
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

        self.Vs = torch.from_numpy(phi_s.astype(np.float64)).to(device)
        qp_aff = getattr(ann_model, "qp_init_mu_affine", None)
        if qp_aff is None:
            raise RuntimeError("qp_init_mode='continuation'/'mu_affine' requires qm_init_mu_affine.npz.")
        self.qp_aff = qp_aff

        q0_const, J0_const = self._eval_qs_and_jac(np.zeros(self.n_primary), np.zeros(3))
        self.q0_const_np, self.J0_const_np = q0_const, J0_const
        q_m_scale = ann_model.input_scaler.std.detach().cpu().numpy().reshape(-1).astype(float)
        self.q_m_scale = np.maximum(np.abs(q_m_scale), 1.0e-12)
        a_m = np.asarray(getattr(ann_model, "a_m_np", None), dtype=float)
        phi_master = phi_p @ a_m
        self.phi_p_eff_np = phi_master + phi_s @ J0_const
        self.w0_const_np = phi_s @ q0_const

    def _eval_qs_only(self, qp_vec):
        qp_t = torch.from_numpy(np.asarray(qp_vec, dtype=float).astype(np.float64)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_s_map = self.ann_model(qp_t)
        return q_s_map.detach().cpu().numpy().reshape(-1)

    def _eval_qs_and_jac(self, qp_vec, e_vec):
        qp_t = torch.from_numpy(np.asarray(qp_vec, dtype=float).astype(np.float64)).reshape(-1).to(self.device)
        qp_in = qp_t.clone().detach()

        def ann_from_qvec(qvec):
            return self.ann_model(qvec.view(1, -1)).reshape(-1)

        # Forward-mode AD (jacfwd), not reverse-mode: n_primary << n_secondary
        # here, so this needs n_primary forward passes instead of
        # n_secondary backward passes for the identical Jacobian -- this
        # call sits inside the inner Newton correction, so it runs more
        # often per query than D-HPROM-ANN's own single evaluation (see
        # dhprom_ann_direct_law_float64_claude.py for the verified,
        # end-to-end numerical-equivalence check of this exact swap).
        with torch.no_grad():
            q_s_map = ann_from_qvec(qp_in)
        jac = torch.func.jacfwd(ann_from_qvec)(qp_in).reshape(self.n_secondary, self.n_primary)
        return q_s_map.detach().cpu().numpy().reshape(-1), jac.detach().cpu().numpy()

    def _compute_weighted_decoder_hessian(self, qp_vec, output_weights):
        qp_t = torch.from_numpy(np.asarray(qp_vec, dtype=float).astype(np.float64)).to(self.device)
        weights = torch.from_numpy(np.asarray(output_weights, dtype=float).astype(np.float64)).to(self.device)
        with torch.enable_grad():
            qp_in = qp_t.clone().detach().requires_grad_(True)

            def weighted_ann_output(qvec):
                q_s_raw = self.ann_model(qvec.view(1, -1)).reshape(-1)
                return torch.dot(weights, q_s_raw)

            hessian = torch.autograd.functional.hessian(weighted_ann_output, qp_in, vectorize=True)
        return hessian.detach().cpu().numpy().reshape(self.n_primary, self.n_primary)

    def _eval_qs_and_jac_batch(self, qp_batch):
        """Batched form of _eval_qs_and_jac: one vmap(jacfwd(...)) call
        covering every row of qp_batch (n_active, n_primary) instead of
        n_active separate jacfwd calls -- verified this session (see
        _prototype_batched_decoder_claude.py / this file's own
        evaluate_with_tangent_batch verification) to reproduce the serial
        per-row result to ~1e-14 relative error, ~60x faster per row (the
        serial cost is dominated by per-call dispatch/tracing overhead on a
        tiny network, not actual FLOPs)."""
        qp_t = torch.from_numpy(np.asarray(qp_batch, dtype=np.float64)).to(self.device)

        def ann_from_qvec(qvec):
            return self.ann_model(qvec.view(1, -1)).reshape(-1)

        with torch.no_grad():
            q_s_map_batch = torch.func.vmap(ann_from_qvec)(qp_t)
        jac_batch = torch.func.vmap(torch.func.jacfwd(ann_from_qvec))(qp_t)
        return q_s_map_batch.detach().cpu().numpy(), jac_batch.detach().cpu().numpy()

    def _compute_weighted_decoder_hessian_batch(self, qp_batch, weights_batch):
        """Batched form of _compute_weighted_decoder_hessian: switches from
        torch.autograd.functional.hessian (one point at a time) to
        torch.func.hessian + torch.func.vmap, which compose -- verified this
        session on real captured (q_p, weights) pairs to ~1e-15 relative
        error, ~10x faster per row."""
        qp_t = torch.from_numpy(np.asarray(qp_batch, dtype=np.float64)).to(self.device)
        w_t = torch.from_numpy(np.asarray(weights_batch, dtype=np.float64)).to(self.device)

        def weighted_ann_output(qvec, w):
            q_s_raw = self.ann_model(qvec.view(1, -1)).reshape(-1)
            return torch.dot(w, q_s_raw)

        hessian_fn = torch.func.hessian(weighted_ann_output, argnums=0)
        with torch.enable_grad():
            H = torch.func.vmap(hessian_fn, in_dims=(0, 0))(qp_t, w_t)
        return H.detach().cpu().numpy()

    @staticmethod
    def _affine(e_vec, x_loc, y_loc, is_x_loc):
        F = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (F[0, 0] - 1.0) * x_loc + F[0, 1] * y_loc
        uy = F[1, 0] * x_loc + (F[1, 1] - 1.0) * y_loc
        return np.where(is_x_loc, ux, uy)

    def _dirichlet_sensitivity(self, e_vec, heps=1.0e-6):
        """d(u_dirichlet)/de_k, k=0,1,2, via central finite difference on the
        closed-form affine map -- same formula/step heuristic as
        pann/direct_energy/reaction_force_direct_stress.py's
        DirectStressGenerator.dirichlet_strain_sensitivity, but evaluated on
        THIS instance's own (possibly hyper-reduced) dir_dofs/x_dir/y_dir."""
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
        """Per-element reaction-force stress integrand c_e (n_current_elements,
        3) -- see DHpromAnnDirectLawFloat64's method of the same name for
        the full derivation/verification note. Requires vec_assembler._f_int
        to already be populated (ComputeLocalArrays, not
        ComputeStrainStressOnly) for the current displacement."""
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

    def _initial_qp_guess(self, e_vec, q_prev, step_index):
        if self.qp_init_mode in ("continuation", "previous") and (self.qp_init_mode != "continuation" or int(step_index) > 1):
            return np.asarray(q_prev, dtype=float).copy()
        if self.qp_init_mode == "zero":
            return np.zeros(self.n_primary, dtype=float)
        mu_dim = int(self.qp_aff["mu_dim"])
        mu = np.asarray(e_vec, dtype=float).reshape(-1)[:mu_dim]
        return np.concatenate([mu, [1.0]]) @ np.asarray(self.qp_aff["b_aff"], dtype=float)

    def _residual_weights_and_jacobian(self, qp_vec):
        w_support_all, dw_support_all = _evaluate_maw_res_support_weights_and_jacobian(qp_vec, self.maw_res)
        w_local = np.asarray(w_support_all[self.z_res_pos], dtype=float)
        dw_local = np.asarray(dw_support_all[self.z_res_pos, :], dtype=float)
        if np.flatnonzero(np.abs(w_local) > WEIGHT_ZERO_TOL).size == 0:
            return self.w_res_anchor_local.copy(), np.zeros((self.Z_res.size, self.n_primary), dtype=float)
        return w_local, dw_local

    def _hom_weights(self, maw_models, q_p, E):
        if self.maw_hom_componentwise:
            return np.vstack([
                _evaluate_maw_hom_weights_current(
                    q_p, E, m, n_elem_reference=self.n_elem_reference,
                    n_current_elements=self.n_current_elements,
                    full_to_local=self.full_to_local_hom, eval_mode=self.maw_hom_eval_mode,
                )
                for m in maw_models
            ])
        return _evaluate_maw_hom_weights_current(
            q_p, E, maw_models, n_elem_reference=self.n_elem_reference,
            n_current_elements=self.n_current_elements,
            full_to_local=self.full_to_local_hom, eval_mode=self.maw_hom_eval_mode,
        )

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
        """Thin driver over _evaluate_impl_gen: serves every yielded request
        with the exact same serial self._eval_qs_and_jac/
        _compute_weighted_decoder_hessian calls the old inline code used, in
        the same order -- so this reproduces the original method exactly.
        The generator itself is the single source of truth for the Newton
        loop's control flow; evaluate_with_tangent_batch drives the SAME
        generator logic across many points at once, batching the requests
        instead of serving them one at a time, so the two code paths cannot
        drift apart."""
        gen = self._evaluate_impl_gen(E, q_prev=q_prev, step_index=step_index, _cache_out=_cache_out)
        try:
            request = next(gen)
            while True:
                kind = request[0]
                if kind == "decode":
                    result = self._eval_qs_and_jac(request[1], E)
                elif kind == "hessian":
                    result = self._compute_weighted_decoder_hessian(request[1], request[2])
                else:
                    raise RuntimeError(f"[HpromAnnIterativeLawFloat64] unknown generator request {kind!r}")
                request = gen.send(result)
        except StopIteration as exc:
            return exc.value

    def _evaluate_impl_gen(self, E, q_prev=None, step_index=1, _cache_out=None):
        """Generator core of _evaluate_impl: identical Newton-correction
        control flow, but yields ("decode", q_p) / ("hessian", q_p, weights)
        at the two torch-heavy call sites instead of calling
        self._eval_qs_and_jac/_compute_weighted_decoder_hessian directly, so
        an external driver (serial: _evaluate_impl; batched: see
        evaluate_with_tangent_batch) can serve those requests however it
        likes -- serially for one point, or batched via vmap across many
        concurrently-running instances of this same generator for a whole
        macro Gauss-point set. Do not duplicate this control flow anywhere
        else; add new callers by driving this generator instead."""
        E = np.asarray(E, dtype=float).reshape(-1)
        if q_prev is None:
            q_prev = np.zeros(self.n_primary, dtype=float)
        u_aff_free = self._affine(E, self.x_free, self.y_free, self.is_x_free)
        disp_base = np.zeros(self.n_total_dof, dtype=float)
        disp_base[self.dir_dofs] = self._affine(E, self.x_dir, self.y_dir, self.is_x_dir)

        q_p = self._initial_qp_guess(E, q_prev, step_index)
        q_step_start = q_p.copy()

        it = 0
        converged = False
        nonfinite_detected = False
        res_norm_0 = None
        Kr_last = None
        dq_norm_prev = None
        prev_res_norm = None
        prev_q_eval = None
        prev_qs_eval = None
        plateau_count = 0
        best_q, best_res, best_rel = q_step_start.copy(), np.inf, np.inf

        while it < self.max_its:
            q_s_map, J_ann_raw = yield ("decode", q_p)
            q_s = q_s_map - self.q0_const_np - self.J0_const_np @ q_p
            J_ann = J_ann_raw - self.J0_const_np

            if not (np.all(np.isfinite(q_p)) and np.all(np.isfinite(q_s))):
                nonfinite_detected = True
                break
            q_eval_delta = np.inf
            q_eval_delta_normalized = np.inf
            if prev_q_eval is not None and prev_q_eval.shape == q_p.shape:
                q_change = q_p - prev_q_eval
                q_eval_delta = float(np.linalg.norm(q_change))
                q_eval_delta_normalized = float(np.linalg.norm(q_change / self.q_m_scale))
            prev_q_eval = q_p.copy()
            q_s_delta = np.inf
            if prev_qs_eval is not None and prev_qs_eval.shape == q_s.shape:
                q_s_delta = float(np.linalg.norm(q_s - prev_qs_eval))
            prev_qs_eval = q_s.copy()

            u_fluc = self.w0_const_np + self.phi_p_eff_np @ q_p + self.phi_s @ q_s
            if not np.all(np.isfinite(u_fluc)):
                nonfinite_detected = True
                break
            u_free = u_aff_free + u_fluc
            J_manifold = self.phi_p_eff_np + self.phi_s @ J_ann
            if not np.all(np.isfinite(J_manifold)):
                nonfinite_detected = True
                break

            u_eq_curr = disp_base.copy()
            u_eq_curr[self.free_dofs] = u_free

            InitializeNonLinearIteration(self.entities, self.mp.ProcessInfo)
            w_res_iter, dw_res_local = self._residual_weights_and_jacobian(q_p)
            KJ, r_full = self.dyn_res_assembler.assemble_reduced_action(u_eq_curr, w_res_iter, J_manifold)
            FinalizeNonLinearIteration(self.entities, self.mp.ProcessInfo)
            # r_full (assemble_reduced_action's own persistent _rhs_free
            # buffer), the reduced assembler's own _K_total, and its own
            # _f_int are all mutated in place on every call, by ANY
            # generator once evaluate_with_tangent_batch interleaves many
            # at once -- must snapshot all three HERE, immediately, in this
            # same uninterrupted span (no yield since the assembly call
            # above). Reading any of them below the "hessian" yield (as the
            # original inline code did, harmlessly for a single serial
            # point) is NOT equivalent once batched: other generators' own
            # assembly calls run in that gap and silently overwrite these
            # buffers first -- caught this session via a real, large (up to
            # 120% relative) corruption on points that finish mid-pack
            # while others are still iterating. This isn't just a caching
            # nicety: the _f_int snapshot feeds K_r, which drives the
            # actual Newton update below, so a stale read here would have
            # corrupted the iteration path itself, not merely the tangent.
            r_full = r_full.copy()
            K_total_support_snapshot = self.dyn_res_assembler._assembler._K_total.copy()
            f_int_snapshot = self.dyn_res_assembler._assembler._f_int.copy()

            if not (np.all(np.isfinite(r_full)) and np.all(np.isfinite(KJ))):
                nonfinite_detected = True
                break

            r_r = J_manifold.T @ r_full
            K_std = J_manifold.T @ KJ
            if self.include_manifold_curvature:
                curvature_weights = self.phi_s.T @ r_full
                K_curv = yield ("hessian", q_p, curvature_weights)
                K_curv = 0.5 * (K_curv + K_curv.T)
                K_r = K_std - K_curv
            else:
                K_r = K_std
            if dw_res_local is not None:
                # Uses the snapshot taken right after assembly above, NOT a
                # live read -- this runs after the "hessian" yield, where
                # another point's own assembly call may have already
                # overwritten the live buffer (see that snapshot's own
                # comment).
                rhs_unit_loc = -np.asarray(f_int_snapshot, dtype=float).reshape(
                    self.dyn_res_assembler.n_sel, int(self.dyn_res_assembler._assembler.n_local_dof)
                )
                r_free_e = _scatter_rhs_loc_by_element_free(
                    rhs_loc=rhs_unit_loc, n_free=self.free_dofs.size, scatter_cache=self.unit_rhs_scatter,
                )
                drw_dqp = r_free_e @ dw_res_local
                K_r = K_r - (J_manifold.T @ drw_dqp)
            Kr_last = K_r

            if not (np.all(np.isfinite(r_r)) and np.all(np.isfinite(K_r))):
                nonfinite_detected = True
                break

            res_norm = float(np.linalg.norm(r_r))
            if not np.isfinite(res_norm):
                nonfinite_detected = True
                break
            if res_norm_0 is None:
                res_norm_0 = max(res_norm, 1e-30)
            rel_res = res_norm / (res_norm_0 + 1e-12)
            if res_norm < best_res:
                best_res, best_q, best_rel = float(res_norm), q_p.copy(), float(rel_res)
            if self.verbose:
                print(f"    it {it:02d}: ||R_r||={res_norm:.3e}, rel={rel_res:.3e}")

            if res_norm < self.abs_res_cutoff:
                converged = True
                break
            if it > 0 and np.isfinite(q_eval_delta_normalized) and q_eval_delta_normalized < self.normalized_dq_cutoff:
                converged = True
                break
            if (dq_norm_prev is not None and dq_norm_prev < self.dq_abs_cutoff
                    and rel_res < self.relnorm_cutoff and res_norm < self.max_res_for_rel_convergence):
                converged = True
                break
            if prev_res_norm is not None:
                rel_drop = abs(prev_res_norm - res_norm) / max(prev_res_norm, 1e-30)
                q_tol = 20.0 * self.dq_abs_cutoff
                q_is_flat = (np.isfinite(q_eval_delta) and np.isfinite(q_s_delta)
                             and q_eval_delta < q_tol and q_s_delta < q_tol)
                if it >= 2 and q_is_flat and rel_res < max(self.relnorm_cutoff, 0.5 * self.stagnation_relnorm_gate):
                    converged = True
                    break
                if rel_drop < self.min_rel_drop_stop and q_is_flat:
                    plateau_count += 1
                else:
                    plateau_count = 0
                if (rel_drop < self.min_rel_drop_stop and rel_res < self.stagnation_relnorm_gate
                        and res_norm < self.max_res_for_rel_convergence):
                    converged = True
                    break
                if (plateau_count >= 3
                        and rel_res < max(3.0 * self.relnorm_cutoff, self.stagnation_relnorm_gate)
                        and (dq_norm_prev is None or dq_norm_prev < 5.0 * self.dq_abs_cutoff)):
                    converged = True
                    break
            prev_res_norm = float(res_norm)

            K_solve = K_r
            dq_p = self._solve_reduced_system(K_solve, r_r, self.regularization)
            if not np.all(np.isfinite(dq_p)):
                nonfinite_detected = True
                break
            dq_norm = float(np.linalg.norm(dq_p))
            if dq_norm > self.max_dq_norm and dq_norm > 0.0:
                dq_p *= self.max_dq_norm / dq_norm
                dq_norm = float(np.linalg.norm(dq_p))
            q_trial = q_p + dq_p
            if not np.all(np.isfinite(q_trial)):
                nonfinite_detected = True
                break
            q_p = q_trial
            dq_norm_prev = dq_norm
            it += 1

        # True only if the loop broke via one of the in-loop convergence
        # checks above -- in that case q_p (not yet updated this iteration)
        # and J_manifold/Kr_last/w_res_iter/u_eq_curr/r_full were all just
        # computed FROM that same q_p, so they're a valid, exact linearization
        # to hand back to evaluate_with_tangent instead of recomputing it via
        # _residual_jacobian_at. NOT true after the fallback block below,
        # which can reassign q_p to an earlier best_q/q_step_start.
        converged_cleanly = converged

        if not converged:
            quasi_converged = np.isfinite(best_res) and best_rel < self.relnorm_cutoff and best_res < self.max_res_for_rel_convergence
            plateau_quasi = (np.isfinite(best_res) and plateau_count >= 3
                              and best_rel < max(3.0 * self.relnorm_cutoff, self.stagnation_relnorm_gate))
            if quasi_converged or plateau_quasi:
                q_p = best_q.copy()
                converged = True
            else:
                if self.fail_on_nonconvergence:
                    raise RuntimeError(f"HPROM-ANN iterative did not converge in {self.max_its} iters.")
                q_p = best_q.copy() if np.isfinite(best_res) else q_step_start.copy()

        if _cache_out is not None and converged_cleanly:
            # r_full/K_total_support are the already-snapshotted (.copy()'d
            # right after assemble_reduced_action, above) values -- do NOT
            # read self.dyn_res_assembler._assembler._K_total live here:
            # this code runs one "hessian" yield after the assembly call,
            # and in evaluate_with_tangent_batch other points' own assembly
            # calls interleave in exactly that gap, so a live read here
            # would silently pick up whichever point ran most recently
            # instead of this point's own state.
            _cache_out.update(
                q_p=q_p.copy(), J_manifold=J_manifold, K_r=Kr_last,
                w_res_iter=w_res_iter, u_eq_curr=u_eq_curr.copy(), r_full=r_full,
                K_total_support=K_total_support_snapshot,
            )

        q_s_final_map = self._eval_qs_only(q_p)
        q_s_final = q_s_final_map - self.q0_const_np - self.J0_const_np @ q_p
        u_fluc_final = self.w0_const_np + self.phi_p_eff_np @ q_p + self.phi_s @ q_s_final
        if not np.all(np.isfinite(u_fluc_final)):
            q_p = q_step_start.copy()
            q_s_final_map = self._eval_qs_only(q_p)
            q_s_final = q_s_final_map - self.q0_const_np - self.J0_const_np @ q_p
            u_fluc_final = self.w0_const_np + self.phi_p_eff_np @ q_p + self.phi_s @ q_s_final

        disp = disp_base.copy()
        disp[self.free_dofs] = u_aff_free + u_fluc_final
        SetDisplacementFromEquationVector(disp, self.eq_id_map, self.ta)
        UpdateCurrentCoordinatesFromDisplacement(self.mp, step=0)
        self.vec_assembler.ComputeLocalArrays(disp)

        w_eps = self._hom_weights(self.maw_eps_hom, q_p, E)
        w_sig = self._hom_weights(self.maw_sig_hom, q_p, E)
        hom_eps, hom_sig = CalculateHomogenizedFromAssemblerWithElementWeights(
            self.vec_assembler, w_eps=w_eps, w_sig=w_sig, reference_measure=self.hom_reference_measure,
        )
        return hom_eps, hom_sig, q_p, it, converged

    def _hom_weights_and_jacobian(self, maw_models, E):
        if self.maw_hom_componentwise:
            results = [
                maw_hom_weight_and_jacobian_single_model(
                    E, m, n_elem_reference=self.n_elem_reference,
                    n_current_elements=self.n_current_elements,
                    full_to_local=self.full_to_local_hom,
                )
                for m in maw_models
            ]
            w = np.vstack([r[0] for r in results])
            dw_dE = np.stack([r[1] for r in results], axis=0)
            return w, dw_dE
        return maw_hom_weight_and_jacobian_single_model(
            E, maw_models, n_elem_reference=self.n_elem_reference,
            n_current_elements=self.n_current_elements,
            full_to_local=self.full_to_local_hom,
        )

    def _residual_jacobian_at(self, q_p, E, disp_base, u_aff_free):
        q_s_map, J_ann_raw = self._eval_qs_and_jac(q_p, E)
        q_s = q_s_map - self.q0_const_np - self.J0_const_np @ q_p
        J_ann = J_ann_raw - self.J0_const_np
        u_fluc = self.w0_const_np + self.phi_p_eff_np @ q_p + self.phi_s @ q_s
        u_free = u_aff_free + u_fluc
        J_manifold = self.phi_p_eff_np + self.phi_s @ J_ann

        u_eq_curr = disp_base.copy()
        u_eq_curr[self.free_dofs] = u_free

        w_res_iter, dw_res_local = self._residual_weights_and_jacobian(q_p)
        KJ, r_full = self.dyn_res_assembler.assemble_reduced_action(u_eq_curr, w_res_iter, J_manifold)

        K_std = J_manifold.T @ KJ
        if self.include_manifold_curvature:
            curvature_weights = self.phi_s.T @ r_full
            K_curv = self._compute_weighted_decoder_hessian(q_p, curvature_weights)
            K_curv = 0.5 * (K_curv + K_curv.T)
            K_r = K_std - K_curv
        else:
            K_r = K_std
        if dw_res_local is not None:
            rhs_unit_loc = -np.asarray(self.dyn_res_assembler._assembler._f_int, dtype=float).reshape(
                self.dyn_res_assembler.n_sel, int(self.dyn_res_assembler._assembler.n_local_dof)
            )
            r_free_e = _scatter_rhs_loc_by_element_free(
                rhs_loc=rhs_unit_loc, n_free=self.free_dofs.size, scatter_cache=self.unit_rhs_scatter,
            )
            drw_dqp = r_free_e @ dw_res_local
            K_r = K_r - (J_manifold.T @ drw_dqp)

        return J_manifold, K_r, w_res_iter, u_eq_curr, r_full

    def dqp_dE_at(self, q_p, E, disp_base, u_aff_free, J_manifold=None, K_r=None, w_res_iter=None,
                  K_total_support=None):
        if J_manifold is None or K_r is None or w_res_iter is None:
            J_manifold, K_r, w_res_iter, _u_eq_curr, _r_full = self._residual_jacobian_at(
                q_p, E, disp_base, u_aff_free,
            )
            # _residual_jacobian_at just refreshed the reduced assembler's
            # own _K_total via its own assemble_reduced_action call -- read
            # it live below rather than trust a caller-supplied snapshot
            # that predates this recompute (a caller that reaches this
            # branch didn't have a valid J_manifold/K_r/w_res_iter either,
            # so any K_total_support it passed is equally stale).
            K_total_support = None

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

        support_assembler = self.dyn_res_assembler._assembler
        local_eq_ids_support = support_assembler.local_eq_ids
        du_local_dE_support = du_aff_dE[local_eq_ids_support, :]
        if K_total_support is None:
            K_total_support = support_assembler._K_total
        d_neg_fint_dE_support = -np.einsum("eij,ejk->eik", K_total_support, du_local_dE_support)

        dr_full_dE = np.zeros((self.free_dofs.size, 3), dtype=float)
        for k in range(3):
            scattered_e = _scatter_rhs_loc_by_element_free(
                rhs_loc=d_neg_fint_dE_support[:, :, k], n_free=self.free_dofs.size,
                scatter_cache=self.unit_rhs_scatter,
            )
            dr_full_dE[:, k] = scattered_e @ w_res_iter

        dG_dE = J_manifold.T @ dr_full_dE
        dq_p_dE = np.linalg.solve(K_r, dG_dE)
        return dq_p_dE, J_manifold, K_r, w_res_iter

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

            # _evaluate_impl's own last Newton iteration already computed this
            # exact linearization (at this exact q_p) right before declaring
            # convergence -- reuse it instead of recomputing decoder+Hessian+
            # assembly a second time. Falls back to the original recompute
            # whenever the cache wasn't populated (non-clean convergence path)
            # or (defensively) doesn't match the returned q_p.
            K_total_support = None
            if _cache and np.array_equal(_cache["q_p"], q_p):
                J_manifold, K_r, w_res_iter, u_eq_curr, _r_full = (
                    _cache["J_manifold"], _cache["K_r"], _cache["w_res_iter"],
                    _cache["u_eq_curr"], _cache["r_full"],
                )
                K_total_support = _cache["K_total_support"]
            else:
                J_manifold, K_r, w_res_iter, u_eq_curr, _r_full = self._residual_jacobian_at(
                    q_p, E, disp_base, u_aff_free,
                )
            dq_p_dE, J_manifold, K_r, w_res_iter = self.dqp_dE_at(
                q_p, E, disp_base, u_aff_free, J_manifold=J_manifold, K_r=K_r, w_res_iter=w_res_iter,
                K_total_support=K_total_support,
            )

            self.vec_assembler.ComputeLocalArrays(u_eq_curr)

        w_eps, dw_eps_dE = self._hom_weights_and_jacobian(self.maw_eps_hom, E)
        w_sig, dw_sig_dE = self._hom_weights_and_jacobian(self.maw_sig_hom, E)
        # Recomputed via u_eq_curr (this method's own second ComputeLocalArrays
        # call above) -- a consistency check between the two displacement-
        # reconstruction paths (_evaluate_impl's own vs. this method's).
        hom_eps_check, hom_sig_check = CalculateHomogenizedFromAssemblerWithElementWeights(
            self.vec_assembler, w_eps=w_eps, w_sig=w_sig, reference_measure=self.hom_reference_measure,
        )

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
        ddisp_dE[self.free_dofs, :] += J_manifold @ dq_p_dE

        local_eq_ids = self.vec_assembler.local_eq_ids
        du_local_dE = ddisp_dE[local_eq_ids, :]
        B = self.vec_assembler._B
        CC = self.vec_assembler._CC
        dE_gp_dE = np.einsum("egvc,ecx->egvx", B, du_local_dE)
        dS_gp_dE = np.einsum("egvw,egwx->egvx", CC, dE_gp_dE)
        dEpsbar_e_dE = np.mean(dE_gp_dE, axis=1)
        dSbar_e_dE = np.mean(dS_gp_dE, axis=1)

        eps_mean_e = np.mean(self.vec_assembler._E_voigt, axis=1)
        sig_mean_e = np.mean(self.vec_assembler._S_voigt, axis=1)
        if hasattr(self.vec_assembler, "area_e"):
            area_e = np.asarray(self.vec_assembler.area_e, dtype=float).reshape(-1)
        else:
            area_e = np.sum(np.asarray(self.vec_assembler.w_detJ, dtype=float), axis=1)
        den = float(self.hom_reference_measure)

        dEps_hom_dE = DHpromAnnDirectLaw._dhom_dE(eps_mean_e, dEpsbar_e_dE, w_eps, dw_eps_dE, area_e, den)
        dSig_hom_dE = DHpromAnnDirectLaw._dhom_dE(sig_mean_e, dSbar_e_dE, w_sig, dw_sig_dE, area_e, den)

        return hom_eps, hom_sig, q_p, n_iters, converged, dEps_hom_dE, dSig_hom_dE, hom_eps_check, hom_sig_check

    def _drive_evaluate_impl_gens_batch(self, E_batch, q_prev_batch, step_index_batch):
        """Runs one _evaluate_impl_gen generator per row of E_batch to
        completion in lockstep: every round, resumes every still-running
        generator once, groups whatever it yields this round by kind
        ("decode"/"hessian"), and serves each group with ONE batched
        _eval_qs_and_jac_batch/_compute_weighted_decoder_hessian_batch call
        instead of one call per generator. A generator's own control flow
        (convergence checks, max_its, early nonfinite breaks, fallback
        logic) is untouched by this -- interleaving with other generators
        changes nothing about what any one of them computes or when it
        declares convergence, only who serves its 2 yield sites and how
        many of them get batched together. Kratos calls made between a
        generator's yields (InitializeNonLinearIteration/
        assemble_reduced_action/FinalizeNonLinearIteration, which mutate
        shared assembler state) always run to completion inside a single
        .send() before the driver moves to the next generator, so they are
        never interleaved across points -- only the torch calls are.
        Returns a list of (hom_eps, hom_sig, q_p, n_iters, converged,
        cache_dict) tuples, one per row of E_batch, in order."""
        n = E_batch.shape[0]
        caches = [dict() for _ in range(n)]
        results = [None] * n
        with true_neo_hookean_active():
            gens = [
                self._evaluate_impl_gen(
                    E_batch[i], q_prev=q_prev_batch[i], step_index=step_index_batch[i], _cache_out=caches[i],
                )
                for i in range(n)
            ]
            pending = list(range(n))
            to_send = {i: None for i in pending}

            while pending:
                decode_owners, decode_qp = [], []
                hessian_owners, hessian_qp, hessian_w = [], [], []
                next_pending = []
                for i in pending:
                    try:
                        y = gens[i].send(to_send[i])
                    except StopIteration as exc:
                        results[i] = exc.value
                        continue
                    kind = y[0]
                    if kind == "decode":
                        decode_owners.append(i)
                        decode_qp.append(y[1])
                    elif kind == "hessian":
                        hessian_owners.append(i)
                        hessian_qp.append(y[1])
                        hessian_w.append(y[2])
                    else:
                        raise RuntimeError(f"[HpromAnnIterativeLawFloat64] unknown generator request {kind!r}")
                    next_pending.append(i)

                if decode_owners:
                    q_s_map_b, jac_b = self._eval_qs_and_jac_batch(np.stack(decode_qp))
                    for j, i in enumerate(decode_owners):
                        to_send[i] = (q_s_map_b[j], jac_b[j])
                if hessian_owners:
                    K_curv_b = self._compute_weighted_decoder_hessian_batch(
                        np.stack(hessian_qp), np.stack(hessian_w),
                    )
                    for j, i in enumerate(hessian_owners):
                        to_send[i] = K_curv_b[j]

                pending = next_pending

        return [(*results[i], caches[i]) for i in range(n)]

    def evaluate_with_tangent_batch(self, E_batch, q_prev_batch=None, step_index_batch=None):
        """Batched form of evaluate_with_tangent across many independent
        macro Gauss points at once. The inner Newton correction for every
        point runs through _drive_evaluate_impl_gens_batch (batching the
        decoder-Jacobian and weighted-Hessian torch calls -- 74.5% of
        per-point cost measured this session -- across whichever points
        are still active at each shared iteration depth); hom_sig/
        dSig_hom_dE themselves are a cheap per-point masked weighted average
        (the same CalculateHomogenizedFromAssemblerWithElementWeights path
        hom_eps always used), so no separate batched stage is needed for
        them -- see dhprom_ann_direct_law_float64_claude.py's own
        dhprom_ann_pk2_2d_vectorized_consistent_float64 docstring for why an
        earlier version of this routed hom_sig through a separate per-
        element reaction-force integrand instead, and why that silently
        produced zero stress against this project's currently-deployed
        MAW-ECM sig weights.
        Kratos assembly (the Z_res-support residual assembly inside the
        Newton loop, and the full-mesh ComputeLocalArrays here) stays a
        per-point Python loop: both are cheap (6.0%/2.1% of per-point cost
        measured this session) and mutate shared assembler state that
        cannot be batched without touching Kratos itself.

        q_prev_batch/step_index_batch: per-point warm-start inputs, same
        contract as evaluate_with_tangent's own q_prev/step_index -- pass
        the previous load step's own converged q_p per point (continuation
        mode) or leave as None for an all-cold-start (mu_affine) batch.

        Returns arrays stacked over the batch axis, matching
        evaluate_with_tangent's own per-point return signature: (hom_eps,
        hom_sig, q_p, n_iters, converged, dEps_hom_dE, dSig_hom_dE,
        hom_eps_check, hom_sig_check)."""
        E_batch = np.asarray(E_batch, dtype=float).reshape(-1, 3)
        n = E_batch.shape[0]
        if q_prev_batch is None:
            q_prev_batch = np.zeros((n, self.n_primary), dtype=float)
        if step_index_batch is None:
            step_index_batch = np.ones(n, dtype=int)

        per_point = self._drive_evaluate_impl_gens_batch(E_batch, q_prev_batch, step_index_batch)
        hom_eps_list = [p[0] for p in per_point]
        hom_sig_list = [p[1] for p in per_point]
        q_p_b = np.stack([p[2] for p in per_point])
        n_iters_b = np.array([p[3] for p in per_point])
        converged_b = np.array([p[4] for p in per_point], dtype=bool)
        caches = [p[5] for p in per_point]

        dEps_hom_dE_list, dSig_hom_dE_list = [], []
        hom_eps_check_list, hom_sig_check_list = [], []

        for i in range(n):
            E = E_batch[i]
            q_p = q_p_b[i]
            _cache = caches[i]

            u_aff_free = self._affine(E, self.x_free, self.y_free, self.is_x_free)
            disp_base = np.zeros(self.n_total_dof, dtype=float)
            disp_base[self.dir_dofs] = self._affine(E, self.x_dir, self.y_dir, self.is_x_dir)

            with true_neo_hookean_active():
                K_total_support = None
                if _cache and np.array_equal(_cache["q_p"], q_p):
                    J_manifold, K_r, w_res_iter, u_eq_curr, _r_full = (
                        _cache["J_manifold"], _cache["K_r"], _cache["w_res_iter"],
                        _cache["u_eq_curr"], _cache["r_full"],
                    )
                    K_total_support = _cache["K_total_support"]
                else:
                    J_manifold, K_r, w_res_iter, u_eq_curr, _r_full = self._residual_jacobian_at(
                        q_p, E, disp_base, u_aff_free,
                    )
                dq_p_dE, J_manifold, K_r, w_res_iter = self.dqp_dE_at(
                    q_p, E, disp_base, u_aff_free, J_manifold=J_manifold, K_r=K_r, w_res_iter=w_res_iter,
                    K_total_support=K_total_support,
                )

                self.vec_assembler.ComputeLocalArrays(u_eq_curr)

            w_eps, dw_eps_dE = self._hom_weights_and_jacobian(self.maw_eps_hom, E)
            w_sig, dw_sig_dE = self._hom_weights_and_jacobian(self.maw_sig_hom, E)
            hom_eps_check, hom_sig_check = CalculateHomogenizedFromAssemblerWithElementWeights(
                self.vec_assembler, w_eps=w_eps, w_sig=w_sig, reference_measure=self.hom_reference_measure,
            )

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
            ddisp_dE[self.free_dofs, :] += J_manifold @ dq_p_dE

            local_eq_ids = self.vec_assembler.local_eq_ids
            du_local_dE = ddisp_dE[local_eq_ids, :]
            B = self.vec_assembler._B
            CC = self.vec_assembler._CC
            dE_gp_dE = np.einsum("egvc,ecx->egvx", B, du_local_dE)
            dS_gp_dE = np.einsum("egvw,egwx->egvx", CC, dE_gp_dE)
            dEpsbar_e_dE = np.mean(dE_gp_dE, axis=1)
            dSbar_e_dE = np.mean(dS_gp_dE, axis=1)

            eps_mean_e = np.mean(self.vec_assembler._E_voigt, axis=1)
            sig_mean_e = np.mean(self.vec_assembler._S_voigt, axis=1)
            if hasattr(self.vec_assembler, "area_e"):
                area_e = np.asarray(self.vec_assembler.area_e, dtype=float).reshape(-1)
            else:
                area_e = np.sum(np.asarray(self.vec_assembler.w_detJ, dtype=float), axis=1)
            den = float(self.hom_reference_measure)
            dEps_hom_dE = DHpromAnnDirectLaw._dhom_dE(eps_mean_e, dEpsbar_e_dE, w_eps, dw_eps_dE, area_e, den)
            dSig_hom_dE = DHpromAnnDirectLaw._dhom_dE(sig_mean_e, dSbar_e_dE, w_sig, dw_sig_dE, area_e, den)

            dEps_hom_dE_list.append(dEps_hom_dE)
            dSig_hom_dE_list.append(dSig_hom_dE)
            hom_eps_check_list.append(hom_eps_check)
            hom_sig_check_list.append(hom_sig_check)

        return (
            np.stack(hom_eps_list), np.stack(hom_sig_list), q_p_b, n_iters_b, converged_b,
            np.stack(dEps_hom_dE_list), np.stack(dSig_hom_dE_list),
            np.stack(hom_eps_check_list), np.stack(hom_sig_check_list),
        )


_DEFAULT_LAW_F64 = None


def get_law_float64(**kwargs):
    global _DEFAULT_LAW_F64
    if _DEFAULT_LAW_F64 is None:
        kwargs.setdefault("qp_init_mode", "mu_affine")
        _DEFAULT_LAW_F64 = HpromAnnIterativeLawFloat64(**kwargs)
    return _DEFAULT_LAW_F64


def hprom_ann_iterative_pk2_2d_vectorized_consistent_float64(E_flat, young=None, poisson=None):
    """Same contract as hprom_ann_iterative_law_claude.py's
    hprom_ann_iterative_pk2_2d_vectorized_consistent, using the float64 law.

    Always a cold (mu_affine) start for every row, same as before this
    session's batching work -- callers that need per-point warm-starting
    across calls (continuation) should use HpromIterativeContinuationWrapper
    in run_cruciform_fe2_claude.py instead, which drives
    HpromAnnIterativeLawFloat64.evaluate_with_tangent_batch directly with
    its own per-point q_prev/step_index bookkeeping."""
    law = get_law_float64()
    E_flat = np.asarray(E_flat, dtype=float)
    n = E_flat.shape[0]
    _, S, _, _, _, _, CC, _, _ = law.evaluate_with_tangent_batch(E_flat)
    return S, CC


def hprom_ann_iterative_pk2_2d_vectorized_float64(E_flat, young=None, poisson=None, h=1.0e-4):
    """Same contract as hprom_ann_iterative_law_claude.py's
    hprom_ann_iterative_pk2_2d_vectorized (finite-difference tangent),
    using the float64 law's evaluate() instead of the float32 original's."""
    law = get_law_float64()
    E_flat = np.asarray(E_flat, dtype=float)
    n = E_flat.shape[0]
    S = np.zeros((n, 3), dtype=float)
    CC = np.zeros((n, 3, 3), dtype=float)
    for i in range(n):
        E = E_flat[i]
        _, S[i], _, _, _ = law.evaluate(E)
        for k in range(3):
            Ep, Em = E.copy(), E.copy()
            Ep[k] += h
            Em[k] -= h
            _, Sp, _, _, _ = law.evaluate(Ep)
            _, Sm, _, _, _ = law.evaluate(Em)
            CC[i, :, k] = (Sp - Sm) / (2.0 * h)
    return S, CC


if __name__ == "__main__":
    print("[hprom_ann_iterative_law_float64] building law instance ...")
    law = HpromAnnIterativeLawFloat64(verbose=True)
    eps0, sig0, q0, n_it, conv = law.evaluate(np.array([0.05, 0.0, 0.0]))
    print(f"[hprom_ann_iterative_law_float64] evaluate([0.05,0,0]) -> eps={eps0}, sig={sig0}, "
          f"iters={n_it}, converged={conv}")
