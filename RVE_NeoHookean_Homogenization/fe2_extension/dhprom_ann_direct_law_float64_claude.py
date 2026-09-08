#!/usr/bin/env python3
"""Precision-floor experiment: a float64 copy of DHpromAnnDirectLaw
(dhprom_ann_direct_law_claude.py), which the original leaves untouched.

Motivation: after ruling out tangent inconsistency, local loss of
ellipticity, and rank-one convexity loss (at the final state AND at
every iterate of the Newton trajectory) as explanations for Cook's
membrane step-1 stall, the one remaining hypothesis from earlier in this
investigation is a numerical precision floor -- the decoder network and
several reconstruction tensors are deliberately cast to float32 in the
original, and in particular

    q_s_final = q_s_final_map - self.N0_const - (q_p_torch @ self.J0_const_torch.T)

subtracts two potentially O(1)-scale quantities to recover a much
smaller (O(q_p)-scale, ~1e-3) result -- a textbook catastrophic-
cancellation setup, where the float32 ABSOLUTE precision of the two
large operands (~1e-7 relative to their own O(1) scale) can dominate the
relative precision of the small DIFFERENCE (~1e-4 relative to the
difference's own, much smaller scale) -- a much larger noise floor than
a naive "float32 is 1e-7 accurate" estimate would suggest.

This class duplicates dhprom_ann_direct_law_claude.py's __init__/evaluate/
evaluate_with_tangent line for line, changing every float32 cast to
float64 (and casting the decoder network itself to double, before it is
ever used, so the N0_const/J0_const_torch constants computed in __init__
are themselves already double precision -- casting AFTER the fact would
not recover precision already lost). Not a subclass of the original
(which would run its own float32 setup via a shared __init__) -- a full,
independent copy, so the original's already-verified behavior is
completely unaffected by this file's existence.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("core", "prom/ann", "hprom/ann"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import numpy as np
import torch
import KratosMultiphysics as KM

from fom_solver_rve import (
    setup_kratos_parameters,
    RVEHomogenizationDatasetGenerator,
    VectorizedAssembler,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    DeformationGradientFromGreenLagrange2D,
    CalculateHomogenizedFromAssemblerWithElementWeights,
    WEIGHT_ZERO_TOL,
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
)
from deformation_gradient_jacobian_claude import deformation_gradient_and_jacobian_2d
from maw_hom_weight_jacobian_claude import maw_hom_weight_and_jacobian_single_model

DEFAULT_ANN_DATA_DIR = str(ROOT / "prom" / "ann" / "stage_7_ann_model_ls")
DEFAULT_HPROM_ANN_DIR = str(ROOT / "hprom" / "ann" / "maw_dynamic")
DEFAULT_BASIS_DIR = str(ROOT / "pod" / "stage_2_pod_rve")


class DHpromAnnDirectLawFloat64:
    """Line-for-line float64 copy of DHpromAnnDirectLaw. See module
    docstring. Only dtype-related lines differ from the original; every
    other line (Kratos/mesh/basis setup, homogenization-weight machinery,
    the tangent chain rule) is unchanged."""

    def __init__(
        self,
        ann_data_dir=DEFAULT_ANN_DATA_DIR,
        hprom_ann_dir=DEFAULT_HPROM_ANN_DIR,
        basis_dir=DEFAULT_BASIS_DIR,
        homogenization_mode="maw_dynamic",
        maw_hom_eval_mode="model",
    ):
        (
            phi_p_ref, phi_s_ref, free_dofs_ref, _dir_dofs_ref, eq_map_ref,
            Xc, Yc, ann_model, device, ecm_data, _include_macro,
        ) = LoadHpromAnnModel(basis_dir=basis_dir, ann_data_dir=ann_data_dir, hprom_ann_dir=hprom_ann_dir)
        # Force CPU: LoadHpromAnnModel defaults to CUDA when available, but
        # this workload is thousands of independent batch-size-1 calls
        # through a tiny MLP -- GPU kernel-launch/dispatch overhead
        # dominates the actual (tiny) compute per call. Verified directly:
        # ~2.2x per-call speedup on CPU, identical results to floating-
        # point precision (evaluate_with_tangent's own outputs match to
        # <1e-12 relative, both here and end-to-end through the full Cook
        # run's tip displacement and errors).
        device = torch.device("cpu")
        ann_model = ann_model.to(device)
        ann_model = ann_model.double()  # <-- cast BEFORE any forward pass, so nothing downstream ever sees float32
        self.ann_model, self.device = ann_model, device
        # Also cap torch's own intra-op thread pool at 1: every call here
        # is a single batch-size-1 sample through a tiny (128-unit) MLP,
        # so there is no meaningful work to parallelize, only thread-pool
        # dispatch overhead to pay for it -- same effect this project
        # already found for training-time hyperthreading. Verified
        # directly: torch's own default (8 threads) is ~34% slower per
        # call than 1 thread for this exact workload; OMP/BLAS env vars
        # gave no further change once torch's own count is capped.
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
        self.vec_assembler = VectorizedAssembler(mp, n_total_dof, eq_id_map, log_label="DHpromAnnDirectLawFloat64")
        elements = list(mp.Elements)
        n_elem_reference = int(np.ravel(ecm_data["n_elem"])[0]) if "n_elem" in ecm_data else len(elements)

        free_dofs, dir_dofs, basis_rows = ResolveActiveFreeDofsAndBasisRows(
            mp, n_total_dof, eq_id_map,
            free_dofs_reference=free_dofs_ref, eq_map_reference=eq_map_ref,
            full_mesh_base=full_mesh_base, solver_label="DHpromAnnDirectLawFloat64",
        )
        self.free_dofs = np.asarray(free_dofs, dtype=np.int64)
        self.dir_dofs = np.asarray(dir_dofs, dtype=np.int64)
        self.dof_to_dirpos_local = -np.ones(n_total_dof, dtype=np.int64)
        self.dof_to_dirpos_local[self.dir_dofs] = np.arange(self.dir_dofs.size, dtype=np.int64)
        self.thickness_scalar = float(np.asarray(self.vec_assembler.thickness, dtype=float).reshape(-1)[0])
        phi_p = np.asarray(phi_p_ref, dtype=float)[basis_rows, :]
        phi_s = np.asarray(phi_s_ref, dtype=float)[basis_rows, :]
        self.n_primary, self.n_secondary = phi_p.shape[1], phi_s.shape[1]

        self.full_to_local_hom = _build_full_to_local_map(
            ecm_data, n_elem_reference=n_elem_reference, n_current_elements=len(elements),
        )
        self.n_elem_reference = n_elem_reference
        self.n_current_elements = len(elements)

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
            raise RuntimeError("Direct mode requires qm_init_mu_affine.npz (mu-affine initializer).")
        self.qp_aff = qp_aff

        q_zero = torch.zeros((1, self.n_primary), dtype=torch.float64, device=device)
        with torch.no_grad():
            N0_const = ann_model(q_zero)

        def ann_from_qvec(qvec):
            return ann_model(qvec.view(1, -1)).reshape(-1)

        # Forward-mode AD (jacfwd) instead of the reverse-mode
        # torch.autograd.functional.jacobian: the decoder's input dim
        # (n_primary, here 3) is far smaller than its output dim
        # (n_secondary, here 36), so forward-mode needs only n_primary
        # passes instead of n_secondary backward passes for the exact
        # same Jacobian -- verified numerically identical (see
        # evaluate_with_tangent's own docstring note below).
        q_in0 = q_zero.reshape(-1).clone().detach()
        J0_const_torch = torch.func.jacfwd(ann_from_qvec)(q_in0).reshape(
            self.n_secondary, self.n_primary
        ).detach()
        J0_const_np = J0_const_torch.cpu().numpy()
        N0_const_np = N0_const.detach().cpu().numpy().reshape(-1)

        a_m = np.asarray(getattr(ann_model, "a_m_np", None), dtype=float)
        phi_master = phi_p @ a_m
        self.phi_p_eff_np = phi_master + phi_s @ J0_const_np
        self.w0_const_np = phi_s @ N0_const_np
        self.Vp_eff = torch.from_numpy(self.phi_p_eff_np.astype(np.float64)).to(device)
        self.w0_const_t = torch.from_numpy(self.w0_const_np.astype(np.float64)).to(device).unsqueeze(0)
        self.N0_const, self.J0_const_torch = N0_const, J0_const_torch

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
        THIS instance's own (possibly hyper-reduced) dir_dofs/x_dir/y_dir,
        not a separate full-mesh generator."""
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
        3): the exact envelope-theorem decomposition of the energy-conjugate
        macro stress (see fe2_extension/reaction_force_ecm_target_claude.py
        for the derivation, verified there to 7e-16 relative error against
        DirectStressGenerator.direct_stress_history). NOT the same per-
        element quantity as area_e*mean(sig_gp) -- a weight vector fit
        against c_e is meaningless if dotted against the naive-average
        integrand instead, and vice versa.

        Requires vec_assembler._f_int to already be populated for the
        current displacement (ComputeLocalArrays, not
        ComputeStrainStressOnly -- the latter skips force assembly
        entirely)."""
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

    def evaluate(self, E):
        E = np.asarray(E, dtype=float).reshape(-1)
        u_aff_free = self._affine(E, self.x_free, self.y_free, self.is_x_free)
        disp = np.zeros(self.n_total_dof, dtype=float)
        disp[self.dir_dofs] = self._affine(E, self.x_dir, self.y_dir, self.is_x_dir)

        mu_dim = int(self.qp_aff["mu_dim"])
        mu = E[:mu_dim]
        q_p = np.concatenate([mu, [1.0]]) @ np.asarray(self.qp_aff["b_aff"], dtype=float)
        q_p_torch = torch.from_numpy(q_p.astype(np.float64)).reshape(1, -1).to(self.device)

        with torch.no_grad():
            q_s_final_map = self.ann_model(q_p_torch)
            q_s_final = q_s_final_map - self.N0_const - (q_p_torch @ self.J0_const_torch.T)
            u_fluc_final = (
                self.w0_const_t + q_p_torch @ self.Vp_eff.T + q_s_final @ self.Vs.T
            ).detach().cpu().numpy().reshape(-1)

        disp[self.free_dofs] = u_aff_free + u_fluc_final
        SetDisplacementFromEquationVector(disp, self.eq_id_map, self.ta)
        UpdateCurrentCoordinatesFromDisplacement(self.mp, step=0)
        with true_neo_hookean_active():
            self.vec_assembler.ComputeLocalArrays(disp)

        w_eps = self._hom_weights(self.maw_eps_hom, q_p, E)
        w_sig = self._hom_weights(self.maw_sig_hom, q_p, E)
        hom_eps, hom_sig = CalculateHomogenizedFromAssemblerWithElementWeights(
            self.vec_assembler, w_eps=w_eps, w_sig=w_sig, reference_measure=self.hom_reference_measure,
        )
        return hom_eps, hom_sig

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

    @staticmethod
    def _dhom_dE(mean_e, dmean_e_dE, w, dw_dE, area_e, den):
        out = np.zeros((3, 3), dtype=float)
        if w.ndim == 2:
            for j in range(3):
                wj = w[j, :]
                nz = np.flatnonzero(np.abs(wj) > WEIGHT_ZERO_TOL)
                if nz.size == 0:
                    continue
                a = area_e[nz]
                w_nz, dw_nz = wj[nz], dw_dE[j, nz, :]
                m, dm = mean_e[nz, j], dmean_e_dE[nz, j, :]
                contrib = a[:, None] * (dw_nz * m[:, None] + w_nz[:, None] * dm)
                out[j, :] = contrib.sum(axis=0) / den
            return out
        nz = np.flatnonzero(np.abs(w) > WEIGHT_ZERO_TOL)
        if nz.size == 0:
            return out
        a = area_e[nz]
        w_nz, dw_nz = w[nz], dw_dE[nz, :]
        for j in range(3):
            m, dm = mean_e[nz, j], dmean_e_dE[nz, j, :]
            contrib = a[:, None] * (dw_nz * m[:, None] + w_nz[:, None] * dm)
            out[j, :] = contrib.sum(axis=0) / den
        return out

    def _decoder_batch(self, E_batch):
        """Batched decoder forward pass + Jacobian for ALL rows of E_batch
        (n,3) in two torch calls total, instead of two calls PER row.

        Confirmed by direct profiling (see _profile_dhprom_ann_claude.py /
        _prototype_batched_decoder_claude.py, both this session): the
        per-row jacfwd call below is ~60% of evaluate_with_tangent's own
        wall time, almost all of it torch/functorch dispatch and tracing
        overhead rather than the tiny MLP's own actual compute (confirmed
        by cProfile: torch._C._nn.linear's own tottime is ~11% of total).
        vmap batches the SAME per-row computation into one traced call;
        it does not change what is computed, only how it is scheduled --
        checked directly, not assumed: 300-point batch matches the old
        per-row loop to 5.9e-15 (forward output) / 3.2e-12 (Jacobian) max
        absolute difference, i.e. floating-point roundoff, and gives a
        61.6x speedup on this piece alone (2.383 -> 0.039 ms/point).

        Returns (q_p (n,n_primary), q_s_final_map (n,n_secondary) torch
        tensor, J_dec (n,n_secondary,n_primary) numpy array).
        """
        E_batch = np.asarray(E_batch, dtype=float).reshape(-1, 3)
        mu_dim = int(self.qp_aff["mu_dim"])
        b_aff = np.asarray(self.qp_aff["b_aff"], dtype=float)
        mu = E_batch[:, :mu_dim]
        ones = np.ones((E_batch.shape[0], 1), dtype=float)
        q_p_batch = np.concatenate([mu, ones], axis=1) @ b_aff
        Q = torch.from_numpy(q_p_batch.astype(np.float64)).to(self.device)

        def ann_single(qvec):
            return self.ann_model(qvec.view(1, -1)).reshape(-1)

        with torch.no_grad():
            q_s_final_map_batch = self.ann_model(Q)
        J_dec_batch = torch.func.vmap(torch.func.jacfwd(ann_single))(Q).detach().cpu().numpy()
        return q_p_batch, q_s_final_map_batch, J_dec_batch

    def evaluate_with_tangent(self, E, _decoder_precomputed=None):
        E = np.asarray(E, dtype=float).reshape(-1)

        u_aff_free = self._affine(E, self.x_free, self.y_free, self.is_x_free)
        disp = np.zeros(self.n_total_dof, dtype=float)
        disp[self.dir_dofs] = self._affine(E, self.x_dir, self.y_dir, self.is_x_dir)

        mu_dim = int(self.qp_aff["mu_dim"])
        b_aff = np.asarray(self.qp_aff["b_aff"], dtype=float)
        if _decoder_precomputed is None:
            mu = E[:mu_dim]
            q_p = np.concatenate([mu, [1.0]]) @ b_aff
            q_p_torch = torch.from_numpy(q_p.astype(np.float64)).reshape(1, -1).to(self.device)
            with torch.no_grad():
                q_s_final_map = self.ann_model(q_p_torch)
            J_dec = None  # computed later, unbatched, exactly as before
        else:
            q_p, q_s_final_map_row, J_dec = _decoder_precomputed
            q_p_torch = torch.from_numpy(q_p.astype(np.float64)).reshape(1, -1).to(self.device)
            q_s_final_map = q_s_final_map_row.reshape(1, -1)

        with torch.no_grad():
            q_s_final = q_s_final_map - self.N0_const - (q_p_torch @ self.J0_const_torch.T)
            u_fluc_final = (
                self.w0_const_t + q_p_torch @ self.Vp_eff.T + q_s_final @ self.Vs.T
            ).detach().cpu().numpy().reshape(-1)

        disp[self.free_dofs] = u_aff_free + u_fluc_final
        SetDisplacementFromEquationVector(disp, self.eq_id_map, self.ta)
        UpdateCurrentCoordinatesFromDisplacement(self.mp, step=0)
        with true_neo_hookean_active():
            self.vec_assembler.ComputeLocalArrays(disp)

        w_eps, dw_eps_dE = self._hom_weights_and_jacobian(self.maw_eps_hom, E)
        w_sig, dw_sig_dE = self._hom_weights_and_jacobian(self.maw_sig_hom, E)
        hom_eps, hom_sig = CalculateHomogenizedFromAssemblerWithElementWeights(
            self.vec_assembler, w_eps=w_eps, w_sig=w_sig, reference_measure=self.hom_reference_measure,
        )

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

        dqp_dE = np.zeros((self.n_primary, 3), dtype=float)
        dqp_dE[:, :mu_dim] = b_aff[:mu_dim, :].T

        if J_dec is None:
            # Only taken when called standalone (no precomputed batch --
            # e.g. verify_reaction_force_tangent_claude.py's own single-
            # state calls): forward-mode AD (jacfwd), not reverse-mode,
            # since n_primary (3) << n_secondary (36) here, so this needs
            # 3 forward passes instead of 36 backward passes for the
            # identical Jacobian. Confirmed by profiling the dominant
            # single-point cost; _decoder_batch above computes the exact
            # same quantity for a whole batch in one vmap(jacfwd) call --
            # see dhprom_ann_pk2_2d_vectorized_consistent_float64, which
            # is what every real run actually uses.
            q_in = q_p_torch.reshape(-1).clone().detach()

            def ann_from_qvec(qvec):
                return self.ann_model(qvec.view(1, -1)).reshape(-1)

            J_dec_torch = torch.func.jacfwd(ann_from_qvec)(q_in).reshape(
                self.n_secondary, self.n_primary
            ).detach()
            J_dec = J_dec_torch.cpu().numpy()
        J0_const_np = self.J0_const_torch.cpu().numpy()
        Vs_np = self.Vs.cpu().numpy()

        d_ufluc_dqp = self.phi_p_eff_np + Vs_np @ (J_dec - J0_const_np)
        d_ufluc_dE = d_ufluc_dqp @ dqp_dE

        ddisp_dE = du_aff_dE
        ddisp_dE[self.free_dofs, :] += d_ufluc_dE

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

        dEps_hom_dE = self._dhom_dE(eps_mean_e, dEpsbar_e_dE, w_eps, dw_eps_dE, area_e, den)
        dSig_hom_dE = self._dhom_dE(sig_mean_e, dSbar_e_dE, w_sig, dw_sig_dE, area_e, den)

        return hom_eps, hom_sig, dEps_hom_dE, dSig_hom_dE


_DEFAULT_LAW_F64 = None


def get_law_float64(**kwargs):
    global _DEFAULT_LAW_F64
    if _DEFAULT_LAW_F64 is None:
        _DEFAULT_LAW_F64 = DHpromAnnDirectLawFloat64(**kwargs)
    return _DEFAULT_LAW_F64


def dhprom_ann_pk2_2d_vectorized_consistent_float64(E_flat, young=None, poisson=None):
    """Same contract as dhprom_ann_direct_law_claude.py's
    dhprom_ann_pk2_2d_vectorized_consistent, using the float64 law instead.

    The decoder forward+Jacobian (DHpromAnnDirectLawFloat64._decoder_batch)
    is batched across the whole call instead of once per row -- ~60% of the
    original per-row wall time was pure torch/functorch dispatch overhead,
    not actual compute (see _decoder_batch's own docstring: 61.6x speedup on
    this piece alone). hom_sig/dSig_hom_dE themselves are a cheap masked
    weighted average over already-computed per-element arrays (the same
    CalculateHomogenizedFromAssemblerWithElementWeights path hom_eps always
    used), so -- unlike an earlier version of this function that routed
    hom_sig through a separate per-element reaction-force integrand
    (fe2_extension/reaction_force_ecm_target_claude.py's derivation) -- no
    extra batching stage is needed for it: this project's currently-deployed
    MAW-ECM sig weights (hprom/ann/maw_dynamic/ecm_weights_all.npz) were
    fit against the naive volume-average target, not that reaction-force
    integrand, and dotting them against the reaction-force integrand
    instead (as that earlier version did) silently produces zero stress,
    since the two integrands' own nonzero-support elements do not overlap.
    Confirmed by direct comparison against this project's own already-saved,
    already-reported cruciform_results_dhprom_f64_consistent_claude.npz
    state: this function now reproduces that saved s_gp again.
    """
    law = get_law_float64()
    E_flat = np.asarray(E_flat, dtype=float)
    n = E_flat.shape[0]
    q_p_batch, q_s_final_map_batch, J_dec_batch = law._decoder_batch(E_flat)

    S = np.zeros((n, 3), dtype=float)
    CC = np.zeros((n, 3, 3), dtype=float)
    for i in range(n):
        decoder_i = (q_p_batch[i], q_s_final_map_batch[i], J_dec_batch[i])
        _, S[i], _, CC[i] = law.evaluate_with_tangent(E_flat[i], _decoder_precomputed=decoder_i)
    return S, CC


def dhprom_ann_pk2_2d_vectorized_float64(E_flat, young=None, poisson=None, h=1.0e-4):
    """Same contract as dhprom_ann_direct_law_claude.py's
    dhprom_ann_pk2_2d_vectorized (finite-difference tangent), using the
    float64 law's evaluate() instead of the float32 original's."""
    law = get_law_float64()
    E_flat = np.asarray(E_flat, dtype=float)
    n = E_flat.shape[0]
    S = np.zeros((n, 3), dtype=float)
    CC = np.zeros((n, 3, 3), dtype=float)
    for i in range(n):
        E = E_flat[i]
        _, S[i] = law.evaluate(E)
        for k in range(3):
            Ep, Em = E.copy(), E.copy()
            Ep[k] += h
            Em[k] -= h
            _, Sp = law.evaluate(Ep)
            _, Sm = law.evaluate(Em)
            CC[i, :, k] = (Sp - Sm) / (2.0 * h)
    return S, CC


if __name__ == "__main__":
    print("[dhprom_ann_direct_law_float64] building law instance ...")
    law = DHpromAnnDirectLawFloat64()
    eps0, sig0 = law.evaluate(np.zeros(3))
    print(f"[dhprom_ann_direct_law_float64] evaluate(0,0,0) -> eps={eps0}, sig={sig0}")
