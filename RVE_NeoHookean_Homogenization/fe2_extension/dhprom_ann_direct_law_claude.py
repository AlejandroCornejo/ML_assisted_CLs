#!/usr/bin/env python3
"""Stage 1: D-HPROM-ANN direct mode (max_its=0), extracted from
RunHpromAnnBatchSimulation (hprom/ann/hprom_ann_solver_rve.py) into a
standalone, build-once/evaluate-many class, so it can be queried at
arbitrary, independent macro strains (what Cook's-membrane Gauss
points need) instead of only along one continuous trajectory.

Mirrors, line for line, the direct-mode block of the original function
(no corrector loop, qp_init_mode forced to "mu_affine" -- the only
config verified to be memoryless/trajectory-independent), reusing every
non-trivial piece (MAW-ECM homogenization-weight models, basis-row
resolution, reference-measure lookup) by importing it read-only from
hprom_ann_solver_rve.py and core/hprom_solver_rve.py. Nothing in those
files is modified.

See /home/kratos/.claude/plans/jazzy-meandering-pebble.md for the
staged plan this implements (Stage 1) and the verification discipline
that governs it.
"""
from __future__ import annotations

import os
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


class DHpromAnnDirectLaw:
    """Build-once / evaluate-many extraction of D-HPROM-ANN's direct
    (max_its=0, qp_init_mode="mu_affine") per-step block."""

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
        self.ann_model, self.device = ann_model, device

        full_mesh_base = str(np.ravel(ecm_data["hrom_full_mesh_base"])[0]) if "hrom_full_mesh_base" in ecm_data else "rve_geometry"
        hrom_mesh_base = str(np.ravel(ecm_data["hrom_mesh_base"])[0]) if "hrom_mesh_base" in ecm_data else full_mesh_base

        parameters = setup_kratos_parameters(hrom_mesh_base)
        model_kratos = KM.Model()
        sim = RVEHomogenizationDatasetGenerator(model_kratos, parameters)
        sim.Initialize()
        mp = sim._GetSolver().GetComputingModelPart()
        n_total_dof, eq_id_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
        self.mp, self.eq_id_map, self.ta, self.n_total_dof = mp, eq_id_map, ta, n_total_dof
        self.vec_assembler = VectorizedAssembler(mp, n_total_dof, eq_id_map, log_label="DHpromAnnDirectLaw")
        elements = list(mp.Elements)
        n_elem_reference = int(np.ravel(ecm_data["n_elem"])[0]) if "n_elem" in ecm_data else len(elements)

        free_dofs, dir_dofs, basis_rows = ResolveActiveFreeDofsAndBasisRows(
            mp, n_total_dof, eq_id_map,
            free_dofs_reference=free_dofs_ref, eq_map_reference=eq_map_ref,
            full_mesh_base=full_mesh_base, solver_label="DHpromAnnDirectLaw",
        )
        self.free_dofs = np.asarray(free_dofs, dtype=np.int64)
        self.dir_dofs = np.asarray(dir_dofs, dtype=np.int64)
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

        self.Vs = torch.from_numpy(phi_s.astype(np.float32)).to(device)
        qp_aff = getattr(ann_model, "qp_init_mu_affine", None)
        if qp_aff is None:
            raise RuntimeError("Direct mode requires qm_init_mu_affine.npz (mu-affine initializer).")
        self.qp_aff = qp_aff

        q_zero = torch.zeros((1, self.n_primary), device=device)
        with torch.no_grad():
            N0_const = ann_model(q_zero)
        with torch.enable_grad():
            q_in = q_zero.reshape(-1).clone().detach().requires_grad_(True)

            def ann_from_qvec(qvec):
                return ann_model(qvec.view(1, -1)).reshape(-1)

            J0_const_torch = torch.autograd.functional.jacobian(ann_from_qvec, q_in).reshape(
                self.n_secondary, self.n_primary
            ).detach()
        J0_const_np = J0_const_torch.cpu().numpy()
        N0_const_np = N0_const.detach().cpu().numpy().reshape(-1)

        a_m = np.asarray(getattr(ann_model, "a_m_np", None), dtype=float)
        phi_master = phi_p @ a_m
        self.phi_p_eff_np = phi_master + phi_s @ J0_const_np
        self.w0_const_np = phi_s @ N0_const_np
        self.Vp_eff = torch.from_numpy(self.phi_p_eff_np.astype(np.float32)).to(device)
        self.w0_const_t = torch.from_numpy(self.w0_const_np.astype(np.float32)).to(device).unsqueeze(0)
        self.N0_const, self.J0_const_torch = N0_const, J0_const_torch

    @staticmethod
    def _affine(e_vec, x_loc, y_loc, is_x_loc):
        F = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (F[0, 0] - 1.0) * x_loc + F[0, 1] * y_loc
        uy = F[1, 0] * x_loc + (F[1, 1] - 1.0) * y_loc
        return np.where(is_x_loc, ux, uy)

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
        q_p_torch = torch.from_numpy(q_p.astype(np.float32)).reshape(1, -1).to(self.device)

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
            self.vec_assembler.ComputeStrainStressOnly(disp)

        w_eps = self._hom_weights(self.maw_eps_hom, q_p, E)
        w_sig = self._hom_weights(self.maw_sig_hom, q_p, E)
        hom_eps, hom_sig = CalculateHomogenizedFromAssemblerWithElementWeights(
            self.vec_assembler, w_eps=w_eps, w_sig=w_sig, reference_measure=self.hom_reference_measure,
        )
        return hom_eps, hom_sig

    def _hom_weights_and_jacobian(self, maw_models, E):
        """Like _hom_weights, but also returns dw/dE (see
        maw_hom_weight_jacobian_claude.py). Only reachable via
        evaluate_with_tangent; evaluate() above never calls this."""
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
        """d/dE of CalculateHomogenizedFromAssemblerWithElementWeights's own
        _avg formula (core/fom_solver_rve.py), with the active weight set
        (nz = |w|>WEIGHT_ZERO_TOL) frozen at its value at the current E --
        exact except on the measure-zero set where a weight sits exactly at
        the zero-tolerance threshold (the same idealization conventionally
        used for a ReLU/max kink). den is always a constant here since
        evaluate_with_tangent always passes an explicit reference_measure.

        mean_e: (n_elem,3); dmean_e_dE: (n_elem,3,3) with [e,j,k]=d(mean_e[e,j])/dE_k.
        Non-componentwise: w (n_elem,), dw_dE (n_elem,3) with [e,k]=dw[e]/dE_k.
        Componentwise:     w (3,n_elem), dw_dE (3,n_elem,3) with [j,e,k].
        Returns (3,3) with out[j,k] = d(hom_j)/dE_k."""
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

    def evaluate_with_tangent(self, E):
        """Same (hom_eps, hom_sig) as evaluate(), plus the consistent
        (non-finite-difference) macro tangents dEps_hom/dE and dSig_hom/dE,
        each (3,3). Does NOT call evaluate() -- that method discards the
        RVE's own CC/B buffers via the cheaper ComputeStrainStressOnly, so
        this replicates its setup with ComputeLocalArrays instead, which
        retains them. evaluate() itself is left completely untouched, so
        nothing already verified (~1e-7 rel. L2 against the saved reference)
        is at risk of being disturbed by this addition.

        Chain rule (derived by hand; each piece verified in isolation against
        finite differences on the real trained models before being wired in
        here -- see deformation_gradient_jacobian_claude.py and
        maw_hom_weight_jacobian_claude.py):

            hom_j(E) = (1/den) * sum_{e in nz} A_e * w_e(E) * mean_gp(value_gp,e)(E)

            d(hom_j)/dE = (1/den) * sum_{e in nz} A_e *
                          [ dw_e/dE * mean_e,j + w_e * d(mean_e,j)/dE ]

        d(mean_e,j)/dE is the Gauss-point mean of CC_gp @ B_gp @ (du_local/dE),
        reusing the RVE's own already-assembled analytic microscale tangent
        (CC = dS_gp/dE_gp) and B-matrix (B = dE_gp/du_local) exactly as they
        already drive the RVE's own Newton solve elsewhere in this project.
        du_local/dE is the affine-lifting derivative (dF/dE, via
        deformation_gradient_and_jacobian_2d) plus the fluctuation-field
        derivative (the decoder's own Jacobian at the current q_p, via torch
        autograd -- mirroring how __init__ already differentiates the same
        decoder at q_p=0 for J0_const -- composed with the constant dq_p/dE
        implied by the mu-affine initializer)."""
        E = np.asarray(E, dtype=float).reshape(-1)

        u_aff_free = self._affine(E, self.x_free, self.y_free, self.is_x_free)
        disp = np.zeros(self.n_total_dof, dtype=float)
        disp[self.dir_dofs] = self._affine(E, self.x_dir, self.y_dir, self.is_x_dir)

        mu_dim = int(self.qp_aff["mu_dim"])
        b_aff = np.asarray(self.qp_aff["b_aff"], dtype=float)
        mu = E[:mu_dim]
        q_p = np.concatenate([mu, [1.0]]) @ b_aff
        q_p_torch = torch.from_numpy(q_p.astype(np.float32)).reshape(1, -1).to(self.device)

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

        w_eps, dw_eps_dE = self._hom_weights_and_jacobian(self.maw_eps_hom, E)
        w_sig, dw_sig_dE = self._hom_weights_and_jacobian(self.maw_sig_hom, E)
        hom_eps, hom_sig = CalculateHomogenizedFromAssemblerWithElementWeights(
            self.vec_assembler, w_eps=w_eps, w_sig=w_sig, reference_measure=self.hom_reference_measure,
        )

        # ---- du_local/dE: affine lifting (dF/dE) + fluctuation (decoder Jacobian) ----
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

        with torch.enable_grad():
            q_in = q_p_torch.reshape(-1).clone().detach().requires_grad_(True)

            def ann_from_qvec(qvec):
                return self.ann_model(qvec.view(1, -1)).reshape(-1)

            J_dec_torch = torch.autograd.functional.jacobian(ann_from_qvec, q_in).reshape(
                self.n_secondary, self.n_primary
            ).detach()
        J_dec = J_dec_torch.cpu().numpy()
        J0_const_np = self.J0_const_torch.cpu().numpy()
        Vs_np = self.Vs.cpu().numpy()

        d_ufluc_dqp = self.phi_p_eff_np + Vs_np @ (J_dec - J0_const_np)  # (n_free_dofs, n_primary)
        d_ufluc_dE = d_ufluc_dqp @ dqp_dE                                 # (n_free_dofs, 3)

        ddisp_dE = du_aff_dE
        ddisp_dE[self.free_dofs, :] += d_ufluc_dE

        # ---- microscale strain/stress sensitivity, per-element Gauss-mean ----
        local_eq_ids = self.vec_assembler.local_eq_ids
        du_local_dE = ddisp_dE[local_eq_ids, :]                # (n_elem, n_local_dof, 3)
        B = self.vec_assembler._B                               # (n_elem, n_gauss, 3, n_local_dof)
        CC = self.vec_assembler._CC                             # (n_elem, n_gauss, 3, 3)

        dE_gp_dE = np.einsum("egvc,ecx->egvx", B, du_local_dE)
        dS_gp_dE = np.einsum("egvw,egwx->egvx", CC, dE_gp_dE)
        dEpsbar_e_dE = np.mean(dE_gp_dE, axis=1)                 # (n_elem,3,3)
        dSbar_e_dE = np.mean(dS_gp_dE, axis=1)                   # (n_elem,3,3)

        eps_mean_e = np.mean(self.vec_assembler._E_voigt, axis=1)  # (n_elem,3)
        sig_mean_e = np.mean(self.vec_assembler._S_voigt, axis=1)  # (n_elem,3)

        if hasattr(self.vec_assembler, "area_e"):
            area_e = np.asarray(self.vec_assembler.area_e, dtype=float).reshape(-1)
        else:
            area_e = np.sum(np.asarray(self.vec_assembler.w_detJ, dtype=float), axis=1)
        den = float(self.hom_reference_measure)

        dEps_hom_dE = self._dhom_dE(eps_mean_e, dEpsbar_e_dE, w_eps, dw_eps_dE, area_e, den)
        dSig_hom_dE = self._dhom_dE(sig_mean_e, dSbar_e_dE, w_sig, dw_sig_dE, area_e, den)

        return hom_eps, hom_sig, dEps_hom_dE, dSig_hom_dE


_DEFAULT_LAW = None


def get_law():
    """Lazily builds the one, process-wide DHpromAnnDirectLaw instance --
    same caching pattern as Cook.gid/pann_constitutive_law_claude.py's
    get_law(which), so the expensive one-time setup (Kratos mesh, MAW
    models) happens exactly once regardless of how many Gauss points
    call the module function below."""
    global _DEFAULT_LAW
    if _DEFAULT_LAW is None:
        _DEFAULT_LAW = DHpromAnnDirectLaw()
    return _DEFAULT_LAW


def dhprom_ann_pk2_2d_vectorized(E_flat, young=None, poisson=None, h=1.0e-4):
    """Matches the (E_flat (N,3), young, poisson) -> (S (N,3), CC (N,3,3))
    signature Cook's VectorizedAssembler expects (young/poisson accepted
    but unused, kept only for monkeypatch signature compatibility with
    core/fom_solver_rve.py's _neo_hookean_pk2_2d_vectorized). Stress via
    DHpromAnnDirectLaw.evaluate; tangent via central finite differences
    on the 3 independent strain components (Stage 1 decision: no
    autodiff tangent available off the shelf for this path -- see
    /home/kratos/.claude/plans/jazzy-meandering-pebble.md)."""
    law = get_law()
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


def dhprom_ann_pk2_2d_vectorized_consistent(E_flat, young=None, poisson=None):
    """Same (E_flat (N,3), young, poisson) -> (S (N,3), CC (N,3,3)) contract as
    dhprom_ann_pk2_2d_vectorized (young/poisson accepted but unused, kept only
    for monkeypatch signature compatibility), but CC is the consistent
    analytic tangent from DHpromAnnDirectLaw.evaluate_with_tangent instead of
    a central finite difference."""
    law = get_law()
    E_flat = np.asarray(E_flat, dtype=float)
    n = E_flat.shape[0]
    S = np.zeros((n, 3), dtype=float)
    CC = np.zeros((n, 3, 3), dtype=float)
    for i in range(n):
        _, S[i], _, CC[i] = law.evaluate_with_tangent(E_flat[i])
    return S, CC


if __name__ == "__main__":
    print("[dhprom_ann_direct_law] building law instance ...")
    law = DHpromAnnDirectLaw()
    eps0, sig0 = law.evaluate(np.zeros(3))
    print(f"[dhprom_ann_direct_law] evaluate(0,0,0) -> eps={eps0}, sig={sig0}")
