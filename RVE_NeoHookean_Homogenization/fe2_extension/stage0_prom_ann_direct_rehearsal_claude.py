#!/usr/bin/env python3
"""Stage 0 (throwaway): extract PROM-ANN's direct-mode (max_its=0)
per-step evaluation into a standalone, build-once/evaluate-many class,
and verify it reproduces the ORIGINAL, trusted RunPromAnnBatchSimulation
exactly. No saved direct-mode PROM-ANN reference exists anywhere in this
project (grep confirms max_its=0 is never actually invoked elsewhere),
so this script generates its own reference by running the original
function once, then checks the extracted class against it.

Purpose: de-risk the extraction PATTERN (what to factor into "build
once" vs "per query") on the simpler, non-hyper-reduced target before
Stage 1 tackles D-HPROM-ANN's added ECM/MAW complexity. Not paper-grade,
not a deliverable -- see /home/kratos/.claude/plans/jazzy-meandering-pebble.md.

Read-only: only imports from core/, prom/ann/; writes exclusively under
fe2_extension/stage0_out/. Nothing outside fe2_extension/ is modified.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("core", "prom/ann"):
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
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from stage6_test_hprom import generate_safe_test_path
from prom_ann_solver_rve import LoadPromAnnModel, RunPromAnnBatchSimulation

ANN_DATA_DIR = str(ROOT / "prom" / "ann" / "stage_7_ann_model_ls")
BASIS_DIR = str(ROOT / "pod" / "stage_2_pod_rve")
OUT_DIR = str(HERE / "stage0_out")


def build_held_out_strain_path():
    """Same construction stage8_test_prom_ann.py uses for its benchmark path."""
    emax = 2.0
    rel6 = [1.0, 0.05, 1.0, 0.05, 0.05, 0.05]
    domain_type = "box"
    bundle_path = ROOT / "trajectories" / "stage_0_trajectory" / "stage_0_trajectories.npz"
    if bundle_path.exists():
        data = np.load(bundle_path, allow_pickle=True)
        rel6 = list(data["relative_boundary"])
        emax = float(np.ravel(data["emax"])[0]) if "emax" in data else float(np.ravel(data["reference_amplitude"])[0])
        if "domain_type" in data:
            domain_type = str(data["domain_type"][0])
    _control_points, full_path = generate_safe_test_path(emax, rel6, domain_type)
    return np.array(full_path, dtype=float), emax


def densified_per_step_strains(strain_path, emax):
    """Reproduces RunPromAnnBatchSimulation's internal per-step E exactly
    (same BuildDynamicSegmentSteps call + linear interpolation, lines
    ~478-481 of prom_ann_solver_rve.py), without modifying that file."""
    E_wp = np.asarray(strain_path, dtype=float)
    n_seg = len(E_wp) - 1
    seg_steps, _ = BuildDynamicSegmentSteps(
        E_wp, reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT, reference_amplitude=emax,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    E_per_step = np.zeros((n_steps_total, 3), dtype=float)
    for step in range(1, n_steps_total + 1):
        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        E_per_step[step - 1] = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]
    return E_per_step


def generate_reference(strain_path, emax):
    """Run the ORIGINAL, trusted RunPromAnnBatchSimulation in direct mode."""
    os.makedirs(OUT_DIR, exist_ok=True)
    phi_p, phi_s, free_dofs, _dir_dofs, _eq_map, ann_model, device, _ = LoadPromAnnModel(
        basis_dir=BASIS_DIR, ann_data_dir=ANN_DATA_DIR,
    )
    parameters = setup_kratos_parameters("rve_geometry")
    eps_ref, sig_ref = RunPromAnnBatchSimulation(
        parameters, phi_p, phi_s, free_dofs, ann_model, device,
        strain_path, out_dir=OUT_DIR,
        reference_amplitude=emax, reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        qp_init_mode="mu_affine", max_its=0, fail_on_nonconvergence=False,
    )
    np.save(os.path.join(OUT_DIR, "reference_sig.npy"), sig_ref)
    np.save(os.path.join(OUT_DIR, "reference_eps.npy"), eps_ref)
    return eps_ref, sig_ref


class PromAnnDirectLaw:
    """Build-once / evaluate-many extraction of PROM-ANN's direct
    (max_its=0) per-step block (prom_ann_solver_rve.py lines ~1569-1682-
    equivalent for PROM-ANN, i.e. lines ~814-844 of that file)."""

    def __init__(self, basis_dir=BASIS_DIR, ann_data_dir=ANN_DATA_DIR, mesh="rve_geometry"):
        phi_p, phi_s, free_dofs, _dir_dofs, _eq_map, ann_model, device, _ = LoadPromAnnModel(
            basis_dir=basis_dir, ann_data_dir=ann_data_dir,
        )
        self.ann_model, self.device = ann_model, device
        self.phi_p, self.phi_s = np.asarray(phi_p, dtype=float), np.asarray(phi_s, dtype=float)
        self.free_dofs = np.asarray(free_dofs, dtype=np.int64)
        self.n_primary, self.n_secondary = self.phi_p.shape[1], self.phi_s.shape[1]

        parameters = setup_kratos_parameters(mesh)
        model_kratos = KM.Model()
        sim = RVEHomogenizationDatasetGenerator(model_kratos, parameters)
        sim.Initialize()
        mp = sim._GetSolver().GetComputingModelPart()
        n_total_dof, eq_id_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
        self.mp, self.eq_id_map, self.ta, self.n_total_dof = mp, eq_id_map, ta, n_total_dof
        self.vec_assembler = VectorizedAssembler(mp, n_total_dof, eq_id_map)
        self.hom_reference_measure = float(np.sum(np.asarray(self.vec_assembler.area_e, dtype=float)))

        sim._InitializeDomainCenterIfNeeded(mp)
        x0c, y0c = float(sim._x0c), float(sim._y0c)
        dof_x = np.zeros(n_total_dof); dof_y = np.zeros(n_total_dof)
        is_x_dof = np.zeros(n_total_dof, dtype=bool)
        for i, node in enumerate(mp.Nodes):
            xr, yr = float(node.X0) - x0c, float(node.Y0) - y0c
            idx_x, idx_y = int(eq_id_map[i, 0]), int(eq_id_map[i, 1])
            if 0 <= idx_x < n_total_dof:
                dof_x[idx_x], dof_y[idx_x], is_x_dof[idx_x] = xr, yr, True
            if 0 <= idx_y < n_total_dof:
                dof_x[idx_y], dof_y[idx_y], is_x_dof[idx_y] = xr, yr, False
        self.x_free, self.y_free, self.is_x_free = dof_x[self.free_dofs], dof_y[self.free_dofs], is_x_dof[self.free_dofs]
        free_mask = np.zeros(n_total_dof, dtype=bool)
        free_mask[self.free_dofs] = True
        self.dir_dofs = np.nonzero(~free_mask)[0].astype(np.int64)
        self.x_dir, self.y_dir, self.is_x_dir = dof_x[self.dir_dofs], dof_y[self.dir_dofs], is_x_dof[self.dir_dofs]

        self.Vs = torch.from_numpy(self.phi_s.astype(np.float32)).to(device)
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
        phi_master = self.phi_p @ a_m
        self.phi_p_eff_np = phi_master + self.phi_s @ J0_const_np
        self.w0_const_np = self.phi_s @ N0_const_np
        self.Vp_eff = torch.from_numpy(self.phi_p_eff_np.astype(np.float32)).to(device)
        self.w0_const_t = torch.from_numpy(self.w0_const_np.astype(np.float32)).to(device).unsqueeze(0)
        self.N0_const, self.J0_const_torch = N0_const, J0_const_torch

    @staticmethod
    def _affine(e_vec, x_loc, y_loc, is_x_loc):
        F = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (F[0, 0] - 1.0) * x_loc + F[0, 1] * y_loc
        uy = F[1, 0] * x_loc + (F[1, 1] - 1.0) * y_loc
        return np.where(is_x_loc, ux, uy)

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
            q_s_final_raw = self.ann_model(q_p_torch)
            q_s_final = q_s_final_raw - self.N0_const - (q_p_torch @ self.J0_const_torch.T)
            u_fluc_final = (
                self.w0_const_t + q_p_torch @ self.Vp_eff.T + q_s_final @ self.Vs.T
            ).detach().cpu().numpy().reshape(-1)

        disp[self.free_dofs] = u_aff_free + u_fluc_final
        SetDisplacementFromEquationVector(disp, self.eq_id_map, self.ta)
        UpdateCurrentCoordinatesFromDisplacement(self.mp, step=0)
        self.vec_assembler.ComputeStrainStressOnly(disp)
        eps_h, sig_h = CalculateHomogenizedFromAssemblerWithElementWeights(
            self.vec_assembler, reference_measure=self.hom_reference_measure,
        )
        return eps_h, sig_h


def relative_l2(pred, ref):
    pred, ref = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-30))


def main():
    strain_path, emax = build_held_out_strain_path()
    print(f"[stage0] held-out path: {len(strain_path)} waypoints, emax={emax}")

    print("[stage0] generating reference via the ORIGINAL RunPromAnnBatchSimulation ...")
    eps_ref, sig_ref = generate_reference(strain_path, emax)
    print(f"[stage0] reference shape: eps={eps_ref.shape}, sig={sig_ref.shape}")

    print("[stage0] building extracted PromAnnDirectLaw ...")
    law = PromAnnDirectLaw()

    E_per_step = densified_per_step_strains(strain_path, emax)
    assert E_per_step.shape[0] == eps_ref.shape[0] - 1, (
        f"step count mismatch: {E_per_step.shape[0]} vs {eps_ref.shape[0] - 1}"
    )

    print(f"[stage0] evaluating extracted law at {E_per_step.shape[0]} independent strains ...")
    eps_ext = np.zeros_like(E_per_step)
    sig_ext = np.zeros_like(E_per_step)
    for i, E in enumerate(E_per_step):
        eps_ext[i], sig_ext[i] = law.evaluate(E)

    rel_eps = relative_l2(eps_ext, eps_ref[1:])
    rel_sig = relative_l2(sig_ext, sig_ref[1:])
    print(f"[stage0] relative L2 (extracted vs original): eps={rel_eps:.3e}, sig={rel_sig:.3e}")

    np.save(os.path.join(OUT_DIR, "extracted_eps.npy"), eps_ext)
    np.save(os.path.join(OUT_DIR, "extracted_sig.npy"), sig_ext)

    tol = 1e-6
    if rel_eps < tol and rel_sig < tol:
        print(f"[stage0] PASS: extraction pattern reproduces the original to within {tol:.0e}.")
    else:
        print(f"[stage0] FAIL: mismatch exceeds {tol:.0e} -- inspect before trusting the pattern.")


if __name__ == "__main__":
    main()
