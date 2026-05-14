#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()

from stage7a_prepare_ann_rbf_dataset import (
    _build_affine_lifting_helpers,
    _compute_affine_free_displacement,
)
from rbf_manifold_model import evaluate_rbf_map_and_jacobian_qp


def _load_training_snapshot_pair(fom_dir, trajectory_index, n_total_dofs, free_dofs):
    traj_dir = os.path.join(fom_dir, f"trajectory_{int(trajectory_index)}")
    u_file = os.path.join(traj_dir, f"trajectory_{int(trajectory_index)}_U.npy")
    e_file = os.path.join(traj_dir, f"trajectory_{int(trajectory_index)}_applied_strain.npy")
    if not os.path.exists(u_file):
        raise FileNotFoundError(f"Missing displacement snapshot file: {u_file}")
    if not os.path.exists(e_file):
        raise FileNotFoundError(f"Missing applied strain file: {e_file}")

    U = np.load(u_file)
    E_hist = np.load(e_file)
    if U.ndim != 2:
        raise ValueError(f"Invalid U shape {U.shape}; expected 2D array.")
    if E_hist.ndim != 2 or E_hist.shape[1] != 3:
        raise ValueError(f"Invalid applied strain shape {E_hist.shape}; expected (n_steps, 3).")

    if U.shape[1] == n_total_dofs:
        U_free = U[:, free_dofs]
    elif U.shape[0] == n_total_dofs:
        U_free = U[free_dofs, :].T
    else:
        raise ValueError(
            f"Cannot infer U layout from shape {U.shape}. Expected one axis to match n_total_dofs={n_total_dofs}."
        )

    n_steps = min(U_free.shape[0], E_hist.shape[0])
    if n_steps <= 0:
        raise RuntimeError("No time steps available in selected trajectory.")
    return U_free[:n_steps], E_hist[:n_steps], traj_dir


def _load_manifold_predictor(model_type, basis_dir, ann_data_dir, rbf_data_dir):
    model_type = str(model_type).strip().lower()
    if model_type == "rbf":
        from prom_rbf_solver_rve import LoadPromRbfModel

        phi_p, phi_s, free_dofs, _, _, rbf_model, include_macro = LoadPromRbfModel(
            basis_dir=basis_dir, rbf_data_dir=rbf_data_dir
        )
        n_p = int(phi_p.shape[1])
        q0_const, j0_const = evaluate_rbf_map_and_jacobian_qp(
            np.zeros(n_p, dtype=float), rbf_model, n_p
        )

        def predict_qs(qp_vec, e_vec):
            qp = np.asarray(qp_vec, dtype=float).reshape(-1)
            q_map, _ = evaluate_rbf_map_and_jacobian_qp(qp, rbf_model, n_p)
            return np.asarray(q_map - q0_const - j0_const @ qp, dtype=float).reshape(-1)

        model_label = "PROM-RBF"
        return phi_p, phi_s, free_dofs, include_macro, predict_qs, model_label, q0_const, j0_const

    if model_type == "ann":
        import torch
        from prom_ann_solver_rve import LoadPromAnnModel

        phi_p, phi_s, free_dofs, _, _, ann_model, device, include_macro = LoadPromAnnModel(
            basis_dir=basis_dir, ann_data_dir=ann_data_dir
        )
        n_p = int(phi_p.shape[1])

        def _eval_ann_raw_and_jac(qp_vec, e_vec):
            qp = np.asarray(qp_vec, dtype=np.float32).reshape(-1)

            q_in = torch.from_numpy(qp).unsqueeze(0).to(device)
            with torch.enable_grad():
                q_var = q_in.clone().detach().requires_grad_(True)

                def _ann_wrap(q_loc):
                    return ann_model(q_loc)

                q_map = _ann_wrap(q_var)
                jac = torch.autograd.functional.jacobian(_ann_wrap, q_var).reshape(-1, n_p)
            return (
                q_map.detach().cpu().numpy().reshape(-1).astype(float),
                jac.detach().cpu().numpy().astype(float),
            )

        q0_const, j0_const = _eval_ann_raw_and_jac(np.zeros(n_p, dtype=np.float32), np.zeros(3, dtype=np.float32))

        def predict_qs(qp_vec, e_vec):
            qp = np.asarray(qp_vec, dtype=np.float32).reshape(-1)
            q_map, _ = _eval_ann_raw_and_jac(qp.astype(np.float32), np.zeros(3, dtype=np.float32))
            return np.asarray(q_map - q0_const - j0_const @ qp, dtype=float).reshape(-1)

        model_label = "PROM-ANN"
        return phi_p, phi_s, free_dofs, include_macro, predict_qs, model_label, q0_const, j0_const

    raise ValueError(f"Unsupported model type '{model_type}'. Use 'rbf' or 'ann'.")


def _load_pod_dl_reconstructor(basis_dir, pod_dl_data_dir):
    import torch
    from pod_dl_manifold_model import load_pod_dl_model

    basis = np.asarray(np.load(os.path.join(basis_dir, "pod_basis_free.npy")), dtype=np.float64)
    free_dofs = np.asarray(np.load(os.path.join(basis_dir, "free_dofs.npy")), dtype=np.int64)

    model, checkpoint, device = load_pod_dl_model(model_dir=pod_dl_data_dir)
    q_dim = int(checkpoint["q_dim"])
    if q_dim < 1 or basis.shape[1] < q_dim:
        raise ValueError(
            f"POD-DL q_dim={q_dim} is invalid for basis with {basis.shape[1]} columns."
        )
    phi_q = basis[:, :q_dim].copy()

    with torch.no_grad():
        q0 = torch.zeros((1, q_dim), dtype=torch.float32, device=device)
        z0 = model.encode(q0)
        q_ref = model.decode_from_latent(z0).cpu().numpy().reshape(-1)

    def reconstruct_q(q_vec):
        q = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != q_dim:
            raise ValueError(f"Input q has {q.shape[1]} entries, expected {q_dim}.")
        with torch.no_grad():
            q_hat_raw = model(torch.from_numpy(q).to(device)).cpu().numpy().reshape(-1)
        # Match online origin anchoring used in PROM-POD-DL solver.
        return np.asarray(q_hat_raw - q_ref, dtype=float).reshape(-1)

    return phi_q, free_dofs, q_dim, reconstruct_q


def run_stage7c_reconstruction_check_pod_dl(
    trajectory_index=1,
    fom_dir="stage_1_training_set_fom",
    basis_dir="stage_2_pod_rve",
    pod_dl_data_dir="stage_7_pod_dl_data",
    out_dir="stage_7c_reconstruction_results",
):
    os.makedirs(out_dir, exist_ok=True)

    phi_q, free_dofs, q_dim, reconstruct_q = _load_pod_dl_reconstructor(
        basis_dir=basis_dir, pod_dl_data_dir=pod_dl_data_dir
    )

    dir_dofs = np.load(os.path.join(basis_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(basis_dir, "eq_map.npy"))
    n_total_dofs = int(len(free_dofs) + len(dir_dofs))

    n_total_runtime, dof_x, dof_y, is_x_dof, eq_map_runtime = _build_affine_lifting_helpers()
    if n_total_runtime != n_total_dofs:
        raise RuntimeError(
            f"Runtime/model DOF mismatch: runtime={n_total_runtime}, expected={n_total_dofs}."
        )
    if eq_map_runtime.shape == eq_map.shape and not np.array_equal(eq_map_runtime, eq_map):
        raise RuntimeError("eq_map mismatch between runtime model and Stage 2 metadata.")
    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]

    U_free, E_hist, traj_dir = _load_training_snapshot_pair(
        fom_dir=fom_dir,
        trajectory_index=trajectory_index,
        n_total_dofs=n_total_dofs,
        free_dofs=free_dofs,
    )
    n_steps = U_free.shape[0]

    q_true = np.zeros((n_steps, q_dim), dtype=float)
    q_pred = np.zeros((n_steps, q_dim), dtype=float)
    w_free_hist = np.zeros_like(U_free)
    w_proj_hist = np.zeros_like(U_free)
    w_pred_hist = np.zeros_like(U_free)

    q_step_rel = np.zeros(n_steps, dtype=float)
    w_map_step_rel = np.zeros(n_steps, dtype=float)
    w_total_step_rel = np.zeros(n_steps, dtype=float)
    proj_floor_step_rel = np.zeros(n_steps, dtype=float)

    for k in range(n_steps):
        u_aff_free = _compute_affine_free_displacement(E_hist[k], x_free, y_free, is_x_free)
        w_free = U_free[k] - u_aff_free

        qk = w_free @ phi_q
        qk_hat = reconstruct_q(qk)

        w_proj = phi_q @ qk
        w_pred = phi_q @ qk_hat

        q_true[k, :] = qk
        q_pred[k, :] = qk_hat
        w_free_hist[k, :] = w_free
        w_proj_hist[k, :] = w_proj
        w_pred_hist[k, :] = w_pred

        q_step_rel[k] = np.linalg.norm(qk_hat - qk) / (np.linalg.norm(qk) + 1e-30)
        w_map_step_rel[k] = np.linalg.norm(w_pred - w_proj) / (np.linalg.norm(w_proj) + 1e-30)
        w_total_step_rel[k] = np.linalg.norm(w_pred - w_free) / (np.linalg.norm(w_free) + 1e-30)
        proj_floor_step_rel[k] = np.linalg.norm(w_proj - w_free) / (np.linalg.norm(w_free) + 1e-30)

    q_rel_l2 = np.linalg.norm(q_pred - q_true) / (np.linalg.norm(q_true) + 1e-30)
    w_map_rel_l2 = np.linalg.norm(w_pred_hist - w_proj_hist) / (np.linalg.norm(w_proj_hist) + 1e-30)
    w_total_rel_l2 = np.linalg.norm(w_pred_hist - w_free_hist) / (np.linalg.norm(w_free_hist) + 1e-30)
    proj_floor_rel_l2 = np.linalg.norm(w_proj_hist - w_free_hist) / (np.linalg.norm(w_free_hist) + 1e-30)

    case_tag = f"pod_dl_traj_{int(trajectory_index)}"
    case_dir = os.path.join(out_dir, case_tag)
    os.makedirs(case_dir, exist_ok=True)

    np.save(os.path.join(case_dir, "q_true.npy"), q_true)
    np.save(os.path.join(case_dir, "q_pred.npy"), q_pred)
    np.save(os.path.join(case_dir, "strain_history.npy"), E_hist)
    np.save(os.path.join(case_dir, "q_step_rel.npy"), q_step_rel)
    np.save(os.path.join(case_dir, "w_map_step_rel.npy"), w_map_step_rel)
    np.save(os.path.join(case_dir, "w_total_step_rel.npy"), w_total_step_rel)
    np.save(os.path.join(case_dir, "w_projection_floor_step_rel.npy"), proj_floor_step_rel)

    plt.figure(figsize=(7, 5))
    plt.plot(q_step_rel, "m-", linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel(r"Relative Error $||q^{pred}-q^{true}||/||q^{true}||$")
    plt.title("PROM-POD-DL Stage 7c: q reconstruction error history")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "q_relative_error_history.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(w_map_step_rel, "m-", linewidth=1.5, label="map-only error vs POD projection")
    plt.plot(w_total_step_rel, "r-", linewidth=1.2, label="total error vs full fluctuation")
    plt.plot(proj_floor_step_rel, "k--", linewidth=1.0, label=f"rank-{q_dim} projection floor")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.title("PROM-POD-DL Stage 7c: fluctuation reconstruction error")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "w_reconstruction_error_history.png"), dpi=180)
    plt.close()

    summary_file = os.path.join(case_dir, "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Stage 7c reconstruction check summary\n")
        f.write("model_type=pod_dl\n")
        f.write("model_label=PROM-POD-DL\n")
        f.write(f"trajectory_index={int(trajectory_index)}\n")
        f.write(f"trajectory_dir={traj_dir}\n")
        f.write(f"n_steps={n_steps}\n")
        f.write(f"q_dim={q_dim}\n")
        f.write(f"q_rel_l2={q_rel_l2:.16e}\n")
        f.write(f"w_map_rel_l2={w_map_rel_l2:.16e}\n")
        f.write(f"w_total_rel_l2={w_total_rel_l2:.16e}\n")
        f.write(f"projection_floor_rel_l2={proj_floor_rel_l2:.16e}\n")

    print("=" * 70)
    print("Stage 7c reconstruction check")
    print("=" * 70)
    print("Model: PROM-POD-DL")
    print(f"Trajectory: {trajectory_index} | steps: {n_steps}")
    print(f"q relative L2 error:            {q_rel_l2:.6e}")
    print(f"w map-only relative L2 error:   {w_map_rel_l2:.6e}")
    print(f"w total relative L2 error:      {w_total_rel_l2:.6e}")
    print(f"rank-{q_dim} projection floor:      {proj_floor_rel_l2:.6e}")
    print(f"Saved results to: {case_dir}")


def run_stage7c_reconstruction_check(
    model_type="rbf",
    trajectory_index=1,
    fom_dir="stage_1_training_set_fom",
    basis_dir="stage_2_pod_rve",
    ann_data_dir="stage_7_ann_data",
    rbf_data_dir="stage_7_rbf_data",
    pod_dl_data_dir="stage_7_pod_dl_data",
    out_dir="stage_7c_reconstruction_results",
):
    model_type = str(model_type).strip().lower()
    if model_type == "pod_dl":
        return run_stage7c_reconstruction_check_pod_dl(
            trajectory_index=trajectory_index,
            fom_dir=fom_dir,
            basis_dir=basis_dir,
            pod_dl_data_dir=pod_dl_data_dir,
            out_dir=out_dir,
        )

    os.makedirs(out_dir, exist_ok=True)

    phi_p, phi_s, free_dofs, include_macro, predict_qs, model_label, q0_const, j0_const = _load_manifold_predictor(
        model_type=model_type,
        basis_dir=basis_dir,
        ann_data_dir=ann_data_dir,
        rbf_data_dir=rbf_data_dir,
    )
    n_p = int(phi_p.shape[1])
    n_s = int(phi_s.shape[1])
    q0_const = np.asarray(q0_const, dtype=float).reshape(-1)
    j0_const = np.asarray(j0_const, dtype=float)
    phi_p_eff = phi_p + phi_s @ j0_const
    w0_const = phi_s @ q0_const
    dir_dofs = np.load(os.path.join(basis_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(basis_dir, "eq_map.npy"))
    n_total_dofs = int(len(free_dofs) + len(dir_dofs))

    n_total_runtime, dof_x, dof_y, is_x_dof, eq_map_runtime = _build_affine_lifting_helpers()
    if n_total_runtime != n_total_dofs:
        raise RuntimeError(
            f"Runtime/model DOF mismatch: runtime={n_total_runtime}, expected={n_total_dofs}."
        )
    if eq_map_runtime.shape == eq_map.shape and not np.array_equal(eq_map_runtime, eq_map):
        raise RuntimeError("eq_map mismatch between runtime model and Stage 2 metadata.")
    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]

    U_free, E_hist, traj_dir = _load_training_snapshot_pair(
        fom_dir=fom_dir,
        trajectory_index=trajectory_index,
        n_total_dofs=n_total_dofs,
        free_dofs=free_dofs,
    )
    n_steps = U_free.shape[0]

    q_p_true = np.zeros((n_steps, n_p), dtype=float)
    q_s_true = np.zeros((n_steps, n_s), dtype=float)
    q_s_pred = np.zeros((n_steps, n_s), dtype=float)

    w_free_hist = np.zeros_like(U_free)
    w_proj_hist = np.zeros_like(U_free)
    w_pred_hist = np.zeros_like(U_free)

    qs_step_rel = np.zeros(n_steps, dtype=float)
    w_map_step_rel = np.zeros(n_steps, dtype=float)
    w_total_step_rel = np.zeros(n_steps, dtype=float)
    proj_floor_step_rel = np.zeros(n_steps, dtype=float)

    for k in range(n_steps):
        u_aff_free = _compute_affine_free_displacement(E_hist[k], x_free, y_free, is_x_free)
        w_free = U_free[k] - u_aff_free

        qp = w_free @ phi_p
        qs_raw = w_free @ phi_s
        qs = qs_raw - q0_const - j0_const @ qp
        qs_hat = predict_qs(qp, E_hist[k])

        w_proj = w0_const + phi_p_eff @ qp + phi_s @ qs
        w_pred = w0_const + phi_p_eff @ qp + phi_s @ qs_hat

        q_p_true[k, :] = qp
        q_s_true[k, :] = qs
        q_s_pred[k, :] = qs_hat
        w_free_hist[k, :] = w_free
        w_proj_hist[k, :] = w_proj
        w_pred_hist[k, :] = w_pred

        qs_step_rel[k] = np.linalg.norm(qs_hat - qs) / (np.linalg.norm(qs) + 1e-30)
        w_map_step_rel[k] = np.linalg.norm(w_pred - w_proj) / (np.linalg.norm(w_proj) + 1e-30)
        w_total_step_rel[k] = np.linalg.norm(w_pred - w_free) / (np.linalg.norm(w_free) + 1e-30)
        proj_floor_step_rel[k] = np.linalg.norm(w_proj - w_free) / (np.linalg.norm(w_free) + 1e-30)

    qs_rel_l2 = np.linalg.norm(q_s_pred - q_s_true) / (np.linalg.norm(q_s_true) + 1e-30)
    w_map_rel_l2 = np.linalg.norm(w_pred_hist - w_proj_hist) / (np.linalg.norm(w_proj_hist) + 1e-30)
    w_total_rel_l2 = np.linalg.norm(w_pred_hist - w_free_hist) / (np.linalg.norm(w_free_hist) + 1e-30)
    proj_floor_rel_l2 = np.linalg.norm(w_proj_hist - w_free_hist) / (np.linalg.norm(w_free_hist) + 1e-30)

    case_tag = f"{str(model_type).lower()}_traj_{int(trajectory_index)}"
    case_dir = os.path.join(out_dir, case_tag)
    os.makedirs(case_dir, exist_ok=True)

    np.save(os.path.join(case_dir, "q_p_true.npy"), q_p_true)
    np.save(os.path.join(case_dir, "q_s_true.npy"), q_s_true)
    np.save(os.path.join(case_dir, "q_s_pred.npy"), q_s_pred)
    np.save(os.path.join(case_dir, "strain_history.npy"), E_hist)
    np.save(os.path.join(case_dir, "qs_step_rel.npy"), qs_step_rel)
    np.save(os.path.join(case_dir, "w_map_step_rel.npy"), w_map_step_rel)
    np.save(os.path.join(case_dir, "w_total_step_rel.npy"), w_total_step_rel)
    np.save(os.path.join(case_dir, "w_projection_floor_step_rel.npy"), proj_floor_step_rel)

    plt.figure(figsize=(7, 5))
    plt.plot(qs_step_rel, "b-", linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel(r"Relative Error $||q_s^{pred}-q_s^{true}||/||q_s^{true}||$")
    plt.title(f"{model_label} Stage 7c: q_s mapping error history")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "qs_relative_error_history.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(w_map_step_rel, "b-", linewidth=1.5, label="map-only error vs rank-9 projection")
    plt.plot(w_total_step_rel, "r-", linewidth=1.2, label="total error vs full fluctuation")
    plt.plot(proj_floor_step_rel, "k--", linewidth=1.0, label="rank-9 projection floor")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.title(f"{model_label} Stage 7c: fluctuation reconstruction error")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "w_reconstruction_error_history.png"), dpi=180)
    plt.close()

    n_plot = min(3, n_s)
    if n_plot > 0:
        fig, axs = plt.subplots(n_plot, 1, figsize=(8, 2.7 * n_plot), sharex=True)
        if n_plot == 1:
            axs = [axs]
        for i in range(n_plot):
            axs[i].plot(q_s_true[:, i], "k-", label=f"q_s_true[{i}]")
            axs[i].plot(q_s_pred[:, i], "b--", label=f"q_s_pred[{i}]")
            axs[i].grid(True, linestyle="--", alpha=0.5)
            axs[i].legend(loc="best")
        axs[-1].set_xlabel("Step")
        fig.suptitle(f"{model_label} Stage 7c: first {n_plot} secondary coordinates")
        fig.tight_layout()
        fig.savefig(os.path.join(case_dir, "qs_components_overlay.png"), dpi=180)
        plt.close(fig)

    summary_file = os.path.join(case_dir, "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Stage 7c reconstruction check summary\n")
        f.write(f"model_type={str(model_type).lower()}\n")
        f.write(f"model_label={model_label}\n")
        f.write(f"trajectory_index={int(trajectory_index)}\n")
        f.write(f"trajectory_dir={traj_dir}\n")
        f.write(f"n_steps={n_steps}\n")
        f.write(f"n_primary={n_p}\n")
        f.write(f"n_secondary={n_s}\n")
        f.write(f"include_macro_strain_input={int(include_macro)}\n")
        f.write(f"qs_rel_l2={qs_rel_l2:.16e}\n")
        f.write(f"w_map_rel_l2={w_map_rel_l2:.16e}\n")
        f.write(f"w_total_rel_l2={w_total_rel_l2:.16e}\n")
        f.write(f"projection_floor_rel_l2={proj_floor_rel_l2:.16e}\n")

    print("=" * 70)
    print("Stage 7c reconstruction check")
    print("=" * 70)
    print(f"Model: {model_label}")
    print(f"Trajectory: {trajectory_index} | steps: {n_steps}")
    print(f"include_macro_strain_input: {int(include_macro)}")
    print(f"q_s relative L2 error:          {qs_rel_l2:.6e}")
    print(f"w map-only relative L2 error:   {w_map_rel_l2:.6e}")
    print(f"w total relative L2 error:      {w_total_rel_l2:.6e}")
    print(f"rank-9 projection floor (L2):   {proj_floor_rel_l2:.6e}")
    print(f"Saved results to: {case_dir}")


def run_stage7c_bounds_comparison(
    trajectory_index=1,
    fom_dir="stage_1_training_set_fom",
    basis_dir="stage_2_pod_rve",
    ann_data_dir="stage_7_ann_data",
    rbf_data_dir="stage_7_rbf_data",
    pod_dl_data_dir="stage_7_pod_dl_data",
    out_dir="stage_7c_reconstruction_results",
):
    """
    Compare linear POD baselines and nonlinear manifold reconstructions on one trajectory:
      - POD rank-4 (lower bound among compared models)
      - POD rank-9 projection floor (upper bound in the rank-9 subspace)
      - PROM-ANN reconstruction
      - PROM-RBF reconstruction
      - PROM-POD-DL reconstruction
    """
    os.makedirs(out_dir, exist_ok=True)

    (
        phi_p_ann,
        phi_s_ann,
        free_dofs_ann,
        inc_ann,
        predict_qs_ann,
        _,
        q0_ann,
        j0_ann,
    ) = _load_manifold_predictor(
        model_type="ann",
        basis_dir=basis_dir,
        ann_data_dir=ann_data_dir,
        rbf_data_dir=rbf_data_dir,
    )
    (
        phi_p_rbf,
        phi_s_rbf,
        free_dofs_rbf,
        inc_rbf,
        predict_qs_rbf,
        _,
        q0_rbf,
        j0_rbf,
    ) = _load_manifold_predictor(
        model_type="rbf",
        basis_dir=basis_dir,
        ann_data_dir=ann_data_dir,
        rbf_data_dir=rbf_data_dir,
    )

    if phi_p_ann.shape != phi_p_rbf.shape or phi_s_ann.shape != phi_s_rbf.shape:
        raise RuntimeError("ANN/RBF basis partitions do not match.")
    if not np.array_equal(free_dofs_ann, free_dofs_rbf):
        raise RuntimeError("ANN/RBF free_dofs do not match.")

    phi_q_dl, free_dofs_dl, q_dim_dl, reconstruct_q_dl = _load_pod_dl_reconstructor(
        basis_dir=basis_dir, pod_dl_data_dir=pod_dl_data_dir
    )
    if not np.array_equal(free_dofs_ann, free_dofs_dl):
        raise RuntimeError("ANN/RBF and POD-DL free_dofs do not match.")

    phi_p = phi_p_ann
    phi_s = phi_s_ann
    free_dofs = free_dofs_ann
    n_p = int(phi_p.shape[1])
    n_s = int(phi_s.shape[1])
    q0_ann = np.asarray(q0_ann, dtype=float).reshape(-1)
    j0_ann = np.asarray(j0_ann, dtype=float)
    q0_rbf = np.asarray(q0_rbf, dtype=float).reshape(-1)
    j0_rbf = np.asarray(j0_rbf, dtype=float)
    phi_p_eff_ann = phi_p + phi_s @ j0_ann
    phi_p_eff_rbf = phi_p + phi_s @ j0_rbf
    w0_ann = phi_s @ q0_ann
    w0_rbf = phi_s @ q0_rbf
    n_rank9 = int(n_p + n_s)
    if int(q_dim_dl) != n_rank9:
        raise RuntimeError(
            f"POD-DL q_dim={int(q_dim_dl)} is not compatible with rank-9 bounds ({n_rank9}). "
            "Retrain Stage 7b POD-DL with --q-dim 9 (or matching n_p+n_s)."
        )

    dir_dofs = np.load(os.path.join(basis_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(basis_dir, "eq_map.npy"))
    n_total_dofs = int(len(free_dofs) + len(dir_dofs))

    n_total_runtime, dof_x, dof_y, is_x_dof, eq_map_runtime = _build_affine_lifting_helpers()
    if n_total_runtime != n_total_dofs:
        raise RuntimeError(
            f"Runtime/model DOF mismatch: runtime={n_total_runtime}, expected={n_total_dofs}."
        )
    if eq_map_runtime.shape == eq_map.shape and not np.array_equal(eq_map_runtime, eq_map):
        raise RuntimeError("eq_map mismatch between runtime model and Stage 2 metadata.")

    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]

    U_free, E_hist, traj_dir = _load_training_snapshot_pair(
        fom_dir=fom_dir,
        trajectory_index=trajectory_index,
        n_total_dofs=n_total_dofs,
        free_dofs=free_dofs,
    )
    n_steps = U_free.shape[0]

    # Histories
    q_p_true = np.zeros((n_steps, n_p), dtype=float)
    q_s_true = np.zeros((n_steps, n_s), dtype=float)
    q_s_true_rbf = np.zeros((n_steps, n_s), dtype=float)
    q_s_ann = np.zeros((n_steps, n_s), dtype=float)
    q_s_rbf = np.zeros((n_steps, n_s), dtype=float)
    q_dl_true = np.zeros((n_steps, n_rank9), dtype=float)
    q_dl_pred = np.zeros((n_steps, n_rank9), dtype=float)

    w_free_hist = np.zeros_like(U_free)
    w_pod4_hist = np.zeros_like(U_free)
    w_pod9_hist = np.zeros_like(U_free)
    w_ann_hist = np.zeros_like(U_free)
    w_rbf_hist = np.zeros_like(U_free)
    w_dl_hist = np.zeros_like(U_free)

    # Stepwise relative errors (vs full fluctuation w_free)
    err_pod4_step = np.zeros(n_steps, dtype=float)
    err_pod9_step = np.zeros(n_steps, dtype=float)
    err_ann_step = np.zeros(n_steps, dtype=float)
    err_rbf_step = np.zeros(n_steps, dtype=float)
    err_dl_step = np.zeros(n_steps, dtype=float)

    # Stepwise map-only errors (vs POD-9 reference)
    err_ann_vs_pod9_step = np.zeros(n_steps, dtype=float)
    err_rbf_vs_pod9_step = np.zeros(n_steps, dtype=float)
    err_dl_vs_pod9_step = np.zeros(n_steps, dtype=float)

    # Stepwise q_s errors
    err_qs_ann_step = np.zeros(n_steps, dtype=float)
    err_qs_rbf_step = np.zeros(n_steps, dtype=float)
    err_q_dl_step = np.zeros(n_steps, dtype=float)

    for k in range(n_steps):
        u_aff_free = _compute_affine_free_displacement(E_hist[k], x_free, y_free, is_x_free)
        w_free = U_free[k] - u_aff_free

        qp = w_free @ phi_p
        qs_raw = w_free @ phi_s
        qs_ann_true = qs_raw - q0_ann - j0_ann @ qp
        qs_rbf_true = qs_raw - q0_rbf - j0_rbf @ qp
        q9 = np.concatenate([qp, qs_raw], axis=0)

        qs_hat_ann = predict_qs_ann(qp, E_hist[k])
        qs_hat_rbf = predict_qs_rbf(qp, E_hist[k])
        q9_hat_dl = reconstruct_q_dl(q9)

        w_pod4 = phi_p @ qp
        w_pod9 = w_pod4 + phi_s @ qs_raw
        w_ann = w0_ann + phi_p_eff_ann @ qp + phi_s @ qs_hat_ann
        w_rbf = w0_rbf + phi_p_eff_rbf @ qp + phi_s @ qs_hat_rbf
        w_dl = phi_q_dl @ q9_hat_dl

        q_p_true[k, :] = qp
        q_s_true[k, :] = qs_ann_true
        q_s_true_rbf[k, :] = qs_rbf_true
        q_s_ann[k, :] = qs_hat_ann
        q_s_rbf[k, :] = qs_hat_rbf
        q_dl_true[k, :] = q9
        q_dl_pred[k, :] = q9_hat_dl

        w_free_hist[k, :] = w_free
        w_pod4_hist[k, :] = w_pod4
        w_pod9_hist[k, :] = w_pod9
        w_ann_hist[k, :] = w_ann
        w_rbf_hist[k, :] = w_rbf
        w_dl_hist[k, :] = w_dl

        denom_w = np.linalg.norm(w_free) + 1e-30
        denom_w9 = np.linalg.norm(w_pod9) + 1e-30
        denom_qs_ann = np.linalg.norm(qs_ann_true) + 1e-30
        denom_qs_rbf = np.linalg.norm(qs_rbf_true) + 1e-30
        denom_q9 = np.linalg.norm(q9) + 1e-30

        err_pod4_step[k] = np.linalg.norm(w_pod4 - w_free) / denom_w
        err_pod9_step[k] = np.linalg.norm(w_pod9 - w_free) / denom_w
        err_ann_step[k] = np.linalg.norm(w_ann - w_free) / denom_w
        err_rbf_step[k] = np.linalg.norm(w_rbf - w_free) / denom_w
        err_dl_step[k] = np.linalg.norm(w_dl - w_free) / denom_w

        err_ann_vs_pod9_step[k] = np.linalg.norm(w_ann - w_pod9) / denom_w9
        err_rbf_vs_pod9_step[k] = np.linalg.norm(w_rbf - w_pod9) / denom_w9
        err_dl_vs_pod9_step[k] = np.linalg.norm(w_dl - w_pod9) / denom_w9

        err_qs_ann_step[k] = np.linalg.norm(qs_hat_ann - qs_ann_true) / denom_qs_ann
        err_qs_rbf_step[k] = np.linalg.norm(qs_hat_rbf - qs_rbf_true) / denom_qs_rbf
        err_q_dl_step[k] = np.linalg.norm(q9_hat_dl - q9) / denom_q9

    # Global relative L2 errors
    rel_pod4 = np.linalg.norm(w_pod4_hist - w_free_hist) / (np.linalg.norm(w_free_hist) + 1e-30)
    rel_pod9 = np.linalg.norm(w_pod9_hist - w_free_hist) / (np.linalg.norm(w_free_hist) + 1e-30)
    rel_ann = np.linalg.norm(w_ann_hist - w_free_hist) / (np.linalg.norm(w_free_hist) + 1e-30)
    rel_rbf = np.linalg.norm(w_rbf_hist - w_free_hist) / (np.linalg.norm(w_free_hist) + 1e-30)
    rel_dl = np.linalg.norm(w_dl_hist - w_free_hist) / (np.linalg.norm(w_free_hist) + 1e-30)

    rel_ann_vs_pod9 = np.linalg.norm(w_ann_hist - w_pod9_hist) / (np.linalg.norm(w_pod9_hist) + 1e-30)
    rel_rbf_vs_pod9 = np.linalg.norm(w_rbf_hist - w_pod9_hist) / (np.linalg.norm(w_pod9_hist) + 1e-30)
    rel_dl_vs_pod9 = np.linalg.norm(w_dl_hist - w_pod9_hist) / (np.linalg.norm(w_pod9_hist) + 1e-30)

    rel_qs_ann = np.linalg.norm(q_s_ann - q_s_true) / (np.linalg.norm(q_s_true) + 1e-30)
    rel_qs_rbf = np.linalg.norm(q_s_rbf - q_s_true_rbf) / (np.linalg.norm(q_s_true_rbf) + 1e-30)
    rel_q_dl = np.linalg.norm(q_dl_pred - q_dl_true) / (np.linalg.norm(q_dl_true) + 1e-30)

    case_dir = os.path.join(out_dir, f"bounds_traj_{int(trajectory_index)}")
    os.makedirs(case_dir, exist_ok=True)

    np.save(os.path.join(case_dir, "q_p_true.npy"), q_p_true)
    np.save(os.path.join(case_dir, "q_s_true.npy"), q_s_true)
    np.save(os.path.join(case_dir, "q_s_true_rbf.npy"), q_s_true_rbf)
    np.save(os.path.join(case_dir, "q_s_ann_pred.npy"), q_s_ann)
    np.save(os.path.join(case_dir, "q_s_rbf_pred.npy"), q_s_rbf)
    np.save(os.path.join(case_dir, "q9_true.npy"), q_dl_true)
    np.save(os.path.join(case_dir, "q9_pod_dl_pred.npy"), q_dl_pred)
    np.save(os.path.join(case_dir, "strain_history.npy"), E_hist)
    np.save(os.path.join(case_dir, "err_pod4_step.npy"), err_pod4_step)
    np.save(os.path.join(case_dir, "err_pod9_step.npy"), err_pod9_step)
    np.save(os.path.join(case_dir, "err_ann_step.npy"), err_ann_step)
    np.save(os.path.join(case_dir, "err_rbf_step.npy"), err_rbf_step)
    np.save(os.path.join(case_dir, "err_pod_dl_step.npy"), err_dl_step)
    np.save(os.path.join(case_dir, "err_ann_vs_pod9_step.npy"), err_ann_vs_pod9_step)
    np.save(os.path.join(case_dir, "err_rbf_vs_pod9_step.npy"), err_rbf_vs_pod9_step)
    np.save(os.path.join(case_dir, "err_pod_dl_vs_pod9_step.npy"), err_dl_vs_pod9_step)
    np.save(os.path.join(case_dir, "err_qs_ann_step.npy"), err_qs_ann_step)
    np.save(os.path.join(case_dir, "err_qs_rbf_step.npy"), err_qs_rbf_step)
    np.save(os.path.join(case_dir, "err_q9_pod_dl_step.npy"), err_q_dl_step)

    # Error history plot
    plt.figure(figsize=(8, 5.5))
    plt.plot(err_pod4_step, "k-", linewidth=1.5, label="POD rank-4 vs full fluctuation")
    plt.plot(err_pod9_step, "k--", linewidth=1.5, label="POD rank-9 projection floor")
    plt.plot(err_ann_step, "b-", linewidth=1.3, label="PROM-ANN reconstruction")
    plt.plot(err_rbf_step, "r-", linewidth=1.3, label="PROM-RBF reconstruction")
    plt.plot(err_dl_step, "m-", linewidth=1.3, label="PROM-POD-DL reconstruction")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.title("Stage 7c: Reconstruction Bounds and Manifold Closures")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "bounds_error_history.png"), dpi=180)
    plt.close()

    # q_s map-only error plot
    plt.figure(figsize=(8, 5.0))
    plt.plot(err_qs_ann_step, "b-", linewidth=1.2, label="ANN q_s relative error")
    plt.plot(err_qs_rbf_step, "r-", linewidth=1.2, label="RBF q_s relative error")
    plt.plot(err_q_dl_step, "m-", linewidth=1.2, label="POD-DL q relative error (full q)")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel(r"Relative Error $||q_s^{pred}-q_s^{true}||/||q_s^{true}||$")
    plt.title("Stage 7c: Secondary Coordinates Mapping Error")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "qs_map_error_history_ann_vs_rbf.png"), dpi=180)
    plt.close()

    summary_file = os.path.join(case_dir, "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Stage 7c bounds comparison summary\n")
        f.write(f"trajectory_index={int(trajectory_index)}\n")
        f.write(f"trajectory_dir={traj_dir}\n")
        f.write(f"n_steps={n_steps}\n")
        f.write(f"n_primary={n_p}\n")
        f.write(f"n_secondary={n_s}\n")
        f.write(f"ann_include_macro_strain_input={int(inc_ann)}\n")
        f.write(f"rbf_include_macro_strain_input={int(inc_rbf)}\n")
        f.write(f"pod_dl_q_dim={int(q_dim_dl)}\n")
        f.write(f"rel_l2_pod4={rel_pod4:.16e}\n")
        f.write(f"rel_l2_pod9={rel_pod9:.16e}\n")
        f.write(f"rel_l2_ann={rel_ann:.16e}\n")
        f.write(f"rel_l2_rbf={rel_rbf:.16e}\n")
        f.write(f"rel_l2_pod_dl={rel_dl:.16e}\n")
        f.write(f"rel_l2_ann_vs_pod9={rel_ann_vs_pod9:.16e}\n")
        f.write(f"rel_l2_rbf_vs_pod9={rel_rbf_vs_pod9:.16e}\n")
        f.write(f"rel_l2_pod_dl_vs_pod9={rel_dl_vs_pod9:.16e}\n")
        f.write(f"rel_l2_qs_ann={rel_qs_ann:.16e}\n")
        f.write(f"rel_l2_qs_rbf={rel_qs_rbf:.16e}\n")
        f.write(f"rel_l2_q_pod_dl={rel_q_dl:.16e}\n")

    print("=" * 70)
    print("Stage 7c bounds comparison")
    print("=" * 70)
    print(f"Trajectory: {trajectory_index} | steps: {n_steps}")
    print(f"POD rank-4 rel L2:          {rel_pod4:.6e}")
    print(f"POD rank-9 floor rel L2:    {rel_pod9:.6e}")
    print(f"PROM-ANN rel L2:            {rel_ann:.6e}")
    print(f"PROM-RBF rel L2:            {rel_rbf:.6e}")
    print(f"PROM-POD-DL rel L2:         {rel_dl:.6e}")
    print(f"ANN vs POD-9 rel L2:        {rel_ann_vs_pod9:.6e}")
    print(f"RBF vs POD-9 rel L2:        {rel_rbf_vs_pod9:.6e}")
    print(f"POD-DL vs POD-9 rel L2:     {rel_dl_vs_pod9:.6e}")
    print(f"ANN q_s rel L2:             {rel_qs_ann:.6e}")
    print(f"RBF q_s rel L2:             {rel_qs_rbf:.6e}")
    print(f"POD-DL q rel L2:            {rel_q_dl:.6e}")
    print(f"Saved results to: {case_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 7c: pure reconstruction check on one training trajectory")
    p.add_argument(
        "--model-type",
        type=str,
        default="rbf",
        choices=["rbf", "ann", "pod_dl"],
        help="Manifold model to evaluate.",
    )
    p.add_argument("--trajectory-index", type=int, default=1, help="Training trajectory index to check.")
    p.add_argument("--fom-dir", type=str, default="stage_1_training_set_fom", help="Stage 1 directory.")
    p.add_argument("--basis-dir", type=str, default="stage_2_pod_rve", help="Stage 2 POD directory.")
    p.add_argument("--ann-data-dir", type=str, default="stage_7_ann_data", help="ANN data/model directory.")
    p.add_argument("--rbf-data-dir", type=str, default="stage_7_rbf_data", help="RBF data/model directory.")
    p.add_argument("--pod-dl-data-dir", type=str, default="stage_7_pod_dl_data", help="POD-DL data/model directory.")
    p.add_argument("--out-dir", type=str, default="stage_7c_reconstruction_results", help="Output directory.")
    args = p.parse_args()

    run_stage7c_reconstruction_check(
        model_type=args.model_type,
        trajectory_index=args.trajectory_index,
        fom_dir=args.fom_dir,
        basis_dir=args.basis_dir,
        ann_data_dir=args.ann_data_dir,
        rbf_data_dir=args.rbf_data_dir,
        pod_dl_data_dir=args.pod_dl_data_dir,
        out_dir=args.out_dir,
    )
