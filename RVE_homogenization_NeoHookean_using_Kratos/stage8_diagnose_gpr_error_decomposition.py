#!/usr/bin/env python3

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style_utils import apply_latex_plot_style
from prom_gpr_solver_rve import LoadPromGprModel
from sparse_gp_manifold_model import evaluate_sparse_gp_map_and_jacobian_qp
from stage7a_prepare_rbf_dataset_ls import (
    _build_affine_lifting_helpers,
    _compute_affine_free_displacement,
)


def _relative_error(value, reference):
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return float(
        np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1.0e-30)
    )


def _component_relative_errors(value, reference):
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if value.shape != reference.shape or value.ndim != 2:
        raise ValueError("Component errors require matching two-dimensional arrays.")
    return np.array(
        [
            np.linalg.norm(value[:, j] - reference[:, j])
            / max(np.linalg.norm(reference[:, j]), 1.0e-30)
            for j in range(reference.shape[1])
        ],
        dtype=float,
    )


def _evaluate_gpr_batch(q_m, gpr_model):
    q_m = np.asarray(q_m, dtype=float)
    n_primary = int(q_m.shape[1])
    q_s = np.empty((q_m.shape[0], int(gpr_model["output_dim"])), dtype=float)
    for i, q_i in enumerate(q_m):
        q_s[i], _ = evaluate_sparse_gp_map_and_jacobian_qp(
            q_i, gpr_model, n_primary
        )
    return q_s


def _save_error_history_plot(path, histories):
    plt.figure(figsize=(8, 5))
    for label, values in histories:
        plt.semilogy(
            np.maximum(np.asarray(values, dtype=float), 1.0e-16),
            linewidth=1.3,
            label=label,
        )
    plt.xlabel("Step")
    plt.ylabel("Absolute error / global RMS reference")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_qm_plot(path, q_ref, q_mu):
    n_primary = q_ref.shape[1]
    fig, axes = plt.subplots(n_primary, 1, figsize=(9, 2.8 * n_primary), sharex=True)
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        ax.plot(q_ref[:, j], "k-", linewidth=1.5, label="FOM projection")
        ax.plot(q_mu[:, j], "r--", linewidth=1.2, label=r"$\mu\rightarrow q_m$")
        ax.set_ylabel(rf"$q_{{{j + 1}}}^m$")
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_diagnostics(
    stage8_dir,
    basis_dir,
    gpr_data_dir,
    out_dir,
    mesh_name,
):
    os.makedirs(out_dir, exist_ok=True)

    u_path = os.path.join(stage8_dir, "single_run_U.npy")
    strain_path = os.path.join(stage8_dir, "single_run_applied_strain.npy")
    if not os.path.exists(u_path) or not os.path.exists(strain_path):
        raise FileNotFoundError(
            "Stage 8 FOM state files are required: single_run_U.npy and "
            "single_run_applied_strain.npy. Run Stage 8 with FOM comparison first."
        )

    U = np.asarray(np.load(u_path), dtype=float)
    mu = np.asarray(np.load(strain_path), dtype=float)
    phi = np.asarray(
        np.load(os.path.join(basis_dir, "pod_basis_free.npy")), dtype=float
    )
    free_dofs = np.asarray(
        np.load(os.path.join(basis_dir, "free_dofs.npy")), dtype=np.int64
    )
    dir_dofs = np.asarray(
        np.load(os.path.join(basis_dir, "dirichlet_dofs.npy")), dtype=np.int64
    )

    phi_m, phi_s, free_model, _, _, gpr_model, _ = LoadPromGprModel(
        basis_dir=basis_dir,
        gpr_data_dir=gpr_data_dir,
    )
    if not np.array_equal(free_dofs, np.asarray(free_model, dtype=np.int64)):
        raise RuntimeError("Stage 2 and GPR free-DOF arrays do not match.")

    A_m = np.asarray(gpr_model["A_m"], dtype=float)
    initializer = gpr_model.get("qp_init_mu_affine")
    if initializer is None:
        raise FileNotFoundError(
            f"Missing qm_init_mu_affine.npz in {gpr_data_dir}."
        )

    n = min(U.shape[0], mu.shape[0])
    U = U[:n]
    mu = mu[:n]
    n_total_dofs = int(free_dofs.size + dir_dofs.size)
    if U.shape[1] != n_total_dofs:
        raise RuntimeError(
            f"FOM state width {U.shape[1]} does not match expected DOFs {n_total_dofs}."
        )

    (
        n_total_runtime,
        dof_x,
        dof_y,
        is_x_dof,
        _,
    ) = _build_affine_lifting_helpers(mesh_name)
    if n_total_runtime != n_total_dofs:
        raise RuntimeError(
            f"Runtime mesh has {n_total_runtime} DOFs, expected {n_total_dofs}."
        )

    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]
    W = np.empty((n, free_dofs.size), dtype=float)
    for i in range(n):
        u_aff = _compute_affine_free_displacement(
            mu[i], x_free, y_free, is_x_free
        )
        W[i] = U[i, free_dofs] - u_aff

    # Orthogonal projection onto the retained POD space.
    q_full_ref = W @ phi
    W_pod = q_full_ref @ phi.T

    # Exact coordinates of the projected FOM state in the LS decoder:
    # W_pod = Phi_m A_m q_m + Phi_s q_s.
    q_master_ref = W @ phi_m
    try:
        q_m_ref = np.linalg.solve(A_m, q_master_ref.T).T
    except np.linalg.LinAlgError:
        q_m_ref = np.linalg.lstsq(A_m, q_master_ref.T, rcond=None)[0].T
    q_s_ref = W @ phi_s

    decoder_ref = (q_m_ref @ A_m.T) @ phi_m.T + q_s_ref @ phi_s.T
    decoder_consistency = _relative_error(decoder_ref, W_pod)

    # Error source 1: physical parameters -> master coordinates.
    b_aff = np.asarray(initializer["b_aff"], dtype=float)
    mu_dim = int(initializer["mu_dim"])
    if mu.shape[1] < mu_dim:
        raise RuntimeError(
            f"Initializer expects {mu_dim} parameters, got {mu.shape[1]}."
        )
    x_aug = np.hstack([mu[:, :mu_dim], np.ones((n, 1), dtype=float)])
    q_m_mu = x_aug @ b_aff

    # Error source 2: GPR slave prediction using exact projected FOM q_m.
    q_s_gpr_oracle = _evaluate_gpr_batch(q_m_ref, gpr_model)

    # Combined direct prediction used by Stage 8 without Newton.
    q_s_gpr_direct = _evaluate_gpr_batch(q_m_mu, gpr_model)

    W_qm_only = (q_m_mu @ A_m.T) @ phi_m.T + q_s_ref @ phi_s.T
    W_gpr_only = (q_m_ref @ A_m.T) @ phi_m.T + q_s_gpr_oracle @ phi_s.T
    W_direct = (q_m_mu @ A_m.T) @ phi_m.T + q_s_gpr_direct @ phi_s.T

    metrics = {
        "pod_projection_vs_fom": _relative_error(W_pod, W),
        "ls_decoder_consistency": decoder_consistency,
        "mu_to_qm": _relative_error(q_m_mu, q_m_ref),
        "gpr_qs_oracle_qm": _relative_error(q_s_gpr_oracle, q_s_ref),
        "combined_qs_direct": _relative_error(q_s_gpr_direct, q_s_ref),
        "field_qm_only_vs_pod": _relative_error(W_qm_only, W_pod),
        "field_gpr_only_vs_pod": _relative_error(W_gpr_only, W_pod),
        "field_direct_vs_pod": _relative_error(W_direct, W_pod),
        "field_direct_vs_fom": _relative_error(W_direct, W),
    }
    q_m_components = _component_relative_errors(q_m_mu, q_m_ref)
    q_s_oracle_components = _component_relative_errors(q_s_gpr_oracle, q_s_ref)
    q_s_direct_components = _component_relative_errors(q_s_gpr_direct, q_s_ref)

    summary_path = os.path.join(out_dir, "gpr_error_decomposition.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 8 PROM-GPR error decomposition\n")
        f.write(f"stage8_dir={stage8_dir}\n")
        f.write(f"basis_dir={basis_dir}\n")
        f.write(f"gpr_data_dir={gpr_data_dir}\n")
        f.write(f"n_steps={n}\n")
        f.write(f"n_pod={phi.shape[1]}\n")
        f.write(f"n_master={q_m_ref.shape[1]}\n")
        f.write(f"n_slave={q_s_ref.shape[1]}\n")
        for key, value in metrics.items():
            f.write(f"{key}={value:.16e}\n")
        f.write(
            "mu_to_qm_component_rel="
            + ",".join(f"{v:.16e}" for v in q_m_components)
            + "\n"
        )
        f.write(
            "gpr_qs_oracle_component_rel="
            + ",".join(f"{v:.16e}" for v in q_s_oracle_components)
            + "\n"
        )
        f.write(
            "combined_qs_direct_component_rel="
            + ",".join(f"{v:.16e}" for v in q_s_direct_components)
            + "\n"
        )

    np.savez(
        os.path.join(out_dir, "gpr_error_decomposition.npz"),
        applied_strain=mu,
        q_m_fom_projection=q_m_ref,
        q_m_from_mu=q_m_mu,
        q_s_fom_projection=q_s_ref,
        q_s_gpr_from_fom_qm=q_s_gpr_oracle,
        q_s_gpr_from_mu_qm=q_s_gpr_direct,
        q_m_component_rel=q_m_components,
        q_s_oracle_component_rel=q_s_oracle_components,
        q_s_direct_component_rel=q_s_direct_components,
        **{key: np.array([value], dtype=float) for key, value in metrics.items()},
    )

    rms_qm = max(
        float(np.linalg.norm(q_m_ref) / np.sqrt(max(q_m_ref.size, 1))), 1.0e-30
    )
    rms_qs = max(
        float(np.linalg.norm(q_s_ref) / np.sqrt(max(q_s_ref.size, 1))), 1.0e-30
    )
    rms_w = max(float(np.linalg.norm(W_pod) / np.sqrt(max(W_pod.size, 1))), 1.0e-30)
    _save_error_history_plot(
        os.path.join(out_dir, "gpr_error_decomposition_history.png"),
        [
            (
                r"$\mu\rightarrow q_m$",
                np.linalg.norm(q_m_mu - q_m_ref, axis=1) / rms_qm,
            ),
            (
                r"GPR with projected $q_m$",
                np.linalg.norm(q_s_gpr_oracle - q_s_ref, axis=1) / rms_qs,
            ),
            (
                "direct combined field",
                np.linalg.norm(W_direct - W_pod, axis=1) / rms_w,
            ),
        ],
    )
    _save_qm_plot(
        os.path.join(out_dir, "mu_to_qm_vs_fom_projection.png"),
        q_m_ref,
        q_m_mu,
    )

    print("=" * 68)
    print("Stage 8 PROM-GPR error decomposition")
    print("=" * 68)
    print(f"POD projection floor, W_POD vs W_FOM       : {metrics['pod_projection_vs_fom']:.4e}")
    print(f"LS decoder consistency                     : {metrics['ls_decoder_consistency']:.4e}")
    print(f"mu -> q_m error                             : {metrics['mu_to_qm']:.4e}")
    print(f"GPR q_s error with exact projected q_m      : {metrics['gpr_qs_oracle_qm']:.4e}")
    print(f"Combined q_s error using q_m(mu)             : {metrics['combined_qs_direct']:.4e}")
    print(f"Field error caused only by q_m               : {metrics['field_qm_only_vs_pod']:.4e}")
    print(f"Field error caused only by GPR (oracle q_m)  : {metrics['field_gpr_only_vs_pod']:.4e}")
    print(f"Combined direct field error vs POD projection: {metrics['field_direct_vs_pod']:.4e}")
    print(f"Combined direct field error vs full FOM      : {metrics['field_direct_vs_fom']:.4e}")
    print("mu -> q_m component errors                  : " + ", ".join(
        f"q_{j + 1}={value:.4e}" for j, value in enumerate(q_m_components)
    ))
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Separate Stage 8 mu-to-q_m and GPR q_m-to-q_s errors."
    )
    parser.add_argument(
        "--stage8-dir",
        default="stage_8_prom_gpr_ls_results",
        help="Stage 8 directory containing single_run_U.npy and applied strain.",
    )
    parser.add_argument("--basis-dir", default="stage_2_pod_rve")
    parser.add_argument("--gpr-data-dir", default="stage_7_gpr_data_ls")
    parser.add_argument("--mesh", default="rve_geometry")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Diagnostic output directory. Defaults to <stage8-dir>/gpr_error_decomposition.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(
        args.stage8_dir, "gpr_error_decomposition"
    )
    run_diagnostics(
        stage8_dir=args.stage8_dir,
        basis_dir=args.basis_dir,
        gpr_data_dir=args.gpr_data_dir,
        out_dir=out_dir,
        mesh_name=args.mesh,
    )


if __name__ == "__main__":
    apply_latex_plot_style()
    main()
