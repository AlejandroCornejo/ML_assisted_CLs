#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 5 (online): run FOM and PROM-RBF on a test trajectory unseen by POD training.

Default behavior builds a new "unseen" trajectory in parameter space (mu), maps it
into strain waypoints, runs FOM (iterative) and PROM-RBF (explicit surrogate), and
compares homogenized stress/strain histories.
"""

import argparse
import json
import os
import sys

import numpy as np

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from fom_solver_rve import (
    setup_kratos_parameters,
    RunFomBatchSimulation,
    LoadStrainWaypointsFromFile,
)
from prom_rbf_solver_rve import LoadPromRbfModel, RunPromRbfBatchSimulation


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage5 online: FOM vs PROM-RBF on unseen test trajectory")
    p.add_argument("--stage0-file", type=str, default="stage_0_trajectory/stage_0_trajectories.npz")
    p.add_argument(
        "--trajectory-source",
        type=str,
        default="unseen",
        choices=["unseen", "stage0"],
        help="Use unseen generated trajectory or existing trajectory_<i> from stage0-file.",
    )
    p.add_argument("--trajectory-index", type=int, default=1, help="Used only when --trajectory-source stage0")
    p.add_argument("--unseen-waypoints", type=int, default=41)

    p.add_argument("--mesh", type=str, default="rve_geometry")
    p.add_argument("--reference-amplitude", type=float, default=None)

    p.add_argument("--run-fom", action="store_true", help="Force FOM recompute.")
    p.add_argument(
        "--run-prom-rbf",
        "--run-prom",
        dest="run_prom_rbf",
        action="store_true",
        help="Force PROM-RBF recompute.",
    )
    p.add_argument("--prom-corrector-iters", type=int, default=6)
    p.add_argument("--prom-corrector-rel-tol", type=float, default=1.0e-5)
    p.add_argument("--prom-corrector-abs-tol", type=float, default=1.0e-10)
    p.add_argument("--prom-corrector-dq-abs-tol", type=float, default=1.0e-7)
    p.add_argument("--prom-corrector-dq-rel-tol", type=float, default=1.0e-6)
    p.add_argument("--prom-corrector-res-floor-for-dq", type=float, default=1.0e-1)
    p.add_argument("--prom-corrector-l2-reg", type=float, default=1.0e-10)

    p.add_argument("--stage2a-dir", type=str, default="stage_2a_pod_data")
    p.add_argument("--stage2b-dir", type=str, default="stage_2b_ls_master")
    p.add_argument(
        "--stage3-dataset-file",
        type=str,
        default="stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    )
    p.add_argument("--stage4-rbf-dir", type=str, default="stage_4_prom_rbf_grid")

    p.add_argument("--out-dir", type=str, default="stage_5_online_compare")
    p.add_argument("--save-plots", type=int, default=1, choices=[0, 1])
    return p.parse_args()


def _map_param_to_strain(mu_path: np.ndarray, mapping: str) -> np.ndarray:
    gx = np.asarray(mu_path[:, 0], dtype=float)
    gxy = np.asarray(mu_path[:, 1], dtype=float)

    if mapping == "small_strain":
        exx = gx
        eyy = np.zeros_like(gx)
        gamma = gxy
    elif mapping == "green_lagrange_upper":
        exx = gx + 0.5 * gx * gx
        eyy = 0.5 * gxy * gxy
        gamma = gxy * (1.0 + gx)
    else:
        raise RuntimeError(f"Unsupported mapping='{mapping}'.")

    return np.column_stack([exx, eyy, gamma])


def _build_unseen_mu_path(stage0, n_waypoints: int) -> np.ndarray:
    gx_vals = np.asarray(stage0["gx_values"], dtype=float)
    gxy_vals = np.asarray(stage0["gxy_values"], dtype=float)

    gx_min = float(np.min(gx_vals))
    gx_max = float(np.max(gx_vals))
    gy_min = float(np.min(gxy_vals))
    gy_max = float(np.max(gxy_vals))

    amp = 0.70 * min(abs(gy_min), abs(gy_max))
    amp = max(amp, 1e-6)

    n = max(int(n_waypoints), 7)
    t = np.linspace(0.0, 1.0, n)

    # Use tension-dominated branch for robust FOM convergence and keep values
    # off the structured nodes via small phase/offset shifts.
    gx_lo = max(0.0, gx_min) + 0.006
    gx_hi = min(gx_max - 0.006, 0.40)
    gx = gx_lo + (gx_hi - gx_lo) * t
    gxy = (
        0.70 * amp * np.sin(2.0 * np.pi * t + 0.23)
        + 0.18 * amp * np.sin(5.0 * np.pi * t + 0.61)
    )

    # Keep strict bounds.
    gxy = np.clip(gxy, gy_min, gy_max)

    mu_curve = np.column_stack([gx, gxy])
    mu_return = mu_curve[::-1].copy()

    # Start/end at origin for consistency with existing workflows.
    origin = np.array([[0.0, 0.0]], dtype=float)
    mu_path = np.vstack([origin, mu_curve, mu_return, origin])
    return mu_path


def _nearest_dist(query: np.ndarray, cloud: np.ndarray):
    d2 = np.sum((query[:, None, :] - cloud[None, :, :]) ** 2, axis=2)
    nn = np.min(d2, axis=1)
    return np.sqrt(nn)


def _compute_rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def _to_mu_space(mu_gx_gxy: np.ndarray, mu_space: str) -> np.ndarray:
    mu = np.asarray(mu_gx_gxy, dtype=float)
    if mu.ndim != 2 or mu.shape[1] != 2:
        raise RuntimeError(f"Expected mu array with shape (n,2), got {mu.shape}.")
    if mu_space == "gx_gxy":
        return mu.copy()
    if mu_space == "f11_f12":
        out = np.empty_like(mu)
        out[:, 0] = 1.0 + mu[:, 0]  # f11
        out[:, 1] = mu[:, 1]        # f12
        return out
    raise RuntimeError(f"Unsupported mu_space='{mu_space}'.")


def _plot_reference_domain_and_test_path(
    out_dir: str,
    stage0,
    mu_space: str,
    mu_train: np.ndarray,
    mu_test: np.ndarray,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        print("[WARN] matplotlib not available. Skipping reference-domain plot.")
        return

    gx_vals = np.asarray(stage0["gx_values"], dtype=float)
    gxy_vals = np.asarray(stage0["gxy_values"], dtype=float)
    gx_min = float(np.min(gx_vals))
    gx_max = float(np.max(gx_vals))
    gxy_min = float(np.min(gxy_vals))
    gxy_max = float(np.max(gxy_vals))

    if mu_space == "gx_gxy":
        x_min, x_max = gx_min, gx_max
        y_min, y_max = gxy_min, gxy_max
        x_label = r"$G_x^{macro}$"
        y_label = r"$G_{xy}^{macro}$"
    elif mu_space == "f11_f12":
        x_min, x_max = 1.0 + gx_min, 1.0 + gx_max
        y_min, y_max = gxy_min, gxy_max
        x_label = r"$F_{11}$"
        y_label = r"$F_{12}$"
    else:
        raise RuntimeError(f"Unsupported mu_space='{mu_space}'.")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    rect = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        facecolor="#dfe8f5",
        edgecolor="#4a6fa5",
        linewidth=1.4,
        alpha=0.35,
        label="Stage0 training domain",
    )
    ax.add_patch(rect)

    ax.scatter(mu_train[:, 0], mu_train[:, 1], s=6, c="#9aa0a6", alpha=0.18, label="Stage3 train points")
    ax.plot(mu_test[:, 0], mu_test[:, 1], color="#d62728", lw=2.0, label="Test trajectory")
    ax.scatter(mu_test[0, 0], mu_test[0, 1], s=45, c="#2ca02c", marker="o", label="Start")
    ax.scatter(mu_test[-1, 0], mu_test[-1, 1], s=45, c="#111111", marker="x", label="End")

    dx = max(1e-12, x_max - x_min)
    dy = max(1e-12, y_max - y_min)
    ax.set_xlim(x_min - 0.06 * dx, x_max + 0.06 * dx)
    ax.set_ylim(y_min - 0.12 * dy, y_max + 0.12 * dy)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Reference training domain and test trajectory")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reference_domain_and_test_trajectory.png"), dpi=180)
    plt.close(fig)


def _maybe_plot(out_dir: str, eps_f: np.ndarray, sig_f: np.ndarray, eps_p: np.ndarray, sig_p: np.ndarray):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available. Skipping plots.")
        return

    n = min(len(eps_f), len(eps_p), len(sig_f), len(sig_p))
    eps_f = eps_f[:n]
    sig_f = sig_f[:n]
    eps_p = eps_p[:n]
    sig_p = sig_p[:n]

    labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "xy"),
    ]
    for i, ylab, xlab, suf in labels:
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        ax.plot(eps_f[:, i], sig_f[:, i], "k-", lw=1.7, label="FOM")
        ax.plot(eps_p[:, i], sig_p[:, i], "r--", lw=1.5, label="PROM-RBF")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"online_compare_{suf}.png"), dpi=180)
        plt.close(fig)

    denom = np.maximum(np.linalg.norm(sig_f, axis=1), 1e-30)
    rel_sig = np.linalg.norm(sig_f - sig_p, axis=1) / denom

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.plot(rel_sig, "b-", lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("step index")
    ax.set_ylabel("relative stress error")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "online_compare_rel_stress_error.png"), dpi=180)
    plt.close(fig)


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.stage0_file):
        raise FileNotFoundError(f"stage0-file not found: {args.stage0_file}")
    stage0 = np.load(args.stage0_file, allow_pickle=True)

    mapping = str(np.ravel(stage0["mapping"])[0]) if "mapping" in stage0 else "green_lagrange_upper"

    if "ref_steps" not in stage0:
        raise RuntimeError(
            "stage0-file does not contain 'ref_steps'. "
            "Stage5 requires fixed reference stepping equal to Stage0/Stage1 training."
        )
    reference_steps = int(np.ravel(stage0["ref_steps"])[0])

    if args.reference_amplitude is None:
        if "reference_amplitude" in stage0:
            reference_amplitude = float(np.ravel(stage0["reference_amplitude"])[0])
        else:
            reference_amplitude = 0.5
    else:
        reference_amplitude = float(args.reference_amplitude)

    stage3 = np.load(args.stage3_dataset_file, allow_pickle=True)
    mu_train = np.asarray(stage3["mu_all"], dtype=float)
    mu_space = str(np.ravel(stage3["mu_space"])[0]) if "mu_space" in stage3 else "gx_gxy"

    if args.trajectory_source == "stage0":
        strain_path, _ = LoadStrainWaypointsFromFile(args.stage0_file, args.trajectory_index)
        key_param = f"trajectory_param_{int(args.trajectory_index)}"
        if key_param not in stage0:
            raise RuntimeError(
                f"Stage0 file missing key '{key_param}'. "
                "Expected parametric trajectory for plotting/novelty checks."
            )
        mu_path_gxgxy = np.asarray(stage0[key_param], dtype=float)
        traj_label = f"stage0_traj{int(args.trajectory_index)}"
    else:
        mu_path_gxgxy = _build_unseen_mu_path(stage0, n_waypoints=int(args.unseen_waypoints))
        strain_path = _map_param_to_strain(mu_path_gxgxy, mapping=mapping)
        traj_label = "unseen"

    mu_eval = _to_mu_space(mu_path_gxgxy, mu_space=mu_space)

    novelty = {
        "enabled": False,
        "min_dist": None,
        "mean_dist": None,
        "max_dist": None,
        "exact_matches_tol_1e-12": None,
    }
    d = _nearest_dist(mu_eval, mu_train)
    novelty = {
        "enabled": True,
        "min_dist": float(np.min(d)),
        "mean_dist": float(np.mean(d)),
        "max_dist": float(np.max(d)),
        "exact_matches_tol_1e-12": int(np.sum(d <= 1e-12)),
    }

    _plot_reference_domain_and_test_path(
        out_dir=args.out_dir,
        stage0=stage0,
        mu_space=mu_space,
        mu_train=mu_train,
        mu_test=mu_eval,
    )

    params_fom = setup_kratos_parameters(args.mesh)
    params_prom = setup_kratos_parameters(args.mesh)

    print("=" * 78)
    print("Stage 5 online: FOM vs PROM-RBF")
    print("=" * 78)
    print(f"trajectory source : {args.trajectory_source}")
    print(f"trajectory label  : {traj_label}")
    print(f"mapping           : {mapping}")
    print(f"reference_steps   : {reference_steps}")
    print(f"reference_amp     : {reference_amplitude}")
    if novelty["enabled"]:
        print(
            "novelty(mu)       : "
            f"min={novelty['min_dist']:.3e}, mean={novelty['mean_dist']:.3e}, "
            f"max={novelty['max_dist']:.3e}, exact@1e-12={novelty['exact_matches_tol_1e-12']}"
        )

    eps_fom = sig_fom = None
    fom_tag = "fom"
    prom_tag = "prom_rbf"
    fom_eps_file = os.path.join(args.out_dir, f"trajectory_{fom_tag}_strain.npy")
    fom_sig_file = os.path.join(args.out_dir, f"trajectory_{fom_tag}_stress.npy")
    prom_eps_file = os.path.join(args.out_dir, f"trajectory_{prom_tag}_strain.npy")
    prom_sig_file = os.path.join(args.out_dir, f"trajectory_{prom_tag}_stress.npy")
    fom_source = "cached"
    prom_source = "cached"

    if args.run_fom or not (os.path.exists(fom_eps_file) and os.path.exists(fom_sig_file)):
        print("\n[RUN] FOM (iterative Newton)...")
        eps_fom, sig_fom = RunFomBatchSimulation(
            params_fom,
            out_dir=args.out_dir,
            save_results=True,
            save_plot=False,
            strain_path=strain_path,
            trajectory_index=fom_tag,
            reference_amplitude=reference_amplitude,
            reference_steps=reference_steps,
        )
        fom_source = "new"
    else:
        print("\n[CACHE] Loading cached FOM.")
        eps_fom = np.load(fom_eps_file)
        sig_fom = np.load(fom_sig_file)

    eps_prom = sig_prom = None
    if args.run_prom_rbf or not (os.path.exists(prom_eps_file) and os.path.exists(prom_sig_file)):
        print("\n[RUN] PROM-RBF (online surrogate + reduced corrector)...")
        model_pack = LoadPromRbfModel(
            stage2a_dir=args.stage2a_dir,
            stage2b_dir=args.stage2b_dir,
            stage3_dataset_file=args.stage3_dataset_file,
            stage4_rbf_dir=args.stage4_rbf_dir,
        )
        eps_prom, sig_prom = RunPromRbfBatchSimulation(
            params_prom,
            model_pack=model_pack,
            strain_path=strain_path,
            out_dir=args.out_dir,
            trajectory_index=prom_tag,
            reference_amplitude=reference_amplitude,
            reference_steps=reference_steps,
            prom_corrector_max_iters=args.prom_corrector_iters,
            prom_corrector_rel_tol=args.prom_corrector_rel_tol,
            prom_corrector_abs_tol=args.prom_corrector_abs_tol,
            prom_corrector_dq_abs_tol=args.prom_corrector_dq_abs_tol,
            prom_corrector_dq_rel_tol=args.prom_corrector_dq_rel_tol,
            prom_corrector_res_floor_for_dq=args.prom_corrector_res_floor_for_dq,
            prom_corrector_l2_reg=args.prom_corrector_l2_reg,
        )
        prom_source = "new"
    else:
        print("\n[CACHE] Loading cached PROM-RBF.")
        eps_prom = np.load(prom_eps_file)
        sig_prom = np.load(prom_sig_file)

    n = min(len(eps_fom), len(eps_prom), len(sig_fom), len(sig_prom))
    eps_fom = np.asarray(eps_fom[:n], dtype=float)
    eps_prom = np.asarray(eps_prom[:n], dtype=float)
    sig_fom = np.asarray(sig_fom[:n], dtype=float)
    sig_prom = np.asarray(sig_prom[:n], dtype=float)

    err_sig = _compute_rel_error(sig_fom, sig_prom)
    err_eps = _compute_rel_error(eps_fom, eps_prom)

    summary = {
        "trajectory_source": args.trajectory_source,
        "trajectory_label": traj_label,
        "mapping": mapping,
        "reference_steps": int(reference_steps),
        "reference_amplitude": float(reference_amplitude),
        "n_steps_compared": int(n),
        "novelty": novelty,
        "rel_error_stress": float(err_sig),
        "rel_error_strain": float(err_eps),
        "out_dir": args.out_dir,
        "fom_tag": fom_tag,
        "prom_tag": prom_tag,
        "fom_source": fom_source,
        "prom_rbf_source": prom_source,
        "prom_corrector_iters": int(args.prom_corrector_iters),
        "prom_corrector_rel_tol": float(args.prom_corrector_rel_tol),
        "prom_corrector_abs_tol": float(args.prom_corrector_abs_tol),
        "prom_corrector_dq_abs_tol": float(args.prom_corrector_dq_abs_tol),
        "prom_corrector_dq_rel_tol": float(args.prom_corrector_dq_rel_tol),
        "prom_corrector_res_floor_for_dq": float(args.prom_corrector_res_floor_for_dq),
        "prom_corrector_l2_reg": float(args.prom_corrector_l2_reg),
        "stage2a_dir": args.stage2a_dir,
        "stage2b_dir": args.stage2b_dir,
        "stage3_dataset_file": args.stage3_dataset_file,
        "stage4_rbf_dir": args.stage4_rbf_dir,
    }

    with open(os.path.join(args.out_dir, "stage5_online_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        os.path.join(args.out_dir, "stage5_online_compare_arrays.npz"),
        eps_fom=eps_fom,
        sig_fom=sig_fom,
        eps_prom=eps_prom,
        sig_prom=sig_prom,
    )

    if int(args.save_plots) == 1:
        _maybe_plot(args.out_dir, eps_fom, sig_fom, eps_prom, sig_prom)

    print("\n[RESULT] Comparison done")
    print(f"  compared steps      : {n}")
    print(f"  rel error stress    : {err_sig:.3e}")
    print(f"  rel error strain    : {err_eps:.3e}")
    print(f"  FOM source          : {fom_source}")
    print(f"  PROM-RBF source     : {prom_source}")
    print(f"  summary file        : {os.path.join(args.out_dir, 'stage5_online_summary.json')}")


if __name__ == "__main__":
    main()
