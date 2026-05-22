#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 8 (2D-MAWECM): online comparison FOM vs PROM-GPR vs HPROM-MAWECM-GPR."""

import argparse
import json
import os
import sys
import time

import numpy as np

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from fom_solver_rve import setup_kratos_parameters, RunFomBatchSimulation, LoadStrainWaypointsFromFile
from prom_gpr_solver_rve import LoadPromGprModel, RunPromGprBatchSimulation
from hprom_mawecm_gpr_solver_rve import LoadHpromMawEcmGprModel, RunHpromMawEcmGprBatchSimulation


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage8 online: FOM vs PROM-GPR vs HPROM-MAWECM-GPR")
    p.add_argument("--stage0-file", type=str, default="stage_0_trajectory/stage_0_trajectories.npz")
    p.add_argument("--trajectory-source", type=str, default="unseen", choices=["unseen", "stage0"])
    p.add_argument("--trajectory-index", type=int, default=1)
    p.add_argument("--unseen-waypoints", type=int, default=41)

    p.add_argument("--mesh", type=str, default="rve_geometry")
    p.add_argument("--reference-amplitude", type=float, default=None)

    p.add_argument("--run-fom", action="store_true")
    p.add_argument("--run-prom-gpr", action="store_true")
    p.add_argument("--run-hprom-mawecm-gpr", action="store_true")

    p.add_argument("--prom-corrector-iters", type=int, default=10)
    p.add_argument("--prom-corrector-rel-tol", type=float, default=1.0e-5)
    p.add_argument("--prom-corrector-abs-tol", type=float, default=1.0e-10)
    p.add_argument("--prom-corrector-dq-abs-tol", type=float, default=1.0e-7)
    p.add_argument("--prom-corrector-dq-rel-tol", type=float, default=1.0e-6)
    p.add_argument("--prom-corrector-res-floor-for-dq", type=float, default=1.0e-1)
    p.add_argument("--prom-corrector-damping-after-iter", type=int, default=10)
    p.add_argument("--prom-corrector-damping-factor", type=float, default=0.5)
    p.add_argument("--prom-use-old-stiffness-in-first-iteration", type=int, default=1, choices=[0, 1])
    p.add_argument("--prom-old-stiffness-residual-cutoff", type=float, default=1.0e5)
    p.add_argument("--prom-fail-on-nonconvergence", type=int, default=1, choices=[0, 1])
    p.add_argument("--prom-corrector-l2-reg", type=float, default=1.0e-10)

    p.add_argument("--hprom-corrector-iters", type=int, default=10)
    p.add_argument("--hprom-corrector-rel-tol", type=float, default=1.0e-5)
    p.add_argument("--hprom-corrector-abs-tol", type=float, default=1.0e-10)
    p.add_argument("--hprom-corrector-dq-abs-tol", type=float, default=1.0e-7)
    p.add_argument("--hprom-corrector-dq-rel-tol", type=float, default=1.0e-6)
    p.add_argument("--hprom-corrector-res-floor-for-dq", type=float, default=1.0e-1)
    p.add_argument("--hprom-corrector-damping-after-iter", type=int, default=10)
    p.add_argument("--hprom-corrector-damping-factor", type=float, default=0.5)
    p.add_argument("--hprom-use-old-stiffness-in-first-iteration", type=int, default=1, choices=[0, 1])
    p.add_argument("--hprom-old-stiffness-residual-cutoff", type=float, default=1.0e5)
    p.add_argument("--hprom-corrector-l2-reg", type=float, default=1.0e-10)
    p.add_argument("--hprom-fail-on-nonconvergence", type=int, default=1, choices=[0, 1])
    p.add_argument("--hprom-update-maw-each-iter", type=int, default=1, choices=[0, 1])
    p.add_argument(
        "--hprom-include-weight-tangent",
        type=int,
        default=1,
        choices=[0, 1],
        help="Include consistent MAW term with analytic d(w_res)/d(q_m) in reduced tangent.",
    )
    p.add_argument("--hprom-clip-nonnegative", type=int, default=1, choices=[0, 1])
    p.add_argument("--hprom-renorm-weights", type=int, default=0, choices=[0, 1])
    p.add_argument(
        "--hprom-homogenization-mode",
        type=str,
        default="ecm_separate",
        choices=["full_fom", "ecm_separate"],
        help="Homogenization backend for HPROM-MAWECM-GPR.",
    )

    p.add_argument("--stage2a-dir", type=str, default="stage_2a_pod_data")
    p.add_argument("--stage2b-dir", type=str, default="stage_2b_ls_master")
    p.add_argument(
        "--stage3-dataset-file",
        type=str,
        default="stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    )
    p.add_argument("--stage4-gpr-dir", type=str, default="stage_4_prom_gpr_sparse")
    p.add_argument("--mawecm-file", type=str, default="stage_8b_hprom_mawecm_res_rbf/ecm_weights_all.npz")
    p.add_argument("--out-dir", type=str, default="stage_8_online_hprom_mawecm_gpr")
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

    gx_lo = max(0.0, gx_min) + 0.006
    gx_hi = min(gx_max - 0.006, 0.40)
    gx = gx_lo + (gx_hi - gx_lo) * t
    gxy = 0.70 * amp * np.sin(2.0 * np.pi * t + 0.23) + 0.18 * amp * np.sin(5.0 * np.pi * t + 0.61)
    gxy = np.clip(gxy, gy_min, gy_max)

    mu_curve = np.column_stack([gx, gxy])
    mu_return = mu_curve[::-1].copy()
    origin = np.array([[0.0, 0.0]], dtype=float)
    return np.vstack([origin, mu_curve, mu_return, origin])


def _to_mu_space(mu_gx_gxy: np.ndarray, mu_space: str) -> np.ndarray:
    mu = np.asarray(mu_gx_gxy, dtype=float)
    if mu_space == "gx_gxy":
        return mu.copy()
    if mu_space == "f11_f12":
        out = np.empty_like(mu)
        out[:, 0] = 1.0 + mu[:, 0]
        out[:, 1] = mu[:, 1]
        return out
    raise RuntimeError(f"Unsupported mu_space='{mu_space}'.")


def _nearest_dist(query: np.ndarray, cloud: np.ndarray):
    d2 = np.sum((query[:, None, :] - cloud[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def _compute_rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def _maybe_plot(out_dir: str, eps_f, sig_f, eps_p, sig_p, eps_h, sig_h):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available. Skipping plots.")
        return

    n = min(len(eps_f), len(eps_p), len(eps_h), len(sig_f), len(sig_p), len(sig_h))
    eps_f = eps_f[:n]
    sig_f = sig_f[:n]
    eps_p = eps_p[:n]
    sig_p = sig_p[:n]
    eps_h = eps_h[:n]
    sig_h = sig_h[:n]

    labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "xy"),
    ]
    for i, ylab, xlab, suf in labels:
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.plot(eps_f[:, i], sig_f[:, i], "k-", lw=1.8, label="FOM")
        ax.plot(eps_p[:, i], sig_p[:, i], "r--", lw=1.5, label="PROM-GPR")
        ax.plot(eps_h[:, i], sig_h[:, i], "b:", lw=1.8, label="HPROM-MAWECM-GPR")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"stage8_compare_{suf}.png"), dpi=180)
        plt.close(fig)

    den = np.maximum(np.linalg.norm(sig_f, axis=1), 1e-30)
    err_p = np.linalg.norm(sig_f - sig_p, axis=1) / den
    err_h = np.linalg.norm(sig_f - sig_h, axis=1) / den
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(err_p, "r-", lw=1.3, label="PROM-GPR rel. stress error")
    ax.plot(err_h, "b-", lw=1.3, label="HPROM-MAWECM-GPR rel. stress error")
    ax.set_yscale("log")
    ax.set_xlabel("step index")
    ax.set_ylabel("relative error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage8_compare_rel_stress_error.png"), dpi=180)
    plt.close(fig)


def _timing_file(out_dir: str, tag: str) -> str:
    return os.path.join(out_dir, f"trajectory_{tag}_runtime.json")


def _write_runtime(path: str, runtime_sec: float, source: str) -> None:
    payload = {"runtime_sec": float(runtime_sec), "source": str(source)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_runtime(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "runtime_sec" not in data:
            return None
        return float(data["runtime_sec"])
    except Exception:
        return None


def _maybe_plot_timing(out_dir: str, t_fom, t_prom, t_hprom):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = ["FOM", "PROM-GPR", "HPROM-MAW"]
    vals = [t_fom, t_prom, t_hprom]
    if any(v is None for v in vals):
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Runtime comparison")
    ax.grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage8_timing_comparison.png"), dpi=180)
    plt.close(fig)


def _strip_mdpa_extension(mesh_name: str) -> str:
    s = str(mesh_name).strip()
    return s[:-5] if s.endswith(".mdpa") else s


def _resolve_hprom_mesh(base_mesh: str, mawecm_file: str, enable_hrom: bool):
    mesh_base = _strip_mdpa_extension(base_mesh)
    if not enable_hrom:
        return mesh_base, "base_mesh"
    if not os.path.exists(mawecm_file):
        return mesh_base, "base_mesh_maw_missing"
    try:
        data = np.load(mawecm_file, allow_pickle=True)
    except Exception:
        return mesh_base, "base_mesh_maw_unreadable"
    if "hrom_mesh_base" not in data.files:
        return mesh_base, "base_mesh_no_hrom_key"
    hrom_mesh = _strip_mdpa_extension(str(np.ravel(data["hrom_mesh_base"])[0]))
    if not os.path.exists(f"{hrom_mesh}.mdpa"):
        return mesh_base, "base_mesh_hrom_mdpa_missing"
    return hrom_mesh, "hrom_mesh"


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stage0 = np.load(args.stage0_file, allow_pickle=True)
    stage3 = np.load(args.stage3_dataset_file, allow_pickle=True)

    mapping = str(np.ravel(stage0["mapping"])[0]) if "mapping" in stage0 else "green_lagrange_upper"
    reference_steps = int(np.ravel(stage0["ref_steps"])[0])
    reference_amplitude = (
        float(np.ravel(stage0["reference_amplitude"])[0])
        if args.reference_amplitude is None
        else float(args.reference_amplitude)
    )

    mu_space = str(np.ravel(stage3["mu_space"])[0]) if "mu_space" in stage3 else "gx_gxy"
    mu_train = np.asarray(stage3["mu_all"], dtype=float)

    if args.trajectory_source == "stage0":
        strain_path, _ = LoadStrainWaypointsFromFile(args.stage0_file, args.trajectory_index)
        key_param = f"trajectory_param_{int(args.trajectory_index)}"
        mu_path_gxgxy = np.asarray(stage0[key_param], dtype=float)
        traj_label = f"stage0_traj{int(args.trajectory_index)}"
    else:
        mu_path_gxgxy = _build_unseen_mu_path(stage0, n_waypoints=int(args.unseen_waypoints))
        strain_path = _map_param_to_strain(mu_path_gxgxy, mapping=mapping)
        traj_label = "unseen"

    mu_eval = _to_mu_space(mu_path_gxgxy, mu_space=mu_space)
    d = _nearest_dist(mu_eval, mu_train)
    novelty = {
        "min_dist": float(np.min(d)),
        "mean_dist": float(np.mean(d)),
        "max_dist": float(np.max(d)),
        "exact_matches_tol_1e-12": int(np.sum(d <= 1e-12)),
    }

    mesh_fom = _strip_mdpa_extension(args.mesh)
    mesh_prom = mesh_fom
    # Strict Stage-8 mode: always run HPROM-MAW on base mesh and compute
    # homogenization on full mesh (no HROM mdpa branch).
    mesh_hprom = mesh_fom
    mesh_hprom_mode = "base_mesh"

    params_fom = setup_kratos_parameters(mesh_fom)
    params_prom = setup_kratos_parameters(mesh_prom)
    params_hprom = setup_kratos_parameters(mesh_hprom)

    print("=" * 80)
    print("Stage 8 online: FOM vs PROM-GPR vs HPROM-MAWECM-GPR")
    print("=" * 80)
    print(f"trajectory source : {args.trajectory_source}")
    print(f"trajectory label  : {traj_label}")
    print(f"mesh FOM/PROM     : {mesh_fom}")
    print(f"mesh HPROM-MAW    : {mesh_hprom} ({mesh_hprom_mode})")
    print(f"reference_steps   : {reference_steps}")
    print(
        "novelty(mu)       : "
        f"min={novelty['min_dist']:.3e}, mean={novelty['mean_dist']:.3e}, "
        f"max={novelty['max_dist']:.3e}, exact@1e-12={novelty['exact_matches_tol_1e-12']}"
    )

    tag_f = "fom"
    tag_p = "prom_gpr"
    tag_h = "hprom_mawecm_gpr"

    f_eps_file = os.path.join(args.out_dir, f"trajectory_{tag_f}_strain.npy")
    f_sig_file = os.path.join(args.out_dir, f"trajectory_{tag_f}_stress.npy")
    p_eps_file = os.path.join(args.out_dir, f"trajectory_{tag_p}_strain.npy")
    p_sig_file = os.path.join(args.out_dir, f"trajectory_{tag_p}_stress.npy")
    h_eps_file = os.path.join(args.out_dir, f"trajectory_{tag_h}_strain.npy")
    h_sig_file = os.path.join(args.out_dir, f"trajectory_{tag_h}_stress.npy")
    p_qs_file = os.path.join(args.out_dir, f"trajectory_{tag_p}_q_s.npy")
    h_qs_file = os.path.join(args.out_dir, f"trajectory_{tag_h}_q_s.npy")

    src_f, src_p, src_h = "cached", "cached", "cached"
    t_f, t_p, t_h = None, None, None

    tfile_f = _timing_file(args.out_dir, tag_f)
    tfile_p = _timing_file(args.out_dir, tag_p)
    tfile_h = _timing_file(args.out_dir, tag_h)

    if args.run_fom or not (os.path.exists(f_eps_file) and os.path.exists(f_sig_file)):
        print("\n[RUN] FOM")
        t0 = time.perf_counter()
        eps_f, sig_f = RunFomBatchSimulation(
            params_fom,
            out_dir=args.out_dir,
            save_results=True,
            save_plot=False,
            strain_path=strain_path,
            trajectory_index=tag_f,
            reference_amplitude=reference_amplitude,
            reference_steps=reference_steps,
        )
        t_f = float(time.perf_counter() - t0)
        _write_runtime(tfile_f, t_f, source="new")
        src_f = "new"
    else:
        print("\n[CACHE] FOM")
        eps_f = np.load(f_eps_file)
        sig_f = np.load(f_sig_file)
        t_f = _read_runtime(tfile_f)

    if args.run_prom_gpr or not (os.path.exists(p_eps_file) and os.path.exists(p_sig_file)):
        print("\n[RUN] PROM-GPR")
        prom_pack = LoadPromGprModel(
            stage2a_dir=args.stage2a_dir,
            stage2b_dir=args.stage2b_dir,
            stage3_dataset_file=args.stage3_dataset_file,
            stage4_gpr_dir=args.stage4_gpr_dir,
        )
        t0 = time.perf_counter()
        eps_p, sig_p = RunPromGprBatchSimulation(
            params_prom,
            model_pack=prom_pack,
            strain_path=strain_path,
            out_dir=args.out_dir,
            trajectory_index=tag_p,
            reference_amplitude=reference_amplitude,
            reference_steps=reference_steps,
            prom_corrector_max_iters=args.prom_corrector_iters,
            prom_corrector_rel_tol=args.prom_corrector_rel_tol,
            prom_corrector_abs_tol=args.prom_corrector_abs_tol,
            prom_corrector_dq_abs_tol=args.prom_corrector_dq_abs_tol,
            prom_corrector_dq_rel_tol=args.prom_corrector_dq_rel_tol,
            prom_corrector_res_floor_for_dq=args.prom_corrector_res_floor_for_dq,
            prom_corrector_damping_after_iter=args.prom_corrector_damping_after_iter,
            prom_corrector_damping_factor=args.prom_corrector_damping_factor,
            prom_use_old_stiffness_in_first_iteration=args.prom_use_old_stiffness_in_first_iteration,
            prom_old_stiffness_residual_cutoff=args.prom_old_stiffness_residual_cutoff,
            prom_corrector_l2_reg=args.prom_corrector_l2_reg,
            prom_fail_on_nonconvergence=args.prom_fail_on_nonconvergence,
            track_q_pod=0,
        )
        t_p = float(time.perf_counter() - t0)
        _write_runtime(tfile_p, t_p, source="new")
        src_p = "new"
    else:
        print("\n[CACHE] PROM-GPR")
        eps_p = np.load(p_eps_file)
        sig_p = np.load(p_sig_file)
        t_p = _read_runtime(tfile_p)

    if args.run_hprom_mawecm_gpr or not (os.path.exists(h_eps_file) and os.path.exists(h_sig_file)):
        print("\n[RUN] HPROM-MAWECM-GPR")
        prom_pack_h, maw_pack = LoadHpromMawEcmGprModel(
            stage2a_dir=args.stage2a_dir,
            stage2b_dir=args.stage2b_dir,
            stage3_dataset_file=args.stage3_dataset_file,
            stage4_gpr_dir=args.stage4_gpr_dir,
            mawecm_file=args.mawecm_file,
        )
        t0 = time.perf_counter()
        eps_h, sig_h = RunHpromMawEcmGprBatchSimulation(
            params_hprom,
            model_pack=prom_pack_h,
            mawecm_data=maw_pack,
            strain_path=strain_path,
            out_dir=args.out_dir,
            trajectory_index=tag_h,
            reference_amplitude=reference_amplitude,
            reference_steps=reference_steps,
            prom_corrector_max_iters=args.hprom_corrector_iters,
            prom_corrector_rel_tol=args.hprom_corrector_rel_tol,
            prom_corrector_abs_tol=args.hprom_corrector_abs_tol,
            prom_corrector_dq_abs_tol=args.hprom_corrector_dq_abs_tol,
            prom_corrector_dq_rel_tol=args.hprom_corrector_dq_rel_tol,
            prom_corrector_res_floor_for_dq=args.hprom_corrector_res_floor_for_dq,
            prom_corrector_damping_after_iter=args.hprom_corrector_damping_after_iter,
            prom_corrector_damping_factor=args.hprom_corrector_damping_factor,
            prom_use_old_stiffness_in_first_iteration=args.hprom_use_old_stiffness_in_first_iteration,
            prom_old_stiffness_residual_cutoff=args.hprom_old_stiffness_residual_cutoff,
            prom_corrector_l2_reg=args.hprom_corrector_l2_reg,
            prom_fail_on_nonconvergence=args.hprom_fail_on_nonconvergence,
            update_maw_weights_each_iter=args.hprom_update_maw_each_iter,
            include_weight_tangent=args.hprom_include_weight_tangent,
            clip_nonnegative=args.hprom_clip_nonnegative,
            renorm_weights=args.hprom_renorm_weights,
            homogenization_mode=args.hprom_homogenization_mode,
            track_q_pod=0,
        )
        t_h = float(time.perf_counter() - t0)
        _write_runtime(tfile_h, t_h, source="new")
        src_h = "new"
    else:
        print("\n[CACHE] HPROM-MAWECM-GPR")
        eps_h = np.load(h_eps_file)
        sig_h = np.load(h_sig_file)
        t_h = _read_runtime(tfile_h)

    n = min(len(eps_f), len(eps_p), len(eps_h), len(sig_f), len(sig_p), len(sig_h))
    eps_f = np.asarray(eps_f[:n], dtype=float)
    sig_f = np.asarray(sig_f[:n], dtype=float)
    eps_p = np.asarray(eps_p[:n], dtype=float)
    sig_p = np.asarray(sig_p[:n], dtype=float)
    eps_h = np.asarray(eps_h[:n], dtype=float)
    sig_h = np.asarray(sig_h[:n], dtype=float)

    err_ps = _compute_rel_error(sig_f, sig_p)
    err_hs = _compute_rel_error(sig_f, sig_h)
    err_pe = _compute_rel_error(eps_f, eps_p)
    err_he = _compute_rel_error(eps_f, eps_h)

    p_qm_file = os.path.join(args.out_dir, f"trajectory_{tag_p}_q_m.npy")
    h_qm_file = os.path.join(args.out_dir, f"trajectory_{tag_h}_q_m.npy")

    err_qm_ph = np.nan
    err_qs_ph = np.nan
    q_m_prom = None
    q_m_hprom = None
    q_s_prom = None
    q_s_hprom = None
    if os.path.exists(p_qm_file) and os.path.exists(h_qm_file):
        q_m_prom = np.asarray(np.load(p_qm_file), dtype=float)
        q_m_hprom = np.asarray(np.load(h_qm_file), dtype=float)
        nq = min(len(q_m_prom), len(q_m_hprom), n)
        err_qm_ph = _compute_rel_error(q_m_prom[:nq], q_m_hprom[:nq])
    else:
        print("[WARN] q_m trajectory files not found; skipping q_m PROM-vs-HPROM comparison.")

    if os.path.exists(p_qs_file) and os.path.exists(h_qs_file):
        q_s_prom = np.asarray(np.load(p_qs_file), dtype=float)
        q_s_hprom = np.asarray(np.load(h_qs_file), dtype=float)
        nq = min(len(q_s_prom), len(q_s_hprom), n)
        err_qs_ph = _compute_rel_error(q_s_prom[:nq], q_s_hprom[:nq])
    else:
        print("[WARN] q_s trajectory files not found; skipping q_s PROM-vs-HPROM comparison.")

    sp_prom = (float(t_f / t_p) if (t_f is not None and t_p is not None and t_p > 0.0) else None)
    sp_hprom = (float(t_f / t_h) if (t_f is not None and t_h is not None and t_h > 0.0) else None)

    summary = {
        "trajectory_source": args.trajectory_source,
        "trajectory_label": traj_label,
        "reference_steps": int(reference_steps),
        "reference_amplitude": float(reference_amplitude),
        "n_steps_compared": int(n),
        "novelty": novelty,
        "rel_error_stress_prom_vs_fom": float(err_ps),
        "rel_error_stress_hprom_maw_vs_fom": float(err_hs),
        "rel_error_strain_prom_vs_fom": float(err_pe),
        "rel_error_strain_hprom_maw_vs_fom": float(err_he),
        "rel_error_q_m_hprom_vs_prom": float(err_qm_ph) if np.isfinite(err_qm_ph) else None,
        "rel_error_q_s_hprom_vs_prom": float(err_qs_ph) if np.isfinite(err_qs_ph) else None,
        "source_fom": src_f,
        "source_prom_gpr": src_p,
        "source_hprom_mawecm_gpr": src_h,
        "mesh_fom": mesh_fom,
        "mesh_prom": mesh_prom,
        "mesh_hprom": mesh_hprom,
        "mesh_hprom_mode": mesh_hprom_mode,
        "runtime_sec_fom": t_f,
        "runtime_sec_prom_gpr": t_p,
        "runtime_sec_hprom_mawecm_gpr": t_h,
        "speedup_fom_over_prom_gpr": sp_prom,
        "speedup_fom_over_hprom_mawecm_gpr": sp_hprom,
        "stage2a_dir": args.stage2a_dir,
        "stage2b_dir": args.stage2b_dir,
        "stage3_dataset_file": args.stage3_dataset_file,
        "stage4_gpr_dir": args.stage4_gpr_dir,
        "mawecm_file": args.mawecm_file,
        "hprom_clip_nonnegative": int(args.hprom_clip_nonnegative),
        "hprom_renorm_weights": int(args.hprom_renorm_weights),
        "hprom_include_weight_tangent": int(args.hprom_include_weight_tangent),
        "hprom_fail_on_nonconvergence": int(args.hprom_fail_on_nonconvergence),
        "hprom_homogenization_mode": str(args.hprom_homogenization_mode),
        "hprom_corrector_dq_abs_tol": float(args.hprom_corrector_dq_abs_tol),
        "hprom_corrector_dq_rel_tol": float(args.hprom_corrector_dq_rel_tol),
        "hprom_corrector_res_floor_for_dq": float(args.hprom_corrector_res_floor_for_dq),
        "out_dir": args.out_dir,
    }

    with open(os.path.join(args.out_dir, "stage8_online_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        os.path.join(args.out_dir, "stage8_online_compare_arrays.npz"),
        eps_fom=eps_f,
        sig_fom=sig_f,
        eps_prom=eps_p,
        sig_prom=sig_p,
        eps_hprom_maw=eps_h,
        sig_hprom_maw=sig_h,
        q_m_prom=q_m_prom if q_m_prom is not None else np.zeros((0, 0), dtype=float),
        q_m_hprom=q_m_hprom if q_m_hprom is not None else np.zeros((0, 0), dtype=float),
        q_s_prom=q_s_prom if q_s_prom is not None else np.zeros((0, 0), dtype=float),
        q_s_hprom=q_s_hprom if q_s_hprom is not None else np.zeros((0, 0), dtype=float),
    )

    if int(args.save_plots) == 1:
        _maybe_plot(args.out_dir, eps_f, sig_f, eps_p, sig_p, eps_h, sig_h)
        _maybe_plot_timing(args.out_dir, t_f, t_p, t_h)

    print("\n[RESULT] Stage 8 comparison done")
    print(f"  compared steps                        : {n}")
    print(f"  rel stress error PROM       vs FOM    : {err_ps:.3e}")
    print(f"  rel stress error HPROM-MAW  vs FOM    : {err_hs:.3e}")
    print(f"  rel strain error PROM       vs FOM    : {err_pe:.3e}")
    print(f"  rel strain error HPROM-MAW  vs FOM    : {err_he:.3e}")
    if np.isfinite(err_qm_ph):
        print(f"  rel q_m error HPROM-MAW vs PROM-GPR   : {err_qm_ph:.3e}")
    if np.isfinite(err_qs_ph):
        print(f"  rel q_s error HPROM-MAW vs PROM-GPR   : {err_qs_ph:.3e}")
    if t_f is not None:
        print(f"  runtime FOM [s]                       : {t_f:.3f}")
    if t_p is not None:
        print(f"  runtime PROM-GPR [s]                  : {t_p:.3f}")
    if t_h is not None:
        print(f"  runtime HPROM-MAW [s]                 : {t_h:.3f}")
    if sp_prom is not None:
        print(f"  speedup FOM/PROM-GPR                  : {sp_prom:.2f}x")
    if sp_hprom is not None:
        print(f"  speedup FOM/HPROM-MAW                 : {sp_hprom:.2f}x")
    print(f"  source FOM / PROM / HPROM-MAW         : {src_f} / {src_p} / {src_h}")
    print(f"  summary file                          : {os.path.join(args.out_dir, 'stage8_online_summary.json')}")


if __name__ == "__main__":
    main()
