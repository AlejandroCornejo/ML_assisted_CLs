#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 7 (2D-MAWECM): online comparison FOM vs PROM-RBF vs HPROM-RBF.

- Uses classical ECM/HPROM first (no MAW in this stage).
- Reuses cached outputs unless --run-* flags force recompute.
"""

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
from prom_rbf_solver_rve import LoadPromRbfModel, RunPromRbfBatchSimulation
from hprom_rbf_solver_rve import LoadHpromRbfModel, RunHpromRbfBatchSimulation


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage7 online: FOM vs PROM-RBF vs HPROM-RBF")
    p.add_argument("--stage0-file", type=str, default="stage_0_trajectory/stage_0_trajectories.npz")
    p.add_argument(
        "--trajectory-source",
        type=str,
        default="unseen",
        choices=["unseen", "stage0"],
        help="Use unseen generated trajectory or existing trajectory_<i> from stage0-file.",
    )
    p.add_argument("--trajectory-index", type=int, default=1, help="Used only with --trajectory-source stage0")
    p.add_argument("--unseen-waypoints", type=int, default=41)

    p.add_argument("--mesh", type=str, default="rve_geometry")
    p.add_argument("--reference-amplitude", type=float, default=None)

    p.add_argument("--run-fom", action="store_true", help="Force FOM recompute.")
    p.add_argument("--run-prom-rbf", action="store_true", help="Force PROM-RBF recompute.")
    p.add_argument("--run-hprom-rbf", action="store_true", help="Force HPROM-RBF recompute.")

    p.add_argument("--prom-corrector-iters", type=int, default=6)
    p.add_argument("--prom-corrector-rel-tol", type=float, default=1.0e-5)
    p.add_argument("--prom-corrector-abs-tol", type=float, default=1.0e-10)
    p.add_argument("--prom-corrector-dq-abs-tol", type=float, default=1.0e-7)
    p.add_argument("--prom-corrector-dq-rel-tol", type=float, default=1.0e-6)
    p.add_argument("--prom-corrector-res-floor-for-dq", type=float, default=1.0e-1)
    p.add_argument("--prom-corrector-l2-reg", type=float, default=1.0e-10)

    p.add_argument("--hprom-corrector-iters", type=int, default=10)
    p.add_argument("--hprom-corrector-rel-tol", type=float, default=1.0e-5)
    p.add_argument("--hprom-corrector-abs-tol", type=float, default=1.0e-10)
    p.add_argument("--hprom-corrector-dq-abs-tol", type=float, default=1.0e-7)
    p.add_argument("--hprom-corrector-dq-rel-tol", type=float, default=1.0e-6)
    p.add_argument("--hprom-corrector-res-floor-for-dq", type=float, default=1.0e-1)
    p.add_argument("--hprom-corrector-l2-reg", type=float, default=1.0e-10)

    p.add_argument("--stage2a-dir", type=str, default="stage_2a_pod_data")
    p.add_argument("--stage2b-dir", type=str, default="stage_2b_ls_master")
    p.add_argument(
        "--stage3-dataset-file",
        type=str,
        default="stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    )
    p.add_argument("--stage4-rbf-dir", type=str, default="stage_4_prom_rbf_grid")
    p.add_argument("--ecm-file", type=str, default="stage_6b_hprom_ecm/ecm_weights_all.npz")
    p.add_argument(
        "--hprom-use-hrom-mdpa",
        type=int,
        default=1,
        choices=[0, 1],
        help="If enabled and ECM file contains hrom_mesh_base, HPROM runs on that reduced mdpa.",
    )

    p.add_argument("--out-dir", type=str, default="stage_7_online_hprom_rbf")
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
    gxy = (
        0.70 * amp * np.sin(2.0 * np.pi * t + 0.23)
        + 0.18 * amp * np.sin(5.0 * np.pi * t + 0.61)
    )
    gxy = np.clip(gxy, gy_min, gy_max)

    mu_curve = np.column_stack([gx, gxy])
    mu_return = mu_curve[::-1].copy()
    origin = np.array([[0.0, 0.0]], dtype=float)
    return np.vstack([origin, mu_curve, mu_return, origin])


def _nearest_dist(query: np.ndarray, cloud: np.ndarray):
    d2 = np.sum((query[:, None, :] - cloud[None, :, :]) ** 2, axis=2)
    nn = np.min(d2, axis=1)
    return np.sqrt(nn)


def _compute_rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


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


def _maybe_plot_comparisons(out_dir: str, eps_f, sig_f, eps_p, sig_p, eps_h, sig_h):
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
        ax.plot(eps_p[:, i], sig_p[:, i], "r--", lw=1.5, label="PROM-RBF")
        ax.plot(eps_h[:, i], sig_h[:, i], "b:", lw=1.7, label="HPROM-RBF")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"stage7_compare_{suf}.png"), dpi=180)
        plt.close(fig)

    den = np.maximum(np.linalg.norm(sig_f, axis=1), 1e-30)
    err_p = np.linalg.norm(sig_f - sig_p, axis=1) / den
    err_h = np.linalg.norm(sig_f - sig_h, axis=1) / den

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(err_p, "r-", lw=1.3, label="PROM-RBF rel. stress error")
    ax.plot(err_h, "b-", lw=1.3, label="HPROM-RBF rel. stress error")
    ax.set_yscale("log")
    ax.set_xlabel("step index")
    ax.set_ylabel("relative error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage7_compare_rel_stress_error.png"), dpi=180)
    plt.close(fig)


def _strip_mdpa_extension(mesh_name: str) -> str:
    s = str(mesh_name).strip()
    return s[:-5] if s.endswith(".mdpa") else s


def _resolve_hprom_mesh(base_mesh: str, ecm_file: str, enable_hrom: bool):
    mesh_base = _strip_mdpa_extension(base_mesh)
    if not enable_hrom:
        return mesh_base, "base_mesh"
    if not os.path.exists(ecm_file):
        return mesh_base, "base_mesh_ecm_missing"
    try:
        data = np.load(ecm_file, allow_pickle=True)
    except Exception:
        return mesh_base, "base_mesh_ecm_unreadable"
    if "hrom_mesh_base" not in data.files:
        return mesh_base, "base_mesh_no_hrom_key"
    hrom_mesh = _strip_mdpa_extension(str(np.ravel(data["hrom_mesh_base"])[0]))
    if not os.path.exists(f"{hrom_mesh}.mdpa"):
        return mesh_base, "base_mesh_hrom_mdpa_missing"
    return hrom_mesh, "hrom_mesh"


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

    vals = [t_fom, t_prom, t_hprom]
    if any(v is None for v in vals):
        return

    labels = ["FOM", "PROM-RBF", "HPROM-RBF"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Runtime comparison")
    ax.grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage7_timing_comparison.png"), dpi=180)
    plt.close(fig)


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.stage0_file):
        raise FileNotFoundError(f"stage0-file not found: {args.stage0_file}")
    stage0 = np.load(args.stage0_file, allow_pickle=True)
    stage3 = np.load(args.stage3_dataset_file, allow_pickle=True)

    mapping = str(np.ravel(stage0["mapping"])[0]) if "mapping" in stage0 else "green_lagrange_upper"
    if "ref_steps" not in stage0:
        raise RuntimeError("stage0-file does not contain 'ref_steps'.")
    reference_steps = int(np.ravel(stage0["ref_steps"])[0])

    if args.reference_amplitude is None:
        reference_amplitude = float(np.ravel(stage0["reference_amplitude"])[0]) if "reference_amplitude" in stage0 else 0.5
    else:
        reference_amplitude = float(args.reference_amplitude)

    mu_space = str(np.ravel(stage3["mu_space"])[0]) if "mu_space" in stage3 else "gx_gxy"
    mu_train = np.asarray(stage3["mu_all"], dtype=float)

    if args.trajectory_source == "stage0":
        strain_path, _ = LoadStrainWaypointsFromFile(args.stage0_file, args.trajectory_index)
        key_param = f"trajectory_param_{int(args.trajectory_index)}"
        if key_param not in stage0:
            raise RuntimeError(f"Missing key '{key_param}' in stage0-file")
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
    mesh_hprom, mesh_hprom_mode = _resolve_hprom_mesh(
        base_mesh=mesh_fom,
        ecm_file=args.ecm_file,
        enable_hrom=bool(int(args.hprom_use_hrom_mdpa)),
    )

    params_fom = setup_kratos_parameters(mesh_fom)
    params_prom = setup_kratos_parameters(mesh_prom)
    params_hprom = setup_kratos_parameters(mesh_hprom)

    print("=" * 78)
    print("Stage 7 online: FOM vs PROM-RBF vs HPROM-RBF")
    print("=" * 78)
    print(f"trajectory source : {args.trajectory_source}")
    print(f"trajectory label  : {traj_label}")
    print(f"mapping           : {mapping}")
    print(f"mesh FOM/PROM     : {mesh_fom}")
    print(f"mesh HPROM        : {mesh_hprom} ({mesh_hprom_mode})")
    print(f"reference_steps   : {reference_steps}")
    print(f"reference_amp     : {reference_amplitude}")
    print(
        "novelty(mu)       : "
        f"min={novelty['min_dist']:.3e}, mean={novelty['mean_dist']:.3e}, "
        f"max={novelty['max_dist']:.3e}, exact@1e-12={novelty['exact_matches_tol_1e-12']}"
    )

    fom_tag = "fom"
    prom_tag = "prom_rbf"
    hprom_tag = "hprom_rbf"

    fom_eps_file = os.path.join(args.out_dir, f"trajectory_{fom_tag}_strain.npy")
    fom_sig_file = os.path.join(args.out_dir, f"trajectory_{fom_tag}_stress.npy")
    prom_eps_file = os.path.join(args.out_dir, f"trajectory_{prom_tag}_strain.npy")
    prom_sig_file = os.path.join(args.out_dir, f"trajectory_{prom_tag}_stress.npy")
    hprom_eps_file = os.path.join(args.out_dir, f"trajectory_{hprom_tag}_strain.npy")
    hprom_sig_file = os.path.join(args.out_dir, f"trajectory_{hprom_tag}_stress.npy")

    src_fom = "cached"
    src_prom = "cached"
    src_hprom = "cached"
    t_fom = None
    t_prom = None
    t_hprom = None
    tfile_f = _timing_file(args.out_dir, fom_tag)
    tfile_p = _timing_file(args.out_dir, prom_tag)
    tfile_h = _timing_file(args.out_dir, hprom_tag)

    if args.run_fom or not (os.path.exists(fom_eps_file) and os.path.exists(fom_sig_file)):
        print("\n[RUN] FOM (iterative Newton)...")
        t0 = time.perf_counter()
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
        t_fom = float(time.perf_counter() - t0)
        _write_runtime(tfile_f, t_fom, source="new")
        src_fom = "new"
    else:
        print("\n[CACHE] Loading cached FOM.")
        eps_fom = np.load(fom_eps_file)
        sig_fom = np.load(fom_sig_file)
        t_fom = _read_runtime(tfile_f)

    if args.run_prom_rbf or not (os.path.exists(prom_eps_file) and os.path.exists(prom_sig_file)):
        print("\n[RUN] PROM-RBF (iterative reduced corrector)...")
        model_pack = LoadPromRbfModel(
            stage2a_dir=args.stage2a_dir,
            stage2b_dir=args.stage2b_dir,
            stage3_dataset_file=args.stage3_dataset_file,
            stage4_rbf_dir=args.stage4_rbf_dir,
        )
        t0 = time.perf_counter()
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
        t_prom = float(time.perf_counter() - t0)
        _write_runtime(tfile_p, t_prom, source="new")
        src_prom = "new"
    else:
        print("\n[CACHE] Loading cached PROM-RBF.")
        eps_prom = np.load(prom_eps_file)
        sig_prom = np.load(prom_sig_file)
        t_prom = _read_runtime(tfile_p)

    if args.run_hprom_rbf or not (os.path.exists(hprom_eps_file) and os.path.exists(hprom_sig_file)):
        print("\n[RUN] HPROM-RBF (classical ECM)...")
        model_pack_h, ecm_data = LoadHpromRbfModel(
            stage2a_dir=args.stage2a_dir,
            stage2b_dir=args.stage2b_dir,
            stage3_dataset_file=args.stage3_dataset_file,
            stage4_rbf_dir=args.stage4_rbf_dir,
            ecm_file=args.ecm_file,
        )
        t0 = time.perf_counter()
        eps_hprom, sig_hprom = RunHpromRbfBatchSimulation(
            params_hprom,
            model_pack=model_pack_h,
            ecm_data=ecm_data,
            strain_path=strain_path,
            out_dir=args.out_dir,
            trajectory_index=hprom_tag,
            reference_amplitude=reference_amplitude,
            reference_steps=reference_steps,
            prom_corrector_max_iters=args.hprom_corrector_iters,
            prom_corrector_rel_tol=args.hprom_corrector_rel_tol,
            prom_corrector_abs_tol=args.hprom_corrector_abs_tol,
            prom_corrector_dq_abs_tol=args.hprom_corrector_dq_abs_tol,
            prom_corrector_dq_rel_tol=args.hprom_corrector_dq_rel_tol,
            prom_corrector_res_floor_for_dq=args.hprom_corrector_res_floor_for_dq,
            prom_corrector_l2_reg=args.hprom_corrector_l2_reg,
        )
        t_hprom = float(time.perf_counter() - t0)
        _write_runtime(tfile_h, t_hprom, source="new")
        src_hprom = "new"
    else:
        print("\n[CACHE] Loading cached HPROM-RBF.")
        eps_hprom = np.load(hprom_eps_file)
        sig_hprom = np.load(hprom_sig_file)
        t_hprom = _read_runtime(tfile_h)

    n = min(len(eps_fom), len(eps_prom), len(eps_hprom), len(sig_fom), len(sig_prom), len(sig_hprom))
    eps_fom = np.asarray(eps_fom[:n], dtype=float)
    sig_fom = np.asarray(sig_fom[:n], dtype=float)
    eps_prom = np.asarray(eps_prom[:n], dtype=float)
    sig_prom = np.asarray(sig_prom[:n], dtype=float)
    eps_hprom = np.asarray(eps_hprom[:n], dtype=float)
    sig_hprom = np.asarray(sig_hprom[:n], dtype=float)

    err_prom_sig = _compute_rel_error(sig_fom, sig_prom)
    err_hprom_sig = _compute_rel_error(sig_fom, sig_hprom)
    err_prom_eps = _compute_rel_error(eps_fom, eps_prom)
    err_hprom_eps = _compute_rel_error(eps_fom, eps_hprom)
    sp_prom = (float(t_fom / t_prom) if (t_fom is not None and t_prom is not None and t_prom > 0.0) else None)
    sp_hprom = (float(t_fom / t_hprom) if (t_fom is not None and t_hprom is not None and t_hprom > 0.0) else None)

    summary = {
        "trajectory_source": args.trajectory_source,
        "trajectory_label": traj_label,
        "mapping": mapping,
        "reference_steps": int(reference_steps),
        "reference_amplitude": float(reference_amplitude),
        "n_steps_compared": int(n),
        "novelty": novelty,
        "rel_error_stress_prom_vs_fom": float(err_prom_sig),
        "rel_error_stress_hprom_vs_fom": float(err_hprom_sig),
        "rel_error_strain_prom_vs_fom": float(err_prom_eps),
        "rel_error_strain_hprom_vs_fom": float(err_hprom_eps),
        "source_fom": src_fom,
        "source_prom_rbf": src_prom,
        "source_hprom_rbf": src_hprom,
        "runtime_sec_fom": t_fom,
        "runtime_sec_prom_rbf": t_prom,
        "runtime_sec_hprom_rbf": t_hprom,
        "speedup_fom_over_prom_rbf": sp_prom,
        "speedup_fom_over_hprom_rbf": sp_hprom,
        "mesh_fom": mesh_fom,
        "mesh_prom": mesh_prom,
        "mesh_hprom": mesh_hprom,
        "mesh_hprom_mode": mesh_hprom_mode,
        "out_dir": args.out_dir,
        "ecm_file": args.ecm_file,
        "stage2a_dir": args.stage2a_dir,
        "stage2b_dir": args.stage2b_dir,
        "stage3_dataset_file": args.stage3_dataset_file,
        "stage4_rbf_dir": args.stage4_rbf_dir,
    }

    with open(os.path.join(args.out_dir, "stage7_online_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        os.path.join(args.out_dir, "stage7_online_compare_arrays.npz"),
        eps_fom=eps_fom,
        sig_fom=sig_fom,
        eps_prom=eps_prom,
        sig_prom=sig_prom,
        eps_hprom=eps_hprom,
        sig_hprom=sig_hprom,
    )

    if int(args.save_plots) == 1:
        _maybe_plot_comparisons(args.out_dir, eps_fom, sig_fom, eps_prom, sig_prom, eps_hprom, sig_hprom)
        _maybe_plot_timing(args.out_dir, t_fom, t_prom, t_hprom)

    print("\n[RESULT] Stage 7 comparison done")
    print(f"  compared steps                : {n}")
    print(f"  rel stress error PROM  vs FOM : {err_prom_sig:.3e}")
    print(f"  rel stress error HPROM vs FOM : {err_hprom_sig:.3e}")
    print(f"  rel strain error PROM  vs FOM : {err_prom_eps:.3e}")
    print(f"  rel strain error HPROM vs FOM : {err_hprom_eps:.3e}")
    if t_fom is not None:
        print(f"  runtime FOM [s]               : {t_fom:.3f}")
    if t_prom is not None:
        print(f"  runtime PROM-RBF [s]          : {t_prom:.3f}")
    if t_hprom is not None:
        print(f"  runtime HPROM-RBF [s]         : {t_hprom:.3f}")
    if sp_prom is not None:
        print(f"  speedup FOM/PROM-RBF          : {sp_prom:.2f}x")
    if sp_hprom is not None:
        print(f"  speedup FOM/HPROM-RBF         : {sp_hprom:.2f}x")
    print(f"  source FOM / PROM / HPROM     : {src_fom} / {src_prom} / {src_hprom}")
    print(f"  summary file                  : {os.path.join(args.out_dir, 'stage7_online_summary.json')}")


if __name__ == "__main__":
    main()
