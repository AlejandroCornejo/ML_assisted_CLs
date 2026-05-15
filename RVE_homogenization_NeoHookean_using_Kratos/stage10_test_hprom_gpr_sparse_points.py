#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 10 (sparse-point variant):
  - Build the same full benchmark strain trajectory as stage10_test_hprom_gpr.py
  - Run HPROM-GPR only at a sparse subset of dynamic points (e.g. 20)
  - Compare against FOM sampled at the same points
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from plot_style_utils import apply_latex_plot_style

apply_latex_plot_style()

from stage6_test_hprom import generate_safe_test_path
from stage4_test_rve import plot_path_in_domain
from fom_solver_rve import (
    setup_kratos_parameters,
    RunFomBatchSimulation,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from hprom_gpr_solver_rve import LoadHpromGprModel, RunHpromGprBatchSimulation


def _load_hrom_mesh_base_from_ecm_dir(hprom_gpr_dir):
    ecm_file = os.path.join(str(hprom_gpr_dir), "ecm_weights_all.npz")
    if not os.path.exists(ecm_file):
        return None
    try:
        ecm = np.load(ecm_file, allow_pickle=True)
    except Exception:
        return None
    if "hrom_mesh_base" not in ecm:
        return None
    try:
        return str(np.ravel(ecm["hrom_mesh_base"])[0])
    except Exception:
        return None


def _compute_equivalent_stress_strain(eps, sig):
    eps = np.asarray(eps, dtype=float)
    sig = np.asarray(sig, dtype=float)
    exx = eps[:, 0]
    eyy = eps[:, 1]
    gxy = eps[:, 2]
    sxx = sig[:, 0]
    syy = sig[:, 1]
    sxy = sig[:, 2]
    sigma_eq = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
    eps_eq = (2.0 / 3.0) * np.sqrt(np.maximum(exx * exx + eyy * eyy - exx * eyy + 0.75 * gxy * gxy, 0.0))
    return eps_eq, sigma_eq


def _build_step_macro_strain_series(strain_path, seg_steps):
    e_wp = np.asarray(strain_path, dtype=float)
    seg_steps = np.asarray(seg_steps, dtype=int).reshape(-1)
    n_steps = int(np.sum(seg_steps))
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    e_hist = np.zeros((n_steps + 1, 3), dtype=float)
    n_seg = int(len(seg_steps))
    for step in range(1, n_steps + 1):
        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        e_hist[step, :] = (1.0 - xi) * e_wp[s, :3] + xi * e_wp[s + 1, :3]
    return e_hist


def _collect_stage1_snapshot_bank(stage1_dir):
    traj_dirs = sorted([d for d in glob(os.path.join(stage1_dir, "trajectory_*")) if os.path.isdir(d)])
    if not traj_dirs:
        raise RuntimeError(f"No trajectory_* directories found in {stage1_dir}")

    infos = []
    mu_blocks = []
    offsets = [0]
    for d in traj_dirs:
        e_path = os.path.join(d, f"{os.path.basename(d)}_applied_strain.npy")
        u_path = os.path.join(d, f"{os.path.basename(d)}_U.npy")
        if not (os.path.exists(e_path) and os.path.exists(u_path)):
            continue
        e = np.load(e_path, mmap_mode="r")
        if e.ndim != 2 or e.shape[1] < 3:
            continue
        n = int(e.shape[0])
        mu = np.asarray(e[:, :3], dtype=float).copy()
        # Consistent metric with trajectory-length logic: Exy = 0.5*Gxy
        mu[:, 2] *= 0.5
        mu_blocks.append(mu)
        infos.append({"dir": d, "e_path": e_path, "u_path": u_path, "n": n})
        offsets.append(offsets[-1] + n)

    if not infos:
        raise RuntimeError(f"No valid trajectory data found in {stage1_dir}")

    mu_all = np.vstack(mu_blocks)
    offsets = np.asarray(offsets, dtype=int)
    bank = {"infos": infos, "mu_all": mu_all, "offsets": offsets, "u_mmap_cache": {}}
    try:
        from scipy.spatial import cKDTree

        bank["tree"] = cKDTree(mu_all)
        bank["has_tree"] = True
    except Exception:
        bank["tree"] = None
        bank["has_tree"] = False
    return bank


def _resolve_global_index_to_local(bank, global_idx):
    off = bank["offsets"]
    j = int(np.searchsorted(off, int(global_idx), side="right") - 1)
    j = max(0, min(j, len(bank["infos"]) - 1))
    loc = int(global_idx - off[j])
    return j, loc


def _get_u_snapshot(bank, traj_id, local_idx):
    info = bank["infos"][traj_id]
    p = info["u_path"]
    if p not in bank["u_mmap_cache"]:
        bank["u_mmap_cache"][p] = np.load(p, mmap_mode="r")
    arr = bank["u_mmap_cache"][p]
    if local_idx < 0 or local_idx >= arr.shape[0]:
        raise IndexError(f"Local snapshot index {local_idx} out of bounds for {p}")
    return np.asarray(arr[local_idx, :], dtype=float)


def _query_nearest_snapshot(bank, e_target):
    mu = np.asarray(e_target, dtype=float).reshape(-1)[:3].copy()
    mu[2] *= 0.5
    if bank.get("has_tree", False):
        dist, gi = bank["tree"].query(mu, k=1)
        gi = int(gi)
        dist = float(dist)
    else:
        d = bank["mu_all"] - mu[None, :]
        d2 = np.einsum("ij,ij->i", d, d)
        gi = int(np.argmin(d2))
        dist = float(np.sqrt(d2[gi]))
    tid, loc = _resolve_global_index_to_local(bank, gi)
    u0 = _get_u_snapshot(bank, tid, loc)
    mu_nn = np.asarray(bank["mu_all"][gi, :], dtype=float).reshape(-1)
    e_nn = np.array([mu_nn[0], mu_nn[1], 2.0 * mu_nn[2]], dtype=float)
    return u0, {
        "distance": dist,
        "global_idx": gi,
        "traj_id": tid,
        "local_idx": loc,
        "nearest_macro_strain": e_nn,
    }


def _pick_sparse_step_indices(n_steps, n_points):
    n_steps = int(n_steps)
    n_points = int(max(1, n_points))
    if n_steps <= 0:
        return np.zeros(0, dtype=int)
    # select over dynamic increments [1..n_steps], keep endpoints
    idx = np.linspace(1, n_steps, num=min(n_points, n_steps), dtype=int)
    idx = np.unique(idx)
    if idx[-1] != n_steps:
        idx[-1] = n_steps
    return idx


def _plot_sparse_vs_fom(
    fom_eps_full,
    fom_sig_full,
    h_eps_sparse,
    h_sig_sparse,
    out_dir,
    fom_sparse_warm_eps=None,
    fom_sparse_warm_sig=None,
    hprom_no_newton_eps=None,
    hprom_no_newton_sig=None,
):
    n_sparse = min(len(h_sig_sparse), len(h_eps_sparse))
    n_sparse_warm = 0
    n_sparse_hn = 0
    if fom_sparse_warm_eps is not None and fom_sparse_warm_sig is not None:
        n_sparse_warm = min(len(fom_sparse_warm_eps), len(fom_sparse_warm_sig))
    if hprom_no_newton_eps is not None and hprom_no_newton_sig is not None:
        n_sparse_hn = min(len(hprom_no_newton_eps), len(hprom_no_newton_sig))
    labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "xy"),
    ]
    for comp, sy, ex, tag in labels:
        plt.figure(figsize=(7, 6))
        # Full FOM trajectory (all points)
        plt.plot(
            fom_eps_full[:, comp],
            fom_sig_full[:, comp],
            "k-",
            lw=1.6,
            label="FOM (all points)",
            zorder=1,
        )
        if n_sparse_warm > 0:
            n_pair = min(n_sparse, n_sparse_warm)
            # Pair connectors make tiny differences visible
            for i in range(1, n_pair):
                plt.plot(
                    [h_eps_sparse[i, comp], fom_sparse_warm_eps[i, comp]],
                    [h_sig_sparse[i, comp], fom_sparse_warm_sig[i, comp]],
                    color="#7f3c8d",
                    lw=0.9,
                    alpha=0.45,
                    zorder=2,
                )
            # Filled warm-FOM points
            plt.scatter(
                fom_sparse_warm_eps[:n_sparse_warm, comp],
                fom_sparse_warm_sig[:n_sparse_warm, comp],
                s=36,
                c="b",
                marker="s",
                edgecolors="white",
                linewidths=0.7,
                alpha=0.9,
                label=f"FOM sparse warm ({n_sparse_warm - 1} dynamic points)",
                zorder=4,
            )
        if n_sparse_hn > 0:
            n_pair_hn = min(n_sparse, n_sparse_hn)
            for i in range(1, n_pair_hn):
                plt.plot(
                    [h_eps_sparse[i, comp], hprom_no_newton_eps[i, comp]],
                    [h_sig_sparse[i, comp], hprom_no_newton_sig[i, comp]],
                    color="#11a579",
                    lw=0.9,
                    alpha=0.40,
                    zorder=2,
                )
            plt.scatter(
                hprom_no_newton_eps[:n_sparse_hn, comp],
                hprom_no_newton_sig[:n_sparse_hn, comp],
                s=42,
                c="#11a579",
                marker="^",
                edgecolors="white",
                linewidths=0.7,
                alpha=0.9,
                label=f"HPROM-GPR no-Newton ({n_sparse_hn - 1} dynamic points)",
                zorder=4,
            )
        # Hollow HPROM circles on top: overlap stays visible
        plt.scatter(
            h_eps_sparse[:n_sparse, comp],
            h_sig_sparse[:n_sparse, comp],
            s=86,
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            marker="o",
            label=f"HPROM-GPR ({n_sparse - 1} dynamic-point scatter)",
            zorder=5,
        )
        plt.xlabel(f"{ex} [-]")
        plt.ylabel(f"{sy} [Pa]")
        plt.title(f"Sparse-Point HPROM-GPR vs FOM: {sy}")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sparse_hprom_gpr_vs_fom_{tag}.png"), dpi=180)
        plt.close()

    # Equivalent curve: full FOM line + sparse HPROM scatter
    eps_eq_f, sig_eq_f = _compute_equivalent_stress_strain(fom_eps_full, fom_sig_full)
    eps_eq_h, sig_eq_h = _compute_equivalent_stress_strain(h_eps_sparse[:n_sparse], h_sig_sparse[:n_sparse])
    plt.figure(figsize=(7, 6))
    plt.plot(eps_eq_f, sig_eq_f, "k-", lw=1.6, label="FOM (all points)", zorder=1)
    if n_sparse_warm > 0:
        eps_eq_fw, sig_eq_fw = _compute_equivalent_stress_strain(
            fom_sparse_warm_eps[:n_sparse_warm], fom_sparse_warm_sig[:n_sparse_warm]
        )
        n_pair = min(len(eps_eq_h), len(eps_eq_fw))
        for i in range(1, n_pair):
            plt.plot(
                [eps_eq_h[i], eps_eq_fw[i]],
                [sig_eq_h[i], sig_eq_fw[i]],
                color="#7f3c8d",
                lw=0.9,
                alpha=0.45,
                zorder=2,
            )
        plt.scatter(
            eps_eq_fw,
            sig_eq_fw,
            s=38,
            c="b",
            marker="s",
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
            label=f"FOM sparse warm ({n_sparse_warm - 1} dynamic points)",
            zorder=4,
        )
    if n_sparse_hn > 0:
        eps_eq_hn, sig_eq_hn = _compute_equivalent_stress_strain(
            hprom_no_newton_eps[:n_sparse_hn], hprom_no_newton_sig[:n_sparse_hn]
        )
        n_pair_hn = min(len(eps_eq_h), len(eps_eq_hn))
        for i in range(1, n_pair_hn):
            plt.plot(
                [eps_eq_h[i], eps_eq_hn[i]],
                [sig_eq_h[i], sig_eq_hn[i]],
                color="#11a579",
                lw=0.9,
                alpha=0.40,
                zorder=2,
            )
        plt.scatter(
            eps_eq_hn,
            sig_eq_hn,
            s=42,
            c="#11a579",
            marker="^",
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
            label=f"HPROM-GPR no-Newton ({n_sparse_hn - 1} dynamic points)",
            zorder=4,
        )
    plt.scatter(
        eps_eq_h,
        sig_eq_h,
        s=88,
        facecolors="none",
        edgecolors="red",
        linewidths=1.8,
        marker="o",
        label=f"HPROM-GPR ({n_sparse - 1} dynamic-point scatter)",
        zorder=5,
    )
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.title(r"Sparse-Point HPROM-GPR vs FOM: $\sigma_{eq}$")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sparse_hprom_gpr_vs_fom_sigma_eq.png"), dpi=180)
    plt.close()


def _plot_timing_comparison(
    out_dir,
    fom_time,
    hprom_time,
    fom_sparse_warm_time=np.nan,
    hprom_no_newton_time=np.nan,
):
    labels = []
    values = []
    colors = []
    if np.isfinite(float(fom_time)):
        labels.append("FOM full")
        values.append(float(fom_time))
        colors.append("gray")
    if np.isfinite(float(fom_sparse_warm_time)):
        labels.append("FOM sparse warm")
        values.append(float(fom_sparse_warm_time))
        colors.append("blue")
    if np.isfinite(float(hprom_no_newton_time)):
        labels.append("HPROM no-Newton")
        values.append(float(hprom_no_newton_time))
        colors.append("#11a579")
    labels.append("HPROM sparse")
    values.append(float(hprom_time))
    colors.append("red")
    if len(values) == 0:
        return
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=colors, alpha=0.85)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + 0.5 * b.get_width(), v, f"{v:.2f}s", ha="center", va="bottom")
    plt.ylabel("Wall time [s]")
    plt.title("Timing Comparison")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sparse_hprom_gpr_timing_comparison.png"), dpi=180)
    plt.close()


def run_stage10_sparse_points(
    n_points=20,
    run_fom=False,
    run_fom_sparse_warm=False,
    run_hprom_gpr=True,
    run_hprom_gpr_no_newton=False,
    stage1_fom_dir="stage_1_training_set_fom",
    fom_warm_bridge_mode="adaptive",
    fom_warm_ref_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    fom_warm_ref_amplitude=None,
    gpr_data_dir="stage_7_gpr_data_ls",
    hprom_gpr_dir="stage_9_hprom_gpr_data_ls",
    out_dir="stage_10_hprom_gpr_sparse_points",
    qp_init_mode="mu_affine",
):
    os.makedirs(out_dir, exist_ok=True)

    emax = 2.0
    rel6 = [1.0, 0.05, 1.0, 0.05, 0.05, 0.05]
    domain_type = "box"
    bundle_path = "stage_0_trajectory/stage_0_trajectories.npz"
    if os.path.exists(bundle_path):
        data = np.load(bundle_path, allow_pickle=True)
        rel6 = list(data["relative_boundary"])
        if "emax" in data:
            emax = float(np.ravel(data["emax"])[0])
        else:
            emax = float(np.ravel(data["reference_amplitude"])[0])
        if "domain_type" in data:
            domain_type = str(data["domain_type"][0])

    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(waypoints, dtype=float)
    plot_path_in_domain(
        control_points,
        waypoints,
        emax,
        rel6,
        domain_type,
        os.path.join(out_dir, "sparse_hprom_gpr_test_path.png"),
    )

    seg_steps_full, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax,
    )
    n_steps_full = int(np.sum(seg_steps_full))
    e_hist_full = _build_step_macro_strain_series(strain_path, seg_steps_full)
    idx_sparse_dyn = _pick_sparse_step_indices(n_steps_full, n_points=n_points)
    idx_sparse_all = np.concatenate(([0], idx_sparse_dyn))
    strain_path_sparse = e_hist_full[idx_sparse_all, :]

    print("=" * 72)
    print("Stage 10 Sparse-Point HPROM-GPR Test")
    print("=" * 72)
    print(f"  Full dynamic steps: {n_steps_full}")
    print(f"  Sparse dynamic points requested: {int(n_points)}")
    print(f"  Sparse dynamic points used: {len(idx_sparse_dyn)}")
    print(f"  q_p init mode: {qp_init_mode}")
    print(f"  Stage1 FOM dir (warm-start bank): {stage1_fom_dir}")
    print(
        f"  FOM sparse-warm bridge: mode={fom_warm_bridge_mode}, "
        f"ref_steps={int(fom_warm_ref_steps)}, ref_amp="
        f"{'auto' if fom_warm_ref_amplitude is None else float(fom_warm_ref_amplitude)}"
    )
    print(f"  GPR data dir: {gpr_data_dir}")
    print(f"  HPROM-GPR dir: {hprom_gpr_dir}")
    print(f"  Output dir: {out_dir}")

    mesh_fom_prom = "rve_geometry"
    mesh_hprom = "rve_geometry"
    auto_hrom_mesh = _load_hrom_mesh_base_from_ecm_dir(hprom_gpr_dir)
    if auto_hrom_mesh:
        mesh_hprom = str(auto_hrom_mesh)
    print(f"  FOM mesh:   {mesh_fom_prom}")
    print(f"  HPROM mesh: {mesh_hprom}")

    parameters_fom = setup_kratos_parameters(mesh_fom_prom)
    parameters_hprom = setup_kratos_parameters(mesh_hprom)

    fom_eps_file = os.path.join(out_dir, "fom_full_strain.npy")
    fom_sig_file = os.path.join(out_dir, "fom_full_stress.npy")
    t_fom = np.nan
    if run_fom or not (os.path.exists(fom_eps_file) and os.path.exists(fom_sig_file)):
        print("\n[Stage10-Sparse] Running full FOM trajectory (reference)...")
        t0 = time.perf_counter()
        f_eps_full, f_sig_full = RunFomBatchSimulation(
            parameters_fom,
            out_dir=out_dir,
            strain_path=strain_path,
            trajectory_index=None,
            save_plot=False,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        )
        t_fom = time.perf_counter() - t0
        print(f"  FOM full time: {t_fom:.2f}s")
        f_eps_full = np.asarray(f_eps_full, dtype=float)
        f_sig_full = np.asarray(f_sig_full, dtype=float)
        np.save(fom_eps_file, f_eps_full)
        np.save(fom_sig_file, f_sig_full)
    else:
        print("\n[Stage10-Sparse] Loading cached full FOM trajectory.")
        f_eps_full = np.load(fom_eps_file)
        f_sig_full = np.load(fom_sig_file)

    f_eps_sparse = f_eps_full[idx_sparse_all, :]
    f_sig_sparse = f_sig_full[idx_sparse_all, :]
    np.save(os.path.join(out_dir, "fom_sparse_indices.npy"), idx_sparse_all)
    np.save(os.path.join(out_dir, "fom_sparse_strain.npy"), f_eps_sparse)
    np.save(os.path.join(out_dir, "fom_sparse_stress.npy"), f_sig_sparse)
    np.save(os.path.join(out_dir, "sparse_macro_strain_path.npy"), strain_path_sparse)

    t_fom_sparse_warm = np.nan
    f_eps_sparse_warm = None
    f_sig_sparse_warm = None
    nn_dist = None
    if run_fom_sparse_warm:
        print("\n[Stage10-Sparse] Running sparse pointwise FOM with nearest-snapshot warm start...")
        bank = _collect_stage1_snapshot_bank(stage1_fom_dir)
        print(
            f"  Warm-start bank: {bank['mu_all'].shape[0]} snapshots from "
            f"{len(bank['infos'])} trajectories."
        )
        eps_list = [np.zeros(3, dtype=float)]
        sig_list = [np.zeros(3, dtype=float)]
        d_list = []
        bridge_steps_list = []
        t0 = time.perf_counter()
        warm_out = os.path.join(out_dir, "fom_sparse_warm_runs")
        os.makedirs(warm_out, exist_ok=True)
        for i_dyn, step_idx in enumerate(idx_sparse_dyn, start=1):
            e_tar = e_hist_full[int(step_idx), :3]
            u0, meta = _query_nearest_snapshot(bank, e_tar)
            d_list.append(float(meta["distance"]))
            e_nn = np.asarray(meta["nearest_macro_strain"], dtype=float).reshape(3)
            mode = str(fom_warm_bridge_mode).strip().lower()
            if mode not in ("adaptive", "single_jump"):
                raise ValueError(
                    f"Unsupported fom_warm_bridge_mode='{fom_warm_bridge_mode}'. "
                    "Use 'adaptive' or 'single_jump'."
                )
            if mode == "adaptive":
                e_path = np.vstack([e_nn, e_tar.reshape(1, 3)])
                seg_steps_local, _ = BuildDynamicSegmentSteps(
                    e_path,
                    reference_steps=int(fom_warm_ref_steps),
                    min_steps=MIN_STEPS_PER_SEGMENT,
                    reference_amplitude=(
                        float(fom_warm_ref_amplitude)
                        if fom_warm_ref_amplitude is not None
                        else float(emax)
                    ),
                )
                n_bridge_steps = int(np.sum(seg_steps_local))
                if n_bridge_steps <= 0:
                    n_bridge_steps = 1
                ref_amp_local = (
                    float(fom_warm_ref_amplitude)
                    if fom_warm_ref_amplitude is not None
                    else float(emax)
                )
                ref_steps_local = int(fom_warm_ref_steps)
            else:
                e_path = np.vstack([np.zeros(3, dtype=float), e_tar.reshape(1, 3)])
                n_bridge_steps = 1
                ref_amp_local = 1.0e12
                ref_steps_local = 1
            bridge_steps_list.append(int(n_bridge_steps))
            print(
                f"  [FOM sparse warm] point {i_dyn:03d}/{len(idx_sparse_dyn)} | "
                f"step={int(step_idx)} | nn_dist={float(meta['distance']):.3e} | "
                f"bridge_steps={int(n_bridge_steps)}"
            )
            e_h, s_h = RunFomBatchSimulation(
                parameters_fom,
                out_dir=warm_out,
                save_results=False,
                save_plot=False,
                strain_path=e_path,
                trajectory_index=None,
                reference_amplitude=ref_amp_local,
                reference_steps=ref_steps_local,
                initial_displacement=u0,
            )
            eps_list.append(np.asarray(e_h, dtype=float)[-1, :3])
            sig_list.append(np.asarray(s_h, dtype=float)[-1, :3])
        t_fom_sparse_warm = time.perf_counter() - t0
        f_eps_sparse_warm = np.asarray(eps_list, dtype=float)
        f_sig_sparse_warm = np.asarray(sig_list, dtype=float)
        nn_dist = np.asarray(d_list, dtype=float)
        bridge_steps_arr = np.asarray(bridge_steps_list, dtype=int)
        np.save(os.path.join(out_dir, "fom_sparse_warm_strain.npy"), f_eps_sparse_warm)
        np.save(os.path.join(out_dir, "fom_sparse_warm_stress.npy"), f_sig_sparse_warm)
        np.save(os.path.join(out_dir, "fom_sparse_warm_nn_dist.npy"), nn_dist)
        np.save(os.path.join(out_dir, "fom_sparse_warm_bridge_steps.npy"), bridge_steps_arr)
        print(f"  FOM sparse warm runtime: {t_fom_sparse_warm:.2f}s")

    if not run_hprom_gpr:
        print("\n[Stage10-Sparse] HPROM-GPR run skipped by flag.")
        return

    print("\n[Stage10-Sparse] Running HPROM-GPR on sparse points only...")
    (
        phi_p_h,
        phi_s_h,
        free_dofs_h,
        _dir_dofs_h,
        eq_map_h,
        Xc_h,
        Yc_h,
        gpr_model_h,
        ecm_data_h,
        _include_macro_h,
    ) = LoadHpromGprModel(
        basis_dir="stage_2_pod_rve",
        gpr_data_dir=gpr_data_dir,
        hprom_gpr_dir=hprom_gpr_dir,
    )
    t0 = time.perf_counter()
    h_eps_sparse, h_sig_sparse = RunHpromGprBatchSimulation(
        parameters_hprom,
        phi_p_h,
        phi_s_h,
        free_dofs_h,
        gpr_model_h,
        ecm_data_h,
        out_dir=out_dir,
        strain_path=strain_path_sparse,
        trajectory_index=None,
        # Force one dynamic solve per sparse segment:
        reference_amplitude=1.0e12,
        reference_steps=1,
        eq_map_full=eq_map_h,
        Xc=Xc_h,
        Yc=Yc_h,
        qp_init_mode=qp_init_mode,
    )
    t_h = time.perf_counter() - t0
    h_eps_sparse = np.asarray(h_eps_sparse, dtype=float)
    h_sig_sparse = np.asarray(h_sig_sparse, dtype=float)
    np.save(os.path.join(out_dir, "hprom_sparse_strain.npy"), h_eps_sparse)
    np.save(os.path.join(out_dir, "hprom_sparse_stress.npy"), h_sig_sparse)

    n = min(len(f_sig_sparse), len(h_sig_sparse))
    err_sig = float(np.linalg.norm(f_sig_sparse[:n] - h_sig_sparse[:n]) / max(np.linalg.norm(f_sig_sparse[:n]), 1e-30))
    err_eps = float(np.linalg.norm(f_eps_sparse[:n] - h_eps_sparse[:n]) / max(np.linalg.norm(f_eps_sparse[:n]), 1e-30))
    f_eps_eq_sparse, f_sig_eq_sparse = _compute_equivalent_stress_strain(f_eps_sparse[:n], f_sig_sparse[:n])
    h_eps_eq_sparse, h_sig_eq_sparse = _compute_equivalent_stress_strain(h_eps_sparse[:n], h_sig_sparse[:n])
    err_sig_eq = float(np.linalg.norm(f_sig_eq_sparse - h_sig_eq_sparse) / max(np.linalg.norm(f_sig_eq_sparse), 1e-30))
    err_eps_eq = float(np.linalg.norm(f_eps_eq_sparse - h_eps_eq_sparse) / max(np.linalg.norm(f_eps_eq_sparse), 1e-30))
    hprom_time_per_dynamic = float(t_h / max(len(idx_sparse_dyn), 1))
    fom_time_per_dynamic = float(t_fom / max(n_steps_full, 1)) if np.isfinite(float(t_fom)) else np.nan
    fom_sparse_warm_time_per_dynamic = (
        float(t_fom_sparse_warm / max(len(idx_sparse_dyn), 1))
        if np.isfinite(float(t_fom_sparse_warm))
        else np.nan
    )
    err_sig_fw = np.nan
    err_eps_fw = np.nan
    err_sig_eq_fw = np.nan
    err_eps_eq_fw = np.nan
    if f_eps_sparse_warm is not None and f_sig_sparse_warm is not None:
        nfw = min(len(f_sig_sparse), len(f_sig_sparse_warm))
        err_sig_fw = float(
            np.linalg.norm(f_sig_sparse[:nfw] - f_sig_sparse_warm[:nfw])
            / max(np.linalg.norm(f_sig_sparse[:nfw]), 1e-30)
        )
        err_eps_fw = float(
            np.linalg.norm(f_eps_sparse[:nfw] - f_eps_sparse_warm[:nfw])
            / max(np.linalg.norm(f_eps_sparse[:nfw]), 1e-30)
        )
        fweps_eq, fwsig_eq = _compute_equivalent_stress_strain(
            f_eps_sparse_warm[:nfw], f_sig_sparse_warm[:nfw]
        )
        feps_eq_ref, fsig_eq_ref = _compute_equivalent_stress_strain(
            f_eps_sparse[:nfw], f_sig_sparse[:nfw]
        )
        err_sig_eq_fw = float(
            np.linalg.norm(fsig_eq_ref - fwsig_eq) / max(np.linalg.norm(fsig_eq_ref), 1e-30)
        )
        err_eps_eq_fw = float(
            np.linalg.norm(feps_eq_ref - fweps_eq) / max(np.linalg.norm(feps_eq_ref), 1e-30)
        )

    t_hn = np.nan
    h_eps_sparse_no_newton = None
    h_sig_sparse_no_newton = None
    err_sig_hn = np.nan
    err_eps_hn = np.nan
    err_sig_eq_hn = np.nan
    err_eps_eq_hn = np.nan
    err_sig_hn_vs_h = np.nan
    err_eps_hn_vs_h = np.nan
    if run_hprom_gpr_no_newton:
        print("\n[Stage10-Sparse] Running HPROM-GPR predictor-only (no Newton iterations)...")
        t0 = time.perf_counter()
        h_eps_sparse_no_newton, h_sig_sparse_no_newton = RunHpromGprBatchSimulation(
            parameters_hprom,
            phi_p_h,
            phi_s_h,
            free_dofs_h,
            gpr_model_h,
            ecm_data_h,
            out_dir=out_dir,
            strain_path=strain_path_sparse,
            trajectory_index=None,
            reference_amplitude=1.0e12,
            reference_steps=1,
            eq_map_full=eq_map_h,
            Xc=Xc_h,
            Yc=Yc_h,
            qp_init_mode=qp_init_mode,
            max_its=0,
        )
        t_hn = time.perf_counter() - t0
        h_eps_sparse_no_newton = np.asarray(h_eps_sparse_no_newton, dtype=float)
        h_sig_sparse_no_newton = np.asarray(h_sig_sparse_no_newton, dtype=float)
        np.save(os.path.join(out_dir, "hprom_sparse_no_newton_strain.npy"), h_eps_sparse_no_newton)
        np.save(os.path.join(out_dir, "hprom_sparse_no_newton_stress.npy"), h_sig_sparse_no_newton)
        nhn = min(len(f_sig_sparse), len(h_sig_sparse_no_newton))
        err_sig_hn = float(
            np.linalg.norm(f_sig_sparse[:nhn] - h_sig_sparse_no_newton[:nhn])
            / max(np.linalg.norm(f_sig_sparse[:nhn]), 1e-30)
        )
        err_eps_hn = float(
            np.linalg.norm(f_eps_sparse[:nhn] - h_eps_sparse_no_newton[:nhn])
            / max(np.linalg.norm(f_eps_sparse[:nhn]), 1e-30)
        )
        feps_eq_hn_ref, fsig_eq_hn_ref = _compute_equivalent_stress_strain(
            f_eps_sparse[:nhn], f_sig_sparse[:nhn]
        )
        heps_eq_hn, hsig_eq_hn = _compute_equivalent_stress_strain(
            h_eps_sparse_no_newton[:nhn], h_sig_sparse_no_newton[:nhn]
        )
        err_sig_eq_hn = float(
            np.linalg.norm(fsig_eq_hn_ref - hsig_eq_hn) / max(np.linalg.norm(fsig_eq_hn_ref), 1e-30)
        )
        err_eps_eq_hn = float(
            np.linalg.norm(feps_eq_hn_ref - heps_eq_hn) / max(np.linalg.norm(feps_eq_hn_ref), 1e-30)
        )
        nh_cmp = min(len(h_sig_sparse), len(h_sig_sparse_no_newton))
        err_sig_hn_vs_h = float(
            np.linalg.norm(h_sig_sparse[:nh_cmp] - h_sig_sparse_no_newton[:nh_cmp])
            / max(np.linalg.norm(h_sig_sparse[:nh_cmp]), 1e-30)
        )
        err_eps_hn_vs_h = float(
            np.linalg.norm(h_eps_sparse[:nh_cmp] - h_eps_sparse_no_newton[:nh_cmp])
            / max(np.linalg.norm(h_eps_sparse[:nh_cmp]), 1e-30)
        )

    with open(os.path.join(out_dir, "sparse_hprom_gpr_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"full_dynamic_steps={n_steps_full}\n")
        f.write(f"sparse_dynamic_points={len(idx_sparse_dyn)}\n")
        f.write(f"qp_init_mode={qp_init_mode}\n")
        f.write(f"fom_full_runtime_s={t_fom:.16e}\n")
        f.write(f"hprom_sparse_runtime_s={t_h:.16e}\n")
        f.write(f"hprom_sparse_no_newton_runtime_s={t_hn:.16e}\n")
        f.write(f"fom_time_per_dynamic_s={fom_time_per_dynamic:.16e}\n")
        f.write(f"fom_sparse_warm_runtime_s={t_fom_sparse_warm:.16e}\n")
        f.write(f"fom_sparse_warm_time_per_dynamic_s={fom_sparse_warm_time_per_dynamic:.16e}\n")
        f.write(f"hprom_time_per_dynamic_s={hprom_time_per_dynamic:.16e}\n")
        f.write(f"rel_stress_error_vs_fom_sparse={err_sig:.16e}\n")
        f.write(f"rel_strain_error_vs_fom_sparse={err_eps:.16e}\n")
        f.write(f"rel_sigma_eq_error_vs_fom_sparse={err_sig_eq:.16e}\n")
        f.write(f"rel_eps_eq_error_vs_fom_sparse={err_eps_eq:.16e}\n")
        f.write(f"rel_stress_error_hprom_no_newton_vs_fom_sparse={err_sig_hn:.16e}\n")
        f.write(f"rel_strain_error_hprom_no_newton_vs_fom_sparse={err_eps_hn:.16e}\n")
        f.write(f"rel_sigma_eq_error_hprom_no_newton_vs_fom_sparse={err_sig_eq_hn:.16e}\n")
        f.write(f"rel_eps_eq_error_hprom_no_newton_vs_fom_sparse={err_eps_eq_hn:.16e}\n")
        f.write(f"rel_stress_error_hprom_no_newton_vs_hprom={err_sig_hn_vs_h:.16e}\n")
        f.write(f"rel_strain_error_hprom_no_newton_vs_hprom={err_eps_hn_vs_h:.16e}\n")
        f.write(f"rel_stress_error_fom_sparse_warm_vs_fom_sparse={err_sig_fw:.16e}\n")
        f.write(f"rel_strain_error_fom_sparse_warm_vs_fom_sparse={err_eps_fw:.16e}\n")
        f.write(f"rel_sigma_eq_error_fom_sparse_warm_vs_fom_sparse={err_sig_eq_fw:.16e}\n")
        f.write(f"rel_eps_eq_error_fom_sparse_warm_vs_fom_sparse={err_eps_eq_fw:.16e}\n")
        if nn_dist is not None and nn_dist.size > 0:
            f.write(f"warm_nn_dist_mean={float(np.mean(nn_dist)):.16e}\n")
            f.write(f"warm_nn_dist_max={float(np.max(nn_dist)):.16e}\n")
            f.write(f"warm_nn_dist_min={float(np.min(nn_dist)):.16e}\n")
        if run_fom_sparse_warm and 'bridge_steps_arr' in locals() and bridge_steps_arr.size > 0:
            f.write(f"warm_bridge_steps_mean={float(np.mean(bridge_steps_arr)):.16e}\n")
            f.write(f"warm_bridge_steps_max={float(np.max(bridge_steps_arr)):.16e}\n")
            f.write(f"warm_bridge_steps_min={float(np.min(bridge_steps_arr)):.16e}\n")
            f.write(f"warm_bridge_steps_total={int(np.sum(bridge_steps_arr))}\n")
        f.write("sparse_indices_dynamic=" + ",".join([str(int(v)) for v in idx_sparse_dyn]) + "\n")

    print("\n" + "=" * 72)
    print("Sparse-Point Summary")
    print("=" * 72)
    if np.isfinite(float(t_fom)):
        print(f"  FOM full runtime: {t_fom:.2f}s")
        print(f"  FOM time per dynamic step: {fom_time_per_dynamic:.4f}s")
    else:
        print("  FOM full runtime: cache loaded (run with --run-fom for timing).")
    if np.isfinite(float(t_fom_sparse_warm)):
        print(f"  FOM sparse warm runtime: {t_fom_sparse_warm:.2f}s")
        print(f"  FOM sparse warm time per dynamic point: {fom_sparse_warm_time_per_dynamic:.4f}s")
    print(f"  HPROM sparse runtime: {t_h:.2f}s")
    print(f"  HPROM time per sparse dynamic point: {hprom_time_per_dynamic:.4f}s")
    print(f"  Rel stress error (sparse vs sampled FOM): {err_sig:.4e}")
    print(f"  Rel strain error (sparse vs sampled FOM): {err_eps:.4e}")
    print(f"  Rel sigma_eq error (sparse vs sampled FOM): {err_sig_eq:.4e}")
    print(f"  Rel eps_eq error (sparse vs sampled FOM): {err_eps_eq:.4e}")
    if np.isfinite(float(t_hn)):
        print(f"  HPROM no-Newton runtime: {t_hn:.2f}s")
    if np.isfinite(float(err_sig_hn)):
        print(f"  Rel stress error (HPROM no-Newton vs sampled FOM): {err_sig_hn:.4e}")
        print(f"  Rel strain error (HPROM no-Newton vs sampled FOM): {err_eps_hn:.4e}")
        print(f"  Rel sigma_eq error (HPROM no-Newton vs sampled FOM): {err_sig_eq_hn:.4e}")
        print(f"  Rel eps_eq error (HPROM no-Newton vs sampled FOM): {err_eps_eq_hn:.4e}")
        print(f"  Rel stress error (HPROM no-Newton vs HPROM): {err_sig_hn_vs_h:.4e}")
        print(f"  Rel strain error (HPROM no-Newton vs HPROM): {err_eps_hn_vs_h:.4e}")
    if np.isfinite(float(err_sig_fw)):
        print(f"  Rel stress error (FOM sparse warm vs sampled FOM): {err_sig_fw:.4e}")
        print(f"  Rel strain error (FOM sparse warm vs sampled FOM): {err_eps_fw:.4e}")
        print(f"  Rel sigma_eq error (FOM sparse warm vs sampled FOM): {err_sig_eq_fw:.4e}")
        print(f"  Rel eps_eq error (FOM sparse warm vs sampled FOM): {err_eps_eq_fw:.4e}")
        if nn_dist is not None and nn_dist.size > 0:
            print(
                f"  Warm-start NN distance: mean={float(np.mean(nn_dist)):.3e}, "
                f"max={float(np.max(nn_dist)):.3e}"
            )
        if 'bridge_steps_arr' in locals() and bridge_steps_arr.size > 0:
            print(
                f"  Warm bridge steps: mean={float(np.mean(bridge_steps_arr)):.2f}, "
                f"max={int(np.max(bridge_steps_arr))}, total={int(np.sum(bridge_steps_arr))}"
            )

    _plot_sparse_vs_fom(
        f_eps_full,
        f_sig_full,
        h_eps_sparse,
        h_sig_sparse,
        out_dir=out_dir,
        fom_sparse_warm_eps=f_eps_sparse_warm,
        fom_sparse_warm_sig=f_sig_sparse_warm,
        hprom_no_newton_eps=h_eps_sparse_no_newton,
        hprom_no_newton_sig=h_sig_sparse_no_newton,
    )
    _plot_timing_comparison(
        out_dir,
        fom_time=t_fom,
        hprom_time=t_h,
        fom_sparse_warm_time=t_fom_sparse_warm,
        hprom_no_newton_time=t_hn,
    )
    print(f"  Saved outputs in: {out_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Stage 10 sparse-point HPROM-GPR test (HPROM solved only on selected macro points)."
    )
    p.add_argument("--n-points", type=int, default=20, help="Number of dynamic points sampled from full 1150-step path.")
    p.add_argument("--run-fom", action="store_true", help="Force full FOM recompute.")
    p.add_argument(
        "--run-fom-sparse-warm",
        action="store_true",
        help="Run pointwise sparse FOM initialized from nearest Stage-1 snapshot in mu-space.",
    )
    p.add_argument("--run-hprom-gpr", action="store_true", help="Run sparse HPROM-GPR solve.")
    p.add_argument(
        "--run-hprom-gpr-no-newton",
        action="store_true",
        help="Run sparse HPROM-GPR predictor-only (no Newton iterations).",
    )
    p.add_argument(
        "--stage1-fom-dir",
        type=str,
        default="stage_1_training_set_fom",
        help="Stage-1 trajectory directory used to build nearest-snapshot warm-start bank.",
    )
    p.add_argument(
        "--fom-warm-bridge-mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "single_jump"],
        help="Adaptive: nearest-snapshot->target bridge with ref-steps logic. single_jump: one-step jump.",
    )
    p.add_argument(
        "--fom-warm-ref-steps",
        type=int,
        default=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        help="Reference steps used for adaptive bridge allocation.",
    )
    p.add_argument(
        "--fom-warm-ref-amplitude",
        type=float,
        default=None,
        help="Reference amplitude for adaptive bridge (default: stage emax).",
    )
    p.add_argument("--gpr-data-dir", type=str, default="stage_7_gpr_data_ls")
    p.add_argument("--hprom-gpr-dir", type=str, default="stage_9_hprom_gpr_data_ls")
    p.add_argument("--out-dir", type=str, default="stage_10_hprom_gpr_sparse_points")
    p.add_argument(
        "--qp-init-mode",
        type=str,
        default="mu_affine",
        choices=["previous", "zero", "mu_affine"],
        help="Initializer for q_p at each sparse point.",
    )
    args = p.parse_args()

    run_stage10_sparse_points(
        n_points=args.n_points,
        run_fom=args.run_fom,
        run_fom_sparse_warm=args.run_fom_sparse_warm,
        run_hprom_gpr=args.run_hprom_gpr,
        run_hprom_gpr_no_newton=args.run_hprom_gpr_no_newton,
        stage1_fom_dir=args.stage1_fom_dir,
        fom_warm_bridge_mode=args.fom_warm_bridge_mode,
        fom_warm_ref_steps=args.fom_warm_ref_steps,
        fom_warm_ref_amplitude=args.fom_warm_ref_amplitude,
        gpr_data_dir=args.gpr_data_dir,
        hprom_gpr_dir=args.hprom_gpr_dir,
        out_dir=args.out_dir,
        qp_init_mode=args.qp_init_mode,
    )
