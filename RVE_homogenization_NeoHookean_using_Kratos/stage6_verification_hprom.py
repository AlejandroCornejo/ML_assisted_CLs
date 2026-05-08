#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 6 verification on a selected training trajectory.

Purpose:
- Re-run PROM and HPROM on one Stage-0 trajectory index.
- Compare against FOM for the same trajectory with matching step controls.
- Report strict error metrics to verify whether HPROM recovers training paths.
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add Kratos path (same convention used in other stage scripts)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from fom_solver_rve import (
    RunFomBatchSimulation,
    LoadStrainWaypointsFromFile,
    setup_kratos_parameters,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from prom_solver_rve import RunPromBatchSimulation
from hprom_solver_rve import RunHpromBatchSimulation


def _load_linear_rom_model(model_dir="stage_2_pod_rve"):
    phi_f = np.load(os.path.join(model_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(model_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(model_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(model_dir, "eq_map.npy"))
    Xc, Yc = np.load(os.path.join(model_dir, "domain_center.npy"))
    return phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc


def _rel_l2(ref, pred):
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = min(len(ref), len(pred))
    if n == 0:
        return np.nan, 0
    num = np.linalg.norm(ref[:n] - pred[:n])
    den = np.linalg.norm(ref[:n]) + 1e-30
    return float(num / den), int(n)


def _stepwise_rel(ref, pred):
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = min(len(ref), len(pred))
    if n == 0:
        return np.zeros(0, dtype=float)
    den = np.linalg.norm(ref[:n], axis=1) + 1e-30
    num = np.linalg.norm(ref[:n] - pred[:n], axis=1)
    return num / den


def _load_q_file_if_exists(path):
    if os.path.exists(path):
        return np.asarray(np.load(path), dtype=float)
    return None


def _sync_solver_q_output(out_dir, trajectory_index, dst_name):
    src = os.path.join(out_dir, f"trajectory_{trajectory_index}_q.npy")
    if not os.path.exists(src):
        return None
    q = np.asarray(np.load(src), dtype=float)
    np.save(os.path.join(out_dir, dst_name), q)
    return q


def _load_fom_reference_from_stage1(trajectory_index, stage1_root):
    traj_dir = os.path.join(stage1_root, f"trajectory_{trajectory_index}")
    eps_file = os.path.join(traj_dir, f"trajectory_{trajectory_index}_strain.npy")
    sig_file = os.path.join(traj_dir, f"trajectory_{trajectory_index}_stress.npy")
    if os.path.exists(eps_file) and os.path.exists(sig_file):
        return np.load(eps_file), np.load(sig_file)
    return None, None


def _load_applied_strain_path_from_stage1(trajectory_index, stage1_root, first_n_steps=0):
    traj_dir = os.path.join(stage1_root, f"trajectory_{trajectory_index}")
    path_file = os.path.join(traj_dir, f"trajectory_{trajectory_index}_applied_strain.npy")
    if not os.path.exists(path_file):
        raise FileNotFoundError(path_file)
    path = np.asarray(np.load(path_file), dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        raise RuntimeError(f"Invalid applied strain path shape: {path.shape}")
    if int(first_n_steps) > 0:
        path = path[: int(first_n_steps), :]
    if path.shape[0] < 2:
        raise RuntimeError("Applied strain path must contain at least 2 points.")
    return path


def _maybe_pick_hrom_mesh(ecm_data, mesh_base, use_hrom_mesh):
    mesh_hprom = str(mesh_base)
    if use_hrom_mesh and "hrom_mesh_base" in ecm_data:
        mesh_hprom = str(np.ravel(ecm_data["hrom_mesh_base"])[0])
    return mesh_hprom


def _load_optional_gappy_operator(gappy_operator_file):
    if gappy_operator_file is None:
        return None
    if not os.path.isfile(gappy_operator_file):
        raise FileNotFoundError(gappy_operator_file)
    g = np.load(gappy_operator_file, allow_pickle=True)
    return {k: g[k] for k in g.files}


def _save_plots(out_dir, f_eps, f_sig, p_eps, p_sig, h_eps, h_sig):
    n = min(len(f_sig), len(p_sig), len(h_sig), len(f_eps), len(p_eps), len(h_eps))
    if n <= 1:
        return

    plt.rcParams.update(
        {
            "font.size": 12,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

    labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "xy"),
    ]

    for i, ylab, xlab, suffix in labels:
        plt.figure(figsize=(7, 6))
        plt.plot(f_eps[:n, i], f_sig[:n, i], "k-", label="FOM", linewidth=2.0)
        plt.plot(p_eps[:n, i], p_sig[:n, i], "r--", label="PROM", linewidth=1.5)
        plt.plot(h_eps[:n, i], h_sig[:n, i], "b:", label="HPROM", linewidth=1.6)
        plt.xlabel(f"{xlab} [-]")
        plt.ylabel(f"{ylab} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"verification_comp_{suffix}.png"), dpi=150)
        plt.close()

    err_prom = _stepwise_rel(f_sig, p_sig)
    err_hprom = _stepwise_rel(f_sig, h_sig)
    nerr = min(len(err_prom), len(err_hprom))
    plt.figure(figsize=(7, 6))
    plt.plot(err_prom[:nerr], "r-", label="PROM Error", linewidth=1.5)
    plt.plot(err_hprom[:nerr], "b-", label="HPROM Error", linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Relative stress error [-]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "verification_error_history.png"), dpi=150)
    plt.close()


def run_verification(
    trajectory_index=1,
    stage0_file="stage_0_trajectory/stage_0_trajectories.npz",
    stage1_root="stage_1_training_set_fom",
    model_dir="stage_2_pod_rve",
    ecm_file="stage_5_hprom_data/ecm_weights_all.npz",
    mesh_base="rve_geometry",
    use_hrom_mesh=False,
    run_fom=False,
    reuse_prom_cache=False,
    reuse_hprom_cache=False,
    hprom_homogenization_method="ecm_weighted",
    fom_homogenization_method="ecm_weighted_full",
    gappy_operator_file=None,
    use_stage1_applied_strain_path=False,
    first_n_steps=0,
    out_tag="",
):
    out_suffix = f"trajectory_{trajectory_index}"
    if str(out_tag).strip():
        out_suffix = f"{out_suffix}_{str(out_tag).strip()}"
    out_dir = os.path.join("stage_6_verification_hprom", out_suffix)
    os.makedirs(out_dir, exist_ok=True)

    if use_stage1_applied_strain_path:
        strain_path = _load_applied_strain_path_from_stage1(
            trajectory_index=trajectory_index,
            stage1_root=stage1_root,
            first_n_steps=first_n_steps,
        )
        # Force one step per segment so n_entries = len(strain_path)
        ref_steps = 1
        ref_amp = 1.0e12
        path_source = "stage1_applied_strain"
    else:
        strain_path, meta = LoadStrainWaypointsFromFile(stage0_file, trajectory_index=trajectory_index)
        if int(first_n_steps) > 0:
            strain_path = np.asarray(strain_path, dtype=float)[: int(first_n_steps), :]
            if strain_path.shape[0] < 2:
                raise RuntimeError("first-n-steps too small for selected Stage-0 trajectory.")
        ref_steps = int(meta.get("ref_steps", REFERENCE_STEPS_FOR_UNIT_AMPLITUDE))
        ref_amp = float(meta.get("reference_amplitude", 0.10))
        path_source = "stage0_waypoints"

    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=ref_steps,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=ref_amp,
    )
    total_steps = int(np.sum(seg_steps))
    print("=" * 72)
    print("Stage 6 Verification HPROM")
    print("=" * 72)
    print(f"Trajectory index: {trajectory_index}")
    print(f"Path source     : {path_source}")
    if use_stage1_applied_strain_path:
        print(f"Stage-1 root    : {stage1_root}")
    else:
        print(f"Trajectory file : {stage0_file}")
    print(f"Waypoints       : {len(strain_path)}")
    if int(first_n_steps) > 0:
        print(f"first_n_steps   : {int(first_n_steps)}")
    print(f"ref_steps       : {ref_steps}")
    print(f"ref_amplitude   : {ref_amp}")
    print(f"Dynamic steps   : {total_steps} (+1 initial = {total_steps + 1})")

    phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc = _load_linear_rom_model(model_dir)
    ecm_npz = np.load(ecm_file, allow_pickle=True)
    ecm_data = {k: ecm_npz[k] for k in ecm_npz.files}
    gappy_data = _load_optional_gappy_operator(gappy_operator_file)
    mesh_hprom = _maybe_pick_hrom_mesh(ecm_data, mesh_base, use_hrom_mesh)

    print(f"Base mesh       : {mesh_base}")
    print(f"HPROM mesh      : {mesh_hprom}")
    print(f"HPROM hom method: {hprom_homogenization_method}")
    print(f"FOM hom method  : {fom_homogenization_method}")
    a0_ref = None
    for k in ("A0_ref", "hom_reference_measure", "A_total"):
        if k in ecm_data:
            val = float(np.ravel(ecm_data[k])[0])
            if np.isfinite(val) and val > 0.0:
                a0_ref = val
                break
    if a0_ref is not None:
        print(f"Reference area A0: {a0_ref:.6e}")
    if str(hprom_homogenization_method).strip().lower() == "gappy_pod" and gappy_data is None:
        print(
            "[WARN] gappy_pod requested without operator file; "
            "solver will fallback unless operator keys exist in ECM npz."
        )
    if gappy_data is not None and "hom_gappy_train_rel_error_total" in gappy_data:
        print(
            "[GAPPY] offline train rel error: "
            f"total={float(np.ravel(gappy_data['hom_gappy_train_rel_error_total'])[0]):.3e}, "
            f"eps={float(np.ravel(gappy_data['hom_gappy_train_rel_error_eps'])[0]):.3e}, "
            f"sig={float(np.ravel(gappy_data['hom_gappy_train_rel_error_sig'])[0]):.3e}"
        )
    if "rel_error_res" in ecm_data:
        print(
            f"ECM offline rel_error_res={float(np.ravel(ecm_data['rel_error_res'])[0]):.3e}, "
            f"rel_error_eps={float(np.ravel(ecm_data['rel_error_eps'])[0]):.3e}, "
            f"rel_error_sig={float(np.ravel(ecm_data['rel_error_sig'])[0]):.3e}"
        )

    fom_hom_mode = str(fom_homogenization_method).strip().lower()

    if fom_hom_mode != "ecm_weighted_full":
        raise RuntimeError(
            "Only 'ecm_weighted_full' is supported for FOM homogenization in Stage 6 verification."
        )

    use_weighted_fom_hom = True
    w_fom_eps = None
    w_fom_sig = None
    if use_weighted_fom_hom:
        if "n_elem" not in ecm_data:
            raise RuntimeError(
                "ECM file does not contain 'n_elem', required for FOM weighted-full homogenization."
            )
        n_elem_ref = int(np.ravel(ecm_data["n_elem"])[0])
        if n_elem_ref <= 0:
            raise RuntimeError(f"Invalid n_elem in ECM file: {n_elem_ref}")
        w_fom_eps = np.ones(n_elem_ref, dtype=float)
        w_fom_sig = np.ones(n_elem_ref, dtype=float)
        if not run_fom:
            print(
                "[FOM] NOTE: Cached Stage-1 homogenized outputs may come from older moving-area runs. "
                "For strict reference-area consistency, a fresh FOM run will be performed "
                "with the ECM full-reference operator (all-ones weights)."
            )
            run_fom = True

    timings = {}

    # ------------------------------------------------------------------
    # FOM reference
    # ------------------------------------------------------------------
    fom_eps_file = os.path.join(out_dir, "fom_strain.npy")
    fom_sig_file = os.path.join(out_dir, "fom_stress.npy")

    f_eps = None
    f_sig = None
    if (not run_fom):
        f_eps, f_sig = _load_fom_reference_from_stage1(trajectory_index, stage1_root)
        if f_eps is not None:
            if use_stage1_applied_strain_path and int(first_n_steps) > 0:
                nclip = int(first_n_steps)
                f_eps = np.asarray(f_eps, dtype=float)[:nclip, :]
                f_sig = np.asarray(f_sig, dtype=float)[:nclip, :]
            print("[FOM] Loaded Stage-1 reference trajectory data.")
            np.save(fom_eps_file, f_eps)
            np.save(fom_sig_file, f_sig)

    if f_eps is None or run_fom:
        print("[FOM] Running fresh FOM for this trajectory...")
        params_fom = setup_kratos_parameters(mesh_base)
        t0 = time.perf_counter()
        f_eps, f_sig = RunFomBatchSimulation(
            params_fom,
            out_dir=out_dir,
            save_plot=False,
            strain_path=strain_path,
            trajectory_index=trajectory_index,
            reference_amplitude=ref_amp,
            reference_steps=ref_steps,
            hom_weights_eps_full=w_fom_eps,
            hom_weights_sig_full=w_fom_sig,
            hom_reference_measure=a0_ref,
        )
        timings["FOM"] = time.perf_counter() - t0
        f_eps = np.asarray(f_eps, dtype=float)
        f_sig = np.asarray(f_sig, dtype=float)
        np.save(fom_eps_file, f_eps)
        np.save(fom_sig_file, f_sig)

    # ------------------------------------------------------------------
    # PROM
    # ------------------------------------------------------------------
    prom_eps_file = os.path.join(out_dir, "prom_strain.npy")
    prom_sig_file = os.path.join(out_dir, "prom_stress.npy")
    prom_q_file = os.path.join(out_dir, "prom_q.npy")
    if reuse_prom_cache and os.path.exists(prom_eps_file) and os.path.exists(prom_sig_file):
        print("[PROM] Loading cached result.")
        p_eps = np.load(prom_eps_file)
        p_sig = np.load(prom_sig_file)
        q_prom = _load_q_file_if_exists(prom_q_file)
    else:
        print("[PROM] Running PROM...")
        params_prom = setup_kratos_parameters(mesh_base)
        t0 = time.perf_counter()
        p_eps, p_sig = RunPromBatchSimulation(
            params_prom,
            phi_f,
            free_dofs,
            dir_dofs,
            eq_map,
            Xc,
            Yc,
            out_dir=out_dir,
            save_plot=False,
            strain_path=strain_path,
            trajectory_index=trajectory_index,
            reference_amplitude=ref_amp,
            reference_steps=ref_steps,
        )
        timings["PROM"] = time.perf_counter() - t0
        p_eps = np.asarray(p_eps, dtype=float)
        p_sig = np.asarray(p_sig, dtype=float)
        np.save(prom_eps_file, p_eps)
        np.save(prom_sig_file, p_sig)
        q_prom = _sync_solver_q_output(out_dir, trajectory_index, "prom_q.npy")

    # ------------------------------------------------------------------
    # HPROM
    # ------------------------------------------------------------------
    method_tag = str(hprom_homogenization_method).strip().lower()
    hprom_prefix = "hprom" if method_tag == "ecm_weighted" else f"hprom_{method_tag}"
    hprom_eps_file = os.path.join(out_dir, f"{hprom_prefix}_strain.npy")
    hprom_sig_file = os.path.join(out_dir, f"{hprom_prefix}_stress.npy")
    hprom_q_file = os.path.join(out_dir, f"{hprom_prefix}_q.npy")
    if reuse_hprom_cache and os.path.exists(hprom_eps_file) and os.path.exists(hprom_sig_file):
        print("[HPROM] Loading cached result.")
        h_eps = np.load(hprom_eps_file)
        h_sig = np.load(hprom_sig_file)
        q_hprom = _load_q_file_if_exists(hprom_q_file)
    else:
        print("[HPROM] Running HPROM...")
        params_hprom = setup_kratos_parameters(mesh_hprom)
        t0 = time.perf_counter()
        h_eps, h_sig = RunHpromBatchSimulation(
            params_hprom,
            phi_f,
            free_dofs,
            dir_dofs,
            eq_map,
            Xc,
            Yc,
            ecm_data=ecm_data,
            out_dir=out_dir,
            save_plot=False,
            strain_path=strain_path,
            trajectory_index=trajectory_index,
            reference_amplitude=ref_amp,
            reference_steps=ref_steps,
            homogenization_method=hprom_homogenization_method,
            homogenization_gappy_data=gappy_data,
        )
        timings["HPROM"] = time.perf_counter() - t0
        h_eps = np.asarray(h_eps, dtype=float)
        h_sig = np.asarray(h_sig, dtype=float)
        np.save(hprom_eps_file, h_eps)
        np.save(hprom_sig_file, h_sig)
        q_hprom = _sync_solver_q_output(out_dir, trajectory_index, f"{hprom_prefix}_q.npy")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    e_prom_sig, n_ps = _rel_l2(f_sig, p_sig)
    e_hprom_sig, n_hs = _rel_l2(f_sig, h_sig)
    e_prom_eps, n_pe = _rel_l2(f_eps, p_eps)
    e_hprom_eps, n_he = _rel_l2(f_eps, h_eps)

    step_err_prom = _stepwise_rel(f_sig, p_sig)
    step_err_hprom = _stepwise_rel(f_sig, h_sig)
    e_q_hprom_prom, n_q = np.nan, 0
    q_step_err = np.zeros(0, dtype=float)
    if ("q_prom" in locals()) and ("q_hprom" in locals()) and (q_prom is not None) and (q_hprom is not None):
        e_q_hprom_prom, n_q = _rel_l2(q_prom, q_hprom)
        q_step_err = _stepwise_rel(q_prom, q_hprom)

    print("\n" + "=" * 72)
    print("Verification Summary")
    print("=" * 72)
    print(f"PROM  vs FOM | stress rel L2: {e_prom_sig:.6e} (n={n_ps})")
    print(f"HPROM vs FOM | stress rel L2: {e_hprom_sig:.6e} (n={n_hs})")
    print(f"PROM  vs FOM | strain rel L2: {e_prom_eps:.6e} (n={n_pe})")
    print(f"HPROM vs FOM | strain rel L2: {e_hprom_eps:.6e} (n={n_he})")
    if int(n_q) > 0 and np.isfinite(e_q_hprom_prom):
        print(f"HPROM vs PROM | q rel L2    : {e_q_hprom_prom:.6e} (n={n_q})")
        if q_step_err.size > 0:
            print(f"q stepwise                 : mean={np.mean(q_step_err):.6e}, max={np.max(q_step_err):.6e}")
    if step_err_hprom.size > 0:
        print(
            f"HPROM stepwise stress error | mean={np.mean(step_err_hprom):.6e}, "
            f"max={np.max(step_err_hprom):.6e}"
        )
    for k, v in timings.items():
        print(f"{k} time: {v:.2f}s")

    summary_file = os.path.join(out_dir, "verification_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Stage6 verification HPROM summary\n")
        f.write(f"trajectory_index={trajectory_index}\n")
        f.write(f"path_source={path_source}\n")
        f.write(f"stage0_file={stage0_file}\n")
        f.write(f"stage1_root={stage1_root}\n")
        f.write(f"use_stage1_applied_strain_path={bool(use_stage1_applied_strain_path)}\n")
        f.write(f"first_n_steps={int(first_n_steps)}\n")
        f.write(f"mesh_base={mesh_base}\n")
        f.write(f"mesh_hprom={mesh_hprom}\n")
        f.write(f"hprom_homogenization_method={hprom_homogenization_method}\n")
        f.write(f"fom_homogenization_method={fom_hom_mode}\n")
        if a0_ref is not None:
            f.write(f"reference_area_A0={a0_ref:.16e}\n")
        if gappy_operator_file is not None:
            f.write(f"gappy_operator_file={gappy_operator_file}\n")
        f.write(f"ref_steps={ref_steps}\n")
        f.write(f"ref_amplitude={ref_amp:.16e}\n")
        f.write(f"total_steps={total_steps}\n")
        f.write(f"entries_expected={total_steps + 1}\n")
        f.write(f"stress_rel_l2_prom={e_prom_sig:.16e}\n")
        f.write(f"stress_rel_l2_hprom={e_hprom_sig:.16e}\n")
        f.write(f"strain_rel_l2_prom={e_prom_eps:.16e}\n")
        f.write(f"strain_rel_l2_hprom={e_hprom_eps:.16e}\n")
        if int(n_q) > 0 and np.isfinite(e_q_hprom_prom):
            f.write(f"q_rel_l2_hprom_vs_prom={e_q_hprom_prom:.16e}\n")
            if q_step_err.size > 0:
                f.write(f"q_step_err_mean={np.mean(q_step_err):.16e}\n")
                f.write(f"q_step_err_max={np.max(q_step_err):.16e}\n")
        if step_err_prom.size > 0:
            f.write(f"step_err_prom_mean={np.mean(step_err_prom):.16e}\n")
            f.write(f"step_err_prom_max={np.max(step_err_prom):.16e}\n")
        if step_err_hprom.size > 0:
            f.write(f"step_err_hprom_mean={np.mean(step_err_hprom):.16e}\n")
            f.write(f"step_err_hprom_max={np.max(step_err_hprom):.16e}\n")
        for k, v in timings.items():
            f.write(f"time_{k}={v:.16e}\n")
        if "rel_error_res" in ecm_data:
            f.write(f"ecm_rel_error_res={float(np.ravel(ecm_data['rel_error_res'])[0]):.16e}\n")
            f.write(f"ecm_rel_error_eps={float(np.ravel(ecm_data['rel_error_eps'])[0]):.16e}\n")
            f.write(f"ecm_rel_error_sig={float(np.ravel(ecm_data['rel_error_sig'])[0]):.16e}\n")

    _save_plots(out_dir, f_eps, f_sig, p_eps, p_sig, h_eps, h_sig)
    print(f"\nSaved summary: {summary_file}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6 trajectory verification for HPROM.")
    parser.add_argument("--trajectory-index", type=int, default=1, help="Stage-0 trajectory index (1-based).")
    parser.add_argument(
        "--stage0-file",
        type=str,
        default="stage_0_trajectory/stage_0_trajectories.npz",
        help="Trajectory bundle file used to define verification path.",
    )
    parser.add_argument(
        "--stage1-root",
        type=str,
        default="stage_1_training_set_fom",
        help="Root directory of Stage-1 FOM trajectories.",
    )
    parser.add_argument("--model-dir", type=str, default="stage_2_pod_rve", help="Linear ROM model directory.")
    parser.add_argument("--ecm-file", type=str, default="stage_5_hprom_data/ecm_weights_all.npz", help="ECM file.")
    parser.add_argument("--mesh-base", type=str, default="rve_geometry", help="Base mesh name.")
    parser.add_argument(
        "--use-hrom-mesh",
        action="store_true",
        help="Use hrom_mesh_base from ECM file for HPROM run.",
    )
    parser.add_argument(
        "--run-fom",
        action="store_true",
        help="Force a fresh FOM run instead of using Stage-1 trajectory cache.",
    )
    parser.add_argument(
        "--reuse-prom-cache",
        action="store_true",
        help="Reuse cached PROM output in stage_6_verification_hprom if available.",
    )
    parser.add_argument(
        "--reuse-hprom-cache",
        action="store_true",
        help="Reuse cached HPROM output in stage_6_verification_hprom if available.",
    )
    parser.add_argument(
        "--hprom-homogenization-method",
        type=str,
        default="ecm_weighted",
        choices=["ecm_weighted", "gappy_pod", "kratos_reference"],
        help="Homogenization output method used by HPROM.",
    )
    parser.add_argument(
        "--fom-homogenization-method",
        type=str,
        default="ecm_weighted_full",
        choices=["ecm_weighted_full"],
        help=(
            "Reference homogenization for FOM comparison. "
            "Use ecm_weighted_full for the same reference-area operator family."
        ),
    )
    parser.add_argument(
        "--gappy-operator-file",
        type=str,
        default=None,
        help="Optional Stage5c gappy operator npz used when method is gappy_pod.",
    )
    parser.add_argument(
        "--use-stage1-applied-strain-path",
        action="store_true",
        help="Use Stage-1 trajectory_X_applied_strain.npy as path (instead of Stage-0 waypoints).",
    )
    parser.add_argument(
        "--first-n-steps",
        type=int,
        default=0,
        help="If >0, truncate selected path to first N points.",
    )
    parser.add_argument(
        "--out-tag",
        type=str,
        default="",
        help="Optional suffix added to output folder name.",
    )
    args = parser.parse_args()

    run_verification(
        trajectory_index=args.trajectory_index,
        stage0_file=args.stage0_file,
        stage1_root=args.stage1_root,
        model_dir=args.model_dir,
        ecm_file=args.ecm_file,
        mesh_base=args.mesh_base,
        use_hrom_mesh=args.use_hrom_mesh,
        run_fom=args.run_fom,
        reuse_prom_cache=args.reuse_prom_cache,
        reuse_hprom_cache=args.reuse_hprom_cache,
        hprom_homogenization_method=args.hprom_homogenization_method,
        fom_homogenization_method=args.fom_homogenization_method,
        gappy_operator_file=args.gappy_operator_file,
        use_stage1_applied_strain_path=args.use_stage1_applied_strain_path,
        first_n_steps=args.first_n_steps,
        out_tag=args.out_tag,
    )
