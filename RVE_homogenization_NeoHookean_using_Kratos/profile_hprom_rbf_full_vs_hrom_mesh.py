#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Short-path diagnostic for HPROM-RBF:
compare full-mesh ECM vs HROM-mesh ECM with identical nonlinear settings.
"""

import argparse
import contextlib
import io
import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt

from stage6_test_hprom import generate_safe_test_path
from fom_solver_rve import (
    setup_kratos_parameters,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from hprom_rbf_solver_rve import LoadHpromRbfModel, RunHpromRbfBatchSimulation


def _load_stage11_domain_inputs():
    emax = 2.0
    rel6 = [1.0, 0.05, 1.0, 0.05, 0.05, 0.05]
    domain_type = "box"
    bundle_path = "stage_0_trajectory/stage_0_trajectories.npz"
    if os.path.exists(bundle_path):
        data = np.load(bundle_path, allow_pickle=True)
        rel6 = list(data["relative_boundary"])
        if "emax" in data:
            emax = float(np.ravel(data["emax"])[0])
        elif "reference_amplitude" in data:
            emax = float(np.ravel(data["reference_amplitude"])[0])
        if "domain_type" in data:
            domain_type = str(np.ravel(data["domain_type"])[0])
    return emax, rel6, domain_type


def _parse_iteration_stats(log_text):
    it_re = re.compile(r"^\s+>\s+It\s+(\d+):")
    n_iters = 0
    for line in log_text.splitlines():
        if it_re.search(line):
            n_iters += 1
    return int(n_iters)


def _run_case(
    tag,
    mesh_name,
    ecm_data,
    strain_path,
    phi_p,
    phi_s,
    free_dofs,
    eq_map_full,
    rbf_model,
    include_macro,
    Xc,
    Yc,
    out_dir,
    max_its,
    ref_amp,
    ref_steps,
    use_old_stiffness,
    expected_steps,
):
    params = setup_kratos_parameters(mesh_name)
    run_out_dir = os.path.join(out_dir, tag)
    os.makedirs(run_out_dir, exist_ok=True)

    log_stream = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(log_stream):
        eps, sig = RunHpromRbfBatchSimulation(
            params,
            phi_p,
            phi_s,
            free_dofs,
            rbf_model,
            ecm_data,
            out_dir=run_out_dir,
            strain_path=strain_path,
            trajectory_index=None,
            include_macro_strain_input=include_macro,
            reference_amplitude=ref_amp,
            reference_steps=ref_steps,
            max_its=max_its,
            use_old_stiffness_in_first_iteration=use_old_stiffness,
            eq_map_full=eq_map_full,
            Xc=Xc,
            Yc=Yc,
        )
    elapsed = time.perf_counter() - t0

    log_text = log_stream.getvalue()
    log_path = os.path.join(run_out_dir, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)

    n_iters = _parse_iteration_stats(log_text)
    eps = np.asarray(eps, dtype=float)
    sig = np.asarray(sig, dtype=float)
    np.save(os.path.join(run_out_dir, "strain.npy"), eps)
    np.save(os.path.join(run_out_dir, "stress.npy"), sig)
    q_path = os.path.join(run_out_dir, "hprom_rbf_run_q_p.npy")
    q_hist = np.asarray(np.load(q_path), dtype=float) if os.path.exists(q_path) else None
    avg_iters = (float(n_iters) / float(expected_steps)) if int(expected_steps) > 0 else 0.0

    return {
        "eps": eps,
        "sig": sig,
        "q": q_hist,
        "time": float(elapsed),
        "n_steps": int(expected_steps),
        "n_iters": int(n_iters),
        "avg_iters": float(avg_iters),
        "log_path": log_path,
    }


def _save_q_diagnostics(q_full, q_hrom, out_dir):
    if q_full is None or q_hrom is None:
        return None

    qf = np.asarray(q_full, dtype=float)
    qh = np.asarray(q_hrom, dtype=float)
    n = min(len(qf), len(qh))
    if n <= 0:
        return None
    qf = qf[:n]
    qh = qh[:n]

    dq = qh - qf
    qf_norm = np.linalg.norm(qf, axis=1) + 1e-30
    dq_norm = np.linalg.norm(dq, axis=1)
    rel_step = dq_norm / qf_norm

    np.save(os.path.join(out_dir, "q_full.npy"), qf)
    np.save(os.path.join(out_dir, "q_hrom.npy"), qh)
    np.save(os.path.join(out_dir, "q_diff.npy"), dq)
    np.save(os.path.join(out_dir, "q_rel_error_per_step.npy"), rel_step)
    np.save(os.path.join(out_dir, "q_abs_error_per_step.npy"), dq_norm)

    plt.figure(figsize=(8, 5))
    plt.plot(rel_step, "m-", linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Relative q error [-]")
    plt.title("HPROM-RBF: per-step q error (HROM vs Full)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "q_rel_error_vs_step.png"), dpi=150)
    plt.close()

    k_show = int(min(5, qf.shape[1]))
    if k_show > 0:
        plt.figure(figsize=(10, 6))
        for i in range(k_show):
            plt.plot(qf[:, i], "-", linewidth=1.2, label=f"q[{i}] full")
            plt.plot(qh[:, i], "--", linewidth=1.2, label=f"q[{i}] hrom")
        plt.xlabel("Step")
        plt.ylabel("q component [-]")
        plt.title(f"HPROM-RBF: first {k_show} q components")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "q_components_first5.png"), dpi=150)
        plt.close()

    max_idx = int(np.argmax(rel_step))
    return {
        "n_steps_q": int(n),
        "max_rel_q_error": float(np.max(rel_step)),
        "max_rel_q_error_step": max_idx,
        "final_rel_q_error": float(rel_step[-1]),
        "mean_rel_q_error": float(np.mean(rel_step)),
        "rel_step": rel_step,
    }


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


def _save_stress_strain_diagnostics(eps_full, sig_full, eps_hrom, sig_hrom, out_dir):
    ef = np.asarray(eps_full, dtype=float)
    sf = np.asarray(sig_full, dtype=float)
    eh = np.asarray(eps_hrom, dtype=float)
    sh = np.asarray(sig_hrom, dtype=float)
    n = min(len(ef), len(sf), len(eh), len(sh))
    if n <= 1:
        return None

    ef = ef[:n]
    sf = sf[:n]
    eh = eh[:n]
    sh = sh[:n]

    np.save(os.path.join(out_dir, "strain_full.npy"), ef)
    np.save(os.path.join(out_dir, "stress_full.npy"), sf)
    np.save(os.path.join(out_dir, "strain_hrom.npy"), eh)
    np.save(os.path.join(out_dir, "stress_hrom.npy"), sh)

    eps_eq_f, sig_eq_f = _compute_equivalent_stress_strain(ef, sf)
    eps_eq_h, sig_eq_h = _compute_equivalent_stress_strain(eh, sh)

    plt.figure(figsize=(7.2, 6.0))
    plt.plot(eps_eq_f, sig_eq_f, "k-", linewidth=2.0, label="Full-mesh ECM")
    plt.plot(eps_eq_h, sig_eq_h, "m--", linewidth=1.8, label="HROM-mesh ECM")
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.title(r"HPROM-RBF: $\sigma_{eq}$ vs $\varepsilon_{eq}$")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comp_sigma_eq.png"), dpi=150)
    plt.close()

    comp = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "xy"),
    ]
    for i, ylab, xlab, tag in comp:
        plt.figure(figsize=(7.2, 6.0))
        plt.plot(ef[:, i], sf[:, i], "k-", linewidth=2.0, label="Full-mesh ECM")
        plt.plot(eh[:, i], sh[:, i], "m--", linewidth=1.8, label="HROM-mesh ECM")
        plt.xlabel(f"{xlab} [-]")
        plt.ylabel(f"{ylab} [Pa]")
        plt.title(f"HPROM-RBF: {ylab}")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"comp_sigma_{tag}.png"), dpi=150)
        plt.close()

    stress_rel_step = np.linalg.norm(sf - sh, axis=1) / (np.linalg.norm(sf, axis=1) + 1e-30)
    strain_rel_step = np.linalg.norm(ef - eh, axis=1) / (np.linalg.norm(ef, axis=1) + 1e-30)
    np.save(os.path.join(out_dir, "stress_rel_error_per_step.npy"), stress_rel_step)
    np.save(os.path.join(out_dir, "strain_rel_error_per_step.npy"), strain_rel_step)

    plt.figure(figsize=(8, 5))
    plt.plot(stress_rel_step, "r-", linewidth=1.6, label="stress rel. error")
    plt.plot(strain_rel_step, "b-", linewidth=1.6, label="strain rel. error")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Relative error [-]")
    plt.title("HPROM-RBF: stepwise state error (HROM vs Full)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "state_rel_error_vs_step.png"), dpi=150)
    plt.close()

    return {
        "stress_rel_step": stress_rel_step,
        "strain_rel_step": strain_rel_step,
        "mean_stress_rel": float(np.mean(stress_rel_step)),
        "max_stress_rel": float(np.max(stress_rel_step)),
        "mean_strain_rel": float(np.mean(strain_rel_step)),
        "max_strain_rel": float(np.max(strain_rel_step)),
    }


def _plot_multi_segment_q_error(out_dir, per_segment_stats):
    valid = [s for s in per_segment_stats if s.get("q_rel_step") is not None]
    if not valid:
        return

    plt.figure(figsize=(9, 5))
    for s in valid:
        rel_step = np.asarray(s["q_rel_step"], dtype=float)
        plt.plot(rel_step, linewidth=1.5, label=str(s.get("run_label", s["segment_index"])))
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Relative q error [-]")
    plt.title("HPROM-RBF: per-step q error by segment (HROM vs Full)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "q_rel_error_vs_step_all_segments.png"), dpi=150)
    plt.close()


def _run_segment(args, seg_idx_start, seg_idx_end, full_path, model_data):
    strain_path = full_path[seg_idx_start:seg_idx_end + 2].copy()
    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=int(args.ref_steps),
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=float(args.reference_amplitude),
    )
    n_dyn = int(np.sum(seg_steps))

    (
        phi_p,
        phi_s,
        free_dofs,
        eq_map_full,
        Xc,
        Yc,
        rbf_model,
        ecm_full,
        ecm_hrom,
        include_macro,
        mesh_hrom,
    ) = model_data

    run_label = f"seg_{seg_idx_start:03d}_{seg_idx_end:03d}"
    seg_out_dir = os.path.join(args.out_dir, run_label)
    os.makedirs(seg_out_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"HPROM-RBF profile: segments {seg_idx_start}..{seg_idx_end}")
    print("=" * 72)
    print(f"Path points used             : {len(strain_path)}")
    print(f"Path segments used           : {seg_idx_end - seg_idx_start + 1}")
    print(f"Dynamic steps                : {n_dyn}")
    print(f"Reference amplitude          : {float(args.reference_amplitude):.6e}")
    print(f"Ref steps                    : {int(args.ref_steps)}")
    print(f"Max Newton iterations        : {int(args.max_its)}")
    print(f"Use old stiffness (first it) : {not bool(args.disable_old_stiffness)}")
    print(f"HROM mesh                    : {mesh_hrom}")

    res_full = _run_case(
        tag="full_mesh_ecm",
        mesh_name="rve_geometry",
        ecm_data=ecm_full,
        strain_path=strain_path,
        phi_p=phi_p,
        phi_s=phi_s,
        free_dofs=free_dofs,
        eq_map_full=eq_map_full,
        rbf_model=rbf_model,
        include_macro=include_macro,
        Xc=Xc,
        Yc=Yc,
        out_dir=seg_out_dir,
        max_its=int(args.max_its),
        ref_amp=float(args.reference_amplitude),
        ref_steps=int(args.ref_steps),
        use_old_stiffness=not bool(args.disable_old_stiffness),
        expected_steps=n_dyn,
    )

    res_hrom = _run_case(
        tag="hrom_mesh_ecm",
        mesh_name=mesh_hrom,
        ecm_data=ecm_hrom,
        strain_path=strain_path,
        phi_p=phi_p,
        phi_s=phi_s,
        free_dofs=free_dofs,
        eq_map_full=eq_map_full,
        rbf_model=rbf_model,
        include_macro=include_macro,
        Xc=Xc,
        Yc=Yc,
        out_dir=seg_out_dir,
        max_its=int(args.max_its),
        ref_amp=float(args.reference_amplitude),
        ref_steps=int(args.ref_steps),
        use_old_stiffness=not bool(args.disable_old_stiffness),
        expected_steps=n_dyn,
    )

    n = min(len(res_full["sig"]), len(res_hrom["sig"]))
    rel_sig_err = np.linalg.norm(res_full["sig"][:n] - res_hrom["sig"][:n]) / (
        np.linalg.norm(res_full["sig"][:n]) + 1e-30
    )
    rel_q_err = np.nan
    q_diag = None
    ss_diag = _save_stress_strain_diagnostics(
        res_full["eps"], res_full["sig"], res_hrom["eps"], res_hrom["sig"], seg_out_dir
    )
    if res_full["q"] is not None and res_hrom["q"] is not None:
        nq = min(len(res_full["q"]), len(res_hrom["q"]))
        rel_q_err = np.linalg.norm(res_full["q"][:nq] - res_hrom["q"][:nq]) / (
            np.linalg.norm(res_full["q"][:nq]) + 1e-30
        )
        q_diag = _save_q_diagnostics(res_full["q"], res_hrom["q"], seg_out_dir)

    print("\n" + "=" * 72)
    print(f"Summary ({run_label})")
    print("=" * 72)
    print(f"FULL-MESH ECM time [s]       : {res_full['time']:.6f}")
    print(f"FULL-MESH ECM steps/iters    : {res_full['n_steps']} / {res_full['n_iters']} (avg {res_full['avg_iters']:.3f})")
    print(f"HROM-MESH ECM time [s]       : {res_hrom['time']:.6f}")
    print(f"HROM-MESH ECM steps/iters    : {res_hrom['n_steps']} / {res_hrom['n_iters']} (avg {res_hrom['avg_iters']:.3f})")
    print(f"HROM/FULL wall ratio         : {res_hrom['time'] / max(res_full['time'], 1e-30):.6f}")
    print(f"Rel. q-history error (HROM/FULL): {rel_q_err:.6e}")
    print(f"Rel. stress error (HROM/FULL): {rel_sig_err:.6e} (n={n})")
    if ss_diag is not None:
        print(
            f"Stepwise stress rel-error    : mean={ss_diag['mean_stress_rel']:.6e}, "
            f"max={ss_diag['max_stress_rel']:.6e}"
        )
        print(
            f"Stepwise strain rel-error    : mean={ss_diag['mean_strain_rel']:.6e}, "
            f"max={ss_diag['max_strain_rel']:.6e}"
        )
    if q_diag is not None:
        print(
            f"Per-step q rel-error stats   : mean={q_diag['mean_rel_q_error']:.6e}, "
            f"final={q_diag['final_rel_q_error']:.6e}, "
            f"max={q_diag['max_rel_q_error']:.6e} @ step {q_diag['max_rel_q_error_step']}"
        )
        print(f"Saved q diagnostics to       : {seg_out_dir}")
    print(f"FULL log: {res_full['log_path']}")
    print(f"HROM log: {res_hrom['log_path']}")

    return {
        "segment_index": int(seg_idx_start),
        "segment_index_end": int(seg_idx_end),
        "run_label": run_label,
        "rel_q_err": float(rel_q_err) if np.isfinite(rel_q_err) else np.nan,
        "rel_sig_err": float(rel_sig_err),
        "time_full": float(res_full["time"]),
        "time_hrom": float(res_hrom["time"]),
        "q_rel_step": None if q_diag is None else q_diag["rel_step"],
    }


def main():
    p = argparse.ArgumentParser(description="Profile HPROM-RBF full-mesh ECM vs HROM-mesh ECM.")
    p.add_argument("--segment-index", type=int, default=None, help="Single Stage-11 segment index.")
    p.add_argument("--segment-indices", type=int, nargs="+", default=None, help="Multiple segment indices in one run.")
    p.add_argument(
        "--path-seg-start",
        type=int,
        default=None,
        help="Start segment index (inclusive) for one contiguous Stage-11 trajectory window.",
    )
    p.add_argument(
        "--path-seg-end",
        type=int,
        default=None,
        help="End segment index (inclusive) for one contiguous Stage-11 trajectory window.",
    )
    p.add_argument("--max-its", type=int, default=25, help="Max Newton iterations.")
    p.add_argument("--ref-steps", type=int, default=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE)
    p.add_argument("--reference-amplitude", type=float, default=2.0)
    p.add_argument("--disable-old-stiffness", action="store_true")
    p.add_argument("--out-dir", default="profile_hprom_rbf_full_vs_hrom_mesh")
    args = p.parse_args()

    emax, rel6, domain_type = _load_stage11_domain_inputs()
    _, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    full_path = np.array(waypoints, dtype=float)
    n_total_segments = len(full_path) - 1

    run_ranges = []
    if args.path_seg_start is not None or args.path_seg_end is not None:
        s0 = 0 if args.path_seg_start is None else int(args.path_seg_start)
        s1 = (n_total_segments - 1) if args.path_seg_end is None else int(args.path_seg_end)
        if s0 > s1:
            raise ValueError(f"path-seg-start ({s0}) must be <= path-seg-end ({s1}).")
        run_ranges = [(s0, s1)]
    else:
        if args.segment_indices is not None and len(args.segment_indices) > 0:
            segment_indices = sorted(set(int(s) for s in args.segment_indices))
        elif args.segment_index is not None:
            segment_indices = [int(args.segment_index)]
        else:
            segment_indices = [0]
        run_ranges = [(s, s) for s in segment_indices]

    for s0, s1 in run_ranges:
        if s0 < 0 or s1 >= n_total_segments:
            raise ValueError(f"Requested segment window [{s0}, {s1}] out of range [0, {n_total_segments - 1}].")

    (
        phi_p,
        phi_s,
        free_dofs,
        _dir_dofs,
        eq_map_full,
        Xc,
        Yc,
        rbf_model,
        ecm_data_base,
        include_macro,
    ) = LoadHpromRbfModel(
        basis_dir="stage_2_pod_rve",
        rbf_data_dir="stage_7_rbf_data",
        hprom_rbf_dir="stage_9_hprom_rbf_data",
    )

    mesh_hrom = str(np.ravel(ecm_data_base["hrom_mesh_base"])[0]) if "hrom_mesh_base" in ecm_data_base else None
    if not mesh_hrom:
        raise RuntimeError("ECM data has no hrom_mesh_base. Build HROM mesh first.")

    os.makedirs(args.out_dir, exist_ok=True)

    ecm_full = {k: v for k, v in ecm_data_base.items() if k != "w_res_hrom"}
    ecm_hrom = dict(ecm_data_base)

    model_data = (
        phi_p,
        phi_s,
        free_dofs,
        eq_map_full,
        Xc,
        Yc,
        rbf_model,
        ecm_full,
        ecm_hrom,
        include_macro,
        mesh_hrom,
    )

    per_segment_stats = []
    for s0, s1 in run_ranges:
        per_segment_stats.append(_run_segment(args, s0, s1, full_path, model_data))

    _plot_multi_segment_q_error(args.out_dir, per_segment_stats)

    print("\n" + "=" * 72)
    print("Global Summary")
    print("=" * 72)
    print(f"Segment windows processed: {run_ranges}")
    for s in per_segment_stats:
        print(
            f"  {s['run_label']:>11s} | q_err={s['rel_q_err']:.6e} | "
            f"sig_err={s['rel_sig_err']:.6e} | "
            f"t_full={s['time_full']:.3f}s | t_hrom={s['time_hrom']:.3f}s"
        )
    print(f"Combined plot: {os.path.join(args.out_dir, 'q_rel_error_vs_step_all_segments.png')}")


if __name__ == "__main__":
    main()
