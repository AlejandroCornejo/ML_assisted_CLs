#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 2b (2D-MAWECM): build LS master map from POD coordinates.

Input: Stage2a POD data (q_pod_train, phi_rom, applied_strain_train).
Output: Joaquín-style master parameterization and LS map.

By default, the master target is NOT strain; it is:
  mu = [Gx, Gxy]
with optional export in deformation-gradient coordinates [F11, F12].
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2b: LS master from POD (2D-MAWECM)")
    p.add_argument("--pod-dir", type=str, default="stage_2a_pod_data")
    p.add_argument("--stage0-file", type=str, default="stage_0_trajectory/stage_0_trajectories.npz")
    p.add_argument("--out-dir", type=str, default="stage_2b_ls_master")
    p.add_argument(
        "--mu-parametrization",
        type=str,
        default="gx_gxy",
        choices=["gx_gxy", "f11_f12"],
        help="Target coordinates for LS master map.",
    )
    p.add_argument(
        "--master-dim",
        type=int,
        default=2,
        help="Number of master coordinates to fit (<=2 in current 2D setup).",
    )
    p.add_argument("--save-plots", type=int, default=1, choices=[0, 1])
    return p.parse_args()


def _load_stage0_mapping(stage0_file: str) -> str:
    if not os.path.exists(stage0_file):
        raise FileNotFoundError(f"Stage0 file not found: {stage0_file}")
    data = np.load(stage0_file, allow_pickle=True)
    if "mapping" not in data:
        raise RuntimeError("Stage0 file does not contain 'mapping' key.")
    mapping = str(np.ravel(data["mapping"])[0])
    return mapping


def _recover_g_from_applied_strain(E: np.ndarray, mapping: str) -> np.ndarray:
    if E.ndim != 2 or E.shape[1] != 3:
        raise RuntimeError(f"Expected E with shape (Ns,3), got {E.shape}")

    exx = np.asarray(E[:, 0], dtype=float)
    gamma = np.asarray(E[:, 2], dtype=float)

    if mapping == "small_strain":
        gx = exx
        gxy = gamma
        return np.column_stack([gx, gxy])

    if mapping != "green_lagrange_upper":
        raise RuntimeError(f"Unsupported Stage0 mapping for inversion: {mapping}")

    disc = 1.0 + 2.0 * exx
    bad = disc <= 0.0
    if np.any(bad):
        idx = int(np.nonzero(bad)[0][0])
        raise RuntimeError(
            "Cannot recover Gx from Exx: found 1 + 2*Exx <= 0 at sample "
            f"{idx} (Exx={exx[idx]:.6e})."
        )

    # Branch around the identity: F11 = 1 + Gx > 0.
    gx = np.sqrt(disc) - 1.0
    den = 1.0 + gx

    near_zero = np.abs(den) < 1e-12
    if np.any(near_zero):
        idx = int(np.nonzero(near_zero)[0][0])
        raise RuntimeError(
            f"Cannot recover Gxy at sample {idx}: denominator (1+Gx) is near zero."
        )

    gxy = gamma / den
    return np.column_stack([gx, gxy])


def _build_mu_targets(E: np.ndarray, mapping: str, parametrization: str) -> Dict[str, np.ndarray]:
    g = _recover_g_from_applied_strain(E, mapping)
    gx = g[:, 0]
    gxy = g[:, 1]

    f11 = 1.0 + gx
    f12 = gxy

    if parametrization == "gx_gxy":
        mu = np.column_stack([gx, gxy])
    elif parametrization == "f11_f12":
        mu = np.column_stack([f11, f12])
    else:
        raise RuntimeError(f"Unsupported mu parametrization: {parametrization}")

    return {
        "mu": mu,
        "g": g,
        "f": np.column_stack([f11, f12]),
    }


def _fit_ls_map(q: np.ndarray, mu: np.ndarray, master_dim: int) -> Dict[str, np.ndarray]:
    if q.ndim != 2:
        raise RuntimeError(f"Expected q with shape (Ns,r), got {q.shape}")
    if mu.ndim != 2:
        raise RuntimeError(f"Expected mu with shape (Ns,d), got {mu.shape}")
    if q.shape[0] != mu.shape[0]:
        raise RuntimeError(f"Sample mismatch: q has {q.shape[0]}, mu has {mu.shape[0]}")

    d_mu = mu.shape[1]
    d = int(master_dim)
    if d < 1 or d > d_mu:
        raise RuntimeError(f"master-dim must be in [1,{d_mu}], got {d}")

    mu_d = np.asarray(mu[:, :d], dtype=float)

    # Solve min ||Q*T^T - Mu||_F with Q=[Ns,r], Mu=[Ns,d].
    t_t, *_ = np.linalg.lstsq(q, mu_d, rcond=None)  # [r,d]
    t = t_t.T  # [d,r]

    mu_hat = q @ t_t

    err_abs = float(np.linalg.norm(mu_hat - mu_d))
    err_rel = err_abs / max(float(np.linalg.norm(mu_d)), 1e-30)
    rmse = np.sqrt(np.mean((mu_hat - mu_d) ** 2, axis=0))

    return {
        "T_m": t,
        "T_m_t": t_t,
        "mu_target": mu_d,
        "q_m": mu_hat,
        "error_rel": float(err_rel),
        "error_abs": float(err_abs),
        "rmse": np.asarray(rmse, dtype=float),
    }


def _build_master_basis(phi_rom: np.ndarray, t_m_t: np.ndarray) -> Dict[str, np.ndarray]:
    # Phi_c maps master coordinates into free-DOF fluctuation space.
    phi_c = np.asarray(phi_rom @ t_m_t, dtype=float)  # [n_free, d]

    u, s, vt = np.linalg.svd(phi_c, full_matrices=False)
    tol = (s[0] if s.size else 1.0) * 1e-12
    rank = int(np.sum(s > tol))
    rank = max(rank, 1)

    phi_m = np.asarray(u[:, :rank], dtype=float)
    sigma_m = np.asarray(s[:rank], dtype=float)
    v_m_t = np.asarray(vt[:rank, :], dtype=float)
    # Phi_c = Phi_m * A_m, so A_m must carry the singular values (not inverse).
    a_m = np.asarray(np.diag(sigma_m) @ v_m_t, dtype=float)

    return {
        "phi_c": phi_c,
        "phi_m": phi_m,
        "sigma_m": sigma_m,
        "v_m_t": v_m_t,
        "A_m": a_m,
        "rank": int(rank),
    }


def _build_slave_manifold_data(
    phi_rom: np.ndarray,
    phi_m: np.ndarray,
    a_m: np.ndarray,
    q_m: np.ndarray,
    q_pod: np.ndarray,
) -> Dict[str, np.ndarray]:
    # Reduced-space embedding of master basis inside POD space.
    c_m = np.asarray(phi_rom.T @ phi_m, dtype=float)  # [r, d_m], already near-orthonormal.
    gram_cm = c_m.T @ c_m
    if np.linalg.norm(gram_cm - np.eye(c_m.shape[1], dtype=float)) > 1.0e-10:
        raise RuntimeError("C_m is not orthonormal enough; cannot build stable master/slave split.")

    # Build orthogonal complement while preserving C_m as-is.
    r = c_m.shape[0]
    proj_perp = np.eye(r, dtype=float) - (c_m @ c_m.T)
    evals, evecs = np.linalg.eigh(proj_perp)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    evals = evals[order]
    d_s = r - c_m.shape[1]
    c_s = np.asarray(evecs[:, :d_s], dtype=float)
    if c_s.size > 0:
        c_s = np.asarray(np.linalg.qr(c_s)[0], dtype=float)

    phi_s = np.asarray(phi_rom @ c_s, dtype=float)    # [n_free, r-d_m]
    q_s = np.asarray(q_pod @ c_s, dtype=float)        # [Ns, r-d_m]
    q_m_proj = np.asarray(q_pod @ c_m, dtype=float)   # [Ns, d_m]

    # Define master coordinates consistent with A_m:
    # q_master = A_m q_m  with q_master = q_pod * C_m.
    try:
        q_m_consistent = np.linalg.solve(a_m, q_m_proj.T).T
    except np.linalg.LinAlgError:
        q_m_consistent = np.linalg.lstsq(a_m, q_m_proj.T, rcond=None)[0].T

    # Diagnostics of decoder consistency using LS-predicted q_m (legacy diagnostic)
    # and with A_m-consistent q_m (the one used downstream).
    d_ref = (phi_rom @ q_pod.T).T
    d_rec_ls = (phi_m @ (a_m @ q_m.T)).T + (phi_s @ q_s.T).T
    d_rec_cons = (phi_m @ (a_m @ q_m_consistent.T)).T + (phi_s @ q_s.T).T
    err_rel_ls = float(np.linalg.norm(d_rec_ls - d_ref) / max(np.linalg.norm(d_ref), 1e-30))
    err_rel_cons = float(np.linalg.norm(d_rec_cons - d_ref) / max(np.linalg.norm(d_ref), 1e-30))

    # Master-block mismatch for legacy LS q_m and consistent q_m.
    a_qm_ls = np.asarray((a_m @ q_m.T).T, dtype=float)  # [Ns, d_m]
    a_qm_cons = np.asarray((a_m @ q_m_consistent.T).T, dtype=float)
    master_mismatch_ls = float(np.linalg.norm(a_qm_ls - q_m_proj) / max(np.linalg.norm(q_m_proj), 1e-30))
    master_mismatch_cons = float(np.linalg.norm(a_qm_cons - q_m_proj) / max(np.linalg.norm(q_m_proj), 1e-30))

    ortho_ms = float(np.linalg.norm(phi_m.T @ phi_s))
    ortho_s = float(np.linalg.norm(phi_s.T @ phi_s - np.eye(phi_s.shape[1], dtype=float))) if phi_s.shape[1] > 0 else 0.0

    return {
        "C_m": c_m,
        "C_s": c_s,
        "phi_s": phi_s,
        "q_s": q_s,
        "q_m_proj": q_m_proj,
        "q_m_consistent": q_m_consistent,
        "decoder_rel_error_ls": err_rel_ls,
        "decoder_rel_error_consistent": err_rel_cons,
        "master_mismatch_rel_ls": master_mismatch_ls,
        "master_mismatch_rel_consistent": master_mismatch_cons,
        "ortho_ms": ortho_ms,
        "ortho_s": ortho_s,
    }


def _fit_affine_mu_to_qm(mu: np.ndarray, q_m: np.ndarray) -> Dict[str, np.ndarray]:
    if mu.shape[0] != q_m.shape[0]:
        raise RuntimeError(f"mu and q_m sample mismatch: {mu.shape[0]} vs {q_m.shape[0]}")
    x = np.asarray(mu, dtype=float)
    y = np.asarray(q_m, dtype=float)
    x_aug = np.hstack([x, np.ones((x.shape[0], 1), dtype=float)])
    w, *_ = np.linalg.lstsq(x_aug, y, rcond=None)  # [(d+1), d_m]
    a = np.asarray(w[:-1, :].T, dtype=float)       # [d_m, d]
    b = np.asarray(w[-1, :], dtype=float)          # [d_m]
    y_hat = x @ a.T + b[None, :]
    rel = float(np.linalg.norm(y_hat - y) / max(np.linalg.norm(y), 1e-30))
    return {"A": a, "b": b, "rel_error": rel}


def _save_plots(out_dir: str, mu_target: np.ndarray, q_m: np.ndarray):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib unavailable. Skipping LS plots.")
        return

    if mu_target.shape[1] < 2 or q_m.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.scatter(mu_target[:, 0], mu_target[:, 1], s=8, alpha=0.45, label="Target master")
    ax.scatter(q_m[:, 0], q_m[:, 1], s=8, alpha=0.45, label="LS predicted")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_title("LS master: target vs predicted")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ls_master_target_vs_pred.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pod_dir = args.pod_dir
    phi_rom = np.load(os.path.join(pod_dir, "pod_basis_free.npy"))
    q_pod = np.load(os.path.join(pod_dir, "q_pod_train.npy"))
    E = np.load(os.path.join(pod_dir, "applied_strain_train.npy"))

    mapping = _load_stage0_mapping(args.stage0_file)
    mu_data = _build_mu_targets(E, mapping, args.mu_parametrization)
    mu_full = mu_data["mu"]

    ls = _fit_ls_map(q_pod, mu_full, int(args.master_dim))
    master = _build_master_basis(phi_rom, ls["T_m_t"])
    slave = _build_slave_manifold_data(
        phi_rom=phi_rom,
        phi_m=master["phi_m"],
        a_m=master["A_m"],
        q_m=ls["q_m"],
        q_pod=q_pod,
    )
    q_m_train = np.asarray(slave["q_m_consistent"], dtype=float)
    qm_init = _fit_affine_mu_to_qm(ls["mu_target"], q_m_train)

    print("=" * 72)
    print("Stage 2b: LS master from POD")
    print("=" * 72)
    print(f"pod_dir            : {args.pod_dir}")
    print(f"stage0_file        : {args.stage0_file}")
    print(f"mapping            : {mapping}")
    print(f"mu_parametrization : {args.mu_parametrization}")
    print(f"Ns, r              : {q_pod.shape[0]}, {q_pod.shape[1]}")
    print(f"master_dim         : {ls['mu_target'].shape[1]}")
    print(f"LS error rel       : {ls['error_rel']:.3e}")
    print(f"LS RMSE            : {ls['rmse']}")
    print(f"master basis rank  : {master['rank']}")
    print(f"decoder rel (LS q_m)        : {slave['decoder_rel_error_ls']:.3e}")
    print(f"decoder rel (consistent q_m): {slave['decoder_rel_error_consistent']:.3e}")
    print(f"master mismatch (LS q_m)    : {slave['master_mismatch_rel_ls']:.3e}")
    print(f"master mismatch (consistent): {slave['master_mismatch_rel_consistent']:.3e}")
    print(f"mu->q_m init map rel error  : {qm_init['rel_error']:.3e}")

    # Save dense arrays for downstream MAW stages.
    np.save(os.path.join(args.out_dir, "phi_rom.npy"), phi_rom)
    np.save(os.path.join(args.out_dir, "q_pod_train.npy"), q_pod)

    np.save(os.path.join(args.out_dir, "mu_train.npy"), ls["mu_target"])
    np.save(os.path.join(args.out_dir, "mu_train_gx_gxy.npy"), mu_data["g"])
    np.save(os.path.join(args.out_dir, "mu_train_f11_f12.npy"), mu_data["f"])
    np.save(os.path.join(args.out_dir, "q_m_train.npy"), q_m_train)
    np.save(os.path.join(args.out_dir, "q_m_ls_pred_from_qpod.npy"), ls["q_m"])

    np.save(os.path.join(args.out_dir, "T_m.npy"), ls["T_m"])
    np.save(os.path.join(args.out_dir, "phi_c.npy"), master["phi_c"])
    np.save(os.path.join(args.out_dir, "phi_m.npy"), master["phi_m"])
    np.save(os.path.join(args.out_dir, "phi_s.npy"), slave["phi_s"])
    np.save(os.path.join(args.out_dir, "sigma_m.npy"), master["sigma_m"])
    np.save(os.path.join(args.out_dir, "v_m_t.npy"), master["v_m_t"])
    np.save(os.path.join(args.out_dir, "A_m.npy"), master["A_m"])
    np.save(os.path.join(args.out_dir, "C_m.npy"), slave["C_m"])
    np.save(os.path.join(args.out_dir, "C_s.npy"), slave["C_s"])
    np.save(os.path.join(args.out_dir, "q_s_train.npy"), slave["q_s"])
    np.save(os.path.join(args.out_dir, "q_m_projected_from_qpod.npy"), slave["q_m_proj"])
    np.save(os.path.join(args.out_dir, "q_m_init_from_mu_A.npy"), qm_init["A"])
    np.save(os.path.join(args.out_dir, "q_m_init_from_mu_b.npy"), qm_init["b"])

    # Copy structured mesh entities from Stage0 (if present) so MAW/graph stages
    # can use the same structured topology directly.
    stage0 = np.load(args.stage0_file, allow_pickle=True)
    for key in (
        "grid_nodes_param",
        "grid_cells_quad",
        "grid_graph_edges",
        "grid_graph_laplacian",
        "structured_mesh_shape",
        "trajectory_param_1",
        "trajectory_param_2",
        "trajectory_labels",
    ):
        if key in stage0:
            np.save(os.path.join(args.out_dir, f"stage0_{key}.npy"), stage0[key])

    summary = {
        "pod_dir": args.pod_dir,
        "stage0_file": args.stage0_file,
        "mapping": mapping,
        "mu_parametrization": args.mu_parametrization,
        "n_samples": int(q_pod.shape[0]),
        "pod_dim": int(q_pod.shape[1]),
        "master_dim": int(ls["mu_target"].shape[1]),
        "ls_error_rel": float(ls["error_rel"]),
        "ls_error_abs": float(ls["error_abs"]),
        "ls_rmse": [float(x) for x in ls["rmse"]],
        "master_rank": int(master["rank"]),
        "slave_dim": int(slave["q_s"].shape[1]),
        "decoder_rel_error_ls_qm": float(slave["decoder_rel_error_ls"]),
        "decoder_rel_error_consistent_qm": float(slave["decoder_rel_error_consistent"]),
        "master_mismatch_rel_ls_qm": float(slave["master_mismatch_rel_ls"]),
        "master_mismatch_rel_consistent_qm": float(slave["master_mismatch_rel_consistent"]),
        "q_m_source_for_stage3": "A_m_inverse_times_projected_master",
        "mu_to_qm_init_rel_error": float(qm_init["rel_error"]),
        "ortho_phi_m_phi_s": float(slave["ortho_ms"]),
        "ortho_phi_s": float(slave["ortho_s"]),
    }
    with open(os.path.join(args.out_dir, "stage2b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if int(args.save_plots) == 1:
        _save_plots(args.out_dir, ls["mu_target"], ls["q_m"])

    print(f"[OK] Stage 2b data saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
