#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5c (alternative): build a Gappy-POD homogenization operator
using the residual ECM sampling set (typically Z_res).

This does NOT replace the existing Stage 5 ECM weights. It produces an
alternative homogenization model that can be tested in Stage 6.
"""

import argparse
import os
import numpy as np

from homogenization_gappy import build_gappy_pod_homogenization_operator_from_chom


def _as_numpy_dict(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _project_sample_elements_to_hrom(data_out, ecm_data):
    if "hrom_element_full_indices" not in ecm_data:
        return
    full_to_local = {
        int(full_idx): i
        for i, full_idx in enumerate(np.asarray(ecm_data["hrom_element_full_indices"], dtype=np.int64))
    }
    z_full = np.asarray(data_out["hom_gappy_sample_elements"], dtype=np.int64).reshape(-1)
    z_loc = []
    missing = []
    for e in z_full:
        key = int(e)
        loc = full_to_local.get(key)
        if loc is None:
            missing.append(key)
        else:
            z_loc.append(int(loc))
    if missing:
        print(
            "[Stage5c][WARN] Could not map all sampled full-mesh elements to HROM local indices. "
            f"missing={len(missing)}"
        )
        return
    data_out["hom_gappy_sample_elements_hrom"] = np.asarray(z_loc, dtype=np.int64)
    data_out["hom_gappy_hrom_mesh_base"] = np.array(
        str(np.ravel(ecm_data["hrom_mesh_base"])[0]) if "hrom_mesh_base" in ecm_data else ""
    )
    data_out["hom_gappy_hrom_full_mesh_base"] = np.array(
        str(np.ravel(ecm_data["hrom_full_mesh_base"])[0]) if "hrom_full_mesh_base" in ecm_data else ""
    )


def main():
    p = argparse.ArgumentParser(
        description="Stage 5c: build alternative Gappy-POD homogenization operator."
    )
    p.add_argument("--data-dir", default="stage_5_ecm_dataset", help="Stage 5 dataset folder.")
    p.add_argument(
        "--ecm-file",
        default="stage_5_hprom_data/ecm_weights_all.npz",
        help="ECM weights npz containing Z_res (and optionally HROM map).",
    )
    p.add_argument(
        "--sampling-key",
        default="Z_res",
        choices=["Z_res", "Z_union", "Z_eps", "Z_sig"],
        help="Which ECM set to use as sampled elements for gappy homogenization.",
    )
    p.add_argument(
        "--out-file",
        default="stage_5_hprom_data/homogenization_gappy_from_residual_ecm.npz",
        help="Output npz containing the alternative homogenization operator.",
    )
    p.add_argument(
        "--energy-loss-tol",
        type=float,
        default=1.0e-10,
        help="POD truncation energy loss tolerance.",
    )
    p.add_argument("--max-modes", type=int, default=256, help="Maximum number of POD modes.")
    p.add_argument("--ridge", type=float, default=1.0e-12, help="Ridge regularization in gappy LS.")
    p.add_argument(
        "--no-center-data",
        action="store_true",
        help="Disable mean-centering of C_hom snapshots before POD.",
    )
    p.add_argument(
        "--update-ecm-file",
        action="store_true",
        help="Also inject operator keys directly into --ecm-file.",
    )
    args = p.parse_args()

    meta_file = os.path.join(args.data_dir, "meta.npz")
    c_file = os.path.join(args.data_dir, "C_hom.dat")
    if not os.path.isfile(meta_file):
        raise FileNotFoundError(meta_file)
    if not os.path.isfile(c_file):
        raise FileNotFoundError(c_file)
    if not os.path.isfile(args.ecm_file):
        raise FileNotFoundError(args.ecm_file)

    meta = np.load(meta_file, allow_pickle=True)
    n_elem = int(np.ravel(meta["n_elem"])[0])
    Ns = int(np.ravel(meta["N_s_hom"])[0])
    if "A_total" not in meta:
        raise KeyError(f"'A_total' not found in {meta_file}.")
    A_total = float(np.ravel(meta["A_total"])[0])
    if abs(A_total) <= 1.0e-30:
        raise RuntimeError(f"Invalid A_total={A_total} in {meta_file}.")
    C_hom = np.memmap(c_file, dtype=np.float64, mode="r", shape=(6 * Ns, n_elem))

    ecm_data = _as_numpy_dict(args.ecm_file)
    if args.sampling_key not in ecm_data:
        raise KeyError(
            f"{args.sampling_key} not found in {args.ecm_file}. "
            f"Available keys: {sorted(ecm_data.keys())}"
        )
    sampled_elements = np.asarray(ecm_data[args.sampling_key], dtype=np.int64).reshape(-1)

    print("=" * 72)
    print("Stage 5c - Alternative homogenization operator (Gappy-POD)")
    print("=" * 72)
    print(f"Dataset dir         : {args.data_dir}")
    print(f"ECM file            : {args.ecm_file}")
    print(f"Sampling key        : {args.sampling_key}")
    print(f"n_elem              : {n_elem}")
    print(f"N_s                 : {Ns}")
    print(f"|sampled elements|  : {sampled_elements.size}")
    print(f"reference measure A : {A_total:.6e}")
    print(f"energy_loss_tol     : {args.energy_loss_tol:.3e}")
    print(f"max_modes           : {int(args.max_modes)}")
    print(f"ridge               : {args.ridge:.3e}")
    print(f"center_data         : {not args.no_center_data}")

    op_data = build_gappy_pod_homogenization_operator_from_chom(
        C_hom=C_hom,
        n_snapshots=Ns,
        n_elem=n_elem,
        sample_elements=sampled_elements,
        energy_loss_tol=float(args.energy_loss_tol),
        max_modes=int(args.max_modes),
        ridge=float(args.ridge),
        center_data=not args.no_center_data,
        reference_measure=A_total,
        normalize_by_reference_measure=True,
    )
    op_data["hom_gappy_sampling_key"] = np.array([str(args.sampling_key)])
    _project_sample_elements_to_hrom(op_data, ecm_data)

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out_file, **op_data)
    print(f"\n[Stage5c] Saved alternative operator -> {args.out_file}")
    print(
        "[Stage5c] Offline train errors: "
        f"total={float(np.ravel(op_data['hom_gappy_train_rel_error_total'])[0]):.3e}, "
        f"eps={float(np.ravel(op_data['hom_gappy_train_rel_error_eps'])[0]):.3e}, "
        f"sig={float(np.ravel(op_data['hom_gappy_train_rel_error_sig'])[0]):.3e}"
    )

    if args.update_ecm_file:
        merged = dict(ecm_data)
        merged.update(op_data)
        np.savez(args.ecm_file, **merged)
        print(f"[Stage5c] Injected operator keys into ECM file -> {args.ecm_file}")


if __name__ == "__main__":
    main()
