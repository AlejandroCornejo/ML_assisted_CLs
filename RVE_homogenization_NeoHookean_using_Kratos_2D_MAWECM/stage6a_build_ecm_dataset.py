#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 6a (2D-MAWECM): build classical ECM dataset from Stage-1 snapshots.

Outputs:
- Q_ecm.dat  : residual projection matrix blocks (nq * Ns_res, n_elem)
- b_full.dat : residual projection target vector (nq * Ns_res,)
- C_hom.dat  : homogenization per-element blocks (6 * Ns_hom, n_elem)
- b_hom.dat  : homogenization target vector (6 * Ns_hom,)
- meta.npz   : dimensions + metadata
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    PrecomputeElementIntegrationWeights,
    setup_kratos_parameters,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage6a: build classical ECM dataset (2D-MAWECM)")
    p.add_argument("--fom-dir", type=str, default="stage_1_training_set_fom")
    p.add_argument("--pod-dir", type=str, default="stage_2a_pod_data")
    p.add_argument("--mesh", type=str, default="rve_geometry")
    p.add_argument("--out-dir", type=str, default="stage_6a_ecm_dataset")
    p.add_argument("--snapshot-percent-res", type=float, default=5.0)
    p.add_argument("--snapshot-percent-hom", type=float, default=5.0)
    p.add_argument("--sampling-mode", type=str, default="uniform", choices=["uniform", "random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--trajectory-indices",
        type=str,
        default="",
        help="Optional comma-separated trajectory IDs (e.g. '1,2'). Empty=all.",
    )
    p.add_argument(
        "--first-n-steps",
        type=int,
        default=0,
        help="If >0, only first N snapshots per selected trajectory are considered.",
    )
    return p.parse_args()


def _discover_trajectory_dirs(root: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"FOM directory not found: {root}")
    out: List[Tuple[int, str]] = []
    pat = re.compile(r"^trajectory_(\d+)$")
    for name in os.listdir(root):
        m = pat.match(name)
        if m is None:
            continue
        idx = int(m.group(1))
        p = os.path.join(root, name)
        if os.path.isdir(p):
            out.append((idx, p))
    out.sort(key=lambda t: t[0])
    if len(out) == 0:
        raise RuntimeError(f"No trajectory_<i> folders found in {root}")
    return out


def _parse_selected_ids(text: str):
    s = str(text).strip()
    if not s:
        return None
    out = sorted(set(int(v.strip()) for v in s.split(",") if v.strip()))
    if len(out) == 0:
        return None
    return out


def _uniform_pick(n_steps: int, n_pick: int) -> np.ndarray:
    n_pick = int(max(1, min(int(n_steps), int(n_pick))))
    if n_pick == 1:
        return np.array([0], dtype=int)
    idx = np.linspace(0, n_steps - 1, n_pick)
    idx = np.round(idx).astype(int)
    idx = np.unique(np.clip(idx, 0, n_steps - 1))
    # guarantee requested cardinality when possible
    if idx.size < n_pick:
        all_idx = np.arange(n_steps, dtype=int)
        missing = [i for i in all_idx.tolist() if i not in set(idx.tolist())]
        add = np.array(missing[: n_pick - idx.size], dtype=int)
        idx = np.sort(np.concatenate([idx, add]))
    return idx


def _random_pick(n_steps: int, n_pick: int, rng: np.random.Generator) -> np.ndarray:
    n_pick = int(max(1, min(int(n_steps), int(n_pick))))
    idx = rng.choice(np.arange(n_steps, dtype=int), size=n_pick, replace=False)
    idx.sort()
    return idx


def _pick_indices(n_steps: int, pct: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    n_pick = int(np.ceil(float(pct) * float(n_steps) / 100.0))
    if str(mode).lower() == "random":
        return _random_pick(n_steps, n_pick, rng)
    return _uniform_pick(n_steps, n_pick)


def _build_free_map(n_dof: int, free_dofs: np.ndarray) -> np.ndarray:
    g2f = -np.ones(int(n_dof), dtype=int)
    for i, gdof in enumerate(np.asarray(free_dofs, dtype=np.int64).reshape(-1)):
        g2f[int(gdof)] = int(i)
    return g2f


def _to_voigt3(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    ncomp = int(arr.shape[1]) if arr.ndim >= 2 else 0
    if ncomp == 3:
        return arr
    if ncomp >= 4:
        return arr[:, [0, 1, 3]]
    out = np.zeros((arr.shape[0], 3), dtype=float)
    if ncomp > 0:
        out[:, : min(3, ncomp)] = arr[:, : min(3, ncomp)]
    return out


def _build_residual_projection_cache(elements, process_info, map_g2f: np.ndarray, phi_f: np.ndarray):
    cache = []
    active = 0
    for elem in elements:
        ids = np.asarray(elem.EquationIdVector(process_info), dtype=int)
        local_pos = np.flatnonzero(ids >= 0)
        if local_pos.size == 0:
            cache.append(None)
            continue

        local_dofs = ids[local_pos]
        rows = map_g2f[local_dofs]
        valid = rows >= 0
        if not np.any(valid):
            cache.append(None)
            continue

        rhs_pick = local_pos[valid].astype(int, copy=False)
        v_e_t = np.ascontiguousarray(phi_f[rows[valid], :].T)
        cache.append((rhs_pick, v_e_t))
        active += 1
    return cache, active


def _fill_hom_block(elements, process_info, area_e: np.ndarray, c_block: np.ndarray):
    for e, elem in enumerate(elements):
        eps = _to_voigt3(elem.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info))
        sig = _to_voigt3(elem.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info))
        a_e = float(area_e[e])
        c_block[0:3, e] = a_e * np.mean(eps, axis=0)
        c_block[3:6, e] = a_e * np.mean(sig, axis=0)


def _alloc_memmaps(out_dir: str, nq: int, n_s_res: int, n_s_hom: int, n_elem: int):
    os.makedirs(out_dir, exist_ok=True)
    q_path = os.path.join(out_dir, "Q_ecm.dat")
    b_path = os.path.join(out_dir, "b_full.dat")
    c_path = os.path.join(out_dir, "C_hom.dat")
    bh_path = os.path.join(out_dir, "b_hom.dat")

    for p in (q_path, b_path, c_path, bh_path):
        if os.path.exists(p):
            os.remove(p)

    q_ecm = np.memmap(q_path, dtype=np.float64, mode="w+", shape=(nq * n_s_res, n_elem))
    b_full = np.memmap(b_path, dtype=np.float64, mode="w+", shape=(nq * n_s_res,))
    c_hom = np.memmap(c_path, dtype=np.float64, mode="w+", shape=(6 * n_s_hom, n_elem))
    b_hom = np.memmap(bh_path, dtype=np.float64, mode="w+", shape=(6 * n_s_hom,))
    return q_ecm, b_full, c_hom, b_hom


def main():
    args = _parse_args()
    if args.snapshot_percent_res <= 0.0 or args.snapshot_percent_res > 100.0:
        raise ValueError("--snapshot-percent-res must be in (0,100].")
    if args.snapshot_percent_hom <= 0.0 or args.snapshot_percent_hom > 100.0:
        raise ValueError("--snapshot-percent-hom must be in (0,100].")
    if int(args.first_n_steps) < 0:
        raise ValueError("--first-n-steps must be >= 0.")

    print("=" * 72)
    print("Stage 6a: Classical ECM dataset build (2D-MAWECM)")
    print("=" * 72)
    print(f"fom_dir             : {args.fom_dir}")
    print(f"pod_dir             : {args.pod_dir}")
    print(f"out_dir             : {args.out_dir}")
    print(f"snapshot_percent    : res={args.snapshot_percent_res:.2f}%, hom={args.snapshot_percent_hom:.2f}%")
    print(f"sampling_mode       : {args.sampling_mode}")

    phi_f = np.asarray(np.load(os.path.join(args.pod_dir, "pod_basis_free.npy")), dtype=float)
    free_dofs = np.asarray(np.load(os.path.join(args.pod_dir, "free_dofs.npy")), dtype=np.int64)
    nq = int(phi_f.shape[1])

    traj_dirs = _discover_trajectory_dirs(args.fom_dir)
    selected_ids = _parse_selected_ids(args.trajectory_indices)
    if selected_ids is not None:
        allowed = set(selected_ids)
        traj_dirs = [(i, p) for (i, p) in traj_dirs if i in allowed]
        if len(traj_dirs) == 0:
            raise RuntimeError("No trajectory left after --trajectory-indices filter.")

    rng = np.random.default_rng(int(args.seed))
    tasks = []
    total_res = 0
    total_hom = 0
    for idx, tdir in traj_dirs:
        u_file = os.path.join(tdir, f"trajectory_{idx}_U.npy")
        e_file = os.path.join(tdir, f"trajectory_{idx}_applied_strain.npy")
        if not (os.path.exists(u_file) and os.path.exists(e_file)):
            raise FileNotFoundError(f"Missing trajectory arrays in {tdir}")
        u_all = np.load(u_file, mmap_mode="r")
        e_all = np.load(e_file, mmap_mode="r")
        n_steps = int(min(u_all.shape[0], e_all.shape[0]))
        if int(args.first_n_steps) > 0:
            n_steps = min(n_steps, int(args.first_n_steps))
        if n_steps <= 0:
            continue

        idx_res = _pick_indices(n_steps, args.snapshot_percent_res, args.sampling_mode, rng)
        idx_hom = _pick_indices(n_steps, args.snapshot_percent_hom, args.sampling_mode, rng)
        tasks.append((idx, tdir, idx_res, idx_hom))
        total_res += int(idx_res.size)
        total_hom += int(idx_hom.size)
        print(f"  trajectory_{idx}: n_steps={n_steps}, picked_res={idx_res.size}, picked_hom={idx_hom.size}")

    if len(tasks) == 0:
        raise RuntimeError("No valid trajectory tasks to process.")

    params = setup_kratos_parameters(args.mesh)
    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, params)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    pi = mp.ProcessInfo

    n_dof, eq_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    elements = list(mp.Elements)
    n_elem = len(elements)
    if n_elem <= 0:
        raise RuntimeError("Mesh has no elements.")

    _, area_e = PrecomputeElementIntegrationWeights(elements)
    a0_ref = float(np.sum(np.asarray(area_e, dtype=float)))
    map_g2f = _build_free_map(n_dof, free_dofs)

    residual_cache, n_active = _build_residual_projection_cache(elements, pi, map_g2f, phi_f)
    rhs_vectors = [KM.Vector() for _ in range(n_elem)]
    q_block = np.zeros((nq, n_elem), dtype=float)
    c_block = np.zeros((6, n_elem), dtype=float)

    print(f"[Info] n_elem={n_elem}, nq={nq}, active residual projections={n_active}/{n_elem}")
    print(f"[Info] total snapshots: Ns_res={total_res}, Ns_hom={total_hom}")

    q_ecm, b_full, c_hom, b_hom = _alloc_memmaps(args.out_dir, nq, total_res, total_hom, n_elem)

    s_res = 0
    s_hom = 0
    for idx, tdir, idx_res, idx_hom in tasks:
        idx_res = np.unique(np.asarray(idx_res, dtype=int).reshape(-1))
        idx_hom = np.unique(np.asarray(idx_hom, dtype=int).reshape(-1))
        idx_union = np.union1d(idx_res, idx_hom)
        set_res = set(int(v) for v in idx_res.tolist())
        set_hom = set(int(v) for v in idx_hom.tolist())

        u_file = os.path.join(tdir, f"trajectory_{idx}_U.npy")
        u_all = np.load(u_file, mmap_mode="r")

        print(f"  > replay trajectory_{idx}: union_steps={idx_union.size}")
        for k in idx_union:
            ks = int(k)
            u_snap = np.asarray(u_all[ks, :], dtype=float)

            SetDisplacementFromEquationVector(u_snap, eq_map, ta)
            UpdateCurrentCoordinatesFromDisplacement(mp, step=0)

            if ks in set_res:
                q_block.fill(0.0)
                for e, elem in enumerate(elements):
                    cached = residual_cache[e]
                    if cached is None:
                        continue
                    rhs_pick, v_e_t = cached
                    rhs = rhs_vectors[e]
                    elem.CalculateRightHandSide(rhs, pi)
                    rhs_arr = np.asarray(rhs, dtype=float)
                    q_block[:, e] = v_e_t @ rhs_arr[rhs_pick]

                r0, r1 = nq * s_res, nq * (s_res + 1)
                q_ecm[r0:r1, :] = q_block
                b_full[r0:r1] = np.sum(q_block, axis=1)
                s_res += 1

            if ks in set_hom:
                _fill_hom_block(elements, pi, area_e, c_block)
                h0, h1 = 6 * s_hom, 6 * (s_hom + 1)
                c_hom[h0:h1, :] = c_block
                b_hom[h0:h1] = np.sum(c_block, axis=1)
                s_hom += 1

    if s_res != total_res:
        raise RuntimeError(f"Residual sample count mismatch: expected {total_res}, got {s_res}")
    if s_hom != total_hom:
        raise RuntimeError(f"Hom sample count mismatch: expected {total_hom}, got {s_hom}")

    q_ecm.flush()
    b_full.flush()
    c_hom.flush()
    b_hom.flush()

    np.savez(
        os.path.join(args.out_dir, "meta.npz"),
        nq=np.array([nq], dtype=np.int64),
        n_elem=np.array([n_elem], dtype=np.int64),
        N_s_res=np.array([total_res], dtype=np.int64),
        N_s_hom=np.array([total_hom], dtype=np.int64),
        snapshot_percent_res=np.array([float(args.snapshot_percent_res)], dtype=float),
        snapshot_percent_hom=np.array([float(args.snapshot_percent_hom)], dtype=float),
        sampling_mode=np.array([str(args.sampling_mode)]),
        seed=np.array([int(args.seed)], dtype=np.int64),
        first_n_steps=np.array([int(args.first_n_steps)], dtype=np.int64),
        trajectory_ids=np.array([t[0] for t in tasks], dtype=np.int64),
        pod_dir=np.array([str(args.pod_dir)]),
        fom_dir=np.array([str(args.fom_dir)]),
        A_total=np.array([a0_ref], dtype=float),
        A0_ref=np.array([a0_ref], dtype=float),
        hom_reference_measure=np.array([a0_ref], dtype=float),
    )

    print("[OK] Stage6a done")
    print(f"  Q_ecm shape: ({nq * total_res}, {n_elem})")
    print(f"  C_hom shape: ({6 * total_hom}, {n_elem})")
    print(f"  A0_ref: {a0_ref:.6e}")


if __name__ == "__main__":
    main()
