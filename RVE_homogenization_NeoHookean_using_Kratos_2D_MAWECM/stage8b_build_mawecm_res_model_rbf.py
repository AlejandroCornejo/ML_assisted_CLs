#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 8b (2D-MAWECM): MAW model build.

Modes:
- res_only: MAW on residual channel only.
- res_eps_sig: MAW on residual + strain + stress channels (separate MAW runs).

Homogenization source is strict/explicit (no fallback):
- full_mesh: hom channels start from full mesh weights (ones).
- fixed_ecm: hom channels start from a provided fixed ECM file.
- ecm_separate: hom channels are built in this stage with separate classic ECM
  for eps/sig (with full ECM progress logs).
Graph stage is optional and controlled explicitly by CLI flags:
- --use-global-graph-2ndstage
- --smooth-laplacian-all-iterations
"""

import argparse
import os
import sys
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from mawecm_graph_utils import (
    build_cell_graph_laplacian,
    build_knn_graph_laplacian,
    edge_jump_metrics,
    edge_list_from_laplacian,
)
from mawecm_pruning import run_mawecm_pruning
from mawecm_rbf_weights import fit_mawecm_rbf, eval_mawecm_rbf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_CLASSIC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "RVE_homogenization_NeoHookean_using_Kratos"))
if _MAIN_CLASSIC_DIR not in sys.path:
    sys.path.append(_MAIN_CLASSIC_DIR)

from empirical_cubature_method import EmpiricalCubatureMethod
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage8b MAW-ECM build. Supports residual-only, res+eps, and "
            "separate res/eps/sig MAW, with optional graph stage for residual channel."
        )
    )
    p.add_argument("--dataset-dir", type=str, default="stage_8a_mawecm_res_dataset")
    p.add_argument("--out-dir", type=str, default="stage_8b_hprom_mawecm_res_rbf")
    p.add_argument(
        "--maw-mode",
        type=str,
        default="res_only",
        choices=["res_only", "res_eps", "res_eps_sig"],
        help=(
            "MAW channel mode. "
            "'res_only' => only residual channel is pruned adaptively. "
            "'res_eps' => residual/strain are pruned, stress keeps classical ECM. "
            "'res_eps_sig' => residual/strain/stress are pruned in separate MAW runs."
        ),
    )
    p.add_argument(
        "--hom-source",
        type=str,
        default="full_mesh",
        choices=["full_mesh", "fixed_ecm", "ecm_separate"],
        help=(
            "Homogenization source in Stage8b payload. "
            "'full_mesh' => hom bootstrap weights are full-mesh ones. "
            "'fixed_ecm' => hom bootstrap weights come from --fixed-ecm-file. "
            "'ecm_separate' => run classic ECM in Stage8b for eps/sig "
            "(separate from residual MAW)."
        ),
    )
    p.add_argument(
        "--fixed-ecm-file",
        type=str,
        default="stage_6b_hprom_ecm/ecm_weights_all.npz",
        help="Used only when --hom-source fixed_ecm. Must contain Z_eps/Z_sig and w_eps_full/w_sig_full.",
    )
    p.add_argument(
        "--res-bootstrap-ecm-file",
        type=str,
        default="",
        help=(
            "Optional path to a classic ECM npz (e.g. Stage6b ecm_weights_all.npz). "
            "When provided, its Z_res/w_res_full is used as bootstrap zINI/wINI."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Global numpy seed for reproducible Stage8b builds.",
    )

    p.add_argument(
        "--res-candidate-pool",
        type=str,
        default="fixed_support",
        choices=["fixed_support"],
        help="Docs-aligned MAW start: candidate support is the bootstrap fixed ECM support zINI.",
    )
    p.add_argument(
        "--res-target-source",
        type=str,
        default="anchor",
        choices=["anchor"],
        help="Docs-aligned local RHS: b_j = A_j * wINI.",
    )
    p.add_argument(
        "--maw-state-space",
        type=str,
        default="q_m",
        choices=["q_m"],
        help="State variable used for graph/RBF in MAW-ECM. Fixed to q_m (docs-aligned).",
    )
    p.add_argument(
        "--use-global-graph-2ndstage",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "If 1, stage-2 positivity enforcement for residual MAW uses "
            "graph-coupled PruneStep/PruneSweep logic. If 0, uses local active-set only."
        ),
    )
    p.add_argument(
        "--smooth-laplacian-all-iterations",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "If 0, phase-1 NOENF runs first and stage-2 is used only when phase-1 stalls. "
            "If 1, stage-2 is forced every iteration (not allowed with graph stage)."
        ),
    )
    p.add_argument(
        "--alpha-smooth",
        type=float,
        default=1.0e4,
        help="Graph regularization weight alpha_G used in residual graph stage.",
    )
    p.add_argument(
        "--graph-mode",
        type=str,
        default="structured_grid",
        choices=["structured_grid", "knn"],
        help=(
            "Residual graph construction mode when --use-global-graph-2ndstage=1. "
            "'structured_grid' uses Stage8a grid cells; 'knn' uses q_m kNN graph."
        ),
    )
    p.add_argument("--graph-knn", type=int, default=8, help="k for kNN graph mode.")
    p.add_argument(
        "--graph-kernel",
        type=str,
        default="gaussian",
        choices=["gaussian", "binary"],
        help="Edge weighting kernel for kNN graph mode.",
    )
    p.add_argument(
        "--graph-sigma",
        type=float,
        default=-1.0,
        help="Gaussian sigma for kNN graph. <=0 means automatic median-kNN distance.",
    )
    p.add_argument(
        "--graph-cell-weight-mode",
        type=str,
        default="binary",
        choices=["binary"],
        help="Weight mode for structured-grid graph.",
    )
    p.add_argument(
        "--max-number-zeros-active-set-loop-maw-ecm",
        type=int,
        default=1,
        help="If >0, enables stage-2 explicit enforcement when NOENF_cl cannot eliminate more points.",
    )
    p.add_argument("--criterion", type=int, default=2)
    p.add_argument("--n-candidates-to-try", type=int, default=0)
    p.add_argument("--incremental-smoothing", type=int, default=1, choices=[0, 1])
    p.add_argument("--use-total-as-criterion", type=int, default=0, choices=[0, 1])
    p.add_argument("--tol-rank-rel", type=float, default=1.0e-12)
    p.add_argument("--tol-neg-factor", type=float, default=10.0)
    p.add_argument("--tol-zero", type=float, default=1.0e-12)
    p.add_argument("--max-as-iters", type=int, default=30)
    p.add_argument("--max-reduced-dim", type=int, default=2500)
    p.add_argument(
        "--maw-min-support-size",
        type=int,
        default=0,
        help=(
            "Legacy alias for residual hard lower bound |Z_res|. "
            "Use --maw-min-support-size-res for explicit per-channel control. "
            "If <=0, uses internal default n_stop=max(local constraint rows)."
        ),
    )
    p.add_argument(
        "--maw-min-support-size-res",
        type=int,
        default=-1,
        help=(
            "Residual hard lower bound |Z_res|. "
            "If <=0, auto n_stop is used. "
            "If >0, overrides --maw-min-support-size."
        ),
    )
    p.add_argument(
        "--maw-min-support-size-eps",
        type=int,
        default=0,
        help="Strain-channel hard lower bound |Z_eps|. <=0 means auto n_stop.",
    )
    p.add_argument(
        "--maw-min-support-size-sig",
        type=int,
        default=0,
        help="Stress-channel hard lower bound |Z_sig|. <=0 means auto n_stop.",
    )
    p.add_argument(
        "--enforce-sum-weights",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "If 1, enforce per-state MAW equality sum(w)=target by augmenting "
            "each local constraint block with one row of ones."
        ),
    )
    p.add_argument(
        "--sum-weights-target",
        type=float,
        default=-1.0,
        help=(
            "Target for sum(w) when --enforce-sum-weights=1. "
            "If <=0, uses sum(w_ini)."
        ),
    )
    p.add_argument("--rsvd-tol-res-bootstrap", type=float, default=1.0e-6)
    p.add_argument(
        "--res-bootstrap-rsvd-randomized",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use randomized RSVD in residual bootstrap (0 = deterministic path).",
    )
    p.add_argument("--ecm-tol-res-bootstrap", type=float, default=0.0)
    p.add_argument(
        "--res-bootstrap-constrain-sum-weights",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "Constraint used only in bootstrap fixed ECM (zINI,wINI). "
            "Default 1 enforces classical ECM sum-weight consistency in bootstrap."
        ),
    )
    p.add_argument("--ecm-max-unsuccessful-it", type=int, default=200)

    p.add_argument("--rbf-centers-res", type=int, default=0)
    p.add_argument("--rbf-poly-mode", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--rbf-lambda", type=float, default=1.0e-10)
    p.add_argument("--rbf-length-scale-factor", type=float, default=1.0)
    p.add_argument("--rbf-clip-nonnegative", type=int, default=1, choices=[0, 1])
    p.add_argument("--rbf-renorm", type=int, default=1, choices=[0, 1])
    p.add_argument(
        "--save-weight-field-plots",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, save one q_m1-q_m2-weight plot per MAW support weight.",
    )
    p.add_argument(
        "--show-weight-field-plots",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, call plt.show() for each plot (interactive sessions only).",
    )
    p.add_argument(
        "--max-weight-field-plots",
        type=int,
        default=0,
        help="Maximum number of weight-field plots to generate. 0 means all.",
    )
    p.add_argument(
        "--weight-plot-format",
        type=str,
        default="png",
        help="File extension for saved weight-field plots (e.g. png, pdf).",
    )

    p.add_argument("--strict-constraint-rel-tol", type=float, default=1.0e-8)
    p.add_argument("--strict-negative-tol", type=float, default=1.0e-12)
    p.add_argument(
        "--maw-res-enforce-nonnegativity",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, residual MAW enforces nonnegative weights. Recommended ON.",
    )
    p.add_argument(
        "--maw-hom-enforce-nonnegativity",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, eps/sig MAW enforces nonnegative weights. Default OFF (signed weights allowed).",
    )
    p.add_argument(
        "--maw-hom-conservative",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "If 1, enforce conservative local-pruning tolerances for eps/sig channels "
            "(rank/zero/neg and active-set iterations)."
        ),
    )
    p.add_argument("--maw-hom-tol-rank-rel", type=float, default=1.0e-12)
    p.add_argument("--maw-hom-tol-neg-factor", type=float, default=12.0)
    p.add_argument("--maw-hom-tol-zero", type=float, default=1.0e-12)
    p.add_argument("--maw-hom-max-as-iters", type=int, default=60)
    p.add_argument("--maw-hom-max-reduced-dim", type=int, default=2500)
    p.add_argument("--maw-hom-criterion", type=int, default=2)
    p.add_argument(
        "--maw-hom-rowspace-compress",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "If 1, each eps/sig local block is compressed to its numerical row-space "
            "before MAW pruning (rank-aware constraints)."
        ),
    )
    p.add_argument("--maw-hom-rowspace-tol-rel", type=float, default=1.0e-12)
    p.add_argument(
        "--maw-hom-cv-max-rel",
        type=float,
        default=5.0e-3,
        help=(
            "Max allowed CV relative local-constraint error for eps/sig MAW (<=0 disables CV guard). "
            "Used to prevent overly aggressive hom pruning in auto mode."
        ),
    )
    p.add_argument("--maw-hom-cv-folds", type=int, default=5)
    p.add_argument("--maw-hom-cv-seed", type=int, default=11)
    p.add_argument(
        "--maw-hom-cv-min-samples",
        type=int,
        default=60,
        help="Minimum structured samples required to activate hom CV guard.",
    )
    p.add_argument(
        "--maw-hom-cv-strict",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1 and auto mode cannot satisfy hom CV guard, Stage8b raises instead of silently accepting.",
    )
    return p.parse_args()


def _meta_int(meta, key):
    return int(np.ravel(meta[key])[0])


def _meta_float(meta, key):
    return float(np.ravel(meta[key])[0])


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))


def _run_rsvd_on_transpose(
    m_t: np.ndarray,
    tol: float,
    label: str,
    use_randomization: bool,
):
    rsvd = RandomizedSingularValueDecomposition(
        COMPUTE_U=True,
        COMPUTE_V=False,
        RELATIVE_SVD=True,
        USE_RANDOMIZATION=bool(use_randomization),
    )
    u, s, _, _ = rsvd.Calculate(np.ascontiguousarray(m_t), truncation_tolerance=float(tol))
    if u.size == 0:
        raise RuntimeError(f"[{label}] RSVD returned empty basis")
    return np.asarray(u, dtype=float), np.asarray(s, dtype=float)


def _run_ecm(
    u_basis: np.ndarray,
    n_elem: int,
    ecm_tol: float,
    max_unsuccessful_it: int,
    constrain_sum_of_weights: bool,
):
    ecm = EmpiricalCubatureMethod(
        ECM_tolerance=float(ecm_tol),
        Filter_tolerance=0.0,
        Plotting=False,
        MaximumNumberUnsuccesfulIterations=int(max_unsuccessful_it),
    )
    ecm.SetUp(
        ResidualsBasis=np.asarray(u_basis, dtype=float),
        InitialCandidatesSet=None,
        constrain_sum_of_weights=bool(constrain_sum_of_weights),
        constrain_conditions=False,
        number_of_conditions=0,
    )
    ecm.Run()
    z = np.asarray(ecm.z, dtype=np.int64).reshape(-1)
    w_sel = np.asarray(ecm.w, dtype=float).reshape(-1)
    w_full = np.zeros(int(n_elem), dtype=float)
    w_full[z] = w_sel
    return z, w_sel, w_full


def _compute_classic_res_bootstrap(
    dataset,
    rsvd_tol,
    rsvd_randomized,
    ecm_tol,
    max_unsuccessful_it,
    constrain_sum_of_weights,
):
    q_res = np.asarray(dataset["Q_res"], dtype=float)
    b_res = np.asarray(dataset["b_res"], dtype=float).reshape(-1)
    n_elem = int(dataset["n_elem"])

    print("  [Stage8b] Residual bootstrap ECM: computing fixed rule from Stage8a dataset...")
    u_res, _ = _run_rsvd_on_transpose(
        np.asarray(q_res.T, dtype=float),
        float(rsvd_tol),
        label="RES-BOOT",
        use_randomization=bool(rsvd_randomized),
    )
    z_res, w_res, w_res_full = _run_ecm(
        u_basis=u_res,
        n_elem=n_elem,
        ecm_tol=float(ecm_tol),
        max_unsuccessful_it=int(max_unsuccessful_it),
        constrain_sum_of_weights=bool(constrain_sum_of_weights),
    )
    err = _rel_error(q_res @ w_res_full, b_res)
    print(
        f"  [Stage8b] Residual bootstrap ECM: |Z_res|={z_res.size} "
        f"({100.0*z_res.size/max(n_elem,1):.1f}% of {n_elem}), rel_err={err:.3e}"
    )
    return {"Z_res": z_res, "w_res": w_res, "w_res_full": w_res_full, "rel_err": float(err)}


def _compute_classic_hom_bootstrap(
    dataset,
    rsvd_tol,
    rsvd_randomized,
    ecm_tol,
    max_unsuccessful_it,
    constrain_sum_of_weights,
):
    c_hom = dataset.get("C_hom", None)
    b_hom = dataset.get("b_hom", None)
    ns_hom = int(dataset.get("ns_hom", 0))
    n_elem = int(dataset["n_elem"])
    if c_hom is None or b_hom is None or ns_hom <= 0:
        raise RuntimeError(
            "Homogenization bootstrap ECM requires C_hom/b_hom in Stage8a dataset. "
            "Rebuild Stage8a with --include-homogenization 1."
        )

    c_eps, b_eps, c_sig, b_sig = _split_hom_blocks(c_hom, b_hom, ns_hom=ns_hom, n_elem=n_elem)

    print("  [Stage8b] Hom bootstrap ECM (eps): computing fixed rule from Stage8a hom dataset...")
    u_eps, _ = _run_rsvd_on_transpose(
        np.asarray(c_eps.T, dtype=float),
        float(rsvd_tol),
        label="HOM-EPS-BOOT",
        use_randomization=bool(rsvd_randomized),
    )
    z_eps, w_eps, w_eps_full = _run_ecm(
        u_basis=u_eps,
        n_elem=n_elem,
        ecm_tol=float(ecm_tol),
        max_unsuccessful_it=int(max_unsuccessful_it),
        constrain_sum_of_weights=bool(constrain_sum_of_weights),
    )
    err_eps = _rel_error(c_eps @ w_eps_full, b_eps)
    print(
        f"  [Stage8b] Hom bootstrap ECM (eps): |Z_eps|={z_eps.size} "
        f"({100.0*z_eps.size/max(n_elem,1):.1f}% of {n_elem}), rel_err={err_eps:.3e}"
    )

    print("  [Stage8b] Hom bootstrap ECM (sig): computing fixed rule from Stage8a hom dataset...")
    u_sig, _ = _run_rsvd_on_transpose(
        np.asarray(c_sig.T, dtype=float),
        float(rsvd_tol),
        label="HOM-SIG-BOOT",
        use_randomization=bool(rsvd_randomized),
    )
    z_sig, w_sig, w_sig_full = _run_ecm(
        u_basis=u_sig,
        n_elem=n_elem,
        ecm_tol=float(ecm_tol),
        max_unsuccessful_it=int(max_unsuccessful_it),
        constrain_sum_of_weights=bool(constrain_sum_of_weights),
    )
    err_sig = _rel_error(c_sig @ w_sig_full, b_sig)
    print(
        f"  [Stage8b] Hom bootstrap ECM (sig): |Z_sig|={z_sig.size} "
        f"({100.0*z_sig.size/max(n_elem,1):.1f}% of {n_elem}), rel_err={err_sig:.3e}"
    )

    return {
        "Z_eps": np.asarray(z_eps, dtype=np.int64),
        "Z_sig": np.asarray(z_sig, dtype=np.int64),
        "w_eps": np.asarray(w_eps, dtype=float),
        "w_sig": np.asarray(w_sig, dtype=float),
        "w_eps_full": np.asarray(w_eps_full, dtype=float),
        "w_sig_full": np.asarray(w_sig_full, dtype=float),
        "rel_err_eps": float(err_eps),
        "rel_err_sig": float(err_sig),
    }


def _load_res_bootstrap_from_ecm_file(ecm_file: str, dataset):
    if not os.path.exists(ecm_file):
        raise FileNotFoundError(ecm_file)
    data = np.load(ecm_file, allow_pickle=True)
    required = ["Z_res"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"Bootstrap ECM file missing keys: {missing}")

    z_res = np.asarray(data["Z_res"], dtype=np.int64).reshape(-1)
    n_elem = int(dataset["n_elem"])
    if np.any(z_res < 0) or np.any(z_res >= n_elem):
        raise RuntimeError(
            f"Bootstrap Z_res out of range for current mesh: "
            f"min={int(np.min(z_res)) if z_res.size else -1}, "
            f"max={int(np.max(z_res)) if z_res.size else -1}, n_elem={n_elem}."
        )

    if "w_res_full" in data.files:
        w_res_full = np.asarray(data["w_res_full"], dtype=float).reshape(-1)
        if w_res_full.size != n_elem:
            raise RuntimeError(
                f"Bootstrap w_res_full size mismatch: got {w_res_full.size}, expected {n_elem}."
            )
    elif "w_res" in data.files:
        w_res = np.asarray(data["w_res"], dtype=float).reshape(-1)
        if w_res.size != z_res.size:
            raise RuntimeError(
                f"Bootstrap w_res size mismatch: |w_res|={w_res.size}, |Z_res|={z_res.size}."
            )
        w_res_full = np.zeros(n_elem, dtype=float)
        w_res_full[z_res] = w_res
    else:
        raise RuntimeError(
            "Bootstrap ECM file must include either w_res_full or w_res."
        )

    w_res = np.asarray(w_res_full[z_res], dtype=float).reshape(-1)
    q_res = np.asarray(dataset["Q_res"], dtype=float)
    b_res = np.asarray(dataset["b_res"], dtype=float).reshape(-1)
    err = _rel_error(q_res @ w_res_full, b_res)
    print(
        "  [Stage8b] Residual bootstrap ECM: loaded from file "
        f"'{ecm_file}'. |Z_res|={z_res.size} "
        f"({100.0*z_res.size/max(n_elem,1):.1f}% of {n_elem}), rel_err={err:.3e}"
    )
    return {"Z_res": z_res, "w_res": w_res, "w_res_full": w_res_full, "rel_err": float(err)}


def _single_block_scales(q_res: np.ndarray, c_eps: np.ndarray, c_sig: np.ndarray, mode: str):
    if mode == "none":
        return 1.0, 1.0, 1.0
    if mode == "row":
        return (
            1.0 / max(np.sqrt(q_res.shape[0]), 1.0),
            1.0 / max(np.sqrt(c_eps.shape[0]), 1.0),
            1.0 / max(np.sqrt(c_sig.shape[0]), 1.0),
        )
    # fro
    return (
        1.0 / max(np.linalg.norm(q_res), 1e-30),
        1.0 / max(np.linalg.norm(c_eps), 1e-30),
        1.0 / max(np.linalg.norm(c_sig), 1e-30),
    )


def _split_hom_blocks(c_hom: np.ndarray, b_hom: np.ndarray, ns_hom: int, n_elem: int):
    c_blk = np.asarray(c_hom, dtype=float).reshape(int(ns_hom), 6, int(n_elem))
    b_blk = np.asarray(b_hom, dtype=float).reshape(int(ns_hom), 6)
    c_eps = c_blk[:, 0:3, :].reshape(3 * int(ns_hom), int(n_elem))
    c_sig = c_blk[:, 3:6, :].reshape(3 * int(ns_hom), int(n_elem))
    b_eps = b_blk[:, 0:3].reshape(3 * int(ns_hom))
    b_sig = b_blk[:, 3:6].reshape(3 * int(ns_hom))
    return c_eps, b_eps, c_sig, b_sig


def _compute_classic_single_bootstrap(
    dataset,
    rsvd_tol,
    ecm_tol,
    max_unsuccessful_it,
    block_norm_mode,
    constrain_sum_of_weights,
):
    q_res = np.asarray(dataset["Q_res"], dtype=float)
    b_res = np.asarray(dataset["b_res"], dtype=float).reshape(-1)
    c_hom = dataset.get("C_hom", None)
    b_hom = dataset.get("b_hom", None)
    ns_hom = int(dataset.get("ns_hom", 0))
    n_elem = int(dataset["n_elem"])

    if c_hom is None or b_hom is None or ns_hom <= 0:
        raise RuntimeError(
            "Stage8a dataset has no homogenization blocks (C_hom/b_hom). "
            "Rebuild Stage8a with --include-homogenization 1."
        )

    c_eps, b_eps, c_sig, b_sig = _split_hom_blocks(c_hom, b_hom, ns_hom=ns_hom, n_elem=n_elem)
    a_res, a_eps, a_sig = _single_block_scales(q_res, c_eps, c_sig, str(block_norm_mode))
    m_all_t = np.ascontiguousarray(
        np.concatenate(
            [
                float(a_res) * np.asarray(q_res.T, dtype=float),
                float(a_eps) * np.asarray(c_eps.T, dtype=float),
                float(a_sig) * np.asarray(c_sig.T, dtype=float),
            ],
            axis=1,
        )
    )

    print("  [Stage8b] Single bootstrap ECM: computing fixed rule from Stage8a residual+hom dataset...")
    u_all, _ = _run_rsvd_on_transpose(m_all_t, float(rsvd_tol), label="SINGLE-BOOT")
    z, w_sel, w_full = _run_ecm(
        u_basis=u_all,
        n_elem=n_elem,
        ecm_tol=float(ecm_tol),
        max_unsuccessful_it=int(max_unsuccessful_it),
        constrain_sum_of_weights=bool(constrain_sum_of_weights),
    )

    err_res = _rel_error(q_res @ w_full, b_res)
    err_eps = _rel_error(c_eps @ w_full, b_eps)
    err_sig = _rel_error(c_sig @ w_full, b_sig)
    print(
        "  [Stage8b] Single bootstrap ECM: "
        f"|Z|={z.size} ({100.0*z.size/max(n_elem,1):.1f}% of {n_elem}), "
        f"rel_err(res/eps/sig)=({err_res:.3e}/{err_eps:.3e}/{err_sig:.3e})"
    )
    return {
        "Z_res": z,
        "w_res": w_sel,
        "w_res_full": w_full,
        "rel_err": float(err_res),
        "rel_err_eps": float(err_eps),
        "rel_err_sig": float(err_sig),
    }


def _load_dataset(dataset_dir):
    meta_path = os.path.join(dataset_dir, "meta.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    meta = np.load(meta_path, allow_pickle=True)

    nq = _meta_int(meta, "nq")
    n_elem = _meta_int(meta, "n_elem")
    ns_res = _meta_int(meta, "N_s_res")
    ns_hom = _meta_int(meta, "N_s_hom") if "N_s_hom" in meta else 0

    q_file = os.path.join(dataset_dir, "Q_ecm.dat")
    b_file = os.path.join(dataset_dir, "b_full.dat")
    for p in (q_file, b_file):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    q_res = np.memmap(q_file, dtype=np.float64, mode="r", shape=(nq * ns_res, n_elem))
    b_res = np.memmap(b_file, dtype=np.float64, mode="r", shape=(nq * ns_res,))

    c_hom = None
    b_hom = None
    if ns_hom > 0:
        c_file = os.path.join(dataset_dir, "C_hom.dat")
        bh_file = os.path.join(dataset_dir, "b_hom.dat")
        if os.path.exists(c_file) and os.path.exists(bh_file):
            c_hom = np.memmap(c_file, dtype=np.float64, mode="r", shape=(6 * ns_hom, n_elem))
            b_hom = np.memmap(bh_file, dtype=np.float64, mode="r", shape=(6 * ns_hom,))

    q_m_file = os.path.join(dataset_dir, "q_m_res.npy")
    if not os.path.exists(q_m_file):
        raise FileNotFoundError(
            f"{q_m_file} not found. Rebuild Stage8a with a Stage3 dataset containing grid_node_q_m."
        )
    q_m_res = np.asarray(np.load(q_m_file), dtype=float)
    mu_res = np.asarray(np.load(os.path.join(dataset_dir, "mu_res.npy")), dtype=float)
    ids_res = np.asarray(np.load(os.path.join(dataset_dir, "sample_ids_res.npy")), dtype=np.int64)
    cells_file = os.path.join(dataset_dir, "structured_mesh_cells_res.npy")
    grid_file = os.path.join(dataset_dir, "structured_mesh_grid_shape_res.npy")
    cells = np.asarray(np.load(cells_file), dtype=np.int64) if os.path.exists(cells_file) else None
    grid = np.asarray(np.load(grid_file), dtype=np.int64) if os.path.exists(grid_file) else None

    if q_m_res.shape[0] != ns_res or mu_res.shape[0] != ns_res:
        raise RuntimeError(
            f"State arrays mismatch: q_m_res={q_m_res.shape}, mu_res={mu_res.shape}, expected first dim={ns_res}."
        )

    return {
        "meta": meta,
        "nq": nq,
        "n_elem": n_elem,
        "ns_res": ns_res,
        "ns_hom": ns_hom,
        "Q_res": np.asarray(q_res, dtype=float),
        "b_res": np.asarray(b_res, dtype=float),
        "C_hom": (np.asarray(c_hom, dtype=float) if c_hom is not None else None),
        "b_hom": (np.asarray(b_hom, dtype=float) if b_hom is not None else None),
        "q_m_res": q_m_res,
        "mu_res": mu_res,
        "ids_res": ids_res,
        "cells_struct_res": cells,
        "grid_struct_res": grid,
    }


def _load_fixed_ecm_file(ecm_file, n_elem: int):
    ecm_file = str(ecm_file).strip()
    if not os.path.exists(ecm_file):
        raise FileNotFoundError(ecm_file)
    ecm = np.load(ecm_file, allow_pickle=True)
    out = {k: ecm[k] for k in ecm.files}
    required = ["Z_eps", "Z_sig", "w_eps", "w_sig", "w_eps_full", "w_sig_full"]
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(f"Fixed ECM file missing homogenization keys: {missing}")
    w_eps_full = np.asarray(out["w_eps_full"], dtype=float).reshape(-1)
    w_sig_full = np.asarray(out["w_sig_full"], dtype=float).reshape(-1)
    if w_eps_full.size != int(n_elem) or w_sig_full.size != int(n_elem):
        raise RuntimeError(
            "Fixed ECM homogenization vector sizes mismatch with current mesh: "
            f"n_elem={int(n_elem)}, len(w_eps_full)={w_eps_full.size}, len(w_sig_full)={w_sig_full.size}."
        )
    return out


def _build_fullmesh_hom_weights(n_elem: int):
    z = np.arange(int(n_elem), dtype=np.int64)
    w = np.ones(int(n_elem), dtype=float)
    return {
        "Z_eps": z.copy(),
        "Z_sig": z.copy(),
        "w_eps": w.copy(),
        "w_sig": w.copy(),
        "w_eps_full": w.copy(),
        "w_sig_full": w.copy(),
    }


def _solve_nonnegative_ls(a: np.ndarray, b: np.ndarray, l2_reg: float = 1.0e-10):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if l2_reg > 0.0:
        n = int(a.shape[1])
        a_aug = np.vstack([a, np.sqrt(float(l2_reg)) * np.eye(n, dtype=float)])
        b_aug = np.concatenate([b, np.zeros(n, dtype=float)])
    else:
        a_aug = a
        b_aug = b
    # Robust solve sequence:
    # 1) trf+lsmr (fast for many cases)
    # 2) trf+exact (more robust dense backend)
    # 3) bvls (active-set, often robust on difficult bounded LS problems)
    attempts = [
        (
            "trf_lsmr",
            dict(
                method="trf",
                lsq_solver="lsmr",
                lsmr_tol="auto",
                tol=1.0e-10,
                max_iter=5000,
                verbose=0,
            ),
        ),
        (
            "trf_exact",
            dict(
                method="trf",
                lsq_solver="exact",
                tol=1.0e-10,
                max_iter=10000,
                verbose=0,
            ),
        ),
        (
            "bvls",
            dict(
                method="bvls",
                tol=1.0e-10,
                max_iter=20000,
                verbose=0,
            ),
        ),
    ]

    fail_msgs = []
    for name, kwargs in attempts:
        res = lsq_linear(
            a_aug,
            b_aug,
            bounds=(0.0, np.inf),
            **kwargs,
        )
        if bool(res.success):
            if name != "trf_lsmr":
                print(
                    f"  [Stage8b][NNLS] solved with fallback '{name}' "
                    f"(status={res.status}, nit={getattr(res, 'nit', 'na')})."
                )
            return np.asarray(res.x, dtype=float)
        fail_msgs.append(
            f"{name}: status={res.status}, nit={getattr(res, 'nit', 'na')}, msg={res.message}"
        )

    raise RuntimeError(
        "Nonnegative least-squares failed in all backends: " + " | ".join(fail_msgs)
    )


def _fit_hom_weights_on_support_nnls(dataset, z_support, n_elem: int):
    c_hom = dataset.get("C_hom", None)
    b_hom = dataset.get("b_hom", None)
    ns_hom = int(dataset.get("ns_hom", 0))
    if c_hom is None or b_hom is None or ns_hom <= 0:
        raise RuntimeError(
            "hom_source='maw_support_nnls' requires C_hom/b_hom in Stage8a dataset. "
            "Rebuild Stage8a with --include-homogenization 1."
        )

    z = np.asarray(z_support, dtype=np.int64).reshape(-1)
    if z.size == 0:
        raise RuntimeError("Cannot fit homogenization weights on empty support.")

    c_eps, b_eps, c_sig, b_sig = _split_hom_blocks(c_hom, b_hom, ns_hom=ns_hom, n_elem=n_elem)
    a_eps = np.asarray(c_eps[:, z], dtype=float)
    a_sig = np.asarray(c_sig[:, z], dtype=float)
    b_eps = np.asarray(b_eps, dtype=float).reshape(-1)
    b_sig = np.asarray(b_sig, dtype=float).reshape(-1)

    w_eps_sel = _solve_nonnegative_ls(a_eps, b_eps, l2_reg=1.0e-10)
    w_sig_sel = _solve_nonnegative_ls(a_sig, b_sig, l2_reg=1.0e-10)

    w_eps_full = np.zeros(int(n_elem), dtype=float)
    w_sig_full = np.zeros(int(n_elem), dtype=float)
    w_eps_full[z] = w_eps_sel
    w_sig_full[z] = w_sig_sel

    tol = 1.0e-14
    z_eps = z[np.abs(w_eps_sel) > tol]
    z_sig = z[np.abs(w_sig_sel) > tol]
    if z_eps.size == 0:
        z_eps = z.copy()
    if z_sig.size == 0:
        z_sig = z.copy()

    err_eps = _rel_error(c_eps @ w_eps_full, b_eps)
    err_sig = _rel_error(c_sig @ w_sig_full, b_sig)
    return {
        "Z_eps": np.asarray(z_eps, dtype=np.int64),
        "Z_sig": np.asarray(z_sig, dtype=np.int64),
        "w_eps": np.asarray(w_eps_full[np.asarray(z_eps, dtype=np.int64)], dtype=float),
        "w_sig": np.asarray(w_sig_full[np.asarray(z_sig, dtype=np.int64)], dtype=float),
        "w_eps_full": w_eps_full,
        "w_sig_full": w_sig_full,
        "rel_err_eps": float(err_eps),
        "rel_err_sig": float(err_sig),
    }


def _select_initial_support_and_weights(res_bootstrap, n_elem, pool):
    p = str(pool).strip().lower()
    if p != "fixed_support":
        raise ValueError(
            f"Unsupported candidate pool '{pool}'. "
            "For MAW_ECM_genV1 alignment, use '--res-candidate-pool fixed_support'."
        )

    z_fix = np.asarray(res_bootstrap["Z_res"], dtype=np.int64).reshape(-1)
    w_fix = np.asarray(res_bootstrap["w_res"], dtype=float).reshape(-1)
    z = z_fix
    w = w_fix

    if z.size != w.size:
        raise RuntimeError(f"Fixed ECM mismatch: |Z|={z.size}, |w|={w.size}")
    return z, w


def _select_initial_support_and_weights_hom(ecm_fixed, n_elem, target):
    t = str(target).strip().lower()
    if t not in {"eps", "sig"}:
        raise ValueError(f"Unsupported hom target '{target}'.")
    z_key = f"Z_{t}"
    w_key = f"w_{t}"
    z = np.asarray(ecm_fixed[z_key], dtype=np.int64).reshape(-1)
    w = np.asarray(ecm_fixed[w_key], dtype=float).reshape(-1)
    if z.size == 0:
        raise RuntimeError(f"Initial hom support {z_key} is empty (strict mode, no fallback).")
    if np.any(z < 0) or np.any(z >= int(n_elem)):
        raise RuntimeError(
            f"Initial hom support {z_key} out of range: "
            f"min={int(np.min(z))}, max={int(np.max(z))}, n_elem={int(n_elem)}."
        )
    if z.size != w.size:
        raise RuntimeError(f"Initial hom weights mismatch for {t}: |Z|={z.size}, |w|={w.size}")
    return z, w


def _build_blocks_res(dataset, z_ini, w_ini, rhs_mode):
    nq = int(dataset["nq"])
    ns = int(dataset["ns_res"])
    q_res = np.asarray(dataset["Q_res"], dtype=float)
    b_res = np.asarray(dataset["b_res"], dtype=float)
    mode = str(rhs_mode).strip().lower()

    a_blocks = []
    b_blocks = []
    for s in range(ns):
        r0, r1 = nq * s, nq * (s + 1)
        a_full = q_res[r0:r1, :]
        a = a_full[:, z_ini]
        if mode == "anchor":
            b = a @ w_ini
        else:
            raise ValueError(
                f"Unsupported residual rhs mode '{rhs_mode}'. "
                "For MAW_ECM_genV1 alignment, use '--res-target-source anchor'."
            )
        a_blocks.append(np.asarray(a, dtype=float))
        b_blocks.append(np.asarray(b, dtype=float))

    q_m_train = np.asarray(dataset["q_m_res"], dtype=float)
    mu_train = np.asarray(dataset["mu_res"], dtype=float)
    ids = np.asarray(dataset["ids_res"], dtype=np.int64)
    return a_blocks, b_blocks, q_m_train, mu_train, ids


def _build_blocks_hom_component(dataset, z_ini, w_ini, rhs_mode, component):
    mode = str(rhs_mode).strip().lower()
    comp = str(component).strip().lower()
    if comp not in {"eps", "sig"}:
        raise ValueError(f"Unsupported hom component '{component}'.")

    ns_hom = int(dataset.get("ns_hom", 0))
    c_hom = dataset.get("C_hom", None)
    if c_hom is None or ns_hom <= 0:
        raise RuntimeError(
            "Hom MAW requires C_hom/b_hom in Stage8a dataset. "
            "Rebuild Stage8a with --include-homogenization 1."
        )

    q_m_train = np.asarray(dataset["q_m_res"], dtype=float)
    mu_train = np.asarray(dataset["mu_res"], dtype=float)
    ids = np.asarray(dataset["ids_res"], dtype=np.int64)
    if q_m_train.shape[0] != ns_hom:
        raise RuntimeError(
            "Hom MAW requires aligned structured samples: "
            f"N_s_hom={ns_hom}, q_m_res rows={q_m_train.shape[0]}."
        )

    c_hom = np.asarray(c_hom, dtype=float)
    n_elem = int(dataset["n_elem"])
    if c_hom.shape != (6 * ns_hom, n_elem):
        raise RuntimeError(
            f"C_hom shape mismatch: got {c_hom.shape}, expected {(6 * ns_hom, n_elem)}."
        )

    if comp == "eps":
        r_sel = slice(0, 3)
    else:
        r_sel = slice(3, 6)

    a_blocks = []
    b_blocks = []
    for s in range(ns_hom):
        h0, h1 = 6 * s, 6 * (s + 1)
        a_full = c_hom[h0:h1, :][r_sel, :]
        a = a_full[:, z_ini]
        if mode == "anchor":
            b = a @ w_ini
        else:
            raise ValueError(
                f"Unsupported hom rhs mode '{rhs_mode}'. "
                "For MAW_ECM_genV1 alignment, use '--res-target-source anchor'."
            )
        a_blocks.append(np.asarray(a, dtype=float))
        b_blocks.append(np.asarray(b, dtype=float))
    return a_blocks, b_blocks, q_m_train, mu_train, ids


def _compress_constraints_rowspace(a_blocks, b_blocks, tol_rel, label):
    """Project each local block to its numerical row-space to remove dependent rows."""
    tol_rel = float(max(tol_rel, 0.0))
    a_eff = []
    b_eff = []
    ranks = []
    m_full = []
    n_rank_zero = 0
    for a, b in zip(a_blocks, b_blocks):
        a_loc = np.asarray(a, dtype=float)
        b_loc = np.asarray(b, dtype=float).reshape(-1)
        m = int(a_loc.shape[0])
        m_full.append(m)
        if m == 0:
            r = 0
            a_hat = np.zeros((0, int(a_loc.shape[1])), dtype=float)
            b_hat = np.zeros((0,), dtype=float)
        else:
            u, s, _ = np.linalg.svd(a_loc, full_matrices=False)
            smax = float(np.max(s)) if s.size else 0.0
            tol = tol_rel * max(smax, 1.0)
            r = int(np.sum(s > tol))
            if r <= 0:
                a_hat = np.zeros((0, int(a_loc.shape[1])), dtype=float)
                b_hat = np.zeros((0,), dtype=float)
            elif r >= m:
                a_hat = a_loc
                b_hat = b_loc
            else:
                u_r = u[:, :r]
                a_hat = u_r.T @ a_loc
                b_hat = u_r.T @ b_loc
        if r == 0:
            n_rank_zero += 1
        ranks.append(r)
        a_eff.append(np.asarray(a_hat, dtype=float))
        b_eff.append(np.asarray(b_hat, dtype=float))

    ranks = np.asarray(ranks, dtype=np.int64)
    m_full = np.asarray(m_full, dtype=np.int64)
    n_blocks = int(ranks.size)
    n_reduced = int(np.sum(ranks < m_full))
    info = {
        "n_blocks": n_blocks,
        "m_full_min": int(np.min(m_full)) if m_full.size else 0,
        "m_full_max": int(np.max(m_full)) if m_full.size else 0,
        "rank_min": int(np.min(ranks)) if ranks.size else 0,
        "rank_mean": float(np.mean(ranks)) if ranks.size else 0.0,
        "rank_max": int(np.max(ranks)) if ranks.size else 0,
        "n_rank_zero": int(n_rank_zero),
        "n_row_reduced": int(n_reduced),
    }
    print(
        f"  [Stage8b][{label}] rowspace compression: "
        f"rank(min/mean/max)=({info['rank_min']}/{info['rank_mean']:.2f}/{info['rank_max']}), "
        f"reduced_blocks={info['n_row_reduced']}/{info['n_blocks']}, rank0={info['n_rank_zero']}"
    )
    return a_eff, b_eff, info


def _kfold_splits(n_samples, n_folds, seed):
    n = int(n_samples)
    k = int(max(2, min(max(2, n_folds), n)))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for f in folds:
        if f.size > 0:
            out.append(np.asarray(np.sort(f), dtype=np.int64))
    return out


def _hom_rbf_constraint_cv(
    q_train,
    w_train,
    a_blocks_red,
    b_blocks_red,
    args,
    renorm_target,
):
    """K-fold CV on local-constraint residuals after RBF interpolation of MAW weights."""
    ns = int(q_train.shape[0])
    if ns <= 2:
        return {"enabled": False, "max_rel": np.nan, "mean_rel": np.nan, "p95_rel": np.nan, "n_eval": 0}
    if ns < int(args.maw_hom_cv_min_samples):
        return {"enabled": False, "max_rel": np.nan, "mean_rel": np.nan, "p95_rel": np.nan, "n_eval": 0}

    folds = _kfold_splits(ns, int(args.maw_hom_cv_folds), int(args.maw_hom_cv_seed))
    rel_all = []
    for val_idx in folds:
        mask = np.ones(ns, dtype=bool)
        mask[val_idx] = False
        tr_idx = np.flatnonzero(mask).astype(np.int64)
        if tr_idx.size < 2 or val_idx.size == 0:
            continue
        model = fit_mawecm_rbf(
            q_train=np.asarray(q_train[tr_idx, :], dtype=float),
            W_train=np.asarray(w_train[:, tr_idx], dtype=float),
            n_centers=int(args.rbf_centers_res),
            poly_mode=int(args.rbf_poly_mode),
            lambda_reg=float(args.rbf_lambda),
            length_scale_factor=float(args.rbf_length_scale_factor),
        )
        w_pred = eval_mawecm_rbf(
            q_query=np.asarray(q_train[val_idx, :], dtype=float),
            model=model,
            clip_nonnegative=bool(int(args.rbf_clip_nonnegative)),
            renorm_target=(float(renorm_target) if bool(int(args.rbf_renorm)) else None),
        )
        for iloc, j in enumerate(val_idx.tolist()):
            a = np.asarray(a_blocks_red[int(j)], dtype=float)
            b = np.asarray(b_blocks_red[int(j)], dtype=float).reshape(-1)
            r = a @ np.asarray(w_pred[:, iloc], dtype=float) - b
            den = max(float(np.linalg.norm(b)), 1.0e-30)
            rel_all.append(float(np.linalg.norm(r) / den))

    if len(rel_all) == 0:
        return {"enabled": False, "max_rel": np.nan, "mean_rel": np.nan, "p95_rel": np.nan, "n_eval": 0}
    arr = np.asarray(rel_all, dtype=float)
    return {
        "enabled": True,
        "max_rel": float(np.max(arr)),
        "mean_rel": float(np.mean(arr)),
        "p95_rel": float(np.quantile(arr, 0.95)),
        "n_eval": int(arr.size),
    }


def _run_hom_channel_maw(dataset, ecm_fixed, n_elem, args, component, n_stop_req):
    comp = str(component).strip().lower()
    z_ini, w_ini = _select_initial_support_and_weights_hom(ecm_fixed, n_elem=n_elem, target=comp)
    a_blocks, b_blocks, q_m, _, ids = _build_blocks_hom_component(
        dataset=dataset,
        z_ini=z_ini,
        w_ini=w_ini,
        rhs_mode=str(args.res_target_source),
        component=comp,
    )

    rowspace_info = {
        "n_blocks": len(a_blocks),
        "m_full_min": int(min(a.shape[0] for a in a_blocks)) if a_blocks else 0,
        "m_full_max": int(max(a.shape[0] for a in a_blocks)) if a_blocks else 0,
        "rank_min": np.nan,
        "rank_mean": np.nan,
        "rank_max": np.nan,
        "n_rank_zero": 0,
        "n_row_reduced": 0,
    }
    a_use = a_blocks
    b_use = b_blocks
    if bool(int(args.maw_hom_rowspace_compress)):
        a_use, b_use, rowspace_info = _compress_constraints_rowspace(
            a_blocks=a_blocks,
            b_blocks=b_blocks,
            tol_rel=float(args.maw_hom_rowspace_tol_rel),
            label=comp,
        )

    w_ini_nodes = np.tile(np.asarray(w_ini, dtype=float).reshape(-1, 1), (1, int(q_m.shape[0])))
    val_ini = _validate_maw_constraints(
        a_blocks=a_use,
        b_blocks=b_use,
        w_train=w_ini_nodes,
        strict_rel_tol=max(float(args.strict_constraint_rel_tol), 1.0e-14),
        strict_neg_tol=float(args.strict_negative_tol),
        enforce_nonnegativity=bool(int(args.maw_hom_enforce_nonnegativity)),
    )
    print(
        f"  [Stage8b][{comp}] init feasibility: max_rel={val_ini['max_rel']:.3e}, "
        f"min_w={val_ini['min_weight']:.3e}"
    )

    n_cand = int(z_ini.size)
    n_stop_base = int(n_stop_req) if (n_stop_req is not None and int(n_stop_req) > 0) else None
    cv_guard_active = float(args.maw_hom_cv_max_rel) > 0.0
    n_stop_candidates = [n_stop_base]
    if n_stop_base is None and cv_guard_active:
        for frac in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00):
            n_stop_candidates.append(max(3, int(np.ceil(frac * n_cand))))
    # Preserve order, remove duplicates.
    seen = set()
    n_stop_candidates_unique = []
    for v in n_stop_candidates:
        key = None if v is None else int(v)
        if key not in seen:
            seen.add(key)
            n_stop_candidates_unique.append(key)

    accepted = None
    last_trial = None
    for n_stop_try in n_stop_candidates_unique:
        prune_opts = _make_prune_opts_hom(args, state_train=q_m, n_stop=n_stop_try)
        maw_loc = run_mawecm_pruning(
            A_blocks=a_use,
            b_blocks=b_use,
            z_ini=z_ini,
            w_ini=w_ini,
            q_train=q_m,
            options=prune_opts,
        )
        i_support_local = np.asarray(maw_loc["i_support_local"], dtype=np.int64).reshape(-1)
        w_train_loc = np.asarray(maw_loc["W_support"], dtype=float)
        z_loc = np.asarray(maw_loc["Z_support"], dtype=np.int64)
        if z_loc.size == 0:
            raise RuntimeError(f"MAW returned empty {comp} support.")
        a_red = [np.asarray(a[:, i_support_local], dtype=float) for a in a_use]
        renorm_target = float(np.sum(w_ini))
        rbf_loc = fit_mawecm_rbf(
            q_train=q_m,
            W_train=w_train_loc,
            n_centers=int(args.rbf_centers_res),
            poly_mode=int(args.rbf_poly_mode),
            lambda_reg=float(args.rbf_lambda),
            length_scale_factor=float(args.rbf_length_scale_factor),
        )
        w_recon_loc = eval_mawecm_rbf(
            q_query=q_m,
            model=rbf_loc,
            clip_nonnegative=bool(int(args.rbf_clip_nonnegative)),
            renorm_target=renorm_target if bool(int(args.rbf_renorm)) else None,
        )
        rel_recon_loc = float(
            np.linalg.norm(w_recon_loc - w_train_loc) / max(np.linalg.norm(w_train_loc), 1.0e-30)
        )
        val_loc = _validate_maw_constraints(
            a_blocks=a_red,
            b_blocks=b_use,
            w_train=w_train_loc,
            strict_rel_tol=float(args.strict_constraint_rel_tol),
            strict_neg_tol=float(args.strict_negative_tol),
            enforce_nonnegativity=bool(int(args.maw_hom_enforce_nonnegativity)),
        )
        cv_loc = _hom_rbf_constraint_cv(
            q_train=q_m,
            w_train=w_train_loc,
            a_blocks_red=a_red,
            b_blocks_red=b_use,
            args=args,
            renorm_target=renorm_target,
        )
        cv_pass = (not bool(cv_loc["enabled"])) or (float(cv_loc["max_rel"]) <= float(args.maw_hom_cv_max_rel))
        print(
            f"  [Stage8b][{comp}] trial n_stop={('auto' if n_stop_try is None else n_stop_try)}: "
            f"|Z|={z_loc.size}, cv_max_rel="
            f"{(cv_loc['max_rel'] if cv_loc['enabled'] else float('nan')):.3e}, pass={cv_pass}"
        )

        trial = {
            "maw": maw_loc,
            "i_support_local": i_support_local,
            "w_train": w_train_loc,
            "z": z_loc,
            "a_blocks_red": a_red,
            "rbf": rbf_loc,
            "rel_recon": rel_recon_loc,
            "val": val_loc,
            "cv": cv_loc,
            "n_stop_used": (-1 if n_stop_try is None else int(n_stop_try)),
            "renorm_target": renorm_target,
        }
        last_trial = trial
        if cv_pass or not cv_guard_active:
            accepted = trial
            break

    if accepted is None:
        if n_stop_base is None and bool(int(args.maw_hom_cv_strict)) and cv_guard_active:
            raise RuntimeError(
                f"[Stage8b][{comp}] CV guard failed for all auto n_stop candidates "
                f"(threshold={float(args.maw_hom_cv_max_rel):.3e})."
            )
        accepted = last_trial
        print(
            f"  [Stage8b][{comp}][WARN] accepting last trial despite CV guard "
            f"(strict={int(args.maw_hom_cv_strict)})."
        )

    smooth = edge_jump_metrics(np.asarray(accepted["w_train"], dtype=float), sparse.csr_matrix((q_m.shape[0], q_m.shape[0])))
    n_plots = 0
    if bool(int(args.save_weight_field_plots)):
        n_plots = int(
            _plot_maw_weight_fields(
                state_train=q_m,
                w_train=accepted["w_train"],
                z_red=accepted["z"],
                rbf_model=accepted["rbf"],
                out_dir=str(args.out_dir),
                show_plots=bool(int(args.show_weight_field_plots)),
                max_plots=int(args.max_weight_field_plots),
                fmt=str(args.weight_plot_format),
                clip_nonnegative=bool(int(args.rbf_clip_nonnegative)),
                renorm=bool(int(args.rbf_renorm)),
                renorm_target=accepted["renorm_target"],
                channel_tag=comp,
            )
        )

    w_anchor = np.mean(accepted["w_train"], axis=1)
    w_full = np.zeros(int(n_elem), dtype=float)
    w_full[accepted["z"]] = w_anchor
    return {
        "component": comp,
        "z_ini": np.asarray(z_ini, dtype=np.int64),
        "w_ini": np.asarray(w_ini, dtype=float),
        "q_m": np.asarray(q_m, dtype=float),
        "ids": np.asarray(ids, dtype=np.int64),
        "maw": accepted["maw"],
        "z": np.asarray(accepted["z"], dtype=np.int64),
        "w_train": np.asarray(accepted["w_train"], dtype=float),
        "rbf": accepted["rbf"],
        "rel_recon": float(accepted["rel_recon"]),
        "val": dict(accepted["val"]),
        "smooth": dict(smooth),
        "cv": dict(accepted["cv"]),
        "n_stop_used": int(accepted["n_stop_used"]),
        "w_anchor": np.asarray(w_anchor, dtype=float),
        "w_full": np.asarray(w_full, dtype=float),
        "n_weight_field_plots": int(n_plots),
        "rowspace_info": dict(rowspace_info),
    }


def _build_blocks_single_res_hom(dataset, z_ini, w_ini, rhs_mode):
    nq = int(dataset["nq"])
    ns = int(dataset["ns_res"])
    ns_hom = int(dataset.get("ns_hom", 0))
    if ns_hom != ns:
        raise RuntimeError(
            "single_res_hom mode requires aligned structured samples: "
            f"N_s_res={ns}, N_s_hom={ns_hom}."
        )
    c_hom = dataset.get("C_hom", None)
    if c_hom is None:
        raise RuntimeError(
            "single_res_hom mode requires C_hom/b_hom in Stage8a dataset. "
            "Rebuild Stage8a with --include-homogenization 1."
        )

    q_res = np.asarray(dataset["Q_res"], dtype=float)
    c_hom = np.asarray(c_hom, dtype=float)
    mode = str(rhs_mode).strip().lower()

    a_blocks = []
    b_blocks = []
    for s in range(ns):
        r0, r1 = nq * s, nq * (s + 1)
        h0, h1 = 6 * s, 6 * (s + 1)
        a_res = q_res[r0:r1, :]
        a_hom = c_hom[h0:h1, :]
        a = np.vstack([a_res[:, z_ini], a_hom[:, z_ini]])
        if mode == "anchor":
            b = a @ w_ini
        else:
            raise ValueError(
                f"Unsupported residual rhs mode '{rhs_mode}'. "
                "For MAW_ECM_genV1 alignment, use '--res-target-source anchor'."
            )
        a_blocks.append(np.asarray(a, dtype=float))
        b_blocks.append(np.asarray(b, dtype=float))

    q_m_train = np.asarray(dataset["q_m_res"], dtype=float)
    mu_train = np.asarray(dataset["mu_res"], dtype=float)
    ids = np.asarray(dataset["ids_res"], dtype=np.int64)
    return a_blocks, b_blocks, q_m_train, mu_train, ids


def _augment_blocks_with_sum_constraint(a_blocks, b_blocks, target_sum):
    target = float(target_sum)
    a_aug = []
    b_aug = []
    for a, b in zip(a_blocks, b_blocks):
        a_loc = np.asarray(a, dtype=float)
        b_loc = np.asarray(b, dtype=float).reshape(-1)
        a_loc_aug = np.vstack([a_loc, np.ones((1, int(a_loc.shape[1])), dtype=float)])
        b_loc_aug = np.concatenate([b_loc, np.array([target], dtype=float)])
        a_aug.append(a_loc_aug)
        b_aug.append(b_loc_aug)
    return a_aug, b_aug


def _build_graph_for_res(dataset, state_train, args):
    q = np.asarray(state_train, dtype=float)
    n_nodes = int(q.shape[0])
    use_graph = bool(int(args.use_global_graph_2ndstage))
    if not use_graph:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=float), {
            "mode": "disabled",
            "knn": 0,
            "kernel": "none",
            "sigma": np.nan,
            "n_edges": 0,
        }

    mode = str(args.graph_mode).strip().lower()
    if mode == "structured_grid":
        cells = dataset.get("cells_struct_res", None)
        if cells is None:
            raise RuntimeError(
                "graph_mode='structured_grid' requires Stage8a structured cells "
                "(structured_mesh_cells_res.npy)."
            )
        k_graph = build_cell_graph_laplacian(
            n_nodes=n_nodes,
            cells=np.asarray(cells, dtype=np.int64),
            cell_type="quad4",
            weight_mode=str(args.graph_cell_weight_mode).strip().lower(),
        )
        sigma_out = np.nan
        knn_out = 0
        kernel_out = "binary"
    elif mode == "knn":
        knn = int(max(1, args.graph_knn))
        sigma_in = float(args.graph_sigma)
        sigma = None if sigma_in <= 0.0 else sigma_in
        k_graph = build_knn_graph_laplacian(
            q_train=q,
            knn=knn,
            metric="euclidean",
            kernel=str(args.graph_kernel).strip().lower(),
            sigma=sigma,
        )
        sigma_out = np.nan if sigma is None else float(sigma)
        knn_out = int(knn)
        kernel_out = str(args.graph_kernel).strip().lower()
    else:
        raise RuntimeError(f"Unsupported graph_mode='{mode}'.")

    k_graph = sparse.csr_matrix(k_graph, dtype=float)
    if k_graph.shape != (n_nodes, n_nodes):
        raise RuntimeError(
            f"Residual graph shape mismatch: got {k_graph.shape}, expected {(n_nodes, n_nodes)}."
        )
    i_edge, _ = edge_list_from_laplacian(k_graph)
    return k_graph, {
        "mode": mode,
        "knn": int(knn_out),
        "kernel": str(kernel_out),
        "sigma": float(sigma_out) if np.isfinite(sigma_out) else np.nan,
        "n_edges": int(i_edge.size),
    }


def _validate_maw_constraints(
    a_blocks,
    b_blocks,
    w_train,
    strict_rel_tol,
    strict_neg_tol,
    enforce_nonnegativity=True,
):
    ns = len(a_blocks)
    if int(w_train.shape[1]) != ns:
        raise RuntimeError(f"W_train columns ({w_train.shape[1]}) != number of blocks ({ns}).")

    rel_err = np.zeros(ns, dtype=float)
    abs_err = np.zeros(ns, dtype=float)
    for j, (a, b) in enumerate(zip(a_blocks, b_blocks)):
        r = np.asarray(a, dtype=float) @ np.asarray(w_train[:, j], dtype=float) - np.asarray(b, dtype=float)
        abs_e = float(np.linalg.norm(r))
        den = max(float(np.linalg.norm(b)), 1.0e-30)
        rel_e = float(abs_e / den)
        abs_err[j] = abs_e
        rel_err[j] = rel_e

    min_w = float(np.min(np.asarray(w_train, dtype=float)))
    max_rel = float(np.max(rel_err)) if rel_err.size else 0.0
    mean_rel = float(np.mean(rel_err)) if rel_err.size else 0.0
    max_abs = float(np.max(abs_err)) if abs_err.size else 0.0
    if max_rel > float(strict_rel_tol):
        raise RuntimeError(
            f"MAW local-constraint validation failed: max rel error {max_rel:.3e} > {strict_rel_tol:.3e}."
        )
    if bool(enforce_nonnegativity) and min_w < -float(strict_neg_tol):
        raise RuntimeError(
            f"MAW non-negativity validation failed: min weight {min_w:.3e} < -{strict_neg_tol:.3e}."
        )
    return {
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "max_abs": max_abs,
        "min_weight": min_w,
    }


def _plot_maw_weight_fields(
    state_train,
    w_train,
    z_red,
    rbf_model,
    out_dir,
    show_plots,
    max_plots,
    fmt,
    clip_nonnegative,
    renorm,
    renorm_target,
    channel_tag="res",
):
    q = np.asarray(state_train, dtype=float)
    w = np.asarray(w_train, dtype=float)
    z = np.asarray(z_red, dtype=np.int64).reshape(-1)
    if q.ndim != 2 or q.shape[1] < 2:
        print("  [MAW-plots][WARN] state_train is not 2D (q_m1,q_m2). Skipping weight-field plots.")
        return 0
    if w.ndim != 2 or w.shape[1] != q.shape[0]:
        print(
            "  [MAW-plots][WARN] W_train shape mismatch with q_m nodes. "
            f"W_train={w.shape}, q_train={q.shape}. Skipping weight-field plots."
        )
        return 0
    if w.shape[0] != z.size:
        print(
            "  [MAW-plots][WARN] W_train rows mismatch with Z_support. "
            f"W_train rows={w.shape[0]}, |Z_support|={z.size}. Skipping weight-field plots."
        )
        return 0

    tag = str(channel_tag).strip().lower()
    save_dir = os.path.join(out_dir, f"maw_weight_fields_{tag}")
    os.makedirs(save_dir, exist_ok=True)

    # Avoid matplotlib warning in restricted environments.
    mpl_cfg = os.path.join(save_dir, ".mplconfig")
    os.makedirs(mpl_cfg, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Keep plot style compact and readable, independent from external rc files.
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titlesize"] = 11
    mpl.rcParams["axes.labelsize"] = 10

    q1 = q[:, 0]
    q2 = q[:, 1]

    # Dense rectangular evaluation grid for true RBF surfaces.
    n_q1 = 140
    n_q2 = 90
    pad_q1 = 0.02 * max(float(np.ptp(q1)), 1.0e-12)
    pad_q2 = 0.02 * max(float(np.ptp(q2)), 1.0e-12)
    q1_vec = np.linspace(float(np.min(q1) - pad_q1), float(np.max(q1) + pad_q1), n_q1)
    q2_vec = np.linspace(float(np.min(q2) - pad_q2), float(np.max(q2) + pad_q2), n_q2)
    q1_grid, q2_grid = np.meshgrid(q1_vec, q2_vec, indexing="xy")
    q_query = np.column_stack([q1_grid.reshape(-1), q2_grid.reshape(-1)])

    renorm_target_eval = float(renorm_target) if bool(renorm) else None
    w_grid = eval_mawecm_rbf(
        q_query=q_query,
        model=rbf_model,
        clip_nonnegative=bool(clip_nonnegative),
        renorm_target=renorm_target_eval,
    )
    if w_grid.shape[0] != w.shape[0]:
        print(
            "  [MAW-plots][WARN] RBF output rows mismatch with support size. "
            f"w_grid={w_grid.shape}, |Z_support|={w.shape[0]}. Skipping weight-field plots."
        )
        return 0

    n_all = int(w.shape[0])
    n_do = n_all if int(max_plots) <= 0 else min(n_all, int(max_plots))
    saved = 0
    n_flat_fields = 0
    rel_flat_tol = 1.0e-6

    for k in range(n_do):
        elem_id = int(z[k])
        wk_grid = np.asarray(w_grid[k, :], dtype=float).reshape(n_q2, n_q1)
        wk_mean = float(np.mean(wk_grid))
        wk_range = float(np.max(wk_grid) - np.min(wk_grid))
        rel_range = wk_range / max(abs(wk_mean), 1.0e-30)
        is_flat = bool(rel_range <= rel_flat_tol)
        if is_flat:
            # Avoid visually misleading contour bands caused by tiny floating-point oscillations.
            wk_grid = np.full_like(wk_grid, wk_mean)
            n_flat_fields += 1

        fig = plt.figure(figsize=(10.8, 4.8), constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        c = ax1.contourf(q1_grid, q2_grid, wk_grid, levels=28, cmap="viridis")
        cbar = fig.colorbar(c, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("weight", fontsize=10)
        cbar.formatter = mticker.ScalarFormatter(useMathText=False)
        cbar.formatter.set_scientific(False)
        cbar.formatter.set_useOffset(False)
        cbar.update_ticks()

        ax1.set_xlabel("q_m1", fontsize=10)
        ax1.set_ylabel("q_m2", fontsize=10)
        title_tag = " [flat]" if is_flat else ""
        ax1.set_title(
            f"RBF weight field ({tag}) - element {elem_id}{title_tag}",
            fontsize=11,
            pad=8.0,
        )
        # Optional guide: training-node locations.
        ax1.scatter(q1, q2, s=6, c="k", alpha=0.15, linewidths=0.0)

        ax2.plot_surface(
            q1_grid,
            q2_grid,
            wk_grid,
            cmap="viridis",
            linewidth=0.0,
            antialiased=True,
            rcount=min(n_q2, 90),
            ccount=min(n_q1, 140),
        )
        ax2.set_xlabel("q_m1", labelpad=4.0, fontsize=10)
        ax2.set_ylabel("q_m2", labelpad=4.0, fontsize=10)
        ax2.set_zlabel("weight", labelpad=4.0, fontsize=10)
        ax2.set_title(f"RBF surface w(q_m1, q_m2)  rel-range={rel_range:.2e}", fontsize=11, pad=8.0)
        ax2.view_init(elev=28.0, azim=-58.0)

        out_file = os.path.join(save_dir, f"weight_field_elem_{elem_id:05d}.{str(fmt).strip('.')}")
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        saved += 1
        # Interactive visualization is intentionally disabled to keep Stage8b
        # non-blocking and batch-friendly even if CLI passes --show-weight-field-plots 1.
        plt.close(fig)

    print(
        f"  [MAW-plots] saved {saved} weight-field plots in: {save_dir}"
        f" (flat fields={n_flat_fields}, tol={rel_flat_tol:.1e})"
    )
    return saved


def _resolve_n_stop_res(args):
    if int(args.maw_min_support_size_res) > 0:
        return int(args.maw_min_support_size_res)
    if int(args.maw_min_support_size) > 0:
        return int(args.maw_min_support_size)
    return None


def _make_prune_opts_base(
    args,
    state_train,
    n_stop,
    enforce_nonnegativity,
    *,
    k_graph=None,
    use_global_graph_2ndstage=False,
    smooth_laplacian_all_iterations=False,
    alpha_smooth=0.0,
):
    if k_graph is None:
        k_graph = sparse.csr_matrix((int(state_train.shape[0]), int(state_train.shape[0])), dtype=float)
    else:
        k_graph = sparse.csr_matrix(k_graph, dtype=float)
    return {
        "K_graph": k_graph,
        "alpha_smooth": float(alpha_smooth),
        "use_global_graph_2ndstage": bool(use_global_graph_2ndstage),
        "smooth_laplacian_all_iterations": bool(smooth_laplacian_all_iterations),
        "max_number_zeros_active_set_loop": int(args.max_number_zeros_active_set_loop_maw_ecm),
        "criterion": int(args.criterion),
        "number_of_candidates_to_try": int(args.n_candidates_to_try) if int(args.n_candidates_to_try) > 0 else None,
        "incremental_smoothing": bool(int(args.incremental_smoothing)),
        "use_total_as_criterion": bool(int(args.use_total_as_criterion)),
        "tol_rank_rel": float(args.tol_rank_rel),
        "tol_neg_factor": float(args.tol_neg_factor),
        "tol_zero": float(args.tol_zero),
        "max_active_set_iters": int(args.max_as_iters),
        "max_reduced_dim": int(args.max_reduced_dim),
        "n_stop": (int(n_stop) if (n_stop is not None and int(n_stop) > 0) else None),
        "verbose": True,
        "enforce_nonnegativity": bool(enforce_nonnegativity),
    }


def _make_prune_opts_hom(args, state_train, n_stop):
    out = _make_prune_opts_base(
        args,
        state_train=state_train,
        n_stop=n_stop,
        enforce_nonnegativity=bool(int(args.maw_hom_enforce_nonnegativity)),
        k_graph=sparse.csr_matrix((int(state_train.shape[0]), int(state_train.shape[0])), dtype=float),
        use_global_graph_2ndstage=False,
        smooth_laplacian_all_iterations=False,
        alpha_smooth=0.0,
    )
    if bool(int(args.maw_hom_conservative)):
        out["criterion"] = int(args.maw_hom_criterion)
        out["tol_rank_rel"] = float(args.maw_hom_tol_rank_rel)
        out["tol_neg_factor"] = float(args.maw_hom_tol_neg_factor)
        out["tol_zero"] = float(args.maw_hom_tol_zero)
        out["max_active_set_iters"] = int(args.maw_hom_max_as_iters)
        out["max_reduced_dim"] = int(args.maw_hom_max_reduced_dim)
    return out


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(int(args.seed))

    dataset = _load_dataset(args.dataset_dir)
    n_elem = int(dataset["n_elem"])
    # Homogenization source is explicit (full_mesh or fixed_ecm), no fallback.
    maw_mode = str(args.maw_mode).strip().lower()
    if maw_mode not in {"res_only", "res_eps", "res_eps_sig"}:
        raise RuntimeError(f"Unsupported maw_mode='{maw_mode}'.")
    hom_source = str(args.hom_source).strip().lower()
    if hom_source not in {"full_mesh", "fixed_ecm", "ecm_separate"}:
        raise RuntimeError(f"Unsupported hom_source='{hom_source}'.")
    boot_rsvd_tol = float(args.rsvd_tol_res_bootstrap)
    boot_ecm_tol = float(args.ecm_tol_res_bootstrap)
    bootstrap_ecm_file = str(args.res_bootstrap_ecm_file).strip()
    if bootstrap_ecm_file:
        res_bootstrap = _load_res_bootstrap_from_ecm_file(
            ecm_file=bootstrap_ecm_file,
            dataset=dataset,
        )
        bootstrap_source_txt = f"file:{bootstrap_ecm_file}"
        bootstrap_source_tag = "external_ecm_file"
    else:
        res_bootstrap = _compute_classic_res_bootstrap(
            dataset,
            rsvd_tol=boot_rsvd_tol,
            rsvd_randomized=bool(int(args.res_bootstrap_rsvd_randomized)),
            ecm_tol=boot_ecm_tol,
            max_unsuccessful_it=int(args.ecm_max_unsuccessful_it),
            constrain_sum_of_weights=bool(int(args.res_bootstrap_constrain_sum_weights)),
        )
        bootstrap_source_txt = "stage8a_dataset_classic_ecm"
        bootstrap_source_tag = "stage8a_dataset_classic_ecm"

    hom_bootstrap_source_tag = hom_source
    hom_bootstrap_source_txt = hom_source
    if hom_source == "fixed_ecm":
        ecm_fixed = _load_fixed_ecm_file(args.fixed_ecm_file, n_elem=n_elem)
        hom_bootstrap_source_txt = f"fixed_ecm file='{args.fixed_ecm_file}'"
        print(f"  [Stage8b] Homogenization source: {hom_bootstrap_source_txt} (strict, explicit).")
    elif hom_source == "ecm_separate":
        ecm_fixed = _compute_classic_hom_bootstrap(
            dataset,
            rsvd_tol=boot_rsvd_tol,
            rsvd_randomized=bool(int(args.res_bootstrap_rsvd_randomized)),
            ecm_tol=boot_ecm_tol,
            max_unsuccessful_it=int(args.ecm_max_unsuccessful_it),
            constrain_sum_of_weights=bool(int(args.res_bootstrap_constrain_sum_weights)),
        )
        hom_bootstrap_source_tag = "stage8a_hom_classic_ecm"
        hom_bootstrap_source_txt = "stage8a_hom_classic_ecm"
        print(
            "  [Stage8b] Homogenization source: ecm_separate "
            "(classic ECM for eps/sig in this Stage8b, strict/explicit)."
        )
    else:
        if maw_mode in {"res_eps", "res_eps_sig"}:
            # For hom MAW channels, first build classic ECM rules on eps/sig.
            ecm_fixed = _compute_classic_hom_bootstrap(
                dataset,
                rsvd_tol=boot_rsvd_tol,
                rsvd_randomized=bool(int(args.res_bootstrap_rsvd_randomized)),
                ecm_tol=boot_ecm_tol,
                max_unsuccessful_it=int(args.ecm_max_unsuccessful_it),
                constrain_sum_of_weights=bool(int(args.res_bootstrap_constrain_sum_weights)),
            )
            hom_bootstrap_source_tag = "stage8a_hom_classic_ecm"
            hom_bootstrap_source_txt = "stage8a_hom_classic_ecm"
            if maw_mode == "res_eps":
                print(
                    "  [Stage8b] Homogenization source: full_mesh + classic ECM bootstrap "
                    "for eps/sig, with MAW only on eps (sig fixed ECM)."
                )
            else:
                print(
                    "  [Stage8b] Homogenization source: full_mesh + classic ECM bootstrap "
                    "for eps/sig (strict, explicit)."
                )
        else:
            ecm_fixed = _build_fullmesh_hom_weights(n_elem=n_elem)
            hom_bootstrap_source_txt = "full_mesh"
            print("  [Stage8b] Homogenization source: full_mesh (strict, explicit).")

    print("=" * 78)
    print("Stage 8b: MAW-ECM model build")
    print("=" * 78)
    print(f"dataset_dir : {args.dataset_dir}")
    if hom_source == "fixed_ecm":
        print(f"fixed_ecm   : {args.fixed_ecm_file}")
    else:
        print("fixed_ecm   : <unused>")
    print(f"hom_source  : {hom_source}")
    print(f"out_dir     : {args.out_dir}")
    print(f"n_elem      : {dataset['n_elem']}")
    print(f"nq          : {dataset['nq']}")
    print(f"N_s_res/hom : {dataset['ns_res']} / {dataset.get('ns_hom', 0)}")
    print(f"seed        : {int(args.seed)}")
    if bool(int(args.enforce_sum_weights)):
        if float(args.sum_weights_target) > 0.0:
            print(f"sum(w) cons.: on (target={float(args.sum_weights_target):.6e})")
        else:
            print("sum(w) cons.: on (target=auto=sum(w_ini))")
    else:
        print("sum(w) cons.: off")
    print(f"maw mode    : {maw_mode}")
    print(f"maw space   : {args.maw_state_space}")
    n_stop_res_req = _resolve_n_stop_res(args)
    if n_stop_res_req is not None and int(n_stop_res_req) > 0:
        print(f"maw min |Z_res| : {int(n_stop_res_req)} (hard stop)")
    else:
        print("maw min |Z_res| : auto (n_stop from local constraints)")
    if maw_mode in {"res_eps", "res_eps_sig"}:
        if int(args.maw_min_support_size_eps) > 0:
            print(f"maw min |Z_eps| : {int(args.maw_min_support_size_eps)} (hard stop)")
        else:
            print("maw min |Z_eps| : auto (n_stop from local constraints)")
    if maw_mode == "res_eps":
        print("maw min |Z_sig| : <disabled, classical ECM fixed>")
    if maw_mode == "res_eps_sig":
        if int(args.maw_min_support_size_sig) > 0:
            print(f"maw min |Z_sig| : {int(args.maw_min_support_size_sig)} (hard stop)")
        else:
            print("maw min |Z_sig| : auto (n_stop from local constraints)")
    res_use_graph = bool(int(args.use_global_graph_2ndstage))
    res_smooth_all = bool(int(args.smooth_laplacian_all_iterations))
    if res_use_graph and res_smooth_all:
        raise RuntimeError(
            "Invalid schedule: graph stage enabled with smooth-all-iters=1. "
            "Use --smooth-laplacian-all-iterations 0 so phase-1 (NOENF) runs first."
        )
    if res_use_graph:
        print("stage2 mode : graph_active_set (PruneStep/PruneSweep-like)")
    else:
        print("stage2 mode : local_active_set (Loop_MAWecmNOENFe-like)")
    print(f"smooth all iters : {int(res_smooth_all)}")
    if res_use_graph:
        print(
            "graph config: "
            f"mode={str(args.graph_mode).strip().lower()}, "
            f"alpha={float(args.alpha_smooth):.3e}, "
            f"knn={int(args.graph_knn)}, kernel={str(args.graph_kernel).strip().lower()}, "
            f"sigma={'auto' if float(args.graph_sigma) <= 0.0 else f'{float(args.graph_sigma):.3e}'}"
        )
    print(
        "nonnegativity: "
        f"res={'on' if bool(int(args.maw_res_enforce_nonnegativity)) else 'off'}, "
        f"hom={'on' if bool(int(args.maw_hom_enforce_nonnegativity)) else 'off'}"
    )
    print(f"bootstrap source: {bootstrap_source_txt}")
    print(
        "bootstrap sum(w): "
        f"{'on' if bool(int(args.res_bootstrap_constrain_sum_weights)) else 'off'}"
    )
    print(
        "bootstrap RSVD randomized: "
        f"{'on' if bool(int(args.res_bootstrap_rsvd_randomized)) else 'off'}"
    )
    print(
        f"res bootstrap: rsvd_tol={boot_rsvd_tol:.1e}, "
        f"ecm_tol={boot_ecm_tol:.1e}"
    )

    z_ini, w_ini = _select_initial_support_and_weights(
        res_bootstrap,
        n_elem=n_elem,
        pool=str(args.res_candidate_pool),
    )

    a_blocks, b_blocks, q_m_train, mu_train, ids = _build_blocks_res(
        dataset,
        z_ini,
        w_ini,
        rhs_mode=str(args.res_target_source),
    )

    enforce_sum_weights = bool(int(args.enforce_sum_weights))
    sum_target = float(args.sum_weights_target)
    if enforce_sum_weights:
        if sum_target <= 0.0:
            sum_target = float(np.sum(w_ini))
        a_blocks, b_blocks = _augment_blocks_with_sum_constraint(
            a_blocks=a_blocks,
            b_blocks=b_blocks,
            target_sum=sum_target,
        )
        print(f"  [Stage8b] sum(w) target resolved to {sum_target:.6e}.")

    state_mode = "q_m"
    state_train = q_m_train

    k_graph_res, graph_info_res = _build_graph_for_res(
        dataset=dataset,
        state_train=state_train,
        args=args,
    )

    prune_opts = _make_prune_opts_base(
        args,
        state_train=state_train,
        n_stop=n_stop_res_req,
        enforce_nonnegativity=bool(int(args.maw_res_enforce_nonnegativity)),
        k_graph=k_graph_res,
        use_global_graph_2ndstage=res_use_graph,
        smooth_laplacian_all_iterations=res_smooth_all,
        alpha_smooth=(float(args.alpha_smooth) if res_use_graph else 0.0),
    )
    k_graph = sparse.csr_matrix(prune_opts["K_graph"])
    if res_use_graph:
        print(f"  [Stage8b] residual graph edges: {int(graph_info_res['n_edges'])}")

    # Docs-aligned feasibility check at initialization:
    # wADAPT(:,j) = wINI must satisfy A_j w_j = b_j for all manifold samples j.
    w_ini_nodes = np.tile(np.asarray(w_ini, dtype=float).reshape(-1, 1), (1, int(dataset["ns_res"])))
    val_ini = _validate_maw_constraints(
        a_blocks=a_blocks,
        b_blocks=b_blocks,
        w_train=w_ini_nodes,
        strict_rel_tol=max(float(args.strict_constraint_rel_tol), 1.0e-14),
        strict_neg_tol=float(args.strict_negative_tol),
        enforce_nonnegativity=bool(int(args.maw_res_enforce_nonnegativity)),
    )
    print(
        f"  [Stage8b] init feasibility: max_rel={val_ini['max_rel']:.3e}, "
        f"min_w={val_ini['min_weight']:.3e}"
    )

    maw = run_mawecm_pruning(
        A_blocks=a_blocks,
        b_blocks=b_blocks,
        z_ini=z_ini,
        w_ini=w_ini,
        q_train=state_train,
        options=prune_opts,
    )

    w_full_nodes = np.asarray(maw["W_full"], dtype=float)
    i_support_local = np.asarray(maw["i_support_local"], dtype=np.int64).reshape(-1)
    w_train = np.asarray(maw["W_support"], dtype=float)
    z_red = np.asarray(maw["Z_support"], dtype=np.int64)
    if z_red.size == 0:
        raise RuntimeError("MAW returned empty residual support.")
    if w_full_nodes.shape[0] != int(z_ini.size):
        raise RuntimeError(
            f"MAW W_full row mismatch: got {w_full_nodes.shape[0]}, expected {int(z_ini.size)}."
        )
    if w_train.shape[0] != int(i_support_local.size):
        raise RuntimeError(
            f"MAW W_support row mismatch: got {w_train.shape[0]}, expected {int(i_support_local.size)}."
        )

    a_blocks_red = [np.asarray(a[:, i_support_local], dtype=float) for a in a_blocks]

    renorm_target = float(np.sum(w_ini))
    rbf = fit_mawecm_rbf(
        q_train=state_train,
        W_train=w_train,
        n_centers=int(args.rbf_centers_res),
        poly_mode=int(args.rbf_poly_mode),
        lambda_reg=float(args.rbf_lambda),
        length_scale_factor=float(args.rbf_length_scale_factor),
    )

    w_recon = eval_mawecm_rbf(
        q_query=state_train,
        model=rbf,
        clip_nonnegative=bool(int(args.rbf_clip_nonnegative)),
        renorm_target=renorm_target if bool(int(args.rbf_renorm)) else None,
    )
    rel_recon = float(np.linalg.norm(w_recon - w_train) / max(np.linalg.norm(w_train), 1.0e-30))
    val = _validate_maw_constraints(
        a_blocks=a_blocks_red,
        b_blocks=b_blocks,
        w_train=w_train,
        strict_rel_tol=float(args.strict_constraint_rel_tol),
        strict_neg_tol=float(args.strict_negative_tol),
        enforce_nonnegativity=bool(int(args.maw_res_enforce_nonnegativity)),
    )
    sum_w_nodes = np.sum(np.asarray(w_train, dtype=float), axis=0)
    sum_w_min = float(np.min(sum_w_nodes)) if sum_w_nodes.size else np.nan
    sum_w_max = float(np.max(sum_w_nodes)) if sum_w_nodes.size else np.nan
    sum_w_mean = float(np.mean(sum_w_nodes)) if sum_w_nodes.size else np.nan
    smooth = edge_jump_metrics(np.asarray(w_train, dtype=float), k_graph)
    n_weight_field_plots = 0
    if bool(int(args.save_weight_field_plots)):
        n_weight_field_plots = int(
            _plot_maw_weight_fields(
                state_train=state_train,
                w_train=w_train,
                z_red=z_red,
                rbf_model=rbf,
                out_dir=str(args.out_dir),
                show_plots=bool(int(args.show_weight_field_plots)),
                max_plots=int(args.max_weight_field_plots),
                fmt=str(args.weight_plot_format),
                clip_nonnegative=bool(int(args.rbf_clip_nonnegative)),
                renorm=bool(int(args.rbf_renorm)),
                renorm_target=renorm_target,
                channel_tag="res",
            )
        )

    # Compatibility vectors for existing consumers:
    # residual anchor is mean of adaptive fields.
    w_res_anchor = np.mean(w_train, axis=1)
    w_res_full_anchor = np.zeros(int(n_elem), dtype=float)
    w_res_full_anchor[z_red] = w_res_anchor

    hom_fit_eps = np.nan
    hom_fit_sig = np.nan
    n_weight_field_plots_eps = 0
    n_weight_field_plots_sig = 0
    maw_eps = None
    maw_sig = None
    rel_recon_eps = np.nan
    rel_recon_sig = np.nan
    val_eps = {"max_rel": np.nan, "mean_rel": np.nan, "max_abs": np.nan, "min_weight": np.nan}
    val_sig = {"max_rel": np.nan, "mean_rel": np.nan, "max_abs": np.nan, "min_weight": np.nan}
    smooth_eps = {"S": np.nan, "R95": np.nan, "Rmax": np.nan, "n_edges": 0}
    smooth_sig = {"S": np.nan, "R95": np.nan, "Rmax": np.nan, "n_edges": 0}

    if maw_mode in {"res_eps", "res_eps_sig"}:
        if maw_mode == "res_eps":
            print("  [Stage8b] Hom MAW mode: eps pruning only (sig stays classical ECM).")
        else:
            print("  [Stage8b] Hom MAW mode: separate eps/sig pruning (first-phase local, no graph).")

        # --- eps channel ---
        n_stop_eps_req = int(args.maw_min_support_size_eps) if int(args.maw_min_support_size_eps) > 0 else None
        eps_out = _run_hom_channel_maw(
            dataset=dataset,
            ecm_fixed=ecm_fixed,
            n_elem=n_elem,
            args=args,
            component="eps",
            n_stop_req=n_stop_eps_req,
        )
        maw_eps = eps_out["maw"]
        z_eps = np.asarray(eps_out["z"], dtype=np.int64)
        w_train_eps = np.asarray(eps_out["w_train"], dtype=float)
        q_m_eps = np.asarray(eps_out["q_m"], dtype=float)
        ids_eps = np.asarray(eps_out["ids"], dtype=np.int64)
        rbf_eps = eps_out["rbf"]
        rel_recon_eps = float(eps_out["rel_recon"])
        val_eps = dict(eps_out["val"])
        smooth_eps = dict(eps_out["smooth"])
        w_eps = np.asarray(eps_out["w_anchor"], dtype=float)
        w_eps_full = np.asarray(eps_out["w_full"], dtype=float)
        n_weight_field_plots_eps = int(eps_out["n_weight_field_plots"])
        cv_eps = dict(eps_out["cv"])
        rowspace_eps = dict(eps_out["rowspace_info"])

        # --- sig channel ---
        if maw_mode == "res_eps_sig":
            n_stop_sig_req = int(args.maw_min_support_size_sig) if int(args.maw_min_support_size_sig) > 0 else None
            sig_out = _run_hom_channel_maw(
                dataset=dataset,
                ecm_fixed=ecm_fixed,
                n_elem=n_elem,
                args=args,
                component="sig",
                n_stop_req=n_stop_sig_req,
            )
            q_m_sig = np.asarray(sig_out["q_m"], dtype=float)
            ids_sig = np.asarray(sig_out["ids"], dtype=np.int64)
            if ids_sig.shape != ids_eps.shape or np.any(ids_sig != ids_eps):
                raise RuntimeError("eps/sig structured IDs mismatch; strict mode requires aligned samples.")
            maw_sig = sig_out["maw"]
            z_sig = np.asarray(sig_out["z"], dtype=np.int64)
            w_train_sig = np.asarray(sig_out["w_train"], dtype=float)
            rbf_sig = sig_out["rbf"]
            rel_recon_sig = float(sig_out["rel_recon"])
            val_sig = dict(sig_out["val"])
            smooth_sig = dict(sig_out["smooth"])
            w_sig = np.asarray(sig_out["w_anchor"], dtype=float)
            w_sig_full = np.asarray(sig_out["w_full"], dtype=float)
            n_weight_field_plots_sig = int(sig_out["n_weight_field_plots"])
            cv_sig = dict(sig_out["cv"])
            rowspace_sig = dict(sig_out["rowspace_info"])
        else:
            z_sig = np.asarray(ecm_fixed["Z_sig"], dtype=np.int64).reshape(-1)
            w_sig = np.asarray(ecm_fixed["w_sig"], dtype=float).reshape(-1)
            w_sig_full = np.asarray(ecm_fixed["w_sig_full"], dtype=float).reshape(-1)
            cv_sig = {"enabled": False, "max_rel": np.nan, "mean_rel": np.nan, "p95_rel": np.nan, "n_eval": 0}
            rowspace_sig = {
                "n_blocks": 0,
                "m_full_min": 0,
                "m_full_max": 0,
                "rank_min": np.nan,
                "rank_mean": np.nan,
                "rank_max": np.nan,
                "n_rank_zero": 0,
                "n_row_reduced": 0,
            }
            print(
                f"  [Stage8b][sig] classical ECM fixed: |Z_sig|={z_sig.size}, "
                "MAW disabled by mode=res_eps."
            )
        rowspace_eps = dict(eps_out["rowspace_info"])
        cv_eps = dict(eps_out["cv"])
        n_stop_used_eps = int(eps_out["n_stop_used"])
        n_stop_used_sig = int(sig_out["n_stop_used"]) if maw_mode == "res_eps_sig" else -2
    else:
        z_eps = np.asarray(ecm_fixed["Z_eps"], dtype=np.int64).reshape(-1)
        z_sig = np.asarray(ecm_fixed["Z_sig"], dtype=np.int64).reshape(-1)
        w_eps = np.asarray(ecm_fixed["w_eps"], dtype=float).reshape(-1)
        w_sig = np.asarray(ecm_fixed["w_sig"], dtype=float).reshape(-1)
        w_eps_full = np.asarray(ecm_fixed["w_eps_full"], dtype=float).reshape(-1)
        w_sig_full = np.asarray(ecm_fixed["w_sig_full"], dtype=float).reshape(-1)
        cv_eps = {"enabled": False, "max_rel": np.nan, "mean_rel": np.nan, "p95_rel": np.nan, "n_eval": 0}
        cv_sig = {"enabled": False, "max_rel": np.nan, "mean_rel": np.nan, "p95_rel": np.nan, "n_eval": 0}
        rowspace_eps = {
            "n_blocks": 0,
            "m_full_min": 0,
            "m_full_max": 0,
            "rank_min": np.nan,
            "rank_mean": np.nan,
            "rank_max": np.nan,
            "n_rank_zero": 0,
            "n_row_reduced": 0,
        }
        rowspace_sig = dict(rowspace_eps)
        n_stop_used_eps = -2
        n_stop_used_sig = -2

    if maw_mode in {"res_eps", "res_eps_sig"}:
        hom_fit_eps = float(val_eps["max_rel"])
    if maw_mode == "res_eps_sig":
        hom_fit_sig = float(val_sig["max_rel"])

    z_union = np.union1d(np.union1d(z_red, z_eps), z_sig).astype(np.int64)

    out_file = os.path.join(args.out_dir, "ecm_weights_all.npz")
    payload = {
        # Compatibility: residual/hom fixed-style keys
        "Z_res": z_red,
        "Z_eps": z_eps,
        "Z_sig": z_sig,
        "Z_union": z_union,
        "w_res": np.asarray(w_res_anchor, dtype=float),
        "w_eps": np.asarray(w_eps, dtype=float),
        "w_sig": np.asarray(w_sig, dtype=float),
        "w_res_full": np.asarray(w_res_full_anchor, dtype=float),
        "w_eps_full": np.asarray(w_eps_full, dtype=float),
        "w_sig_full": np.asarray(w_sig_full, dtype=float),
        "n_elem": np.array([int(n_elem)], dtype=np.int64),
        "nq": np.array([int(dataset["nq"])], dtype=np.int64),
        "Ns_res": np.array([int(dataset["ns_res"])], dtype=np.int64),
        "Ns_hom": np.array([int(dataset.get("ns_hom", 0) if maw_mode == "res_eps_sig" else 0)], dtype=np.int64),
        "A0_ref": np.array([_meta_float(dataset["meta"], "A0_ref")], dtype=float),
        "hom_reference_measure": np.array([_meta_float(dataset["meta"], "A0_ref")], dtype=float),
        "ECM_COUPLING_MODE": np.array(
            [
                "mawecm_residual_only"
                if maw_mode == "res_only"
                else ("mawecm_residual_eps" if maw_mode == "res_eps" else "mawecm_residual_eps_sig")
            ]
        ),
        "ECM_COUPLING_MODE_INPUT": np.array([str(maw_mode)]),
        "hprom_model_type": np.array(
            [
                "MAW_ECM_RBF_RES_ONLY"
                if maw_mode == "res_only"
                else ("MAW_ECM_RBF_RES_EPS" if maw_mode == "res_eps" else "MAW_ECM_RBF_RES_EPS_SIG")
            ]
        ),
        "maw_hom_source": np.array([hom_source]),
        "maw_hom_fit_rel_err_eps": np.array([float(hom_fit_eps)], dtype=float),
        "maw_hom_fit_rel_err_sig": np.array([float(hom_fit_sig)], dtype=float),
        "maw_mode": np.array([maw_mode]),
        "maw_apply_to_hom": np.array([1 if maw_mode in {"res_eps", "res_eps_sig"} else 0], dtype=np.int64),
        # MAW residual payload
        "maw_targets": np.array(
            ["res"]
            if maw_mode == "res_only"
            else (["res", "eps"] if maw_mode == "res_eps" else ["res", "eps", "sig"])
        ),
        "maw_res_state_space": np.array([state_mode]),
        "maw_res_bootstrap_source": np.array([bootstrap_source_tag]),
        "maw_res_bootstrap_rsvd_tol": np.array([float(boot_rsvd_tol)], dtype=float),
        "maw_res_bootstrap_rsvd_randomized": np.array(
            [int(args.res_bootstrap_rsvd_randomized)], dtype=np.int64
        ),
        "maw_res_bootstrap_ecm_tol": np.array([float(boot_ecm_tol)], dtype=float),
        "maw_res_seed": np.array([int(args.seed)], dtype=np.int64),
        "maw_res_bootstrap_rel_err": np.array([float(res_bootstrap["rel_err"])], dtype=float),
        "maw_res_bootstrap_rel_err_eps": np.array([np.nan], dtype=float),
        "maw_res_bootstrap_rel_err_sig": np.array([np.nan], dtype=float),
        "maw_single_block_normalization": np.array(["<disabled_in_strict_res_only>"]),
        "maw_rsvd_tol_single_bootstrap": np.array([np.nan], dtype=float),
        "maw_ecm_tol_single_bootstrap": np.array([np.nan], dtype=float),
        "maw_res_candidate_pool": np.array([str(args.res_candidate_pool)]),
        "maw_res_rhs_mode": np.array([str(args.res_target_source)]),
        "maw_res_graph_mode": np.array([str(graph_info_res["mode"])]),
        "maw_res_graph_knn": np.array([int(graph_info_res["knn"])], dtype=np.int64),
        "maw_res_graph_kernel": np.array([str(graph_info_res["kernel"])]),
        "maw_res_graph_sigma": np.array([float(graph_info_res["sigma"])], dtype=float),
        "maw_res_graph_n_edges": np.array([int(graph_info_res["n_edges"])], dtype=np.int64),
        "maw_res_use_global_graph_2ndstage": np.array([int(res_use_graph)], dtype=np.int64),
        "maw_res_smooth_laplacian_all_iterations": np.array([int(res_smooth_all)], dtype=np.int64),
        "maw_res_max_number_zeros_active_set_loop": np.array([int(args.max_number_zeros_active_set_loop_maw_ecm)], dtype=np.int64),
        "maw_res_bootstrap_constrain_sum_weights": np.array(
            [int(args.res_bootstrap_constrain_sum_weights)], dtype=np.int64
        ),
        "maw_res_enforce_nonnegativity": np.array([int(args.maw_res_enforce_nonnegativity)], dtype=np.int64),
        "maw_hom_enforce_nonnegativity": np.array([int(args.maw_hom_enforce_nonnegativity)], dtype=np.int64),
        "maw_hom_rowspace_compress": np.array([int(args.maw_hom_rowspace_compress)], dtype=np.int64),
        "maw_hom_rowspace_tol_rel": np.array([float(args.maw_hom_rowspace_tol_rel)], dtype=float),
        "maw_hom_cv_max_rel": np.array([float(args.maw_hom_cv_max_rel)], dtype=float),
        "maw_hom_cv_folds": np.array([int(args.maw_hom_cv_folds)], dtype=np.int64),
        "maw_hom_cv_seed": np.array([int(args.maw_hom_cv_seed)], dtype=np.int64),
        "maw_hom_cv_min_samples": np.array([int(args.maw_hom_cv_min_samples)], dtype=np.int64),
        "maw_hom_cv_strict": np.array([int(args.maw_hom_cv_strict)], dtype=np.int64),
        "maw_res_enforce_sum_weights": np.array([int(args.enforce_sum_weights)], dtype=np.int64),
        "maw_res_sum_weights_target": np.array(
            [float(sum_target) if enforce_sum_weights else np.nan], dtype=float
        ),
        "maw_res_alpha_smooth": np.array([float(prune_opts["alpha_smooth"])], dtype=float),
        "maw_res_criterion": np.array([int(args.criterion)], dtype=np.int64),
        "maw_res_incremental_smoothing": np.array([int(args.incremental_smoothing)], dtype=np.int64),
        "maw_res_tol_neg": np.array([float(maw["tol_neg"])], dtype=float),
        "maw_res_n_stop": np.array([int(maw["n_stop"])], dtype=np.int64),
        "maw_res_requested_min_support_size": np.array(
            [int(n_stop_res_req) if n_stop_res_req is not None else 0], dtype=np.int64
        ),
        "maw_res_elapsed_sec": np.array([float(maw["elapsed_sec"])], dtype=float),
        "maw_res_recon_rel": np.array([float(rel_recon)], dtype=float),
        "maw_res_constraint_rel_max": np.array([float(val["max_rel"])], dtype=float),
        "maw_res_constraint_rel_mean": np.array([float(val["mean_rel"])], dtype=float),
        "maw_res_constraint_abs_max": np.array([float(val["max_abs"])], dtype=float),
        "maw_res_weight_min": np.array([float(val["min_weight"])], dtype=float),
        "maw_res_smooth_S": np.array([float(smooth["S"])], dtype=float),
        "maw_res_smooth_R95": np.array([float(smooth["R95"])], dtype=float),
        "maw_res_smooth_Rmax": np.array([float(smooth["Rmax"])], dtype=float),
        "maw_res_smooth_n_edges": np.array([int(smooth["n_edges"])], dtype=np.int64),
        "maw_res_renorm_target": np.array([float(renorm_target)], dtype=float),
        "maw_res_sum_w_min": np.array([float(sum_w_min)], dtype=float),
        "maw_res_sum_w_max": np.array([float(sum_w_max)], dtype=float),
        "maw_res_sum_w_mean": np.array([float(sum_w_mean)], dtype=float),
        "maw_res_z_ini": np.asarray(z_ini, dtype=np.int64),
        "maw_res_w_ini": np.asarray(w_ini, dtype=float),
        "maw_res_removed_local": np.asarray(maw["removed_local"], dtype=np.int64),
        "maw_res_active_counts": np.asarray(maw["active_counts"], dtype=np.int64),
        "maw_res_stage_history": np.asarray(maw["stage_history"]),
        "maw_res_no_enforcement_attempts": np.array([int(maw["no_enforcement_attempts"])], dtype=np.int64),
        "maw_res_local_active_set_attempts": np.array([int(maw.get("local_active_set_attempts", 0))], dtype=np.int64),
        "maw_res_graph_attempts": np.array([int(maw["graph_attempts"])], dtype=np.int64),
        "maw_res_z_support": np.asarray(z_red, dtype=np.int64),
        "maw_res_W_train": np.asarray(w_train, dtype=float),
        "maw_res_q_train": np.asarray(state_train, dtype=float),
        "maw_res_q_m_train": np.asarray(q_m_train, dtype=float),
        "maw_res_ids": np.asarray(ids, dtype=np.int64),
        "maw_res_structured_grid_shape": np.asarray(dataset["grid_struct_res"], dtype=np.int64)
        if dataset["grid_struct_res"] is not None
        else np.zeros(0, dtype=np.int64),
        # RBF for adaptive residual weights
        "maw_res_rbf_centers": np.asarray(rbf["centers"], dtype=float),
        "maw_res_rbf_center_ids": np.asarray(rbf["center_ids"], dtype=np.int64),
        "maw_res_rbf_length_scales": np.asarray(rbf["length_scales"], dtype=float),
        "maw_res_rbf_alpha": np.asarray(rbf["Alpha"], dtype=float),
        "maw_res_rbf_beta": np.asarray(rbf["Beta"], dtype=float),
        "maw_res_rbf_scale": np.asarray(rbf["scale"], dtype=float),
        "maw_res_rbf_poly_mode": np.array([int(rbf["poly_mode"])], dtype=np.int64),
        "maw_res_rbf_lambda": np.array([float(rbf["lambda_reg"])], dtype=float),
        "maw_res_rbf_n_centers": np.array([int(rbf["n_centers"])], dtype=np.int64),
        "maw_res_rbf_train_rel_error": np.array([float(rbf["train_rel_error"])], dtype=float),
        "maw_res_weight_plots_saved": np.array([int(n_weight_field_plots)], dtype=np.int64),
        "maw_res_weight_plot_format": np.array([str(args.weight_plot_format)]),
        "maw_res_save_weight_field_plots": np.array([int(args.save_weight_field_plots)], dtype=np.int64),
        "maw_res_show_weight_field_plots": np.array([int(args.show_weight_field_plots)], dtype=np.int64),
        "maw_res_max_weight_field_plots": np.array([int(args.max_weight_field_plots)], dtype=np.int64),
        "data_dir": np.array([str(args.dataset_dir)]),
        "fixed_ecm_dir": np.array([str(args.fixed_ecm_file) if hom_source == "fixed_ecm" else ""]),
    }

    if maw_mode == "res_eps_sig":
        payload.update(
            {
                "maw_eps_n_stop_requested": np.array([int(args.maw_min_support_size_eps)], dtype=np.int64),
                "maw_sig_n_stop_requested": np.array([int(args.maw_min_support_size_sig)], dtype=np.int64),
                "maw_eps_bootstrap_source": np.array([str(hom_bootstrap_source_tag)]),
                "maw_sig_bootstrap_source": np.array([str(hom_bootstrap_source_tag)]),
                "maw_eps_z_ini": np.asarray(np.asarray(ecm_fixed["Z_eps"], dtype=np.int64).reshape(-1), dtype=np.int64),
                "maw_sig_z_ini": np.asarray(np.asarray(ecm_fixed["Z_sig"], dtype=np.int64).reshape(-1), dtype=np.int64),
                "maw_eps_w_ini": np.asarray(np.asarray(ecm_fixed["w_eps"], dtype=float).reshape(-1), dtype=float),
                "maw_sig_w_ini": np.asarray(np.asarray(ecm_fixed["w_sig"], dtype=float).reshape(-1), dtype=float),
                "maw_eps_z_support": np.asarray(z_eps, dtype=np.int64),
                "maw_sig_z_support": np.asarray(z_sig, dtype=np.int64),
                "maw_eps_W_train": np.asarray(w_train_eps, dtype=float),
                "maw_sig_W_train": np.asarray(w_train_sig, dtype=float),
                "maw_eps_q_train": np.asarray(q_m_eps, dtype=float),
                "maw_sig_q_train": np.asarray(q_m_sig, dtype=float),
                "maw_eps_ids": np.asarray(ids_eps, dtype=np.int64),
                "maw_sig_ids": np.asarray(ids_sig, dtype=np.int64),
                "maw_eps_elapsed_sec": np.array([float(maw_eps["elapsed_sec"])], dtype=float),
                "maw_sig_elapsed_sec": np.array([float(maw_sig["elapsed_sec"])], dtype=float),
                "maw_eps_recon_rel": np.array([float(rel_recon_eps)], dtype=float),
                "maw_sig_recon_rel": np.array([float(rel_recon_sig)], dtype=float),
                "maw_eps_constraint_rel_max": np.array([float(val_eps["max_rel"])], dtype=float),
                "maw_sig_constraint_rel_max": np.array([float(val_sig["max_rel"])], dtype=float),
                "maw_eps_cv_enabled": np.array([int(bool(cv_eps.get("enabled", False)))], dtype=np.int64),
                "maw_sig_cv_enabled": np.array([int(bool(cv_sig.get("enabled", False)))], dtype=np.int64),
                "maw_eps_cv_rel_max": np.array([float(cv_eps.get("max_rel", np.nan))], dtype=float),
                "maw_sig_cv_rel_max": np.array([float(cv_sig.get("max_rel", np.nan))], dtype=float),
                "maw_eps_cv_rel_mean": np.array([float(cv_eps.get("mean_rel", np.nan))], dtype=float),
                "maw_sig_cv_rel_mean": np.array([float(cv_sig.get("mean_rel", np.nan))], dtype=float),
                "maw_eps_cv_rel_p95": np.array([float(cv_eps.get("p95_rel", np.nan))], dtype=float),
                "maw_sig_cv_rel_p95": np.array([float(cv_sig.get("p95_rel", np.nan))], dtype=float),
                "maw_eps_cv_n_eval": np.array([int(cv_eps.get("n_eval", 0))], dtype=np.int64),
                "maw_sig_cv_n_eval": np.array([int(cv_sig.get("n_eval", 0))], dtype=np.int64),
                "maw_eps_n_stop_used": np.array([int(n_stop_used_eps)], dtype=np.int64),
                "maw_sig_n_stop_used": np.array([int(n_stop_used_sig)], dtype=np.int64),
                "maw_eps_rowspace_rank_min": np.array([float(rowspace_eps.get("rank_min", np.nan))], dtype=float),
                "maw_sig_rowspace_rank_min": np.array([float(rowspace_sig.get("rank_min", np.nan))], dtype=float),
                "maw_eps_rowspace_rank_mean": np.array([float(rowspace_eps.get("rank_mean", np.nan))], dtype=float),
                "maw_sig_rowspace_rank_mean": np.array([float(rowspace_sig.get("rank_mean", np.nan))], dtype=float),
                "maw_eps_rowspace_rank_max": np.array([float(rowspace_eps.get("rank_max", np.nan))], dtype=float),
                "maw_sig_rowspace_rank_max": np.array([float(rowspace_sig.get("rank_max", np.nan))], dtype=float),
                "maw_eps_rowspace_rank0_count": np.array([int(rowspace_eps.get("n_rank_zero", 0))], dtype=np.int64),
                "maw_sig_rowspace_rank0_count": np.array([int(rowspace_sig.get("n_rank_zero", 0))], dtype=np.int64),
                "maw_eps_weight_min": np.array([float(val_eps["min_weight"])], dtype=float),
                "maw_sig_weight_min": np.array([float(val_sig["min_weight"])], dtype=float),
                "maw_eps_smooth_S": np.array([float(smooth_eps["S"])], dtype=float),
                "maw_sig_smooth_S": np.array([float(smooth_sig["S"])], dtype=float),
                "maw_eps_smooth_R95": np.array([float(smooth_eps["R95"])], dtype=float),
                "maw_sig_smooth_R95": np.array([float(smooth_sig["R95"])], dtype=float),
                "maw_eps_smooth_Rmax": np.array([float(smooth_eps["Rmax"])], dtype=float),
                "maw_sig_smooth_Rmax": np.array([float(smooth_sig["Rmax"])], dtype=float),
                "maw_eps_weight_plots_saved": np.array([int(n_weight_field_plots_eps)], dtype=np.int64),
                "maw_sig_weight_plots_saved": np.array([int(n_weight_field_plots_sig)], dtype=np.int64),
                "maw_eps_rbf_centers": np.asarray(rbf_eps["centers"], dtype=float),
                "maw_sig_rbf_centers": np.asarray(rbf_sig["centers"], dtype=float),
                "maw_eps_rbf_center_ids": np.asarray(rbf_eps["center_ids"], dtype=np.int64),
                "maw_sig_rbf_center_ids": np.asarray(rbf_sig["center_ids"], dtype=np.int64),
                "maw_eps_rbf_length_scales": np.asarray(rbf_eps["length_scales"], dtype=float),
                "maw_sig_rbf_length_scales": np.asarray(rbf_sig["length_scales"], dtype=float),
                "maw_eps_rbf_alpha": np.asarray(rbf_eps["Alpha"], dtype=float),
                "maw_sig_rbf_alpha": np.asarray(rbf_sig["Alpha"], dtype=float),
                "maw_eps_rbf_beta": np.asarray(rbf_eps["Beta"], dtype=float),
                "maw_sig_rbf_beta": np.asarray(rbf_sig["Beta"], dtype=float),
                "maw_eps_rbf_scale": np.asarray(rbf_eps["scale"], dtype=float),
                "maw_sig_rbf_scale": np.asarray(rbf_sig["scale"], dtype=float),
                "maw_eps_rbf_poly_mode": np.array([int(rbf_eps["poly_mode"])], dtype=np.int64),
                "maw_sig_rbf_poly_mode": np.array([int(rbf_sig["poly_mode"])], dtype=np.int64),
                "maw_eps_rbf_lambda": np.array([float(rbf_eps["lambda_reg"])], dtype=float),
                "maw_sig_rbf_lambda": np.array([float(rbf_sig["lambda_reg"])], dtype=float),
                "maw_eps_rbf_n_centers": np.array([int(rbf_eps["n_centers"])], dtype=np.int64),
                "maw_sig_rbf_n_centers": np.array([int(rbf_sig["n_centers"])], dtype=np.int64),
                "maw_eps_rbf_train_rel_error": np.array([float(rbf_eps["train_rel_error"])], dtype=float),
                "maw_sig_rbf_train_rel_error": np.array([float(rbf_sig["train_rel_error"])], dtype=float),
            }
        )

    if maw_mode == "res_eps":
        payload.update(
            {
                "maw_eps_n_stop_requested": np.array([int(args.maw_min_support_size_eps)], dtype=np.int64),
                "maw_eps_bootstrap_source": np.array([str(hom_bootstrap_source_tag)]),
                "maw_eps_z_ini": np.asarray(np.asarray(ecm_fixed["Z_eps"], dtype=np.int64).reshape(-1), dtype=np.int64),
                "maw_eps_w_ini": np.asarray(np.asarray(ecm_fixed["w_eps"], dtype=float).reshape(-1), dtype=float),
                "maw_eps_z_support": np.asarray(z_eps, dtype=np.int64),
                "maw_eps_W_train": np.asarray(w_train_eps, dtype=float),
                "maw_eps_q_train": np.asarray(q_m_eps, dtype=float),
                "maw_eps_ids": np.asarray(ids_eps, dtype=np.int64),
                "maw_eps_elapsed_sec": np.array([float(maw_eps["elapsed_sec"])], dtype=float),
                "maw_eps_recon_rel": np.array([float(rel_recon_eps)], dtype=float),
                "maw_eps_constraint_rel_max": np.array([float(val_eps["max_rel"])], dtype=float),
                "maw_eps_cv_enabled": np.array([int(bool(cv_eps.get("enabled", False)))], dtype=np.int64),
                "maw_eps_cv_rel_max": np.array([float(cv_eps.get("max_rel", np.nan))], dtype=float),
                "maw_eps_cv_rel_mean": np.array([float(cv_eps.get("mean_rel", np.nan))], dtype=float),
                "maw_eps_cv_rel_p95": np.array([float(cv_eps.get("p95_rel", np.nan))], dtype=float),
                "maw_eps_cv_n_eval": np.array([int(cv_eps.get("n_eval", 0))], dtype=np.int64),
                "maw_eps_n_stop_used": np.array([int(n_stop_used_eps)], dtype=np.int64),
                "maw_eps_rowspace_rank_min": np.array([float(rowspace_eps.get("rank_min", np.nan))], dtype=float),
                "maw_eps_rowspace_rank_mean": np.array([float(rowspace_eps.get("rank_mean", np.nan))], dtype=float),
                "maw_eps_rowspace_rank_max": np.array([float(rowspace_eps.get("rank_max", np.nan))], dtype=float),
                "maw_eps_rowspace_rank0_count": np.array([int(rowspace_eps.get("n_rank_zero", 0))], dtype=np.int64),
                "maw_eps_weight_min": np.array([float(val_eps["min_weight"])], dtype=float),
                "maw_eps_smooth_S": np.array([float(smooth_eps["S"])], dtype=float),
                "maw_eps_smooth_R95": np.array([float(smooth_eps["R95"])], dtype=float),
                "maw_eps_smooth_Rmax": np.array([float(smooth_eps["Rmax"])], dtype=float),
                "maw_eps_weight_plots_saved": np.array([int(n_weight_field_plots_eps)], dtype=np.int64),
                "maw_eps_rbf_centers": np.asarray(rbf_eps["centers"], dtype=float),
                "maw_eps_rbf_center_ids": np.asarray(rbf_eps["center_ids"], dtype=np.int64),
                "maw_eps_rbf_length_scales": np.asarray(rbf_eps["length_scales"], dtype=float),
                "maw_eps_rbf_alpha": np.asarray(rbf_eps["Alpha"], dtype=float),
                "maw_eps_rbf_beta": np.asarray(rbf_eps["Beta"], dtype=float),
                "maw_eps_rbf_scale": np.asarray(rbf_eps["scale"], dtype=float),
                "maw_eps_rbf_poly_mode": np.array([int(rbf_eps["poly_mode"])], dtype=np.int64),
                "maw_eps_rbf_lambda": np.array([float(rbf_eps["lambda_reg"])], dtype=float),
                "maw_eps_rbf_n_centers": np.array([int(rbf_eps["n_centers"])], dtype=np.int64),
                "maw_eps_rbf_train_rel_error": np.array([float(rbf_eps["train_rel_error"])], dtype=float),
            }
        )

    np.savez(out_file, **payload)

    print(
        f"  [MAW] mode={maw_mode} |Z_ini|={z_ini.size} -> |Z_red|={z_red.size}, "
        f"RBF train-rel={rel_recon:.3e}, prune-elapsed={maw['elapsed_sec']:.2f}s"
    )
    print(
        f"  [MAW] stage usage: NOENF={maw['no_enforcement_attempts']}, "
        f"NOENFe-local={maw.get('local_active_set_attempts', 0)}, graph={maw['graph_attempts']}"
    )
    print(
        f"  [MAW-res] validation: max_rel={val['max_rel']:.3e}, "
        f"min_w={val['min_weight']:.3e}, smooth(R95={smooth['R95']:.3e}, Rmax={smooth['Rmax']:.3e})"
    )
    if maw_mode in {"res_eps", "res_eps_sig"}:
        print(
            f"  [MAW-eps] |Z_ini|={np.asarray(ecm_fixed['Z_eps']).size} -> |Z_red|={z_eps.size}, "
            f"RBF train-rel={rel_recon_eps:.3e}, max_rel={val_eps['max_rel']:.3e}, "
            f"cv_max_rel={float(cv_eps.get('max_rel', np.nan)):.3e}, n_stop_used={int(n_stop_used_eps)}"
        )
    if maw_mode == "res_eps":
        print(
            f"  [MAW-sig] classical ECM fixed: |Z_sig|={np.asarray(ecm_fixed['Z_sig']).size}, "
            "MAW disabled."
        )
    if maw_mode == "res_eps_sig":
        print(
            f"  [MAW-sig] |Z_ini|={np.asarray(ecm_fixed['Z_sig']).size} -> |Z_red|={z_sig.size}, "
            f"RBF train-rel={rel_recon_sig:.3e}, max_rel={val_sig['max_rel']:.3e}, "
            f"cv_max_rel={float(cv_sig.get('max_rel', np.nan)):.3e}, n_stop_used={int(n_stop_used_sig)}"
        )
    if enforce_sum_weights:
        print(
            f"  [MAW-res] sum(w) stats over states: min={sum_w_min:.6e}, "
            f"max={sum_w_max:.6e}, mean={sum_w_mean:.6e}, target={sum_target:.6e}"
        )
    print(f"[DONE] Saved -> {out_file}")


if __name__ == "__main__":
    main()
