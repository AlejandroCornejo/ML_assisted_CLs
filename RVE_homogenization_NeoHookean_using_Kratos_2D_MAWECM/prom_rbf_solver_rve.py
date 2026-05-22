#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PROM-RBF solver (2D-MAWECM, online trajectory evaluation).

Joaquin-style master/slave decoder:
    q_s = N_slave(q_m)
    u = u_aff + Phi_m (A_m q_m) + Phi_s q_s

Reduced Newton unknown is q_m, with Jacobian:
    du/dq_m = Phi_m A_m + Phi_s dN_slave/dq_m
"""

import os
import pickle
import sys
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.interpolate._rbfinterp import _build_and_solve_system

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    DeformationGradientFromGreenLagrange2D,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    BuildDynamicSegmentSteps,
    CalculateHomogenizedFromAssemblerWithElementWeights,
    VectorizedAssembler,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)


def _recover_mu_from_E(e_vec: np.ndarray, mapping: str, mu_space: str) -> np.ndarray:
    exx = float(e_vec[0])
    gamma = float(e_vec[2])

    if mapping == "small_strain":
        gx = exx
        gxy = gamma
    elif mapping == "green_lagrange_upper":
        disc = 1.0 + 2.0 * exx
        if disc <= 0.0:
            raise RuntimeError(
                f"Invalid strain state for Green-Lagrange inversion: 1+2*Exx={disc:.3e} <= 0."
            )
        gx = np.sqrt(disc) - 1.0
        den = 1.0 + gx
        if abs(den) <= 1e-14:
            raise RuntimeError("Invalid strain state: denominator (1+Gx) too small for Gxy recovery.")
        gxy = gamma / den
    else:
        raise RuntimeError(f"Unsupported mapping='{mapping}' for mu recovery.")

    if mu_space == "gx_gxy":
        return np.array([gx, gxy], dtype=float)
    if mu_space == "f11_f12":
        return np.array([1.0 + gx, gxy], dtype=float)
    raise RuntimeError(f"Unsupported mu_space='{mu_space}'.")


def _extract_qs_from_y(y: np.ndarray, target_space: str, q_m_dim: int, q_s_dim: int) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    if target_space == "q_s":
        if y.size != q_s_dim:
            raise RuntimeError(f"Expected q_s dim={q_s_dim}, got {y.size}")
        return y.copy()
    if target_space == "both_ms":
        if y.size < (q_m_dim + q_s_dim):
            raise RuntimeError(
                f"Target size too small for split both_ms=[q_m({q_m_dim}), q_s({q_s_dim})], got {y.size}."
            )
        return y[q_m_dim:q_m_dim + q_s_dim].copy()
    raise RuntimeError(
        f"Stage4 model target_space='{target_space}' is not master/slave. "
        "Retrain Stage3/Stage4 with target-space='q_s' (or 'both_ms')."
    )


def _assert_qm_input_dimension(q_m_dim: int, x_mean: np.ndarray) -> None:
    n_in = int(np.asarray(x_mean, dtype=float).size)
    if int(q_m_dim) != n_in:
        raise RuntimeError(
            "PROM-RBF q_m-Newton requires q_m dimension equal to decoder input dimension. "
            f"Got q_m_dim={q_m_dim}, dim(input)={n_in}."
        )


def _decode_qs_from_qm(
    q_m: np.ndarray,
    *,
    rbf_model,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    target_space: str,
    q_m_dim: int,
    q_s_dim: int,
) -> np.ndarray:
    q_m = np.asarray(q_m, dtype=float).reshape(-1)
    if q_m.size != int(q_m_dim):
        raise RuntimeError(f"q_m size mismatch: expected {q_m_dim}, got {q_m.size}.")
    x_z = (q_m - x_mean) / x_std
    y_z = np.asarray(rbf_model(x_z.reshape(1, -1)), dtype=float).reshape(-1)
    y = y_z * y_std + y_mean
    return _extract_qs_from_y(y, target_space=target_space, q_m_dim=q_m_dim, q_s_dim=q_s_dim)


def _rbf_kernel_phi_and_dphi_dr(r: np.ndarray, kernel_name: str) -> Tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    k = str(kernel_name).strip().lower()
    if k == "linear":
        phi = -r
        dphi_dr = -np.ones_like(r)
    elif k == "thin_plate_spline":
        phi = np.zeros_like(r)
        dphi_dr = np.zeros_like(r)
        mask = r > 0.0
        rm = r[mask]
        phi[mask] = (rm ** 2) * np.log(rm)
        dphi_dr[mask] = rm * (2.0 * np.log(rm) + 1.0)
    elif k == "cubic":
        phi = r ** 3
        dphi_dr = 3.0 * (r ** 2)
    elif k == "quintic":
        phi = -(r ** 5)
        dphi_dr = -5.0 * (r ** 4)
    elif k == "multiquadric":
        z = np.sqrt(1.0 + r ** 2)
        phi = -z
        dphi_dr = -(r / z)
    elif k == "inverse_multiquadric":
        z = 1.0 + r ** 2
        phi = z ** (-0.5)
        dphi_dr = -r * (z ** (-1.5))
    elif k == "inverse_quadratic":
        z = 1.0 + r ** 2
        phi = z ** (-1.0)
        dphi_dr = -2.0 * r * (z ** (-2.0))
    elif k == "gaussian":
        phi = np.exp(-(r ** 2))
        dphi_dr = -2.0 * r * phi
    else:
        raise RuntimeError(f"Unsupported RBF kernel for analytic Jacobian: '{kernel_name}'.")
    return phi, dphi_dr


def _rbf_polynomial_terms_and_gradients(
    x: np.ndarray,
    powers: np.ndarray,
    shift: np.ndarray,
    scale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    powers = np.asarray(powers, dtype=np.int64)
    shift = np.asarray(shift, dtype=float).reshape(-1)
    scale = np.asarray(scale, dtype=float).reshape(-1)
    x_scaled = (x - shift) / scale

    n_poly = int(powers.shape[0])
    dim = int(x.size)
    poly = np.ones(n_poly, dtype=float)
    grad = np.zeros((n_poly, dim), dtype=float)

    for j in range(n_poly):
        pj = powers[j]
        val = 1.0
        for b in range(dim):
            p = int(pj[b])
            if p != 0:
                val *= x_scaled[b] ** p
        poly[j] = val

        for a in range(dim):
            pa = int(pj[a])
            if pa == 0:
                continue
            dval = float(pa) / float(scale[a])
            for b in range(dim):
                p = int(pj[b])
                if b == a:
                    p = p - 1
                if p != 0:
                    dval *= x_scaled[b] ** p
            grad[j, a] = dval
    return poly, grad


def _rbf_eval_and_jacobian_standardized(
    x_z: np.ndarray,
    rbf_model: RBFInterpolator,
) -> Tuple[np.ndarray, np.ndarray]:
    x_z = np.asarray(x_z, dtype=float).reshape(-1)
    if x_z.ndim != 1:
        raise RuntimeError("x_z must be a 1D vector.")

    y_data = np.asarray(rbf_model.y, dtype=float)
    d_data = np.asarray(rbf_model.d, dtype=float)
    smoothing_all = np.asarray(rbf_model.smoothing, dtype=float).reshape(-1)
    powers = np.asarray(rbf_model.powers, dtype=np.int64)
    eps = float(rbf_model.epsilon)
    kernel = str(rbf_model.kernel)

    if rbf_model.neighbors is None:
        y_nbr = y_data
        if hasattr(rbf_model, "_coeffs") and hasattr(rbf_model, "_shift") and hasattr(rbf_model, "_scale"):
            coeffs = np.asarray(rbf_model._coeffs, dtype=float)
            shift = np.asarray(rbf_model._shift, dtype=float)
            scale = np.asarray(rbf_model._scale, dtype=float)
        else:
            shift, scale, coeffs = _build_and_solve_system(
                y_nbr,
                d_data,
                smoothing_all,
                kernel,
                eps,
                powers,
            )
            coeffs = np.asarray(coeffs, dtype=float)
            shift = np.asarray(shift, dtype=float)
            scale = np.asarray(scale, dtype=float)
    else:
        k = int(rbf_model.neighbors)
        _, y_idx = rbf_model._tree.query(x_z.reshape(1, -1), k)
        if k == 1:
            y_idx = y_idx[:, None]
        y_idx = np.sort(y_idx, axis=1)[0]
        y_nbr = y_data[y_idx]
        d_nbr = d_data[y_idx]
        s_nbr = smoothing_all[y_idx]
        shift, scale, coeffs = _build_and_solve_system(
            y_nbr,
            d_nbr,
            s_nbr,
            kernel,
            eps,
            powers,
        )
        coeffs = np.asarray(coeffs, dtype=float)
        shift = np.asarray(shift, dtype=float)
        scale = np.asarray(scale, dtype=float)

    n_nbr = int(y_nbr.shape[0])
    n_poly = int(powers.shape[0])
    if coeffs.shape[0] != n_nbr + n_poly:
        raise RuntimeError(
            f"Unexpected RBF coeff shape {coeffs.shape}; expected first dim {n_nbr + n_poly}."
        )

    alpha = coeffs[:n_nbr, :]  # kernel coefficients
    beta = coeffs[n_nbr:, :]   # polynomial coefficients

    diff = x_z.reshape(1, -1) - y_nbr  # (n_nbr, dim)
    dist = np.linalg.norm(diff, axis=1)
    r = eps * dist
    phi, dphi_dr = _rbf_kernel_phi_and_dphi_dr(r, kernel)

    dist_safe = np.where(dist > 0.0, dist, 1.0)
    dr_dx = eps * diff / dist_safe[:, None]
    dr_dx[dist <= 0.0, :] = 0.0
    dphi_dx = dphi_dr[:, None] * dr_dx  # (n_nbr, dim)

    poly, dpoly_dx = _rbf_polynomial_terms_and_gradients(x_z, powers, shift, scale)

    y_z = (phi[:, None] * alpha).sum(axis=0) + (poly[:, None] * beta).sum(axis=0)
    jac_yz_xz = (dphi_dx.T @ alpha).T + (dpoly_dx.T @ beta).T  # (n_out, dim)
    return y_z, jac_yz_xz


def _decode_qs_and_jacobian_from_qm(
    q_m: np.ndarray,
    *,
    rbf_model: RBFInterpolator,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    target_space: str,
    q_m_dim: int,
    q_s_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    q_m = np.asarray(q_m, dtype=float).reshape(-1)
    if q_m.size != int(q_m_dim):
        raise RuntimeError(f"q_m size mismatch: expected {q_m_dim}, got {q_m.size}.")

    x_z = (q_m - x_mean) / x_std
    y_z, jac_yz_xz = _rbf_eval_and_jacobian_standardized(x_z, rbf_model=rbf_model)
    y = y_z * y_std + y_mean

    jac_y_qm = (jac_yz_xz * y_std[:, None]) / x_std[None, :]
    if target_space == "q_s":
        if y.size != int(q_s_dim):
            raise RuntimeError(f"Expected q_s dim={q_s_dim}, got {y.size}")
        q_s = y.copy()
        jac = jac_y_qm
    elif target_space == "both_ms":
        off = int(q_m_dim)
        end = off + int(q_s_dim)
        if y.size < end:
            raise RuntimeError(
                f"Target size too small for split both_ms=[q_m({q_m_dim}), q_s({q_s_dim})], got {y.size}."
            )
        q_s = y[off:end].copy()
        jac = jac_y_qm[off:end, :]
    else:
        raise RuntimeError(
            f"Stage4 model target_space='{target_space}' is not master/slave. "
            "Retrain Stage3/Stage4 with target-space='q_s' (or 'both_ms')."
        )
    return q_s, jac


def _decoder_jacobian_qs_wrt_qm(
    q_m: np.ndarray,
    decode_qs: Callable[[np.ndarray], np.ndarray],
    q_s_dim: int,
) -> np.ndarray:
    # This pipeline enforces analytic decoder Jacobians.
    if not hasattr(decode_qs, "_analytic_kwargs"):
        raise RuntimeError(
            "Missing analytic decoder metadata. "
            "PROM/HPROM-RBF now require analytic dq_s/dq_m (no finite-difference fallback)."
        )
    akw = getattr(decode_qs, "_analytic_kwargs")
    _, jac = _decode_qs_and_jacobian_from_qm(q_m, **akw)
    jac = np.asarray(jac, dtype=float)
    if jac.shape != (int(q_s_dim), int(np.asarray(q_m, dtype=float).size)):
        raise RuntimeError(
            f"Analytic decoder Jacobian shape mismatch: got {jac.shape}, "
            f"expected ({int(q_s_dim)}, {int(np.asarray(q_m, dtype=float).size)})."
        )
    return jac


def _load_stage2b_array(stage2b_dir: str, name: str) -> np.ndarray:
    p = os.path.join(stage2b_dir, name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.asarray(np.load(p), dtype=float)


def _initial_qm_from_mu(mu_vec: np.ndarray, model_pack: Dict[str, object]) -> np.ndarray:
    mu = np.asarray(mu_vec, dtype=float).reshape(-1)
    q_m_dim = int(model_pack["q_m_dim"])
    a_init = model_pack.get("q_m_init_from_mu_A", None)
    b_init = model_pack.get("q_m_init_from_mu_b", None)
    if a_init is not None and b_init is not None:
        a_init = np.asarray(a_init, dtype=float)
        b_init = np.asarray(b_init, dtype=float).reshape(-1)
        if a_init.shape[0] == q_m_dim and a_init.shape[1] == mu.size and b_init.size == q_m_dim:
            return (a_init @ mu + b_init).astype(float, copy=False)
    if mu.size == q_m_dim:
        return mu.copy()
    return np.zeros(q_m_dim, dtype=float)


def LoadPromRbfModel(
    stage2a_dir: str = "stage_2a_pod_data",
    stage3_dataset_file: str = "stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    stage4_rbf_dir: str = "stage_4_prom_rbf_grid",
    stage2b_dir: str = "stage_2b_ls_master",
):
    phi_rom = np.asarray(np.load(os.path.join(stage2a_dir, "pod_basis_free.npy")), dtype=float)
    free_dofs = np.asarray(np.load(os.path.join(stage2a_dir, "free_dofs.npy")), dtype=np.int64)
    dir_dofs = np.asarray(np.load(os.path.join(stage2a_dir, "dirichlet_dofs.npy")), dtype=np.int64)
    eq_map = np.asarray(np.load(os.path.join(stage2a_dir, "eq_map.npy")), dtype=np.int64)

    stage3 = np.load(stage3_dataset_file, allow_pickle=True)
    mu_mean = np.asarray(stage3["mu_mean"], dtype=float)
    mu_std = np.asarray(stage3["mu_std"], dtype=float)
    y_mean = np.asarray(stage3["y_mean"], dtype=float)
    y_std = np.asarray(stage3["y_std"], dtype=float)
    target_space = str(np.ravel(stage3["target_space"])[0]) if "target_space" in stage3 else "both"
    mu_space = str(np.ravel(stage3["mu_space"])[0]) if "mu_space" in stage3 else "gx_gxy"
    mapping = str(np.ravel(stage3["mapping"])[0]) if "mapping" in stage3 else "green_lagrange_upper"

    q_m_dim = int(np.asarray(stage3["q_m_all"]).shape[1])
    q_pod_dim = int(np.asarray(stage3["q_pod_all"]).shape[1])
    if "q_s_all" not in stage3:
        raise RuntimeError(
            "Stage3 dataset missing q_s_all. Rebuild pipeline with:\n"
            "  Stage2b (master/slave outputs),\n"
            "  Stage3 --target-space q_s,\n"
            "  Stage4 --rbf-input-space q_m."
        )
    q_s_dim = int(np.asarray(stage3["q_s_all"]).shape[1])

    a_m = np.asarray(stage3["A_m"], dtype=float) if "A_m" in stage3 else _load_stage2b_array(stage2b_dir, "A_m.npy")
    c_m = np.asarray(stage3["C_m"], dtype=float) if "C_m" in stage3 else _load_stage2b_array(stage2b_dir, "C_m.npy")
    c_s = np.asarray(stage3["C_s"], dtype=float) if "C_s" in stage3 else _load_stage2b_array(stage2b_dir, "C_s.npy")

    phi_m = _load_stage2b_array(stage2b_dir, "phi_m.npy")
    phi_s = _load_stage2b_array(stage2b_dir, "phi_s.npy")
    q_m_init_A_file = os.path.join(stage2b_dir, "q_m_init_from_mu_A.npy")
    q_m_init_b_file = os.path.join(stage2b_dir, "q_m_init_from_mu_b.npy")
    q_m_init_A = np.asarray(np.load(q_m_init_A_file), dtype=float) if os.path.exists(q_m_init_A_file) else None
    q_m_init_b = np.asarray(np.load(q_m_init_b_file), dtype=float) if os.path.exists(q_m_init_b_file) else None

    model_file = os.path.join(stage4_rbf_dir, "rbf_model.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"RBF model not found: {model_file}")
    with open(model_file, "rb") as f:
        rbf_model = pickle.load(f)

    meta_file = os.path.join(stage4_rbf_dir, "rbf_model_meta.npz")
    if os.path.exists(meta_file):
        meta = np.load(meta_file, allow_pickle=True)
        rbf_input_space = str(np.ravel(meta["input_space"])[0]).strip().lower()
        x_mean = np.asarray(meta["x_mean"], dtype=float)
        x_std = np.asarray(meta["x_std"], dtype=float)
        y_mean = np.asarray(meta["y_mean"], dtype=float)
        y_std = np.asarray(meta["y_std"], dtype=float)
    else:
        rbf_input_space = "mu"
        x_mean = mu_mean.copy()
        x_std = mu_std.copy()

    if phi_rom.shape[1] != q_pod_dim:
        raise RuntimeError(
            f"POD basis columns ({phi_rom.shape[1]}) do not match q_pod_dim from Stage3 ({q_pod_dim})."
        )

    if phi_m.shape[0] != phi_rom.shape[0] or phi_s.shape[0] != phi_rom.shape[0]:
        raise RuntimeError("phi_m/phi_s row size mismatch with Stage2a free-DOF basis.")
    if phi_m.shape[1] != q_m_dim:
        raise RuntimeError(
            f"phi_m columns ({phi_m.shape[1]}) must match q_m_dim ({q_m_dim})."
        )
    if phi_s.shape[1] != q_s_dim:
        raise RuntimeError(
            f"phi_s columns ({phi_s.shape[1]}) must match q_s_dim ({q_s_dim})."
        )
    if a_m.shape != (q_m_dim, q_m_dim):
        raise RuntimeError(f"A_m shape mismatch: expected ({q_m_dim},{q_m_dim}), got {a_m.shape}.")
    if c_m.shape != (q_pod_dim, q_m_dim):
        raise RuntimeError(
            f"C_m shape mismatch: expected ({q_pod_dim},{q_m_dim}), got {c_m.shape}."
        )
    if c_s.shape != (q_pod_dim, q_s_dim):
        raise RuntimeError(
            f"C_s shape mismatch: expected ({q_pod_dim},{q_s_dim}), got {c_s.shape}."
        )

    if rbf_input_space == "q_m":
        if x_mean.size != q_m_dim:
            raise RuntimeError(
                f"RBF input metadata mismatch for q_m space: dim(x_mean)={x_mean.size}, q_m_dim={q_m_dim}."
            )
    elif rbf_input_space == "mu":
        if x_mean.size != mu_mean.size:
            raise RuntimeError(
                f"RBF input metadata mismatch for mu space: dim(x_mean)={x_mean.size}, dim(mu)={mu_mean.size}."
            )
    else:
        raise RuntimeError(f"Unsupported RBF input_space='{rbf_input_space}'.")

    return {
        "phi_rom": phi_rom,
        "phi_m": phi_m,
        "phi_s": phi_s,
        "A_m": a_m,
        "C_m": c_m,
        "C_s": c_s,
        "free_dofs": free_dofs,
        "dir_dofs": dir_dofs,
        "eq_map": eq_map,
        "mu_mean": mu_mean,
        "mu_std": np.where(mu_std < 1e-14, 1.0, mu_std),
        "rbf_input_space": rbf_input_space,
        "x_mean": np.asarray(x_mean, dtype=float),
        "x_std": np.where(np.asarray(x_std, dtype=float) < 1e-14, 1.0, np.asarray(x_std, dtype=float)),
        "y_mean": y_mean,
        "y_std": np.where(y_std < 1e-14, 1.0, y_std),
        "target_space": target_space,
        "mu_space": mu_space,
        "mapping": mapping,
        "q_m_dim": q_m_dim,
        "q_s_dim": q_s_dim,
        "q_pod_dim": q_pod_dim,
        "q_m_init_from_mu_A": q_m_init_A,
        "q_m_init_from_mu_b": q_m_init_b,
        "rbf_model": rbf_model,
    }


def RunPromRbfBatchSimulation(
    parameters,
    model_pack: Dict[str, object],
    strain_path,
    out_dir=None,
    trajectory_index=None,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    prom_corrector_max_iters=6,
    prom_corrector_rel_tol=1.0e-5,
    prom_corrector_abs_tol=1.0e-10,
    prom_corrector_dq_abs_tol=1.0e-7,
    prom_corrector_dq_rel_tol=1.0e-6,
    prom_corrector_res_floor_for_dq=1.0e-1,
    prom_corrector_min_rel_drop_stop=1.0e-2,
    prom_corrector_stagnation_relnorm_gate=1.0e-4,
    prom_corrector_max_dq_norm=0.5,
    prom_use_old_stiffness_in_first_iteration=True,
    prom_old_stiffness_residual_cutoff=1.0e5,
    prom_corrector_l2_reg=1.0e-10,
    track_q_pod=0,
):
    if strain_path is None:
        raise ValueError("strain_path must be provided.")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    phi_m = np.asarray(model_pack["phi_m"], dtype=float)
    phi_s = np.asarray(model_pack["phi_s"], dtype=float)
    a_m = np.asarray(model_pack["A_m"], dtype=float)
    c_m = np.asarray(model_pack["C_m"], dtype=float)
    c_s = np.asarray(model_pack["C_s"], dtype=float)

    free_dofs = np.asarray(model_pack["free_dofs"], dtype=np.int64)
    target_space = str(model_pack["target_space"])
    mu_space = str(model_pack["mu_space"])
    mapping = str(model_pack["mapping"])
    q_m_dim = int(model_pack["q_m_dim"])
    q_s_dim = int(model_pack["q_s_dim"])
    q_pod_dim = int(model_pack["q_pod_dim"])
    track_qpod = bool(int(track_q_pod))
    rbf_input_space = str(model_pack.get("rbf_input_space", "mu")).strip().lower()

    mu_mean = np.asarray(model_pack["mu_mean"], dtype=float)
    mu_std = np.asarray(model_pack["mu_std"], dtype=float)
    x_mean = np.asarray(model_pack.get("x_mean", mu_mean), dtype=float)
    x_std = np.asarray(model_pack.get("x_std", mu_std), dtype=float)
    y_mean = np.asarray(model_pack["y_mean"], dtype=float)
    y_std = np.asarray(model_pack["y_std"], dtype=float)
    rbf_model = model_pack["rbf_model"]
    _assert_qm_input_dimension(q_m_dim, x_mean)
    if rbf_input_space != "q_m":
        raise RuntimeError(
            "PROM-RBF online Newton in q_m requires Stage4 RBF trained with '--rbf-input-space q_m'. "
            f"Current model input_space='{rbf_input_space}'. Retrain Stage4."
        )

    dt = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
    e_wp = np.asarray(strain_path, dtype=float)
    if e_wp.ndim != 2 or e_wp.shape[1] != 3:
        raise ValueError(f"strain_path must have shape [n,3], got {e_wp.shape}")

    n_seg = e_wp.shape[0] - 1
    seg_steps, _ = BuildDynamicSegmentSteps(
        e_wp,
        reference_steps=reference_steps,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=reference_amplitude,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    end_time = dt * float(n_steps_total)

    if n_steps_total <= 0:
        raise RuntimeError("Dynamic step allocation produced zero total steps.")

    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()

    mp = sim._GetSolver().GetComputingModelPart()
    entities = list(mp.Elements) + list(mp.Conditions)
    n_dof, eq_map_runtime, ta_disp = SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    eq_map_ref = np.asarray(model_pack["eq_map"], dtype=np.int64)
    dir_dofs_ref = np.asarray(model_pack["dir_dofs"], dtype=np.int64)
    if eq_map_runtime.shape == eq_map_ref.shape and not np.array_equal(eq_map_runtime, eq_map_ref):
        raise RuntimeError("eq_map mismatch between runtime model and Stage2a metadata.")

    free_mask = np.zeros(n_dof, dtype=bool)
    free_mask[free_dofs] = True
    dir_dofs_runtime = np.nonzero(~free_mask)[0].astype(np.int64)
    if dir_dofs_ref.size == dir_dofs_runtime.size and not np.array_equal(dir_dofs_ref, dir_dofs_runtime):
        raise RuntimeError("dirichlet_dofs mismatch between runtime model and Stage2a metadata.")

    assembler = VectorizedAssembler(mp, n_dof, eq_map_runtime)

    sim._InitializeDomainCenterIfNeeded(mp)
    x0c, y0c = float(sim._x0c), float(sim._y0c)

    dof_x = np.zeros(n_dof, dtype=float)
    dof_y = np.zeros(n_dof, dtype=float)
    is_x_dof = np.zeros(n_dof, dtype=bool)
    for i, node in enumerate(mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        ix = int(eq_map_runtime[i, 0])
        iy = int(eq_map_runtime[i, 1])
        if 0 <= ix < n_dof:
            dof_x[ix] = xr
            dof_y[ix] = yr
            is_x_dof[ix] = True
        if 0 <= iy < n_dof:
            dof_x[iy] = xr
            dof_y[iy] = yr
            is_x_dof[iy] = False

    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]

    dir_dofs = dir_dofs_runtime
    x_dir = dof_x[dir_dofs]
    y_dir = dof_y[dir_dofs]
    is_x_dir = is_x_dof[dir_dofs]

    def _affine_component(e_vec, xx, yy, is_x):
        f = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (f[0, 0] - 1.0) * xx + f[0, 1] * yy
        uy = f[1, 0] * xx + (f[1, 1] - 1.0) * yy
        return np.where(is_x, ux, uy)

    results_eps = [np.zeros(3, dtype=float)]
    results_sig = [np.zeros(3, dtype=float)]
    u_hist = [np.zeros(n_dof, dtype=float)]
    e_applied_hist = [np.zeros(3, dtype=float)]
    qm_hist = [np.zeros(q_m_dim, dtype=float)]
    qs_hist = [np.zeros(q_s_dim, dtype=float)]
    qpod_hist = [np.zeros(q_pod_dim, dtype=float)] if track_qpod else None

    n_corr = max(0, int(prom_corrector_max_iters))
    print(
        f"  [PROM-RBF] Solving trajectory with {n_steps_total} increments "
        f"(reduced corrector iters={n_corr})."
    )

    q_m_state = None
    k_red_old = None
    for step in range(1, n_steps_total + 1):
        time_val = float(step) * float(dt)
        mp.CloneTimeStep(time_val)
        mp.ProcessInfo[KM.DELTA_TIME] = dt
        mp.ProcessInfo[KM.TIME] = time_val
        mp.ProcessInfo[KM.STEP] = step

        sim.time, sim.step, sim.end_time = time_val, step, end_time
        sim.InitializeSolutionStep()

        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        e = (1.0 - xi) * e_wp[s, :] + xi * e_wp[s + 1, :]

        mu = _recover_mu_from_E(e, mapping=mapping, mu_space=mu_space)
        if q_m_state is None:
            q_m = _initial_qm_from_mu(mu, model_pack=model_pack)
        else:
            q_m = np.asarray(q_m_state, dtype=float).reshape(-1).copy()

        q_s = np.zeros(q_s_dim, dtype=float)
        q_pod = np.zeros(q_pod_dim, dtype=float) if track_qpod else None

        def _decode_qs_local(qm_vec: np.ndarray) -> np.ndarray:
            return _decode_qs_from_qm(
                qm_vec,
                rbf_model=rbf_model,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                target_space=target_space,
                q_m_dim=q_m_dim,
                q_s_dim=q_s_dim,
            )
        _decode_qs_local._analytic_kwargs = {
            "rbf_model": rbf_model,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "target_space": target_space,
            "q_m_dim": q_m_dim,
            "q_s_dim": q_s_dim,
        }

        u_aff_free = _affine_component(e, x_free, y_free, is_x_free)
        u_aff_dir = _affine_component(e, x_dir, y_dir, is_x_dir)
        u = np.zeros(n_dof, dtype=float)
        u[dir_dofs] = u_aff_dir

        converged = False
        nonfinite_detected = False
        nrm0 = None
        nrf_last = np.nan
        nrm_last = np.nan
        ndq_last = np.nan
        ndq0 = None
        nrm_prev = None
        nrm_best = np.inf
        rrel_best = np.inf
        q_m_step_start = q_m.copy()
        q_m_best = q_m.copy()
        k_red_last = None

        for it in range(n_corr + 1):
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1

            q_s = _decode_qs_local(q_m)
            q_master = a_m @ q_m
            if track_qpod:
                q_pod = (c_m @ q_master) + (c_s @ q_s)
            u[free_dofs] = u_aff_free + (phi_m @ q_master) + (phi_s @ q_s)

            SetDisplacementFromEquationVector(u, eq_map_runtime, ta_disp)
            UpdateCurrentCoordinatesFromDisplacement(mp, step=0)

            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            k_full, rhs = assembler.Assemble(u)
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            r_f = rhs[free_dofs]
            dqs_dqm = _decoder_jacobian_qs_wrt_qm(
                q_m,
                _decode_qs_local,
                q_s_dim=q_s_dim,
            )
            du_dqm = (phi_m @ a_m) + (phi_s @ dqs_dqm)
            r_red = du_dqm.T @ r_f

            nrf = float(np.linalg.norm(r_f))
            nrm = float(np.linalg.norm(r_red))
            if (not np.isfinite(nrf)) or (not np.isfinite(nrm)):
                nonfinite_detected = True
                break
            nrf_last = nrf
            nrm_last = nrm

            if nrm0 is None:
                nrm0 = max(nrm, 1.0e-30)
            r_rel = nrm / nrm0
            if nrm < nrm_best:
                nrm_best = nrm
                rrel_best = r_rel
                q_m_best = q_m.copy()

            kff = k_full[free_dofs][:, free_dofs]
            k_red = du_dqm.T @ (kff @ du_dqm)
            if float(prom_corrector_l2_reg) > 0.0:
                k_red = k_red + float(prom_corrector_l2_reg) * np.eye(q_m_dim, dtype=float)
            k_red_last = k_red

            if nrm <= float(prom_corrector_abs_tol):
                converged = True
                print(f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e}")
                print(f"  > Converged in {it} iterations.")
                break

            if (
                ndq_last == ndq_last
                and ndq_last <= float(prom_corrector_dq_abs_tol)
                and r_rel <= float(prom_corrector_rel_tol)
                and nrm <= float(prom_corrector_res_floor_for_dq)
            ):
                converged = True
                print(
                    f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e}, "
                    f"||dq_m|| = {ndq_last:.3e} -> converged(small-dq+rel)"
                )
                print(f"  > Converged in {it} iterations.")
                break

            if nrm_prev is not None:
                rel_drop = abs(nrm_prev - nrm) / max(nrm_prev, 1.0e-30)
                if (
                    rel_drop <= float(prom_corrector_min_rel_drop_stop)
                    and r_rel <= float(prom_corrector_stagnation_relnorm_gate)
                    and nrm <= float(prom_corrector_res_floor_for_dq)
                ):
                    converged = True
                    print(
                        f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e}, "
                        f"rel_drop={rel_drop:.3e} -> converged(stagnation)"
                    )
                    print(f"  > Converged in {it} iterations.")
                    break
            nrm_prev = nrm

            if r_rel <= float(prom_corrector_rel_tol) and nrm <= float(prom_corrector_res_floor_for_dq):
                converged = True
                print(
                    f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e} -> converged(rel)"
                )
                print(f"  > Converged in {it} iterations.")
                break

            if it >= n_corr:
                break

            k_solve = k_red
            used_old = False
            if (
                it == 0
                and bool(prom_use_old_stiffness_in_first_iteration)
                and k_red_old is not None
                and k_red_old.shape == k_red.shape
                and nrm <= float(prom_old_stiffness_residual_cutoff)
            ):
                k_solve = k_red_old
                used_old = True

            try:
                dq = np.linalg.solve(k_solve, r_red)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(k_solve, r_red, rcond=None)[0]
            if used_old and (not np.all(np.isfinite(dq))):
                try:
                    dq = np.linalg.solve(k_red, r_red)
                    used_old = False
                except np.linalg.LinAlgError:
                    dq = np.linalg.lstsq(k_red, r_red, rcond=None)[0]
                    used_old = False

            ndq = float(np.linalg.norm(dq))
            ndq_last = ndq
            if ndq0 is None:
                ndq0 = max(ndq, 1.0e-30)
            dq_rel = ndq / ndq0
            if ndq > float(prom_corrector_max_dq_norm) and ndq > 0.0:
                scale = float(prom_corrector_max_dq_norm) / ndq
                dq = dq * scale
                ndq = float(np.linalg.norm(dq))
                ndq_last = ndq
                dq_rel = ndq / ndq0
            q_m = q_m + dq
            if used_old:
                print("    > using previous reduced stiffness (K_old) at first iteration")

            dq_ok = (ndq <= float(prom_corrector_dq_abs_tol)) or (dq_rel <= float(prom_corrector_dq_rel_tol))
            if dq_ok and nrm <= float(prom_corrector_res_floor_for_dq):
                converged = True
                print(
                    f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e}, "
                    f"||dq_m|| = {ndq:.3e} (rel {dq_rel:.3e}) -> converged(dq)"
                )
                print(f"  > Converged in {it} iterations.")
                break

            print(
                f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e}"
            )

        if not converged:
            quasi_converged = (
                np.isfinite(nrm_best)
                and np.isfinite(rrel_best)
                and (rrel_best <= float(prom_corrector_rel_tol))
                and (nrm_best <= float(prom_corrector_res_floor_for_dq))
            )
            if quasi_converged:
                q_m = q_m_best.copy()
                converged = True
                print(
                    "  [PROM-RBF] Step accepted as quasi-converged: "
                    f"best ||R_r||={nrm_best:.3e}, rel={rrel_best:.3e}"
                )
            else:
                if np.isfinite(nrm_best):
                    q_m = q_m_best.copy()
                else:
                    q_m = q_m_step_start.copy()
                print(
                    f"  [PROM-RBF][WARN] Step {step:03d}/{n_steps_total} not converged in {n_corr} corrector iters. "
                    f"last ||R_r||={nrm_last:.3e}, ||dq_m||={ndq_last:.3e}"
                )
                if nonfinite_detected:
                    print("  [PROM-RBF][WARN] non-finite state detected; rolled back to best finite iterate.")
            k_red_old = None
        elif k_red_last is not None and np.all(np.isfinite(k_red_last)):
            k_red_old = k_red_last.copy()
        else:
            k_red_old = None

        q_s = _decode_qs_local(q_m)
        q_master = a_m @ q_m
        if track_qpod:
            q_pod = (c_m @ q_master) + (c_s @ q_s)
        u[free_dofs] = u_aff_free + (phi_m @ q_master) + (phi_s @ q_s)

        eps_h, sig_h = CalculateHomogenizedFromAssemblerWithElementWeights(
            assembler,
            w_eps=None,
            w_sig=None,
            reference_measure=float(np.sum(np.asarray(assembler.area_e, dtype=float))),
        )

        sim.FinalizeSolutionStep()

        results_eps.append(np.asarray(eps_h, dtype=float))
        results_sig.append(np.asarray(sig_h, dtype=float))
        u_hist.append(u.copy())
        e_applied_hist.append(e.copy())
        qm_hist.append(q_m.copy())
        qs_hist.append(q_s.copy())
        if track_qpod:
            qpod_hist.append(q_pod.copy())
        q_m_state = q_m.copy()
        print(f"  [PROM-RBF] Step {step:03d}/{n_steps_total}: ||q_m||={np.linalg.norm(q_m):.3e}, ||q_s||={np.linalg.norm(q_s):.3e}")

    sim.Finalize()

    strain_hist = np.stack(results_eps)
    stress_hist = np.stack(results_sig)
    u_hist = np.stack(u_hist)
    e_applied_hist = np.stack(e_applied_hist)
    qm_hist = np.stack(qm_hist)
    qs_hist = np.stack(qs_hist)
    if track_qpod:
        qpod_hist = np.stack(qpod_hist)

    if out_dir is not None:
        tag = f"trajectory_{trajectory_index}" if trajectory_index is not None else "prom_rbf_run"
        np.save(os.path.join(out_dir, f"{tag}_strain.npy"), strain_hist)
        np.save(os.path.join(out_dir, f"{tag}_stress.npy"), stress_hist)
        np.save(os.path.join(out_dir, f"{tag}_U.npy"), u_hist)
        np.save(os.path.join(out_dir, f"{tag}_applied_strain.npy"), e_applied_hist)
        np.save(os.path.join(out_dir, f"{tag}_q_m.npy"), qm_hist)
        np.save(os.path.join(out_dir, f"{tag}_q_s.npy"), qs_hist)
        if track_qpod:
            np.save(os.path.join(out_dir, f"{tag}_q_pod.npy"), qpod_hist)

    return strain_hist, stress_hist
