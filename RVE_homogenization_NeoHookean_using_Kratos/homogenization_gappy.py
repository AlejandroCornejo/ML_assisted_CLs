#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gappy-POD utilities for homogenization output reconstruction.

Goal:
  keep residual assembly hyper-reduced with ECM (Z_res, w_res) and build an
  alternative homogenization evaluator that only uses data sampled on Z_res.

Offline idea:
  c in R^(6*n_elem)   -> per-snapshot concatenated element contributions
  y = H c in R^6      -> homogenized (eps_xx, eps_yy, eps_xy, sig_xx, sig_yy, sig_xy)
  c_s ~ U_r a_s       -> POD basis over full c snapshots
  a_s ~ argmin ||P^T(U_r a - c)||_2  (gappy least squares on sampled entries)

Then y is reconstructed with a precomputed affine map:
  y_hat = M * c_sample + b
"""

from __future__ import annotations

import numpy as np

N_HOM_COMPONENTS = 6


def build_component_element_sample_indices(sample_elements, n_elem, n_components=N_HOM_COMPONENTS):
    """
    Build flat indices for vectors shaped as (n_components, n_elem) flattened
    in row-major order, using all components for each sampled element.
    """
    elems = np.asarray(sample_elements, dtype=np.int64).reshape(-1)
    n_elem = int(n_elem)
    n_components = int(n_components)
    if n_elem <= 0:
        raise ValueError("n_elem must be > 0.")
    if n_components <= 0:
        raise ValueError("n_components must be > 0.")
    if elems.size == 0:
        raise ValueError("sample_elements is empty.")
    if np.min(elems) < 0 or np.max(elems) >= n_elem:
        raise ValueError(
            f"sample_elements out of range [0, {n_elem - 1}]. "
            f"min={int(np.min(elems))}, max={int(np.max(elems))}"
        )

    # Keep stable, unique ordering.
    elems = np.unique(elems)
    # Flat index for c[comp, elem] with c.reshape(-1, order='C'):
    # idx = comp * n_elem + elem
    sample_idx = np.empty(n_components * elems.size, dtype=np.int64)
    k = 0
    for comp in range(n_components):
        base = comp * n_elem
        sample_idx[k : k + elems.size] = base + elems
        k += elems.size
    return elems, sample_idx


def _snapshot_matrix_from_chom(C_hom, n_snapshots, n_elem, n_components=N_HOM_COMPONENTS):
    """
    Convert Stage5/Stage9 C_hom layout:
      C_hom[(n_components*Ns), n_elem]
    into snapshot matrix:
      X[(n_components*n_elem), Ns]
    """
    n_snapshots = int(n_snapshots)
    n_elem = int(n_elem)
    n_components = int(n_components)
    X = np.empty((n_components * n_elem, n_snapshots), dtype=float)
    for s in range(n_snapshots):
        b0 = n_components * s
        b1 = n_components * (s + 1)
        X[:, s] = np.asarray(C_hom[b0:b1, :], dtype=float).reshape(-1, order="C")
    return X


def build_gappy_pod_homogenization_operator_from_chom(
    C_hom,
    n_snapshots,
    n_elem,
    sample_elements,
    energy_loss_tol=1.0e-10,
    max_modes=256,
    ridge=1.0e-12,
    center_data=True,
    reference_measure=None,
    normalize_by_reference_measure=True,
):
    """
    Build affine map y_hat = M @ c_sample + b from C_hom snapshots.

    Returns a dict ready to be stored in npz (values are numpy arrays/scalars).
    """
    n_snapshots = int(n_snapshots)
    n_elem = int(n_elem)
    if n_snapshots <= 0:
        raise ValueError("n_snapshots must be > 0.")
    if n_elem <= 0:
        raise ValueError("n_elem must be > 0.")

    z_res, sample_indices = build_component_element_sample_indices(
        sample_elements=sample_elements,
        n_elem=n_elem,
        n_components=N_HOM_COMPONENTS,
    )
    X = _snapshot_matrix_from_chom(C_hom, n_snapshots, n_elem, n_components=N_HOM_COMPONENTS)
    Y_integral = np.sum(X.reshape(N_HOM_COMPONENTS, n_elem, n_snapshots), axis=1)
    ref_measure = None
    if normalize_by_reference_measure:
        if reference_measure is None:
            raise ValueError(
                "reference_measure is required when normalize_by_reference_measure=True."
            )
        ref_measure = float(reference_measure)
        if abs(ref_measure) <= 1.0e-30:
            raise ValueError("reference_measure must be non-zero.")
        Y_true = Y_integral / ref_measure
    else:
        Y_true = Y_integral

    mu = np.mean(X, axis=1) if center_data else np.zeros(X.shape[0], dtype=float)
    Xc = X - mu[:, None]

    U, svals, _ = np.linalg.svd(Xc, full_matrices=False)
    rank = int(np.count_nonzero(svals > 0.0))
    if rank <= 0:
        raise RuntimeError("SVD rank is zero; cannot build gappy operator.")

    max_modes = int(max_modes) if max_modes is not None else rank
    if max_modes <= 0:
        max_modes = rank
    max_modes = min(max_modes, rank)

    energy = svals[:rank] ** 2
    energy_total = float(np.sum(energy))
    if energy_total <= 0.0:
        r_energy = 1
        energy_captured = 1.0
    else:
        cumulative = np.cumsum(energy) / energy_total
        target = 1.0 - float(max(0.0, energy_loss_tol))
        r_energy = int(np.searchsorted(cumulative, target, side="left") + 1)
        r_energy = max(1, min(rank, r_energy))
        energy_captured = float(cumulative[r_energy - 1])

    r = min(max_modes, r_energy)
    m = int(sample_indices.size)
    if r > m:
        r = m
    if r <= 0:
        raise RuntimeError("Number of retained modes is zero.")

    U_r = U[:, :r]  # (6*n_elem, r)
    U_s = U_r[sample_indices, :]  # (m, r)
    mu_s = mu[sample_indices]

    # Gappy least-squares operator (with tiny ridge regularization).
    G = U_s.T @ U_s
    reg = float(ridge)
    if reg > 0.0:
        G = G + reg * np.eye(G.shape[0], dtype=float)
    A = np.linalg.solve(G, U_s.T)  # (r, m)

    # H map: sums per component over all elements.
    H = np.zeros((N_HOM_COMPONENTS, N_HOM_COMPONENTS * n_elem), dtype=float)
    for comp in range(N_HOM_COMPONENTS):
        i0 = comp * n_elem
        H[comp, i0 : i0 + n_elem] = 1.0

    M = H @ U_r @ A
    b = H @ mu - M @ mu_s
    if normalize_by_reference_measure:
        M = M / ref_measure
        b = b / ref_measure

    Y_pred = M @ X[sample_indices, :] + b[:, None]
    err_total = float(
        np.linalg.norm(Y_pred - Y_true) / (np.linalg.norm(Y_true) + 1.0e-30)
    )
    err_eps = float(
        np.linalg.norm(Y_pred[0:3, :] - Y_true[0:3, :])
        / (np.linalg.norm(Y_true[0:3, :]) + 1.0e-30)
    )
    err_sig = float(
        np.linalg.norm(Y_pred[3:6, :] - Y_true[3:6, :])
        / (np.linalg.norm(Y_true[3:6, :]) + 1.0e-30)
    )
    step_err = np.linalg.norm((Y_pred - Y_true).T, axis=1) / (
        np.linalg.norm(Y_true.T, axis=1) + 1.0e-30
    )

    return {
        "hom_gappy_enabled": np.array([1], dtype=np.int8),
        "hom_gappy_method": np.array(["gappy_pod_residual_sampling"]),
        "hom_gappy_n_elem": np.array([n_elem], dtype=np.int64),
        "hom_gappy_n_components": np.array([N_HOM_COMPONENTS], dtype=np.int64),
        "hom_gappy_center_data": np.array([1 if center_data else 0], dtype=np.int8),
        "hom_gappy_n_modes": np.array([r], dtype=np.int64),
        "hom_gappy_energy_captured": np.array([energy_captured], dtype=float),
        "hom_gappy_energy_loss_tol": np.array([float(energy_loss_tol)], dtype=float),
        "hom_gappy_ridge": np.array([float(ridge)], dtype=float),
        "hom_gappy_output_is_average": np.array(
            [1 if normalize_by_reference_measure else 0], dtype=np.int8
        ),
        "hom_gappy_reference_measure": np.array(
            [float(ref_measure) if ref_measure is not None else np.nan], dtype=float
        ),
        "hom_gappy_sample_elements": z_res.astype(np.int64),
        "hom_gappy_sample_indices": sample_indices.astype(np.int64),
        "hom_gappy_matrix": np.asarray(M, dtype=float),
        "hom_gappy_offset": np.asarray(b, dtype=float),
        "hom_gappy_train_rel_error_total": np.array([err_total], dtype=float),
        "hom_gappy_train_rel_error_eps": np.array([err_eps], dtype=float),
        "hom_gappy_train_rel_error_sig": np.array([err_sig], dtype=float),
        "hom_gappy_train_rel_error_step_mean": np.array([float(np.mean(step_err))], dtype=float),
        "hom_gappy_train_rel_error_step_max": np.array([float(np.max(step_err))], dtype=float),
    }


def extract_sampled_hom_vector_from_assembler(assembler, sample_element_indices):
    """
    Build sampled c vector (all 6 components per selected element) from the
    latest VectorizedAssembler state.
    """
    z = np.asarray(sample_element_indices, dtype=np.int64).reshape(-1)
    if z.size == 0:
        raise ValueError("sample_element_indices is empty.")
    if getattr(assembler, "n_elems", 0) <= 0:
        raise RuntimeError("Assembler has no active elements.")
    if np.min(z) < 0 or np.max(z) >= int(assembler.n_elems):
        raise RuntimeError(
            f"Sample element index out of range for active mesh: "
            f"min={int(np.min(z))}, max={int(np.max(z))}, n_elems={int(assembler.n_elems)}"
        )

    if hasattr(assembler, "area_e"):
        area_e = np.asarray(assembler.area_e, dtype=float).reshape(-1)
    else:
        area_e = np.sum(np.asarray(assembler.w_detJ, dtype=float), axis=1)

    eps_e = np.mean(np.asarray(assembler._E_voigt, dtype=float), axis=1)  # (n_elem,3)
    sig_e = np.mean(np.asarray(assembler._S_voigt, dtype=float), axis=1)  # (n_elem,3)
    c_e = np.zeros((N_HOM_COMPONENTS, int(assembler.n_elems)), dtype=float)
    c_e[0:3, :] = (area_e[:, None] * eps_e).T
    c_e[3:6, :] = (area_e[:, None] * sig_e).T

    return c_e[:, z].reshape(-1, order="C")


def evaluate_gappy_homogenization_from_sample(sample_vec, matrix_M, offset_b):
    """Evaluate y_hat = M @ sample_vec + b."""
    s = np.asarray(sample_vec, dtype=float).reshape(-1)
    M = np.asarray(matrix_M, dtype=float)
    b = np.asarray(offset_b, dtype=float).reshape(-1)
    y = M @ s + b
    return np.asarray(y, dtype=float).reshape(-1)
