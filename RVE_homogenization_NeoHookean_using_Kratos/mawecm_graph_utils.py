#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for MAW-ECM graph construction and smoothness diagnostics.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _as_2d(arr):
    x = np.asarray(arr, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}.")
    return x


def _pairwise_sqeuclidean(x):
    xx = np.sum(x * x, axis=1, keepdims=True)
    d2 = xx + xx.T - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def build_knn_graph_laplacian(q_train, knn=8, metric="euclidean", kernel="gaussian", sigma=None):
    """
    Build a symmetric kNN graph Laplacian L = D - W.

    Parameters
    ----------
    q_train : array, shape (n_samples, q_dim)
    knn : int
        Number of neighbors (excluding self).
    metric : str
        Only 'euclidean' is currently supported.
    kernel : str
        'gaussian' or 'binary'.
    sigma : float or None
        Gaussian bandwidth. If None, uses median kNN distance.
    """
    q = _as_2d(q_train)
    n = int(q.shape[0])
    if n < 2:
        return sparse.csr_matrix((n, n), dtype=float)

    if metric.strip().lower() != "euclidean":
        raise ValueError("Only euclidean metric is supported.")

    k = int(max(1, min(int(knn), n - 1)))
    d2 = _pairwise_sqeuclidean(q)
    np.fill_diagonal(d2, np.inf)

    nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = nn_idx.reshape(-1)
    d2_vals = d2[rows, cols]

    ker = str(kernel).strip().lower()
    if ker == "binary":
        vals = np.ones_like(d2_vals, dtype=float)
    elif ker == "gaussian":
        if sigma is None:
            d = np.sqrt(np.maximum(d2_vals, 0.0))
            sigma_eff = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
            sigma_eff = max(sigma_eff, 1.0e-12)
        else:
            sigma_eff = max(float(sigma), 1.0e-12)
        vals = np.exp(-0.5 * d2_vals / (sigma_eff * sigma_eff))
    else:
        raise ValueError("kernel must be 'gaussian' or 'binary'.")

    w_dir = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # Symmetrize by max to preserve local strong links.
    w = w_dir.maximum(w_dir.T).tocsr()
    w.setdiag(0.0)
    w.eliminate_zeros()

    deg = np.asarray(w.sum(axis=1)).reshape(-1)
    dmat = sparse.diags(deg, offsets=0, shape=(n, n), dtype=float)
    lap = (dmat - w).tocsr()
    return lap


def build_cell_graph_laplacian(n_nodes, cells, cell_type="auto", weight_mode="binary"):
    """
    Build graph Laplacian L = D - W from mesh connectivity.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes.
    cells : array, shape (n_cells, n_conn)
        Cell connectivity (0-based node ids).
    cell_type : str
        "auto", "hex8", or "quad4".
    weight_mode : str
        Currently only "binary" is supported.
    """
    n = int(n_nodes)
    if n <= 0:
        raise ValueError("n_nodes must be > 0.")
    conn = np.asarray(cells, dtype=np.int64)
    if conn.ndim != 2 or conn.shape[0] <= 0 or conn.shape[1] <= 0:
        raise ValueError(f"Invalid cells shape: {conn.shape}.")
    if np.min(conn) < 0 or np.max(conn) >= n:
        raise ValueError(
            f"Connectivity indices out of range: min={int(np.min(conn))}, max={int(np.max(conn))}, n_nodes={n}."
        )

    ctype = str(cell_type).strip().lower()
    if ctype == "auto":
        if conn.shape[1] == 8:
            ctype = "hex8"
        elif conn.shape[1] == 4:
            ctype = "quad4"
        else:
            raise ValueError(
                f"Cannot auto-detect cell type from n_conn={conn.shape[1]}. Use cell_type explicitly."
            )
    if ctype == "hex8":
        local_edges = np.asarray(
            [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ],
            dtype=np.int64,
        )
    elif ctype == "quad4":
        local_edges = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported cell_type='{cell_type}'.")

    if str(weight_mode).strip().lower() != "binary":
        raise ValueError("Only weight_mode='binary' is supported.")

    i_all = []
    j_all = []
    for e in range(conn.shape[0]):
        c = conn[e, :]
        i_all.append(c[local_edges[:, 0]])
        j_all.append(c[local_edges[:, 1]])

    i = np.concatenate(i_all).astype(np.int64, copy=False)
    j = np.concatenate(j_all).astype(np.int64, copy=False)
    v = np.ones(i.shape[0], dtype=float)
    w_dir = sparse.coo_matrix((v, (i, j)), shape=(n, n)).tocsr()
    w = w_dir.maximum(w_dir.T).tocsr()
    w.setdiag(0.0)
    w.data[:] = 1.0
    w.eliminate_zeros()

    deg = np.asarray(w.sum(axis=1)).reshape(-1)
    dmat = sparse.diags(deg, offsets=0, shape=(n, n), dtype=float)
    lap = (dmat - w).tocsr()
    return lap


def edge_list_from_laplacian(k_graph):
    """
    Extract undirected edge list (i,j) from a Laplacian matrix.
    """
    k = sparse.csr_matrix(k_graph)
    upper = sparse.triu(-k, k=1).tocoo()
    mask = np.asarray(upper.data > 0.0).reshape(-1)
    i_edge = np.asarray(upper.row[mask], dtype=np.int64)
    j_edge = np.asarray(upper.col[mask], dtype=np.int64)
    return i_edge, j_edge


def edge_jump_metrics(delta_w, k_graph):
    """
    Compute roughness metrics from edge jumps.

    Parameters
    ----------
    delta_w : array, shape (n_rows, n_nodes)
    k_graph : Laplacian, shape (n_nodes, n_nodes)

    Returns
    -------
    dict with keys: S, R95, Rmax, n_edges
    """
    dw = _as_2d(delta_w)
    k = np.asarray(k_graph.todense() if sparse.issparse(k_graph) else k_graph, dtype=float)
    if dw.shape[1] != k.shape[0] or k.shape[0] != k.shape[1]:
        raise ValueError("delta_w columns must match k_graph size.")

    i_edge, j_edge = edge_list_from_laplacian(k_graph)
    if i_edge.size == 0:
        s = float(np.sum(dw * (dw @ k)))
        return {"S": s, "R95": 0.0, "Rmax": 0.0, "n_edges": 0}

    jumps = dw[:, i_edge] - dw[:, j_edge]
    abs_j = np.abs(jumps.reshape(-1))
    rmax = float(np.max(abs_j)) if abs_j.size else 0.0
    r95 = float(np.percentile(abs_j, 95)) if abs_j.size else 0.0
    s = float(np.sum(dw * (dw @ k)))
    return {"S": s, "R95": r95, "Rmax": rmax, "n_edges": int(i_edge.size)}
