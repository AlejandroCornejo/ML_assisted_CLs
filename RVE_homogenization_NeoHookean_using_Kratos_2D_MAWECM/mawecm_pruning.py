#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAW-ECM pruning routines:
- stage 1: no-enforcement least-change pruning
- stage 2: explicit positivity enforcement (local active-set, no graph)
- stage 3: graph-regularized active-set pruning (Option B)
"""

from __future__ import annotations

import time
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from mawecm_graph_utils import edge_jump_metrics


DEFAULT_OPTIONS = {
    "alpha_smooth": 0.1,
    "criterion": 2,  # 0: zero-clamp nodes, 1: smooth score S, 2: R95 tie Rmax
    "number_of_candidates_to_try": None,
    "incremental_smoothing": True,
    "use_total_as_criterion": False,
    "tol_rank_rel": 1.0e-12,
    "tol_neg_factor": 10.0,
    "tol_zero": 1.0e-12,
    "max_active_set_iters": 30,
    "max_reduced_dim": 2500,
    "weight_pos_tol": 0.0,
    "enforce_nonnegativity": True,
    # Joaquín-style control flags
    "smooth_laplacian_all_iterations": False,
    "use_global_graph_2ndstage": False,
    "max_number_zeros_active_set_loop": 1,
    "verbose": True,
    "n_stop": None,
}


def _merge_options(options):
    out = dict(DEFAULT_OPTIONS)
    if options:
        out.update(options)
    if out["number_of_candidates_to_try"] is not None:
        out["number_of_candidates_to_try"] = int(out["number_of_candidates_to_try"])
    out["criterion"] = int(out["criterion"])
    out["max_active_set_iters"] = int(out["max_active_set_iters"])
    out["max_reduced_dim"] = int(out["max_reduced_dim"])
    out["smooth_laplacian_all_iterations"] = bool(out["smooth_laplacian_all_iterations"])
    out["use_global_graph_2ndstage"] = bool(out["use_global_graph_2ndstage"])
    out["max_number_zeros_active_set_loop"] = int(out["max_number_zeros_active_set_loop"])
    out["verbose"] = bool(out["verbose"])
    out["enforce_nonnegativity"] = bool(out["enforce_nonnegativity"])
    return out


def _as_constraints(A_blocks, b_blocks, n_cand):
    if len(A_blocks) != len(b_blocks):
        raise ValueError("A_blocks and b_blocks length mismatch.")
    A_out = []
    b_out = []
    m_list = []
    for j, (A, b) in enumerate(zip(A_blocks, b_blocks)):
        Aj = np.asarray(A, dtype=float)
        bj = np.asarray(b, dtype=float).reshape(-1)
        if Aj.ndim != 2:
            raise ValueError(f"A[{j}] must be 2D, got {Aj.shape}.")
        if Aj.shape[1] != int(n_cand):
            raise ValueError(f"A[{j}] columns={Aj.shape[1]} expected {n_cand}.")
        if Aj.shape[0] != bj.size:
            raise ValueError(f"A[{j}] rows={Aj.shape[0]} != len(b[{j}])={bj.size}.")
        A_out.append(Aj)
        b_out.append(bj)
        m_list.append(int(bj.size))
    return A_out, b_out, m_list


def _negativity_tolerance(w_ini, tol_factor):
    w = np.asarray(w_ini, dtype=float).reshape(-1)
    vol = float(np.sum(w))
    r = max(int(w.size), 1)
    return float(tol_factor) * (np.finfo(float).eps ** (2.0 / 3.0)) * (vol / r)


def _nullspace_basis(A, tol_rel=1.0e-12):
    """
    Nullspace basis of A (m x n) using SVD. Returns n x (n-rank).
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    if s.size == 0:
        return np.eye(A.shape[1], dtype=float)
    tol = float(tol_rel) * float(np.max(s))
    rank = int(np.sum(s > tol))
    return Vh[rank:, :].T


def _try_no_enforcement_pass(
    W,
    A_blocks,
    b_blocks,
    i_cand,
    iloc_pos,
    ind_sort,
    tol_neg,
    tol_rank_rel,
    enforce_nonnegativity,
):
    """
    Try candidates (in ind_sort order) using local least-change update + rejection on negatives.
    Returns (ok, W_new, i_cand_new, removed_index, tested_count).
    """
    tested = 0
    for k_loc in ind_sort:
        tested += 1
        cand_global = int(iloc_pos[int(k_loc)])
        i_cand_new = np.array([i for i in i_cand if int(i) != cand_global], dtype=np.int64)
        if i_cand_new.size == 0:
            continue

        W_new = np.asarray(W, dtype=float).copy()
        feasible = True

        for j, (Aj, bj) in enumerate(zip(A_blocks, b_blocks)):
            A_keep = Aj[:, i_cand_new]  # (m, r_keep)
            if A_keep.shape[1] < A_keep.shape[0]:
                feasible = False
                break
            rank = np.linalg.matrix_rank(A_keep, tol=float(tol_rank_rel) * max(np.linalg.norm(A_keep, 2), 1.0))
            if rank < A_keep.shape[0]:
                feasible = False
                break

            w_before = W_new[i_cand_new, j]
            rhs = bj - (A_keep @ w_before)
            delta, *_ = np.linalg.lstsq(A_keep, rhs, rcond=None)
            w_after = w_before + delta
            if bool(enforce_nonnegativity) and np.any(w_after < -tol_neg):
                feasible = False
                break
            W_new[i_cand_new, j] = w_after

        if feasible:
            W_new[cand_global, :] = 0.0
            return True, W_new, i_cand_new, cand_global, tested

    return False, W, i_cand, None, tested


def _prune_step_optionb_np(w_old, A_loc_blocks, b_blocks, K_graph, alpha, p_local, opts):
    """
    Python counterpart of prune_step_optionB.m for one candidate.

    w_old      : (r, Nm)
    A_loc      : list length Nm, each (m_j, r)
    b_blocks   : list length Nm, each (m_j,)
    p_local    : local prune index in [0, r-1]
    """
    tol_rank_rel = float(opts["tol_rank_rel"])
    tol_neg = float(opts["tol_neg"])
    max_as = int(opts["max_active_set_iters"])
    incremental = bool(opts["incremental_smoothing"])
    max_reduced_dim = int(opts["max_reduced_dim"])

    w_old = np.asarray(w_old, dtype=float)
    r, n_nodes = w_old.shape
    if p_local < 0 or p_local >= r:
        return w_old, {"ok": False, "reason": "invalid prune index", "nASiter": 0, "Dj": None}

    K = sparse.csr_matrix(K_graph)
    if K.shape != (n_nodes, n_nodes):
        return w_old, {"ok": False, "reason": "Kgraph shape mismatch", "nASiter": 0, "Dj": None}

    N = r * n_nodes
    H = sparse.eye(N, format="csc") + float(alpha) * sparse.kron(K, sparse.eye(r, format="csc"), format="csc")
    z_old = np.reshape(w_old, (-1,), order="F")
    g = H @ z_old if incremental else z_old.copy()

    Dj = [np.zeros(r, dtype=bool) for _ in range(n_nodes)]
    for j in range(n_nodes):
        Dj[j][p_local] = True

    for it in range(1, max_as + 1):
        wp_full = np.zeros((r, n_nodes), dtype=float)
        Nj_list = []
        ok = True
        reason = ""

        for j in range(n_nodes):
            Aj = np.asarray(A_loc_blocks[j], dtype=float)
            bj = np.asarray(b_blocks[j], dtype=float).reshape(-1)
            m = int(bj.size)
            F_j = ~Dj[j]
            nF = int(np.sum(F_j))
            if nF < m:
                ok = False
                reason = f"node {j}: |F|={nF} < m={m}"
                break

            A_jF = Aj[:, F_j]
            U, s, Vh = np.linalg.svd(A_jF, full_matrices=False)
            if s.size == 0:
                ok = False
                reason = f"node {j}: empty svd"
                break
            rnk = int(np.sum(s > tol_rank_rel * max(float(np.max(s)), 1.0)))
            if rnk < m:
                ok = False
                reason = f"node {j}: rank(A_F)={rnk} < m={m}"
                break

            Gm = A_jF @ A_jF.T
            try:
                wpj = A_jF.T @ np.linalg.solve(Gm, bj)
            except np.linalg.LinAlgError:
                wpj = A_jF.T @ np.linalg.lstsq(Gm, bj, rcond=None)[0]

            N_j = _nullspace_basis(A_jF, tol_rel=tol_rank_rel)
            wpj_full = np.zeros(r, dtype=float)
            wpj_full[F_j] = wpj
            Nj_full = sparse.csr_matrix((r, N_j.shape[1]), dtype=float)
            if N_j.size > 0:
                rr = np.flatnonzero(F_j)
                cc = np.arange(N_j.shape[1], dtype=np.int64)
                row_rep = np.repeat(rr, len(cc))
                col_tile = np.tile(cc, len(rr))
                vals = N_j.reshape(-1)
                Nj_full = sparse.coo_matrix((vals, (row_rep, col_tile)), shape=(r, N_j.shape[1])).tocsr()

            wp_full[:, j] = wpj_full
            Nj_list.append(Nj_full)

        if not ok:
            return w_old, {"ok": False, "reason": reason, "nASiter": it, "Dj": Dj}

        z_p = np.reshape(wp_full, (-1,), order="F")
        total_d = int(sum(Nj.shape[1] for Nj in Nj_list))

        if total_d == 0:
            z = z_p
        else:
            if total_d > max_reduced_dim:
                return w_old, {
                    "ok": False,
                    "reason": f"reduced dim too large ({total_d}>{max_reduced_dim})",
                    "nASiter": it,
                    "Dj": Dj,
                }

            Nmat = sparse.block_diag(Nj_list, format="csc")
            rhs_full = g - (H @ z_p)
            Hr = (Nmat.T @ H @ Nmat).tocsc()
            fr = Nmat.T @ rhs_full
            try:
                y = spla.spsolve(Hr, fr)
            except Exception:
                y = np.linalg.lstsq(Hr.toarray(), np.asarray(fr).reshape(-1), rcond=None)[0]
            z = z_p + (Nmat @ y)

        w = np.reshape(np.asarray(z, dtype=float), (r, n_nodes), order="F")

        changed = False
        for j in range(n_nodes):
            F_j = ~Dj[j]
            neg_idx = np.flatnonzero(F_j & (w[:, j] < -tol_neg))
            if neg_idx.size:
                Dj[j][neg_idx] = True
                changed = True

        if not changed:
            w_new = np.asarray(w, dtype=float)
            w_new[np.abs(w_new) < tol_neg] = 0.0
            w_new[w_new < 0.0] = 0.0
            return w_new, {"ok": True, "reason": "", "nASiter": it, "Dj": Dj}

    return w_old, {"ok": False, "reason": f"active-set did not converge ({max_as})", "nASiter": max_as, "Dj": Dj}


def _try_graph_pass(W, A_blocks, b_blocks, i_cand, iloc_pos, ind_sort, K_graph, options):
    n_try = int(options["number_of_candidates_to_try"] or len(ind_sort))
    n_try = max(1, min(n_try, len(ind_sort)))

    cand_records = []
    tol_neg = float(options["tol_neg"])

    for ord_pos, k_loc in enumerate(ind_sort[:n_try]):
        p_local = int(k_loc)
        cand_global = int(iloc_pos[p_local])

        A_loc_blocks = [Aj[:, iloc_pos] for Aj in A_blocks]
        w_old_loc = np.asarray(W[iloc_pos, :], dtype=float)
        w_trial_loc, info = _prune_step_optionb_np(
            w_old=w_old_loc,
            A_loc_blocks=A_loc_blocks,
            b_blocks=b_blocks,
            K_graph=K_graph,
            alpha=float(options["alpha_smooth"]),
            p_local=p_local,
            opts=options,
        )
        if not info.get("ok", False):
            continue

        i_cand_new = np.array([i for i in i_cand if int(i) != cand_global], dtype=np.int64)
        W_new = np.asarray(W, dtype=float).copy()
        W_new[iloc_pos, :] = w_trial_loc
        W_new[cand_global, :] = 0.0

        rows_keep_loc = np.array([i for i in range(iloc_pos.size) if i != p_local], dtype=np.int64)
        if bool(options["use_total_as_criterion"]):
            delta_w = w_trial_loc[rows_keep_loc, :]
        else:
            delta_w = w_trial_loc[rows_keep_loc, :] - w_old_loc[rows_keep_loc, :]

        metrics = edge_jump_metrics(delta_w, K_graph)
        w_keep = w_trial_loc[rows_keep_loc, :]
        zero_nodes = int(np.sum(np.any(w_keep <= max(tol_neg, float(options["tol_zero"])), axis=0)))

        cand_records.append(
            {
                "W_new": W_new,
                "i_cand_new": i_cand_new,
                "removed": cand_global,
                "zero_nodes": zero_nodes,
                "S": float(metrics["S"]),
                "R95": float(metrics["R95"]),
                "Rmax": float(metrics["Rmax"]),
                "info": info,
                "tested_rank": int(ord_pos),
            }
        )

    if not cand_records:
        return False, W, i_cand, None, {"tested": n_try}

    crit = int(options["criterion"])
    if crit == 0:
        order = sorted(cand_records, key=lambda c: (c["zero_nodes"], c["tested_rank"]))
    elif crit == 1:
        order = sorted(cand_records, key=lambda c: (c["S"], c["tested_rank"]))
    elif crit == 2:
        order = sorted(cand_records, key=lambda c: (c["R95"], c["Rmax"], c["tested_rank"]))
    else:
        raise ValueError(f"Unsupported criterion={crit}.")

    best = order[0]
    return True, best["W_new"], best["i_cand_new"], best["removed"], {
        "tested": n_try,
        "candidate_metrics": best,
    }


def _try_local_active_set_pass(W, A_blocks, b_blocks, i_cand, iloc_pos, ind_sort, options):
    """
    Python counterpart of Loop_MAWecmNOENFe (local explicit enforcement).
    No graph coupling is used here; each node is solved with local active-set
    constraints and the first feasible candidate in the sorted order is accepted.
    """
    n_try = int(options["number_of_candidates_to_try"] or len(ind_sort))
    n_try = max(1, min(n_try, len(ind_sort)))

    n_nodes = int(np.asarray(W).shape[1])
    K_zero = sparse.csr_matrix((n_nodes, n_nodes), dtype=float)
    opts_loc = dict(options)
    opts_loc["alpha_smooth"] = 0.0

    for ord_pos, k_loc in enumerate(ind_sort[:n_try]):
        p_local = int(k_loc)
        cand_global = int(iloc_pos[p_local])

        A_loc_blocks = [Aj[:, iloc_pos] for Aj in A_blocks]
        w_old_loc = np.asarray(W[iloc_pos, :], dtype=float)
        w_trial_loc, info = _prune_step_optionb_np(
            w_old=w_old_loc,
            A_loc_blocks=A_loc_blocks,
            b_blocks=b_blocks,
            K_graph=K_zero,
            alpha=0.0,
            p_local=p_local,
            opts=opts_loc,
        )
        if not info.get("ok", False):
            continue

        i_cand_new = np.array([i for i in i_cand if int(i) != cand_global], dtype=np.int64)
        W_new = np.asarray(W, dtype=float).copy()
        W_new[iloc_pos, :] = w_trial_loc
        W_new[cand_global, :] = 0.0
        return True, W_new, i_cand_new, cand_global, {
            "tested": int(ord_pos + 1),
            "candidate_metrics": {
                "tested_rank": int(ord_pos),
                "info": info,
            },
        }

    return False, W, i_cand, None, {"tested": int(n_try)}


def run_mawecm_pruning(A_blocks, b_blocks, z_ini, w_ini, q_train, options=None):
    """
    Main MAW pruning driver.

    Parameters
    ----------
    A_blocks : list of arrays
        Per-node local operators A_j with shape (m_j, n_cand).
    b_blocks : list of arrays
        Per-node local targets b_j with shape (m_j,).
    z_ini : array (n_cand,)
        Candidate full-mesh element indices.
    w_ini : array (n_cand,)
        Initial fixed ECM weights on z_ini.
    q_train : array (n_nodes, q_dim)
        Manifold coordinates for sampled nodes.
    options : dict
        Algorithm options (see DEFAULT_OPTIONS).

    Returns
    -------
    dict containing support, weights, and diagnostics.
    """
    t0 = time.perf_counter()
    opts = _merge_options(options)

    z_ini = np.asarray(z_ini, dtype=np.int64).reshape(-1)
    w_ini = np.asarray(w_ini, dtype=float).reshape(-1)
    q_train = np.asarray(q_train, dtype=float)

    if z_ini.size != w_ini.size:
        raise ValueError("z_ini and w_ini size mismatch.")
    if q_train.ndim != 2:
        raise ValueError("q_train must be 2D (n_nodes, q_dim).")

    n_cand = int(z_ini.size)
    n_nodes = int(q_train.shape[0])
    if n_nodes <= 0:
        raise ValueError("q_train must have at least one node.")

    A_blocks, b_blocks, m_list = _as_constraints(A_blocks, b_blocks, n_cand=n_cand)
    if len(A_blocks) != n_nodes:
        raise ValueError(
            f"Constraint blocks ({len(A_blocks)}) must match n_nodes ({n_nodes})."
        )

    n_stop_default = int(max(max(m_list), 1))
    n_stop = int(opts["n_stop"]) if opts.get("n_stop") is not None else n_stop_default
    n_stop = max(n_stop_default, n_stop)

    use_global_graph = bool(opts["use_global_graph_2ndstage"])
    smooth_all = bool(opts["smooth_laplacian_all_iterations"])
    max_zeros_loop = int(opts["max_number_zeros_active_set_loop"])
    enforce_nonnegativity = bool(opts["enforce_nonnegativity"])

    if use_global_graph:
        if "K_graph" not in opts:
            raise ValueError(
                "use_global_graph_2ndstage=1 requires options['K_graph'] (graph Laplacian)."
            )
        K_graph = sparse.csr_matrix(opts["K_graph"])
    else:
        # Local NOENFe stage does not use graph coupling.
        K_graph = sparse.csr_matrix((n_nodes, n_nodes), dtype=float)

    tol_neg = _negativity_tolerance(w_ini, opts["tol_neg_factor"]) if enforce_nonnegativity else 0.0
    opts["tol_neg"] = float(tol_neg)

    W = np.tile(w_ini[:, None], (1, n_nodes))
    i_cand = np.arange(n_cand, dtype=np.int64)
    removed = []

    active_counts = [int(i_cand.size)]
    removed_history = []
    stage_history = []
    no_enf_attempts = 0
    local_as_attempts = 0
    graph_attempts = 0

    while int(i_cand.size) > int(n_stop):
        if enforce_nonnegativity:
            activity = np.sum(W, axis=1)
            iloc_pos = np.array(
                [idx for idx in i_cand if activity[int(idx)] > float(opts["weight_pos_tol"])],
                dtype=np.int64,
            )
        else:
            # Signed-weight mode: use absolute activity to define active rows.
            activity = np.max(np.abs(W), axis=1)
            iloc_pos = np.array(
                [idx for idx in i_cand if activity[int(idx)] > float(opts["tol_zero"])],
                dtype=np.int64,
            )
        if iloc_pos.size == 0 or iloc_pos.size <= n_stop:
            break

        ind_sort = np.argsort(activity[iloc_pos])

        k_loc = 1
        if not smooth_all:
            ok_no, W_new, i_cand_new, removed_idx, tested_no = _try_no_enforcement_pass(
                W=W,
                A_blocks=A_blocks,
                b_blocks=b_blocks,
                i_cand=i_cand,
                iloc_pos=iloc_pos,
                ind_sort=ind_sort,
                tol_neg=tol_neg,
                tol_rank_rel=opts["tol_rank_rel"],
                enforce_nonnegativity=enforce_nonnegativity,
            )
            no_enf_attempts += int(tested_no)
            if ok_no:
                W = W_new
                i_cand = i_cand_new
                removed.append(int(removed_idx))
                removed_history.append(int(removed_idx))
                stage_history.append("no_enforcement")
                active_counts.append(int(i_cand.size))
                continue
            # MATLAB convention: if all were tested and failed, kLOC > length(ilocPOS)
            k_loc = int(tested_no + 1)

        need_stage2 = ((k_loc > int(iloc_pos.size) and max_zeros_loop > 0) or smooth_all)
        if not enforce_nonnegativity:
            # Stage-2 NOENFe / graph step is a positivity-enforcement mechanism.
            # In signed-weight mode we keep only stage-1 least-change eliminations.
            need_stage2 = False
        if not need_stage2:
            break

        if use_global_graph:
            graph_attempts += 1
            ok_s2, W_new, i_cand_new, removed_idx, _diag = _try_graph_pass(
                W=W,
                A_blocks=A_blocks,
                b_blocks=b_blocks,
                i_cand=i_cand,
                iloc_pos=iloc_pos,
                ind_sort=ind_sort,
                K_graph=K_graph,
                options=opts,
            )
            stage_label = "graph_active_set"
        else:
            local_as_attempts += 1
            ok_s2, W_new, i_cand_new, removed_idx, _diag = _try_local_active_set_pass(
                W=W,
                A_blocks=A_blocks,
                b_blocks=b_blocks,
                i_cand=i_cand,
                iloc_pos=iloc_pos,
                ind_sort=ind_sort,
                options=opts,
            )
            stage_label = "local_active_set"

        if not ok_s2:
            break

        W = W_new
        i_cand = i_cand_new
        removed.append(int(removed_idx))
        removed_history.append(int(removed_idx))
        stage_history.append(stage_label)
        active_counts.append(int(i_cand.size))

    # Ensure strict zeros on removed set.
    if removed:
        W[np.asarray(removed, dtype=np.int64), :] = 0.0

    if enforce_nonnegativity:
        support_mask = np.sum(W, axis=1) > float(opts["tol_zero"])
    else:
        support_mask = np.max(np.abs(W), axis=1) > float(opts["tol_zero"])
    i_support = np.flatnonzero(support_mask).astype(np.int64)

    out = {
        "z_ini": z_ini,
        "w_ini": w_ini,
        "W_full": W,
        "i_support_local": i_support,
        "Z_support": z_ini[i_support],
        "W_support": W[i_support, :],
        "i_cand_final": np.asarray(i_cand, dtype=np.int64),
        "removed_local": np.asarray(removed_history, dtype=np.int64),
        "active_counts": np.asarray(active_counts, dtype=np.int64),
        "stage_history": np.asarray(stage_history, dtype="U32"),
        "n_stop": int(n_stop),
        "tol_neg": float(tol_neg),
        "options": opts,
        "no_enforcement_attempts": int(no_enf_attempts),
        "local_active_set_attempts": int(local_as_attempts),
        "graph_attempts": int(graph_attempts),
        "elapsed_sec": float(time.perf_counter() - t0),
    }

    if opts["verbose"]:
        print(
            "[MAW-ECM] pruning finished: "
            f"|Z_ini|={n_cand}, |Z_final|={i_support.size}, n_stop={n_stop}, "
            f"removed={len(removed_history)}, elapsed={out['elapsed_sec']:.2f}s"
        )

    return out
