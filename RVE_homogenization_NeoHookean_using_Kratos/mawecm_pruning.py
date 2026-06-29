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
import warnings
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
    "warn_max_reduced_dim": False,
    "weight_pos_tol": 0.0,
    "enforce_nonnegativity": True,
    # Joaquín-style control flags
    "smooth_laplacian_all_iterations": False,
    "use_global_graph_2ndstage": False,
    "max_number_zeros_active_set_loop": 1,
    "verbose": True,
    "n_stop": None,
    "phase1_stop_size": None,
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
    if out.get("phase1_stop_size") is not None:
        out["phase1_stop_size"] = int(out["phase1_stop_size"])
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
            if A_keep.shape[0] == 0:
                # No local constraints at this node after rank compression.
                # Feasible by construction; keep weights unchanged here.
                continue
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


def _apply_graph_hessian(W, K_graph, alpha):
    W = np.asarray(W, dtype=float)
    return W + float(alpha) * (sparse.csr_matrix(K_graph) @ W.T).T


def _solve_graph_nullspace_matrix_free(
    w_old,
    wp_full,
    A_loc_blocks,
    b_blocks,
    K_graph,
    alpha,
    active_masks,
    incremental,
    tol_rank_rel,
):
    """
    Solve the graph QP by projected CG without explicitly constructing N.

    The iterate is w = w_p + y, where every local correction y_j is
    projected onto ker(A_j). Therefore A_j w_j = b_j is preserved by
    construction while applying the graph Hessian matrix-free.
    """
    w_ref = np.asarray(w_old, dtype=float)
    wp = np.asarray(wp_full, dtype=float)
    r, n_nodes = w_ref.shape
    m_sizes = [int(np.asarray(b).size) for b in b_blocks]
    if not m_sizes or min(m_sizes) <= 0 or len(set(m_sizes)) != 1:
        return w_ref, {
            "ok": False,
            "reason": "matrix-free graph solve requires equal nonzero local constraint sizes",
            "iterations": 0,
        }

    A = np.stack([np.asarray(Aj, dtype=float) for Aj in A_loc_blocks], axis=0)
    active = np.stack(
        [np.asarray(mask, dtype=bool) for mask in active_masks],
        axis=1,
    )
    free = ~active

    # Row scaling improves the local projector conditioning without changing
    # either the equality constraints or their null spaces.
    A_free = A * free.T[:, None, :]
    row_scale = np.linalg.norm(A_free, axis=2)
    if np.any(~np.isfinite(row_scale)) or np.any(row_scale <= float(tol_rank_rel)):
        return w_ref, {
            "ok": False,
            "reason": "invalid local constraint scaling in matrix-free graph solve",
            "iterations": 0,
        }
    A_scaled = A / row_scale[:, :, None]
    A_scaled_free = A_scaled * free.T[:, None, :]
    gram = np.einsum(
        "jmr,jnr->jmn",
        A_scaled_free,
        A_scaled_free,
        optimize=True,
    )
    try:
        gram_inv = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        gram_inv = np.linalg.pinv(
            gram,
            rcond=max(float(tol_rank_rel), 1.0e-14),
        )

    free_idx = np.flatnonzero(free.reshape(-1, order="F"))
    n_free = int(free_idx.size)
    if n_free <= 0:
        return w_ref, {
            "ok": False,
            "reason": "matrix-free graph solve has no free variables",
            "iterations": 0,
        }

    K = sparse.csr_matrix(K_graph)

    def _scatter_free(x_free):
        z = np.zeros(r * n_nodes, dtype=float)
        z[free_idx] = np.asarray(x_free, dtype=float)
        return z.reshape((r, n_nodes), order="F")

    def _gather_free(W):
        return np.asarray(W, dtype=float).reshape(-1, order="F")[free_idx]

    def _project(W):
        W_free = np.asarray(W, dtype=float) * free
        local_rhs = np.einsum(
            "jmr,rj->jm",
            A_scaled,
            W_free,
            optimize=True,
        )
        local_lam = np.einsum(
            "jmn,jn->jm",
            gram_inv,
            local_rhs,
            optimize=True,
        )
        correction = np.einsum(
            "jmr,jm->rj",
            A_scaled,
            local_lam,
            optimize=True,
        )
        return (W_free - correction) * free

    g_full = (
        _apply_graph_hessian(w_ref, K, alpha)
        if bool(incremental)
        else w_ref
    )
    rhs_full = _project(g_full - _apply_graph_hessian(wp, K, alpha))
    rhs = _gather_free(rhs_full)

    def _matvec(x):
        Y = _project(_scatter_free(x))
        HY = _apply_graph_hessian(Y, K, alpha)
        return _gather_free(_project(HY))

    operator = spla.LinearOperator(
        shape=(n_free, n_free),
        matvec=_matvec,
        rmatvec=_matvec,
        dtype=float,
    )

    degree = np.asarray(K.diagonal(), dtype=float).reshape(-1)
    h_diag_node = 1.0 + float(alpha) * degree

    def _precondition(x):
        Y = _scatter_free(x) / h_diag_node[None, :]
        return _gather_free(_project(Y))

    preconditioner = spla.LinearOperator(
        shape=(n_free, n_free),
        matvec=_precondition,
        rmatvec=_precondition,
        dtype=float,
    )

    iteration_count = 0

    def _count_iteration(_):
        nonlocal iteration_count
        iteration_count += 1

    solution, info = spla.cg(
        operator,
        rhs,
        x0=np.zeros(n_free, dtype=float),
        rtol=1.0e-9,
        atol=0.0,
        maxiter=1000,
        M=preconditioner,
        callback=_count_iteration,
    )
    if int(info) != 0 or np.any(~np.isfinite(solution)):
        return w_ref, {
            "ok": False,
            "reason": f"matrix-free projected CG did not converge (info={int(info)})",
            "iterations": int(iteration_count),
        }

    W = wp + _project(_scatter_free(solution))
    constraint_rel = np.zeros(n_nodes, dtype=float)
    for j, (Aj, bj) in enumerate(zip(A_loc_blocks, b_blocks)):
        bj_arr = np.asarray(bj, dtype=float).reshape(-1)
        residual = np.asarray(Aj, dtype=float) @ W[:, j] - bj_arr
        constraint_rel[j] = np.linalg.norm(residual) / max(np.linalg.norm(bj_arr), 1.0)
    max_constraint_rel = float(np.max(constraint_rel))
    if max_constraint_rel > 1.0e-7:
        return w_ref, {
            "ok": False,
            "reason": (
                "matrix-free projected CG constraint error too large "
                f"({max_constraint_rel:.3e})"
            ),
            "iterations": int(iteration_count),
            "max_constraint_rel": max_constraint_rel,
        }

    projected_residual = _project(
        _apply_graph_hessian(W, K, alpha) - g_full
    )
    rel_projected_residual = float(
        np.linalg.norm(projected_residual)
        / max(np.linalg.norm(_project(g_full)), 1.0)
    )
    return W, {
        "ok": True,
        "reason": "",
        "iterations": int(iteration_count),
        "max_constraint_rel": max_constraint_rel,
        "rel_projected_residual": rel_projected_residual,
        "n_free": n_free,
    }

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

    H = None
    g = None

    Dj = [np.zeros(r, dtype=bool) for _ in range(n_nodes)]
    for j in range(n_nodes):
        Dj[j][p_local] = True

    for it in range(1, max_as + 1):
        wp_full = np.zeros((r, n_nodes), dtype=float)
        Nj_list = []
        ok = True
        reason = ""
        total_d = 0
        use_matrix_free = False

        for j in range(n_nodes):
            Aj = np.asarray(A_loc_blocks[j], dtype=float)
            bj = np.asarray(b_blocks[j], dtype=float).reshape(-1)
            m = int(bj.size)
            if m == 0:
                # No local constraints at this node; keep wp=0.
                total_d += int(np.sum(~Dj[j]))
                continue
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

            total_d += int(nF - m)
            Gm = A_jF @ A_jF.T
            try:
                wpj = A_jF.T @ np.linalg.solve(Gm, bj)
            except np.linalg.LinAlgError:
                wpj = A_jF.T @ np.linalg.lstsq(Gm, bj, rcond=None)[0]

            wpj_full = np.zeros(r, dtype=float)
            wpj_full[F_j] = wpj
            wp_full[:, j] = wpj_full

        if not ok:
            return w_old, {"ok": False, "reason": reason, "nASiter": it, "Dj": Dj}

        use_matrix_free = int(total_d) > int(max_reduced_dim)
        if use_matrix_free:
            if bool(opts.get("verbose", False)) and it == 1:
                print(
                    "[MAW-ECM][Phase2][graph] "
                    f"matrix-free solve: r={r}, nodes={n_nodes}, "
                    f"free_dim={int(total_d)}, max_reduced_dim={int(max_reduced_dim)}",
                    flush=True,
                )
            w, solve_info = _solve_graph_nullspace_matrix_free(
                w_old=w_old,
                wp_full=wp_full,
                A_loc_blocks=A_loc_blocks,
                b_blocks=b_blocks,
                K_graph=K,
                alpha=float(alpha),
                active_masks=Dj,
                incremental=incremental,
                tol_rank_rel=tol_rank_rel,
            )
            if not bool(solve_info["ok"]):
                return w_old, {
                    "ok": False,
                    "reason": str(solve_info["reason"]),
                    "nASiter": it,
                    "Dj": Dj,
                    "linear_solver": solve_info,
                }
        else:
            if H is None:
                N = r * n_nodes
                H = sparse.eye(N, format="csc") + float(alpha) * sparse.kron(
                    K,
                    sparse.eye(r, format="csc"),
                    format="csc",
                )
                z_old = np.reshape(w_old, (-1,), order="F")
                g = H @ z_old if incremental else z_old.copy()
            for j in range(n_nodes):
                Aj = np.asarray(A_loc_blocks[j], dtype=float)
                F_j = ~Dj[j]
                if Aj.shape[0] == 0:
                    Nj_full = sparse.csr_matrix(
                        np.eye(r, dtype=float)[:, F_j]
                    )
                    Nj_list.append(Nj_full)
                    continue
                N_j = _nullspace_basis(Aj[:, F_j], tol_rel=tol_rank_rel)
                Nj_full = sparse.csr_matrix((r, N_j.shape[1]), dtype=float)
                if N_j.size > 0:
                    rr = np.flatnonzero(F_j)
                    cc = np.arange(N_j.shape[1], dtype=np.int64)
                    row_rep = np.repeat(rr, len(cc))
                    col_tile = np.tile(cc, len(rr))
                    vals = N_j.reshape(-1)
                    Nj_full = sparse.coo_matrix(
                        (vals, (row_rep, col_tile)),
                        shape=(r, N_j.shape[1]),
                    ).tocsr()
                Nj_list.append(Nj_full)

        z_p = np.reshape(wp_full, (-1,), order="F")
        if not use_matrix_free:
            total_d = int(sum(Nj.shape[1] for Nj in Nj_list))
            if total_d == 0:
                z = z_p
            else:
                if total_d > max_reduced_dim and bool(opts.get("warn_max_reduced_dim", False)):
                    warnings.warn(
                        "MAW-ECM reduced active-set dimension exceeds "
                        f"--max-reduced-dim ({total_d}>{max_reduced_dim}); "
                        "continuing with the solve.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

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
            out_info = {
                "ok": True,
                "reason": "",
                "nASiter": it,
                "Dj": Dj,
            }
            if use_matrix_free:
                out_info["linear_solver"] = solve_info
            return w_new, out_info

    return w_old, {"ok": False, "reason": f"active-set did not converge ({max_as})", "nASiter": max_as, "Dj": Dj}


def _try_graph_pass(W, A_blocks, b_blocks, i_cand, iloc_pos, ind_sort, K_graph, options):
    n_try = int(options["number_of_candidates_to_try"] or len(ind_sort))
    n_try = max(1, min(n_try, len(ind_sort)))

    cand_records = []
    fail_reasons = {}
    tol_neg = float(options["tol_neg"])

    for ord_pos, k_loc in enumerate(ind_sort[:n_try]):
        p_local = int(k_loc)
        cand_global = int(iloc_pos[p_local])
        if bool(options.get("verbose", False)):
            if ord_pos == 0 or ord_pos + 1 == n_try or (ord_pos + 1) % 10 == 0:
                print(
                    "[MAW-ECM][Phase2][graph] "
                    f"testing candidate {ord_pos + 1}/{n_try} "
                    f"(local={p_local}, global={cand_global}, active={int(iloc_pos.size)}, "
                    f"nodes={len(b_blocks)})",
                    flush=True,
                )

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
            rs = str(info.get("reason", "unknown"))
            fail_reasons[rs] = int(fail_reasons.get(rs, 0)) + 1
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
        return False, W, i_cand, None, {
            "tested": int(n_try),
            "n_feasible": 0,
            "fail_reasons": fail_reasons,
        }

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
        "tested": int(n_try),
        "n_feasible": int(len(cand_records)),
        "fail_reasons": fail_reasons,
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

    fail_reasons = {}
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
            reason = str(info.get("reason", "unknown"))
            fail_reasons[reason] = int(fail_reasons.get(reason, 0)) + 1
            continue

        i_cand_new = np.array([i for i in i_cand if int(i) != cand_global], dtype=np.int64)
        W_new = np.asarray(W, dtype=float).copy()
        W_new[iloc_pos, :] = w_trial_loc
        W_new[cand_global, :] = 0.0
        return True, W_new, i_cand_new, cand_global, {
            "tested": int(ord_pos + 1),
            "fail_reasons": fail_reasons,
            "candidate_metrics": {
                "tested_rank": int(ord_pos),
                "info": info,
            },
        }

    return False, W, i_cand, None, {"tested": int(n_try), "fail_reasons": fail_reasons}


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
    phase1_stop_size = opts.get("phase1_stop_size", None)
    if phase1_stop_size is not None and int(phase1_stop_size) > 0:
        phase1_stop_size = max(int(phase1_stop_size), int(n_stop))
    else:
        phase1_stop_size = None

    use_global_graph = bool(opts["use_global_graph_2ndstage"])
    smooth_all = bool(opts["smooth_laplacian_all_iterations"])
    max_zeros_loop = int(opts["max_number_zeros_active_set_loop"])
    enforce_nonnegativity = bool(opts["enforce_nonnegativity"])

    # Default policy: stage-1 NOENF runs first and stage-2 regularization is
    # only used after phase-1 stalls. If smooth_all=True, stage-2 is explicitly
    # forced from the first elimination attempt.

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
    phase1_start_size = int(n_cand)
    phase1_end_size = int(n_cand)
    phase2_started = False
    phase2_mode = "graph_regularization" if use_global_graph else "local_active_set_regularization"
    phase2_start_size = -1
    phase2_end_size = -1
    phase2_attempts = 0
    phase2_successes = 0

    if opts["verbose"]:
        if smooth_all:
            print(
                "[MAW-ECM][Phase1] skipped "
                "(smooth_laplacian_all_iterations=1): "
                f"stage-2 regularization is forced from |Z|={phase1_start_size}, "
                f"target n_stop={n_stop}"
            )
        else:
            extra = ""
            if phase1_stop_size is not None:
                extra = f", phase1_stop={phase1_stop_size}"
            print(
                "[MAW-ECM][Phase1] start (no regularization): "
                f"|Z|={phase1_start_size}, target n_stop={n_stop}{extra}"
            )

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
        force_stage2_by_phase1_stop = (
            phase1_stop_size is not None and int(i_cand.size) <= int(phase1_stop_size)
        )
        if not smooth_all and not force_stage2_by_phase1_stop:
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
                phase1_end_size = int(i_cand.size)
                continue
            # MATLAB convention: if all were tested and failed, kLOC > length(ilocPOS)
            k_loc = int(tested_no + 1)

        need_stage2 = (
            (k_loc > int(iloc_pos.size) and max_zeros_loop > 0)
            or smooth_all
            or (force_stage2_by_phase1_stop and max_zeros_loop > 0)
        )
        if not enforce_nonnegativity:
            # Stage-2 NOENFe / graph step is a positivity-enforcement mechanism.
            # In signed-weight mode we keep only stage-1 least-change eliminations.
            need_stage2 = False
        if not need_stage2:
            break

        if not phase2_started:
            phase2_started = True
            phase1_end_size = int(i_cand.size)
            phase2_start_size = int(i_cand.size)
            if opts["verbose"]:
                if not smooth_all:
                    print(
                        "[MAW-ECM][Phase1] end: "
                        f"|Z|={phase1_end_size}, removed={phase1_start_size - phase1_end_size}"
                    )
                print(
                    "[MAW-ECM][Phase2] starting regularization "
                    f"({phase2_mode}), |Z|={phase2_start_size}"
                )

        if use_global_graph:
            graph_attempts += 1
            phase2_attempts += 1
            ok_s2, W_new, i_cand_new, removed_idx, diag = _try_graph_pass(
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
            phase2_attempts += 1
            ok_s2, W_new, i_cand_new, removed_idx, diag = _try_local_active_set_pass(
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
            if opts["verbose"]:
                reason_txt = ""
                fail_reasons = diag.get("fail_reasons", {}) if isinstance(diag, dict) else {}
                if isinstance(fail_reasons, dict) and fail_reasons:
                    top = sorted(fail_reasons.items(), key=lambda kv: kv[1], reverse=True)
                    top_txt = ", ".join([f"{k} x{int(v)}" for k, v in top[:3]])
                    reason_txt = f" Top fail reasons: {top_txt}."
                print(
                    "[MAW-ECM][Phase2] no feasible elimination found; stopping "
                    f"after {phase2_attempts} phase-2 attempt(s). "
                    f"Candidates tested in last attempt={int(diag.get('tested', 0))}."
                    f"{reason_txt}"
                )
            break

        W = W_new
        i_cand = i_cand_new
        phase2_successes += 1
        removed.append(int(removed_idx))
        removed_history.append(int(removed_idx))
        stage_history.append(stage_label)
        active_counts.append(int(i_cand.size))
        if opts["verbose"]:
            solver_txt = ""
            candidate_metrics = (
                diag.get("candidate_metrics", {})
                if isinstance(diag, dict)
                else {}
            )
            solve_info = (
                candidate_metrics.get("info", {}).get("linear_solver", {})
                if isinstance(candidate_metrics, dict)
                else {}
            )
            if solve_info:
                solver_txt = (
                    f", projected-CG iters={int(solve_info.get('iterations', 0))}, "
                    f"constraint_rel="
                    f"{float(solve_info.get('max_constraint_rel', np.nan)):.2e}"
                )
            print(
                "[MAW-ECM][Phase2] accepted elimination: "
                f"removed local idx={int(removed_idx)}, |Z|={int(i_cand.size)}"
                f"{solver_txt}"
            )

    # Ensure strict zeros on removed set.
    if removed:
        W[np.asarray(removed, dtype=np.int64), :] = 0.0

    if enforce_nonnegativity:
        support_mask = np.sum(W, axis=1) > float(opts["tol_zero"])
    else:
        support_mask = np.max(np.abs(W), axis=1) > float(opts["tol_zero"])
    i_support = np.flatnonzero(support_mask).astype(np.int64)
    if not phase2_started:
        phase1_end_size = int(i_support.size)
    phase2_end_size = int(i_support.size) if phase2_started else -1

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
        "phase1_stop_size": (-1 if phase1_stop_size is None else int(phase1_stop_size)),
        "tol_neg": float(tol_neg),
        "options": opts,
        "no_enforcement_attempts": int(no_enf_attempts),
        "local_active_set_attempts": int(local_as_attempts),
        "graph_attempts": int(graph_attempts),
        "phase1_start_size": int(phase1_start_size),
        "phase1_end_size": int(phase1_end_size),
        "phase2_started": bool(phase2_started),
        "phase2_mode": str(phase2_mode),
        "phase2_start_size": int(phase2_start_size),
        "phase2_end_size": int(phase2_end_size),
        "phase2_attempts": int(phase2_attempts),
        "phase2_successes": int(phase2_successes),
        "elapsed_sec": float(time.perf_counter() - t0),
    }

    if opts["verbose"]:
        if phase2_started:
            print(
                "[MAW-ECM][Phase2] end: "
                f"|Z_start|={phase2_start_size} -> |Z_final|={phase2_end_size}, "
                f"attempts={phase2_attempts}, accepted={phase2_successes}"
            )
        else:
            print(
                "[MAW-ECM][Phase2] not triggered "
                f"(phase-1 reached |Z|={phase1_end_size})."
            )
        print(
            "[MAW-ECM] pruning finished: "
            f"|Z_ini|={n_cand}, |Z_final|={i_support.size}, n_stop={n_stop}, "
            f"removed={len(removed_history)}, elapsed={out['elapsed_sec']:.2f}s"
        )

    return out
