#!/usr/bin/env python3
"""Stage 00A: the effective zero-strain tangent C0 under PERIODIC boundary
conditions, from an independent linear-elastic assembly.

Why this exists. The affine-Dirichlet cell-count study measured a clean 1/n
boundary-layer law (R^2 = 0.99992 over n = 2..5), which means the shear-normal
coupling -- the very quantity that is the paper's group-free signature --
cannot be converged by growing the block: error = 0.441/n while cost grows as
n^2, i.e. cost ~ 1/error^2. Reaching 2% would need n ~ 22, about 135,000
elements. Periodic BCs instead give the converged answer on ONE cell, which is
exactly why they are the standard in this literature.

Why an independent implementation rather than Kratos constraints. Only the
LINEAR effective stiffness is needed here, so no nonlinear solve and no
tangent derivation is involved, and writing it standalone buys a strong
verification: run the same assembly with AFFINE boundary conditions and it
must reproduce the Kratos analytic-tangent values already measured
(1x1: Ex = 1.26502e9, nu_xy = 0.4991, g12/E11 = -0.1863). That validates this
code against the project's validated path before it is trusted for the
periodic result, which has no independent reference except the 1/n
extrapolation it is meant to test.

Two gates, therefore:
  gate 1  affine here  == Kratos affine        (validates this assembly)
  gate 2  periodic here == the 1/n extrapolation (validates both methods)

Small strain is the right setting: at E = 0 a hyperelastic tangent coincides
with the linear elastic stiffness, since the geometric/initial-stress terms
carry a factor of the stress, which vanishes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from gen_rve_mesh import build_mesh  # noqa: E402

# Degree-4, 6-point Gauss rule on the triangle. Degree 4 rather than 2 because
# elements on the elliptical rim are isoparametrically curved, so the Jacobian
# is not constant over the element and B^T D B is not merely quadratic.
_A1, _W1 = 0.445948490915965, 0.223381589678011
_A2, _W2 = 0.091576213509771, 0.109951743655322
GAUSS = np.array([
    [_A1, _A1, _W1], [1 - 2 * _A1, _A1, _W1], [_A1, 1 - 2 * _A1, _W1],
    [_A2, _A2, _W2], [1 - 2 * _A2, _A2, _W2], [_A2, 1 - 2 * _A2, _W2],
])


def plane_strain_D(young=None, poisson=None):
    """Voigt [exx, eyy, gxy] with gxy the ENGINEERING shear (= 2*e12), which is
    the convention core/fom_solver_rve.py uses (GreenLagrangeVoigtToTensor2D
    comments the third slot as `engineering shear = 2 * E12`)."""
    E = cfg.MATRIX_YOUNG if young is None else young
    nu = cfg.MATRIX_POISSON if poisson is None else poisson
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return np.array([[lam + 2 * mu, lam, 0.0],
                     [lam, lam + 2 * mu, 0.0],
                     [0.0, 0.0, mu]])


def _t6_shape_derivs(xi, eta):
    """dN/dxi, dN/deta for the T6 in the project's node order: corners 0,1,2
    then mid(0,1), mid(1,2), mid(2,0)."""
    l1, l2, l3 = 1.0 - xi - eta, xi, eta
    dxi = np.array([1 - 4 * l1, 4 * l2 - 1, 0.0,
                    4 * (l1 - l2), 4 * l3, -4 * l3])
    det = np.array([1 - 4 * l1, 0.0, 4 * l3 - 1,
                    -4 * l2, 4 * l2, 4 * (l1 - l3)])
    return dxi, det


def assemble(coords, tris, D):
    """Global stiffness K, plus per-Gauss-point (B, weight) for the stress
    average. Unit thickness: it cancels, since the homogenized stress divides
    by thickness * A0 and the assembly multiplies by the same thickness."""
    n_dof = 2 * coords.shape[0]
    rows, cols, vals = [], [], []
    gp_B, gp_w, gp_dofs = [], [], []
    for t in tris:
        xy = coords[t]
        dofs = np.empty(12, dtype=np.int64)
        dofs[0::2] = 2 * t
        dofs[1::2] = 2 * t + 1
        Ke = np.zeros((12, 12))
        for (xi, eta, w) in GAUSS:
            dxi, det = _t6_shape_derivs(xi, eta)
            J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                          [det @ xy[:, 0], det @ xy[:, 1]]])
            detJ = np.linalg.det(J)
            if detJ <= 0.0:
                raise RuntimeError("non-positive Jacobian: inverted element")
            dN = np.linalg.solve(J, np.vstack((dxi, det)))  # (2, 6) -> d/dx, d/dy
            B = np.zeros((3, 12))
            B[0, 0::2] = dN[0]
            B[1, 1::2] = dN[1]
            B[2, 0::2] = dN[1]
            B[2, 1::2] = dN[0]
            wgt = 0.5 * w * detJ
            Ke += wgt * (B.T @ D @ B)
            gp_B.append(B)
            gp_w.append(wgt)
            gp_dofs.append(dofs)
        rows.append(np.repeat(dofs, 12))
        cols.append(np.tile(dofs, 12))
        vals.append(Ke.ravel())
    K = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_dof, n_dof)).tocsr()
    return K, np.array(gp_B), np.array(gp_w), np.array(gp_dofs)


def _voigt_to_tensor(e):
    return np.array([[e[0], 0.5 * e[2]], [0.5 * e[2], e[1]]])


def _periodic_pairs(coords, half, tol):
    """slave dof -> (master dof, geometric offset vector index) for the two
    translation directions, matching opposite faces by the free coordinate."""
    x, y = coords[:, 0], coords[:, 1]
    pairs = []  # (slave_node, master_node)
    for (sel_s, sel_m, key) in (
            (np.abs(x - half) < tol, np.abs(x + half) < tol, y),
            (np.abs(y - half) < tol, np.abs(y + half) < tol, x)):
        s_idx = np.where(sel_s)[0]
        m_idx = np.where(sel_m)[0]
        if s_idx.size != m_idx.size:
            raise RuntimeError(f"non-matching face node counts: "
                               f"{s_idx.size} vs {m_idx.size} -- mesh is not periodic")
        s_sorted = s_idx[np.argsort(key[s_idx])]
        m_sorted = m_idx[np.argsort(key[m_idx])]
        mism = np.max(np.abs(key[s_sorted] - key[m_sorted]))
        if mism > tol:
            raise RuntimeError(f"face node positions do not match to {tol:.1e} "
                               f"(worst {mism:.3e}) -- mesh is not periodic")
        pairs.extend(zip(s_sorted.tolist(), m_sorted.tolist()))
    return pairs


def solve_C0(coords, tris, mode, half, tol=1.0e-7, verbose=False):
    """C0 (3,3) column by column: impose unit macro strain, average the stress.

    mode 'affine'   -- every boundary node prescribed u = eps.X (what the
                       project's Kratos path does)
    mode 'periodic' -- opposite faces tied, u+ - u- = eps.(X+ - X-), one node
                       pinned to remove the free translation
    """
    D = plane_strain_D()
    K, gp_B, gp_w, gp_dofs = assemble(coords, tris, D)
    n_node = coords.shape[0]
    n_dof = 2 * n_node
    A0 = (2.0 * half) ** 2

    x, y = coords[:, 0], coords[:, 1]
    on_outer = (np.abs(np.abs(x) - half) < tol) | (np.abs(np.abs(y) - half) < tol)

    # Resolve each dof to a root plus an accumulated geometric offset. Corner
    # nodes are slaves in BOTH directions, so chains must be followed rather
    # than assumed one deep.
    root = np.arange(n_dof, dtype=np.int64)
    off_node = np.zeros((n_dof, 2))   # offset expressed as a REFERENCE vector
    pinned = np.zeros(n_dof, dtype=bool)

    if mode == "affine":
        pinned[2 * np.where(on_outer)[0]] = True
        pinned[2 * np.where(on_outer)[0] + 1] = True
    elif mode == "periodic":
        for (s, m) in _periodic_pairs(coords, half, tol):
            for c in (0, 1):
                sd, md = 2 * s + c, 2 * m + c
                if root[sd] != sd:      # already tied; chain will resolve it
                    continue
                root[sd] = md
                off_node[sd] = coords[s] - coords[m]
        # Pin the corner that masters both directions: removes exactly the
        # rigid translation. Rotation is already excluded, since a symmetric
        # macro strain gives a symmetric average displacement gradient.
        pin_node = int(np.argmin((x + half) ** 2 + (y + half) ** 2))
        pinned[2 * pin_node] = True
        pinned[2 * pin_node + 1] = True
    else:
        raise ValueError(mode)

    # Follow chains to a fixed point, accumulating offsets.
    final_root = np.arange(n_dof, dtype=np.int64)
    final_off = np.zeros((n_dof, 2))
    for d in range(n_dof):
        r, acc = d, np.zeros(2)
        guard = 0
        while root[r] != r:
            acc = acc + off_node[r]
            r = root[r]
            guard += 1
            if guard > 8:
                raise RuntimeError("periodicity chain does not terminate")
        final_root[d] = r
        final_off[d] = acc

    # Independent dofs: roots that are not pinned.
    is_ind = (final_root == np.arange(n_dof)) & (~pinned)
    red = -np.ones(n_dof, dtype=np.int64)
    red[is_ind] = np.arange(int(np.sum(is_ind)))
    n_ind = int(np.sum(is_ind))

    t_rows, t_cols = [], []
    for d in range(n_dof):
        r = final_root[d]
        if not pinned[r]:
            t_rows.append(d)
            t_cols.append(red[r])
    T = sp.coo_matrix((np.ones(len(t_rows)), (t_rows, t_cols)),
                      shape=(n_dof, n_ind)).tocsr()
    Kr = (T.T @ K @ T).tocsc()
    lu = spla.splu(Kr)

    C0 = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1.0
        eps = _voigt_to_tensor(e)

        # Inhomogeneous part: prescribed value at a pinned root, plus the
        # accumulated periodic jump eps.(X_slave - X_master).
        g = np.zeros(n_dof)
        for d in range(n_dof):
            r = final_root[d]
            c = d % 2
            val = float(eps[c] @ final_off[d])
            if pinned[r]:
                node_r = r // 2
                val += float(eps[c] @ coords[node_r])
            g[d] = val

        u = T @ lu.solve(-(T.T @ (K @ g))) + g
        sig = np.zeros(3)
        for i in range(gp_B.shape[0]):
            sig += gp_w[i] * (D @ (gp_B[i] @ u[gp_dofs[i]]))
        C0[:, j] = sig / A0
        if verbose:
            print(f"   column {j}: {C0[:, j]}")
    return C0, dict(n_dof=n_dof, n_ind=n_ind, A0=A0)


def eng(C0):
    S = np.linalg.inv(C0)
    return dict(Ex=1.0 / S[0, 0], nu_xy=-S[0, 1] / S[0, 0],
                shear=S[0, 2] / S[0, 0])


# Kratos affine reference at 1x1, coarse mesh (compute_C0.py --mode cells,
# n=1, size_far=0.30 size_hole=0.14) and the 1/n extrapolated periodic limit
# (least squares over n=2..5, R^2 = 0.99992).
KRATOS_AFFINE_1X1 = dict(Ex=1.26502e9, nu_xy=0.4991, shear=-0.1863)
EXTRAPOLATED_LIMIT = dict(Ex=1.21182e9, nu_xy=0.47149, shear=-0.31024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-far", type=float, default=0.30)
    ap.add_argument("--size-hole", type=float, default=0.14)
    a = ap.parse_args()

    print("gate 1: this assembly with AFFINE BCs vs the Kratos analytic tangent")
    coords, tris, _outer, geom = build_mesh(size_far=a.size_far, size_hole=a.size_hole)
    half = geom["block_side"] / 2.0
    C_aff, info = solve_C0(coords, tris, "affine", half)
    e_aff = eng(C_aff)
    for k, ref in KRATOS_AFFINE_1X1.items():
        rel = abs(e_aff[k] - ref) / abs(ref)
        print(f"  {k:8s} {e_aff[k]:+.6e}  kratos {ref:+.6e}  rel {rel:.3e}"
              f"  {'OK' if rel < 5.0e-3 else 'FAIL'}")
    gate1 = all(abs(e_aff[k] - r) / abs(r) < 5.0e-3 for k, r in KRATOS_AFFINE_1X1.items())

    print("\ngate 2: PERIODIC BCs vs the affine 1/n extrapolation")
    pc, pt, _po, pg = build_mesh(size_far=a.size_far, size_hole=a.size_hole,
                                 periodic=True)
    phalf = pg["block_side"] / 2.0
    C_per, pinfo = solve_C0(pc, pt, "periodic", phalf)
    e_per = eng(C_per)
    print(f"  periodic mesh: {pg['n_nodes']} nodes, {pg['n_elements']} elems, "
          f"{pinfo['n_ind']}/{pinfo['n_dof']} independent dofs")
    for k, ref in EXTRAPOLATED_LIMIT.items():
        rel = abs(e_per[k] - ref) / abs(ref)
        print(f"  {k:8s} {e_per[k]:+.6e}  extrap {ref:+.6e}  rel {rel:.3e}")
    gate2 = all(abs(e_per[k] - r) / abs(r) < 0.05 for k, r in EXTRAPOLATED_LIMIT.items())

    with np.printoptions(precision=4, linewidth=120):
        print("\nC0 periodic (Pa) =\n", C_per)
    asym = np.max(np.abs(C_per - C_per.T)) / np.max(np.abs(C_per))
    eigv = np.linalg.eigvalsh(0.5 * (C_per + C_per.T))
    print(f"  symmetry rel asym {asym:.3e}   eigenvalues {eigv}")
    print(f"  |C13|/C11 {abs(C_per[0, 2]) / abs(C_per[0, 0]):.4e}   "
          f"|C23|/C22 {abs(C_per[1, 2]) / abs(C_per[1, 1]):.4e}")

    print(f"\ngate 1 (assembly validated)  {'PASS' if gate1 else 'FAIL'}")
    print(f"gate 2 (periodic vs extrap)  {'PASS' if gate2 else 'FAIL'}")
    np.savez(HERE / "C0_periodic.npz", C0_periodic=C_per, C0_affine=C_aff)
    print("PERIODIC_C0_PASS" if (gate1 and gate2) else "PERIODIC_C0_FAIL")
    return 0 if (gate1 and gate2) else 1


if __name__ == "__main__":
    sys.exit(main())
