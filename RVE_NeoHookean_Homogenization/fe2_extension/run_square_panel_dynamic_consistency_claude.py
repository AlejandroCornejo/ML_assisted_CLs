#!/usr/bin/env python3
"""Dynamic analog of As'ad, Avery & Farhat (2022, IJNME 123:2738-2759,
Section 4.3/Figure 9): a completely FREE-FLOATING square panel (no
Dirichlet constraints at all -- unlike every other test this session),
given only a uniform initial velocity and zero external force. The exact
solution is rigid-body translation forever, no deformation. Explicit
central-difference time integration:

    u[n+1] = 2 u[n] - u[n-1] + dt^2 * M_lumped^{-1} * (f_ext - f_int(u[n]))

f_int(u) is exactly VectorizedAssembler.Assemble(u)'s own rhs_int (the same
call every other driver this session already uses) -- the material law is
the only thing that changes between runs. M_lumped is a real Kratos mass
matrix (TotalLagrangianElement2D6N.CalculateMassMatrix, verified this
session to work natively -- row-sum lumped here for a trivial diagonal
inverse, the standard choice for explicit dynamics).

Diagnostic: deviation from the rigid-translation reference (u(t) - v0*t,
which should stay exactly 0 for a consistent law) and total kinetic
energy (should stay bounded; As'ad's own finding is that it grows
unboundedly for the inconsistent regression ANN)."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.chdir(str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.insert(0, KRATOS_PATH)

import numpy as np

from run_cruciform_fe2_claude import MATERIAL_FUNCS  # noqa: E402
import fom_solver_rve as fom  # noqa: E402
from build_square_panel_mesh_claude import build_square_panel_mesh  # noqa: E402
import KratosMultiphysics as KM  # noqa: E402

N_BODY = 8
L_BODY = 12.0
DENSITY = 2500.0
V0 = np.array([1.0, 0.0])  # uniform initial velocity, m/s
DT = 5.0e-6
N_STEPS = 3000  # short stability check first (T=0.015s); extend once dt is confirmed stable


def build_free_model_part():
    coords, tris, edge_nodes, center_node = build_square_panel_mesh(n_body=N_BODY, L_body=L_BODY)
    model = KM.Model()
    mp = model.CreateModelPart("Structure")
    mp.SetBufferSize(1)
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    for i, (x, y) in enumerate(coords):
        mp.CreateNewNode(i + 1, float(x), float(y), 0.0)
    prop = mp.GetProperties()[1]
    prop.SetValue(KM.YOUNG_MODULUS, 1.0)
    prop.SetValue(KM.POISSON_RATIO, 0.3)
    prop.SetValue(KM.THICKNESS, 1.0)
    prop.SetValue(KM.DENSITY, DENSITY)
    for e, conn in enumerate(tris):
        node_ids = [int(c) + 1 for c in conn]
        mp.CreateNewElement("TotalLagrangianElement2D6N", e + 1, node_ids, prop)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, mp)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, mp)
    return mp, coords, tris  # NO Fix() calls anywhere -- fully free-floating


def lumped_mass_vector(mp, n_dof, eq_map):
    """HRZ (Hinton-Rock-Zienkiewicz) lumping, not naive row-sum: for
    quadratic (Tri6) elements, row-sum lumping is a known-unsafe scheme --
    off-diagonal terms in the consistent mass matrix can be large enough
    and negative enough that some row sums come out non-positive (hit
    exactly this failure empirically here). HRZ instead rescales the
    (guaranteed-positive) diagonal terms by a single per-element factor so
    they sum to the element's own true total mass (itself obtained from
    the SAME consistent matrix via partition-of-unity: summing the full
    x-x block equals rho*Volume exactly, no separate geometry query
    needed) -- always positive by construction."""
    m = np.zeros(n_dof)
    pi = mp.ProcessInfo
    M_local = KM.Matrix()
    for elem in mp.Elements:
        elem.CalculateMassMatrix(M_local, pi)
        n_local = M_local.Size1()
        x_idx = list(range(0, n_local, 2))
        total_element_mass = sum(M_local[i, j] for i in x_idx for j in x_idx)
        diag_x_sum = sum(M_local[i, i] for i in x_idx)
        scale = total_element_mass / diag_x_sum
        diag = np.array([M_local[i, i] for i in range(n_local)]) * scale

        node_ids = [node.Id - 1 for node in elem.GetGeometry()]
        for local_i, nid in enumerate(node_ids):
            m[eq_map[nid, 0]] += diag[2 * local_i]
            m[eq_map[nid, 1]] += diag[2 * local_i + 1]
    return m


def run_dynamic_test(which, verbose_every=20):
    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")

    mp, coords, tris = build_free_model_part()
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    n_nodes = coords.shape[0]

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"DynamicConsistency[{which}]")

    m_vec = lumped_mass_vector(mp, n_dof, eq_map)
    assert np.all(m_vec > 0), "expected strictly positive lumped mass at every DOF"
    m_inv = 1.0 / m_vec

    v0_eq = np.zeros(n_dof)
    v0_eq[eq_map[:, 0]] = V0[0]
    v0_eq[eq_map[:, 1]] = V0[1]

    u_curr = np.zeros(n_dof)
    _, f_int0 = assembler.Assemble(u_curr)
    a0 = -m_inv * f_int0
    u_prev = u_curr - DT * v0_eq + 0.5 * DT ** 2 * a0

    u_rigid = np.zeros(n_dof)  # v0*t, the exact rigid-translation reference
    max_deviation_history = []
    ke_history = []
    t0 = time.perf_counter()
    blew_up_at = None

    for n in range(N_STEPS):
        t = n * DT
        u_rigid[:] = v0_eq * t
        deviation = u_curr - u_rigid
        max_dev = float(np.max(np.abs(deviation)))
        max_deviation_history.append(max_dev)

        if not np.all(np.isfinite(u_curr)) or max_dev > 1.0e6:
            blew_up_at = n
            break

        _, f_int = assembler.Assemble(u_curr)
        if not np.all(np.isfinite(f_int)):
            blew_up_at = n
            break
        a_curr = -m_inv * f_int
        u_next = 2.0 * u_curr - u_prev + DT ** 2 * a_curr

        v_mid = (u_next - u_prev) / (2.0 * DT)
        ke = float(0.5 * np.sum(m_vec * v_mid ** 2))
        ke_history.append(ke)

        if n % verbose_every == 0:
            print(f"  [{which}] step {n:4d}  t={t:.4f}s  max|dev from rigid|={max_dev:.4e}  KE={ke:.4e}", flush=True)

        u_prev, u_curr = u_curr, u_next

    wall = time.perf_counter() - t0
    status = f"BLEW UP at step {blew_up_at}" if blew_up_at is not None else "completed all steps"
    print(f"[{which}] {status}, wall={wall:.1f}s, final max|dev|={max_deviation_history[-1]:.4e}, "
          f"peak max|dev|={max(max_deviation_history):.4e}, "
          f"KE: initial={ke_history[0]:.4e} final={ke_history[-1] if ke_history else float('nan'):.4e}")
    return {"which": which, "max_deviation_history": max_deviation_history, "ke_history": ke_history,
            "blew_up_at": blew_up_at}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--which", type=str, default="pann_certified")
    a = p.parse_args()
    run_dynamic_test(a.which)
