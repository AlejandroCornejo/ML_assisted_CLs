#!/usr/bin/env python3
"""Stage 01: the cheap macro pre-pass on the ASTM coupon.

Purpose: measure the strain envelope the coupon actually visits, BEFORE any
expensive RVE data exists. This is the step whose absence caused the previous
project's inverted design, where the sampling box was chosen from what the RVE
could survive and a macro problem was then hunted to fit inside it.

Constitutive model: St Venant-Kirchhoff with the measured periodic C0,
S = C0 : E with E the Green-Lagrange strain. Exact at small strain, and it
carries the geometric nonlinearity exactly, at the cost of one matrix-vector
product per Gauss point.

DISPLACEMENT control, deliberately, and this is not a detail. Measured against
the true RVE uniaxial path, SVK-C0 over-predicts S11 by +19.7% at E11 = 0.10,
+40.8% at 0.20 and +109.6% at 0.50, because the real perforated cell softens
as its ligaments reorient. Inverted for FORCE control that becomes a 28% / 76%
/ 236% UNDER-prediction of strain, which would undersize the envelope by a
factor of up to 3.4. What SVK-C0 does get right is the kinematic ratios --
gamma12/E11 to within 2.5% and E22/E11 to within 6-31% -- so under
displacement control, where the strain field is set kinematically, the strain
cloud is reliable and a too-stiff material merely reports higher forces.

The FE^2 runs stay force-controlled; the force is calibrated later against the
real model, and the envelope measured here already covers it.

Implementation reuses the project's validated VectorizedAssembler by replacing
its constitutive function, so the element, the total-Lagrangian kinematics and
the tangent assembly are all the already-verified ones. The only new code is
the Newton loop and the Dirichlet handling.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(HERE), str(ROOT / "00_rve"),
          str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import config as cfg  # noqa: E402
from gen_coupon_mesh import build_mesh  # noqa: E402

NEWTON_TOL = 1.0e-10
NEWTON_MAX_IT = 300   # modified Newton on the softening probe converges linearly


def load_C0():
    d = np.load(ROOT / "00_rve" / "C0_periodic.npz")
    return np.asarray(d["C0_periodic"], dtype=float)


@contextlib.contextmanager
def softening_probe(C):
    """SVK-C0 scaled by a strain-dependent factor calibrated to the MEASURED
    true RVE uniaxial response: S = phi(||E||) C E.

    This is a SENSITIVITY PROBE, not a constitutive model -- it has no
    potential and is not claimed to be admissible. Its only job is to move the
    macro strain field by a realistic amount, so the envelope margin can be
    measured rather than guessed.

    It exists because the obvious test is a null test: scaling C by a scalar
    leaves the strain field EXACTLY unchanged, since the equilibrium equations
    are homogeneous in C. What genuinely moves the field under displacement
    control is the material's NONLINEARITY, because a softening material lets
    the more-strained gauge take an even larger share of the elongation.

    phi is taken from the measured curve by interpolation in ||E||_F, with no
    fitted functional form: phi = S11_true(E11) / (Ex * E11) sampled along the
    true uniaxial path.
    """
    import fom_solver_rve as fom
    original = fom._neo_hookean_pk2_2d_vectorized

    d = np.load(ROOT / "00_rve" / "uniaxial_path.npz")
    ok = d["ok"].astype(bool)
    e11, e22, g12, s11 = d["e11"][ok], d["e22"][ok], d["g12"][ok], d["s11"][ok]
    Ex = 1.0 / np.linalg.inv(C)[0, 0]
    phi_tab = s11 / (Ex * e11)
    # ||E||_F with the Voigt third slot being engineering shear (= 2*E12)
    nrm_tab = np.sqrt(e11 ** 2 + e22 ** 2 + 0.5 * g12 ** 2)

    def law(E_flat, young, poisson):
        E_flat = np.asarray(E_flat, dtype=float)
        nrm = np.sqrt(E_flat[:, 0] ** 2 + E_flat[:, 1] ** 2 + 0.5 * E_flat[:, 2] ** 2)
        phi = np.interp(nrm, nrm_tab, phi_tab, left=phi_tab[0], right=phi_tab[-1])
        dphi = np.gradient(phi_tab, nrm_tab)
        dphi_i = np.interp(nrm, nrm_tab, dphi, left=0.0, right=0.0)
        CE = E_flat @ C.T
        S = phi[:, None] * CE
        # MODIFIED Newton: phi*C only, dropping the rank-1 term
        # dphi * (C E) (x) d||E||/dE. That term is the consistent one -- it was
        # verified against finite differences to 1e-11 at small strain -- but it
        # is ASYMMETRIC (7.9e-02 relative at mid strain, 1.7e-01 at high),
        # which is the signature of this probe having no potential, and with a
        # large initial residual it pushed Newton out of its basin. phi*C is
        # symmetric and positive definite, so iteration is robust; it converges
        # linearly rather than quadratically, and Newton's FIXED POINT does not
        # depend on the tangent, so the converged strain field is unchanged.
        del dphi_i
        CC = phi[:, None, None] * np.broadcast_to(C, (E_flat.shape[0], 3, 3))
        return S, CC

    fom._neo_hookean_pk2_2d_vectorized = law
    try:
        yield
    finally:
        fom._neo_hookean_pk2_2d_vectorized = original


@contextlib.contextmanager
def svk_material(C):
    """Replace the assembler's constitutive function with SVK: S = C E,
    tangent C, constant. Restores the original on exit so nothing leaks into
    another user of the module in the same process."""
    import fom_solver_rve as fom
    original = fom._neo_hookean_pk2_2d_vectorized

    def svk(E_flat, young, poisson):
        E_flat = np.asarray(E_flat, dtype=float)
        S = E_flat @ C.T
        CC = np.broadcast_to(C, (E_flat.shape[0], 3, 3))
        return S, CC

    fom._neo_hookean_pk2_2d_vectorized = svk
    try:
        yield
    finally:
        fom._neo_hookean_pk2_2d_vectorized = original


def write_coupon_mdpa(path, coords, tris, dirichlet_nodes):
    """`dirichlet` sub-model-part is required because the shared
    ProjectParameters.json runs an AssignVectorVariableProcess on
    `Structure.dirichlet`. Its VALUES are irrelevant here -- this solver
    imposes its own boundary conditions in the Newton loop -- but the part
    must exist, and providing it is preferable to editing the shared file."""
    lines = ["Begin ModelPartData", "//  VARIABLE_NAME value", "End ModelPartData", "",
             "Begin Properties 0", "End Properties", "Begin Nodes"]
    for i, (x, y) in enumerate(coords, start=1):
        lines.append(f"    {i}  {x:.12f}  {y:.12f}  0.0000000000")
    lines += ["End Nodes", "",
              "Begin Geometries Triangle2D6 // GUI group identifier: material"]
    for e, t in enumerate(tris, start=1):
        lines.append("    " + str(e) + "   " + "  ".join(str(int(v) + 1) for v in t))
    lines += ["End Geometries", "",
              "Begin SubModelPart material // Group material",
              "    Begin SubModelPartNodes"]
    lines += [f"        {i}" for i in range(1, coords.shape[0] + 1)]
    lines += ["    End SubModelPartNodes", "    Begin SubModelPartGeometries"]
    lines += [f"        {e}" for e in range(1, tris.shape[0] + 1)]
    lines += ["    End SubModelPartGeometries", "End SubModelPart",
              "Begin SubModelPart dirichlet // Group dirichlet",
              "    Begin SubModelPartNodes"]
    lines += [f"        {int(i) + 1}" for i in dirichlet_nodes]
    lines += ["    End SubModelPartNodes", "End SubModelPart", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


class MacroCoupon:
    def __init__(self, mesh_base, verbose=False):
        import KratosMultiphysics as KM
        import fom_solver_rve as fom
        from fom_nested_consistent_law_claude import make_parameters

        parameters = make_parameters(mesh_base=str(mesh_base))
        model = KM.Model()
        sim = fom.RVEHomogenizationDatasetGenerator(model, parameters)
        sim.Initialize()
        mp = sim._GetSolver().GetComputingModelPart()
        self._sim, self._mp = sim, mp

        n_dof, eq_map, _ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)
        self.n_dof = int(n_dof)
        self.assembler = fom.VectorizedAssembler(mp, n_dof, eq_map)
        self.xy = np.array([[n.X0, n.Y0] for n in mp.Nodes], dtype=float)
        self.eq_map = np.asarray(eq_map, dtype=np.int64)

        x_end = cfg.COUPON_L_TOTAL / 2.0
        tol = 1.0e-9 * cfg.COUPON_L_TOTAL
        self.left = np.where(np.abs(self.xy[:, 0] + x_end) < tol)[0]
        self.right = np.where(np.abs(self.xy[:, 0] - x_end) < tol)[0]
        if self.left.size == 0 or self.right.size == 0:
            raise RuntimeError("end faces not found")
        if verbose:
            print(f"[macro] {self.n_dof} dofs, {self.left.size}/{self.right.size} "
                  f"end-face nodes")

    def _bc(self, delta):
        """Symmetric axial displacement control: -delta/2 on the left face,
        +delta/2 on the right. Equivalent to fixed-plus-delta up to a rigid
        translation, and symmetric is the honest picture of a two-sided pull.
        y is left FREE on the end faces, so the faces are not clamped and the
        free-edge/clamped-face shear artefact of the previous project does not
        arise. One node's y is pinned to remove the remaining translation."""
        dofs, vals = [], []
        for nodes, u in ((self.left, -0.5 * delta), (self.right, +0.5 * delta)):
            for n in nodes:
                dofs.append(int(self.eq_map[n, 0]))
                vals.append(u)
        n_pin = int(self.left[np.argmin(np.abs(self.xy[self.left, 1]))])
        dofs.append(int(self.eq_map[n_pin, 1]))
        vals.append(0.0)
        return np.asarray(dofs, dtype=np.int64), np.asarray(vals, dtype=float)

    def solve(self, delta, n_steps=20, record=True, verbose=False):
        """Ramp the end displacement to `delta`, recording the Gauss-point
        strain cloud at every step."""
        free = np.ones(self.n_dof, dtype=bool)
        dir_dofs, dir_vals_full = self._bc(delta)
        free[dir_dofs] = False
        free_idx = np.where(free)[0]

        u = np.zeros(self.n_dof)
        cloud = []
        for k in range(1, n_steps + 1):
            frac = k / n_steps
            u[dir_dofs] = dir_vals_full * frac
            for it in range(NEWTON_MAX_IT):
                K, R = self.assembler.Assemble(u)
                Kff = K[free_idx, :][:, free_idx].tocsc()
                du = spla.spsolve(Kff, R[free_idx])
                u[free_idx] += du
                nrm = np.linalg.norm(du) / max(np.linalg.norm(u), 1e-30)
                if nrm < NEWTON_TOL:
                    break
            else:
                raise RuntimeError(f"macro Newton failed at step {k}, "
                                   f"last rel du {nrm:.3e}")
            if record:
                cloud.append(self.assembler._E_voigt.reshape(-1, 3).copy())
            if verbose:
                E = self.assembler._E_voigt.reshape(-1, 3)
                print(f"  step {k:3d}/{n_steps} delta={delta * frac * 1e3:7.3f}mm "
                      f"{it + 1} its  E11[{E[:, 0].min():+.4f},{E[:, 0].max():+.4f}] "
                      f"E22[{E[:, 1].min():+.4f},{E[:, 1].max():+.4f}] "
                      f"g12[{E[:, 2].min():+.4f},{E[:, 2].max():+.4f}]")
        return u, (np.concatenate(cloud, axis=0) if record else None)

    def gauge_mask(self):
        """Gauss points inside the narrow section, by element centroid."""
        conn = np.asarray(self.assembler.connectivity, dtype=np.int64)
        xc = self.xy[conn[:, :3], 0].mean(axis=1)
        inside = np.abs(xc) < cfg.COUPON_L_GAUGE / 2.0
        return np.repeat(inside, self.assembler.n_gauss)

    def gauge_median_e11(self):
        E = self.assembler._E_voigt.reshape(-1, 3)
        return float(np.median(E[self.gauge_mask(), 0]))


def gauge_strain_for(target_e11):
    """First estimate of the end displacement for a target GAUGE strain.

    The naive `delta = (lambda-1) * L_gauge` is wrong by a factor of ~2.4,
    because the imposed displacement is shared over the WHOLE 165 mm specimen,
    not just the 57 mm gauge, and the wider grips strain less. Effective
    length, weighting each region by w_gauge/w(x) as uniaxial equilibrium
    requires:

        gauge  57.0 * 1.000                 = 57.0
        fillet 42.3 * (6.5/7.8 avg)         = 35.2
        grip   65.7 * (6.5/9.5)             = 44.9
                                       total ~ 137 mm

    This is only a starting point; `calibrate_delta` then iterates on the
    measured gauge strain, since the estimate above is itself approximate.
    """
    p = __import__("gen_coupon_mesh").profile()
    l_gauge = cfg.COUPON_L_GAUGE
    l_fillet = 2.0 * p["dx"]
    l_grip = 2.0 * (p["x_end"] - p["x_fillet_end"])
    l_eff = (l_gauge
             + l_fillet * (p["w_gauge"] / (0.5 * (p["w_gauge"] + p["w_grip"])))
             + l_grip * (p["w_gauge"] / p["w_grip"]))
    lam = np.sqrt(1.0 + 2.0 * target_e11)
    return (lam - 1.0) * l_eff


def calibrate_delta(m, target_e11, n_steps, tol=2.0e-3, max_it=6, verbose=True):
    """Secant iteration on the end displacement until the MEDIAN gauge E11
    hits the target. The median rather than the maximum, because the target is
    the gauge's own strain level, not the concentration at the fillet."""
    delta = gauge_strain_for(target_e11)
    prev = None
    for it in range(max_it):
        _u, _c = m.solve(delta, n_steps=n_steps, record=False)
        got = m.gauge_median_e11()
        if verbose:
            print(f"  calib {it}: delta={delta * 1e3:8.4f} mm -> gauge E11 median "
                  f"{got:.5f} (target {target_e11})")
        if abs(got - target_e11) < tol * max(target_e11, 1e-9):
            return delta, got
        if prev is None:
            delta_new = delta * target_e11 / max(got, 1e-12)
        else:
            d0, g0 = prev
            slope = (got - g0) / max(delta - d0, 1e-30)
            delta_new = delta + (target_e11 - got) / slope
        prev = (delta, got)
        delta = float(delta_new)
    return delta, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-e11", type=float, default=0.15)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--material", choices=("svk", "softening"), default="svk",
                    help="svk = S = C0 E; softening = the measured-response "
                         "sensitivity probe")
    ap.add_argument("--size-gauge", type=float, default=None)
    a = ap.parse_args()

    C0 = load_C0()
    coords, tris, left, right, geom = build_mesh(size_gauge=a.size_gauge)
    base = HERE / "coupon_mesh"
    write_coupon_mdpa(str(base) + ".mdpa", coords, tris,
                      np.concatenate((left, right)))
    print(f"macro mesh: {geom['n_nodes']} nodes, {geom['n_elements']} elements "
          f"({geom['n_elements'] * 3} Gauss points)")

    print(f"target gauge E11 {a.target_e11}, material '{a.material}'")
    ctx = svk_material(C0) if a.material == "svk" else softening_probe(C0)
    with ctx:
        m = MacroCoupon(base, verbose=True)
        delta, got = calibrate_delta(m, a.target_e11, a.n_steps)
        print(f"calibrated end displacement {delta * 1e3:.4f} mm "
              f"-> gauge E11 median {got:.5f}")
        u, cloud = m.solve(delta, n_steps=a.n_steps, verbose=False)

    np.savez(HERE / f"prepass_cloud_{a.material}.npz", cloud=cloud,
             delta=delta, target_e11=a.target_e11, C0=C0)
    print(f"\nrecorded cloud: {cloud.shape[0]} (Gauss point, step) samples")
    for j, nm in enumerate(("E11", "E22", "g12")):
        c = cloud[:, j]
        print(f"  {nm}  min {c.min():+.5f}  max {c.max():+.5f}  "
              f"p1 {np.percentile(c, 1):+.5f}  p99 {np.percentile(c, 99):+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
