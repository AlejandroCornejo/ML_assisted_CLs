#!/usr/bin/env python3
"""Stage 00B: nonlinear FOM solve of the cell under PERIODIC boundary
conditions, plus the homogenized stress and its macro tangent.

Construction, and why it is built this way. Kratos's own validated element and
material do all the physics: VectorizedAssembler.Assemble(u) returns (K, R)
with R = -f_int, using the true Neo-Hookean law, total Lagrangian, and the
[E11, E22, 2*E12] Voigt convention. Only the CONSTRAINT handling is new here.
The project's own path partitions dofs into prescribed/free and sets the
prescribed ones from the affine formula, which is valid for KUBC and false for
periodicity, where only the JUMP across a face pair is known.

So the periodic constraint enters as an elimination:

    u = T @ u_ind + g(E)

with T mapping independent dofs (interior plus one face of each pair) to all
dofs, and g(E) carrying the periodic jumps eps.(X+ - X-) plus one pinned node
to remove the free translation. Newton then runs in the reduced space:

    (T^T K T) du_ind = T^T R

Note this is the SAME decomposition the reduced-order machinery already uses,
u = u_affine(E) + Phi q -- the textbook periodic split. KUBC is the special
case where T is a plain selection onto interior dofs and the fluctuation is
forced to vanish on the boundary. Nothing about the ROM ansatz changes; a POD
basis built from periodic snapshots spans periodic fluctuations automatically,
since periodicity is a linear constraint.

Gate 3 (the reason this file can be trusted): the tangent of this nonlinear
solve at small strain must reproduce the LINEAR periodic C0 from
periodic_C0.py, which is itself already verified two independent ways -- its
affine mode against Kratos's analytic tangent (1e-4), and its periodic mode
against the affine 1/n extrapolation (2e-4).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Same Kratos location the project's own laws use; kept as one line rather
# than duplicated logic, and appended so a system install still wins.
_kratos_candidates = (
    Path("/home/sares/Kratos_Eigen_Check/bin/Release"),
    Path("/home/kratos/Kratos_Eigen_Check/bin/Release"),
)
KRATOS_PATH = next((p for p in _kratos_candidates if p.is_dir()), _kratos_candidates[0])
if str(KRATOS_PATH) not in sys.path:
    sys.path.append(str(KRATOS_PATH))

import config as cfg  # noqa: E402
from gen_rve_mesh import build_mesh, write_mdpa  # noqa: E402

NEWTON_TOL = 1.0e-10      # on ||du_ind|| relative to ||u_ind||
NEWTON_MAX_IT = 30
SUBSTEPS_PER_UNIT_STRAIN = 200.0   # matches the project's own ramp density


class PeriodicRVE:
    """Nonlinear periodic homogenization of one cell, reusing Kratos assembly."""

    def __init__(self, mesh_base, cell_area, verbose=False):
        import KratosMultiphysics as KM  # noqa: F401
        import fom_solver_rve as fom
        from fom_nested_consistent_law_claude import make_parameters

        self._fom = fom
        parameters = make_parameters(mesh_base=str(mesh_base))
        model = KM.Model()
        sim = fom.RVEHomogenizationDatasetGenerator(model, parameters)
        sim.Initialize()
        mp = sim._GetSolver().GetComputingModelPart()
        self._sim = sim
        self._mp = mp

        n_dof, eq_map, _ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)
        self.n_dof = int(n_dof)
        # Kept so a hyperreduced assembler can be built on the same model part
        # with a weighted element subset (VectorizedAssembler needs eq_map).
        self._eq_map = eq_map
        self.assembler = fom.VectorizedAssembler(mp, n_dof, eq_map)
        self.A0 = float(cell_area)
        # w_detJ already carries the element THICKNESS (sum(w_detJ) came out
        # as 0.16 = 0.05 * 3.2, thickness times solid area), so the
        # homogenization denominator is thickness * A0 -- the same product the
        # project's own laws use as `thickness * hom_reference_measure`.
        th = np.unique(np.asarray(self.assembler.thickness, dtype=float))
        if th.size != 1:
            raise RuntimeError(f"non-uniform thickness {th}: denominator ambiguous")
        self.thickness = float(th[0])
        self.denom = self.thickness * self.A0

        # Reference coordinates, indexed the same way eq_map is.
        xy = np.array([[n.X0, n.Y0] for n in mp.Nodes], dtype=float)
        self.dof_xy = np.zeros((self.n_dof, 2))
        self.dof_comp = np.zeros(self.n_dof, dtype=np.int64)
        self.dof_xy[eq_map[:, 0]] = xy
        self.dof_xy[eq_map[:, 1]] = xy
        self.dof_comp[eq_map[:, 0]] = 0
        self.dof_comp[eq_map[:, 1]] = 1

        self._build_periodicity(xy, eq_map, verbose=verbose)

    # -- constraint construction -------------------------------------------
    def _build_periodicity(self, xy, eq_map, tol=None, verbose=False):
        half = 0.5 * np.sqrt(self.A0)
        tol = 1.0e-7 * max(1.0, 2.0 * half) if tol is None else tol
        x, y = xy[:, 0], xy[:, 1]

        root = np.arange(self.n_dof, dtype=np.int64)
        off = np.zeros((self.n_dof, 2))
        for (sel_s, sel_m, key) in ((np.abs(x - half) < tol, np.abs(x + half) < tol, y),
                                    (np.abs(y - half) < tol, np.abs(y + half) < tol, x)):
            si = np.where(sel_s)[0]
            mi = np.where(sel_m)[0]
            if si.size != mi.size:
                raise RuntimeError(f"face node counts differ ({si.size} vs {mi.size}): "
                                   "mesh is not periodic")
            si = si[np.argsort(key[si])]
            mi = mi[np.argsort(key[mi])]
            worst = float(np.max(np.abs(key[si] - key[mi])))
            if worst > tol:
                raise RuntimeError(f"opposite-face nodes mismatch by {worst:.3e}: "
                                   "mesh is not periodic")
            for (s, m) in zip(si.tolist(), mi.tolist()):
                for c in (0, 1):
                    sd, md = int(eq_map[s, c]), int(eq_map[m, c])
                    if root[sd] != sd:
                        continue          # already tied; the chain resolves it
                    root[sd] = md
                    off[sd] = xy[s] - xy[m]

        # Corners are slaves in both directions, so chains must be followed.
        final_root = np.arange(self.n_dof, dtype=np.int64)
        final_off = np.zeros((self.n_dof, 2))
        for d in range(self.n_dof):
            r, acc, guard = d, np.zeros(2), 0
            while root[r] != r:
                acc = acc + off[r]
                r = root[r]
                guard += 1
                if guard > 8:
                    raise RuntimeError("periodicity chain does not terminate")
            final_root[d] = r
            final_off[d] = acc

        # Pin the corner that masters both directions: kills exactly the free
        # translation. Rotation is already excluded, since a symmetric macro
        # strain gives a symmetric average displacement gradient.
        pin_node = int(np.argmin((x + half) ** 2 + (y + half) ** 2))
        pinned = np.zeros(self.n_dof, dtype=bool)
        pinned[int(eq_map[pin_node, 0])] = True
        pinned[int(eq_map[pin_node, 1])] = True

        is_ind = (final_root == np.arange(self.n_dof)) & (~pinned)
        red = -np.ones(self.n_dof, dtype=np.int64)
        red[is_ind] = np.arange(int(np.sum(is_ind)))
        rows = [d for d in range(self.n_dof) if not pinned[final_root[d]]]
        cols = [red[final_root[d]] for d in rows]
        self.T = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(self.n_dof, int(np.sum(is_ind)))).tocsr()
        self.final_root = final_root
        self.final_off = final_off
        self.pinned = pinned
        self.n_ind = int(np.sum(is_ind))

        # Independent-dof index -> (reference position, component). Needed to
        # subtract the affine field E.X from a snapshot, which is what the POD
        # convention that compresses the FLUCTUATION requires.
        ind_dofs = np.where(is_ind)[0]
        order = np.argsort(red[ind_dofs])
        ind_dofs = ind_dofs[order]
        self.ind_dof = ind_dofs
        self.ind_xy = self.dof_xy[ind_dofs]
        self.ind_comp = self.dof_comp[ind_dofs]
        self.n_slave = int(np.sum(final_root != np.arange(self.n_dof)))
        if verbose:
            print(f"[periodic] {self.n_dof} dofs -> {self.n_ind} independent, "
                  f"{self.n_slave} tied, 2 pinned")

    def _g(self, E):
        """Inhomogeneous part: periodic jumps plus the pinned node's own
        affine value.

        The jump is u+ - u- = (F - I).(X+ - X-) with F the DEFORMATION
        GRADIENT, not the Green-Lagrange strain. An earlier version of this
        method used E directly as if it were the displacement gradient. Since
        E = (F^T F - I)/2, the two agree only to first order, so the error is
        O(E^2): at E = 0.15, sqrt(1+2E) - 1 = 0.1402 against E = 0.15, i.e.
        7%, which showed up as an 18.6% violation of S = dW/dE and a 9.5%
        asymmetry in dS/dE -- with the wrong kinematics the discrete problem
        is no longer the gradient of a potential in E, so its tangent need not
        be symmetric.

        Worth recording WHY the three verification gates missed it: all of
        them were at small strain or against small-strain references (gate 3
        compared the tangent at E = 1e-5 to the linear C0), and at E -> 0 the
        bug vanishes identically. Only a finite-strain identity could catch
        it, which is what the stage-03 acceptance test S = dW/dE is.

        Uses the project's own DeformationGradientFromGreenLagrange2D, the
        same routine the validated affine-Dirichlet path uses, rather than a
        second implementation.
        """
        from fom_solver_rve import DeformationGradientFromGreenLagrange2D
        H = DeformationGradientFromGreenLagrange2D(np.asarray(E, dtype=float)) \
            - np.eye(2)
        g = np.einsum("ij,dj->di", H, self.final_off)[
            np.arange(self.n_dof), self.dof_comp]
        pin = self.pinned[self.final_root]
        if np.any(pin):
            root_xy = self.dof_xy[self.final_root]
            g = g + np.where(pin,
                             np.einsum("ij,dj->di", H, root_xy)[
                                 np.arange(self.n_dof), self.dof_comp],
                             0.0)
        return g

    def _g_at(self, E, sel):
        """`_g(E)[sel]`, computed only at the requested dofs.

        PURELY ADDITIVE: `_g` above is untouched, since the FOM and the PROM
        need the whole field. This exists because the hyperreduced models need
        the affine part at ~120 dofs of 6320 and were computing all of them and
        discarding 98%. Profiled, that waste was 215.9 us of the 684 us per
        state in the D-HPROM-ANN (31.6%), and ~31 ms of the 78.7 ms in the
        MAW-HPROM-ANN, where `_g` is called once per load substep.

        Verified against `_g(E)[sel]` rather than assumed equal to it.
        """
        from fom_solver_rve import DeformationGradientFromGreenLagrange2D
        H = DeformationGradientFromGreenLagrange2D(np.asarray(E, dtype=float)) \
            - np.eye(2)
        sel = np.asarray(sel, dtype=np.int64)
        k = np.arange(sel.size)
        comp = self.dof_comp[sel]
        g = np.einsum("ij,dj->di", H, self.final_off[sel])[k, comp]
        root = self.final_root[sel]
        pin = self.pinned[root]
        if np.any(pin):
            g = g + np.where(
                pin,
                np.einsum("ij,dj->di", H, self.dof_xy[root])[k, comp],
                0.0)
        return g

    # -- solve --------------------------------------------------------------
    def solve(self, E, u_ind_init=None, E_start=None, verbose=False):
        """(S (3,), u_ind) at macro strain E. Ramped Newton in the reduced
        space, warm-startable exactly as the affine path is -- the material is
        hyperelastic with no history, so the converged state is a pure
        function of E and the ramp is only a Newton aid."""
        E = np.asarray(E, dtype=float).reshape(3)
        E0 = np.zeros(3) if E_start is None else np.asarray(E_start, float).reshape(3)
        n_sub = max(1, int(np.ceil(SUBSTEPS_PER_UNIT_STRAIN * np.linalg.norm(E - E0))))
        u_ind = np.zeros(self.T.shape[1]) if u_ind_init is None else u_ind_init.copy()

        for k in range(1, n_sub + 1):
            Et = E0 + (E - E0) * (k / n_sub)
            g = self._g(Et)
            for it in range(NEWTON_MAX_IT):
                u = self.T @ u_ind + g
                K, R = self.assembler.Assemble(u)
                r_red = self.T.T @ R
                K_red = (self.T.T @ K @ self.T).tocsc()
                du = spla.spsolve(K_red, r_red)
                u_ind = u_ind + du
                nrm = np.linalg.norm(du) / max(np.linalg.norm(u_ind), 1.0e-30)
                if nrm < NEWTON_TOL:
                    break
            else:
                raise RuntimeError(f"periodic Newton failed at substep {k}/{n_sub}, "
                                   f"E={Et}, last rel du {nrm:.3e}")
            if verbose:
                print(f"   substep {k}/{n_sub}: {it + 1} Newton its, rel du {nrm:.2e}")

        u = self.T @ u_ind + self._g(E)
        self.assembler.Assemble(u)
        return self.homogenized_stress(E), u_ind

    def homogenized_stress(self, E, assembler=None):
        """Macro 2nd PK stress from the assembler's current converged state.

        THE AVERAGE MUST BE TAKEN ON THE FIRST PIOLA-KIRCHHOFF STRESS:

            Pbar = <P>,      Sbar = Fbar^-1 Pbar

        It is <P> that homogenizes exactly, P being work-conjugate to F. Since
        S = F^-1 P pointwise, <F^-1 P> is NOT Fbar^-1 <P> unless F is uniform,
        so averaging the microscopic 2nd PK stress directly is wrong by
        O(fluctuation^2).

        Measured, at h-independent finite differences of the energy:

            state                    naive <S>     Fbar^-1 <P>
            [0.05, -0.090,  0.10]    5.286e-03     7.08e-07
            [0.19, -0.090, -0.16]    7.045e-03     5.48e-07
            [0.15, -0.065, -0.048]   1.540e-03     4.82e-07

        as the relative violation of S = dW/dE. Four orders of magnitude, and
        Sbar comes out SYMMETRIC to 1e-16 on its own rather than by
        construction, which is the signature that the formulation is right.

        The error is O(fluctuation^2), hence invisible at small strain -- which
        is how it survived gates 1 to 3, all of which sat at or near E = 0.
        Hernandez reports Pbar rather than Sbar for exactly this reason.

        `assembler` lets the average be taken on a HYPERREDUCED assembler --
        a weighted element subset whose ECM rule was fitted for THIS integrand.
        It must not be the residual rule's assembler: those weights reproduce
        the projected residual, a different integral, and using them here is
        exactly the previous project's S == 0 bug.

        Note this project's inherited naive-average helper
        (CalculateHomogenizedStressAndStrain) averages the 2nd PK stress and
        so carries the same bias; its reaction-force path
        (S = -(sens @ R)/denom) does not, being an envelope-theorem
        derivative.
        """
        a = self.assembler if assembler is None else assembler
        Sv = a._S_voigt
        St = np.zeros(Sv.shape[:2] + (2, 2))
        St[..., 0, 0] = Sv[..., 0]
        St[..., 1, 1] = Sv[..., 1]
        St[..., 0, 1] = Sv[..., 2]
        St[..., 1, 0] = Sv[..., 2]
        P_bar = np.einsum("eg,egij->ij", a.w_detJ, np.matmul(a._F, St)) / self.denom
        F_bar = self._fom.DeformationGradientFromGreenLagrange2D(
            np.asarray(E, dtype=float))
        S_bar = np.linalg.solve(F_bar, P_bar)
        # Symmetric to roundoff by construction of the formulation; the
        # symmetrization below only removes that roundoff.
        return np.array([S_bar[0, 0], S_bar[1, 1],
                         0.5 * (S_bar[0, 1] + S_bar[1, 0])])

    def homogenized_energy(self, E=None):
        """Macro strain energy density, integrated with the SAME quadrature
        weights the stress uses. The project's own helper weights Gauss points
        by `mean(psi) * area_e`, which differs from `sum(psi * w_detJ)` on the
        isoparametrically CURVED rim elements where detJ is not constant --
        measured at 1.3e-03 relative. Small, but there is no reason to carry
        an inconsistency between the two outputs whose identity is being
        tested."""
        a = self.assembler
        c11 = 1.0 + 2.0 * a._E_voigt[..., 0]
        c22 = 1.0 + 2.0 * a._E_voigt[..., 1]
        c12 = a._E_voigt[..., 2]
        det_c = c11 * c22 - c12 * c12
        if np.min(det_c) <= 0.0:
            raise RuntimeError("det(C) <= 0 at a Gauss point")
        log_j = 0.5 * np.log(det_c)
        young = np.asarray(a.young, dtype=float)[:, None]
        poisson = np.asarray(a.poisson, dtype=float)[:, None]
        mu = young / (2.0 * (1.0 + poisson))
        lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        psi = 0.5 * mu * (c11 + c22 - 2.0) - mu * log_j + 0.5 * lam * log_j ** 2
        return float(np.sum(psi * a.w_detJ) / self.denom)

    def _periodic_lifting_jacobian(self, E):
        """Return ``g_E = d g / dE`` for the periodic lifting.

        The periodic constraint is ``u = T q + g(E)``.  Unlike an affine
        Dirichlet lifting, ``g`` is built with ``F(E)-I`` because the input is
        Green--Lagrange strain.  Its first derivative is therefore *not* a
        constant affine matrix.  This small helper keeps that derivative in
        the same place as the constraint construction, so the FE2 tangent can
        use the implicit-function theorem without finite-differencing six
        additional RVE solves per macro Gauss point.
        """
        from deformation_gradient_jacobian_claude import deformation_gradient_and_jacobian_2d

        _F, dF_dE = deformation_gradient_and_jacobian_2d(
            np.asarray(E, dtype=float).reshape(3))

        # ``final_off`` carries the periodic jump.  A dof whose ultimate
        # master is pinned also carries the affine value of that master; this
        # is exactly the second term in _g(), written once as an arm vector.
        arm = self.final_off.copy()
        root = self.final_root
        pin_root = self.pinned[root]
        arm[pin_root] += self.dof_xy[root[pin_root]]

        out = np.empty((self.n_dof, 3), dtype=float)
        rows = np.arange(self.n_dof)
        for k in range(3):
            disp = np.einsum("ij,dj->di", dF_dE[:, :, k], arm)
            out[:, k] = disp[rows, self.dof_comp]
        return out

    def stress_and_tangent_consistent(self, E, u_ind_init=None, E_start=None,
                                      heps=1.0e-6, check_stress=True,
                                      return_state=False):
        """Homogenized ``(S, dS/dE)`` using one nonlinear periodic solve.

        This is the periodic counterpart of the reaction-force/implicit-
        function tangent used by the established FE2 drivers.  Let

        ``u(E) = T q(E) + g(E)`` and ``f_int(u)`` be the assembled internal
        force.  At equilibrium ``T.T @ f_int = 0``.  Hence

        ``dq/dE = -(T.T K T)^-1 T.T K dg/dE``.

        The envelope theorem gives ``S = (dg/dE).T f_int / denom``.  Its
        derivative needs only the already assembled microscopic stiffness and
        the curvature of the closed-form 2x2 map ``F(E)``.  The latter is
        finite-differenced here *only at the lifting level*; no extra RVE
        solves are performed.  This reduces a macro material call from seven
        nonlinear RVE solves to one while retaining a consistent tangent.
        """
        E = np.asarray(E, dtype=float).reshape(3)
        _S_average, q = self.solve(E, u_ind_init=u_ind_init, E_start=E_start)

        # ``solve`` leaves the assembler at the converged state.  Assemble
        # once explicitly so K/R and the stress arrays are tied to the same
        # state even if this method is called after a future refactor.
        u = self.T @ q + self._g(E)
        K, rhs = self.assembler.Assemble(u)       # rhs == -f_int
        f_int = -np.asarray(rhs, dtype=float)

        g_E = self._periodic_lifting_jacobian(E)  # (n_dof, 3)
        K_red = (self.T.T @ K @ self.T).tocsc()
        rhs_q = -(self.T.T @ (K @ g_E))
        dq_dE = np.column_stack([
            spla.spsolve(K_red, rhs_q[:, k]) for k in range(3)
        ])
        du_dE = g_E + self.T @ dq_dE

        S = (g_E.T @ f_int) / self.denom

        # d²g/dE_jdE_k.  This is cheap algebra on the imposed macro map;
        # perturbing it must never be confused with perturbing the RVE solve.
        d2g = np.empty((self.n_dof, 3, 3), dtype=float)
        for k in range(3):
            h = heps * max(1.0, abs(E[k]))
            Ep, Em = E.copy(), E.copy()
            Ep[k] += h
            Em[k] -= h
            d2g[:, :, k] = (
                self._periodic_lifting_jacobian(Ep)
                - self._periodic_lifting_jacobian(Em)
            ) / (2.0 * h)

        CC = (np.einsum("djk,d->jk", d2g, f_int)
              + g_E.T @ (K @ du_dE)) / self.denom
        # Roundoff in the lifting's numerical second derivative can leave a
        # tiny antisymmetric component.  A hyperelastic macro tangent is
        # symmetric, so remove only that roundoff-level part.
        CC = 0.5 * (CC + CC.T)

        if check_stress:
            S_average = self.homogenized_stress(E)
            rel = np.linalg.norm(S - S_average) / max(np.linalg.norm(S_average), 1.0)
            if rel > 2.0e-7:
                raise RuntimeError(
                    "periodic reaction-force stress does not match Fbar^-1<Pbar>: "
                    f"relative mismatch {rel:.3e}")
        # ``q`` is an algorithmic Newton aid, not an internal material
        # variable: this hyperelastic RVE has no history.  Exposing it on
        # request lets a nested FE2 caller continue from its last converged
        # microscopic state without changing the FOM problem or its answer.
        return (S, CC, q) if return_state else (S, CC)

    def stress_and_tangent(self, E, h=1.0e-6, verbose=False):
        """(S, dS/dE) with a central finite difference on E. Each perturbed
        solve is warm-started from the base state, so the six extra solves are
        cheap: the increment is O(h) and needs one substep."""
        E = np.asarray(E, dtype=float).reshape(3)
        S0, u0 = self.solve(E, verbose=verbose)
        C = np.zeros((3, 3))
        for j in range(3):
            step = h * max(1.0, abs(E[j]))
            Ep, Em = E.copy(), E.copy()
            Ep[j] += step
            Em[j] -= step
            Sp, _ = self.solve(Ep, u_ind_init=u0, E_start=E)
            Sm, _ = self.solve(Em, u_ind_init=u0, E_start=E)
            C[:, j] = (Sp - Sm) / (2.0 * step)
        return S0, C


def _build(size_far=0.30, size_hole=0.14, tag="periodic"):
    coords, tris, outer, geom = build_mesh(size_far=size_far, size_hole=size_hole,
                                           periodic=True)
    base = HERE / f"mesh_{tag}"
    write_mdpa(str(base) + ".mdpa", coords, tris, outer)
    return str(base), geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-far", type=float, default=0.30)
    ap.add_argument("--size-hole", type=float, default=0.14)
    ap.add_argument("--eps", type=float, default=1.0e-5,
                    help="small strain at which the tangent must reproduce C0")
    a = ap.parse_args()

    ref = np.load(HERE / "C0_periodic.npz")["C0_periodic"]
    base, geom = _build(a.size_far, a.size_hole)
    print(f"periodic mesh: {geom['n_nodes']} nodes, {geom['n_elements']} elements, "
          f"porosity {geom['porosity_mesh'] * 100:.3f}%")

    from _material_law_guard_claude import true_neo_hookean_active
    with true_neo_hookean_active():
        rve = PeriodicRVE(base, cell_area=geom["block_area"], verbose=True)

        print("\ngate 3: nonlinear periodic tangent at small strain vs linear periodic C0")
        t0 = time.perf_counter()
        S, C = rve.stress_and_tangent(np.full(3, a.eps))
        dt = time.perf_counter() - t0
        rel = np.max(np.abs(C - ref)) / np.max(np.abs(ref))
        with np.printoptions(precision=4, linewidth=120):
            print(" tangent (Pa) =\n", C)
            print(" linear C0    =\n", ref)
        print(f" max rel difference {rel:.3e}   ({dt:.1f}s for 7 solves)")
        gate3 = rel < 1.0e-3
        print(f"gate 3  {'PASS' if gate3 else 'FAIL'}")

        print("\nfinite strain: uniaxial-ish states, checking the solve holds up")
        for e11 in (0.02, 0.05, 0.10, 0.20):
            E = np.array([e11, -0.4718 * e11, 0.0])
            t0 = time.perf_counter()
            Sf, uf = rve.solve(E)
            print(f"  E11={e11:5.2f}  S=[{Sf[0]:+.4e} {Sf[1]:+.4e} {Sf[2]:+.4e}]  "
                  f"{time.perf_counter() - t0:5.1f}s")

    print("PERIODIC_FOM_PASS" if gate3 else "PERIODIC_FOM_FAIL")
    return 0 if gate3 else 1


if __name__ == "__main__":
    sys.exit(main())
