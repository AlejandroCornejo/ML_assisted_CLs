#!/usr/bin/env python3
"""Build a genuinely REDUCED model part for the HPROM, instead of iterating a
subset of a full one.

WHY. The earlier HPROM kept the full model part and passed
`elements=[subset]` to VectorizedAssembler. That is CORRECT -- only the
selected elements' contributions are assembled, with their ECM weights -- but
it is not cheap: the global sparse system is still built at full size
(6480 x 6480) and the projection passes through full-size objects. Measured
symptom: at 135 elements the HPROM took 0.24 s/state where assembly alone
should account for about 0.15 s, so roughly 40% was avoidable overhead and the
34.4x speedup was a LOWER BOUND on its own implementation.

THE KEY SIMPLIFICATION, which is what makes this tractable: the reduced mesh
does NOT need its own periodic constraint. The HPROM's unknown is q, and the
displacement it implies is

    u = T Phi q + g(E)

on the FULL dof set. The reduced assembly only needs u at the nodes its own
elements touch, so the reduced problem needs a SELECTION OF ROWS of T Phi and
of g -- not a rebuilt constraint. That matters because a reduced mesh
generally has no matching periodic face pairs at all, so rebuilding the
constraint on it would be ill-posed.

Node IDs are preserved from the full mesh, so the reduced-to-full dof map is
by ID rather than by coordinate matching.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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


def read_mdpa(path):
    """(node_id -> (x, y), [(elem_id, [6 node ids])]) from the project's format."""
    nodes, elems, sec = {}, [], None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("Begin Nodes"):
            sec = "n"
            continue
        if s.startswith("End Nodes"):
            sec = None
            continue
        if s.startswith("Begin Geometries"):
            sec = "e"
            continue
        if s.startswith("End Geometries"):
            sec = None
            continue
        if sec == "n":
            p = s.split()
            if len(p) >= 4:
                nodes[int(p[0])] = (float(p[1]), float(p[2]))
        elif sec == "e":
            p = s.split()
            if len(p) == 7:
                elems.append((int(p[0]), [int(v) for v in p[1:7]]))
    return nodes, elems


def write_reduced_mdpa(out_path, nodes, elems, keep_elem_idx):
    """Reduced mdpa with ORIGINAL node ids preserved, so the reduced-to-full
    dof map is by id. Element ids are renumbered from 1, since only their
    order matters for the ECM weight alignment."""
    kept = [elems[int(i)] for i in keep_elem_idx]
    node_ids = sorted({n for _e, conn in kept for n in conn})
    lines = ["Begin ModelPartData", "//  VARIABLE_NAME value", "End ModelPartData", "",
             "Begin Properties 0", "End Properties", "Begin Nodes"]
    for nid in node_ids:
        x, y = nodes[nid]
        lines.append(f"    {nid}  {x:.12f}  {y:.12f}  0.0000000000")
    lines += ["End Nodes", "",
              "Begin Geometries Triangle2D6 // GUI group identifier: material"]
    for k, (_eid, conn) in enumerate(kept, start=1):
        lines.append("    " + str(k) + "   " + "  ".join(str(n) for n in conn))
    lines += ["End Geometries", "",
              "Begin SubModelPart material // Group material",
              "    Begin SubModelPartNodes"]
    lines += [f"        {n}" for n in node_ids]
    lines += ["    End SubModelPartNodes", "    Begin SubModelPartGeometries"]
    lines += [f"        {k}" for k in range(1, len(kept) + 1)]
    lines += ["    End SubModelPartGeometries", "End SubModelPart",
              "Begin SubModelPart dirichlet // Group dirichlet",
              "    Begin SubModelPartNodes"]
    # Non-empty because the shared ProjectParameters runs a process on it; its
    # values are irrelevant, this solver imposes its own state.
    lines += [f"        {node_ids[0]}"]
    lines += ["    End SubModelPartNodes", "End SubModelPart", ""]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return len(node_ids), len(kept)


class ReducedAssembly:
    """Assembler on a reduced mesh, plus the row selection that connects it to
    the full-mesh reduced basis."""

    def __init__(self, full_mdpa, out_base, keep_elem_idx, weights,
                 full_rve, TPhi_full):
        import KratosMultiphysics as KM
        import fom_solver_rve as fom
        from fom_nested_consistent_law_claude import make_parameters

        nodes, elems = read_mdpa(full_mdpa)
        n_nod, n_el = write_reduced_mdpa(str(out_base) + ".mdpa", nodes, elems,
                                         keep_elem_idx)
        self.n_nodes, self.n_elements = n_nod, n_el

        model = KM.Model()
        sim = fom.RVEHomogenizationDatasetGenerator(
            model, make_parameters(mesh_base=str(out_base)))
        sim.Initialize()
        mp = sim._GetSolver().GetComputingModelPart()
        self._sim = sim
        n_dof, eq_map, _ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)
        self.n_dof = int(n_dof)
        self.asm = fom.VectorizedAssembler(
            mp, n_dof, eq_map, element_scales=np.asarray(weights, dtype=float),
            log_label="ReducedAssembler")

        # Reduced dof -> full dof, by NODE ID.
        full_id_to_row = {int(nd.Id): i for i, nd in enumerate(full_rve._mp.Nodes)}
        eqf = np.asarray(full_rve._eq_map, dtype=np.int64)
        sel = np.empty(self.n_dof, dtype=np.int64)
        for i, nd in enumerate(mp.Nodes):
            fr = full_id_to_row[int(nd.Id)]
            for c in (0, 1):
                sel[int(eq_map[i, c])] = int(eqf[fr, c])
        self.sel = sel
        self.TPhi = np.ascontiguousarray(TPhi_full[sel])   # (n_dof_red, r)

    def u_reduced(self, q, g_full):
        return self.TPhi @ q + g_full[self.sel]

    def assemble(self, q, g_full):
        return self.asm.Assemble(self.u_reduced(q, g_full))
