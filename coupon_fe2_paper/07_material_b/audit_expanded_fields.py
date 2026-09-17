"""Read-only FE-map/contour diagnostic; sampled checks are not contact proofs."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon-b-expanded-mpl")
import argparse
import json
from pathlib import Path
import numpy as np
from summarize_box_extension import digest
from run_preflight import rim_indices
from audit_pilot import cross

HERE = Path(__file__).resolve().parent


def crossings(polygon):
    """Proper segment crossings, independently localized by a 2x2 solve."""
    found = []
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i+1) % len(polygon)]
        for j in range(i+2, len(polygon)):
            if i == 0 and j == len(polygon)-1:
                continue
            c, d = polygon[j], polygon[(j+1) % len(polygon)]
            if cross(b-a, c-a)*cross(b-a, d-a) < 0 and cross(d-c, a-c)*cross(d-c, b-c) < 0:
                parameters = np.linalg.solve(np.column_stack((b-a, -(d-c))), c-a)
                found.append(dict(segments=[i, j], fractions=parameters.tolist(),
                                  point=(a+parameters[0]*(b-a)).tolist()))
    return found


def sampled_map(xy, deformed, triangles):
    minimum, location, inverted = np.inf, None, set()
    reference, current = xy[triangles], deformed[triangles]
    for i in range(11):
        for j in range(11-i):
            r, s = i/10, j/10
            ell = 1-r-s
            grad = np.array([[1-4*ell, 1-4*ell], [4*r-1, 0], [0, 4*s-1],
                             [4*(ell-r), -4*r], [4*s, 4*r], [-4*s, 4*(ell-s)]])
            initial = np.linalg.det(np.einsum("eni,nj->eij", reference, grad))
            if initial.min() <= 0:
                raise ValueError("Nonpositive sampled reference Jacobian")
            ratio = np.linalg.det(np.einsum("eni,nj->eij", current, grad))/initial
            inverted.update(np.flatnonzero(ratio <= 0).tolist())
            element = int(np.argmin(ratio))
            if ratio[element] < minimum:
                minimum = float(ratio[element])
                location = dict(element_index_zero_based=element, local_coordinates=[r, s])
    return dict(minimum_sampled_det_F=minimum, minimum_location=location,
                nonpositive_elements=len(inverted), reference_points_per_element=66)


def rim_topology(ids, triangles):
    """Check the sorted polygon follows actual corner--midnode--corner FE edges."""
    rim = set(ids.tolist())
    edges = [tuple(e) for t in triangles for e in (t[[0, 3, 1]], t[[1, 4, 2]], t[[2, 5, 0]]) if set(e) <= rim]
    parts = {frozenset((a, b)) for edge in edges for a, b in zip(edge[:-1], edge[1:])}
    valid = len(parts) == len(ids) and all(frozenset((int(ids[i]), int(ids[(i+1) % len(ids)]))) in parts
                                         for i in range(len(ids)))
    return valid, edges


def curved_rim(ids, edges, deformed):
    """Eight straight diagnostic subdivisions per quadratic boundary edge."""
    by_pair = {}
    for edge in edges:
        by_pair[frozenset((edge[0], edge[1]))] = edge
        by_pair[frozenset((edge[1], edge[2]))] = edge[::-1]
    pieces = []
    t = np.arange(8)/8
    for i in range(len(ids)):
        key = frozenset((ids[i], ids[(i+1) % len(ids)]))
        if key not in by_pair:
            continue
        edge = by_pair[key]
        if ids[i] != edge[0]:
            continue
        a, m, b = deformed[list(edge)]
        pieces.append((2*(t-.5)*(t-1))[:, None]*a+(4*t*(1-t))[:, None]*m+(2*t*(t-.5))[:, None]*b)
    if len(pieces) != len(edges):
        raise ValueError("Cannot order quadratic cavity edges")
    return np.vstack(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=HERE / "expanded_reference_v1")
    parser.add_argument("--check", type=Path, default=HERE / "expanded_check_v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads((p / "report.json").read_text()) for p in (args.reference, args.check)]
    if any(r["status"] != "complete" for r in reports):
        raise RuntimeError("Wait for both complete reports")
    spec = reports[0]["spec"]
    original = json.loads((HERE / spec["original_spec"]).read_text())
    geometry = json.loads((HERE / original["parent_refinement"] / "report.json").read_text())["parent_spec"]
    records, displays, sources = [], {}, {}
    for folder, report in zip((args.reference, args.check), reports):
        directories = spec["source_directories"][report["mesh_role"]]
        mesh = HERE / directories["primary"] / (directories["mesh_name"]+".npz")
        sources[str(mesh)] = digest(mesh)
        sources[str(folder / "report.json")] = digest(folder / "report.json")
        with np.load(mesh) as data:
            xy, triangles = data["xy"], data["triangles"]
        rims = rim_indices(geometry, xy)
        topology = [rim_topology(ids, triangles) for ids in rims]
        if not all(valid for valid, _edges in topology):
            raise ValueError("Reference angular rim order does not follow FE boundary topology")
        for s in report["states"]:
            if not s["ok"]:
                continue
            file = Path(s["fields_file"])
            sources[str(file)] = digest(file)
            with np.load(file) as data:
                deformed = xy+data["node_displacement"]
            row = dict(mesh_role=report["mesh_role"], name=s["name"], strain=s["strain"],
                       rim_order_matches_FE_topology=True, **sampled_map(xy, deformed, triangles))
            row["polygon_crossings"] = [dict(cavity_index_zero_based=k, **c) for k, ids in enumerate(rims)
                                       for c in crossings(deformed[ids])]
            row["sampled_quadratic_crossings"] = []
            for k in {c["cavity_index_zero_based"] for c in row["polygon_crossings"]}:
                curve = curved_rim(rims[k], topology[k][1], deformed)
                row["sampled_quadratic_crossings"].extend(dict(cavity_index_zero_based=k, **c)
                                                          for c in crossings(curve))
            records.append(row)
            if s["name"] == "corner_low_low_negative":
                displays[report["mesh_role"]] = [(deformed[ids], curved_rim(ids, edges, deformed))
                                                for ids, (_valid, edges) in zip(rims, topology)]
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="complete", script_sha256=digest(__file__), source_sha256=sources, states=records,
        purpose="Additional saved-field diagnostic after the declared endpoint contour screen failed; does not revise its flags or thresholds.",
        scope="Det F sampled at 66 reference points per quadratic triangle, including its boundary. Positive values do not prove "
        "positivity everywhere; negative values identify a local FE orientation failure numerically. No inference of physical buckling "
        "or constitutive instability. Crossings are straight node-polygon diagnostics; displayed quadratic boundaries are sampled guides, not contact certificates.")
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    for axis, role in zip(axes, ("reference", "check")):
        for k, (polygon, curved) in enumerate(displays[role]):
            axis.plot(*np.vstack((curved, curved[:1])).T, color=f"C{k}", lw=1.3)
            axis.plot(*np.vstack((polygon, polygon[:1])).T, color=f"C{k}", lw=.5, alpha=.5)
        row = next(s for s in records if s["mesh_role"] == role and s["name"] == "corner_low_low_negative")
        for c in row["polygon_crossings"]:
            axis.scatter(*c["point"], color="red", marker="x", s=50, zorder=4)
        axis.set_title(("Referencia: 4,621 elementos" if role == "reference" else "Comprobación: 8,961 elementos")
                       +f"\nMínimo det F muestreado: {row['minimum_sampled_det_F']:.3f}")
        axis.set(xlabel="x [escala de la celda]", ylabel="y [escala de la celda]")
        axis.set_aspect("equal")
        axis.grid(alpha=.2)
    fig.suptitle("B · esquina con compresión biaxial y cortante negativo\n"+r"$e=(-0.04,-0.04,-0.16)$", fontsize=13)
    fig.text(.5, .015, "Contornos FE cuadráticos muestreados · Cruces rojas: intersecciones del polígono de nodos\n"
             "Diagnóstico de la discretización; no demuestra una inestabilidad física", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .88))
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("compression_shear_contours."+suffix), dpi=160)
    plt.close(fig)
    print(json.dumps([s for s in records if s["nonpositive_elements"] or s["polygon_crossings"]], indent=2))


if __name__ == "__main__":
    main()
