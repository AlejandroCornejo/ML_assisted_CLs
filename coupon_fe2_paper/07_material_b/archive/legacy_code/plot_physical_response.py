#!/usr/bin/env python3
"""Postprocess saved B FOM fields only; no solver imports or new simulations.

Recover plane-strain sigma33 from the underlying compressible 3D Neo-Hookean
matrix, not from an arbitrary 2D learned energy. Compute von Mises AFTER
homogenizing the complete Cauchy tensor, not by averaging local equivalents.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon-b-response-mpl")
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PATHS = {
    "axial_x_extension": ("Tracción X", "#0072B2", "o", "-"),
    "axial_y_extension": ("Tracción Y", "#D55E00", "s", "-"),
    "axial_x_compression": ("Compresión X", "#56B4E9", "o", "--"),
    "axial_y_compression": ("Compresión Y", "#CC79A7", "s", "--"),
    "positive_green_shear": ("Green-cortante +", "#009E73", "^", "-"),
    "negative_green_shear": ("Green-cortante −", "#E69F00", "v", "--"),
    "combined_x": ("Combinada X", "#5E3C99", "D", "-"),
    "combined_y": ("Combinada Y", "#8C6D31", "P", "--"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deformation_gradient(e):
    C = np.array([[1+2*e[0], e[2]], [e[2], 1+2*e[1]]])
    values, vectors = np.linalg.eigh(C)
    if values.min() <= 0:
        raise ValueError("Nonpositive macroscopic C")
    return (vectors*np.sqrt(values)) @ vectors.T


def von_mises(stress):
    stress = np.asarray(stress, dtype=float)
    deviator = stress-np.eye(3)*np.trace(stress)/3
    return float(np.sqrt(1.5*np.sum(deviator**2)))


def cauchy_from_saved(e, s, F_micro, P_micro, weights, young, poisson, denominator):
    F = deformation_gradient(e)
    J = float(np.linalg.det(F))
    S = np.array([[s[0], s[2]], [s[2], s[1]]])
    in_plane = F @ S @ F.T / J
    local_J = np.linalg.det(F_micro)
    if np.min(local_J) <= 0 or np.min(weights) <= 0:
        raise ValueError("Nonpositive saved microscopic determinant or integration weight")
    # <sigma>_current = integral_ref(P F^T)/(Jbar V0), including void volume.
    integrated = np.einsum("eg,egij->ij", weights,
                          P_micro @ np.swapaxes(F_micro, -1, -2))/(J*denominator)
    error = float(np.linalg.norm(in_plane-integrated)/max(np.linalg.norm(in_plane), 1.))
    if error > 1e-7:
        raise ValueError(f"Saved-field/macroscopic Cauchy inconsistency: {error:g}")
    lam = young*poisson/((1+poisson)*(1-2*poisson))
    # F33=1: P33 = lambda log(Jmicro); sigma33 = P33/Jmicro locally.
    # Weighting by current volume Jmicro dV0 cancels the local denominator.
    sigma = np.zeros((3, 3))
    sigma[:2, :2] = in_plane
    sigma[2, 2] = lam*np.sum(weights*np.log(local_J))/(J*denominator)
    if not np.isfinite(sigma).all():
        raise ValueError("Nonfinite homogenized Cauchy stress")
    return sigma, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "preflight_check_v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.source / "report.json"
    report = json.loads(report_path.read_text())
    if report["stage"] != "check" or report["status"] != "complete" or not report["numerical_states_complete"]:
        raise ValueError("Require a completed check-stage report with all states reached")
    parent_path = HERE / report["spec"]["parent_refinement"] / "report.json"
    if digest(parent_path) != report["parent_report_sha256"]:
        raise ValueError("Historical parent report changed")
    geometry = json.loads(parent_path.read_text())["parent_spec"]
    # Read literal material constants from the frozen original config, not A's
    # geometry defaults or a subsequently modified shared configuration.
    config_path = HERE / report["spec"]["parent_pilot"] / "source_snapshot/config.py"
    parent_pilot = json.loads((HERE / report["spec"]["parent_pilot"] / "report.json").read_text())
    if digest(config_path) != parent_pilot["source_sha256"]["coupon_fe2_paper/config.py"]:
        raise ValueError("Frozen material configuration hash does not match")
    wanted = {"MATRIX_YOUNG", "MATRIX_POISSON", "MATRIX_THICKNESS"}
    constants = {node.targets[0].id:ast.literal_eval(node.value)
                 for node in ast.parse(config_path.read_text()).body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in wanted}
    young, poisson, thickness = [constants[k] for k in ("MATRIX_YOUNG", "MATRIX_POISSON", "MATRIX_THICKNESS")]
    denominator = geometry["cell_side"]**2*thickness
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    points, sources = [], {str(report_path):digest(report_path), str(parent_path):digest(parent_path),
                           str(config_path):digest(config_path)}
    for path in PATHS:
        endpoint = np.array(geometry["paths"][path])
        rows = [s for s in report["states"] if s["name"].startswith("pilot__"+path+"__")]
        rows.sort(key=lambda s:float(s["name"].split("__")[-1]))
        if [float(s["name"].split("__")[-1]) for s in rows] != geometry["path_fractions"]:
            raise ValueError("Missing or duplicated ray target: "+path)
        # An analytically normalized reference, explicitly NOT another FOM solve.
        points.append(dict(path=path, t=0., origin="analytic_reference", e=[0., 0., 0.],
                           stress=[0., 0., 0.], energy=0., cauchy=np.zeros((3, 3)).tolist(),
                           von_mises=0., cauchy_consistency_error=0.))
        for row in rows:
            fraction = float(row["name"].split("__")[-1])
            if not row["ok"] or not np.allclose(row["strain"], fraction*endpoint, atol=1e-14, rtol=0):
                raise ValueError("Incorrect ray state: "+row["name"])
            state_path = args.source / (row["name"]+".npz")
            with np.load(state_path) as fields:
                for key, expected in (("strain", row["strain"]), ("stress", row["stress"])):
                    if not np.array_equal(fields[key], expected):
                        raise ValueError("Saved/report mismatch: "+key)
                sigma, error = cauchy_from_saved(row["strain"], row["stress"], fields["F_micro"],
                    fields["P_micro"], fields["weights"], young, poisson, denominator)
            points.append(dict(path=path, t=fraction, origin="saved_FOM", e=row["strain"],
                               stress=row["stress"], energy=row["energy"], cauchy=sigma.tolist(),
                               von_mises=von_mises(sigma), cauchy_consistency_error=error))
            sources[str(state_path)] = digest(state_path)
    manifest = dict(source_sha256=sources, script_sha256=digest(Path(__file__)),
        mesh_elements=report["geometry"]["n_elements"], material=constants, cell_side=geometry["cell_side"],
        points=points, endpoints=geometry["paths"],
        scope="Saved FOM postprocessing only. Origin is analytic; straight connecting lines are visual guides. "
              "t is load fraction, not time. Von Mises of homogenized 3D Cauchy stress includes sigma33 "
              "reconstructed from the underlying 3D Neo-Hookean matrix in plane strain. Not the mean/max "
              "of microscopic von Mises, not a yield criterion and not a proof of hyperelasticity or a "
              "3D guarantee for the learned in-plane energy.")
    (output / "response.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    with (output / "response.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "load_fraction", "origin", "E11", "E22", "2E12",
                         "S11_Pa", "S22_Pa", "S12_Pa", "sigma11_Pa", "sigma22_Pa", "sigma12_Pa",
                         "sigma33_Pa", "von_mises_Pa", "energy_J_per_m3"])
        for p in points:
            sigma = np.array(p["cauchy"])
            writer.writerow([p["path"], p["t"], p["origin"], *p["e"], *p["stress"],
                             sigma[0, 0], sigma[1, 1], sigma[0, 1], sigma[2, 2], p["von_mises"], p["energy"]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex":False, "font.family":"DejaVu Sans", "font.size":11,
        "axes.spines.top":False, "axes.spines.right":False, "axes.titleweight":"bold"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios":[1., 1.05, 1.05]})
    for component, (ext, comp, color) in enumerate((
            ("axial_x_extension", "axial_x_compression", "#0072B2"),
            ("axial_y_extension", "axial_y_compression", "#D55E00"))):
        values = [p for p in points if p["path"] in (ext, comp) and p["t"] > 0]
        values.append(next(p for p in points if p["path"] == ext and p["t"] == 0))
        values.sort(key=lambda p:p["e"][component])
        axes[0].plot([100*p["e"][component] for p in values],
            [p["stress"][component]/1e6 for p in values], marker="o" if component == 0 else "s",
            color=color, lw=1.8, ms=5, label="Dirección "+("X" if component == 0 else "Y"))
    for path, (label, color, marker, style) in PATHS.items():
        values = [p for p in points if p["path"] == path]
        for ax, key, scale in ((axes[1], "von_mises", 1e6), (axes[2], "energy", 1e6)):
            ax.plot([p["t"] for p in values], [p[key]/scale for p in values],
                    color=color, marker=marker, ls=style, ms=5, lw=1.6, label=label)
    axes[0].axhline(0, color="0.6", lw=.7)
    axes[0].axvline(0, color="0.6", lw=.7)
    axes[0].set(title="(a) Respuesta axial con signo", xlabel=r"Deformación axial $E_{ii}$ [%]",
                ylabel=r"Tensión material $S_{ii}$ [MPa]")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].set(title="(b) Tensión equivalente", xlabel=r"Fracción de carga $t$",
                ylabel=r"Von Mises de $\overline{\boldsymbol{\sigma}}$ [MPa]", xlim=(0, 1.04))
    axes[2].set(title="(c) Energía de deformación", xlabel=r"Fracción de carga $t$",
                ylabel=r"Energía $W$ [MJ/m³]", xlim=(0, 1.04))
    for ax in axes:
        ax.grid(alpha=.18)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5, .065), frameon=False, fontsize=10)
    fig.suptitle("Material B · respuesta FOM hiperelástica · 8,961 elementos cuadráticos", y=.97, fontsize=15)
    fig.text(.5, .025, "t = fracción de cada trayectoria, no tiempo · Marcadores: estados guardados; origen analítico · Líneas: guías visuales\n"
             "Panel (a): deformación normal transversal y cortante impuestas cero; no es un ensayo de tensión uniaxial",
             ha="center", fontsize=9, color="0.35")
    fig.subplots_adjust(left=.065, right=.99, top=.84, bottom=.29, wspace=.34)
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("physical_response."+suffix), dpi=160)
    plt.close(fig)
    print("Output:", output)
    print("Worst saved-field Cauchy consistency error:", max(p["cauchy_consistency_error"] for p in points))
    print("Endpoint equivalent stresses [MPa]:", {p["path"]:p["von_mises"]/1e6 for p in points if p["t"] == 1})


if __name__ == "__main__":
    main()
