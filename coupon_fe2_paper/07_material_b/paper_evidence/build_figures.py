"""Build the four reader-facing Material-B figures from approved records.

Only statistical coordinates are read for the test split. No test/path FOM
target, neural checkpoint, or neural prediction is loaded.
"""
from __future__ import annotations

import csv
import hashlib
import json
from math import pi, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Ellipse, Rectangle


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
GEOMETRY = BASE / "pilot_spec.json"
MESH = BASE / "preflight_reference_v1/reference.npz"
DESIGN = BASE / "results/data_protocol_design_v1.npz"
PROTOCOL = BASE / "protocol/data_protocol_v1.json"
NONLINEAR = BASE / "results/nonlinear_response_v1/decision.json"
PILOT_RESPONSE = BASE / "results/physical_response_v2/response.csv"

NAVY = "#175D8C"
ORANGE = "#D55E00"
TEAL = "#008B8B"
PURPLE = "#7A4EAB"
GRAY = "#626B73"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def draw_geometry(ax, spec):
    ax.add_patch(Rectangle((-1, -1), 2, 2, facecolor="#DCE8F1",
                           edgecolor="#27465E", linewidth=1.5))
    for i, cavity in enumerate(spec["cavities"], start=1):
        area = 4*cavity["area_over_cell"]
        minor = sqrt(area/(pi*cavity["aspect"]))
        major = cavity["aspect"]*minor
        center = 2*np.asarray(cavity["center_over_L"])
        ax.add_patch(Ellipse(center, 2*major, 2*minor, angle=cavity["angle_deg"],
                             facecolor="white", edgecolor=TEAL, linewidth=2))
        ax.text(*center, str(i), ha="center", va="center", color=TEAL,
                fontsize=11, fontweight="bold")
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), aspect="equal",
           xlabel=r"$X_1/L$", ylabel=r"$X_2/L$")
    ax.set_title("Material B periodic unit cell", fontsize=15, fontweight="bold")


def geometry_figure():
    spec = json.loads(GEOMETRY.read_text())
    fig, ax = plt.subplots(figsize=(5.6, 5.5))
    draw_geometry(ax, spec)
    fig.text(.5, .015, "Four elliptical cavities · 20% total void area",
             ha="center", fontsize=10.5, color=GRAY)
    fig.subplots_adjust(bottom=.17, top=.90)
    save(fig, "01_geometry")


def mesh_figure():
    with np.load(MESH, allow_pickle=False) as mesh:
        xy = np.asarray(mesh["xy"])
        triangles = np.asarray(mesh["triangles"])
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles[:, :3])
    fig, ax = plt.subplots(figsize=(6.1, 5.7))
    ax.set_facecolor("#F7F9FA")
    ax.triplot(triangulation, color="#35566B", linewidth=.18, alpha=.68)
    ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal",
           xlabel=r"$X_1/L$", ylabel=r"$X_2/L$")
    ax.set_title("Finite-element mesh", fontsize=15, fontweight="bold")
    fig.text(.5, .015,
             f"Working FOM mesh · {len(triangles):,} six-node triangles · {len(xy):,} nodes",
             ha="center", fontsize=10.5, color=GRAY)
    fig.subplots_adjust(bottom=.17, top=.90)
    save(fig, "02_mesh")


def parameter_space_figure():
    with np.load(DESIGN, allow_pickle=False) as design:
        fit = np.asarray(design["E_fit"])*100
        validation = np.asarray(design["E_validation"])*100
        test = np.asarray(design["E_test"])*100

    fig = plt.figure(figsize=(7.4, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(fit[:, 0], fit[:, 1], fit[:, 2], s=6, color=NAVY, alpha=.75,
               linewidths=0, rasterized=True, label=f"Training ({len(fit):,})")
    ax.scatter(validation[:, 0], validation[:, 1], validation[:, 2], s=19,
               facecolors="none", edgecolors=ORANGE, linewidths=.85, alpha=.95,
               rasterized=True, label=f"Validation ({len(validation):,})")
    ax.scatter(test[:, 0], test[:, 1], test[:, 2], s=15, color=PURPLE, marker="x",
               linewidths=.80, alpha=.95, rasterized=True, label=f"Test ({len(test):,})")
    ax.set_xlabel(r"$E_{11}$ [%]", labelpad=8)
    ax.set_ylabel(r"$E_{22}$ [%]", labelpad=8)
    ax.set_zlabel(r"$\gamma_{12}=2E_{12}$ [%]", labelpad=8)
    ax.set_box_aspect((1, 1, .78))
    ax.view_init(elev=23, azim=-57)
    ax.grid(alpha=.22)
    ax.set_title("Strain-coordinate design", fontsize=15, fontweight="bold", pad=14)
    ax.legend(loc="upper left", framealpha=.88, borderpad=.7)
    fig.text(.5, .015,
             "Each marker is one prescribed macroscopic strain coordinate; no stress labels are shown.",
             ha="center", fontsize=10.5, color=GRAY)
    fig.subplots_adjust(left=.02, right=.96, bottom=.10, top=.91)
    save(fig, "03_parameter_space")


def load_response():
    decision = json.loads(NONLINEAR.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    if decision["status"] != "complete" or not decision["passed"]:
        raise ValueError("Nonlinear FOM record is not approved")
    records = decision["baseline"]+decision["extended_states"]
    by_path = {}
    accepted_max = dict(axial_x_extension=protocol["bounds"]["upper"][0],
                        axial_y_extension=protocol["bounds"]["upper"][1])
    components = dict(axial_x_extension=0, axial_y_extension=1)
    for name, component in components.items():
        rows = [row for row in records if row["path"] == name
                and row["strain"][component] <= accepted_max[name]+1e-15]
        if not rows:
            raise ValueError(f"Missing path {name}")
        by_path[name] = rows
    with PILOT_RESPONSE.open(newline="") as stream:
        pilot = list(csv.DictReader(stream))
    return by_path, pilot


def axial_series(by_path, pilot, axis):
    suffix = "x" if axis == 0 else "y"
    compression = [r for r in pilot if r["path"] == f"axial_{suffix}_compression"]
    extension = by_path[f"axial_{suffix}_extension"]
    rows = [(float(r["E11"] if axis == 0 else r["E22"]),
             float(r[f"S{axis+1}{axis+1}_Pa"])/1e6) for r in compression]
    rows += [(float(r["strain"][axis]), float(r["stress"][axis])/1e6) for r in extension]
    rows.append((0., 0.))
    rows = np.asarray(sorted(set(rows)))
    return rows[:, 0]*100, rows[:, 1]


def shear_series(pilot):
    paths = {"negative_green_shear", "positive_green_shear"}
    rows = [(float(r["2E12"]), float(r["S12_Pa"])/1e6)
            for r in pilot if r["path"] in paths]
    rows.append((0., 0.))
    rows = np.asarray(sorted(set(rows)))
    return rows[:, 0]*100, rows[:, 1]


def response_figure():
    by_path, pilot = load_response()
    series = [axial_series(by_path, pilot, 0),
              axial_series(by_path, pilot, 1),
              shear_series(pilot)]
    titles = [r"Horizontal loading: $E_{11}$", r"Vertical loading: $E_{22}$",
              r"Shear loading: $\gamma_{12}=2E_{12}$"]
    ylabels = [r"$S_{11}$ [MPa]", r"$S_{22}$ [MPa]", r"$S_{12}$ [MPa]"]
    colors = [NAVY, ORANGE, TEAL]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, (strain, fom), title, ylabel, color in zip(axes, series, titles, ylabels, colors):
        ax.plot(strain, fom, "o-", color=color, linewidth=2.5, markersize=5.5,
                label="FOM response")
        ax.axhline(0, color="#B7BEC4", linewidth=.8)
        ax.axvline(0, color="#B7BEC4", linewidth=.8)
        ax.grid(alpha=.20)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Imposed Green strain [%]")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", framealpha=.74)
    fig.suptitle("Material B: FOM stress response in the accepted domain",
                 fontsize=16, fontweight="bold", x=.04, ha="left")
    fig.text(.04, .015,
             "Prescribed-strain paths; transverse strains are zero. Markers are saved FOM states. No neural predictions.",
             fontsize=10, color=GRAY)
    fig.subplots_adjust(top=.79, bottom=.18, wspace=.30)
    save(fig, "04_fom_response")


def main():
    geometry_figure()
    mesh_figure()
    parameter_space_figure()
    response_figure()
    source_paths = (GEOMETRY, MESH, DESIGN, PROTOCOL, NONLINEAR, PILOT_RESPONSE)
    manifest = {
        "purpose": "Reader-facing pretraining Material-B evidence",
        "generated": [
            f"{stem}.{extension}"
            for stem in ("01_geometry", "02_mesh", "03_parameter_space", "04_fom_response")
            for extension in ("png", "pdf")
        ],
        "sources_sha256": {
            str(path.relative_to(BASE)): digest(path) for path in source_paths
        },
        "test_coordinates_loaded": True,
        "test_labels_loaded": False,
        "reserved_40_point_path_labels_loaded": False,
        "neural_checkpoint_loaded": False,
        "accepted_ranges": {"E11": [-.04, .20], "E22": [-.04, .20], "2E12": [-.08, .08]},
        "limits": "Pretraining geometry, mesh, statistical coordinates, and selected prescribed-strain FOM states; no surrogate evaluation."
    }
    (HERE / "source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
