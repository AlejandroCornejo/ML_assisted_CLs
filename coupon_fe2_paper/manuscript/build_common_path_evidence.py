#!/usr/bin/env python3
"""Build the common-state FOM-field figure for Section 5."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE = PROJECT / "07_material_b/results/common_path_evidence_v1"
FIGURES = HERE / "figures"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{bm}",
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.5,
        "ytick.labelsize": 7,
        "ytick.major.width": 0.5,
        "ytick.major.size": 2.5,
    })


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    subprocess.run(["pdfinfo", str(pdf)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "220", str(pdf),
                    str(FIGURES / name)], check=True, stdout=subprocess.DEVNULL)


def positive_sqrt_from_green(strain: np.ndarray) -> np.ndarray:
    C = np.array([[1.0 + 2.0 * strain[0], strain[2]],
                  [strain[2], 1.0 + 2.0 * strain[1]]])
    values, vectors = np.linalg.eigh(C)
    if np.min(values) <= 0.0:
        raise ValueError("Endpoint is not an admissible Green strain")
    return (vectors * np.sqrt(values)) @ vectors.T


def nodal_average(triangles: np.ndarray, values: np.ndarray, nodes: int) -> np.ndarray:
    """Average element values to the corner nodes for smooth display; unused nodes get zero."""
    total = np.zeros(nodes)
    count = np.zeros_like(total)
    np.add.at(total, triangles.ravel(), np.repeat(values, 3))
    np.add.at(count, triangles.ravel(), 1.0)
    return total / np.maximum(count, 1.0)


def field_figure(data: np.lib.npyio.NpzFile) -> None:
    F = positive_sqrt_from_green(data["strain"][-1])
    fluctuation, deformed, triangulations = {}, {}, {}
    for key in ("sc", "mc"):
        xy, u = data[f"{key}_xy"], data[f"{key}_u_nodal"]
        # Normalized by the cell side, as the coordinates of the geometry figures.
        side = float(np.ptp(xy[:, 0]))
        if not np.isclose(side, np.ptp(xy[:, 1])):
            raise ValueError("Expected a square cell")
        fluctuation[key] = np.linalg.norm(u - xy @ (F - np.eye(2)).T, axis=1) / side
        deformed[key] = xy + u
        triangulations[key] = mtri.Triangulation(
            deformed[key][:, 0], deformed[key][:, 1], data[f"{key}_triangles"])
    # Drawn at its printed size, 0.7 of the text width.
    fig, axes = plt.subplots(2, 2, figsize=(4.42, 3.27), layout="constrained")
    for column, (key, title) in enumerate((("sc", "SC-RVE"), ("mc", "MC-RVE"))):
        tri = triangulations[key]
        top = axes[0, column]
        displacement_max = float(fluctuation[key].max())
        im_u = top.tripcolor(tri, fluctuation[key], shading="gouraud", cmap="coolwarm",
                             vmin=0.0, vmax=displacement_max, alpha=0.82)
        top.triplot(tri, color="#334155", linewidth=0.075, alpha=0.30)
        top.set_title(title, fontsize=9, pad=3)

        bottom = axes[1, column]
        stress = nodal_average(data[f"{key}_triangles"], data[f"{key}_von_mises_element"] / 1e6,
                               len(data[f"{key}_xy"]))
        im_s = bottom.tripcolor(tri, stress, shading="gouraud", cmap="jet",
                                vmin=0.0, vmax=float(stress.max()), alpha=0.82)
        bottom.triplot(tri, color="#334155", linewidth=0.075, alpha=0.30)
        for ax in (top, bottom):
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#7C838C")
                spine.set_linewidth(0.5)
        for image, ax in ((im_u, top), (im_s, bottom)):
            bar = fig.colorbar(image, ax=ax, location="right", shrink=0.86, pad=0.018)
            bar.outline.set_linewidth(0.5)
            bar.locator = MaxNLocator(5, steps=[1, 2, 5, 10])
    axes[0, 0].set_ylabel(r"Periodic fluctuation $\|\bm w\|/\ell$")
    axes[1, 0].set_ylabel(r"Local Cauchy $\sigma_{\mathrm{vm}}$ [MPa]")
    fig.get_layout_engine().set(h_pad=0.05, w_pad=0.04, hspace=0.05, wspace=0.04)
    save(fig, "rve_common_diagnostic_fields")


def main() -> None:
    report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if (report.get("status") != "complete"
            or report.get("evidence_sha256") != digest(SOURCE / "evidence.npz")):
        raise ValueError("Invalid FOM-field evidence receipt")
    configure()
    with np.load(SOURCE / "evidence.npz", allow_pickle=False) as data:
        field_figure(data)


if __name__ == "__main__":
    main()
