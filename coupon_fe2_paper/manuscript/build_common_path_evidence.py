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

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE = PROJECT / "07_material_b/results/common_path_evidence_v1"
FIGURES = HERE / "figures"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 8.5,
        "axes.linewidth": 0.65,
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


def field_figure(data: np.lib.npyio.NpzFile) -> None:
    F = positive_sqrt_from_green(data["strain"][-1])
    fluctuation, deformed, triangulations = {}, {}, {}
    for key in ("sc", "mc"):
        xy, u = data[f"{key}_xy"], data[f"{key}_u_nodal"]
        fluctuation[key] = np.linalg.norm(u - xy @ (F - np.eye(2)).T, axis=1)
        deformed[key] = xy + u
        triangulations[key] = mtri.Triangulation(
            deformed[key][:, 0], deformed[key][:, 1], data[f"{key}_triangles"])
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.0), layout="constrained")
    for column, (key, title) in enumerate((("sc", "SC--RVE"), ("mc", "MC--RVE"))):
        tri = triangulations[key]
        top = axes[0, column]
        displacement_max = float(fluctuation[key].max())
        im_u = top.tripcolor(tri, fluctuation[key], shading="gouraud", cmap="coolwarm",
                             vmin=0.0, vmax=displacement_max, alpha=0.82)
        top.triplot(tri, color="#334155", linewidth=0.075, alpha=0.30)
        top.set_title(title, fontsize=10, fontweight="bold", pad=5)

        bottom = axes[1, column]
        stress_max = float(data[f"{key}_von_mises_element"].max()) / 1e6
        im_s = bottom.tripcolor(tri, facecolors=data[f"{key}_von_mises_element"] / 1e6,
                                shading="flat", cmap="jet", vmin=0.0, vmax=stress_max, alpha=0.82)
        bottom.triplot(tri, color="#334155", linewidth=0.075, alpha=0.30)
        for ax in (top, bottom):
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#7C838C")
                spine.set_linewidth(0.6)
        fig.colorbar(im_u, ax=top, location="right", shrink=0.86, pad=0.018)
        fig.colorbar(im_s, ax=bottom, location="right", shrink=0.86, pad=0.018)
    axes[0, 0].set_ylabel(r"Periodic fluctuation $\|u_\mu-(F-I)X\|$")
    axes[1, 0].set_ylabel(r"Local Cauchy $\sigma_{\mathrm{vm}}$")
    fig.suptitle(r"FOM microscopic fields at a common diagnostic state "
                 r"$(E_{11},E_{22},2E_{12})=(18,-4,8)\%$",
                 fontsize=10.2, fontweight="bold")
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
