#!/usr/bin/env python3
"""Cook-style field plots for the clean FOM--FE2 coupon reference.

The FOM--FE2 result is available at macro Gauss points.  For plotting only,
the three values of each macro T6 element are averaged and patch-recovered to
its nodes.  This gives continuous fields without changing the FE2 solution.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


HERE = Path(__file__).resolve().parent
RESULT = HERE / "fom_fe2_clean_timing_fomfe2_w4_f100kn.npz"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    }
)


def tri6_to_tri3(tris6: np.ndarray) -> np.ndarray:
    """Subdivide one quadratic triangle into four linear triangles."""
    return np.vstack(
        (
            tris6[:, [0, 3, 5]],
            tris6[:, [3, 1, 4]],
            tris6[:, [5, 4, 2]],
            tris6[:, [3, 4, 5]],
        )
    )


def recover_to_nodes(element_values: np.ndarray, tris6: np.ndarray, n_nodes: int) -> np.ndarray:
    """Patch-average one macro-element scalar to the T6 nodes."""
    total = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    np.add.at(total, tris6.ravel(), np.repeat(element_values, 6))
    np.add.at(count, tris6.ravel(), 1.0)
    return total / np.maximum(count, 1.0)


def symmetric_limits(field: np.ndarray) -> tuple[float, float]:
    bound = float(np.max(np.abs(field)))
    return -bound, bound


def draw_field(
    ax: plt.Axes,
    deformed_m: np.ndarray,
    tris6: np.ndarray,
    nodal_field: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    show_xlabel: bool = True,
) -> None:
    """Draw a smooth, deformed field with the same visual language as Cook."""
    deformed_mm = 1.0e3 * deformed_m
    fine = mtri.Triangulation(deformed_mm[:, 0], deformed_mm[:, 1], tri6_to_tri3(tris6))
    coarse = mtri.Triangulation(deformed_mm[:, 0], deformed_mm[:, 1], tris6[:, :3])
    artist = ax.tripcolor(
        fine,
        nodal_field,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.triplot(coarse, color="white", linewidth=0.28, alpha=0.46)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=5)
    if show_xlabel:
        ax.set_xlabel(r"$x$ [mm]")
    else:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"$y$ [mm]")
    ax.tick_params(direction="in", top=True, right=True, length=3)
    colorbar = ax.figure.colorbar(artist, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label(colorbar_label, rotation=90, labelpad=8)
    colorbar.ax.tick_params(labelsize=8)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.png", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not RESULT.exists():
        raise FileNotFoundError(f"Missing clean reference: {RESULT}")

    data = np.load(RESULT)
    nodes = data["coords"]
    tris6 = data["connectivity"].astype(int)
    displacement = data["u_nodal"]
    E_gp = data["E_final"].reshape(len(tris6), 3, 3).mean(axis=1)
    S_gp_mpa = 1.0e-6 * data["S_final"].reshape(len(tris6), 3, 3).mean(axis=1)

    deformed = nodes + displacement
    umag_mm = 1.0e3 * np.linalg.norm(displacement, axis=1)
    e11 = recover_to_nodes(E_gp[:, 0], tris6, len(nodes))
    e22 = recover_to_nodes(E_gp[:, 1], tris6, len(nodes))
    gamma12 = recover_to_nodes(E_gp[:, 2], tris6, len(nodes))
    s11 = recover_to_nodes(S_gp_mpa[:, 0], tris6, len(nodes))
    s22 = recover_to_nodes(S_gp_mpa[:, 1], tris6, len(nodes))
    s12 = recover_to_nodes(S_gp_mpa[:, 2], tris6, len(nodes))

    # This is an in-plane S-equivalent, not a full 3-D von Mises stress:
    # the macro FE2 result supplies (S11, S22, S12), but not S33.
    seq2d_element = np.sqrt(
        np.maximum(
            S_gp_mpa[:, 0] ** 2
            - S_gp_mpa[:, 0] * S_gp_mpa[:, 1]
            + S_gp_mpa[:, 1] ** 2
            + 3.0 * S_gp_mpa[:, 2] ** 2,
            0.0,
        )
    )
    seq2d = recover_to_nodes(seq2d_element, tris6, len(nodes))

    fig, axes = plt.subplots(4, 1, figsize=(13.8, 7.4), constrained_layout=True)
    fig.suptitle(
        r"ASTM D638 coupon: clean FOM--FE$^2$ reference at $F=100\,\mathrm{kN}$",
        fontsize=12.5,
    )
    draw_field(
        axes[0], deformed, tris6, umag_mm, r"(a) Displacement magnitude", r"$\|\mathbf{u}\|$ [mm]", "viridis", show_xlabel=False
    )
    draw_field(
        axes[1], deformed, tris6, e11, r"(b) Axial Green--Lagrange strain", r"$E_{11}$ [-]", "magma", show_xlabel=False
    )
    draw_field(
        axes[2], deformed, tris6, s11, r"(c) Axial second Piola stress", r"$S_{11}$ [MPa]", "viridis", show_xlabel=False
    )
    draw_field(
        axes[3], deformed, tris6, seq2d,
        r"(d) In-plane equivalent $S_{\mathrm{eq}}^{2D}$ (not 3-D von Mises)",
        r"$S_{\mathrm{eq}}^{2D}$ [MPa]", "viridis",
    )
    save(fig, "fom_fe2_axial_fields")

    gamma_lims = symmetric_limits(gamma12)
    s22_lims = symmetric_limits(s22)
    s12_lims = symmetric_limits(s12)
    fig, axes = plt.subplots(4, 1, figsize=(13.8, 7.4), constrained_layout=True)
    fig.suptitle(
        r"ASTM D638 coupon: transverse and shear response, clean FOM--FE$^2$ at $F=100\,\mathrm{kN}$",
        fontsize=12.5,
    )
    draw_field(
        axes[0], deformed, tris6, e22, r"(a) Transverse Green--Lagrange strain", r"$E_{22}$ [-]", "viridis_r", show_xlabel=False
    )
    draw_field(
        axes[1], deformed, tris6, gamma12, r"(b) Engineering shear strain", r"$\gamma_{12}$ [-]", "RdBu_r", *gamma_lims,
        show_xlabel=False,
    )
    draw_field(
        axes[2], deformed, tris6, s22, r"(c) Transverse second Piola stress", r"$S_{22}$ [MPa]", "RdBu_r", *s22_lims,
        show_xlabel=False,
    )
    draw_field(
        axes[3], deformed, tris6, s12, r"(d) Shear second Piola stress", r"$S_{12}$ [MPa]", "RdBu_r", *s12_lims,
    )
    save(fig, "fom_fe2_transverse_shear")

    print("Wrote Cook-style figures:")
    for stem in ("fom_fe2_axial_fields", "fom_fe2_transverse_shear"):
        print(HERE / f"{stem}.png")
        print(HERE / f"{stem}.pdf")


if __name__ == "__main__":
    main()
