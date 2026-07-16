#!/usr/bin/env python3
"""Static Manim figure for the HPROM-ANN / D-HPROM-ANN online paths."""

from pathlib import Path

import numpy as np
from manim import (
    BLACK,
    BLUE_D,
    BLUE_E,
    GRAY_B,
    GRAY_C,
    GRAY_E,
    ORANGE,
    WHITE,
    Arrow,
    CurvedArrow,
    Dot,
    Line,
    MathTex,
    Polygon,
    Rectangle,
    RoundedRectangle,
    Scene,
    Tex,
    VGroup,
    config,
)


config.background_color = WHITE

SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "RVE_homogenization_NeoHookean_using_Kratos").is_dir():
    RVE_DIR = SCRIPT_DIR / "RVE_homogenization_NeoHookean_using_Kratos"
elif (SCRIPT_DIR.parent / "rve_geometry.mdpa").exists():
    RVE_DIR = SCRIPT_DIR.parent
else:
    raise FileNotFoundError("Could not locate the RVE_homogenization_NeoHookean_using_Kratos directory.")
MESH_FILE = RVE_DIR / "rve_geometry.mdpa"
ECM_FILE = RVE_DIR / (
    "stage_12_hprom_ann_mawecm_res_eps_sig_phase1to40_phase2to10_sum990_ann/"
    "ecm_weights_all.npz"
)

INK = "#18212B"
BLUE = "#1976D2"
BLUE_FILL = "#EAF3FB"
ORANGE_DARK = "#D66522"
ORANGE_FILL = "#FCEDE4"
MESH_GRAY = "#D7DBDF"


def load_quadratic_triangle_mesh(path):
    nodes = {}
    triangles = []
    section = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "Begin Nodes":
            section = "nodes"
            continue
        if line == "End Nodes":
            section = None
            continue
        if line.startswith("Begin Geometries Triangle2D6"):
            section = "triangles"
            continue
        if line == "End Geometries":
            section = None
            continue
        if not line or line.startswith("//"):
            continue
        fields = line.split()
        if section == "nodes" and len(fields) >= 4:
            nodes[int(fields[0])] = np.array([float(fields[1]), float(fields[2])])
        elif section == "triangles" and len(fields) >= 7:
            triangles.append(tuple(int(v) for v in fields[1:4]))
    return nodes, triangles


def make_formula_panel(content, width, height, edge_color, fill_color):
    frame = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.10,
        stroke_color=edge_color,
        stroke_width=1.8,
        fill_color=fill_color,
        fill_opacity=0.72,
    )
    content.move_to(frame)
    return VGroup(frame, content)


def make_rve_mesh(nodes, triangles, selected, color, width=1.82, gauss_points=False):
    selected = set(int(i) for i in selected)
    scale = width / 2.0
    background = VGroup()
    highlights = VGroup()
    gp_group = VGroup()

    for index, tri in enumerate(triangles):
        xy = [nodes[node_id] for node_id in tri]
        points = [np.array([scale * p[0], scale * p[1], 0.0]) for p in xy]
        background.add(
            Polygon(
                *points,
                stroke_color=MESH_GRAY,
                stroke_width=0.28,
                fill_opacity=0.0,
            )
        )
        if index not in selected:
            continue
        highlights.add(
            Polygon(
                *points,
                stroke_color=color,
                stroke_width=1.0,
                fill_color=color,
                fill_opacity=0.82,
            )
        )
        if gauss_points:
            barycentric = ((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3))
            for b0, b1, b2 in barycentric:
                gp = b0 * points[0] + b1 * points[1] + b2 * points[2]
                gp_group.add(Dot(gp, radius=0.012, color=INK))

    outline = Rectangle(
        width=width,
        height=width,
        stroke_color=GRAY_B,
        stroke_width=0.8,
        fill_opacity=0.0,
    )
    return VGroup(background, highlights, gp_group, outline)


def labeled_arrow(start, end, color, label=None, label_shift=(0.0, 0.0, 0.0), buff=0.08):
    arr = Arrow(start, end, buff=buff, color=color, stroke_width=3.0, max_tip_length_to_length_ratio=0.13)
    if label is None:
        return VGroup(arr)
    tex = MathTex(label, color=color, font_size=25)
    tex.move_to((np.array(start) + np.array(end)) / 2 + np.array(label_shift))
    return VGroup(arr, tex)


class HpromGalerkinStatic(Scene):
    def construct(self):
        nodes, triangles = load_quadratic_triangle_mesh(MESH_FILE)
        ecm = np.load(ECM_FILE, allow_pickle=True)
        z_res = np.asarray(ecm["Z_res"], dtype=int)
        z_sig = np.asarray(ecm["Z_sig"], dtype=int)

        # Shared multiscale-to-manifold path.
        f_tex = MathTex(r"\overline{\mathbf F}", color=INK, font_size=42)
        f_caption = Tex("macro", color=GRAY_C, font_size=23).next_to(f_tex, np.array([0, -1, 0]), buff=0.08)
        f_group = VGroup(f_tex, f_caption).move_to(np.array([-6.52, 2.78, 0]))

        e_formula = MathTex(
            r"\overline{\mathbf E}=\frac12(\overline{\mathbf F}^{T}\overline{\mathbf F}-\mathbf I)",
            color=INK,
            font_size=31,
        )
        mu_formula = MathTex(
            r"\boldsymbol\mu=(E_{xx},E_{yy},G_{xy})",
            color=INK,
            font_size=29,
        )
        e_content = VGroup(e_formula, mu_formula).arrange(np.array([0, -1, 0]), buff=0.10)
        e_group = make_formula_panel(e_content, 3.16, 1.14, BLUE, BLUE_FILL).move_to(np.array([-4.15, 2.78, 0]))

        q_formula = MathTex(r"\mathbf q^{0}(\boldsymbol\mu)", color=INK, font_size=39)
        q_caption = Tex("initial coordinates", color=GRAY_C, font_size=21).next_to(q_formula, np.array([0, -1, 0]), buff=0.10)
        q_content = VGroup(q_formula, q_caption)
        q_group = make_formula_panel(q_content, 2.15, 1.14, BLUE, BLUE_FILL).move_to(np.array([-1.27, 2.78, 0]))

        decoder_1 = MathTex(
            r"\widetilde{\mathbf u}(\mathbf q,\boldsymbol\mu)=\mathbf u_{\rm aff}(\boldsymbol\mu)",
            color=INK,
            font_size=29,
        )
        decoder_2 = MathTex(
            r"+\,\mathbf V\mathbf A\mathbf q+\overline{\mathbf V}\mathcal N_{\theta}(\mathbf q)",
            color=INK,
            font_size=29,
        )
        decoder_content = VGroup(decoder_1, decoder_2).arrange(np.array([0, -1, 0]), buff=0.09)
        decoder = make_formula_panel(decoder_content, 4.02, 1.28, INK, WHITE).move_to(np.array([2.16, 2.78, 0]))

        # Explicit arrows avoid relying on panel bounding-box internals.
        top_arrows = VGroup(
            Arrow(f_group.get_right(), e_group.get_left(), buff=0.10, color=BLUE, stroke_width=3.0),
            Arrow(e_group.get_right(), q_group.get_left(), buff=0.10, color=BLUE, stroke_width=3.0),
            Arrow(q_group.get_right(), decoder.get_left(), buff=0.10, color=BLUE, stroke_width=3.0),
        )

        # Galerkin-corrected route.
        h_label = Tex(r"\textbf{HPROM-ANN}", color=ORANGE_DARK, font_size=31)
        w_def = MathTex(
            r"\mathbf W(\mathbf q):=\frac{\partial\widetilde{\mathbf u}}{\partial\mathbf q}",
            r"=\mathbf V\mathbf A+\overline{\mathbf V}\frac{\partial\mathcal N_\theta}{\partial\mathbf q}",
            color=INK,
            font_size=30,
        )
        newton = MathTex(
            r"\big[\mathbf W^{T}\mathbf J\mathbf W\big]_{\!\mathbf q^k}\,\Delta\mathbf q^k",
            r"=-\,\mathbf W^{T}\mathbf r(\widetilde{\mathbf u}^{,k})",
            color=INK,
            font_size=34,
        )
        update = MathTex(r"\mathbf q^{k+1}=\mathbf q^k+\Delta\mathbf q^k", color=ORANGE_DARK, font_size=27)
        j_def = MathTex(r"\mathbf J:=\partial\mathbf r/\partial\mathbf u", color=GRAY_C, font_size=23)
        h_content = VGroup(h_label, w_def, newton, VGroup(update, j_def).arrange(np.array([1, 0, 0]), buff=0.42))
        h_content.arrange(np.array([0, -1, 0]), buff=0.09)
        h_panel = make_formula_panel(h_content, 7.02, 2.08, ORANGE_DARK, ORANGE_FILL).move_to(
            np.array([-2.55, 0.67, 0])
        )

        # Direct route.
        d_label = Tex(r"\textbf{D-HPROM-ANN}", color=BLUE, font_size=31)
        d_formula = MathTex(r"\mathbf q=\mathbf q^0(\boldsymbol\mu)", color=INK, font_size=34)
        d_caption = Tex("direct state", color=GRAY_C, font_size=22)
        d_content = VGroup(d_label, d_formula, d_caption).arrange(np.array([0, -1, 0]), buff=0.11)
        d_panel = make_formula_panel(d_content, 3.40, 1.55, BLUE, BLUE_FILL).move_to(np.array([4.95, 0.80, 0]))

        decoder_to_h = CurvedArrow(
            decoder.get_bottom() + np.array([-0.55, 0.02, 0]),
            h_panel.get_right() + np.array([-0.02, 0.48, 0]),
            angle=-0.18,
            color=ORANGE_DARK,
            stroke_width=3.2,
        )
        decoder_to_d = CurvedArrow(
            decoder.get_right() + np.array([-0.02, -0.20, 0]),
            d_panel.get_top() + np.array([0.25, 0.00, 0]),
            angle=-0.32,
            color=BLUE,
            stroke_width=3.2,
        )

        # Actual element supports.
        residual_mesh = make_rve_mesh(nodes, triangles, z_res, ORANGE_DARK, width=1.82)
        residual_mesh.move_to(np.array([-5.63, -2.10, 0]))
        residual_label = VGroup(
            Tex(r"\textbf{residual support}", color=ORANGE_DARK, font_size=25),
            MathTex(r"\mathcal Z_{\rm res}", color=ORANGE_DARK, font_size=25),
        ).arrange(np.array([1, 0, 0]), buff=0.12)
        residual_count = Tex("10 / 990 elements", color=GRAY_C, font_size=20)
        residual_label.next_to(residual_mesh, np.array([0, 1, 0]), buff=0.14)
        residual_count.next_to(residual_mesh, np.array([0, -1, 0]), buff=0.10)

        residual_sum = MathTex(
            r"\mathbf W^T\mathbf r",
            r"=\sum_{e\in\mathcal E}\mathbf W_e^T\mathbf r_e",
            r"\approx\sum_{e\in\mathcal Z_{\rm res}}\xi_e^{\rm res}\mathbf W_e^T\mathbf r_e",
            color=INK,
            font_size=29,
        )
        tangent_sum = MathTex(
            r"\mathbf W^T\mathbf J\mathbf W",
            r"\approx\sum_{e\in\mathcal Z_{\rm res}}\xi_e^{\rm res}\mathbf W_e^T\mathbf J_e\mathbf W_e",
            color=INK,
            font_size=27,
        )
        residual_formulas = VGroup(residual_sum, tangent_sum).arrange(np.array([0, -1, 0]), buff=0.22)
        residual_formulas.move_to(np.array([-2.57, -2.05, 0]))

        stress_mesh = make_rve_mesh(nodes, triangles, z_sig, BLUE, width=1.82, gauss_points=True)
        stress_mesh.move_to(np.array([1.05, -2.10, 0]))
        stress_label = VGroup(
            Tex(r"\textbf{stress support}", color=BLUE, font_size=25),
            MathTex(r"\mathcal Z_{\sigma}", color=BLUE, font_size=25),
        ).arrange(np.array([1, 0, 0]), buff=0.12)
        stress_count = Tex("10 / 990 elements", color=GRAY_C, font_size=20)
        stress_label.next_to(stress_mesh, np.array([0, 1, 0]), buff=0.14)
        stress_count.next_to(stress_mesh, np.array([0, -1, 0]), buff=0.10)

        hom_label = Tex(r"\textbf{homogenization}", color=BLUE, font_size=27)
        hom_formula = MathTex(
            r"\overline{\mathbf S}",
            r"=\frac{1}{A_0}\sum_{e\in\mathcal E}A_e\langle\mathbf S_e\rangle",
            r"\approx\frac{1}{A_0}\sum_{e\in\mathcal Z_{\sigma}}\xi_e^{\sigma}A_e\langle\mathbf S_e\rangle",
            color=INK,
            font_size=28,
        )
        gp_formula = MathTex(
            r"\langle\mathbf S_e\rangle",
            r"=\frac{1}{n_{\rm gp}}\sum_{g=1}^{n_{\rm gp}}\mathbf S_{e,g}",
            color=GRAY_C,
            font_size=23,
        )
        hom_group = VGroup(hom_label, hom_formula, gp_formula).arrange(np.array([0, -1, 0]), buff=0.18)
        hom_group.move_to(np.array([4.50, -2.02, 0]))

        h_to_res = CurvedArrow(
            h_panel.get_bottom() + np.array([-2.38, 0.0, 0]),
            residual_mesh.get_left() + np.array([0.02, 0.42, 0]),
            angle=0.18,
            color=ORANGE_DARK,
            stroke_width=3.0,
        )
        res_to_stress = CurvedArrow(
            h_panel.get_bottom() + np.array([2.50, 0.0, 0]),
            stress_mesh.get_left() + np.array([0.02, 0.45, 0]),
            angle=-0.18,
            color=ORANGE_DARK,
            stroke_width=3.0,
        )
        q_star = MathTex(r"\mathbf q^{\star}", color=ORANGE_DARK, font_size=24).move_to(
            np.array([-0.03, -1.43, 0])
        )
        d_to_stress = CurvedArrow(
            d_panel.get_bottom() + np.array([-0.15, 0.02, 0]),
            stress_mesh.get_top() + np.array([0.25, 0.02, 0]),
            angle=-0.35,
            color=BLUE,
            stroke_width=3.0,
        )
        q_zero = MathTex(r"\mathbf q^0", color=BLUE, font_size=24).move_to(np.array([3.0, -0.55, 0]))
        stress_to_hom = Arrow(
            stress_mesh.get_right(),
            hom_group.get_left(),
            buff=0.14,
            color=BLUE,
            stroke_width=3.0,
        )

        objects = VGroup(
            f_group,
            e_group,
            q_group,
            decoder,
            top_arrows,
            h_panel,
            d_panel,
            decoder_to_h,
            decoder_to_d,
            residual_mesh,
            residual_label,
            residual_count,
            residual_formulas,
            stress_mesh,
            stress_label,
            stress_count,
            hom_group,
            h_to_res,
            res_to_stress,
            q_star,
            d_to_stress,
            q_zero,
            stress_to_hom,
        )
        self.add(objects)


if __name__ == "__main__":
    pass
