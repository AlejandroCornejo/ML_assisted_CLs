#!/usr/bin/env python3
"""Manim workflow animation for the HPROM-ANN and D-HPROM-ANN online paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    AnimationGroup,
    Arrow,
    BLUE_E,
    Create,
    CurvedArrow,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    GRAY_B,
    GRAY_C,
    Group,
    GrowArrow,
    ImageMobject,
    Indicate,
    LaggedStart,
    LEFT,
    Line,
    MathTex,
    ORANGE,
    RIGHT,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    UP,
    VGroup,
    VMobject,
    WHITE,
    config,
)


config.background_color = WHITE
config.pixel_width = 1280
config.pixel_height = 720
config.frame_rate = 30

SCRIPT_DIR = Path(__file__).resolve().parent
RESIDUAL_SUPPORT = SCRIPT_DIR / "residual_support_10_elements.png"
STRESS_SUPPORT = SCRIPT_DIR / "stress_support_10_elements.png"

# Match the existing PROM-ANN coordinate animation exactly.
INK = "#111827"
MUTED = "#9ca3af"
BLUE = "#2563eb"
BLUE_FILL = "#eff6ff"
ORANGE_DARK = "#c2410c"
ORANGE_FILL = "#fff7ed"


def formula(tex: str, scale: float = 0.42, color: str = INK) -> MathTex:
    return MathTex(tex, color=color).scale(scale)


def label(text: str, size: int = 18, color: str = INK, bold: bool = False) -> Text:
    return Text(text, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def make_panel(title: str, width: float, height: float, edge: str, fill: str, center: np.ndarray) -> tuple[RoundedRectangle, Text]:
    frame = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.16,
        stroke_color=edge,
        stroke_width=2.6,
        fill_color=fill,
        fill_opacity=0.68,
    ).move_to(center)
    title_mobject = label(title, 17, edge, bold=True)
    title_mobject.move_to(frame.get_top() + DOWN * 0.26 + RIGHT * (0.5 * title_mobject.width - 0.5 * width + 0.18))
    return frame, title_mobject


def polyline(points: list[np.ndarray], color: str, width: float = 3.2) -> VMobject:
    path = VMobject()
    path.set_points_as_corners(points)
    path.set_stroke(color=color, width=width)
    return path


def token(tex: str, color: str, point: np.ndarray) -> VGroup:
    dot = Dot(point, radius=0.115, color=color, fill_opacity=0.08, stroke_width=3.0)
    text = formula(tex, scale=0.30, color=color).move_to(dot)
    return VGroup(dot, text)


def make_domain(center: np.ndarray, axis_labels: tuple[str, str, str], color: str, structured: bool) -> VGroup:
    def project(point: tuple[float, float, float]) -> np.ndarray:
        x, y, z = point
        return center + np.array([0.76 * x + 0.44 * y, 0.25 * y - 0.82 * z, 0.0])

    corners = [
        (sx, sy, sz)
        for sx in (-0.72, 0.72)
        for sy in (-0.72, 0.72)
        for sz in (-0.46, 0.46)
    ]
    projected = [project(corner) for corner in corners]
    edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]
    cube = VGroup(*[Line(projected[left], projected[right], color=color, stroke_width=1.25, stroke_opacity=0.60) for left, right in edges])

    trajectories = VGroup()
    for z_value in np.linspace(-0.26, 0.26, 5):
        if structured:
            vertices = [
                project((-0.58, -0.58, z_value)),
                project((0.58, -0.58, z_value)),
                project((0.58, 0.58, z_value)),
                project((-0.58, 0.58, z_value)),
                project((-0.58, -0.58, z_value)),
            ]
            trajectories.add(polyline(vertices, color, width=1.7).set_stroke(opacity=0.62))
        else:
            vertices = [
                project((x_value, 0.34 * np.sin(3.1 * x_value + 3.0 * z_value), z_value))
                for x_value in np.linspace(-0.58, 0.58, 20)
            ]
            trajectories.add(polyline(vertices, ORANGE_DARK, width=1.7).set_stroke(opacity=0.72))

    base = center + np.array([-0.64, -0.86, 0.0])
    ends = [base + RIGHT * 0.40, base + np.array([0.26, 0.18, 0.0]), base + UP * 0.38]
    triad = VGroup()
    for end, axis_label in zip(ends, axis_labels):
        triad.add(Arrow(base, end, buff=0, color=INK, stroke_width=1.7, max_tip_length_to_length_ratio=0.20))
        triad.add(formula(axis_label, scale=0.25).next_to(end, RIGHT * 0.15 + UP * 0.02, buff=0.02))
    return VGroup(cube, trajectories, triad)


class HpromDhpromWorkflow(Scene):
    def construct(self) -> None:
        # Left: macro input and the affine coordinate initialization.  The
        # preceding coordinate-model movie already introduces the 3D domains,
        # so this scene keeps the handoff compact and makes the two branches clear.
        macro_frame, macro_title = make_panel("macro input", 1.90, 1.65, BLUE, BLUE_FILL, np.array([-5.95, 0.18, 0.0]))
        f_value = formula(r"\overline{\mathbf F}", 0.50).move_to([-5.95, 0.45, 0])
        e_value = formula(r"\overline{\mathbf E}=\frac12(\overline{\mathbf F}^T\overline{\mathbf F}-\mathbf I)", 0.29).move_to([-5.95, -0.04, 0])
        mu_value = formula(r"\boldsymbol\mu=(E_{xx},E_{yy},G_{xy})", 0.30).move_to([-5.95, -0.48, 0])
        macro_arrows = VGroup(
            Arrow(f_value.get_bottom(), e_value.get_top(), buff=0.04, color=BLUE, stroke_width=2.3, max_tip_length_to_length_ratio=0.18),
            Arrow(e_value.get_bottom(), mu_value.get_top(), buff=0.06, color=BLUE, stroke_width=2.3, max_tip_length_to_length_ratio=0.18),
        )
        macro_group = VGroup(macro_frame, macro_title, f_value, e_value, mu_value, macro_arrows)

        affine_frame, affine_title = make_panel(
            "affine initialization",
            2.86,
            1.42,
            BLUE,
            BLUE_FILL,
            np.array([-3.38, 0.18, 0.0]),
        )
        affine_formula = formula(r"\mathbf q^0(\boldsymbol\mu)=[\boldsymbol\mu,1]\mathbf B_{\rm aff}", 0.39).move_to([-3.38, 0.10, 0])
        affine_group = VGroup(affine_frame, affine_title, affine_formula)
        macro_to_affine = Arrow(macro_frame.get_right(), affine_frame.get_left(), buff=0.10, color=BLUE, stroke_width=3.0)

        # Upper orange route: intrusive HPROM correction on the residual support.
        hprom_frame, hprom_title = make_panel("HPROM-ANN: Galerkin", 3.16, 2.38, ORANGE_DARK, ORANGE_FILL, np.array([0.00, 1.83, 0.0]))
        residual_mesh = ImageMobject(str(RESIDUAL_SUPPORT)).set_width(1.42).move_to([-0.72, 1.66, 0])
        residual_caption = label("10 selected elements", 12, ORANGE_DARK).next_to(residual_mesh, DOWN, buff=0.04)
        residual_formula = formula(r"\mathbf W(\mathbf q)^T\mathbf r", 0.41).move_to([0.68, 2.10, 0])
        residual_sum = formula(r"\approx\sum_{e\in\mathcal Z_{\rm res}}\xi_e^{\rm res}\,\mathbf W_e(\mathbf q)^T\mathbf r_e", 0.235).move_to([0.68, 1.75, 0])
        update_caption = label("Galerkin update", 15, ORANGE_DARK, bold=True).move_to([0.68, 1.31, 0])
        update_formula = formula(r"\mathbf q^{k+1}=\mathbf q^k+\Delta\mathbf q^k", 0.30).move_to([0.68, 1.02, 0])
        correction_loop = CurvedArrow(np.array([1.18, 0.84, 0]), np.array([0.96, 1.52, 0]), angle=-1.25, color=ORANGE_DARK, stroke_width=2.6)
        hprom_group = Group(hprom_frame, hprom_title, residual_mesh, residual_caption, residual_formula, residual_sum, update_caption, update_formula, correction_loop)

        # Lower blue route: accepts q=q0 and does not visit the residual support.
        direct_frame, direct_title = make_panel("D-HPROM-ANN: direct state", 3.16, 1.40, BLUE, BLUE_FILL, np.array([0.00, -1.73, 0.0]))
        direct_formula = formula(r"\mathbf q=\mathbf q^0(\boldsymbol\mu)", 0.42).move_to([0.00, -1.55, 0])
        direct_caption = label("skip residual correction", 16, BLUE, bold=True).move_to([0.00, -1.97, 0])
        direct_note = label("one PROM-ANN state evaluation", 13, MUTED).move_to([0.00, -2.26, 0])
        direct_group = VGroup(direct_frame, direct_title, direct_formula, direct_caption, direct_note)

        # Shared state map and shared hyper-reduced stress evaluation.
        decoder_frame, decoder_title = make_panel("PROM-ANN state map", 2.86, 1.88, BLUE, BLUE_FILL, np.array([3.70, 0.62, 0.0]))
        decoder_formula_1 = formula(r"\widetilde{\mathbf u}(\mathbf q,\boldsymbol\mu)", 0.41).move_to([3.70, 0.91, 0])
        decoder_formula_2 = formula(r"=\mathbf u_{\rm aff}(\boldsymbol\mu)+\mathbf V\mathbf A\mathbf q", 0.27).move_to([3.70, 0.53, 0])
        decoder_formula_3 = formula(r"+\overline{\mathbf V}\mathcal N_\theta(\mathbf q)", 0.29).move_to([3.70, 0.18, 0])
        decoder_caption = label("nonlinear RVE state", 13, MUTED).move_to([3.70, -0.03, 0])
        decoder_group = VGroup(decoder_frame, decoder_title, decoder_formula_1, decoder_formula_2, decoder_formula_3, decoder_caption)

        stress_frame, _ = make_panel("", 2.55, 2.62, BLUE, BLUE_FILL, np.array([5.52, -2.14, 0.0]))
        stress_title = label("hyper-reduced\nhomogenization", 14, BLUE, bold=True).move_to(stress_frame.get_top() + DOWN * 0.34)
        stress_mesh = ImageMobject(str(STRESS_SUPPORT)).set_width(1.22).move_to([4.91, -2.18, 0])
        stress_caption = label("10 stress elements", 12, BLUE).next_to(stress_mesh, DOWN, buff=0.04)
        stress_value = formula(r"\overline{\mathbf S}", 0.48).move_to([6.05, -1.73, 0])
        stress_formula = formula(r"\approx\frac{1}{A_0}\sum_{e\in\mathcal Z_\sigma}\xi_e^\sigma\,A_e\langle\mathbf S_e\rangle", 0.23).move_to([6.05, -2.12, 0])
        stress_caption_2 = label("homogenized stress", 13, MUTED).move_to([6.05, -2.52, 0])
        stress_group = Group(stress_frame, stress_title, stress_mesh, stress_caption, stress_value, stress_formula, stress_caption_2)

        # The final routing arrows are deliberately separate so the paths are visible.
        q_split = affine_frame.get_right() + RIGHT * 0.10
        split_to_hprom = Arrow(q_split, hprom_frame.get_left() + UP * 0.48, buff=0.08, color=ORANGE_DARK, stroke_width=3.4)
        split_to_direct = Arrow(q_split, direct_frame.get_left() + UP * 0.18, buff=0.08, color=BLUE, stroke_width=3.4)
        hprom_to_decoder = Arrow(hprom_frame.get_right() + DOWN * 0.32, decoder_frame.get_left() + UP * 0.35, buff=0.08, color=ORANGE_DARK, stroke_width=3.4)
        direct_to_decoder = Arrow(direct_frame.get_right() + UP * 0.18, decoder_frame.get_left() + DOWN * 0.36, buff=0.08, color=BLUE, stroke_width=3.4)
        decoder_to_stress = Arrow(decoder_frame.get_bottom() + RIGHT * 0.20, stress_frame.get_top() + LEFT * 0.37, buff=0.08, color=BLUE, stroke_width=3.4)
        route_labels = VGroup(
            formula(r"\mathbf q^0", 0.30, ORANGE_DARK).next_to(split_to_hprom, LEFT, buff=0.04),
            formula(r"\mathbf q^0", 0.30, BLUE).next_to(split_to_direct, LEFT, buff=0.04),
            formula(r"\mathbf q^*", 0.30, ORANGE_DARK).next_to(hprom_to_decoder, UP, buff=0.04),
        )
        final_note_frame = RoundedRectangle(width=8.85, height=0.44, corner_radius=0.12, stroke_color=GRAY_B, stroke_width=1.0, fill_color="#F5F7FA", fill_opacity=0.96).move_to([0.30, -3.42, 0])
        final_note = label("same decoder and stress support; only HPROM-ANN performs online residual correction", 14, INK).move_to(final_note_frame)
        final_note_group = VGroup(final_note_frame, final_note)

        # 1. Macro input and the affine q0 map.
        self.play(FadeIn(macro_group, shift=UP * 0.10), run_time=0.75)
        self.wait(5.0)
        self.play(
            GrowArrow(macro_to_affine),
            FadeIn(affine_group, shift=RIGHT * 0.08),
            run_time=1.10,
        )
        q0_dot = token(r"\mathbf q^0", ORANGE_DARK, affine_frame.get_right() + LEFT * 0.36 + DOWN * 0.18)
        self.play(FadeIn(q0_dot, scale=0.85), run_time=0.50)
        self.wait(7.0)

        # 2. Orange intrusive route: residual support -> correction -> q*.
        self.play(
            FadeIn(hprom_group, shift=UP * 0.12),
            FadeIn(decoder_group, shift=RIGHT * 0.12),
            GrowArrow(split_to_hprom),
            GrowArrow(hprom_to_decoder),
            FadeIn(route_labels[0]),
            run_time=1.10,
        )
        hprom_token = token(r"\mathbf q^0", ORANGE_DARK, split_to_hprom.get_start())
        self.play(Transform(q0_dot, hprom_token), run_time=0.40)
        self.play(hprom_token.animate.move_to(residual_mesh.get_center() + UP * 0.22), run_time=1.30)
        self.play(Indicate(residual_mesh, color=ORANGE_DARK, scale_factor=1.05), run_time=0.75)
        q1_token = token(r"\mathbf q^1", ORANGE_DARK, np.array([0.86, 0.85, 0.0]))
        self.play(Transform(hprom_token, q1_token), Indicate(correction_loop, color=ORANGE_DARK), run_time=0.90)
        self.play(hprom_token.animate.move_to(residual_mesh.get_center() + DOWN * 0.16), run_time=0.85)
        self.play(Indicate(residual_mesh, color=ORANGE_DARK, scale_factor=1.07), run_time=0.75)
        qstar_token = token(r"\mathbf q^*", ORANGE_DARK, hprom_to_decoder.get_start())
        self.play(Transform(hprom_token, qstar_token), FadeIn(route_labels[2]), run_time=0.90)
        self.play(hprom_token.animate.move_to(decoder_frame.get_left() + UP * 0.35), run_time=1.10)
        self.play(Indicate(decoder_frame, color=ORANGE_DARK, scale_factor=1.02), run_time=0.65)
        self.play(FadeOut(hprom_token), run_time=0.25)
        self.wait(8.0)

        # 3. Blue direct route: q=q0, no residual loop, one state evaluation.
        self.play(
            FadeIn(direct_group, shift=DOWN * 0.10),
            GrowArrow(split_to_direct),
            GrowArrow(direct_to_decoder),
            FadeIn(route_labels[1]),
            run_time=0.95,
        )
        direct_token = token(r"\mathbf q^0", BLUE, split_to_direct.get_start())
        self.play(FadeIn(direct_token, scale=0.85), run_time=0.25)
        self.play(direct_token.animate.move_to(direct_frame.get_center() + UP * 0.10), run_time=1.15)
        self.play(Indicate(direct_frame, color=BLUE, scale_factor=1.03), run_time=0.65)
        self.play(direct_token.animate.move_to(decoder_frame.get_left() + DOWN * 0.36), run_time=1.30)
        self.play(Indicate(decoder_frame, color=BLUE, scale_factor=1.02), run_time=0.65)
        self.play(FadeOut(direct_token), run_time=0.25)
        self.wait(8.0)

        # 4. Shared stress support and final comparison cue.
        self.play(FadeIn(stress_group, shift=DOWN * 0.12), GrowArrow(decoder_to_stress), run_time=1.10)
        state_token = token(r"\widetilde{\mathbf u}", BLUE, decoder_to_stress.get_start())
        self.play(FadeIn(state_token, scale=0.85), run_time=0.25)
        self.play(state_token.animate.move_to(stress_mesh.get_center() + UP * 0.12), run_time=1.35)
        self.play(Indicate(stress_mesh, color=BLUE, scale_factor=1.06), run_time=0.80)
        stress_token = token(r"\overline{\mathbf S}", BLUE, stress_value.get_center())
        self.play(Transform(state_token, stress_token), Indicate(stress_value, color=BLUE, scale_factor=1.10), run_time=0.90)
        self.play(FadeOut(state_token), FadeIn(final_note_group, shift=UP * 0.06), run_time=0.70)
        self.wait(10.0)
