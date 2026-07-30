#!/usr/bin/env python3
"""Standalone Manim animation for the affine initializer and PROM-ANN closure."""

from __future__ import annotations

import numpy as np
from manim import (
    AnimationGroup,
    Arrow,
    BLACK,
    Circle,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    GrowArrow,
    Indicate,
    LaggedStart,
    LEFT,
    Line,
    MathTex,
    RoundedRectangle,
    RIGHT,
    Scene,
    UP,
    VGroup,
    WHITE,
    config,
)


config.background_color = WHITE
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30

INK = "#111827"
BLUE = "#2563eb"
LIGHT_BLUE = "#93c5fd"
ORANGE = "#c2410c"


def tex(value: str, scale: float = 0.4, color: str = INK) -> MathTex:
    return MathTex(value, color=color).scale(scale)


def make_layer(count: int, center: np.ndarray, radius: float, buff: float, color: str) -> VGroup:
    layer = VGroup(*[Circle(radius=radius) for _ in range(count)]).arrange(DOWN, buff=buff).move_to(center)
    for node in layer:
        node.set_stroke(color, width=2.1).set_fill(color, opacity=0.05)
    return layer


def connections(left: VGroup, right: VGroup) -> VGroup:
    rng = np.random.default_rng(26 + len(left) * 11 + len(right))
    edges = VGroup()
    weights = rng.normal(size=(len(right), len(left)))
    weights /= np.max(np.abs(weights))
    for right_index, right_node in enumerate(right):
        for left_index, left_node in enumerate(left):
            weight = float(weights[right_index, left_index])
            color = ORANGE if weight >= 0.0 else BLUE
            edges.add(
                Line(
                    left_node.get_edge_center(RIGHT),
                    right_node.get_edge_center(LEFT),
                    color=color,
                    stroke_width=0.7 + 0.75 * abs(weight),
                    stroke_opacity=0.12 + 0.30 * abs(weight),
                )
            )
    return edges


def activate(layer: VGroup, color: str) -> AnimationGroup:
    return AnimationGroup(
        *[Indicate(node, color=color, scale_factor=1.30) for node in layer],
        lag_ratio=0.05,
    )


class PromAnnNetwork(Scene):
    def construct(self) -> None:
        input_layer = make_layer(3, np.array([-3.20, 0.10, 0.0]), 0.145, 0.42, ORANGE)
        hidden_1 = make_layer(6, np.array([-1.28, 0.10, 0.0]), 0.115, 0.20, INK)
        hidden_2 = make_layer(6, np.array([0.72, 0.10, 0.0]), 0.115, 0.20, INK)
        output_layer = make_layer(7, np.array([2.78, 0.10, 0.0]), 0.090, 0.16, BLUE)
        edge_1 = connections(input_layer, hidden_1)
        edge_2 = connections(hidden_1, hidden_2)
        edge_3 = connections(hidden_2, output_layer)
        all_edges = VGroup(edge_1, edge_2, edge_3)

        input_labels = VGroup(
            tex(r"q_1", 0.42).next_to(input_layer[0], LEFT, buff=0.16),
            tex(r"q_2", 0.42).next_to(input_layer[1], LEFT, buff=0.16),
            tex(r"q_3", 0.42).next_to(input_layer[2], LEFT, buff=0.16),
        )
        input_caption = tex(r"\mathbf q\in\mathbb R^3", 0.52, ORANGE).next_to(input_layer, UP, buff=0.36)
        network_caption = tex(r"\mathcal N_\theta(\mathbf q)", 0.70).move_to([-0.15, 2.15, 0])
        output_caption = tex(r"\overline{\mathbf q}\in\mathbb R^{36}", 0.54, BLUE).next_to(output_layer, UP, buff=0.36)

        closure_formula = tex(
            r"\widetilde{\mathbf u}(\mathbf q,\boldsymbol\mu)"
            r"\approx"
            r"\mathbf u_{\rm aff}(\boldsymbol\mu)"
            r"+\mathbf V\mathbf A\mathbf q"
            r"+\overline{\mathbf V}\mathcal N_\theta(\mathbf q)",
            0.58,
        ).move_to([0.0, -2.80, 0])

        self.play(
            FadeIn(input_caption, shift=UP * 0.08),
            FadeIn(input_labels, shift=RIGHT * 0.05),
            FadeIn(input_layer, scale=0.85),
            run_time=0.55,
        )
        self.play(
            FadeIn(network_caption, shift=UP * 0.08),
            FadeIn(output_caption, shift=UP * 0.08),
            LaggedStart(Create(edge_1), Create(edge_2), Create(edge_3), lag_ratio=0.20),
            FadeIn(VGroup(hidden_1, hidden_2, output_layer), scale=0.88),
            run_time=1.05,
        )

        for _ in range(2):
            forward = all_edges.copy().set_stroke(ORANGE, width=2.3, opacity=0.55)
            self.play(activate(input_layer, ORANGE), FadeIn(forward), run_time=0.30)
            self.play(activate(hidden_1, ORANGE), run_time=0.30)
            self.play(activate(hidden_2, ORANGE), run_time=0.30)
            self.play(activate(output_layer, ORANGE), run_time=0.30)
            self.play(FadeOut(forward), run_time=0.12)

            backward = all_edges.copy().set_stroke(BLUE, width=2.1, opacity=0.45)
            self.play(activate(output_layer, BLUE), FadeIn(backward), run_time=0.28)
            self.play(activate(hidden_2, BLUE), run_time=0.28)
            self.play(activate(hidden_1, BLUE), run_time=0.28)
            self.play(activate(input_layer, BLUE), FadeOut(backward), run_time=0.28)

        self.play(FadeIn(closure_formula, shift=UP * 0.08), run_time=0.65)
        self.wait(2.40)
