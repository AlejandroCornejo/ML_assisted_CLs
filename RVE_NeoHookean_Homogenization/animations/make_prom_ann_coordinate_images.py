#!/usr/bin/env python3
"""Static WCCM assets for the raw POD and structured affine-coordinate domains."""

from __future__ import annotations

import numpy as np
from manim import Arrow, MathTex, Scene, Text, VGroup, WHITE, config

from make_mu_to_qm_transform_animation import (
    BLACKISH,
    BLUE,
    LIGHT_BLUE,
    ORANGE,
    Q_CENTER,
    axis_triad,
    cube_edges,
    make_path_groups,
    make_raw_pod_domain,
)


config.background_color = WHITE
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30


def heading(value: str, center: np.ndarray) -> Text:
    return Text(value, font_size=29, color=BLACKISH).move_to(center)


def subheading(value: str, center: np.ndarray, color: str) -> Text:
    return Text(value, font_size=19, color=color).move_to(center)


def raw_domain(center: np.ndarray) -> VGroup:
    surface, points = make_raw_pod_domain(center)
    cube = cube_edges(center, scale=2.15).set_stroke(color="#9ca3af", width=1.2, opacity=0.62)
    axes = axis_triad(center, (r"q_{{\rm tot},1}", r"q_{{\rm tot},2}", r"q_{{\rm tot},3}"))
    return VGroup(cube, surface, points, axes)


def structured_domain(center: np.ndarray, scale_factor: float = 1.0) -> VGroup:
    _, q_paths, _, _ = make_path_groups()
    q_paths.set_stroke(color=BLUE, width=1.5, opacity=0.68)
    q_paths.shift(center - Q_CENTER)
    cube = cube_edges(center).set_stroke(color=LIGHT_BLUE, width=1.25, opacity=0.72)
    axes = axis_triad(center, (r"q_1", r"q_2", r"q_3"))
    domain = VGroup(cube, q_paths, axes)
    domain.scale(scale_factor, about_point=center)
    return domain


class FirstThreePODCoordinates(Scene):
    def construct(self) -> None:
        center = np.array([0.0, -0.35, 0.0])
        title = heading("first 3 POD coordinates", np.array([0.0, 2.72, 0.0]))
        subtitle = subheading("POD-optimal for reconstruction\nnot ideal for regression", np.array([0.0, 2.18, 0.0]), ORANGE)
        formula = MathTex(r"\mathbf Q_{\rm tot}=\mathbf V_{\rm tot}^{\top}\mathbf S", color=BLACKISH).scale(0.62).move_to([0.0, -3.22, 0.0])
        self.add(title, subtitle, raw_domain(center), formula)


class StructuredAffineCoordinates(Scene):
    def construct(self) -> None:
        center = np.array([0.0, -0.35, 0.0])
        title = heading("structured affine coordinates", np.array([0.0, 2.72, 0.0]))
        subtitle = subheading("structured coordinates\nbetter suited to regression", np.array([0.0, 2.18, 0.0]), BLUE)
        coordinates = MathTex(r"\mathbf q=(q_1,q_2,q_3)", color=BLUE).scale(0.56).move_to([0.0, 1.70, 0.0])
        least_squares = MathTex(
            r"\mathbf T=\underset{\mathbf T}{\operatorname*{arg\,min}}"
            r"\left\|\mathbf T\mathbf Q_{\rm tot}-\mathbf M\right\|_F^2",
            color=BLACKISH,
        ).scale(0.45).move_to([0.0, -3.00, 0.0])
        factorization = MathTex(
            r"\mathbf V_{\rm tot}\mathbf T^{\top}=\mathbf V\mathbf A",
            color=BLACKISH,
        ).scale(0.57).move_to([0.0, -3.55, 0.0])
        self.add(title, subtitle, coordinates, structured_domain(center, scale_factor=1.30), least_squares, factorization)


class PODToStructuredCoordinates(Scene):
    def construct(self) -> None:
        left_center = np.array([-3.20, -0.35, 0.0])
        right_center = np.array([3.20, -0.35, 0.0])
        raw_title = heading("first 3 POD coordinates", np.array([-3.20, 2.72, 0.0]))
        structured_title = heading("structured affine coordinates", np.array([3.20, 2.72, 0.0]))
        raw_subtitle = subheading("POD-optimal for reconstruction\nnot ideal for regression", np.array([-3.20, 2.18, 0.0]), ORANGE)
        structured_subtitle = subheading("structured coordinates\nbetter suited to regression", np.array([3.20, 2.18, 0.0]), BLUE)
        structured_formula = MathTex(r"\mathbf q=(q_1,q_2,q_3)", color=BLUE).scale(0.48).move_to([3.20, 1.70, 0.0])
        arrow = Arrow([-0.70, 0.40, 0.0], [0.70, 0.40, 0.0], color=ORANGE, stroke_width=3.6, buff=0.05)
        raw_formula = MathTex(r"\mathbf Q_{\rm tot}=\mathbf V_{\rm tot}^{\top}\mathbf S", color=BLACKISH).scale(0.41).move_to([-3.20, -3.25, 0.0])
        least_squares = MathTex(
            r"\mathbf T=\underset{\mathbf T}{\operatorname*{arg\,min}}"
            r"\left\|\mathbf T\mathbf Q_{\rm tot}-\mathbf M\right\|_F^2",
            color=BLACKISH,
        ).scale(0.31).move_to([3.20, -3.07, 0.0])
        factorization = MathTex(
            r"\mathbf V_{\rm tot}\mathbf T^{\top}=\mathbf V\mathbf A",
            color=BLACKISH,
        ).scale(0.42).move_to([3.20, -3.55, 0.0])
        self.add(
            raw_title,
            raw_subtitle,
            structured_title,
            structured_subtitle,
            structured_formula,
            raw_domain(left_center),
            structured_domain(right_center, scale_factor=1.30),
            arrow,
            raw_formula,
            least_squares,
            factorization,
        )
