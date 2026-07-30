#!/usr/bin/env python3
"""Animate all training samples through the affine macro-strain to q0 map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    Arrow,
    Create,
    DOWN,
    FadeIn,
    GrowArrow,
    MathTex,
    PMobject,
    RIGHT,
    Scene,
    Transform,
    UP,
    VGroup,
    VMobject,
    WHITE,
    config,
)


config.background_color = WHITE
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30

INK = "#111827"
GRAY = "#9ca3af"
BLUE = "#2563eb"
LIGHT_BLUE = "#93c5fd"
ORANGE = "#c2410c"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "stage_7_ann_data_ls"


def tex(value: str, scale: float = 0.4, color: str = INK) -> MathTex:
    return MathTex(value, color=color).scale(scale)


def normalize_paths(paths: list[np.ndarray]) -> list[np.ndarray]:
    points = np.vstack(paths)
    center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
    span = np.max(np.max(points, axis=0) - np.min(points, axis=0))
    return [(path - center[None, :]) / max(float(span), 1.0e-12) for path in paths]


def load_parameter_mesh_points() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return every actual Stage 7 parameter-mesh node and its affine image."""
    macro_points = np.load(DATA_DIR / "parameter_mesh_nodes_mu.npy")[:, :3]
    affine_a = np.load(DATA_DIR / "q_m_init_from_mu_A.npy")
    affine_b = np.load(DATA_DIR / "q_m_init_from_mu_b.npy")
    q0_points = macro_points @ affine_a + affine_b[None, :]
    return normalize_paths([macro_points]), normalize_paths([q0_points])


def project_data(center: np.ndarray, point: np.ndarray, scale: float) -> np.ndarray:
    x, y, z = np.asarray(point, dtype=float)
    return center + scale * np.array([0.88 * x + 0.46 * y, 0.28 * y + 0.86 * z, 0.0])


def polyline(points: list[np.ndarray], color: str, width: float = 2.0, opacity: float = 1.0) -> VMobject:
    line = VMobject()
    line.set_points_as_corners(points)
    return line.set_stroke(color=color, width=width, opacity=opacity)


def cube_edges(center: np.ndarray, color: str, scale: float) -> VGroup:
    corners = [(sx, sy, sz) for sx in (-0.54, 0.54) for sy in (-0.54, 0.54) for sz in (-0.28, 0.28)]
    vertices = [project_data(center, np.array(corner), scale) for corner in corners]
    edge_ids = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]
    return VGroup(*[polyline([vertices[a], vertices[b]], color, 1.25, 0.62) for a, b in edge_ids])


def axis_triad(center: np.ndarray, axes: tuple[str, str, str], scale: float) -> VGroup:
    base = center + np.array([-0.58 * scale, -0.80 * scale, 0.0])
    directions = [
        np.array([0.88, 0.00, 0.0]),
        np.array([0.46, 0.28, 0.0]),
        np.array([0.00, 0.86, 0.0]),
    ]
    triad = VGroup()
    for direction, axis in zip(directions, axes):
        direction = direction / np.linalg.norm(direction)
        end = base + 0.27 * scale * direction
        triad.add(Arrow(base, end, buff=0, color=INK, stroke_width=1.8, max_tip_length_to_length_ratio=0.18))
        triad.add(tex(axis, 0.30).next_to(end, RIGHT * 0.10 + UP * 0.02, buff=0.01))
    return triad


def domain_frame(center: np.ndarray, axes: tuple[str, str, str], color: str, scale: float) -> VGroup:
    return VGroup(cube_edges(center, color, scale), axis_triad(center, axes, scale))


def projected_cloud(paths: list[np.ndarray], center: np.ndarray, scale: float, color: str) -> PMobject:
    points = np.vstack([project_data(center, point, scale) for path in paths for point in path])
    cloud = PMobject(stroke_width=5)
    cloud.add_points(points, color=color, alpha=0.78)
    cloud.z_index = 1
    return cloud


class MuToQ0AllPoints(Scene):
    def construct(self) -> None:
        left_center = np.array([-3.10, -0.35, 0.0])
        right_center = np.array([3.10, -0.35, 0.0])
        macro_domain_scale = 3.00
        q_domain_scale = 3.90
        macro_paths, q0_paths = load_parameter_mesh_points()

        least_squares = tex(
            r"\mathbf B_{\rm aff}="
            r"\underset{\mathbf B}{\operatorname*{arg\,min}}"
            r"\left\|[\mathbf M^{\top},\mathbf 1]\mathbf B-\mathbf Q^{\top}\right\|_F^2",
            0.58,
        ).move_to([0.0, 3.22, 0.0])
        formula = tex(r"\mathbf q^0(\boldsymbol\mu)=[\boldsymbol\mu,1]\mathbf B_{\rm aff}", 0.68)
        formula.move_to([0.0, 2.37, 0.0])
        left_caption = tex(r"\boldsymbol\mu=(E_{xx},E_{yy},G_{xy})", 0.48).move_to(left_center + np.array([0.0, 1.55, 0.0]))
        right_caption = tex(r"\mathbf q^0=(q_1^0,q_2^0,q_3^0)", 0.48, BLUE).move_to(right_center + np.array([0.0, 1.70, 0.0]))
        macro_domain = domain_frame(left_center, (r"E_{xx}", r"E_{yy}", r"G_{xy}"), GRAY, macro_domain_scale)
        q_domain = domain_frame(right_center, (r"q_1", r"q_2", r"q_3"), LIGHT_BLUE, q_domain_scale)
        macro_cloud = projected_cloud(macro_paths, left_center, macro_domain_scale, GRAY)
        q_cloud = projected_cloud(q0_paths, right_center, q_domain_scale, BLUE)

        affine_arrow = Arrow(
            np.array([-0.72, 1.58, 0.0]),
            np.array([0.72, 1.58, 0.0]),
            buff=0.08,
            color=ORANGE,
            stroke_width=3.6,
            max_tip_length_to_length_ratio=0.14,
        )
        affine_label = tex(r"\mathbf B_{\rm aff}", 0.38, ORANGE).next_to(affine_arrow, DOWN, buff=0.08)

        self.add(least_squares, formula, macro_cloud)
        self.play(
            FadeIn(VGroup(left_caption, right_caption), shift=UP * 0.06),
            Create(macro_domain),
            Create(q_domain),
            run_time=1.00,
        )
        self.wait(1.80)
        self.play(GrowArrow(affine_arrow), FadeIn(affine_label), run_time=0.72)
        self.play(Transform(macro_cloud, q_cloud), run_time=2.50)
        self.wait(2.40)
