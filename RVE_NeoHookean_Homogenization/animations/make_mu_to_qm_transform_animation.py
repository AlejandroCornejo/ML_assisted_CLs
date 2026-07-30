#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Manim scene for the PROM-ANN coordinate animation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import *
from scipy.spatial import Delaunay


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
STAGE0_FILE = PROJECT_DIR / "stage_0_trajectory" / "stage_0_trajectories.npz"
DATA_DIR = PROJECT_DIR / "stage_7_ann_data_ls"

MACRO_CENTER = np.array([-4.85, -0.35, 0.0])
Q_CENTER = np.array([-1.55, -0.35, 0.0])
OPEN_RAW_CENTER = np.array([-3.70, -0.35, 0.0])
Q_PREVIEW_SHIFT = np.array([2.90, 0.0, 0.0])
HIGHLIGHT_TRAJECTORIES = {1, 4, 5, 8, 9}

BLACKISH = "#111827"
GRAY = "#9ca3af"
BLUE = "#2563eb"
LIGHT_BLUE = "#93c5fd"
ORANGE = "#c2410c"


def load_stage0_paths() -> list[np.ndarray]:
    data = np.load(STAGE0_FILE)
    count = int(np.ravel(data["trajectory_count"])[0])
    return [np.asarray(data[f"trajectory_{i}"], dtype=float) for i in range(1, count + 1)]


def normalize_paths(paths: list[np.ndarray]) -> list[np.ndarray]:
    pts = np.vstack(paths)
    center = 0.5 * (np.min(pts, axis=0) + np.max(pts, axis=0))
    span = np.max(np.max(pts, axis=0) - np.min(pts, axis=0))
    return [(p - center[None, :]) / max(float(span), 1.0e-12) for p in paths]


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    lo = np.percentile(pts, 2.0, axis=0)
    hi = np.percentile(pts, 98.0, axis=0)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 1.0e-12)
    return np.clip((pts - center[None, :]) / span[None, :], -0.62, 0.62)


def project_point(point: np.ndarray, center: np.ndarray, scale: float = 1.78) -> np.ndarray:
    x, y, z = np.asarray(point, dtype=float)
    px = center[0] + scale * (0.88 * x + 0.46 * y)
    py = center[1] + scale * (0.00 * x + 0.28 * y + 0.86 * z)
    return np.array([float(px), float(py), 0.0])


def project_path(path: np.ndarray, center: np.ndarray, scale: float = 1.78) -> list[np.ndarray]:
    return [project_point(p, center, scale) for p in np.asarray(path, dtype=float)]


def make_polyline(points: list[np.ndarray], color: str, width: float, opacity: float) -> VMobject:
    line = VMobject()
    line.set_points_as_corners(points)
    line.set_stroke(color=color, width=width, opacity=opacity)
    return line


def cube_edges(center: np.ndarray, scale: float = 1.78) -> VGroup:
    corners = np.array(
        [
            [sx, sy, sz]
            for sx in (-0.54, 0.54)
            for sy in (-0.54, 0.54)
            for sz in (-0.28, 0.28)
        ],
        dtype=float,
    )
    points = project_path(corners, center, scale)
    edge_ids = [
        (0, 1),
        (0, 2),
        (0, 4),
        (3, 1),
        (3, 2),
        (3, 7),
        (5, 1),
        (5, 4),
        (5, 7),
        (6, 2),
        (6, 4),
        (6, 7),
    ]
    return VGroup(*[Line(points[i], points[j]) for i, j in edge_ids])


def unit_direction(axis: np.ndarray) -> np.ndarray:
    start = project_point(np.zeros(3), np.zeros(3), scale=1.0)
    end = project_point(axis, np.zeros(3), scale=1.0)
    direction = end - start
    return direction / max(np.linalg.norm(direction), 1.0e-12)


def axis_triad(center: np.ndarray, labels: tuple[str, str, str]) -> VGroup:
    base = center + np.array([-1.03, -1.42, 0.0])
    directions = [
        unit_direction(np.array([1.0, 0.0, 0.0])),
        unit_direction(np.array([0.0, 1.0, 0.0])),
        unit_direction(np.array([0.0, 0.0, 1.0])),
    ]
    offsets = [
        np.array([0.12, -0.05, 0.0]),
        np.array([0.08, 0.07, 0.0]),
        np.array([0.03, 0.12, 0.0]),
    ]
    group = VGroup()
    for direction, label, offset in zip(directions, labels, offsets):
        end = base + 0.46 * direction
        group.add(Arrow(base, end, buff=0, color=BLACK, stroke_width=2.2, max_tip_length_to_length_ratio=0.18))
        group.add(MathTex(label, color=BLACK).scale(0.34).move_to(end + offset))
    return group


def pca_surface_mesh(points: np.ndarray, u_bins: int = 18, v_bins: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Build a sparse 2D Delaunay mesh in the first two POD-cloud principal directions."""
    centered = np.asarray(points, dtype=float) - np.mean(points, axis=0, keepdims=True)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ basis.T
    display_points = normalize_point_cloud(points)
    uv_scale = np.maximum(np.percentile(coordinates[:, :2], 98.0, axis=0) - np.percentile(coordinates[:, :2], 2.0, axis=0), 1.0e-12)
    uv = coordinates[:, :2] / uv_scale[None, :]

    u_edges = np.linspace(np.min(uv[:, 0]), np.max(uv[:, 0]), u_bins + 1)
    v_edges = np.linspace(np.min(uv[:, 1]), np.max(uv[:, 1]), v_bins + 1)
    selected: list[int] = []
    for i in range(u_bins):
        for j in range(v_bins):
            mask = (
                (uv[:, 0] >= u_edges[i])
                & (uv[:, 0] <= u_edges[i + 1])
                & (uv[:, 1] >= v_edges[j])
                & (uv[:, 1] <= v_edges[j + 1])
            )
            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                continue
            target = np.array([(u_edges[i] + u_edges[i + 1]) / 2.0, (v_edges[j] + v_edges[j + 1]) / 2.0])
            nearest = candidates[np.argmin(np.sum((uv[candidates] - target[None, :]) ** 2, axis=1))]
            selected.append(int(nearest))

    selected_ids = np.array(sorted(set(selected)), dtype=np.int64)
    sampled_uv = uv[selected_ids]
    simplices = Delaunay(sampled_uv).simplices
    triangle_points = sampled_uv[simplices]
    edge_lengths = np.stack(
        [
            np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 1], axis=1),
            np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 2], axis=1),
            np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 0], axis=1),
        ],
        axis=1,
    )
    return display_points[selected_ids], simplices[np.max(edge_lengths, axis=1) < 0.22]


def make_raw_pod_domain(center: np.ndarray) -> tuple[VGroup, VGroup]:
    points = np.load(DATA_DIR / "parameter_mesh_nodes_q_pod.npy")[:, :3]
    nodes, faces = pca_surface_mesh(points)

    surface_group = VGroup(
        *[
            Polygon(*[project_point(nodes[index], center, scale=2.15) for index in face])
            .set_fill(ORANGE, opacity=0.30)
            .set_stroke(ORANGE, width=0.0, opacity=0.0)
            for face in faces
        ]
    )

    display_points = normalize_point_cloud(points)
    rng = np.random.default_rng(71)
    point_ids = np.sort(rng.choice(display_points.shape[0], size=min(700, display_points.shape[0]), replace=False))
    point_cloud = VGroup(
        *[
            Dot(project_point(display_points[index], center, scale=2.15), radius=0.013, color="#9a3412")
            for index in point_ids
        ]
    ).set_opacity(0.78)
    return surface_group, point_cloud


def pulse_layer(layer: VGroup, color: str, scale_factor: float = 1.52) -> LaggedStart:
    return LaggedStart(
        *[Indicate(node, color=color, scale_factor=scale_factor) for node in layer],
        lag_ratio=0.035,
    )


def make_path_groups() -> tuple[VGroup, VGroup, VGroup, VGroup]:
    macro_raw = load_stage0_paths()
    a_mu = np.load(DATA_DIR / "q_m_init_from_mu_A.npy")
    b_mu = np.load(DATA_DIR / "q_m_init_from_mu_b.npy")
    q_raw = [path @ a_mu + b_mu[None, :] for path in macro_raw]

    macro_paths = [project_path(path, MACRO_CENTER) for path in normalize_paths(macro_raw)]
    q_paths = [project_path(path, Q_CENTER) for path in normalize_paths(q_raw)]

    macro_background = VGroup()
    q_background = VGroup()
    macro_highlight = VGroup()
    q_highlight = VGroup()
    for i, (macro_path, q_path) in enumerate(zip(macro_paths, q_paths), start=1):
        macro_background.add(make_polyline(macro_path, GRAY, 1.15, 0.32))
        q_background.add(make_polyline(q_path, LIGHT_BLUE, 1.15, 0.42))
        if i in HIGHLIGHT_TRAJECTORIES:
            macro_highlight.add(make_polyline(macro_path, ORANGE, 2.7, 0.98))
            q_highlight.add(make_polyline(q_path, ORANGE, 2.7, 0.98))
    return macro_background, q_background, macro_highlight, q_highlight


def make_network() -> tuple[list[VGroup], VGroup, VGroup, list[np.ndarray]]:
    input_layer = VGroup(*[Circle(radius=0.105) for _ in range(3)]).arrange(DOWN, buff=0.33).move_to([2.10, 0.0, 0])
    hidden_1 = VGroup(*[Circle(radius=0.095) for _ in range(6)]).arrange(DOWN, buff=0.20).move_to([3.12, 0.0, 0])
    hidden_2 = VGroup(*[Circle(radius=0.095) for _ in range(6)]).arrange(DOWN, buff=0.20).move_to([4.20, 0.0, 0])
    output_nodes = VGroup(*[Circle(radius=0.080) for _ in range(7)]).arrange(DOWN, buff=0.18).move_to([5.38, 0.0, 0])

    for node in input_layer:
        node.set_stroke(ORANGE, width=2.1).set_fill(ORANGE, opacity=0.08)
    for layer in (hidden_1, hidden_2):
        for node in layer:
            node.set_stroke(BLACKISH, width=1.45).set_fill(BLUE, opacity=0.06)
    for node in output_nodes:
        node.set_stroke(BLUE, width=1.7).set_fill(BLUE, opacity=0.06)

    rng = np.random.default_rng(29)
    edge_layers: list[VGroup] = []
    for left, right in [(input_layer, hidden_1), (hidden_1, hidden_2), (hidden_2, output_nodes)]:
        weights = rng.normal(size=(len(right), len(left)))
        weights /= np.max(np.abs(weights))
        edges = VGroup()
        for right_index, right_node in enumerate(right):
            for left_index, left_node in enumerate(left):
                weight = float(weights[right_index, left_index])
                color = ORANGE if weight >= 0.0 else BLUE
                edges.add(
                    Line(
                        left_node.get_edge_center(RIGHT),
                        right_node.get_edge_center(LEFT),
                        color=color,
                        stroke_width=0.70 + 0.45 * abs(weight),
                        stroke_opacity=0.10 + 0.32 * abs(weight),
                    )
                )
        edge_layers.append(edges)

    nodes = VGroup(input_layer, hidden_1, hidden_2, output_nodes)
    input_labels = VGroup(
        MathTex(r"q_1", color=BLACK).scale(0.38).next_to(input_layer[0], LEFT, buff=0.14),
        MathTex(r"q_2", color=BLACK).scale(0.38).next_to(input_layer[1], LEFT, buff=0.14),
        MathTex(r"q_3", color=BLACK).scale(0.38).next_to(input_layer[2], LEFT, buff=0.14),
    )
    activations = [
        np.array([0.88, 0.42, 0.72]),
        np.array([0.20, 0.76, 0.54, 0.92, 0.34, 0.66]),
        np.array([0.70, 0.27, 0.88, 0.44, 0.62, 0.38]),
        np.array([0.58, 0.26, 0.84, 0.40, 0.70, 0.32, 0.62]),
    ]
    return edge_layers, nodes, input_labels, activations


def activate_layer(layer: VGroup, values: np.ndarray, color: str) -> AnimationGroup:
    return AnimationGroup(
        *[
            node.animate.set_fill(color, opacity=0.14 + 0.78 * float(value))
            for node, value in zip(layer, values)
        ],
        lag_ratio=0.07,
    )


def pod_matrix_block(
    center: np.ndarray,
    width: float,
    height: float,
    color: str,
    symbol: str,
    dimension: str,
    coordinates: str | None = None,
) -> VGroup:
    block = Rectangle(width=width, height=height, stroke_color=color, stroke_width=2.0)
    block.set_fill(color, opacity=0.58).move_to(center)
    symbol_label = MathTex(symbol, color=BLACK).scale(0.62).move_to(block)
    dimension_label = MathTex(dimension, color=BLACKISH).scale(0.40).next_to(block, DOWN, buff=0.13)
    group = VGroup(block, symbol_label, dimension_label)
    if coordinates is not None:
        coordinate_label = MathTex(coordinates, color=BLACKISH).scale(0.40).next_to(dimension_label, DOWN, buff=0.12)
        group.add(coordinate_label)
    return group


class PromAnnCoordinatePipeline(Scene):
    def construct(self) -> None:
        self.camera.background_color = WHITE

        qbar_dim = int(np.load(DATA_DIR / "q_s_train.npy").shape[1])
        n_total = 3 + qbar_dim
        macro_background, q_background, macro_highlight, q_highlight = make_path_groups()
        raw_pod_domain, raw_pod_points = make_raw_pod_domain(OPEN_RAW_CENTER)
        raw_cube = cube_edges(OPEN_RAW_CENTER, scale=2.15).set_stroke(GRAY, width=1.2, opacity=0.62)
        raw_axes = axis_triad(OPEN_RAW_CENTER, (r"q_1", r"q_2", r"q_3"))
        macro_cube = cube_edges(MACRO_CENTER).set_stroke(GRAY, width=1.2, opacity=0.62)
        q_cube = cube_edges(Q_CENTER).set_stroke(LIGHT_BLUE, width=1.2, opacity=0.68)
        macro_axes = axis_triad(MACRO_CENTER, (r"E_{xx}", r"E_{yy}", r"G_{xy}"))
        q_axes = axis_triad(Q_CENTER, (r"q_1", r"q_2", r"q_3"))
        network_edge_layers, network_nodes, input_labels, activation_values = make_network()

        title = Text("PROM-ANN coordinate model", font_size=30, color=BLACKISH, weight=BOLD)
        title.to_edge(UP, buff=0.22)

        total_basis = pod_matrix_block(
            np.array([0.0, -0.10, 0.0]),
            width=2.85,
            height=2.65,
            color="#94a3b8",
            symbol=r"\mathbf V_{\rm tot}",
            dimension=rf"N\times {n_total}",
        )
        primary_basis = pod_matrix_block(
            np.array([-0.95, -0.10, 0.0]),
            width=0.62,
            height=2.65,
            color="#6366f1",
            symbol=r"\mathbf V",
            dimension=r"N\times 3",
            coordinates=r"\mathbf q\in\mathbb R^3",
        )
        secondary_basis = pod_matrix_block(
            np.array([0.95, -0.10, 0.0]),
            width=1.75,
            height=2.65,
            color="#dc7f85",
            symbol=r"\overline{\mathbf V}",
            dimension=rf"N\times {qbar_dim}",
            coordinates=rf"\overline{{\mathbf q}}\in\mathbb R^{{{qbar_dim}}}",
        )
        parameter_dimension = MathTex(
            r"\boldsymbol\mu\in\mathbb R^3\ \Longrightarrow\ \mathbf V\in\mathbb R^{N\times3}",
            color=BLACK,
        ).scale(0.54).move_to([0.0, 2.25, 0])
        partition_formula = MathTex(
            r"\mathbf V_{\rm tot}=[\,\mathbf V\ \ \overline{\mathbf V}\,]",
            color=BLACKISH,
        ).scale(0.54).move_to([0.0, -2.58, 0])
        pod_intro_group = VGroup(total_basis, primary_basis, secondary_basis, parameter_dimension, partition_formula)

        macro_label = MathTex(r"\boldsymbol\mu=(E_{xx},E_{yy},G_{xy})", color=BLACK).scale(0.58)
        macro_label.move_to([-4.85, 2.42, 0])
        q_label = MathTex(r"\mathbf q=(q_1,q_2,q_3)", color=BLACK).scale(0.58)
        q_label.move_to([-1.55, 2.42, 0])
        raw_label = Text("first 3 POD coordinates", font_size=21, color=BLACKISH)
        raw_label.move_to(OPEN_RAW_CENTER + np.array([0.0, 2.77, 0.0]))
        raw_note = Tex(r"curved manifold", color=ORANGE).scale(0.44)
        raw_note.move_to(OPEN_RAW_CENTER + np.array([0.0, 2.40, 0.0]))
        q_preview_label = Tex(r"seek: structured coordinates", color=BLACKISH).scale(0.40)
        q_preview_label.move_to(Q_CENTER + Q_PREVIEW_SHIFT + np.array([0.70, 2.77, 0.0]))
        q_preview_note = MathTex(r"\mathbf q=(q_1,q_2,q_3)", color=BLACK).scale(0.54)
        q_preview_note.move_to(Q_CENTER + Q_PREVIEW_SHIFT + np.array([0.0, 2.39, 0.0]))

        q_preview_group = VGroup(q_cube, q_axes, q_background)
        q_preview_group.shift(Q_PREVIEW_SHIFT)
        q_background.set_stroke(color=BLUE, width=1.35, opacity=0.62)
        raw_to_target = Arrow(
            start=[-2.05, 0.45, 0],
            end=[-0.10, 0.45, 0],
            buff=0.05,
            color=ORANGE,
            stroke_width=3.6,
        )
        raw_to_target_label = Tex(r"seek", color=ORANGE).scale(0.38).next_to(raw_to_target, UP, buff=0.10)

        opt_box = RoundedRectangle(
            corner_radius=0.13,
            width=2.76,
            height=0.88,
            stroke_color=BLACK,
            stroke_width=1.6,
            fill_color=WHITE,
            fill_opacity=0.94,
        ).move_to([-3.20, 1.25, 0])
        opt_text = VGroup(
            MathTex(r"\min_{\mathbf T}\|\mathbf Q\mathbf T^T-\mathbf M\|_F\Rightarrow\mathbf A", color=BLACK).scale(0.34),
            MathTex(r"\mathbf q^0(\boldsymbol\mu)=[\boldsymbol\mu,1]\mathbf B_{\rm aff}", color=BLACK).scale(0.38),
        ).arrange(DOWN, buff=0.11).move_to(opt_box)
        opt_caption = Tex(r"fit linear combinations to macro strains", color=BLACKISH).scale(0.31)
        opt_caption.next_to(opt_box, DOWN, buff=0.08)
        opt_group = VGroup(opt_box, opt_text, opt_caption)

        arrow_macro_to_q_back = Arrow(
            start=[-3.74, 0.45, 0],
            end=[-2.54, 0.45, 0],
            buff=0.08,
            stroke_width=9.0,
            color=WHITE,
        ).set_z_index(8)
        arrow_macro_to_q = Arrow(
            start=[-3.74, 0.45, 0],
            end=[-2.54, 0.45, 0],
            buff=0.08,
            stroke_width=4.0,
            color=ORANGE,
        ).set_z_index(9)
        arrow_q_to_ann = Arrow(
            start=[0.12, 0.45, 0],
            end=[1.70, 0.45, 0],
            buff=0.06,
            stroke_width=4.0,
            color=ORANGE,
        ).set_z_index(9)
        q_arrow_label = MathTex(r"\mathbf q", color=ORANGE).scale(0.72).next_to(arrow_q_to_ann, UP, buff=0.14)

        ann_label = MathTex(r"\mathcal N_\theta(\mathbf q)", color=BLACK).scale(0.72).move_to([3.65, 2.28, 0])
        qs_label = MathTex(r"\overline{\mathbf q}\in\mathbb R^{36}", color=BLUE).scale(0.60).move_to([5.42, 1.70, 0])

        decoder_formula = MathTex(
            r"\widetilde{\mathbf u}(\mathbf q,\boldsymbol\mu)",
            r"=",
            r"\mathbf u_{\rm aff}(\boldsymbol\mu)",
            r"+",
            r"\mathbf V\mathbf A\mathbf q",
            r"+",
            r"\overline{\mathbf V}\mathcal N_\theta(\mathbf q)",
            color=BLACK,
        ).scale(0.62).to_edge(DOWN, buff=0.72)

        self.play(FadeIn(title, shift=UP * 0.18), run_time=0.55)
        self.play(FadeIn(total_basis, scale=0.92), run_time=0.75)
        self.wait(5.0)
        self.play(
            FadeIn(parameter_dimension, shift=UP * 0.08),
            FadeIn(primary_basis, shift=LEFT * 0.12),
            FadeIn(secondary_basis, shift=RIGHT * 0.12),
            FadeIn(partition_formula, shift=UP * 0.08),
            FadeOut(total_basis),
            run_time=1.10,
        )
        self.wait(8.0)
        self.play(FadeOut(pod_intro_group, shift=UP * 0.08), run_time=0.55)
        self.play(
            FadeIn(raw_label, shift=UP * 0.10),
            Create(raw_cube),
            FadeIn(raw_axes),
            FadeIn(raw_pod_domain),
            FadeIn(raw_pod_points),
            run_time=1.15,
        )
        self.play(FadeIn(raw_note, shift=UP * 0.05), run_time=0.45)
        self.wait(5.0)
        self.play(
            FadeIn(q_preview_label, shift=UP * 0.08),
            FadeIn(q_preview_note, shift=UP * 0.08),
            Create(q_cube),
            FadeIn(q_axes),
            LaggedStart(*[Create(path) for path in q_background], lag_ratio=0.035),
            GrowArrow(raw_to_target),
            FadeIn(raw_to_target_label, shift=UP * 0.04),
            run_time=1.15,
        )
        self.wait(5.0)
        self.play(
            FadeOut(VGroup(raw_label, raw_note, raw_cube, raw_axes, raw_pod_domain, raw_pod_points, raw_to_target, raw_to_target_label), shift=LEFT * 0.10),
            q_preview_group.animate.shift(-Q_PREVIEW_SHIFT),
            Transform(q_preview_label, q_label),
            FadeOut(q_preview_note),
            run_time=0.70,
        )
        self.play(
            FadeIn(macro_label, shift=UP * 0.10),
            Create(macro_cube),
            FadeIn(macro_axes),
            LaggedStart(*[Create(path) for path in macro_background], lag_ratio=0.035),
            run_time=1.20,
        )
        self.play(
            LaggedStart(*[Create(path) for path in macro_highlight], lag_ratio=0.07),
            run_time=0.75,
        )
        self.play(FadeIn(opt_group, shift=UP * 0.10), GrowArrow(arrow_macro_to_q_back), GrowArrow(arrow_macro_to_q), run_time=0.75)
        self.play(
            LaggedStart(
                *[TransformFromCopy(macro_path, q_path) for macro_path, q_path in zip(macro_highlight, q_highlight)],
                lag_ratio=0.14,
            ),
            run_time=2.20,
        )
        self.wait(0.45)
        q_repeat = VGroup(*[path.copy().set_stroke(ORANGE, width=3.2, opacity=1.0) for path in q_highlight])
        self.play(
            LaggedStart(
                *[TransformFromCopy(macro_path, q_path) for macro_path, q_path in zip(macro_highlight, q_repeat)],
                lag_ratio=0.14,
            ),
            run_time=2.20,
        )
        self.play(FadeOut(q_repeat), run_time=0.25)
        self.wait(8.0)
        self.play(GrowArrow(arrow_q_to_ann), FadeIn(q_arrow_label, shift=UP * 0.08), run_time=0.55)
        self.play(
            FadeIn(ann_label, shift=UP * 0.10),
            LaggedStart(*[Create(layer) for layer in network_edge_layers], lag_ratio=0.22),
            FadeIn(network_nodes, scale=0.88),
            FadeIn(input_labels),
            run_time=1.05,
        )

        input_layer = network_nodes[0]
        hidden_1 = network_nodes[1]
        hidden_2 = network_nodes[2]
        output_nodes = network_nodes[3]
        self.play(activate_layer(input_layer, activation_values[0], ORANGE), run_time=0.36)
        first_glow = network_edge_layers[0].copy().set_stroke(ORANGE, width=1.9, opacity=0.52)
        self.play(
            FadeIn(first_glow),
            activate_layer(hidden_1, activation_values[1], BLUE),
            run_time=0.52,
        )
        self.play(FadeOut(first_glow), run_time=0.12)
        second_glow = network_edge_layers[1].copy().set_stroke(ORANGE, width=1.9, opacity=0.52)
        self.play(
            FadeIn(second_glow),
            activate_layer(hidden_2, activation_values[2], BLUE),
            run_time=0.52,
        )
        self.play(FadeOut(second_glow), run_time=0.12)
        third_glow = network_edge_layers[2].copy().set_stroke(ORANGE, width=1.9, opacity=0.52)
        self.play(
            FadeIn(third_glow),
            activate_layer(output_nodes, activation_values[3], BLUE),
            FadeIn(qs_label, shift=RIGHT * 0.08),
            run_time=0.60,
        )
        self.play(FadeOut(third_glow), run_time=0.20)
        self.play(FadeIn(decoder_formula, shift=UP * 0.18), run_time=0.70)
        self.wait(0.35)
        training_label = Text("training", font_size=16, color=BLACKISH).move_to([3.60, -1.52, 0])
        self.play(FadeIn(training_label, shift=UP * 0.05), run_time=0.20)
        all_network_edges = VGroup(*network_edge_layers)
        for _ in range(2):
            forward_glow = all_network_edges.copy().set_stroke(ORANGE, width=1.8, opacity=0.42)
            self.play(FadeIn(forward_glow), pulse_layer(input_layer, ORANGE), run_time=0.28)
            self.play(pulse_layer(hidden_1, ORANGE), run_time=0.28)
            self.play(pulse_layer(hidden_2, ORANGE), run_time=0.28)
            self.play(pulse_layer(output_nodes, ORANGE, scale_factor=1.36), run_time=0.28)
            self.play(FadeOut(forward_glow), run_time=0.12)
            backward_glow = all_network_edges.copy().set_stroke(BLUE, width=1.8, opacity=0.38)
            self.play(FadeIn(backward_glow), pulse_layer(output_nodes, BLUE, scale_factor=1.36), run_time=0.28)
            self.play(pulse_layer(hidden_2, BLUE), run_time=0.28)
            self.play(pulse_layer(hidden_1, BLUE), run_time=0.28)
            self.play(pulse_layer(input_layer, BLUE), run_time=0.28)
            self.play(FadeOut(backward_glow), run_time=0.12)
        self.play(FadeOut(training_label), run_time=0.18)
        self.wait(4.0)
