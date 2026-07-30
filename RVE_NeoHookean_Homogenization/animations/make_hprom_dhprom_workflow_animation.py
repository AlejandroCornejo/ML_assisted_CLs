#!/usr/bin/env python3
"""Create a paced visual workflow for HPROM-ANN and D-HPROM-ANN.

The animation is deliberately formula-light.  It shows how the affine map
provides q^0, then makes the one online difference explicit: HPROM-ANN uses a
Galerkin correction on the residual support while D-HPROM-ANN bypasses it.
Both routes share the PROM-ANN state map and hyper-reduced stress evaluation.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MESH_FILE = PROJECT_DIR / "rve_geometry.mdpa"
ECM_FILE = PROJECT_DIR / (
    "stage_12_hprom_ann_mawecm_res_eps_sig_phase1to40_phase2to10_sum990_ann/"
    "ecm_weights_all.npz"
)
OUTPUT_MP4 = SCRIPT_DIR / "hprom_dhprom_ann_workflow.mp4"
OUTPUT_GIF = SCRIPT_DIR / "hprom_dhprom_ann_workflow.gif"

WIDTH, HEIGHT = 1280, 720
FPS = 12
DURATION_SECONDS = 66.0

INK = "#17212B"
MUTED = "#657281"
PANEL_EDGE = "#2F73C6"
BLUE = "#1676D2"
BLUE_FILL = "#EAF3FB"
ORANGE = "#D66522"
ORANGE_FILL = "#FCEDE4"
MESH_GRAY = "#AEB8C3"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{suffix}", size)


FONT_SMALL = font(13)
FONT_TINY = font(11)
FONT_BODY = font(16)
FONT_BODY_BOLD = font(16, bold=True)
FONT_TITLE = font(20, bold=True)
FONT_FORMULA = font(15)
FONT_FORMULA_SMALL = font(13)


def hex_rgba(color: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4)) + (int(255 * alpha),)


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def fade_in(time_s: float, start: float, duration: float = 0.65) -> float:
    return smoothstep((time_s - start) / duration)


def mix(a: np.ndarray | tuple[float, float], b: np.ndarray | tuple[float, float], value: float) -> np.ndarray:
    return (1.0 - value) * np.asarray(a, dtype=float) + value * np.asarray(b, dtype=float)


def alpha_layer(base: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 0.999:
        return base
    out = base.copy()
    out.putalpha(out.getchannel("A").point(lambda pixel: int(pixel * alpha)))
    return out


def paste(canvas: Image.Image, layer: Image.Image, xy: tuple[int, int] = (0, 0), alpha: float = 1.0) -> None:
    canvas.alpha_composite(alpha_layer(layer, alpha), xy)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font_value: ImageFont.FreeTypeFont, fill: str, anchor: str = "la") -> None:
    draw.text(xy, text, font=font_value, fill=hex_rgba(fill), anchor=anchor)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    edge_color: str,
    fill_color: str,
    alpha: float,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(
        box,
        radius=18,
        fill=hex_rgba(fill_color, 0.78 * alpha),
        outline=hex_rgba(edge_color, alpha),
        width=3,
    )
    draw_text(draw, (x0 + 16, y0 + 15), title, FONT_BODY_BOLD, edge_color)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    alpha: float = 1.0,
    width: int = 4,
    head: float = 12.0,
) -> None:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    direction = end_array - start_array
    length = float(np.linalg.norm(direction))
    if length < 1.0e-9:
        return
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    head_base = end_array - head * unit
    draw.line([tuple(start_array), tuple(head_base)], fill=hex_rgba(color, alpha), width=width)
    triangle = [tuple(end_array), tuple(head_base + 0.55 * head * normal), tuple(head_base - 0.55 * head * normal)]
    draw.polygon(triangle, fill=hex_rgba(color, alpha))


def draw_poly_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str,
    alpha: float = 1.0,
    width: int = 4,
) -> None:
    for start, end in zip(points[:-2], points[1:-1]):
        draw.line([start, end], fill=hex_rgba(color, alpha), width=width)
    draw_arrow(draw, points[-2], points[-1], color, alpha=alpha, width=width)


def draw_token(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    label: str,
    color: str,
    alpha: float = 1.0,
) -> None:
    x, y = center
    radius = 14
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=hex_rgba(WHITE, alpha),
        outline=hex_rgba(color, alpha),
        width=3,
    )
    draw_text(draw, (x, y + 0.5), label, FONT_TINY, color, anchor="mm")


def load_linear_mesh(path: Path) -> tuple[dict[int, np.ndarray], list[tuple[int, int, int]]]:
    nodes: dict[int, np.ndarray] = {}
    triangles: list[tuple[int, int, int]] = []
    section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
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
        if section == "nodes":
            nodes[int(fields[0])] = np.asarray([float(fields[1]), float(fields[2])])
        elif section == "triangles":
            triangles.append((int(fields[1]), int(fields[2]), int(fields[3])))
    return nodes, triangles


def render_support_mesh(
    nodes: dict[int, np.ndarray],
    triangles: list[tuple[int, int, int]],
    selected: np.ndarray,
    color: str,
    emphasize: float,
    size: int = 150,
) -> Image.Image:
    output = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(output, "RGBA")
    selected_ids = {int(index) for index in selected}
    all_points = np.vstack(list(nodes.values()))
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    span = maximum - minimum
    scale = 0.90 * size / float(max(span))
    offset = 0.5 * (minimum + maximum)

    def transform(point: np.ndarray) -> tuple[float, float]:
        centered = point - offset
        return (0.5 * size + scale * centered[0], 0.5 * size - scale * centered[1])

    for index, triangle in enumerate(triangles):
        polygon = [transform(nodes[node_id]) for node_id in triangle]
        draw.line(polygon + [polygon[0]], fill=hex_rgba(MESH_GRAY, 0.72), width=1)
        if index in selected_ids:
            fill_alpha = 0.86 if emphasize > 0.6 else 0.66
            draw.polygon(polygon, fill=hex_rgba(color, fill_alpha))
            draw.line(polygon + [polygon[0]], fill=hex_rgba(WHITE, 0.95), width=2)
            if emphasize > 0.05:
                centroid = np.mean(np.asarray(polygon), axis=0)
                radius = 3.0 + 4.0 * emphasize
                draw.ellipse(
                    (
                        centroid[0] - radius,
                        centroid[1] - radius,
                        centroid[0] + radius,
                        centroid[1] + radius,
                    ),
                    outline=hex_rgba(color, 0.35 * emphasize),
                    width=2,
                )
    return output


def iso_point(center: tuple[float, float], point: tuple[float, float, float], scale: float) -> tuple[float, float]:
    x, y, z = point
    return (center[0] + scale * (0.90 * x + 0.48 * y), center[1] + scale * (0.28 * y - 0.92 * z))


def draw_domain(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    labels: tuple[str, str, str],
    color: str,
    alpha: float,
    structured: bool,
) -> None:
    corners = [
        (sx, sy, sz)
        for sx in (-0.58, 0.58)
        for sy in (-0.58, 0.58)
        for sz in (-0.36, 0.36)
    ]
    points = [iso_point(center, point, 58.0) for point in corners]
    edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]
    for left, right in edges:
        draw.line([points[left], points[right]], fill=hex_rgba(color, 0.52 * alpha), width=2)

    if structured:
        for z_value in np.linspace(-0.23, 0.23, 6):
            line_points = [iso_point(center, (x_value, y_value, z_value), 58.0) for x_value, y_value in [(-0.48, -0.48), (0.48, -0.48), (0.48, 0.48), (-0.48, 0.48), (-0.48, -0.48)]]
            draw.line(line_points, fill=hex_rgba(color, 0.68 * alpha), width=2)
    else:
        for z_value in np.linspace(-0.22, 0.22, 5):
            line_points = [
                iso_point(center, (x_value, 0.33 * math.sin(3.2 * x_value + 3.0 * z_value), z_value), 58.0)
                for x_value in np.linspace(-0.48, 0.48, 18)
            ]
            draw.line(line_points, fill=hex_rgba(ORANGE, 0.76 * alpha), width=2)

    base = (center[0] - 43, center[1] + 55)
    axis_ends = [(base[0] + 37, base[1]), (base[0] + 20, base[1] - 18), (base[0], base[1] - 34)]
    for end, label in zip(axis_ends, labels):
        draw_arrow(draw, base, end, INK, alpha=alpha, width=2, head=7)
        draw_text(draw, (end[0] + 3, end[1] - 4), label, FONT_TINY, INK)


def point_on_polyline(points: list[tuple[float, float]], fraction: float) -> tuple[float, float]:
    segments = [float(np.linalg.norm(np.asarray(end) - np.asarray(start))) for start, end in zip(points[:-1], points[1:])]
    total = max(sum(segments), 1.0e-12)
    distance = max(0.0, min(1.0, fraction)) * total
    for start, end, segment in zip(points[:-1], points[1:], segments):
        if distance <= segment:
            return tuple(mix(start, end, distance / max(segment, 1.0e-12)))
        distance -= segment
    return points[-1]


def draw_macro_input(draw: ImageDraw.ImageDraw, alpha: float) -> None:
    box = (30, 250, 238, 415)
    draw_panel(draw, box, "macro input", PANEL_EDGE, BLUE_FILL, alpha)
    draw_text(draw, (134, 302), "Fbar", FONT_TITLE, INK, anchor="ma")
    draw_arrow(draw, (134, 322), (134, 342), PANEL_EDGE, alpha=alpha, width=3, head=8)
    draw_text(draw, (134, 358), "Ebar = 1/2(Fbar^T Fbar - I)", FONT_FORMULA_SMALL, INK, anchor="ma")
    draw_arrow(draw, (134, 372), (134, 389), PANEL_EDGE, alpha=alpha, width=3, head=8)
    draw_text(draw, (134, 401), "mu = (E_xx, E_yy, G_xy)", FONT_FORMULA_SMALL, INK, anchor="ma")


def draw_affine_map(draw: ImageDraw.ImageDraw, alpha: float, time_s: float) -> None:
    draw_text(draw, (398, 145), "affine coordinate initialization", FONT_BODY_BOLD, INK, anchor="ma")
    draw_domain(draw, (330, 276), ("E_xx", "E_yy", "G_xy"), MUTED, alpha, structured=False)
    draw_domain(draw, (478, 276), ("q1", "q2", "q3"), BLUE, alpha, structured=True)
    draw_text(draw, (330, 188), "macro strain", FONT_SMALL, MUTED, anchor="ma")
    draw_text(draw, (478, 188), "structured q domain", FONT_SMALL, BLUE, anchor="ma")
    draw_arrow(draw, (385, 255), (423, 255), ORANGE, alpha=alpha, width=4)
    draw_text(draw, (404, 236), "B_aff", FONT_SMALL, ORANGE, anchor="ma")
    draw_text(draw, (405, 390), "q^0(mu) = [mu, 1] B_aff", FONT_FORMULA, INK, anchor="ma")

    phase = (time_s - 6.3) / 6.1
    if 0.0 <= phase <= 1.0:
        source = iso_point((330, 276), (0.20, -0.15, 0.08), 58.0)
        target = iso_point((478, 276), (0.20, -0.15, 0.08), 58.0)
        draw_token(draw, tuple(mix(source, target, smoothstep(phase))), "mu", ORANGE, alpha=alpha)
    elif time_s > 12.4:
        target = iso_point((478, 276), (0.20, -0.15, 0.08), 58.0)
        draw_token(draw, target, "q0", ORANGE, alpha=alpha)


def draw_decoder(draw: ImageDraw.ImageDraw, alpha: float, active_color: str | None = None) -> None:
    box = (930, 172, 1230, 365)
    draw_panel(draw, box, "PROM-ANN state map", PANEL_EDGE, BLUE_FILL, alpha)
    draw_text(draw, (1080, 223), "q, mu  ->  u~(q, mu)", FONT_BODY_BOLD, INK, anchor="ma")
    draw_text(draw, (1080, 258), "u~ = u_aff(mu) + V A q", FONT_FORMULA, INK, anchor="ma")
    draw_text(draw, (1080, 284), "+ Vbar N_theta(q)", FONT_FORMULA, INK, anchor="ma")
    draw_text(draw, (1080, 327), "nonlinear RVE state", FONT_SMALL, MUTED, anchor="ma")
    if active_color is not None:
        draw.rounded_rectangle((944, 186, 1216, 350), radius=13, outline=hex_rgba(active_color, 0.92 * alpha), width=4)


def draw_hprom_panel(draw: ImageDraw.ImageDraw, residual_mesh: Image.Image, alpha: float, time_s: float) -> None:
    box = (550, 56, 900, 310)
    draw_panel(draw, box, "HPROM-ANN: Galerkin correction", ORANGE, ORANGE_FILL, alpha)
    pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * max(0.0, time_s - 26.0) / 3.4)
    mesh_alpha = alpha * (0.72 + 0.28 * pulse)
    paste(draw._image, residual_mesh, (568, 120), alpha=mesh_alpha)
    draw_text(draw, (643, 274), "10 residual elements", FONT_SMALL, ORANGE, anchor="ma")
    draw_text(draw, (748, 130), "W(q)^T r", FONT_TITLE, INK, anchor="ma")
    draw_text(draw, (748, 158), "approx. sum over Z_res", FONT_FORMULA_SMALL, INK, anchor="ma")
    draw_text(draw, (748, 188), "Galerkin update", FONT_BODY_BOLD, ORANGE, anchor="ma")
    draw_text(draw, (748, 218), "q^k + delta q^k", FONT_FORMULA, INK, anchor="ma")
    draw_text(draw, (748, 258), "q0  ->  q1  ->  q*", FONT_FORMULA, ORANGE, anchor="ma")
    draw_poly_arrow(draw, [(835, 225), (866, 225), (866, 170), (806, 170)], ORANGE, alpha=alpha, width=3)


def draw_direct_panel(draw: ImageDraw.ImageDraw, alpha: float) -> None:
    box = (550, 420, 900, 558)
    draw_panel(draw, box, "D-HPROM-ANN: direct state", BLUE, BLUE_FILL, alpha)
    draw_text(draw, (725, 476), "q = q0(mu)", FONT_TITLE, INK, anchor="ma")
    draw_text(draw, (725, 509), "skip residual correction", FONT_BODY_BOLD, BLUE, anchor="ma")
    draw_text(draw, (725, 534), "one decoder evaluation", FONT_SMALL, MUTED, anchor="ma")


def draw_stress_panel(draw: ImageDraw.ImageDraw, stress_mesh: Image.Image, alpha: float, time_s: float) -> None:
    box = (950, 402, 1235, 666)
    draw_panel(draw, box, "hyper-reduced homogenization", BLUE, BLUE_FILL, alpha)
    pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * max(0.0, time_s - 54.0) / 2.8)
    paste(draw._image, stress_mesh, (968, 461), alpha=alpha * (0.72 + 0.28 * pulse))
    draw_text(draw, (1043, 620), "10 stress elements", FONT_SMALL, BLUE, anchor="ma")
    draw_text(draw, (1144, 468), "Sbar", FONT_TITLE, INK, anchor="ma")
    draw_text(draw, (1144, 496), "approx. weighted", FONT_FORMULA_SMALL, INK, anchor="ma")
    draw_text(draw, (1144, 515), "stress average", FONT_FORMULA_SMALL, INK, anchor="ma")
    draw_text(draw, (1144, 560), "Sbar = 1/A0", FONT_FORMULA, BLUE, anchor="ma")
    draw_text(draw, (1144, 583), "sum over Z_sigma", FONT_FORMULA_SMALL, INK, anchor="ma")
    draw_text(draw, (1144, 633), "homogenized stress", FONT_SMALL, MUTED, anchor="ma")


def frame_at(time_s: float, residual_mesh: Image.Image, stress_mesh: Image.Image) -> np.ndarray:
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), hex_rgba(WHITE))
    draw = ImageDraw.Draw(canvas, "RGBA")

    macro_alpha = fade_in(time_s, 0.5)
    affine_alpha = fade_in(time_s, 5.2)
    decoder_alpha = fade_in(time_s, 17.8)
    hprom_alpha = fade_in(time_s, 24.3)
    direct_alpha = fade_in(time_s, 41.2)
    stress_alpha = fade_in(time_s, 52.5)

    if macro_alpha > 0.0:
        draw_macro_input(draw, macro_alpha)
    if affine_alpha > 0.0:
        draw_affine_map(draw, affine_alpha, time_s)
        draw_arrow(draw, (238, 331), (270, 331), PANEL_EDGE, alpha=min(macro_alpha, affine_alpha), width=4)

    if time_s >= 13.0:
        branch_alpha = min(affine_alpha, max(hprom_alpha, direct_alpha, decoder_alpha))
        draw_poly_arrow(draw, [(536, 330), (536, 177), (550, 177)], ORANGE, alpha=branch_alpha, width=4)
        draw_poly_arrow(draw, [(536, 330), (536, 490), (550, 490)], BLUE, alpha=branch_alpha, width=4)
        draw_text(draw, (538, 347), "q0", FONT_SMALL, ORANGE, anchor="ma")

    if hprom_alpha > 0.0:
        draw_hprom_panel(draw, residual_mesh, hprom_alpha, time_s)
        draw_poly_arrow(draw, [(900, 178), (914, 178), (914, 266), (930, 266)], ORANGE, alpha=hprom_alpha, width=4)
        draw_text(draw, (913, 206), "q*", FONT_SMALL, ORANGE, anchor="ma")

    if direct_alpha > 0.0:
        draw_direct_panel(draw, direct_alpha)
        draw_poly_arrow(draw, [(900, 490), (916, 490), (916, 312), (930, 312)], BLUE, alpha=direct_alpha, width=4)
        draw_text(draw, (910, 448), "q = q0", FONT_SMALL, BLUE, anchor="ma")

    if decoder_alpha > 0.0:
        if 27.0 <= time_s < 40.5:
            active = ORANGE
        elif 43.0 <= time_s < 52.0:
            active = BLUE
        else:
            active = None
        draw_decoder(draw, decoder_alpha, active)

    if time_s >= 27.0 and time_s < 40.5:
        correction_phase = ((time_s - 27.0) % 4.0) / 4.0
        route = [(536, 177), (576, 177), (638, 190), (770, 205), (853, 205), (853, 266), (808, 266)]
        draw_token(draw, point_on_polyline(route, correction_phase), "qk", ORANGE, alpha=1.0)
    elif time_s >= 40.5:
        draw_token(draw, (808, 266), "q*", ORANGE, alpha=0.92)

    if 43.0 <= time_s < 52.0:
        direct_phase = min(1.0, (time_s - 43.0) / 5.6)
        route = [(536, 490), (610, 490), (735, 490), (875, 490), (916, 352), (946, 312)]
        draw_token(draw, point_on_polyline(route, direct_phase), "q0", BLUE, alpha=1.0)
    elif time_s >= 52.0:
        draw_token(draw, (946, 312), "q0", BLUE, alpha=0.86)

    if stress_alpha > 0.0:
        draw_arrow(draw, (1080, 365), (1080, 402), BLUE, alpha=stress_alpha, width=4)
        draw_stress_panel(draw, stress_mesh, stress_alpha, time_s)
        if time_s >= 56.0:
            progress = min(1.0, (time_s - 56.0) / 4.0)
            draw_token(draw, tuple(mix((1080, 372), (1080, 542), progress)), "u", BLUE, alpha=1.0)

    if time_s >= 59.5:
        final_alpha = fade_in(time_s, 59.5, 0.7)
        draw.rounded_rectangle((28, 645, 908, 688), radius=12, fill=hex_rgba("#F5F7FA", 0.92 * final_alpha), outline=hex_rgba("#D3DAE2", final_alpha), width=1)
        draw_text(
            draw,
            (468, 667),
            "same decoder + stress support; only HPROM-ANN performs online residual correction",
            FONT_SMALL,
            INK,
            anchor="mm",
        )

    return np.asarray(canvas.convert("RGB"))


def create_gif(mp4_file: Path, gif_file: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create the GIF.")
    temporary_file = gif_file.with_name(f"{gif_file.stem}.tmp.gif")
    filter_graph = "fps=10,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=sierra2_4a"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(mp4_file),
            "-vf",
            filter_graph,
            "-loop",
            "0",
            str(temporary_file),
        ],
        check=True,
    )
    temporary_file.replace(gif_file)


def main() -> None:
    nodes, triangles = load_linear_mesh(MESH_FILE)
    ecm = np.load(ECM_FILE, allow_pickle=True)
    residual_selected = np.asarray(ecm["Z_res"], dtype=int)
    stress_selected = np.asarray(ecm["Z_sig"], dtype=int)
    residual_mesh = render_support_mesh(nodes, triangles, residual_selected, ORANGE, emphasize=1.0)
    stress_mesh = render_support_mesh(nodes, triangles, stress_selected, BLUE, emphasize=1.0)

    frame_count = int(round(DURATION_SECONDS * FPS))
    with imageio.get_writer(
        OUTPUT_MP4,
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        ffmpeg_log_level="error",
    ) as writer:
        for frame_index in range(frame_count):
            time_s = frame_index / FPS
            writer.append_data(frame_at(time_s, residual_mesh, stress_mesh))
            if frame_index % FPS == 0 or frame_index == frame_count - 1:
                print(f"Rendered {time_s:05.1f} / {DURATION_SECONDS:.1f} s", flush=True)

    create_gif(OUTPUT_MP4, OUTPUT_GIF)
    print(f"Saved MP4: {OUTPUT_MP4}")
    print(f"Saved GIF: {OUTPUT_GIF}")


if __name__ == "__main__":
    main()
