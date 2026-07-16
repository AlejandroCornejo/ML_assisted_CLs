#!/usr/bin/env python3
"""Create a portrait LinkedIn MP4 for the WCCM 2026 HPROM-ANN animation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_GIF = SCRIPT_DIR / "hprom_ann_three_panel_response.gif"
WCCM_SCREENSHOT = Path("/home/kratos/Pictures/Screenshots/WCCM2026_Screenshot.png")
SESSION_CARD = SCRIPT_DIR / "WCCM2026_MS106B_session_card.png"
OUTPUT_MP4 = SCRIPT_DIR / "WCCM2026_MS106B_hprom_ann_linkedin.mp4"

CARD_SIZE = (1430, 1056)
SCALE = 2


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{suffix}", size * SCALE)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font_value: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_gap: int = 3,
) -> int:
    """Draw a compact word-wrapped text block and return its lower y-coordinate."""
    x, y = xy
    words = text.split()
    line = ""
    lines: list[str] = []
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if draw.textlength(candidate, font=font_value) <= max_width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    bbox = draw.textbbox((0, 0), "Ag", font=font_value)
    line_height = (bbox[3] - bbox[1]) + line_gap * SCALE
    for index, line_text in enumerate(lines):
        draw.text((x, y + index * line_height), line_text, font=font_value, fill=fill)
    return y + len(lines) * line_height


def render_session_card(output_file: Path) -> None:
    """Recreate the WCCM 2026 session listing shown in the supplied image."""
    image = Image.new("RGB", CARD_SIZE, "#ffffff")
    draw = ImageDraw.Draw(image)
    margin = 10 * SCALE
    card_box = (margin, margin, CARD_SIZE[0] - margin, CARD_SIZE[1] - margin)
    draw.rounded_rectangle(card_box, radius=5 * SCALE, fill="#cdd9ec", outline="#083d8c", width=2 * SCALE)

    blue = "#315cff"
    ink = "#1e293b"
    left_x = 22 * SCALE
    right_x = 256 * SCALE
    y_top = 18 * SCALE

    date_font = font(17)
    title_font = font(18, bold=True)
    meta_font = font(13)
    talk_font = font(16)
    author_font = font(13)

    for index, value in enumerate(("Thu, 23/07/2026", "15:00 - 16:30", "H- 4 (Alois)", "MS106B")):
        draw.text((left_x, y_top + index * 23 * SCALE), value, font=date_font, fill=blue if index < 3 else ink)

    title_y = draw_wrapped(
        draw,
        "MS106B Advances in Model Order Reduction: Bridging Physics and Machine Learning II",
        (right_x, y_top),
        title_font,
        blue,
        430 * SCALE,
        line_gap=2,
    )
    organizer_y = draw_wrapped(
        draw,
        "Main Organizer: Dr. Youngsoo Choi (Lawrence Livermore National Laboratory, United States)",
        (right_x, title_y + 7 * SCALE),
        meta_font,
        ink,
        430 * SCALE,
        line_gap=1,
    )
    draw_wrapped(
        draw,
        "Chaired by: Dr. Siu Wun Cheung (Lawrence Livermore National Laboratory, United States), Dr. Youngsoo Choi (Lawrence Livermore National Laboratory, United States)",
        (right_x, organizer_y + 6 * SCALE),
        meta_font,
        ink,
        430 * SCALE,
        line_gap=1,
    )

    separator_y = 162 * SCALE
    draw.line((22 * SCALE, separator_y, CARD_SIZE[0] - 22 * SCALE, separator_y), fill="#94a3b8", width=1 * SCALE)

    talks = [
        ("Optimal Learning in Shallow AutoEncoders", "R. Bousquet *, C. Nore, D. Lefeuve"),
        ("A Multi-layer POD-FFT Bridging Framework for Stable Neural Compression of Mesh-based Finite Element Data with Complex Geometry Boundaries", "P. Yu, C. Wang *, Z. You, L. Feng"),
        ("Sparse Model Selection and Manifold Dimensionality Reduction with Neural Networks", "T. Koike *, P. Mohan, M. Henry de Frahan, E. Qian, J. Bessac"),
        ("Local Nonlinear Reduced-Order Models with Regression-Based Latent-Space Closure", "S. Ares de Parga Regalado *, R. Rossi, J. Hernandez Ortega, A. Cornejo Velazquez, R. Tezaur, C. Farhat"),
        ("Interpretable Latent Dynamics via Graph Convolutional Networks", "F. Pichi *, L. Tomada, G. Rozza"),
        ("Nonlinear Performance Bounds for Reduced Order Models", "W. Cho, K. Lee, N. Park, D. Rim, G. Welper *"),
    ]

    bullet_x = 38 * SCALE
    text_x = 53 * SCALE
    y = 181 * SCALE
    for title, authors in talks:
        draw.ellipse((bullet_x - 3 * SCALE, y + 7 * SCALE, bullet_x + 3 * SCALE, y + 13 * SCALE), fill="#475569")
        title_end = draw_wrapped(draw, title, (text_x, y), talk_font, blue, 630 * SCALE, line_gap=1)
        author_end = draw_wrapped(draw, authors, (text_x, title_end + 1 * SCALE), author_font, ink, 630 * SCALE, line_gap=1)
        y = author_end + 6 * SCALE

    image.save(output_file, quality=95)


def copy_supplied_session_card(source_file: Path, output_file: Path) -> None:
    """Use the supplied WCCM screenshot without modifying its visual content."""
    if not source_file.exists():
        raise FileNotFoundError(source_file)
    with Image.open(source_file) as source:
        source.convert("RGB").save(output_file)


def create_video(card_file: Path, source_gif: Path, output_file: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to produce the LinkedIn MP4.")
    if not source_gif.exists():
        raise FileNotFoundError(source_gif)

    # GIFs carry an infinite-loop flag.  Count their encoded frame durations so
    # the portrait MP4 ends after one presentation loop.
    with Image.open(source_gif) as gif:
        duration_seconds = 0.0
        for frame_index in range(gif.n_frames):
            gif.seek(frame_index)
            duration_seconds += float(gif.info.get("duration", 0)) / 1000.0

    filters = (
        "color=c=white:s=1080x1350[base];"
        "[0:v]scale=1000:736:flags=lanczos[card];"
        "[1:v]fps=15,scale=1000:562:flags=lanczos[animation];"
        "[base][card]overlay=40:20[with_card];"
        "[with_card][animation]overlay=40:780,format=yuv420p[out]"
    )
    temporary_file = output_file.with_name(f"{output_file.stem}.tmp.mp4")
    command = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-i",
        str(card_file),
        "-i",
        str(source_gif),
        "-filter_complex",
        filters,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        "-r",
        "15",
        "-t",
        f"{duration_seconds:.3f}",
        str(temporary_file),
    ]
    subprocess.run(command, check=True)
    temporary_file.replace(output_file)


def main() -> None:
    copy_supplied_session_card(WCCM_SCREENSHOT, SESSION_CARD)
    create_video(SESSION_CARD, SOURCE_GIF, OUTPUT_MP4)
    print(f"Saved session card: {SESSION_CARD}")
    print(f"Saved LinkedIn MP4: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
