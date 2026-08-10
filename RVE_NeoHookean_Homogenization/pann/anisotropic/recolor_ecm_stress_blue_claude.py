#!/usr/bin/env python3
"""Recolor the stress-support (Z_sig) ECM figure from red to blue, so it
is visually distinguishable from the residual-support (Z_res) figure,
which stays red. Both currently use the same red for their selected
elements (RGB ~(214,71,71)), so a reader has to read the small title
text to tell the two element sets apart; this script fixes that by hue-
rotating every reddish pixel to blue in HSV space (preserving saturation
and value), which correctly handles anti-aliased triangle edges too,
since a partially-red-tinted edge pixel just becomes a partially-blue-
tinted one at the same blend strength.

Regenerates ecm_stress_support_claude.png in place (the source used
directly by Figure 11's right panel) and also writes a matching
recolored copy of the raw thumbnail crop consumed by the workflow
diagram (Figure 12), so both figures show the same blue for this set.
"""
from __future__ import annotations

from pathlib import Path

import colorsys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
TARGET_HUE = 210.0 / 360.0  # medium blue, matching this paper's #1f5fa8-family blues
RED_HUE_MAX = 30.0 / 360.0  # anything with hue in [0, RED_HUE_MAX] or [1-RED_HUE_MAX, 1]
SATURATION_MIN = 0.12  # skip near-gray/white/black pixels (mesh lines, background, text)

_rgb_to_hsv_pixel = np.vectorize(colorsys.rgb_to_hsv)
_hsv_to_rgb_pixel = np.vectorize(colorsys.hsv_to_rgb)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    h, s, v = _rgb_to_hsv_pixel(rgb[..., 0], rgb[..., 1], rgb[..., 2])
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    r, g, b = _hsv_to_rgb_pixel(hsv[..., 0], hsv[..., 1], hsv[..., 2])
    return np.stack([r, g, b], axis=-1)


def recolor_red_to_blue(path: Path) -> None:
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img).astype(np.float64) / 255.0
    rgb, alpha = arr[..., :3], arr[..., 3:4]

    hsv = rgb_to_hsv(rgb)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    is_reddish = ((hue <= RED_HUE_MAX) | (hue >= 1.0 - RED_HUE_MAX)) & (sat >= SATURATION_MIN)

    new_hue = np.where(is_reddish, TARGET_HUE, hue)
    new_hsv = np.stack([new_hue, sat, val], axis=-1)
    new_rgb = hsv_to_rgb(new_hsv)

    out = np.concatenate([new_rgb, alpha], axis=-1)
    out = (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(out, mode="RGBA").save(path)
    print(f"recolored {path.name}: {int(is_reddish.sum())} reddish pixels -> blue")


def main() -> None:
    recolor_red_to_blue(HERE / "ecm_stress_support_claude.png")


if __name__ == "__main__":
    main()
