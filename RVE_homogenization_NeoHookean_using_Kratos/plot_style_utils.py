#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared Matplotlib styling for LaTeX-friendly figures."""

import os


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def apply_latex_plot_style(use_tex: bool = True) -> None:
    """
    Apply a consistent paper-style Matplotlib setup.

    By default LaTeX rendering is enabled. Set environment variable
    `RVE_PLOT_USE_TEX=0` to force-disable it globally when needed.
    """
    use_tex_env = _env_flag("RVE_PLOT_USE_TEX", True)
    use_tex_final = bool(use_tex) and bool(use_tex_env)

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "text.usetex": use_tex_final,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 2.0,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )

