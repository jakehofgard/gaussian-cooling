"""Shared publication-plot helpers for lattice phi4 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from .lattice_phi4 import (
    COOLING_COMPARISON,
    EMPIRICAL_COMPARISON,
    RAW_ULMC,
    TRANSLATION_AVERAGED_ULMC,
)


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#6B6B6B"

METHOD_STYLES = {
    COOLING_COMPARISON: (BLUE, "o", "-"),
    EMPIRICAL_COMPARISON: (ORANGE, "s", "--"),
    TRANSLATION_AVERAGED_ULMC: (GREEN, "^", "-."),
    RAW_ULMC: (GRAY, "X", ":"),
}
PLOT_LABELS = {
    COOLING_COMPARISON: "Translation-invariant Gaussian cooling",
    EMPIRICAL_COMPARISON: ("Translation-invariant empirical preconditioning"),
    TRANSLATION_AVERAGED_ULMC: "Translation-averaged ULMC covariance",
    RAW_ULMC: "Full empirical ULMC covariance",
}


def _configure_plot_style() -> None:
    """Apply the common typography and line styling to subsequent figures."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.labelsize": 10.2,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "figure.dpi": 140,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _phi4_parameter_subtitle(args: argparse.Namespace) -> str:
    """Return concise target and method-budget metadata for phi4 plots."""

    radius_text = r"\infty" if np.isposinf(args.radius) else f"{args.radius:g}"
    design_text = ""
    return (
        rf"$\beta={args.beta:g},\ \lambda={args.quartic:g},\ "
        + rf"m={args.mass:g},\ R={radius_text}"
        + design_text
        + rf";\ n={args.chains},\ N={args.steps},\ K={args.stages}$"
    )


def _plot_median_iqr(
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    marker: str,
    linestyle: str,
    label: str,
) -> None:
    """Plot rowwise medians and interquartile bands across repeated runs.

    ``values`` has one row per x coordinate and one column per repeat.
    Rows without finite observations are omitted.
    """

    valid_rows = np.any(np.isfinite(values), axis=1)
    if not np.any(valid_rows):
        return
    x_valid = x[valid_rows]
    data = values[valid_rows]
    median = np.nanmedian(data, axis=1)
    lower, upper = np.nanquantile(data, (0.25, 0.75), axis=1)
    ax.fill_between(
        x_valid,
        lower,
        upper,
        color=color,
        alpha=0.15,
        linewidth=0,
    )
    ax.plot(
        x_valid,
        median,
        color=color,
        marker=marker,
        linestyle=linestyle,
        markersize=4.2,
        label=label,
    )


def _logarithmic_cell_edges(values: np.ndarray) -> np.ndarray:
    """Return geometric cell edges centered on positive axis values."""

    values = np.asarray(values, dtype=float)
    logarithms = np.log(values)
    if len(values) == 1:
        half_width = 0.5 * np.log(2.0)
        return np.exp([logarithms[0] - half_width, logarithms[0] + half_width])
    midpoints = 0.5 * (logarithms[:-1] + logarithms[1:])
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = midpoints
    edges[0] = logarithms[0] - (midpoints[0] - logarithms[0])
    edges[-1] = logarithms[-1] + (logarithms[-1] - midpoints[-1])
    return np.exp(edges)


def _configure_hardness_axes(
    ax: plt.Axes,
    masses: np.ndarray,
    quartics: np.ndarray,
) -> None:
    """Label the two physical hardness-map axes without ambiguity."""

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(masses, [f"{value:g}" for value in masses])
    ax.set_yticks(quartics, [f"{value:g}" for value in quartics])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Mass $m$")
    ax.set_ylabel(r"Quartic coupling $\lambda$")


def save_publication_figures(
    figures: dict[str, plt.Figure],
    output: Path,
) -> dict[str, Path]:
    """Save each plot as ``<output_stem>_<plot_name>.pdf`` and return its path.

    Any suffix on ``output`` is replaced; parent directories are created
    as needed. Figures remain open for the caller to manage.
    """

    output = output.expanduser().resolve()
    stem = output.with_suffix("") if output.suffix else output
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for plot_name, figure in figures.items():
        path = stem.with_name(f"{stem.name}_{plot_name}").with_suffix(".pdf")
        figure.savefig(path, bbox_inches="tight")
        paths[plot_name] = path
    return paths


__all__ = [
    "BLUE",
    "ORANGE",
    "GREEN",
    "PURPLE",
    "GRAY",
    "METHOD_STYLES",
    "PLOT_LABELS",
    "save_publication_figures",
]
