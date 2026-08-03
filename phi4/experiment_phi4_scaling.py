"""Lattice-size quality and runtime comparison for phi4 preconditioners.

This focused entry point owns the lattice-side scaling experiment.  It
compares the four primary covariance estimators, with dense Gaussian cooling
as an ancillary small-dimensional benchmark, and writes one vector PDF for
each quality or timing panel.  Other phi4 workflows have their own scripts.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import random

from gaussian_cooling_algs import (
    sample_power_spectrum,
    translation_invariant_covariance,
)
from .lattice_phi4 import (
    COOLING_COMPARISON,
    EMPIRICAL_COMPARISON,
    FOUR_METHODS,
    GPU_LATTICE_SIDES,
    RAW_ULMC,
    TRANSLATION_AVERAGED_ULMC,
    _dense_cooling_call,
    _fourier_cooling_call,
    _fourier_empirical_baseline_call,
    _raw_ulmc_covariance_call,
    _run_timed_repeats,
    _translation_averaged_ulmc_call,
    dense_relative_condition,
    make_lattice_model,
    reference_samples,
    spectral_relative_condition,
    validate_lattice_model,
)
from .phi4_cli import (
    _available_gpu_devices,
    _explicit_cli_destinations,
    _parse_sides,
)
from .phi4_plotting import (
    BLUE,
    METHOD_STYLES,
    PLOT_LABELS,
    PURPLE,
    _configure_plot_style,
    _phi4_parameter_subtitle,
    _plot_median_iqr,
    save_publication_figures,
)

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


@dataclass
class ScalingResult:
    """Quality and post-compilation timings across lattice sides."""

    sides: np.ndarray
    relative_conditions: dict[str, np.ndarray]
    runtimes: dict[str, np.ndarray]
    dense_cooling_conditions: np.ndarray
    dense_cooling_runtimes: np.ndarray
    dense_skip_reasons: dict[int, str]
    raw_skip_reasons: dict[int, str]
    elapsed_seconds: float


def run_scaling_experiment(args: argparse.Namespace) -> ScalingResult:
    """Run only the lattice-size quality and timing experiment."""

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    sides = np.asarray(args.sides, dtype=int)
    shape = (len(sides), args.repeats)
    conditions = {
        method: np.full(shape, np.nan, dtype=float) for method in FOUR_METHODS
    }
    runtimes = {method: np.full(shape, np.nan, dtype=float) for method in FOUR_METHODS}
    dense_conditions = np.full(shape, np.nan, dtype=float)
    dense_runtimes = np.full(shape, np.nan, dtype=float)
    dense_skip_reasons: dict[int, str] = {}
    raw_skip_reasons: dict[int, str] = {}
    root_key = random.PRNGKey(args.seed)
    started = time.perf_counter()

    for side_index, side_value in enumerate(sides):
        side = int(side_value)
        model = make_lattice_model(
            side,
            args.beta,
            args.quartic,
            args.mass,
            args.radius,
            dtype,
            design_radius=(
                args.cooling_design_radius if np.isposinf(args.radius) else None
            ),
        )
        validate_lattice_model(model)
        base_key = random.fold_in(root_key, side_index)
        fourier_keys = list(
            random.split(
                random.fold_in(base_key, 10_004),
                args.repeats,
            )
        )
        empirical_root = random.fold_in(base_key, 80_000)
        key_warm_empirical, key_empirical_runs = random.split(empirical_root)
        empirical_keys = list(random.split(key_empirical_runs, args.repeats))
        plain_root = random.fold_in(base_key, 70_000)
        key_warm_plain, key_plain_runs = random.split(plain_root)
        plain_keys = list(random.split(key_plain_runs, args.repeats))

        print(
            f"side={side:4d} (D={model.dimension:8d}): "
            "translation-invariant Gaussian cooling",
            flush=True,
        )
        spectra, elapsed = _run_timed_repeats(
            lambda key, model=model: _fourier_cooling_call(key, model, args),
            fourier_keys,
            random.fold_in(base_key, 10_001),
        )
        runtimes[COOLING_COMPARISON][side_index] = elapsed

        print("  Translation-invariant empirical preconditioning", flush=True)
        empirical_spectra, elapsed = _run_timed_repeats(
            lambda key, model=model: _fourier_empirical_baseline_call(key, model, args),
            empirical_keys,
            key_warm_empirical,
        )
        runtimes[EMPIRICAL_COMPARISON][side_index] = elapsed

        print("  Translation-averaged unpreconditioned ULMC", flush=True)
        translation_spectra, elapsed = _run_timed_repeats(
            lambda key, model=model: _translation_averaged_ulmc_call(key, model, args),
            plain_keys,
            key_warm_plain,
        )
        runtimes[TRANSLATION_AVERAGED_ULMC][side_index] = elapsed

        raw_reason: str | None = None
        if args.chains <= model.dimension:
            raw_reason = f"n={args.chains} <= D={model.dimension} (rank deficient)"
        elif side > args.dense_max_side:
            raw_reason = f"side>{args.dense_max_side} full-covariance cutoff"
        raw_covariances: list[np.ndarray] = []
        if raw_reason is None:
            print("  Full empirical covariance from plain ULMC", flush=True)
            raw_covariances, elapsed = _run_timed_repeats(
                lambda key, model=model: _raw_ulmc_covariance_call(key, model, args),
                plain_keys,
                key_warm_plain,
            )
            runtimes[RAW_ULMC][side_index] = elapsed
        else:
            raw_skip_reasons[side] = raw_reason
            print(f"  Full covariance omitted: {raw_reason}", flush=True)

        dense_reason: str | None = None
        if side > args.dense_max_side:
            dense_reason = f"side>{args.dense_max_side} dense cutoff"
        elif args.chains <= model.dimension:
            dense_reason = f"n={args.chains} <= D={model.dimension} (rank deficient)"

        reference = reference_samples(random.fold_in(base_key, 10_003), model, args)
        reference.block_until_ready()
        reference_spectrum = np.asarray(
            sample_power_spectrum(reference, model.lattice_shape)
        )
        reference_covariance: np.ndarray | None = None
        if raw_reason is None or dense_reason is None:
            reference_covariance = np.asarray(
                translation_invariant_covariance(
                    jnp.asarray(reference_spectrum, dtype=dtype),
                    model.lattice_shape,
                )
            )

        spectral_batches = {
            COOLING_COMPARISON: spectra,
            EMPIRICAL_COMPARISON: empirical_spectra,
            TRANSLATION_AVERAGED_ULMC: translation_spectra,
        }
        for method, estimates in spectral_batches.items():
            for repeat, estimate in enumerate(estimates):
                conditions[method][side_index, repeat] = spectral_relative_condition(
                    estimate,
                    reference_spectrum,
                    args.metric_floor,
                )
        if raw_reason is None:
            assert reference_covariance is not None
            for repeat, covariance in enumerate(raw_covariances):
                conditions[RAW_ULMC][side_index, repeat] = dense_relative_condition(
                    covariance,
                    reference_covariance,
                    0.0,
                )

        if dense_reason is not None:
            dense_skip_reasons[side] = dense_reason
            print(f"  Dense Gaussian cooling omitted: {dense_reason}", flush=True)
            continue

        print("  Dense Gaussian cooling", flush=True)
        dense_keys = list(
            random.split(
                random.fold_in(base_key, 10_005),
                args.repeats,
            )
        )
        dense_covariances, elapsed = _run_timed_repeats(
            lambda key, model=model: _dense_cooling_call(key, model, args),
            dense_keys,
            random.fold_in(base_key, 10_002),
        )
        dense_runtimes[side_index] = elapsed
        assert reference_covariance is not None
        for repeat, covariance in enumerate(dense_covariances):
            dense_conditions[side_index, repeat] = dense_relative_condition(
                covariance,
                reference_covariance,
                args.metric_ridge,
            )

    return ScalingResult(
        sides=sides,
        relative_conditions=conditions,
        runtimes=runtimes,
        dense_cooling_conditions=dense_conditions,
        dense_cooling_runtimes=dense_runtimes,
        dense_skip_reasons=dense_skip_reasons,
        raw_skip_reasons=raw_skip_reasons,
        elapsed_seconds=time.perf_counter() - started,
    )


def make_scaling_figures(
    result: ScalingResult,
    args: argparse.Namespace,
) -> dict[str, plt.Figure]:
    """Create the four standalone scaling figures."""

    _configure_plot_style()
    figures: dict[str, plt.Figure] = {}
    sides = result.sides.astype(float)
    dimensions = sides**2

    quality_figure, ax = plt.subplots(figsize=(6.5, 4.7), constrained_layout=True)
    for method in FOUR_METHODS:
        color, marker, linestyle = METHOD_STYLES[method]
        _plot_median_iqr(
            ax,
            sides,
            result.relative_conditions[method],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=PLOT_LABELS[method],
        )
    ax.axhline(1.0, color="#222222", linestyle=":", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sides, [str(int(side)) for side in sides])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Lattice side length $d$")
    ax.set_ylabel(r"Relative condition number $\kappa_{\mathrm{rel}}$")
    ax.set_title(r"$\kappa_{\mathrm{rel}}$ vs. $d$\n" + _phi4_parameter_subtitle(args))
    ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncols=2,
        fontsize=7.4,
    )
    figures["preconditioner_quality"] = quality_figure

    runtime_figure, ax = plt.subplots(figsize=(6.5, 4.7), constrained_layout=True)
    for method in FOUR_METHODS:
        color, marker, linestyle = METHOD_STYLES[method]
        _plot_median_iqr(
            ax,
            dimensions,
            result.runtimes[method],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=PLOT_LABELS[method],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of lattice sites $D=d^2$")
    ax.set_ylabel("Wall time after JIT compilation (s)")
    ax.set_title("Covariance-estimation cost\n" + _phi4_parameter_subtitle(args))
    ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncols=2,
        fontsize=7.4,
    )
    figures["covariance_estimation_cost"] = runtime_figure

    if np.any(np.isfinite(result.dense_cooling_conditions)):
        dense_quality, ax = plt.subplots(figsize=(5.2, 3.9), constrained_layout=True)
        _plot_median_iqr(
            ax,
            sides,
            result.relative_conditions[COOLING_COMPARISON],
            color=BLUE,
            marker="o",
            linestyle="-",
            label=COOLING_COMPARISON,
        )
        _plot_median_iqr(
            ax,
            sides,
            result.dense_cooling_conditions,
            color=PURPLE,
            marker="D",
            linestyle="--",
            label="Dense Gaussian cooling",
        )
        ax.axhline(1.0, color="#222222", linestyle=":", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sides, [str(int(side)) for side in sides])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"Lattice side length $d$")
        ax.set_ylabel(r"Relative condition number $\kappa_{\mathrm{rel}}$")
        ax.set_title(
            "Dense vs. translation-invariant Gaussian cooling\n"
            + _phi4_parameter_subtitle(args)
        )
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        ax.legend(
            frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2
        )
        figures["dense_cooling_quality"] = dense_quality

        dense_cost, ax = plt.subplots(figsize=(5.2, 3.9), constrained_layout=True)
        _plot_median_iqr(
            ax,
            dimensions,
            result.runtimes[COOLING_COMPARISON],
            color=BLUE,
            marker="o",
            linestyle="-",
            label=COOLING_COMPARISON,
        )
        _plot_median_iqr(
            ax,
            dimensions,
            result.dense_cooling_runtimes,
            color=PURPLE,
            marker="D",
            linestyle="--",
            label="Dense Gaussian cooling",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Number of lattice sites $D=d^2$")
        ax.set_ylabel("Wall time after JIT compilation (s)")
        ax.set_title("Dense Gaussian-cooling cost\n" + _phi4_parameter_subtitle(args))
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        ax.legend(
            frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2
        )
        figures["dense_cooling_cost"] = dense_cost

    return figures


def build_parser() -> argparse.ArgumentParser:
    """Build the focused scaling CLI."""

    parser = argparse.ArgumentParser(
        description="Compare phi4 preconditioner quality and cost across lattice sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sides", type=_parse_sides, default=[5, 10, 20, 50, 100])
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument(
        "--quartic", type=float, default=0.5, help="Quartic coupling lambda."
    )
    parser.add_argument("--mass", type=float, default=0.25)
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Use inf for the genuine quartic target.",
    )
    parser.add_argument("--cooling-design-radius", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chains", type=int, default=512)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--cooling-gamma", type=float, default=0.35)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--covariance-ridge", type=float, default=0.0)
    parser.add_argument("--metric-floor", type=float, default=1e-10)
    parser.add_argument("--metric-ridge", type=float, default=0.0)
    parser.add_argument("--dense-max-side", type=int, default=20)
    parser.add_argument("--reference-chains", type=int, default=256)
    parser.add_argument("--reference-steps", type=int, default=256)
    parser.add_argument("--reference-step-size", type=float, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--output-prefix",
        "--output",
        dest="output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "phi4_scaling",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use the H100-scale side preset and require a visible JAX GPU.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small deterministic smoke-test preset.",
    )
    return parser


def apply_presets(
    args: argparse.Namespace,
    explicit: set[str],
) -> None:
    """Apply quick or GPU values without overriding explicit options."""

    if args.quick and args.gpu:
        raise ValueError("--quick and --gpu are mutually exclusive presets.")
    values: dict[str, object] = {}
    if args.quick:
        values = {
            "sides": [4, 6, 8],
            "repeats": 1,
            "chains": 64,
            "steps": 20,
            "stages": 4,
            "dense_max_side": 8,
            "reference_chains": 96,
            "reference_steps": 48,
            "cooling_design_radius": 2.0,
            "dtype": "float32",
        }
    elif args.gpu:
        values = {
            "sides": list(GPU_LATTICE_SIDES),
            "repeats": 1,
            "chains": 64,
            "steps": 32,
            "stages": 8,
            "dense_max_side": 0,
            "reference_chains": 128,
            "reference_steps": 128,
            "dtype": "float32",
        }
    for destination, value in values.items():
        if destination not in explicit:
            setattr(args, destination, value)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate only options consumed by this experiment."""

    positive = {
        "quartic": args.quartic,
        "mass": args.mass,
        "cooling_design_radius": args.cooling_design_radius,
        "delta": args.delta,
        "friction": args.friction,
        "step_size": args.step_size,
        "metric_floor": args.metric_floor,
    }
    invalid = [
        name
        for name, value in positive.items()
        if not np.isfinite(value) or value <= 0.0
    ]
    if invalid:
        raise ValueError(
            "These options must be finite and positive: " + ", ".join(invalid)
        )
    if not np.isfinite(args.beta) or args.beta < 0.0:
        raise ValueError("--beta must be finite and nonnegative.")
    if not (np.isfinite(args.radius) or np.isposinf(args.radius)) or args.radius <= 0.0:
        raise ValueError("--radius must be positive or inf.")
    if not 0.0 < args.cooling_gamma < 1.0:
        raise ValueError("--cooling-gamma must lie in (0, 1).")
    ridges = (args.covariance_ridge, args.metric_ridge)
    if any(not np.isfinite(value) or value < 0.0 for value in ridges):
        raise ValueError("Ridge parameters must be finite and nonnegative.")
    integer_positive = (
        args.repeats,
        args.chains,
        args.steps,
        args.stages,
        args.reference_chains,
        args.reference_steps,
    )
    if any(value < 1 for value in integer_positive) or args.chains < 2:
        raise ValueError(
            "Repeats, chains, steps, stages, and reference budgets must be positive; --chains must be at least two."
        )
    if args.reference_chains < 4:
        raise ValueError("--reference-chains must be at least four.")
    if args.dense_max_side < 0:
        raise ValueError("--dense-max-side must be nonnegative.")
    if args.reference_step_size is not None and (
        not np.isfinite(args.reference_step_size) or args.reference_step_size <= 0.0
    ):
        raise ValueError("--reference-step-size must be finite and positive.")
    if np.isposinf(args.radius):
        warnings.warn(
            "R=inf is a genuine quartic target; the design radius is an operational tuning scale, not a global Hessian bound.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.gpu and args.dtype == "float64":
        warnings.warn(
            "Float64 roughly doubles large-lattice state and workspace memory relative to float32.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.gpu and not _available_gpu_devices():
        raise RuntimeError("--gpu requires a visible JAX GPU device.")


def print_summary(result: ScalingResult) -> None:
    """Print compact quality and omission summaries."""

    print("\nMedian relative condition numbers")
    header = "side".ljust(7) + "".join(
        PLOT_LABELS[m][:17].rjust(19) for m in FOUR_METHODS
    )
    print(header)
    for index, side in enumerate(result.sides):
        fields = []
        for method in FOUR_METHODS:
            finite = result.relative_conditions[method][index]
            finite = finite[np.isfinite(finite)]
            fields.append(
                (f"{np.median(finite):.4g}" if finite.size else "omitted").rjust(19)
            )
        print(str(int(side)).ljust(7) + "".join(fields))
    for side, reason in result.raw_skip_reasons.items():
        print(f"Raw covariance at d={side}: {reason}")
    for side, reason in result.dense_skip_reasons.items():
        print(f"Dense cooling at d={side}: {reason}")
    print(f"Wall time including compilation: {result.elapsed_seconds:.2f}s")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    explicit = _explicit_cli_destinations(parser, arguments)
    try:
        apply_presets(args, explicit)
        validate_arguments(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    result = run_scaling_experiment(args)
    figures = make_scaling_figures(result, args)
    paths = save_publication_figures(figures, args.output)
    for figure in figures.values():
        plt.close(figure)
    print_summary(result)
    for name, path in paths.items():
        print(f"Saved {name.replace('_', ' ')} PDF: {path}")


if __name__ == "__main__":
    main()
