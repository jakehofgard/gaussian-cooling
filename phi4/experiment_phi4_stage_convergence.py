"""Stagewise convergence of lattice phi4 covariance preconditioners."""

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
    GPU_STAGE_COMPARISON_SIDES,
    RAW_ULMC,
    TRANSLATION_AVERAGED_ULMC,
    _fourier_stage_history,
    _make_unpreconditioned_stage_history_call,
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
    METHOD_STYLES,
    PLOT_LABELS,
    _configure_plot_style,
    _phi4_parameter_subtitle,
    _plot_median_iqr,
    save_publication_figures,
)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


@dataclass
class StageConvergenceResult:
    """Condition histories for all feasible methods at each lattice side."""

    sides: np.ndarray
    conditions: dict[int, dict[str, np.ndarray]]
    raw_skip_reasons: dict[int, str]
    elapsed_seconds: float


def run_stage_convergence(args: argparse.Namespace) -> StageConvergenceResult:
    """Run equal-budget stage histories without the size-scaling benchmark."""

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    sides = np.asarray(args.sides, dtype=int)
    root_key = random.PRNGKey(args.seed)
    conditions: dict[int, dict[str, np.ndarray]] = {}
    raw_skip_reasons: dict[int, str] = {}
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
        key_side_index = side_index
        if args.key_side_order is not None:
            key_side_index = args.key_side_order.index(side)
        base_key = random.fold_in(root_key, key_side_index)
        print(
            f"Stage convergence at side={side} (D={model.dimension}), "
            f"{args.repeats} repeat(s)",
            flush=True,
        )
        reference = reference_samples(random.fold_in(base_key, 10_003), model, args)
        reference.block_until_ready()
        reference_spectrum = np.asarray(
            sample_power_spectrum(reference, model.lattice_shape)
        )
        raw_reason: str | None = None
        if args.chains <= model.dimension:
            raw_reason = f"n={args.chains} <= D={model.dimension} (rank deficient)"
        elif side > args.dense_max_side:
            raw_reason = f"side>{args.dense_max_side} full-covariance cutoff"
        if raw_reason is not None:
            raw_skip_reasons[side] = raw_reason
            print(f"  Raw full covariance omitted: {raw_reason}", flush=True)

        reference_covariance = None
        if raw_reason is None:
            reference_covariance = np.asarray(
                translation_invariant_covariance(
                    jnp.asarray(reference_spectrum, dtype=dtype),
                    model.lattice_shape,
                )
            )
        stage_shape = (args.stages + 1, args.repeats)
        comparison = {
            method: np.full(stage_shape, np.nan, dtype=float) for method in FOUR_METHODS
        }
        initial_spectrum = (
            np.ones(model.dimension, dtype=float) / model.design_smoothness
        )
        initial_condition = spectral_relative_condition(
            initial_spectrum,
            reference_spectrum,
            args.metric_floor,
        )
        for values in comparison.values():
            values[0, :] = initial_condition

        plain_history_call = _make_unpreconditioned_stage_history_call(
            model,
            args,
            include_raw_covariances=(raw_reason is None),
        )
        for repeat in range(args.repeats):
            comparison_key = random.fold_in(base_key, 60_000 + repeat)
            adaptive_histories = {
                COOLING_COMPARISON: _fourier_stage_history(
                    comparison_key,
                    model,
                    args,
                    use_cooling=True,
                ),
                EMPIRICAL_COMPARISON: _fourier_stage_history(
                    comparison_key,
                    model,
                    args,
                    use_cooling=False,
                ),
            }
            plain_output = jax.block_until_ready(
                plain_history_call(random.fold_in(comparison_key, 70_001))
            )
            if raw_reason is None:
                translation_history, raw_history = plain_output
                raw_history_array: np.ndarray | None = np.asarray(raw_history)
            else:
                translation_history = plain_output
                raw_history_array = None
            adaptive_histories[TRANSLATION_AVERAGED_ULMC] = np.asarray(
                translation_history
            )

            for method, history in adaptive_histories.items():
                for stage_index, spectrum in enumerate(history):
                    comparison[method][stage_index, repeat] = (
                        spectral_relative_condition(
                            spectrum,
                            reference_spectrum,
                            args.metric_floor,
                        )
                    )
            if raw_history_array is not None:
                assert reference_covariance is not None
                for stage_index, covariance in enumerate(raw_history_array, start=1):
                    comparison[RAW_ULMC][stage_index, repeat] = (
                        dense_relative_condition(
                            covariance,
                            reference_covariance,
                            0.0,
                        )
                    )
        conditions[side] = comparison

    return StageConvergenceResult(
        sides=sides,
        conditions=conditions,
        raw_skip_reasons=raw_skip_reasons,
        elapsed_seconds=time.perf_counter() - started,
    )


def make_stage_figures(
    result: StageConvergenceResult,
    args: argparse.Namespace,
) -> dict[str, plt.Figure]:
    """Create one stagewise-convergence PDF per lattice side."""

    _configure_plot_style()
    figures: dict[str, plt.Figure] = {}
    stages = np.arange(args.stages + 1)
    for side in result.sides:
        side = int(side)
        figure, ax = plt.subplots(figsize=(6.6, 4.9), constrained_layout=True)
        for method in FOUR_METHODS:
            if method == RAW_ULMC and side in result.raw_skip_reasons:
                continue
            color, marker, linestyle = METHOD_STYLES[method]
            _plot_median_iqr(
                ax,
                stages,
                result.conditions[side][method],
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=PLOT_LABELS[method],
            )
        ax.axhline(1.0, color="#222222", linestyle=":", linewidth=0.9)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.set_xlabel(r"Cumulative stage $k$")
        ax.set_ylabel(r"Relative condition number $\kappa_{\mathrm{rel}}$")
        ax.set_title(
            rf"Stagewise preconditioner convergence, $d={side}$"
            + "\n"
            + _phi4_parameter_subtitle(args)
        )
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            fontsize=7.2,
            ncols=2,
        )
        figures[f"stage_comparison_d{side}"] = figure
    return figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare stagewise convergence of phi4 preconditioners.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sides",
        "--comparison-sides",
        dest="sides",
        type=_parse_sides,
        default=[10, 100],
    )
    parser.add_argument(
        "--key-side-order",
        type=_parse_sides,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--quartic", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=0.25)
    parser.add_argument("--radius", type=float, default=2.0)
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
        default=Path(__file__).resolve().parents[1] / "figures" / "phi4_stages",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use sides 512 and 1024 and require a visible GPU.",
    )
    parser.add_argument("--quick", action="store_true")
    return parser


def apply_presets(args: argparse.Namespace, explicit: set[str]) -> None:
    if args.quick and args.gpu:
        raise ValueError("--quick and --gpu are mutually exclusive presets.")
    values: dict[str, object] = {}
    if args.quick:
        values = {
            "sides": [4, 8],
            "key_side_order": [4, 6, 8],
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
            "sides": list(GPU_STAGE_COMPARISON_SIDES),
            "key_side_order": [64, 128, 256, 512, 1024],
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
    if "sides" in explicit and "key_side_order" not in explicit:
        args.key_side_order = None


def validate_arguments(args: argparse.Namespace) -> None:
    scalars = (
        args.quartic,
        args.mass,
        args.cooling_design_radius,
        args.delta,
        args.friction,
        args.step_size,
        args.metric_floor,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in scalars):
        raise ValueError(
            "Physical, sampler, and metric scales must be finite and positive."
        )
    if not np.isfinite(args.beta) or args.beta < 0.0:
        raise ValueError("--beta must be finite and nonnegative.")
    if not (np.isfinite(args.radius) or np.isposinf(args.radius)) or args.radius <= 0.0:
        raise ValueError("--radius must be positive or inf.")
    if not 0.0 < args.cooling_gamma < 1.0:
        raise ValueError("--cooling-gamma must lie in (0, 1).")
    if not np.isfinite(args.covariance_ridge) or args.covariance_ridge < 0.0:
        raise ValueError("--covariance-ridge must be finite and nonnegative.")
    if args.dense_max_side < 0:
        raise ValueError("--dense-max-side must be nonnegative.")
    counts = (
        args.repeats,
        args.chains,
        args.steps,
        args.stages,
        args.reference_chains,
        args.reference_steps,
    )
    if any(value < 1 for value in counts) or args.chains < 2:
        raise ValueError("Counts must be positive and --chains at least two.")
    if args.reference_chains < 4:
        raise ValueError("--reference-chains must be at least four.")
    if args.key_side_order is not None:
        missing_sides = sorted(set(args.sides) - set(args.key_side_order))
        if missing_sides:
            raise ValueError(
                "--key-side-order must contain every requested side; missing "
                + ", ".join(map(str, missing_sides))
                + "."
            )
    if args.reference_step_size is not None and (
        not np.isfinite(args.reference_step_size) or args.reference_step_size <= 0.0
    ):
        raise ValueError("--reference-step-size must be finite and positive.")
    if np.isposinf(args.radius):
        warnings.warn(
            "R=inf uses the design radius only as an operational scale.",
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


def print_summary(result: StageConvergenceResult) -> None:
    print("\nFinal-stage median relative condition numbers")
    for side in result.sides:
        side = int(side)
        print(f"d={side}")
        for method in FOUR_METHODS:
            values = result.conditions[side][method][-1]
            values = values[np.isfinite(values)]
            text = f"{np.median(values):.4g}" if values.size else "omitted"
            print(f"  {PLOT_LABELS[method]}: {text}")
        if side in result.raw_skip_reasons:
            print(f"  Raw omission: {result.raw_skip_reasons[side]}")
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
    result = run_stage_convergence(args)
    figures = make_stage_figures(result, args)
    paths = save_publication_figures(figures, args.output)
    for figure in figures.values():
        plt.close(figure)
    print_summary(result)
    for name, path in paths.items():
        print(f"Saved {name.replace('_', ' ')} PDF: {path}")


if __name__ == "__main__":
    main()
