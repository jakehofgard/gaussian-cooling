r"""Controlled parameter sweeps for periodic lattice :math:`\phi^4` targets.

This focused experiment compares four equal-budget covariance-preconditioning
methods while varying one physical parameter at a time: the quartic coupling
``lambda``, the mass ``m``, or the truncation radius ``R``.  The radius sweep
includes ``R=inf`` by default, corresponding to the genuine quartic target.

Every figure is written as a separate vector PDF.  Run ``--quick`` before a
production sweep to exercise the complete numerical and plotting path at
small lattice sizes.
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
import matplotlib
import numpy as np
from jax import random

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from gaussian_cooling_algs import (
    sample_power_spectrum,
    translation_invariant_covariance,
)
from .lattice_phi4 import (
    COOLING_COMPARISON,
    EMPIRICAL_COMPARISON,
    FOUR_METHODS,
    RAW_ULMC,
    TRANSLATION_AVERAGED_ULMC,
    LatticeModel,
    _fourier_cooling_call,
    _fourier_empirical_baseline_call,
    _run_timed_repeats,
    _safe_spectrum,
    _unpreconditioned_estimators_call,
    dense_relative_condition,
    make_lattice_model,
    reference_samples,
    spectral_relative_condition,
    validate_lattice_model,
)
from .phi4_cli import (
    _explicit_cli_destinations,
    _parse_positive_values,
    _parse_target_radii,
)
from .phi4_plotting import (
    METHOD_STYLES,
    PLOT_LABELS,
    PURPLE,
    _configure_plot_style,
    _plot_median_iqr,
    save_publication_figures,
)


@dataclass
class ParameterSweepResult:
    """Controlled one-parameter sweeps for the four primary methods.

    ``relative_conditions[sweep][method]`` has axes ``(value, repeat)``.
    Other dictionaries hold one reference or target diagnostic per value.
    """

    sweep_side: int
    values: dict[str, np.ndarray]
    relative_conditions: dict[str, dict[str, np.ndarray]]
    hessian_condition_bounds: dict[str, np.ndarray]
    reference_conditions: dict[str, np.ndarray]
    continuation_fractions: dict[str, np.ndarray]
    design_exceedance_fractions: dict[str, np.ndarray]
    truncation_to_quartic_conditions: np.ndarray
    reference_step_sizes: dict[str, np.ndarray]
    reference_steps: dict[str, np.ndarray]
    stages: int
    reference_chains: int
    unique_target_count: int
    elapsed_seconds: float


@dataclass
class _PrimaryMethodEvaluation:
    """Four-method output for one lattice target in a parameter sweep."""

    relative_conditions: dict[str, np.ndarray]
    hessian_condition_bound: float
    reference_condition: float
    continuation_fraction: float
    design_exceedance_fraction: float
    reference_spectrum: np.ndarray
    reference_step_size: float
    reference_steps: int


def _parameter_reference_arguments(
    model: LatticeModel,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Limit the reference step size while preserving its integration time.

    The step count is rounded up so each target is sampled for at least
    ``reference_time``, even when its curvature requires smaller steps.
    """

    reference_args = argparse.Namespace(**vars(args))
    transformed_design_smoothness = (
        1.0 + 3.0 * model.quartic * model.design_radius**2 / model.mass
    )
    requested_step_size = (
        args.step_size if args.reference_step_size is None else args.reference_step_size
    )
    stable_step_size = args.reference_margin / np.sqrt(transformed_design_smoothness)
    reference_step_size = min(requested_step_size, stable_step_size)
    reference_args.reference_step_size = reference_step_size
    reference_args.reference_steps = max(
        1,
        int(np.ceil(args.reference_time / reference_step_size)),
    )
    return reference_args


def _evaluate_primary_methods(
    key: jax.Array,
    model: LatticeModel,
    args: argparse.Namespace,
) -> _PrimaryMethodEvaluation:
    """Evaluate all four primary methods for one target configuration."""

    if args.chains <= model.dimension:
        raise ValueError(
            "The four-method parameter sweep requires n>D so the raw "
            "empirical covariance is nonsingular."
        )

    # Adaptive methods share keys; plain ULMC and the reference use separate streams.
    adaptive_root, plain_root, reference_key = random.split(key, 3)
    adaptive_warm_key, adaptive_run_root = random.split(adaptive_root)
    adaptive_keys = list(random.split(adaptive_run_root, args.repeats))
    plain_warm_key, plain_run_root = random.split(plain_root)
    plain_keys = list(random.split(plain_run_root, args.repeats))

    def cooling_call(run_key: jax.Array) -> jax.Array:
        return _fourier_cooling_call(run_key, model, args)

    cooling_spectra, _ = _run_timed_repeats(
        cooling_call,
        adaptive_keys,
        adaptive_warm_key,
    )

    def empirical_call(run_key: jax.Array) -> jax.Array:
        return _fourier_empirical_baseline_call(run_key, model, args)

    empirical_spectra, _ = _run_timed_repeats(
        empirical_call,
        adaptive_keys,
        adaptive_warm_key,
    )

    def plain_call(run_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _unpreconditioned_estimators_call(run_key, model, args)

    # Both plain-ULMC estimators use the same sampled endpoints.
    jax.block_until_ready(plain_call(plain_warm_key))
    translation_averaged_spectra: list[np.ndarray] = []
    raw_covariances: list[np.ndarray] = []
    for plain_key in plain_keys:
        spectrum, covariance = jax.block_until_ready(plain_call(plain_key))
        translation_averaged_spectra.append(np.asarray(spectrum))
        raw_covariances.append(np.asarray(covariance))

    reference_args = _parameter_reference_arguments(model, args)
    reference = reference_samples(reference_key, model, reference_args)
    reference.block_until_ready()
    reference_spectrum = np.asarray(
        sample_power_spectrum(reference, model.lattice_shape)
    )
    reference_covariance = np.asarray(
        translation_invariant_covariance(
            jnp.asarray(reference_spectrum, dtype=model.dtype),
            model.lattice_shape,
        )
    )

    relative_conditions = {
        COOLING_COMPARISON: np.asarray(
            [
                spectral_relative_condition(
                    spectrum,
                    reference_spectrum,
                    args.metric_floor,
                )
                for spectrum in cooling_spectra
            ]
        ),
        EMPIRICAL_COMPARISON: np.asarray(
            [
                spectral_relative_condition(
                    spectrum,
                    reference_spectrum,
                    args.metric_floor,
                )
                for spectrum in empirical_spectra
            ]
        ),
        TRANSLATION_AVERAGED_ULMC: np.asarray(
            [
                spectral_relative_condition(
                    spectrum,
                    reference_spectrum,
                    args.metric_floor,
                )
                for spectrum in translation_averaged_spectra
            ]
        ),
        RAW_ULMC: np.asarray(
            [
                dense_relative_condition(
                    covariance,
                    reference_covariance,
                    0.0,
                )
                for covariance in raw_covariances
            ]
        ),
    }
    safe_reference = _safe_spectrum(
        reference_spectrum,
        args.metric_floor,
    )
    reference_array = np.asarray(reference)
    continuation_fraction = (
        float(np.mean(np.abs(reference_array) > model.radius))
        if model.is_truncated
        else np.nan
    )
    design_exceedance_fraction = float(
        np.mean(np.abs(reference_array) > model.design_radius)
    )
    if not model.is_truncated and design_exceedance_fraction > 0.01:
        warnings.warn(
            "More than 1% of the genuine-quartic reference coordinates "
            f"exceeded R_design={model.design_radius:g}. Repeat with a "
            "larger --cooling-design-radius and a smaller step size to "
            "check tuning-scale sensitivity.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _PrimaryMethodEvaluation(
        relative_conditions=relative_conditions,
        hessian_condition_bound=(
            model.global_smoothness_bound / model.strong_convexity
        ),
        reference_condition=float(np.max(safe_reference) / np.min(safe_reference)),
        continuation_fraction=continuation_fraction,
        design_exceedance_fraction=design_exceedance_fraction,
        reference_spectrum=reference_spectrum,
        reference_step_size=float(reference_args.reference_step_size),
        reference_steps=int(reference_args.reference_steps),
    )


def run_parameter_sweeps(
    args: argparse.Namespace,
) -> ParameterSweepResult:
    """Run controlled lambda, mass, and radius sweeps at one lattice size."""

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    sweep_specs = {
        "quartic": np.asarray(args.quartic_sweep_values, dtype=float),
        "mass": np.asarray(args.mass_sweep_values, dtype=float),
        "radius": np.asarray(args.radius_sweep_values, dtype=float),
    }
    relative_conditions = {
        sweep_name: {
            method: np.empty((len(values), args.repeats), dtype=float)
            for method in FOUR_METHODS
        }
        for sweep_name, values in sweep_specs.items()
    }
    hessian_bounds = {
        name: np.empty(len(values), dtype=float) for name, values in sweep_specs.items()
    }
    reference_conditions = {
        name: np.empty(len(values), dtype=float) for name, values in sweep_specs.items()
    }
    continuation_fractions = {
        name: np.empty(len(values), dtype=float) for name, values in sweep_specs.items()
    }
    design_exceedance_fractions = {
        name: np.empty(len(values), dtype=float) for name, values in sweep_specs.items()
    }
    reference_spectra: dict[str, list[np.ndarray | None]] = {
        name: [None] * len(values) for name, values in sweep_specs.items()
    }
    reference_step_sizes = {
        name: np.empty(len(values), dtype=float) for name, values in sweep_specs.items()
    }
    reference_steps = {
        name: np.empty(len(values), dtype=int) for name, values in sweep_specs.items()
    }

    root_key = random.fold_in(random.PRNGKey(args.seed), 300_000)
    # Reuse targets appearing in multiple sweeps, including their reference samples.
    cache: dict[
        tuple[float, float, float, float],
        _PrimaryMethodEvaluation,
    ] = {}
    started = time.perf_counter()

    def configuration(
        sweep_name: str,
        value: float,
    ) -> tuple[float, float, float, float]:
        """Return the target and design radius with only one parameter varied."""

        quartic = value if sweep_name == "quartic" else args.parameter_sweep_quartic
        mass = value if sweep_name == "mass" else args.parameter_sweep_mass
        radius = value if sweep_name == "radius" else args.parameter_sweep_radius
        # The radius sweep uses one curvature scale, including its R=infinity target.
        design_radius = args.cooling_design_radius if sweep_name == "radius" else radius
        return (
            float(quartic),
            float(mass),
            float(radius),
            float(design_radius),
        )

    for sweep_name, values in sweep_specs.items():
        for value_index, value in enumerate(values):
            target_configuration = configuration(sweep_name, float(value))
            if target_configuration not in cache:
                quartic, mass, radius, design_radius = target_configuration
                model = make_lattice_model(
                    args.side,
                    args.beta,
                    quartic,
                    mass,
                    radius,
                    dtype,
                    design_radius=design_radius,
                )
                validate_lattice_model(model)
                # Common random numbers pair every target configuration,
                # reducing Monte Carlo noise in parameter-to-parameter trends.
                print(
                    "Parameter sweep target "
                    f"{len(cache) + 1}: d={args.side}, "
                    f"lambda={quartic:g}, m={mass:g}, R={radius:g}, "
                    f"R_design={design_radius:g}",
                    flush=True,
                )
                cache[target_configuration] = _evaluate_primary_methods(
                    root_key,
                    model,
                    args,
                )

            evaluation = cache[target_configuration]
            for method in FOUR_METHODS:
                relative_conditions[sweep_name][method][value_index] = (
                    evaluation.relative_conditions[method]
                )
            hessian_bounds[sweep_name][value_index] = evaluation.hessian_condition_bound
            reference_conditions[sweep_name][value_index] = (
                evaluation.reference_condition
            )
            continuation_fractions[sweep_name][value_index] = (
                evaluation.continuation_fraction
            )
            design_exceedance_fractions[sweep_name][value_index] = (
                evaluation.design_exceedance_fraction
            )
            reference_spectra[sweep_name][value_index] = evaluation.reference_spectrum
            reference_step_sizes[sweep_name][value_index] = (
                evaluation.reference_step_size
            )
            reference_steps[sweep_name][value_index] = evaluation.reference_steps

    radius_values = sweep_specs["radius"]
    truncation_to_quartic_conditions = np.full(
        len(radius_values),
        np.nan,
        dtype=float,
    )
    quartic_indices = np.flatnonzero(np.isposinf(radius_values))
    if quartic_indices.size:
        # Compare target covariances themselves, separately from estimator quality.
        quartic_spectrum = reference_spectra["radius"][int(quartic_indices[0])]
        assert quartic_spectrum is not None
        for value_index, finite_spectrum in enumerate(reference_spectra["radius"]):
            assert finite_spectrum is not None
            truncation_to_quartic_conditions[value_index] = spectral_relative_condition(
                finite_spectrum,
                quartic_spectrum,
                args.metric_floor,
            )

    return ParameterSweepResult(
        sweep_side=args.side,
        values=sweep_specs,
        relative_conditions=relative_conditions,
        hessian_condition_bounds=hessian_bounds,
        reference_conditions=reference_conditions,
        continuation_fractions=continuation_fractions,
        design_exceedance_fractions=design_exceedance_fractions,
        truncation_to_quartic_conditions=truncation_to_quartic_conditions,
        reference_step_sizes=reference_step_sizes,
        reference_steps=reference_steps,
        stages=args.stages,
        reference_chains=args.reference_chains,
        unique_target_count=len(cache),
        elapsed_seconds=time.perf_counter() - started,
    )


def _parameter_sweep_subtitle(
    sweep_name: str,
    result: ParameterSweepResult,
    args: argparse.Namespace,
) -> str:
    """Return fixed target and budget parameters for one sweep plot."""

    if sweep_name == "quartic":
        fixed_parameters = (
            rf"m={args.parameter_sweep_mass:g},\ " rf"R={args.parameter_sweep_radius:g}"
        )
    elif sweep_name == "mass":
        fixed_parameters = (
            rf"\lambda={args.parameter_sweep_quartic:g},\ "
            rf"R={args.parameter_sweep_radius:g}"
        )
    else:
        fixed_parameters = (
            rf"\lambda={args.parameter_sweep_quartic:g},\ "
            rf"m={args.parameter_sweep_mass:g},\ "
            rf"R_{{\rm design}}={args.cooling_design_radius:g}"
        )
    return (
        rf"$d={result.sweep_side},\ \beta={args.beta:g},\ "
        + fixed_parameters
        + rf";\ n={args.chains},\ N={args.steps},\ K={result.stages}$"
    )


def make_parameter_sweep_figures(
    result: ParameterSweepResult,
    args: argparse.Namespace,
) -> dict[str, plt.Figure]:
    """Create standalone sweep and truncation-to-quartic figures."""

    _configure_plot_style()
    plot_metadata = {
        "quartic": (
            r"Quartic coupling $\lambda$",
            "Four-method comparison across quartic coupling",
            "parameter_sweep_lambda",
        ),
        "mass": (
            r"Mass $m$",
            "Four-method comparison across mass",
            "parameter_sweep_mass",
        ),
        "radius": (
            r"Target radius $R$ ($\infty$: genuine quartic)",
            "Four-method comparison: truncated and genuine quartic targets",
            "parameter_sweep_radius",
        ),
    }
    figures: dict[str, plt.Figure] = {}
    for sweep_name, values in result.values.items():
        x_label, title, figure_name = plot_metadata[sweep_name]
        is_radius_sweep = sweep_name == "radius"
        # Treat radii as categories so the genuine-quartic endpoint fits on the axis.
        plot_values = np.arange(len(values), dtype=float) if is_radius_sweep else values
        figure, ax = plt.subplots(
            figsize=(6.5, 4.9),
            constrained_layout=True,
        )
        for method in FOUR_METHODS:
            color, marker, linestyle = METHOD_STYLES[method]
            _plot_median_iqr(
                ax,
                plot_values,
                result.relative_conditions[sweep_name][method],
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=PLOT_LABELS[method],
            )
        ax.axhline(1.0, color="#222222", linestyle=":", linewidth=0.9)
        if not is_radius_sweep:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(
            plot_values,
            [r"$\infty$" if np.isposinf(value) else f"{value:g}" for value in values],
        )
        if not is_radius_sweep:
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"Relative condition number $\kappa_{\mathrm{rel}}$")
        ax.set_title(title + "\n" + _parameter_sweep_subtitle(sweep_name, result, args))
        ax.grid(
            which="major",
            color="#D8D8D8",
            linewidth=0.55,
            alpha=0.8,
        )
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            ncols=2,
            fontsize=7.4,
        )
        figures[figure_name] = figure

    radius_conditions = result.truncation_to_quartic_conditions
    if np.any(np.isfinite(radius_conditions)):
        radius_values = result.values["radius"]
        categorical_values = np.arange(len(radius_values), dtype=float)
        figure, ax = plt.subplots(
            figsize=(6.5, 4.5),
            constrained_layout=True,
        )
        ax.plot(
            categorical_values,
            radius_conditions,
            color=PURPLE,
            marker="o",
            linestyle="-",
        )
        ax.axhline(1.0, color="#222222", linestyle=":", linewidth=0.9)
        ax.set_yscale("log")
        ax.set_xticks(
            categorical_values,
            [
                r"$\infty$" if np.isposinf(value) else f"{value:g}"
                for value in radius_values
            ],
        )
        ax.set_xlabel(r"Target radius $R$ ($\infty$: genuine quartic)")
        ax.set_ylabel(r"$\kappa_{\mathrm{rel}}(\Sigma_R,\Sigma_\infty)$")
        ax.set_title(
            "Reference convergence to quartic covariance\n"
            + _parameter_sweep_subtitle("radius", result, args)
        )
        ax.grid(
            which="major",
            color="#D8D8D8",
            linewidth=0.55,
            alpha=0.8,
        )
        figures["truncation_to_quartic_covariance"] = figure
    return figures


def print_parameter_sweep_summary(result: ParameterSweepResult) -> None:
    """Print target difficulty and all four sweep metrics."""

    parameter_labels = {
        "quartic": "lambda",
        "mass": "m",
        "radius": "R",
    }
    print(
        "\nControlled four-method parameter sweeps "
        f"(d={result.sweep_side}, K={result.stages}, "
        f"{result.unique_target_count} unique targets)"
    )
    for sweep_name, values in result.values.items():
        print(f"\nSweep over {parameter_labels[sweep_name]}")
        print(
            "value".rjust(8)
            + "bound".rjust(11)
            + "ref kappa".rjust(12)
            + "outside %".rjust(12)
            + ">Rdes %".rjust(10)
            + "h_ref".rjust(11)
            + "N_ref".rjust(8)
            + "cool".rjust(11)
            + "emp".rjust(11)
            + "TI avg".rjust(11)
            + "raw".rjust(11)
        )
        for value_index, value in enumerate(values):
            medians = {
                method: float(
                    np.median(
                        result.relative_conditions[sweep_name][method][value_index]
                    )
                )
                for method in FOUR_METHODS
            }
            condition_bound = result.hessian_condition_bounds[sweep_name][value_index]
            reference_condition = result.reference_conditions[sweep_name][value_index]
            continuation_percent = 100.0 * (
                result.continuation_fractions[sweep_name][value_index]
            )
            design_exceedance_percent = 100.0 * (
                result.design_exceedance_fractions[sweep_name][value_index]
            )
            reference_step_size = result.reference_step_sizes[sweep_name][value_index]
            reference_step_count = result.reference_steps[sweep_name][value_index]
            continuation_text = (
                "n/a".rjust(12)
                if not np.isfinite(continuation_percent)
                else f"{continuation_percent:>12.3g}"
            )
            print(
                f"{value:>8g}"
                f"{condition_bound:>11.4g}"
                f"{reference_condition:>12.4g}"
                + continuation_text
                + f"{design_exceedance_percent:>10.3g}"
                f"{reference_step_size:>11.4g}"
                f"{reference_step_count:>8d}"
                f"{medians[COOLING_COMPARISON]:>11.4g}"
                f"{medians[EMPIRICAL_COMPARISON]:>11.4g}"
                f"{medians[TRANSLATION_AVERAGED_ULMC]:>11.4g}"
                f"{medians[RAW_ULMC]:>11.4g}"
            )
        if sweep_name == "radius" and np.any(
            np.isfinite(result.truncation_to_quartic_conditions)
        ):
            convergence_text = ", ".join(
                ("inf" if np.isposinf(value) else f"{value:g}") + f": {condition:.4g}"
                for value, condition in zip(
                    values,
                    result.truncation_to_quartic_conditions,
                    strict=True,
                )
            )
            print("  reference kappa_rel(Sigma_R, Sigma_inf): " + convergence_text)
    print(
        "\nParameter-sweep wall time (including compilation): "
        f"{result.elapsed_seconds:.2f}s"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the focused parameter-sweep command-line interface."""

    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compare four covariance preconditioners in controlled lambda, "
            "mass, and truncation-radius sweeps for lattice phi4 targets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Lattice, sweep axes, and values held fixed in each one-parameter sweep.
    parser.add_argument(
        "--side",
        "--parameter-sweep-side",
        dest="side",
        type=int,
        default=10,
        help="Periodic square-lattice side length d.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Nearest-neighbor Laplacian coupling.",
    )
    parser.add_argument(
        "--lambda-sweep-values",
        "--quartic-sweep-values",
        dest="quartic_sweep_values",
        type=_parse_positive_values,
        default=[0.1, 0.5, 2.0],
        help="Quartic couplings in the lambda sweep.",
    )
    parser.add_argument(
        "--mass-sweep-values",
        type=_parse_positive_values,
        default=[0.01, 0.05, 0.25],
        help="Masses in the mass sweep.",
    )
    parser.add_argument(
        "--radius-sweep-values",
        type=_parse_target_radii,
        default=[0.5, 2.0, 4.0, np.inf],
        help=(
            "Target radii in the radius sweep; use 'inf' for the genuine "
            "quartic target."
        ),
    )
    parser.add_argument(
        "--quartic",
        "--parameter-sweep-quartic",
        dest="parameter_sweep_quartic",
        type=float,
        default=0.5,
        help="Fixed quartic coupling at the sweep anchor.",
    )
    parser.add_argument(
        "--mass",
        "--parameter-sweep-mass",
        dest="parameter_sweep_mass",
        type=float,
        default=0.05,
        help="Fixed mass at the sweep anchor.",
    )
    parser.add_argument(
        "--radius",
        "--parameter-sweep-radius",
        dest="parameter_sweep_radius",
        type=float,
        default=2.0,
        help="Fixed truncation radius at the sweep anchor.",
    )
    parser.add_argument(
        "--cooling-design-radius",
        type=float,
        default=4.0,
        help=(
            "Common finite operational curvature radius for every target "
            "in the radius sweep."
        ),
    )

    # Equal method budgets and the independently tuned reference sampler.
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Independent repeats for each learned preconditioner.",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=512,
        help="Independent chains n used by each method at each stage.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="ULMC transitions N per stage.",
    )
    parser.add_argument(
        "--stages",
        "--parameter-sweep-stages",
        dest="stages",
        type=int,
        default=12,
        help="Equal stage count K used by all four methods.",
    )
    parser.add_argument(
        "--reference-chains",
        "--parameter-sweep-reference-chains",
        dest="reference_chains",
        type=int,
        default=1024,
        help="Independent endpoint count for each reference estimate.",
    )
    parser.add_argument(
        "--reference-time",
        "--parameter-sweep-reference-time",
        dest="reference_time",
        type=float,
        default=7.68,
        help="Physical integration time retained by each reference run.",
    )
    parser.add_argument(
        "--reference-margin",
        "--parameter-sweep-reference-margin",
        dest="reference_margin",
        type=float,
        default=0.15,
        help="Maximum h_ref*sqrt(L_ref) for each reference run.",
    )
    parser.add_argument(
        "--reference-step-size",
        type=float,
        default=None,
        help=(
            "Maximum reference ULMC step size; by default reuse "
            "--step-size before applying the stability margin."
        ),
    )

    # Integration settings, numerical metrics, and output controls.
    parser.add_argument(
        "--cooling-gamma",
        type=float,
        default=0.35,
        help="Geometric Gaussian-cooling factor in (0,1).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.25,
        help="Theoretical tolerance retained as fixed-budget metadata.",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="ULMC friction coefficient.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.03,
        help="ULMC integration step size for compared methods.",
    )
    parser.add_argument(
        "--covariance-ridge",
        type=float,
        default=0.0,
        help="Optional ridge used in adaptive covariance updates.",
    )
    parser.add_argument(
        "--metric-floor",
        type=float,
        default=1e-10,
        help="Positive relative floor used in spectral condition metrics.",
    )
    parser.add_argument(
        "--dense-max-side",
        type=int,
        default=20,
        help="Largest side allowed for the raw full-covariance baseline.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Floating-point precision.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=271828,
        help="Root random seed.",
    )
    parser.add_argument(
        "--output-prefix",
        "--output",
        dest="output",
        type=Path,
        default=project_dir / "figures" / "phi4_parameter_sweeps",
        help=(
            "Base output prefix; each plot name is appended and saved as "
            "a separate PDF."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use a deterministic reduced-size smoke-test preset; explicit "
            "numerical options override its values."
        ),
    )
    return parser


def apply_quick_configuration(
    args: argparse.Namespace,
    explicit_destinations: set[str] | None = None,
) -> None:
    """Reduce the sweep and sampling budgets, preserving explicit overrides."""

    if not args.quick:
        return
    explicitly_set = set() if explicit_destinations is None else explicit_destinations
    quick_values = {
        "side": 4,
        "repeats": 1,
        "chains": 64,
        "steps": 20,
        "stages": 4,
        "reference_chains": 96,
        "reference_time": 1.44,
        "quartic_sweep_values": [0.5, 1.0],
        "mass_sweep_values": [0.05, 0.25],
        "radius_sweep_values": [1.0, 2.0, np.inf],
        "cooling_design_radius": 2.0,
        "dense_max_side": 8,
        "dtype": "float32",
    }
    for destination, value in quick_values.items():
        if destination not in explicitly_set:
            setattr(args, destination, value)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate the focused sweep configuration before compiling JAX work."""

    counts = {
        "side": args.side,
        "repeats": args.repeats,
        "chains": args.chains,
        "steps": args.steps,
        "stages": args.stages,
        "reference-chains": args.reference_chains,
    }
    invalid_counts = [name for name, value in counts.items() if value <= 0]
    if invalid_counts:
        raise ValueError(
            "These counts must be positive: " + ", ".join(invalid_counts) + "."
        )
    if args.side < 2:
        raise ValueError("--side must be at least two.")
    if args.chains <= args.side**2:
        raise ValueError(
            "The four-method parameter sweeps require --chains > --side^2; "
            f"got n={args.chains} and D={args.side**2}."
        )
    if args.side > args.dense_max_side:
        raise ValueError(
            "The four-method parameter sweeps require --side <= "
            "--dense-max-side so the raw full-covariance method is feasible."
        )
    if args.reference_chains < 4:
        raise ValueError("--reference-chains must be at least four.")
    if not np.isfinite(args.cooling_gamma) or not (0.0 < args.cooling_gamma < 1.0):
        raise ValueError("--cooling-gamma must lie in (0,1).")
    positive_scalars = {
        "quartic": args.parameter_sweep_quartic,
        "mass": args.parameter_sweep_mass,
        "radius": args.parameter_sweep_radius,
        "cooling-design-radius": args.cooling_design_radius,
        "reference-time": args.reference_time,
        "reference-margin": args.reference_margin,
        "delta": args.delta,
        "friction": args.friction,
        "step-size": args.step_size,
        "metric-floor": args.metric_floor,
    }
    invalid_positive = [
        name
        for name, value in positive_scalars.items()
        if not np.isfinite(value) or value <= 0.0
    ]
    if invalid_positive:
        raise ValueError(
            "These values must be finite and positive: "
            + ", ".join(invalid_positive)
            + "."
        )
    if args.reference_step_size is not None and (
        not np.isfinite(args.reference_step_size) or args.reference_step_size <= 0.0
    ):
        raise ValueError("--reference-step-size must be finite and positive.")
    nonnegative_scalars = {
        "beta": args.beta,
        "covariance-ridge": args.covariance_ridge,
    }
    invalid_nonnegative = [
        name
        for name, value in nonnegative_scalars.items()
        if not np.isfinite(value) or value < 0.0
    ]
    if invalid_nonnegative:
        raise ValueError(
            "These values must be finite and nonnegative: "
            + ", ".join(invalid_nonnegative)
            + "."
        )
    if args.dense_max_side < 0:
        raise ValueError("--dense-max-side must be nonnegative.")

    largest_finite_radius = max(
        (value for value in args.radius_sweep_values if np.isfinite(value)),
        default=0.0,
    )
    if args.cooling_design_radius < largest_finite_radius:
        raise ValueError(
            "--cooling-design-radius must be at least the largest finite "
            "--radius-sweep-values entry."
        )

    includes_genuine_quartic = any(
        np.isposinf(value) for value in args.radius_sweep_values
    )
    if includes_genuine_quartic:
        warnings.warn(
            "R=inf selects the genuine quartic target. Its Hessian is "
            "unbounded, so --cooling-design-radius supplies only an "
            "operational stage-zero/step-size scale; finite-global-L "
            "Gaussian-cooling guarantees do not apply.",
            RuntimeWarning,
            stacklevel=2,
        )

    configurations = (
        [
            (
                value,
                args.parameter_sweep_mass,
                args.parameter_sweep_radius,
                args.parameter_sweep_radius,
            )
            for value in args.quartic_sweep_values
        ]
        + [
            (
                args.parameter_sweep_quartic,
                value,
                args.parameter_sweep_radius,
                args.parameter_sweep_radius,
            )
            for value in args.mass_sweep_values
        ]
        + [
            (
                args.parameter_sweep_quartic,
                args.parameter_sweep_mass,
                value,
                args.cooling_design_radius,
            )
            for value in args.radius_sweep_values
        ]
    )
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    max_design_smoothness = max(
        make_lattice_model(
            args.side,
            args.beta,
            quartic,
            mass,
            radius,
            dtype,
            design_radius=design_radius,
        ).design_smoothness
        for quartic, mass, radius, design_radius in configurations
    )
    stiffness_margin = args.step_size * np.sqrt(max_design_smoothness)
    if stiffness_margin > 0.5:
        warnings.warn(
            "The parameter grid has "
            f"h*sqrt(L)={stiffness_margin:.3g}; use a smaller --step-size "
            "or a less extreme grid.",
            RuntimeWarning,
            stacklevel=2,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the focused parameter-sweep experiment."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    explicit_destinations = _explicit_cli_destinations(parser, arguments)
    apply_quick_configuration(args, explicit_destinations)
    validate_arguments(args)

    result = run_parameter_sweeps(args)
    figures = make_parameter_sweep_figures(result, args)
    output_paths = save_publication_figures(figures, args.output)
    for figure in figures.values():
        plt.close(figure)

    print_parameter_sweep_summary(result)
    for plot_name, path in output_paths.items():
        print(f"Saved {plot_name.replace('_', ' ')} PDF: {path}")


if __name__ == "__main__":
    main()
