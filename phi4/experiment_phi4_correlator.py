"""Center-site two-point correlators for periodic lattice phi4 targets.

The experiment first learns covariance preconditioners with four equal-budget
methods.  It then runs a matched serial ULMC chain for each feasible learned
preconditioner and estimates the connected two-point correlator relative to
the center of the periodic grid.  Exact toroidal-distance shells turn the
center covariance row into a radial correlation-decay curve.

All method curves target the same physical correlator.  Their differences
therefore diagnose finite burn-in, Monte Carlo error, and ULMC discretization
bias rather than different equilibrium quantities.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import lax, random

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from emcee.autocorr import AutocorrError, integrated_time
except ImportError:  # pragma: no cover - validated before the experiment.
    AutocorrError = RuntimeError
    integrated_time = None

from gaussian_cooling_algs import (
    apply_translation_invariant_spectrum,
    sample_power_spectrum,
    translation_invariant_covariance,
    ulmc_coefficients,
)
from .lattice_phi4 import (
    COOLING_COMPARISON,
    EMPIRICAL_COMPARISON,
    FOUR_METHODS,
    RAW_ULMC,
    TRANSLATION_AVERAGED_ULMC,
    Array,
    LatticeModel,
    _fourier_cooling_call,
    _fourier_empirical_baseline_call,
    _normalized_preconditioner_operators,
    _periodic_distance_shells,
    _raw_ulmc_covariance_call,
    _representative_repeat_index,
    _run_timed_repeats,
    _translation_averaged_ulmc_call,
    dense_relative_condition,
    make_lattice_model,
    reference_samples,
    spectral_relative_condition,
    validate_lattice_model,
)
from .phi4_cli import (
    _explicit_cli_destinations,
    add_execution_arguments,
    add_reference_arguments,
    add_sampler_arguments,
    add_target_arguments,
    apply_gpu_configuration,
    apply_quick_configuration,
    validate_common_arguments,
    validate_trajectory_arguments,
)
from .phi4_plotting import (
    METHOD_STYLES,
    PLOT_LABELS,
    _configure_plot_style,
    _phi4_parameter_subtitle,
    save_publication_figures,
)


@dataclass
class TwoPointCorrelatorResult:
    """Center-site correlators from matched preconditioned ULMC chains."""

    side: int
    center: tuple[int, int]
    correlation_fields: dict[str, np.ndarray]
    shell_distances: np.ndarray
    radial_correlations: dict[str, np.ndarray]
    radial_standard_errors: dict[str, np.ndarray]
    radial_reliable: dict[str, np.ndarray]
    reference_correlation_field: np.ndarray
    reference_radial_correlation: np.ndarray
    skipped_methods: dict[str, str]


@dataclass
class _CenterCorrelatorEstimate:
    """Streaming center-site correlator and shellwise uncertainty output."""

    correlation_field: np.ndarray
    shell_distances: np.ndarray
    radial_correlation: np.ndarray
    radial_standard_error: np.ndarray
    radial_reliable: np.ndarray


@dataclass
class _LearnedPreconditioners:
    """Representative preconditioners and their selection diagnostics."""

    preconditioners: dict[str, tuple[np.ndarray, bool]]
    reference_spectrum: np.ndarray
    relative_conditions: dict[str, np.ndarray]
    skipped_methods: dict[str, str]


def _estimate_integrated_times(
    observables: np.ndarray,
    iat_tolerance: int,
) -> np.ndarray:
    """Estimate one IAT per observable, retaining short-chain estimates."""

    if integrated_time is None:
        raise RuntimeError(
            "Two-point-correlator uncertainty estimates require `emcee`. "
            "Install it with `python -m pip install emcee`."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            estimate = integrated_time(
                observables,
                tol=iat_tolerance,
                quiet=False,
                has_walkers=False,
            )
            times = np.atleast_1d(estimate)
        except AutocorrError as exc:
            times = np.atleast_1d(exc.tau)
    times = np.asarray(times, dtype=np.float64).reshape((-1,))
    if times.size != observables.shape[1]:
        raise RuntimeError(
            "emcee returned an unexpected number of autocorrelation times: "
            f"expected {observables.shape[1]}, received {times.size}."
        )
    return times


def _estimate_center_correlator(
    key: Array,
    model: LatticeModel,
    preconditioner: np.ndarray,
    args: argparse.Namespace,
    *,
    translation_invariant: bool,
) -> _CenterCorrelatorEstimate:
    """Estimate a center covariance row and shellwise MC uncertainty."""

    center = (model.side // 2, model.side // 2)
    center_index = center[0] * model.side + center[1]
    shell_distances, shell_indices = _periodic_distance_shells(
        model.side,
        center,
    )
    to_physical, transform_gradient = _normalized_preconditioner_operators(
        model,
        preconditioner,
        translation_invariant=translation_invariant,
        relative_floor=args.metric_floor,
    )

    key_position, key_momentum, key_steps = random.split(key, 3)
    position = random.normal(
        key_position,
        (model.dimension,),
        dtype=model.dtype,
    )
    momentum = random.normal(
        key_momentum,
        (model.dimension,),
        dtype=model.dtype,
    )
    physical_field = to_physical(position)
    state = (position, momentum, physical_field, key_steps)
    (
        momentum_decay,
        position_coefficient,
        gradient_coefficient,
        noise_cholesky,
    ) = ulmc_coefficients(
        args.friction,
        args.step_size,
        model.dtype,
    )

    def advance(carry):
        (
            current_position,
            current_momentum,
            current_physical,
            current_key,
        ) = carry
        next_key, noise_key = random.split(current_key)
        standard_noise = random.normal(
            noise_key,
            (model.dimension, 2),
            dtype=model.dtype,
        )
        correlated_noise = standard_noise @ noise_cholesky.T
        transformed_gradient = transform_gradient(model.gradient(current_physical))
        next_position = (
            current_position
            + position_coefficient * current_momentum
            - gradient_coefficient * transformed_gradient
            + correlated_noise[:, 0]
        )
        next_momentum = (
            momentum_decay * current_momentum
            - position_coefficient * transformed_gradient
            + correlated_noise[:, 1]
        )
        next_physical = to_physical(next_position)
        return next_position, next_momentum, next_physical, next_key

    def burnin_transition(carry, _):
        return advance(carry), None

    def sampling_transition(carry, _):
        next_state = advance(carry)
        return next_state, next_state[2]

    runners: dict[tuple[int, bool], Callable] = {}

    def run_chunk(
        current_state: tuple[Array, Array, Array, Array],
        length: int,
        *,
        emit_fields: bool,
    ) -> tuple[tuple[Array, Array, Array, Array], Array | None]:
        runner_key = (length, emit_fields)
        if runner_key not in runners:
            transition = sampling_transition if emit_fields else burnin_transition

            def scan_chunk(scan_state):
                return lax.scan(
                    transition,
                    scan_state,
                    xs=None,
                    length=length,
                )

            runners[runner_key] = jax.jit(scan_chunk)
        return runners[runner_key](current_state)

    remaining_burnin = args.trajectory_burnin
    while remaining_burnin:
        length = min(args.trajectory_chunk_size, remaining_burnin)
        state, _ = run_chunk(state, length, emit_fields=False)
        remaining_burnin -= length

    num_samples = args.trajectory_samples
    center_series = np.empty(num_samples, dtype=np.float64)
    shell_series = np.empty(
        (num_samples, len(shell_indices)),
        dtype=np.float64,
    )
    field_sum = np.zeros(model.dimension, dtype=np.float64)
    center_field_sum = np.zeros(model.dimension, dtype=np.float64)
    sample_start = 0
    while sample_start < num_samples:
        length = min(
            args.trajectory_chunk_size,
            num_samples - sample_start,
        )
        state, fields = run_chunk(state, length, emit_fields=True)
        assert fields is not None
        field_block = np.asarray(fields.block_until_ready())
        if np.any(~np.isfinite(field_block)):
            raise FloatingPointError("A center-correlator trajectory became nonfinite.")
        sample_stop = sample_start + length
        center_values = np.asarray(
            field_block[:, center_index],
            dtype=np.float64,
        )
        center_series[sample_start:sample_stop] = center_values
        for shell_index, sites in enumerate(shell_indices):
            shell_series[sample_start:sample_stop, shell_index] = np.mean(
                field_block[:, sites],
                axis=1,
                dtype=np.float64,
            )
        field_sum += np.sum(field_block, axis=0, dtype=np.float64)
        center_field_sum += np.sum(
            field_block * center_values[:, None],
            axis=0,
            dtype=np.float64,
        )
        sample_start = sample_stop

    center_mean = float(np.mean(center_series))
    field_mean = field_sum / num_samples
    correlation_row = center_field_sum / num_samples - center_mean * field_mean
    centered_shell_observables = (center_series - center_mean)[:, None] * (
        shell_series - np.mean(shell_series, axis=0, keepdims=True)
    )
    radial_correlation = np.mean(centered_shell_observables, axis=0)
    field_radial_correlation = np.asarray(
        [np.mean(correlation_row[sites]) for sites in shell_indices]
    )
    if not np.allclose(
        radial_correlation,
        field_radial_correlation,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError("Streaming field and shell correlator estimates disagree.")

    radial_iats = _estimate_integrated_times(
        centered_shell_observables,
        args.iat_tolerance,
    )
    radial_variances = np.var(
        centered_shell_observables,
        axis=0,
        ddof=1,
    )
    valid_iats = (
        np.isfinite(radial_iats)
        & (radial_iats > 0.0)
        & np.isfinite(radial_variances)
        & (radial_variances >= 0.0)
    )
    radial_standard_error = np.full(len(shell_indices), np.nan)
    radial_standard_error[valid_iats] = np.sqrt(
        radial_variances[valid_iats] * radial_iats[valid_iats] / num_samples
    )
    radial_reliable = valid_iats & (num_samples >= args.iat_tolerance * radial_iats)
    return _CenterCorrelatorEstimate(
        correlation_field=correlation_row.reshape(model.lattice_shape),
        shell_distances=shell_distances,
        radial_correlation=radial_correlation,
        radial_standard_error=radial_standard_error,
        radial_reliable=radial_reliable,
    )


def run_two_point_correlator_experiment(
    key: Array,
    model: LatticeModel,
    preconditioners: dict[str, tuple[np.ndarray, bool]],
    reference_spectrum: np.ndarray,
    skipped_methods: dict[str, str],
    args: argparse.Namespace,
) -> TwoPointCorrelatorResult:
    """Compare center-site correlators under matched sampling budgets."""

    center = (model.side // 2, model.side // 2)
    shell_distances, shell_indices = _periodic_distance_shells(
        model.side,
        center,
    )
    correlation_fields: dict[str, np.ndarray] = {}
    radial_correlations: dict[str, np.ndarray] = {}
    radial_standard_errors: dict[str, np.ndarray] = {}
    radial_reliable: dict[str, np.ndarray] = {}
    for method_index, method in enumerate(FOUR_METHODS):
        if method not in preconditioners:
            continue
        preconditioner, translation_invariant = preconditioners[method]
        print(
            "  Center-site correlator chain: " + PLOT_LABELS[method],
            flush=True,
        )
        estimate = _estimate_center_correlator(
            random.fold_in(key, method_index),
            model,
            preconditioner,
            args,
            translation_invariant=translation_invariant,
        )
        if not np.array_equal(estimate.shell_distances, shell_distances):
            raise AssertionError("Correlator shell definitions disagree.")
        correlation_fields[method] = estimate.correlation_field
        radial_correlations[method] = estimate.radial_correlation
        radial_standard_errors[method] = estimate.radial_standard_error
        radial_reliable[method] = estimate.radial_reliable

    center_basis = (
        jnp.zeros(model.dimension, dtype=model.dtype)
        .at[center[0] * model.side + center[1]]
        .set(1.0)
    )
    reference_field = np.asarray(
        apply_translation_invariant_spectrum(
            center_basis,
            jnp.asarray(reference_spectrum, dtype=model.dtype),
            model.lattice_shape,
        )
    ).reshape(model.lattice_shape)
    flattened_reference = reference_field.reshape((-1,))
    reference_radial = np.asarray(
        [np.mean(flattened_reference[sites]) for sites in shell_indices]
    )
    return TwoPointCorrelatorResult(
        side=model.side,
        center=center,
        correlation_fields=correlation_fields,
        shell_distances=shell_distances,
        radial_correlations=radial_correlations,
        radial_standard_errors=radial_standard_errors,
        radial_reliable=radial_reliable,
        reference_correlation_field=reference_field,
        reference_radial_correlation=reference_radial,
        skipped_methods=skipped_methods,
    )


def _learn_preconditioners(
    model: LatticeModel,
    args: argparse.Namespace,
    root_key: Array,
) -> _LearnedPreconditioners:
    """Learn and select one representative preconditioner per method."""

    # A focused one-side invocation matches the legacy `--sides d` key path.
    base_key = random.fold_in(root_key, 0)
    key_warm_fourier = random.fold_in(base_key, 10_001)
    key_reference = random.fold_in(base_key, 10_003)
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
        f"d={model.side} (D={model.dimension}): learning "
        f"{args.repeats} preconditioner repeat(s)",
        flush=True,
    )
    print("  Translation-invariant Gaussian cooling", flush=True)

    def cooling_call(key: Array) -> Array:
        return _fourier_cooling_call(key, model, args)

    cooling_spectra, _ = _run_timed_repeats(
        cooling_call,
        fourier_keys,
        key_warm_fourier,
    )

    print(
        "  Translation-invariant empirical preconditioning (no cooling)",
        flush=True,
    )

    def empirical_call(key: Array) -> Array:
        return _fourier_empirical_baseline_call(key, model, args)

    empirical_spectra, _ = _run_timed_repeats(
        empirical_call,
        empirical_keys,
        key_warm_empirical,
    )

    print(
        "  Translation-averaged covariance from unpreconditioned ULMC",
        flush=True,
    )

    def translation_averaged_call(key: Array) -> Array:
        return _translation_averaged_ulmc_call(key, model, args)

    translation_averaged_spectra, _ = _run_timed_repeats(
        translation_averaged_call,
        plain_keys,
        key_warm_plain,
    )

    raw_skip_reason: str | None = None
    if args.chains <= model.dimension:
        raw_skip_reason = f"n={args.chains} <= D={model.dimension} (rank deficient)"
    elif model.side > args.dense_max_side:
        raw_skip_reason = f"side>{args.dense_max_side} full-covariance cutoff"

    raw_covariances: list[np.ndarray] = []
    if raw_skip_reason is None:
        print(
            "  Full empirical covariance from the same unpreconditioned "
            "ULMC endpoints",
            flush=True,
        )

        def raw_call(key: Array) -> Array:
            return _raw_ulmc_covariance_call(key, model, args)

        raw_covariances, _ = _run_timed_repeats(
            raw_call,
            plain_keys,
            key_warm_plain,
        )
    else:
        print(
            f"  Full empirical ULMC covariance omitted: {raw_skip_reason}",
            flush=True,
        )

    reference = reference_samples(key_reference, model, args)
    reference.block_until_ready()
    reference_spectrum = np.asarray(
        sample_power_spectrum(reference, model.lattice_shape)
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
    }

    if raw_skip_reason is None:
        reference_covariance = np.asarray(
            translation_invariant_covariance(
                jnp.asarray(reference_spectrum, dtype=model.dtype),
                model.lattice_shape,
            )
        )
        relative_conditions[RAW_ULMC] = np.asarray(
            [
                dense_relative_condition(
                    covariance,
                    reference_covariance,
                    0.0,
                )
                for covariance in raw_covariances
            ]
        )

    preconditioners = {
        COOLING_COMPARISON: (
            np.asarray(
                cooling_spectra[
                    _representative_repeat_index(
                        relative_conditions[COOLING_COMPARISON]
                    )
                ]
            ),
            True,
        ),
        EMPIRICAL_COMPARISON: (
            np.asarray(
                empirical_spectra[
                    _representative_repeat_index(
                        relative_conditions[EMPIRICAL_COMPARISON]
                    )
                ]
            ),
            True,
        ),
        TRANSLATION_AVERAGED_ULMC: (
            np.asarray(
                translation_averaged_spectra[
                    _representative_repeat_index(
                        relative_conditions[TRANSLATION_AVERAGED_ULMC]
                    )
                ]
            ),
            True,
        ),
    }
    skipped_methods: dict[str, str] = {}
    if raw_skip_reason is None:
        preconditioners[RAW_ULMC] = (
            np.asarray(
                raw_covariances[
                    _representative_repeat_index(relative_conditions[RAW_ULMC])
                ]
            ),
            False,
        )
    else:
        skipped_methods[RAW_ULMC] = raw_skip_reason

    return _LearnedPreconditioners(
        preconditioners=preconditioners,
        reference_spectrum=reference_spectrum,
        relative_conditions=relative_conditions,
        skipped_methods=skipped_methods,
    )


def run_experiment(
    args: argparse.Namespace,
) -> tuple[TwoPointCorrelatorResult, _LearnedPreconditioners, float]:
    """Learn preconditioners and run the matched correlator trajectories."""

    started = time.perf_counter()
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    model = make_lattice_model(
        args.side,
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
    root_key = random.PRNGKey(args.seed)
    learned = _learn_preconditioners(model, args, root_key)
    print(
        "Center-site two-point-correlator comparison: "
        f"d={args.side}, c=({args.side // 2},{args.side // 2}), "
        f"T={args.trajectory_samples:,}, "
        f"burn-in={args.trajectory_burnin:,}",
        flush=True,
    )
    result = run_two_point_correlator_experiment(
        random.fold_in(root_key, 20_000),
        model,
        learned.preconditioners,
        learned.reference_spectrum,
        learned.skipped_methods,
        args,
    )
    return result, learned, time.perf_counter() - started


def make_correlator_figures(
    result: TwoPointCorrelatorResult,
    args: argparse.Namespace,
) -> dict[str, plt.Figure]:
    """Create the standalone center-correlator decay figure."""

    _configure_plot_style()
    figure, ax = plt.subplots(
        figsize=(6.6, 4.9),
        constrained_layout=True,
    )
    distances = result.shell_distances
    for method in FOUR_METHODS:
        if method not in result.radial_correlations:
            continue
        color, marker, linestyle = METHOD_STYLES[method]
        correlations = result.radial_correlations[method]
        standard_errors = result.radial_standard_errors[method]
        reliable = result.radial_reliable[method]
        band_width = np.where(
            reliable,
            1.96 * standard_errors,
            np.nan,
        )
        ax.fill_between(
            distances,
            correlations - band_width,
            correlations + band_width,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            distances,
            correlations,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=3.8,
            markevery=max(1, len(distances) // 10),
            label=PLOT_LABELS[method],
        )
    ax.plot(
        distances,
        result.reference_radial_correlation,
        color="#222222",
        linestyle=(0, (4, 2)),
        linewidth=1.35,
        label="Independent reference",
    )
    ax.axhline(
        0.0,
        color="#555555",
        linestyle=":",
        linewidth=0.8,
        zorder=0,
    )
    ax.set_xlabel(r"Periodic distance $r$ from the center site")
    ax.set_ylabel(r"Connected two-point correlator $C(r)$")
    center_row, center_column = result.center
    ax.set_title(
        "Center-site two-point correlation after preconditioned ULMC\n"
        rf"$d={result.side},\ c=({center_row},{center_column}),\ "
        rf"T={args.trajectory_samples},\ "
        rf"N_{{\rm burn}}={args.trajectory_burnin}$"
        + "\n"
        + _phi4_parameter_subtitle(args)
    )
    ax.grid(
        which="major",
        color="#D8D8D8",
        linewidth=0.55,
        alpha=0.8,
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncols=2,
        fontsize=7.3,
    )
    return {f"two_point_correlator_decay_d{result.side}": figure}


def print_summary(
    result: TwoPointCorrelatorResult,
    learned: _LearnedPreconditioners,
    elapsed_seconds: float,
) -> None:
    """Report representative quality and correlator agreement."""

    print(
        "\nCenter-site two-point correlators, "
        f"side={result.side}, center={result.center}:"
    )
    for method in FOUR_METHODS:
        if method not in result.radial_correlations:
            continue
        difference = (
            result.radial_correlations[method] - result.reference_radial_correlation
        )
        reliable_count = int(np.count_nonzero(result.radial_reliable[method]))
        selected_index = _representative_repeat_index(
            learned.relative_conditions[method]
        )
        selected_condition = learned.relative_conditions[method][selected_index]
        print(
            f"  {PLOT_LABELS[method]}: "
            f"selected kappa_rel={selected_condition:.4g}, "
            "shell RMSE vs reference="
            f"{np.sqrt(np.mean(difference**2)):.4g}, "
            f"reliable MCSE bands={reliable_count}/"
            f"{len(result.shell_distances)}"
        )
    for method, reason in result.skipped_methods.items():
        print(f"  {PLOT_LABELS[method]} omitted: {reason}")
    print(
        "\nCorrelator-experiment wall time (including compilation): "
        f"{elapsed_seconds:.2f}s"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the focused center-correlator command-line interface."""

    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the lattice phi4 center-site two-point correlator "
            "using matched ULMC chains under learned preconditioners."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--side",
        "--correlator-side",
        dest="side",
        type=int,
        default=10,
        help="Periodic square-lattice side length d.",
    )
    add_target_arguments(parser, include_sides=False)
    add_sampler_arguments(parser)
    add_reference_arguments(parser)
    parser.add_argument(
        "--trajectory-burnin",
        type=int,
        default=1024,
        help="Burn-in transitions for each matched post-learning chain.",
    )
    parser.add_argument(
        "--trajectory-samples",
        type=int,
        default=8192,
        help="Retained samples in each matched post-learning chain.",
    )
    parser.add_argument(
        "--iat-tolerance",
        type=int,
        default=50,
        help="Minimum chain-length-to-IAT ratio required by emcee.",
    )
    parser.add_argument(
        "--trajectory-chunk-size",
        type=int,
        default=256,
        help="Consecutive trajectory transitions per compiled JAX scan.",
    )
    add_execution_arguments(
        parser,
        default_output=(project_dir / "figures" / "phi4_correlator"),
    )
    return parser


def apply_presets(
    args: argparse.Namespace,
    explicit_destinations: set[str] | None = None,
) -> None:
    """Apply focused quick/GPU presets while preserving explicit options."""

    explicitly_set = set() if explicit_destinations is None else explicit_destinations
    apply_quick_configuration(args, explicitly_set)
    apply_gpu_configuration(args, explicitly_set)
    if args.quick and "side" not in explicitly_set:
        # The legacy quick suite selected d=6: d=8 has D=n and therefore no
        # full-rank raw empirical covariance.
        args.side = 6
    if args.gpu and "side" not in explicitly_set:
        # This matches the legacy GPU opt-in's smallest requested side.
        args.side = 64


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate model, trajectory, and backend settings."""

    validate_common_arguments(args)
    validate_trajectory_arguments(
        args,
        emcee_available=(integrated_time is not None),
    )
    if args.side < 2:
        raise ValueError("--side must be at least two.")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the focused center-site correlator experiment."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.quick and args.gpu:
        parser.error("--quick and --gpu are mutually exclusive presets.")
    explicit_destinations = _explicit_cli_destinations(parser, arguments)
    apply_presets(args, explicit_destinations)
    validate_arguments(args)

    result, learned, elapsed_seconds = run_experiment(args)
    figures = make_correlator_figures(result, args)
    output_paths = save_publication_figures(figures, args.output)
    for figure in figures.values():
        plt.close(figure)

    print_summary(result, learned, elapsed_seconds)
    for plot_name, path in output_paths.items():
        print(f"Saved {plot_name.replace('_', ' ')} PDF: {path}")


if __name__ == "__main__":
    main()
