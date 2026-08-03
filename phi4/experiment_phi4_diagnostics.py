"""Post-cooling mixing diagnostics for periodic lattice phi4 targets.

The experiment first learns a translation-invariant Gaussian-cooling
preconditioner, choosing the repeat nearest the median independently measured
relative condition number.  A serial preconditioned ULMC trajectory then
estimates the integrated autocorrelation time (IAT) of every origin-anchored
two-point observable and checks the exact ``phi -> -phi`` symmetry through
IAT-adjusted final-half site-mean z scores.

The retained trajectory is stored temporarily in a site-major memory map, so
the analysis does not require holding the complete trajectory in RAM.  Every
plot is saved as a separate vector PDF.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
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
except ImportError:  # pragma: no cover - exercised only without dependency.
    AutocorrError = RuntimeError
    integrated_time = None

from gaussian_cooling_algs import (
    apply_translation_invariant_spectrum,
    sample_power_spectrum,
    ulmc_coefficients,
)
from .lattice_phi4 import (
    LatticeModel,
    _fourier_cooling_call,
    _representative_repeat_index,
    _run_timed_repeats,
    _safe_spectrum,
    make_lattice_model,
    periodic_distances,
    reference_samples,
    spectral_relative_condition,
    validate_lattice_model,
)
from .phi4_cli import _available_gpu_devices, _explicit_cli_destinations
from .phi4_plotting import (
    _configure_plot_style,
    _phi4_parameter_subtitle,
    save_publication_figures,
)


Array = jax.Array


@dataclass
class SiteDiagnostics:
    """Bounded-memory diagnostics derived from one retained trajectory."""

    distances: np.ndarray
    covariance_row: np.ndarray
    correlator_autocorrelation_times: np.ndarray
    correlator_autocorrelation_reliable: np.ndarray
    final_half_site_means: np.ndarray
    site_field_autocorrelation_times: np.ndarray
    site_field_autocorrelation_reliable: np.ndarray
    site_mean_standard_errors: np.ndarray
    site_mean_effective_sample_sizes: np.ndarray
    site_mean_z_scores: np.ndarray
    final_half_sample_count: int


@dataclass
class DiagnosticsResult:
    """Learned-preconditioner metadata and post-learning diagnostics."""

    side: int
    selected_relative_condition: float
    all_relative_conditions: np.ndarray
    diagnostics: SiteDiagnostics
    elapsed_seconds: float


def _write_preconditioned_site_series(
    key: Array,
    model: LatticeModel,
    spectrum: np.ndarray,
    args: argparse.Namespace,
    store: np.memmap,
) -> np.ndarray:
    """Write every post-burn-in lattice field in bounded memory.

    The site-major disk layout makes every site's time series contiguous.
    The PRNG key is part of the scan carry, so changing the chunk size does
    not change the trajectory.
    """

    sqrt_spectrum = jnp.sqrt(
        jnp.asarray(
            _safe_spectrum(spectrum, args.metric_floor),
            dtype=model.dtype,
        )
    )

    def to_physical(y: Array) -> Array:
        return apply_translation_invariant_spectrum(
            y,
            sqrt_spectrum,
            model.lattice_shape,
        )

    preconditioned_smoothness = max(
        model.design_smoothness * float(np.max(spectrum)),
        1e-6,
    )
    key_position, key_momentum, key_steps = random.split(key, 3)
    position = random.normal(
        key_position,
        (model.dimension,),
        dtype=model.dtype,
    ) / jnp.sqrt(jnp.asarray(preconditioned_smoothness, dtype=model.dtype))
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
        gradient = apply_translation_invariant_spectrum(
            model.gradient(current_physical),
            sqrt_spectrum,
            model.lattice_shape,
        )
        next_position = (
            current_position
            + position_coefficient * current_momentum
            - gradient_coefficient * gradient
            + correlated_noise[:, 0]
        )
        next_momentum = (
            momentum_decay * current_momentum
            - position_coefficient * gradient
            + correlated_noise[:, 1]
        )
        next_physical = to_physical(next_position)
        return (
            next_position,
            next_momentum,
            next_physical,
            next_key,
        )

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

    final_half_start = args.trajectory_samples // 2
    final_half_sum = np.zeros(model.dimension, dtype=np.float64)
    sample_start = 0
    while sample_start < args.trajectory_samples:
        length = min(
            args.trajectory_chunk_size,
            args.trajectory_samples - sample_start,
        )
        state, fields = run_chunk(state, length, emit_fields=True)
        assert fields is not None
        field_block = np.asarray(fields.block_until_ready())
        if np.any(~np.isfinite(field_block)):
            raise FloatingPointError(
                "The preconditioned diagnostic trajectory became nonfinite."
            )
        sample_stop = sample_start + length
        store[:, sample_start:sample_stop] = field_block.T
        if sample_stop > final_half_start:
            local_start = max(0, final_half_start - sample_start)
            final_half_sum += np.sum(
                field_block[local_start:],
                axis=0,
                dtype=np.float64,
            )
        sample_start = sample_stop

    store.flush()
    final_half_sample_count = args.trajectory_samples - final_half_start
    return (final_half_sum / final_half_sample_count).reshape(model.lattice_shape)


def _estimate_integrated_times(
    observables: np.ndarray,
    iat_tolerance: int,
) -> np.ndarray:
    """Estimate one IAT per observable, retaining short-chain estimates."""

    if integrated_time is None:
        raise RuntimeError(
            "Autocorrelation diagnostics require `emcee`. Install it with "
            "`python -m pip install emcee`."
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


def _analyze_site_store(
    store: np.memmap,
    side: int,
    num_samples: int,
    iat_tolerance: int,
    iat_batch_size: int,
) -> SiteDiagnostics:
    """Analyze correlator mixing and final-half site-mean uncertainty."""

    if integrated_time is None:
        raise RuntimeError(
            "Autocorrelation diagnostics require `emcee`. Install it with "
            "`python -m pip install emcee`."
        )

    dimension = side * side
    if store.shape != (dimension, num_samples):
        raise ValueError(
            "The site-series store must have shape " "(side**2, trajectory_samples)."
        )
    final_half_start = num_samples // 2
    final_half_sample_count = num_samples - final_half_start
    if final_half_sample_count < 2:
        raise ValueError(
            "At least two final-half samples are required for site-mean "
            "standard errors."
        )

    covariance_row = np.empty(dimension, dtype=float)
    autocorrelation_times = np.empty(dimension, dtype=float)
    final_half_site_means = np.empty(dimension, dtype=float)
    site_field_autocorrelation_times = np.empty(dimension, dtype=float)
    site_mean_variances = np.empty(dimension, dtype=float)
    origin_series = np.array(store[0, :], dtype=np.float64, copy=True)
    centered_origin = origin_series - np.mean(origin_series)

    for start in range(0, dimension, iat_batch_size):
        stop = min(start + iat_batch_size, dimension)
        site_series = np.array(
            store[start:stop, :],
            dtype=np.float64,
            copy=True,
        )
        if np.any(~np.isfinite(site_series)):
            raise FloatingPointError(
                "The stored diagnostic trajectory contains nonfinite values."
            )

        final_half = site_series[:, final_half_start:]
        final_half_site_means[start:stop] = np.mean(
            final_half,
            axis=1,
        )
        site_mean_variances[start:stop] = np.var(
            final_half,
            axis=1,
            ddof=1,
        )
        site_field_autocorrelation_batch = _estimate_integrated_times(
            final_half.T,
            iat_tolerance,
        )
        site_field_autocorrelation_times[start:stop] = site_field_autocorrelation_batch

        centered_sites = site_series - np.mean(site_series, axis=1, keepdims=True)
        correlators = centered_origin[None, :] * centered_sites
        covariance_row[start:stop] = np.mean(correlators, axis=1)
        autocorrelation_times[start:stop] = _estimate_integrated_times(
            correlators.T,
            iat_tolerance,
        )

    correlator_reliable = (
        np.isfinite(autocorrelation_times)
        & (autocorrelation_times > 0.0)
        & (num_samples >= iat_tolerance * autocorrelation_times)
    )
    valid_site_estimates = (
        np.isfinite(site_field_autocorrelation_times)
        & (site_field_autocorrelation_times > 0.0)
        & np.isfinite(site_mean_variances)
        & (site_mean_variances > 0.0)
    )
    site_field_reliable = valid_site_estimates & (
        final_half_sample_count >= iat_tolerance * site_field_autocorrelation_times
    )
    standard_errors = np.full(dimension, np.nan, dtype=float)
    effective_sample_sizes = np.full(dimension, np.nan, dtype=float)
    z_scores = np.full(dimension, np.nan, dtype=float)
    effective_sample_sizes[valid_site_estimates] = (
        final_half_sample_count / site_field_autocorrelation_times[valid_site_estimates]
    )
    standard_errors[valid_site_estimates] = np.sqrt(
        site_mean_variances[valid_site_estimates]
        / effective_sample_sizes[valid_site_estimates]
    )
    valid_standard_errors = (
        valid_site_estimates & np.isfinite(standard_errors) & (standard_errors > 0.0)
    )
    z_scores[valid_standard_errors] = (
        final_half_site_means[valid_standard_errors]
        / standard_errors[valid_standard_errors]
    )

    lattice_shape = (side, side)
    return SiteDiagnostics(
        distances=periodic_distances(side),
        covariance_row=covariance_row,
        correlator_autocorrelation_times=autocorrelation_times,
        correlator_autocorrelation_reliable=correlator_reliable,
        final_half_site_means=final_half_site_means.reshape(lattice_shape),
        site_field_autocorrelation_times=(
            site_field_autocorrelation_times.reshape(lattice_shape)
        ),
        site_field_autocorrelation_reliable=(
            site_field_reliable.reshape(lattice_shape)
        ),
        site_mean_standard_errors=standard_errors.reshape(lattice_shape),
        site_mean_effective_sample_sizes=(
            effective_sample_sizes.reshape(lattice_shape)
        ),
        site_mean_z_scores=z_scores.reshape(lattice_shape),
        final_half_sample_count=final_half_sample_count,
    )


def run_preconditioned_diagnostics(
    key: Array,
    model: LatticeModel,
    spectrum: np.ndarray,
    args: argparse.Namespace,
) -> SiteDiagnostics:
    """Generate and analyze all first-row correlators, cleaning up storage."""

    storage_dtype = np.float64 if model.dtype == jnp.float64 else np.float32
    shape = (model.dimension, args.trajectory_samples)
    temporary_root = Path(tempfile.gettempdir())
    required_bytes = int(np.prod(shape)) * np.dtype(storage_dtype).itemsize
    safety_margin = max(64 * 2**20, required_bytes // 10)
    free_bytes = shutil.disk_usage(temporary_root).free
    if free_bytes < required_bytes + safety_margin:
        raise OSError(
            "Insufficient temporary disk space for full covariance-row "
            f"diagnostics: need at least "
            f"{(required_bytes + safety_margin) / 2**20:.1f} MiB, "
            f"found {free_bytes / 2**20:.1f} MiB in {temporary_root}."
        )
    with tempfile.TemporaryDirectory(
        prefix="gaussian-cooling-phi4-iat-",
        dir=temporary_root,
    ) as temporary_directory:
        store_path = Path(temporary_directory) / "site-series.dat"
        store = np.memmap(
            store_path,
            mode="w+",
            dtype=storage_dtype,
            shape=shape,
        )
        try:
            final_half_site_means = _write_preconditioned_site_series(
                key,
                model,
                spectrum,
                args,
                store,
            )
            diagnostics = _analyze_site_store(
                store,
                model.side,
                args.trajectory_samples,
                args.iat_tolerance,
                args.iat_batch_size,
            )
            storage_epsilon = np.finfo(storage_dtype).eps
            if not np.allclose(
                final_half_site_means,
                diagnostics.final_half_site_means,
                rtol=50.0 * storage_epsilon,
                atol=50.0 * storage_epsilon,
            ):
                raise AssertionError(
                    "Streaming and stored final-half site means disagree."
                )
        finally:
            store.flush()
            del store

    return diagnostics


def run_experiment(args: argparse.Namespace) -> DiagnosticsResult:
    """Learn a representative cooling preconditioner and diagnose its chain."""

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
    # A focused one-side invocation corresponds to side_index=0 in the old
    # suite.  Retain its independent key tags for learning, reference, and
    # the post-learning serial trajectory.
    base_key = random.fold_in(root_key, 0)
    warmup_key = random.fold_in(base_key, 10_001)
    reference_key = random.fold_in(base_key, 10_003)
    run_keys = list(
        random.split(
            random.fold_in(base_key, 10_004),
            args.repeats,
        )
    )
    started = time.perf_counter()

    print(
        "Learning translation-invariant Gaussian-cooling preconditioners: "
        f"d={model.side}, D={model.dimension}, repeats={args.repeats}",
        flush=True,
    )
    call = lambda key: _fourier_cooling_call(key, model, args)
    spectra, _ = _run_timed_repeats(call, run_keys, warmup_key)

    reference = reference_samples(reference_key, model, args)
    reference.block_until_ready()
    reference_spectrum = np.asarray(
        sample_power_spectrum(reference, model.lattice_shape)
    )
    conditions = np.asarray(
        [
            spectral_relative_condition(
                spectrum,
                reference_spectrum,
                args.metric_floor,
            )
            for spectrum in spectra
        ],
        dtype=float,
    )
    representative_index = _representative_repeat_index(conditions)
    representative_spectrum = spectra[representative_index]

    storage_bytes = (
        model.dimension
        * args.trajectory_samples
        * np.dtype(np.float64 if dtype == jnp.float64 else np.float32).itemsize
    )
    print(
        "Post-cooling diagnostics: full first covariance row at "
        f"d={model.side}, T={args.trajectory_samples:,} "
        f"(temporary storage {storage_bytes / 2**20:.1f} MiB)",
        flush=True,
    )
    diagnostics = run_preconditioned_diagnostics(
        random.fold_in(root_key, 10_000),
        model,
        representative_spectrum,
        args,
    )
    return DiagnosticsResult(
        side=model.side,
        selected_relative_condition=float(conditions[representative_index]),
        all_relative_conditions=conditions,
        diagnostics=diagnostics,
        elapsed_seconds=time.perf_counter() - started,
    )


def make_figures(
    result: DiagnosticsResult,
    args: argparse.Namespace,
) -> dict[str, plt.Figure]:
    """Create the standalone IAT and ergodicity figures."""

    _configure_plot_style()
    diagnostics = result.diagnostics
    figures: dict[str, plt.Figure] = {}

    iat_figure, ax = plt.subplots(
        figsize=(6.3, 4.7),
        constrained_layout=True,
    )
    distances = diagnostics.distances
    autocorrelation_times = diagnostics.correlator_autocorrelation_times
    finite = np.isfinite(autocorrelation_times) & (autocorrelation_times > 0.0)
    reliable = finite & diagnostics.correlator_autocorrelation_reliable
    unreliable = finite & ~diagnostics.correlator_autocorrelation_reliable
    if np.any(reliable):
        ax.scatter(
            distances[reliable],
            autocorrelation_times[reliable],
            s=8,
            color="#CC79A7",
            alpha=0.22,
            linewidth=0,
            rasterized=True,
            label="IAT criterion met",
        )
    if np.any(unreliable):
        ax.scatter(
            distances[unreliable],
            autocorrelation_times[unreliable],
            s=10,
            facecolors="none",
            edgecolors="#CC79A7",
            alpha=0.28,
            linewidth=0.45,
            rasterized=True,
            label="Short-chain estimate",
        )

    if np.any(finite):
        radial_bins = np.floor(distances[finite]).astype(int)
        unique_bins = np.unique(radial_bins)
        radial_centers = np.asarray(
            [
                np.median(distances[finite][radial_bins == radial_bin])
                for radial_bin in unique_bins
            ]
        )
        radial_median = np.asarray(
            [
                np.median(autocorrelation_times[finite][radial_bins == radial_bin])
                for radial_bin in unique_bins
            ]
        )
        radial_lower = np.asarray(
            [
                np.quantile(
                    autocorrelation_times[finite][radial_bins == radial_bin],
                    0.25,
                )
                for radial_bin in unique_bins
            ]
        )
        radial_upper = np.asarray(
            [
                np.quantile(
                    autocorrelation_times[finite][radial_bins == radial_bin],
                    0.75,
                )
                for radial_bin in unique_bins
            ]
        )
        ax.fill_between(
            radial_centers,
            radial_lower,
            radial_upper,
            color="#009E73",
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            radial_centers,
            radial_median,
            color="#009E73",
            linewidth=1.8,
            label="Annular median and IQR",
        )
    ax.set_xlabel("Periodic distance from the origin")
    ax.set_ylabel(r"Integrated autocorrelation time $\tau_{\rm int}$ (ULMC steps)")
    ax.set_title(
        "Two-point-correlator IATs after "
        "translation-invariant Gaussian cooling\n"
        rf"$d={result.side},\ T={args.trajectory_samples},\ "
        rf"N_{{\rm burn}}={args.trajectory_burnin}$"
        + "\n"
        + _phi4_parameter_subtitle(args)
    )
    ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncols=3,
            fontsize=7.6,
        )
    figures[f"two_point_iat_d{result.side}"] = iat_figure

    mean_figure, ax = plt.subplots(
        figsize=(6.2, 4.9),
        constrained_layout=True,
    )
    site_reliable = diagnostics.site_field_autocorrelation_reliable & np.isfinite(
        diagnostics.site_mean_z_scores
    )
    displayed_z_scores = np.ma.masked_where(
        ~site_reliable,
        diagnostics.site_mean_z_scores,
    )
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad(color="#D7D7D7")
    image = ax.imshow(
        displayed_z_scores,
        origin="lower",
        cmap=colormap,
        vmin=-4.0,
        vmax=4.0,
        interpolation="nearest",
        rasterized=True,
    )
    colorbar = mean_figure.colorbar(
        image,
        ax=ax,
        shrink=0.88,
        pad=0.03,
        extend="both",
    )
    colorbar.set_label(r"Final-half site-mean $z_x$")
    ax.set_xlabel("Lattice coordinate $x_2$")
    ax.set_ylabel("Lattice coordinate $x_1$")
    ax.set_title(
        "Final-half site-mean z-scores\n"
        rf"$d={result.side}$, "
        rf"$T_{{\rm retained}}={args.trajectory_samples}$, "
        rf"$N_{{\rm half}}={diagnostics.final_half_sample_count}$, "
        rf"$N_{{\rm burn}}={args.trajectory_burnin}$"
        + "\n"
        + _phi4_parameter_subtitle(args)
    )
    figures[f"ergodicity_site_mean_z_scores_d{result.side}"] = mean_figure
    return figures


def print_summary(result: DiagnosticsResult) -> None:
    """Report preconditioner quality and trajectory diagnostics."""

    diagnostics = result.diagnostics
    print(
        "\nRepresentative translation-invariant Gaussian-cooling "
        f"preconditioner: kappa_rel={result.selected_relative_condition:.4g} "
        f"from {len(result.all_relative_conditions)} repeat(s)."
    )
    finite_times = diagnostics.correlator_autocorrelation_times[
        np.isfinite(diagnostics.correlator_autocorrelation_times)
        & (diagnostics.correlator_autocorrelation_times > 0.0)
    ]
    print(
        "First covariance-row IATs after translation-invariant "
        f"Gaussian cooling, side={result.side}:"
    )
    if finite_times.size:
        print(
            f"  entries={len(diagnostics.correlator_autocorrelation_times):,}, "
            f"median tau={np.median(finite_times):.3f}, "
            f"max tau={np.max(finite_times):.3f}, "
            f"reliable="
            f"{np.count_nonzero(diagnostics.correlator_autocorrelation_reliable):,}"
            f"/{len(diagnostics.correlator_autocorrelation_reliable):,}"
        )
    else:
        print(
            f"  entries={len(diagnostics.correlator_autocorrelation_times):,}; "
            "no finite positive IAT estimates"
        )
    print(
        "  final-half site-mean RMS="
        f"{np.sqrt(np.mean(diagnostics.final_half_site_means**2)):.4g} "
        f"from {diagnostics.final_half_sample_count:,} samples"
    )
    site_reliable = diagnostics.site_field_autocorrelation_reliable & np.isfinite(
        diagnostics.site_mean_z_scores
    )
    reliable_count = int(np.count_nonzero(site_reliable))
    if reliable_count:
        reliable_iats = diagnostics.site_field_autocorrelation_times[site_reliable]
        reliable_ess = diagnostics.site_mean_effective_sample_sizes[site_reliable]
        reliable_z = diagnostics.site_mean_z_scores[site_reliable]
        print(
            "  final-half site-field diagnostics: "
            f"reliable={reliable_count:,}/{site_reliable.size:,}, "
            f"median tau={np.median(reliable_iats):.3f}, "
            f"median ESS={np.median(reliable_ess):.1f}, "
            f"|z|<=1.96="
            f"{100.0 * np.mean(np.abs(reliable_z) <= 1.96):.1f}%"
        )
    else:
        print(
            "  final-half site-field diagnostics: no reliable IAT "
            "estimates; the retained chain is too short for standardized "
            "mean inference"
        )
    print(
        "Diagnostic wall time (including compilation): "
        f"{result.elapsed_seconds:.2f}s"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the focused diagnostic command-line interface."""

    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Learn a translation-invariant Gaussian-cooling preconditioner "
            "and diagnose two-point and site-mean mixing on lattice phi4."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--side",
        "--diagnostic-side",
        dest="side",
        type=int,
        default=100,
        help="Lattice side used for all diagnostics.",
    )
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument(
        "--quartic",
        type=float,
        default=0.5,
        help="Quartic coupling lambda.",
    )
    parser.add_argument("--mass", type=float, default=0.25)
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Truncation radius R; use 'inf' for the genuine quartic.",
    )
    parser.add_argument(
        "--cooling-design-radius",
        type=float,
        default=4.0,
        help="Operational curvature radius when R=inf.",
    )
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
    parser.add_argument("--reference-chains", type=int, default=256)
    parser.add_argument("--reference-steps", type=int, default=256)
    parser.add_argument(
        "--reference-step-size",
        type=float,
        default=None,
        help="Reference ULMC step size; by default reuse --step-size.",
    )
    parser.add_argument("--trajectory-burnin", type=int, default=1024)
    parser.add_argument("--trajectory-samples", type=int, default=8192)
    parser.add_argument("--iat-tolerance", type=int, default=50)
    parser.add_argument("--trajectory-chunk-size", type=int, default=256)
    parser.add_argument("--iat-batch-size", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--output-prefix",
        "--output",
        dest="output",
        type=Path,
        default=project_dir / "figures" / "phi4_diagnostics",
        help="Base prefix for the two separate PDF outputs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced CPU smoke-test configuration.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Require a visible JAX GPU and use the old moderate-size GPU "
            "diagnostic preset unless options are explicitly overridden."
        ),
    )
    return parser


def apply_quick_configuration(
    args: argparse.Namespace,
    explicit_destinations: set[str] | None = None,
) -> None:
    """Apply the old quick diagnostic preset, preserving overrides."""

    if not args.quick:
        return
    explicitly_set = set() if explicit_destinations is None else explicit_destinations
    quick_values = {
        "side": 8,
        "repeats": 1,
        "chains": 64,
        "steps": 20,
        "stages": 4,
        "reference_chains": 96,
        "reference_steps": 48,
        "trajectory_burnin": 100,
        "trajectory_samples": 500,
        "cooling_design_radius": 2.0,
        "dtype": "float32",
    }
    for destination, value in quick_values.items():
        if destination not in explicitly_set:
            setattr(args, destination, value)
    if "cooling_design_radius" not in explicitly_set:
        args.cooling_design_radius = 2.0


def apply_gpu_configuration(
    args: argparse.Namespace,
    explicit_destinations: set[str] | None = None,
) -> None:
    """Apply the prior opt-in GPU diagnostic settings."""

    if not args.gpu:
        return
    explicitly_set = set() if explicit_destinations is None else explicit_destinations
    gpu_values = {
        "side": 64,
        "repeats": 1,
        "chains": 64,
        "steps": 32,
        "stages": 8,
        "reference_chains": 128,
        "reference_steps": 128,
        "dtype": "float32",
    }
    for destination, value in gpu_values.items():
        if destination not in explicitly_set:
            setattr(args, destination, value)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate the focused learning and diagnostic configuration."""

    if integrated_time is None:
        raise RuntimeError(
            "Autocorrelation diagnostics require `emcee`. Install it with "
            "`python -m pip install emcee`."
        )
    if args.gpu and not _available_gpu_devices():
        raise RuntimeError(
            "--gpu requires a JAX GPU backend, but no GPU device was found."
        )
    if args.gpu and args.dtype == "float64":
        warnings.warn(
            "Float64 doubles trajectory storage relative to float32.",
            RuntimeWarning,
            stacklevel=2,
        )
    counts = {
        "repeats": args.repeats,
        "chains": args.chains,
        "steps": args.steps,
        "stages": args.stages,
        "reference-chains": args.reference_chains,
        "reference-steps": args.reference_steps,
        "trajectory-samples": args.trajectory_samples,
        "iat-tolerance": args.iat_tolerance,
        "trajectory-chunk-size": args.trajectory_chunk_size,
        "iat-batch-size": args.iat_batch_size,
    }
    invalid = [name for name, value in counts.items() if value <= 0]
    if invalid:
        raise ValueError("These counts must be positive: " + ", ".join(invalid))
    if args.side < 2:
        raise ValueError("--side must be at least two.")
    if args.chains < 2 or args.reference_chains < 4:
        raise ValueError(
            "--chains must be at least two and --reference-chains at least four."
        )
    if args.trajectory_burnin < 0:
        raise ValueError("--trajectory-burnin must be nonnegative.")
    if args.trajectory_samples < 4:
        raise ValueError("--trajectory-samples must be at least four.")
    positive_scalars = {
        "quartic": args.quartic,
        "mass": args.mass,
        "cooling-design-radius": args.cooling_design_radius,
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
            "These values must be finite and positive: " + ", ".join(invalid_positive)
        )
    if not np.isfinite(args.cooling_gamma) or not 0 < args.cooling_gamma < 1:
        raise ValueError("--cooling-gamma must lie in (0,1).")
    if args.radius <= 0.0 or np.isnan(args.radius) or np.isneginf(args.radius):
        raise ValueError("--radius must be positive finite or 'inf'.")
    nonnegative = {
        "beta": args.beta,
        "covariance-ridge": args.covariance_ridge,
    }
    invalid_nonnegative = [
        name
        for name, value in nonnegative.items()
        if not np.isfinite(value) or value < 0.0
    ]
    if invalid_nonnegative:
        raise ValueError(
            "These values must be finite and nonnegative: "
            + ", ".join(invalid_nonnegative)
        )
    if args.reference_step_size is not None and (
        not np.isfinite(args.reference_step_size) or args.reference_step_size <= 0.0
    ):
        raise ValueError("--reference-step-size must be finite and positive.")
    if np.isposinf(args.radius):
        warnings.warn(
            "R=inf is the genuine quartic target. The finite design radius "
            "is operational and does not provide a global Hessian bound.",
            RuntimeWarning,
            stacklevel=2,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the focused diagnostic experiment and save its PDFs."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.quick and args.gpu:
        parser.error("--quick and --gpu are mutually exclusive presets.")
    explicit_destinations = _explicit_cli_destinations(parser, arguments)
    apply_quick_configuration(args, explicit_destinations)
    apply_gpu_configuration(args, explicit_destinations)
    validate_arguments(args)

    result = run_experiment(args)
    figures = make_figures(result, args)
    output_paths = save_publication_figures(figures, args.output)
    for figure in figures.values():
        plt.close(figure)
    print_summary(result)
    for plot_name, path in output_paths.items():
        print(f"Saved {plot_name.replace('_', ' ')} PDF: {path}")


if __name__ == "__main__":
    main()
