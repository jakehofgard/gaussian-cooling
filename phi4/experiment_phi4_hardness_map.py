"""Fixed-budget hardness maps for periodic lattice phi4 targets.

This focused experiment fixes ``beta=2`` and the truncation radius ``R=4``,
then varies the mass ``m``, quartic coupling ``lambda``, and lattice side
length.  At every grid cell, three translation-invariant preconditioners use
the same practitioner-selected gradient budget.  The default budget is
``n=512`` chains, ``N=128`` ULMC transitions per stage, and ``K=12`` stages.

The script writes two vector PDFs per lattice side plus a compressed NPZ file
containing all repeats and reference-sampler metadata.  Use ``--quick`` for a
small CPU smoke test.  ``--gpu`` only requires a visible JAX GPU; it does not
change the scientific grid or budgets.
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

from gaussian_cooling_algs import sample_power_spectrum
from .lattice_phi4 import (
    COOLING_COMPARISON,
    EMPIRICAL_COMPARISON,
    HARDNESS_BETA,
    HARDNESS_RADIUS,
    SCALABLE_METHODS,
    TRANSLATION_AVERAGED_ULMC,
    LatticeModel,
    _fourier_cooling_call,
    _fourier_empirical_baseline_call,
    _run_timed_repeats,
    _safe_spectrum,
    _translation_averaged_ulmc_call,
    make_lattice_model,
    reference_samples,
    spectral_relative_condition,
    validate_lattice_model,
)
from .phi4_cli import (
    _available_gpu_devices,
    _explicit_cli_destinations,
    _parse_positive_values,
    _parse_sides,
)
from .phi4_plotting import (
    _configure_hardness_axes,
    _configure_plot_style,
    _logarithmic_cell_edges,
    save_publication_figures,
)

import matplotlib.pyplot as plt


@dataclass
class HardnessMapResult:
    """Scalable ``(mass, quartic coupling)`` maps at several lattice sides."""

    sides: np.ndarray
    masses: np.ndarray
    quartics: np.ndarray
    relative_conditions: dict[str, np.ndarray]
    hessian_condition_bounds: np.ndarray
    chains: int
    steps: int
    reference_conditions: np.ndarray
    reference_split_conditions: np.ndarray
    reference_chains: np.ndarray
    reference_step_sizes: np.ndarray
    reference_steps: np.ndarray
    stages: int
    repeats: int
    elapsed_seconds: float


@dataclass
class _HardnessCellEvaluation:
    """Three scalable-method estimates and reference metadata for one cell."""

    relative_conditions: dict[str, np.ndarray]
    reference_condition: float
    reference_split_condition: float
    reference_chains: int
    reference_step_size: float
    reference_steps: int


def _hardness_budget(
    model: LatticeModel,
    args: argparse.Namespace,
) -> tuple[float, int, int]:
    """Return ``(kappa_H, n, N)`` for one fixed-budget map target."""

    condition_bound = model.global_smoothness_bound / model.strong_convexity
    return (
        float(condition_bound),
        int(args.hardness_chains),
        int(args.hardness_steps),
    )


def _hardness_reference_arguments(
    model: LatticeModel,
    method_chains: int,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Build stable, independently controlled reference-sampler settings."""

    reference_args = argparse.Namespace(**vars(args))
    transformed_smoothness = 1.0 + 3.0 * model.quartic * model.radius**2 / model.mass
    stable_step_size = args.hardness_reference_margin / np.sqrt(transformed_smoothness)
    reference_step_size = min(
        args.hardness_reference_max_step,
        stable_step_size,
    )
    reference_args.reference_step_size = reference_step_size
    reference_args.reference_steps = max(
        1,
        int(np.ceil(args.hardness_reference_time / reference_step_size)),
    )
    reference_args.reference_chains = max(
        args.hardness_reference_min_chains,
        int(np.ceil(args.hardness_reference_chain_factor * method_chains)),
    )
    return reference_args


def _evaluate_hardness_cell(
    key: jax.Array,
    model: LatticeModel,
    args: argparse.Namespace,
) -> _HardnessCellEvaluation:
    """Evaluate the three O(D)-storage methods at one map cell."""

    adaptive_root, plain_root, reference_key = random.split(key, 3)
    adaptive_warm_key, adaptive_run_root = random.split(adaptive_root)
    adaptive_keys = list(random.split(adaptive_run_root, args.hardness_repeats))
    plain_warm_key, plain_run_root = random.split(plain_root)
    plain_keys = list(random.split(plain_run_root, args.hardness_repeats))

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

    def plain_call(run_key: jax.Array) -> jax.Array:
        return _translation_averaged_ulmc_call(run_key, model, args)

    plain_spectra, _ = _run_timed_repeats(
        plain_call,
        plain_keys,
        plain_warm_key,
    )

    reference_args = _hardness_reference_arguments(
        model,
        args.chains,
        args,
    )
    reference = reference_samples(reference_key, model, reference_args)
    reference.block_until_ready()
    reference_spectrum = np.asarray(
        sample_power_spectrum(reference, model.lattice_shape)
    )
    reference_midpoint = reference.shape[0] // 2
    first_reference_spectrum = np.asarray(
        sample_power_spectrum(
            reference[:reference_midpoint],
            model.lattice_shape,
        )
    )
    second_reference_spectrum = np.asarray(
        sample_power_spectrum(
            reference[reference_midpoint:],
            model.lattice_shape,
        )
    )
    reference_split_condition = spectral_relative_condition(
        first_reference_spectrum,
        second_reference_spectrum,
        args.metric_floor,
    )
    safe_reference = _safe_spectrum(
        reference_spectrum,
        args.metric_floor,
    )
    spectra_by_method = {
        COOLING_COMPARISON: cooling_spectra,
        EMPIRICAL_COMPARISON: empirical_spectra,
        TRANSLATION_AVERAGED_ULMC: plain_spectra,
    }
    relative_conditions = {
        method: np.asarray(
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
        for method, spectra in spectra_by_method.items()
    }
    return _HardnessCellEvaluation(
        relative_conditions=relative_conditions,
        reference_condition=float(np.max(safe_reference) / np.min(safe_reference)),
        reference_split_condition=reference_split_condition,
        reference_chains=int(reference_args.reference_chains),
        reference_step_size=float(reference_args.reference_step_size),
        reference_steps=int(reference_args.reference_steps),
    )


def run_hardness_map(args: argparse.Namespace) -> HardnessMapResult:
    """Run the fixed-``R``, fixed-``beta`` hardness-map experiment."""

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    sides = np.asarray(args.hardness_sides, dtype=int)
    masses = np.asarray(args.hardness_masses, dtype=float)
    quartics = np.asarray(args.hardness_quartics, dtype=float)
    shape = (
        len(sides),
        len(quartics),
        len(masses),
        args.hardness_repeats,
    )
    relative_conditions = {
        method: np.empty(shape, dtype=float) for method in SCALABLE_METHODS
    }
    cell_shape = shape[:-1]
    condition_bounds = np.empty(cell_shape, dtype=float)
    reference_conditions = np.empty(cell_shape, dtype=float)
    reference_split_conditions = np.empty(cell_shape, dtype=float)
    reference_chains = np.empty(cell_shape, dtype=int)
    reference_step_sizes = np.empty(cell_shape, dtype=float)
    reference_steps = np.empty(cell_shape, dtype=int)

    root_key = random.fold_in(random.PRNGKey(args.seed), 400_000)
    started = time.perf_counter()
    num_cells_per_side = len(quartics) * len(masses)
    for side_index, side_value in enumerate(sides):
        side = int(side_value)
        for quartic_index, quartic in enumerate(quartics):
            for mass_index, mass in enumerate(masses):
                model = make_lattice_model(
                    side,
                    HARDNESS_BETA,
                    float(quartic),
                    float(mass),
                    HARDNESS_RADIUS,
                    dtype,
                )
                validate_lattice_model(model)
                condition, num_chains, num_steps = _hardness_budget(
                    model,
                    args,
                )
                condition_bounds[
                    side_index,
                    quartic_index,
                    mass_index,
                ] = condition
                cell_args = argparse.Namespace(**vars(args))
                cell_args.chains = num_chains
                cell_args.steps = num_steps
                cell_args.stages = args.hardness_stages
                flat_cell_index = (
                    side_index * num_cells_per_side
                    + quartic_index * len(masses)
                    + mass_index
                )
                cell_key = random.fold_in(root_key, flat_cell_index)
                print(
                    "Hardness cell: "
                    f"side={side} (D={model.dimension}), "
                    f"m={mass:g}, lambda={quartic:g}, "
                    f"kappa_H={condition:.4g}, n={num_chains}, "
                    f"N={num_steps}, K={args.hardness_stages}",
                    flush=True,
                )
                evaluation = _evaluate_hardness_cell(
                    cell_key,
                    model,
                    cell_args,
                )
                for method in SCALABLE_METHODS:
                    relative_conditions[method][
                        side_index,
                        quartic_index,
                        mass_index,
                    ] = evaluation.relative_conditions[method]
                reference_conditions[
                    side_index,
                    quartic_index,
                    mass_index,
                ] = evaluation.reference_condition
                reference_split_conditions[
                    side_index,
                    quartic_index,
                    mass_index,
                ] = evaluation.reference_split_condition
                reference_chains[
                    side_index,
                    quartic_index,
                    mass_index,
                ] = evaluation.reference_chains
                reference_step_sizes[
                    side_index,
                    quartic_index,
                    mass_index,
                ] = evaluation.reference_step_size
                reference_steps[
                    side_index,
                    quartic_index,
                    mass_index,
                ] = evaluation.reference_steps

    maximum_reference_disagreement = float(np.max(reference_split_conditions))
    if (
        not args.quick
        and maximum_reference_disagreement
        > args.hardness_reference_consistency_threshold
    ):
        warnings.warn(
            "The maximum half-sample reference consistency condition is "
            f"{maximum_reference_disagreement:.3g}, above "
            f"{args.hardness_reference_consistency_threshold:g}. Increase "
            "the hardness reference time and chain settings before "
            "interpreting method differences.",
            RuntimeWarning,
            stacklevel=2,
        )

    return HardnessMapResult(
        sides=sides,
        masses=masses,
        quartics=quartics,
        relative_conditions=relative_conditions,
        hessian_condition_bounds=condition_bounds,
        chains=args.hardness_chains,
        steps=args.hardness_steps,
        reference_conditions=reference_conditions,
        reference_split_conditions=reference_split_conditions,
        reference_chains=reference_chains,
        reference_step_sizes=reference_step_sizes,
        reference_steps=reference_steps,
        stages=args.hardness_stages,
        repeats=args.hardness_repeats,
        elapsed_seconds=time.perf_counter() - started,
    )


def make_hardness_map_figures(
    result: HardnessMapResult,
    args: argparse.Namespace,
) -> dict[str, plt.Figure]:
    """Create absolute-quality and adaptive-gain maps for every side."""

    del args  # Plot metadata is carried by the result and fixed constants.
    _configure_plot_style()
    medians = {
        method: np.median(values, axis=-1)
        for method, values in result.relative_conditions.items()
    }
    absolute_log_values = np.concatenate(
        [np.log10(values).reshape((-1,)) for values in medians.values()]
    )
    absolute_min = float(np.min(absolute_log_values))
    absolute_max = float(np.max(absolute_log_values))
    if np.isclose(absolute_min, absolute_max):
        absolute_min -= 0.5
        absolute_max += 0.5

    baseline = medians[TRANSLATION_AVERAGED_ULMC]
    gain_methods = (COOLING_COMPARISON, EMPIRICAL_COMPARISON)
    gains = {method: np.log10(baseline / medians[method]) for method in gain_methods}
    gain_limit = max(float(np.max(np.abs(values))) for values in gains.values())
    gain_limit = max(gain_limit, 1e-6)

    mass_edges = _logarithmic_cell_edges(result.masses)
    quartic_edges = _logarithmic_cell_edges(result.quartics)
    figures: dict[str, plt.Figure] = {}
    short_titles = {
        COOLING_COMPARISON: ("Translation-invariant\nGaussian cooling"),
        EMPIRICAL_COMPARISON: ("Translation-invariant\nempirical preconditioning"),
        TRANSLATION_AVERAGED_ULMC: ("Translation-averaged\nULMC covariance"),
    }
    for side_index, side_value in enumerate(result.sides):
        side = int(side_value)
        dimension = side * side
        budget_text = (
            rf"$R={HARDNESS_RADIUS:g},\ \beta={HARDNESS_BETA:g},\ "
            rf"d={side},\ D=d^2={dimension},\ "
            rf"n={result.chains},\ N={result.steps},\ "
            rf"K={result.stages}$"
        )

        absolute_figure, axes = plt.subplots(
            1,
            len(SCALABLE_METHODS),
            figsize=(12.2, 4.6),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        absolute_image = None
        for ax, method in zip(axes, SCALABLE_METHODS, strict=True):
            absolute_image = ax.pcolormesh(
                mass_edges,
                quartic_edges,
                np.log10(medians[method][side_index]),
                cmap="viridis",
                vmin=absolute_min,
                vmax=absolute_max,
                shading="flat",
            )
            _configure_hardness_axes(
                ax,
                result.masses,
                result.quartics,
            )
            ax.set_title(short_titles[method])
        for ax in axes[1:]:
            ax.set_ylabel("")
        assert absolute_image is not None
        absolute_colorbar = absolute_figure.colorbar(
            absolute_image,
            ax=axes,
            shrink=0.88,
            pad=0.02,
        )
        absolute_colorbar.set_label(r"$\log_{10}(\widetilde{\kappa}_{\mathrm{rel}})$")
        absolute_figure.suptitle(
            r"Lattice $\phi^4$ preconditioner quality" + "\n" + budget_text,
            fontsize=10.6,
        )
        figures[f"hardness_absolute_d{side}"] = absolute_figure

        gain_figure, axes = plt.subplots(
            1,
            len(gain_methods),
            figsize=(8.5, 4.6),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        gain_image = None
        for ax, method in zip(axes, gain_methods, strict=True):
            gain_image = ax.pcolormesh(
                mass_edges,
                quartic_edges,
                gains[method][side_index],
                cmap="RdBu_r",
                vmin=-gain_limit,
                vmax=gain_limit,
                shading="flat",
            )
            _configure_hardness_axes(
                ax,
                result.masses,
                result.quartics,
            )
            ax.set_title(short_titles[method])
        for ax in axes[1:]:
            ax.set_ylabel("")
        assert gain_image is not None
        gain_colorbar = gain_figure.colorbar(
            gain_image,
            ax=axes,
            shrink=0.88,
            pad=0.02,
        )
        gain_colorbar.set_label(
            r"$\log_{10}(\widetilde{\kappa}_{\rm rel,\,baseline}/"
            r"\widetilde{\kappa}_{\rm rel,\,method})$"
        )
        gain_figure.suptitle(
            "Relative performance against translation-averaged ULMC\n" + budget_text,
            fontsize=10.6,
        )
        figures[f"hardness_adaptive_gain_d{side}"] = gain_figure

    return figures


def save_hardness_map_data(
    result: HardnessMapResult,
    output: Path,
    args: argparse.Namespace,
) -> Path:
    """Save numerical map values and fixed budgets beside the PDFs."""

    output = output.expanduser().resolve()
    stem = output.with_suffix("") if output.suffix else output
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = stem.with_name(f"{stem.name}_hardness_map_data.npz")
    np.savez_compressed(
        path,
        sides=result.sides,
        dimensions=result.sides**2,
        masses=result.masses,
        quartics=result.quartics,
        beta=np.asarray(HARDNESS_BETA),
        radius=np.asarray(HARDNESS_RADIUS),
        seed=np.asarray(args.seed),
        dtype=np.asarray(args.dtype),
        repeats=np.asarray(result.repeats),
        cooling_gamma=np.asarray(args.cooling_gamma),
        delta=np.asarray(args.delta),
        friction=np.asarray(args.friction),
        step_size=np.asarray(args.step_size),
        covariance_ridge=np.asarray(args.covariance_ridge),
        metric_floor=np.asarray(args.metric_floor),
        budget_mode=np.asarray("fixed"),
        reference_chain_factor=np.asarray(args.hardness_reference_chain_factor),
        reference_minimum_chains=np.asarray(args.hardness_reference_min_chains),
        reference_max_step=np.asarray(args.hardness_reference_max_step),
        reference_margin=np.asarray(args.hardness_reference_margin),
        reference_time=np.asarray(args.hardness_reference_time),
        reference_consistency_threshold=np.asarray(
            args.hardness_reference_consistency_threshold
        ),
        cooling=result.relative_conditions[COOLING_COMPARISON],
        empirical=result.relative_conditions[EMPIRICAL_COMPARISON],
        translation_averaged=result.relative_conditions[TRANSLATION_AVERAGED_ULMC],
        hessian_condition_bounds=result.hessian_condition_bounds,
        chains=result.chains,
        steps=result.steps,
        stages=np.asarray(result.stages),
        reference_conditions=result.reference_conditions,
        reference_split_conditions=result.reference_split_conditions,
        reference_chains=result.reference_chains,
        reference_step_sizes=result.reference_step_sizes,
        reference_steps=result.reference_steps,
    )
    return path


def print_hardness_map_summary(result: HardnessMapResult) -> None:
    """Report axes, budgets, and cell winners for the hardness-map test."""

    print(
        "\nHardness-map axes: horizontal mass m="
        + ",".join(f"{value:g}" for value in result.masses)
        + "; vertical quartic coupling lambda="
        + ",".join(f"{value:g}" for value in result.quartics)
    )
    print(
        f"Fixed beta={HARDNESS_BETA:g}, R={HARDNESS_RADIUS:g}. "
        f"Every cell uses fixed n={result.chains}, N={result.steps}, "
        f"K={result.stages}. "
        "Ambient dimension is D=d^2."
    )
    median_conditions = {
        method: np.median(values, axis=-1)
        for method, values in result.relative_conditions.items()
    }
    stacked = np.stack(
        [median_conditions[method] for method in SCALABLE_METHODS],
        axis=0,
    )
    winners = np.argmin(stacked, axis=0)
    for side_index, side_value in enumerate(result.sides):
        side = int(side_value)
        reference_chain_grid = result.reference_chains[side_index]
        reference_step_grid = result.reference_steps[side_index]
        reference_step_size_grid = result.reference_step_sizes[side_index]
        reference_split_grid = result.reference_split_conditions[side_index]
        winner_counts = {
            short_name: int(np.count_nonzero(winners[side_index] == index))
            for index, short_name in enumerate(
                ("cooling", "TI empirical", "TI-averaged ULMC")
            )
        }
        print(
            f"  side={side} (D={side * side}): "
            f"N={result.steps}, n={result.chains}, "
            + ", ".join(
                f"{name} wins {count} cell(s)" for name, count in winner_counts.items()
            )
        )
        print(
            "    reference ranges: "
            f"n_ref={np.min(reference_chain_grid)}--"
            f"{np.max(reference_chain_grid)}, "
            f"N_ref={np.min(reference_step_grid)}--"
            f"{np.max(reference_step_grid)}, "
            f"h_ref={np.min(reference_step_size_grid):.3g}--"
            f"{np.max(reference_step_size_grid):.3g}, "
            f"half-sample kappa_rel="
            f"{np.min(reference_split_grid):.3g}--"
            f"{np.max(reference_split_grid):.3g}"
        )
    print(
        "Hardness-map wall time (including compilation): "
        f"{result.elapsed_seconds:.2f}s"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the focused hardness-map command-line interface."""

    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Map fixed-budget lattice phi4 preconditioner quality over mass "
            "and quartic coupling at fixed beta=2 and R=4."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sides",
        "--hardness-sides",
        "--hardness-map-sides",
        dest="hardness_sides",
        type=_parse_sides,
        default=[8, 16, 32, 64, 128],
        help="Comma-separated lattice side lengths.",
    )
    parser.add_argument(
        "--masses",
        "--hardness-masses",
        "--hardness-map-masses",
        dest="hardness_masses",
        type=_parse_positive_values,
        default=[0.01, 0.05, 0.25],
        help="Horizontal-axis mass values.",
    )
    parser.add_argument(
        "--lambdas",
        "--hardness-lambdas",
        "--hardness-map-lambdas",
        dest="hardness_quartics",
        type=_parse_positive_values,
        default=[0.1, 0.5, 2.0],
        help="Vertical-axis quartic couplings.",
    )
    parser.add_argument(
        "--stages",
        "--hardness-stages",
        dest="hardness_stages",
        type=int,
        default=12,
        help="Fixed equal stage count K at every grid cell.",
    )
    parser.add_argument(
        "--repeats",
        "--hardness-repeats",
        dest="hardness_repeats",
        type=int,
        default=1,
        help="Independent repeats at every grid cell.",
    )
    parser.add_argument(
        "--chains",
        "--hardness-chains",
        dest="hardness_chains",
        type=int,
        default=512,
        help="Fixed chain count n used by every method at every grid cell.",
    )
    parser.add_argument(
        "--steps",
        "--hardness-steps",
        dest="hardness_steps",
        type=int,
        default=128,
        help="Fixed transitions N per stage for every method.",
    )
    parser.add_argument(
        "--reference-chain-factor",
        "--hardness-reference-chain-factor",
        dest="hardness_reference_chain_factor",
        type=float,
        default=2.0,
        help="Reference chains as a multiple of the method chain count.",
    )
    parser.add_argument(
        "--reference-min-chains",
        "--hardness-reference-min-chains",
        dest="hardness_reference_min_chains",
        type=int,
        default=512,
        help="Minimum independent reference endpoint count at each cell.",
    )
    parser.add_argument(
        "--reference-max-step",
        "--hardness-reference-max-step",
        dest="hardness_reference_max_step",
        type=float,
        default=0.03,
        help="Maximum step size for the independently tuned reference.",
    )
    parser.add_argument(
        "--reference-margin",
        "--hardness-reference-margin",
        dest="hardness_reference_margin",
        type=float,
        default=0.15,
        help="Maximum h_ref*sqrt(L_ref) for the reference.",
    )
    parser.add_argument(
        "--reference-time",
        "--hardness-reference-time",
        dest="hardness_reference_time",
        type=float,
        default=2.0,
        help="Physical integration time retained by each reference run.",
    )
    parser.add_argument(
        "--reference-consistency-threshold",
        "--hardness-reference-consistency-threshold",
        dest="hardness_reference_consistency_threshold",
        type=float,
        default=1.5,
        help=(
            "Warn when kappa_rel between two reference half-samples exceeds "
            "this value."
        ),
    )
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
        help="Theoretical preconditioning tolerance retained as metadata.",
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
        help="ULMC integration step size.",
    )
    parser.add_argument(
        "--covariance-ridge",
        type=float,
        default=0.0,
        help="Optional ridge inside adaptive covariance updates.",
    )
    parser.add_argument(
        "--metric-floor",
        type=float,
        default=1e-10,
        help="Positive relative floor in spectral condition metrics.",
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
        default=project_dir / "figures" / "phi4_hardness",
        help="Base prefix for separate PDF and NPZ outputs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use a reduced deterministic CPU smoke-test grid. Explicit "
            "options override the corresponding preset values."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Require a visible JAX GPU without changing the map grid, "
            "budgets, or dtype."
        ),
    )
    parser.add_argument(
        "--hardness-map",
        "--hardness-map-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def apply_quick_configuration(
    args: argparse.Namespace,
    explicit_destinations: set[str] | None = None,
) -> None:
    """Shrink the map and budgets while preserving explicit overrides."""

    if not args.quick:
        return
    explicitly_set = set() if explicit_destinations is None else explicit_destinations
    quick_values = {
        "hardness_sides": [4, 8],
        "hardness_masses": [0.05, 0.25],
        "hardness_quartics": [0.1, 0.5],
        "hardness_stages": 2,
        "hardness_repeats": 1,
        "hardness_chains": 8,
        "hardness_steps": 2,
        "hardness_reference_chain_factor": 1.0,
        "hardness_reference_min_chains": 16,
        "hardness_reference_margin": 0.2,
        "hardness_reference_time": 0.06,
    }
    for destination, value in quick_values.items():
        if destination not in explicitly_set:
            setattr(args, destination, value)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate only settings used by this focused experiment."""

    if args.gpu and not _available_gpu_devices():
        raise RuntimeError(
            "--gpu requires a JAX GPU backend, but no GPU device was found. "
            "Without --gpu, the experiment automatically uses whichever "
            "JAX backend is available."
        )
    if args.gpu and args.dtype == "float64":
        warnings.warn(
            "Float64 approximately doubles hardness-map state storage. Run "
            "the float32 map first to establish the memory margin.",
            RuntimeWarning,
            stacklevel=2,
        )

    positive_counts = {
        "stages": args.hardness_stages,
        "repeats": args.hardness_repeats,
        "chains": args.hardness_chains,
        "steps": args.hardness_steps,
        "reference-min-chains": args.hardness_reference_min_chains,
    }
    invalid_counts = [name for name, value in positive_counts.items() if value <= 0]
    if invalid_counts:
        raise ValueError(
            "These hardness-map counts must be positive: "
            + ", ".join(invalid_counts)
            + "."
        )
    if args.hardness_chains < 2:
        raise ValueError("--chains must be at least two.")
    if args.hardness_reference_min_chains < 4:
        raise ValueError("--reference-min-chains must be at least four.")

    positive_scalars = {
        "reference-chain-factor": args.hardness_reference_chain_factor,
        "reference-max-step": args.hardness_reference_max_step,
        "reference-margin": args.hardness_reference_margin,
        "reference-time": args.hardness_reference_time,
        "reference-consistency-threshold": (
            args.hardness_reference_consistency_threshold
        ),
        "cooling-gamma": args.cooling_gamma,
        "delta": args.delta,
        "friction": args.friction,
        "step-size": args.step_size,
        "metric-floor": args.metric_floor,
    }
    invalid_scalars = [
        name
        for name, value in positive_scalars.items()
        if not np.isfinite(value) or value <= 0.0
    ]
    if invalid_scalars:
        raise ValueError(
            "These hardness-map values must be finite and positive: "
            + ", ".join(invalid_scalars)
            + "."
        )
    if args.cooling_gamma >= 1.0:
        raise ValueError("--cooling-gamma must lie in (0,1).")
    if not np.isfinite(args.covariance_ridge) or args.covariance_ridge < 0.0:
        raise ValueError("--covariance-ridge must be finite and nonnegative.")

    map_models = [
        make_lattice_model(
            side,
            HARDNESS_BETA,
            quartic,
            mass,
            HARDNESS_RADIUS,
            jnp.float32,
        )
        for side in args.hardness_sides
        for quartic in args.hardness_quartics
        for mass in args.hardness_masses
    ]
    maximum_smoothness = max(model.design_smoothness for model in map_models)
    stiffness_margin = args.step_size * np.sqrt(maximum_smoothness)
    if stiffness_margin > 0.5:
        warnings.warn(
            "The hardest map cell has "
            f"h*sqrt(L)={stiffness_margin:.3g}; reduce --step-size if "
            "method trajectories become unstable.",
            RuntimeWarning,
            stacklevel=2,
        )
    maximum_condition = max(
        model.global_smoothness_bound / model.strong_convexity for model in map_models
    )
    final_cooling_residual = (
        args.cooling_gamma**args.hardness_stages * maximum_condition
    )
    if not args.quick and final_cooling_residual > args.delta:
        warnings.warn(
            "The hardest map cell has "
            f"gamma_cool^K*kappa_H={final_cooling_residual:.3g}, above "
            f"delta={args.delta:g}. Increase --stages so the map does not "
            "mistake unfinished cooling for estimator failure.",
            RuntimeWarning,
            stacklevel=2,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the hardness map and save all publication outputs."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.quick and args.gpu:
        parser.error("--quick and --gpu are mutually exclusive presets.")
    explicit_destinations = _explicit_cli_destinations(parser, arguments)
    apply_quick_configuration(args, explicit_destinations)
    validate_arguments(args)

    backend_text = jax.default_backend()
    if args.gpu:
        backend_text = ", ".join(
            str(getattr(device, "device_kind", device))
            for device in _available_gpu_devices()
        )
    print(
        "Hardness map: horizontal axis=m, vertical axis=lambda; "
        f"beta={HARDNESS_BETA:g}, R={HARDNESS_RADIUS:g}; "
        f"sides={','.join(map(str, args.hardness_sides))}; "
        "D=side^2; "
        f"fixed n={args.hardness_chains}, "
        f"N={args.hardness_steps}, K={args.hardness_stages}; "
        f"backend={backend_text}",
        flush=True,
    )

    result = run_hardness_map(args)
    figures = make_hardness_map_figures(result, args)
    output_paths = save_publication_figures(figures, args.output)
    data_path = save_hardness_map_data(result, args.output, args)
    for figure in figures.values():
        plt.close(figure)

    print_hardness_map_summary(result)
    for plot_name, path in output_paths.items():
        print(f"Saved {plot_name.replace('_', ' ')} PDF: {path}")
    print(f"Saved hardness-map numerical data: {data_path}")


if __name__ == "__main__":
    main()
