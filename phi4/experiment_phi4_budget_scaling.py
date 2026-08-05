"""Chain-and-step budget heatmap for translation-invariant Gaussian cooling.

This focused experiment fixes one periodic lattice :math:`\phi^4` target and
varies the number of independent chains ``n`` and ULMC transitions per chain
and stage ``N`` over a Cartesian grid.  Every estimate is evaluated against
the same independently sampled, translation-invariant reference covariance.
The default axes are

``n = 64, 128, 256, 512, 1024`` and ``N = 32, 64, 128, 256, 512``

at lattice side length ``d=100``, giving 25 method configurations.  The
experiment writes one vector-PDF heatmap and a compressed NPZ file containing
all repeats, learned spectra, timings, and reference metadata.
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
    _fourier_cooling_call,
    make_lattice_model,
    reference_samples,
    spectral_relative_condition,
    validate_lattice_model,
)
from .phi4_cli import _available_gpu_devices, _explicit_cli_destinations
from .phi4_plotting import (
    _configure_plot_style,
    _logarithmic_cell_edges,
    save_publication_figures,
)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


DEFAULT_CHAIN_COUNTS = (64, 128, 256, 512, 1024)
DEFAULT_STEP_COUNTS = (32, 64, 128, 256, 512)
QUICK_CHAIN_COUNTS = (8, 16)
QUICK_STEP_COUNTS = (4, 8)


@dataclass
class BudgetScalingResult:
    """Preconditioner quality on a Cartesian ``N``-by-``n`` budget grid."""

    side: int
    chains: np.ndarray
    steps: np.ndarray
    relative_conditions: np.ndarray
    estimated_spectra: np.ndarray
    runtimes: np.ndarray
    reference_spectrum: np.ndarray
    reference_split_condition: float
    reference_chains: int
    reference_steps: int
    reference_step_size: float
    design_smoothness: float
    strong_convexity: float
    design_conditioning_alpha: float
    transformed_reference_smoothness: float
    elapsed_seconds: float


def _parse_integer_grid(value: str) -> tuple[int, ...]:
    """Parse a comma-separated grid of distinct positive integers."""

    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated positive integers, for example 64,128,256."
        ) from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Grid values must be positive integers.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Grid values must be unique.")
    return values


def _parse_budget_pairs(value: str) -> tuple[tuple[int, int], ...]:
    """Parse the former comma-separated ``chains:steps`` path syntax."""

    pairs: list[tuple[int, int]] = []
    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            fields = item.split(":")
            if len(fields) != 2:
                raise ValueError
            pairs.append((int(fields[0]), int(fields[1])))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected chains:steps pairs, for example 64:32,128:64."
        ) from exc
    if not pairs or any(chains < 2 or steps < 1 for chains, steps in pairs):
        raise argparse.ArgumentTypeError(
            "Each legacy budget pair requires at least two chains and one step."
        )
    if len(set(pairs)) != len(pairs):
        raise argparse.ArgumentTypeError("Legacy budget pairs must be unique.")
    return tuple(pairs)


def _reference_arguments(
    transformed_design_smoothness: float,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Resolve a stable reference step size and fixed physical run time."""

    reference_args = argparse.Namespace(**vars(args))
    stable_step_size = args.reference_stability_margin / np.sqrt(
        transformed_design_smoothness
    )
    if args.reference_step_size is None:
        reference_step_size = min(args.reference_max_step_size, stable_step_size)
    else:
        reference_step_size = args.reference_step_size
        if reference_step_size > stable_step_size:
            warnings.warn(
                "The requested reference step size exceeds the configured "
                "stability margin. Consider omitting --reference-step-size "
                "to use the automatic value.",
                RuntimeWarning,
                stacklevel=2,
            )
    reference_args.reference_step_size = reference_step_size
    reference_args.reference_steps = (
        int(args.reference_steps)
        if args.reference_steps is not None
        else max(1, int(np.ceil(args.reference_time / reference_step_size)))
    )
    return reference_args


def run_budget_scaling_experiment(
    args: argparse.Namespace,
) -> BudgetScalingResult:
    """Run cooling repeats at every budget against one shared reference."""

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

    chains = np.asarray(args.chain_counts, dtype=int)
    steps = np.asarray(args.step_counts, dtype=int)
    grid_shape = (len(steps), len(chains), args.repeats)
    relative_conditions = np.empty(grid_shape, dtype=float)
    estimated_spectra = np.empty(
        grid_shape + (model.dimension,),
        dtype=np.float64,
    )
    runtimes = np.empty_like(relative_conditions)
    root_key = random.fold_in(random.PRNGKey(args.seed), 510_000)
    reference_key = random.fold_in(root_key, 1_000)
    method_key = random.fold_in(root_key, 2_000)
    started = time.perf_counter()

    transformed_design_smoothness = (
        1.0 + 3.0 * model.quartic * model.design_radius**2 / model.mass
    )
    reference_args = _reference_arguments(transformed_design_smoothness, args)

    print(
        "Generating one shared independent reference: "
        f"n_ref={reference_args.reference_chains}, "
        f"N_ref={reference_args.reference_steps}, "
        f"h_ref={reference_args.reference_step_size:.4g}",
        flush=True,
    )
    reference = reference_samples(
        reference_key,
        model,
        reference_args,
    )
    reference.block_until_ready()
    reference_spectrum = np.asarray(
        sample_power_spectrum(reference, model.lattice_shape)
    )
    midpoint = reference.shape[0] // 2
    first_half_spectrum = np.asarray(
        sample_power_spectrum(reference[:midpoint], model.lattice_shape)
    )
    second_half_spectrum = np.asarray(
        sample_power_spectrum(reference[midpoint:], model.lattice_shape)
    )
    reference_split_condition = spectral_relative_condition(
        first_half_spectrum,
        second_half_spectrum,
        args.metric_floor,
    )

    num_cells = len(chains) * len(steps)
    for step_index, num_steps in enumerate(steps):
        for chain_index, num_chains in enumerate(chains):
            cell_args = argparse.Namespace(**vars(args))
            cell_args.chains = int(num_chains)
            cell_args.steps = int(num_steps)
            cell_key = random.fold_in(
                random.fold_in(method_key, int(num_chains)),
                int(num_steps),
            )
            repeat_keys = [
                random.fold_in(cell_key, repeat) for repeat in range(args.repeats)
            ]
            cell_index = step_index * len(chains) + chain_index
            print(
                f"Budget cell {cell_index + 1}/{num_cells}: "
                f"n={num_chains}, N={num_steps}, K={args.stages}",
                flush=True,
            )
            for repeat, key in enumerate(repeat_keys):
                repeat_started = time.perf_counter()
                spectrum = _fourier_cooling_call(
                    key,
                    model,
                    cell_args,
                )
                spectrum.block_until_ready()
                runtimes[step_index, chain_index, repeat] = (
                    time.perf_counter() - repeat_started
                )
                spectrum_array = np.asarray(spectrum)
                estimated_spectra[step_index, chain_index, repeat] = spectrum_array
                relative_conditions[step_index, chain_index, repeat] = (
                    spectral_relative_condition(
                        spectrum_array,
                        reference_spectrum,
                        args.metric_floor,
                    )
                )

    return BudgetScalingResult(
        side=args.side,
        chains=chains,
        steps=steps,
        relative_conditions=relative_conditions,
        estimated_spectra=estimated_spectra,
        runtimes=runtimes,
        reference_spectrum=reference_spectrum,
        reference_split_condition=reference_split_condition,
        reference_chains=int(reference_args.reference_chains),
        reference_steps=int(reference_args.reference_steps),
        reference_step_size=float(reference_args.reference_step_size),
        design_smoothness=float(model.design_smoothness),
        strong_convexity=float(model.strong_convexity),
        design_conditioning_alpha=float(model.design_conditioning_alpha),
        transformed_reference_smoothness=float(transformed_design_smoothness),
        elapsed_seconds=time.perf_counter() - started,
    )


def make_budget_scaling_figure(
    result: BudgetScalingResult,
    args: argparse.Namespace,
) -> dict[str, plt.Figure]:
    """Create the median relative-condition heatmap over the budget grid."""

    _configure_plot_style()
    figure, ax = plt.subplots(figsize=(6.5, 5.25), constrained_layout=True)
    medians = np.median(result.relative_conditions, axis=-1)
    if np.any(~np.isfinite(medians)) or np.any(medians <= 0.0):
        raise FloatingPointError(
            "Budget-grid relative condition numbers must be finite and positive."
        )
    maximum = float(np.max(medians))
    maximum = max(maximum, 1.01)
    normalization = LogNorm(vmin=1.0, vmax=maximum)
    image = ax.pcolormesh(
        _logarithmic_cell_edges(result.chains),
        _logarithmic_cell_edges(result.steps),
        medians,
        cmap="viridis",
        norm=normalization,
        shading="flat",
        edgecolors=(1.0, 1.0, 1.0, 0.65),
        linewidth=0.55,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(result.chains, [str(value) for value in result.chains])
    ax.set_yticks(result.steps, [str(value) for value in result.steps])
    ax.minorticks_off()
    ax.set_xlabel(r"Number of chains per stage $n$")
    ax.set_ylabel(r"ULMC steps per chain and stage $N$")
    ax.set_box_aspect(1.0)

    for step_index, num_steps in enumerate(result.steps):
        for chain_index, num_chains in enumerate(result.chains):
            value = medians[step_index, chain_index]
            red, green, blue, _ = image.cmap(normalization(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "white" if luminance < 0.5 else "#111111"
            ax.text(
                num_chains,
                num_steps,
                f"{value:.3g}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.4,
            )

    colorbar = figure.colorbar(image, ax=ax, shrink=0.88, pad=0.025)
    colorbar.set_label(r"Median relative condition number $\kappa_{\mathrm{rel}}$")
    radius_text = r"\infty" if np.isposinf(args.radius) else f"{args.radius:g}"
    repeat_label = "repeat" if args.repeats == 1 else "repeats"
    ax.set_title(
        "Translation-invariant Gaussian cooling: budget sensitivity\n"
        + rf"$d={result.side},\ D=d^2={result.side**2},\ "
        + rf"\beta={args.beta:g},\ \lambda={args.quartic:g},\ "
        + rf"m={args.mass:g},\ R={radius_text},\ K={args.stages}$; "
        + f"median of {args.repeats} {repeat_label}"
    )
    return {f"budget_heatmap_d{result.side}": figure}


def save_budget_scaling_data(
    result: BudgetScalingResult,
    output: Path,
    args: argparse.Namespace,
) -> Path:
    """Save all repeats and experimental metadata beside the PDF."""

    output = output.expanduser().resolve()
    stem = output.with_suffix("") if output.suffix else output
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = stem.with_name(f"{stem.name}_budget_grid_d{result.side}_data.npz")
    np.savez_compressed(
        path,
        side=np.asarray(result.side),
        dimension=np.asarray(result.side**2),
        chain_counts=result.chains,
        step_counts=result.steps,
        stages=np.asarray(args.stages),
        chain_transitions=(
            result.steps[:, None] * result.chains[None, :] * args.stages
        ),
        repeats=np.asarray(args.repeats),
        relative_conditions=result.relative_conditions,
        estimated_spectra=result.estimated_spectra,
        runtimes_seconds=result.runtimes,
        reference_spectrum=result.reference_spectrum,
        reference_split_condition=np.asarray(result.reference_split_condition),
        reference_chains=np.asarray(result.reference_chains),
        reference_steps=np.asarray(result.reference_steps),
        reference_step_size=np.asarray(result.reference_step_size),
        reference_requested_time=np.asarray(args.reference_time),
        reference_realized_time=np.asarray(
            result.reference_steps * result.reference_step_size
        ),
        reference_time_mode=np.asarray(
            "explicit_steps" if args.reference_steps is not None else "fixed_time"
        ),
        reference_max_step_size=np.asarray(args.reference_max_step_size),
        reference_stability_margin=np.asarray(args.reference_stability_margin),
        reference_consistency_threshold=np.asarray(
            args.reference_consistency_threshold
        ),
        design_smoothness=np.asarray(result.design_smoothness),
        stage_zero_spectrum=np.asarray(1.0 / result.design_smoothness),
        strong_convexity=np.asarray(result.strong_convexity),
        design_conditioning_alpha=np.asarray(result.design_conditioning_alpha),
        transformed_reference_smoothness=np.asarray(
            result.transformed_reference_smoothness
        ),
        beta=np.asarray(args.beta),
        quartic=np.asarray(args.quartic),
        mass=np.asarray(args.mass),
        radius=np.asarray(args.radius),
        cooling_design_radius=np.asarray(args.cooling_design_radius),
        cooling_gamma=np.asarray(args.cooling_gamma),
        delta=np.asarray(args.delta),
        friction=np.asarray(args.friction),
        step_size=np.asarray(args.step_size),
        covariance_ridge=np.asarray(args.covariance_ridge),
        metric_floor=np.asarray(args.metric_floor),
        dtype=np.asarray(args.dtype),
        seed=np.asarray(args.seed),
        method=np.asarray(COOLING_COMPARISON),
        budget_mode=np.asarray("cartesian_chain_step_grid"),
        grid_axis_order=np.asarray("step_counts,chain_counts,repeats"),
        relative_conditions_axis_order=np.asarray("step_counts,chain_counts,repeats"),
        runtimes_axis_order=np.asarray("step_counts,chain_counts,repeats"),
        estimated_spectra_axis_order=np.asarray(
            "step_counts,chain_counts,repeats,fourier_mode"
        ),
        chain_transitions_axis_order=np.asarray("step_counts,chain_counts"),
        runtime_first_repeat_includes_compilation=np.asarray(True),
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the Cartesian budget-grid CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure translation-invariant Gaussian-cooling quality while "
            "varying chains n and steps per chain and stage N over a grid."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--side", type=int, default=100)
    parser.add_argument(
        "--chain-counts",
        type=_parse_integer_grid,
        default=DEFAULT_CHAIN_COUNTS,
        metavar="C1,C2,...",
        help="Increasing chain counts n forming the horizontal heatmap axis.",
    )
    parser.add_argument(
        "--step-counts",
        type=_parse_integer_grid,
        default=DEFAULT_STEP_COUNTS,
        metavar="S1,S2,...",
        help="Increasing ULMC step counts N forming the vertical heatmap axis.",
    )
    parser.add_argument(
        "--budgets",
        type=_parse_budget_pairs,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--quartic", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=0.01)
    parser.add_argument(
        "--radius",
        type=float,
        default=4.0,
        help="Quadratic-continuation radius R; use inf for a genuine quartic.",
    )
    parser.add_argument("--cooling-design-radius", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--stages", type=int, default=12)
    parser.add_argument("--cooling-gamma", type=float, default=0.35)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--covariance-ridge", type=float, default=0.0)
    parser.add_argument("--metric-floor", type=float, default=1e-10)
    parser.add_argument("--reference-chains", type=int, default=4096)
    parser.add_argument(
        "--reference-steps",
        type=int,
        default=None,
        help="Reference transitions; by default derive these from its run time.",
    )
    parser.add_argument(
        "--reference-time",
        type=float,
        default=5.0,
        help="Reference integration time when --reference-steps is omitted.",
    )
    parser.add_argument(
        "--reference-step-size",
        type=float,
        default=None,
        help="By default choose a stable value from the transformed curvature.",
    )
    parser.add_argument("--reference-max-step-size", type=float, default=0.01)
    parser.add_argument("--reference-stability-margin", type=float, default=0.15)
    parser.add_argument(
        "--reference-consistency-threshold",
        type=float,
        default=1.5,
        help="Warn when the two reference halves disagree beyond this value.",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--output-prefix",
        "--output",
        dest="output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1] / "figures" / "phi4_budget_scaling"
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Require a visible JAX GPU; scientific settings are unchanged.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small deterministic CPU smoke-test preset.",
    )
    return parser


def apply_presets(args: argparse.Namespace, explicit: set[str]) -> None:
    """Apply the quick preset without overriding explicit options."""

    if args.quick and args.gpu:
        raise ValueError("--quick and --gpu are mutually exclusive.")
    if not args.quick:
        return
    quick_values: dict[str, object] = {
        "side": 4,
        "chain_counts": QUICK_CHAIN_COUNTS,
        "step_counts": QUICK_STEP_COUNTS,
        "repeats": 1,
        "stages": 2,
        "reference_chains": 64,
        "reference_steps": 32,
        "dtype": "float32",
    }
    for destination, value in quick_values.items():
        if destination not in explicit:
            setattr(args, destination, value)


def apply_legacy_budget_grid(args: argparse.Namespace, explicit: set[str]) -> None:
    """Map the former diagonal-path syntax to a Cartesian grid with a warning."""

    if args.budgets is None:
        return
    if "chain_counts" in explicit or "step_counts" in explicit:
        raise ValueError(
            "Do not combine deprecated --budgets with --chain-counts or "
            "--step-counts."
        )
    args.chain_counts = tuple(sorted({pair[0] for pair in args.budgets}))
    args.step_counts = tuple(sorted({pair[1] for pair in args.budgets}))
    warnings.warn(
        "--budgets is deprecated. Its unique chain and step coordinates now "
        "define a Cartesian grid, not a diagonal path; use --chain-counts "
        "and --step-counts instead.",
        FutureWarning,
        stacklevel=2,
    )


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate target, budget-grid, sampler, and execution settings."""

    if args.side < 2:
        raise ValueError("--side must be at least two.")
    chains = np.asarray(args.chain_counts, dtype=int)
    steps = np.asarray(args.step_counts, dtype=int)
    if len(chains) == 0 or len(steps) == 0:
        raise ValueError("The chain-count and step-count grids cannot be empty.")
    if len(set(chains)) != len(chains) or len(set(steps)) != len(steps):
        raise ValueError("Chain-count and step-count grid values must be unique.")
    if np.any(chains < 2) or np.any(steps < 1):
        raise ValueError("The grid requires every n >= 2 and every N >= 1.")
    if len(chains) > 1 and np.any(np.diff(chains) <= 0):
        raise ValueError("--chain-counts must be strictly increasing.")
    if len(steps) > 1 and np.any(np.diff(steps) <= 0):
        raise ValueError("--step-counts must be strictly increasing.")
    positive = {
        "quartic": args.quartic,
        "mass": args.mass,
        "cooling_design_radius": args.cooling_design_radius,
        "delta": args.delta,
        "friction": args.friction,
        "step_size": args.step_size,
        "metric_floor": args.metric_floor,
        "reference_time": args.reference_time,
        "reference_max_step_size": args.reference_max_step_size,
        "reference_stability_margin": args.reference_stability_margin,
        "reference_consistency_threshold": args.reference_consistency_threshold,
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
    if not (np.isfinite(args.radius) or np.isposinf(args.radius)) or args.radius <= 0:
        raise ValueError("--radius must be positive or inf.")
    if not 0.0 < args.cooling_gamma < 1.0:
        raise ValueError("--cooling-gamma must lie in (0, 1).")
    if not np.isfinite(args.covariance_ridge) or args.covariance_ridge < 0.0:
        raise ValueError("--covariance-ridge must be finite and nonnegative.")
    integers = (args.repeats, args.stages, args.reference_chains)
    if any(value < 1 for value in integers) or args.reference_chains < 4:
        raise ValueError(
            "Repeats and stages must be positive; reference chains must be "
            "at least four."
        )
    if args.reference_steps is not None and args.reference_steps < 1:
        raise ValueError("--reference-steps must be positive when supplied.")
    if args.reference_step_size is not None and (
        not np.isfinite(args.reference_step_size) or args.reference_step_size <= 0.0
    ):
        raise ValueError("--reference-step-size must be finite and positive.")
    if np.isposinf(args.radius):
        warnings.warn(
            "R=inf uses --cooling-design-radius only as an operational "
            "curvature scale; it is not a global Hessian bound.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.gpu and args.dtype == "float64":
        warnings.warn(
            "Float64 approximately doubles device memory relative to float32.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.gpu and not _available_gpu_devices():
        raise RuntimeError("--gpu requires a visible JAX GPU device.")


def print_summary(
    result: BudgetScalingResult,
    args: argparse.Namespace,
) -> None:
    """Print quality, runtime, reference, and trend summaries."""

    print("\nMedian relative condition numbers (rows N, columns n)")
    medians = np.median(result.relative_conditions, axis=-1)
    header = "N \\ n".rjust(10) + "".join(
        f"{num_chains:>12d}" for num_chains in result.chains
    )
    print(header)
    for step_index, num_steps in enumerate(result.steps):
        print(
            f"{num_steps:>10d}"
            + "".join(f"{value:>12.4g}" for value in medians[step_index])
        )

    improving_chains = int(np.count_nonzero(np.diff(medians, axis=1) < 0.0))
    chain_comparisons = medians.shape[0] * max(medians.shape[1] - 1, 0)
    improving_steps = int(np.count_nonzero(np.diff(medians, axis=0) < 0.0))
    step_comparisons = max(medians.shape[0] - 1, 0) * medians.shape[1]
    print(
        "Improving adjacent medians when increasing n at fixed N: "
        f"{improving_chains}/{chain_comparisons}."
    )
    print(
        "Improving adjacent medians when increasing N at fixed n: "
        f"{improving_steps}/{step_comparisons}."
    )
    best_step, best_chain = np.unravel_index(np.argmin(medians), medians.shape)
    worst_step, worst_chain = np.unravel_index(np.argmax(medians), medians.shape)
    print(
        "Best median cell: "
        f"n={result.chains[best_chain]}, N={result.steps[best_step]}, "
        f"kappa_rel={medians[best_step, best_chain]:.4g}."
    )
    print(
        "Worst median cell: "
        f"n={result.chains[worst_chain]}, N={result.steps[worst_step]}, "
        f"kappa_rel={medians[worst_step, worst_chain]:.4g}."
    )
    print(
        "Half-reference consistency kappa_rel: "
        f"{result.reference_split_condition:.4g}"
    )
    print(
        f"Reference: n_ref={result.reference_chains}, "
        f"N_ref={result.reference_steps}, h_ref={result.reference_step_size:.4g}, "
        f"T_ref={result.reference_steps * result.reference_step_size:.4g}. "
        "The first retained repeat at each grid cell includes JIT compilation."
    )
    print(f"Wall time including compilation: {result.elapsed_seconds:.2f}s")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    explicit = _explicit_cli_destinations(parser, arguments)
    try:
        apply_presets(args, explicit)
        apply_legacy_budget_grid(args, explicit)
        validate_arguments(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    result = run_budget_scaling_experiment(args)
    if (
        not args.quick
        and result.reference_split_condition > args.reference_consistency_threshold
    ):
        warnings.warn(
            "The two halves of the independent reference have relative "
            f"condition {result.reference_split_condition:.3g}, above "
            f"{args.reference_consistency_threshold:g}. Increase "
            "--reference-chains or --reference-steps before interpreting "
            "small method differences.",
            RuntimeWarning,
            stacklevel=2,
        )
    figures = make_budget_scaling_figure(result, args)
    paths = save_publication_figures(figures, args.output)
    data_path = save_budget_scaling_data(result, args.output, args)
    for figure in figures.values():
        plt.close(figure)
    print_summary(result, args)
    for name, path in paths.items():
        print(f"Saved {name.replace('_', ' ')} PDF: {path}")
    print(f"Saved numerical data: {data_path}")


if __name__ == "__main__":
    main()
