"""Composable command-line helpers for lattice phi4 experiments."""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path
from typing import Sequence

import jax
import numpy as np


def _parse_sides(value: str) -> list[int]:
    sides = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sides or any(side < 2 for side in sides):
        raise argparse.ArgumentTypeError(
            "Expected comma-separated lattice side lengths, each at least two."
        )
    if len(set(sides)) != len(sides):
        raise argparse.ArgumentTypeError("Lattice side lengths must be unique.")
    return sorted(sides)


def _parse_positive_values(value: str) -> list[float]:
    """Parse a sorted CSV of distinct, finite, positive floats."""

    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated floating-point values."
        ) from exc
    if not values or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("Sweep values must be finite and positive.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Sweep values must be unique.")
    return sorted(values)


def _parse_target_radii(value: str) -> list[float]:
    """Parse distinct positive radii, allowing positive infinity."""

    try:
        radii = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated radii; use 'inf' for the genuine "
            "quartic target."
        ) from exc
    if not radii or any(
        item <= 0.0 or np.isnan(item) or np.isneginf(item) for item in radii
    ):
        raise argparse.ArgumentTypeError(
            "Target radii must be positive finite values or 'inf'."
        )
    if len(set(radii)) != len(radii):
        raise argparse.ArgumentTypeError("Target radii must be unique.")
    return sorted(radii)


def add_target_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_sides: bool = True,
) -> None:
    """Add the physical lattice-target options shared by experiments."""

    if include_sides:
        parser.add_argument(
            "--sides",
            type=_parse_sides,
            default=[5, 10, 20, 50, 100],
            help=("Comma-separated lattice side lengths for the experiment."),
        )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Nearest-neighbor Laplacian coupling.",
    )
    parser.add_argument(
        "--quartic",
        type=float,
        default=0.5,
        help="Quartic coupling lambda.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=0.25,
        help="Mass m.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help=(
            "Quadratic-continuation radius R; use 'inf' for the genuine "
            "quartic target."
        ),
    )
    parser.add_argument(
        "--cooling-design-radius",
        type=float,
        default=4.0,
        help=(
            "Finite operational curvature radius used for R=inf and the "
            "radius sweep; it does not truncate the target."
        ),
    )


def add_sampler_arguments(parser: argparse.ArgumentParser) -> None:
    """Add method budgets, ULMC tuning, and condition-metric options."""

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Independent repeats for each preconditioner estimate.",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=512,
        help="Independent chains n used at every stage.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="ULMC transitions N per stage.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=8,
        help="Stages K.",
    )
    parser.add_argument(
        "--cooling-gamma",
        type=float,
        default=0.35,
        help="Geometric cooling factor in (0, 1).",
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
        help="Optional ridge used inside adaptive covariance updates.",
    )
    parser.add_argument(
        "--metric-floor",
        type=float,
        default=1e-10,
        help="Positive spectral floor used in condition metrics.",
    )
    parser.add_argument(
        "--metric-ridge",
        type=float,
        default=0.0,
        help="Optional dense evaluation-metric ridge.",
    )
    parser.add_argument(
        "--dense-max-side",
        type=int,
        default=20,
        help=(
            "Largest side at which a full D-by-D covariance is formed; "
            "use 0 to disable dense paths."
        ),
    )


def add_reference_arguments(parser: argparse.ArgumentParser) -> None:
    """Add independently preconditioned reference-run options."""

    parser.add_argument(
        "--reference-chains",
        type=int,
        default=256,
        help="Independent chains used for each reference estimate.",
    )
    parser.add_argument(
        "--reference-steps",
        type=int,
        default=256,
        help="ULMC transitions used for each reference estimate.",
    )
    parser.add_argument(
        "--reference-step-size",
        type=float,
        default=None,
        help=("Reference ULMC step size; by default reuse --step-size."),
    )


def add_execution_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output: Path,
) -> None:
    """Add precision, seed, output, and execution-preset options."""

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
        default=default_output,
        help="Base prefix used for separate PDF outputs.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Require a JAX GPU backend and apply the GPU preset.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small deterministic smoke-test preset.",
    )


def _explicit_cli_destinations(
    parser: argparse.ArgumentParser,
    arguments: Sequence[str],
) -> set[str]:
    """Return parser destinations explicitly present on the command line."""

    destinations: set[str] = set()
    for action in parser._actions:
        if any(
            argument == option or argument.startswith(f"{option}=")
            for argument in arguments
            for option in action.option_strings
        ):
            destinations.add(action.dest)
    return destinations


def _apply_present_preset_values(
    args: argparse.Namespace,
    values: dict[str, object],
    explicit_destinations: set[str],
) -> None:
    """Set non-explicit preset values that exist on a focused parser."""

    for destination, value in values.items():
        if destination not in explicit_destinations and hasattr(args, destination):
            setattr(args, destination, value)


def apply_quick_configuration(
    args: argparse.Namespace,
    explicit_destinations: set[str] | None = None,
) -> None:
    """Apply the quick preset while preserving explicit CLI overrides."""

    if not getattr(args, "quick", False):
        return
    explicitly_set = set() if explicit_destinations is None else explicit_destinations
    quick_values: dict[str, object] = {
        "repeats": 1,
        "chains": 64,
        "steps": 20,
        "stages": 4,
        "dense_max_side": 8,
        "reference_chains": 96,
        "reference_steps": 48,
        "trajectory_burnin": 100,
        "trajectory_samples": 500,
        "dtype": "float32",
    }
    _apply_present_preset_values(args, quick_values, explicitly_set)
    if (
        hasattr(args, "cooling_design_radius")
        and "cooling_design_radius" not in explicitly_set
    ):
        args.cooling_design_radius = 2.0


def apply_gpu_configuration(
    args: argparse.Namespace,
    explicit_destinations: set[str] | None = None,
) -> None:
    """Apply the large-lattice GPU preset without overriding CLI choices."""

    if not getattr(args, "gpu", False):
        return
    explicitly_set = set() if explicit_destinations is None else explicit_destinations
    gpu_values: dict[str, object] = {
        "repeats": 1,
        "chains": 64,
        "steps": 32,
        "stages": 8,
        "reference_chains": 128,
        "reference_steps": 128,
        "dense_max_side": 0,
        "dtype": "float32",
    }
    _apply_present_preset_values(args, gpu_values, explicitly_set)


def _available_gpu_devices() -> list[object]:
    """Return JAX GPU devices, with a focused backend error."""

    try:
        devices = jax.devices()
    except RuntimeError as exc:
        raise RuntimeError(
            "JAX could not initialize its device backend while validating "
            "--gpu. Check the CUDA-enabled JAX installation and driver."
        ) from exc
    gpu_platforms = {"gpu", "cuda", "rocm"}
    return [
        device for device in devices if str(device.platform).lower() in gpu_platforms
    ]


def _validate_gpu_request(args: argparse.Namespace) -> None:
    if getattr(args, "gpu", False) and not _available_gpu_devices():
        raise RuntimeError(
            "--gpu requires a JAX GPU backend, but no GPU device was found. "
            "Install CUDA-enabled JAX and ensure the pod exposes its GPU."
        )
    if getattr(args, "gpu", False) and getattr(args, "dtype", "float32") == "float64":
        warnings.warn(
            "The GPU preset is sized for float32. Float64 approximately "
            "doubles real state storage; establish the memory margin first.",
            RuntimeWarning,
            stacklevel=3,
        )


def validate_common_arguments(args: argparse.Namespace) -> None:
    """Validate target, method, reference, and execution options."""

    _validate_gpu_request(args)
    count_names = (
        "repeats",
        "chains",
        "steps",
        "stages",
        "reference_chains",
        "reference_steps",
    )
    invalid_counts = [
        name for name in count_names if hasattr(args, name) and getattr(args, name) <= 0
    ]
    if invalid_counts:
        raise ValueError(
            "These counts must be positive: " + ", ".join(invalid_counts) + "."
        )
    if hasattr(args, "chains") and args.chains < 2:
        raise ValueError("--chains must be at least two.")
    if hasattr(args, "reference_chains") and args.reference_chains < 4:
        raise ValueError("--reference-chains must be at least four.")

    if hasattr(args, "cooling_gamma") and (
        not np.isfinite(args.cooling_gamma) or not 0.0 < args.cooling_gamma < 1.0
    ):
        raise ValueError("--cooling-gamma must lie in (0, 1).")
    positive_names = (
        "quartic",
        "mass",
        "cooling_design_radius",
        "delta",
        "friction",
        "step_size",
        "metric_floor",
    )
    invalid_positive = [
        name.replace("_", "-")
        for name in positive_names
        if hasattr(args, name)
        and (not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0)
    ]
    if invalid_positive:
        raise ValueError(
            "These values must be finite and positive: "
            + ", ".join(invalid_positive)
            + "."
        )
    if hasattr(args, "radius") and (
        args.radius <= 0.0 or np.isnan(args.radius) or np.isneginf(args.radius)
    ):
        raise ValueError(
            "--radius must be positive and finite, or 'inf' for the genuine "
            "quartic target."
        )
    if (
        hasattr(args, "reference_step_size")
        and args.reference_step_size is not None
        and (
            not np.isfinite(args.reference_step_size) or args.reference_step_size <= 0.0
        )
    ):
        raise ValueError("--reference-step-size must be finite and positive.")
    nonnegative_names = (
        "beta",
        "covariance_ridge",
        "metric_ridge",
    )
    invalid_nonnegative = [
        name.replace("_", "-")
        for name in nonnegative_names
        if hasattr(args, name)
        and (not np.isfinite(getattr(args, name)) or getattr(args, name) < 0.0)
    ]
    if invalid_nonnegative:
        raise ValueError(
            "These values must be finite and nonnegative: "
            + ", ".join(invalid_nonnegative)
            + "."
        )
    if hasattr(args, "dense_max_side") and args.dense_max_side < 0:
        raise ValueError("--dense-max-side must be nonnegative.")
    if hasattr(args, "radius") and np.isposinf(args.radius):
        warnings.warn(
            "R=inf selects the genuine quartic target. Its Hessian is "
            "unbounded, so --cooling-design-radius supplies only an "
            "operational stage-zero/step-size scale; finite-global-L "
            "Gaussian-cooling guarantees do not apply.",
            RuntimeWarning,
            stacklevel=2,
        )


def validate_trajectory_arguments(
    args: argparse.Namespace,
    *,
    emcee_available: bool | None = None,
) -> None:
    """Validate trajectory sizes, side selection, and emcee availability."""

    count_names = (
        "trajectory_samples",
        "iat_tolerance",
        "trajectory_chunk_size",
        "iat_batch_size",
    )
    invalid_counts = [
        name for name in count_names if hasattr(args, name) and getattr(args, name) <= 0
    ]
    if invalid_counts:
        raise ValueError(
            "These trajectory counts must be positive: "
            + ", ".join(invalid_counts)
            + "."
        )
    if hasattr(args, "trajectory_burnin") and args.trajectory_burnin < 0:
        raise ValueError("--trajectory-burnin must be nonnegative.")
    if hasattr(args, "trajectory_samples") and args.trajectory_samples < 4:
        raise ValueError(
            "--trajectory-samples must be at least four so the final-half "
            "variance and IAT are defined."
        )
    for destination in ("diagnostic_side", "correlator_side"):
        if not hasattr(args, destination):
            continue
        side = getattr(args, destination)
        if side is None:
            continue
        option = "--" + destination.replace("_", "-")
        if side < 2:
            raise ValueError(f"{option} must be at least two.")
        if hasattr(args, "sides") and side not in args.sides:
            raise ValueError(f"{option} must be included in --sides; got d={side}.")

    if emcee_available is None:
        emcee_available = importlib.util.find_spec("emcee") is not None
    if not emcee_available:
        raise RuntimeError(
            "Autocorrelation diagnostics and the center-correlator "
            "experiment require `emcee`; install it with "
            "`python -m pip install emcee`."
        )


__all__ = [
    "add_target_arguments",
    "add_sampler_arguments",
    "add_reference_arguments",
    "add_execution_arguments",
    "apply_quick_configuration",
    "apply_gpu_configuration",
    "validate_common_arguments",
    "validate_trajectory_arguments",
]
