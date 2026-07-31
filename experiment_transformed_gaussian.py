"""Experiment comparing covariance preconditioners on transformed Gaussians.

The target density is

    pi_B(x) propto exp(-U(B^{1/2} x)),    U(z) = ||z||^2 / 2.

Consequently ``pi_B = N(0, B^{-1})``.  The eigenvalues of ``B`` are
geometrically spaced in ``[1 / kappa, 1]`` and its eigenvectors are Haar
orthogonal. Gaussian cooling is compared at an equal ULMC budget with:

* one unpreconditioned ULMC run of length ``steps * stages``; and
* staged covariance adaptation on the uncooled target.

The figures report the scale-invariant relative condition number

    kappa(Sigma_hat^{-1/2} Sigma Sigma_hat^{-1/2}),

and stagewise convergence for the hardest transformation, including its exact
cooled-Gaussian oracle.  Each plot is written to its own vector PDF.

Here ``n`` is ``--chains``, ``N`` is ``--steps``, and ``K`` is ``--stages``.
Every method therefore uses the same ``n*N*K`` gradient-evaluation budget.
Sampler tuning and Monte Carlo effort are explicit command-line controls.
``--delta`` records the theoretical preconditioning tolerance but does not
otherwise alter a run whose sampler and stage counts are supplied directly.

Examples
--------
Fast deterministic smoke test:

    python experiment_transformed_gaussian.py --quick

Publication run with custom Monte Carlo effort:

    python experiment_transformed_gaussian.py \
        --chains 512 --steps 128 --stages 12 --repeats 12
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import random

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaussian_cooling_algs import (
    gaussian_cooling,
    sample_covariance,
    symmetric_matrix_sqrt,
    transformed_ulmc,
    ulmc,
)


Array = jax.Array
PotentialFn = Callable[[Array], Array]

METHODS = (
    "Gaussian cooling",
    "K-stage empirical preconditioning",
    "Unpreconditioned ULMC",
)
COLORS = {
    "Gaussian cooling": "#0072B2",
    "K-stage empirical preconditioning": "#D55E00",
    "Unpreconditioned ULMC": "#6B6B6B",
}
MARKERS = {
    "Gaussian cooling": "o",
    "K-stage empirical preconditioning": "s",
    "Unpreconditioned ULMC": "^",
}


@dataclass(frozen=True)
class GaussianTarget:
    """A fixed transformed Gaussian and its analytic covariance."""

    precision: Array
    covariance: np.ndarray
    potential: PotentialFn
    gradient: PotentialFn


@dataclass
class ExperimentResult:
    kappas: np.ndarray
    relative_conditions: dict[str, np.ndarray]
    convergence_conditions: dict[str, np.ndarray]
    exact_cooling_oracle: np.ndarray
    elapsed_seconds: float


def haar_orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    """Sample an orthogonal matrix from Haar measure using a signed QR."""

    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def make_target(
    dimension: int,
    condition_number: float,
    rng: np.random.Generator,
    dtype: jnp.dtype,
) -> GaussianTarget:
    """Construct ``B`` with spectrum ``[1/kappa, 1]`` and ``Sigma=B^-1``."""

    eigenvalues = np.geomspace(
        1.0 / condition_number,
        1.0,
        dimension,
    )
    eigenvectors = haar_orthogonal(rng, dimension)
    precision_np = (eigenvectors * eigenvalues) @ eigenvectors.T
    precision = jnp.asarray(precision_np, dtype=dtype)
    # Evaluate against the inverse of the matrix actually used by JAX.  This
    # distinction matters only for deliberately extreme float32 condition
    # numbers, where casting the nominal construction slightly moves B.
    covariance = np.linalg.inv(np.asarray(precision, dtype=np.float64))
    covariance = 0.5 * (covariance + covariance.T)

    def potential(x: Array) -> Array:
        return 0.5 * x @ precision @ x

    def gradient(x: Array) -> Array:
        return precision @ x

    return GaussianTarget(
        precision=precision,
        covariance=covariance,
        potential=potential,
        gradient=gradient,
    )


def _positive_definite(
    covariance: np.ndarray,
    relative_ridge: float,
) -> np.ndarray:
    """Symmetrize and add only the requested scale-relative numerical ridge."""

    covariance = 0.5 * (covariance + covariance.T)
    dimension = covariance.shape[0]
    scale = max(float(np.trace(covariance)) / dimension, 1.0)
    return covariance + relative_ridge * scale * np.eye(dimension)


def generalized_covariance_eigenvalues(
    estimated_covariance: np.ndarray,
    target_covariance: np.ndarray,
    relative_ridge: float,
) -> np.ndarray:
    """Eigenvalues of ``Sigma_hat^-1/2 Sigma Sigma_hat^-1/2``."""

    estimate = _positive_definite(estimated_covariance, relative_ridge)
    try:
        cholesky = np.linalg.cholesky(estimate)
    except np.linalg.LinAlgError as exc:
        smallest = float(np.linalg.eigvalsh(estimate)[0])
        raise ValueError(
            "The empirical covariance is not positive definite "
            f"(smallest eigenvalue {smallest:.3e}). Increase --chains or "
            "--metric-ridge."
        ) from exc

    inverse_cholesky = np.linalg.solve(
        cholesky,
        np.eye(cholesky.shape[0]),
    )
    whitened = inverse_cholesky @ target_covariance @ inverse_cholesky.T
    eigenvalues = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))
    if eigenvalues[0] <= 0.0:
        raise FloatingPointError("The generalized covariance spectrum is nonpositive.")
    return eigenvalues


def relative_condition_number(
    estimated_covariance: np.ndarray,
    target_covariance: np.ndarray,
    relative_ridge: float,
) -> float:
    eigenvalues = generalized_covariance_eigenvalues(
        estimated_covariance,
        target_covariance,
        relative_ridge,
    )
    return float(eigenvalues[-1] / eigenvalues[0])


def validate_target_and_metric(
    target: GaussianTarget,
    requested_condition: float,
    relative_ridge: float,
) -> None:
    """Check the analytic covariance and three exact metric invariants."""

    original_precision = np.asarray(target.precision)
    is_float32 = original_precision.dtype == np.float32
    precision = np.asarray(original_precision, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(precision)
    obtained_condition = float(eigenvalues[-1] / eigenvalues[0])
    condition_tolerance = 2e-4 if is_float32 else 2e-9
    if not np.isclose(
        obtained_condition,
        requested_condition,
        rtol=condition_tolerance,
    ):
        raise AssertionError(
            f"Constructed kappa(B)={obtained_condition:.12g}, expected "
            f"{requested_condition:.12g}."
        )
    inverse_error = float(
        np.linalg.norm(precision @ target.covariance - np.eye(len(precision)))
    )
    inverse_tolerance = 5e-2 if is_float32 else 1e-7
    if inverse_error > inverse_tolerance:
        raise AssertionError(
            "Analytic covariance inverse error is "
            f"{inverse_error:.3e}."
        )

    exact = relative_condition_number(
        target.covariance,
        target.covariance,
        relative_ridge,
    )
    rescaled = relative_condition_number(
        3.0 * target.covariance,
        target.covariance,
        relative_ridge,
    )
    identity = relative_condition_number(
        np.eye(len(precision)),
        target.covariance,
        relative_ridge,
    )
    if not np.isclose(exact, 1.0, rtol=2e-8):
        raise AssertionError(f"kappa_rel(Sigma) should be one, got {exact:.8g}.")
    if not np.isclose(rescaled, 1.0, rtol=2e-8):
        raise AssertionError(
            f"kappa_rel(c Sigma) should be one, got {rescaled:.8g}."
        )
    if not np.isclose(identity, requested_condition, rtol=2e-8):
        raise AssertionError(
            f"kappa_rel(I) should be {requested_condition:g}, got {identity:.8g}."
        )


def staged_target_covariance(
    key: Array,
    target: GaussianTarget,
    minimizer: Array,
    smoothness: float,
    friction: float,
    step_size: float,
    num_steps: int,
    num_chains: int,
    num_stages: int,
    covariance_ridge: float,
) -> Array:
    """Adapt the covariance in stages without Gaussian cooling."""

    dimension = minimizer.shape[0]
    dtype = minimizer.dtype
    covariance = jnp.eye(dimension, dtype=dtype) / smoothness
    ridge_identity = covariance_ridge * jnp.eye(dimension, dtype=dtype)

    for _ in range(num_stages):
        key, stage_key = random.split(key)
        covariance_sqrt, _ = symmetric_matrix_sqrt(
            covariance + ridge_identity,
        )
        samples = transformed_ulmc(
            stage_key,
            target.potential,
            target.gradient,
            minimizer,
            covariance_sqrt,
            friction,
            smoothness,
            step_size,
            num_steps,
            num_chains,
        )
        covariance = sample_covariance(samples)

    return 0.5 * (covariance + covariance.T)


def estimate_all_methods(
    key: Array,
    target: GaussianTarget,
    *,
    dimension: int,
    condition_number: float,
    cooling_gamma: float,
    delta: float,
    friction: float,
    step_size: float,
    num_steps: int,
    num_chains: int,
    num_stages: int,
    covariance_ridge: float,
    dtype: jnp.dtype,
) -> dict[str, np.ndarray]:
    """Run all three procedures with equal ``(chains, steps, stages)``."""

    minimizer = jnp.zeros(dimension, dtype=dtype)
    key_cooling, key_staged, key_plain = random.split(key, 3)

    cooling_covariance = gaussian_cooling(
        key_cooling,
        target.potential,
        target.gradient,
        minimizer,
        alpha=min(1.0 - 1e-12, 1.0 / condition_number),
        delta=delta,
        cooling_gamma=cooling_gamma,
        smoothness_L=1.0,
        num_stages=num_stages,
        num_chains=num_chains,
        num_ulmc_steps=num_steps,
        ulmc_step_size=step_size,
        ulmc_friction_gamma=friction,
        covariance_regularization=covariance_ridge,
    )

    staged_covariance = staged_target_covariance(
        key_staged,
        target,
        minimizer,
        smoothness=1.0,
        friction=friction,
        step_size=step_size,
        num_steps=num_steps,
        num_chains=num_chains,
        num_stages=num_stages,
        covariance_ridge=covariance_ridge,
    )

    # Baseline (a) uses the same number of chains and total ULMC transitions.
    plain_samples = ulmc(
        key_plain,
        target.potential,
        target.gradient,
        minimizer,
        friction,
        1.0,
        step_size,
        num_steps * num_stages,
        num_chains,
    )
    plain_covariance = sample_covariance(plain_samples)

    outputs = {
        "Gaussian cooling": cooling_covariance,
        "K-stage empirical preconditioning": staged_covariance,
        "Unpreconditioned ULMC": plain_covariance,
    }
    return {
        name: np.asarray(value.block_until_ready(), dtype=np.float64)
        for name, value in outputs.items()
    }


def run_experiment(args: argparse.Namespace) -> ExperimentResult:
    """Execute all target condition numbers and independent sampler repeats."""

    if args.chains <= args.dimension:
        raise ValueError(
            "--chains must exceed --dimension so the dense empirical "
            "covariances can be positive definite."
        )
    if args.covariance_ridge < 0.0 or args.metric_ridge < 0.0:
        raise ValueError("Covariance ridges must be nonnegative.")

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    kappas = np.sort(np.asarray(args.kappas, dtype=float))
    root_key = random.PRNGKey(args.seed)
    relative_conditions = {
        method: np.empty((len(kappas), args.repeats), dtype=float)
        for method in METHODS
    }
    hardest_target: GaussianTarget | None = None

    started = time.perf_counter()
    for kappa_index, kappa in enumerate(kappas):
        target_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, 1729, kappa_index])
        )
        target = make_target(args.dimension, float(kappa), target_rng, dtype)
        validate_target_and_metric(
            target,
            float(kappa),
            0.0,
        )
        if kappa_index == len(kappas) - 1:
            hardest_target = target
        print(
            f"kappa(B)={kappa:g}: "
            f"{args.repeats} repeats, d={args.dimension}, "
            f"chains={args.chains}, steps={args.steps}, stages={args.stages}",
            flush=True,
        )

        for repeat in range(args.repeats):
            run_key = random.fold_in(root_key, kappa_index * args.repeats + repeat)
            estimates = estimate_all_methods(
                run_key,
                target,
                dimension=args.dimension,
                condition_number=float(kappa),
                cooling_gamma=args.cooling_gamma,
                delta=args.delta,
                friction=args.friction,
                step_size=args.step_size,
                num_steps=args.steps,
                num_chains=args.chains,
                num_stages=args.stages,
                covariance_ridge=args.covariance_ridge,
                dtype=dtype,
            )

            for method, estimate in estimates.items():
                spectrum = generalized_covariance_eigenvalues(
                    estimate,
                    target.covariance,
                    args.metric_ridge,
                )
                relative_conditions[method][kappa_index, repeat] = (
                    spectrum[-1] / spectrum[0]
                )

    # A fixed-key prefix experiment exposes convergence versus cumulative
    # stage/gradient budget.  Reusing the key makes every prefix nested: the
    # first k stages or k*N plain transitions are identical across prefixes.
    assert hardest_target is not None
    hardest_kappa = float(kappas[-1])
    convergence_conditions = {
        method: np.empty(
            (args.stages + 1, args.convergence_repeats),
            dtype=float,
        )
        for method in METHODS
    }
    for method in METHODS:
        convergence_conditions[method][0, :] = hardest_kappa

    print(
        f"hardest-case convergence: stages 1..{args.stages}, "
        f"{args.convergence_repeats} nested repeat(s)",
        flush=True,
    )
    for repeat in range(args.convergence_repeats):
        convergence_key = random.fold_in(root_key, 1_000_000 + repeat)
        for prefix_stages in range(1, args.stages + 1):
            estimates = estimate_all_methods(
                convergence_key,
                hardest_target,
                dimension=args.dimension,
                condition_number=hardest_kappa,
                cooling_gamma=args.cooling_gamma,
                delta=args.delta,
                friction=args.friction,
                step_size=args.step_size,
                num_steps=args.steps,
                num_chains=args.chains,
                num_stages=prefix_stages,
                covariance_ridge=args.covariance_ridge,
                dtype=dtype,
            )
            for method, estimate in estimates.items():
                convergence_conditions[method][prefix_stages, repeat] = (
                    relative_condition_number(
                        estimate,
                        hardest_target.covariance,
                        args.metric_ridge,
                    )
                )

    stage_indices = np.arange(args.stages + 1)
    cooling_strengths = args.cooling_gamma**stage_indices
    exact_cooling_oracle = (
        1.0 + hardest_kappa * cooling_strengths
    ) / (1.0 + cooling_strengths)
    # Stage zero is the implementation's initial I/L preconditioner, not a
    # sampled gamma**0 cooled covariance.
    exact_cooling_oracle[0] = hardest_kappa

    elapsed = time.perf_counter() - started
    return ExperimentResult(
        kappas=kappas,
        relative_conditions=relative_conditions,
        convergence_conditions=convergence_conditions,
        exact_cooling_oracle=exact_cooling_oracle,
        elapsed_seconds=elapsed,
    )


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "figure.dpi": 140,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figures(
    result: ExperimentResult,
    *,
    dimension: int,
    num_chains: int,
    num_steps: int,
    num_stages: int,
    cooling_gamma: float,
    step_size: float,
    friction: float,
) -> dict[str, plt.Figure]:
    """Create standalone publication figures for both diagnostics."""

    _configure_plot_style()
    quality_figure, ax = plt.subplots(
        figsize=(4.7, 3.8),
        constrained_layout=True,
    )

    for method in METHODS:
        values = result.relative_conditions[method]
        median = np.median(values, axis=1)
        lower, upper = np.quantile(values, (0.25, 0.75), axis=1)
        ax.fill_between(
            result.kappas,
            lower,
            upper,
            color=COLORS[method],
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            result.kappas,
            median,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4.5,
            label=method,
        )
    ax.axhline(1.0, color="#222222", linewidth=0.9, linestyle=":", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Transformation condition number $\kappa(B)$")
    ax.set_ylabel(r"Relative condition number $\kappa_{\mathrm{rel}}$")
    ax.set_title(
        "Preconditioner quality under affine transformations\n"
        rf"$d={dimension}$, $n={num_chains}$, $N={num_steps}$, "
        rf"$K={num_stages}$"
        + "\n"
        + rf"$\gamma_{{\rm cool}}={cooling_gamma:g}$, "
        rf"$h={step_size:g}$, "
        rf"$\gamma_{{\rm fric}}={friction:g}$"
    )
    ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax.legend(frameon=False, loc="upper left")

    convergence_figure, ax = plt.subplots(
        figsize=(4.7, 3.8),
        constrained_layout=True,
    )
    stages = np.arange(num_stages + 1)
    for method in METHODS:
        values = result.convergence_conditions[method]
        median = np.median(values, axis=1)
        lower, upper = np.quantile(values, (0.25, 0.75), axis=1)
        ax.fill_between(
            stages,
            lower,
            upper,
            color=COLORS[method],
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(
            stages,
            median,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=3.7,
            label=method,
        )
    ax.plot(
        stages,
        result.exact_cooling_oracle,
        color="#222222",
        linestyle="--",
        linewidth=1.25,
        label="Exact Gaussian cooling (oracle)",
    )
    ax.axhline(1.0, color="#222222", linewidth=0.9, linestyle=":", zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative stages")
    ax.set_ylabel(r"Relative condition number $\kappa_{\mathrm{rel}}$")
    ax.set_title(
        "Hardest-case convergence\n"
        rf"$\kappa(B)={result.kappas[-1]:g}$, "
        rf"$d={dimension}$, $n={num_chains}$, $N={num_steps}$, "
        rf"$K={num_stages}$"
        + "\n"
        + rf"$\gamma_{{\rm cool}}={cooling_gamma:g}$, "
        rf"$h={step_size:g}$, "
        rf"$\gamma_{{\rm fric}}={friction:g}$"
    )
    ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=7.7)

    return {
        "preconditioner_quality": quality_figure,
        "stage_convergence": convergence_figure,
    }


def save_publication_figures(
    figures: dict[str, plt.Figure],
    output: Path,
) -> dict[str, Path]:
    """Save every plot as a separate vector PDF."""

    output = output.expanduser().resolve()
    stem = output.with_suffix("") if output.suffix else output
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for plot_name, figure in figures.items():
        path = stem.with_name(f"{stem.name}_{plot_name}").with_suffix(".pdf")
        figure.savefig(path, bbox_inches="tight")
        paths[plot_name] = path
    return paths


def _parse_positive_floats(value: str) -> list[float]:
    try:
        values = [
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated floating-point condition numbers."
        ) from exc
    if not values or any(
        not np.isfinite(item) or item < 1.0 for item in values
    ):
        raise argparse.ArgumentTypeError(
            "Expected finite comma-separated condition numbers, each at least one."
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Condition numbers must be unique.")
    return sorted(values)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gaussian cooling and equal-budget baselines on "
            "affinely transformed Gaussian targets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=20,
        help="Target dimension.",
    )
    parser.add_argument(
        "--kappas",
        type=_parse_positive_floats,
        default=[1.0, 10.0, 100.0, 1_000.0, 10_000.0],
        help="Comma-separated condition numbers for B.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help="Independent sampler repeats per condition number.",
    )
    parser.add_argument(
        "--convergence-repeats",
        type=int,
        default=3,
        help="Independent nested runs used for the hardest-case stage panel.",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=256,
        help="Independent chains n used at each stage.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=96,
        help="ULMC transitions N per stage.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=12,
        help="Number K of covariance-adaptation stages.",
    )
    parser.add_argument(
        "--cooling-gamma",
        type=float,
        default=0.05,
        help="Geometric cooling factor in (0, 1).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.25,
        help=(
            "Theoretical preconditioning tolerance retained as metadata; "
            "fixed sampler and stage counts control this experiment."
        ),
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
        default=0.06,
        help="ULMC integration step size.",
    )
    parser.add_argument(
        "--covariance-ridge",
        type=float,
        default=0.0,
        help="Optional ridge used inside covariance adaptation.",
    )
    parser.add_argument(
        "--metric-ridge",
        type=float,
        default=0.0,
        help="Optional ridge used only in the evaluation metric.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Floating-point precision.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=202407,
        help="Root random seed.",
    )
    parser.add_argument(
        "--output-prefix",
        "--output",
        dest="output",
        type=Path,
        default=script_dir / "figures" / "transformed_gaussian",
        help=(
            "Base output prefix; each plot name is appended and saved as a "
            "separate PDF."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use a small deterministic smoke-test preset. Explicit numerical "
            "options override the corresponding preset values."
        ),
    )
    return parser


def _explicit_cli_destinations(
    parser: argparse.ArgumentParser,
    arguments: Sequence[str],
) -> set[str]:
    """Return parser destinations explicitly present on the command line."""

    destinations: set[str] = set()
    for action in parser._actions:
        if any(
            argument == option
            or argument.startswith(f"{option}=")
            for argument in arguments
            for option in action.option_strings
        ):
            destinations.add(action.dest)
    return destinations


def apply_quick_configuration(
    args: argparse.Namespace,
    explicit_destinations: set[str] | None = None,
) -> None:
    """Apply the quick preset while preserving explicit CLI overrides."""

    if not args.quick:
        return
    explicitly_set = (
        set()
        if explicit_destinations is None
        else explicit_destinations
    )
    quick_values = {
        "dimension": 6,
        "kappas": [1.0, 30.0, 1_000.0],
        "repeats": 2,
        "convergence_repeats": 2,
        "chains": 192,
        "steps": 80,
        "stages": 8,
        "dtype": "float64",
    }
    for destination, value in quick_values.items():
        if destination not in explicitly_set:
            setattr(args, destination, value)


def validate_arguments(args: argparse.Namespace) -> None:
    counts = {
        "dimension": args.dimension,
        "repeats": args.repeats,
        "convergence_repeats": args.convergence_repeats,
        "chains": args.chains,
        "steps": args.steps,
        "stages": args.stages,
    }
    invalid = [name for name, value in counts.items() if value <= 0]
    if invalid:
        raise ValueError(f"These counts must be positive: {', '.join(invalid)}.")
    if args.chains <= args.dimension:
        raise ValueError(
            "--chains must exceed --dimension so the dense empirical "
            "covariances can be positive definite."
        )
    if (
        not np.isfinite(args.cooling_gamma)
        or not 0.0 < args.cooling_gamma < 1.0
    ):
        raise ValueError("--cooling-gamma must lie in (0, 1).")
    positive_scalars = {
        "delta": args.delta,
        "friction": args.friction,
        "step-size": args.step_size,
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
    nonnegative_scalars = {
        "covariance-ridge": args.covariance_ridge,
        "metric-ridge": args.metric_ridge,
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


def print_summary(result: ExperimentResult) -> None:
    column_widths = {
        method: max(29, len(method) + 2)
        for method in METHODS
    }
    header = "kappa(B)".ljust(12) + "".join(
        method.rjust(column_widths[method]) for method in METHODS
    )
    print("\nMedian relative condition number")
    print(header)
    for index, kappa in enumerate(result.kappas):
        row = f"{kappa:g}".ljust(12)
        for method in METHODS:
            median = np.median(result.relative_conditions[method][index])
            row += f"{median:{column_widths[method]}.4g}"
        print(row)
    print(
        "\nTotal wall time (including JIT compilation): "
        f"{result.elapsed_seconds:.2f}s"
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    apply_quick_configuration(
        args,
        _explicit_cli_destinations(parser, arguments),
    )
    validate_arguments(args)

    result = run_experiment(args)
    figures = make_figures(
        result,
        dimension=args.dimension,
        num_chains=args.chains,
        num_steps=args.steps,
        num_stages=args.stages,
        cooling_gamma=args.cooling_gamma,
        step_size=args.step_size,
        friction=args.friction,
    )
    output_paths = save_publication_figures(figures, args.output)
    for figure in figures.values():
        plt.close(figure)

    print_summary(result)
    for plot_name, path in output_paths.items():
        print(f"Saved {plot_name.replace('_', ' ')} PDF: {path}")


if __name__ == "__main__":
    main()
