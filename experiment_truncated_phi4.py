"""Experiment comparing preconditioners on lattice phi^4 targets.

The target is defined on a two-dimensional periodic square lattice. The
script compares four primary estimators: translation-invariant Gaussian
cooling, translation-invariant empirical preconditioning, translation
averaging applied only after unpreconditioned ULMC, and the full empirical
covariance from the same unpreconditioned endpoints.  Dense Gaussian cooling
is retained as a separate ancillary comparison at feasible dimensions.  The
sample count ``n``, ULMC step count ``N``, and stage count ``K`` determine
equal gradient budgets across the four primary methods.

The non-Gaussian target covariance is not analytic. The relative condition
metric is therefore evaluated against an independent, longer reference run
preconditioned by the known quadratic operator
``Delta_beta + m I``.  Translation averaging produces a reference covariance
spectrum without materializing a dense matrix.

Dense Gaussian cooling is deliberately omitted when either:

* ``side > --dense-max-side``; or
* ``n <= side**2``, which makes its empirical covariance rank deficient.

In particular, a literal dense run at side 100 would require multiple
10,000-by-10,000 matrices and an O(10^12) eigensquare-root. The
translation-invariant method keeps only a length-10,000 spectrum and runs at
every requested side length. At side lengths 10 and 100, a separate stage
plot places all four methods on the same axes. The two unpreconditioned
curves use the same continuous chains after ``k * N`` transitions; one
projects their empirical covariance onto the translation-invariant Fourier
diagonal and the other retains the full covariance. The latter is reported
as rank deficient whenever ``n <= d**2``.

The production run also performs controlled one-factor sweeps of ``lambda``,
``m``, and ``R`` at a feasible fixed lattice side.  The radius comparison
includes ``R=inf``, which is the genuine (untruncated) quartic target, and
uses a common finite operational curvature radius for all of its targets.
This keeps stage zero and the cooling schedule comparable.  For the genuine
quartic this operational radius is a tuning scale, not a global Hessian
bound: the quartic Hessian is unbounded and the finite-smoothness theory does
not apply.  Each sweep is saved as its own vector PDF.

The script exposes step size, friction, sample count, transition count, and
stage count directly. ``--delta`` records the theoretical preconditioning
tolerance but does not otherwise alter a fixed-budget recurrence.

The independently preconditioned reference target uses the operational
curvature scale ``1 + 3 * lambda * R_design**2 / m``.  For small-mass sweeps,
``--reference-step-size`` can therefore be reduced without changing the
step size or physical budget of any method being compared.

The default physical parameters ``beta=2``, ``lambda=0.5``, ``m=0.25``, and
``R=2`` give a uniform raw Hessian-condition bound of 89, compared with 12
for the former all-ones defaults.  The default step size is reduced to 0.03
to preserve a similar dimensionless curvature margin.

After cooling the selected diagnostic lattice, a serial preconditioned ULMC
trajectory is used for two sampling diagnostics. The largest requested side
is used ordinarily. The ``--gpu`` scaling preset skips these diagnostics by
default while extending the scalable comparisons through side 1024; an
explicit opt-in uses side 64. ``emcee`` estimates the integrated
autocorrelation time of every entry in the first sample-covariance row, using
``g_x(t) = (phi_t(0) - phi_bar(0)) (phi_t(x) - phi_bar(x))``.
The resulting plot contains every lattice site and displays mixing versus
periodic distance from the origin.  Ergodicity under the exact
``phi -> -phi`` symmetry is checked from every site's mean over the final
half of the post-burn-in samples.  Site-field IATs turn those means into
Monte Carlo standard errors, effective sample sizes, and standardized mean
diagnostics.  Every plot is saved to its own vector PDF.

Examples
--------
Fast end-to-end smoke test:

    python experiment_truncated_phi4.py --quick

Production defaults:

    python experiment_truncated_phi4.py

H100-scale defaults through side 1024:

    python experiment_truncated_phi4.py --gpu
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
from matplotlib.ticker import MaxNLocator, NullFormatter

try:
    from emcee.autocorr import AutocorrError, integrated_time
except ImportError:  # pragma: no cover - exercised only without an optional dep.
    AutocorrError = RuntimeError
    integrated_time = None

from gaussian_cooling_algs import (
    apply_translation_invariant_spectrum,
    gaussian_cooling,
    sample_covariance,
    sample_power_spectrum,
    translation_invariant_covariance,
    translation_invariant_gaussian_cooling,
    ulmc,
    ulmc_coefficients,
)


Array = jax.Array
PotentialFn = Callable[[Array], Array]

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#6B6B6B"

GPU_LATTICE_SIDES = (64, 128, 256, 512, 1024)
GPU_DIAGNOSTIC_MAX_SIDE = 100

COOLING_COMPARISON = "Translation-invariant Gaussian cooling"
EMPIRICAL_COMPARISON = (
    "Translation-invariant empirical preconditioning (no cooling)"
)
TRANSLATION_AVERAGED_ULMC = "Translation-averaged unpreconditioned ULMC"
RAW_ULMC = "Full empirical covariance from unpreconditioned ULMC"

FOUR_METHODS = (
    COOLING_COMPARISON,
    EMPIRICAL_COMPARISON,
    TRANSLATION_AVERAGED_ULMC,
    RAW_ULMC,
)
METHOD_STYLES = {
    COOLING_COMPARISON: (BLUE, "o", "-"),
    EMPIRICAL_COMPARISON: (ORANGE, "s", "--"),
    TRANSLATION_AVERAGED_ULMC: (GREEN, "^", "-."),
    RAW_ULMC: (GRAY, "X", ":"),
}
PLOT_LABELS = {
    COOLING_COMPARISON: "Translation-invariant Gaussian cooling",
    EMPIRICAL_COMPARISON: (
        "Translation-invariant empirical preconditioning"
    ),
    TRANSLATION_AVERAGED_ULMC: "Translation-averaged ULMC covariance",
    RAW_ULMC: "Full empirical ULMC covariance",
}


@dataclass(frozen=True)
class LatticeModel:
    """Periodic lattice phi^4 target and its algorithmic tuning scales.

    ``radius`` is the physical continuation radius; positive infinity means
    the genuine quartic target. ``design_radius`` determines the finite
    stage-zero covariance and fixed-step ULMC tuning. For a truncated target,
    ``design_radius >= radius`` gives a valid (possibly conservative) global
    smoothness bound. For the genuine quartic it is only an operational scale.
    """

    side: int
    beta: float
    quartic: float
    mass: float
    radius: float
    design_radius: float
    dtype: jnp.dtype
    potential: PotentialFn
    gradient: PotentialFn
    laplacian_spectrum: np.ndarray
    design_smoothness: float
    global_smoothness_bound: float
    strong_convexity: float
    design_conditioning_alpha: float

    @property
    def dimension(self) -> int:
        return self.side * self.side

    @property
    def lattice_shape(self) -> tuple[int, int]:
        return (self.side, self.side)

    @property
    def is_truncated(self) -> bool:
        return bool(np.isfinite(self.radius))


@dataclass
class LatticeResult:
    sides: np.ndarray
    relative_condition_fourier: np.ndarray
    relative_condition_dense: np.ndarray
    relative_condition_empirical: np.ndarray
    relative_condition_translation_averaged_ulmc: np.ndarray
    relative_condition_raw_ulmc: np.ndarray
    comparison_conditions: dict[int, dict[str, np.ndarray]]
    runtime_fourier: np.ndarray
    runtime_dense: np.ndarray
    runtime_empirical: np.ndarray
    runtime_translation_averaged_ulmc: np.ndarray
    runtime_raw_ulmc: np.ndarray
    dense_skip_reasons: dict[int, str]
    raw_ulmc_skip_reasons: dict[int, str]
    diagnostic_side: int
    diagnostic_distances: np.ndarray
    two_point_correlator_row: np.ndarray
    autocorrelation_times: np.ndarray
    autocorrelation_reliable: np.ndarray
    final_half_site_means: np.ndarray
    site_field_autocorrelation_times: np.ndarray
    site_field_autocorrelation_reliable: np.ndarray
    site_mean_standard_errors: np.ndarray
    site_mean_effective_sample_sizes: np.ndarray
    site_mean_z_scores: np.ndarray
    final_half_sample_count: int
    iat_tolerance: int
    elapsed_seconds: float


@dataclass
class ParameterSweepResult:
    """Controlled one-parameter sweeps for the four primary methods."""

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


def _periodic_laplacian(field: Array, beta: float) -> Array:
    """Apply the weighted nearest-neighbor Laplacian on a square torus."""

    return beta * (
        4.0 * field
        - jnp.roll(field, 1, axis=0)
        - jnp.roll(field, -1, axis=0)
        - jnp.roll(field, 1, axis=1)
        - jnp.roll(field, -1, axis=1)
    )


def _truncated_scalar_potential(
    values: Array,
    quartic: float,
    mass: float,
    radius: float,
) -> tuple[Array, Array]:
    """Return ``(w_R, w_R')`` for a finite or infinite target radius.

    Clipping to the truncation boundary gives a compact expression exactly
    equivalent to the interior and both quadratic continuation branches. At
    ``radius=inf`` this function evaluates the genuine quartic directly; no
    clipping or continuation is applied.
    """

    if np.isposinf(radius):
        value = 0.25 * quartic * values**4 + 0.5 * mass * values**2
        gradient = quartic * values**3 + mass * values
        return value, gradient

    anchor = jnp.clip(values, -radius, radius)
    offset = values - anchor
    value_at_anchor = (
        0.25 * quartic * anchor**4 + 0.5 * mass * anchor**2
    )
    gradient_at_anchor = quartic * anchor**3 + mass * anchor
    hessian_at_anchor = 3.0 * quartic * anchor**2 + mass
    continued_value = (
        value_at_anchor
        + gradient_at_anchor * offset
        + 0.5 * hessian_at_anchor * offset**2
    )
    continued_gradient = gradient_at_anchor + hessian_at_anchor * offset
    return continued_value, continued_gradient


def make_lattice_model(
    side: int,
    beta: float,
    quartic: float,
    mass: float,
    radius: float,
    dtype: jnp.dtype,
    *,
    design_radius: float | None = None,
) -> LatticeModel:
    """Construct a truncated or genuine quartic lattice target.

    A finite target defaults to the tight choice ``design_radius=radius``.
    The genuine quartic (``radius=inf``) requires a finite design radius,
    because it has no finite global smoothness constant. In that case the
    resulting ``design_smoothness`` is used only for stage zero and numerical
    tuning and carries no global-smoothness guarantee.
    """

    if side < 2:
        raise ValueError("Lattice side length must be at least two.")
    radius = float(radius)
    is_truncated = bool(np.isfinite(radius))
    if (
        not np.isfinite(beta)
        or beta < 0.0
        or not np.isfinite(quartic)
        or quartic <= 0.0
        or not np.isfinite(mass)
        or mass <= 0.0
        or radius <= 0.0
        or (not is_truncated and not np.isposinf(radius))
    ):
        raise ValueError(
            "Require finite beta >= 0, finite quartic and mass > 0, and "
            "radius > 0 or radius=inf."
        )
    if design_radius is None:
        if not is_truncated:
            raise ValueError(
                "The genuine quartic target requires a finite positive "
                "design_radius for stage-zero and ULMC tuning."
            )
        design_radius = radius
    design_radius = float(design_radius)
    if not np.isfinite(design_radius) or design_radius <= 0.0:
        raise ValueError("design_radius must be finite and positive.")
    if is_truncated and design_radius < radius:
        raise ValueError(
            "A truncated target requires design_radius >= radius so its "
            "design smoothness remains a valid global upper bound."
        )

    def potential(phi: Array) -> Array:
        field = phi.reshape((side, side))
        laplacian = _periodic_laplacian(field, beta)
        scalar_values, _ = _truncated_scalar_potential(
            field,
            quartic,
            mass,
            radius,
        )
        return 0.5 * jnp.vdot(field, laplacian) + jnp.sum(scalar_values)

    def gradient(phi: Array) -> Array:
        field = phi.reshape((side, side))
        laplacian = _periodic_laplacian(field, beta)
        _, scalar_gradient = _truncated_scalar_potential(
            field,
            quartic,
            mass,
            radius,
        )
        return (laplacian + scalar_gradient).reshape((side * side,))

    frequencies = np.arange(side, dtype=float)
    one_dimensional = 4.0 * beta * np.sin(np.pi * frequencies / side) ** 2
    laplacian_spectrum = (
        one_dimensional[:, None] + one_dimensional[None, :]
    )
    max_laplacian = float(np.max(laplacian_spectrum))
    design_smoothness = (
        max_laplacian + mass + 3.0 * quartic * design_radius**2
    )
    global_smoothness_bound = (
        max_laplacian + mass + 3.0 * quartic * radius**2
        if is_truncated
        else np.inf
    )
    design_conditioning_alpha = 1.0 / (
        1.0 + 3.0 * quartic * design_radius**2 / mass
    )
    return LatticeModel(
        side=side,
        beta=beta,
        quartic=quartic,
        mass=mass,
        radius=radius,
        design_radius=design_radius,
        dtype=dtype,
        potential=potential,
        gradient=gradient,
        laplacian_spectrum=laplacian_spectrum,
        design_smoothness=design_smoothness,
        global_smoothness_bound=global_smoothness_bound,
        strong_convexity=mass,
        design_conditioning_alpha=design_conditioning_alpha,
    )


def validate_lattice_model(model: LatticeModel) -> None:
    """Run inexpensive analytic-gradient, spectrum, and symmetry checks."""

    side = model.side
    test_values = jnp.linspace(
        -1.7 * model.design_radius,
        1.7 * model.design_radius,
        model.dimension,
        dtype=model.dtype,
    )
    analytic = model.gradient(test_values)
    automatic = jax.grad(model.potential)(test_values)
    tolerance = 2e-5 if model.dtype == jnp.float32 else 2e-10
    gradient_error = float(jnp.max(jnp.abs(analytic - automatic)))
    if gradient_error > tolerance:
        raise AssertionError(
            f"Analytic lattice gradient error {gradient_error:.3e} "
            f"exceeds {tolerance:.1e}."
        )

    if not model.is_truncated:
        field = test_values.reshape((side, side))
        expected_value = float(
            0.5 * jnp.vdot(
                field,
                _periodic_laplacian(field, model.beta),
            )
            + jnp.sum(
                0.25 * model.quartic * field**4
                + 0.5 * model.mass * field**2
            )
        )
        direct_error = abs(float(model.potential(test_values)) - expected_value)
        if direct_error > 10.0 * tolerance * max(abs(expected_value), 1.0):
            raise AssertionError(
                "The R=inf target does not match the genuine quartic "
                f"formula (error {direct_error:.3e})."
            )

    field = test_values.reshape((side, side))
    shifted = jnp.roll(field, shift=(1, -1), axis=(0, 1)).reshape((-1,))
    original_value = float(model.potential(test_values))
    shifted_value = float(model.potential(shifted))
    symmetry_error = abs(original_value - shifted_value)
    symmetry_scale = max(abs(original_value), abs(shifted_value), 1.0)
    if symmetry_error > 10.0 * tolerance * symmetry_scale:
        raise AssertionError(
            f"Periodic-translation invariance error is {symmetry_error:.3e}."
        )

    impulse = jnp.zeros((side, side), dtype=model.dtype).at[0, 0].set(1.0)
    numerical_spectrum = np.asarray(
        jnp.real(
            jnp.fft.fftn(
                _periodic_laplacian(impulse, model.beta),
                norm=None,
            )
        )
    )
    spectrum_error = float(
        np.max(np.abs(numerical_spectrum - model.laplacian_spectrum))
    )
    if spectrum_error > 20.0 * tolerance:
        raise AssertionError(
            f"Periodic Laplacian spectrum error is {spectrum_error:.3e}."
        )


def _fourier_cooling_call(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
    num_stages: int | None = None,
) -> Array:
    if num_stages is None:
        num_stages = args.stages
    return translation_invariant_gaussian_cooling(
        key,
        model.potential,
        model.gradient,
        jnp.zeros(model.dimension, dtype=model.dtype),
        alpha=model.design_conditioning_alpha,
        delta=args.delta,
        cooling_gamma=args.cooling_gamma,
        smoothness_L=model.design_smoothness,
        num_stages=num_stages,
        num_chains=args.chains,
        num_ulmc_steps=args.steps,
        ulmc_step_size=args.step_size,
        ulmc_friction_gamma=args.friction,
        lattice_shape=model.lattice_shape,
        covariance_regularization=args.covariance_ridge,
        return_spectrum=True,
    )


def _unpreconditioned_samples_call(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
    num_stages: int | None = None,
) -> Array:
    """Return endpoints from an equal-budget unpreconditioned ULMC run."""

    if num_stages is None:
        num_stages = args.stages
    return ulmc(
        key,
        model.potential,
        model.gradient,
        jnp.zeros(model.dimension, dtype=model.dtype),
        args.friction,
        model.design_smoothness,
        args.step_size,
        args.steps * num_stages,
        args.chains,
    )


def _translation_averaged_ulmc_call(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
    num_stages: int | None = None,
) -> Array:
    """Project an unpreconditioned endpoint covariance onto translations."""

    samples = _unpreconditioned_samples_call(
        key,
        model,
        args,
        num_stages,
    )
    return sample_power_spectrum(samples, model.lattice_shape)


def _raw_ulmc_covariance_call(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
    num_stages: int | None = None,
) -> Array:
    """Return the full empirical covariance of unpreconditioned endpoints."""

    samples = _unpreconditioned_samples_call(
        key,
        model,
        args,
        num_stages,
    )
    return sample_covariance(samples)


def _unpreconditioned_estimators_call(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
    num_stages: int | None = None,
) -> tuple[Array, Array]:
    """Return both plain-ULMC estimators from exactly the same endpoints."""

    samples = _unpreconditioned_samples_call(
        key,
        model,
        args,
        num_stages,
    )
    return (
        sample_power_spectrum(samples, model.lattice_shape),
        sample_covariance(samples),
    )


def _make_unpreconditioned_stage_history_call(
    model: LatticeModel,
    args: argparse.Namespace,
    *,
    include_raw_covariances: bool,
) -> Callable[[Array], Array | tuple[Array, Array]]:
    """Return a compiled nested ULMC run sampled after each N-step block.

    The endpoint at stage ``k`` is the same chain after ``k * N`` continuous
    unpreconditioned transitions.  This avoids rerunning all prefixes while
    retaining the Gaussian experiment's fixed-key convergence semantics.
    """

    dimension = model.dimension
    num_chains = args.chains
    dtype = model.dtype
    initial_spectrum = (
        jnp.ones(dimension, dtype=dtype) / model.design_smoothness
    )
    (
        momentum_decay,
        position_coefficient,
        gradient_coefficient,
        noise_cholesky,
    ) = ulmc_coefficients(
        args.friction,
        args.step_size,
        dtype,
    )
    batched_gradient = jax.vmap(model.gradient)

    def run_history(key: Array) -> Array | tuple[Array, Array]:
        key_position, key_momentum, key_scan = random.split(key, 3)
        positions = random.normal(
            key_position,
            (num_chains, dimension),
            dtype=dtype,
        ) / jnp.sqrt(jnp.asarray(model.design_smoothness, dtype=dtype))
        momenta = random.normal(
            key_momentum,
            (num_chains, dimension),
            dtype=dtype,
        )

        def transition(carry, _):
            current_positions, current_momenta, current_key = carry
            next_key, noise_key = random.split(current_key)
            standard_noise = random.normal(
                noise_key,
                (num_chains, dimension, 2),
                dtype=dtype,
            )
            correlated_noise = standard_noise @ noise_cholesky.T
            gradient = jnp.asarray(
                batched_gradient(current_positions),
                dtype=dtype,
            )
            next_positions = (
                current_positions
                + position_coefficient * current_momenta
                - gradient_coefficient * gradient
                + correlated_noise[..., 0]
            )
            next_momenta = (
                momentum_decay * current_momenta
                - position_coefficient * gradient
                + correlated_noise[..., 1]
            )
            return (next_positions, next_momenta, next_key), None

        def stage(carry, _):
            next_carry, _ = lax.scan(
                transition,
                carry,
                xs=None,
                length=args.steps,
            )
            spectrum = sample_power_spectrum(
                next_carry[0],
                model.lattice_shape,
            )
            if include_raw_covariances:
                raw_covariance = sample_covariance(next_carry[0])
                return next_carry, (spectrum, raw_covariance)
            return next_carry, spectrum

        _, sampled_outputs = lax.scan(
            stage,
            (positions, momenta, key_scan),
            xs=None,
            length=args.stages,
        )
        if include_raw_covariances:
            sampled_spectra, raw_covariances = sampled_outputs
            spectrum_history = jnp.concatenate(
                (initial_spectrum[None, :], sampled_spectra),
                axis=0,
            )
            return spectrum_history, raw_covariances
        return jnp.concatenate(
            (initial_spectrum[None, :], sampled_outputs),
            axis=0,
        )

    return jax.jit(run_history)


def _fourier_empirical_baseline_call(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
    num_stages: int | None = None,
    initial_spectrum: Array | None = None,
) -> Array:
    """Run K translation-invariant empirical stages without cooling.

    This is the scalable lattice analogue of the Gaussian experiment's
    K-stage empirical-preconditioning baseline. It uses exactly the same
    stage transform, samples, transitions, and Fourier covariance estimator
    as translation-invariant Gaussian cooling, with ``cooling_gamma=0``.
    Since stages are one-based, every artificial quadratic strength is then
    identically zero.
    """

    if num_stages is None:
        num_stages = args.stages
    return translation_invariant_gaussian_cooling(
        key,
        model.potential,
        model.gradient,
        jnp.zeros(model.dimension, dtype=model.dtype),
        alpha=model.design_conditioning_alpha,
        delta=args.delta,
        cooling_gamma=0.0,
        smoothness_L=model.design_smoothness,
        num_stages=num_stages,
        num_chains=args.chains,
        num_ulmc_steps=args.steps,
        ulmc_step_size=args.step_size,
        ulmc_friction_gamma=args.friction,
        lattice_shape=model.lattice_shape,
        initial_spectrum=initial_spectrum,
        covariance_regularization=args.covariance_ridge,
        return_spectrum=True,
    )


def _fourier_stage_history(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
    *,
    use_cooling: bool,
) -> np.ndarray:
    """Return spectra after stages 0 through K without rerunning prefixes.

    A one-stage call accepts the preceding spectrum as its initial
    preconditioner. For cooling stage ``k``, supplying
    ``cooling_gamma ** k`` to that one-stage call reproduces the quadratic
    strength ``cooling_gamma ** k * L``. The no-cooling baseline instead
    supplies zero at every stage.
    """

    spectrum: Array = (
        jnp.ones(model.dimension, dtype=model.dtype)
        / model.design_smoothness
    )
    history: list[Array] = [spectrum]
    for stage_index in range(1, args.stages + 1):
        key, stage_key = random.split(key)
        if use_cooling:
            spectrum = translation_invariant_gaussian_cooling(
                stage_key,
                model.potential,
                model.gradient,
                jnp.zeros(model.dimension, dtype=model.dtype),
                alpha=model.design_conditioning_alpha,
                delta=args.delta,
                cooling_gamma=args.cooling_gamma**stage_index,
                smoothness_L=model.design_smoothness,
                num_stages=1,
                num_chains=args.chains,
                num_ulmc_steps=args.steps,
                ulmc_step_size=args.step_size,
                ulmc_friction_gamma=args.friction,
                lattice_shape=model.lattice_shape,
                initial_spectrum=spectrum,
                covariance_regularization=args.covariance_ridge,
                return_spectrum=True,
            )
        else:
            spectrum = _fourier_empirical_baseline_call(
                stage_key,
                model,
                args,
                num_stages=1,
                initial_spectrum=spectrum,
            )
        history.append(spectrum)
    history[-1].block_until_ready()
    return np.asarray(jnp.stack(history))


def _dense_cooling_call(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
) -> Array:
    return gaussian_cooling(
        key,
        model.potential,
        model.gradient,
        jnp.zeros(model.dimension, dtype=model.dtype),
        alpha=model.design_conditioning_alpha,
        delta=args.delta,
        cooling_gamma=args.cooling_gamma,
        smoothness_L=model.design_smoothness,
        num_stages=args.stages,
        num_chains=args.chains,
        num_ulmc_steps=args.steps,
        ulmc_step_size=args.step_size,
        ulmc_friction_gamma=args.friction,
        covariance_regularization=args.covariance_ridge,
    )


def _run_timed_repeats(
    call: Callable[[Array], Array],
    keys: Sequence[Array],
    warmup_key: Array,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return synchronized outputs and wall times for cached JAX executions."""

    call(warmup_key).block_until_ready()

    outputs: list[np.ndarray] = []
    runtimes = np.empty(len(keys), dtype=float)
    for index, key in enumerate(keys):
        started = time.perf_counter()
        output = call(key)
        output.block_until_ready()
        runtimes[index] = time.perf_counter() - started
        outputs.append(np.asarray(output))
    return outputs, runtimes


def _apply_spectrum_batch(
    samples: Array,
    spectrum: Array,
    lattice_shape: tuple[int, int],
) -> Array:
    return jax.vmap(
        lambda sample: apply_translation_invariant_spectrum(
            sample,
            spectrum,
            lattice_shape,
        )
    )(samples)


def reference_samples(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
) -> Array:
    """Generate reference endpoints using the known quadratic preconditioner."""

    quadratic_spectrum = jnp.asarray(
        1.0 / (model.laplacian_spectrum + model.mass),
        dtype=model.dtype,
    ).reshape((-1,))
    sqrt_spectrum = jnp.sqrt(quadratic_spectrum)

    def transformed_potential(y: Array) -> Array:
        physical = apply_translation_invariant_spectrum(
            y,
            sqrt_spectrum,
            model.lattice_shape,
        )
        return model.potential(physical)

    def transformed_gradient(y: Array) -> Array:
        physical = apply_translation_invariant_spectrum(
            y,
            sqrt_spectrum,
            model.lattice_shape,
        )
        return apply_translation_invariant_spectrum(
            model.gradient(physical),
            sqrt_spectrum,
            model.lattice_shape,
        )

    transformed_design_smoothness = (
        1.0
        + 3.0 * model.quartic * model.design_radius**2 / model.mass
    )
    reference_step_size = (
        args.step_size
        if args.reference_step_size is None
        else args.reference_step_size
    )
    reference_stiffness_margin = (
        reference_step_size * np.sqrt(transformed_design_smoothness)
    )
    if reference_stiffness_margin > 0.5:
        warnings.warn(
            "The reference sampler has "
            f"h_ref * sqrt(L_ref,design)={reference_stiffness_margin:.3g}. "
            "Use --reference-step-size to reduce reference discretization "
            "error for this small-mass target.",
            RuntimeWarning,
            stacklevel=2,
        )
    latent_samples = ulmc(
        key,
        transformed_potential,
        transformed_gradient,
        jnp.zeros(model.dimension, dtype=model.dtype),
        args.friction,
        transformed_design_smoothness,
        reference_step_size,
        args.reference_steps,
        args.reference_chains,
    )
    samples = _apply_spectrum_batch(
        latent_samples,
        sqrt_spectrum,
        model.lattice_shape,
    )
    if not bool(jnp.all(jnp.isfinite(samples))):
        target_name = "truncated" if model.is_truncated else "genuine quartic"
        raise FloatingPointError(
            f"The {target_name} reference run became nonfinite. Reduce "
            "--reference-step-size (and the method --step-size), or increase "
            "--cooling-design-radius for R=inf."
        )
    return samples


def _safe_spectrum(
    spectrum: np.ndarray,
    relative_floor: float,
) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.float64).reshape((-1,))
    scale = max(float(np.median(spectrum)), np.finfo(float).tiny)
    floor = relative_floor * scale
    if np.any(~np.isfinite(spectrum)):
        raise FloatingPointError("A covariance spectrum contains nonfinite values.")
    return np.maximum(spectrum, floor)


def spectral_relative_condition(
    estimate: np.ndarray,
    reference: np.ndarray,
    relative_floor: float,
) -> float:
    estimate = _safe_spectrum(estimate, relative_floor)
    reference = _safe_spectrum(reference, relative_floor)
    ratios = reference / estimate
    return float(np.max(ratios) / np.min(ratios))


def dense_relative_condition(
    estimate: np.ndarray,
    reference: np.ndarray,
    relative_ridge: float,
) -> float:
    """Compute the generalized covariance condition number."""

    estimate = 0.5 * (estimate + estimate.T)
    dimension = estimate.shape[0]
    scale = max(float(np.trace(estimate)) / dimension, 1.0)
    estimate = estimate + relative_ridge * scale * np.eye(dimension)
    try:
        cholesky = np.linalg.cholesky(estimate)
    except np.linalg.LinAlgError as exc:
        smallest = float(np.linalg.eigvalsh(estimate)[0])
        raise ValueError(
            "The empirical method produced a singular covariance "
            f"(smallest eigenvalue {smallest:.3e})."
        ) from exc
    inverse_cholesky = np.linalg.solve(cholesky, np.eye(dimension))
    whitened = inverse_cholesky @ reference @ inverse_cholesky.T
    eigenvalues = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))
    if eigenvalues[0] <= 0.0:
        raise FloatingPointError("Reference-relative eigenvalues are nonpositive.")
    return float(eigenvalues[-1] / eigenvalues[0])


def uniform_hessian_condition_bound(
    spectrum: np.ndarray,
    model: LatticeModel,
    relative_floor: float,
) -> float:
    """Return the reference-free bound from the target's Hessian inequalities."""

    if not model.is_truncated:
        return np.inf
    spectrum = _safe_spectrum(spectrum, relative_floor).reshape(
        model.lattice_shape
    )
    lower = spectrum * (model.laplacian_spectrum + model.mass)
    upper = spectrum * (
        model.laplacian_spectrum
        + model.mass
        + 3.0 * model.quartic * model.radius**2
    )
    return float(np.max(upper) / np.min(lower))


def periodic_distances(side: int) -> np.ndarray:
    """Return origin-to-site distances in flattened row-major order."""

    rows = np.repeat(np.arange(side), side)
    columns = np.tile(np.arange(side), side)
    row_distances = np.minimum(rows, side - rows)
    column_distances = np.minimum(columns, side - columns)
    return np.hypot(row_distances, column_distances)


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
    ) / jnp.sqrt(
        jnp.asarray(preconditioned_smoothness, dtype=model.dtype)
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
            transition = (
                sampling_transition if emit_fields else burnin_transition
            )

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
    return (
        final_half_sum / final_half_sample_count
    ).reshape(model.lattice_shape)


def _estimate_integrated_times(
    observables: np.ndarray,
    iat_tolerance: int,
) -> np.ndarray:
    """Estimate one IAT per observable, retaining short-chain estimates."""

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
            "The site-series store must have shape "
            "(side**2, trajectory_samples)."
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
        site_field_autocorrelation_times[start:stop] = (
            site_field_autocorrelation_batch
        )

        centered_sites = (
            site_series - np.mean(site_series, axis=1, keepdims=True)
        )
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
    site_field_reliable = (
        valid_site_estimates
        & (
            final_half_sample_count
            >= iat_tolerance * site_field_autocorrelation_times
        )
    )
    standard_errors = np.full(dimension, np.nan, dtype=float)
    effective_sample_sizes = np.full(dimension, np.nan, dtype=float)
    z_scores = np.full(dimension, np.nan, dtype=float)
    effective_sample_sizes[valid_site_estimates] = (
        final_half_sample_count
        / site_field_autocorrelation_times[valid_site_estimates]
    )
    standard_errors[valid_site_estimates] = np.sqrt(
        site_mean_variances[valid_site_estimates]
        / effective_sample_sizes[valid_site_estimates]
    )
    valid_standard_errors = (
        valid_site_estimates
        & np.isfinite(standard_errors)
        & (standard_errors > 0.0)
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

    storage_dtype = (
        np.float64 if model.dtype == jnp.float64 else np.float32
    )
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


def run_experiment(args: argparse.Namespace) -> LatticeResult:
    """Run cooling, equal-budget baselines, references, and diagnostics."""

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    sides = np.asarray(args.sides, dtype=int)
    diagnostic_side = int(
        sides[-1] if args.diagnostic_side is None else args.diagnostic_side
    )
    shape = (len(sides), args.repeats)
    relative_fourier = np.full(shape, np.nan)
    relative_dense = np.full(shape, np.nan)
    relative_empirical = np.full(shape, np.nan)
    relative_translation_averaged_ulmc = np.full(shape, np.nan)
    relative_raw_ulmc = np.full(shape, np.nan)
    runtime_fourier = np.full(shape, np.nan)
    runtime_dense = np.full(shape, np.nan)
    runtime_empirical = np.full(shape, np.nan)
    runtime_translation_averaged_ulmc = np.full(shape, np.nan)
    runtime_raw_ulmc = np.full(shape, np.nan)
    dense_skip_reasons: dict[int, str] = {}
    raw_ulmc_skip_reasons: dict[int, str] = {}
    comparison_conditions: dict[int, dict[str, np.ndarray]] = {}
    comparison_sides = (
        set() if args.skip_comparisons else set(args.comparison_sides)
    )
    representative_spectrum: np.ndarray | None = None

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
                args.cooling_design_radius
                if np.isposinf(args.radius)
                else None
            ),
        )
        validate_lattice_model(model)
        base_key = random.fold_in(root_key, side_index)
        key_warm_fourier = random.fold_in(base_key, 10_001)
        key_warm_dense = random.fold_in(base_key, 10_002)
        key_reference = random.fold_in(base_key, 10_003)
        fourier_keys = list(
            random.split(
                random.fold_in(base_key, 10_004),
                args.repeats,
            )
        )
        dense_keys = list(
            random.split(
                random.fold_in(base_key, 10_005),
                args.repeats,
            )
        )
        empirical_root = random.fold_in(base_key, 80_000)
        key_warm_empirical, key_empirical_runs = random.split(empirical_root)
        empirical_keys = list(
            random.split(key_empirical_runs, args.repeats)
        )
        plain_root = random.fold_in(base_key, 70_000)
        key_warm_plain, key_plain_runs = random.split(plain_root)
        plain_keys = list(random.split(key_plain_runs, args.repeats))

        print(
            f"side={side:3d} (D={model.dimension:5d}): "
            "translation-invariant Gaussian cooling, "
            f"{args.repeats} repeat(s)",
            flush=True,
        )
        fourier_call = lambda key, model=model: _fourier_cooling_call(
            key,
            model,
            args,
        )
        spectra, fourier_times = _run_timed_repeats(
            fourier_call,
            fourier_keys,
            key_warm_fourier,
        )
        runtime_fourier[side_index] = fourier_times

        print(
            "  Translation-invariant empirical preconditioning "
            "(no cooling)",
            flush=True,
        )
        empirical_call = lambda key, model=model: (
            _fourier_empirical_baseline_call(key, model, args)
        )
        empirical_spectra, empirical_times = _run_timed_repeats(
            empirical_call,
            empirical_keys,
            key_warm_empirical,
        )
        runtime_empirical[side_index] = empirical_times

        print(
            "  Translation-averaged covariance from unpreconditioned ULMC",
            flush=True,
        )
        translation_averaged_call = lambda key, model=model: (
            _translation_averaged_ulmc_call(key, model, args)
        )
        (
            translation_averaged_spectra,
            translation_averaged_times,
        ) = _run_timed_repeats(
            translation_averaged_call,
            plain_keys,
            key_warm_plain,
        )
        runtime_translation_averaged_ulmc[side_index] = (
            translation_averaged_times
        )

        raw_skip_reason: str | None = None
        if args.chains <= model.dimension:
            raw_skip_reason = (
                f"n={args.chains} <= D={model.dimension} (rank deficient)"
            )
        elif side > args.dense_max_side:
            raw_skip_reason = (
                f"side>{args.dense_max_side} full-covariance cutoff"
            )

        raw_covariances: list[np.ndarray] = []
        if raw_skip_reason is None:
            print(
                "  Full empirical covariance from the same "
                "unpreconditioned ULMC endpoints",
                flush=True,
            )
            raw_call = lambda key, model=model: _raw_ulmc_covariance_call(
                key,
                model,
                args,
            )
            raw_covariances, raw_times = _run_timed_repeats(
                raw_call,
                plain_keys,
                key_warm_plain,
            )
            runtime_raw_ulmc[side_index] = raw_times
        else:
            raw_ulmc_skip_reasons[side] = raw_skip_reason
            print(
                f"  Full empirical ULMC covariance omitted: {raw_skip_reason}",
                flush=True,
            )

        dense_skip_reason: str | None = None
        if side > args.dense_max_side:
            dense_skip_reason = (
                f"side>{args.dense_max_side} dense cutoff"
            )
        elif args.chains <= model.dimension:
            dense_skip_reason = (
                f"n={args.chains} <= D={model.dimension} (rank deficient)"
            )

        reference = reference_samples(key_reference, model, args)
        reference.block_until_ready()
        reference_spectrum = np.asarray(
            sample_power_spectrum(reference, model.lattice_shape)
        )
        reference_covariance: np.ndarray | None = None
        if raw_skip_reason is None or dense_skip_reason is None:
            reference_covariance = np.asarray(
                translation_invariant_covariance(
                    jnp.asarray(reference_spectrum, dtype=dtype),
                    model.lattice_shape,
                )
            )

        for repeat, spectrum in enumerate(spectra):
            relative_fourier[side_index, repeat] = spectral_relative_condition(
                spectrum,
                reference_spectrum,
                args.metric_floor,
            )
        for repeat, spectrum in enumerate(empirical_spectra):
            relative_empirical[side_index, repeat] = (
                spectral_relative_condition(
                    spectrum,
                    reference_spectrum,
                    args.metric_floor,
                )
            )
        for repeat, spectrum in enumerate(translation_averaged_spectra):
            relative_translation_averaged_ulmc[side_index, repeat] = (
                spectral_relative_condition(
                    spectrum,
                    reference_spectrum,
                    args.metric_floor,
                )
            )
        if raw_skip_reason is None:
            assert reference_covariance is not None
            for repeat, covariance in enumerate(raw_covariances):
                relative_raw_ulmc[side_index, repeat] = (
                    dense_relative_condition(
                        covariance,
                        reference_covariance,
                        0.0,
                    )
                )

        if side in comparison_sides:
            print(
                "  Stage comparison: four equal-budget preconditioners",
                flush=True,
            )
            stage_shape = (args.stages + 1, args.repeats)
            comparison = {
                method: np.full(stage_shape, np.nan, dtype=float)
                for method in FOUR_METHODS
            }
            initial_spectrum = (
                np.ones(model.dimension, dtype=float)
                / model.design_smoothness
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
                include_raw_covariances=(raw_skip_reason is None),
            )
            for repeat in range(args.repeats):
                # The adaptive methods use common random numbers.  The two
                # plain-ULMC estimators share a separate chain, so their only
                # difference is the covariance structure imposed afterward.
                comparison_key = random.fold_in(
                    base_key,
                    60_000 + repeat,
                )
                plain_comparison_key = random.fold_in(
                    comparison_key,
                    70_001,
                )
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
                    plain_history_call(plain_comparison_key)
                )
                if raw_skip_reason is None:
                    translation_averaged_history, raw_history = plain_output
                    raw_history_array: np.ndarray | None = np.asarray(
                        raw_history
                    )
                else:
                    translation_averaged_history = plain_output
                    raw_history_array = None
                adaptive_histories[TRANSLATION_AVERAGED_ULMC] = np.asarray(
                    translation_averaged_history
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
                    for stage_index, covariance in enumerate(
                        raw_history_array,
                        start=1,
                    ):
                        comparison[RAW_ULMC][stage_index, repeat] = (
                            dense_relative_condition(
                                covariance,
                                reference_covariance,
                                0.0,
                            )
                        )
            comparison_conditions[side] = comparison

        if side == diagnostic_side:
            median_condition = np.median(relative_fourier[side_index])
            representative_index = int(
                np.argmin(
                    np.abs(
                        relative_fourier[side_index] - median_condition
                    )
                )
            )
            representative_spectrum = spectra[representative_index]

        if dense_skip_reason is not None:
            dense_skip_reasons[side] = dense_skip_reason
            print(
                f"  Dense Gaussian cooling omitted: {dense_skip_reason}",
                flush=True,
            )
            continue

        print("  Dense Gaussian cooling comparison", flush=True)
        dense_call = lambda key, model=model: _dense_cooling_call(
            key,
            model,
            args,
        )
        dense_covariances, dense_times = _run_timed_repeats(
            dense_call,
            dense_keys,
            key_warm_dense,
        )
        runtime_dense[side_index] = dense_times
        assert reference_covariance is not None
        for repeat, covariance in enumerate(dense_covariances):
            relative_dense[side_index, repeat] = dense_relative_condition(
                covariance,
                reference_covariance,
                args.metric_ridge,
            )

    if args.skip_diagnostics:
        distances = np.asarray([])
        covariance_row = np.asarray([])
        autocorrelation_times = np.asarray([])
        reliable = np.asarray([], dtype=bool)
        final_half_site_means = np.empty((0, 0))
        site_field_autocorrelation_times = np.empty((0, 0))
        site_field_autocorrelation_reliable = np.empty(
            (0, 0),
            dtype=bool,
        )
        site_mean_standard_errors = np.empty((0, 0))
        site_mean_effective_sample_sizes = np.empty((0, 0))
        site_mean_z_scores = np.empty((0, 0))
        final_half_sample_count = 0
    else:
        assert representative_spectrum is not None
        diagnostic_model = make_lattice_model(
            diagnostic_side,
            args.beta,
            args.quartic,
            args.mass,
            args.radius,
            dtype,
            design_radius=(
                args.cooling_design_radius
                if np.isposinf(args.radius)
                else None
            ),
        )
        diagnostic_key = random.fold_in(root_key, 10_000)
        storage_bytes = (
            diagnostic_model.dimension
            * args.trajectory_samples
            * np.dtype(
                np.float64 if dtype == jnp.float64 else np.float32
            ).itemsize
        )
        print(
            "Post-cooling diagnostics: full first covariance row at "
            f"d={diagnostic_side}, T={args.trajectory_samples:,} "
            f"(temporary storage {storage_bytes / 2**20:.1f} MiB)",
            flush=True,
        )
        diagnostics = run_preconditioned_diagnostics(
            diagnostic_key,
            diagnostic_model,
            representative_spectrum,
            args,
        )
        distances = diagnostics.distances
        covariance_row = diagnostics.covariance_row
        autocorrelation_times = (
            diagnostics.correlator_autocorrelation_times
        )
        reliable = diagnostics.correlator_autocorrelation_reliable
        final_half_site_means = diagnostics.final_half_site_means
        site_field_autocorrelation_times = (
            diagnostics.site_field_autocorrelation_times
        )
        site_field_autocorrelation_reliable = (
            diagnostics.site_field_autocorrelation_reliable
        )
        site_mean_standard_errors = diagnostics.site_mean_standard_errors
        site_mean_effective_sample_sizes = (
            diagnostics.site_mean_effective_sample_sizes
        )
        site_mean_z_scores = diagnostics.site_mean_z_scores
        final_half_sample_count = diagnostics.final_half_sample_count

    return LatticeResult(
        sides=sides,
        relative_condition_fourier=relative_fourier,
        relative_condition_dense=relative_dense,
        relative_condition_empirical=relative_empirical,
        relative_condition_translation_averaged_ulmc=(
            relative_translation_averaged_ulmc
        ),
        relative_condition_raw_ulmc=relative_raw_ulmc,
        comparison_conditions=comparison_conditions,
        runtime_fourier=runtime_fourier,
        runtime_dense=runtime_dense,
        runtime_empirical=runtime_empirical,
        runtime_translation_averaged_ulmc=(
            runtime_translation_averaged_ulmc
        ),
        runtime_raw_ulmc=runtime_raw_ulmc,
        dense_skip_reasons=dense_skip_reasons,
        raw_ulmc_skip_reasons=raw_ulmc_skip_reasons,
        diagnostic_side=diagnostic_side,
        diagnostic_distances=distances,
        two_point_correlator_row=covariance_row,
        autocorrelation_times=autocorrelation_times,
        autocorrelation_reliable=reliable,
        final_half_site_means=final_half_site_means,
        site_field_autocorrelation_times=site_field_autocorrelation_times,
        site_field_autocorrelation_reliable=(
            site_field_autocorrelation_reliable
        ),
        site_mean_standard_errors=site_mean_standard_errors,
        site_mean_effective_sample_sizes=(
            site_mean_effective_sample_sizes
        ),
        site_mean_z_scores=site_mean_z_scores,
        final_half_sample_count=final_half_sample_count,
        iat_tolerance=args.iat_tolerance,
        elapsed_seconds=time.perf_counter() - started,
    )


def _parameter_reference_arguments(
    model: LatticeModel,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Return per-target reference settings with a fixed stability margin."""

    reference_args = argparse.Namespace(**vars(args))
    transformed_design_smoothness = (
        1.0
        + 3.0 * model.quartic * model.design_radius**2 / model.mass
    )
    requested_step_size = (
        args.step_size
        if args.reference_step_size is None
        else args.reference_step_size
    )
    stable_step_size = (
        args.parameter_sweep_reference_margin
        / np.sqrt(transformed_design_smoothness)
    )
    reference_step_size = min(requested_step_size, stable_step_size)
    reference_args.reference_step_size = reference_step_size
    reference_args.reference_steps = max(
        1,
        int(
            np.ceil(
                args.parameter_sweep_reference_time
                / reference_step_size
            )
        ),
    )
    reference_args.reference_chains = (
        args.parameter_sweep_reference_chains
    )
    return reference_args


def _evaluate_primary_methods(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
) -> _PrimaryMethodEvaluation:
    """Evaluate all four primary methods for one parameter configuration."""

    if args.chains <= model.dimension:
        raise ValueError(
            "The four-method parameter sweep requires n>D so the raw "
            "empirical covariance is nonsingular."
        )

    adaptive_root, plain_root, reference_key = random.split(key, 3)
    adaptive_warm_key, adaptive_run_root = random.split(adaptive_root)
    adaptive_keys = list(
        random.split(adaptive_run_root, args.repeats)
    )
    plain_warm_key, plain_run_root = random.split(plain_root)
    plain_keys = list(random.split(plain_run_root, args.repeats))

    cooling_call = lambda run_key: _fourier_cooling_call(
        run_key,
        model,
        args,
    )
    cooling_spectra, _ = _run_timed_repeats(
        cooling_call,
        adaptive_keys,
        adaptive_warm_key,
    )

    empirical_call = lambda run_key: _fourier_empirical_baseline_call(
        run_key,
        model,
        args,
    )
    empirical_spectra, _ = _run_timed_repeats(
        empirical_call,
        adaptive_keys,
        adaptive_warm_key,
    )

    plain_call = lambda run_key: _unpreconditioned_estimators_call(
        run_key,
        model,
        args,
    )
    jax.block_until_ready(plain_call(plain_warm_key))
    translation_averaged_spectra: list[np.ndarray] = []
    raw_covariances: list[np.ndarray] = []
    for plain_key in plain_keys:
        spectrum, covariance = jax.block_until_ready(
            plain_call(plain_key)
        )
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
        reference_condition=float(
            np.max(safe_reference) / np.min(safe_reference)
        ),
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
        "quartic": np.asarray(
            args.quartic_sweep_values,
            dtype=float,
        ),
        "mass": np.asarray(args.mass_sweep_values, dtype=float),
        "radius": np.asarray(args.radius_sweep_values, dtype=float),
    }
    relative_conditions = {
        sweep_name: {
            method: np.empty(
                (len(values), args.repeats),
                dtype=float,
            )
            for method in FOUR_METHODS
        }
        for sweep_name, values in sweep_specs.items()
    }
    hessian_bounds = {
        name: np.empty(len(values), dtype=float)
        for name, values in sweep_specs.items()
    }
    reference_conditions = {
        name: np.empty(len(values), dtype=float)
        for name, values in sweep_specs.items()
    }
    continuation_fractions = {
        name: np.empty(len(values), dtype=float)
        for name, values in sweep_specs.items()
    }
    design_exceedance_fractions = {
        name: np.empty(len(values), dtype=float)
        for name, values in sweep_specs.items()
    }
    reference_spectra: dict[str, list[np.ndarray | None]] = {
        name: [None] * len(values)
        for name, values in sweep_specs.items()
    }
    reference_step_sizes = {
        name: np.empty(len(values), dtype=float)
        for name, values in sweep_specs.items()
    }
    reference_steps = {
        name: np.empty(len(values), dtype=int)
        for name, values in sweep_specs.items()
    }

    sweep_args = argparse.Namespace(**vars(args))
    sweep_args.stages = args.parameter_sweep_stages
    root_key = random.fold_in(random.PRNGKey(args.seed), 300_000)
    cache: dict[
        tuple[float, float, float, float],
        _PrimaryMethodEvaluation,
    ] = {}
    started = time.perf_counter()

    def configuration(
        sweep_name: str,
        value: float,
    ) -> tuple[float, float, float, float]:
        quartic = (
            value
            if sweep_name == "quartic"
            else args.parameter_sweep_quartic
        )
        mass = (
            value
            if sweep_name == "mass"
            else args.parameter_sweep_mass
        )
        radius = (
            value
            if sweep_name == "radius"
            else args.parameter_sweep_radius
        )
        design_radius = (
            args.cooling_design_radius
            if sweep_name == "radius"
            else radius
        )
        return (
            float(quartic),
            float(mass),
            float(radius),
            float(design_radius),
        )

    for sweep_name, values in sweep_specs.items():
        for value_index, value in enumerate(values):
            target_configuration = configuration(
                sweep_name,
                float(value),
            )
            if target_configuration not in cache:
                quartic, mass, radius, design_radius = target_configuration
                model = make_lattice_model(
                    args.parameter_sweep_side,
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
                target_key = root_key
                print(
                    "Parameter sweep target "
                    f"{len(cache) + 1}: d={args.parameter_sweep_side}, "
                    f"lambda={quartic:g}, m={mass:g}, R={radius:g}, "
                    f"R_design={design_radius:g}",
                    flush=True,
                )
                cache[target_configuration] = _evaluate_primary_methods(
                    target_key,
                    model,
                    sweep_args,
                )

            evaluation = cache[target_configuration]
            for method in FOUR_METHODS:
                relative_conditions[sweep_name][method][
                    value_index
                ] = evaluation.relative_conditions[method]
            hessian_bounds[sweep_name][value_index] = (
                evaluation.hessian_condition_bound
            )
            reference_conditions[sweep_name][value_index] = (
                evaluation.reference_condition
            )
            continuation_fractions[sweep_name][value_index] = (
                evaluation.continuation_fraction
            )
            design_exceedance_fractions[sweep_name][value_index] = (
                evaluation.design_exceedance_fraction
            )
            reference_spectra[sweep_name][value_index] = (
                evaluation.reference_spectrum
            )
            reference_step_sizes[sweep_name][value_index] = (
                evaluation.reference_step_size
            )
            reference_steps[sweep_name][value_index] = (
                evaluation.reference_steps
            )

    radius_values = sweep_specs["radius"]
    truncation_to_quartic_conditions = np.full(
        len(radius_values),
        np.nan,
        dtype=float,
    )
    quartic_indices = np.flatnonzero(np.isposinf(radius_values))
    if quartic_indices.size:
        quartic_spectrum = reference_spectra["radius"][
            int(quartic_indices[0])
        ]
        assert quartic_spectrum is not None
        for value_index, finite_spectrum in enumerate(
            reference_spectra["radius"]
        ):
            assert finite_spectrum is not None
            truncation_to_quartic_conditions[value_index] = (
                spectral_relative_condition(
                    finite_spectrum,
                    quartic_spectrum,
                    args.metric_floor,
                )
            )

    return ParameterSweepResult(
        sweep_side=args.parameter_sweep_side,
        values=sweep_specs,
        relative_conditions=relative_conditions,
        hessian_condition_bounds=hessian_bounds,
        reference_conditions=reference_conditions,
        continuation_fractions=continuation_fractions,
        design_exceedance_fractions=design_exceedance_fractions,
        truncation_to_quartic_conditions=(
            truncation_to_quartic_conditions
        ),
        reference_step_sizes=reference_step_sizes,
        reference_steps=reference_steps,
        stages=args.parameter_sweep_stages,
        reference_chains=args.parameter_sweep_reference_chains,
        unique_target_count=len(cache),
        elapsed_seconds=time.perf_counter() - started,
    )


def _configure_plot_style() -> None:
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
    """Return target and cooling parameters shown on every phi4 plot."""

    reference_step_size = (
        args.step_size
        if args.reference_step_size is None
        else args.reference_step_size
    )
    reference_step_text = (
        ""
        if np.isclose(reference_step_size, args.step_size)
        else rf",\ h_{{\rm ref}}={reference_step_size:g}"
    )
    radius_text = (
        r"\infty" if np.isposinf(args.radius) else f"{args.radius:g}"
    )
    design_text = (
        rf",\ R_{{\rm design}}={args.cooling_design_radius:g}"
        if np.isposinf(args.radius)
        else ""
    )
    return (
        rf"$\beta={args.beta:g},\ \lambda={args.quartic:g},\ "
        + rf"m={args.mass:g},\ R={radius_text}"
        + design_text
        + "$"
        + "\n"
        + rf"$n={args.chains},\ N={args.steps},\ K={args.stages},\ "
        rf"\gamma_{{\rm cool}}={args.cooling_gamma:g},\ "
        rf"h={args.step_size:g},\ "
        rf"\gamma_{{\rm fric}}={args.friction:g}"
        + reference_step_text
        + "$"
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
    valid_rows = np.any(np.isfinite(values), axis=1)
    if not np.any(valid_rows):
        ax.plot(
            [],
            [],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
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


def make_figures(
    result: LatticeResult,
    args: argparse.Namespace,
) -> dict[str, plt.Figure]:
    """Create one standalone publication figure for each lattice plot."""

    _configure_plot_style()
    figures: dict[str, plt.Figure] = {}
    sides = result.sides.astype(float)

    quality_figure, ax = plt.subplots(
        figsize=(6.5, 4.7),
        constrained_layout=True,
    )
    quality_values = {
        COOLING_COMPARISON: result.relative_condition_fourier,
        EMPIRICAL_COMPARISON: result.relative_condition_empirical,
        TRANSLATION_AVERAGED_ULMC: (
            result.relative_condition_translation_averaged_ulmc
        ),
        RAW_ULMC: result.relative_condition_raw_ulmc,
    }
    for method in FOUR_METHODS:
        color, marker, linestyle = METHOD_STYLES[method]
        _plot_median_iqr(
            ax,
            sides,
            quality_values[method],
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
    ax.set_title(
        "Preconditioner quality for the lattice $\\phi^4$ target\n"
        + _phi4_parameter_subtitle(args)
    )
    ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.88,
        edgecolor="none",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncols=2,
        fontsize=7.4,
    )
    if result.raw_ulmc_skip_reasons:
        ax.text(
            0.98,
            0.13,
            "Full covariance omitted where rank deficient\n"
            "or beyond the full-covariance cutoff",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=GRAY,
            fontsize=7.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.5,
            },
        )
    figures["preconditioner_quality"] = quality_figure

    runtime_figure, ax = plt.subplots(
        figsize=(6.5, 4.7),
        constrained_layout=True,
    )
    dimensions = sides**2
    runtime_values = {
        COOLING_COMPARISON: result.runtime_fourier,
        EMPIRICAL_COMPARISON: result.runtime_empirical,
        TRANSLATION_AVERAGED_ULMC: (
            result.runtime_translation_averaged_ulmc
        ),
        RAW_ULMC: result.runtime_raw_ulmc,
    }
    for method in FOUR_METHODS:
        color, marker, linestyle = METHOD_STYLES[method]
        _plot_median_iqr(
            ax,
            dimensions,
            runtime_values[method],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=PLOT_LABELS[method],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of lattice sites $D=d^2$")
    ax.set_ylabel("Cached wall time (s)")
    ax.set_title(
        "Covariance-estimation cost\n" + _phi4_parameter_subtitle(args)
    )
    ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.88,
        edgecolor="none",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncols=2,
        fontsize=7.4,
    )
    if result.raw_ulmc_skip_reasons:
        ax.text(
            0.98,
            0.05,
            "Full covariance omitted where rank deficient\n"
            "or beyond the full-covariance cutoff",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=GRAY,
            fontsize=7.8,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.5,
            },
        )
    figures["covariance_estimation_cost"] = runtime_figure

    if np.any(np.isfinite(result.relative_condition_dense)):
        dense_quality_figure, ax = plt.subplots(
            figsize=(5.2, 3.9),
            constrained_layout=True,
        )
        _plot_median_iqr(
            ax,
            sides,
            result.relative_condition_fourier,
            color=BLUE,
            marker="o",
            linestyle="-",
            label=COOLING_COMPARISON,
        )
        _plot_median_iqr(
            ax,
            sides,
            result.relative_condition_dense,
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
            "Dense versus translation-invariant Gaussian cooling\n"
            + _phi4_parameter_subtitle(args)
        )
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        ax.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.88,
            edgecolor="none",
            loc="best",
            fontsize=7.6,
        )
        figures["dense_cooling_quality"] = dense_quality_figure

        dense_runtime_figure, ax = plt.subplots(
            figsize=(5.2, 3.9),
            constrained_layout=True,
        )
        _plot_median_iqr(
            ax,
            dimensions,
            result.runtime_fourier,
            color=BLUE,
            marker="o",
            linestyle="-",
            label=COOLING_COMPARISON,
        )
        _plot_median_iqr(
            ax,
            dimensions,
            result.runtime_dense,
            color=PURPLE,
            marker="D",
            linestyle="--",
            label="Dense Gaussian cooling",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Number of lattice sites $D=d^2$")
        ax.set_ylabel("Cached wall time (s)")
        ax.set_title(
            "Dense Gaussian-cooling cost\n"
            + _phi4_parameter_subtitle(args)
        )
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        ax.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.88,
            edgecolor="none",
            loc="best",
            fontsize=7.6,
        )
        figures["dense_cooling_cost"] = dense_runtime_figure

    stages = np.arange(args.stages + 1)
    for side, comparison in sorted(result.comparison_conditions.items()):
        comparison_figure, ax = plt.subplots(
            figsize=(6.6, 4.9),
            constrained_layout=True,
        )
        for method in FOUR_METHODS:
            color, marker, linestyle = METHOD_STYLES[method]
            _plot_median_iqr(
                ax,
                stages,
                comparison[method],
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=PLOT_LABELS[method],
            )
        ax.axhline(1.0, color="#222222", linestyle=":", linewidth=0.9)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.set_xlabel(
            r"Cumulative stage budget $k$ "
            r"(plain ULMC uses $kN$ transitions)"
        )
        ax.set_ylabel(r"Relative condition number $\kappa_{\mathrm{rel}}$")
        ax.set_title(
            rf"Four-method preconditioner comparison, $d={side}$"
            + "\n"
            + _phi4_parameter_subtitle(args)
        )
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        ax.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.88,
            edgecolor="none",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            fontsize=7.2,
            ncols=2,
        )
        raw_reason = result.raw_ulmc_skip_reasons.get(side)
        if raw_reason is not None:
            ax.text(
                0.98,
                0.12,
                f"Full empirical covariance unavailable after stage 0:\n"
                f"{raw_reason}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color=GRAY,
                fontsize=7.3,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.5,
                },
            )
        figures[f"stage_comparison_d{side}"] = comparison_figure

    if result.autocorrelation_times.size:
        iat_figure, ax = plt.subplots(
            figsize=(6.3, 4.7),
            constrained_layout=True,
        )
        distances = result.diagnostic_distances
        autocorrelation_times = result.autocorrelation_times
        finite = (
            np.isfinite(autocorrelation_times)
            & (autocorrelation_times > 0.0)
        )
        reliable = finite & result.autocorrelation_reliable
        unreliable = finite & ~result.autocorrelation_reliable
        if np.any(reliable):
            ax.scatter(
                distances[reliable],
                autocorrelation_times[reliable],
                s=8,
                color=PURPLE,
                alpha=0.22,
                linewidth=0,
                rasterized=True,
                label=rf"Reliable ($T\geq {result.iat_tolerance}\tau_{{\rm int}}$)",
            )
        if np.any(unreliable):
            ax.scatter(
                distances[unreliable],
                autocorrelation_times[unreliable],
                s=10,
                facecolors="none",
                edgecolors=PURPLE,
                alpha=0.28,
                linewidth=0.45,
                rasterized=True,
                label=rf"Short chain ($T<{result.iat_tolerance}\tau_{{\rm int}}$)",
            )

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
                np.median(
                    autocorrelation_times[finite][radial_bins == radial_bin]
                )
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
            color=GREEN,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            radial_centers,
            radial_median,
            color=GREEN,
            linewidth=1.8,
            label="Annular median and IQR",
        )
        ax.set_xlabel("Periodic distance from the origin")
        ax.set_ylabel(
            r"Integrated autocorrelation time $\tau_{\rm int}$ (ULMC steps)"
        )
        ax.set_title(
            "First covariance-row IATs after "
            "translation-invariant Gaussian cooling\n"
            rf"$g_x(t)=\delta\phi_t(0)\delta\phi_t(x)$, "
            rf"$\delta\phi=\phi-\bar{{\phi}}$, "
            rf"$d={result.diagnostic_side}$, "
            rf"$T={args.trajectory_samples}$, "
            rf"$N_{{\rm burn}}={args.trajectory_burnin}$"
            + "\n"
            + _phi4_parameter_subtitle(args)
        )
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        ax.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.86,
            edgecolor="none",
            loc="upper left",
            fontsize=7.6,
        )
        ax.text(
            0.98,
            0.03,
            f"{np.count_nonzero(reliable):,}/{len(reliable):,} "
            "entries reliable",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=GRAY,
            fontsize=7.6,
        )
        figures[
            f"two_point_iat_d{result.diagnostic_side}"
        ] = iat_figure

    if result.site_mean_z_scores.size:
        mean_figure, ax = plt.subplots(
            figsize=(6.2, 4.9),
            constrained_layout=True,
        )
        site_reliable = (
            result.site_field_autocorrelation_reliable
            & np.isfinite(result.site_mean_z_scores)
        )
        displayed_z_scores = np.ma.masked_where(
            ~site_reliable,
            result.site_mean_z_scores,
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
        colorbar.set_label(
            r"Standardized mean "
            r"$z_x=\bar{\phi}_x/\operatorname{MCSE}(\bar{\phi}_x)$"
        )
        ax.set_xlabel("Lattice coordinate $x_2$")
        ax.set_ylabel("Lattice coordinate $x_1$")
        ax.set_title(
            "Ergodicity diagnostic: final-half site-mean z-scores\n"
            rf"$d={result.diagnostic_side}$, "
            rf"$T_{{\rm retained}}={args.trajectory_samples}$, "
            rf"$N_{{\rm half}}={result.final_half_sample_count}$, "
            rf"$N_{{\rm burn}}={args.trajectory_burnin}$"
            + "\n"
            + _phi4_parameter_subtitle(args)
        )
        reliable_count = int(np.count_nonzero(site_reliable))
        if reliable_count:
            reliable_effective_sizes = (
                result.site_mean_effective_sample_sizes[site_reliable]
            )
            reliable_iats = result.site_field_autocorrelation_times[
                site_reliable
            ]
            reliable_standard_errors = result.site_mean_standard_errors[
                site_reliable
            ]
            within_two_standard_errors = np.mean(
                np.abs(result.site_mean_z_scores[site_reliable]) <= 1.96
            )
            diagnostic_text = (
                rf"reliable ($N_{{\rm half}}\geq "
                rf"{result.iat_tolerance}\tau_x$): "
                f"{reliable_count:,}/"
                f"{site_reliable.size:,}\n"
                rf"median $\tau_x$: {np.median(reliable_iats):.1f}; "
                f"ESS: {np.median(reliable_effective_sizes):.1f}; "
                f"MCSE: {np.median(reliable_standard_errors):.3g}\n"
                rf"$|z_x|\leq1.96$: "
                f"{100.0 * within_two_standard_errors:.1f}%\n"
                rf"RMS$(\bar{{\phi}}_x)$: "
                f"{np.sqrt(np.mean(result.final_half_site_means**2)):.3g}"
            )
        else:
            diagnostic_text = (
                rf"reliable ($N_{{\rm half}}\geq "
                rf"{result.iat_tolerance}\tau_x$): "
                f"0/{site_reliable.size:,}\n"
                "chain too short for standardized inference\n"
                rf"RMS$(\bar{{\phi}}_x)$: "
                f"{np.sqrt(np.mean(result.final_half_site_means**2)):.3g}"
            )
        ax.text(
            0.02,
            0.02,
            diagnostic_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="#222222",
            fontsize=7.8,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 2.0,
            },
        )
        figures[
            f"ergodicity_site_mean_z_scores_d{result.diagnostic_side}"
        ] = mean_figure

    return figures


def _parameter_sweep_subtitle(
    sweep_name: str,
    result: ParameterSweepResult,
    args: argparse.Namespace,
) -> str:
    """Return fixed target and budget parameters for one sweep plot."""

    if sweep_name == "quartic":
        fixed_parameters = (
            rf"m={args.parameter_sweep_mass:g},\ "
            rf"R={args.parameter_sweep_radius:g}"
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
        + "$"
        + "\n"
        + rf"$n={args.chains},\ N={args.steps},\ K={result.stages},\ "
        + rf"h={args.step_size:g},\ "
        + rf"\gamma_{{\rm cool}}={args.cooling_gamma:g},\ "
        + rf"n_{{\rm ref}}={result.reference_chains}$"
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
        plot_values = (
            np.arange(len(values), dtype=float)
            if is_radius_sweep
            else values
        )
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
            [
                r"$\infty$" if np.isposinf(value) else f"{value:g}"
                for value in values
            ],
        )
        if not is_radius_sweep:
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"Relative condition number $\kappa_{\mathrm{rel}}$")
        title_text = (
            title
            + "\n"
            + _parameter_sweep_subtitle(sweep_name, result, args)
        )
        if is_radius_sweep:
            continuation_values = ", ".join(
                (
                    r"$\infty$: n/a"
                    if np.isposinf(value)
                    else f"{value:g}: {100.0 * fraction:.2g}%"
                )
                for value, fraction in zip(
                    values,
                    result.continuation_fractions[sweep_name],
                    strict=True,
                )
            )
            title_text += (
                "\n"
                + r"Reference $\Pr(|\phi|>R)$ by $R$: "
                + continuation_values
            )
        ax.set_title(title_text)
        ax.grid(
            which="major",
            color="#D8D8D8",
            linewidth=0.55,
            alpha=0.8,
        )
        ax.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.88,
            edgecolor="none",
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
        ax.set_ylabel(
            r"$\kappa_{\mathrm{rel}}(\Sigma_R,\Sigma_\infty)$"
        )
        ax.set_title(
            "Reference-covariance convergence to the genuine quartic target\n"
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
        values = [
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated floating-point values."
        ) from exc
    if not values or any(
        not np.isfinite(item) or item <= 0.0
        for item in values
    ):
        raise argparse.ArgumentTypeError(
            "Sweep values must be finite and positive."
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Sweep values must be unique.")
    return sorted(values)


def _parse_target_radii(value: str) -> list[float]:
    """Parse distinct positive radii, allowing positive infinity."""

    try:
        radii = [
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated radii; use 'inf' for the genuine "
            "quartic target."
        ) from exc
    if not radii or any(
        item <= 0.0 or np.isnan(item) or np.isneginf(item)
        for item in radii
    ):
        raise argparse.ArgumentTypeError(
            "Target radii must be positive finite values or 'inf'."
        )
    if len(set(radii)) != len(radii):
        raise argparse.ArgumentTypeError("Target radii must be unique.")
    return sorted(radii)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Compare four covariance-preconditioning methods on truncated "
            "and genuine-quartic periodic lattice phi4 targets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sides",
        type=_parse_sides,
        default=[5, 10, 20, 50, 100],
        help="Comma-separated lattice side lengths for the main experiment.",
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
        help="Quartic coupling lambda for the main lattice-size experiment.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=0.25,
        help="Mass m for the main lattice-size experiment.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help=(
            "Quadratic-continuation radius R for the main experiment; use "
            "'inf' for the genuine quartic target."
        ),
    )
    parser.add_argument(
        "--cooling-design-radius",
        type=float,
        default=4.0,
        help=(
            "Finite operational curvature radius used for R=inf and, for a "
            "fair comparison, for every target in the radius sweep. It sets "
            "stage zero and ULMC tuning but does not truncate the target."
        ),
    )
    parser.add_argument(
        "--lambda-sweep-values",
        "--quartic-sweep-values",
        dest="quartic_sweep_values",
        type=_parse_positive_values,
        default=[0.1, 0.5, 2.0],
        help=(
            "Quartic couplings for the controlled lambda sweep. The fixed "
            "mass and radius are set by the parameter-sweep anchor."
        ),
    )
    parser.add_argument(
        "--mass-sweep-values",
        type=_parse_positive_values,
        default=[0.01, 0.05, 0.25],
        help=(
            "Masses for the controlled mass sweep. The fixed lambda and "
            "radius are set by the parameter-sweep anchor."
        ),
    )
    parser.add_argument(
        "--radius-sweep-values",
        type=_parse_target_radii,
        default=[0.5, 2.0, 4.0, np.inf],
        help=(
            "Target radii for the controlled comparison; use 'inf' for the "
            "genuine quartic. All entries share --cooling-design-radius."
        ),
    )
    parser.add_argument(
        "--parameter-sweep-quartic",
        type=float,
        default=0.5,
        help="Fixed quartic coupling at the parameter-sweep anchor.",
    )
    parser.add_argument(
        "--parameter-sweep-mass",
        type=float,
        default=0.05,
        help="Fixed mass at the parameter-sweep anchor.",
    )
    parser.add_argument(
        "--parameter-sweep-radius",
        type=float,
        default=2.0,
        help="Fixed truncation radius at the parameter-sweep anchor.",
    )
    parser.add_argument(
        "--parameter-sweep-side",
        type=int,
        default=10,
        help=(
            "Lattice side for the controlled parameter sweeps. The default "
            "keeps the raw empirical covariance full rank."
        ),
    )
    parser.add_argument(
        "--parameter-sweep-stages",
        type=int,
        default=12,
        help=(
            "Equal stage count used by all four methods in every parameter "
            "sweep target."
        ),
    )
    parser.add_argument(
        "--parameter-sweep-reference-chains",
        type=int,
        default=1024,
        help="Reference endpoint count for each parameter-sweep target.",
    )
    parser.add_argument(
        "--parameter-sweep-reference-time",
        type=float,
        default=7.68,
        help=(
            "Reference physical time preserved while its step size is "
            "automatically reduced for stiff sweep targets."
        ),
    )
    parser.add_argument(
        "--parameter-sweep-reference-margin",
        type=float,
        default=0.15,
        help=(
            "Maximum h_ref*sqrt(L_ref) used to tune each sweep reference."
        ),
    )
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
        help="Stages K in the main lattice-size experiment.",
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
            "Largest side at which any full D-by-D covariance method is "
            "formed; use 0 to disable every dense/full-covariance path. "
            "Rank-deficient empirical covariances remain omitted."
        ),
    )
    parser.add_argument(
        "--reference-chains",
        type=int,
        default=256,
        help="Independent chains used for each main reference estimate.",
    )
    parser.add_argument(
        "--reference-steps",
        type=int,
        default=256,
        help="ULMC transitions used for each main reference estimate.",
    )
    parser.add_argument(
        "--reference-step-size",
        type=float,
        default=None,
        help=(
            "ULMC step size for the independently preconditioned reference "
            "run. By default, reuse --step-size; decreasing this is important "
            "when small mass makes the transformed reference target stiff."
        ),
    )
    parser.add_argument(
        "--trajectory-burnin",
        type=int,
        default=1024,
        help="Burn-in transitions for the post-cooling diagnostic chain.",
    )
    parser.add_argument(
        "--diagnostic-side",
        type=int,
        default=None,
        help=(
            "Lattice side used for post-cooling IAT and ergodicity "
            "diagnostics. By default use the largest requested side; the "
            "--gpu preset skips diagnostics, or uses d=64 when explicitly "
            "enabled."
        ),
    )
    parser.add_argument(
        "--trajectory-samples",
        type=int,
        default=8192,
        help="Retained post-burn-in diagnostic samples.",
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
        help="Number of consecutive diagnostic transitions run per JAX scan.",
    )
    parser.add_argument(
        "--iat-batch-size",
        type=int,
        default=64,
        help=(
            "Number of first-row covariance observables processed together "
            "when estimating IATs."
        ),
    )
    parser.add_argument(
        "--comparison-sides",
        type=_parse_sides,
        default=[10, 100],
        help=(
            "Comma-separated lattice sides for the four-method stage "
            "comparison: translation-invariant cooling, translation-"
            "invariant empirical preconditioning, translation averaging "
            "of plain ULMC, and the raw plain-ULMC covariance."
        ),
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
    diagnostic_group = parser.add_mutually_exclusive_group()
    diagnostic_group.add_argument(
        "--skip-diagnostics",
        dest="skip_diagnostics",
        action="store_true",
        help=(
            "Skip the full covariance-row IAT and uncertainty-aware "
            "final-half mean diagnostics."
        ),
    )
    diagnostic_group.add_argument(
        "--run-diagnostics",
        dest="skip_diagnostics",
        action="store_false",
        help="Run diagnostics even when a preset would skip them.",
    )
    comparison_group = parser.add_mutually_exclusive_group()
    comparison_group.add_argument(
        "--skip-comparisons",
        "--skip-stage-convergence",
        dest="skip_comparisons",
        action="store_true",
        help="Skip the separate stage-comparison plots.",
    )
    comparison_group.add_argument(
        "--run-comparisons",
        dest="skip_comparisons",
        action="store_false",
        help="Run stage comparisons even when a preset would skip them.",
    )
    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument(
        "--skip-parameter-sweeps",
        dest="skip_parameter_sweeps",
        action="store_true",
        help=(
            "Skip the controlled lambda, mass, and radius comparisons."
        ),
    )
    sweep_group.add_argument(
        "--run-parameter-sweeps",
        dest="skip_parameter_sweeps",
        action="store_false",
        help="Run parameter sweeps even when a preset would skip them.",
    )
    parser.set_defaults(
        skip_diagnostics=False,
        skip_comparisons=False,
        skip_parameter_sweeps=False,
    )
    parser.add_argument(
        "--output-prefix",
        "--output",
        dest="output",
        type=Path,
        default=script_dir / "figures" / "truncated_phi4",
        help=(
            "Base output prefix; each plot name is appended and saved as a "
            "separate PDF."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Require a JAX GPU backend and run the scalable lattice-size "
            "feasibility test at sides 64,128,256,512,1024 using conservative "
            "float32 budgets. Dense paths, diagnostics, stage comparisons, "
            "and parameter sweeps are skipped unless explicitly enabled; "
            "explicit numerical options override the preset."
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
        "sides": [4, 6, 8],
        "comparison_sides": [4, 8],
        "repeats": 1,
        "chains": 64,
        "steps": 20,
        "stages": 4,
        "dense_max_side": 8,
        "reference_chains": 96,
        "reference_steps": 48,
        "quartic_sweep_values": [0.5, 1.0],
        "mass_sweep_values": [0.05, 0.25],
        "radius_sweep_values": [1.0, 2.0, np.inf],
        "parameter_sweep_side": 4,
        "parameter_sweep_stages": 4,
        "parameter_sweep_reference_chains": 96,
        "parameter_sweep_reference_time": 1.44,
        "trajectory_burnin": 100,
        "trajectory_samples": 500,
        "dtype": "float32",
    }
    for destination, value in quick_values.items():
        if destination not in explicitly_set:
            setattr(args, destination, value)
    if (
        "cooling_design_radius" not in explicitly_set
        and "radius_sweep_values" not in explicitly_set
    ):
        args.cooling_design_radius = 2.0
    if (
        "sides" in explicitly_set
        and "comparison_sides" not in explicitly_set
    ):
        args.comparison_sides = sorted(
            {args.sides[0], args.sides[-1]}
        )


def apply_gpu_configuration(
    args: argparse.Namespace,
    explicit_destinations: set[str] | None = None,
) -> None:
    """Apply the large-lattice GPU preset without overriding CLI choices."""

    if not args.gpu:
        return
    explicitly_set = (
        set()
        if explicit_destinations is None
        else explicit_destinations
    )
    gpu_values = {
        "sides": list(GPU_LATTICE_SIDES),
        "repeats": 1,
        "chains": 64,
        "steps": 32,
        "stages": 8,
        "reference_chains": 128,
        "reference_steps": 128,
        "dense_max_side": 0,
        "parameter_sweep_side": 4,
        "parameter_sweep_reference_chains": 128,
        "skip_diagnostics": True,
        "skip_comparisons": True,
        "skip_parameter_sweeps": True,
        "dtype": "float32",
    }
    for destination, value in gpu_values.items():
        if destination not in explicitly_set:
            setattr(args, destination, value)
    if (
        not args.skip_parameter_sweeps
        and "dense_max_side" not in explicitly_set
    ):
        args.dense_max_side = args.parameter_sweep_side

    moderate_sides = [
        side for side in args.sides if side <= GPU_DIAGNOSTIC_MAX_SIDE
    ]
    if "diagnostic_side" not in explicitly_set:
        if moderate_sides:
            args.diagnostic_side = max(moderate_sides)
        elif (
            "skip_diagnostics" in explicitly_set
            and not args.skip_diagnostics
        ):
            args.diagnostic_side = min(args.sides)
        else:
            args.skip_diagnostics = True

    if "comparison_sides" not in explicitly_set:
        args.comparison_sides = sorted(
            {args.sides[0], args.sides[-1]}
        )


def _available_gpu_devices() -> list[object]:
    """Return JAX GPU devices, with a focused error for backend failures."""

    try:
        devices = jax.devices()
    except RuntimeError as exc:
        raise RuntimeError(
            "JAX could not initialize its device backend while validating "
            "--gpu. Check the CUDA-enabled JAX installation and driver."
        ) from exc
    gpu_platforms = {"gpu", "cuda", "rocm"}
    return [
        device
        for device in devices
        if str(device.platform).lower() in gpu_platforms
    ]


def validate_arguments(args: argparse.Namespace) -> None:
    if args.gpu and not _available_gpu_devices():
        raise RuntimeError(
            "--gpu requires a JAX GPU backend, but no GPU device was found. "
            "Install CUDA-enabled JAX and ensure the Pod exposes its GPU "
            "before starting this experiment."
        )
    if args.gpu and args.dtype == "float64":
        warnings.warn(
            "The --gpu preset is sized for float32. Float64 approximately "
            "doubles real state storage and should be enabled only after a "
            "successful one-repeat memory benchmark.",
            RuntimeWarning,
            stacklevel=2,
        )

    counts = {
        "repeats": args.repeats,
        "chains": args.chains,
        "steps": args.steps,
        "stages": args.stages,
        "reference_chains": args.reference_chains,
        "reference_steps": args.reference_steps,
        "parameter_sweep_stages": args.parameter_sweep_stages,
        "parameter_sweep_reference_chains": (
            args.parameter_sweep_reference_chains
        ),
        "trajectory_samples": args.trajectory_samples,
        "iat_tolerance": args.iat_tolerance,
        "trajectory_chunk_size": args.trajectory_chunk_size,
        "iat_batch_size": args.iat_batch_size,
    }
    invalid = [name for name, value in counts.items() if value <= 0]
    if invalid:
        raise ValueError(f"These counts must be positive: {', '.join(invalid)}.")
    if args.trajectory_burnin < 0:
        raise ValueError("--trajectory-burnin must be nonnegative.")
    if args.trajectory_samples < 4:
        raise ValueError(
            "--trajectory-samples must be at least four so the final-half "
            "variance and IAT are defined."
        )
    if (
        not np.isfinite(args.cooling_gamma)
        or not 0.0 < args.cooling_gamma < 1.0
    ):
        raise ValueError("--cooling-gamma must lie in (0, 1).")
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
            "These values must be finite and positive: "
            + ", ".join(invalid_positive)
            + "."
        )
    if (
        args.radius <= 0.0
        or np.isnan(args.radius)
        or np.isneginf(args.radius)
    ):
        raise ValueError(
            "--radius must be positive and finite, or 'inf' for the genuine "
            "quartic target."
        )
    if (
        args.reference_step_size is not None
        and (
            not np.isfinite(args.reference_step_size)
            or args.reference_step_size <= 0.0
        )
    ):
        raise ValueError(
            "--reference-step-size must be finite and positive."
        )
    nonnegative_scalars = {
        "beta": args.beta,
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
    if args.chains < 2 or args.reference_chains < 4:
        raise ValueError(
            "--chains must be at least two and --reference-chains at least four."
        )
    if args.dense_max_side < 0:
        raise ValueError("--dense-max-side must be nonnegative.")
    sweep_scalars = {
        "parameter-sweep-quartic": args.parameter_sweep_quartic,
        "parameter-sweep-mass": args.parameter_sweep_mass,
        "parameter-sweep-radius": args.parameter_sweep_radius,
        "parameter-sweep-reference-time": (
            args.parameter_sweep_reference_time
        ),
        "parameter-sweep-reference-margin": (
            args.parameter_sweep_reference_margin
        ),
    }
    invalid_sweep_scalars = [
        name
        for name, value in sweep_scalars.items()
        if not np.isfinite(value) or value <= 0.0
    ]
    if invalid_sweep_scalars:
        raise ValueError(
            "These parameter-sweep values must be finite and positive: "
            + ", ".join(invalid_sweep_scalars)
            + "."
        )
    if args.parameter_sweep_side < 2:
        raise ValueError("--parameter-sweep-side must be at least two.")
    if args.diagnostic_side is not None:
        if args.diagnostic_side < 2:
            raise ValueError("--diagnostic-side must be at least two.")
        if args.diagnostic_side not in args.sides:
            raise ValueError(
                "--diagnostic-side must be included in --sides; got "
                f"d={args.diagnostic_side}."
            )
    includes_genuine_quartic = np.isposinf(args.radius) or (
        not args.skip_parameter_sweeps
        and any(np.isposinf(value) for value in args.radius_sweep_values)
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
    if not args.skip_parameter_sweeps:
        sweep_dimension = args.parameter_sweep_side**2
        if args.parameter_sweep_reference_chains < 4:
            raise ValueError(
                "--parameter-sweep-reference-chains must be at least four."
            )
        if args.chains <= sweep_dimension:
            raise ValueError(
                "The four-method parameter sweeps require --chains > "
                f"--parameter-sweep-side^2; got n={args.chains} and "
                f"D={sweep_dimension}."
            )
        if args.parameter_sweep_side > args.dense_max_side:
            raise ValueError(
                "The four-method parameter sweeps require "
                "--parameter-sweep-side <= --dense-max-side so the raw "
                "full-covariance method is feasible."
            )
        largest_finite_radius = max(
            (
                value
                for value in args.radius_sweep_values
                if np.isfinite(value)
            ),
            default=0.0,
        )
        if args.cooling_design_radius < largest_finite_radius:
            raise ValueError(
                "--cooling-design-radius must be at least the largest finite "
                "--radius-sweep-values entry so it is a valid common "
                "smoothness bound for every truncated comparison target."
            )
        sweep_configurations = (
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
        max_design_smoothness = max(
            make_lattice_model(
                args.parameter_sweep_side,
                args.beta,
                quartic,
                mass,
                radius,
                jnp.float32,
                design_radius=design_radius,
            ).design_smoothness
            for quartic, mass, radius, design_radius in sweep_configurations
        )
        method_stiffness_margin = (
            args.step_size * np.sqrt(max_design_smoothness)
        )
        if method_stiffness_margin > 0.5:
            warnings.warn(
                "The parameter grid has "
                f"h*sqrt(L)={method_stiffness_margin:.3g}; use a smaller "
                "--step-size or a less extreme grid.",
                RuntimeWarning,
                stacklevel=2,
            )
    missing_comparison_sides = sorted(
        set(args.comparison_sides) - set(args.sides)
    )
    if not args.skip_comparisons and missing_comparison_sides:
        raise ValueError(
            "--comparison-sides must be included in --sides; missing "
            + ", ".join(str(side) for side in missing_comparison_sides)
            + "."
        )
    if not args.skip_diagnostics and integrated_time is None:
        raise RuntimeError(
            "Autocorrelation diagnostics require `emcee`; install it with "
            "`python -m pip install emcee`."
        )


def print_summary(result: LatticeResult) -> None:
    def median_text(
        values: np.ndarray,
        *,
        suffix: str = "",
        unavailable: str = "omitted",
    ) -> str:
        finite = np.asarray(values)[np.isfinite(values)]
        if not finite.size:
            return unavailable
        return f"{np.median(finite):.4g}{suffix}"

    print("\nMedian relative condition numbers")
    print(
        "side".ljust(7)
        + "TI cooling".rjust(15)
        + "TI empirical".rjust(15)
        + "TI-avg ULMC".rjust(16)
        + "Raw ULMC".rjust(17)
        + "Dense cooling".rjust(17)
    )
    for index, side in enumerate(result.sides):
        side_int = int(side)
        raw_unavailable = (
            "rank deficient"
            if "rank deficient"
            in result.raw_ulmc_skip_reasons.get(side_int, "")
            else "omitted"
        )
        dense_unavailable = (
            "rank deficient"
            if "rank deficient"
            in result.dense_skip_reasons.get(side_int, "")
            else "omitted"
        )
        translation_averaged_text = median_text(
            result.relative_condition_translation_averaged_ulmc[index]
        )
        raw_text = median_text(
            result.relative_condition_raw_ulmc[index],
            unavailable=raw_unavailable,
        )
        dense_text = median_text(
            result.relative_condition_dense[index],
            unavailable=dense_unavailable,
        )
        print(
            f"{side_int:<7d}"
            f"{median_text(result.relative_condition_fourier[index]):>15}"
            f"{median_text(result.relative_condition_empirical[index]):>15}"
            f"{translation_averaged_text:>16}"
            f"{raw_text:>17}"
            f"{dense_text:>17}"
        )

    print("\nMedian cached wall times (seconds)")
    print(
        "side".ljust(7)
        + "TI cooling".rjust(15)
        + "TI empirical".rjust(15)
        + "TI-avg ULMC".rjust(16)
        + "Raw ULMC".rjust(17)
        + "Dense cooling".rjust(17)
    )
    for index, side in enumerate(result.sides):
        side_int = int(side)
        raw_unavailable = (
            "rank deficient"
            if "rank deficient"
            in result.raw_ulmc_skip_reasons.get(side_int, "")
            else "omitted"
        )
        dense_unavailable = (
            "rank deficient"
            if "rank deficient"
            in result.dense_skip_reasons.get(side_int, "")
            else "omitted"
        )
        translation_averaged_text = median_text(
            result.runtime_translation_averaged_ulmc[index],
            suffix="s",
        )
        raw_text = median_text(
            result.runtime_raw_ulmc[index],
            suffix="s",
            unavailable=raw_unavailable,
        )
        dense_text = median_text(
            result.runtime_dense[index],
            suffix="s",
            unavailable=dense_unavailable,
        )
        print(
            f"{side_int:<7d}"
            f"{median_text(result.runtime_fourier[index], suffix='s'):>15}"
            f"{median_text(result.runtime_empirical[index], suffix='s'):>15}"
            f"{translation_averaged_text:>16}"
            f"{raw_text:>17}"
            f"{dense_text:>17}"
        )

    for side, comparison in sorted(result.comparison_conditions.items()):
        stage_zero = np.median(comparison[COOLING_COMPARISON][0])
        print(
            f"\nStage comparison, side={side}: "
            rf"stage-0 kappa_rel={stage_zero:.4g}"
        )
        for method in FOUR_METHODS:
            unavailable = (
                "rank deficient"
                if method == RAW_ULMC
                and "rank deficient"
                in result.raw_ulmc_skip_reasons.get(side, "")
                else "omitted"
            )
            print(
                f"  final {method}: "
                f"{median_text(comparison[method][-1], unavailable=unavailable)}"
            )

    if result.autocorrelation_times.size:
        finite_times = result.autocorrelation_times[
            np.isfinite(result.autocorrelation_times)
            & (result.autocorrelation_times > 0.0)
        ]
        print(
            "\nFirst covariance-row IATs after translation-invariant "
            f"Gaussian cooling, side={result.diagnostic_side}:"
        )
        if finite_times.size:
            print(
                f"  entries={len(result.autocorrelation_times):,}, "
                f"median tau={np.median(finite_times):.3f}, "
                f"max tau={np.max(finite_times):.3f}, "
                f"reliable="
                f"{np.count_nonzero(result.autocorrelation_reliable):,}"
                f"/{len(result.autocorrelation_reliable):,}"
            )
        else:
            print(
                f"  entries={len(result.autocorrelation_times):,}; "
                "no finite positive IAT estimates"
            )
        print(
            "  final-half site-mean RMS="
            f"{np.sqrt(np.mean(result.final_half_site_means**2)):.4g} "
            f"from {result.final_half_sample_count:,} samples"
        )
        site_reliable = (
            result.site_field_autocorrelation_reliable
            & np.isfinite(result.site_mean_z_scores)
        )
        reliable_count = int(np.count_nonzero(site_reliable))
        if reliable_count:
            reliable_iats = result.site_field_autocorrelation_times[
                site_reliable
            ]
            reliable_ess = result.site_mean_effective_sample_sizes[
                site_reliable
            ]
            reliable_z = result.site_mean_z_scores[site_reliable]
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
    print(f"\nTotal wall time (including compilation): {result.elapsed_seconds:.2f}s")


def print_parameter_sweep_summary(
    result: ParameterSweepResult,
) -> None:
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
                        result.relative_conditions[sweep_name][method][
                            value_index
                        ]
                    )
                )
                for method in FOUR_METHODS
            }
            condition_bound = result.hessian_condition_bounds[
                sweep_name
            ][value_index]
            reference_condition = result.reference_conditions[
                sweep_name
            ][value_index]
            continuation_percent = 100.0 * (
                result.continuation_fractions[sweep_name][value_index]
            )
            design_exceedance_percent = 100.0 * (
                result.design_exceedance_fractions[sweep_name][value_index]
            )
            reference_step_size = result.reference_step_sizes[
                sweep_name
            ][value_index]
            reference_step_count = result.reference_steps[
                sweep_name
            ][value_index]
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
                (
                    "inf"
                    if np.isposinf(value)
                    else f"{value:g}"
                )
                + f": {condition:.4g}"
                for value, condition in zip(
                    values,
                    result.truncation_to_quartic_conditions,
                    strict=True,
                )
            )
            print(
                "  reference kappa_rel(Sigma_R, Sigma_inf): "
                + convergence_text
            )
    print(
        "\nParameter-sweep wall time (including compilation): "
        f"{result.elapsed_seconds:.2f}s"
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.quick and args.gpu:
        parser.error("--quick and --gpu are mutually exclusive presets.")
    explicit_destinations = _explicit_cli_destinations(parser, arguments)
    apply_quick_configuration(
        args,
        explicit_destinations,
    )
    apply_gpu_configuration(args, explicit_destinations)
    validate_arguments(args)

    if args.gpu:
        gpu_devices = _available_gpu_devices()
        device_names = ", ".join(
            str(getattr(device, "device_kind", device))
            for device in gpu_devices
        )
        diagnostic_text = (
            "disabled"
            if args.skip_diagnostics
            else f"d={args.diagnostic_side}"
        )
        print(
            "GPU large-lattice preset: "
            f"devices={device_names}; sides={','.join(map(str, args.sides))}; "
            f"n={args.chains}; N={args.steps}; K={args.stages}; "
            f"diagnostics={diagnostic_text}",
            flush=True,
        )

    result = run_experiment(args)
    parameter_sweep_result = (
        None
        if args.skip_parameter_sweeps
        else run_parameter_sweeps(args)
    )
    figures = make_figures(result, args)
    if parameter_sweep_result is not None:
        figures.update(
            make_parameter_sweep_figures(
                parameter_sweep_result,
                args,
            )
        )
    output_paths = save_publication_figures(figures, args.output)
    for figure in figures.values():
        plt.close(figure)

    print_summary(result)
    if parameter_sweep_result is not None:
        print_parameter_sweep_summary(parameter_sweep_result)
    for plot_name, path in output_paths.items():
        print(f"Saved {plot_name.replace('_', ' ')} PDF: {path}")


if __name__ == "__main__":
    main()
