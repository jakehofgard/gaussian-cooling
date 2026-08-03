"""Shared lattice :math:`\phi^4` models, samplers, and condition metrics.

This module contains the numerical building blocks used by the standalone
lattice experiments.  It deliberately has no plotting or autocorrelation
dependencies, so importing it does not require Matplotlib or ``emcee``.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import lax, random

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

GPU_LATTICE_SIDES = (64, 128, 256, 512, 1024)
GPU_STAGE_COMPARISON_SIDES = (512, 1024)
GPU_DIAGNOSTIC_MAX_SIDE = 100
HARDNESS_BETA = 2.0
HARDNESS_RADIUS = 4.0

COOLING_COMPARISON = "Translation-invariant Gaussian cooling"
EMPIRICAL_COMPARISON = "Translation-invariant empirical preconditioning (no cooling)"
TRANSLATION_AVERAGED_ULMC = "Translation-averaged unpreconditioned ULMC"
RAW_ULMC = "Full empirical covariance from unpreconditioned ULMC"

FOUR_METHODS = (
    COOLING_COMPARISON,
    EMPIRICAL_COMPARISON,
    TRANSLATION_AVERAGED_ULMC,
    RAW_ULMC,
)
SCALABLE_METHODS = (
    COOLING_COMPARISON,
    EMPIRICAL_COMPARISON,
    TRANSLATION_AVERAGED_ULMC,
)


@dataclass(frozen=True)
class LatticeModel:
    """Periodic lattice phi4 target and its algorithmic tuning scales."""

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
    """Return ``(w_R, w_R')`` for a finite or infinite target radius."""

    if np.isposinf(radius):
        value = 0.25 * quartic * values**4 + 0.5 * mass * values**2
        gradient = quartic * values**3 + mass * values
        return value, gradient

    anchor = jnp.clip(values, -radius, radius)
    offset = values - anchor
    value_at_anchor = 0.25 * quartic * anchor**4 + 0.5 * mass * anchor**2
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
    """Construct a truncated or genuine-quartic lattice target."""

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
    laplacian_spectrum = one_dimensional[:, None] + one_dimensional[None, :]
    max_laplacian = float(np.max(laplacian_spectrum))
    design_smoothness = max_laplacian + mass + 3.0 * quartic * design_radius**2
    global_smoothness_bound = (
        max_laplacian + mass + 3.0 * quartic * radius**2 if is_truncated else np.inf
    )
    design_conditioning_alpha = 1.0 / (1.0 + 3.0 * quartic * design_radius**2 / mass)
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
            0.5
            * jnp.vdot(
                field,
                _periodic_laplacian(field, model.beta),
            )
            + jnp.sum(0.25 * model.quartic * field**4 + 0.5 * model.mass * field**2)
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

    samples = _unpreconditioned_samples_call(key, model, args, num_stages)
    return sample_power_spectrum(samples, model.lattice_shape)


def _raw_ulmc_covariance_call(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
    num_stages: int | None = None,
) -> Array:
    """Return the full empirical covariance of unpreconditioned endpoints."""

    samples = _unpreconditioned_samples_call(key, model, args, num_stages)
    return sample_covariance(samples)


def _unpreconditioned_estimators_call(
    key: Array,
    model: LatticeModel,
    args: argparse.Namespace,
    num_stages: int | None = None,
) -> tuple[Array, Array]:
    """Return both plain-ULMC estimators from exactly the same endpoints."""

    samples = _unpreconditioned_samples_call(key, model, args, num_stages)
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
    """Return a compiled nested ULMC run sampled after each N-step block."""

    dimension = model.dimension
    num_chains = args.chains
    dtype = model.dtype
    initial_spectrum = jnp.ones(dimension, dtype=dtype) / model.design_smoothness
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
    """Run K translation-invariant empirical stages without cooling."""

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
    """Return spectra after stages zero through K without rerunning prefixes."""

    spectrum: Array = (
        jnp.ones(model.dimension, dtype=model.dtype) / model.design_smoothness
    )
    history: list[Array] = [spectrum]
    for stage_index in range(1, args.stages + 1):
        stage_key = key
        key, _ = random.split(key)
        if use_cooling:
            stage_cooling_gamma = jnp.power(
                jnp.asarray(args.cooling_gamma, dtype=model.dtype),
                stage_index,
            )
            spectrum = translation_invariant_gaussian_cooling(
                stage_key,
                model.potential,
                model.gradient,
                jnp.zeros(model.dimension, dtype=model.dtype),
                alpha=model.design_conditioning_alpha,
                delta=args.delta,
                cooling_gamma=stage_cooling_gamma,
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
    """Return synchronized outputs and post-compilation JAX wall times."""

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
        1.0 + 3.0 * model.quartic * model.design_radius**2 / model.mass
    )
    reference_step_size = (
        args.step_size if args.reference_step_size is None else args.reference_step_size
    )
    reference_stiffness_margin = reference_step_size * np.sqrt(
        transformed_design_smoothness
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
    """Return the reference-free bound from the target Hessian inequalities."""

    if not model.is_truncated:
        return np.inf
    spectrum = _safe_spectrum(spectrum, relative_floor).reshape(model.lattice_shape)
    lower = spectrum * (model.laplacian_spectrum + model.mass)
    upper = spectrum * (
        model.laplacian_spectrum + model.mass + 3.0 * model.quartic * model.radius**2
    )
    return float(np.max(upper) / np.min(lower))


def periodic_distances(side: int) -> np.ndarray:
    """Return origin-to-site distances in flattened row-major order."""

    rows = np.repeat(np.arange(side), side)
    columns = np.tile(np.arange(side), side)
    row_distances = np.minimum(rows, side - rows)
    column_distances = np.minimum(columns, side - columns)
    return np.hypot(row_distances, column_distances)


def _periodic_distance_shells(
    side: int,
    center: tuple[int, int],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return exact toroidal radii and flattened sites in each shell."""

    center_row, center_column = center
    if not (0 <= center_row < side and 0 <= center_column < side):
        raise ValueError("center must be a valid lattice coordinate")
    rows = np.repeat(np.arange(side), side)
    columns = np.tile(np.arange(side), side)
    row_offsets = np.abs(rows - center_row)
    column_offsets = np.abs(columns - center_column)
    row_offsets = np.minimum(row_offsets, side - row_offsets)
    column_offsets = np.minimum(column_offsets, side - column_offsets)
    squared_distances = row_offsets**2 + column_offsets**2
    unique_squared_distances = np.unique(squared_distances)
    shell_indices = tuple(
        np.flatnonzero(squared_distances == squared_distance)
        for squared_distance in unique_squared_distances
    )
    return np.sqrt(unique_squared_distances.astype(float)), shell_indices


def _normalized_preconditioner_operators(
    model: LatticeModel,
    preconditioner: np.ndarray,
    *,
    translation_invariant: bool,
    relative_floor: float,
) -> tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    """Return square-root actions after a scalar smoothness normalization."""

    upper_hessian_spectrum = (
        model.laplacian_spectrum
        + model.mass
        + 3.0 * model.quartic * model.design_radius**2
    )
    if translation_invariant:
        spectrum = _safe_spectrum(
            np.asarray(preconditioner, dtype=float).reshape((-1,)),
            relative_floor,
        )
        if spectrum.size != model.dimension:
            raise ValueError(
                "A translation-invariant preconditioner must have D entries."
            )
        transformed_smoothness = float(
            np.max(spectrum.reshape(model.lattice_shape) * upper_hessian_spectrum)
        )
        normalized_sqrt_spectrum = jnp.sqrt(
            jnp.asarray(
                spectrum / transformed_smoothness,
                dtype=model.dtype,
            )
        )

        def to_physical(y: Array) -> Array:
            return apply_translation_invariant_spectrum(
                y,
                normalized_sqrt_spectrum,
                model.lattice_shape,
            )

        def transform_gradient(gradient: Array) -> Array:
            return apply_translation_invariant_spectrum(
                gradient,
                normalized_sqrt_spectrum,
                model.lattice_shape,
            )

        return to_physical, transform_gradient

    covariance = np.asarray(preconditioner, dtype=float)
    if covariance.shape != (model.dimension, model.dimension):
        raise ValueError("A dense preconditioner must have shape (D, D).")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalue_scale = max(float(np.max(eigenvalues)), 1.0)
    if eigenvalues[0] <= 100.0 * np.finfo(float).eps * eigenvalue_scale:
        raise ValueError(
            "The dense empirical preconditioner is not numerically positive "
            "definite; increase n or choose a smaller correlator side."
        )
    covariance_sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    upper_hessian = np.asarray(
        translation_invariant_covariance(
            jnp.asarray(
                upper_hessian_spectrum.reshape((-1,)),
                dtype=model.dtype,
            ),
            model.lattice_shape,
        )
    )
    transformed_upper = covariance_sqrt @ upper_hessian @ covariance_sqrt
    transformed_smoothness = float(
        np.linalg.eigvalsh(0.5 * (transformed_upper + transformed_upper.T))[-1]
    )
    normalized_factor = jnp.asarray(
        covariance_sqrt / np.sqrt(transformed_smoothness),
        dtype=model.dtype,
    )

    def to_physical(y: Array) -> Array:
        return normalized_factor @ y

    def transform_gradient(gradient: Array) -> Array:
        return normalized_factor.T @ gradient

    return to_physical, transform_gradient


def _representative_repeat_index(values: np.ndarray) -> int:
    """Return the repeat nearest the finite sample median."""

    values = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not finite.size:
        raise ValueError("Cannot choose a representative from no finite values.")
    median = np.median(values[finite])
    return int(finite[np.argmin(np.abs(values[finite] - median))])


__all__ = [
    "Array",
    "PotentialFn",
    "LatticeModel",
    "GPU_LATTICE_SIDES",
    "GPU_STAGE_COMPARISON_SIDES",
    "GPU_DIAGNOSTIC_MAX_SIDE",
    "HARDNESS_BETA",
    "HARDNESS_RADIUS",
    "COOLING_COMPARISON",
    "EMPIRICAL_COMPARISON",
    "TRANSLATION_AVERAGED_ULMC",
    "RAW_ULMC",
    "FOUR_METHODS",
    "SCALABLE_METHODS",
    "make_lattice_model",
    "validate_lattice_model",
    "reference_samples",
    "spectral_relative_condition",
    "dense_relative_condition",
    "uniform_hessian_condition_bound",
    "periodic_distances",
]
