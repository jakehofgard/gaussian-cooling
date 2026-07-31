"""JAX implementations of underdamped Langevin and Gaussian-cooling methods.

The module provides standard and linearly transformed exact-integration
underdamped Langevin Monte Carlo (ULMC), dense Gaussian cooling, and a
translation-invariant Gaussian-cooling variant for periodic lattices.

Sampler functions run independent chains and return their final positions,
not complete trajectories. Cooling functions return covariance
preconditioners or, when requested, their Fourier eigenvalues. Randomness is
controlled entirely through explicit JAX PRNG keys.

The library does not enable JAX 64-bit mode globally. Applications requiring
float64 should call ``jax.config.update("jax_enable_x64", True)`` before
constructing their arrays.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax, random


Array = jax.Array
PotentialFn = Callable[[Array], Array]

__all__ = [
    "ulmc",
    "transformed_ulmc",
    "ulmc_coefficients",
    "sample_mean",
    "sample_covariance",
    "sample_power_spectrum",
    "symmetric_matrix_sqrt",
    "apply_translation_invariant_spectrum",
    "translation_invariant_covariance",
    "gaussian_cooling",
    "translation_invariant_gaussian_cooling",
]


def _floating_array(x: Array) -> Array:
    """Convert ``x`` to a JAX array, promoting integer inputs to float."""

    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.asarray(0.0).dtype)
    return x


def ulmc_coefficients(
    friction_gamma: float,
    step_size_h: float,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array]:
    """Return the exact transition coefficients and noise Cholesky factor.

    Direct evaluation of the position-noise variance subtracts three nearly
    equal terms when ``friction_gamma * step_size_h`` is small.  The formulas
    below are algebraically identical to the specification, with short Taylor
    expansions used near zero to retain float32 accuracy.
    """

    g = jnp.asarray(friction_gamma, dtype=dtype)
    h = jnp.asarray(step_size_h, dtype=dtype)
    x = g * h

    exp_neg_x = jnp.exp(-x)
    one_minus_exp_neg_x = -jnp.expm1(-x)
    one_minus_exp_neg_2x = -jnp.expm1(-2.0 * x)

    # (1 - exp(-x)) / x
    phi1_series = 1.0 + x * (
        -1.0 / 2.0
        + x
        * (
            1.0 / 6.0
            + x
            * (
                -1.0 / 24.0
                + x
                * (
                    1.0 / 120.0
                    + x
                    * (
                        -1.0 / 720.0
                        + x
                        * (
                            1.0 / 5040.0
                            + x * (-1.0 / 40320.0 + x * (1.0 / 362880.0))
                        )
                    )
                )
            )
        )
    )

    # (x - 1 + exp(-x)) / x**2
    phi2_series = 1.0 / 2.0 + x * (
        -1.0 / 6.0
        + x
        * (
            1.0 / 24.0
            + x
            * (
                -1.0 / 120.0
                + x
                * (
                    1.0 / 720.0
                    + x
                    * (
                        -1.0 / 5040.0
                        + x
                        * (
                            1.0 / 40320.0
                            + x * (-1.0 / 362880.0 + x * (1.0 / 3628800.0))
                        )
                    )
                )
            )
        )
    )

    # 2 * (x - 2(1-e^-x) + (1-e^-2x)/2) / x**2
    position_variance_series = x * (
        2.0 / 3.0
        + x
        * (
            -1.0 / 2.0
            + x
            * (
                7.0 / 30.0
                + x
                * (
                    -1.0 / 12.0
                    + x
                    * (
                        31.0 / 1260.0
                        + x
                        * (
                            -1.0 / 160.0
                            + x * (127.0 / 90720.0 + x * (-17.0 / 60480.0))
                        )
                    )
                )
            )
        )
    )

    # Float32 loses several digits in the three-term expression for ``a`` at
    # substantially larger x than float64 does.  The eighth-order polynomial
    # is more accurate than direct evaluation throughout these intervals.
    series_threshold = 0.5 if jnp.finfo(dtype).eps > 1e-10 else 2e-2
    use_series = jnp.abs(x) <= jnp.asarray(series_threshold, dtype=dtype)

    # Outside the series regime, use dimensional formulas with sequential
    # division.  This avoids the otherwise unnecessary x**2 and h**2
    # overflows for large, but still representable, parameter values.  These
    # forms also remain finite when the product x = g*h itself overflows.
    direct_position_coefficient = one_minus_exp_neg_x / g
    position_coefficient = jnp.where(
        use_series,
        h * phi1_series,
        direct_position_coefficient,
    )
    gradient_coefficient = jnp.where(
        use_series,
        h * h * phi2_series,
        (h - direct_position_coefficient) / g,
    )
    a = jnp.where(
        use_series,
        h * h * position_variance_series,
        2.0
        * (
            h / g
            - 2.0 * direct_position_coefficient / g
            + 0.5 * (one_minus_exp_neg_2x / g) / g
        ),
    )

    # These are the entries (a, b, c) in the specified 2-by-2 covariance.
    b = position_coefficient * one_minus_exp_neg_x
    c = one_minus_exp_neg_2x
    covariance = jnp.stack(
        (jnp.stack((a, b)), jnp.stack((b, c))),
    )
    zero_noise = x == 0
    covariance_for_cholesky = jnp.where(
        zero_noise,
        jnp.eye(2, dtype=dtype),
        covariance,
    )
    noise_cholesky = jnp.where(
        zero_noise,
        jnp.zeros((2, 2), dtype=dtype),
        jnp.linalg.cholesky(covariance_for_cholesky),
    )

    return exp_neg_x, position_coefficient, gradient_coefficient, noise_cholesky


def _ulmc_impl(
    rng_key: Array,
    potential_fn: PotentialFn,
    grad_potential_fn: PotentialFn,
    minimizer: Array,
    friction_gamma: float,
    smoothness_L: float,
    step_size_h: float,
    num_steps: int,
    num_chains: int,
) -> Array:
    """Unjitted ULMC body shared by the public and cooling samplers."""

    del potential_fn  # The exact transition uses the supplied gradient only.

    minimizer = _floating_array(minimizer)
    if minimizer.ndim != 1:
        raise ValueError("minimizer must be a one-dimensional array")
    if num_steps < 0:
        raise ValueError("num_steps must be nonnegative")
    if num_chains <= 0:
        raise ValueError("num_chains must be positive")
    dtype = minimizer.dtype
    dimension = minimizer.shape[0]

    key_position, key_momentum, key_scan = random.split(rng_key, 3)
    initial_positions = minimizer + random.normal(
        key_position,
        shape=(num_chains, dimension),
        dtype=dtype,
    ) / jnp.sqrt(jnp.asarray(smoothness_L, dtype=dtype))
    initial_momenta = random.normal(
        key_momentum,
        shape=(num_chains, dimension),
        dtype=dtype,
    )

    (
        momentum_decay,
        position_coefficient,
        gradient_coefficient,
        noise_cholesky,
    ) = ulmc_coefficients(friction_gamma, step_size_h, dtype)

    batched_gradient = jax.vmap(grad_potential_fn)

    def transition(carry, _):
        positions, momenta, key = carry
        key, noise_key = random.split(key)

        standard_noise = random.normal(
            noise_key,
            shape=(num_chains, dimension, 2),
            dtype=dtype,
        )
        correlated_noise = standard_noise @ noise_cholesky.T
        position_noise = correlated_noise[..., 0]
        momentum_noise = correlated_noise[..., 1]

        gradient = jnp.asarray(batched_gradient(positions), dtype=dtype)
        next_positions = (
            positions
            + position_coefficient * momenta
            - gradient_coefficient * gradient
            + position_noise
        )
        next_momenta = (
            momentum_decay * momenta
            - position_coefficient * gradient
            + momentum_noise
        )
        return (next_positions, next_momenta, key), None

    (final_positions, _, _), _ = lax.scan(
        transition,
        (initial_positions, initial_momenta, key_scan),
        xs=None,
        length=num_steps,
    )
    return final_positions


@partial(
    jax.jit,
    static_argnames=(
        "potential_fn",
        "grad_potential_fn",
        "num_steps",
        "num_chains",
    ),
)
def ulmc(
    rng_key: Array,
    potential_fn: PotentialFn,
    grad_potential_fn: PotentialFn,
    minimizer: Array,
    friction_gamma: float,
    smoothness_L: float,
    step_size_h: float,
    num_steps: int,
    num_chains: int,
) -> Array:
    """Run exact-integration ULMC and return final positions.

    The chains are initialized independently with
    ``X_0 ~ N(minimizer, smoothness_L**-1 I)`` and ``P_0 ~ N(0, I)``.
    The returned array has shape ``(num_chains, d)``.
    """

    return _ulmc_impl(
        rng_key,
        potential_fn,
        grad_potential_fn,
        minimizer,
        friction_gamma,
        smoothness_L,
        step_size_h,
        num_steps,
        num_chains,
    )


def _transformed_ulmc_impl(
    rng_key: Array,
    potential_fn: PotentialFn,
    grad_potential_fn: PotentialFn,
    minimizer: Array,
    factor_C: Array,
    friction_gamma: float,
    smoothness_L: float,
    step_size_h: float,
    num_steps: int,
    num_chains: int,
    noise_cholesky: Array | None,
) -> Array:
    """Unjitted transformed-ULMC body with ``A = C @ C.T``."""

    del potential_fn

    minimizer = _floating_array(minimizer)
    factor_C = _floating_array(factor_C)
    if minimizer.ndim != 1:
        raise ValueError("minimizer must be a one-dimensional array")
    if factor_C.ndim != 2:
        raise ValueError("factor_C must be a two-dimensional array")
    if factor_C.shape[0] != minimizer.shape[0]:
        raise ValueError(
            "factor_C must have one row per coordinate of minimizer"
        )
    if factor_C.shape[1] < factor_C.shape[0]:
        raise ValueError(
            "factor_C must have at least d columns to have full row rank"
        )
    if num_steps < 0:
        raise ValueError("num_steps must be nonnegative")
    if num_chains <= 0:
        raise ValueError("num_chains must be positive")
    dtype = jnp.result_type(minimizer.dtype, factor_C.dtype)
    minimizer = minimizer.astype(dtype)
    factor_C = factor_C.astype(dtype)

    dimension, latent_dimension = factor_C.shape
    key_position, key_momentum, key_scan = random.split(rng_key, 3)
    latent_positions = random.normal(
        key_position,
        shape=(num_chains, latent_dimension),
        dtype=dtype,
    )
    latent_momenta = random.normal(
        key_momentum,
        shape=(num_chains, latent_dimension),
        dtype=dtype,
    )
    initial_positions = (
        minimizer
        + (latent_positions @ factor_C.T)
        / jnp.sqrt(jnp.asarray(smoothness_L, dtype=dtype))
    )
    initial_momenta = latent_momenta @ factor_C.T

    (
        momentum_decay,
        position_coefficient,
        gradient_coefficient,
        computed_noise_cholesky,
    ) = ulmc_coefficients(friction_gamma, step_size_h, dtype)
    if noise_cholesky is None:
        noise_cholesky = computed_noise_cholesky
    else:
        noise_cholesky = jnp.asarray(noise_cholesky, dtype=dtype)
        if noise_cholesky.shape != (2, 2):
            raise ValueError("noise_cholesky must have shape (2, 2)")

    batched_gradient = jax.vmap(grad_potential_fn)

    def transition(carry, _):
        positions, momenta, key = carry
        key, noise_key = random.split(key)

        standard_noise = random.normal(
            noise_key,
            shape=(num_chains, latent_dimension, 2),
            dtype=dtype,
        )
        latent_noise = standard_noise @ noise_cholesky.T
        transformed_noise = jnp.einsum(
            "dr,brq->bdq",
            factor_C,
            latent_noise,
        )
        position_noise = transformed_noise[..., 0]
        momentum_noise = transformed_noise[..., 1]

        gradient = jnp.asarray(batched_gradient(positions), dtype=dtype)
        transformed_gradient = (gradient @ factor_C) @ factor_C.T
        next_positions = (
            positions
            + position_coefficient * momenta
            - gradient_coefficient * transformed_gradient
            + position_noise
        )
        next_momenta = (
            momentum_decay * momenta
            - position_coefficient * transformed_gradient
            + momentum_noise
        )
        return (next_positions, next_momenta, key), None

    (final_positions, _, _), _ = lax.scan(
        transition,
        (initial_positions, initial_momenta, key_scan),
        xs=None,
        length=num_steps,
    )
    return final_positions


@partial(
    jax.jit,
    static_argnames=(
        "potential_fn",
        "grad_potential_fn",
        "num_steps",
        "num_chains",
    ),
)
def transformed_ulmc(
    rng_key: Array,
    potential_fn: PotentialFn,
    grad_potential_fn: PotentialFn,
    minimizer: Array,
    factor_C: Array,
    friction_gamma: float,
    smoothness_L: float,
    step_size_h: float,
    num_steps: int,
    num_chains: int,
    *,
    noise_cholesky: Array | None = None,
) -> Array:
    """Run transformed ULMC with ``A = factor_C @ factor_C.T``.

    ``factor_C`` has shape ``(d, r)`` and must have full row rank; normally it
    is a square, nonsingular covariance factor. Initialization, momentum,
    gradient drift, and correlated noise are transformed without explicitly
    forming either ``A`` or a Kronecker covariance. If supplied,
    ``noise_cholesky`` must be the 2-by-2 Cholesky factor for one exact ULMC
    transition; otherwise it is computed once.
    """

    return _transformed_ulmc_impl(
        rng_key,
        potential_fn,
        grad_potential_fn,
        minimizer,
        factor_C,
        friction_gamma,
        smoothness_L,
        step_size_h,
        num_steps,
        num_chains,
        noise_cholesky,
    )


def sample_mean(samples: Array) -> Array:
    """Return the sample mean of an ``(n, d)`` array."""

    samples = _floating_array(samples)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("samples must have shape (n, d) with n > 0")
    return jnp.mean(samples, axis=0)


def sample_covariance(samples: Array) -> Array:
    """Return the biased sample covariance, normalized by ``n``.

    ``samples`` must have shape ``(n, d)`` with ``n > 0``. The estimator has
    rank at most ``n - 1``.
    """

    samples = _floating_array(samples)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("samples must have shape (n, d) with n > 0")
    # Shifting first is algebraically neutral and avoids loss of precision when
    # all samples share a large offset relative to their spread.
    shifted = samples - samples[0]
    centered = shifted - sample_mean(shifted)
    return (centered.T @ centered) / samples.shape[0]


def _resolve_lattice_shape(
    dimension: int,
    lattice_shape: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Return a validated periodic-lattice shape."""

    shape = (dimension,) if lattice_shape is None else tuple(lattice_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("lattice_shape must contain positive dimensions")
    if math.prod(shape) != dimension:
        raise ValueError(
            "the product of lattice_shape must equal the sample dimension"
        )
    return shape


def sample_power_spectrum(
    samples: Array,
    lattice_shape: tuple[int, ...] | None = None,
) -> Array:
    """Return the biased Fourier-domain covariance diagonal.

    The Fourier basis is represented by JAX's unitary inverse DFT
    (``ifftn(..., norm="ortho")``); the sign convention does not affect the
    real translation-averaged covariance.  Centering is only across the sample
    axis.  For a multidimensional periodic lattice, ``lattice_shape``
    specifies its translation group; for example ``(m, m)`` represents
    ``Z_m x Z_m`` rather than the flattened group ``Z_{m**2}``.  The returned
    spectrum is flattened to shape ``(d,)``.
    """

    samples = _floating_array(samples)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("samples must have shape (n, d) with n > 0")
    num_samples, dimension = samples.shape
    shape = _resolve_lattice_shape(dimension, lattice_shape)

    shifted = samples - samples[0]
    centered = shifted - sample_mean(shifted)
    fields = centered.reshape((num_samples,) + shape)
    spatial_axes = tuple(range(1, fields.ndim))
    fourier_samples = jnp.fft.ifftn(
        fields,
        axes=spatial_axes,
        norm="ortho",
    )
    spectrum = jnp.mean(
        jnp.real(fourier_samples * jnp.conj(fourier_samples)),
        axis=0,
    )
    return spectrum.reshape((dimension,))


def apply_translation_invariant_spectrum(
    vector: Array,
    spectrum: Array,
    lattice_shape: tuple[int, ...] | None = None,
) -> Array:
    """Apply ``F diag(spectrum) F*`` without forming a dense matrix.

    ``spectrum`` should be produced by :func:`sample_power_spectrum` or obey
    the corresponding nonnegative reversal symmetry for a real covariance.
    """

    vector = _floating_array(vector)
    spectrum = jnp.asarray(spectrum, dtype=vector.dtype)
    dimension = vector.shape[0]
    shape = _resolve_lattice_shape(dimension, lattice_shape)
    spatial_axes = tuple(range(len(shape)))

    field = vector.reshape(shape)
    multiplier = spectrum.reshape(shape)
    transformed = jnp.fft.fftn(field, axes=spatial_axes, norm="ortho")
    result = jnp.fft.ifftn(
        multiplier * transformed,
        axes=spatial_axes,
        norm="ortho",
    )
    return jnp.real(result).reshape((dimension,))


def translation_invariant_covariance(
    spectrum: Array,
    lattice_shape: tuple[int, ...] | None = None,
) -> Array:
    """Materialize the covariance represented by Fourier eigenvalues.

    This allocates a dense ``(d, d)`` array. For large lattices, retain the
    length-``d`` spectrum and use :func:`apply_translation_invariant_spectrum`
    instead.
    """

    spectrum = _floating_array(spectrum)
    dimension = spectrum.shape[0]
    shape = _resolve_lattice_shape(dimension, lattice_shape)
    identity = jnp.eye(dimension, dtype=spectrum.dtype)
    covariance = jax.vmap(
        lambda column: apply_translation_invariant_spectrum(
            column,
            spectrum,
            shape,
        ),
        in_axes=1,
        out_axes=1,
    )(identity)
    return 0.5 * (covariance + covariance.T)


def symmetric_matrix_sqrt(
    A: Array,
    regularize: bool = False,
    epsilon: float = 1e-8,
) -> tuple[Array, Array]:
    """Return the symmetric square root and inverse square root of ``A``.

    If ``regularize`` is true, ``epsilon * I`` is added before the symmetric
    eigendecomposition.
    """

    A = _floating_array(A)
    symmetric_A = 0.5 * (A + A.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(symmetric_A)
    eigenvalue_regularization = jnp.where(
        jnp.asarray(regularize),
        jnp.asarray(epsilon, dtype=A.dtype),
        jnp.asarray(0.0, dtype=A.dtype),
    )
    # Adding epsilon after ``eigh`` is algebraically identical to adding
    # epsilon*I beforehand, but it prevents a small float32 ridge from being
    # rounded away in the matrix entries.
    eigenvalues = eigenvalues + eigenvalue_regularization
    sqrt_eigenvalues = jnp.sqrt(eigenvalues)

    sqrt_A = (eigenvectors * sqrt_eigenvalues) @ eigenvectors.T
    inv_sqrt_A = (eigenvectors * (1.0 / sqrt_eigenvalues)) @ eigenvectors.T
    return sqrt_A, inv_sqrt_A


@partial(
    jax.jit,
    static_argnames=(
        "U",
        "grad_U",
        "num_stages",
        "num_chains",
        "num_ulmc_steps",
    ),
)
def gaussian_cooling(
    rng_key: Array,
    U: PotentialFn,
    grad_U: PotentialFn,
    minimizer: Array,
    alpha: float,
    delta: float,
    cooling_gamma: float,
    smoothness_L: float,
    num_stages: int,
    num_chains: int,
    num_ulmc_steps: int,
    ulmc_step_size: float,
    ulmc_friction_gamma: float,
    *,
    initial_preconditioner: Array | None = None,
    covariance_regularization: float = 0.0,
) -> Array:
    """Run dense Gaussian cooling and return its covariance preconditioner.

    The initial preconditioner defaults to ``smoothness_L**-1 * I``, matching
    the covariance scale used to initialize ULMC. ``initial_preconditioner``
    may instead supply a symmetric positive-definite ``(d, d)`` covariance
    matrix, not its square-root factor.

    ``covariance_regularization`` is a nonnegative, opt-in ridge added to the
    previous covariance while forming and lifting through the next stage.
    It can stabilize inverse roots but does not separately ridge the final
    empirical covariance, which may still be rank-deficient.  Its zero default
    gives the unregularized empirical recurrence.

    At one-based stage ``k = 1, ..., K``, the cooled potential is
    ``U_k(x) = U(x) + cooling_gamma**k * smoothness_L
    * ||x-minimizer||**2 / 2``. ``alpha`` and ``delta`` are theoretical
    tuning metadata: they motivate the caller-supplied sampler parameters but
    do not otherwise enter this fixed-budget recurrence. The transition
    evaluates ``grad_U``; ``U`` is retained for a consistent
    potential/gradient interface.
    """

    # These parameters determine theoretical choices of the supplied stage and
    # ULMC parameters, but do not otherwise enter the specified cooling loop.
    del alpha, delta

    minimizer = _floating_array(minimizer)
    dtype = minimizer.dtype
    dimension = minimizer.shape[0]
    smoothness_L = jnp.asarray(smoothness_L, dtype=dtype)
    cooling_gamma = jnp.asarray(cooling_gamma, dtype=dtype)
    covariance_regularization = jnp.asarray(
        covariance_regularization,
        dtype=dtype,
    )

    if initial_preconditioner is None:
        initial_covariance = (
            jnp.eye(dimension, dtype=dtype) / smoothness_L
        )
    else:
        initial_covariance = jnp.asarray(
            initial_preconditioner,
            dtype=dtype,
        )
        initial_covariance = 0.5 * (
            initial_covariance + initial_covariance.T
        )

    def cooling_stage(carry, stage_index):
        covariance_previous, key = carry
        covariance_sqrt, covariance_inv_sqrt = symmetric_matrix_sqrt(
            covariance_previous,
            regularize=covariance_regularization > 0,
            epsilon=covariance_regularization,
        )
        stage_number = stage_index + 1
        regularization_strength = (
            jnp.power(cooling_gamma, stage_number) * smoothness_L
        )

        def cooled_potential(x):
            displacement = x - minimizer
            return U(x) + 0.5 * regularization_strength * jnp.vdot(
                displacement, displacement
            )

        def cooled_gradient(x):
            return grad_U(x) + regularization_strength * (x - minimizer)

        def preconditioned_potential(x):
            return cooled_potential(covariance_sqrt @ x)

        def preconditioned_gradient(x):
            transformed_x = covariance_sqrt @ x
            return covariance_sqrt @ cooled_gradient(transformed_x)

        preconditioned_minimizer = covariance_inv_sqrt @ minimizer
        key, stage_key = random.split(key)
        samples = _ulmc_impl(
            stage_key,
            preconditioned_potential,
            preconditioned_gradient,
            preconditioned_minimizer,
            ulmc_friction_gamma,
            smoothness_L,
            ulmc_step_size,
            num_ulmc_steps,
            num_chains,
        )

        preconditioned_covariance = sample_covariance(samples)
        covariance = (
            covariance_sqrt
            @ preconditioned_covariance
            @ covariance_sqrt
        )
        covariance = 0.5 * (covariance + covariance.T)
        return (covariance, key), None

    (final_covariance, _), _ = lax.scan(
        cooling_stage,
        (initial_covariance, rng_key),
        xs=jnp.arange(num_stages),
    )
    return final_covariance


@partial(
    jax.jit,
    static_argnames=(
        "U",
        "grad_U",
        "num_stages",
        "num_chains",
        "num_ulmc_steps",
        "lattice_shape",
        "return_spectrum",
    ),
)
def translation_invariant_gaussian_cooling(
    rng_key: Array,
    U: PotentialFn,
    grad_U: PotentialFn,
    minimizer: Array,
    alpha: float,
    delta: float,
    cooling_gamma: float,
    smoothness_L: float,
    num_stages: int,
    num_chains: int,
    num_ulmc_steps: int,
    ulmc_step_size: float,
    ulmc_friction_gamma: float,
    *,
    lattice_shape: tuple[int, ...] | None = None,
    initial_spectrum: Array | None = None,
    covariance_regularization: float = 0.0,
    return_spectrum: bool = False,
) -> Array:
    """Run translation-invariant Gaussian cooling in the Fourier basis.

    ``lattice_shape`` identifies the periodic translation group.  For the
    square lattice with side length ``m``, pass ``(m, m)``; omitting it uses
    the different one-dimensional group ``Z_d``.  Flattening and
    reconstruction use ordinary C/row-major order and unshifted JAX FFT bins.
    Any supplied spectrum must use the same layout and be real, nonnegative,
    and conjugate/reversal symmetric.

    The cumulative spectrum is updated as
    ``s_k = s_{k-1} * tilde_s_k``. ``initial_spectrum`` defaults to the
    constant spectrum ``1 / smoothness_L``.
    ``covariance_regularization`` has the same opt-in stage-transform
    semantics as in :func:`gaussian_cooling`.

    Set ``return_spectrum=True`` to retain the algorithm's O(d) storage and
    FFT-based operator representation.  The default materializes the dense
    covariance so its return type agrees with :func:`gaussian_cooling`.
    """

    del alpha, delta

    minimizer = _floating_array(minimizer)
    dtype = minimizer.dtype
    dimension = minimizer.shape[0]
    shape = _resolve_lattice_shape(dimension, lattice_shape)
    smoothness_L = jnp.asarray(smoothness_L, dtype=dtype)
    cooling_gamma = jnp.asarray(cooling_gamma, dtype=dtype)
    covariance_regularization = jnp.asarray(
        covariance_regularization,
        dtype=dtype,
    )

    if initial_spectrum is None:
        spectrum_initial = (
            jnp.ones((dimension,), dtype=dtype) / smoothness_L
        )
    else:
        spectrum_initial = jnp.asarray(
            initial_spectrum,
            dtype=dtype,
        ).reshape((dimension,))

    def cooling_stage(carry, stage_index):
        spectrum_previous, key = carry
        effective_spectrum = spectrum_previous + jnp.where(
            covariance_regularization > 0,
            covariance_regularization,
            jnp.asarray(0.0, dtype=dtype),
        )
        sqrt_spectrum = jnp.sqrt(effective_spectrum)
        inv_sqrt_spectrum = 1.0 / sqrt_spectrum

        def apply_sqrt(x):
            return apply_translation_invariant_spectrum(
                x,
                sqrt_spectrum,
                shape,
            )

        def apply_inv_sqrt(x):
            return apply_translation_invariant_spectrum(
                x,
                inv_sqrt_spectrum,
                shape,
            )

        stage_number = stage_index + 1
        regularization_strength = (
            jnp.power(cooling_gamma, stage_number) * smoothness_L
        )

        def cooled_potential(x):
            displacement = x - minimizer
            return U(x) + 0.5 * regularization_strength * jnp.vdot(
                displacement,
                displacement,
            )

        def cooled_gradient(x):
            return grad_U(x) + regularization_strength * (x - minimizer)

        def preconditioned_potential(x):
            return cooled_potential(apply_sqrt(x))

        def preconditioned_gradient(x):
            transformed_x = apply_sqrt(x)
            return apply_sqrt(cooled_gradient(transformed_x))

        preconditioned_minimizer = apply_inv_sqrt(minimizer)
        key, stage_key = random.split(key)
        samples = _ulmc_impl(
            stage_key,
            preconditioned_potential,
            preconditioned_gradient,
            preconditioned_minimizer,
            ulmc_friction_gamma,
            smoothness_L,
            ulmc_step_size,
            num_ulmc_steps,
            num_chains,
        )
        stage_spectrum = sample_power_spectrum(samples, shape)
        spectrum = effective_spectrum * stage_spectrum
        return (spectrum, key), None

    (final_spectrum, _), _ = lax.scan(
        cooling_stage,
        (spectrum_initial, rng_key),
        xs=jnp.arange(num_stages),
    )

    if return_spectrum:
        return final_spectrum
    return translation_invariant_covariance(final_spectrum, shape)
