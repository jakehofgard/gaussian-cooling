r"""Fixed-alpha, increasing-kappa lattice phi4 comparison on CPU or GPU.

The target fixes lambda=.5, R=2 and m=6 and sweeps the Laplacian coupling.
All covariances stay in Fourier form: no D-by-D matrix is constructed.
The GPU stage-comparison budget is retained; only ordinary ULMC's step is
capped as requested to avoid frozen-force integrator instability.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from gaussian_cooling_algs import sample_power_spectrum, ulmc_coefficients
from .experiment_phi4_stage_convergence import (
    StageConvergenceResult,
    make_stage_figures,
)
from .lattice_phi4 import (
    SCALABLE_METHODS,
    _fourier_stage_history,
    _make_unpreconditioned_stage_history_call,
    make_lattice_model,
    reference_samples,
    spectral_relative_condition,
    validate_lattice_model,
)
from .phi4_cli import (
    _explicit_cli_destinations,
    _parse_positive_values,
    add_execution_arguments,
    add_reference_arguments,
    validate_common_arguments,
)
from .phi4_plotting import (
    METHOD_STYLES,
    PLOT_LABELS,
    _configure_plot_style,
    _plot_median_iqr,
    save_publication_figures,
)

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


QUARTIC = 0.5
MASS = 6.0
RADIUS = 2.0
DEFAULT_BETAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)


@dataclass
class BetaSweepResult:
    """Metrics have axes (beta, method, stage, repeat), including stage zero.

    ``estimated_spectra`` appends a flattened Fourier-mode axis. Reference
    and GFF spectra have axes (beta, mode). Invalid estimates remain in the
    archive, with NaN metrics and a false entry in ``valid_estimates``.
    """

    betas: np.ndarray
    estimated_spectra: np.ndarray
    relative_conditions: np.ndarray
    gff_relative_conditions: np.ndarray
    gff_ratio_min: np.ndarray
    gff_ratio_max: np.ndarray
    valid_estimates: np.ndarray
    reference_spectra: np.ndarray
    gff_spectra: np.ndarray
    reference_split_conditions: np.ndarray
    reference_gff_conditions: np.ndarray
    reference_gff_ratio_min: np.ndarray
    reference_gff_ratio_max: np.ndarray
    kappas: np.ndarray
    alphas: np.ndarray
    ulmc_step_sizes: np.ndarray
    ulmc_max_mode_radii: np.ndarray
    elapsed_seconds: float


def harmonic_drift_radius(curvature: float, step_size: float, friction: float) -> float:
    """Spectral radius of the actual frozen-force ULMC harmonic transition."""

    decay, position, gradient, _ = ulmc_coefficients(
        friction, step_size, jnp.float64
    )
    drift = np.array([
        [1.0 - float(gradient) * curvature, float(position)],
        [-float(position) * curvature, float(decay)],
    ])
    return float(np.max(np.abs(np.linalg.eigvals(drift))))


def stable_ulmc_step_size(
    smoothness: float, requested: float, friction: float, factor: float = 1.0,
) -> float:
    """Cap h at factor*friction/L, keeping a margin inside harmonic stability.

    Frozen-force ULMC requires h=O(friction/L), not merely O(1/sqrt(L)).
    The conservative factor is restricted to (0,1]. Check the actual drift
    as well, including unusual user-supplied friction/step combinations.
    This controls linear stability; it does not remove discretization bias.
    """

    if any(not np.isfinite(x) or x <= 0 for x in (smoothness, requested, friction)):
        raise ValueError("ULMC smoothness, step size, and friction must be positive.")
    if not np.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("--ulmc-stability-factor must lie in (0,1].")
    step = min(requested, factor * friction / smoothness)
    while harmonic_drift_radius(smoothness, step, friction) >= 1.0:
        step *= 0.5
        if step < np.finfo(float).eps:
            raise ValueError("Unable to resolve a stable ULMC step in float64.")
    return step


def gff_covariance_diagnostics(
    spectrum: np.ndarray, gff_spectrum: np.ndarray, metric_floor: float,
) -> tuple[float, float, float]:
    """Return kappa_rel and extrema of M^(1/2) Sigma_hat M^(1/2).

    A condition of one permits any scalar multiple of the GFF covariance.
    The unfloored extrema retain the scale information needed to see this.
    """

    spectrum = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    gff_spectrum = np.asarray(gff_spectrum, dtype=np.float64).reshape(-1)
    if spectrum.shape != gff_spectrum.shape:
        raise ValueError("Estimate and GFF spectra must have matching shapes.")
    if any(np.any(~np.isfinite(s)) or np.any(s <= 0) for s in (spectrum, gff_spectrum)):
        raise FloatingPointError("Covariance spectra must be finite and strictly positive.")
    ratios = spectrum / gff_spectrum
    return (
        spectral_relative_condition(spectrum, gff_spectrum, metric_floor),
        float(np.min(ratios)),
        float(np.max(ratios)),
    )


def reference_gff_variance_calibration(args: argparse.Namespace) -> float:
    """Exact discrete harmonic calibration for the reference initialization.

    Replace the whitened target by N(0,I), retain the reference initial
    variance 1/L_ref=alpha and all integrator settings, and propagate its
    2-by-2 position/momentum covariance. This is a bias diagnostic, not a
    correction applied to the nonlinear target or the reported metrics.
    """

    decay, position, gradient, noise = ulmc_coefficients(
        args.friction, args.reference_step_size, jnp.float64
    )
    drift = np.array([[1.0 - float(gradient), float(position)],
                      [-float(position), float(decay)]])
    covariance = np.diag([MASS / (MASS + 3 * QUARTIC * RADIUS**2), 1.0])
    noise = np.asarray(noise)
    for _ in range(args.reference_steps):
        covariance = drift @ covariance @ drift.T + noise @ noise.T
    return float(covariance[0, 0])


def run_beta_sweep(args: argparse.Namespace) -> BetaSweepResult:
    """Run paired adaptive methods and continuous, equal-query plain ULMC."""

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    betas = np.asarray(args.betas, dtype=float)
    shape = (len(betas), len(SCALABLE_METHODS), args.stages + 1, args.repeats)
    spectra = np.full(shape + (args.side**2,), np.nan, dtype=args.dtype)
    conditions, gff_conditions, ratio_min, ratio_max = (
        np.full(shape, np.nan) for _ in range(4)
    )
    valid = np.zeros(shape, dtype=bool)
    references = np.empty((len(betas), args.side**2), dtype=args.dtype)
    gff_spectra = np.empty_like(references, dtype=np.float64)
    split, ref_gff, ref_min, ref_max, kappas, alphas, ulmc_steps, radii = (
        np.empty(len(betas)) for _ in range(8)
    )
    root_key = random.PRNGKey(args.seed)
    started = time.perf_counter()
    for beta_index, beta in enumerate(betas):
        model = make_lattice_model(args.side, float(beta), QUARTIC, MASS, RADIUS, dtype)
        validate_lattice_model(model)
        # Key by the floating-point beta value, so extending a grid preserves runs.
        beta_bits = np.asarray([beta], dtype=np.float64).view(np.uint32)
        key = random.fold_in(random.fold_in(root_key, int(beta_bits[0])), int(beta_bits[1]))
        kappas[beta_index] = model.global_smoothness_bound / model.strong_convexity
        alphas[beta_index] = model.design_conditioning_alpha
        gff = (1.0 / (model.laplacian_spectrum + MASS)).reshape(-1)
        gff_spectra[beta_index] = gff
        plain_args = argparse.Namespace(**vars(args))
        plain_args.step_size = stable_ulmc_step_size(
            model.design_smoothness, args.step_size, args.friction,
            args.ulmc_stability_factor,
        )
        ulmc_steps[beta_index] = plain_args.step_size
        radii[beta_index] = harmonic_drift_radius(
            model.design_smoothness, plain_args.step_size, args.friction
        )
        print(
            f"beta={beta:g}, side={args.side}, kappa={kappas[beta_index]:.6g}, "
            f"alpha={alphas[beta_index]:g}, h_ULMC={plain_args.step_size:.6g}",
            flush=True,
        )
        reference = reference_samples(random.fold_in(key, 10_003), model, args)
        ref = np.asarray(sample_power_spectrum(reference, model.lattice_shape))
        references[beta_index] = ref
        midpoint = args.reference_chains // 2
        halves = [np.asarray(sample_power_spectrum(part, model.lattice_shape))
                  for part in (reference[:midpoint], reference[midpoint:])]
        split[beta_index] = spectral_relative_condition(*halves, args.metric_floor)
        ref_gff[beta_index], ref_min[beta_index], ref_max[beta_index] = (
            gff_covariance_diagnostics(ref, gff, args.metric_floor)
        )
        del reference
        plain_call = _make_unpreconditioned_stage_history_call(
            model, plain_args, include_raw_covariances=False
        )
        for repeat in range(args.repeats):
            method_key = random.fold_in(key, 60_000 + repeat)
            histories = (
                _fourier_stage_history(method_key, model, args, use_cooling=True),
                _fourier_stage_history(method_key, model, args, use_cooling=False),
                np.asarray(plain_call(random.fold_in(method_key, 70_001))),
            )
            for method_index, history in enumerate(histories):
                spectra[beta_index, method_index, :, repeat] = history
                for stage, spectrum in enumerate(history):
                    index = (beta_index, method_index, stage, repeat)
                    try:
                        gff_conditions[index], ratio_min[index], ratio_max[index] = (
                            gff_covariance_diagnostics(spectrum, gff, args.metric_floor)
                        )
                        conditions[index] = spectral_relative_condition(
                            spectrum, ref, args.metric_floor
                        )
                        valid[index] = True
                    except FloatingPointError:
                        warnings.warn(
                            f"Invalid covariance at beta={beta:g}, "
                            f"method={SCALABLE_METHODS[method_index]}, "
                            f"stage={stage}, repeat={repeat}; saved as an invalid estimate.",
                            RuntimeWarning,
                        )
            print(f"  completed repeat {repeat + 1}/{args.repeats}", flush=True)
        # Each beta closes over a different model. Release compiled executables
        # after synchronization to avoid accumulating GPU compilation caches.
        del histories, plain_call
        jax.clear_caches()
    return BetaSweepResult(
        betas, spectra, conditions, gff_conditions, ratio_min, ratio_max, valid,
        references, gff_spectra, split, ref_gff, ref_min, ref_max, kappas, alphas,
        ulmc_steps, radii, time.perf_counter() - started,
    )


def make_beta_figures(result: BetaSweepResult, args: argparse.Namespace) -> dict[str, plt.Figure]:
    """Reuse the existing phi4 stage plotter and method comparison conventions."""

    _configure_plot_style()
    figures = {}
    for beta_index, beta in enumerate(result.betas):
        stage_args = argparse.Namespace(**vars(args), beta=float(beta),
                                        quartic=QUARTIC, mass=MASS, radius=RADIUS)
        stage_result = StageConvergenceResult(
            np.array([args.side]),
            {args.side: {method: result.relative_conditions[beta_index, index]
                         for index, method in enumerate(SCALABLE_METHODS)}},
            {args.side: "This sweep compares the three Fourier methods."},
            result.elapsed_seconds,
        )
        stage_figures = make_stage_figures(stage_result, stage_args)
        beta_label = f"{beta:.8g}".replace(".", "p")
        figures[f"beta_stage_comparison_d{args.side}_beta{beta_label}"] = next(iter(stage_figures.values()))
    subtitle = (
        rf"$d={args.side},\ \lambda={QUARTIC:g},\ m={MASS:g},\ R={RADIUS:g}"
        rf";\ n={args.chains},\ N={args.steps},\ K={args.stages}$"
    )
    for name, metrics, title in (
        ("beta_comparison", result.relative_conditions,
         r"Preconditioner quality vs. $\beta$"),
        ("beta_gff_comparison", result.gff_relative_conditions,
         r"Relative condition number $\kappa_{\mathrm{rel}}$ vs. $\beta$"),
    ):
        figure, ax = plt.subplots(figsize=(6.5, 4.9), constrained_layout=True)
        for method_index, method in enumerate(SCALABLE_METHODS):
            color, marker, linestyle = METHOD_STYLES[method]
            # Keep invalid beta rows as NaNs so lines do not bridge failures.
            values = metrics[:, method_index, -1, :]
            valid_rows = np.any(np.isfinite(values), axis=1)
            valid_indices = np.flatnonzero(valid_rows)
            segments = np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) > 1) + 1)
            for segment_index, segment in enumerate(segments):
                if segment.size:
                    _plot_median_iqr(
                        ax, result.betas[segment], values[segment], color=color,
                        marker=marker, linestyle=linestyle,
                        label=PLOT_LABELS[method] if segment_index == 0 else "_nolegend_",
                    )
        ax.axhline(1.0, color="#222222", linestyle=":", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(result.betas, [f"{beta:g}" for beta in result.betas])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"Laplacian coupling $\beta$")
        ax.set_ylabel(
            r"$\kappa_{\mathrm{rel}}(M_\beta^{-1},\widehat\Sigma)$"
            if name == "beta_gff_comparison"
            else r"Relative condition number $\kappa_{\mathrm{rel}}$"
        )
        ax.set_title(title + "\n" + subtitle)
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24),
                  ncols=2, fontsize=7.4)
        figures[f"{name}_d{args.side}"] = figure
    return figures


def save_beta_sweep_data(result: BetaSweepResult, args: argparse.Namespace) -> tuple[Path, Path]:
    """Archive every repeat and stage plus a readable final-stage summary."""

    output = args.output.expanduser().resolve()
    stem = output.with_suffix("") if output.suffix else output
    stem.parent.mkdir(parents=True, exist_ok=True)
    base = stem.with_name(f"{stem.name}_beta_sweep_d{args.side}")
    data_path = base.with_name(base.name + "_data.npz")
    csv_path = base.with_name(base.name + "_summary.csv")
    metadata = dict(vars(args), output=str(args.output), quartic=QUARTIC,
                    mass=MASS, radius=RADIUS, backend=jax.default_backend(),
                    devices=[str(device) for device in jax.devices()],
                    jax_version=jax.__version__, numpy_version=np.__version__,
                    spectrum_axes=["beta", "method", "stage", "repeat", "mode"],
                    gradient_evaluations_per_method=args.chains * args.steps * args.stages,
                    reference_gff_variance_calibration=reference_gff_variance_calibration(args),
                    reference_integration_time=args.reference_steps * args.reference_step_size)
    np.savez_compressed(data_path, **vars(result), methods=np.asarray(SCALABLE_METHODS),
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    fields = ["beta", "method", "repeat", "valid", "kappa", "alpha", "step_size",
              "integration_time_per_chain", "gradient_evaluations", "relative_condition",
              "gff_relative_condition", "gff_ratio_min", "gff_ratio_max",
              "reference_split_condition"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for beta_index, beta in enumerate(result.betas):
            common = dict(beta=beta, kappa=result.kappas[beta_index], alpha=result.alphas[beta_index],
                          reference_split_condition=result.reference_split_conditions[beta_index])
            for method_index, method in enumerate(SCALABLE_METHODS):
                h = result.ulmc_step_sizes[beta_index] if method_index == 2 else args.step_size
                for repeat in range(args.repeats):
                    index = (beta_index, method_index, -1, repeat)
                    writer.writerow(dict(common, method=method, repeat=repeat,
                        valid=result.valid_estimates[index], step_size=h,
                        integration_time_per_chain=h * args.steps * args.stages,
                        gradient_evaluations=args.chains * args.steps * args.stages,
                        relative_condition=result.relative_conditions[index],
                        gff_relative_condition=result.gff_relative_conditions[index],
                        gff_ratio_min=result.gff_ratio_min[index], gff_ratio_max=result.gff_ratio_max[index]))
            writer.writerow(dict(common, method="Independent reference", repeat=0, valid=True,
                step_size=args.reference_step_size,
                integration_time_per_chain=args.reference_step_size * args.reference_steps,
                gradient_evaluations=args.reference_chains * args.reference_steps,
                relative_condition=1.0, gff_relative_condition=result.reference_gff_conditions[beta_index],
                gff_ratio_min=result.reference_gff_ratio_min[beta_index],
                gff_ratio_max=result.reference_gff_ratio_max[beta_index]))
    return data_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep beta with lambda=.5, R=2, m=6 and the GPU comparison budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--side", type=int, default=100)
    parser.add_argument("--betas", type=_parse_positive_values, default=list(DEFAULT_BETAS))
    for name, default in (("chains", 512), ("steps", 256), ("stages", 12), ("repeats", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("cooling-gamma", .05), ("delta", .25), ("friction", 1.),
                          ("step-size", .1), ("covariance-ridge", 0.), ("metric-floor", 1e-10)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--ulmc-stability-factor", type=float, default=1.0,
                        help="Ordinary ULMC h cap is this factor times friction/L; must be in (0,1].")
    add_reference_arguments(parser)
    add_execution_arguments(parser, default_output=Path(__file__).resolve().parents[1] / "figures" / "phi4_beta")
    parser.set_defaults(
        reference_chains=2048,
        reference_steps=2048,
        reference_step_size=.001,
    )
    return parser


def apply_presets(args: argparse.Namespace, explicit: set[str]) -> None:
    if args.quick and args.gpu:
        raise ValueError("--quick and --gpu are mutually exclusive presets.")
    if args.quick:
        for name, value in dict(side=4, betas=[1., 100., 10_000.], chains=16,
                                steps=8, stages=3, repeats=1,
                                reference_chains=32, reference_steps=32).items():
            if name not in explicit:
                setattr(args, name, value)


def validate_arguments(args: argparse.Namespace) -> None:
    validate_common_arguments(args)
    if args.side < 2:
        raise ValueError("--side must be at least two.")
    if not np.isfinite(args.ulmc_stability_factor) or not 0 < args.ulmc_stability_factor <= 1:
        raise ValueError("--ulmc-stability-factor must lie in (0,1].")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        apply_presets(args, _explicit_cli_destinations(parser, arguments))
        validate_arguments(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    calibration = reference_gff_variance_calibration(args)
    print(f"Backend: {jax.default_backend()}; devices: {jax.devices()}", flush=True)
    print("Fixed lambda=.5, R=2, m=6; alpha=.5. Equal gradient budgets; "
          "ordinary ULMC uses h=min(h_requested, factor*friction/L).", flush=True)
    print(f"Reference harmonic variance calibration: {calibration:.6g} (equilibrium=1). "
          "GFF comparisons are finite-sample covariance diagnostics, not a Gaussianity certificate.", flush=True)
    if abs(calibration - 1.) > .05:
        warnings.warn("The reference budget has appreciable harmonic transient/step-size bias. "
                      "For collapse analysis, check convergence with longer reference runs "
                      "and more reference chains.", RuntimeWarning)
    result = run_beta_sweep(args)
    data_path, csv_path = save_beta_sweep_data(result, args)
    figures = make_beta_figures(result, args)
    paths = save_publication_figures(figures, args.output)
    for figure in figures.values():
        plt.close(figure)
    for path in (*paths.values(), data_path, csv_path):
        print(f"Saved: {path}")
    print(f"Valid sampled estimates: {np.count_nonzero(result.valid_estimates[:, :, 1:])}"
          f"/{result.valid_estimates[:, :, 1:].size}; elapsed including compilation: "
          f"{result.elapsed_seconds:.2f}s")


if __name__ == "__main__":
    main()
