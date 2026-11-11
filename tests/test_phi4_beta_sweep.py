"""Mathematical and configuration checks for the fixed-alpha beta experiment.

Run from the project root with:
    python -m unittest discover -s tests -v
The end-to-end CPU smoke run is documented separately in PHI4_EXPERIMENTS.md.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import unittest

import jax.numpy as jnp
import numpy as np

from gaussian_cooling_algs import translation_invariant_covariance
from phi4 import experiment_phi4_beta_sweep as sweep
from phi4 import experiment_phi4_stage_convergence as stages
from phi4.lattice_phi4 import (
    dense_relative_condition,
    make_lattice_model,
    validate_lattice_model,
)
from phi4.phi4_cli import _explicit_cli_destinations


class BetaSweepMathematicsTests(unittest.TestCase):
    def test_conditioning_regime_on_the_requested_lattice(self):
        kappas = []
        for beta in sweep.DEFAULT_BETAS:
            with self.subTest(beta=beta):
                model = make_lattice_model(100, beta, .5, 6., 2., jnp.float64)
                self.assertEqual(model.design_conditioning_alpha, .5)
                condition = model.global_smoothness_bound / model.strong_convexity
                self.assertAlmostEqual(condition, (8 * beta + 12) / 6)
                kappas.append(condition)
        self.assertTrue(np.all(np.diff(kappas) > 0))

    def test_large_coupling_float32_validation_accepts_roundoff_not_wrong_spectra(self):
        # FFT roundoff grows with beta even when the analytic spectrum is exact.
        # A normwise tolerance must still reject an actual error in that spectrum.
        for side in (6, 100):
            with self.subTest(side=side):
                model = make_lattice_model(side, 10000., .5, 6., 2., jnp.float32)
                validate_lattice_model(model)
                corrupted = replace(model, laplacian_spectrum=1.01 * model.laplacian_spectrum)
                with self.assertRaisesRegex(AssertionError, "Periodic Laplacian spectrum error"):
                    validate_lattice_model(corrupted)

    def test_gff_relative_condition_keeps_scale_information_separate(self):
        gff = np.array([.5, .25, .125, .0625])
        condition, lower, upper = sweep.gff_covariance_diagnostics(.3 * gff, gff, 1e-10)
        self.assertAlmostEqual(condition, 1.)
        self.assertAlmostEqual(lower, .3)
        self.assertAlmostEqual(upper, .3)
        estimate = gff * np.array([.8, 1., 1.2, 1.6])
        forward = sweep.gff_covariance_diagnostics(estimate, gff, 1e-10)
        reverse = sweep.gff_covariance_diagnostics(gff, estimate, 1e-10)
        scaled = sweep.gff_covariance_diagnostics(3 * estimate, 7 * gff, 1e-10)
        self.assertAlmostEqual(forward[0], 2.)
        self.assertAlmostEqual(forward[0], reverse[0])
        self.assertAlmostEqual(forward[0], scaled[0])
        np.testing.assert_allclose(forward[1:], [.8, 1.6])

    def test_fourier_metric_agrees_with_dense_generalized_covariance(self):
        # Two real symmetric periodic covariances with distinct Fourier ratios.
        model = make_lattice_model(4, 2., .5, 6., 2., jnp.float64)
        gff = (1. / (model.laplacian_spectrum + model.mass)).reshape(-1)
        phase = 2 * np.pi * np.arange(4) / 4
        ratios = 1. + .2 * np.cos(phase)[:, None] + .1 * np.cos(phase)[None, :]
        estimate = gff * ratios.reshape(-1)
        dense_estimate = np.asarray(translation_invariant_covariance(jnp.asarray(estimate), (4, 4)))
        dense_reference = np.asarray(translation_invariant_covariance(jnp.asarray(gff), (4, 4)))
        expected = dense_relative_condition(dense_estimate, dense_reference, 0.)
        actual = sweep.gff_covariance_diagnostics(estimate, gff, 1e-10)[0]
        self.assertAlmostEqual(actual, expected, places=12)

    def test_nonpositive_or_nonfinite_spectra_are_rejected(self):
        for spectrum in ([1., 0.], [1., -1.], [1., np.nan], [1., np.inf]):
            with self.subTest(spectrum=spectrum), self.assertRaises(FloatingPointError):
                sweep.gff_covariance_diagnostics(np.array(spectrum), np.ones(2), 1e-10)
        with self.assertRaises(ValueError):
            sweep.gff_covariance_diagnostics(np.ones(2), np.ones(3), 1e-10)

    def test_stability_cap_controls_all_harmonic_modes_across_beta_grid(self):
        steps = []
        for beta in sweep.DEFAULT_BETAS:
            smoothness = 8 * beta + 12
            step = sweep.stable_ulmc_step_size(smoothness, .1, 1.)
            self.assertGreater(step, 0.)
            self.assertLessEqual(step, .1)
            # Sample the full Hessian spectral interval, including the zero mode.
            for curvature in np.geomspace(6., smoothness, 12):
                with self.subTest(beta=beta, curvature=curvature):
                    self.assertLess(sweep.harmonic_drift_radius(curvature, step, 1.), 1.)
            steps.append(step)
        self.assertTrue(np.all(np.diff(steps) < 0))
        self.assertGreater(sweep.harmonic_drift_radius(80006., .1, 1.), 1.)
        self.assertAlmostEqual(steps[-1], 1. / 80012.)

    def test_reference_calibration_detects_shared_initialization_bias(self):
        args = sweep.build_parser().parse_args([])
        short = sweep.reference_gff_variance_calibration(args)
        self.assertAlmostEqual(short, .5295369125, places=8)
        args.reference_steps = 10000
        long = sweep.reference_gff_variance_calibration(args)
        self.assertLess(abs(long - 1.), .001)
        self.assertGreater(long, short)


class BetaSweepConfigurationTests(unittest.TestCase):
    def test_full_budget_matches_existing_gpu_stage_comparison(self):
        args = sweep.build_parser().parse_args(["--gpu"])
        sweep.apply_presets(args, {"gpu"})
        baseline = stages.build_parser().parse_args(["--gpu"])
        stages.apply_presets(baseline, {"gpu"})
        names = ("chains", "steps", "stages", "repeats", "cooling_gamma", "delta",
                 "friction", "step_size", "covariance_ridge", "metric_floor",
                 "reference_chains", "reference_steps", "reference_step_size", "dtype", "seed")
        for name in names:
            with self.subTest(setting=name):
                self.assertEqual(getattr(args, name), getattr(baseline, name))
        self.assertEqual(args.side, 100)
        self.assertEqual(args.betas[0], 1.)
        self.assertEqual(args.betas[-1], 10000.)

    def test_quick_preset_preserves_explicit_arguments(self):
        argv = ["--quick", "--side=6", "--betas", "10000,1", "--chains", "25",
                "--steps=11", "--reference-steps", "40", "--dtype", "float64"]
        parser = sweep.build_parser()
        args = parser.parse_args(argv)
        sweep.apply_presets(args, _explicit_cli_destinations(parser, argv))
        sweep.validate_arguments(args)
        self.assertEqual(args.side, 6)
        self.assertEqual(args.betas, [1., 10000.])
        self.assertEqual(args.chains, 25)
        self.assertEqual(args.steps, 11)
        self.assertEqual(args.reference_steps, 40)
        self.assertEqual(args.dtype, "float64")
        self.assertEqual(args.stages, 3)

    def test_invalid_budgets_and_stability_parameters(self):
        parser = sweep.build_parser()
        for argv in (["--side", "1"], ["--chains", "1"], ["--steps", "0"],
                     ["--reference-chains", "3"], ["--reference-step-size", "0"],
                     ["--ulmc-stability-factor", "0"], ["--ulmc-stability-factor", "1.1"],
                     ["--ulmc-stability-factor", "nan"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                sweep.validate_arguments(parser.parse_args(argv))
        args = parser.parse_args(["--gpu", "--quick"])
        with self.assertRaises(ValueError):
            sweep.apply_presets(args, {"gpu", "quick"})
        for grid in ("0,1", "1,1", "1,nan", "-1,2", ""):
            with self.subTest(grid=grid), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--betas", grid])


if __name__ == "__main__":
    unittest.main()
