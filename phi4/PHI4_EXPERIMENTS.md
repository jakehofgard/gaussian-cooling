# Lattice $\phi^4$ experiments

This directory contains the lattice experiments for **Efficient Mass Matrix
Estimation with Gaussian Cooling**. They test how well the methods estimate
covariances, how much sampling they need, and how the resulting estimates
help rescale the sampler's updates. Such a rescaling matrix is called a
preconditioner. The model is a scalar field on a square lattice whose edges
wrap around. A lattice of side length $d$ has $D=d^2$ variables.

Follow the installation steps in the [project README](../README.md), then run
the commands below from the **project root**. Each experiment has a `--quick`
option that uses smaller lattices or fewer samples to check that the code and
plots run.

`--output-prefix` sets the start of each output filename. The experiment adds
the plot name and creates the destination directory if needed. All plots are
PDFs; the hardness map and budget study also save numerical results as NumPy
`.npz` files.

## Convergence across stages

[Stage convergence](experiment_phi4_stage_convergence.py) compares the learned
covariance after each round of sampling and adaptation. Its plots show the
relative condition number; values closer to 1 indicate a better match to the
reference covariance.

```bash
# Small test run
python -m phi4.experiment_phi4_stage_convergence --quick \
  --output-prefix results/phi4_stages_quick

# Default experiment
python -m phi4.experiment_phi4_stage_convergence \
  --output-prefix figures/phi4_stages

# Larger lattices on a GPU
python -m phi4.experiment_phi4_stage_convergence --gpu \
  --output-prefix figures/phi4_stages_gpu
```

The default experiment uses sides 10 and 100, with 12 stages, 2,048 chains,
2,048 steps per chain per stage, and three repeats. The GPU preset uses sides
256 and 512, 512 chains, 256 steps per stage, and one repeat. Both produce one
`_stage_comparison_d{d}.pdf` per lattice size.

To study a smaller mass or a quartic potential with no quadratic continuation:

```bash
python -m phi4.experiment_phi4_stage_convergence --mass .001 \
  --output-prefix figures/phi4_stages_small_mass

python -m phi4.experiment_phi4_stage_convergence --radius inf \
  --output-prefix figures/phi4_stages_quartic
```

These options can be combined. Add `--quick` for a short test or `--gpu` for
the larger lattice preset. Explicit settings, such as `--mass .001` or
`--sides 8,16`, take precedence over preset values. The meaning of the radius
and its tuning parameter is explained below.

## Mixing and autocorrelation

[Diagnostics](experiment_phi4_diagnostics.py) first learns a covariance with
Gaussian cooling, then uses it to run a sampling chain. It measures integrated
autocorrelation time (IAT): roughly, how many sampling steps correspond to one
independent measurement. It also checks whether the estimated field mean at
each site is consistent with the model's zero mean.

```bash
python -m phi4.experiment_phi4_diagnostics --quick \
  --output-prefix results/phi4_diagnostics_quick

python -m phi4.experiment_phi4_diagnostics \
  --output-prefix figures/phi4_diagnostics
```

The default uses side 100, discards 10,000 initial steps, and retains 100,000
samples. Change these with `--side`, `--trajectory-burnin`, and
`--trajectory-samples`. Adding `--gpu` selects the diagnostic GPU preset,
which uses side 64 and a smaller learning budget.

The two outputs are `_two_point_iat_d{d}.pdf` and
`_ergodicity_site_mean_z_scores_d{d}.pdf`. Short chains may leave many sites
without reliable estimates; the site-mean plot marks these in gray.

## Sampling budget heatmap

[Budget scaling](experiment_phi4_budget_scaling.py) runs Gaussian cooling for
every combination of a chosen list of chain counts and a chosen list of step
counts. The heatmap shows covariance quality for each combination.

```bash
python -m phi4.experiment_phi4_budget_scaling --quick \
  --output-prefix results/phi4_budget_quick

python -m phi4.experiment_phi4_budget_scaling --gpu \
  --output-prefix figures/phi4_budget
```

The full experiment uses side 100, 12 stages, and five repeats. It tests chain
counts `64,128,256,512,1024` against step counts `32,64,128,256,512`, for 25
combinations. Set different lists with `--chain-counts` and `--step-counts`.
This is a substantial computation; `--quick` reduces it to four combinations
on a side-4 lattice. Here, `--gpu` requires an available GPU but does not change
the experiment's settings. Omit it to use JAX's available device.

Outputs are `_budget_heatmap_d{d}.pdf` and `_budget_grid_d{d}_data.npz`. The
data file includes every repeat, the learned covariances in Fourier form,
timings, and reference settings. The first repeat's timing includes compilation.

## Other implemented experiments

| Experiment | What it tests |
| --- | --- |
| [Lattice-size scaling](experiment_phi4_scaling.py) | Covariance quality and runtime as the lattice grows. Timings exclude the initial compilation. Dense-matrix methods are included where feasible. |
| [Two-point correlator](experiment_phi4_correlator.py) | How correlations with the center site change with distance, under each available preconditioner. |
| [Parameter sweeps](experiment_phi4_parameter_sweeps.py) | Changes in quality when varying the quartic coupling, mass, or radius one at a time. The default radius sweep includes the unmodified quartic potential. |
| [Hardness map](experiment_phi4_hardness_map.py) | Quality across pairs of mass and quartic coupling, with the same sampling budget throughout. It fixes the neighbor coupling at 2 and the radius at 4. |

Run a small version of each with:

```bash
python -m phi4.experiment_phi4_scaling --quick \
  --output-prefix results/phi4_scaling
python -m phi4.experiment_phi4_correlator --quick \
  --output-prefix results/phi4_correlator
python -m phi4.experiment_phi4_parameter_sweeps --quick \
  --output-prefix results/phi4_sweeps
python -m phi4.experiment_phi4_hardness_map --quick \
  --output-prefix results/phi4_hardness
```

Remove `--quick` to run each experiment's default settings. Every script prints
the paths to its saved figures. The hardness map also saves
`_hardness_map_data.npz`, containing each repeat and its reference settings.

Defaults differ between experiments. Use `--help` to see the available controls:

```bash
python -m phi4.experiment_phi4_stage_convergence --help
```

For most experiments, `--chains`, `--steps`, `--stages`, and `--repeats` control
the sampling budget. `--reference-*` options control a separate run used to
judge covariance quality. `--seed` sets the random seed and `--dtype` chooses
`float32` or `float64`. The sweep and map scripts provide lists of parameter
values instead of a single target.

## Running several experiments together

The following command runs small versions of scaling, stage convergence,
diagnostics, correlators, and parameter sweeps:

```bash
python -m phi4 --quick --output-prefix results/phi4_suite
```

Remove `--quick` for the default experiments, or select a subset:

```bash
python -m phi4 --only scaling --only stages --quick \
  --output-prefix results/phi4_selected
```

The available names are `scaling`, `stages`, `diagnostics`, `correlator`,
`sweeps`, `hardness`, and `budget`. Hardness maps and budget studies run only
when selected. Use the individual scripts when changing target parameters
or sampling budgets.

GPU runs require JAX to recognize an available GPU. `python -m phi4 --gpu`
runs scaling and stages only: sides 64, 128, 256, and 512 for scaling, and
sides 256 and 512 for stages. Diagnostics and correlators have their own GPU
presets when run individually. Hardness and budget studies use `--gpu` only
to require a GPU. Parameter sweeps have no `--gpu` flag. `--quick` and `--gpu`
cannot be combined.

## Model and numerical details

The potential is

```math
U(\phi)=\frac12\phi^\mathsf T\Delta_\beta\phi+\sum_x u_R(\phi_x),
\qquad
u_R(z)=\lambda z^4/4+mz^2/2\quad (|z|\leq R).
```

Here $\Delta_\beta$ couples neighboring sites, $\lambda$ is the quartic
coupling, and $m$ is the mass. Outside $[-R,R]$, the single-site potential is
continued as a quadratic. This gives curvature bounds $\mu=m$ and
$L=\lambda_{\max}(\Delta_\beta)+m+3\lambda R^2$ for finite $R$.
`--radius inf` keeps the quartic potential everywhere. In that case,
`--cooling-design-radius` supplies a finite scale for tuning the sampler;
it does not change the target or bound its curvature everywhere.

The comparisons use underdamped Langevin Monte Carlo (ULMC) to estimate
covariances. Three methods use the lattice's translation symmetry: covariance
depends on the separation between sites, so it can be stored and applied
efficiently using Fourier transforms. They are Gaussian cooling, repeated
empirical covariance updates without cooling, and a covariance estimated by
averaging over translations of ordinary ULMC samples. A fourth method keeps
the full empirical covariance matrix when the chain count exceeds $D$ and
the configured size limit allows it. Scaling also compares dense Gaussian
cooling at feasible sizes. The budget heatmap tests only the Fourier version
of Gaussian cooling.

Reference covariances come from separate ULMC samples, using
$(\Delta_\beta+mI)^{-1}$ as the preconditioner and averaging over translations.
The [project README](../README.md) defines the relative condition number used
to compare an estimate with this reference.

Diagnostic IATs are measured for products of the field at the origin and at
each other site. Site-mean checks use the final half of the retained chain
and standard errors adjusted for autocorrelation. The correlator experiment
subtracts the product of the estimated means, averages sites at the same
periodic distance from the center, and shows 95% uncertainty bands where
the IAT estimates are reliable. These bands account for sampling with the
learned covariance fixed; they do not include variation from learning that
covariance. Longer chains and smaller integration steps help check sampling
error and errors caused by the finite step size.

## Where to find the code

| File | Contents |
| --- | --- |
| [lattice_phi4.py](lattice_phi4.py) | Lattice potential and gradient, sampler calls, covariance references, and quality measures. |
| [phi4_cli.py](phi4_cli.py) | Shared command-line options, checks, and presets. Some experiments define their own defaults. |
| [phi4_plotting.py](phi4_plotting.py) | Shared plot styles, labels, uncertainty bands, and PDF saving. |
| `experiment_phi4_*.py` | The sampling, measurements, plots, and command-line options for each experiment. |
| [experiment_truncated_phi4.py](experiment_truncated_phi4.py), [__main__.py](__main__.py) | Selection of experiments for `python -m phi4`. |

The core ULMC and cooling algorithms are in
[../gaussian_cooling_algs.py](../gaussian_cooling_algs.py).
