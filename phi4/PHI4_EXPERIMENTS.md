# Lattice $\phi^4$ experiments

These experiments learn and evaluate covariance preconditioners for a scalar
field on a periodic square lattice. The side length is $d$ and the ambient
dimension is $D=d^2$. See the [project README](../README.md) for installation
and the relative-condition metric.

Run every command here from the **project root**, the directory containing
`gaussian_cooling_algs.py`.

## Choose an experiment

| Module | What it measures | Main output suffixes |
| --- | --- | --- |
| [experiment_phi4_scaling](experiment_phi4_scaling.py) | Preconditioner quality and post-JIT runtime versus lattice size. | `_preconditioner_quality.pdf`, `_covariance_estimation_cost.pdf`; ancillary dense comparisons. |
| [experiment_phi4_stage_convergence](experiment_phi4_stage_convergence.py) | Quality versus cumulative cooling/adaptation stage at selected sizes. | `_stage_comparison_d{d}.pdf` |
| [experiment_phi4_diagnostics](experiment_phi4_diagnostics.py) | Two-point integrated autocorrelation times (IATs) and site-mean diagnostics after cooling. | `_two_point_iat_d{d}.pdf`, `_ergodicity_site_mean_z_scores_d{d}.pdf` |
| [experiment_phi4_correlator](experiment_phi4_correlator.py) | Center-site connected correlator decay under each feasible preconditioner. | `_two_point_correlator_decay_d{d}.pdf` |
| [experiment_phi4_parameter_sweeps](experiment_phi4_parameter_sweeps.py) | Separate quartic-coupling, mass, and truncation-radius sweeps. | `_parameter_sweep_lambda.pdf`, `_parameter_sweep_mass.pdf`, `_parameter_sweep_radius.pdf`, `_truncation_to_quartic_covariance.pdf` |
| [experiment_phi4_hardness_map](experiment_phi4_hardness_map.py) | Fixed-budget quality and improvement over the baseline across mass/coupling pairs. | `_hardness_absolute_d{d}.pdf`, `_hardness_adaptive_gain_d{d}.pdf`, `_hardness_map_data.npz` |
| [experiment_phi4_budget_scaling](experiment_phi4_budget_scaling.py) | Cooling quality over a Cartesian grid of chain counts and steps per stage. | `_budget_heatmap_d{d}.pdf`, `_budget_grid_d{d}_data.npz` |

Each module runs independently. These reduced commands exercise every study;
omit `--quick` for its full default configuration:

```bash
python -m phi4.experiment_phi4_scaling --quick \
  --output-prefix results/phi4_scaling
python -m phi4.experiment_phi4_stage_convergence --quick \
  --output-prefix results/phi4_stages
python -m phi4.experiment_phi4_diagnostics --quick \
  --output-prefix results/phi4_diagnostics
python -m phi4.experiment_phi4_correlator --quick \
  --output-prefix results/phi4_correlator
python -m phi4.experiment_phi4_parameter_sweeps --quick \
  --output-prefix results/phi4_sweeps
python -m phi4.experiment_phi4_hardness_map --quick \
  --output-prefix results/phi4_hardness
python -m phi4.experiment_phi4_budget_scaling --quick \
  --output-prefix results/phi4_budget
```

Defaults are specific to each experiment. Use its `--help` and `build_parser`
function to inspect the options and defaults. Common controls include:

| Controls | Meaning |
| --- | --- |
| `--side` or `--sides` | Lattice side length(s), depending on the experiment. |
| `--beta`, `--quartic`, `--mass`, `--radius` | Target parameters; the sweep and map modules expose their own grids. |
| `--chains`, `--steps`, `--stages`, `--repeats` | Sampling budget and independent repetitions. Budget scaling instead uses `--chain-counts` and `--step-counts`. |
| `--reference-*` | Budget and tuning for the independent covariance reference. |
| `--trajectory-burnin`, `--trajectory-samples` | Post-learning trajectory lengths for diagnostics and correlators. |
| `--seed`, `--dtype`, `--output-prefix` | Reproducibility, precision, and output location. |

Explicit numerical options override preset values. For example:

```bash
python -m phi4.experiment_phi4_scaling --quick --sides 4,8 \
  --chains 128 --steps 64 --stages 6 \
  --output-prefix results/phi4_custom
```

## Run several experiments

The suite launcher runs **scaling, stages, diagnostics, correlators, and
parameter sweeps** by default. Hardness maps and budget scaling are opt-in.

```bash
python -m phi4 --quick --output-prefix results/phi4_suite
python -m phi4 --output-prefix figures/phi4_suite

# Select one or more workflows
python -m phi4 --only scaling --only stages --quick \
  --output-prefix results/phi4_selected
python -m phi4 --only budget --quick --output-prefix results/phi4_budget
```

The selection names are `scaling`, `stages`, `diagnostics`, `correlator`,
`sweeps`, `hardness`, and `budget`. The launcher accepts selection, presets,
and an output prefix; use the focused modules to set scientific parameters.

## GPU runs

GPU execution requires a GPU-enabled JAX installation and a visible GPU.
The scaling and stage presets select larger lattices, use float32, and retain
the scalable Fourier methods:

```bash
python -m phi4 --gpu --output-prefix figures/phi4_gpu
python -m phi4.experiment_phi4_budget_scaling --gpu \
  --output-prefix figures/phi4_budget
```

`python -m phi4 --gpu` runs scaling and stages only. Their preset sides are
64, 128, 256, and 512 for scaling, and 256 and 512 for stages.
The hardness-map and budget-scaling `--gpu` flags require a GPU without
changing scientific settings. Diagnostics and correlators have their own
GPU presets, available through their focused modules. Parameter sweeps have
no `--gpu` flag. `--quick` and `--gpu` are mutually exclusive.

The full budget study uses side 100, 12 stages, five repeats, and all 25 pairs
of $n\in\{64,128,256,512,1024\}$ chains and
$N\in\{32,64,128,256,512\}$ steps. Its default run is intended for a
large-memory accelerator; the reduced command above is the local smoke test.

## Model and comparisons

The target potential is

```math
U(\phi)=\frac12\phi^\mathsf T\Delta_\beta\phi+\sum_x u_R(\phi_x),
\qquad
u_R(z)=\lambda z^4/4+mz^2/2\quad (|z|\leq R).
```

Outside $[-R,R]$, the onsite potential continues quadratically. For finite
$R$, the curvature bounds are $\mu=m$ and
$L=\lambda_{\max}(\Delta_\beta)+m+3\lambda R^2$.
`--radius inf` selects the genuine quartic target. In that case,
`--cooling-design-radius` controls the operational curvature scale used by
the sampler; it neither truncates the target nor supplies a global bound.

The primary comparisons use translation-invariant Gaussian cooling, staged
empirical adaptation without cooling, and translation-averaged endpoints
from equal-budget unpreconditioned ULMC. A raw full empirical covariance is
included where the chain count exceeds $D$ and the dense size limit permits
it. Scaling also includes dense Gaussian cooling at feasible sizes. Budget
scaling evaluates only translation-invariant Gaussian cooling.

Covariance references come from independent ULMC endpoints preconditioned
by $(\Delta_\beta+mI)^{-1}$ and translation-averaged in Fourier space.
Hardness and budget NPZ files preserve repeat-level measurements and run
settings alongside the plots.

The diagnostic IAT maps use origin-site two-point products. Their mean maps
use the final half of each retained chain and an IAT-adjusted standard error;
gray sites lack a reliable standardized mean. The correlator experiment
averages connected center-site correlations over toroidal-distance shells
and shows IAT-adjusted 95% bands where reliable. Those bands condition on
the learned preconditioner and exclude learning variability. Longer chains
and smaller step sizes help assess sampling error and finite-step bias.

## Code organization

| File | Responsibility |
| --- | --- |
| [lattice_phi4.py](lattice_phi4.py) | Periodic target, analytic gradient and curvature, sampler wrappers, covariance references, and condition metrics. |
| [phi4_cli.py](phi4_cli.py) | Reusable argument groups, parsers, validation, and presets. Some experiments define their own options and defaults. |
| [phi4_plotting.py](phi4_plotting.py) | Shared figure style, labels, interval bands, and PDF saving. |
| `experiment_phi4_*.py` | Each study's results, run logic, figures, summary, and CLI. |
| [experiment_truncated_phi4.py](experiment_truncated_phi4.py), [__main__.py](__main__.py) | Suite selection and the `python -m phi4` entry point. |

Core ULMC and cooling implementations live in
[../gaussian_cooling_algs.py](../gaussian_cooling_algs.py).
