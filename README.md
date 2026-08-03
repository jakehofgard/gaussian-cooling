# Gaussian cooling

JAX implementations of ULMC, Gaussian cooling, and reproducible Gaussian and
lattice $\phi^4$ experiments. Preconditioner quality is measured by

```math
\kappa_{\mathrm{rel}}(\widehat\Sigma,\Sigma)
=
\kappa\!\left(
\widehat\Sigma^{-1/2}\Sigma\widehat\Sigma^{-1/2}
\right).
```

Here $\Sigma$ is an exact or independently estimated target covariance and
$\widehat\Sigma$ is the learned preconditioner. Smaller is better; the ideal
value is one. Every figure is saved as a separate vector PDF.

## Files

| File | Purpose |
| --- | --- |
| `gaussian_cooling_algs.py` | Reusable ULMC, dense Gaussian cooling, and translation-invariant Gaussian cooling. |
| `experiment_transformed_gaussian.py` | Exact transformed-Gaussian benchmark. |
| `phi4/lattice_phi4.py` | Shared lattice model, samplers, reference estimator, and condition metrics. |
| `phi4/phi4_cli.py`, `phi4/phi4_plotting.py` | Shared command-line, validation, and publication-plot helpers. |
| `phi4/experiment_phi4_scaling.py` | Preconditioner quality and runtime versus lattice size. |
| `phi4/experiment_phi4_stage_convergence.py` | Stagewise convergence at selected lattice sizes. |
| `phi4/experiment_phi4_diagnostics.py` | Post-cooling correlator-IAT and ergodicity diagnostics. |
| `phi4/experiment_phi4_correlator.py` | Center-site two-point correlators under each preconditioner. |
| `phi4/experiment_phi4_parameter_sweeps.py` | Controlled $\lambda$, $m$, and $R$ sweeps. |
| `phi4/experiment_phi4_hardness_map.py` | Fixed-budget $(m,\lambda)$ hardness maps. |
| `phi4/__main__.py` | Convenience launcher for the complete lattice $\phi^4$ suite. |
| `RUNPOD.md` | Cost-conscious H100 setup and production commands. |
| `Dockerfile.runpod` | Optional reproducible NVIDIA container. |

No datasets are required.

## Install

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA GPUs, install CUDA-enabled JAX using the
[official JAX instructions](https://docs.jax.dev/en/latest/installation.html)
before installing the remaining requirements. JAX automatically uses an
available GPU. See [RUNPOD.md](RUNPOD.md) for the H100 workflow.

## Quick checks

Run the module commands below from the repository root.

```bash
python experiment_transformed_gaussian.py --quick \
  --output-prefix results/gaussian_quick

python -m phi4 --quick \
  --output-prefix results/phi4_suite_quick

python -m phi4.experiment_phi4_scaling --quick \
  --output-prefix results/phi4_scaling_quick

python -m phi4.experiment_phi4_hardness_map --quick \
  --output-prefix results/phi4_hardness_quick
```

## Transformed Gaussian

The well-conditioned base target is
$z\sim\mathcal N(0,\Sigma_0)$, where $\kappa(\Sigma_0)=10$ by default.
Its eigenvectors are Haar-random and its eigenvalues are geometrically spaced
in $[1,10]$. With $z=B^{1/2}x$, the transformed target is

```math
x\sim\mathcal N\!\left(0,B^{-1/2}\Sigma_0B^{-1/2}\right),
```

where $B$ has independent Haar eigenvectors and eigenvalues in
$[1/\kappa(B),1]$. Gaussian cooling, empirical adaptation without cooling,
and equal-budget unpreconditioned ULMC use the same $nNK$ gradient budget.

```bash
python experiment_transformed_gaussian.py \
  --output-prefix figures/transformed_gaussian
```

| Output suffix | Horizontal axis | Vertical axis |
| --- | --- | --- |
| `_preconditioner_quality.pdf` | Transformation condition $\kappa(B)$ | $\kappa_{\mathrm{rel}}$ for all methods |
| `_stage_convergence.pdf` | Cumulative stage $k$ at the hardest $\kappa(B)$ | $\kappa_{\mathrm{rel}}$, including the exact cooling oracle |

Defaults are dimension 20, $\kappa(B)\in\{1,10,100,10^3,10^4\}$,
$n=256$, $N=96$, and $K=12$.

## Lattice phi4 experiments

The target is defined on a periodic square lattice of side length $d$ and
ambient dimension $D=d^2$:

```math
U(\phi)
=
\frac12\phi^\mathsf T\Delta_\beta\phi
+\sum_x u_R(\phi_x).
```

Inside $[-R,R]$, $u_R(z)=\lambda z^4/4+mz^2/2$; outside, the truncated
target uses its quadratic continuation. For finite $R$,

```math
\mu=m,
\qquad
L=\lambda_{\max}(\Delta_\beta)+m+3\lambda R^2,
\qquad
\kappa_H=L/\mu.
```

The non-Gaussian covariance is not analytic. Each relative-condition estimate
therefore uses independent reference endpoints sampled after preconditioning
by $(\Delta_\beta+mI)^{-1}$, followed by translation averaging in Fourier
space.

The four primary methods are translation-invariant Gaussian cooling,
translation-invariant empirical preconditioning without cooling, translation
averaging of equal-budget unpreconditioned ULMC endpoints, and the raw full
empirical covariance of those endpoints where $n>D$ and the matrix is small
enough to form. Dense Gaussian cooling is an ancillary small-$D$ comparison.

### Focused entry points

Each experiment can be run independently with only its relevant options.

| Module file | Experiment | Main outputs |
| --- | --- | --- |
| `phi4/experiment_phi4_scaling.py` | Quality and post-JIT runtime versus $d$ | `_preconditioner_quality.pdf`, `_covariance_estimation_cost.pdf`, and dense ancillary PDFs |
| `phi4/experiment_phi4_stage_convergence.py` | Equal-budget convergence versus cumulative stage $k$ | `_stage_comparison_d{d}.pdf` |
| `phi4/experiment_phi4_diagnostics.py` | Correlator IATs and final-half mean diagnostics after cooling | `_two_point_iat_d{d}.pdf`, `_ergodicity_site_mean_z_scores_d{d}.pdf` |
| `phi4/experiment_phi4_correlator.py` | Center-site correlator decay under each feasible preconditioner | `_two_point_correlator_decay_d{d}.pdf` |
| `phi4/experiment_phi4_parameter_sweeps.py` | One-factor $\lambda$, $m$, and $R$ comparisons | `_parameter_sweep_lambda.pdf`, `_parameter_sweep_mass.pdf`, `_parameter_sweep_radius.pdf`, `_truncation_to_quartic_covariance.pdf` |
| `phi4/experiment_phi4_hardness_map.py` | Fixed-budget map over $(m,\lambda)$ at $R=4$, $\beta=2$ | `_hardness_absolute_d{d}.pdf`, `_hardness_adaptive_gain_d{d}.pdf`, `_hardness_map_data.npz` |

Default runs:

```bash
python -m phi4.experiment_phi4_scaling \
  --output-prefix figures/phi4_scaling

python -m phi4.experiment_phi4_stage_convergence \
  --output-prefix figures/phi4_stages

python -m phi4.experiment_phi4_diagnostics \
  --output-prefix figures/phi4_diagnostics

python -m phi4.experiment_phi4_correlator \
  --output-prefix figures/phi4_correlator

python -m phi4.experiment_phi4_parameter_sweeps \
  --output-prefix figures/phi4_parameter_sweeps

python -m phi4.experiment_phi4_hardness_map \
  --output-prefix figures/phi4_hardness
```

Use `python -m phi4.MODULE --help` for focused scientific controls. The suite
launcher intentionally accepts only suite selection, quick/GPU presets, and
an output prefix; it runs the complete ordinary suite with:

```bash
python -m phi4 \
  --output-prefix figures/truncated_phi4
```

The ordinary target defaults are $\beta=2$, $\lambda=0.5$, $m=0.25$, and
$R=2$. Method defaults are $n=512$, $N=64$, and three repeats. Scaling and
stage convergence use $K=8$; parameter sweeps use $K=12$.

### Sampling diagnostics and correlators

The diagnostic experiment learns a translation-invariant Gaussian-cooling
preconditioner and estimates the IAT of

```math
g_x(t)=\delta\phi_t(0)\,\delta\phi_t(x)
```

for every lattice site $x$. Its ergodicity map uses the mean of the final half
of the retained chain and an IAT-adjusted Monte Carlo standard error. Gray
sites do not have a sufficiently long chain for a reliable standardized mean.

The separate correlator experiment uses
$c=(\lfloor d/2\rfloor,\lfloor d/2\rfloor)$ and estimates the connected
correlator

```math
C(c,x)
=
\mathbb E[(\phi(c)-\mathbb E\phi(c))(\phi(x)-\mathbb E\phi(x))].
```

It averages over exact toroidal-distance shells and shows IAT-adjusted 95%
uncertainty bands where reliable. At equilibrium, $\phi\mapsto-\phi$ symmetry
makes the connected and ordinary correlators equal. All method curves target
the same observable; disagreement can indicate burn-in, Monte Carlo error, or
finite-step ULMC bias. Repeat publication runs with longer trajectories and a
smaller step size. The uncertainty bands condition on the selected learned
preconditioner and do not include learning variability.

### Genuine quartic target

`--radius inf` removes the truncation. The quartic Hessian is then unbounded,
so `--cooling-design-radius` is only an operational stage-zero and step-size
scale; it does not change the target or provide a global smoothness bound.

```bash
python -m phi4.experiment_phi4_scaling --quick --radius inf \
  --cooling-design-radius 2 \
  --output-prefix figures/phi4_genuine_quartic
```

Check important runs with a larger design radius and smaller step size.

### Parameter sweeps and hardness map

The parameter-sweep defaults use side 10. They vary
$\lambda\in\{0.1,0.5,2\}$ at $m=0.05,R=2$,
$m\in\{0.01,0.05,0.25\}$ at $\lambda=0.5,R=2$, and
$R\in\{0.5,2,4,\infty\}$ at $\lambda=0.5,m=0.05$, with common
$R_{\mathrm{design}}=4$.

The hardness map fixes $R=4$ and $\beta=2$. Its horizontal axis is mass $m$,
its vertical axis is quartic coupling $\lambda$, and both are logarithmic. For
the default even sides,

```math
\kappa_H=1+\frac{16+48\lambda}{m}.
```

Every cell uses the same practitioner budget, by default $n=512$, $N=128$,
and $K=12$. The three scalable methods receive equal $nNK$ gradient budgets;
the raw full covariance is omitted. Absolute maps show median
$\log_{10}\kappa_{\mathrm{rel}}$. Gain maps show
$\log_{10}(\kappa_{\mathrm{baseline}}/\kappa_{\mathrm{method}})$, using
translation-averaged ULMC as the baseline. The NPZ file preserves all repeats,
axes, budgets, and reference settings. Use at least three repeats and increase
the reference budget for publication results.

### GPU scaling

Run the two large-lattice experiments independently:

```bash
python -m phi4.experiment_phi4_scaling --gpu \
  --output-prefix figures/phi4_gpu_scaling

python -m phi4.experiment_phi4_stage_convergence --gpu \
  --output-prefix figures/phi4_gpu_stages
```

The scaling preset uses sides 64, 128, 256, 512, and 1024 with float32,
$n=64$, $N=32$, $K=8$, and one repeat. The stage preset uses sides 512 and
1024. Both require a visible JAX GPU and retain only scalable Fourier methods
at these ranks. The suite-launcher alternative runs both presets in one
command:

```bash
python -m phi4 --gpu \
  --output-prefix figures/phi4_gpu_suite
```

The hardness map uses an available GPU automatically; its optional `--gpu`
flag only requires that a GPU be visible and does not change its grid.

## Library use and reproducibility

`gaussian_cooling_algs.py` exports `ulmc`, `transformed_ulmc`,
`gaussian_cooling`, `translation_invariant_gaussian_cooling`, covariance
estimators, Fourier operators, and symmetric matrix-square-root utilities.
`phi4/lattice_phi4.py` supplies the shared periodic target, reference sampler,
method wrappers, and condition metrics used by every focused lattice module.
The experiment modules contain only their own run, result, plot, summary, and
CLI logic; `phi4/phi4_cli.py` and `phi4/phi4_plotting.py` centralize reusable
interface and presentation helpers.

Samplers return independent-chain endpoints rather than trajectories.
`sample_covariance` uses the biased $1/n$ normalization. At large lattice
sizes, retain Fourier spectra rather than materializing dense covariances.
All experiment modules use explicit seeds. Small numerical differences can
occur across JAX versions and hardware.

Add the copyright holder's chosen license and paper citation before release.
