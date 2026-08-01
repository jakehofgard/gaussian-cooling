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
| `experiment_truncated_phi4.py` | Truncated/genuine lattice $\phi^4$ benchmarks, diagnostics, sweeps, hardness maps, and GPU scaling. |
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

```bash
# Transformed Gaussian
python experiment_transformed_gaussian.py --quick \
  --output-prefix results/gaussian_quick

# Full phi4 code path at reduced sizes
python experiment_truncated_phi4.py --quick \
  --output-prefix results/phi4_quick

# Hardness-map code path only
python experiment_truncated_phi4.py --hardness-map --quick \
  --output-prefix results/hardness_quick
```

The ordinary phi4 `--quick` preset still includes its diagnostics, stage
comparisons, and reduced parameter sweeps. Add the relevant `--skip-*` flags
if only a faster main-size smoke test is wanted.

## Experiment 1: transformed Gaussian

The well-conditioned base target is
$z\sim\mathcal N(0,\Sigma_0)$, where $\kappa(\Sigma_0)=10$ by default.
Its eigenvectors are Haar-random and its eigenvalues are geometrically spaced
in $[1,10]$. With $z=B^{1/2}x$, the transformed target is

```math
x\sim\mathcal N\!\left(0,B^{-1/2}\Sigma_0B^{-1/2}\right),
```

where $B$ has independent Haar eigenvectors and eigenvalues in
$[1/\kappa(B),1]$. Gaussian cooling, empirical adaptation without cooling,
and equal-budget unpreconditioned ULMC all use the same $nNK$ gradient budget.

```bash
python experiment_transformed_gaussian.py \
  --output-prefix figures/transformed_gaussian
```

| Output suffix | Horizontal axis | Vertical axis |
| --- | --- | --- |
| `_preconditioner_quality.pdf` | Transformation condition $\kappa(B)$ | $\kappa_{\mathrm{rel}}$ for all three methods |
| `_stage_convergence.pdf` | Cumulative stage $k$ for the hardest $\kappa(B)$ | $\kappa_{\mathrm{rel}}$, including the exact cooling oracle |

Defaults are vector dimension 20,
$\kappa(B)\in\{1,10,100,10^3,10^4\}$, $n=256$, $N=96$, $K=12$, eight quality
repeats, and three convergence repeats. Customize with `--dimension`,
`--kappas`, `--chains`, `--steps`, `--stages`, `--repeats`, and
`--convergence-repeats`.

## Experiment 2: lattice phi4 suite

The target is defined on a periodic square lattice of side length $d$ and
ambient dimension $D=d^2$:

```math
U(\phi)
=
\frac12\phi^\mathsf T\Delta_\beta\phi
+\sum_x u_R(\phi_x).
```

Inside $[-R,R]$,
$u_R(z)=\lambda z^4/4+mz^2/2$; outside, the truncated target uses its
quadratic continuation. For finite $R$,

```math
\mu=m,
\qquad
L=\lambda_{\max}(\Delta_\beta)+m+3\lambda R^2,
\qquad
\kappa_H=L/\mu.
```

Because the non-Gaussian covariance is not analytic, every
$\kappa_{\mathrm{rel}}$ uses independent reference endpoints sampled after
preconditioning by the known quadratic covariance
$(\Delta_\beta+mI)^{-1}$, followed by translation averaging in Fourier space.

The primary methods are:

1. translation-invariant Gaussian cooling;
2. translation-invariant empirical preconditioning, which feeds each learned
   Fourier spectrum into the next stage but applies no cooling;
3. translation averaging applied only after an equal-budget unpreconditioned
   ULMC run; and
4. the raw full covariance from those same ULMC endpoints, when it is
   full-rank and small enough to materialize.

Dense Gaussian cooling is an ancillary small-$D$ comparison.

Run the complete ordinary suite with:

```bash
python experiment_truncated_phi4.py \
  --output-prefix figures/truncated_phi4
```

Ordinary defaults are sides $5,10,20,50,100$; $\beta=2$, $\lambda=0.5$,
$m=0.25$, $R=2$; $n=512$, $N=64$, $K=8$; and three repeats. Dense work is
capped at side 20, stage plots use sides 10 and 100, and diagnostics use side
100 with 1,024 burn-in and 8,192 retained transitions.

It produces the following experiments.

| Output suffix | What is plotted |
| --- | --- |
| `_preconditioner_quality.pdf` | Horizontal: side length $d$. Vertical: $\kappa_{\mathrm{rel}}$ for the scalable primary methods and raw covariance where feasible. |
| `_covariance_estimation_cost.pdf` | Horizontal: number of sites $D=d^2$. Vertical: wall time after JIT compilation. |
| `_dense_cooling_quality.pdf`, `_dense_cooling_cost.pdf` | Ancillary dense-cooling quality and time where side is at or below `--dense-max-side` and $n>D$. |
| `_stage_comparison_d{d}.pdf` | Horizontal: cumulative stage $k$; plain ULMC has used $kN$ transitions. Vertical: $\kappa_{\mathrm{rel}}$. One PDF per `--comparison-sides`. |
| `_two_point_iat_d{d}.pdf` | Horizontal: periodic distance from the origin. Vertical: IAT of each first covariance-row/two-point-correlator observable. |
| `_ergodicity_site_mean_z_scores_d{d}.pdf` | Lattice coordinates on the two axes; color is the final-half site-mean z score using IAT-adjusted MCSE. |
| `_parameter_sweep_lambda.pdf` | Horizontal: $\lambda$; vertical: $\kappa_{\mathrm{rel}}$, with $m,R$ fixed. |
| `_parameter_sweep_mass.pdf` | Horizontal: $m$; vertical: $\kappa_{\mathrm{rel}}$, with $\lambda,R$ fixed. |
| `_parameter_sweep_radius.pdf` | Horizontal: $R$, including $R=\infty$; vertical: $\kappa_{\mathrm{rel}}$. |
| `_truncation_to_quartic_covariance.pdf` | If the radius grid contains $\infty$: horizontal $R$; vertical $\kappa_{\mathrm{rel}}(\Sigma_R,\Sigma_\infty)$. |

The two-point panel estimates the IAT of
$g_x(t)=\delta\phi_t(0)\delta\phi_t(x)$ for every site $x$. In the ergodicity
map, gray sites are masked because the final half-chain is too short for a
reliable IAT-adjusted z score; counts and summary statistics remain in the
console output rather than covering the figure.

The default one-factor sweeps use side 10 and $K=12$: $\lambda$ runs over
$\{0.1,0.5,2\}$ at $m=0.05,R=2$; $m$ runs over
$\{0.01,0.05,0.25\}$ at $\lambda=0.5,R=2$; and $R$ runs over
$\{0.5,2,4,\infty\}$ at $\lambda=0.5,m=0.05$, with common
$R_{\mathrm{design}}=4$.

Useful focused invocations are:

```bash
# Stage convergence at selected sides; the main size plots are also produced.
python experiment_truncated_phi4.py \
  --sides 10,100 --comparison-sides 10,100 \
  --skip-diagnostics --skip-parameter-sweeps \
  --output-prefix figures/phi4_stages

# Diagnostics at side 100; the main benchmark is run first.
python experiment_truncated_phi4.py \
  --sides 100 --repeats 1 --dense-max-side 0 \
  --skip-comparisons --skip-parameter-sweeps \
  --output-prefix figures/phi4_diagnostics

# One-factor lambda, mass, and radius sweeps at a dense-feasible side.
python experiment_truncated_phi4.py \
  --sides 2 --skip-diagnostics --skip-comparisons \
  --output-prefix figures/phi4_parameter_sweeps
```

The phi4 script always runs its main lattice-size benchmark unless the
separate `--hardness-map` mode is selected. Optional ordinary components are
controlled by `--skip-diagnostics`, `--skip-comparisons`, and
`--skip-parameter-sweeps` (with corresponding `--run-*` overrides).

### Genuine quartic target

`--radius inf` removes the truncation. Because the true quartic Hessian is
unbounded, `--cooling-design-radius` supplies only a finite operational scale
for stage zero and step-size tuning; it does not alter the target.

```bash
python experiment_truncated_phi4.py --quick --radius inf \
  --cooling-design-radius 2 --skip-diagnostics \
  --skip-comparisons --skip-parameter-sweeps \
  --output-prefix figures/phi4_genuine_quartic
```

Repeat important genuine-quartic runs with a larger design radius and smaller
step size to check tuning-scale sensitivity.

## Experiment 3: fixed-R, fixed-beta hardness map

This is a separate, scalable test:

```bash
python experiment_truncated_phi4.py --hardness-map --gpu \
  --output-prefix figures/phi4_hardness
```

The axes and fixed quantities are:

- horizontal: mass $m$;
- vertical: quartic coupling $\lambda$;
- both axes are logarithmic, with defaults
  $m\in\{0.01,0.05,0.25\}$ and $\lambda\in\{0.1,0.5,2\}$;
- one pair of PDFs for each lattice side $d\in\{8,16,32,64,128\}$;
- fixed $R=4$ and $\beta=2$; and
- ambient sampling dimension $D=d^2$.

For the default even sides,

```math
\kappa_H
=\frac{L}{m}
=1+\frac{16+48\lambda}{m}.
```

Every cell uses the fixed practitioner budget $n=512$, $N=128$, and $K=12$
by default. Holding $n,N,K$ constant makes the map a fixed-compute regime
comparison: differences across $(d,m,\lambda)$ are not confounded by assigning
more work to harder targets. The dimension-dependent prescriptions motivated
by theoretical accuracy bounds are therefore not used here.

All three scalable methods receive the same $nNK$ gradient budget. Adaptive
methods initialize $n$ chains at each stage; plain ULMC evolves $n$ continuous
chains for $KN$ transitions. The independent reference run has its own
reported budget and is not counted as method work. Run `--quick` first: the
full 45-cell map is intended for a GPU.

For custom harder grids, the CLI warns when
$\gamma_{\mathrm{cool}}^K\max\kappa_H>\delta$ and recommends increasing
`--hardness-stages`.

For each side, the mode writes:

- `_hardness_absolute_d{side}.pdf`: three panels of median
  $\log_{10}\kappa_{\mathrm{rel}}$ on one shared scale;
- `_hardness_adaptive_gain_d{side}.pdf`: cooling and empirical-adaptation gains
  over translation-averaged plain ULMC, computed as the log ratio of the two
  median $\kappa_{\mathrm{rel}}$ values. The plotted ratio is baseline over
  method, so values above zero correspond to a smaller condition number for
  the adaptive method; and
- `_hardness_map_data.npz`: raw repeats, fixed budgets, reference settings,
  axes, and fixed parameters.

The raw full covariance is intentionally absent: it is rank-deficient whenever
$n\leq D$, and materializing a $D\times D$ matrix at side 128 would defeat the
scalable test. Change the grid or fixed budget with `--hardness-sides`,
`--hardness-masses`, `--hardness-lambdas`, `--hardness-chains`,
`--hardness-steps`, `--hardness-stages`, and `--hardness-repeats`. Shared
sampler controls include `--step-size`, `--friction`, `--cooling-gamma`,
`--dtype`, and `--seed`; reference controls start with
`--hardness-reference-`.

The one-repeat default is exploratory; use `--hardness-repeats 3` or more for
publication runs. Check reference sensitivity by increasing
`--hardness-reference-time`, `--hardness-reference-min-chains`, and
`--hardness-reference-chain-factor`. Each cell also executes one unmeasured
JAX warm-up per compared method, so compilation wall time exceeds the nominal
comparison budget.

The default reference uses at least 512 chains, otherwise twice the method
count, and physical time 2. The console reports
$\kappa_{\mathrm{rel}}$ between the two reference half-samples as a Monte
Carlo consistency check (nearer one is better). This check does not prove that
the reference has mixed, so the sensitivity run remains important.

## Experiment 4: H100 scaling

```bash
python experiment_truncated_phi4.py --gpu \
  --output-prefix figures/phi4_gpu_scale
```

`--gpu` requires a visible JAX GPU and runs sides 64, 128, 256, 512, and 1024
with float32, $n=64$, $N=32$, $K=8$, and one repeat. It writes the scalable
quality/time plots plus stagewise convergence at sides 512 and 1024. Dense
methods, diagnostics, and parameter sweeps are disabled. The full empirical
covariance is rank-deficient and omitted. Use `--skip-comparisons` for a
timing-only run or `--comparison-sides` to change the history sizes.
The reference uses 128 chains and 128 transitions. Stage histories rerun the
three scalable methods at sides 512 and 1024, so skipping them materially
reduces work. The target retains the ordinary $\beta,\lambda,m,R$ defaults.

The hardness map uses a GPU automatically when available. Adding `--gpu` to
`--hardness-map` only enforces that a GPU is visible; it does not apply the
large-lattice scaling preset. `--quick` and `--gpu` are mutually exclusive.

## Library use and reproducibility

`gaussian_cooling_algs.py` exports `ulmc`, `transformed_ulmc`,
`gaussian_cooling`, `translation_invariant_gaussian_cooling`, covariance
estimators, Fourier operators, and symmetric matrix-square-root utilities.
Samplers return independent-chain endpoints rather than trajectories.
`sample_covariance` uses the biased $1/n$ normalization. At large lattice
sizes, retain Fourier spectra rather than materializing dense covariances.

Both scripts use explicit seeds. Repeated method curves report medians and
interquartile ranges; heatmaps report medians and preserve every repeat in the
NPZ output. Small numerical differences can occur across JAX versions and
hardware.

Add the copyright holder's chosen license and paper citation before release.
