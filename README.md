# Gaussian cooling

JAX implementations of ULMC, Gaussian cooling, and two reproducible
preconditioning experiments. Preconditioner quality is measured by

```math
\kappa_{\mathrm{rel}}
=
\kappa\!\left(
\widehat{\Sigma}^{-1/2}\Sigma\widehat{\Sigma}^{-1/2}
\right).
```

## Contents

| File | Purpose |
| --- | --- |
| `gaussian_cooling_algs.py` | Reusable ULMC and Gaussian-cooling algorithms. |
| `experiment_transformed_gaussian.py` | Transformed-Gaussian benchmark with an exact reference. |
| `experiment_truncated_phi4.py` | Translation-invariant lattice $\phi^4$ benchmark and diagnostics. |
| `Dockerfile.runpod` | Reproducible NVIDIA container for RunPod deployment. |
| `RUNPOD.md` | Cost-conscious setup and scaling guide for a RunPod H100. |

No datasets are required. Every plot is saved as a separate vector PDF.

## Install

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install CUDA-enabled JAX using the
[official JAX instructions](https://docs.jax.dev/en/latest/installation.html)
before installing the requirements. The algorithms use one GPU; plotting,
dense metrics, and autocorrelation analysis remain CPU-side.

For an H100, follow the [RunPod setup and scaling guide](RUNPOD.md). It covers
the network-volume layout, persistent JAX cache, GPU verification, production
commands, and shutdown safeguards.

## Run

Smoke tests:

```bash
python experiment_transformed_gaussian.py \
  --quick --output-prefix results/transformed_gaussian

python experiment_truncated_phi4.py \
  --quick --output-prefix results/truncated_phi4
```

Production defaults:

```bash
python experiment_transformed_gaussian.py \
  --output-prefix figures/transformed_gaussian

python experiment_truncated_phi4.py \
  --output-prefix figures/truncated_phi4
```

H100-scale lattice comparison through side 1024:

```bash
python experiment_truncated_phi4.py --gpu \
  --output-prefix figures/phi4_gpu_scale
```

`--gpu` refuses to run without a JAX GPU backend. Its conservative preset uses
sides 64, 128, 256, 512, and 1024; 64 chains; 128 reference chains; one repeat;
float32; and no dense, diagnostic, stage-comparison, or parameter-sweep paths.
This is a feasibility and timing run, not a statistically final configuration.
Explicit options override these values. Use `--run-diagnostics`,
`--run-comparisons`, or `--run-parameter-sweeps` to opt those components back
in; diagnostics then use side 64 unless `--diagnostic-side` says otherwise.
These components are intentionally excluded from the million-site scaling
run. Add `--radius inf --cooling-design-radius 4` to use the genuine quartic
target at the same GPU sizes.

The first run includes JAX compilation. Explicit options override the
`--quick` preset. Use `--help` for all controls and defaults.

`--output-prefix results/run` creates files such as
`results/run_preconditioner_quality.pdf`; parent directories are created
automatically.

## Experiments

### Transformed Gaussian

The base target is $z\sim\mathcal{N}(0,\Sigma_0)$, where $\Sigma_0$ has
Haar-random eigenvectors and eigenvalues in $[1,10]$. The coordinate change
$z=B^{1/2}x$ produces

```math
x\sim\mathcal{N}\!\left(
0,\,B^{-1/2}\Sigma_0B^{-1/2}
\right).
```

The matrix $B$ has independent Haar-random eigenvectors and geometrically
spaced eigenvalues in $[1/\kappa(B),1]$. The experiment compares:

- Gaussian cooling;
- $K$-stage empirical preconditioning without cooling;
- equal-budget unpreconditioned ULMC.

All methods use the same $nNK$ gradient-evaluation budget. The default is
$K=12$.

### Lattice $\phi^4$

On a periodic $d\times d$ lattice,

```math
U(\phi)
=
\frac{1}{2}\phi^\mathsf{T}\Delta_\beta\phi
+ \sum_x u_R(\phi_x),
```

where $u_R(z)=\lambda z^4/4+mz^2/2$ for $|z|\leq R$, with a quadratic
continuation outside that interval. For finite $R$, the curvature parameters
are

```math
\mu=m,\qquad
L=\lambda_{\max}(\Delta_\beta)+m+3\lambda R^2,\qquad
\alpha=\left(1+\frac{3\lambda R^2}{m}\right)^{-1}.
```

Setting `--radius inf` selects the genuine quartic potential everywhere. Its
Hessian is unbounded, so no finite global $L$ exists. In this case,
`--cooling-design-radius` supplies a finite operational scale

```math
L_{\mathrm{design}}
=\lambda_{\max}(\Delta_\beta)+m+3\lambda R_{\mathrm{design}}^2
```

for stage zero and fixed-step tuning only. It does not truncate the target,
and the finite-global-$L$ guarantee does not apply. The console reports the
reference fraction beyond $R_{\mathrm{design}}$; repeat important runs with a
larger design radius and a smaller step size as a sensitivity check.

The four primary methods are translation-invariant Gaussian cooling,
translation-invariant empirical preconditioning, translation averaging of
unpreconditioned ULMC endpoints, and the full covariance from those same
endpoints.

The production run also compares all four methods across
$\lambda\in\{0.1,0.5,2\}$, $m\in\{0.01,0.05,0.25\}$, and
$R\in\{0.5,2,4,\infty\}$. The radius comparison uses one common
$R_{\mathrm{design}}=4$ and creates a second PDF showing
$\kappa_{\mathrm{rel}}(\Sigma_R,\Sigma_\infty)$ for the reference spectra.

To run only a small genuine-quartic main experiment:

```bash
python experiment_truncated_phi4.py --quick --radius inf \
  --cooling-design-radius 2 --skip-diagnostics --skip-comparisons \
  --skip-parameter-sweeps
```

The full defaults include lattices through $d=100$, parameter sweeps,
two-point-correlator IATs, and symmetry diagnostics. Optional components can
be disabled with:

```text
--skip-diagnostics
--skip-comparisons
--skip-parameter-sweeps
```

Dense methods are skipped automatically when infeasible. For large GPU runs,
use the `--gpu` preset and increase chains or repeats only after the
one-repeat run fits in device memory.

## Library use

`gaussian_cooling_algs.py` exports `ulmc`, `transformed_ulmc`,
`gaussian_cooling`, `translation_invariant_gaussian_cooling`, covariance
estimators, Fourier operators, and matrix-square-root utilities.

Samplers return independent-chain endpoints, not trajectories.
`sample_covariance` uses the biased $1/n$ normalization. At large lattice
sizes, retain the Fourier spectrum instead of materializing a dense
covariance.

## Reproducibility and license

Both experiments use explicit seeds. Curves show medians and interquartile
ranges across repeats; small numerical differences can occur across JAX
versions and hardware.

Add the copyright holder's chosen license and the appropriate paper citation
before publishing the repository.
