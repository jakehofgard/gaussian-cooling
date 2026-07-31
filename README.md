# Gaussian cooling experiments

This repository contains JAX implementations of Gaussian cooling and two
reproducible numerical experiments:

1. an affinely transformed Gaussian target with known covariance; and
2. a truncated, translation-invariant lattice \(\phi^4\) target.

The experiments compare covariance preconditioners using the scale-invariant
metric

\[
\kappa_{\mathrm{rel}}
=
\kappa\!\left(
\widehat{\Sigma}^{-1/2}
\Sigma
\widehat{\Sigma}^{-1/2}
\right).
\]

All plots use recognizable method names, include their simulation parameters,
and are written as separate vector PDF files. No datasets or network access
are required.

## Repository contents

| File | Purpose |
| --- | --- |
| `gaussian_cooling_algs.py` | Reusable ULMC, transformed ULMC, dense Gaussian-cooling, and translation-invariant Gaussian-cooling routines. |
| `experiment_transformed_gaussian.py` | Controlled transformed-Gaussian comparison with an analytic covariance reference. |
| `experiment_truncated_phi4.py` | Lattice \(\phi^4\) comparisons, parameter sweeps, two-point-correlator IATs, and ergodicity diagnostics. |
| `requirements.txt` | Runtime dependencies. |

The two experiment files are executable scripts, while
`gaussian_cooling_algs.py` can also be imported as a small library.

## Installation

Python 3.10 or newer is required. Python 3.11 is the tested version.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The tested environment used JAX/JAXlib 0.4.23, NumPy 1.26.4, Matplotlib
3.7.1, and emcee 3.1.6. The bounds in `requirements.txt` allow compatible
newer releases.

The requirements install the standard JAX build. Researchers using NVIDIA,
AMD, or TPU accelerators should first follow the
[platform-specific JAX installation instructions](https://docs.jax.dev/en/latest/installation.html),
then install the remaining requirements. The experiments do not require an
accelerator.

## Quick verification

Run these commands from the repository root:

```bash
python experiment_transformed_gaussian.py \
  --quick \
  --output-prefix results/transformed_gaussian

python experiment_truncated_phi4.py \
  --quick \
  --output-prefix results/truncated_phi4
```

`--quick` selects a small deterministic smoke-test preset. Explicit numerical
options still take precedence, so, for example,

```bash
python experiment_transformed_gaussian.py \
  --quick \
  --chains 256 \
  --output-prefix results/transformed_gaussian
```

keeps the quick preset but uses 256 chains.

The first run can pause while JAX compiles array programs. Later calls with
the same shapes are typically faster.

## Production experiments

The default transformed-Gaussian experiment is:

```bash
python experiment_transformed_gaussian.py \
  --output-prefix figures/transformed_gaussian
```

It uses 12 Gaussian-cooling stages and compares:

- Gaussian cooling;
- \(K\)-stage empirical preconditioning without cooling; and
- an equal-budget unpreconditioned ULMC chain.

The target precision has Haar-random eigenvectors and geometrically spaced
eigenvalues. The same \(nNK\) gradient-evaluation budget is used for all
three methods. Curves are medians across repeats and shaded bands are
interquartile ranges.

Run the default lattice experiment with:

```bash
python experiment_truncated_phi4.py \
  --output-prefix figures/truncated_phi4
```

The lattice experiment compares four primary methods:

- translation-invariant Gaussian cooling;
- translation-invariant empirical preconditioning without cooling;
- translation averaging of covariance endpoints from unpreconditioned ULMC;
- the full empirical covariance from those same unpreconditioned endpoints.

The two unpreconditioned estimates share exactly the same chain endpoints.
Dense Gaussian cooling is also reported where the covariance is small enough
and the empirical estimate is full rank.

### Lattice target and conditioning parameters

For a periodic \(d\times d\) lattice, the potential is

\[
U(\phi)
=
\frac12\phi^\mathsf{T}\Delta_\beta\phi
+\sum_x u_R(\phi_x),
\]

where \(u_R(z)=\lambda z^4/4+mz^2/2\) for \(|z|\le R\), with a \(C^2\)
quadratic continuation outside that interval. The uniform curvature
parameters used by the scripts are

\[
\mu=m,\qquad
L=\lambda_{\max}(\Delta_\beta)+m+3\lambda R^2,\qquad
\alpha=\left(1+\frac{3\lambda R^2}{m}\right)^{-1}.
\]

The default production run includes controlled one-factor sweeps around
\((\lambda,m,R)=(0.5,0.05,2)\):

- \(\lambda\in\{0.1,0.5,2\}\);
- \(m\in\{0.01,0.05,0.25\}\);
- \(R\in\{0.5,2,4\}\).

Each sweep uses \(d=10\), \(K=12\), common random numbers, the same four
methods, and a cached shared anchor. It creates three additional standalone
PDFs. The radius plot also reports the reference fraction
\(\Pr(|\phi|>R)\), because increasing \(R\) can worsen the uniform Hessian
bound even when the truncation boundary lies outside the typical set.

Custom grids are comma-separated:

```bash
python experiment_truncated_phi4.py \
  --lambda-sweep-values 0.05,0.1,0.5,1,2 \
  --mass-sweep-values 0.01,0.02,0.05,0.1,0.25 \
  --radius-sweep-values 0.5,1,2,3,4 \
  --output-prefix figures/truncated_phi4_extended
```

## Output naming

`--output-prefix` is a filename prefix, not a directory. The scripts append a
descriptive suffix and `.pdf`. For example,

```text
--output-prefix results/transformed_gaussian
```

produces:

```text
results/transformed_gaussian_preconditioner_quality.pdf
results/transformed_gaussian_stage_convergence.pdf
```

The lattice script similarly produces separate quality, cost, stage,
parameter-sweep, autocorrelation, and ergodicity PDFs. Parent directories are
created automatically. The legacy spelling `--output` remains accepted as an
alias.

## Important command-line controls

Use each script's full help for defaults and descriptions:

```bash
python experiment_transformed_gaussian.py --help
python experiment_truncated_phi4.py --help
```

The main shared controls are:

| Option | Meaning |
| --- | --- |
| `--chains` | Independent endpoints \(n\) used at each stage. |
| `--steps` | ULMC transitions \(N\) per stage. |
| `--stages` | Number of adaptive stages \(K\). |
| `--step-size` | ULMC integration step size \(h\). |
| `--friction` | ULMC friction coefficient. |
| `--cooling-gamma` | Geometric cooling factor. |
| `--repeats` | Independent Monte Carlo repetitions. |
| `--dtype` | `float32` or `float64`. |
| `--seed` | Root JAX/NumPy random seed. |

In the algorithm API, `alpha` and `delta` are theoretical tuning metadata.
They motivate caller-selected budgets and integrator settings but do not
alter a recurrence once those settings are supplied explicitly. ULMC
transitions evaluate the supplied gradient; the potential callable is kept
in the interface so targets are represented consistently.

### Reducing the lattice workload

The full lattice defaults are intentionally substantial: they include lattice
sides through \(d=100\), diagnostic trajectories, four-method stage
comparisons, and three parameter sweeps. The \(d=100\), 8,192-sample
diagnostic trajectory uses a temporary on-disk array of roughly 313 MiB,
before the script's free-space safety margin. Temporary trajectory storage is
deleted automatically.

Use the following flags to omit independent components:

```text
--skip-diagnostics
--skip-comparisons
--skip-parameter-sweeps
```

Dense covariance methods are skipped automatically when the lattice exceeds
`--dense-max-side` or when \(n\le d^2\), which would make the empirical
covariance rank deficient.

`emcee` is needed for integrated autocorrelation-time diagnostics. If it is
not available, the rest of the phi-4 experiment can still be run with
`--skip-diagnostics`.

## Library API

The public functions exported by `gaussian_cooling_algs.py` are:

- `ulmc`
- `transformed_ulmc`
- `ulmc_coefficients`
- `sample_mean`
- `sample_covariance`
- `sample_power_spectrum`
- `symmetric_matrix_sqrt`
- `apply_translation_invariant_spectrum`
- `translation_invariant_covariance`
- `gaussian_cooling`
- `translation_invariant_gaussian_cooling`

Example:

```python
import jax
import jax.numpy as jnp

from gaussian_cooling_algs import ulmc


def potential(x):
    return 0.5 * jnp.vdot(x, x)


def gradient(x):
    return x


samples = ulmc(
    jax.random.PRNGKey(0),
    potential,
    gradient,
    minimizer=jnp.zeros(4),
    friction_gamma=1.0,
    smoothness_L=1.0,
    step_size_h=0.05,
    num_steps=100,
    num_chains=256,
)
print(samples.shape)  # (256, 4)
```

Sampler calls return independent-chain endpoints rather than complete
trajectories. `sample_covariance` uses the biased \(1/n\) normalization and
therefore has rank at most \(n-1\). For large periodic lattices, retain the
Fourier spectrum returned by translation-invariant Gaussian cooling instead
of materializing a dense covariance.

The library does not globally enable JAX 64-bit mode. Enable it before
constructing arrays when needed:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

Changing target function objects, array shapes, or static stage counts can
trigger a new JAX compilation.

## Reproducibility

- Every experiment accepts a root seed and uses explicit JAX PRNG keys.
- Parameter sweeps use common random numbers to reduce noise in comparisons.
- Reported lines are medians and uncertainty bands are interquartile ranges.
- Floating-point reductions and compiled kernels can vary slightly across
  JAX versions and hardware backends.
- The exact package versions used during development are listed above.

## Citation and license

No license or canonical citation was supplied with the research code. Before
publishing the repository, add the license chosen by the copyright holder and
the appropriate paper/preprint citation. A public GitHub repository without a
license does not grant general reuse rights.
