# Gaussian cooling

JAX implementations of underdamped Langevin Monte Carlo (ULMC) and Gaussian
cooling for learning covariance preconditioners. Two experiment groups compare
cooling with empirical adaptation and unpreconditioned sampling:

- **Transformed Gaussians**: an exact-covariance benchmark in the project root.
- **Lattice $\phi^4$**: non-Gaussian experiments on periodic square lattices in
  [phi4/](phi4/PHI4_EXPERIMENTS.md).

No datasets are required. Experiments use explicit random seeds and save each
figure as a separate vector PDF.

## Project guide

| Location | What is implemented |
| --- | --- |
| [gaussian_cooling_algs.py](gaussian_cooling_algs.py) | Standard and transformed ULMC, dense and translation-invariant Gaussian cooling, covariance estimators, and matrix/Fourier utilities. |
| [experiment_transformed_gaussian.py](experiment_transformed_gaussian.py) | Gaussian benchmark, equal-budget method comparisons, and stage convergence against an exact oracle. |
| [phi4/](phi4/PHI4_EXPERIMENTS.md) | Lattice model, shared sampling helpers, and seven focused experiments: scaling, stages, diagnostics, correlators, parameter sweeps, hardness maps, and budget scaling. |
| `figures/`, `results/` | Generated output; ignored by Git. Scripts create output directories as needed. |

## Install and run

Use Python 3.10 or newer. Run all commands below from this directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start with the reduced experiments:

```bash
python experiment_transformed_gaussian.py --quick \
  --output-prefix results/gaussian_quick
python -m phi4 --quick --output-prefix results/phi4_quick
```

`--quick` reduces the experiment size for a smoke test; omit it to use the
full defaults. `--output-prefix` controls the directory and filename prefix,
and `--help` lists each entry point's options. Explicit numerical options take
precedence over presets. The [lattice experiment guide](phi4/PHI4_EXPERIMENTS.md) gives
commands for each study and describes GPU presets.

## Transformed-Gaussian experiment

The target starts with $z\sim\mathcal N(0,\Sigma_0)$ and applies the coordinate
change $z=B^{1/2}x$, giving covariance $B^{-1/2}\Sigma_0B^{-1/2}$. The benchmark
varies the condition number of $B$ while holding the eigenvectors fixed.

Three methods use the same $nNK$ gradient-evaluation budget, where $n$ is the
chain count, $N$ the steps per stage, and $K$ the number of stages:

1. Gaussian cooling.
2. Staged empirical covariance adaptation on the uncooled target.
3. Unpreconditioned ULMC with $NK$ steps per chain.

```bash
python experiment_transformed_gaussian.py \
  --output-prefix figures/transformed_gaussian

# Example with explicit sampling budgets
python experiment_transformed_gaussian.py \
  --chains 512 --steps 128 --stages 12 --repeats 12 \
  --output-prefix figures/gaussian_custom
```

| Output suffix | What it shows |
| --- | --- |
| `_preconditioner_quality.pdf` | Preconditioner quality versus the condition number of $B$. |
| `_stage_convergence.pdf` | Quality versus cumulative stage at the hardest transformation, including the exact cooled-Gaussian oracle. |

Use `--dimension`, `--base-condition-number`, and `--kappas` to change the
target; `--chains`, `--steps`, `--stages`, and `--repeats` control sampling.
See `python experiment_transformed_gaussian.py --help` for all controls.

## Reading the results and using the library

Preconditioner quality is the relative condition number

```math
\kappa_{\mathrm{rel}}(\widehat\Sigma,\Sigma)
=\kappa\!\left(\widehat\Sigma^{-1/2}\Sigma\widehat\Sigma^{-1/2}\right).
```

Here $\widehat\Sigma$ is the learned preconditioner and $\Sigma$ is the target
covariance. Smaller is better; the ideal value is one. The Gaussian benchmark
uses an analytic covariance; the lattice studies use independent reference
samples, so their quality estimates also have Monte Carlo error.

The reusable samplers in `gaussian_cooling_algs.py` return independent-chain
endpoints. `sample_covariance` uses the biased $1/n$ normalization. For large
lattices, retain Fourier spectra instead of forming dense covariance matrices.
The library leaves JAX precision configuration to the caller; experiment
scripts expose `--dtype` and `--seed`. Results can differ slightly across JAX
versions and hardware.
