# Efficient Mass Matrix Estimation with Gaussian Cooling

This repository contains the experiments for *Efficient Mass Matrix Estimation with Gaussian Cooling*.

There are two groups of experiments, corresponding to Sections 4.1 and 4.2 in the paper.

- **Transformed Gaussians:** well-conditioned Gaussians under a random, ill-conditioned linear map (Section 4.1).
- **Lattice $\phi^4$:** a simplified lattice field theory, with a quartic term (Section 4.2). See the [lattice experiment guide](phi4/PHI4_EXPERIMENTS.md) for a list of experiments and explanations of the figures present in the paper.

## Installation

Run the commands below from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To test the installation, you can run a small version of each experiment group:

```bash
python experiment_transformed_gaussian.py --quick \
  --output-prefix results/gaussian_quick
python -m phi4.experiment_phi4_stage_convergence --quick \
  --output-prefix results/phi4_quick
```

The scripts create the output directory and save the plots as PDF files. `--quick` uses smaller problems and fewer samples so you can check that the code runs. Omit it to use the full experiment settings. `--output-prefix` sets the directory and the first part of each output filename.

## Transformed Gaussian experiments

```bash
python experiment_transformed_gaussian.py \
  --output-prefix figures/transformed_gaussian
```

This command compares three methods:

1. Gaussian cooling.
2. Adaptive preconditioning, which repeatedly estimates the covariance and uses it to precondition the sampler, with no cooling.
3. Underdamped Langevin Monte Carlo (ULMC) without any cooling or preconditioning.

Each method uses the same number of gradient evaluations, $nNK$, where $n$ is the number of chains, $N$ is the number of steps per stage, and $K$ is the number of stages. The third method runs each chain for $NK$ steps to ensure that the total number of first-order queries is the same across all methods.

The command produces two figures:

| File | What it shows |
| --- | --- |
| `transformed_gaussian_preconditioner_quality.pdf` | How the methods perform as the conditioning of the transformation gets increasingly large. |
| `transformed_gaussian_stage_convergence.pdf` | How the estimates improve from stage to stage for a transformation with condition number $\kappa = 10^6$.|

The target is constructed from $z\sim\mathcal N(0,\Sigma_0)$ by setting $z=B^{1/2}x$. Its covariance is therefore $B^{-1/2}\Sigma_0B^{-1/2}$. The experiment increases the condition number of $B$ while keeping its eigenvectors fixed.

To change the problem or the sampling budget, use the following options:

| Options | Description |
| --- | --- |
| `--dimension`, `--base-condition-number`, `--kappas` | The target dimension, the condition number of $\Sigma_0$, and the condition numbers to test for $B$. |
| `--chains`, `--steps`, `--stages` | The sampling budget $n$, $N$, and $K$. |
| `--repeats`, `--convergence-repeats` | The number of independent runs for the first and second figures, respectively. |
| `--seed`, `--dtype` | The random seed and numerical precision. |

For example:

```bash
python experiment_transformed_gaussian.py \
  --chains 512 --steps 128 --stages 12 --repeats 12 \
  --output-prefix figures/gaussian_custom
```

Use `python experiment_transformed_gaussian.py --help` to list all options.
Values supplied on the command line override the corresponding `--quick`
settings. Each lattice experiment has its own options and defaults, described
in [phi4/PHI4_EXPERIMENTS.md](phi4/PHI4_EXPERIMENTS.md).

## Figures

The plots measure how well the estimated covariance rescales the target.
Specifically, they report the relative condition number

```math
\kappa_{\mathrm{rel}}(\widehat\Sigma,\Sigma)
=\kappa\!\left(\widehat\Sigma^{-1/2}\Sigma\widehat\Sigma^{-1/2}\right),
```

where $\widehat\Sigma$ is the estimate and $\Sigma$ is the target covariance.
A smaller value is better; one is ideal. The Gaussian experiments use the exact
target covariance. The lattice experiments estimate it from a separate set of
samples, so those comparisons also depend on the accuracy of the reference.

## Code Organization

| File or directory | Contents |
| --- | --- |
| [gaussian_cooling_algs.py](gaussian_cooling_algs.py) | ULMC samplers, Gaussian cooling, covariance estimates, and the matrix and Fourier calculations used by the experiments. |
| [experiment_transformed_gaussian.py](experiment_transformed_gaussian.py) | The Gaussian experiments and their figures. |
| [phi4/](phi4/PHI4_EXPERIMENTS.md) | The lattice model and experiments for convergence, sampling diagnostics, and changes in model size or parameters. |
| `figures/`, `results/` | Output directories created when experiments run. |
