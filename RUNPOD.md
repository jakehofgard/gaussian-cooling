# Running the experiments on a RunPod H100

The cost-efficient setup is one H100, a pinned container image, and a
persistent network volume. Put Python, GPU-enabled JAX, and the repository in
the image. Use the network volume for compilation caches, job definitions,
logs, and results. This avoids package installation after the Pod starts and
allows it to be terminated as soon as the experiment finishes. Scheduling and
a cold multi-GB image pull can still add startup time.

The scripts currently use one JAX device. Requesting multiple H100s will not
make them faster.

## 1. Create the network volume first

RunPod network volumes are tied to one data center. Check that the data center
offers the desired H100 before creating the volume. The volume must be selected
when the Pod is created, and its location restricts the available GPU choices.
Network volumes are available to Pods in Secure Cloud and are mounted at
`/workspace`. See RunPod's
[storage overview](https://docs.runpod.io/pods/storage/types) and
[network-volume guide](https://docs.runpod.io/storage/network-volumes).

Use a 30 GB standard network volume initially. That is enough for the runtime
fallback below, JAX caches, logs, and PDFs, and the volume can be enlarged
later. Create this layout once:

```text
/workspace/
├── gaussian-cooling-runtime/
│   ├── cache/
│   │   ├── jax/
│   │   ├── matplotlib/
│   │   ├── pip/
│   │   └── xdg/
│   └── venv/                 # only needed without a custom image
├── jobs/                     # saved commands or runner scripts
├── runs/                     # logs, metadata, and output PDFs
└── src/gaussian-cooling/     # only needed without image-baked source
```

Do not store API keys, S3 credentials, or registry credentials on the volume.
Use RunPod Secrets or template environment variables.

If the selected data center supports RunPod's S3-compatible API, the volume
can be populated and results can be downloaded without paying for a Pod. The
[S3 API guide](https://docs.runpod.io/storage/s3-api) lists the supported data
centers and endpoint format. With the separate S3 credentials configured in
the AWS CLI, pre-stage job files with:

```bash
export GC_VOLUME_ID="YOUR_NETWORK_VOLUME_ID"
export GC_DC_ID="YOUR_DATA_CENTER_ID"
aws s3 sync ./runpod-assets/ "s3://${GC_VOLUME_ID}/" \
  --region "${GC_DC_ID}" \
  --endpoint-url "https://s3api-${GC_DC_ID}.runpod.io/"
```

## 2. Prepare the software before renting an H100

### Recommended: a pinned custom image

Build a versioned Docker image containing:

- Python 3.12;
- the dependencies in `requirements.txt`;
- GPU-enabled JAX, pinned to an explicit version;
- `bash` and `curl` for the unattended runner and self-termination step;
- this repository at `/opt/gaussian-cooling`.

Use a Git-SHA tag that is never overwritten, or preferably an image digest;
do not use `latest`. Do not place the image-baked repository under
`/workspace`, because mounting the network volume hides anything the image put
there. RunPod documents this workflow in
[custom templates](https://docs.runpod.io/pods/templates/create-custom-template).

The repository includes `Dockerfile.runpod`. After logging in to a container
registry, build and push the Linux/AMD64 image before renting the H100:

```bash
export GC_IMAGE_REPO="ghcr.io/YOUR_ACCOUNT/gaussian-cooling"
export GC_GIT_SHA="$(git rev-parse HEAD)"
export GC_IMAGE_REF="${GC_IMAGE_REPO}:${GC_GIT_SHA}"

docker buildx build --platform linux/amd64 --push \
  --file Dockerfile.runpod \
  --build-arg GC_GIT_SHA="${GC_GIT_SHA}" \
  --build-arg GC_IMAGE_REF="${GC_IMAGE_REF}" \
  --tag "${GC_IMAGE_REF}" .
```

The explicit platform is important when building from an Apple Silicon Mac.
The Dockerfile installs `jax[cuda13]==0.11.0` with JAX's pip-supplied CUDA
libraries. Do not install `requirements.txt` alone on the GPU Pod: its generic
JAX constraint does not select an NVIDIA backend.

CUDA 13 wheels require an NVIDIA driver at least 580. On a Pod with a driver
from 525 through 579, add `--build-arg JAX_CUDA_EXTRA=cuda12` to the build and
give it a separate cache directory. The version is deliberately pinned for
reproducibility; update the pin and cache name together. JAX recommends the
pip-supplied CUDA libraries, which avoid depending on a matching CUDA toolkit
inside the image. Do not let `LD_LIBRARY_PATH` point to an incompatible CUDA
or cuDNN installation. See the official
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

### Fallback: a persistent virtual environment

If building an image is inconvenient, attach the volume to an inexpensive
preparation Pod and install the environment there before starting the H100.
Use the same Linux image and Python minor version that the H100 Pod will use.

```bash
mkdir -p /workspace/gaussian-cooling-runtime/cache/{jax,matplotlib,pip,xdg}
mkdir -p /workspace/jobs /workspace/runs /workspace/src

export GC_REPOSITORY_URL="YOUR_REPOSITORY_URL"
git clone "${GC_REPOSITORY_URL}" /workspace/src/gaussian-cooling

python3.12 -m venv /workspace/gaussian-cooling-runtime/venv
source /workspace/gaussian-cooling-runtime/venv/bin/activate
export PIP_CACHE_DIR=/workspace/gaussian-cooling-runtime/cache/pip

python -m pip install --upgrade pip wheel
python -m pip install \
  -r /workspace/src/gaussian-cooling/requirements.txt \
  "jax[cuda13]==0.11.0"
python -m pip check
python -m pip freeze \
  > /workspace/gaussian-cooling-runtime/requirements.lock
```

This avoids installation time on the H100. An image is still faster to start
and import from because a network filesystem is relatively slow for the many
small files in a Python environment.

## 3. Create a reusable Pod template

Configure a private NVIDIA RunPod template with:

- the pinned image from the previous section;
- about 20--30 GB of container disk;
- `/workspace` as the volume mount path;
- enough host RAM and CPU for NumPy eigendecompositions and IAT analysis;
- SSH only if interactive debugging is needed;
- no Jupyter server for unattended production jobs.

The template UI accepts environment variables as key/value entries. Add these
before Python imports JAX (use `export KEY=value` only when placing them in a
shell startup script):

```text
PYTHONUNBUFFERED=1
MPLBACKEND=Agg
JAX_PLATFORMS=cuda
JAX_COMPILATION_CACHE_DIR=/workspace/gaussian-cooling-runtime/cache/jax/h100-sxm-jax-0.11.0-cuda13
JAX_COMPILATION_CACHE_MAX_SIZE=10000000000
MPLCONFIGDIR=/workspace/gaussian-cooling-runtime/cache/matplotlib
XDG_CACHE_HOME=/workspace/gaussian-cooling-runtime/cache/xdg
PIP_CACHE_DIR=/workspace/gaussian-cooling-runtime/cache/pip
```

Use a different JAX cache directory when the JAX version, CUDA wheel, or H100
variant changes. JAX cache keys include compiled code, shapes, JAX/XLA
versions, flags, and GPU topology, so an H100 SXM cache should not be assumed
to work for H100 PCIe or H100 NVL. See the
[JAX persistent-cache documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).
The 10 GB limit prevents an unbounded cache from filling the recommended
volume; monitor it and change the limit deliberately for larger shape grids.

Do not let two Pods write the same JAX, Matplotlib, or output directory
simultaneously. If concurrent Pods are ever needed, give each Pod directories
that include its Pod ID. RunPod warns that concurrent writes to a shared
network volume can corrupt data.

Leave JAX's default GPU-memory preallocation enabled. It is usually the
fastest and least fragmented setting for a single process. Change it only
after observing an out-of-memory error; the alternatives are documented in
[JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

Save the production runner at `/workspace/jobs/run_h100.sh` and set the
template start command to:

```text
bash /workspace/jobs/run_h100.sh
```

For an initial interactive debug Pod, temporarily use a keep-alive start
command and enable SSH. Always retain a hard termination deadline.

Select the GPU, Secure Cloud, data center, and network volume when deploying
the Pod; those are deployment choices, not template fields. From the local
machine, a CUDA 13 deployment is:

```bash
export GC_TEMPLATE_ID="YOUR_TEMPLATE_ID"
runpodctl pod create \
  --name gaussian-cooling-production \
  --template-id "${GC_TEMPLATE_ID}" \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --cloud-type SECURE \
  --data-center-ids "${GC_DC_ID}" \
  --network-volume-id "${GC_VOLUME_ID}" \
  --min-cuda-version 13.0 \
  --terminate-after 6h
```

For a `jax[cuda12]` image, use `--min-cuda-version 12.1`. This filter prevents
a CUDA 13 image from being scheduled on a host with an incompatible driver.
Treat `--terminate-after` as a safety limit, not the normal shutdown path. The
runner should terminate the Pod as soon as its outputs have been flushed. See
the
[RunPod Pod CLI reference](https://docs.runpod.io/runpodctl/reference/runpodctl-pod).

## 4. Verify the H100 immediately

Run this as soon as the Pod starts. `JAX_PLATFORMS=cuda` makes a missing CUDA
backend fail instead of silently running on the CPU.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader

python - <<'PY'
import jax
import jax.numpy as jnp

device = jax.devices()[0]
print("JAX:", jax.__version__)
print("devices:", jax.devices())
assert "H100" in device.device_kind, device

x = jnp.ones((4096, 4096), dtype=jnp.float32)
y = jax.jit(lambda a: a @ a)(x).block_until_ready()
print("GPU check passed:", float(y[0, 0]))
PY
```

If using the persistent-venv fallback, activate it first:

```bash
source /workspace/gaussian-cooling-runtime/venv/bin/activate
export GC_REPO=/workspace/src/gaussian-cooling
```

With an image-baked repository, use:

```bash
export GC_REPO=/opt/gaussian-cooling
```

## 5. Smoke-test, then benchmark the exact production shape

Create a unique output directory and retain the complete terminal log:

```bash
set -Eeuo pipefail
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${RUNPOD_POD_ID:-manual}"
export RUN_DIR="/workspace/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
cd "${GC_REPO}"
```

Run both smoke tests:

```bash
python -u experiment_transformed_gaussian.py \
  --quick --dtype float32 --repeats 1 --convergence-repeats 1 \
  --steps 16 --stages 3 \
  --output-prefix "${RUN_DIR}/gaussian_smoke" \
  2>&1 | tee "${RUN_DIR}/gaussian_smoke.log"

python -u experiment_truncated_phi4.py \
  --quick --repeats 1 --steps 8 --stages 2 --dense-max-side 0 \
  --skip-diagnostics --skip-comparisons --skip-parameter-sweeps \
  --output-prefix "${RUN_DIR}/phi4_smoke" \
  2>&1 | tee "${RUN_DIR}/phi4_smoke.log"
```

A smoke test validates the installation but does not warm the production JAX
cache: changing lattice size, dimension, chain count, stage count, step count,
or dtype can trigger a new compilation. Before committing to a long run, time
one repeat with the exact production shapes and inspect `nvidia-smi` while it
runs. Keep that compiled cache for the full job. A cheaper GPU can validate the
image but cannot usefully warm this cache for an H100, because GPU identity and
topology participate in the cache key.

At the quoted H100 price of $2.99/hour, the measured cost is

```text
cost in dollars = wall time in seconds * 2.99 / 3600
```

so one idle minute costs about $0.05.

## 6. Suggested large-run commands

These are starting points, not statistically sufficient settings for every
study. The candidate production commands use three repeats. First make an
otherwise identical benchmark run with `--repeats 1` and a `_benchmark`
output prefix, then increase the sampling budget after inspecting its plots,
timing, and peak memory. The phi4 section spells out that ladder explicitly.

### Transformed Gaussian

```bash
python -u experiment_transformed_gaussian.py \
  --dimension 512 --chains 1024 --steps 96 --stages 12 \
  --repeats 3 --convergence-repeats 1 --dtype float64 \
  --output-prefix "${RUN_DIR}/transformed_gaussian_d512" \
  2>&1 | tee "${RUN_DIR}/transformed_gaussian_d512.log"
```

`--chains` must exceed `--dimension`. Algorithmic dense linear algebra runs in
JAX on the GPU, while target construction, validation, and evaluation metrics
contain dense NumPy work on the CPU. Both paths contain cubic-in-dimension
operations, so monitor CPU as well as GPU use. The convergence panel reruns
every prefix from stage 1 through stage K. At K=12 that is 78 stage-equivalents
for each of three sampled methods, or roughly 234 method-stage budgets per
convergence repeat, in addition to evaluation and compilation. Keep
`--convergence-repeats 1` while scaling and increase it only for the final
convergence figure.

### Large translation-invariant lattice phi4 run

The conservative one-command H100 run is:

```bash
python -u experiment_truncated_phi4.py --gpu \
  --output-prefix "${RUN_DIR}/phi4_gpu_scale" \
  2>&1 | tee "${RUN_DIR}/phi4_gpu_scale.log"
```

The flag verifies that JAX sees a GPU, then runs sides 64, 128, 256, 512, and
1024 with 64 chains, 128 reference chains, 32 transitions per stage, eight
stages, one repeat, and float32. It also writes stagewise-convergence plots for
sides 512 and 1024. Those histories use only Fourier spectra and therefore
remain linear in `D`, but they add separate runs for each of the three scalable
methods. The full empirical covariance baseline is omitted because 64 samples
give a rank-deficient covariance at these dimensions. The preset disables
dense matrices, diagnostics, and parameter sweeps. Explicit flags override
every preset value; add `--skip-comparisons` when only feasibility and timing
matter, or use `--comparison-sides` to select different history sizes.

If the first run fits comfortably, increase precision deliberately, for
example:

```bash
python -u experiment_truncated_phi4.py --gpu \
  --chains 128 --reference-chains 256 --repeats 3 \
  --output-prefix "${RUN_DIR}/phi4_gpu_scale_n128" \
  2>&1 | tee "${RUN_DIR}/phi4_gpu_scale_n128.log"
```

For a side length `d`, the state dimension is `D=d^2`. The Fourier covariance
representation is `O(D)`, but live chain storage is `O(chains * D)` and the
batched FFT work is approximately `O(chains * D log D)`. In float32 at
`d=1024`, `chains=64`, one real chain-state array is 256 MiB and the explicit
`(chains, D, 2)` Gaussian-noise tensor is 512 MiB; complex FFT workspaces and
other live arrays add substantially to this. Float64 doubles the real-array
sizes. A dense `D`-by-`D` float32 matrix would require 4 TiB, which is why the
preset disables every dense path. Each method also executes one unmeasured
warmup before its requested repeat.

Run smaller, qualitatively different outputs separately when needed:

```bash
# Stage comparisons; raw full covariance appears only where rank/size permit.
python -u experiment_truncated_phi4.py \
  --sides 10,100 --comparison-sides 10,100 \
  --skip-diagnostics --skip-parameter-sweeps \
  --output-prefix "${RUN_DIR}/phi4_stage_comparison" \
  2>&1 | tee "${RUN_DIR}/phi4_stage_comparison.log"

# Lambda, mass, and radius sweeps; the default sweep lattice is d=10.
python -u experiment_truncated_phi4.py \
  --sides 2 --repeats 3 --steps 64 --stages 2 \
  --parameter-sweep-stages 12 \
  --skip-diagnostics --skip-comparisons \
  --output-prefix "${RUN_DIR}/phi4_parameter_sweeps" \
  2>&1 | tee "${RUN_DIR}/phi4_parameter_sweeps.log"

# Post-cooling two-point-correlator IAT and ergodicity diagnostics.
export TMPDIR="/tmp/gaussian-cooling-${RUN_ID}"
mkdir -p "${TMPDIR}"
python -u experiment_truncated_phi4.py \
  --sides 100 --repeats 1 --dense-max-side 0 \
  --skip-comparisons --skip-parameter-sweeps \
  --output-prefix "${RUN_DIR}/phi4_diagnostics_d100" \
  2>&1 | tee "${RUN_DIR}/phi4_diagnostics_d100.log"
```

The parameter sweeps use small dense problems, and the diagnostics generate a
single trajectory followed by CPU-side autocorrelation analysis. They can
leave an H100 underutilized; after validating the environment, benchmark them
on a less expensive GPU. Keep diagnostic memmaps in local `/tmp`, as above,
and only persist the logs and PDFs. Their storage is approximately
`trajectory_samples * d^2 * bytes_per_value`: the defaults use about 313 MiB
at `d=100` in float32, while `d=500` would use about 7.6 GiB. Ensure the Pod's
local disk has room.

The parameter-sweep reference sampler automatically reduces its step size for
stiff targets, which can substantially increase its transition count at small
mass, large quartic coupling, or large radius. Benchmark the hardest requested
sweep points before running all repeats. For a cheap exploratory sweep, reduce
`--repeats` and `--steps`; retain `--parameter-sweep-stages 12` unless the
stage budget itself is deliberately under study.

With the stage-comparison defaults, the raw full-covariance curve is available
at `d=10` but omitted at `d=100`, where `chains <= D` and the dense cutoff is
exceeded. The diagnostics command also reruns all main preconditioner methods
and the reference estimate at `d=100` before generating its single diagnostic
trajectory; include that duplicated work in the cost estimate.

For exploratory phi4 scaling, float32 is substantially smaller and is the
script default. Use float64 for final runs when sensitivity of the spectral
metrics warrants it. The transformed-Gaussian experiment defaults to float64.

## 7. Record provenance and terminate promptly

Before the production command, save enough information to reproduce it:

```bash
{
  date -u
  nvidia-smi
  python -m pip freeze
  printf 'image_ref=%s\n' "${GC_IMAGE_REF:-unknown}"
  printf 'git_sha=%s\n' "${GC_GIT_SHA:-unknown}"
  git -C "${GC_REPO}" rev-parse HEAD 2>/dev/null || true
} > "${RUN_DIR}/environment.txt"
```

`Dockerfile.runpod` bakes `GC_IMAGE_REF` and `GC_GIT_SHA` into the image. This
is more reliable than `git rev-parse` alone, because production images usually
exclude the `.git` directory.

Write the exact command to `${RUN_DIR}/command.txt`. After a successful run:

```bash
touch "${RUN_DIR}/_SUCCESS"
sync
```

For a fully unattended job, put the commands in
`/workspace/jobs/run_h100.sh`, start that script from the Pod template, and add
this cleanup trap after defining `RUN_DIR`. It records success or failure,
flushes the network volume, and deletes the Pod through RunPod's REST API:

```bash
set -Eeuo pipefail

finish_run() {
  status=$?
  trap - EXIT
  set +e
  if (( status == 0 )); then
    touch "${RUN_DIR}/_SUCCESS"
  else
    touch "${RUN_DIR}/_FAILED"
  fi
  sync
  if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
    curl -fsS -X DELETE \
      "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" || true
  fi
  exit "${status}"
}
trap finish_run EXIT
```

RunPod supplies the Pod ID and a Pod-scoped API key as runtime environment
variables; do not copy the key into a log. See the
[Pod environment-variable reference](https://docs.runpod.io/pods/templates/environment-variables).
Keep the independent `--terminate-after` deadline in case the runner or API
request fails.

On failure, preserve the log and write `_FAILED`. The experiment scripts save
their PDFs only near the end, so splitting independent components as above
reduces the amount of work lost to an interruption.

Stopping a Pod preserves `/workspace`, but releases its GPU allocation, so a
restart may return with no GPU available. For unattended batch work, terminate
the Pod immediately after `sync` and redeploy against the same persistent
volume when more work is needed. See RunPod's
[Pod lifecycle documentation](https://docs.runpod.io/pods/manage-pods).

Retrieve results through the S3 API without starting compute. Run these
commands on the local machine; list the volume first because the `RUN_ID` was
created inside the Pod:

```bash
aws s3 ls "s3://${GC_VOLUME_ID}/runs/" \
  --region "${GC_DC_ID}" \
  --endpoint-url "https://s3api-${GC_DC_ID}.runpod.io/"

export RUN_ID="THE_RUN_ID_TO_DOWNLOAD"
aws s3 sync \
  "s3://${GC_VOLUME_ID}/runs/${RUN_ID}/" "./runs/${RUN_ID}/" \
  --region "${GC_DC_ID}" \
  --endpoint-url "https://s3api-${GC_DC_ID}.runpod.io/"
```

Copy publication outputs to durable storage. RunPod describes network volumes
as active-workload storage, not a substitute for a backup.

## Preflight checklist

- Network volume is in a data center with the intended H100 available.
- Image tag, Python, JAX, CUDA wheel, and cache directory are versioned.
- The CUDA assertion reports the expected H100.
- Exactly one GPU and one experiment process are running.
- Every job has a unique output directory and log.
- Exact production shapes were benchmarked with one repeat.
- Large phi4 jobs use `--dense-max-side 0`.
- A hard termination deadline is set, and normal shutdown happens immediately
  after `sync`.
