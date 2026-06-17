# DynamiQ SIGCOMM Artifact

This repository contains both:

* the multi-node testbed evaluation used for the SIGCOMM artifact reproduction under `testbed_evaluation/`, and
* the LLM simulation workloads under `simulations_llm/`.

## Testbed Assumptions

This artifact targets the testbed configuration used for the SIGCOMM evaluation:

* 4 GPU nodes.
* 2 NVIDIA RTX A6000 Ada-class GPUs per node.
* The two GPUs on each node are connected by an intra-node NVLink link reported as `NV4`.
* The reproduced paper-artifact configuration uses one 100 Gbps RDMA NIC rail per node.
* Each node launches 2 ranks, so the full 4-node evaluation runs with `WORLD_SIZE=8`.

The code also supports a two-rail RDMA configuration via `--rails 2`, `--iface0`, and `--iface1`. However, the validated testbed reproduction uses `--rails 1`. Treat the two-rail path as an additional supported code path rather than the default paper-reproduction configuration.

Before running the full evaluation, select four idle nodes and export them as a comma-separated list:

```bash
export DYNAMIQ_NODES=node0,node1,node2,node3
```

Each selected node should have an available NV4-connected GPU pair. The launcher can try several candidate GPU pairs via `--gpu-pair-starts`.

## Environment

Set the repository root as:

```bash
export DYNAMIQ_HOME=/path/to/dynamiq_artifact
cd "$DYNAMIQ_HOME"
```

Create the Python environment and install the Python dependencies:

```bash
conda create -n llm python=3.9 pip
conda activate llm

# Install a CUDA 11.8 PyTorch build, then verify NCCL reports 2.19.3.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

See `ENVIRONMENT_REQUIREMENTS.md` for the tested package versions and the
non-Python requirements, including CUDA, NCCL, compiler, RDMA, and build tools.

The qsub wrappers assume a conda environment with PyTorch, CUDA, NCCL, and a compatible C++ compiler. As an example, the default values in our local environments are:

```bash
CONDA_ENV=llm
CUDA_HOME=/share/apps/cuda-11.8
GCC_HOME=/share/apps/gcc-8.3
TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9'
```

Override these to your own paths:

```bash
--conda-env <env-name>
--cuda-home <cuda-path>
--gcc-home <gcc-path>
--cuda-arch-list <arch-list>
```

The same submit wrappers can be used in two ways:

* `--launch-mode qsub` (default): submit one SGE job per node.
* `--launch-mode direct`: `ssh` into each selected host and invoke the same `qsub_*_node.zsh` launcher directly.

For `--launch-mode direct`, ensure that:

* the repository is visible at the same `$DYNAMIQ_HOME` path on every selected node,
* the launching machine can `ssh` to each selected node without interactive prompts, and
* the conda/CUDA/GCC paths passed to the wrapper are valid on every selected node.

For an interactive sanity check:

```bash
source ~/.zshrc
conda activate llm
cd "$DYNAMIQ_HOME"

python - <<'PY'
import torch
print("python ok")
print("torch", torch.__version__)
print("cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("nccl", torch.cuda.nccl.version() if torch.cuda.is_available() else "n/a")
PY
```

The training scripts download datasets and models through Hugging Face when they
are not already cached. By default they use Hugging Face's normal cache
locations, usually under `~/.cache/huggingface`.

To use explicit cache directories, either export:

```bash
export DYNAMIQ_DATA_CACHE="$DYNAMIQ_HOME/hf_data"
export DYNAMIQ_MODEL_CACHE="$DYNAMIQ_HOME/hf_models"
```

or pass cache paths to the launchers:

```bash
--data-cache-dir "$DYNAMIQ_HOME/hf_data"
--model-cache-dir "$DYNAMIQ_HOME/hf_models"
```

The Llama and Gemma runs use gated Hugging Face model repositories. Before
running those jobs on a new machine, accept the model licenses on Hugging Face
and authenticate once:

```bash
huggingface-cli login
```


# Testbed evaluation

We first provide instructions on how to run the throughput performance evaluation in the `testbed_evaluation/` folder. These are used to reproduce the main experiments in `figure 6`.

Before launching the testbed jobs, generate the fixed sampling-order files:

```bash
cd "$DYNAMIQ_HOME/testbed_evaluation"
python gen_rand.py
```

This writes deterministic sampling-order files under:

```bash
testbed_evaluation/models/
```

In particular, `gen_rand.py` creates the `indices_2_3.pkl`, `indices_4_3.pkl`, and `indices_8_3.pkl` files used by the testbed training runs.

## Build the Kernels

Build the CUDA compression extension and copy it into the testbed hook package:

```bash
cd "$DYNAMIQ_HOME/testbed_evaluation"
./build_eden_utils_llm.zsh
```

This runs `cuda_kernels/setup.py build_ext --inplace` and copies the resulting `eden_utils*.so` into:

```bash
testbed_evaluation/new_comm_hooks/
```

The RDMA all-reduce extension under `rdma_comm_compress/ring_allreduce_ext/` is JIT-built on first import by PyTorch. To prebuild it explicitly:

```bash
cd "$DYNAMIQ_HOME"
source ~/.zshrc
conda activate llm

python - <<'PY'
from rdma_comm_compress.ring_allreduce_ext import load_ring_allreduce_ext
load_ring_allreduce_ext(verbose=True)
PY
```

## Smoke Test

Run a small communication-hook smoke test before launching the full evaluation:

```bash
cd "$DYNAMIQ_HOME/testbed_evaluation"

./submit_qsub_smoke_new_comm_hooks.zsh \
  --nodes "$DYNAMIQ_NODES" \
  --aggregation-method dynamiQ_aee_5bit \
  --steps 1 \
  --numel 1048576 \
  --rails 1 \
  --sync
```

If you are not using SGE, launch the same smoke test directly over `ssh` with:

```bash
cd "$DYNAMIQ_HOME/testbed_evaluation"

./submit_qsub_smoke_new_comm_hooks.zsh \
  --launch-mode direct \
  --nodes "$DYNAMIQ_NODES" \
  --aggregation-method dynamiQ_aee_5bit \
  --steps 1 \
  --numel 1048576 \
  --rails 1
```

In direct mode, the wrapper uses `ssh` to invoke `qsub_smoke_new_comm_hooks_node.zsh` on each selected host and waits for those node-level launchers to finish.

To exercise the butterfly all-reduce path in the smoke test, add `--topology butterfly`.
For example:

```bash
cd "$DYNAMIQ_HOME/testbed_evaluation"

./submit_qsub_smoke_new_comm_hooks.zsh \
  --topology butterfly \
  --nodes "$DYNAMIQ_NODES" \
  --aggregation-method dynamiQ_mee_5bit \
  --steps 1 \
  --numel 1048576 \
  --rails 1 \
  --expect-butterfly-rdma \
  --sync
```

With `--topology butterfly`, the launcher automatically rewrites methods such as
`bf16`, `omnireduce`, `thc`, or `dynamiQ_mee_5bit_dynamic_bitrate` to their
`*_butterfly` variants if needed.

The reproduced artifact configuration uses `--rails 1`.

On systems with two working RDMA interfaces, the two-rail path can be exercised with:

```bash
--rails 2 --iface0 <iface0> --iface1 <iface1>
```

Smoke-test outputs are written to:

```bash
testbed_evaluation/smoke_job_results/<run_id>/
```

A successful run should produce per-node stdout/stderr files and `node_<rank>.status` files without failures.

## End-to-End Evaluation

The full testbed launcher is:

```bash
cd "$DYNAMIQ_HOME/testbed_evaluation"

./submit_qsub_llm_hook_matrix.zsh \
  --nodes "$DYNAMIQ_NODES" \
  --rails 1 \
  --dynamic-pipeline-rdma 1 \
  --pipeline-chunk-mb 8 \
  --pipeline-inflight 2
```

If you are not using SGE, launch the same evaluation directly over `ssh` with:

```bash
cd "$DYNAMIQ_HOME/testbed_evaluation"

./submit_qsub_llm_hook_matrix.zsh \
  --launch-mode direct \
  --nodes "$DYNAMIQ_NODES" \
  --rails 1 \
  --dynamic-pipeline-rdma 1 \
  --pipeline-chunk-mb 8 \
  --pipeline-inflight 2
```

In direct mode, the wrapper uses `ssh` to invoke `qsub_llm_hook_matrix_node.zsh` on each selected host. The per-node `ssh` launcher logs are written as `*.direct.out` and `*.direct.err`, while the per-method training logs remain under the usual run directory.

By default, this runs four task/model combinations:

* `causal:llama:meta-llama/Llama-3.2-1B`
* `causal:gemma:gemma`
* `mmlu:llama:meta-llama/Llama-3.2-1B`
* `wikitext:bert-large:bert-large-cased`

The task names map to the following training scripts:

* `causal` → `train_llm_causal.py`
* `mmlu` → `train_llm_mmlu.py`
* `wikitext`, `maskedlm`, or `mlm` → `train_llm_maskedlm_bert.py`

Results are written to:

```bash
testbed_evaluation/llm_hook_matrix_results/<run_id>/
```

Each run directory contains per-method logs, `command.log`, scheduler stdout/stderr, and `node_<rank>.status` files.

For a shorter command-generation check:

```bash
./submit_qsub_llm_hook_matrix.zsh \
  --nodes "$DYNAMIQ_NODES" \
  --combos wikitext:bert-large:bert-large-cased \
  --aggregation-methods bf16,dynamiQ_mee_5bit_dynamic_bitrate \
  --num-train-epochs 1 \
  --rails 1 \
  --dry-run
```

Remove `--dry-run` to submit the job.

The same `--dry-run` flag also works with `--launch-mode direct`; in that case the wrapper prints the exact `ssh` commands instead of running them.

## Butterfly All-Reduce

The testbed launchers also expose:

```bash
--topology butterfly
```

This keeps the usual method names at the command line and maps them to butterfly
variants internally, for example:

* `bf16` → `bf16_butterfly`
* `omnireduce` → `omnireduce_butterfly`
* `dynamiQ_mee_5bit_dynamic_bitrate` → `dynamiQ_mee_5bit_dynamic_bitrate_butterfly`

To reproduce the butterfly-allreduce testbed runs corresponding to Table 5, run:

```bash
cd "$DYNAMIQ_HOME/testbed_evaluation"

./submit_qsub_llm_hook_matrix.zsh \
  --topology butterfly \
  --nodes "$DYNAMIQ_NODES" \
  --aggregation-methods bf16,omnireduce,thc,dynamiQ_mee_5bit_dynamic_bitrate \
  --rails 1 \
  --dynamic-pipeline-rdma 0 \
  --pipeline-chunk-mb 8 \
  --pipeline-inflight 2
```

If you are not using SGE, add `--launch-mode direct` to the same command.

## Important Parameters

* `--nodes`: comma-separated host list. Use four nodes for the full paper-artifact reproduction. Each node launches two ranks.
* `--launch-mode`: `qsub` submits through SGE; `direct` launches the same node-level scripts over `ssh`.
* `--topology`: `ring` keeps the default topology; `butterfly` rewrites the selected methods to butterfly all-reduce variants.
* `--aggregation-methods`: comma-separated communication hooks. The default is `bf16,MXfp8,fp4,fp6,zero,dynamiQ_aee_5bit,dynamiQ_mee_5bit,dynamiQ_mee_5bit_dynamic_bitrate,omnireduce,thc`. To reproduce figure 7 and table 4 with varied bitrate for DynamiQ, simply change `5bit` to `3bit`, `4bit`, `5bit`, `6bit` and `7bit`.
* `--rails`: number of RDMA rails. Use `--rails 1` for the reproduced paper-artifact configuration. Use `--rails 2` only on systems with two working RDMA interfaces.
* `--iface0`, `--iface1`: network interface names for one-rail or two-rail RDMA runs.
* `--dynamic-pipeline-rdma`: enables the pipelined dynamic AEE/MEE RDMA path.
* `--pipeline-chunk-mb`: RDMA pipeline tile size.
* `--pipeline-inflight`: number of in-flight tiles per rail.
* `--gpu-pair-starts`: ordered GPU pair starts to try on each node. The selected pair should be connected by NVLink/NV4.
* `--max-used-mb`: treats a GPU as busy if its memory usage is above this threshold.
* `--data-cache-dir`, `--model-cache-dir`: optional Hugging Face cache directories for automatic dataset and model downloads.
* `--per-device-train-batch-size`, `--learning-rate`, `--num-train-epochs`: forwarded to the training scripts.
* `--extra-train-args`: raw extra arguments appended to each training command.

## Bandwidth Microbenchmarks

The communication module can be tested independently:

```bash
cd "$DYNAMIQ_HOME/rdma_comm_compress"

./submit_qsub_ring_allreduce.zsh \
  --nodes node0,node1 \
  --mode quantized \
  --nbits 4 \
  --rails 1 \
  --iface0 <iface0> \
  --numel 268435456 \
  --iters 20 \
  --warmup-iters 3 \
  --sync
```

For two-rail experiments on a system with two RDMA interfaces:

```bash
--rails 2 --iface0 <iface0> --iface1 <iface1>
```

See `rdma_comm_compress/README.md` for the ring all-reduce topology, bandwidth metrics, and debugging notes.

## Simulation Jobs

The repository also includes simulation-based LLM experiments under `simulations_llm/`. These jobs are used for the end-to-end simulated train without caring about the performance (speed), getting the round-to-accuracy results. These results, together with the testbed evaluation where we can measure the time-per-round data, will be used to reproduce the end-to-end time-to-accuracy (TTA) figures such as `figure 4` and `figure 5`.

For the full simulation workflow, including:

* required pre-generated correlated-random data,
* `qsub` and direct `ssh` launch modes,
* ring and butterfly topology selection,
* worker-count and node-count conventions, and
* concrete command-line launch examples, including the butterfly-allreduce runs for Figure 9,

see [simulations_llm/README.md](simulations_llm/README.md).

Typical simulation entry points from the repository root are:

```bash
python simulations_llm/gen_correlated_rand.py
simulations_llm/submit_qsub_simulation_matrix.zsh --dry-run
simulations_llm/submit_qsub_simulation_matrix.zsh --launch-mode direct --dry-run
```

## Scope of This Artifact

This repository contains the testbed and simulation artifact code for DynamiQ compression-aware distributed training.

* `cuda_kernels/`: CUDA compression and decompression kernels used by the DynamiQ communication hooks.
* `rdma_comm_compress/`: RDMA/NVLink communication microbenchmarks and the staged GPU→CPU→RDMA→CPU→GPU transport used by the hooks.
* `simulations_llm/`: simulation-based distributed LLM training experiments and their launch scripts.
* `testbed_evaluation/`: end-to-end training evaluations for causal language modeling, MMLU, and Wikitext-103 masked language modeling.
