# DynamiQ_SIGCOMM_Artifact

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

The training scripts use the following default cache locations:

```bash
DATA_CACHE=/cluster/project2/gcreduce_data/data
MODEL_CACHE=/cluster/project2/gcreduce_data/pretrained_models/language_model
```

If your datasets or HuggingFace models are stored elsewhere, pass the appropriate cache overrides through `--extra-train-args`.

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
  --aggregation-methods bf16,dynamiQ_aee_5bit \
  --num-train-epochs 1 \
  --rails 1 \
  --dry-run
```

Remove `--dry-run` to submit the job.

## Important Parameters

* `--nodes`: comma-separated host list. Use four nodes for the full paper-artifact reproduction. Each node launches two ranks.
* `--aggregation-methods`: comma-separated communication hooks. The default is `bf16,MXfp8,fp4,fp6,zero,dynamiQ_aee_5bit,dynamiQ_mee_5bit,omnireduce,thc`.
* `--rails`: number of RDMA rails. Use `--rails 1` for the reproduced paper-artifact configuration. Use `--rails 2` only on systems with two working RDMA interfaces.
* `--iface0`, `--iface1`: network interface names for one-rail or two-rail RDMA runs.
* `--dynamic-pipeline-rdma`: enables the pipelined dynamic AEE/MEE RDMA path.
* `--pipeline-chunk-mb`: RDMA pipeline tile size.
* `--pipeline-inflight`: number of in-flight tiles per rail.
* `--gpu-pair-starts`: ordered GPU pair starts to try on each node. The selected pair should be connected by NVLink/NV4.
* `--max-used-mb`: treats a GPU as busy if its memory usage is above this threshold.
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

## Scope of This Artifact

This repository contains the testbed artifact for DynamiQ compression-aware distributed training.

* `cuda_kernels/`: CUDA compression and decompression kernels used by the DynamiQ communication hooks.
* `rdma_comm_compress/`: RDMA/NVLink communication microbenchmarks and the staged GPU→CPU→RDMA→CPU→GPU transport used by the hooks.
* `testbed_evaluation/`: end-to-end training evaluations for causal language modeling, MMLU, and Wikitext-103 masked language modeling.

The refactored open-source artifact path is the testbed workflow above. The `simulations_llm/` tree is not required for the current testbed artifact reproduction.
