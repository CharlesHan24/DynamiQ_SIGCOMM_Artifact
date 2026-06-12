#!/usr/bin/env zsh
unsetopt errexit nounset
source ~/.zshrc
setopt errexit nounset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPO_DIR=""
RUN_DIR=""
NODE_INDEX=""
MASTER_ADDR=""
MASTER_PORT=""
BASE_PORT=""
AGGREGATION_METHOD=""
DTYPE="bf16"
STEPS=25
NUMEL=1073741824
BUCKET_CAP_MB=512
CHECKSUM_ATOL="1e-2"
NUM_NODES=4
PROCS_PER_NODE=2
WORLD_SIZE=4
SKIP_GRAD_CHECK=0
EXPECT_BUTTERFLY_RDMA=0
NSYS_PROFILE_STEP=-1
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
RAILS=1
IFACE0="eth0"
IFACE1="eth1"
GID_INDEX=-1
GPU_PAIR_STARTS="4,6,2,0"
MAX_USED_MB=512
CONDA_ENV="llm"
CUDA_HOME_DIR="/share/apps/cuda-11.8"
GCC_HOME="/share/apps/gcc-8.3"
CUDA_ARCH_LIST="8.0;8.6;8.9"
DYNAMIC_AEE_PIPELINE_RDMA="${DYNAMIC_AEE_PIPELINE_RDMA:-0}"
RING_RDMA_PIPELINE_CHUNK_MB="${RING_RDMA_PIPELINE_CHUNK_MB:-8}"
RING_RDMA_PIPELINE_INFLIGHT="${RING_RDMA_PIPELINE_INFLIGHT:-2}"
NUMA_NODE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --run-dir)
      RUN_DIR="$2"; shift 2 ;;
    --node-index)
      NODE_INDEX="$2"; shift 2 ;;
    --master-addr)
      MASTER_ADDR="$2"; shift 2 ;;
    --master-port)
      MASTER_PORT="$2"; shift 2 ;;
    --base-port)
      BASE_PORT="$2"; shift 2 ;;
    --num-nodes)
      NUM_NODES="$2"; shift 2 ;;
    --procs-per-node)
      PROCS_PER_NODE="$2"; shift 2 ;;
    --aggregation-method|--aggregation_method)
      AGGREGATION_METHOD="$2"; shift 2 ;;
    --dtype)
      DTYPE="$2"; shift 2 ;;
    --steps)
      STEPS="$2"; shift 2 ;;
    --numel)
      NUMEL="$2"; shift 2 ;;
    --bucket-cap-mb|--bucket_cap_mb)
      BUCKET_CAP_MB="$2"; shift 2 ;;
    --checksum-atol|--checksum_atol)
      CHECKSUM_ATOL="$2"; shift 2 ;;
    --skip-grad-check|--skip_grad_check)
      SKIP_GRAD_CHECK=1; shift ;;
    --expect-butterfly-rdma|--expect_butterfly_rdma)
      EXPECT_BUTTERFLY_RDMA=1; shift ;;
    --nsys-profile-step|--nsys_profile_step)
      NSYS_PROFILE_STEP="$2"; shift 2 ;;
    --nsys-trace|--nsys_trace)
      NSYS_TRACE="$2"; shift 2 ;;
    --rails)
      RAILS="$2"; shift 2 ;;
    --iface0)
      IFACE0="$2"; shift 2 ;;
    --iface1)
      IFACE1="$2"; shift 2 ;;
    --gid-index)
      GID_INDEX="$2"; shift 2 ;;
    --gpu-pair-starts)
      GPU_PAIR_STARTS="$2"; shift 2 ;;
    --max-used-mb)
      MAX_USED_MB="$2"; shift 2 ;;
    --conda-env)
      CONDA_ENV="$2"; shift 2 ;;
    --cuda-home)
      CUDA_HOME_DIR="$2"; shift 2 ;;
    --gcc-home)
      GCC_HOME="$2"; shift 2 ;;
    --cuda-arch-list)
      CUDA_ARCH_LIST="$2"; shift 2 ;;
    --dynamic-pipeline-rdma)
      DYNAMIC_AEE_PIPELINE_RDMA="$2"; shift 2 ;;
    --pipeline-chunk-mb)
      RING_RDMA_PIPELINE_CHUNK_MB="$2"; shift 2 ;;
    --pipeline-inflight)
      RING_RDMA_PIPELINE_INFLIGHT="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="${SGE_O_WORKDIR:-$SCRIPT_DIR/..}"
fi
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

missing=()
for name in RUN_DIR NODE_INDEX MASTER_ADDR MASTER_PORT BASE_PORT AGGREGATION_METHOD; do
  if [[ -z "${(P)name}" ]]; then
    missing+=("$name")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "Missing required settings: ${missing[*]}" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
WORLD_SIZE=$(( NUM_NODES * PROCS_PER_NODE ))

gpu_exists() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1
}

gpu_memory_used_mb() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

gpu_has_compute_process() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[[:space:]]*[0-9]+'
}

pair_is_available() {
  local first="$1"
  local second="$2"
  local used_first=""
  local used_second=""

  gpu_exists "$first" || return 1
  gpu_exists "$second" || return 1

  used_first="$(gpu_memory_used_mb "$first")"
  used_second="$(gpu_memory_used_mb "$second")"
  [[ -n "$used_first" && -n "$used_second" ]] || return 1
  (( used_first <= MAX_USED_MB )) || return 1
  (( used_second <= MAX_USED_MB )) || return 1

  if gpu_has_compute_process "$first" || gpu_has_compute_process "$second"; then
    return 1
  fi

  return 0
}

select_gpu_pair() {
  local starts
  starts=(${(s:,:)GPU_PAIR_STARTS})
  local first=""
  local second=""

  for first in "${starts[@]}"; do
    first="${first//[[:space:]]/}"
    [[ -n "$first" ]] || continue
    second=$(( first ^ 1 ))
    if pair_is_available "$first" "$second"; then
      echo "$first,$second"
      return 0
    fi
  done

  return 1
}

PAIR="$(select_gpu_pair || true)"
if [[ -z "$PAIR" ]]; then
  {
    echo "No available XOR GPU pair found on $(hostname)."
    echo "Tried pair starts: $GPU_PAIR_STARTS"
    echo "Maximum allowed memory per GPU: ${MAX_USED_MB} MiB"
    nvidia-smi || true
  } > "$RUN_DIR/node_${NODE_INDEX}_gpu_selection_failed.log" 2>&1
  exit 2
fi

IFS=',' read -r GPU0 GPU1 <<< "$PAIR"

conda activate "$CONDA_ENV"

export CUDA_HOME="$CUDA_HOME_DIR"
export PATH="$GCC_HOME/bin:$CUDA_HOME_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$GCC_HOME/lib64:$CUDA_HOME_DIR/lib64:${LD_LIBRARY_PATH:-}"
export CC="$GCC_HOME/bin/gcc"
export CXX="$GCC_HOME/bin/g++"
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST"

export CUDA_VISIBLE_DEVICES="$GPU0,$GPU1"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export WORLD_SIZE="$WORLD_SIZE"

export RING_RDMA_RAILS="$RAILS"
export RING_RDMA_IFACE0="$IFACE0"
export RING_RDMA_IFACE1="$IFACE1"
export RING_RDMA_BASE_PORT="$BASE_PORT"
export RING_RDMA_GID_INDEX="$GID_INDEX"
export RING_VALIDATE_NVLINK="${RING_VALIDATE_NVLINK:-1}"
export DYNAMIC_AEE_PIPELINE_RDMA="$DYNAMIC_AEE_PIPELINE_RDMA"
export RING_RDMA_PIPELINE_CHUNK_MB="$RING_RDMA_PIPELINE_CHUNK_MB"
export RING_RDMA_PIPELINE_INFLIGHT="$RING_RDMA_PIPELINE_INFLIGHT"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IFACE0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"

{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "job_id=${JOB_ID:-unknown}"
  echo "node_index=$NODE_INDEX"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "base_port=$BASE_PORT"
  echo "num_nodes=$NUM_NODES"
  echo "procs_per_node=$PROCS_PER_NODE"
  echo "world_size=$WORLD_SIZE"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "aggregation_method=$AGGREGATION_METHOD"
  echo "dtype=$DTYPE"
  echo "steps=$STEPS"
  echo "numel=$NUMEL"
  echo "bucket_cap_mb=$BUCKET_CAP_MB"
  echo "checksum_atol=$CHECKSUM_ATOL"
  echo "skip_grad_check=$SKIP_GRAD_CHECK"
  echo "expect_butterfly_rdma=$EXPECT_BUTTERFLY_RDMA"
  echo "nsys_profile_step=$NSYS_PROFILE_STEP"
  echo "nsys_trace=$NSYS_TRACE"
  echo "rails=$RAILS"
  echo "iface0=$IFACE0"
  echo "iface1=$IFACE1"
  echo "gid_index=$GID_INDEX"
  echo "conda_env=$CONDA_ENV"
  echo "python=$(command -v python)"
  echo "cuda_home=$CUDA_HOME"
  echo "gcc_home=$GCC_HOME"
  echo "torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST"
  echo "dynamic_aee_pipeline_rdma=$DYNAMIC_AEE_PIPELINE_RDMA"
  echo "ring_rdma_pipeline_chunk_mb=$RING_RDMA_PIPELINE_CHUNK_MB"
  echo "ring_rdma_pipeline_inflight=$RING_RDMA_PIPELINE_INFLIGHT"
  echo "numa_node=$NUMA_NODE"
  nvidia-smi -L || true
} > "$RUN_DIR/node_${NODE_INDEX}_metadata.log" 2>&1

cmd=(
  numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE"
  python -m torch.distributed.run
  --nnodes "$NUM_NODES"
  --nproc_per_node "$PROCS_PER_NODE"
  --node_rank "$NODE_INDEX"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  "$REPO_DIR/testbed_evaluation/smoke_test_new_comm_hooks.py"
  --aggregation_method "$AGGREGATION_METHOD"
  --dtype "$DTYPE"
  --steps "$STEPS"
  --numel "$NUMEL"
  --bucket_cap_mb "$BUCKET_CAP_MB"
  --checksum_atol "$CHECKSUM_ATOL"
  --nclients "$WORLD_SIZE"
  --rdma_max_bucket_elems "$NUMEL"
)

if (( NSYS_PROFILE_STEP >= 0 )); then
  cmd+=(--nsys_profile_step "$NSYS_PROFILE_STEP")
fi

if (( SKIP_GRAD_CHECK )); then
  cmd+=(--skip_grad_check)
fi
if (( EXPECT_BUTTERFLY_RDMA )); then
  cmd+=(--expect_butterfly_rdma)
fi

run_cmd=("${cmd[@]}")
if (( NSYS_PROFILE_STEP >= 0 )); then
  run_cmd=(
    /usr/local/cuda/bin/nsys profile
    --force-overwrite=true
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --trace "$NSYS_TRACE"
    --output "$RUN_DIR/node_${NODE_INDEX}_nsys_step${NSYS_PROFILE_STEP}"
    "${cmd[@]}"
  )
fi

print -r -- "${(q)run_cmd[@]}" > "$RUN_DIR/node_${NODE_INDEX}_torchrun_command.log"
cd "$REPO_DIR"
"${run_cmd[@]}"
