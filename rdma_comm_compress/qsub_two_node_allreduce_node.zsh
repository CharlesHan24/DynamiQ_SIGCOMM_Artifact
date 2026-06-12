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
NUMEL=""
ITERS=""
WARMUP_ITERS=""
MODE="bf16"
NBITS=4
RAILS=2
IFACE0="eth0"
IFACE1=""
GPU_IDS="7,6,5,4,3,2,1,0"
MAX_USED_MB=512
NUMA_NODE=1
VERIFY_FLAG=""
GID_INDEX=-1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --node-index)
      NODE_INDEX="$2"
      shift 2
      ;;
    --master-addr)
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --numel)
      NUMEL="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --warmup-iters)
      WARMUP_ITERS="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --nbits)
      NBITS="$2"
      shift 2
      ;;
    --rails)
      RAILS="$2"
      shift 2
      ;;
    --iface0)
      IFACE0="$2"
      shift 2
      ;;
    --iface1)
      IFACE1="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --max-used-mb)
      MAX_USED_MB="$2"
      shift 2
      ;;
    --numa-node)
      NUMA_NODE="$2"
      shift 2
      ;;
    --gid-index)
      GID_INDEX="$2"
      shift 2
      ;;
    --verify)
      VERIFY_FLAG="--verify"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="${SGE_O_WORKDIR:-$SCRIPT_DIR}"
fi
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

missing=()
for name in RUN_DIR NODE_INDEX MASTER_ADDR MASTER_PORT BASE_PORT NUMEL ITERS WARMUP_ITERS RAILS IFACE0; do
  if [[ -z "${(P)name}" ]]; then
    missing+=("$name")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "Missing required settings: ${missing[*]}" >&2
  exit 1
fi

if [[ "$RAILS" == "2" && -z "$IFACE1" ]]; then
  echo "--iface1 is required when --rails 2" >&2
  exit 1
fi

if [[ "$MODE" != "quantized" && "$MODE" != "bf16" ]]; then
  echo "--mode must be quantized or bf16" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

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

gpu_is_available() {
  local gpu="$1"
  local used=""

  gpu_exists "$gpu" || return 1
  used="$(gpu_memory_used_mb "$gpu")"
  [[ -n "$used" ]] || return 1
  (( used <= MAX_USED_MB )) || return 1

  if gpu_has_compute_process "$gpu"; then
    return 1
  fi
  return 0
}

select_gpu() {
  local entries
  local gpu=""

  entries=(${(s:,:)GPU_IDS})
  for gpu in "${entries[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -n "$gpu" ]] || continue
    if gpu_is_available "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done

  return 1
}

GPU_ID="$(select_gpu || true)"
if [[ -z "$GPU_ID" ]]; then
  {
    echo "No available GPU found on $(hostname)."
    echo "Tried GPU preference order: $GPU_IDS"
    echo "Maximum allowed memory per GPU: ${MAX_USED_MB} MiB"
    nvidia-smi || true
  } > "$RUN_DIR/node_${NODE_INDEX}_gpu_selection_failed.log" 2>&1
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MASTER_ADDR="$MASTER_ADDR"
export WORLD_SIZE=2
export RANK="$NODE_INDEX"
export LOCAL_RANK=0

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IFACE0}"

if [[ "$RAILS" == "2" ]]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE0,$IFACE1}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"
else
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE0}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
fi

{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "job_id=${JOB_ID:-unknown}"
  echo "task_id=${SGE_TASK_ID:-none}"
  echo "node_index=$NODE_INDEX"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "base_port=$BASE_PORT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "selected_gpu=$GPU_ID"
  echo "gpu_id_order=$GPU_IDS"
  echo "max_used_mb=$MAX_USED_MB"
  echo "numa_node=$NUMA_NODE"
  echo "repo_dir=$REPO_DIR"
  echo "script_dir=$SCRIPT_DIR"
  echo "sge_o_workdir=${SGE_O_WORKDIR:-}"
  echo "mode=$MODE"
  echo "nbits=$NBITS"
  echo "rails=$RAILS"
  echo "iface0=$IFACE0"
  echo "iface1=$IFACE1"
  echo "gid_index=$GID_INDEX"
  echo "ring_rdma_debug=${RING_RDMA_DEBUG:-}"
  nvidia-smi -L || true
} > "$RUN_DIR/node_${NODE_INDEX}_metadata.log" 2>&1

run_bench() {
  local label="$1"
  local master_port_value="$2"
  shift 2
  local -a cmd
  cmd=("$@")

  export MASTER_PORT="$master_port_value"

  if [[ "$NUMA_NODE" != "none" && -x "$(command -v numactl)" ]]; then
    numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" "${cmd[@]}" \
      > "$RUN_DIR/${label}_rank_${NODE_INDEX}.out" \
      2> "$RUN_DIR/${label}_rank_${NODE_INDEX}.err"
  else
    "${cmd[@]}" \
      > "$RUN_DIR/${label}_rank_${NODE_INDEX}.out" \
      2> "$RUN_DIR/${label}_rank_${NODE_INDEX}.err"
  fi
}

nccl_cmd=(
  python "$REPO_DIR/single_node_nccl_allreduce.py"
  --numel "$NUMEL"
  --iters "$ITERS"
  --warmup-iters "$WARMUP_ITERS"
  --dtype bf16
)
if [[ -n "$VERIFY_FLAG" ]]; then
  nccl_cmd+=("$VERIFY_FLAG")
fi

ring_cmd=(
  python "$REPO_DIR/sample_allreduce.py"
  --expected-world-size 2
  --numel "$NUMEL"
  --iters "$ITERS"
  --warmup-iters "$WARMUP_ITERS"
  --mode "$MODE"
  --nbits "$NBITS"
  --rails "$RAILS"
  --iface0 "$IFACE0"
  --base-port "$BASE_PORT"
  --gid-index "$GID_INDEX"
)
if [[ "$RAILS" == "2" ]]; then
  ring_cmd+=(--iface1 "$IFACE1")
fi
if [[ -n "$VERIFY_FLAG" ]]; then
  ring_cmd+=("$VERIFY_FLAG")
fi

if run_bench nccl "$MASTER_PORT" "${nccl_cmd[@]}"; then
  nccl_status=0
else
  nccl_status=$?
fi

if run_bench ring "$(( MASTER_PORT + 1 ))" "${ring_cmd[@]}"; then
  ring_status=0
else
  ring_status=$?
fi

{
  echo "date=$(date -Is)"
  echo "nccl_status=$nccl_status"
  echo "ring_status=$ring_status"
} > "$RUN_DIR/node_${NODE_INDEX}_status.log"

if (( nccl_status != 0 )); then
  exit "$nccl_status"
fi
exit "$ring_status"
