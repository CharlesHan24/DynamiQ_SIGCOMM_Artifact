#!/usr/bin/env zsh
unsetopt errexit nounset
source ~/.zshrc
setopt errexit nounset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPO_DIR=""
RUN_DIR=""
NODE_INDEX=""
RANK_BASE=""
WORLD_SIZE=""
MASTER_ADDR=""
MASTER_PORT=""
BASE_PORT=""
NUMEL=""
ITERS=""
WARMUP_ITERS=""
NBITS="4"
MODE="bf16"
RAILS="1"
IFACE0=""
IFACE1=""
NSYS_ENABLED=0
NSYS_RANKS="1"
NSYS_TRACE="cuda,nvtx,osrt"
NSYS_BIN="/usr/local/cuda/bin/nsys"
GPU_PAIR_STARTS="6,4,2,0"
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
    --rank-base)
      RANK_BASE="$2"
      shift 2
      ;;
    --world-size)
      WORLD_SIZE="$2"
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
    --nbits)
      NBITS="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --nsys)
      NSYS_ENABLED=1
      shift
      ;;
    --nsys-ranks)
      NSYS_RANKS="$2"
      shift 2
      ;;
    --nsys-trace)
      NSYS_TRACE="$2"
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
    --gpu-pair-starts)
      GPU_PAIR_STARTS="$2"
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
for name in RUN_DIR NODE_INDEX RANK_BASE WORLD_SIZE MASTER_ADDR MASTER_PORT BASE_PORT NUMEL ITERS WARMUP_ITERS RAILS IFACE0; do
  if [[ -z "${(P)name}" ]]; then
    missing+=("$name")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Missing required settings: ${missing[*]}" >&2
  exit 1
fi

if [[ "$RAILS" != "1" && "$RAILS" != "2" ]]; then
  echo "ring all-reduce currently supports only --rails 1 or --rails 2" >&2
  exit 1
fi

if [[ "$RAILS" == "2" && -z "$IFACE1" ]]; then
  echo "--iface1 is required with --rails 2" >&2
  exit 1
fi

if [[ "$MODE" != "bf16" && "$MODE" != "quantized" ]]; then
  echo "ring all-reduce currently supports only --mode bf16 or --mode quantized" >&2
  exit 1
fi

if (( NSYS_ENABLED )) && [[ ! -x "$NSYS_BIN" ]]; then
  echo "Nsight Systems binary not found at $NSYS_BIN" >&2
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
    echo "No available GPU pair found on $(hostname)."
    echo "Tried pair starts: $GPU_PAIR_STARTS"
    echo "Maximum allowed memory per GPU: ${MAX_USED_MB} MiB"
    nvidia-smi || true
  } > "$RUN_DIR/node_${NODE_INDEX}_gpu_selection_failed.log" 2>&1
  exit 2
fi

IFS=',' read -r GPU0 GPU1 <<< "$PAIR"

export CUDA_VISIBLE_DEVICES="$GPU0,$GPU1"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export WORLD_SIZE="$WORLD_SIZE"

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
  echo "task_id=${SGE_TASK_ID:-none}"
  echo "node_index=$NODE_INDEX"
  echo "rank_base=$RANK_BASE"
  echo "world_size=$WORLD_SIZE"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "base_port=$BASE_PORT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "gpu_pair_start_order=$GPU_PAIR_STARTS"
  echo "max_used_mb=$MAX_USED_MB"
  echo "numa_node=$NUMA_NODE"
  echo "repo_dir=$REPO_DIR"
  echo "script_dir=$SCRIPT_DIR"
  echo "sge_o_workdir=${SGE_O_WORKDIR:-}"
  echo "mode=$MODE"
  echo "nbits=$NBITS"
  echo "rails=$RAILS"
  echo "nccl_p2p_level=$NCCL_P2P_LEVEL"
  echo "nsys_enabled=$NSYS_ENABLED"
  echo "nsys_ranks=$NSYS_RANKS"
  echo "nsys_trace=$NSYS_TRACE"
  echo "nsys_bin=$NSYS_BIN"
  echo "iface0=$IFACE0"
  echo "iface1=$IFACE1"
  nvidia-smi -L || true
} > "$RUN_DIR/node_${NODE_INDEX}_metadata.log" 2>&1

rank_should_profile() {
  local global_rank="$1"
  local entry=""
  local entries

  if (( ! NSYS_ENABLED )); then
    return 1
  fi
  if [[ "$NSYS_RANKS" == "all" ]]; then
    return 0
  fi

  entries=(${(s:,:)NSYS_RANKS})
  for entry in "${entries[@]}"; do
    entry="${entry//[[:space:]]/}"
    [[ -n "$entry" ]] || continue
    if [[ "$entry" == "$global_rank" ]]; then
      return 0
    fi
  done

  return 1
}

rank_cmd() {
  local local_rank="$1"
  local global_rank="$2"
  local cmd
  local profile_dir="$RUN_DIR/nsys_reports"
  cmd=(
    python "$REPO_DIR/sample_allreduce.py"
    --numel "$NUMEL"
    --iters "$ITERS"
    --warmup-iters "$WARMUP_ITERS"
    --mode "$MODE"
    --nbits "$NBITS"
    --rails "$RAILS"
    --expected-world-size "$WORLD_SIZE"
    --iface0 "$IFACE0"
    --base-port "$BASE_PORT"
    --gid-index "$GID_INDEX"
  )

  if [[ "$RAILS" == "2" ]]; then
    cmd+=(--iface1 "$IFACE1")
  fi
  if [[ -n "$VERIFY_FLAG" ]]; then
    cmd+=("$VERIFY_FLAG")
  fi
  if rank_should_profile "$global_rank"; then
    mkdir -p "$profile_dir"
    cmd=(
      "$NSYS_BIN"
      profile
      --force-overwrite=true
      --stats=false
      --sample=none
      --cpuctxsw=none
      --show-output=true
      --trace="$NSYS_TRACE"
      --capture-range=cudaProfilerApi
      --capture-range-end=stop
      -o "$profile_dir/rank_${global_rank}"
        "${cmd[@]}"
        --enable-nvtx
        --cuda-profiler-range
        )
  fi

  export RANK="$global_rank"
  export LOCAL_RANK="$local_rank"

  if [[ "$NUMA_NODE" != "none" && -x "$(command -v numactl)" ]]; then
    numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

pids=()
rank_ids=()
statuses=()
exit_codes=()

for local_rank in 0 1; do
  global_rank=$(( RANK_BASE + local_rank ))
  (
    unsetopt errexit
    rank_cmd "$local_rank" "$global_rank"
    rc=$?
    print -r -- "$rc" > "$RUN_DIR/rank_${global_rank}.exitcode"
    exit "$rc"
  ) > "$RUN_DIR/rank_${global_rank}.out" 2> "$RUN_DIR/rank_${global_rank}.err" &
  pids+=("$!")
  rank_ids+=("$global_rank")
done

idx=1
for pid in "${pids[@]}"; do
  global_rank="${rank_ids[$idx]}"
  if wait "$pid"; then
    rc=0
  else
    rc=$?
  fi
  if [[ -s "$RUN_DIR/rank_${global_rank}.exitcode" ]]; then
    rc="$(< "$RUN_DIR/rank_${global_rank}.exitcode")"
  fi
  statuses+=("${global_rank}:${rc}")
  exit_codes+=("$rc")
  idx=$(( idx + 1 ))
done

{
  echo "date=$(date -Is)"
  echo "statuses=${statuses[*]}"
} > "$RUN_DIR/node_${NODE_INDEX}_status.log"

for exit_status in "${exit_codes[@]}"; do
  if (( exit_status != 0 )); then
    exit "$exit_status"
  fi
done

exit 0
