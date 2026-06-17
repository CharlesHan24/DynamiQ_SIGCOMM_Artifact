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
TASK=""
MODEL_NAME_OR_PATH=""
MODEL_LABEL=""
AGGREGATION_METHODS_RAW=""

NUM_NODES=4
PROCS_PER_NODE=2
WORLD_SIZE=4
METHOD_PORT_STRIDE=200
RDZV_TIMEOUT=360000

NORMALIZED=1
NORMALIZED_CHUNK_SIZE=1024
QUANTIZATION_LEVELS=120
NBITS_COMMUNICATION=8
SEED=63
COMPRESSION="None"
RESUME_DIR="None"
ROTATION="True"
FINE_TUNING="False"
TO_PERM="False"
SPARSITY="None"
AGG_CHUNK_SIZE=16
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=""
CAUSAL_BLOCK_SIZE=3000
PER_DEVICE_TRAIN_BATCH_SIZE=""
TO_SHRIMP=""
EXTRA_TRAIN_ARGS=""
DATA_CACHE_DIR="${DYNAMIQ_DATA_CACHE:-}"
MODEL_CACHE_DIR="${DYNAMIQ_MODEL_CACHE:-}"

GPU_PAIR_STARTS="4,2,6,0"
MAX_USED_MB=512
NUMA_NODE=1
CONDA_ENV="llm"
CUDA_HOME_DIR="/share/apps/cuda-11.8"
GCC_HOME="/share/apps/gcc-8.3"
CUDA_ARCH_LIST="8.0;8.6;8.9"

RAILS=1
IFACE0="eth0"
IFACE1="eth1"
GID_INDEX=-1
DYNAMIC_AEE_PIPELINE_RDMA="${DYNAMIC_AEE_PIPELINE_RDMA:-0}"
RING_RDMA_PIPELINE_CHUNK_MB="${RING_RDMA_PIPELINE_CHUNK_MB:-8}"
RING_RDMA_PIPELINE_INFLIGHT="${RING_RDMA_PIPELINE_INFLIGHT:-2}"
RING_LOCAL_P2P_CHUNK_MB="${RING_LOCAL_P2P_CHUNK_MB:-8}"

STOP_ON_FAILURE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --node-index) NODE_INDEX="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --model) MODEL_NAME_OR_PATH="$2"; shift 2 ;;
    --model-label) MODEL_LABEL="$2"; shift 2 ;;
    --aggregation-methods) AGGREGATION_METHODS_RAW="$2"; shift 2 ;;
    --num-nodes) NUM_NODES="$2"; shift 2 ;;
    --procs-per-node) PROCS_PER_NODE="$2"; shift 2 ;;
    --method-port-stride) METHOD_PORT_STRIDE="$2"; shift 2 ;;
    --rdzv-timeout) RDZV_TIMEOUT="$2"; shift 2 ;;
    --normalized) NORMALIZED="$2"; shift 2 ;;
    --normalized-chunk-size|--normalized_chunk_size) NORMALIZED_CHUNK_SIZE="$2"; shift 2 ;;
    --quantization-levels|--quantization_levels) QUANTIZATION_LEVELS="$2"; shift 2 ;;
    --nbits-communication|--nbits_communication) NBITS_COMMUNICATION="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --compression) COMPRESSION="$2"; shift 2 ;;
    --resume-dir|--resume_dir) RESUME_DIR="$2"; shift 2 ;;
    --rotation) ROTATION="$2"; shift 2 ;;
    --fine-tuning|--fine_tuning) FINE_TUNING="$2"; shift 2 ;;
    --to-perm|--to_perm) TO_PERM="$2"; shift 2 ;;
    --sparsity) SPARSITY="$2"; shift 2 ;;
    --agg-chunk-size|--agg_chunk_size) AGG_CHUNK_SIZE="$2"; shift 2 ;;
    --num-train-epochs|--num_train_epochs) NUM_TRAIN_EPOCHS="$2"; shift 2 ;;
    --learning-rate|--learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --causal-block-size|--block-size|--block_size) CAUSAL_BLOCK_SIZE="$2"; shift 2 ;;
    --per-device-train-batch-size|--per_device_train_batch_size) PER_DEVICE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --to-shrimp|--to_shrimp) TO_SHRIMP="$2"; shift 2 ;;
    --data-cache-dir|--data_cache_dir) DATA_CACHE_DIR="$2"; shift 2 ;;
    --model-cache-dir|--model_cache_dir) MODEL_CACHE_DIR="$2"; shift 2 ;;
    --extra-train-args) EXTRA_TRAIN_ARGS="$2"; shift 2 ;;
    --gpu-pair-starts) GPU_PAIR_STARTS="$2"; shift 2 ;;
    --max-used-mb) MAX_USED_MB="$2"; shift 2 ;;
    --numa-node) NUMA_NODE="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --cuda-home) CUDA_HOME_DIR="$2"; shift 2 ;;
    --gcc-home) GCC_HOME="$2"; shift 2 ;;
    --cuda-arch-list) CUDA_ARCH_LIST="$2"; shift 2 ;;
    --rails) RAILS="$2"; shift 2 ;;
    --iface0) IFACE0="$2"; shift 2 ;;
    --iface1) IFACE1="$2"; shift 2 ;;
    --gid-index) GID_INDEX="$2"; shift 2 ;;
    --dynamic-pipeline-rdma) DYNAMIC_AEE_PIPELINE_RDMA="$2"; shift 2 ;;
    --pipeline-chunk-mb) RING_RDMA_PIPELINE_CHUNK_MB="$2"; shift 2 ;;
    --pipeline-inflight) RING_RDMA_PIPELINE_INFLIGHT="$2"; shift 2 ;;
    --local-p2p-chunk-mb) RING_LOCAL_P2P_CHUNK_MB="$2"; shift 2 ;;
    --continue-on-failure) STOP_ON_FAILURE=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="${SGE_O_WORKDIR:-$SCRIPT_DIR/..}"
fi
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

missing=()
for name in RUN_DIR NODE_INDEX MASTER_ADDR MASTER_PORT BASE_PORT TASK MODEL_NAME_OR_PATH AGGREGATION_METHODS_RAW; do
  if [[ -z "${(P)name}" ]]; then
    missing+=("$name")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "Missing required settings: ${missing[*]}" >&2
  exit 1
fi

WORLD_SIZE=$(( NUM_NODES * PROCS_PER_NODE ))
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

calc_learning_rate() {
  local task="$1"
  local world_size="$2"
  local model="$3"
  local base="1e-5"
  [[ "$task" == "mmlu" ]] && base="3e-6"
  if [[ "$task" == "wikitext" || "$task" == "maskedlm" || "$task" == "mlm" ]]; then
    printf "5e-5"
    return
  fi
  awk -v base="$base" -v n="$world_size" -v task="$task" -v model="$model" '
    BEGIN {
      lr = base * sqrt(n / 2.0) * exp(log(8.0 / n) * 0.4)
      if (task == "causal" && index(tolower(model), "gemma") > 0) {
        lr *= 0.7
      }
      printf "%.12g", lr
    }
  '
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
export WORLD_SIZE="$WORLD_SIZE"

export RING_RDMA_RAILS="$RAILS"
export RING_RDMA_IFACE0="$IFACE0"
export RING_RDMA_IFACE1="$IFACE1"
export RING_RDMA_GID_INDEX="$GID_INDEX"
export RING_VALIDATE_NVLINK="${RING_VALIDATE_NVLINK:-1}"
export DYNAMIC_AEE_PIPELINE_RDMA="$DYNAMIC_AEE_PIPELINE_RDMA"
export RING_RDMA_PIPELINE_CHUNK_MB="$RING_RDMA_PIPELINE_CHUNK_MB"
export RING_RDMA_PIPELINE_INFLIGHT="$RING_RDMA_PIPELINE_INFLIGHT"
export RING_LOCAL_P2P_CHUNK_MB="$RING_LOCAL_P2P_CHUNK_MB"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IFACE0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"

if [[ -z "$PER_DEVICE_TRAIN_BATCH_SIZE" ]]; then
  PER_DEVICE_TRAIN_BATCH_SIZE=1
  [[ "$TASK" == "mmlu" ]] && PER_DEVICE_TRAIN_BATCH_SIZE=4
  [[ "$TASK" == "wikitext" || "$TASK" == "maskedlm" || "$TASK" == "mlm" ]] && PER_DEVICE_TRAIN_BATCH_SIZE=4
fi
if [[ -z "$TO_SHRIMP" ]]; then
  TO_SHRIMP="False"
  [[ "$TASK" == "mmlu" ]] && TO_SHRIMP="True"
fi

case "$TASK" in
  causal)
    TRAIN_SCRIPT="$REPO_DIR/testbed_evaluation/train_llm_causal.py"
    ;;
  mmlu)
    TRAIN_SCRIPT="$REPO_DIR/testbed_evaluation/train_llm_mmlu.py"
    ;;
  wikitext|maskedlm|mlm)
    TRAIN_SCRIPT="$REPO_DIR/testbed_evaluation/train_llm_maskedlm_bert.py"
    ;;
  *)
    echo "Unsupported task '$TASK'. Use causal, mmlu, or wikitext/maskedlm." >&2
    exit 1
    ;;
esac

METHODS=(${(s:,:)AGGREGATION_METHODS_RAW})
LR_VALUE="$LEARNING_RATE"
if [[ -z "$LR_VALUE" ]]; then
  LR_VALUE="$(calc_learning_rate "$TASK" "$WORLD_SIZE" "$MODEL_NAME_OR_PATH")"
fi

{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "job_id=${JOB_ID:-unknown}"
  echo "node_index=$NODE_INDEX"
  echo "task=$TASK"
  echo "model=$MODEL_NAME_OR_PATH"
  echo "model_label=$MODEL_LABEL"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "base_port=$BASE_PORT"
  echo "world_size=$WORLD_SIZE"
  echo "num_nodes=$NUM_NODES"
  echo "procs_per_node=$PROCS_PER_NODE"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "gpu_pair_start_order=$GPU_PAIR_STARTS"
  echo "numa_node=$NUMA_NODE"
  echo "methods=${METHODS[*]}"
  echo "learning_rate=$LR_VALUE"
  echo "per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE"
  echo "data_cache_dir=${DATA_CACHE_DIR:-hf_default}"
  echo "model_cache_dir=${MODEL_CACHE_DIR:-hf_default}"
  echo "num_train_epochs=$NUM_TRAIN_EPOCHS"
  echo "conda_env=$CONDA_ENV"
  echo "python=$(command -v python)"
  echo "cuda_home=$CUDA_HOME"
  echo "gcc_home=$GCC_HOME"
  echo "torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST"
  echo "rails=$RAILS"
  echo "iface0=$IFACE0"
  echo "iface1=$IFACE1"
  echo "gid_index=$GID_INDEX"
  echo "dynamic_aee_pipeline_rdma=$DYNAMIC_AEE_PIPELINE_RDMA"
  echo "ring_rdma_pipeline_chunk_mb=$RING_RDMA_PIPELINE_CHUNK_MB"
  echo "ring_rdma_pipeline_inflight=$RING_RDMA_PIPELINE_INFLIGHT"
  echo "ring_local_p2p_chunk_mb=$RING_LOCAL_P2P_CHUNK_MB"
  nvidia-smi -L || true
  nvidia-smi topo -m || true
} > "$RUN_DIR/node_${NODE_INDEX}_metadata.log" 2>&1

overall_status=0
method_index=0
for method in "${METHODS[@]}"; do
  method="${method//[[:space:]]/}"
  [[ -n "$method" ]] || continue

  safe_method="${method//[^A-Za-z0-9_.-]/_}"
  METHOD_RUN_DIR="$RUN_DIR/method_${method_index}_${safe_method}"
  mkdir -p "$METHOD_RUN_DIR"

  method_master_port=$(( MASTER_PORT + method_index * METHOD_PORT_STRIDE ))
  method_base_port=$(( BASE_PORT + method_index * METHOD_PORT_STRIDE ))
  export MASTER_PORT="$method_master_port"
  export RING_RDMA_BASE_PORT="$method_base_port"

  cmd=(
    accelerate launch
    --multi_gpu
    --main_process_ip "$MASTER_ADDR"
    --main_process_port "$method_master_port"
    --num_processes "$WORLD_SIZE"
    --num_machines "$NUM_NODES"
    --machine_rank "$NODE_INDEX"
    --rdzv_conf "timeout=$RDZV_TIMEOUT"
    "$TRAIN_SCRIPT"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --quantization_levels "$QUANTIZATION_LEVELS"
    --nclients "$WORLD_SIZE"
    --normalized "$NORMALIZED"
    --normalized_chunk_size "$NORMALIZED_CHUNK_SIZE"
    --aggregation_method "$method"
    --seed "$SEED"
    --learning_rate "$LR_VALUE"
    --compression "$COMPRESSION"
    --nbits_communication "$NBITS_COMMUNICATION"
    --resume_dir "$RESUME_DIR"
    --rotation "$ROTATION"
    --fine_tuning "$FINE_TUNING"
    --to_perm "$TO_PERM"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --sparsity "$SPARSITY"
    --agg_chunk_size "$AGG_CHUNK_SIZE"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --to_shrimp "$TO_SHRIMP"
    --output_dir "$METHOD_RUN_DIR/output"
  )

  if [[ "$TASK" == "causal" ]]; then
    cmd+=(--block_size "$CAUSAL_BLOCK_SIZE")
  fi
  if [[ -n "$DATA_CACHE_DIR" ]]; then
    cmd+=(--data_cache_dir "$DATA_CACHE_DIR")
  fi
  if [[ -n "$MODEL_CACHE_DIR" ]]; then
    cmd+=(--model_cache_dir "$MODEL_CACHE_DIR")
  fi
  if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
    cmd+=(${=EXTRA_TRAIN_ARGS})
  fi

  run_cmd=("${cmd[@]}")
  if [[ "$NUMA_NODE" != "none" ]]; then
    run_cmd=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" "${cmd[@]}")
  fi

  echo $run_cmd

  {
    echo "date=$(date -Is)"
    echo "method_index=$method_index"
    echo "aggregation_method=$method"
    echo "master_port=$method_master_port"
    echo "ring_rdma_base_port=$method_base_port"
    echo "command=${(q)run_cmd[@]}"
  } > "$METHOD_RUN_DIR/command.log"

  echo "[$(date -Is)] node=$NODE_INDEX task=$TASK model=$MODEL_LABEL method=$method start"
  set +e
  (cd "$REPO_DIR" && "${run_cmd[@]}") > "$METHOD_RUN_DIR/node_${NODE_INDEX}.out" 2> "$METHOD_RUN_DIR/node_${NODE_INDEX}.err"
  method_status=$?
  set -e
  echo "[$(date -Is)] node=$NODE_INDEX task=$TASK model=$MODEL_LABEL method=$method exit=$method_status"
  echo "$method_status" > "$METHOD_RUN_DIR/node_${NODE_INDEX}.status"

  if (( method_status != 0 )); then
    overall_status=$method_status
    if (( STOP_ON_FAILURE )); then
      exit "$method_status"
    fi
  fi

  method_index=$(( method_index + 1 ))
done

exit "$overall_status"
