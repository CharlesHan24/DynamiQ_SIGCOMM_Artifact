#!/usr/bin/env zsh
unsetopt errexit nounset
[[ -f ~/.zshrc ]] && source ~/.zshrc
setopt errexit nounset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPO_DIR=""
RUN_DIR=""
NODE_INDEX=""
MASTER_ADDR=""
MASTER_PORT=""
TASK=""
MODEL_NAME_OR_PATH=""
MODEL_LABEL=""
AGGREGATION_METHODS_RAW=""

WORKERS=""
NUM_NODES=""
PROCS_PER_NODE=1
METHOD_PORT_STRIDE=200
RDZV_TIMEOUT=360000

NORMALIZED=1
NORMALIZED_CHUNK_SIZE=1024
QUANTIZATION_LEVELS=120
NBITS_COMMUNICATION=8
SEED=63
RESUME_DIR="None"
ROTATION="True"
FINE_TUNING="False"
TO_PERM="False"
SPARSITY="None"
AGG_CHUNK_SIZE=16
CAUSAL_BLOCK_SIZE=3000
PER_DEVICE_TRAIN_BATCH_SIZE=""
LEARNING_RATE=""
EXTRA_TRAIN_ARGS=""
DATA_CACHE_DIR="${DYNAMIQ_DATA_CACHE:-}"
MODEL_CACHE_DIR="${DYNAMIQ_MODEL_CACHE:-}"

GPU_MAX_USED_MB=512
NUMA_NODE="none"
CONDA_ENV="llm"
STOP_ON_FAILURE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --node-index) NODE_INDEX="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --model) MODEL_NAME_OR_PATH="$2"; shift 2 ;;
    --model-label) MODEL_LABEL="$2"; shift 2 ;;
    --aggregation-methods) AGGREGATION_METHODS_RAW="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --num-nodes) NUM_NODES="$2"; shift 2 ;;
    --procs-per-node) PROCS_PER_NODE="$2"; shift 2 ;;
    --method-port-stride) METHOD_PORT_STRIDE="$2"; shift 2 ;;
    --rdzv-timeout|--rdzv_timeout) RDZV_TIMEOUT="$2"; shift 2 ;;
    --normalized) NORMALIZED="$2"; shift 2 ;;
    --normalized-chunk-size|--normalized_chunk_size) NORMALIZED_CHUNK_SIZE="$2"; shift 2 ;;
    --quantization-levels|--quantization_levels) QUANTIZATION_LEVELS="$2"; shift 2 ;;
    --nbits-communication|--nbits_communication) NBITS_COMMUNICATION="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --resume-dir|--resume_dir) RESUME_DIR="$2"; shift 2 ;;
    --rotation) ROTATION="$2"; shift 2 ;;
    --fine-tuning|--fine_tuning) FINE_TUNING="$2"; shift 2 ;;
    --to-perm|--to_perm) TO_PERM="$2"; shift 2 ;;
    --sparsity) SPARSITY="$2"; shift 2 ;;
    --agg-chunk-size|--agg_chunk_size) AGG_CHUNK_SIZE="$2"; shift 2 ;;
    --causal-block-size|--block-size|--block_size) CAUSAL_BLOCK_SIZE="$2"; shift 2 ;;
    --per-device-train-batch-size|--per_device_train_batch_size) PER_DEVICE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate|--learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --data-cache-dir|--data_cache_dir) DATA_CACHE_DIR="$2"; shift 2 ;;
    --model-cache-dir|--model_cache_dir) MODEL_CACHE_DIR="$2"; shift 2 ;;
    --extra-train-args) EXTRA_TRAIN_ARGS="$2"; shift 2 ;;
    --gpu-max-used-mb|--max-used-mb) GPU_MAX_USED_MB="$2"; shift 2 ;;
    --numa-node) NUMA_NODE="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --continue-on-failure) STOP_ON_FAILURE=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="${SGE_O_WORKDIR:-$SCRIPT_DIR/..}"
fi
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

missing=()
for name in RUN_DIR NODE_INDEX MASTER_ADDR MASTER_PORT TASK MODEL_NAME_OR_PATH AGGREGATION_METHODS_RAW WORKERS NUM_NODES; do
  if [[ -z "${(P)name}" ]]; then
    missing+=("$name")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "Missing required settings: ${missing[*]}" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

gpu_memory_used_mb() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

gpu_has_compute_process() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[[:space:]]*[0-9]+'
}

select_gpus() {
  local selected=()
  local gpu used
  local all_gpus
  all_gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null))
  for gpu in "${all_gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -n "$gpu" ]] || continue
    used="$(gpu_memory_used_mb "$gpu")"
    [[ -n "$used" ]] || continue
    (( used <= GPU_MAX_USED_MB )) || continue
    gpu_has_compute_process "$gpu" && continue
    selected+=("$gpu")
    (( ${#selected[@]} >= PROCS_PER_NODE )) && break
  done
  (( ${#selected[@]} == PROCS_PER_NODE )) || return 1
  print -r -- "${(j:,:)selected}"
}

calc_learning_rate() {
  local task="$1"
  local workers="$2"
  local model="$3"
  local base="1e-5"
  [[ "$task" == "mmlu" ]] && base="3e-6"
  if [[ "$task" == "wikitext" || "$task" == "maskedlm" || "$task" == "mlm" ]]; then
    printf "5e-5"
    return
  fi
  awk -v base="$base" -v n="$workers" -v task="$task" -v model="$model" '
    BEGIN {
      lr = base * sqrt(n / 2.0) * exp(log(8.0 / n) * 0.4)
      if (task == "causal" && index(tolower(model), "gemma") > 0) {
        lr *= 0.7
      }
      printf "%.12g", lr
    }
  '
}

calc_decay_epochs() {
  local task="$1"
  local workers="$2"
  if [[ "$task" == "wikitext" || "$task" == "maskedlm" || "$task" == "mlm" ]]; then
    (( workers <= 4 )) && print 8 || print 15
    return
  fi
  (( workers <= 4 )) && print 1 || print 2
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  selected_gpus="$(select_gpus || true)"
  if [[ -z "$selected_gpus" ]]; then
    {
      echo "No available GPU set of size $PROCS_PER_NODE found on $(hostname)."
      echo "Maximum allowed memory per GPU: ${GPU_MAX_USED_MB} MiB"
      nvidia-smi || true
    } > "$RUN_DIR/node_${NODE_INDEX}_gpu_selection_failed.log" 2>&1
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$selected_gpus"
fi

if [[ -n "$CONDA_ENV" ]]; then
  conda activate "$CONDA_ENV"
fi

export MASTER_ADDR="$MASTER_ADDR"
export WORLD_SIZE="$WORKERS"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "$PER_DEVICE_TRAIN_BATCH_SIZE" ]]; then
  PER_DEVICE_TRAIN_BATCH_SIZE=1
  [[ "$TASK" == "mmlu" ]] && PER_DEVICE_TRAIN_BATCH_SIZE=4
  [[ "$TASK" == "wikitext" || "$TASK" == "maskedlm" || "$TASK" == "mlm" ]] && PER_DEVICE_TRAIN_BATCH_SIZE=4
fi

LR_VALUE="$LEARNING_RATE"
[[ -n "$LR_VALUE" ]] || LR_VALUE="$(calc_learning_rate "$TASK" "$WORKERS" "$MODEL_NAME_OR_PATH")"
DECAY_EPOCHS="$(calc_decay_epochs "$TASK" "$WORKERS")"

case "$TASK" in
  causal)
    TRAIN_SCRIPT="$REPO_DIR/simulations_llm/distributed_llm_clm.py"
    ;;
  mmlu)
    TRAIN_SCRIPT="$REPO_DIR/simulations_llm/distributed_llm_mmlu.py"
    ;;
  wikitext|maskedlm|mlm)
    TRAIN_SCRIPT="$REPO_DIR/simulations_llm/distributed_maskedlm_bert.py"
    ;;
  *)
    echo "Unsupported task '$TASK'. Use causal, mmlu, or wikitext/maskedlm." >&2
    exit 1
    ;;
esac

methods=()
for raw_method in ${(s:,:)AGGREGATION_METHODS_RAW}; do
  method="${raw_method//[[:space:]]/}"
  [[ -n "$method" ]] && methods+=("$method")
done

{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "job_id=${JOB_ID:-unknown}"
  echo "node_index=$NODE_INDEX"
  echo "workers=$WORKERS"
  echo "num_nodes=$NUM_NODES"
  echo "procs_per_node=$PROCS_PER_NODE"
  echo "task=$TASK"
  echo "model=$MODEL_NAME_OR_PATH"
  echo "model_label=$MODEL_LABEL"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "methods=${methods[*]}"
  echo "learning_rate=$LR_VALUE"
  echo "decay_epochs=$DECAY_EPOCHS"
  echo "per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE"
  echo "data_cache_dir=${DATA_CACHE_DIR:-hf_default}"
  echo "model_cache_dir=${MODEL_CACHE_DIR:-hf_default}"
  echo "conda_env=$CONDA_ENV"
  echo "python=$(command -v python)"
  nvidia-smi -L || true
} > "$RUN_DIR/node_${NODE_INDEX}_metadata.log" 2>&1

overall_status=0
method_index=0
for method in "${methods[@]}"; do
  safe_method="${method//[^A-Za-z0-9_.-]/_}"
  method_run_dir="$RUN_DIR/method_${method_index}_${safe_method}"
  mkdir -p "$method_run_dir"

  method_master_port=$(( MASTER_PORT + method_index * METHOD_PORT_STRIDE ))
  cmd=(
    accelerate launch
    --multi_gpu
    --main_process_ip "$MASTER_ADDR"
    --main_process_port "$method_master_port"
    --num_processes "$WORKERS"
    --num_machines "$NUM_NODES"
    --machine_rank "$NODE_INDEX"
    --rdzv_conf "timeout=$RDZV_TIMEOUT"
    "$TRAIN_SCRIPT"
    --num_train_epochs "$DECAY_EPOCHS"
    --quantization_levels "$QUANTIZATION_LEVELS"
    --nclients "$WORKERS"
    --normalized "$NORMALIZED"
    --normalized_chunk_size "$NORMALIZED_CHUNK_SIZE"
    --aggregation_method "$method"
    --seed "$SEED"
    --learning_rate "$LR_VALUE"
    --nbits_communication "$NBITS_COMMUNICATION"
    --resume_dir "$RESUME_DIR"
    --rotation "$ROTATION"
    --fine_tuning "$FINE_TUNING"
    --to_perm "$TO_PERM"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --sparsity "$SPARSITY"
    --agg_chunk_size "$AGG_CHUNK_SIZE"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --output_dir "$method_run_dir/output"
  )

  [[ "$TASK" == "wikitext" || "$TASK" == "maskedlm" || "$TASK" == "mlm" ]] && cmd+=(--fixed_lr_epochs 6)
  [[ "$TASK" == "causal" ]] && cmd+=(--block_size "$CAUSAL_BLOCK_SIZE")
  [[ -n "$DATA_CACHE_DIR" ]] && cmd+=(--data_cache_dir "$DATA_CACHE_DIR")
  [[ -n "$MODEL_CACHE_DIR" ]] && cmd+=(--model_cache_dir "$MODEL_CACHE_DIR")
  [[ -n "$EXTRA_TRAIN_ARGS" ]] && cmd+=(${=EXTRA_TRAIN_ARGS})

  run_cmd=("${cmd[@]}")
  if [[ "$NUMA_NODE" != "none" ]]; then
    run_cmd=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" "${cmd[@]}")
  fi

  {
    echo "date=$(date -Is)"
    echo "method_index=$method_index"
    echo "aggregation_method=$method"
    echo "master_port=$method_master_port"
    echo "command=${(q)run_cmd[@]}"
  } > "$method_run_dir/command.log"

  echo "[$(date -Is)] node=$NODE_INDEX workers=$WORKERS task=$TASK model=$MODEL_LABEL method=$method start"
  set +e
  (cd "$REPO_DIR/simulations_llm" && "${run_cmd[@]}") > "$method_run_dir/node_${NODE_INDEX}.out" 2> "$method_run_dir/node_${NODE_INDEX}.err"
  method_status=$?
  set -e
  echo "[$(date -Is)] node=$NODE_INDEX workers=$WORKERS task=$TASK model=$MODEL_LABEL method=$method exit=$method_status"
  echo "$method_status" > "$method_run_dir/node_${NODE_INDEX}.status"

  if (( method_status != 0 )); then
    overall_status=$method_status
    (( STOP_ON_FAILURE )) && exit "$method_status"
  fi

  method_index=$(( method_index + 1 ))
done

exit "$overall_status"
