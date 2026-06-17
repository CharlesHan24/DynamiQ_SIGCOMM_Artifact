#!/usr/bin/env zsh
setopt errexit nounset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NODE_SCRIPT="$SCRIPT_DIR/qsub_simulation_matrix_node.zsh"
RESULT_ROOT="$SCRIPT_DIR/simulation_matrix_results"

NODES_RAW="chip-207-3,chip-207-4,chip-207-5,chip-207-6,chip-207-7,chip-207-8,chip-207-9,chip-207-10"
WORKERS_RAW="2,4,8"
PROCS_PER_NODE=1
PROJECT="cmic_hpc"
H_RT=60000
TMEM="1K"
PE_GPU=1
CONDA_ENV="llm"

COMBOS_RAW="causal:llama:meta-llama/Llama-3.2-1B,causal:gemma:gemma,mmlu:llama:meta-llama/Llama-3.2-1B,wikitext:bert-large:bert-large-cased"
AGGREGATION_METHODS_RAW="dynamiQ_mee_5bit_dynamic_bitrate"

BASE_PORT=""
METHOD_PORT_STRIDE=200
EXPERIMENT_PORT_STRIDE=4000
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
SYNC_EACH_EXPERIMENT=1
CONTINUE_ON_FAILURE=0
DRY_RUN=0
LAUNCH_MODE="qsub"
SSH_BIN="ssh"
TOPOLOGY="ring"

usage() {
  cat <<'EOF'
Usage:
  simulations_llm/submit_qsub_simulation_matrix.zsh [options]

Submits simulation jobs for:
  causal + Llama, causal + Gemma, MMLU + Llama, and Wikitext-103 + BERT-large-cased.
Each experiment is run for the requested worker counts, defaulting to 2, 4, and 8.

Important options:
  --nodes <csv>                 Hosts available to use. Need at least workers / procs-per-node.
  --workers <csv>               Worker counts to run. Default: 2,4,8
  --combos <csv>                task:label:model triples.
  --aggregation-methods <csv>   Simulation aggregation methods.
  --topology <ring|butterfly>   Communication topology. Default: ring
  --result-root <path>          Output root. Default: simulations_llm/simulation_matrix_results
  --launch-mode <qsub|direct>   Launch via SGE qsub or direct ssh. Default: qsub
  --ssh-bin <path>              SSH client for --launch-mode direct. Default: ssh
  --procs-per-node <n>          Processes per submitted node job. Default: 1
  --learning-rate <x>           Override task/model LR formula.
  --per-device-train-batch-size <n>
                                Override task defaults.
  --data-cache-dir <path>       Dataset cache passed to train scripts. Default: env/HF cache
  --model-cache-dir <path>      Model cache passed to train scripts. Default: env/HF cache
  --extra-train-args <string>   Raw extra args appended to each train command.
  --dry-run                     Print launcher commands without submitting or running.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) NODES_RAW="$2"; shift 2 ;;
    --workers) WORKERS_RAW="$2"; shift 2 ;;
    --combos) COMBOS_RAW="$2"; shift 2 ;;
    --aggregation-methods|--aggregation_methods) AGGREGATION_METHODS_RAW="$2"; shift 2 ;;
    --topology) TOPOLOGY="$2"; shift 2 ;;
    --result-root|--result_root) RESULT_ROOT="$2"; shift 2 ;;
    --launch-mode) LAUNCH_MODE="$2"; shift 2 ;;
    --ssh-bin) SSH_BIN="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --method-port-stride) METHOD_PORT_STRIDE="$2"; shift 2 ;;
    --experiment-port-stride) EXPERIMENT_PORT_STRIDE="$2"; shift 2 ;;
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
    --procs-per-node) PROCS_PER_NODE="$2"; PE_GPU="$2"; shift 2 ;;
    --pe-gpu) PE_GPU="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --h-rt) H_RT="$2"; shift 2 ;;
    --tmem) TMEM="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --continue-on-failure) CONTINUE_ON_FAILURE=1; shift ;;
    --no-sync) SYNC_EACH_EXPERIMENT=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -x "$NODE_SCRIPT" ]]; then
  echo "Node runner is missing or not executable: $NODE_SCRIPT" >&2
  exit 1
fi

nodes=()
for raw_node in ${(s:,:)NODES_RAW}; do
  node="${raw_node//[[:space:]]/}"
  [[ -n "$node" ]] && nodes+=("$node")
done
if (( ${#nodes[@]} == 0 )); then
  echo "--nodes must contain at least one host" >&2
  exit 1
fi
typeset -A seen_nodes
for node in "${nodes[@]}"; do
  if [[ "$node" == *[[:space:]]* ]]; then
    echo "Invalid node '$node'. Hostnames must not contain whitespace." >&2
    exit 1
  fi
  if [[ -n "${seen_nodes[$node]:-}" ]]; then
    echo "Duplicate node '$node' in --nodes" >&2
    exit 1
  fi
  seen_nodes[$node]=1
done

if [[ "$LAUNCH_MODE" != "qsub" && "$LAUNCH_MODE" != "direct" ]]; then
  echo "--launch-mode must be qsub or direct" >&2
  exit 1
fi

if [[ "$TOPOLOGY" != "ring" && "$TOPOLOGY" != "butterfly" ]]; then
  echo "--topology must be ring or butterfly" >&2
  exit 1
fi

normalize_aggregation_methods_csv() {
  local raw="$1"
  local -a methods
  local method=""
  local normalized=""

  methods=(${(s:,:)raw})
  for method in "${methods[@]}"; do
    method="${method//[[:space:]]/}"
    [[ -n "$method" ]] || continue
    if [[ "$TOPOLOGY" == "butterfly" && "$method" != *butterfly* ]]; then
      method="${method}_butterfly"
    fi
    normalized+="${method},"
  done

  print -r -- "${normalized%,}"
}

AGGREGATION_METHODS_RAW="$(normalize_aggregation_methods_csv "$AGGREGATION_METHODS_RAW")"

workers_list=()
for raw_workers in ${(s:,:)WORKERS_RAW}; do
  workers="${raw_workers//[[:space:]]/}"
  [[ -n "$workers" ]] || continue
  case "$workers" in
    2|4|8) workers_list+=("$workers") ;;
    *) echo "Unsupported worker count '$workers'. Expected 2, 4, or 8." >&2; exit 1 ;;
  esac
done
if (( ${#workers_list[@]} == 0 )); then
  echo "--workers must include at least one of 2,4,8" >&2
  exit 1
fi

if [[ -z "$BASE_PORT" ]]; then
  BASE_PORT=$((23000 + ($(date +%s) % 300) * 100))
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_${JOB_ID:-manual}_$$"
RUN_DIR="$RESULT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

{
  echo "date=$(date -Is)"
  echo "run_id=$RUN_ID"
  echo "nodes=${nodes[*]}"
  echo "workers=$WORKERS_RAW"
  echo "procs_per_node=$PROCS_PER_NODE"
  echo "combos=$COMBOS_RAW"
  echo "aggregation_methods=$AGGREGATION_METHODS_RAW"
  echo "topology=$TOPOLOGY"
  echo "launch_mode=$LAUNCH_MODE"
  echo "ssh_bin=$SSH_BIN"
  echo "base_port=$BASE_PORT"
  echo "method_port_stride=$METHOD_PORT_STRIDE"
  echo "experiment_port_stride=$EXPERIMENT_PORT_STRIDE"
  echo "normalized=$NORMALIZED"
  echo "normalized_chunk_size=$NORMALIZED_CHUNK_SIZE"
  echo "quantization_levels=$QUANTIZATION_LEVELS"
  echo "learning_rate=${LEARNING_RATE:-auto}"
  echo "per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-task_default}"
  echo "data_cache_dir=${DATA_CACHE_DIR:-hf_default}"
  echo "model_cache_dir=${MODEL_CACHE_DIR:-hf_default}"
} > "$RUN_DIR/submit_metadata.log"

quote_command() {
  local quoted=""
  local word=""
  for word in "$@"; do
    quoted+="${(q)word} "
  done
  print -r -- "${quoted% }"
}

DIRECT_PIDS=()
DIRECT_LABELS=()

wait_for_direct_batch() {
  local batch_status=0
  local idx=0
  local pid=""
  local label=""
  local status=0

  for (( idx = 1; idx <= ${#DIRECT_PIDS[@]}; idx++ )); do
    pid="${DIRECT_PIDS[$idx]}"
    label="${DIRECT_LABELS[$idx]}"
    set +e
    wait "$pid"
    status=$?
    set -e
    if (( status != 0 )); then
      echo "Direct launch failed for $label with status $status" >&2
      if (( batch_status == 0 )); then
        batch_status=$status
      fi
    fi
  done

  DIRECT_PIDS=()
  DIRECT_LABELS=()
  return "$batch_status"
}

submit_one_node() {
  local experiment_index="$1"
  local workers="$2"
  local task="$3"
  local model_label="$4"
  local model_name="$5"
  local experiment_dir="$6"
  local node_index="$7"
  local node="$8"
  local sync_this="$9"
  local num_nodes=$(( workers / PROCS_PER_NODE ))
  local master_addr
  local master_port=$(( BASE_PORT + experiment_index * EXPERIMENT_PORT_STRIDE ))
  local safe_task="${task//[^A-Za-z0-9_.-]/_}"
  local safe_model="${model_label//[^A-Za-z0-9_.-]/_}"
  local name="sim_${workers}w_${safe_task}_${safe_model}_${node_index}"
  local -a node_args
  local -a qsub_args
  local -a direct_args
  local remote_cmd=""

  master_addr="$(getent ahostsv4 "${nodes[1]}" | awk '{print $1; exit}')"
  [[ -n "$master_addr" ]] || master_addr="${nodes[1]}"

  node_args=(
    "$NODE_SCRIPT"
    --repo-dir "$REPO_DIR"
    --run-dir "$experiment_dir"
    --node-index "$node_index"
    --master-addr "$master_addr"
    --master-port "$master_port"
    --task "$task"
    --model "$model_name"
    --model-label "$model_label"
    --aggregation-methods "$AGGREGATION_METHODS_RAW"
    --workers "$workers"
    --num-nodes "$num_nodes"
    --procs-per-node "$PROCS_PER_NODE"
    --method-port-stride "$METHOD_PORT_STRIDE"
    --rdzv-timeout "$RDZV_TIMEOUT"
    --normalized "$NORMALIZED"
    --normalized-chunk-size "$NORMALIZED_CHUNK_SIZE"
    --quantization-levels "$QUANTIZATION_LEVELS"
    --nbits-communication "$NBITS_COMMUNICATION"
    --seed "$SEED"
    --resume-dir "$RESUME_DIR"
    --rotation "$ROTATION"
    --fine-tuning "$FINE_TUNING"
    --to-perm "$TO_PERM"
    --sparsity "$SPARSITY"
    --agg-chunk-size "$AGG_CHUNK_SIZE"
    --causal-block-size "$CAUSAL_BLOCK_SIZE"
    --gpu-max-used-mb "$GPU_MAX_USED_MB"
    --numa-node "$NUMA_NODE"
    --conda-env "$CONDA_ENV"
  )
  [[ -n "$LEARNING_RATE" ]] && node_args+=(--learning-rate "$LEARNING_RATE")
  [[ -n "$PER_DEVICE_TRAIN_BATCH_SIZE" ]] && node_args+=(--per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE")
  [[ -n "$DATA_CACHE_DIR" ]] && node_args+=(--data-cache-dir "$DATA_CACHE_DIR")
  [[ -n "$MODEL_CACHE_DIR" ]] && node_args+=(--model-cache-dir "$MODEL_CACHE_DIR")
  [[ -n "$EXTRA_TRAIN_ARGS" ]] && node_args+=(--extra-train-args "$EXTRA_TRAIN_ARGS")
  (( CONTINUE_ON_FAILURE )) && node_args+=(--continue-on-failure)

  if [[ "$LAUNCH_MODE" == "qsub" ]]; then
    qsub_args=(
      qsub
      -cwd
      -S /bin/zsh
      -N "$name"
      -P "$PROJECT"
      -R y
      -l "gpu=true,hostname=($node),tmem=$TMEM,h_rt=$H_RT"
      -pe gpu "$PE_GPU"
      -o "$experiment_dir/node_${node_index}_${node}.qsub.out"
      -e "$experiment_dir/node_${node_index}_${node}.qsub.err"
    )
    (( sync_this )) && qsub_args+=(-sync y)

    qsub_args+=("${node_args[@]}")

    print -r -- "${(q)qsub_args[@]}" >> "$RUN_DIR/qsub_commands.log"
    if (( DRY_RUN )); then
      print -r -- "${(q)qsub_args[@]}"
    else
      "${qsub_args[@]}"
    fi
  else
    remote_cmd="cd ${(q)REPO_DIR} && $(quote_command "${node_args[@]}")"
    direct_args=("$SSH_BIN" "$node" "$remote_cmd")
    print -r -- "${(q)direct_args[@]}" >> "$RUN_DIR/direct_commands.log"
    if (( DRY_RUN )); then
      print -r -- "${(q)direct_args[@]}"
    else
      "${direct_args[@]}" > "$experiment_dir/node_${node_index}_${node}.direct.out" 2> "$experiment_dir/node_${node_index}_${node}.direct.err" &
      DIRECT_PIDS+=("$!")
      DIRECT_LABELS+=("experiment=$experiment_index workers=$workers task=$task model=$model_label node=$node_index host=$node")
    fi
  fi
}

experiment_index=0
script_status=0
for workers in "${workers_list[@]}"; do
  if (( workers % PROCS_PER_NODE != 0 )); then
    echo "workers=$workers is not divisible by procs_per_node=$PROCS_PER_NODE" >&2
    exit 1
  fi
  num_nodes=$(( workers / PROCS_PER_NODE ))
  if (( num_nodes > ${#nodes[@]} )); then
    echo "Need $num_nodes hosts for $workers workers, but only ${#nodes[@]} hosts were provided." >&2
    exit 1
  fi

  for raw_combo in ${(s:,:)COMBOS_RAW}; do
    combo="${raw_combo//[[:space:]]/}"
    [[ -n "$combo" ]] || continue
    task="${combo%%:*}"
    rest="${combo#*:}"
    model_label="${rest%%:*}"
    model_name="${rest#*:}"
    if [[ "$task" == "$combo" || "$model_label" == "$rest" || -z "$task" || -z "$model_label" || -z "$model_name" ]]; then
      echo "Invalid combo '$combo'. Expected task:label:model." >&2
      exit 1
    fi

    safe_combo="${task}_${model_label}_${workers}w"
    safe_combo="${safe_combo//[^A-Za-z0-9_.-]/_}"
    experiment_dir="$RUN_DIR/experiment_${experiment_index}_${safe_combo}"
    mkdir -p "$experiment_dir"
    {
      echo "date=$(date -Is)"
      echo "workers=$workers"
      echo "num_nodes=$num_nodes"
      echo "procs_per_node=$PROCS_PER_NODE"
      echo "task=$task"
      echo "model_label=$model_label"
      echo "model_name=$model_name"
      echo "aggregation_methods=$AGGREGATION_METHODS_RAW"
    } > "$experiment_dir/experiment_metadata.log"

    echo "Submitting workers=$workers task=$task model=$model_label dir=$experiment_dir"
    for (( node_index = 0; node_index < num_nodes; node_index++ )); do
      node="${nodes[$((node_index + 1))]}"
      sync_this=0
      (( SYNC_EACH_EXPERIMENT && node_index == num_nodes - 1 )) && sync_this=1
      set +e
      submit_one_node "$experiment_index" "$workers" "$task" "$model_label" "$model_name" "$experiment_dir" "$node_index" "$node" "$sync_this"
      submit_status=$?
      set -e
      if (( submit_status != 0 )); then
        if (( script_status == 0 )); then
          script_status=$submit_status
        fi
        if (( ! CONTINUE_ON_FAILURE )); then
          exit "$submit_status"
        fi
      fi
    done

    if [[ "$LAUNCH_MODE" == "direct" && "$DRY_RUN" -eq 0 && "$SYNC_EACH_EXPERIMENT" -eq 1 ]]; then
      set +e
      wait_for_direct_batch
      experiment_status=$?
      set -e
      if (( experiment_status != 0 )); then
        if (( script_status == 0 )); then
          script_status=$experiment_status
        fi
        if (( ! CONTINUE_ON_FAILURE )); then
          exit "$experiment_status"
        fi
      fi
    fi

    experiment_index=$(( experiment_index + 1 ))
  done
done

if [[ "$LAUNCH_MODE" == "direct" && "$DRY_RUN" -eq 0 && "$SYNC_EACH_EXPERIMENT" -eq 0 && ${#DIRECT_PIDS[@]} -gt 0 ]]; then
  set +e
  wait_for_direct_batch
  direct_status=$?
  set -e
  if (( direct_status != 0 && script_status == 0 )); then
    script_status=$direct_status
  fi
fi

echo "Run directory: $RUN_DIR"
exit "$script_status"
