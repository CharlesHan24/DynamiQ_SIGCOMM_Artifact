#!/usr/bin/env zsh
unsetopt errexit nounset
source ~/.zshrc
setopt errexit nounset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NODE_SCRIPT="$SCRIPT_DIR/qsub_llm_hook_matrix_node.zsh"
RESULT_ROOT="$SCRIPT_DIR/llm_hook_matrix_results"

NODES_RAW="chip-207-3,chip-207-4,chip-207-5,chip-207-6"
NUM_NODES=0
PROCS_PER_NODE=2
PROJECT="cmic_hpc"
H_RT=60000
TMEM="1K"
PE_GPU=2
CONDA_ENV="llm"

COMBOS_RAW="causal:llama:meta-llama/Llama-3.2-1B,causal:gemma:gemma,mmlu:llama:meta-llama/Llama-3.2-1B,wikitext:bert-large:bert-large-cased"
AGGREGATION_METHODS_RAW="bf16,MXfp8,fp4,fp6,zero,dynamiQ_aee_5bit,dynamiQ_mee_5bit,dynamiQ_mee_5bit_dynamic_bitrate,omnireduce,thc"

BASE_PORT=""
MASTER_PORT=""
METHOD_PORT_STRIDE=200
COMBO_PORT_STRIDE=4000
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
RAILS=1
IFACE0="eth0"
IFACE1="eth1"
GID_INDEX=-1
DYNAMIC_AEE_PIPELINE_RDMA="${DYNAMIC_AEE_PIPELINE_RDMA:-0}"
RING_RDMA_PIPELINE_CHUNK_MB="${RING_RDMA_PIPELINE_CHUNK_MB:-8}"
RING_RDMA_PIPELINE_INFLIGHT="${RING_RDMA_PIPELINE_INFLIGHT:-2}"
RING_LOCAL_P2P_CHUNK_MB="${RING_LOCAL_P2P_CHUNK_MB:-8}"

CUDA_HOME_DIR="${CUDA_HOME:-/share/apps/cuda-11.8}"
GCC_HOME="${GCC_HOME:-/share/apps/gcc-8.3}"
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9}"

SYNC_EACH_COMBO=1
CONTINUE_ON_FAILURE=0
DRY_RUN=0
LAUNCH_MODE="qsub"
SSH_BIN="ssh"
TOPOLOGY="ring"

usage() {
  cat <<'EOF'
Usage:
  ./submit_qsub_llm_hook_matrix.zsh [options]

Runs four LLM test matrices by default:
  causal + llama, causal + gemma, mmlu + llama, and wikitext-103 + bert-large.
Each matrix launches one node runner per node, uses two GPUs per node, and walks the
hook list sequentially inside the node runner.

Options:
  --nodes <csv>                 Two or four hostnames. Default: chip-207-3,chip-207-4,chip-207-5,chip-207-6
  --combos <csv>                task:label:model triples. Tasks: causal, mmlu, wikitext/maskedlm
  --aggregation-methods <csv>   Hook method list. Default: bf16,MXfp8,fp4,fp6,zero,dynamiQ_aee_5bit,dynamiQ_mee_5bit,omnireduce,thc
  --topology <ring|butterfly>   Communication topology. Default: ring
  --result-root <path>          Result root. Default: testbed_evaluation/llm_hook_matrix_results
  --launch-mode <qsub|direct>   Launch via SGE qsub or direct ssh. Default: qsub
  --ssh-bin <path>              SSH client for --launch-mode direct. Default: ssh
  --base-port <port>            First RDMA base port. Default: derived from timestamp
  --master-port <port>          First accelerate master port. Default: base-port + 90
  --method-port-stride <n>      Port stride between methods. Default: 200
  --combo-port-stride <n>       Port stride between scenario matrices. Default: 4000
  --rdzv-timeout <seconds>      Accelerate rendezvous timeout. Default: 360000
  --gpu-pair-starts <csv>       XOR GPU pair starts to try. Default: 4,2,6,0
  --max-used-mb <MiB>           Treat a GPU as busy above this memory. Default: 512
  --numa-node <n|none>          numactl CPU/memory node binding. Default: 1
  --rails <1|2>                 RDMA rails. Default: 1
  --iface0 <name>               First NIC interface. Default: eth0
  --iface1 <name>               Second NIC interface. Default: eth1
  --gid-index <idx>             RoCE GID index. Default: -1
  --dynamic-pipeline-rdma <0|1> Enable pipelined dynamic AEE/MEE RDMA. Default: env or 0
  --pipeline-chunk-mb <n>       Pipelined RDMA tile size per rail in MiB. Default: env or 8
  --pipeline-inflight <n>       Pipelined RDMA in-flight tiles per rail. Default: env or 2
  --local-p2p-chunk-mb <n>      Local torch P2P chunk size in MiB. Default: env or 8
  --normalized <0|1>            Train script normalized flag. Default: 1
  --normalized-chunk-size <n>   Train script normalized chunk size. Default: 1024
  --quantization-levels <n>     Train script quantization levels. Default: 120
  --nbits-communication <n>     Train script communication bit-width. Default: 8
  --sparsity <x|None>           Train script sparsity. Default: None
  --agg-chunk-size <n>          Train script agg_chunk_size. Default: 16
  --num-train-epochs <n>        Train epochs. Default: 3
  --learning-rate <x>           Override LR for every scenario. Default: per-task/model formula
  --causal-block-size <n>       Causal sequence block size. Default: 3000
  --per-device-train-batch-size <n>
                                  Override per-device batch size for every scenario
  --to-shrimp <True|False>      Override to_shrimp for every scenario
  --data-cache-dir <path>       Dataset cache passed to train scripts. Default: env/HF cache
  --model-cache-dir <path>      Model cache passed to train scripts. Default: env/HF cache
  --extra-train-args <string>   Extra raw train args appended by the node runner
  --conda-env <name>            Conda env activated inside qsub jobs. Default: llm
  --cuda-home <path>            CUDA toolkit root. Default: /share/apps/cuda-11.8
  --gcc-home <path>             GCC root. Default: /share/apps/gcc-8.3
  --cuda-arch-list <list>       TORCH_CUDA_ARCH_LIST. Default: 8.0;8.6;8.9
  --pe-gpu <n>                  qsub -pe gpu value. Default: 2
  --project <name>              SGE project. Default: cmic_hpc
  --h-rt <seconds>              SGE runtime limit. Default: 60000
  --tmem <amount>               SGE memory request. Default: 1K
  --continue-on-failure         Keep walking methods and later scenario matrices after failures
  --no-sync                     Submit all scenario matrices without waiting between them
  --dry-run                     Print launcher commands without submitting or running
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) NODES_RAW="$2"; shift 2 ;;
    --combos) COMBOS_RAW="$2"; shift 2 ;;
    --aggregation-methods|--aggregation_methods) AGGREGATION_METHODS_RAW="$2"; shift 2 ;;
    --topology) TOPOLOGY="$2"; shift 2 ;;
    --result-root|--result_root) RESULT_ROOT="$2"; shift 2 ;;
    --launch-mode) LAUNCH_MODE="$2"; shift 2 ;;
    --ssh-bin) SSH_BIN="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --method-port-stride) METHOD_PORT_STRIDE="$2"; shift 2 ;;
    --combo-port-stride) COMBO_PORT_STRIDE="$2"; shift 2 ;;
    --rdzv-timeout|--rdzv_timeout) RDZV_TIMEOUT="$2"; shift 2 ;;
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
    --rails) RAILS="$2"; shift 2 ;;
    --iface0) IFACE0="$2"; shift 2 ;;
    --iface1) IFACE1="$2"; shift 2 ;;
    --gid-index) GID_INDEX="$2"; shift 2 ;;
    --dynamic-pipeline-rdma) DYNAMIC_AEE_PIPELINE_RDMA="$2"; shift 2 ;;
    --pipeline-chunk-mb) RING_RDMA_PIPELINE_CHUNK_MB="$2"; shift 2 ;;
    --pipeline-inflight) RING_RDMA_PIPELINE_INFLIGHT="$2"; shift 2 ;;
    --local-p2p-chunk-mb) RING_LOCAL_P2P_CHUNK_MB="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --cuda-home) CUDA_HOME_DIR="$2"; shift 2 ;;
    --gcc-home) GCC_HOME="$2"; shift 2 ;;
    --cuda-arch-list) CUDA_ARCH_LIST="$2"; shift 2 ;;
    --pe-gpu) PE_GPU="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --h-rt) H_RT="$2"; shift 2 ;;
    --tmem) TMEM="$2"; shift 2 ;;
    --continue-on-failure) CONTINUE_ON_FAILURE=1; shift ;;
    --no-sync) SYNC_EACH_COMBO=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

nodes=()
for raw_node in ${(s:,:)NODES_RAW}; do
  node="${raw_node//[[:space:]]/}"
  [[ -n "$node" ]] && nodes+=("$node")
done

if (( ${#nodes[@]} != 2 && ${#nodes[@]} != 4 )); then
  echo "--nodes must contain either two or four hosts" >&2
  exit 1
fi
NUM_NODES=${#nodes[@]}

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

if [[ "$RAILS" != "1" && "$RAILS" != "2" ]]; then
  echo "--rails must be 1 or 2" >&2
  exit 1
fi

if [[ "$RAILS" == "2" && -z "$IFACE1" ]]; then
  echo "--iface1 is required with --rails 2" >&2
  exit 1
fi

if [[ ! -x "$NODE_SCRIPT" ]]; then
  echo "Node script is missing or not executable: $NODE_SCRIPT" >&2
  exit 1
fi

WORLD_SIZE=$(( NUM_NODES * PROCS_PER_NODE ))
if [[ -z "$BASE_PORT" ]]; then
  BASE_PORT=$((19000 + ($(date +%s) % 300) * 100))
fi
if [[ -z "$MASTER_PORT" ]]; then
  MASTER_PORT=$((BASE_PORT + 90))
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_${JOB_ID:-manual}_$$"
RUN_DIR="$RESULT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

MASTER_ADDR="$(getent ahostsv4 "${nodes[1]}" | awk '{print $1; exit}')"
if [[ -z "$MASTER_ADDR" ]]; then
  MASTER_ADDR="${nodes[1]}"
fi

cat > "$RUN_DIR/build_eden_utils_command.zsh" <<EOF
#!/usr/bin/env zsh
CONDA_ENV=$CONDA_ENV \\
CUDA_HOME=$CUDA_HOME_DIR \\
GCC_HOME=$GCC_HOME \\
TORCH_CUDA_ARCH_LIST='$CUDA_ARCH_LIST' \\
"$SCRIPT_DIR/build_eden_utils_llm.zsh"
EOF
chmod +x "$RUN_DIR/build_eden_utils_command.zsh"

{
  echo "date=$(date -Is)"
  echo "run_id=$RUN_ID"
  echo "nodes=${nodes[*]}"
  echo "num_nodes=$NUM_NODES"
  echo "procs_per_node=$PROCS_PER_NODE"
  echo "world_size=$WORLD_SIZE"
  echo "master_addr=$MASTER_ADDR"
  echo "base_port=$BASE_PORT"
  echo "master_port=$MASTER_PORT"
  echo "method_port_stride=$METHOD_PORT_STRIDE"
  echo "combo_port_stride=$COMBO_PORT_STRIDE"
  echo "combos=$COMBOS_RAW"
  echo "aggregation_methods=$AGGREGATION_METHODS_RAW"
  echo "topology=$TOPOLOGY"
  echo "launch_mode=$LAUNCH_MODE"
  echo "ssh_bin=$SSH_BIN"
  echo "gpu_pair_starts=$GPU_PAIR_STARTS"
  echo "max_used_mb=$MAX_USED_MB"
  echo "numa_node=$NUMA_NODE"
  echo "rails=$RAILS"
  echo "iface0=$IFACE0"
  echo "iface1=$IFACE1"
  echo "gid_index=$GID_INDEX"
  echo "dynamic_aee_pipeline_rdma=$DYNAMIC_AEE_PIPELINE_RDMA"
  echo "ring_rdma_pipeline_chunk_mb=$RING_RDMA_PIPELINE_CHUNK_MB"
  echo "ring_rdma_pipeline_inflight=$RING_RDMA_PIPELINE_INFLIGHT"
  echo "ring_local_p2p_chunk_mb=$RING_LOCAL_P2P_CHUNK_MB"
  echo "normalized=$NORMALIZED"
  echo "normalized_chunk_size=$NORMALIZED_CHUNK_SIZE"
  echo "quantization_levels=$QUANTIZATION_LEVELS"
  echo "nbits_communication=$NBITS_COMMUNICATION"
  echo "sparsity=$SPARSITY"
  echo "agg_chunk_size=$AGG_CHUNK_SIZE"
  echo "num_train_epochs=$NUM_TRAIN_EPOCHS"
  echo "learning_rate=${LEARNING_RATE:-auto}"
  echo "causal_block_size=$CAUSAL_BLOCK_SIZE"
  echo "per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-task_default}"
  echo "to_shrimp=${TO_SHRIMP:-task_default}"
  echo "data_cache_dir=${DATA_CACHE_DIR:-hf_default}"
  echo "model_cache_dir=${MODEL_CACHE_DIR:-hf_default}"
  echo "extra_train_args=$EXTRA_TRAIN_ARGS"
  echo "conda_env=$CONDA_ENV"
  echo "cuda_home=$CUDA_HOME_DIR"
  echo "gcc_home=$GCC_HOME"
  echo "cuda_arch_list=$CUDA_ARCH_LIST"
  echo "project=$PROJECT"
  echo "h_rt=$H_RT"
  echo "tmem=$TMEM"
  echo "pe_gpu=$PE_GPU"
  echo "sync_each_combo=$SYNC_EACH_COMBO"
  echo "continue_on_failure=$CONTINUE_ON_FAILURE"
  echo "build_command=$RUN_DIR/build_eden_utils_command.zsh"
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

submit_one() {
  local combo_index="$1"
  local task="$2"
  local model_label="$3"
  local model_name="$4"
  local combo_run_dir="$5"
  local combo_master_port="$6"
  local combo_base_port="$7"
  local node_index="$8"
  local node="$9"
  local sync_this="${10}"
  local safe_model="${model_label//[^A-Za-z0-9_.-]/_}"
  local name="llmhm_${combo_index}_${task}_${safe_model}_${node_index}"
  local -a node_args
  local -a qsub_args
  local -a direct_args
  local remote_cmd=""

  node_args=(
    "$NODE_SCRIPT"
    --repo-dir "$REPO_DIR"
    --run-dir "$combo_run_dir"
    --node-index "$node_index"
    --master-addr "$MASTER_ADDR"
    --master-port "$combo_master_port"
    --base-port "$combo_base_port"
    --task "$task"
    --model "$model_name"
    --model-label "$model_label"
    --aggregation-methods "$AGGREGATION_METHODS_RAW"
    --num-nodes "$NUM_NODES"
    --procs-per-node "$PROCS_PER_NODE"
    --method-port-stride "$METHOD_PORT_STRIDE"
    --rdzv-timeout "$RDZV_TIMEOUT"
    --normalized "$NORMALIZED"
    --normalized-chunk-size "$NORMALIZED_CHUNK_SIZE"
    --quantization-levels "$QUANTIZATION_LEVELS"
    --nbits-communication "$NBITS_COMMUNICATION"
    --seed "$SEED"
    --compression "$COMPRESSION"
    --resume-dir "$RESUME_DIR"
    --rotation "$ROTATION"
    --fine-tuning "$FINE_TUNING"
    --to-perm "$TO_PERM"
    --sparsity "$SPARSITY"
    --agg-chunk-size "$AGG_CHUNK_SIZE"
    --num-train-epochs "$NUM_TRAIN_EPOCHS"
    --causal-block-size "$CAUSAL_BLOCK_SIZE"
    --gpu-pair-starts "$GPU_PAIR_STARTS"
    --max-used-mb "$MAX_USED_MB"
    --numa-node "$NUMA_NODE"
    --rails "$RAILS"
    --iface0 "$IFACE0"
    --iface1 "$IFACE1"
    --gid-index "$GID_INDEX"
    --dynamic-pipeline-rdma "$DYNAMIC_AEE_PIPELINE_RDMA"
    --pipeline-chunk-mb "$RING_RDMA_PIPELINE_CHUNK_MB"
    --pipeline-inflight "$RING_RDMA_PIPELINE_INFLIGHT"
    --local-p2p-chunk-mb "$RING_LOCAL_P2P_CHUNK_MB"
    --conda-env "$CONDA_ENV"
    --cuda-home "$CUDA_HOME_DIR"
    --gcc-home "$GCC_HOME"
    --cuda-arch-list "$CUDA_ARCH_LIST"
  )

  if [[ -n "$LEARNING_RATE" ]]; then
    node_args+=(--learning-rate "$LEARNING_RATE")
  fi
  if [[ -n "$PER_DEVICE_TRAIN_BATCH_SIZE" ]]; then
    node_args+=(--per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE")
  fi
  if [[ -n "$TO_SHRIMP" ]]; then
    node_args+=(--to-shrimp "$TO_SHRIMP")
  fi
  if [[ -n "$DATA_CACHE_DIR" ]]; then
    node_args+=(--data-cache-dir "$DATA_CACHE_DIR")
  fi
  if [[ -n "$MODEL_CACHE_DIR" ]]; then
    node_args+=(--model-cache-dir "$MODEL_CACHE_DIR")
  fi
  if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
    node_args+=(--extra-train-args "$EXTRA_TRAIN_ARGS")
  fi
  if (( CONTINUE_ON_FAILURE )); then
    node_args+=(--continue-on-failure)
  fi

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
      -o "$combo_run_dir/node_${node_index}_${node}.qsub.out"
      -e "$combo_run_dir/node_${node_index}_${node}.qsub.err"
    )

    if (( sync_this )); then
      qsub_args+=(-sync y)
    fi

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
      "${direct_args[@]}" > "$combo_run_dir/node_${node_index}_${node}.direct.out" 2> "$combo_run_dir/node_${node_index}_${node}.direct.err" &
      DIRECT_PIDS+=("$!")
      DIRECT_LABELS+=("combo=$combo_index task=$task model=$model_label node=$node_index host=$node")
    fi
  fi
}

combo_index=0
script_status=0
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
  case "$task" in
    causal|mmlu|wikitext|maskedlm|mlm) ;;
    *)
      echo "Invalid task '$task' in combo '$combo'. Use causal, mmlu, or wikitext/maskedlm." >&2
      exit 1
      ;;
  esac

  safe_combo="${task}_${model_label}"
  safe_combo="${safe_combo//[^A-Za-z0-9_.-]/_}"
  combo_run_dir="$RUN_DIR/combo_${combo_index}_${safe_combo}"
  mkdir -p "$combo_run_dir"

  combo_master_port=$(( MASTER_PORT + combo_index * COMBO_PORT_STRIDE ))
  combo_base_port=$(( BASE_PORT + combo_index * COMBO_PORT_STRIDE ))

  {
    echo "date=$(date -Is)"
    echo "combo_index=$combo_index"
    echo "task=$task"
    echo "model_label=$model_label"
    echo "model_name=$model_name"
    echo "master_port=$combo_master_port"
    echo "base_port=$combo_base_port"
    echo "aggregation_methods=$AGGREGATION_METHODS_RAW"
  } > "$combo_run_dir/combo_metadata.log"

  echo "Submitting combo=$combo_index task=$task model=$model_label run_dir=$combo_run_dir"

  for (( idx = 0; idx < NUM_NODES; idx++ )); do
    node="${nodes[$((idx + 1))]}"
    sync_this=0
    if (( SYNC_EACH_COMBO && idx == NUM_NODES - 1 )); then
      sync_this=1
    fi

    set +e
    submit_one "$combo_index" "$task" "$model_label" "$model_name" "$combo_run_dir" "$combo_master_port" "$combo_base_port" "$idx" "$node" "$sync_this"
    node_status=$?
    set -e
    if (( node_status != 0 )); then
      echo "Node-$idx qsub failed for combo $combo_index with status $node_status" >&2
      if (( script_status == 0 )); then
        script_status=$node_status
      fi
      if (( ! CONTINUE_ON_FAILURE )); then
        exit "$node_status"
      fi
    fi
  done

  if [[ "$LAUNCH_MODE" == "direct" && "$DRY_RUN" -eq 0 && "$SYNC_EACH_COMBO" -eq 1 ]]; then
    set +e
    wait_for_direct_batch
    combo_status=$?
    set -e
    if (( combo_status != 0 )); then
      if (( script_status == 0 )); then
        script_status=$combo_status
      fi
      if (( ! CONTINUE_ON_FAILURE )); then
        exit "$combo_status"
      fi
    fi
  fi

  combo_index=$(( combo_index + 1 ))
done

if [[ "$LAUNCH_MODE" == "direct" && "$DRY_RUN" -eq 0 && "$SYNC_EACH_COMBO" -eq 0 && ${#DIRECT_PIDS[@]} -gt 0 ]]; then
  set +e
  wait_for_direct_batch
  direct_status=$?
  set -e
  if (( direct_status != 0 && script_status == 0 )); then
    script_status=$direct_status
  fi
fi

echo "Run directory: $RUN_DIR"
echo "Build command: $RUN_DIR/build_eden_utils_command.zsh"
exit "$script_status"
