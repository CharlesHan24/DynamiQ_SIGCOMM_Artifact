#!/usr/bin/env zsh
unsetopt errexit nounset
source ~/.zshrc
setopt errexit nounset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NODE_SCRIPT="$SCRIPT_DIR/qsub_smoke_new_comm_hooks_node.zsh"
RESULT_ROOT="$SCRIPT_DIR/smoke_job_results"

NODES_RAW="chip-207-3,chip-207-4,chip-207-5,chip-207-6"
NUM_NODES=0
PROCS_PER_NODE=2
PROJECT="cmic_hpc"
H_RT=60000
TMEM="1K"
PE_GPU=2
CONDA_ENV="llm"

AGGREGATION_METHOD="dynamiQ_mee_5bit"
DTYPE="bf16"
STEPS=25
NUMEL=1073741824
BUCKET_CAP_MB=512
CHECKSUM_ATOL="1e-2"
SKIP_GRAD_CHECK=0
EXPECT_BUTTERFLY_RDMA=0
NSYS_PROFILE_STEP=-1
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"

RAILS=1
IFACE0="eth0"
IFACE1="eth1"
BASE_PORT=""
MASTER_PORT=""
GID_INDEX=-1
GPU_PAIR_STARTS="6,4,2,0"
MAX_USED_MB=512
DYNAMIC_AEE_PIPELINE_RDMA="${DYNAMIC_AEE_PIPELINE_RDMA:-0}"
RING_RDMA_PIPELINE_CHUNK_MB="${RING_RDMA_PIPELINE_CHUNK_MB:-8}"
RING_RDMA_PIPELINE_INFLIGHT="${RING_RDMA_PIPELINE_INFLIGHT:-2}"

CUDA_HOME_DIR="${CUDA_HOME:-/share/apps/cuda-11.8}"
GCC_HOME="${GCC_HOME:-/share/apps/gcc-8.3}"
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9}"

SYNC_FLAG=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./submit_qsub_smoke_new_comm_hooks.zsh [options]

Options:
  --nodes <csv>              Two or four chip-207 hosts. Default: chip-207-3,chip-207-4
  --result-root <path>       Result root. Default: testbed_evaluation/smoke_job_results
  --aggregation-method <s>   Smoke hook method. Default: dynamiQ_aee_5bit
  --dtype <bf16|fp16|fp32>   Model/bucket dtype. Default: bf16
  --steps <n>                Backward passes. Default: 3
  --numel <n>                Synthetic parameter elements. Default: 262144
  --bucket-cap-mb <mb>       DDP bucket cap. Default: 25
  --checksum-atol <x>        Cross-rank checksum tolerance. Default: 1e-2
  --skip-grad-check          Skip checksum all_gather check
  --expect-butterfly-rdma    Fail unless the butterfly RDMA callback initializes
  --nsys-profile-step <n>    Run torchrun under nsys and capture only this step via cudaProfilerApi
  --nsys-trace <list>        nsys trace domains. Default: cuda,nvtx,osrt
  --rails <1|2>              RDMA rails. Default: 1
  --iface0 <name>            First NIC interface. Default: eth0
  --iface1 <name>            Second NIC interface. Default: eth1
  --base-port <port>         RDMA base port. Default: derived from timestamp
  --master-port <port>       torch.distributed master port. Default: base-port + 90
  --gid-index <idx>          RoCE GID index. Default: -1
  --gpu-pair-starts <csv>    XOR GPU pair starts to try. Default: 6,4,2,0
  --max-used-mb <MiB>        Treat a GPU as busy above this memory. Default: 512
  --dynamic-pipeline-rdma <0|1>
                              Enable pipelined compression/D2H/RDMA for dynamic AEE/MEE. Default: 0
  --pipeline-chunk-mb <n>     Pipelined RDMA tile size per rail in MiB. Default: 8
  --pipeline-inflight <n>     Pipelined RDMA in-flight tiles per rail. Default: 2
  --conda-env <name>         Conda env activated inside qsub jobs. Default: llm
  --cuda-home <path>         CUDA toolkit root. Default: /share/apps/cuda-11.8
  --gcc-home <path>          GCC root. Default: /share/apps/gcc-8.3
  --cuda-arch-list <list>    TORCH_CUDA_ARCH_LIST. Default: 8.0;8.6;8.9
  --pe-gpu <n>               qsub -pe gpu value. Default: 2, matching requested qrsh flags
  --project <name>           SGE project. Default: cmic_hpc
  --h-rt <seconds>           SGE runtime limit. Default: 60000
  --tmem <amount>            SGE memory request. Default: 1K
  --sync                     Add qsub -sync y to the final node submission
  --dry-run                  Print qsub commands without submitting
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes)
      NODES_RAW="$2"; shift 2 ;;
    --result-root|--result_root)
      RESULT_ROOT="$2"; shift 2 ;;
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
    --base-port)
      BASE_PORT="$2"; shift 2 ;;
    --master-port)
      MASTER_PORT="$2"; shift 2 ;;
    --gid-index)
      GID_INDEX="$2"; shift 2 ;;
    --gpu-pair-starts)
      GPU_PAIR_STARTS="$2"; shift 2 ;;
    --max-used-mb)
      MAX_USED_MB="$2"; shift 2 ;;
    --dynamic-pipeline-rdma)
      DYNAMIC_AEE_PIPELINE_RDMA="$2"; shift 2 ;;
    --pipeline-chunk-mb)
      RING_RDMA_PIPELINE_CHUNK_MB="$2"; shift 2 ;;
    --pipeline-inflight)
      RING_RDMA_PIPELINE_INFLIGHT="$2"; shift 2 ;;
    --conda-env)
      CONDA_ENV="$2"; shift 2 ;;
    --cuda-home)
      CUDA_HOME_DIR="$2"; shift 2 ;;
    --gcc-home)
      GCC_HOME="$2"; shift 2 ;;
    --cuda-arch-list)
      CUDA_ARCH_LIST="$2"; shift 2 ;;
    --pe-gpu)
      PE_GPU="$2"; shift 2 ;;
    --project)
      PROJECT="$2"; shift 2 ;;
    --h-rt)
      H_RT="$2"; shift 2 ;;
    --tmem)
      TMEM="$2"; shift 2 ;;
    --sync)
      SYNC_FLAG="-sync y"; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1 ;;
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
  if [[ ! "$node" =~ '^chip-207-[0-9]+$' ]]; then
    echo "Unsupported node '$node'. Expected chip-207-*." >&2
    exit 1
  fi
  if [[ -n "${seen_nodes[$node]:-}" ]]; then
    echo "Duplicate node '$node' in --nodes" >&2
    exit 1
  fi
  seen_nodes[$node]=1
done

if [[ "$RAILS" != "1" && "$RAILS" != "2" ]]; then
  echo "--rails must be 1 or 2" >&2
  exit 1
fi

if [[ "$RAILS" == "2" && -z "$IFACE1" ]]; then
  echo "--iface1 is required with --rails 2" >&2
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
  echo "master_port=$MASTER_PORT"
  echo "base_port=$BASE_PORT"
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
  echo "gpu_pair_starts=$GPU_PAIR_STARTS"
  echo "max_used_mb=$MAX_USED_MB"
  echo "dynamic_aee_pipeline_rdma=$DYNAMIC_AEE_PIPELINE_RDMA"
  echo "ring_rdma_pipeline_chunk_mb=$RING_RDMA_PIPELINE_CHUNK_MB"
  echo "ring_rdma_pipeline_inflight=$RING_RDMA_PIPELINE_INFLIGHT"
  echo "conda_env=$CONDA_ENV"
  echo "cuda_home=$CUDA_HOME_DIR"
  echo "gcc_home=$GCC_HOME"
  echo "cuda_arch_list=$CUDA_ARCH_LIST"
  echo "project=$PROJECT"
  echo "h_rt=$H_RT"
  echo "tmem=$TMEM"
  echo "pe_gpu=$PE_GPU"
  echo "build_command=$RUN_DIR/build_eden_utils_command.zsh"
} > "$RUN_DIR/submit_metadata.log"

submit_one() {
  local node_index="$1"
  local node="$2"
  local sync_this="$3"
  local name="hooksmk_${RUN_ID}_${node_index}"
  local qsub_args

  qsub_args=(
    qsub
    -cwd
    -S /bin/zsh
    -N "$name"
    -P "$PROJECT"
    -R y
    -l "gpu=true,hostname=($node),tmem=$TMEM,h_rt=$H_RT"
    -pe gpu "$PE_GPU"
    -o "$RUN_DIR/node_${node_index}_${node}.qsub.out"
    -e "$RUN_DIR/node_${node_index}_${node}.qsub.err"
  )

  if [[ -n "$sync_this" ]]; then
    qsub_args+=(${=sync_this})
  fi

  qsub_args+=(
    "$NODE_SCRIPT"
    --repo-dir "$REPO_DIR"
    --run-dir "$RUN_DIR"
    --node-index "$node_index"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    --base-port "$BASE_PORT"
    --num-nodes "$NUM_NODES"
    --procs-per-node "$PROCS_PER_NODE"
    --aggregation-method "$AGGREGATION_METHOD"
    --dtype "$DTYPE"
    --steps "$STEPS"
    --numel "$NUMEL"
    --bucket-cap-mb "$BUCKET_CAP_MB"
    --checksum-atol "$CHECKSUM_ATOL"
    --rails "$RAILS"
    --iface0 "$IFACE0"
    --iface1 "$IFACE1"
    --gid-index "$GID_INDEX"
    --gpu-pair-starts "$GPU_PAIR_STARTS"
    --max-used-mb "$MAX_USED_MB"
    --dynamic-pipeline-rdma "$DYNAMIC_AEE_PIPELINE_RDMA"
    --pipeline-chunk-mb "$RING_RDMA_PIPELINE_CHUNK_MB"
    --pipeline-inflight "$RING_RDMA_PIPELINE_INFLIGHT"
    --nsys-profile-step "$NSYS_PROFILE_STEP"
    --nsys-trace "$NSYS_TRACE"
    --conda-env "$CONDA_ENV"
    --cuda-home "$CUDA_HOME_DIR"
    --gcc-home "$GCC_HOME"
    --cuda-arch-list "$CUDA_ARCH_LIST"
  )

  if (( SKIP_GRAD_CHECK )); then
    qsub_args+=(--skip-grad-check)
  fi
  if (( EXPECT_BUTTERFLY_RDMA )); then
    qsub_args+=(--expect-butterfly-rdma)
  fi

  print -r -- "${(q)qsub_args[@]}" >> "$RUN_DIR/qsub_commands.log"
  if (( DRY_RUN )); then
    print -r -- "${(q)qsub_args[@]}"
  else
    "${qsub_args[@]}"
  fi
}

for (( idx = 0; idx < NUM_NODES; idx++ )); do
  node="${nodes[$((idx + 1))]}"
  sync_this=""
  if [[ -n "$SYNC_FLAG" && "$idx" == "$((NUM_NODES - 1))" ]]; then
    sync_this="$SYNC_FLAG"
  fi
  submit_one "$idx" "$node" "$sync_this"
done

echo "Run directory: $RUN_DIR"
echo "Build command: $RUN_DIR/build_eden_utils_command.zsh"
