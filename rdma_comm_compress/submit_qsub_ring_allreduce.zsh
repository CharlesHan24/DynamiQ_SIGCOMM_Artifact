#!/usr/bin/env zsh
unsetopt errexit nounset
source ~/.zshrc
setopt errexit nounset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NODE_SCRIPT="$SCRIPT_DIR/qsub_ring_allreduce_node.zsh"
RESULT_ROOT="$SCRIPT_DIR/job_results_dump"

NODES_RAW="chip-207-3,chip-207-4,chip-207-5,chip-207-6"
PROJECT="cmic_hpc"
H_RT=60000
TMEM="8G"
NUMEL=$((8 * 1024 * 1024))
ITERS=20
WARMUP_ITERS=3
NBITS=4
MODE="bf16"
NSYS_ENABLED=0
NSYS_RANKS="1"
NSYS_TRACE="cuda,nvtx,osrt"
RAILS=1
IFACE0="eth0"
IFACE1="eth1"
BASE_PORT=""
MASTER_PORT=""
GPU_PAIR_STARTS="6,4,2,0"
MAX_USED_MB=512
NUMA_NODE=1
GID_INDEX=-1
VERIFY_FLAG=""
SYNC_FLAG=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./submit_qsub_ring_allreduce.zsh [options]

Options:
  --nodes <csv>              Hosts from chip-207-1,2,3,4,6.
                             Default: chip-207-3,chip-207-4
  --numel <count>            BF16 elements per rank. Default: 8388608
  --iters <count>            Timed iterations. Default: 20
  --warmup-iters <count>     Warmup iterations. Default: 3
  --mode <bf16|quantized>    Payload mode. Default: bf16
  --nsys                     Run selected ranks under /usr/local/cuda/bin/nsys
  --nsys-ranks <csv|all>     Global ranks to profile. Default: 1
  --nsys-trace <csv>         Nsight Systems trace set. Default: cuda,nvtx
  --nbits <2|4|8>            MEE bitwidth for quantized mode. Default: 4
  --rails <1|2>              RDMA rails. Default: 1
  --iface0 <name>            First NIC interface. Default: eth0
  --iface1 <name>            Second NIC interface. Default: eth1
  --base-port <port>         RDMA base port. Default: derived from timestamp
  --master-port <port>       torch.distributed master port. Default: base-port + 90
  --gpu-pair-starts <csv>    Pair starts tried on each node. Default: 6,4,2,0
  --max-used-mb <MiB>        Treat a GPU as busy above this memory. Default: 512
  --numa-node <node|none>    Run ranks under numactl. Default: 1
  --gid-index <idx>          Optional RoCE GID index. Default: -1
  --project <name>           SGE project. Default: cmic_hpc
  --h-rt <seconds>           SGE runtime limit. Default: 60000
  --tmem <amount>            SGE memory request. Default: 8G
  --verify                   Enable cross-rank result identity check
  --sync                     Submit with qsub -sync y
  --dry-run                  Print qsub commands without submitting
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes)
      NODES_RAW="$2"
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
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
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
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --h-rt)
      H_RT="$2"
      shift 2
      ;;
    --tmem)
      TMEM="$2"
      shift 2
      ;;
    --verify)
      VERIFY_FLAG="--verify"
      shift
      ;;
    --sync)
      SYNC_FLAG="-sync y"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

nodes=(${(s:,:)NODES_RAW})
if (( ${#nodes[@]} < 1 )); then
  echo "--nodes must contain at least one host" >&2
  exit 1
fi
WORLD_SIZE=$(( ${#nodes[@]} * 2 ))

typeset -A seen_nodes
for node in "${nodes[@]}"; do
  node="${node//[[:space:]]/}"
  if [[ ! "$node" =~ '^chip-207-(1|2|3|4|6)$' ]]; then
    echo "Unsupported node '$node'. Allowed: chip-207-1, chip-207-2, chip-207-3, chip-207-4, chip-207-6" >&2
    exit 1
  fi
  if [[ -n "${seen_nodes[$node]:-}" ]]; then
    echo "Duplicate node '$node' in --nodes" >&2
    exit 1
  fi
  seen_nodes[$node]=1
done

if [[ "$RAILS" != "1" && "$RAILS" != "2" ]]; then
  echo "ring all-reduce currently supports only --rails 1 or --rails 2" >&2
  exit 1
fi

if [[ "$RAILS" == "2" && -z "$IFACE1" ]]; then
  echo "--iface1 is required with --rails 2" >&2
  exit 1
fi

if [[ "$NBITS" != "2" && "$NBITS" != "4" && "$NBITS" != "8" ]]; then
  echo "--nbits must be 2, 4, or 8" >&2
  exit 1
fi

if [[ "$MODE" != "bf16" && "$MODE" != "quantized" ]]; then
  echo "ring all-reduce currently supports only --mode bf16 or --mode quantized" >&2
  exit 1
fi

if (( NUMEL % WORLD_SIZE != 0 )); then
  echo "--numel must be divisible by WORLD_SIZE=$WORLD_SIZE" >&2
  exit 1
fi

if [[ "$MODE" == "quantized" ]] && (( (NUMEL / WORLD_SIZE) % 256 != 0 )); then
  echo "--numel / WORLD_SIZE must be divisible by 256 for quantized mode" >&2
  exit 1
fi

if [[ -z "$BASE_PORT" ]]; then
  BASE_PORT=$((18000 + ($(date +%s) % 300) * 100))
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

{
  echo "date=$(date -Is)"
  echo "run_id=$RUN_ID"
  echo "nodes=${nodes[*]}"
  echo "world_size=$WORLD_SIZE"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "base_port=$BASE_PORT"
  echo "numel=$NUMEL"
  echo "iters=$ITERS"
  echo "warmup_iters=$WARMUP_ITERS"
  echo "mode=$MODE"
  echo "nsys_enabled=$NSYS_ENABLED"
  echo "nsys_ranks=$NSYS_RANKS"
  echo "nsys_trace=$NSYS_TRACE"
  echo "nbits=$NBITS"
  echo "rails=$RAILS"
  echo "iface0=$IFACE0"
  echo "iface1=$IFACE1"
  echo "gpu_pair_starts=$GPU_PAIR_STARTS"
  echo "max_used_mb=$MAX_USED_MB"
  echo "numa_node=$NUMA_NODE"
  echo "project=$PROJECT"
  echo "h_rt=$H_RT"
  echo "tmem=$TMEM"
} > "$RUN_DIR/submit_metadata.log"

submit_one() {
  local node_index="$1"
  local node="$2"
  local sync_this="$3"
  local rank_base=$(( node_index * 2 ))
  local name="ringar_${RUN_ID}_${node_index}"
  local qsub_args
  qsub_args=(
    qsub
    -cwd
    -S /bin/zsh
    -N "$name"
    -P "$PROJECT"
    -R y
    -l "gpu=true"
    -pe gpu 2
    -l "hostname=$node"
    -l "tmem=$TMEM"
    -l "h_rt=$H_RT"
    -o "$RUN_DIR/node_${node_index}_${node}.qsub.out"
    -e "$RUN_DIR/node_${node_index}_${node}.qsub.err"
  )

  if [[ -n "$sync_this" ]]; then
    qsub_args+=(${=sync_this})
  fi

  qsub_args+=(
    "$NODE_SCRIPT"
    --repo-dir "$SCRIPT_DIR"
    --run-dir "$RUN_DIR"
    --node-index "$node_index"
    --rank-base "$rank_base"
    --world-size "$WORLD_SIZE"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    --base-port "$BASE_PORT"
    --numel "$NUMEL"
    --iters "$ITERS"
    --warmup-iters "$WARMUP_ITERS"
    --mode "$MODE"
    --nbits "$NBITS"
    --rails "$RAILS"
    --iface0 "$IFACE0"
    --iface1 "$IFACE1"
    --gpu-pair-starts "$GPU_PAIR_STARTS"
    --max-used-mb "$MAX_USED_MB"
    --numa-node "$NUMA_NODE"
    --gid-index "$GID_INDEX"
  )

  if (( NSYS_ENABLED )); then
    qsub_args+=(
      --nsys
      --nsys-ranks "$NSYS_RANKS"
      --nsys-trace "$NSYS_TRACE"
    )
  fi

  if [[ -n "$VERIFY_FLAG" ]]; then
    qsub_args+=("$VERIFY_FLAG")
  fi

  print -r -- "${(q)qsub_args[@]}" >> "$RUN_DIR/qsub_commands.log"
  if (( DRY_RUN )); then
    print -r -- "${(q)qsub_args[@]}"
  else
    "${qsub_args[@]}"
  fi
}

last_idx=$(( ${#nodes[@]} - 1 ))
for (( idx = 0; idx <= last_idx; idx++ )); do
  node="${nodes[$((idx + 1))]}"
  sync_this=""
  if [[ -n "$SYNC_FLAG" && "$idx" == "$last_idx" ]]; then
    sync_this="$SYNC_FLAG"
  fi
  submit_one "$idx" "$node" "$sync_this"
done

echo "Run directory: $RUN_DIR"
