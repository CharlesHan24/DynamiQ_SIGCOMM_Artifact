#!/usr/bin/env zsh
unsetopt errexit nounset
source ~/.zshrc
setopt errexit nounset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NODE_SCRIPT="$SCRIPT_DIR/qsub_two_node_allreduce_node.zsh"
RESULT_ROOT="$SCRIPT_DIR/job_results_dump"

NODES_RAW="chip-207-3,chip-207-6"
PROJECT="cmic_hpc"
H_RT=60000
TMEM="8G"
NUMEL=$((8 * 1024 * 1024))
ITERS=20
WARMUP_ITERS=3
MODE="bf16"
NBITS=4
RAILS=2
IFACE0="eth0"
IFACE1="eth1"
BASE_PORT=""
MASTER_PORT=""
GPU_IDS="7,6,5,4,3,2,1,0"
MAX_USED_MB=512
NUMA_NODE=1
GID_INDEX=-1
VERIFY_FLAG=""
SYNC_FLAG=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./submit_qsub_two_node_allreduce.zsh [options]

Options:
  --nodes <csv>            Exactly two hosts from chip-207-1,2,3,4,6.
                           Default: chip-207-3,chip-207-6
  --numel <count>          BF16 elements per rank. Default: 8388608
  --iters <count>          Timed iterations. Default: 20
  --warmup-iters <count>   Warmup iterations. Default: 3
  --mode <quantized|bf16>  Ring payload mode. Default: bf16
  --nbits <2|4|8>          MEE bitwidth for quantized mode. Default: 4
  --rails <1|2>            RDMA rails for the ring path. Default: 2
  --iface0 <name>          First NIC interface. Default: eth0
  --iface1 <name>          Second NIC interface. Default: eth1
  --base-port <port>       RDMA base port. Default: derived from timestamp
  --master-port <port>     torch.distributed master port. Default: base-port + 90
  --gpu-ids <csv>          GPU preference order. Default: 7,6,5,4,3,2,1,0
  --max-used-mb <MiB>      Treat a GPU as busy above this memory. Default: 512
  --numa-node <node|none>  Run under numactl. Default: 1
  --gid-index <idx>        Optional RoCE GID index. Default: -1
  --project <name>         SGE project. Default: cmic_hpc
  --h-rt <seconds>         SGE runtime limit. Default: 60000
  --tmem <amount>          SGE memory request. Default: 8G
  --verify                 Enable correctness checks in both NCCL and ring runs
  --sync                   Submit the second qsub with -sync y
  --dry-run                Print qsub commands without submitting
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
if (( ${#nodes[@]} != 2 )); then
  echo "--nodes must contain exactly two hosts" >&2
  exit 1
fi

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
  echo "--rails must be 1 or 2" >&2
  exit 1
fi

if [[ "$NBITS" != "2" && "$NBITS" != "4" && "$NBITS" != "8" ]]; then
  echo "--nbits must be 2, 4, or 8" >&2
  exit 1
fi

if [[ "$MODE" != "quantized" && "$MODE" != "bf16" ]]; then
  echo "--mode must be quantized or bf16" >&2
  exit 1
fi

if (( NUMEL % 2 != 0 )); then
  echo "--numel must be divisible by 2" >&2
  exit 1
fi

if [[ "$MODE" == "quantized" ]] && (( NUMEL % (2 * 256) != 0 )); then
  echo "--numel must be divisible by 2 * 256 for quantized mode" >&2
  exit 1
fi

if [[ -z "$BASE_PORT" ]]; then
  BASE_PORT=$((18000 + ($(date +%s) % 300) * 100))
fi
if [[ -z "$MASTER_PORT" ]]; then
  MASTER_PORT=$((BASE_PORT + 90))
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_two_node_${JOB_ID:-manual}_$$"
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
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "base_port=$BASE_PORT"
  echo "numel=$NUMEL"
  echo "iters=$ITERS"
  echo "warmup_iters=$WARMUP_ITERS"
  echo "mode=$MODE"
  echo "nbits=$NBITS"
  echo "rails=$RAILS"
  echo "iface0=$IFACE0"
  echo "iface1=$IFACE1"
  echo "gpu_ids=$GPU_IDS"
  echo "max_used_mb=$MAX_USED_MB"
  echo "numa_node=$NUMA_NODE"
  echo "gid_index=$GID_INDEX"
  echo "project=$PROJECT"
  echo "h_rt=$H_RT"
  echo "tmem=$TMEM"
} > "$RUN_DIR/submit_metadata.log"

submit_one() {
  local node_index="$1"
  local node="$2"
  local sync_this="$3"
  local name="twoar_${RUN_ID}_${node_index}"
  local qsub_args
  qsub_args=(
    qsub
    -cwd
    -S /bin/zsh
    -N "$name"
    -P "$PROJECT"
    -R y
    -l "gpu=true"
    -pe gpu 1
    -l "hostname=$node"
    -l "tmem=$TMEM"
    -l "h_rt=$H_RT"
    -o "$RUN_DIR/node_${node_index}_${node}.qsub.out"
    -e "$RUN_DIR/node_${node_index}_${node}.qsub.err"
  )

  if [[ -n "${RING_RDMA_DEBUG:-}" ]]; then
    qsub_args+=(-v "RING_RDMA_DEBUG=$RING_RDMA_DEBUG")
  fi

  if [[ -n "$sync_this" ]]; then
    qsub_args+=(${=sync_this})
  fi

  qsub_args+=(
    "$NODE_SCRIPT"
    --repo-dir "$SCRIPT_DIR"
    --run-dir "$RUN_DIR"
    --node-index "$node_index"
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
    --gpu-ids "$GPU_IDS"
    --max-used-mb "$MAX_USED_MB"
    --numa-node "$NUMA_NODE"
    --gid-index "$GID_INDEX"
  )

  if [[ "$RAILS" == "2" ]]; then
    qsub_args+=(--iface1 "$IFACE1")
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

for idx in 0 1; do
  node="${nodes[$((idx + 1))]}"
  sync_this=""
  if [[ -n "$SYNC_FLAG" && "$idx" == "1" ]]; then
    sync_this="$SYNC_FLAG"
  fi
  submit_one "$idx" "$node" "$sync_this"
done

echo "Run directory: $RUN_DIR"
