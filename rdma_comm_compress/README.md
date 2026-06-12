# Fast BF16 And Quantized Ring All-Reduce

This directory contains a custom ring all-reduce prototype for the chip-207 GPU
testbed. The implementation is intentionally focused: two GPUs per node,
standard ring order, BF16 tensors, and either raw BF16 or hierarchical MEE
quantized payloads. Inter-node edges reuse the high-bandwidth staged
GPU->CPU->RDMA->CPU->GPU transport from
`ring_allreduce_ext`; intra-node edges use NCCL
P2P over NVLink.

## Files

- `sample_allreduce.py`: Python driver for the standard ring all-reduce. It
  supports raw BF16 payloads and hierarchical MEE quantized payloads.
- `ring_allreduce_ext/`: native CUDA/C++/verbs extension used by the Python
  driver.
- `submit_qsub_ring_allreduce.zsh`: top-level SGE/qsub submitter.
- `qsub_ring_allreduce_node.zsh`: per-node qsub script. It selects an idle
  XOR GPU pair, sets NCCL/RDMA environment variables, and starts two ranks.
- `job_results_dump/`: per-run logs, metadata, qsub commands, rank stdout,
  rank stderr, and exit codes.

## Rank And Topology Model

For `k` nodes, the implementation runs `2*k` ranks:

```text
rank0 = node0 GPU0
rank1 = node0 GPU1
rank2 = node1 GPU0
rank3 = node1 GPU1
...
rank(2*k-2) = node(k-1) GPU0
rank(2*k-1) = node(k-1) GPU1
```

The ring order is the standard rank order:

```text
0 -> 1 -> 2 -> 3 -> ... -> WORLD_SIZE-1 -> 0
```

This means every node has one local edge and one or two remote ring edges:

- Local edge inside a node: `GPU0 <-> GPU1`, handled by NCCL P2P.
- Remote edge between nodes: `node n GPU1 -> node n+1 GPU0`, handled by the
  custom staged RDMA extension. The final node wraps to `node0 GPU0`.

The qsub path enforces that each node exposes exactly one XOR GPU pair through
`CUDA_VISIBLE_DEVICES`, such as `0,1`, `2,3`, `4,5`, or `6,7`. The Python driver
then verifies the pair with `nvidia-smi topo -m` and requires the reported link
to start with `NV`. On the current fast runs, `GPU6 <-> GPU7` reports `NV4`.

## Communication Path

The all-reduce is a standard ring all-reduce:

1. Reduce-scatter for `WORLD_SIZE - 1` steps.
2. All-gather for `WORLD_SIZE - 1` steps.

Each rank splits its BF16 tensor into `WORLD_SIZE` equal chunks. In the
reduce-scatter phase, a received chunk is accumulated in place with the native
`bf16_add_` CUDA kernel. The kernel performs the addition in float and stores
back to BF16.

In `mode=quantized`, each ring chunk is still logically BF16, but the remote
payload is compressed with the hierarchical MEE CUDA kernels from
`/cluster/project2/gcreduce_data/dynamiq_artifact/cuda_kernels/src/srrcomp/eden_utils/`.
For `nbits=4`, the compressed RDMA message is about `3.51x` smaller than the
raw BF16 chunk. The reduce-scatter path uses fused decompress/add/compress
where possible, and the all-gather path forwards compressed payloads.

For each ring step, the exchange path chooses the edge type:

- If the neighbor is on the same node, `torch.distributed` NCCL P2P sends or
  receives the BF16 chunk directly between the two GPUs.
- If the neighbor is on another node, the native extension performs staged RDMA:

```text
source GPU -> pinned host ring buffer -> RDMA write with immediate
           -> peer pinned host ring buffer -> destination GPU
```

The RDMA extension uses:

- `8 MiB` pipeline chunks.
- Up to `64` pinned host slots per rail. For small messages, the slot count is
  capped at the number of chunks in that rail to avoid wasting CUDA streams and
  pinned host memory.
- One or two RDMA rails.
- With two rails, the message bytes are split across the two rail spans and the
  rail ports are `base_port + rail * 10`.
- Nonblocking CUDA streams and CUDA events for D2H/H2D progress.
- Receiver credits returned only after the H2D copy has completed.
- A no-progress timeout controlled by `RING_RDMA_TIMEOUT_SECONDS`, default
  `300` seconds.
- Optional RDMA debug logging through `RING_RDMA_DEBUG=1`.

The sender now waits for all of the following before `send()` returns:

- All D2H copies have completed.
- All RDMA write completions have been observed.
- All receiver credits have returned.

The receiver waits for:

- All RDMA write notifications.
- All H2D copies.
- All credit send completions.

This strict message boundary is important. Earlier versions returned from
`send()` after only local RDMA write completions, which allowed the next ring
message to start before the receiver had fully completed the previous `recv()`.
Because the receiver-side immediate data identifies the slot/chunk but not a
high-level Python ring step, that caused hangs or low, unstable bandwidth.

## Dual-Rail RDMA

The ring all-reduce can use both NIC rails for cross-node edges:

```bash
--rails 2 --iface0 eth0 --iface1 eth1
```

Each rank gathers both interface IPs during setup. A remote receiver listens on
both rail addresses, and a remote sender connects to the matching peer rail
addresses. The native transport creates a separate RDMA CM/QP/CQ, pinned host
ring buffer, CUDA stream pool, and credit loop for each rail. The payload split
is visible in the topology line:

```text
ring_topology ... mode=quantized rails=2 ... message_bytes=149504
rail_message_bytes=74752,74752 ...
```

For large quantized runs, `rails=2` has measured around `60-65Gbps` reported
single-direction remote-edge bandwidth, or `120-130Gbps` bidirectional per-node
remote bandwidth. This is the compressed-wire remote-edge bandwidth printed by
`sample_allreduce.py`; the logical BF16 input bandwidth can be higher because
the payload is compressed before crossing RDMA.

## Local P2P Overlap

Local NCCL P2P is overlapped with the remote staged RDMA path by default.

In `sample_allreduce.py`, each exchange records a CUDA event for the remote send
buffer before posting local NCCL P2P. The remote send worker waits on that event
and can then progress while local NCCL P2P is still running. This avoids
serializing:

```text
local P2P, then RDMA
```

and instead allows:

```text
local P2P || RDMA
```

The debug fallback is:

```bash
--no-overlap-local-p2p
```

When overlap is disabled, the code pipelines local NCCL P2P in
`--local-p2p-chunk-mb` slices. The default slice size is `8 MiB`.

## NVLink Enforcement

The qsub node script sets:

```bash
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
```

The Python driver independently checks the physical pair with:

```bash
nvidia-smi topo -m
```

The expected startup line looks like:

```text
ring_topology world=4 nodes=2 mode=bf16 rails=1 cuda_visible_devices=6,7
nvlink=NV4 ... overlap_local_p2p=True
```

NCCL logs may also show:

```text
NCCL_P2P_LEVEL set by environment to NVL
Channel 00/0 : 0[6] -> 1[7] via P2P/CUMEM
```

`via P2P/CUMEM` means NCCL selected CUDA peer-to-peer transport using CUDA
memory handles. By itself it does not spell out the physical link. In this
setup it corresponds to NVLink because `NCCL_P2P_LEVEL=NVL` is set and the
topology guard has already verified `GPU6 <-> GPU7 = NV4`.

## Launch Commands

Use `chip-207-3` and `chip-207-4` for the current fast two-node runs, or include chip-207-5,chip-207-6 for the fast 4-node runs. Avoid
`chip-207-1` and `chip-207-2`; it has shown much lower bandwidth.

Raw BF16 correctness smoke test:

```bash
cd /cluster/project2/gcreduce_data/dynamiq_artifact/rdma_comm_compress

./submit_qsub_ring_allreduce.zsh \
  --nodes chip-207-3,chip-207-4 \
  --numel 67108864 \
  --iters 1 \
  --warmup-iters 1 \
  --verify \
  --sync
```

Raw BF16 single-rail performance run:

```bash
cd /cluster/project2/gcreduce_data/dynamiq_artifact/rdma_comm_compress

./submit_qsub_ring_allreduce.zsh \
  --nodes chip-207-3,chip-207-4,chip-207-5,chip-207-6 \
  --numel 268435456 \
  --iters 20 \
  --warmup-iters 3 \
  --sync
```

## Current Limitations

- Input tensors are BF16 only.
- Payload modes are raw BF16 and hierarchical MEE quantized BF16.
- Quantized mode currently supports `nbits=2`, `4`, or `8`; the tested fast
  path is `nbits=4`.
- RDMA supports one or two rails.
- Exactly two ranks per node.
- GPU pair must be an XOR pair and must be NVLink-connected.
- The standard ring pattern is fixed; no tree, hierarchical, or reordered ring
  pattern is implemented.
- Cross-node all-reduce data uses the custom staged RDMA extension. NCCL is used
  for local P2P, process-group support, timing reductions, and optional
  correctness verification.
- `--verify` is most meaningful for raw BF16. Quantized mode is lossy, so use
  bandwidth runs and separate numerical-quality checks when evaluating the
  compression strategy.
- The qsub script allows supported chip-207 hosts, but current fast defaults
  avoid `chip-207-1` and `chip-207-2`.

## Troubleshooting

If the job hangs or exits with a timeout:

- Check `rank_*.out` for the last `ring_progress` marker. This usually tells
  whether the job is stuck in setup, warmup, the timed loop, or verification.
- Set `RING_RDMA_DEBUG=1` to print sender/receiver RDMA progress messages.
- Adjust `RING_RDMA_TIMEOUT_SECONDS` if a long debug run needs more time.
- Confirm all `rank_*.exitcode` files are `0`.
- Confirm the startup line reports `nvlink=NV...` and
  `overlap_local_p2p=True`.
- For two-rail runs, confirm the startup line reports `rails=2` and
  `rail_message_bytes=...,...`.
- Confirm `node_*_metadata.log` selected `cuda_visible_devices=6,7` or another
  valid XOR pair.
- For two-rail runs, confirm `node_*_metadata.log` has `iface0=eth0` and
  `iface1=eth1`.

If NCCL prints `via P2P/CUMEM`, combine it with the topology line:

```text
ring_topology ... nvlink=NV4 ...
NCCL_P2P_LEVEL set by environment to NVL
... via P2P/CUMEM
```

Together these indicate that the local NCCL P2P transport is constrained to the
NVLink-connected GPU pair.
