# Refactored DDP Communication Hooks

This directory contains the refactored PyTorch DDP communication hooks used by
the testbed smoke jobs.  The custom RDMA work focuses on ring all-reduce for the
dynamiQ AEE/MEE hook, plus a simple butterfly all-reduce prototype.
The optimized pipelined GPU->CPU->RDMA path is still ring-only.

## dynamiQ Ring RDMA Path

The dynamiQ hook first mean-centers each 256-element superchunk,
reduces per-superchunk statistics, sorts superchunks by summed L2 norm, and then
splits them into 8-bit, 4-bit, and 2-bit groups.  Each selected group is passed
through the ring all-reduce callback independently.

Set `DYNAMIC_AEE_PIPELINE_RDMA=1` to use the pipelined RDMA callback:

```bash
DYNAMIC_AEE_PIPELINE_RDMA=1 \
RING_RDMA_RAILS=2 \
torchrun ...
```

The qsub smoke wrapper exposes the same switch:

```bash
./testbed_evaluation/submit_qsub_smoke_new_comm_hooks.zsh \
  --aggregation-method dynamiQ_mee_5bit \
  --dynamic-pipeline-rdma 1
```

The callback is `composable_ring_rdma_dynamiQ_allreduce_callback` in
`ring_rdma.py`.  It uses `PipelineRdmaSender` and `PipelineRdmaReceiver` from
`rdma_comm_compress/ring_allreduce_ext`.

## Native Pipeline Classes

The regular `RdmaSender` and `RdmaReceiver` remain available.  The pipelined
dynamiQ path uses separate native classes:

- `PipelineRdmaSender.send_compress_bf16`: compresses BF16 input tile-by-tile,
  copies each compressed tile D2H, then posts the RDMA write.
- `PipelineRdmaSender.send_dec_comp_bf16`: runs fused
  decompress+add+compress tile-by-tile before D2H/RDMA.
- `PipelineRdmaSender.send`: sends an already-compressed uint8 payload.
- `PipelineRdmaReceiver.recv`: receives compressed uint8 payloads.
- `PipelineRdmaReceiver.recv_decompress_bf16`: receives compressed payloads and
  decompresses each tile into BF16 output during H2D progress.
- `PipelineRdmaReceiver.recv_dec_comp_bf16`: receives compressed payloads and
  runs fused decompress+add+compress during H2D progress, writing the next
  compressed send buffer.

The last method is the receive-side lookahead path.  It fixes the bottleneck on
ranks whose right edge is local and left edge is remote: those ranks previously
performed `dec_compression` as a separate step before local P2P.  They now
receive the remote compressed tile and produce the next compressed send tile in
the same pipelined receive loop.

## Ring Scheduling

The code assumes the two-GPUs-per-node rank layout:

```text
rank0, rank1 on node0
rank2, rank3 on node1
...
```

Ranks with a remote right edge use the sender-side compression pipeline.  Ranks
with a remote left edge and local right edge use the receiver-side lookahead
pipeline with an extra compressed buffer named `pipeline_next_send_chunk`.
Local P2P remains handled by torch distributed P2P, not by the RDMA pipeline.

## Quantization Strategy Values

The dynamic AEE/MEE compressor objects pass their strategy value into the native
pipeline at runtime:

- AEE uses `strategy = 2`.
- MEE uses `strategy = 3`.

The pipeline validates that only strategies 2 and 3 are used.

## Tile And Buffer Configuration

There is one tile-size knob for the pipelined remote RDMA path:

```bash
RING_RDMA_PIPELINE_CHUNK_MB=8
```

This is the per-rail pipeline tile limit in MiB.  The same tile boundary is used
for compression/decompression work, D2H/H2D copies, and RDMA writes.  Tiles are
rounded down to compressed superchunk-record boundaries so the native kernels
never split a compressed superchunk record.

There is currently no separate compression-only tile-size knob.  Compression is
scheduled on the communication tile.  The quantization chunk size is separate:
dynamic AEE/MEE currently uses `agg_chunk_size = 16`, and one superchunk is
`16 * 16 = 256` elements.

The number of in-flight remote tiles per rail is controlled by:

```bash
RING_RDMA_PIPELINE_INFLIGHT=2
```

Increasing this may improve overlap, but it also increases pinned host buffer
pressure and outstanding work per rail.

Local P2P has a separate chunking knob:

```bash
RING_LOCAL_P2P_CHUNK_MB=8
```

This only affects local torch P2P chunking.  It does not change RDMA pipeline
tile size or compression kernel tile size.

The qsub smoke wrapper exposes the RDMA pipeline knobs:

```bash
./testbed_evaluation/submit_qsub_smoke_new_comm_hooks.zsh \
  --dynamic-pipeline-rdma 1 \
  --pipeline-chunk-mb 8 \
  --pipeline-inflight 2
```

## Butterfly RDMA Prototype

Butterfly all-reduce now has a simple RDMA callback:
`composable_butterfly_rdma_allreduce_callback` in `butterfly_rdma.py`.  It keeps
the original `calc_order` butterfly schedule from `utils.py`, but routes each
peer exchange by topology:

- inter-node edges use `RdmaSender` / `RdmaReceiver` from
  `rdma_comm_compress/ring_allreduce_ext`, i.e. GPU->CPU->RDMA->CPU->GPU.
- intra-node edges use torch distributed P2P over the NCCL process group, so the
  two local GPUs communicate over NVLink/NV4.
- compression, decompression, and decompression-add remain explicit CUDA kernel
  calls around communication.  There is no dynamic pipeline, lookahead, or fused
  RDMA compression in this prototype.

Supported testbed layouts are two GPUs per node with `WORLD_SIZE=4` or
`WORLD_SIZE=8`:

```text
rank0, rank1 on node0
rank2, rank3 on node1
rank4, rank5 on node2
rank6, rank7 on node3
```

Set an aggregation method containing `butterfly`, for example:

```bash
./testbed_evaluation/launch_butterfly_rdma_smoke.zsh
```

The launch wrapper defaults to a two-node, 1M-element
`dynamiQ_aee_5bit_butterfly` smoke and passes `--expect-butterfly-rdma` so
the smoke test fails if the RDMA butterfly manager is not initialized.  You can
override the common knobs with environment variables:

```bash
NODES=chip-207-3,chip-207-4 \
NUMEL=67108864 \
STEPS=10 \
RAILS=2 \
./testbed_evaluation/launch_butterfly_rdma_smoke.zsh
```

Relevant butterfly-specific knobs:

- `BUTTERFLY_RDMA_DISABLE=1`: fall back to the legacy torch-P2P butterfly
  callback.
- `BUTTERFLY_RDMA_BASE_PORT`: override the butterfly RDMA base port.  By
  default, it uses `RING_RDMA_BASE_PORT + 4000`.
- `BUTTERFLY_RDMA_EDGE_PORT_STRIDE=20`: spacing between directed peer-edge
  ports.  Keep this at least 20 when using two rails because rail 1 uses
  `port + 10`.
- `BUTTERFLY_LOCAL_P2P_CHUNK_MB=8`: local NCCL P2P chunk size.

## OmniReduce Top-K Prototype

The testbed includes `P2P_omnireduce_topk_hook`, ported from the simulation
`distributed_omnireduce_topk_hook`.  It chunks each bucket into 256-element rows,
selects locally large rows, reduces the row-selection bitmap across ranks, and
then reduces only the selected dense rows.

The simulation uses `dist.all_reduce` for the selected dense rows.  The testbed
hook routes that main all-reduce phase through the RDMA topology callback:

- `omnireduce`: selected rows use ring RDMA.
- `omnireduce_butterfly`: selected rows use butterfly RDMA.

The bitmap all-reduce stays as a normal distributed collective because it is
small control metadata.  The selected dense rows are reduced as FP16 payloads,
matching the simulation compressor buffer.

Smoke launch examples:

```bash
./testbed_evaluation/launch_omni_rdma_smoke.zsh
```

```bash
TOPOLOGY=butterfly \
RAILS=2 \
NUMEL=67108864 \
STEPS=10 \
./testbed_evaluation/launch_omni_rdma_smoke.zsh
```

## THC oldINCA Prototype

The testbed also includes `P2P_THC_compress_hook`, based on the simulation
`thc_hooks.py` oldINCA path.  Each partition is randomly Hadamard transformed,
quantized with the oldINCA coordinate range, directly summed as BF16 bin values
through the selected RDMA topology, then dequantized and inverse transformed.

Unlike the simulation hook, the testbed RHT/IRHT uses
`new_comm_hooks.eden_utils.Hadamard`, built from
`cuda_kernels/src/srrcomp/eden_utils`.  The bucket partition threshold follows
the testbed hooks (`1 << 30`).  The oldINCA table metadata is read from
`simulations_llm/compression/new_tables`; set `THC_TABLE_DIR` to override it.

Smoke launch examples:

```bash
./testbed_evaluation/launch_thc_rdma_smoke.zsh
```

```bash
TOPOLOGY=butterfly \
RAILS=2 \
NUMEL=1048576 \
STEPS=5 \
./testbed_evaluation/launch_thc_rdma_smoke.zsh
```

## Extension Loading

`rdma_comm_compress/ring_allreduce_ext/__init__.py` now has a fast path that
loads the prebuilt `ring_allreduce_native.so` directly when it is newer than its
sources.  This avoids PyTorch JIT extension lock stalls in qsub runs.

Relevant knobs:

- `RING_EXT_FORCE_JIT=1` forces `torch.utils.cpp_extension.load`.
- `RING_EXT_STALE_LOCK_SECONDS=60` controls stale lock cleanup age.

## Smoke Tests Run

The following smoke tests passed on two nodes (`chip-207-3`, `chip-207-4`) with
two ranks per node and two RDMA rails:

- `dynamiQ_mee_5bit`, 1M elements, checksum delta 0.
- `dynamiQ_mee_5bit`, 64M elements, checksum delta 0.
- `dynamiQ_mee_5bit`, 1B elements, 25 steps, skip grad check.

The 1B timing run reported about `0.1870 s/step` over 22 measured custom-ring
steps:

```text
testbed_evaluation/smoke_job_results/20260611_222809_manual_11467
```
