#!/usr/bin/env python3
import argparse
import os
import re
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import torch
import torch.distributed as dist

from ring_allreduce_ext import load_ring_allreduce_ext


BF16_BYTES = 2
DEFAULT_LOCAL_P2P_CHUNK_MB = 8
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MEE_ALIGNMENT_ELEMS = 16 * 16


@dataclass(frozen=True)
class RankMeta:
    rank: int
    hostname: str
    ip0: str
    ip1: str = ""


@contextmanager
def maybe_nvtx_range(enabled: bool, label: str):
    if enabled:
        torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def cuda_profiler_start(enabled: bool) -> None:
    if not enabled:
        return
    status = torch.cuda.cudart().cudaProfilerStart()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {status}")


def cuda_profiler_stop(enabled: bool) -> None:
    if not enabled:
        return
    status = torch.cuda.cudart().cudaProfilerStop()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStop failed with status {status}")


def concurrent_remote_exchange(
    remote_sender,
    send_buf: torch.Tensor,
    remote_receiver,
    recv_buf: torch.Tensor,
    send_ready_event: Optional[torch.cuda.Event] = None,
) -> None:
    errors = []

    def send_worker() -> None:
        try:
            if send_ready_event is not None:
                send_ready_event.synchronize()
            remote_sender.send(send_buf)
        except BaseException as exc:
            errors.append(exc)

    send_thread = threading.Thread(target=send_worker, name="rdma-send")
    send_thread.start()
    remote_receiver.recv(recv_buf)
    send_thread.join()
    if errors:
        raise errors[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BF16 standard ring all-reduce over local NCCL P2P and staged inter-node RDMA."
    )
    parser.add_argument(
        "--numel",
        type=int,
        default=8 * 1024 * 1024,
        help="Number of BF16 elements per rank. Must be divisible by WORLD_SIZE.",
    )
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations.")
    parser.add_argument("--warmup-iters", type=int, default=3, help="Warmup iterations.")
    parser.add_argument(
        "--mode",
        choices=("bf16", "quantized"),
        default="bf16",
        help="Payload mode. bf16 sends raw BF16 chunks; quantized sends hierarchical MEE-compressed BF16 chunks.",
    )
    parser.add_argument("--nbits", type=int, default=4, choices=(2, 4, 8), help="MEE bitwidth for --mode quantized.")
    parser.add_argument("--rails", type=int, default=1, choices=(1, 2), help="RDMA rail count.")
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=0,
        help="Optional WORLD_SIZE assertion. 0 means accept the environment value.",
    )
    parser.add_argument(
        "--iface0",
        default="eth0",
        help="RDMA/bootstrap interface used for inter-node receiver setup.",
    )
    parser.add_argument(
        "--iface1",
        default="",
        help="Second RDMA interface used when --rails 2.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=18740,
        help="Base RDMA port. Edge ending at node n uses base_port + n*100.",
    )
    parser.add_argument("--gid-index", type=int, default=-1, help="Optional RoCE GID index override.")
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=30,
        help="Default process group timeout.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare the final output with a NCCL all_reduce reference and check rank identity.",
    )
    parser.add_argument("--build-verbose", action="store_true", help="Print native extension build output.")
    parser.add_argument("--enable-nvtx", action="store_true", help="Annotate ring phases with NVTX ranges.")
    overlap_group = parser.add_mutually_exclusive_group()
    overlap_group.add_argument(
        "--overlap-local-p2p",
        dest="overlap_local_p2p",
        action="store_true",
        default=True,
        help="Overlap local NCCL P2P with RDMA. Enabled by default.",
    )
    overlap_group.add_argument(
        "--no-overlap-local-p2p",
        dest="overlap_local_p2p",
        action="store_false",
        help="Debug fallback: complete local NCCL P2P before entering the RDMA transfer.",
    )
    parser.add_argument(
        "--local-p2p-chunk-mb",
        type=int,
        default=DEFAULT_LOCAL_P2P_CHUNK_MB,
        help="Pipeline local NCCL P2P in this many MiB per slice. Default: 8.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Bracket the timed region with cudaProfilerStart/Stop for Nsight Systems capture-range mode.",
    )
    return parser.parse_args()


def env_int(name: str, default: Optional[int] = None) -> int:
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise RuntimeError(f"missing required environment variable: {name}")
        return default
    return int(value)


def get_interface_ipv4(ifname: str) -> str:
    cmd = f"ip -4 -o addr show dev {ifname} | awk '{{print $4}}' | cut -d/ -f1"
    output = subprocess.check_output(["bash", "-lc", cmd], text=True).strip()
    if not output:
        raise RuntimeError(f"could not determine IPv4 address for interface {ifname}")
    return output.splitlines()[0].strip()


def parse_cuda_visible_devices_pair() -> tuple[int, int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly two physical GPU ids per node, "
            f"got {raw!r}"
        )
    try:
        gpu0, gpu1 = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must contain numeric GPU ids, got {raw!r}") from exc
    if gpu1 != (gpu0 ^ 1):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be an XOR pair such as 0,1 or 6,7, "
            f"got {gpu0},{gpu1}"
        )
    return gpu0, gpu1


def parse_topology_link(output: str, source: str, target: str) -> str:
    header = None
    target_column = -1
    for raw_line in output.splitlines():
        fields = ANSI_ESCAPE_RE.sub("", raw_line).split()
        if not fields:
            continue
        if fields[0].startswith("GPU") and header is None and target in fields:
            header = fields
            target_column = fields.index(target) + 1
            continue
        if fields[0] == source:
            if target_column <= 0 or target_column >= len(fields):
                break
            return fields[target_column]
    raise RuntimeError(f"could not parse topology link {source}->{target}")


def gpu_topology_link(gpu0: int, gpu1: int) -> str:
    errors = []

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        output = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, env=env)
        return parse_topology_link(output, f"GPU{gpu0}", f"GPU{gpu1}")
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        errors.append(f"physical GPU{gpu0}->GPU{gpu1}: {exc}")

    try:
        output = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        return parse_topology_link(output, "GPU0", "GPU1")
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        errors.append(f"visible GPU0->GPU1: {exc}")

    raise RuntimeError(
        "could not parse nvidia-smi topo -m link between "
        f"physical GPU{gpu0} and GPU{gpu1}; attempts: {'; '.join(errors)}"
    )


def validate_nvlink_pair(gpu0: int, gpu1: int) -> str:
    link = gpu_topology_link(gpu0, gpu1)
    if not link.upper().startswith("NV"):
        raise RuntimeError(
            f"physical GPU{gpu0} and GPU{gpu1} are not connected by NVLink "
            f"according to nvidia-smi topo -m: link={link!r}"
        )
    return link


def validate_static_args(args: argparse.Namespace) -> None:
    if args.rails not in (1, 2):
        raise RuntimeError("the rewritten ring all-reduce engine supports only --rails 1 or --rails 2")
    if args.rails == 2 and not args.iface1:
        raise RuntimeError("--iface1 is required with --rails 2")
    if args.numel <= 0:
        raise RuntimeError("--numel must be positive")
    if args.iters <= 0:
        raise RuntimeError("--iters must be positive")
    if args.warmup_iters < 0:
        raise RuntimeError("--warmup-iters must be non-negative")
    if args.local_p2p_chunk_mb <= 0:
        raise RuntimeError("--local-p2p-chunk-mb must be positive")


def validate_rank_topology(rank: int, world_size: int, local_rank: int, metas: List[RankMeta]) -> None:
    if world_size % 2 != 0:
        raise RuntimeError("standard two-GPU-per-node ring requires an even WORLD_SIZE")
    if local_rank not in (0, 1):
        raise RuntimeError(f"LOCAL_RANK must be 0 or 1, got {local_rank}")
    if rank % 2 != local_rank:
        raise RuntimeError(
            "rank mapping must be rank0=node0/GPU0, rank1=node0/GPU1, "
            f"... but rank={rank} LOCAL_RANK={local_rank}"
        )
    if len(metas) != world_size:
        raise RuntimeError("rank metadata gather returned an incomplete world")
    for node_id in range(world_size // 2):
        left = metas[2 * node_id]
        right = metas[2 * node_id + 1]
        if left.hostname != right.hostname:
            raise RuntimeError(
                f"ranks {left.rank} and {right.rank} must share one node, "
                f"got {left.hostname!r} and {right.hostname!r}"
            )
    if world_size > 2:
        node_hosts = [metas[2 * node_id].hostname for node_id in range(world_size // 2)]
        if len(set(node_hosts)) != len(node_hosts):
            raise RuntimeError(f"expected one unique host per node pair, got {node_hosts}")


def build_rank_meta(args: argparse.Namespace, rank: int) -> RankMeta:
    return RankMeta(
        rank=rank,
        hostname=socket.gethostname(),
        ip0=get_interface_ipv4(args.iface0),
        ip1=get_interface_ipv4(args.iface1) if args.rails == 2 else "",
    )


def remote_edge_id(rank: int, world_size: int) -> int:
    right = (rank + 1) % world_size
    if rank % 2 == 1:
        return right // 2
    return rank // 2


class Bf16RingAllreduce:
    def __init__(
        self,
        ext,
        nccl_group,
        rank: int,
        world_size: int,
        remote_sender,
        remote_receiver,
        left_is_local: bool,
        right_is_local: bool,
        enable_nvtx: bool,
        overlap_local_p2p: bool,
        local_p2p_chunk_elems: int,
    ) -> None:
        self.ext = ext
        self.nccl_group = nccl_group
        self.rank = rank
        self.world_size = world_size
        self.remote_sender = remote_sender
        self.remote_receiver = remote_receiver
        self.left_is_local = left_is_local
        self.right_is_local = right_is_local
        self.enable_nvtx = enable_nvtx
        self.overlap_local_p2p = overlap_local_p2p
        self.local_p2p_chunk_elems = local_p2p_chunk_elems
        self.left = (rank - 1 + world_size) % world_size
        self.right = (rank + 1) % world_size

    def local_p2p_exchange(self, send_buf: torch.Tensor, recv_buf: torch.Tensor) -> None:
        if not self.left_is_local and not self.right_is_local:
            return

        send_flat = send_buf.view(-1)
        recv_flat = recv_buf.view(-1)
        if send_flat.numel() != recv_flat.numel():
            raise ValueError("local P2P send and receive buffers must have the same element count")

        for offset in range(0, send_flat.numel(), self.local_p2p_chunk_elems):
            count = min(self.local_p2p_chunk_elems, send_flat.numel() - offset)
            ops = []
            if self.right_is_local:
                ops.append(
                    dist.P2POp(
                        dist.isend,
                        send_flat.narrow(0, offset, count),
                        self.right,
                        group=self.nccl_group,
                    )
                )
            if self.left_is_local:
                ops.append(
                    dist.P2POp(
                        dist.irecv,
                        recv_flat.narrow(0, offset, count),
                        self.left,
                        group=self.nccl_group,
                    )
                )
            reqs = dist.batch_isend_irecv(ops) if ops else []
            for req in reqs:
                req.wait()

    def exchange(self, send_buf: torch.Tensor, recv_buf: torch.Tensor) -> None:
        send_ready_event = None
        if self.remote_sender is not None:
            send_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
            send_ready_event.record(torch.cuda.current_stream(send_buf.device))

        ops = []
        if not self.overlap_local_p2p:
            self.local_p2p_exchange(send_buf, recv_buf)
            reqs = []
        else:
            if self.right_is_local:
                ops.append(dist.P2POp(dist.isend, send_buf, self.right, group=self.nccl_group))
            if self.left_is_local:
                ops.append(dist.P2POp(dist.irecv, recv_buf, self.left, group=self.nccl_group))
            reqs = dist.batch_isend_irecv(ops) if ops else []

        same_remote_peer = (
            self.remote_sender is not None
            and self.remote_receiver is not None
            and self.left == self.right
        )
        if same_remote_peer:
            concurrent_remote_exchange(
                self.remote_sender,
                send_buf,
                self.remote_receiver,
                recv_buf,
                send_ready_event,
            )
        else:
            if self.remote_sender is not None:
                send_ready_event.synchronize()
                self.remote_sender.send(send_buf)
            if self.remote_receiver is not None:
                self.remote_receiver.recv(recv_buf)
        for req in reqs:
            req.wait()

    def allreduce_(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.bfloat16:
            raise TypeError("BF16 ring all-reduce expects torch.bfloat16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("tensor must be contiguous CUDA memory")
        if tensor.numel() % self.world_size != 0:
            raise ValueError("tensor length must be divisible by WORLD_SIZE")

        chunk_elems = tensor.numel() // self.world_size
        if chunk_elems <= 0:
            raise ValueError("each rank must own at least one BF16 element per ring chunk")

        chunks = [tensor[i * chunk_elems : (i + 1) * chunk_elems] for i in range(self.world_size)]
        recv_buf = torch.empty(chunk_elems, dtype=torch.bfloat16, device=tensor.device)

        send_idx = self.rank
        for step in range(self.world_size - 1):
            recv_idx = (self.rank - step - 1 + self.world_size) % self.world_size
            with maybe_nvtx_range(self.enable_nvtx, f"rank{self.rank}.rs.exchange.{step}"):
                self.exchange(chunks[send_idx], recv_buf)
            with maybe_nvtx_range(self.enable_nvtx, f"rank{self.rank}.rs.add.{step}"):
                self.ext.bf16_add_(chunks[recv_idx], recv_buf)
            send_idx = recv_idx

        send_idx = (self.rank + 1) % self.world_size
        for step in range(self.world_size - 1):
            recv_idx = (self.rank - step + self.world_size) % self.world_size
            with maybe_nvtx_range(self.enable_nvtx, f"rank{self.rank}.ag.exchange.{step}"):
                self.exchange(chunks[send_idx], chunks[recv_idx])
            send_idx = recv_idx

        return tensor


class QuantizedMeeRingAllreduce:
    def __init__(
        self,
        ext,
        exchange_engine: Bf16RingAllreduce,
        rank: int,
        world_size: int,
        nbits: int,
        enable_nvtx: bool,
    ) -> None:
        self.ext = ext
        self.exchange_engine = exchange_engine
        self.rank = rank
        self.world_size = world_size
        self.nbits = nbits
        self.enable_nvtx = enable_nvtx

    def exchange(self, send_buf: torch.Tensor, recv_buf: torch.Tensor) -> None:
        self.exchange_engine.exchange(send_buf, recv_buf)

    def allreduce_(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.bfloat16:
            raise TypeError("quantized MEE ring all-reduce expects torch.bfloat16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("tensor must be contiguous CUDA memory")
        if tensor.numel() % self.world_size != 0:
            raise ValueError("tensor length must be divisible by WORLD_SIZE")

        chunk_elems = tensor.numel() // self.world_size
        if chunk_elems % MEE_ALIGNMENT_ELEMS != 0:
            raise ValueError("each ring chunk must be divisible by 256 elements for hierarchical MEE")

        comp_bytes = self.ext.mee_compressed_bytes(chunk_elems, self.nbits)
        chunks = [tensor[i * chunk_elems : (i + 1) * chunk_elems] for i in range(self.world_size)]
        rand_pool = torch.rand(chunk_elems, dtype=torch.bfloat16, device=tensor.device)
        send_comp = torch.empty(comp_bytes, dtype=torch.uint8, device=tensor.device)
        recv_comp = torch.empty_like(send_comp)

        with maybe_nvtx_range(self.enable_nvtx, f"rank{self.rank}.q.rs.compress_init"):
            self.ext.mee_compress_bf16(chunks[self.rank], send_comp, rand_pool, self.nbits)

        for step in range(self.world_size - 1):
            recv_idx = (self.rank - step - 1 + self.world_size) % self.world_size
            with maybe_nvtx_range(self.enable_nvtx, f"rank{self.rank}.q.rs.exchange.{step}"):
                self.exchange(send_comp, recv_comp)
            if step == self.world_size - 2:
                with maybe_nvtx_range(self.enable_nvtx, f"rank{self.rank}.q.rs.decompress_add.{step}"):
                    self.ext.mee_decompress_add_bf16(recv_comp, chunks[recv_idx], self.nbits)
            else:
                with maybe_nvtx_range(self.enable_nvtx, f"rank{self.rank}.q.rs.dec_comp.{step}"):
                    self.ext.mee_dec_comp_bf16(recv_comp, chunks[recv_idx], send_comp, rand_pool, self.nbits)

        owned_idx = (self.rank + 1) % self.world_size
        with maybe_nvtx_range(self.enable_nvtx, f"rank{self.rank}.q.ag.compress_owned"):
            self.ext.mee_compress_bf16(chunks[owned_idx], send_comp, rand_pool, self.nbits)

        for step in range(self.world_size - 1):
            recv_idx = (self.rank - step + self.world_size) % self.world_size
            with maybe_nvtx_range(self.enable_nvtx, f"rank{self.rank}.q.ag.exchange.{step}"):
                self.exchange(send_comp, recv_comp)
            with maybe_nvtx_range(self.enable_nvtx, f"rank{self.rank}.q.ag.decompress.{step}"):
                self.ext.mee_decompress_bf16(recv_comp, chunks[recv_idx], self.nbits)
            send_comp, recv_comp = recv_comp, send_comp

        return tensor


def verify_result(tensor: torch.Tensor, base: torch.Tensor, nccl_group) -> None:
    ref = base.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=nccl_group)

    local_max_abs = (tensor.float() - ref.float()).abs().max()
    max_abs = local_max_abs.clone()
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=nccl_group)

    ref_rank0 = tensor.clone() if dist.get_rank() == 0 else torch.empty_like(tensor)
    dist.broadcast(ref_rank0, src=0, group=nccl_group)
    identity_mismatch = torch.tensor(
        [int(not torch.equal(tensor, ref_rank0))],
        dtype=torch.int32,
        device=tensor.device,
    )
    dist.all_reduce(identity_mismatch, op=dist.ReduceOp.MAX, group=nccl_group)

    ref_mismatch = torch.tensor(
        [int(not torch.allclose(tensor.float(), ref.float(), rtol=1e-2, atol=1.0))],
        dtype=torch.int32,
        device=tensor.device,
    )
    dist.all_reduce(ref_mismatch, op=dist.ReduceOp.MAX, group=nccl_group)

    if identity_mismatch.item() != 0:
        raise RuntimeError("ring all-reduce results differed across ranks")
    if ref_mismatch.item() != 0:
        raise RuntimeError(f"ring all-reduce differed from NCCL reference, max_abs={max_abs.item():.6f}")
    if dist.get_rank() == 0:
        print(f"verify max_abs_vs_nccl={max_abs.item():.6f}", flush=True)


def make_input(rank: int, numel: int, device: torch.device) -> torch.Tensor:
    return torch.full((numel,), float(rank + 1), dtype=torch.bfloat16, device=device)


def main() -> None:
    args = parse_args()
    validate_static_args(args)

    rank = env_int("RANK")
    world_size = env_int("WORLD_SIZE")
    local_rank = env_int("LOCAL_RANK")

    if args.expected_world_size > 0 and world_size != args.expected_world_size:
        raise RuntimeError(f"expected WORLD_SIZE={args.expected_world_size}, got {world_size}")
    if args.numel % world_size != 0:
        raise RuntimeError("--numel must be divisible by WORLD_SIZE")
    if args.mode == "quantized" and (args.numel // world_size) % MEE_ALIGNMENT_ELEMS != 0:
        raise RuntimeError("--numel / WORLD_SIZE must be divisible by 256 for hierarchical MEE")

    physical_gpu0, physical_gpu1 = parse_cuda_visible_devices_pair()
    nvlink_link = validate_nvlink_pair(physical_gpu0, physical_gpu1)

    dist.init_process_group(backend="gloo", timeout=timedelta(minutes=args.timeout_minutes))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    nccl_group = dist.new_group(backend="nccl")

    if rank == 0:
        load_ring_allreduce_ext(verbose=args.build_verbose)
    dist.barrier()
    ext = load_ring_allreduce_ext(verbose=False)
    dist.barrier()

    local_meta = build_rank_meta(args, rank)
    gathered: List[Optional[RankMeta]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_meta)
    metas = [meta for meta in gathered if meta is not None]
    metas.sort(key=lambda meta: meta.rank)
    validate_rank_topology(rank, world_size, local_rank, metas)

    node_id = rank // 2
    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size
    left_is_local = left // 2 == node_id
    right_is_local = right // 2 == node_id

    chunk_elems = args.numel // world_size
    bf16_message_bytes = chunk_elems * BF16_BYTES
    compressed_message_bytes = (
        ext.mee_compressed_bytes(chunk_elems, args.nbits) if args.mode == "quantized" else 0
    )
    message_bytes = compressed_message_bytes if args.mode == "quantized" else bf16_message_bytes
    rail_message_bytes = [
        message_bytes // args.rails + (1 if rail < message_bytes % args.rails else 0)
        for rail in range(args.rails)
    ]
    local_p2p_chunk_elems = max(1, args.local_p2p_chunk_mb * 1024 * 1024 // BF16_BYTES)

    dist.barrier()
    remote_sender = None
    remote_receiver = None

    if not left_is_local:
        pair_id = remote_edge_id(rank, world_size)
        remote_receiver = ext.RdmaReceiver(
            local_meta.ip0,
            local_meta.ip1,
            args.base_port + pair_id * 100,
            args.rails,
            device.index,
            message_bytes,
            args.gid_index,
        )
    if not right_is_local:
        pair_id = remote_edge_id(rank, world_size)
        remote_sender = ext.RdmaSender(
            metas[right].ip0,
            metas[right].ip1,
            args.base_port + pair_id * 100,
            args.rails,
            device.index,
            message_bytes,
            args.gid_index,
        )
    dist.barrier()

    exchange_engine = Bf16RingAllreduce(
        ext=ext,
        nccl_group=nccl_group,
        rank=rank,
        world_size=world_size,
        remote_sender=remote_sender,
        remote_receiver=remote_receiver,
        left_is_local=left_is_local,
        right_is_local=right_is_local,
        enable_nvtx=args.enable_nvtx,
        overlap_local_p2p=args.overlap_local_p2p,
        local_p2p_chunk_elems=local_p2p_chunk_elems,
    )
    if args.mode == "quantized":
        ring = QuantizedMeeRingAllreduce(
            ext=ext,
            exchange_engine=exchange_engine,
            rank=rank,
            world_size=world_size,
            nbits=args.nbits,
            enable_nvtx=args.enable_nvtx,
        )
    else:
        ring = exchange_engine

    if rank == 0:
        compression_text = ""
        if args.mode == "quantized":
            compression_ratio = bf16_message_bytes / max(compressed_message_bytes, 1)
            compression_text = (
                f" nbits={args.nbits} compressed_message_bytes={compressed_message_bytes} "
                f"compression_ratio={compression_ratio:.2f}"
            )
        print(
            "ring_topology "
            f"world={world_size} nodes={world_size // 2} mode={args.mode} rails={args.rails} "
            f"cuda_visible_devices={physical_gpu0},{physical_gpu1} "
            f"nvlink={nvlink_link} "
            f"numel={args.numel} chunk_elems={chunk_elems} bf16_message_bytes={bf16_message_bytes} "
            f"message_bytes={message_bytes} rail_message_bytes={','.join(map(str, rail_message_bytes))}"
            f"{compression_text} "
            f"local_p2p_chunk_mib={args.local_p2p_chunk_mb} "
            f"overlap_local_p2p={args.overlap_local_p2p}",
            flush=True,
        )

    base = make_input(rank, args.numel, device)
    work = base.clone()
    if rank == 0:
        print("ring_progress input_ready", flush=True)

    for warmup_idx in range(args.warmup_iters):
        work.copy_(base)
        ring.allreduce_(work)
        if rank == 0:
            print(f"ring_progress warmup_done={warmup_idx + 1}/{args.warmup_iters}", flush=True)

    torch.cuda.synchronize(device)
    time.sleep(1)
    dist.barrier()
    if rank == 0:
        print("ring_progress timed_start", flush=True)
    cuda_profiler_start(args.cuda_profiler_range)

    start = time.perf_counter()
    for _ in range(args.iters):
        work.copy_(base)
        ring.allreduce_(work)
        if rank == 0:
            print(f"ring_progress main_done={_ + 1}/{args.iters}", flush=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"rank{rank} ring_progress timed_stop elapsed={elapsed:.6f}s", flush=True)
    cuda_profiler_stop(args.cuda_profiler_range)

    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=nccl_group)
    elapsed_max = float(elapsed_tensor.item())
    dist.barrier()

    if args.verify:
        verify_result(work, base, nccl_group)

    seconds_per_iter = elapsed_max / max(args.iters, 1)
    input_bytes = args.numel * BF16_BYTES
    algo_bytes = input_bytes * 2 * (world_size - 1) / world_size
    remote_edge_bytes = message_bytes * 2 * (world_size - 1)
    algbw_gbps = (algo_bytes / seconds_per_iter) * 8.0 / 1e9
    remote_edge_gbps = (remote_edge_bytes / seconds_per_iter) * 8.0 / 1e9
    per_rail_remote_edge_gbps = remote_edge_gbps / args.rails
    per_node_duplex_gbps = 2.0 * remote_edge_gbps if world_size > 2 else 0.0

    if rank == 0:
        print(
            "ring_allreduce "
            f"world={world_size} nodes={world_size // 2} mode={args.mode} rails={args.rails} nbits={args.nbits} "
            f"numel={args.numel} chunk_elems={chunk_elems} input_bytes={input_bytes} "
            f"message_bytes={message_bytes} iters={args.iters} avg_s={seconds_per_iter:.6f} "
            f"algbw={algbw_gbps:.2f}Gbps remote_edge_bw={remote_edge_gbps:.2f}Gbps "
            f"per_rail_remote_edge_bw={per_rail_remote_edge_gbps:.2f}Gbps "
            f"per_node_duplex_remote_bw={per_node_duplex_gbps:.2f}Gbps",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
