import os
import re
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist


BF16_BYTES = 2
DEFAULT_BASE_PORT = 18740
DEFAULT_IFACE0 = "eth0"
DEFAULT_IFACE1 = "eth1"
DEFAULT_LOCAL_P2P_CHUNK_MB = 8
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class RankMeta:
    rank: int
    hostname: str
    ip0: str
    ip1: str = ""


def _artifact_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_ring_ext(verbose: bool):
    root = _artifact_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rdma_comm_compress.ring_allreduce_ext import load_ring_allreduce_ext

    return load_ring_allreduce_ext(verbose=verbose)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() not in ("0", "false", "no", "off")


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _payload_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _payload_view(tensor: torch.Tensor, sliced_portion: float) -> torch.Tensor:
    payload = tensor
    if payload.dtype not in (torch.bfloat16, torch.uint8):
        payload = payload.view(torch.uint8)

    if sliced_portion < 1:
        keep = int(payload.numel() * sliced_portion)
        payload = payload[payload.numel() - keep :] if keep > 0 else payload[payload.numel() :]

    if not payload.is_contiguous():
        raise ValueError("ring RDMA payload view must be contiguous")
    return payload


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
    gpu0, gpu1 = int(parts[0]), int(parts[1])
    if gpu1 != (gpu0 ^ 1):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be an XOR pair such as 0,1 or 6,7, "
            f"got {gpu0},{gpu1}"
        )
    return gpu0, gpu1


def parse_topology_link(output: str, source: str, target: str) -> str:
    target_column = -1
    for raw_line in output.splitlines():
        fields = ANSI_ESCAPE_RE.sub("", raw_line).split()
        if not fields:
            continue
        if fields[0].startswith("GPU") and target in fields:
            target_column = fields.index(target) + 1
            continue
        if fields[0] == source and target_column > 0 and target_column < len(fields):
            return fields[target_column]
    raise RuntimeError(f"could not parse topology link {source}->{target}")


def gpu_topology_link(gpu0: int, gpu1: int) -> str:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    output = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, env=env)
    return parse_topology_link(output, f"GPU{gpu0}", f"GPU{gpu1}")


def validate_nvlink_pair(gpu0: int, gpu1: int) -> str:
    link = gpu_topology_link(gpu0, gpu1)
    if not link.upper().startswith("NV"):
        raise RuntimeError(
            f"physical GPU{gpu0} and GPU{gpu1} are not connected by NVLink: link={link!r}"
        )
    return link


def build_rank_meta(iface0: str, iface1: str, rails: int, rank: int) -> RankMeta:
    return RankMeta(
        rank=rank,
        hostname=socket.gethostname(),
        ip0=get_interface_ipv4(iface0),
        ip1=get_interface_ipv4(iface1) if rails == 2 else "",
    )


def validate_rank_topology(rank: int, world_size: int, local_rank: int, metas: List[RankMeta]) -> None:
    if world_size % 2 != 0:
        raise RuntimeError("two-GPUs-per-node ring requires an even WORLD_SIZE")
    if local_rank not in (0, 1):
        raise RuntimeError(f"LOCAL_RANK must be 0 or 1, got {local_rank}")
    if rank % 2 != local_rank:
        raise RuntimeError(
            "expected rank0/rank1 on node0 GPU0/GPU1, rank2/rank3 on node1 GPU0/GPU1, "
            f"but rank={rank} LOCAL_RANK={local_rank}"
        )
    for node_id in range(world_size // 2):
        left = metas[2 * node_id]
        right = metas[2 * node_id + 1]
        if left.hostname != right.hostname:
            raise RuntimeError(
                f"ranks {left.rank} and {right.rank} must share one node, "
                f"got {left.hostname!r} and {right.hostname!r}"
            )


def remote_edge_id(rank: int, world_size: int) -> int:
    right = (rank + 1) % world_size
    if rank % 2 == 1:
        return right // 2
    return rank // 2


def concurrent_remote_exchange(
    remote_sender,
    send_buf: torch.Tensor,
    remote_receiver,
    recv_buf: torch.Tensor,
    send_ready_event: Optional[torch.cuda.Event],
) -> None:
    errors = []

    def send_worker() -> None:
        try:
            if send_ready_event is not None:
                send_ready_event.synchronize()
            remote_sender.send(send_buf)
        except BaseException as exc:
            errors.append(exc)

    send_thread = threading.Thread(target=send_worker, name="ring-rdma-send")
    send_thread.start()
    remote_receiver.recv(recv_buf)
    send_thread.join()
    if errors:
        raise errors[0]


class RingRdmaExchange:
    def __init__(
        self,
        process_group,
        rank: int,
        world_size: int,
        remote_sender,
        remote_receiver,
        left_is_local: bool,
        right_is_local: bool,
        overlap_local_p2p: bool,
        local_p2p_chunk_bytes: int,
    ) -> None:
        self.process_group = process_group
        self.rank = rank
        self.world_size = world_size
        self.remote_sender = remote_sender
        self.remote_receiver = remote_receiver
        self.left_is_local = left_is_local
        self.right_is_local = right_is_local
        self.overlap_local_p2p = overlap_local_p2p
        self.local_p2p_chunk_bytes = local_p2p_chunk_bytes
        self.left = (rank - 1 + world_size) % world_size
        self.right = (rank + 1) % world_size
        print("OVERLAP_LOCAL_P2P={}".format(self.overlap_local_p2p))

    def local_p2p_exchange(self, send_buf: torch.Tensor, recv_buf: torch.Tensor, tag: int) -> None:
        if not self.left_is_local and not self.right_is_local:
            return
        send_flat = send_buf.view(-1)
        recv_flat = recv_buf.view(-1)
        if send_flat.numel() != recv_flat.numel() or send_flat.element_size() != recv_flat.element_size():
            raise ValueError("local P2P send and receive buffers must have the same byte size")

        chunk_elems = max(1, self.local_p2p_chunk_bytes // max(send_flat.element_size(), 1))
        for offset in range(0, send_flat.numel(), chunk_elems):
            count = min(chunk_elems, send_flat.numel() - offset)
            ops = []
            if self.right_is_local:
                ops.append(
                    dist.P2POp(
                        dist.isend,
                        send_flat.narrow(0, offset, count),
                        self.right,
                        group=self.process_group,
                        tag=tag,
                    )
                )
            if self.left_is_local:
                ops.append(
                    dist.P2POp(
                        dist.irecv,
                        recv_flat.narrow(0, offset, count),
                        self.left,
                        group=self.process_group,
                        tag=tag,
                    )
                )
            for req in dist.batch_isend_irecv(ops) if ops else []:
                req.wait()

    def exchange(
        self,
        send_buf: torch.Tensor,
        recv_buf: torch.Tensor,
        tag: int,
    ) -> None:
        if _payload_nbytes(send_buf) != _payload_nbytes(recv_buf):
            raise ValueError("ring exchange send and receive payloads must have the same byte size")
        if _payload_nbytes(send_buf) == 0:
            return

        send_ready_event = None
        if self.remote_sender is not None:
            send_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
            send_ready_event.record(torch.cuda.current_stream(send_buf.device))

        
        if not self.overlap_local_p2p:
            self.local_p2p_exchange(send_buf, recv_buf, tag)
            reqs = []
        else:
            ops = []
            if self.right_is_local:
                ops.append(
                    dist.P2POp(dist.isend, send_buf, self.right, group=self.process_group, tag=tag)
                )
            if self.left_is_local:
                ops.append(
                    dist.P2POp(dist.irecv, recv_buf, self.left, group=self.process_group, tag=tag)
                )
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

    def exchange_pipeline(
        self,
        send_buf: torch.Tensor,
        recv_buf: torch.Tensor,
        tag: int,
        remote_send_fn=None,
        remote_recv_fn=None,
    ) -> None:
        if _payload_nbytes(send_buf) != _payload_nbytes(recv_buf):
            raise ValueError("ring exchange send and receive payloads must have the same byte size")
        if _payload_nbytes(send_buf) == 0:
            return

        if not self.overlap_local_p2p:
            self.local_p2p_exchange(send_buf, recv_buf, tag)
            reqs = []
        else:
            ops = []
            if self.right_is_local:
                ops.append(
                    dist.P2POp(dist.isend, send_buf, self.right, group=self.process_group, tag=tag)
                )
            if self.left_is_local:
                ops.append(
                    dist.P2POp(dist.irecv, recv_buf, self.left, group=self.process_group, tag=tag)
                )
            reqs = dist.batch_isend_irecv(ops) if ops else []

        def run_remote_send() -> None:
            if self.remote_sender is None:
                return
            if remote_send_fn is None:
                self.remote_sender.send(send_buf)
            else:
                remote_send_fn(self.remote_sender)

        def run_remote_recv() -> None:
            if self.remote_receiver is None:
                return
            if remote_recv_fn is None:
                self.remote_receiver.recv(recv_buf)
            else:
                remote_recv_fn(self.remote_receiver)

        same_remote_peer = (
            self.remote_sender is not None
            and self.remote_receiver is not None
            and self.left == self.right
        )
        if same_remote_peer:
            errors = []

            def send_worker() -> None:
                try:
                    run_remote_send()
                except BaseException as exc:
                    errors.append(exc)

            send_thread = threading.Thread(target=send_worker, name="ring-rdma-pipeline-send")
            send_thread.start()
            run_remote_recv()
            send_thread.join()
            if errors:
                raise errors[0]
        else:
            if self.remote_sender is not None:
                run_remote_send()
            if self.remote_receiver is not None:
                run_remote_recv()

        for req in reqs:
            req.wait()


class RingRdmaManager:
    def __init__(
        self,
        params,
        capacity_bytes: int,
        manager_key: str = "ring_rdma_manager",
        pipeline_dynamiQ: bool = False,
    ) -> None:
        self.params = params
        self.manager_key = manager_key
        self.pipeline_dynamiQ = pipeline_dynamiQ
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = _env_int("LOCAL_RANK", self.rank % 2)
        self.rails = _env_int("RING_RDMA_RAILS", 1)
        self.iface0 = os.environ.get("RING_RDMA_IFACE0", DEFAULT_IFACE0)
        self.iface1 = os.environ.get("RING_RDMA_IFACE1", DEFAULT_IFACE1)
        self.base_port = _env_int("RING_RDMA_BASE_PORT", DEFAULT_BASE_PORT)
        self.gid_index = _env_int("RING_RDMA_GID_INDEX", -1)
        self.capacity_bytes = max(capacity_bytes, self.rails)
        self.collective_lock = threading.Lock()

        if self.rails not in (1, 2):
            raise RuntimeError("RING_RDMA_RAILS must be 1 or 2")
        if self.rails == 2 and not self.iface1:
            raise RuntimeError("RING_RDMA_IFACE1 is required with two RDMA rails")

        if self.rank == 0:
            _load_ring_ext(verbose=_env_bool("RING_EXT_BUILD_VERBOSE", False))
        dist.barrier()
        self.ext = _load_ring_ext(verbose=False)
        dist.barrier()
        self.local_process_group = dist.new_group(ranks=list(range(self.world_size)))
        warmup = torch.zeros(1, device=torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(warmup, group=self.local_process_group)
        torch.cuda.synchronize(warmup.device)
        dist.barrier()

        local_meta = build_rank_meta(self.iface0, self.iface1, self.rails, self.rank)
        gathered: List[Optional[RankMeta]] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_meta)
        metas = [meta for meta in gathered if meta is not None]
        metas.sort(key=lambda meta: meta.rank)
        validate_rank_topology(self.rank, self.world_size, self.local_rank, metas)

        nvlink_link = "unchecked"
        if _env_bool("RING_VALIDATE_NVLINK", True):
            physical_gpu0, physical_gpu1 = parse_cuda_visible_devices_pair()
            nvlink_link = validate_nvlink_pair(physical_gpu0, physical_gpu1)

        node_id = self.rank // 2
        left = (self.rank - 1 + self.world_size) % self.world_size
        right = (self.rank + 1) % self.world_size
        left_is_local = left // 2 == node_id
        right_is_local = right // 2 == node_id

        dist.barrier()
        remote_receiver = None
        remote_sender = None
        receiver_cls = self.ext.PipelineRdmaReceiver if self.pipeline_dynamiQ else self.ext.RdmaReceiver
        sender_cls = self.ext.PipelineRdmaSender if self.pipeline_dynamiQ else self.ext.RdmaSender
        if not left_is_local:
            pair_id = remote_edge_id(self.rank, self.world_size)
            remote_receiver = receiver_cls(
                local_meta.ip0,
                local_meta.ip1,
                self.base_port + pair_id * 100,
                self.rails,
                torch.cuda.current_device(),
                self.capacity_bytes,
                self.gid_index,
            )
        if not right_is_local:
            pair_id = remote_edge_id(self.rank, self.world_size)
            remote_sender = sender_cls(
                metas[right].ip0,
                metas[right].ip1,
                self.base_port + pair_id * 100,
                self.rails,
                torch.cuda.current_device(),
                self.capacity_bytes,
                self.gid_index,
            )
        dist.barrier()

        local_p2p_chunk_mb = _env_int("RING_LOCAL_P2P_CHUNK_MB", DEFAULT_LOCAL_P2P_CHUNK_MB)
        self.exchange_engine = RingRdmaExchange(
            process_group=self.local_process_group,
            rank=self.rank,
            world_size=self.world_size,
            remote_sender=remote_sender,
            remote_receiver=remote_receiver,
            left_is_local=left_is_local,
            right_is_local=right_is_local,
            overlap_local_p2p=_env_bool("RING_OVERLAP_LOCAL_P2P", True),
            local_p2p_chunk_bytes=local_p2p_chunk_mb * 1024 * 1024,
        )

        if self.rank == 0:
            print(
                "ring_rdma_hooks "
                f"manager={self.manager_key} "
                f"world={self.world_size} nodes={self.world_size // 2} rails={self.rails} "
                f"base_port={self.base_port} "
                f"pipeline_dynamiQ={self.pipeline_dynamiQ} "
                f"capacity_bytes={self.capacity_bytes} iface0={self.iface0} iface1={self.iface1} "
                f"nvlink={nvlink_link}",
                flush=True,
            )


def default_capacity_bytes(params) -> int:
    env_capacity = os.environ.get("RING_RDMA_CAPACITY_BYTES")
    if env_capacity:
        return int(env_capacity)

    world_size = dist.get_world_size()
    max_bucket_elems = int(params.get("MAX_BUCKET_SIZE", 1 << 30))
    alignment = max(1, int(params.get("chunk_size", 1)))
    chunk_elems = _align_up((max_bucket_elems + world_size - 1) // world_size, alignment)
    return chunk_elems * BF16_BYTES


def get_ring_rdma_manager(
    params,
    manager_key: str = "ring_rdma_manager",
    pipeline_dynamiQ: bool = False,
) -> RingRdmaManager:
    capacity_bytes = default_capacity_bytes(params)
    manager = params.get(manager_key)
    if manager is not None:
        if getattr(manager, "pipeline_dynamiQ", False) != pipeline_dynamiQ:
            raise RuntimeError("existing ring RDMA manager pipeline mode does not match requested mode")
        if manager.capacity_bytes < capacity_bytes:
            raise RuntimeError(
                "existing ring RDMA capacity is too small; set RING_RDMA_CAPACITY_BYTES "
                "before initializing hooks"
            )
        return manager

    manager = RingRdmaManager(
        params,
        capacity_bytes,
        manager_key=manager_key,
        pipeline_dynamiQ=pipeline_dynamiQ,
    )
    params[manager_key] = manager
    return manager


def _init_buffers(tag, input_chunk, dtype, callback_comm, params):
    if "buffer" not in params:
        params["buffer"] = {}
    if tag not in params["buffer"]:
        params["buffer"][tag] = {}

    expected_send = callback_comm.create_tensor(input_chunk[0], dtype)
    cached = params["buffer"][tag]
    if (
        "send_chunk" not in cached
        or cached["send_chunk"].dtype != expected_send.dtype
        or cached["send_chunk"].numel() != expected_send.numel()
    ):
        cached["send_chunk"] = expected_send
        cached["recv_chunk"] = torch.empty_like(expected_send)
    return cached["send_chunk"], cached["recv_chunk"]


def _init_extra_like_buffer(tag, name: str, template: torch.Tensor, params):
    cached = params["buffer"][tag]
    if (
        name not in cached
        or cached[name].dtype != template.dtype
        or cached[name].numel() != template.numel()
    ):
        cached[name] = torch.empty_like(template)
    return cached[name]


def composable_ring_rdma_allreduce_callback(
    send_vec: torch.Tensor,
    callback_comm,
    params,
    tag=0,
    dtype=torch.uint8,
    sliced_portion=1,
    manager_key="ring_rdma_manager",
):
    if send_vec.numel() == 0:
        return send_vec

    manager = get_ring_rdma_manager(
        params,
        manager_key=manager_key,
    )
    rank = dist.get_rank()
    size = dist.get_world_size()

    if "coov" in params["args"].aggregation_method:
        sliced_portion = 0

    if "chunk_size" not in params:
        params["chunk_size"] = 64

    chunk_size = (send_vec.numel() + size - 1) // size
    chunk_size = _align_up(chunk_size, params["chunk_size"])
    input_chunk = [
        send_vec[chunk_size * i : min(chunk_size * (i + 1), send_vec.numel())]
        for i in range(size)
    ]

    send_chunk, recv_chunk = _init_buffers(tag, input_chunk, dtype, callback_comm, params)
    wire_tag = tag << 8

    def exchange(src: torch.Tensor, dst: torch.Tensor, step_tag: int) -> None:
        manager.exchange_engine.exchange(
            _payload_view(src, sliced_portion),
            _payload_view(dst, sliced_portion),
            step_tag,
        )

    with manager.collective_lock:
        for step in range(size - 1):
            if step == 0:
                callback_comm.compression(input_chunk[rank], send_chunk)
            else:
                callback_comm.dec_compression(
                    recv_chunk,
                    input_chunk[(rank - step + size) % size],
                    send_chunk,
                )
            exchange(send_chunk, recv_chunk, wire_tag | step)

        owner_chunk = input_chunk[(rank + 1) % size]
        callback_comm.dec_compression(recv_chunk, owner_chunk, send_chunk)
        callback_comm.decompression(send_chunk, owner_chunk)

        chunks = [send_chunk, recv_chunk]
        for step in range(size - 1):
            exchange(chunks[step & 1], chunks[(step ^ 1) & 1], wire_tag | (size + step))
            callback_comm.decompression(
                chunks[(step ^ 1) & 1],
                input_chunk[(rank - step + size) % size],
            )

    return send_vec


def composable_ring_rdma_dynamiQ_allreduce_callback(
    send_vec: torch.Tensor,
    callback_comm,
    params,
    tag=0,
    dtype=torch.uint8,
    manager_key="ring_rdma_pipeline_manager",
):
    if send_vec.numel() == 0:
        return send_vec

    required_attrs = ("nbits", "chunk_size", "strategy", "rand_pool")
    if dtype != torch.uint8 or not all(hasattr(callback_comm, attr) for attr in required_attrs):
        return composable_ring_rdma_allreduce_callback(
            send_vec,
            callback_comm,
            params,
            tag=tag,
            dtype=dtype,
            manager_key=manager_key,
        )

    manager = get_ring_rdma_manager(
        params,
        manager_key=manager_key,
        pipeline_dynamiQ=True,
    )
    exchange_engine = manager.exchange_engine
    rank = dist.get_rank()
    size = dist.get_world_size()

    if "chunk_size" not in params:
        params["chunk_size"] = 64

    chunk_size = (send_vec.numel() + size - 1) // size
    chunk_size = _align_up(chunk_size, params["chunk_size"])
    input_chunk = [
        send_vec[chunk_size * i : min(chunk_size * (i + 1), send_vec.numel())]
        for i in range(size)
    ]

    send_chunk, recv_chunk = _init_buffers(tag, input_chunk, dtype, callback_comm, params)
    wire_tag = tag << 8
    nbits = int(callback_comm.nbits)
    quant_chunk_size = int(callback_comm.chunk_size)
    strategy = int(callback_comm.strategy)
    recv_dec_comp_lookahead = exchange_engine.right_is_local and not exchange_engine.left_is_local
    next_send_chunk = (
        _init_extra_like_buffer(tag, "pipeline_next_send_chunk", send_chunk, params)
        if recv_dec_comp_lookahead
        else None
    )

    def rand_pool_for(tensor: torch.Tensor) -> torch.Tensor:
        return callback_comm.rand_pool(tensor.numel(), tensor.dtype)

    def exchange_reduce_scatter(op_name: str, input_tensor: torch.Tensor, step_tag: int) -> None:
        remote_send_fn = None
        if exchange_engine.right_is_local:
            if op_name == "compress":
                callback_comm.compression(input_tensor, send_chunk)
            else:
                callback_comm.dec_compression(recv_chunk, input_tensor, send_chunk)
        else:
            rand_pool = rand_pool_for(input_tensor)
            if op_name == "compress":
                remote_send_fn = lambda sender: sender.send_compress_bf16(
                    input_tensor,
                    send_chunk,
                    rand_pool,
                    nbits,
                    quant_chunk_size,
                    strategy,
                )
            else:
                remote_send_fn = lambda sender: sender.send_dec_comp_bf16(
                    recv_chunk,
                    input_tensor,
                    send_chunk,
                    rand_pool,
                    nbits,
                    quant_chunk_size,
                    strategy,
                )

        remote_recv_fn = None
        if not exchange_engine.left_is_local:
            remote_recv_fn = lambda receiver: receiver.recv(
                recv_chunk,
                nbits,
                quant_chunk_size,
                strategy,
            )

        exchange_engine.exchange_pipeline(
            send_chunk,
            recv_chunk,
            step_tag,
            remote_send_fn=remote_send_fn,
            remote_recv_fn=remote_recv_fn,
        )

    def exchange_reduce_scatter_lookahead(
        send_src: torch.Tensor,
        next_input_tensor: torch.Tensor,
        next_send_dst: torch.Tensor,
        step_tag: int,
    ) -> None:
        rand_pool = rand_pool_for(next_input_tensor)
        remote_recv_fn = lambda receiver: receiver.recv_dec_comp_bf16(
            recv_chunk,
            next_input_tensor,
            next_send_dst,
            rand_pool,
            nbits,
            quant_chunk_size,
            strategy,
        )
        exchange_engine.exchange_pipeline(
            send_src,
            recv_chunk,
            step_tag,
            remote_recv_fn=remote_recv_fn,
        )

    def exchange_allgather(src: torch.Tensor, dst: torch.Tensor, output_tensor: torch.Tensor, step_tag: int) -> None:
        remote_send_fn = None
        if not exchange_engine.right_is_local:
            remote_send_fn = lambda sender: sender.send(
                src,
                nbits,
                quant_chunk_size,
                strategy,
            )

        remote_recv_fn = None
        remote_decompressed = not exchange_engine.left_is_local
        if remote_decompressed:
            remote_recv_fn = lambda receiver: receiver.recv_decompress_bf16(
                dst,
                output_tensor,
                nbits,
                quant_chunk_size,
                strategy,
            )

        exchange_engine.exchange_pipeline(
            src,
            dst,
            step_tag,
            remote_send_fn=remote_send_fn,
            remote_recv_fn=remote_recv_fn,
        )
        if not remote_decompressed:
            callback_comm.decompression(dst, output_tensor)

    with manager.collective_lock:
        if recv_dec_comp_lookahead:
            current_send = send_chunk
            next_send = next_send_chunk
            callback_comm.compression(input_chunk[rank], current_send)
            for step in range(size - 1):
                next_input = input_chunk[(rank - step - 1 + size) % size]
                exchange_reduce_scatter_lookahead(
                    current_send,
                    next_input,
                    next_send,
                    wire_tag | step,
                )
                current_send, next_send = next_send, current_send

            owner_chunk = input_chunk[(rank + 1) % size]
            callback_comm.decompression(current_send, owner_chunk)
            chunks = [current_send, recv_chunk]
        else:
            for step in range(size - 1):
                if step == 0:
                    exchange_reduce_scatter("compress", input_chunk[rank], wire_tag | step)
                else:
                    exchange_reduce_scatter(
                        "dec_comp",
                        input_chunk[(rank - step + size) % size],
                        wire_tag | step,
                    )

            owner_chunk = input_chunk[(rank + 1) % size]
            callback_comm.dec_compression(recv_chunk, owner_chunk, send_chunk)
            callback_comm.decompression(send_chunk, owner_chunk)
            chunks = [send_chunk, recv_chunk]

        for step in range(size - 1):
            exchange_allgather(
                chunks[step & 1],
                chunks[(step ^ 1) & 1],
                input_chunk[(rank - step + size) % size],
                wire_tag | (size + step),
            )

    return send_vec
