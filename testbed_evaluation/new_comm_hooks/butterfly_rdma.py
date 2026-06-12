import os
import threading
from typing import Dict, Optional, Set

import torch
import torch.distributed as dist

from .ring_rdma import (
    DEFAULT_BASE_PORT,
    DEFAULT_IFACE0,
    DEFAULT_IFACE1,
    RankMeta,
    _env_bool,
    _env_int,
    _load_ring_ext,
    _payload_nbytes,
    _payload_view,
    build_rank_meta,
    parse_cuda_visible_devices_pair,
    validate_nvlink_pair,
    validate_rank_topology,
)
from .utils import calc_order, decompression_add


DEFAULT_BUTTERFLY_PORT_OFFSET = 4000
DEFAULT_BUTTERFLY_EDGE_PORT_STRIDE = 20
DEFAULT_BUTTERFLY_LOCAL_P2P_CHUNK_MB = 8


def _node_id(rank: int) -> int:
    return rank // 2


def _is_local_peer(rank: int, peer: int) -> bool:
    return _node_id(rank) == _node_id(peer)


def _edge_port(base_port: int, world_size: int, src: int, dst: int) -> int:
    stride = _env_int("BUTTERFLY_RDMA_EDGE_PORT_STRIDE", DEFAULT_BUTTERFLY_EDGE_PORT_STRIDE)
    return base_port + (src * world_size + dst) * stride


def _butterfly_remote_peers(rank: int, world_size: int) -> tuple[Set[int], Set[int]]:
    edges = _butterfly_remote_edges(world_size)
    return (
        {dst for src, dst in edges if src == rank},
        {src for src, dst in edges if dst == rank},
    )


def _butterfly_remote_edges(world_size: int) -> Set[tuple[int, int]]:
    edges: Set[tuple[int, int]] = set()

    for rank in range(world_size):
        order = calc_order(rank, world_size)
        for _, right, _, left in order:
            if not _is_local_peer(rank, right):
                edges.add((rank, right))
            if not _is_local_peer(rank, left):
                edges.add((left, rank))

        allgather_send = (rank - 1 + world_size) % world_size
        allgather_recv = (rank + 1) % world_size
        if not _is_local_peer(rank, allgather_send):
            edges.add((rank, allgather_send))
        if not _is_local_peer(rank, allgather_recv):
            edges.add((allgather_recv, rank))

    return edges


def _local_p2p_exchange(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    left: Optional[int],
    right: Optional[int],
    process_group,
    tag: int,
    chunk_bytes: int,
) -> None:
    send_flat = send_buf.view(-1)
    recv_flat = recv_buf.view(-1)
    if send_flat.numel() != recv_flat.numel() or send_flat.element_size() != recv_flat.element_size():
        raise ValueError("butterfly local P2P buffers must have the same byte size")

    chunk_elems = max(1, chunk_bytes // max(send_flat.element_size(), 1))
    for offset in range(0, send_flat.numel(), chunk_elems):
        count = min(chunk_elems, send_flat.numel() - offset)
        ops = []
        if right is not None:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_flat.narrow(0, offset, count),
                    right,
                    group=process_group,
                    tag=tag,
                )
            )
        if left is not None:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_flat.narrow(0, offset, count),
                    left,
                    group=process_group,
                    tag=tag,
                )
            )
        for req in dist.batch_isend_irecv(ops) if ops else []:
            req.wait()


class ButterflyRdmaManager:
    def __init__(
        self,
        params,
        capacity_bytes: int,
        manager_key: str = "butterfly_rdma_manager",
        port_offset: int = DEFAULT_BUTTERFLY_PORT_OFFSET,
    ) -> None:
        self.params = params
        self.manager_key = manager_key
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = _env_int("LOCAL_RANK", self.rank % 2)
        self.rails = _env_int("RING_RDMA_RAILS", 1)
        self.iface0 = os.environ.get("RING_RDMA_IFACE0", DEFAULT_IFACE0)
        self.iface1 = os.environ.get("RING_RDMA_IFACE1", DEFAULT_IFACE1)
        self.base_port = _env_int("BUTTERFLY_RDMA_BASE_PORT", _env_int("RING_RDMA_BASE_PORT", DEFAULT_BASE_PORT)) + port_offset
        self.gid_index = _env_int("RING_RDMA_GID_INDEX", -1)
        self.cuda_device = torch.cuda.current_device()
        self.capacity_bytes = max(capacity_bytes, self.rails)
        self.collective_lock = threading.Lock()
        self.local_p2p_chunk_bytes = (
            _env_int("BUTTERFLY_LOCAL_P2P_CHUNK_MB", DEFAULT_BUTTERFLY_LOCAL_P2P_CHUNK_MB)
            * 1024
            * 1024
        )

        if self.world_size not in (4, 8):
            raise RuntimeError("butterfly RDMA prototype supports WORLD_SIZE 4 or 8")
        if self.world_size % 2 != 0:
            raise RuntimeError("butterfly RDMA prototype requires two ranks per node")
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
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_meta)
        metas = [meta for meta in gathered if isinstance(meta, RankMeta)]
        metas.sort(key=lambda meta: meta.rank)
        validate_rank_topology(self.rank, self.world_size, self.local_rank, metas)

        nvlink_link = "unchecked"
        if _env_bool("RING_VALIDATE_NVLINK", True):
            physical_gpu0, physical_gpu1 = parse_cuda_visible_devices_pair()
            nvlink_link = validate_nvlink_pair(physical_gpu0, physical_gpu1)

        self.send_peers, self.recv_peers = _butterfly_remote_peers(self.rank, self.world_size)
        self.remote_receivers: Dict[int, object] = {}
        self.remote_senders: Dict[int, object] = {}

        dist.barrier()
        for src, dst in sorted(_butterfly_remote_edges(self.world_size)):
            receiver_error = []
            receiver_thread = None

            if self.rank == dst:
                def start_receiver() -> None:
                    try:
                        torch.cuda.set_device(self.cuda_device)
                        self.remote_receivers[src] = self.ext.RdmaReceiver(
                            local_meta.ip0,
                            local_meta.ip1,
                            _edge_port(self.base_port, self.world_size, src, dst),
                            self.rails,
                            self.cuda_device,
                            self.capacity_bytes,
                            self.gid_index,
                        )
                    except BaseException as exc:
                        receiver_error.append(exc)

                receiver_thread = threading.Thread(
                    target=start_receiver,
                    name=f"butterfly-rdma-recv-{src}-{dst}",
                )
                receiver_thread.start()

            dist.barrier()

            if self.rank == src:
                self.remote_senders[dst] = self.ext.RdmaSender(
                    metas[dst].ip0,
                    metas[dst].ip1,
                    _edge_port(self.base_port, self.world_size, src, dst),
                    self.rails,
                    self.cuda_device,
                    self.capacity_bytes,
                    self.gid_index,
                )

            if receiver_thread is not None:
                receiver_thread.join()
            if receiver_error:
                raise receiver_error[0]

            dist.barrier()
        dist.barrier()

        if self.rank == 0:
            print(
                "butterfly_rdma_hooks "
                f"manager={self.manager_key} world={self.world_size} nodes={self.world_size // 2} "
                f"rails={self.rails} base_port={self.base_port} capacity_bytes={self.capacity_bytes} "
                f"iface0={self.iface0} iface1={self.iface1} nvlink={nvlink_link}",
                flush=True,
            )

    def exchange(self, send_buf: torch.Tensor, recv_buf: torch.Tensor, left: int, right: int, tag: int) -> None:
        if _payload_nbytes(send_buf) != _payload_nbytes(recv_buf):
            raise ValueError("butterfly exchange payloads must have the same byte size")
        if _payload_nbytes(send_buf) == 0:
            return

        local_left = left if _is_local_peer(self.rank, left) else None
        local_right = right if _is_local_peer(self.rank, right) else None
        remote_receiver = None if local_left is not None else self.remote_receivers[left]
        remote_sender = None if local_right is not None else self.remote_senders[right]

        errors = []

        def send_remote() -> None:
            try:
                remote_sender.send(send_buf)
            except BaseException as exc:
                errors.append(exc)

        send_thread = None
        if remote_sender is not None:
            send_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
            send_ready_event.record(torch.cuda.current_stream(send_buf.device))

            def wait_and_send() -> None:
                torch.cuda.set_device(send_buf.device)
                send_ready_event.synchronize()
                send_remote()

            send_thread = threading.Thread(target=wait_and_send, name="butterfly-rdma-send")
            send_thread.start()

        if local_left is not None or local_right is not None:
            _local_p2p_exchange(
                send_buf,
                recv_buf,
                local_left,
                local_right,
                self.local_process_group,
                tag,
                self.local_p2p_chunk_bytes,
            )

        if remote_receiver is not None:
            remote_receiver.recv(recv_buf)

        if send_thread is not None:
            send_thread.join()
        if errors:
            raise errors[0]


def get_butterfly_rdma_manager(
    params,
    capacity_bytes: int,
    manager_key: str = "butterfly_rdma_manager",
    port_offset: int = DEFAULT_BUTTERFLY_PORT_OFFSET,
) -> ButterflyRdmaManager:
    env_capacity = os.environ.get("RING_RDMA_CAPACITY_BYTES")
    if env_capacity:
        capacity_bytes = max(capacity_bytes, int(env_capacity))

    manager = params.get(manager_key)
    if manager is not None:
        if manager.capacity_bytes < capacity_bytes:
            raise RuntimeError(
                "existing butterfly RDMA capacity is too small; set RING_RDMA_CAPACITY_BYTES "
                "or rerun with a larger first bucket"
            )
        return manager

    manager = ButterflyRdmaManager(
        params,
        capacity_bytes,
        manager_key=manager_key,
        port_offset=port_offset,
    )
    params[manager_key] = manager
    return manager


def _decompression_add(recv_chunk: torch.Tensor, input_chunk: torch.Tensor, callback_comm) -> None:
    if hasattr(callback_comm, "decompression_add"):
        callback_comm.decompression_add(recv_chunk, input_chunk)
    else:
        decompression_add(recv_chunk, input_chunk, callback_comm)


def composable_butterfly_rdma_allreduce_callback(
    send_vec: torch.Tensor,
    callback_comm,
    params,
    tag=0,
    dtype=torch.uint8,
    sliced_portion=1,
    manager_key="butterfly_rdma_manager",
    manager_port_offset=DEFAULT_BUTTERFLY_PORT_OFFSET,
):
    if send_vec.numel() == 0:
        return send_vec

    rank = dist.get_rank()
    size = dist.get_world_size()
    if size not in (4, 8):
        raise RuntimeError("butterfly RDMA prototype supports two-node/four-rank or four-node/eight-rank jobs")

    if "coov" in params["args"].aggregation_method:
        sliced_portion = 0
    if "chunk_size" not in params:
        params["chunk_size"] = 64

    chunk_size = (send_vec.numel() + size - 1) // size
    chunk_size = ((chunk_size + params["chunk_size"] - 1) // params["chunk_size"]) * params["chunk_size"]
    input_chunk = [
        send_vec[chunk_size * i : min(chunk_size * (i + 1), send_vec.numel())]
        for i in range(size)
    ]

    from .ring_rdma import _init_buffers

    send_chunk, recv_chunk = _init_buffers(tag, input_chunk, dtype, callback_comm, params)
    capacity_bytes = max(
        _payload_nbytes(_payload_view(send_chunk, sliced_portion)),
        _payload_nbytes(_payload_view(recv_chunk, sliced_portion)),
    )
    manager = get_butterfly_rdma_manager(
        params,
        capacity_bytes=capacity_bytes,
        manager_key=manager_key,
        port_offset=manager_port_offset,
    )

    order = params.setdefault("butterfly_order", {})
    if order.get("rank") != rank or order.get("size") != size:
        order.clear()
        order.update({"rank": rank, "size": size, "steps": calc_order(rank, size)})

    wire_tag = tag << 8

    def exchange(src: torch.Tensor, dst: torch.Tensor, left: int, right: int, step_tag: int) -> None:
        manager.exchange(
            _payload_view(src, sliced_portion),
            _payload_view(dst, sliced_portion),
            left,
            right,
            step_tag,
        )

    with manager.collective_lock:
        for step, (src_rank, right, dst_rank, left) in enumerate(order["steps"]):
            callback_comm.compression(input_chunk[src_rank], send_chunk)
            exchange(send_chunk, recv_chunk, left, right, wire_tag | step)
            _decompression_add(recv_chunk, input_chunk[dst_rank], callback_comm)

        src_rank = order["steps"][-1][2]
        left = (rank + 1) % size
        right = (rank - 1 + size) % size
        callback_comm.compression(input_chunk[src_rank], send_chunk)
        callback_comm.decompression(send_chunk, input_chunk[src_rank])

        chunks = [send_chunk, recv_chunk]
        for step in range(size - 1):
            exchange(chunks[step & 1], chunks[(step ^ 1) & 1], left, right, wire_tag | (size + step))
            callback_comm.decompression(
                chunks[(step ^ 1) & 1],
                input_chunk[(src_rank + step + 1) % size],
            )

    return send_vec
