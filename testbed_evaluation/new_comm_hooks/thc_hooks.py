import ast
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from . import eden_utils
from .butterfly_rdma import composable_butterfly_rdma_allreduce_callback
from .ring_rdma import composable_ring_rdma_allreduce_callback
from .utils import composable_butterfly_allreduce_callback
from .utils_aggregation import Direct_Summation


INTEG_PARTITION_LAYER = 1
CHUNK_SIZE_THRESHOLD = 1 << 30


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() not in ("0", "false", "no", "off")


def _topology_callback(params):
    if "butterfly" not in params["args"].aggregation_method:
        return composable_ring_rdma_allreduce_callback
    if _env_bool("BUTTERFLY_RDMA_DISABLE", False):
        return composable_butterfly_allreduce_callback
    return composable_butterfly_rdma_allreduce_callback


def _next_power_of_two_at_least_64(n: int) -> int:
    if n <= 64:
        return 64
    return 1 << ((n - 1).bit_length())


def _resolve_table_data_path(params) -> Path:
    table_size = params.get("table_size", 10001)
    max_val = params.get("max_val", 42)
    overflow_frequency = params.get("overflow_frequency", 1024)
    stem = f"{table_size}_tablesize_{max_val}_maxval_16_qlevels_{overflow_frequency}_ofreq_"

    root = Path(__file__).resolve().parents[2]
    candidates = []
    if os.environ.get("THC_TABLE_DIR"):
        candidates.append(Path(os.environ["THC_TABLE_DIR"]))
    if params.get("table_dir"):
        table_dir = Path(params["table_dir"])
        candidates.extend([table_dir, root / table_dir])
    candidates.extend(
        [
            root / "simulations_llm" / "compression" / "new_tables",
            Path("/home/wenchhan/cloned_thc_simu/simulation_llm/compression/new_tables"),
        ]
    )

    for table_dir in candidates:
        path = table_dir / f"{stem}data.txt"
        if path.exists():
            return path

    searched = ", ".join(str(path / f"{stem}data.txt") for path in candidates)
    raise FileNotFoundError(
        "THC oldINCA table metadata not found. Set THC_TABLE_DIR to a directory containing "
        f"{stem}data.txt. Searched: {searched}"
    )


def _load_table_data(params):
    data_path = _resolve_table_data_path(params)
    with data_path.open("r", encoding="utf-8") as fin:
        data = ast.literal_eval(fin.read())
    if "T" not in data:
        raise KeyError(f"THC table metadata at {data_path} does not contain key 'T'")
    return data


class EdenHadamardRHT:
    def __init__(self, dim: int, seed: int, device: str, key):
        if dim & (dim - 1):
            raise ValueError("eden_utils.Hadamard requires a power-of-two dimension")
        self.dim = dim
        self.device = device
        self.key = key
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed((seed + 1000003 * int(key[0]) + 9176 * int(key[1])) & 0xFFFFFFFF)
        self.random_diagonal = torch.empty(dim, dtype=torch.float32, device=device)
        self.random_diagonal.bernoulli_(0.5, generator=self.generator)
        self.random_diagonal.mul_(2.0).sub_(1.0)

    def rht(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.numel() != self.dim:
            raise ValueError(f"expected {self.dim} elements, got {vec.numel()}")
        vec = vec.mul_(self.random_diagonal)
        vec = eden_utils.Hadamard(vec, torch.cuda.current_stream().cuda_stream)
        return vec.div_(math.sqrt(self.dim))

    def irht(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.numel() != self.dim:
            raise ValueError(f"expected {self.dim} elements, got {vec.numel()}")
        vec = eden_utils.Hadamard(vec, torch.cuda.current_stream().cuda_stream)
        vec.div_(math.sqrt(self.dim))
        return vec.mul_(self.random_diagonal)


class OldINCACudaHadamardCompressor:
    def __init__(self, params):
        try:
            eden_utils.Hadamard_init()
        except AttributeError:
            pass

        self.device = params.get("device", "cuda")
        self.ds = params["d"]
        self.original_size = params["size"]
        self.seed = params.get("seed", 42)
        self.quantization_levels = params.get("quantization_levels", 64)
        self.nclients = params.get("nclients", dist.get_world_size())
        self.data = _load_table_data(params)
        self.max_norm_dict = {}

        self.sender_prng = torch.Generator(device=self.device)
        self.sender_prng.manual_seed((self.seed + dist.get_rank() * 104729) & 0xFFFFFFFF)

        self.hadamards = {
            key: EdenHadamardRHT(dim, self.seed, self.device, key)
            for key, dim in self.ds.items()
        }
        self.work_buffers = {}

    def _work_buffer(self, name, dtype=torch.float32):
        dim = self.ds[name]
        buf = self.work_buffers.get((name, dtype))
        if buf is None or buf.numel() != dim or buf.dtype != dtype:
            buf = torch.empty(dim, dtype=dtype, device=self.device)
            self.work_buffers[(name, dtype)] = buf
        return buf

    def _coordinate_range(self, max_norm, dim):
        max_coordinate = self.data["T"] * max_norm / math.sqrt(dim)
        min_coordinate = -max_coordinate
        delta = (max_coordinate - min_coordinate) / (self.quantization_levels - 1) + 1e-23
        return min_coordinate, delta

    def compress(self, tensor: torch.Tensor, name, max_norm: torch.Tensor) -> torch.Tensor:
        orig_size = self.original_size[name]
        dim = self.ds[name]
        self.max_norm_dict[name] = max_norm

        work = self._work_buffer(name)
        work[:orig_size].copy_(tensor.float())
        if dim > orig_size:
            work[orig_size:].zero_()

        rotated = self.hadamards[name].rht(work)
        min_coordinate, delta = self._coordinate_range(max_norm, dim)

        rotated.sub_(min_coordinate).div_(delta)
        rotated.clamp_(min=0, max=self.quantization_levels - 1)
        floored = rotated.floor()
        probs = rotated.sub_(floored)
        floored.add_(probs.bernoulli_(generator=self.sender_prng))
        return floored.to(torch.bfloat16)

    def decompress(self, tensor: torch.Tensor, name, max_norm: torch.Tensor = None) -> torch.Tensor:
        if max_norm is None:
            max_norm = self.max_norm_dict[name]

        dim = self.ds[name]
        orig_size = self.original_size[name]
        min_coordinate, delta = self._coordinate_range(max_norm, dim)

        work = self._work_buffer(name)
        work.copy_(tensor.float())
        work.mul_(delta / self.nclients).add_(min_coordinate)
        restored = self.hadamards[name].irht(work)
        return restored[:orig_size]


def _partition_bucket(state, index: int, total_size: int, dtype, device) -> None:
    orig_total_size = total_size
    start_interm_idx = 0
    part = 0

    while total_size > 0:
        if part >= INTEG_PARTITION_LAYER - 1 and total_size <= CHUNK_SIZE_THRESHOLD:
            cur_size = total_size
        else:
            cur_size = CHUNK_SIZE_THRESHOLD

        cur_d = _next_power_of_two_at_least_64(cur_size)
        key = (index, part)
        state["params"]["d"][key] = cur_d
        state["params"]["size"][key] = cur_size
        state["start_idx"][key] = orig_total_size - total_size
        state["start_interm_idx"][key] = start_interm_idx

        start_interm_idx += cur_d
        total_size -= cur_size
        part += 1

    state["partition_len"][index] = part


def P2P_THC_compress_hook(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    def compress_reduce_decompress(fut, state, index, l, r, vec, ret_tensor):
        fut.wait()
        callback = state["params"]["callback_func"]
        callback_comm = state["params"]["callback_comm"]
        compressor = state["params"]["compressor"]

        for part in range(l, r):
            sl = state["start_idx"][(index, part)]
            sr = sl + state["params"]["size"][(index, part)]
            tensor = vec[sl:sr]

            norm_tensor = tensor.float().norm(2)
            dist.all_reduce(norm_tensor, async_op=False, op=dist.ReduceOp.MAX)

            first_coordinate = tensor[0].float().clone().view(1)
            dist.all_reduce(first_coordinate, async_op=False, op=dist.ReduceOp.SUM)
            first_coordinate.div_(dist.get_world_size())

            compressed_tensor = compressor.compress(tensor, (index, part), norm_tensor)
            aggregated_tensor = callback(
                compressed_tensor,
                callback_comm,
                state["params"],
                dtype=torch.bfloat16,
                tag=(index << 8) | part,
            )
            aggregated_tensor = compressor.decompress(aggregated_tensor, (index, part), norm_tensor)
            aggregated_tensor[0] = first_coordinate[0]
            ret_tensor[sl:sr].copy_(aggregated_tensor.view(-1)[: sr - sl])

        return ret_tensor

    state = state[0]

    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    if state["batch_idx"] == 1:
        index = bucket.index()
        _partition_bucket(state, index, bucket.buffer().numel(), bucket.buffer().dtype, bucket.buffer().device)
        state["params"]["chunk_size"] = 64

        if bucket.is_last():
            state["params"]["callback_comm"] = Direct_Summation(state["params"])
            state["params"]["callback_func"] = _topology_callback(state["params"])
            state["params"]["compressor"] = OldINCACudaHadamardCompressor(state["params"])

        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    index = bucket.index()
    vec = bucket.buffer()
    ret_tensor = bucket.buffer()
    init_future = torch.futures.Future()
    init_future.set_result(0)
    return init_future.then(
        lambda fut: compress_reduce_decompress(
            fut,
            state,
            index,
            0,
            state["partition_len"][index],
            vec,
            ret_tensor,
        )
    )
