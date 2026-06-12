import os

import torch
import torch.distributed as dist

from .utils import composable_allreduce_arbitrary_callback, composable_butterfly_allreduce_callback
from .butterfly_rdma import composable_butterfly_rdma_allreduce_callback
from .ring_rdma import (
    composable_ring_rdma_dynamiQ_allreduce_callback,
    composable_ring_rdma_allreduce_callback,
)
from .utils_aggregation import (
    Aee_Dynamic_Range_GPU_kern,
    BFloat16_compression,
    Float16_compression,
    Float32_compression,
    Float8_compression,
    Mee_Dynamic_Range_GPU_kern,
    superchunk_add_mean_copy,
    superchunk_mean_center,
)
from .compressor_hierarchical import NewINCACompressor
from .thc_hooks import P2P_THC_compress_hook

INTEG_PARTITION_LAYER = 1
CHUNK_SIZE_THRESHOLD = 3 << 29 # 1.5*2^30, 1.6G parameters


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() not in ("0", "false", "no", "off")


def _butterfly_callback():
    if _env_bool("BUTTERFLY_RDMA_DISABLE", False):
        return composable_butterfly_allreduce_callback
    return composable_butterfly_rdma_allreduce_callback


def _topology_callback(params):
    if "butterfly" in params["args"].aggregation_method:
        return _butterfly_callback()
    return composable_ring_rdma_allreduce_callback


DYNAMIC_BITRATE_HEADROOM = 0.625
DYNAMIC_BITRATE_U_MIN = -10.0
DYNAMIC_BITRATE_U_MAX = 60.0
DYNAMIC_BITRATE_TOL = 0.1
DYNAMIC_BITRATE_MAX_ITERS = 32
DYNAMIC_BITRATE_NORM_SCALE = 4.0 / (512.0 / 17.0)


def _dynamic_bitrate_target_bitrate(args):
    method = args.aggregation_method.lower()
    for nbits in (8, 7, 6, 5, 4, 3, 2):
        if f"{nbits}bit" in method:
            return float(nbits)
    return 5.0


def _dynamic_bitrate_qbits(norms, u):
    eps = 1e-20
    log_norms = torch.log2(torch.clamp(norms, min=eps))
    score = log_norms.mul(DYNAMIC_BITRATE_NORM_SCALE).add(float(u))
    z = torch.floor(torch.log2(torch.clamp(score, min=eps))).to(torch.int64)
    z.clamp_(1, 3)
    return torch.bitwise_left_shift(torch.ones_like(z), z)


def _dynamic_bitrate_average_bits(norms_list, u):
    total_chunks = 0
    total_bits = None
    for norms in norms_list:
        if norms.numel() == 0:
            continue
        qbits = _dynamic_bitrate_qbits(norms, u)
        qbits_sum = qbits.sum()
        total_bits = qbits_sum if total_bits is None else total_bits + qbits_sum
        total_chunks += qbits.numel()

    if total_bits is None or total_chunks == 0:
        return 0.0
    return float(total_bits.item()) / total_chunks


def _dynamic_bitrate_avg_delta(norms_list, u, target_avg_bits):
    avg_bits = _dynamic_bitrate_average_bits(norms_list, u)
    return avg_bits, target_avg_bits - avg_bits


def _dynamic_bitrate_binary_search_u(norms_list, target_avg_bits):
    low = DYNAMIC_BITRATE_U_MIN
    high = DYNAMIC_BITRATE_U_MAX
    best_u = high
    best_avg, best_delta = _dynamic_bitrate_avg_delta(norms_list, best_u, target_avg_bits)

    for _ in range(DYNAMIC_BITRATE_MAX_ITERS):
        mid = (low + high) * 0.5
        avg_bits, delta = _dynamic_bitrate_avg_delta(norms_list, mid, target_avg_bits)
        if abs(delta) < abs(best_delta):
            best_u = mid
            best_avg = avg_bits
            best_delta = delta
        if abs(delta) <= DYNAMIC_BITRATE_TOL:
            return mid, avg_bits, delta
        if delta > 0:
            low = mid
        else:
            high = mid

    return best_u, best_avg, best_delta


def _dynamic_bitrate_adjust_u(norms_list, prev_u, target_avg_bits):
    prev_u = max(DYNAMIC_BITRATE_U_MIN, min(DYNAMIC_BITRATE_U_MAX, float(prev_u)))
    old_avg, old_delta = _dynamic_bitrate_avg_delta(norms_list, prev_u, target_avg_bits)
    if abs(old_delta) <= DYNAMIC_BITRATE_TOL:
        return prev_u, old_avg, old_delta

    step = old_delta
    best_u = prev_u
    best_avg = old_avg
    best_delta = old_delta

    for _ in range(DYNAMIC_BITRATE_MAX_ITERS):
        candidate_u = max(DYNAMIC_BITRATE_U_MIN, min(DYNAMIC_BITRATE_U_MAX, prev_u + step))
        avg_bits, delta = _dynamic_bitrate_avg_delta(norms_list, candidate_u, target_avg_bits)
        if abs(delta) < abs(best_delta):
            best_u = candidate_u
            best_avg = avg_bits
            best_delta = delta
        if abs(delta) <= DYNAMIC_BITRATE_TOL or old_delta * delta >= 0:
            return candidate_u, avg_bits, delta
        step *= 0.5

    return best_u, best_avg, best_delta


def P2P_fp16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    def compress_and_reduce_and_decompress(fut, bucket, buffer, params):
        nelem = buffer.numel()
        comm_buff = buffer[:nelem]
        comm_buff = composable_allreduce_arbitrary_callback(comm_buff, params["callback_comm"], params, dtype=torch.float16, tag=bucket.index())
        return buffer

    group_to_use = process_group if process_group is not None else dist.group.WORLD
    params = group_to_use[0]["params"]
    world_size = group_to_use[0]["params"]["nclients"]

    if group_to_use[0]["batch_idx"] <= 1:
        params["callback_comm"] = Float16_compression(params)

        to_reduce = bucket.buffer() / dist.get_world_size()

        handle = dist.all_reduce(to_reduce, async_op=True).get_future()
        return handle.then(lambda fut: fut.value()[0])

    buffer = bucket.buffer()
    buffer.copy_(buffer.to(torch.float16).div_(world_size))

    init_fut = torch.futures.Future()
    init_fut.set_result(0)

    reduce_future = init_fut.then(lambda fut: compress_and_reduce_and_decompress(fut, bucket, buffer, params))

    return reduce_future



def P2P_bf16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    def compress_and_reduce_and_decompress(fut, bucket, buffer, params):
        nelem = buffer.numel()
        comm_buff = buffer
        callback = _topology_callback(params)

        comm_buff = callback(comm_buff, params["callback_comm"], params, dtype=torch.bfloat16, tag=bucket.index())
        return buffer

    group_to_use = process_group if process_group is not None else dist.group.WORLD
    params = group_to_use[0]["params"]
    world_size = group_to_use[0]["params"]["nclients"]

    if group_to_use[0]["batch_idx"] <= 1:
        if group_to_use[0]["batch_idx"] == 1:
            print(bucket.index(), bucket.buffer().numel())
        params["callback_comm"] = BFloat16_compression(params)

        to_reduce = bucket.buffer() / dist.get_world_size()
        print("index={}".format(bucket.index()))

        handle = dist.all_reduce(to_reduce, async_op=True).get_future()
        return handle.then(lambda fut: fut.value()[0])


    buffer = bucket.buffer()
    div_buffer = buffer.to(torch.bfloat16) / float(world_size)
    buffer.copy_(div_buffer)

    init_fut = torch.futures.Future()
    init_fut.set_result(0)

    reduce_future = init_fut.then(lambda fut: compress_and_reduce_and_decompress(fut, bucket, buffer, params))

    return reduce_future


def P2P_slicing_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    def compress_and_reduce_and_decompress(fut, bucket, buffer, sliced_portion, params):
        callback = _topology_callback(params)
        callback(buffer, params["callback_comm"], params, dtype=torch.bfloat16, tag=bucket.index(), sliced_portion=sliced_portion)
        return buffer

    group_to_use = process_group if process_group is not None else dist.group.WORLD
    params = group_to_use[0]["params"]
    world_size = group_to_use[0]["params"]["nclients"]

    if group_to_use[0]["batch_idx"] <= 1:
        params["callback_comm"] = BFloat16_compression(params)

        to_reduce = bucket.buffer() / dist.get_world_size()

        handle = dist.all_reduce(to_reduce, async_op=True).get_future()
        return handle.then(lambda fut: fut.value()[0])


    buffer = bucket.buffer()
    div_buffer = buffer.to(torch.bfloat16) / float(world_size)
    buffer.copy_(div_buffer)

    sliced_portion = (4.25 / 16) if "fp4" in params["args"].aggregation_method else (6.25 / 16)
    if "zero" in params["args"].aggregation_method:
        sliced_portion = 0.0

    init_fut = torch.futures.Future()
    init_fut.set_result(0)

    reduce_future = init_fut.then(lambda fut: compress_and_reduce_and_decompress(fut, bucket, buffer, sliced_portion, params))

    return reduce_future



def P2P_fp8_compress_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    MX_FP8_E4M3_MAX = 448.0

    def compress_and_reduce_and_decompress(fut, state, index, l, r, no_hadamard=False):
        padded_tensor_list = []
        norm_max_list = []
        max_reduce_handles = []

        for i in range(l, r):
            sl = state["start_idx"][(index, i)]
            sr = sl + state["params"]["size"][(index, i)]
            
            padded_tensor = vec[sl:sr]
            padded_tensor_list.append(padded_tensor)

            norm_max_memory = torch.amax(padded_tensor.abs())
            norm_max_list.append(norm_max_memory)
            
            max_reduce_handles.append(dist.all_reduce(norm_max_memory, async_op=True, op=dist.ReduceOp.MAX))

        for i in range(l, r):
            max_reduce_handles[i].wait()
            sl = state["start_idx"][(index, i)]
            sr = sl + state["params"]["size"][(index, i)]

            norm_max = norm_max_list[i]

            padded_tensor = padded_tensor_list[i]

            padded_tensor.div_(norm_max + 5e-7).mul_(MX_FP8_E4M3_MAX / dist.get_world_size())

            aggregated_tensor = composable_ring_rdma_allreduce_callback(padded_tensor, state["params"]["callback_comm"], state["params"], dtype=torch.float8_e4m3fn, tag=(index << 8))
            aggregated_tensor.mul_(dist.get_world_size() / MX_FP8_E4M3_MAX).mul_(norm_max + 5e-7)
            ret_tensor[sl:sr].copy_(aggregated_tensor.view(-1)[:sr - sl])
        
        return ret_tensor
    
    state = state[0]

    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    elif state["batch_idx"] == 1:
        
        index = bucket.index()
        print(index, bucket.buffer().numel())

        total_size = bucket.buffer().numel()
        orig_total_size = total_size
        start_interm_idx = 0

        i = 0
        while True:
            if i >= INTEG_PARTITION_LAYER - 1 and total_size <= CHUNK_SIZE_THRESHOLD:
                cur_size = total_size
                cur_d = cur_size
            else:
                cur_size = CHUNK_SIZE_THRESHOLD
                cur_d = cur_size

            state["params"]["d"][(index, i)] = cur_d
            state["params"]["size"][(index, i)] = cur_size
            state["start_idx"][(index, i)] = orig_total_size - total_size
            state["start_interm_idx"][(index, i)] = start_interm_idx
            
            start_interm_idx += cur_d

            total_size -= cur_size
            i += 1
            if total_size == 0:
                break

        state["partition_len"][index] = i

        state["params"]["chunk_size"] = 32
        
        if bucket.is_last():
            state["params"]["callback_comm"] = Float8_compression(state["params"])

            state["params"]["compressor"] = NewINCACompressor(state["params"])

        return (
            dist.all_reduce(bucket.buffer() / dist.get_world_size(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )
        
    else:
        
        index = bucket.index()
        
        vec = bucket.buffer()
        ret_tensor = bucket.buffer()


        init_future = torch.futures.Future()
        init_future.set_result(0)

        l = 0
        r = state["partition_len"][index]

        reduce_future = init_future.then(lambda fut: compress_and_reduce_and_decompress(fut, state, index, l, r, True))

        return reduce_future



def P2P_MXfp8_compress_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    MX_FP8_E4M3_MAX = 336

    def compress_and_reduce_and_decompress(fut, state, index, l, r, no_hadamard=False):
        max_chunk_size = state["params"]["chunk_size"]
        
        padded_tensor_list = []
        chunk_max_list = []
        chunk_max_reduce_handles = []

        for i in range(l, r):
            sl = state["start_idx"][(index, i)]
            sr = sl + state["params"]["size"][(index, i)]

            chunk_max_memory = state["params"]["compressor"].max_memory_tensor((index, i))
            padding_length = (sr - sl + max_chunk_size - 1) // max_chunk_size * max_chunk_size - (sr - sl)
            if padding_length > 0:
                padded_tensor = torch.nn.functional.pad(vec[sl:sr], (0, padding_length), "constant", 0)
            else:
                padded_tensor = vec[sl:sr]
            padded_tensor = padded_tensor.view(-1, max_chunk_size)
            padded_tensor_list.append(padded_tensor)

            torch.amax(padded_tensor.abs(), dim=1, keepdim=False, out=chunk_max_memory)
            chunk_max_list.append(chunk_max_memory.view(-1, 1))
            chunk_max_reduce_handles.append(
                dist.all_reduce(chunk_max_memory, async_op=True, op=dist.ReduceOp.MAX)
            )
            

        for i in range(l, r):
            sl = state["start_idx"][(index, i)]
            sr = sl + state["params"]["size"][(index, i)]

            chunk_max_reduce_handles[i].wait()
            chunk_max = chunk_max_list[i]
            padded_tensor = padded_tensor_list[i]

            chunk_max.add_(5e-7).mul_(MX_FP8_E4M3_MAX / dist.get_world_size())

            padded_tensor.div_(chunk_max)
            padded_tensor = padded_tensor.view(-1)

            aggregated_tensor = state["params"]["callback_func"](padded_tensor, state["params"]["callback_comm"], state["params"], dtype=torch.float8_e4m3fn, tag=(index << 8))
            aggregated_tensor.view(-1, max_chunk_size).mul_(chunk_max).div_(dist.get_world_size())
            ret_tensor[sl:sr].copy_(aggregated_tensor.view(-1)[:sr - sl])
        
        return ret_tensor
    
    state = state[0]

    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    elif state["batch_idx"] == 1:
        
        index = bucket.index()
        total_size = bucket.buffer().numel()
        orig_total_size = total_size
        start_interm_idx = 0

        i = 0
        while True:
            if i >= INTEG_PARTITION_LAYER - 1 and total_size <= CHUNK_SIZE_THRESHOLD:
                cur_size = total_size
                cur_d = cur_size
            else:
                cur_size = CHUNK_SIZE_THRESHOLD
                cur_d = cur_size

            state["params"]["d"][(index, i)] = cur_d
            state["params"]["size"][(index, i)] = cur_size
            state["start_idx"][(index, i)] = orig_total_size - total_size
            state["start_interm_idx"][(index, i)] = start_interm_idx
            
            start_interm_idx += cur_d

            total_size -= cur_size
            i += 1
            if total_size == 0:
                break

        state["partition_len"][index] = i

        state["params"]["chunk_size"] = 32
        
        if bucket.is_last():
            state["params"]["callback_comm"] = Float8_compression(state["params"])

            state["params"]["compressor"] = NewINCACompressor(state["params"])

            if "butterfly" in state["params"]["args"].aggregation_method:
                state["params"]["callback_func"] = _butterfly_callback()
            else:
                state["params"]["callback_func"] = composable_ring_rdma_allreduce_callback

        return (
            dist.all_reduce(bucket.buffer(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )
        
    else:
        
        index = bucket.index()
        
        vec = bucket.buffer()
        ret_tensor = bucket.buffer()


        init_future = torch.futures.Future()
        init_future.set_result(0)

        l = 0
        r = state["partition_len"][index]

        reduce_future = init_future.then(lambda fut: compress_and_reduce_and_decompress(fut, state, index, l, r, True))

        return reduce_future


def P2P_fp32_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    def compress_and_reduce_and_decompress(fut, bucket, buffer, params):
        buffer = composable_allreduce_arbitrary_callback(buffer, params["callback_comm"], params, dtype=torch.float32, tag=bucket.index())
        return buffer

    group_to_use = process_group if process_group is not None else dist.group.WORLD
    params = group_to_use[0]["params"]
    if group_to_use[0]["batch_idx"] <= 1:
        params["callback_comm"] = Float32_compression(params)
        return (
            dist.all_reduce(bucket.buffer() / dist.get_world_size(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    world_size = group_to_use[0]["params"]["nclients"]
    
    buffer = bucket.buffer()
    buffer.copy_(buffer.div_(world_size))

    init_fut = torch.futures.Future()
    init_fut.set_result(0)

    reduce_future = init_fut.then(lambda fut: compress_and_reduce_and_decompress(fut, bucket, buffer, params))

    return reduce_future


def P2P_omnireduce_topk_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    def compress_reduce_decompress(fut, state, index, l, r, vec, ret_tensor):
        fut.wait()
        max_chunk_size = state["params"]["max_chunk_size"]
        callback = state["params"]["callback_func"]
        callback_comm = state["params"]["callback_comm"]

        bitmap_list = []
        bitmap_handles = []
        padded_tensor_list = []

        total_chunk_cnt = 0
        dense_chunk_cnt = 0

        for part in range(l, r):
            sl = state["start_idx"][(index, part)]
            sr = sl + state["params"]["size"][(index, part)]

            padded_tensor, bitmap_memory = state["params"]["compressor"].padding_tensor(vec[sl:sr], (index, part))
            padded_tensor = padded_tensor.view(-1, max_chunk_size)

            local_norm = torch.norm(padded_tensor, p="fro", dim=1, keepdim=False)
            nchunks = local_norm.numel()
            topk_chunks = int(nchunks * state["params"]["local_topks"][(index, part)])
            topk_chunks = max(0, min(topk_chunks, nchunks))

            bitmap_memory.zero_()
            if topk_chunks > 0:
                _, topk_indices = torch.topk(local_norm, topk_chunks, largest=True)
                bitmap_memory[topk_indices] = 1

            bitmap_list.append(bitmap_memory)
            bitmap_handles.append(dist.all_reduce(bitmap_memory, async_op=True, op=dist.ReduceOp.SUM))
            padded_tensor_list.append(padded_tensor)

        for part in range(l, r):
            sl = state["start_idx"][(index, part)]
            sr = sl + state["params"]["size"][(index, part)]
            bitmap_handles[part - l].wait()

            padded_tensor = padded_tensor_list[part - l]
            bitmap_memory = bitmap_list[part - l]
            indice = bitmap_memory.nonzero().view(-1)

            dense_chunk_cnt += indice.numel()
            total_chunk_cnt += bitmap_memory.numel()

            if indice.numel() > 0:
                to_reduce_tensor = padded_tensor[indice].contiguous().view(-1)
                to_reduce_tensor = callback(
                    to_reduce_tensor,
                    callback_comm,
                    state["params"],
                    dtype=torch.bfloat16,
                    tag=(index << 8) | part,
                )
                padded_tensor.zero_()
                padded_tensor[indice] = to_reduce_tensor.view(-1, max_chunk_size)
            else:
                padded_tensor.zero_()

            ret_tensor[sl:sr].copy_(padded_tensor.view(-1)[: sr - sl])

            target_topk = state["params"]["target_topk"] * bitmap_memory.numel()
            if target_topk >= 1:
                real_topk = indice.numel()
                proportion = real_topk / target_topk
                proportion = max(0.6, min(proportion, 1 / 0.6))
                state["params"]["local_topks"][(index, part)] = (
                    0.75 * state["params"]["local_topks"][(index, part)]
                    + 0.25 * state["params"]["local_topks"][(index, part)] / proportion
                )

        if total_chunk_cnt == 0:
            state["params"]["sparsity"].append([0, 0, 0])
        else:
            sparse_chunks = total_chunk_cnt - dense_chunk_cnt
            state["params"]["sparsity"].append(
                [total_chunk_cnt, sparse_chunks, 1 - dense_chunk_cnt / total_chunk_cnt]
            )
            print(total_chunk_cnt, sparse_chunks, 1 - dense_chunk_cnt / total_chunk_cnt)
            state["params"]["sparsity"][0][0] += total_chunk_cnt
            state["params"]["sparsity"][0][1] += sparse_chunks
            state["params"]["sparsity"][0][2] = (
                0
                if state["params"]["sparsity"][0][0] == 0
                else state["params"]["sparsity"][0][1] / state["params"]["sparsity"][0][0]
            )

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
        total_size = bucket.buffer().numel()
        orig_total_size = total_size

        state["params"]["max_chunk_size"] = state["params"]["chunk_size"] = 256
        state["params"]["sparsity"] = [[0, 0, 0]]
        state["params"].setdefault("target_topk", 0.5)

        part = 0
        while total_size > 0:
            cur_size = min(total_size, CHUNK_SIZE_THRESHOLD)
            cur_d = cur_size

            state["params"]["d"][(index, part)] = (
                (cur_d + state["params"]["max_chunk_size"] - 1)
                // state["params"]["max_chunk_size"]
                * state["params"]["max_chunk_size"]
            )
            state["params"]["size"][(index, part)] = cur_size
            state["start_idx"][(index, part)] = orig_total_size - total_size

            total_size -= cur_size
            part += 1

        state["partition_len"][index] = part

        if bucket.is_last():
            state["params"]["compressor"] = NewINCACompressor(state["params"])
            state["params"]["local_topks"] = {}
            for key in state["params"]["d"]:
                state["params"]["local_topks"][key] = state["params"]["target_topk"] / dist.get_world_size()
            state["params"]["callback_comm"] = Float16_compression(state["params"])
            state["params"]["callback_func"] = _topology_callback(state["params"])

        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    index = bucket.index()
    vec = bucket.buffer()
    ret_tensor = bucket.buffer()

    if bucket.is_last() and state["params"]["sparsity"]:
        print("Sparsity: ", state["params"]["sparsity"][0], flush=True)

    l = 0
    r = state["partition_len"][index]

    init_future = dist.all_reduce(vec[0].norm(2).view(-1), async_op=True, op=dist.ReduceOp.MAX).get_future()
    return init_future.then(lambda fut: compress_reduce_decompress(fut, state, index, l, r, vec, ret_tensor))


def P2P_dynamiQ_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    def compress_and_reduce_and_decompress(fut, state, index, l, r, no_hadamard=False):
        superchunk_size = state["params"]["chunk_size"]
        residuals = {}
        stats_tensors = {}
        stat_reduce_handles = []
        world_size = dist.get_world_size()

        for i in range(l, r):
            sl = state["start_idx"][(index, i)]
            sr = sl + state["params"]["size"][(index, i)]
            n = sr - sl
            residual = state["dynamic_residual"][(index, i)]
            stats = state["dynamic_superchunk_stats"][(index, i)]
            superchunk_mean_center(vec[sl:sr], residual, stats, n, world_size, superchunk_size)
            residuals[i] = residual
            stats_tensors[i] = stats
            stat_reduce_handles.append(dist.all_reduce(stats, async_op=True, op=dist.ReduceOp.SUM))

        for handle in stat_reduce_handles:
            handle.wait()

        for i in range(l, r):
            stats = stats_tensors[i]
            residual = residuals[i]
            residual_2d = residual.view(-1, superchunk_size)
            norms = stats[:, 1].float()
            num_superchunks = norms.numel()

            top_count = int(num_superchunks * state["params"]["thresholds"][8])
            bottom_count = int(num_superchunks * state["params"]["thresholds"][2])
            if top_count + bottom_count > num_superchunks:
                bottom_count = max(0, num_superchunks - top_count)

            sorted_indices = torch.argsort(norms, descending=True, stable=True)
            indice_8 = sorted_indices[:top_count]
            indice_4 = sorted_indices[top_count:num_superchunks - bottom_count]
            indice_2 = sorted_indices[num_superchunks - bottom_count:] if bottom_count > 0 else sorted_indices[:0]

            def build_group(indices, nbits, tag_suffix):
                if indices.numel() == 0:
                    return None

                nrows = indices.numel()
                nelem = nrows * superchunk_size
                buffer_key = (index, i, nbits)
                buffer = state["dynamic_select_buffers"].get(buffer_key)
                if (
                    buffer is None
                    or buffer.numel() < nelem
                    or buffer.dtype != residual.dtype
                    or buffer.device != residual.device
                ):
                    buffer = torch.empty(nelem, dtype=residual.dtype, device=residual.device)
                    state["dynamic_select_buffers"][buffer_key] = buffer

                group_tensor = buffer[:nelem]
                return {
                    "indices": indices,
                    "nbits": nbits,
                    "tag_suffix": tag_suffix,
                    "nrows": nrows,
                    "group_tensor": group_tensor,
                }

            def communicate_group(group):
                indices = group["indices"]
                nbits = group["nbits"]
                tag_suffix = group["tag_suffix"]
                nrows = group["nrows"]
                group_tensor = group["group_tensor"]
                torch.index_select(residual_2d, 0, indices, out=group_tensor.view(nrows, superchunk_size))
                if "no_compress" not in state["params"]["args"].aggregation_method:
                    group_tensor = state["params"]["callback_func"](
                        group_tensor,
                        state["params"][f"callback_comm_{nbits}bit"],
                        state["params"],
                        dtype=torch.uint8,
                        tag=(index << 10) | (i << 2) | tag_suffix,
                    )
                residual_2d.index_copy_(0, indices, group_tensor.view(nrows, superchunk_size))

            group_specs = []
            if "fixed" not in state["args"].aggregation_method:
                group_specs.append(build_group(indice_8, 8, 0))
            group_specs.append(build_group(indice_4, 4, 1))
            if "fixed" not in state["args"].aggregation_method:
                group_specs.append(build_group(indice_2, 2, 2))
            group_specs = [group for group in group_specs if group is not None]

            for group in group_specs:
                communicate_group(group)

            sl = state["start_idx"][(index, i)]
            sr = sl + state["params"]["size"][(index, i)]
            superchunk_add_mean_copy(residual, ret_tensor[sl:sr], stats, sr - sl, superchunk_size)
        
        return ret_tensor
    

    state = state[0]


    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer() / dist.get_world_size(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    elif state["batch_idx"] == 1:
        local_rank = dist.get_rank()
        print(local_rank, bucket.index(), bucket.buffer().numel())
        
        index = bucket.index()
        total_size = bucket.buffer().numel()
        orig_total_size = total_size
        start_interm_idx = 0
        state.setdefault("dynamic_select_buffers", {})

        i = 0
        while True:
            if i >= INTEG_PARTITION_LAYER - 1 and total_size <= CHUNK_SIZE_THRESHOLD:
                cur_size = total_size
                cur_d = cur_size
            else:
                cur_size = CHUNK_SIZE_THRESHOLD
                cur_d = cur_size

            state["params"]["d"][(index, i)] = cur_d
            state["params"]["size"][(index, i)] = cur_size
            state["start_idx"][(index, i)] = orig_total_size - total_size
            state["start_interm_idx"][(index, i)] = start_interm_idx
            padded_size = (cur_size + 255) // 256 * 256
            num_superchunks = padded_size // 256
            state.setdefault("dynamic_residual", {})[(index, i)] = torch.empty(
                padded_size,
                dtype=bucket.buffer().dtype,
                device=bucket.buffer().device,
            )
            state.setdefault("dynamic_superchunk_stats", {})[(index, i)] = torch.empty(
                (num_superchunks, 2),
                dtype=torch.bfloat16,
                device=bucket.buffer().device,
            )

            start_interm_idx += cur_d

            total_size -= cur_size
            i += 1
            if total_size == 0:
                break
        state["partition_len"][index] = i

        state["params"]["chunk_size"] = 16
        
        if bucket.is_last():
            print(state["params"]["d"])
            state["params"]["thresholds"] = {8: 0.2, 4: 0.6, 2: 0.2}
            if "4bit" in state["args"].aggregation_method:
                if "fixed" in state["args"].aggregation_method:
                    state["params"]["thresholds"] = {4: 1, 8: 0, 2: 0}
                else:
                    state["params"]["thresholds"] = {8: 0.05, 4: 0.6, 2: 0.35}
            
            elif "3bit" in state["args"].aggregation_method:
                state["params"]["thresholds"] = {8: 1 / 40, 4: 13 / 40, 2: 26 / 40} 
            elif "6bit" in state["args"].aggregation_method:
                state["params"]["thresholds"] = {8: 0.55, 4: 0.35, 2: 0.1}
            elif "7bit" in state["args"].aggregation_method:
                state["params"]["thresholds"] = {8: 0.75, 4: 0.22, 2: 0.03}
            else:
                state["params"]["thresholds"] = {8: 0.35, 4: 0.45, 2: 0.2}


            state["params"]["agg_chunk_size"] = state["params"]["chunk_size"]
            state["params"]["supergroup"] = 16
            
            state["params"]["chunk_size"] = state["params"]["agg_chunk_size"] * state["params"]["supergroup"]
            
            if "aee" in state["params"]["args"].aggregation_method:
                state["params"]["callback_comm_2bit"] = Aee_Dynamic_Range_GPU_kern(2, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_4bit"] = Aee_Dynamic_Range_GPU_kern(4, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_8bit"] = Aee_Dynamic_Range_GPU_kern(8, state["params"]["agg_chunk_size"], state["params"])
            else:
                state["params"]["callback_comm_2bit"] = Mee_Dynamic_Range_GPU_kern(2, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_4bit"] = Mee_Dynamic_Range_GPU_kern(4, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_8bit"] = Mee_Dynamic_Range_GPU_kern(8, state["params"]["agg_chunk_size"], state["params"])
            if "butterfly" in state["params"]["args"].aggregation_method:
                state["params"]["callback_func"] = _butterfly_callback()
            elif _env_bool("DYNAMIC_AEE_PIPELINE_RDMA", False):
                state["params"]["callback_func"] = composable_ring_rdma_dynamiQ_allreduce_callback
            else:
                state["params"]["callback_func"] = composable_ring_rdma_allreduce_callback

        return (
            dist.all_reduce(bucket.buffer() / dist.get_world_size(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )
        
    else:
        index = bucket.index()
        
        vec = bucket.buffer()
        ret_tensor = bucket.buffer()


        init_future = torch.futures.Future()
        init_future.set_result(0)

        l = 0
        r = state["partition_len"][index]

        reduce_future = init_future.then(lambda fut: compress_and_reduce_and_decompress(fut, state, index, l, r, True))

        return reduce_future


def P2P_dynamiQ_dynamic_bitrate_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    def update_next_u_if_last_bucket(is_last_bucket):
        if not is_last_bucket:
            return

        norms_list = state.get("dynamic_bitrate_norms", [])
        if not norms_list:
            state["dynamic_bitrate_norms"] = []
            return

        target_avg_bits = state["params"].setdefault(
            "dynamic_bitrate_target_avg_bits",
            _dynamic_bitrate_target_bitrate(state["params"]["args"]) - DYNAMIC_BITRATE_HEADROOM,
        )
        prev_u = state["params"].get("dynamic_bitrate_u", DYNAMIC_BITRATE_U_MAX)

        if state["batch_idx"] == 2:
            next_u, avg_bits, delta = _dynamic_bitrate_binary_search_u(norms_list, target_avg_bits)
        else:
            next_u, avg_bits, delta = _dynamic_bitrate_adjust_u(norms_list, prev_u, target_avg_bits)

        state["params"]["dynamic_bitrate_u"] = float(next_u)
        state["params"]["dynamic_bitrate_last_avg_bits"] = float(avg_bits)
        state["params"]["dynamic_bitrate_last_delta"] = float(delta)
        state["dynamic_bitrate_norms"] = []

    def compress_and_reduce_and_decompress(
        fut, state, index, l, r, is_last_bucket=False, no_hadamard=False
    ):
        superchunk_size = state["params"]["chunk_size"]
        residuals = {}
        stats_tensors = {}
        stat_reduce_handles = []
        world_size = dist.get_world_size()
        current_u = state["params"].get("dynamic_bitrate_u", DYNAMIC_BITRATE_U_MAX)

        for i in range(l, r):
            sl = state["start_idx"][(index, i)]
            sr = sl + state["params"]["size"][(index, i)]
            n = sr - sl
            residual = state["dynamic_residual"][(index, i)]
            stats = state["dynamic_superchunk_stats"][(index, i)]
            superchunk_mean_center(vec[sl:sr], residual, stats, n, world_size, superchunk_size)
            residuals[i] = residual
            stats_tensors[i] = stats
            stat_reduce_handles.append(dist.all_reduce(stats, async_op=True, op=dist.ReduceOp.SUM))

        for handle in stat_reduce_handles:
            handle.wait()

        for i in range(l, r):
            stats = stats_tensors[i]
            residual = residuals[i]
            residual_2d = residual.view(-1, superchunk_size)
            norms = stats[:, 1]
            state.setdefault("dynamic_bitrate_norms", []).append(norms)

            qbits = _dynamic_bitrate_qbits(norms, current_u)
            indice_8 = torch.nonzero(qbits == 8, as_tuple=False).view(-1)
            indice_4 = torch.nonzero(qbits == 4, as_tuple=False).view(-1)
            indice_2 = torch.nonzero(qbits == 2, as_tuple=False).view(-1)

            def build_group(indices, nbits, tag_suffix):
                if indices.numel() == 0:
                    return None
                nrows = indices.numel()
                nelem = nrows * superchunk_size
                buffer_key = (index, i, nbits)
                buffer = state["dynamic_select_buffers"].get(buffer_key)
                if (
                    buffer is None
                    or buffer.numel() < nelem
                    or buffer.dtype != residual.dtype
                    or buffer.device != residual.device
                ):
                    buffer = torch.empty(nelem, dtype=residual.dtype, device=residual.device)
                    state["dynamic_select_buffers"][buffer_key] = buffer

                group_tensor = buffer[:nelem]
                return {
                    "indices": indices,
                    "nbits": nbits,
                    "tag_suffix": tag_suffix,
                    "nrows": nrows,
                    "group_tensor": group_tensor,
                }

            def communicate_group(group):
                indices = group["indices"]
                nbits = group["nbits"]
                tag_suffix = group["tag_suffix"]
                nrows = group["nrows"]
                group_tensor = group["group_tensor"]
                torch.index_select(residual_2d, 0, indices, out=group_tensor.view(nrows, superchunk_size))
                if "no_compress" not in state["params"]["args"].aggregation_method:
                    group_tensor = state["params"]["callback_func"](
                        group_tensor,
                        state["params"][f"callback_comm_{nbits}bit"],
                        state["params"],
                        dtype=torch.uint8,
                        tag=(index << 10) | (i << 2) | tag_suffix,
                    )
                residual_2d.index_copy_(0, indices, group_tensor.view(nrows, superchunk_size))

            group_specs = [
                build_group(indice_8, 8, 0),
                build_group(indice_4, 4, 1),
                build_group(indice_2, 2, 2),
            ]
            group_specs = [group for group in group_specs if group is not None]

            for group in group_specs:
                communicate_group(group)

            sl = state["start_idx"][(index, i)]
            sr = sl + state["params"]["size"][(index, i)]
            superchunk_add_mean_copy(residual, ret_tensor[sl:sr], stats, sr - sl, superchunk_size)

        update_next_u_if_last_bucket(is_last_bucket)
        return ret_tensor

    state = state[0]

    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer() / dist.get_world_size(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    elif state["batch_idx"] == 1:
        local_rank = dist.get_rank()
        print(local_rank, bucket.index(), bucket.buffer().numel())

        index = bucket.index()
        total_size = bucket.buffer().numel()
        orig_total_size = total_size
        start_interm_idx = 0
        state.setdefault("dynamic_select_buffers", {})

        i = 0
        while True:
            if i >= INTEG_PARTITION_LAYER - 1 and total_size <= CHUNK_SIZE_THRESHOLD:
                cur_size = total_size
                cur_d = cur_size
            else:
                cur_size = CHUNK_SIZE_THRESHOLD
                cur_d = cur_size

            state["params"]["d"][(index, i)] = cur_d
            state["params"]["size"][(index, i)] = cur_size
            state["start_idx"][(index, i)] = orig_total_size - total_size
            state["start_interm_idx"][(index, i)] = start_interm_idx
            padded_size = (cur_size + 255) // 256 * 256
            num_superchunks = padded_size // 256
            state.setdefault("dynamic_residual", {})[(index, i)] = torch.empty(
                padded_size,
                dtype=bucket.buffer().dtype,
                device=bucket.buffer().device,
            )
            state.setdefault("dynamic_superchunk_stats", {})[(index, i)] = torch.empty(
                (num_superchunks, 2),
                dtype=torch.bfloat16,
                device=bucket.buffer().device,
            )

            start_interm_idx += cur_d

            total_size -= cur_size
            i += 1
            if total_size == 0:
                break
        state["partition_len"][index] = i

        state["params"]["chunk_size"] = 16

        if bucket.is_last():
            print(state["params"]["d"])
            target_bitrate = _dynamic_bitrate_target_bitrate(state["params"]["args"])
            state["params"]["dynamic_bitrate_target_bitrate"] = target_bitrate
            state["params"]["dynamic_bitrate_target_avg_bits"] = target_bitrate - DYNAMIC_BITRATE_HEADROOM
            state["params"]["dynamic_bitrate_u"] = DYNAMIC_BITRATE_U_MAX
            state["dynamic_bitrate_norms"] = []
            state["dynamic_bitrate_stats_batch_idx"] = None

            state["params"]["agg_chunk_size"] = state["params"]["chunk_size"]
            state["params"]["supergroup"] = 16

            state["params"]["chunk_size"] = state["params"]["agg_chunk_size"] * state["params"]["supergroup"]

            if "aee" in state["params"]["args"].aggregation_method:
                state["params"]["callback_comm_2bit"] = Aee_Dynamic_Range_GPU_kern(2, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_4bit"] = Aee_Dynamic_Range_GPU_kern(4, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_8bit"] = Aee_Dynamic_Range_GPU_kern(8, state["params"]["agg_chunk_size"], state["params"])
            else:
                state["params"]["callback_comm_2bit"] = Mee_Dynamic_Range_GPU_kern(2, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_4bit"] = Mee_Dynamic_Range_GPU_kern(4, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_8bit"] = Mee_Dynamic_Range_GPU_kern(8, state["params"]["agg_chunk_size"], state["params"])
            if "butterfly" in state["params"]["args"].aggregation_method:
                state["params"]["callback_func"] = _butterfly_callback()
            elif _env_bool("DYNAMIC_AEE_PIPELINE_RDMA", False):
                state["params"]["callback_func"] = composable_ring_rdma_dynamiQ_allreduce_callback
            else:
                state["params"]["callback_func"] = composable_ring_rdma_allreduce_callback

        return (
            dist.all_reduce(bucket.buffer() / dist.get_world_size(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    else:
        index = bucket.index()

        vec = bucket.buffer()
        ret_tensor = bucket.buffer()

        state["params"].setdefault("dynamic_bitrate_u", DYNAMIC_BITRATE_U_MAX)
        state["params"].setdefault(
            "dynamic_bitrate_target_avg_bits",
            _dynamic_bitrate_target_bitrate(state["params"]["args"]) - DYNAMIC_BITRATE_HEADROOM,
        )
        if state.get("dynamic_bitrate_stats_batch_idx") != state["batch_idx"]:
            state["dynamic_bitrate_norms"] = []
            state["dynamic_bitrate_stats_batch_idx"] = state["batch_idx"]

        init_future = torch.futures.Future()
        init_future.set_result(0)

        l = 0
        r = state["partition_len"][index]

        reduce_future = init_future.then(
            lambda fut: compress_and_reduce_and_decompress(
                fut, state, index, l, r, bucket.is_last(), True
            )
        )

        return reduce_future
