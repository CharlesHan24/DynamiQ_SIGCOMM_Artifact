from pathlib import Path

import torch
import torch.distributed as dist

from .utils import composable_allreduce_arbitrary_callback, composable_butterfly_allreduce_callback
from .utils_aggregation import Float8_compression, BFloat16_compression, Mee_Dynamic_Range, Mee_Dynamic_Range_Hierarchical, Fpx_arithmetics
from .compressor_hierarchical import NewINCACompressor
from .thc_hooks import P2P_THC_compress_hook


INTEG_PARTITION_LAYER = 1
CHUNK_SIZE_THRESHOLD = 1 << 28 # 2^23 for language tasks and 2^24 for image tasks


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float64).cpu().item()
    return float(value)


def _write_comm_error_vnmse(state):
    params = state["params"]
    args = state["args"]
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        return

    l2_error = _to_float(params["sum_l2_error"])
    l2_norm = _to_float(params["sum_l2_norm"])
    pred_l2_norm = _to_float(params["sum_l2_pred_norm"])
    vnmse = l2_error / (l2_norm + 1e-11)
    pred_norm_ratio = pred_l2_norm / (l2_norm + 1e-11)

    if "comm_error_vnmse_file" not in params:
        out_path = Path(output_dir) / "comm_error_vnmse.tsv"
        needs_header = not out_path.exists() or out_path.stat().st_size == 0
        params["comm_error_vnmse_file"] = open(out_path, "a")
        if needs_header:
            params["comm_error_vnmse_file"].write(
                "batch_idx\taggregation_method\tl2_error\tl2_norm\tpred_l2_norm\tvnmse\tpred_norm_ratio\n"
            )

    comm_error_file = params["comm_error_vnmse_file"]
    comm_error_file.write(
        f"{state['batch_idx']}\t{args.aggregation_method}\t"
        f"{l2_error:.12e}\t{l2_norm:.12e}\t{pred_l2_norm:.12e}\t"
        f"{vnmse:.12e}\t{pred_norm_ratio:.12e}\n"
    )
    comm_error_file.flush()

def normalize_mean(tensor, dim, chunk_size):
    tensor = tensor.reshape(dim // chunk_size, chunk_size)
    mean_tensor = tensor.mean(dim=1, keepdim=True)

    tensor = tensor - mean_tensor
    tensor = tensor.reshape(-1)

    return tensor, mean_tensor

def add_back_mean(tensor, mean_tensor, dim, chunk_size):
    tensor = tensor.reshape(dim // chunk_size, chunk_size)
    tensor += mean_tensor
    tensor = tensor.reshape(-1)
    return tensor

def P2P_fp16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    group_to_use = process_group if process_group is not None else dist.group.WORLD
    
    world_size = group_to_use["params"]["nclients"]
    compressed_tensor = bucket.buffer().to(torch.float16).div_(world_size)

    


    fut = dist.all_reduce(
        compressed_tensor, async_op=True
    ).get_future()

    def decompress(fut):
        fut.wait()
        decompressed_tensor = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        copied_tensor = fut.value()[0]
        decompressed_tensor[:copied_tensor.numel()].copy_(copied_tensor)
        return decompressed_tensor

    return fut.then(decompress)

def P2P_bf16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    
    world_size = group_to_use["params"]["nclients"]
    compressed_tensor = bucket.buffer().to(torch.bfloat16).div_(world_size)

    if "butterfly" in group_to_use["args"].aggregation_method:
        params = group_to_use["params"]
        init_future = torch.futures.Future()
        init_future.set_result(0)

        buffer = bucket.buffer()
        buffer.copy_(compressed_tensor)

        def compress_and_decompress(fut, bucket, buffer, params):
            buffer = composable_allreduce_arbitrary_callback(buffer, params["callback_comm"], params, dtype=torch.bfloat16, tag=bucket.index())
            return buffer
        
        if "callback_comm" not in params:
            params["callback_comm"] = BFloat16_compression(params)
        fut = init_future.then(lambda fut: compress_and_decompress(fut, bucket, buffer, params))

        return fut



    fut = dist.all_reduce(
        compressed_tensor, async_op=True
    ).get_future()

    def decompress(fut):
        fut.wait()
        decompressed_tensor = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        copied_tensor = fut.value()[0]
        decompressed_tensor[:copied_tensor.numel()].copy_(copied_tensor)
        return decompressed_tensor

    return fut.then(decompress)


def P2P_MXfp8_compress_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    # MX_FP8_E4M3_MAX = 448.0
    MX_FP8_E4M3_MAX = 335 # be a bit more conservative to prevent NAN overflows.

    def compress_and_reduce_and_decompress(fut, state, index, l, r, no_hadamard=False):
        fut.wait()
        max_chunk_size = state["params"]["chunk_size"]
        
        try:
            src_tensors = []
            all_reduce_handles = []

            padded_tensor_list = []
            norm_max_list = []
            max_reduce_handles = []
            for i in range(l, r):
                sl = state["start_idx"][(index, i)]
                sr = sl + state["params"]["size"][(index, i)]
                
                padded_tensor, norm_max_memory = state["params"]["compressor"].padding_tensor(vec[sl:sr], (index, i))   # division done here!!!
                padded_tensor = padded_tensor.view(-1, max_chunk_size)
                padded_tensor_list.append(padded_tensor)

                torch.amax(padded_tensor.abs(), dim=1, keepdim=False, out=norm_max_memory)
                norm_max_memory = norm_max_memory.view(-1, 1)
                norm_max_list.append(norm_max_memory)
                
                max_reduce_handles.append(dist.all_reduce(norm_max_memory, async_op=True, op=dist.ReduceOp.MAX))

            for i in range(l, r):
                max_reduce_handles[i].wait()
                sl = state["start_idx"][(index, i)]
                sr = sl + state["params"]["size"][(index, i)]

                norm_max = norm_max_list[i]

                padded_tensor = padded_tensor_list[i]

                padded_tensor.div_(norm_max + 5e-7).mul_(MX_FP8_E4M3_MAX / dist.get_world_size()) # MX_FP8_E4M3_MAX  / dist.get_world_size() is the maximum value to scale the tensor up.
                padded_tensor = padded_tensor.view(-1)

                aggregated_tensor = state["params"]["callback_func"](padded_tensor, state["params"]["callback_comm"], state["params"], dtype=torch.float8_e4m3fn, tag=(index << 8))
                aggregated_tensor.view(-1, max_chunk_size).mul_(dist.get_world_size() / MX_FP8_E4M3_MAX * state["params"]["ratio"]).mul_(norm_max + 5e-7)
                
                ret_tensor[sl:sr].copy_(aggregated_tensor.view(-1)[:sr - sl])
            
            return ret_tensor
                

        except Exception:
            raise
    
    def return_func(fut):
        return ret_tensor
    

    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
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
            # if i >= INTEG_PARTITION_LAYER - 1:
                cur_size = total_size
                cur_d = cur_size
            else:
                cur_size = CHUNK_SIZE_THRESHOLD
                # cur_size = 1 << (total_size.bit_length() - 1)
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

        # state["params"]["chunk_size"] = 64
        # state["params"]["max_chunk_size"] = 64
        # state["params"]["agg_chunk_size"] = 64
        state["params"]["chunk_size"] = 32
        state["params"]["max_chunk_size"] = 32
        state["params"]["agg_chunk_size"] = 32
        
        if bucket.is_last():
            state["params"]["callback_comm"] = Float8_compression(state["params"])

            state["params"]["compressor"] = NewINCACompressor(state["params"])

            if "butterfly" in state["args"].aggregation_method:
                state["params"]["callback_func"] = composable_butterfly_allreduce_callback
            else:
                state["params"]["callback_func"] = composable_allreduce_arbitrary_callback

        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )
        
    else:
        index = bucket.index()
        
        vec = bucket.buffer()
        total_size = bucket.buffer().numel()
        ret_tensor = bucket.buffer()


        init_future = torch.futures.Future()
        init_future.set_result(0)

        l = 0
        r = state["partition_len"][index]

        reduce_future = init_future.then(lambda fut: compress_and_reduce_and_decompress(fut, state, index, l, r, True)) #bucket.is_last()))

        return reduce_future


def P2P_dynamiQ_fixed_percentage_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    def compress_and_reduce_and_decompress(fut, state, index, l, r, bucket):
        fut.wait()
        max_chunk_size = state["params"]["chunk_size"]
        norm2_sum_chunk_size = state["params"]["max_chunk_size"]

        try:
            padded_tensor_list = []
            norm2_sum_list = []
            mean_tensor_list = []

            for i in range(l, r):
                sl = state["start_idx"][(index, i)]
                sr = sl + state["params"]["size"][(index, i)]
                
                norm2_sum_memory = state["params"]["compressor"].max_memory_tensor((index, i))
                padding_length = (sr - sl + max_chunk_size - 1) // max_chunk_size * max_chunk_size - (sr - sl)
                if padding_length > 0:
                    padded_tensor = torch.nn.functional.pad(vec[sl:sr], (0, padding_length), "constant", 0).div_(dist.get_world_size())
                else:
                    padded_tensor = vec[sl:sr].div_(dist.get_world_size())

                normalized_padded_tensor, mean_tensor = normalize_mean(padded_tensor, padded_tensor.numel(), max_chunk_size)
                padded_tensor = normalized_padded_tensor.view(-1, norm2_sum_chunk_size)

                padded_tensor_list.append(padded_tensor)

                dist.all_reduce(mean_tensor, async_op=False, op=dist.ReduceOp.SUM)
                mean_tensor_list.append(mean_tensor)

                torch.norm(padded_tensor, dim=1, keepdim=False, out=norm2_sum_memory)
                norm2_sum_memory **= 2 # L2 squared norm for each chunk
                
                norm2_sum_list.append(norm2_sum_memory)

                dist.all_reduce(norm2_sum_memory, async_op=False, op=dist.ReduceOp.SUM)

            for i in range(l, r):
                sl = state["start_idx"][(index, i)]
                sr = sl + state["params"]["size"][(index, i)]

                norm2_sum = norm2_sum_list[i]

                state["params"]["list_of_max_norm"].append(norm2_sum)
                if "fixed" not in state["args"].aggregation_method:
                    indice_8 = torch.argwhere(norm2_sum.view(-1) > state["params"]["last_kth"]).view(-1)
                    indice_2 = torch.argwhere(norm2_sum.view(-1) < state["params"]["last_kth_smaller"]).view(-1)
                
                    indice_4 = torch.argwhere((norm2_sum.view(-1) <= state["params"]["last_kth"]) & (norm2_sum.view(-1) >= state["params"]["last_kth_smaller"])).view(-1) # TO CHECK

                # to ensure that indices are synchronized and are not affected due to heterogeneous devices among different ranks, we broadcast the indices
                if "fixed" not in state["args"].aggregation_method:
                    dist.broadcast(indice_8, src=0, async_op=False)
                    dist.broadcast(indice_4, src=0, async_op=False)
                    dist.broadcast(indice_2, src=0, async_op=False)

                if "fixed" not in state["args"].aggregation_method:
                    comp_tensor_8 = padded_tensor_list[i][indice_8].reshape(-1)
                    comp_tensor_2 = padded_tensor_list[i][indice_2].reshape(-1)
                    comp_tensor_4 = padded_tensor_list[i][indice_4].reshape(-1)
                else:
                    comp_tensor_4 = padded_tensor_list[i].reshape(-1)
                
                if "fixed" not in state["args"].aggregation_method:
                    comp_tensor_8 = state["params"]["callback_func"](comp_tensor_8, state["params"]["callback_comm_8bit"], state["params"], dtype=torch.bfloat16, tag=(index << 10) | (i << 2))
                comp_tensor_4 = state["params"]["callback_func"](comp_tensor_4, state["params"]["callback_comm_4bit"], state["params"], dtype=torch.bfloat16, tag=(index << 10) | (i << 2) | 1)
                
                if "fixed" not in state["args"].aggregation_method:
                    comp_tensor_2 = state["params"]["callback_func"](comp_tensor_2, state["params"]["callback_comm_2bit"], state["params"], dtype=torch.bfloat16, tag=(index << 10) | (i << 2) | 2)
                    if "twoconservative" in state["args"].aggregation_method:
                        comp_tensor_2 *= 0.8

                if "fixed" not in state["args"].aggregation_method:
                    padded_tensor_list[i][indice_8] = comp_tensor_8.view(-1, norm2_sum_chunk_size)
                    padded_tensor_list[i][indice_4] = comp_tensor_4.view(-1, norm2_sum_chunk_size)
                    padded_tensor_list[i][indice_2] = comp_tensor_2.view(-1, norm2_sum_chunk_size)
                else:
                    padded_tensor_list[i] = comp_tensor_4.view(-1, norm2_sum_chunk_size)

                padded_tensor_list[i] = add_back_mean(padded_tensor_list[i], mean_tensor_list[i], padded_tensor_list[i].numel(), max_chunk_size) # returned tensor is of shape (-1)

                ret_tensor[sl:sr].copy_(padded_tensor_list[i][:sr - sl] * state["params"]["ratio"])

            if bucket.is_last() and "fixed" not in state["args"].aggregation_method:
                list_of_max_norm = torch.cat(state["params"]["list_of_max_norm"]).view(-1)
                list_of_max_norm = torch.sort(list_of_max_norm, descending=True)[0]
                
                # momentum
                state["params"]["last_kth"] = 0.25 * list_of_max_norm[int(list_of_max_norm.numel() * state["params"]["thresholds"][7])] + 0.75 * state["params"]["last_kth"]
                state["params"]["last_kth_smaller"] = 0.25 * list_of_max_norm[int(list_of_max_norm.numel() * (state["params"]["thresholds"][7] + state["params"]["thresholds"][4]))] + 0.75 * state["params"]["last_kth_smaller"]

            if bucket.is_last():    
                state["params"]["list_of_max_norm"] = []
            
            return ret_tensor
                

        except Exception:
            raise
    
    def return_func(fut):
        return ret_tensor
    

    state = state


    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
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
            # if i >= INTEG_PARTITION_LAYER - 1:
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
        
        if bucket.is_last():
            

            if "hierarchical" in state["args"].aggregation_method:
                state["params"]["supergroup"] = 16
            else:
                state["params"]["supergroup"] = 1
            
            state["params"]["chunk_size"] = max(max(state["params"]["chunk_size"], state["params"]["max_chunk_size"]), state["params"]["agg_chunk_size"] * state["params"]["supergroup"])
            state["params"]["max_chunk_size"] = max(state["params"]["max_chunk_size"], state["params"]["agg_chunk_size"] * state["params"]["supergroup"])

            if "hierarchical" in state["args"].aggregation_method:  # currently only implemented MEE_Dynamic_Range and AEE_Dynamic_Range
                agg_func = Mee_Dynamic_Range_Hierarchical
            else:
                agg_func = Mee_Dynamic_Range
            
            if "4bit" in state["args"].aggregation_method:
                if "fixed" in state["args"].aggregation_method:
                    state["params"]["thresholds"] = {4: 1, 8: 0, 2: 0}
                else:
                    state["params"]["thresholds"] = {8: 0.05, 4: 0.6, 2: 0.35}
                    
                state["params"]["callback_comm_2bit"] = agg_func(2, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_4bit"] = agg_func(4, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_8bit"] = agg_func(8, state["params"]["agg_chunk_size"], state["params"])

            elif "3bit" in state["args"].aggregation_method:
                if "fixed" in state["args"].aggregation_method:
                    state["params"]["thresholds"] = {8: 0, 4: 1, 2: 0}
                else:
                    state["params"]["thresholds"] = {8: 1 / 40, 4: 13 / 40, 2: 26 / 40} 
                    
                state["params"]["callback_comm_2bit"] = agg_func(2, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_4bit"] = agg_func(4, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_8bit"] = agg_func(8, state["params"]["agg_chunk_size"], state["params"])

            elif "6bit" in state["args"].aggregation_method:
                if "fixed" in state["args"].aggregation_method:
                    state["params"]["thresholds"] = {8: 0, 4: 1, 2: 0}
                else:
                    state["params"]["thresholds"] = {8: 0.75, 4: 0.22, 2: 0.03}
                    
                state["params"]["callback_comm_2bit"] = agg_func(2, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_4bit"] = agg_func(4, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_8bit"] = agg_func(8, state["params"]["agg_chunk_size"], state["params"])

            elif "7bit" in state["args"].aggregation_method:
                if "fixed" in state["args"].aggregation_method:
                    state["params"]["thresholds"] = {8: 1, 4: 0, 2: 0}
                else:
                    state["params"]["thresholds"] = {8: 0.86, 4: 0.12, 2: 0.02}
                    
                state["params"]["callback_comm_2bit"] = agg_func(2, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_4bit"] = agg_func(4, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_8bit"] = agg_func(8, state["params"]["agg_chunk_size"], state["params"])

            else: # 5bit
                if "fixed" in state["args"].aggregation_method:
                    state["params"]["thresholds"] = {8: 0, 4: 1, 2: 0}
                else:
                    state["params"]["thresholds"] = {8: 0.2, 4: 0.6, 2: 0.2}

                state["params"]["callback_comm_2bit"] = agg_func(2, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_4bit"] = agg_func(4, state["params"]["agg_chunk_size"], state["params"])
                state["params"]["callback_comm_8bit"] = agg_func(8, state["params"]["agg_chunk_size"], state["params"])
            
            if "butterfly" in state["args"].aggregation_method:
                state["params"]["callback_func"] = composable_butterfly_allreduce_callback
            else:
                state["params"]["callback_func"] = composable_allreduce_arbitrary_callback

            state["params"]["compressor"] = NewINCACompressor(state["params"])

            state["params"]["last_kth"] = 1.1562e-4
            state["params"]["last_kth_smaller"] = 2.6375e-06
            state["params"]["list_of_max_norm"] = []
            state["params"]["ratio"] = 1

        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )
        
    else:
        index = bucket.index()
        
        vec = bucket.buffer()
        total_size = bucket.buffer().numel()
        ret_tensor = bucket.buffer()


        init_future = torch.futures.Future()
        init_future.set_result(0)

        l = 0
        r = state["partition_len"][index]

        reduce_future = init_future.then(lambda fut: compress_and_reduce_and_decompress(fut, state, index, l, r, bucket)) #bucket.is_last()))

        return reduce_future





def P2P_dynamiQ_dynamic_bitrate_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    DYNAMIC_BITRATE_HEADROOM = 0.625
    DYNAMIC_BITRATE_U_MIN = -10.0
    DYNAMIC_BITRATE_U_MAX = 60.0
    DYNAMIC_BITRATE_TOL = 0.1
    DYNAMIC_BITRATE_MAX_ITERS = 32
    DYNAMIC_BITRATE_NORM_SCALE = 4.0 / (512.0 / 17.0)

    def dynamic_bitrate_target_bitrate(args):
        method = args.aggregation_method.lower()
        for nbits in (8, 7, 6, 5, 4, 3, 2):
            if f"{nbits}bit" in method:
                return float(nbits)
        return 5.0

    def dynamic_bitrate_qbits(norms, u):
        eps = 1e-20
        log_norms = torch.log2(torch.clamp(norms.to(torch.float32), min=eps))
        score = log_norms.mul(DYNAMIC_BITRATE_NORM_SCALE).add(float(u))
        z = torch.floor(torch.log2(torch.clamp(score, min=eps))).to(torch.int64)
        z.clamp_(1, 3)
        return torch.bitwise_left_shift(torch.ones_like(z), z)

    def dynamic_bitrate_average_bits(norms_list, u):
        total_chunks = 0
        total_bits = None
        for norms in norms_list:
            if norms.numel() == 0:
                continue
            qbits = dynamic_bitrate_qbits(norms, u)
            qbits_sum = qbits.sum()
            total_bits = qbits_sum if total_bits is None else total_bits + qbits_sum
            total_chunks += qbits.numel()

        if total_bits is None or total_chunks == 0:
            return 0.0
        return float(total_bits.item()) / total_chunks

    def dynamic_bitrate_avg_delta(norms_list, u, target_avg_bits):
        avg_bits = dynamic_bitrate_average_bits(norms_list, u)
        return avg_bits, target_avg_bits - avg_bits

    def dynamic_bitrate_binary_search_u(norms_list, target_avg_bits):
        low = DYNAMIC_BITRATE_U_MIN
        high = DYNAMIC_BITRATE_U_MAX
        best_u = high
        best_avg, best_delta = dynamic_bitrate_avg_delta(norms_list, best_u, target_avg_bits)

        for _ in range(DYNAMIC_BITRATE_MAX_ITERS):
            mid = (low + high) * 0.5
            avg_bits, delta = dynamic_bitrate_avg_delta(norms_list, mid, target_avg_bits)
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

    def dynamic_bitrate_adjust_u(norms_list, prev_u, target_avg_bits):
        prev_u = max(DYNAMIC_BITRATE_U_MIN, min(DYNAMIC_BITRATE_U_MAX, float(prev_u)))
        old_avg, old_delta = dynamic_bitrate_avg_delta(norms_list, prev_u, target_avg_bits)
        if abs(old_delta) <= DYNAMIC_BITRATE_TOL:
            return prev_u, old_avg, old_delta

        step = old_delta
        best_u = prev_u
        best_avg = old_avg
        best_delta = old_delta

        for _ in range(DYNAMIC_BITRATE_MAX_ITERS):
            candidate_u = max(DYNAMIC_BITRATE_U_MIN, min(DYNAMIC_BITRATE_U_MAX, prev_u + step))
            avg_bits, delta = dynamic_bitrate_avg_delta(norms_list, candidate_u, target_avg_bits)
            if abs(delta) < abs(best_delta):
                best_u = candidate_u
                best_avg = avg_bits
                best_delta = delta
            if abs(delta) <= DYNAMIC_BITRATE_TOL or old_delta * delta >= 0:
                return candidate_u, avg_bits, delta
            step *= 0.5

        return best_u, best_avg, best_delta

    def update_next_u_if_last_bucket(is_last_bucket):
        if not is_last_bucket:
            return

        norms_list = state.get("dynamic_bitrate_norms", [])
        if not norms_list:
            state["dynamic_bitrate_norms"] = []
            return

        target_avg_bits = state["params"].setdefault(
            "dynamic_bitrate_target_avg_bits",
            dynamic_bitrate_target_bitrate(state["args"]) - DYNAMIC_BITRATE_HEADROOM,
        )
        prev_u = state["params"].get("dynamic_bitrate_u", DYNAMIC_BITRATE_U_MAX)

        if state["batch_idx"] == 2:
            next_u, avg_bits, delta = dynamic_bitrate_binary_search_u(norms_list, target_avg_bits)
        else:
            next_u, avg_bits, delta = dynamic_bitrate_adjust_u(norms_list, prev_u, target_avg_bits)

        state["params"]["dynamic_bitrate_u"] = float(next_u)
        state["params"]["dynamic_bitrate_last_avg_bits"] = float(avg_bits)
        state["params"]["dynamic_bitrate_last_delta"] = float(delta)
        state["dynamic_bitrate_norms"] = []

    def compress_and_reduce_and_decompress(fut, state, index, l, r, bucket):
        fut.wait()
        max_chunk_size = state["params"]["chunk_size"]
        norm2_sum_chunk_size = state["params"]["max_chunk_size"]
        current_u = state["params"].get("dynamic_bitrate_u", DYNAMIC_BITRATE_U_MAX)

        try:
            padded_tensor_list = []
            norm2_sum_list = []
            mean_tensor_list = []

            for i in range(l, r):
                sl = state["start_idx"][(index, i)]
                sr = sl + state["params"]["size"][(index, i)]

                norm2_sum_memory = state["params"]["compressor"].max_memory_tensor((index, i))
                padding_length = (sr - sl + max_chunk_size - 1) // max_chunk_size * max_chunk_size - (sr - sl)
                if padding_length > 0:
                    padded_tensor = torch.nn.functional.pad(vec[sl:sr], (0, padding_length), "constant", 0).div_(dist.get_world_size())
                else:
                    padded_tensor = vec[sl:sr].div_(dist.get_world_size())

                normalized_padded_tensor, mean_tensor = normalize_mean(padded_tensor, padded_tensor.numel(), max_chunk_size)
                padded_tensor = normalized_padded_tensor.view(-1, norm2_sum_chunk_size)

                padded_tensor_list.append(padded_tensor)

                dist.all_reduce(mean_tensor, async_op=False, op=dist.ReduceOp.SUM)
                mean_tensor_list.append(mean_tensor)

                torch.norm(padded_tensor, dim=1, keepdim=False, out=norm2_sum_memory)
                norm2_sum_memory **= 2

                norm2_sum_list.append(norm2_sum_memory)

                dist.all_reduce(norm2_sum_memory, async_op=False, op=dist.ReduceOp.SUM)

            for i in range(l, r):
                sl = state["start_idx"][(index, i)]
                sr = sl + state["params"]["size"][(index, i)]

                norm2_sum = norm2_sum_list[i].view(-1)
                state.setdefault("dynamic_bitrate_norms", []).append(norm2_sum)

                qbits = dynamic_bitrate_qbits(norm2_sum, current_u)
                indice_8 = torch.argwhere(qbits == 8).view(-1)
                indice_4 = torch.argwhere(qbits == 4).view(-1)
                indice_2 = torch.argwhere(qbits == 2).view(-1)

                comp_tensor_8 = padded_tensor_list[i][indice_8].reshape(-1)
                comp_tensor_4 = padded_tensor_list[i][indice_4].reshape(-1)
                comp_tensor_2 = padded_tensor_list[i][indice_2].reshape(-1)

                comp_tensor_8 = state["params"]["callback_func"](comp_tensor_8, state["params"]["callback_comm_8bit"], state["params"], dtype=torch.bfloat16, tag=(index << 10) | (i << 2))
                comp_tensor_4 = state["params"]["callback_func"](comp_tensor_4, state["params"]["callback_comm_4bit"], state["params"], dtype=torch.bfloat16, tag=(index << 10) | (i << 2) | 1)
                comp_tensor_2 = state["params"]["callback_func"](comp_tensor_2, state["params"]["callback_comm_2bit"], state["params"], dtype=torch.bfloat16, tag=(index << 10) | (i << 2) | 2)
                if "twoconservative" in state["args"].aggregation_method:
                    comp_tensor_2 *= 0.8

                padded_tensor_list[i][indice_8] = comp_tensor_8.view(-1, norm2_sum_chunk_size)
                padded_tensor_list[i][indice_4] = comp_tensor_4.view(-1, norm2_sum_chunk_size)
                padded_tensor_list[i][indice_2] = comp_tensor_2.view(-1, norm2_sum_chunk_size)

                padded_tensor_list[i] = add_back_mean(padded_tensor_list[i], mean_tensor_list[i], padded_tensor_list[i].numel(), max_chunk_size)

                ret_tensor[sl:sr].copy_(padded_tensor_list[i][:sr - sl] * state["params"]["ratio"])

            update_next_u_if_last_bucket(bucket.is_last())

            return ret_tensor

        except Exception:
            raise

    def return_func(fut):
        return ret_tensor

    state = state

    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
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
            # if i >= INTEG_PARTITION_LAYER - 1:
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

        if bucket.is_last():

            if "hierarchical" in state["args"].aggregation_method:
                state["params"]["supergroup"] = 16
            else:
                state["params"]["supergroup"] = 1

            state["params"]["chunk_size"] = max(max(state["params"]["chunk_size"], state["params"]["max_chunk_size"]), state["params"]["agg_chunk_size"] * state["params"]["supergroup"])
            state["params"]["max_chunk_size"] = max(state["params"]["max_chunk_size"], state["params"]["agg_chunk_size"] * state["params"]["supergroup"])

            if "hierarchical" in state["args"].aggregation_method:
                agg_func = Mee_Dynamic_Range_Hierarchical
            else:
                agg_func = Mee_Dynamic_Range

            state["params"]["callback_comm_2bit"] = agg_func(2, state["params"]["agg_chunk_size"], state["params"])
            state["params"]["callback_comm_4bit"] = agg_func(4, state["params"]["agg_chunk_size"], state["params"])
            state["params"]["callback_comm_8bit"] = agg_func(8, state["params"]["agg_chunk_size"], state["params"])

            if "butterfly" in state["args"].aggregation_method:
                state["params"]["callback_func"] = composable_butterfly_allreduce_callback
            else:
                state["params"]["callback_func"] = composable_allreduce_arbitrary_callback

            state["params"]["compressor"] = NewINCACompressor(state["params"])
            state["params"]["ratio"] = 1

            target_bitrate = dynamic_bitrate_target_bitrate(state["args"])
            state["params"]["dynamic_bitrate_target_bitrate"] = target_bitrate
            state["params"]["dynamic_bitrate_target_avg_bits"] = target_bitrate - DYNAMIC_BITRATE_HEADROOM
            state["params"]["dynamic_bitrate_u"] = DYNAMIC_BITRATE_U_MAX
            state["params"]["dynamic_bitrate_last_avg_bits"] = None
            state["params"]["dynamic_bitrate_last_delta"] = None
            state["dynamic_bitrate_norms"] = []
            state["dynamic_bitrate_stats_batch_idx"] = None

        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    else:
        index = bucket.index()

        vec = bucket.buffer()
        total_size = bucket.buffer().numel()
        ret_tensor = bucket.buffer()

        state["params"].setdefault("dynamic_bitrate_u", DYNAMIC_BITRATE_U_MAX)
        state["params"].setdefault(
            "dynamic_bitrate_target_avg_bits",
            dynamic_bitrate_target_bitrate(state["args"]) - DYNAMIC_BITRATE_HEADROOM,
        )
        if state.get("dynamic_bitrate_stats_batch_idx") != state["batch_idx"]:
            state["dynamic_bitrate_norms"] = []
            state["dynamic_bitrate_stats_batch_idx"] = state["batch_idx"]

        init_future = torch.futures.Future()
        init_future.set_result(0)

        l = 0
        r = state["partition_len"][index]

        reduce_future = init_future.then(lambda fut: compress_and_reduce_and_decompress(fut, state, index, l, r, bucket)) #bucket.is_last()))

        return reduce_future



def distributed_omnireduce_topk_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    
    def compress_reduce_decompress(fut, state, index, l, r, vec, ret_tensor):
        all_reduce_handles = []
        all_reduce_tensors = []
        all_reduce_indice = []
        max_chunk_size = state["params"]["max_chunk_size"]

        bitmap_list = []
        bitmap_handles = []
        padded_tensor_list = []

        total_chunk_cnt = 0
        dense_chunk_cnt = 0

        for i in range(l, r):
            sl = state["start_idx"][(index, i)]
            sr = sl + state["params"]["size"][(index, i)]

            padded_tensor, bitmap_memory = state["params"]["compressor"].padding_tensor(vec[sl:sr], (index, i))
            padded_tensor = padded_tensor.view(-1, max_chunk_size)
            
            local_norm = torch.norm(padded_tensor, p="fro", dim=1, keepdim=False)
            nchunks = local_norm.numel()
            topk_chunks = int(nchunks * state["params"]["local_topks"][(index, i)])

            _, topk_indices = torch.topk(local_norm, topk_chunks, largest=True)
            bitmap_memory *= 0
            bitmap_memory[topk_indices] = 1
            
            bitmap_list.append(bitmap_memory)
            bitmap_handles.append(dist.all_reduce(bitmap_memory, async_op=True, op=dist.ReduceOp.SUM))
            padded_tensor_list.append(padded_tensor)
        for i in range(l, r):
            sl = state["start_idx"][(index, i)]
            sr = sl + state["params"]["size"][(index, i)]
            bitmap_handles[i].wait()

            padded_tensor = padded_tensor_list[i]
            bitmap_memory = bitmap_list[i]

            indice = bitmap_memory.nonzero().view(-1)

            dense_chunk_cnt += indice.numel()
            total_chunk_cnt += bitmap_memory.numel()
            
            to_reduce_tensor = padded_tensor[indice].view(-1)

            dist.all_reduce(to_reduce_tensor, async_op=False, op=dist.ReduceOp.SUM)
            padded_tensor *= 0
            padded_tensor[indice] = to_reduce_tensor.view(-1, max_chunk_size)

            ret_tensor[sl:sr].copy_(padded_tensor.view(-1)[:sr - sl])

            # handling local sparsity.
            target_topk = state["params"]["target_topk"] * bitmap_memory.numel()
            if target_topk >= 1:
                real_topk = indice.numel()
                proportion = real_topk / target_topk
                proportion = max(0.6, min(proportion, 1 / 0.6))
                state["params"]["local_topks"][(index, i)] = 0.75 * state["params"]["local_topks"][(index, i)] + 0.25 * state["params"]["local_topks"][(index, i)] / proportion # momentum



        if total_chunk_cnt == 0:
            state["params"]["sparsity"].append([0, 0, 0])
        else:
            state["params"]["sparsity"].append([total_chunk_cnt, total_chunk_cnt - dense_chunk_cnt, 1 - dense_chunk_cnt / total_chunk_cnt])
            state["params"]["sparsity"][0][0] += total_chunk_cnt
            state["params"]["sparsity"][0][1] += total_chunk_cnt - dense_chunk_cnt
            state["params"]["sparsity"][0][2] = 0 if state["params"]["sparsity"][0][0] == 0 else state["params"]["sparsity"][0][1] / state["params"]["sparsity"][0][0]

        return ret_tensor


    def return_func(fut):
        return ret_tensor
    

    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    elif state["batch_idx"] == 1:
        
        index = bucket.index()
        total_size = bucket.buffer().numel()
        orig_total_size = total_size


        state["params"]["max_chunk_size"] = state["params"]["chunk_size"] = 256   # based on what omnireduce recommends
        state["params"]["sparsity"] = [[0, 0, 0]]
        if "target_topk" not in state["params"]:
            state["params"]["target_topk"] = 0.5 # let's try 8 bits per coordinate. Then 5 bit

        i = 0
        while True:
            cur_size = min(total_size, CHUNK_SIZE_THRESHOLD)
            cur_d = cur_size

            state["params"]["d"][(index, i)] = ((cur_d - 1) // state["params"]["max_chunk_size"] + 1) * state["params"]["max_chunk_size"]
            state["params"]["size"][(index, i)] = cur_size
            state["start_idx"][(index, i)] = orig_total_size - total_size
            

            total_size -= cur_size
            i += 1
            if total_size == 0:
                break
        
        state["partition_len"][index] = i

        if bucket.is_last() == True:

            state["params"]["compressor"] = NewINCACompressor(state["params"])
            state["params"]["local_topks"] = dict()
            for key in state["params"]["d"]:
                state["params"]["local_topks"][key] = state["params"]["target_topk"] #initially set the same target topk

        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )
        
    else:



        index = bucket.index()
        
        vec = bucket.buffer()

        if bucket.is_last() == True and len(state["params"]["sparsity"]) > 0:
            if dist.get_rank() == 0:
                fout = state["params"]["txt_log_file"]
                fout.write("Sparsity: {} {}. Target: {}. ".format(1 - state["params"]["sparsity"][0][2], 1 - state["params"]["sparsity"][-1][2], state["params"]["target_topk"]))
        
        total_size = bucket.buffer().numel()
        ret_tensor = bucket.buffer() # state["ret_tensor"][index]

        l = 0
        r = state["partition_len"][index]

        init_future = dist.all_reduce(vec[0].norm(2).view(-1), async_op=True, op=dist.ReduceOp.MAX).get_future()

        reduce_future = init_future.then(lambda fut: compress_reduce_decompress(fut, state, index, l, r, vec, ret_tensor)) #bucket.is_last()))

        return reduce_future




def P2P_MXfpx_compress_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    def update_policy(aggregated_tensor, key, state, fpx_max, agg_method):
        if "fp4" not in agg_method:
            return
        
        # only update for fp4
        num_elem = aggregated_tensor.numel()
        num_overflowed_elem = torch.sum(aggregated_tensor.abs() >= fpx_max - 5e-8).item()

        if num_overflowed_elem > num_elem * state["params"]["proportion"]: # proportion
            state["params"]["mu"][key] = max(state["params"]["mu"][key] * 0.95, fpx_max / dist.get_world_size() / 2)
            # state["params"]["mu"][key] = max(state["params"]["mu"][key] * 0.9, fpx_max / dist.get_world_size() / 2)
        else:
            state["params"]["mu"][key] = min(fpx_max, state["params"]["mu"][key] * (2 ** 0.02))

    def compress_and_reduce_and_decompress(fut, state, index, l, r, no_hadamard=False):
        fut.wait()
        max_chunk_size = state["params"]["chunk_size"]

        try:
            src_tensors = []
            all_reduce_handles = []

            padded_tensor_list = []
            norm_max_list = []
            max_reduce_handles = []
            for i in range(l, r):
                sl = state["start_idx"][(index, i)]
                sr = sl + state["params"]["size"][(index, i)]
                
                padded_tensor, norm_max_memory = state["params"]["compressor"].padding_tensor(vec[sl:sr], (index, i))   # division done here!!!
                padded_tensor = padded_tensor.view(-1, max_chunk_size)
                padded_tensor_list.append(padded_tensor)

                torch.amax(padded_tensor.abs(), dim=1, keepdim=False, out=norm_max_memory)
                norm_max_memory = norm_max_memory.view(-1, 1)

                # round up to the power of 2
                norm_max_memory = 2 ** torch.ceil(torch.log2(norm_max_memory))
                norm_max_list.append(norm_max_memory)
                
                max_reduce_handles.append(dist.all_reduce(norm_max_memory, async_op=True, op=dist.ReduceOp.MAX))

            for i in range(l, r):
                max_reduce_handles[i].wait()
                sl = state["start_idx"][(index, i)]
                sr = sl + state["params"]["size"][(index, i)]

                norm_max = norm_max_list[i]

                padded_tensor = padded_tensor_list[i]

                eps = 5e-7 if "fp4" in state["args"].aggregation_method else 1e-23

                padded_tensor.div_(norm_max + eps).mul_(state["params"]["mu"][(index, i)]) # MX_FP8_E4M3_MAX  / dist.get_world_size() is the maximum value to scale the tensor up.
                padded_tensor = padded_tensor.view(-1)

                aggregated_tensor = state["params"]["callback_func"](padded_tensor, state["params"]["callback_comm"], state["params"], dtype=torch.bfloat16, tag=(index << 8))


                old_mu = state["params"]["mu"][(index, i)]
                update_policy(aggregated_tensor, key=(index, i), state=state, fpx_max=state["params"]["callback_comm"].fpx_max, agg_method=state["args"].aggregation_method)

                aggregated_tensor.view(-1, max_chunk_size).div_(old_mu).mul_(norm_max + eps)

                
                ret_tensor[sl:sr].copy_(aggregated_tensor.view(-1)[:sr - sl])
            
            return ret_tensor
                

        except Exception:
            raise
    
    def return_func(fut):
        return ret_tensor
    

    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
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
            # if i >= INTEG_PARTITION_LAYER - 1:
                cur_size = total_size
                cur_d = cur_size
            else:
                cur_size = CHUNK_SIZE_THRESHOLD
                # cur_size = 1 << (total_size.bit_length() - 1)
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
        state["params"]["max_chunk_size"] = 32
        state["params"]["agg_chunk_size"] = 32

        
        if bucket.is_last():
            state["params"]["mu"] = dict()
            if "fp4" in state["args"].aggregation_method:
                state["params"]["callback_comm"] = Fpx_arithmetics(2, 1, state["params"])
                state["params"]["proportion"] = 2e-2
            elif "fp6" in state["args"].aggregation_method:
                state["params"]["callback_comm"] = Fpx_arithmetics(3, 2, state["params"])
                state["params"]["proportion"] = 1e-2

            for key in state["params"]["d"]:
                if "fp4" in state["args"].aggregation_method:
                    state["params"]["mu"][key] = state["params"]["callback_comm"].fpx_max / ((2 * float(dist.get_world_size())) ** 0.5)
                else:
                    state["params"]["mu"][key] = state["params"]["callback_comm"].fpx_max / (1.4 * float(dist.get_world_size()))
            

            state["params"]["compressor"] = NewINCACompressor(state["params"])

            if "butterfly" in state["args"].aggregation_method:
                state["params"]["callback_func"] = composable_butterfly_allreduce_callback
            else:
                state["params"]["callback_func"] = composable_allreduce_arbitrary_callback 

        return (
            dist.all_reduce(bucket.buffer().div_(dist.get_world_size()), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )
        
    else:
        index = bucket.index()
        
        vec = bucket.buffer()
        total_size = bucket.buffer().numel()
        ret_tensor = bucket.buffer()


        init_future = torch.futures.Future()
        init_future.set_result(0)

        l = 0
        r = state["partition_len"][index]

        reduce_future = init_future.then(lambda fut: compress_and_reduce_and_decompress(fut, state, index, l, r, True)) #bucket.is_last()))

        return reduce_future




def wrapper_hook(state, bucket):
    def calc_l2_norm(fut, state, bucket, gt_tensor):
        fut.wait()
        ret_tensor = fut.value()
        params = state["params"]
        cur_l2_error = torch.norm(ret_tensor - gt_tensor / params["lr_adjust_param"]) ** 2
        cur_l2_norm = torch.norm(gt_tensor.to(torch.float32)) ** 2
        cur_pred_l2_norm = torch.norm(ret_tensor.to(torch.float32)) ** 2

        state["params"]["sum_l2_error"] += cur_l2_error
        state["params"]["sum_l2_norm"] += cur_l2_norm
        state["params"]["sum_l2_pred_norm"] += cur_pred_l2_norm

        if bucket.is_last():
            local_rank = dist.get_rank()
            if local_rank == 0:
                txt_log_file = params["txt_log_file"]
                txt_log_file.write("{} {} {}\n".format(state["params"]["sum_l2_error"], state["params"]["sum_l2_norm"], state["params"]["sum_l2_pred_norm"]))
                txt_log_file.flush()
                _write_comm_error_vnmse(state)

            if "rescale" in state["args"].aggregation_method:
                state["params"]["ratio"] = state["params"]["ratio"] * 0.75 + 0.25 * ((state["params"]["sum_l2_norm"] / (state["params"]["sum_l2_pred_norm"] + 1e-11)) ** 0.5)
            else:
                state["params"]["ratio"] = 1
            state["params"]["sum_l2_error"] = 0
            state["params"]["sum_l2_norm"] = 0
            state["params"]["sum_l2_pred_norm"] = 0

            # momentum
            

        return ret_tensor

    aggregation_method = state["args"].aggregation_method
    aggregation_method_lower = aggregation_method.lower()
    track_error = state["params"].get("measure_comm_error", False) or "rescale" in aggregation_method

    if state["batch_idx"] == 0:
        state["params"]["sum_l2_error"] = 0
        state["params"]["sum_l2_norm"] = 0
        state["params"]["sum_l2_pred_norm"] = 0
        state["params"]["ratio"] = 1

    copied_tensor = None
    if track_error:
        copied_tensor = bucket.buffer().clone() / dist.get_world_size()
        dist.all_reduce(copied_tensor, async_op=False, op=dist.ReduceOp.SUM)

    if "bf16" in aggregation_method:
        reduce_future = P2P_bf16_compress_hook(state, bucket)
    elif "THC" in aggregation_method or "thc" in aggregation_method:
        reduce_future = P2P_THC_compress_hook(state, bucket)
    elif aggregation_method == "fp16":
        reduce_future = P2P_fp16_compress_hook(state, bucket)
    elif "MXfp8" in aggregation_method:
        reduce_future = P2P_MXfp8_compress_hook(state, bucket)
    elif "fp4" in aggregation_method or "fp6" in aggregation_method:
        reduce_future = P2P_MXfpx_compress_hook(state, bucket)
    elif "omnireduce" in aggregation_method:
        reduce_future = distributed_omnireduce_topk_hook(state, bucket)
    elif "dynamiq" in aggregation_method_lower:
        if "dynamic_bitrate" in aggregation_method_lower or "bitrate" in aggregation_method_lower:
            reduce_future = P2P_dynamiQ_dynamic_bitrate_hook(state, bucket)
        else:
            reduce_future = P2P_dynamiQ_fixed_percentage_hook(state, bucket)
    else:
        raise ValueError(
            "Unsupported aggregation method: "
            f"{aggregation_method}. Use bf16, fp16, MXfp8, fp4/fp6, omnireduce, or dynamiQ methods, "
            "optionally with a _butterfly suffix."
        )

    if not track_error:
        return reduce_future

    return reduce_future.then(lambda fut: calc_l2_norm(fut, state, bucket, copied_tensor))
