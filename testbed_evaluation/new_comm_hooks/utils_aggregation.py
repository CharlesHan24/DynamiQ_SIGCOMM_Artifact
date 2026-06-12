import torch.distributed as dist
import torch
import pdb
import sys
import time
import torch.profiler


class overflow_pickup_summation(object):
    def __init__(self, quant_levels, bits=8):
        self.overflow_val = (1 << bits) - quant_levels - 1

    def summation(self, src_tensor: torch.Tensor, trans_tensor: torch.Tensor, overflow_index: torch.Tensor, overflow_value: torch.Tensor):
        # pdb.set_trace()
        overflow_value.add_(trans_tensor[overflow_index].type(torch.int32))
        trans_tensor.sub_(trans_tensor[overflow_index])
        
        src_tensor.add_(trans_tensor)
        
        new_overflow_index = torch.where(src_tensor >= self.overflow_val).nonzero()
        new_overflow_value = src_tensor[new_overflow_index]
        overflow_index = torch.cat([overflow_index, new_overflow_index])
        overflow_value = torch.cat([overflow_value, new_overflow_value])
        

        return src_tensor.where(src_tensor < self.overflow_val, 0), overflow_index, overflow_value

    def broadcast(self, dst_tensor: torch.Tensor, trans_tensor: torch.Tensor, overflow_index: torch.Tensor, overflow_value: torch.Tensor):
        dst_tensor.copy_(trans_tensor.type(torch.int32))
        dst_tensor[overflow_index] = overflow_value





def floattoint2(tensor1, tensor2, nelem): # tensor 1's type is float16, the data is integers between [-1, 1], tensor2's type is uint8. tensor2 contains exactly nelem elements

    tensor1.add_(1)
    tensor1 = tensor1.type(torch.uint8)
    tensor2.copy_(tensor1[:nelem >> 2])
    tensor2.add_(tensor1[nelem >> 2: nelem >> 1].mul_(4))
    tensor2.add_(tensor1[nelem >> 1: (nelem >> 2) * 3].mul_(16))
    tensor2.add_(tensor1[(nelem >> 2) * 3: nelem].mul_(64))
    return tensor2

def floattoint4(tensor1, tensor2, nelem):  # tensor2 contains exactly nelem elements
    tensor1.add_(7)
    tensor1 = tensor1.type(torch.uint8)
    tensor2.copy_(tensor1[:nelem >> 1])
    tensor2.add_(tensor1[nelem >> 1: nelem].mul_(16))
    return tensor2

def int2tofloat(tensor1, tensor2, nelem):
    # torch.bitwise_and(tensor1, 3, out=tensor2[:nelem >> 2])
    # torch.bitwise_right_shift(tensor1, 2, out=tensor1)
    # torch.bitwise_and(tensor1, 3, out=tensor2[nelem >> 2: nelem >> 1])
    # torch.bitwise_right_shift(tensor1, 2, out=tensor1)
    # torch.bitwise_and(tensor1, 3, out=tensor2[nelem >> 1: (nelem >> 1) * 3])
    # torch.bitwise_right_shift(tensor1, 2, out=tensor1)
    # torch.bitwise_and(tensor1, 3, out=tensor2[(nelem >> 1) * 3: nelem])
    tensor2[:nelem >> 2].copy_(tensor1 & 3)
    tensor1 >>= 2
    tensor2[nelem >> 2: nelem >> 1].copy_(tensor1 & 3)
    tensor1 >>= 2
    tensor2[nelem >> 1: (nelem >> 2) * 3].copy_(tensor1 & 3)
    tensor1 >>= 2
    tensor2[(nelem >> 2) * 3: nelem].copy_(tensor1)
    tensor2.sub_(1)
    return tensor2

def int4tofloat(tensor1, tensor2, nelem):
    # pdb.set_trace()
    # torch.bitwise_and(tensor1, 15, out=tensor2[:nelem >> 1])
    # torch.bitwise_right_shift(tensor1, 4, out=tensor1)
    # torch.bitwise_and(tensor1, 15, out=tensor2[nelem >> 1: nelem])
    tensor2[:nelem >> 1].copy_(tensor1 & 15)
    tensor1 >>= 4
    tensor2[nelem >> 1: nelem].copy_(tensor1 & 15)
    tensor2.sub_(7) # [-7, 7]
    return tensor2

"""
class Aee_Dynamic_Range(object):
    def __init__(self, nbits, chunk_size, params):
        self.nbits = nbits
        self.chunk_size = chunk_size
        self.range = 1 << self.nbits
        self.params = params
        self.cur_max_size = 0
        self.nclients = dist.get_world_size()
        self.client_rank = dist.get_rank()

        for d_index in params["d"]:
            self.cur_max_size = max(self.cur_max_size, (params["d"][d_index] + chunk_size) // chunk_size)
        
        self.three_times_cur_max_size = 3 * self.cur_max_size + 2 * self.chunk_size
        
        self.randomized_vec_pool = torch.randint(0, 65536, dtype=torch.float32, size=(self.three_times_cur_max_size // chunk_size, chunk_size), device="cuda") / 65536
        self.rand_seed = (self.nbits * 998244353) & 0xFFFFFFFF

    def gen_next_rand(self):
        self.rand_seed = (self.rand_seed * 671431 + 1000000007) & 0xFFFFFFFF
        return self.rand_seed
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()
        padded_n = (n + self.chunk_size - 1) // self.chunk_size * self.chunk_size

        if dtype == torch.uint8: # scales are encoded as torch.float16
            padded_vec_size = padded_n // (8 // self.nbits)
            padded_vec_size += (padded_n // self.chunk_size) * 2 # 2 bytes for each chunk

        else:
            padded_vec_size = padded_n

        return torch.zeros(padded_vec_size, dtype=dtype, device="cuda")
    
    def compress(self, interm_chunk, send_chunk, n):
        padded_n = (n + self.chunk_size - 1) // self.chunk_size * self.chunk_size
        padded_vec_size = padded_n // (8 // self.nbits)

        two_dimen_interm_chunk = interm_chunk.view(-1, self.chunk_size)

        max_val = (torch.max(torch.abs(two_dimen_interm_chunk), dim=1, keepdim=True)[0] + 1e-15)
        prob_pow = max_val.div_(self.range)

        send_chunk_scales = send_chunk[padded_vec_size:].view(torch.float16)
        send_chunk_scales.copy_(prob_pow.view(-1))

        two_dimen_interm_chunk.div_(prob_pow)

        two_dimen_interm_chunk2 = two_dimen_interm_chunk.clone()
        two_dimen_interm_chunk.floor_()
        prob = two_dimen_interm_chunk2.sub_(two_dimen_interm_chunk)

        randl_mod = self.randomized_vec_pool.numel() // chunk_size - two_dimen_interm_chunk.numel() // self.chunk_size
        randl = self.gen_next_rand() % randl_mod
        rand_vec = self.randomized_vec_pool[rand_l:rand_l + two_dimen_interm_chunk.numel() // self.chunk_size, :]

        two_dimen_interm_chunk.add_(rand_vec <= prob)
        
        if self.nbits == 8:
            send_chunk[:padded_vec_size].copy_(interm_chunk)
        elif self.nbits == 4:
            floattoint4(interm_chunk, send_chunk[:padded_vec_size])
        else:
            floattoint2(interm_chunk, send_chunk[:padded_vec_size])

    def compression(self, input_chunk, send_chunk, interm_chunk): # compress input_chunk to send_chunk
        n = input_chunk.numel()
        interm_chunk[:n].copy_(input_chunk)
        interm_chunk[n:].fill_(0)
        
        self.compress(interm_chunk, send_chunk, n)
    
    def decompress(self, recv_chunk, interm_chunk, n):  
        padded_n = (n + self.chunk_size - 1) // self.chunk_size * self.chunk_size
        padded_vec_size = padded_n // (8 // self.nbits)

        scale_chunk = recv_chunk[padded_vec_size:].view(torch.float16)  # shared memory

        if self.nbits == 8:
            interm_chunk.copy_(recv_chunk[:padded_vec_size])
        elif self.nbits == 4:
            int4tofloat(recv_chunk[:padded_vec_size], interm_chunk)
        else:
            int2tofloat(recv_chunk[:padded_vec_size], interm_chunk)

        data_chunk = interm_chunk.view(-1, self.chunk_size)  # shared memory
        data_chunk.mul_(scale_chunk)

    def decompression(self, recv_chunk, input_chunk, interm_chunk): # recv_chunk, input_chunk, interm chunk are contiguous tensors. Decompress recv_chunk to input_chunk
        n = input_chunk.numel()

        self.decompress(recv_chunk, interm_chunk, n)

        input_chunk.copy_(interm_chunk[:n])

    def dec_compression(self, recv_chunk, input_chunk, send_chunk, interm_chunk): # recv_chunk, input_chunk, send_chunk, interm_chunk are contiguous tensors. Decompress recv_chunk to input_chunk, add input_chunk to interm_chunk, compress interm_chunk to send_chunk
        n = input_chunk.numel()
        self.decompress(recv_chunk, interm_chunk, n)

        interm_chunk[:n].add_(input_chunk)
        self.compress(interm_chunk, send_chunk, n)
"""

"""
input_tensor is guaranteed to be a multiple of self.chunk_size
"""
class Aee_Dynamic_Range(object):
    def __init__(self, nbits, chunk_size, params):
        self.nbits = nbits
        self.chunk_size = chunk_size
        self.range = (1 << (self.nbits - 1)) - 1
        self.params = params
        self.cur_max_size = 0
        self.nclients = dist.get_world_size()
        self.client_rank = dist.get_rank()

        for d_index in params["d"]:
            self.cur_max_size = max(self.cur_max_size, (params["d"][d_index] + chunk_size) // chunk_size)
        
        self.three_times_cur_max_size = 3 * self.cur_max_size + 2 * self.chunk_size
        
        self.randomized_vec_pool = torch.rand(dtype=torch.float16, size=(self.three_times_cur_max_size, chunk_size), device="cuda") + 1 # NOTE: range is [1, 2]. Formula: floor(x) + ceil(prob - frac_x) = floor(x) + floor(prob + 1 - (x-floor(x)) = 2floor(x) + floor(prob + 1 - x)
        self.rand_seed = (self.nbits * 998244353) & 0xFFFFFFFF

    def gen_next_rand(self):
        self.rand_seed = (self.rand_seed * 671431 + 1000000007) & 0xFFFFFFFF
        return self.rand_seed
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()
        padded_n = (n + self.chunk_size - 1) // self.chunk_size * self.chunk_size

        if dtype == torch.uint8: # scales are encoded as torch.float16
            padded_vec_size = padded_n // (8 // self.nbits)
            padded_vec_size += (padded_n // self.chunk_size) * 2 # 2 bytes for each chunk

        else:
            padded_vec_size = padded_n

        return torch.zeros(padded_vec_size, dtype=dtype, device="cuda")
    
    def compress(self, interm_chunk, send_chunk, tmp_chunk, n): # tmp chunk is of the same type as interm_chunk: float16
        padded_n = (n + self.chunk_size - 1) // self.chunk_size * self.chunk_size
        padded_vec_size = padded_n // (8 // self.nbits)
        n_scale = padded_n // self.chunk_size * 2
        # pdb.set_trace()
        
        two_dimen_interm_chunk = interm_chunk.view(-1, self.chunk_size)

        randl_mod = self.randomized_vec_pool.numel() // self.chunk_size - two_dimen_interm_chunk.numel() // self.chunk_size
        
        randl = self.gen_next_rand() % randl_mod
        rand_vec = self.randomized_vec_pool[randl:randl + two_dimen_interm_chunk.numel() // self.chunk_size, :]
        
        tmp = torch.abs(two_dimen_interm_chunk)
        max_val = torch.max(tmp, dim=1, keepdim=True)[0]
        # max_val = torch.max(two_dimen_interm_chunk, dim=1, keepdim=True)[0]
        prob_pow = max_val.div_(self.range).add_(1e-7)   # NOTE: 5.9e-8 is the smallest positive subnormal number in float16

        send_chunk_scales = send_chunk[padded_vec_size:padded_vec_size + n_scale].view(torch.float16)
        send_chunk_scales.copy_(prob_pow.view(-1))

        two_dimen_interm_chunk.div_(prob_pow)

        two_dimen_interm_chunk2 = tmp_chunk[:padded_n].view(-1, self.chunk_size)
        
        # two_dimen_interm_chunk2.copy_(two_dimen_interm_chunk)
        # two_dimen_interm_chunk.floor_()
        # prob = two_dimen_interm_chunk2.sub_(two_dimen_interm_chunk)

        # two_dimen_interm_chunk.add_(rand_vec <= prob)
        # # two_dimen_interm_chunk.add_(prob.sub_(rand_vec).ceil_())  # prob, rand_vec in (0, 1). Goal: return 1 if rand_vec <= prob and 0 otherwise. if rand_vec <= prob, then prob - rand_vec >= 0, so ceil is equivalent to adding 1. if rand_vec > prob, then prob - rand_vec < 0, so ceil is equivalent to adding 0


        two_dimen_interm_chunk2.copy_(two_dimen_interm_chunk)
        torch.floor(two_dimen_interm_chunk2, out=two_dimen_interm_chunk)
        two_dimen_interm_chunk.mul_(2).sub_(two_dimen_interm_chunk2).add_(rand_vec)  # 2floor(x) + floor(prob + 1 - x)


        if self.nbits == 8:
            send_chunk[:padded_vec_size].copy_(interm_chunk)
        elif self.nbits == 4:
            floattoint4(interm_chunk, send_chunk[:padded_vec_size], padded_n)
        else:
            floattoint2(interm_chunk, send_chunk[:padded_vec_size], padded_n)

    def compression(self, input_chunk, send_chunk, interm_chunk=None): # compress input_chunk to send_chunk
        n = input_chunk.numel()
        # interm_chunk[:n].copy_(input_chunk)
        # interm_chunk[n:].fill_(0)
        # pdb.set_trace()
        
        # m = min(input_chunk.numel(), send_chunk.numel())
        # send_chunk[:m].copy_(input_chunk[:m])
        self.compress(input_chunk, send_chunk, interm_chunk, n)
        
    
    def decompress(self, recv_chunk, interm_chunk, n):
        padded_n = (n + self.chunk_size - 1) // self.chunk_size * self.chunk_size
        padded_vec_size = padded_n // (8 // self.nbits)

        n_scale = padded_n // self.chunk_size * 2
        scale_chunk = recv_chunk[padded_vec_size:padded_vec_size + n_scale].view(torch.float16).view(-1, 1)  # shared memory

        if self.nbits == 8:
            interm_chunk.copy_(recv_chunk[:padded_vec_size])
        elif self.nbits == 4:
            int4tofloat(recv_chunk[:padded_vec_size], interm_chunk, padded_n)
        else:
            int2tofloat(recv_chunk[:padded_vec_size], interm_chunk, padded_n)

        data_chunk = interm_chunk.view(-1, self.chunk_size)  # shared memory
        data_chunk.mul_(scale_chunk)

    def decompression(self, recv_chunk, input_chunk, interm_chunk=None): # recv_chunk, input_chunk, interm chunk are contiguous tensors. Decompress recv_chunk to input_chunk
        n = input_chunk.numel()
        # m = min(recv_chunk.numel(), input_chunk.numel())
        # input_chunk[:m].copy_(recv_chunk[:m])
        self.decompress(recv_chunk, input_chunk, n)

    def dec_compression(self, recv_chunk, input_chunk, send_chunk, interm_chunk=None, interm_chunk2=None): # recv_chunk, input_chunk, send_chunk, interm_chunk are contiguous tensors. Decompress recv_chunk to interm_chunk, add interm_chunk to  input_chunk, compress input_chunk to send_chunk
        n = input_chunk.numel()
        self.decompress(recv_chunk, interm_chunk[:n], n)
        
        input_chunk.add_(interm_chunk[:n])
        if interm_chunk2 != None: # input_chunk is not modifiable
            interm_chunk2[:n].copy_(input_chunk)
            self.compress(interm_chunk2[:n], send_chunk, interm_chunk, n)
        else: # input_chunk is modifiable
            self.compress(input_chunk, send_chunk, interm_chunk, n)
        

from . import eden_utils


def superchunk_mean_center(input_tensor, residual, stats, n, world_size, superchunk_size):
    superchunk_op = getattr(eden_utils, "superchunk_mean_center", None)
    if superchunk_op is None:
        raise RuntimeError("eden_utils must be rebuilt with superchunk_mean_center support")
    superchunk_op(
        input_tensor,
        residual,
        stats,
        n,
        world_size,
        superchunk_size,
        torch.cuda.current_stream().cuda_stream,
    )


def superchunk_add_mean_copy(residual, output_tensor, stats, n, superchunk_size):
    add_mean_op = getattr(eden_utils, "superchunk_add_mean_copy", None)
    if add_mean_op is None:
        raise RuntimeError("eden_utils must be rebuilt with superchunk_add_mean_copy support")
    add_mean_op(
        residual,
        output_tensor,
        stats,
        n,
        superchunk_size,
        torch.cuda.current_stream().cuda_stream,
    )


class Aee_Dynamic_Range_GPU_kern(object):
    def __init__(self, nbits, chunk_size, params):
        self.nbits = nbits
        self.chunk_size = chunk_size
        self.strategy = 2
        self.supergroup = 16
        self.range = (1 << (self.nbits - 1)) - 1
        self.params = params
        self.cur_max_size = 0
        self.nclients = dist.get_world_size()
        self.client_rank = dist.get_rank()

        for d_index in params["d"]:
            self.cur_max_size = max(self.cur_max_size, (params["d"][d_index] + chunk_size) // chunk_size)
        
        self.three_times_cur_max_size = (3 * self.cur_max_size) // self.nclients
        if "correlated" in params["args"].aggregation_method:
            random_obj = torch.load("/cluster/project2/gcreduce_data/data/correlated_rand/obj_{}_{}.pt".format(self.nclients, self.client_rank))
            self.randomized_vec_pool = random_obj.to(device="cuda")[:self.three_times_cur_max_size * self.chunk_size].view(self.three_times_cur_max_size, self.chunk_size)
        else:
            self.randomized_vec_pool = torch.rand(dtype=torch.bfloat16, size=(self.three_times_cur_max_size, chunk_size), device="cuda") # NOTE: range is [0, 1]
        self.randomized_vec_pools = {self.randomized_vec_pool.dtype: self.randomized_vec_pool}
        self.rand_seed = (self.nbits * 998244353) & 0xFFFFFFFF



    def gen_next_rand(self):
        self.rand_seed = (self.rand_seed * 671431 + 1000000007) & 0xFFFFFFFF
        return self.rand_seed

    def padded_n(self, n):
        superchunk_elems = self.chunk_size * self.supergroup
        return (n + superchunk_elems - 1) // superchunk_elems * superchunk_elems

    def compressed_numel(self, n):
        padded_n = self.padded_n(n)
        packed_bytes = (self.chunk_size * self.nbits) // 8
        superchunks = padded_n // (self.chunk_size * self.supergroup)
        return superchunks * (self.supergroup * (packed_bytes + 1) + 2)

    def rand_pool(self, n, dtype):
        pool = self.randomized_vec_pools.get(dtype)
        if pool is None:
            pool = self.randomized_vec_pool.to(dtype=dtype)
            self.randomized_vec_pools[dtype] = pool

        randl_mod = pool.numel() // self.chunk_size - n // self.chunk_size - 1
        if randl_mod <= 0:
            raise RuntimeError("randomized_vec_pool is too small for scaling quantization")

        randl = self.gen_next_rand() % randl_mod
        return pool[randl:randl + n // self.chunk_size, :]
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()
        padded_n = self.padded_n(n)

        if dtype == torch.uint8:
            padded_vec_size = self.compressed_numel(n)

        else:
            padded_vec_size = padded_n

        return torch.zeros(padded_vec_size, dtype=dtype, device="cuda")
    
    def compression(self, input_chunk, send_chunk, interm_chunk=None): # compress input_chunk to send_chunk
        n = input_chunk.numel()
        if n == 0:
            return

        rand_vec = self.rand_pool(n, input_chunk.dtype)
        eden_utils.scaling_compress(input_chunk, send_chunk, rand_vec, n, self.nbits, self.chunk_size, torch.cuda.current_stream().cuda_stream, self.strategy)
        
    
    
    def decompression(self, recv_chunk, input_chunk, interm_chunk=None): # recv_chunk, input_chunk, interm chunk are contiguous tensors. Decompress recv_chunk to input_chunk
        n = input_chunk.numel()
        if n == 0:
            return
        
        # pdb.set_trace()
        eden_utils.scaling_decompress(recv_chunk, input_chunk, n, self.nbits, self.chunk_size, torch.cuda.current_stream().cuda_stream, self.strategy)
        

    def dec_compression(self, recv_chunk, input_chunk, send_chunk, interm_chunk=None, interm_chunk2=None): # recv_chunk, input_chunk, send_chunk, interm_chunk are contiguous tensors. Decompress recv_chunk to interm_chunk, add interm_chunk to  input_chunk, compress input_chunk to send_chunk
        n = input_chunk.numel()
        if n == 0:
            return

        rand_vec = self.rand_pool(n, input_chunk.dtype)
        eden_utils.scaling_dec_comp(recv_chunk, input_chunk, send_chunk, rand_vec, n, self.nbits, self.chunk_size, torch.cuda.current_stream().cuda_stream, self.strategy)

    def decompression_add(self, recv_chunk, input_chunk, interm_chunk=None):
        n = input_chunk.numel()
        if n == 0:
            return
        
        eden_utils.scaling_decompress_add(recv_chunk, input_chunk, n, self.nbits, self.chunk_size, torch.cuda.current_stream().cuda_stream, self.strategy)



class Mee_Dynamic_Range_GPU_kern(object):
    def __init__(self, nbits, chunk_size, params):
        self.nbits = nbits
        self.chunk_size = chunk_size
        self.strategy = 3
        self.supergroup = 16
        self.range = (1 << (self.nbits - 1)) - 1
        self.params = params
        self.cur_max_size = 0
        self.nclients = dist.get_world_size()
        self.client_rank = dist.get_rank()

        for d_index in params["d"]:
            self.cur_max_size = max(self.cur_max_size, (params["d"][d_index] + chunk_size) // chunk_size)
        
        self.three_times_cur_max_size = (3 * self.cur_max_size) // self.nclients
        if "correlated" in params["args"].aggregation_method:
            random_obj = torch.load("/cluster/project2/gcreduce_data/data/correlated_rand/obj_{}_{}.pt".format(self.nclients, self.client_rank))
            self.randomized_vec_pool = random_obj.to(device="cuda")[:self.three_times_cur_max_size * self.chunk_size].view(self.three_times_cur_max_size, self.chunk_size).to(dtype=torch.float16)
        else:
            self.randomized_vec_pool = torch.rand(dtype=torch.bfloat16, size=(self.three_times_cur_max_size, chunk_size), device="cuda") # NOTE: range is [0, 1]
        self.randomized_vec_pools = {self.randomized_vec_pool.dtype: self.randomized_vec_pool}
        self.rand_seed = (self.nbits * 998244353) & 0xFFFFFFFF


        eden_utils.init_lookup_table()

    def gen_next_rand(self):
        self.rand_seed = (self.rand_seed * 671431 + 1000000007) & 0xFFFFFFFF
        return self.rand_seed

    def padded_n(self, n):
        superchunk_elems = self.chunk_size * self.supergroup
        return (n + superchunk_elems - 1) // superchunk_elems * superchunk_elems

    def compressed_numel(self, n):
        padded_n = self.padded_n(n)
        packed_bytes = (self.chunk_size * self.nbits) // 8
        superchunks = padded_n // (self.chunk_size * self.supergroup)
        return superchunks * (self.supergroup * (packed_bytes + 1) + 2)

    def rand_pool(self, n, dtype):
        pool = self.randomized_vec_pools.get(dtype)
        if pool is None:
            pool = self.randomized_vec_pool.to(dtype=dtype)
            self.randomized_vec_pools[dtype] = pool

        randl_mod = pool.numel() // self.chunk_size - n // self.chunk_size - 1
        if randl_mod <= 0:
            raise RuntimeError("randomized_vec_pool is too small for scaling quantization")

        randl = self.gen_next_rand() % randl_mod
        return pool[randl:randl + n // self.chunk_size, :]
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()
        padded_n = self.padded_n(n)

        if dtype == torch.uint8:
            padded_vec_size = self.compressed_numel(n)

        else:
            padded_vec_size = padded_n

        return torch.zeros(padded_vec_size, dtype=dtype, device="cuda")
    
    def compression(self, input_chunk, send_chunk, interm_chunk=None): # compress input_chunk to send_chunk
        n = input_chunk.numel()

        if n == 0:
            return

        rand_vec = self.rand_pool(n, input_chunk.dtype)
        eden_utils.scaling_compress(input_chunk, send_chunk, rand_vec, n, self.nbits, self.chunk_size, torch.cuda.current_stream().cuda_stream, self.strategy)
        
    
    
    def decompression(self, recv_chunk, input_chunk, interm_chunk=None): # recv_chunk, input_chunk, interm chunk are contiguous tensors. Decompress recv_chunk to input_chunk
        n = input_chunk.numel()
        if n == 0:
            return

        # pdb.set_trace()
        eden_utils.scaling_decompress(recv_chunk, input_chunk, n, self.nbits, self.chunk_size, torch.cuda.current_stream().cuda_stream, self.strategy)
        

    def dec_compression(self, recv_chunk, input_chunk, send_chunk, interm_chunk=None, interm_chunk2=None): # recv_chunk, input_chunk, send_chunk, interm_chunk are contiguous tensors. Decompress recv_chunk to interm_chunk, add interm_chunk to  input_chunk, compress input_chunk to send_chunk
        n = input_chunk.numel()
        if n == 0:
            return

        rand_vec = self.rand_pool(n, input_chunk.dtype)
        eden_utils.scaling_dec_comp(recv_chunk, input_chunk, send_chunk, rand_vec, n, self.nbits, self.chunk_size, torch.cuda.current_stream().cuda_stream, self.strategy)

    def decompression_add(self, recv_chunk, input_chunk, interm_chunk=None):
        n = input_chunk.numel()

        if n == 0:
            return
        eden_utils.scaling_decompress_add(recv_chunk, input_chunk, n, self.nbits, self.chunk_size, torch.cuda.current_stream().cuda_stream, self.strategy)





class Float16_compression(object):
    def __init__(self, params):
        self.params = params
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()

        return torch.zeros(n, dtype=dtype, device="cuda")
    
    def compression(self, input_chunk, send_chunk, interm_chunk=None): # send_chunk in float16; input_chunk in float32
        n = input_chunk.numel()
        send_chunk[:n].copy_(input_chunk)

    def decompression(self, recv_chunk, input_chunk, interm_chunk=None):
        n = input_chunk.numel()
        input_chunk.copy_(recv_chunk[:n])
    
    def dec_compression(self, recv_chunk, input_chunk, send_chunk, interm_chunk=None, interm_chunk2=None):
        n = input_chunk.numel()
        input_chunk.add_(recv_chunk[:n])
        send_chunk[:n].copy_(input_chunk)


class BFloat16_compression(object):
    def __init__(self, params):
        self.params = params
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()

        return torch.zeros(n, dtype=dtype, device="cuda")
    
    def compression(self, input_chunk, send_chunk, interm_chunk=None): # send_chunk in float16; input_chunk in float32
        n = input_chunk.numel()
        send_chunk[:n].copy_(input_chunk)

    def decompression(self, recv_chunk, input_chunk, interm_chunk=None):
        n = input_chunk.numel()
        input_chunk.copy_(recv_chunk[:n])
    
    def dec_compression(self, recv_chunk, input_chunk, send_chunk, interm_chunk=None, interm_chunk2=None):
        n = input_chunk.numel()
        input_chunk.add_(recv_chunk[:n])
        send_chunk[:n].copy_(input_chunk)


class Direct_Summation(object):
    def __init__(self, params):
        self.params = params

    def create_tensor(self, tensor, dtype):
        return torch.zeros(tensor.numel(), dtype=dtype, device="cuda")

    def compression(self, input_chunk, send_chunk, interm_chunk=None):
        n = input_chunk.numel()
        send_chunk[:n].copy_(input_chunk)

    def decompression(self, recv_chunk, input_chunk, interm_chunk=None):
        n = input_chunk.numel()
        input_chunk.copy_(recv_chunk[:n])

    def dec_compression(self, recv_chunk, input_chunk, send_chunk, interm_chunk=None, interm_chunk2=None):
        n = input_chunk.numel()
        input_chunk.add_(recv_chunk[:n])
        send_chunk[:n].copy_(input_chunk)


class Float32_compression(object):
    def __init__(self, params):
        self.params = params
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()

        return torch.zeros(n, dtype=dtype, device="cuda")
    
    def compression(self, input_chunk, send_chunk, interm_chunk=None): # send_chunk in float32; input_chunk in float32
        n = input_chunk.numel()
        send_chunk[:n].copy_(input_chunk)

    def decompression(self, recv_chunk, input_chunk, interm_chunk=None):
        n = input_chunk.numel()
        input_chunk.copy_(recv_chunk[:n])
    
    def dec_compression(self, recv_chunk, input_chunk, send_chunk, interm_chunk=None, interm_chunk2=None):
        n = input_chunk.numel()
        input_chunk.add_(recv_chunk[:n])
        send_chunk[:n].copy_(input_chunk)


class Float8_compression(object):
    def __init__(self, params):
        self.params = params
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()

        return torch.zeros(n, dtype=dtype, device="cuda")
    
    def compression(self, input_chunk, send_chunk, interm_chunk=None): # send_chunk in float8_e4m3fn
        n = input_chunk.numel()
        send_chunk[:n].copy_(input_chunk)

    def decompression(self, recv_chunk, input_chunk, interm_chunk=None): # recv_chunk in float8_e4m3fn
        n = input_chunk.numel()
        input_chunk.copy_(recv_chunk[:n])
    
    def dec_compression(self, recv_chunk, input_chunk, send_chunk, interm_chunk=None, interm_chunk2=None):
        n = input_chunk.numel()
        dtype = input_chunk.dtype
        input_chunk.add_(recv_chunk[:n].to(dtype))
        send_chunk[:n].copy_(input_chunk)
