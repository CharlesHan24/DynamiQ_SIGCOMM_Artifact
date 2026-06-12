import torch.distributed as dist
import torch
import pdb
import sys
import time



def async_send_recv(send_buff, recv_buff, left, right, tag, sliced_portion, params):
    # pdb.set_trace()
    if send_buff.dtype == torch.float8_e4m3fn:
        send_buff = send_buff.view(torch.uint8)
        recv_buff = recv_buff.view(torch.uint8)

    
    orig_send_buff = send_buff
    orig_recv_buff = recv_buff
    m = send_buff.numel()
    n = int(send_buff.numel() * sliced_portion)
    
    send_buff = send_buff[m-n:]
    recv_buff = recv_buff[m-n:]

    # send_op = dist.isend(send_buff, right, tag=tag, group=params["pg1"])
    # recv_op = dist.irecv(recv_buff, left, tag=tag + 1, group=params["pg2"])

    # send_op.wait()
    # recv_op.wait()
    
    # for i in range(4): # TODO
    send_op = dist.P2POp(dist.isend, send_buff, right, tag=tag)
    recv_op = dist.P2POp(dist.irecv, recv_buff, left, tag=tag)

    reqs = dist.batch_isend_irecv([send_op, recv_op])
    
    for req in reqs:
        req.wait()

def decompression_add(recv_chunk, input_chunk, callback_comm, **kwargs):
    interm_chunk = input_chunk.clone()
    callback_comm.decompression(recv_chunk, interm_chunk, **kwargs)
    input_chunk.add_(interm_chunk)

def calc_order(rank, size):
    execution_list = []
    for i in range(size): # nclients
        execution_list.append([])

    all_workers = [i for i in range(size)]
    relative_order = []
    while len(all_workers) != 1:
        new_all_workers = []
        for i in range(len(all_workers) // 2):
            relative_order.append([all_workers[i * 2], all_workers[i * 2 + 1]])
            new_all_workers.append(all_workers[i * 2 + 1])
        all_workers = new_all_workers

    for i in range(size):
        for j in range(len(relative_order)):
            execution_list[j].append([(relative_order[j][0] + i) % size, (relative_order[j][1] + i) % size, i]) # at time j, what happens at chunk i
    
    res_orders = []
    for j in range(size - 1):
        for i in range(len(execution_list[j])):
            if execution_list[j][i][0] == rank:
                src_rank = execution_list[j][i][2]
                right = execution_list[j][i][1]
        
        for i in range(len(execution_list[j])):
            if execution_list[j][i][1] == rank:
                dst_rank = execution_list[j][i][2]
                left = execution_list[j][i][0]
        res_orders.append([src_rank, right, dst_rank, left])
    
    return res_orders


def composable_allreduce_arbitrary_callback(send_vec: torch.Tensor, callback_comm, params, tag=0, dtype=torch.uint8, sliced_portion = 1):    # callback_comm should consist of exactly 3 functions: compression, decompression, and aggregation
    # buffers for intermediate tensor are always in int8; we reinterpret the tensor to int4, int 32 etc. in the callback functions
    # tag = index << 8 | i
    # IMPORTANT: ensure that the chunk_size is a multiple of params["chunk_size"]. For FP16 and FP32 we don't actually care; for AEE, the preprocessing ensures that send_vec.numel() is a multiple of chunk_size
    # pdb.set_trace()
    def _init(tag, input_chunk, dtype, callback_comm, rank, size):
        if "buffer" not in params:
            params["buffer"] = {}

        if tag not in params["buffer"]:
            params["buffer"][tag] = {}
            
            params["buffer"][tag]["send_chunk"] = callback_comm.create_tensor(input_chunk[0], dtype)
            params["buffer"][tag]["recv_chunk"] = callback_comm.create_tensor(input_chunk[0], dtype)
        


    if send_vec.numel() == 0:
        return send_vec

    rank = dist.get_rank()
    size = dist.get_world_size()

    if "coov" in params["args"].aggregation_method:
        sliced_portion = 0

    

    if "chunk_size" not in params:
        params["chunk_size"] = 64

    chunk_size = (send_vec.numel() + size - 1) // size
    chunk_size = (chunk_size + params["chunk_size"] - 1) // params["chunk_size"] * params["chunk_size"]

    input_chunk = []
    for i in range(size):
        input_chunk.append(send_vec[chunk_size * i: min(chunk_size * (i + 1), send_vec.numel())])
    
    # send_chunk, receive_chunk, interm_chunk

    _init(tag, input_chunk, dtype, callback_comm, rank, size)
    
    
    send_chunk = params["buffer"][tag]["send_chunk"]
    recv_chunk = params["buffer"][tag]["recv_chunk"]

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    # if tag & 1 == 0:
    #     return send_vec

    tag <<= 8 # world size <= 128. the actual tag is tag | step_num
    # pdb.set_trace()

    for i in range(size - 1):
        if i == 0:
            callback_comm.compression(input_chunk[rank], send_chunk)
        else:
            # callback_comm.compression(input_chunk[rank], send_chunk, interm_chunk)
            # send_chunk.add_(recv_chunk)
            callback_comm.dec_compression(recv_chunk, input_chunk[(rank - i + size) % size], send_chunk)
        async_send_recv(send_chunk, recv_chunk, left, right, tag | i, sliced_portion=sliced_portion, params=params) 
   
    # callback_comm.dec_compression(recv_chunk, input_chunk[(rank + 1) % size], send_chunk) # decompression
    callback_comm.decompression(recv_chunk, input_chunk[(rank + 1) % size]) # decompression
    callback_comm.compression(input_chunk[(rank + 1) % size], send_chunk) # compression

    chunks = [send_chunk, recv_chunk]
    for i in range(size - 1):
        async_send_recv(chunks[(i & 1)], chunks[(i ^ 1) & 1], left, right, tag=tag | (size + i), sliced_portion=sliced_portion, params=params)
        callback_comm.decompression(chunks[(i ^ 1) & 1], input_chunk[(rank - i + size) % size])
    
    return send_vec


def composable_butterfly_allreduce_callback(send_vec: torch.Tensor, callback_comm, params, tag=0, dtype=torch.uint8, sliced_portion = 1):
    # buffers for intermediate tensor are always in int8; we reinterpret the tensor to int4, int 32 etc. in the callback functions
    # tag = index << 8 | i
    # IMPORTANT: ensure that the chunk_size is a multiple of params["chunk_size"]. For FP16 and FP32 we don't actually care; for AEE, the preprocessing ensures that send_vec.numel() is a multiple of chunk_size

    def _init(tag, input_chunk, dtype, callback_comm, rank, size):
        if "buffer" not in params:
            params["buffer"] = {}

        if tag not in params["buffer"]:
            params["buffer"][tag] = {}
            params["buffer"][tag]["send_chunk"] = callback_comm.create_tensor(input_chunk[0], dtype)
            params["buffer"][tag]["recv_chunk"] = callback_comm.create_tensor(input_chunk[0], dtype)

        if "order" not in params:
            params["order"] = calc_order(rank, size)

    if send_vec.numel() == 0:
        return send_vec

    rank = dist.get_rank()
    size = dist.get_world_size()

    # pdb.set_trace()

    if "coov" in params["args"].aggregation_method:
        sliced_portion = 0
    
    if "chunk_size" not in params:
        params["chunk_size"] = 64

    chunk_size = (send_vec.numel() + size - 1) // size
    chunk_size = (chunk_size + params["chunk_size"] - 1) // params["chunk_size"] * params["chunk_size"]

    input_chunk = []
    for i in range(size):
        input_chunk.append(send_vec[chunk_size * i: min(chunk_size * (i + 1), send_vec.numel())])
    
    # send_chunk, receive_chunk, interm_chunk

    _init(tag, input_chunk, dtype, callback_comm, rank, size)

    order = params["order"]
    
    
    send_chunk = params["buffer"][tag]["send_chunk"]
    recv_chunk = params["buffer"][tag]["recv_chunk"]

    tag <<= 8 # world size <= 128. the actual tag is tag | step_num

    for i in range(size - 1):   # compress -> transmit -> decompress
        src_rank = order[i][0]
        right = order[i][1]
        dst_rank = order[i][2]
        left = order[i][3]

        callback_comm.compression(input_chunk[src_rank], send_chunk)

        async_send_recv(send_chunk, recv_chunk, left, right, tag | i, sliced_portion=sliced_portion, params=params) # compression

        decompression_add(recv_chunk, input_chunk[dst_rank], callback_comm) # decompression
    
    src_rank = order[-1][2]
    left = ((rank + 1) + size) % size
    right = (rank - 1) % size
    callback_comm.compression(input_chunk[src_rank], send_chunk) # decompression

    chunks = [send_chunk, recv_chunk]
    for i in range(size - 1):
        async_send_recv(chunks[(i & 1)], chunks[(i ^ 1) & 1], left, right, tag=tag | (size + i), sliced_portion=sliced_portion, params=params)
        callback_comm.decompression(chunks[(i ^ 1) & 1], input_chunk[(src_rank - 1 - i + (size << 1)) % size])
    
    return send_vec

