import torch
import eden_utils
import random
import pdb

n = 512 + 64
src = torch.tensor([i for i in range(n * 2)], dtype=torch.float16, device='cuda')

dst = torch.zeros(357284, dtype=torch.uint8, device='cuda')

res = torch.zeros(n * 2, dtype=torch.float16, device='cuda')

rand_pool = torch.rand(n * 2, dtype=torch.float16, device='cuda')

eden_utils.scaling_compress(src, dst, rand_pool, n, 4, 64, torch.cuda.current_stream().cuda_stream)
eden_utils.scaling_decompress(dst, res, n, 4, 64, torch.cuda.current_stream().cuda_stream)

pdb.set_trace()
print(dst.numel())
print(dst)