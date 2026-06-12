import importlib
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def import_eden_utils():
    for path in (ROOT, ROOT / "src", ROOT / "src" / "srrcomp"):
        sys.path.insert(0, str(path))
    try:
        return importlib.import_module("eden_utils")
    except ImportError:
        return importlib.import_module("srrcomp.eden_utils")


eden_utils = import_eden_utils()


def required_bytes(n, nbits, chunk_size, strategy):
    chunks = n // chunk_size
    packed = (chunk_size * nbits) // 8
    if strategy >= 2:
        return (chunks // 16) * (16 * (packed + 1) + 2)
    return chunks * (packed + 2)


def make_input(n, dtype):
    idx = torch.arange(n, dtype=torch.float32, device="cuda")
    values = torch.sin(idx * 0.017) * 1.7 + torch.cos(idx * 0.0071) * 0.3
    values += (idx.remainder(7) - 3.0) * 0.013
    return values.to(dtype)


def make_rand(n, dtype):
    return torch.rand(n, dtype=torch.float32, device="cuda").to(dtype)

def time_cuda(name, fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push(name)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    return start.elapsed_time(end) / iters


def main():
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"
    torch.manual_seed(2026)

    stream = torch.cuda.current_stream().cuda_stream

    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    print("dtype,chunk_size,nbits,strategy,compress_ms,decompress_ms,split_dec_comp_ms,fused_dec_comp_ms", flush=True)

    for chunk_size in (16, ):
        n = chunk_size * 16 * 65536 * 16
        for dtype in (torch.bfloat16,):
            src = make_input(n, dtype).contiguous()
            inp_template = (src.float() * -0.25).to(dtype).contiguous()
            rand_pool = make_rand(n, dtype).contiguous()
            for nbits in (2, 4, 8):
                for strategy in (2, 3):
                    encoded = torch.zeros(required_bytes(n, nbits, chunk_size, strategy), dtype=torch.uint8, device="cuda")
                    send = torch.zeros_like(encoded)
                    send_split = torch.zeros_like(encoded)
                    decoded = torch.empty_like(src)
                    inp = inp_template.clone()
                    inp_split = inp_template.clone()

                    eden_utils.scaling_compress(src, encoded, rand_pool, n, nbits, chunk_size, stream, strategy)
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_push("profile")
                    compress_ms = time_cuda(f"compress_chunk{chunk_size}_bits{nbits}",
                        lambda: eden_utils.scaling_compress(src, encoded, rand_pool, n, nbits, chunk_size, stream, strategy)
                    )
                    decompress_ms = time_cuda(f"decompress_chunk{chunk_size}_bits{nbits}",
                        lambda: eden_utils.scaling_decompress(encoded, decoded, n, nbits, chunk_size, stream, strategy)
                    )
                    split_dec_comp_ms = time_cuda(f"split_dec_comp_chunk{chunk_size}_bits{nbits}",
                        lambda: (
                            eden_utils.scaling_decompress_add(encoded, inp_split, n, nbits, chunk_size, stream, strategy),
                            eden_utils.scaling_compress(inp_split, send_split, rand_pool, n, nbits, chunk_size, stream, strategy),
                        ),
                        warmup=5,
                        iters=25,
                    )
                    fused_dec_comp_ms = time_cuda(f"fused_dec_comp_chunk{chunk_size}_bits{nbits}",
                        lambda: eden_utils.scaling_dec_comp(encoded, inp, send, rand_pool, n, nbits, chunk_size, stream, strategy),
                        warmup=5,
                        iters=25,
                    )
                    torch.cuda.nvtx.range_pop()

                    print(
                        f"{dtype},{chunk_size},{nbits},{strategy},"
                        f"{compress_ms:.4f},{decompress_ms:.4f},{split_dec_comp_ms:.4f},{fused_dec_comp_ms:.4f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
