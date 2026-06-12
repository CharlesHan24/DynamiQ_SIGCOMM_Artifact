import importlib
import math
import sys
from pathlib import Path

import numpy as np
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


def h(value):
    return np.float16(value)


def h_add(a, b):
    return h(h(a) + h(b))


def h_sub(a, b):
    return h(h(a) - h(b))


def h_mul(a, b):
    return h(h(a) * h(b))


def h_div(a, b):
    return h(h(a) / h(b))


def h_log2(a):
    return h(math.log2(float(h(a))))


def required_bytes(n, nbits, chunk_size, strategy):
    chunks = n // chunk_size
    packed = (chunk_size * nbits) // 8
    if strategy >= 2:
        return (chunks // 16) * (16 * (packed + 1) + 2)
    return chunks * (packed + 2)


def mee_params(nbits):
    epses = {2: 0.23, 4: 0.18, 8: 0.1}
    eps = epses[nbits]
    base = 1.0 + 2.0 * eps * eps
    one_plus_eps2_over_2eps2 = h((1.0 + eps * eps) / (2.0 * eps * eps))
    log2_base = h(math.log2(base))
    anchor = (1 << (nbits - 1)) - 1
    table = [h(0.0) for _ in range(1 << nbits)]
    table[anchor] = h(0.0)
    pow_eps2 = 1.0
    for j in range(1, anchor + 1):
        pow_eps2 *= base
        value = h((pow_eps2 - 1.0) / (2.0 * eps * eps) * (1.0 + eps * eps))
        table[anchor + j] = value
        table[anchor - j] = h(-value)
    return {
        "table": table,
        "chunk_max_mee": table[2 * anchor],
        "one_plus_eps2_over_2eps2": one_plus_eps2_over_2eps2,
        "log2_base": log2_base,
    }


def stochastic_code(value, rand_value, nbits):
    value = h(value)
    floored = math.floor(float(value))
    frac = h_sub(value, h(floored))
    increment = math.ceil(float(h_sub(frac, rand_value)))
    signed_value = floored + increment
    value_range = (1 << (nbits - 1)) - 1
    signed_value = max(-value_range, min(value_range, signed_value))
    return signed_value + value_range


def mee_inverse_lattice(value, params):
    value = h(value)
    abs_value = h(abs(float(value)))
    if not (float(abs_value) > 0.0):
        return h(0.0)
    log_arg = h_add(h_div(abs_value, params["one_plus_eps2_over_2eps2"]), h(1.0))
    lattice_value = h_div(h_log2(log_arg), params["log2_base"])
    return h(-lattice_value if float(value) < 0.0 else lattice_value)


def bf16(value):
    return float(torch.tensor(float(value), dtype=torch.float32).to(torch.bfloat16).float().item())


def stochastic_code_float(value, rand_value, nbits):
    floored = math.floor(float(value))
    increment = math.ceil((float(value) - floored) - float(rand_value))
    signed_value = floored + increment
    value_range = (1 << (nbits - 1)) - 1
    signed_value = max(-value_range, min(value_range, signed_value))
    return signed_value + value_range


def mee_inverse_lattice_float(value, params):
    abs_value = abs(float(value))
    if not (abs_value > 0.0):
        return 0.0
    lattice_value = math.log2(abs_value / float(params["one_plus_eps2_over_2eps2"]) + 1.0) / float(params["log2_base"])
    return -lattice_value if float(value) < 0.0 else lattice_value


def reference_decompressed(src_np, rand_np, nbits, chunk_size, strategy):
    src_np = src_np.astype(np.float16, copy=False)
    rand_np = rand_np.astype(np.float16, copy=False)
    out = np.zeros_like(src_np, dtype=np.float16)
    value_range = (1 << (nbits - 1)) - 1
    chunks = src_np.size // chunk_size

    if strategy in (0, 2):
        raw_scales = []
        if strategy == 2:
            for chunk_id in range(chunks):
                start = chunk_id * chunk_size
                end = start + chunk_size
                chunk_absmax = h(max(abs(float(v)) for v in src_np[start:end]))
                raw_scales.append(h_add(h_div(chunk_absmax, h(value_range)), h(1e-7)))

        for chunk_id in range(chunks):
            start = chunk_id * chunk_size
            end = start + chunk_size
            chunk = src_np[start:end]
            chunk_absmax = h(max(abs(float(v)) for v in chunk))
            if strategy == 0:
                scale = h_add(h_div(chunk_absmax, h(value_range)), h(1e-7))
            else:
                super_start = (chunk_id // 16) * 16
                super_scales = raw_scales[super_start : super_start + 16]
                super_scale = h(max(float(v) for v in super_scales))
                q_scale = math.ceil(float(h_mul(h_div(raw_scales[chunk_id], super_scale), h(255.0))))
                q_scale = max(0, min(255, q_scale))
                super_scale_unit = h_div(super_scale, h(255.0))
                scale = h_mul(h(q_scale), super_scale_unit)
            for i, value in enumerate(chunk):
                normalized = h_div(value, scale)
                code = stochastic_code(normalized, rand_np[start + i], nbits)
                signed_code = code - value_range
                out[start + i] = h_mul(h(signed_code), scale)
        return out

    params = mee_params(nbits)
    raw_scales = []
    if strategy == 3:
        for chunk_id in range(chunks):
            start = chunk_id * chunk_size
            end = start + chunk_size
            chunk_absmax = h(max(abs(float(v)) for v in src_np[start:end]))
            raw_scales.append(h_add(h_div(chunk_absmax, params["chunk_max_mee"]), h(1e-7)))

    for chunk_id in range(chunks):
        start = chunk_id * chunk_size
        end = start + chunk_size
        chunk = src_np[start:end]
        chunk_absmax = h(max(abs(float(v)) for v in chunk))
        if strategy == 1:
            scale = h_add(h_div(chunk_absmax, params["chunk_max_mee"]), h(1e-7))
        else:
            super_start = (chunk_id // 16) * 16
            super_scales = raw_scales[super_start : super_start + 16]
            super_scale = h(max(float(v) for v in super_scales))
            q_scale = math.ceil(float(h_mul(h_div(raw_scales[chunk_id], super_scale), h(255.0))))
            q_scale = max(0, min(255, q_scale))
            super_scale_unit = h_div(super_scale, h(255.0))
            scale = h_mul(h(q_scale), super_scale_unit)
        for i, value in enumerate(chunk):
            normalized = h_div(value, scale)
            lattice_coord = mee_inverse_lattice(normalized, params)
            code = stochastic_code(lattice_coord, rand_np[start + i], nbits)
            out[start + i] = h_mul(params["table"][code], scale)
    return out


def reference_decompressed_bf16(src_np, rand_np, nbits, chunk_size, strategy):
    src_np = src_np.astype(np.float32, copy=False)
    rand_np = rand_np.astype(np.float32, copy=False)
    out = np.zeros_like(src_np, dtype=np.float32)
    value_range = (1 << (nbits - 1)) - 1
    chunks = src_np.size // chunk_size

    if strategy in (0, 2):
        raw_scales = []
        if strategy == 2:
            for chunk_id in range(chunks):
                start = chunk_id * chunk_size
                end = start + chunk_size
                chunk_absmax = max(abs(float(v)) for v in src_np[start:end])
                raw_scales.append(chunk_absmax / float(value_range) + 1e-7)

        for chunk_id in range(chunks):
            start = chunk_id * chunk_size
            end = start + chunk_size
            chunk = src_np[start:end]
            chunk_absmax = max(abs(float(v)) for v in chunk)
            if strategy == 0:
                scale = bf16(chunk_absmax / float(value_range) + 1e-7)
            else:
                super_start = (chunk_id // 16) * 16
                super_scales = raw_scales[super_start : super_start + 16]
                super_scale = max(float(v) for v in super_scales)
                q_scale = math.ceil(raw_scales[chunk_id] / super_scale * 255.0)
                q_scale = max(0, min(255, q_scale))
                super_scale_unit = bf16(super_scale / 255.0)
                scale = float(q_scale) * super_scale_unit
            for i, value in enumerate(chunk):
                normalized = float(value) / scale
                code = stochastic_code_float(normalized, rand_np[start + i], nbits)
                signed_code = code - value_range
                out[start + i] = bf16(float(signed_code) * scale)
        return out

    params = mee_params(nbits)
    raw_scales = []
    if strategy == 3:
        for chunk_id in range(chunks):
            start = chunk_id * chunk_size
            end = start + chunk_size
            chunk_absmax = max(abs(float(v)) for v in src_np[start:end])
            raw_scales.append(chunk_absmax / float(params["chunk_max_mee"]) + 1e-7)

    for chunk_id in range(chunks):
        start = chunk_id * chunk_size
        end = start + chunk_size
        chunk = src_np[start:end]
        chunk_absmax = max(abs(float(v)) for v in chunk)
        if strategy == 1:
            scale = bf16(chunk_absmax / float(params["chunk_max_mee"]) + 1e-7)
        else:
            super_start = (chunk_id // 16) * 16
            super_scales = raw_scales[super_start : super_start + 16]
            super_scale = max(float(v) for v in super_scales)
            q_scale = math.ceil(raw_scales[chunk_id] / super_scale * 255.0)
            q_scale = max(0, min(255, q_scale))
            super_scale_unit = bf16(super_scale / 255.0)
            scale = float(q_scale) * super_scale_unit
        for i, value in enumerate(chunk):
            normalized = float(value) / scale
            lattice_coord = mee_inverse_lattice_float(normalized, params)
            code = stochastic_code_float(lattice_coord, rand_np[start + i], nbits)
            out[start + i] = bf16(float(params["table"][code]) * scale)
    return out


def reference_decompressed_for_dtype(src, rand_pool, nbits, chunk_size, strategy, dtype):
    if dtype == torch.float16:
        return reference_decompressed(src.cpu().numpy(), rand_pool.cpu().numpy(), nbits, chunk_size, strategy)
    return reference_decompressed_bf16(
        src.float().cpu().numpy(),
        rand_pool.float().cpu().numpy(),
        nbits,
        chunk_size,
        strategy,
    )


def make_input(n, dtype):
    idx = torch.arange(n, dtype=torch.float32, device="cuda")
    values = torch.sin(idx * 0.17) * 1.7 + torch.cos(idx * 0.071) * 0.3
    values += ((idx.remainder(7) - 3.0) * 0.013)
    return values.to(dtype)


def make_rand(n, seed, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(n, generator=generator, dtype=torch.float32).to(dtype).cuda()


def tolerances(dtype, strategy):
    if dtype == torch.float16:
        return (2e-3, 2e-3) if strategy in (0, 2) else (2e-2, 6e-2)
    return (4e-2, 4e-2) if strategy in (0, 2) else (8e-2, 8e-2)


def rounded_sum(left, right, dtype):
    return (left.float() + right.float()).to(dtype).float()


def assert_close(name, actual, expected, strategy, dtype):
    expected_t = torch.as_tensor(expected, dtype=torch.float32, device=actual.device)
    atol, rtol = tolerances(dtype, strategy)
    diff = (actual.float() - expected_t.float()).abs()
    allowed = atol + rtol * expected_t.float().abs()
    bad = (~torch.isfinite(actual.float())) | (~torch.isfinite(expected_t.float())) | (diff > allowed)
    if bool(bad.any().item()):
        flat_idx = int(diff.argmax().item())
        if not bool(torch.isfinite(diff.flatten()[flat_idx]).item()):
            flat_idx = int(bad.flatten().nonzero()[0].item())
        print(
            f"{name}: max_abs_diff={float(diff.flatten()[flat_idx])} "
            f"actual={float(actual.flatten()[flat_idx])} "
            f"expected={float(expected_t.flatten()[flat_idx])} "
            f"index={flat_idx}",
            flush=True,
        )
    torch.testing.assert_close(
        actual.float(),
        expected_t.float(),
        atol=atol,
        rtol=rtol,
        msg=f"{name} failed",
    )


def run_roundtrip(nbits, chunk_size, strategy, dtype):
    n = chunk_size * (32 if strategy >= 2 else 19)
    src = make_input(n, dtype)
    rand_pool = make_rand(n, 1000 + nbits * 31 + chunk_size * 7 + strategy, dtype)
    encoded = torch.zeros(required_bytes(n, nbits, chunk_size, strategy), dtype=torch.uint8, device="cuda")
    decoded = torch.empty_like(src)

    stream = torch.cuda.current_stream().cuda_stream
    eden_utils.scaling_compress(src, encoded, rand_pool, n, nbits, chunk_size, stream, strategy)
    eden_utils.scaling_decompress(encoded, decoded, n, nbits, chunk_size, stream, strategy)
    torch.cuda.synchronize()

    expected = reference_decompressed_for_dtype(src, rand_pool, nbits, chunk_size, strategy, dtype)
    assert_close(f"roundtrip dtype={dtype} nbits={nbits} chunk={chunk_size} strategy={strategy}", decoded, expected, strategy, dtype)

    base = (make_input(n, dtype).float() * 0.125).to(dtype).contiguous()
    added = base.clone()
    eden_utils.scaling_decompress_add(encoded, added, n, nbits, chunk_size, stream, strategy)
    torch.cuda.synchronize()
    atol, rtol = tolerances(dtype, strategy)
    torch.testing.assert_close(
        added.float(),
        rounded_sum(base, decoded, dtype),
        atol=atol,
        rtol=rtol,
        msg=f"decompress_add dtype={dtype} nbits={nbits} chunk={chunk_size} strategy={strategy} failed",
    )


def run_dec_comp(nbits, chunk_size, strategy, dtype):

    n = chunk_size * (32 if strategy >= 2 else 19)
    recv_src = (make_input(n, dtype).float() * 0.5).to(dtype)
    inp = (make_input(n, dtype).float() * -0.25).to(dtype).contiguous()
    inp_before = inp.clone()
    rand_recv = make_rand(n, 2000 + nbits * 17 + chunk_size + strategy, dtype)
    rand_send = make_rand(n, 3000 + nbits * 19 + chunk_size + strategy, dtype)
    recv = torch.zeros(required_bytes(n, nbits, chunk_size, strategy), dtype=torch.uint8, device="cuda")
    send = torch.zeros_like(recv)
    send_split = torch.zeros_like(recv)

    stream = torch.cuda.current_stream().cuda_stream
    eden_utils.scaling_compress(recv_src, recv, rand_recv, n, nbits, chunk_size, stream, strategy)
    eden_utils.scaling_dec_comp(recv, inp, send, rand_send, n, nbits, chunk_size, stream, strategy)
    inp_split = inp_before.clone()
    eden_utils.scaling_decompress_add(recv, inp_split, n, nbits, chunk_size, stream, strategy)
    eden_utils.scaling_compress(inp_split, send_split, rand_send, n, nbits, chunk_size, stream, strategy)
    torch.cuda.synchronize()

    if not bool(torch.equal(send, send_split)):
        mismatch = int((send != send_split).flatten().nonzero()[0].item())
        raise AssertionError(
            f"dec_comp packed output dtype={dtype} nbits={nbits} chunk={chunk_size} "
            f"strategy={strategy} failed at byte {mismatch}: "
            f"actual={int(send.flatten()[mismatch].item())} expected={int(send_split.flatten()[mismatch].item())}"
        )

    send_decoded = torch.empty_like(inp)
    eden_utils.scaling_decompress(send, send_decoded, n, nbits, chunk_size, stream, strategy)
    torch.cuda.synchronize()
    expected = reference_decompressed_for_dtype(inp_split, rand_send, nbits, chunk_size, strategy, dtype)
    assert_close(f"dec_comp send dtype={dtype} nbits={nbits} chunk={chunk_size} strategy={strategy}", send_decoded, expected, strategy, dtype)


def main():
    assert torch.cuda.is_available(), "CUDA is required for this test"
    torch.manual_seed(1234)
    for dtype in (torch.float16, torch.bfloat16):
        for chunk_size in (16, 32):
            for nbits in (2, 4, 8):
                for strategy in (0, 1, 2, 3):
                    run_roundtrip(nbits, chunk_size, strategy, dtype)
                    run_dec_comp(nbits, chunk_size, strategy, dtype)
                    print(f"ok dtype={dtype} nbits={nbits} chunk_size={chunk_size} strategy={strategy}", flush=True)


if __name__ == "__main__":
    main()
