# Scaling Quantization Algorithm

This document describes the `scaling_*` quantization algorithms implemented by
the CUDA extension. The public operations are:

- `scaling_compress`: compress a `torch.float16` or `torch.bfloat16` gradient
  vector into a packed byte buffer.
- `scaling_decompress`: decompress a packed buffer into a destination buffer
  with the same floating dtype family, overwriting the destination.
- `scaling_decompress_add`: decompress a packed buffer and add the result into
  the destination buffer.
- `scaling_dec_comp`: decompress a received packed buffer, add it to an
  existing `torch.float16` or `torch.bfloat16` input value, then compress the
  sum into a send buffer. The operation does not require the sum to be written
  back to the input buffer.

The CUDA implementation exposes these operations through two lower-level
launchers:

```cpp
void scaling_compress_with_cuda(
    __half* src,
    uint8_t* dst,
    __half* rand_pool,
    int n,
    int nbits,
    int chunk_size,
    int device,
    int strategy,
    cudaStream_t stream);

void scaling_decompress_with_cuda(
    uint8_t* src,
    __half* dst,
    int n,
    int nbits,
    int chunk_size,
    int device,
    int strategy,
    cudaStream_t stream,
    int add_original);

void scaling_compress_with_cuda(
    __nv_bfloat16* src,
    uint8_t* dst,
    __nv_bfloat16* rand_pool,
    int n,
    int nbits,
    int chunk_size,
    int device,
    int strategy,
    cudaStream_t stream);

void scaling_decompress_with_cuda(
    uint8_t* src,
    __nv_bfloat16* dst,
    int n,
    int nbits,
    int chunk_size,
    int device,
    int strategy,
    cudaStream_t stream,
    int add_original);
```

`add_original == 0` means decompression overwrites `dst`; `add_original != 0`
means decompression adds into the existing `dst` values.

## Inputs and Constraints

The source gradient is a one-dimensional `torch.float16` or `torch.bfloat16`
vector. The C++ wrapper dispatches to the matching CUDA overload, and
`rand_pool` must use the same dtype as the source/input tensor. The
implementation assumes:

- `nbits` is one of `2`, `4`, or `8`.
- `chunk_size <= 32`.
- `chunk_size` is a power of two.
- `n` is a multiple of `chunk_size`.
- Hierarchical mode additionally assumes `n` is a multiple of
  `16 * chunk_size`.

`rand_pool` is a dtype-matched vector of random values in `[0, 1]`, indexed by
the same coordinate id as `src`. It supplies the stochastic rounding decisions.
For `bfloat16`, the kernels convert source and random values to `float` for
arithmetic, store the scale fields as `bfloat16`, and round decompressed outputs
back to `bfloat16`. The MEE representative table is still stored as `__half`,
then converted to `float` inside the `bfloat16` decompression kernels.

The `strategy` argument selects the quantizer:

- `strategy == 0`: AEE, linear quantization.
- `strategy == 1`: MEE, nonlinear quantization.
- `strategy == 2`: hierarchical AEE, where the per-chunk AEE scaling factors
  are themselves quantized inside groups of 16 chunks.
- `strategy == 3`: hierarchical MEE, where the per-chunk MEE scaling factors
  are themselves quantized inside groups of 16 chunks.

## Chunking

The vector is partitioned into chunks of `chunk_size` coordinates. A chunk id is

```text
chunk_id = element_id / chunk_size
```

and the coordinate inside a chunk is

```text
inner_id = element_id % chunk_size
```

Each chunk is quantized independently. The chunk scale is derived from the
absolute maximum value in that chunk:

```text
absmax = max(abs(src[i])) for i in the chunk
```

## AEE Linear Quantization

For `nbits`, the signed integer range is symmetric around zero:

```text
range = (1 << (nbits - 1)) - 1
```

For example:

- `nbits == 2`: representable signed values are `[-1, 1]`.
- `nbits == 4`: representable signed values are `[-7, 7]`.
- `nbits == 8`: representable signed values are `[-127, 127]`.

The chunk scale is:

```text
scale = absmax / range + eps
```

where `eps` is a small positive constant used to avoid division by zero.

Each value is normalized:

```text
x = src[i] / scale
```

Then stochastic rounding maps `x` to one of its two nearest integers. Let:

```text
lo = floor(x)
frac = x - lo
r = rand_pool[i]
```

The quantized signed integer is:

```text
q_signed = lo + 1 if r <= frac else lo
```

Equivalently, the CUDA implementation can compute:

```text
q_signed = floor(x) + ceil(frac - r)
```

The signed integer is offset into an unsigned code:

```text
q_code = q_signed + range
```

This uses `2 * range + 1` valid codes. Since the physical storage uses `nbits`
bits, one unsigned code is unused for each supported bit width.

The `q_code` values are packed into bytes:

- INT8: one value per byte.
- INT4: two values per byte.
- INT2: four values per byte.

The non-hierarchical AEE chunk layout is:

```text
| packed quantized values | dtype scale |
```

The chunk byte size is:

```text
chunk_bytes = (chunk_size * nbits) / 8 + 2
```

Both supported floating dtypes occupy two bytes for the stored scale, so the
packed size is the same for `float16` and `bfloat16`.

Decompression unpacks `q_code`, converts it back to a signed integer,
multiplies by the stored scale, and either writes or adds the result:

```text
q_signed = q_code - range
dequantized = q_signed * scale
```

## MEE Nonlinear Quantization

MEE uses the same per-chunk absmax idea, but the integer lattice is mapped to a
nonlinear set of floating-point representatives.

For a given `epsilon = e`, define:

```text
mee_lattice(l) = ((1 + 2e^2)^l - 1) / (2e^2) * (1 + e^2)
```

where `l` is an integer. Negative `l` values map symmetrically:

```text
mee_lattice(-l) = -mee_lattice(l)
```

The largest positive representative for `nbits` is:

```text
max_l = (1 << (nbits - 1)) - 1
chunk_max_mee = mee_lattice(max_l)
```

The chunk scale is:

```text
scale = absmax / chunk_max_mee + eps
```

Each source value is normalized:

```text
t = src[i] / scale
```

Before stochastic rounding, MEE applies the inverse lattice map. For
`abs(t) > 0`:

```text
l_float = log(abs(t) * (2e^2) / (1 + e^2) + 1) / log(1 + 2e^2)
```

and the original sign is restored:

```text
l_float = sign(t) * l_float
```

For `t == 0`, `l_float == 0`.

The floating-point lattice coordinate `l_float` is then stochastically rounded
exactly like AEE:

```text
lo = floor(l_float)
frac = l_float - lo
q_l = lo + 1 if rand_pool[i] <= frac else lo
q_code = q_l + max_l
```

Decompression uses the MEE lookup table:

```text
q_l = q_code - max_l
representative = mee_lattice(q_l)
dequantized = representative * scale
```

The non-hierarchical MEE memory layout is the same as AEE:

```text
| packed quantized values | dtype scale |
```

## Hierarchical Scale Quantization

Hierarchical mode groups exactly 16 chunks into one superchunk. The packed
coordinate values are still AEE or MEE codes depending on the selected
strategy, but each chunk scale is quantized to one byte instead of being stored
directly as the floating dtype. One dtype-sized superchunk scale is shared by
all 16 chunks.

For hierarchical AEE (`strategy == 2`), compute the ordinary AEE chunk scale:

```text
chunk_scale[j] = absmax(chunk j) / range + eps
```

For hierarchical MEE (`strategy == 3`), compute the ordinary MEE chunk scale:

```text
chunk_scale[j] = absmax(chunk j) / chunk_max_mee + eps
```

The superchunk maximum is:

```text
super_scale = max(chunk_scale[0], ..., chunk_scale[15])
```

Because all chunk scales are nonnegative, each one is quantized into an unsigned
8-bit value in `[0, 255]`. The algorithm uses round-up quantization:

```text
q_scale[j] = ceil(chunk_scale[j] / super_scale * 255)
```

with clamping to `[0, 255]`. The stored floating value is the scale unit:

```text
super_scale_unit = super_scale / 255
```

During decompression:

```text
chunk_scale[j] = q_scale[j] * super_scale_unit
```

During compression, the packed coordinate values should be computed against this
reconstructed chunk scale rather than the unquantized chunk scale. Because
`q_scale[j]` is rounded up, the reconstructed scale is at least as large as the
original scale, so every normalized value remains inside the representable
range.

During AEE decompression:

```text
q_signed = q_code - range
dequantized = q_signed * chunk_scale[j]
```

During MEE decompression, the reconstructed `chunk_scale[j]` is used exactly
like the ordinary MEE scale:

```text
representative = mee_lattice(q_l)
dequantized = representative * chunk_scale[j]
```

The hierarchical superchunk layout is:

```text
| chunk 0 packed values | chunk 0 uint8 scale |
| chunk 1 packed values | chunk 1 uint8 scale |
...
| chunk 15 packed values | chunk 15 uint8 scale |
| dtype super_scale_unit |
```

Let:

```text
packed_bytes = (chunk_size * nbits) / 8
per_chunk_bytes = packed_bytes + 1
superchunk_size = 16
superchunk_bytes = superchunk_size * per_chunk_bytes + 2
```

Then for a `chunk_id`:

```text
super_id = chunk_id / superchunk_size
local_chunk_id = chunk_id % superchunk_size
super_start = super_id * superchunk_bytes
chunk_start = super_start + local_chunk_id * per_chunk_bytes
chunk_scale_addr = chunk_start + packed_bytes
super_scale_addr = super_start + superchunk_size * per_chunk_bytes
```

## Packed Value Ordering

Within a chunk, values are stored in increasing coordinate order. For an element
with `inner_id = element_id % chunk_size`:

```text
byte_offset = (inner_id * nbits) / 8
bit_offset = (inner_id * nbits) % 8
```

The `q_code` occupies `nbits` bits starting at `bit_offset` in the corresponding
byte.

## CUDA Runtime Notes

The current implementation relies on the guarantee that `chunk_size <= 32`.
Each chunk fits inside one warp, so chunk `absmax` reductions use
`__shfl_xor_sync` instead of block-wide shared-memory reductions. Packed
coordinates are also assembled within the warp with `__shfl_down_sync`, leaving
one lane to write each output byte.

The MEE lookup table is initialized lazily once per CUDA device. Nonlinear
decompression stages only the active `(1 << nbits)` table entries into dynamic
shared memory. Hierarchical decompression uses the same precomputed device table
instead of recomputing the MEE representative with `pow`/`log` per element.

Hierarchical compression launches one block per 16-chunk superchunk with
`16 * chunk_size` threads. The block stores only the 16 raw chunk scales plus one
superchunk maximum in shared memory, then quantizes chunk scales to `[0, 255]`
and compresses coordinates against the reconstructed per-chunk scale.

For the production bf16 hierarchical paths with `chunk_size == 16`, the launcher
uses specialized `nbits in {2, 4, 8}` kernels. These kernels hard-code the
16-coordinate chunk and 16-chunk superchunk layout, avoid per-element divisions
and modulo operations, compute the local scale once per chunk, and use packed
byte decoding/encoding directly. The MEE decompression specialization stages the
MEE table as `float` shared memory; AEE decompression decodes signed linear
values directly. Both write bf16 pairs with 32-bit stores.

The same bf16 hierarchical `chunk_size == 16` paths also have a fused
`scaling_dec_comp` kernel. It decompresses the received superchunk, adds it to
the bf16 input, rounds the sum to bf16 in registers, and compresses that rounded
value into the send buffer without storing and reloading an intermediate tensor.
Other dtype, strategy, and chunk-size combinations continue to use the generic
two-kernel decompress-add then compress sequence.
