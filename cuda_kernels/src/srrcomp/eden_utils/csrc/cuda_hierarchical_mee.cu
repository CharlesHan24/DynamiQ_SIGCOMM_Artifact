#include "cuda_hierarchical_mee.h"
#include <unistd.h>
#include <cuda_fp16.h>

#include <cuda_bf16.h>

__device__ __forceinline__ void store_bf16_pair(
    __nv_bfloat16* __restrict__ dst,
    unsigned int out_idx,
    float v0,
    float v1
) {
    // out_idx must be even for 4-byte alignment.
    __nv_bfloat162 packed = __floats2bfloat162_rn(v0, v1);
    *reinterpret_cast<__nv_bfloat162*>(dst + out_idx) = packed;
}

constexpr float SCALING_HIER_MIN_POSITIVE_SCALE = 1.0e-30f;

__device__ __forceinline__ bool scaling_hier_is_finite(float value) {
    return isfinite(value);
}

__device__ __forceinline__ float scaling_hier_finite_or(float value, float fallback) {
    return scaling_hier_is_finite(value) ? value : fallback;
}

__device__ __forceinline__ float scaling_hier_positive_scale(float value) {
    value = scaling_hier_finite_or(value, SCALING_HIER_MIN_POSITIVE_SCALE);
    return fmaxf(value, SCALING_HIER_MIN_POSITIVE_SCALE);
}

__device__ __forceinline__ float scaling_hier_nonnegative_scale(float value) {
    value = scaling_hier_finite_or(value, 0.0f);
    return fmaxf(value, 0.0f);
}

__device__ __forceinline__ float scaling_hier_bf16_positive_scale(float value) {
    __nv_bfloat16 stored = __float2bfloat16(scaling_hier_positive_scale(value));
    return scaling_hier_positive_scale(__bfloat162float(stored));
}

__device__ __forceinline__ __nv_bfloat16 scaling_hier_bf16_scale_to_store(float value) {
    return __float2bfloat16(scaling_hier_positive_scale(value));
}

__device__ __forceinline__ uint8_t scaling_hier_quantized_chunk_scale(float chunk_scale, float global_chunk_scale) {
    float ratio = scaling_hier_positive_scale(chunk_scale) / scaling_hier_positive_scale(global_chunk_scale) * 255.0f;
    if (!scaling_hier_is_finite(ratio)) {
        ratio = ratio > 0.0f ? 255.0f : 1.0f;
    }
    uint16_t quantized = (uint16_t)ceilf(ratio);
    if (quantized < 1) {
        quantized = 1;
    }
    if (quantized > 255) {
        quantized = 255;
    }
    return (uint8_t)quantized;
}

__device__ __forceinline__ unsigned int scaling_hier_chunk_mask(int chunk_size) {
    unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
    if (chunk_size == WARP_SIZE) {
        return 0xffffffffu;
    }

    unsigned int group_start = lane & ~(chunk_size - 1);
    return ((1u << chunk_size) - 1u) << group_start;
}

__device__ __forceinline__ __half scaling_hier_chunk_absmax(__half local_abs, int chunk_size) {
    unsigned int mask = scaling_hier_chunk_mask(chunk_size);
    for (int stride = 1; stride < chunk_size; stride <<= 1) {
        local_abs = __hmax(local_abs, __shfl_xor_sync(mask, local_abs, stride));
    }
    return local_abs;
}

__device__ __forceinline__ int16_t scaling_hier_stochastic_round_to_signed(__half val, __half rand_val) {
    __half floored_val = hfloor(val);
    int16_t signed_val = __half2short_rd(floored_val);
    __half frac = __hsub(val, floored_val);
    signed_val += __half2short_ru(__hsub(frac, rand_val));
    return signed_val;
}

__device__ __forceinline__ uint8_t scaling_hier_signed_to_code(int16_t signed_val, int range) {
    if (signed_val < -range) {
        signed_val = -range;
    }
    if (signed_val > range) {
        signed_val = range;
    }
    return (uint8_t)(signed_val + range);
}

__device__ __forceinline__ __half scaling_hier_mee_inverse_lattice(__half normalized_val, __half one_plus_eps2_over_2eps2, __half log2_one_plus_2eps2) {
    __half abs_val = __habs(normalized_val);
    if (!__hgt(abs_val, __float2half(0.0f))) {
        return __float2half(0.0f);
    }

    __half log_arg = __hadd(__hdiv(abs_val, one_plus_eps2_over_2eps2), __float2half(1.0f));
    __half lattice_val = __hdiv(hlog2(log_arg), log2_one_plus_2eps2);
    return __hlt(normalized_val, __float2half(0.0f)) ? __hneg(lattice_val) : lattice_val;
}

__device__ __forceinline__ float scaling_hier_chunk_absmax_float(float local_abs, int chunk_size) {
    unsigned int mask = scaling_hier_chunk_mask(chunk_size);
    for (int stride = 1; stride < chunk_size; stride <<= 1) {
        local_abs = fmaxf(local_abs, __shfl_xor_sync(mask, local_abs, stride));
    }
    return local_abs;
}

__device__ __forceinline__ int16_t scaling_hier_stochastic_round_to_signed_float(float val, float rand_val) {
    if (!scaling_hier_is_finite(val)) {
        if (val > 0.0f) {
            return 32767;
        }
        if (val < 0.0f) {
            return -32767;
        }
        return 0;
    }
    if (val >= 32767.0f) {
        return 32767;
    }
    if (val <= -32767.0f) {
        return -32767;
    }
    rand_val = scaling_hier_finite_or(rand_val, 0.0f);
    float floored_val = floorf(val);
    int16_t signed_val = (int16_t)floored_val;
    signed_val += (int16_t)ceilf((val - floored_val) - rand_val);
    return signed_val;
}

__device__ __forceinline__ float scaling_hier_mee_inverse_lattice_float(float normalized_val, float one_plus_eps2_over_2eps2, float log2_one_plus_2eps2) {
    if (!scaling_hier_is_finite(normalized_val)) {
        if (normalized_val > 0.0f) {
            return 32767.0f;
        }
        if (normalized_val < 0.0f) {
            return -32767.0f;
        }
        return 0.0f;
    }
    float abs_val = fabsf(normalized_val);
    if (!(abs_val > 0.0f)) {
        return 0.0f;
    }

    float lattice_val = log2f(abs_val / one_plus_eps2_over_2eps2 + 1.0f) / log2_one_plus_2eps2;
    return normalized_val < 0.0f ? -lattice_val : lattice_val;
}

__device__ __forceinline__ float scaling_hier_chunk_absmax_16(float local_abs) {
    unsigned int mask = ((threadIdx.x & 16) == 0) ? 0x0000ffffu : 0xffff0000u;
    local_abs = fmaxf(local_abs, __shfl_xor_sync(mask, local_abs, 1, 16));
    local_abs = fmaxf(local_abs, __shfl_xor_sync(mask, local_abs, 2, 16));
    local_abs = fmaxf(local_abs, __shfl_xor_sync(mask, local_abs, 4, 16));
    local_abs = fmaxf(local_abs, __shfl_xor_sync(mask, local_abs, 8, 16));
    return local_abs;
}

template<int NBITS>
__device__ __forceinline__ void scaling_hier_store_packed_code_16(
    uint8_t* __restrict__ dst,
    unsigned int chunk_start_addr,
    unsigned int inner_chunk_idx,
    uint8_t code
) {
    unsigned int mask = ((threadIdx.x & 16) == 0) ? 0x0000ffffu : 0xffff0000u;
    unsigned int code_u = (unsigned int)code;

    if constexpr (NBITS == 2) {
        unsigned int group_base = inner_chunk_idx & ~3u;
        unsigned int code0 = __shfl_sync(mask, code_u, group_base, 16);
        unsigned int code1 = __shfl_sync(mask, code_u, group_base + 1, 16);
        unsigned int code2 = __shfl_sync(mask, code_u, group_base + 2, 16);
        unsigned int code3 = __shfl_sync(mask, code_u, group_base + 3, 16);
        if (inner_chunk_idx == group_base) {
            uint8_t packed = (uint8_t)(code0 | (code1 << 2) | (code2 << 4) | (code3 << 6));
            dst[chunk_start_addr + (inner_chunk_idx >> 2)] = packed;
        }
    }
    else if constexpr (NBITS == 4) {
        unsigned int group_base = inner_chunk_idx & ~1u;
        unsigned int code0 = __shfl_sync(mask, code_u, group_base, 16);
        unsigned int code1 = __shfl_sync(mask, code_u, group_base + 1, 16);
        if (inner_chunk_idx == group_base) {
            uint8_t packed = (uint8_t)(code0 | (code1 << 4));
            dst[chunk_start_addr + (inner_chunk_idx >> 1)] = packed;
        }
    }
    else {
        dst[chunk_start_addr + inner_chunk_idx] = code;
    }
}

template<int NBITS>
__global__ void scaling_compress_kernel_aee_hierarchical_bf16_opt(
    __nv_bfloat16* __restrict__ src,
    uint8_t* __restrict__ dst,
    __nv_bfloat16* __restrict__ rand_pool,
    int n
) {
    (void)n;
    constexpr unsigned int CHUNK_SIZE = 16;
    constexpr unsigned int SUPERCHUNK_SIZE = 16;
    constexpr unsigned int PACKED_BYTES = (CHUNK_SIZE * NBITS) >> 3;
    constexpr unsigned int PER_CHUNK_BYTES = PACKED_BYTES + 1;
    constexpr unsigned int SUPERCHUNK_BYTES = SUPERCHUNK_SIZE * PER_CHUNK_BYTES + 2;
    constexpr int RANGE = (1 << (NBITS - 1)) - 1;

    extern __shared__ float sdata_opt[];
    float* raw_chunk_scales = sdata_opt;
    float* decoded_chunk_scales = sdata_opt + SUPERCHUNK_SIZE;
    float* global_chunk_scale = sdata_opt + SUPERCHUNK_SIZE * 2;
    float* super_scale_unit = global_chunk_scale + 1;

    unsigned int sid = threadIdx.x;
    unsigned int local_chunk_id = sid >> 4;
    unsigned int inner_chunk_idx = sid & (CHUNK_SIZE - 1);
    unsigned int tid = blockIdx.x * (SUPERCHUNK_SIZE * CHUNK_SIZE) + sid;

    float local_val = __bfloat162float(src[tid]);
    float chunk_scale = scaling_hier_chunk_absmax_16(fabsf(local_val));
    chunk_scale = scaling_hier_positive_scale(chunk_scale / (float)RANGE + 1e-7f);

    unsigned int super_start_addr = blockIdx.x * SUPERCHUNK_BYTES;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * PER_CHUNK_BYTES;
    unsigned int chunk_scale_addr = chunk_start_addr + PACKED_BYTES;
    unsigned int super_scale_addr = super_start_addr + SUPERCHUNK_SIZE * PER_CHUNK_BYTES;

    if (inner_chunk_idx == 0) {
        raw_chunk_scales[local_chunk_id] = chunk_scale;
    }
    __syncthreads();

    if (sid == 0) {
        float global_chunk_max = raw_chunk_scales[0];
        #pragma unroll
        for (int i = 1; i < SUPERCHUNK_SIZE; i++) {
            global_chunk_max = fmaxf(global_chunk_max, raw_chunk_scales[i]);
        }
        *global_chunk_scale = scaling_hier_positive_scale(global_chunk_max);
        __nv_bfloat16 stored_super_scale_unit = scaling_hier_bf16_scale_to_store(*global_chunk_scale / 255.0f);
        *super_scale_unit = scaling_hier_bf16_positive_scale(*global_chunk_scale / 255.0f);
        *reinterpret_cast<__nv_bfloat16*>(dst + super_scale_addr) = stored_super_scale_unit;
    }
    __syncthreads();

    if (inner_chunk_idx == 0) {
        uint8_t quantized_chunk_max = scaling_hier_quantized_chunk_scale(raw_chunk_scales[local_chunk_id], *global_chunk_scale);
        dst[chunk_scale_addr] = (uint8_t)quantized_chunk_max;
        decoded_chunk_scales[local_chunk_id] = scaling_hier_positive_scale((float)quantized_chunk_max * (*super_scale_unit));
    }
    __syncthreads();

    local_val /= scaling_hier_positive_scale(decoded_chunk_scales[local_chunk_id]);
    uint8_t code = scaling_hier_signed_to_code(
        scaling_hier_stochastic_round_to_signed_float(local_val, __bfloat162float(rand_pool[tid])),
        RANGE
    );

    scaling_hier_store_packed_code_16<NBITS>(dst, chunk_start_addr, inner_chunk_idx, code);
}

template<int NBITS>
__global__ void scaling_compress_kernel_mee_hierarchical_bf16_opt(
    __nv_bfloat16* __restrict__ src,
    uint8_t* __restrict__ dst,
    __nv_bfloat16* __restrict__ rand_pool,
    int n,
    float one_plus_eps2_over_2eps2,
    float chunk_max_mee,
    float log2_one_plus_2eps2
) {
    (void)n;
    constexpr unsigned int CHUNK_SIZE = 16;
    constexpr unsigned int SUPERCHUNK_SIZE = 16;
    constexpr unsigned int PACKED_BYTES = (CHUNK_SIZE * NBITS) >> 3;
    constexpr unsigned int PER_CHUNK_BYTES = PACKED_BYTES + 1;
    constexpr unsigned int SUPERCHUNK_BYTES = SUPERCHUNK_SIZE * PER_CHUNK_BYTES + 2;
    constexpr int RANGE = (1 << (NBITS - 1)) - 1;

    extern __shared__ float sdata_opt[];
    float* raw_chunk_scales = sdata_opt;
    float* decoded_chunk_scales = sdata_opt + SUPERCHUNK_SIZE;
    float* global_chunk_scale = sdata_opt + SUPERCHUNK_SIZE * 2;
    float* super_scale_unit = global_chunk_scale + 1;

    unsigned int sid = threadIdx.x;
    unsigned int local_chunk_id = sid >> 4;
    unsigned int inner_chunk_idx = sid & (CHUNK_SIZE - 1);
    unsigned int tid = blockIdx.x * (SUPERCHUNK_SIZE * CHUNK_SIZE) + sid;

    float local_val = __bfloat162float(src[tid]);
    float chunk_scale = scaling_hier_chunk_absmax_16(fabsf(local_val));
    chunk_scale = scaling_hier_positive_scale(chunk_scale / scaling_hier_positive_scale(chunk_max_mee) + 1e-7f);

    unsigned int super_start_addr = blockIdx.x * SUPERCHUNK_BYTES;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * PER_CHUNK_BYTES;
    unsigned int chunk_scale_addr = chunk_start_addr + PACKED_BYTES;
    unsigned int super_scale_addr = super_start_addr + SUPERCHUNK_SIZE * PER_CHUNK_BYTES;

    if (inner_chunk_idx == 0) {
        raw_chunk_scales[local_chunk_id] = chunk_scale;
    }
    __syncthreads();

    if (sid == 0) {
        float global_chunk_max = raw_chunk_scales[0];
        #pragma unroll
        for (int i = 1; i < SUPERCHUNK_SIZE; i++) {
            global_chunk_max = fmaxf(global_chunk_max, raw_chunk_scales[i]);
        }
        *global_chunk_scale = scaling_hier_positive_scale(global_chunk_max);
        __nv_bfloat16 stored_super_scale_unit = scaling_hier_bf16_scale_to_store(*global_chunk_scale / 255.0f);
        *super_scale_unit = scaling_hier_bf16_positive_scale(*global_chunk_scale / 255.0f);
        *reinterpret_cast<__nv_bfloat16*>(dst + super_scale_addr) = stored_super_scale_unit;
    }
    __syncthreads();

    if (inner_chunk_idx == 0) {
        uint8_t quantized_chunk_max = scaling_hier_quantized_chunk_scale(raw_chunk_scales[local_chunk_id], *global_chunk_scale);
        dst[chunk_scale_addr] = (uint8_t)quantized_chunk_max;
        decoded_chunk_scales[local_chunk_id] = scaling_hier_positive_scale((float)quantized_chunk_max * (*super_scale_unit));
    }
    __syncthreads();

    local_val /= scaling_hier_positive_scale(decoded_chunk_scales[local_chunk_id]);
    float rev_local_val = scaling_hier_mee_inverse_lattice_float(local_val, one_plus_eps2_over_2eps2, log2_one_plus_2eps2);
    uint8_t code = scaling_hier_signed_to_code(
        scaling_hier_stochastic_round_to_signed_float(rev_local_val, __bfloat162float(rand_pool[tid])),
        RANGE
    );

    scaling_hier_store_packed_code_16<NBITS>(dst, chunk_start_addr, inner_chunk_idx, code);
}

template<int NBITS, bool ADD>
__global__ void scaling_decompress_kernel_aee_hierarchical_bf16_opt(
    const uint8_t* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n
) {
    (void)n;
    constexpr unsigned int CHUNK_SIZE = 16;
    constexpr unsigned int SUPERCHUNK_SIZE = 16;
    constexpr unsigned int PACKED_BYTES = (CHUNK_SIZE * NBITS) >> 3;
    constexpr unsigned int PER_CHUNK_BYTES = PACKED_BYTES + 1;
    constexpr unsigned int SUPERCHUNK_BYTES = SUPERCHUNK_SIZE * PER_CHUNK_BYTES + 2;
    constexpr unsigned int WORKERS_PER_CHUNK = (NBITS == 2) ? 4 : 8;
    constexpr int RANGE = (1 << (NBITS - 1)) - 1;

    extern __shared__ float sdata_opt[];
    float* s_chunk_scales = sdata_opt;
    float* s_super_scale = s_chunk_scales + SUPERCHUNK_SIZE;

    unsigned int super_start_addr = blockIdx.x * SUPERCHUNK_BYTES;
    unsigned int super_scale_addr = super_start_addr + SUPERCHUNK_SIZE * PER_CHUNK_BYTES;
    if (threadIdx.x == 0) {
        *s_super_scale = scaling_hier_nonnegative_scale(__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(src + super_scale_addr)));
    }
    __syncthreads();

    unsigned int local_chunk_id = threadIdx.x / WORKERS_PER_CHUNK;
    unsigned int worker_lane = threadIdx.x - local_chunk_id * WORKERS_PER_CHUNK;
    unsigned int chunk_id = blockIdx.x * SUPERCHUNK_SIZE + local_chunk_id;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * PER_CHUNK_BYTES;

    if (worker_lane == 0) {
        s_chunk_scales[local_chunk_id] = scaling_hier_nonnegative_scale((float)src[chunk_start_addr + PACKED_BYTES] * (*s_super_scale));
    }
    __syncthreads();

    float chunk_scale = s_chunk_scales[local_chunk_id];

    if constexpr (NBITS == 2) {
        uint8_t packed = src[chunk_start_addr + worker_lane];
        unsigned int out_idx = chunk_id * CHUNK_SIZE + worker_lane * 4;
        float v0 = (float)((int)(packed & 0x03) - RANGE) * chunk_scale;
        float v1 = (float)((int)((packed >> 2) & 0x03) - RANGE) * chunk_scale;
        float v2 = (float)((int)((packed >> 4) & 0x03) - RANGE) * chunk_scale;
        float v3 = (float)((int)((packed >> 6) & 0x03) - RANGE) * chunk_scale;
        if constexpr (ADD) {
            v0 += __bfloat162float(dst[out_idx]);
            v1 += __bfloat162float(dst[out_idx + 1]);
            v2 += __bfloat162float(dst[out_idx + 2]);
            v3 += __bfloat162float(dst[out_idx + 3]);
        }
        store_bf16_pair(dst, out_idx, v0, v1);
        store_bf16_pair(dst, out_idx + 2, v2, v3);
    }
    else if constexpr (NBITS == 4) {
        uint8_t packed = src[chunk_start_addr + worker_lane];
        unsigned int out_idx = chunk_id * CHUNK_SIZE + worker_lane * 2;
        float v0 = (float)((int)(packed & 0x0f) - RANGE) * chunk_scale;
        float v1 = (float)((int)(packed >> 4) - RANGE) * chunk_scale;
        if constexpr (ADD) {
            v0 += __bfloat162float(dst[out_idx]);
            v1 += __bfloat162float(dst[out_idx + 1]);
        }
        store_bf16_pair(dst, out_idx, v0, v1);
    }
    else {
        unsigned int packed_idx = worker_lane * 2;
        unsigned int out_idx = chunk_id * CHUNK_SIZE + packed_idx;
        float v0 = (float)((int)src[chunk_start_addr + packed_idx] - RANGE) * chunk_scale;
        float v1 = (float)((int)src[chunk_start_addr + packed_idx + 1] - RANGE) * chunk_scale;
        if constexpr (ADD) {
            v0 += __bfloat162float(dst[out_idx]);
            v1 += __bfloat162float(dst[out_idx + 1]);
        }
        store_bf16_pair(dst, out_idx, v0, v1);
    }
}

template<int NBITS, bool ADD>
__global__ void scaling_decompress_kernel_mee_hierarchical_bf16_opt(
    const uint8_t* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n,
    const __half* __restrict__ mee_table
) {
    (void)n;
    constexpr unsigned int CHUNK_SIZE = 16;
    constexpr unsigned int SUPERCHUNK_SIZE = 16;
    constexpr unsigned int PACKED_BYTES = (CHUNK_SIZE * NBITS) >> 3;
    constexpr unsigned int PER_CHUNK_BYTES = PACKED_BYTES + 1;
    constexpr unsigned int SUPERCHUNK_BYTES = SUPERCHUNK_SIZE * PER_CHUNK_BYTES + 2;
    constexpr unsigned int TABLE_SIZE = 1u << NBITS;
    constexpr unsigned int WORKERS_PER_CHUNK = (NBITS == 2) ? 4 : 8;

    extern __shared__ float sdata_opt[];
    float* s_table = sdata_opt;
    float* s_chunk_scales = sdata_opt + TABLE_SIZE;
    float* s_super_scale = s_chunk_scales + SUPERCHUNK_SIZE;

    for (unsigned int table_idx = threadIdx.x; table_idx < TABLE_SIZE; table_idx += blockDim.x) {
        s_table[table_idx] = __half2float(mee_table[NBITS * 256 + table_idx]);
    }

    unsigned int super_start_addr = blockIdx.x * SUPERCHUNK_BYTES;
    unsigned int super_scale_addr = super_start_addr + SUPERCHUNK_SIZE * PER_CHUNK_BYTES;
    if (threadIdx.x == 0) {
        *s_super_scale = scaling_hier_nonnegative_scale(__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(src + super_scale_addr)));
    }
    __syncthreads();

    unsigned int local_chunk_id = threadIdx.x / WORKERS_PER_CHUNK;
    unsigned int worker_lane = threadIdx.x - local_chunk_id * WORKERS_PER_CHUNK;
    unsigned int chunk_id = blockIdx.x * SUPERCHUNK_SIZE + local_chunk_id;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * PER_CHUNK_BYTES;

    if (worker_lane == 0) {
        s_chunk_scales[local_chunk_id] = scaling_hier_nonnegative_scale((float)src[chunk_start_addr + PACKED_BYTES] * (*s_super_scale));
    }
    __syncthreads();

    float chunk_scale = s_chunk_scales[local_chunk_id];

    if constexpr (NBITS == 2) {
        uint8_t packed = src[chunk_start_addr + worker_lane];
        unsigned int out_idx = chunk_id * CHUNK_SIZE + worker_lane * 4;
        float v0 = s_table[packed & 0x03] * chunk_scale;
        float v1 = s_table[(packed >> 2) & 0x03] * chunk_scale;
        float v2 = s_table[(packed >> 4) & 0x03] * chunk_scale;
        float v3 = s_table[(packed >> 6) & 0x03] * chunk_scale;
        if constexpr (ADD) {
            v0 += __bfloat162float(dst[out_idx]);
            v1 += __bfloat162float(dst[out_idx + 1]);
            v2 += __bfloat162float(dst[out_idx + 2]);
            v3 += __bfloat162float(dst[out_idx + 3]);
        }
        store_bf16_pair(dst, out_idx, v0, v1);
        store_bf16_pair(dst, out_idx + 2, v2, v3);
    }
    else if constexpr (NBITS == 4) {
        uint8_t packed = src[chunk_start_addr + worker_lane];
        unsigned int out_idx = chunk_id * CHUNK_SIZE + worker_lane * 2;
        float v0 = s_table[packed & 0x0f] * chunk_scale;
        float v1 = s_table[packed >> 4] * chunk_scale;
        if constexpr (ADD) {
            v0 += __bfloat162float(dst[out_idx]);
            v1 += __bfloat162float(dst[out_idx + 1]);
        }
        store_bf16_pair(dst, out_idx, v0, v1);
    }
    else {
        unsigned int packed_idx = worker_lane * 2;
        unsigned int out_idx = chunk_id * CHUNK_SIZE + packed_idx;
        float v0 = s_table[src[chunk_start_addr + packed_idx]] * chunk_scale;
        float v1 = s_table[src[chunk_start_addr + packed_idx + 1]] * chunk_scale;
        if constexpr (ADD) {
            v0 += __bfloat162float(dst[out_idx]);
            v1 += __bfloat162float(dst[out_idx + 1]);
        }
        store_bf16_pair(dst, out_idx, v0, v1);
    }
}

template<int NBITS>
__device__ __forceinline__ uint8_t scaling_hier_load_packed_code_16(
    const uint8_t* __restrict__ src,
    unsigned int chunk_start_addr,
    unsigned int inner_chunk_idx
) {
    constexpr unsigned int VALUES_PER_BYTE = 8 / NBITS;
    constexpr unsigned int CODE_MASK = (1u << NBITS) - 1u;

    unsigned int mask = ((threadIdx.x & 16) == 0) ? 0x0000ffffu : 0xffff0000u;
    unsigned int group_base = inner_chunk_idx & ~(VALUES_PER_BYTE - 1u);
    unsigned int packed = 0;
    if (inner_chunk_idx == group_base) {
        packed = src[chunk_start_addr + ((group_base * NBITS) >> 3)];
    }
    packed = __shfl_sync(mask, packed, group_base, 16);
    return (uint8_t)((packed >> ((inner_chunk_idx - group_base) * NBITS)) & CODE_MASK);
}

template<int NBITS>
__global__ void scaling_dec_comp_kernel_aee_hierarchical_bf16_opt(
    const uint8_t* __restrict__ recv,
    const __nv_bfloat16* __restrict__ inp,
    uint8_t* __restrict__ send,
    __nv_bfloat16* __restrict__ rand_pool,
    int n
) {
    (void)n;
    constexpr unsigned int CHUNK_SIZE = 16;
    constexpr unsigned int SUPERCHUNK_SIZE = 16;
    constexpr unsigned int PACKED_BYTES = (CHUNK_SIZE * NBITS) >> 3;
    constexpr unsigned int PER_CHUNK_BYTES = PACKED_BYTES + 1;
    constexpr unsigned int SUPERCHUNK_BYTES = SUPERCHUNK_SIZE * PER_CHUNK_BYTES + 2;
    constexpr int RANGE = (1 << (NBITS - 1)) - 1;

    extern __shared__ float sdata_fused[];
    float* send_raw_chunk_scales = sdata_fused;
    float* send_global_chunk_scale = send_raw_chunk_scales + SUPERCHUNK_SIZE;
    float* send_super_scale_unit = send_global_chunk_scale + 1;

    unsigned int sid = threadIdx.x;
    unsigned int local_chunk_id = sid >> 4;
    unsigned int inner_chunk_idx = sid & (CHUNK_SIZE - 1);
    unsigned int tid = blockIdx.x * (SUPERCHUNK_SIZE * CHUNK_SIZE) + sid;
    unsigned int half_mask = ((threadIdx.x & 16) == 0) ? 0x0000ffffu : 0xffff0000u;

    unsigned int super_start_addr = blockIdx.x * SUPERCHUNK_BYTES;
    unsigned int recv_chunk_start_addr = super_start_addr + local_chunk_id * PER_CHUNK_BYTES;
    unsigned int send_chunk_start_addr = recv_chunk_start_addr;
    unsigned int super_scale_addr = super_start_addr + SUPERCHUNK_SIZE * PER_CHUNK_BYTES;

    float recv_chunk_scale = 0.0f;
    if (inner_chunk_idx == 0) {
        float recv_super_scale = scaling_hier_nonnegative_scale(__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(recv + super_scale_addr)));
        recv_chunk_scale = scaling_hier_nonnegative_scale((float)recv[recv_chunk_start_addr + PACKED_BYTES] * recv_super_scale);
    }
    recv_chunk_scale = __shfl_sync(half_mask, recv_chunk_scale, 0, CHUNK_SIZE);

    uint8_t recv_code = scaling_hier_load_packed_code_16<NBITS>(recv, recv_chunk_start_addr, inner_chunk_idx);
    float decompressed = (float)((int)recv_code - RANGE) * recv_chunk_scale;
    float updated = __bfloat162float(inp[tid]) + decompressed;
    float local_val = __bfloat162float(__float2bfloat16(updated));
    float raw_chunk_scale = scaling_hier_chunk_absmax_16(fabsf(local_val));
    raw_chunk_scale = scaling_hier_positive_scale(raw_chunk_scale / (float)RANGE + 1e-7f);

    if (inner_chunk_idx == 0) {
        send_raw_chunk_scales[local_chunk_id] = raw_chunk_scale;
    }
    __syncthreads();

    if (sid == 0) {
        float global_chunk_max = send_raw_chunk_scales[0];
        #pragma unroll
        for (int i = 1; i < SUPERCHUNK_SIZE; i++) {
            global_chunk_max = fmaxf(global_chunk_max, send_raw_chunk_scales[i]);
        }
        *send_global_chunk_scale = scaling_hier_positive_scale(global_chunk_max);
        __nv_bfloat16 stored_super_scale_unit = scaling_hier_bf16_scale_to_store(*send_global_chunk_scale / 255.0f);
        *send_super_scale_unit = scaling_hier_bf16_positive_scale(*send_global_chunk_scale / 255.0f);
        *reinterpret_cast<__nv_bfloat16*>(send + super_scale_addr) = stored_super_scale_unit;
    }
    __syncthreads();

    float decoded_chunk_scale = 0.0f;
    if (inner_chunk_idx == 0) {
        uint8_t quantized_chunk_max = scaling_hier_quantized_chunk_scale(send_raw_chunk_scales[local_chunk_id], *send_global_chunk_scale);
        send[send_chunk_start_addr + PACKED_BYTES] = (uint8_t)quantized_chunk_max;
        decoded_chunk_scale = scaling_hier_positive_scale((float)quantized_chunk_max * (*send_super_scale_unit));
    }
    decoded_chunk_scale = __shfl_sync(half_mask, decoded_chunk_scale, 0, CHUNK_SIZE);

    local_val /= scaling_hier_positive_scale(decoded_chunk_scale);
    uint8_t send_code = scaling_hier_signed_to_code(
        scaling_hier_stochastic_round_to_signed_float(local_val, __bfloat162float(rand_pool[tid])),
        RANGE
    );
    scaling_hier_store_packed_code_16<NBITS>(send, send_chunk_start_addr, inner_chunk_idx, send_code);
}

template<int NBITS>
__global__ void scaling_dec_comp_kernel_mee_hierarchical_bf16_opt(
    const uint8_t* __restrict__ recv,
    const __nv_bfloat16* __restrict__ inp,
    uint8_t* __restrict__ send,
    __nv_bfloat16* __restrict__ rand_pool,
    int n,
    const __half* __restrict__ mee_table,
    float one_plus_eps2_over_2eps2,
    float chunk_max_mee,
    float log2_one_plus_2eps2
) {
    (void)n;
    constexpr unsigned int CHUNK_SIZE = 16;
    constexpr unsigned int SUPERCHUNK_SIZE = 16;
    constexpr unsigned int PACKED_BYTES = (CHUNK_SIZE * NBITS) >> 3;
    constexpr unsigned int PER_CHUNK_BYTES = PACKED_BYTES + 1;
    constexpr unsigned int SUPERCHUNK_BYTES = SUPERCHUNK_SIZE * PER_CHUNK_BYTES + 2;
    constexpr unsigned int TABLE_SIZE = 1u << NBITS;
    constexpr int RANGE = (1 << (NBITS - 1)) - 1;

    extern __shared__ float sdata_fused[];
    float* s_table = sdata_fused;
    float* send_raw_chunk_scales = s_table + TABLE_SIZE;
    float* send_global_chunk_scale = send_raw_chunk_scales + SUPERCHUNK_SIZE;
    float* send_super_scale_unit = send_global_chunk_scale + 1;

    for (unsigned int table_idx = threadIdx.x; table_idx < TABLE_SIZE; table_idx += blockDim.x) {
        s_table[table_idx] = __half2float(mee_table[NBITS * 256 + table_idx]);
    }
    __syncthreads();

    unsigned int sid = threadIdx.x;
    unsigned int local_chunk_id = sid >> 4;
    unsigned int inner_chunk_idx = sid & (CHUNK_SIZE - 1);
    unsigned int tid = blockIdx.x * (SUPERCHUNK_SIZE * CHUNK_SIZE) + sid;
    unsigned int half_mask = ((threadIdx.x & 16) == 0) ? 0x0000ffffu : 0xffff0000u;

    unsigned int super_start_addr = blockIdx.x * SUPERCHUNK_BYTES;
    unsigned int recv_chunk_start_addr = super_start_addr + local_chunk_id * PER_CHUNK_BYTES;
    unsigned int send_chunk_start_addr = recv_chunk_start_addr;
    unsigned int super_scale_addr = super_start_addr + SUPERCHUNK_SIZE * PER_CHUNK_BYTES;

    float recv_chunk_scale = 0.0f;
    if (inner_chunk_idx == 0) {
        float recv_super_scale = scaling_hier_nonnegative_scale(__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(recv + super_scale_addr)));
        recv_chunk_scale = scaling_hier_nonnegative_scale((float)recv[recv_chunk_start_addr + PACKED_BYTES] * recv_super_scale);
    }
    recv_chunk_scale = __shfl_sync(half_mask, recv_chunk_scale, 0, CHUNK_SIZE);

    uint8_t recv_code = scaling_hier_load_packed_code_16<NBITS>(recv, recv_chunk_start_addr, inner_chunk_idx);
    float decompressed = s_table[recv_code] * recv_chunk_scale;
    float updated = __bfloat162float(inp[tid]) + decompressed;
    float local_val = __bfloat162float(__float2bfloat16(updated));
    float raw_chunk_scale = scaling_hier_chunk_absmax_16(fabsf(local_val));
    raw_chunk_scale = scaling_hier_positive_scale(raw_chunk_scale / scaling_hier_positive_scale(chunk_max_mee) + 1e-7f);

    if (inner_chunk_idx == 0) {
        send_raw_chunk_scales[local_chunk_id] = raw_chunk_scale;
    }
    __syncthreads();

    if (sid == 0) {
        float global_chunk_max = send_raw_chunk_scales[0];
        #pragma unroll
        for (int i = 1; i < SUPERCHUNK_SIZE; i++) {
            global_chunk_max = fmaxf(global_chunk_max, send_raw_chunk_scales[i]);
        }
        *send_global_chunk_scale = scaling_hier_positive_scale(global_chunk_max);
        __nv_bfloat16 stored_super_scale_unit = scaling_hier_bf16_scale_to_store(*send_global_chunk_scale / 255.0f);
        *send_super_scale_unit = scaling_hier_bf16_positive_scale(*send_global_chunk_scale / 255.0f);
        *reinterpret_cast<__nv_bfloat16*>(send + super_scale_addr) = stored_super_scale_unit;
    }
    __syncthreads();

    float decoded_chunk_scale = 0.0f;
    if (inner_chunk_idx == 0) {
        uint8_t quantized_chunk_max = scaling_hier_quantized_chunk_scale(send_raw_chunk_scales[local_chunk_id], *send_global_chunk_scale);
        send[send_chunk_start_addr + PACKED_BYTES] = (uint8_t)quantized_chunk_max;
        decoded_chunk_scale = scaling_hier_positive_scale((float)quantized_chunk_max * (*send_super_scale_unit));
    }
    decoded_chunk_scale = __shfl_sync(half_mask, decoded_chunk_scale, 0, CHUNK_SIZE);

    local_val /= scaling_hier_positive_scale(decoded_chunk_scale);
    float rev_local_val = scaling_hier_mee_inverse_lattice_float(local_val, one_plus_eps2_over_2eps2, log2_one_plus_2eps2);
    uint8_t send_code = scaling_hier_signed_to_code(
        scaling_hier_stochastic_round_to_signed_float(rev_local_val, __bfloat162float(rand_pool[tid])),
        RANGE
    );
    scaling_hier_store_packed_code_16<NBITS>(send, send_chunk_start_addr, inner_chunk_idx, send_code);
}

__global__ void scaling_compress_kernel_aee_hierarchical(__half* src, uint8_t* dst, __half* rand_pool, int n, int nbits, int chunk_size, int superchunk_size) {
    (void)n;
    extern __shared__ __half sdata[];
    __half* superchunk_data = sdata;

    unsigned int sid = threadIdx.x;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_start_addr = blockIdx.x * superchunk_bytes;

    unsigned int local_chunk_id = sid / chunk_size;
    unsigned int inner_chunk_idx = sid & (chunk_size - 1);
    unsigned int chunk_id = blockIdx.x * superchunk_size + local_chunk_id;
    unsigned int tid = chunk_id * chunk_size + inner_chunk_idx;

    __half local_val = src[tid];
    __half chunk_max = __habs(local_val);
    chunk_max = scaling_hier_chunk_absmax(chunk_max, chunk_size);

    uint16_t range = (1 << (nbits - 1)) - 1;
    chunk_max = __hadd(__hdiv(chunk_max, __ushort2half_rn(range)), __float2half(1e-7f));

    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    if (inner_chunk_idx == 0) {
        superchunk_data[local_chunk_id] = chunk_max;
    }
    __syncthreads();

    if (sid == 0) {
        __half global_chunk_max = superchunk_data[0];
        for (int i = 1; i < superchunk_size; i++) {
            global_chunk_max = __hmax(global_chunk_max, superchunk_data[i]);
        }
        superchunk_data[superchunk_size] = global_chunk_max;
    }
    __syncthreads();

    __half global_chunk_max = superchunk_data[superchunk_size];
    __half super_scale_unit = __hdiv(global_chunk_max, __float2half(255.0f));
    uint16_t quantized_chunk_max = __half2ushort_ru(__hmul(__hdiv(chunk_max, global_chunk_max), __float2half(255.0f)));
    if (quantized_chunk_max > 255) {
        quantized_chunk_max = 255;
    }
    __half decoded_chunk_max = __hmul(__ushort2half_rn(quantized_chunk_max), super_scale_unit);

    if (inner_chunk_idx == 0) {
        dst[chunk_end_addr] = (uint8_t)quantized_chunk_max;

        if (local_chunk_id == 0) {
            *(__half*)&dst[superchunk_end_addr] = super_scale_unit;
        }
    }

    local_val = __hdiv(local_val, decoded_chunk_max);
    uint8_t int_floored_local_val = scaling_hier_signed_to_code(
        scaling_hier_stochastic_round_to_signed(local_val, rand_pool[tid]),
        range
    );

    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint32_t packed_local_val = ((uint32_t)int_floored_local_val) << bit_offset;
    unsigned int pack_mask = scaling_hier_chunk_mask(chunk_size);

    for (int stride = 4 / nbits; stride >= 1; stride >>= 1) {
        packed_local_val |= __shfl_down_sync(pack_mask, packed_local_val, stride);
    }

    if (bit_offset == 0) {
        dst[byte_addr] = (uint8_t)packed_local_val;
    }
}

__global__ void scaling_decompress_kernel_add_aee_hierarchical(uint8_t* src, __half* dst, int n, int nbits, int chunk_size, int superchunk_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) {
        return;
    }

    unsigned int chunk_id = tid / chunk_size;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_id = chunk_id / superchunk_size;
    unsigned int local_chunk_id = chunk_id - super_id * superchunk_size;
    unsigned int super_start_addr = super_id * superchunk_bytes;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint16_t int_chunk_max = (uint16_t)src[chunk_end_addr];
    __half chunk_max = __ushort2half_rn(int_chunk_max);
    __half global_chunk_max = *(__half*)&src[superchunk_end_addr];
    chunk_max = __hmul(chunk_max, global_chunk_max);

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    int_floored_local_val -= (1 << (nbits - 1)) - 1;
    __half local_val = __short2half_rn(int_floored_local_val);
    local_val = __hmul(local_val, chunk_max);

    dst[tid] = __hadd(dst[tid], local_val);
}

__global__ void scaling_decompress_kernel_unadd_aee_hierarchical(uint8_t* src, __half* dst, int n, int nbits, int chunk_size, int superchunk_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) {
        return;
    }

    unsigned int chunk_id = tid / chunk_size;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_id = chunk_id / superchunk_size;
    unsigned int local_chunk_id = chunk_id - super_id * superchunk_size;
    unsigned int super_start_addr = super_id * superchunk_bytes;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint16_t int_chunk_max = (uint16_t)src[chunk_end_addr];
    __half chunk_max = __ushort2half_rn(int_chunk_max);
    __half global_chunk_max = *(__half*)&src[superchunk_end_addr];
    chunk_max = __hmul(chunk_max, global_chunk_max);

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    int_floored_local_val -= (1 << (nbits - 1)) - 1;
    __half local_val = __short2half_rn(int_floored_local_val);
    local_val = __hmul(local_val, chunk_max);

    dst[tid] = local_val;
}

__global__ void scaling_compress_kernel_mee_hierarchical(__half* src, uint8_t* dst, __half* rand_pool, int n, int nbits, int chunk_size, int superchunk_size, __half one_plus_eps2_over_2eps2, __half one_plus_2eps2, __half chunk_max_mee, __half log2_one_plus_2eps2) {
    extern __shared__ __half sdata[];
    __half* superchunk_data = sdata;

    unsigned int sid = threadIdx.x;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_start_addr = blockIdx.x * superchunk_bytes;

    unsigned int local_chunk_id = sid / chunk_size;
    unsigned int inner_chunk_idx = sid & (chunk_size - 1);
    unsigned int chunk_id = blockIdx.x * superchunk_size + local_chunk_id;
    unsigned int tid = chunk_id * chunk_size + inner_chunk_idx;

    __half local_val = src[tid];
    __half chunk_max = __habs(local_val);

    chunk_max = scaling_hier_chunk_absmax(chunk_max, chunk_size);

    // normalizing
    uint16_t range = (1 << (nbits - 1)) - 1;
    chunk_max = __hadd(__hdiv(chunk_max, chunk_max_mee), __float2half(1e-7f));

    // chunk_max is the reduced max value of the chunk
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    if (inner_chunk_idx == 0) {
        superchunk_data[local_chunk_id] = chunk_max;
    }
    __syncthreads();

    if (sid == 0) {
        __half global_chunk_max = superchunk_data[0];
        for (int i = 1; i < superchunk_size; i++) {
            global_chunk_max = __hmax(global_chunk_max, superchunk_data[i]);
        }
        superchunk_data[superchunk_size] = global_chunk_max;
    }
    __syncthreads();

    __half global_chunk_max = superchunk_data[superchunk_size];
    __half super_scale_unit = __hdiv(global_chunk_max, __float2half(255.0f));
    uint16_t quantized_chunk_max = __half2ushort_ru(__hmul(__hdiv(chunk_max, global_chunk_max), __float2half(255.0f)));
    if (quantized_chunk_max > 255) {
        quantized_chunk_max = 255;
    }
    __half decoded_chunk_max = __hmul(__ushort2half_rn(quantized_chunk_max), super_scale_unit);

    if (inner_chunk_idx == 0) {
        dst[chunk_end_addr] = (uint8_t)quantized_chunk_max;

        if (local_chunk_id == 0) {
            *(__half*)&dst[superchunk_end_addr] = super_scale_unit;
        }
    }

    local_val = __hdiv(local_val, decoded_chunk_max);
    __half rev_local_val = scaling_hier_mee_inverse_lattice(local_val, one_plus_eps2_over_2eps2, log2_one_plus_2eps2);

    // stochastic quantization
    __half rand_val = rand_pool[tid];

    // if (__hlt(rand_val, local_val)) {
    //     int_floored_local_val += 1;
    // }
    uint8_t int_floored_local_val = scaling_hier_signed_to_code(scaling_hier_stochastic_round_to_signed(rev_local_val, rand_val), range);



    // write back by packing 8/nbits values into a byte
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint32_t packed_local_val = ((uint32_t)int_floored_local_val) << bit_offset;
    unsigned int pack_mask = scaling_hier_chunk_mask(chunk_size);

    for (int stride = 4 / nbits; stride >= 1; stride >>= 1) {
        packed_local_val |= __shfl_down_sync(pack_mask, packed_local_val, stride);
    }

    if (bit_offset == 0) {
        dst[byte_addr] = (uint8_t)packed_local_val;
    }
}

__global__ void scaling_decompress_kernel_add_mee_hierarchical(uint8_t* src, __half* dst, int n, int nbits, int chunk_size, int superchunk_size, const __half* mee_table) { // added to dst
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; // for dst
    unsigned int sid = threadIdx.x;

    extern __shared__ __half sdata[];
    const __half* table = mee_table + nbits * 256;
    for (int table_idx = sid; table_idx < (1 << nbits); table_idx += blockDim.x) {
        sdata[table_idx] = table[table_idx];
    }
    __syncthreads();

    if (tid >= n) {
        return;
    }

    unsigned int chunk_id = tid / chunk_size;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_id = chunk_id / superchunk_size;
    unsigned int local_chunk_id = chunk_id - super_id * superchunk_size;
    unsigned int super_start_addr = super_id * superchunk_bytes;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint16_t int_chunk_max = (uint16_t)src[chunk_end_addr];
    __half chunk_max = __ushort2half_rn(int_chunk_max);
    __half global_chunk_max = *(__half*)&src[superchunk_end_addr];

    chunk_max = __hmul(chunk_max, global_chunk_max); // direct mul

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    __half local_val = sdata[int_floored_local_val];
    local_val = __hmul(local_val, chunk_max);  // direct mul

    dst[tid] = __hadd(dst[tid], local_val);
}

__global__ void scaling_decompress_kernel_unadd_mee_hierarchical(uint8_t* src, __half* dst, int n, int nbits, int chunk_size, int superchunk_size, const __half* mee_table) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; // for dst
    unsigned int sid = threadIdx.x;

    extern __shared__ __half sdata[];
    const __half* table = mee_table + nbits * 256;
    for (int table_idx = sid; table_idx < (1 << nbits); table_idx += blockDim.x) {
        sdata[table_idx] = table[table_idx];
    }
    __syncthreads();

    if (tid >= n) { 
        return;
    }

    unsigned int chunk_id = tid / chunk_size;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_id = chunk_id / superchunk_size;
    unsigned int local_chunk_id = chunk_id - super_id * superchunk_size;
    unsigned int super_start_addr = super_id * superchunk_bytes;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint16_t int_chunk_max = (uint16_t)src[chunk_end_addr];
    __half chunk_max = __ushort2half_rn(int_chunk_max);
    __half global_chunk_max = *(__half*)&src[superchunk_end_addr];
    
    chunk_max = __hmul(chunk_max, global_chunk_max); // direct mul

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    __half local_val = sdata[int_floored_local_val];
    local_val = __hmul(local_val, chunk_max);

    dst[tid] = local_val;
}

__global__ void scaling_compress_kernel_aee_hierarchical_bf16(__nv_bfloat16* src, uint8_t* dst, __nv_bfloat16* rand_pool, int n, int nbits, int chunk_size, int superchunk_size) {
    (void)n;
    extern __shared__ float sdata_float[];
    float* superchunk_data = sdata_float;

    unsigned int sid = threadIdx.x;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_start_addr = blockIdx.x * superchunk_bytes;

    unsigned int local_chunk_id = sid / chunk_size;
    unsigned int inner_chunk_idx = sid & (chunk_size - 1);
    unsigned int chunk_id = blockIdx.x * superchunk_size + local_chunk_id;
    unsigned int tid = chunk_id * chunk_size + inner_chunk_idx;

    float local_val = __bfloat162float(src[tid]);
    float chunk_max = scaling_hier_chunk_absmax_float(fabsf(local_val), chunk_size);

    int range = (1 << (nbits - 1)) - 1;
    chunk_max = scaling_hier_positive_scale(chunk_max / (float)range + 1e-7f);

    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    if (inner_chunk_idx == 0) {
        superchunk_data[local_chunk_id] = chunk_max;
    }
    __syncthreads();

    if (sid == 0) {
        float global_chunk_max = superchunk_data[0];
        for (int i = 1; i < superchunk_size; i++) {
            global_chunk_max = fmaxf(global_chunk_max, superchunk_data[i]);
        }
        superchunk_data[superchunk_size] = global_chunk_max;
    }
    __syncthreads();

    float global_chunk_max = scaling_hier_positive_scale(superchunk_data[superchunk_size]);
    __nv_bfloat16 stored_super_scale_unit = scaling_hier_bf16_scale_to_store(global_chunk_max / 255.0f);
    float super_scale_unit = scaling_hier_bf16_positive_scale(global_chunk_max / 255.0f);
    uint8_t quantized_chunk_max = scaling_hier_quantized_chunk_scale(chunk_max, global_chunk_max);
    float decoded_chunk_max = scaling_hier_positive_scale((float)quantized_chunk_max * super_scale_unit);

    if (inner_chunk_idx == 0) {
        dst[chunk_end_addr] = (uint8_t)quantized_chunk_max;

        if (local_chunk_id == 0) {
            *(__nv_bfloat16*)&dst[superchunk_end_addr] = stored_super_scale_unit;
        }
    }

    local_val /= scaling_hier_positive_scale(decoded_chunk_max);
    uint8_t int_floored_local_val = scaling_hier_signed_to_code(
        scaling_hier_stochastic_round_to_signed_float(local_val, __bfloat162float(rand_pool[tid])),
        range
    );

    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint32_t packed_local_val = ((uint32_t)int_floored_local_val) << bit_offset;
    unsigned int pack_mask = scaling_hier_chunk_mask(chunk_size);

    for (int stride = 4 / nbits; stride >= 1; stride >>= 1) {
        packed_local_val |= __shfl_down_sync(pack_mask, packed_local_val, stride);
    }

    if (bit_offset == 0) {
        dst[byte_addr] = (uint8_t)packed_local_val;
    }
}

__global__ void scaling_decompress_kernel_add_aee_hierarchical_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, int superchunk_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) {
        return;
    }

    unsigned int chunk_id = tid / chunk_size;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_id = chunk_id / superchunk_size;
    unsigned int local_chunk_id = chunk_id - super_id * superchunk_size;
    unsigned int super_start_addr = super_id * superchunk_bytes;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    float chunk_max = scaling_hier_nonnegative_scale((float)src[chunk_end_addr] * scaling_hier_nonnegative_scale(__bfloat162float(*(__nv_bfloat16*)&src[superchunk_end_addr])));
    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    int_floored_local_val -= (1 << (nbits - 1)) - 1;
    float local_val = (float)int_floored_local_val * chunk_max;

    dst[tid] = __float2bfloat16(__bfloat162float(dst[tid]) + local_val);
}

__global__ void scaling_decompress_kernel_unadd_aee_hierarchical_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, int superchunk_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) {
        return;
    }

    unsigned int chunk_id = tid / chunk_size;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_id = chunk_id / superchunk_size;
    unsigned int local_chunk_id = chunk_id - super_id * superchunk_size;
    unsigned int super_start_addr = super_id * superchunk_bytes;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    float chunk_max = scaling_hier_nonnegative_scale((float)src[chunk_end_addr] * scaling_hier_nonnegative_scale(__bfloat162float(*(__nv_bfloat16*)&src[superchunk_end_addr])));
    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    int_floored_local_val -= (1 << (nbits - 1)) - 1;
    float local_val = (float)int_floored_local_val * chunk_max;

    dst[tid] = __float2bfloat16(local_val);
}

__global__ void scaling_compress_kernel_mee_hierarchical_bf16(__nv_bfloat16* src, uint8_t* dst, __nv_bfloat16* rand_pool, int n, int nbits, int chunk_size, int superchunk_size, __half one_plus_eps2_over_2eps2, __half one_plus_2eps2, __half chunk_max_mee, __half log2_one_plus_2eps2) {
    extern __shared__ float sdata_float[];
    float* superchunk_data = sdata_float;

    unsigned int sid = threadIdx.x;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_start_addr = blockIdx.x * superchunk_bytes;

    unsigned int local_chunk_id = sid / chunk_size;
    unsigned int inner_chunk_idx = sid & (chunk_size - 1);
    unsigned int chunk_id = blockIdx.x * superchunk_size + local_chunk_id;
    unsigned int tid = chunk_id * chunk_size + inner_chunk_idx;

    float local_val = __bfloat162float(src[tid]);
    float chunk_max = fabsf(local_val);

    chunk_max = scaling_hier_chunk_absmax_float(chunk_max, chunk_size);

    int range = (1 << (nbits - 1)) - 1;
    chunk_max = scaling_hier_positive_scale(chunk_max / scaling_hier_positive_scale(__half2float(chunk_max_mee)) + 1e-7f);

    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    if (inner_chunk_idx == 0) {
        superchunk_data[local_chunk_id] = chunk_max;
    }
    __syncthreads();

    if (sid == 0) {
        float global_chunk_max = superchunk_data[0];
        for (int i = 1; i < superchunk_size; i++) {
            global_chunk_max = fmaxf(global_chunk_max, superchunk_data[i]);
        }
        superchunk_data[superchunk_size] = global_chunk_max;
    }
    __syncthreads();

    float global_chunk_max = scaling_hier_positive_scale(superchunk_data[superchunk_size]);
    __nv_bfloat16 stored_super_scale_unit = scaling_hier_bf16_scale_to_store(global_chunk_max / 255.0f);
    float super_scale_unit = scaling_hier_bf16_positive_scale(global_chunk_max / 255.0f);
    uint8_t quantized_chunk_max = scaling_hier_quantized_chunk_scale(chunk_max, global_chunk_max);
    float decoded_chunk_max = scaling_hier_positive_scale((float)quantized_chunk_max * super_scale_unit);

    if (inner_chunk_idx == 0) {
        dst[chunk_end_addr] = (uint8_t)quantized_chunk_max;

        if (local_chunk_id == 0) {
            *(__nv_bfloat16*)&dst[superchunk_end_addr] = stored_super_scale_unit;
        }
    }

    local_val /= scaling_hier_positive_scale(decoded_chunk_max);
    float rev_local_val = scaling_hier_mee_inverse_lattice_float(local_val, __half2float(one_plus_eps2_over_2eps2), __half2float(log2_one_plus_2eps2));

    uint8_t int_floored_local_val = scaling_hier_signed_to_code(scaling_hier_stochastic_round_to_signed_float(rev_local_val, __bfloat162float(rand_pool[tid])), range);

    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint32_t packed_local_val = ((uint32_t)int_floored_local_val) << bit_offset;
    unsigned int pack_mask = scaling_hier_chunk_mask(chunk_size);

    for (int stride = 4 / nbits; stride >= 1; stride >>= 1) { 
        packed_local_val |= __shfl_down_sync(pack_mask, packed_local_val, stride);
    }

    if (bit_offset == 0) {
        dst[byte_addr] = (uint8_t)packed_local_val;
    }
}

__global__ void scaling_decompress_kernel_add_mee_hierarchical_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, int superchunk_size, const __half* mee_table) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sid = threadIdx.x;

    extern __shared__ __half sdata[];
    const __half* table = mee_table + nbits * 256;
    for (int table_idx = sid; table_idx < (1 << nbits); table_idx += blockDim.x) {
        sdata[table_idx] = table[table_idx];
    }
    __syncthreads();

    if (tid >= n) {
        return;
    }

    unsigned int chunk_id = tid / chunk_size;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_id = chunk_id / superchunk_size;
    unsigned int local_chunk_id = chunk_id - super_id * superchunk_size;
    unsigned int super_start_addr = super_id * superchunk_bytes;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    float chunk_max = scaling_hier_nonnegative_scale((float)src[chunk_end_addr] * scaling_hier_nonnegative_scale(__bfloat162float(*(__nv_bfloat16*)&src[superchunk_end_addr])));
    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    float local_val = __half2float(sdata[int_floored_local_val]) * chunk_max;

    dst[tid] = __float2bfloat16(__bfloat162float(dst[tid]) + local_val);
}

__global__ void scaling_decompress_kernel_unadd_mee_hierarchical_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, int superchunk_size, const __half* mee_table) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sid = threadIdx.x;

    extern __shared__ __half sdata[];
    const __half* table = mee_table + nbits * 256;
    for (int table_idx = sid; table_idx < (1 << nbits); table_idx += blockDim.x) {
        sdata[table_idx] = table[table_idx];
    }
    __syncthreads();

    if (tid >= n) {
        return;
    }

    unsigned int chunk_id = tid / chunk_size;
    unsigned int packed_bytes = (chunk_size * nbits) >> 3;
    unsigned int per_chunk_bytes = packed_bytes + 1;
    unsigned int superchunk_bytes = superchunk_size * per_chunk_bytes + 2;
    unsigned int super_id = chunk_id / superchunk_size;
    unsigned int local_chunk_id = chunk_id - super_id * superchunk_size;
    unsigned int super_start_addr = super_id * superchunk_bytes;
    unsigned int chunk_start_addr = super_start_addr + local_chunk_id * per_chunk_bytes;
    unsigned int chunk_end_addr = chunk_start_addr + packed_bytes;
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    unsigned int superchunk_end_addr = super_start_addr + superchunk_size * per_chunk_bytes;

    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    float chunk_max = scaling_hier_nonnegative_scale((float)src[chunk_end_addr] * scaling_hier_nonnegative_scale(__bfloat162float(*(__nv_bfloat16*)&src[superchunk_end_addr])));
    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    float local_val = __half2float(sdata[int_floored_local_val]) * chunk_max;

    dst[tid] = __float2bfloat16(local_val);
}

void launch_scaling_compress_kernel_aee_hierarchical_bf16_optimized(
    __nv_bfloat16* src,
    uint8_t* dst,
    __nv_bfloat16* rand_pool,
    int n,
    int nbits,
    cudaStream_t stream
) {
    constexpr int threads = 16 * 16;
    int num_superchunks = n / threads;
    size_t shared_mem = (16 * 2 + 2) * sizeof(float);

    if (nbits == 2) {
        scaling_compress_kernel_aee_hierarchical_bf16_opt<2><<<num_superchunks, threads, shared_mem, stream>>>(
            src, dst, rand_pool, n
        );
    }
    else if (nbits == 4) {
        scaling_compress_kernel_aee_hierarchical_bf16_opt<4><<<num_superchunks, threads, shared_mem, stream>>>(
            src, dst, rand_pool, n
        );
    }
    else {
        scaling_compress_kernel_aee_hierarchical_bf16_opt<8><<<num_superchunks, threads, shared_mem, stream>>>(
            src, dst, rand_pool, n
        );
    }
}

void launch_scaling_decompress_kernel_aee_hierarchical_bf16_optimized(
    uint8_t* src,
    __nv_bfloat16* dst,
    int n,
    int nbits,
    cudaStream_t stream,
    int add_original
) {
    int num_superchunks = n / (16 * 16);

    if (nbits == 2) {
        constexpr int threads = 16 * 4;
        size_t shared_mem = (16 + 1) * sizeof(float);
        if (add_original) {
            scaling_decompress_kernel_aee_hierarchical_bf16_opt<2, true><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n);
        }
        else {
            scaling_decompress_kernel_aee_hierarchical_bf16_opt<2, false><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n);
        }
    }
    else if (nbits == 4) {
        constexpr int threads = 16 * 8;
        size_t shared_mem = (16 + 1) * sizeof(float);
        if (add_original) {
            scaling_decompress_kernel_aee_hierarchical_bf16_opt<4, true><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n);
        }
        else {
            scaling_decompress_kernel_aee_hierarchical_bf16_opt<4, false><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n);
        }
    }
    else {
        constexpr int threads = 16 * 8;
        size_t shared_mem = (16 + 1) * sizeof(float);
        if (add_original) {
            scaling_decompress_kernel_aee_hierarchical_bf16_opt<8, true><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n);
        }
        else {
            scaling_decompress_kernel_aee_hierarchical_bf16_opt<8, false><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n);
        }
    }
}

void launch_scaling_dec_comp_kernel_aee_hierarchical_bf16_optimized(
    uint8_t* recv,
    const __nv_bfloat16* inp,
    uint8_t* send,
    __nv_bfloat16* rand_pool,
    int n,
    int nbits,
    cudaStream_t stream
) {
    constexpr int threads = 16 * 16;
    int num_superchunks = n / threads;
    size_t shared_mem = (16 + 2) * sizeof(float);

    if (nbits == 2) {
        scaling_dec_comp_kernel_aee_hierarchical_bf16_opt<2><<<num_superchunks, threads, shared_mem, stream>>>(
            recv, inp, send, rand_pool, n
        );
    }
    else if (nbits == 4) {
        scaling_dec_comp_kernel_aee_hierarchical_bf16_opt<4><<<num_superchunks, threads, shared_mem, stream>>>(
            recv, inp, send, rand_pool, n
        );
    }
    else {
        scaling_dec_comp_kernel_aee_hierarchical_bf16_opt<8><<<num_superchunks, threads, shared_mem, stream>>>(
            recv, inp, send, rand_pool, n
        );
    }
}

void launch_scaling_compress_kernel_mee_hierarchical_bf16_optimized(
    __nv_bfloat16* src,
    uint8_t* dst,
    __nv_bfloat16* rand_pool,
    int n,
    int nbits,
    cudaStream_t stream,
    __half one_plus_eps2_over_2eps2,
    __half chunk_max_mee,
    __half log2_one_plus_2eps2
) {
    constexpr int threads = 16 * 16;
    int num_superchunks = n / threads;
    size_t shared_mem = (16 * 2 + 2) * sizeof(float);
    float one_plus_eps2_over_2eps2_f = __half2float(one_plus_eps2_over_2eps2);
    float chunk_max_mee_f = __half2float(chunk_max_mee);
    float log2_one_plus_2eps2_f = __half2float(log2_one_plus_2eps2);

    if (nbits == 2) {
        scaling_compress_kernel_mee_hierarchical_bf16_opt<2><<<num_superchunks, threads, shared_mem, stream>>>(
            src, dst, rand_pool, n, one_plus_eps2_over_2eps2_f, chunk_max_mee_f, log2_one_plus_2eps2_f
        );
    }
    else if (nbits == 4) {
        scaling_compress_kernel_mee_hierarchical_bf16_opt<4><<<num_superchunks, threads, shared_mem, stream>>>(
            src, dst, rand_pool, n, one_plus_eps2_over_2eps2_f, chunk_max_mee_f, log2_one_plus_2eps2_f
        );
    }
    else {
        scaling_compress_kernel_mee_hierarchical_bf16_opt<8><<<num_superchunks, threads, shared_mem, stream>>>(
            src, dst, rand_pool, n, one_plus_eps2_over_2eps2_f, chunk_max_mee_f, log2_one_plus_2eps2_f
        );
    }
}

void launch_scaling_decompress_kernel_mee_hierarchical_bf16_optimized(
    uint8_t* src,
    __nv_bfloat16* dst,
    int n,
    int nbits,
    cudaStream_t stream,
    int add_original,
    const __half* mee_table
) {
    int num_superchunks = n / (16 * 16);

    if (nbits == 2) {
        constexpr int threads = 16 * 4;
        size_t shared_mem = ((1 << 2) + 16 + 1) * sizeof(float);
        if (add_original) {
            scaling_decompress_kernel_mee_hierarchical_bf16_opt<2, true><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n, mee_table);
        }
        else {
            scaling_decompress_kernel_mee_hierarchical_bf16_opt<2, false><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n, mee_table);
        }
    }
    else if (nbits == 4) {
        constexpr int threads = 16 * 8;
        size_t shared_mem = ((1 << 4) + 16 + 1) * sizeof(float);
        if (add_original) {
            scaling_decompress_kernel_mee_hierarchical_bf16_opt<4, true><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n, mee_table);
        }
        else {
            scaling_decompress_kernel_mee_hierarchical_bf16_opt<4, false><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n, mee_table);
        }
    }
    else {
        constexpr int threads = 16 * 8;
        size_t shared_mem = ((1 << 8) + 16 + 1) * sizeof(float);
        if (add_original) {
            scaling_decompress_kernel_mee_hierarchical_bf16_opt<8, true><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n, mee_table);
        }
        else {
            scaling_decompress_kernel_mee_hierarchical_bf16_opt<8, false><<<num_superchunks, threads, shared_mem, stream>>>(src, dst, n, mee_table);
        }
    }
}

void launch_scaling_dec_comp_kernel_mee_hierarchical_bf16_optimized(
    uint8_t* recv,
    const __nv_bfloat16* inp,
    uint8_t* send,
    __nv_bfloat16* rand_pool,
    int n,
    int nbits,
    cudaStream_t stream,
    const __half* mee_table,
    __half one_plus_eps2_over_2eps2,
    __half chunk_max_mee,
    __half log2_one_plus_2eps2
) {
    constexpr int threads = 16 * 16;
    int num_superchunks = n / threads;
    float one_plus_eps2_over_2eps2_f = __half2float(one_plus_eps2_over_2eps2);
    float chunk_max_mee_f = __half2float(chunk_max_mee);
    float log2_one_plus_2eps2_f = __half2float(log2_one_plus_2eps2);

    if (nbits == 2) {
        size_t shared_mem = ((1 << 2) + 16 + 2) * sizeof(float);
        scaling_dec_comp_kernel_mee_hierarchical_bf16_opt<2><<<num_superchunks, threads, shared_mem, stream>>>(
            recv, inp, send, rand_pool, n, mee_table, one_plus_eps2_over_2eps2_f, chunk_max_mee_f, log2_one_plus_2eps2_f
        );
    }
    else if (nbits == 4) {
        size_t shared_mem = ((1 << 4) + 16 + 2) * sizeof(float);
        scaling_dec_comp_kernel_mee_hierarchical_bf16_opt<4><<<num_superchunks, threads, shared_mem, stream>>>(
            recv, inp, send, rand_pool, n, mee_table, one_plus_eps2_over_2eps2_f, chunk_max_mee_f, log2_one_plus_2eps2_f
        );
    }
    else {
        size_t shared_mem = ((1 << 8) + 16 + 2) * sizeof(float);
        scaling_dec_comp_kernel_mee_hierarchical_bf16_opt<8><<<num_superchunks, threads, shared_mem, stream>>>(
            recv, inp, send, rand_pool, n, mee_table, one_plus_eps2_over_2eps2_f, chunk_max_mee_f, log2_one_plus_2eps2_f
        );
    }
}
