#ifndef _CUDA_HIERARCHICAL_MEE_H
#define _CUDA_HIERARCHICAL_MEE_H

#include "defs.h"

__global__ void scaling_compress_kernel_mee_hierarchical(__half* src, uint8_t* dst, __half* rand_pool, int n, int nbits, int chunk_size, int superchunk_size, __half one_plus_eps2_over_2eps2, __half one_plus_2eps2, __half chunk_max_mee, __half log2_one_plus_2eps2);

__global__ void scaling_compress_kernel_aee_hierarchical(__half* src, uint8_t* dst, __half* rand_pool, int n, int nbits, int chunk_size, int superchunk_size);

__global__ void scaling_decompress_kernel_add_aee_hierarchical(uint8_t* src, __half* dst, int n, int nbits, int chunk_size, int superchunk_size);

__global__ void scaling_decompress_kernel_unadd_aee_hierarchical(uint8_t* src, __half* dst, int n, int nbits, int chunk_size, int superchunk_size);

__global__ void scaling_decompress_kernel_add_mee_hierarchical(uint8_t* src, __half* dst, int n, int nbits, int chunk_size, int superchunk_size, const __half* mee_table);

__global__ void scaling_decompress_kernel_unadd_mee_hierarchical(uint8_t* src, __half* dst, int n, int nbits, int chunk_size, int superchunk_size, const __half* mee_table);

__global__ void scaling_compress_kernel_aee_hierarchical_bf16(__nv_bfloat16* src, uint8_t* dst, __nv_bfloat16* rand_pool, int n, int nbits, int chunk_size, int superchunk_size);

__global__ void scaling_decompress_kernel_add_aee_hierarchical_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, int superchunk_size);

__global__ void scaling_decompress_kernel_unadd_aee_hierarchical_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, int superchunk_size);

__global__ void scaling_compress_kernel_mee_hierarchical_bf16(__nv_bfloat16* src, uint8_t* dst, __nv_bfloat16* rand_pool, int n, int nbits, int chunk_size, int superchunk_size, __half one_plus_eps2_over_2eps2, __half one_plus_2eps2, __half chunk_max_mee, __half log2_one_plus_2eps2);

__global__ void scaling_decompress_kernel_add_mee_hierarchical_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, int superchunk_size, const __half* mee_table);

__global__ void scaling_decompress_kernel_unadd_mee_hierarchical_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, int superchunk_size, const __half* mee_table);

void launch_scaling_compress_kernel_aee_hierarchical_bf16_optimized(__nv_bfloat16* src, uint8_t* dst, __nv_bfloat16* rand_pool, int n, int nbits, cudaStream_t stream);

void launch_scaling_decompress_kernel_aee_hierarchical_bf16_optimized(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, cudaStream_t stream, int add_original);

void launch_scaling_dec_comp_kernel_aee_hierarchical_bf16_optimized(uint8_t* recv, const __nv_bfloat16* inp, uint8_t* send, __nv_bfloat16* rand_pool, int n, int nbits, cudaStream_t stream);

void launch_scaling_compress_kernel_mee_hierarchical_bf16_optimized(__nv_bfloat16* src, uint8_t* dst, __nv_bfloat16* rand_pool, int n, int nbits, cudaStream_t stream, __half one_plus_eps2_over_2eps2, __half chunk_max_mee, __half log2_one_plus_2eps2);

void launch_scaling_decompress_kernel_mee_hierarchical_bf16_optimized(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, cudaStream_t stream, int add_original, const __half* mee_table);

void launch_scaling_dec_comp_kernel_mee_hierarchical_bf16_optimized(uint8_t* recv, const __nv_bfloat16* inp, uint8_t* send, __nv_bfloat16* rand_pool, int n, int nbits, cudaStream_t stream, const __half* mee_table, __half one_plus_eps2_over_2eps2, __half chunk_max_mee, __half log2_one_plus_2eps2);

#endif
