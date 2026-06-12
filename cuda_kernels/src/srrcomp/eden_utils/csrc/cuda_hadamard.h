/* Copyright 2022 VMware, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef CUDA_HADAMARD
#define CUDA_HADAMARD

#include "defs.h"
#include "cuda_hierarchical_mee.h"



void HadamardWithCudaNoSharedMemory(float* vec, int n, int device, cudaStream_t stream);
void HadamardWithCuda(float* vec, int n, int device, cudaStream_t stream, int depth);
void initialize_curand_states(curandState_t* states, int device);
void quantization_with_cuda(float* vec, int n, int device, float div_factor, float tensor_max, curandState_t* states, cudaStream_t stream);

void scaling_compress_with_cuda(__half* src, uint8_t* dst, __half* rand_pool, int n, int nbits, int chunk_size, int device, int strategy, cudaStream_t stream);
void scaling_decompress_with_cuda(uint8_t* src, __half* dst, int n, int nbits, int chunk_size, int device, int strategy, cudaStream_t stream, int add_original);
void scaling_compress_with_cuda(__nv_bfloat16* src, uint8_t* dst, __nv_bfloat16* rand_pool, int n, int nbits, int chunk_size, int device, int strategy, cudaStream_t stream);
void scaling_decompress_with_cuda(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, int device, int strategy, cudaStream_t stream, int add_original);
void scaling_dec_comp_with_cuda(uint8_t* recv, __nv_bfloat16* inp, uint8_t* send, __nv_bfloat16* rand_pool, int n, int nbits, int chunk_size, int device, int strategy, cudaStream_t stream);
void superchunk_mean_center_with_cuda(__half* src, __half* residual, __nv_bfloat16* stats, int n, int world_size, int superchunk_size, int device, cudaStream_t stream);
void superchunk_mean_center_with_cuda(__nv_bfloat16* src, __nv_bfloat16* residual, __nv_bfloat16* stats, int n, int world_size, int superchunk_size, int device, cudaStream_t stream);
void superchunk_add_mean_copy_with_cuda(__half* residual, __half* dst, __nv_bfloat16* stats, int n, int superchunk_size, int device, cudaStream_t stream);
void superchunk_add_mean_copy_with_cuda(__nv_bfloat16* residual, __nv_bfloat16* dst, __nv_bfloat16* stats, int n, int superchunk_size, int device, cudaStream_t stream);

void init_lookup_table();


#endif
