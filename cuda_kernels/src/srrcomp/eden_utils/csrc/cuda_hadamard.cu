/* Copyright 2022 VMware, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* @author: Shay Vargaftik (VMware Research) */

/* 
 * Inspired by CUDA samples https://docs.nvidia.com/cuda/cuda-samples/index.html (see notice below). 
 * 
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "cuda_hadamard.h"
#include <unistd.h>

extern cudaEvent_t cu_event;

#define CHUNK_SIZE (1 << 28)

struct floatpack {
    union {
        float x[4];
        uint64_t storage[2];
    };
};


inline __device__ void load128(const uint64_t* ptr, floatpack& v) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
      : "=l"(v.storage[0]), "=l"(v.storage[1]) : "l"(ptr));
}

inline __device__ void store128(const uint64_t* ptr, floatpack v) {
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(ptr), "l"(v.storage[0]), "l"(v.storage[1]) : "memory");
}

inline __device__ void load128_shmem(const uint64_t* ptr, floatpack& v) {
  asm volatile("ld.volatile.shared.v2.u64 {%0,%1}, [%2];"
      : "=l"(v.storage[0]), "=l"(v.storage[1]) : "l"(ptr));
}

inline __device__ void store128_shmem(const uint64_t* ptr, floatpack v) {
  asm volatile("st.volatile.shared.v2.u64 [%2], {%0,%1};"
      :: "l"(v.storage[0]), "l"(v.storage[1]), "l"(ptr));
}

inline __device__ uint64_t* shmemCvtPtr(volatile uint64_t* shmemGenericPtr) {
  uint64_t* shmemAsmPtr;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(shmemAsmPtr) : "l"(shmemGenericPtr));
  return shmemAsmPtr;
}

struct floatpack2 {
    union {
        float x[2];
        uint64_t storage;
    };
};
__global__ void HadamardSharedMemoryIterations(float* vec, unsigned int iters)
{
	// see https://developer.nvidia.com/blog/cooperative-groups/
	namespace cg = cooperative_groups;

	// handle to thread block
	cg::thread_block block = cg::this_thread_block();

	// block's shared memory
	extern __shared__ float shared_memory[];

	// each block perform fwht to a chuck of size n using n / blockDim.x threads
	unsigned int n = 1 << iters;

	// This is the offset of the current block's chunk
	float* block_vec = vec + (blockIdx.x << iters);

	// copy block values to shared memory - each thread copies n / blockDim.x values
	for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
	{
		shared_memory[i] = block_vec[i];
	}

	// initial stride size
	unsigned int stride = 1;

	// requires radix2 step
	if (iters % 2 != 0) {

		// make sure all block values are available in shared memory 
		cg::sync(block);

		for (unsigned int h = threadIdx.x; h < (n >> 1); h += blockDim.x)
		{
			unsigned int index_a = h << 1;
			unsigned int index_b = index_a + 1;

			float a = shared_memory[index_a];
			float b = shared_memory[index_b];

			shared_memory[index_a] = a + b;
			shared_memory[index_b] = a - b;
		}

		stride <<= 1;
	}

	// the rest are radix4 steps
	for (; stride <= (n >> 2); stride <<= 2)
	{
		for (unsigned int h = threadIdx.x; h < (n >> 2); h += blockDim.x)
		{
			unsigned int offset = h & (stride - 1);

			unsigned int index_a = ((h - offset) << 2) + offset;
			unsigned int index_b = index_a + stride;
			unsigned int index_c = index_b + stride;
			unsigned int index_d = index_c + stride;

			// make sure all block threads' updated values are available in shared memory 
			cg::sync(block);

			float a = shared_memory[index_a];
			float b = shared_memory[index_b];
			float c = shared_memory[index_c];
			float d = shared_memory[index_d];

			// radix 2 for [a,b]
			float temp1 = a + b;
			float temp2 = a - b;

			// radix 2 for [c,d]
			float temp3 = c + d;
			float temp4 = c - d;

			// radix 2 for [a,b] and [c,d]
			shared_memory[index_a] = temp1 + temp3;
			shared_memory[index_b] = temp2 + temp4;
			shared_memory[index_c] = temp1 - temp3;
			shared_memory[index_d] = temp2 - temp4;
		}
	}

	// all block threads' values are available in shared memory 
	cg::sync(block);

	// copy values from shared memory
	for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
	{
		block_vec[i] = shared_memory[i];
	}
}

__global__ void HadamardRadix2Iteration(float* vec, unsigned int stride)
{
	unsigned int h = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int index_a = (h << 1) - (h & (stride - 1));
	unsigned int index_b = index_a + stride;

	float a = vec[index_a];
	float b = vec[index_b];

	vec[index_a] = a + b;
	vec[index_b] = a - b;
}

__global__ void HadamardRadix4Iteration(float* vec, unsigned int stride)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	// same as index % stride
	unsigned int offset = index & (stride - 1);

	unsigned int index_a = ((index - offset) << 2) + offset;
	unsigned int index_b = index_a + stride;
	unsigned int index_c = index_b + stride;
	unsigned int index_d = index_c + stride;

	float a = vec[index_a];
	float b = vec[index_b];
	float c = vec[index_c];
	float d = vec[index_d];

	// radix 2 for [a,b]
	float temp1 = a + b;
	float temp2 = a - b;

	// radix 2 for [c,d]
	float temp3 = c + d;
	float temp4 = c - d;

	// radix 2 for [a,b] and [c,d]
	vec[index_a] = temp1 + temp3;
	vec[index_b] = temp2 + temp4;
	vec[index_c] = temp1 - temp3;
	vec[index_d] = temp2 - temp4;
}

__global__ void HadamardRadix8Iteration(float* vec, unsigned int stride)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int offset = index & (stride - 1);
    unsigned int index_a = ((index - offset) << 3) + offset;

    // uint32_t chunk_offset = 0;

    // two cases
    // case #1: len <= CHUNK_SIZE, then #block * #thread == len
    // case #2: len > CHUNK_SIZE, then len must be a multiple of CHUNK_SIZE and #block * #thread == CHUNK_SIZE
    // while (chunk_offset < len) {
        // same as index % stride
    unsigned int index_b = index_a + stride;
    unsigned int index_c = index_b + stride;
    unsigned int index_d = index_c + stride;
    unsigned int index_e = index_d + stride;
    unsigned int index_f = index_e + stride;
    unsigned int index_g = index_f + stride;
    unsigned int index_h = index_g + stride;

    // radix 4 for [a,b,c,d]
    float a = vec[index_a];
    float b = vec[index_b];
    float c = vec[index_c];
    float d = vec[index_d];

    float temp1 = a + b;
    float temp2 = a - b;
    float temp3 = c + d;
    float temp4 = c - d;

    a = temp1 + temp3;
    b = temp2 + temp4;
    c = temp1 - temp3;
    d = temp2 - temp4;

    // radix 4 for [e,f,g,h]
    float e = vec[index_e];
    float f = vec[index_f];
    float g = vec[index_g];
    float h = vec[index_h];

    temp1 = e + f;
    temp2 = e - f;
    temp3 = g + h;
    temp4 = g - h;

    e = temp1 + temp3;
    f = temp2 + temp4;
    g = temp1 - temp3;
    h = temp2 - temp4;

    // radix 2 for [a,b,c,d] and [e,f,g,h]
    // if (index_a == 0) {
    //     printf("ok, %f %f %f %f", vec[index_a], vec[index_b], vec[index_c], vec[index_d]);
    // }
    vec[index_a] = a + e;
    vec[index_b] = b + f;
    vec[index_c] = c + g;
    vec[index_d] = d + h;
    vec[index_e] = a - e;
    vec[index_f] = b - f;
    vec[index_g] = c - g;
    vec[index_h] = d - h;

    //     chunk_offset += CHUNK_SIZE;
    //     index_a += CHUNK_SIZE;
    // }
}

__global__ void HadamardRadix8Iteration_load128(float* vec, unsigned int stride, unsigned int len) { // assuming 16 bytes aligned and stride >= 4
	unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) << 2; // 4 floats per pack
    unsigned int offset = index & (stride - 1);
    unsigned int index_a = ((index - offset) << 3) + offset;

    float temp1, temp2, temp3, temp4;
    unsigned int index_b, index_c, index_d, index_e, index_f, index_g, index_h;

    floatpack pa, pb, pc, pd, pe, pf, pg, ph;

    uint32_t chunk_offset = 0;

    // two cases
    // case #1: len <= CHUNK_SIZE, then #block * #thread == len
    // case #2: len > CHUNK_SIZE, then len must be a multiple of CHUNK_SIZE and #block * #thread == CHUNK_SIZE
    while (chunk_offset < len) {
        // same as index % stride
        index_b = index_a + stride;
        index_c = index_b + stride;
        index_d = index_c + stride;
        index_e = index_d + stride;
        index_f = index_e + stride;
        index_g = index_f + stride;
        index_h = index_g + stride;

        // if (index_h >= len) {
        //     printf("error. index_h = %u, index_a = %u, stride = %u, address = %llu\n", index_h, index_a, stride, vec + index_a);
        // }
        

        load128((uint64_t*)(vec + index_a), pa);
        load128((uint64_t*)(vec + index_b), pb);
        load128((uint64_t*)(vec + index_c), pc);
        load128((uint64_t*)(vec + index_d), pd);
        load128((uint64_t*)(vec + index_e), pe);
        load128((uint64_t*)(vec + index_f), pf);
        load128((uint64_t*)(vec + index_g), pg);
        load128((uint64_t*)(vec + index_h), ph);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            temp1 = pa.x[j] + pb.x[j];
            temp2 = pa.x[j] - pb.x[j];
            temp3 = pc.x[j] + pd.x[j];
            temp4 = pc.x[j] - pd.x[j];

            pa.x[j] = temp1 + temp3;
            pb.x[j] = temp2 + temp4;
            pc.x[j] = temp1 - temp3;
            pd.x[j] = temp2 - temp4;
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {

            temp1 = pe.x[j] + pf.x[j];
            temp2 = pe.x[j] - pf.x[j];
            temp3 = pg.x[j] + ph.x[j];
            temp4 = pg.x[j] - ph.x[j];

            pe.x[j] = temp1 + temp3;
            pf.x[j] = temp2 + temp4;
            pg.x[j] = temp1 - temp3;
            ph.x[j] = temp2 - temp4;

            // radix 2 for [a,b,c,d] and [e,f,g,h]
            // if (index_a == 0) {
            //     printf("ok, %f %f %f %f", vec[index_a], vec[index_b], vec[index_c], vec[index_d]);
            // }
            temp1 = pe.x[j];
            temp2 = pf.x[j];
            temp3 = pg.x[j];
            temp4 = ph.x[j];
            pe.x[j] = pa.x[j] - temp1;
            pa.x[j] = pa.x[j] + temp1;
            pf.x[j] = pb.x[j] - temp2;
            pb.x[j] = pb.x[j] + temp2;
            pg.x[j] = pc.x[j] - temp3;
            pc.x[j] = pc.x[j] + temp3;
            ph.x[j] = pd.x[j] - temp4;
            pd.x[j] = pd.x[j] + temp4;
        }

        store128((uint64_t*)(vec + index_a), pa);
        store128((uint64_t*)(vec + index_b), pb);
        store128((uint64_t*)(vec + index_c), pc);
        store128((uint64_t*)(vec + index_d), pd);
        store128((uint64_t*)(vec + index_e), pe);
        store128((uint64_t*)(vec + index_f), pf);
        store128((uint64_t*)(vec + index_g), pg);
        store128((uint64_t*)(vec + index_h), ph);

        chunk_offset += CHUNK_SIZE;
        index_a += CHUNK_SIZE;
    }

	
}

__global__ void HadamardRadix8Iteration_load64(float* vec, unsigned int stride, unsigned int len) { // assuming 16 bytes aligned and stride >= 2
	unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1; // 2 floats per pack
    unsigned int offset = index & (stride - 1);
    unsigned int index_a = ((index - offset) << 3) + offset;

    float temp1, temp2, temp3, temp4;
    unsigned int index_b, index_c, index_d, index_e, index_f, index_g, index_h;

    floatpack2 pa, pb, pc, pd, pe, pf, pg, ph;

    uint32_t chunk_offset = 0;

    // two cases
    // case #1: len <= CHUNK_SIZE, then #block * #thread == len
    // case #2: len > CHUNK_SIZE, then len must be a multiple of CHUNK_SIZE and #block * #thread == CHUNK_SIZE
    while (chunk_offset < len) {
        // same as index % stride
        index_b = index_a + stride;
        index_c = index_b + stride;
        index_d = index_c + stride;
        index_e = index_d + stride;
        index_f = index_e + stride;
        index_g = index_f + stride;
        index_h = index_g + stride;

        // if (index_h >= len) {
        //     printf("error. index_h = %u, index_a = %u, stride = %u, address = %llu\n", index_h, index_a, stride, vec + index_a);
        // }
        

        pa.storage = *(uint64_t*)(vec + index_a);
        pb.storage = *(uint64_t*)(vec + index_b);
        pc.storage = *(uint64_t*)(vec + index_c);
        pd.storage = *(uint64_t*)(vec + index_d);
        pe.storage = *(uint64_t*)(vec + index_e);
        pf.storage = *(uint64_t*)(vec + index_f);
        pg.storage = *(uint64_t*)(vec + index_g);
        ph.storage = *(uint64_t*)(vec + index_h);

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            temp1 = pa.x[j] + pb.x[j];
            temp2 = pa.x[j] - pb.x[j];
            temp3 = pc.x[j] + pd.x[j];
            temp4 = pc.x[j] - pd.x[j];

            pa.x[j] = temp1 + temp3;
            pb.x[j] = temp2 + temp4;
            pc.x[j] = temp1 - temp3;
            pd.x[j] = temp2 - temp4;
        }

        #pragma unroll
        for (int j = 0; j < 2; j++) {

            temp1 = pe.x[j] + pf.x[j];
            temp2 = pe.x[j] - pf.x[j];
            temp3 = pg.x[j] + ph.x[j];
            temp4 = pg.x[j] - ph.x[j];

            pe.x[j] = temp1 + temp3;
            pf.x[j] = temp2 + temp4;
            pg.x[j] = temp1 - temp3;
            ph.x[j] = temp2 - temp4;

            // radix 2 for [a,b,c,d] and [e,f,g,h]
            // if (index_a == 0) {
            //     printf("ok, %f %f %f %f", vec[index_a], vec[index_b], vec[index_c], vec[index_d]);
            // }
            temp1 = pe.x[j];
            temp2 = pf.x[j];
            temp3 = pg.x[j];
            temp4 = ph.x[j];
            pe.x[j] = pa.x[j] - temp1;
            pa.x[j] = pa.x[j] + temp1;
            pf.x[j] = pb.x[j] - temp2;
            pb.x[j] = pb.x[j] + temp2;
            pg.x[j] = pc.x[j] - temp3;
            pc.x[j] = pc.x[j] + temp3;
            ph.x[j] = pd.x[j] - temp4;
            pd.x[j] = pd.x[j] + temp4;
        }

        *(uint64_t*)(vec + index_a) = pa.storage;
        *(uint64_t*)(vec + index_b) = pb.storage;
        *(uint64_t*)(vec + index_c) = pc.storage;
        *(uint64_t*)(vec + index_d) = pd.storage;
        *(uint64_t*)(vec + index_e) = pe.storage;
        *(uint64_t*)(vec + index_f) = pf.storage;
        *(uint64_t*)(vec + index_g) = pg.storage;
        *(uint64_t*)(vec + index_h) = ph.storage;

        chunk_offset += CHUNK_SIZE;
        index_a += CHUNK_SIZE;
    }

	
}

void HadamardWithCudaNoSharedMemory(float* vec, int n, int device, cudaStream_t stream)
{	
	cudaError_t cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCudaNoSharedMemory failed. %s ***\n", cudaGetErrorString(cudaStatus));
	}
	
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	int radix2_num_threads = (n >> 1) < prop.maxThreadsPerBlock ? (n >> 1) : prop.maxThreadsPerBlock;
	int radix4_num_threads = (n >> 2) < prop.maxThreadsPerBlock ? (n >> 2) : prop.maxThreadsPerBlock;
    int block_n = min(n, CHUNK_SIZE);
	int radix8_num_threads = (block_n >> 3) < prop.maxThreadsPerBlock ? (block_n >> 3) : prop.maxThreadsPerBlock;

	int log2n = (int)log2((double)n);
	int len = 1;

    // printf("%lld %d\n", (long long)vec, n);
    
    // cudaStatus = cudaDeviceSynchronize();
    // cudaEventRecord(cu_event);
    // cudaEventSynchronize(cu_event);
    
    
	if ((log2n % 3) % 2 != 0) 
	{
        // printf("Success 1 %d\n", n);
		HadamardRadix2Iteration <<< (n >> 1) / radix2_num_threads, radix2_num_threads, 0, stream >>> (vec, len);
		len <<= 1;
		log2n -= 1;
	}

	if (log2n % 3 != 0) 
	{
        // printf("Success 2 %d\n", n);
		HadamardRadix4Iteration <<< (n >> 2) / radix4_num_threads, radix4_num_threads, 0, stream >>> (vec, len);
		len <<= 2;
	}

	for (; len < n; len <<= 3) 
	{
        // printf("Success 3 %d\n", n);
		HadamardRadix8Iteration <<< (block_n >> 3) / radix8_num_threads, radix8_num_threads, 0, stream >>> (vec, len);
	}
    // cudaStatus = cudaDeviceSynchronize();
    // usleep(100000);
    // printf("%f\n", vec[0]);
    // cudaEventRecord(cu_event);
    // cudaEventSynchronize(cu_event);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCudaNoSharedMemory failed. %s ***\n", cudaGetErrorString(cudaStatus));
	}
}

void HadamardWithCuda(float* vec, int n, int device, cudaStream_t stream, int depth)
{
	if ((n & (n - 1)) != 0)
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. Input size is not a power of 2 ***\n");
		return;
	}
	
	cudaError_t cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. %s ***\n", cudaGetErrorString(cudaStatus));
	}

	int maxThreadsPerBlock;
	int sharedMemPerBlock;

	// read the following attributes from the cuda device
	cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device);
	cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);

	int log2n = (int)log2(n);

	// constraint on block's shared memory size
	int sharedMemPerBlockfFloats = sharedMemPerBlock / sizeof(float);
	int log2nSharedMemPerBlockfFloats = (int)log2(sharedMemPerBlockfFloats);
	int sharedMemIters = log2nSharedMemPerBlockfFloats > log2n ? log2n : log2nSharedMemPerBlockfFloats;

	// must ensure that only radix8 iterations remain after shared memory iterations
	if ((log2n - sharedMemIters) % 3 == 2) 
	{
		sharedMemIters -= 1;
	}
	else if ((log2n - sharedMemIters) % 3 == 1) 
	{
		sharedMemIters -= 2;
	}

	// the shared memory allocated size per block
	int sharedMemSize = 1 << sharedMemIters;

	// the number of transformed chunks after sm iterations
	int num_blocks = 1 << (log2n - sharedMemIters);

	// number of threads per block
	int num_threads;

	if (sharedMemSize == 2) 
	{
		num_threads = 1;
	}
	else 
	{
		num_threads = (sharedMemSize >> 2) < maxThreadsPerBlock ? (sharedMemSize >> 2) : maxThreadsPerBlock;
	}

	HadamardSharedMemoryIterations <<< num_blocks, num_threads, sharedMemSize * sizeof(float), stream >>> (vec, sharedMemIters);

	int radix8_num_threads = (n >> 3) < maxThreadsPerBlock ? (n >> 3) : maxThreadsPerBlock;

	// complete the transform	
	for (int len = sharedMemSize; len < n; len <<= 3) 
	{
		HadamardRadix8Iteration <<< (n >> 3) / radix8_num_threads, radix8_num_threads, 0, stream >>> (vec, len);
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. %s ***\n", cudaGetErrorString(cudaGetLastError()));
	}
}

__global__ void quantization_kernel(float* vec, int n, float div_factor, float tensor_max, curandState_t* states) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    states += index; // state[index]
    unsigned int dim = gridDim.x * blockDim.x;

    float prob;

    while (index < n){
        float temp = vec[index];
        temp /= div_factor; // assertion in python that div_factor != 0
        temp = temp < tensor_max? temp: tensor_max;
        temp = temp > -tensor_max? temp: -tensor_max;

        prob = temp - floor(temp);
        temp = floor(temp);
        temp += curand_uniform(states) < prob;
        vec[index] = temp;

        index += dim; // quantization
    }
}

void quantization_with_cuda(float* vec, int n, int device, float div_factor, float tensor_max, curandState_t* states, cudaStream_t stream)
{
	if ((n & (n - 1)) != 0)
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. Input size is not a power of 2 ***\n");
		return;
	}
	
    cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "%d %d \n*** (CPP) HadamardWithCuda failed. %s ***\n", cudaStatus, cudaSuccess, cudaGetErrorString(cudaStatus));
	}

	// struct cudaDeviceProp prop;
	// cudaGetDeviceProperties(&prop, device);
    int max_threads_per_block;

    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device);

	int num_threads = n < max_threads_per_block ? n : max_threads_per_block;
    int num_blocks = (n + max_threads_per_block - 1) / max_threads_per_block;
    num_blocks = 512 < num_blocks? 512: num_blocks;

    // printf("%d\n", num_blocks * num_threads);
	
	quantization_kernel <<< num_blocks, num_threads, 0, stream>>> (vec, n, div_factor, tensor_max, states);

    // cudaStatus = cudaDeviceSynchronize();
    // cudaStatus = cudaStreamSynchronize(stream1);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. status: %d, %s ***\n", cudaStatus, cudaGetErrorString(cudaGetLastError()));
	}
}


__global__ void initialize_curand_kernel(curandState_t* states) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(12, index, 0, &states[index]);
}

void initialize_curand_states(curandState_t* states, int device) {
	cudaSetDevice(device);

    int max_threads_per_block;

    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device);

    printf("%d\n", 512 * max_threads_per_block);

    // cudaMalloc(&states, sizeof(curandState_t) * 512 * max_threads_per_block);

    // initialize_curand_kernel <<<512, max_threads_per_block, >>> (states);
}



////////////////////////////////////////////////////////////////////////////////////////
// compression and decompression with scaling factors.
////////////////////////////////////////////////////////////////////////////////////////
// MEE: $\ln(f(l,\epsilon) / (1+\epsilon^2) * (2\epsilon^2) + 1) / (\ln(1+2\epsilon^2))$

__device__ __forceinline__ unsigned int scaling_chunk_mask(int chunk_size) {
    unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
    if (chunk_size == WARP_SIZE) {
        return 0xffffffffu;
    }

    unsigned int group_start = lane & ~(chunk_size - 1);
    return ((1u << chunk_size) - 1u) << group_start;
}

__device__ __forceinline__ __half scaling_chunk_absmax(__half local_abs, int chunk_size) {
    unsigned int mask = scaling_chunk_mask(chunk_size);
    for (int stride = 1; stride < chunk_size; stride <<= 1) {
        local_abs = __hmax(local_abs, __shfl_xor_sync(mask, local_abs, stride));
    }
    return local_abs;
}

__device__ __forceinline__ int16_t scaling_stochastic_round_to_signed(__half val, __half rand_val) {
    __half floored_val = hfloor(val);
    int16_t signed_val = __half2short_rd(floored_val);
    __half frac = __hsub(val, floored_val);
    signed_val += __half2short_ru(__hsub(frac, rand_val));
    return signed_val;
}

__device__ __forceinline__ uint8_t scaling_signed_to_code(int16_t signed_val, int range) {
    if (signed_val < -range) {
        signed_val = -range;
    }
    if (signed_val > range) {
        signed_val = range;
    }
    return (uint8_t)(signed_val + range);
}

__device__ __forceinline__ __half scaling_mee_inverse_lattice(__half normalized_val, __half one_plus_eps2_over_2eps2, __half log2_one_plus_2eps2) {
    __half abs_val = __habs(normalized_val);
    if (!__hgt(abs_val, __float2half(0.0f))) {
        return __float2half(0.0f);
    }

    __half log_arg = __hadd(__hdiv(abs_val, one_plus_eps2_over_2eps2), __float2half(1.0f));
    __half lattice_val = __hdiv(hlog2(log_arg), log2_one_plus_2eps2);
    return __hlt(normalized_val, __float2half(0.0f)) ? __hneg(lattice_val) : lattice_val;
}

__device__ __forceinline__ float scaling_chunk_absmax_float(float local_abs, int chunk_size) {
    unsigned int mask = scaling_chunk_mask(chunk_size);
    for (int stride = 1; stride < chunk_size; stride <<= 1) {
        local_abs = fmaxf(local_abs, __shfl_xor_sync(mask, local_abs, stride));
    }
    return local_abs;
}

__device__ __forceinline__ int16_t scaling_stochastic_round_to_signed_float(float val, float rand_val) {
    float floored_val = floorf(val);
    int16_t signed_val = (int16_t)floored_val;
    signed_val += (int16_t)ceilf((val - floored_val) - rand_val);
    return signed_val;
}

__device__ __forceinline__ float scaling_mee_inverse_lattice_float(float normalized_val, float one_plus_eps2_over_2eps2, float log2_one_plus_2eps2) {
    float abs_val = fabsf(normalized_val);
    if (!(abs_val > 0.0f)) {
        return 0.0f;
    }

    float lattice_val = log2f(abs_val / one_plus_eps2_over_2eps2 + 1.0f) / log2_one_plus_2eps2;
    return normalized_val < 0.0f ? -lattice_val : lattice_val;
}


__global__ void scaling_compress_kernel(__half* src, uint8_t* dst, __half* rand_pool, int n, int nbits, int chunk_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; // for src

    if (tid >= n) {
        return;
    }

    unsigned int inner_chunk_idx = tid & (chunk_size - 1);

    __half local_val = src[tid];
    __half chunk_max = __habs(local_val);

    chunk_max = scaling_chunk_absmax(chunk_max, chunk_size);

    // normalizing
    uint16_t range = (1 << (nbits - 1)) - 1;
    chunk_max = __hadd(__hdiv(chunk_max, __ushort2half_rn(range)), __float2half(1e-7f)); // the scaling factor to be encoded is chunk_max / range (such that input_data / scaling factor is what to be encoded; see line 834 local_val = __hdiv(local_val, chunk_max);)

    // chunk_max is the reduced max value of the chunk
    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + ((chunk_size * nbits) >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + ((chunk_size * nbits) >> 3);

    if (inner_chunk_idx == chunk_size - 1) {
        *(__half*)&dst[chunk_end_addr] = chunk_max;
    }

    local_val = __hdiv(local_val, chunk_max);

    // stochastic quantization
    __half rand_val = rand_pool[tid]; // in [0, 1]
    uint8_t int_floored_local_val = scaling_signed_to_code(scaling_stochastic_round_to_signed(local_val, rand_val), range);
    // if (__hlt(rand_val, local_val)) {
    //     int_floored_local_val += 1;
    // }

    // write back by packing 8/nbits values into a byte
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint32_t packed_local_val = ((uint32_t)int_floored_local_val) << bit_offset;
    unsigned int pack_mask = scaling_chunk_mask(chunk_size);

    for (int stride = 4 / nbits; stride >= 1; stride >>= 1) { 
        packed_local_val |= __shfl_down_sync(pack_mask, packed_local_val, stride);
    }

    if (bit_offset == 0) {
        dst[byte_addr] = (uint8_t)packed_local_val;
    }
}

__global__ void scaling_decompress_kernel_add(uint8_t* src, __half* dst, int n, int nbits, int chunk_size) { // added to dst
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; // for dst

    if (tid >= n) {
        return;
    }

    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + (chunk_size * nbits >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + (chunk_size * nbits >> 3);
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    __half chunk_max = *(__half*)&src[chunk_end_addr];

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    int_floored_local_val -= (1 << (nbits - 1)) - 1;
    __half local_val = __short2half_rn(int_floored_local_val);
    local_val = __hmul(local_val, chunk_max);  // direct mul

    dst[tid] = __hadd(dst[tid], local_val);
}

__global__ void scaling_decompress_kernel_unadd(uint8_t* src, __half* dst, int n, int nbits, int chunk_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; // for dst

    if (tid >= n) { 
        return;
    }

    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + ((chunk_size * nbits) >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + ((chunk_size * nbits) >> 3);
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    __half chunk_max = *(__half*)&src[chunk_end_addr];

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    int_floored_local_val -= (1 << (nbits - 1)) - 1;
    __half local_val = __short2half_rn(int_floored_local_val);
    local_val = __hmul(local_val, chunk_max);

    dst[tid] = local_val;
}


lookup_table lut[9];

__device__ lookup_table cu_lut[9];

constexpr int SCALING_MAX_TRACKED_DEVICES = 32;
bool scaling_lut_host_initialized = false;
bool scaling_lut_device_initialized[SCALING_MAX_TRACKED_DEVICES] = {false};
__half* scaling_lut_device_tables[SCALING_MAX_TRACKED_DEVICES] = {nullptr};

void build_lookup_table_host() {
    if (scaling_lut_host_initialized) {
        return;
    }

    float epses[9] = {0, 0.23, 0.23, 0.18, 0.18, 0.18, 0.1, 0.1, 0.1};
    for (int i = 1; i < 9; i++) {
        float eps = epses[i];
        lut[i].one_plus_2eps2 = __float2half(1.0f + 2.0f * eps * eps);
        lut[i].one_plus_eps2_over_2eps2 = __float2half((1.0f + eps * eps) / (2.0f * eps * eps));
        lut[i].log2_one_plus_2eps2 = __float2half(log2f(1.0f + 2.0f * eps * eps));

        float one_plus_2eps2 = 1.0f + 2.0f * eps * eps;
        float pow_eps2 = 1;

        int anchor = (1 << (i - 1)) - 1;
        lut[i].table[anchor] = __float2half(0.0f);
        for (int j = 1; j <= (1 << (i - 1)) - 1; j++) {
            pow_eps2 *= one_plus_2eps2;
            lut[i].table[j + anchor] = __float2half((pow_eps2 - 1) / (2.0f * eps * eps) * (1.0f + eps * eps));
            lut[i].table[anchor - j] = __float2half(-(pow_eps2 - 1) / (2.0f * eps * eps) * (1.0f + eps * eps));
        }

        lut[i].chunk_max_mee = lut[i].table[((1 << (i - 1)) - 1) * 2];
    }
    scaling_lut_host_initialized = true;
}

void init_lookup_table_for_device(int device) {
    build_lookup_table_host();
    if (device >= 0 && device < SCALING_MAX_TRACKED_DEVICES && scaling_lut_device_initialized[device]) {
        return;
    }

    cudaSetDevice(device);
    cudaMemcpyToSymbol(cu_lut, lut, sizeof(lookup_table) * 9);

    __half host_tables[9 * 256];
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 256; j++) {
            host_tables[i * 256 + j] = lut[i].table[j];
        }
    }

    __half* device_table = nullptr;
    cudaMalloc(&device_table, sizeof(host_tables));
    cudaMemcpy(device_table, host_tables, sizeof(host_tables), cudaMemcpyHostToDevice);

    if (device >= 0 && device < SCALING_MAX_TRACKED_DEVICES) {
        scaling_lut_device_tables[device] = device_table;
        scaling_lut_device_initialized[device] = true;
    }
}

__half* get_lookup_table_device_ptr(int device) {
    init_lookup_table_for_device(device);
    if (device >= 0 && device < SCALING_MAX_TRACKED_DEVICES) {
        return scaling_lut_device_tables[device];
    }
    return nullptr;
}

void init_lookup_table() {
    int device = 0;
    cudaGetDevice(&device);
    init_lookup_table_for_device(device);
}

__global__ void scaling_compress_kernel_mee(__half* src, uint8_t* dst, __half* rand_pool, int n, int nbits, int chunk_size, __half one_plus_eps2_over_2eps2, __half one_plus_2eps2, __half chunk_max_mee, __half log2_one_plus_2eps2) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; // for src

    if (tid >= n) {
        return;
    }

    unsigned int inner_chunk_idx = tid & (chunk_size - 1);

    __half local_val = src[tid];
    __half chunk_max = __habs(local_val);

    // __half *shared_table_data = sdata + blockDim.x;
    // if (sid <= (1 << (nbits - 1))) { // nbits = 4-> sid = 0~8
    //     shared_table_data[sid] = cu_lut[nbits].table[sid];
    // }

    chunk_max = scaling_chunk_absmax(chunk_max, chunk_size);

    // normalizing
    uint16_t range = (1 << (nbits - 1)) - 1;
    chunk_max = __hadd(__hdiv(chunk_max, chunk_max_mee), __float2half(1e-7f)); // the scaling factor to be encoded is chunk_max / range (such that input_data / scaling factor is what to be encoded; see line 834 local_val = __hdiv(local_val, chunk_max);)

    // chunk_max is the reduced max value of the chunk
    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + ((chunk_size * nbits) >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + ((chunk_size * nbits) >> 3);

    if (inner_chunk_idx == chunk_size - 1) {
        *(__half*)&dst[chunk_end_addr] = chunk_max;
    }

    local_val = __hdiv(local_val, chunk_max);
    __half rev_local_val = scaling_mee_inverse_lattice(local_val, one_plus_eps2_over_2eps2, log2_one_plus_2eps2);

    // stochastic quantization
    __half rand_val = rand_pool[tid];
    
    // if (__hlt(rand_val, local_val)) {
    //     int_floored_local_val += 1;
    // }
    uint8_t int_floored_local_val = scaling_signed_to_code(scaling_stochastic_round_to_signed(rev_local_val, rand_val), range);

    // write back by packing 8/nbits values into a byte
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint32_t packed_local_val = ((uint32_t)int_floored_local_val) << bit_offset;
    unsigned int pack_mask = scaling_chunk_mask(chunk_size);

    for (int stride = 4 / nbits; stride >= 1; stride >>= 1) { 
        packed_local_val |= __shfl_down_sync(pack_mask, packed_local_val, stride);
    }

    if (bit_offset == 0) {
        dst[byte_addr] = (uint8_t)packed_local_val;
    }
}

__global__ void scaling_decompress_kernel_add_mee(uint8_t* src, __half* dst, int n, int nbits, int chunk_size) { // added to dst
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; // for dst
    unsigned int sid = threadIdx.x; // for sdata

    extern __shared__ __half sdata[];
    for (int table_idx = sid; table_idx < (1 << nbits); table_idx += blockDim.x) {
        sdata[table_idx] = cu_lut[nbits].table[table_idx];
    }
    __syncthreads();

    if (tid >= n) {
        return;
    }

    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + (chunk_size * nbits >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + (chunk_size * nbits >> 3);
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    __half chunk_max = *(__half*)&src[chunk_end_addr];

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    __half local_val = sdata[int_floored_local_val];
    local_val = __hmul(local_val, chunk_max);  // direct mul

    dst[tid] = __hadd(dst[tid], local_val);
}

__global__ void scaling_decompress_kernel_unadd_mee(uint8_t* src, __half* dst, int n, int nbits, int chunk_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; // for dst
    unsigned int sid = threadIdx.x; // for sdata

    extern __shared__ __half sdata[];
    for (int table_idx = sid; table_idx < (1 << nbits); table_idx += blockDim.x) {
        sdata[table_idx] = cu_lut[nbits].table[table_idx];
    }
    __syncthreads();

    if (tid >= n) { 
        return;
    }

    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + ((chunk_size * nbits) >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + ((chunk_size * nbits) >> 3);
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    __half chunk_max = *(__half*)&src[chunk_end_addr];

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    __half local_val = sdata[int_floored_local_val];
    local_val = __hmul(local_val, chunk_max);

    dst[tid] = local_val;
}

__global__ void scaling_compress_kernel_bf16(__nv_bfloat16* src, uint8_t* dst, __nv_bfloat16* rand_pool, int n, int nbits, int chunk_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) {
        return;
    }

    unsigned int inner_chunk_idx = tid & (chunk_size - 1);

    float local_val = __bfloat162float(src[tid]);
    float chunk_max = scaling_chunk_absmax_float(fabsf(local_val), chunk_size);

    int range = (1 << (nbits - 1)) - 1;
    __nv_bfloat16 stored_scale = __float2bfloat16(chunk_max / (float)range + 1e-7f);
    float scale = __bfloat162float(stored_scale);

    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + ((chunk_size * nbits) >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + ((chunk_size * nbits) >> 3);

    if (inner_chunk_idx == chunk_size - 1) {
        *(__nv_bfloat16*)&dst[chunk_end_addr] = stored_scale;
    }

    local_val /= scale;
    uint8_t int_floored_local_val = scaling_signed_to_code(scaling_stochastic_round_to_signed_float(local_val, __bfloat162float(rand_pool[tid])), range);

    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint32_t packed_local_val = ((uint32_t)int_floored_local_val) << bit_offset;
    unsigned int pack_mask = scaling_chunk_mask(chunk_size);

    for (int stride = 4 / nbits; stride >= 1; stride >>= 1) {
        packed_local_val |= __shfl_down_sync(pack_mask, packed_local_val, stride);
    }

    if (bit_offset == 0) {
        dst[byte_addr] = (uint8_t)packed_local_val;
    }
}

__global__ void scaling_decompress_kernel_add_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) {
        return;
    }

    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + (chunk_size * nbits >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + (chunk_size * nbits >> 3);
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    float chunk_max = __bfloat162float(*(__nv_bfloat16*)&src[chunk_end_addr]);

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    int_floored_local_val -= (1 << (nbits - 1)) - 1;
    float local_val = (float)int_floored_local_val * chunk_max;

    dst[tid] = __float2bfloat16(__bfloat162float(dst[tid]) + local_val);
}

__global__ void scaling_decompress_kernel_unadd_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) {
        return;
    }

    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + ((chunk_size * nbits) >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + ((chunk_size * nbits) >> 3);
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    float chunk_max = __bfloat162float(*(__nv_bfloat16*)&src[chunk_end_addr]);

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    int_floored_local_val -= (1 << (nbits - 1)) - 1;
    float local_val = (float)int_floored_local_val * chunk_max;

    dst[tid] = __float2bfloat16(local_val);
}

__global__ void scaling_compress_kernel_mee_bf16(__nv_bfloat16* src, uint8_t* dst, __nv_bfloat16* rand_pool, int n, int nbits, int chunk_size, __half one_plus_eps2_over_2eps2, __half one_plus_2eps2, __half chunk_max_mee, __half log2_one_plus_2eps2) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) {
        return;
    }

    unsigned int inner_chunk_idx = tid & (chunk_size - 1);

    float local_val = __bfloat162float(src[tid]);
    float chunk_max = scaling_chunk_absmax_float(fabsf(local_val), chunk_size);

    int range = (1 << (nbits - 1)) - 1;
    __nv_bfloat16 stored_scale = __float2bfloat16(chunk_max / __half2float(chunk_max_mee) + 1e-7f);
    float scale = __bfloat162float(stored_scale);

    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + ((chunk_size * nbits) >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + ((chunk_size * nbits) >> 3);

    if (inner_chunk_idx == chunk_size - 1) {
        *(__nv_bfloat16*)&dst[chunk_end_addr] = stored_scale;
    }

    local_val /= scale;
    float rev_local_val = scaling_mee_inverse_lattice_float(local_val, __half2float(one_plus_eps2_over_2eps2), __half2float(log2_one_plus_2eps2));
    uint8_t int_floored_local_val = scaling_signed_to_code(scaling_stochastic_round_to_signed_float(rev_local_val, __bfloat162float(rand_pool[tid])), range);

    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    uint32_t packed_local_val = ((uint32_t)int_floored_local_val) << bit_offset;
    unsigned int pack_mask = scaling_chunk_mask(chunk_size);

    for (int stride = 4 / nbits; stride >= 1; stride >>= 1) {
        packed_local_val |= __shfl_down_sync(pack_mask, packed_local_val, stride);
    }

    if (bit_offset == 0) {
        dst[byte_addr] = (uint8_t)packed_local_val;
    }
}

__global__ void scaling_decompress_kernel_add_mee_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, const __half* mee_table) {
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

    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + (chunk_size * nbits >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + (chunk_size * nbits >> 3);
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    float chunk_max = __bfloat162float(*(__nv_bfloat16*)&src[chunk_end_addr]);

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    float local_val = __half2float(sdata[int_floored_local_val]) * chunk_max;

    dst[tid] = __float2bfloat16(__bfloat162float(dst[tid]) + local_val);
}

__global__ void scaling_decompress_kernel_unadd_mee_bf16(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, const __half* mee_table) {
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

    unsigned int chunk_start_addr = (tid / chunk_size) * (2 + ((chunk_size * nbits) >> 3));
    unsigned int chunk_end_addr = chunk_start_addr + ((chunk_size * nbits) >> 3);
    unsigned int byte_addr = chunk_start_addr + (((tid & (chunk_size - 1)) * nbits) >> 3);
    uint8_t bit_offset = ((tid & 7) * nbits) & 7;

    float chunk_max = __bfloat162float(*(__nv_bfloat16*)&src[chunk_end_addr]);

    int16_t int_floored_local_val = (src[byte_addr] >> bit_offset) & ((1 << nbits) - 1);
    float local_val = __half2float(sdata[int_floored_local_val]) * chunk_max;

    dst[tid] = __float2bfloat16(local_val);
}

__device__ __forceinline__ float superchunk_to_float(__half value) {
    return __half2float(value);
}

__device__ __forceinline__ float superchunk_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t superchunk_from_float(float value);

template <>
__device__ __forceinline__ __half superchunk_from_float<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 superchunk_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

__global__ void superchunk_mean_center_kernel(
    const __half* src,
    __half* residual,
    __nv_bfloat16* stats,
    int n,
    int world_size,
    int superchunk_size
) {
    extern __shared__ float shared[];
    float* shared_sum = shared;
    float* shared_sq_norm = shared + superchunk_size;

    int superchunk_id = blockIdx.x;
    int lane = threadIdx.x;
    int start = superchunk_id * superchunk_size;
    int valid = n - start;
    valid = valid > superchunk_size ? superchunk_size : valid;
    valid = valid < 0 ? 0 : valid;

    float value = 0.0f;
    int global_idx = start + lane;
    if (lane < valid) {
        value = __half2float(src[global_idx]) / static_cast<float>(world_size);
    }

    shared_sum[lane] = value;
    shared_sq_norm[lane] = value * value;
    __syncthreads();

    for (int stride = superchunk_size >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared_sum[lane] += shared_sum[lane + stride];
            shared_sq_norm[lane] += shared_sq_norm[lane + stride];
        }
        __syncthreads();
    }

    float mean = valid > 0 ? shared_sum[0] / static_cast<float>(valid) : 0.0f;
    float centered = lane < valid ? value - mean : 0.0f;
    residual[global_idx] = __float2half(centered);

    if (lane == 0) {
        stats[(superchunk_id << 1)] = __float2bfloat16(mean);
        stats[(superchunk_id << 1) + 1] = __float2bfloat16(shared_sq_norm[0]);
    }
}

__global__ void superchunk_mean_center_kernel(
    const __nv_bfloat16* src,
    __nv_bfloat16* residual,
    __nv_bfloat16* stats,
    int n,
    int world_size,
    int superchunk_size
) {
    extern __shared__ float shared[];
    float* shared_sum = shared;
    float* shared_sq_norm = shared + superchunk_size;

    int superchunk_id = blockIdx.x;
    int lane = threadIdx.x;
    int start = superchunk_id * superchunk_size;
    int valid = n - start;
    valid = valid > superchunk_size ? superchunk_size : valid;
    valid = valid < 0 ? 0 : valid;

    float value = 0.0f;
    int global_idx = start + lane;
    if (lane < valid) {
        value = __bfloat162float(src[global_idx]) / static_cast<float>(world_size);
    }

    shared_sum[lane] = value;
    shared_sq_norm[lane] = value * value;
    __syncthreads();

    for (int stride = superchunk_size >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared_sum[lane] += shared_sum[lane + stride];
            shared_sq_norm[lane] += shared_sq_norm[lane + stride];
        }
        __syncthreads();
    }

    float mean = valid > 0 ? shared_sum[0] / static_cast<float>(valid) : 0.0f;
    float centered = lane < valid ? value - mean : 0.0f;
    residual[global_idx] = __float2bfloat16(centered);

    if (lane == 0) {
        stats[(superchunk_id << 1)] = __float2bfloat16(mean);
        stats[(superchunk_id << 1) + 1] = __float2bfloat16(shared_sq_norm[0]);
    }
}

template <typename scalar_t>
__global__ void superchunk_add_mean_copy_kernel(
    const scalar_t* residual,
    scalar_t* dst,
    const __nv_bfloat16* stats,
    int n,
    int superchunk_size
) {
    int superchunk_id = blockIdx.x;
    int lane = threadIdx.x;
    int global_idx = superchunk_id * superchunk_size + lane;
    if (global_idx >= n) {
        return;
    }

    float mean = __bfloat162float(stats[(superchunk_id << 1)]);
    float value = superchunk_to_float(residual[global_idx]) + mean;
    dst[global_idx] = superchunk_from_float<scalar_t>(value);
}

template <typename scalar_t>
void launch_superchunk_mean_center(
    scalar_t* src,
    scalar_t* residual,
    __nv_bfloat16* stats,
    int n,
    int world_size,
    int superchunk_size,
    int device,
    cudaStream_t stream
) {
    cudaError_t cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%d %d \n*** (CPP) superchunk_mean_center failed. %s ***\n", cudaStatus, cudaSuccess, cudaGetErrorString(cudaStatus));
    }

    int num_superchunks = (n + superchunk_size - 1) / superchunk_size;
    superchunk_mean_center_kernel<<<num_superchunks, superchunk_size, 2 * superchunk_size * sizeof(float), stream>>>(
        src,
        residual,
        stats,
        n,
        world_size,
        superchunk_size
    );

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n*** (CPP) superchunk_mean_center failed. status: %d, %s ***\n", cudaStatus, cudaGetErrorString(cudaGetLastError()));
    }
}

template <typename scalar_t>
void launch_superchunk_add_mean_copy(
    scalar_t* residual,
    scalar_t* dst,
    __nv_bfloat16* stats,
    int n,
    int superchunk_size,
    int device,
    cudaStream_t stream
) {
    cudaError_t cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%d %d \n*** (CPP) superchunk_add_mean_copy failed. %s ***\n", cudaStatus, cudaSuccess, cudaGetErrorString(cudaStatus));
    }

    int num_superchunks = (n + superchunk_size - 1) / superchunk_size;
    superchunk_add_mean_copy_kernel<scalar_t><<<num_superchunks, superchunk_size, 0, stream>>>(
        residual,
        dst,
        stats,
        n,
        superchunk_size
    );

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n*** (CPP) superchunk_add_mean_copy failed. status: %d, %s ***\n", cudaStatus, cudaGetErrorString(cudaGetLastError()));
    }
}

void superchunk_mean_center_with_cuda(__half* src, __half* residual, __nv_bfloat16* stats, int n, int world_size, int superchunk_size, int device, cudaStream_t stream) {
    launch_superchunk_mean_center(src, residual, stats, n, world_size, superchunk_size, device, stream);
}

void superchunk_mean_center_with_cuda(__nv_bfloat16* src, __nv_bfloat16* residual, __nv_bfloat16* stats, int n, int world_size, int superchunk_size, int device, cudaStream_t stream) {
    launch_superchunk_mean_center(src, residual, stats, n, world_size, superchunk_size, device, stream);
}

void superchunk_add_mean_copy_with_cuda(__half* residual, __half* dst, __nv_bfloat16* stats, int n, int superchunk_size, int device, cudaStream_t stream) {
    launch_superchunk_add_mean_copy(residual, dst, stats, n, superchunk_size, device, stream);
}

void superchunk_add_mean_copy_with_cuda(__nv_bfloat16* residual, __nv_bfloat16* dst, __nv_bfloat16* stats, int n, int superchunk_size, int device, cudaStream_t stream) {
    launch_superchunk_add_mean_copy(residual, dst, stats, n, superchunk_size, device, stream);
}

void scaling_compress_with_cuda(__half* src, uint8_t* dst, __half* rand_pool, int n, int nbits, int chunk_size, int device, int strategy, cudaStream_t stream) { // 0: aee. 1: mee
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%d %d \n*** (CPP) HadamardWithCuda failed. %s ***\n", cudaStatus, cudaSuccess, cudaGetErrorString(cudaStatus));
    }
    if (strategy == 1 || strategy == 3) {
        init_lookup_table_for_device(device);
    }

    int max_threads_per_block;
    int shared_mem_per_block; // in bytes
    int num_blocks;


    int superchunk_size = 16;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device);
    cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device);

    max_threads_per_block = std::min(max_threads_per_block, n);

    num_blocks = (n + max_threads_per_block - 1) / max_threads_per_block;
    
    if (strategy == 0) {
        scaling_compress_kernel <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, rand_pool, n, nbits, chunk_size);
    }

    else if (strategy == 1) {
        scaling_compress_kernel_mee <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, rand_pool, n, nbits, chunk_size, lut[nbits].one_plus_eps2_over_2eps2, lut[nbits].one_plus_2eps2, lut[nbits].chunk_max_mee, lut[nbits].log2_one_plus_2eps2);
    }

    else if (strategy == 2) {
        int hierarchical_threads_per_block = superchunk_size * chunk_size;
        int hierarchical_num_blocks = n / hierarchical_threads_per_block;
        int hierarchical_shared_mem_per_block = (superchunk_size + 1) * (int)sizeof(__half);
        scaling_compress_kernel_aee_hierarchical <<< hierarchical_num_blocks, hierarchical_threads_per_block, hierarchical_shared_mem_per_block, stream>>> (src, dst, rand_pool, n, nbits, chunk_size, superchunk_size);
    }

    else {
        int hierarchical_threads_per_block = superchunk_size * chunk_size;
        int hierarchical_num_blocks = n / hierarchical_threads_per_block;
        int hierarchical_shared_mem_per_block = (superchunk_size + 1) * (int)sizeof(__half);
        scaling_compress_kernel_mee_hierarchical <<< hierarchical_num_blocks, hierarchical_threads_per_block, hierarchical_shared_mem_per_block, stream>>> (src, dst, rand_pool, n, nbits, chunk_size, superchunk_size, lut[nbits].one_plus_eps2_over_2eps2, lut[nbits].one_plus_2eps2, lut[nbits].chunk_max_mee, lut[nbits].log2_one_plus_2eps2);
    }

    

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. status: %d, %s ***\n", cudaStatus, cudaGetErrorString(cudaGetLastError()));
    }
}



void scaling_decompress_with_cuda(uint8_t* src, __half* dst, int n, int nbits, int chunk_size, int device, int strategy, cudaStream_t stream, int add_original) {
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%d %d \n*** (CPP) HadamardWithCuda failed. %s ***\n", cudaStatus, cudaSuccess, cudaGetErrorString(cudaStatus));
    }
    __half* mee_table = nullptr;
    if (strategy == 1 || strategy == 3) {
        mee_table = get_lookup_table_device_ptr(device);
    }

    int max_threads_per_block;
    int num_blocks;

    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device);

    max_threads_per_block = std::min(max_threads_per_block, n);

    num_blocks = (n + max_threads_per_block - 1) / max_threads_per_block;

    int shared_mem_per_block = (1 << nbits) * sizeof(__half);

    int superchunk_size = 16;
    
    
    if (add_original) {
        if (strategy == 0){
            scaling_decompress_kernel_add <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, n, nbits, chunk_size);
        }
        else if (strategy == 1){
            scaling_decompress_kernel_add_mee <<< num_blocks, max_threads_per_block, shared_mem_per_block, stream>>> (src, dst, n, nbits, chunk_size);
        }
        else if (strategy == 2) {
            scaling_decompress_kernel_add_aee_hierarchical <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, n, nbits, chunk_size, superchunk_size);
        }
        else {
            scaling_decompress_kernel_add_mee_hierarchical <<< num_blocks, max_threads_per_block, shared_mem_per_block, stream>>> (src, dst, n, nbits, chunk_size, superchunk_size, mee_table);
        }
    }
    else {
        if (strategy == 0){ 
            scaling_decompress_kernel_unadd <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, n, nbits, chunk_size);
        }
        else if (strategy == 1) {
            scaling_decompress_kernel_unadd_mee <<< num_blocks, max_threads_per_block, shared_mem_per_block, stream>>> (src, dst, n, nbits, chunk_size);
        }
        else if (strategy == 2) {
            scaling_decompress_kernel_unadd_aee_hierarchical <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, n, nbits, chunk_size, superchunk_size);
        }
        else {
            scaling_decompress_kernel_unadd_mee_hierarchical <<< num_blocks, max_threads_per_block, shared_mem_per_block, stream>>> (src, dst, n, nbits, chunk_size, superchunk_size, mee_table);
        }
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. status: %d, %s ***\n", cudaStatus, cudaGetErrorString(cudaGetLastError()));
    }
}

void scaling_compress_with_cuda(__nv_bfloat16* src, uint8_t* dst, __nv_bfloat16* rand_pool, int n, int nbits, int chunk_size, int device, int strategy, cudaStream_t stream) {
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%d %d \n*** (CPP) HadamardWithCuda failed. %s ***\n", cudaStatus, cudaSuccess, cudaGetErrorString(cudaStatus));
    }
    if (strategy == 1 || strategy == 3) {
        init_lookup_table_for_device(device);
    }

    int max_threads_per_block;
    int num_blocks;
    int superchunk_size = 16;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device);

    max_threads_per_block = std::min(max_threads_per_block, n);
    num_blocks = (n + max_threads_per_block - 1) / max_threads_per_block;

    if (strategy == 0) {
        scaling_compress_kernel_bf16 <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, rand_pool, n, nbits, chunk_size);
    }
    else if (strategy == 1) {
        scaling_compress_kernel_mee_bf16 <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, rand_pool, n, nbits, chunk_size, lut[nbits].one_plus_eps2_over_2eps2, lut[nbits].one_plus_2eps2, lut[nbits].chunk_max_mee, lut[nbits].log2_one_plus_2eps2);
    }
    else if (strategy == 2) {
        if (chunk_size == 16) {
            launch_scaling_compress_kernel_aee_hierarchical_bf16_optimized(
                src,
                dst,
                rand_pool,
                n,
                nbits,
                stream
            );
        }
        else {
            int hierarchical_threads_per_block = superchunk_size * chunk_size;
            int hierarchical_num_blocks = n / hierarchical_threads_per_block;
            int hierarchical_shared_mem_per_block = (superchunk_size + 1) * (int)sizeof(float);
            scaling_compress_kernel_aee_hierarchical_bf16 <<< hierarchical_num_blocks, hierarchical_threads_per_block, hierarchical_shared_mem_per_block, stream>>> (src, dst, rand_pool, n, nbits, chunk_size, superchunk_size);
        }
    }
    else {
        if (chunk_size == 16) {
            launch_scaling_compress_kernel_mee_hierarchical_bf16_optimized(
                src,
                dst,
                rand_pool,
                n,
                nbits,
                stream,
                lut[nbits].one_plus_eps2_over_2eps2,
                lut[nbits].chunk_max_mee,
                lut[nbits].log2_one_plus_2eps2
            );
        }
        else {
            int hierarchical_threads_per_block = superchunk_size * chunk_size;
            int hierarchical_num_blocks = n / hierarchical_threads_per_block;
            int hierarchical_shared_mem_per_block = (superchunk_size + 1) * (int)sizeof(float);
            scaling_compress_kernel_mee_hierarchical_bf16 <<< hierarchical_num_blocks, hierarchical_threads_per_block, hierarchical_shared_mem_per_block, stream>>> (src, dst, rand_pool, n, nbits, chunk_size, superchunk_size, lut[nbits].one_plus_eps2_over_2eps2, lut[nbits].one_plus_2eps2, lut[nbits].chunk_max_mee, lut[nbits].log2_one_plus_2eps2);
        }
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. status: %d, %s ***\n", cudaStatus, cudaGetErrorString(cudaGetLastError()));
    }
}

void scaling_decompress_with_cuda(uint8_t* src, __nv_bfloat16* dst, int n, int nbits, int chunk_size, int device, int strategy, cudaStream_t stream, int add_original) {
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%d %d \n*** (CPP) HadamardWithCuda failed. %s ***\n", cudaStatus, cudaSuccess, cudaGetErrorString(cudaStatus));
    }
    __half* mee_table = nullptr;
    if (strategy == 1 || strategy == 3) {
        mee_table = get_lookup_table_device_ptr(device);
    }

    int max_threads_per_block;
    int num_blocks;

    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device);
    max_threads_per_block = std::min(max_threads_per_block, n);
    num_blocks = (n + max_threads_per_block - 1) / max_threads_per_block;

    int shared_mem_per_block = (1 << nbits) * sizeof(__half);
    int superchunk_size = 16;

    if (add_original) {
        if (strategy == 0) {
            scaling_decompress_kernel_add_bf16 <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, n, nbits, chunk_size);
        }
        else if (strategy == 1) {
            scaling_decompress_kernel_add_mee_bf16 <<< num_blocks, max_threads_per_block, shared_mem_per_block, stream>>> (src, dst, n, nbits, chunk_size, mee_table);
        }
        else if (strategy == 2) {
            if (chunk_size == 16) {
                launch_scaling_decompress_kernel_aee_hierarchical_bf16_optimized(src, dst, n, nbits, stream, 1);
            }
            else {
                scaling_decompress_kernel_add_aee_hierarchical_bf16 <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, n, nbits, chunk_size, superchunk_size);
            }
        }
        else {
            if (chunk_size == 16) {
                launch_scaling_decompress_kernel_mee_hierarchical_bf16_optimized(src, dst, n, nbits, stream, 1, mee_table);
            }
            else {
                scaling_decompress_kernel_add_mee_hierarchical_bf16 <<< num_blocks, max_threads_per_block, shared_mem_per_block, stream>>> (src, dst, n, nbits, chunk_size, superchunk_size, mee_table);
            }
        }
    }
    else {
        if (strategy == 0) {
            scaling_decompress_kernel_unadd_bf16 <<< num_blocks, max_threads_per_block, 0, stream>>> (src, dst, n, nbits, chunk_size);
        }
        else if (strategy == 1) {
            scaling_decompress_kernel_unadd_mee_bf16 <<< num_blocks, max_threads_per_block, shared_mem_per_block, stream>>> (src, dst, n, nbits, chunk_size, mee_table);
        }
        else if (strategy == 2) {
            if (chunk_size == 16) {
                launch_scaling_decompress_kernel_aee_hierarchical_bf16_optimized(src, dst, n, nbits, stream, 0);
            }
            else {
                scaling_decompress_kernel_unadd_aee_hierarchical_bf16 <<< num_blocks, max_threads_per_block, 0, stream >>> (src, dst, n, nbits, chunk_size, superchunk_size);
            }
        }
        else {
            if (chunk_size == 16) {
                launch_scaling_decompress_kernel_mee_hierarchical_bf16_optimized(src, dst, n, nbits, stream, 0, mee_table);
            }
            else {
                scaling_decompress_kernel_unadd_mee_hierarchical_bf16 <<< num_blocks, max_threads_per_block, shared_mem_per_block, stream >>> (src, dst, n, nbits, chunk_size, superchunk_size, mee_table);
            }
        }
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. status: %d, %s ***\n", cudaStatus, cudaGetErrorString(cudaGetLastError()));
    }
}

void scaling_dec_comp_with_cuda(uint8_t* recv, __nv_bfloat16* inp, uint8_t* send, __nv_bfloat16* rand_pool, int n, int nbits, int chunk_size, int device, int strategy, cudaStream_t stream) {
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%d %d \n*** (CPP) HadamardWithCuda failed. %s ***\n", cudaStatus, cudaSuccess, cudaGetErrorString(cudaStatus));
    }

    if (strategy == 2 && chunk_size == 16) {
        launch_scaling_dec_comp_kernel_aee_hierarchical_bf16_optimized(
            recv,
            inp,
            send,
            rand_pool,
            n,
            nbits,
            stream
        );
    }
    else if (strategy == 3 && chunk_size == 16) {
        __half* mee_table = get_lookup_table_device_ptr(device);
        launch_scaling_dec_comp_kernel_mee_hierarchical_bf16_optimized(
            recv,
            inp,
            send,
            rand_pool,
            n,
            nbits,
            stream,
            mee_table,
            lut[nbits].one_plus_eps2_over_2eps2,
            lut[nbits].chunk_max_mee,
            lut[nbits].log2_one_plus_2eps2
        );
    }
    else {
        scaling_decompress_with_cuda(recv, inp, n, nbits, chunk_size, device, strategy, stream, 1);
        scaling_compress_with_cuda(inp, send, rand_pool, n, nbits, chunk_size, device, strategy, stream);
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. status: %d, %s ***\n", cudaStatus, cudaGetErrorString(cudaGetLastError()));
    }
}
