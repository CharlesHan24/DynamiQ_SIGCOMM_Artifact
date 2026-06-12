/* Copyright 2022 VMware, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* @author: Shay Vargaftik (VMware Research) */

#include "cuda_packing.h"

/* EDEN packing */

__global__ void BinsToBitsKernel(int* bins, int nbins, uint32_t* arr, int narr, int nbits)
{
	unsigned int binsIndex = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int arrIndex = binsIndex >> 5;

	int mask = 1;

	int offset = 0;
	int offsetStep = nbins >> 5;

	for (int b = 0; b < nbits; b++)
	{
		int predicate = bins[binsIndex] & mask ? 1 : 0;
		arr[arrIndex + offset] = __ballot_sync(0xFFFFFFFF, predicate);

		mask <<= 1;
		offset += offsetStep;
	}
}

void BinsToBits(int* bins, int nbins, uint32_t* arr, int narr, int nbits, int device)
{
	if ((nbins & 31) != 0)
	{
		fprintf(stderr, "\n*** (CPP) BinsToBits failed. Input is not a multiple of 32 ***\n");
	}
	if ((narr << 5) / nbits != nbins)
	{
		fprintf(stderr, "\n*** (CPP) BinsToBits failed. Size missmatch ***\n");
	}

	cudaError_t cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "\n*** (CPP) BinsToBits failed. %s ***\n", cudaGetErrorString(cudaStatus));
	}

	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	if (prop.warpSize != 32)
	{
		fprintf(stderr, "\n*** (CPP) BinsToBits error: warp size != 32 ***\n");
	}

	int nthreads = prop.maxThreadsPerBlock;

	if (nbins > nthreads)
	{
		BinsToBitsKernel <<< nbins / nthreads, nthreads >>> (bins, nbins, arr, narr, nbits);
	}
	else
	{
		BinsToBitsKernel <<< 1, nbins >>> (bins, nbins, arr, narr, nbits);
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "\n*** (CPP) BinsToBitsKernel failed. %s ***\n", cudaGetErrorString(cudaGetLastError()));
	}
}

/* EDEN unpacking */

__global__ void BitsToBinsKernel(int* bins, int nbins, uint32_t* arr, int narr, int nbits)
{
	unsigned int binsIndex = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int arrIndex = binsIndex >> 5;
	unsigned int arrIndexBit = binsIndex & 31;

	bins[binsIndex] = 0;

	int offset = 0;
	int offsetStep = nbins >> 5; 

	for (int b = 0; b < nbits; b++)
	{
		if (arr[arrIndex + offset] & (1 << arrIndexBit))
		{
			bins[binsIndex] += 1 << b;
		}
		offset += offsetStep;
	}
}

void BitsToBins(int* bins, int nbins, uint32_t* arr, int narr, int nbits, int device)
{
	if ((nbins & 31) != 0)
	{
		fprintf(stderr, "\n*** (CPP) BinsToBits failed. Input is not a multiple of 32 ***\n");
	}
	if ((narr << 5) / nbits != nbins)
	{
		fprintf(stderr, "\n*** (CPP) BinsToBits failed. Size missmatch ***\n");
	}

	cudaError_t cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "\n*** (CPP) BitsToBins failed. %s ***\n", cudaGetErrorString(cudaStatus));
	}

	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	int nthreads = prop.maxThreadsPerBlock;

	if (nbins > nthreads)
	{
		BitsToBinsKernel <<< nbins / nthreads, nthreads >>> (bins, nbins, arr, narr, nbits);
	}
	else
	{
		BitsToBinsKernel <<< 1, nbins >>> (bins, nbins, arr, narr, nbits);
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "\n*** (CPP) BitsToBins failed. %s ***\n", cudaGetErrorString(cudaGetLastError()));
	}
}
