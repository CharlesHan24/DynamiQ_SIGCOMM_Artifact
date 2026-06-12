/* Copyright 2022 VMware, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef CUDA_PACKING
#define CUDA_PACKING

#include "defs.h"

void BinsToBits(int* bins, int nbins, uint32_t* arr, int narr, int nbits, int device);
void BitsToBins(int* bins, int nbins, uint32_t* arr, int narr, int nbits, int device);

#endif