/* Copyright 2022 VMware, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 */


#ifndef _DEFS_EDEN_H
#define _DEFS_EDEN_H

#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/random.h>

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iomanip>
#include <time.h>
#include <chrono>
#include <algorithm>

using std::cout;
using std::endl;


struct lookup_table {
    __half one_plus_2eps2;
    __half one_plus_eps2_over_2eps2;
    __half chunk_max_mee;
    __half log2_one_plus_2eps2;
    __half table[256];
};

#define WARP_SIZE 32
#endif
