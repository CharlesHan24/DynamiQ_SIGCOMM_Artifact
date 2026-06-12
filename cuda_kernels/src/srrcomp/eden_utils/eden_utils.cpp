// Copyright 2022 VMware, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/* @author: Shay Vargaftik (VMware Research) */

#include <torch/extension.h>
#include "./csrc/cuda_hadamard.h"
#include "./csrc/cuda_packing.h"
#

using namespace std;
using namespace torch::indexing;

cudaEvent_t cu_event;

curandState_t* curand_states;

volatile int first_time = 0;


bool scaling_is_supported_data_dtype(c10::ScalarType dtype) {
    return dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

int64_t superchunk_num_chunks(int n, int superchunk_size) {
    return (static_cast<int64_t>(n) + superchunk_size - 1) / superchunk_size;
}

int64_t superchunk_padded_numel(int n, int superchunk_size) {
    return superchunk_num_chunks(n, superchunk_size) * superchunk_size;
}

int64_t scaling_required_bytes(int n, int nbits, int chunk_size, int strategy) {
    int64_t chunks = n / chunk_size;
    int64_t packed_bytes = (chunk_size * nbits) >> 3;
    if (strategy >= 2) {
        return (chunks / 16) * (16 * (packed_bytes + 1) + 2);
    }
    return chunks * (packed_bytes + 2);
}

void scaling_check_common(int n, int nbits, int chunk_size, int strategy) {
    TORCH_CHECK(n > 0, "n must be positive");
    TORCH_CHECK(nbits == 2 || nbits == 4 || nbits == 8, "nbits must be one of {2, 4, 8}");
    TORCH_CHECK(strategy >= 0 && strategy <= 3, "strategy must be 0 (AEE), 1 (MEE), 2 (hierarchical AEE), or 3 (hierarchical MEE)");
    TORCH_CHECK(chunk_size > 0 && chunk_size <= 32, "chunk_size must be in [1, 32]");
    TORCH_CHECK((chunk_size & (chunk_size - 1)) == 0, "chunk_size must be a power of two");
    TORCH_CHECK((chunk_size * nbits) % 8 == 0, "chunk_size * nbits must be divisible by 8");
    TORCH_CHECK(n % chunk_size == 0, "n must be a multiple of chunk_size");
    if (strategy >= 2) {
        TORCH_CHECK((n / chunk_size) % 16 == 0, "hierarchical mode requires n to contain a whole number of 16-chunk superchunks");
    }
}


torch::Tensor Hadamard(torch::Tensor vec, uint64_t stream_p=0, int depth=0)
{
	TORCH_CHECK(vec.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(vec.dtype() == torch::kFloat32, "input must be a torch.float32 CUDA tensor");

	// size of last dimension
	auto n = vec.size(-1);

	TORCH_CHECK(n == vec.numel(), "input must be 1D");
	TORCH_CHECK((n & (n - 1)) == 0 && n > 0, "input size must be a power of 2");

	// cloning makes the output vector contiguous
	auto output = vec.clone(); 

	// device number
	int device = output.device().index();
	
	// invoke Hadamard kernel
	HadamardWithCuda(output.data_ptr<float>(), n, device, (cudaStream_t)stream_p, depth);
    
	return output;
}


void Hadamard_inplace(torch::Tensor vec, uint64_t stream_p=0, int depth=0)
{
	TORCH_CHECK(vec.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(vec.dtype() == torch::kFloat32, "input must be a torch.float32 CUDA tensor");

	// size of last dimension
	auto n = vec.size(-1);

	TORCH_CHECK(n == vec.numel(), "input must be 1D");
	TORCH_CHECK((n & (n - 1)) == 0 && n > 0, "input size must be a power of 2");

	// cloning makes the output vector contiguous

	// device number
	int device = vec.device().index();
	
	// invoke Hadamard kernel
	HadamardWithCuda(vec.data_ptr<float>(), n, device, (cudaStream_t)stream_p, depth);
}

void Hadamard_inplace_ptr(long long vec, int n, int device, uint64_t stream_p=0, int depth=0)
{
	
	// invoke Hadamard kernel
	HadamardWithCuda((float*)vec, n, device, (cudaStream_t)stream_p, depth);
}


void quantization_compression(torch::Tensor vec, float div_factor, float tensor_max, uint64_t stream_p=0) {
	TORCH_CHECK(vec.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(vec.dtype() == torch::kFloat32, "input must be a torch.float32 CUDA tensor");

	// size of last dimension
	auto n = vec.size(-1);

	// cloning makes the output vector contiguous

	// device number
	int device = vec.device().index();
	
	// invoke Hadamard kernel
	quantization_with_cuda(vec.data_ptr<float>(), n, device, div_factor, tensor_max, curand_states, (cudaStream_t)stream_p);
}

/*
 * compress from src to dst in nbits. 
 * @param src: a read only float16 or bfloat16 tensor.
 * @param dst: in uint8_t. Format: chunk_size number of nbits numbers, followed by one dtype-sized scaling factor.
 * @param rand_pool: a read only tensor with the same dtype as src. Random numbers for scaling.
 * @chunk_size: must be a power of two and <= WARP_SIZE
 * @param n: process the first n elements of src. n is guaranteed to be a multiple of chunk size
 * @param chunk_size: the chunk size, which must be <= 32 and must be a power of 2.
 */
void scaling_compress(torch::Tensor src, torch::Tensor dst, torch::Tensor rand_pool, int n, int nbits, int chunk_size, uint64_t stream_p=0, int strategy=0) { 
    TORCH_CHECK(src.device().type() == torch::kCUDA, "input must be a CUDA tensor");
    TORCH_CHECK(dst.device().type() == torch::kCUDA, "output must be a CUDA tensor");
    TORCH_CHECK(rand_pool.device().type() == torch::kCUDA, "rand_pool must be a CUDA tensor");
    TORCH_CHECK(scaling_is_supported_data_dtype(src.scalar_type()), "input must be torch.float16 or torch.bfloat16");
    TORCH_CHECK(dst.dtype() == torch::kUInt8, "output must be torch.uint8");
    TORCH_CHECK(rand_pool.dtype() == src.dtype(), "rand_pool dtype must match input dtype");
    TORCH_CHECK(src.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(rand_pool.is_contiguous(), "rand_pool must be contiguous");
    scaling_check_common(n, nbits, chunk_size, strategy);
    TORCH_CHECK(src.numel() >= n, "input has fewer than n elements");
    TORCH_CHECK(rand_pool.numel() >= n, "rand_pool has fewer than n elements");
    TORCH_CHECK(dst.numel() >= scaling_required_bytes(n, nbits, chunk_size, strategy), "output buffer is too small");

    int device = src.device().index();
    if (src.scalar_type() == torch::kFloat16) {
        scaling_compress_with_cuda(reinterpret_cast<__half*>(src.data_ptr<at::Half>()), dst.data_ptr<uint8_t>(), reinterpret_cast<__half*>(rand_pool.data_ptr<at::Half>()), n, nbits, chunk_size, device, strategy, (cudaStream_t)stream_p);
    }
    else {
        scaling_compress_with_cuda(reinterpret_cast<__nv_bfloat16*>(src.data_ptr<at::BFloat16>()), dst.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(rand_pool.data_ptr<at::BFloat16>()), n, nbits, chunk_size, device, strategy, (cudaStream_t)stream_p);
    }
}

/*
 * decompress from src to dst in nbits. 
 * @param src: a read only uint8_t tensor. Format: chunk_size number of nbits numbers, followed by one dtype-sized scale.
 * @param dst: in float16 or bfloat16.
 * @param n: process the first n elements of src.
 */
void scaling_decompress(torch::Tensor src, torch::Tensor dst, int n, int nbits, int chunk_size, uint64_t stream_p=0, int strategy=0) { 
    TORCH_CHECK(src.device().type() == torch::kCUDA, "input must be a CUDA tensor");
    TORCH_CHECK(dst.device().type() == torch::kCUDA, "output must be a CUDA tensor");
    TORCH_CHECK(src.dtype() == torch::kUInt8, "input must be torch.uint8");
    TORCH_CHECK(scaling_is_supported_data_dtype(dst.scalar_type()), "output must be torch.float16 or torch.bfloat16");
    TORCH_CHECK(src.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "output must be contiguous");
    scaling_check_common(n, nbits, chunk_size, strategy);
    TORCH_CHECK(src.numel() >= scaling_required_bytes(n, nbits, chunk_size, strategy), "input buffer is too small");
    TORCH_CHECK(dst.numel() >= n, "output has fewer than n elements");

    int device = src.device().index();
    if (dst.scalar_type() == torch::kFloat16) {
        scaling_decompress_with_cuda(src.data_ptr<uint8_t>(), reinterpret_cast<__half*>(dst.data_ptr<at::Half>()), n, nbits, chunk_size, device, strategy, (cudaStream_t)stream_p, 0); // not adding dst's original value to dst.
    }
    else {
        scaling_decompress_with_cuda(src.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()), n, nbits, chunk_size, device, strategy, (cudaStream_t)stream_p, 0);
    }
}


/*
 * decompress recv and add the results to inp. Then compress inp to send.
 * @param recv, send: uint8_t tensors. recv is read only.
 * @param inp: float16 or bfloat16 tensor.
 * @param rand_pool: a read only tensor with the same dtype as inp. Random numbers for scaling.
 * @param n: process the first n elements of recv, inp, send.
 */
void scaling_dec_comp(torch::Tensor recv, torch::Tensor inp, torch::Tensor send, torch::Tensor rand_pool, int n, int nbits, int chunk_size, uint64_t stream_p=0, int strategy=0) {
    TORCH_CHECK(recv.device().type() == torch::kCUDA, "recv must be a CUDA tensor");
    TORCH_CHECK(inp.device().type() == torch::kCUDA, "inp must be a CUDA tensor");
    TORCH_CHECK(send.device().type() == torch::kCUDA, "send must be a CUDA tensor");
    TORCH_CHECK(rand_pool.device().type() == torch::kCUDA, "rand_pool must be a CUDA tensor");
    TORCH_CHECK(recv.dtype() == torch::kUInt8, "recv must be torch.uint8");
    TORCH_CHECK(scaling_is_supported_data_dtype(inp.scalar_type()), "inp must be torch.float16 or torch.bfloat16");
    TORCH_CHECK(send.dtype() == torch::kUInt8, "send must be torch.uint8");
    TORCH_CHECK(rand_pool.dtype() == inp.dtype(), "rand_pool dtype must match inp dtype");
    TORCH_CHECK(recv.is_contiguous(), "recv must be contiguous");
    TORCH_CHECK(inp.is_contiguous(), "inp must be contiguous");
    TORCH_CHECK(send.is_contiguous(), "send must be contiguous");
    TORCH_CHECK(rand_pool.is_contiguous(), "rand_pool must be contiguous");
    scaling_check_common(n, nbits, chunk_size, strategy);
    int64_t required_bytes = scaling_required_bytes(n, nbits, chunk_size, strategy);
    TORCH_CHECK(recv.numel() >= required_bytes, "recv buffer is too small");
    TORCH_CHECK(send.numel() >= required_bytes, "send buffer is too small");
    TORCH_CHECK(inp.numel() >= n, "inp has fewer than n elements");
    TORCH_CHECK(rand_pool.numel() >= n, "rand_pool has fewer than n elements");

    int device = recv.device().index();
    if (inp.scalar_type() == torch::kFloat16) {
        scaling_decompress_with_cuda(recv.data_ptr<uint8_t>(), reinterpret_cast<__half*>(inp.data_ptr<at::Half>()), n, nbits, chunk_size, device, strategy, (cudaStream_t)stream_p, 1); // also adding inp's original value to inp.
        scaling_compress_with_cuda(reinterpret_cast<__half*>(inp.data_ptr<at::Half>()), send.data_ptr<uint8_t>(), reinterpret_cast<__half*>(rand_pool.data_ptr<at::Half>()), n, nbits, chunk_size, device, strategy, (cudaStream_t)stream_p);
    }
    else {
        scaling_dec_comp_with_cuda(recv.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(inp.data_ptr<at::BFloat16>()), send.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(rand_pool.data_ptr<at::BFloat16>()), n, nbits, chunk_size, device, strategy, (cudaStream_t)stream_p);
    }
}

/*
 * decompress from src and add to dst 
 * @param src: a read only uint8_t tensor. Format: chunk_size number of nbits numbers, followed by one dtype-sized scale.
 * @param dst: in float16 or bfloat16.
 * @param n: process the first n elements of src.
 */
void scaling_decompress_add(torch::Tensor src, torch::Tensor dst, int n, int nbits, int chunk_size, uint64_t stream_p=0, int strategy=0) { 
    TORCH_CHECK(src.device().type() == torch::kCUDA, "input must be a CUDA tensor");
    TORCH_CHECK(dst.device().type() == torch::kCUDA, "output must be a CUDA tensor");
    TORCH_CHECK(src.dtype() == torch::kUInt8, "input must be torch.uint8");
    TORCH_CHECK(scaling_is_supported_data_dtype(dst.scalar_type()), "output must be torch.float16 or torch.bfloat16");
    TORCH_CHECK(src.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "output must be contiguous");
    scaling_check_common(n, nbits, chunk_size, strategy);
    TORCH_CHECK(src.numel() >= scaling_required_bytes(n, nbits, chunk_size, strategy), "input buffer is too small");
    TORCH_CHECK(dst.numel() >= n, "output has fewer than n elements");

    int device = src.device().index();
    if (dst.scalar_type() == torch::kFloat16) {
        scaling_decompress_with_cuda(src.data_ptr<uint8_t>(), reinterpret_cast<__half*>(dst.data_ptr<at::Half>()), n, nbits, chunk_size, device, strategy, (cudaStream_t)stream_p, 1); // adding dst's original value to dst.
    }
    else {
        scaling_decompress_with_cuda(src.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()), n, nbits, chunk_size, device, strategy, (cudaStream_t)stream_p, 1);
    }
}

void superchunk_mean_center(torch::Tensor src, torch::Tensor residual, torch::Tensor stats, int n, int world_size, int superchunk_size=256, uint64_t stream_p=0) {
    TORCH_CHECK(src.device().type() == torch::kCUDA, "src must be a CUDA tensor");
    TORCH_CHECK(residual.device().type() == torch::kCUDA, "residual must be a CUDA tensor");
    TORCH_CHECK(stats.device().type() == torch::kCUDA, "stats must be a CUDA tensor");
    TORCH_CHECK(residual.device() == src.device(), "residual must be on the same CUDA device as src");
    TORCH_CHECK(stats.device() == src.device(), "stats must be on the same CUDA device as src");
    TORCH_CHECK(scaling_is_supported_data_dtype(src.scalar_type()), "src must be torch.float16 or torch.bfloat16");
    TORCH_CHECK(residual.scalar_type() == src.scalar_type(), "residual dtype must match src dtype");
    TORCH_CHECK(stats.dtype() == torch::kBFloat16, "stats must be torch.bfloat16");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(stats.is_contiguous(), "stats must be contiguous");
    TORCH_CHECK(n > 0, "n must be positive");
    TORCH_CHECK(world_size > 0, "world_size must be positive");
    TORCH_CHECK(superchunk_size == 256, "superchunk_size must be 256");
    TORCH_CHECK(src.numel() >= n, "src has fewer than n elements");

    int64_t num_superchunks = superchunk_num_chunks(n, superchunk_size);
    TORCH_CHECK(residual.numel() >= superchunk_padded_numel(n, superchunk_size), "residual buffer is too small");
    TORCH_CHECK(stats.numel() >= num_superchunks * 2, "stats buffer is too small");

    int device = src.device().index();
    if (src.scalar_type() == torch::kFloat16) {
        superchunk_mean_center_with_cuda(
            reinterpret_cast<__half*>(src.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(residual.data_ptr<at::Half>()),
            reinterpret_cast<__nv_bfloat16*>(stats.data_ptr<at::BFloat16>()),
            n,
            world_size,
            superchunk_size,
            device,
            (cudaStream_t)stream_p
        );
    }
    else {
        superchunk_mean_center_with_cuda(
            reinterpret_cast<__nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(stats.data_ptr<at::BFloat16>()),
            n,
            world_size,
            superchunk_size,
            device,
            (cudaStream_t)stream_p
        );
    }
}

void superchunk_add_mean_copy(torch::Tensor residual, torch::Tensor dst, torch::Tensor stats, int n, int superchunk_size=256, uint64_t stream_p=0) {
    TORCH_CHECK(residual.device().type() == torch::kCUDA, "residual must be a CUDA tensor");
    TORCH_CHECK(dst.device().type() == torch::kCUDA, "dst must be a CUDA tensor");
    TORCH_CHECK(stats.device().type() == torch::kCUDA, "stats must be a CUDA tensor");
    TORCH_CHECK(dst.device() == residual.device(), "dst must be on the same CUDA device as residual");
    TORCH_CHECK(stats.device() == residual.device(), "stats must be on the same CUDA device as residual");
    TORCH_CHECK(scaling_is_supported_data_dtype(residual.scalar_type()), "residual must be torch.float16 or torch.bfloat16");
    TORCH_CHECK(dst.scalar_type() == residual.scalar_type(), "dst dtype must match residual dtype");
    TORCH_CHECK(stats.dtype() == torch::kBFloat16, "stats must be torch.bfloat16");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(stats.is_contiguous(), "stats must be contiguous");
    TORCH_CHECK(n > 0, "n must be positive");
    TORCH_CHECK(superchunk_size == 256, "superchunk_size must be 256");
    TORCH_CHECK(dst.numel() >= n, "dst has fewer than n elements");

    int64_t num_superchunks = superchunk_num_chunks(n, superchunk_size);
    TORCH_CHECK(residual.numel() >= superchunk_padded_numel(n, superchunk_size), "residual buffer is too small");
    TORCH_CHECK(stats.numel() >= num_superchunks * 2, "stats buffer is too small");

    int device = residual.device().index();
    if (residual.scalar_type() == torch::kFloat16) {
        superchunk_add_mean_copy_with_cuda(
            reinterpret_cast<__half*>(residual.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(dst.data_ptr<at::Half>()),
            reinterpret_cast<__nv_bfloat16*>(stats.data_ptr<at::BFloat16>()),
            n,
            superchunk_size,
            device,
            (cudaStream_t)stream_p
        );
    }
    else {
        superchunk_add_mean_copy_with_cuda(
            reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(stats.data_ptr<at::BFloat16>()),
            n,
            superchunk_size,
            device,
            (cudaStream_t)stream_p
        );
    }
}



// void Hadamard(long long vec, int n, int device, long long stream_p)
// {
// 	HadamardWithCudaNoSharedMemory((float*)vec, n, device, (cudaStream_t)stream_p);
// }

void Hadamard_init(torch::Tensor vec) {
    int device = vec.device().index();
    //if (!first_time) {
        //first_time = 1;
    curand_states = (curandState_t*)vec.data_ptr<int>(); // address malloc by torch application 
    initialize_curand_states(curand_states, device);
    //}
}

void Hadamard_init_ptr(long long vec, int device=0) {
    curand_states = (curandState_t*)vec; // address malloc by torch application 
    initialize_curand_states(curand_states, device);
}


torch::Tensor EdenBinsToBits(torch::Tensor bins, int nbits)
{
	TORCH_CHECK(bins.device().type() == torch::kCUDA, "bins must be a CUDA tensor");
	TORCH_CHECK(bins.dtype() == torch::kInt32, "bins must be a torch.int32 CUDA tensor");

	// size of last dimension
	auto nbins = bins.size(-1);

	TORCH_CHECK(nbins == bins.numel(), "bins must be 1D");
	TORCH_CHECK((nbins & 31) == 0, "bins size must be a multiple of 32");

	// cloning makes the tensor contiguous
	auto cbins = bins.clone();  
	
	// device number
	int device = cbins.device().index();

	// allocate tensor for packed bits
	auto carr_options = torch::TensorOptions().device(bins.device().type(), device).dtype(torch::kInt32);
	torch::Tensor carr = torch::zeros({(nbins>>5) * nbits}, carr_options);
	
	// call cuda kernel
	BinsToBits((int *)cbins.data_ptr(), nbins, (uint32_t *)carr.data_ptr(), (nbins>>5) * nbits, nbits, device);  

	return carr;
}


torch::Tensor EdenBitsToBins(torch::Tensor arr, int nbits)
{
	TORCH_CHECK(arr.device().type() == torch::kCUDA, "arr must be a CUDA tensor");
	TORCH_CHECK(arr.dtype() == torch::kInt32, "arr must be a torch.int32 CUDA tensor");

	// size of last dimension
	auto narr = arr.size(-1);

	TORCH_CHECK(narr == arr.numel(), "arr must be 1D");

	// cloning makes the tensor contiguous
	auto carr = arr.clone();  

	// device number
	int device = carr.device().index();
	
	// allocate tensor for unpacked bins
	auto cbins_options = torch::TensorOptions().device(arr.device().type(), device).dtype(torch::kInt32);
	torch::Tensor cbins = torch::zeros({(narr / nbits) << 5}, cbins_options);
	
	// call cuda kernel
	BitsToBins((int *)cbins.data_ptr(), (narr / nbits) << 5, (uint32_t *)carr.data_ptr(), narr, nbits, device);  

	return cbins;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, modul) {
	modul.def("Hadamard", &Hadamard, "fast Hadamard transform", py::arg("vec"), py::arg("stream_p") = 0, py::arg("depth") = 0);
	modul.def("EdenBinsToBits", &EdenBinsToBits, "EDEN's packing bins to bits");
	modul.def("EdenBitsToBins", &EdenBitsToBins, "EDEN's unpacking bits to bins");
    modul.def("Hadamard_init", &Hadamard_init, "hadamard init", py::arg("vec"));
    modul.def("Hadamard_init_ptr", &Hadamard_init_ptr, "hadamard init", py::arg("vec"), py::arg("device") = 0);
    modul.def("Hadamard_inplace", &Hadamard_inplace, "fast Hadamard transform", py::arg("vec"), py::arg("stream_p") = 0, py::arg("depth") = 0);
    modul.def("Hadamard_inplace_ptr", &Hadamard_inplace_ptr, "fast Hadamard transform", py::arg("vec"), py::arg("n"), py::arg("device"), py::arg("stream_p") = 0, py::arg("depth") = 0);
    modul.def("quantization_compression", &quantization_compression, "quantization compression", py::arg("vec"), py::arg("div_factor"), py::arg("tensor_max"), py::arg("stream_p") = 0);
    modul.def("scaling_compress", &scaling_compress, "scaling compress", py::arg("src"), py::arg("dst"), py::arg("rand_pool"), py::arg("n"), py::arg("nbits"), py::arg("chunk_size"), py::arg("stream_p") = 0, py::arg("strategy") = 0);
    modul.def("scaling_decompress", &scaling_decompress, "scaling decompress", py::arg("src"), py::arg("dst"), py::arg("n"), py::arg("nbits"), py::arg("chunk_size"), py::arg("stream_p") = 0, py::arg("strategy") = 0);
    modul.def("scaling_decompress_add", &scaling_decompress_add, "scaling decompress and add", py::arg("src"), py::arg("dst"), py::arg("n"), py::arg("nbits"), py::arg("chunk_size"), py::arg("stream_p") = 0, py::arg("strategy") = 0);
    modul.def("scaling_dec_comp", &scaling_dec_comp, "scaling decompress and compress", py::arg("recv"), py::arg("inp"), py::arg("send"), py::arg("rand_pool"), py::arg("n"), py::arg("nbits"), py::arg("chunk_size"), py::arg("stream_p") = 0, py::arg("strategy") = 0);
    modul.def("superchunk_mean_center", &superchunk_mean_center, "mean-center 256-element superchunks and write per-superchunk mean/norm stats", py::arg("src"), py::arg("residual"), py::arg("stats"), py::arg("n"), py::arg("world_size"), py::arg("superchunk_size") = 256, py::arg("stream_p") = 0);
    modul.def("superchunk_add_mean_copy", &superchunk_add_mean_copy, "add reduced per-superchunk means to residuals and copy back valid elements", py::arg("residual"), py::arg("dst"), py::arg("stats"), py::arg("n"), py::arg("superchunk_size") = 256, py::arg("stream_p") = 0);
    modul.def("init_lookup_table", &init_lookup_table, "init lookup table");
}
