#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <torch/extension.h>

namespace ring_allreduce {

void bf16_add_(const torch::Tensor& dst, const torch::Tensor& src);
std::int64_t mee_compressed_bytes(std::int64_t n, int nbits);
void mee_compress_bf16(const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& rand_pool, int nbits);
void mee_decompress_bf16(const torch::Tensor& src, const torch::Tensor& dst, int nbits);
void mee_decompress_add_bf16(const torch::Tensor& src, const torch::Tensor& dst, int nbits);
void mee_dec_comp_bf16(
    const torch::Tensor& recv,
    const torch::Tensor& inp,
    const torch::Tensor& send,
    const torch::Tensor& rand_pool,
    int nbits);

class RdmaSender {
 public:
  RdmaSender(
      std::string server_addr,
      std::string server_addr2,
      int port,
      int rails,
      int gpu,
      std::size_t total_bytes,
      int gid_index);
  ~RdmaSender();

  void send(const torch::Tensor& src);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class RdmaReceiver {
 public:
  RdmaReceiver(
      std::string server_addr,
      std::string server_addr2,
      int port,
      int rails,
      int gpu,
      std::size_t total_bytes,
      int gid_index);
  ~RdmaReceiver();

  void recv(const torch::Tensor& dst);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class PipelineRdmaSender {
 public:
  PipelineRdmaSender(
      std::string server_addr,
      std::string server_addr2,
      int port,
      int rails,
      int gpu,
      std::size_t total_bytes,
      int gid_index);
  ~PipelineRdmaSender();

  void send(const torch::Tensor& src, int nbits, int chunk_size, int strategy);
  void send_compress_bf16(
      const torch::Tensor& src,
      const torch::Tensor& dst,
      const torch::Tensor& rand_pool,
      int nbits,
      int chunk_size,
      int strategy);
  void send_dec_comp_bf16(
      const torch::Tensor& recv,
      const torch::Tensor& inp,
      const torch::Tensor& send,
      const torch::Tensor& rand_pool,
      int nbits,
      int chunk_size,
      int strategy);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class PipelineRdmaReceiver {
 public:
  PipelineRdmaReceiver(
      std::string server_addr,
      std::string server_addr2,
      int port,
      int rails,
      int gpu,
      std::size_t total_bytes,
      int gid_index);
  ~PipelineRdmaReceiver();

  void recv(const torch::Tensor& dst, int nbits, int chunk_size, int strategy);
  void recv_decompress_bf16(
      const torch::Tensor& src,
      const torch::Tensor& dst,
      int nbits,
      int chunk_size,
      int strategy);
  void recv_dec_comp_bf16(
      const torch::Tensor& recv,
      const torch::Tensor& inp,
      const torch::Tensor& send,
      const torch::Tensor& rand_pool,
      int nbits,
      int chunk_size,
      int strategy);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ring_allreduce
