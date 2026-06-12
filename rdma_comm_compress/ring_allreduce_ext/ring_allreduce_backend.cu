#include "ring_allreduce_backend.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_hadamard.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ring_allreduce {
namespace {

constexpr std::uint64_t kSenderRdmaWr = 0x1000000000000000ULL;
constexpr std::uint64_t kSenderCreditRecv = 0x2000000000000000ULL;
constexpr std::uint64_t kReceiverNotifyRecv = 0x3000000000000000ULL;
constexpr std::uint64_t kReceiverCreditSend = 0x4000000000000000ULL;
constexpr std::uint32_t kMagic = 0x43505244;
constexpr std::size_t kPipelineChunkBytes = 8ULL * 1024ULL * 1024ULL;
constexpr std::size_t kPipelineSlots = 64;
constexpr int kMeeChunkSize = 16;
constexpr int kMeeStrategy = 3;  // AEE: 2, MEE: 3

struct RemoteInfo {
  std::uint32_t magic = kMagic;
  std::uint64_t addr = 0;
  std::uint32_t rkey = 0;
  std::uint32_t slots = 1;
  std::uint64_t chunk_bytes = 0;
};

struct RdmaContext {
  rdma_event_channel* ec = nullptr;
  rdma_cm_id* listen_id = nullptr;
  rdma_cm_id* id = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_mr* mr = nullptr;
  char* host_buf = nullptr;
};

struct ByteSpan {
  std::size_t offset = 0;
  std::size_t bytes = 0;
};

struct ChunkSpan {
  std::size_t offset = 0;
  std::size_t bytes = 0;
};

struct PendingChunk {
  int rail = 0;
  int slot = 0;
  std::size_t chunk_idx = 0;
};

void validate_nbits(int nbits);

[[noreturn]] void die(const std::string& msg) {
  throw std::runtime_error(msg);
}

void check(bool ok, const std::string& msg) {
  if (!ok) {
    die(msg);
  }
}

void check_cuda(cudaError_t status, const std::string& msg) {
  if (status != cudaSuccess) {
    die(msg + ": " + cudaGetErrorString(status));
  }
}

void check_tensor(bool ok, const std::string& msg) {
  TORCH_CHECK(ok, msg);
}

bool rdma_debug_enabled() {
  const char* value = std::getenv("RING_RDMA_DEBUG");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

double rdma_timeout_seconds() {
  const char* value = std::getenv("RING_RDMA_TIMEOUT_SECONDS");
  if (value == nullptr || value[0] == '\0') {
    return 300.0;
  }
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value) {
    return 300.0;
  }
  return parsed;
}

void check_no_progress_timeout(
    const std::chrono::steady_clock::time_point& last_progress,
    double timeout_seconds,
    const std::string& detail) {
  if (timeout_seconds <= 0.0) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  const double idle_seconds =
      std::chrono::duration<double>(now - last_progress).count();
  if (idle_seconds > timeout_seconds) {
    die("RDMA operation made no progress for " + std::to_string(idle_seconds) +
        "s: " + detail);
  }
}

sockaddr_in make_sockaddr(const std::string& ip, int port) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(port));
  check(inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) == 1, "invalid IPv4 address: " + ip);
  return addr;
}

void wait_cm_event(rdma_event_channel* ec, rdma_cm_event_type expected, rdma_cm_event** out) {
  rdma_cm_event* event = nullptr;
  check(rdma_get_cm_event(ec, &event) == 0, "rdma_get_cm_event failed");
  if (event->event != expected) {
    std::string msg = "unexpected RDMA CM event: ";
    msg += rdma_event_str(event->event);
    msg += ", expected ";
    msg += rdma_event_str(expected);
    rdma_ack_cm_event(event);
    die(msg);
  }
  *out = event;
}

void create_qp_and_memory(RdmaContext& ctx, std::size_t ring_bytes, int cq_depth) {
  ctx.pd = ibv_alloc_pd(ctx.id->verbs);
  check(ctx.pd != nullptr, "ibv_alloc_pd failed");
  ctx.cq = ibv_create_cq(ctx.id->verbs, cq_depth, nullptr, nullptr, 0);
  check(ctx.cq != nullptr, "ibv_create_cq failed");

  ibv_qp_init_attr qp_attr{};
  qp_attr.send_cq = ctx.cq;
  qp_attr.recv_cq = ctx.cq;
  qp_attr.qp_type = IBV_QPT_RC;
  qp_attr.cap.max_send_wr = cq_depth;
  qp_attr.cap.max_recv_wr = cq_depth;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;
  check(rdma_create_qp(ctx.id, ctx.pd, &qp_attr) == 0, "rdma_create_qp failed");

  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ctx.host_buf), ring_bytes, cudaHostAllocPortable),
             "cudaHostAlloc failed");
  std::memset(ctx.host_buf, 0, ring_bytes);
  ctx.mr = ibv_reg_mr(ctx.pd, ctx.host_buf, ring_bytes,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  check(ctx.mr != nullptr, "ibv_reg_mr failed");
}

void set_gid_index_if_requested(rdma_cm_id* id, int gid_index) {
  if (gid_index < 0) {
    return;
  }
  ibv_qp_attr attr{};
  ibv_qp_init_attr init_attr{};
  check(ibv_query_qp(id->qp, &attr, IBV_QP_AV, &init_attr) == 0, "ibv_query_qp failed");
  attr.ah_attr.grh.sgid_index = gid_index;
  check(ibv_modify_qp(id->qp, &attr, IBV_QP_AV) == 0, "ibv_modify_qp gid_index failed");
}

void post_recv(rdma_cm_id* id, std::uint64_t wr_id) {
  ibv_recv_wr wr{};
  ibv_recv_wr* bad = nullptr;
  wr.wr_id = wr_id;
  wr.num_sge = 0;
  check(ibv_post_recv(id->qp, &wr, &bad) == 0, "ibv_post_recv failed");
}

void post_credit_send(rdma_cm_id* id, int slot) {
  ibv_send_wr wr{};
  ibv_send_wr* bad = nullptr;
  wr.wr_id = kReceiverCreditSend | static_cast<std::uint64_t>(slot);
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = htonl(static_cast<std::uint32_t>(slot));
  check(ibv_post_send(id->qp, &wr, &bad) == 0, "ibv_post_send credit failed");
}

std::uint32_t pack_chunk_imm(int slot, std::size_t chunk_idx) {
  check(slot >= 0 && slot <= 0xFFFF, "pipeline slot does not fit in immediate data");
  check(chunk_idx <= 0xFFFF, "pipeline chunk index does not fit in immediate data");
  return (static_cast<std::uint32_t>(chunk_idx) << 16) | static_cast<std::uint32_t>(slot);
}

int unpack_imm_slot(std::uint32_t imm) {
  return static_cast<int>(imm & 0xFFFFU);
}

std::size_t unpack_imm_chunk_idx(std::uint32_t imm) {
  return static_cast<std::size_t>((imm >> 16) & 0xFFFFU);
}

std::uint32_t pack_pipeline_imm(int slot, std::size_t chunk_idx, std::int64_t message_idx) {
  check(slot >= 0 && slot <= 0xFF, "pipeline slot does not fit in immediate data");
  check(chunk_idx <= 0xFFFF, "pipeline chunk index does not fit in immediate data");
  const auto seq = static_cast<std::uint32_t>(message_idx) & 0xFFU;
  return (seq << 24) |
         (static_cast<std::uint32_t>(chunk_idx) << 8) |
         static_cast<std::uint32_t>(slot);
}

int unpack_pipeline_imm_slot(std::uint32_t imm) {
  return static_cast<int>(imm & 0xFFU);
}

std::size_t unpack_pipeline_imm_chunk_idx(std::uint32_t imm) {
  return static_cast<std::size_t>((imm >> 8) & 0xFFFFU);
}

std::uint32_t unpack_pipeline_imm_seq(std::uint32_t imm) {
  return (imm >> 24) & 0xFFU;
}

void post_rdma_write_at(rdma_cm_id* id, ibv_mr* mr, const RemoteInfo& remote,
                        std::size_t local_offset, std::size_t remote_offset,
                        std::uint64_t chunk_idx, std::size_t bytes, bool with_imm,
                        std::uint32_t imm_value) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<std::uintptr_t>(static_cast<char*>(mr->addr) + local_offset);
  sge.length = static_cast<std::uint32_t>(bytes);
  sge.lkey = mr->lkey;

  ibv_send_wr wr{};
  ibv_send_wr* bad = nullptr;
  wr.wr_id = kSenderRdmaWr | chunk_idx;
  wr.opcode = with_imm ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  if (with_imm) {
    wr.imm_data = htonl(static_cast<std::uint32_t>(imm_value));
  }
  wr.wr.rdma.remote_addr = remote.addr + remote_offset;
  wr.wr.rdma.rkey = remote.rkey;
  check(ibv_post_send(id->qp, &wr, &bad) == 0, "ibv_post_send rdma write failed");
}

int poll_one(ibv_cq* cq, ibv_wc* wc) {
  int n = ibv_poll_cq(cq, 1, wc);
  check(n >= 0, "ibv_poll_cq failed");
  if (n == 0) {
    return 0;
  }
  if (wc->status != IBV_WC_SUCCESS) {
    die(std::string("work completion failed: ") + ibv_wc_status_str(wc->status));
  }
  return 1;
}

void cleanup(RdmaContext& ctx) {
  if (ctx.id && ctx.id->qp) {
    rdma_destroy_qp(ctx.id);
  }
  if (ctx.mr) {
    ibv_dereg_mr(ctx.mr);
  }
  if (ctx.host_buf) {
    cudaFreeHost(ctx.host_buf);
  }
  if (ctx.cq) {
    ibv_destroy_cq(ctx.cq);
  }
  if (ctx.pd) {
    ibv_dealloc_pd(ctx.pd);
  }
  if (ctx.id) {
    rdma_destroy_id(ctx.id);
  }
  if (ctx.listen_id) {
    rdma_destroy_id(ctx.listen_id);
  }
  if (ctx.ec) {
    rdma_destroy_event_channel(ctx.ec);
  }
}

std::vector<ByteSpan> split_bytes(std::size_t total_bytes, int rails) {
  check(rails >= 1, "rails must be >= 1");
  check(total_bytes >= static_cast<std::size_t>(rails),
        "message is too small for the selected rail count");
  std::vector<ByteSpan> spans;
  spans.reserve(rails);
  std::size_t offset = 0;
  std::size_t base = total_bytes / static_cast<std::size_t>(rails);
  std::size_t rem = total_bytes % static_cast<std::size_t>(rails);
  for (int rail = 0; rail < rails; ++rail) {
    std::size_t bytes = base + (static_cast<std::size_t>(rail) < rem ? 1 : 0);
    spans.push_back(ByteSpan{offset, bytes});
    offset += bytes;
  }
  return spans;
}

std::vector<ChunkSpan> split_rail_into_chunks(std::size_t rail_bytes) {
  std::vector<ChunkSpan> chunks;
  std::size_t offset = 0;
  while (offset < rail_bytes) {
    std::size_t bytes = kPipelineChunkBytes;
    if (bytes > rail_bytes - offset) {
      bytes = rail_bytes - offset;
    }
    chunks.push_back(ChunkSpan{offset, bytes});
    offset += bytes;
  }
  return chunks;
}

std::size_t align_down(std::size_t value, std::size_t alignment) {
  if (alignment <= 1) {
    return value;
  }
  return (value / alignment) * alignment;
}

std::size_t pipeline_window() {
  const char* value = std::getenv("RING_RDMA_PIPELINE_INFLIGHT");
  if (value == nullptr || value[0] == '\0') {
    return 2;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0) {
    return 2;
  }
  return static_cast<std::size_t>(parsed);
}

std::size_t pipeline_chunk_bytes() {
  const char* value = std::getenv("RING_RDMA_PIPELINE_CHUNK_MB");
  if (value == nullptr || value[0] == '\0') {
    return kPipelineChunkBytes;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0) {
    return kPipelineChunkBytes;
  }
  return static_cast<std::size_t>(parsed) * 1024ULL * 1024ULL;
}

void validate_pipeline_scaling(int nbits, int chunk_size, int strategy) {
  validate_nbits(nbits);
  check(chunk_size > 0, "chunk_size must be positive");
  check(strategy == 2 || strategy == 3,
        "pipeline scaling supports hierarchical AEE/MEE only");
  check((chunk_size * nbits) % 8 == 0,
        "chunk_size * nbits must be byte aligned");
}

std::size_t pipeline_superchunk_elems(int chunk_size) {
  return static_cast<std::size_t>(chunk_size) * 16ULL;
}

std::size_t pipeline_record_bytes(int nbits, int chunk_size, int strategy) {
  validate_pipeline_scaling(nbits, chunk_size, strategy);
  const std::size_t packed_bytes =
      (static_cast<std::size_t>(chunk_size) * static_cast<std::size_t>(nbits)) / 8ULL;
  return 16ULL * (packed_bytes + 1ULL) + 2ULL;
}

std::size_t pipeline_required_bytes(std::int64_t n, int nbits, int chunk_size, int strategy) {
  check(n >= 0, "n must be non-negative");
  const std::size_t superchunk_elems = pipeline_superchunk_elems(chunk_size);
  const std::size_t record_bytes = pipeline_record_bytes(nbits, chunk_size, strategy);
  const std::size_t nn = static_cast<std::size_t>(n);
  const std::size_t superchunks = (nn + superchunk_elems - 1ULL) / superchunk_elems;
  return superchunks * record_bytes;
}

std::vector<ByteSpan> split_bytes_aligned(std::size_t total_bytes, int rails, std::size_t alignment) {
  check(rails == 1 || rails == 2, "aligned split supports one or two rails");
  if (rails == 1) {
    return {ByteSpan{0, total_bytes}};
  }
  check(alignment > 0, "alignment must be positive");
  check(total_bytes % alignment == 0, "aligned split requires aligned total bytes");
  const std::size_t first = align_down(total_bytes / 2ULL, alignment);
  return {ByteSpan{0, first}, ByteSpan{first, total_bytes - first}};
}

std::vector<ChunkSpan> split_rail_into_chunks_aligned(
    std::size_t rail_bytes,
    std::size_t alignment,
    std::size_t chunk_limit_bytes) {
  check(alignment > 0, "alignment must be positive");
  check(chunk_limit_bytes > 0, "chunk limit must be positive");
  check(rail_bytes % alignment == 0, "rail bytes must be aligned");
  std::vector<ChunkSpan> chunks;
  std::size_t offset = 0;
  while (offset < rail_bytes) {
    std::size_t bytes = std::min(chunk_limit_bytes, rail_bytes - offset);
    if (bytes < rail_bytes - offset) {
      bytes = align_down(bytes, alignment);
      check(bytes > 0, "pipeline chunk alignment is larger than chunk size");
    }
    chunks.push_back(ChunkSpan{offset, bytes});
    offset += bytes;
  }
  return chunks;
}

std::size_t slot_count_for_chunks(std::size_t chunk_count) {
  check(chunk_count > 0, "rail must have at least one pipeline chunk");
  return chunk_count < kPipelineSlots ? chunk_count : kPipelineSlots;
}

void start_receiver_listener(RdmaContext& ctx, const std::string& ip, int port) {
  ctx.ec = rdma_create_event_channel();
  check(ctx.ec != nullptr, "rdma_create_event_channel failed");
  check(rdma_create_id(ctx.ec, &ctx.listen_id, nullptr, RDMA_PS_TCP) == 0,
        "rdma_create_id listen failed");
  sockaddr_in listen_addr = make_sockaddr(ip, port);
  check(rdma_bind_addr(ctx.listen_id, reinterpret_cast<sockaddr*>(&listen_addr)) == 0,
        "rdma_bind_addr failed");
  check(rdma_listen(ctx.listen_id, 1) == 0, "rdma_listen failed");
}

RemoteInfo accept_receiver_rail(RdmaContext& ctx, std::size_t ring_bytes, int cq_depth,
                                int gid_index, std::size_t slots, std::size_t chunk_bytes) {
  rdma_cm_event* event = nullptr;
  wait_cm_event(ctx.ec, RDMA_CM_EVENT_CONNECT_REQUEST, &event);
  ctx.id = event->id;
  rdma_ack_cm_event(event);
  create_qp_and_memory(ctx, ring_bytes, cq_depth);

  RemoteInfo info{};
  info.addr = reinterpret_cast<std::uint64_t>(ctx.host_buf);
  info.rkey = ctx.mr->rkey;
  info.slots = static_cast<std::uint32_t>(slots);
  info.chunk_bytes = chunk_bytes;

  rdma_conn_param param{};
  param.private_data = &info;
  param.private_data_len = sizeof(info);
  param.responder_resources = 1;
  param.initiator_depth = 1;
  param.retry_count = 7;
  check(rdma_accept(ctx.id, &param) == 0, "rdma_accept failed");
  wait_cm_event(ctx.ec, RDMA_CM_EVENT_ESTABLISHED, &event);
  rdma_ack_cm_event(event);
  set_gid_index_if_requested(ctx.id, gid_index);
  return info;
}

RemoteInfo setup_sender_rail(RdmaContext& ctx, const std::string& ip, int port,
                             std::size_t ring_bytes, int cq_depth, int gid_index,
                             std::size_t slots, std::size_t chunk_bytes) {
  ctx.ec = rdma_create_event_channel();
  check(ctx.ec != nullptr, "rdma_create_event_channel failed");
  check(rdma_create_id(ctx.ec, &ctx.id, nullptr, RDMA_PS_TCP) == 0, "rdma_create_id client failed");
  sockaddr_in dst = make_sockaddr(ip, port);
  check(rdma_resolve_addr(ctx.id, nullptr, reinterpret_cast<sockaddr*>(&dst), 20000) == 0,
        "rdma_resolve_addr failed");
  rdma_cm_event* event = nullptr;
  wait_cm_event(ctx.ec, RDMA_CM_EVENT_ADDR_RESOLVED, &event);
  rdma_ack_cm_event(event);

  create_qp_and_memory(ctx, ring_bytes, cq_depth);
  check(rdma_resolve_route(ctx.id, 20000) == 0, "rdma_resolve_route failed");
  wait_cm_event(ctx.ec, RDMA_CM_EVENT_ROUTE_RESOLVED, &event);
  rdma_ack_cm_event(event);

  rdma_conn_param param{};
  param.responder_resources = 1;
  param.initiator_depth = 1;
  param.retry_count = 7;
  check(rdma_connect(ctx.id, &param) == 0, "rdma_connect failed");
  wait_cm_event(ctx.ec, RDMA_CM_EVENT_ESTABLISHED, &event);
  RemoteInfo remote{};
  check(event->param.conn.private_data_len >= sizeof(remote), "missing receiver memory info");
  std::memcpy(&remote, event->param.conn.private_data, sizeof(remote));
  rdma_ack_cm_event(event);
  set_gid_index_if_requested(ctx.id, gid_index);
  check(remote.magic == kMagic, "bad receiver memory info");
  check(static_cast<std::size_t>(remote.slots) == slots, "receiver slot count mismatch");
  check(remote.chunk_bytes == chunk_bytes, "receiver chunk size mismatch");
  return remote;
}

RemoteInfo setup_sender_rail_with_retry(RdmaContext& ctx, const std::string& ip, int port,
                                        std::size_t ring_bytes, int cq_depth, int gid_index,
                                        std::size_t slots, std::size_t chunk_bytes) {
  constexpr int kMaxAttempts = 100;
  std::string last_error;
  for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
    try {
      return setup_sender_rail(ctx, ip, port, ring_bytes, cq_depth, gid_index, slots, chunk_bytes);
    } catch (const std::exception& e) {
      last_error = e.what();
      cleanup(ctx);
      ctx = RdmaContext{};
      if (attempt == kMaxAttempts) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  die("RDMA sender failed after retries: " + last_error);
}

void validate_bf16_tensor(const torch::Tensor& tensor, const std::string& name) {
  check_tensor(tensor.is_cuda(), name + " must be on CUDA");
  check_tensor(tensor.scalar_type() == torch::kBFloat16, name + " must be bfloat16");
  check_tensor(tensor.is_contiguous(), name + " must be contiguous");
}

void validate_rdma_payload_tensor(const torch::Tensor& tensor, const std::string& name) {
  check_tensor(tensor.is_cuda(), name + " must be on CUDA");
  check_tensor(tensor.is_contiguous(), name + " must be contiguous");
  check_tensor(
      tensor.scalar_type() == torch::kBFloat16 || tensor.scalar_type() == torch::kUInt8,
      name + " must be bfloat16 or uint8");
}

std::size_t rdma_payload_bytes(const torch::Tensor& tensor) {
  validate_rdma_payload_tensor(tensor, "rdma payload");
  return static_cast<std::size_t>(tensor.numel()) *
      static_cast<std::size_t>(tensor.element_size());
}

int checked_rails(int rails) {
  check(rails == 1 || rails == 2, "RDMA rail count must be 1 or 2");
  return rails;
}

__nv_bfloat16* bf16_ptr(const torch::Tensor& tensor) {
  return reinterpret_cast<__nv_bfloat16*>(tensor.data_ptr<at::BFloat16>());
}

cudaStream_t current_stream_for(const torch::Tensor& tensor) {
  return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

void validate_u8_tensor(const torch::Tensor& tensor, const std::string& name) {
  check_tensor(tensor.is_cuda(), name + " must be on CUDA");
  check_tensor(tensor.scalar_type() == torch::kUInt8, name + " must be uint8");
  check_tensor(tensor.is_contiguous(), name + " must be contiguous");
}

void validate_nbits(int nbits) {
  check(nbits == 2 || nbits == 4 || nbits == 8, "nbits must be one of 2, 4, or 8");
}

int checked_mee_numel(const torch::Tensor& tensor, const std::string& name) {
  check_tensor(tensor.numel() > 0, name + " must be non-empty");
  check_tensor(tensor.numel() <= std::numeric_limits<int>::max(), name + " is too large for eden kernels");
  check_tensor(tensor.numel() % (kMeeChunkSize * 16) == 0,
               name + " numel must be divisible by 256 for hierarchical MEE");
  return static_cast<int>(tensor.numel());
}

__global__ void bf16_add_kernel(__nv_bfloat16* dst, const __nv_bfloat16* src, std::int64_t numel) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    const float sum = __bfloat162float(dst[idx]) + __bfloat162float(src[idx]);
    dst[idx] = __float2bfloat16(sum);
  }
}

}  // namespace

struct RdmaSender::Impl {
  std::vector<RdmaContext> contexts;
  std::vector<RemoteInfo> remotes;
  std::vector<std::vector<cudaStream_t>> streams;
  std::vector<ByteSpan> spans;
  std::vector<std::vector<ChunkSpan>> chunks;
  std::vector<std::vector<cudaEvent_t>> copy_done_events;
  std::vector<std::size_t> slot_counts;
  std::vector<std::size_t> slot_bytes;
  std::vector<std::size_t> ring_bytes;
  std::vector<std::deque<int>> credits;
  std::int64_t message_idx = 1;
  int gpu = 0;
  int rails = 1;
  std::size_t total_bytes = 0;
  std::size_t chunk_bytes = kPipelineChunkBytes;
  bool debug = false;

  Impl(std::string server_addr, std::string server_addr2, int port, int rails_in, int gpu_in,
       std::size_t total_bytes_in, int gid_index)
      : contexts(checked_rails(rails_in)),
        remotes(checked_rails(rails_in)),
        streams(checked_rails(rails_in)),
        spans(split_bytes(total_bytes_in, checked_rails(rails_in))),
        chunks(checked_rails(rails_in)),
        copy_done_events(checked_rails(rails_in)),
        slot_counts(checked_rails(rails_in)),
        slot_bytes(checked_rails(rails_in)),
        ring_bytes(checked_rails(rails_in)),
        credits(checked_rails(rails_in)),
        gpu(gpu_in),
        rails(rails_in),
        total_bytes(total_bytes_in),
        chunk_bytes(pipeline_chunk_bytes()),
        debug(rdma_debug_enabled()) {
    check(!server_addr.empty(), "first rail address must not be empty");
    if (rails == 2) {
      check(!server_addr2.empty(), "second rail address must not be empty with --rails 2");
    }

    check_cuda(cudaSetDevice(gpu), "cudaSetDevice failed");
    for (int rail = 0; rail < rails; ++rail) {
      chunks[rail] = split_rail_into_chunks(spans[rail].bytes);
      slot_counts[rail] = slot_count_for_chunks(chunks[rail].size());
      slot_bytes[rail] = kPipelineChunkBytes;
      ring_bytes[rail] = slot_counts[rail] * slot_bytes[rail];
      streams[rail].resize(slot_counts[rail]);
      copy_done_events[rail].resize(slot_counts[rail]);
      for (std::size_t slot = 0; slot < slot_counts[rail]; ++slot) {
        check_cuda(cudaStreamCreateWithFlags(&streams[rail][slot], cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags failed");
        check_cuda(cudaEventCreateWithFlags(&copy_done_events[rail][slot], cudaEventDisableTiming),
                   "cudaEventCreateWithFlags failed");
        credits[rail].push_back(static_cast<int>(slot));
      }
    }

    std::vector<std::string> addrs = {server_addr, server_addr2};
    for (int rail = 0; rail < rails; ++rail) {
      const int cq_depth = static_cast<int>(slot_counts[rail] * 8 + 128);
      remotes[rail] = setup_sender_rail_with_retry(
          contexts[rail], addrs[rail], port + rail * 10, ring_bytes[rail], cq_depth,
          gid_index, slot_counts[rail], slot_bytes[rail]);
      for (std::size_t slot = 0; slot < slot_counts[rail]; ++slot) {
        post_recv(contexts[rail].id, kSenderCreditRecv | static_cast<std::uint64_t>(slot));
      }
    }
    if (debug) {
      std::fprintf(stderr, "[rdma] sender ready port=%d rails=%d bytes=%zu chunks0=%zu slots0=%zu\n",
                   port, rails, total_bytes, chunks[0].size(), slot_counts[0]);
      std::fflush(stderr);
    }
  }

  ~Impl() {
    for (auto& rail_events : copy_done_events) {
      for (auto& event : rail_events) {
        if (event != nullptr) {
          cudaEventDestroy(event);
        }
      }
    }
    for (auto& rail_streams : streams) {
      for (auto& stream : rail_streams) {
        if (stream != nullptr) {
          cudaStreamDestroy(stream);
        }
      }
    }
    for (auto& ctx : contexts) {
      cleanup(ctx);
    }
  }

  void send(const torch::Tensor& src) {
    validate_rdma_payload_tensor(src, "src");
    check_tensor(src.get_device() == gpu, "src must live on the configured CUDA device");
    const std::size_t payload_bytes = rdma_payload_bytes(src);
    check_tensor(payload_bytes <= total_bytes, "src exceeds configured RDMA capacity");

    check_cuda(cudaSetDevice(gpu), "cudaSetDevice failed");
    const auto* src_ptr = reinterpret_cast<const std::uint8_t*>(src.data_ptr());
    const auto message_spans = split_bytes(payload_bytes, rails);
    std::vector<std::vector<ChunkSpan>> message_chunks(static_cast<std::size_t>(rails));
    for (int rail = 0; rail < rails; ++rail) {
      message_chunks[rail] = split_rail_into_chunks(message_spans[rail].bytes);
      check(message_chunks[rail].size() <= chunks[rail].size(),
            "runtime message has more chunks than the configured RDMA capacity");
    }

    std::vector<std::size_t> next_to_copy(static_cast<std::size_t>(rails), 0);
    std::vector<std::size_t> posted_writes(static_cast<std::size_t>(rails), 0);
    std::vector<std::size_t> completed_writes(static_cast<std::size_t>(rails), 0);
    std::vector<PendingChunk> pending;

    std::size_t total_chunks = 0;
    for (int rail = 0; rail < rails; ++rail) {
      total_chunks += message_chunks[rail].size();
    }
    std::size_t total_completed = 0;
    const double timeout_seconds = rdma_timeout_seconds();
    auto last_progress = std::chrono::steady_clock::now();

    if (debug) {
      std::fprintf(stderr, "[rdma] send begin bytes=%zu chunks=%zu credits0=%zu\n",
                   payload_bytes, total_chunks, credits[0].size());
      std::fflush(stderr);
    }

    auto poll_sender_rail = [&](int rail) -> bool {
      bool progressed = false;
      ibv_wc wc{};
      while (poll_one(contexts[rail].cq, &wc)) {
        progressed = true;
        const std::uint64_t kind = wc.wr_id & 0xF000000000000000ULL;
        if (kind == kSenderRdmaWr) {
          ++completed_writes[rail];
          ++total_completed;
          if (debug && (total_completed == 1 || total_completed == total_chunks)) {
            std::fprintf(stderr, "[rdma] send write completions %zu/%zu\n",
                         total_completed, total_chunks);
            std::fflush(stderr);
          }
        } else if (kind == kSenderCreditRecv) {
          const int slot = static_cast<int>(ntohl(wc.imm_data));
          credits[rail].push_back(slot);
          post_recv(contexts[rail].id, kSenderCreditRecv | static_cast<std::uint64_t>(slot));
        }
      }
      return progressed;
    };

    auto all_credits_returned = [&]() {
      for (int rail = 0; rail < rails; ++rail) {
        if (credits[rail].size() != slot_counts[rail]) {
          return false;
        }
      }
      return true;
    };

    while (total_completed < total_chunks || !pending.empty() || !all_credits_returned()) {
      bool progressed = false;
      for (int rail = 0; rail < rails; ++rail) {
        progressed = poll_sender_rail(rail) || progressed;
        while (!credits[rail].empty() && next_to_copy[rail] < message_chunks[rail].size()) {
          const int slot = credits[rail].front();
          credits[rail].pop_front();
          const std::size_t idx = next_to_copy[rail];
          const ChunkSpan chunk = message_chunks[rail][idx];
          check_cuda(cudaMemcpyAsync(contexts[rail].host_buf +
                                         static_cast<std::size_t>(slot) * slot_bytes[rail],
                                     src_ptr + message_spans[rail].offset + chunk.offset,
                                     chunk.bytes,
                                     cudaMemcpyDeviceToHost,
                                     streams[rail][slot]),
                     "cudaMemcpyAsync D2H failed");
          check_cuda(cudaEventRecord(copy_done_events[rail][slot], streams[rail][slot]),
                     "cudaEventRecord failed");
          pending.push_back(PendingChunk{rail, slot, idx});
          ++next_to_copy[rail];
          progressed = true;
        }
      }

      for (auto it = pending.begin(); it != pending.end();) {
        cudaError_t query = cudaEventQuery(copy_done_events[it->rail][it->slot]);
        if (query == cudaSuccess) {
          const ChunkSpan chunk = message_chunks[it->rail][it->chunk_idx];
          const std::uint64_t wr_payload =
              ((static_cast<std::uint64_t>(message_idx) & 0x0FFFFFFFULL) << 32) |
              static_cast<std::uint64_t>(it->chunk_idx);
          post_rdma_write_at(
              contexts[it->rail].id,
              contexts[it->rail].mr,
              remotes[it->rail],
              static_cast<std::size_t>(it->slot) * slot_bytes[it->rail],
              static_cast<std::size_t>(it->slot) * slot_bytes[it->rail],
              wr_payload,
              chunk.bytes,
              true,
              pack_chunk_imm(it->slot, it->chunk_idx));
          ++posted_writes[it->rail];
          it = pending.erase(it);
          progressed = true;
        } else if (query == cudaErrorNotReady) {
          ++it;
        } else {
          check_cuda(query, "cudaEventQuery failed");
        }
      }

      if (!progressed) {
        check_no_progress_timeout(
            last_progress,
            timeout_seconds,
            "sender completed=" + std::to_string(total_completed) +
                "/" + std::to_string(total_chunks) +
                " pending=" + std::to_string(pending.size()) +
                " credits0=" + std::to_string(credits[0].size()) +
                "/" + std::to_string(slot_counts[0]));
        std::this_thread::yield();
      } else {
        last_progress = std::chrono::steady_clock::now();
      }
    }

    ++message_idx;
    if (debug) {
      std::fprintf(stderr, "[rdma] send end chunks=%zu\n", total_completed);
      std::fflush(stderr);
    }
  }
};

struct RdmaReceiver::Impl {
  std::vector<RdmaContext> contexts;
  std::vector<std::vector<cudaStream_t>> streams;
  std::vector<ByteSpan> spans;
  std::vector<std::vector<ChunkSpan>> chunks;
  std::vector<std::vector<cudaEvent_t>> h2d_done_events;
  std::vector<std::size_t> slot_counts;
  std::vector<std::size_t> slot_bytes;
  std::vector<std::size_t> ring_bytes;
  int gpu = 0;
  int rails = 1;
  std::size_t total_bytes = 0;
  std::size_t chunk_bytes = kPipelineChunkBytes;
  bool debug = false;

  Impl(std::string server_addr, std::string server_addr2, int port, int rails_in, int gpu_in,
       std::size_t total_bytes_in, int gid_index)
      : contexts(checked_rails(rails_in)),
        streams(checked_rails(rails_in)),
        spans(split_bytes(total_bytes_in, checked_rails(rails_in))),
        chunks(checked_rails(rails_in)),
        h2d_done_events(checked_rails(rails_in)),
        slot_counts(checked_rails(rails_in)),
        slot_bytes(checked_rails(rails_in)),
        ring_bytes(checked_rails(rails_in)),
        gpu(gpu_in),
        rails(rails_in),
        total_bytes(total_bytes_in),
        chunk_bytes(pipeline_chunk_bytes()),
        debug(rdma_debug_enabled()) {
    check(!server_addr.empty(), "first rail address must not be empty");
    if (rails == 2) {
      check(!server_addr2.empty(), "second rail address must not be empty with --rails 2");
    }

    check_cuda(cudaSetDevice(gpu), "cudaSetDevice failed");
    for (int rail = 0; rail < rails; ++rail) {
      chunks[rail] = split_rail_into_chunks(spans[rail].bytes);
      slot_counts[rail] = slot_count_for_chunks(chunks[rail].size());
      slot_bytes[rail] = kPipelineChunkBytes;
      ring_bytes[rail] = slot_counts[rail] * slot_bytes[rail];
      streams[rail].resize(slot_counts[rail]);
      h2d_done_events[rail].resize(slot_counts[rail]);
      for (std::size_t slot = 0; slot < slot_counts[rail]; ++slot) {
        check_cuda(cudaStreamCreateWithFlags(&streams[rail][slot], cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags failed");
        check_cuda(cudaEventCreateWithFlags(&h2d_done_events[rail][slot], cudaEventDisableTiming),
                   "cudaEventCreateWithFlags failed");
      }
    }

    std::vector<std::string> addrs = {server_addr, server_addr2};
    for (int rail = 0; rail < rails; ++rail) {
      start_receiver_listener(contexts[rail], addrs[rail], port + rail * 10);
    }
    for (int rail = 0; rail < rails; ++rail) {
      const int cq_depth = static_cast<int>(slot_counts[rail] * 8 + 128);
      accept_receiver_rail(
          contexts[rail], ring_bytes[rail], cq_depth, gid_index, slot_counts[rail], slot_bytes[rail]);
      for (std::size_t slot = 0; slot < slot_counts[rail]; ++slot) {
        post_recv(contexts[rail].id, kReceiverNotifyRecv | static_cast<std::uint64_t>(slot));
      }
    }
    if (debug) {
      std::fprintf(stderr, "[rdma] receiver ready port=%d rails=%d bytes=%zu chunks0=%zu slots0=%zu\n",
                   port, rails, total_bytes, chunks[0].size(), slot_counts[0]);
      std::fflush(stderr);
    }
  }

  ~Impl() {
    for (auto& rail_events : h2d_done_events) {
      for (auto& event : rail_events) {
        if (event != nullptr) {
          cudaEventDestroy(event);
        }
      }
    }
    for (auto& rail_streams : streams) {
      for (auto& stream : rail_streams) {
        if (stream != nullptr) {
          cudaStreamDestroy(stream);
        }
      }
    }
    for (auto& ctx : contexts) {
      cleanup(ctx);
    }
  }

  void recv(const torch::Tensor& dst) {
    validate_rdma_payload_tensor(dst, "dst");
    check_tensor(dst.get_device() == gpu, "dst must live on the configured CUDA device");
    const std::size_t payload_bytes = rdma_payload_bytes(dst);
    check_tensor(payload_bytes <= total_bytes, "dst exceeds configured RDMA capacity");

    check_cuda(cudaSetDevice(gpu), "cudaSetDevice failed");
    auto* dst_ptr = reinterpret_cast<std::uint8_t*>(dst.data_ptr());
    const auto message_spans = split_bytes(payload_bytes, rails);
    std::vector<std::vector<ChunkSpan>> message_chunks(static_cast<std::size_t>(rails));
    for (int rail = 0; rail < rails; ++rail) {
      message_chunks[rail] = split_rail_into_chunks(message_spans[rail].bytes);
      check(message_chunks[rail].size() <= chunks[rail].size(),
            "runtime message has more chunks than the configured RDMA capacity");
    }

    std::vector<PendingChunk> h2d_pending;
    std::size_t total_chunks = 0;
    for (int rail = 0; rail < rails; ++rail) {
      total_chunks += message_chunks[rail].size();
    }
    std::size_t total_received = 0;
    std::size_t credit_sends_pending = 0;
    const double timeout_seconds = rdma_timeout_seconds();
    auto last_progress = std::chrono::steady_clock::now();

    if (debug) {
      std::fprintf(stderr, "[rdma] recv begin bytes=%zu chunks=%zu\n", payload_bytes, total_chunks);
      std::fflush(stderr);
    }

    while (total_received < total_chunks || !h2d_pending.empty() || credit_sends_pending > 0) {
      bool progressed = false;
      for (int rail = 0; rail < rails; ++rail) {
        ibv_wc wc{};
        while (poll_one(contexts[rail].cq, &wc)) {
          progressed = true;
          const std::uint64_t kind = wc.wr_id & 0xF000000000000000ULL;
          if (kind == kReceiverNotifyRecv) {
            const std::uint32_t imm = ntohl(wc.imm_data);
            const int slot = unpack_imm_slot(imm);
            const std::size_t idx = unpack_imm_chunk_idx(imm);
            check(idx < message_chunks[rail].size(), "received too many RDMA chunks");
            const ChunkSpan chunk = message_chunks[rail][idx];
            check_cuda(cudaMemcpyAsync(
                           dst_ptr + message_spans[rail].offset + chunk.offset,
                           contexts[rail].host_buf + static_cast<std::size_t>(slot) * slot_bytes[rail],
                           chunk.bytes,
                           cudaMemcpyHostToDevice,
                           streams[rail][slot]),
                       "cudaMemcpyAsync H2D failed");
            check_cuda(cudaEventRecord(h2d_done_events[rail][slot], streams[rail][slot]),
                       "cudaEventRecord failed");
            h2d_pending.push_back(PendingChunk{rail, slot, idx});
            ++total_received;
            if (debug && (total_received == 1 || total_received == total_chunks)) {
              std::fprintf(stderr, "[rdma] recv notifies %zu/%zu slot=%d idx=%zu\n",
                           total_received, total_chunks, slot, idx);
              std::fflush(stderr);
            }
            post_recv(contexts[rail].id, kReceiverNotifyRecv | static_cast<std::uint64_t>(slot));
          } else if (kind == kReceiverCreditSend) {
            check(credit_sends_pending > 0, "unexpected receiver credit send completion");
            --credit_sends_pending;
          }
        }
      }

      for (auto it = h2d_pending.begin(); it != h2d_pending.end();) {
        cudaError_t query = cudaEventQuery(h2d_done_events[it->rail][it->slot]);
        if (query == cudaSuccess) {
          post_credit_send(contexts[it->rail].id, it->slot);
          ++credit_sends_pending;
          it = h2d_pending.erase(it);
          progressed = true;
        } else if (query == cudaErrorNotReady) {
          ++it;
        } else {
          check_cuda(query, "cudaEventQuery failed");
        }
      }

      if (!progressed) {
        check_no_progress_timeout(
            last_progress,
            timeout_seconds,
            "receiver received=" + std::to_string(total_received) +
                "/" + std::to_string(total_chunks) +
                " h2d_pending=" + std::to_string(h2d_pending.size()) +
                " credit_sends_pending=" + std::to_string(credit_sends_pending));
        std::this_thread::yield();
      } else {
        last_progress = std::chrono::steady_clock::now();
      }
    }
    if (debug) {
      std::fprintf(stderr, "[rdma] recv end chunks=%zu\n", total_received);
      std::fflush(stderr);
    }
  }
};

struct PipelineRdmaSender::Impl {
  std::vector<RdmaContext> contexts;
  std::vector<RemoteInfo> remotes;
  std::vector<std::vector<cudaStream_t>> streams;
  std::vector<ByteSpan> capacity_spans;
  std::vector<std::vector<cudaEvent_t>> copy_done_events;
  std::vector<std::size_t> slot_counts;
  std::vector<std::size_t> slot_bytes;
  std::vector<std::size_t> ring_bytes;
  std::vector<std::deque<int>> credits;
  cudaEvent_t input_ready_event = nullptr;
  std::int64_t message_idx = 1;
  int gpu = 0;
  int rails = 1;
  std::size_t total_bytes = 0;
  std::size_t chunk_bytes = kPipelineChunkBytes;
  std::size_t max_inflight = 2;
  bool debug = false;

  Impl(std::string server_addr, std::string server_addr2, int port, int rails_in, int gpu_in,
       std::size_t total_bytes_in, int gid_index)
      : contexts(checked_rails(rails_in)),
        remotes(checked_rails(rails_in)),
        streams(checked_rails(rails_in)),
        capacity_spans(split_bytes(total_bytes_in, checked_rails(rails_in))),
        copy_done_events(checked_rails(rails_in)),
        slot_counts(checked_rails(rails_in)),
        slot_bytes(checked_rails(rails_in)),
        ring_bytes(checked_rails(rails_in)),
        credits(checked_rails(rails_in)),
        gpu(gpu_in),
        rails(rails_in),
        total_bytes(total_bytes_in),
        chunk_bytes(pipeline_chunk_bytes()),
        max_inflight(pipeline_window()),
        debug(rdma_debug_enabled()) {
    check(!server_addr.empty(), "first rail address must not be empty");
    if (rails == 2) {
      check(!server_addr2.empty(), "second rail address must not be empty with --rails 2");
    }

    check_cuda(cudaSetDevice(gpu), "cudaSetDevice failed");
    check_cuda(cudaEventCreateWithFlags(&input_ready_event, cudaEventDisableTiming),
               "cudaEventCreateWithFlags input_ready_event failed");
    for (int rail = 0; rail < rails; ++rail) {
      const auto capacity_chunks = split_rail_into_chunks_aligned(capacity_spans[rail].bytes, 1, chunk_bytes);
      slot_counts[rail] = slot_count_for_chunks(capacity_chunks.size());
      slot_bytes[rail] = chunk_bytes;
      ring_bytes[rail] = slot_counts[rail] * slot_bytes[rail];
      streams[rail].resize(slot_counts[rail]);
      copy_done_events[rail].resize(slot_counts[rail]);
      for (std::size_t slot = 0; slot < slot_counts[rail]; ++slot) {
        check_cuda(cudaStreamCreateWithFlags(&streams[rail][slot], cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags failed");
        check_cuda(cudaEventCreateWithFlags(&copy_done_events[rail][slot], cudaEventDisableTiming),
                   "cudaEventCreateWithFlags failed");
        credits[rail].push_back(static_cast<int>(slot));
      }
    }

    std::vector<std::string> addrs = {server_addr, server_addr2};
    for (int rail = 0; rail < rails; ++rail) {
      const int cq_depth = static_cast<int>(slot_counts[rail] * 8 + 128);
      remotes[rail] = setup_sender_rail_with_retry(
          contexts[rail], addrs[rail], port + rail * 10, ring_bytes[rail], cq_depth,
          gid_index, slot_counts[rail], slot_bytes[rail]);
      for (std::size_t slot = 0; slot < slot_counts[rail]; ++slot) {
        post_recv(contexts[rail].id, kSenderCreditRecv | static_cast<std::uint64_t>(slot));
      }
    }
    if (debug) {
      std::fprintf(stderr,
                   "[rdma] pipeline sender ready port=%d rails=%d capacity=%zu chunk_bytes=%zu inflight=%zu\n",
                   port, rails, total_bytes, chunk_bytes, max_inflight);
      std::fflush(stderr);
    }
  }

  ~Impl() {
    if (input_ready_event != nullptr) {
      cudaEventDestroy(input_ready_event);
    }
    for (auto& rail_events : copy_done_events) {
      for (auto& event : rail_events) {
        if (event != nullptr) {
          cudaEventDestroy(event);
        }
      }
    }
    for (auto& rail_streams : streams) {
      for (auto& stream : rail_streams) {
        if (stream != nullptr) {
          cudaStreamDestroy(stream);
        }
      }
    }
    for (auto& ctx : contexts) {
      cleanup(ctx);
    }
  }

  enum class SendMode {
    Raw,
    Compress,
    DecComp,
  };

  void send_raw(const torch::Tensor& src, int nbits, int chunk_size, int strategy) {
    validate_u8_tensor(src, "src");
    check_tensor(src.get_device() == gpu, "src must live on the configured CUDA device");
    send_impl(SendMode::Raw, torch::Tensor(), torch::Tensor(), src, torch::Tensor(), src.numel(),
              nbits, chunk_size, strategy);
  }

  void send_compress(
      const torch::Tensor& src,
      const torch::Tensor& dst,
      const torch::Tensor& rand_pool,
      int nbits,
      int chunk_size,
      int strategy) {
    validate_bf16_tensor(src, "src");
    validate_u8_tensor(dst, "dst");
    validate_bf16_tensor(rand_pool, "rand_pool");
    check_tensor(src.get_device() == gpu, "src must live on the configured CUDA device");
    check_tensor(dst.get_device() == gpu, "dst must live on the configured CUDA device");
    check_tensor(rand_pool.get_device() == gpu, "rand_pool must live on the configured CUDA device");
    check_tensor(rand_pool.numel() >= src.numel(), "rand_pool is too small");
    send_impl(SendMode::Compress, src, torch::Tensor(), dst, rand_pool, src.numel(),
              nbits, chunk_size, strategy);
  }

  void send_dec_comp(
      const torch::Tensor& recv,
      const torch::Tensor& inp,
      const torch::Tensor& send,
      const torch::Tensor& rand_pool,
      int nbits,
      int chunk_size,
      int strategy) {
    validate_u8_tensor(recv, "recv");
    validate_bf16_tensor(inp, "inp");
    validate_u8_tensor(send, "send");
    validate_bf16_tensor(rand_pool, "rand_pool");
    check_tensor(recv.get_device() == gpu, "recv must live on the configured CUDA device");
    check_tensor(inp.get_device() == gpu, "inp must live on the configured CUDA device");
    check_tensor(send.get_device() == gpu, "send must live on the configured CUDA device");
    check_tensor(rand_pool.get_device() == gpu, "rand_pool must live on the configured CUDA device");
    check_tensor(rand_pool.numel() >= inp.numel(), "rand_pool is too small");
    send_impl(SendMode::DecComp, inp, recv, send, rand_pool, inp.numel(),
              nbits, chunk_size, strategy);
  }

  void send_impl(
      SendMode mode,
      const torch::Tensor& input,
      const torch::Tensor& recv,
      const torch::Tensor& wire,
      const torch::Tensor& rand_pool,
      std::int64_t n,
      int nbits,
      int chunk_size,
      int strategy) {
    validate_pipeline_scaling(nbits, chunk_size, strategy);
    validate_u8_tensor(wire, "wire");
    check_tensor(wire.get_device() == gpu, "wire must live on the configured CUDA device");

    const std::size_t payload_bytes = rdma_payload_bytes(wire);
    check_tensor(payload_bytes <= total_bytes, "wire exceeds configured RDMA capacity");
    const std::size_t required_bytes =
        mode == SendMode::Raw ? payload_bytes : pipeline_required_bytes(n, nbits, chunk_size, strategy);
    check_tensor(payload_bytes >= required_bytes, "wire buffer is too small for compressed payload");

    const std::size_t record_bytes = pipeline_record_bytes(nbits, chunk_size, strategy);
    const std::size_t superchunk_elems = pipeline_superchunk_elems(chunk_size);
    check_tensor(payload_bytes % record_bytes == 0, "wire payload must be superchunk-record aligned");

    check_cuda(cudaSetDevice(gpu), "cudaSetDevice failed");
    check_cuda(cudaEventRecord(input_ready_event, current_stream_for(wire)),
               "cudaEventRecord input_ready_event failed");

    const auto message_spans = split_bytes_aligned(payload_bytes, rails, record_bytes);
    std::vector<std::vector<ChunkSpan>> message_chunks(static_cast<std::size_t>(rails));
    for (int rail = 0; rail < rails; ++rail) {
      message_chunks[rail] = split_rail_into_chunks_aligned(message_spans[rail].bytes, record_bytes, chunk_bytes);
      check(message_spans[rail].bytes <= total_bytes,
            "runtime message rail span exceeds configured RDMA capacity");
    }

    const auto* wire_ptr = reinterpret_cast<const std::uint8_t*>(wire.data_ptr());
    auto* send_ptr = const_cast<std::uint8_t*>(wire_ptr);
    auto* input_ptr = input.defined() ? bf16_ptr(input) : nullptr;
    auto* recv_ptr = recv.defined() ? reinterpret_cast<std::uint8_t*>(recv.data_ptr()) : nullptr;
    auto* rand_ptr = rand_pool.defined() ? bf16_ptr(rand_pool) : nullptr;

    std::vector<std::size_t> next_to_copy(static_cast<std::size_t>(rails), 0);
    std::vector<std::size_t> completed_writes(static_cast<std::size_t>(rails), 0);
    std::vector<std::size_t> active_slots(static_cast<std::size_t>(rails), 0);
    std::vector<PendingChunk> pending;

    std::size_t total_chunks = 0;
    for (int rail = 0; rail < rails; ++rail) {
      total_chunks += message_chunks[rail].size();
    }
    std::size_t total_completed = 0;
    const double timeout_seconds = rdma_timeout_seconds();
    auto last_progress = std::chrono::steady_clock::now();

    auto poll_sender_rail = [&](int rail) -> bool {
      bool progressed = false;
      ibv_wc wc{};
      while (poll_one(contexts[rail].cq, &wc)) {
        progressed = true;
        const std::uint64_t kind = wc.wr_id & 0xF000000000000000ULL;
        if (kind == kSenderRdmaWr) {
          ++completed_writes[rail];
          ++total_completed;
        } else if (kind == kSenderCreditRecv) {
          const int slot = static_cast<int>(ntohl(wc.imm_data));
          credits[rail].push_back(slot);
          check(active_slots[rail] > 0, "pipeline sender received unexpected credit");
          --active_slots[rail];
          post_recv(contexts[rail].id, kSenderCreditRecv | static_cast<std::uint64_t>(slot));
        }
      }
      return progressed;
    };

    auto all_credits_returned = [&]() {
      for (int rail = 0; rail < rails; ++rail) {
        if (credits[rail].size() != slot_counts[rail]) {
          return false;
        }
      }
      return true;
    };

    auto launch_producer = [&](int rail, int slot, std::size_t idx, const ChunkSpan& chunk) {
      const std::size_t global_offset = message_spans[rail].offset + chunk.offset;
      cudaStream_t stream = streams[rail][slot];
      check_cuda(cudaStreamWaitEvent(stream, input_ready_event, 0), "cudaStreamWaitEvent failed");
      if (mode != SendMode::Raw && global_offset < required_bytes) {
        const std::size_t compressed_bytes = std::min(chunk.bytes, required_bytes - global_offset);
        check(compressed_bytes % record_bytes == 0, "compressed tile must be record aligned");
        const std::size_t elem_offset = (global_offset / record_bytes) * superchunk_elems;
        check(elem_offset < static_cast<std::size_t>(n), "compressed tile maps past input");
        const std::size_t tile_elems = (compressed_bytes / record_bytes) * superchunk_elems;
        const std::size_t actual_elems =
            std::min(tile_elems, static_cast<std::size_t>(n) - elem_offset);
        if (mode == SendMode::Compress) {
          scaling_compress_with_cuda(
              input_ptr + elem_offset,
              send_ptr + global_offset,
              rand_ptr + elem_offset,
              static_cast<int>(actual_elems),
              nbits,
              chunk_size,
              gpu,
              strategy,
              stream);
        } else {
          scaling_dec_comp_with_cuda(
              recv_ptr + global_offset,
              input_ptr + elem_offset,
              send_ptr + global_offset,
              rand_ptr + elem_offset,
              static_cast<int>(actual_elems),
              nbits,
              chunk_size,
              gpu,
              strategy,
              stream);
        }
      }
      check_cuda(cudaMemcpyAsync(
                     contexts[rail].host_buf + static_cast<std::size_t>(slot) * slot_bytes[rail],
                     wire_ptr + global_offset,
                     chunk.bytes,
                     cudaMemcpyDeviceToHost,
                     stream),
                 "cudaMemcpyAsync pipeline D2H failed");
      check_cuda(cudaEventRecord(copy_done_events[rail][slot], stream),
                 "cudaEventRecord pipeline D2H failed");
    };

    while (total_completed < total_chunks || !pending.empty() || !all_credits_returned()) {
      bool progressed = false;
      for (int rail = 0; rail < rails; ++rail) {
        progressed = poll_sender_rail(rail) || progressed;
        while (!credits[rail].empty() &&
               active_slots[rail] < std::min(max_inflight, slot_counts[rail]) &&
               next_to_copy[rail] < message_chunks[rail].size()) {
          const int slot = credits[rail].front();
          credits[rail].pop_front();
          const std::size_t idx = next_to_copy[rail];
          const ChunkSpan chunk = message_chunks[rail][idx];
          launch_producer(rail, slot, idx, chunk);
          pending.push_back(PendingChunk{rail, slot, idx});
          ++next_to_copy[rail];
          ++active_slots[rail];
          progressed = true;
        }
      }

      for (auto it = pending.begin(); it != pending.end();) {
        cudaError_t query = cudaEventQuery(copy_done_events[it->rail][it->slot]);
        if (query == cudaSuccess) {
          const ChunkSpan chunk = message_chunks[it->rail][it->chunk_idx];
          const std::uint64_t wr_payload =
              ((static_cast<std::uint64_t>(message_idx) & 0x0FFFFFFFULL) << 32) |
              static_cast<std::uint64_t>(it->chunk_idx);
          post_rdma_write_at(
              contexts[it->rail].id,
              contexts[it->rail].mr,
              remotes[it->rail],
              static_cast<std::size_t>(it->slot) * slot_bytes[it->rail],
              static_cast<std::size_t>(it->slot) * slot_bytes[it->rail],
              wr_payload,
              chunk.bytes,
              true,
              pack_pipeline_imm(it->slot, it->chunk_idx, message_idx));
          it = pending.erase(it);
          progressed = true;
        } else if (query == cudaErrorNotReady) {
          ++it;
        } else {
          check_cuda(query, "cudaEventQuery pipeline D2H failed");
        }
      }

	      if (!progressed) {
	        check_no_progress_timeout(
	            last_progress,
	            timeout_seconds,
	            "pipeline sender completed=" + std::to_string(total_completed) +
	                "/" + std::to_string(total_chunks) +
	                " pending=" + std::to_string(pending.size()) +
	                " active0=" + std::to_string(active_slots[0]) +
	                " message_idx=" + std::to_string(message_idx));
        std::this_thread::yield();
      } else {
        last_progress = std::chrono::steady_clock::now();
      }
    }
    ++message_idx;
  }
};

struct PipelineRdmaReceiver::Impl {
  std::vector<RdmaContext> contexts;
  std::vector<std::vector<cudaStream_t>> streams;
  std::vector<ByteSpan> capacity_spans;
  std::vector<std::vector<cudaEvent_t>> h2d_done_events;
  std::vector<std::vector<cudaEvent_t>> decomp_done_events;
  std::vector<std::size_t> slot_counts;
  std::vector<std::size_t> slot_bytes;
  std::vector<std::size_t> ring_bytes;
  cudaEvent_t input_ready_event = nullptr;
  std::int64_t message_idx = 1;
  std::size_t credit_sends_pending = 0;
  int gpu = 0;
  int rails = 1;
  std::size_t total_bytes = 0;
  std::size_t chunk_bytes = kPipelineChunkBytes;
  bool debug = false;

  Impl(std::string server_addr, std::string server_addr2, int port, int rails_in, int gpu_in,
       std::size_t total_bytes_in, int gid_index)
      : contexts(checked_rails(rails_in)),
        streams(checked_rails(rails_in)),
        capacity_spans(split_bytes(total_bytes_in, checked_rails(rails_in))),
        h2d_done_events(checked_rails(rails_in)),
        decomp_done_events(checked_rails(rails_in)),
        slot_counts(checked_rails(rails_in)),
        slot_bytes(checked_rails(rails_in)),
        ring_bytes(checked_rails(rails_in)),
        gpu(gpu_in),
        rails(rails_in),
        total_bytes(total_bytes_in),
        chunk_bytes(pipeline_chunk_bytes()),
        debug(rdma_debug_enabled()) {
    check(!server_addr.empty(), "first rail address must not be empty");
    if (rails == 2) {
      check(!server_addr2.empty(), "second rail address must not be empty with --rails 2");
    }

    check_cuda(cudaSetDevice(gpu), "cudaSetDevice failed");
    check_cuda(cudaEventCreateWithFlags(&input_ready_event, cudaEventDisableTiming),
               "cudaEventCreateWithFlags receiver input_ready_event failed");
    for (int rail = 0; rail < rails; ++rail) {
      const auto capacity_chunks = split_rail_into_chunks_aligned(capacity_spans[rail].bytes, 1, chunk_bytes);
      slot_counts[rail] = slot_count_for_chunks(capacity_chunks.size());
      slot_bytes[rail] = chunk_bytes;
      ring_bytes[rail] = slot_counts[rail] * slot_bytes[rail];
      streams[rail].resize(slot_counts[rail]);
      h2d_done_events[rail].resize(slot_counts[rail]);
      decomp_done_events[rail].resize(slot_counts[rail]);
      for (std::size_t slot = 0; slot < slot_counts[rail]; ++slot) {
        check_cuda(cudaStreamCreateWithFlags(&streams[rail][slot], cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags failed");
        check_cuda(cudaEventCreateWithFlags(&h2d_done_events[rail][slot], cudaEventDisableTiming),
                   "cudaEventCreateWithFlags failed");
        check_cuda(cudaEventCreateWithFlags(&decomp_done_events[rail][slot], cudaEventDisableTiming),
                   "cudaEventCreateWithFlags failed");
      }
    }

    std::vector<std::string> addrs = {server_addr, server_addr2};
    for (int rail = 0; rail < rails; ++rail) {
      start_receiver_listener(contexts[rail], addrs[rail], port + rail * 10);
    }
    for (int rail = 0; rail < rails; ++rail) {
      const int cq_depth = static_cast<int>(slot_counts[rail] * 8 + 128);
      accept_receiver_rail(
          contexts[rail], ring_bytes[rail], cq_depth, gid_index, slot_counts[rail], slot_bytes[rail]);
      for (std::size_t slot = 0; slot < slot_counts[rail]; ++slot) {
        post_recv(contexts[rail].id, kReceiverNotifyRecv | static_cast<std::uint64_t>(slot));
      }
    }
    if (debug) {
      std::fprintf(stderr, "[rdma] pipeline receiver ready port=%d rails=%d capacity=%zu chunk_bytes=%zu\n",
                   port, rails, total_bytes, chunk_bytes);
      std::fflush(stderr);
    }
  }

  ~Impl() {
    if (input_ready_event != nullptr) {
      cudaEventDestroy(input_ready_event);
    }
    for (auto& rail_events : h2d_done_events) {
      for (auto& event : rail_events) {
        if (event != nullptr) {
          cudaEventDestroy(event);
        }
      }
    }
    for (auto& rail_events : decomp_done_events) {
      for (auto& event : rail_events) {
        if (event != nullptr) {
          cudaEventDestroy(event);
        }
      }
    }
    for (auto& rail_streams : streams) {
      for (auto& stream : rail_streams) {
        if (stream != nullptr) {
          cudaStreamDestroy(stream);
        }
      }
    }
    for (auto& ctx : contexts) {
      cleanup(ctx);
    }
  }

  enum class RecvMode {
    Raw,
    Decompress,
    DecComp,
  };

  static const char* recv_mode_name(RecvMode mode) {
    switch (mode) {
      case RecvMode::Raw:
        return "raw";
      case RecvMode::Decompress:
        return "decompress";
      case RecvMode::DecComp:
        return "dec_comp";
    }
    return "unknown";
  }

  void recv_raw(const torch::Tensor& dst, int nbits, int chunk_size, int strategy) {
    validate_u8_tensor(dst, "dst");
    check_tensor(dst.get_device() == gpu, "dst must live on the configured CUDA device");
    recv_impl(
        RecvMode::Raw,
        dst,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        nbits,
        chunk_size,
        strategy);
  }

  void recv_decompress(
      const torch::Tensor& src,
      const torch::Tensor& dst,
      int nbits,
      int chunk_size,
      int strategy) {
    validate_u8_tensor(src, "src");
    validate_bf16_tensor(dst, "dst");
    check_tensor(src.get_device() == gpu, "src must live on the configured CUDA device");
    check_tensor(dst.get_device() == gpu, "dst must live on the configured CUDA device");
    recv_impl(
        RecvMode::Decompress,
        src,
        dst,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        nbits,
        chunk_size,
        strategy);
  }

  void recv_dec_comp(
      const torch::Tensor& recv,
      const torch::Tensor& inp,
      const torch::Tensor& send,
      const torch::Tensor& rand_pool,
      int nbits,
      int chunk_size,
      int strategy) {
    validate_u8_tensor(recv, "recv");
    validate_bf16_tensor(inp, "inp");
    validate_u8_tensor(send, "send");
    validate_bf16_tensor(rand_pool, "rand_pool");
    check_tensor(recv.get_device() == gpu, "recv must live on the configured CUDA device");
    check_tensor(inp.get_device() == gpu, "inp must live on the configured CUDA device");
    check_tensor(send.get_device() == gpu, "send must live on the configured CUDA device");
    check_tensor(rand_pool.get_device() == gpu, "rand_pool must live on the configured CUDA device");
    check_tensor(rand_pool.numel() >= inp.numel(), "rand_pool is too small");
    recv_impl(
        RecvMode::DecComp,
        recv,
        torch::Tensor(),
        inp,
        send,
        rand_pool,
        nbits,
        chunk_size,
        strategy);
  }

  void recv_impl(
      RecvMode mode,
      const torch::Tensor& wire,
      const torch::Tensor& decomp_dst,
      const torch::Tensor& deccomp_input,
      const torch::Tensor& deccomp_send,
      const torch::Tensor& rand_pool,
      int nbits,
      int chunk_size,
      int strategy) {
    validate_pipeline_scaling(nbits, chunk_size, strategy);
    validate_u8_tensor(wire, "wire");
    check_tensor(wire.get_device() == gpu, "wire must live on the configured CUDA device");

    const bool do_decompress = mode == RecvMode::Decompress;
    const bool do_dec_comp = mode == RecvMode::DecComp;
    if (do_decompress) {
      validate_bf16_tensor(decomp_dst, "decomp_dst");
      check_tensor(decomp_dst.get_device() == gpu, "decomp_dst must live on the configured CUDA device");
    }
    if (do_dec_comp) {
      validate_bf16_tensor(deccomp_input, "deccomp_input");
      validate_u8_tensor(deccomp_send, "deccomp_send");
      validate_bf16_tensor(rand_pool, "rand_pool");
      check_tensor(deccomp_input.get_device() == gpu,
                   "deccomp_input must live on the configured CUDA device");
      check_tensor(deccomp_send.get_device() == gpu,
                   "deccomp_send must live on the configured CUDA device");
      check_tensor(rand_pool.get_device() == gpu, "rand_pool must live on the configured CUDA device");
      check_tensor(rand_pool.numel() >= deccomp_input.numel(), "rand_pool is too small");
    }

    const std::size_t payload_bytes = rdma_payload_bytes(wire);
    check_tensor(payload_bytes <= total_bytes, "wire exceeds configured RDMA capacity");
    const std::size_t record_bytes = pipeline_record_bytes(nbits, chunk_size, strategy);
    const std::size_t superchunk_elems = pipeline_superchunk_elems(chunk_size);
    const std::int64_t n = do_decompress ? decomp_dst.numel()
        : (do_dec_comp ? deccomp_input.numel() : wire.numel());
    const std::size_t required_bytes =
        (do_decompress || do_dec_comp) ? pipeline_required_bytes(n, nbits, chunk_size, strategy)
                                       : payload_bytes;
    check_tensor(payload_bytes >= required_bytes, "wire buffer is too small for pipeline receive");
    if (do_dec_comp) {
      check_tensor(rdma_payload_bytes(deccomp_send) >= required_bytes,
                   "deccomp_send buffer is too small");
    }
    check_tensor(payload_bytes % record_bytes == 0, "wire payload must be superchunk-record aligned");

    check_cuda(cudaSetDevice(gpu), "cudaSetDevice failed");
    if (do_dec_comp) {
      check_cuda(cudaEventRecord(input_ready_event, current_stream_for(deccomp_input)),
                 "cudaEventRecord receiver input_ready_event failed");
    }
    auto* wire_ptr = reinterpret_cast<std::uint8_t*>(wire.data_ptr());
    auto* dst_ptr = do_decompress ? bf16_ptr(decomp_dst) : nullptr;
    auto* input_ptr = do_dec_comp ? bf16_ptr(deccomp_input) : nullptr;
    auto* send_ptr = do_dec_comp ? reinterpret_cast<std::uint8_t*>(deccomp_send.data_ptr()) : nullptr;
    auto* rand_ptr = do_dec_comp ? bf16_ptr(rand_pool) : nullptr;
    const auto message_spans = split_bytes_aligned(payload_bytes, rails, record_bytes);
    std::vector<std::vector<ChunkSpan>> message_chunks(static_cast<std::size_t>(rails));
    for (int rail = 0; rail < rails; ++rail) {
      message_chunks[rail] = split_rail_into_chunks_aligned(message_spans[rail].bytes, record_bytes, chunk_bytes);
      check(message_spans[rail].bytes <= total_bytes,
            "runtime message rail span exceeds configured RDMA capacity");
    }

    std::vector<PendingChunk> h2d_pending;
    std::vector<PendingChunk> decomp_pending;
    std::size_t total_chunks = 0;
    for (int rail = 0; rail < rails; ++rail) {
      total_chunks += message_chunks[rail].size();
    }
    std::size_t total_received = 0;
    const std::uint32_t expected_seq = static_cast<std::uint32_t>(message_idx) & 0xFFU;
    const double timeout_seconds = rdma_timeout_seconds();
    auto last_progress = std::chrono::steady_clock::now();

    while (total_received < total_chunks ||
           !h2d_pending.empty() ||
           !decomp_pending.empty()) {
      bool progressed = false;
      for (int rail = 0; rail < rails; ++rail) {
        ibv_wc wc{};
        while (poll_one(contexts[rail].cq, &wc)) {
          progressed = true;
          const std::uint64_t kind = wc.wr_id & 0xF000000000000000ULL;
          if (kind == kReceiverNotifyRecv) {
            const std::uint32_t imm = ntohl(wc.imm_data);
            const std::uint32_t seq = unpack_pipeline_imm_seq(imm);
            check(
                seq == expected_seq,
                "pipeline receiver message sequence mismatch: expected=" +
                    std::to_string(expected_seq) +
                    " got=" + std::to_string(seq) +
                    " message_idx=" + std::to_string(message_idx) +
                    " mode=" + recv_mode_name(mode));
            const int slot = unpack_pipeline_imm_slot(imm);
            const std::size_t idx = unpack_pipeline_imm_chunk_idx(imm);
            check(idx < message_chunks[rail].size(), "received too many RDMA chunks");
            const ChunkSpan chunk = message_chunks[rail][idx];
            const std::size_t global_offset = message_spans[rail].offset + chunk.offset;
            check_cuda(cudaMemcpyAsync(
                           wire_ptr + global_offset,
                           contexts[rail].host_buf + static_cast<std::size_t>(slot) * slot_bytes[rail],
                           chunk.bytes,
                           cudaMemcpyHostToDevice,
                           streams[rail][slot]),
                       "cudaMemcpyAsync pipeline H2D failed");
            check_cuda(cudaEventRecord(h2d_done_events[rail][slot], streams[rail][slot]),
                       "cudaEventRecord pipeline H2D failed");
            h2d_pending.push_back(PendingChunk{rail, slot, idx});
            ++total_received;
            post_recv(contexts[rail].id, kReceiverNotifyRecv | static_cast<std::uint64_t>(slot));
          } else if (kind == kReceiverCreditSend) {
            check(credit_sends_pending > 0, "unexpected receiver credit send completion");
            --credit_sends_pending;
          }
        }
      }

      for (auto it = h2d_pending.begin(); it != h2d_pending.end();) {
        cudaError_t query = cudaEventQuery(h2d_done_events[it->rail][it->slot]);
        if (query == cudaSuccess) {
          const ChunkSpan chunk = message_chunks[it->rail][it->chunk_idx];
          const std::size_t global_offset = message_spans[it->rail].offset + chunk.offset;
          const bool transform_tile = (do_decompress || do_dec_comp) && global_offset < required_bytes;
          if (transform_tile) {
            const std::size_t compressed_bytes = std::min(chunk.bytes, required_bytes - global_offset);
            check(compressed_bytes % record_bytes == 0, "pipeline receive tile must be record aligned");
            const std::size_t elem_offset = (global_offset / record_bytes) * superchunk_elems;
            check(elem_offset < static_cast<std::size_t>(n), "pipeline receive tile maps past output");
            const std::size_t tile_elems = (compressed_bytes / record_bytes) * superchunk_elems;
            const std::size_t actual_elems =
                std::min(tile_elems, static_cast<std::size_t>(n) - elem_offset);
            if (do_decompress) {
              scaling_decompress_with_cuda(
                  wire_ptr + global_offset,
                  dst_ptr + elem_offset,
                  static_cast<int>(actual_elems),
                  nbits,
                  chunk_size,
                  gpu,
                  strategy,
                  streams[it->rail][it->slot],
                  0);
            } else {
              check_cuda(cudaStreamWaitEvent(streams[it->rail][it->slot], input_ready_event, 0),
                         "cudaStreamWaitEvent receiver input_ready_event failed");
              scaling_dec_comp_with_cuda(
                  wire_ptr + global_offset,
                  input_ptr + elem_offset,
                  send_ptr + global_offset,
                  rand_ptr + elem_offset,
                  static_cast<int>(actual_elems),
                  nbits,
                  chunk_size,
                  gpu,
                  strategy,
                  streams[it->rail][it->slot]);
            }
            check_cuda(cudaEventRecord(decomp_done_events[it->rail][it->slot],
                                       streams[it->rail][it->slot]),
                       "cudaEventRecord pipeline receive transform failed");
            decomp_pending.push_back(*it);
          } else {
            post_credit_send(contexts[it->rail].id, it->slot);
            ++credit_sends_pending;
          }
          it = h2d_pending.erase(it);
          progressed = true;
        } else if (query == cudaErrorNotReady) {
          ++it;
        } else {
          check_cuda(query, "cudaEventQuery pipeline H2D failed");
        }
      }

      for (auto it = decomp_pending.begin(); it != decomp_pending.end();) {
        cudaError_t query = cudaEventQuery(decomp_done_events[it->rail][it->slot]);
        if (query == cudaSuccess) {
          post_credit_send(contexts[it->rail].id, it->slot);
          ++credit_sends_pending;
          it = decomp_pending.erase(it);
          progressed = true;
        } else if (query == cudaErrorNotReady) {
          ++it;
        } else {
          check_cuda(query, "cudaEventQuery pipeline decompression failed");
        }
      }

	      if (!progressed) {
	        check_no_progress_timeout(
	            last_progress,
	            timeout_seconds,
            "pipeline receiver received=" + std::to_string(total_received) +
                "/" + std::to_string(total_chunks) +
                " h2d_pending=" + std::to_string(h2d_pending.size()) +
                " decomp_pending=" + std::to_string(decomp_pending.size()) +
                " credit_sends_pending=" + std::to_string(credit_sends_pending) +
                " message_idx=" + std::to_string(message_idx) +
                " mode=" + recv_mode_name(mode));
        std::this_thread::yield();
      } else {
        last_progress = std::chrono::steady_clock::now();
      }
    }
    ++message_idx;
  }
};

void bf16_add_(const torch::Tensor& dst, const torch::Tensor& src) {
  validate_bf16_tensor(dst, "dst");
  validate_bf16_tensor(src, "src");
  check_tensor(dst.get_device() == src.get_device(), "dst and src must be on the same CUDA device");
  check_tensor(dst.numel() == src.numel(), "dst and src must have the same number of elements");

  if (dst.numel() == 0) {
    return;
  }

  check_cuda(cudaSetDevice(dst.get_device()), "cudaSetDevice failed");
  const int threads = 256;
  const int blocks = static_cast<int>((dst.numel() + threads - 1) / threads);
  bf16_add_kernel<<<blocks, threads, 0, current_stream_for(dst)>>>(bf16_ptr(dst), bf16_ptr(src), dst.numel());
  check_cuda(cudaGetLastError(), "bf16_add_kernel launch failed");
}

std::int64_t mee_compressed_bytes(std::int64_t n, int nbits) {
  validate_nbits(nbits);
  check(n > 0, "n must be positive");
  check(n % (kMeeChunkSize * 16) == 0, "n must be divisible by 256 for hierarchical MEE");

  const std::int64_t chunks = n / kMeeChunkSize;
  const std::int64_t packed_bytes = (kMeeChunkSize * nbits) >> 3;
  return (chunks / 16) * (16 * (packed_bytes + 1) + 2);
}

void mee_compress_bf16(
    const torch::Tensor& src,
    const torch::Tensor& dst,
    const torch::Tensor& rand_pool,
    int nbits) {
  validate_nbits(nbits);
  validate_bf16_tensor(src, "src");
  validate_u8_tensor(dst, "dst");
  validate_bf16_tensor(rand_pool, "rand_pool");
  check_tensor(src.get_device() == dst.get_device(), "src and dst must be on the same CUDA device");
  check_tensor(src.get_device() == rand_pool.get_device(), "src and rand_pool must be on the same CUDA device");

  const int n = checked_mee_numel(src, "src");
  check_tensor(rand_pool.numel() >= src.numel(), "rand_pool is too small");
  check_tensor(dst.numel() >= mee_compressed_bytes(src.numel(), nbits), "dst is too small");

  check_cuda(cudaSetDevice(src.get_device()), "cudaSetDevice failed");
  scaling_compress_with_cuda(
      bf16_ptr(src),
      dst.data_ptr<std::uint8_t>(),
      bf16_ptr(rand_pool),
      n,
      nbits,
      kMeeChunkSize,
      src.get_device(),
      kMeeStrategy,
      current_stream_for(src));
}

void mee_decompress_bf16(const torch::Tensor& src, const torch::Tensor& dst, int nbits) {
  validate_nbits(nbits);
  validate_u8_tensor(src, "src");
  validate_bf16_tensor(dst, "dst");
  const int n = checked_mee_numel(dst, "dst");
  check_tensor(src.get_device() == dst.get_device(), "src and dst must be on the same CUDA device");
  check_tensor(src.numel() >= mee_compressed_bytes(dst.numel(), nbits), "src is too small");

  check_cuda(cudaSetDevice(dst.get_device()), "cudaSetDevice failed");
  scaling_decompress_with_cuda(
      src.data_ptr<std::uint8_t>(),
      bf16_ptr(dst),
      n,
      nbits,
      kMeeChunkSize,
      dst.get_device(),
      kMeeStrategy,
      current_stream_for(dst),
      0);
}

void mee_decompress_add_bf16(const torch::Tensor& src, const torch::Tensor& dst, int nbits) {
  validate_nbits(nbits);
  validate_u8_tensor(src, "src");
  validate_bf16_tensor(dst, "dst");
  const int n = checked_mee_numel(dst, "dst");
  check_tensor(src.get_device() == dst.get_device(), "src and dst must be on the same CUDA device");
  check_tensor(src.numel() >= mee_compressed_bytes(dst.numel(), nbits), "src is too small");

  check_cuda(cudaSetDevice(dst.get_device()), "cudaSetDevice failed");
  scaling_decompress_with_cuda(
      src.data_ptr<std::uint8_t>(),
      bf16_ptr(dst),
      n,
      nbits,
      kMeeChunkSize,
      dst.get_device(),
      kMeeStrategy,
      current_stream_for(dst),
      1);
}

void mee_dec_comp_bf16(
    const torch::Tensor& recv,
    const torch::Tensor& inp,
    const torch::Tensor& send,
    const torch::Tensor& rand_pool,
    int nbits) {
  validate_nbits(nbits);
  validate_u8_tensor(recv, "recv");
  validate_bf16_tensor(inp, "inp");
  validate_u8_tensor(send, "send");
  validate_bf16_tensor(rand_pool, "rand_pool");
  const int n = checked_mee_numel(inp, "inp");
  const auto required = mee_compressed_bytes(inp.numel(), nbits);
  check_tensor(recv.get_device() == inp.get_device(), "recv and inp must be on the same CUDA device");
  check_tensor(send.get_device() == inp.get_device(), "send and inp must be on the same CUDA device");
  check_tensor(rand_pool.get_device() == inp.get_device(), "rand_pool and inp must be on the same CUDA device");
  check_tensor(recv.numel() >= required, "recv is too small");
  check_tensor(send.numel() >= required, "send is too small");
  check_tensor(rand_pool.numel() >= inp.numel(), "rand_pool is too small");

  check_cuda(cudaSetDevice(inp.get_device()), "cudaSetDevice failed");
  scaling_dec_comp_with_cuda(
      recv.data_ptr<std::uint8_t>(),
      bf16_ptr(inp),
      send.data_ptr<std::uint8_t>(),
      bf16_ptr(rand_pool),
      n,
      nbits,
      kMeeChunkSize,
      inp.get_device(),
      kMeeStrategy,
      current_stream_for(inp));
}

RdmaSender::RdmaSender(std::string server_addr, std::string server_addr2, int port, int rails,
                       int gpu, std::size_t total_bytes, int gid_index)
    : impl_(std::make_unique<Impl>(
          std::move(server_addr), std::move(server_addr2), port, rails, gpu, total_bytes, gid_index)) {}

RdmaSender::~RdmaSender() = default;

void RdmaSender::send(const torch::Tensor& src) {
  impl_->send(src);
}

RdmaReceiver::RdmaReceiver(std::string server_addr, std::string server_addr2, int port, int rails,
                           int gpu, std::size_t total_bytes, int gid_index)
    : impl_(std::make_unique<Impl>(
          std::move(server_addr), std::move(server_addr2), port, rails, gpu, total_bytes, gid_index)) {}

RdmaReceiver::~RdmaReceiver() = default;

void RdmaReceiver::recv(const torch::Tensor& dst) {
  impl_->recv(dst);
}

PipelineRdmaSender::PipelineRdmaSender(std::string server_addr, std::string server_addr2, int port,
                                       int rails, int gpu, std::size_t total_bytes, int gid_index)
    : impl_(std::make_unique<Impl>(
          std::move(server_addr), std::move(server_addr2), port, rails, gpu, total_bytes, gid_index)) {}

PipelineRdmaSender::~PipelineRdmaSender() = default;

void PipelineRdmaSender::send(const torch::Tensor& src, int nbits, int chunk_size, int strategy) {
  impl_->send_raw(src, nbits, chunk_size, strategy);
}

void PipelineRdmaSender::send_compress_bf16(
    const torch::Tensor& src,
    const torch::Tensor& dst,
    const torch::Tensor& rand_pool,
    int nbits,
    int chunk_size,
    int strategy) {
  impl_->send_compress(src, dst, rand_pool, nbits, chunk_size, strategy);
}

void PipelineRdmaSender::send_dec_comp_bf16(
    const torch::Tensor& recv,
    const torch::Tensor& inp,
    const torch::Tensor& send,
    const torch::Tensor& rand_pool,
    int nbits,
    int chunk_size,
    int strategy) {
  impl_->send_dec_comp(recv, inp, send, rand_pool, nbits, chunk_size, strategy);
}

PipelineRdmaReceiver::PipelineRdmaReceiver(std::string server_addr, std::string server_addr2, int port,
                                           int rails, int gpu, std::size_t total_bytes, int gid_index)
    : impl_(std::make_unique<Impl>(
          std::move(server_addr), std::move(server_addr2), port, rails, gpu, total_bytes, gid_index)) {}

PipelineRdmaReceiver::~PipelineRdmaReceiver() = default;

void PipelineRdmaReceiver::recv(const torch::Tensor& dst, int nbits, int chunk_size, int strategy) {
  impl_->recv_raw(dst, nbits, chunk_size, strategy);
}

void PipelineRdmaReceiver::recv_decompress_bf16(
    const torch::Tensor& src,
    const torch::Tensor& dst,
    int nbits,
    int chunk_size,
    int strategy) {
  impl_->recv_decompress(src, dst, nbits, chunk_size, strategy);
}

void PipelineRdmaReceiver::recv_dec_comp_bf16(
    const torch::Tensor& recv,
    const torch::Tensor& inp,
    const torch::Tensor& send,
    const torch::Tensor& rand_pool,
    int nbits,
    int chunk_size,
    int strategy) {
  impl_->recv_dec_comp(recv, inp, send, rand_pool, nbits, chunk_size, strategy);
}

}  // namespace ring_allreduce
