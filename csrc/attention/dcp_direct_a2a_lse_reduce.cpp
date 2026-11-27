// Direct symmetric-memory DCP A2A + LSE reduce for XPU.
// Mirrors vLLM's CUDA op `_C.direct_dcp_a2a_lse_reduce`: each rank writes the
// partial output/LSE slices straight into peer buffers, publishes an epoch
// signal with release semantics, then waits on all peers and combines.
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <sycl/sycl.hpp>

#include "sycl_launch.h"
#include "utils.h"

namespace vllm {
namespace direct_dcp {

constexpr uint64_t kSpinLimit = 100000000;
// Peers write neighbouring signal slots and LSE/output partitions across
// devices; sharing a cache line between writers loses stores on XPU.
constexpr int64_t kCacheLineBytes = 64;
constexpr float kLog2E = 1.4426950408889634f;

using signal_ref_t = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;

template <typename T>
inline T* get_peer_ptr(const uint64_t* table, int64_t rank) {
  return reinterpret_cast<T*>(table[rank]);
}

inline void store_release_system(uint32_t* ptr, uint32_t value) {
  signal_ref_t ref(*ptr);
  ref.store(value, sycl::memory_order::release, sycl::memory_scope::system);
}

inline uint32_t load_acquire_system(const uint32_t* ptr) {
  signal_ref_t ref(*const_cast<uint32_t*>(ptr));
  return ref.load(sycl::memory_order::acquire, sycl::memory_scope::system);
}

inline bool wait_for_epoch(const uint32_t* ptr, uint32_t epoch) {
  for (uint64_t spins = 0; spins < kSpinLimit; ++spins) {
    // Wrap-safe >=: a peer that already moved on must not strand the waiter.
    if (static_cast<int32_t>(load_acquire_system(ptr) - epoch) >= 0) {
      return true;
    }
  }
  return false;
}

inline int64_t find_sequence(
    const int32_t* query_start_loc, int64_t token_idx, int64_t num_seqs) {
  int64_t left = 0;
  int64_t right = num_seqs;
  while (left < right) {
    const int64_t mid = (left + right) / 2;
    if (query_start_loc[mid] <= token_idx) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

struct IncrementEpoch {
  int64_t* epoch;
  void operator()(sycl::nd_item<1>) const { epoch[0] += 1; }
};

template <typename lse_t>
class DispatchOutputLse : public xpu::SyclKernelConfigConvention {
 public:
  DispatchOutputLse(
      const sycl::uint4* partial_output,
      const lse_t* partial_lse,
      const int32_t* seq_lens,
      const int32_t* query_start_loc,
      const uint64_t* peer_output_ptrs,
      const uint64_t* peer_lse_ptrs,
      const int64_t* epoch_ptr,
      int64_t world_size,
      int64_t rank,
      int64_t num_tokens,
      int64_t max_num_tokens,
      int64_t num_seqs,
      int64_t heads_per_rank,
      int64_t head_dim,
      int64_t output_token_stride,
      int64_t lse_token_stride,
      int64_t lse_head_stride)
      : partial_output_(partial_output),
        partial_lse_(partial_lse),
        seq_lens_(seq_lens),
        query_start_loc_(query_start_loc),
        peer_output_ptrs_(peer_output_ptrs),
        peer_lse_ptrs_(peer_lse_ptrs),
        epoch_ptr_(epoch_ptr),
        world_size_(world_size),
        rank_(rank),
        num_tokens_(num_tokens),
        max_num_tokens_(max_num_tokens),
        num_seqs_(num_seqs),
        heads_per_rank_(heads_per_rank),
        head_dim_(head_dim),
        output_token_stride_(output_token_stride),
        lse_token_stride_(lse_token_stride),
        lse_head_stride_(lse_head_stride) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    empty_kv_ = sycl::local_accessor<int32_t, 1>(1, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t tid = item.get_local_id(0);
    const int64_t block_size = item.get_local_range(0);
    const int64_t work_item = item.get_group(0);
    // Work-groups roughly start in id order; rotate destinations per rank so
    // peers do not all write the same device at once (PCIe incast).
    const int64_t step = work_item / num_tokens_;
    const int64_t token_idx = work_item - step * num_tokens_;
    const int64_t destination_rank = (rank_ + 1 + step) % world_size_;

    if (tid == 0) {
      int32_t empty = 0;
      if (seq_lens_ != nullptr) {
        const int64_t seq_idx =
            find_sequence(query_start_loc_, token_idx, num_seqs_);
        empty = seq_lens_[seq_idx] == 0 ? 1 : 0;
      }
      empty_kv_[0] = empty;
    }
    sycl::group_barrier(item.get_group());
    const bool empty_kv = empty_kv_[0] != 0;

    const uint32_t epoch = static_cast<uint32_t>(epoch_ptr_[0]);
    const int64_t parity = static_cast<int64_t>(epoch & 1u);
    const int64_t destination_item =
        ((parity * world_size_ + rank_) * max_num_tokens_ + token_idx) *
        heads_per_rank_;
    const int64_t source_head = destination_rank * heads_per_rank_;

    if (!empty_kv) {
      auto* peer_output =
          get_peer_ptr<sycl::uint4>(peer_output_ptrs_, destination_rank);
      const int64_t vectors_per_item = heads_per_rank_ * head_dim_ / 8;
      const int64_t source_vector =
          (token_idx * output_token_stride_ + source_head * head_dim_) / 8;
      const int64_t destination_vector = destination_item * head_dim_ / 8;
      for (int64_t v = tid; v < vectors_per_item; v += block_size) {
        peer_output[destination_vector + v] =
            partial_output_[source_vector + v];
      }
    }

    float* peer_lse = get_peer_ptr<float>(peer_lse_ptrs_, destination_rank);
    const int64_t source_lse =
        token_idx * lse_token_stride_ + source_head * lse_head_stride_;
    for (int64_t h = tid; h < heads_per_rank_; h += block_size) {
      peer_lse[destination_item + h] =
          empty_kv ? -std::numeric_limits<float>::infinity()
                   : static_cast<float>(
                         partial_lse_[source_lse + h * lse_head_stride_]);
    }
  }

 private:
  const sycl::uint4* partial_output_;
  const lse_t* partial_lse_;
  const int32_t* seq_lens_;
  const int32_t* query_start_loc_;
  const uint64_t* peer_output_ptrs_;
  const uint64_t* peer_lse_ptrs_;
  const int64_t* epoch_ptr_;
  int64_t world_size_, rank_, num_tokens_, max_num_tokens_, num_seqs_;
  int64_t heads_per_rank_, head_dim_;
  int64_t output_token_stride_, lse_token_stride_, lse_head_stride_;
  sycl::local_accessor<int32_t, 1> empty_kv_;
};

struct SignalPeers {
  const uint64_t* peer_signal_ptrs;
  const int64_t* epoch_ptr;
  int64_t world_size;
  int64_t rank;
  int64_t signal_stride;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t destination_rank = item.get_global_id(0);
    if (destination_rank >= world_size) {
      return;
    }
    const uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
    const int64_t parity = static_cast<int64_t>(epoch & 1u);
    auto* peer_signal =
        get_peer_ptr<uint32_t>(peer_signal_ptrs, destination_rank);
    store_release_system(
        peer_signal + (parity * world_size + rank) * signal_stride, epoch);
  }
};

template <typename scalar_t>
class WaitLseCombine : public xpu::SyclKernelConfigConvention {
 public:
  WaitLseCombine(
      const scalar_t* received_output,
      const float* received_lse,
      const uint32_t* received_signal,
      const int64_t* epoch_ptr,
      scalar_t* combined_output,
      int64_t world_size,
      int64_t max_num_tokens,
      int64_t heads_per_rank,
      int64_t head_dim,
      bool is_lse_base_on_e,
      int64_t signal_stride,
      int64_t heads_per_group)
      : received_output_(received_output),
        received_lse_(received_lse),
        received_signal_(received_signal),
        epoch_ptr_(epoch_ptr),
        combined_output_(combined_output),
        world_size_(world_size),
        max_num_tokens_(max_num_tokens),
        heads_per_rank_(heads_per_rank),
        head_dim_(head_dim),
        is_lse_base_on_e_(is_lse_base_on_e),
        signal_stride_(signal_stride),
        heads_per_group_(heads_per_group) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    weights_ =
        sycl::local_accessor<float, 1>(world_size_ * heads_per_group_, cgh);
    timed_out_ = sycl::local_accessor<int32_t, 1>(1, cgh);
  }

  // Each work-group covers heads_per_group heads of one token, sized so the
  // grid stays small: thousands of waiters on the same signal words serialize.
  void operator()(sycl::nd_item<1> item) const {
    const int64_t tid = item.get_local_id(0);
    const int64_t block_size = item.get_local_range(0);
    const int64_t groups_per_token =
        (heads_per_rank_ + heads_per_group_ - 1) / heads_per_group_;
    const int64_t token_idx = item.get_group(0) / groups_per_token;
    const int64_t head_begin =
        (item.get_group(0) - token_idx * groups_per_token) * heads_per_group_;
    const int64_t head_count =
        sycl::min(heads_per_group_, heads_per_rank_ - head_begin);
    const uint32_t epoch = static_cast<uint32_t>(epoch_ptr_[0]);
    const int64_t parity = static_cast<int64_t>(epoch & 1u);
    constexpr float inf = std::numeric_limits<float>::infinity();

    if (tid == 0) {
      int32_t timed_out = 0;
      for (int64_t src = 0; src < world_size_; ++src) {
        const uint32_t* signal =
            received_signal_ + (parity * world_size_ + src) * signal_stride_;
        if (!wait_for_epoch(signal, epoch)) {
          sycl::ext::oneapi::experimental::printf(
              "direct DCP A2A timeout source=%ld token=%ld epoch=%u\n",
              static_cast<long>(src),
              static_cast<long>(token_idx),
              epoch);
          timed_out = 1;
          break;
        }
      }
      timed_out_[0] = timed_out;
    }
    sycl::group_barrier(item.get_group());
    // Only tid 0 acquired the signals; make peer data visible to every thread.
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    const bool timed_out = timed_out_[0] != 0;

    for (int64_t local_head = tid; local_head < head_count;
         local_head += block_size) {
      const int64_t head = head_begin + local_head;
      float* weights = &weights_[local_head * world_size_];
      if (timed_out) {
        // Surface the failure as NaNs instead of hanging the device.
        for (int64_t src = 0; src < world_size_; ++src) {
          weights[src] = std::numeric_limits<float>::quiet_NaN();
        }
        continue;
      }
      float lse_max = -inf;
      for (int64_t src = 0; src < world_size_; ++src) {
        float value = received_lse_
            [((parity * world_size_ + src) * max_num_tokens_ + token_idx) *
                 heads_per_rank_ +
             head];
        if (sycl::isnan(value) || value == inf) {
          value = -inf;
        }
        if (is_lse_base_on_e_) {
          value *= kLog2E;
        }
        weights[src] = value;
        lse_max = sycl::fmax(lse_max, value);
      }
      if (lse_max == -inf) {
        lse_max = 0.0f;
      }
      float lse_sum = 0.0f;
      for (int64_t src = 0; src < world_size_; ++src) {
        lse_sum += sycl::exp2(weights[src] - lse_max);
      }
      const float inverse_lse_sum = lse_sum > 0.0f ? 1.0f / lse_sum : 0.0f;
      for (int64_t src = 0; src < world_size_; ++src) {
        weights[src] = sycl::exp2(weights[src] - lse_max) * inverse_lse_sum;
      }
    }
    sycl::group_barrier(item.get_group());

    // head_dim % 8 == 0, so each 16-byte vector stays within one head.
    constexpr int64_t kVec = sizeof(sycl::uint4) / sizeof(scalar_t);
    const int64_t vectors_per_head = head_dim_ / kVec;
    const int64_t vectors_per_token = heads_per_rank_ * vectors_per_head;
    const int64_t vector_begin = head_begin * vectors_per_head;
    const auto* received =
        reinterpret_cast<const sycl::uint4*>(received_output_);
    auto* combined = reinterpret_cast<sycl::uint4*>(combined_output_);
    for (int64_t local_vector = tid;
         local_vector < head_count * vectors_per_head;
         local_vector += block_size) {
      const int64_t v = vector_begin + local_vector;
      const float* weights =
          &weights_[(local_vector / vectors_per_head) * world_size_];
      float accumulator[kVec] = {};
      for (int64_t src = 0; src < world_size_; ++src) {
        const float weight = weights[src];
        // Empty-KV peers skip their output write, so never read it.
        if (weight == 0.0f) {
          continue;
        }
        const int64_t source_vector =
            ((parity * world_size_ + src) * max_num_tokens_ + token_idx) *
                vectors_per_token +
            v;
        const sycl::uint4 packed = received[source_vector];
        const auto* values = reinterpret_cast<const scalar_t*>(&packed);
#pragma unroll
        for (int64_t i = 0; i < kVec; ++i) {
          accumulator[i] += static_cast<float>(values[i]) * weight;
        }
      }
      sycl::uint4 packed;
      auto* values = reinterpret_cast<scalar_t*>(&packed);
#pragma unroll
      for (int64_t i = 0; i < kVec; ++i) {
        values[i] = static_cast<scalar_t>(accumulator[i]);
      }
      combined[token_idx * vectors_per_token + v] = packed;
    }
  }

 private:
  const scalar_t* received_output_;
  const float* received_lse_;
  const uint32_t* received_signal_;
  const int64_t* epoch_ptr_;
  scalar_t* combined_output_;
  int64_t world_size_, max_num_tokens_, heads_per_rank_, head_dim_;
  bool is_lse_base_on_e_;
  int64_t signal_stride_;
  int64_t heads_per_group_;
  sycl::local_accessor<float, 1> weights_;
  sycl::local_accessor<int32_t, 1> timed_out_;
};

inline const uint64_t* ptr_table(
    const torch::Tensor& t,
    const char* name,
    const at::Device& device,
    int64_t world_size) {
  TORCH_CHECK(
      (t.scalar_type() == at::kLong || t.scalar_type() == at::kUInt64) &&
          t.is_contiguous() && t.device() == device && t.numel() >= world_size,
      name,
      " must be a contiguous int64/uint64 tensor with world_size entries on ",
      "the input device");
  return reinterpret_cast<const uint64_t*>(t.data_ptr());
}

}  // namespace direct_dcp
}  // namespace vllm

void direct_dcp_a2a_lse_reduce(
    torch::Tensor const& partial_output,
    torch::Tensor const& partial_lse,
    std::optional<torch::Tensor> const& seq_lens,
    std::optional<torch::Tensor> const& query_start_loc,
    torch::Tensor const& peer_output_ptrs,
    torch::Tensor const& peer_lse_ptrs,
    torch::Tensor const& peer_signal_ptrs,
    torch::Tensor& received_output,
    torch::Tensor& received_lse,
    torch::Tensor& received_signal,
    torch::Tensor& epoch,
    torch::Tensor& combined_output,
    int64_t world_size,
    int64_t rank,
    int64_t max_num_tokens,
    bool is_lse_base_on_e) {
  using namespace vllm::direct_dcp;
  const at::DeviceGuard device_guard(partial_output.device());
  const auto device = partial_output.device();
  TORCH_CHECK(
      partial_output.is_xpu() && partial_lse.device() == device,
      "partial output and LSE must be on the same XPU device");
  const auto output_dtype = partial_output.scalar_type();
  TORCH_CHECK(
      output_dtype == at::kHalf || output_dtype == at::kBFloat16,
      "direct DCP A2A only supports FP16 and BF16 output");
  const auto lse_dtype = partial_lse.scalar_type();
  TORCH_CHECK(
      lse_dtype == at::kFloat || lse_dtype == at::kHalf ||
          lse_dtype == at::kBFloat16,
      "partial LSE must be FP32, FP16, or BF16");
  TORCH_CHECK(
      partial_output.dim() == 3 && partial_lse.dim() == 2,
      "expected output [T,H,D] and LSE [T,H]");
  TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");

  const int64_t num_tokens = partial_output.size(0);
  const int64_t total_heads = partial_output.size(1);
  const int64_t head_dim = partial_output.size(2);
  const int64_t output_token_stride = partial_output.stride(0);
  const int64_t lse_token_stride = partial_lse.stride(0);
  const int64_t lse_head_stride = partial_lse.stride(1);
  TORCH_CHECK(
      partial_output.stride(2) == 1 && partial_output.stride(1) == head_dim &&
          output_token_stride >= total_heads * head_dim &&
          output_token_stride % 8 == 0,
      "partial output must have packed heads and an aligned token stride");
  TORCH_CHECK(
      num_tokens > 0 && num_tokens <= max_num_tokens,
      "token count exceeds symmetric buffer capacity");
  TORCH_CHECK(
      total_heads % world_size == 0,
      "attention heads must divide evenly across DCP ranks");
  TORCH_CHECK(
      partial_lse.size(0) == num_tokens && partial_lse.size(1) == total_heads,
      "LSE shape must match attention output");
  TORCH_CHECK(
      head_dim % 8 == 0, "head_dim must be divisible by 8 for 16-byte stores");
  const int64_t heads_per_rank = total_heads / world_size;
  TORCH_CHECK(
      combined_output.scalar_type() == output_dtype &&
          combined_output.is_contiguous() &&
          combined_output.device() == device &&
          reinterpret_cast<uintptr_t>(combined_output.data_ptr()) % 16 == 0,
      "combined output must be a contiguous, 16-byte aligned tensor on the "
      "input device with the input dtype");
  TORCH_CHECK(
      combined_output.size(0) == num_tokens &&
          combined_output.size(1) == heads_per_rank &&
          combined_output.size(2) == head_dim,
      "combined output has the wrong shape");
  TORCH_CHECK(
      received_output.scalar_type() == output_dtype &&
          received_lse.scalar_type() == at::kFloat &&
          received_signal.scalar_type() == at::kInt,
      "invalid symmetric staging dtypes");
  const int64_t staging_entries = 2 * world_size * max_num_tokens;
  TORCH_CHECK(
      received_output.is_contiguous() && received_lse.is_contiguous() &&
          received_signal.is_contiguous() &&
          received_output.device() == device &&
          received_lse.device() == device && received_signal.device() == device,
      "symmetric staging buffers must be contiguous and on the input device");
  TORCH_CHECK(
      received_output.numel() >= staging_entries * heads_per_rank * head_dim &&
          received_lse.numel() >= staging_entries * heads_per_rank,
      "symmetric staging buffers are smaller than 2 * world_size * "
      "max_num_tokens entries");
  // Signal slots are strided so each one owns a cache line; the stride is
  // inferred from the buffer size to keep the op schema unchanged.
  const int64_t signal_slots = 2 * world_size;
  TORCH_CHECK(
      received_signal.numel() % signal_slots == 0,
      "received_signal must hold a multiple of 2 * world_size elements");
  const int64_t signal_stride = received_signal.numel() / signal_slots;
  TORCH_CHECK(
      signal_stride * static_cast<int64_t>(sizeof(uint32_t)) >= kCacheLineBytes,
      "received_signal slots must be padded to ",
      kCacheLineBytes,
      " bytes, got stride ",
      signal_stride);
  // Each source rank's LSE partition (and hence output partition) must start
  // on a cache line so different peers never write the same line.
  TORCH_CHECK(
      (max_num_tokens * heads_per_rank * static_cast<int64_t>(sizeof(float))) %
              kCacheLineBytes ==
          0,
      "max_num_tokens * heads_per_rank * 4 must be a multiple of ",
      kCacheLineBytes,
      " bytes");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(partial_output.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(received_output.data_ptr()) %
                  kCacheLineBytes ==
              0 &&
          reinterpret_cast<uintptr_t>(received_lse.data_ptr()) %
                  kCacheLineBytes ==
              0 &&
          reinterpret_cast<uintptr_t>(received_signal.data_ptr()) %
                  kCacheLineBytes ==
              0,
      "partial_output must be 16-byte aligned and staging buffers ",
      kCacheLineBytes,
      "-byte aligned");
  TORCH_CHECK(
      epoch.scalar_type() == at::kLong && epoch.numel() >= 1 &&
          epoch.device() == device,
      "epoch must be an int64 tensor on the input device");

  const int32_t* seq_lens_ptr = nullptr;
  const int32_t* query_start_loc_ptr = nullptr;
  int64_t num_seqs = 0;
  TORCH_CHECK(
      seq_lens.has_value() == query_start_loc.has_value(),
      "seq_lens and query_start_loc must be provided together");
  if (seq_lens.has_value()) {
    TORCH_CHECK(
        seq_lens->device() == device && seq_lens->dim() == 1 &&
            seq_lens->stride(0) == 1 && seq_lens->scalar_type() == at::kInt,
        "seq_lens must be a contiguous 1-D int32 tensor on the input device");
    TORCH_CHECK(
        query_start_loc->device() == device && query_start_loc->dim() == 1 &&
            query_start_loc->stride(0) == 1 &&
            query_start_loc->scalar_type() == at::kInt,
        "query_start_loc must be a contiguous 1-D int32 tensor on the input "
        "device");
    num_seqs = seq_lens->size(0);
    TORCH_CHECK(
        num_seqs > 0 && query_start_loc->size(0) == num_seqs + 1,
        "query_start_loc must contain one boundary per sequence");
    seq_lens_ptr = seq_lens->const_data_ptr<int32_t>();
    query_start_loc_ptr = query_start_loc->const_data_ptr<int32_t>();
  }

  const uint64_t* output_ptrs =
      ptr_table(peer_output_ptrs, "peer_output_ptrs", device, world_size);
  const uint64_t* lse_ptrs =
      ptr_table(peer_lse_ptrs, "peer_lse_ptrs", device, world_size);
  const uint64_t* signal_ptrs =
      ptr_table(peer_signal_ptrs, "peer_signal_ptrs", device, world_size);

  auto& queue = vllm::xpu::vllmGetQueue();
  constexpr int64_t kExchangeThreads = 256;
  constexpr int64_t kCombineThreads = 256;
  int64_t* epoch_ptr = epoch.data_ptr<int64_t>();

  vllm::xpu::sycl_kernel_submit(queue, 1, 1, IncrementEpoch{epoch_ptr});

  auto launch_dispatch = [&](auto lse_tag) {
    using lse_t = decltype(lse_tag);
    vllm::xpu::sycl_kernel_submit(
        queue,
        world_size * num_tokens * kExchangeThreads,
        kExchangeThreads,
        DispatchOutputLse<lse_t>(
            reinterpret_cast<const sycl::uint4*>(partial_output.data_ptr()),
            reinterpret_cast<const lse_t*>(partial_lse.data_ptr()),
            seq_lens_ptr,
            query_start_loc_ptr,
            output_ptrs,
            lse_ptrs,
            epoch_ptr,
            world_size,
            rank,
            num_tokens,
            max_num_tokens,
            num_seqs,
            heads_per_rank,
            head_dim,
            output_token_stride,
            lse_token_stride,
            lse_head_stride));
  };
  if (lse_dtype == at::kFloat) {
    launch_dispatch(float{});
  } else if (lse_dtype == at::kBFloat16) {
    launch_dispatch(sycl::ext::oneapi::bfloat16{});
  } else {
    launch_dispatch(sycl::half{});
  }

  const int64_t signal_groups =
      (world_size + kExchangeThreads - 1) / kExchangeThreads;
  vllm::xpu::sycl_kernel_submit(
      queue,
      signal_groups * kExchangeThreads,
      kExchangeThreads,
      SignalPeers{signal_ptrs, epoch_ptr, world_size, rank, signal_stride});

  // Split each token's heads across groups while the grid stays near
  // combine_target_groups. Every group polls the same signal words, so small
  // batches keep the grid small: on B60, 8 tokens drop from 17 to 15 us
  // (DCP2) with 32 instead of 128 groups.
  const int64_t combine_target_groups =
      std::clamp<int64_t>(4 * num_tokens, 32, 128);
  const int64_t heads_per_group = std::min(
      heads_per_rank,
      std::max<int64_t>(
          1,
          (num_tokens * heads_per_rank + combine_target_groups - 1) /
              combine_target_groups));
  const int64_t combine_groups =
      num_tokens * ((heads_per_rank + heads_per_group - 1) / heads_per_group);
  auto launch_combine = [&](auto scalar_tag) {
    using scalar_t = decltype(scalar_tag);
    vllm::xpu::sycl_kernel_submit(
        queue,
        combine_groups * kCombineThreads,
        kCombineThreads,
        WaitLseCombine<scalar_t>(
            reinterpret_cast<const scalar_t*>(received_output.data_ptr()),
            received_lse.const_data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(
                received_signal.const_data_ptr<int32_t>()),
            epoch_ptr,
            reinterpret_cast<scalar_t*>(combined_output.data_ptr()),
            world_size,
            max_num_tokens,
            heads_per_rank,
            head_dim,
            is_lse_base_on_e,
            signal_stride,
            heads_per_group));
  };
  if (output_dtype == at::kBFloat16) {
    launch_combine(sycl::ext::oneapi::bfloat16{});
  } else {
    launch_combine(sycl::half{});
  }
}
