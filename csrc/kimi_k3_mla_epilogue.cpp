/*
 * XPU port of the CUDA Kimi-K3 MLA epilogue kernels
 * (vllm/csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu).
 * bf16/fp16 only — the fp8 and fp8_ds_mla cache-layout variants, and the
 * sm_90+ PDL launch overlap, are not ported.
 */
#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>

#include "dispatch_utils.h"
#include "utils.h"

namespace vllm {
namespace kimi_k3 {

constexpr int kKvLoraRank = 512;
constexpr int kQkNopeHeadDim = 128;
constexpr int kQkRopeHeadDim = 64;
constexpr int kQkHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;  // 192
constexpr int kCacheEntry = kKvLoraRank + kQkRopeHeadDim;    // 576
constexpr int kVecElems = 8;  // 8 bf16/fp16 elems == 16B vector
constexpr int kSubGroupSize = 32;

template <typename scalar_t, bool APPLY_ROPE>
static inline void copy_chunk8(
    scalar_t* dst, const scalar_t* src, int elem_base, const float* cos_sin) {
  using vllm::xpu::from_float;
  using vllm::xpu::to_float;
  auto v =
      *reinterpret_cast<const vllm::xpu::aligned_vec<scalar_t, kVecElems>*>(
          src);
  if constexpr (APPLY_ROPE) {
#pragma unroll
    for (int i = 0; i < kVecElems / 2; i++) {
      int const pair = elem_base / 2 + i;
      float const cos = cos_sin[pair];
      float const sin = cos_sin[pair + kQkRopeHeadDim / 2];
      float const x = to_float(v[2 * i]);
      float const y = to_float(v[2 * i + 1]);
      from_float(v[2 * i], x * cos - y * sin);
      from_float(v[2 * i + 1], x * sin + y * cos);
    }
  }
  *reinterpret_cast<vllm::xpu::aligned_vec<scalar_t, kVecElems>*>(dst) = v;
}

// One sub-group per (token, head) writes q-RoPE + full-key concat; one
// extra sub-group per token (slot == num_heads) writes the latent cache row.
template <typename scalar_t, bool APPLY_ROPE>
class KeyConcatKvCacheKernel {
 public:
  KeyConcatKvCacheKernel(
      scalar_t* q_,
      int64_t q_tok_stride_,
      int64_t q_head_stride_,
      const scalar_t* k_nope_,
      int64_t kn_tok_stride_,
      int64_t kn_head_stride_,
      const scalar_t* k_pe_,
      int64_t k_pe_tok_stride_,
      const scalar_t* kv_c_,
      int64_t kv_c_tok_stride_,
      scalar_t* k_out_,
      int64_t ko_tok_stride_,
      int64_t ko_head_stride_,
      scalar_t* k_cache_,
      int64_t cache_block_stride_,
      int64_t cache_token_stride_,
      const int64_t* slot_mapping_,
      const int64_t* position_ids_,
      const float* cos_sin_cache_,
      int num_tokens_,
      int num_heads_,
      int cache_block_size_)
      : q(q_),
        q_tok_stride(q_tok_stride_),
        q_head_stride(q_head_stride_),
        k_nope(k_nope_),
        kn_tok_stride(kn_tok_stride_),
        kn_head_stride(kn_head_stride_),
        k_pe(k_pe_),
        k_pe_tok_stride(k_pe_tok_stride_),
        kv_c(kv_c_),
        kv_c_tok_stride(kv_c_tok_stride_),
        k_out(k_out_),
        ko_tok_stride(ko_tok_stride_),
        ko_head_stride(ko_head_stride_),
        k_cache(k_cache_),
        cache_block_stride(cache_block_stride_),
        cache_token_stride(cache_token_stride_),
        slot_mapping(slot_mapping_),
        position_ids(position_ids_),
        cos_sin_cache(cos_sin_cache_),
        num_tokens(num_tokens_),
        num_heads(num_heads_),
        cache_block_size(cache_block_size_) {}

  void operator() [[sycl::reqd_sub_group_size(kSubGroupSize)]] (
      sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    int const lane = sg.get_local_linear_id();
    int const global_sg = item.get_group(0) * sg.get_group_linear_range() +
                          sg.get_group_linear_id();
    int const slots_per_token = num_heads + 1;
    int const token = global_sg / slots_per_token;
    int const slot = global_sg % slots_per_token;
    if (token >= num_tokens) return;

    const float* rope = nullptr;
    if constexpr (APPLY_ROPE) {
      rope = cos_sin_cache + position_ids[token] * kQkRopeHeadDim;
    }

    if (slot < num_heads) {
      scalar_t* qh = q + token * q_tok_stride + slot * q_head_stride;
      if constexpr (APPLY_ROPE) {
        for (int e = lane * kVecElems; e < kQkRopeHeadDim;
             e += kSubGroupSize * kVecElems) {
          copy_chunk8<scalar_t, true>(
              qh + kQkNopeHeadDim + e, qh + kQkNopeHeadDim + e, e, rope);
        }
      }
      scalar_t* ko = k_out + token * ko_tok_stride + slot * ko_head_stride;
      const scalar_t* kn =
          k_nope + token * kn_tok_stride + slot * kn_head_stride;
      const scalar_t* kp = k_pe + token * k_pe_tok_stride;
      for (int e = lane * kVecElems; e < kQkHeadDim;
           e += kSubGroupSize * kVecElems) {
        if (e < kQkNopeHeadDim) {
          copy_chunk8<scalar_t, false>(ko + e, kn + e, 0, nullptr);
        } else {
          int const rope_e = e - kQkNopeHeadDim;
          copy_chunk8<scalar_t, APPLY_ROPE>(ko + e, kp + rope_e, rope_e, rope);
        }
      }
    } else {
      int64_t const slot_id = slot_mapping[token];
      if (slot_id < 0) return;
      scalar_t* row = k_cache +
                      (slot_id / cache_block_size) * cache_block_stride +
                      (slot_id % cache_block_size) * cache_token_stride;
      const scalar_t* kvc = kv_c + token * kv_c_tok_stride;
      const scalar_t* kp = k_pe + token * k_pe_tok_stride;
      for (int e = lane * kVecElems; e < kCacheEntry;
           e += kSubGroupSize * kVecElems) {
        if (e < kKvLoraRank) {
          copy_chunk8<scalar_t, false>(row + e, kvc + e, 0, nullptr);
        } else {
          int const rope_e = e - kKvLoraRank;
          copy_chunk8<scalar_t, APPLY_ROPE>(row + e, kp + rope_e, rope_e, rope);
        }
      }
    }
  }

 private:
  scalar_t* q;
  int64_t q_tok_stride, q_head_stride;
  const scalar_t* k_nope;
  int64_t kn_tok_stride, kn_head_stride;
  const scalar_t* k_pe;
  int64_t k_pe_tok_stride;
  const scalar_t* kv_c;
  int64_t kv_c_tok_stride;
  scalar_t* k_out;
  int64_t ko_tok_stride, ko_head_stride;
  scalar_t* k_cache;
  int64_t cache_block_stride, cache_token_stride;
  const int64_t* slot_mapping;
  const int64_t* position_ids;
  const float* cos_sin_cache;
  int num_tokens, num_heads, cache_block_size;
};

// One sub-group per (token, head) writes mqa_q = [ql_nope | q_pe]; one extra
// sub-group per token (slot == num_heads) writes the latent cache row.
template <typename scalar_t, bool APPLY_ROPE>
class DecodeQConcatKvCacheKernel {
 public:
  DecodeQConcatKvCacheKernel(
      const scalar_t* ql_nope_,
      int64_t qn_tok_stride_,
      int64_t qn_head_stride_,
      const scalar_t* q_pe_,
      int64_t qpe_tok_stride_,
      int64_t qpe_head_stride_,
      const scalar_t* kv_c_,
      int64_t kv_c_tok_stride_,
      const scalar_t* k_pe_,
      int64_t k_pe_tok_stride_,
      scalar_t* mqa_q_,
      int64_t mq_tok_stride_,
      int64_t mq_head_stride_,
      scalar_t* k_cache_,
      int64_t cache_block_stride_,
      int64_t cache_token_stride_,
      const int64_t* slot_mapping_,
      const int64_t* position_ids_,
      const float* cos_sin_cache_,
      int num_tokens_,
      int num_heads_,
      int cache_block_size_)
      : ql_nope(ql_nope_),
        qn_tok_stride(qn_tok_stride_),
        qn_head_stride(qn_head_stride_),
        q_pe(q_pe_),
        qpe_tok_stride(qpe_tok_stride_),
        qpe_head_stride(qpe_head_stride_),
        kv_c(kv_c_),
        kv_c_tok_stride(kv_c_tok_stride_),
        k_pe(k_pe_),
        k_pe_tok_stride(k_pe_tok_stride_),
        mqa_q(mqa_q_),
        mq_tok_stride(mq_tok_stride_),
        mq_head_stride(mq_head_stride_),
        k_cache(k_cache_),
        cache_block_stride(cache_block_stride_),
        cache_token_stride(cache_token_stride_),
        slot_mapping(slot_mapping_),
        position_ids(position_ids_),
        cos_sin_cache(cos_sin_cache_),
        num_tokens(num_tokens_),
        num_heads(num_heads_),
        cache_block_size(cache_block_size_) {}

  void operator() [[sycl::reqd_sub_group_size(kSubGroupSize)]] (
      sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    int const lane = sg.get_local_linear_id();
    int const global_sg = item.get_group(0) * sg.get_group_linear_range() +
                          sg.get_group_linear_id();
    int const slots_per_token = num_heads + 1;
    int const token = global_sg / slots_per_token;
    int const slot = global_sg % slots_per_token;
    if (token >= num_tokens) return;

    const float* rope = nullptr;
    if constexpr (APPLY_ROPE) {
      rope = cos_sin_cache + position_ids[token] * kQkRopeHeadDim;
    }

    if (slot < num_heads) {
      scalar_t* dst = mqa_q + token * mq_tok_stride + slot * mq_head_stride;
      const scalar_t* a =
          ql_nope + token * qn_tok_stride + slot * qn_head_stride;
      const scalar_t* b =
          q_pe + token * qpe_tok_stride + slot * qpe_head_stride;
      for (int e = lane * kVecElems; e < kCacheEntry;
           e += kSubGroupSize * kVecElems) {
        if (e < kKvLoraRank) {
          copy_chunk8<scalar_t, false>(dst + e, a + e, 0, nullptr);
        } else {
          int const rope_e = e - kKvLoraRank;
          copy_chunk8<scalar_t, APPLY_ROPE>(dst + e, b + rope_e, rope_e, rope);
        }
      }
    } else {
      int64_t const slot_id = slot_mapping[token];
      if (slot_id < 0) return;
      scalar_t* row = k_cache +
                      (slot_id / cache_block_size) * cache_block_stride +
                      (slot_id % cache_block_size) * cache_token_stride;
      const scalar_t* kvc = kv_c + token * kv_c_tok_stride;
      const scalar_t* kp = k_pe + token * k_pe_tok_stride;
      for (int e = lane * kVecElems; e < kCacheEntry;
           e += kSubGroupSize * kVecElems) {
        if (e < kKvLoraRank) {
          copy_chunk8<scalar_t, false>(row + e, kvc + e, 0, nullptr);
        } else {
          int const rope_e = e - kKvLoraRank;
          copy_chunk8<scalar_t, APPLY_ROPE>(row + e, kp + rope_e, rope_e, rope);
        }
      }
    }
  }

 private:
  const scalar_t* ql_nope;
  int64_t qn_tok_stride, qn_head_stride;
  const scalar_t* q_pe;
  int64_t qpe_tok_stride, qpe_head_stride;
  const scalar_t* kv_c;
  int64_t kv_c_tok_stride;
  const scalar_t* k_pe;
  int64_t k_pe_tok_stride;
  scalar_t* mqa_q;
  int64_t mq_tok_stride, mq_head_stride;
  scalar_t* k_cache;
  int64_t cache_block_stride, cache_token_stride;
  const int64_t* slot_mapping;
  const int64_t* position_ids;
  const float* cos_sin_cache;
  int num_tokens, num_heads, cache_block_size;
};

}  // namespace kimi_k3
}  // namespace vllm

namespace {
// position_ids/cos_sin_cache are provided together or not at all; validated
// the same way as the CUDA op (int64 [num_tokens], fp32 [max_position, 64]).
bool check_rope_inputs(
    std::optional<torch::Tensor> const& position_ids,
    std::optional<torch::Tensor> const& cos_sin_cache,
    int64_t num_tokens) {
  TORCH_CHECK(
      position_ids.has_value() == cos_sin_cache.has_value(),
      "position_ids and cos_sin_cache must be provided together");
  if (!position_ids.has_value()) return false;
  auto const& positions = position_ids.value();
  auto const& rope_cache = cos_sin_cache.value();
  CHECK_DEVICE(positions);
  CHECK_DEVICE(rope_cache);
  TORCH_CHECK(
      positions.dim() == 1 && positions.scalar_type() == torch::kInt64 &&
          positions.size(0) == num_tokens,
      "position_ids must be int64 XPU with shape [num_tokens]");
  TORCH_CHECK(
      rope_cache.dim() == 2 && rope_cache.size(1) == 64 &&
          rope_cache.stride(1) == 1 &&
          rope_cache.scalar_type() == torch::kFloat32,
      "cos_sin_cache must have shape [max_position, 64], unit last-dim "
      "stride, and be fp32");
  return true;
}
}  // namespace

void fused_kimi_k3_mla_key_concat_kv_cache_insert(
    torch::Tensor& q,
    torch::Tensor const& k_nope,
    torch::Tensor const& k_pe,
    torch::Tensor const& kv_c_normed,
    torch::Tensor& k_out,
    torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping,
    int64_t cache_block_size,
    std::optional<torch::Tensor> position_ids,
    std::optional<torch::Tensor> cos_sin_cache) {
  CHECK_DEVICE(q);
  CHECK_DEVICE(k_nope);
  CHECK_DEVICE(k_pe);
  CHECK_DEVICE(kv_c_normed);
  CHECK_DEVICE(k_out);
  CHECK_DEVICE(k_cache);
  CHECK_DEVICE(slot_mapping);
  TORCH_CHECK(k_nope.dim() == 3 && k_nope.size(2) == 128, "k_nope [Tp,H,128]");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 192, "q [Tp,H,192]");
  CHECK_STRIDE_ALIGNMENT(k_pe);
  TORCH_CHECK(k_pe.dim() == 2 && k_pe.size(1) == 64, "k_pe [Tp,64]");
  CHECK_CONTIGUOUS(kv_c_normed);
  TORCH_CHECK(
      kv_c_normed.dim() == 2 && kv_c_normed.size(1) == 512,
      "kv_c_normed [Tp,512] contiguous");
  CHECK_CONTIGUOUS(k_out);
  TORCH_CHECK(k_out.dim() == 3 && k_out.size(2) == 192, "k_out [Tp,H,192]");
  CHECK_CONTIGUOUS(k_cache);
  TORCH_CHECK(
      k_cache.dim() == 3 && k_cache.size(1) == cache_block_size &&
          k_cache.size(2) == 576,
      "k_cache [nblk,block_size,576] contiguous");
  TORCH_CHECK(
      slot_mapping.scalar_type() == torch::kInt64, "slot_mapping int64");
  auto const dt = k_nope.scalar_type();
  TORCH_CHECK(
      q.scalar_type() == dt && k_pe.scalar_type() == dt &&
          kv_c_normed.scalar_type() == dt && k_out.scalar_type() == dt &&
          k_cache.scalar_type() == dt,
      "all tensors must share k_nope's dtype");

  int const num_tokens = static_cast<int>(k_nope.size(0));
  int const num_heads = static_cast<int>(k_nope.size(1));
  bool const apply_rope =
      check_rope_inputs(position_ids, cos_sin_cache, num_tokens);
  if (num_tokens == 0) return;

  const at::DeviceGuard device_guard(k_nope.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  constexpr int kSgsPerWg = 8;
  int64_t const total_sgs = static_cast<int64_t>(num_tokens) * (num_heads + 1);
  int64_t const grid_sgs = (total_sgs + kSgsPerWg - 1) / kSgsPerWg;

  VLLM_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_key_concat_kv_cache_insert", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        auto q_ptr = reinterpret_cast<sycl_t*>(q.data_ptr<scalar_t>());
        auto kn_ptr =
            reinterpret_cast<const sycl_t*>(k_nope.data_ptr<scalar_t>());
        auto kp_ptr =
            reinterpret_cast<const sycl_t*>(k_pe.data_ptr<scalar_t>());
        auto kvc_ptr =
            reinterpret_cast<const sycl_t*>(kv_c_normed.data_ptr<scalar_t>());
        auto ko_ptr = reinterpret_cast<sycl_t*>(k_out.data_ptr<scalar_t>());
        auto kc_ptr = reinterpret_cast<sycl_t*>(k_cache.data_ptr<scalar_t>());
        auto sm_ptr = slot_mapping.data_ptr<int64_t>();
        auto pid_ptr =
            apply_rope ? position_ids.value().data_ptr<int64_t>() : nullptr;
        auto cs_ptr = apply_rope ? reinterpret_cast<const float*>(
                                       cos_sin_cache.value().data_ptr<float>())
                                 : nullptr;

        auto launch = [&](auto apply_rope_tag) {
          constexpr bool kApplyRope = decltype(apply_rope_tag)::value;
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(
                    grid_sgs * kSgsPerWg * vllm::kimi_k3::kSubGroupSize,
                    kSgsPerWg * vllm::kimi_k3::kSubGroupSize),
                vllm::kimi_k3::KeyConcatKvCacheKernel<sycl_t, kApplyRope>(
                    q_ptr,
                    q.stride(0),
                    q.stride(1),
                    kn_ptr,
                    k_nope.stride(0),
                    k_nope.stride(1),
                    kp_ptr,
                    k_pe.stride(0),
                    kvc_ptr,
                    kv_c_normed.stride(0),
                    ko_ptr,
                    k_out.stride(0),
                    k_out.stride(1),
                    kc_ptr,
                    k_cache.stride(0),
                    k_cache.stride(1),
                    sm_ptr,
                    pid_ptr,
                    cs_ptr,
                    num_tokens,
                    num_heads,
                    static_cast<int>(cache_block_size)));
          });
        };
        if (apply_rope) {
          launch(std::bool_constant<true>{});
        } else {
          launch(std::bool_constant<false>{});
        }
      });
}

void fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
    torch::Tensor const& ql_nope,
    torch::Tensor const& q_pe,
    torch::Tensor const& kv_c_normed,
    torch::Tensor const& k_pe,
    torch::Tensor& mqa_q,
    torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping,
    int64_t cache_block_size,
    std::optional<torch::Tensor> position_ids,
    std::optional<torch::Tensor> cos_sin_cache) {
  CHECK_DEVICE(ql_nope);
  CHECK_DEVICE(q_pe);
  CHECK_DEVICE(kv_c_normed);
  CHECK_DEVICE(k_pe);
  CHECK_DEVICE(mqa_q);
  CHECK_DEVICE(k_cache);
  CHECK_DEVICE(slot_mapping);
  auto const dt = ql_nope.scalar_type();
  TORCH_CHECK(
      ql_nope.dim() == 3 && ql_nope.size(2) == 512, "ql_nope [B,H,512]");
  TORCH_CHECK(
      q_pe.scalar_type() == dt && q_pe.dim() == 3 && q_pe.size(2) == 64,
      "q_pe [B,H,64]");
  CHECK_CONTIGUOUS(kv_c_normed);
  TORCH_CHECK(
      kv_c_normed.scalar_type() == dt && kv_c_normed.dim() == 2 &&
          kv_c_normed.size(1) == 512,
      "kv_c_normed [B,512] contiguous");
  CHECK_STRIDE_ALIGNMENT(k_pe);
  TORCH_CHECK(
      k_pe.scalar_type() == dt && k_pe.dim() == 2 && k_pe.size(1) == 64,
      "k_pe [B,64]");
  CHECK_CONTIGUOUS(mqa_q);
  TORCH_CHECK(
      mqa_q.scalar_type() == dt && mqa_q.dim() == 3 && mqa_q.size(2) == 576,
      "mqa_q [B,H,576] contiguous");
  CHECK_CONTIGUOUS(k_cache);
  TORCH_CHECK(
      k_cache.scalar_type() == dt && k_cache.dim() == 3 &&
          k_cache.size(1) == cache_block_size && k_cache.size(2) == 576,
      "k_cache [nblk,block_size,576] contiguous, matches ql_nope dtype");
  TORCH_CHECK(
      slot_mapping.scalar_type() == torch::kInt64, "slot_mapping int64");

  int const num_tokens = static_cast<int>(ql_nope.size(0));
  int const num_heads = static_cast<int>(ql_nope.size(1));
  bool const apply_rope =
      check_rope_inputs(position_ids, cos_sin_cache, num_tokens);
  if (num_tokens == 0) return;

  const at::DeviceGuard device_guard(ql_nope.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  constexpr int kSgsPerWg = 8;
  int64_t const total_sgs = static_cast<int64_t>(num_tokens) * (num_heads + 1);
  int64_t const grid_sgs = (total_sgs + kSgsPerWg - 1) / kSgsPerWg;

  VLLM_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        auto qn_ptr =
            reinterpret_cast<const sycl_t*>(ql_nope.data_ptr<scalar_t>());
        auto qp_ptr =
            reinterpret_cast<const sycl_t*>(q_pe.data_ptr<scalar_t>());
        auto kvc_ptr =
            reinterpret_cast<const sycl_t*>(kv_c_normed.data_ptr<scalar_t>());
        auto kp_ptr =
            reinterpret_cast<const sycl_t*>(k_pe.data_ptr<scalar_t>());
        auto mq_ptr = reinterpret_cast<sycl_t*>(mqa_q.data_ptr<scalar_t>());
        auto kc_ptr = reinterpret_cast<sycl_t*>(k_cache.data_ptr<scalar_t>());
        auto sm_ptr = slot_mapping.data_ptr<int64_t>();
        auto pid_ptr =
            apply_rope ? position_ids.value().data_ptr<int64_t>() : nullptr;
        auto cs_ptr = apply_rope ? reinterpret_cast<const float*>(
                                       cos_sin_cache.value().data_ptr<float>())
                                 : nullptr;

        auto launch = [&](auto apply_rope_tag) {
          constexpr bool kApplyRope = decltype(apply_rope_tag)::value;
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(
                    grid_sgs * kSgsPerWg * vllm::kimi_k3::kSubGroupSize,
                    kSgsPerWg * vllm::kimi_k3::kSubGroupSize),
                vllm::kimi_k3::DecodeQConcatKvCacheKernel<sycl_t, kApplyRope>(
                    qn_ptr,
                    ql_nope.stride(0),
                    ql_nope.stride(1),
                    qp_ptr,
                    q_pe.stride(0),
                    q_pe.stride(1),
                    kvc_ptr,
                    kv_c_normed.stride(0),
                    kp_ptr,
                    k_pe.stride(0),
                    mq_ptr,
                    mqa_q.stride(0),
                    mqa_q.stride(1),
                    kc_ptr,
                    k_cache.stride(0),
                    k_cache.stride(1),
                    sm_ptr,
                    pid_ptr,
                    cs_ptr,
                    num_tokens,
                    num_heads,
                    static_cast<int>(cache_block_size)));
          });
        };
        if (apply_rope) {
          launch(std::bool_constant<true>{});
        } else {
          launch(std::bool_constant<false>{});
        }
      });
}
