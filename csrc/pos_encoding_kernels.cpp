#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"
#include "xpu/sycl/rotary_embedding.hpp"
#include <cmath>
#include <limits>
#include <c10/macros/Macros.h>

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
inline void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const scalar_t* cache_ptr,
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int token_idx,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride,
    const sycl::nd_item<3>& item_ct1) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = item_ct1.get_local_id(2); i < nq;
       i += item_ct1.get_local_range(2)) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query + token_head,
        query + token_head,
        cos_ptr,
        sin_ptr,
        rot_offset,
        embed_dim);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = item_ct1.get_local_id(2); i < nk;
         i += item_ct1.get_local_range(2)) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, IS_NEOX>(
          key + token_head,
          key + token_head,
          cos_ptr,
          sin_ptr,
          rot_offset,
          embed_dim);
    }
  }
}

template <typename scalar_t, bool IS_NEOX>
class rotary_embedding_kernel {
 public:
  rotary_embedding_kernel(
      const int64_t* __restrict__ positions_,  // [batch_size, seq_len] or
                                               // [num_tokens]
      scalar_t* __restrict__ query_,  // [batch_size, seq_len, num_heads,
                                      // head_size] or [num_tokens, num_heads,
                                      // head_size]
      scalar_t* __restrict__ key_,    // nullptr or
                                      // [batch_size, seq_len, num_kv_heads,
      // head_size] or [num_tokens, num_kv_heads,
      // head_size]
      const scalar_t* __restrict__ cos_sin_cache_,  // [max_position, 2, rot_dim
                                                    // // 2]
      const int rot_dim_,
      const int64_t query_stride_,
      const int64_t key_stride_,
      const int64_t head_stride_,
      const int num_heads_,
      const int num_kv_heads_,
      const int head_size_)
      : positions(positions_),
        query(query_),
        key(key_),
        cos_sin_cache(cos_sin_cache_),
        rot_dim(rot_dim_),
        query_stride(query_stride_),
        key_stride(key_stride_),
        head_stride(head_stride_),
        num_heads(num_heads_),
        num_kv_heads(num_kv_heads_),
        head_size(head_size_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    // Each thread block is responsible for one token.
    const int token_idx = item_ct1.get_group(2);
    int64_t pos = positions[token_idx];
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

    apply_rotary_embedding<scalar_t, IS_NEOX>(
        query,
        key,
        cache_ptr,
        head_size,
        num_heads,
        num_kv_heads,
        rot_dim,
        token_idx,
        query_stride,
        key_stride,
        head_stride,
        item_ct1);
  }

 private:
  const int64_t* __restrict__ positions;  // [batch_size, seq_len] or
                                          // [num_tokens]
  scalar_t* __restrict__ query;           // [batch_size, seq_len, num_heads,
                                 // head_size] or [num_tokens, num_heads,
                                 // head_size]
  scalar_t* __restrict__ key;  // nullptr or
                               // [batch_size, seq_len, num_kv_heads,
                               // head_size] or [num_tokens, num_kv_heads,
                               // head_size]
  const scalar_t* __restrict__ cos_sin_cache;  // [max_position, 2, rot_dim //
                                               // 2]
  const int rot_dim;
  const int64_t query_stride;
  const int64_t key_stride;
  const int64_t head_stride;
  const int num_heads;
  const int num_kv_heads;
  const int head_size;
};

// Vectorized rotary embedding.
//
// The reference kernel above maps one work-group to one token and lets every
// work-item rotate a single (x, y) element pair.  On Crescent Island that is
// heavily memory-latency bound: each work-item only issues 2-byte accesses,
// so a 32-wide sub-group requests 64 bytes per message instead of the 512
// bytes a fully vectorized sub-group can move, and the number of resident
// work-groups is pinned to the token count rather than to the 32 Xe cores.
//
// This kernel instead flattens (token, head, rot_offset) for query *and* key
// into a single 1-D index space where each work-item owns VEC_SIZE contiguous
// rotation offsets.  With VEC_SIZE * sizeof(scalar_t) == kRopeVecBytes every
// sub-group lane contributes a full-width access, query and key are drained by
// one dispatch, and the grid is sized by total work instead of token count.
template <typename scalar_t, bool IS_NEOX, int VEC_SIZE>
class rotary_embedding_vec_kernel {
 public:
  rotary_embedding_vec_kernel(
      const int64_t* __restrict__ positions_,
      scalar_t* __restrict__ query_,
      scalar_t* __restrict__ key_,
      const scalar_t* __restrict__ cos_sin_cache_,
      const int rot_dim_,
      const int64_t query_stride_,
      const int64_t key_stride_,
      const int64_t head_stride_,
      const int vecs_per_head_,
      const int query_vecs_per_token_,
      const int vecs_per_token_,
      const int total_vecs_)
      : positions(positions_),
        query(query_),
        key(key_),
        cos_sin_cache(cos_sin_cache_),
        rot_dim(rot_dim_),
        query_stride(query_stride_),
        key_stride(key_stride_),
        head_stride(head_stride_),
        vecs_per_head(vecs_per_head_),
        query_vecs_per_token(query_vecs_per_token_),
        vecs_per_token(vecs_per_token_),
        total_vecs(total_vecs_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<1>& item_ct1) const {
    const int idx = static_cast<int>(item_ct1.get_global_id(0));
    if (idx >= total_vecs) return;

    const int token_idx = idx / vecs_per_token;
    int rem = idx - token_idx * vecs_per_token;

    // The first query_vecs_per_token slots of a token belong to the query
    // tensor, the remainder (if any) to the key tensor.
    scalar_t* __restrict__ base;
    int64_t token_stride;
    if (rem < query_vecs_per_token) {
      base = query;
      token_stride = query_stride;
    } else {
      base = key;
      token_stride = key_stride;
      rem -= query_vecs_per_token;
    }

    const int head_idx = rem / vecs_per_head;
    const int rot_offset = (rem - head_idx * vecs_per_head) * VEC_SIZE;

    const int embed_dim = rot_dim >> 1;
    const int64_t pos = positions[token_idx];
    const scalar_t* __restrict__ cos_ptr = cos_sin_cache + pos * rot_dim;
    scalar_t* __restrict__ ptr =
        base + token_idx * token_stride + head_idx * head_stride;

    if constexpr (IS_NEOX) {
      apply_token_rotary_embedding_vec<scalar_t, VEC_SIZE>(
          ptr, ptr, cos_ptr, cos_ptr + embed_dim, rot_offset, embed_dim);
    } else {
      apply_token_rotary_embedding_gptj_vec<scalar_t, VEC_SIZE>(
          ptr, ptr, cos_ptr, cos_ptr + embed_dim, rot_offset);
    }
  }

 private:
  const int64_t* __restrict__ positions;
  scalar_t* __restrict__ query;
  scalar_t* __restrict__ key;
  const scalar_t* __restrict__ cos_sin_cache;
  const int rot_dim;
  const int64_t query_stride;
  const int64_t key_stride;
  const int64_t head_stride;
  const int vecs_per_head;
  const int query_vecs_per_token;
  const int vecs_per_token;
  const int total_vecs;
};

}  // namespace vllm

namespace {

// Tuned on Crescent Island (32 Xe cores / 256 XVEs, sub-group size 32).
//
// kRopeVecBytes: bytes touched per work-item per access.  A sweep of
// {8, 16, 32} x work-group {64 … 1024} on the Qwen3-30B-A3B shape
// (4096 tokens, 32 q heads, 4 kv heads, head_dim 128, bf16) gave
// 97 us / 58 us / 55.7 us for 8 / 16 / 32 bytes, so the widest access wins:
// it halves the number of memory messages and keeps the HBM pipe saturated.
// kRopeVecWorkGroupSize: flat within 55.0-55.7 us for 64…1024; 256 is picked
// as the middle of that plateau.
constexpr int kRopeVecBytes = 32;
constexpr int kRopeVecWorkGroupSize = 256;

template <typename sycl_t, bool IS_NEOX, int VEC_SIZE>
bool try_launch_rotary_embedding_vec(
    sycl::queue& queue,
    const int64_t* positions_ptr,
    sycl_t* query_ptr,
    sycl_t* key_ptr,
    const sycl_t* cos_sin_cache_ptr,
    int64_t num_tokens,
    int rot_dim,
    int64_t query_stride,
    int64_t key_stride,
    int64_t head_stride,
    int num_heads,
    int num_kv_heads) {
  const int embed_dim = rot_dim / 2;
  if (embed_dim % VEC_SIZE != 0) return false;

  // NeoX touches VEC_SIZE contiguous elements at two offsets that are both
  // VEC_SIZE-aligned; GPT-J touches 2 * VEC_SIZE contiguous elements.
  constexpr int kElemAlign = IS_NEOX ? VEC_SIZE : 2 * VEC_SIZE;
  constexpr uintptr_t kByteAlign = kElemAlign * sizeof(sycl_t);
  constexpr uintptr_t kCacheByteAlign = VEC_SIZE * sizeof(sycl_t);

  if (query_stride % kElemAlign != 0 || head_stride % kElemAlign != 0)
    return false;
  if (reinterpret_cast<uintptr_t>(query_ptr) % kByteAlign != 0) return false;
  if (key_ptr != nullptr) {
    if (key_stride % kElemAlign != 0) return false;
    if (reinterpret_cast<uintptr_t>(key_ptr) % kByteAlign != 0) return false;
  }
  if (reinterpret_cast<uintptr_t>(cos_sin_cache_ptr) % kCacheByteAlign != 0)
    return false;

  const int vecs_per_head = embed_dim / VEC_SIZE;
  const int64_t query_vecs_per_token =
      static_cast<int64_t>(num_heads) * vecs_per_head;
  const int64_t key_vecs_per_token =
      key_ptr != nullptr ? static_cast<int64_t>(num_kv_heads) * vecs_per_head
                         : 0;
  const int64_t vecs_per_token = query_vecs_per_token + key_vecs_per_token;
  const int64_t total_vecs = num_tokens * vecs_per_token;
  if (total_vecs <= 0) return false;
  // The kernel indexes with 32-bit math to keep the per-work-item divisions
  // cheap; fall back to the reference kernel for anything larger.
  if (total_vecs > std::numeric_limits<int>::max()) return false;

  const int wg = kRopeVecWorkGroupSize;
  const size_t global =
      static_cast<size_t>((total_vecs + wg - 1) / wg) * static_cast<size_t>(wg);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        vllm::rotary_embedding_vec_kernel<sycl_t, IS_NEOX, VEC_SIZE>(
            positions_ptr,
            query_ptr,
            key_ptr,
            cos_sin_cache_ptr,
            rot_dim,
            query_stride,
            key_stride,
            head_stride,
            vecs_per_head,
            static_cast<int>(query_vecs_per_token),
            static_cast<int>(vecs_per_token),
            static_cast<int>(total_vecs)));
  });
  return true;
}

template <typename sycl_t, bool IS_NEOX, int BYTES>
struct RopeVecCandidate {
  static constexpr int kVec = BYTES / static_cast<int>(sizeof(sycl_t));
  // sycl::vec tops out at 16 elements; GPT-J loads 2 * kVec at once.
  static constexpr int kMaxVec = IS_NEOX ? 16 : 8;
  static constexpr bool kValid = kVec >= 2 && kVec <= kMaxVec;
};

template <typename sycl_t, bool IS_NEOX, int BYTES, typename... Args>
bool try_rope_vec_width(Args&&... args) {
  using C = RopeVecCandidate<sycl_t, IS_NEOX, BYTES>;
  if constexpr (C::kValid && BYTES <= kRopeVecBytes) {
    return try_launch_rotary_embedding_vec<sycl_t, IS_NEOX, C::kVec>(args...);
  }
  return false;
}

template <typename sycl_t, bool IS_NEOX, typename... Args>
bool launch_rotary_embedding_vec(Args&&... args) {
  // Widest access first, then progressively narrower ones so that shapes or
  // alignments that cannot sustain the wide access still get vectorized.
  return try_rope_vec_width<sycl_t, IS_NEOX, 32>(args...) ||
         try_rope_vec_width<sycl_t, IS_NEOX, 16>(args...) ||
         try_rope_vec_width<sycl_t, IS_NEOX, 8>(args...) ||
         try_rope_vec_width<sycl_t, IS_NEOX, 4>(args...);
}

}  // namespace

template <typename scalar_t>
void call_rotary_embedding_kernel(
    torch::Tensor& positions,
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  // num_tokens = batch_size * seq_len
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();

  // Make sure num_tokens dim is consistent across positions, query, and key
  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)),
        "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) &&
            (!key.has_value() || key->size(1) == positions.size(1)),
        "query, key and positions must have the same batch_size and seq_len");
  }

  // Make sure head_size is valid for query and key
  // hidden_size = num_heads * head_size
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  // Make sure query and key have consistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  // Determine head stride: for [*, heads, head_size] use stride of last dim;
  // for flat [*, heads*head_size], heads blocks are contiguous of size
  // head_size
  int query_ndim = query.dim();
  int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  auto positions_ptr = positions.data_ptr<int64_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_ptr = key.has_value() ? key->data_ptr<scalar_t>() : nullptr;
  auto cos_sin_cache_ptr = cos_sin_cache.data_ptr<scalar_t>();

  at::DeviceGuard device_guard(query.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  // Fast path: fully vectorized, flat index space over query + key.
  const bool launched = is_neox ? launch_rotary_embedding_vec<sycl_t, true>(
                                      queue,
                                      positions_ptr,
                                      (sycl_t*)query_ptr,
                                      (sycl_t*)key_ptr,
                                      (const sycl_t*)cos_sin_cache_ptr,
                                      num_tokens,
                                      rot_dim,
                                      query_stride,
                                      key_stride,
                                      head_stride,
                                      num_heads,
                                      num_kv_heads)
                                : launch_rotary_embedding_vec<sycl_t, false>(
                                      queue,
                                      positions_ptr,
                                      (sycl_t*)query_ptr,
                                      (sycl_t*)key_ptr,
                                      (const sycl_t*)cos_sin_cache_ptr,
                                      num_tokens,
                                      rot_dim,
                                      query_stride,
                                      key_stride,
                                      head_stride,
                                      num_heads,
                                      num_kv_heads);
  if (launched) return;

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min<int64_t>(num_heads * rot_dim / 2, 512));

  if (is_neox) {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::rotary_embedding_kernel<sycl_t, true>(
              positions_ptr,
              (sycl_t*)query_ptr,
              (sycl_t*)key_ptr,
              (sycl_t*)cos_sin_cache_ptr,
              rot_dim,
              query_stride,
              key_stride,
              head_stride,
              num_heads,
              num_kv_heads,
              head_size));
    });
  } else {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::rotary_embedding_kernel<sycl_t, false>(
              positions_ptr,
              (sycl_t*)query_ptr,
              (sycl_t*)key_ptr,
              (sycl_t*)cos_sin_cache_ptr,
              rot_dim,
              query_stride,
              key_stride,
              head_stride,
              num_heads,
              num_kv_heads,
              head_size));
    });
  }
}

void rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    std::optional<torch::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    call_rotary_embedding_kernel<scalar_t>(
        positions, query, key, head_size, cos_sin_cache, is_neox);
  });
}
