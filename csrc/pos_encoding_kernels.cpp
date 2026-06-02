#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"
#include <cmath>
#include <c10/macros/Macros.h>

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
inline void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = cos_ptr[x_index / 2];
    sin = sin_ptr[x_index / 2];
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

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
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
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
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
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

}  // namespace vllm

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

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min<int64_t>(num_heads * rot_dim / 2, 512));

  at::DeviceGuard device_guard(query.device());
  auto& queue = vllm::xpu::vllmGetQueue();
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

// ─── apply_rotary_emb: takes cos/sin directly (flash_attn style) ────────────

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
class apply_rotary_emb_kernel {
 public:
  apply_rotary_emb_kernel(
      scalar_t* __restrict__ output_,
      const scalar_t* __restrict__ input_,
      const scalar_t* __restrict__ cos_,
      const scalar_t* __restrict__ sin_,
      const int num_tokens_,
      const int num_heads_,
      const int head_size_,
      const int rot_dim_,
      const int64_t input_stride_,
      const int64_t head_stride_,
      const int64_t cos_stride_)
      : output(output_),
        input(input_),
        cos_ptr(cos_),
        sin_ptr(sin_),
        num_tokens(num_tokens_),
        num_heads(num_heads_),
        head_size(head_size_),
        rot_dim(rot_dim_),
        input_stride(input_stride_),
        head_stride(head_stride_),
        cos_stride(cos_stride_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item) const {
    const int token_idx = item.get_group(2);
    const int embed_dim = rot_dim / 2;

    const scalar_t* token_cos = cos_ptr + token_idx * cos_stride;
    const scalar_t* token_sin = sin_ptr + token_idx * cos_stride;

    const int nq = num_heads * embed_dim;
    for (int i = item.get_local_id(2); i < nq; i += item.get_local_range(2)) {
      const int head_idx = i / embed_dim;
      const int rot_offset = i % embed_dim;

      const int64_t in_base = token_idx * input_stride + head_idx * head_stride;

      int x_index, y_index;
      scalar_t cos_val, sin_val;
      if (IS_NEOX) {
        x_index = rot_offset;
        y_index = embed_dim + rot_offset;
        cos_val = token_cos[rot_offset];
        sin_val = token_sin[rot_offset];
      } else {
        x_index = 2 * rot_offset;
        y_index = 2 * rot_offset + 1;
        cos_val = token_cos[rot_offset];
        sin_val = token_sin[rot_offset];
      }

      const scalar_t x = input[in_base + x_index];
      const scalar_t y = input[in_base + y_index];
      output[in_base + x_index] = x * cos_val - y * sin_val;
      output[in_base + y_index] = y * cos_val + x * sin_val;
    }

    if (rot_dim < head_size) {
      const int remaining = head_size - rot_dim;
      for (int i = item.get_local_id(2); i < num_heads * remaining;
           i += item.get_local_range(2)) {
        const int head_idx = i / remaining;
        const int dim_offset = rot_dim + (i % remaining);
        const int64_t idx =
            token_idx * input_stride + head_idx * head_stride + dim_offset;
        output[idx] = input[idx];
      }
    }
  }

 private:
  scalar_t* __restrict__ output;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ cos_ptr;
  const scalar_t* __restrict__ sin_ptr;
  const int num_tokens;
  const int num_heads;
  const int head_size;
  const int rot_dim;
  const int64_t input_stride;
  const int64_t head_stride;
  const int64_t cos_stride;
};

}  // namespace vllm

void apply_rotary_emb(
    torch::Tensor& output,   // [num_tokens, num_heads, head_size]
    torch::Tensor& input,    // [num_tokens, num_heads, head_size]
    torch::Tensor& cos,      // [num_tokens, rot_dim/2]
    torch::Tensor& sin,      // [num_tokens, rot_dim/2]
    bool is_neox) {
  int num_tokens = input.size(0);
  int num_heads = input.size(1);
  int head_size = input.size(2);
  int rot_dim = cos.size(1) * 2;

  int64_t input_stride = input.stride(0);
  int64_t head_stride = input.stride(1);
  int64_t cos_stride = cos.stride(0);

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min<int64_t>(num_heads * rot_dim / 2, 512));

  at::DeviceGuard device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "apply_rotary_emb", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        auto input_ptr = (sycl_t*)input.data_ptr<scalar_t>();
        auto output_ptr = (sycl_t*)output.data_ptr<scalar_t>();
        auto cos_ptr = (sycl_t*)cos.data_ptr<scalar_t>();
        auto sin_ptr = (sycl_t*)sin.data_ptr<scalar_t>();

        if (is_neox) {
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                vllm::apply_rotary_emb_kernel<sycl_t, true>(
                    output_ptr, input_ptr, cos_ptr, sin_ptr,
                    num_tokens, num_heads, head_size, rot_dim,
                    input_stride, head_stride, cos_stride));
          });
        } else {
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                vllm::apply_rotary_emb_kernel<sycl_t, false>(
                    output_ptr, input_ptr, cos_ptr, sin_ptr,
                    num_tokens, num_heads, head_size, rot_dim,
                    input_stride, head_stride, cos_stride));
          });
        }
      });
}
