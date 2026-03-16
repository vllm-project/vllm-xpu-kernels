#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"
#include "rotary_embedding.hpp"
#include <cmath>
#include <c10/macros/Macros.h>

namespace vllm {

// Maximum number of M-RoPE sections supported (e.g. 3 for Qwen2-VL:
// temporal / height / width).
constexpr int MROPE_MAX_SECTIONS = 4;
// Maximum rot_dim supported for the merged cache buffer (cos + sin).
constexpr int MROPE_MAX_ROT_DIM = 512;

template <typename scalar_t, bool IS_NEOX>
class multimodal_rotary_embedding_kernel {
 public:
  multimodal_rotary_embedding_kernel(
      const int64_t* __restrict__ positions_,  // [num_mrope_sections,
                                               // num_tokens]
      scalar_t* __restrict__ query_,
      scalar_t* __restrict__ key_,
      const scalar_t* __restrict__ cos_sin_cache_,  // [max_position, rot_dim]
      const int* mrope_section_data,  // host array [num_mrope_sections]
      const int num_mrope_sections_,
      const int num_tokens_,
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
        num_mrope_sections(num_mrope_sections_),
        num_tokens(num_tokens_),
        rot_dim(rot_dim_),
        query_stride(query_stride_),
        key_stride(key_stride_),
        head_stride(head_stride_),
        num_heads(num_heads_),
        num_kv_heads(num_kv_heads_),
        head_size(head_size_) {
    for (int s = 0; s < num_mrope_sections_; ++s)
      mrope_section[s] = mrope_section_data[s];
  }

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    // Each work-group handles one token.
    const int token_idx = item_ct1.get_group(2);
    const int embed_dim = rot_dim / 2;

    // Build cos/sin cache for this token (private memory per thread).
    scalar_t merged_cache[MROPE_MAX_ROT_DIM];  // [cos | sin], size = rot_dim
    int cumsum = 0;
    for (int s = 0; s < num_mrope_sections; ++s) {
      const int lo = cumsum;
      const int hi = lo + mrope_section[s];
      cumsum = hi;
      const int64_t pos = positions[s * num_tokens + token_idx];
      const scalar_t* src = cos_sin_cache + pos * rot_dim;
      for (int r = lo; r < hi; ++r) {
        merged_cache[r]             = src[r];              // cos slice
        merged_cache[embed_dim + r] = src[embed_dim + r];  // sin slice
      }
    }
    const scalar_t* cache_ptr = merged_cache;

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
  const int64_t* __restrict__ positions;
  scalar_t* __restrict__ query;
  scalar_t* __restrict__ key;
  const scalar_t* __restrict__ cos_sin_cache;
  int mrope_section[MROPE_MAX_SECTIONS];  // embedded, copied from host list
  const int num_mrope_sections;
  const int num_tokens;
  const int rot_dim;
  const int64_t query_stride;
  const int64_t key_stride;
  const int64_t head_stride;
  const int num_heads;
  const int num_kv_heads;
  const int head_size;
};

}  // namespace vllm


// ── Multi-Modal Rotary Embedding (M-RoPE) ──────────────────────────────────
// Used by models such as Qwen2-VL that need per-section position encoding.
//
// positions      : [num_mrope_sections, num_tokens]  int64, on device
// query          : [num_tokens, num_heads * head_size] or
//                  [num_tokens, num_heads, head_size]
// key            : same shapes as query but kv_heads, or nullopt
// cos_sin_cache  : [max_position, rot_dim]
// mrope_section  : [num_mrope_sections]  int32, on device;
//                  values in embed_dim units summing to rot_dim / 2

template <typename scalar_t>
void call_multimodal_rotary_embedding_kernel(
    torch::Tensor& positions,
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox,
    const std::vector<int64_t>& mrope_section) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  TORCH_CHECK(
      positions.dim() == 2,
      "positions must have shape [num_mrope_sections, num_tokens]");
  const int num_mrope_sections = positions.size(0);
  const int64_t num_tokens = positions.size(1);

  TORCH_CHECK(
      (int)mrope_section.size() == num_mrope_sections,
      "mrope_section length must equal positions.size(0)");
  TORCH_CHECK(
      num_mrope_sections <= vllm::MROPE_MAX_SECTIONS,
      "num_mrope_sections exceeds MROPE_MAX_SECTIONS=",
      vllm::MROPE_MAX_SECTIONS);

  const int query_hidden_size = query.numel() / num_tokens;
  const int key_hidden_size =
      key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  const int num_heads = query_hidden_size / head_size;
  const int num_kv_heads =
      key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  const int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(
      rot_dim <= vllm::MROPE_MAX_ROT_DIM,
      "rot_dim exceeds MROPE_MAX_ROT_DIM=",
      vllm::MROPE_MAX_ROT_DIM,
      ", got rot_dim=",
      rot_dim);

  // query is always [num_tokens, ...] in the M-RoPE path.
  const int64_t query_stride = query.stride(0);
  const int64_t key_stride = key.has_value() ? key->stride(0) : 0;
  const int query_ndim = query.dim();
  // For [num_tokens, num_heads, head_size] use stride(-2), else head_size.
  const int64_t head_stride =
      (query_ndim == 3) ? query.stride(-2) : head_size;

  // Ensure positions is contiguous so that raw pointer arithmetic
  // s * num_tokens + t correctly addresses positions[s, t].
  at::Tensor positions_contig = positions.contiguous();
  auto positions_ptr = positions_contig.data_ptr<int64_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_ptr = key.has_value() ? key->data_ptr<scalar_t>() : nullptr;
  auto cos_sin_cache_ptr = cos_sin_cache.data_ptr<scalar_t>();

  // Convert int64 list to int array for the kernel.
  int mrope_section_arr[vllm::MROPE_MAX_SECTIONS] = {};
  for (int s = 0; s < num_mrope_sections; ++s)
    mrope_section_arr[s] = static_cast<int>(mrope_section[s]);

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min<int64_t>(num_heads * rot_dim / 2, 512));

  at::DeviceGuard device_guard(query.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  if (is_neox) {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::multimodal_rotary_embedding_kernel<sycl_t, true>(
              positions_ptr,
              (sycl_t*)query_ptr,
              (sycl_t*)key_ptr,
              (sycl_t*)cos_sin_cache_ptr,
              mrope_section_arr,
              num_mrope_sections,
              num_tokens,
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
          vllm::multimodal_rotary_embedding_kernel<sycl_t, false>(
              positions_ptr,
              (sycl_t*)query_ptr,
              (sycl_t*)key_ptr,
              (sycl_t*)cos_sin_cache_ptr,
              mrope_section_arr,
              num_mrope_sections,
              num_tokens,
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

void multimodal_rotary_embedding(
    torch::Tensor& positions,           // [num_mrope_sections, num_tokens]
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,       // [max_position, rot_dim]
    bool is_neox,
    std::vector<int64_t> mrope_section) // host int list [num_mrope_sections]
{
  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "multimodal_rotary_embedding", [&] {
        call_multimodal_rotary_embedding_kernel<scalar_t>(
            positions, query, key, head_size, cos_sin_cache, is_neox,
            mrope_section);
  });
}