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

template <typename scalar_t, bool IS_NEOX>
class multimodal_rotary_embedding_kernel {
 public:
  multimodal_rotary_embedding_kernel(
      const int64_t* __restrict__ positions_,
      scalar_t* __restrict__ query_,
      scalar_t* __restrict__ key_,
      const scalar_t* __restrict__ cos_sin_cache_,
      const int* mrope_section_data,
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
    // Pre-compute cumulative section boundaries for fast lookup.
    int cumsum = 0;
    for (int s = 0; s < num_mrope_sections_; ++s) {
      mrope_section[s] = mrope_section_data[s];
      cumsum += mrope_section_data[s];
      section_end[s] = cumsum;
    }
  }

  static constexpr int VEC_SIZE = 4;
  // GPT-J vec4 processes 2 rotation offsets per call.
  static constexpr int GPTJ_PAIRS_PER_VEC = 2;

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    const int token_idx = item_ct1.get_group(2);
    const int embed_dim = rot_dim / 2;
    const int local_id = item_ct1.get_local_id(2);
    const int local_range = item_ct1.get_local_range(2);

    // Pre-fetch per-section source pointers for this token.
    // All threads in the WG share the same token → same positions →
    // L1-cached across threads.
    const scalar_t* section_src[MROPE_MAX_SECTIONS];
    for (int s = 0; s < num_mrope_sections; ++s) {
      const int64_t pos = positions[s * num_tokens + token_idx];
      section_src[s] = cos_sin_cache + pos * rot_dim;
    }

    // ── 2D grid mode: each WG handles one (token, head) pair ──
    // Active when grid dim 1 > 1 (small num_tokens path).
    if (item_ct1.get_group_range(1) > 1) {
      const int head_group = item_ct1.get_group(1);
      const bool is_query_head = (head_group < num_heads);
      const int head_idx =
          is_query_head ? head_group : (head_group - num_heads);
      if (!is_query_head && key == nullptr) return;

      scalar_t* data = is_query_head ? query : key;
      const int64_t stride = is_query_head ? query_stride : key_stride;
      const int64_t token_head = token_idx * stride + head_idx * head_stride;
      rotate_single_head(
          data + token_head, embed_dim, local_id, local_range, section_src);
      return;
    }

    // ── 1D grid mode: each WG handles all heads for one token ──
    rotate_all_heads(
        query,
        query_stride,
        num_heads,
        embed_dim,
        local_id,
        local_range,
        token_idx,
        section_src);
    if (key != nullptr) {
      rotate_all_heads(
          key,
          key_stride,
          num_kv_heads,
          embed_dim,
          local_id,
          local_range,
          token_idx,
          section_src);
    }
  }

 private:
  // ── Section lookup: find which section owns `rot_offset` ──
  // Returns index s such that section_end[s-1] <= rot_offset < section_end[s].
  // At most num_mrope_sections-1 comparisons (typically ≤ 3).
  inline int find_section(int rot_offset) const {
    int s = 0;
    for (; s < num_mrope_sections - 1; ++s) {
      if (rot_offset < section_end[s]) break;
    }
    return s;
  }

  // ── Rotate one head (used by 2D grid mode) ──
  // All work-items in the WG cooperatively process the embed_dim of
  // a single head, with no loops over heads.
  inline void rotate_single_head(
      scalar_t* data,
      int embed_dim,
      int local_id,
      int local_range,
      const scalar_t* section_src[MROPE_MAX_SECTIONS]) const {
    if constexpr (IS_NEOX) {
      const int vph = embed_dim / VEC_SIZE;
      for (int v = local_id; v < vph; v += local_range) {
        const int rot_offset = v * VEC_SIZE;
        const int s = find_section(rot_offset);
        // Guard: if the vec4 straddles a section boundary,
        // fall back to scalar to use the correct cos/sin for each.
        if (find_section(rot_offset + VEC_SIZE - 1) != s) {
          for (int j = 0; j < VEC_SIZE; ++j) {
            const int sj = find_section(rot_offset + j);
            apply_token_rotary_embedding<scalar_t, true>(
                data,
                data,
                section_src[sj],
                section_src[sj] + embed_dim,
                rot_offset + j,
                embed_dim);
          }
        } else {
          apply_token_rotary_embedding_vec<scalar_t, VEC_SIZE>(
              data,
              data,
              section_src[s],
              section_src[s] + embed_dim,
              rot_offset,
              embed_dim);
        }
      }
      // Scalar tail: embed_dim % VEC_SIZE remaining elements.
      const int tail_start = vph * VEC_SIZE;
      for (int r = tail_start + local_id; r < embed_dim; r += local_range) {
        const int s = find_section(r);
        apply_token_rotary_embedding<scalar_t, true>(
            data,
            data,
            section_src[s],
            section_src[s] + embed_dim,
            r,
            embed_dim);
      }
    } else {
      const int vph = embed_dim / GPTJ_PAIRS_PER_VEC;
      for (int v = local_id; v < vph; v += local_range) {
        const int rot_offset = v * GPTJ_PAIRS_PER_VEC;
        const int s = find_section(rot_offset);
        // Guard: if the two offsets straddle a section boundary,
        // fall back to scalar to use the correct cos/sin for each.
        if (rot_offset + 1 < embed_dim && find_section(rot_offset + 1) != s) {
          apply_token_rotary_embedding<scalar_t, false>(
              data,
              data,
              section_src[s],
              section_src[s] + embed_dim,
              rot_offset,
              embed_dim);
          const int s2 = find_section(rot_offset + 1);
          apply_token_rotary_embedding<scalar_t, false>(
              data,
              data,
              section_src[s2],
              section_src[s2] + embed_dim,
              rot_offset + 1,
              embed_dim);
        } else {
          apply_token_rotary_embedding_gptj_vec4<scalar_t>(
              data,
              data,
              section_src[s],
              section_src[s] + embed_dim,
              rot_offset);
        }
      }
      // Scalar tail: if embed_dim is odd, handle last element.
      if (embed_dim % GPTJ_PAIRS_PER_VEC != 0 && local_id == 0) {
        const int s = find_section(embed_dim - 1);
        apply_token_rotary_embedding<scalar_t, false>(
            data,
            data,
            section_src[s],
            section_src[s] + embed_dim,
            embed_dim - 1,
            embed_dim);
      }
    }
  }

  // ── Rotate all heads of one Q/K tensor (used by 1D grid mode) ──
  // Distributes head × embed_dim work items across the WG.
  inline void rotate_all_heads(
      scalar_t* data,
      int64_t data_stride,
      int n_heads,
      int embed_dim,
      int local_id,
      int local_range,
      int token_idx,
      const scalar_t* section_src[MROPE_MAX_SECTIONS]) const {
    if constexpr (IS_NEOX) {
      const int vph = embed_dim / VEC_SIZE;
      const int n_vecs = n_heads * vph;
      for (int i = local_id; i < n_vecs; i += local_range) {
        const int head_idx = i / vph;
        const int rot_offset = (i % vph) * VEC_SIZE;
        const int s = find_section(rot_offset);
        const int64_t token_head =
            token_idx * data_stride + head_idx * head_stride;
        // Guard: if the vec4 straddles a section boundary,
        // fall back to scalar to use the correct cos/sin for each.
        if (find_section(rot_offset + VEC_SIZE - 1) != s) {
          for (int j = 0; j < VEC_SIZE; ++j) {
            const int sj = find_section(rot_offset + j);
            apply_token_rotary_embedding<scalar_t, true>(
                data + token_head,
                data + token_head,
                section_src[sj],
                section_src[sj] + embed_dim,
                rot_offset + j,
                embed_dim);
          }
        } else {
          apply_token_rotary_embedding_vec<scalar_t, VEC_SIZE>(
              data + token_head,
              data + token_head,
              section_src[s],
              section_src[s] + embed_dim,
              rot_offset,
              embed_dim);
        }
      }
      // Scalar tail for embed_dim % VEC_SIZE != 0.
      const int tail_start = vph * VEC_SIZE;
      if (tail_start < embed_dim) {
        const int tail_width = embed_dim - tail_start;
        const int n_tail = n_heads * tail_width;
        for (int i = local_id; i < n_tail; i += local_range) {
          const int head_idx = i / tail_width;
          const int rot_offset = tail_start + i % tail_width;
          const int s = find_section(rot_offset);
          const int64_t token_head =
              token_idx * data_stride + head_idx * head_stride;
          apply_token_rotary_embedding<scalar_t, true>(
              data + token_head,
              data + token_head,
              section_src[s],
              section_src[s] + embed_dim,
              rot_offset,
              embed_dim);
        }
      }
    } else {
      const int vph = embed_dim / GPTJ_PAIRS_PER_VEC;
      const int n_vecs = n_heads * vph;
      for (int i = local_id; i < n_vecs; i += local_range) {
        const int head_idx = i / vph;
        const int rot_offset = (i % vph) * GPTJ_PAIRS_PER_VEC;
        const int s = find_section(rot_offset);
        const int64_t token_head =
            token_idx * data_stride + head_idx * head_stride;
        // Guard: section boundary crossing.
        if (rot_offset + 1 < embed_dim && find_section(rot_offset + 1) != s) {
          apply_token_rotary_embedding<scalar_t, false>(
              data + token_head,
              data + token_head,
              section_src[s],
              section_src[s] + embed_dim,
              rot_offset,
              embed_dim);
          const int s2 = find_section(rot_offset + 1);
          apply_token_rotary_embedding<scalar_t, false>(
              data + token_head,
              data + token_head,
              section_src[s2],
              section_src[s2] + embed_dim,
              rot_offset + 1,
              embed_dim);
        } else {
          apply_token_rotary_embedding_gptj_vec4<scalar_t>(
              data + token_head,
              data + token_head,
              section_src[s],
              section_src[s] + embed_dim,
              rot_offset);
        }
      }
      // Scalar tail for odd embed_dim.
      if (embed_dim % GPTJ_PAIRS_PER_VEC != 0) {
        for (int h = local_id; h < n_heads; h += local_range) {
          const int s = find_section(embed_dim - 1);
          const int64_t token_head = token_idx * data_stride + h * head_stride;
          apply_token_rotary_embedding<scalar_t, false>(
              data + token_head,
              data + token_head,
              section_src[s],
              section_src[s] + embed_dim,
              embed_dim - 1,
              embed_dim);
        }
      }
    }
  }

  const int64_t* __restrict__ positions;
  scalar_t* __restrict__ query;
  scalar_t* __restrict__ key;
  const scalar_t* __restrict__ cos_sin_cache;
  int mrope_section[MROPE_MAX_SECTIONS];
  int section_end[MROPE_MAX_SECTIONS];
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
// Used by models such as Qwen2-VL / Qwen3-VL that need per-section position
// encoding (temporal / height / width).
//
// positions      : [num_mrope_sections, num_tokens]  int64, on device
// query          : [num_tokens, num_heads * head_size]  or
//                  [num_tokens, num_heads, head_size]
// key            : same shapes as query but kv_heads, or nullopt
// cos_sin_cache  : [max_position, rot_dim]
// mrope_section  : host int list [num_mrope_sections], values in embed_dim
//                  units summing to rot_dim / 2

namespace {

// Align `v` up to the next multiple of `alignment`.
inline int64_t align_up(int64_t v, int64_t alignment) {
  return ((v + alignment - 1) / alignment) * alignment;
}

}  // namespace

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

  if (num_tokens == 0) return;

  TORCH_CHECK(
      static_cast<int>(mrope_section.size()) == num_mrope_sections,
      "mrope_section length must equal positions.size(0)");
  TORCH_CHECK(
      num_mrope_sections <= vllm::MROPE_MAX_SECTIONS,
      "num_mrope_sections exceeds MROPE_MAX_SECTIONS=",
      vllm::MROPE_MAX_SECTIONS);

  const int query_hidden_size = query.numel() / num_tokens;
  const int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  const int num_heads = query_hidden_size / head_size;
  const int num_kv_heads =
      key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  const int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim % 2 == 0, "rot_dim must be even, got ", rot_dim);
  TORCH_CHECK(rot_dim > 0, "rot_dim must be positive");

  // Strides.
  const int64_t query_stride = query.stride(0);
  const int64_t key_stride = key.has_value() ? key->stride(0) : 0;
  const int64_t head_stride = (query.dim() == 3) ? query.stride(-2) : head_size;

  // Ensure positions is contiguous for raw pointer arithmetic.
  at::Tensor positions_contig = positions.contiguous();

  // Verify sections sum to embed_dim (= rot_dim / 2).
  int mrope_section_arr[vllm::MROPE_MAX_SECTIONS] = {};
  int section_sum = 0;
  for (int s = 0; s < num_mrope_sections; ++s) {
    mrope_section_arr[s] = static_cast<int>(mrope_section[s]);
    section_sum += mrope_section_arr[s];
  }
  TORCH_CHECK(
      section_sum == rot_dim / 2,
      "mrope_section values must sum to rot_dim / 2 (embed_dim=",
      rot_dim / 2,
      "), but got ",
      section_sum);

  // ── Grid and block configuration ──
  // NeoX vec4 processes 4 rot_offsets/call; GPT-J vec4 processes 2.
  const int effective_vec_size = is_neox ? 4 : 2;
  const int vecs_per_head = (rot_dim / 2) / effective_vec_size;
  const int total_head_groups =
      num_heads + (key.has_value() ? num_kv_heads : 0);

  // For small num_tokens use 2D grid: one WG per (token, head).
  // Eliminates head loops and maximises EU occupancy.
  constexpr int64_t SMALL_TOKEN_THRESHOLD = 128;
  sycl::range<3> grid(1, 1, 1);
  sycl::range<3> block(1, 1, 1);

  if (num_tokens <= SMALL_TOKEN_THRESHOLD) {
    grid = sycl::range<3>(1, total_head_groups, num_tokens);
    block = sycl::range<3>(
        1,
        1,
        std::min<int64_t>(
            align_up(std::max<int64_t>(vecs_per_head, 32), 32), 512));
  } else {
    grid = sycl::range<3>(1, 1, num_tokens);
    const int nk_vecs =
        (key.has_value() ? num_kv_heads : num_heads) * vecs_per_head;
    block = sycl::range<3>(
        1,
        1,
        std::min<int64_t>(align_up(std::max<int64_t>(nk_vecs, 32), 32), 512));
  }

  // ── Launch kernel ──
  auto positions_ptr = positions_contig.data_ptr<int64_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_ptr = key.has_value() ? key->data_ptr<scalar_t>() : nullptr;
  auto cos_sin_cache_ptr = cos_sin_cache.data_ptr<scalar_t>();

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
    torch::Tensor& positions,
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox,
    std::vector<int64_t> mrope_section) {
  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "multimodal_rotary_embedding", [&] {
        call_multimodal_rotary_embedding_kernel<scalar_t>(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
            mrope_section);
      });
}