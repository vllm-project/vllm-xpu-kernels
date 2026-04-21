#include <limits>
#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "utils.h"
#include "mqa_logits_interface.h"

#include "xe_2/mqa_logits_xe2.h"

namespace {

void check_fp8_mqa_inputs(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seqlen_ks,
    const torch::Tensor& cu_seqlen_ke) {
  CHECK_DEVICE(q);
  CHECK_DEVICE(kv);
  CHECK_DEVICE(kv_scales);
  CHECK_DEVICE(weights);
  CHECK_DEVICE(cu_seqlen_ks);
  CHECK_DEVICE(cu_seqlen_ke);

  TORCH_CHECK(q.dim() == 3, "q must be [seq_len, num_heads, head_dim]");
  TORCH_CHECK(kv.dim() == 2, "kv must be [seq_len_kv, head_dim]");
  TORCH_CHECK(kv_scales.dim() == 1, "kv_scales must be [seq_len_kv]");
  TORCH_CHECK(weights.dim() == 2, "weights must be [seq_len, num_heads]");
  TORCH_CHECK(cu_seqlen_ks.dim() == 1, "cu_seqlen_ks must be [seq_len]");
  TORCH_CHECK(cu_seqlen_ke.dim() == 1, "cu_seqlen_ke must be [seq_len]");

  TORCH_CHECK(
      q.scalar_type() == at::kFloat8_e4m3fn, "q must be torch.float8_e4m3fn");
  TORCH_CHECK(
      kv.scalar_type() == at::kFloat8_e4m3fn, "kv must be torch.float8_e4m3fn");
  TORCH_CHECK(
      kv_scales.scalar_type() == at::kFloat, "kv_scales must be torch.float32");
  TORCH_CHECK(
      weights.scalar_type() == at::kFloat, "weights must be torch.float32");
  TORCH_CHECK(
      cu_seqlen_ks.scalar_type() == at::kInt,
      "cu_seqlen_ks must be torch.int32");
  TORCH_CHECK(
      cu_seqlen_ke.scalar_type() == at::kInt,
      "cu_seqlen_ke must be torch.int32");

  const int64_t seq_len = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t seq_len_kv = kv.size(0);

  TORCH_CHECK(kv.size(1) == head_dim, "kv head_dim mismatch");
  TORCH_CHECK(weights.size(0) == seq_len, "weights seq_len mismatch");
  TORCH_CHECK(weights.size(1) == num_heads, "weights num_heads mismatch");
  TORCH_CHECK(kv_scales.size(0) == seq_len_kv, "kv_scales shape mismatch");
  TORCH_CHECK(cu_seqlen_ks.size(0) == seq_len, "cu_seqlen_ks shape mismatch");
  TORCH_CHECK(cu_seqlen_ke.size(0) == seq_len, "cu_seqlen_ke shape mismatch");
}

void check_fp8_paged_mqa_inputs(
    const torch::Tensor& q_fp8,
    const torch::Tensor& kv_cache_fp8,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_tables,
    int64_t max_model_len) {
  CHECK_DEVICE(q_fp8);
  CHECK_DEVICE(kv_cache_fp8);
  CHECK_DEVICE(weights);
  CHECK_DEVICE(context_lens);
  CHECK_DEVICE(block_tables);

  TORCH_CHECK(
      q_fp8.dim() == 4, "q_fp8 must be [batch_size, next_n, heads, index_dim]");
  TORCH_CHECK(
      kv_cache_fp8.dim() == 4,
      "kv_cache_fp8 must be [num_blocks, block_size, 1, index_dim+4]");
  TORCH_CHECK(
      weights.dim() == 2, "weights must be [batch_size * next_n, heads]");
  TORCH_CHECK(context_lens.dim() == 1, "context_lens must be [batch_size]");
  TORCH_CHECK(
      block_tables.dim() == 2, "block_tables must be [batch_size, max_blocks]");

  TORCH_CHECK(
      q_fp8.scalar_type() == at::kFloat8_e4m3fn,
      "q_fp8 must be torch.float8_e4m3fn");
  TORCH_CHECK(
      kv_cache_fp8.scalar_type() == at::kByte,
      "kv_cache_fp8 must be torch.uint8 packed cache");
  TORCH_CHECK(
      weights.scalar_type() == at::kFloat, "weights must be torch.float32");
  TORCH_CHECK(
      context_lens.scalar_type() == at::kInt,
      "context_lens must be torch.int32");
  TORCH_CHECK(
      block_tables.scalar_type() == at::kInt,
      "block_tables must be torch.int32");

  TORCH_CHECK(kv_cache_fp8.size(2) == 1, "kv_cache_fp8 size(2) must be 1");

  const int64_t batch_size = q_fp8.size(0);
  const int64_t next_n = q_fp8.size(1);
  const int64_t heads = q_fp8.size(2);
  const int64_t index_dim = q_fp8.size(3);

  TORCH_CHECK(
      kv_cache_fp8.size(3) == index_dim + 4,
      "kv_cache_fp8 packed last dim must be index_dim + 4");
  TORCH_CHECK(
      weights.size(0) == batch_size * next_n, "weights first dim mismatch");
  TORCH_CHECK(weights.size(1) == heads, "weights heads mismatch");
  TORCH_CHECK(
      context_lens.size(0) == batch_size, "context_lens shape mismatch");
  TORCH_CHECK(
      block_tables.size(0) == batch_size, "block_tables shape mismatch");
  TORCH_CHECK(max_model_len > 0, "max_model_len must be > 0");
}

}  // namespace

torch::Tensor fp8_mqa_logits(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seqlen_ks,
    const torch::Tensor& cu_seqlen_ke) {
  check_fp8_mqa_inputs(q, kv, kv_scales, weights, cu_seqlen_ks, cu_seqlen_ke);

  const int64_t seq_len = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t seq_len_kv = kv.size(0);

  if (vllm::xpu::is_xe2_arch()) {
    return fp8_mqa_logits_xe2(
        q,
        kv,
        kv_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        seq_len,
        num_heads,
        head_dim,
        seq_len_kv);
  }

  TORCH_CHECK(false, "Only XE2 mqa logits kernel is supported currently.");
}

torch::Tensor fp8_paged_mqa_logits(
    const torch::Tensor& q_fp8,
    const torch::Tensor& kv_cache_fp8,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_tables,
    const c10::optional<at::Tensor>& schedule_metadata,
    int64_t max_model_len) {
  check_fp8_paged_mqa_inputs(
      q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len);
  (void)schedule_metadata;

  const int64_t batch_size = q_fp8.size(0);
  const int64_t next_n = q_fp8.size(1);
  const int64_t heads = q_fp8.size(2);
  const int64_t index_dim = q_fp8.size(3);
  const int64_t num_blocks = kv_cache_fp8.size(0);
  const int64_t block_size = kv_cache_fp8.size(1);
  const int64_t max_blocks = block_tables.size(1);

  auto kv_cache_flat =
      kv_cache_fp8.view({num_blocks, block_size * (index_dim + 4)});
  auto kv_value_u8 = kv_cache_flat.slice(1, 0, block_size * index_dim)
                         .view({num_blocks, block_size, 1, index_dim});
  auto kv_scale_u8 = kv_cache_flat.slice(1, block_size * index_dim)
                         .view({num_blocks, block_size, 1, 4});

  auto kv_fp8 = kv_value_u8.view(torch::kFloat8_e4m3fn);
  auto kv_scales_f32 = kv_scale_u8.view(torch::kFloat);

  if (vllm::xpu::is_xe2_arch()) {
    return fp8_paged_mqa_logits_xe2(
        q_fp8,
        kv_fp8,
        kv_scales_f32,
        weights,
        context_lens,
        block_tables,
        batch_size,
        next_n,
        heads,
        index_dim,
        num_blocks,
        block_size,
        max_blocks,
        max_model_len);
  }

  TORCH_CHECK(
      false, "Only XE2 paged mqa logits kernel is supported currently.");
}
