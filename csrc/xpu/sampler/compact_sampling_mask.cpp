#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "compact_sampling_mask_kernels.hpp"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compact_sampling_mask(
    const torch::Tensor& logits,
    const torch::Tensor& cu_num_logits,
    const torch::Tensor& num_sampled_tokens,
    int64_t max_num_kept,
    int64_t max_compact_support,
    int64_t rows_per_request) {
  CHECK_DEVICE(logits);
  CHECK_CONTIGUOUS(logits);
  TORCH_CHECK(
      logits.dim() == 2,
      "logits tensor must be 2D [num_source_rows, vocab_size], but got dim ",
      logits.dim());
  TORCH_CHECK(
      logits.dtype() == torch::kFloat32,
      "logits tensor must be float32, but got ",
      logits.dtype());

  CHECK_DEVICE(num_sampled_tokens);
  CHECK_CONTIGUOUS(num_sampled_tokens);
  TORCH_CHECK(
      num_sampled_tokens.dim() == 1,
      "num_sampled_tokens tensor must be 1D [num_reqs], but got dim ",
      num_sampled_tokens.dim());
  TORCH_CHECK(
      num_sampled_tokens.dtype() == torch::kInt32 ||
          num_sampled_tokens.dtype() == torch::kInt64,
      "num_sampled_tokens tensor must be int32 or int64, but got ",
      num_sampled_tokens.dtype());

  CHECK_DEVICE(cu_num_logits);
  CHECK_CONTIGUOUS(cu_num_logits);
  TORCH_CHECK(
      cu_num_logits.dim() == 1,
      "cu_num_logits tensor must be 1D [num_reqs + 1], but got dim ",
      cu_num_logits.dim());
  TORCH_CHECK(
      cu_num_logits.dtype() == torch::kInt32 ||
          cu_num_logits.dtype() == torch::kInt64,
      "cu_num_logits tensor must be int32 or int64, but got ",
      cu_num_logits.dtype());
  // cu_num_logits is the CSR prefix sum of source-logits rows, so it holds
  // num_reqs + 1 entries; only the first num_reqs are indexed by the kernel.
  TORCH_CHECK(
      cu_num_logits.size(0) >= num_sampled_tokens.size(0),
      "cu_num_logits size(0) (",
      cu_num_logits.size(0),
      ") must be >= num_sampled_tokens size(0) (",
      num_sampled_tokens.size(0),
      ")");
  TORCH_CHECK(
      rows_per_request >= 1,
      "rows_per_request must be >= 1, but got ",
      rows_per_request);
  TORCH_CHECK(
      num_sampled_tokens.device() == logits.device() &&
          cu_num_logits.device() == logits.device(),
      "cu_num_logits, num_sampled_tokens and logits must be on the same "
      "device, got ",
      cu_num_logits.device(),
      ", ",
      num_sampled_tokens.device(),
      " and ",
      logits.device());

  int num_reqs = num_sampled_tokens.size(0);
  int vocab_size = logits.size(1);
  int rows_per_request_ = static_cast<int>(rows_per_request);
  int num_output_rows = num_reqs * rows_per_request_;
  int real_max_num_kept = std::min(
      {static_cast<int64_t>(max_num_kept),
       static_cast<int64_t>(vocab_size),
       max_compact_support});
  int aligned_vocab_size = (vocab_size + 7) / 8;

  auto device = logits.device();

  torch::Tensor token_ids = torch::empty(
      {num_output_rows, real_max_num_kept},
      torch::dtype(torch::kInt32).device(device).requires_grad(false));
  torch::Tensor packed_mask = torch::empty(
      {num_output_rows, aligned_vocab_size},
      torch::dtype(torch::kUInt8).device(device).requires_grad(false));
  torch::Tensor counts = torch::empty(
      {num_output_rows},
      torch::dtype(torch::kInt32).device(device).requires_grad(false));

  auto& queue = vllm::xpu::vllmGetQueue();

  // num_sampled_tokens and cu_num_logits may carry different integer dtypes
  // (int32/int64), so resolve each index type independently.
  auto launch = [&](auto num_sampled_dummy, auto cu_dummy) {
    using index_t = decltype(num_sampled_dummy);
    using cu_index_t = decltype(cu_dummy);
    CompactSamplingMaskImpl::
        compact_sampling_mask_kernel_launcher<index_t, cu_index_t>(
            queue,
            logits.data_ptr<float>(),
            cu_num_logits.data_ptr<cu_index_t>(),
            num_sampled_tokens.data_ptr<index_t>(),
            token_ids.data_ptr<int>(),
            packed_mask.data_ptr<uint8_t>(),
            counts.data_ptr<int>(),
            num_reqs,
            rows_per_request_,
            real_max_num_kept,
            vocab_size);
  };

  const bool sampled_is_int64 =
      num_sampled_tokens.scalar_type() == torch::kInt64;
  const bool cu_is_int64 = cu_num_logits.scalar_type() == torch::kInt64;
  if (sampled_is_int64) {
    if (cu_is_int64) {
      launch(int64_t{}, int64_t{});
    } else {
      launch(int64_t{}, int32_t{});
    }
  } else {
    if (cu_is_int64) {
      launch(int32_t{}, int64_t{});
    } else {
      launch(int32_t{}, int32_t{});
    }
  }

  return {token_ids, packed_mask, counts};
}
