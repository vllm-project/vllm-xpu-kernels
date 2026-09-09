#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "compact_sampling_mask_kernels.hpp"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compact_sampling_mask(
    const torch::Tensor& logits,
    const torch::Tensor& num_sampled_tokens,
    int64_t max_num_kept,
    int64_t max_compact_support) {
  CHECK_DEVICE(logits);
  CHECK_CONTIGUOUS(logits);
  TORCH_CHECK(
      logits.dim() == 2,
      "logits tensor must be 2D [num_reqs, vocab_size], but got dim ",
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
      num_sampled_tokens.dtype() == torch::kInt32,
      "num_sampled_tokens tensor must be int32, but got ",
      num_sampled_tokens.dtype());
  TORCH_CHECK(
      num_sampled_tokens.size(0) == logits.size(0),
      "num_sampled_tokens size(0) (",
      num_sampled_tokens.size(0),
      ") must match logits size(0) (",
      logits.size(0),
      ")");
  TORCH_CHECK(
      num_sampled_tokens.device() == logits.device(),
      "num_sampled_tokens and logits must be on the same device, got ",
      num_sampled_tokens.device(),
      " and ",
      logits.device());

  int num_reqs = logits.size(0);
  int vocab_size = logits.size(1);
  int real_max_num_kept = std::min(
      {static_cast<int64_t>(max_num_kept),
       static_cast<int64_t>(vocab_size),
       max_compact_support});
  int aligned_vocab_size = (vocab_size + 7) / 8;

  auto device = logits.device();

  torch::Tensor token_ids = torch::empty(
      {num_reqs, real_max_num_kept},
      torch::dtype(torch::kInt32).device(device).requires_grad(false));
  torch::Tensor packed_mask = torch::empty(
      {num_reqs, aligned_vocab_size},
      torch::dtype(torch::kUInt8).device(device).requires_grad(false));
  torch::Tensor counts = torch::empty(
      {num_reqs},
      torch::dtype(torch::kInt32).device(device).requires_grad(false));

  auto& queue = vllm::xpu::vllmGetQueue();

  CompactSamplingMaskImpl::compact_sampling_mask_kernel_launcher(
      queue,
      logits.data_ptr<float>(),
      num_sampled_tokens.data_ptr<int>(),
      token_ids.data_ptr<int>(),
      packed_mask.data_ptr<uint8_t>(),
      counts.data_ptr<int>(),
      num_reqs,
      real_max_num_kept,
      vocab_size);

  return {token_ids, packed_mask, counts};
}
