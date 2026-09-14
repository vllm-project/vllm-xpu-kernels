#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "topk_topp_sampler_kernels.hpp"

void topk_topp_sampler(
    torch::Tensor& random_sampled,
    const std::optional<torch::Tensor>& logits_to_return,
    torch::Tensor& logits,
    const std::optional<torch::Tensor>& k,
    const std::optional<torch::Tensor>& p,
    const std::string& logprobs_mode,
    torch::Tensor& seeds,
    const double lambda) {
  int batch_size = logits.size(0);
  int vocab_size = logits.size(1);

  TORCH_CHECK(
      seeds.dtype() == torch::kInt64,
      "seeds tensor must be int64, but got ",
      seeds.dtype());
  TORCH_CHECK(seeds.is_contiguous(), "seeds tensor must be contiguous");
  TORCH_CHECK(
      seeds.dim() == 1 || seeds.dim() == 2,
      "seeds tensor must have rank 1 or 2, but got ",
      seeds.dim());
  bool has_per_row_rng = seeds.dim() == 2;
  if (has_per_row_rng) {
    TORCH_CHECK(
        seeds.device() == logits.device(),
        "per-row seeds tensor must be on the same device as logits, but got ",
        seeds.device(),
        " and ",
        logits.device());
    TORCH_CHECK(
        seeds.size(0) == batch_size && seeds.size(1) == 2,
        "per-row seeds tensor must have shape [batch_size, 2], but got ",
        seeds.sizes());
  } else {
    TORCH_CHECK(
        seeds.device().is_cpu(),
        "shared seeds tensor must be on CPU, but got device: ",
        seeds.device());
    TORCH_CHECK(
        seeds.dim() == 1 && seeds.numel() == 2,
        "shared seeds tensor must have shape [2], but got ",
        seeds.sizes());
  }
  TORCH_CHECK(
      logits.dtype() == torch::kFloat32,
      "Logits tensor must be float32, but got ",
      logits.dtype())
  TORCH_CHECK(logits.is_contiguous(), "Logitss tensor must be contiguous")
  if (logits_to_return.has_value()) {
    TORCH_CHECK(
        logits_to_return->dtype() == torch::kFloat32,
        "Logits_to_return tensor must be float32, but got ",
        logits_to_return->dtype())
    TORCH_CHECK(
        logits_to_return->is_contiguous(),
        "Logits_to_return tensor must be contiguous")
  }
  TORCH_CHECK(
      random_sampled.dtype() == torch::kInt64,
      "random_sampled tensor must be int64, but got ",
      random_sampled.dtype())
  TORCH_CHECK(
      random_sampled.is_contiguous(),
      "random_sampled tensor must be contiguous")

  auto& queue = vllm::xpu::vllmGetQueue();

  const int64_t* seed_offsets = nullptr;
  int64_t seed = 0;
  int64_t offset = 0;
  if (has_per_row_rng) {
    seed_offsets = seeds.data_ptr<int64_t>();
  } else {
    auto seeds_ptr = seeds.data_ptr<int64_t>();
    seed = seeds_ptr[0];
    offset = seeds_ptr[1];
  }

  torch::Tensor buffer = torch::empty_like(logits);

#define LAUNCHER(logprobs_mode)                                          \
  TopkToppSamplerImpl::topk_topp_sampler_kernel_launcher<logprobs_mode>( \
      queue,                                                             \
      random_sampled.data_ptr<int64_t>(),                                \
      logits_to_return.has_value() ? logits_to_return->data_ptr<float>() \
                                   : nullptr,                            \
      logits.data_ptr<float>(),                                          \
      buffer.data_ptr<float>(),                                          \
      k.has_value() ? k->data_ptr<int64_t>() : nullptr,                  \
      p.has_value() ? p->data_ptr<float>() : nullptr,                    \
      seed_offsets,                                                      \
      batch_size,                                                        \
      vocab_size,                                                        \
      seed,                                                              \
      offset,                                                            \
      lambda);

  if (logprobs_mode == "raw_logits" || logprobs_mode == "raw_logprobs") {
    LAUNCHER(TopkToppSamplerImpl::LogprobsMode::default_mode)
  } else if (logprobs_mode == "processed_logits") {
    LAUNCHER(TopkToppSamplerImpl::LogprobsMode::processed_logits)
  } else if (logprobs_mode == "processed_logprobs") {
    LAUNCHER(TopkToppSamplerImpl::LogprobsMode::processed_logprobs)
  } else {
    TORCH_CHECK(false, "Unsupported logprobs_mode: ", logprobs_mode);
  }
#undef LAUNCHER
}
