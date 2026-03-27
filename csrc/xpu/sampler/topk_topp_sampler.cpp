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
    torch::Tensor& seeds,  // should on CPU
    const double lambda) {
  int batch_size = logits.size(0);
  int vocal_size = logits.size(1);

  auto& queue = vllm::xpu::vllmGetQueue();

  auto seeds_ptr = reinterpret_cast<int64_t*>(seeds.data_ptr());
  int64_t seed = seeds_ptr[0];
  int64_t offset = seeds_ptr[1];

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
      batch_size,                                                        \
      vocal_size,                                                        \
      seed,                                                              \
      offset,                                                            \
      lambda);

  TopkToppSamplerImpl::LogprobsMode logprobs_mode_enum;
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
