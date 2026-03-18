#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "topk_topp_sampler_kernels.hpp"

void topk_topp_sampler(
    torch::Tensor& random_sampled,
    const std::optional<torch::Tensor>& processed_logprobs,
    const torch::Tensor& logits,
    const std::optional<torch::Tensor>& k,
    const std::optional<torch::Tensor>& p,
    const std::string& logprobs_mode) {

    int batch_size = logits.size(0);
    int vocal_size = logits.size(1);

    auto& queue = vllm::xpu::vllmGetQueue();

    TopkToppSamplerImpl::LogprobsMode logprobs_mode_enum;
    if(logprobs_mode == "raw_logits"){
        logprobs_mode_enum = TopkToppSamplerImpl::LogprobsMode::raw_logits;
    } else if(logprobs_mode == "raw_logprobs"){
        logprobs_mode_enum = TopkToppSamplerImpl::LogprobsMode::raw_logprobs;
    } else if(logprobs_mode == "processed_logits"){
        logprobs_mode_enum = TopkToppSamplerImpl::LogprobsMode::processed_logits;
    } else if(logprobs_mode == "processed_logprobs"){
        logprobs_mode_enum = TopkToppSamplerImpl::LogprobsMode::processed_logprobs;
    } else {
        TORCH_CHECK(false, "Unsupported logprobs_mode: ", logprobs_mode);
    }

    TopkToppSamplerImpl::kernel_launcher(
        queue,
        random_sampled.data_ptr<int64_t>(),
        processed_logprobs.has_value() ? processed_logprobs->data_ptr<float>() : nullptr,
        logits.data_ptr<float>(),
        k.has_value() ? k->data_ptr<int64_t>() : nullptr,
        p.has_value() ? p->data_ptr<float>() : nullptr,
        batch_size,
        vocal_size,
        logprobs_mode_enum);


}