#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "topk_topp_sampler_kernels.hpp"

void exponential_2d_(
    torch::Tensor& tensor,
    int64_t seed,
    int64_t offset,
    const double lambda) {

    auto dtype = tensor.dtype();
    int batch_size = tensor.size(0);
    int vocal_size = tensor.size(1);

    auto& queue = vllm::xpu::vllmGetQueue();

#define EXPONENTIAL_2D_LAUNCHER(T) \
    using scalar_t = T; \
    TopkToppSamplerImpl::exponential_2d_kernel_launcher( \
        queue, \
        reinterpret_cast<T*>(tensor.data_ptr()), \
        batch_size, \
        vocal_size, \
        seed, \
        offset, \
        lambda);
    
    if(dtype == torch::kFloat32){
        EXPONENTIAL_2D_LAUNCHER(float)
    } else if(dtype == torch::kFloat16){
        EXPONENTIAL_2D_LAUNCHER(sycl::half)
    } else if(dtype == torch::kBFloat16){
        EXPONENTIAL_2D_LAUNCHER(sycl::ext::oneapi::bfloat16)
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", dtype);
    }
}

void exponential_2d_(
    torch::Tensor& tensor,
    torch::Tensor& seeds, // should on CPU
    const double lambda) {

    auto seeds_ptr = reinterpret_cast<int64_t*>(seeds.data_ptr());
    int64_t seed = seeds_ptr[0];
    int64_t offset = seeds_ptr[1];
    exponential_2d_(tensor, seed, offset, lambda);
}

void topk_topp_sampler(
    torch::Tensor& random_sampled,
    const std::optional<torch::Tensor>& processed_logprobs,
    torch::Tensor& logits,
    const std::optional<torch::Tensor>& k,
    const std::optional<torch::Tensor>& p,
    const std::string& logprobs_mode,
    torch::Tensor& seeds, // should on CPU
    const double lambda) {

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

    auto seeds_ptr = reinterpret_cast<int64_t*>(seeds.data_ptr());
    int64_t seed = seeds_ptr[0];
    int64_t offset = seeds_ptr[1];

    TopkToppSamplerImpl::topk_topp_sampler_kernel_launcher(
        queue,
        random_sampled.data_ptr<int64_t>(),
        processed_logprobs.has_value() ? processed_logprobs->data_ptr<float>() : nullptr,
        logits.data_ptr<float>(),
        k.has_value() ? k->data_ptr<int64_t>() : nullptr,
        p.has_value() ? p->data_ptr<float>() : nullptr,
        batch_size,
        vocal_size,
        logprobs_mode_enum,
        seed,
        offset,
        lambda);
}
