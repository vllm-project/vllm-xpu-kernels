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

void top_k_mask_logits(
    torch::Tensor& logits,
    const std::optional<torch::Tensor>& k) {

    TORCH_CHECK(k.has_value(), "k must be provided for top_k_mask_logits");

    int batch_size = logits.size(0);
    int vocal_size = logits.size(1);

    auto& queue = vllm::xpu::vllmGetQueue();

    TopkToppSamplerImpl::top_k_mask_logits_kernel_launcher(
        queue,
        logits.data_ptr<float>(),
        k->data_ptr<int64_t>(),
        batch_size,
        vocal_size);
}


void topk_topp_sampler(
    torch::Tensor& random_sampled,
    const std::optional<torch::Tensor>& logits_to_return,
    torch::Tensor& logits,
    const std::optional<torch::Tensor>& k,
    const std::optional<torch::Tensor>& p,
    const std::string& logprobs_mode,
    torch::Tensor& seeds, // should on CPU
    const double lambda) {

    int batch_size = logits.size(0);
    int vocal_size = logits.size(1);

    auto& queue = vllm::xpu::vllmGetQueue();

    auto seeds_ptr = reinterpret_cast<int64_t*>(seeds.data_ptr());
    int64_t seed = seeds_ptr[0];
    int64_t offset = seeds_ptr[1];

#define LAUNCHER(logprobs_mode) \
    TopkToppSamplerImpl::topk_topp_sampler_kernel_launcher<logprobs_mode>( \
        queue, \
        random_sampled.data_ptr<int64_t>(), \
        logits_to_return.has_value() ? logits_to_return->data_ptr<float>() : nullptr, \
        logits.data_ptr<float>(), \
        k.has_value() ? k->data_ptr<int64_t>() : nullptr, \
        p.has_value() ? p->data_ptr<float>() : nullptr, \
        batch_size, \
        vocal_size, \
        seed, \
        offset, \
        lambda);

    TopkToppSamplerImpl::LogprobsMode logprobs_mode_enum;
    if(logprobs_mode == "raw_logits" || logprobs_mode == "raw_logprobs"){
        LAUNCHER(TopkToppSamplerImpl::LogprobsMode::default_mode)
    } else if(logprobs_mode == "processed_logits"){
        LAUNCHER(TopkToppSamplerImpl::LogprobsMode::processed_logits)
    } else if(logprobs_mode == "processed_logprobs"){
        LAUNCHER(TopkToppSamplerImpl::LogprobsMode::processed_logprobs)
    } else {
        TORCH_CHECK(false, "Unsupported logprobs_mode: ", logprobs_mode);
    }
}
