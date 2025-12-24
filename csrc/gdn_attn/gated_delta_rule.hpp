#pragma once

#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace gdn{

template<typename T, int Width>
struct gated_delta_rule_kernel
{
public:
    static constexpr int group_size = 256;
    static constexpr int elems_per_item = 1;
    static constexpr int elems_per_group = group_size * elems_per_item;

    gated_delta_rule_kernel(
    ):
    {}

    static inline sycl::nd_range<2>
    get_nd_range(const int seqlen, const int input_dim) {
        const int groups_per_dim = (input_dim + elems_per_group - 1) / elems_per_group;
        sycl::range<2> local(1, group_size);
        sycl::range<2> global(seqlen, groups_per_dim);
        return sycl::nd_range<2>(global * local, local);
    }

    static inline void act_swish(float& x, float beta=1.0f){
        x = x / (1.0f + sycl::exp(-x * beta));
    }
    static inline void act_silu(float& x){
        act_swish(x, 1.0f);
    }

  [[sycl::reqd_sub_group_size(32)]] void
  operator()(sycl::nd_item<2> item) const {

  }
private:

};

template<typename T>
void kernel_launcher(
    sycl::queue& queue,
){
    using KERNEL = gated_delta_rule_kernel<T>;
    auto range = KERNEL::get_nd_range();
    queue.submit([&](sycl::handler& cgh) {
        KERNEL task(
        );
        cgh.parallel_for(range, task);
    });
}

void causal_conv1d(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& projected_states_ba,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& initial_state,
    const torch::Tensor& ssm_state,
    const torch::Tensor& dt_bias,
    const torch::Tensor& dt_bias,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
){
    const int batch_size = query_start_loc.size(0) - 1;

#define KERNEL_LAUNCHER(scalar_t)       \
    kernel_launcher<scalar_t>(       \
        queue,       \
    );

    if(input.scalar_type() == at::kBFloat16){
        using scalar_t = sycl::ext::oneapi::bfloat16;
        KERNEL_LAUNCHER(scalar_t)
    }else if(input.scalar_type() == at::kHalf){
        using scalar_t = sycl::half;
        KERNEL_LAUNCHER(scalar_t, width)
    }else{
        using scalar_t = float;
        KERNEL_LAUNCHER(scalar_t, width)
    }
}

}
