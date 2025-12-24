#pragma once

#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace gdn{

template<typename T, int Width>
struct gated_delta_rule_kernel
{
public:
    static constexpr int sub_group_size = 32;
    static constexpr int group_size = 256;
    static constexpr int sg_per_group = group_size / sub_group_size;

    gated_delta_rule_kernel(
    ):
    {}

    static inline sycl::nd_range<2>
    get_nd_range(const int batch_size, const int num_heads) {
        sycl::range<3> local(1, 1, group_size);
        sycl::range<3> global(batch_size, num_heads, v_dim / sg_per_group);
        return sycl::nd_range<2>(global * local, local);
    }

    static inline float act_sigmiod(float& x){
        return 1.0f / (1.0f + sycl::exp(-x));
    }

    static inline float act_softplus(float& x){
        return sycl::log(1.0f + sycl::exp(x));
    }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    int batch_id = item.get_group(0);
    int num_heads_id = item.get_group(1);

    auto sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    int v_dim_id = sg_id;
    int k_dim_loop = k_dim / sub_group_size;

    float A_log_local = A_log[num_heads_id];
    float dt_bias_local = dt_bias[num_heads_id];
    A_log_local = sycl::exp(A_log_local);

    // q [tokens, num_k_heads, head_k_dim]
    // k [tokens, num_k_heads, head_k_dim]
    // projected_states_ba; [tokens, num_k_heads, 2 * num_v_heads / num_k_heads]
    // b [tokens, num_k_heads, num_v_heads / num_k_heads]
    // a [tokens, num_k_heads, num_v_heads / num_k_heads]
    // v [tokens, num_k_heads, head_v_dim *  num_v_heads / num_k_heads]
    // --->>
    // b [tokens, num_v_heads]
    // a [tokens, num_v_heads]
    // v [tokens, num_v_heads, head_v_dim]
    // repeat_interleave(num_k_heads) == num_v_heads

    float state_local[4];  // 128 / 32: k_dim / sub group size
    float k_local[4];

    if(ssm_state_indices != nullptr){
        for(int i = 0; i < 4; ++i){
            state_local[i] = ssm_state[i];
        } 
    }else {
        for(int i = 0; i < 4; ++i){
            state_local[i] = 0.0f;
        } 
    }

    for(int t = 0; t < num_tokens, ++t){
        float b_local = b[t * num_v_heads + num_heads_id];
        float beta = act_sigmiod(b_local);
        float a_local = a[t * num_v_heads + num_heads_id] + dt_bias_local;
        float g = -A_log_local * act_softplus(a_local);

        float kv_mem = 0.0f;
        for(int i = 0; i < 4; ++i){
            k_local[i] = k[t * num_k_heads * head_k_dim + num_heads_id * head_k_dim + i * sub_group_size + sg_local_id];
            state_local[i] *= g;
            kv_mem += state_local[i] * k_local[i];
        }
        kv_mem = sycl::reduce_over_group(sg, kv_mem, sycl::plus<>());

        float v_local = v[t * num_v_heads * head_v_dim + num_heads_id * head_v_dim + v_dim_id];
        delta = (v_local - kv_mem) * beta;

        float res = 0.0f;
        for(int i = 0; i < 4; ++i){
            state_local[i] += k_local[i] * delta;
            res += state_local[i] * q[t * num_k_heads * head_k_dim + num_heads_id * head_k_dim + i * sub_group_size + sg_local_id];
        }
        res = sycl::reduce_over_group(sg, res, sycl::plus<>());

        if(sg_local_id == 0){
            core_attn_out[batch_id * num_v_heads * num_tokens * head_v_dim + num_heads_id * num_tokens * head_v_dim + t * head_v_dim + v_dim_id] = res;
        }
    }

    for(int i = 0; i < 4; ++i){
        ssm_state[i] = state_local[i];
    } 
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
