#pragma once

#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace gdn_attention{

enum class ActMode{
    silu = 0;
    swish = 1;
}

template<typename T, int Width>
struct causal_conv1d_kernel
{
public:
    static constexpr int group_size = 256;
    static constexpr int elems_per_item = 1;
    static constexpr int elems_per_group = group_size * elems_per_item;

    causal_conv1d_kernel(
        T* q_out,
        T* k_out,
        T* v_out,
        T* z_out,
        const int head_k_dim,
        const int head_v_dim,
        const int num_v_heads,
        const int num_k_heads,
        const T* input,
        const T* weight,
        const T* bias,
        T* conv_states,
        const int* query_start_loc,
        const T* cache_indices,
        const bool* has_initial_state,
        const ActMode& act_mode,
        const int& pad_slot_id,
        const int& num_actual_tokens,
        const int& batch_size,
        const int& dim,
        const int& input_dim
    ):
        q_out(q_out),
        k_out(k_out),
        v_out(v_out),
        z_out(z_out),
        head_k_dim(head_k_dim),
        head_v_dim(head_v_dim),
        num_v_heads(num_v_heads),
        num_k_heads(num_k_heads),
        input(input),
        weight(weight),
        bias(bias),
        conv_states(conv_states),
        query_start_loc(query_start_loc),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
        act_mode(act_mode),
        pad_slot_id(pad_slot_id),
        num_actual_tokens(num_actual_tokens),
        batch_size(batch_size),
        dim(dim),
        input_dim(input_dim)
    {}

    static inline sycl::nd_range<2>
    get_nd_range(const int seqlen, const int input_dim) {
        const int groups_per_dim = (input_dim + elems_per_group - 1) / elems_per_group;
        sycl::range<2> local(1, group_size);
        sycl::range<2> global(seqlen, groups_per_dim);
        return sycl::nd_range<2>(global * local, local);
    }

    static inline void silu(float& x){}
    static inline void swish(float& x){}

  [[sycl::reqd_sub_group_size(32)]] void
  operator()(sycl::nd_item<2> item) const {
    const int token_id = item.get_group_id(0);
    const int local_group_id = item.get_group_id(1);
    const int local_id = item.get_group_linear_id();
    const int input_dim_id = local_group_id * elems_per_group + local_id;

    const int q_dim = head_k_dim;
    const int k_dim = head_k_dim;
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int z_dim = head_v_dim * num_v_heads / num_k_heads;
    const int qkvz_dim = q_dim + k_dim + v_dim + z_dim;


    int batch_id = batch_size - 1;
    int seq_start_offset = 0;
    int seq_end_offset = 0;
    for(int i = batch_size; i > 0; --i){
        if(token_id < query_start_loc[i]){
            batch_id = i - 1;
            seq_start_offset = query_start_loc[i - 1];
            seq_end_offset = query_start_loc[i];
            break;
        }
    }

    int states_id = cache_indices[batch_id];

    if (states_id == pad_slot_id){
        return;
    }

    if(input_dim_id >= input_dim){
        return;
    }

    int k_heads_id = input_dim_id / qkvz_dim;
    int qkvz_dim_id = input_dim_id % qkvz_dim;

    bool is_q = false;
    bool is_k = false;
    bool is_v = false;
    bool is_z = false;

    if(qkvz_dim_id < q_dim){
        is_q = true;
    }else if(qkvz_dim_id < q_dim + k_dim){
        is_k = true;
    }else if(qkvz_dim_id < q_dim + k_dim + v_dim){
        is_v = true;
    }else{
        is_z = true;
    }

    if(is_z){
        int z_dim_id = num_k_heads * z_dim + qkvz_dim_id - (q_dim + k_dim + v_dim);
        z_out[token_id * z_dim + z_dim_id] = input[token_id * qkvz_dim + qkvz_dim_id];
        return;
    }

    int reordered_dim_id = 0;
    if(is_q){
        reordered_dim_id = k_heads_id * q_dim + qkvz_dim_id;
    }else if(is_k){
        reordered_dim_id = num_k_heads * q_dim + k_heads_id * k_dim + qkvz_dim_id - q_dim;
    }else if(is_v){
        reordered_dim_id = num_k_heads * (q_dim + v_dim) + k_heads_id * v_dim + qkvz_dim_id - (q_dim + k_dim);
    }

    const T* initial_states_ptr = 
        has_initial_state[batch_id]
        ? conv_states + batch_id * (Width - 1) * dim
        : nullptr;

    T* states_out_ptr = conv_states + batch_id * (Width - 1) * dim;

    T local_weights[Width];
    #pragma unroll
    for(int i = 0; i < Width; ++i){
        local_weights[i] = weight[reordered_dim_id * Width + i];
    }

    T local_input[Width];
    #pragma unroll
    for(int i = 0; i < Width; ++i){
        local_input[i] = 0;
    }

    int input_start_seq_offset = token_id - seq_start_offset - Width;

    for(int i = 0; i < Width; ++i){
        local_input[i] = input[token_id * qkvz_dim + qkvz_dim_id];
    }

    float res = 0.0f;
    #pragma unroll
    for(int i = 0; i < Width; ++i){
        res += local_input[i] * local_weights[i];
    }

    if(bias != nullptr){
        res += bias[reordered_dim_id];
    }

    if(act_mode == ActMode::silu){
        act_silu(res);
    }else if(act_mode == ActMode::swish){
        act_swish(res);
    }

    if(is_q){
        q_out[token_id * num_k_heads * q_dim + k_heads_id * q_dim + qkvz_dim_id] = res;
    }else if(is_k){
        k_out[token_id * num_k_heads * k_dim + k_heads_id * k_dim + qkvz_dim_id - q_dim] = res;
    }else if(is_v){
        v_out[token_id * num_k_heads * v_dim + k_heads_id * v_dim + qkvz_dim_id - (q_dim + k_dim)] = res;
    }

  }
private:
    const T* q_out;
    const T* k_out;
    const T* v_out;
    const T* z_out;
    const int head_k_dim;
    const int head_v_dim;
    const int num_v_heads;
    const int num_k_heads;
    const T* input;
    const T* weight;
    const T* bias;
    T* conv_states;
    const int32_t* query_start_loc;
    const T* cache_indices;
    const bool* has_initial_state;
    const ActMode* act_mode;
    const int pad_slot_id;
    const int num_actual_tokens;
    const int batch_size;
    const int dim;
    const int input_dim;
};

template<typename T, int Width>
void kernel_launcher(
    sycl::queue& queue,
    T* q_out,
    T* k_out,
    T* v_out,
    T* z_out,
    const int head_k_dim,
    const int head_v_dim,
    const int num_v_heads,
    const int num_k_heads,
    const T* input,
    const T* weight,
    const T* bias,
    T* conv_states,
    int* query_start_loc,
    int* cache_indices,
    bool* has_initial_state,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int& num_actual_tokens,
    const int& batch_size,
    const int& dim
){
    using KERNEL = causal_conv1d_kernel<T, Width>;
    const int input_dim = num_k_heads * (2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads);
    auto range = KERNEL::get_nd_range(seqlen, input_dim);
    queue.submit([&](sycl::handler& cgh) {
        Kernel task(
            q_out,
            k_out,
            v_out,
            z_out,
            head_k_dim,
            head_v_dim,
            num_v_heads,
            num_k_heads,
            input,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            act_mode,
            pad_slot_id,
            num_actual_tokens,
            batch_size,
            dim,
            input_dim
        );
        cgh.parallel_for(range, task);
    });
}

void causal_conv1d(
    sycl::queue& queue,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out,
    torch::Tensor& z_out,
    const int head_k_dim,
    const int head_v_dim,
    const int num_v_heads,
    const int num_k_heads,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const torch::Tensor& has_initial_state,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int& num_actual_tokens
){
    const int batch_size = query_start_loc.size() - 1;
    const int dim = weight.size(0);
    const int width = weight.size(1);

#define KERNEL_LAUNCHER(scalar_t, width)       \
    kernel_launcher<scalar_t, width>(       \
        queue,       \
        reinterpret_cast<scalar_t*>(q_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(k_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(v_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(z_out.data_ptr()),       \
        head_k_dim,       \
        head_v_dim,       \
        num_v_heads,       \
        num_k_heads,       \
        reinterpret_cast<scalar_t*>(input.data_ptr()),       \
        reinterpret_cast<scalar_t*>(weight.data_ptr()),       \
        reinterpret_cast<scalar_t*>(bias.data_ptr()),       \
        reinterpret_cast<scalar_t*>(conv_states.data_ptr()),       \
        reinterpret_cast<int*>(query_start_loc.data_ptr()),       \
        reinterpret_cast<int*>(cache_indices.data_ptr()),       \
        reinterpret_cast<bool*>(has_initial_state.data_ptr()),       \
        act_mode,       \
        pad_slot_id,       \
        num_actual_tokens,            \
        batch_size,          \
        dim \
    );

#define WIDTH_DISPATCH(scalar_t, width)       \
    switch (width){       \
        case width == 1:       \
            KERNEL_LAUNCHER(scalar_t, 1)       \
            break;       \
        case width == 2:       \
            KERNEL_LAUNCHER(scalar_t, 2)       \
            break;       \
        case width == 3:       \
            KERNEL_LAUNCHER(scalar_t, 3)       \
            break;       \
        case width == 4:       \
            KERNEL_LAUNCHER(scalar_t, 4)       \
            break;       \
        case width == 5:       \
            KERNEL_LAUNCHER(scalar_t, 5)       \
            break;       \
        default:       \
            break;       \
    }

    if(input.scalar_type() == at::kBFloat16){
        using scalar_t = sycl::ext::oneapi::bfloat16;
        WIDTH_DISPATCH(scalar_t, width)
    }else if(input.scalar_type() == at::kHalf){
        using scalar_t = sycl::half;
        WIDTH_DISPATCH(scalar_t, width)
    }else{
        using scalar_t = float;
        WIDTH_DISPATCH(scalar_t, width)
    }
}

}
