#pragma once

#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace gdn{

enum class ActMode{
    silu = 0,
    swish = 1,
};

template<typename T, int Width>
struct causal_conv1d_kernel
{
public:
    static constexpr int sub_group_size = 32;
    static constexpr int group_size = 256;
    static constexpr int elems_per_item = 1;
    static constexpr int elems_per_group = group_size * elems_per_item;

    causal_conv1d_kernel(
        T* q_out,
        T* k_out,
        T* v_out,
        T* z_out,
        T* b_out,
        T* a_out,
        const T* mixed_qkvz,
        const T* mixed_ba,
        const T* conv_weights,
        const T* conv_bias,
        T* conv_states,
        int* query_start_loc,
        int* cache_indices,
        bool* has_initial_state,
        const ActMode& act_mode,
        const int& pad_slot_id,
        const int& batch_size,
        const int& num_actual_tokens,
        const int& num_k_heads,
        const int& head_k_dim,
        const int& num_v_heads,
        const int& head_v_dim,
        const int & qkvz_dim,
        const int& conv_dim
    ):
        q_out(q_out),
        k_out(k_out),
        v_out(v_out),
        z_out(z_out),
        b_out(b_out),
        a_out(a_out),
        mixed_qkvz(mixed_qkvz),
        mixed_ba(mixed_ba),
        conv_weights(conv_weights),
        conv_bias(conv_bias),
        conv_states(conv_states),
        query_start_loc(query_start_loc),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
        act_mode(act_mode),
        pad_slot_id(pad_slot_id),
        batch_size(batch_size),
        num_actual_tokens(num_actual_tokens),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim),
        qkvz_dim(qkvz_dim),
        conv_dim(conv_dim)
    {}

    static inline sycl::nd_range<2>
    get_nd_range(const int seqlen, const int qkvz_dim) {
        const int groups_per_dim = (qkvz_dim + elems_per_group - 1) / elems_per_group;
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

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int token_id = item.get_group(0);
    const int local_group_id = item.get_group(1);
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
        int z_dim_id = k_heads_id * z_dim + qkvz_dim_id - (q_dim + k_dim + v_dim);
        z_out[token_id * num_k_heads * z_dim + z_dim_id] = mixed_qkvz[token_id * input_dim + input_dim_id];
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
        (has_initial_state != nullptr && has_initial_state[batch_id])
        ? conv_states + states_id * (Width - 1) * dim
        : nullptr;

    T* states_out_ptr = conv_states + states_id * (Width - 1) * dim;

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

    int seq_cu_len = token_id - seq_start_offset + 1;
    int input_load_len = seq_cu_len > Width ? Width : seq_cu_len;
    int states_load_len = seq_cu_len > Width ? 0 : Width - input_load_len;
    if(states_load_len != 0 && initial_states_ptr != nullptr){
        for(int i = 0; i < states_load_len; ++i){
            local_input[i] = initial_states_ptr[(Width - states_load_len + i) * dim + reordered_dim_id];
        }
    }

    for(int i = 0; i < input_load_len; ++i){
        local_input[states_load_len + i] = mixed_qkvz[(token_id - input_load_len + i) * input_dim + input_dim_id];
    }

    float res = 0.0f;
    #pragma unroll
    for(int i = 0; i < Width; ++i){
        res += static_cast<float>(local_input[i]) * static_cast<float>(local_weights[i]);
    }

    if(bias != nullptr){
        res += bias[reordered_dim_id];
    }

    if(seq_end_offset - token_id < Width){
        states_out_ptr[(Width - 1 - (seq_end_offset - token_id)) * dim + reordered_dim_id] = res;
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
    T* q_out;
    T* k_out;
    T* v_out;
    T* z_out;
    T* b_out;
    T* a_out;
    const T* mixed_qkvz;
    const T* mixed_ba;
    const T* conv_weights;
    const T* conv_bias;
    T* conv_states;
    const int32_t* query_start_loc;
    const int* cache_indices;
    const bool* has_initial_state;
    const ActMode act_mode;
    const int pad_slot_id;
    const int batch_size;
    const int num_actual_tokens;
    const int num_k_heads;
    const int head_k_dim;
    const int num_v_heads;
    const int head_v_dim;
    const int qkvz_dim;
    const int conv_dim;
};

template<typename T, int Width>
void kernel_launcher(
    sycl::queue& queue,
    T* q_out,
    T* k_out,
    T* v_out,
    T* z_out,
    T* b_out,
    T* a_out,
    const T* mixed_qkvz,
    const T* mixed_ba,
    const T* conv_weights,
    const T* conv_bias,
    T* conv_states,
    int* query_start_loc,
    int* cache_indices,
    bool* has_initial_state,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int& batch_size,
    const int& num_actual_tokens,
    const int& num_k_heads,
    const int& head_k_dim,
    const int& num_v_heads,
    const int& head_v_dim,
    const int & qkvz_dim,
    const int& conv_dim
){
    using KERNEL = causal_conv1d_kernel<T, Width>;
    auto range = KERNEL::get_nd_range(num_actual_tokens, qkvz_dim);
    queue.submit([&](sycl::handler& cgh) {
        KERNEL task(
            q_out,
            k_out,
            v_out,
            z_out,
            b_out,
            a_out,
            mixed_qkvz,
            mixed_ba,
            conv_weights,
            conv_bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            act_mode,
            pad_slot_id,
            batch_size,
            num_actual_tokens,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            qkvz_dim,
            conv_dim
        );
        cgh.parallel_for(range, task);
    });
}

void causal_conv1d(
    sycl::queue& queue,
    torch::Tensor& q_out, // [total_seqlen, num_k_heads, head_k_dim]
    torch::Tensor& k_out, // [total_seqlen, num_k_heads, head_k_dim]
    torch::Tensor& v_out, // [total_seqlen, num_v_heads, head_v_dim]
    torch::Tensor& z_out, // [total_seqlen, num_v_heads, head_v_dim]
    torch::Tensor& b_out, // [total_seqlen, num_v_heads]
    torch::Tensor& a_out, // [total_seqlen, num_v_heads]
    const torch::Tensor& mixed_qkvz, // [total_seqlen, num_k_heads * (2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads)]
    const torch::Tensor& mixed_ba,  // [total_seqlen, num_k_heads * (2 * num_v_heads / num_k_heads)]
    const torch::Tensor& conv_weights, // [2 * num_k_heads *  head_k_dim, width]
    const std::optional<torch::Tensor>& conv_bias, // [2 * num_k_heads *  head_k_dim] or None
    torch::Tensor& conv_states, // [cache_batch_size, width - 1, 2 * num_k_heads *  head_k_dim]
    const torch::Tensor& query_start_loc, // [batch_size + 1]
    const torch::Tensor& cache_indices, // [batch_size]
    const std::optional<torch::Tensor>& has_initial_state, // [batch_size] or None
    const ActMode& act_mode, // silu or swish
    const int& pad_slot_id, // -1
){
    const int batch_size = query_start_loc.size(0) - 1;
    const int num_actual_tokens = q_out.size(0);
    const int num_k_heads = q_out.size(1);
    const int head_k_dim = q_out.size(2);
    const int num_v_heads = v_out.size(1);
    const int head_v_dim = v_out.size(2);
    const int qkvz_dim = mixed_qkvz.size(2);
    const int conv_dim = conv_weights.size(0);
    const int width = conv_weights.size(1);

#define KERNEL_LAUNCHER(scalar_t, width)       \
    kernel_launcher<scalar_t, width>(       \
        queue,       \
        reinterpret_cast<scalar_t*>(q_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(k_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(v_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(z_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(b_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(a_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(mixed_qkvz.data_ptr()),       \
        reinterpret_cast<scalar_t*>(mixed_ba.data_ptr()),       \
        reinterpret_cast<scalar_t*>(conv_weights.data_ptr()),       \
        conv_bias.has_value()? reinterpret_cast<scalar_t*>(conv_bias->data_ptr()) : nullptr,       \
        reinterpret_cast<scalar_t*>(conv_states.data_ptr()),       \
        reinterpret_cast<int*>(query_start_loc.data_ptr()),       \
        reinterpret_cast<int*>(cache_indices.data_ptr()),       \
        has_initial_state.has_value()? reinterpret_cast<bool*>(has_initial_state->data_ptr()) : nullptr,       \
        act_mode,       \
        pad_slot_id,       \
        batch_size,          \
        num_actual_tokens,            \
        num_k_heads, \
        head_k_dim, \
        num_v_heads, \
        num_k_heads, \
        head_v_dim, \
        qkvz_dim, \
        conv_dim \
    );

#define WIDTH_DISPATCH(scalar_t, width)       \
    switch (width){       \
        case 1:       \
            KERNEL_LAUNCHER(scalar_t, 1)       \
            break;       \
        case 2:       \
            KERNEL_LAUNCHER(scalar_t, 2)       \
            break;       \
        case 3:       \
            KERNEL_LAUNCHER(scalar_t, 3)       \
            break;       \
        case 4:       \
            KERNEL_LAUNCHER(scalar_t, 4)       \
            break;       \
        case 5:       \
            KERNEL_LAUNCHER(scalar_t, 5)       \
            break;       \
        default:       \
            break;       \
    }

    if(mixed_qkvz.scalar_type() == at::kBFloat16){
        using scalar_t = sycl::ext::oneapi::bfloat16;
        WIDTH_DISPATCH(scalar_t, width)
    }else if(mixed_qkvz.scalar_type() == at::kHalf){
        using scalar_t = sycl::half;
        WIDTH_DISPATCH(scalar_t, width)
    }else{
        using scalar_t = float;
        WIDTH_DISPATCH(scalar_t, width)
    }
}

}
