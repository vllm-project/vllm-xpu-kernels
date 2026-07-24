#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace kda {

static constexpr int sub_group_size = 32;
static constexpr int work_group_size = 256;
static constexpr int values_per_sub_group = 4;
static constexpr int values_per_work_group =
    work_group_size / sub_group_size * values_per_sub_group;
static constexpr float l2norm_eps = 0.000001f;
static constexpr int pad_slot_id = -1;

inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }

inline float softplus(float x) {
  return x < 20.0f ? sycl::log(1.0f + sycl::exp(x)) : x;
}

template <typename T, typename CacheT, int Width, bool IsSpec>
struct causal_conv1d_kernel {
  causal_conv1d_kernel(
      T* q,
      T* k,
      T* v,
      const T* q_proj,
      const T* k_proj,
      const T* v_proj,
      const float* q_weight,
      const float* k_weight,
      const float* v_weight,
      CacheT* conv_state,
      int64_t conv_state_stride_0,
      int64_t conv_state_dim_stride,
      int64_t conv_state_time_stride,
      const int* query_start_loc,
      const int* token_indx,
      const int* state_indices,
      int64_t state_indices_stride_0,
      const bool* has_initial_state,
      const int* num_accepted_tokens,
      int batch_size,
      int hidden_dim)
      : q(q),
        k(k),
        v(v),
        q_proj(q_proj),
        k_proj(k_proj),
        v_proj(v_proj),
        q_weight(q_weight),
        k_weight(k_weight),
        v_weight(v_weight),
        conv_state(conv_state),
        conv_state_stride_0(conv_state_stride_0),
        conv_state_dim_stride(conv_state_dim_stride),
        conv_state_time_stride(conv_state_time_stride),
        query_start_loc(query_start_loc),
        token_indx(token_indx),
        state_indices(state_indices),
        state_indices_stride_0(state_indices_stride_0),
        has_initial_state(has_initial_state),
        num_accepted_tokens(num_accepted_tokens),
        batch_size(batch_size),
        hidden_dim(hidden_dim) {}

  static sycl::nd_range<2> get_nd_range(int batch_size, int hidden_dim) {
    const int combined_dim = 3 * hidden_dim;
    const int rounded_dim = (combined_dim + work_group_size - 1) /
                            work_group_size * work_group_size;
    return sycl::nd_range<2>(
        sycl::range<2>(batch_size, rounded_dim),
        sycl::range<2>(1, work_group_size));
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int combined_channel = item.get_global_id(1);
    if (batch_id >= batch_size || combined_channel >= 3 * hidden_dim) {
      return;
    }

    const int stream = combined_channel / hidden_dim;
    const int channel = combined_channel % hidden_dim;
    const int seq_start = query_start_loc[batch_id];
    const int seq_end = query_start_loc[batch_id + 1];

    int initial_state_id;
    bool load_initial_state;
    if constexpr (IsSpec) {
      int initial_col = num_accepted_tokens[batch_id] - 1;
      if (initial_col < 0) {
        initial_col = 0;
      }
      initial_state_id =
          state_indices[batch_id * state_indices_stride_0 + initial_col];
      load_initial_state = true;
    } else {
      initial_state_id = state_indices[batch_id];
      load_initial_state =
          has_initial_state == nullptr || has_initial_state[batch_id];
    }
    if (initial_state_id == pad_slot_id) {
      return;
    }

    const T* input = stream == 0 ? q_proj : (stream == 1 ? k_proj : v_proj);
    T* output = stream == 0 ? q : (stream == 1 ? k : v);
    const float* weight =
        stream == 0 ? q_weight : (stream == 1 ? k_weight : v_weight);

    CacheT history[Width - 1];
#pragma unroll
    for (int i = 0; i < Width - 1; ++i) {
      history[i] = static_cast<CacheT>(0.0f);
      if (load_initial_state) {
        history[i] = conv_state
            [static_cast<int64_t>(initial_state_id) * conv_state_stride_0 +
             static_cast<int64_t>(combined_channel) * conv_state_dim_stride +
             static_cast<int64_t>(i) * conv_state_time_stride];
      }
    }

    for (int local_token = seq_start; local_token < seq_end; ++local_token) {
      const int global_token =
          token_indx == nullptr ? local_token : token_indx[local_token];
      const CacheT current = static_cast<CacheT>(
          input[static_cast<int64_t>(global_token) * hidden_dim + channel]);

      float acc = static_cast<float>(current) *
                  weight[static_cast<int64_t>(channel) * Width + Width - 1];
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
        acc += static_cast<float>(history[i]) *
               weight[static_cast<int64_t>(channel) * Width + i];
      }
      output[static_cast<int64_t>(global_token) * hidden_dim + channel] =
          static_cast<T>(silu(acc));

#pragma unroll
      for (int i = 0; i < Width - 2; ++i) {
        history[i] = history[i + 1];
      }
      history[Width - 2] = current;

      if constexpr (IsSpec) {
        const int token_in_sequence = local_token - seq_start;
        const int save_state_id = state_indices
            [batch_id * state_indices_stride_0 + token_in_sequence];
        if (save_state_id != pad_slot_id) {
#pragma unroll
          for (int i = 0; i < Width - 1; ++i) {
            conv_state
                [static_cast<int64_t>(save_state_id) * conv_state_stride_0 +
                 static_cast<int64_t>(combined_channel) *
                     conv_state_dim_stride +
                 static_cast<int64_t>(i) * conv_state_time_stride] = history[i];
          }
        }
      }
    }

    if constexpr (!IsSpec) {
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
        conv_state
            [static_cast<int64_t>(initial_state_id) * conv_state_stride_0 +
             static_cast<int64_t>(combined_channel) * conv_state_dim_stride +
             static_cast<int64_t>(i) * conv_state_time_stride] = history[i];
      }
    }
  }

 private:
  T* q;
  T* k;
  T* v;
  const T* q_proj;
  const T* k_proj;
  const T* v_proj;
  const float* q_weight;
  const float* k_weight;
  const float* v_weight;
  CacheT* conv_state;
  int64_t conv_state_stride_0;
  int64_t conv_state_dim_stride;
  int64_t conv_state_time_stride;
  const int* query_start_loc;
  const int* token_indx;
  const int* state_indices;
  int64_t state_indices_stride_0;
  const bool* has_initial_state;
  const int* num_accepted_tokens;
  int batch_size;
  int hidden_dim;
};

template <typename T, typename CacheT, int Width, bool IsSpec>
void launch_causal_conv1d(
    sycl::queue& queue,
    T* q,
    T* k,
    T* v,
    const T* q_proj,
    const T* k_proj,
    const T* v_proj,
    const float* q_weight,
    const float* k_weight,
    const float* v_weight,
    CacheT* conv_state,
    int64_t conv_state_stride_0,
    int64_t conv_state_dim_stride,
    int64_t conv_state_time_stride,
    const int* query_start_loc,
    const int* token_indx,
    const int* state_indices,
    int64_t state_indices_stride_0,
    const bool* has_initial_state,
    const int* num_accepted_tokens,
    int batch_size,
    int hidden_dim) {
  using Kernel = causal_conv1d_kernel<T, CacheT, Width, IsSpec>;
  const auto range = Kernel::get_nd_range(batch_size, hidden_dim);
  queue.submit([&](sycl::handler& cgh) {
    Kernel task(
        q,
        k,
        v,
        q_proj,
        k_proj,
        v_proj,
        q_weight,
        k_weight,
        v_weight,
        conv_state,
        conv_state_stride_0,
        conv_state_dim_stride,
        conv_state_time_stride,
        query_start_loc,
        token_indx,
        state_indices,
        state_indices_stride_0,
        has_initial_state,
        num_accepted_tokens,
        batch_size,
        hidden_dim);
    cgh.parallel_for(range, task);
  });
}

template <typename T, int KBucketSize, bool IsSpec>
struct recurrent_kda_kernel {
  recurrent_kda_kernel(
      T* output,
      const T* q,
      const T* k,
      const T* v,
      const T* raw_gate,
      const float* beta,
      const float* a_log,
      const float* dt_bias,
      float* recurrent_state,
      int64_t recurrent_state_stride_0,
      const int* query_start_loc,
      const int* token_indx,
      const int* state_indices,
      int64_t state_indices_stride_0,
      const bool* has_initial_state,
      const int* num_accepted_tokens,
      int batch_size,
      int num_heads,
      int head_dim)
      : output(output),
        q(q),
        k(k),
        v(v),
        raw_gate(raw_gate),
        beta(beta),
        a_log(a_log),
        dt_bias(dt_bias),
        recurrent_state(recurrent_state),
        recurrent_state_stride_0(recurrent_state_stride_0),
        query_start_loc(query_start_loc),
        token_indx(token_indx),
        state_indices(state_indices),
        state_indices_stride_0(state_indices_stride_0),
        has_initial_state(has_initial_state),
        num_accepted_tokens(num_accepted_tokens),
        batch_size(batch_size),
        num_heads(num_heads),
        head_dim(head_dim) {}

  static sycl::nd_range<3>
  get_nd_range(int batch_size, int num_heads, int head_dim) {
    const int value_buckets =
        (head_dim + values_per_work_group - 1) / values_per_work_group;
    return sycl::nd_range<3>(
        sycl::range<3>(batch_size, num_heads, value_buckets * work_group_size),
        sycl::range<3>(1, 1, work_group_size));
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    const int batch_id = item.get_group(0);
    const int head_id = item.get_group(1);
    const int value_bucket = item.get_group(2);
    if (batch_id >= batch_size || head_id >= num_heads) {
      return;
    }

    const auto sub_group = item.get_sub_group();
    const int sub_group_id = sub_group.get_group_id();
    const int lane_id = sub_group.get_local_id();
    const int value_start = value_bucket * values_per_work_group +
                            sub_group_id * values_per_sub_group;

    int initial_state_id;
    bool load_initial_state;
    if constexpr (IsSpec) {
      int initial_col = num_accepted_tokens[batch_id] - 1;
      if (initial_col < 0) {
        initial_col = 0;
      }
      initial_state_id =
          state_indices[batch_id * state_indices_stride_0 + initial_col];
      load_initial_state = true;
    } else {
      initial_state_id = state_indices[batch_id];
      load_initial_state =
          has_initial_state == nullptr || has_initial_state[batch_id];
    }
    if (initial_state_id == pad_slot_id) {
      return;
    }

    float state[values_per_sub_group * KBucketSize];
#pragma unroll
    for (int value = 0; value < values_per_sub_group; ++value) {
#pragma unroll
      for (int key = 0; key < KBucketSize; ++key) {
        const int key_id = lane_id * KBucketSize + key;
        const int64_t state_offset =
            static_cast<int64_t>(head_id) * head_dim * head_dim +
            static_cast<int64_t>(value_start + value) * head_dim + key_id;
        state[value * KBucketSize + key] =
            load_initial_state ? recurrent_state
                                     [static_cast<int64_t>(initial_state_id) *
                                          recurrent_state_stride_0 +
                                      state_offset]
                               : 0.0f;
      }
    }

    const int seq_start = query_start_loc[batch_id];
    const int seq_end = query_start_loc[batch_id + 1];
    const float q_scale = sycl::rsqrt(static_cast<float>(head_dim));
    const float head_a = -sycl::exp(a_log[head_id]);

    for (int local_token = seq_start; local_token < seq_end; ++local_token) {
      const int global_token =
          token_indx == nullptr ? local_token : token_indx[local_token];

      float q_local[KBucketSize];
      float k_local[KBucketSize];
      float decay[KBucketSize];
      float q_sum = 0.0f;
      float k_sum = 0.0f;
#pragma unroll
      for (int key = 0; key < KBucketSize; ++key) {
        const int key_id = lane_id * KBucketSize + key;
        const int64_t qk_offset =
            (static_cast<int64_t>(global_token) * num_heads + head_id) *
                head_dim +
            key_id;
        q_local[key] = static_cast<float>(q[qk_offset]);
        k_local[key] = static_cast<float>(k[qk_offset]);
        q_sum += q_local[key] * q_local[key];
        k_sum += k_local[key] * k_local[key];

        const float gate =
            static_cast<float>(raw_gate[qk_offset]) +
            dt_bias[static_cast<int64_t>(head_id) * head_dim + key_id];
        decay[key] = sycl::exp(head_a * softplus(gate));
      }
      q_sum = sycl::reduce_over_group(sub_group, q_sum, sycl::plus<>());
      k_sum = sycl::reduce_over_group(sub_group, k_sum, sycl::plus<>());
      const float q_inv_norm = sycl::rsqrt(q_sum + l2norm_eps) * q_scale;
      const float k_inv_norm = sycl::rsqrt(k_sum + l2norm_eps);
#pragma unroll
      for (int key = 0; key < KBucketSize; ++key) {
        q_local[key] *= q_inv_norm;
        k_local[key] *= k_inv_norm;
      }

      float kv_memory[values_per_sub_group] = {};
#pragma unroll
      for (int value = 0; value < values_per_sub_group; ++value) {
#pragma unroll
        for (int key = 0; key < KBucketSize; ++key) {
          float& state_value = state[value * KBucketSize + key];
          state_value *= decay[key];
          kv_memory[value] += state_value * k_local[key];
        }
        kv_memory[value] = sycl::reduce_over_group(
            sub_group, kv_memory[value], sycl::plus<>());
      }

      const float beta_value =
          beta[static_cast<int64_t>(global_token) * num_heads + head_id];
      float result[values_per_sub_group] = {};
#pragma unroll
      for (int value = 0; value < values_per_sub_group; ++value) {
        const int64_t value_offset =
            (static_cast<int64_t>(global_token) * num_heads + head_id) *
                head_dim +
            value_start + value;
        const float delta =
            (static_cast<float>(v[value_offset]) - kv_memory[value]) *
            beta_value;
#pragma unroll
        for (int key = 0; key < KBucketSize; ++key) {
          float& state_value = state[value * KBucketSize + key];
          state_value += delta * k_local[key];
          result[value] += state_value * q_local[key];
        }
        result[value] =
            sycl::reduce_over_group(sub_group, result[value], sycl::plus<>());
        if (lane_id == 0) {
          output[value_offset] = static_cast<T>(result[value]);
        }
      }

      if constexpr (IsSpec) {
        const int token_in_sequence = local_token - seq_start;
        const int save_state_id = state_indices
            [batch_id * state_indices_stride_0 + token_in_sequence];
        if (save_state_id != pad_slot_id) {
#pragma unroll
          for (int value = 0; value < values_per_sub_group; ++value) {
#pragma unroll
            for (int key = 0; key < KBucketSize; ++key) {
              const int key_id = lane_id * KBucketSize + key;
              const int64_t state_offset =
                  static_cast<int64_t>(head_id) * head_dim * head_dim +
                  static_cast<int64_t>(value_start + value) * head_dim + key_id;
              recurrent_state
                  [static_cast<int64_t>(save_state_id) *
                       recurrent_state_stride_0 +
                   state_offset] = state[value * KBucketSize + key];
            }
          }
        }
      }
    }

    if constexpr (!IsSpec) {
#pragma unroll
      for (int value = 0; value < values_per_sub_group; ++value) {
#pragma unroll
        for (int key = 0; key < KBucketSize; ++key) {
          const int key_id = lane_id * KBucketSize + key;
          const int64_t state_offset =
              static_cast<int64_t>(head_id) * head_dim * head_dim +
              static_cast<int64_t>(value_start + value) * head_dim + key_id;
          recurrent_state
              [static_cast<int64_t>(initial_state_id) *
                   recurrent_state_stride_0 +
               state_offset] = state[value * KBucketSize + key];
        }
      }
    }
  }

 private:
  T* output;
  const T* q;
  const T* k;
  const T* v;
  const T* raw_gate;
  const float* beta;
  const float* a_log;
  const float* dt_bias;
  float* recurrent_state;
  int64_t recurrent_state_stride_0;
  const int* query_start_loc;
  const int* token_indx;
  const int* state_indices;
  int64_t state_indices_stride_0;
  const bool* has_initial_state;
  const int* num_accepted_tokens;
  int batch_size;
  int num_heads;
  int head_dim;
};

template <typename T, int KBucketSize, bool IsSpec>
void launch_recurrent_kda(
    sycl::queue& queue,
    T* output,
    const T* q,
    const T* k,
    const T* v,
    const T* raw_gate,
    const float* beta,
    const float* a_log,
    const float* dt_bias,
    float* recurrent_state,
    int64_t recurrent_state_stride_0,
    const int* query_start_loc,
    const int* token_indx,
    const int* state_indices,
    int64_t state_indices_stride_0,
    const bool* has_initial_state,
    const int* num_accepted_tokens,
    int batch_size,
    int num_heads,
    int head_dim) {
  using Kernel = recurrent_kda_kernel<T, KBucketSize, IsSpec>;
  const auto range = Kernel::get_nd_range(batch_size, num_heads, head_dim);
  queue.submit([&](sycl::handler& cgh) {
    Kernel task(
        output,
        q,
        k,
        v,
        raw_gate,
        beta,
        a_log,
        dt_bias,
        recurrent_state,
        recurrent_state_stride_0,
        query_start_loc,
        token_indx,
        state_indices,
        state_indices_stride_0,
        has_initial_state,
        num_accepted_tokens,
        batch_size,
        num_heads,
        head_dim);
    cgh.parallel_for(range, task);
  });
}

}  // namespace kda
