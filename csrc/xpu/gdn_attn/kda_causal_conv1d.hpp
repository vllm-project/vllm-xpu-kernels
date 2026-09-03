#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kda_common.hpp"

// Depthwise causal convolution over the three KDA projections (q, k, v), fused
// with the SiLU activation and the rolling conv-state cache vLLM keeps per
// sequence.
//
// `causal_conv1d_kernel<Width, IsSpec>` runs one thread per (sequence,
// channel) and walks the sequence serially, keeping the `Width` tap window in
// registers. `IsSpec` switches the state cache between one slot per sequence
// (non-spec) and one slot per accepted token (spec decode), which covers every
// batch shape the scheduler produces.

namespace kda {

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
      int hidden_dim,
      int64_t qkv_row_stride)
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
        hidden_dim(hidden_dim),
        qkv_row_stride(qkv_row_stride) {}

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
          input[static_cast<int64_t>(global_token) * qkv_row_stride + channel]);

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
  // Distance between consecutive tokens in q/k/v. Equals `hidden_dim` for
  // standalone projections and `3 * hidden_dim` when they are views into a
  // single fused mixed-QKV buffer.
  int64_t qkv_row_stride;
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
    int hidden_dim,
    int64_t qkv_row_stride) {
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
        hidden_dim,
        qkv_row_stride);
    cgh.parallel_for(range, task);
  });
}

}  // namespace kda
