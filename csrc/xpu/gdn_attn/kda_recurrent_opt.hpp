#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kda_attention.hpp"

// Optimised variants of `recurrent_kda_kernel` (kda_attention.hpp).
//
// The reference kernel is kept untouched as the correctness/fallback path
// (`VLLM_XPU_KDA_RECURRENT_MODE=recurrent`). This file specialises it along
// three axes:
//
//   * `head_dim` becomes a compile-time constant (`KBucket * sub_group_size`),
//     so every offset computation constant-folds instead of emitting runtime
//     64-bit multiplies inside the token loop.
//   * All global accesses (recurrent state, q/k/raw_gate, v, output) use wide
//     `sycl::vec` messages instead of scalar element accesses. Each lane owns
//     `KBucket` consecutive key channels and `values_per_sub_group`
//     consecutive value channels, so both are naturally vectorisable.
//   * The recurrent state may be stored in fp32, bf16 or fp16. The state is
//     the dominant memory traffic during decode (head_dim^2 per head, read and
//     written every step), so a narrower cache dtype directly buys bandwidth.
//
// `Mode` additionally drops the per-sequence token loop for pure decode and
// hoists loop-invariant work (dt_bias, head decay coefficient) out of it.

namespace kda {

static constexpr int recurrent_mode_general = 0;
static constexpr int recurrent_mode_spec = 1;
static constexpr int recurrent_mode_decode = 2;

// `N` is always a power of two in {1, 2, 4, 8}, so these map onto a single
// wide load/store message rather than N scalar accesses.
template <typename ElementT, int N>
inline void load_to_float(const ElementT* ptr, float (&dst)[N]) {
  if constexpr (N == 1) {
    dst[0] = static_cast<float>(*ptr);
  } else {
    const sycl::vec<ElementT, N> in =
        *reinterpret_cast<const sycl::vec<ElementT, N>*>(ptr);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<float>(in[i]);
    }
  }
}

template <typename ElementT, int N>
inline void store_from_float(ElementT* ptr, const float (&src)[N]) {
  if constexpr (N == 1) {
    *ptr = static_cast<ElementT>(src[0]);
  } else {
    sycl::vec<ElementT, N> out;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out[i] = static_cast<ElementT>(src[i]);
    }
    *reinterpret_cast<sycl::vec<ElementT, N>*>(ptr) = out;
  }
}

// The state cache may be page strided with an arbitrary slot stride, in which
// case the per-slot base is not guaranteed to satisfy the alignment a wide
// message needs. `vectorized` is uniform across the whole kernel, so the branch
// costs nothing beyond code size.
template <typename ElementT, int N>
inline void
load_state_to_float(const ElementT* ptr, float (&dst)[N], bool vectorized) {
  if (vectorized) {
    load_to_float<ElementT, N>(ptr, dst);
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<float>(ptr[i]);
    }
  }
}

template <typename ElementT, int N>
inline void
store_state_from_float(ElementT* ptr, const float (&src)[N], bool vectorized) {
  if (vectorized) {
    store_from_float<ElementT, N>(ptr, src);
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      ptr[i] = static_cast<ElementT>(src[i]);
    }
  }
}

template <typename T, typename StateT, int KBucket, int Mode>
struct recurrent_kda_opt_kernel {
  static constexpr bool is_spec = (Mode == recurrent_mode_spec);
  static constexpr bool is_decode = (Mode == recurrent_mode_decode);
  static constexpr int head_dim = KBucket * sub_group_size;
  static constexpr int value_buckets = head_dim / values_per_work_group > 0
                                           ? head_dim / values_per_work_group
                                           : 1;

  recurrent_kda_opt_kernel(
      T* output,
      const T* q,
      const T* k,
      const T* v,
      const T* raw_gate,
      const float* beta,
      const float* a_log,
      const float* dt_bias,
      float lower_bound,
      StateT* recurrent_state,
      int64_t recurrent_state_stride_0,
      const int* query_start_loc,
      const int* token_indx,
      const int* state_indices,
      int64_t state_indices_stride_0,
      const bool* has_initial_state,
      const int* num_accepted_tokens,
      int batch_size,
      int num_heads,
      bool state_vectorized)
      : output(output),
        q(q),
        k(k),
        v(v),
        raw_gate(raw_gate),
        beta(beta),
        a_log(a_log),
        dt_bias(dt_bias),
        lower_bound(lower_bound),
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
        state_vectorized(state_vectorized) {}

  static sycl::nd_range<3> get_nd_range(int batch_size, int num_heads) {
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
    if constexpr (is_spec) {
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

    // Lane-local slice of the [head][value][key] state block.
    const int64_t state_lane_offset =
        static_cast<int64_t>(head_id) * head_dim * head_dim +
        static_cast<int64_t>(value_start) * head_dim + lane_id * KBucket;
    StateT* initial_state_ptr =
        recurrent_state +
        static_cast<int64_t>(initial_state_id) * recurrent_state_stride_0 +
        state_lane_offset;

    float state[values_per_sub_group][KBucket];
#pragma unroll
    for (int value = 0; value < values_per_sub_group; ++value) {
      if (load_initial_state) {
        load_state_to_float<StateT, KBucket>(
            initial_state_ptr + value * head_dim,
            state[value],
            state_vectorized);
      } else {
#pragma unroll
        for (int key = 0; key < KBucket; ++key) {
          state[value][key] = 0.0f;
        }
      }
    }

    const float q_scale = sycl::rsqrt(static_cast<float>(head_dim));
    const float head_a = -sycl::native::exp(a_log[head_id]);
    // dt_bias only depends on (head, key channel): hoist it out of the token
    // loop instead of re-reading it for every token.
    float bias_local[KBucket];
    load_to_float<float, KBucket>(
        dt_bias + static_cast<int64_t>(head_id) * head_dim + lane_id * KBucket,
        bias_local);

    const int seq_start = query_start_loc[batch_id];
    const int seq_end =
        is_decode ? seq_start + 1 : query_start_loc[batch_id + 1];

    for (int local_token = seq_start; local_token < seq_end; ++local_token) {
      const int global_token =
          token_indx == nullptr ? local_token : token_indx[local_token];
      const int64_t token_head_offset =
          (static_cast<int64_t>(global_token) * num_heads + head_id) * head_dim;
      const int64_t qk_offset = token_head_offset + lane_id * KBucket;

      float q_local[KBucket];
      float k_local[KBucket];
      float decay[KBucket];
      load_to_float<T, KBucket>(q + qk_offset, q_local);
      load_to_float<T, KBucket>(k + qk_offset, k_local);
      load_to_float<T, KBucket>(raw_gate + qk_offset, decay);

      float q_sum = 0.0f;
      float k_sum = 0.0f;
#pragma unroll
      for (int key = 0; key < KBucket; ++key) {
        q_sum += q_local[key] * q_local[key];
        k_sum += k_local[key] * k_local[key];
        decay[key] = sycl::native::exp(
            kda_gate::native_log_gate(
                decay[key] + bias_local[key], head_a, lower_bound));
      }
      q_sum = sycl::reduce_over_group(sub_group, q_sum, sycl::plus<>());
      k_sum = sycl::reduce_over_group(sub_group, k_sum, sycl::plus<>());
      const float q_inv_norm =
          sycl::native::rsqrt(q_sum + l2norm_eps) * q_scale;
      const float k_inv_norm = sycl::native::rsqrt(k_sum + l2norm_eps);
#pragma unroll
      for (int key = 0; key < KBucket; ++key) {
        q_local[key] *= q_inv_norm;
        k_local[key] *= k_inv_norm;
      }

      float kv_memory[values_per_sub_group] = {};
#pragma unroll
      for (int value = 0; value < values_per_sub_group; ++value) {
#pragma unroll
        for (int key = 0; key < KBucket; ++key) {
          state[value][key] *= decay[key];
          kv_memory[value] += state[value][key] * k_local[key];
        }
      }
      // Reductions are issued after all FMAs so the independent cross-lane
      // shuffles pipeline instead of stalling the value loop one at a time.
#pragma unroll
      for (int value = 0; value < values_per_sub_group; ++value) {
        kv_memory[value] = sycl::reduce_over_group(
            sub_group, kv_memory[value], sycl::plus<>());
      }

      const float beta_value =
          beta[static_cast<int64_t>(global_token) * num_heads + head_id];
      const int64_t value_offset = token_head_offset + value_start;
      float v_local[values_per_sub_group];
      load_to_float<T, values_per_sub_group>(v + value_offset, v_local);

      float delta[values_per_sub_group];
#pragma unroll
      for (int value = 0; value < values_per_sub_group; ++value) {
        delta[value] = (v_local[value] - kv_memory[value]) * beta_value;
      }

      float result[values_per_sub_group] = {};
#pragma unroll
      for (int value = 0; value < values_per_sub_group; ++value) {
#pragma unroll
        for (int key = 0; key < KBucket; ++key) {
          state[value][key] += delta[value] * k_local[key];
          result[value] += state[value][key] * q_local[key];
        }
      }
#pragma unroll
      for (int value = 0; value < values_per_sub_group; ++value) {
        result[value] =
            sycl::reduce_over_group(sub_group, result[value], sycl::plus<>());
      }
      if (lane_id == 0) {
        store_from_float<T, values_per_sub_group>(
            output + value_offset, result);
      }

      if constexpr (is_spec) {
        const int token_in_sequence = local_token - seq_start;
        const int save_state_id = state_indices
            [batch_id * state_indices_stride_0 + token_in_sequence];
        if (save_state_id != pad_slot_id) {
          StateT* save_ptr =
              recurrent_state +
              static_cast<int64_t>(save_state_id) * recurrent_state_stride_0 +
              state_lane_offset;
#pragma unroll
          for (int value = 0; value < values_per_sub_group; ++value) {
            store_state_from_float<StateT, KBucket>(
                save_ptr + value * head_dim, state[value], state_vectorized);
          }
        }
      }
    }

    if constexpr (!is_spec) {
#pragma unroll
      for (int value = 0; value < values_per_sub_group; ++value) {
        store_state_from_float<StateT, KBucket>(
            initial_state_ptr + value * head_dim,
            state[value],
            state_vectorized);
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
  float lower_bound;
  StateT* recurrent_state;
  int64_t recurrent_state_stride_0;
  const int* query_start_loc;
  const int* token_indx;
  const int* state_indices;
  int64_t state_indices_stride_0;
  const bool* has_initial_state;
  const int* num_accepted_tokens;
  int batch_size;
  int num_heads;
  bool state_vectorized;
};

template <typename T, typename StateT, int KBucket, int Mode>
void launch_recurrent_kda_opt(
    sycl::queue& queue,
    T* output,
    const T* q,
    const T* k,
    const T* v,
    const T* raw_gate,
    const float* beta,
    const float* a_log,
    const float* dt_bias,
    float lower_bound,
    StateT* recurrent_state,
    int64_t recurrent_state_stride_0,
    const int* query_start_loc,
    const int* token_indx,
    const int* state_indices,
    int64_t state_indices_stride_0,
    const bool* has_initial_state,
    const int* num_accepted_tokens,
    int batch_size,
    int num_heads,
    bool state_vectorized) {
  using Kernel = recurrent_kda_opt_kernel<T, StateT, KBucket, Mode>;
  const auto range = Kernel::get_nd_range(batch_size, num_heads);
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
        lower_bound,
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
        state_vectorized);
    cgh.parallel_for(range, task);
  });
}

}  // namespace kda
