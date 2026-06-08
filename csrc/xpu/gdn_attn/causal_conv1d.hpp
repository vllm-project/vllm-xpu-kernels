#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "gdn_attn_utils.h"

namespace gdn {

template <typename T, int Width, bool ReorderInput>
struct causal_conv1d_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = 4;
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
      const int conv_states_stride_0,
      T* conv_states_tmp,
      int* query_start_loc,
      int* token_indx,
      int* cache_indices,
      bool* has_initial_state,
      int* num_accepted_tokens,
      const ActMode& act_mode,
      const int& pad_slot_id,
      const int& batch_size,
      const int& num_virtual_tokens,
      const int& num_actual_tokens,
      const int& num_k_heads,
      const int& head_k_dim,
      const int& num_v_heads,
      const int& head_v_dim,
      const int& qkvz_elems,
      const int& conv_elems)
      : q_out(q_out),
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
        conv_states_stride_0(conv_states_stride_0),
        conv_states_tmp(conv_states_tmp),
        query_start_loc(query_start_loc),
        token_indx(token_indx),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
        num_accepted_tokens(num_accepted_tokens),
        act_mode(act_mode),
        pad_slot_id(pad_slot_id),
        batch_size(batch_size),
        num_actual_tokens(num_actual_tokens),
        num_virtual_tokens(num_virtual_tokens),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim),
        qkvz_elems(qkvz_elems),
        conv_elems(conv_elems) {}

  static inline sycl::nd_range<2>
  get_nd_range(const int num_virtual_tokens, const int qkvz_elems) {
    const int groups_per_token =
        (qkvz_elems + elems_per_group - 1) / elems_per_group;
    sycl::range<2> local(1, group_size);
    sycl::range<2> global(num_virtual_tokens, groups_per_token);
    return sycl::nd_range<2>(global * local, local);
  }

  static inline void act_swish(float& x, float beta = 1.0f) {
    x = x / (1.0f + sycl::exp(-x * beta));
  }
  static inline void act_silu(float& x) { act_swish(x, 1.0f); }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int token_id = item.get_group(0);
    const int local_group_id = item.get_group(1);
    const int local_id = item.get_local_linear_id();
    const int qkvz_elems_id =
        local_group_id * elems_per_group + local_id * elems_per_item;

    if (qkvz_elems_id >= qkvz_elems) {
      return;
    }

    // When q/k/v/b/a (LOCAL outputs) are a subset of mixed_qkvz/mixed_ba
    // (GLOBAL inputs), token_indx maps local token positions back to global
    // positions. When nullptr, local == global.
    const int global_token_id =
        (token_indx != nullptr) ? token_indx[token_id] : token_id;

    const int q_dim = head_k_dim;
    const int k_dim = head_k_dim;
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int z_dim = head_v_dim * num_v_heads / num_k_heads;
    const int qkvz_dim = q_dim + k_dim + v_dim + z_dim;

    int k_heads_id = qkvz_elems_id / qkvz_dim;
    int qkvz_dim_id = qkvz_elems_id % qkvz_dim;

    // reorder b,a (mixed_ba is GLOBAL → indexed by global_token_id;
    //              b_out/a_out are LOCAL → indexed by token_id)
    if constexpr (ReorderInput) {
      if (qkvz_elems_id < num_v_heads) {
        int step_local = token_id * num_v_heads;
        int step_global = global_token_id * num_v_heads;
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          b_out[step_local + qkvz_elems_id + e] =
              mixed_ba[step_global * 2 + qkvz_elems_id + e];
          a_out[step_local + qkvz_elems_id + e] =
              mixed_ba[step_global * 2 + num_v_heads + qkvz_dim_id + e];
        }
      }
    } else {
      if (qkvz_dim_id < (num_v_heads / num_k_heads)) {
        int step_local =
            token_id * num_v_heads + k_heads_id * num_v_heads / num_k_heads;
        int step_global = global_token_id * num_v_heads +
                          k_heads_id * num_v_heads / num_k_heads;
        const int ba_elems_per_item =
            sycl::min(elems_per_item, num_v_heads / num_k_heads);
#pragma unroll
        for (int e = 0; e < ba_elems_per_item; ++e) {
          b_out[step_local + qkvz_dim_id + e] =
              mixed_ba[step_global * 2 + qkvz_dim_id + e];
          a_out[step_local + qkvz_dim_id + e] = mixed_ba
              [step_global * 2 + num_v_heads / num_k_heads + qkvz_dim_id + e];
        }
      }
    }

    // get current seq start, end
    int batch_id = batch_size - 1;
    int seq_start_offset = 0;
    int seq_end_offset = 0;
    for (int i = 0; i < batch_size; ++i) {
      if (token_id < query_start_loc[i + 1]) {
        batch_id = i;
        seq_start_offset = query_start_loc[i];
        seq_end_offset = query_start_loc[i + 1];
        break;
      }
    }

    // get states cache location
    int states_id = cache_indices[batch_id];

    if (states_id == pad_slot_id) {
      return;
    }

    int mixed_qkvz_id = qkvz_elems_id;

    bool is_q = false;
    bool is_k = false;
    bool is_v = false;
    bool is_z = false;

    if (qkvz_dim_id < q_dim) {
      is_q = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = k_heads_id * k_dim + qkvz_dim_id;
      }
    } else if (qkvz_dim_id < q_dim + k_dim) {
      is_k = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = num_k_heads * head_k_dim + k_heads_id * k_dim +
                        qkvz_dim_id - (q_dim);
      }
    } else if (qkvz_dim_id < q_dim + k_dim + v_dim) {
      is_v = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = 2 * num_k_heads * head_k_dim + k_heads_id * v_dim +
                        qkvz_dim_id - (q_dim + k_dim);
      }
    } else {
      is_z = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = 2 * num_k_heads * head_k_dim +
                        num_v_heads * head_v_dim + k_heads_id * z_dim +
                        qkvz_dim_id - (q_dim + k_dim + v_dim);
      }
    }

    // reorder z (z_out is GLOBAL → use global_token_id; mixed_qkvz is GLOBAL)
    if (is_z) {
      int z_elems_id =
          k_heads_id * z_dim + qkvz_dim_id - (q_dim + k_dim + v_dim);
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        z_out[global_token_id * num_k_heads * z_dim + z_elems_id + e] =
            mixed_qkvz[global_token_id * qkvz_elems + mixed_qkvz_id + e];
      }
      return;
    }

    // reorder index to map weights
    int reordered_elems_id = 0;
    if (is_q) {
      reordered_elems_id = k_heads_id * q_dim + qkvz_dim_id;
    } else if (is_k) {
      reordered_elems_id =
          num_k_heads * q_dim + k_heads_id * k_dim + qkvz_dim_id - q_dim;
    } else if (is_v) {
      reordered_elems_id = num_k_heads * (q_dim + k_dim) + k_heads_id * v_dim +
                           qkvz_dim_id - (q_dim + k_dim);
    }

    // get states cache ptr
    const bool has_init_conv_states =
        (has_initial_state == nullptr ||
         (has_initial_state != nullptr && has_initial_state[batch_id]));
    T* conv_states_ptr = conv_states + states_id * conv_states_stride_0;

    // load weights
    T local_weights[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_weights[Width * e + i] =
            conv_weights[(reordered_elems_id + e) * Width + i];
      }
    }

    // load input
    T local_input[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + i] = 0.0f;
      }
    }

    int seq_cu_len = token_id - seq_start_offset + 1;
    int input_load_len = seq_cu_len >= Width ? Width : seq_cu_len;
    int states_load_len = seq_cu_len >= Width ? 0 : Width - input_load_len;
    if (states_load_len != 0 && has_init_conv_states) {
#pragma unroll
      for (int i = 0; i < states_load_len; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          local_input[Width * e + i] = conv_states_ptr
              [(Width - 1 - states_load_len + i) * conv_elems +
               reordered_elems_id + e];
        }
      }
    }

#pragma unroll
    for (int i = 0; i < input_load_len; ++i) {
      const int load_local = token_id - input_load_len + 1 + i;
      const int load_global =
          (token_indx != nullptr) ? token_indx[load_local] : load_local;
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + states_load_len + i] =
            mixed_qkvz[load_global * qkvz_elems + mixed_qkvz_id + e];
      }
    }

    float res[elems_per_item];
#pragma unroll
    for (int i = 0; i < elems_per_item; ++i) {
      res[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        res[e] += static_cast<float>(local_input[Width * e + i]) *
                  static_cast<float>(local_weights[Width * e + i]);
      }
    }

    if (conv_bias != nullptr) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        res[e] += conv_bias[reordered_elems_id + e];
      }
    }

    // save states
    if (seq_end_offset - seq_start_offset > 1) {
      // because current group is unable to know if old states are needed by
      // other group, hard to update states inplace if prefill
      if (seq_end_offset - 1 == token_id) {
#pragma unroll
        for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            conv_states_tmp
                [batch_id * (Width - 1) * conv_elems + i * conv_elems +
                 reordered_elems_id + e] = local_input[Width * e + i + 1];
          }
        }
      }
    } else {
// update states inplace if decode
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          conv_states_ptr[i * conv_elems + reordered_elems_id + e] =
              local_input[Width * e + i + 1];
        }
      }
    }

    if (act_mode == ActMode::silu) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        act_silu(res[e]);
      }
    } else if (act_mode == ActMode::swish) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        act_swish(res[e]);
      }
    }

    // reorder q, k, v
    if (is_q) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        q_out
            [token_id * num_k_heads * q_dim + k_heads_id * q_dim + qkvz_dim_id +
             e] = res[e];
      }
    } else if (is_k) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        k_out
            [token_id * num_k_heads * k_dim + k_heads_id * k_dim + qkvz_dim_id -
             q_dim + e] = res[e];
      }
    } else if (is_v) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        v_out
            [token_id * num_k_heads * v_dim + k_heads_id * v_dim + qkvz_dim_id -
             (q_dim + k_dim) + e] = res[e];
      }
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
  const int conv_states_stride_0;
  T* conv_states_tmp;
  const int32_t* query_start_loc;
  const int* token_indx;
  const int* cache_indices;
  const bool* has_initial_state;
  const int* num_accepted_tokens;
  const ActMode act_mode;
  const int pad_slot_id;
  const int batch_size;
  const int num_virtual_tokens;
  const int num_actual_tokens;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
  const int qkvz_elems;
  const int conv_elems;
};

template <typename T>
struct update_states_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = 4;
  static constexpr int elems_per_group = group_size * elems_per_item;

  update_states_kernel(
      T* conv_states,
      const int conv_states_stride_0,
      const T* conv_states_tmp,
      const int* cache_indices,
      const int width,
      const int conv_elems,
      const int32_t* query_start_loc,
      const int* token_indx,
      const int batch_size)
      : conv_states(conv_states),
        conv_states_stride_0(conv_states_stride_0),
        conv_states_tmp(conv_states_tmp),
        cache_indices(cache_indices),
        width(width),
        conv_elems(conv_elems),
        query_start_loc(query_start_loc),
        token_indx(token_indx),
        batch_size(batch_size) {}

  static inline sycl::nd_range<3>
  get_nd_range(const int batch_size, const int width, const int conv_elems) {
    const int groups_per_token =
        (conv_elems + elems_per_group - 1) / elems_per_group;
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(batch_size, (width - 1), groups_per_token);
    return sycl::nd_range<3>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    const int batch_id = item.get_group(0);
    const int width_id = item.get_group(1);
    const int local_group_id = item.get_group(2);
    const int local_id = item.get_local_linear_id();
    const int elems_start_offset_group = local_group_id * elems_per_group;

    int seq_start_offset = query_start_loc[batch_id];
    int seq_end_offset = query_start_loc[batch_id + 1];
    if (seq_end_offset - seq_start_offset == 1) {
      // only update if prefill
      return;
    }

    int states_id = cache_indices[batch_id];
    T* conv_states_ptr = conv_states + states_id * conv_states_stride_0;
    const T* conv_states_tmp_ptr =
        conv_states_tmp + batch_id * (width - 1) * conv_elems;
    for (int i = elems_start_offset_group + local_id;
         i < sycl::min((local_group_id + 1) * elems_per_group, conv_elems);
         i += group_size) {
      conv_states_ptr[width_id * conv_elems + i] =
          conv_states_tmp_ptr[width_id * conv_elems + i];
    }
  }

 private:
  T* conv_states;
  const int conv_states_stride_0;
  const T* conv_states_tmp;
  const int* cache_indices;
  const int width;
  const int conv_elems;
  const int32_t* query_start_loc;
  const int* token_indx;
  const int batch_size;
};

// -----------------------------------------------------------------------------
// Spec-decoding kernel for causal_conv1d.
//
// Each spec sequence contributes exactly num_spec_tokens (== num_spec + 1)
// contiguous tokens in the LOCAL q/k/v/b/a buffers, ordered by
// spec_query_start_loc; their corresponding GLOBAL positions in
// mixed_qkvz / mixed_ba / z_out are given by token_indx.
//
// Grid: (num_spec_decodes, groups_per_token). Each thread owns a feature
// chunk of `elems_per_item` lanes and walks all num_spec_tokens tokens of
// one sequence in a sliding window:
//   - q/k/v lanes: load Width-1 prior values from
//     conv_states[cache_indices[batch, num_accepted_tokens[batch] - 1]],
//     then for every t_local in [0, num_spec_tokens) shift the window left
//     by one, load the new input from mixed_qkvz[token_indx[...]], apply the
//     conv (+bias +activation) and store to local q/k/v outputs.
//   - z lanes: pure reorder, no conv.
//   - b/a lanes: pure reorder.
// After processing all tokens we write the trailing Width-1 input values to
// ONLY the last cache slot: conv_states[cache_indices[batch, num_spec]],
// matching the Triton fused_recurrent path which keeps a single rollback
// reference for the next forward.
// -----------------------------------------------------------------------------
template <typename T, int Width, bool ReorderInput>
struct causal_conv1d_spec_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = 4;
  static constexpr int elems_per_group = group_size * elems_per_item;

  causal_conv1d_spec_kernel(
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
      const int conv_states_stride_0,
      const int* token_indx,
      const int* cache_indices,
      const int cache_indices_stride_0,
      const int* num_accepted_tokens,
      const ActMode& act_mode,
      const int& pad_slot_id,
      const int& num_spec_decodes,
      const int& num_spec_tokens,
      const int& num_k_heads,
      const int& head_k_dim,
      const int& num_v_heads,
      const int& head_v_dim,
      const int& qkvz_elems,
      const int& conv_elems)
      : q_out(q_out),
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
        conv_states_stride_0(conv_states_stride_0),
        token_indx(token_indx),
        cache_indices(cache_indices),
        cache_indices_stride_0(cache_indices_stride_0),
        num_accepted_tokens(num_accepted_tokens),
        act_mode(act_mode),
        pad_slot_id(pad_slot_id),
        num_spec_decodes(num_spec_decodes),
        num_spec_tokens(num_spec_tokens),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim),
        qkvz_elems(qkvz_elems),
        conv_elems(conv_elems) {}

  static inline sycl::nd_range<2>
  get_nd_range(const int num_spec_decodes, const int qkvz_elems) {
    const int groups_per_token =
        (qkvz_elems + elems_per_group - 1) / elems_per_group;
    sycl::range<2> local(1, group_size);
    sycl::range<2> global(num_spec_decodes, groups_per_token);
    return sycl::nd_range<2>(global * local, local);
  }

  static inline void act_swish(float& x, float beta = 1.0f) {
    x = x / (1.0f + sycl::exp(-x * beta));
  }
  static inline void act_silu(float& x) { act_swish(x, 1.0f); }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int local_group_id = item.get_group(1);
    const int local_id = item.get_local_linear_id();
    const int qkvz_elems_id =
        local_group_id * elems_per_group + local_id * elems_per_item;

    if (qkvz_elems_id >= qkvz_elems) {
      return;
    }

    const int q_dim = head_k_dim;
    const int k_dim = head_k_dim;
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int z_dim = head_v_dim * num_v_heads / num_k_heads;
    const int qkvz_dim = q_dim + k_dim + v_dim + z_dim;

    int k_heads_id = qkvz_elems_id / qkvz_dim;
    int qkvz_dim_id = qkvz_elems_id % qkvz_dim;

    // Determine role (q/k/v/z) and the mixed_qkvz offset for this feature.
    bool is_q = false, is_k = false, is_v = false, is_z = false;
    int mixed_qkvz_id = qkvz_elems_id;
    int reordered_elems_id = 0;
    if (qkvz_dim_id < q_dim) {
      is_q = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = k_heads_id * k_dim + qkvz_dim_id;
      }
      reordered_elems_id = k_heads_id * q_dim + qkvz_dim_id;
    } else if (qkvz_dim_id < q_dim + k_dim) {
      is_k = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id =
            num_k_heads * head_k_dim + k_heads_id * k_dim + qkvz_dim_id - q_dim;
      }
      reordered_elems_id =
          num_k_heads * q_dim + k_heads_id * k_dim + qkvz_dim_id - q_dim;
    } else if (qkvz_dim_id < q_dim + k_dim + v_dim) {
      is_v = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = 2 * num_k_heads * head_k_dim + k_heads_id * v_dim +
                        qkvz_dim_id - (q_dim + k_dim);
      }
      reordered_elems_id = num_k_heads * (q_dim + k_dim) + k_heads_id * v_dim +
                           qkvz_dim_id - (q_dim + k_dim);
    } else {
      is_z = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = 2 * num_k_heads * head_k_dim +
                        num_v_heads * head_v_dim + k_heads_id * z_dim +
                        qkvz_dim_id - (q_dim + k_dim + v_dim);
      }
    }

    // -- b / a reorder (mixed_ba GLOBAL → b_out/a_out LOCAL) ----------------
    for (int t_local = 0; t_local < num_spec_tokens; ++t_local) {
      const int token_id_local = batch_id * num_spec_tokens + t_local;
      const int global_t = token_indx[token_id_local];
      if constexpr (ReorderInput) {
        if (qkvz_elems_id < num_v_heads) {
          int step_local = token_id_local * num_v_heads;
          int step_global = global_t * num_v_heads;
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            b_out[step_local + qkvz_elems_id + e] =
                mixed_ba[step_global * 2 + qkvz_elems_id + e];
            a_out[step_local + qkvz_elems_id + e] =
                mixed_ba[step_global * 2 + num_v_heads + qkvz_dim_id + e];
          }
        }
      } else {
        if (qkvz_dim_id < (num_v_heads / num_k_heads)) {
          int step_local = token_id_local * num_v_heads +
                           k_heads_id * num_v_heads / num_k_heads;
          int step_global =
              global_t * num_v_heads + k_heads_id * num_v_heads / num_k_heads;
          const int ba_elems_per_item =
              sycl::min(elems_per_item, num_v_heads / num_k_heads);
#pragma unroll
          for (int e = 0; e < ba_elems_per_item; ++e) {
            b_out[step_local + qkvz_dim_id + e] =
                mixed_ba[step_global * 2 + qkvz_dim_id + e];
            a_out[step_local + qkvz_dim_id + e] = mixed_ba
                [step_global * 2 + num_v_heads / num_k_heads + qkvz_dim_id + e];
          }
        }
      }
    }

    // -- z reorder: simple copy through token_indx, no convolution. ---------
    if (is_z) {
      int z_elems_id =
          k_heads_id * z_dim + qkvz_dim_id - (q_dim + k_dim + v_dim);
      for (int t_local = 0; t_local < num_spec_tokens; ++t_local) {
        const int token_id_local = batch_id * num_spec_tokens + t_local;
        const int global_t = token_indx[token_id_local];
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          z_out[global_t * num_k_heads * z_dim + z_elems_id + e] =
              mixed_qkvz[global_t * qkvz_elems + mixed_qkvz_id + e];
        }
      }
      return;
    }

    if (!(is_q || is_k || is_v)) {
      return;
    }

    // -- conv1d on q/k/v lanes ----------------------------------------------
    // Pick the initial conv-state slot using num_accepted_tokens. The slot
    // at column num_accepted_tokens[batch_id] - 1 was written by the previous
    // step's last accepted token; we fall back to col 0 defensively when
    // num_accepted_tokens == 0.
    int init_col = num_accepted_tokens[batch_id] - 1;
    if (init_col < 0) init_col = 0;
    const int init_state_id =
        cache_indices[batch_id * cache_indices_stride_0 + init_col];

    // Load weights
    T local_weights[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_weights[Width * e + i] =
            conv_weights[(reordered_elems_id + e) * Width + i];
      }
    }

    // Sliding window of size Width. Positions [0, Width-1) start with the
    // prior conv_state; position Width-1 is filled per-iter with the new
    // input. When the cache slot is the pad slot the prior is zeroed.
    T local_input[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width * elems_per_item; ++i) {
      local_input[i] = static_cast<T>(0);
    }
    if (Width > 1 && init_state_id != pad_slot_id) {
      const T* init_state_ptr =
          conv_states + init_state_id * conv_states_stride_0;
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          local_input[Width * e + i] =
              init_state_ptr[i * conv_elems + reordered_elems_id + e];
        }
      }
    }

    for (int t_local = 0; t_local < num_spec_tokens; ++t_local) {
      const int token_id_local = batch_id * num_spec_tokens + t_local;
      const int global_t = token_indx[token_id_local];

      // Shift window left by 1 (for t_local == 0 the trailing slot is fresh).
      if (t_local > 0) {
#pragma unroll
        for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            local_input[Width * e + i] = local_input[Width * e + i + 1];
          }
        }
      }
      // Load new input at the trailing slot.
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + Width - 1] =
            mixed_qkvz[global_t * qkvz_elems + mixed_qkvz_id + e];
      }

      float res[elems_per_item];
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        res[e] = 0.0f;
      }
#pragma unroll
      for (int i = 0; i < Width; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          res[e] += static_cast<float>(local_input[Width * e + i]) *
                    static_cast<float>(local_weights[Width * e + i]);
        }
      }
      if (conv_bias != nullptr) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          res[e] += conv_bias[reordered_elems_id + e];
        }
      }
      if (act_mode == ActMode::silu) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          act_silu(res[e]);
        }
      } else if (act_mode == ActMode::swish) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          act_swish(res[e]);
        }
      }

      // Write to LOCAL q/k/v at token_id_local.
      if (is_q) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          q_out
              [token_id_local * num_k_heads * q_dim + k_heads_id * q_dim +
               qkvz_dim_id + e] = res[e];
        }
      } else if (is_k) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          k_out
              [token_id_local * num_k_heads * k_dim + k_heads_id * k_dim +
               qkvz_dim_id - q_dim + e] = res[e];
        }
      } else {  // is_v
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          v_out
              [token_id_local * num_k_heads * v_dim + k_heads_id * v_dim +
               qkvz_dim_id - (q_dim + k_dim) + e] = res[e];
        }
      }

      // Checkpoint the rolling conv state at every step into the cache slot
      // that the next decoding round will read when `num_accepted_tokens ==
      // t_local + 1`. Writing each column (not only the last one) keeps the
      // scheduler's per-acceptance rollback consistent: column `t_local`
      // holds the conv state right after consuming spec token `t_local`.
      if (Width > 1) {
        const int save_state_id =
            cache_indices[batch_id * cache_indices_stride_0 + t_local];
        if (save_state_id != pad_slot_id) {
          T* save_state_ptr =
              conv_states + save_state_id * conv_states_stride_0;
#pragma unroll
          for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
            for (int e = 0; e < elems_per_item; ++e) {
              save_state_ptr[i * conv_elems + reordered_elems_id + e] =
                  local_input[Width * e + i + 1];
            }
          }
        }
      }
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
  const int conv_states_stride_0;
  const int* token_indx;
  const int* cache_indices;
  const int cache_indices_stride_0;
  const int* num_accepted_tokens;
  const ActMode act_mode;
  const int pad_slot_id;
  const int num_spec_decodes;
  const int num_spec_tokens;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
  const int qkvz_elems;
  const int conv_elems;
};

template <typename T, int Width, bool ReorderInput>
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
    const int conv_states_stride_0,
    T* conv_states_tmp,
    int* query_start_loc,
    int* token_indx,
    int* cache_indices,
    const int cache_indices_stride_0,
    bool* has_initial_state,
    int* num_accepted_tokens,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int& batch_size,
    const int& num_virtual_tokens,
    const int& num_actual_tokens,
    const int& num_k_heads,
    const int& head_k_dim,
    const int& num_v_heads,
    const int& head_v_dim,
    const int& qkvz_elems,
    const int& conv_elems,
    const int& num_prefills,
    const int& num_decodes,
    const int& num_spec_decodes,
    const int& num_spec_tokens) {
  // Spec path: state writeback happens inside the spec kernel, so we do not
  // run the auxiliary update_states_kernel for prefills here.
  if (num_accepted_tokens != nullptr) {
    using KERNEL_SPEC = causal_conv1d_spec_kernel<T, Width, ReorderInput>;
    auto range_spec = KERNEL_SPEC::get_nd_range(num_spec_decodes, qkvz_elems);
    assert(head_k_dim % KERNEL_SPEC::elems_per_item == 0);
    assert(num_v_heads % KERNEL_SPEC::elems_per_item == 0);
    queue.submit([&](sycl::handler& cgh) {
      KERNEL_SPEC task(
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
          conv_states_stride_0,
          token_indx,
          cache_indices,
          cache_indices_stride_0,
          num_accepted_tokens,
          act_mode,
          pad_slot_id,
          num_spec_decodes,
          num_spec_tokens,
          num_k_heads,
          head_k_dim,
          num_v_heads,
          head_v_dim,
          qkvz_elems,
          conv_elems);
      cgh.parallel_for(range_spec, task);
    });
    return;
  }

  using KERNEL_MAIN = causal_conv1d_kernel<T, Width, ReorderInput>;
  auto range_main = KERNEL_MAIN::get_nd_range(num_virtual_tokens, qkvz_elems);
  assert(head_k_dim % KERNEL_MAIN::elems_per_item == 0);
  assert(num_v_heads % KERNEL_MAIN::elems_per_item == 0);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL_MAIN task(
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
        conv_states_stride_0,
        conv_states_tmp,
        query_start_loc,
        token_indx,
        cache_indices,
        has_initial_state,
        num_accepted_tokens,
        act_mode,
        pad_slot_id,
        batch_size,
        num_virtual_tokens,
        num_actual_tokens,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        qkvz_elems,
        conv_elems);
    cgh.parallel_for(range_main, task);
  });
  if (num_prefills > 0) {
    using KERNEL_UPDATE = update_states_kernel<T>;
    auto range_update =
        KERNEL_UPDATE::get_nd_range(batch_size, Width, conv_elems);
    queue.submit([&](sycl::handler& cgh) {
      KERNEL_UPDATE task(
          conv_states,
          conv_states_stride_0,
          conv_states_tmp,
          cache_indices,
          Width,
          conv_elems,
          query_start_loc,
          token_indx,
          batch_size);
      cgh.parallel_for(range_update, task);
    });
  }
}

void causal_conv1d(
    sycl::queue& queue,
    torch::Tensor& q_out,  // [num_virtual_tokens, num_k_heads, head_k_dim]
    torch::Tensor& k_out,  // [num_virtual_tokens, num_k_heads, head_k_dim]
    torch::Tensor& v_out,  // [num_virtual_tokens, num_v_heads, head_v_dim]
    torch::Tensor& z_out,  // [num_actual_tokens, num_v_heads, head_v_dim]
    torch::Tensor& b_out,  // [num_virtual_tokens, num_v_heads]
    torch::Tensor& a_out,  // [num_virtual_tokens, num_v_heads]
    const torch::Tensor&
        mixed_qkvz,  // [num_actual_tokens, num_k_heads * (2 * head_k_dim + 2 *
                     // head_v_dim * num_v_heads / num_k_heads)]
    const torch::Tensor& mixed_ba,  // [num_actual_tokens, num_k_heads * (2 *
                                    // num_v_heads / num_k_heads)]
    const torch::Tensor&
        conv_weights,  // [num_k_heads * (2 * head_k_dim + head_v_dim *
                       // num_v_heads / num_k_heads), width]
    const std::optional<torch::Tensor>&
        conv_bias,  // [num_k_heads * (2 * head_k_dim + head_v_dim * num_v_heads
                    // / num_k_heads)] or None
    torch::Tensor&
        conv_states,  // [cache_batch_size, width - 1, num_k_heads * (2 *
                      // head_k_dim + head_v_dim * num_v_heads / num_k_heads)]
    const std::optional<torch::Tensor>& query_start_loc,  // [batch_size + 1]
    const std::optional<torch::Tensor>&
        token_indx,  // [num_virtual_tokens] or None
    const std::optional<torch::Tensor>& cache_indices,  // [batch_size]
    const std::optional<torch::Tensor>&
        has_initial_state,  // [batch_size] or None
    const std::optional<torch::Tensor>&
        num_accepted_tokens,  // [batch_size] or None
    const ActMode& act_mode,  // silu or swish
    const int& pad_slot_id,   // -1
    const int num_prefills,
    const int num_decodes,
    const int num_spec_decodes,
    const bool reorder_input) {
  TORCH_CHECK(query_start_loc.has_value() && cache_indices.has_value());

  const bool is_spec = num_accepted_tokens.has_value();
  const int batch_size = query_start_loc->size(0) - 1;
  const int num_virtual_tokens = q_out.size(0);
  const int num_actual_tokens = mixed_qkvz.size(0);
  const int num_k_heads = q_out.size(1);
  const int head_k_dim = q_out.size(2);
  const int num_v_heads = v_out.size(1);
  const int head_v_dim = v_out.size(2);
  const int qkvz_elems = mixed_qkvz.size(1);
  const int conv_elems = conv_weights.size(0);
  const int width = conv_weights.size(1);
  const int conv_states_stride_0 = conv_states.stride(0);

  int num_spec_tokens = 0;
  int cache_indices_stride_0 = 0;
  if (is_spec) {
    TORCH_CHECK(cache_indices->dim() == 2);
    TORCH_CHECK(cache_indices->size(0) == num_spec_decodes);
    num_spec_tokens = cache_indices->size(1);
    cache_indices_stride_0 = cache_indices->stride(0);
    TORCH_CHECK(num_accepted_tokens->size(0) == num_spec_decodes);
  }

  auto dtype = conv_states.dtype();
  auto device = conv_states.device();
  torch::Tensor conv_states_tmp = torch::empty(
      {std::max(batch_size, 1), width - 1, conv_elems},
      torch::dtype(dtype).device(device).requires_grad(false));

#define KERNEL_LAUNCHER(scalar_t, width, reorder_input)                       \
  kernel_launcher<scalar_t, width, reorder_input>(                            \
      queue,                                                                  \
      reinterpret_cast<scalar_t*>(q_out.data_ptr()),                          \
      reinterpret_cast<scalar_t*>(k_out.data_ptr()),                          \
      reinterpret_cast<scalar_t*>(v_out.data_ptr()),                          \
      reinterpret_cast<scalar_t*>(z_out.data_ptr()),                          \
      reinterpret_cast<scalar_t*>(b_out.data_ptr()),                          \
      reinterpret_cast<scalar_t*>(a_out.data_ptr()),                          \
      reinterpret_cast<scalar_t*>(mixed_qkvz.data_ptr()),                     \
      reinterpret_cast<scalar_t*>(mixed_ba.data_ptr()),                       \
      reinterpret_cast<scalar_t*>(conv_weights.data_ptr()),                   \
      conv_bias.has_value()                                                   \
          ? reinterpret_cast<scalar_t*>(conv_bias->data_ptr())                \
          : nullptr,                                                          \
      reinterpret_cast<scalar_t*>(conv_states.data_ptr()),                    \
      conv_states_stride_0,                                                   \
      reinterpret_cast<scalar_t*>(conv_states_tmp.data_ptr()),                \
      reinterpret_cast<int*>(query_start_loc->data_ptr()),                    \
      token_indx.has_value() ? reinterpret_cast<int*>(token_indx->data_ptr()) \
                             : nullptr,                                       \
      reinterpret_cast<int*>(cache_indices->data_ptr()),                      \
      cache_indices_stride_0,                                                 \
      has_initial_state.has_value()                                           \
          ? reinterpret_cast<bool*>(has_initial_state->data_ptr())            \
          : nullptr,                                                          \
      num_accepted_tokens.has_value()                                         \
          ? reinterpret_cast<int*>(num_accepted_tokens->data_ptr())           \
          : nullptr,                                                          \
      act_mode,                                                               \
      pad_slot_id,                                                            \
      batch_size,                                                             \
      num_virtual_tokens,                                                     \
      num_actual_tokens,                                                      \
      num_k_heads,                                                            \
      head_k_dim,                                                             \
      num_v_heads,                                                            \
      head_v_dim,                                                             \
      qkvz_elems,                                                             \
      conv_elems,                                                             \
      num_prefills,                                                           \
      num_decodes,                                                            \
      num_spec_decodes,                                                       \
      num_spec_tokens);

#define WIDTH_DISPATCH(scalar_t, width, reorder_input) \
  switch (width) {                                     \
    case 1:                                            \
      KERNEL_LAUNCHER(scalar_t, 1, reorder_input)      \
      break;                                           \
    case 2:                                            \
      KERNEL_LAUNCHER(scalar_t, 2, reorder_input)      \
      break;                                           \
    case 3:                                            \
      KERNEL_LAUNCHER(scalar_t, 3, reorder_input)      \
      break;                                           \
    case 4:                                            \
      KERNEL_LAUNCHER(scalar_t, 4, reorder_input)      \
      break;                                           \
    case 5:                                            \
      KERNEL_LAUNCHER(scalar_t, 5, reorder_input)      \
      break;                                           \
    default:                                           \
      break;                                           \
  }

#define SPLIT_DISPATCH(scalar_t, width, reorder_input) \
  if (reorder_input) {                                 \
    WIDTH_DISPATCH(scalar_t, width, true)              \
  } else {                                             \
    WIDTH_DISPATCH(scalar_t, width, false)             \
  }

  if (mixed_qkvz.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    SPLIT_DISPATCH(scalar_t, width, reorder_input)
  } else if (mixed_qkvz.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    SPLIT_DISPATCH(scalar_t, width, reorder_input)
  } else {
    using scalar_t = float;
    SPLIT_DISPATCH(scalar_t, width, reorder_input)
  }
#undef SPLIT_DISPATCH
#undef WIDTH_DISPATCH
#undef KERNEL_LAUNCHER
}

}  // namespace gdn
