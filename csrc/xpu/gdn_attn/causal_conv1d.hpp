#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "gdn_attn_utils.h"

namespace gdn {

template <typename T, int Width, bool ReorderInput, bool IS_SPEC = false>
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
      const int& qkvz_elems,
      const int& conv_elems,
      // Spec-decoding (only consulted when IS_SPEC=true). Per-token store
      // slot comes from spec_state_indices[batch_id, t - seq_start]; the
      // load slot for the partial-window branch comes from
      // spec_state_indices[batch_id, num_accepted_tokens[batch_id]-1].
      const int* spec_state_indices = nullptr,
      const int* num_accepted_tokens = nullptr,
      const int spec_stride = 0,
      const int state_len = 0)
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
        qkvz_elems(qkvz_elems),
        conv_elems(conv_elems),
        spec_state_indices(spec_state_indices),
        num_accepted_tokens(num_accepted_tokens),
        spec_stride(spec_stride),
        state_len(state_len) {}

  static inline sycl::nd_range<2>
  get_nd_range(const int total_seqlen, const int qkvz_elems) {
    const int groups_per_token =
        (qkvz_elems + elems_per_group - 1) / elems_per_group;
    sycl::range<2> local(1, group_size);
    sycl::range<2> global(total_seqlen, groups_per_token);
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

    const int q_dim = head_k_dim;
    const int k_dim = head_k_dim;
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int z_dim = head_v_dim * num_v_heads / num_k_heads;
    const int qkvz_dim = q_dim + k_dim + v_dim + z_dim;

    int k_heads_id = qkvz_elems_id / qkvz_dim;
    int qkvz_dim_id = qkvz_elems_id % qkvz_dim;

    // reorder b,a
    if constexpr (ReorderInput) {
      if (qkvz_elems_id < num_v_heads) {
        int step = token_id * num_v_heads;
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          b_out[step + qkvz_elems_id + e] =
              mixed_ba[step * 2 + qkvz_elems_id + e];
          a_out[step + qkvz_elems_id + e] =
              mixed_ba[step * 2 + num_v_heads + qkvz_dim_id + e];
        }
      }
    } else {
      if (qkvz_dim_id < (num_v_heads / num_k_heads)) {
        int step =
            token_id * num_v_heads + k_heads_id * num_v_heads / num_k_heads;
        const int ba_elems_per_item =
            sycl::min(elems_per_item, num_v_heads / num_k_heads);
#pragma unroll
        for (int e = 0; e < ba_elems_per_item; ++e) {
          b_out[step + qkvz_dim_id + e] = mixed_ba[step * 2 + qkvz_dim_id + e];
          a_out[step + qkvz_dim_id + e] =
              mixed_ba[step * 2 + num_v_heads / num_k_heads + qkvz_dim_id + e];
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

    // get states cache location. For spec-decoding (FLA-aligned, tick 40):
    // ALL spec tokens read/write the single slot at cache_indices[batch_id]
    // (= spec_state_indices[batch_id, 0] per the dispatcher), with a rolling
    // history of state_len positions inside it. The history starting offset
    // within the slot is (num_accepted_tokens - 1).
    int states_id;
    int spec_store_slot = pad_slot_id;
    bool spec_skip_init_load = false;
    if constexpr (IS_SPEC) {
      const int n_acc = num_accepted_tokens[batch_id];
      if (n_acc <= 0) {
        // (Rung 6) num_accepted=0: no valid prior state to load.
        states_id = 0;
        spec_skip_init_load = true;
      } else {
        states_id = cache_indices[batch_id];
        if (states_id <= 0) {
          // FLA NULL_BLOCK_ID convention: treat 0/-1 as no prior state.
          states_id = 0;
          spec_skip_init_load = true;
        }
      }
      const int t_in_seq = token_id - seq_start_offset;
      spec_store_slot = spec_state_indices[batch_id * spec_stride + t_in_seq];
    } else {
      states_id = cache_indices[batch_id];
    }

    if constexpr (!IS_SPEC) {
      if (states_id == pad_slot_id) {
        return;
      }
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

    // reorder z
    if (is_z) {
      int z_elems_id =
          k_heads_id * z_dim + qkvz_dim_id - (q_dim + k_dim + v_dim);
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        z_out[token_id * num_k_heads * z_dim + z_elems_id + e] =
            mixed_qkvz[token_id * qkvz_elems + mixed_qkvz_id + e];
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
    bool has_init_conv_states =
        (has_initial_state == nullptr ||
         (has_initial_state != nullptr && has_initial_state[batch_id]));
    if constexpr (IS_SPEC) {
      // (Rung 6) Spec overrides: when num_accepted=0 or the load slot is
      // the NULL sentinel, the partial-window load must NOT touch
      // conv_states_ptr (no valid prior data); the kernel still proceeds
      // and writes per-token state.
      if (spec_skip_init_load) {
        has_init_conv_states = false;
      }
    }
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
    // For IS_SPEC, the slot holds state_len > Width-1 positions and the
    // history of interest starts at offset (n_acc - 1). For non-spec, the
    // slot is exactly Width-1 positions and offset is 0.
    int load_offset_shift = 0;
    if constexpr (IS_SPEC) {
      const int n_acc_local = num_accepted_tokens[batch_id];
      if (n_acc_local > 0) {
        load_offset_shift = n_acc_local - 1;
      }
    }
    if (states_load_len != 0 && has_init_conv_states) {
#pragma unroll
      for (int i = 0; i < states_load_len; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          local_input[Width * e + i] = conv_states_ptr
              [(Width - 1 - states_load_len + i + load_offset_shift) *
                   conv_elems +
               reordered_elems_id + e];
        }
      }
    }

#pragma unroll
    for (int i = 0; i < input_load_len; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + states_load_len + i] = mixed_qkvz
            [(token_id - input_load_len + 1 + i) * qkvz_elems + mixed_qkvz_id +
             e];
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
    if constexpr (IS_SPEC) {
      // (Tick 40) FLA-aligned spec writeback: stage the rolled state into
      // conv_states_tmp[batch_id, state_len, conv_elems]. A second kernel
      // (spec_state_roll_kernel) copies tmp -> slot[cache_indices[batch_id]]
      // after the main kernel finishes (race-free because slot is read by
      // pass 1 but only written by pass 2).
      //
      // Per-position semantics for slot[cache_indices[batch_id]]:
      //   new[p] = pre[n_acc + p]                  for p < state_len - K_call
      //   new[p] = current_input[p - (state_len - K_call)]  for p >= state_len
      //   - K_call
      //
      // Each work-item handles t_in_seq = token_id - seq_start_offset:
      //   - Always: writes tmp at position (state_len - K_call + t_in_seq) =
      //     the spec token's contribution to the rolled state.
      //   - If t_in_seq < state_len - K_call: also writes tmp at position
      //     t_in_seq with the shift-copy of pre[n_acc + t_in_seq].
      const int K_call = seq_end_offset - seq_start_offset;
      const int n_acc_safe = (num_accepted_tokens[batch_id] > 0)
                                 ? num_accepted_tokens[batch_id]
                                 : 0;
      const int t_in_seq = token_id - seq_start_offset;
      const int new_input_pos = state_len - K_call + t_in_seq;
      // Latest input value for this token = local_input[Width-1].
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        if (new_input_pos >= 0 && new_input_pos < state_len) {
          conv_states_tmp
              [batch_id * state_len * conv_elems + new_input_pos * conv_elems +
               reordered_elems_id + e] = local_input[Width * e + Width - 1];
        }
      }
      // Shift-copy: only the first (state_len - K_call) work items per
      // dim chunk read pre[n_acc + t_in_seq] and stage it at position
      // t_in_seq in tmp.
      if (t_in_seq < state_len - K_call) {
        const int shift_src_pos = n_acc_safe + t_in_seq;
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          T pre_val = T(0);
          if (!spec_skip_init_load && shift_src_pos < state_len) {
            pre_val = conv_states_ptr
                [shift_src_pos * conv_elems + reordered_elems_id + e];
          }
          conv_states_tmp
              [batch_id * state_len * conv_elems + t_in_seq * conv_elems +
               reordered_elems_id + e] = pre_val;
        }
      }
    } else if (seq_end_offset - seq_start_offset > 1) {
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
  const int qkvz_elems;
  const int conv_elems;
  // Spec-decoding inputs (consulted when IS_SPEC=true). See ctor docstring.
  const int* spec_state_indices;
  const int* num_accepted_tokens;
  const int spec_stride;
  // state_len = conv_states_stride_0 / conv_elems. Number of time positions
  // per slot in the (slot, time, dim) conv state buffer. For non-spec this
  // equals Width-1; for spec it can be larger (e.g. state_len = (Width-1) +
  // (K_max - 1)). Plumbed in to drive the FLA-aligned rolled writeback.
  const int state_len;
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
      const int batch_size)
      : conv_states(conv_states),
        conv_states_stride_0(conv_states_stride_0),
        conv_states_tmp(conv_states_tmp),
        cache_indices(cache_indices),
        width(width),
        conv_elems(conv_elems),
        query_start_loc(query_start_loc),
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
         i < (local_group_id + 1) * elems_per_group;
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
  const int batch_size;
};

// Tick-40 spec-state roll kernel. Pass 2 of the FLA-aligned conv1d_update:
// copies the rolled state staged by pass 1 in conv_states_tmp[batch_id,
// state_len, conv_elems] to conv_states[cache_indices[batch_id], state_len,
// conv_elems]. Race-free because pass 1 only writes tmp; pass 2 only writes
// the slot.
template <typename T>
struct spec_state_roll_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = 4;
  static constexpr int elems_per_group = group_size * elems_per_item;

  spec_state_roll_kernel(
      T* conv_states,
      const int conv_states_stride_0,
      const T* conv_states_tmp,
      const int* cache_indices,
      const int state_len,
      const int conv_elems,
      const int batch_size)
      : conv_states(conv_states),
        conv_states_stride_0(conv_states_stride_0),
        conv_states_tmp(conv_states_tmp),
        cache_indices(cache_indices),
        state_len(state_len),
        conv_elems(conv_elems),
        batch_size(batch_size) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int batch_size, const int state_len, const int conv_elems) {
    const int groups_per_token =
        (conv_elems + elems_per_group - 1) / elems_per_group;
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(batch_size, state_len, groups_per_token);
    return sycl::nd_range<3>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    const int batch_id = item.get_group(0);
    const int pos_id = item.get_group(1);
    const int local_group_id = item.get_group(2);
    const int local_id = item.get_local_linear_id();
    const int elems_start_offset_group = local_group_id * elems_per_group;

    const int states_id = cache_indices[batch_id];
    if (states_id <= 0) {
      return;  // null sentinel: no slot to write
    }
    T* conv_states_ptr =
        conv_states + static_cast<int64_t>(states_id) * conv_states_stride_0;
    const T* conv_states_tmp_ptr =
        conv_states_tmp + batch_id * state_len * conv_elems;
    for (int i = elems_start_offset_group + local_id;
         i < elems_start_offset_group + elems_per_group && i < conv_elems;
         i += group_size) {
      conv_states_ptr[pos_id * conv_elems + i] =
          conv_states_tmp_ptr[pos_id * conv_elems + i];
    }
  }

 private:
  T* conv_states;
  const int conv_states_stride_0;
  const T* conv_states_tmp;
  const int* cache_indices;
  const int state_len;
  const int conv_elems;
  const int batch_size;
};

template <typename T, int Width, bool ReorderInput, bool IS_SPEC = false>
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
    const int& qkvz_elems,
    const int& conv_elems,
    const int& num_prefills,
    const int& num_decodes,
    const int* spec_state_indices = nullptr,
    const int* num_accepted_tokens = nullptr,
    const int spec_stride = 0,
    const int state_len = 0) {
  using KERNEL_MAIN = causal_conv1d_kernel<T, Width, ReorderInput, IS_SPEC>;
  auto range_main = KERNEL_MAIN::get_nd_range(num_actual_tokens, qkvz_elems);
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
        qkvz_elems,
        conv_elems,
        spec_state_indices,
        num_accepted_tokens,
        spec_stride,
        state_len);
    cgh.parallel_for(range_main, task);
  });
  if constexpr (!IS_SPEC) {
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
            batch_size);
        cgh.parallel_for(range_update, task);
      });
    }
  } else {
    // Tick-40 pass 2: copy the rolled state staged in conv_states_tmp
    // (shape [batch, state_len, conv_elems]) into the slot at
    // cache_indices[batch_id]. Race-free vs pass 1 because pass 1 only
    // wrote tmp; this pass is the only writer of slot.
    using KERNEL_ROLL = spec_state_roll_kernel<T>;
    auto range_roll =
        KERNEL_ROLL::get_nd_range(batch_size, state_len, conv_elems);
    queue.submit([&](sycl::handler& cgh) {
      KERNEL_ROLL task(
          conv_states,
          conv_states_stride_0,
          conv_states_tmp,
          cache_indices,
          state_len,
          conv_elems,
          batch_size);
      cgh.parallel_for(range_roll, task);
    });
  }
}

void causal_conv1d(
    sycl::queue& queue,
    torch::Tensor& q_out,  // [total_seqlen, num_k_heads, head_k_dim]
    torch::Tensor& k_out,  // [total_seqlen, num_k_heads, head_k_dim]
    torch::Tensor& v_out,  // [total_seqlen, num_v_heads, head_v_dim]
    torch::Tensor& z_out,  // [total_seqlen, num_v_heads, head_v_dim]
    torch::Tensor& b_out,  // [total_seqlen, num_v_heads]
    torch::Tensor& a_out,  // [total_seqlen, num_v_heads]
    const torch::Tensor&
        mixed_qkvz,  // [total_seqlen, num_k_heads * (2 * head_k_dim + 2 *
                     // head_v_dim * num_v_heads / num_k_heads)]
    const torch::Tensor& mixed_ba,  // [total_seqlen, num_k_heads * (2 *
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
    const torch::Tensor& query_start_loc,  // [batch_size + 1]
    const torch::Tensor& cache_indices,    // [batch_size]
    const std::optional<torch::Tensor>&
        has_initial_state,    // [batch_size] or None
    const ActMode& act_mode,  // silu or swish
    const int& pad_slot_id,   // -1
    const int num_prefills,
    const int num_decodes,
    const bool reorder_input,
    // Spec-decoding extension. When is_spec=true the per-sequence load slot
    // and per-token store slot come from these tensors; cache_indices is
    // ignored. Both tensors are int32 [batch_size, K] / [batch_size].
    const bool is_spec = false,
    const int* spec_state_indices_ptr = nullptr,
    const int* num_accepted_tokens_ptr = nullptr,
    const int spec_stride = 0) {
  if (num_prefills == 0 && num_decodes == 0) {
    return;
  }

  const int batch_size = query_start_loc.size(0) - 1;
  const int num_actual_tokens = q_out.size(0);
  const int num_k_heads = q_out.size(1);
  const int head_k_dim = q_out.size(2);
  const int num_v_heads = v_out.size(1);
  const int head_v_dim = v_out.size(2);
  const int qkvz_elems = mixed_qkvz.size(1);
  const int conv_elems = conv_weights.size(0);
  const int width = conv_weights.size(1);
  const int conv_states_stride_0 = conv_states.stride(0);
  // (Tick 40) state_len = number of time positions per slot. For non-spec,
  // the buffer is sized to width-1 by convention; for spec, the runtime
  // sizes it larger (e.g. (Width-1) + (K_max-1) = 6 for Width=4 K_max=4).
  // Derive from stride; conv_states is row-major (slot, time, dim).
  const int state_len = conv_states_stride_0 / conv_elems;
  const int tmp_rows = is_spec ? state_len : (width - 1);

  auto dtype = conv_states.dtype();
  auto device = conv_states.device();
  torch::Tensor conv_states_tmp = torch::empty(
      {batch_size, tmp_rows, conv_elems},
      torch::dtype(dtype).device(device).requires_grad(false));

#define KERNEL_LAUNCHER_IMPL(scalar_t, width, reorder_input, IS_SPEC) \
  kernel_launcher<scalar_t, width, reorder_input, IS_SPEC>(           \
      queue,                                                          \
      reinterpret_cast<scalar_t*>(q_out.data_ptr()),                  \
      reinterpret_cast<scalar_t*>(k_out.data_ptr()),                  \
      reinterpret_cast<scalar_t*>(v_out.data_ptr()),                  \
      reinterpret_cast<scalar_t*>(z_out.data_ptr()),                  \
      reinterpret_cast<scalar_t*>(b_out.data_ptr()),                  \
      reinterpret_cast<scalar_t*>(a_out.data_ptr()),                  \
      reinterpret_cast<scalar_t*>(mixed_qkvz.data_ptr()),             \
      reinterpret_cast<scalar_t*>(mixed_ba.data_ptr()),               \
      reinterpret_cast<scalar_t*>(conv_weights.data_ptr()),           \
      conv_bias.has_value()                                           \
          ? reinterpret_cast<scalar_t*>(conv_bias->data_ptr())        \
          : nullptr,                                                  \
      reinterpret_cast<scalar_t*>(conv_states.data_ptr()),            \
      conv_states_stride_0,                                           \
      reinterpret_cast<scalar_t*>(conv_states_tmp.data_ptr()),        \
      reinterpret_cast<int*>(query_start_loc.data_ptr()),             \
      reinterpret_cast<int*>(cache_indices.data_ptr()),               \
      has_initial_state.has_value()                                   \
          ? reinterpret_cast<bool*>(has_initial_state->data_ptr())    \
          : nullptr,                                                  \
      act_mode,                                                       \
      pad_slot_id,                                                    \
      batch_size,                                                     \
      num_actual_tokens,                                              \
      num_k_heads,                                                    \
      head_k_dim,                                                     \
      num_v_heads,                                                    \
      head_v_dim,                                                     \
      qkvz_elems,                                                     \
      conv_elems,                                                     \
      num_prefills,                                                   \
      num_decodes,                                                    \
      spec_state_indices_ptr,                                         \
      num_accepted_tokens_ptr,                                        \
      spec_stride,                                                    \
      state_len);

#define KERNEL_LAUNCHER(scalar_t, width, reorder_input)         \
  if (is_spec) {                                                \
    KERNEL_LAUNCHER_IMPL(scalar_t, width, reorder_input, true)  \
  } else {                                                      \
    KERNEL_LAUNCHER_IMPL(scalar_t, width, reorder_input, false) \
  }

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
#undef KERNEL_LAUNCHER_IMPL
}

}  // namespace gdn