#pragma once

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "utils.h"
#include "gdn_attn_utils.h"

namespace gdn {
static constexpr int chunk_size = gdn::chunk_size_xe2;

template <typename VecT>
static inline bool is_aligned_vec_ptr(const void* ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignof(VecT) == 0;
}

template <typename T, int Width, bool ReorderInput>
struct chunk_causal_conv1d_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int elems_per_item = 4;

  static inline float act_sigmoid(float& x) {
    return 1.0f / (1.0f + sycl::exp(-x));
  }

  chunk_causal_conv1d_kernel(
      T* q_out,
      T* k_out,
      T* v_out,
      T* z_out,
      float* b_out,
      float* a_out,
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
      const int& num_virtual_tokens,
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
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
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

  static inline sycl::nd_range<2> get_nd_range(
      const int total_seqlen,
      const int num_k_heads,
      const int head_k_dim,
      const int num_v_heads,
      const int head_v_dim) {
    const int group_size =
        (2 * head_k_dim + head_v_dim * num_v_heads / num_k_heads) /
        elems_per_item;
    assert(num_v_heads % num_k_heads == 0);
    assert(head_k_dim % elems_per_item == 0);
    assert(head_v_dim % elems_per_item == 0);
    sycl::range<2> local(1, group_size);
    sycl::range<2> global(total_seqlen, num_k_heads);
    return sycl::nd_range<2>(global * local, local);
  }

  static inline void act_swish(float& x, float beta = 1.0f) {
    x = x / (1.0f + sycl::exp(-x * beta));
  }
  static inline void act_silu(float& x) { act_swish(x, 1.0f); }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int token_id = item.get_group(0);
    const int k_head_id = item.get_group(1);
    const int local_id = item.get_local_linear_id();
    const int qkv_dim_offset = local_id * elems_per_item;

    const int q_dim = head_k_dim;
    const int k_dim = head_k_dim;
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int z_dim = head_v_dim * num_v_heads / num_k_heads;
    const int qkv_dim = q_dim + k_dim + v_dim;
    const int qkvz_dim = q_dim + k_dim + v_dim + z_dim;

    int qkvz_elems_offset = k_head_id * qkvz_dim + qkv_dim_offset;

    // get current seq start, end
    int batch_id = batch_size - 1;
    int seq_start_offset = 0;
    int seq_end_offset = 0;
    int out_token_id = token_id;
    int pre_chunks = 0;
    for (int i = 0; i < batch_size; ++i) {
      int current_seq_start = query_start_loc[i];
      int current_seq_end = query_start_loc[i + 1];
      int current_seq_len = current_seq_end - current_seq_start;
      if (token_id < current_seq_end) {
        batch_id = i;
        seq_start_offset = current_seq_start;
        seq_end_offset = current_seq_end;
        out_token_id = token_id - seq_start_offset + pre_chunks * chunk_size;
        break;
      }
      pre_chunks += (current_seq_len + chunk_size - 1) / chunk_size;
    }

    // get states cache location
    int states_id = cache_indices[batch_id];

    if (states_id == pad_slot_id) {
      return;
    }

    bool is_q = false;
    bool is_k = false;
    bool is_v = false;

    if (qkv_dim_offset < q_dim) {
      is_q = true;
      if constexpr (ReorderInput) {
        qkvz_elems_offset = k_head_id * k_dim + qkv_dim_offset;
      }
    } else if (qkv_dim_offset < q_dim + k_dim) {
      is_k = true;
      if constexpr (ReorderInput) {
        qkvz_elems_offset = num_k_heads * head_k_dim + k_head_id * k_dim +
                            qkv_dim_offset - (q_dim);
      }
    } else {
      is_v = true;
      if constexpr (ReorderInput) {
        qkvz_elems_offset = 2 * num_k_heads * head_k_dim + k_head_id * v_dim +
                            qkv_dim_offset - (q_dim + k_dim);
      }
    }

    // reorder index to map weights
    int reordered_elems_offset = 0;
    if (is_q) {
      reordered_elems_offset = k_head_id * q_dim + qkv_dim_offset;
    } else if (is_k) {
      reordered_elems_offset =
          num_k_heads * q_dim + k_head_id * k_dim + qkv_dim_offset - q_dim;
    } else if (is_v) {
      reordered_elems_offset = num_k_heads * (q_dim + k_dim) +
                               k_head_id * v_dim + qkv_dim_offset -
                               (q_dim + k_dim);
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
            conv_weights[(reordered_elems_offset + e) * Width + i];
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
               reordered_elems_offset + e];
        }
      }
    }

#pragma unroll
    for (int i = 0; i < input_load_len; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + states_load_len + i] = mixed_qkvz
            [(token_id - input_load_len + 1 + i) * qkvz_elems +
             qkvz_elems_offset + e];
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
        res[e] += conv_bias[reordered_elems_offset + e];
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
                 reordered_elems_offset + e] = local_input[Width * e + i + 1];
          }
        }
      }
    } else {
// update states inplace if decode
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          conv_states_ptr[i * conv_elems + reordered_elems_offset + e] =
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
            [out_token_id * num_k_heads * q_dim + k_head_id * q_dim +
             qkv_dim_offset + e] = res[e];
      }
    } else if (is_k) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        k_out
            [out_token_id * num_k_heads * k_dim + k_head_id * k_dim +
             qkv_dim_offset - q_dim + e] = res[e];
      }
    } else if (is_v) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        v_out
            [out_token_id * num_k_heads * v_dim + k_head_id * v_dim +
             qkv_dim_offset - (q_dim + k_dim) + e] = res[e];
      }
    }
  }

 private:
  T* q_out;
  T* k_out;
  T* v_out;
  T* z_out;
  float* b_out;
  float* a_out;
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
  const int num_virtual_tokens;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
  const int qkvz_elems;
  const int conv_elems;
};

template <typename T, int Width, bool ReorderInput>
struct chunk_causal_conv1d_opt_kernel {
 public:
  // Optimized kernel for ReorderInput=true only.
  //
  // High-level mapping:
  //   - group(0): one chunk-program (one sequence chunk).
  //   - group(1): one feature tile (block_n contiguous conv channels).
  //   - local_id: cooperatively loads/computes/stores within the tile.
  //
  // High-level dataflow:
  //   1) Stage [state_len + segment_len, block_n] window into SLM.
  //   2) Do depthwise causal conv on staged window for this feature tile.
  //   3) Write q/k/v to virtual-token-major output layout.
  //   4) Update conv_states once per sequence (current policy: first chunk).
  static constexpr int sub_group_size = 32;
  static constexpr int sg_num = 4;
  static constexpr int wg_size = sub_group_size * sg_num;
  static constexpr int vec_size = 4;
  static constexpr int block_m = 8;
  static constexpr int block_n = 128;
  static constexpr int state_len = Width - 1;
  static constexpr int vec_channels = block_n / vec_size;
  static constexpr int smem_elems = (state_len + block_m) * vec_channels;
  using vec_t = vllm::xpu::aligned_vec<T, vec_size>;
  using vecf_t = vllm::xpu::aligned_vec<float, vec_size>;
  using smem_elem_t = vec_t;

  chunk_causal_conv1d_opt_kernel(
      T* q_out,
      T* k_out,
      T* v_out,
      T* z_out,
      float* b_out,
      float* a_out,
      const T* mixed_qkvz,
      const T* mixed_ba,
      const T* conv_weights,
      const T* conv_bias,
      T* conv_states,
      const int conv_states_stride_0,
      T* conv_states_tmp,
      const int32_t* batch_ptr,
      const int32_t* token_chunk_offset_ptr,
      const int32_t* out_token_base_ptr,
      int* query_start_loc,
      int* cache_indices,
      bool* has_initial_state,
      const ActMode& act_mode,
      const int& pad_slot_id,
      const int& batch_size,
      const int& num_actual_tokens,
      const int& num_virtual_tokens,
      const int& num_k_heads,
      const int& head_k_dim,
      const int& num_v_heads,
      const int& head_v_dim,
      const int& qkvz_elems,
      const int& conv_elems,
      sycl::local_accessor<vec_t, 1> smem_x)
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
        batch_ptr(batch_ptr),
        token_chunk_offset_ptr(token_chunk_offset_ptr),
        out_token_base_ptr(out_token_base_ptr),
        query_start_loc(query_start_loc),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
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
        conv_elems(conv_elems),
        smem_x(smem_x) {
    static_assert(ReorderInput, "chunk_causal_conv1d_opt_kernel only supports ReorderInput=true");
    static_assert(vec_size == 1 || vec_size == 2 || vec_size == 4 || vec_size == 8,
                  "vec_size must be one of {1,2,4,8}");
      static_assert(
        std::is_trivially_copyable_v<vec_t>,
        "aligned_vec must be trivially copyable");
      // static_assert(block_n == wg_size, "block_n must match wg_size");
      static_assert(
        block_n % vec_size == 0,
        "block_n must be divisible by vec_size");
  }

  // mixed_qkvz layout used by this kernel (ReorderInput=true only), per token:
  //   [all Q heads][all K heads][all V heads][all Z heads]
  // Conv path in this kernel consumes only Q/K/V part (conv_num_feats).

  static inline sycl::nd_range<2> get_nd_range(
      const int total_seqlen_chunks,
      const int num_k_heads,
      const int head_k_dim,
      const int num_v_heads,
      const int head_v_dim) {
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int conv_feats_per_head = 2 * head_k_dim + v_dim;
    const int conv_num_feats = conv_feats_per_head * num_k_heads;
    const int feat_groups =
        (conv_num_feats + block_n - 1) / block_n;

    // Each work-group computes one [block_m, block_n] output tile:
    //   - block_m: chunk axis (token axis inside one sequence chunk)
    //   - block_n: conv-channel axis (feature tile)
    sycl::range<2> local(1, wg_size);
    sycl::range<2> global(total_seqlen_chunks, feat_groups);
    return sycl::nd_range<2>(global * local, local);
  }

  static inline void act_swish(float& x, float beta = 1.0f) {
    constexpr float log2e = 1.44269504089f;
    x = x / (1.0f + sycl::native::exp2(-x * beta * log2e));
  }

  static inline void act_silu(float& x) { act_swish(x, 1.0f); }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    // token_chunk_id maps to one host-generated chunk program:
    //   token_chunk_id -> (batch_id, chunk_offset_in_sequence)
    const int token_chunk_id = item.get_group(0);
    const int local_id = item.get_local_linear_id();
    vec_t* smem_ptr =
        smem_x.template get_multi_ptr<sycl::access::decorated::no>().get();
    vec_t* smem_in = smem_ptr;

    const int q_dim = head_k_dim;
    const int k_dim = head_k_dim;
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int conv_feats_per_head = q_dim + k_dim + v_dim;
    const int conv_num_feats = conv_feats_per_head * num_k_heads;

    const int feat_group = item.get_group(1);

    // batch_id for this token chunk
    const int batch_id = batch_ptr[token_chunk_id];
    if (batch_id < 0 || batch_id >= batch_size) {
      return;
    }

    // chunk_offset is INTRA-SEQUENCE chunk id (0..num_chunks-1),
    // not a global chunk id across all sequences.
    const int chunk_offset = token_chunk_offset_ptr[token_chunk_id];
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seqlen = seq_end_offset - seq_start_offset;
    if (seqlen <= 0) {
      return;
    }

    const int states_id = cache_indices[batch_id];
    if (states_id == pad_slot_id) {
      return;
    }

    const bool has_init_conv_states =
        (has_initial_state == nullptr ||
         (has_initial_state != nullptr && has_initial_state[batch_id]));
    T* conv_states_ptr = conv_states + states_id * conv_states_stride_0;

    const int token_offset = block_m * chunk_offset;
    // segment_len can be smaller than block_m for the tail chunk.
    const int segment_len = sycl::max(0, sycl::min(block_m, seqlen - token_offset));
    if (segment_len <= 0) {
      return;
    }

    // Stage one conv window into SLM for this feature tile:
    //   window = [history(state_len) + current_chunk(segment_len)]
    //   shape  = [window_len, vec_channels] where vec_channels=block_n/vec_size
    // Source token mapping:
    //   src_token < 0  -> read from conv_states (if has initial state)
    //   src_token >= 0 -> read from mixed_qkvz at sequence-local position
    const int window_len = state_len + segment_len;
    const int vec_stride = vec_channels;
    const int load_tasks = window_len * vec_channels;
    for (int task_id = local_id; task_id < load_tasks; task_id += wg_size) {
      const int pos = task_id / vec_channels;
      const int vec_idx = task_id % vec_channels;
      const int feat_base = feat_group * block_n + vec_idx * vec_size;
      const int src_token = token_offset - state_len + pos;

      vec_t src_vec{};
      if (src_token >= 0 && src_token < seqlen) {
        const T* src_ptr =
            mixed_qkvz + (seq_start_offset + src_token) * qkvz_elems + feat_base;
        src_vec = *reinterpret_cast<const vec_t*>(src_ptr);
      } else if (has_init_conv_states) {
        const int state_pos = state_len + src_token;
        if (state_pos >= 0 && state_pos < state_len) {
          const T* state_ptr = conv_states_ptr + state_pos * conv_elems + feat_base;
          src_vec = *reinterpret_cast<const vec_t*>(state_ptr);
        }
      }
      *(smem_in + pos * vec_stride + vec_idx) = src_vec;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Parallelism over token axis inside a chunk tile:
    // each lane-group handles t = m_lane, m_lane + m_parallel, ...
    const int m_parallel = wg_size / vec_channels;
    const int vec_idx = local_id % vec_channels;
    const int m_lane = local_id / vec_channels;
    const int feat_base = feat_group * block_n + vec_idx * vec_size;

    // Load bias
    vec_t bias_vec{};
    if (conv_bias != nullptr) {
      bias_vec = *reinterpret_cast<const vec_t*>(conv_bias + feat_base);
    }

    // Load weights for channels of this work-item into registers.
    vec_t w_reg[Width];
#pragma unroll
    for (int k = 0; k < Width; ++k) {
#pragma unroll
      for (int lane = 0; lane < vec_size; ++lane) {
        const int c = feat_base + lane;
        w_reg[k][lane] = conv_weights[c * Width + k];
      }
    }

    for (int t = m_lane; t < segment_len; t += m_parallel) {
      // acc computes one output channel vector for token t.
      vecf_t acc;
#pragma unroll
      for (int lane = 0; lane < vec_size; ++lane) {
        acc[lane] = vllm::xpu::to_float(bias_vec[lane]);
      }

#pragma unroll
      for (int k = 0; k < Width; ++k) {
        // Read staged input x[t + k, c] from SLM.
        vec_t xv_vec = *(smem_in + (t + k) * vec_stride + vec_idx);
#pragma unroll
        for (int lane = 0; lane < vec_size; ++lane) {
          acc[lane] += vllm::xpu::to_float(xv_vec[lane]) * 
                       vllm::xpu::to_float(w_reg[k][lane]);
        }
      }
      vec_t out_vec{};
#pragma unroll
      for (int lane = 0; lane < vec_size; ++lane) {
        if (act_mode == ActMode::silu) {
          act_silu(acc[lane]);
        } else if (act_mode == ActMode::swish) {
          act_swish(acc[lane]);
        }
        vllm::xpu::from_float(out_vec[lane], acc[lane]);
      }

      // Store q/k/v into virtual-token-major layout directly.
      // Mapping is decoupled from program scheduling granularity:
      //   - scheduling uses block_m-sized micro chunks,
      //   - logical output layout remains chunk_size(64)-based.
      // Host precomputes per-program out_token_base in logical layout.
      const int out_token_base = out_token_base_ptr[token_chunk_id];
      const int out_token_id = out_token_base + t;
      if (out_token_id >= num_virtual_tokens) {
        continue;
      }

      // Channel partition inside conv feature vector:
      //   [0, q_total)                  -> Q
      //   [q_total, q_total + k_total)  -> K
      //   [q_total + k_total, end)      -> V
      const int q_total = num_k_heads * q_dim;
      const int k_total = num_k_heads * k_dim;
      const int v_total = num_k_heads * v_dim;

      const int feat_end_store = feat_base + vec_size;
      if (feat_end_store <= q_total) {
        *reinterpret_cast<vec_t*>(q_out + out_token_id * q_total + feat_base) =
            out_vec;
      } else if (feat_base >= q_total && feat_end_store <= q_total + k_total) {
        *reinterpret_cast<vec_t*>(
            k_out + out_token_id * k_total + (feat_base - q_total)) = out_vec;
      } else {
        *reinterpret_cast<vec_t*>(
            v_out + out_token_id * v_total + (feat_base - q_total - k_total)) =
            out_vec;
      }
    }

    // conv_states update policy (current implementation):
    //   - only chunk_offset == 0 performs update (single writer per sequence),
    //   - updated state is built from sequence tail tokens (or previous state
    //     when sequence shorter than state_len).
    if (chunk_offset == 0 && state_len > 0) {
      if (local_id < vec_channels) {
        const int feat_base_state = feat_group * block_n + local_id * vec_size;
#pragma unroll
        for (int s = 0; s < state_len; ++s) {
          const int src_token = seqlen - state_len + s;
          vec_t new_state_vec{};
          if (src_token >= 0) {
            const T* src_ptr =
                mixed_qkvz + (seq_start_offset + src_token) * qkvz_elems +
                feat_base_state;
            new_state_vec = *reinterpret_cast<const vec_t*>(src_ptr);
          } else if (has_init_conv_states) {
            const int prev_pos = state_len + src_token;
            if (prev_pos >= 0 && prev_pos < state_len) {
              const T* prev_ptr = conv_states_ptr + prev_pos * conv_elems +
                                  feat_base_state;
              new_state_vec = *reinterpret_cast<const vec_t*>(prev_ptr);
            }
          }
          *reinterpret_cast<vec_t*>(
              conv_states_ptr + s * conv_elems + feat_base_state) =
              new_state_vec;
        }
      }
    }
  }

 private:
  T* q_out;
  T* k_out;
  T* v_out;
  T* z_out;
  float* b_out;
  float* a_out;
  const T* mixed_qkvz;
  const T* mixed_ba;
  const T* conv_weights;
  const T* conv_bias;
  T* conv_states;
  const int conv_states_stride_0;
  T* conv_states_tmp;
  const int32_t* batch_ptr;
  const int32_t* token_chunk_offset_ptr;
  const int32_t* out_token_base_ptr;
  const int32_t* query_start_loc;
  const int* cache_indices;
  const bool* has_initial_state;
  const ActMode act_mode;
  const int pad_slot_id;
  const int batch_size;
  const int num_actual_tokens;
  const int num_virtual_tokens;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
  const int qkvz_elems;
  const int conv_elems;
  sycl::local_accessor<vec_t, 1> smem_x;
};

template <typename T, int Width>
static inline bool can_use_chunk_causal_conv1d_opt(
    const T* q_out,
    const T* k_out,
    const T* v_out,
    const T* mixed_qkvz,
    const T* conv_weights,
    const T* conv_bias,
    const T* conv_states,
    const int conv_states_stride_0,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim,
    const int qkvz_elems,
    const int conv_elems) {
  using kernel_t = chunk_causal_conv1d_opt_kernel<T, Width, true>;
  using aligned_vec_t = typename kernel_t::vec_t;

  const int v_dim = head_v_dim * num_v_heads / num_k_heads;
  const int expected_conv_elems = num_k_heads * (2 * head_k_dim + v_dim);
  const int q_total = num_k_heads * head_k_dim;
  const int k_total = num_k_heads * head_k_dim;
  const int v_total = num_v_heads * head_v_dim;

  return num_v_heads % num_k_heads == 0 &&
         conv_elems == expected_conv_elems &&
         conv_elems % kernel_t::block_n == 0 &&
         q_total % kernel_t::vec_size == 0 &&
         k_total % kernel_t::vec_size == 0 &&
         v_total % kernel_t::vec_size == 0 &&
         qkvz_elems % kernel_t::vec_size == 0 &&
         conv_states_stride_0 % kernel_t::vec_size == 0 &&
         is_aligned_vec_ptr<aligned_vec_t>(q_out) &&
         is_aligned_vec_ptr<aligned_vec_t>(k_out) &&
         is_aligned_vec_ptr<aligned_vec_t>(v_out) &&
         is_aligned_vec_ptr<aligned_vec_t>(mixed_qkvz) &&
         is_aligned_vec_ptr<aligned_vec_t>(conv_weights) &&
         is_aligned_vec_ptr<aligned_vec_t>(conv_states) &&
         (conv_bias == nullptr || is_aligned_vec_ptr<aligned_vec_t>(conv_bias));
}

template <typename T, bool ReorderInput>
struct chunk_reorder_zba_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int load_size = 4;

  static inline float act_sigmoid(float& x) {
    return 1.0f / (1.0f + sycl::exp(-x));
  }

  chunk_reorder_zba_kernel(
      T* z_out,
      float* b_out,
      float* a_out,
      const T* mixed_qkvz,
      const T* mixed_ba,
      const int32_t* query_start_loc,
      const int batch_size,
      const int& num_virtual_tokens,
      const int& num_k_heads,
      const int& head_k_dim,
      const int& num_v_heads,
      const int& head_v_dim)
      : z_out(z_out),
        b_out(b_out),
        a_out(a_out),
        mixed_qkvz(mixed_qkvz),
        mixed_ba(mixed_ba),
        query_start_loc(query_start_loc),
        batch_size(batch_size),
        num_virtual_tokens(num_virtual_tokens),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim) {}

  static inline sycl::nd_range<2>
  get_nd_range(const int total_seqlen, const int num_k_heads, const int z_dim) {
    assert(z_dim % load_size == 0);
    sycl::range<2> local(1, z_dim / load_size);
    sycl::range<2> global(total_seqlen, num_k_heads);
    return sycl::nd_range<2>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int token_id = item.get_group(0);
    const int k_head_id = item.get_group(1);
    const int dim_offset = item.get_local_id(1);

    const int q_dim = head_k_dim;
    const int k_dim = head_k_dim;
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int z_dim = head_v_dim * num_v_heads / num_k_heads;
    const int qkv_dim = q_dim + k_dim + v_dim;
    const int qkvz_dim = q_dim + k_dim + v_dim + z_dim;

    const int kv_ratio = num_v_heads / num_k_heads;

    // reorder b,a
    if (dim_offset == 0) {
      int pre_chunks = 0;
      int out_token_id = token_id;
      for (int i = 0; i < batch_size; ++i) {
        int current_seq_start = query_start_loc[i];
        int current_seq_end = query_start_loc[i + 1];
        int current_seq_len = current_seq_end - current_seq_start;
        if (token_id < current_seq_end) {
          out_token_id = token_id - current_seq_start + pre_chunks * chunk_size;
          break;
        }
        pre_chunks += (current_seq_len + chunk_size - 1) / chunk_size;
      }

      if constexpr (ReorderInput) {
        int step = token_id * num_v_heads * 2;
#pragma unroll
        for (int e = 0; e < kv_ratio; ++e) {
          float b_value = mixed_ba[step + k_head_id * kv_ratio + e];
          float a_value =
              mixed_ba[step + num_v_heads + k_head_id * kv_ratio + e];
          b_value = act_sigmoid(b_value);
          b_out
              [(k_head_id * kv_ratio + e) * num_virtual_tokens + out_token_id] =
                  b_value;
          a_out
              [(k_head_id * kv_ratio + e) * num_virtual_tokens + out_token_id] =
                  a_value;
        }
      } else {
        int step =
            (token_id * num_v_heads + k_head_id * num_v_heads / num_k_heads) *
            2;
#pragma unroll
        for (int e = 0; e < kv_ratio; ++e) {
          float b_value = mixed_ba[step + e];
          float a_value = mixed_ba[step + kv_ratio + e];
          b_value = act_sigmoid(b_value);
          b_out
              [(k_head_id * kv_ratio + e) * num_virtual_tokens + out_token_id] =
                  b_value;
          a_out
              [(k_head_id * kv_ratio + e) * num_virtual_tokens + out_token_id] =
                  a_value;
        }
      }
    }

    // reorder z
#pragma unroll
    for (int e = 0; e < load_size; ++e) {
      int z_dim_id = dim_offset * load_size + e;
      int mixed_z_id = token_id * num_k_heads * qkvz_dim +
                       k_head_id * qkvz_dim + qkv_dim + z_dim_id;
      if constexpr (ReorderInput) {
        mixed_z_id = token_id * num_k_heads * qkvz_dim +
                     2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim +
                     k_head_id * z_dim + z_dim_id;
      }
      z_out[token_id * num_k_heads * z_dim + k_head_id * z_dim + z_dim_id] =
          mixed_qkvz[mixed_z_id];
    }
  }

 private:
  T* z_out;
  float* b_out;
  float* a_out;
  const T* mixed_qkvz;
  const T* mixed_ba;
  const int32_t* query_start_loc;
  const int batch_size;
  const int num_virtual_tokens;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
};

template <typename T>
struct chunk_update_states_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = 4;
  static constexpr int elems_per_group = group_size * elems_per_item;

  chunk_update_states_kernel(
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


struct build_program_meta_device_kernel {
 public:
  build_program_meta_device_kernel(
      const int32_t* query_start_loc,
      int32_t* batch_ptr,
      int32_t* token_chunk_offset_ptr,
      int32_t* out_token_base_ptr,
      int32_t pad_slot_id,
      int block_m,
      int batch_size,
      int num_program_slots)
      : query_start_loc(query_start_loc),
        batch_ptr(batch_ptr),
        token_chunk_offset_ptr(token_chunk_offset_ptr),
        out_token_base_ptr(out_token_base_ptr),
        pad_slot_id(pad_slot_id),
        block_m(block_m),
        batch_size(batch_size),
        num_program_slots(num_program_slots) {}

  void operator()(sycl::id<1> id) const {
    const int32_t program_id = static_cast<int32_t>(id[0]);
    if (program_id >= num_program_slots) {
      return;
    }

    int32_t pre_micro_chunks = 0;
    int32_t pre_chunks_64 = 0;
    bool found = false;

    for (int32_t seq = 0; seq < batch_size; ++seq) {
      const int32_t seqlen = query_start_loc[seq + 1] - query_start_loc[seq];
      const int32_t n_micro_chunks = (seqlen + block_m - 1) / block_m;
      const int32_t cumsum_micro = pre_micro_chunks + n_micro_chunks;

      if (program_id < cumsum_micro) {
        const int32_t chunk_offset = program_id - pre_micro_chunks;
        batch_ptr[program_id] = seq;
        token_chunk_offset_ptr[program_id] = chunk_offset;
        out_token_base_ptr[program_id] =
            pre_chunks_64 * chunk_size + chunk_offset * block_m;
        found = true;
        break;
      }

      pre_micro_chunks = cumsum_micro;
      pre_chunks_64 += (seqlen + chunk_size - 1) / chunk_size;
    }

    if (!found) {
      batch_ptr[program_id] = pad_slot_id;
      token_chunk_offset_ptr[program_id] = 0;
      out_token_base_ptr[program_id] = 0;
    }
  }

 private:
  const int32_t* query_start_loc;
  int32_t* batch_ptr;
  int32_t* token_chunk_offset_ptr;
  int32_t* out_token_base_ptr;
  const int32_t pad_slot_id;
  const int block_m;
  const int batch_size;
  const int num_program_slots;
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
build_program_meta_device(
    sycl::queue& queue,
    const torch::Tensor& query_start_loc,
    int32_t pad_slot_id,
    int block_m,
    int num_program_slots) {
  auto opts = torch::TensorOptions()
                  .dtype(torch::kInt32)
                  .device(query_start_loc.device())
                  .requires_grad(false);
  auto batch_ptr = torch::empty({num_program_slots}, opts);
  auto token_chunk_offset_ptr = torch::empty({num_program_slots}, opts);
  auto out_token_base_ptr = torch::empty({num_program_slots}, opts);

  const int batch_size = query_start_loc.size(0) - 1;
  queue.submit([&](sycl::handler& cgh) {
    build_program_meta_device_kernel task(
        reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
        reinterpret_cast<int32_t*>(batch_ptr.data_ptr()),
        reinterpret_cast<int32_t*>(token_chunk_offset_ptr.data_ptr()),
        reinterpret_cast<int32_t*>(out_token_base_ptr.data_ptr()),
        pad_slot_id,
        block_m,
        batch_size,
        num_program_slots);
    cgh.parallel_for(sycl::range<1>(num_program_slots), task);
  });

  return {batch_ptr, token_chunk_offset_ptr, out_token_base_ptr};
}

template <typename T, int Width, bool ReorderInput>
void kernel_launcher(
    sycl::queue& queue,
    T* q_out,
    T* k_out,
    T* v_out,
    T* z_out,
    float* b_out,
    float* a_out,
    const T* mixed_qkvz,
    const T* mixed_ba,
    const T* conv_weights,
    const T* conv_bias,
    T* conv_states,
    const int conv_states_stride_0,
    T* conv_states_tmp,
    const torch::Tensor& query_start_loc_tensor,
    int* query_start_loc,
    int* cache_indices,
    bool* has_initial_state,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int& batch_size,
    const int& num_actual_tokens,
    const int& num_virtual_tokens,
    const int& num_k_heads,
    const int& head_k_dim,
    const int& num_v_heads,
    const int& head_v_dim,
    const int& qkvz_elems,
    const int& conv_elems,
    const int& num_prefills,
    const int& num_decodes) {
  bool use_opt_kernel = false;
  if constexpr (ReorderInput) {
    use_opt_kernel = can_use_chunk_causal_conv1d_opt<T, Width>(
        q_out,
        k_out,
        v_out,
        mixed_qkvz,
        conv_weights,
        conv_bias,
        conv_states,
        conv_states_stride_0,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        qkvz_elems,
        conv_elems);

    if (use_opt_kernel) {
      using KERNEL_MAIN = chunk_causal_conv1d_opt_kernel<T, Width, ReorderInput>;

      // Use a approximate number of micro-chunks to avoid D2H synchronization
      const int approx_total_micro_chunks =
          (num_actual_tokens + batch_size * (KERNEL_MAIN::block_m - 1) +
           KERNEL_MAIN::block_m - 1) /
          KERNEL_MAIN::block_m;

      auto [batch_tensor, token_chunk_offset_tensor, out_token_base_tensor] =
          build_program_meta_device(
              queue,
              query_start_loc_tensor,
              static_cast<int32_t>(pad_slot_id),
              KERNEL_MAIN::block_m,
              approx_total_micro_chunks);

      const int32_t* batch_ptr =
          reinterpret_cast<const int32_t*>(batch_tensor.data_ptr());
      const int32_t* token_chunk_offset_ptr =
          reinterpret_cast<const int32_t*>(token_chunk_offset_tensor.data_ptr());
      const int32_t* out_token_base_ptr =
          reinterpret_cast<const int32_t*>(out_token_base_tensor.data_ptr());

      auto range_main = KERNEL_MAIN::get_nd_range(
          approx_total_micro_chunks,
          num_k_heads,
          head_k_dim,
          num_v_heads,
          head_v_dim);
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<typename KERNEL_MAIN::smem_elem_t, 1> smem_x(
            sycl::range<1>(KERNEL_MAIN::smem_elems), cgh);
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
            batch_ptr,
            token_chunk_offset_ptr,
            out_token_base_ptr,
            query_start_loc,
            cache_indices,
            has_initial_state,
            act_mode,
            pad_slot_id,
            batch_size,
            num_actual_tokens,
            num_virtual_tokens,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            qkvz_elems,
            conv_elems,
            smem_x);
        cgh.parallel_for(range_main, task);
      });
    }
  }

  if (!use_opt_kernel) {
    using KERNEL_MAIN = chunk_causal_conv1d_kernel<T, Width, ReorderInput>;
    auto range_main = KERNEL_MAIN::get_nd_range(
        num_actual_tokens, num_k_heads, head_k_dim, num_v_heads, head_v_dim);
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
          num_virtual_tokens,
          num_k_heads,
          head_k_dim,
          num_v_heads,
          head_v_dim,
          qkvz_elems,
          conv_elems);
      cgh.parallel_for(range_main, task);
    });

    if (num_prefills > 0) {
      using KERNEL_UPDATE = chunk_update_states_kernel<T>;
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
  }

  using KERNEL_ZBA = chunk_reorder_zba_kernel<T, ReorderInput>;
  const int z_dim = head_v_dim * num_v_heads / num_k_heads;
  auto range_zba =
      KERNEL_ZBA::get_nd_range(num_actual_tokens, num_k_heads, z_dim);
  assert(
      (head_v_dim * num_v_heads / num_k_heads) %
          (KERNEL_ZBA::group_size / num_k_heads) ==
      0);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL_ZBA task(
        z_out,
        b_out,
        a_out,
        mixed_qkvz,
        mixed_ba,
        query_start_loc,
        batch_size,
        num_virtual_tokens,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim);
    cgh.parallel_for(range_zba, task);
  });

}

void chunk_causal_conv1d_xe2(
    sycl::queue& queue,
    torch::Tensor& q_out,  // [total_seqlen, num_k_heads, head_k_dim]
    torch::Tensor& k_out,  // [total_seqlen, num_k_heads, head_k_dim]
    torch::Tensor& v_out,  // [total_seqlen, num_v_heads, head_v_dim]
    torch::Tensor& z_out,  // [total_seqlen, num_v_heads, head_v_dim]
    torch::Tensor& b_out,  // [num_v_heads, total_seqlen]
    torch::Tensor& a_out,  // [num_v_heads, total_seqlen]
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
    const bool reorder_input) {
  if (num_prefills == 0 && num_decodes == 0) {
    return;
  }

  const int batch_size = query_start_loc.size(0) - 1;
  const int num_actual_tokens = mixed_qkvz.size(0);
  const int num_virtual_tokens = q_out.size(0);
  const int num_k_heads = q_out.size(1);
  const int head_k_dim = q_out.size(2);
  const int num_v_heads = v_out.size(1);
  const int head_v_dim = v_out.size(2);
  const int qkvz_elems = mixed_qkvz.size(1);
  const int conv_elems = conv_weights.size(0);
  const int width = conv_weights.size(1);
  const int conv_states_stride_0 = conv_states.stride(0);

  auto dtype = conv_states.dtype();
  auto device = conv_states.device();
  torch::Tensor conv_states_tmp = torch::empty(
      {batch_size, width - 1, conv_elems},
      torch::dtype(dtype).device(device).requires_grad(false));

#define KERNEL_LAUNCHER(scalar_t, width, reorder_input)            \
  kernel_launcher<scalar_t, width, reorder_input>(                 \
      queue,                                                       \
      reinterpret_cast<scalar_t*>(q_out.data_ptr()),               \
      reinterpret_cast<scalar_t*>(k_out.data_ptr()),               \
      reinterpret_cast<scalar_t*>(v_out.data_ptr()),               \
      reinterpret_cast<scalar_t*>(z_out.data_ptr()),               \
      reinterpret_cast<float*>(b_out.data_ptr()),                  \
      reinterpret_cast<float*>(a_out.data_ptr()),                  \
      reinterpret_cast<scalar_t*>(mixed_qkvz.data_ptr()),          \
      reinterpret_cast<scalar_t*>(mixed_ba.data_ptr()),            \
      reinterpret_cast<scalar_t*>(conv_weights.data_ptr()),        \
      conv_bias.has_value()                                        \
          ? reinterpret_cast<scalar_t*>(conv_bias->data_ptr())     \
          : nullptr,                                               \
      reinterpret_cast<scalar_t*>(conv_states.data_ptr()),         \
      conv_states_stride_0,                                        \
      reinterpret_cast<scalar_t*>(conv_states_tmp.data_ptr()),     \
      query_start_loc,                                             \
      reinterpret_cast<int*>(query_start_loc.data_ptr()),          \
      reinterpret_cast<int*>(cache_indices.data_ptr()),            \
      has_initial_state.has_value()                                \
          ? reinterpret_cast<bool*>(has_initial_state->data_ptr()) \
          : nullptr,                                               \
      act_mode,                                                    \
      pad_slot_id,                                                 \
      batch_size,                                                  \
      num_actual_tokens,                                           \
      num_virtual_tokens,                                          \
      num_k_heads,                                                 \
      head_k_dim,                                                  \
      num_v_heads,                                                 \
      head_v_dim,                                                  \
      qkvz_elems,                                                  \
      conv_elems,                                                  \
      num_prefills,                                                \
      num_decodes);

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