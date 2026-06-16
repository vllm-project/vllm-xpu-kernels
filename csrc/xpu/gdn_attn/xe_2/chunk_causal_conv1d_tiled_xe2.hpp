#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "gdn_attn_utils.h"

namespace gdn {

static constexpr int conv1d_tile_size = 8;

// SLM-tiled conv1d kernel for XE2 prefill path.
//
// Tiles TileT=8 consecutive tokens per workgroup with cooperative SLM loading.
// The qkv feature dimension is split across num_feat_chunks workgroups, each
// handling feats_per_wg = wg_size * elems_per_item = 256 features.
//
// Grid: (num_tiles * num_feat_chunks, num_k_heads), Local: (1, wg_size=64)
// SLM: meta_size + (TileT + Width - 1) * feats_per_wg elements
//
// Phase 0: Item 0 computes tile metadata (batch_id, offsets) → SLM, barrier
// Phase 1: All items cooperatively load (TileT+Width-1) slots into SLM
//           with L2 prefetch 2 slots ahead
// Phase 2: Each item computes conv1d for its 4 feature lanes from SLM
// Phase 3: Last tile writes conv_state

template <typename T, int Width, int TileT, bool ReorderInput>
struct chunk_causal_conv1d_tiled_kernel {
 public:
  static constexpr int sub_group_size = 16;
  static constexpr int elems_per_item = 4;
  static constexpr int wg_size = 64;  // 4 subgroups per WG

  chunk_causal_conv1d_tiled_kernel(
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
      const int* token_indx,
      const ActMode& act_mode,
      const int& pad_slot_id,
      const int& batch_size,
      const int& num_k_heads,
      const int& head_k_dim,
      const int& num_v_heads,
      const int& head_v_dim,
      const int& qkvz_elems,
      const int& conv_elems,
      const int& num_virtual_tokens,
      char* slm_data,
      const bool fuse_l2norm)
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
        token_indx(token_indx),
        act_mode(act_mode),
        pad_slot_id(pad_slot_id),
        batch_size(batch_size),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim),
        qkvz_elems(qkvz_elems),
        conv_elems(conv_elems),
        num_virtual_tokens(num_virtual_tokens),
        slm_data(slm_data),
        fuse_l2norm(fuse_l2norm) {}

  inline int lookup(int t) const { return token_indx ? token_indx[t] : t; }

  static inline int get_num_feat_chunks(
      const int head_k_dim,
      const int num_v_heads,
      const int num_k_heads,
      const int head_v_dim) {
    int qkv_dim = 2 * head_k_dim + head_v_dim * num_v_heads / num_k_heads;
    int feats_per_wg = wg_size * elems_per_item;
    return (qkv_dim + feats_per_wg - 1) / feats_per_wg;
  }

  static inline sycl::nd_range<2> get_nd_range(
      const int num_tiles,
      const int num_k_heads,
      const int head_k_dim,
      const int num_v_heads,
      const int head_v_dim) {
    int num_feat_chunks =
        get_num_feat_chunks(head_k_dim, num_v_heads, num_k_heads, head_v_dim);
    sycl::range<2> local(1, wg_size);
    sycl::range<2> global(num_tiles * num_feat_chunks, num_k_heads);
    return sycl::nd_range<2>(global * local, local);
  }

  static constexpr int meta_ints = 5;
  // SLM sizes in bytes: metadata (ints) + input data (T elements) + norm
  // (floats)
  static constexpr int slm_meta_bytes = meta_ints * sizeof(int);
  static constexpr int feats_per_wg = wg_size * elems_per_item;
  static constexpr int slm_data_elems = (TileT + Width - 1) * feats_per_wg;
  static constexpr int num_subgroups_per_wg = wg_size / sub_group_size;
  // 2 floats per subgroup: one for Q partial sum, one for K partial sum
  static constexpr int norm_slm_bytes =
      2 * num_subgroups_per_wg * static_cast<int>(sizeof(float));

  static inline int get_slm_bytes() {
    return slm_meta_bytes + slm_data_elems * sizeof(T) + norm_slm_bytes;
  }

  static inline void act_swish(float& x, float beta = 1.0f) {
    x = x / (1.0f + sycl::exp(-x * beta));
  }
  static inline void act_silu(float& x) { act_swish(x, 1.0f); }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int k_head_id = item.get_group(1);
    const int local_id = item.get_local_linear_id();

    const int q_dim = head_k_dim;
    const int k_dim = head_k_dim;
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int qkv_dim = q_dim + k_dim + v_dim;
    const int qkvz_dim =
        q_dim + k_dim + v_dim + head_v_dim * num_v_heads / num_k_heads;
    const int num_slots = TileT + Width - 1;
    const int feats_per_wg = wg_size * elems_per_item;
    const int num_feat_chunks = (qkv_dim + feats_per_wg - 1) / feats_per_wg;

    // Decode tile_id and feat_chunk_id from group(0)
    const int combined_id = item.get_group(0);
    const int tile_id = combined_id / num_feat_chunks;
    const int feat_chunk_id = combined_id % num_feat_chunks;
    const int feat_base = feat_chunk_id * feats_per_wg;

    // SLM layout: [meta_ints ints | slm_data_elems T elements]
    // slm_data is allocated as raw bytes, cast to appropriate types.
    int* slm_meta = reinterpret_cast<int*>(slm_data);
    T* slm_input = reinterpret_cast<T*>(
        reinterpret_cast<char*>(slm_data) + slm_meta_bytes);

    // Item 0 computes tile metadata once; others skip
    if (local_id == 0) {
      int batch_id_local = -1;
      int tile_start_local = 0;
      int seq_start_local = 0;
      int seq_end_local = 0;
      int pre_chunks_local = 0;
      int tiles_before = 0;
      for (int i = 0; i < batch_size; ++i) {
        int s_start = query_start_loc[i];
        int s_end = query_start_loc[i + 1];
        int seq_len_i = s_end - s_start;
        int tiles_in_seq = (seq_len_i + TileT - 1) / TileT;
        if (tile_id < tiles_before + tiles_in_seq) {
          batch_id_local = i;
          tile_start_local = (tile_id - tiles_before) * TileT;
          seq_start_local = s_start;
          seq_end_local = s_end;
          break;
        }
        pre_chunks_local += (seq_len_i + chunk_size_xe2 - 1) / chunk_size_xe2;
        tiles_before += tiles_in_seq;
      }
      slm_meta[0] = batch_id_local;
      slm_meta[1] = tile_start_local;
      slm_meta[2] = seq_start_local;
      slm_meta[3] = seq_end_local;
      slm_meta[4] = pre_chunks_local;
    }

    sycl::group_barrier(item.get_group());

    // All items read shared metadata from SLM
    int batch_id = slm_meta[0];
    int tile_start_in_seq = slm_meta[1];
    int seq_start = slm_meta[2];
    int seq_end = slm_meta[3];
    int pre_chunks = slm_meta[4];

    if (batch_id < 0) {
      return;
    }

    int states_id = cache_indices[batch_id];
    if (states_id == pad_slot_id) {
      return;
    }

    int seq_len = seq_end - seq_start;
    int tile_tokens = sycl::min(TileT, seq_len - tile_start_in_seq);

    const bool has_init_conv_states =
        (has_initial_state == nullptr ||
         (has_initial_state != nullptr && has_initial_state[batch_id]));
    T* conv_states_ptr = conv_states + states_id * conv_states_stride_0;

    // ========================================================================
    // Phase 1: COOPERATIVE load of (TileT + Width - 1) tokens into SLM
    // ========================================================================
    // This WG handles features [feat_base, feat_base + feats_per_wg).
    // SLM layout: slm_input[slot * feats_per_wg + local_feat]
    // Total SLM elements = num_slots * feats_per_wg = 11 * 256 = 2816
    // With 64 items × 4 elems = 256 per iter → 11 iterations (one per slot).

    // Compute per-item global feature offset (constant across slots).
    int local_feat = local_id * elems_per_item;
    int feat = feat_base + local_feat;

    // Guard: last chunk may have items beyond qkv_dim if not evenly divisible
    bool feat_valid = (feat < qkv_dim);

    int global_feat_offset = 0;
    int reordered_feat = 0;
    if (!feat_valid) {
      // Skip all computation for out-of-range items; still participate in
      // barriers
    } else if (feat < q_dim) {
      if constexpr (ReorderInput) {
        global_feat_offset = k_head_id * k_dim + feat;
      } else {
        global_feat_offset = k_head_id * qkvz_dim + feat;
      }
      reordered_feat = k_head_id * q_dim + feat;
    } else if (feat < q_dim + k_dim) {
      int feat_in_k = feat - q_dim;
      if constexpr (ReorderInput) {
        global_feat_offset =
            num_k_heads * head_k_dim + k_head_id * k_dim + feat_in_k;
      } else {
        global_feat_offset = k_head_id * qkvz_dim + feat;
      }
      reordered_feat = num_k_heads * q_dim + k_head_id * k_dim + feat_in_k;
    } else {
      int feat_in_v = feat - (q_dim + k_dim);
      if constexpr (ReorderInput) {
        global_feat_offset =
            2 * num_k_heads * head_k_dim + k_head_id * v_dim + feat_in_v;
      } else {
        global_feat_offset = k_head_id * qkvz_dim + feat;
      }
      reordered_feat =
          num_k_heads * (q_dim + k_dim) + k_head_id * v_dim + feat_in_v;
    }

    // Prefetch first 2 slots into L2 before entering the loop
    if (feat_valid)
      for (int pf = 0; pf < 2; ++pf) {
        int pf_token = tile_start_in_seq + pf - (Width - 1);
        if (pf_token >= 0 && pf_token < seq_len) {
          int pf_tok = lookup(seq_start + pf_token);
          auto pf_ptr = &mixed_qkvz[pf_tok * qkvz_elems + global_feat_offset];
          sycl::ext::oneapi::experimental::prefetch(
              pf_ptr, elems_per_item * sizeof(T));
        }
      }

    // Load loop: one iteration per slot, prefetch 2 slots ahead
    if (feat_valid)
      for (int slot = 0; slot < num_slots; ++slot) {
        int token_in_seq = tile_start_in_seq + slot - (Width - 1);

        // Prefetch slot+2 data to L2
        if (slot + 2 < num_slots) {
          int pf_token = tile_start_in_seq + (slot + 2) - (Width - 1);
          if (pf_token >= 0 && pf_token < seq_len) {
            int pf_tok = lookup(seq_start + pf_token);
            auto pf_ptr = &mixed_qkvz[pf_tok * qkvz_elems + global_feat_offset];
            sycl::ext::oneapi::experimental::prefetch(
                pf_ptr, elems_per_item * sizeof(T));
          }
        }

        T vals[elems_per_item];
        if (token_in_seq < 0) {
          int state_row = (Width - 1) + token_in_seq;
          if (has_init_conv_states && state_row >= 0 && state_row < Width - 1) {
#pragma unroll
            for (int e = 0; e < elems_per_item; ++e) {
              vals[e] =
                  conv_states_ptr[state_row * conv_elems + reordered_feat + e];
            }
          } else {
#pragma unroll
            for (int e = 0; e < elems_per_item; ++e) {
              vals[e] = static_cast<T>(0);
            }
          }
        } else if (token_in_seq < seq_len) {
          int global_tok = lookup(seq_start + token_in_seq);
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            vals[e] =
                mixed_qkvz[global_tok * qkvz_elems + global_feat_offset + e];
          }
        } else {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            vals[e] = static_cast<T>(0);
          }
        }

#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          slm_input[slot * feats_per_wg + local_feat + e] = vals[e];
        }
      }

    // Barrier: all items must finish writing SLM before any reads
    sycl::group_barrier(item.get_group());

    // ========================================================================
    // Phase 2: Each item computes conv1d for its own feature lanes
    // ========================================================================
    if (!feat_valid) return;

    bool is_q = (feat < q_dim);
    bool is_k = (!is_q && feat < q_dim + k_dim);

    // Load weights from global (only Width * elems_per_item = 16 values)
    T local_weights[Width * elems_per_item];
#pragma unroll
    for (int w = 0; w < Width; ++w) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_weights[w * elems_per_item + e] =
            conv_weights[(reordered_feat + e) * Width + w];
      }
    }

    float local_bias[elems_per_item];
    if (conv_bias != nullptr) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_bias[e] = conv_bias[reordered_feat + e];
      }
    }

    // Conv1d: for each output token t, read SLM slots [t, t+Width)
    for (int t = 0; t < tile_tokens; ++t) {
      float res[elems_per_item];
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        res[e] = 0.0f;
      }

#pragma unroll
      for (int w = 0; w < Width; ++w) {
        int slot = t + w;
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          res[e] += static_cast<float>(
                        slm_input[slot * feats_per_wg + local_feat + e]) *
                    static_cast<float>(local_weights[w * elems_per_item + e]);
        }
      }

      if (conv_bias != nullptr) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          res[e] += local_bias[e];
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

      // ---- Fused L2 norm for Q and K (only in feat_chunk 0) ----
      if (fuse_l2norm && feat_chunk_id == 0) {
        // l2norm_eps is defined in gdn_attn_utils.h
        float* norm_slm = reinterpret_cast<float*>(
            reinterpret_cast<char*>(slm_data) + slm_meta_bytes +
            slm_data_elems * sizeof(T));

        float q_local_sq = 0.0f;
        float k_local_sq = 0.0f;
        if (is_q) {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e)
            q_local_sq += res[e] * res[e];
        }
        if (is_k) {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e)
            k_local_sq += res[e] * res[e];
        }

        // Subgroup reduce
        auto sg = item.get_sub_group();
        float q_sg_sum =
            sycl::reduce_over_group(sg, q_local_sq, sycl::plus<float>());
        float k_sg_sum =
            sycl::reduce_over_group(sg, k_local_sq, sycl::plus<float>());

        // Write subgroup partial sums to SLM
        int sg_id = sg.get_group_linear_id();
        if (sg.get_local_linear_id() == 0) {
          norm_slm[sg_id * 2] = q_sg_sum;
          norm_slm[sg_id * 2 + 1] = k_sg_sum;
        }
        sycl::group_barrier(item.get_group());

        // Combine all subgroup partial sums
        float q_total = 0.0f;
        float k_total = 0.0f;
        for (int i = 0; i < num_subgroups_per_wg; ++i) {
          q_total += norm_slm[i * 2];
          k_total += norm_slm[i * 2 + 1];
        }
        sycl::group_barrier(item.get_group());  // protect SLM for next token

        float q_inv = sycl::rsqrt(q_total + l2norm_eps) *
                      sycl::rsqrt(static_cast<float>(q_dim));
        float k_inv = sycl::rsqrt(k_total + l2norm_eps);

        if (is_q) {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e)
            res[e] *= q_inv;
        }
        if (is_k) {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e)
            res[e] *= k_inv;
        }
      }

      // Write output
      int token_in_seq = tile_start_in_seq + t;
      int out_token_id = pre_chunks * chunk_size_xe2 + token_in_seq;

      if (is_q) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          q_out
              [out_token_id * num_k_heads * q_dim + k_head_id * q_dim + feat +
               e] = res[e];
        }
      } else if (is_k) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          k_out
              [out_token_id * num_k_heads * k_dim + k_head_id * k_dim + feat -
               q_dim + e] = res[e];
        }
      } else {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          v_out
              [out_token_id * num_k_heads * v_dim + k_head_id * v_dim + feat -
               (q_dim + k_dim) + e] = res[e];
        }
      }
    }

    // ========================================================================
    // Phase 2b: Fused z/b/a reorder (only first feat chunk does this)
    // ========================================================================
    if (feat_chunk_id == 0) {
      const int z_dim = head_v_dim * num_v_heads / num_k_heads;
      const int qkv_dim_full = q_dim + k_dim + v_dim;
      const int qkvz_dim_full = qkv_dim_full + z_dim;
      const int kv_ratio = num_v_heads / num_k_heads;

      for (int t = 0; t < tile_tokens; ++t) {
        int token_in_seq = tile_start_in_seq + t;
        int out_token_id = pre_chunks * chunk_size_xe2 + token_in_seq;
        int global_tok = lookup(seq_start + token_in_seq);

        // z reorder: each item handles elems_per_item z features per pass.
        // The 64 items cover feats_per_wg (256) features per pass, so when
        // z_dim exceeds 256 -- e.g. GQA ratio num_v_heads/num_k_heads == 3
        // gives z_dim = head_v_dim * 3 = 384 for Qwen3.6 -- a single pass would
        // drop the tail z features (the 3rd v-head of each k-group). Stride by
        // feats_per_wg so every z feature is written for any ratio.
        for (int z_dim_id = local_feat; z_dim_id < z_dim;
             z_dim_id += feats_per_wg) {
          int mixed_z_id;
          if constexpr (ReorderInput) {
            mixed_z_id = global_tok * num_k_heads * qkvz_dim_full +
                         2 * num_k_heads * head_k_dim +
                         num_v_heads * head_v_dim + k_head_id * z_dim +
                         z_dim_id;
          } else {
            mixed_z_id = global_tok * num_k_heads * qkvz_dim_full +
                         k_head_id * qkvz_dim_full + qkv_dim_full + z_dim_id;
          }
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            z_out
                [global_tok * num_k_heads * z_dim + k_head_id * z_dim +
                 z_dim_id + e] = mixed_qkvz[mixed_z_id + e];
          }
        }

        // b/a reorder: only item 0 does this (kv_ratio=2 elements per head)
        if (local_id == 0) {
          if constexpr (ReorderInput) {
            int step = global_tok * num_v_heads * 2;
#pragma unroll
            for (int e = 0; e < kv_ratio; ++e) {
              float b_val =
                  static_cast<float>(mixed_ba[step + k_head_id * kv_ratio + e]);
              float a_val = static_cast<float>(
                  mixed_ba[step + num_v_heads + k_head_id * kv_ratio + e]);
              b_val = 1.0f / (1.0f + sycl::exp(-b_val));
              b_out
                  [(k_head_id * kv_ratio + e) * num_virtual_tokens +
                   out_token_id] = b_val;
              a_out
                  [(k_head_id * kv_ratio + e) * num_virtual_tokens +
                   out_token_id] = a_val;
            }
          } else {
            int step = (global_tok * num_v_heads +
                        k_head_id * num_v_heads / num_k_heads) *
                       2;
#pragma unroll
            for (int e = 0; e < kv_ratio; ++e) {
              float b_val = static_cast<float>(mixed_ba[step + e]);
              float a_val = static_cast<float>(mixed_ba[step + kv_ratio + e]);
              b_val = 1.0f / (1.0f + sycl::exp(-b_val));
              b_out
                  [(k_head_id * kv_ratio + e) * num_virtual_tokens +
                   out_token_id] = b_val;
              a_out
                  [(k_head_id * kv_ratio + e) * num_virtual_tokens +
                   out_token_id] = a_val;
            }
          }
        }
      }
    }

    // ========================================================================
    // Phase 3: Save conv_state for the last tile of each sequence
    // ========================================================================
    if (tile_start_in_seq + TileT >= seq_len && seq_len > 1) {
      int last_slot = (tile_tokens - 1) + (Width - 1);
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
        int slot = last_slot - (Width - 2) + i;
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          conv_states_tmp
              [batch_id * (Width - 1) * conv_elems + i * conv_elems +
               reordered_feat + e] =
                  slm_input[slot * feats_per_wg + local_feat + e];
        }
      }
    } else if (seq_len == 1) {
      T* st = conv_states + states_id * conv_states_stride_0;
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
        int slot = i + 1;
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          st[i * conv_elems + reordered_feat + e] =
              slm_input[slot * feats_per_wg + local_feat + e];
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
  const int32_t* query_start_loc;
  const int* cache_indices;
  const bool* has_initial_state;
  const int* token_indx;
  const ActMode act_mode;
  const int pad_slot_id;
  const int batch_size;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
  const int qkvz_elems;
  const int conv_elems;
  const int num_virtual_tokens;
  char* slm_data;
  const bool fuse_l2norm;
};

template <typename T, int Width, int TileT, bool ReorderInput>
void tiled_kernel_launcher(
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
    int* query_start_loc,
    int* cache_indices,
    bool* has_initial_state,
    const int* token_indx,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int& batch_size,
    const int& num_actual_tokens,
    const int& num_virtual_tokens,
    const int& num_tiles,
    const int& num_k_heads,
    const int& head_k_dim,
    const int& num_v_heads,
    const int& head_v_dim,
    const int& qkvz_elems,
    const int& conv_elems,
    const int& num_prefills,
    const int& num_decodes,
    const bool fuse_l2norm) {
  // Note: z_out, b_out, a_out, mixed_ba are passed through to the
  // ZBA reorder kernel below, not to the tiled conv1d kernel itself.
  using KERNEL_MAIN =
      chunk_causal_conv1d_tiled_kernel<T, Width, TileT, ReorderInput>;

  auto range_main = KERNEL_MAIN::get_nd_range(
      num_tiles, num_k_heads, head_k_dim, num_v_heads, head_v_dim);

  int slm_bytes = KERNEL_MAIN::get_slm_bytes();

  queue.submit([&](sycl::handler& cgh) {
    auto slm = sycl::local_accessor<char, 1>(sycl::range<1>(slm_bytes), cgh);
    cgh.parallel_for(range_main, [=](sycl::nd_item<2> item) {
      char* slm_ptr =
          slm.template get_multi_ptr<sycl::access::decorated::no>().get_raw();

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
          token_indx,
          act_mode,
          pad_slot_id,
          batch_size,
          num_k_heads,
          head_k_dim,
          num_v_heads,
          head_v_dim,
          qkvz_elems,
          conv_elems,
          num_virtual_tokens,
          slm_ptr,
          fuse_l2norm);
      task(item);
    });
  });

  // Update conv states from tmp buffer
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

void chunk_causal_conv1d_tiled_xe2(
    sycl::queue& queue,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out,
    torch::Tensor& z_out,
    torch::Tensor& b_out,
    torch::Tensor& a_out,
    const torch::Tensor& mixed_qkvz,
    const torch::Tensor& mixed_ba,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int num_prefills,
    const int num_decodes,
    const bool reorder_input,
    const int* token_indx = nullptr,
    int num_actual_tokens_override = -1,
    const bool fuse_l2norm = false) {
  if (num_prefills == 0 && num_decodes == 0) {
    return;
  }

  const int batch_size = query_start_loc.size(0) - 1;
  const int num_actual_tokens = num_actual_tokens_override >= 0
                                    ? num_actual_tokens_override
                                    : static_cast<int>(mixed_qkvz.size(0));
  const int num_virtual_tokens = q_out.size(0);
  const int num_k_heads = q_out.size(1);
  const int head_k_dim = q_out.size(2);
  const int num_v_heads = v_out.size(1);
  const int head_v_dim = v_out.size(2);
  const int qkvz_elems = mixed_qkvz.size(1);
  const int conv_elems = conv_weights.size(0);
  const int width = conv_weights.size(1);
  const int conv_states_stride_0 = conv_states.stride(0);

  // Upper bound on tile count: each token can start a new tile at worst,
  // plus each batch boundary can add one partial tile. The kernel handles
  // out-of-range tiles gracefully (tile_id >= actual tiles just won't match
  // any batch in the loop and hits pad_slot_id early-exit).
  // Tighter bound: total_tokens/TileT + batch_size (one extra per seq).
  int num_tiles =
      (num_actual_tokens + conv1d_tile_size - 1) / conv1d_tile_size +
      batch_size;

  auto dtype = conv_states.dtype();
  auto device = conv_states.device();
  torch::Tensor conv_states_tmp = torch::empty(
      {batch_size, width - 1, conv_elems},
      torch::dtype(dtype).device(device).requires_grad(false));

  constexpr int TileT = conv1d_tile_size;

#define TILED_KERNEL_LAUNCHER(scalar_t, width, reorder_input)      \
  tiled_kernel_launcher<scalar_t, width, TileT, reorder_input>(    \
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
      reinterpret_cast<int*>(query_start_loc.data_ptr()),          \
      reinterpret_cast<int*>(cache_indices.data_ptr()),            \
      has_initial_state.has_value()                                \
          ? reinterpret_cast<bool*>(has_initial_state->data_ptr()) \
          : nullptr,                                               \
      token_indx,                                                  \
      act_mode,                                                    \
      pad_slot_id,                                                 \
      batch_size,                                                  \
      num_actual_tokens,                                           \
      num_virtual_tokens,                                          \
      num_tiles,                                                   \
      num_k_heads,                                                 \
      head_k_dim,                                                  \
      num_v_heads,                                                 \
      head_v_dim,                                                  \
      qkvz_elems,                                                  \
      conv_elems,                                                  \
      num_prefills,                                                \
      num_decodes,                                                 \
      fuse_l2norm);

#define TILED_WIDTH_DISPATCH(scalar_t, width, reorder_input) \
  switch (width) {                                           \
    case 1:                                                  \
      TILED_KERNEL_LAUNCHER(scalar_t, 1, reorder_input)      \
      break;                                                 \
    case 2:                                                  \
      TILED_KERNEL_LAUNCHER(scalar_t, 2, reorder_input)      \
      break;                                                 \
    case 3:                                                  \
      TILED_KERNEL_LAUNCHER(scalar_t, 3, reorder_input)      \
      break;                                                 \
    case 4:                                                  \
      TILED_KERNEL_LAUNCHER(scalar_t, 4, reorder_input)      \
      break;                                                 \
    case 5:                                                  \
      TILED_KERNEL_LAUNCHER(scalar_t, 5, reorder_input)      \
      break;                                                 \
    default:                                                 \
      break;                                                 \
  }

#define TILED_SPLIT_DISPATCH(scalar_t, width, reorder_input) \
  if (reorder_input) {                                       \
    TILED_WIDTH_DISPATCH(scalar_t, width, true)              \
  } else {                                                   \
    TILED_WIDTH_DISPATCH(scalar_t, width, false)             \
  }

  if (mixed_qkvz.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    TILED_SPLIT_DISPATCH(scalar_t, width, reorder_input)
  } else if (mixed_qkvz.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    TILED_SPLIT_DISPATCH(scalar_t, width, reorder_input)
  } else {
    using scalar_t = float;
    TILED_SPLIT_DISPATCH(scalar_t, width, reorder_input)
  }
#undef TILED_SPLIT_DISPATCH
#undef TILED_WIDTH_DISPATCH
#undef TILED_KERNEL_LAUNCHER
}

}  // namespace gdn
