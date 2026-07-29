#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "gemm.hpp"
#include "gdn_attn_utils.h"
#include "csrc/utils.h"

// Chunked (WY-representation) Kimi Delta Attention recurrence for the prefill /
// mixed path, running the heavy contractions on the XMX matrix engine.
//
// The sequential `recurrent_kda_kernel` walks one token at a time, which makes
// prefill latency bound (~0.5 M tok/s at head_dim=128 on Xe2). This pipeline
// mirrors the proven GDN chunk pipeline
// (`chunk_gated_delta_rule_kernels_xe2.hpp`): the only sequential dependency
// left is the inter-chunk state carry, everything inside a chunk becomes dense
// GEMMs.
//
// KDA differs from GDN in one essential way: the forget gate is per **key
// channel** rather than a scalar per (token, head). GDN can therefore apply the
// decay after the GEMM as `exp(g[m] - g[n])`; KDA cannot, and instead folds the
// decay into the GEMM operands. `beta` is folded in as well, which removes
// every post-GEMM scaling from the inner stages:
//
//   g[c,j]  = -exp(A_log[h]) * softplus(raw_gate[c,j] + dt_bias[h,j])
//   G[c,j]  = inclusive cumsum of g over the chunk (reset at each chunk)
//   Ka[c,j] = k_hat[c,j] * exp( G[c,j])
//   Kb[c,j] = k_hat[c,j] * exp(-G[c,j]) * beta[c]      <- beta folded in
//   Qt[c,j] = q_hat[c,j] * exp( G[c,j]) / sqrt(head_dim)
//   Tl[j]   = exp(G[n-1,j])                            <- end-of-chunk decay
//
//   A       = I + tril_strict(Ka @ Kb^T)
//   W       = A^-1 @ Ka          U0 = A^-1 @ V
//   U       = U0 - W  @ S0^T
//   O       = Qt @ S0^T + tril(Qt @ Kb^T) @ U
//   S_new   = (S0 + U^T @ Kb) * diag(Tl)               <- per-channel, on keys
//
// Because `exp(-G)` grows without bound, `G` is clamped to `g_floor` and the
// cumsum restarts every chunk, which keeps both operands inside the bf16/fp16
// exponent range. The products that actually matter (`m > n`) are unaffected
// since the two exponentials cancel there.
//
// Everything the GEMM stages read is written by `prepare` into a chunk-aligned,
// zero-padded workspace, so no stage below needs predication and the KDA conv
// kernel and op schema stay untouched.

namespace kda_xe2 {
using namespace cute;
// The DPAS GEMM helpers (gemm_TTS, gemm_TTS_fused_2A, ...) live in `gdn` and
// are pure templates, so they are safe to share across translation units.
using gdn::gemm_STS;
using gdn::gemm_TTS;
using gdn::gemm_TTS_fused_2A;
using gdn::gemm_TTS_fused_2B;

static constexpr int MaxThreadsPerSM = 512;
static constexpr int sub_group_size = 16;
static constexpr int chunk_size = gdn::chunk_size_xe2;
static constexpr float l2norm_eps = 0.000001f;
static constexpr int pad_slot_id = -1;
// Numerical precondition of every chunked delta-rule formulation: the
// intra-chunk term needs exp(G[s] - G[r]) expressed as exp(G[s]) * exp(-G[r]),
// so the per-channel cumulative log-decay across one chunk must fit in the
// float exponent range. exp(80) ~ 5.5e34 leaves headroom below the bf16/fp32
// maximum of 3.4e38 (both have an 8-bit exponent, so fp32 operands would not
// help). A channel that decays by more than e^-80 within 64 tokens forgets its
// state ~35 times over inside a single chunk, which no trained model does; the
// clamp is inert in practice and the dispatcher keeps the recurrent kernel
// available as an exact fallback.
static constexpr float g_floor = -80.0f;

static constexpr int prepare_sub_group_size = 32;
static constexpr int prepare_work_group_size = 256;

// DPAS tiling policies. These mirror the GDN chunk pipeline's proven shapes but
// are declared locally so this file does not have to include
// `chunk_gated_delta_rule_kernels_xe2.hpp` (which defines non-inline symbols
// and would clash at link time).
struct chunk_gemm_policy_64x64x32_4x2 {
  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
};

struct chunk_gemm_policy_64x64x32_8x2 {
  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
};

struct chunk_gemm_policy_16x16x16 {
  using WGTile = Shape<_16, _16, _16>;
  using SGLayout = Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>;
};

// The three chunk GEMM stages were swept over the 2x1 / 2x2 / 4x2 / 8x2
// sub-group splits on Battlemage. They are memory bound, so the work-group tile
// - and with it the operand traffic - is the same whatever the split; what
// changes is that more sub-groups issue more concurrent block-2d loads to hide
// latency behind, and that smaller per-lane accumulators leave register
// headroom for the fused two-accumulator products below (a 32x64 sub-group tile
// would need the whole 256-GRF budget for two accumulators alone, and spills).
//
// 4x2 is the sweet spot for the two chunk-parallel stages: 2x2 measured ~10%
// slower for `compute_wu`, while 8x2 splits their already-short K loops too
// thinly and costs up to 17% on head_dim == 64 shapes. `fwd_o` is the exception
// and wants 8x2, because its grid is by far the narrowest in the pipeline (see
// `chunk_kda_fwd_o_dv_groups`), so per-work-group width is the only occupancy
// knob it has.
using chunk_gemm_policy_compute_A = chunk_gemm_policy_64x64x32_4x2;
using chunk_gemm_policy_inverse = chunk_gemm_policy_16x16x16;
using chunk_gemm_policy_compute_wu = chunk_gemm_policy_64x64x32_4x2;
using chunk_gemm_policy_fwd_o = chunk_gemm_policy_64x64x32_8x2;

// A Battlemage tile holds 32 Xe-cores x 2048 work-item slots, so ~65536
// in-flight work items saturate it. The budget is expressed in work items
// rather than work-groups so that it stays correct when the tiling policy
// above changes the work-group width.
static constexpr int fwd_o_target_work_items = 32 * 2048;
static constexpr int fwd_o_wg_size =
    cute::size(chunk_gemm_policy_fwd_o::SGLayout{}) * sub_group_size;

// `fwd_o` is the only stage whose grid is capped at batch_size x num_heads,
// because it walks the chunks of a sequence in order. The recurrence is however
// separable along the value dimension: state row block `dv` only ever reads
// `U[:, dv]` and writes `S[dv, :]`, and the output column block `dv` only reads
// those. Splitting `dv` across work-groups therefore needs no extra
// synchronisation - it only duplicates the intra-chunk `O2 = tril(Qt @ Kb^T)`
// tile, which is why each group gets its own scratch plane and why the split is
// skipped once the unsplit grid already fills the machine.
inline int
chunk_kda_fwd_o_dv_groups(int batch_size, int num_heads, int head_dim) {
  const int ndv = head_dim / chunk_size;
  const int64_t base_items =
      static_cast<int64_t>(batch_size) * num_heads * fwd_o_wg_size;
  int groups = 1;
  for (int g = ndv; g > 1; --g) {
    if (ndv % g != 0) {
      continue;
    }
    if (base_items * g <= fwd_o_target_work_items) {
      groups = g;
      break;
    }
  }
  return groups;
}

// Branch-free so the `prepare` cumsum unrolls cleanly: for x >= 20 the
// exponential saturates to +inf and log(1 + inf) = inf, which the select then
// discards, so no NaN can escape.
CUTE_DEVICE float native_softplus(float x) {
  const float saturated = sycl::native::log(1.0f + sycl::native::exp(x));
  return x < 20.0f ? saturated : x;
}

// How many tokens the per-channel cumsum processes per unrolled step. The loads
// of one step are independent of the recurrence, so unrolling is what gives the
// serial scan enough memory-level parallelism to cover DRAM latency.
static constexpr int prepare_unroll = 4;

// Naturally aligned POD pack. `sycl::vec` cannot be instantiated on
// `cutlass::bfloat16_t`, but an aligned trivially copyable struct still lowers
// to one wide load/store message, which is the only property that matters here.
template <typename T, int N>
struct alignas(sizeof(T) * N) VecPack {
  T data[N];
};

template <typename T, int N>
CUTE_DEVICE VecPack<T, N> load_pack(const T* ptr) {
  return *reinterpret_cast<const VecPack<T, N>*>(ptr);
}

template <typename T, int N>
CUTE_DEVICE void store_pack(T* ptr, const VecPack<T, N>& value) {
  *reinterpret_cast<VecPack<T, N>*>(ptr) = value;
}

template <typename T, int N>
CUTE_DEVICE void pack_to_float(const VecPack<T, N>& src, float (&dst)[N]) {
  CUTE_UNROLL
  for (int i = 0; i < N; ++i) {
    dst[i] = static_cast<float>(src.data[i]);
  }
}

template <typename T, int N>
CUTE_DEVICE void store_from_float(T* ptr, const float (&src)[N]) {
  VecPack<T, N> out;
  CUTE_UNROLL
  for (int i = 0; i < N; ++i) {
    out.data[i] = static_cast<T>(src[i]);
  }
  store_pack<T, N>(ptr, out);
}

// ---------------------------------------------------------------------------
// Stage 1 (vectorized): prepare, one sub-group per (chunk, head)
//
// The scalar variant below spends one 2-byte message per lane per array per
// token, which makes `prepare` message-bound rather than bandwidth-bound. Here
// lane L owns the `V = head_dim / sub_group_size` *consecutive* channels
// starting at L * V, so every access is a single fully coalesced wide message
// covering the whole row. Because one sub-group then covers all `head_dim`
// channels, the L2 norms collapse to a plain sub-group reduction: q and k are
// read once instead of twice, no SLM and no barrier is needed anywhere, and the
// chunk mapping can be made per sub-group exactly like the GDN prepare stage.
// ---------------------------------------------------------------------------
template <typename T, int V>
CUTE_DEVICE void chunk_kda_prepare_vec_kernel(
    T* Ka,
    T* Kb,
    T* Qt,
    T* Vp,
    float* Tl,
    const T* q,
    const T* k,
    const T* v,
    const T* raw_gate,
    const float* beta,
    const float* a_log,
    const float* dt_bias,
    const int* query_start_loc,
    const int* token_indx,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_heads,
    const int head_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto sg = item.get_sub_group();
  const int sg_id = sg.get_group_linear_id();
  const int sg_range = sg.get_group_linear_range();
  const int lane = sg.get_local_linear_id();

  const int total_sg_range = item.get_group_range(1) * sg_range;
  const int total_sg_id = item.get_group(1) * sg_range + sg_id;

  const int chunk_range = total_sg_range / num_heads;
  if (chunk_range == 0) {
    return;
  }
  int chunk_id = total_sg_id % chunk_range;
  const int head_id = total_sg_id / chunk_range;
  if (head_id >= num_heads) {
    return;
  }

  const float head_a = -sycl::native::exp(a_log[head_id]);
  const float q_scale = sycl::native::rsqrt(static_cast<float>(head_dim));
  const int64_t token_stride = static_cast<int64_t>(num_heads) * head_dim;
  const int64_t lane_off = static_cast<int64_t>(head_id) * head_dim + lane * V;
  const int num_virtual_chunks = total_virtual_seqlen / chunk_size;

  float bias[V];
  {
    const VecPack<float, V> bias_pack = load_pack<float, V>(
        dt_bias + static_cast<int64_t>(head_id) * head_dim + lane * V);
    CUTE_UNROLL
    for (int e = 0; e < V; ++e) {
      bias[e] = bias_pack.data[e];
    }
  }

  int pre_chunks = 0;
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start = query_start_loc[batch_id];
    const int seq_len = query_start_loc[batch_id + 1] - seq_start;
    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      const int chunk_token_start = (chunk_id - pre_chunks) * chunk_size;
      const int valid = sycl::min(chunk_size, seq_len - chunk_token_start);
      const int64_t out_base =
          static_cast<int64_t>(head_id) * total_virtual_seqlen * head_dim +
          static_cast<int64_t>(chunk_id) * chunk_size * head_dim + lane * V;

      float log_cum[V];
      float last_decay[V];
      CUTE_UNROLL
      for (int e = 0; e < V; ++e) {
        log_cum[e] = 0.0f;
        last_decay[e] = 1.0f;
      }

      for (int c = 0; c < valid; ++c) {
        const int local_token = seq_start + chunk_token_start + c;
        const int global_token =
            token_indx == nullptr ? local_token : token_indx[local_token];
        const int64_t in_off =
            static_cast<int64_t>(global_token) * token_stride + lane_off;

        const VecPack<T, V> v_pack = load_pack<T, V>(v + in_off);
        float qv[V];
        float kv[V];
        float gv[V];
        pack_to_float<T, V>(load_pack<T, V>(q + in_off), qv);
        pack_to_float<T, V>(load_pack<T, V>(k + in_off), kv);
        pack_to_float<T, V>(load_pack<T, V>(raw_gate + in_off), gv);
        const float beta_value =
            beta[static_cast<int64_t>(global_token) * num_heads + head_id];

        float q_sum = 0.0f;
        float k_sum = 0.0f;
        CUTE_UNROLL
        for (int e = 0; e < V; ++e) {
          q_sum += qv[e] * qv[e];
          k_sum += kv[e] * kv[e];
        }
        // One sub-group spans the whole row, so this single reduction is the
        // complete L2 norm; the scalar variant needs a separate SLM phase.
        q_sum = sycl::reduce_over_group(sg, q_sum, sycl::plus<>());
        k_sum = sycl::reduce_over_group(sg, k_sum, sycl::plus<>());
        const float q_inv = sycl::native::rsqrt(q_sum + l2norm_eps) * q_scale;
        const float k_inv = sycl::native::rsqrt(k_sum + l2norm_eps);

        float ka[V];
        float kb[V];
        float qt[V];
        CUTE_UNROLL
        for (int e = 0; e < V; ++e) {
          log_cum[e] = sycl::fmax(
              log_cum[e] + head_a * native_softplus(gv[e] + bias[e]), g_floor);
          const float decay = sycl::native::exp(log_cum[e]);
          const float inv_decay = sycl::native::recip(decay);
          last_decay[e] = decay;
          const float k_hat = kv[e] * k_inv;
          const float q_hat = qv[e] * q_inv;
          ka[e] = k_hat * decay;
          kb[e] = k_hat * inv_decay * beta_value;
          qt[e] = q_hat * decay;
        }

        const int64_t out_off = out_base + static_cast<int64_t>(c) * head_dim;
        store_from_float<T, V>(Ka + out_off, ka);
        store_from_float<T, V>(Kb + out_off, kb);
        store_from_float<T, V>(Qt + out_off, qt);
        store_pack<T, V>(Vp + out_off, v_pack);
      }

      // The GEMM stages read whole chunks unpredicated, so the tail past the
      // sequence end has to be zeroed.
      if (valid < chunk_size) {
        VecPack<T, V> zero_pack;
        CUTE_UNROLL
        for (int e = 0; e < V; ++e) {
          zero_pack.data[e] = static_cast<T>(0.0f);
        }
        for (int c = valid; c < chunk_size; ++c) {
          const int64_t out_off = out_base + static_cast<int64_t>(c) * head_dim;
          store_pack<T, V>(Ka + out_off, zero_pack);
          store_pack<T, V>(Kb + out_off, zero_pack);
          store_pack<T, V>(Qt + out_off, zero_pack);
          store_pack<T, V>(Vp + out_off, zero_pack);
        }
      }

      VecPack<float, V> tl_pack;
      CUTE_UNROLL
      for (int e = 0; e < V; ++e) {
        tl_pack.data[e] = last_decay[e];
      }
      store_pack<float, V>(
          Tl + static_cast<int64_t>(head_id) * num_virtual_chunks * head_dim +
              static_cast<int64_t>(chunk_id) * head_dim + lane * V,
          tl_pack);

      chunk_id += chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

// ---------------------------------------------------------------------------
// Stage 1: prepare
//
// Packs the compact, ragged activations into the chunk-aligned workspace and
// applies the L2 norm, the per-channel decay cumsum and the beta folding.
// One work-group owns one (chunk, head); phase A derives the per-token inverse
// norms with cheap sub-group reductions, phase B walks the chunk serially with
// one thread per key channel so the cumsum needs no communication at all.
// ---------------------------------------------------------------------------
template <typename T>
CUTE_DEVICE void chunk_kda_prepare_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,
    T* Ka,
    T* Kb,
    T* Qt,
    T* Vp,
    float* Tl,
    const T* q,
    const T* k,
    const T* v,
    const T* raw_gate,
    const float* beta,
    const float* a_log,
    const float* dt_bias,
    const int* query_start_loc,
    const int* token_indx,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_heads,
    const int head_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int local_id = item.get_local_linear_id();
  const int local_range = item.get_local_range(2);
  const int wg_id = item.get_group(1);
  const int wg_range = item.get_group_range(1);

  auto sg = item.get_sub_group();
  const int sg_id = sg.get_group_linear_id();
  const int sg_range = sg.get_group_linear_range();
  const int sg_local_id = sg.get_local_linear_id();

  const int chunk_range = wg_range / num_heads;
  if (chunk_range == 0) {
    return;
  }
  int chunk_id = wg_id % chunk_range;
  const int head_id = wg_id / chunk_range;
  if (head_id >= num_heads) {
    return;
  }

  float* slm = static_cast<float*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());
  float* q_inv_slm = slm;                     // [chunk_size]
  float* k_inv_slm = q_inv_slm + chunk_size;  // [chunk_size]
  float* beta_slm = k_inv_slm + chunk_size;   // [chunk_size]
  // Phase A already resolves the ragged->packed token mapping, so caching it
  // keeps the serial cumsum in phase B free of dependent gathers.
  int* token_slm = reinterpret_cast<int*>(beta_slm + chunk_size);

  const float head_a = -sycl::native::exp(a_log[head_id]);
  const float q_scale = sycl::native::rsqrt(static_cast<float>(head_dim));
  const int keys_per_lane = head_dim / prepare_sub_group_size;
  const bool vectorizable = (keys_per_lane & 3) == 0;
  const int64_t token_stride = static_cast<int64_t>(num_heads) * head_dim;

  int pre_chunks = 0;
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start = query_start_loc[batch_id];
    const int seq_len = query_start_loc[batch_id + 1] - seq_start;
    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      const int local_chunk = chunk_id - pre_chunks;
      const int chunk_token_start = local_chunk * chunk_size;
      const int valid = sycl::min(chunk_size, seq_len - chunk_token_start);

      item.barrier(sycl::access::fence_space::local_space);

      // --- phase A: per-token inverse L2 norms (one sub-group per token) ---
      for (int c = sg_id; c < chunk_size; c += sg_range) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        float beta_value = 0.0f;
        int global_token = 0;
        if (c < valid) {
          const int local_token = seq_start + chunk_token_start + c;
          global_token =
              token_indx == nullptr ? local_token : token_indx[local_token];
          const int64_t base =
              static_cast<int64_t>(global_token) * token_stride +
              static_cast<int64_t>(head_id) * head_dim;
          if (vectorizable) {
            // head_dim is a multiple of 4 here, and the row base is a multiple
            // of head_dim, so the vec4 view is always correctly aligned. One
            // 8-byte load per lane replaces four strided 2-byte gathers.
            using Vec4 = sycl::vec<T, 4>;
            const Vec4* q4 = reinterpret_cast<const Vec4*>(q + base);
            const Vec4* k4 = reinterpret_cast<const Vec4*>(k + base);
            const int vecs_per_lane = keys_per_lane >> 2;
            for (int e = 0; e < vecs_per_lane; ++e) {
              const int idx = e * prepare_sub_group_size + sg_local_id;
              const Vec4 qv = q4[idx];
              const Vec4 kv = k4[idx];
              CUTE_UNROLL
              for (int t = 0; t < 4; ++t) {
                const float qf = static_cast<float>(qv[t]);
                const float kf = static_cast<float>(kv[t]);
                q_sum += qf * qf;
                k_sum += kf * kf;
              }
            }
          } else {
            for (int e = 0; e < keys_per_lane; ++e) {
              const int j = e * prepare_sub_group_size + sg_local_id;
              const float qv = static_cast<float>(q[base + j]);
              const float kv = static_cast<float>(k[base + j]);
              q_sum += qv * qv;
              k_sum += kv * kv;
            }
          }
          beta_value =
              beta[static_cast<int64_t>(global_token) * num_heads + head_id];
        }
        q_sum = sycl::reduce_over_group(sg, q_sum, sycl::plus<>());
        k_sum = sycl::reduce_over_group(sg, k_sum, sycl::plus<>());
        if (sg_local_id == 0) {
          q_inv_slm[c] = sycl::native::rsqrt(q_sum + l2norm_eps) * q_scale;
          k_inv_slm[c] = sycl::native::rsqrt(k_sum + l2norm_eps);
          beta_slm[c] = beta_value;
          token_slm[c] = global_token;
        }
      }

      item.barrier(sycl::access::fence_space::local_space);

      // --- phase B: serial per-channel cumsum (one thread per key channel) ---
      for (int j = local_id; j < head_dim; j += local_range) {
        const float bias =
            dt_bias[static_cast<int64_t>(head_id) * head_dim + j];
        const int64_t out_base =
            static_cast<int64_t>(head_id) * total_virtual_seqlen * head_dim +
            static_cast<int64_t>(chunk_id) * chunk_size * head_dim + j;
        const int64_t in_head_off =
            static_cast<int64_t>(head_id) * head_dim + j;
        float log_cum = 0.0f;
        float last_decay = 1.0f;

        // `log_cum` only ever decreases (head_a < 0, softplus > 0) and is
        // clamped at `g_floor`, so max(x + d, floor) composes to
        // max(x + sum(d), floor): the unrolled steps stay bit-equivalent to the
        // scalar recurrence. Every address in a step is known up front, so the
        // four token loads issue back to back instead of one per iteration.
        auto emit = [&](int c, float gate, float kv, float qv, T vv) {
          log_cum += head_a * native_softplus(gate + bias);
          log_cum = sycl::fmax(log_cum, g_floor);
          const float decay = sycl::native::exp(log_cum);
          // exp(-log_cum) costs a second transcendental and, worse, does not
          // cancel exactly against `decay`; the reciprocal does both better.
          const float inv_decay = sycl::native::recip(decay);
          last_decay = decay;

          const float k_hat = kv * k_inv_slm[c];
          const float q_hat = qv * q_inv_slm[c];
          const int64_t out_off = out_base + static_cast<int64_t>(c) * head_dim;
          Ka[out_off] = static_cast<T>(k_hat * decay);
          Kb[out_off] = static_cast<T>(k_hat * inv_decay * beta_slm[c]);
          Qt[out_off] = static_cast<T>(q_hat * decay);
          Vp[out_off] = vv;
        };

        int c = 0;
        for (; c + prepare_unroll <= valid; c += prepare_unroll) {
          int64_t in_off[prepare_unroll];
          float gate[prepare_unroll];
          float kv[prepare_unroll];
          float qv[prepare_unroll];
          T vv[prepare_unroll];
          CUTE_UNROLL
          for (int u = 0; u < prepare_unroll; ++u) {
            in_off[u] = static_cast<int64_t>(token_slm[c + u]) * token_stride +
                        in_head_off;
          }
          CUTE_UNROLL
          for (int u = 0; u < prepare_unroll; ++u) {
            gate[u] = static_cast<float>(raw_gate[in_off[u]]);
            kv[u] = static_cast<float>(k[in_off[u]]);
            qv[u] = static_cast<float>(q[in_off[u]]);
            vv[u] = v[in_off[u]];
          }
          CUTE_UNROLL
          for (int u = 0; u < prepare_unroll; ++u) {
            emit(c + u, gate[u], kv[u], qv[u], vv[u]);
          }
        }
        for (; c < valid; ++c) {
          const int64_t in_off =
              static_cast<int64_t>(token_slm[c]) * token_stride + in_head_off;
          emit(
              c,
              static_cast<float>(raw_gate[in_off]),
              static_cast<float>(k[in_off]),
              static_cast<float>(q[in_off]),
              v[in_off]);
        }
        // Tail padding: the GEMM stages read the whole chunk unpredicated, so
        // the workspace has to be zero-filled past the sequence end.
        for (; c < chunk_size; ++c) {
          const int64_t out_off = out_base + static_cast<int64_t>(c) * head_dim;
          Ka[out_off] = static_cast<T>(0.0f);
          Kb[out_off] = static_cast<T>(0.0f);
          Qt[out_off] = static_cast<T>(0.0f);
          Vp[out_off] = static_cast<T>(0.0f);
        }
        Tl[static_cast<int64_t>(head_id) * (total_virtual_seqlen / chunk_size) *
               head_dim +
           static_cast<int64_t>(chunk_id) * head_dim + j] = last_decay;
      }

      chunk_id += chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

// ---------------------------------------------------------------------------
// Stage 2: A = I + tril_strict(Ka @ Kb^T)
// ---------------------------------------------------------------------------
template <typename T, class TiledMMA>
CUTE_DEVICE void chunk_kda_compute_A_kernel(
    T* A,
    const T* Ka,
    const T* Kb,
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_heads,
    const int head_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int local_id = item.get_local_linear_id();
  // One work-group per (chunk, head): looping every head inside a
  // chunk-indexed work-group would cap parallelism at the chunk count.
  const int head_id = item.get_group(1) % num_heads;
  int chunk_id = item.get_group(1) / num_heads;
  const int global_chunk_range = item.get_group_range(1) / num_heads;

  auto sg = item.get_sub_group();
  const int sg_local_id = sg.get_local_linear_id();

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(local_id);

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto SG_M = tile_m / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;

  const auto sg_local_m_coord = cutlass::get_sub_group_id() / ATOM_N;
  const auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  const int m_sg_start = sg_local_m_coord * SG_M;
  const int n_sg_start = sg_local_n_coord * SG_N;

  int pre_chunks = 0;
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start = query_start_loc[batch_id];
    const int seq_len = query_start_loc[batch_id + 1] - seq_start;
    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      {
        const int64_t operand_offset =
            static_cast<int64_t>(head_id) * total_virtual_seqlen * head_dim +
            static_cast<int64_t>(chunk_id) * chunk_size * head_dim;

        auto Ka_tensor = make_tensor(
            make_gmem_ptr(Ka + operand_offset),
            make_layout(
                make_shape(chunk_size, head_dim), make_stride(head_dim, _1{})));
        auto Kb_tensor = make_tensor(
            make_gmem_ptr(Kb + operand_offset),
            make_layout(
                make_shape(chunk_size, head_dim), make_stride(head_dim, _1{})));

        auto A_ptr =
            A +
            static_cast<int64_t>(head_id) * total_virtual_seqlen * chunk_size +
            static_cast<int64_t>(chunk_id) * chunk_size * chunk_size;
        auto A_tensor = make_tensor(
            make_gmem_ptr(A_ptr),
            make_layout(
                make_shape(chunk_size, chunk_size),
                make_stride(chunk_size, _1{})));

        Tensor cA = make_identity_tensor(A_tensor.shape());
        Tensor gA_C =
            local_tile(cA, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});

        auto copy_A_c = get_block_2d_copy_D<void>(mma, A_tensor);
        auto thr_copy_A_c = copy_A_c.get_slice(local_id);
        auto tCrA_c = thr_copy_A_c.partition_sg_fragment_S(gA_C);
        auto tCgA_c = thr_copy_A_c.partition_D(gA_C);
        auto tSrA_c = thr_mma.partition_sg_fragment_C(gA_C);

        clear(tSrA_c);
        // Sub-group tiles that sit entirely above the diagonal are thrown away
        // by the masked epilogue below, but they cannot be skipped: gemm_TTS
        // uses work-group split barriers, so every sub-group has to run it.
        gemm_TTS(Ka_tensor, Kb_tensor, tSrA_c, 0, 0, mma);

        // beta and the decay are already baked into Ka/Kb, so the epilogue is
        // the unit-lower-triangular mask only.
        CUTE_UNROLL
        for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
          const int n_idx = n_sg_start + sn * sub_group_size + sg_local_id;
          CUTE_UNROLL
          for (int sm = 0; sm < SG_M; ++sm) {
            const int m_idx = m_sg_start + sm;
            if (m_idx == n_idx) {
              tSrA_c(sn * SG_M + sm) = 1.0f;
            } else if (m_idx < n_idx) {
              tSrA_c(sn * SG_M + sm) = 0.0f;
            }
          }
        }

        reorder(tSrA_c, tCrA_c);
        copy(copy_A_c, tCrA_c, tCgA_c);
      }
      chunk_id += global_chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

// ---------------------------------------------------------------------------
// Stage 3: A := (I + A)^-1 by unit-lower-triangular forward substitution.
//
// This is O(chunk^3 / 6) ~ 43 K ops per (chunk, head), well under 1% of the
// pipeline, so the scalar formulation costs nothing measurable and avoids the
// blocked-DPAS inverse.
// ---------------------------------------------------------------------------
template <typename T>
CUTE_DEVICE void chunk_kda_inverse_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,
    T* A,
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_heads) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int local_id = item.get_local_linear_id();
  const int local_range = item.get_local_range(2);
  int chunk_id = item.get_group(1);
  const int global_chunk_range = item.get_group_range(1);

  float* slm = static_cast<float*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());
  float* A_load = slm;
  float* A_save = A_load + chunk_size * chunk_size;

  int pre_chunks = 0;
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start = query_start_loc[batch_id];
    const int seq_len = query_start_loc[batch_id + 1] - seq_start;
    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      for (int head_id = 0; head_id < num_heads; ++head_id) {
        T* A_ptr =
            A +
            static_cast<int64_t>(head_id) * total_virtual_seqlen * chunk_size +
            static_cast<int64_t>(chunk_id) * chunk_size * chunk_size;

        item.barrier(sycl::access::fence_space::local_space);
        for (int idx = local_id; idx < chunk_size * chunk_size;
             idx += local_range) {
          A_load[idx] = static_cast<float>(A_ptr[idx]);
          A_save[idx] = 0.0f;
        }
        for (int idx = local_id; idx < chunk_size; idx += local_range) {
          A_save[idx * chunk_size + idx] = 1.0f;
        }
        item.barrier(sycl::access::fence_space::local_space);

        // Column-parallel forward substitution: column n of the inverse only
        // depends on rows above the current one within that same column.
        for (int n_idx = local_id; n_idx < chunk_size; n_idx += local_range) {
          for (int m_idx = n_idx + 1; m_idx < chunk_size; ++m_idx) {
            // X[m][n] = -( L[m][n] * X[n][n] + sum_{l=n+1}^{m-1} L[m][l]
            // X[l][n] ) and X[n][n] == 1, so the l == n term is the seed above.
            float sum = A_load[m_idx * chunk_size + n_idx];
            for (int l = n_idx + 1; l < m_idx; ++l) {
              sum += A_save[l * chunk_size + n_idx] *
                     A_load[m_idx * chunk_size + l];
            }
            A_save[m_idx * chunk_size + n_idx] = -sum;
          }
        }
        item.barrier(sycl::access::fence_space::local_space);

        for (int idx = local_id; idx < chunk_size * chunk_size;
             idx += local_range) {
          A_ptr[idx] = static_cast<T>(A_save[idx]);
        }
      }
      chunk_id += global_chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

// Blocked DPAS inversion of the unit lower-triangular (I + A), ported from the
// GDN pipeline (`chunk_inverse_opt_kernel`). The 64x64 matrix is split into
// 4x4 blocks of 16: the diagonal blocks are inverted with sub-group
// broadcasts and the off-diagonal blocks fall out of small DPAS GEMMs. This
// replaces a scalar forward substitution whose 64-step serial dependency and
// per-lane divergence made it the dominant cost of the whole pipeline.
template <typename T, class TiledMMA>
CUTE_DEVICE void chunk_kda_inverse_opt_kernel(
    T* A,
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_heads) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  int local_range = item.get_local_range(2);
  int head_id = item.get_group(1) % num_heads;
  int chunk_id = item.get_group(1) / num_heads;
  const int global_chunk_range = item.get_group_range(1) / num_heads;

  // l2norm for q, k
  int group_id = item.get_group(1);
  int group_range = item.get_group_range(1);
  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_linear_id();
  int sg_range = sg.get_group_linear_range();
  int sg_local_id = sg.get_local_linear_id();

  int pre_chunks = 0;

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(local_id);

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seq_len = seq_end_offset - seq_start_offset;

    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      const int chunk_start_offset = chunk_id * chunk_size;

      auto A_ptr =
          A +
          static_cast<int64_t>(head_id) * total_virtual_seqlen * chunk_size +
          chunk_start_offset * chunk_size;

      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) {
        int offset = i * 16;
        T* A_ptr_xx = A_ptr + offset * chunk_size + offset;
        float A_local[16];
        float A_other[16];
        float A_sum;
        CUTE_UNROLL
        for (int e = 0; e < sg_local_id + 1; ++e) {
          A_local[e] = 0.0f;
        }

        T A_load[16];
        CUTE_UNROLL
        for (int e = 0; e < sg_local_id; ++e) {
          A_load[e] = A_ptr_xx[sg_local_id * chunk_size + e];
        }

        CUTE_UNROLL
        for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
          CUTE_UNROLL
          for (int nn_idx = 0; nn_idx < mm_idx; ++nn_idx) {
            float send_value = static_cast<float>(A_load[nn_idx]);
            float receive_value = sycl::group_broadcast(sg, send_value, mm_idx);
            if (sg_local_id == nn_idx) {
              A_local[mm_idx] = receive_value;
            }
          }
        }

        CUTE_UNROLL
        for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
          A_sum = 0.0f;
          CUTE_UNROLL
          for (int e = 1; e < mm_idx + 1; ++e) {
            A_other[e] = sycl::group_broadcast(sg, A_local[mm_idx], e);
          }

          CUTE_UNROLL
          for (int e = 1; e < mm_idx + 1; ++e) {
            A_sum += A_local[e] * A_other[e];
          }

          A_local[mm_idx] = -A_local[mm_idx] - A_sum;
        }

        CUTE_UNROLL
        for (int e = sg_local_id + 1; e < 16; ++e) {
          A_ptr_xx[e * chunk_size + sg_local_id] = static_cast<T>(A_local[e]);
        }
      }

      auto A_ptr_11 = A_ptr;

      auto A_ptr_21 = A_ptr + 16 * chunk_size;
      auto A_ptr_22 = A_ptr + 16 * chunk_size + 16;

      auto A_ptr_31 = A_ptr + 32 * chunk_size;
      auto A_ptr_32 = A_ptr + 32 * chunk_size + 16;
      auto A_ptr_33 = A_ptr + 32 * chunk_size + 32;

      auto A_ptr_41 = A_ptr + 48 * chunk_size;
      auto A_ptr_42 = A_ptr + 48 * chunk_size + 16;
      auto A_ptr_43 = A_ptr + 48 * chunk_size + 32;
      auto A_ptr_44 = A_ptr + 48 * chunk_size + 48;

      auto A_XX_tensor_shape = make_shape(16, 16);

      auto A_11_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_11),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

      auto A_21_tensor = make_tensor(
          make_gmem_ptr(A_ptr_21),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_21_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_21),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_22_tensor = make_tensor(
          make_gmem_ptr(A_ptr_22),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_22_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_22),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

      auto A_31_tensor = make_tensor(
          make_gmem_ptr(A_ptr_31),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_31_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_31),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_32_tensor = make_tensor(
          make_gmem_ptr(A_ptr_32),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_32_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_32),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_33_tensor = make_tensor(
          make_gmem_ptr(A_ptr_33),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_33_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_33),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

      auto A_41_tensor = make_tensor(
          make_gmem_ptr(A_ptr_41),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_41_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_41),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_42_tensor = make_tensor(
          make_gmem_ptr(A_ptr_42),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_42_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_42),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_43_tensor = make_tensor(
          make_gmem_ptr(A_ptr_43),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_43_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_43),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_44_tensor = make_tensor(
          make_gmem_ptr(A_ptr_44),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));

      Tensor cA = make_identity_tensor(A_XX_tensor_shape);
      Tensor cB = make_identity_tensor(A_XX_tensor_shape);
      Tensor cC = make_identity_tensor(A_XX_tensor_shape);
      Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(0, _));
      Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(0, _));
      Tensor gC =
          local_tile(cC, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
      auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
      auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
      auto tCrC = thr_mma.partition_sg_fragment_C(gC);

      auto copy_D_21 = get_block_2d_copy_D<void>(mma, A_21_tensor);
      auto thr_copy_D_21 = copy_D_21.get_slice(local_id);
      auto tCrD_21 = thr_copy_D_21.partition_sg_fragment_S(gC);
      auto tCgD_21 = thr_copy_D_21.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_22_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrA);
      clear(tCrC);
      gemm_STS(tCrA, A_11_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_21);
      copy(copy_D_21, tCrD_21, tCgD_21);

      auto copy_D_31 = get_block_2d_copy_D<void>(mma, A_31_tensor);
      auto thr_copy_D_31 = copy_D_31.get_slice(local_id);
      auto tCrD_31 = thr_copy_D_31.partition_sg_fragment_S(gC);
      auto tCgD_31 = thr_copy_D_31.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_31_tensor, A_11_tensor_T, tCrC, 0, 0, mma);
      gemm_TTS(A_32_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrD_31);
      copy(copy_D_31, tCrD_31, tCgD_31);
      clear(tCrC);
      gemm_TTS(A_33_tensor, A_31_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_31);
      copy(copy_D_31, tCrD_31, tCgD_31);

      auto copy_D_41 = get_block_2d_copy_D<void>(mma, A_41_tensor);
      auto thr_copy_D_41 = copy_D_41.get_slice(local_id);
      auto tCrD_41 = thr_copy_D_41.partition_sg_fragment_S(gC);
      auto tCgD_41 = thr_copy_D_41.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_41_tensor, A_11_tensor_T, tCrC, 0, 0, mma);
      gemm_TTS(A_42_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
      gemm_TTS(A_43_tensor, A_31_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrD_41);
      copy(copy_D_41, tCrD_41, tCgD_41);
      clear(tCrC);
      gemm_TTS(A_44_tensor, A_41_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_41);
      copy(copy_D_41, tCrD_41, tCgD_41);

      auto copy_D_32 = get_block_2d_copy_D<void>(mma, A_32_tensor);
      auto thr_copy_D_32 = copy_D_32.get_slice(local_id);
      auto tCrD_32 = thr_copy_D_32.partition_sg_fragment_S(gC);
      auto tCgD_32 = thr_copy_D_32.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_33_tensor, A_32_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrA);
      clear(tCrC);
      gemm_STS(tCrA, A_22_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_32);
      copy(copy_D_32, tCrD_32, tCgD_32);

      auto copy_D_42 = get_block_2d_copy_D<void>(mma, A_42_tensor);
      auto thr_copy_D_42 = copy_D_42.get_slice(local_id);
      auto tCrD_42 = thr_copy_D_42.partition_sg_fragment_S(gC);
      auto tCgD_42 = thr_copy_D_42.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_42_tensor, A_22_tensor_T, tCrC, 0, 0, mma);
      gemm_TTS(A_43_tensor, A_32_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrD_42);
      copy(copy_D_42, tCrD_42, tCgD_42);
      clear(tCrC);
      gemm_TTS(A_44_tensor, A_42_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_42);
      copy(copy_D_42, tCrD_42, tCgD_42);

      auto copy_D_43 = get_block_2d_copy_D<void>(mma, A_43_tensor);
      auto thr_copy_D_43 = copy_D_43.get_slice(local_id);
      auto tCrD_43 = thr_copy_D_43.partition_sg_fragment_S(gC);
      auto tCgD_43 = thr_copy_D_43.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_44_tensor, A_43_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrA);
      clear(tCrC);
      gemm_STS(tCrA, A_33_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_43);
      copy(copy_D_43, tCrD_43, tCgD_43);

      chunk_id += global_chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

// ---------------------------------------------------------------------------
// Stage 4: W = A^-1 @ Ka, U = A^-1 @ Vp
// ---------------------------------------------------------------------------
template <typename T, class TiledMMA>
CUTE_DEVICE void chunk_kda_compute_wu_kernel(
    const T* A,
    T* W,
    T* U,
    const T* Ka,
    const T* Vp,
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_heads,
    const int head_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int local_id = item.get_local_linear_id();
  const int head_id = item.get_group(1) % num_heads;
  int chunk_id = item.get_group(1) / num_heads;
  const int global_chunk_range = item.get_group_range(1) / num_heads;

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(local_id);

  int pre_chunks = 0;
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start = query_start_loc[batch_id];
    const int seq_len = query_start_loc[batch_id + 1] - seq_start;
    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      {
        const int64_t operand_offset =
            static_cast<int64_t>(head_id) * total_virtual_seqlen * head_dim +
            static_cast<int64_t>(chunk_id) * chunk_size * head_dim;
        auto A_ptr =
            A +
            static_cast<int64_t>(head_id) * total_virtual_seqlen * chunk_size +
            static_cast<int64_t>(chunk_id) * chunk_size * chunk_size;
        auto A_tensor = make_tensor(
            make_gmem_ptr(A_ptr),
            make_layout(
                make_shape(chunk_size, chunk_size),
                make_stride(chunk_size, _1{})));

        // (N, K) operands: gemm_TTS contracts over the trailing mode, so the
        // transposed views give A^-1 @ Ka and A^-1 @ Vp directly.
        auto Ka_tensor_T = make_tensor(
            make_gmem_ptr(Ka + operand_offset),
            make_layout(
                make_shape(head_dim, chunk_size), make_stride(_1{}, head_dim)));
        auto Vp_tensor_T = make_tensor(
            make_gmem_ptr(Vp + operand_offset),
            make_layout(
                make_shape(head_dim, chunk_size), make_stride(_1{}, head_dim)));

        auto W_tensor = make_tensor(
            make_gmem_ptr(W + operand_offset),
            make_layout(
                make_shape(chunk_size, head_dim), make_stride(head_dim, _1{})));
        auto U_tensor = make_tensor(
            make_gmem_ptr(U + operand_offset),
            make_layout(
                make_shape(chunk_size, head_dim), make_stride(head_dim, _1{})));

        Tensor cW = make_identity_tensor(W_tensor.shape());
        Tensor cU = make_identity_tensor(U_tensor.shape());
        auto copy_W_d = get_block_2d_copy_D<void>(mma, W_tensor);
        auto copy_U_d = get_block_2d_copy_D<void>(mma, U_tensor);
        auto thr_copy_W_d = copy_W_d.get_slice(local_id);
        auto thr_copy_U_d = copy_U_d.get_slice(local_id);

        for (int d = 0; d < head_dim / chunk_size; ++d) {
          Tensor gW_C =
              local_tile(cW, wg_tile, make_coord(0, d, 0), Step<_1, _1, X>{});
          Tensor gU_C =
              local_tile(cU, wg_tile, make_coord(0, d, 0), Step<_1, _1, X>{});
          auto tSrW_d = thr_mma.partition_sg_fragment_C(gW_C);
          auto tSrU_d = thr_mma.partition_sg_fragment_C(gU_C);
          clear(tSrW_d);
          clear(tSrU_d);

          // Both products share the triangular inverse, so it is loaded once
          // per k-tile instead of once per product.
          gemm_TTS_fused_2B(
              A_tensor, Ka_tensor_T, Vp_tensor_T, tSrW_d, tSrU_d, 0, d, d, mma);

          auto tCrW_d = thr_copy_W_d.partition_sg_fragment_S(gW_C);
          auto tCgW_d = thr_copy_W_d.partition_D(gW_C);
          reorder(tSrW_d, tCrW_d);
          copy(copy_W_d, tCrW_d, tCgW_d);

          auto tCrU_d = thr_copy_U_d.partition_sg_fragment_S(gU_C);
          auto tCgU_d = thr_copy_U_d.partition_D(gU_C);
          reorder(tSrU_d, tCrU_d);
          copy(copy_U_d, tCrU_d, tCgU_d);
        }
      }
      chunk_id += global_chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

// ---------------------------------------------------------------------------
// Stage 5: sequential inter-chunk carry.
//
//   U  = U0 - W @ S0^T
//   O  = Qt @ S0^T + tril(Qt @ Kb^T) @ U
//   S  = (S0 + U^T @ Kb) * diag(Tl)
// ---------------------------------------------------------------------------
template <typename T, typename StateT, class TiledMMA>
CUTE_DEVICE void chunk_kda_fwd_o_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,
    T* core_attn_out,
    T* A,  // reused as the [chunk, chunk] attention buffer
    T* W,
    T* U,
    const T* Qt,
    const T* Kb,
    const float* Tl,
    StateT* recurrent_state,
    const int64_t recurrent_state_stride_0,
    const int* query_start_loc,
    const int* state_indices,
    const bool* has_initial_state,
    const int* token_indx,
    const int batch_size,
    const int total_virtual_seqlen,
    const int num_heads,
    const int head_dim,
    const int dv_groups) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int local_id = item.get_local_linear_id();
  const int local_range = item.get_local_range(2);
  const int current_batch_id = item.get_group(0);
  const int head_id = item.get_group(1) / dv_groups;
  // Value-dimension slice owned by this work-group. `dv_groups == 1` restores
  // the original behaviour of walking every slice in one work-group.
  const int dv_group = item.get_group(1) % dv_groups;

  auto sg = item.get_sub_group();
  const int sg_local_id = sg.get_local_linear_id();

  float* Tl_slm = static_cast<float*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(local_id);

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto SG_M = tile_m / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;

  const auto sg_local_m_coord = cutlass::get_sub_group_id() / ATOM_N;
  const auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  const int m_sg_start = sg_local_m_coord * SG_M;
  const int n_sg_start = sg_local_n_coord * SG_N;

  const int num_virtual_chunks = total_virtual_seqlen / chunk_size;

  int pre_chunks = 0;
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start = query_start_loc[batch_id];
    const int seq_len = query_start_loc[batch_id + 1] - seq_start;
    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    if (current_batch_id != batch_id) {
      pre_chunks += current_chunks;
      continue;
    }

    const int slot = state_indices[batch_id];
    if (slot == pad_slot_id) {
      return;
    }
    const bool initial_state =
        has_initial_state == nullptr || has_initial_state[batch_id];

    StateT* state_ptr = recurrent_state +
                        static_cast<int64_t>(slot) * recurrent_state_stride_0 +
                        static_cast<int64_t>(head_id) * head_dim * head_dim;
    auto S_tensor = make_tensor(
        make_gmem_ptr(state_ptr),
        make_layout(
            make_shape(head_dim, head_dim), make_stride(head_dim, _1{})));

    for (int local_chunk = 0; local_chunk < current_chunks; ++local_chunk) {
      const bool has_prev_state = (local_chunk != 0) || initial_state;
      const int chunk_id = pre_chunks + local_chunk;
      const int out_token_start = seq_start + local_chunk * chunk_size;
      const int current_chunk_size =
          sycl::min(chunk_size, seq_len - local_chunk * chunk_size);

      const int64_t operand_offset =
          static_cast<int64_t>(head_id) * total_virtual_seqlen * head_dim +
          static_cast<int64_t>(chunk_id) * chunk_size * head_dim;

      // Ends the previous chunk: its `S` stores have to be visible before this
      // chunk reads `S` back as a GEMM operand, and `Tl_slm` must not be
      // refilled while a sub-group is still reading it.
      sycl::group_barrier(item.get_group());
      for (int j = local_id; j < head_dim; j += local_range) {
        Tl_slm[j] =
            Tl[static_cast<int64_t>(head_id) * num_virtual_chunks * head_dim +
               static_cast<int64_t>(chunk_id) * head_dim + j];
      }
      item.barrier(sycl::access::fence_space::local_space);

      auto W_tensor = make_tensor(
          make_gmem_ptr(W + operand_offset),
          make_layout(
              make_shape(chunk_size, head_dim), make_stride(head_dim, _1{})));
      auto U_tensor = make_tensor(
          make_gmem_ptr(U + operand_offset),
          make_layout(
              make_shape(chunk_size, head_dim), make_stride(head_dim, _1{})));
      auto U_tensor_T = make_tensor(
          make_gmem_ptr(U + operand_offset),
          make_layout(
              make_shape(head_dim, chunk_size), make_stride(_1{}, head_dim)));
      auto Qt_tensor = make_tensor(
          make_gmem_ptr(Qt + operand_offset),
          make_layout(
              make_shape(current_chunk_size, head_dim),
              make_stride(head_dim, _1{})));
      auto Kb_tensor = make_tensor(
          make_gmem_ptr(Kb + operand_offset),
          make_layout(
              make_shape(chunk_size, head_dim), make_stride(head_dim, _1{})));
      auto Kb_tensor_T = make_tensor(
          make_gmem_ptr(Kb + operand_offset),
          make_layout(
              make_shape(head_dim, chunk_size), make_stride(_1{}, head_dim)));

      auto O2_ptr =
          A +
          static_cast<int64_t>(dv_group) * num_heads * total_virtual_seqlen *
              chunk_size +
          static_cast<int64_t>(head_id) * total_virtual_seqlen * chunk_size +
          static_cast<int64_t>(chunk_id) * chunk_size * chunk_size;
      auto O2_tensor = make_tensor(
          make_gmem_ptr(O2_ptr),
          make_layout(
              make_shape(current_chunk_size, chunk_size),
              make_stride(chunk_size, _1{})));

      // `token_indx` maps each row of the chunk to an arbitrary row of
      // `core_attn_out`, so a base pointer plus a fixed row stride only
      // describes the destination when this chunk's rows happen to land
      // consecutively. Probe that here and fall back to a per-row scatter
      // otherwise; the probe reduces over the same indices in every sub-group,
      // so its result is uniform across the work-group.
      const int out_token =
          token_indx == nullptr ? out_token_start : token_indx[out_token_start];
      bool out_contiguous = true;
      if (token_indx != nullptr) {
        bool lane_contiguous = true;
        for (int i = sg_local_id; i < current_chunk_size; i += sub_group_size) {
          lane_contiguous &= token_indx[out_token_start + i] == out_token + i;
        }
        out_contiguous = sycl::all_of_group(sg, lane_contiguous);
      }
      auto O_tensor = make_tensor(
          make_gmem_ptr(
              core_attn_out +
              static_cast<int64_t>(out_token) * num_heads * head_dim +
              static_cast<int64_t>(head_id) * head_dim),
          make_layout(
              make_shape(current_chunk_size, head_dim),
              make_stride(num_heads * head_dim, _1{})));

      Tensor cO2 = make_identity_tensor(O2_tensor.shape());
      Tensor cU = make_identity_tensor(U_tensor.shape());
      Tensor cO = make_identity_tensor(O_tensor.shape());
      Tensor cS = make_identity_tensor(S_tensor.shape());

      auto copy_O2_d = get_block_2d_copy_D<void>(mma, O2_tensor);
      auto thr_copy_O2_d = copy_O2_d.get_slice(local_id);
      auto copy_U_c = get_block_2d_copy_C<void>(mma, U_tensor);
      auto copy_U_d = get_block_2d_copy_D<void>(mma, U_tensor);
      auto thr_copy_U_c = copy_U_c.get_slice(local_id);
      auto thr_copy_U_d = copy_U_d.get_slice(local_id);
      auto copy_O_d = get_block_2d_copy_D<void>(mma, O_tensor);
      auto thr_copy_O_d = copy_O_d.get_slice(local_id);
      auto copy_S_c = get_block_2d_copy_C<void>(mma, S_tensor);
      auto copy_S_d = get_block_2d_copy_D<void>(mma, S_tensor);
      auto thr_copy_S_c = copy_S_c.get_slice(local_id);
      auto thr_copy_S_d = copy_S_d.get_slice(local_id);

      // --- O2 = tril(Qt @ Kb^T); decay and beta are already in the operands.
      {
        Tensor gO2_C =
            local_tile(cO2, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
        auto tCrO2_d = thr_copy_O2_d.partition_sg_fragment_S(gO2_C);
        auto tCgO2_d = thr_copy_O2_d.partition_D(gO2_C);
        auto tSrO2 = thr_mma.partition_sg_fragment_C(gO2_C);
        clear(tSrO2);
        gemm_TTS(Qt_tensor, Kb_tensor, tSrO2, 0, 0, mma);

        CUTE_UNROLL
        for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
          const int n_idx = n_sg_start + sn * sub_group_size + sg_local_id;
          CUTE_UNROLL
          for (int sm = 0; sm < SG_M; ++sm) {
            const int m_idx = m_sg_start + sm;
            if (m_idx < n_idx) {
              tSrO2(sn * SG_M + sm) = 0.0f;
            }
          }
        }
        reorder(tSrO2, tCrO2_d);
        copy(copy_O2_d, tCrO2_d, tCgO2_d);
      }

      // Each sub-group stores only its own block of O2 but reads the whole
      // tile back as a GEMM A operand below, so the store has to be ordered
      // against the loads of every other sub-group. The split barriers inside
      // `gemm_TTS` only pace its k-loop and cannot stand in for that.
      sycl::group_barrier(item.get_group());

      const int num_d_tiles = head_dim / chunk_size;

      // O += tril(Qt @ Kb^T) @ U for one value block, then write it out. `U`
      // must already be published to global memory by the caller.
      auto accumulate_and_store_o = [&](auto& tSrO, auto const& gO_C, int dv) {
        gemm_TTS(O2_tensor, U_tensor_T, tSrO, 0, dv, mma);
        if (out_contiguous) {
          auto tCrO_d = thr_copy_O_d.partition_sg_fragment_S(gO_C);
          auto tCgO_d = thr_copy_O_d.partition_D(gO_C);
          reorder(tSrO, tCrO_d);
          copy(copy_O_d, tCrO_d, tCgO_d);
          return;
        }
        // Permuted destination rows cannot be expressed as a strided tile, so
        // scatter the accumulator one row at a time. Lanes still cover
        // consecutive channels, so each row stays coalesced.
        CUTE_UNROLL
        for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
          const int n_idx =
              dv * chunk_size + n_sg_start + sn * sub_group_size + sg_local_id;
          CUTE_UNROLL
          for (int sm = 0; sm < SG_M; ++sm) {
            const int m_idx = m_sg_start + sm;
            if (m_idx >= current_chunk_size) {
              continue;
            }
            const int64_t out_row = token_indx[out_token_start + m_idx];
            core_attn_out
                [out_row * num_heads * head_dim +
                 static_cast<int64_t>(head_id) * head_dim + n_idx] =
                    static_cast<T>(tSrO(sn * SG_M + sm));
          }
        }
      };

      if (has_prev_state) {
        for (int dv = dv_group; dv < num_d_tiles; dv += dv_groups) {
          Tensor gU_C =
              local_tile(cU, wg_tile, make_coord(0, dv, 0), Step<_1, _1, X>{});
          Tensor gO_C =
              local_tile(cO, wg_tile, make_coord(0, dv, 0), Step<_1, _1, X>{});

          auto tSrWS = thr_mma.partition_sg_fragment_C(gU_C);
          clear(tSrWS);
          auto tSrO = thr_mma.partition_sg_fragment_C(gO_C);
          clear(tSrO);

          // S is loaded once and reused for both W x S^T and Qt x S^T.
          gemm_TTS_fused_2A(
              W_tensor, Qt_tensor, S_tensor, tSrWS, tSrO, 0, 0, dv, mma);

          // U := U0 - W @ S0^T
          auto tCrWS = thr_copy_U_c.partition_sg_fragment_D(gU_C);
          reorder(tSrWS, tCrWS);
          auto tCgU_c = thr_copy_U_c.partition_S(gU_C);
          auto tCrU_c = thr_copy_U_c.partition_sg_fragment_D(gU_C);
          copy(copy_U_c, tCgU_c, tCrU_c);
          CUTE_UNROLL
          for (int i = 0; i < tCrWS.size(); ++i) {
            tCrU_c(i) -= tCrWS(i);
          }
          auto tCrU_d = thr_copy_U_d.partition_sg_fragment_S(gU_C);
          auto tCgU_d = thr_copy_U_d.partition_D(gU_C);
          reorder(tCrU_c, tCrU_d);
          copy(copy_U_d, tCrU_d, tCgU_d);

          // U must be visible to every sub-group before O2 @ U reads it back.
          // It lives in global memory, so the fence has to cover it - a
          // local-space fence would leave the store unordered.
          sycl::group_barrier(item.get_group());

          accumulate_and_store_o(tSrO, gO_C, dv);
        }
      } else {
        for (int dv = dv_group; dv < num_d_tiles; dv += dv_groups) {
          Tensor gO_C =
              local_tile(cO, wg_tile, make_coord(0, dv, 0), Step<_1, _1, X>{});
          auto tSrO = thr_mma.partition_sg_fragment_C(gO_C);
          clear(tSrO);
          accumulate_and_store_o(tSrO, gO_C, dv);
        }
      }

      // --- S := (S0 + U^T @ Kb) * diag(Tl) -------------------------------
      // Separates the reads of the carried `S` above from the stores below,
      // and publishes the final `U` to the sub-groups that read other blocks
      // of it. Both operands are in global memory.
      sycl::group_barrier(item.get_group());
      // Load the carried state into the accumulator (or start from zero on the
      // first chunk of a fresh sequence).
      auto init_state = [&](auto& tSrS, auto const& gS_C) {
        if (has_prev_state) {
          auto tCgS_c = thr_copy_S_c.partition_S(gS_C);
          auto tCrS_c = thr_copy_S_c.partition_sg_fragment_D(gS_C);
          copy(copy_S_c, tCgS_c, tCrS_c);
          reorder(tCrS_c, tSrS);
        } else {
          clear(tSrS);
        }
      };
      // The end-of-chunk decay is per key channel, i.e. per output column.
      auto scale_and_store = [&](auto& tSrS, auto const& gS_C, int dk) {
        CUTE_UNROLL
        for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
          const int n_idx =
              dk * chunk_size + n_sg_start + sn * sub_group_size + sg_local_id;
          const float scale = Tl_slm[n_idx];
          CUTE_UNROLL
          for (int sm = 0; sm < SG_M; ++sm) {
            tSrS(sn * SG_M + sm) *= scale;
          }
        }
        auto tCrS_d = thr_copy_S_d.partition_sg_fragment_S(gS_C);
        auto tCgS_d = thr_copy_S_d.partition_D(gS_C);
        reorder(tSrS, tCrS_d);
        copy(copy_S_d, tCrS_d, tCgS_d);
      };

      for (int dv = dv_group; dv < num_d_tiles; dv += dv_groups) {
        int dk = 0;
        // Both key blocks contract against the same U^T tile, so pairing them
        // loads it once and runs one k-loop instead of two.
        for (; dk + 1 < num_d_tiles; dk += 2) {
          Tensor gS_C0 =
              local_tile(cS, wg_tile, make_coord(dv, dk, 0), Step<_1, _1, X>{});
          Tensor gS_C1 = local_tile(
              cS, wg_tile, make_coord(dv, dk + 1, 0), Step<_1, _1, X>{});
          auto tSrS0 = thr_mma.partition_sg_fragment_C(gS_C0);
          auto tSrS1 = thr_mma.partition_sg_fragment_C(gS_C1);
          init_state(tSrS0, gS_C0);
          init_state(tSrS1, gS_C1);

          gemm_TTS_fused_2B(
              U_tensor_T,
              Kb_tensor_T,
              Kb_tensor_T,
              tSrS0,
              tSrS1,
              dv,
              dk,
              dk + 1,
              mma);

          scale_and_store(tSrS0, gS_C0, dk);
          scale_and_store(tSrS1, gS_C1, dk + 1);
        }
        for (; dk < num_d_tiles; ++dk) {
          Tensor gS_C =
              local_tile(cS, wg_tile, make_coord(dv, dk, 0), Step<_1, _1, X>{});
          auto tSrS = thr_mma.partition_sg_fragment_C(gS_C);
          init_state(tSrS, gS_C);
          gemm_TTS(U_tensor_T, Kb_tensor_T, tSrS, dv, dk, mma);
          scale_and_store(tSrS, gS_C, dk);
        }
      }
    }
    return;
  }
}

}  // namespace kda_xe2
