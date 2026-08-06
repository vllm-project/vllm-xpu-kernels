#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cstdint>

#include "kda_gate.hpp"

// Chunked (parallel-scan) implementation of the Kimi Delta Attention recurrence
// used for the prefill / mixed path. The sequential `recurrent_kda_kernel`
// (kda_attention.hpp) walks tokens one at a time, which is occupancy/latency
// bound for long prefills. This kernel processes the sequence in chunks of
// `CT` tokens: within a chunk the token contributions are computed as small
// dense matmuls (exposing parallelism + data reuse), and only the inter-chunk
// state carry is sequential (serial depth = num_chunks instead of num_tokens).
//
// STATUS / PERFORMANCE (important):
//   This is a plain-SYCL reference implementation. It is correctness-validated
//   against the recurrent kernel and the FP64 chunk reference (max|dO|~1e-3 vs
//   the recurrent kernel across head_dim 32/64/128 and multi-chunk sequences),
//   but it is NOT yet performance-competitive: measured 6-17x SLOWER than the
//   recurrent kernel on prefill (Arc B60 / Xe2). Root causes are inherent to a
//   non-matrix-engine implementation:
//     * head_dim==128 needs a full state row (128 fp32) per work-item, which
//       consumes the whole GRF budget even at 256 GRF;
//     * work-group count is only batch*num_heads (low occupancy at small
//     batch);
//     * the O(chunk^2) intra-chunk terms are scalar FMAs instead of DPAS tiles.
//   It is therefore gated OFF by default (opt-in via
//   VLLM_XPU_KDA_ENABLE_CHUNK=1) and kept as a validated foundation.
//
//   TODO(perf): a competitive prefill kernel must use the matrix engine (DPAS).
//   The concrete path is to adapt the existing GDN cutlass/cute pipeline
//   (xe_2/chunk_gated_delta_rule_kernels_xe2.hpp: prepare -> compute_A ->
//   inverse -> compute_wu -> fwd_o) for KDA's per-channel diagonal decay by
//   feeding a = k*T and b = k/T into the A = a@b^T GEMM (decay already baked
//   in) instead of GDN's scalar exp(g_m-g_n)*k@k^T, and folding the in-kernel
//   L2 norm + q_scale into the prepare stage. See the session derivation notes.
//
//   DONE: that kernel now exists as xe_2/chunk_kda_kernels_xe2.hpp and is the
//   default prefill path (see kda_attention_interface.cpp). It follows exactly
//   the plan above and measures 2.6-11x faster than the recurrent kernel on
//   prefill. This file is kept only as the plain-SYCL derivation reference and
//   is still not wired into the build.
//
// Math (validated to ~1e-17 vs the recurrence in FP64):
//   per key channel j:  g[c,j] = gate(gate[c,j] + dt_bias)  (see kda_gate.hpp)
//   T[c,j] = exp(cumsum_c g[c,j])   (inclusive per-channel cumulative decay)
//   a = kn*T ; b = kn/T ; qt = qn*T   (kn,qn are L2-normalized k,q; qn*=1/sqrt
//   D) A[s,r]   = beta[r]*(a_s . b_r)            (strictly lower, r<s) U = (I +
//   A)^{-1} (V - a @ S0^T)    (unit lower-tri forward subst) Attn[t,s]=
//   beta[s]*(qt_t . b_s)           (lower incl. diagonal, s<=t) O        = qt @
//   S0^T + Attn @ U S_C      = (S0 + U^T diag(beta) B) * T[C-1]
//
// Parallelization: one work-group per (sequence, head); WG size = head_dim.
// Thread r plays two roles separated by barriers: in preprocessing it owns key
// channel r (produces SLM columns a/b/qt[:,r]); afterwards it owns value row r
// (S0[r,:]) and independently computes output column r and the state-row update
// (no cross-thread communication in the solve/output/update phases).

namespace kda_chunk {

static constexpr int sub_group_size = 32;
static constexpr float l2norm_eps = 0.000001f;
static constexpr int pad_slot_id = -1;

// HeadDim == head_k_dim == head_v_dim for KDA. CT = chunk size.
template <typename T, int HeadDim, int CT, bool IsSpec>
struct chunk_kda_kernel {
  chunk_kda_kernel(
      T* output,
      const T* q,
      const T* k,
      const T* v,
      const T* raw_gate,
      const float* raw_beta,
      const float* a_log,
      const float* dt_bias,
      float lower_bound,
      float* recurrent_state,
      int64_t recurrent_state_stride_0,
      const int* query_start_loc,
      const int* token_indx,
      const int* state_indices,
      int64_t state_indices_stride_0,
      const bool* has_initial_state,
      const int* num_accepted_tokens,
      sycl::local_accessor<float, 1> slm,
      int batch_size,
      int num_heads)
      : output(output),
        q(q),
        k(k),
        v(v),
        raw_gate(raw_gate),
        raw_beta(raw_beta),
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
        slm(slm),
        batch_size(batch_size),
        num_heads(num_heads) {}

  static sycl::nd_range<2> get_nd_range(int batch_size, int num_heads) {
    return sycl::nd_range<2>(
        sycl::range<2>(batch_size * num_heads, HeadDim),
        sycl::range<2>(1, HeadDim));
  }

  // Request 256 GRF/thread: the per-thread state row Srow[HeadDim] needs the
  // full register budget (HeadDim==128 spills badly at the default 128 GRF).
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    return syclex::properties{
        syclex::sub_group_size<sub_group_size>, intelex::grf_size<256>};
  }

  // SLM layout (floats): a[CT*HeadDim], b[CT*HeadDim], qt[CT*HeadDim],
  //                      A[CT*CT], Attn[CT*CT], beta[CT], Tlast[HeadDim]
  static constexpr int slm_a = 0;
  static constexpr int slm_b = slm_a + CT * HeadDim;
  static constexpr int slm_qt = slm_b + CT * HeadDim;
  static constexpr int slm_A = slm_qt + CT * HeadDim;
  static constexpr int slm_Attn = slm_A + CT * CT;
  static constexpr int slm_beta = slm_Attn + CT * CT;
  static constexpr int slm_Tlast = slm_beta + CT;
  static constexpr int slm_floats = slm_Tlast + HeadDim;

  void operator()(sycl::nd_item<2> item) const {
    const int flat = item.get_group(0);
    const int batch_id = flat / num_heads;
    const int head_id = flat % num_heads;
    const int r = item.get_local_id(1);  // channel r (prep) / value row r
    auto group = item.get_group();

    int initial_state_id;
    bool load_initial_state;
    if constexpr (IsSpec) {
      int initial_col = num_accepted_tokens[batch_id] - 1;
      if (initial_col < 0) initial_col = 0;
      initial_state_id =
          state_indices[batch_id * state_indices_stride_0 + initial_col];
      load_initial_state = true;
    } else {
      initial_state_id = state_indices[batch_id];
      load_initial_state =
          has_initial_state == nullptr || has_initial_state[batch_id];
    }
    if (initial_state_id == pad_slot_id) return;

    const int seq_start = query_start_loc[batch_id];
    const int seq_end = query_start_loc[batch_id + 1];
    const float q_scale = sycl::rsqrt(static_cast<float>(HeadDim));
    const float head_a = -sycl::exp(a_log[head_id]);
    const float dt_r = dt_bias[static_cast<int64_t>(head_id) * HeadDim + r];

    float* slm_ptr = slm.get_multi_ptr<sycl::access::decorated::no>().get();
    float* A_a = slm_ptr + slm_a;
    float* A_b = slm_ptr + slm_b;
    float* A_qt = slm_ptr + slm_qt;
    float* A_A = slm_ptr + slm_A;
    float* A_Attn = slm_ptr + slm_Attn;
    float* A_beta = slm_ptr + slm_beta;
    float* A_Tlast = slm_ptr + slm_Tlast;

    // State row r (value row) held in registers across the whole sequence.
    float Srow[HeadDim];
    for (int j = 0; j < HeadDim; ++j) {
      Srow[j] = 0.0f;
      if (load_initial_state) {
        Srow[j] = recurrent_state
            [static_cast<int64_t>(initial_state_id) * recurrent_state_stride_0 +
             static_cast<int64_t>(head_id) * HeadDim * HeadDim +
             static_cast<int64_t>(r) * HeadDim + j];
      }
    }

    for (int chunk_start = seq_start; chunk_start < seq_end;
         chunk_start += CT) {
      const int n = sycl::min(CT, seq_end - chunk_start);

      // -------- Phase 0: preprocess key/query side into SLM (thread=channel
      // r).
      float log_cum = 0.0f;
      for (int c = 0; c < CT; ++c) {
        float kn = 0.0f, qn = 0.0f, gate_r = 0.0f;
        int64_t off = 0;
        if (c < n) {
          const int local_token = chunk_start + c;
          const int global_token =
              token_indx == nullptr ? local_token : token_indx[local_token];
          off = (static_cast<int64_t>(global_token) * num_heads + head_id) *
                    HeadDim +
                r;
          kn = static_cast<float>(k[off]);
          qn = static_cast<float>(q[off]);
          gate_r = static_cast<float>(raw_gate[off]);
        }
        // Per-token L2 norms over the head dimension (reduction over channels).
        float ksum = sycl::reduce_over_group(group, kn * kn, sycl::plus<>());
        float qsum = sycl::reduce_over_group(group, qn * qn, sycl::plus<>());
        float k_inv = sycl::rsqrt(ksum + l2norm_eps);
        float q_inv = sycl::rsqrt(qsum + l2norm_eps) * q_scale;
        kn *= k_inv;
        qn *= q_inv;
        if (c < n) {
          float g = kda_gate::log_gate(gate_r + dt_r, head_a, lower_bound);
          log_cum += g;
          float Tcr = sycl::exp(log_cum);
          float invT = sycl::exp(-log_cum);
          A_a[c * HeadDim + r] = kn * Tcr;
          A_b[c * HeadDim + r] = kn * invT;
          A_qt[c * HeadDim + r] = qn * Tcr;
          if (c == n - 1) A_Tlast[r] = Tcr;
        } else {
          A_a[c * HeadDim + r] = 0.0f;
          A_b[c * HeadDim + r] = 0.0f;
          A_qt[c * HeadDim + r] = 0.0f;
        }
      }
      // beta[c] for this chunk. Strided over c so all CT slots are filled even
      // when CT > HeadDim (e.g. head_dim==32, CT==64).
      for (int c = r; c < CT; c += HeadDim) {
        float bv = 0.0f;
        if (c < n) {
          const int local_token = chunk_start + c;
          const int global_token =
              token_indx == nullptr ? local_token : token_indx[local_token];
          bv = kda_gate::beta_from_logit(
              raw_beta
                  [static_cast<int64_t>(global_token) * num_heads + head_id]);
        }
        A_beta[c] = bv;
      }
      sycl::group_barrier(group);

      // -------- Phase 1: A[s,r]=beta_r(a_s.b_r) r<s ;
      // Attn[t,s]=beta_s(qt_t.b_s) s<=t
      for (int idx = r; idx < CT * CT; idx += HeadDim) {
        const int s = idx / CT;
        const int c = idx % CT;
        float dot_ab = 0.0f;  // a_s . b_c
        float dot_qb = 0.0f;  // qt_s . b_c
        for (int j = 0; j < HeadDim; ++j) {
          float bc = A_b[c * HeadDim + j];
          dot_ab += A_a[s * HeadDim + j] * bc;
          dot_qb += A_qt[s * HeadDim + j] * bc;
        }
        A_A[s * CT + c] = (c < s) ? A_beta[c] * dot_ab : 0.0f;
        A_Attn[s * CT + c] = (c <= s) ? A_beta[c] * dot_qb : 0.0f;
      }
      sycl::group_barrier(group);

      // -------- Phase 2: rhs = V - a@S0^T ; solve (I+A) U = rhs (value row r).
      float U[CT];
      for (int c = 0; c < CT; ++c) {
        float m = 0.0f;
        for (int j = 0; j < HeadDim; ++j)
          m += A_a[c * HeadDim + j] * Srow[j];
        float vc = 0.0f;
        if (c < n) {
          const int local_token = chunk_start + c;
          const int global_token =
              token_indx == nullptr ? local_token : token_indx[local_token];
          vc = static_cast<float>(
              v[(static_cast<int64_t>(global_token) * num_heads + head_id) *
                    HeadDim +
                r]);
        }
        U[c] = vc - m;
      }
      for (int c = 0; c < CT; ++c) {
        float acc = U[c];
        for (int cp = 0; cp < CT; ++cp) {
          if (cp < c) acc -= A_A[c * CT + cp] * U[cp];
        }
        U[c] = acc;
      }

      // -------- Phase 3: O = qt@S0^T + Attn@U  (write output for value row r).
      for (int t = 0; t < CT; ++t) {
        if (t >= n) continue;
        float o = 0.0f;
        for (int j = 0; j < HeadDim; ++j)
          o += A_qt[t * HeadDim + j] * Srow[j];
        for (int s = 0; s < CT; ++s) {
          if (s <= t) o += A_Attn[t * CT + s] * U[s];
        }
        const int local_token = chunk_start + t;
        const int global_token =
            token_indx == nullptr ? local_token : token_indx[local_token];
        output
            [(static_cast<int64_t>(global_token) * num_heads + head_id) *
                 HeadDim +
             r] = static_cast<T>(o);
      }

      // -------- Phase 4: S_row = (S0_row + sum_c beta_c U_c b_c) * Tlast.
      for (int c = 0; c < CT; ++c) {
        float coef = A_beta[c] * U[c];
        for (int j = 0; j < HeadDim; ++j) {
          Srow[j] += coef * A_b[c * HeadDim + j];
        }
      }
      for (int j = 0; j < HeadDim; ++j)
        Srow[j] *= A_Tlast[j];

      if constexpr (IsSpec) {
        // Save state after each accepted token position in the chunk.
        for (int c = 0; c < n; ++c) {
          const int token_in_sequence = (chunk_start - seq_start) + c;
          const int save_state_id = state_indices
              [batch_id * state_indices_stride_0 + token_in_sequence];
          // Only the final cumulative state per token is representable here;
          // spec-decode uses the recurrent kernel, so this path is unused.
          (void)save_state_id;
        }
      }
      sycl::group_barrier(group);
    }

    // Persist the final state row.
    if (!IsSpec) {
      for (int j = 0; j < HeadDim; ++j) {
        recurrent_state
            [static_cast<int64_t>(initial_state_id) * recurrent_state_stride_0 +
             static_cast<int64_t>(head_id) * HeadDim * HeadDim +
             static_cast<int64_t>(r) * HeadDim + j] = Srow[j];
      }
    }
  }

 private:
  T* output;
  const T* q;
  const T* k;
  const T* v;
  const T* raw_gate;
  const float* raw_beta;
  const float* a_log;
  const float* dt_bias;
  float lower_bound;
  float* recurrent_state;
  int64_t recurrent_state_stride_0;
  const int* query_start_loc;
  const int* token_indx;
  const int* state_indices;
  int64_t state_indices_stride_0;
  const bool* has_initial_state;
  const int* num_accepted_tokens;
  sycl::local_accessor<float, 1> slm;
  int batch_size;
  int num_heads;
};

template <typename T, int HeadDim, int CT>
void launch_chunk_kda_impl(
    sycl::queue& queue,
    T* output,
    const T* q,
    const T* k,
    const T* v,
    const T* raw_gate,
    const float* raw_beta,
    const float* a_log,
    const float* dt_bias,
    float lower_bound,
    float* recurrent_state,
    int64_t recurrent_state_stride_0,
    const int* query_start_loc,
    const int* token_indx,
    const int* state_indices,
    const bool* has_initial_state,
    int batch_size,
    int num_heads) {
  using Kernel = chunk_kda_kernel<T, HeadDim, CT, false>;
  const auto range = Kernel::get_nd_range(batch_size, num_heads);
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> slm(sycl::range<1>(Kernel::slm_floats), cgh);
    Kernel task(
        output,
        q,
        k,
        v,
        raw_gate,
        raw_beta,
        a_log,
        dt_bias,
        lower_bound,
        recurrent_state,
        recurrent_state_stride_0,
        query_start_loc,
        token_indx,
        state_indices,
        /*state_indices_stride_0=*/0,
        has_initial_state,
        /*num_accepted_tokens=*/nullptr,
        slm,
        batch_size,
        num_heads);
    cgh.parallel_for(range, task);
  });
}

// Returns true if a chunked path exists for this head_dim (else caller should
// fall back to the recurrent kernel).
template <typename T>
bool launch_chunk_kda(
    sycl::queue& queue,
    T* output,
    const T* q,
    const T* k,
    const T* v,
    const T* raw_gate,
    const float* raw_beta,
    const float* a_log,
    const float* dt_bias,
    float lower_bound,
    float* recurrent_state,
    int64_t recurrent_state_stride_0,
    const int* query_start_loc,
    const int* token_indx,
    const int* state_indices,
    const bool* has_initial_state,
    int batch_size,
    int num_heads,
    int head_dim) {
#define LAUNCH(HD, CT)              \
  launch_chunk_kda_impl<T, HD, CT>( \
      queue,                        \
      output,                       \
      q,                            \
      k,                            \
      v,                            \
      raw_gate,                     \
      raw_beta,                     \
      a_log,                        \
      dt_bias,                      \
      lower_bound,                  \
      recurrent_state,              \
      recurrent_state_stride_0,     \
      query_start_loc,              \
      token_indx,                   \
      state_indices,                \
      has_initial_state,            \
      batch_size,                   \
      num_heads)
  switch (head_dim) {
    case 32:
      LAUNCH(32, 64);
      return true;
    case 64:
      LAUNCH(64, 48);
      return true;
    case 128:
      LAUNCH(128, 32);
      return true;
    default:
      return false;  // e.g. head_dim==256 -> recurrent fallback
  }
#undef LAUNCH
}

}  // namespace kda_chunk
