#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

namespace gdn {
static constexpr int sub_group_size = 32;
template <typename T, typename StateT, int k_bucket_size>
struct gated_delta_rule_kernel {
 public:
  static constexpr int group_size = 256;
  static constexpr int sg_per_group = group_size / sub_group_size;
  static constexpr int v_dim_per_sg = 4;
  static constexpr int v_dim_per_group = v_dim_per_sg * sg_per_group;
  static constexpr float eps = 0.000001;

  gated_delta_rule_kernel(
      T* core_attn_out,
      const T* q,
      const T* k,
      const T* v,
      const T* b,
      const T* a,
      const float* A_log,
      const T* dt_bias,
      StateT* ssm_state,
      const int ssm_state_stride_0,
      const int* query_start_loc,
      const int* token_indx,
      const int* cache_indices,
      const bool* has_initial_state,
      const int* num_accepted_tokens,
      const int batch_size,
      const int total_seqlen,
      const int num_k_heads,
      const int head_k_dim,
      const int num_v_heads,
      const int head_v_dim)
      : core_attn_out(core_attn_out),
        q(q),
        k(k),
        v(v),
        b(b),
        a(a),
        A_log(A_log),
        dt_bias(dt_bias),
        ssm_state(ssm_state),
        ssm_state_stride_0(ssm_state_stride_0),
        query_start_loc(query_start_loc),
        token_indx(token_indx),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
        num_accepted_tokens(num_accepted_tokens),
        batch_size(batch_size),
        total_seqlen(total_seqlen),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int batch_size, const int num_v_heads, const int head_v_dim) {
    int num_v_bucket = (head_v_dim + v_dim_per_group - 1) / v_dim_per_group;
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(batch_size, num_v_heads, num_v_bucket);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline float act_sigmoid(float& x) {
    return 1.0f / (1.0f + sycl::exp(-x));
  }

  static inline float
  act_softplus(float& x, float beta = 1.0f, float threshold = 20.0f) {
    if (beta * x < threshold) {
      return sycl::log(1.0f + sycl::exp(beta * x)) / beta;
    } else
      return x;
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    int batch_id = item.get_group(0);
    int num_v_heads_id = item.get_group(1);
    int v_bucket_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    // assume num_v_heads is always bigger than num_k_heads
    int kv_ratio = num_v_heads / num_k_heads;
    int head_v_dim_id = v_bucket_id * v_dim_per_group + sg_id * v_dim_per_sg;

    if (head_v_dim_id >= head_v_dim) {
      return;
    }

    const float scale = 1.0f / sycl::sqrt(float(head_k_dim));
    float A_log_local = A_log[num_v_heads_id];
    float dt_bias_local = dt_bias[num_v_heads_id];
    A_log_local = -sycl::exp(A_log_local);

    float state_local[v_dim_per_sg * k_bucket_size];
    float q_local[k_bucket_size];
    float k_local[k_bucket_size];
    float v_local[v_dim_per_sg];

    StateT* ssm_state_ptr =
        ssm_state +
        static_cast<int64_t>(cache_indices[batch_id]) * ssm_state_stride_0;

    // load state
    if (has_initial_state == nullptr || has_initial_state[batch_id]) {
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] =
              static_cast<float>(ssm_state_ptr
                                     [num_v_heads_id * head_k_dim * head_v_dim +
                                      (k_bucket_size * sg_local_id + i) +
                                      (head_v_dim_id + j) * head_k_dim]);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
#pragma unroll
        for (int j = 0; j < v_dim_per_sg; ++j) {
          state_local[i * v_dim_per_sg + j] = 0.0f;
        }
      }
    }

    int seq_start_offset = query_start_loc[batch_id];
    int seq_end_offset = query_start_loc[batch_id + 1];

    // The state of each token is calculated iteratively.
    for (int t = seq_start_offset; t < seq_end_offset; ++t) {
      // act beta(t), g(t)
      float b_local = b[t * num_v_heads + num_v_heads_id];
      float beta = act_sigmoid(b_local);
      float a_local = a[t * num_v_heads + num_v_heads_id] + dt_bias_local;
      float g = sycl::exp(A_log_local * act_softplus(a_local));

      float q_sum = 0.0f;
      float k_sum = 0.0f;
// load q(t), k(t) and l2norm
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        q_local[i] =
            q[t * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        k_local[i] =
            k[t * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        q_sum += q_local[i] * q_local[i];
        k_sum += k_local[i] * k_local[i];
      }
      q_sum = sycl::reduce_over_group(sg, q_sum, sycl::plus<>());
      k_sum = sycl::reduce_over_group(sg, k_sum, sycl::plus<>());
      q_sum += eps;
      k_sum += eps;
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        q_local[i] /= sycl::sqrt(q_sum);
        q_local[i] *= scale;
        k_local[i] /= sycl::sqrt(k_sum);
      }

      float kv_mem[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        kv_mem[i] = 0.0f;
      }

#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] *= g;
          kv_mem[j] += state_local[j * k_bucket_size + i] * k_local[i];
        }
      }
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        kv_mem[i] = sycl::reduce_over_group(sg, kv_mem[i], sycl::plus<>());
      }

#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        v_local[i] =
            v[t * num_v_heads * head_v_dim + num_v_heads_id * head_v_dim +
              head_v_dim_id + i];
      }
      float delta[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        delta[i] = (v_local[i] - kv_mem[i]) * beta;
      }

      float res[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        res[i] = 0.0f;
      }
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          // get S(t)
          state_local[j * k_bucket_size + i] += k_local[i] * delta[j];
          // get O(t)
          res[j] += state_local[j * k_bucket_size + i] * q_local[i];
        }
      }
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        res[i] = sycl::reduce_over_group(sg, res[i], sycl::plus<>());
      }

      // store O(t) -- core_attn_out is the GLOBAL active buffer; remap
      // local token id `t` through token_indx if it was provided.
      if (sg_local_id == 0) {
        const int global_t = (token_indx != nullptr) ? token_indx[t] : t;
#pragma unroll
        for (int i = 0; i < v_dim_per_sg; ++i) {
          core_attn_out
              [global_t * num_v_heads * head_v_dim +
               num_v_heads_id * head_v_dim + head_v_dim_id + i] = res[i];
        }
      }
    }

// update state
#pragma unroll
    for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        ssm_state_ptr
            [num_v_heads_id * head_k_dim * head_v_dim +
             (k_bucket_size * sg_local_id + i) +
             (head_v_dim_id + j) * head_k_dim] =
                static_cast<StateT>(state_local[j * k_bucket_size + i]);
      }
    }
  }

 private:
  T* core_attn_out;
  const T* q;
  const T* k;
  const T* v;
  const T* b;
  const T* a;
  const float* A_log;
  const T* dt_bias;
  StateT* ssm_state;
  const int ssm_state_stride_0;
  const int* query_start_loc;
  const int* token_indx;
  const int* cache_indices;
  const bool* has_initial_state;
  const int* num_accepted_tokens;
  const int batch_size;
  const int total_seqlen;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
};

// -----------------------------------------------------------------------------
// Spec-decoding kernel: every spec sequence contributes exactly
// `num_spec_tokens` (== num_speculative_tokens + 1) tokens, laid out
// contiguously inside q/k/v/b/a (which are LOCAL buffers of size
// `num_spec_decodes * num_spec_tokens`).
//
// - The initial SSM state for sequence n is read from cache slot
//   `cache_indices[n, num_accepted_tokens[n] - 1]` (the last accepted token of
//   the previous step). When num_accepted_tokens[n] is 0 we fall back to col 0
//   (matches the rejection-sampler invariant that the bonus token is always
//   accepted, so this branch is defensive only).
// - After processing every local token t, the new SSM state is written into
//   `cache_indices[n, t]`, mirroring the Triton kernel's per-step writeback
//   used to support multiple rollback positions on the next iteration.
// - `core_attn_out` is the GLOBAL active buffer; we route writes through
//   `token_indx`, which maps a local token index back to its global position.
// -----------------------------------------------------------------------------
template <typename T, typename StateT, int k_bucket_size>
struct gated_delta_rule_spec_kernel {
 public:
  static constexpr int group_size = 256;
  static constexpr int sg_per_group = group_size / sub_group_size;
  static constexpr int v_dim_per_sg = 4;
  static constexpr int v_dim_per_group = v_dim_per_sg * sg_per_group;
  static constexpr float eps = 0.000001;

  gated_delta_rule_spec_kernel(
      T* core_attn_out,
      const T* q,
      const T* k,
      const T* v,
      const T* b,
      const T* a,
      const float* A_log,
      const T* dt_bias,
      StateT* ssm_state,
      const int ssm_state_stride_0,
      const int* token_indx,
      const int* cache_indices,
      const int cache_indices_stride_0,
      const int* num_accepted_tokens,
      const int num_spec_decodes,
      const int num_spec_tokens,
      const int num_k_heads,
      const int head_k_dim,
      const int num_v_heads,
      const int head_v_dim)
      : core_attn_out(core_attn_out),
        q(q),
        k(k),
        v(v),
        b(b),
        a(a),
        A_log(A_log),
        dt_bias(dt_bias),
        ssm_state(ssm_state),
        ssm_state_stride_0(ssm_state_stride_0),
        token_indx(token_indx),
        cache_indices(cache_indices),
        cache_indices_stride_0(cache_indices_stride_0),
        num_accepted_tokens(num_accepted_tokens),
        num_spec_decodes(num_spec_decodes),
        num_spec_tokens(num_spec_tokens),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int num_spec_decodes, const int num_v_heads, const int head_v_dim) {
    int num_v_bucket = (head_v_dim + v_dim_per_group - 1) / v_dim_per_group;
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(num_spec_decodes, num_v_heads, num_v_bucket);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline float act_sigmoid(float& x) {
    return 1.0f / (1.0f + sycl::exp(-x));
  }

  static inline float
  act_softplus(float& x, float beta = 1.0f, float threshold = 20.0f) {
    if (beta * x < threshold) {
      return sycl::log(1.0f + sycl::exp(beta * x)) / beta;
    } else
      return x;
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    int batch_id = item.get_group(0);
    int num_v_heads_id = item.get_group(1);
    int v_bucket_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    int kv_ratio = num_v_heads / num_k_heads;
    int head_v_dim_id = v_bucket_id * v_dim_per_group + sg_id * v_dim_per_sg;
    if (head_v_dim_id >= head_v_dim) {
      return;
    }

    const float scale = 1.0f / sycl::sqrt(float(head_k_dim));
    float A_log_local = A_log[num_v_heads_id];
    float dt_bias_local = dt_bias[num_v_heads_id];
    A_log_local = -sycl::exp(A_log_local);

    float state_local[v_dim_per_sg * k_bucket_size];
    float q_local[k_bucket_size];
    float k_local[k_bucket_size];
    float v_local[v_dim_per_sg];

    // -- Load initial state ---------------------------------------------------
    // init_col = num_accepted_tokens[batch_id] - 1 (clamped to >= 0).
    int init_col = num_accepted_tokens[batch_id] - 1;
    if (init_col < 0) init_col = 0;
    const int init_state_idx =
        cache_indices[batch_id * cache_indices_stride_0 + init_col];
    const StateT* init_state_ptr =
        ssm_state + static_cast<int64_t>(init_state_idx) * ssm_state_stride_0;
#pragma unroll
    for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        state_local[j * k_bucket_size + i] =
            static_cast<float>(init_state_ptr
                                   [num_v_heads_id * head_k_dim * head_v_dim +
                                    (k_bucket_size * sg_local_id + i) +
                                    (head_v_dim_id + j) * head_k_dim]);
      }
    }

    // -- Iterate over the num_spec_tokens tokens of this sequence -------------
    for (int t_local = 0; t_local < num_spec_tokens; ++t_local) {
      // Local token index inside q/k/v/b/a (which were sized
      // num_spec_decodes * num_spec_tokens and ordered by
      // spec_query_start_loc).
      const int t = batch_id * num_spec_tokens + t_local;

      float b_local = b[t * num_v_heads + num_v_heads_id];
      float beta = act_sigmoid(b_local);
      float a_local = a[t * num_v_heads + num_v_heads_id] + dt_bias_local;
      float g = sycl::exp(A_log_local * act_softplus(a_local));

      float q_sum = 0.0f;
      float k_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        q_local[i] =
            q[t * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        k_local[i] =
            k[t * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        q_sum += q_local[i] * q_local[i];
        k_sum += k_local[i] * k_local[i];
      }
      q_sum = sycl::reduce_over_group(sg, q_sum, sycl::plus<>());
      k_sum = sycl::reduce_over_group(sg, k_sum, sycl::plus<>());
      q_sum += eps;
      k_sum += eps;
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        q_local[i] /= sycl::sqrt(q_sum);
        q_local[i] *= scale;
        k_local[i] /= sycl::sqrt(k_sum);
      }

      float kv_mem[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        kv_mem[i] = 0.0f;
      }
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] *= g;
          kv_mem[j] += state_local[j * k_bucket_size + i] * k_local[i];
        }
      }
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        kv_mem[i] = sycl::reduce_over_group(sg, kv_mem[i], sycl::plus<>());
      }

#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        v_local[i] =
            v[t * num_v_heads * head_v_dim + num_v_heads_id * head_v_dim +
              head_v_dim_id + i];
      }
      float delta[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        delta[i] = (v_local[i] - kv_mem[i]) * beta;
      }

      float res[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        res[i] = 0.0f;
      }
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] += k_local[i] * delta[j];
          res[j] += state_local[j * k_bucket_size + i] * q_local[i];
        }
      }
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        res[i] = sycl::reduce_over_group(sg, res[i], sycl::plus<>());
      }

      // Write O(t) to GLOBAL core_attn_out via token_indx remap.
      if (sg_local_id == 0) {
        const int global_t = (token_indx != nullptr) ? token_indx[t] : t;
#pragma unroll
        for (int i = 0; i < v_dim_per_sg; ++i) {
          core_attn_out
              [global_t * num_v_heads * head_v_dim +
               num_v_heads_id * head_v_dim + head_v_dim_id + i] = res[i];
        }
      }

      // Write the new SSM state back to this token's dedicated cache slot,
      // so that the next forward can pick the right rollback column based on
      // num_accepted_tokens.
      const int final_state_idx =
          cache_indices[batch_id * cache_indices_stride_0 + t_local];
      StateT* final_state_ptr =
          ssm_state +
          static_cast<int64_t>(final_state_idx) * ssm_state_stride_0;
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          final_state_ptr
              [num_v_heads_id * head_k_dim * head_v_dim +
               (k_bucket_size * sg_local_id + i) +
               (head_v_dim_id + j) * head_k_dim] =
                  static_cast<StateT>(state_local[j * k_bucket_size + i]);
        }
      }
    }
  }

 private:
  T* core_attn_out;
  const T* q;
  const T* k;
  const T* v;
  const T* b;
  const T* a;
  const float* A_log;
  const T* dt_bias;
  StateT* ssm_state;
  const int ssm_state_stride_0;
  const int* token_indx;
  const int* cache_indices;
  const int cache_indices_stride_0;
  const int* num_accepted_tokens;
  const int num_spec_decodes;
  const int num_spec_tokens;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
};

template <typename T, typename StateT, int k_bucket_size>
void kernel_launcher(
    sycl::queue& queue,
    T* core_attn_out,
    const T* q,
    const T* k,
    const T* v,
    const T* b,
    const T* a,
    const float* A_log,
    const T* dt_bias,
    StateT* ssm_state,
    const int ssm_state_stride_0,
    const int* query_start_loc,
    const int* token_indx,
    const int* cache_indices,
    const bool* has_initial_state,
    const int* num_accepted_tokens,
    const int batch_size,
    const int total_seqlen,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  using KERNEL = gated_delta_rule_kernel<T, StateT, k_bucket_size>;
  auto range = KERNEL::get_nd_range(batch_size, num_v_heads, head_v_dim);
  assert(head_v_dim % KERNEL::v_dim_per_group == 0);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL task(
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_stride_0,
        query_start_loc,
        token_indx,
        cache_indices,
        has_initial_state,
        num_accepted_tokens,
        batch_size,
        total_seqlen,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim);
    cgh.parallel_for(range, task);
  });
}

template <typename T, typename StateT, int k_bucket_size>
void kernel_launcher_spec(
    sycl::queue& queue,
    T* core_attn_out,
    const T* q,
    const T* k,
    const T* v,
    const T* b,
    const T* a,
    const float* A_log,
    const T* dt_bias,
    StateT* ssm_state,
    const int ssm_state_stride_0,
    const int* token_indx,
    const int* cache_indices,
    const int cache_indices_stride_0,
    const int* num_accepted_tokens,
    const int num_spec_decodes,
    const int num_spec_tokens,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  using KERNEL = gated_delta_rule_spec_kernel<T, StateT, k_bucket_size>;
  auto range = KERNEL::get_nd_range(num_spec_decodes, num_v_heads, head_v_dim);
  assert(head_v_dim % KERNEL::v_dim_per_group == 0);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL task(
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_stride_0,
        token_indx,
        cache_indices,
        cache_indices_stride_0,
        num_accepted_tokens,
        num_spec_decodes,
        num_spec_tokens,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim);
    cgh.parallel_for(range, task);
  });
}

void gated_delta_rule(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,  // [total_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& q,        // [total_seqlen, num_k_heads, head_k_dim]
    const torch::Tensor& k,        // [total_seqlen, num_k_heads, head_k_dim]
    const torch::Tensor& v,        // [total_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& b,        // [total_seqlen, num_v_heads]
    const torch::Tensor& a,        // [total_seqlen, num_v_heads]
    const torch::Tensor& A_log,    // [num_v_heads]
    const torch::Tensor& dt_bias,  // [num_v_heads]
    torch::Tensor&
        ssm_state,  // [cache_batch_size, num_v_heads, head_v_dim, head_k_dim]
    const std::optional<torch::Tensor>& query_start_loc,  // [batch_size + 1]
    const std::optional<torch::Tensor>&
        token_indx,  // [num_virtual_tokens] or None
    const std::optional<torch::Tensor>& cache_indices,  // [batch_size]
    const std::optional<torch::Tensor>&
        has_initial_state,  // [batch_size] or None
    const std::optional<torch::Tensor>&
        num_accepted_tokens,  // [batch_size] or None
    const int num_prefills,
    const int num_decodes,
    const int num_spec_decodes) {
  TORCH_CHECK(query_start_loc.has_value() && cache_indices.has_value());

  // Spec path is selected when num_accepted_tokens is provided. In that case
  // cache_indices is the 2D spec_state_indices_tensor of shape
  // [num_spec_decodes, num_speculative_tokens + 1]; otherwise it is the 1D
  // non_spec_state_indices_tensor.
  const bool is_spec = num_accepted_tokens.has_value();

  int batch_size = query_start_loc->size(0) - 1;
  if (num_prefills == 0 && num_decodes > 0) {
    batch_size = num_decodes;
  }
  const int total_seqlen = q.size(0);
  const int num_k_heads = q.size(1);
  const int head_k_dim = q.size(2);
  const int num_v_heads = v.size(1);
  const int head_v_dim = v.size(2);
  const int ssm_state_stride_0 = ssm_state.stride(0);

  int num_spec_tokens = 0;
  int cache_indices_stride_0 = 0;
  if (is_spec) {
    TORCH_CHECK(cache_indices->dim() == 2);
    TORCH_CHECK(cache_indices->size(0) == num_spec_decodes);
    num_spec_tokens = cache_indices->size(1);
    cache_indices_stride_0 = cache_indices->stride(0);
    TORCH_CHECK(num_accepted_tokens->size(0) == num_spec_decodes);
  }

  TORCH_CHECK(num_v_heads % num_k_heads == 0);
  TORCH_CHECK(
      A_log.scalar_type() == at::kFloat,
      "A_log dtype must be float32, but got ",
      A_log.scalar_type());
  TORCH_CHECK(
      dt_bias.scalar_type() == core_attn_out.scalar_type(),
      "dt_bias dtype must match core_attn_out dtype (float16/bfloat16), but "
      "got dt_bias=",
      dt_bias.scalar_type(),
      ", core_attn_out=",
      core_attn_out.scalar_type());

  TORCH_CHECK(head_k_dim % sub_group_size == 0);
  const int k_bucket_size = head_k_dim / sub_group_size;

#define KERNEL_LAUNCHER(scalar_t, state_scalar_t, k_bucket_size)     \
  if (is_spec) {                                                     \
    kernel_launcher_spec<scalar_t, state_scalar_t, k_bucket_size>(   \
        queue,                                                       \
        reinterpret_cast<scalar_t*>(core_attn_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(q.data_ptr()),                   \
        reinterpret_cast<scalar_t*>(k.data_ptr()),                   \
        reinterpret_cast<scalar_t*>(v.data_ptr()),                   \
        reinterpret_cast<scalar_t*>(b.data_ptr()),                   \
        reinterpret_cast<scalar_t*>(a.data_ptr()),                   \
        reinterpret_cast<float*>(A_log.data_ptr()),                  \
        reinterpret_cast<scalar_t*>(dt_bias.data_ptr()),             \
        reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),     \
        ssm_state_stride_0,                                          \
        token_indx.has_value()                                       \
            ? reinterpret_cast<int*>(token_indx->data_ptr())         \
            : nullptr,                                               \
        reinterpret_cast<int*>(cache_indices->data_ptr()),           \
        cache_indices_stride_0,                                      \
        reinterpret_cast<int*>(num_accepted_tokens->data_ptr()),     \
        num_spec_decodes,                                            \
        num_spec_tokens,                                             \
        num_k_heads,                                                 \
        head_k_dim,                                                  \
        num_v_heads,                                                 \
        head_v_dim);                                                 \
  } else {                                                           \
    kernel_launcher<scalar_t, state_scalar_t, k_bucket_size>(        \
        queue,                                                       \
        reinterpret_cast<scalar_t*>(core_attn_out.data_ptr()),       \
        reinterpret_cast<scalar_t*>(q.data_ptr()),                   \
        reinterpret_cast<scalar_t*>(k.data_ptr()),                   \
        reinterpret_cast<scalar_t*>(v.data_ptr()),                   \
        reinterpret_cast<scalar_t*>(b.data_ptr()),                   \
        reinterpret_cast<scalar_t*>(a.data_ptr()),                   \
        reinterpret_cast<float*>(A_log.data_ptr()),                  \
        reinterpret_cast<scalar_t*>(dt_bias.data_ptr()),             \
        reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),     \
        ssm_state_stride_0,                                          \
        reinterpret_cast<int*>(query_start_loc->data_ptr()),         \
        token_indx.has_value()                                       \
            ? reinterpret_cast<int*>(token_indx->data_ptr())         \
            : nullptr,                                               \
        reinterpret_cast<int*>(cache_indices->data_ptr()),           \
        has_initial_state.has_value()                                \
            ? reinterpret_cast<bool*>(has_initial_state->data_ptr()) \
            : nullptr,                                               \
        /*num_accepted_tokens=*/nullptr,                             \
        batch_size,                                                  \
        total_seqlen,                                                \
        num_k_heads,                                                 \
        head_k_dim,                                                  \
        num_v_heads,                                                 \
        head_v_dim);                                                 \
  }

#define BUCKET_DISPATCH(scalar_t, state_scalar_t, k_bucket_size) \
  switch (k_bucket_size) {                                       \
    case 1:                                                      \
      KERNEL_LAUNCHER(scalar_t, state_scalar_t, 1)               \
      break;                                                     \
    case 2:                                                      \
      KERNEL_LAUNCHER(scalar_t, state_scalar_t, 2)               \
      break;                                                     \
    case 4:                                                      \
      KERNEL_LAUNCHER(scalar_t, state_scalar_t, 4)               \
      break;                                                     \
    case 8:                                                      \
      KERNEL_LAUNCHER(scalar_t, state_scalar_t, 8)               \
      break;                                                     \
    default:                                                     \
      TORCH_CHECK(false);                                        \
  }

#define DISPATCH_STATE_DTYPE(scalar_t)                                  \
  do {                                                                  \
    if (ssm_state.scalar_type() == at::kFloat) {                        \
      using state_scalar_t = float;                                     \
      BUCKET_DISPATCH(scalar_t, state_scalar_t, k_bucket_size)          \
    } else if (ssm_state.scalar_type() == at::kBFloat16) {              \
      using state_scalar_t = sycl::ext::oneapi::bfloat16;               \
      BUCKET_DISPATCH(scalar_t, state_scalar_t, k_bucket_size)          \
    } else if (ssm_state.scalar_type() == at::kHalf) {                  \
      using state_scalar_t = sycl::half;                                \
      BUCKET_DISPATCH(scalar_t, state_scalar_t, k_bucket_size)          \
    } else {                                                            \
      TORCH_CHECK(                                                      \
          false,                                                        \
          "ssm_state dtype must be float32/float16/bfloat16, but got ", \
          ssm_state.scalar_type());                                     \
    }                                                                   \
  } while (0)

  if (core_attn_out.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    DISPATCH_STATE_DTYPE(scalar_t);
  } else if (core_attn_out.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    DISPATCH_STATE_DTYPE(scalar_t);
  } else {
    using scalar_t = float;
    DISPATCH_STATE_DTYPE(scalar_t);
  }
#undef DISPATCH_STATE_DTYPE
#undef BUCKET_DISPATCH
#undef KERNEL_LAUNCHER
}

}  // namespace gdn