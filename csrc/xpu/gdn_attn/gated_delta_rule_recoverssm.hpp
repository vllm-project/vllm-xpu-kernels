#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "gated_delta_rule.hpp"

namespace gdn {

// -----------------------------------------------------------------------------
// RecoverSSM spec-decoding kernel (vLLM's --use-replayssm, vllm#58863).
//
// Same recurrence as gated_delta_rule_spec_kernel, but with RecoverSSM's state
// contract instead of one state slot per draft position:
//
// - The initial SSM state for sequence n is read from its single state block
//   `state_indices[n]` and is never written here. vLLM commits the accepted
//   prefix into that block after sampling, from the replay records below.
// - For every local token t the kernel stores a float32 replay record at
//   `replay_state[state_indices[n], hv, t, :]`:
//     [0, V)      corr  = (v_t - S_{t-1}^T k_t) * beta_t   (after the decay)
//     [V, V + K)  k_t, L2-normalised
//     [V + K]     decay = exp(-exp(A_log) * softplus(a_t + dt_bias))
//   so that S_t = decay * S_{t-1} + k_t corr^T can be replayed exactly.
// - A sequence whose state index is <= null_block_id writes zeros to
//   core_attn_out and leaves its replay block untouched.
// - `core_attn_out` is the GLOBAL active buffer; writes go through
//   `token_indx` (local token -> global position) when it is given.
// -----------------------------------------------------------------------------
template <typename T, typename StateT, int k_bucket_size>
struct gated_delta_rule_spec_recoverssm_kernel {
 public:
  static constexpr int group_size = 256;
  static constexpr int sg_per_group = group_size / sub_group_size;
  static constexpr int v_dim_per_sg = 4;
  static constexpr int v_dim_per_group = v_dim_per_sg * sg_per_group;
  static constexpr float eps = 0.000001;

  gated_delta_rule_spec_recoverssm_kernel(
      T* core_attn_out,
      const T* q,
      const T* k,
      const T* v,
      const T* b,
      const T* a,
      const float* A_log,
      const T* dt_bias,
      const StateT* ssm_state,
      const int64_t ssm_state_stride_0,
      float* replay_state,
      const int64_t replay_stride_0,
      const int64_t replay_stride_1,
      const int64_t replay_stride_2,
      const int64_t replay_stride_3,
      const int* query_start_loc,
      const int* token_indx,
      const int* state_indices,
      const int null_block_id,
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
        replay_state(replay_state),
        replay_stride_0(replay_stride_0),
        replay_stride_1(replay_stride_1),
        replay_stride_2(replay_stride_2),
        replay_stride_3(replay_stride_3),
        query_start_loc(query_start_loc),
        token_indx(token_indx),
        state_indices(state_indices),
        null_block_id(null_block_id),
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

    const int token_start = query_start_loc[batch_id];
    const int token_end = query_start_loc[batch_id + 1];
    const int state_idx = state_indices[batch_id];

    // Padded / freed request: zero its outputs, touch no state.
    if (state_idx <= null_block_id) {
      if (sg_local_id == 0) {
        for (int t = token_start; t < token_end; ++t) {
          const int global_t = (token_indx != nullptr) ? token_indx[t] : t;
#pragma unroll
          for (int i = 0; i < v_dim_per_sg; ++i) {
            core_attn_out
                [static_cast<int64_t>(global_t) * num_v_heads * head_v_dim +
                 num_v_heads_id * head_v_dim + head_v_dim_id + i] = T(0.0f);
          }
        }
      }
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

    // -- Load the checkpoint state (read-only) --------------------------------
    const StateT* init_state_ptr =
        ssm_state + static_cast<int64_t>(state_idx) * ssm_state_stride_0;
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

    float* replay_base = replay_state +
                         static_cast<int64_t>(state_idx) * replay_stride_0 +
                         static_cast<int64_t>(num_v_heads_id) * replay_stride_1;

    for (int t = token_start, t_local = 0; t < token_end; ++t, ++t_local) {
      float b_local = static_cast<float>(b[t * num_v_heads + num_v_heads_id]);
      float beta = act_sigmoid(b_local);
      // Add in float32 so that the recorded decay matches vLLM's float32
      // verify/commit.
      float a_local = static_cast<float>(a[t * num_v_heads + num_v_heads_id]) +
                      dt_bias_local;
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

      // Write O(t) to GLOBAL core_attn_out via token_indx remap, and this
      // sub-group's slice of the corr record.
      float* rec =
          replay_base + static_cast<int64_t>(t_local) * replay_stride_2;
      if (sg_local_id == 0) {
        const int global_t = (token_indx != nullptr) ? token_indx[t] : t;
#pragma unroll
        for (int i = 0; i < v_dim_per_sg; ++i) {
          core_attn_out
              [static_cast<int64_t>(global_t) * num_v_heads * head_v_dim +
               num_v_heads_id * head_v_dim + head_v_dim_id + i] = res[i];
          rec[static_cast<int64_t>(head_v_dim_id + i) * replay_stride_3] =
              delta[i];
        }
      }
      // k and the decay are shared by every V slice of this head; one
      // sub-group writes them.
      if (v_bucket_id == 0 && sg_id == 0) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          rec[static_cast<int64_t>(
                  head_v_dim + k_bucket_size * sg_local_id + i) *
              replay_stride_3] = k_local[i];
        }
        if (sg_local_id == 0) {
          rec[static_cast<int64_t>(head_v_dim + head_k_dim) * replay_stride_3] =
              g;
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
  const StateT* ssm_state;
  const int64_t ssm_state_stride_0;
  float* replay_state;
  const int64_t replay_stride_0;
  const int64_t replay_stride_1;
  const int64_t replay_stride_2;
  const int64_t replay_stride_3;
  const int* query_start_loc;
  const int* token_indx;
  const int* state_indices;
  const int null_block_id;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
};

template <typename T, typename StateT, int k_bucket_size>
void kernel_launcher_spec_recoverssm(
    sycl::queue& queue,
    T* core_attn_out,
    const T* q,
    const T* k,
    const T* v,
    const T* b,
    const T* a,
    const float* A_log,
    const T* dt_bias,
    const StateT* ssm_state,
    const int64_t ssm_state_stride_0,
    float* replay_state,
    const int64_t replay_stride_0,
    const int64_t replay_stride_1,
    const int64_t replay_stride_2,
    const int64_t replay_stride_3,
    const int* query_start_loc,
    const int* token_indx,
    const int* state_indices,
    const int null_block_id,
    const int num_spec_decodes,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  using KERNEL =
      gated_delta_rule_spec_recoverssm_kernel<T, StateT, k_bucket_size>;
  auto range = KERNEL::get_nd_range(num_spec_decodes, num_v_heads, head_v_dim);
  TORCH_CHECK(
      head_v_dim % KERNEL::v_dim_per_group == 0,
      "head_v_dim must be a multiple of ",
      KERNEL::v_dim_per_group);
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
        replay_state,
        replay_stride_0,
        replay_stride_1,
        replay_stride_2,
        replay_stride_3,
        query_start_loc,
        token_indx,
        state_indices,
        null_block_id,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim);
    cgh.parallel_for(range, task);
  });
}

inline void gated_delta_rule_recoverssm(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,  // [total_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& q,        // [total_seqlen, num_k_heads, head_k_dim]
    const torch::Tensor& k,        // [total_seqlen, num_k_heads, head_k_dim]
    const torch::Tensor& v,        // [total_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& b,        // [total_seqlen, num_v_heads]
    const torch::Tensor& a,        // [total_seqlen, num_v_heads]
    const torch::Tensor& A_log,    // [num_v_heads]
    const torch::Tensor& dt_bias,  // [num_v_heads]
    const torch::Tensor&
        ssm_state,  // [cache_batch_size, num_v_heads, head_v_dim, head_k_dim]
    torch::Tensor&
        replay_state,  // [cache_batch_size, num_v_heads,
                       //  max_query_len, head_v_dim + head_k_dim + 1]
    const torch::Tensor& query_start_loc,            // [num_spec_decodes + 1]
    const std::optional<torch::Tensor>& token_indx,  // [total_seqlen] or None
    const torch::Tensor& state_indices,              // [num_spec_decodes]
    const int64_t null_block_id) {
  const int num_spec_decodes = state_indices.size(0);
  const int num_k_heads = q.size(1);
  const int head_k_dim = q.size(2);
  const int num_v_heads = v.size(1);
  const int head_v_dim = v.size(2);

  TORCH_CHECK(num_v_heads % num_k_heads == 0);
  TORCH_CHECK(head_k_dim % sub_group_size == 0);
  const int k_bucket_size = head_k_dim / sub_group_size;

#define KERNEL_LAUNCHER(scalar_t, state_scalar_t, k_bucket_size)              \
  kernel_launcher_spec_recoverssm<scalar_t, state_scalar_t, k_bucket_size>(   \
      queue,                                                                  \
      reinterpret_cast<scalar_t*>(core_attn_out.data_ptr()),                  \
      reinterpret_cast<scalar_t*>(q.data_ptr()),                              \
      reinterpret_cast<scalar_t*>(k.data_ptr()),                              \
      reinterpret_cast<scalar_t*>(v.data_ptr()),                              \
      reinterpret_cast<scalar_t*>(b.data_ptr()),                              \
      reinterpret_cast<scalar_t*>(a.data_ptr()),                              \
      reinterpret_cast<float*>(A_log.data_ptr()),                             \
      reinterpret_cast<scalar_t*>(dt_bias.data_ptr()),                        \
      reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),                \
      ssm_state.stride(0),                                                    \
      reinterpret_cast<float*>(replay_state.data_ptr()),                      \
      replay_state.stride(0),                                                 \
      replay_state.stride(1),                                                 \
      replay_state.stride(2),                                                 \
      replay_state.stride(3),                                                 \
      reinterpret_cast<int*>(query_start_loc.data_ptr()),                     \
      token_indx.has_value() ? reinterpret_cast<int*>(token_indx->data_ptr()) \
                             : nullptr,                                       \
      reinterpret_cast<int*>(state_indices.data_ptr()),                       \
      static_cast<int>(null_block_id),                                        \
      num_spec_decodes,                                                       \
      num_k_heads,                                                            \
      head_k_dim,                                                             \
      num_v_heads,                                                            \
      head_v_dim);

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
