#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

namespace gdn {
static constexpr int sub_group_size = 32;
template <typename T, int k_bucket_size>
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
      const T* A_log,
      const T* dt_bias,
      T* ssm_state,
      const int ssm_state_stride_0,
      const int* query_start_loc,
      const int* cache_indices,
      const bool* has_initial_state,
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
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
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

    T* ssm_state_ptr =
        ssm_state +
        static_cast<int64_t>(cache_indices[batch_id]) * ssm_state_stride_0;

    // load state
    if (has_initial_state == nullptr || has_initial_state[batch_id]) {
// #pragma unroll
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] = ssm_state_ptr
              [num_v_heads_id * head_k_dim * head_v_dim +
               (k_bucket_size * sg_local_id + i) +
               (head_v_dim_id + j) * head_k_dim];
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
    // O(t) = S(t) * q(t)
    // S(t) = g(t)* S(t - 1) + (v(t) - g(t) * S(t - 1)* k(t)) * beta(t) * k(t)
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
// get g(t) * S(t - 1)* k(t)
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

// get (v(t) - g(t) * S(t - 1)* k(t)) * beta(t)
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

      // store O(t)
      if (sg_local_id == 0) {
#pragma unroll
        for (int i = 0; i < v_dim_per_sg; ++i) {
          core_attn_out
              [t * num_v_heads * head_v_dim + num_v_heads_id * head_v_dim +
               head_v_dim_id + i] = res[i];
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
                state_local[j * k_bucket_size + i];
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
  const T* A_log;
  const T* dt_bias;
  T* ssm_state;
  const int ssm_state_stride_0;
  const int* query_start_loc;
  const int* cache_indices;
  const bool* has_initial_state;
  const int batch_size;
  const int total_seqlen;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
};

template <typename T, int k_bucket_size>
void kernel_launcher(
    sycl::queue& queue,
    T* core_attn_out,
    const T* q,
    const T* k,
    const T* v,
    const T* b,
    const T* a,
    const T* A_log,
    const T* dt_bias,
    T* ssm_state,
    const int ssm_state_stride_0,
    const int* query_start_loc,
    const int* cache_indices,
    const bool* has_initial_state,
    const int batch_size,
    const int total_seqlen,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  using KERNEL = gated_delta_rule_kernel<T, k_bucket_size>;
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
        cache_indices,
        has_initial_state,
        batch_size,
        total_seqlen,
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
    const torch::Tensor& query_start_loc,  // [batch_size + 1]
    const torch::Tensor& cache_indices,    // [batch_size]
    const std::optional<torch::Tensor>&
        has_initial_state,  // [batch_size] or None
    const int num_prefills,
    const int num_decodes) {
  if (num_prefills == 0 && num_decodes == 0) {
    return;
  }

  int batch_size = query_start_loc.size(0) - 1;
  if (num_prefills == 0 && num_decodes > 0) {
    batch_size = num_decodes;
  }
  const int total_seqlen = q.size(0);
  const int num_k_heads = q.size(1);
  const int head_k_dim = q.size(2);
  const int num_v_heads = v.size(1);
  const int head_v_dim = v.size(2);
  const int ssm_state_stride_0 = ssm_state.stride(0);

  TORCH_CHECK(num_v_heads % num_k_heads == 0);
  TORCH_CHECK(head_k_dim % sub_group_size == 0);
  const int k_bucket_size = head_k_dim / sub_group_size;

#define KERNEL_LAUNCHER(scalar_t, k_bucket_size)                   \
  kernel_launcher<scalar_t, k_bucket_size>(                        \
      queue,                                                       \
      reinterpret_cast<scalar_t*>(core_attn_out.data_ptr()),       \
      reinterpret_cast<scalar_t*>(q.data_ptr()),                   \
      reinterpret_cast<scalar_t*>(k.data_ptr()),                   \
      reinterpret_cast<scalar_t*>(v.data_ptr()),                   \
      reinterpret_cast<scalar_t*>(b.data_ptr()),                   \
      reinterpret_cast<scalar_t*>(a.data_ptr()),                   \
      reinterpret_cast<scalar_t*>(A_log.data_ptr()),               \
      reinterpret_cast<scalar_t*>(dt_bias.data_ptr()),             \
      reinterpret_cast<scalar_t*>(ssm_state.data_ptr()),           \
      ssm_state_stride_0,                                          \
      reinterpret_cast<int*>(query_start_loc.data_ptr()),          \
      reinterpret_cast<int*>(cache_indices.data_ptr()),            \
      has_initial_state.has_value()                                \
          ? reinterpret_cast<bool*>(has_initial_state->data_ptr()) \
          : nullptr,                                               \
      batch_size,                                                  \
      total_seqlen,                                                \
      num_k_heads,                                                 \
      head_k_dim,                                                  \
      num_v_heads,                                                 \
      head_v_dim);

#define BUCKET_DISPATCH(scalar_t, k_bucket_size) \
  switch (k_bucket_size) {                       \
    case 1:                                      \
      KERNEL_LAUNCHER(scalar_t, 1)               \
      break;                                     \
    case 2:                                      \
      KERNEL_LAUNCHER(scalar_t, 2)               \
      break;                                     \
    case 4:                                      \
      KERNEL_LAUNCHER(scalar_t, 4)               \
      break;                                     \
    case 8:                                      \
      KERNEL_LAUNCHER(scalar_t, 8)               \
      break;                                     \
    default:                                     \
      TORCH_CHECK(false);                        \
  }

  if (core_attn_out.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    BUCKET_DISPATCH(scalar_t, k_bucket_size)
  } else if (core_attn_out.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    BUCKET_DISPATCH(scalar_t, k_bucket_size)
  } else {
    using scalar_t = float;
    BUCKET_DISPATCH(scalar_t, k_bucket_size)
  }
#undef BUCKET_DISPATCH
#undef KERNEL_LAUNCHER
}

}  // namespace gdn
