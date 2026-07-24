#pragma once

#include <sycl/sycl.hpp>
#include <type_traits>
#include <torch/all.h>

namespace gdn {
namespace xe2 {

static constexpr int decode_sub_group_size = 32;

template <typename T, typename StateT, int k_bucket_size>
struct gated_delta_rule_decode_kernel_xe2 {
 public:
  static constexpr int group_size = 256;
  static constexpr int sg_per_group = group_size / decode_sub_group_size;
  static constexpr int v_dim_per_sg = 4;
  static constexpr int v_dim_per_group = v_dim_per_sg * sg_per_group;
  static constexpr float eps = 0.000001f;

  gated_delta_rule_decode_kernel_xe2(
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
      const int batch_size,
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
        batch_size(batch_size),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int batch_size, const int num_v_heads, const int head_v_dim) {
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(batch_size, num_v_heads, 1);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline float act_sigmoid(float& x) {
    return 1.0f / (1.0f + sycl::exp(-x));
  }

  static inline float
  act_softplus(float& x, float beta = 1.0f, float threshold = 20.0f) {
    if (beta * x < threshold) {
      return sycl::log(1.0f + sycl::exp(beta * x)) / beta;
    } else {
      return x;
    }
  }

  template <typename VecT, int vec_size>
  static inline void load_vec(const VecT* ptr, VecT (&dst)[vec_size]) {
    sycl::vec<VecT, vec_size> in =
        *reinterpret_cast<const sycl::vec<VecT, vec_size>*>(ptr);
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      dst[i] = in[i];
    }
  }

  template <typename VecT, int vec_size>
  static inline void load_vec_to_float(const VecT* ptr, float (&dst)[vec_size]) {
    VecT tmp[vec_size];
    load_vec<VecT, vec_size>(ptr, tmp);
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      dst[i] = static_cast<float>(tmp[i]);
    }
  }

  template <typename VecT, int vec_size>
  static inline void store_vec(VecT* ptr, const VecT (&src)[vec_size]) {
    sycl::vec<VecT, vec_size> out;
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      out[i] = src[i];
    }
    *reinterpret_cast<sycl::vec<VecT, vec_size>*>(ptr) = out;
  }

  template <typename VecT, int vec_size>
  static inline void store_vec_from_float(
      VecT* ptr, const float (&src)[vec_size]) {
    VecT tmp[vec_size];
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      tmp[i] = static_cast<VecT>(src[i]);
    }
    store_vec<VecT, vec_size>(ptr, tmp);
  }

  [[sycl::reqd_sub_group_size(decode_sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    const int decode_id = item.get_group(0);
    const int num_v_heads_id = item.get_group(1);

    auto sg = item.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_local_id = sg.get_local_id();
    const int kv_ratio = num_v_heads / num_k_heads;
    const float scale = 1.0f / sycl::sqrt(float(head_k_dim));
    float A_log_local = A_log[num_v_heads_id];
    float dt_bias_local = dt_bias[num_v_heads_id];
    A_log_local = -sycl::exp(A_log_local);

    const int state_head_offset = num_v_heads_id * head_k_dim * head_v_dim;
    const int global_t =
        (token_indx != nullptr) ? token_indx[decode_id] : decode_id;

    StateT* ssm_state_ptr =
        ssm_state +
        static_cast<int64_t>(cache_indices[decode_id]) * ssm_state_stride_0;

    for (int head_v_dim_id = sg_id * v_dim_per_sg; head_v_dim_id < head_v_dim;
         head_v_dim_id += v_dim_per_group) {
      alignas(16) float state_local[v_dim_per_sg * k_bucket_size];
      float q_local[k_bucket_size];
      float k_local[k_bucket_size];
      float v_local[v_dim_per_sg];

#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
        const int state_offset =
            state_head_offset + (head_v_dim_id + j) * head_k_dim +
            k_bucket_size * sg_local_id;
        load_vec_to_float<StateT, k_bucket_size>(
            ssm_state_ptr + state_offset,
            reinterpret_cast<float (&)[k_bucket_size]>(
                state_local[j * k_bucket_size]));
      }

      float b_local = b[decode_id * num_v_heads + num_v_heads_id];
      float beta = act_sigmoid(b_local);
      float a_local = a[decode_id * num_v_heads + num_v_heads_id] +
                      dt_bias_local;
      float g = sycl::exp(A_log_local * act_softplus(a_local));

      float q_sum = 0.0f;
      float k_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        q_local[i] =
            q[decode_id * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        k_local[i] =
            k[decode_id * num_k_heads * head_k_dim +
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

      load_vec_to_float<T, v_dim_per_sg>(
          v + decode_id * num_v_heads * head_v_dim +
              num_v_heads_id * head_v_dim + head_v_dim_id,
          v_local);

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

      if (sg_local_id == 0) {
        store_vec_from_float<T, v_dim_per_sg>(
            core_attn_out + global_t * num_v_heads * head_v_dim +
                num_v_heads_id * head_v_dim + head_v_dim_id,
            res);
      }

#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
        const int state_offset =
            state_head_offset + (head_v_dim_id + j) * head_k_dim +
            k_bucket_size * sg_local_id;
        store_vec_from_float<StateT, k_bucket_size>(
            ssm_state_ptr + state_offset,
            reinterpret_cast<float (&)[k_bucket_size]>(
                state_local[j * k_bucket_size]));
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
  const int batch_size;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
};

template <typename T, typename StateT, int k_bucket_size>
struct gated_delta_rule_decode_spec_kernel_xe2 {
 public:
  static constexpr int group_size = 256;
  static constexpr int sg_per_group = group_size / decode_sub_group_size;
  static constexpr int v_dim_per_sg = 4;
  static constexpr int v_dim_per_group = v_dim_per_sg * sg_per_group;
  static constexpr float eps = 0.000001f;

  gated_delta_rule_decode_spec_kernel_xe2(
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
      const int num_spec_decodes, const int num_v_heads,
      const int head_v_dim) {
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(num_spec_decodes, num_v_heads, 1);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline float act_sigmoid(float& x) {
    return 1.0f / (1.0f + sycl::exp(-x));
  }

  static inline float
  act_softplus(float& x, float beta = 1.0f, float threshold = 20.0f) {
    if (beta * x < threshold) {
      return sycl::log(1.0f + sycl::exp(beta * x)) / beta;
    } else {
      return x;
    }
  }

  template <typename VecT, int vec_size>
  static inline void load_vec(const VecT* ptr, VecT (&dst)[vec_size]) {
    sycl::vec<VecT, vec_size> in =
        *reinterpret_cast<const sycl::vec<VecT, vec_size>*>(ptr);
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      dst[i] = in[i];
    }
  }

  template <typename VecT, int vec_size>
  static inline void load_vec_to_float(const VecT* ptr, float (&dst)[vec_size]) {
    VecT tmp[vec_size];
    load_vec<VecT, vec_size>(ptr, tmp);
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      dst[i] = static_cast<float>(tmp[i]);
    }
  }

  template <typename VecT, int vec_size>
  static inline void store_vec(VecT* ptr, const VecT (&src)[vec_size]) {
    sycl::vec<VecT, vec_size> out;
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      out[i] = src[i];
    }
    *reinterpret_cast<sycl::vec<VecT, vec_size>*>(ptr) = out;
  }

  template <typename VecT, int vec_size>
  static inline void store_vec_from_float(
      VecT* ptr, const float (&src)[vec_size]) {
    VecT tmp[vec_size];
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      tmp[i] = static_cast<VecT>(src[i]);
    }
    store_vec<VecT, vec_size>(ptr, tmp);
  }

  [[sycl::reqd_sub_group_size(decode_sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    const int decode_id = item.get_group(0);
    const int num_v_heads_id = item.get_group(1);

    auto sg = item.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_local_id = sg.get_local_id();
    const int kv_ratio = num_v_heads / num_k_heads;
    const float scale = 1.0f / sycl::sqrt(float(head_k_dim));
    float A_log_local = A_log[num_v_heads_id];
    float dt_bias_local = dt_bias[num_v_heads_id];
    A_log_local = -sycl::exp(A_log_local);

    const int state_head_offset = num_v_heads_id * head_k_dim * head_v_dim;

    int init_col = num_accepted_tokens[decode_id] - 1;
    if (init_col < 0) {
      init_col = 0;
    }
    const int init_state_idx =
        cache_indices[decode_id * cache_indices_stride_0 + init_col];
    const StateT* init_state_ptr =
        ssm_state + static_cast<int64_t>(init_state_idx) * ssm_state_stride_0;

    for (int head_v_dim_id = sg_id * v_dim_per_sg; head_v_dim_id < head_v_dim;
         head_v_dim_id += v_dim_per_group) {
      alignas(16) float state_local[v_dim_per_sg * k_bucket_size];
      float q_local[k_bucket_size];
      float k_local[k_bucket_size];
      float v_local[v_dim_per_sg];

#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
        const int state_offset =
            state_head_offset + (head_v_dim_id + j) * head_k_dim +
            k_bucket_size * sg_local_id;
        load_vec_to_float<StateT, k_bucket_size>(
            init_state_ptr + state_offset,
            reinterpret_cast<float (&)[k_bucket_size]>(
                state_local[j * k_bucket_size]));
      }

      for (int t_local = 0; t_local < num_spec_tokens; ++t_local) {
        const int t = decode_id * num_spec_tokens + t_local;

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

        load_vec_to_float<T, v_dim_per_sg>(
            v + t * num_v_heads * head_v_dim +
                num_v_heads_id * head_v_dim + head_v_dim_id,
            v_local);

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

        if (sg_local_id == 0) {
          const int global_t = (token_indx != nullptr) ? token_indx[t] : t;
          store_vec_from_float<T, v_dim_per_sg>(
              core_attn_out + global_t * num_v_heads * head_v_dim +
                  num_v_heads_id * head_v_dim + head_v_dim_id,
              res);
        }

        const int writeback_idx =
            cache_indices[decode_id * cache_indices_stride_0 + t_local];
        StateT* writeback_ptr =
            ssm_state + static_cast<int64_t>(writeback_idx) * ssm_state_stride_0;
#pragma unroll
        for (int j = 0; j < v_dim_per_sg; ++j) {
          const int state_offset =
              state_head_offset + (head_v_dim_id + j) * head_k_dim +
              k_bucket_size * sg_local_id;
          store_vec_from_float<StateT, k_bucket_size>(
              writeback_ptr + state_offset,
              reinterpret_cast<float (&)[k_bucket_size]>(
                  state_local[j * k_bucket_size]));
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
void kernel_launcher_decode_xe2(
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
    const int batch_size,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  using KERNEL = gated_delta_rule_decode_kernel_xe2<T, StateT, k_bucket_size>;
  auto range = KERNEL::get_nd_range(batch_size, num_v_heads, head_v_dim);
  TORCH_CHECK(
      head_v_dim % KERNEL::v_dim_per_group == 0,
      "head_v_dim must be divisible by ",
      KERNEL::v_dim_per_group,
      " for XE2 decode, but got ",
      head_v_dim);
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
        batch_size,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim);
    cgh.parallel_for(range, task);
  });
}

template <typename T>
void dispatch_state_dtype_decode_xe2(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,
    const int ssm_state_stride_0,
    const int* token_indx,
    const int* cache_indices,
    const int batch_size,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim,
    const int k_bucket_size) {
#define XE2_DECODE_BUCKET_DISPATCH(state_scalar_t)                           \
  switch (k_bucket_size) {                                                   \
    case 1:                                                                  \
      kernel_launcher_decode_xe2<T, state_scalar_t, 1>(                      \
          queue,                                                              \
          reinterpret_cast<T*>(core_attn_out.data_ptr()),                    \
          reinterpret_cast<T*>(q.data_ptr()),                                \
          reinterpret_cast<T*>(k.data_ptr()),                                \
          reinterpret_cast<T*>(v.data_ptr()),                                \
          reinterpret_cast<T*>(b.data_ptr()),                                \
          reinterpret_cast<T*>(a.data_ptr()),                                \
          reinterpret_cast<float*>(A_log.data_ptr()),                        \
          reinterpret_cast<T*>(dt_bias.data_ptr()),                          \
          reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),           \
          ssm_state_stride_0,                                                 \
          token_indx,                                                         \
          cache_indices,                                                      \
          batch_size,                                                         \
          num_k_heads,                                                        \
          head_k_dim,                                                         \
          num_v_heads,                                                        \
          head_v_dim);                                                        \
      break;                                                                  \
    case 2:                                                                  \
      kernel_launcher_decode_xe2<T, state_scalar_t, 2>(                      \
          queue,                                                              \
          reinterpret_cast<T*>(core_attn_out.data_ptr()),                    \
          reinterpret_cast<T*>(q.data_ptr()),                                \
          reinterpret_cast<T*>(k.data_ptr()),                                \
          reinterpret_cast<T*>(v.data_ptr()),                                \
          reinterpret_cast<T*>(b.data_ptr()),                                \
          reinterpret_cast<T*>(a.data_ptr()),                                \
          reinterpret_cast<float*>(A_log.data_ptr()),                        \
          reinterpret_cast<T*>(dt_bias.data_ptr()),                          \
          reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),           \
          ssm_state_stride_0,                                                 \
          token_indx,                                                         \
          cache_indices,                                                      \
          batch_size,                                                         \
          num_k_heads,                                                        \
          head_k_dim,                                                         \
          num_v_heads,                                                        \
          head_v_dim);                                                        \
      break;                                                                  \
    case 4:                                                                  \
      kernel_launcher_decode_xe2<T, state_scalar_t, 4>(                      \
          queue,                                                              \
          reinterpret_cast<T*>(core_attn_out.data_ptr()),                    \
          reinterpret_cast<T*>(q.data_ptr()),                                \
          reinterpret_cast<T*>(k.data_ptr()),                                \
          reinterpret_cast<T*>(v.data_ptr()),                                \
          reinterpret_cast<T*>(b.data_ptr()),                                \
          reinterpret_cast<T*>(a.data_ptr()),                                \
          reinterpret_cast<float*>(A_log.data_ptr()),                        \
          reinterpret_cast<T*>(dt_bias.data_ptr()),                          \
          reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),           \
          ssm_state_stride_0,                                                 \
          token_indx,                                                         \
          cache_indices,                                                      \
          batch_size,                                                         \
          num_k_heads,                                                        \
          head_k_dim,                                                         \
          num_v_heads,                                                        \
          head_v_dim);                                                        \
      break;                                                                  \
    case 8:                                                                  \
      kernel_launcher_decode_xe2<T, state_scalar_t, 8>(                      \
          queue,                                                              \
          reinterpret_cast<T*>(core_attn_out.data_ptr()),                    \
          reinterpret_cast<T*>(q.data_ptr()),                                \
          reinterpret_cast<T*>(k.data_ptr()),                                \
          reinterpret_cast<T*>(v.data_ptr()),                                \
          reinterpret_cast<T*>(b.data_ptr()),                                \
          reinterpret_cast<T*>(a.data_ptr()),                                \
          reinterpret_cast<float*>(A_log.data_ptr()),                        \
          reinterpret_cast<T*>(dt_bias.data_ptr()),                          \
          reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),           \
          ssm_state_stride_0,                                                 \
          token_indx,                                                         \
          cache_indices,                                                      \
          batch_size,                                                         \
          num_k_heads,                                                        \
          head_k_dim,                                                         \
          num_v_heads,                                                        \
          head_v_dim);                                                        \
      break;                                                                  \
    default:                                                                  \
      TORCH_CHECK(false, "unsupported k_bucket_size: ", k_bucket_size);      \
  }

  if (ssm_state.scalar_type() == at::kFloat) {
    using state_scalar_t = float;
    XE2_DECODE_BUCKET_DISPATCH(state_scalar_t)
  } else if (ssm_state.scalar_type() == at::kBFloat16) {
    using state_scalar_t = sycl::ext::oneapi::bfloat16;
    XE2_DECODE_BUCKET_DISPATCH(state_scalar_t)
  } else if (ssm_state.scalar_type() == at::kHalf) {
    using state_scalar_t = sycl::half;
    XE2_DECODE_BUCKET_DISPATCH(state_scalar_t)
  } else {
    TORCH_CHECK(
        false,
        "ssm_state dtype must be float32/float16/bfloat16, but got ",
        ssm_state.scalar_type());
  }

#undef XE2_DECODE_BUCKET_DISPATCH
}

template <typename T, typename StateT, int k_bucket_size>
void kernel_launcher_decode_xe2_spec(
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
  using KERNEL =
      gated_delta_rule_decode_spec_kernel_xe2<T, StateT, k_bucket_size>;
  auto range = KERNEL::get_nd_range(num_spec_decodes, num_v_heads, head_v_dim);
  TORCH_CHECK(
      head_v_dim % KERNEL::v_dim_per_group == 0,
      "head_v_dim must be divisible by ",
      KERNEL::v_dim_per_group,
      " for XE2 spec decode, but got ",
      head_v_dim);
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

template <typename T>
void dispatch_state_dtype_decode_xe2_spec(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,
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
    const int head_v_dim,
    const int k_bucket_size) {
#define XE2_DECODE_SPEC_BUCKET_DISPATCH(state_scalar_t)                    \
  switch (k_bucket_size) {                                                 \
    case 1:                                                                \
      kernel_launcher_decode_xe2_spec<T, state_scalar_t, 1>(               \
          queue,                                                            \
          reinterpret_cast<T*>(core_attn_out.data_ptr()),                  \
          reinterpret_cast<T*>(q.data_ptr()),                              \
          reinterpret_cast<T*>(k.data_ptr()),                              \
          reinterpret_cast<T*>(v.data_ptr()),                              \
          reinterpret_cast<T*>(b.data_ptr()),                              \
          reinterpret_cast<T*>(a.data_ptr()),                              \
          reinterpret_cast<float*>(A_log.data_ptr()),                      \
          reinterpret_cast<T*>(dt_bias.data_ptr()),                        \
          reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),         \
          ssm_state_stride_0,                                               \
          token_indx,                                                       \
          cache_indices,                                                    \
          cache_indices_stride_0,                                           \
          num_accepted_tokens,                                              \
          num_spec_decodes,                                                 \
          num_spec_tokens,                                                  \
          num_k_heads,                                                      \
          head_k_dim,                                                       \
          num_v_heads,                                                      \
          head_v_dim);                                                      \
      break;                                                                \
    case 2:                                                                \
      kernel_launcher_decode_xe2_spec<T, state_scalar_t, 2>(               \
          queue,                                                            \
          reinterpret_cast<T*>(core_attn_out.data_ptr()),                  \
          reinterpret_cast<T*>(q.data_ptr()),                              \
          reinterpret_cast<T*>(k.data_ptr()),                              \
          reinterpret_cast<T*>(v.data_ptr()),                              \
          reinterpret_cast<T*>(b.data_ptr()),                              \
          reinterpret_cast<T*>(a.data_ptr()),                              \
          reinterpret_cast<float*>(A_log.data_ptr()),                      \
          reinterpret_cast<T*>(dt_bias.data_ptr()),                        \
          reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),         \
          ssm_state_stride_0,                                               \
          token_indx,                                                       \
          cache_indices,                                                    \
          cache_indices_stride_0,                                           \
          num_accepted_tokens,                                              \
          num_spec_decodes,                                                 \
          num_spec_tokens,                                                  \
          num_k_heads,                                                      \
          head_k_dim,                                                       \
          num_v_heads,                                                      \
          head_v_dim);                                                      \
      break;                                                                \
    case 4:                                                                \
      kernel_launcher_decode_xe2_spec<T, state_scalar_t, 4>(               \
          queue,                                                            \
          reinterpret_cast<T*>(core_attn_out.data_ptr()),                  \
          reinterpret_cast<T*>(q.data_ptr()),                              \
          reinterpret_cast<T*>(k.data_ptr()),                              \
          reinterpret_cast<T*>(v.data_ptr()),                              \
          reinterpret_cast<T*>(b.data_ptr()),                              \
          reinterpret_cast<T*>(a.data_ptr()),                              \
          reinterpret_cast<float*>(A_log.data_ptr()),                      \
          reinterpret_cast<T*>(dt_bias.data_ptr()),                        \
          reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),         \
          ssm_state_stride_0,                                               \
          token_indx,                                                       \
          cache_indices,                                                    \
          cache_indices_stride_0,                                           \
          num_accepted_tokens,                                              \
          num_spec_decodes,                                                 \
          num_spec_tokens,                                                  \
          num_k_heads,                                                      \
          head_k_dim,                                                       \
          num_v_heads,                                                      \
          head_v_dim);                                                      \
      break;                                                                \
    case 8:                                                                \
      kernel_launcher_decode_xe2_spec<T, state_scalar_t, 8>(               \
          queue,                                                            \
          reinterpret_cast<T*>(core_attn_out.data_ptr()),                  \
          reinterpret_cast<T*>(q.data_ptr()),                              \
          reinterpret_cast<T*>(k.data_ptr()),                              \
          reinterpret_cast<T*>(v.data_ptr()),                              \
          reinterpret_cast<T*>(b.data_ptr()),                              \
          reinterpret_cast<T*>(a.data_ptr()),                              \
          reinterpret_cast<float*>(A_log.data_ptr()),                      \
          reinterpret_cast<T*>(dt_bias.data_ptr()),                        \
          reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),         \
          ssm_state_stride_0,                                               \
          token_indx,                                                       \
          cache_indices,                                                    \
          cache_indices_stride_0,                                           \
          num_accepted_tokens,                                              \
          num_spec_decodes,                                                 \
          num_spec_tokens,                                                  \
          num_k_heads,                                                      \
          head_k_dim,                                                       \
          num_v_heads,                                                      \
          head_v_dim);                                                      \
      break;                                                                \
    default:                                                                \
      TORCH_CHECK(false, "unsupported k_bucket_size: ", k_bucket_size);    \
  }

  if (ssm_state.scalar_type() == at::kFloat) {
    using state_scalar_t = float;
    XE2_DECODE_SPEC_BUCKET_DISPATCH(state_scalar_t)
  } else if (ssm_state.scalar_type() == at::kBFloat16) {
    using state_scalar_t = sycl::ext::oneapi::bfloat16;
    XE2_DECODE_SPEC_BUCKET_DISPATCH(state_scalar_t)
  } else if (ssm_state.scalar_type() == at::kHalf) {
    using state_scalar_t = sycl::half;
    XE2_DECODE_SPEC_BUCKET_DISPATCH(state_scalar_t)
  } else {
    TORCH_CHECK(
        false,
        "ssm_state dtype must be float32/float16/bfloat16, but got ",
        ssm_state.scalar_type());
  }

#undef XE2_DECODE_SPEC_BUCKET_DISPATCH
}

}  // namespace xe2
}  // namespace gdn