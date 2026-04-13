// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <sycl/sycl.hpp>
#include <iostream>
#include <numeric>

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"
#include "quantization/fp8/quant_utils.h"
#include <ATen/DeviceGuard.h>

namespace vllm {

template <typename scalar_t>
struct alignas(8) vec4_t {
  scalar_t val[4];
};

template <typename scalar_t, typename out_t, bool has_residual>
class rms_norm_dynamic_per_token_quant_kernel {
 public:
  rms_norm_dynamic_per_token_quant_kernel(
      out_t* __restrict__ out_,
      scalar_t* __restrict__ residual_,
      const scalar_t* __restrict__ input_,
      const scalar_t* __restrict__ weight_,
      const float* __restrict__ scale_ub_,
      float* __restrict__ scales_,
      const float epsilon_,
      const int hidden_size_)
      : out(out_),
        residual(residual_),
        input(input_),
        weight(weight_),
        scale_ub(scale_ub_),
        scales(scales_),
        epsilon(epsilon_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int tid = item.get_local_id(0);
    const int local_range = item.get_local_range(0);
    const int64_t token_idx = item.get_group(0);

    // s_local[0] = inv_rms,  s_local[1] = scale
    auto& s_local =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2]>(
            item.get_group());

    const scalar_t* token_input = input + token_idx * hidden_size;
    out_t* token_output = out + token_idx * hidden_size;
    scalar_t* token_residual = nullptr;
    if constexpr (has_residual) {
      token_residual = residual + token_idx * hidden_size;
    }

    // Pass 1: optional residual add + compute variance
    float variance = 0.0f;
    for (int i = tid; i < hidden_size; i += local_range) {
      float x = static_cast<float>(token_input[i]);
      if constexpr (has_residual) {
        x += static_cast<float>(token_residual[i]);
        token_residual[i] = static_cast<scalar_t>(x);
        // Read back the dtype-rounded value so variance is consistent with
        // what pass 2 will use when reading token_residual.
        x = static_cast<float>(token_residual[i]);
      }
      variance += x * x;
    }
    variance = sycl::reduce_over_group(
        item.get_group(), variance, sycl::plus<float>());
    if (tid == 0) {
      s_local[0] = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    sycl::group_barrier(item.get_group());
    const float inv_rms = s_local[0];

    // Pass 2: compute max |norm(x)| across the row → token scale
    float absmax = 0.0f;
    for (int i = tid; i < hidden_size; i += local_range) {
      const float x = has_residual ? static_cast<float>(token_residual[i])
                                   : static_cast<float>(token_input[i]);
      const float norm_x = x * inv_rms * static_cast<float>(weight[i]);
      absmax = sycl::max(absmax, sycl::fabs(norm_x));
    }
    absmax = sycl::reduce_over_group(
        item.get_group(), absmax, sycl::maximum<float>());

    if (tid == 0) {
      float computed_scale;
      if constexpr (std::is_same_v<out_t, int8_t>) {
        computed_scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
      } else {
        // FP8
        const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);
        const float clamped_absmax =
            (scale_ub != nullptr) ? sycl::min(absmax, *scale_ub) : absmax;
        computed_scale = sycl::max(
            clamped_absmax / fp8_max, fp8::min_scaling_factor<out_t>::val());
      }
      s_local[1] = computed_scale;
      scales[token_idx] = computed_scale;
    }
    sycl::group_barrier(item.get_group());
    const float inv_scale = 1.0f / s_local[1];

    // Pass 3: normalize and quantize
    for (int i = tid; i < hidden_size; i += local_range) {
      const float x = has_residual ? static_cast<float>(token_residual[i])
                                   : static_cast<float>(token_input[i]);
      const float norm_x = x * inv_rms * static_cast<float>(weight[i]);
      const float q = norm_x * inv_scale;

      if constexpr (std::is_same_v<out_t, int8_t>) {
        token_output[i] = static_cast<int8_t>(
            sycl::max(sycl::min(sycl::rint(q), 127.0f), -128.0f));
      } else {
        // FP8
        const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);
        token_output[i] =
            static_cast<out_t>(sycl::max(sycl::min(q, fp8_max), -fp8_max));
      }
    }
  }

 private:
  out_t* __restrict__ out;
  scalar_t* __restrict__ residual;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ weight;
  const float* __restrict__ scale_ub;
  float* __restrict__ scales;
  const float epsilon;
  const int hidden_size;
};

template <typename scalar_t, typename out_t, bool has_residual>
class rms_norm_per_block_quant_kernel {
 public:
  rms_norm_per_block_quant_kernel(
      out_t* __restrict__ out_,
      scalar_t* __restrict__ residual_,
      const scalar_t* __restrict__ input_,
      const scalar_t* __restrict__ weight_,
      float* __restrict__ scales_,
      const float epsilon_,
      const int hidden_size_,
      const int group_size_,
      const int num_tokens_,
      const int64_t scale_stride_token_,
      const int64_t scale_stride_group_)
      : out(out_),
        residual(residual_),
        input(input_),
        weight(weight_),
        scales(scales_),
        epsilon(epsilon_),
        hidden_size(hidden_size_),
        group_size(group_size_),
        num_tokens(num_tokens_),
        scale_stride_token(scale_stride_token_),
        scale_stride_group(scale_stride_group_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int tid = item.get_local_id(0);
    const int local_range = item.get_local_range(0);
    const int64_t token_idx = item.get_group(0);

    // s_local[0] = inv_rms,  s_local[1] = current group scale
    auto& s_local =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2]>(
            item.get_group());

    const scalar_t* token_input = input + token_idx * hidden_size;
    out_t* token_output = out + token_idx * hidden_size;
    scalar_t* token_residual = nullptr;
    if constexpr (has_residual) {
      token_residual = residual + token_idx * hidden_size;
    }

    // Pass 1: optional residual add + compute full-row variance
    float variance = 0.0f;
    for (int i = tid; i < hidden_size; i += local_range) {
      float x = static_cast<float>(token_input[i]);
      if constexpr (has_residual) {
        x += static_cast<float>(token_residual[i]);
        token_residual[i] = static_cast<scalar_t>(x);
        // Read back the dtype-rounded value so variance is consistent with
        // what passes 2+3 will use when reading token_residual.
        x = static_cast<float>(token_residual[i]);
      }
      variance += x * x;
    }
    variance = sycl::reduce_over_group(
        item.get_group(), variance, sycl::plus<float>());
    if (tid == 0) {
      s_local[0] = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    sycl::group_barrier(item.get_group());
    const float inv_rms = s_local[0];

    // Pass 2+3: for each column group, compute scale then quantize
    const int num_groups = hidden_size / group_size;
    for (int g = 0; g < num_groups; g++) {
      const int group_start = g * group_size;

      // Find max |norm(x)| within this group
      float group_absmax = 0.0f;
      for (int i = tid; i < group_size; i += local_range) {
        const int col = group_start + i;
        const float x = has_residual ? static_cast<float>(token_residual[col])
                                     : static_cast<float>(token_input[col]);
        const float norm_x = x * inv_rms * static_cast<float>(weight[col]);
        group_absmax = sycl::max(group_absmax, sycl::fabs(norm_x));
      }
      group_absmax = sycl::reduce_over_group(
          item.get_group(), group_absmax, sycl::maximum<float>());

      if (tid == 0) {
        float group_scale;
        if constexpr (std::is_same_v<out_t, int8_t>) {
          group_scale = (group_absmax > 0.0f) ? (group_absmax / 127.0f) : 1.0f;
        } else {
          // FP8
          const float fp8_max =
              static_cast<float>(fp8::quant_type_max_v<out_t>);
          group_scale = sycl::max(
              group_absmax / fp8_max, fp8::min_scaling_factor<out_t>::val());
        }
        s_local[1] = group_scale;
        const int64_t scale_idx = token_idx * scale_stride_token +
                                  static_cast<int64_t>(g) * scale_stride_group;
        scales[scale_idx] = group_scale;
      }
      sycl::group_barrier(item.get_group());
      const float inv_scale = 1.0f / s_local[1];

      // Quantize this group
      for (int i = tid; i < group_size; i += local_range) {
        const int col = group_start + i;
        const float x = has_residual ? static_cast<float>(token_residual[col])
                                     : static_cast<float>(token_input[col]);
        const float norm_x = x * inv_rms * static_cast<float>(weight[col]);
        const float q = norm_x * inv_scale;

        if constexpr (std::is_same_v<out_t, int8_t>) {
          token_output[col] = static_cast<int8_t>(
              sycl::max(sycl::min(sycl::rint(q), 127.0f), -128.0f));
        } else {
          // FP8
          const float fp8_max =
              static_cast<float>(fp8::quant_type_max_v<out_t>);
          token_output[col] =
              static_cast<out_t>(sycl::max(sycl::min(q, fp8_max), -fp8_max));
        }
      }
      sycl::group_barrier(item.get_group());
    }
  }

 private:
  out_t* __restrict__ out;
  scalar_t* __restrict__ residual;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ weight;
  float* __restrict__ scales;
  const float epsilon;
  const int hidden_size;
  const int group_size;
  const int num_tokens;
  const int64_t scale_stride_token;
  const int64_t scale_stride_group;
};

template <typename scalar_t, typename out_t, int VEC_SIZE>
class rms_norm_static_fp8_quant_kernel {
 public:
  rms_norm_static_fp8_quant_kernel(
      out_t* __restrict__ out_,
      const scalar_t* __restrict__ input_,
      const int input_stride_,
      const scalar_t* __restrict__ weight_,
      const float* __restrict__ scale_,
      const float epsilon_,
      const int hidden_size_)
      : out(out_),
        input(input_),
        input_stride(input_stride_),
        weight(weight_),
        scale(scale_),
        epsilon(epsilon_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    using vec_t = vllm::xpu::aligned_vec<scalar_t, VEC_SIZE>;

    const int tid = item.get_local_id(0);
    const int local_range = item.get_local_range(0);
    const int64_t token_idx = item.get_group(0);

    auto& s_variance =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
            item.get_group());

    const scalar_t* token_input = input + token_idx * input_stride;
    out_t* token_output = out + token_idx * hidden_size;
    const int nvec = hidden_size / VEC_SIZE;

    // Pass 1: compute variance — VEC_SIZE elements per work-item per iteration
    float variance = 0.0f;
    const auto* v_in = reinterpret_cast<const vec_t*>(token_input);
    for (int i = tid; i < nvec; i += local_range) {
      vec_t v = v_in[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(v[j]);
        variance += x * x;
      }
    }
    variance = sycl::reduce_over_group(
        item.get_group(), variance, sycl::plus<float>());
    if (tid == 0) {
      s_variance = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    sycl::group_barrier(item.get_group());
    const float inv_rms = s_variance;

    // Invert scale to avoid division
    const float scale_inv = 1.0f / (*scale);
    const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);

    // Pass 2: normalize, apply weight, quantize — same vectorization as Pass 1
    const auto* v_w = reinterpret_cast<const vec_t*>(weight);
    for (int i = tid; i < nvec; i += local_range) {
      vec_t src = v_in[i];
      vec_t wgt = v_w[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(src[j]);
        // Weight multiply in scalar_t precision to match unfused path
        float norm_x =
            static_cast<float>(static_cast<scalar_t>(x * inv_rms) * wgt[j]);
        float q = norm_x * scale_inv;
        token_output[i * VEC_SIZE + j] =
            static_cast<out_t>(sycl::max(sycl::min(q, fp8_max), -fp8_max));
      }
    }
  }

 private:
  out_t* __restrict__ out;
  const scalar_t* __restrict__ input;
  const int input_stride;
  const scalar_t* __restrict__ weight;
  const float* __restrict__ scale;
  const float epsilon;
  const int hidden_size;
};

template <typename scalar_t, typename out_t, int VEC_SIZE>
class fused_add_rms_norm_static_fp8_quant_kernel {
 public:
  fused_add_rms_norm_static_fp8_quant_kernel(
      out_t* __restrict__ out_,
      const scalar_t* __restrict__ input_,
      const int input_stride_,
      scalar_t* __restrict__ residual_,
      const scalar_t* __restrict__ weight_,
      const float* __restrict__ scale_,
      const float epsilon_,
      const int hidden_size_)
      : out(out_),
        input(input_),
        input_stride(input_stride_),
        residual(residual_),
        weight(weight_),
        scale(scale_),
        epsilon(epsilon_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    using vec_t = vllm::xpu::aligned_vec<scalar_t, VEC_SIZE>;

    const int tid = item.get_local_id(0);
    const int local_range = item.get_local_range(0);
    const int64_t token_idx = item.get_group(0);

    auto& s_variance =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
            item.get_group());

    const scalar_t* token_input = input + token_idx * input_stride;
    scalar_t* token_residual = residual + token_idx * hidden_size;
    out_t* token_output = out + token_idx * hidden_size;
    const int nvec = hidden_size / VEC_SIZE;

    // Pass 1: add residual + compute variance, VEC_SIZE elements per iteration
    float variance = 0.0f;
    const auto* v_in = reinterpret_cast<const vec_t*>(token_input);
    auto* v_res = reinterpret_cast<vec_t*>(token_residual);
    for (int i = tid; i < nvec; i += local_range) {
      vec_t inp = v_in[i];
      vec_t res = v_res[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        // Add in scalar_t precision to match fused_add_rms_norm kernel
        scalar_t z = inp[j] + res[j];
        res[j] = z;
        float xf = static_cast<float>(z);
        variance += xf * xf;
      }
      v_res[i] = res;  // write updated residual back
    }
    variance = sycl::reduce_over_group(
        item.get_group(), variance, sycl::plus<float>());
    if (tid == 0) {
      s_variance = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    sycl::group_barrier(item.get_group());
    const float inv_rms = s_variance;

    // Invert scale to avoid division
    const float scale_inv = 1.0f / (*scale);
    const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);

    // Pass 2: normalize from residual, apply weight, quantize
    const auto* v_w = reinterpret_cast<const vec_t*>(weight);
    for (int i = tid; i < nvec; i += local_range) {
      vec_t res = v_res[i];
      vec_t wgt = v_w[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(res[j]);
        // Weight multiply in scalar_t precision to match unfused path
        float norm_x =
            static_cast<float>(static_cast<scalar_t>(x * inv_rms) * wgt[j]);
        float q = norm_x * scale_inv;
        token_output[i * VEC_SIZE + j] =
            static_cast<out_t>(sycl::max(sycl::min(q, fp8_max), -fp8_max));
      }
    }
  }

 private:
  out_t* __restrict__ out;
  const scalar_t* __restrict__ input;
  const int input_stride;
  scalar_t* __restrict__ residual;
  const scalar_t* __restrict__ weight;
  const float* __restrict__ scale;
  const float epsilon;
  const int hidden_size;
};

template <typename scalar_t, typename out_t>
void call_rms_norm_dynamic_per_token_quant_kernel(
    torch::Tensor& out,
    std::optional<torch::Tensor>& residual,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    std::optional<torch::Tensor> const& scale_ub,
    torch::Tensor& scales,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int hidden_size = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_size;
  const int block_size = std::min(hidden_size, 1024);

  auto* out_ptr = out.data_ptr<out_t>();
  auto* input_ptr = input.data_ptr<scalar_t>();
  auto* weight_ptr = weight.data_ptr<scalar_t>();
  auto* scales_ptr = scales.data_ptr<float>();
  const float* scale_ub_ptr =
      scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr;

  auto& queue = vllm::xpu::vllmGetQueue();

  auto launch = [&](auto has_residual_tag) {
    constexpr bool has_residual = decltype(has_residual_tag)::value;
    sycl_t* residual_ptr = nullptr;
    if constexpr (has_residual) {
      residual_ptr = (sycl_t*)residual->data_ptr<scalar_t>();
    }
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(num_tokens * block_size, block_size),
          rms_norm_dynamic_per_token_quant_kernel<sycl_t, out_t, has_residual>(
              out_ptr,
              residual_ptr,
              (const sycl_t*)input_ptr,
              (const sycl_t*)weight_ptr,
              scale_ub_ptr,
              scales_ptr,
              epsilon,
              hidden_size));
    });
  };

  if (residual.has_value()) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

template <typename scalar_t, typename out_t>
void call_rms_norm_per_block_quant_kernel(
    torch::Tensor& out,
    std::optional<torch::Tensor>& residual,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor& scales,
    float epsilon,
    int group_size,
    bool is_scale_transposed) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int hidden_size = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_size;
  const int num_groups = hidden_size / group_size;
  const int block_size = std::min(hidden_size, 1024);

  // Compute scale strides based on transposition
  const int64_t scale_stride_token =
      is_scale_transposed ? 1 : static_cast<int64_t>(num_groups);
  const int64_t scale_stride_group = is_scale_transposed ? num_tokens : 1LL;

  auto* out_ptr = out.data_ptr<out_t>();
  auto* input_ptr = input.data_ptr<scalar_t>();
  auto* weight_ptr = weight.data_ptr<scalar_t>();
  auto* scales_ptr = scales.data_ptr<float>();

  auto& queue = vllm::xpu::vllmGetQueue();

  auto launch = [&](auto has_residual_tag) {
    constexpr bool has_residual = decltype(has_residual_tag)::value;
    sycl_t* residual_ptr = nullptr;
    if constexpr (has_residual) {
      residual_ptr = (sycl_t*)residual->data_ptr<scalar_t>();
    }
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(num_tokens * block_size, block_size),
          rms_norm_per_block_quant_kernel<sycl_t, out_t, has_residual>(
              out_ptr,
              residual_ptr,
              (const sycl_t*)input_ptr,
              (const sycl_t*)weight_ptr,
              scales_ptr,
              epsilon,
              hidden_size,
              group_size,
              static_cast<int>(num_tokens),
              scale_stride_token,
              scale_stride_group));
    });
  };

  if (residual.has_value()) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

template <typename scalar_t, typename out_t>
void call_rms_norm_static_fp8_quant_kernel(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& scale,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int hidden_size = input.size(-1);
  const int input_stride = input.stride(-2);
  const int64_t num_tokens = input.numel() / hidden_size;

  // Match CUDA: smaller blocks when num_tokens is large for better occupancy
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;

  // Dispatch VEC_SIZE = gcd(16 / sizeof(sycl_t), hidden_size) matching CUDA
  const int candidate_vec_size =
      std::gcd(static_cast<int>(16 / sizeof(sycl_t)), hidden_size);
  const int vec_size =
      (input_stride % candidate_vec_size == 0) ? candidate_vec_size : 1;
  const int block_size = std::min(hidden_size / vec_size, max_block_size);

  auto& queue = vllm::xpu::vllmGetQueue();

  auto launch = [&](auto vec_tag) {
    constexpr int VS = decltype(vec_tag)::value;
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(num_tokens * block_size, block_size),
          rms_norm_static_fp8_quant_kernel<sycl_t, out_t, VS>(
              out.data_ptr<out_t>(),
              (const sycl_t*)input.data_ptr<scalar_t>(),
              input_stride,
              (const sycl_t*)weight.data_ptr<scalar_t>(),
              scale.data_ptr<float>(),
              epsilon,
              hidden_size));
    });
  };

  // Dispatch on vec_size (gcd guarantees hidden_size % vec_size == 0)
  switch (vec_size) {
    case 8:
      launch(std::integral_constant<int, 8>{});
      break;
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    case 2:
      launch(std::integral_constant<int, 2>{});
      break;
    default:
      launch(std::integral_constant<int, 1>{});
      break;
  }
}

template <typename scalar_t, typename out_t>
void call_fused_add_rms_norm_static_fp8_quant_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor const& weight,
    torch::Tensor const& scale,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int hidden_size = input.size(-1);
  const int input_stride = input.stride(-2);
  const int64_t num_tokens = input.numel() / hidden_size;

  // Match CUDA: smaller blocks when num_tokens is large for better occupancy
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;

  // Dispatch VEC_SIZE = gcd(16 / sizeof(sycl_t), hidden_size) matching CUDA.
  // Also require input_stride % vec_size == 0 for safe vectorized input reads.
  int vec_size = std::gcd(static_cast<int>(16 / sizeof(sycl_t)), hidden_size);
  if (input_stride % vec_size != 0) {
    vec_size = 1;
  }
  const int block_size = std::min(hidden_size / vec_size, max_block_size);

  auto& queue = vllm::xpu::vllmGetQueue();

  auto launch = [&](auto vec_tag) {
    constexpr int VS = decltype(vec_tag)::value;
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(num_tokens * block_size, block_size),
          fused_add_rms_norm_static_fp8_quant_kernel<sycl_t, out_t, VS>(
              out.data_ptr<out_t>(),
              (const sycl_t*)input.data_ptr<scalar_t>(),
              input_stride,
              (sycl_t*)residual.data_ptr<scalar_t>(),
              (const sycl_t*)weight.data_ptr<scalar_t>(),
              scale.data_ptr<float>(),
              epsilon,
              hidden_size));
    });
  };

  // Dispatch on vec_size (gcd guarantees hidden_size % vec_size == 0)
  switch (vec_size) {
    case 8:
      launch(std::integral_constant<int, 8>{});
      break;
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    case 2:
      launch(std::integral_constant<int, 2>{});
      break;
    default:
      launch(std::integral_constant<int, 1>{});
      break;
  }
}

}  // namespace vllm

void rms_norm_dynamic_per_token_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor& scales,
    double const epsilon,
    std::optional<torch::Tensor> scale_ub,
    std::optional<torch::Tensor> residual) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(
      weight.dtype() == input.dtype(),
      "weight and input must have the same dtype");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      out.dtype() == torch::kFloat8_e4m3fn || out.dtype() == torch::kInt8,
      "output must be float8_e4m3fn or int8");
  if (scale_ub.has_value()) {
    TORCH_CHECK(
        out.dtype() == torch::kFloat8_e4m3fn,
        "scale_ub is only supported for FP8 output");
  }
  if (residual.has_value()) {
    TORCH_CHECK(
        residual->scalar_type() == input.scalar_type(),
        "residual and input must have the same dtype");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
  }

  if (out.dtype() == torch::kFloat8_e4m3fn) {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "rms_norm_dynamic_per_token_quant", [&] {
          vllm::call_rms_norm_dynamic_per_token_quant_kernel<
              scalar_t,
              at::Float8_e4m3fn>(
              out,
              residual,
              input,
              weight,
              scale_ub,
              scales,
              static_cast<float>(epsilon));
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "rms_norm_dynamic_per_token_quant_int8", [&] {
          vllm::call_rms_norm_dynamic_per_token_quant_kernel<scalar_t, int8_t>(
              out,
              residual,
              input,
              weight,
              scale_ub,
              scales,
              static_cast<float>(epsilon));
        });
  }
}

void rms_norm_per_block_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor& scales,
    double const epsilon,
    std::optional<torch::Tensor> scale_ub,
    std::optional<torch::Tensor> residual,
    int64_t group_size,
    bool is_scale_transposed) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(
      weight.dtype() == input.dtype(),
      "weight and input must have the same dtype");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      out.dtype() == torch::kFloat8_e4m3fn || out.dtype() == torch::kInt8,
      "output must be float8_e4m3fn or int8");
  TORCH_CHECK(
      input.size(-1) % group_size == 0,
      "hidden_size must be divisible by group_size");
  if (residual.has_value()) {
    TORCH_CHECK(
        residual->scalar_type() == input.scalar_type(),
        "residual and input must have the same dtype");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
  }

  if (out.dtype() == torch::kFloat8_e4m3fn) {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "rms_norm_per_block_quant", [&] {
          vllm::
              call_rms_norm_per_block_quant_kernel<scalar_t, at::Float8_e4m3fn>(
                  out,
                  residual,
                  input,
                  weight,
                  scales,
                  static_cast<float>(epsilon),
                  static_cast<int>(group_size),
                  is_scale_transposed);
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "rms_norm_per_block_quant_int8", [&] {
          vllm::call_rms_norm_per_block_quant_kernel<scalar_t, int8_t>(
              out,
              residual,
              input,
              weight,
              scales,
              static_cast<float>(epsilon),
              static_cast<int>(group_size),
              is_scale_transposed);
        });
  }
}

void rms_norm_static_fp8_quant(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& scale,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(
      weight.dtype() == input.dtype(),
      "weight and input must have the same dtype");

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_static_fp8_quant", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "rms_norm_static_fp8_quant_fp8", [&] {
              vllm::call_rms_norm_static_fp8_quant_kernel<scalar_t, fp8_t>(
                  out, input, weight, scale, static_cast<float>(epsilon));
            });
      });
}

void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& scale,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(
      residual.scalar_type() == input.scalar_type(),
      "residual and input must have the same dtype");
  TORCH_CHECK(
      weight.scalar_type() == input.scalar_type(),
      "weight and input must have the same dtype");

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "fused_add_rms_norm_static_fp8_quant", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "fused_add_rms_norm_static_fp8_quant_fp8", [&] {
              vllm::call_fused_add_rms_norm_static_fp8_quant_kernel<
                  scalar_t,
                  fp8_t>(
                  out,
                  input,
                  residual,
                  weight,
                  scale,
                  static_cast<float>(epsilon));
            });
      });
}
