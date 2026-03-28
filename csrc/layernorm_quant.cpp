/*
 * SYCL kernels for fused quantized layernorm.
 *
 * These kernels correspond to the CUDA kernels in vllm's
 * layernorm_quant_kernels.cu. They perform RMS normalization and produce
 * FP8-quantized output directly, fusing the two operations into one kernel
 * pass to save memory bandwidth.
 *
 * Currently, only static FP8 quantization is supported.
 */

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>

#include "dispatch_utils.h"
#include "quantization/fp8/quant_utils.h"
#include "utils.h"

namespace vllm {

// ============================================================================
// rms_norm_static_fp8_quant_kernel
//
// Computes: out[i] = fp8_quant( rms_norm(input[i]) * weight[i] )
// where rms_norm(x) = x * rsqrt( mean(x^2) + epsilon )
// ============================================================================
template <typename scalar_t, typename fp8_type, int VEC_SIZE = 4>
class rms_norm_static_fp8_quant_kernel {
 public:
  rms_norm_static_fp8_quant_kernel(
      fp8_type* out_,
      const scalar_t* input_,
      const int input_stride_,
      const scalar_t* weight_,
      const float* scale_,
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        input_stride(input_stride_),
        weight(weight_),
        scale(scale_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    const scalar_t* input_row = input + item_ct1.get_group(2) * input_stride;

    // ---------- Phase 1: compute sum of squares ----------
    using vec_t = typename vllm::xpu::aligned_vec<scalar_t, VEC_SIZE>;
    constexpr int WIDTH = VEC_SIZE * sizeof(scalar_t);
    uintptr_t addr_in = reinterpret_cast<uintptr_t>(input_row);

    bool can_vec =
        ((addr_in & (WIDTH - 1)) == 0) && ((hidden_size & (VEC_SIZE - 1)) == 0);

    if (can_vec) {
      int64_t const num_vec_elems = hidden_size / VEC_SIZE;
      auto const* vec_in = reinterpret_cast<const vec_t*>(input_row);
      for (int i = item_ct1.get_local_id(2); i < num_vec_elems;
           i += item_ct1.get_local_range(2)) {
        vec_t tmp = vec_in[i];
        for (int j = 0; j < VEC_SIZE; ++j) {
          float x = static_cast<float>(tmp.val[j]);
          variance += x * x;
        }
      }
    } else {
      for (int i = item_ct1.get_local_id(2); i < hidden_size;
           i += item_ct1.get_local_range(2)) {
        float x = static_cast<float>(input_row[i]);
        variance += x * x;
      }
    }

    // ---------- Phase 2: work-group reduction ----------
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // ---------- Phase 3: normalize + quantize ----------
    // invert scale to avoid division
    float const scale_inv = 1.0f / (*scale);
    fp8::ConvertWithScaleOp<true, fp8_type> convert_op{scale_inv};

    if (can_vec) {
      auto* v_in = reinterpret_cast<const vec_t*>(input_row);
      auto* v_w = reinterpret_cast<const vec_t*>(weight);
      int64_t const num_vec_elems = hidden_size / VEC_SIZE;
      float s_var = *s_variance_ptr;

      for (int idx = item_ct1.get_local_id(2); idx < num_vec_elems;
           idx += item_ct1.get_local_range(2)) {
        vec_t src1 = v_in[idx];
        vec_t src2 = v_w[idx];
        for (int j = 0; j < VEC_SIZE; j++) {
          float x = static_cast<float>(src1.val[j]);
          float out_norm =
              static_cast<float>(static_cast<scalar_t>(x * s_var)) *
              static_cast<float>(src2.val[j]);
          fp8_type dst;
          convert_op(dst, out_norm);
          out[item_ct1.get_group(2) * hidden_size + idx * VEC_SIZE + j] = dst;
        }
      }
    } else {
      float s_var = *s_variance_ptr;
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        float x = static_cast<float>(input_row[idx]);
        float out_norm = static_cast<float>(static_cast<scalar_t>(x * s_var)) *
                         static_cast<float>(weight[idx]);
        fp8_type dst;
        convert_op(dst, out_norm);
        out[item_ct1.get_group(2) * hidden_size + idx] = dst;
      }
    }
  }

 private:
  fp8_type* __restrict__ out;          // [..., hidden_size]
  const scalar_t* __restrict__ input;  // [..., hidden_size]
  const int input_stride;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float* __restrict__ scale;      // [1]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

// ============================================================================
// fused_add_rms_norm_static_fp8_quant_kernel
//
// Computes:
//   residual[i] = input[i] + residual[i]
//   out[i] = fp8_quant( rms_norm(residual[i]) * weight[i] )
//
// This fuses the residual addition, RMS normalization and FP8 quantization
// into a single kernel to minimise memory traffic.
// ============================================================================
template <typename scalar_t, typename fp8_type, int VEC_SIZE = 4>
class fused_add_rms_norm_static_fp8_quant_kernel {
 public:
  fused_add_rms_norm_static_fp8_quant_kernel(
      fp8_type* out_,
      scalar_t* input_,
      const int input_stride_,
      scalar_t* residual_,
      const scalar_t* weight_,
      const float* scale_,
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        input_stride(input_stride_),
        residual(residual_),
        weight(weight_),
        scale(scale_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    // ---------- Phase 1: fused add + compute sum of squares ----------
    using vec_t = typename vllm::xpu::aligned_vec<scalar_t, VEC_SIZE>;
    constexpr int WIDTH = VEC_SIZE * sizeof(scalar_t);

    uintptr_t addr_in = reinterpret_cast<uintptr_t>(
        input + item_ct1.get_group(2) * input_stride);
    uintptr_t addr_res = reinterpret_cast<uintptr_t>(
        residual + item_ct1.get_group(2) * hidden_size);
    uintptr_t addr_w = reinterpret_cast<uintptr_t>(weight);

    bool can_vec = ((addr_in & (WIDTH - 1)) == 0) &&
                   ((addr_res & (WIDTH - 1)) == 0) &&
                   ((addr_w & (WIDTH - 1)) == 0) &&
                   ((hidden_size & (VEC_SIZE - 1)) == 0) &&
                   ((input_stride & (VEC_SIZE - 1)) == 0);

    if (can_vec) {
      int64_t const vec_hidden = hidden_size / VEC_SIZE;
      int64_t const vec_stride = input_stride / VEC_SIZE;

      auto* input_v = reinterpret_cast<vec_t*>(
          input + item_ct1.get_group(2) * input_stride);
      auto* residual_v = reinterpret_cast<vec_t*>(
          residual + item_ct1.get_group(2) * hidden_size);

      for (int idx = item_ct1.get_local_id(2); idx < vec_hidden;
           idx += item_ct1.get_local_range(2)) {
        vec_t in_vec = input_v[idx];
        vec_t res_vec = residual_v[idx];
        for (int j = 0; j < VEC_SIZE; ++j) {
          float z = static_cast<float>(in_vec.val[j]) +
                    static_cast<float>(res_vec.val[j]);
          variance += z * z;
          res_vec.val[j] = static_cast<scalar_t>(z);
        }
        residual_v[idx] = res_vec;
      }
    } else {
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        scalar_t z = input[item_ct1.get_group(2) * input_stride + idx];
        z += residual[item_ct1.get_group(2) * hidden_size + idx];
        float x = static_cast<float>(z);
        variance += x * x;
        residual[item_ct1.get_group(2) * hidden_size + idx] = z;
      }
    }

    // ---------- Phase 2: work-group reduction ----------
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // ---------- Phase 3: normalize + quantize ----------
    float const scale_inv = 1.0f / (*scale);
    fp8::ConvertWithScaleOp<true, fp8_type> convert_op{scale_inv};
    float s_var = *s_variance_ptr;

    if (can_vec) {
      int64_t const vec_hidden = hidden_size / VEC_SIZE;
      auto* residual_v = reinterpret_cast<const vec_t*>(
          residual + item_ct1.get_group(2) * hidden_size);
      auto* weight_v = reinterpret_cast<const vec_t*>(weight);

      for (int idx = item_ct1.get_local_id(2); idx < vec_hidden;
           idx += item_ct1.get_local_range(2)) {
        vec_t res_vec = residual_v[idx];
        vec_t w_vec = weight_v[idx];
        for (int j = 0; j < VEC_SIZE; j++) {
          float x = static_cast<float>(res_vec.val[j]);
          float out_norm =
              static_cast<float>(static_cast<scalar_t>(x * s_var)) *
              static_cast<float>(w_vec.val[j]);
          fp8_type dst;
          convert_op(dst, out_norm);
          out[item_ct1.get_group(2) * hidden_size + idx * VEC_SIZE + j] = dst;
        }
      }
    } else {
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        float x = static_cast<float>(
            residual[item_ct1.get_group(2) * hidden_size + idx]);
        float out_norm = static_cast<float>(static_cast<scalar_t>(x * s_var)) *
                         static_cast<float>(weight[idx]);
        fp8_type dst;
        convert_op(dst, out_norm);
        out[item_ct1.get_group(2) * hidden_size + idx] = dst;
      }
    }
  }

 private:
  fp8_type* __restrict__ out;    // [..., hidden_size]
  scalar_t* __restrict__ input;  // [..., hidden_size]
  const int input_stride;
  scalar_t* __restrict__ residual;      // [..., hidden_size]
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float* __restrict__ scale;      // [1]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

}  // namespace vllm

// ============================================================================
// Host-side entry points
// ============================================================================

void rms_norm_static_fp8_quant(
    torch::Tensor& out,     // [..., hidden_size]  fp8
    torch::Tensor& input,   // [..., hidden_size]
    torch::Tensor& weight,  // [hidden_size]
    torch::Tensor& scale,   // [1]
    double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  int hidden_size = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, max_block_size));

  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_static_fp8_quant_scalar_type", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "rms_norm_static_fp8_quant_fp8_type", [&] {
              queue.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> s_variance(
                    sycl::range<1>(1), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    vllm::rms_norm_static_fp8_quant_kernel<sycl_t, fp8_t>(
                        out.data_ptr<fp8_t>(),
                        reinterpret_cast<const sycl_t*>(
                            input.data_ptr<scalar_t>()),
                        input_stride,
                        reinterpret_cast<const sycl_t*>(
                            weight.data_ptr<scalar_t>()),
                        scale.data_ptr<float>(),
                        static_cast<float>(epsilon),
                        num_tokens,
                        hidden_size,
                        s_variance));
              });
            });
      });
}

void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,       // [..., hidden_size]  fp8
    torch::Tensor& input,     // [..., hidden_size]
    torch::Tensor& residual,  // [..., hidden_size]
    torch::Tensor& weight,    // [hidden_size]
    torch::Tensor& scale,     // [1]
    double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(residual.scalar_type() == input.scalar_type());
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());

  int hidden_size = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  const int max_block_size = (num_tokens < 256) ? 1024 : 256;

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, max_block_size));

  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "fused_add_rms_norm_static_fp8_quant_scalar_type",
      [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "fused_add_rms_norm_static_fp8_quant_fp8_type",
            [&] {
              queue.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> s_variance(
                    sycl::range<1>(1), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    vllm::fused_add_rms_norm_static_fp8_quant_kernel<
                        sycl_t,
                        fp8_t>(
                        out.data_ptr<fp8_t>(),
                        reinterpret_cast<sycl_t*>(input.data_ptr<scalar_t>()),
                        input_stride,
                        reinterpret_cast<sycl_t*>(
                            residual.data_ptr<scalar_t>()),
                        reinterpret_cast<const sycl_t*>(
                            weight.data_ptr<scalar_t>()),
                        scale.data_ptr<float>(),
                        static_cast<float>(epsilon),
                        num_tokens,
                        hidden_size,
                        s_variance));
              });
            });
      });
}
