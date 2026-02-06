#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"

#include "quantization/fp8/fp8_quant.h"
#include "quantization/fp8/quant_utils.h"

namespace vllm {

// STRIDE_I_ZERO: true if scale_stride_i == 0 (per-tensor or per-channel)
// STRIDE_J_ZERO: true if scale_stride_j == 0 (per-tensor or per-token)
template <
    typename scalar_t,
    typename fp8_type,
    bool STRIDE_I_ZERO,
    bool STRIDE_J_ZERO>
class scaled_fp8_quant_kernel_strided_group_shape {
 private:
  fp8_type* out;
  const scalar_t* input;
  const float* scale;
  int hidden_size;
  int64_t in_row_stride;
  int64_t out_row_stride;
  int group_m;
  int group_n;
  int64_t scale_stride_i;
  int64_t scale_stride_j;

 public:
  scaled_fp8_quant_kernel_strided_group_shape(
      fp8_type* out_,
      const scalar_t* input_,
      const float* scale_,
      int hidden_size,
      int64_t in_row_stride_,
      int64_t out_row_stride_,
      int group_m_,
      int group_n_,
      int64_t scale_stride_i_,
      int64_t scale_stride_j_)
      : out(out_),
        input(input_),
        scale(scale_),
        hidden_size(hidden_size),
        in_row_stride(in_row_stride_),
        out_row_stride(out_row_stride_),
        group_m(group_m_),
        group_n(group_n_),
        scale_stride_i(scale_stride_i_),
        scale_stride_j(scale_stride_j_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    const int tid = item.get_local_id(0);

    const scalar_t* token_in = input + token_idx * in_row_stride;
    fp8_type* token_out = out + token_idx * out_row_stride;

    // Precompute row-level base offset for scale access (compile-time
    // eliminated
    // when STRIDE_I_ZERO)
    const int64_t scale_row_base =
        STRIDE_I_ZERO ? 0
                      : static_cast<int>(token_idx) / group_m * scale_stride_i;
    auto get_inv_scale = [&](int gj) {
      return 1.0f / scale[scale_row_base + gj * scale_stride_j];
    };

    int cached_gj = -1;
    float cached_inv_scale = 0.0f;
    auto get_inv_scale_cached = [&](int gj) {
      if (gj != cached_gj) {
        cached_inv_scale = 1.0f / scale[scale_row_base + gj * scale_stride_j];
        cached_gj = gj;
      }
      return cached_inv_scale;
    };

    constexpr int VEC_SIZE = 4;
    if (STRIDE_J_ZERO && hidden_size % VEC_SIZE == 0) {
      // Per-tensor or per-token: single scale per row, vectorize full row
      fp8::ConvertWithScaleOp<true, fp8_type> op{get_inv_scale(0)};
      fp8::scaled_convert_vec(
          token_in, token_out, hidden_size, tid, item.get_local_range(0), op);
    } else if (group_n % VEC_SIZE == 0) {
      // Multiple column groups with vectorization
      const int num_groups_n = hidden_size / group_n;

      for (int gj = 0; gj < num_groups_n; gj++) {
        fp8::ConvertWithScaleOp<true, fp8_type> op{get_inv_scale(gj)};
        fp8::scaled_convert_vec(
            token_in + gj * group_n,
            token_out + gj * group_n,
            group_n,
            tid,
            item.get_local_range(gj),
            op);
      }
    } else {
      // Scalar path for small column groups (group_n < VEC_SIZE)
      for (int n = tid; n < hidden_size; n += item.get_local_range(0)) {
        const int gj = n / group_n;
        fp8::ConvertWithScaleOp<true, fp8_type> op{get_inv_scale_cached(gj)};
        op(token_out[n], token_in[n]);
      }
    }
  }
};

template <typename scalar_t, typename fp8_type>
class scaled_fp8_quant_kernel_strided_dynamic {
 private:
  fp8_type* out;
  const scalar_t* input;
  const float* scale;
  int64_t hidden_size;
  int64_t in_row_stride;
  int64_t out_row_stride;

 public:
  scaled_fp8_quant_kernel_strided_dynamic(
      fp8_type* out_,
      const scalar_t* input_,
      const float* scale_,
      int hidden_size_,
      int64_t in_row_stride_,
      int64_t out_row_stride_)
      : out(out_),
        input(input_),
        scale(scale_),
        hidden_size(hidden_size_),
        in_row_stride(in_row_stride_),
        out_row_stride(out_row_stride_) {}
  void operator()(sycl::nd_item<1> item) const {
    int64_t token_idx = item.get_group(0);
    int tid = item.get_local_id(0);

    const scalar_t* token_in = input + token_idx * in_row_stride;
    fp8_type* token_out = out + token_idx * out_row_stride;
    // Invert the scale so that we can use multiplications to avoid expensive
    // division.
    const float inverted_scale = 1.0f / (*scale);
    fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
    fp8::scaled_convert_vec(
        token_in, token_out, hidden_size, tid, item.get_local_range(0), op);
  }
};

template <typename scalar_t, typename fp8_type>
class dynamic_per_token_scaled_fp8_quant_kernel {
 private:
  fp8_type* out;
  float* scale;
  scalar_t const* input;
  float const* scale_ub;
  const int hidden_size;

 public:
  dynamic_per_token_scaled_fp8_quant_kernel(
      fp8_type* out_,
      float* scale_,
      scalar_t const* input_,
      float const* scale_ub_,
      const int hidden_size_)
      : out(out_),
        scale(scale_),
        input(input_),
        scale_ub(scale_ub_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    int const tid = item.get_local_id(0);
    int const token_idx = item.get_group(0);

    // Use int64 to avoid overflowing an int32 when calculating this offset
    int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
    scalar_t const* token_input = &input[offset];
    fp8_type* token_output = &out[offset];

    // For vectorization, token_input and token_output pointers need to be
    // aligned at 8-byte and 4-byte addresses respectively.
    bool const can_vectorize = hidden_size % 4 == 0;

    float absmax_val = 0.0f;
    if (can_vectorize) {
      absmax_val = thread_max_vec(
          token_input, hidden_size, tid, item.get_local_range(0));
    } else {
      for (int i = tid; i < hidden_size; i += item.get_local_range(0)) {
        float const x = static_cast<float>(token_input[i]);
        absmax_val = sycl::max(absmax_val, sycl::fabs(x));
      }
    }

    float const block_absmax_val_maybe = sycl::reduce_over_group(
        item.get_group(), absmax_val, sycl::maximum<float>());
    // __shared__ float token_scale;
    auto& token_scale =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(
            item.get_group());
    if (tid == 0) {
      if (scale_ub) {
        token_scale[0] = sycl::min(block_absmax_val_maybe, *scale_ub);
      } else {
        token_scale[0] = block_absmax_val_maybe;
      }
      // token scale computation
      token_scale[0] = sycl::max(
          token_scale[0] / fp8::quant_type_max_v<fp8_type>,
          fp8::min_scaling_factor<fp8_type>::val());
      scale[token_idx] = token_scale[0];
    }
    group_barrier(item.get_group());

    // Note that we don't use inverted scales so we can match FBGemm impl.
    const float inverted_scale = 1.0f / (token_scale[0]);
    if (can_vectorize) {
      fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
      fp8::scaled_convert_vec(
          token_input,
          token_output,
          hidden_size,
          tid,
          item.get_local_range(0),
          op);
    } else {
      for (int i = tid; i < hidden_size; i += item.get_local_range(0)) {
        fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
        op(token_output[i], token_input[i]);
      }
    }
  }
};

}  // namespace vllm

void static_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor const& scale,  // various shapes
    std::optional<std::tuple<int64_t, int64_t>>
        opt_group_shape)  // optional explicit (group_m, group_n)
{
  TORCH_CHECK(
      input.stride(-1) == 1, "last dimension of input must be contiguous");
  TORCH_CHECK(
      out.stride(-1) == 1, "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);              // N (columns)
  const int num_tokens = input.numel() / hidden_size;  // M (rows)

  // Determine group_m, group_n, and scale strides from scale shape
  // Scale indexing: scale[gi * scale_stride_j + gj * scale_stride_i]
  // where gi = m / group_m, gj = n / group_n
  int group_m, group_n;
  int64_t scale_stride_i, scale_stride_j;

  if (scale.dim() == 0 || scale.numel() == 1) {
    // Per-tensor: one scale for the entire tensor
    group_m = num_tokens;
    group_n = hidden_size;
    scale_stride_i = 0;
    scale_stride_j = 0;
  } else if (scale.dim() == 1) {
    // 1D scale: require explicit group_shape to disambiguate per-channel vs
    // per-token (avoids edge case where num_tokens == hidden_size)
    TORCH_CHECK(
        opt_group_shape.has_value(),
        "1D scale requires explicit group_shape to disambiguate "
        "per-channel vs per-token quantization. "
        "Use group_shape=(-1, 1) for per-channel or group_shape=(1, "
        "-1) for per-token.");

    const auto& [opt_group_m, opt_group_n] = opt_group_shape.value();
    group_m = opt_group_m == -1 ? num_tokens : static_cast<int>(opt_group_m);
    group_n = opt_group_n == -1 ? hidden_size : static_cast<int>(opt_group_n);

    // Validate the explicit group shape matches the 1D scale
    const int64_t scale_len = scale.numel();
    const int64_t expected_scale_m = num_tokens / group_m;
    const int64_t expected_scale_n = hidden_size / group_n;
    const int64_t expected_scale_numel = expected_scale_m * expected_scale_n;

    TORCH_CHECK(
        scale_len == expected_scale_numel,
        "1D scale length (",
        scale_len,
        ") does not match expected size (",
        expected_scale_numel,
        ") for group_shape (",
        opt_group_m,
        ", ",
        opt_group_n,
        ") with input shape (",
        num_tokens,
        ", ",
        hidden_size,
        ")");

    // For 1D scale, determine strides based on which dim is trivial
    // Scale indexing: scale[gi * scale_stride_i + gj * scale_stride_j]
    // where gi = m / group_m (row group), gj = n / group_n (col group)
    if (expected_scale_m == 1) {
      // Per-channel style: one scale in M dim, scale varies along N
      // gi = 0 always, gj varies, so stride_1 traverses the scale
      scale_stride_i = 0;
      scale_stride_j = scale.stride(0);
    } else if (expected_scale_n == 1) {
      // Per-token style: one scale in N dim, scale varies along M
      // gj = 0 always, gi varies, so stride_0 traverses the scale
      scale_stride_i = scale.stride(0);
      scale_stride_j = 0;
    } else {
      TORCH_CHECK(
          false,
          "1D scale can only be used when one of the scale dimensions is 1. "
          "For 2D group scaling, use a 2D scale tensor.");
    }
  } else if (scale.dim() == 2) {
    // 2D scale: infer group sizes from scale dimensions (or use explicit if
    // provided)
    const int64_t scale_size_0 = scale.size(0);
    const int64_t scale_size_1 = scale.size(1);

    TORCH_CHECK(
        num_tokens % scale_size_0 == 0,
        "num_tokens (",
        num_tokens,
        ") must be divisible by scale.size(0) (",
        scale_size_0,
        ")");
    TORCH_CHECK(
        hidden_size % scale_size_1 == 0,
        "hidden_size (",
        hidden_size,
        ") must be divisible by scale.size(1) (",
        scale_size_1,
        ")");

    // Infer from 2D scale shape
    int inferred_group_m = num_tokens / scale_size_0;
    int inferred_group_n = hidden_size / scale_size_1;

    // Use explicit if provided, otherwise use inferred
    if (opt_group_shape.has_value()) {
      const auto& [opt_group_m, opt_group_n] = opt_group_shape.value();
      group_m = opt_group_m == -1 ? num_tokens : static_cast<int>(opt_group_m);
      group_n = opt_group_n == -1 ? hidden_size : static_cast<int>(opt_group_n);

      // Validate explicit matches inferred
      TORCH_CHECK(
          group_m == inferred_group_m && group_n == inferred_group_n,
          "Explicit group_shape (",
          opt_group_m,
          ", ",
          opt_group_n,
          ") does not match inferred group shape (",
          inferred_group_m,
          ", ",
          inferred_group_n,
          ") from 2D scale tensor shape (",
          scale_size_0,
          ", ",
          scale_size_1,
          ")");
    } else {
      group_m = inferred_group_m;
      group_n = inferred_group_n;
    }

    scale_stride_i = scale.stride(0);
    scale_stride_j = scale.stride(1);
  } else {
    TORCH_CHECK(
        false,
        "scale must be 0D, 1D, or 2D tensor, but got ",
        scale.dim(),
        "D");
  }

  int64_t num_elems = input.numel();
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(1024);

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto& queue = vllm::xpu::vllmGetQueue();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              VLLM_DISPATCH_BOOL(scale_stride_i == 0, S0_ZERO, [&] {
                VLLM_DISPATCH_BOOL(scale_stride_j == 0, S1_ZERO, [&] {
                  // Launch the kernel
                  queue.submit([&](sycl::handler& cgh) {
                    auto kernel =
                        vllm::scaled_fp8_quant_kernel_strided_group_shape<
                            scalar_t,
                            fp8_t,
                            S0_ZERO,
                            S1_ZERO>(
                            out.data_ptr<fp8_t>(),
                            input.data_ptr<scalar_t>(),
                            scale.data_ptr<float>(),
                            hidden_size,
                            in_row_stride,
                            out_row_stride,
                            group_m,
                            group_n,
                            scale_stride_i,
                            scale_stride_j);
                    cgh.parallel_for(
                        sycl::nd_range<1>(grid * block, block), kernel);
                  });
                });
              });
            });
      });
}

void dynamic_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor& scale)        // [1]
{
  TORCH_CHECK(
      input.stride(-1) == 1, "last dimension of input must be contiguous");
  TORCH_CHECK(
      out.stride(-1) == 1, "last dimension of output must be contiguous");

  int64_t hidden_size = input.size(-1);
  int64_t num_tokens = input.numel() / hidden_size;
  int64_t num_elems = input.numel();
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(1024);

  int64_t in_row_stride = input.stride(-2);
  int64_t out_row_stride = out.stride(-2);

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto& queue = vllm::xpu::vllmGetQueue();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              // Launch the kernel
              queue.submit([&](sycl::handler& cgh) {
                auto max_reduce_kernel =
                    vllm::segmented_max_reduction_strided<scalar_t, fp8_t>(
                        scale.data_ptr<float>(),
                        input.data_ptr<scalar_t>(),
                        hidden_size,
                        in_row_stride,
                        static_cast<int64_t>(num_tokens));
                cgh.parallel_for(
                    sycl::nd_range<1>(grid * block, block), max_reduce_kernel);
              });
              queue.submit([&](sycl::handler& cgh) {
                auto kernel = vllm::
                    scaled_fp8_quant_kernel_strided_dynamic<scalar_t, fp8_t>(
                        out.data_ptr<fp8_t>(),
                        input.data_ptr<scalar_t>(),
                        scale.data_ptr<float>(),
                        hidden_size,
                        in_row_stride,
                        out_row_stride);
                cgh.parallel_for(
                    sycl::nd_range<1>(grid * block, block), kernel);
              });
            });
      });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor& scales,
    std::optional<at::Tensor> const& scale_ub) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(std::min(hidden_size, 1024));

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto& queue = vllm::xpu::vllmGetQueue();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type",
      [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "dynamic_per_token_scaled_fp8_quant_kernel_fp8_type",
            [&] {
              // Launch the kernel
              queue
                  .submit([&](sycl::handler& cgh) {
                    auto kernel =
                        vllm::dynamic_per_token_scaled_fp8_quant_kernel<
                            scalar_t,
                            fp8_t>(
                            out.data_ptr<fp8_t>(),
                            scales.data_ptr<float>(),
                            input.data_ptr<scalar_t>(),
                            scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                                 : nullptr,
                            hidden_size);
                    cgh.parallel_for(
                        sycl::nd_range<1>(grid * block, block), kernel);
                  })
                  .wait();
            });
      });
}
