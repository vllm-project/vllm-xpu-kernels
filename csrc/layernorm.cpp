#include <sycl/sycl.hpp>

#include <algorithm>
#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t>
struct alignas(8) vec4_t {
  scalar_t val[4];
};

// The vector width is fixed at 4 to avoid excessive branching in the kernel,
// which could degrade performance.
template <typename scalar_t, int NUM_DIMS, int VEC_SIZE = 4>
class rms_norm_kernel {
 public:
  rms_norm_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const int64_t input_stride_d2_,  // input.stride(-2)
      const int64_t input_stride_d3_,  // input.stride(-3)
      const int64_t input_stride_d4_,  // input.stride(-4)
      const int64_t input_shape_d2_,   // input.size(-2)
      const int64_t input_shape_d3_,   // input.size(-3)
      const scalar_t* weight_,
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        input_stride_d2(input_stride_d2_),
        input_stride_d3(input_stride_d3_),
        input_stride_d4(input_stride_d4_),
        input_shape_d2(input_shape_d2_),
        input_shape_d3(input_shape_d3_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    const scalar_t* input_row;
    if constexpr (NUM_DIMS == 2) {
      // 2D for layernorm normal case [batch_size, hidden]
      input_row = input + item_ct1.get_group(2) * input_stride_d2;
    } else if constexpr (NUM_DIMS == 3) {
      // 3D for q/k norm [batch_size, num_heads, head_size]
      int batch_idx = item_ct1.get_group(2) / input_shape_d2;
      int head_idx = item_ct1.get_group(2) % input_shape_d2;
      input_row =
          input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    } else if constexpr (NUM_DIMS == 4) {
      // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
      int batch_idx = item_ct1.get_group(2) / (input_shape_d3 * input_shape_d2);
      int remaining = item_ct1.get_group(2) % (input_shape_d3 * input_shape_d2);
      int seq_idx = remaining / input_shape_d2;
      int head_idx = remaining % input_shape_d2;
      input_row = input + batch_idx * input_stride_d4 +
                  seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    }

    auto vec_op = [&variance](
                      const vec4_t<scalar_t>& vec, int vec_size = VEC_SIZE) {
      for (int i = 0; i < vec_size; ++i) {
        float x = static_cast<float>(vec.val[i]);
        variance += x * x;
      }
    };
    auto scalar_op = [&variance](const scalar_t& val) {
      float x = static_cast<float>(val);
      variance += x * x;
    };

    constexpr int WIDTH = VEC_SIZE * sizeof(scalar_t);
    uintptr_t addr = reinterpret_cast<uintptr_t>(input_row);

    // fast path when the whole region is already aligned
    bool can_vec =
        ((addr & (WIDTH - 1)) == 0) && ((hidden_size & (VEC_SIZE - 1)) == 0);
    if (can_vec) {
      int64_t const num_vec_elems = hidden_size / VEC_SIZE;
      auto const* vec_in = reinterpret_cast<const vec4_t<scalar_t>*>(input_row);
      for (int i = item_ct1.get_local_id(2); i < num_vec_elems;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> tmp = vec_in[i];
        vec_op(tmp);
      }
    } else {
      int misalignment_offset = addr & (WIDTH - 1);
      int alignment_bytes = WIDTH - misalignment_offset;
      int prefix_elems = alignment_bytes & (WIDTH - 1);
      prefix_elems /= sizeof(scalar_t);
      prefix_elems = prefix_elems < hidden_size ? prefix_elems : hidden_size;

      // 1. handle the possibly unaligned prefix with scalar access.
      for (int i = item_ct1.get_local_id(2); i < prefix_elems;
           i += item_ct1.get_local_range(2)) {
        scalar_op(input_row[i]);
      }

      int64_t const num_vec_elems = (hidden_size - prefix_elems) / VEC_SIZE;
      auto const* vec_in =
          reinterpret_cast<const vec4_t<scalar_t>*>(input_row + prefix_elems);
      for (int i = item_ct1.get_local_id(2); i < num_vec_elems;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> tmp = vec_in[i];
        vec_op(tmp);
      }

      // 3. handle remaining tail elements.
      for (int i = item_ct1.get_local_id(2) + num_vec_elems * VEC_SIZE;
           i < hidden_size - prefix_elems;
           i += item_ct1.get_local_range(2)) {
        scalar_op(input_row[i]);
      }
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    scalar_t* out_row = out + item_ct1.get_group(2) * hidden_size;
    addr = reinterpret_cast<uintptr_t>(out_row);
    can_vec =
        ((addr & (WIDTH - 1)) == 0) && ((hidden_size & (VEC_SIZE - 1)) == 0);
    if (can_vec) {
      auto* v_in = reinterpret_cast<const vec4_t<scalar_t>*>(input_row);
      auto* v_w = reinterpret_cast<const vec4_t<scalar_t>*>(weight);
      auto* v_out = reinterpret_cast<vec4_t<scalar_t>*>(out_row);
      int64_t const out_num_vec_elems = hidden_size / VEC_SIZE;
      for (int idx = item_ct1.get_local_id(2); idx < out_num_vec_elems;
           idx += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> dst;
        vec4_t<scalar_t> src1 = v_in[idx];
        vec4_t<scalar_t> src2 = v_w[idx];
        for (int j = 0; j < VEC_SIZE; j++) {
          float x = static_cast<float>(src1.val[j]);
          dst.val[j] = ((scalar_t)(x * (*s_variance_ptr))) * src2.val[j];
        }
        v_out[idx] = dst;
      }
    } else {
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        float x = (float)input_row[idx];
        out_row[idx] = ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
      }
    }
  }

 private:
  scalar_t* __restrict__ out;          // [..., hidden_size]
  const scalar_t* __restrict__ input;  // [..., hidden_size]
  const int64_t input_stride_d2;
  const int64_t input_stride_d3;
  const int64_t input_stride_d4;
  const int64_t input_shape_d2;
  const int64_t input_shape_d3;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

template <typename scalar_t>
void call_rms_norm_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int num_dims = input.dim();
  int64_t input_stride_d2 = input.stride(-2);
  int64_t input_stride_d3 = (num_dims >= 3) ? input.stride(-3) : 0;
  int64_t input_stride_d4 = (num_dims >= 4) ? input.stride(-4) : 0;
  int64_t input_shape_d2 = (num_dims >= 3) ? input.size(-2) : 0;
  int64_t input_shape_d3 = (num_dims >= 4) ? input.size(-3) : 0;

  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_RANK234(num_dims, [&]() {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::rms_norm_kernel<sycl_t, tensor_rank>(
              (sycl_t*)out_ptr,
              (const sycl_t*)input_ptr,
              input_stride_d2,
              input_stride_d3,
              input_stride_d4,
              input_shape_d2,
              input_shape_d3,
              (const sycl_t*)weight_ptr,
              epsilon,
              num_tokens,
              hidden_size,
              s_variance));
    });
  });
}

template <typename scalar_t>
class fused_add_rms_norm_kernel {
 public:
  fused_add_rms_norm_kernel(
      scalar_t* __restrict__ input_,     // [..., hidden_size]
      scalar_t* __restrict__ residual_,  // [..., hidden_size]
      const int64_t input_stride_,
      const scalar_t* __restrict__ weight_,  // [hidden_size]
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : input(input_),
        residual(residual_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      scalar_t z = (scalar_t)input[item_ct1.get_group(2) * input_stride + idx];
      z += residual[item_ct1.get_group(2) * hidden_size + idx];
      float x = (float)z;
      variance += x * x;
      residual[item_ct1.get_group(2) * hidden_size + idx] = z;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      float x = (float)residual[item_ct1.get_group(2) * hidden_size + idx];
      input[item_ct1.get_group(2) * input_stride + idx] =
          ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
    }
  }

 private:
  scalar_t* __restrict__ input;     // [..., hidden_size]
  scalar_t* __restrict__ residual;  // [..., hidden_size]
  const int64_t input_stride;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;  // local memory for variance
};

template <typename scalar_t>
void call_fused_add_rms_norm_kernel(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  auto input_ptr = input.data_ptr<scalar_t>();
  auto residual_ptr = residual.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  int64_t input_stride = input.stride(-2);
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        fused_add_rms_norm_kernel<sycl_t>(
            (sycl_t*)input_ptr,
            (sycl_t*)residual_ptr,
            input_stride,
            (const sycl_t*)weight_ptr,
            epsilon,
            num_tokens,
            hidden_size,
            s_variance));
  });
}

}  // namespace vllm

void rms_norm(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = input.contiguous();
  }
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_rms_norm_kernel", [&] {
        vllm::call_rms_norm_kernel<scalar_t>(out, input, weight, epsilon);
      });
}

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_fused_add_rms_norm_kernel", [&] {
        vllm::call_fused_add_rms_norm_kernel<scalar_t>(
            input, residual, weight, epsilon);
      });
}