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
template <typename scalar_t, int NUM_DIMS, int VEC_SIZE = 4, bool IS_GEMMA = false,
          bool HAS_Z = false, bool NORM_BEFORE_GATE = false>
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
      sycl::local_accessor<float, 1> s_variance_,
      const scalar_t* z_ = nullptr)
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
        s_variance(s_variance_),
        z(z_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    const scalar_t* input_row, * z_row;
    if constexpr (NUM_DIMS == 2) {
      // 2D for layernorm normal case [batch_size, hidden]
      input_row = input + item_ct1.get_group(2) * input_stride_d2;
      if constexpr (HAS_Z) {
        z_row = z + item_ct1.get_group(2) * input_stride_d2;
      }
    } else if constexpr (NUM_DIMS == 3) {
      // 3D for q/k norm [batch_size, num_heads, head_size]
      int batch_idx = item_ct1.get_group(2) / input_shape_d2;
      int head_idx = item_ct1.get_group(2) % input_shape_d2;
      input_row =
          input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
      if constexpr (HAS_Z) {
        z_row = z + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
      }
    } else if constexpr (NUM_DIMS == 4) {
      // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
      int batch_idx = item_ct1.get_group(2) / (input_shape_d3 * input_shape_d2);
      int remaining = item_ct1.get_group(2) % (input_shape_d3 * input_shape_d2);
      int seq_idx = remaining / input_shape_d2;
      int head_idx = remaining % input_shape_d2;
      input_row = input + batch_idx * input_stride_d4 +
                  seq_idx * input_stride_d3 + head_idx * input_stride_d2;
      if constexpr (HAS_Z) {
        z_row = z + batch_idx * input_stride_d4 +
                seq_idx * input_stride_d3 + head_idx * input_stride_d2;
      }
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

    auto silu_op = [](const scalar_t& val) {
      scalar_t x = val;
      x = x / ((scalar_t)1.0f + (scalar_t)sycl::exp((float)(-x)));
      return x;
    };

    constexpr int WIDTH = VEC_SIZE * sizeof(scalar_t);
    uintptr_t addr_in = reinterpret_cast<uintptr_t>(input_row);

    // fast path when the whole region is already aligned
    bool can_vec =
        ((addr_in & (WIDTH - 1)) == 0) && ((hidden_size & (VEC_SIZE - 1)) == 0);
    uintptr_t addr_z_in = 0;
    if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
      addr_z_in = reinterpret_cast<uintptr_t>(z_row);
      can_vec = can_vec && ((addr_z_in & (WIDTH - 1)) == 0);
    }
    if (can_vec) {
      int64_t const num_vec_elems = hidden_size / VEC_SIZE;
      auto const* vec_in = reinterpret_cast<const vec4_t<scalar_t>*>(input_row);
      const vec4_t<scalar_t>* vec_z;
      if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
        vec_z = reinterpret_cast<const vec4_t<scalar_t>*>(z_row);
      }
      for (int i = item_ct1.get_local_id(2); i < num_vec_elems;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> tmp = vec_in[i];
        if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
          vec4_t<scalar_t> z_tmp = vec_z[i];
          #pragma unroll
          for (int j = 0; j < VEC_SIZE; ++j) {
            tmp.val[j] = silu_op(z_tmp.val[j]) * tmp.val[j];
          }
        }
        vec_op(tmp);
      }
    } else {
      int misalignment_offset = addr_in & (WIDTH - 1);
      int alignment_bytes = WIDTH - misalignment_offset;
      int prefix_elems = alignment_bytes & (WIDTH - 1);
      prefix_elems /= sizeof(scalar_t);
      prefix_elems = prefix_elems < hidden_size ? prefix_elems : hidden_size;

      // 1. handle the possibly unaligned prefix with scalar access.
      for (int i = item_ct1.get_local_id(2); i < prefix_elems;
           i += item_ct1.get_local_range(2)) {
        auto tmp = input_row[i];
        if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
          auto z_tmp = z_row[i];
          tmp = silu_op(z_tmp) * tmp;
        }
        scalar_op(tmp);
      }

      int64_t const num_vec_elems = (hidden_size - prefix_elems) / VEC_SIZE;
      auto const* vec_in =
          reinterpret_cast<const vec4_t<scalar_t>*>(input_row + prefix_elems);
      const vec4_t<scalar_t>* vec_z;
      bool can_vec_z;
      if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
        int z_misalignment_offset = addr_z_in & (WIDTH - 1);
        can_vec_z = z_misalignment_offset == misalignment_offset;
        vec_z = reinterpret_cast<const vec4_t<scalar_t>*>(z_row + prefix_elems);
      }
      for (int i = item_ct1.get_local_id(2); i < num_vec_elems;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> tmp = vec_in[i];
        if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
          vec4_t<scalar_t> z_tmp;
          if (can_vec_z) {
            z_tmp = vec_z[i];
          } else {
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
              z_tmp.val[j] = (z_row + prefix_elems)[i * VEC_SIZE + j];
            }
          }
          #pragma unroll
          for (int j = 0; j < VEC_SIZE; ++j) {
            tmp.val[j] = silu_op(z_tmp.val[j]) * tmp.val[j];
          }
        }
        vec_op(tmp);
      }

      // 3. handle remaining tail elements.
      for (int i = item_ct1.get_local_id(2) + num_vec_elems * VEC_SIZE;
           i < hidden_size - prefix_elems;
           i += item_ct1.get_local_range(2)) {
        auto tmp = (input_row + prefix_elems)[i];
        if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
          auto z_tmp = (z_row + prefix_elems)[i];
          tmp = silu_op(z_tmp) * tmp;
        }
        scalar_op(tmp);
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
    uintptr_t addr_weight = reinterpret_cast<uintptr_t>(weight);
    uintptr_t addr_out = reinterpret_cast<uintptr_t>(out_row);
    bool can_vec_out = ((addr_in & (WIDTH - 1)) == 0) &&
                       ((addr_weight & (WIDTH - 1)) == 0) &&
                       ((addr_out & (WIDTH - 1)) == 0) &&
                       ((hidden_size & (VEC_SIZE - 1)) == 0);
    if constexpr (HAS_Z) {
      addr_z_in = reinterpret_cast<uintptr_t>(z_row);
      can_vec_out = can_vec_out && ((addr_z_in & (WIDTH - 1)) == 0);
    }
    if (can_vec_out) {
      auto* v_in = reinterpret_cast<const vec4_t<scalar_t>*>(input_row);
      const vec4_t<scalar_t>* v_z = nullptr;
      if constexpr (HAS_Z) {
        v_z = reinterpret_cast<const vec4_t<scalar_t>*>(z_row);
      }
      auto* v_w = reinterpret_cast<const vec4_t<scalar_t>*>(weight);
      auto* v_out = reinterpret_cast<vec4_t<scalar_t>*>(out_row);
      int64_t const out_num_vec_elems = hidden_size / VEC_SIZE;
      float s_variance_val = *s_variance_ptr;
      for (int idx = item_ct1.get_local_id(2); idx < out_num_vec_elems;
           idx += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> dst;
        vec4_t<scalar_t> src1 = v_in[idx];
        vec4_t<scalar_t> src2 = v_w[idx];
        vec4_t<scalar_t> z_src;
        if constexpr (HAS_Z) {
          z_src = v_z[idx];
        }
        for (int j = 0; j < VEC_SIZE; j++) {
          scalar_t x_val = src1.val[j];
          if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
            x_val *= silu_op(z_src.val[j]);
          }
          if constexpr (IS_GEMMA) {
            float w = (float)src2.val[j] + 1.0f;
            dst.val[j] = (scalar_t)(x_val * s_variance_val * w);
          } else {
            dst.val[j] = ((scalar_t)(x_val * s_variance_val)) * src2.val[j];
          }
          if constexpr (HAS_Z && NORM_BEFORE_GATE) {
            auto z_tmp = z_src.val[j];
            dst.val[j] *= silu_op(z_tmp);
          }
        }
        v_out[idx] = dst;
      }
    } else {
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        scalar_t x_val = input_row[idx];
        if constexpr (HAS_Z && !NORM_BEFORE_GATE) {
          x_val = silu_op(z_row[idx]) * x_val;
        }
        scalar_t out_val;
        if constexpr (IS_GEMMA) {
          float w = (float)weight[idx] + 1.0f;
          out_val = (scalar_t)(x_val * (*s_variance_ptr) * w);
        } else {
          out_val = ((scalar_t)(x_val * (*s_variance_ptr))) * weight[idx];
        }
        if constexpr (HAS_Z && NORM_BEFORE_GATE) {
          auto z_tmp = z_row[idx];
          out_val *= silu_op(z_tmp);
        }
        out_row[idx] = out_val;
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
  const scalar_t* __restrict__ z;       // [..., hidden_size] gating tensor (HAS_Z only)
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

template <typename scalar_t, bool IS_GEMMA = false,
          bool HAS_Z = false, bool NORM_BEFORE_GATE = false>
void call_rms_norm_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon,
    const scalar_t* z_ptr = nullptr) {
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
          vllm::rms_norm_kernel<sycl_t, tensor_rank, 4, IS_GEMMA, HAS_Z, NORM_BEFORE_GATE>(
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
              s_variance,
              (const sycl_t*)z_ptr));
    });
  });
}

template <typename scalar_t, bool IS_GEMMA = false>
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
      if constexpr (IS_GEMMA) {
        float w = (float)weight[idx] + 1.0f;
        input[item_ct1.get_group(2) * input_stride + idx] =
          (scalar_t)(x * (*s_variance_ptr) * w);
      } else {
        input[item_ct1.get_group(2) * input_stride + idx] =
          ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
      }
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

template <typename scalar_t, bool IS_GEMMA = false>
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
        fused_add_rms_norm_kernel<sycl_t, IS_GEMMA>(
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

void gemma_rms_norm(
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
  TORCH_CHECK(input.scalar_type() == weight.scalar_type());

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_gemma_rms_norm_kernel", [&] {
        vllm::call_rms_norm_kernel<scalar_t, true>(out, input, weight, epsilon);
      });
}

void fused_add_gemma_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    double epsilon) {
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(input.scalar_type() == weight.scalar_type());

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_fused_add_gemma_rms_norm_kernel", [&] {
        vllm::call_fused_add_rms_norm_kernel<scalar_t, true>(
            input, residual, weight, epsilon);
      });
}

void rms_norm_gated(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    std::optional<torch::Tensor> z,
    double epsilon,
    bool norm_before_gate,
    const std::string& activation) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(input.dim() >= 1);

  if (input.stride(-1) != 1 || !input.is_contiguous()) {
    input = input.contiguous();
  }
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.scalar_type() == input.scalar_type());
  TORCH_CHECK(out.numel() == input.numel());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  TORCH_CHECK(
      activation == "swish" || activation == "silu",
      "Unsupported rms_norm_gated activation '",
      activation,
      "'. Expected one of: swish, silu.");

  int64_t hidden_size = input.size(-1);
  TORCH_CHECK(weight.numel() == hidden_size);

  if (z.has_value() && z->defined()) {
    if (z->stride(-1) != 1 || !z->is_contiguous()) {
      z = z->contiguous();
    }
    TORCH_CHECK(z->sizes() == input.sizes());
    TORCH_CHECK(z->scalar_type() == input.scalar_type());
  }

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_rms_norm_gated_kernel", [&] {
        auto z_ptr =
            (z.has_value() && z->defined()) ? z->data_ptr<scalar_t>() : nullptr;
        if (z_ptr == nullptr) {
            vllm::call_rms_norm_kernel<scalar_t, false, false, false>(
                out, input, weight, epsilon, z_ptr);
        } else {
          if (norm_before_gate) {
            vllm::call_rms_norm_kernel<scalar_t, false, true, true>(
                out, input, weight, epsilon, z_ptr);
          } else {
            vllm::call_rms_norm_kernel<scalar_t, false, true, false>(
                out, input, weight, epsilon, z_ptr);
          }
        }
      });
}
