#include <sycl/sycl.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <ATen/DeviceGuard.h>
#include "utils.h"
#include "dispatch_utils.h"
#include "quantization/utils.h"

// Fused gated RMSNorm.
//
//   out = rms_norm(input) * weight * act(gate)
//
// with `act` being either `sigmoid(g)` or `swish(g) = g * sigmoid(g)`.
//
// This is the output normalization used by linear-attention layers: KDA
// (Kimi-Linear / Kimi-K3) uses the `sigmoid` variant while the Gated DeltaNet
// family (Qwen3-Next, Olmo3) uses `swish`. It matches
// `fla.modules.FusedRMSNormGated` / vLLM's `RMSNormGated`, including their
// numerics: the normalization, the weight scaling and the gate are all applied
// in fp32 and the result is rounded to the input dtype exactly once.
//
// Normalization is always over the last dimension, so a `[num_tokens,
// num_heads, head_dim]` activation normalizes per (token, head) with a
// `[head_dim]` weight shared across heads.

namespace vllm {

enum class GateActivation { Sigmoid, Swish };

template <GateActivation ACT>
inline float apply_gate_activation(float g) {
  const float s = 1.0f / (1.0f + sycl::exp(-g));
  if constexpr (ACT == GateActivation::Swish) {
    return g * s;
  } else {
    return s;
  }
}

// One work-group per row, vectorized loads/stores.
template <typename scalar_t, int VEC_SIZE, bool HasWeight, GateActivation ACT>
class rms_norm_gated_kernel {
 public:
  rms_norm_gated_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const scalar_t* gate_,
      const scalar_t* weight_,
      const float epsilon_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        gate(gate_),
        weight(weight_),
        epsilon(epsilon_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    using vec_t = vec_n_t<scalar_t, VEC_SIZE>;

    const int64_t row = item_ct1.get_group(2);
    const int64_t row_offset = row * hidden_size;
    const int tid = item_ct1.get_local_id(2);
    const int num_threads = item_ct1.get_local_range(2);
    const int64_t num_vec_elems = hidden_size / VEC_SIZE;

    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();

    auto const* v_in = reinterpret_cast<const vec_t*>(input + row_offset);
    auto const* v_g = reinterpret_cast<const vec_t*>(gate + row_offset);
    auto const* v_w = reinterpret_cast<const vec_t*>(weight);
    auto* v_out = reinterpret_cast<vec_t*>(out + row_offset);

    float variance = 0.0f;
    for (int i = tid; i < num_vec_elems; i += num_threads) {
      vec_t tmp = v_in[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float x = static_cast<float>(tmp.val[j]);
        variance += x * x;
      }
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (tid == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float rstd = *s_variance_ptr;

    for (int idx = tid; idx < num_vec_elems; idx += num_threads) {
      vec_t dst;
      vec_t src_x = v_in[idx];
      vec_t src_g = v_g[idx];
      vec_t src_w;
      if constexpr (HasWeight) {
        src_w = v_w[idx];
      }
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float y = static_cast<float>(src_x.val[j]) * rstd;
        if constexpr (HasWeight) {
          y *= static_cast<float>(src_w.val[j]);
        }
        y *= apply_gate_activation<ACT>(static_cast<float>(src_g.val[j]));
        dst.val[j] = static_cast<scalar_t>(y);
      }
      v_out[idx] = dst;
    }
  }

 private:
  scalar_t* __restrict__ out;           // [num_tokens, hidden_size]
  const scalar_t* __restrict__ input;   // [num_tokens, hidden_size]
  const scalar_t* __restrict__ gate;    // [num_tokens, hidden_size]
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

// Scalar fallback for hidden sizes / pointers that cannot be vectorized.
template <typename scalar_t, bool HasWeight, GateActivation ACT>
class rms_norm_gated_scalar_kernel {
 public:
  rms_norm_gated_scalar_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const scalar_t* gate_,
      const scalar_t* weight_,
      const float epsilon_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        gate(gate_),
        weight(weight_),
        epsilon(epsilon_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    const int64_t row_offset =
        static_cast<int64_t>(item_ct1.get_group(2)) * hidden_size;
    const int tid = item_ct1.get_local_id(2);
    const int num_threads = item_ct1.get_local_range(2);

    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();

    float variance = 0.0f;
    for (int i = tid; i < hidden_size; i += num_threads) {
      float x = static_cast<float>(input[row_offset + i]);
      variance += x * x;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (tid == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float rstd = *s_variance_ptr;

    for (int i = tid; i < hidden_size; i += num_threads) {
      float y = static_cast<float>(input[row_offset + i]) * rstd;
      if constexpr (HasWeight) {
        y *= static_cast<float>(weight[i]);
      }
      y *= apply_gate_activation<ACT>(static_cast<float>(gate[row_offset + i]));
      out[row_offset + i] = static_cast<scalar_t>(y);
    }
  }

 private:
  scalar_t* __restrict__ out;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ gate;
  const scalar_t* __restrict__ weight;
  const float epsilon;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

// Multi-row kernel: a single row of `head_dim` elements is far too small to
// fill a work-group, so pack ROWS_PER_WG rows into one work-group. Organized
// as (ROWS_PER_WG, 1, items_per_row); dim 0 selects the row, dim 2 the column.
template <
    typename scalar_t,
    int VEC_SIZE,
    int ROWS_PER_WG,
    bool HasWeight,
    GateActivation ACT>
class rms_norm_gated_multi_row_kernel {
 public:
  rms_norm_gated_multi_row_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const scalar_t* gate_,
      const scalar_t* weight_,
      const float epsilon_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        gate(gate_),
        weight(weight_),
        epsilon(epsilon_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    using vec_t = vec_n_t<scalar_t, VEC_SIZE>;

    const int row_in_wg = item_ct1.get_local_id(0);
    const int col_id = item_ct1.get_local_id(2);
    const int col_range = item_ct1.get_local_range(2);
    // Valid by construction: dispatch requires num_tokens % ROWS_PER_WG == 0.
    const int64_t global_row =
        static_cast<int64_t>(item_ct1.get_group(2)) * ROWS_PER_WG + row_in_wg;
    const int64_t row_offset = global_row * hidden_size;
    const int64_t num_vec_elems = hidden_size / VEC_SIZE;

    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();

    auto const* v_in = reinterpret_cast<const vec_t*>(input + row_offset);
    auto const* v_g = reinterpret_cast<const vec_t*>(gate + row_offset);
    auto const* v_w = reinterpret_cast<const vec_t*>(weight);
    auto* v_out = reinterpret_cast<vec_t*>(out + row_offset);

    float variance = 0.0f;
    for (int i = col_id; i < num_vec_elems; i += col_range) {
      vec_t tmp = v_in[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float x = static_cast<float>(tmp.val[j]);
        variance += x * x;
      }
    }

    // Lanes of one row are contiguous within a sub-group, so reduce over the
    // `col_range` lanes that belong to this row.
    auto sg = item_ct1.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int row_lane_offset = lane % col_range;
    for (int offset = col_range / 2; offset > 0; offset >>= 1) {
      float other = sycl::shift_group_left(sg, variance, offset);
      if (row_lane_offset < offset) variance += other;
    }
    if (row_lane_offset == 0) {
      s_variance_ptr[row_in_wg] = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float rstd = s_variance_ptr[row_in_wg];

    for (int idx = col_id; idx < num_vec_elems; idx += col_range) {
      vec_t dst;
      vec_t src_x = v_in[idx];
      vec_t src_g = v_g[idx];
      vec_t src_w;
      if constexpr (HasWeight) {
        src_w = v_w[idx];
      }
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float y = static_cast<float>(src_x.val[j]) * rstd;
        if constexpr (HasWeight) {
          y *= static_cast<float>(src_w.val[j]);
        }
        y *= apply_gate_activation<ACT>(static_cast<float>(src_g.val[j]));
        dst.val[j] = static_cast<scalar_t>(y);
      }
      v_out[idx] = dst;
    }
  }

 private:
  scalar_t* __restrict__ out;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ gate;
  const scalar_t* __restrict__ weight;
  const float epsilon;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

template <typename scalar_t, bool HasWeight, GateActivation ACT>
void call_rms_norm_gated_kernel(
    torch::Tensor& out,
    const torch::Tensor& input,
    const torch::Tensor& gate,
    const scalar_t* weight_ptr,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  const int hidden_size = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_size;

  auto* out_ptr = out.data_ptr<scalar_t>();
  const auto* input_ptr = input.data_ptr<scalar_t>();
  const auto* gate_ptr = gate.data_ptr<scalar_t>();

  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  auto& queue = vllm::xpu::vllmGetQueue();

  constexpr int vec_size = (sizeof(scalar_t) == 2) ? 8 : 4;
  constexpr int req_alignment_bytes = vec_size * sizeof(scalar_t);
  auto aligned = [](const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % req_alignment_bytes == 0;
  };
  const bool can_vec = (hidden_size % vec_size == 0) && aligned(input_ptr) &&
                       aligned(gate_ptr) && aligned(out_ptr) &&
                       (weight_ptr == nullptr || aligned(weight_ptr));

  if (can_vec) {
    const int items_per_row = hidden_size / vec_size;
    constexpr int subgroup_size = 32;
    constexpr int ROWS_PER_WG = 16;
    if (items_per_row <= subgroup_size && num_tokens % ROWS_PER_WG == 0 &&
        subgroup_size % items_per_row == 0) {
      const int64_t num_groups = num_tokens / ROWS_PER_WG;
      sycl::range<3> block(ROWS_PER_WG, 1, items_per_row);
      sycl::range<3> grid(1, 1, num_groups);
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> s_variance(
            sycl::range<1>(ROWS_PER_WG), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            rms_norm_gated_multi_row_kernel<
                sycl_t,
                vec_size,
                ROWS_PER_WG,
                HasWeight,
                ACT>(
                (sycl_t*)out_ptr,
                (const sycl_t*)input_ptr,
                (const sycl_t*)gate_ptr,
                (const sycl_t*)weight_ptr,
                epsilon,
                hidden_size,
                s_variance));
      });
      return;
    }

    sycl::range<3> grid(1, 1, num_tokens);
    sycl::range<3> block(1, 1, std::min(items_per_row, max_block_size));
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          rms_norm_gated_kernel<sycl_t, vec_size, HasWeight, ACT>(
              (sycl_t*)out_ptr,
              (const sycl_t*)input_ptr,
              (const sycl_t*)gate_ptr,
              (const sycl_t*)weight_ptr,
              epsilon,
              hidden_size,
              s_variance));
    });
    return;
  }

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, max_block_size));
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        rms_norm_gated_scalar_kernel<sycl_t, HasWeight, ACT>(
            (sycl_t*)out_ptr,
            (const sycl_t*)input_ptr,
            (const sycl_t*)gate_ptr,
            (const sycl_t*)weight_ptr,
            epsilon,
            hidden_size,
            s_variance));
  });
}

}  // namespace vllm

void fused_rms_norm_gated(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& gate,
    std::optional<torch::Tensor> weight,
    double epsilon,
    const std::string& activation) {
  const at::DeviceGuard device_guard(input.device());

  TORCH_CHECK(
      input.sizes() == gate.sizes(),
      "fused_rms_norm_gated: gate must have the same shape as input, got ",
      gate.sizes(),
      " vs ",
      input.sizes());
  TORCH_CHECK(
      out.sizes() == input.sizes(),
      "fused_rms_norm_gated: out must have the same shape as input");
  TORCH_CHECK(
      gate.scalar_type() == input.scalar_type() &&
          out.scalar_type() == input.scalar_type(),
      "fused_rms_norm_gated: input, gate and out must share a dtype");
  TORCH_CHECK(
      out.is_contiguous(), "fused_rms_norm_gated: out must be contiguous");

  vllm::GateActivation act;
  if (activation == "sigmoid") {
    act = vllm::GateActivation::Sigmoid;
  } else if (activation == "swish" || activation == "silu") {
    act = vllm::GateActivation::Swish;
  } else {
    TORCH_CHECK(
        false,
        "fused_rms_norm_gated: unsupported activation '",
        activation,
        "', expected one of 'sigmoid', 'swish', 'silu'");
  }

  // The kernels index rows as `row * hidden_size`, so any leading-dimension
  // striding (e.g. a head-major view of a fused QKVG projection) is
  // materialized here.
  const torch::Tensor input_c = input.contiguous();
  const torch::Tensor gate_c = gate.contiguous();

  const bool has_weight = weight.has_value();
  if (has_weight) {
    TORCH_CHECK(weight->is_contiguous());
    TORCH_CHECK(
        weight->numel() == input.size(-1),
        "fused_rms_norm_gated: weight must have hidden_size elements");
    TORCH_CHECK(
        weight->scalar_type() == input.scalar_type(),
        "fused_rms_norm_gated: weight must share the input dtype");
  }

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_rms_norm_gated_kernel", [&] {
        const scalar_t* weight_ptr =
            has_weight ? weight->data_ptr<scalar_t>() : nullptr;
        using Act = vllm::GateActivation;
        if (act == Act::Sigmoid) {
          if (has_weight) {
            vllm::call_rms_norm_gated_kernel<scalar_t, true, Act::Sigmoid>(
                out, input_c, gate_c, weight_ptr, epsilon);
          } else {
            vllm::call_rms_norm_gated_kernel<scalar_t, false, Act::Sigmoid>(
                out, input_c, gate_c, weight_ptr, epsilon);
          }
        } else {
          if (has_weight) {
            vllm::call_rms_norm_gated_kernel<scalar_t, true, Act::Swish>(
                out, input_c, gate_c, weight_ptr, epsilon);
          } else {
            vllm::call_rms_norm_gated_kernel<scalar_t, false, Act::Swish>(
                out, input_c, gate_c, weight_ptr, epsilon);
          }
        }
      });
}
