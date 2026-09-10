// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>
#include <cstdint>

namespace vllm {

// Describes the per-expert, MN-major, M-padded-to-4 scale-A surface that the
// mxfp grouped-GEMM mainloop indexes (see
// grouped_gemm/xe_3/collective/moe_array_mma_mxfp.hpp: expert e's block is
// based at sum_{i<e} round_up_4(rows[i]) * scale_k and has leading dim
// round_up_4(rows[e])).
//
// The MoE activation for the second grouped GEMM is already laid out with its
// rows grouped per expert, so the quantization kernel knows which expert each
// row belongs to and can place its scale directly. That removes the separate
// reorder pass (and its prefix-scan kernel) from the runtime path.
//
// The whole descriptor is the int32 [2, num_experts + 1] tensor that
// remap_hidden_states publishes for free: row 0 is the exclusive prefix sum of
// the per-expert row counts and row 1 the exclusive prefix sum of
// round_up_4(rows), each with the total appended. Per-expert row counts and
// leading dimensions are the successive differences, so nothing else is needed.
//
// Scales are written as raw e8m0 bytes. Every scale this describes is an exact
// power of two, so the encoding is just the float32 biased exponent field,
// which is the same bit pattern torch produces for float8_e8m0fnu.
struct MoEScaleLayout {
  const int* desc;      // [2, num_experts + 1]
  uint8_t* scale_e8m0;  // destination surface
  int num_experts;
  int scale_k;

  bool enabled() const { return scale_e8m0 != nullptr; }

  // Element offset of scale (row, k) in the padded surface, or -1 when `row`
  // is past the last routed row. Expert parallelism can leave a tail of the
  // activation buffer unrouted, and those rows must not be written.
  int64_t offset(int row, int k) const {
    const int* src = desc;
    const int* dst = desc + num_experts + 1;

    // Last entry that starts at or before `row`. Experts with zero rows share
    // a prefix with their successor, and taking the last such entry skips
    // them. A row past the total lands on the appended total and is rejected.
    int lo = 0;
    int hi = num_experts + 1;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (src[mid] <= row) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    const int e = lo - 1;
    if (e < 0 || e >= num_experts) return -1;

    const int ld = dst[e + 1] - dst[e];
    return static_cast<int64_t>(dst[e]) * scale_k +
           static_cast<int64_t>(k) * ld + (row - src[e]);
  }

  void store(int row, int k, float pow2_scale) const {
    const int64_t idx = offset(row, k);
    if (idx < 0) return;
    scale_e8m0[idx] = static_cast<uint8_t>(
        (sycl::bit_cast<uint32_t>(pow2_scale) >> 23) & 0xFFu);
  }
};

// Validates the optional MoE descriptor and builds the device-side layout.
// Returns a disabled layout when the descriptor is absent, which is the case
// for every non-MoE caller.
inline MoEScaleLayout make_moe_scale_layout(
    const torch::Tensor& output_s,
    const c10::optional<torch::Tensor>& expert_scale_desc,
    int scale_k) {
  MoEScaleLayout layout{};
  if (!expert_scale_desc.has_value()) {
    return layout;
  }

  const auto& desc = expert_scale_desc.value();
  TORCH_CHECK(
      desc.scalar_type() == at::ScalarType::Int && desc.is_contiguous() &&
          desc.dim() == 2 && desc.size(0) == 2 && desc.size(1) >= 2,
      "expert_scale_desc must be a contiguous int32 [2, num_experts + 1] "
      "tensor");
  TORCH_CHECK(
      output_s.scalar_type() == at::ScalarType::Float8_e8m0fnu ||
          output_s.scalar_type() == at::ScalarType::Byte,
      "expert_scale_desc writes raw e8m0 bytes, so output_s must be "
      "float8_e8m0fnu or uint8");
  TORCH_CHECK(
      output_s.is_contiguous(),
      "expert_scale_desc requires a contiguous output_s surface");

  layout.desc = desc.data_ptr<int32_t>();
  layout.scale_e8m0 = reinterpret_cast<uint8_t*>(output_s.data_ptr());
  layout.num_experts = static_cast<int>(desc.size(1)) - 1;
  layout.scale_k = scale_k;
  return layout;
}

}  // namespace vllm
