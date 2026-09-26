// SPDX-License-Identifier: Apache-2.0
//
// Runtime dispatcher for deepseek_inv_rope_fp8_quant, which has an Xe3
// hardware-convert (`fcvt`) path.
//
// `fcvt` with an e4m3 destination only exists on Xe3, and _xpu_C is
// preprocessed once and AOT compiled for every device in AOT_DEVICES, so
// enabling it inside _xpu_C would make ocloc reject the instruction for the Xe2
// targets and fail the link. The kernel is therefore compiled twice, exactly
// like the quant asm ops (see csrc/quant_dispatch.cpp): software convert into
// _xpu_C, hardware convert into libinv_rope_asm_xe_3.so. The latter is used
// when the device is one of the archs it was AOT built for and
// VLLM_XPU_DISABLE_QUANT_ASM is not 1 (the escape hatch the other asm quant
// paths share).
//
// The two paths are numerically equivalent; where they differ, the hardware
// convert is the more accurate one (it emits e4m3 subnormals where the
// software convert flushes to zero).

#include <cstdlib>
#include <tuple>

#include <torch/all.h>

#include "ops.h"
#include "utils.h"

// ----------------------------------------------------------------------------
// Default implementation (software convert), always linked into _xpu_C.
// ----------------------------------------------------------------------------
namespace vllm_inv_rope_fallback {

std::tuple<torch::Tensor, torch::Tensor> deepseek_inv_rope_fp8_quant(
    const torch::Tensor& attn_output,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    int64_t n_groups,
    int64_t heads_per_group,
    int64_t nope_dim,
    int64_t rope_dim,
    bool use_mxfp8);

}  // namespace vllm_inv_rope_fallback

#ifdef VLLM_INV_ROPE_ASM_AVAILABLE
// ----------------------------------------------------------------------------
// Xe3 hardware-convert implementation, from libinv_rope_asm_xe_3.so.
// ----------------------------------------------------------------------------
namespace vllm_inv_rope_asm {

std::tuple<torch::Tensor, torch::Tensor> deepseek_inv_rope_fp8_quant(
    const torch::Tensor& attn_output,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    int64_t n_groups,
    int64_t heads_per_group,
    int64_t nope_dim,
    int64_t rope_dim,
    bool use_mxfp8);

}  // namespace vllm_inv_rope_asm

namespace {

bool use_inv_rope_asm(const at::Device& device) {
  const bool enabled = [&] {
    if (!vllm::xpu::is_xe3p_arch(
            static_cast<at::DeviceIndex>(device.index()))) {
      return false;
    }

    const char* v = std::getenv("VLLM_XPU_DISABLE_QUANT_ASM");
    if (v == nullptr || v[0] != '1') return true;
    return false;
  }();
  return enabled;
}

}  // namespace
#endif  // VLLM_INV_ROPE_ASM_AVAILABLE

std::tuple<torch::Tensor, torch::Tensor> deepseek_inv_rope_fp8_quant(
    const torch::Tensor& attn_output,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    int64_t n_groups,
    int64_t heads_per_group,
    int64_t nope_dim,
    int64_t rope_dim,
    bool use_mxfp8) {
#ifdef VLLM_INV_ROPE_ASM_AVAILABLE
  if (use_inv_rope_asm(attn_output.device())) {
    return vllm_inv_rope_asm::deepseek_inv_rope_fp8_quant(
        attn_output,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        use_mxfp8);
  }
#endif
  return vllm_inv_rope_fallback::deepseek_inv_rope_fp8_quant(
      attn_output,
      positions,
      cos_sin_cache,
      n_groups,
      heads_per_group,
      nope_dim,
      rope_dim,
      use_mxfp8);
}
