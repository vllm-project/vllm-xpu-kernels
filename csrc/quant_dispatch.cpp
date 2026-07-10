// SPDX-License-Identifier: Apache-2.0
//
// Runtime dispatcher for quant ops that have an Xe3p hand-written asm path.
//
// The asm implementation is only linked/available on Xe3p -capable builds
// (VLLM_QUANT_ASM_AVAILABLE). When present, it is taken by default and only
// when BOTH:
//   1. the current device is one of the archs libquant_asm_xe_3.so was
//   AOT-built
//      for (Xe3p arch devices), and
//   2. the env var VLLM_XPU_DISABLE_QUANT_ASM is NOT set to 1 (i.e. the asm
//   path
//      is on by default and can be turned off to fall back to the _C impl).

#include <cstdlib>
#include <optional>
#include <string>

#include "ops.h"
#include "utils.h"

// ----------------------------------------------------------------------------
// Default implementations
// ----------------------------------------------------------------------------
namespace vllm_quant_fallback {

void static_scaled_fp8_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& scale,
    std::optional<std::tuple<int64_t, int64_t>> group_shape);

void dynamic_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    std::optional<at::Tensor> const& scale_ub);

void per_token_group_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0,
    bool dummy_is_scale_transposed,
    bool dummy_is_tma_aligned,
    const c10::optional<torch::Tensor>& expert_scale_desc);

void per_token_group_quant_mxfp4(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps,
    const c10::optional<torch::Tensor>& expert_scale_desc);

void reshape_and_cache_flash(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale);

}  // namespace vllm_quant_fallback

#ifdef VLLM_QUANT_ASM_AVAILABLE
// ----------------------------------------------------------------------------
// ASM implementations
// ----------------------------------------------------------------------------
namespace vllm_quant_asm {

void static_scaled_fp8_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& scale,
    std::optional<std::tuple<int64_t, int64_t>> group_shape);

void dynamic_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    std::optional<at::Tensor> const& scale_ub);

void per_token_group_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0,
    bool dummy_is_scale_transposed,
    bool dummy_is_tma_aligned,
    const c10::optional<torch::Tensor>& expert_scale_desc);

void per_token_group_quant_mxfp4(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps,
    const c10::optional<torch::Tensor>& expert_scale_desc);

void reshape_and_cache_flash(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale);

}  // namespace vllm_quant_asm
#endif  // VLLM_QUANT_ASM_AVAILABLE

namespace {

#ifdef VLLM_QUANT_ASM_AVAILABLE
bool use_quant_asm(const at::Device& device) {
  static const bool enabled = [&] {
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
#endif  // VLLM_QUANT_ASM_AVAILABLE

}  // namespace

// ----------------------------------------------------------------------------
// Public op entry points (global namespace) — bound by torch_bindings.cpp.
//
// Each op keeps its own signature (C++ requires it), but the dispatch body is a
// single VLLM_QUANT_ROUTE(fn, device_tensor, forwarded-args...). To add a new
// asm-routed op: declare it in both namespaces above and write one forwarder
// using the macro below.
// ----------------------------------------------------------------------------

#ifdef VLLM_QUANT_ASM_AVAILABLE
  #define VLLM_QUANT_ROUTE(fn, dev, ...)           \
    do {                                           \
      if (use_quant_asm((dev).device()))           \
        return vllm_quant_asm::fn(__VA_ARGS__);    \
      return vllm_quant_fallback::fn(__VA_ARGS__); \
    } while (0)
#else
  #define VLLM_QUANT_ROUTE(fn, dev, ...) \
    return vllm_quant_fallback::fn(__VA_ARGS__)
#endif

void static_scaled_fp8_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& scale,
    std::optional<std::tuple<int64_t, int64_t>> group_shape) {
  VLLM_QUANT_ROUTE(
      static_scaled_fp8_quant, input, out, input, scale, group_shape);
}

void dynamic_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale) {
  VLLM_QUANT_ROUTE(dynamic_scaled_fp8_quant, input, out, input, scale);
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    std::optional<at::Tensor> const& scale_ub) {
  VLLM_QUANT_ROUTE(
      dynamic_per_token_scaled_fp8_quant, input, out, input, scales, scale_ub);
}

void per_token_group_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0,
    bool dummy_is_scale_transposed,
    bool dummy_is_tma_aligned,
    const c10::optional<torch::Tensor>& expert_scale_desc) {
  VLLM_QUANT_ROUTE(
      per_token_group_quant_fp8,
      input,
      input,
      output_q,
      output_s,
      group_size,
      eps,
      fp8_min,
      fp8_max,
      scale_ue8m0,
      dummy_is_scale_transposed,
      dummy_is_tma_aligned,
      expert_scale_desc);
}

void per_token_group_quant_mxfp4(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps,
    const c10::optional<torch::Tensor>& expert_scale_desc) {
  VLLM_QUANT_ROUTE(
      per_token_group_quant_mxfp4,
      input,
      input,
      output_q,
      output_s,
      group_size,
      eps,
      expert_scale_desc);
}

void reshape_and_cache_flash(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  VLLM_QUANT_ROUTE(
      reshape_and_cache_flash,
      key,
      key,
      value,
      key_cache,
      value_cache,
      slot_mapping,
      kv_cache_dtype,
      k_scale,
      v_scale);
}

#undef VLLM_QUANT_ROUTE