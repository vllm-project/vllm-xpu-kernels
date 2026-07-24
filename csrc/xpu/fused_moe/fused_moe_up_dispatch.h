#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>

#include "activation_utils.h"

namespace FusedMOE {

enum class FusedMOEWeightType { W4A16, W8A16, W16A16 };

struct FusedMOEUpLaunchParams {
  sycl::queue& queue;
  const void* activations;
  const void* weights;
  const void* scales;
  const void* bias;
  void* outputs;
  const int* rows_per_expert;
  int gemm_n;
  int gemm_k;
  int num_experts;
  int group_size;
  int32_t* atomic_buffer;
  double gemm1_clamp_limit;
  int average_rows_per_expert;
  int weight_n;
  bool has_clamping;
  bool activations_are_bfloat16;
  bool uses_first_weight_encoding;
};

using FusedMOEUpDispatchFn = void (*)(const FusedMOEUpLaunchParams&);

void fused_moe_up_w4a16_silu(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w4a16_gelu(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w4a16_gelu_tanh(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w4a16_swigluoai(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w4a16_relu2_no_mul(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w4a16_swiglustep(const FusedMOEUpLaunchParams& params);

void fused_moe_up_w8a16_silu(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w8a16_gelu(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w8a16_gelu_tanh(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w8a16_swigluoai(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w8a16_relu2_no_mul(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w8a16_swiglustep(const FusedMOEUpLaunchParams& params);

void fused_moe_up_w16a16_silu(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w16a16_gelu(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w16a16_gelu_tanh(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w16a16_swigluoai(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w16a16_relu2_no_mul(const FusedMOEUpLaunchParams& params);
void fused_moe_up_w16a16_swiglustep(const FusedMOEUpLaunchParams& params);

}  // namespace FusedMOE
