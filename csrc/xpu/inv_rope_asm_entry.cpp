// SPDX-License-Identifier: Apache-2.0
//
// ASM entry translation unit: re-includes the kernel source under
// VLLM_INV_ROPE_NS = vllm_inv_rope_asm so its symbols do not clash with the
// software-convert copy in _xpu_C. The cmake target, not this file, defines
// VLLM_XPU_SYCL_FCVT_ENABLED, so BUILD_QUANT_ASM_XE3=OFF can still opt out.
#define VLLM_INV_ROPE_NS vllm_inv_rope_asm

#include "sycl/deepseek_inv_rope_fp8_quant.cpp"
