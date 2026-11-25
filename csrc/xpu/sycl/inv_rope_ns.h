// SPDX-License-Identifier: Apache-2.0
#pragma once

// Namespace-suffix mechanism used to compile deepseek_inv_rope_fp8_quant.cpp
// twice, mirroring csrc/quant_ns.h: the software convert goes into _xpu_C and
// the Xe3 `fcvt` convert into libinv_rope_asm_xe_3.so. See
// csrc/xpu/inv_rope_dispatch.cpp for why, and for the runtime selection.
#ifndef VLLM_INV_ROPE_NS
  #define VLLM_INV_ROPE_NS vllm_inv_rope_fallback
#endif

// Both copies define the same SYCL kernel class. The inline namespace gives the
// two device images distinct kernel names, so the runtime never sees two images
// registered under one name, while use sites keep spelling the class
// `vllm::deepseek_inv_rope_fp8_quant_kernel`.
#ifdef VLLM_XPU_SYCL_FCVT_ENABLED
  #define VLLM_INV_ROPE_IMPL_NS impl_fcvt
#else
  #define VLLM_INV_ROPE_IMPL_NS impl_sw
#endif
