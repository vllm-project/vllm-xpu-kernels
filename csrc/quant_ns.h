// SPDX-License-Identifier: Apache-2.0
#pragma once

// Namespace-suffix mechanism used to compile the shared quant sources
// (fp8_quant.cpp / mxfp4_quant.cpp / cache.cpp) twice:
//
//   * once into the _C extension with the ASM path DISABLED  -> fallback impl
//   * once into libquant_asm_xe_3.so with the ASM path ENABLED -> Xe3p asm impl
//
#ifndef VLLM_QUANT_NS
  #define VLLM_QUANT_NS vllm_quant_fallback
#endif
