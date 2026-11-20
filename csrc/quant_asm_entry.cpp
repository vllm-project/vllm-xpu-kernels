// SPDX-License-Identifier: Apache-2.0
//
// ASM entry translation unit. Compiled ONLY into libquant_asm_xe_3.so on
// Xe3p-capable builds. Re-includes the shared quant sources verbatim under the
// asm namespace (VLLM_QUANT_NS = vllm_quant_asm) with the asm path enabled (Xe3
// -D + AOT), so their symbols do not clash with the fallback copies linked into
// _C.
//
// VLLM_QUANT_ASM_BUILD drops the public ops without an asm path (e.g. every
// cache op but reshape_and_cache_flash); those stay in _C only.
#define VLLM_QUANT_NS vllm_quant_asm
#define VLLM_QUANT_ASM_BUILD 1

#include "quantization/fp4/mxfp4_quant.cpp"
#include "quantization/fp8/fp8_quant.cpp"
#include "quantization/fused_kernels/fused_silu_mul_block_quant.cpp"
#include "quantization/fused_kernels/fused_silu_mul_mxfp4_quant.cpp"
#include "layernorm_quant.cpp"
#include "cache.cpp"
