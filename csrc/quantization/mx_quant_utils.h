// SPDX-License-Identifier: Apache-2.0
#pragma once

// Helpers for the MX quant kernels (32-element groups, e8m0 scales). Each
// lane handles one 16-byte vector of 16-bit input, so a group consists of 4
// lanes.

#include <sycl/sycl.hpp>
#include <cstdint>
#include <type_traits>

#include "quantization/fp4/mxfp4_quant.h"
#include "quantization/fp8/fp8_quant_asm.h"
#include "quantization/fp8/quant_utils.h"
#include "quantization/utils.h"
#include "quant_ns.h"

namespace vllm {
namespace mx {
inline namespace VLLM_QUANT_KERNEL_NS {

static constexpr int kGroupSize = 32;
static constexpr int kVec = 8;  // 16-bit elements per 16-byte vector
static constexpr int kLanesPerGroup = kGroupSize / kVec;
static constexpr int kWorkGroupSize = 256;

inline int flat_index(const sycl::nd_item<1>& item) {
  return static_cast<int>(item.get_group(0)) * kWorkGroupSize +
         static_cast<int>(item.get_local_id(0));
}

enum class ScaleLayout { kRowMajor, kColMajor, kMoE };

inline float group_absmax(sycl::sub_group sg, float v, int lanes) {
  for (int off = 1; off < lanes; off <<= 1) {
    v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, off));
  }
  return v;
}

template <typename fp8_t>
inline float mxfp8_scale(float absmax, float& inv_scale) {
  const float y = sycl::fmax(
      absmax / fp8::quant_type_max_v<fp8_t>,
      fp8::min_scaling_factor<fp8_t>::val());
  return mxfp4::pow2_ceil_scale(y, inv_scale);
}

using fp8x8_t = vec_n_t<uint8_t, kVec>;
// EXACT_IN_FP16: the scaled values are known to be exact in fp16 (16-bit
// input times a power-of-two scale), so the Xe3p path skips round-to-odd.
template <typename fp8_t, bool EXACT_IN_FP16 = false>
inline fp8x8_t mxfp8_pack8(const float (&val)[kVec], float inv_scale) {
#ifdef VLLM_FP8_QUANT_ASM_ENABLED
  vec_n_t<uint32_t, kVec / 4> w;
  #pragma unroll
  for (int i = 0; i < kVec / 4; ++i) {
    const float x4[4] = {
        val[4 * i], val[4 * i + 1], val[4 * i + 2], val[4 * i + 3]};
    w.val[i] = fp8::scaled_quant4_asm<fp8_t, EXACT_IN_FP16>(x4, inv_scale);
  }
  return sycl::bit_cast<fp8x8_t>(w);
#else
  const float fp8_max = static_cast<float>(fp8::quant_type_max_v<fp8_t>);
  fp8x8_t o;
  #pragma unroll
  for (int k = 0; k < kVec; ++k) {
    const float x =
        sycl::fmax(-fp8_max, sycl::fmin(val[k] * inv_scale, fp8_max));
    o.val[k] = sycl::bit_cast<uint8_t>(static_cast<fp8_t>(x));
  }
  return o;
#endif
}

}  // namespace VLLM_QUANT_KERNEL_NS
}  // namespace mx
}  // namespace vllm
