// SPDX-License-Identifier: Apache-2.0
//
// Architecture guard of the mono decode ops: the kernels are written for Xe2
// (Battlemage: 2D block loads, DPAS shapes, 256-GRF work-groups).
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include "utils.h"

namespace vllm::mono_decode {

inline bool xe2_device(int64_t device) {
  static std::array<std::atomic<int8_t>, 64> known{};  // 0 unknown, 1 / -1
  const at::DeviceIndex d =
      device < 0 ? c10::xpu::current_device() : (at::DeviceIndex)device;
  if (d < 0 || d >= (at::DeviceIndex)known.size()) return vllm::xpu::is_bmg(d);
  int8_t v = known[d].load(std::memory_order_relaxed);
  if (v == 0) {
    v = vllm::xpu::is_bmg(d) ? 1 : -1;
    known[d].store(v, std::memory_order_relaxed);
  }
  return v == 1;
}

inline void require_xe2(const char* op) {
  TORCH_CHECK(
      xe2_device(-1),
      op,
      ": the mono decode kernels need an Xe2 (Battlemage) GPU; check "
      "mono_decode_supported() first");
}

}  // namespace vllm::mono_decode
