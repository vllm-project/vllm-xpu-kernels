#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <memory>
#include <sycl/sycl.hpp>

namespace vllm {
namespace xpu {

static inline sycl::queue& vllmGetQueue() {
  auto current_stream = c10::xpu::getCurrentXPUStream();
  auto& queue = current_stream.queue();
  return queue;
}

template <typename T>
struct SyclTypeTrait {
  using Type = T;
};

template <>
struct SyclTypeTrait<c10::Half> {
  using Type = sycl::half;
};

template <>
struct SyclTypeTrait<c10::BFloat16> {
  using Type = sycl::ext::oneapi::bfloat16;
};

}  // namespace xpu

}  // namespace vllm
