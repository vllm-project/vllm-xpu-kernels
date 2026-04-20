#include "csrc/utils.h"
#include "gemm_interface.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/gemm_xe2.h"
#endif

torch::Tensor
cutlass_gemm_interface(const torch::Tensor& A, const torch::Tensor& B) {
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& queue = vllm::xpu::vllmGetQueue();
  if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    return cutlass_gemm_xe2(queue, A, B);
#else
    TORCH_CHECK(false, "XE2 cutlass GEMM kernel is not enabled in this build.");
#endif
  } else {
    TORCH_CHECK(false, "cutlass_gemm only supports XE2 architecture.");
  }
}
