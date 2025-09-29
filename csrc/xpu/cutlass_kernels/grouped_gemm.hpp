

// #include <sycl/sycl.hpp>
// #include <cassert>
// #include <vector>

// #include <ATen/Tensor.h>
/* #include "pytorch_shim.h" */

#include <torch/all.h>
#include "utils.h"

namespace gpu::cutlass_kernel {

namespace grouped_gemm {
void kernel_functor(sycl::queue& stream, void* ptr_A, void* ptr_B, void* ptr_D,
                    void* offset, int32_t N, int32_t K, int32_t groups);
}

/* gemm2(group_A, w2, output, offset) */

at::Tensor grouped_gemm_func(at::Tensor& ptr_A, at::Tensor& ptr_B,
                             at::Tensor& ptr_D, at::Tensor& tokens_per_expert,
                             int64_t N, int64_t K, int64_t groups) {
  auto& dpcpp_queue = vllm::xpu::vllmGetQueue();
  grouped_gemm::kernel_functor(dpcpp_queue, ptr_A.data_ptr(), ptr_B.data_ptr(),
                               ptr_D.data_ptr(), tokens_per_expert.data_ptr(), (int32_t)N, (int32_t)K,
                               (int32_t)groups);
  return ptr_D;
}

}  // namespace gpu::cutlass_kernel
