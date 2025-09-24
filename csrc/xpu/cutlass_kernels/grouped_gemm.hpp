

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
                    void* ptr_alpha, void* ptr_beta, void* offset, int64_t N,
                    int64_t K, int64_t groups);
}

/* gemm2(group_A, w2, output, offset) */

at::Tensor grouped_gemm_func(at::Tensor& ptr_A, at::Tensor& ptr_B,
                             at::Tensor& ptr_D, at::Tensor& ptr_alpha,
                             at::Tensor& ptr_beta, at::Tensor& offset,
                             int64_t N, int64_t K, int64_t groups) {
  auto& dpcpp_queue = vllm::xpu::vllmGetQueue();
  grouped_gemm::kernel_functor(dpcpp_queue, ptr_A.data_ptr(), ptr_B.data_ptr(),
                               ptr_D.data_ptr(), ptr_alpha.data_ptr(),
                               ptr_beta.data_ptr(), offset.data_ptr(), N, K,
                               groups);
  return ptr_D;
}

}  // namespace gpu::cutlass_kernel
