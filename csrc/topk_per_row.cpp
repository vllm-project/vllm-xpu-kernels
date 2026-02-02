#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <string>

#include "dispatch_utils.h"
#include "utils.h"

namespace vllm {

class top_k_per_row_prefill_kernel {
 public:
  top_k_per_row_prefill_kernel(
      const int32_t* indices,
      const float* logits,
      const int32_t* rowStart,
      const int32_t* rowEnd,
      const int64_t stride0,
      const int64_t stride1,
      const int64_t topK)
      : indices_(indices),
        logits_(logits),
        rowStart_(rowStart),
        rowEnd_(rowEnd),
        stride0_(stride0),
        stride1_(stride1),
        topK_(topK) {}
  void operator()(const sycl::nd_item<1>& item) const {
    int64_t group_idx = item.get_group(0);
    int64_t local_idx = item.get_local_id(0);
    int local_range = item.get_local_range(0);
  }

 private:
  const int32_t *indices_; // [num_tokens, topK]
  const float *logits_;   // [num_tokens, num_max_logits]
  const int32_t *rowStart_; // [num_tokens]
  const int32_t *rowEnd_; // [num_tokens]
  const int64_t stride0_;
  const int64_t stride1_;
  const int64_t topK_;
};
} // namespace vllm

void top_k_per_row_decode(
    const torch::Tensor& logits,
    int64_t next_n,
    const torch::Tensor& seqLens,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK) {
  return;
}

void top_k_per_row_prefill(
    const torch::Tensor& logits,
    const torch::Tensor& rowStarts,
    const torch::Tensor& rowEnds,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK) {
  TORCH_CHECK(logits.size(0) == numRows);
  TORCH_CHECK(logits.stride(0) == stride0);
  TORCH_CHECK(logits.stride(1) == stride1);

  TORCH_CHECK(rowStarts.size(0) == numRows);
  TORCH_CHECK(rowEnds.size(0) == numRows);

  TORCH_CHECK(indices.size(0) == numRows);
  TORCH_CHECK(indices.size(1) == topK);

  sycl::range<1> grid(numRows);
  sycl::range<1> block(512);
  const at::DeviceGuard device_guard(logits.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    // SLM allocation
    // sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(grid * block, block),
        vllm::top_k_per_row_prefill_kernel(
            indices.data_ptr<int32_t>(),
            logits.data_ptr<float>(),
            rowStarts.data_ptr<int32_t>(),
            rowEnds.data_ptr<int32_t>(),
            stride0,
            stride1,
            topK));
  });
}
