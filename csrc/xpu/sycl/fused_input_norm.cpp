#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <ATen/DeviceGuard.h>
#include "utils.h"
#include "dispatch_utils.h"
#include "quantization/utils.h"

namespace vllm {

// Fused image rescale + normalize for multimodal preprocessing.
//
// Mirrors ``FusedInputNorm.forward`` in vLLM's ``vision.py`` which, with
// ``running_mean=0``, ``running_var=1`` and ``eps=0``, reduces
// ``F.batch_norm`` to a per-channel affine transform:
//
//   out[p, c * patch_size + i] = input[p, c * patch_size + i] * weight[c]
//                                 + bias[c]
//
// ``input`` is ``uint8`` (rescale + normalize folded into weight/bias) and the
// output is cast to the visual encoder dtype (bf16 / fp16 / fp32). This avoids
// the fp32 intermediate + generic ``batch_norm`` that regresses throughput on
// XPU.
//
// Each work-item handles one output element. ``weight`` and ``bias`` are
// per-channel (small, e.g. channel == 3) so they are read from global memory
// directly and stay resident in cache.
template <typename scalar_t>
class fused_input_norm_kernel {
 public:
  fused_input_norm_kernel(
      scalar_t* __restrict__ out_,
      const uint8_t* __restrict__ input_,
      const float* __restrict__ weight_,  // [channel]
      const float* __restrict__ bias_,    // [channel]
      const int64_t num_elems_,
      const int channel_,
      const int patch_size_)
      : out(out_),
        input(input_),
        weight(weight_),
        bias(bias_),
        num_elems(num_elems_),
        channel(channel_),
        patch_size(patch_size_) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int64_t stride = item.get_global_range(0);
    const int64_t row_size = static_cast<int64_t>(channel) * patch_size;
    for (int64_t idx = item.get_global_id(0); idx < num_elems; idx += stride) {
      // Row is laid out as [c0 x patch_size, c1 x patch_size, ...], so the
      // channel of a flat index is ((idx % row_size) / patch_size).
      const int col = static_cast<int>(idx % row_size);
      const int c = col / patch_size;
      const float x = static_cast<float>(input[idx]);
      out[idx] = static_cast<scalar_t>(x * weight[c] + bias[c]);
    }
  }

 private:
  scalar_t* __restrict__ out;
  const uint8_t* __restrict__ input;
  const float* __restrict__ weight;
  const float* __restrict__ bias;
  const int64_t num_elems;
  const int channel;
  const int patch_size;
};

template <typename scalar_t>
void call_fused_input_norm_kernel(
    torch::Tensor& out,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int channel,
    const int patch_size) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  const int64_t num_elems = input.numel();
  auto out_ptr = reinterpret_cast<sycl_t*>(out.data_ptr<scalar_t>());
  auto input_ptr = input.data_ptr<uint8_t>();
  auto weight_ptr = weight.data_ptr<float>();
  auto bias_ptr = bias.data_ptr<float>();

  auto& queue = vllm::xpu::vllmGetQueue();

  // The kernel is memory-bandwidth bound, so a bounded grid with a grid-stride
  // loop is preferred over one work-item per element: it keeps launch overhead
  // low without hurting bandwidth on very large tensors.
  constexpr int block_size = 256;
  const int64_t num_wg =
      std::min<int64_t>((num_elems + block_size - 1) / block_size, 65535);
  sycl::range<1> global(num_wg * block_size);
  sycl::range<1> local(block_size);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(global, local),
        fused_input_norm_kernel<sycl_t>(
            out_ptr,
            input_ptr,
            weight_ptr,
            bias_ptr,
            num_elems,
            channel,
            patch_size));
  });
}

}  // namespace vllm

void fused_input_norm(
    torch::Tensor& out,     // [num_patches, channel * patch_size]
    torch::Tensor& input,   // uint8, same shape as out
    torch::Tensor& weight,  // float32 [channel]
    torch::Tensor& bias) {  // float32 [channel]
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Byte,
      "fused_input_norm: input must be uint8");
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float,
      "fused_input_norm: weight must be float32");
  TORCH_CHECK(
      bias.scalar_type() == at::ScalarType::Float,
      "fused_input_norm: bias must be float32");
  TORCH_CHECK(input.dim() == 2, "fused_input_norm: input must be 2D");
  TORCH_CHECK(
      out.sizes() == input.sizes(),
      "fused_input_norm: out and input must have the same shape");
  TORCH_CHECK(
      input.is_contiguous() && out.is_contiguous(),
      "fused_input_norm: input and out must be contiguous");
  TORCH_CHECK(
      weight.is_contiguous() && bias.is_contiguous(),
      "fused_input_norm: weight and bias must be contiguous");

  const int channel = weight.numel();
  TORCH_CHECK(channel > 0, "fused_input_norm: weight must be non-empty");
  TORCH_CHECK(
      bias.numel() == channel,
      "fused_input_norm: weight and bias must have the same length");
  const int64_t row_size = input.size(1);
  TORCH_CHECK(
      row_size % channel == 0,
      "fused_input_norm: row size must be divisible by channel");
  const int patch_size = static_cast<int>(row_size / channel);

  VLLM_DISPATCH_FLOATING_TYPES(
      out.scalar_type(), "call_fused_input_norm_kernel", [&] {
        vllm::call_fused_input_norm_kernel<scalar_t>(
            out, input, weight, bias, channel, patch_size);
      });
}
