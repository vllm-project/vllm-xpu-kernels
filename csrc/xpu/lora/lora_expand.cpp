#include <iostream>
#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include "dispatch_utils.h"
#include "utils.h"
#include "xpu/lora/lora_ops.h"
#include "utils/dpcpp.h"

namespace vllm::lora {
namespace constants {
constexpr uint32_t kSmallProblemThreshold = 8192;
constexpr uint32_t kMediumProblemThreshold = 65536;
constexpr uint32_t kLargeProblemThreshold = 1048576;  // 1M
constexpr uint32_t kMaxWorkgroupsLimit = 8192;
constexpr uint32_t kMaxElementsPerThread = 16;
constexpr uint32_t kVectorAlignmentBytes = 16;
constexpr uint32_t kDefaultWorkgroupSize = 256;
constexpr uint32_t kSmallWorkgroupSize = 64;
constexpr uint32_t kMediumWorkgroupSize = 128;
}  // namespace constants
/**
 * BGMV Expand Slice Kernel
 * ---------------------------------
 * Performs batched grouped matrix–vector multiplication with an expand
 * operation, but only writes to a slice of the output tensor.
 *
 * Template parameters:
 *   - output_t : Output tensor data type
 *   - input_t  : Input tensor data type
 *   - vec_size : Number of elements processed per vectorized loop
 *
 * Tensor dimensions:
 *   - inputs   : [batch_size, rank]
 *   - weights  : [num_loras, slice_size, rank]   (LoRA B matrix slice)
 *   - indices  : [batch_size]
 *   - outputs  : [batch_size, hidden_size] (only slice is updated)
 *
 * Mathematical operation:
 *   For each sample b and slice dimension s:
 *     outputs[b, slice_offset + s] += Σ_r (inputs[b, r] * weights[indices[b],
 * s, r])
 */
template <typename output_t, typename input_t, uint32_t vec_size>
class bgmv_expand_slice_kernel {
 private:
  output_t* outputs_;
  const input_t* inputs_;
  const input_t* weights_;
  const int64_t* indices_;
  const uint32_t hidden_;
  const uint32_t rank_;
  const uint32_t slice_offset_;
  const uint32_t slice_size_;
  const bool add_inputs_;
  const uint32_t batch_size_;

 public:
  using acc_t = vllm::xpu::acc_type<input_t>;
  using vec_t = vllm::xpu::aligned_vec<input_t, vec_size>;

  /**
   * Constructor
   * @param outputs Output tensor [batch_size, hidden_size]
   * @param inputs Input tensor [batch_size, rank]
   * @param weights LoRA weights [num_loras, slice_size, rank]
   * @param indices LoRA indices [batch_size]
   * @param hidden Hidden dimension
   * @param rank LoRA rank
   * @param slice_offset Starting offset in output dimension
   * @param slice_size Size of the slice to update
   * @param add_inputs Whether to accumulate to existing output
   */
  bgmv_expand_slice_kernel(
      output_t* outputs,
      const input_t* inputs,
      const input_t* weights,
      const int64_t* indices,
      const uint32_t hidden,
      const uint32_t rank,
      const uint32_t slice_offset,
      const uint32_t slice_size,
      const bool add_inputs,
      const uint32_t batch_size)
      : outputs_(outputs),
        inputs_(inputs),
        weights_(weights),
        indices_(indices),
        hidden_(hidden),
        rank_(rank),
        slice_offset_(slice_offset),
        slice_size_(slice_size),
        add_inputs_(add_inputs),
        batch_size_(batch_size) {}

  void operator()(sycl::nd_item<1> item) const {
    const uint32_t global_id = item.get_global_linear_id();
    const uint32_t group_size = item.get_local_range(0);
    const uint32_t group_id = item.get_group_linear_id();

    const uint32_t total_threads = group_size * item.get_group_range(0);

    const uint32_t elements_per_thread =
        (batch_size_ * slice_size_ + total_threads - 1) / total_threads;

    for (uint32_t i = 0; i < elements_per_thread; i++) {
      const uint32_t element_idx = global_id + i * total_threads;
      if (element_idx >= batch_size_ * slice_size_) break;

      const uint32_t batch_id = element_idx / slice_size_;
      const uint32_t slice_id = element_idx % slice_size_;

      const int64_t lora_idx = indices_[batch_id];
      if (lora_idx < 0) continue;

      const input_t* input_base = inputs_ + batch_id * rank_;
      const input_t* weight_base =
          weights_ + lora_idx * slice_size_ * rank_ + slice_id * rank_;

      acc_t sum = 0;

      const uint32_t full_vectors = rank_ / vec_size;
      for (uint32_t j = 0; j < full_vectors; j++) {
        const vec_t input_vec =
            *reinterpret_cast<const vec_t*>(input_base + j * vec_size);
        const vec_t weight_vec =
            *reinterpret_cast<const vec_t*>(weight_base + j * vec_size);

#pragma unroll
        for (uint32_t k = 0; k < vec_size; k++) {
          sum = sycl::mad(
              static_cast<acc_t>(input_vec[k]),
              static_cast<acc_t>(weight_vec[k]),
              sum);
        }
      }

      const uint32_t remaining = rank_ % vec_size;
      for (uint32_t j = 0; j < remaining; j++) {
        const uint32_t offset = full_vectors * vec_size + j;
        sum += static_cast<acc_t>(input_base[offset]) *
               static_cast<acc_t>(weight_base[offset]);
      }

      const uint32_t output_hidden_id = slice_offset_ + slice_id;
      if (output_hidden_id < hidden_) {
        const uint32_t output_idx = batch_id * hidden_ + output_hidden_id;
        const output_t result = static_cast<output_t>(sum);

        if (add_inputs_) {
          outputs_[output_idx] += result;
        } else {
          outputs_[output_idx] = result;
        }
      }
    }
  }
};

}  // namespace vllm::lora

// Vector size dispatch function
template <typename Fn>
void dispatch_vec_size(int vec_size, Fn&& fn) {
  switch (vec_size) {
    case 8:
      fn(std::integral_constant<int, 8>{});
      break;
    case 4:
      [[likely]] fn(std::integral_constant<int, 4>{});
      break;
    case 2:
      fn(std::integral_constant<int, 2>{});
      break;
    case 1:
      [[unlikely]] fn(std::integral_constant<int, 1>{});
      break;
    default:
      [[unlikely]] TORCH_CHECK(false, "Unsupported vector size: ", vec_size);
  }
}

/**
 * Compute optimal workgroup configuration
 */
std::pair<uint32_t, uint32_t>
compute_workgroup(uint32_t total_elements, uint32_t max_workgroup_size) {
  using namespace vllm::lora::constants;

  uint32_t workgroup_size;
  uint32_t num_workgroups;

  // Choose workgroup size based on problem size
  if (total_elements <= kSmallProblemThreshold) {
    workgroup_size = kSmallWorkgroupSize;
  } else if (total_elements <= kMediumProblemThreshold) {
    workgroup_size = kMediumWorkgroupSize;
  } else {
    workgroup_size = kDefaultWorkgroupSize;
  }

  // Calculate number of workgroups
  if (total_elements <= kLargeProblemThreshold) {
    // Small to medium problems: one thread per element
    num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
  } else {
    // Large problems: use multiple elements per thread to limit workgroups
    const uint32_t elements_per_thread = kMaxElementsPerThread;
    num_workgroups =
        (total_elements + elements_per_thread - 1) / elements_per_thread;
    num_workgroups = std::min(num_workgroups, kMaxWorkgroupsLimit);
  }

  // Apply device constraints
  workgroup_size = std::min(workgroup_size, max_workgroup_size);
  num_workgroups = std::max(num_workgroups, 1u);

  // Ensure workgroup size doesn't exceed total elements for small problems
  if (total_elements < workgroup_size) {
    workgroup_size = std::max(total_elements, 1u);
    num_workgroups = 1;
  }

  return {workgroup_size, num_workgroups};
}

/**
 * Launch BGMV expand slice kernel
 *
 * @param outputs  [batch_size, hidden_size] - Output tensor
 * @param inputs   [batch_size, rank] - Input tensor
 * @param weights  [num_loras, slice_size, rank] - LoRA weights slice
 * @param indices  [batch_size] - LoRA index mapping
 * @param batch_size - Batch size
 * @param hidden     - Hidden dimension size
 * @param rank       - LoRA rank
 * @param slice_offset - Starting offset in output dimension
 * @param slice_size   - Size of the slice to update
 * @param add_inputs - Whether to accumulate to existing output
 */
template <typename output_t, typename input_t>
void launch_bgmv_expand_slice(
    output_t* outputs,
    input_t* inputs,
    input_t* weights,
    int64_t* indices,
    const uint32_t batch_size,
    const uint32_t hidden,
    const uint32_t rank,
    const uint32_t slice_offset,
    const uint32_t slice_size,
    const bool add_inputs) {
  uint32_t vec_bytes = 16;
  const auto input_align = reinterpret_cast<uintptr_t>(inputs);
  const auto weight_align = reinterpret_cast<uintptr_t>(weights);
  const uint32_t data_bytes = rank * sizeof(input_t);

  while (vec_bytes > sizeof(input_t) &&
         (data_bytes % vec_bytes != 0 || input_align % vec_bytes != 0 ||
          weight_align % vec_bytes != 0)) {
    vec_bytes /= 2;
  }
  // Fallback to scalar operations
  if (vec_bytes < sizeof(input_t)) {
    vec_bytes = sizeof(input_t);
  }
  uint32_t vec_size = vec_bytes / sizeof(input_t);

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& dpcpp_queue = vllm::xpu::vllmGetQueue();
  const auto dev_id = at::xpu::current_device();
  const uint32_t max_workgroup_size = vllm::xpu::getMaxWorkGroupSize(dev_id);

  const uint32_t total_elements = batch_size * slice_size;

  // Compute workgroup configuration
  auto [workgroup_size, num_workgroups] =
      compute_workgroup(total_elements, max_workgroup_size);

  const sycl::range<1> local_range{workgroup_size};
  const sycl::range<1> global_range{num_workgroups * workgroup_size};

  dpcpp_queue.submit([&](sycl::handler& cgh) {
    dispatch_vec_size(vec_size, [&](auto vec_c) {
      constexpr int V = vec_c.value;
      vllm::lora::bgmv_expand_slice_kernel<output_t, input_t, V> kfn(
          outputs,
          inputs,
          weights,
          indices,
          hidden,
          rank,
          slice_offset,
          slice_size,
          add_inputs,
          batch_size);
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
    });
  });
}

void validate_lora_b_slice_tensors(
    const at::Tensor& inputs,
    const at::Tensor& lora_b_weights,
    const at::Tensor& output_tensor,
    const at::Tensor& lora_indices_tensor,
    const uint32_t slice_offset,
    const uint32_t slice_size) {
  // Device checks
  TORCH_CHECK(inputs.is_xpu(), "inputs must be on XPU");
  TORCH_CHECK(lora_b_weights.is_xpu(), "lora_b_weights must be on XPU");
  TORCH_CHECK(output_tensor.is_xpu(), "output_tensor must be on XPU");
  TORCH_CHECK(
      lora_indices_tensor.is_xpu(), "lora_indices_tensor must be on XPU");

  // Contiguous checks
  TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(
      lora_b_weights.is_contiguous(), "lora_b_weights must be contiguous");
  TORCH_CHECK(
      output_tensor.is_contiguous(), "output_tensor must be contiguous");
  TORCH_CHECK(
      lora_indices_tensor.is_contiguous(),
      "lora_indices_tensor must be contiguous");

  // Dtype checks
  TORCH_CHECK(
      inputs.scalar_type() == lora_b_weights.scalar_type(),
      "inputs dtype must match lora_b_weights dtype");

  TORCH_CHECK(
      inputs.scalar_type() == at::kHalf ||
          inputs.scalar_type() == at::kBFloat16,
      "inputs must be float16 or bfloat16");

  TORCH_CHECK(
      output_tensor.scalar_type() == at::kHalf ||
          output_tensor.scalar_type() == at::kBFloat16 ||
          output_tensor.scalar_type() == at::kFloat,
      "output_tensor must be float16, bfloat16, or float32");

  TORCH_CHECK(
      lora_indices_tensor.scalar_type() == at::kLong,
      "lora_indices_tensor must be int64");

  // Dimension checks
  TORCH_CHECK(inputs.dim() == 2, "inputs must be 2D [batch_size, rank]");
  TORCH_CHECK(
      output_tensor.dim() == 2,
      "output_tensor must be 2D [batch_size, hidden_size]");
  TORCH_CHECK(
      lora_indices_tensor.dim() == 1,
      "lora_indices_tensor must be 1D [batch_size]");

  at::Tensor lora_weights = lora_b_weights;
  if (lora_b_weights.dim() == 4) {  // shape: (lora_num,1,slice_size,rank)
    TORCH_CHECK(
        lora_b_weights.size(1) == 1, "lora_b_weights.size(1) must be 1");
  } else {
    TORCH_CHECK(
        lora_b_weights.dim() == 3,
        "lora_b_weights must be 3D [lora_num, slice_size, rank]");
  }

  // Shape consistency checks
  TORCH_CHECK(
      inputs.size(1) == lora_weights.size(-1),
      "inputs.size(1) must match lora_b_weights.size(-1)");
  TORCH_CHECK(
      lora_weights.size(-2) == slice_size,
      "lora_b_weights.size(-2) must match slice_size");
  TORCH_CHECK(
      inputs.size(0) == output_tensor.size(0),
      "inputs.size(0) must match output_tensor.size(0)");
  TORCH_CHECK(
      inputs.size(0) == lora_indices_tensor.size(0),
      "inputs.size(0) must match lora_indices_tensor.size(0)");

  // Slice bounds check
  TORCH_CHECK(
      slice_offset + slice_size <= output_tensor.size(1),
      "slice [",
      slice_offset,
      ":",
      slice_offset + slice_size,
      ") is out of bounds for output dimension ",
      output_tensor.size(1));
}

/**
 * BGMV expand slice operation main function
 *
 * @param outputs [batch_size, hidden_size] - Output tensor
 * @param inputs  [batch_size, rank] - Input feature tensor
 * @param weights [num_loras, slice_size, rank] - LoRA weight matrix B slice
 * @param indices [batch_size] - LoRA index for each sample
 * @param slice_offset - Starting offset in output dimension
 * @param slice_size - Size of the slice to update
 * @param add_inputs - Whether to accumulate to existing output
 */
void bgmv_expand_slice(
    torch::Tensor& outputs,
    const torch::Tensor& inputs,
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    const int64_t slice_offset,
    const int64_t slice_size,
    bool add_inputs) {
  // 1. Input validation
  validate_lora_b_slice_tensors(
      inputs, weights, outputs, indices, slice_offset, slice_size);

  at::Tensor lora_weights = weights;
  if (lora_weights.dim() == 4) {  // shape: (lora_num,1,slice_size,rank)
    lora_weights = lora_weights.squeeze(1);  // squeeze dim 1
  }

  // 2. Get dimension information
  uint32_t batch_size = inputs.size(0);
  uint32_t rank = inputs.size(1);
  uint32_t hidden = outputs.size(1);

  // 3. Dispatch based on INPUT type
  VLLM_DISPATCH_FLOATING_TYPES(
      inputs.scalar_type(), "bgmv_expand_slice", [&]() {
        using input_t = scalar_t;
        switch (outputs.scalar_type()) {
          case at::ScalarType::Half:
            launch_bgmv_expand_slice<at::Half, input_t>(
                outputs.data_ptr<at::Half>(),
                inputs.data_ptr<input_t>(),
                lora_weights.data_ptr<input_t>(),
                indices.data_ptr<int64_t>(),
                batch_size,
                hidden,
                rank,
                slice_offset,
                slice_size,
                add_inputs);
            break;
          case at::ScalarType::BFloat16:
            launch_bgmv_expand_slice<at::BFloat16, input_t>(
                outputs.data_ptr<at::BFloat16>(),
                inputs.data_ptr<input_t>(),
                lora_weights.data_ptr<input_t>(),
                indices.data_ptr<int64_t>(),
                batch_size,
                hidden,
                rank,
                slice_offset,
                slice_size,
                add_inputs);
            break;
          case at::ScalarType::Float:
            launch_bgmv_expand_slice<float, input_t>(
                outputs.data_ptr<float>(),
                inputs.data_ptr<input_t>(),
                lora_weights.data_ptr<input_t>(),
                indices.data_ptr<int64_t>(),
                batch_size,
                hidden,
                rank,
                slice_offset,
                slice_size,
                add_inputs);
            break;
          default:
            TORCH_CHECK(
                false, "Unsupported output type: ", outputs.scalar_type());
            break;
        }
      });
}

/**
 * BGMV expand operation main function
 *
 * @param outputs [batch_size, hidden_size] - Output tensor
 * @param inputs  [batch_size, rank] - Input feature tensor
 * @param weights [num_loras, hidden_size, rank] - LoRA weight matrix B
 * @param indices [batch_size] - LoRA index for each sample
 * @param add_inputs - Whether to accumulate to existing output
 */
void bgmv_expand(
    torch::Tensor& outputs,
    const torch::Tensor& inputs,
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    bool add_inputs) {
  uint32_t hidden = outputs.size(1);

  bgmv_expand_slice(
      outputs,
      inputs,
      weights,
      indices,
      0,       // slice_offset = 0
      hidden,  // slice_size
      add_inputs);
}