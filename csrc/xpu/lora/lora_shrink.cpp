#include <iostream>
#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include "dispatch_utils.h"
#include "utils.h"
#include "xpu/lora/lora_ops.h"
#include "utils/dpcpp.h"

namespace vllm::lora {

/**
 * BGMV Shrink Kernel
 * ---------------------------------
 * Performs batched grouped matrix–vector multiplication with a shrink
 * operation, projecting inputs to a lower-rank space.
 *
 * Template parameters:
 *   - output_t : Output tensor data type
 *   - input_t  : Input tensor data type
 *   - vec_size : Number of elements processed per vectorized loop
 *
 * Tensor dimensions:
 *   - inputs   : [batch_size, hidden_size]
 *   - weights  : [num_loras, rank, hidden_size]   (LoRA B matrix)
 *   - indices  : [batch_size]
 *   - outputs  : [batch_size, rank]
 *
 * Mathematical operation:
 *   For each sample b and rank r:
 *     outputs[b, r] = scale * Σ_h (inputs[b, h] * weights[indices[b], r, h])
 */
template <typename output_t, typename input_t, uint32_t vec_size>
class bgmv_shrink_kernel {
 private:
  output_t* outputs_;
  const input_t* inputs_;
  const input_t* weights_;
  const int64_t* indices_;
  const uint32_t hidden_;
  const uint32_t rank_;
  const float scale_;

 public:
  using acc_t = vllm::xpu::acc_type<input_t>;
  using vec_t = vllm::xpu::aligned_vec<input_t, vec_size>;
  /**
   * Constructor
   * @param outputs Output tensor [batch_size, rank]
   * @param inputs Input tensor [batch_size, hidden_size]
   * @param weights LoRA weights [num_loras, rank, hidden_size]
   * @param indices LoRA indices [batch_size]
   * @param hidden Hidden dimension
   * @param rank LoRA rank
   * @param scale Scaling factor
   */
  bgmv_shrink_kernel(
      output_t* outputs,
      const input_t* inputs,
      const input_t* weights,
      const int64_t* indices,
      const uint32_t hidden,
      const uint32_t rank,
      const float scale)
      : outputs_(outputs),
        inputs_(inputs),
        weights_(weights),
        indices_(indices),
        hidden_(hidden),
        rank_(rank),
        scale_(scale) {}

  void operator()(sycl::nd_item<1> item) const {
    // Thread indexing
    const uint32_t local_id =
        item.get_local_linear_id();  // Local thread ID within workgroup
    const uint32_t group_id = item.get_group_linear_id();  // Workgroup ID
    const uint32_t group_size = item.get_local_range(0);   // Workgroup size

    // Calculate batch and rank indices for current thread
    // Workgroup mapping: group_id = batch_id * rank + rank_id
    const uint32_t batch_id = group_id / rank_;  // Batch index [0, batch_size)
    const uint32_t rank_id = group_id % rank_;   // Rank index [0, rank)

    // Get LoRA index for current batch
    const int64_t lora_idx =
        indices_[batch_id];  // indices[batch_id] -> lora_id

    // Skip invalid LoRA indices
    if (lora_idx < 0) return;

    // Calculate data pointers
    // inputs: [batch_size, hidden_size] -> inputs[batch_id, :]
    const input_t* input_base = inputs_ + batch_id * hidden_;

    // weights: [num_loras, rank, hidden_size] -> weights[lora_idx, rank_id, :]
    const input_t* weight_base =
        weights_ + lora_idx * rank_ * hidden_ + rank_id * hidden_;

    // Initialize accumulator
    acc_t local_sum = 0;

    // Vectorized parallel processing
    // Each thread processes vec_size consecutive elements
    uint32_t offset = local_id * vec_size;          // Thread starting position
    const uint32_t stride = group_size * vec_size;  // Stride between threads

    // Main computation loop: dot product along hidden dimension
    while (offset < hidden_) {  // Iterate over hidden_size dimension
      const uint32_t remaining = hidden_ - offset;

      if (remaining >= vec_size) {
        // Full vector processing: can safely load vec_size elements
        const vec_t input_vec =
            *reinterpret_cast<const vec_t*>(input_base + offset);
        const vec_t weight_vec =
            *reinterpret_cast<const vec_t*>(weight_base + offset);

// Vectorized dot product computation
#pragma unroll
        for (uint32_t i = 0; i < vec_size; i++) {
          // input_vec[i] * weight_vec[i] corresponds to:
          // inputs[batch_id, offset+i] * weights[lora_idx, rank_id, offset+i]
          local_sum += static_cast<acc_t>(input_vec[i]) *
                       static_cast<acc_t>(weight_vec[i]);
        }
      } else {
        // Partial vector processing: handle remaining elements less than
        // vec_size
        for (uint32_t i = 0; i < remaining; i++) {
          // inputs[batch_id, offset+i] * weights[lora_idx, rank_id, offset+i]
          local_sum += static_cast<acc_t>(input_base[offset + i]) *
                       static_cast<acc_t>(weight_base[offset + i]);
        }
      }
      offset += stride;  // Jump to next position handled by current thread
    }

    // Workgroup reduction: sum all local_sum values within workgroup
    // Result: complete dot product sum_h(inputs[batch_id, h] *
    // weights[lora_idx, rank_id, h])
    const acc_t group_sum = sycl::reduce_over_group(
        item.get_group(), local_sum, sycl::plus<acc_t>());

    // Only first thread in workgroup writes the result
    if (local_id == 0) {
      // Write to outputs[batch_id, rank_id] += scale * group_sum
      outputs_[batch_id * rank_ + rank_id] +=
          static_cast<output_t>(group_sum * scale_);
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
 * Launch BGMV shrink kernel
 *
 * @param outputs  [batch_size, rank] - Output tensor
 * @param inputs   [batch_size, hidden_size] - Input tensor
 * @param weights  [num_loras, rank, hidden_size] - LoRA weights
 * @param indices  [batch_size] - LoRA index mapping
 * @param batch_size - Batch size
 * @param hidden     - Hidden dimension size
 * @param rank       - LoRA rank
 * @param scale      - Scaling factor
 */
template <typename output_t, typename input_t>
void launch_bgmv_shrink(
    output_t* outputs,
    input_t* inputs,
    input_t* weights,
    int64_t* indices,
    const uint32_t batch_size,
    const uint32_t hidden,
    const uint32_t rank,
    const float scale) {
  // 1. Calculate optimal vector size
  uint32_t vec_bytes = 16;  // Start with 16 bytes
  const auto input_align = reinterpret_cast<uintptr_t>(inputs);
  const auto weight_align = reinterpret_cast<uintptr_t>(weights);
  const uint32_t data_bytes = hidden * sizeof(input_t);  // Bytes per row

  // Choose vector size based on data size and pointer alignment
  while (vec_bytes > sizeof(input_t) &&
         (data_bytes % vec_bytes != 0 || input_align % vec_bytes != 0 ||
          weight_align % vec_bytes != 0)) {
    vec_bytes /= 2;
  }
  uint32_t vec_size = vec_bytes / sizeof(input_t);

  // 2. Get device information
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& dpcpp_queue = vllm::xpu::vllmGetQueue();
  const auto dev_id = at::xpu::current_device();
  const uint32_t max_workgroup_size = vllm::xpu::getMaxWorkGroupSize(dev_id);

  // 3. Calculate optimal workgroup size
  uint32_t workgroup_size = 64;  // Starting workgroup size
  const uint32_t elements_per_thread = (hidden + vec_size - 1) / vec_size;
  // Dynamically adjust workgroup size for better occupancy
  while (workgroup_size < max_workgroup_size) {
    const uint32_t next_size = workgroup_size * 2;
    if (next_size > max_workgroup_size) break;
    // Ensure each thread has sufficient workload
    if (elements_per_thread < next_size / workgroup_size * 4) break;

    workgroup_size = next_size;
  }
  // Limit maximum workgroup size to avoid resource contention
  workgroup_size = sycl::min(workgroup_size, 256u);

  // 4. Set execution configuration
  // Total workgroups = batch_size * rank (one workgroup per (batch, rank) pair
  const uint32_t workgroup_num = batch_size * rank;
  const sycl::range<1> local_range{workgroup_size};
  const sycl::range<1> global_range{workgroup_num * workgroup_size};
  // 5. Submit kernel execution
  dpcpp_queue.submit([&](sycl::handler& cgh) {
    dispatch_vec_size(vec_size, [&](auto vec_c) {
      constexpr int V = vec_c.value;
      vllm::lora::bgmv_shrink_kernel<output_t, input_t, V> kfn(
          outputs, inputs, weights, indices, hidden, rank, scale);
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
    });
  });
}

void validate_lora_a_tensors(
    const at::Tensor& inputs,
    const at::Tensor& lora_a_weights,
    const at::Tensor& output_tensor,
    const at::Tensor& lora_indices_tensor) {
  // Device checks
  TORCH_CHECK(inputs.is_xpu(), "inputs must be on XPU");
  TORCH_CHECK(lora_a_weights.is_xpu(), "lora_a_weights must be on XPU");
  TORCH_CHECK(output_tensor.is_xpu(), "output_tensor must be on XPU");
  TORCH_CHECK(
      lora_indices_tensor.is_xpu(), "lora_indices_tensor must be on XPU");

  // Contiguous checks
  TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(
      lora_a_weights.is_contiguous(), "lora_a_weights must be contiguous");
  TORCH_CHECK(
      output_tensor.is_contiguous(), "output_tensor must be contiguous");
  TORCH_CHECK(
      lora_indices_tensor.is_contiguous(),
      "lora_indices_tensor must be contiguous");

  // Dtype checks
  TORCH_CHECK(
      inputs.scalar_type() == lora_a_weights.scalar_type(),
      "inputs dtype must match lora_a_weights dtype");

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
  TORCH_CHECK(inputs.dim() == 2, "inputs must be 2D [batch_size, hidden_size]");
  TORCH_CHECK(
      output_tensor.dim() == 2, "output_tensor must be 2D [batch_size, rank]");
  TORCH_CHECK(
      lora_indices_tensor.dim() == 1,
      "lora_indices_tensor must be 1D [batch_size]");

  at::Tensor lora_weights = lora_a_weights;
  if (lora_a_weights.dim() == 4) {  // shape: (lora_num,1,rank,hidden_size)
    TORCH_CHECK(
        lora_a_weights.size(1) == 1, "lora_a_weights.size(1) must be 1");
    lora_weights = lora_a_weights.squeeze(1);  // squeeze dim 1
  } else {
    TORCH_CHECK(
        lora_a_weights.dim() == 3,
        "lora_a_weights must be 3D [lora_num, rank, hidden_size]");
  }

  // Shape consistency checks
  TORCH_CHECK(
      inputs.size(1) == lora_weights.size(-1),
      "inputs.size(1) must match lora_a_weights.size(-1)");
  TORCH_CHECK(
      output_tensor.size(1) == lora_weights.size(-2),
      "output_tensor.size(1) must match lora_a_weights.size(-2)");
  TORCH_CHECK(
      inputs.size(0) == output_tensor.size(0),
      "inputs.size(0) must match output_tensor.size(0)");
  TORCH_CHECK(
      inputs.size(0) == lora_indices_tensor.size(0),
      "inputs.size(0) must match lora_indices_tensor.size(0)");
}

/**
 * BGMV shrink operation main function
 *
 * @param outputs [batch_size, rank] - Output tensor, accumulates results
 * @param inputs  [batch_size, hidden_size] - Input feature tensor
 * @param weights [num_loras, rank, hidden_size] - LoRA weight matrix B
 * @param indices [batch_size] - LoRA index for each sample
 * @param scale - Scaling factor
 */
void bgmv_shrink(
    torch::Tensor& outputs,
    const torch::Tensor& inputs,
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    double scale) {
  // 1. Input validation
  validate_lora_a_tensors(inputs, weights, outputs, indices);

  // 2. Get dimension information
  uint32_t batch_size = inputs.size(0);
  uint32_t hidden = inputs.size(1);
  uint32_t rank = outputs.size(1);

  auto scale_f = static_cast<float>(scale);
  // 5. Dispatch based on output type
  VLLM_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "bgmv_shrink", [&]() {
    using output_t = scalar_t;
    switch (inputs.scalar_type()) {
      case at::ScalarType::Half:
        launch_bgmv_shrink<output_t, at::Half>(
            outputs.data_ptr<output_t>(),
            inputs.data_ptr<at::Half>(),
            weights.data_ptr<at::Half>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            hidden,
            rank,
            scale_f);
        break;
      case at::ScalarType::BFloat16:
        launch_bgmv_shrink<output_t, at::BFloat16>(
            outputs.data_ptr<output_t>(),
            inputs.data_ptr<at::BFloat16>(),
            weights.data_ptr<at::BFloat16>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            hidden,
            rank,
            scale_f);
        break;
      default:
        TORCH_CHECK(false, "Unsupported input type: ", inputs.scalar_type());
        break;
    }
  });
}
