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
 * BGMV Expand Kernel with slice
 * ---------------------------------
 * Performs batched matrix–vector multiplication and writes the result
 * into a slice of the output tensor.
 *
 * Template parameters:
 *   - output_t           : Output tensor data type
 *   - input_t            : Input tensor data type
 *   - vec_size           : Number of elements processed per vectorized loop
 *   - subgroup_size      : Subgroup size in SYCL
 *   - use_aligned_vector : Whether to use aligned vector load/store
 *
 * Tensor dimensions:
 *   - inputs   : [batch_size, hidden_size]
 *   - weights  : [num_loras, hidden_size, hidden_size_out]  (LoRA B matrix)
 *   - indices  : [batch_size]
 *   - outputs  : [batch_size, hidden_size_out]
 *
 * Mathematical operation:
 *   For each sample b:
 *     outputs[b] = inputs[b] @ weights[indices[b]]
 *                  + (add_inputs ? inputs[b] : 0)
 */
template <
    typename output_t,
    typename input_t,
    uint32_t vec_size,
    uint32_t subgroup_size,
    bool use_aligned_vector>
class bgmv_expand_kernel {
  using accscalar_t = vllm::xpu::acc_type<output_t>;
  using input_vec_t = vllm::xpu::aligned_vec<input_t, vec_size>;
  using weight_vec_t = vllm::xpu::aligned_vec<output_t, vec_size>;

 private:
  output_t* outputs_;
  const input_t* __restrict__ inputs_;
  const output_t* __restrict__ weights_;
  const int64_t* __restrict__ indices_;
  const uint32_t batch_size_;
  const uint32_t rank_;
  const uint32_t hidden_;
  const uint32_t output_hidden_;
  const uint32_t slice_offset_;
  const bool add_to_output_;
  WorkgroupLocal<accscalar_t> slm_;
  const uint32_t workitem_per_hidden_;
  const uint32_t hidden_per_subgroup_;
  const uint32_t subgroup_num_;
  const uint32_t sg_per_wg_;

 public:
  /**
   * Constructor
   * @param outputs Output tensor [batch_size, output_hidden]
   * @param inputs Input tensor [batch_size, rank]
   * @param weights Weight tensor [num_loras, hidden, rank]
   * @param indices LoRA indices [batch_size]
   * @param batch_size Batch size
   * @param rank LoRA rank
   * @param hidden Hidden dimension
   * @param output_hidden Output hidden dimension
   * @param slice_offset Start offset in output tensor
   * @param add_to_output Whether to accumulate to existing output
   * @param slm Workgroup local memory accessor
   * @param workitem_per_hidden Number of workitems per hidden element
   * @param hidden_per_subgroup Number of hidden elements per subgroup
   * @param subgroup_num Total number of subgroups
   * @param sg_per_wg Number of subgroups per workgroup
   */
  bgmv_expand_kernel(
      output_t* __restrict__ outputs,
      const input_t* __restrict__ inputs,
      const output_t* __restrict__ weights,
      const int64_t* __restrict__ indices,
      const uint32_t batch_size,
      const uint32_t rank,
      const uint32_t hidden,
      const uint32_t output_hidden,
      const uint32_t slice_offset,
      const bool add_to_output,
      WorkgroupLocal<accscalar_t> slm,
      const uint32_t workitem_per_hidden,
      const uint32_t hidden_per_subgroup,
      const uint32_t subgroup_num,
      const uint32_t sg_per_wg)
      : outputs_(outputs),
        inputs_(inputs),
        weights_(weights),
        indices_(indices),
        batch_size_(batch_size),
        rank_(rank),
        hidden_(hidden),
        output_hidden_(output_hidden),
        slice_offset_(slice_offset),
        add_to_output_(add_to_output),
        slm_(slm),
        workitem_per_hidden_(workitem_per_hidden),
        hidden_per_subgroup_(hidden_per_subgroup),
        subgroup_num_(subgroup_num),
        sg_per_wg_(sg_per_wg) {}

  void operator() [[sycl::reqd_sub_group_size(subgroup_size)]] (
      sycl::nd_item<1> item) const {
    sycl::group<1> g = item.get_group();
    sycl::sub_group sg = item.get_sub_group();
    uint32_t group_id = g.get_group_linear_id();
    uint32_t subgroup_id = sg.get_group_linear_id() + group_id * sg_per_wg_;
    if (subgroup_id >= subgroup_num_) {
      return;
    }

    const uint32_t item_id = g.get_local_linear_id();
    const uint32_t line_id = sg.get_local_linear_id();
    const uint32_t hidden_id_in_subgroup = line_id / workitem_per_hidden_;
    const uint32_t vec_id = line_id % workitem_per_hidden_;
    const uint32_t hidden_linear_id =
        subgroup_id * hidden_per_subgroup_ + hidden_id_in_subgroup;

    if (hidden_id_in_subgroup >= hidden_per_subgroup_ ||
        hidden_linear_id >= batch_size_ * hidden_) {
      return;
    }

    const uint32_t batch_id = hidden_linear_id / hidden_;
    const uint32_t hidden_id = hidden_linear_id % hidden_;

    const int64_t lora_idx = indices_[batch_id];
    if (lora_idx < 0) return;

    const input_t* __restrict__ input_ptr =
        inputs_ + static_cast<size_t>(batch_id) * rank_;
    const output_t* __restrict__ weight_ptr =
        weights_ +
        (static_cast<size_t>(lora_idx) * hidden_ + hidden_id) * rank_;

    accscalar_t local_result = 0;

    uint32_t offset = vec_id * vec_size;
    while (offset < rank_) {
      const input_t* __restrict__ input_base = input_ptr + offset;
      const output_t* __restrict__ weight_base = weight_ptr + offset;
      if constexpr (use_aligned_vector) {
        const input_vec_t in_vec =
            *reinterpret_cast<const input_vec_t*>(input_base);
        const weight_vec_t wt_vec =
            *reinterpret_cast<const weight_vec_t*>(weight_base);
#pragma unroll(vec_size)
        for (uint32_t i = 0; i < vec_size; ++i) {
          local_result += static_cast<accscalar_t>(in_vec[i]) *
                          static_cast<accscalar_t>(wt_vec[i]);
        }
      } else {
#pragma unroll(vec_size)
        for (uint32_t i = 0; i < vec_size; ++i) {
          const uint32_t idx = offset + i;
          if (idx >= rank_) break;
          local_result += static_cast<accscalar_t>(input_base[i]) *
                          static_cast<accscalar_t>(weight_base[i]);
        }
      }
      offset += workitem_per_hidden_ * vec_size;
    }

    const uint32_t slm_base = item_id - vec_id;
    slm_[slm_base + vec_id] = local_result;
    sycl::group_barrier(sg);

    if (vec_id == 0) {
      accscalar_t result = 0;
#pragma unroll
      for (uint32_t i = 0; i < workitem_per_hidden_; ++i) {
        result += slm_[slm_base + i];
      }
      const size_t out_off = static_cast<size_t>(batch_id) * output_hidden_ +
                             slice_offset_ + hidden_id;
      if (add_to_output_) {
        outputs_[out_off] = static_cast<output_t>(
            static_cast<accscalar_t>(outputs_[out_off]) + result);
      } else {
        outputs_[out_off] = static_cast<output_t>(result);
      }
    }
  }
};
}  // namespace vllm::lora

/**
 * Launch BGMV expand kernel for a sliced output
 * @param outputs Output tensor [batch_size, output_hidden]
 * @param inputs Input tensor [batch_size, rank]
 * @param weights Weight tensor [num_loras, hidden, rank]
 * @param indices LoRA indices [batch_size]
 * @param batch_size Batch size
 * @param rank LoRA rank
 * @param hidden Hidden dimension
 * @param output_hidden Output hidden dimension
 * @param slice_offset Start offset in output
 * @param add_to_output Whether to accumulate to output
 */
template <typename output_t, typename input_t>
void launch_bgmv_expand_with_slice(
    output_t* outputs,
    const input_t* inputs,
    const output_t* weights,
    const int64_t* indices,
    const uint32_t batch_size,
    const uint32_t rank,
    const uint32_t hidden,
    const uint32_t output_hidden,
    const uint32_t slice_offset,
    const bool add_to_output) {
  if (batch_size == 0 || rank == 0 || hidden == 0) return;

  constexpr uint32_t vec_size = 16 / sizeof(input_t);
  constexpr uint32_t subgroup_size = 32;
  const bool use_aligned_vector =
      (rank % vec_size == 0 && reinterpret_cast<uintptr_t>(inputs) % 16 == 0 &&
       reinterpret_cast<uintptr_t>(weights) % 16 == 0);

  // Use several workitems to write one element to output with [batch_size,
  // hidden]. Use at most one subgroup to process one element.
  const uint32_t workitem_per_hidden =
      std::min<uint32_t>((rank + vec_size - 1) / vec_size, subgroup_size);

  const uint32_t hidden_per_subgroup = subgroup_size / workitem_per_hidden;
  const uint32_t subgroup_num =
      (batch_size * hidden + hidden_per_subgroup - 1) / hidden_per_subgroup;

  at::DeviceGuard device_guard(at::Device(at::kXPU, at::xpu::current_device()));
  auto& q = vllm::xpu::vllmGetQueue();
  const auto dev_id = at::xpu::current_device();
  const uint32_t max_workgroup_size = vllm::xpu::getMaxWorkGroupSize(dev_id);

  const uint32_t workgroup_size =
      std::min<uint32_t>(max_workgroup_size, subgroup_num * subgroup_size);
  const uint32_t sg_per_wg = workgroup_size / subgroup_size;
  const uint32_t workgroup_num = (subgroup_num + sg_per_wg - 1) / sg_per_wg;

  sycl::range<1> local_range{workgroup_size};
  sycl::range<1> global_range{workgroup_num * workgroup_size};
  q.submit([&](sycl::handler& cgh) {
    WorkgroupLocal<vllm::xpu::acc_type<output_t>> slm(
        sycl::range(workgroup_size), cgh);
    if (use_aligned_vector) {
      vllm::lora::
          bgmv_expand_kernel<output_t, input_t, vec_size, subgroup_size, true>
              kfn(outputs,
                  inputs,
                  weights,
                  indices,
                  batch_size,
                  rank,
                  hidden,
                  output_hidden,
                  slice_offset,
                  add_to_output,
                  slm,
                  workitem_per_hidden,
                  hidden_per_subgroup,
                  subgroup_num,
                  sg_per_wg);
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
    } else {
      vllm::lora::
          bgmv_expand_kernel<output_t, input_t, vec_size, subgroup_size, false>
              kfn(outputs,
                  inputs,
                  weights,
                  indices,
                  batch_size,
                  rank,
                  hidden,
                  output_hidden,
                  slice_offset,
                  add_to_output,
                  slm,
                  workitem_per_hidden,
                  hidden_per_subgroup,
                  subgroup_num,
                  sg_per_wg);
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
    }
  });
}

void validate_lora_b_tensors(
    const at::Tensor& inputs,
    const at::Tensor& lora_b_weights,
    const at::Tensor& output_tensor,
    const at::Tensor& lora_indices_tensor) {
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
      inputs.scalar_type() == at::kFloat ||
          inputs.scalar_type() == at::kBFloat16 ||
          inputs.scalar_type() == at::kHalf,
      "inputs must be float16, bfloat16, or float32");

  TORCH_CHECK(
      lora_b_weights.scalar_type() == at::kBFloat16 ||
          lora_b_weights.scalar_type() == at::kHalf,
      "lora_b_weights must be float16 or bfloat16");

  TORCH_CHECK(
      output_tensor.scalar_type() == lora_b_weights.scalar_type(),
      "output_tensor dtype must match lora_b_weights dtype");

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

  at::Tensor lora_weights = lora_b_weights;
  if (lora_b_weights.dim() == 4) {  // shape: (lora_num,1,hidden_size,rank)
    TORCH_CHECK(
        lora_b_weights.size(1) == 1, "lora_b_weights.size(1) must be 1");
    lora_weights = lora_b_weights.squeeze(1);  // squeeze dim 1
  } else {
    TORCH_CHECK(
        lora_b_weights.dim() == 3,
        "lora_b_weights must be 3D [lora_num, hidden_size, rank]");
  }

  // Shape consistency checks
  TORCH_CHECK(
      inputs.size(1) == lora_weights.size(-1),
      "inputs.size(1) must match lora_b_weights.size(-1)");
  TORCH_CHECK(
      inputs.size(0) == output_tensor.size(0),
      "inputs.size(0) must match output_tensor.size(0)");
  TORCH_CHECK(
      inputs.size(0) == lora_indices_tensor.size(0),
      "inputs.size(0) must match lora_indices_tensor.size(0)");
}

void bgmv_expand_slice(
    torch::Tensor& outputs,
    const torch::Tensor& inputs,
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    const int64_t slice_offset,
    const bool add_to_output) {
  validate_lora_b_tensors(inputs, weights, outputs, indices);
  TORCH_CHECK(slice_offset >= 0, "slice_offset must be non-negative");

  uint32_t batch_size = inputs.size(0);
  uint32_t rank = inputs.size(1);
  uint32_t hidden = weights.size(1);
  uint32_t output_hidden = outputs.size(1);
  VLLM_DISPATCH_HALF_TYPES(inputs.scalar_type(), "bgmv_expand_slice", [&]() {
    using input_t = scalar_t;
    switch (outputs.scalar_type()) {
      case at::ScalarType::Half:
        launch_bgmv_expand_with_slice<at::Half, input_t>(
            outputs.data_ptr<at::Half>(),
            inputs.data_ptr<input_t>(),
            weights.data_ptr<at::Half>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            rank,
            hidden,
            output_hidden,
            slice_offset,
            add_to_output);
        break;
      case at::ScalarType::BFloat16:
        launch_bgmv_expand_with_slice<at::BFloat16, input_t>(
            outputs.data_ptr<at::BFloat16>(),
            inputs.data_ptr<input_t>(),
            weights.data_ptr<at::BFloat16>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            rank,
            hidden,
            output_hidden,
            slice_offset,
            add_to_output);
        break;
      default:
        TORCH_CHECK(false, "Unsupported output type: ", inputs.scalar_type());
    }
  });
};

void bgmv_expand(
    torch::Tensor& outputs,
    const torch::Tensor& inputs,
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    bool add_to_output) {
  validate_lora_b_tensors(inputs, weights, outputs, indices);
  bgmv_expand_slice(
      outputs, inputs, weights, indices, /*slice_offset=*/0, add_to_output);
};