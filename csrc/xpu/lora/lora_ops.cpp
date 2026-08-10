// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Multi-slice LoRA shrink and expand kernels for XPU.
// Processes ALL slices in a single kernel launch, avoiding the per-slice
// Python loop that causes view-mutation issues under torch.compile.
//
// Shrink (lora_shrink):
//   For each slice s, batch b, and rank r:
//     output[s, b, r] = scale * Σ_h (inputs[b, h] * weights_s[indices[b], r,
//     h])
//
//   Tensor shapes:
//     inputs  : [batch_size, hidden_size]
//     weights : num_slices tensors, each [num_loras, 1, rank, hidden_size]
//     output  : [num_slices, batch_size, rank]
//     indices : [batch_size]
//
// Expand (lora_expand):
//   For each slice s, batch b:
//     output[b, offset_s : offset_s + slice_size_s] +=
//         inputs[s, b, :] @ weights_s[indices[b], :, :]
//
//   Tensor shapes:
//     inputs  : [num_slices, batch_size, rank]
//     weights : num_slices tensors, each [num_loras, 1, slice_size, rank]
//     output  : [batch_size, total_output_dim]
//     indices : [batch_size]

#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <type_traits>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include "dispatch_utils.h"
#include "utils.h"
#include "utils/dpcpp.h"

namespace vllm::lora {

// Maximum number of slices supported (QKV typically has 3).
// Weight pointers are stored in a fixed-size array passed to the kernel.
constexpr int kMaxSlices = 8;

// ============================================================================
// Shrink kernel
// ============================================================================

/**
 * Multi-slice BGMV Shrink Kernel
 *
 * Each workgroup computes one (slice, batch, rank) output element by
 * performing a collaborative dot-product over the hidden dimension.
 *
 * Template parameters:
 *   output_t  : output element type
 *   input_t   : input element type
 *   weight_t  : weight element type
 *   vec_size  : number of elements per vectorized load
 */
template <
    typename output_t,
    typename input_t,
    typename weight_t,
    uint32_t vec_size>
class lora_shrink_kernel {
 private:
  output_t* __restrict__ output_;
  const input_t* __restrict__ input_;
  const int32_t* __restrict__ indices_;
  const uint32_t batch_size_;
  const uint32_t hidden_;
  const uint32_t rank_;
  const float scale_;
  const uint32_t num_loras_;

  // Per-slice weight pointers and strides (rank * hidden per LoRA).
  // Using a fixed-size array avoids dynamic allocation on device.
  const weight_t* __restrict__ weight_ptrs_[kMaxSlices];
  uint64_t weight_lora_stride_[kMaxSlices];  // stride between LoRA adapters

 public:
  using acc_t = float;
  using input_vec_t = vllm::xpu::aligned_vec<input_t, vec_size>;
  using weight_vec_t = vllm::xpu::aligned_vec<weight_t, vec_size>;

  lora_shrink_kernel(
      output_t* output,
      const input_t* input,
      const int32_t* indices,
      const uint32_t batch_size,
      const uint32_t hidden,
      const uint32_t rank,
      const float scale,
      const uint32_t num_loras,
      const uint32_t num_slices,
      const weight_t* const* weight_ptrs,
      const uint64_t* weight_lora_strides)
      : output_(output),
        input_(input),
        indices_(indices),
        batch_size_(batch_size),
        hidden_(hidden),
        rank_(rank),
        scale_(scale),
        num_loras_(num_loras) {
    for (uint32_t i = 0; i < num_slices && i < kMaxSlices; ++i) {
      weight_ptrs_[i] = weight_ptrs[i];
      weight_lora_stride_[i] = weight_lora_strides[i];
    }
  }

  [[sycl::reqd_sub_group_size(32)]] void
  operator()(sycl::nd_item<1> item) const {
    const uint32_t local_id = item.get_local_linear_id();
    const uint32_t group_id = item.get_group_linear_id();
    const uint32_t group_size = item.get_local_range(0);

    // Decompose group_id → (slice_id, batch_id, rank_id)
    // Layout: group_id = slice_id * (batch_size * rank) + batch_id * rank +
    // rank_id
    const uint32_t batch_rank = batch_size_ * rank_;
    const uint32_t slice_id = group_id / batch_rank;
    const uint32_t remainder = group_id % batch_rank;
    const uint32_t batch_id = remainder / rank_;
    const uint32_t rank_id = remainder % rank_;

    const int32_t lora_idx = indices_[batch_id];

    acc_t local_sum = 0;
    const bool valid =
        (lora_idx >= 0 && static_cast<uint32_t>(lora_idx) < num_loras_);

    if (valid) {
      const input_t* __restrict__ input_base = input_ + batch_id * hidden_;
      const weight_t* __restrict__ weight_base =
          weight_ptrs_[slice_id] +
          static_cast<uint64_t>(lora_idx) * weight_lora_stride_[slice_id] +
          rank_id * hidden_;

      uint32_t offset = local_id * vec_size;
      const uint32_t stride = group_size * vec_size;

      while (offset < hidden_) {
        const uint32_t remaining = hidden_ - offset;
        if (remaining >= vec_size) {
          const input_vec_t input_vec =
              *reinterpret_cast<const input_vec_t*>(input_base + offset);
          const weight_vec_t weight_vec =
              *reinterpret_cast<const weight_vec_t*>(weight_base + offset);
#pragma unroll
          for (uint32_t i = 0; i < vec_size; i++) {
            local_sum = sycl::mad(
                static_cast<acc_t>(input_vec[i]),
                static_cast<acc_t>(weight_vec[i]),
                local_sum);
          }
        } else {
          for (uint32_t i = 0; i < remaining; i++) {
            local_sum = sycl::mad(
                static_cast<acc_t>(input_base[offset + i]),
                static_cast<acc_t>(weight_base[offset + i]),
                local_sum);
          }
        }
        offset += stride;
      }
    }

    const acc_t group_sum = sycl::reduce_over_group(
        item.get_group(), local_sum, sycl::plus<acc_t>());

    if (local_id == 0) {
      // output layout: [num_slices, batch_size, rank]
      // Always write (including 0 for invalid lora_idx) so the host no
      // longer needs a separate output_tensor.zero_() pass before launch.
      const uint64_t out_idx =
          static_cast<uint64_t>(slice_id) * batch_size_ * rank_ +
          batch_id * rank_ + rank_id;
      output_[out_idx] =
          valid ? static_cast<output_t>(group_sum * scale_)
                : static_cast<output_t>(0);
    }
  }
};

// ============================================================================
// Expand kernel
// ============================================================================

/**
 * Multi-slice BGMV Expand Kernel
 *
 * Uses a 2D launch: dimension 1 (the outer group dimension in this 2D
 * nd_range) selects the slice, so slice_id = item.get_group(1) directly —
 * no runtime division/modulo or linear prefix-sum search is needed to
 * recover it (the per-slice bgmv_expand_slice loop this replaces also
 * never needed one, since each launch only ever served a single,
 * statically-known slice). Dimension 1 iterates over that slice's
 * (batch, col) elements with a grid-stride loop, and each slice gets its
 * own workgroup size (see launch_lora_expand), so small slices don't pay
 * for oversized launches.
 *
 * Each thread computes one output element by performing a dot-product
 * over the LoRA rank dimension.
 *
 * Template parameters:
 *   output_t   : output element type
 *   input_t    : input element type
 *   weight_t   : weight element type
 *   vec_size   : number of elements per vectorized load
 *   add_inputs : whether to accumulate (true) or overwrite (false)
 */
template <
    typename output_t,
    typename input_t,
    typename weight_t,
    uint32_t vec_size,
    bool add_inputs>
class lora_expand_kernel {
 private:
  output_t* __restrict__ output_;
  const input_t* __restrict__ input_;
  const int32_t* __restrict__ indices_;
  const uint32_t batch_size_;
  const uint32_t output_dim_;  // total output columns
  const uint32_t rank_;
  const uint32_t num_loras_;

  // Per-slice metadata
  const weight_t* __restrict__ weight_ptrs_[kMaxSlices];
  uint64_t weight_lora_stride_[kMaxSlices];  // stride between LoRAs
  uint32_t slice_offsets_[kMaxSlices];
  uint32_t slice_sizes_[kMaxSlices];

 public:
  using acc_t = float;
  using input_vec_t = vllm::xpu::aligned_vec<input_t, vec_size>;
  using weight_vec_t = vllm::xpu::aligned_vec<weight_t, vec_size>;

  lora_expand_kernel(
      output_t* output,
      const input_t* input,
      const int32_t* indices,
      const uint32_t batch_size,
      const uint32_t output_dim,
      const uint32_t rank,
      const uint32_t num_loras,
      const uint32_t num_slices,
      const weight_t* const* weight_ptrs,
      const uint64_t* weight_lora_strides,
      const uint32_t* slice_offsets,
      const uint32_t* slice_sizes)
      : output_(output),
        input_(input),
        indices_(indices),
        batch_size_(batch_size),
        output_dim_(output_dim),
        rank_(rank),
        num_loras_(num_loras) {
    for (uint32_t i = 0; i < num_slices && i < kMaxSlices; ++i) {
      weight_ptrs_[i] = weight_ptrs[i];
      weight_lora_stride_[i] = weight_lora_strides[i];
      slice_offsets_[i] = slice_offsets[i];
      slice_sizes_[i] = slice_sizes[i];
    }
  }

  // 2D launch: dim 1 = slice_id (one "row" of workgroups per slice), so
  // slice_id is read directly from item.get_group(1) — no per-thread
  // division/modulo or linear search over cumulative slice sizes is
  // needed to recover it.
  [[sycl::reqd_sub_group_size(32)]] void
  operator()(sycl::nd_item<2> item) const {
    const uint32_t slice_id = item.get_group(1);
    const uint32_t slice_size = slice_sizes_[slice_id];
    const uint32_t slice_offset = slice_offsets_[slice_id];
    const uint32_t slice_elements = batch_size_ * slice_size;

    const uint32_t local_id = item.get_global_id(0);
    const uint32_t total_threads_per_slice =
        item.get_local_range(0) * item.get_group_range(0);

    const weight_t* __restrict__ weight_ptr = weight_ptrs_[slice_id];
    const uint64_t weight_lora_stride = weight_lora_stride_[slice_id];
    const input_t* __restrict__ slice_input_base =
        input_ + static_cast<uint64_t>(slice_id) * batch_size_ * rank_;

    for (uint32_t elem = local_id; elem < slice_elements;
         elem += total_threads_per_slice) {
      const uint32_t batch_id = elem / slice_size;
      const uint32_t col_in_slice = elem % slice_size;

      // Early exit: skip tokens that are not assigned to any LoRA.
      const int32_t lora_idx = indices_[batch_id];
      if (lora_idx < 0 || static_cast<uint32_t>(lora_idx) >= num_loras_)
        continue;

      const input_t* __restrict__ input_base =
          slice_input_base + batch_id * rank_;

      // Weight for this slice and LoRA
      const weight_t* __restrict__ weight_base =
          weight_ptr + static_cast<uint64_t>(lora_idx) * weight_lora_stride +
          col_in_slice * rank_;

      // Dot product over rank
      acc_t sum = 0;
      const uint32_t full_vectors = rank_ / vec_size;
      for (uint32_t j = 0; j < full_vectors; j++) {
        const input_vec_t iv =
            *reinterpret_cast<const input_vec_t*>(input_base + j * vec_size);
        const weight_vec_t wv =
            *reinterpret_cast<const weight_vec_t*>(weight_base + j * vec_size);
#pragma unroll
        for (uint32_t k = 0; k < vec_size; k++) {
          sum = sycl::mad(
              static_cast<acc_t>(iv[k]), static_cast<acc_t>(wv[k]), sum);
        }
      }
      const uint32_t tail = rank_ % vec_size;
      if (tail) {
        const uint32_t base_off = full_vectors * vec_size;
        for (uint32_t j = 0; j < tail; j++) {
          sum += static_cast<acc_t>(input_base[base_off + j]) *
                 static_cast<acc_t>(weight_base[base_off + j]);
        }
      }

      // Write to output
      const uint32_t out_col = slice_offset + col_in_slice;
      if (out_col < output_dim_) {
        const uint64_t out_idx =
            static_cast<uint64_t>(batch_id) * output_dim_ + out_col;
        if constexpr (add_inputs) {
          output_[out_idx] += static_cast<output_t>(sum);
        } else {
          output_[out_idx] = static_cast<output_t>(sum);
        }
      }
    }
  }
};

}  // namespace vllm::lora

// ============================================================================
// Vector-size helpers (shared by shrink and expand launchers)
// ============================================================================

// Returns the largest power-of-2 vec_size such that the base pointer is
// vec_size*sizeof(T)-aligned AND row_elems*sizeof(T) is divisible by
// vec_size*sizeof(T).  Falls back to 1 when no wider width is safe.
template <typename T>
static uint32_t aligned_vec_size(const T* ptr, uint32_t row_elems) {
  uint32_t vec_bytes = 16;
  const uint32_t row_bytes = row_elems * static_cast<uint32_t>(sizeof(T));
  const auto base_align = reinterpret_cast<uintptr_t>(ptr);
  while (vec_bytes > sizeof(T) &&
         (row_bytes % vec_bytes != 0 || base_align % vec_bytes != 0)) {
    vec_bytes /= 2;
  }
  if (vec_bytes < sizeof(T)) vec_bytes = static_cast<uint32_t>(sizeof(T));
  return vec_bytes / static_cast<uint32_t>(sizeof(T));
}

// Dispatch a runtime vec_size (1/2/4/8) to a compile-time constant for
// kernel template instantiation.
template <typename Fn>
static void dispatch_vec_size(uint32_t vec_size, Fn&& fn) {
  switch (vec_size) {
    case 8:
      fn(std::integral_constant<uint32_t, 8>{});
      break;
    case 4:
      fn(std::integral_constant<uint32_t, 4>{});
      break;
    case 2:
      fn(std::integral_constant<uint32_t, 2>{});
      break;
    default:
      fn(std::integral_constant<uint32_t, 1>{});
      break;
  }
}

// ============================================================================
// Shrink launcher
// ============================================================================

template <typename output_t, typename input_t, typename weight_t>
void launch_lora_shrink(
    output_t* output,
    const input_t* input,
    const int32_t* indices,
    const uint32_t batch_size,
    const uint32_t hidden,
    const uint32_t rank,
    const float scale,
    const uint32_t num_loras,
    const uint32_t num_slices,
    const weight_t* const* weight_ptrs,
    const uint64_t* weight_lora_strides) {
  // Compute runtime vec_size: must be safe for all input rows (hidden wide)
  // and all weight rows (hidden wide) across every slice.
  // For row offset batch_id*hidden to be 16-byte aligned, hidden*sizeof(T)
  // must be divisible by 16; similarly for rank_id*hidden weight offsets.
  uint32_t vec_size = aligned_vec_size(input, hidden);
  for (uint32_t i = 0; i < num_slices; ++i)
    vec_size = std::min(vec_size, aligned_vec_size(weight_ptrs[i], hidden));

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& dpcpp_queue = vllm::xpu::vllmGetQueue();
  const auto dev_id = at::xpu::current_device();
  const uint32_t max_wg_size = vllm::xpu::getMaxWorkGroupSize(dev_id);

  const uint32_t wg_num = num_slices * batch_size * rank;

  using sycl_output_t = typename vllm::xpu::SyclTypeTrait<output_t>::Type;
  using sycl_input_t = typename vllm::xpu::SyclTypeTrait<input_t>::Type;
  using sycl_weight_t = typename vllm::xpu::SyclTypeTrait<weight_t>::Type;

  const sycl_weight_t* sycl_weight_ptrs[vllm::lora::kMaxSlices];
  for (uint32_t i = 0; i < num_slices; ++i) {
    sycl_weight_ptrs[i] =
        reinterpret_cast<const sycl_weight_t*>(weight_ptrs[i]);
  }

  dispatch_vec_size(vec_size, [&](auto vec_c) {
    constexpr uint32_t VS = decltype(vec_c)::value;

    // Workgroup sizing: each workgroup reduces over the hidden dim.
    // Pick a power-of-2 size so that every thread has at least a few
    // vector iterations of real work; oversized workgroups waste EU lanes.
    const uint32_t vec_elems = (hidden + VS - 1) / VS;
    uint32_t wg_size = 32;  // minimum: one sub-group
    while (wg_size * 2 <= max_wg_size && wg_size * 2 <= 256u &&
           vec_elems >= wg_size * 2) {
      wg_size *= 2;
    }

    const sycl::range<1> local_range{wg_size};
    const sycl::range<1> global_range{wg_num * wg_size};

    dpcpp_queue.submit([&](sycl::handler& cgh) {
      vllm::lora::
          lora_shrink_kernel<sycl_output_t, sycl_input_t, sycl_weight_t, VS>
              kfn(reinterpret_cast<sycl_output_t*>(output),
                  reinterpret_cast<const sycl_input_t*>(input),
                  indices,
                  batch_size,
                  hidden,
                  rank,
                  scale,
                  num_loras,
                  num_slices,
                  sycl_weight_ptrs,
                  weight_lora_strides);
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
    });
  });
}

// ============================================================================
// Expand launcher
// ============================================================================

template <
    typename output_t,
    typename input_t,
    typename weight_t,
    bool add_inputs>
void launch_lora_expand(
    output_t* output,
    const input_t* input,
    const int32_t* indices,
    const uint32_t batch_size,
    const uint32_t output_dim,
    const uint32_t rank,
    const uint32_t num_loras,
    const uint32_t num_slices,
    const weight_t* const* weight_ptrs,
    const uint64_t* weight_lora_strides,
    const uint32_t* slice_offsets,
    const uint32_t* slice_sizes) {
  // Compute runtime vec_size: must be safe for all input rows (rank wide)
  // and all weight rows (rank wide) across every slice.
  // For row offset batch_id*rank (and col_in_slice*rank for weights) to be
  // 16-byte aligned, rank*sizeof(T) must be divisible by 16.
  uint32_t vec_size = aligned_vec_size(input, rank);
  for (uint32_t i = 0; i < num_slices; ++i)
    vec_size = std::min(vec_size, aligned_vec_size(weight_ptrs[i], rank));

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& dpcpp_queue = vllm::xpu::vllmGetQueue();
  const auto dev_id = at::xpu::current_device();
  const uint32_t max_wg_size = vllm::xpu::getMaxWorkGroupSize(dev_id);

  using sycl_output_t = typename vllm::xpu::SyclTypeTrait<output_t>::Type;
  using sycl_input_t = typename vllm::xpu::SyclTypeTrait<input_t>::Type;
  using sycl_weight_t = typename vllm::xpu::SyclTypeTrait<weight_t>::Type;

  const sycl_weight_t* sycl_weight_ptrs[vllm::lora::kMaxSlices];
  for (uint32_t i = 0; i < num_slices; ++i) {
    sycl_weight_ptrs[i] =
        reinterpret_cast<const sycl_weight_t*>(weight_ptrs[i]);
  }

  dispatch_vec_size(vec_size, [&](auto vec_c) {
    constexpr uint32_t VS = decltype(vec_c)::value;

    // Per-slice workgroup configuration: pick a workgroup size based on
    // how many (batch, col) elements the *largest* slice has, then size
    // dim 0 so each slice gets its own row of workgroups (dim 1). Since
    // slice_id == item.get_group(1), no per-thread division/modulo or
    // linear search is needed to recover it inside the kernel — mirroring
    // how the old per-slice bgmv_expand_slice loop never needed to know
    // "which slice" at all (a single launch only ever served one,
    // statically-known slice).
    uint32_t max_slice_elements = 0;
    for (uint32_t s = 0; s < num_slices; ++s) {
      max_slice_elements =
          std::max(max_slice_elements, batch_size * slice_sizes[s]);
    }

    // Workgroup sizing: identical doubling pattern to launch_lora_shrink's
    // hidden-dim reduction sizing above, just driven by the largest
    // slice's (batch * slice_size) element count instead of `hidden`.
    uint32_t wg_size = 32;  // minimum: one sub-group
    while (wg_size * 2 <= max_wg_size && wg_size * 2 <= 256u &&
           max_slice_elements >= wg_size * 2) {
      wg_size *= 2;
    }

    const uint32_t vecs_per_dot = std::max((rank + VS - 1) / VS, 1u);
    constexpr uint32_t kMaxVecLoadsPerThread = 256;
    const uint32_t max_elems_per_thread =
        std::max(1u, kMaxVecLoadsPerThread / vecs_per_dot);
    const uint32_t min_total_threads =
        (max_slice_elements + max_elems_per_thread - 1) / max_elems_per_thread;
    const uint32_t ideal_wgs =
        (max_slice_elements + wg_size - 1) / wg_size;
    uint32_t num_wgs_per_slice =
        std::min((min_total_threads + wg_size - 1) / wg_size, ideal_wgs);
    num_wgs_per_slice = std::max(num_wgs_per_slice, 1u);

    const sycl::range<2> local_range{wg_size, 1};
    const sycl::range<2> global_range{
        num_wgs_per_slice * wg_size, num_slices};

    dpcpp_queue.submit([&](sycl::handler& cgh) {
      vllm::lora::lora_expand_kernel<
          sycl_output_t,
          sycl_input_t,
          sycl_weight_t,
          VS,
          add_inputs>
          kfn(reinterpret_cast<sycl_output_t*>(output),
              reinterpret_cast<const sycl_input_t*>(input),
              indices,
              batch_size,
              output_dim,
              rank,
              num_loras,
              num_slices,
              sycl_weight_ptrs,
              weight_lora_strides,
              slice_offsets,
              slice_sizes);
      cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
    });
  });
}

// ============================================================================
// Host entry points
// ============================================================================

/**
 * Multi-slice LoRA shrink op entry point.
 *
 * @param inputs          [batch_size, hidden_size]
 * @param lora_a_weights  list of tensors, each [num_loras, 1, rank,
 * hidden_size]
 * @param output_tensor   [num_slices, batch_size, rank]  (zeroed, accumulated
 * in-place)
 * @param lora_indices    [batch_size]
 * @param scaling         scaling factor
 */
void lora_shrink(
    const at::Tensor& inputs,
    const std::vector<at::Tensor>& lora_a_weights,
    at::Tensor& output_tensor,
    const at::Tensor& lora_indices,
    double scaling) {
  const int64_t num_slices = static_cast<int64_t>(lora_a_weights.size());
  TORCH_CHECK(num_slices > 0, "lora_a_weights must be non-empty");
  TORCH_CHECK(
      num_slices <= vllm::lora::kMaxSlices,
      "Too many slices (",
      num_slices,
      "), max is ",
      vllm::lora::kMaxSlices);

  // Validate inputs
  TORCH_CHECK(
      inputs.is_xpu() && inputs.is_contiguous(),
      "inputs must be contiguous XPU tensor");
  TORCH_CHECK(inputs.dim() == 2, "inputs must be 2D [batch_size, hidden_size]");
  TORCH_CHECK(
      output_tensor.is_xpu() && output_tensor.is_contiguous(),
      "output_tensor must be contiguous XPU tensor");
  TORCH_CHECK(
      output_tensor.dim() == 3,
      "output_tensor must be 3D [num_slices, batch_size, rank]");
  TORCH_CHECK(
      output_tensor.size(0) == num_slices,
      "output_tensor.size(0) must match number of weight slices");
  TORCH_CHECK(
      lora_indices.is_xpu() && lora_indices.is_contiguous(),
      "lora_indices must be contiguous XPU tensor");
  TORCH_CHECK(
      lora_indices.scalar_type() == at::kInt, "lora_indices must be int32");

  const uint32_t batch_size = inputs.size(0);
  const uint32_t hidden = inputs.size(1);

  TORCH_CHECK(
      lora_indices.numel() == static_cast<int64_t>(batch_size),
      "lora_indices.numel() (",
      lora_indices.numel(),
      ") must equal batch_size (",
      batch_size,
      ")");

  if (batch_size == 0) return;

  // Process weight tensors: squeeze dim-1 if 4D, validate shapes.
  // Fixed-size array (num_slices <= kMaxSlices) avoids a heap allocation
  // on this hot per-call path.
  std::array<at::Tensor, vllm::lora::kMaxSlices> weights_3d;
  for (int64_t s = 0; s < num_slices; ++s) {
    at::Tensor w = lora_a_weights[s];
    TORCH_CHECK(
        w.is_xpu() && w.is_contiguous(),
        "lora_a_weights[",
        s,
        "] must be contiguous XPU tensor");
    if (w.dim() == 4) {
      TORCH_CHECK(w.size(1) == 1, "lora_a_weights[", s, "].size(1) must be 1");
      w = w.squeeze(1);
    }
    TORCH_CHECK(
        w.dim() == 3, "lora_a_weights[", s, "] must be 3D after squeeze");
    TORCH_CHECK(
        w.size(-1) == static_cast<int64_t>(hidden),
        "weight hidden dim mismatch at slice ",
        s);
    weights_3d[s] = std::move(w);
  }

  const uint32_t rank = static_cast<uint32_t>(weights_3d[0].size(-2));
  const uint32_t num_loras = static_cast<uint32_t>(weights_3d[0].size(0));
  const auto weight_dtype = weights_3d[0].scalar_type();
  for (int64_t s = 1; s < num_slices; ++s) {
    TORCH_CHECK(
        weights_3d[s].size(-2) == static_cast<int64_t>(rank),
        "lora_a_weights[",
        s,
        "] rank mismatch: expected ",
        rank,
        " but got ",
        weights_3d[s].size(-2));
    TORCH_CHECK(
        weights_3d[s].size(0) == static_cast<int64_t>(num_loras),
        "lora_a_weights[",
        s,
        "] num_loras mismatch: expected ",
        num_loras,
        " but got ",
        weights_3d[s].size(0));
    TORCH_CHECK(
        weights_3d[s].scalar_type() == weight_dtype,
        "lora_a_weights[",
        s,
        "] dtype mismatches slice 0");
  }

  TORCH_CHECK(
      output_tensor.size(1) == static_cast<int64_t>(batch_size),
      "output_tensor batch dim mismatch");
  TORCH_CHECK(
      output_tensor.size(2) == static_cast<int64_t>(rank),
      "output_tensor rank dim mismatch");

  // NOTE: the kernel itself writes 0 for tokens with an invalid lora_idx,
  // so no separate output_tensor.zero_() pass is needed here (avoids an
  // extra device-wide memset kernel launch on every call).

  auto scale_f = static_cast<float>(scaling);

  // Dispatch on types
  VLLM_DISPATCH_FLOATING_TYPES(
      inputs.scalar_type(), "lora_shrink_input", [&]() {
        using input_t = scalar_t;
        // Weight type dispatch (Half or BFloat16)
        auto dispatch_weight = [&](auto* dummy_weight_ptr) {
          using weight_t = std::remove_const_t<
              std::remove_pointer_t<decltype(dummy_weight_ptr)>>;

          // Collect raw weight pointers and strides
          const weight_t* weight_ptrs[vllm::lora::kMaxSlices];
          uint64_t weight_lora_strides[vllm::lora::kMaxSlices];
          for (int64_t s = 0; s < num_slices; ++s) {
            weight_ptrs[s] = weights_3d[s].data_ptr<weight_t>();
            // stride from one LoRA adapter to the next = rank * hidden
            weight_lora_strides[s] =
                static_cast<uint64_t>(weights_3d[s].stride(0));
          }

          VLLM_DISPATCH_FLOATING_TYPES(
              output_tensor.scalar_type(), "lora_shrink_output", [&]() {
                using output_t = scalar_t;
                launch_lora_shrink<output_t, input_t, weight_t>(
                    output_tensor.data_ptr<output_t>(),
                    inputs.data_ptr<input_t>(),
                    lora_indices.data_ptr<int32_t>(),
                    batch_size,
                    hidden,
                    rank,
                    scale_f,
                    num_loras,
                    static_cast<uint32_t>(num_slices),
                    weight_ptrs,
                    weight_lora_strides);
              });
        };

        switch (weights_3d[0].scalar_type()) {
          case at::ScalarType::Half:
            dispatch_weight(static_cast<at::Half*>(nullptr));
            break;
          case at::ScalarType::BFloat16:
            dispatch_weight(static_cast<at::BFloat16*>(nullptr));
            break;
          default:
            TORCH_CHECK(
                false,
                "Unsupported weight type: ",
                weights_3d[0].scalar_type());
        }
      });
}

/**
 * Multi-slice LoRA expand op entry point.
 *
 * @param inputs          [num_slices, batch_size, rank]
 * @param lora_b_weights  list of tensors, each [num_loras, 1, slice_size, rank]
 * @param output_tensor   [batch_size, total_output_dim]  (mutated in-place)
 * @param lora_indices    [batch_size]
 * @param offset_start    starting column offset in output
 * @param add_inputs      whether to accumulate (true) or overwrite (false)
 */
void lora_expand(
    const at::Tensor& inputs,
    const std::vector<at::Tensor>& lora_b_weights,
    at::Tensor& output_tensor,
    const at::Tensor& lora_indices,
    int64_t offset_start,
    bool add_inputs) {
  const int64_t num_slices = static_cast<int64_t>(lora_b_weights.size());
  TORCH_CHECK(num_slices > 0, "lora_b_weights must be non-empty");
  TORCH_CHECK(
      num_slices <= vllm::lora::kMaxSlices,
      "Too many slices (",
      num_slices,
      "), max is ",
      vllm::lora::kMaxSlices);

  TORCH_CHECK(
      inputs.is_xpu() && inputs.is_contiguous(),
      "inputs must be contiguous XPU tensor");
  TORCH_CHECK(
      inputs.dim() == 3, "inputs must be 3D [num_slices, batch_size, rank]");
  TORCH_CHECK(
      inputs.size(0) == num_slices, "inputs.size(0) must match num_slices");
  TORCH_CHECK(
      output_tensor.is_xpu() && output_tensor.is_contiguous(),
      "output_tensor must be contiguous XPU tensor");
  TORCH_CHECK(
      output_tensor.dim() == 2,
      "output_tensor must be 2D [batch_size, total_output_dim]");
  TORCH_CHECK(
      lora_indices.is_xpu() && lora_indices.is_contiguous(),
      "lora_indices must be contiguous XPU tensor");
  TORCH_CHECK(
      lora_indices.scalar_type() == at::kInt, "lora_indices must be int32");

  const uint32_t batch_size = inputs.size(1);
  const uint32_t rank = inputs.size(2);
  const uint32_t output_dim = output_tensor.size(1);

  TORCH_CHECK(
      output_tensor.size(0) == static_cast<int64_t>(batch_size),
      "output_tensor.size(0) (",
      output_tensor.size(0),
      ") must equal batch_size (",
      batch_size,
      ")");
  TORCH_CHECK(
      lora_indices.numel() == static_cast<int64_t>(batch_size),
      "lora_indices.numel() (",
      lora_indices.numel(),
      ") must equal batch_size (",
      batch_size,
      ")");
  TORCH_CHECK(
      offset_start >= 0 && offset_start <= static_cast<int64_t>(output_dim),
      "offset_start (",
      offset_start,
      ") out of range [0, ",
      output_dim,
      "]");

  if (batch_size == 0) return;

  // Process weight tensors. Fixed-size array (num_slices <= kMaxSlices)
  // avoids a heap allocation on this hot per-call path.
  std::array<at::Tensor, vllm::lora::kMaxSlices> weights_3d;
  uint32_t slice_offsets[vllm::lora::kMaxSlices];
  uint32_t slice_sizes[vllm::lora::kMaxSlices];
  uint32_t cur_offset = static_cast<uint32_t>(offset_start);

  for (int64_t s = 0; s < num_slices; ++s) {
    at::Tensor w = lora_b_weights[s];
    TORCH_CHECK(
        w.is_xpu() && w.is_contiguous(),
        "lora_b_weights[",
        s,
        "] must be contiguous XPU tensor");
    if (w.dim() == 4) {
      TORCH_CHECK(w.size(1) == 1);
      w = w.squeeze(1);
    }
    TORCH_CHECK(w.dim() == 3);
    TORCH_CHECK(
        w.size(-1) == static_cast<int64_t>(rank),
        "weight rank dim mismatch at slice ",
        s);

    uint32_t ss = static_cast<uint32_t>(w.size(-2));
    slice_offsets[s] = cur_offset;
    slice_sizes[s] = ss;
    cur_offset += ss;

    weights_3d[s] = std::move(w);
  }

  const uint32_t num_loras = static_cast<uint32_t>(weights_3d[0].size(0));
  for (int64_t s = 1; s < num_slices; ++s) {
    TORCH_CHECK(
        weights_3d[s].size(0) == static_cast<int64_t>(num_loras),
        "lora_b_weights[",
        s,
        "] num_loras mismatch: expected ",
        num_loras,
        " but got ",
        weights_3d[s].size(0));
  }
  TORCH_CHECK(
      cur_offset <= output_dim,
      "offset_start (",
      offset_start,
      ") + sum(slice_sizes) (",
      cur_offset,
      ") exceeds output_tensor.size(1) (",
      output_dim,
      ")");

  // Dispatch on types — add_inputs is dispatched as a template parameter
  // to eliminate the runtime branch in the hot kernel loop.
  auto dispatch_all = [&](auto add_inputs_c) {
    constexpr bool ADD = decltype(add_inputs_c)::value;
    VLLM_DISPATCH_FLOATING_TYPES(
        inputs.scalar_type(), "lora_expand_input", [&]() {
          using input_t = scalar_t;
          auto dispatch_weight = [&](auto* dummy_weight_ptr) {
            using weight_t = std::remove_const_t<
                std::remove_pointer_t<decltype(dummy_weight_ptr)>>;

            const weight_t* weight_ptrs[vllm::lora::kMaxSlices];
            uint64_t weight_lora_strides[vllm::lora::kMaxSlices];
            for (int64_t s = 0; s < num_slices; ++s) {
              weight_ptrs[s] = weights_3d[s].data_ptr<weight_t>();
              weight_lora_strides[s] =
                  static_cast<uint64_t>(weights_3d[s].stride(0));
            }

            switch (output_tensor.scalar_type()) {
              case at::ScalarType::Half:
                launch_lora_expand<at::Half, input_t, weight_t, ADD>(
                    output_tensor.data_ptr<at::Half>(),
                    inputs.data_ptr<input_t>(),
                    lora_indices.data_ptr<int32_t>(),
                    batch_size,
                    output_dim,
                    rank,
                    num_loras,
                    static_cast<uint32_t>(num_slices),
                    weight_ptrs,
                    weight_lora_strides,
                    slice_offsets,
                    slice_sizes);
                break;
              case at::ScalarType::BFloat16:
                launch_lora_expand<at::BFloat16, input_t, weight_t, ADD>(
                    output_tensor.data_ptr<at::BFloat16>(),
                    inputs.data_ptr<input_t>(),
                    lora_indices.data_ptr<int32_t>(),
                    batch_size,
                    output_dim,
                    rank,
                    num_loras,
                    static_cast<uint32_t>(num_slices),
                    weight_ptrs,
                    weight_lora_strides,
                    slice_offsets,
                    slice_sizes);
                break;
              case at::ScalarType::Float:
                launch_lora_expand<float, input_t, weight_t, ADD>(
                    output_tensor.data_ptr<float>(),
                    inputs.data_ptr<input_t>(),
                    lora_indices.data_ptr<int32_t>(),
                    batch_size,
                    output_dim,
                    rank,
                    num_loras,
                    static_cast<uint32_t>(num_slices),
                    weight_ptrs,
                    weight_lora_strides,
                    slice_offsets,
                    slice_sizes);
                break;
              default:
                TORCH_CHECK(
                    false,
                    "Unsupported output type: ",
                    output_tensor.scalar_type());
            }
          };

          switch (weights_3d[0].scalar_type()) {
            case at::ScalarType::Half:
              dispatch_weight(static_cast<at::Half*>(nullptr));
              break;
            case at::ScalarType::BFloat16:
              dispatch_weight(static_cast<at::BFloat16*>(nullptr));
              break;
            default:
              TORCH_CHECK(
                  false,
                  "Unsupported weight type: ",
                  weights_3d[0].scalar_type());
          }
        });
  };

  if (add_inputs) {
    dispatch_all(std::true_type{});
  } else {
    dispatch_all(std::false_type{});
  }
}
