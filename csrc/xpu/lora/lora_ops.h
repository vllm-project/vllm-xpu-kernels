#pragma once
#include <torch/all.h>

//------------------------------------------------------------------------------
// bgmv_shrink
//------------------------------------------------------------------------------
// Batched Grouped Matrix–Vector multiplication with shrink (projection to low
// rank).
//
// Mathematical operation:
//   outputs[b, r] = scale * Σ_h (inputs[b, h] * weights[indices[b], r, h])
//
// Tensor shapes:
//   outputs : [batch_size, rank]                 // result of projection
//   inputs  : [batch_size, hidden_size]          // input features
//   weights : [num_loras, rank, hidden_size]     // LoRA B matrices
//   indices : [batch_size]                       // LoRA index per sample
//
// Parameters:
//   outputs  - Output tensor (preallocated, written in-place)
//   inputs   - Input features
//   weights  - LoRA weight matrix B
//   indices  - LoRA index mapping
//   scale    - Scaling factor applied to the result
//------------------------------------------------------------------------------
void bgmv_shrink(
    torch::Tensor& outputs,
    const torch::Tensor& inputs,
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    double scale);

//------------------------------------------------------------------------------
// bgmv_expand_slice
//------------------------------------------------------------------------------
// Batched Matrix–Vector multiplication with slice write.
// Expands inputs to higher dimension and writes results into a slice of the
// output.
//
// Mathematical operation:
//   outputs[b, slice_offset : slice_offset+slice_len] =
//       inputs[b] @ weights[indices[b]]
//       + (add_to_output ? outputs[b, slice_offset : ...] : 0)
//
// Tensor shapes:
//   outputs : [batch_size, hidden_size_out]         // output activations
//   (updated partially) inputs  : [batch_size, hidden_size]             //
//   input features weights : [num_loras, hidden_size, slice_len]   // LoRA B
//   slice indices : [batch_size]                          // LoRA index per
//   sample
//
// Parameters:
//   outputs       - Output tensor (updated in-place for a slice)
//   inputs        - Input features
//   weights       - LoRA weight slice
//   indices       - LoRA index mapping
//   slice_offset  - Starting column index of the output slice
//   add_to_output - If true, add results to existing output; otherwise
//   overwrite
//------------------------------------------------------------------------------
void bgmv_expand_slice(
    torch::Tensor& outputs,
    const torch::Tensor& inputs,
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    const int64_t slice_offset,
    const int64_t slice_size,
    bool add_to_output);

//------------------------------------------------------------------------------
// bgmv_expand
//------------------------------------------------------------------------------
// Batched Matrix–Vector multiplication (full expand).
// Expands inputs to higher dimension and writes the full result to the output.
//
// Mathematical operation:
//   outputs[b] = inputs[b] @ weights[indices[b]]
//                + (add_to_output ? outputs[b] : 0)
//
// Tensor shapes:
//   outputs : [batch_size, hidden_size_out]            // expanded activations
//   inputs  : [batch_size, hidden_size]                // input features
//   weights : [num_loras, hidden_size, hidden_size_out]// LoRA B matrices
//   indices : [batch_size]                             // LoRA index per sample
//
// Parameters:
//   outputs       - Output tensor (preallocated, updated in-place)
//   inputs        - Input features
//   weights       - LoRA weight matrix B
//   indices       - LoRA index mapping
//   add_to_output - If true, add results to existing output; otherwise
//   overwrite
//------------------------------------------------------------------------------
void bgmv_expand(
    torch::Tensor& outputs,
    const torch::Tensor& inputs,
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    bool add_to_output);

//------------------------------------------------------------------------------
// lora_shrink
//------------------------------------------------------------------------------
// Multi-slice LoRA shrink: processes ALL slices in a single kernel launch.
//
//   output[s, b, r] = scale * Σ_h (inputs[b, h] * weights_s[indices[b], r, h])
//
// Tensor shapes:
//   inputs        : [batch_size, hidden_size]
//   lora_a_weights: list of [num_loras, 1, rank, hidden_size]
//   output_tensor : [num_slices, batch_size, rank]
//   lora_indices  : [batch_size]
//------------------------------------------------------------------------------
void lora_shrink(
    const torch::Tensor& inputs,
    const std::vector<torch::Tensor>& lora_a_weights,
    torch::Tensor& output_tensor,
    const torch::Tensor& lora_indices,
    double scaling);

//------------------------------------------------------------------------------
// lora_expand
//------------------------------------------------------------------------------
// Multi-slice LoRA expand: processes ALL slices in a single kernel launch.
//
//   output[b, offset_s + c] += inputs[s, b, :] @ weights_s[indices[b], c, :]
//
// Tensor shapes:
//   inputs        : [num_slices, batch_size, rank]
//   lora_b_weights: list of [num_loras, 1, slice_size, rank]
//   output_tensor : [batch_size, total_output_dim]
//   lora_indices  : [batch_size]
//------------------------------------------------------------------------------
void lora_expand(
    const torch::Tensor& inputs,
    const std::vector<torch::Tensor>& lora_b_weights,
    torch::Tensor& output_tensor,
    const torch::Tensor& lora_indices,
    int64_t offset_start,
    bool add_inputs);