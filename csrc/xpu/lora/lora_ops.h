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
void bgmv_shrink(torch::Tensor& outputs, const torch::Tensor& inputs,
                 const torch::Tensor& weights, const torch::Tensor& indices,
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
void bgmv_expand_slice(torch::Tensor& outputs, const torch::Tensor& inputs,
                       const torch::Tensor& weights,
                       const torch::Tensor& indices, int64_t slice_offset,
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
void bgmv_expand(torch::Tensor& outputs, const torch::Tensor& inputs,
                 const torch::Tensor& weights, const torch::Tensor& indices,
                 bool add_to_output);