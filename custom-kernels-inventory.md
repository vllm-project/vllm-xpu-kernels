# Custom kernels inventory

## LoRA entry points

These operators are registered in the `_xpu_C` namespace for XPU tensors.
All output tensors are supplied by the caller and mutated in place.

| Operator | Operation |
| --- | --- |
| `bgmv_shrink` | Single-slice LoRA A projection |
| `bgmv_expand` | Single-slice LoRA B projection |
| `bgmv_expand_slice` | Single-slice B projection into an output column range |
| `lora_shrink` | Multi-slice A projection with an active-rank output |
| `lora_expand` | Multi-slice B projection from an active-rank input |
| `lora_linear` | Combined dispatch of multi-slice shrink and expand |

### Active-rank linear contract

`lora_linear(inputs, lora_a_weights, lora_b_weights, output_tensor,
lora_indices, scaling, active_rank, add_inputs)` accepts contiguous weights
stored at a rank at least as large as `active_rank`. Each slice retains its
physical row and adapter strides. The temporary has shape
`[num_slices, batch_size, active_rank]` and the input activation dtype.
The token-to-adapter mapping is contiguous int32.

The operation applies the A projection and scaling, then the B projection.
B slices occupy consecutive columns starting at zero. `add_inputs` selects
addition or replacement for rows with a valid adapter. Rows with negative or
out-of-range adapter indices, and columns outside the slices, remain unchanged.
The caller must initialize these rows if it needs zero output there.

This entry point combines host dispatch; it does not fuse the two GPU kernels.
Callers can detect its registration before enabling active-rank dispatch and
retain their existing path with older kernel packages.
