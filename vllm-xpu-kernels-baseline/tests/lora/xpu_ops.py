# SPDX-License-Identifier: Apache-2.0
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
) -> None:
    torch.ops._xpu_C.bgmv_shrink(
        output_tensor,
        inputs,
        lora_a_weights,
        lora_indices_tensor,
        scaling,
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    """
    Args:
        inputs (torch.Tensor): Shape: `[batch_size, hidden_size]`.
        lora_b_weights (torch.Tensor): Shape: `[lora_num, rank, hidden_size]`.
        output_tensor (torch.Tensor): Shape: `[batch_size, rank]`.
        lora_indices_tensor (torch.Tensor): Shape: `[batch_size]`.
         The LoRA index corresponding to each batch. An index of -1 means
            no lora should be applied.
        add_inputs (bool, optional):  Defaults to False. adds the final lora
            results to the output.

    Semantics:
      for i in range(inputs.size(0)):
        output_tensor[i] =
            inputs[i] @ lora_b_weights[lora_indices_tensor[i]]
            + (inputs[i] if add_inputs else 0)
    """
    torch.ops._xpu_C.bgmv_expand(
        output_tensor,
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        add_inputs,
    )


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    """
    Args:
        inputs (torch.Tensor): Shape: `[batch_size, hidden_size]`.
        lora_b_weights (torch.Tensor): Shape: `[lora_num, rank, hidden_size]`.
        output_tensor (torch.Tensor): Shape: `[batch_size, rank]`.
        lora_indices_tensor (torch.Tensor): Shape: `[batch_size]`.
            The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        slice_offset (int): output_tensor's offset
        slice_size (int): current output_tensor's size
        add_inputs (bool, optional):  Defaults to False. adds the final lora
            results to the output.

    Semantics:
      for i in range(inputs.size(0)):
        output_tensor[i][slice_offset:slice_offset+slice_size] =
            inputs[i] @ lora_b_weights[lora_indices_tensor[i]]
            + (inputs[i] if add_inputs else 0)
    """
    torch.ops._xpu_C.bgmv_expand_slice(
        output_tensor,
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        slice_offset,
        slice_size,
        add_inputs,
    )
