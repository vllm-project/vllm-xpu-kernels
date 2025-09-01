# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def seed_everything(seed: Optional[int] = None) -> None:
    """
    Set the seed of each random module.
    `torch.manual_seed` will set seed on all devices.

    Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


@dataclass
class PunicaTensors:
    inputs_tensor: torch.Tensor
    lora_weights: Union[torch.Tensor, list[torch.Tensor]]
    our_out_tensor: torch.Tensor
    ref_out_tensor: torch.Tensor
    b_seq_start_loc: torch.Tensor
    prompt_lora_mapping: torch.Tensor
    seq_len_tensor: torch.Tensor
    token_lora_mapping: torch.Tensor

    def meta(self) -> tuple[int, int]:
        """
        Infer max_seq_length and token_nums from the tensors
        and return them.
        """
        max_seq_length = self.seq_len_tensor.max()
        token_nums = self.seq_len_tensor.sum().item()
        if isinstance(max_seq_length, tuple):
            max_seq_length = max_seq_length[0].item()
        else:
            max_seq_length = max_seq_length.item()
        return max_seq_length, token_nums


def generate_data(
    batches,
    hidden_size,
    lora_nums,
    max_rank,
    seq_length,
    dtype,
    op_type,
    device,
) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1,
                                   (batches, )).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()
    if op_type == "shrink":
        inputs_tensor = torch.rand((total_tokens, hidden_size),
                                   dtype=dtype).to(device)
        lora_weights = torch.rand(
            (lora_nums, max_rank, hidden_size),  # col-major
            dtype=dtype,
        ).to(device)
        # shrink op need atomic_add, so output is initinized by 0
        ref_out_tensor = torch.zeros((total_tokens, max_rank),
                                     dtype=dtype,
                                     device=inputs_tensor.device)
        # NOTE  shrink kernel using torch.float32 as output type
        our_out_tensor = torch.zeros((total_tokens, max_rank),
                                     dtype=torch.float32).to(device)
    else:
        inputs_tensor = torch.rand(
            (total_tokens, max_rank),
            dtype=dtype,
        ).to(device)
        lora_weights = torch.rand(
            (lora_nums, hidden_size, max_rank),  # col-major
            dtype=dtype,
        ).to(device)
        # expand op needs to complete y+=a@lora_b, so output is
        # initinized randomly
        ref_out_tensor = torch.rand(
            (total_tokens, hidden_size),
            dtype=dtype,
        ).to(device)
        # Ensure the same input.
        our_out_tensor = ref_out_tensor.clone()
    lora_indices_tensor = torch.randint(0,
                                        lora_nums - 1 if lora_nums > 1 else 1,
                                        (batches, )).to(device)
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset:current_offset +
                seq_len_tensor[b_id]].copy_(lora_index)
        current_offset += seq_len_tensor[b_id].item()

    return PunicaTensors(
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )


def generate_data_for_expand_nslices(
    batches,
    hidden_size,
    lora_nums,
    max_rank,
    seq_length,
    dtype,
    nslices,
    device,
) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1,
                                   (batches, )).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()
    inputs_tensor = torch.rand(
        (total_tokens, max_rank),
        dtype=dtype,
    ).to(device)
    lora_weights_lst = []
    for _ in range(nslices):
        lora_weights_lst.append(
            torch.rand(
                (lora_nums, hidden_size, max_rank),  # col-major
                dtype=dtype,
            ).to(device))
    # expand op needs to complete y+=a@lora_b, so output is
    # initinized randomly
    ref_out_tensor = torch.rand((total_tokens, hidden_size * nslices),
                                dtype=dtype).to(device)
    # Ensure the same input.
    our_out_tensor = ref_out_tensor.clone()
    lora_indices_tensor = torch.randint(0,
                                        lora_nums - 1 if lora_nums > 1 else 1,
                                        (batches, ))
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset:current_offset +
                seq_len_tensor[b_id]] = (lora_index.item())
        current_offset += seq_len_tensor[b_id].item()

    lora_indices_tensor = lora_indices_tensor.to(device)
    return PunicaTensors(
        inputs_tensor,
        lora_weights_lst,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )
