# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.utils.math_utils import next_power_of_2

_MIN_LAUNCH_GRID_2D = 128
_NUM_PAR_SOFTMAX_SEGMENTS = 16


def make_triton_softmax_buffers(num_query_heads, num_kv_heads, head_size):
    seq_threshold_3D = _MIN_LAUNCH_GRID_2D // num_kv_heads
    headdim_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, _NUM_PAR_SOFTMAX_SEGMENTS,
         headdim_padded),
        dtype=torch.float32)
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, _NUM_PAR_SOFTMAX_SEGMENTS),
        dtype=torch.float32)
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, _NUM_PAR_SOFTMAX_SEGMENTS),
        dtype=torch.float32)
    return (seq_threshold_3D, _NUM_PAR_SOFTMAX_SEGMENTS,
            softmax_segm_output, softmax_segm_max, softmax_segm_expsum)
