# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random

import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels._xpu_C  # noqa: F401

# QWEN NEXT shape
NUM_TOKENS = [1, 32, 1024, 8192]
BATCH_SIZE = [32]
NUM_K_HEADS = [16]
NUM_K_DIMS = [128]
NUM_V_HEADS = [32]
NUM_V_DIMS = [128]
WIDTH = [4]
HAS_BIAS = [True, False]
ACTIVATION = ["silu"]
MODE = ["prefill", "decode", "mix_mode"]
REORDER_INPUT = [True, False]
DTYPES = [torch.float16]

# Override pytest parameters when enabling mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_actual_tokens": [32],
    },
}


def ref_causal_conv1d(
    q_out,
    k_out,
    v_out,
    z_out,
    b_out,
    a_out,
    mixed_qkvz,
    mixed_ba,
    conv_weights,
    conv_bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    activation,
    num_prefills,
    num_decodes,
    reorder_input,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    dtype = mixed_qkvz.dtype
    num_actual_tokens = mixed_qkvz.size(0)
    batch_size = query_start_loc.shape[0] - 1
    kv_ratio = num_v_heads // num_k_heads

    if reorder_input:
        q_size = num_k_heads * head_k_dim
        k_size = q_size
        v_size = num_v_heads * head_v_dim
        z_size = v_size
        q_tmp, k_tmp, v_tmp, z_tmp = mixed_qkvz.split(
            [q_size, k_size, v_size, z_size], dim=-1)
        q_tmp = q_tmp.reshape(q_tmp.size(0), -1, head_k_dim)
        k_tmp = k_tmp.reshape(k_tmp.size(0), -1, head_k_dim)
        v_tmp = v_tmp.reshape(v_tmp.size(0), -1, kv_ratio * head_v_dim)
        z_tmp = z_tmp.reshape(z_tmp.size(0), -1, kv_ratio * head_v_dim)
        mixed_qkvz = torch.cat([q_tmp, k_tmp, v_tmp, z_tmp],
                               dim=-1).reshape(q_tmp.size(0),
                                               -1).contiguous()

        b_tmp, a_tmp = mixed_ba.chunk(2, dim=-1)
        b_tmp = b_tmp.reshape(b_tmp.size(0), -1, kv_ratio)
        a_tmp = a_tmp.reshape(a_tmp.size(0), -1, kv_ratio)
        mixed_ba = torch.cat([b_tmp, a_tmp],
                             dim=-1).reshape(b_tmp.size(0), -1).contiguous()

    # Parse mixed_ba into b, a: [total_seqlen, num_k_heads, 2 * kv_ratio]
    mixed_ba_3d = mixed_ba.reshape(num_actual_tokens, num_k_heads, 2 * kv_ratio)
    b, a = torch.split(mixed_ba_3d, [kv_ratio, kv_ratio], dim=-1)
    b = b.reshape(num_actual_tokens, num_v_heads)
    a = a.reshape(num_actual_tokens, num_v_heads)
    b_out.copy_(b)
    a_out.copy_(a)

    # Parse mixed_qkvz: [total_seqlen, num_k_heads, 2*head_k_dim + 2*kv_ratio*head_v_dim]
    qkvz_per_head = 2 * head_k_dim + 2 * kv_ratio * head_v_dim
    mixed_qkvz_3d = mixed_qkvz.reshape(num_actual_tokens, num_k_heads,
                                       qkvz_per_head)
    q_split, k_split, v_split, z_split = torch.split(
        mixed_qkvz_3d,
        [head_k_dim, head_k_dim, kv_ratio * head_v_dim, kv_ratio * head_v_dim],
        dim=-1)

    # z_out is directly from z_split with no conv and no activation
    z_out.copy_(z_split.reshape(num_actual_tokens, num_v_heads, head_v_dim))

    # Build flat qkv (without z) for conv
    q_flat = q_split.reshape(num_actual_tokens, num_k_heads * head_k_dim)
    k_flat = k_split.reshape(num_actual_tokens, num_k_heads * head_k_dim)
    v_flat = v_split.reshape(num_actual_tokens,
                             num_k_heads * kv_ratio * head_v_dim)
    qkv = torch.cat((q_flat, k_flat, v_flat), dim=-1)
    qkv_elems_size = qkv.shape[-1]

    for batch in range(batch_size):
        if has_initial_state is not None and has_initial_state[batch]:
            conv_state_batch = conv_states[cache_indices[batch]]
        else:
            conv_state_batch = torch.zeros_like(conv_states[0])

        batch_start_id = query_start_loc[batch]
        batch_end_id = query_start_loc[batch + 1]
        batch_num_tokens = batch_end_id - batch_start_id

        qkv_batch = qkv[batch_start_id:batch_end_id]
        qkv_conv_input = torch.cat([conv_state_batch, qkv_batch], dim=0)
        conv_states[cache_indices[batch]] = qkv_conv_input[batch_num_tokens:]

        qkv_conv_input = qkv_conv_input.transpose(0, 1).unsqueeze(0)
        conv_bias_f32 = (conv_bias.to(torch.float32)
                         if conv_bias is not None else None)
        qkv_conv_out = F.conv1d(
            qkv_conv_input.to(torch.float32),
            conv_weights.unsqueeze(1).to(torch.float32),
            conv_bias_f32,
            padding=0,
            groups=qkv_elems_size,
        )
        qkv_conv_out = (qkv_conv_out
                        if activation is None else F.silu(qkv_conv_out)).to(
                            dtype=dtype)
        qkv_conv_out = qkv_conv_out.transpose(-2, -1).reshape(
            batch_num_tokens, qkv_elems_size)

        q_conv, k_conv, v_conv = torch.split(
            qkv_conv_out,
            [
                num_k_heads * head_k_dim,
                num_k_heads * head_k_dim,
                num_k_heads * kv_ratio * head_v_dim,
            ],
            dim=-1,
        )
        q_out[batch_start_id:batch_end_id] = q_conv.reshape(
            batch_num_tokens, num_k_heads, head_k_dim)
        k_out[batch_start_id:batch_end_id] = k_conv.reshape(
            batch_num_tokens, num_k_heads, head_k_dim)
        v_out[batch_start_id:batch_end_id] = v_conv.reshape(
            batch_num_tokens, num_v_heads, head_v_dim)


def simple_random_distribute(N, batch_size):
    distribution = torch.ones([batch_size])
    for i in range(N - batch_size):
        selected_idx = random.randint(0, batch_size - 1)
        distribution[selected_idx] += 1
    return distribution


@pytest.mark.parametrize("num_actual_tokens", NUM_TOKENS)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("num_k_heads", NUM_K_HEADS)
@pytest.mark.parametrize("head_k_dim", NUM_K_DIMS)
@pytest.mark.parametrize("num_v_heads", NUM_V_HEADS)
@pytest.mark.parametrize("head_v_dim", NUM_V_DIMS)
@pytest.mark.parametrize("width", WIDTH)
@pytest.mark.parametrize("has_bias", HAS_BIAS)
@pytest.mark.parametrize("activation", ACTIVATION)
@pytest.mark.parametrize("mode", MODE)
@pytest.mark.parametrize("reorder_input", REORDER_INPUT)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_causal_conv1d(num_actual_tokens, batch_size, num_k_heads, head_k_dim,
                       num_v_heads, head_v_dim, width, has_bias, activation,
                       reorder_input, mode, dtype):
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn kernels testing on PVC.")

    device = "xpu"
    random.seed(42)
    torch.manual_seed(42)

    assert head_k_dim == head_v_dim

    if batch_size > num_actual_tokens:
        batch_size = num_actual_tokens

    if mode == "prefill":
        num_prefills = batch_size
    elif mode == "decode":
        num_prefills = 0
        if batch_size < num_actual_tokens:
            return
    else:
        num_prefills = random.randint(1, batch_size -
                                      1) if batch_size > 1 else 1

    num_decodes = batch_size - num_prefills
    cache_batch_size = 200

    mixed_qkvz_size = num_k_heads * (2 * head_k_dim +
                                     2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads * (2 * num_v_heads // num_k_heads)

    mixed_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                              dtype=dtype,
                              device=device)
    mixed_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                           dtype=dtype,
                           device=device)

    mixed_qkv_size = num_k_heads * (2 * head_k_dim +
                                    head_v_dim * num_v_heads // num_k_heads)
    conv_states = torch.randn((cache_batch_size, width - 1, mixed_qkv_size),
                              dtype=dtype,
                              device=device)
    ref_conv_states = conv_states.clone()

    conv_weights = torch.randn((mixed_qkv_size, width), dtype=dtype,
                                device=device)
    conv_bias = None
    if has_bias:
        conv_bias = torch.randn((mixed_qkv_size, ), dtype=dtype, device=device)

    prefill_batches = simple_random_distribute(num_actual_tokens - num_decodes,
                                               batch_size - num_decodes)
    token_batches = torch.cat([torch.ones([num_decodes]),
                               prefill_batches]).to(device)
    perm = torch.randperm(token_batches.size(0)).to(device)
    shuffled_tensor = token_batches[perm]
    query_start_loc = torch.cat([
        torch.zeros([1], device=device),
        torch.cumsum(shuffled_tensor, dim=0)
    ]).to(torch.int32)
    has_initial_state = perm >= num_decodes
    cache_indices = torch.tensor(random.sample(range(cache_batch_size),
                                               batch_size),
                                 device=device,
                                 dtype=torch.int32)

    q_out = torch.empty((num_actual_tokens, num_k_heads, head_k_dim),
                         dtype=dtype,
                         device=device)
    k_out = torch.empty((num_actual_tokens, num_k_heads, head_k_dim),
                         dtype=dtype,
                         device=device)
    v_out = torch.empty((num_actual_tokens, num_v_heads, head_v_dim),
                         dtype=dtype,
                         device=device)
    z_out = torch.empty((num_actual_tokens, num_v_heads, head_v_dim),
                         dtype=dtype,
                         device=device)
    b_out = torch.empty((num_actual_tokens, num_v_heads),
                         dtype=dtype,
                         device=device)
    a_out = torch.empty((num_actual_tokens, num_v_heads),
                         dtype=dtype,
                         device=device)

    torch.ops._xpu_C.causal_conv1d(
        q_out,
        k_out,
        v_out,
        z_out,
        b_out,
        a_out,
        mixed_qkvz,
        mixed_ba,
        conv_weights,
        conv_bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        num_prefills,
        num_decodes,
        reorder_input,
    )

    ref_q_out = torch.empty_like(q_out)
    ref_k_out = torch.empty_like(k_out)
    ref_v_out = torch.empty_like(v_out)
    ref_z_out = torch.empty_like(z_out)
    ref_b_out = torch.empty_like(b_out)
    ref_a_out = torch.empty_like(a_out)

    ref_causal_conv1d(
        ref_q_out,
        ref_k_out,
        ref_v_out,
        ref_z_out,
        ref_b_out,
        ref_a_out,
        mixed_qkvz,
        mixed_ba,
        conv_weights,
        conv_bias,
        ref_conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        num_prefills,
        num_decodes,
        reorder_input,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )

    atol = 5e-2
    rtol = 5e-2

    torch.testing.assert_close(q_out, ref_q_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_out, ref_k_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_out, ref_v_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(z_out, ref_z_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(b_out, ref_b_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(a_out, ref_a_out, atol=atol, rtol=rtol)

    for i in range(batch_size):
        state_id = cache_indices[i]
        torch.testing.assert_close(conv_states[state_id],
                                   ref_conv_states[state_id],
                                   atol=atol,
                                   rtol=rtol)
