# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
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
TP_SIZE = [1]
HAS_BIAS = [True, False]
ACTIVATION = ["silu"]
MODE = ["prefell", "decode", "mix_mode"]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_actual_tokens": [32],
    },
}


def ref_gdn_attention(
    core_attn_out,
    z,
    projected_states_qkvz,
    projected_states_ba,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    conv_state,
    ssm_state,
    conv_weights,
    conv_bias,
    activation,
    A_log,
    dt_bias,
    num_prefills,
    num_decodes,
    has_initial_state,
    non_spec_query_start_loc,
    non_spec_state_indices_tensor,
    num_actual_tokens,
    tp_size,
):
    eps = 0.000001
    scale = 1.0 / math.sqrt(head_k_dim)
    dtype = projected_states_qkvz.dtype
    batch_size = non_spec_query_start_loc.shape[0] - 1

    split_arg_list_ba = [
        num_v_heads // num_k_heads,
        num_v_heads // num_k_heads,
    ]
    projected_states_ba = projected_states_ba.reshape(
        num_actual_tokens, num_k_heads // tp_size,
        (2 * num_v_heads // num_k_heads))
    (b, a) = torch.split(projected_states_ba, split_arg_list_ba, dim=-1)
    b = b.reshape(num_actual_tokens, num_v_heads // tp_size)
    a = a.reshape(num_actual_tokens, num_v_heads // tp_size)

    split_arg_list_qkvz = [
        head_k_dim,
        head_k_dim,
        num_v_heads // num_k_heads * head_v_dim,
        num_v_heads // num_k_heads * head_v_dim,
    ]
    projected_states_qkvz = projected_states_qkvz.reshape(
        num_actual_tokens, num_k_heads // tp_size,
        (2 * head_k_dim + 2 * num_v_heads // num_k_heads * head_v_dim))
    (q_split, k_split, v_split, z_spilt) = torch.split(projected_states_qkvz,
                                                       split_arg_list_qkvz,
                                                       dim=-1)
    q_split = q_split.reshape(num_actual_tokens,
                              num_k_heads // tp_size * head_k_dim)
    k_split = k_split.reshape(num_actual_tokens,
                              num_k_heads // tp_size * head_k_dim)
    v_split = v_split.reshape(
        num_actual_tokens,
        num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim)
    qkv = torch.cat((q_split, k_split, v_split), dim=-1).reshape(
        num_actual_tokens, num_k_heads // tp_size *
        (2 * head_k_dim + num_v_heads // num_k_heads * head_v_dim))
    qkv_elems_size = qkv.shape[-1]
    z.copy_(
        z_spilt.reshape(num_actual_tokens, num_v_heads // tp_size, head_v_dim))

    A_log_exp = -torch.exp(A_log)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)
    if conv_bias is not None:
        conv_bias = conv_bias.to(torch.float)

    for batch in range(batch_size):
        if has_initial_state[batch]:
            conv_state_batch = conv_state[non_spec_state_indices_tensor[batch]]
        else:
            conv_state_batch = torch.zeros_like(conv_state[0])

        batch_start_id = non_spec_query_start_loc[batch]
        batch_end_id = non_spec_query_start_loc[batch + 1]
        batch_num_tokens = batch_end_id - batch_start_id

        qkv_batch = qkv[batch_start_id:batch_end_id]
        qkv_conv_input = torch.cat([conv_state_batch, qkv_batch], dim=0)
        conv_state[non_spec_state_indices_tensor[batch]] = qkv_conv_input[
            batch_num_tokens:]

        qkv_conv_input = qkv_conv_input.transpose(0, 1).unsqueeze(0)

        qkv_conv_out = F.conv1d(qkv_conv_input.to(torch.float32),
                                conv_weights.unsqueeze(1).to(torch.float32),
                                conv_bias,
                                padding=0,
                                groups=qkv_elems_size)
        qkv_conv_out = (qkv_conv_out if activation is None else
                        F.silu(qkv_conv_out)).to(dtype=dtype)
        qkv_conv_out = qkv_conv_out.transpose(-2, -1).reshape(
            batch_num_tokens, qkv_elems_size)

        split_arg_list_qkv = [
            num_k_heads // tp_size * head_k_dim,
            num_k_heads // tp_size * head_k_dim,
            num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim,
        ]
        (q_out, k_out, v_out) = torch.split(qkv_conv_out,
                                            split_arg_list_qkv,
                                            dim=-1)
        q_out = q_out.reshape(batch_num_tokens, num_k_heads // tp_size,
                              head_k_dim)
        k_out = k_out.reshape(batch_num_tokens, num_k_heads // tp_size,
                              head_k_dim)
        v_out = v_out.reshape(batch_num_tokens, num_v_heads // tp_size,
                              head_v_dim)

        if has_initial_state[batch]:
            ssm_state_batch = ssm_state[non_spec_state_indices_tensor[
                batch]]  # [num_v_heads // tp_size, head_v_dim, head_k_dim]
        else:
            ssm_state_batch = torch.zeros_like(ssm_state[0])
        ssm_state_batch = ssm_state_batch.to(torch.float32)

        # O(t) = S(t) * q(t)
        # S(t) = g(t)*S(t - 1) + (v(t) - g(t)*S(t - 1)*k(t))*beta(t)*k(t)
        for token_id in range(batch_num_tokens):
            b_t = b[batch_start_id + token_id].to(
                torch.float32)  # [num_v_heads // tp_size]
            beta_t = torch.sigmoid(b_t)  # [num_v_heads // tp_size]
            a_t = a[batch_start_id + token_id].to(
                torch.float32)  # [num_v_heads // tp_size]
            g_t = torch.exp(
                A_log_exp *
                softplus(a_t + dt_bias))  # [num_v_heads // tp_size]

            q_t = q_out[token_id].to(
                torch.float32)  # [num_k_heads // tp_size, head_k_dim]
            k_t = k_out[token_id].to(
                torch.float32)  # [num_k_heads // tp_size, head_k_dim]
            v_t = v_out[token_id].to(
                torch.float32)  # [num_v_heads // tp_size, head_v_dim]

            # l2norm
            q_t = q_t / torch.sqrt(torch.sum(q_t * q_t, dim=-1) +
                                   eps).unsqueeze(-1)
            k_t = k_t / torch.sqrt(torch.sum(k_t * k_t, dim=-1) +
                                   eps).unsqueeze(-1)
            q_t *= scale

            q_t = torch.repeat_interleave(
                q_t, repeats=num_v_heads // num_k_heads,
                dim=0)  # [num_v_heads // tp_size, head_k_dim]
            k_t = torch.repeat_interleave(
                k_t, repeats=num_v_heads // num_k_heads,
                dim=0)  # [num_v_heads // tp_size, head_k_dim]

            ssm_state_batch *= g_t.unsqueeze(-1).unsqueeze(-1)

            kv_mem_t = (ssm_state_batch * k_t.unsqueeze(1)).sum(
                dim=-1)  # [num_v_heads // tp_size, head_v_dim]
            delta_t = (v_t - kv_mem_t) * beta_t.unsqueeze(
                -1)  # [num_v_heads // tp_size, head_v_dim]

            ssm_state_batch += k_t.unsqueeze(1) * delta_t.unsqueeze(2)

            core_attn_out[batch_start_id +
                          token_id] = (ssm_state_batch *
                                       q_t.unsqueeze(1)).sum(dim=-1).to(dtype)

        ssm_state[non_spec_state_indices_tensor[batch]] = ssm_state_batch.to(
            dtype)


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
@pytest.mark.parametrize("tp_size", TP_SIZE)
@pytest.mark.parametrize("has_bias", HAS_BIAS)
@pytest.mark.parametrize("activation", ACTIVATION)
@pytest.mark.parametrize("mode", MODE)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_gdn_attention(num_actual_tokens, batch_size, num_k_heads, head_k_dim,
                       num_v_heads, head_v_dim, width, tp_size, has_bias,
                       activation, mode, dtype):
    device = "xpu"
    random.seed(42)
    torch.manual_seed(42)

    assert head_k_dim == head_v_dim

    if batch_size > num_actual_tokens:
        batch_size = num_actual_tokens

    if mode == "prefell":
        num_prefills = batch_size
    elif mode == "decode":
        num_prefills = 0
        if batch_size < num_actual_tokens:
            return
    else:
        num_prefills = random.randint(1, batch_size -
                                      1) if batch_size > 1 else 1

    num_decodes = batch_size - num_prefills
    cache_batch_size = 100

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                                        dtype=dtype,
                                        device=device)
    projected_states_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                                      dtype=dtype,
                                      device=device)

    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)
    conv_state = torch.randn((cache_batch_size, width - 1, mixed_qkv_size),
                             dtype=dtype,
                             device=device)
    ref_conv_state = conv_state.clone()
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=dtype,
        device=device)
    ref_ssm_state = ssm_state.clone()

    conv_weights = torch.randn((mixed_qkv_size, width),
                               dtype=dtype,
                               device=device)
    conv_bias = None
    if has_bias:
        conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)

    A_log = torch.randn((num_v_heads // tp_size), dtype=dtype, device=device)
    dt_bias = torch.randn((num_v_heads // tp_size), dtype=dtype, device=device)

    prefill_batches = simple_random_distribute(num_actual_tokens - num_decodes,
                                               batch_size - num_decodes)
    token_batches = torch.cat([torch.ones([num_decodes]),
                               prefill_batches]).to(device)
    perm = torch.randperm(token_batches.size(0)).to(device)
    shuffled_tensor = token_batches[perm]
    non_spec_query_start_loc = torch.cat([
        torch.zeros([1], device=device),
        torch.cumsum(shuffled_tensor, dim=0)
    ]).to(torch.int32)
    has_initial_state = perm >= num_decodes
    non_spec_state_indices_tensor = torch.tensor(random.sample(
        range(cache_batch_size), batch_size),
                                                 device=device,
                                                 dtype=torch.int32)

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=dtype,
        device=device,
    )
    z = torch.empty_like(core_attn_out)

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
    )

    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.empty_like(core_attn_out)

    ref_gdn_attention(
        ref_core_attn_out,
        ref_z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=ref_conv_state,
        ssm_state=ref_ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
    )

    atol = 1e-2
    rtol = 1e-2

    torch.testing.assert_close(z, ref_z, atol=atol, rtol=rtol)
    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=atol,
                               rtol=rtol)
    for i in range(batch_size):
        state_id = non_spec_state_indices_tensor[i]
        torch.testing.assert_close(conv_state[state_id],
                                   ref_conv_state[state_id],
                                   atol=atol,
                                   rtol=rtol)
        torch.testing.assert_close(ssm_state[state_id],
                                   ref_ssm_state[state_id],
                                   atol=atol,
                                   rtol=rtol)
