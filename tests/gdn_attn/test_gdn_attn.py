# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import random

import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels._xpu_C  # noqa: F401
from tests.utils import format_tc

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
MODE = ["prefill", "decode", "mix_mode"]
REORDER_INPUT = [True, False]
DTYPES = [torch.float16, torch.bfloat16]
SSM_STATE_IS_FP32 = [False, True]
GDN_ATOL = 5e-2
GDN_RTOL = 5e-2

# Override pytest parameters when enabling mini pytest
MINI_PYTEST_PARAMS = {
"default": {
    "num_actual_tokens": [16],
    "batch_size": [16],
    "num_k_heads": [1],
    "head_k_dim": [32],
    "num_v_heads": [1],
    "head_v_dim": [32],
    "width": [2],
    "tp_size": [1],
    "has_bias": [False],
    "activation": ["silu"],
    "mode": ["prefill", "decode", "mix_mode"],
    "reorder_input": [False],
    "dtype": [torch.float16], 
    "ssm_state_is_fp32": [False],
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
    reorder_input,
):
    eps = 0.000001
    scale = 1.0 / math.sqrt(head_k_dim)
    dtype = projected_states_qkvz.dtype
    batch_size = non_spec_query_start_loc.shape[0] - 1

    if reorder_input:
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        q_size = key_dim // tp_size
        k_size = q_size
        v_size = value_dim // tp_size
        z_size = v_size
        q_tmp, k_tmp, v_tmp, z_tmp = projected_states_qkvz.split(
            [q_size, k_size, v_size, z_size], dim=-1)
        q_tmp = q_tmp.reshape(q_tmp.size(0), -1, head_k_dim)
        k_tmp = k_tmp.reshape(k_tmp.size(0), -1, head_k_dim)
        v_tmp = v_tmp.reshape(v_tmp.size(0), -1,
                              num_v_heads // num_k_heads * head_v_dim)
        z_tmp = z_tmp.reshape(z_tmp.size(0), -1,
                              num_v_heads // num_k_heads * head_v_dim)
        projected_states_qkvz = torch.cat([q_tmp, k_tmp, v_tmp, z_tmp],
                                          dim=-1).reshape(q_tmp.size(0),
                                                          -1).contiguous()

        b, a = projected_states_ba.chunk(2, dim=-1)
        b = b.reshape(b.size(0), -1, num_v_heads // num_k_heads)
        a = a.reshape(a.size(0), -1, num_v_heads // num_k_heads)
        projected_states_ba = torch.cat([b, a],
                                        dim=-1).reshape(b.size(0),
                                                        -1).contiguous()

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
            ssm_state_batch = ssm_state[
                non_spec_state_indices_tensor[batch]].to(
                    torch.float32
                )  # [num_v_heads // tp_size, head_v_dim, head_k_dim]
        else:
            ssm_state_batch = torch.zeros_like(ssm_state[0],
                                               dtype=torch.float32)

        # ------------------------------------------------------------------
        # Hoist all per-token elementwise work out of the recurrence loop.
        # Only the SSM state update itself is sequential.
        # ------------------------------------------------------------------
        rep = num_v_heads // num_k_heads
        b_batch = b[batch_start_id:batch_end_id].to(
            torch.float32)  # [T, NV]
        a_batch = a[batch_start_id:batch_end_id].to(torch.float32)
        beta_batch = torch.sigmoid(b_batch)  # [T, NV]
        g_batch = torch.exp(A_log_exp *
                            softplus(a_batch + dt_bias))  # [T, NV]

        q_all = q_out.to(torch.float32)  # [T, NK, Hk]
        k_all = k_out.to(torch.float32)
        v_all = v_out.to(torch.float32)  # [T, NV, Hv]

        # l2norm along head dim, then scale q.
        q_all = q_all * torch.rsqrt(q_all.pow(2).sum(-1, keepdim=True) + eps)
        k_all = k_all * torch.rsqrt(k_all.pow(2).sum(-1, keepdim=True) + eps)
        q_all = q_all * scale

        # GQA: replicate K/Q heads NK -> NV.
        if rep > 1:
            q_all = q_all.repeat_interleave(rep, dim=1)  # [T, NV, Hk]
            k_all = k_all.repeat_interleave(rep, dim=1)

        out_buf = torch.empty(batch_num_tokens,
                              num_v_heads // tp_size,
                              head_v_dim,
                              dtype=torch.float32,
                              device=core_attn_out.device)

        # O(t) = S(t) * q(t)
        # S(t) = g(t)*S(t - 1) + (v(t) - g(t)*S(t - 1)*k(t))*beta(t)*k(t)
        for token_id in range(batch_num_tokens):
            g_t = g_batch[token_id]  # [NV]
            beta_t = beta_batch[token_id]  # [NV]
            q_t = q_all[token_id]  # [NV, Hk]
            k_t = k_all[token_id]  # [NV, Hk]
            v_t = v_all[token_id]  # [NV, Hv]

            ssm_state_batch *= g_t.unsqueeze(-1).unsqueeze(-1)

            # kv_mem_t[v, h] = sum_k S[v, h, k] * k_t[v, k]
            kv_mem_t = torch.einsum("vhk,vk->vh", ssm_state_batch, k_t)
            delta_t = (v_t - kv_mem_t) * beta_t.unsqueeze(-1)  # [NV, Hv]

            # outer product update: S[v, h, k] += delta[v, h] * k_t[v, k]
            ssm_state_batch.add_(
                torch.einsum("vh,vk->vhk", delta_t, k_t))

            out_buf[token_id] = torch.einsum("vhk,vk->vh", ssm_state_batch,
                                             q_t)

        core_attn_out[batch_start_id:batch_end_id] = out_buf.to(dtype)
        ssm_state[non_spec_state_indices_tensor[batch]] = ssm_state_batch.to(
            ssm_state.dtype)


def simple_random_distribute(N, batch_size):
    distribution = torch.ones([batch_size])
    for i in range(N - batch_size):
        selected_idx = random.randint(0, batch_size - 1)
        distribution[selected_idx] += 1

    return distribution


@torch.inference_mode()
def _run_gdn_attention_reference_case(
    *,
    num_actual_tokens,
    batch_size,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    width,
    tp_size=1,
    has_bias=False,
    activation="silu",
    reorder_input=False,
    mode="prefill",
    dtype=torch.float16,
    ssm_state_is_fp32=False,
    seed=42,
    token_batches=None,
    active_state_indices=None,
    cache_batch_size=200,
    check_inactive_states=False,
    check_core_and_states=True,
):
    device = "xpu"
    random.seed(seed)
    torch.manual_seed(seed)
    ssm_state_dtype = torch.float32 if ssm_state_is_fp32 else dtype

    assert head_k_dim == head_v_dim

    if token_batches is not None:
        batch_size = len(token_batches)
        num_actual_tokens = sum(token_batches)
    elif batch_size > num_actual_tokens:
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
    initial_conv_state = conv_state.clone()
    ref_conv_state = conv_state.clone()
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=ssm_state_dtype,
        device=device)
    initial_ssm_state = ssm_state.clone()
    ref_ssm_state = ssm_state.clone()

    conv_weights = torch.randn((mixed_qkv_size, width),
                               dtype=dtype,
                               device=device)
    conv_bias = None
    if has_bias:
        conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)

    A_log = torch.randn((num_v_heads // tp_size),
                        dtype=torch.float32,
                        device=device)
    dt_bias = torch.randn((num_v_heads // tp_size), dtype=dtype, device=device)

    if token_batches is None:
        prefill_batches = simple_random_distribute(
            num_actual_tokens - num_decodes, batch_size - num_decodes)
        token_batches_tensor = torch.cat(
            [torch.ones([num_decodes]), prefill_batches]).to(device)
        perm = torch.randperm(token_batches_tensor.size(0)).to(device)
        shuffled_tensor = token_batches_tensor[perm]
        has_initial_state = perm >= num_decodes
    else:
        token_batches_tensor = torch.tensor(token_batches,
                                           dtype=torch.float32,
                                           device=device)
        shuffled_tensor = token_batches_tensor
        has_initial_state = (torch.arange(batch_size, device=device) >=
                             num_decodes)

    non_spec_query_start_loc = torch.cat([
        torch.zeros([1], device=device),
        torch.cumsum(shuffled_tensor, dim=0)
    ]).to(torch.int32)

    if active_state_indices is None:
        state_indices = random.sample(range(cache_batch_size), batch_size)
    else:
        assert len(active_state_indices) == batch_size
        state_indices = active_state_indices
    non_spec_state_indices_tensor = torch.tensor(state_indices,
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
        num_spec_decodes=0,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input)

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
        reorder_input=reorder_input,
    )

    torch.testing.assert_close(z, ref_z, atol=GDN_ATOL, rtol=GDN_RTOL)

    if not check_core_and_states:
        return

    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=GDN_ATOL,
                               rtol=GDN_RTOL,
                               equal_nan=True)

    for i in range(batch_size):
        state_id = non_spec_state_indices_tensor[i]
        torch.testing.assert_close(conv_state[state_id],
                                   ref_conv_state[state_id],
                                   atol=GDN_ATOL,
                                   rtol=GDN_RTOL)
        torch.testing.assert_close(ssm_state[state_id],
                                   ref_ssm_state[state_id],
                                   atol=GDN_ATOL,
                                   rtol=GDN_RTOL)

    if check_inactive_states:
        inactive_state_mask = torch.ones(cache_batch_size,
                                         dtype=torch.bool,
                                         device=device)
        inactive_state_mask[non_spec_state_indices_tensor.long()] = False
        torch.testing.assert_close(conv_state[inactive_state_mask],
                                   initial_conv_state[inactive_state_mask],
                                   atol=0,
                                   rtol=0)
        torch.testing.assert_close(ssm_state[inactive_state_mask],
                                   initial_ssm_state[inactive_state_mask],
                                   atol=0,
                                   rtol=0)


@pytest.mark.parametrize("mode", ["prefill", "mix_mode"])
@torch.inference_mode()
def test_gdn_attention_fallback_head_dim_boundary(mode):
    _run_gdn_attention_reference_case(
        num_actual_tokens=16,
        batch_size=16,
        num_k_heads=1,
        head_k_dim=32,
        num_v_heads=1,
        head_v_dim=32,
        width=2,
        mode=mode,
    )


@pytest.mark.parametrize("mode", ["prefill", "mix_mode"])
@pytest.mark.parametrize("head_dim", [64, 128])
@torch.inference_mode()
def test_gdn_attention_xe2_chunk_supported_boundary(head_dim, mode):
    _run_gdn_attention_reference_case(
        num_actual_tokens=64,
        batch_size=16,
        num_k_heads=1,
        head_k_dim=head_dim,
        num_v_heads=1,
        head_v_dim=head_dim,
        width=2,
        mode=mode,
        seed=43,
    )


@pytest.mark.parametrize("head_dim", [32, 64])
@torch.inference_mode()
def test_gdn_attention_state_update_preserves_inactive_cache_slots(head_dim):
    active_slots = [4, 12, 20, 28, 36, 44, 52, 56]
    _run_gdn_attention_reference_case(
        num_actual_tokens=64,
        batch_size=len(active_slots),
        num_k_heads=1,
        head_k_dim=head_dim,
        num_v_heads=1,
        head_v_dim=head_dim,
        width=2,
        mode="prefill",
        token_batches=[1, 2, 7, 8, 9, 10, 11, 16],
        active_state_indices=active_slots,
        cache_batch_size=64,
        check_inactive_states=True,
        seed=44,
    )


@pytest.mark.parametrize("num_actual_tokens", [7, 8, 9, 63, 64, 65])
@torch.inference_mode()
def test_gdn_attention_prefill_token_boundaries(num_actual_tokens):
    batch_size = min(num_actual_tokens, 4)
    _run_gdn_attention_reference_case(
        num_actual_tokens=num_actual_tokens,
        batch_size=batch_size,
        num_k_heads=1,
        head_k_dim=64,
        num_v_heads=1,
        head_v_dim=64,
        width=2,
        mode="prefill",
        seed=45,
    )


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
@pytest.mark.parametrize("reorder_input", REORDER_INPUT)
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("ssm_state_is_fp32", SSM_STATE_IS_FP32)
@torch.inference_mode()
def test_gdn_attention(num_actual_tokens, batch_size, num_k_heads, head_k_dim,
                       num_v_heads, head_v_dim, width, tp_size, has_bias,
                       activation, reorder_input, mode, dtype,
                       ssm_state_is_fp32):
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn attention kernels testing on PVC.")

    _run_gdn_attention_reference_case(
        num_actual_tokens=num_actual_tokens,
        batch_size=batch_size,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
        width=width,
        tp_size=tp_size,
        has_bias=has_bias,
        activation=activation,
        reorder_input=reorder_input,
        mode=mode,
        dtype=dtype,
        ssm_state_is_fp32=ssm_state_is_fp32,
        check_core_and_states=num_actual_tokens != 8192,
    )


NUM_SPEC_DECODES = [1, 4]
NUM_SPEC_TOKENS = [2, 3]  # num_speculative_tokens + 1


def _extract_qkv_b_a_z(projected_states_qkvz, projected_states_ba,
                       num_actual_tokens, num_k_heads, num_v_heads,
                       head_k_dim, head_v_dim, tp_size, reorder_input):
    """Replicate the qkv/ba split done in `ref_gdn_attention` and return
    the post-reorder qkv (for conv1d), b, a, z tensors plus split sizes."""
    if reorder_input:
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        q_size = key_dim // tp_size
        k_size = q_size
        v_size = value_dim // tp_size
        z_size = v_size
        q_tmp, k_tmp, v_tmp, z_tmp = projected_states_qkvz.split(
            [q_size, k_size, v_size, z_size], dim=-1)
        q_tmp = q_tmp.reshape(q_tmp.size(0), -1, head_k_dim)
        k_tmp = k_tmp.reshape(k_tmp.size(0), -1, head_k_dim)
        v_tmp = v_tmp.reshape(v_tmp.size(0), -1,
                              num_v_heads // num_k_heads * head_v_dim)
        z_tmp = z_tmp.reshape(z_tmp.size(0), -1,
                              num_v_heads // num_k_heads * head_v_dim)
        projected_states_qkvz = torch.cat(
            [q_tmp, k_tmp, v_tmp, z_tmp],
            dim=-1).reshape(q_tmp.size(0), -1).contiguous()

        b, a = projected_states_ba.chunk(2, dim=-1)
        b = b.reshape(b.size(0), -1, num_v_heads // num_k_heads)
        a = a.reshape(a.size(0), -1, num_v_heads // num_k_heads)
        projected_states_ba = torch.cat(
            [b, a], dim=-1).reshape(b.size(0), -1).contiguous()

    projected_states_ba = projected_states_ba.reshape(
        num_actual_tokens, num_k_heads // tp_size,
        (2 * num_v_heads // num_k_heads))
    b, a = torch.split(
        projected_states_ba,
        [num_v_heads // num_k_heads, num_v_heads // num_k_heads],
        dim=-1)
    b = b.reshape(num_actual_tokens, num_v_heads // tp_size)
    a = a.reshape(num_actual_tokens, num_v_heads // tp_size)

    split_qkvz = [
        head_k_dim,
        head_k_dim,
        num_v_heads // num_k_heads * head_v_dim,
        num_v_heads // num_k_heads * head_v_dim,
    ]
    projected_states_qkvz = projected_states_qkvz.reshape(
        num_actual_tokens, num_k_heads // tp_size,
        (2 * head_k_dim + 2 * num_v_heads // num_k_heads * head_v_dim))
    q_split, k_split, v_split, z_split = torch.split(
        projected_states_qkvz, split_qkvz, dim=-1)
    q_split = q_split.reshape(num_actual_tokens,
                              num_k_heads // tp_size * head_k_dim)
    k_split = k_split.reshape(num_actual_tokens,
                              num_k_heads // tp_size * head_k_dim)
    v_split = v_split.reshape(
        num_actual_tokens,
        num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim)
    qkv = torch.cat((q_split, k_split, v_split), dim=-1).reshape(
        num_actual_tokens,
        num_k_heads // tp_size *
        (2 * head_k_dim + num_v_heads // num_k_heads * head_v_dim))
    z_global = z_split.reshape(num_actual_tokens, num_v_heads // tp_size,
                               head_v_dim)
    return qkv, b, a, z_global


def ref_gdn_attention_spec(
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
    num_spec_decodes,
    spec_query_start_loc,
    spec_token_indx,
    spec_state_indices_tensor,
    num_accepted_tokens,
    num_actual_tokens,
    tp_size,
    reorder_input,
):
    """Spec-decode reference. Mirrors the non-spec ref but routes gather/
    scatter through spec_token_indx and writes per-step ssm-state + final
    conv-state into spec_state_indices_tensor."""
    eps = 0.000001
    scale = 1.0 / math.sqrt(head_k_dim)
    dtype = projected_states_qkvz.dtype
    width = conv_weights.shape[-1]
    rep = num_v_heads // num_k_heads
    K = spec_state_indices_tensor.shape[1]
    qkv_elems_size = num_k_heads // tp_size * (
        2 * head_k_dim + num_v_heads // num_k_heads * head_v_dim)

    qkv, b, a, z_global = _extract_qkv_b_a_z(
        projected_states_qkvz, projected_states_ba, num_actual_tokens,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tp_size,
        reorder_input)

    # Scatter z into output at the spec token positions.
    spec_indx_long = spec_token_indx.to(torch.long)
    z[spec_indx_long] = z_global[spec_indx_long]

    A_log_exp = -torch.exp(A_log)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)
    conv_bias_f = (conv_bias.to(torch.float)
                   if conv_bias is not None else None)

    split_qkv = [
        num_k_heads // tp_size * head_k_dim,
        num_k_heads // tp_size * head_k_dim,
        num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim,
    ]

    for n in range(num_spec_decodes):
        start = int(spec_query_start_loc[n].item())
        end = int(spec_query_start_loc[n + 1].item())
        assert end - start == K, (end - start, K)
        globals_ = spec_token_indx[start:end].to(torch.long)

        naccepted = int(num_accepted_tokens[n].item())
        init_col = max(naccepted - 1, 0)
        init_slot = int(spec_state_indices_tensor[n, init_col].item())
        final_conv_slot = int(spec_state_indices_tensor[n, K - 1].item())

        # ---- conv1d on the K gathered tokens (with Width-1 history) ----
        conv_state_batch = conv_state[init_slot].clone()
        qkv_batch = qkv[globals_]  # [K, qkv_elems]
        qkv_conv_input = torch.cat([conv_state_batch, qkv_batch], dim=0)
        # Final conv state goes ONLY to the last cache slot.
        conv_state[final_conv_slot] = qkv_conv_input[-(width - 1):]

        qkv_conv_in = qkv_conv_input.transpose(0, 1).unsqueeze(0).to(
            torch.float32)
        qkv_conv_out = F.conv1d(qkv_conv_in,
                                conv_weights.unsqueeze(1).to(torch.float32),
                                conv_bias_f,
                                padding=0,
                                groups=qkv_elems_size)
        qkv_conv_out = (qkv_conv_out if activation is None else
                        F.silu(qkv_conv_out)).to(dtype=dtype)
        qkv_conv_out = qkv_conv_out.transpose(-2, -1).reshape(
            K, qkv_elems_size)

        q_out, k_out, v_out = torch.split(qkv_conv_out, split_qkv, dim=-1)
        q_out = q_out.reshape(K, num_k_heads // tp_size, head_k_dim)
        k_out = k_out.reshape(K, num_k_heads // tp_size, head_k_dim)
        v_out = v_out.reshape(K, num_v_heads // tp_size, head_v_dim)

        # ---- SSM recurrence (same as non-spec, just per-step writeback) ----
        ssm_state_batch = ssm_state[init_slot].to(torch.float32).clone()

        b_batch = b[globals_].to(torch.float32)
        a_batch = a[globals_].to(torch.float32)
        beta_batch = torch.sigmoid(b_batch)
        g_batch = torch.exp(A_log_exp * softplus(a_batch + dt_bias))

        q_all = q_out.to(torch.float32)
        k_all = k_out.to(torch.float32)
        v_all = v_out.to(torch.float32)
        q_all = q_all * torch.rsqrt(q_all.pow(2).sum(-1, keepdim=True) + eps)
        k_all = k_all * torch.rsqrt(k_all.pow(2).sum(-1, keepdim=True) + eps)
        q_all = q_all * scale
        if rep > 1:
            q_all = q_all.repeat_interleave(rep, dim=1)
            k_all = k_all.repeat_interleave(rep, dim=1)

        for t in range(K):
            g_t = g_batch[t]
            beta_t = beta_batch[t]
            q_t = q_all[t]
            k_t = k_all[t]
            v_t = v_all[t]

            ssm_state_batch *= g_t.unsqueeze(-1).unsqueeze(-1)
            kv_mem_t = torch.einsum("vhk,vk->vh", ssm_state_batch, k_t)
            delta_t = (v_t - kv_mem_t) * beta_t.unsqueeze(-1)
            ssm_state_batch.add_(torch.einsum("vh,vk->vhk", delta_t, k_t))

            out_t = torch.einsum("vhk,vk->vh", ssm_state_batch, q_t).to(dtype)
            core_attn_out[globals_[t]] = out_t
            # Per-step ssm-state writeback to cache_indices[n, t].
            ssm_state[int(spec_state_indices_tensor[n, t].item())] = (
                ssm_state_batch.to(ssm_state.dtype))


@pytest.mark.parametrize("num_spec_decodes", NUM_SPEC_DECODES)
@pytest.mark.parametrize("num_spec_tokens", NUM_SPEC_TOKENS)
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("activation", ["silu"])
@pytest.mark.parametrize("reorder_input", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=format_tc)
@pytest.mark.parametrize("ssm_state_is_fp32", [False, True])
@torch.inference_mode()
def test_gdn_attention_mtp(num_spec_decodes, num_spec_tokens, num_k_heads,
                           head_k_dim, num_v_heads, head_v_dim, width,
                           tp_size, has_bias, activation, reorder_input,
                           dtype, ssm_state_is_fp32):
    """Pure spec-decode batch: num_prefills == num_decodes == 0,
    num_spec_decodes sequences each contributing num_spec_tokens tokens.
    Token positions are shuffled in the global buffer via spec_token_indx
    so the kernel's gather/scatter is exercised."""
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn attention kernels testing on PVC.")

    device = "xpu"
    random.seed(123)
    torch.manual_seed(123)
    ssm_state_dtype = torch.float32 if ssm_state_is_fp32 else dtype

    assert head_k_dim == head_v_dim
    K = num_spec_tokens
    num_actual_tokens = num_spec_decodes * K
    cache_batch_size = 200

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn(num_actual_tokens,
                                        mixed_qkvz_size,
                                        dtype=dtype,
                                        device=device)
    projected_states_ba = torch.randn(num_actual_tokens,
                                      mixed_ba_size,
                                      dtype=dtype,
                                      device=device)
    conv_state = torch.randn(cache_batch_size,
                             width - 1,
                             mixed_qkv_size,
                             dtype=dtype,
                             device=device)
    ref_conv_state = conv_state.clone()
    ssm_state = torch.randn(cache_batch_size,
                            num_v_heads // tp_size,
                            head_v_dim,
                            head_k_dim,
                            dtype=ssm_state_dtype,
                            device=device)
    ref_ssm_state = ssm_state.clone()
    conv_weights = torch.randn(mixed_qkv_size,
                               width,
                               dtype=dtype,
                               device=device)
    conv_bias = (torch.randn(mixed_qkv_size, dtype=dtype, device=device)
                 if has_bias else None)
    A_log = torch.randn(num_v_heads // tp_size,
                        dtype=torch.float32,
                        device=device)
    dt_bias = torch.randn(num_v_heads // tp_size, dtype=dtype, device=device)

    # Each spec seq owns K consecutive cache slots (cols 0..K-1).
    state_slots = random.sample(range(cache_batch_size), num_spec_decodes * K)
    spec_state_indices_tensor = torch.tensor(state_slots,
                                             dtype=torch.int32,
                                             device=device).reshape(
                                                 num_spec_decodes, K)
    # Mix of acceptance counts including the 0 edge case.
    num_accepted_tokens = torch.tensor(
        [random.randint(0, K) for _ in range(num_spec_decodes)],
        dtype=torch.int32,
        device=device)

    # Shuffle global token positions across the K-tokens-per-seq layout.
    perm = torch.randperm(num_actual_tokens, device=device).to(torch.int32)
    spec_token_indx = perm.contiguous()
    spec_query_start_loc = (torch.arange(
        num_spec_decodes + 1, dtype=torch.int32, device=device) * K)

    core_attn_out = torch.zeros(num_actual_tokens,
                                num_v_heads // tp_size,
                                head_v_dim,
                                dtype=dtype,
                                device=device)
    z = torch.zeros_like(core_attn_out)

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
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=num_spec_decodes,
        has_initial_state=None,
        non_spec_query_start_loc=None,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=None,
        spec_query_start_loc=spec_query_start_loc,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input)

    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.zeros_like(core_attn_out)
    ref_gdn_attention_spec(
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
        num_spec_decodes=num_spec_decodes,
        spec_query_start_loc=spec_query_start_loc,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input,
    )

    atol = 5e-2
    rtol = 5e-2

    torch.testing.assert_close(z, ref_z, atol=atol, rtol=rtol)
    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=atol,
                               rtol=rtol,
                               equal_nan=True)

    # Final conv-state slot per seq (col K-1) must match the reference.
    for n in range(num_spec_decodes):
        final_slot = int(spec_state_indices_tensor[n, K - 1].item())
        torch.testing.assert_close(conv_state[final_slot],
                                   ref_conv_state[final_slot],
                                   atol=atol,
                                   rtol=rtol)
        # All K ssm-state slots are written per-step.
        for t in range(K):
            slot = int(spec_state_indices_tensor[n, t].item())
            torch.testing.assert_close(ssm_state[slot],
                                       ref_ssm_state[slot],
                                       atol=atol,
                                       rtol=rtol)


@torch.inference_mode()
def test_gdn_attention_mtp_accepts_padded_spec_metadata():
    """Graph capture can pad spec metadata rows; only active rows are valid."""
    device = "xpu"
    random.seed(321)
    torch.manual_seed(321)

    dtype = torch.float16
    num_spec_decodes = 1
    num_spec_tokens = 2
    num_actual_tokens = num_spec_decodes * num_spec_tokens
    num_k_heads = 16
    head_k_dim = 128
    num_v_heads = 32
    head_v_dim = 128
    width = 4
    tp_size = 1
    cache_batch_size = 32

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn(num_actual_tokens,
                                        mixed_qkvz_size,
                                        dtype=dtype,
                                        device=device)
    projected_states_ba = torch.randn(num_actual_tokens,
                                      mixed_ba_size,
                                      dtype=dtype,
                                      device=device)
    conv_state = torch.randn(cache_batch_size,
                             width - 1,
                             mixed_qkv_size,
                             dtype=dtype,
                             device=device)
    ssm_state = torch.randn(cache_batch_size,
                            num_v_heads // tp_size,
                            head_v_dim,
                            head_k_dim,
                            dtype=dtype,
                            device=device)
    conv_weights = torch.randn(mixed_qkv_size,
                               width,
                               dtype=dtype,
                               device=device)
    conv_bias = torch.randn(mixed_qkv_size, dtype=dtype, device=device)
    A_log = torch.randn(num_v_heads // tp_size,
                        dtype=torch.float32,
                        device=device)
    dt_bias = torch.randn(num_v_heads // tp_size, dtype=dtype, device=device)

    state_slots = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device=device)
    spec_query_start_loc = torch.tensor([0, num_spec_tokens],
                                        dtype=torch.int32,
                                        device=device)
    spec_token_indx = torch.tensor([1, 0], dtype=torch.int32, device=device)

    def run_with_metadata(query_start_loc, state_indices, accepted_tokens):
        core_attn_out = torch.zeros(num_actual_tokens,
                                    num_v_heads // tp_size,
                                    head_v_dim,
                                    dtype=dtype,
                                    device=device)
        z = torch.zeros_like(core_attn_out)
        conv_state_run = conv_state.clone()
        ssm_state_run = ssm_state.clone()
        torch.ops._xpu_C.gdn_attention(
            core_attn_out,
            z,
            projected_states_qkvz.clone(),
            projected_states_ba.clone(),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_state=conv_state_run,
            ssm_state=ssm_state_run,
            conv_weights=conv_weights,
            conv_bias=conv_bias,
            activation="silu",
            A_log=A_log,
            dt_bias=dt_bias,
            num_prefills=0,
            num_decodes=0,
            num_spec_decodes=num_spec_decodes,
            has_initial_state=None,
            non_spec_query_start_loc=None,
            non_spec_token_indx=None,
            non_spec_state_indices_tensor=None,
            spec_query_start_loc=query_start_loc,
            spec_token_indx=spec_token_indx,
            spec_state_indices_tensor=state_indices,
            num_accepted_tokens=accepted_tokens,
            num_actual_tokens=num_actual_tokens,
            tp_size=tp_size,
            reorder_input=False)
        return core_attn_out, z, conv_state_run, ssm_state_run

    exact = run_with_metadata(spec_query_start_loc, state_slots,
                              num_accepted_tokens)

    padded_query_start_loc = torch.tensor([0, num_spec_tokens, 99, 99],
                                          dtype=torch.int32,
                                          device=device)
    padded_state_slots = torch.tensor([[3, 5], [7, 11], [13, 17]],
                                      dtype=torch.int32,
                                      device=device)
    padded_num_accepted = torch.tensor([1, 0, 0],
                                       dtype=torch.int32,
                                       device=device)
    padded = run_with_metadata(padded_query_start_loc, padded_state_slots,
                               padded_num_accepted)

    for exact_tensor, padded_tensor in zip(exact, padded):
        torch.testing.assert_close(padded_tensor, exact_tensor,
                                   atol=5e-2, rtol=5e-2)
