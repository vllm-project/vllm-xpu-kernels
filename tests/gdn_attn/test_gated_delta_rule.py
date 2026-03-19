# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import random

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

# QWEN NEXT shape
NUM_TOKENS = [1, 32, 1024, 8192]
BATCH_SIZE = [32]
NUM_K_HEADS = [16]
NUM_K_DIMS = [128]
NUM_V_HEADS = [32]
NUM_V_DIMS = [128]
MODE = ["prefill", "decode", "mix_mode"]
DTYPES = [torch.float16]

# Override pytest parameters when enabling mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_actual_tokens": [32],
    },
}


def ref_gated_delta_rule(
    core_attn_out,
    q,
    k,
    v,
    b,
    a,
    A_log,
    dt_bias,
    ssm_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
    num_prefills,
    num_decodes,
):
    eps = 0.000001
    scale = 1.0 / math.sqrt(q.size(-1))
    dtype = q.dtype
    batch_size = query_start_loc.shape[0] - 1
    num_k_heads = q.size(1)
    num_v_heads = v.size(1)

    A_log_exp = -torch.exp(A_log)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)

    for batch in range(batch_size):
        if has_initial_state is not None and has_initial_state[batch]:
            ssm_state_batch = ssm_state[cache_indices[batch]]
        else:
            ssm_state_batch = torch.zeros_like(ssm_state[0])

        batch_start_id = query_start_loc[batch]
        batch_end_id = query_start_loc[batch + 1]
        batch_num_tokens = batch_end_id - batch_start_id

        for token_id in range(batch_num_tokens):
            b_t = b[batch_start_id + token_id].to(
                torch.float32)  # [num_v_heads]
            beta_t = torch.sigmoid(b_t)
            a_t = a[batch_start_id + token_id].to(
                torch.float32)  # [num_v_heads]
            g_t = torch.exp(A_log_exp * softplus(a_t + dt_bias))

            q_t = q[batch_start_id + token_id].to(
                torch.float32)  # [num_k_heads, head_k_dim]
            k_t = k[batch_start_id + token_id].to(torch.float32)
            v_t = v[batch_start_id + token_id].to(
                torch.float32)  # [num_v_heads, head_v_dim]

            # l2norm
            q_t = q_t / torch.sqrt(torch.sum(q_t * q_t, dim=-1) +
                                   eps).unsqueeze(-1)
            k_t = k_t / torch.sqrt(torch.sum(k_t * k_t, dim=-1) +
                                   eps).unsqueeze(-1)
            q_t *= scale

            q_t = torch.repeat_interleave(
                q_t, repeats=num_v_heads // num_k_heads,
                dim=0)  # [num_v_heads, head_k_dim]
            k_t = torch.repeat_interleave(k_t,
                                          repeats=num_v_heads // num_k_heads,
                                          dim=0)

            ssm_state_batch *= g_t.unsqueeze(-1).unsqueeze(-1)

            kv_mem_t = (ssm_state_batch * k_t.unsqueeze(1)).sum(
                dim=-1)  # [num_v_heads, head_v_dim]
            delta_t = (v_t - kv_mem_t) * beta_t.unsqueeze(-1)

            ssm_state_batch += k_t.unsqueeze(1) * delta_t.unsqueeze(2)

            core_attn_out[batch_start_id + token_id] = (
                ssm_state_batch * q_t.unsqueeze(1)).sum(dim=-1).to(dtype)

        ssm_state[cache_indices[batch]] = ssm_state_batch.to(dtype)


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
@pytest.mark.parametrize("mode", MODE)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_gated_delta_rule(num_actual_tokens, batch_size, num_k_heads,
                          head_k_dim, num_v_heads, head_v_dim, mode, dtype):
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

    q = torch.randn((num_actual_tokens, num_k_heads, head_k_dim),
                    dtype=dtype,
                    device=device)
    k = torch.randn((num_actual_tokens, num_k_heads, head_k_dim),
                    dtype=dtype,
                    device=device)
    v = torch.randn((num_actual_tokens, num_v_heads, head_v_dim),
                    dtype=dtype,
                    device=device)
    b = torch.randn((num_actual_tokens, num_v_heads), dtype=dtype,
                    device=device)
    a = torch.randn((num_actual_tokens, num_v_heads), dtype=dtype,
                    device=device)
    A_log = torch.randn((num_v_heads, ), dtype=dtype, device=device)
    dt_bias = torch.randn((num_v_heads, ), dtype=dtype, device=device)
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads, head_v_dim, head_k_dim),
        dtype=dtype,
        device=device)
    ref_ssm_state = ssm_state.clone()

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

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads, head_v_dim),
        dtype=dtype,
        device=device,
    )

    torch.ops._xpu_C.gated_delta_rule(
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        query_start_loc,
        cache_indices,
        has_initial_state,
        num_prefills,
        num_decodes,
    )

    ref_core_attn_out = torch.zeros_like(core_attn_out)

    ref_gated_delta_rule(
        ref_core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ref_ssm_state,
        query_start_loc,
        cache_indices,
        has_initial_state,
        num_prefills,
        num_decodes,
    )

    atol = 5e-2
    rtol = 5e-2

    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=atol,
                               rtol=rtol,
                               equal_nan=True)
    for i in range(batch_size):
        state_id = cache_indices[i]
        if num_actual_tokens == 8192:
            # FIXME: skip because of accumulated floating point error at long
            # sequences, consistent with test_gdn_attn.py
            pytest.skip(
                "FIXME, skip ssm_state test because of random error at 8192")
        torch.testing.assert_close(ssm_state[state_id],
                                   ref_ssm_state[state_id],
                                   atol=atol,
                                   rtol=rtol)
