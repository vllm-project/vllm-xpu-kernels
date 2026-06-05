# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401
from tests.utils import format_tc

# Reuse the fused golden reference and helpers from the gdn attention test.
# causal_conv1d is the conv stage of gdn_attention, so we drive the same
# reference and assert only on the conv-stage outputs (z and conv_state).
from test_gdn_attn import (  # noqa: E402
    ref_gdn_attention,
    ref_gdn_attention_spec,
    simple_random_distribute,
)

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

NUM_SPEC_DECODES = [1, 4]
NUM_SPEC_TOKENS = [2, 3]  # num_speculative_tokens + 1

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
def test_causal_conv1d(num_actual_tokens, batch_size, num_k_heads, head_k_dim,
                       num_v_heads, head_v_dim, width, tp_size, has_bias,
                       activation, reorder_input, mode, dtype,
                       ssm_state_is_fp32):
    # FIXME: remove skip
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn attention kernels testing on PVC.")

    device = "xpu"
    random.seed(42)
    torch.manual_seed(42)
    ssm_state_dtype = torch.float32 if ssm_state_is_fp32 else dtype

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
        dtype=ssm_state_dtype,
        device=device)
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

    torch.ops._xpu_C.causal_conv1d(
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=conv_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
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

    atol = 5e-2
    rtol = 5e-2

    # Conv stage writes z and updates conv_state.
    torch.testing.assert_close(z, ref_z, atol=atol, rtol=rtol)

    for i in range(batch_size):
        state_id = non_spec_state_indices_tensor[i]
        torch.testing.assert_close(conv_state[state_id],
                                   ref_conv_state[state_id],
                                   atol=atol,
                                   rtol=rtol)


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
def test_causal_conv1d_mtp(num_spec_decodes, num_spec_tokens, num_k_heads,
                           head_k_dim, num_v_heads, head_v_dim, width,
                           tp_size, has_bias, activation, reorder_input,
                           dtype, ssm_state_is_fp32):
    """Pure spec-decode batch conv stage. Asserts z (scattered) and the final
    per-seq conv-state slot match the fused reference."""
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

    torch.ops._xpu_C.causal_conv1d(
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=conv_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
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

    # Final conv-state slot per seq (col K-1) must match the reference.
    for n in range(num_spec_decodes):
        final_slot = int(spec_state_indices_tensor[n, K - 1].item())
        torch.testing.assert_close(conv_state[final_slot],
                                   ref_conv_state[final_slot],
                                   atol=atol,
                                   rtol=rtol)
