# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random

import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels._xpu_C  # noqa: F401
from tests.utils import format_tc

# XE2 prefill conv1d emits q/k/v/b/a in a chunk-padded layout (see
# csrc/xpu/gdn_attn/gdn_attn_utils.h). Used to recover the valid token rows
# from the kernel return when the prefill path is taken.
CHUNK_SIZE_XE2 = 64

# causal_conv1d is the conv stage of gdn_attention. Reuse the generic qkv/ba
# split helper and token-distribution helper from the gdn attention test, but
# assert against a standalone conv-stage reference defined below (no call into
# the fused ref_gdn_attention).
from test_gdn_attn import (  # noqa: E402
    _extract_qkv_b_a_z,
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


def ref_causal_conv1d(
    z,
    conv_state,
    projected_states_qkvz,
    projected_states_ba,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    conv_weights,
    conv_bias,
    activation,
    num_prefills,
    num_decodes,
    has_initial_state,
    non_spec_query_start_loc,
    non_spec_state_indices_tensor,
    num_actual_tokens,
    tp_size,
    reorder_input,
):
    """Standalone reference for the causal_conv1d conv stage (non-spec path).

    causal_conv1d splits the fused projections, copies the gate ``z`` out,
    advances the per-sequence conv cache, and returns the per-token conv
    intermediates ``{q, k, v, b, a}`` consumed downstream by
    gated_delta_rule. ``z`` is the gate passthrough from the qkvz projection,
    the new ``conv_state`` is the trailing ``width - 1`` rows of
    ``[conv_history, qkv]`` (the raw-input sliding window), ``q``/``k``/``v``
    are the (grouped) causal convolution of that window followed by the
    activation, and ``b``/``a`` are passthrough from the ba projection.

    Writes ``z`` and updates ``conv_state`` in place, and returns
    ``(q, k, v, b, a)`` in the native per-token layout
    (``[num_actual_tokens, heads, dim]`` for q/k/v and
    ``[num_actual_tokens, heads]`` for b/a).
    """
    dtype = projected_states_qkvz.dtype
    batch_size = non_spec_query_start_loc.shape[0] - 1
    qkv, b, a, z_global = _extract_qkv_b_a_z(
        projected_states_qkvz, projected_states_ba, num_actual_tokens,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tp_size,
        reorder_input)
    z.copy_(z_global)

    qkv_elems_size = qkv.shape[-1]
    conv_bias_f = None if conv_bias is None else conv_bias.to(torch.float)

    ref_q = torch.empty((num_actual_tokens, num_k_heads // tp_size, head_k_dim),
                        dtype=dtype,
                        device=qkv.device)
    ref_k = torch.empty_like(ref_q)
    ref_v = torch.empty((num_actual_tokens, num_v_heads // tp_size, head_v_dim),
                        dtype=dtype,
                        device=qkv.device)

    split_arg_list_qkv = [
        num_k_heads // tp_size * head_k_dim,
        num_k_heads // tp_size * head_k_dim,
        num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim,
    ]

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
        # New cache = trailing (width - 1) rows of [history, qkv].
        conv_state[non_spec_state_indices_tensor[batch]] = qkv_conv_input[
            batch_num_tokens:]

        # Grouped causal conv1d over the [history, qkv] window + activation.
        conv_in = qkv_conv_input.transpose(0, 1).unsqueeze(0)
        conv_out = F.conv1d(conv_in.to(torch.float32),
                            conv_weights.unsqueeze(1).to(torch.float32),
                            conv_bias_f,
                            padding=0,
                            groups=qkv_elems_size)
        conv_out = (conv_out if activation is None else
                    F.silu(conv_out)).to(dtype=dtype)
        conv_out = conv_out.transpose(-2, -1).reshape(batch_num_tokens,
                                                      qkv_elems_size)

        q_out, k_out, v_out = torch.split(conv_out, split_arg_list_qkv, dim=-1)
        ref_q[batch_start_id:batch_end_id] = q_out.reshape(
            batch_num_tokens, num_k_heads // tp_size, head_k_dim)
        ref_k[batch_start_id:batch_end_id] = k_out.reshape(
            batch_num_tokens, num_k_heads // tp_size, head_k_dim)
        ref_v[batch_start_id:batch_end_id] = v_out.reshape(
            batch_num_tokens, num_v_heads // tp_size, head_v_dim)

    return ref_q, ref_k, ref_v, b, a


def ref_causal_conv1d_spec(
    z,
    conv_state,
    projected_states_qkvz,
    projected_states_ba,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    conv_weights,
    conv_bias,
    activation,
    num_spec_decodes,
    spec_query_start_loc,
    spec_token_indx,
    spec_state_indices_tensor,
    num_accepted_tokens,
    num_actual_tokens,
    tp_size,
    reorder_input,
):
    """Standalone reference for the causal_conv1d conv stage (spec-decode path).

    Mirrors :func:`ref_causal_conv1d` but routes the gather/scatter through
    ``spec_token_indx`` and writes the final conv cache into the last slot of
    each speculative sequence (``spec_state_indices_tensor[n, K - 1]``).
    ``z`` and ``conv_state`` are weight-independent; the returned
    ``{q, k, v, b, a}`` intermediates use the spec native layout, indexed by
    the LOCAL token position ``n * K + t`` (not the scattered global position).
    """
    dtype = projected_states_qkvz.dtype
    width = conv_weights.shape[-1]
    K = spec_state_indices_tensor.shape[1]
    qkv, b, a, z_global = _extract_qkv_b_a_z(
        projected_states_qkvz, projected_states_ba, num_actual_tokens,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tp_size,
        reorder_input)

    # Scatter the gate into the spec token positions.
    spec_indx_long = spec_token_indx.to(torch.long)
    z[spec_indx_long] = z_global[spec_indx_long]

    qkv_elems_size = qkv.shape[-1]
    conv_bias_f = None if conv_bias is None else conv_bias.to(torch.float)

    spec_token = num_spec_decodes * K
    ref_q = torch.empty((spec_token, num_k_heads // tp_size, head_k_dim),
                        dtype=dtype,
                        device=qkv.device)
    ref_k = torch.empty_like(ref_q)
    ref_v = torch.empty((spec_token, num_v_heads // tp_size, head_v_dim),
                        dtype=dtype,
                        device=qkv.device)
    ref_b = torch.empty((spec_token, num_v_heads // tp_size),
                        dtype=dtype,
                        device=qkv.device)
    ref_a = torch.empty_like(ref_b)

    split_arg_list_qkv = [
        num_k_heads // tp_size * head_k_dim,
        num_k_heads // tp_size * head_k_dim,
        num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim,
    ]

    for n in range(num_spec_decodes):
        start = int(spec_query_start_loc[n].item())
        end = int(spec_query_start_loc[n + 1].item())
        globals_ = spec_token_indx[start:end].to(torch.long)

        naccepted = int(num_accepted_tokens[n].item())
        init_col = max(naccepted - 1, 0)
        init_slot = int(spec_state_indices_tensor[n, init_col].item())
        final_conv_slot = int(spec_state_indices_tensor[n, K - 1].item())

        conv_state_batch = conv_state[init_slot].clone()
        qkv_batch = qkv[globals_]
        qkv_conv_input = torch.cat([conv_state_batch, qkv_batch], dim=0)
        # Final conv state goes ONLY to the last cache slot.
        conv_state[final_conv_slot] = qkv_conv_input[-(width - 1):]

        # Grouped causal conv1d + activation over the [history, qkv] window.
        conv_in = qkv_conv_input.transpose(0, 1).unsqueeze(0)
        conv_out = F.conv1d(conv_in.to(torch.float32),
                            conv_weights.unsqueeze(1).to(torch.float32),
                            conv_bias_f,
                            padding=0,
                            groups=qkv_elems_size)
        conv_out = (conv_out if activation is None else
                    F.silu(conv_out)).to(dtype=dtype)
        conv_out = conv_out.transpose(-2, -1).reshape(K, qkv_elems_size)

        q_out, k_out, v_out = torch.split(conv_out, split_arg_list_qkv, dim=-1)
        # LOCAL layout: seq n occupies rows [start, end).
        ref_q[start:end] = q_out.reshape(K, num_k_heads // tp_size, head_k_dim)
        ref_k[start:end] = k_out.reshape(K, num_k_heads // tp_size, head_k_dim)
        ref_v[start:end] = v_out.reshape(K, num_v_heads // tp_size, head_v_dim)
        ref_b[start:end] = b[globals_]
        ref_a[start:end] = a[globals_]

    return ref_q, ref_k, ref_v, ref_b, ref_a


def _nonspec_valid_rows(non_spec_query_start_loc):
    """Map each token to its row in the (possibly chunk-padded) conv output.

    The native (decode) path returns one row per token contiguously, while the
    XE2 prefill path lays out each sequence at ``pre_chunks * CHUNK_SIZE_XE2``
    where ``pre_chunks`` accumulates ``ceil(seq_len / CHUNK_SIZE_XE2)`` over the
    preceding sequences. Returns a 1D LongTensor of length ``num_actual_tokens``
    giving the padded-output row index for every token.
    """
    starts = non_spec_query_start_loc.tolist()
    rows = []
    pre_chunks = 0
    for i in range(len(starts) - 1):
        seq_len = starts[i + 1] - starts[i]
        base = pre_chunks * CHUNK_SIZE_XE2
        rows.extend(range(base, base + seq_len))
        pre_chunks += (seq_len + CHUNK_SIZE_XE2 - 1) // CHUNK_SIZE_XE2
    return torch.tensor(rows, dtype=torch.long)


def _assert_conv_intermediates(intermediates, ref_q, ref_k, ref_v, ref_b,
                               ref_a, num_actual_tokens,
                               non_spec_query_start_loc, atol, rtol):
    """Compare the kernel-returned {q, k, v, b, a} against the reference.

    Handles both the native per-token layout and the XE2 chunk-padded prefill
    layout (q/k/v padded along dim 0, b/a transposed to ``[heads, padded]`` in
    float32). Valid token rows are gathered before comparison.
    """
    q, k, v, b, a = intermediates

    if q.shape[0] == num_actual_tokens:
        # Native (decode) layout: one contiguous row per token.
        torch.testing.assert_close(q, ref_q, atol=atol, rtol=rtol)
        torch.testing.assert_close(k, ref_k, atol=atol, rtol=rtol)
        torch.testing.assert_close(v, ref_v, atol=atol, rtol=rtol)
        torch.testing.assert_close(b, ref_b, atol=atol, rtol=rtol)
        torch.testing.assert_close(a, ref_a, atol=atol, rtol=rtol)
        return

    # XE2 prefill layout: gather the valid (non-padded) token rows.
    valid = _nonspec_valid_rows(non_spec_query_start_loc).to(q.device)
    torch.testing.assert_close(q[valid], ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(k[valid], ref_k, atol=atol, rtol=rtol)
    torch.testing.assert_close(v[valid], ref_v, atol=atol, rtol=rtol)
    # b/a are emitted transposed ([heads, padded]) in float32, and this path
    # pre-applies sigmoid to b (a is passthrough).
    torch.testing.assert_close(b.transpose(0, 1)[valid],
                               torch.sigmoid(ref_b.to(torch.float32)),
                               atol=atol,
                               rtol=rtol)
    torch.testing.assert_close(a.transpose(0, 1)[valid],
                               ref_a.to(torch.float32),
                               atol=atol,
                               rtol=rtol)


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
@torch.inference_mode()
def test_causal_conv1d(num_actual_tokens, batch_size, num_k_heads, head_k_dim,
                       num_v_heads, head_v_dim, width, tp_size, has_bias,
                       activation, reorder_input, mode, dtype):
    # FIXME: remove skip
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn attention kernels testing on PVC.")

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

    conv_weights = torch.randn((mixed_qkv_size, width),
                               dtype=dtype,
                               device=device)
    conv_bias = None
    if has_bias:
        conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)

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

    intermediates = torch.ops._xpu_C.causal_conv1d(
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

    ref_z = torch.empty_like(core_attn_out)

    ref_q, ref_k, ref_v, ref_b, ref_a = ref_causal_conv1d(
        ref_z,
        ref_conv_state,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
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

    # Conv stage also returns the {q, k, v, b, a} intermediates.
    _assert_conv_intermediates(intermediates, ref_q, ref_k, ref_v, ref_b,
                               ref_a, num_actual_tokens,
                               non_spec_query_start_loc, atol, rtol)


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
@torch.inference_mode()
def test_causal_conv1d_mtp(num_spec_decodes, num_spec_tokens, num_k_heads,
                           head_k_dim, num_v_heads, head_v_dim, width,
                           tp_size, has_bias, activation, reorder_input,
                           dtype):
    """Pure spec-decode batch conv stage. Asserts z (scattered) and the final
    per-seq conv-state slot match a standalone conv-stage reference."""
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn attention kernels testing on PVC.")

    device = "xpu"
    random.seed(123)
    torch.manual_seed(123)

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
    conv_weights = torch.randn(mixed_qkv_size,
                               width,
                               dtype=dtype,
                               device=device)
    conv_bias = (torch.randn(mixed_qkv_size, dtype=dtype, device=device)
                 if has_bias else None)

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

    intermediates = torch.ops._xpu_C.causal_conv1d(
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

    ref_z = torch.zeros_like(core_attn_out)
    ref_q, ref_k, ref_v, ref_b, ref_a = ref_causal_conv1d_spec(
        ref_z,
        ref_conv_state,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
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

    # Conv stage also returns the {q, k, v, b, a} intermediates (spec native
    # layout: one row per token, indexed by local position n * K + t).
    _assert_conv_intermediates(intermediates, ref_q, ref_k, ref_v, ref_b,
                               ref_a, num_actual_tokens, spec_query_start_loc,
                               atol, rtol)
