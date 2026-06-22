# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import random

import pytest
import torch
# gated_delta_rule is the recurrent (SSM) stage of gdn_attention. It consumes
# the post-conv {q, k, v, b, a} intermediates produced by causal_conv1d. We run
# causal_conv1d once to obtain correctly-shaped/laid-out intermediates (their
# layout is path-specific under XE2), then feed those very tensors to BOTH the
# kernel and the standalone references below. The references contain no conv1d
# logic at all -- they only run the delta-rule recurrence on the intermediates.
# Reuse only the token-distribution helper.
from test_gdn_attn import simple_random_distribute  # noqa: E402

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


def unpad_intermediates(intermediates, query_start_loc, num_actual_tokens):
    """Reshape the (possibly XE2 chunk-padded) causal_conv1d intermediates into
    a clean native ``[token, ...]`` layout for the references.

    Returns ``(q, k, v, beta, a)`` where ``q/k/v`` are ``[T, H, D]``, ``beta``
    already has the sigmoid folded in, and ``a`` is the raw gate -- both
    ``[T, num_v_heads]`` float32 and indexed by native token position. This
    moves all path-specific intermediate-layout knowledge out of the references:

      * native (decode/spec): ``q/k/v`` are already ``[T, H, D]`` and ``b`` is
        the raw gate, so ``beta = sigmoid(b)``.
      * XE2 prefill/mix: ``q/k/v`` are zero-padded to ``[T + pad, H, D]`` with
        tokens scattered at ``pre_chunks * chunk + token_in_seq``; ``b/a`` are
        transposed float32 ``[num_v_heads, T + pad]`` with the sigmoid already
        folded into ``b`` (used as beta directly). Real tokens are gathered back
        into native order here.
    """
    q, k, v, b, a = intermediates
    chunk = 64

    if q.shape[0] == num_actual_tokens:
        # Native layout; b is the raw gate.
        return (q.clone(), k.clone(), v.clone(),
                torch.sigmoid(b.to(torch.float32)), a.to(torch.float32))

    # XE2 chunk-padded layout: gather real tokens back into native order.
    num_v = b.shape[0]
    q_n = torch.zeros((num_actual_tokens, *q.shape[1:]),
                      dtype=q.dtype,
                      device=q.device)
    k_n = torch.zeros((num_actual_tokens, *k.shape[1:]),
                      dtype=k.dtype,
                      device=k.device)
    v_n = torch.zeros((num_actual_tokens, *v.shape[1:]),
                      dtype=v.dtype,
                      device=v.device)
    beta_n = torch.zeros((num_actual_tokens, num_v),
                         dtype=torch.float32,
                         device=b.device)
    a_n = torch.zeros((num_actual_tokens, num_v),
                      dtype=torch.float32,
                      device=a.device)

    pre_chunks = 0
    for i in range(query_start_loc.shape[0] - 1):
        s = int(query_start_loc[i].item())
        e = int(query_start_loc[i + 1].item())
        seq_len = e - s
        base = pre_chunks * chunk
        q_n[s:e] = q[base:base + seq_len]
        k_n[s:e] = k[base:base + seq_len]
        v_n[s:e] = v[base:base + seq_len]
        # b is already sigmoid'd (beta) in the XE2 conv kernel; a is raw.
        beta_slice = b[:, base:base + seq_len].transpose(0, 1)
        beta_n[s:e] = beta_slice.to(torch.float32)
        a_n[s:e] = a[:, base:base + seq_len].transpose(0, 1).to(torch.float32)
        pre_chunks += (seq_len + chunk - 1) // chunk

    return q_n, k_n, v_n, beta_n, a_n


def ref_gated_delta_rule(
    core_attn_out,
    ssm_state,
    q,
    k,
    v,
    beta,
    a,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    A_log,
    dt_bias,
    has_initial_state,
    non_spec_query_start_loc,
    non_spec_state_indices_tensor,
    tp_size,
):
    """Standalone reference for the gated_delta_rule (recurrent) stage,
    non-spec path.

    Consumes the post-conv ``{q, k, v}`` and the resolved gates ``beta``
    (already sigmoid'd) and ``a`` (raw), all in native ``[token, ...]`` layout
    (see :func:`unpad_intermediates`). There is NO convolution and no padded
    handling here -- it only runs the chunk-free GatedDeltaNet recurrence to
    write ``core_attn_out`` and advance ``ssm_state`` in place. The gate ``z``
    is applied outside the kernel and is not part of the asserted output.
    """
    eps = 0.000001
    scale = 1.0 / math.sqrt(head_k_dim)
    dtype = core_attn_out.dtype
    batch_size = non_spec_query_start_loc.shape[0] - 1
    rep = num_v_heads // num_k_heads
    num_v = num_v_heads // tp_size

    A_log_exp = -torch.exp(A_log)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)

    for batch in range(batch_size):
        batch_start_id = int(non_spec_query_start_loc[batch].item())
        batch_end_id = int(non_spec_query_start_loc[batch + 1].item())
        seq_len = batch_end_id - batch_start_id

        q_batch = q[batch_start_id:batch_end_id]
        k_batch = k[batch_start_id:batch_end_id]
        v_batch = v[batch_start_id:batch_end_id]
        beta_batch = beta[batch_start_id:batch_end_id]
        a_batch = a[batch_start_id:batch_end_id]

        state_idx = int(non_spec_state_indices_tensor[batch].item())
        if has_initial_state[batch]:
            ssm_state_batch = ssm_state[state_idx].to(torch.float32)
        else:
            ssm_state_batch = torch.zeros_like(ssm_state[0],
                                               dtype=torch.float32)

        g_batch = torch.exp(A_log_exp * softplus(a_batch + dt_bias))

        q_all = q_batch.to(torch.float32)
        k_all = k_batch.to(torch.float32)
        v_all = v_batch.to(torch.float32)
        q_all = q_all * torch.rsqrt(q_all.pow(2).sum(-1, keepdim=True) + eps)
        k_all = k_all * torch.rsqrt(k_all.pow(2).sum(-1, keepdim=True) + eps)
        q_all = q_all * scale
        if rep > 1:
            q_all = q_all.repeat_interleave(rep, dim=1)
            k_all = k_all.repeat_interleave(rep, dim=1)

        out_buf = torch.empty(seq_len,
                              num_v,
                              head_v_dim,
                              dtype=torch.float32,
                              device=core_attn_out.device)

        # O(t) = S(t) * q(t)
        # S(t) = g(t)*S(t-1) + (v(t) - g(t)*S(t-1)*k(t))*beta(t)*k(t)
        for token_id in range(seq_len):
            g_t = g_batch[token_id]
            beta_t = beta_batch[token_id]
            q_t = q_all[token_id]
            k_t = k_all[token_id]
            v_t = v_all[token_id]

            ssm_state_batch *= g_t.unsqueeze(-1).unsqueeze(-1)
            kv_mem_t = torch.einsum("vhk,vk->vh", ssm_state_batch, k_t)
            delta_t = (v_t - kv_mem_t) * beta_t.unsqueeze(-1)
            ssm_state_batch.add_(torch.einsum("vh,vk->vhk", delta_t, k_t))
            out_buf[token_id] = torch.einsum("vhk,vk->vh", ssm_state_batch, q_t)

        core_attn_out[batch_start_id:batch_end_id] = out_buf.to(dtype)
        ssm_state[state_idx] = ssm_state_batch.to(ssm_state.dtype)


def ref_gated_delta_rule_spec(
    core_attn_out,
    ssm_state,
    q,
    k,
    v,
    beta,
    a,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    A_log,
    dt_bias,
    num_spec_decodes,
    spec_query_start_loc,
    spec_token_indx,
    spec_state_indices_tensor,
    num_accepted_tokens,
    tp_size,
):
    """Standalone reference for the gated_delta_rule stage, spec-decode path.

    Consumes the post-conv ``{q, k, v}`` and the resolved gates ``beta``
    (already sigmoid'd) and ``a`` (raw), in native local token order
    (``t = n * K + t_local``); see :func:`unpad_intermediates`. No convolution
    and no padded-layout handling here. Routes the output scatter through
    ``spec_token_indx`` and writes the per-step ssm-state into
    ``spec_state_indices_tensor[n, t]`` for every step.
    """
    eps = 0.000001
    scale = 1.0 / math.sqrt(head_k_dim)
    dtype = core_attn_out.dtype
    rep = num_v_heads // num_k_heads
    K = spec_state_indices_tensor.shape[1]

    A_log_exp = -torch.exp(A_log)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)

    for n in range(num_spec_decodes):
        start = int(spec_query_start_loc[n].item())
        end = int(spec_query_start_loc[n + 1].item())
        assert end - start == K, (end - start, K)
        globals_ = spec_token_indx[start:end].to(torch.long)

        naccepted = int(num_accepted_tokens[n].item())
        init_col = max(naccepted - 1, 0)
        init_slot = int(spec_state_indices_tensor[n, init_col].item())

        # Intermediates are stored in local token order; this sequence owns the
        # contiguous local slice [start, end).
        q_out = q[start:end]
        k_out = k[start:end]
        v_out = v[start:end]

        # ---- SSM recurrence with per-step ssm-state writeback ----
        ssm_state_batch = ssm_state[init_slot].to(torch.float32).clone()

        beta_batch = beta[start:end]
        a_batch = a[start:end]
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
            ssm_state[int(spec_state_indices_tensor[n, t].item())] = (
                ssm_state_batch.to(ssm_state.dtype))


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
def test_gated_delta_rule(num_actual_tokens, batch_size, num_k_heads,
                          head_k_dim, num_v_heads, head_v_dim, width, tp_size,
                          has_bias, activation, reorder_input, mode, dtype,
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

    # Decode mode should always have initial_state
    if mode == "decode":
        has_initial_state = torch.ones(batch_size,
                                       dtype=torch.bool,
                                       device=device)
    else:
        has_initial_state = perm < num_decodes

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

    # Conv stage produces the intermediates consumed by gated_delta_rule.
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

    # Reshape the (possibly XE2 chunk-padded) intermediates into native layout
    # for the reference. This also snapshots them before the delta kernel, which
    # normalizes q/k in place.
    ref_q, ref_k, ref_v, ref_beta, ref_a = unpad_intermediates(
        intermediates, non_spec_query_start_loc, num_actual_tokens)

    torch.ops._xpu_C.gated_delta_rule(
        core_attn_out,
        *intermediates,
        num_v_heads,
        head_v_dim,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
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
        tp_size=tp_size)

    ref_core_attn_out = torch.zeros_like(core_attn_out)

    ref_gated_delta_rule(
        ref_core_attn_out,
        ref_ssm_state,
        ref_q,
        ref_k,
        ref_v,
        ref_beta,
        ref_a,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        A_log=A_log,
        dt_bias=dt_bias,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        tp_size=tp_size,
    )

    atol = 5e-2
    rtol = 5e-2

    if num_actual_tokens == 8192:
        pytest.skip("FIXME, skip core_attn_out test because of random error")

    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=atol,
                               rtol=rtol,
                               equal_nan=True)
    for i in range(batch_size):
        state_id = non_spec_state_indices_tensor[i]
        # FIXME: the ssm_state check is skipped for num_actual_tokens == 8192
        # due to random error; will be fixed in future. (8192 is already
        # skipped earlier, so the assertion runs for the reached cases.)
        if num_actual_tokens != 8192:
            torch.testing.assert_close(ssm_state[state_id],
                                       ref_ssm_state[state_id],
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
def test_gated_delta_rule_mtp(num_spec_decodes, num_spec_tokens, num_k_heads,
                              head_k_dim, num_v_heads, head_v_dim, width,
                              tp_size, has_bias, activation, reorder_input,
                              dtype, ssm_state_is_fp32):
    """Pure spec-decode batch delta stage. Feeds causal_conv1d intermediates
    to gated_delta_rule and asserts core_attn_out and the per-step ssm-state
    writebacks match the fused reference."""
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

    # Reshape intermediates into native layout for the reference (also snapshots
    # them before the delta kernel, which may normalize q/k in place). The spec
    # path never pads, but the same padding-aware helper is used for symmetry.
    ref_q, ref_k, ref_v, ref_beta, ref_a = unpad_intermediates(
        intermediates, spec_query_start_loc, num_actual_tokens)

    torch.ops._xpu_C.gated_delta_rule(
        core_attn_out,
        *intermediates,
        num_v_heads,
        head_v_dim,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
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
        tp_size=tp_size)

    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_gated_delta_rule_spec(
        ref_core_attn_out,
        ref_ssm_state,
        ref_q,
        ref_k,
        ref_v,
        ref_beta,
        ref_a,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        A_log=A_log,
        dt_bias=dt_bias,
        num_spec_decodes=num_spec_decodes,
        spec_query_start_loc=spec_query_start_loc,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        tp_size=tp_size,
    )

    atol = 5e-2
    rtol = 5e-2

    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=atol,
                               rtol=rtol,
                               equal_nan=True)

    # All K ssm-state slots are written per-step.
    for n in range(num_spec_decodes):
        for t in range(K):
            slot = int(spec_state_indices_tensor[n, t].item())
            torch.testing.assert_close(ssm_state[slot],
                                       ref_ssm_state[slot],
                                       atol=atol,
                                       rtol=rtol)
