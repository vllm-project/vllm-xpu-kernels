# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# gated_delta_rule_spec_recoverssm is the RecoverSSM variant of the spec-decode
# delta stage (vLLM --use-replayssm). It reads one checkpoint state block per
# request, never writes it, and stores a float32 replay record
# [corr (V), normalised k (K), decay (1)] per token. The reference below runs
# the same recurrence in float32 on the {q, k, v, b, a} intermediates.

import math
import os
import random

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401
from tests.utils import format_tc

NULL_BLOCK_ID = 0

NUM_SPEC_DECODES = [1, 4, 13]
NUM_SPEC_TOKENS = [2, 3]  # num_speculative_tokens + 1
NUM_V_HEADS = [32, 48]

MINI_PYTEST_PARAMS = {
    "default": {
        "num_spec_decodes": [4],
        "num_spec_tokens": [2],
        "num_v_heads": [32],
        "dtype": [torch.bfloat16],
        "ssm_state_is_fp32": [False],
        "use_token_indx": [True],
        "ragged": [True],
        "with_null_block": [True],
    },
}


def ref_gated_delta_rule_spec_recoverssm(q, k, v, b, a, A_log, dt_bias,
                                         ssm_state, query_start_loc,
                                         token_indx, state_indices,
                                         num_actual_tokens, out_dtype):
    """Float32 reference. Returns (core_attn_out, records, final_states) where
    records[n] is [HV, len_n, V + K + 1] and final_states[n] is the state after
    the request's last token (None for null-block rows)."""
    eps = 0.000001
    num_k_heads, head_k_dim = q.shape[1], q.shape[2]
    num_v_heads, head_v_dim = v.shape[1], v.shape[2]
    rep = num_v_heads // num_k_heads
    scale = 1.0 / math.sqrt(head_k_dim)
    neg_A = -torch.exp(A_log.float())
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)

    out = torch.zeros(num_actual_tokens,
                      num_v_heads,
                      head_v_dim,
                      dtype=out_dtype,
                      device=q.device)
    records, finals = [], []
    for n in range(state_indices.shape[0]):
        s = int(query_start_loc[n].item())
        e = int(query_start_loc[n + 1].item())
        idx = int(state_indices[n].item())
        glob = (token_indx[s:e].long() if token_indx is not None else
                torch.arange(s, e, device=q.device))
        if idx <= NULL_BLOCK_ID:
            out[glob] = 0
            records.append(None)
            finals.append(None)
            continue
        qq = q[s:e].float()
        kk = k[s:e].float()
        qq = qq * torch.rsqrt(qq.pow(2).sum(-1, keepdim=True) + eps) * scale
        kk = kk * torch.rsqrt(kk.pow(2).sum(-1, keepdim=True) + eps)
        if rep > 1:
            qq = qq.repeat_interleave(rep, dim=1)
            kk = kk.repeat_interleave(rep, dim=1)
        vv = v[s:e].float()
        beta = torch.sigmoid(b[s:e].float())
        g = torch.exp(neg_A * softplus(a[s:e].float() + dt_bias.float()))

        state = ssm_state[idx].float().clone()  # [HV, V, K]
        rec = torch.zeros(num_v_heads,
                          e - s,
                          head_v_dim + head_k_dim + 1,
                          dtype=torch.float32,
                          device=q.device)
        for t in range(e - s):
            state *= g[t].view(-1, 1, 1)
            kv_mem = torch.einsum("hvk,hk->hv", state, kk[t])
            corr = (vv[t] - kv_mem) * beta[t].unsqueeze(-1)
            state += torch.einsum("hv,hk->hvk", corr, kk[t])
            out[glob[t]] = torch.einsum("hvk,hk->hv", state,
                                        qq[t]).to(out_dtype)
            rec[:, t, :head_v_dim] = corr
            rec[:, t, head_v_dim:head_v_dim + head_k_dim] = kk[t]
            rec[:, t, head_v_dim + head_k_dim] = g[t]
        records.append(rec)
        finals.append(state)
    return out, records, finals


def replay(state0, rec, head_v_dim):
    """Fold replay records into the checkpoint, as vLLM's commit does:
    S_t = decay_t * S_{t-1} + corr_t k_t^T."""
    state = state0.float().clone()
    for t in range(rec.shape[1]):
        corr = rec[:, t, :head_v_dim]
        kk = rec[:, t, head_v_dim:-1]
        decay = rec[:, t, -1]
        state = decay.view(-1, 1, 1) * state + torch.einsum(
            "hv,hk->hvk", corr, kk)
    return state


@pytest.mark.parametrize("num_spec_decodes", NUM_SPEC_DECODES)
@pytest.mark.parametrize("num_spec_tokens", NUM_SPEC_TOKENS)
@pytest.mark.parametrize("num_v_heads", NUM_V_HEADS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=format_tc)
@pytest.mark.parametrize("ssm_state_is_fp32", [False, True])
@pytest.mark.parametrize("use_token_indx", [True, False])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("with_null_block", [False, True])
@torch.inference_mode()
def test_gated_delta_rule_spec_recoverssm(num_spec_decodes, num_spec_tokens,
                                          num_v_heads, dtype,
                                          ssm_state_is_fp32, use_token_indx,
                                          ragged, with_null_block):
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn attention kernels testing on PVC.")

    device = "xpu"
    random.seed(123)
    torch.manual_seed(123)
    num_k_heads, head_k_dim, head_v_dim, tp_size = 16, 128, 128, 1
    S = num_spec_tokens
    ssm_state_dtype = torch.float32 if ssm_state_is_fp32 else dtype
    cache_batch_size = 64

    # Per-request token counts; ragged batches shorten some requests.
    lens = [S] * num_spec_decodes
    if ragged:
        for n in range(0, num_spec_decodes, 2):
            lens[n] = random.randint(1, S)
    num_actual_tokens = sum(lens)
    query_start_loc = torch.tensor([0] + lens,
                                   dtype=torch.int32,
                                   device=device).cumsum(0).to(torch.int32)

    # One state block per request; block 0 is the null block.
    blocks = random.sample(range(1, cache_batch_size), num_spec_decodes)
    if with_null_block:
        blocks[num_spec_decodes // 2] = NULL_BLOCK_ID
    state_indices = torch.tensor(blocks, dtype=torch.int32, device=device)

    token_indx = (torch.randperm(num_actual_tokens, device=device).to(
        torch.int32) if use_token_indx else None)

    q = torch.randn(num_actual_tokens,
                    num_k_heads,
                    head_k_dim,
                    dtype=dtype,
                    device=device)
    k = torch.randn_like(q)
    v = torch.randn(num_actual_tokens,
                    num_v_heads,
                    head_v_dim,
                    dtype=dtype,
                    device=device)
    b = torch.randn(num_actual_tokens, num_v_heads, dtype=dtype, device=device)
    a = torch.randn_like(b)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device=device)
    ssm_state = torch.randn(cache_batch_size,
                            num_v_heads,
                            head_v_dim,
                            head_k_dim,
                            dtype=ssm_state_dtype,
                            device=device) * 0.1
    ssm_state_before = ssm_state.clone()
    replay_state = torch.zeros(cache_batch_size,
                               num_v_heads,
                               S,
                               head_v_dim + head_k_dim + 1,
                               dtype=torch.float32,
                               device=device)
    # Sentinel so that rows the kernel must write are detectable.
    core_attn_out = torch.full((num_actual_tokens, num_v_heads, head_v_dim),
                               7.0,
                               dtype=dtype,
                               device=device)

    torch.ops._xpu_C.gated_delta_rule_spec_recoverssm(
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        num_v_heads,
        head_v_dim,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
        replay_state=replay_state,
        num_spec_decodes=num_spec_decodes,
        spec_query_start_loc=query_start_loc,
        spec_token_indx=token_indx,
        spec_state_indices_tensor=state_indices,
        null_block_id=NULL_BLOCK_ID,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size)

    ref_out, ref_records, ref_finals = ref_gated_delta_rule_spec_recoverssm(
        q, k, v, b, a, A_log, dt_bias, ssm_state_before, query_start_loc,
        token_indx, state_indices, num_actual_tokens, dtype)

    atol = 5e-2
    rtol = 5e-2
    torch.testing.assert_close(core_attn_out,
                               ref_out,
                               atol=atol,
                               rtol=rtol,
                               equal_nan=True)

    # The checkpoint is read-only.
    assert torch.equal(ssm_state, ssm_state_before)

    for n, idx in enumerate(blocks):
        if idx <= NULL_BLOCK_ID:
            continue
        L = lens[n]
        rec = replay_state[idx, :, :L]
        # The records are float32 from the same inputs on both sides, so they
        # are held much tighter than the bf16/fp16 outputs.
        torch.testing.assert_close(rec, ref_records[n], atol=1e-4, rtol=1e-4)
        # Positions past the request's length are not written.
        assert torch.all(replay_state[idx, :, L:] == 0)
        # Folding the records reproduces the recurrence state.
        torch.testing.assert_close(replay(ssm_state_before[idx], rec,
                                          head_v_dim),
                                   ref_finals[n],
                                   atol=1e-3,
                                   rtol=1e-3)

    # Blocks no live request owns (including the null block) stay untouched.
    used = {idx for idx in blocks if idx > NULL_BLOCK_ID}
    untouched = [i for i in range(cache_batch_size) if i not in used]
    assert torch.all(replay_state[untouched] == 0)
