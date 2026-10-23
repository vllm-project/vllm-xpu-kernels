# SPDX-License-Identifier: Apache-2.0
"""Mixed spec-decode + non-spec batches through the fused gdn_attention op.

Differential test: a single mixed-batch invocation must be bitwise identical
to chaining the two single-population invocations (each independently
validated against python references by test_gdn_attn.py) on cloned states —
including with fully shuffled token index sets and regardless of allocator
history. Historically this test exposed, in order: the mixed-batch rejection
(engine crash under concurrent MTP), uninitialized spec intermediates, and
the XE2 delta epilogue's undocumented assumption that token_indx values are
chunk-contiguous (single lookup + block write, corrupting neighboring rows /
overrunning the tensor for arbitrary index sets — fixed via compact staging
+ per-row scatter in the interface).
"""

import os
import random

import pytest
import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401


@pytest.mark.parametrize("num_prefills,num_decodes", [(1, 2), (2, 0), (0, 3)])
@pytest.mark.parametrize("num_spec_decodes,num_spec_tokens", [(2, 2), (1, 3)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("reorder_input", [True, False])
def test_gdn_attention_mixed_batch(num_prefills, num_decodes,
                                   num_spec_decodes, num_spec_tokens, dtype,
                                   reorder_input):
    if os.getenv("SKIP_ACC_ERROR_KERNEL") == "1":
        pytest.skip("skip gdn attention kernels testing on PVC.")

    device = "xpu"
    random.seed(7)
    torch.manual_seed(7)

    num_k_heads, head_k_dim = 8, 64
    num_v_heads, head_v_dim = 16, 64
    width, tp_size = 4, 1
    activation = "silu"
    ssm_state_dtype = torch.float32

    K = num_spec_tokens
    prefill_lens = [random.randint(9, 50) for _ in range(num_prefills)]
    non_spec_lens = prefill_lens + [1] * num_decodes
    non_spec_token = sum(non_spec_lens)
    spec_token = num_spec_decodes * K
    num_actual_tokens = non_spec_token + spec_token
    cache_batch_size = 200

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn(num_actual_tokens, mixed_qkvz_size,
                                        dtype=dtype, device=device)
    projected_states_ba = torch.randn(num_actual_tokens, mixed_ba_size,
                                      dtype=dtype, device=device)
    conv_weights = torch.randn(mixed_qkv_size, width, dtype=dtype,
                               device=device)
    conv_bias = torch.randn(mixed_qkv_size, dtype=dtype, device=device)
    A_log = torch.randn(num_v_heads // tp_size, dtype=torch.float32,
                        device=device)
    dt_bias = torch.randn(num_v_heads // tp_size, dtype=dtype, device=device)

    # Sliding-window conv-state convention (upstream #544/#545): the spec path
    # needs width - 1 history rows plus K - 1 draft rows per cache line.
    conv_state0 = torch.randn(cache_batch_size, width - 1 + (K - 1),
                              mixed_qkv_size, dtype=dtype, device=device)
    ssm_state0 = torch.randn(cache_batch_size, num_v_heads // tp_size,
                             head_v_dim, head_k_dim, dtype=ssm_state_dtype,
                             device=device)

    # Disjoint global token positions for the two populations.
    perm = torch.randperm(num_actual_tokens, device=device).to(torch.int32)
    non_spec_token_indx = perm[:non_spec_token].contiguous()
    spec_token_indx = perm[non_spec_token:].contiguous()

    non_spec_query_start_loc = torch.tensor(
        [0] + list(torch.tensor(non_spec_lens).cumsum(0)),
        dtype=torch.int32, device=device)
    has_initial_state = torch.tensor(
        [random.random() < 0.5 for _ in range(len(non_spec_lens))],
        dtype=torch.bool, device=device)

    # Disjoint cache slots between the populations.
    slots = random.sample(range(cache_batch_size),
                          len(non_spec_lens) + num_spec_decodes * K)
    non_spec_state_indices_tensor = torch.tensor(
        slots[:len(non_spec_lens)], dtype=torch.int32, device=device)
    spec_state_indices_tensor = torch.tensor(
        slots[len(non_spec_lens):], dtype=torch.int32,
        device=device).reshape(num_spec_decodes, K)
    # The engine always accepts at least the bonus token; the sliding-window
    # kernel indexes row num_accepted - 1, so 0 is out of contract.
    num_accepted_tokens = torch.tensor(
        [random.randint(1, K) for _ in range(num_spec_decodes)],
        dtype=torch.int32, device=device)
    spec_query_start_loc = (torch.arange(
        num_spec_decodes + 1, dtype=torch.int32, device=device) * K)

    def run(mode):
        conv_state = conv_state0.clone()
        ssm_state = ssm_state0.clone()
        out = torch.zeros(num_actual_tokens, num_v_heads // tp_size,
                          head_v_dim, dtype=dtype, device=device)
        z = torch.zeros_like(out)

        def call(with_non_spec, with_spec):
            torch.ops._xpu_C.gdn_attention(
                out, z, projected_states_qkvz, projected_states_ba,
                num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                conv_state=conv_state, ssm_state=ssm_state,
                conv_weights=conv_weights, conv_bias=conv_bias,
                activation=activation, A_log=A_log, dt_bias=dt_bias,
                num_prefills=num_prefills if with_non_spec else 0,
                num_decodes=num_decodes if with_non_spec else 0,
                num_spec_decodes=num_spec_decodes if with_spec else 0,
                has_initial_state=has_initial_state if with_non_spec else None,
                non_spec_query_start_loc=(non_spec_query_start_loc
                                          if with_non_spec else None),
                non_spec_token_indx=(non_spec_token_indx
                                     if with_non_spec else None),
                non_spec_state_indices_tensor=(non_spec_state_indices_tensor
                                               if with_non_spec else None),
                spec_query_start_loc=(spec_query_start_loc
                                      if with_spec else None),
                spec_token_indx=spec_token_indx if with_spec else None,
                spec_state_indices_tensor=(spec_state_indices_tensor
                                           if with_spec else None),
                num_accepted_tokens=(num_accepted_tokens
                                     if with_spec else None),
                num_actual_tokens=num_actual_tokens, tp_size=tp_size,
                reorder_input=reorder_input)

        if mode == "mixed":
            call(True, True)
        else:
            call(True, False)
            call(False, True)
        torch.xpu.synchronize()
        return out, z, conv_state, ssm_state

    out_m, z_m, conv_m, ssm_m = run("mixed")
    out_s, z_s, conv_s, ssm_s = run("sequential")

    assert torch.equal(out_m, out_s), "core_attn_out differs (STRICT)"
    assert torch.equal(z_m, z_s), "z differs (STRICT)"
    assert torch.equal(conv_m, conv_s), "conv_state differs from composition"
    assert torch.equal(ssm_m, ssm_s), "ssm_state differs from composition"
