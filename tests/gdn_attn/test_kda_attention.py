# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels._xpu_C  # noqa: F401
from tests.utils import format_tc


def _conv_history(
    conv_state: torch.Tensor,
    slot: int,
    stream: int,
    hidden_dim: int,
) -> torch.Tensor:
    start = stream * hidden_dim
    end = start + hidden_dim
    if conv_state.shape[1] == 3 * hidden_dim:
        return conv_state[slot, start:end, :].transpose(0, 1)
    return conv_state[slot, :, start:end]


def _reference_sequence(
    output: torch.Tensor,
    conv_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    projections: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    conv_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    global_tokens: list[int],
    initial_slot: int,
    load_initial_state: bool,
    save_slots: list[int] | None,
    num_heads: int,
    head_dim: int,
) -> None:
    hidden_dim = num_heads * head_dim
    width = conv_weights[0].shape[1]

    for stream, (projection, weight, conv_output) in enumerate(
        zip(projections, conv_weights, conv_outputs)
    ):
        if load_initial_state:
            history = _conv_history(
                conv_state, initial_slot, stream, hidden_dim
            ).float().clone()
        else:
            history = torch.zeros(width - 1, hidden_dim)

        for step, global_token in enumerate(global_tokens):
            current = projection[global_token].to(conv_state.dtype).float()
            window = torch.cat((history, current.unsqueeze(0)), dim=0)
            convolved = F.silu((window * weight.t()).sum(dim=0))
            conv_output[global_token].copy_(convolved.to(conv_output.dtype))
            history = torch.cat((history[1:], current.unsqueeze(0)), dim=0)
            if save_slots is not None:
                _conv_history(
                    conv_state, save_slots[step], stream, hidden_dim
                ).copy_(history.to(conv_state.dtype))

        if save_slots is None:
            _conv_history(
                conv_state, initial_slot, stream, hidden_dim
            ).copy_(history.to(conv_state.dtype))

    if load_initial_state:
        state = recurrent_state[initial_slot].clone()
    else:
        state = torch.zeros_like(recurrent_state[initial_slot])

    dt = dt_bias.reshape(num_heads, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    q_all, k_all, v_all = conv_outputs
    for step, global_token in enumerate(global_tokens):
        q = q_all[global_token].reshape(num_heads, head_dim).float()
        k = k_all[global_token].reshape(num_heads, head_dim).float()
        v = v_all[global_token].reshape(num_heads, head_dim).float()
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        q = q * scale

        gate = -torch.exp(a_log.reshape(num_heads, 1)) * F.softplus(
            raw_gate[0, global_token].float() + dt
        )
        state *= gate.exp().unsqueeze(1)
        kv_memory = (state * k.unsqueeze(1)).sum(-1)
        delta = (
            v - kv_memory
        ) * beta[0, global_token].float().unsqueeze(-1)
        state += delta.unsqueeze(-1) * k.unsqueeze(1)
        result = (state * q.unsqueeze(1)).sum(-1)
        output[0, global_token].copy_(result.to(output.dtype))

        if save_slots is not None:
            recurrent_state[save_slots[step]].copy_(state)

    if save_slots is None:
        recurrent_state[initial_slot].copy_(state)


def _reference_kda(
    core_attn_out: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    q_conv_weight: torch.Tensor,
    k_conv_weight: torch.Tensor,
    v_conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor | None,
    non_spec_token_indx: torch.Tensor | None,
    non_spec_state_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    spec_query_start_loc: torch.Tensor | None,
    spec_token_indx: torch.Tensor | None,
    spec_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actual_tokens = q_proj.shape[0]
    hidden_dim = num_heads * head_dim
    conv_outputs = tuple(
        torch.empty(actual_tokens, hidden_dim, dtype=q_proj.dtype)
        for _ in range(3)
    )
    projections = (q_proj, k_proj, v_proj)
    weights = (q_conv_weight, k_conv_weight, v_conv_weight)

    if non_spec_query_start_loc is not None:
        assert non_spec_state_indices is not None
        for batch in range(non_spec_state_indices.numel()):
            start = int(non_spec_query_start_loc[batch])
            end = int(non_spec_query_start_loc[batch + 1])
            global_tokens = [
                (
                    int(non_spec_token_indx[token])
                    if non_spec_token_indx is not None
                    else token
                )
                for token in range(start, end)
            ]
            load_initial = (
                has_initial_state is None or bool(has_initial_state[batch])
            )
            _reference_sequence(
                core_attn_out,
                conv_outputs,
                projections,
                weights,
                raw_gate,
                beta,
                conv_state,
                recurrent_state,
                a_log,
                dt_bias,
                global_tokens,
                int(non_spec_state_indices[batch]),
                load_initial,
                None,
                num_heads,
                head_dim,
            )
    if spec_query_start_loc is not None:
        assert spec_token_indx is not None
        assert spec_state_indices is not None
        assert num_accepted_tokens is not None
        for batch in range(spec_state_indices.shape[0]):
            start = int(spec_query_start_loc[batch])
            end = int(spec_query_start_loc[batch + 1])
            global_tokens = [
                int(spec_token_indx[token]) for token in range(start, end)
            ]
            initial_col = max(int(num_accepted_tokens[batch]) - 1, 0)
            save_slots = [int(slot) for slot in spec_state_indices[batch]]
            _reference_sequence(
                core_attn_out,
                conv_outputs,
                projections,
                weights,
                raw_gate,
                beta,
                conv_state,
                recurrent_state,
                a_log,
                dt_bias,
                global_tokens,
                int(spec_state_indices[batch, initial_col]),
                True,
                save_slots,
                num_heads,
                head_dim,
            )
    return conv_outputs


def _make_inputs(
    num_actual_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    dim_first: bool,
):
    torch.manual_seed(42)
    capture_tokens = num_actual_tokens + 2
    hidden_dim = num_heads * head_dim
    width = 4
    num_slots = 12

    projections = tuple(
        (torch.randn(capture_tokens, hidden_dim) * 0.2).to(dtype)
        for _ in range(3)
    )
    raw_gate = (
        torch.randn(1, capture_tokens, num_heads, head_dim) * 0.2
    ).to(dtype)
    beta = torch.randn(1, capture_tokens, num_heads).sigmoid()
    weights = tuple(
        torch.randn(hidden_dim, width, dtype=torch.float32) * 0.1
        for _ in range(3)
    )
    if dim_first:
        conv_shape = (num_slots, 3 * hidden_dim, width - 1)
    else:
        conv_shape = (num_slots, width - 1, 3 * hidden_dim)
    conv_state = (torch.randn(conv_shape) * 0.1).to(dtype)
    recurrent_state = (
        torch.randn(num_slots, num_heads, head_dim, head_dim) * 0.05
    )
    a_log = torch.randn(1, 1, num_heads, 1) * 0.1
    dt_bias = torch.randn(hidden_dim) * 0.1
    core_attn_out = torch.full(
        (1, capture_tokens, num_heads, head_dim), 7.0, dtype=dtype
    )
    return (
        projections,
        raw_gate,
        beta,
        weights,
        conv_state,
        recurrent_state,
        a_log,
        dt_bias,
        core_attn_out,
    )


def _to_page_strided_xpu_cache(tensor: torch.Tensor) -> torch.Tensor:
    slot_numel = tensor[0].numel()
    slot_stride = slot_numel + 17
    storage = torch.empty(
        tensor.shape[0] * slot_stride, dtype=tensor.dtype, device="xpu"
    )
    cache = torch.as_strided(
        storage,
        tensor.shape,
        (slot_stride, *tensor.stride()[1:]),
    )
    cache.copy_(tensor)
    assert not cache.is_contiguous()
    assert cache[0].is_contiguous()
    return cache


@pytest.mark.parametrize(
    "page_strided_cache",
    [False, True],
    ids=["contiguous-cache", "page-strided-cache"],
)
@pytest.mark.parametrize(
    ("dtype", "head_dim", "dim_first", "mode"),
    [
        (torch.float16, 32, False, "prefill"),
        (torch.bfloat16, 128, True, "decode"),
        (torch.float16, 64, True, "prefill+decode"),
    ],
    ids=lambda value: (
        format_tc(value) if isinstance(value, torch.dtype) else str(value)
    ),
)
@torch.inference_mode()
def test_kda_attention_non_spec(
    dtype, head_dim, dim_first, mode, page_strided_cache
):
    num_actual_tokens = {
        "prefill": 5,
        "decode": 3,
        "prefill+decode": 5,
    }[mode]
    num_heads = 2
    (
        projections,
        raw_gate,
        beta,
        weights,
        conv_state,
        recurrent_state,
        a_log,
        dt_bias,
        core_attn_out,
    ) = _make_inputs(
        num_actual_tokens,
        num_heads,
        head_dim,
        dtype,
        dim_first,
    )
    if mode == "prefill":
        query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
        state_indices = torch.tensor([1, 3], dtype=torch.int32)
        has_initial_state = torch.tensor([False, True])
        num_prefills, num_decodes = 2, 0
    elif mode == "decode":
        query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        state_indices = torch.tensor([1, 3, 5], dtype=torch.int32)
        has_initial_state = None
        num_prefills, num_decodes = 0, 3
    else:
        query_start_loc = torch.tensor([0, 3, 4, 5], dtype=torch.int32)
        state_indices = torch.tensor([1, 3, 5], dtype=torch.int32)
        has_initial_state = torch.tensor([False, True, True])
        num_prefills, num_decodes = 1, 2

    reference_output = core_attn_out.clone()
    reference_conv_state = conv_state.clone()
    reference_recurrent_state = recurrent_state.clone()
    _reference_kda(
        reference_output,
        *(projection[:num_actual_tokens] for projection in projections),
        raw_gate[:, :num_actual_tokens],
        beta[:, :num_actual_tokens],
        reference_conv_state,
        reference_recurrent_state,
        *weights,
        a_log,
        dt_bias,
        query_start_loc,
        None,
        state_indices,
        has_initial_state,
        None,
        None,
        None,
        None,
        num_heads,
        head_dim,
    )

    device = "xpu"
    actual_output = core_attn_out.to(device)
    if page_strided_cache:
        actual_conv_state = _to_page_strided_xpu_cache(conv_state)
        actual_recurrent_state = _to_page_strided_xpu_cache(recurrent_state)
    else:
        actual_conv_state = conv_state.to(device)
        actual_recurrent_state = recurrent_state.to(device)
    torch.ops._xpu_C.kda_attention(
        actual_output,
        *(projection.to(device) for projection in projections),
        raw_gate.to(device),
        beta.to(device),
        actual_conv_state,
        actual_recurrent_state,
        *(weight.to(device) for weight in weights),
        a_log.to(device),
        dt_bias.to(device),
        num_prefills,
        num_decodes,
        0,
        None if has_initial_state is None else has_initial_state.to(device),
        query_start_loc.to(device),
        None,
        state_indices.to(device),
        None,
        None,
        None,
        None,
        num_actual_tokens,
    )

    tolerance = 6e-2 if dtype == torch.bfloat16 else 3e-2
    torch.testing.assert_close(
        actual_output.cpu(),
        reference_output,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        actual_conv_state.cpu(),
        reference_conv_state,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        actual_recurrent_state.cpu(),
        reference_recurrent_state,
        atol=tolerance,
        rtol=tolerance,
    )


@torch.inference_mode()
def test_kda_split_ops_compose_to_reference():
    num_actual_tokens = 5
    num_heads = 2
    head_dim = 32
    dtype = torch.bfloat16
    (
        projections,
        raw_gate,
        beta,
        weights,
        conv_state,
        recurrent_state,
        a_log,
        dt_bias,
        core_attn_out,
    ) = _make_inputs(
        num_actual_tokens,
        num_heads,
        head_dim,
        dtype,
        dim_first=False,
    )
    query_start_loc = torch.tensor([0, 3, 4, 5], dtype=torch.int32)
    state_indices = torch.tensor([1, 3, 5], dtype=torch.int32)
    has_initial_state = torch.tensor([False, True, True])

    reference_output = core_attn_out.clone()
    reference_conv_state = conv_state.clone()
    reference_recurrent_state = recurrent_state.clone()
    reference_qkv = _reference_kda(
        reference_output,
        *(projection[:num_actual_tokens] for projection in projections),
        raw_gate[:, :num_actual_tokens],
        beta[:, :num_actual_tokens],
        reference_conv_state,
        reference_recurrent_state,
        *weights,
        a_log,
        dt_bias,
        query_start_loc,
        None,
        state_indices,
        has_initial_state,
        None,
        None,
        None,
        None,
        num_heads,
        head_dim,
    )

    device = "xpu"
    actual_output = core_attn_out.to(device)
    actual_conv_state = _to_page_strided_xpu_cache(conv_state)
    actual_recurrent_state = _to_page_strided_xpu_cache(recurrent_state)
    device_query_start_loc = query_start_loc.to(device)
    device_state_indices = state_indices.to(device)
    device_has_initial_state = has_initial_state.to(device)
    actual_qkv = torch.ops._xpu_C.kda_causal_conv1d(
        *(projection.to(device) for projection in projections),
        actual_conv_state,
        *(weight.to(device) for weight in weights),
        1,
        2,
        0,
        device_has_initial_state,
        device_query_start_loc,
        None,
        device_state_indices,
        None,
        None,
        None,
        None,
        num_actual_tokens,
    )
    torch.ops._xpu_C.kda_gated_delta_rule(
        actual_output,
        *actual_qkv,
        raw_gate.to(device),
        beta.to(device),
        actual_recurrent_state,
        a_log.to(device),
        dt_bias.to(device),
        1,
        2,
        0,
        device_has_initial_state,
        device_query_start_loc,
        None,
        device_state_indices,
        None,
        None,
        None,
        None,
        num_actual_tokens,
    )

    tolerance = 6e-2
    for actual, reference in zip(actual_qkv, reference_qkv):
        torch.testing.assert_close(
            actual.cpu(),
            reference,
            atol=tolerance,
            rtol=tolerance,
        )
    torch.testing.assert_close(
        actual_output.cpu(),
        reference_output,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        actual_conv_state.cpu(),
        reference_conv_state,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        actual_recurrent_state.cpu(),
        reference_recurrent_state,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize(
    "mode",
    ["spec-decode", "spec-decode+prefill+decode"],
)
@torch.inference_mode()
def test_kda_attention_spec_decode(mode):
    combined_batch = mode == "spec-decode+prefill+decode"
    num_actual_tokens = 10 if combined_batch else 6
    num_heads = 2
    head_dim = 32
    dtype = torch.float16
    (
        projections,
        raw_gate,
        beta,
        weights,
        conv_state,
        recurrent_state,
        a_log,
        dt_bias,
        core_attn_out,
    ) = _make_inputs(
        num_actual_tokens,
        num_heads,
        head_dim,
        dtype,
        dim_first=False,
    )
    query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)
    token_indx = torch.tensor([1, 3, 5, 0, 2, 4], dtype=torch.int32)
    state_indices = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    accepted = torch.tensor([2, 1], dtype=torch.int32)
    if combined_batch:
        non_spec_query_start_loc = torch.tensor(
            [0, 3, 4], dtype=torch.int32
        )
        non_spec_token_indx = torch.tensor(
            [6, 7, 8, 9], dtype=torch.int32
        )
        non_spec_state_indices = torch.tensor([8, 9], dtype=torch.int32)
        has_initial_state = torch.tensor([False, True])
        num_prefills = 1
        num_decodes = 1
    else:
        non_spec_query_start_loc = None
        non_spec_token_indx = None
        non_spec_state_indices = None
        has_initial_state = None
        num_prefills = 0
        num_decodes = 0

    reference_output = core_attn_out.clone()
    reference_conv_state = conv_state.clone()
    reference_recurrent_state = recurrent_state.clone()
    _reference_kda(
        reference_output,
        *(projection[:num_actual_tokens] for projection in projections),
        raw_gate[:, :num_actual_tokens],
        beta[:, :num_actual_tokens],
        reference_conv_state,
        reference_recurrent_state,
        *weights,
        a_log,
        dt_bias,
        non_spec_query_start_loc,
        non_spec_token_indx,
        non_spec_state_indices,
        has_initial_state,
        query_start_loc,
        token_indx,
        state_indices,
        accepted,
        num_heads,
        head_dim,
    )

    device = "xpu"
    actual_output = core_attn_out.to(device)
    actual_conv_state = conv_state.to(device)
    actual_recurrent_state = recurrent_state.to(device)
    torch.ops._xpu_C.kda_attention(
        actual_output,
        *(projection.to(device) for projection in projections),
        raw_gate.to(device),
        beta.to(device),
        actual_conv_state,
        actual_recurrent_state,
        *(weight.to(device) for weight in weights),
        a_log.to(device),
        dt_bias.to(device),
        num_prefills,
        num_decodes,
        2,
        (
            None
            if has_initial_state is None
            else has_initial_state.to(device)
        ),
        (
            None
            if non_spec_query_start_loc is None
            else non_spec_query_start_loc.to(device)
        ),
        (
            None
            if non_spec_token_indx is None
            else non_spec_token_indx.to(device)
        ),
        (
            None
            if non_spec_state_indices is None
            else non_spec_state_indices.to(device)
        ),
        query_start_loc.to(device),
        token_indx.to(device),
        state_indices.to(device),
        accepted.to(device),
        num_actual_tokens,
    )

    torch.testing.assert_close(
        actual_output.cpu(), reference_output, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        actual_conv_state.cpu(),
        reference_conv_state,
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        actual_recurrent_state.cpu(),
        reference_recurrent_state,
        atol=3e-2,
        rtol=3e-2,
    )
