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


def _reference_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    gate_lower_bound: float | None,
) -> torch.Tensor:
    """Log-domain KDA decay, mirroring csrc/xpu/gdn_attn/kda_gate.hpp."""
    x = raw_gate + dt_bias
    if gate_lower_bound is None:
        return -torch.exp(a_log) * F.softplus(x)
    return gate_lower_bound * torch.sigmoid(torch.exp(a_log) * x)


def _reference_sequence(
    output: torch.Tensor,
    conv_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    projections: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    conv_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
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
    gate_lower_bound: float | None = None,
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
            history = torch.zeros(width - 1,
                                  hidden_dim,
                                  device=conv_state.device)

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

        gate = _reference_gate(
            raw_gate[0, global_token].float(),
            a_log.reshape(num_heads, 1),
            dt,
            gate_lower_bound,
        )
        state *= gate.exp().unsqueeze(1)
        kv_memory = (state * k.unsqueeze(1)).sum(-1)
        delta = (
            v - kv_memory
        ) * raw_beta[0, global_token].float().sigmoid().unsqueeze(-1)
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
    raw_beta: torch.Tensor,
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
    gate_lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actual_tokens = q_proj.shape[0]
    hidden_dim = num_heads * head_dim
    conv_outputs = tuple(
        torch.empty(actual_tokens,
                    hidden_dim,
                    dtype=q_proj.dtype,
                    device=q_proj.device)
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
                raw_beta,
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
                gate_lower_bound,
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
                raw_beta,
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
                gate_lower_bound,
            )
    return conv_outputs


def _make_inputs(
    num_actual_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    dim_first: bool,
    device: str = "xpu",
):
    torch.manual_seed(42)
    capture_tokens = num_actual_tokens + 2
    hidden_dim = num_heads * head_dim
    width = 4
    num_slots = 12

    projections = tuple(
        (torch.randn(capture_tokens, hidden_dim, device=device) * 0.2).to(
            dtype
        )
        for _ in range(3)
    )
    raw_gate = (
        torch.randn(1, capture_tokens, num_heads, head_dim, device=device)
        * 0.2
    ).to(dtype)
    # Raw logits: the kernel applies the sigmoid itself.
    raw_beta = torch.randn(1, capture_tokens, num_heads, device=device)
    weights = tuple(
        torch.randn(hidden_dim, width, dtype=torch.float32, device=device)
        * 0.1
        for _ in range(3)
    )
    if dim_first:
        conv_shape = (num_slots, 3 * hidden_dim, width - 1)
    else:
        conv_shape = (num_slots, width - 1, 3 * hidden_dim)
    conv_state = (torch.randn(conv_shape, device=device) * 0.1).to(dtype)
    recurrent_state = (
        torch.randn(num_slots, num_heads, head_dim, head_dim, device=device)
        * 0.05
    )
    a_log = torch.randn(1, 1, num_heads, 1, device=device) * 0.1
    dt_bias = torch.randn(hidden_dim, device=device) * 0.1
    core_attn_out = torch.full(
        (1, capture_tokens, num_heads, head_dim),
        7.0,
        dtype=dtype,
        device=device,
    )
    return (
        projections,
        raw_gate,
        raw_beta,
        weights,
        conv_state,
        recurrent_state,
        a_log,
        dt_bias,
        core_attn_out,
    )


def _as_fused_qkv_views(projections):
    """Repack q/k/v as row-strided slices of one fused mixed-QKV buffer.

    This is the layout vLLM hands us: `mixed_qkv` is a slice of the wider
    `in_proj_qkvgfab` output, so the padding here keeps the row stride from
    accidentally equalling 3 * hidden_dim.
    """
    tokens, hidden = projections[0].shape
    fused = torch.zeros(
        tokens,
        3 * hidden + 17,
        dtype=projections[0].dtype,
        device=projections[0].device,
    )
    for index, projection in enumerate(projections):
        fused[:, index * hidden : (index + 1) * hidden] = projection
    return fused[:, : 3 * hidden].split(hidden, dim=-1)


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
    "gate_lower_bound",
    [None, -5.0],
    ids=["softplus-gate", "sigmoid-gate"],
)
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
        (torch.bfloat16, 128, True, "long-prefill"),
    ],
    ids=lambda value: (
        format_tc(value) if isinstance(value, torch.dtype) else str(value)
    ),
)
@torch.inference_mode()
def test_kda_attention_non_spec(
    dtype, head_dim, dim_first, mode, page_strided_cache, gate_lower_bound
):
    device = torch.device("xpu")
    num_actual_tokens = {
        "prefill": 5,
        "decode": 3,
        "prefill+decode": 5,
        "long-prefill": 131,
    }[mode]
    num_heads = 2
    (
        projections,
        raw_gate,
        raw_beta,
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
        device=device,
    )
    if mode == "long-prefill":
        query_start_loc = torch.tensor(
            [0, num_actual_tokens], dtype=torch.int32, device=device
        )
        state_indices = torch.tensor([1], dtype=torch.int32, device=device)
        has_initial_state = torch.tensor([True], device=device)
        num_prefills, num_decodes = 1, 0
    elif mode == "prefill":
        query_start_loc = torch.tensor(
            [0, 2, 5], dtype=torch.int32, device=device
        )
        state_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
        has_initial_state = torch.tensor([False, True], device=device)
        num_prefills, num_decodes = 2, 0
    elif mode == "decode":
        query_start_loc = torch.tensor(
            [0, 1, 2, 3], dtype=torch.int32, device=device
        )
        state_indices = torch.tensor(
            [1, 3, 5], dtype=torch.int32, device=device
        )
        has_initial_state = None
        num_prefills, num_decodes = 0, 3
    else:
        query_start_loc = torch.tensor(
            [0, 3, 4, 5], dtype=torch.int32, device=device
        )
        state_indices = torch.tensor(
            [1, 3, 5], dtype=torch.int32, device=device
        )
        has_initial_state = torch.tensor(
            [False, True, True], device=device
        )
        num_prefills, num_decodes = 1, 2

    reference_output = core_attn_out.clone()
    reference_conv_state = conv_state.clone()
    reference_recurrent_state = recurrent_state.clone()
    _reference_kda(
        reference_output,
        *(projection[:num_actual_tokens] for projection in projections),
        raw_gate[:, :num_actual_tokens],
        raw_beta[:, :num_actual_tokens],
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
        gate_lower_bound,
    )

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
        raw_beta.to(device),
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
        gate_lower_bound,
    )

    tolerance = 6e-2 if dtype == torch.bfloat16 else 3e-2
    torch.testing.assert_close(
        actual_output,
        reference_output,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        actual_conv_state,
        reference_conv_state,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        actual_recurrent_state,
        reference_recurrent_state,
        atol=tolerance,
        rtol=tolerance,
    )


@torch.inference_mode()
def test_kda_long_prefill_mixed_conv_cache_dtype():
    device = torch.device("xpu")
    num_actual_tokens = 131
    num_heads = 2
    head_dim = 128
    (
        projections,
        raw_gate,
        raw_beta,
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
        torch.bfloat16,
        True,
        device=device,
    )
    conv_state = conv_state.float()
    query_start_loc = torch.tensor(
        [0, num_actual_tokens], dtype=torch.int32, device=device
    )
    state_indices = torch.tensor([1], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True], device=device)

    reference_output = core_attn_out.clone()
    reference_conv_state = conv_state.clone()
    reference_recurrent_state = recurrent_state.clone()
    reference_conv_outputs = _reference_kda(
        reference_output,
        *projections,
        raw_gate,
        raw_beta,
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

    actual_conv_state = conv_state.clone()
    actual_conv_outputs = torch.ops._xpu_C.kda_causal_conv1d(
        *projections,
        actual_conv_state,
        *weights,
        1,
        0,
        0,
        has_initial_state,
        query_start_loc,
        None,
        state_indices,
        None,
        None,
        None,
        None,
        num_actual_tokens,
    )

    for actual, reference in zip(
        actual_conv_outputs, reference_conv_outputs
    ):
        torch.testing.assert_close(
            actual,
            reference[:num_actual_tokens],
            atol=1e-2,
            rtol=1e-2,
        )
    torch.testing.assert_close(
        actual_conv_state, reference_conv_state, atol=1e-6, rtol=1e-6
    )


@torch.inference_mode()
def test_kda_split_ops_compose_to_reference():
    device = torch.device("xpu")
    num_actual_tokens = 5
    num_heads = 2
    head_dim = 32
    dtype = torch.bfloat16
    (
        projections,
        raw_gate,
        raw_beta,
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
        device=device,
    )
    query_start_loc = torch.tensor(
        [0, 3, 4, 5], dtype=torch.int32, device=device
    )
    state_indices = torch.tensor(
        [1, 3, 5], dtype=torch.int32, device=device
    )
    has_initial_state = torch.tensor(
        [False, True, True], device=device
    )

    reference_output = core_attn_out.clone()
    reference_conv_state = conv_state.clone()
    reference_recurrent_state = recurrent_state.clone()
    reference_qkv = _reference_kda(
        reference_output,
        *(projection[:num_actual_tokens] for projection in projections),
        raw_gate[:, :num_actual_tokens],
        raw_beta[:, :num_actual_tokens],
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
        raw_beta.to(device),
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
            actual,
            reference,
            atol=tolerance,
            rtol=tolerance,
        )
    torch.testing.assert_close(
        actual_output,
        reference_output,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        actual_conv_state,
        reference_conv_state,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        actual_recurrent_state,
        reference_recurrent_state,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize(
    "gate_lower_bound",
    [None, -5.0],
    ids=["softplus-gate", "sigmoid-gate"],
)
@pytest.mark.parametrize(
    "mode",
    ["spec-decode", "spec-decode+prefill+decode"],
)
@pytest.mark.parametrize("qkv_layout", ["split", "fused"])
@torch.inference_mode()
def test_kda_attention_spec_decode(mode, gate_lower_bound, qkv_layout):
    device = torch.device("xpu")
    combined_batch = mode == "spec-decode+prefill+decode"
    num_actual_tokens = 10 if combined_batch else 6
    num_heads = 2
    head_dim = 32
    dtype = torch.float16
    (
        projections,
        raw_gate,
        raw_beta,
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
        device=device,
    )
    if qkv_layout == "fused":
        projections = _as_fused_qkv_views(projections)
    query_start_loc = torch.tensor(
        [0, 3, 6], dtype=torch.int32, device=device
    )
    token_indx = torch.tensor(
        [1, 3, 5, 0, 2, 4], dtype=torch.int32, device=device
    )
    state_indices = torch.tensor(
        [[1, 2, 3], [4, 5, 6]], dtype=torch.int32, device=device
    )
    accepted = torch.tensor([2, 1], dtype=torch.int32, device=device)
    if combined_batch:
        non_spec_query_start_loc = torch.tensor(
            [0, 3, 4], dtype=torch.int32, device=device
        )
        non_spec_token_indx = torch.tensor(
            [6, 7, 8, 9], dtype=torch.int32, device=device
        )
        non_spec_state_indices = torch.tensor(
            [8, 9], dtype=torch.int32, device=device
        )
        has_initial_state = torch.tensor([False, True], device=device)
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
        raw_beta[:, :num_actual_tokens],
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
        gate_lower_bound,
    )

    actual_output = core_attn_out.to(device)
    actual_conv_state = conv_state.to(device)
    actual_recurrent_state = recurrent_state.to(device)
    torch.ops._xpu_C.kda_attention(
        actual_output,
        *(projection.to(device) for projection in projections),
        raw_gate.to(device),
        raw_beta.to(device),
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
        gate_lower_bound,
    )

    torch.testing.assert_close(
        actual_output, reference_output, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        actual_conv_state,
        reference_conv_state,
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        actual_recurrent_state,
        reference_recurrent_state,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("gate_lower_bound", [None, -5.0])
@torch.inference_mode()
def test_kda_attention_accepts_fused_mixed_qkv(mode, gate_lower_bound):
    """Row-strided views of a fused QKV projection must match contiguous ones.

    vLLM produces one fused ``mixed_qkv`` projection and slices q/k/v out of
    it. Consuming those slices directly avoids materializing a QKV-major copy,
    so the kernel has to give bit-identical results for either layout.
    """
    device = "xpu"
    num_heads, head_dim, width = 8, 64, 4
    hidden_dim = num_heads * head_dim
    batch_size = 2
    seq_len = 96 if mode == "prefill" else 1
    num_actual_tokens = batch_size * seq_len

    torch.manual_seed(7)
    # Mimic vLLM: mixed_qkv is itself a slice of a wider fused projection, so
    # its row stride is larger than 3 * hidden_dim.
    fused = (
        torch.randn(num_actual_tokens, 3 * hidden_dim + 17, device=device) * 0.2
    ).to(torch.bfloat16)
    mixed_qkv = fused[:, : 3 * hidden_dim]
    strided = mixed_qkv.split(hidden_dim, dim=-1)
    contiguous = tuple(projection.contiguous() for projection in strided)

    raw_gate = (
        torch.randn(
            1, num_actual_tokens, num_heads, head_dim, device=device
        )
        * 0.2
    ).to(torch.bfloat16)
    raw_beta = torch.randn(1, num_actual_tokens, num_heads, device=device)
    weights = tuple(
        torch.randn(hidden_dim, width, dtype=torch.float32, device=device)
        * 0.1
        for _ in range(3)
    )
    a_log = torch.randn(1, 1, num_heads, 1, device=device) * 0.1
    dt_bias = torch.randn(hidden_dim, device=device) * 0.1
    query_start_loc = torch.arange(
        0, num_actual_tokens + 1, seq_len, device=device, dtype=torch.int32
    )
    state_indices = torch.arange(
        batch_size, device=device, dtype=torch.int32
    )
    has_initial_state = torch.zeros(
        batch_size, device=device, dtype=torch.bool
    )
    num_prefills = batch_size if mode == "prefill" else 0
    num_decodes = 0 if mode == "prefill" else batch_size

    results = []
    for projections in (strided, contiguous):
        conv_state = torch.zeros(
            batch_size, 3 * hidden_dim, width - 1, device=device
        )
        recurrent_state = torch.zeros(
            batch_size, num_heads, head_dim, head_dim, device=device
        )
        output = torch.zeros(
            1,
            num_actual_tokens,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        torch.ops._xpu_C.kda_attention(
            output,
            *projections,
            raw_gate,
            raw_beta,
            conv_state,
            recurrent_state,
            *weights,
            a_log,
            dt_bias,
            num_prefills,
            num_decodes,
            0,
            has_initial_state,
            query_start_loc,
            None,
            state_indices,
            None,
            None,
            None,
            None,
            num_actual_tokens,
            gate_lower_bound,
        )
        results.append((output, conv_state, recurrent_state))

    for from_strided, from_contiguous in zip(*results):
        torch.testing.assert_close(
            from_strided, from_contiguous, atol=0.0, rtol=0.0
        )


@pytest.mark.parametrize("a_log_shape", [(8, ), (1, 1, 8, 1), (1, 8)])
@torch.inference_mode()
def test_kda_attention_accepts_any_contiguous_a_log_layout(a_log_shape):
    """``A_log`` carries one value per head; its shape is the caller's choice.

    vLLM keeps it as ``[num_heads]`` while this repository's tests grew up
    around the ``[1, 1, heads, 1]`` broadcast layout. The kernels only ever
    index it linearly, so requiring one of them would force a reshape on the
    caller for no reason.
    """
    device = "xpu"
    num_heads, head_dim, width = 8, 64, 4
    hidden_dim = num_heads * head_dim
    batch_size, seq_len = 2, 96
    num_actual_tokens = batch_size * seq_len

    torch.manual_seed(11)
    projections = tuple(
        (torch.randn(num_actual_tokens, hidden_dim, device=device) * 0.2).to(
            torch.bfloat16
        )
        for _ in range(3)
    )
    raw_gate = (
        torch.randn(1, num_actual_tokens, num_heads, head_dim, device=device)
        * 0.2
    ).to(torch.bfloat16)
    raw_beta = torch.randn(1, num_actual_tokens, num_heads, device=device)
    weights = tuple(
        torch.randn(hidden_dim, width, dtype=torch.float32, device=device)
        * 0.1
        for _ in range(3)
    )
    a_log_values = torch.randn(num_heads, device=device) * 0.1
    dt_bias = torch.randn(hidden_dim, device=device) * 0.1
    query_start_loc = torch.arange(
        0, num_actual_tokens + 1, seq_len, device=device, dtype=torch.int32
    )
    state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
    has_initial_state = torch.zeros(
        batch_size, device=device, dtype=torch.bool
    )

    def run(a_log):
        conv_state = torch.zeros(
            batch_size, 3 * hidden_dim, width - 1, device=device
        )
        recurrent_state = torch.zeros(
            batch_size, num_heads, head_dim, head_dim, device=device
        )
        output = torch.zeros(
            1,
            num_actual_tokens,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        torch.ops._xpu_C.kda_attention(
            output,
            *projections,
            raw_gate,
            raw_beta,
            conv_state,
            recurrent_state,
            *weights,
            a_log,
            dt_bias,
            batch_size,
            0,
            0,
            has_initial_state,
            query_start_loc,
            None,
            state_indices,
            None,
            None,
            None,
            None,
            num_actual_tokens,
            None,
        )
        return output, recurrent_state

    reference = run(a_log_values.reshape(1, 1, num_heads, 1))
    actual = run(a_log_values.reshape(a_log_shape))
    for from_reference, from_actual in zip(reference, actual):
        torch.testing.assert_close(
            from_actual, from_reference, atol=0.0, rtol=0.0
        )


@pytest.mark.parametrize(
    "mode,seq_len,num_prefills,num_decodes",
    [
        ("prefill", 256, 2, 0),
        ("decode", 1, 0, 2),
        ("short_prefill", 96, 2, 0),
    ],
)
@pytest.mark.parametrize("gate_lower_bound", [None, -5.0])
@torch.inference_mode()
def test_kda_gated_delta_rule_accepts_fused_mixed_qkv(
    mode, seq_len, num_prefills, num_decodes, gate_lower_bound
):
    """``kda_gated_delta_rule`` must accept the same row-strided q/k/v as conv.

    A caller that runs the convolution itself ends up with a packed
    ``[tokens, 3 * heads * dim]`` activation and slices q/k/v out of it. The
    recurrent and chunked kernels index q/k/v with a packed stride, so the op
    densifies such views itself; either way the result must be identical.
    ``seq_len`` spans both sides of the chunked-backend admission threshold so
    every backend is exercised.
    """
    device = "xpu"
    num_heads, head_dim = 8, 64
    hidden_dim = num_heads * head_dim
    batch_size = num_prefills + num_decodes
    num_actual_tokens = batch_size * seq_len

    torch.manual_seed(23)
    # The stride is a multiple of head_dim but not of 3 * hidden_dim, so the
    # chunked backend's alignment precondition is met without the slices being
    # trivially adjacent.
    fused = (
        torch.randn(
            num_actual_tokens, 3 * hidden_dim + head_dim, device=device
        )
        * 0.2
    ).to(torch.bfloat16)
    strided = tuple(
        fused[:, i * hidden_dim:(i + 1) * hidden_dim] for i in range(3)
    )
    contiguous = tuple(
        projection.contiguous() for projection in strided
    )
    assert strided[0].stride(0) == 3 * hidden_dim + head_dim

    raw_gate = (
        torch.randn(1, num_actual_tokens, num_heads, head_dim, device=device)
        * 0.2
    ).to(torch.bfloat16)
    raw_beta = torch.randn(1, num_actual_tokens, num_heads, device=device)
    a_log = torch.randn(num_heads, device=device) * 0.1
    dt_bias = torch.randn(hidden_dim, device=device) * 0.1
    query_start_loc = torch.arange(
        0, num_actual_tokens + 1, seq_len, device=device, dtype=torch.int32
    )
    state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
    has_initial_state = torch.zeros(
        batch_size, device=device, dtype=torch.bool
    )
    initial_state = (
        torch.randn(
            batch_size, num_heads, head_dim, head_dim, device=device
        )
        * 0.1
    )

    results = []
    for projections in (strided, contiguous):
        recurrent_state = initial_state.clone()
        output = torch.zeros(
            1,
            num_actual_tokens,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        torch.ops._xpu_C.kda_gated_delta_rule(
            output,
            *projections,
            raw_gate,
            raw_beta,
            recurrent_state,
            a_log,
            dt_bias,
            num_prefills,
            num_decodes,
            0,
            has_initial_state,
            query_start_loc,
            None,
            state_indices,
            None,
            None,
            None,
            None,
            num_actual_tokens,
            gate_lower_bound,
        )
        results.append((output, recurrent_state))

    for from_strided, from_contiguous in zip(*results):
        torch.testing.assert_close(
            from_strided, from_contiguous, atol=0.0, rtol=0.0
        )


@torch.inference_mode()
def test_kda_gated_delta_rule_accepts_independently_strided_qkv():
    """q/k/v need not share a buffer or a row pitch.

    The op densifies whatever it is given, so unlike ``kda_causal_conv1d`` --
    whose kernels really do index by a shared row stride -- there is no
    matching-pitch requirement here.
    """
    device = "xpu"
    num_heads, head_dim = 4, 64
    hidden_dim = num_heads * head_dim
    num_actual_tokens = 128

    torch.manual_seed(29)
    fused = (
        torch.randn(
            num_actual_tokens, 3 * hidden_dim, device=device
        ) * 0.2
    ).to(torch.bfloat16)
    # q and k share one pitch, v another.
    wide = (
        torch.randn(num_actual_tokens, hidden_dim + 32, device=device) * 0.2
    ).to(torch.bfloat16)
    strided = (
        fused[:, :hidden_dim],
        fused[:, hidden_dim:2 * hidden_dim],
        wide[:, :hidden_dim],
    )
    assert strided[0].stride(0) != strided[2].stride(0)
    contiguous = tuple(x.contiguous() for x in strided)

    raw_gate = (
        torch.randn(1, num_actual_tokens, num_heads, head_dim, device=device)
        * 0.2
    ).to(torch.bfloat16)
    raw_beta = torch.randn(1, num_actual_tokens, num_heads, device=device)
    a_log = torch.randn(num_heads, device=device) * 0.1
    dt_bias = torch.randn(hidden_dim, device=device) * 0.1
    query_start_loc = torch.tensor(
        [0, num_actual_tokens], device=device, dtype=torch.int32
    )
    state_indices = torch.zeros(1, device=device, dtype=torch.int32)
    has_initial_state = torch.zeros(1, device=device, dtype=torch.bool)

    results = []
    for projections in (strided, contiguous):
        recurrent_state = torch.zeros(
            1, num_heads, head_dim, head_dim, device=device
        )
        output = torch.zeros(
            1,
            num_actual_tokens,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        torch.ops._xpu_C.kda_gated_delta_rule(
            output,
            *projections,
            raw_gate,
            raw_beta,
            recurrent_state,
            a_log,
            dt_bias,
            1,
            0,
            0,
            has_initial_state,
            query_start_loc,
            None,
            state_indices,
            None,
            None,
            None,
            None,
            num_actual_tokens,
            -5.0,
        )
        results.append((output, recurrent_state))

    for from_strided, from_contiguous in zip(*results):
        torch.testing.assert_close(
            from_strided, from_contiguous, atol=0.0, rtol=0.0
        )
