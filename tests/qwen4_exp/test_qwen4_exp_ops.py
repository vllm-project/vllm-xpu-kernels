# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_xpu_kernels.qwen4_exp import (ple_conv, ple_ngram_ids,
                                        qsa_compress_groups,
                                        qsa_expand_block_indices,
                                        qsa_store_cache_rows,
                                        qsa_visible_blocks)

DEVICE = "xpu"
EOS_TOKEN_ID = 99
INT64_MASK = (1 << 64) - 1


def _wrap_int64(value):
    value &= INT64_MASK
    return value - (1 << 64) if value >= 1 << 63 else value


def _ngram_reference(
    input_ids,
    query_start_loc,
    ngram_context,
    multipliers,
    sizes,
    offsets,
    heads_per_ngram,
):
    context_length = ngram_context.shape[1]
    output = torch.empty(
        input_ids.numel(),
        context_length * heads_per_ngram,
        dtype=torch.int64,
    )
    for request in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        for token in range(start, end):
            chunk_position = token - start
            for head in range(output.shape[1]):
                order = head // heads_per_ngram + 2
                mixed = (
                    int(input_ids[token]) * int(multipliers[0])
                ) & INT64_MASK
                crossed_eos = False
                for shift in range(1, context_length + 1):
                    if chunk_position >= shift:
                        candidate = int(input_ids[token - shift])
                    else:
                        column = context_length - shift + chunk_position
                        candidate = int(ngram_context[request, column])
                    if crossed_eos:
                        candidate = EOS_TOKEN_ID
                    crossed_eos |= candidate == EOS_TOKEN_ID
                    if order > shift:
                        mixed ^= (
                            candidate * int(multipliers[shift])
                        ) & INT64_MASK
                output[token, head] = (
                    _wrap_int64(mixed) % int(sizes[head])
                    + int(offsets[head])
                )
    return output


def test_ple_ngram_ids_cross_request_and_eos_boundaries():
    input_ids = torch.tensor(
        [11, 12, 13, 14, 21, 22, 23], dtype=torch.int64
    )
    query_start_loc = torch.tensor([0, 4, 7], dtype=torch.int32)
    ngram_context = torch.tensor(
        [[7, EOS_TOKEN_ID, 9], [31, 32, 33]],
        dtype=torch.int64,
    )
    multipliers = torch.tensor([17, 31, 43, 59], dtype=torch.int64)
    sizes = torch.arange(101, 125, dtype=torch.int64)
    offsets = torch.arange(24, dtype=torch.int64) * 1000
    expected = _ngram_reference(
        input_ids,
        query_start_loc,
        ngram_context,
        multipliers,
        sizes,
        offsets,
        heads_per_ngram=8,
    )
    actual = ple_ngram_ids(
        input_ids.to(DEVICE),
        query_start_loc.to(DEVICE),
        ngram_context.to(DEVICE),
        multipliers.to(DEVICE),
        sizes.to(DEVICE),
        offsets.to(DEVICE),
        EOS_TOKEN_ID,
        8,
    )
    assert torch.equal(actual.cpu(), expected)


def test_ple_ngram_ids_has_defined_int64_wraparound():
    input_ids = torch.tensor(
        [(1 << 62) + 17, (1 << 61) + 29],
        dtype=torch.int64,
    )
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    ngram_context = torch.tensor(
        [[(1 << 60) + 3, (1 << 59) + 5]],
        dtype=torch.int64,
    )
    multipliers = torch.tensor(
        [(1 << 62) + 1, (1 << 61) + 3, (1 << 60) + 7],
        dtype=torch.int64,
    )
    sizes = torch.tensor([101, 103, 107, 109], dtype=torch.int64)
    offsets = torch.tensor([0, 1000, 2000, 3000], dtype=torch.int64)
    expected = _ngram_reference(
        input_ids,
        query_start_loc,
        ngram_context,
        multipliers,
        sizes,
        offsets,
        heads_per_ngram=2,
    )

    actual = ple_ngram_ids(
        input_ids.to(DEVICE),
        query_start_loc.to(DEVICE),
        ngram_context.to(DEVICE),
        multipliers.to(DEVICE),
        sizes.to(DEVICE),
        offsets.to(DEVICE),
        EOS_TOKEN_ID,
        2,
    )

    assert torch.equal(actual.cpu(), expected)


def test_ple_ngram_ids_guards_invalid_metadata_values():
    output = ple_ngram_ids(
        torch.tensor([7], dtype=torch.int64, device=DEVICE),
        torch.tensor([1, 1], dtype=torch.int32, device=DEVICE),
        torch.tensor([[3]], dtype=torch.int64, device=DEVICE),
        torch.tensor([5, 7], dtype=torch.int64, device=DEVICE),
        torch.tensor([0], dtype=torch.int64, device=DEVICE),
        torch.tensor([0], dtype=torch.int64, device=DEVICE),
        EOS_TOKEN_ID,
        1,
    )

    assert torch.count_nonzero(output).item() == 0


def _request_for_token(query_start_loc, token):
    for request in range(query_start_loc.numel() - 1):
        if token < int(query_start_loc[request + 1]):
            return request
    raise AssertionError("token is outside query ranges")


def _conv_reference(
    inputs,
    state,
    weights,
    state_indices,
    mode,
    dilation,
    query_start_loc=None,
    num_accepted_tokens=None,
    has_initial_states=None,
    spec_query_len=1,
):
    inputs = inputs.clone()
    original_state = state.clone()
    output = torch.empty_like(inputs)
    kernel_size = weights.shape[1]
    state_length = (kernel_size - 1) * dilation
    if mode == "decode":
        token_count = inputs.shape[0]
        num_requests = token_count
    else:
        token_count = int(query_start_loc[-1])
        num_requests = query_start_loc.numel() - 1

    for token in range(token_count):
        if mode == "decode":
            request = token
            query_start = token
            position = 0
            slot_offset = 0
        else:
            request = _request_for_token(query_start_loc, token)
            query_start = int(query_start_loc[request])
            position = token - query_start
            slot_offset = 0
            if mode == "spec":
                slot_offset = min(
                    max(int(num_accepted_tokens[request]) - 1, 0),
                    spec_query_len - 1,
                )
        state_id = int(state_indices[request])
        state_ok = 0 < state_id < state.shape[0]
        has_initial = (
            bool(has_initial_states[request])
            if has_initial_states is not None
            else state_ok
        )
        read_state = state_ok and has_initial
        output_ok = True if mode == "spec" else state_ok
        accumulator = torch.zeros(inputs.shape[1], dtype=torch.float32)
        for tap in range(kernel_size):
            history = position + dilation * tap
            if history < state_length:
                value = (
                    original_state[state_id, :, history + slot_offset].float()
                    if read_state
                    else torch.zeros_like(accumulator)
                )
            else:
                input_token = query_start + history - state_length
                value = inputs[input_token].float()
            accumulator += value * weights[:, tap].float()
        conv = accumulator.to(torch.bfloat16).float()
        output[token] = (
            torch.nn.functional.silu(conv).to(torch.bfloat16)
            if output_ok
            else torch.zeros_like(output[token])
        )

    if mode == "decode":
        for request in range(num_requests):
            state_id = int(state_indices[request])
            if not 0 < state_id < state.shape[0]:
                continue
            has_initial = (
                bool(has_initial_states[request])
                if has_initial_states is not None
                else True
            )
            if has_initial:
                state[state_id, :, :-1] = original_state[state_id, :, 1:]
            else:
                state[state_id, :, :-1].zero_()
            state[state_id, :, -1] = inputs[request]
        return output, state

    for request in range(num_requests):
        state_id = int(state_indices[request])
        if not 0 < state_id < state.shape[0]:
            continue
        query_start = int(query_start_loc[request])
        query_end = int(query_start_loc[request + 1])
        query_length = query_end - query_start
        if mode == "prefill" and query_length <= 0:
            continue
        slot_offset = 0
        shift = query_length
        write_width = state_length
        has_initial = (
            bool(has_initial_states[request])
            if has_initial_states is not None
            else True
        )
        if mode == "spec":
            slot_offset = min(
                max(int(num_accepted_tokens[request]) - 1, 0),
                spec_query_len - 1,
            )
            shift = 1
            write_width = state_length + spec_query_len - 1
        for index in range(write_width):
            source = shift + index
            if mode == "spec" and index >= state_length + query_length - 1:
                continue
            if source < state_length:
                value = (
                    original_state[state_id, :, slot_offset + source]
                    if has_initial
                    else torch.zeros(inputs.shape[1], dtype=inputs.dtype)
                )
            else:
                value = inputs[query_start + source - state_length]
            state[state_id, :, index] = value
    return output, state


@pytest.mark.parametrize("mode", ["decode", "prefill", "spec"])
def test_ple_conv_qwen_shape_and_state_update(mode):
    torch.manual_seed(20260922)
    channels = 10240
    dilation = 3
    kernel_size = 4
    spec_query_len = 3
    if mode == "decode":
        query_start_loc = None
        lengths = [1, 1, 1]
    elif mode == "prefill":
        lengths = [5, 3]
        query_start_loc = torch.tensor([0, 5, 8], dtype=torch.int32)
    else:
        lengths = [3, 2]
        query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)
    tokens = sum(lengths)
    requests = len(lengths)
    state_length = (kernel_size - 1) * dilation
    state_width = state_length + (spec_query_len - 1 if mode == "spec" else 0)
    inputs = torch.randn(tokens, channels, dtype=torch.bfloat16)
    weights = torch.randn(channels, kernel_size, dtype=torch.bfloat16)
    state = torch.randn(requests + 1, channels, state_width).to(
        torch.bfloat16
    )
    state_indices = torch.arange(1, requests + 1, dtype=torch.int32)
    has_initial = torch.tensor(
        [True, False, True][:requests], dtype=torch.bool
    )
    accepted = (
        torch.tensor([2, 1], dtype=torch.int32)
        if mode == "spec"
        else None
    )
    expected_output, expected_state = _conv_reference(
        inputs,
        state.clone(),
        weights,
        state_indices,
        mode,
        dilation,
        query_start_loc,
        accepted,
        has_initial if mode != "spec" else None,
        spec_query_len,
    )
    actual_state = state.to(DEVICE)
    actual_output = ple_conv(
        inputs.to(DEVICE),
        actual_state,
        weights.to(DEVICE),
        state_indices.to(DEVICE),
        mode=mode,
        dilation=dilation,
        query_start_loc=(
            query_start_loc.to(DEVICE)
            if query_start_loc is not None
            else None
        ),
        num_accepted_tokens=(
            accepted.to(DEVICE) if accepted is not None else None
        ),
        has_initial_states=(
            has_initial.to(DEVICE) if mode != "spec" else None
        ),
        spec_query_len=spec_query_len,
    )
    torch.testing.assert_close(
        actual_output.cpu(),
        expected_output,
        atol=2**-7,
        rtol=2**-7,
    )
    assert torch.equal(actual_state.cpu(), expected_state)


def test_ple_conv_spec_honors_missing_initial_state():
    torch.manual_seed(17)
    channels = 257
    dilation = 2
    spec_query_len = 2
    inputs = torch.randn(2, channels, dtype=torch.bfloat16)
    weights = torch.randn(channels, 3, dtype=torch.bfloat16)
    state = torch.randn(2, channels, 5, dtype=torch.bfloat16)
    state_indices = torch.tensor([1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    accepted = torch.tensor([1], dtype=torch.int32)
    has_initial = torch.tensor([False])
    expected_output, expected_state = _conv_reference(
        inputs,
        state.clone(),
        weights,
        state_indices,
        "spec",
        dilation,
        query_start_loc,
        accepted,
        has_initial,
        spec_query_len,
    )
    actual_state = state.to(DEVICE)

    actual_output = ple_conv(
        inputs.to(DEVICE),
        actual_state,
        weights.to(DEVICE),
        state_indices.to(DEVICE),
        mode="spec",
        dilation=dilation,
        query_start_loc=query_start_loc.to(DEVICE),
        num_accepted_tokens=accepted.to(DEVICE),
        has_initial_states=has_initial.to(DEVICE),
        spec_query_len=spec_query_len,
    )

    torch.testing.assert_close(
        actual_output.cpu(),
        expected_output,
        atol=2**-7,
        rtol=2**-7,
    )
    assert torch.equal(actual_state.cpu(), expected_state)


def test_ple_conv_zeroes_unmapped_default_output_rows():
    channels = 257
    inputs = torch.ones(3, channels, dtype=torch.bfloat16, device=DEVICE)
    state = torch.zeros(2, channels, 1, dtype=torch.bfloat16, device=DEVICE)
    weights = torch.ones(channels, 2, dtype=torch.bfloat16, device=DEVICE)

    output = ple_conv(
        inputs,
        state,
        weights,
        torch.tensor([1], dtype=torch.int32, device=DEVICE),
        mode="decode",
        dilation=1,
        token_indices=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )

    assert torch.count_nonzero(output[[0, 2]]).item() == 0


def test_ple_conv_rejects_tokens_without_requests():
    with pytest.raises(
        RuntimeError,
        match="requires a request for every scheduled token",
    ):
        ple_conv(
            torch.ones(1, 4, dtype=torch.bfloat16, device=DEVICE),
            torch.zeros(1, 4, 1, dtype=torch.bfloat16, device=DEVICE),
            torch.ones(4, 2, dtype=torch.bfloat16, device=DEVICE),
            torch.empty(0, dtype=torch.int32, device=DEVICE),
            mode="prefill",
            dilation=1,
            query_start_loc=torch.tensor([0], dtype=torch.int32, device=DEVICE),
            has_initial_states=torch.empty(
                0,
                dtype=torch.bool,
                device=DEVICE,
            ),
        )


def test_ple_conv_rejects_aliased_tensors():
    inputs = torch.ones(2, 4, dtype=torch.bfloat16, device=DEVICE)
    state = torch.zeros(1, 4, 1, dtype=torch.bfloat16, device=DEVICE)
    weights = torch.ones(4, 2, dtype=torch.bfloat16, device=DEVICE)
    indices = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    kwargs = dict(
        mode="prefill",
        dilation=1,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=DEVICE),
        has_initial_states=torch.zeros(1, dtype=torch.bool, device=DEVICE),
    )
    with pytest.raises(RuntimeError, match="single memory location"):
        ple_conv(inputs, state, weights, indices, output=inputs, **kwargs)
    with pytest.raises(RuntimeError, match="single memory location"):
        ple_conv(
            state.view(1, 4),
            state,
            weights,
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            mode="decode",
            dilation=1,
        )


@pytest.mark.parametrize("state_id", [0, -1, 3])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_ple_conv_preserves_null_and_out_of_range_state_slots(state_id, mode):
    channels = 257
    state = torch.ones(2, channels, 1, dtype=torch.bfloat16, device=DEVICE)
    output = ple_conv(
        torch.ones(1, channels, dtype=torch.bfloat16, device=DEVICE),
        state,
        torch.ones(channels, 2, dtype=torch.bfloat16, device=DEVICE),
        torch.tensor([state_id], dtype=torch.int32, device=DEVICE),
        mode=mode,
        dilation=1,
        query_start_loc=(
            torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
            if mode == "prefill"
            else None
        ),
        has_initial_states=torch.tensor([True], device=DEVICE),
    )

    assert torch.count_nonzero(output).item() == 0
    assert torch.equal(state, torch.ones_like(state))


def test_ple_conv_zeroes_tokens_before_first_query_start():
    channels = 257
    output = ple_conv(
        torch.ones(2, channels, dtype=torch.bfloat16, device=DEVICE),
        torch.ones(2, channels, 1, dtype=torch.bfloat16, device=DEVICE),
        torch.ones(channels, 2, dtype=torch.bfloat16, device=DEVICE),
        torch.tensor([1], dtype=torch.int32, device=DEVICE),
        mode="prefill",
        dilation=1,
        query_start_loc=torch.tensor(
            [1, 2],
            dtype=torch.int32,
            device=DEVICE,
        ),
        has_initial_states=torch.tensor([True], device=DEVICE),
    )

    assert torch.count_nonzero(output[0]).item() == 0


def test_qsa_visibility_and_expansion_at_2048_budget():
    compress_ratio = 4
    token_topk = 2048
    query_positions = torch.tensor(
        [3, 4095, 8194, 23], dtype=torch.int64
    )
    sequence_lengths = torch.tensor([8195, 24], dtype=torch.int32)
    token_to_request = torch.tensor([0, 0, 0, 1], dtype=torch.int32)
    block_topk = token_topk // compress_ratio
    blocks = torch.arange(block_topk, dtype=torch.int32).repeat(4, 1)
    blocks[2, -3:] = -1
    # Blocks past the causal boundary (visible = 1024 for row 1, 6 for
    # row 3) must not be expanded even if they are inside the sequence.
    blocks[1, 0] = 1500
    blocks[3, 1] = 6

    expected_visible = torch.tensor([1, 1024, 2048, 6], dtype=torch.int32)
    actual_visible = qsa_visible_blocks(
        query_positions.to(DEVICE),
        sequence_lengths.to(DEVICE),
        token_to_request.to(DEVICE),
        compress_ratio,
    )
    assert torch.equal(actual_visible.cpu(), expected_visible)

    expected = torch.full(
        (4, token_topk + compress_ratio - 1),
        -1,
        dtype=torch.int32,
    )
    for row in range(4):
        request = int(token_to_request[row])
        sequence_length = int(sequence_lengths[request])
        query_position = int(query_positions[row])
        visible_blocks = min(
            (query_position + 1) // compress_ratio,
            sequence_length // compress_ratio,
        )
        complete_blocks = min(visible_blocks, block_topk)
        expanded_count = complete_blocks * compress_ratio
        for column in range(expanded_count):
            block = int(blocks[row, column // compress_ratio])
            token = block * compress_ratio + column % compress_ratio
            if 0 <= block < visible_blocks and token < sequence_length:
                expected[row, column] = token
        tail_start = (
            (query_position + 1) // compress_ratio * compress_ratio
        )
        tail_count = query_position + 1 - tail_start
        for offset in range(min(tail_count, compress_ratio - 1)):
            column = expanded_count + offset
            if column < expected.shape[1]:
                token = tail_start + offset
                if token < sequence_length:
                    expected[row, column] = token
    actual = qsa_expand_block_indices(
        blocks.to(DEVICE),
        query_positions.to(DEVICE),
        sequence_lengths.to(DEVICE),
        token_to_request.to(DEVICE),
        compress_ratio,
        token_topk,
    )
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float8_e4m3fn, torch.int64],
)
def test_qsa_store_cache_rows_preserves_invalid_slots(dtype):
    cache = torch.zeros(3, 8, 1, 7, dtype=dtype, device=DEVICE)
    rows = torch.arange(28, dtype=torch.int64).reshape(4, 1, 7).to(dtype)
    slots = torch.tensor([0, 9, -1, 23], dtype=torch.int64)
    expected = cache.cpu()
    for row, slot in enumerate(slots):
        if int(slot) >= 0:
            expected[int(slot) // 8, int(slot) % 8] = rows[row]
    qsa_store_cache_rows(cache, slots.to(DEVICE), rows.to(DEVICE))
    assert torch.equal(cache.cpu(), expected)


@pytest.mark.skipif(
    torch.xpu.device_count() < 2,
    reason="requires two XPU devices",
)
def test_qsa_metadata_rejects_mixed_devices():
    first, second = "xpu:0", "xpu:1"
    positions = torch.zeros(2, dtype=torch.int64, device=first)
    lengths = torch.ones(1, dtype=torch.int32, device=first)
    requests = torch.zeros(2, dtype=torch.int32, device=first)
    with pytest.raises(RuntimeError, match="token_to_request must be on"):
        qsa_visible_blocks(positions, lengths, requests.to(second), 4)
    with pytest.raises(RuntimeError, match="sequence_lengths must be on"):
        qsa_expand_block_indices(
            torch.zeros(2, 1, dtype=torch.int32, device=first),
            positions,
            lengths.to(second),
            requests,
            4,
            4,
        )
    cache = torch.zeros(1, 8, 1, 4, dtype=torch.bfloat16, device=first)
    with pytest.raises(RuntimeError, match="slot_mapping must be on"):
        qsa_store_cache_rows(
            cache,
            torch.zeros(1, dtype=torch.int64, device=second),
            torch.zeros(1, 4, dtype=torch.bfloat16, device=first),
        )


def _compress_reference(
    raw_keys,
    raw_positions,
    state_cache,
    state_table,
    token_to_request,
    query_start_loc,
    logical_positions,
    compressed_slots,
    compress_ratio,
    rope_cache,
):
    pooled = torch.zeros_like(raw_keys)
    first_positions = torch.zeros(
        raw_keys.shape[0], 3, dtype=torch.int64
    )
    for row in range(raw_keys.shape[0]):
        request = int(token_to_request[row])
        end_position = int(logical_positions[row])
        query_start = int(query_start_loc[request])
        chunk_start_position = end_position - (row - query_start)
        if end_position < compress_ratio - 1:
            continue
        if int(compressed_slots[row]) < 0:
            continue
        values = []
        for offset in range(compress_ratio):
            position = end_position - compress_ratio + 1 + offset
            if position >= chunk_start_position:
                raw_row = query_start + position - chunk_start_position
                values.append(raw_keys[raw_row, 0].float())
            else:
                block = int(state_table[request, 0])
                values.append(
                    state_cache[
                        block,
                        position % state_cache.shape[1],
                        0,
                    ].float()
                )
        pooled[row, 0] = torch.stack(values).mean(0).to(torch.bfloat16)
        first_position = end_position - compress_ratio + 1
        if rope_cache is None:
            first_positions[row].fill_(first_position)
        elif first_position >= chunk_start_position:
            raw_row = query_start + first_position - chunk_start_position
            first_positions[row] = raw_positions[raw_row, 0]
        else:
            block = int(state_table[request, 0])
            first_positions[row] = rope_cache[
                block,
                first_position % state_cache.shape[1],
                0,
            ]
    return pooled, first_positions


@pytest.mark.parametrize("with_rope_cache", [False, True])
def test_qsa_group_compression_crosses_state_ring(with_rope_cache):
    torch.manual_seed(20260923)
    rows = 7
    head_dim = 128
    state_size = 8
    compress_ratio = 4
    raw_keys = torch.randn(rows, 1, head_dim).to(torch.bfloat16)
    raw_positions = torch.arange(rows * 3, dtype=torch.int64).reshape(
        rows, 1, 3
    )
    state_cache = torch.randn(2, state_size, 1, head_dim).to(
        torch.bfloat16
    )
    rope_cache = torch.arange(
        2 * state_size * 3, dtype=torch.int64
    ).reshape(2, state_size, 1, 3)
    state_table = torch.tensor([[0], [1]], dtype=torch.int32)
    token_to_request = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1], dtype=torch.int32
    )
    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)
    logical_positions = torch.tensor(
        [6, 7, 8, 1, 2, 3, 4], dtype=torch.int64
    )
    compressed_slots = torch.tensor(
        [-1, -1, 2, -1, -1, 4, -1], dtype=torch.int64
    )
    selected_rope_cache = rope_cache if with_rope_cache else None
    expected_pooled, expected_positions = _compress_reference(
        raw_keys,
        raw_positions,
        state_cache,
        state_table,
        token_to_request,
        query_start_loc,
        logical_positions,
        compressed_slots,
        compress_ratio,
        selected_rope_cache,
    )
    actual_pooled, actual_positions = qsa_compress_groups(
        raw_keys.to(DEVICE),
        raw_positions.to(DEVICE),
        state_cache.to(DEVICE),
        state_table.to(DEVICE),
        token_to_request.to(DEVICE),
        query_start_loc.to(DEVICE),
        logical_positions.to(DEVICE),
        compressed_slots.to(DEVICE),
        compress_ratio,
        (
            selected_rope_cache.to(DEVICE)
            if selected_rope_cache is not None
            else None
        ),
    )
    assert torch.equal(actual_pooled.cpu(), expected_pooled)
    assert torch.equal(actual_positions.cpu(), expected_positions)
