# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional

import torch

from . import _xpu_C  # noqa: F401


def ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    return torch.ops._xpu_C.qwen4_exp_ple_ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
        layer_multipliers,
        ngram_heads_vocab_sizes,
        ngram_heads_offsets,
        eos_token_id,
        heads_per_ngram,
    )


def ple_conv(
    inputs: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    mode: Literal["decode", "prefill", "spec"],
    dilation: int,
    query_start_loc: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    has_initial_states: Optional[torch.Tensor] = None,
    spec_query_len: int = 1,
    token_indices: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if output is None:
        output = (
            torch.zeros_like(inputs)
            if token_indices is not None
            else torch.empty_like(inputs)
        )
    torch.ops._xpu_C.qwen4_exp_ple_conv(
        inputs,
        output,
        conv_state,
        conv_weights,
        state_indices,
        mode,
        dilation,
        query_start_loc,
        num_accepted_tokens,
        has_initial_states,
        spec_query_len,
        token_indices,
    )
    return output


def qsa_visible_blocks(
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_request: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    return torch.ops._xpu_C.qwen4_exp_qsa_visible_blocks(
        query_positions,
        sequence_lengths,
        token_to_request,
        compress_ratio,
    )


def qsa_expand_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_request: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    return torch.ops._xpu_C.qwen4_exp_qsa_expand_block_indices(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_request,
        compress_ratio,
        token_topk,
    )


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    torch.ops._xpu_C.qwen4_exp_qsa_store_cache_rows(
        cache, slot_mapping, rows
    )


def qsa_compress_groups(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._xpu_C.qwen4_exp_qsa_compress_groups(
        raw_keys,
        raw_positions,
        compressor_state_cache,
        compressor_state_block_table,
        token_to_request,
        query_start_loc,
        logical_positions,
        compressed_slots,
        compress_ratio,
        rope_cache,
    )
