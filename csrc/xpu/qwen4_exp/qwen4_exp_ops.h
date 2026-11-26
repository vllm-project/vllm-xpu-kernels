// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>

#include <optional>
#include <string>
#include <tuple>

torch::Tensor qwen4_exp_ple_ngram_ids(
    const torch::Tensor& input_ids,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& ngram_context,
    const torch::Tensor& layer_multipliers,
    const torch::Tensor& ngram_heads_vocab_sizes,
    const torch::Tensor& ngram_heads_offsets,
    int64_t eos_token_id,
    int64_t heads_per_ngram);

void qwen4_exp_ple_conv(
    const torch::Tensor& inputs,
    torch::Tensor& output,
    torch::Tensor& conv_state,
    const torch::Tensor& conv_weights,
    const torch::Tensor& state_indices,
    const std::string& mode,
    int64_t dilation,
    const std::optional<torch::Tensor>& query_start_loc,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& has_initial_states,
    int64_t spec_query_len,
    const std::optional<torch::Tensor>& token_indices);

torch::Tensor qwen4_exp_qsa_visible_blocks(
    const torch::Tensor& query_positions,
    const torch::Tensor& sequence_lengths,
    const torch::Tensor& token_to_request,
    int64_t compress_ratio);

torch::Tensor qwen4_exp_qsa_expand_block_indices(
    const torch::Tensor& block_indices,
    const torch::Tensor& query_positions,
    const torch::Tensor& sequence_lengths,
    const torch::Tensor& token_to_request,
    int64_t compress_ratio,
    int64_t token_topk);

void qwen4_exp_qsa_store_cache_rows(
    torch::Tensor& cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& rows);

std::tuple<torch::Tensor, torch::Tensor> qwen4_exp_qsa_compress_groups(
    const torch::Tensor& raw_keys,
    const torch::Tensor& raw_positions,
    const torch::Tensor& compressor_state_cache,
    const torch::Tensor& compressor_state_block_table,
    const torch::Tensor& token_to_request,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& logical_positions,
    const torch::Tensor& compressed_slots,
    int64_t compress_ratio,
    const std::optional<torch::Tensor>& rope_cache);
