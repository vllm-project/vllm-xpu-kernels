// SPDX-License-Identifier: Apache-2.0

#include "xpu/qwen4_exp/qwen4_exp_ops.h"

#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int64_t kWorkGroupSize = 256;
// vLLM reserves slot zero for padded requests.
constexpr int64_t kNullStateId = 0;

int64_t round_up(int64_t value, int64_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

void check_xpu(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_xpu(), name, " must be on XPU");
}

void check_bf16_xpu(const torch::Tensor& tensor, const char* name) {
  check_xpu(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16");
}

void check_int32_xpu(const torch::Tensor& tensor, const char* name) {
  check_xpu(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
}

void check_int64_xpu(const torch::Tensor& tensor, const char* name) {
  check_xpu(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64");
}

void check_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(
      tensor.device() == reference.device(),
      name,
      " must be on ",
      reference.device());
}

class PleNgramIdsKernel {
 public:
  PleNgramIdsKernel(
      const int64_t* input_ids,
      const int32_t* query_start_loc,
      const int64_t* ngram_context,
      const int64_t* multipliers,
      const int64_t* vocab_sizes,
      const int64_t* offsets,
      int64_t* output,
      int64_t num_tokens,
      int64_t num_requests,
      int64_t context_length,
      int64_t heads_per_ngram,
      int64_t num_heads,
      int64_t eos_token_id)
      : input_ids_(input_ids),
        query_start_loc_(query_start_loc),
        ngram_context_(ngram_context),
        multipliers_(multipliers),
        vocab_sizes_(vocab_sizes),
        offsets_(offsets),
        output_(output),
        num_tokens_(num_tokens),
        num_requests_(num_requests),
        context_length_(context_length),
        heads_per_ngram_(heads_per_ngram),
        num_heads_(num_heads),
        eos_token_id_(eos_token_id) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int64_t linear = item.get_global_id(0);
    const int64_t elements = num_tokens_ * num_heads_;
    if (linear >= elements) {
      return;
    }
    const int64_t token = linear / num_heads_;
    const int64_t head = linear % num_heads_;

    int64_t low = 0;
    int64_t high = num_requests_;
    while (low + 1 < high) {
      const int64_t middle = (low + high) / 2;
      if (query_start_loc_[middle] <= token) {
        low = middle;
      } else {
        high = middle;
      }
    }
    const int64_t request = low;
    const int64_t chunk_position = token - query_start_loc_[request];
    if (query_start_loc_[0] != 0 ||
        query_start_loc_[num_requests_] != num_tokens_ || chunk_position < 0) {
      output_[linear] = 0;
      return;
    }
    const int64_t ngram_order = head / heads_per_ngram_ + 2;
    uint64_t mixed = static_cast<uint64_t>(input_ids_[token]) *
                     static_cast<uint64_t>(multipliers_[0]);
    bool crossed_eos = false;

    for (int64_t shift = 1; shift <= context_length_; ++shift) {
      int64_t candidate;
      if (chunk_position >= shift) {
        candidate = input_ids_[token - shift];
      } else {
        const int64_t context_column = context_length_ - shift + chunk_position;
        candidate = ngram_context_[request * context_length_ + context_column];
      }
      if (crossed_eos) {
        candidate = eos_token_id_;
      }
      crossed_eos = crossed_eos || candidate == eos_token_id_;
      if (ngram_order > shift) {
        mixed ^= static_cast<uint64_t>(candidate) *
                 static_cast<uint64_t>(multipliers_[shift]);
      }
    }

    const int64_t size = vocab_sizes_[head];
    if (size <= 0) {
      output_[linear] = 0;
      return;
    }
    const int64_t signed_mixed = sycl::bit_cast<int64_t>(mixed);
    int64_t remainder = signed_mixed % size;
    if (remainder < 0) {
      remainder += size;
    }
    output_[linear] = offsets_[head] + remainder;
  }

 private:
  const int64_t* input_ids_;
  const int32_t* query_start_loc_;
  const int64_t* ngram_context_;
  const int64_t* multipliers_;
  const int64_t* vocab_sizes_;
  const int64_t* offsets_;
  int64_t* output_;
  int64_t num_tokens_;
  int64_t num_requests_;
  int64_t context_length_;
  int64_t heads_per_ngram_;
  int64_t num_heads_;
  int64_t eos_token_id_;
};

class PleConvKernel {
 public:
  PleConvKernel(
      const bf16* inputs,
      bf16* output,
      bf16* state,
      const bf16* weights,
      const int32_t* state_indices,
      const int32_t* query_start_loc,
      const int32_t* num_accepted_tokens,
      const bool* has_initial_states,
      const int64_t* token_indices_i64,
      const int32_t* token_indices_i32,
      int64_t token_count,
      int64_t input_rows,
      int64_t channels,
      int64_t channel_groups,
      int64_t kernel_size,
      int64_t dilation,
      int64_t state_length,
      int64_t spec_query_len,
      int64_t num_requests,
      int64_t num_state_slots,
      int64_t token_indices_count,
      int64_t state_stride_slot,
      int64_t state_stride_channel,
      int64_t state_stride_window,
      int mode,
      bool has_initial_metadata,
      bool has_token_indices,
      bool token_indices_are_i64)
      : inputs_(inputs),
        output_(output),
        state_(state),
        weights_(weights),
        state_indices_(state_indices),
        query_start_loc_(query_start_loc),
        num_accepted_tokens_(num_accepted_tokens),
        has_initial_states_(has_initial_states),
        token_indices_i64_(token_indices_i64),
        token_indices_i32_(token_indices_i32),
        token_count_(token_count),
        input_rows_(input_rows),
        channels_(channels),
        channel_groups_(channel_groups),
        kernel_size_(kernel_size),
        dilation_(dilation),
        state_length_(state_length),
        spec_query_len_(spec_query_len),
        num_requests_(num_requests),
        num_state_slots_(num_state_slots),
        token_indices_count_(token_indices_count),
        state_stride_slot_(state_stride_slot),
        state_stride_channel_(state_stride_channel),
        state_stride_window_(state_stride_window),
        mode_(mode),
        has_initial_metadata_(has_initial_metadata),
        has_token_indices_(has_token_indices),
        token_indices_are_i64_(token_indices_are_i64) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int64_t group = item.get_group(0);
    const int64_t token = group / channel_groups_;
    const int64_t channel =
        (group % channel_groups_) * kWorkGroupSize + item.get_local_id(0);
    if (token >= token_count_ || channel >= channels_) {
      return;
    }

    const int64_t output_token = mapped_token(token);
    if (output_token < 0 || output_token >= input_rows_) {
      return;
    }

    int64_t request;
    int64_t query_start;
    int64_t position_in_query;
    int64_t slot_offset = 0;
    bool in_range = true;
    if (mode_ == 0) {
      request = token;
      query_start = token;
      position_in_query = 0;
    } else {
      request = find_request(token);
      query_start = query_start_loc_[request];
      position_in_query = token - query_start;
      in_range = token >= query_start_loc_[0] &&
                 token < query_start_loc_[num_requests_];
      if (mode_ == 2) {
        const int64_t accepted =
            static_cast<int64_t>(num_accepted_tokens_[request]) - 1;
        slot_offset = sycl::min<int64_t>(
            accepted > 0 ? accepted : 0, spec_query_len_ - 1);
      }
    }
    if (!in_range) {
      output_[output_token * channels_ + channel] = static_cast<bf16>(0.0f);
      return;
    }

    const int64_t state_id = state_indices_[request];
    const bool state_ok = state_id != kNullStateId && state_id >= 0 &&
                          state_id < num_state_slots_;
    const bool has_initial =
        has_initial_metadata_ ? has_initial_states_[request] : state_ok;
    const bool read_state = state_ok && has_initial;
    const bool output_ok = mode_ == 2 ? in_range : in_range && state_ok;

    float accumulator = 0.0f;
    for (int64_t tap = 0; tap < kernel_size_; ++tap) {
      const int64_t history = position_in_query + dilation_ * tap;
      const bool from_state = history <= state_length_ - 1;
      float value = 0.0f;
      if (from_state && read_state) {
        const int64_t state_offset =
            state_id * state_stride_slot_ +
            (history + slot_offset) * state_stride_window_ +
            channel * state_stride_channel_;
        value = static_cast<float>(state_[state_offset]);
      } else if (!from_state && output_ok) {
        const int64_t query_token = query_start + history - state_length_;
        const int64_t input_token = mapped_token(query_token);
        if (input_token >= 0 && input_token < input_rows_) {
          value =
              static_cast<float>(inputs_[input_token * channels_ + channel]);
        }
      }
      accumulator +=
          value * static_cast<float>(weights_[channel * kernel_size_ + tap]);
    }

    const bf16 rounded = static_cast<bf16>(accumulator);
    const float conv = static_cast<float>(rounded);
    output_[output_token * channels_ + channel] =
        output_ok ? static_cast<bf16>(conv / (1.0f + sycl::exp(-conv)))
                  : static_cast<bf16>(0.0f);

    if (mode_ == 0 && state_ok) {
      for (int64_t index = 0; index < state_length_; ++index) {
        bf16 value;
        if (index + 1 < state_length_ && has_initial) {
          value = state_
              [state_id * state_stride_slot_ +
               (index + 1) * state_stride_window_ +
               channel * state_stride_channel_];
        } else if (index + 1 < state_length_) {
          value = static_cast<bf16>(0.0f);
        } else {
          value = inputs_[output_token * channels_ + channel];
        }
        state_
            [state_id * state_stride_slot_ + index * state_stride_window_ +
             channel * state_stride_channel_] = value;
      }
    }
  }

 private:
  int64_t mapped_token(int64_t token) const {
    if (token < 0 || (has_token_indices_ && token >= token_indices_count_)) {
      return -1;
    }
    if (!has_token_indices_) {
      return token;
    }
    return token_indices_are_i64_
               ? token_indices_i64_[token]
               : static_cast<int64_t>(token_indices_i32_[token]);
  }

  int64_t find_request(int64_t token) const {
    int64_t low = 0;
    int64_t high = num_requests_;
    while (low + 1 < high) {
      const int64_t middle = (low + high) / 2;
      if (query_start_loc_[middle] <= token) {
        low = middle;
      } else {
        high = middle;
      }
    }
    return low;
  }

  const bf16* inputs_;
  bf16* output_;
  bf16* state_;
  const bf16* weights_;
  const int32_t* state_indices_;
  const int32_t* query_start_loc_;
  const int32_t* num_accepted_tokens_;
  const bool* has_initial_states_;
  const int64_t* token_indices_i64_;
  const int32_t* token_indices_i32_;
  int64_t token_count_;
  int64_t input_rows_;
  int64_t channels_;
  int64_t channel_groups_;
  int64_t kernel_size_;
  int64_t dilation_;
  int64_t state_length_;
  int64_t spec_query_len_;
  int64_t num_requests_;
  int64_t num_state_slots_;
  int64_t token_indices_count_;
  int64_t state_stride_slot_;
  int64_t state_stride_channel_;
  int64_t state_stride_window_;
  int mode_;
  bool has_initial_metadata_;
  bool has_token_indices_;
  bool token_indices_are_i64_;
};

class PleConvWritebackKernel {
 public:
  PleConvWritebackKernel(
      const bf16* inputs,
      bf16* state,
      const int32_t* state_indices,
      const int32_t* query_start_loc,
      const int32_t* num_accepted_tokens,
      const bool* has_initial_states,
      const int64_t* token_indices_i64,
      const int32_t* token_indices_i32,
      int64_t input_rows,
      int64_t channels,
      int64_t channel_groups,
      int64_t state_length,
      int64_t state_width,
      int64_t spec_query_len,
      int64_t num_requests,
      int64_t num_state_slots,
      int64_t token_indices_count,
      int64_t state_stride_slot,
      int64_t state_stride_channel,
      int64_t state_stride_window,
      int mode,
      bool has_initial_metadata,
      bool has_token_indices,
      bool token_indices_are_i64)
      : inputs_(inputs),
        state_(state),
        state_indices_(state_indices),
        query_start_loc_(query_start_loc),
        num_accepted_tokens_(num_accepted_tokens),
        has_initial_states_(has_initial_states),
        token_indices_i64_(token_indices_i64),
        token_indices_i32_(token_indices_i32),
        input_rows_(input_rows),
        channels_(channels),
        channel_groups_(channel_groups),
        state_length_(state_length),
        state_width_(state_width),
        spec_query_len_(spec_query_len),
        num_requests_(num_requests),
        num_state_slots_(num_state_slots),
        token_indices_count_(token_indices_count),
        state_stride_slot_(state_stride_slot),
        state_stride_channel_(state_stride_channel),
        state_stride_window_(state_stride_window),
        mode_(mode),
        has_initial_metadata_(has_initial_metadata),
        has_token_indices_(has_token_indices),
        token_indices_are_i64_(token_indices_are_i64) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int64_t group = item.get_group(0);
    const int64_t request = group / channel_groups_;
    const int64_t channel =
        (group % channel_groups_) * kWorkGroupSize + item.get_local_id(0);
    if (request >= num_requests_ || channel >= channels_) {
      return;
    }
    const int64_t state_id = state_indices_[request];
    if (state_id == kNullStateId || state_id < 0 ||
        state_id >= num_state_slots_) {
      return;
    }
    const int64_t query_start = query_start_loc_[request];
    const int64_t query_end = query_start_loc_[request + 1];
    const int64_t query_length = query_end - query_start;
    if (query_length <= 0) {
      return;
    }

    const int64_t accepted =
        mode_ == 2 ? static_cast<int64_t>(num_accepted_tokens_[request]) - 1
                   : 0;
    const int64_t slot_offset =
        mode_ == 2 ? sycl::min<int64_t>(
                         accepted > 0 ? accepted : 0, spec_query_len_ - 1)
                   : 0;
    const int64_t shift = mode_ == 2 ? 1 : query_length;
    const bool has_initial =
        has_initial_metadata_ ? has_initial_states_[request] : true;
    const int64_t write_width = mode_ == 2 ? state_width_ : state_length_;

    for (int64_t index = 0; index < write_width; ++index) {
      const int64_t source = shift + index;
      const bool from_state = source <= state_length_ - 1;
      const bool do_write =
          mode_ != 2 || index < state_length_ + query_length - 1;
      if (!do_write) {
        continue;
      }
      bf16 value = static_cast<bf16>(0.0f);
      if (from_state && has_initial) {
        value = state_
            [state_id * state_stride_slot_ +
             (slot_offset + source) * state_stride_window_ +
             channel * state_stride_channel_];
      } else if (!from_state) {
        const int64_t token = query_start + source - state_length_;
        const int64_t input_token = mapped_token(token);
        if (input_token >= 0 && input_token < input_rows_) {
          value = inputs_[input_token * channels_ + channel];
        }
      }
      state_
          [state_id * state_stride_slot_ + index * state_stride_window_ +
           channel * state_stride_channel_] = value;
    }
  }

 private:
  int64_t mapped_token(int64_t token) const {
    if (token < 0 || (has_token_indices_ && token >= token_indices_count_)) {
      return -1;
    }
    if (!has_token_indices_) {
      return token;
    }
    return token_indices_are_i64_
               ? token_indices_i64_[token]
               : static_cast<int64_t>(token_indices_i32_[token]);
  }

  const bf16* inputs_;
  bf16* state_;
  const int32_t* state_indices_;
  const int32_t* query_start_loc_;
  const int32_t* num_accepted_tokens_;
  const bool* has_initial_states_;
  const int64_t* token_indices_i64_;
  const int32_t* token_indices_i32_;
  int64_t input_rows_;
  int64_t channels_;
  int64_t channel_groups_;
  int64_t state_length_;
  int64_t state_width_;
  int64_t spec_query_len_;
  int64_t num_requests_;
  int64_t num_state_slots_;
  int64_t token_indices_count_;
  int64_t state_stride_slot_;
  int64_t state_stride_channel_;
  int64_t state_stride_window_;
  int mode_;
  bool has_initial_metadata_;
  bool has_token_indices_;
  bool token_indices_are_i64_;
};

class QsaVisibleBlocksKernel {
 public:
  QsaVisibleBlocksKernel(
      const int64_t* query_positions,
      const int32_t* sequence_lengths,
      const int32_t* token_to_request,
      int32_t* output,
      int64_t rows,
      int64_t num_requests,
      int64_t compress_ratio)
      : query_positions_(query_positions),
        sequence_lengths_(sequence_lengths),
        token_to_request_(token_to_request),
        output_(output),
        rows_(rows),
        num_requests_(num_requests),
        compress_ratio_(compress_ratio) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int64_t row = item.get_global_id(0);
    if (row >= rows_) {
      return;
    }
    const int32_t request = token_to_request_[row];
    if (request < 0 || request >= num_requests_) {
      output_[row] = 0;
      return;
    }
    const int64_t by_position = (query_positions_[row] + 1) / compress_ratio_;
    const int64_t by_length =
        static_cast<int64_t>(sequence_lengths_[request]) / compress_ratio_;
    const int64_t visible = sycl::min<int64_t>(by_position, by_length);
    output_[row] = static_cast<int32_t>(visible > 0 ? visible : 0);
  }

 private:
  const int64_t* query_positions_;
  const int32_t* sequence_lengths_;
  const int32_t* token_to_request_;
  int32_t* output_;
  int64_t rows_;
  int64_t num_requests_;
  int64_t compress_ratio_;
};

class QsaExpandBlockIndicesKernel {
 public:
  QsaExpandBlockIndicesKernel(
      const int32_t* block_indices,
      const int64_t* query_positions,
      const int32_t* sequence_lengths,
      const int32_t* token_to_request,
      int32_t* output,
      int64_t rows,
      int64_t num_requests,
      int64_t block_topk,
      int64_t compress_ratio,
      int64_t output_width)
      : block_indices_(block_indices),
        query_positions_(query_positions),
        sequence_lengths_(sequence_lengths),
        token_to_request_(token_to_request),
        output_(output),
        rows_(rows),
        num_requests_(num_requests),
        block_topk_(block_topk),
        compress_ratio_(compress_ratio),
        output_width_(output_width) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int64_t linear = item.get_global_id(0);
    const int64_t elements = rows_ * output_width_;
    if (linear >= elements) {
      return;
    }
    const int64_t row = linear / output_width_;
    const int64_t column = linear % output_width_;
    const int32_t request = token_to_request_[row];
    if (request < 0 || request >= num_requests_) {
      output_[linear] = -1;
      return;
    }
    const int64_t sequence_length = sequence_lengths_[request];
    const int64_t query_position = query_positions_[row];
    const int64_t complete_blocks = sycl::min<int64_t>(
        sycl::min<int64_t>(
            (query_position + 1) / compress_ratio_,
            sequence_length / compress_ratio_),
        block_topk_);
    const int64_t expanded_count = complete_blocks * compress_ratio_;
    const int64_t tail_start =
        ((query_position + 1) / compress_ratio_) * compress_ratio_;
    const int64_t tail_count = query_position + 1 - tail_start;

    int64_t token = -1;
    if (column < expanded_count) {
      const int64_t block_rank = column / compress_ratio_;
      const int64_t offset = column % compress_ratio_;
      const int32_t block = block_indices_[row * block_topk_ + block_rank];
      if (block >= 0) {
        token = static_cast<int64_t>(block) * compress_ratio_ + offset;
      }
    } else {
      const int64_t tail_offset = column - expanded_count;
      if (tail_offset < tail_count && tail_offset < compress_ratio_ - 1) {
        token = tail_start + tail_offset;
      }
    }
    output_[linear] = token >= 0 && token < sequence_length
                          ? static_cast<int32_t>(token)
                          : -1;
  }

 private:
  const int32_t* block_indices_;
  const int64_t* query_positions_;
  const int32_t* sequence_lengths_;
  const int32_t* token_to_request_;
  int32_t* output_;
  int64_t rows_;
  int64_t num_requests_;
  int64_t block_topk_;
  int64_t compress_ratio_;
  int64_t output_width_;
};

class QsaStoreCacheRowsKernel {
 public:
  QsaStoreCacheRowsKernel(
      uint8_t* cache,
      const int64_t* slots_i64,
      const int32_t* slots_i32,
      const uint8_t* rows,
      int64_t num_rows,
      int64_t cache_slots,
      int64_t row_bytes,
      bool slots_are_i64)
      : cache_(cache),
        slots_i64_(slots_i64),
        slots_i32_(slots_i32),
        rows_(rows),
        num_rows_(num_rows),
        cache_slots_(cache_slots),
        row_bytes_(row_bytes),
        slots_are_i64_(slots_are_i64) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int64_t linear = item.get_global_id(0);
    const int64_t bytes = num_rows_ * row_bytes_;
    if (linear >= bytes) {
      return;
    }
    const int64_t row = linear / row_bytes_;
    const int64_t byte = linear % row_bytes_;
    const int64_t slot = slots_are_i64_ ? slots_i64_[row]
                                        : static_cast<int64_t>(slots_i32_[row]);
    if (slot >= 0 && slot < cache_slots_) {
      cache_[slot * row_bytes_ + byte] = rows_[linear];
    }
  }

 private:
  uint8_t* cache_;
  const int64_t* slots_i64_;
  const int32_t* slots_i32_;
  const uint8_t* rows_;
  int64_t num_rows_;
  int64_t cache_slots_;
  int64_t row_bytes_;
  bool slots_are_i64_;
};

class QsaCompressGroupsKernel {
 public:
  QsaCompressGroupsKernel(
      const bf16* raw_keys,
      const int64_t* raw_positions,
      const bf16* state_cache,
      const int32_t* state_block_table,
      const int32_t* token_to_request,
      const int32_t* query_start_loc,
      const int64_t* logical_positions,
      const int64_t* compressed_slots,
      const int64_t* rope_cache,
      bf16* pooled,
      int64_t* first_positions,
      int64_t rows,
      int64_t head_dim,
      int64_t state_blocks,
      int64_t state_size,
      int64_t state_table_stride,
      int64_t num_requests,
      int64_t compress_ratio,
      int64_t row_groups,
      bool load_rope_positions)
      : raw_keys_(raw_keys),
        raw_positions_(raw_positions),
        state_cache_(state_cache),
        state_block_table_(state_block_table),
        token_to_request_(token_to_request),
        query_start_loc_(query_start_loc),
        logical_positions_(logical_positions),
        compressed_slots_(compressed_slots),
        rope_cache_(rope_cache),
        pooled_(pooled),
        first_positions_(first_positions),
        rows_(rows),
        head_dim_(head_dim),
        state_blocks_(state_blocks),
        state_size_(state_size),
        state_table_stride_(state_table_stride),
        num_requests_(num_requests),
        compress_ratio_(compress_ratio),
        row_groups_(row_groups),
        load_rope_positions_(load_rope_positions) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int64_t group = item.get_group(0);
    const int64_t row = group / row_groups_;
    const int64_t dimension =
        (group % row_groups_) * kWorkGroupSize + item.get_local_id(0);
    if (row >= rows_) {
      return;
    }
    const int32_t request = token_to_request_[row];
    const bool valid_request = request >= 0 && request < num_requests_;
    const int64_t query_start = valid_request ? query_start_loc_[request] : 0;
    const int64_t query_end = valid_request ? query_start_loc_[request + 1] : 0;
    const int64_t end_position = logical_positions_[row];
    const int64_t chunk_start_position = end_position - (row - query_start);
    const int32_t state_block =
        valid_request ? state_block_table_[request * state_table_stride_] : -1;
    const bool valid_state_block =
        state_block >= 0 && state_block < state_blocks_;
    const bool valid_row =
        valid_request && row >= query_start && row < query_end &&
        end_position >= compress_ratio_ - 1 && compressed_slots_[row] >= 0;

    if (dimension < head_dim_) {
      float accumulator = 0.0f;
      if (valid_row) {
        for (int64_t group_offset = 0; group_offset < compress_ratio_;
             ++group_offset) {
          const int64_t position =
              end_position - (compress_ratio_ - 1 - group_offset);
          const bool use_raw = position >= chunk_start_position;
          if (use_raw) {
            const int64_t raw_row =
                query_start + position - chunk_start_position;
            if (raw_row >= query_start && raw_row < query_end &&
                raw_row < rows_) {
              accumulator += static_cast<float>(
                  raw_keys_[raw_row * head_dim_ + dimension]);
            }
          } else if (valid_state_block) {
            const int64_t state_row =
                ((position % state_size_) + state_size_) % state_size_;
            accumulator += static_cast<float>(
                state_cache_
                    [(static_cast<int64_t>(state_block) * state_size_ +
                      state_row) *
                         head_dim_ +
                     dimension]);
          }
        }
      }
      pooled_[row * head_dim_ + dimension] =
          valid_row ? static_cast<bf16>(
                          accumulator / static_cast<float>(compress_ratio_))
                    : static_cast<bf16>(0.0f);
    }

    if (group % row_groups_ == 0 && item.get_local_id(0) < 3) {
      const int64_t axis = item.get_local_id(0);
      const int64_t first_position = end_position - compress_ratio_ + 1;
      int64_t value = 0;
      if (valid_row) {
        if (!load_rope_positions_) {
          value = first_position;
        } else if (first_position >= chunk_start_position) {
          const int64_t raw_row =
              query_start + first_position - chunk_start_position;
          if (raw_row >= query_start && raw_row < query_end &&
              raw_row < rows_) {
            value = raw_positions_[raw_row * 3 + axis];
          }
        } else if (valid_state_block) {
          const int64_t state_row =
              ((first_position % state_size_) + state_size_) % state_size_;
          value = rope_cache_
              [(static_cast<int64_t>(state_block) * state_size_ + state_row) *
                   3 +
               axis];
        }
      }
      first_positions_[row * 3 + axis] = value;
    }
  }

 private:
  const bf16* raw_keys_;
  const int64_t* raw_positions_;
  const bf16* state_cache_;
  const int32_t* state_block_table_;
  const int32_t* token_to_request_;
  const int32_t* query_start_loc_;
  const int64_t* logical_positions_;
  const int64_t* compressed_slots_;
  const int64_t* rope_cache_;
  bf16* pooled_;
  int64_t* first_positions_;
  int64_t rows_;
  int64_t head_dim_;
  int64_t state_blocks_;
  int64_t state_size_;
  int64_t state_table_stride_;
  int64_t num_requests_;
  int64_t compress_ratio_;
  int64_t row_groups_;
  bool load_rope_positions_;
};

}  // namespace

torch::Tensor qwen4_exp_ple_ngram_ids(
    const torch::Tensor& input_ids,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& ngram_context,
    const torch::Tensor& layer_multipliers,
    const torch::Tensor& ngram_heads_vocab_sizes,
    const torch::Tensor& ngram_heads_offsets,
    int64_t eos_token_id,
    int64_t heads_per_ngram) {
  check_int64_xpu(input_ids, "input_ids");
  check_int32_xpu(query_start_loc, "query_start_loc");
  check_int64_xpu(ngram_context, "ngram_context");
  check_int64_xpu(layer_multipliers, "layer_multipliers");
  check_int64_xpu(ngram_heads_vocab_sizes, "ngram_heads_vocab_sizes");
  check_int64_xpu(ngram_heads_offsets, "ngram_heads_offsets");
  check_same_device(input_ids, query_start_loc, "query_start_loc");
  check_same_device(input_ids, ngram_context, "ngram_context");
  check_same_device(input_ids, layer_multipliers, "layer_multipliers");
  check_same_device(
      input_ids, ngram_heads_vocab_sizes, "ngram_heads_vocab_sizes");
  check_same_device(input_ids, ngram_heads_offsets, "ngram_heads_offsets");
  TORCH_CHECK(
      input_ids.dim() == 1 && query_start_loc.dim() == 1 &&
          ngram_context.dim() == 2,
      "PLE IDs expect input_ids [tokens], query starts [requests + 1], "
      "and context [requests, context_length]");
  const int64_t num_requests = query_start_loc.numel() - 1;
  const int64_t context_length = ngram_context.size(1);
  const int64_t num_heads = context_length * heads_per_ngram;
  TORCH_CHECK(
      num_requests > 0 && ngram_context.size(0) == num_requests,
      "ngram_context request count must match query_start_loc");
  TORCH_CHECK(heads_per_ngram > 0, "heads_per_ngram must be positive");
  TORCH_CHECK(
      layer_multipliers.numel() >= context_length + 1,
      "layer_multipliers must cover every ngram term");
  TORCH_CHECK(
      ngram_heads_vocab_sizes.numel() == num_heads &&
          ngram_heads_offsets.numel() == num_heads,
      "PLE vocabulary sizes and offsets must match ngram head count");
  TORCH_CHECK(
      input_ids.is_contiguous() && query_start_loc.is_contiguous() &&
          ngram_context.is_contiguous() && layer_multipliers.is_contiguous() &&
          ngram_heads_vocab_sizes.is_contiguous() &&
          ngram_heads_offsets.is_contiguous(),
      "PLE ID inputs must be contiguous");
  auto output =
      torch::empty({input_ids.numel(), num_heads}, input_ids.options());
  if (output.numel() == 0) {
    return output;
  }
  at::DeviceGuard guard(input_ids.device());
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const int64_t global = round_up(output.numel(), kWorkGroupSize);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global), sycl::range<1>(kWorkGroupSize)),
        PleNgramIdsKernel(
            input_ids.data_ptr<int64_t>(),
            query_start_loc.data_ptr<int32_t>(),
            ngram_context.data_ptr<int64_t>(),
            layer_multipliers.data_ptr<int64_t>(),
            ngram_heads_vocab_sizes.data_ptr<int64_t>(),
            ngram_heads_offsets.data_ptr<int64_t>(),
            output.data_ptr<int64_t>(),
            input_ids.numel(),
            num_requests,
            context_length,
            heads_per_ngram,
            num_heads,
            eos_token_id));
  });
  return output;
}

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
    const std::optional<torch::Tensor>& token_indices) {
  check_bf16_xpu(inputs, "inputs");
  check_bf16_xpu(output, "output");
  check_bf16_xpu(conv_state, "conv_state");
  check_bf16_xpu(conv_weights, "conv_weights");
  check_int32_xpu(state_indices, "state_indices");
  check_same_device(inputs, output, "output");
  check_same_device(inputs, conv_state, "conv_state");
  check_same_device(inputs, conv_weights, "conv_weights");
  check_same_device(inputs, state_indices, "state_indices");
  TORCH_CHECK(
      inputs.dim() == 2 && output.sizes() == inputs.sizes(),
      "PLE convolution input and output must have matching [tokens, C] "
      "shapes");
  TORCH_CHECK(
      conv_state.dim() == 3, "conv_state must be [slots, C, state_width]");
  TORCH_CHECK(
      conv_weights.dim() == 2 && conv_weights.size(0) == inputs.size(1) &&
          conv_weights.size(1) > 0,
      "conv_weights must be [C, kernel_size]");
  TORCH_CHECK(
      inputs.is_contiguous() && output.is_contiguous() &&
          conv_weights.is_contiguous() && state_indices.is_contiguous(),
      "PLE convolution input, output, weights, and indices must be "
      "contiguous");
  // The kernels address conv_state purely through its strides, so either the
  // window (ds, stride(2)==1) or the channel (sd, stride(1)==1) axis may be
  // innermost.
  TORCH_CHECK(
      conv_state.size(1) == inputs.size(1) &&
          (conv_state.stride(1) == 1 || conv_state.stride(2) == 1),
      "conv_state channels must match inputs and either the channel or the "
      "window axis must be contiguous");
  TORCH_CHECK(dilation > 0, "dilation must be positive");
  TORCH_CHECK(spec_query_len > 0, "spec_query_len must be positive");
  const int64_t state_length = (conv_weights.size(1) - 1) * dilation;
  const int64_t state_width =
      state_length + (mode == "spec" ? spec_query_len - 1 : 0);
  TORCH_CHECK(
      conv_state.size(2) >= state_width,
      "conv_state window is too short for this convolution mode");

  int mode_code;
  int64_t token_count = inputs.size(0);
  int64_t num_requests;
  if (mode == "decode") {
    mode_code = 0;
    num_requests =
        token_indices.has_value() ? token_indices->numel() : inputs.size(0);
    token_count = num_requests;
  } else if (mode == "prefill" || mode == "spec") {
    mode_code = mode == "prefill" ? 1 : 2;
    TORCH_CHECK(
        query_start_loc.has_value(),
        "query_start_loc is required for prefill/spec");
    check_int32_xpu(*query_start_loc, "query_start_loc");
    check_same_device(inputs, *query_start_loc, "query_start_loc");
    TORCH_CHECK(
        query_start_loc->dim() == 1 && query_start_loc->is_contiguous(),
        "query_start_loc must be contiguous [requests + 1]");
    num_requests = query_start_loc->numel() - 1;
    token_count =
        token_indices.has_value() ? token_indices->numel() : inputs.size(0);
    if (mode_code == 2) {
      TORCH_CHECK(
          num_accepted_tokens.has_value(),
          "num_accepted_tokens is required for spec");
    } else {
      TORCH_CHECK(
          has_initial_states.has_value(),
          "has_initial_states is required for prefill");
    }
  } else {
    TORCH_CHECK(false, "mode must be decode, prefill, or spec");
  }
  TORCH_CHECK(
      state_indices.dim() == 1 && state_indices.numel() == num_requests,
      "state_indices must contain one entry per request");
  TORCH_CHECK(
      token_count == 0 || num_requests > 0,
      "PLE convolution requires a request for every scheduled token");

  const int32_t* accepted_ptr = nullptr;
  if (num_accepted_tokens.has_value()) {
    check_int32_xpu(*num_accepted_tokens, "num_accepted_tokens");
    check_same_device(inputs, *num_accepted_tokens, "num_accepted_tokens");
    TORCH_CHECK(
        num_accepted_tokens->dim() == 1 &&
            num_accepted_tokens->numel() == num_requests &&
            num_accepted_tokens->is_contiguous(),
        "num_accepted_tokens must contain one contiguous entry per request");
    accepted_ptr = num_accepted_tokens->data_ptr<int32_t>();
  }
  const bool* initial_ptr = nullptr;
  if (has_initial_states.has_value()) {
    check_xpu(*has_initial_states, "has_initial_states");
    check_same_device(inputs, *has_initial_states, "has_initial_states");
    TORCH_CHECK(
        has_initial_states->scalar_type() == at::kBool &&
            has_initial_states->dim() == 1 &&
            has_initial_states->numel() == num_requests &&
            has_initial_states->is_contiguous(),
        "has_initial_states must contain one contiguous bool per request");
    initial_ptr = has_initial_states->data_ptr<bool>();
  }
  const int64_t* token_indices_i64 = nullptr;
  const int32_t* token_indices_i32 = nullptr;
  bool token_indices_are_i64 = false;
  if (token_indices.has_value()) {
    check_xpu(*token_indices, "token_indices");
    check_same_device(inputs, *token_indices, "token_indices");
    TORCH_CHECK(
        token_indices->dim() == 1 && token_indices->is_contiguous(),
        "token_indices must be contiguous and one-dimensional");
    TORCH_CHECK(
        token_indices->scalar_type() == at::kLong ||
            token_indices->scalar_type() == at::kInt,
        "token_indices must be int32 or int64");
    token_indices_are_i64 = token_indices->scalar_type() == at::kLong;
    if (token_indices_are_i64) {
      token_indices_i64 = token_indices->data_ptr<int64_t>();
    } else {
      token_indices_i32 = token_indices->data_ptr<int32_t>();
    }
  }
  if (token_count == 0) {
    return;
  }

  const int64_t channels = inputs.size(1);
  const int64_t channel_groups =
      (channels + kWorkGroupSize - 1) / kWorkGroupSize;
  const int64_t token_indices_count =
      token_indices.has_value() ? token_indices->numel() : inputs.size(0);
  at::DeviceGuard guard(inputs.device());
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(token_count * channel_groups * kWorkGroupSize),
            sycl::range<1>(kWorkGroupSize)),
        PleConvKernel(
            reinterpret_cast<const bf16*>(inputs.data_ptr()),
            reinterpret_cast<bf16*>(output.data_ptr()),
            reinterpret_cast<bf16*>(conv_state.data_ptr()),
            reinterpret_cast<const bf16*>(conv_weights.data_ptr()),
            state_indices.data_ptr<int32_t>(),
            query_start_loc.has_value() ? query_start_loc->data_ptr<int32_t>()
                                        : nullptr,
            accepted_ptr,
            initial_ptr,
            token_indices_i64,
            token_indices_i32,
            token_count,
            inputs.size(0),
            channels,
            channel_groups,
            conv_weights.size(1),
            dilation,
            state_length,
            spec_query_len,
            num_requests,
            conv_state.size(0),
            token_indices_count,
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            mode_code,
            has_initial_states.has_value(),
            token_indices.has_value(),
            token_indices_are_i64));
  });
  if (mode_code != 0) {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(num_requests * channel_groups * kWorkGroupSize),
              sycl::range<1>(kWorkGroupSize)),
          PleConvWritebackKernel(
              reinterpret_cast<const bf16*>(inputs.data_ptr()),
              reinterpret_cast<bf16*>(conv_state.data_ptr()),
              state_indices.data_ptr<int32_t>(),
              query_start_loc->data_ptr<int32_t>(),
              accepted_ptr,
              initial_ptr,
              token_indices_i64,
              token_indices_i32,
              inputs.size(0),
              channels,
              channel_groups,
              state_length,
              state_width,
              spec_query_len,
              num_requests,
              conv_state.size(0),
              token_indices_count,
              conv_state.stride(0),
              conv_state.stride(1),
              conv_state.stride(2),
              mode_code,
              has_initial_states.has_value(),
              token_indices.has_value(),
              token_indices_are_i64));
    });
  }
}

torch::Tensor qwen4_exp_qsa_visible_blocks(
    const torch::Tensor& query_positions,
    const torch::Tensor& sequence_lengths,
    const torch::Tensor& token_to_request,
    int64_t compress_ratio) {
  check_int64_xpu(query_positions, "query_positions");
  check_int32_xpu(sequence_lengths, "sequence_lengths");
  check_int32_xpu(token_to_request, "token_to_request");
  check_same_device(query_positions, sequence_lengths, "sequence_lengths");
  check_same_device(query_positions, token_to_request, "token_to_request");
  TORCH_CHECK(
      query_positions.dim() == 1 &&
          token_to_request.sizes() == query_positions.sizes() &&
          sequence_lengths.dim() == 1,
      "QSA visibility metadata has invalid shapes");
  TORCH_CHECK(
      query_positions.is_contiguous() && sequence_lengths.is_contiguous() &&
          token_to_request.is_contiguous(),
      "QSA visibility metadata must be contiguous");
  TORCH_CHECK(compress_ratio > 0, "compress_ratio must be positive");
  auto output = torch::empty(
      query_positions.sizes(), query_positions.options().dtype(torch::kInt32));
  if (output.numel() == 0) {
    return output;
  }
  at::DeviceGuard guard(query_positions.device());
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const int64_t global = round_up(output.numel(), kWorkGroupSize);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global), sycl::range<1>(kWorkGroupSize)),
        QsaVisibleBlocksKernel(
            query_positions.data_ptr<int64_t>(),
            sequence_lengths.data_ptr<int32_t>(),
            token_to_request.data_ptr<int32_t>(),
            output.data_ptr<int32_t>(),
            output.numel(),
            sequence_lengths.numel(),
            compress_ratio));
  });
  return output;
}

torch::Tensor qwen4_exp_qsa_expand_block_indices(
    const torch::Tensor& block_indices,
    const torch::Tensor& query_positions,
    const torch::Tensor& sequence_lengths,
    const torch::Tensor& token_to_request,
    int64_t compress_ratio,
    int64_t token_topk) {
  check_int32_xpu(block_indices, "block_indices");
  check_int64_xpu(query_positions, "query_positions");
  check_int32_xpu(sequence_lengths, "sequence_lengths");
  check_int32_xpu(token_to_request, "token_to_request");
  check_same_device(block_indices, query_positions, "query_positions");
  check_same_device(block_indices, sequence_lengths, "sequence_lengths");
  check_same_device(block_indices, token_to_request, "token_to_request");
  TORCH_CHECK(
      block_indices.dim() == 2 && query_positions.dim() == 1 &&
          block_indices.size(0) == query_positions.numel() &&
          token_to_request.sizes() == query_positions.sizes(),
      "QSA expansion metadata has invalid shapes");
  TORCH_CHECK(
      block_indices.is_contiguous() && query_positions.is_contiguous() &&
          sequence_lengths.is_contiguous() && token_to_request.is_contiguous(),
      "QSA expansion inputs must be contiguous");
  TORCH_CHECK(
      compress_ratio > 0 && token_topk > 0 && token_topk % compress_ratio == 0,
      "token_topk must be positive and divisible by compress_ratio");
  const int64_t block_topk = token_topk / compress_ratio;
  TORCH_CHECK(
      block_indices.size(1) == block_topk,
      "block_indices width must be token_topk / compress_ratio");
  const int64_t output_width = token_topk + compress_ratio - 1;
  auto output = torch::empty(
      {block_indices.size(0), output_width}, block_indices.options());
  if (output.numel() == 0) {
    return output;
  }
  at::DeviceGuard guard(block_indices.device());
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const int64_t global = round_up(output.numel(), kWorkGroupSize);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global), sycl::range<1>(kWorkGroupSize)),
        QsaExpandBlockIndicesKernel(
            block_indices.data_ptr<int32_t>(),
            query_positions.data_ptr<int64_t>(),
            sequence_lengths.data_ptr<int32_t>(),
            token_to_request.data_ptr<int32_t>(),
            output.data_ptr<int32_t>(),
            block_indices.size(0),
            sequence_lengths.numel(),
            block_topk,
            compress_ratio,
            output_width));
  });
  return output;
}

void qwen4_exp_qsa_store_cache_rows(
    torch::Tensor& cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& rows) {
  check_xpu(cache, "cache");
  check_xpu(slot_mapping, "slot_mapping");
  check_xpu(rows, "rows");
  check_same_device(cache, slot_mapping, "slot_mapping");
  check_same_device(cache, rows, "rows");
  TORCH_CHECK(
      slot_mapping.scalar_type() == at::kLong ||
          slot_mapping.scalar_type() == at::kInt,
      "slot_mapping must be int32 or int64");
  TORCH_CHECK(
      cache.dim() == 4 && cache.size(2) == 1,
      "cache must be [blocks, page_size, 1, width]");
  TORCH_CHECK(
      rows.scalar_type() == cache.scalar_type(),
      "cache rows and cache must have the same dtype");
  const int64_t num_rows = slot_mapping.numel();
  const int64_t width = cache.size(3);
  const bool rows_are_3d = rows.dim() == 3;
  TORCH_CHECK(
      (rows.dim() == 2 && rows.size(0) == num_rows && rows.size(1) == width) ||
          (rows_are_3d && rows.size(0) == num_rows && rows.size(1) == 1 &&
           rows.size(2) == width),
      "rows must be [rows, width] or [rows, 1, width]");
  TORCH_CHECK(
      cache.is_contiguous() && rows.is_contiguous() &&
          slot_mapping.is_contiguous(),
      "QSA cache store inputs must be contiguous");
  if (num_rows == 0) {
    return;
  }
  const int64_t row_bytes = width * cache.element_size();
  const int64_t bytes = num_rows * row_bytes;
  at::DeviceGuard guard(cache.device());
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const int64_t global = round_up(bytes, kWorkGroupSize);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global), sycl::range<1>(kWorkGroupSize)),
        QsaStoreCacheRowsKernel(
            static_cast<uint8_t*>(cache.data_ptr()),
            slot_mapping.scalar_type() == at::kLong
                ? slot_mapping.data_ptr<int64_t>()
                : nullptr,
            slot_mapping.scalar_type() == at::kInt
                ? slot_mapping.data_ptr<int32_t>()
                : nullptr,
            static_cast<const uint8_t*>(rows.data_ptr()),
            num_rows,
            cache.size(0) * cache.size(1),
            row_bytes,
            slot_mapping.scalar_type() == at::kLong));
  });
}

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
    const std::optional<torch::Tensor>& rope_cache) {
  check_bf16_xpu(raw_keys, "raw_keys");
  check_int64_xpu(raw_positions, "raw_positions");
  check_bf16_xpu(compressor_state_cache, "compressor_state_cache");
  check_int32_xpu(compressor_state_block_table, "compressor_state_block_table");
  check_int32_xpu(token_to_request, "token_to_request");
  check_int32_xpu(query_start_loc, "query_start_loc");
  check_int64_xpu(logical_positions, "logical_positions");
  check_int64_xpu(compressed_slots, "compressed_slots");
  check_same_device(raw_keys, raw_positions, "raw_positions");
  check_same_device(raw_keys, compressor_state_cache, "compressor_state_cache");
  check_same_device(
      raw_keys, compressor_state_block_table, "compressor_state_block_table");
  check_same_device(raw_keys, token_to_request, "token_to_request");
  check_same_device(raw_keys, query_start_loc, "query_start_loc");
  check_same_device(raw_keys, logical_positions, "logical_positions");
  check_same_device(raw_keys, compressed_slots, "compressed_slots");
  const int64_t rows = raw_keys.size(0);
  TORCH_CHECK(
      raw_keys.dim() == 3 && raw_keys.size(1) == 1,
      "raw_keys must be [rows, 1, head_dim]");
  TORCH_CHECK(
      raw_positions.sizes() == at::IntArrayRef({rows, 1, 3}),
      "raw_positions must be [rows, 1, 3]");
  TORCH_CHECK(
      compressor_state_cache.dim() == 4 &&
          compressor_state_cache.size(2) == 1 &&
          compressor_state_cache.size(3) == raw_keys.size(2),
      "compressor state must be [blocks, state_size, 1, head_dim]");
  TORCH_CHECK(
      compressor_state_block_table.dim() == 2 &&
          compressor_state_block_table.size(1) >= 1,
      "compressor state block table must be [requests, >=1]");
  TORCH_CHECK(
      token_to_request.sizes() == at::IntArrayRef({rows}) &&
          logical_positions.sizes() == at::IntArrayRef({rows}) &&
          compressed_slots.sizes() == at::IntArrayRef({rows}),
      "QSA compression metadata must match raw key rows");
  TORCH_CHECK(
      query_start_loc.dim() == 1 && query_start_loc.numel() >= 2,
      "query_start_loc must contain request starts and a terminal offset");
  const int64_t num_requests = query_start_loc.numel() - 1;
  TORCH_CHECK(
      compressor_state_block_table.size(0) >= num_requests,
      "compressor state block table has too few requests");
  TORCH_CHECK(compress_ratio > 0, "compress_ratio must be positive");
  TORCH_CHECK(
      compressor_state_cache.size(1) >= compress_ratio,
      "compressor state ring must hold at least one compression group");
  TORCH_CHECK(
      raw_keys.is_contiguous() && raw_positions.is_contiguous() &&
          compressor_state_cache.is_contiguous() &&
          compressor_state_block_table.is_contiguous() &&
          token_to_request.is_contiguous() && query_start_loc.is_contiguous() &&
          logical_positions.is_contiguous() && compressed_slots.is_contiguous(),
      "QSA compression inputs must be contiguous");
  const int64_t* rope_ptr = nullptr;
  if (rope_cache.has_value()) {
    check_int64_xpu(*rope_cache, "rope_cache");
    check_same_device(raw_keys, *rope_cache, "rope_cache");
    TORCH_CHECK(
        rope_cache->sizes() == at::IntArrayRef(
                                   {compressor_state_cache.size(0),
                                    compressor_state_cache.size(1),
                                    1,
                                    3}) &&
            rope_cache->is_contiguous(),
        "rope_cache must be contiguous [blocks, state_size, 1, 3]");
    rope_ptr = rope_cache->data_ptr<int64_t>();
  }
  auto pooled = torch::empty_like(raw_keys);
  auto first_positions = torch::empty({rows, 3}, raw_positions.options());
  if (rows == 0) {
    return {pooled, first_positions};
  }
  const int64_t head_dim = raw_keys.size(2);
  const int64_t row_groups = (head_dim + kWorkGroupSize - 1) / kWorkGroupSize;
  at::DeviceGuard guard(raw_keys.device());
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(rows * row_groups * kWorkGroupSize),
            sycl::range<1>(kWorkGroupSize)),
        QsaCompressGroupsKernel(
            reinterpret_cast<const bf16*>(raw_keys.data_ptr()),
            raw_positions.data_ptr<int64_t>(),
            reinterpret_cast<const bf16*>(compressor_state_cache.data_ptr()),
            compressor_state_block_table.data_ptr<int32_t>(),
            token_to_request.data_ptr<int32_t>(),
            query_start_loc.data_ptr<int32_t>(),
            logical_positions.data_ptr<int64_t>(),
            compressed_slots.data_ptr<int64_t>(),
            rope_ptr,
            reinterpret_cast<bf16*>(pooled.data_ptr()),
            first_positions.data_ptr<int64_t>(),
            rows,
            head_dim,
            compressor_state_cache.size(0),
            compressor_state_cache.size(1),
            compressor_state_block_table.stride(0),
            num_requests,
            compress_ratio,
            row_groups,
            rope_cache.has_value()));
  });
  return {pooled, first_positions};
}
