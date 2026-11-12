#pragma once

#include <torch/all.h>

namespace gdn {

static constexpr float l2norm_eps = 0.000001f;

#ifdef VLLM_XPU_ENABLE_XE2
static constexpr int chunk_size_xe2 = 64;
#endif

enum class ActMode {
  silu = 0,
  swish = 1,
};

// Element strides of the conv-state cache. vLLM stores each slot either as
// SD [state_len, conv_elems] or, with VLLM_SSM_CONV_STATE_LAYOUT=DS, as
// DS [conv_elems, state_len]. Kernels address (row, elem) through these
// strides so both layouts read and write the same logical state.
struct ConvStateStrides {
  int slot;  // between cache slots
  int row;   // between state rows (time steps)
  int elem;  // between features

  // ElemContig = true assumes elem == 1 (SD), so the compiler sees a
  // compile-time feature stride and keeps adjacent-feature accesses
  // contiguous.
  template <bool ElemContig = false>
  inline int offset(int row_id, int elem_id) const {
    return row_id * row + elem_id * (ElemContig ? 1 : elem);
  }
};

// True for DS [cache_batch_size, conv_elems, state_len], false for SD
// [cache_batch_size, state_len, conv_elems]. The layout is told apart by
// which axis has conv_elems features, not by state_len, which vLLM pads to
// width - 1 + num_spec.
inline bool is_conv_state_dim_first(
    const torch::Tensor& conv_states,
    const int64_t conv_elems,
    const int64_t width) {
  TORCH_CHECK(
      conv_states.dim() == 3,
      "conv_state must be 3D [cache_batch_size, state_len, conv_elems] (SD) "
      "or [cache_batch_size, conv_elems, state_len] (DS)");
  if (conv_states.size(2) == conv_elems && conv_states.size(1) >= width - 1) {
    return false;
  }
  TORCH_CHECK(
      conv_states.size(1) == conv_elems && conv_states.size(2) >= width - 1,
      "conv_state shape ",
      conv_states.sizes(),
      " matches neither SD [cache_batch_size, state_len, ",
      conv_elems,
      "] nor DS [cache_batch_size, ",
      conv_elems,
      ", state_len] with state_len >= ",
      width - 1);
  return true;
}

inline int64_t get_conv_state_len(
    const torch::Tensor& conv_states,
    const int64_t conv_elems,
    const int64_t width) {
  return conv_states.size(
      is_conv_state_dim_first(conv_states, conv_elems, width) ? 2 : 1);
}

inline ConvStateStrides get_conv_state_strides(
    const torch::Tensor& conv_states,
    const int64_t conv_elems,
    const int64_t width) {
  const bool dim_first =
      is_conv_state_dim_first(conv_states, conv_elems, width);
  return {
      static_cast<int>(conv_states.stride(0)),
      static_cast<int>(conv_states.stride(dim_first ? 2 : 1)),
      static_cast<int>(conv_states.stride(dim_first ? 1 : 2))};
}

}  // namespace gdn