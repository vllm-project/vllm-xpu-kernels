#pragma once

#include "dispatch_utils.h"
#include "utils.h"

#include <cstdint>
#include <sycl/sycl.hpp>

namespace vllm {
namespace ggml {

constexpr int64_t GGML_TYPE_Q4_0 = 2;
constexpr int64_t GGML_TYPE_Q5_0 = 6;
constexpr int64_t GGML_TYPE_Q8_0 = 8;
constexpr int64_t QK4_0 = 32;
constexpr int64_t QK5_0 = 32;
constexpr int64_t QK8_0 = 32;

struct block_q4_0 {
  sycl::half d;
  uint8_t qs[QK4_0 / 2];
};

static_assert(sizeof(block_q4_0) == 18, "Unexpected Q4_0 block size");

struct block_q5_0 {
  sycl::half d;
  uint8_t qh[4];
  uint8_t qs[QK5_0 / 2];
};

static_assert(sizeof(block_q5_0) == 22, "Unexpected Q5_0 block size");

struct block_q8_0 {
  sycl::half d;
  int8_t qs[QK8_0];
};

static_assert(sizeof(block_q8_0) == 34, "Unexpected Q8_0 block size");

inline uint32_t load_u32_le(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

template <typename scalar_t>
class ggml_dequantize_q4_0_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q4_0_kernel(
      const block_q4_0* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK4_0;
    const int64_t block_offset = i % QK4_0;
    const block_q4_0& block = blocks_[block_index];
    const bool is_high_half = block_offset >= (QK4_0 / 2);
    const int64_t quant_index =
        is_high_half ? (block_offset - QK4_0 / 2) : block_offset;
    const uint8_t packed = block.qs[quant_index];
    const int quant = is_high_half ? (packed >> 4) : (packed & 0x0F);
    const float value =
        (static_cast<float>(quant) - 8.0f) * static_cast<float>(block.d);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q4_0* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

template <typename scalar_t>
class ggml_dequantize_q5_0_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q5_0_kernel(
      const block_q5_0* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK5_0;
    const int64_t block_offset = i % QK5_0;
    const block_q5_0& block = blocks_[block_index];
    const bool is_high_half = block_offset >= (QK5_0 / 2);
    const int64_t quant_index =
        is_high_half ? (block_offset - QK5_0 / 2) : block_offset;
    const uint8_t packed = block.qs[quant_index];
    const uint32_t qh = load_u32_le(block.qh);
    const int xh = is_high_half ? ((qh >> (quant_index + 12)) & 0x10)
                                : (((qh >> quant_index) << 4) & 0x10);
    const int base_quant = is_high_half ? (packed >> 4) : (packed & 0x0F);
    const float value = (static_cast<float>(base_quant | xh) - 16.0f) *
                        static_cast<float>(block.d);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q5_0* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

template <typename scalar_t>
class ggml_dequantize_q8_0_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q8_0_kernel(
      const block_q8_0* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK8_0;
    const int64_t block_offset = i % QK8_0;
    const block_q8_0& block = blocks_[block_index];
    const float value = static_cast<float>(block.qs[block_offset]) *
                        static_cast<float>(block.d);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q8_0* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

inline int64_t get_expected_nbytes(int64_t type, int64_t numel) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      return (numel / QK4_0) * static_cast<int64_t>(sizeof(block_q4_0));
    case GGML_TYPE_Q5_0:
      return (numel / QK5_0) * static_cast<int64_t>(sizeof(block_q5_0));
    case GGML_TYPE_Q8_0:
      return (numel / QK8_0) * static_cast<int64_t>(sizeof(block_q8_0));
    default:
      return -1;
  }
}

inline int64_t get_quant_block_size(int64_t type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      return QK4_0;
    case GGML_TYPE_Q5_0:
      return QK5_0;
    case GGML_TYPE_Q8_0:
      return QK8_0;
    default:
      return -1;
  }
}

inline const char* ggml_type_name(int64_t type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      return "Q4_0";
    case GGML_TYPE_Q5_0:
      return "Q5_0";
    case GGML_TYPE_Q8_0:
      return "Q8_0";
    default:
      return "unknown";
  }
}

}  // namespace ggml
}  // namespace vllm