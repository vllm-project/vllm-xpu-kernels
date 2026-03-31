#include "dispatch_utils.h"
#include "utils.h"
#include "xpu/ops.h"

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <sycl/sycl.hpp>

namespace {

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

}  // namespace

torch::Tensor ggml_dequantize(
    const torch::Tensor& W,
    int64_t type,
    int64_t m,
    int64_t n,
    std::optional<c10::ScalarType> out_dtype) {
  CHECK_DEVICE(W);
  CHECK_CONTIGUOUS(W);

  TORCH_CHECK(
      type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 ||
          type == GGML_TYPE_Q8_0,
      "XPU ggml_dequantize currently only supports Q4_0 (type=2), "
      "Q5_0 (type=6) and Q8_0 (type=8), got ",
      type);
  TORCH_CHECK(
      W.scalar_type() == at::ScalarType::Byte,
      "XPU ggml_dequantize expects uint8 weights, got ",
      W.scalar_type());
  TORCH_CHECK(m >= 0 && n >= 0, "m and n must be non-negative");

  const int64_t numel = m * n;
  TORCH_CHECK(
      numel % QK4_0 == 0,
      ggml_type_name(type),
      " dequantize expects m * n to be divisible by ",
      QK4_0,
      ", got ",
      numel);

  const int64_t expected_nbytes = get_expected_nbytes(type, numel);
  const int64_t weight_nbytes = W.numel() * W.element_size();
  TORCH_CHECK(
      weight_nbytes == expected_nbytes,
      ggml_type_name(type),
      " packed weight size mismatch: expected ",
      expected_nbytes,
      " bytes for shape (",
      m,
      ", ",
      n,
      "), got ",
      weight_nbytes,
      " bytes");

  const auto dtype = out_dtype.value_or(torch::kFloat16);
  TORCH_CHECK(
      dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
          dtype == torch::kFloat32,
      "XPU ggml_dequantize only supports fp16, bf16 or fp32 outputs, got ",
      dtype);

  auto options = torch::TensorOptions().dtype(dtype).device(W.device());
  auto output = torch::empty({m, n}, options);
  if (numel == 0) {
    return output;
  }

  at::DeviceGuard device_guard(W.device());
  auto& queue = vllm::xpu::vllmGetQueue(W.device().index());
  const auto* weight_ptr = W.data_ptr<uint8_t>();

  VLLM_DISPATCH_FLOATING_TYPES(output.scalar_type(), "ggml_dequantize", [&] {
    using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
    auto* out_ptr = reinterpret_cast<sycl_t*>(output.data_ptr<scalar_t>());

    switch (type) {
      case GGML_TYPE_Q4_0: {
        auto* blocks = reinterpret_cast<const block_q4_0*>(weight_ptr);
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::range<1>(static_cast<size_t>(numel)),
              ggml_dequantize_q4_0_kernel<scalar_t>(blocks, out_ptr, numel));
        });
        break;
      }
      case GGML_TYPE_Q5_0: {
        auto* blocks = reinterpret_cast<const block_q5_0*>(weight_ptr);
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::range<1>(static_cast<size_t>(numel)),
              ggml_dequantize_q5_0_kernel<scalar_t>(blocks, out_ptr, numel));
        });
        break;
      }
      case GGML_TYPE_Q8_0: {
        auto* blocks = reinterpret_cast<const block_q8_0*>(weight_ptr);
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::range<1>(static_cast<size_t>(numel)),
              ggml_dequantize_q8_0_kernel<scalar_t>(blocks, out_ptr, numel));
        });
        break;
      }
      default:
        TORCH_CHECK(
            false, "Unsupported GGML type for XPU ggml_dequantize: ", type);
    }
  });

  return output;
}