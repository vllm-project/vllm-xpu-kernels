// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <cstdint>

#ifdef MINICPM5_MONO_FP16
  #define MINICPM5_MONO_NAMESPACE mono_fp16
#else
  #define MINICPM5_MONO_NAMESPACE mono
#endif

namespace vllm::minicpm5::MINICPM5_MONO_NAMESPACE {
#ifdef MINICPM5_MONO_FP16
using scalar_t = sycl::half;
#else
using scalar_t = sycl::ext::oneapi::bfloat16;
#endif
using Float8 = float __attribute__((ext_vector_type(8)));
using UInt8 = unsigned int __attribute__((ext_vector_type(8)));
using UInt4 = unsigned int __attribute__((ext_vector_type(4)));
using UInt2 = unsigned int __attribute__((ext_vector_type(2)));
using Int2 = int __attribute__((ext_vector_type(2)));

inline uint16_t fp8_bits(uint8_t value) {
  const uint16_t sign = (value & 128) << 8;
  const uint16_t magnitude = value & 127;
  if (magnitude == 127) return sign | 0x7fc0;
#ifdef MINICPM5_MONO_FAST_FP8_ENABLED
  // The seven finite subnormals need only fixed shifts; normal values retain
  // the exponent rebias used by the scalar reference.
  const uint16_t subnormal =
      magnitude < 2 ? magnitude * 0x3b00u
                    : (magnitude < 4 ? 0x3b80u + ((magnitude - 2u) << 6)
                                     : 0x3c00u + ((magnitude - 4u) << 5));
#else
  // An indexed constexpr table is copied to private memory by this compiler.
  // Normalize the three-bit subnormal mantissa entirely in registers.
  const uint32_t highest = 31u - sycl::clz(uint32_t(magnitude) | 1u);
  const uint16_t subnormal = ((118u + highest) << 7) |
                             ((uint32_t(magnitude) << (7u - highest)) & 127u);
#endif
  const uint16_t normal = (magnitude << 4) + 0x3c00;
  return sign | (magnitude == 0 ? 0 : (magnitude < 8 ? subnormal : normal));
}

constexpr int operand_row(int lane, int word) { return 2 * word + lane / 8; }

constexpr int operand_column(int lane) { return 2 * (lane % 8); }

inline uint32_t pack16(scalar_t a, scalar_t b) {
  return uint32_t(sycl::bit_cast<uint16_t>(a)) |
         (uint32_t(sycl::bit_cast<uint16_t>(b)) << 16);
}

// E4M3 maps exactly to FP16/256, including signed zeros and subnormals.
inline uint32_t fp8_half_lanes(uint32_t lanes) {
  const uint32_t magnitude = lanes & 0x007f007fu;
  const uint32_t nan = (magnitude + 0x00010001u) & 0x00800080u;
  // Only E4M3 magnitude127 needs correction: 0x3f80 xor 0x4180=0x7e00.
  // Each half's sign is disjoint from the correction; all other codes keep
  // the exact scaled-half expansion, including zero and subnormals.
  return (((lanes & 0x00800080u) << 8) | (magnitude << 7)) ^
         (nan | (nan << 1) | (nan << 7));
}

inline uint32_t fp8_pair_half(uint32_t bytes) {
  return fp8_half_lanes((bytes & 255u) | ((bytes & 0xff00u) << 8));
}

inline uint32_t half_pair_to_bf16(uint32_t pair) {
  const auto halves = sycl::bit_cast<sycl::vec<sycl::half, 2>>(pair);
  const auto bits =
      (halves.convert<float>() * 256.f).as<sycl::vec<uint32_t, 2>>();
  return (bits[0] >> 16) | (bits[1] & 0xffff0000u);
}

inline uint32_t fp8_pair_bf16(uint32_t bytes) {
#ifdef MINICPM5_MONO_BF16_FP8_HALF_ENABLED
  // Weight conversion alone uses FP16; BF16 activations retain their range.
  return half_pair_to_bf16(fp8_pair_half(bytes));
#elif defined(MINICPM5_MONO_FAST_FP8_ENABLED)
  // Vector masks keep both exceptional bytes in straight-line integer code.
  // Scalar early returns/branches inflate every unrolled DPAS tile and stop
  // the reference register pipelines from being fully scalarized by IGC.
  const UInt2 raw{bytes & 255u, (bytes >> 8) & 255u};
  const UInt2 magnitude = raw & 127u;
  const auto pick = [](UInt2 a, UInt2 b, Int2 predicate) {
    const UInt2 mask = sycl::bit_cast<UInt2>(predicate);
    return (a & mask) | (b & ~mask);
  };
  const UInt2 subnormal = pick(
      magnitude * 0x3b00u,
      pick(
          0x3b80u + ((magnitude - 2u) << 6),
          0x3c00u + ((magnitude - 4u) << 5),
          magnitude < 4u),
      magnitude < 2u);
  const UInt2 finite =
      pick(subnormal, (magnitude << 4) + 0x3c00u, magnitude < 8u);
  const UInt2 result =
      pick(UInt2{0x7fc0u, 0x7fc0u}, finite, magnitude == 127u) |
      ((raw & 128u) << 8);
  return result[0] | (result[1] << 16);
#else
  return uint32_t(fp8_bits(bytes & 255u)) |
         (uint32_t(fp8_bits((bytes >> 8) & 255u)) << 16);
#endif
}

// Compensate the FP16 weight scaling once per output projection.
inline uint32_t fp8_pair(uint32_t bytes) {
#ifdef MINICPM5_MONO_FP16
  return fp8_pair_half(bytes);
#else
  return fp8_pair_bf16(bytes);
#endif
}

// Two E4M3 bytes already occupy bits0..7 and16..23. NK projections pair
// K(0,2) and K(1,3), matching the reference's permuted packed-A layout.
inline uint32_t fp8_lanes(uint32_t lanes) {
#ifdef MINICPM5_MONO_FP16
  return fp8_half_lanes(lanes);
#elif defined(MINICPM5_MONO_BF16_FP8_HALF_ENABLED)
  return half_pair_to_bf16(fp8_half_lanes(lanes));
#else
  return fp8_pair_bf16((lanes & 255u) | ((lanes >> 8) & 0xff00u));
#endif
}

// Keep BF16 M8's measured shift/OR schedule. Other shapes use an exactly
// equivalent packed NaN correction with fewer integer instructions.
template <int Rows>
constexpr bool FP8NanMultiply =
#ifdef MINICPM5_MONO_FP16
    true;
#elif defined(MINICPM5_MONO_BF16_FP8_HALF_ENABLED)
    Rows != 8;
#else
    false;
#endif

inline uint32_t fp8_half_lanes_multiply(uint32_t lanes) {
  const uint32_t magnitude = lanes & 0x007f007fu;
  const uint32_t nan = (magnitude + 0x00010001u) & 0x00800080u;
  return (((lanes & 0x00800080u) << 8) | (magnitude << 7)) ^ (nan * 0x83u);
}

template <int Rows>
inline uint32_t fp8_lanes_rows(uint32_t lanes) {
  static_assert(Rows >= 1 && Rows <= 8);
  if constexpr (FP8NanMultiply<Rows>) {
    const uint32_t halves = fp8_half_lanes_multiply(lanes);
#ifdef MINICPM5_MONO_FP16
    return halves;
#else
    return half_pair_to_bf16(halves);
#endif
  } else {
    return fp8_lanes(lanes);
  }
}

template <int Rows>
inline uint32_t fp8_pair_rows(uint32_t bytes) {
  if constexpr (FP8NanMultiply<Rows>)
    return fp8_lanes_rows<Rows>((bytes & 255u) | ((bytes & 0xff00u) << 8));
  else
    return fp8_pair(bytes);
}

inline float fp8_scale(float scale) {
#ifdef MINICPM5_MONO_FP16
  return scale * 256.f;
#else
  return scale;
#endif
}

inline void dpas16(Float8& out, const UInt4& a, const UInt8& b) {
#ifdef __SYCL_DEVICE_ONLY__
  asm("{\n"
      ".decl DST v_type=G type=f num_elts=128 alias=<%0,0>\n"
      ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"
      ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"
  #ifdef MINICPM5_MONO_FP16
      "dpas.hf.hf.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"
  #else
      "dpas.bf.bf.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"
  #endif
      "}\n"
      : "+rw"(out)
      : "rw"(a), "rw"(b));
#else
  (void)out;
  (void)a;
  (void)b;
#endif
}

template <int Rows>
using ProjectionAccumulator = float __attribute__((ext_vector_type(Rows)));

// Projection rows are batch items, unlike the eight simultaneous Q heads in
// attention. Use the smallest supported physical repeat; non-power-of-two
// batches ignore the extra physical rows and store only real request rows.
template <int Rows>
inline void dpas_rows(Float8& out, const UInt4& a, const UInt8& b) {
  static_assert(Rows >= 1 && Rows <= 8);
  if constexpr (Rows == 3) {
    dpas_rows<4>(out, a, b);
  } else if constexpr (Rows > 4 && Rows < 8) {
    dpas_rows<8>(out, a, b);
  } else if constexpr (Rows == 8) {
    dpas16(out, a, b);
  } else {
    ProjectionAccumulator<Rows> values;
#pragma unroll
    for (int r = 0; r < Rows; ++r)
      values[r] = out[r];
#ifdef __SYCL_DEVICE_ONLY__
  #ifdef MINICPM5_MONO_FP16
    #define MINICPM5_ROW_DPAS_KIND "dpas.hf.hf.8."
  #else
    #define MINICPM5_ROW_DPAS_KIND "dpas.bf.bf.8."
  #endif
  #define MINICPM5_ROW_DPAS_CASE(R, N)                               \
    asm("{\n"                                                        \
        ".decl DST v_type=G type=f num_elts=" #N                     \
        " alias=<%0,0>\n"                                            \
        ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n" \
        ".decl SRC2_UD v_type=G type=UD num_elts=64 "                \
        "alias=<%1,0>\n" MINICPM5_ROW_DPAS_KIND #R                   \
        " (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"             \
        "}\n"                                                        \
        : "+rw"(values)                                              \
        : "rw"(a), "rw"(b))
    if constexpr (Rows == 1) MINICPM5_ROW_DPAS_CASE(1, 16);
    if constexpr (Rows == 2) MINICPM5_ROW_DPAS_CASE(2, 32);
    if constexpr (Rows == 4) MINICPM5_ROW_DPAS_CASE(4, 64);
  #undef MINICPM5_ROW_DPAS_CASE
  #undef MINICPM5_ROW_DPAS_KIND
#else
    (void)a;
    (void)b;
#endif
#pragma unroll
    for (int r = 0; r < Rows; ++r)
      out[r] = values[r];
  }
}

template <int Rows>
inline UInt4
input_tile(int lane, const scalar_t* input, int k, const int (&offsets)[Rows]) {
  UInt4 a;
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    const int row = operand_row(lane, word);
    const int column = k + operand_column(lane);
    a[word] = row < Rows && offsets[row] >= 0
                  ? pack16(
                        input[offsets[row] + column],
                        input[offsets[row] + column + 1])
                  : 0u;
  }
  return a;
}

// NK storage: one lane reads sixteen contiguous K elements of one output.
// FP8 is converted exactly to BF16 before FP32 DPAS accumulation.
template <int Rows, int K>
inline Float8 projection_nk(
    sycl::sub_group sg,
    const uint8_t* weight,
    const scalar_t* input,
    const int (&offsets)[Rows]) {
  static_assert(K % 16 == 0);
  const int lane = sg.get_local_linear_id();
  weight += size_t(lane) * K;
  Float8 result{};
  auto current = *reinterpret_cast<const sycl::uint4*>(weight);
  for (int k = 0; k < K; k += 16) {
    sycl::uint4 next;
    if (k + 16 < K)
      next = *reinterpret_cast<const sycl::uint4*>(weight + k + 16);
    const auto bytes = sycl::bit_cast<sycl::vec<uint8_t, 16>>(current);
    UInt8 b;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
#if defined(MINICPM5_MONO_FAST_FP8_ENABLED) || defined(MINICPM5_MONO_FP16)
      b[p] = fp8_pair_rows<Rows>(current[p / 2] >> (16 * (p % 2)));
#else
      b[p] = uint32_t(fp8_bits(bytes[2 * p])) |
             (uint32_t(fp8_bits(bytes[2 * p + 1])) << 16);
#endif
    }
    dpas_rows<Rows>(result, input_tile<Rows>(lane, input, k, offsets), b);
    if (k + 16 < K) current = next;
  }
#ifdef MINICPM5_MONO_FP16
  result *= 256.f;
#endif
  return result;
}

#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL UInt8
__builtin_IB_subgroup_block_read_flat_transform_u8_k32(
    intptr_t base, int width_m1, int height_m1, int pitch_m1, Int2 xy);
#endif

// KN storage: each dword contains four consecutive K rows for a lane's
// output column. The 2D read supplies two consecutive 16-K DPAS operands.
template <int K, int N>
inline UInt8 weight_kn(const uint8_t* weight, int column, int k) {
#ifdef __SYCL_DEVICE_ONLY__
  return __builtin_IB_subgroup_block_read_flat_transform_u8_k32(
      intptr_t(weight), N - 1, K - 1, N - 1, Int2{column, k});
#else
  (void)weight;
  (void)column;
  (void)k;
  return UInt8{};
#endif
}

template <int Rows, int K, int N>
inline Float8 projection_kn(
    sycl::sub_group sg,
    const uint8_t* weight,
    int column,
    const scalar_t* input,
    const int (&offsets)[Rows],
    int begin,
    int end) {
  static_assert(K % 32 == 0 && N % 16 == 0);
  const int lane = sg.get_local_linear_id();
  Float8 result{};
  UInt8 current = weight_kn<K, N>(weight, column, begin);
  for (int k = begin; k < end; k += 32) {
    UInt8 next;
    if (k + 32 < end) next = weight_kn<K, N>(weight, column, k + 32);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      UInt8 b;
#pragma unroll
      for (int p = 0; p < 8; ++p)
        b[p] = fp8_pair_rows<Rows>(current[4 * half + p / 2] >> (16 * (p % 2)));
      dpas_rows<Rows>(
          result, input_tile<Rows>(lane, input, k + half * 16, offsets), b);
    }
    if (k + 32 < end) current = next;
  }
#ifdef MINICPM5_MONO_FP16
  result *= 256.f;
#endif
  return result;
}
}  // namespace vllm::minicpm5::MINICPM5_MONO_NAMESPACE
