// SPDX-License-Identifier: Apache-2.0
#include "minicpm5/mono.h"
#include "minicpm5/sampling.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

template <int Rows>
void check_fp8_row_policy() {
#ifdef MINICPM5_MONO_FP16
  using namespace vllm::minicpm5::mono_fp16;
#else
  using namespace vllm::minicpm5::mono;
#endif
  for (uint32_t bytes = 0; bytes < 65536; ++bytes) {
    const uint32_t expected = fp8_pair(bytes);
    assert(fp8_pair_rows<Rows>(bytes) == expected);
    assert(
        fp8_lanes_rows<Rows>((bytes & 255u) | ((bytes & 0xff00u) << 8)) ==
        expected);
  }
}

template <int TP, int Rows>
void check_reference_context_ownership() {
#ifdef MINICPM5_MONO_FP16
  using namespace vllm::minicpm5::mono_fp16;
#else
  using namespace vllm::minicpm5::mono;
#endif
  const int boundaries[] = {
      0, 1, 15, 16, 17, 63, 64, 65, 129, 1024, 16385, 32768};
  for (int offset = 0; offset < 12; ++offset) {
    int lengths[Rows];
    int64_t slots[Rows];
    for (int r = 0; r < Rows; ++r) {
      lengths[r] = boundaries[(offset + 3 * r) % 12];
      slots[r] = lengths[r] ? r : -1;
    }
    Step step{};
    step.lengths = lengths;
    step.slots = slots;
    AttentionChunks<TP, Rows> chunks(step);
    assert(chunks.first[Rows] <= chunks.Slots);
    assert(chunks.width > 0 && chunks.width % 16 == 0);
    if (chunks.dynamic) {
      assert(chunks.first[Rows] == chunks.Slots);
      for (int r = 0; r < Rows; ++r)
        assert(
            lengths[r] > 0 ? chunks.first[r + 1] > chunks.first[r]
                           : chunks.first[r + 1] == chunks.first[r]);
      continue;
    }
    for (int r = 0; r < Rows; ++r) {
      std::vector<int> visits(lengths[r]);
      for (int slot = chunks.first[r]; slot < chunks.first[r + 1]; ++slot) {
        const int begin = (slot - chunks.first[r]) * chunks.width;
        const int end = std::min(begin + chunks.width, lengths[r]);
        for (int token = begin; token < end; ++token)
          ++visits[token];
      }
      for (int count : visits)
        assert(count == 1);
    }
  }
}

int main() {
#ifdef MINICPM5_MONO_FP16
  using namespace vllm::minicpm5::mono_fp16;
#else
  using namespace vllm::minicpm5::mono;
#endif
  check_fp8_row_policy<1>();
  check_fp8_row_policy<2>();
  check_fp8_row_policy<3>();
  check_fp8_row_policy<4>();
  check_fp8_row_policy<5>();
  check_fp8_row_policy<6>();
  check_fp8_row_policy<7>();
  check_fp8_row_policy<8>();
  uint32_t control[ControlWords]{};
  uint32_t health[8]{};
  fail(control, 3, 12, 7, 11, health);
  assert(control[Failed] == 1 && health[0] == 1);
  assert(control[FailureKind] == 3 && control[FailurePhase] == 12);
  assert(control[FailureCall] == 7 && control[FailureObserved] == 11);
  // Later failures retain the first local diagnosis and still notify the
  // shared rank status, even when the local control was already failed.
  health[0] = 0;
  fail(control, 2, 8, 9, 13, health);
  assert(health[0] == 1 && control[FailureKind] == 3);
  assert(control[FailureCall] == 7 && control[FailureObserved] == 11);
  // All finite e4m3 encodings, including both signed zeros and subnormals.
  for (int value = 0; value < 256; ++value) {
    const int exponent = (value >> 3) & 15, mantissa = value & 7;
    const float sign = value & 128 ? -1.f : 1.f;
    const float expected =
        exponent == 0
            ? sign * std::ldexp(float(mantissa), -9)
            : sign * std::ldexp(1.f + float(mantissa) / 8, exponent - 7);
    const float actual = fp8(value);
    if ((value & 127) == 127)
      assert(std::isnan(actual));
    else {
      assert(actual == expected);
      assert(std::signbit(actual) == std::signbit(expected));
    }
  }
  for (uint32_t tag : {0u, 1u, 65535u, 65536u, 0xffffffffu}) {
    for (uint32_t bits :
         {0u, 0x80000000u, 0x3f800000u, 0x7f800000u, 0x7fc00001u}) {
      const uint64_t encoded = encode_mail(sycl::bit_cast<float>(bits), tag);
      assert(uint32_t(encoded >> 48) == (tag & 0xffffu));
      assert(((encoded >> 16) & 0xffffu) == (tag & 0xffffu));
      const uint32_t decoded =
          (uint32_t(encoded >> 32) << 16) | (encoded & 0xffffu);
      assert(decoded == bits);
    }
  }
  // BF16's range must survive scratch staging; FP16 cannot hold this value.
#ifndef MINICPM5_MONO_FP16
  assert(float(scalar_t(131072.f)) == 131072.f);
#else
  assert(std::isinf(float(scalar_t(131072.f))));
#endif
  // Reconstruct every DPAS A element from the lane/word mapping. Zero-padded
  // rows and experts absent from a request must never read another row.
  scalar_t inputs[8 * 32];
  int offsets[8];
  for (int r = 0; r < 8; ++r) {
    offsets[r] = r == 3 ? -1 : r * 32;
    for (int k = 0; k < 32; ++k)
      inputs[r * 32 + k] = scalar_t(r * 64 + k);
  }
  int visited[8][16] = {};
  for (int lane = 0; lane < 16; ++lane) {
    const auto tile = input_tile<8>(lane, inputs, 16, offsets);
    for (int word = 0; word < 4; ++word) {
      const int row = 2 * word + lane / 8;
      const int column = 2 * (lane % 8);
      for (int part = 0; part < 2; ++part) {
        const uint16_t actual = uint16_t(tile[word] >> (16 * part));
        const uint16_t expected =
            row == 3 ? 0
                     : sycl::bit_cast<uint16_t>(
                           inputs[row * 32 + 16 + column + part]);
        assert(actual == expected);
        ++visited[row][column + part];
      }
    }
    const int short_offsets[2] = {0, 32};
    const auto short_tile = input_tile<2>(lane, inputs, 0, short_offsets);
    for (int word = 1; word < 4; ++word)
      assert(short_tile[word] == 0u);
  }
  for (const auto& row : visited)
    for (int count : row)
      assert(count == 1);
  for (uint32_t value = 0; value < 65536; ++value) {
    const uint32_t packed = fp8_pair(value);
    assert(fp8_lanes((value & 255u) | ((value & 0xff00u) << 8)) == packed);
#ifdef MINICPM5_MONO_FP16
    for (int part = 0; part < 2; ++part) {
      const float actual =
          float(sycl::bit_cast<scalar_t>(uint16_t(packed >> (16 * part)))) *
          256.f;
      const float expected = fp8(uint8_t(value >> (8 * part)));
      assert(std::isnan(expected) ? std::isnan(actual) : actual == expected);
      if (!std::isnan(expected))
        assert(std::signbit(actual) == std::signbit(expected));
    }
#else
    assert(uint16_t(packed) == fp8_bits(value & 255u));
    assert(uint16_t(packed >> 16) == fp8_bits(value >> 8));
#endif
  }
  // Exercise the actual M1 packed decoder/AXPY, including BF16-only range.
  for (int byte = 0; byte < 256; ++byte)
    for (float input : {0.f, -2.f, 1.e-12f, 131072.f}) {
      float actual[16]{};
      const uint32_t word = uint32_t(byte) * 0x01010101u;
      reference_axpy16(actual, sycl::uint4(word), input);
      const float expected = fp8(byte) * input;
      for (float value : actual)
        assert(std::isnan(expected) ? std::isnan(value) : value == expected);
    }
  check_reference_context_ownership<1, 1>();
  check_reference_context_ownership<1, 2>();
  check_reference_context_ownership<1, 4>();
  check_reference_context_ownership<1, 8>();
  check_reference_context_ownership<2, 1>();
  check_reference_context_ownership<2, 2>();
  check_reference_context_ownership<2, 4>();
  check_reference_context_ownership<2, 8>();
  int experts[2 * Top];
  for (int k = 0; k < Top; ++k) {
    experts[k] = k;
    experts[Top + k] = k + 8;
  }
  int union_size = 0;
  for (int i = 0; i < 2 * Top; ++i)
    union_size += first_expert(experts, i);
  assert(union_size == 24);
  assert(expert_choice(experts, 0, 12) == 12);
  assert(expert_choice(experts, 1, 12) == 4);
  assert(expert_choice(experts, 1, 0) == -1);
  assert(row_metadata_valid(32768, 32767, 32767, 64, 512, 512, 0));
  assert(row_metadata_valid(0, 0, -1, 64, 512, 512, 0));
  assert(!row_metadata_valid(1, 0, -2, 64, 512, 512, 0));
  assert(!row_metadata_valid(32769, 32768, 32768, 64, 513, 513, 32768));
  assert(row_metadata_valid(32769, 32768, 32768, 64, 513, 513, 32769));
  assert(!row_metadata_valid(65, 64, 64, 64, 2, 1, 0));
  assert(!row_metadata_valid(1, 0, 128, 64, 2, 1, 0));
  assert(!row_metadata_valid(10, 8, 9, 64, 1, 1, 0));
  assert(!row_metadata_valid(0, -1, 0, 64, 1, 1, 0));
  assert(!row_metadata_valid(1, 0, 0, 0, 1, 1, 0));
  // Candidate ordering must agree with greedy argmax, including signed zero
  // ties and global token IDs across the two contiguous vocabulary shards.
  using namespace vllm::minicpm5::sampling;
  assert(make_key(-0.f, 0) > make_key(0.f, 1));
  assert(make_key(-0.f, 1) == make_key(0.f, 1));
  for (float value : {-INFINITY, -3.f, -0.f, 0.f, 1.f, INFINITY}) {
    for (int token : {0, 65279, 65280, 130559}) {
      const auto key = make_key(value, token);
      assert(key_tok(key) == token);
      assert(key_val(key) == value);
      assert(make_key(value, token) > make_key(value, token + 1));
    }
  }
  assert(make_key(-3.f, 0) < make_key(0.f, 130559));
  assert(make_key(0.f, 0) < make_key(1.f, 130559));
  // At the target shard size a slice has at most 32 scalar logits per lane.
  // At most K lanes' logits can exceed the K-th lane-maximum threshold.
  assert(KMAX * ((65280 / NWR + WGS - 1) / WGS) <= CAP);
}
