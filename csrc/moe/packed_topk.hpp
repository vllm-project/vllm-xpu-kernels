// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <type_traits>

// Unbiased 512-expert softmax routing uses ordered score/index keys so one
// subgroup reduction selects both the winning score and its lowest index.
namespace vllm::moe::packed_topk {
template <typename T>
struct SoftmaxScoring {
  bool renormalize;
  float row_max = 0.0f;
  float inv_z = 1.0f;

  bool needs_full_z() const { return !renormalize; }
  void set_row_max(float m) { row_max = m; }
  void set_inv_z(float iz) { inv_z = iz; }
  float norm_accum(float x) const { return sycl::native::exp(x - row_max); }
  float winner_weight(float x, int) const { return norm_accum(x); }
  float output_scale() const { return inv_z; }
};

inline float ieee_div(float a, float b) {
  volatile float denom = b;
  return a / denom;
}

template <int LANES>
inline float lane_group_sum(const sycl::sub_group& sg, float s) {
#pragma unroll
  for (int m = 1; m < LANES; m <<= 1)
    s += sycl::permute_group_by_xor(sg, s, static_cast<size_t>(m));
  return s;
}

template <typename T>
struct OrderedKey16 {
  static_assert(sizeof(T) == 2, "packed keys need a 16-bit score");
  static constexpr uint32_t kSign = 0x8000u;
  // Bit pattern of +inf: 0x7F80 (bf16) / 0x7C00 (fp16).
  static constexpr uint32_t kInfBits =
      std::is_same_v<T, sycl::half> ? 0x7C00u : 0x7F80u;

  static uint32_t order(uint32_t b) {
    const bool is_nan = (b & 0x7FFFu) > kInfBits;
    b = b == kSign ? 0u : b;  // -0 -> +0
    const uint32_t o = (b & kSign) ? (~b & 0xFFFFu) : (b | kSign);
    return is_nan ? 0u : o;
  }

  static float value(uint32_t key) {
    const uint32_t o = key >> 16;
    const uint16_t b =
        static_cast<uint16_t>((o & kSign) ? (o & 0x7FFFu) : (~o & 0xFFFFu));
    return static_cast<float>(sycl::bit_cast<T>(b));
  }

  static uint32_t pack(T x, int idx) {
    return order(sycl::bit_cast<uint16_t>(x)) << 16 |
           (0xFFFFu - static_cast<uint32_t>(idx));
  }

  static int index(uint32_t key) {
    return static_cast<int>(0xFFFFu - (key & 0xFFFFu));
  }
};

template <typename InputT, int TOPK, int SG_SIZE = 32>
struct PackedTopK {
  static_assert(TOPK == 3 || TOPK == 10);
  static_assert(SG_SIZE == 16 || SG_SIZE == 32);
  static constexpr int N = 512, GROUPS = 4;
  using Policy = SoftmaxScoring<InputT>;
  using Key = OrderedKey16<InputT>;
  static constexpr int kVpl = N / SG_SIZE;
  static constexpr int kGroups = GROUPS;
  static constexpr int kPerGroup = kVpl / kGroups;
  static_assert(N % SG_SIZE == 0 && kVpl % kGroups == 0);
  static_assert(N <= 0xFFFF, "index field is 16 bits");

  const InputT* __restrict__ gating;
  float* __restrict__ weights;
  int* __restrict__ indices;
  int* __restrict__ source_rows;
  const bool* __restrict__ is_padding;
  Policy policy;
  const double routed_scaling_factor;
  int64_t num_tokens;

  static uint32_t max_of(const uint32_t* k, int n) {
    // Balanced tree: log2(n) dependent steps.
    uint32_t t[kVpl];
#pragma unroll
    for (int i = 0; i < n; ++i)
      t[i] = k[i];
#pragma unroll
    for (int s = 1; s < n; s <<= 1)
#pragma unroll
      for (int i = 0; i + s < n; i += 2 * s)
        t[i] = sycl::max(t[i], t[i + s]);
    return t[0];
  }

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void
  operator()(sycl::nd_item<1> it) const {
    const sycl::sub_group sg = it.get_sub_group();
    const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
    const int lane = static_cast<int>(gid & (SG_SIZE - 1));
    const int64_t token = gid / SG_SIZE;
    // A whole sub-group shares one token, so this exit is uniform.
    if (token >= num_tokens) return;
    const InputT* __restrict__ row = gating + token * N;

    Policy policy = this->policy;

    uint32_t key[kVpl];
#pragma unroll
    for (int j = 0; j < kVpl; ++j) {
      const int idx = j * SG_SIZE + lane;
      key[j] = Key::pack(row[idx], idx);
    }
    uint32_t gm[kGroups];
#pragma unroll
    for (int g = 0; g < kGroups; ++g)
      gm[g] = max_of(key + g * kPerGroup, kPerGroup);

    uint32_t win[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      uint32_t best = max_of(gm, kGroups);
      best = sycl::reduce_over_group(sg, best, sycl::maximum<uint32_t>());
      win[k] = best;

      if (k == 0) {
        // Softmax without bias: the top-1 logit is the row max M.
        policy.set_row_max(Key::value(best));
        if (policy.needs_full_z()) {
          // Same per-lane order (i = lane, lane + 16, ...) and lane
          // reduction as the fallback engine.
          float zs = 0.0f;
#pragma unroll
          for (int j = 0; j < kVpl; ++j)
            zs += policy.norm_accum(Key::value(key[j]));
          zs = lane_group_sum<SG_SIZE>(sg, zs);
          policy.set_inv_z(ieee_div(1.0f, zs));
        }
      }

      if (k + 1 < TOPK) {
        // Evict the winner and re-reduce its group only. The group is a
        // sub-group-uniform value, so this is a uniform branch.
        const int wg = (Key::index(best) / SG_SIZE) / kPerGroup;
#pragma unroll
        for (int g = 0; g < kGroups; ++g) {
          if (g == wg) {
#pragma unroll
            for (int s = 0; s < kPerGroup; ++s) {
              uint32_t& kv = key[g * kPerGroup + s];
              kv = kv == best ? 0u : kv;
            }
            gm[g] = max_of(key + g * kPerGroup, kPerGroup);
          }
        }
      }
    }

    // Weights in pick order (same accumulation order as the other engines).
    float num[TOPK];
    int wid[TOPK];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      wid[k] = Key::index(win[k]);
      num[k] = policy.winner_weight(Key::value(win[k]), wid[k]);
      sum += num[k];
    }

    float scale =
        static_cast<float>(routed_scaling_factor) * policy.output_scale();
    if (policy.renormalize) {
      const float denom = sum > 0.0f ? sum : 1.0f;
      scale = ieee_div(scale, denom);
    }

    // Lane k stores output slot k: one store message per output array.
    float my_w = num[0];
    int my_i = wid[0];
#pragma unroll
    for (int k = 1; k < TOPK; ++k) {
      my_w = lane == k ? num[k] : my_w;
      my_i = lane == k ? wid[k] : my_i;
    }
    if (lane < TOPK) {
      const bool is_pad = is_padding != nullptr && is_padding[token];
      const int64_t out = token * TOPK + lane;
      weights[out] = is_pad ? 0.0f : my_w * scale;
      indices[out] = is_pad ? -1 : my_i;
      source_rows[out] = static_cast<int>(lane * num_tokens + token);
    }
  }
};

}  // namespace vllm::moe::packed_topk
