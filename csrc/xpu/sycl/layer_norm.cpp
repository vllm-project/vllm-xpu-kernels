// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <sycl/sycl.hpp>

#include <cstdint>

#include <ATen/DeviceGuard.h>

#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

constexpr int kLayerNormSgSize = 32;
// Elements-per-lane for the flat sub-group load/store used by the small
// (norm_dim <= 512) kernels: 512 bytes / (lanes * 4-byte uint32 unit).
constexpr int kLayerNormSmallNU = 512 / (kLayerNormSgSize * 4);

template <int N>
inline void layer_norm_ld_flat(const uint32_t* p, int lane, uint32_t* d) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    d[i] = p[i * kLayerNormSgSize + lane];
  }
}

template <int N>
inline void layer_norm_st_flat(uint32_t* p, int lane, const uint32_t* s) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    p[i * kLayerNormSgSize + lane] = s[i];
  }
}

// Two half/bfloat16 elements packed into one 32-bit lane.
template <typename Scalar>
struct LayerNormPacked16 {
  static inline float lo(uint32_t value) {
    return sycl::bit_cast<Scalar>(static_cast<uint16_t>(value & 0xFFFFu));
  }
  static inline float hi(uint32_t value) {
    return sycl::bit_cast<Scalar>(static_cast<uint16_t>(value >> 16));
  }
  static inline uint32_t pack(float lo, float hi) {
    uint16_t packed_lo = sycl::bit_cast<uint16_t>(Scalar(lo));
    uint16_t packed_hi = sycl::bit_cast<uint16_t>(Scalar(hi));
    return static_cast<uint32_t>(packed_lo) |
        (static_cast<uint32_t>(packed_hi) << 16);
  }
};

// norm_dim == 128: packs 2 heads into one 512-byte sub-group block.
template <typename Scalar>
class layer_norm_small_head128_kernel {
 public:
  layer_norm_small_head128_kernel(
      const uint8_t* input_buf,
      uint8_t* output_buf,
      const float* weight,
      int batch,
      float epsilon)
      : input_buf_(input_buf),
        output_buf_(output_buf),
        weight_(weight),
        batch_(batch),
        epsilon_(epsilon) {}

  void operator() [[sycl::reqd_sub_group_size(kLayerNormSgSize)]] (
      sycl::nd_item<1> it) const {
    using pio = LayerNormPacked16<Scalar>;
    constexpr int HEADS_PER_BLOCK = 2;
    constexpr int PACKED_PER_HEAD = 128 / (2 * kLayerNormSgSize);
    int gsg = it.get_group_linear_id() *
            (it.get_local_range(0) / kLayerNormSgSize) +
        it.get_sub_group().get_group_linear_id();
    if (gsg >= (batch_ + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK) return;
    auto sg = it.get_sub_group();
    int lane = sg.get_local_linear_id();

    const uint32_t* input_block =
        reinterpret_cast<const uint32_t*>(input_buf_) +
        (int64_t)gsg * (kLayerNormSmallNU * kLayerNormSgSize);
    uint32_t* output_block = reinterpret_cast<uint32_t*>(output_buf_) +
        (int64_t)gsg * (kLayerNormSmallNU * kLayerNormSgSize);

    uint32_t blk[kLayerNormSmallNU], sd[kLayerNormSmallNU];
    bool valid_heads[HEADS_PER_BLOCK] = {
        gsg * HEADS_PER_BLOCK < batch_,
        gsg * HEADS_PER_BLOCK + 1 < batch_,
    };
#pragma unroll
    for (int head = 0; head < HEADS_PER_BLOCK; head++) {
      int base = head * PACKED_PER_HEAD;
      if (valid_heads[head]) {
        layer_norm_ld_flat<PACKED_PER_HEAD>(
            input_block + base * kLayerNormSgSize, lane, blk + base);
      } else {
#pragma unroll
        for (int i = 0; i < PACKED_PER_HEAD; i++) {
          blk[base + i] = 0;
        }
      }
    }

    // weight (same for both heads)
    float wlo[PACKED_PER_HEAD], whi[PACKED_PER_HEAD];
#pragma unroll
    for (int i = 0; i < PACKED_PER_HEAD; i++) {
      int idx = (i * kLayerNormSgSize + lane) * 2;
      wlo[i] = weight_[idx];
      whi[i] = weight_[idx + 1];
    }

    // Sum and sum-of-squares per head.
    sycl::vec<float, HEADS_PER_BLOCK> sum(0.0f);
    sycl::vec<float, HEADS_PER_BLOCK> sum_sq(0.0f);
#pragma unroll
    for (int head = 0; head < HEADS_PER_BLOCK; head++) {
      int base = head * PACKED_PER_HEAD;
#pragma unroll
      for (int i = 0; i < PACKED_PER_HEAD; i++) {
        float lo = pio::lo(blk[base + i]);
        float hi = pio::hi(blk[base + i]);
        sum[head] += lo + hi;
        sum_sq[head] += lo * lo + hi * hi;
      }
    }
    sum = sycl::reduce_over_group(
        sg, sum, sycl::plus<sycl::vec<float, HEADS_PER_BLOCK>>());
    sum_sq = sycl::reduce_over_group(
        sg, sum_sq, sycl::plus<sycl::vec<float, HEADS_PER_BLOCK>>());

    // normalise each head
#pragma unroll
    for (int head = 0; head < HEADS_PER_BLOCK; head++) {
      int base = head * PACKED_PER_HEAD;
      float mean = sum[head] / 128.0f;
      float inv_std = sycl::rsqrt(
          sycl::fmax(sum_sq[head] / 128.0f - mean * mean, 0.0f) + epsilon_);
#pragma unroll
      for (int i = 0; i < PACKED_PER_HEAD; i++) {
        float lo = (pio::lo(blk[base + i]) - mean) * inv_std * wlo[i];
        float hi = (pio::hi(blk[base + i]) - mean) * inv_std * whi[i];
        sd[base + i] = pio::pack(lo, hi);
      }
    }

#pragma unroll
    for (int head = 0; head < HEADS_PER_BLOCK; head++) {
      if (!valid_heads[head]) continue;
      int base = head * PACKED_PER_HEAD;
      layer_norm_st_flat<PACKED_PER_HEAD>(
          output_block + base * kLayerNormSgSize, lane, sd + base);
    }
  }

 private:
  const uint8_t* input_buf_;
  uint8_t* output_buf_;
  const float* weight_;
  int batch_;
  float epsilon_;
};

// norm_dim in {256, 512}: BLOCKS sub-group-flat blocks per row.
template <typename Scalar, int BLOCKS>
class layer_norm_small_kernel {
 public:
  layer_norm_small_kernel(
      const uint8_t* input_buf,
      uint8_t* output_buf,
      const float* weight,
      int batch,
      float epsilon)
      : input_buf_(input_buf),
        output_buf_(output_buf),
        weight_(weight),
        batch_(batch),
        epsilon_(epsilon) {}

  void operator() [[sycl::reqd_sub_group_size(kLayerNormSgSize)]] (
      sycl::nd_item<1> it) const {
    using pio = LayerNormPacked16<Scalar>;
    constexpr int PACKED = BLOCKS * kLayerNormSmallNU;
    constexpr float NORM_DIM = PACKED * kLayerNormSgSize * 2;
    int gsg = it.get_group_linear_id() *
            (it.get_local_range(0) / kLayerNormSgSize) +
        it.get_sub_group().get_group_linear_id();
    if (gsg >= batch_) return;
    auto sg = it.get_sub_group();
    int lane = sg.get_local_linear_id();

    const uint32_t* input = reinterpret_cast<const uint32_t*>(input_buf_) +
        (int64_t)gsg * (PACKED * kLayerNormSgSize);
    uint32_t* output = reinterpret_cast<uint32_t*>(output_buf_) +
        (int64_t)gsg * (PACKED * kLayerNormSgSize);
    uint32_t values[PACKED];
    float wlo[PACKED], whi[PACKED];
#pragma unroll
    for (int block = 0; block < BLOCKS; block++)
      layer_norm_ld_flat<kLayerNormSmallNU>(
          input + block * kLayerNormSmallNU * kLayerNormSgSize,
          lane,
          values + block * kLayerNormSmallNU);

    float sum = 0.0f;
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < PACKED; i++) {
      int idx = (i * kLayerNormSgSize + lane) * 2;
      float lo = pio::lo(values[i]);
      float hi = pio::hi(values[i]);
      wlo[i] = weight_[idx];
      whi[i] = weight_[idx + 1];
      sum += lo + hi;
      sum_sq += lo * lo + hi * hi;
    }
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
    float mean = sum / NORM_DIM;
    float inv_std = sycl::rsqrt(
        sycl::fmax(sum_sq / NORM_DIM - mean * mean, 0.0f) + epsilon_);

#pragma unroll
    for (int i = 0; i < PACKED; i++) {
      float lo = (pio::lo(values[i]) - mean) * inv_std * wlo[i];
      float hi = (pio::hi(values[i]) - mean) * inv_std * whi[i];
      values[i] = pio::pack(lo, hi);
    }
#pragma unroll
    for (int block = 0; block < BLOCKS; block++)
      layer_norm_st_flat<kLayerNormSmallNU>(
          output + block * kLayerNormSmallNU * kLayerNormSgSize,
          lane,
          values + block * kLayerNormSmallNU);
  }

 private:
  const uint8_t* input_buf_;
  uint8_t* output_buf_;
  const float* weight_;
  int batch_;
  float epsilon_;
};

// ==========================================================================
// Small kernel: 1 sub-group per row (norm_dim in {128, 256, 512}).
// ==========================================================================
template <typename Scalar>
void layer_norm_small(
    sycl::queue& q,
    const uint8_t* input_buf,
    uint8_t* output_buf,
    const float* weight,
    int batch,
    int norm_dim,
    float epsilon) {
  constexpr int WG = 1024;
  constexpr int SGS_PER_WG = WG / kLayerNormSgSize;

  if (norm_dim == 128) {
    constexpr int HEADS_PER_BLOCK = 2;
    int total_sgs = (batch + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;
    int n_wg = (total_sgs + SGS_PER_WG - 1) / SGS_PER_WG;
    int global = n_wg * WG;

    q.submit([&](sycl::handler& h) {
      layer_norm_small_head128_kernel<Scalar> kfn(
          input_buf, output_buf, weight, batch, epsilon);
      h.parallel_for(sycl::nd_range<1>(global, WG), kfn);
    });

  } else {
    int n_wg = (batch + SGS_PER_WG - 1) / SGS_PER_WG;
    int global = n_wg * WG;

    auto launch = [&]<int BLOCKS>() {
      q.submit([&](sycl::handler& h) {
        layer_norm_small_kernel<Scalar, BLOCKS> kfn(
            input_buf, output_buf, weight, batch, epsilon);
        h.parallel_for(sycl::nd_range<1>(global, WG), kfn);
      });
    };

    if (norm_dim == 256)
      launch.template operator()<1>();
    else if (norm_dim == 512)
      launch.template operator()<2>();
  }
}

// ==========================================================================
// Medium kernel: 1 work-group per row, data retained in registers.
// Specialised for norm_dim in {1024, 2048, 4096, 7168, 8192, 16384, 32768}.
// ==========================================================================
template <typename Scalar, int NORM_DIM, int WG>
class layer_norm_medium_kernel {
 public:
  layer_norm_medium_kernel(
      const uint8_t* input_buf,
      uint8_t* output_buf,
      const float* weight,
      float epsilon)
      : input_buf_(input_buf),
        output_buf_(output_buf),
        weight_(weight),
        epsilon_(epsilon) {}

  void operator()(sycl::nd_item<1> it) const {
    using pio = LayerNormPacked16<Scalar>;
    constexpr int PACKED = NORM_DIM / 2;
    constexpr int N = PACKED / WG;
    int row = it.get_group_linear_id();
    int tid = it.get_local_linear_id();
    const uint32_t* input = reinterpret_cast<const uint32_t*>(input_buf_) +
        (int64_t)row * PACKED;
    uint32_t* output = reinterpret_cast<uint32_t*>(output_buf_) +
        (int64_t)row * PACKED;

    uint32_t values[N];
    float wlo[N], whi[N];
    float sum = 0.0f;
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < N; i++) {
      int packed_idx = i * WG + tid;
      uint32_t value = input[packed_idx];
      values[i] = value;
      wlo[i] = weight_[packed_idx * 2];
      whi[i] = weight_[packed_idx * 2 + 1];
      float lo = pio::lo(value);
      float hi = pio::hi(value);
      sum += lo + hi;
      sum_sq += lo * lo + hi * hi;
    }

    sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
    sum_sq =
        sycl::reduce_over_group(it.get_group(), sum_sq, sycl::plus<float>());
    float mean = sum / (float)NORM_DIM;
    float inv_std = sycl::rsqrt(
        sycl::fmax(sum_sq / (float)NORM_DIM - mean * mean, 0.0f) + epsilon_);

#pragma unroll
    for (int i = 0; i < N; i++) {
      float lo = (pio::lo(values[i]) - mean) * inv_std * wlo[i];
      float hi = (pio::hi(values[i]) - mean) * inv_std * whi[i];
      output[i * WG + tid] = pio::pack(lo, hi);
    }
  }

 private:
  const uint8_t* input_buf_;
  uint8_t* output_buf_;
  const float* weight_;
  float epsilon_;
};

template <typename Scalar>
void layer_norm_medium(
    sycl::queue& q,
    const uint8_t* input_buf,
    uint8_t* output_buf,
    const float* weight,
    int batch,
    int norm_dim,
    float epsilon) {
  auto launch = [&]<int NORM_DIM, int WG>() {
    q.submit([&](sycl::handler& h) {
      layer_norm_medium_kernel<Scalar, NORM_DIM, WG> kfn(
          input_buf, output_buf, weight, epsilon);
      h.parallel_for(sycl::nd_range<1>((int64_t)batch * WG, WG), kfn);
    });
  };

  if (norm_dim == 1024)
    launch.template operator()<1024, 256>();
  else if (norm_dim == 2048)
    launch.template operator()<2048, 256>();
  else if (norm_dim == 4096)
    launch.template operator()<4096, 256>();
  else if (norm_dim == 7168)
    launch.template operator()<7168, 256>();
  else if (norm_dim == 8192)
    launch.template operator()<8192, 256>();
  else if (norm_dim == 16384)
    launch.template operator()<16384, 1024>();
  else if (norm_dim == 32768)
    launch.template operator()<32768, 1024>();
  else
    TORCH_CHECK(false, "layer_norm: unsupported norm_dim ", norm_dim);
}

// norm_dim == 131072 keeps only half of the row in registers; the other
// half is staged through SLM (each lane only ever reads what it wrote, so
// no barrier is required) to bound register pressure.
template <typename Scalar>
class layer_norm_131072_kernel {
 public:
  layer_norm_131072_kernel(
      const uint8_t* input_buf,
      uint8_t* output_buf,
      const float* weight,
      float epsilon,
      sycl::local_accessor<uint32_t, 1> input_slm)
      : input_buf_(input_buf),
        output_buf_(output_buf),
        weight_(weight),
        epsilon_(epsilon),
        input_slm_(input_slm) {}

  void operator()(sycl::nd_item<1> it) const {
    using pio = LayerNormPacked16<Scalar>;
    constexpr int NORM_DIM = 131072;
    constexpr int PACKED = NORM_DIM / 2;
    constexpr int WG = 1024;
    constexpr int N = PACKED / WG;
    constexpr int HALF_N = N / 2;
    constexpr int SLM_PACKED = PACKED / 2;

    int row = it.get_group_linear_id();
    int tid = it.get_local_linear_id();
    const uint32_t* input = reinterpret_cast<const uint32_t*>(input_buf_) +
        (int64_t)row * PACKED;
    uint32_t* output = reinterpret_cast<uint32_t*>(output_buf_) +
        (int64_t)row * PACKED;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    uint32_t register_values[HALF_N];
#pragma unroll
    for (int i = 0; i < HALF_N; i++) {
      int packed_idx = i * WG + tid;
      uint32_t value = input[packed_idx];
      register_values[i] = value;
      float lo = pio::lo(value);
      float hi = pio::hi(value);
      sum += lo + hi;
      sum_sq += lo * lo + hi * hi;
    }

#pragma unroll
    for (int i = 0; i < HALF_N; i++) {
      int slm_idx = i * WG + tid;
      uint32_t value = input[SLM_PACKED + slm_idx];
      float lo = pio::lo(value);
      float hi = pio::hi(value);
      sum += lo + hi;
      sum_sq += lo * lo + hi * hi;
      input_slm_[slm_idx] = value;
    }

    sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
    sum_sq =
        sycl::reduce_over_group(it.get_group(), sum_sq, sycl::plus<float>());
    float mean = sum / (float)NORM_DIM;
    float inv_std = sycl::rsqrt(
        sycl::fmax(sum_sq / (float)NORM_DIM - mean * mean, 0.0f) + epsilon_);

#pragma unroll
    for (int i = 0; i < HALF_N; i++) {
      int packed_idx = i * WG + tid;
      uint32_t value = register_values[i];
      float lo =
          (pio::lo(value) - mean) * inv_std * weight_[packed_idx * 2];
      float hi = (pio::hi(value) - mean) * inv_std *
          weight_[packed_idx * 2 + 1];
      output[packed_idx] = pio::pack(lo, hi);
    }

#pragma unroll
    for (int i = 0; i < HALF_N; i++) {
      int slm_idx = i * WG + tid;
      int packed_idx = SLM_PACKED + slm_idx;
      uint32_t value = input_slm_[slm_idx];
      float lo =
          (pio::lo(value) - mean) * inv_std * weight_[packed_idx * 2];
      float hi = (pio::hi(value) - mean) * inv_std *
          weight_[packed_idx * 2 + 1];
      output[packed_idx] = pio::pack(lo, hi);
    }
  }

 private:
  const uint8_t* input_buf_;
  uint8_t* output_buf_;
  const float* weight_;
  float epsilon_;
  sycl::local_accessor<uint32_t, 1> input_slm_;
};

template <typename Scalar>
void layer_norm_131072(
    sycl::queue& q,
    const uint8_t* input_buf,
    uint8_t* output_buf,
    const float* weight,
    int batch,
    float epsilon) {
  constexpr int NORM_DIM = 131072;
  constexpr int PACKED = NORM_DIM / 2;
  constexpr int WG = 1024;
  constexpr int SLM_PACKED = PACKED / 2;

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> input_slm(
        sycl::range<1>(SLM_PACKED), h);
    layer_norm_131072_kernel<Scalar> kfn(
        input_buf, output_buf, weight, epsilon, input_slm);
    h.parallel_for(sycl::nd_range<1>((int64_t)batch * WG, WG), kfn);
  });
}

// ==========================================================================
// Large kernel: 1 work-group per row (norm_dim in {65536, 131072}).
// ==========================================================================
template <typename Scalar, int NORM_DIM, int WG>
class layer_norm_large_kernel {
 public:
  layer_norm_large_kernel(
      const uint8_t* input_buf,
      uint8_t* output_buf,
      const float* weight,
      float epsilon)
      : input_buf_(input_buf),
        output_buf_(output_buf),
        weight_(weight),
        epsilon_(epsilon) {}

  void operator()(sycl::nd_item<1> it) const {
    using pio = LayerNormPacked16<Scalar>;
    constexpr int PACKED = NORM_DIM / 2;
    constexpr int N = PACKED / WG;
    int row = it.get_group_linear_id();
    int tid = it.get_local_linear_id();
    const uint32_t* input = reinterpret_cast<const uint32_t*>(input_buf_) +
        (int64_t)row * PACKED;
    uint32_t* output = reinterpret_cast<uint32_t*>(output_buf_) +
        (int64_t)row * PACKED;

    uint32_t values[N];
    float sum = 0.0f;
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < N; i++) {
      int packed_idx = i * WG + tid;
      values[i] = input[packed_idx];
    }
#pragma unroll
    for (int i = 0; i < N; i++) {
      uint32_t value = values[i];
      float lo = pio::lo(value);
      float hi = pio::hi(value);
      sum += lo + hi;
      sum_sq += lo * lo + hi * hi;
    }

    sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
    sum_sq =
        sycl::reduce_over_group(it.get_group(), sum_sq, sycl::plus<float>());
    float mean = sum / (float)NORM_DIM;
    float inv_std = sycl::rsqrt(
        sycl::fmax(sum_sq / (float)NORM_DIM - mean * mean, 0.0f) + epsilon_);

#pragma unroll
    for (int i = 0; i < N; i++) {
      int packed_idx = i * WG + tid;
      float lo =
          (pio::lo(values[i]) - mean) * inv_std * weight_[packed_idx * 2];
      float hi = (pio::hi(values[i]) - mean) * inv_std *
          weight_[packed_idx * 2 + 1];
      output[i * WG + tid] = pio::pack(lo, hi);
    }
  }

 private:
  const uint8_t* input_buf_;
  uint8_t* output_buf_;
  const float* weight_;
  float epsilon_;
};

template <typename Scalar>
void layer_norm_large(
    sycl::queue& q,
    const uint8_t* input_buf,
    uint8_t* output_buf,
    const float* weight,
    int batch,
    int norm_dim,
    float epsilon) {
  auto launch = [&]<int NORM_DIM, int WG>() {
    q.submit([&](sycl::handler& h) {
      layer_norm_large_kernel<Scalar, NORM_DIM, WG> kfn(
          input_buf, output_buf, weight, epsilon);
      h.parallel_for(sycl::nd_range<1>((int64_t)batch * WG, WG), kfn);
    });
  };

  if (norm_dim == 65536)
    launch.template operator()<65536, 1024>();
  else if (norm_dim == 131072)
    layer_norm_131072<Scalar>(
        q, input_buf, output_buf, weight, batch, epsilon);
  else
    TORCH_CHECK(false, "layer_norm: unsupported norm_dim ", norm_dim);
}

// ==========================================================================
// float32 path: no 16-bit packing needed, straightforward per-lane
// reduction. Mirrors the norm_dim specialisations above.
// ==========================================================================
class layer_norm_fp32_large_kernel {
 public:
  layer_norm_fp32_large_kernel(
      const float* input,
      float* output,
      const float* weight,
      float epsilon)
      : input_(input), output_(output), weight_(weight), epsilon_(epsilon) {}

  void operator()(sycl::nd_item<1> it) const {
    constexpr int NORM_DIM = 131072;
    constexpr int WG = 1024;
    constexpr int N = NORM_DIM / WG;
    int row = it.get_group_linear_id();
    int tid = it.get_local_linear_id();
    const float* row_input = input_ + (int64_t)row * NORM_DIM;
    float* row_output = output_ + (int64_t)row * NORM_DIM;

    float sum = 0.0f;
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < N; i++) {
      float value = row_input[i * WG + tid];
      sum += value;
      sum_sq += value * value;
    }

    sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
    sum_sq =
        sycl::reduce_over_group(it.get_group(), sum_sq, sycl::plus<float>());
    float mean = sum / (float)NORM_DIM;
    float inv_std = sycl::rsqrt(
        sycl::fmax(sum_sq / (float)NORM_DIM - mean * mean, 0.0f) + epsilon_);

#pragma unroll
    for (int i = 0; i < N; i++) {
      int index = i * WG + tid;
      row_output[index] =
          (row_input[index] - mean) * inv_std * weight_[index];
    }
  }

 private:
  const float* input_;
  float* output_;
  const float* weight_;
  float epsilon_;
};

inline void layer_norm_fp32_large(
    sycl::queue& q,
    const float* input,
    float* output,
    const float* weight,
    int batch,
    float epsilon) {
  constexpr int WG = 1024;
  q.submit([&](sycl::handler& h) {
    layer_norm_fp32_large_kernel kfn(input, output, weight, epsilon);
    h.parallel_for(sycl::nd_range<1>((int64_t)batch * WG, WG), kfn);
  });
}

template <int NORM_DIM, int WG>
class layer_norm_non_packed_dtype_kernel {
 public:
  layer_norm_non_packed_dtype_kernel(
      const float* input,
      float* output,
      const float* weight,
      float epsilon)
      : input_(input), output_(output), weight_(weight), epsilon_(epsilon) {}

  void operator()(sycl::nd_item<1> it) const {
    constexpr int N = NORM_DIM / WG;
    int row = it.get_group_linear_id();
    int tid = it.get_local_linear_id();
    const float* row_input = input_ + (int64_t)row * NORM_DIM;
    float* row_output = output_ + (int64_t)row * NORM_DIM;

    float values[N];
    float sum = 0.0f;
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < N; i++) {
      int index = i * WG + tid;
      values[i] = row_input[index];
      sum += values[i];
      sum_sq += values[i] * values[i];
    }

    sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
    sum_sq =
        sycl::reduce_over_group(it.get_group(), sum_sq, sycl::plus<float>());
    float mean = sum / (float)NORM_DIM;
    float inv_std = sycl::rsqrt(
        sycl::fmax(sum_sq / (float)NORM_DIM - mean * mean, 0.0f) + epsilon_);

#pragma unroll
    for (int i = 0; i < N; i++) {
      int index = i * WG + tid;
      row_output[index] = (values[i] - mean) * inv_std * weight_[index];
    }
  }

 private:
  const float* input_;
  float* output_;
  const float* weight_;
  float epsilon_;
};

inline void layer_norm_non_packed_dtype(
    sycl::queue& q,
    const float* input,
    float* output,
    const float* weight,
    int batch,
    int norm_dim,
    float epsilon) {
  auto launch = [&]<int NORM_DIM, int WG>() {
    q.submit([&](sycl::handler& h) {
      layer_norm_non_packed_dtype_kernel<NORM_DIM, WG> kfn(
          input, output, weight, epsilon);
      h.parallel_for(sycl::nd_range<1>((int64_t)batch * WG, WG), kfn);
    });
  };

  switch (norm_dim) {
    case 128: launch.template operator()<128, 32>(); break;
    case 256: launch.template operator()<256, 32>(); break;
    case 512: launch.template operator()<512, 32>(); break;
    case 1024: launch.template operator()<1024, 256>(); break;
    case 2048: launch.template operator()<2048, 256>(); break;
    case 4096: launch.template operator()<4096, 256>(); break;
    // NOTE: the xpu-perf reference kernel omits this case (norm_dim==7168
    // is only handled for the half/bfloat16 medium path there), which would
    // make the fp32 path throw despite the op's shape check allowing it.
    // Added here for parity with layer_norm_medium.
    case 7168: launch.template operator()<7168, 256>(); break;
    case 8192: launch.template operator()<8192, 256>(); break;
    case 16384: launch.template operator()<16384, 1024>(); break;
    case 32768: launch.template operator()<32768, 1024>(); break;
    case 65536: launch.template operator()<65536, 1024>(); break;
    case 131072:
      layer_norm_fp32_large(q, input, output, weight, batch, epsilon);
      break;
    default: TORCH_CHECK(false, "layer_norm: unsupported norm_dim ", norm_dim);
  }
}

template <typename scalar_t>
void launch_layer_norm_kernel(
    torch::Tensor& out,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int batch = static_cast<int>(input.size(0));
  const int norm_dim = static_cast<int>(input.size(1));
  const float* weight_ptr = weight.const_data_ptr<float>();
  auto& q = vllm::xpu::vllmGetQueue();

  if constexpr (sizeof(sycl_t) == 4) {
    layer_norm_non_packed_dtype(
        q,
        input.const_data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        weight_ptr,
        batch,
        norm_dim,
        epsilon);
  } else {
    const uint8_t* input_buf =
        reinterpret_cast<const uint8_t*>(input.const_data_ptr<scalar_t>());
    uint8_t* output_buf =
        reinterpret_cast<uint8_t*>(out.data_ptr<scalar_t>());
    if (norm_dim <= 512)
      layer_norm_small<sycl_t>(
          q, input_buf, output_buf, weight_ptr, batch, norm_dim, epsilon);
    else if (norm_dim <= 32768)
      layer_norm_medium<sycl_t>(
          q, input_buf, output_buf, weight_ptr, batch, norm_dim, epsilon);
    else
      layer_norm_large<sycl_t>(
          q, input_buf, output_buf, weight_ptr, batch, norm_dim, epsilon);
  }
}

}  // namespace vllm

// Out-of-place weighted LayerNorm (no bias): out = (x - mean) / sqrt(var +
// eps) * weight, reduced over the last dimension of a 2D [batch, norm_dim]
// input. weight is always float32; input/output share the same dtype
// (half/bfloat16/float32).
void layer_norm(
    torch::Tensor& out,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  CHECK_DEVICE(input);
  CHECK_DEVICE(out);
  TORCH_CHECK(
      input.device() == out.device(),
      "input/output must be on the same device");
  CHECK_DEVICE(weight);
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16 ||
          input.scalar_type() == at::ScalarType::Float,
      "input must be half, bfloat16, or float32");
  TORCH_CHECK(
      input.dim() == 2 && out.dim() == 2,
      "input/output must be 2D [batch, norm_dim]");
  TORCH_CHECK(
      input.size(0) == out.size(0) && input.size(1) == out.size(1),
      "input/output shape mismatch");
  TORCH_CHECK(
      input.is_contiguous() && out.is_contiguous(),
      "input/output must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == out.scalar_type(), "input/output dtype mismatch");
  TORCH_CHECK(
      input.size(0) > 0 && input.size(1) > 0, "batch/norm_dim must be positive");

  const int64_t norm_dim = input.size(1);
  TORCH_CHECK(
      norm_dim >= 128 && norm_dim <= 131072,
      "norm_dim must be in [128, 131072]");
  TORCH_CHECK(
      norm_dim == 7168 || (norm_dim & (norm_dim - 1)) == 0,
      "norm_dim must be 7168 or a power of two");

  TORCH_CHECK(
      input.device() == weight.device(),
      "weight must be on the same device as input");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float,
              "weight must be float32");
  TORCH_CHECK(
      weight.dim() == 1 && weight.size(0) == norm_dim,
      "weight shape must be (norm_dim,)");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "layer_norm", [&] {
    vllm::launch_layer_norm_kernel<scalar_t>(
        out, input, weight, static_cast<float>(epsilon));
  });
}
