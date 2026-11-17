// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <sycl/sycl.hpp>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <cstdint>
#include <limits>

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"
#include "quantization/fp8/quant_utils.h"
#include "quantization/fp4/mxfp4_quant.h"
#include "quantization/mx_quant_utils.h"
#include "quantization/utils.h"
#include <ATen/DeviceGuard.h>

namespace vllm {
template <typename scalar_t>
struct alignas(8) vec4_t {
  scalar_t val[4];
};

template <typename scalar_t, typename out_t, bool has_residual, int VEC_SIZE>
class rms_norm_dynamic_per_token_quant_kernel {
 public:
  rms_norm_dynamic_per_token_quant_kernel(
      out_t* __restrict__ out_,
      scalar_t* __restrict__ residual_,
      const scalar_t* __restrict__ input_,
      const scalar_t* __restrict__ weight_,
      const float* __restrict__ scale_ub_,
      float* __restrict__ scales_,
      const float epsilon_,
      const int hidden_size_)
      : out(out_),
        residual(residual_),
        input(input_),
        weight(weight_),
        scale_ub(scale_ub_),
        scales(scales_),
        epsilon(epsilon_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    using vec_t = vllm::xpu::aligned_vec<scalar_t, VEC_SIZE>;
    using out_vec_t = vllm::xpu::aligned_vec<out_t, VEC_SIZE>;

    const int tid = item.get_local_id(0);
    const int local_range = item.get_local_range(0);
    const int64_t token_idx = item.get_group(0);

    // s_local[0] = inv_rms,  s_local[1] = scale
    auto& s_local =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2]>(
            item.get_group());

    const scalar_t* token_input = input + token_idx * hidden_size;
    out_t* token_output = out + token_idx * hidden_size;
    scalar_t* token_residual = nullptr;
    if constexpr (has_residual) {
      token_residual = residual + token_idx * hidden_size;
    }

    // The row is walked in units of VEC_SIZE elements so that every access is
    // a full 16-byte load/store instead of a 2-byte one.
    const int num_vec = hidden_size / VEC_SIZE;
    const auto* v_in = reinterpret_cast<const vec_t*>(token_input);
    const auto* v_weight = reinterpret_cast<const vec_t*>(weight);
    auto* v_out = reinterpret_cast<out_vec_t*>(token_output);
    auto* v_residual = reinterpret_cast<vec_t*>(token_residual);

    // Pass 1: optional residual add + compute variance
    float variance = 0.0f;
    for (int i = tid; i < num_vec; i += local_range) {
      vec_t xv = v_in[i];
      if constexpr (has_residual) {
        vec_t rv = v_residual[i];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          // Store the dtype-rounded sum, then read it back below so the
          // variance matches what pass 2 and 3 will see.
          xv[j] = static_cast<scalar_t>(
              static_cast<float>(xv[j]) + static_cast<float>(rv[j]));
        }
        v_residual[i] = xv;
      }
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float x = static_cast<float>(xv[j]);
        variance += x * x;
      }
    }
    variance = sycl::reduce_over_group(
        item.get_group(), variance, sycl::plus<float>());
    if (tid == 0) {
      s_local[0] = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    sycl::group_barrier(item.get_group());
    const float inv_rms = s_local[0];

    const vec_t* v_src =
        has_residual ? const_cast<const vec_t*>(v_residual) : v_in;

    // Pass 2: compute max |norm(x)| across the row -> token scale
    float absmax = 0.0f;
    for (int i = tid; i < num_vec; i += local_range) {
      vec_t xv = v_src[i];
      vec_t wv = v_weight[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float norm_x =
            static_cast<float>(xv[j]) * inv_rms * static_cast<float>(wv[j]);
        absmax = sycl::max(absmax, sycl::fabs(norm_x));
      }
    }
    absmax = sycl::reduce_over_group(
        item.get_group(), absmax, sycl::maximum<float>());

    if (tid == 0) {
      float computed_scale;
      if constexpr (std::is_same_v<out_t, int8_t>) {
        computed_scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
      } else {
        // FP8
        const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);
        const float clamped_absmax =
            (scale_ub != nullptr) ? sycl::min(absmax, *scale_ub) : absmax;
        computed_scale = sycl::max(
            clamped_absmax / fp8_max, fp8::min_scaling_factor<out_t>::val());
      }
      s_local[1] = computed_scale;
      scales[token_idx] = computed_scale;
    }
    sycl::group_barrier(item.get_group());
    const float inv_scale = 1.0f / s_local[1];

    // Pass 3: normalize and quantize
    for (int i = tid; i < num_vec; i += local_range) {
      vec_t xv = v_src[i];
      vec_t wv = v_weight[i];
      out_vec_t ov;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float norm_x =
            static_cast<float>(xv[j]) * inv_rms * static_cast<float>(wv[j]);
        const float q = norm_x * inv_scale;

        if constexpr (std::is_same_v<out_t, int8_t>) {
          ov[j] = static_cast<int8_t>(
              sycl::max(sycl::min(sycl::rint(q), 127.0f), -128.0f));
        } else {
          // FP8
          const float fp8_max =
              static_cast<float>(fp8::quant_type_max_v<out_t>);
          ov[j] =
              static_cast<out_t>(sycl::max(sycl::min(q, fp8_max), -fp8_max));
        }
      }
      v_out[i] = ov;
    }
  }

 private:
  out_t* __restrict__ out;
  scalar_t* __restrict__ residual;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ weight;
  const float* __restrict__ scale_ub;
  float* __restrict__ scales;
  const float epsilon;
  const int hidden_size;
};

// ue8m0 is a uniform runtime switch to keep the instantiation count (and
// binary size) down.
template <typename scalar_t, typename out_t, bool has_residual, int CPT>
class rms_norm_per_block_quant_kernel {
 public:
  rms_norm_per_block_quant_kernel(
      out_t* __restrict__ out_,
      scalar_t* __restrict__ residual_,
      const scalar_t* __restrict__ input_,
      const scalar_t* __restrict__ weight_,
      float* __restrict__ scales_,
      const float epsilon_,
      const int hidden_size_,
      const int group_size_,
      const int num_tokens_,
      const int64_t scale_stride_token_,
      const int64_t scale_stride_group_,
      const int64_t input_stride_,
      const bool scale_ue8m0_,
      sycl::local_accessor<scalar_t, 1> row_smem_)
      : out(out_),
        residual(residual_),
        input(input_),
        weight(weight_),
        scales(scales_),
        epsilon(epsilon_),
        hidden_size(hidden_size_),
        group_size(group_size_),
        num_tokens(num_tokens_),
        scale_stride_token(scale_stride_token_),
        scale_stride_group(scale_stride_group_),
        input_stride(input_stride_),
        scale_ue8m0(scale_ue8m0_),
        row_smem(row_smem_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    const int tid = item.get_local_id(0);
    const int local_range = item.get_local_range(0);
    const int64_t token_idx = item.get_group(0);

    auto& s_local =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(
            item.get_group());

    const scalar_t* token_input = input + token_idx * input_stride;
    out_t* token_output = out + token_idx * hidden_size;
    scalar_t* token_residual = nullptr;
    if constexpr (has_residual) {
      token_residual = residual + token_idx * hidden_size;
    }

    scalar_t* srow =
        row_smem.template get_multi_ptr<sycl::access::decorated::no>().get();

    // Pass 1: optional residual add + compute full-row variance.
    constexpr int VEC = 16 / sizeof(scalar_t);
    const int num_vec = hidden_size / VEC;
    float variance = 0.0f;
    const auto* vin =
        reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(token_input);
    auto* vsrow = reinterpret_cast<vec_n_t<scalar_t, VEC>*>(srow);
    if constexpr (has_residual) {
      auto* vres = reinterpret_cast<vec_n_t<scalar_t, VEC>*>(token_residual);
      for (int i = tid; i < num_vec; i += local_range) {
        vec_n_t<scalar_t, VEC> xi = vin[i];
        vec_n_t<scalar_t, VEC> ri = vres[i];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          float x =
              static_cast<float>(xi.val[k]) + static_cast<float>(ri.val[k]);
          ri.val[k] = static_cast<scalar_t>(x);
          float rx = static_cast<float>(ri.val[k]);
          variance += rx * rx;
        }
        vres[i] = ri;
        vsrow[i] = ri;
      }
    } else {
      for (int i = tid; i < num_vec; i += local_range) {
        vec_n_t<scalar_t, VEC> xi = vin[i];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          float x = static_cast<float>(xi.val[k]);
          variance += x * x;
        }
        vsrow[i] = xi;
      }
    }
    variance = sycl::reduce_over_group(
        item.get_group(), variance, sycl::plus<float>());
    if (tid == 0) {
      s_local[0] = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    sycl::group_barrier(item.get_group());
    const float inv_rms = s_local[0];

    auto sg = item.get_sub_group();
    const int lane = sg.get_local_id()[0];
    const int chunks_per_group = group_size / VEC;
    const int lanes_per_group = chunks_per_group / CPT;
    const int threads_per_row = num_vec / CPT;
    const auto* vweight =
        reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(weight);
    const auto* vsrow_r = reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(srow);
    auto* vout = reinterpret_cast<vec_n_t<out_t, VEC>*>(token_output);

    for (int base = 0; base < threads_per_row; base += local_range) {
      const int t = base + tid;
      const bool active = t < threads_per_row;
      const int c0 = t * CPT;

      float nrm[CPT][VEC];
      float lane_absmax = 0.0f;
      if (active) {
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          vec_n_t<scalar_t, VEC> xv = vsrow_r[c0 + j];
          vec_n_t<scalar_t, VEC> wv = vweight[c0 + j];
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            const float nx = static_cast<float>(xv.val[k]) * inv_rms *
                             static_cast<float>(wv.val[k]);
            nrm[j][k] = nx;
            lane_absmax = sycl::max(lane_absmax, sycl::fabs(nx));
          }
        }
      }

      float gmax = lane_absmax;
#pragma unroll
      for (int off = 1; off < lanes_per_group; off <<= 1) {
        gmax = sycl::max(gmax, sycl::permute_group_by_xor(sg, gmax, off));
      }

      float group_scale;
      if constexpr (std::is_same_v<out_t, int8_t>) {
        group_scale = (gmax > 0.0f) ? (gmax / 127.0f) : 1.0f;
      } else {
        const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);
        group_scale =
            sycl::max(gmax / fp8_max, fp8::min_scaling_factor<out_t>::val());
        if (scale_ue8m0) {
          group_scale = sycl::exp2(
              sycl::ceil(
                  sycl::log2(sycl::fmax(sycl::fabs(group_scale), 1e-10f))));
        }
      }

      if (active && (lanes_per_group <= 1 || (lane % lanes_per_group) == 0)) {
        const int g_idx = c0 / chunks_per_group;
        const int64_t scale_idx =
            token_idx * scale_stride_token +
            static_cast<int64_t>(g_idx) * scale_stride_group;
        scales[scale_idx] = group_scale;
      }

      if (active) {
        const float inv_scale = 1.0f / group_scale;
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          vec_n_t<out_t, VEC> o;
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            const float q = nrm[j][k] * inv_scale;
            if constexpr (std::is_same_v<out_t, int8_t>) {
              o.val[k] = static_cast<int8_t>(
                  sycl::max(sycl::min(sycl::rint(q), 127.0f), -128.0f));
            } else {
              const float fp8_max =
                  static_cast<float>(fp8::quant_type_max_v<out_t>);
              o.val[k] = static_cast<out_t>(
                  sycl::max(sycl::min(q, fp8_max), -fp8_max));
            }
          }
          vout[c0 + j] = o;
        }
      }
    }
  }

 private:
  out_t* __restrict__ out;
  scalar_t* __restrict__ residual;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ weight;
  float* __restrict__ scales;
  const float epsilon;
  const int hidden_size;
  const int group_size;
  const int num_tokens;
  const int64_t scale_stride_token;
  const int64_t scale_stride_group;
  const int64_t input_stride;
  const bool scale_ue8m0;
  sycl::local_accessor<scalar_t, 1> row_smem;
};

template <typename scalar_t, bool has_residual>
class rms_norm_mxfp4_quant_kernel {
 public:
  rms_norm_mxfp4_quant_kernel(
      uint8_t* __restrict__ out_,
      scalar_t* __restrict__ residual_,
      const scalar_t* __restrict__ input_,
      const scalar_t* __restrict__ weight_,
      float* __restrict__ scales_,
      const float epsilon_,
      const int hidden_size_,
      const int group_size_,
      const int64_t scale_stride_token_,
      const int64_t scale_stride_group_,
      sycl::local_accessor<scalar_t, 1> row_smem_)
      : out(out_),
        residual(residual_),
        input(input_),
        weight(weight_),
        scales(scales_),
        epsilon(epsilon_),
        hidden_size(hidden_size_),
        group_size(group_size_),
        scale_stride_token(scale_stride_token_),
        scale_stride_group(scale_stride_group_),
        row_smem(row_smem_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    const int tid = item.get_local_id(0);
    const int local_range = item.get_local_range(0);
    const int64_t token_idx = item.get_group(0);

    auto& s_local =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(
            item.get_group());

    const scalar_t* token_input = input + token_idx * hidden_size;
    uint8_t* token_output = out + token_idx * (hidden_size / 2);
    scalar_t* token_residual = nullptr;
    if constexpr (has_residual) {
      token_residual = residual + token_idx * hidden_size;
    }

    scalar_t* srow =
        row_smem.template get_multi_ptr<sycl::access::decorated::no>().get();

    constexpr int VEC = 16 / sizeof(scalar_t);
    const int num_vec = hidden_size / VEC;
    float variance = 0.0f;
    const auto* vin =
        reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(token_input);
    auto* vsrow = reinterpret_cast<vec_n_t<scalar_t, VEC>*>(srow);
    if constexpr (has_residual) {
      auto* vres = reinterpret_cast<vec_n_t<scalar_t, VEC>*>(token_residual);
      for (int i = tid; i < num_vec; i += local_range) {
        vec_n_t<scalar_t, VEC> xi = vin[i];
        vec_n_t<scalar_t, VEC> ri = vres[i];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          float x =
              static_cast<float>(xi.val[k]) + static_cast<float>(ri.val[k]);
          ri.val[k] = static_cast<scalar_t>(x);
          float rx = static_cast<float>(ri.val[k]);
          variance += rx * rx;
        }
        vres[i] = ri;
        vsrow[i] = ri;
      }
    } else {
      for (int i = tid; i < num_vec; i += local_range) {
        vec_n_t<scalar_t, VEC> xi = vin[i];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          float x = static_cast<float>(xi.val[k]);
          variance += x * x;
        }
        vsrow[i] = xi;
      }
    }
    variance = sycl::reduce_over_group(
        item.get_group(), variance, sycl::plus<float>());
    if (tid == 0) {
      s_local[0] = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    sycl::group_barrier(item.get_group());
    const float inv_rms = s_local[0];

    constexpr float FP4_MAX = vllm::mxfp4::FP4_MAX;

    auto sg = item.get_sub_group();
    const int lane = sg.get_local_id()[0];
    const int lanes_per_group = group_size / VEC;  // 32 / 8 = 4
    const auto* vweight =
        reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(weight);
    const auto* vsrow_r = reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(srow);
    auto* vout = reinterpret_cast<vec_n_t<uint8_t, VEC / 2>*>(token_output);

    for (int base = 0; base < num_vec; base += local_range) {
      const int c = base + tid;
      const bool active = c < num_vec;

      float nrm[VEC];
      float lane_absmax = 0.0f;
      if (active) {
        vec_n_t<scalar_t, VEC> xv = vsrow_r[c];
        vec_n_t<scalar_t, VEC> wv = vweight[c];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          const float nx = static_cast<float>(xv.val[k]) * inv_rms *
                           static_cast<float>(wv.val[k]);
          nrm[k] = nx;
          lane_absmax = sycl::max(lane_absmax, sycl::fabs(nx));
        }
      }

      float gmax = lane_absmax;
#pragma unroll
      for (int off = 1; off < lanes_per_group; off <<= 1) {
        gmax = sycl::max(gmax, sycl::permute_group_by_xor(sg, gmax, off));
      }

      float y_s = gmax / FP4_MAX;
      y_s = sycl::exp2(
          sycl::ceil(sycl::log2(sycl::fmax(sycl::fabs(y_s), epsilon))));

      if (active && (lane % lanes_per_group) == 0) {
        const int g_idx = c / lanes_per_group;
        const int64_t scale_idx =
            token_idx * scale_stride_token +
            static_cast<int64_t>(g_idx) * scale_stride_group;
        scales[scale_idx] = y_s;
      }

      if (active) {
        const float inv_scale = 1.0f / y_s;
        vec_n_t<uint8_t, VEC / 2> o;
#pragma unroll
        for (int k = 0; k < VEC; k += 2) {
          float q0 =
              sycl::fmax(-FP4_MAX, sycl::fmin(nrm[k] * inv_scale, FP4_MAX));
          float q1 =
              sycl::fmax(-FP4_MAX, sycl::fmin(nrm[k + 1] * inv_scale, FP4_MAX));
          const uint8_t fp4_lo = vllm::mxfp4::float_to_fp4_e2m1(q0);
          const uint8_t fp4_hi = vllm::mxfp4::float_to_fp4_e2m1(q1);
          o.val[k / 2] =
              static_cast<uint8_t>(((fp4_hi & 0x0Fu) << 4) | (fp4_lo & 0x0Fu));
        }
        vout[c] = o;
      }
    }
  }

 private:
  uint8_t* __restrict__ out;
  scalar_t* __restrict__ residual;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ weight;
  float* __restrict__ scales;
  const float epsilon;
  const int hidden_size;
  const int group_size;
  const int64_t scale_stride_token;
  const int64_t scale_stride_group;
  sycl::local_accessor<scalar_t, 1> row_smem;
};

// RMSNorm + MX quant to FP8, or packed FP4 when out_t is uint8_t.
template <
    typename scalar_t,
    typename out_t,
    bool has_residual,
    int VPL,
    bool COL_MAJOR>
class rms_norm_mx_quant_kernel {
 public:
  rms_norm_mx_quant_kernel(
      out_t* __restrict__ out_,
      scalar_t* __restrict__ residual_,
      const scalar_t* __restrict__ input_,
      const scalar_t* __restrict__ weight_,
      float* __restrict__ scales_,
      const float epsilon_,
      const int hidden_size_,
      const int input_stride_,
      const int num_tokens_)
      : out(out_),
        residual(residual_),
        input(input_),
        weight(weight_),
        scales(scales_),
        epsilon(epsilon_),
        hidden_size(hidden_size_),
        input_stride(input_stride_),
        num_tokens(num_tokens_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    constexpr int VEC = mx::kVec;
    constexpr int LPG = mx::kLanesPerGroup / VPL;
    constexpr bool kFp4 = std::is_same_v<out_t, uint8_t>;
    using vec_t = vec_n_t<scalar_t, VEC>;

    const int tid = item.get_local_id(0);
    const int64_t token_idx = item.get_group(0);
    const bool active = tid < hidden_size / (VEC * VPL);
    auto sg = item.get_sub_group();
    auto& partial =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            item.get_group());

    const int64_t row = token_idx * hidden_size;
    vec_t xv[VPL] = {};
    vec_t wv[VPL] = {};
    float x[VPL][VEC] = {};
    float variance = 0.0f;
    if (active) {
      const auto* vin =
          reinterpret_cast<const vec_t*>(input + token_idx * input_stride);
      const auto* vweight = reinterpret_cast<const vec_t*>(weight);
#pragma unroll
      for (int j = 0; j < VPL; ++j) {
        xv[j] = vin[tid * VPL + j];
        wv[j] = vweight[tid * VPL + j];
      }
      if constexpr (has_residual) {
        auto* vres = reinterpret_cast<vec_t*>(residual + row);
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
          vec_t r = vres[tid * VPL + j];
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            r.val[k] = static_cast<scalar_t>(
                static_cast<float>(xv[j].val[k]) +
                static_cast<float>(r.val[k]));
          }
          vres[tid * VPL + j] = r;
          xv[j] = r;
        }
      }
#pragma unroll
      for (int j = 0; j < VPL; ++j) {
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          x[j][k] = static_cast<float>(xv[j].val[k]);
          variance += x[j][k] * x[j][k];
        }
      }
    }

    variance = sycl::reduce_over_group(sg, variance, sycl::plus<float>());
    if (sg.leader()) partial[sg.get_group_linear_id()] = variance;
    sycl::group_barrier(item.get_group());
    const int num_sg = (hidden_size / (VEC * VPL) + 31) / 32;
    const int lane = sg.get_local_linear_id();
    variance = lane < num_sg ? partial[lane] : 0.0f;
    variance = sycl::reduce_over_group(sg, variance, sycl::plus<float>());
    const float inv_rms = sycl::rsqrt(variance / hidden_size + epsilon);

    float absmax = 0.0f;
#pragma unroll
    for (int j = 0; j < VPL; ++j) {
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        x[j][k] *= inv_rms * static_cast<float>(wv[j].val[k]);
        absmax = sycl::fmax(absmax, sycl::fabs(x[j][k]));
      }
    }
    absmax = mx::group_absmax(sg, absmax, LPG);

    float inv_scale;
    float y_s;
    using word_t = std::conditional_t<kFp4, uint32_t, mx::fp8x8_t>;
    vec_n_t<word_t, VPL> o;
    if constexpr (kFp4) {
      y_s = mxfp4::mxfp4_pow2_scale(absmax, epsilon, inv_scale);
#pragma unroll
      for (int j = 0; j < VPL; ++j) {
        o.val[j] = mxfp4::mxfp4_pack8<float>(x[j], inv_scale);
      }
    } else {
      y_s = mx::mxfp8_scale<out_t>(absmax, inv_scale);
#pragma unroll
      for (int j = 0; j < VPL; ++j) {
        o.val[j] = mx::mxfp8_pack8<out_t>(x[j], inv_scale);
      }
    }
    if (!active) return;

    if (tid % LPG == 0) {
      if constexpr (COL_MAJOR) {
        scales[(tid / LPG) * num_tokens + token_idx] = y_s;
      } else {
        scales[token_idx * (hidden_size / mx::kGroupSize) + tid / LPG] = y_s;
      }
    }
    constexpr int kOutPerElem = kFp4 ? 2 : 1;
    reinterpret_cast<vec_n_t<word_t, VPL>*>(
        reinterpret_cast<uint8_t*>(out) + row / kOutPerElem)[tid] = o;
  }

 private:
  out_t* __restrict__ out;
  scalar_t* __restrict__ residual;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ weight;
  float* __restrict__ scales;
  const float epsilon;
  const int hidden_size;
  const int input_stride;
  const int num_tokens;
};

template <
    typename scalar_t,
    typename out_t,
    bool has_residual,
    int HIDDEN,
    bool COL_MAJOR>
class rms_norm_mx_quant_short_kernel {
 public:
  rms_norm_mx_quant_short_kernel(
      out_t* __restrict__ out_,
      scalar_t* __restrict__ residual_,
      const scalar_t* __restrict__ input_,
      const scalar_t* __restrict__ weight_,
      float* __restrict__ scales_,
      const float epsilon_,
      const int num_rows_,
      const int input_stride_)
      : out(out_),
        residual(residual_),
        input(input_),
        weight(weight_),
        scales(scales_),
        epsilon(epsilon_),
        num_rows(num_rows_),
        input_stride(input_stride_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32),
        sycl::reqd_work_group_size(mx::kWorkGroupSize)]] (
          sycl::nd_item<1> item) const {
    constexpr int VEC = mx::kVec;
    constexpr int LANES_PER_ROW = HIDDEN / VEC;
    constexpr int GROUPS_PER_ROW = HIDDEN / mx::kGroupSize;
    constexpr bool kFp4 = std::is_same_v<out_t, uint8_t>;
    static_assert(
        LANES_PER_ROW >= mx::kLanesPerGroup && LANES_PER_ROW <= 32 &&
        (LANES_PER_ROW & (LANES_PER_ROW - 1)) == 0);
    using vec_t = vec_n_t<scalar_t, VEC>;

    const int v = mx::flat_index(item);
    const int row = v / LANES_PER_ROW;
    const int c = v % LANES_PER_ROW;
    const bool active = row < num_rows;
    auto sg = item.get_sub_group();

    vec_t xv = {};
    vec_t wv = {};
    float x[VEC] = {};
    float variance = 0.0f;
    if (active) {
      xv = reinterpret_cast<const vec_t*>(
          input + static_cast<int64_t>(row) * input_stride)[c];
      wv = reinterpret_cast<const vec_t*>(weight)[c];
      if constexpr (has_residual) {
        auto* vres = reinterpret_cast<vec_t*>(
            residual + static_cast<int64_t>(row) * HIDDEN);
        vec_t r = vres[c];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          r.val[k] = static_cast<scalar_t>(
              static_cast<float>(xv.val[k]) + static_cast<float>(r.val[k]));
        }
        vres[c] = r;
        xv = r;
      }
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        x[k] = static_cast<float>(xv.val[k]);
        variance += x[k] * x[k];
      }
    }
#pragma unroll
    for (int off = 1; off < LANES_PER_ROW; off <<= 1) {
      variance += sycl::permute_group_by_xor(sg, variance, off);
    }
    const float inv_rms = sycl::rsqrt(variance / HIDDEN + epsilon);

    float absmax = 0.0f;
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      x[k] *= inv_rms * static_cast<float>(wv.val[k]);
      absmax = sycl::fmax(absmax, sycl::fabs(x[k]));
    }
    absmax = mx::group_absmax(sg, absmax, mx::kLanesPerGroup);

    float inv_scale;
    float y_s;
    using word_t = std::conditional_t<kFp4, uint32_t, mx::fp8x8_t>;
    word_t o;
    if constexpr (kFp4) {
      y_s = mxfp4::mxfp4_pow2_scale(absmax, epsilon, inv_scale);
      o = mxfp4::mxfp4_pack8<float>(x, inv_scale);
    } else {
      y_s = mx::mxfp8_scale<out_t>(absmax, inv_scale);
      o = mx::mxfp8_pack8<out_t>(x, inv_scale);
    }
    if (!active) return;

    if (c % mx::kLanesPerGroup == 0) {
      const int g = c / mx::kLanesPerGroup;
      if constexpr (COL_MAJOR) {
        scales[g * num_rows + row] = y_s;
      } else {
        scales[row * GROUPS_PER_ROW + g] = y_s;
      }
    }
    reinterpret_cast<word_t*>(out)[v] = o;
  }

 private:
  out_t* __restrict__ out;
  scalar_t* __restrict__ residual;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ weight;
  float* __restrict__ scales;
  const float epsilon;
  const int num_rows;
  const int input_stride;
};

template <typename scalar_t, typename out_t, int VEC_SIZE>
class rms_norm_static_fp8_quant_kernel {
 public:
  rms_norm_static_fp8_quant_kernel(
      out_t* __restrict__ out_,
      const scalar_t* __restrict__ input_,
      const int input_stride_,
      const scalar_t* __restrict__ weight_,
      const float* __restrict__ scale_,
      const float epsilon_,
      const int hidden_size_)
      : out(out_),
        input(input_),
        input_stride(input_stride_),
        weight(weight_),
        scale(scale_),
        epsilon(epsilon_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    using vec_t = vllm::xpu::aligned_vec<scalar_t, VEC_SIZE>;

    const int tid = item.get_local_id(0);
    const int local_range = item.get_local_range(0);
    const int64_t token_idx = item.get_group(0);

    auto& s_variance =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
            item.get_group());

    const scalar_t* token_input = input + token_idx * input_stride;
    out_t* token_output = out + token_idx * hidden_size;
    const int nvec = hidden_size / VEC_SIZE;

    // Pass 1: compute variance — VEC_SIZE elements per work-item per iteration
    float variance = 0.0f;
    const auto* v_in = reinterpret_cast<const vec_t*>(token_input);
    for (int i = tid; i < nvec; i += local_range) {
      vec_t v = v_in[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(v[j]);
        variance += x * x;
      }
    }
    variance = sycl::reduce_over_group(
        item.get_group(), variance, sycl::plus<float>());
    if (tid == 0) {
      s_variance = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    sycl::group_barrier(item.get_group());
    const float inv_rms = s_variance;

    // Invert scale to avoid division
    const float scale_inv = 1.0f / (*scale);
    const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);

    // Pass 2: normalize, apply weight, quantize — same vectorization as Pass 1
    const auto* v_w = reinterpret_cast<const vec_t*>(weight);
    constexpr bool kPackStore = VEC_SIZE % 4 == 0;
    for (int i = tid; i < nvec; i += local_range) {
      vec_t src = v_in[i];
      vec_t wgt = v_w[i];
      PackedBytes<kPackStore ? VEC_SIZE : 4> packed;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(src[j]);
        // Weight multiply in fp32 (promoting weight) to match unfused path and
        // CUDA, applying a single rounding at store time.
        float norm_x = x * inv_rms * static_cast<float>(wgt[j]);
        float q = norm_x * scale_inv;
        const out_t o =
            static_cast<out_t>(sycl::max(sycl::min(q, fp8_max), -fp8_max));
        if constexpr (kPackStore) {
          packed.set(j, o.x);
        } else {
          token_output[i * VEC_SIZE + j] = o;
        }
      }
      if constexpr (kPackStore) {
        packed.store(token_output + i * VEC_SIZE);
      }
    }
  }

 private:
  out_t* __restrict__ out;
  const scalar_t* __restrict__ input;
  const int input_stride;
  const scalar_t* __restrict__ weight;
  const float* __restrict__ scale;
  const float epsilon;
  const int hidden_size;
};

template <typename scalar_t, typename out_t, int VEC_SIZE>
class fused_add_rms_norm_static_fp8_quant_kernel {
 public:
  fused_add_rms_norm_static_fp8_quant_kernel(
      out_t* __restrict__ out_,
      const scalar_t* __restrict__ input_,
      const int input_stride_,
      scalar_t* __restrict__ residual_,
      const scalar_t* __restrict__ weight_,
      const float* __restrict__ scale_,
      const float epsilon_,
      const int hidden_size_)
      : out(out_),
        input(input_),
        input_stride(input_stride_),
        residual(residual_),
        weight(weight_),
        scale(scale_),
        epsilon(epsilon_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    using vec_t = vllm::xpu::aligned_vec<scalar_t, VEC_SIZE>;

    const int tid = item.get_local_id(0);
    const int local_range = item.get_local_range(0);
    const int64_t token_idx = item.get_group(0);

    auto& s_variance =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
            item.get_group());

    const scalar_t* token_input = input + token_idx * input_stride;
    scalar_t* token_residual = residual + token_idx * hidden_size;
    out_t* token_output = out + token_idx * hidden_size;
    const int nvec = hidden_size / VEC_SIZE;

    // Pass 1: add residual + compute variance, VEC_SIZE elements per iteration
    float variance = 0.0f;
    const auto* v_in = reinterpret_cast<const vec_t*>(token_input);
    auto* v_res = reinterpret_cast<vec_t*>(token_residual);
    for (int i = tid; i < nvec; i += local_range) {
      vec_t inp = v_in[i];
      vec_t res = v_res[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        // Add in scalar_t precision to match fused_add_rms_norm kernel
        scalar_t z = inp[j] + res[j];
        res[j] = z;
        float xf = static_cast<float>(z);
        variance += xf * xf;
      }
      v_res[i] = res;  // write updated residual back
    }
    variance = sycl::reduce_over_group(
        item.get_group(), variance, sycl::plus<float>());
    if (tid == 0) {
      s_variance = sycl::rsqrt(variance / hidden_size + epsilon);
    }
    sycl::group_barrier(item.get_group());
    const float inv_rms = s_variance;

    // Invert scale to avoid division
    const float scale_inv = 1.0f / (*scale);
    const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);

    // Pass 2: normalize from residual, apply weight, quantize
    const auto* v_w = reinterpret_cast<const vec_t*>(weight);
    constexpr bool kPackStore = VEC_SIZE % 4 == 0;
    for (int i = tid; i < nvec; i += local_range) {
      vec_t res = v_res[i];
      vec_t wgt = v_w[i];
      PackedBytes<kPackStore ? VEC_SIZE : 4> packed;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(res[j]);
        // Weight multiply in fp32 (promoting weight) to match unfused path and
        // CUDA, applying a single rounding at store time.
        float norm_x = x * inv_rms * static_cast<float>(wgt[j]);
        float q = norm_x * scale_inv;
        const out_t o =
            static_cast<out_t>(sycl::max(sycl::min(q, fp8_max), -fp8_max));
        if constexpr (kPackStore) {
          packed.set(j, o.x);
        } else {
          token_output[i * VEC_SIZE + j] = o;
        }
      }
      if constexpr (kPackStore) {
        packed.store(token_output + i * VEC_SIZE);
      }
    }
  }

 private:
  out_t* __restrict__ out;
  const scalar_t* __restrict__ input;
  const int input_stride;
  scalar_t* __restrict__ residual;
  const scalar_t* __restrict__ weight;
  const float* __restrict__ scale;
  const float epsilon;
  const int hidden_size;
};

template <typename scalar_t, typename out_t>
void call_rms_norm_dynamic_per_token_quant_kernel(
    torch::Tensor& out,
    std::optional<torch::Tensor>& residual,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    std::optional<torch::Tensor> const& scale_ub,
    torch::Tensor& scales,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int hidden_size = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_size;

  // Walk the row in 16-byte vectors whenever it divides evenly; other hidden
  // sizes take the original scalar traversal (VEC_SIZE == 1).
  constexpr int kFullVec = 16 / sizeof(sycl_t);
  const int vec_size = hidden_size % kFullVec == 0 ? kFullVec : 1;

  auto* out_ptr = out.data_ptr<out_t>();
  auto* input_ptr = input.data_ptr<scalar_t>();
  auto* weight_ptr = weight.data_ptr<scalar_t>();
  auto* scales_ptr = scales.data_ptr<float>();
  const float* scale_ub_ptr =
      scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr;

  auto& queue = vllm::xpu::vllmGetQueue();
  const int eu_count = static_cast<int>(
      queue.get_device().get_info<sycl::info::device::max_compute_units>());
  const int block_size = static_cast<int>(fused_norm_quant_block_size(
      num_tokens, eu_count, hidden_size / vec_size, 4, true));

  sycl_t* residual_ptr =
      residual.has_value() ? (sycl_t*)residual->data_ptr<scalar_t>() : nullptr;

  auto launch_res = [&](auto vec_tag, auto res_tag) {
    constexpr int VS = decltype(vec_tag)::value;
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(num_tokens * block_size, block_size),
          rms_norm_dynamic_per_token_quant_kernel<
              sycl_t,
              out_t,
              decltype(res_tag)::value,
              VS>(
              out_ptr,
              residual_ptr,
              (const sycl_t*)input_ptr,
              (const sycl_t*)weight_ptr,
              scale_ub_ptr,
              scales_ptr,
              epsilon,
              hidden_size));
    });
  };

  auto launch = [&](auto vec_tag) {
    if (residual.has_value()) {
      launch_res(vec_tag, std::true_type{});
    } else {
      launch_res(vec_tag, std::false_type{});
    }
  };

  if (vec_size == kFullVec) {
    launch(std::integral_constant<int, kFullVec>{});
  } else {
    launch(std::integral_constant<int, 1>{});
  }
}

inline int64_t flat_row_stride(torch::Tensor const& input) {
  if (input.dim() < 2) return input.size(-1);
  for (int64_t d = 0; d + 2 < input.dim(); ++d) {
    TORCH_CHECK(
        input.size(d) == 1 ||
            input.stride(d) == input.stride(d + 1) * input.size(d + 1),
        "input rows must be uniformly strided over the leading dimensions");
  }
  return input.stride(-2);
}

template <typename scalar_t, typename out_t>
bool try_rms_norm_mx_quant(
    out_t* out_ptr,
    std::optional<torch::Tensor>& residual,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    float* scales_ptr,
    float epsilon,
    bool is_scale_transposed) {
  // Only 16-bit inputs get the dedicated MX kernels; fp32 rows take the
  // per-block / MXFP4 fallback kernels (binary size).
  if constexpr (sizeof(scalar_t) != 2) {
    return false;
  } else {
    using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
    constexpr int VEC = mx::kVec;
    const int hidden_size = input.size(-1);
    const int64_t input_stride = flat_row_stride(input);
    const int64_t num_tokens = input.numel() / hidden_size;
    const auto aligned = [](const void* p, int n) {
      return reinterpret_cast<uintptr_t>(p) % n == 0;
    };
    constexpr int kOutBytesPerVec = std::is_same_v<out_t, uint8_t> ? 4 : 8;
    if (hidden_size % mx::kGroupSize != 0 || input_stride % VEC != 0 ||
        input_stride > std::numeric_limits<int>::max() ||
        num_tokens > std::numeric_limits<int>::max() ||
        !aligned(input.data_ptr(), 16) || !aligned(weight.data_ptr(), 16) ||
        (residual.has_value() && !aligned(residual->data_ptr(), 16))) {
      return false;
    }
    auto& queue = vllm::xpu::vllmGetQueue();
    sycl_t* const residual_ptr =
        residual.has_value() ? reinterpret_cast<sycl_t*>(residual->data_ptr())
                             : nullptr;
    if ((hidden_size == 32 || hidden_size == 64 || hidden_size == 128 ||
         hidden_size == 256) &&
        aligned(out_ptr, kOutBytesPerVec) &&
        num_tokens * (hidden_size / VEC) <= std::numeric_limits<int>::max()) {
      if (num_tokens == 0) return true;
      const int64_t num_vecs = num_tokens * (hidden_size / VEC);
      const int64_t wg = mx::kWorkGroupSize;
      auto launch_s = [&](auto hidden_tag, auto res_tag, auto col_tag) {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>((num_vecs + wg - 1) / wg * wg, wg),
              rms_norm_mx_quant_short_kernel<
                  sycl_t,
                  out_t,
                  decltype(res_tag)::value,
                  decltype(hidden_tag)::value,
                  decltype(col_tag)::value>(
                  out_ptr,
                  residual_ptr,
                  reinterpret_cast<const sycl_t*>(input.data_ptr()),
                  reinterpret_cast<const sycl_t*>(weight.data_ptr()),
                  scales_ptr,
                  epsilon,
                  static_cast<int>(num_tokens),
                  static_cast<int>(input_stride)));
        });
      };
      auto launch_res = [&](auto hidden_tag, auto res_tag) {
        if (is_scale_transposed) {
          launch_s(hidden_tag, res_tag, std::true_type{});
        } else {
          launch_s(hidden_tag, res_tag, std::false_type{});
        }
      };
      auto launch_short = [&](auto hidden_tag) {
        if (residual.has_value()) {
          launch_res(hidden_tag, std::true_type{});
        } else {
          launch_res(hidden_tag, std::false_type{});
        }
      };
      switch (hidden_size) {
        case 32:
          launch_short(std::integral_constant<int, 32>{});
          break;
        case 64:
          launch_short(std::integral_constant<int, 64>{});
          break;
        case 128:
          launch_short(std::integral_constant<int, 128>{});
          break;
        default:
          launch_short(std::integral_constant<int, 256>{});
      }
      return true;
    }

    // Rows wider than 2 * VEC * 1024 fall back to the per-block kernel; a
    // VPL = 4 variant is not worth its binary size for such rare widths.
    int vpl = 1;
    while (vpl < 2 && hidden_size / (VEC * vpl) > 1024)
      vpl *= 2;
    if (vpl == 1 && !residual.has_value() && num_tokens >= 128) vpl = 2;
    const int lanes = hidden_size / (VEC * vpl);
    if (lanes > 1024 || !aligned(out_ptr, kOutBytesPerVec * vpl)) return false;
    if (num_tokens == 0) return true;
    const int64_t wg_size = (lanes + 31) / 32 * 32;

    auto launch_c = [&](auto vpl_tag, auto res_tag, auto col_tag) {
      constexpr int VPL = decltype(vpl_tag)::value;
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_tokens * wg_size, wg_size),
            rms_norm_mx_quant_kernel<
                sycl_t,
                out_t,
                decltype(res_tag)::value,
                VPL,
                decltype(col_tag)::value>(
                out_ptr,
                residual_ptr,
                reinterpret_cast<const sycl_t*>(input.data_ptr()),
                reinterpret_cast<const sycl_t*>(weight.data_ptr()),
                scales_ptr,
                epsilon,
                hidden_size,
                static_cast<int>(input_stride),
                static_cast<int>(num_tokens)));
      });
    };
    auto launch = [&](auto vpl_tag, auto res_tag) {
      if (is_scale_transposed) {
        launch_c(vpl_tag, res_tag, std::true_type{});
      } else {
        launch_c(vpl_tag, res_tag, std::false_type{});
      }
    };
    auto with_res = [&](auto vpl_tag) {
      if (residual.has_value()) {
        launch(vpl_tag, std::true_type{});
      } else {
        launch(vpl_tag, std::false_type{});
      }
    };
    if (vpl == 1) {
      with_res(std::integral_constant<int, 1>{});
    } else {
      with_res(std::integral_constant<int, 2>{});
    }
    return true;
  }
}

template <typename scalar_t, typename out_t>
void call_rms_norm_per_block_quant_kernel(
    torch::Tensor& out,
    std::optional<torch::Tensor>& residual,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor& scales,
    float epsilon,
    int group_size,
    bool is_scale_transposed,
    bool scale_ue8m0) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int hidden_size = input.size(-1);
  const int64_t input_stride = flat_row_stride(input);
  const int64_t num_tokens = input.numel() / hidden_size;
  const int num_groups = hidden_size / group_size;
  constexpr int vec_width = 16 / sizeof(scalar_t);  // matches kernel VEC
  const int num_chunks = hidden_size / vec_width;

  // Compute scale strides based on transposition
  const int64_t scale_stride_token =
      is_scale_transposed ? 1 : static_cast<int64_t>(num_groups);
  const int64_t scale_stride_group = is_scale_transposed ? num_tokens : 1LL;

  auto* out_ptr = out.data_ptr<out_t>();
  auto* input_ptr = input.data_ptr<scalar_t>();
  auto* weight_ptr = weight.data_ptr<scalar_t>();
  auto* scales_ptr = scales.data_ptr<float>();

  if constexpr (!std::is_same_v<out_t, int8_t>) {
    if (scale_ue8m0 && group_size == mx::kGroupSize &&
        try_rms_norm_mx_quant<scalar_t, out_t>(
            out_ptr,
            residual,
            input,
            weight,
            scales_ptr,
            epsilon,
            is_scale_transposed)) {
      return;
    }
  }

  auto& queue = vllm::xpu::vllmGetQueue();

  const int chunks_per_group = group_size / vec_width;
  const int eu_count = static_cast<int>(
      queue.get_device().get_info<sycl::info::device::max_compute_units>());

  // The int8 entry point always passes scale_ue8m0 = false.
  TORCH_CHECK(
      !(scale_ue8m0 && std::is_same_v<out_t, int8_t>),
      "scale_ue8m0 is not supported for int8 output");
  sycl_t* residual_ptr =
      residual.has_value() ? (sycl_t*)residual->data_ptr<scalar_t>() : nullptr;

  int cpt = 1;
  if constexpr (sizeof(scalar_t) == 2) {
    dispatch_quant_chunks(
        num_tokens, num_chunks, chunks_per_group, eu_count, [&](auto tag) {
          cpt = decltype(tag)::value;
        });
  }
  // The quant loop reduces a group within one 32-lane sub-group, so wide
  // groups need enough chunks per thread to fit (fp32 group 256 needs 64
  // lanes at CPT == 1).
  constexpr int kMaxLanesPerGroup = 32;
  if (chunks_per_group / cpt > kMaxLanesPerGroup) cpt = 4;
  TORCH_CHECK(
      chunks_per_group % cpt == 0 &&
          chunks_per_group / cpt <= kMaxLanesPerGroup,
      "rms_norm_per_block_quant: group_size ",
      group_size,
      " is too large for ",
      input.scalar_type(),
      " input");

  auto launch_res = [&](auto cpt_tag, auto res_tag) {
    // fp32 input only uses CPT == 1, or CPT == 4 for groups wider than a
    // sub-group; fold CPT == 2 onto 1 so it is not instantiated (binary size).
    constexpr int CPT = sizeof(scalar_t) != 2 && decltype(cpt_tag)::value == 2
                            ? 1
                            : decltype(cpt_tag)::value;
    // Only the variance pass reduces over the group; the quantized store loop
    // reduces within a sub-group, so it uses the barrier-free sizing rule.
    const int64_t block_size = fused_norm_quant_block_size(
        num_tokens, eu_count, num_chunks / CPT, 1, false);
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> row_smem(
          sycl::range<1>(hidden_size), cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(num_tokens * block_size, block_size),
          rms_norm_per_block_quant_kernel<
              sycl_t,
              out_t,
              decltype(res_tag)::value,
              CPT>(
              out_ptr,
              residual_ptr,
              (const sycl_t*)input_ptr,
              (const sycl_t*)weight_ptr,
              scales_ptr,
              epsilon,
              hidden_size,
              group_size,
              static_cast<int>(num_tokens),
              scale_stride_token,
              scale_stride_group,
              input_stride,
              scale_ue8m0,
              row_smem));
    });
  };
  auto launch = [&](auto cpt_tag) {
    if (residual_ptr != nullptr) {
      launch_res(cpt_tag, std::true_type{});
    } else {
      launch_res(cpt_tag, std::false_type{});
    }
  };

  if (cpt == 4) {
    launch(std::integral_constant<int, 4>{});
  } else if (cpt == 2) {
    launch(std::integral_constant<int, 2>{});
  } else {
    launch(std::integral_constant<int, 1>{});
  }
}

template <typename scalar_t, typename out_t>
void call_rms_norm_static_fp8_quant_kernel(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& scale,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int hidden_size = input.size(-1);
  const int input_stride = input.stride(-2);
  const int64_t num_tokens = input.numel() / hidden_size;

  // Match CUDA: smaller blocks when num_tokens is large for better occupancy
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;

  // Dispatch VEC_SIZE = gcd(16 / sizeof(sycl_t), hidden_size) matching CUDA
  const int candidate_vec_size =
      std::gcd(static_cast<int>(16 / sizeof(sycl_t)), hidden_size);
  const int vec_size =
      (input_stride % candidate_vec_size == 0) ? candidate_vec_size : 1;
  const int block_size = std::min(hidden_size / vec_size, max_block_size);

  auto& queue = vllm::xpu::vllmGetQueue();

  auto launch = [&](auto vec_tag) {
    constexpr int VS = decltype(vec_tag)::value;
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(num_tokens * block_size, block_size),
          rms_norm_static_fp8_quant_kernel<sycl_t, out_t, VS>(
              out.data_ptr<out_t>(),
              (const sycl_t*)input.data_ptr<scalar_t>(),
              input_stride,
              (const sycl_t*)weight.data_ptr<scalar_t>(),
              scale.data_ptr<float>(),
              epsilon,
              hidden_size));
    });
  };

  // Dispatch on vec_size (gcd guarantees hidden_size % vec_size == 0)
  switch (vec_size) {
    case 8:
      launch(std::integral_constant<int, 8>{});
      break;
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    case 2:
      launch(std::integral_constant<int, 2>{});
      break;
    default:
      launch(std::integral_constant<int, 1>{});
      break;
  }
}

template <typename scalar_t, typename out_t>
void call_fused_add_rms_norm_static_fp8_quant_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor const& weight,
    torch::Tensor const& scale,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int hidden_size = input.size(-1);
  const int input_stride = input.stride(-2);
  const int64_t num_tokens = input.numel() / hidden_size;

  // Match CUDA: smaller blocks when num_tokens is large for better occupancy
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;

  // Dispatch VEC_SIZE = gcd(16 / sizeof(sycl_t), hidden_size) matching CUDA.
  // Also require input_stride % vec_size == 0 for safe vectorized input reads.
  int vec_size = std::gcd(static_cast<int>(16 / sizeof(sycl_t)), hidden_size);
  if (input_stride % vec_size != 0) {
    vec_size = 1;
  }
  const int block_size = std::min(hidden_size / vec_size, max_block_size);

  auto& queue = vllm::xpu::vllmGetQueue();

  auto launch = [&](auto vec_tag) {
    constexpr int VS = decltype(vec_tag)::value;
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(num_tokens * block_size, block_size),
          fused_add_rms_norm_static_fp8_quant_kernel<sycl_t, out_t, VS>(
              out.data_ptr<out_t>(),
              (const sycl_t*)input.data_ptr<scalar_t>(),
              input_stride,
              (sycl_t*)residual.data_ptr<scalar_t>(),
              (const sycl_t*)weight.data_ptr<scalar_t>(),
              scale.data_ptr<float>(),
              epsilon,
              hidden_size));
    });
  };

  // Dispatch on vec_size (gcd guarantees hidden_size % vec_size == 0)
  switch (vec_size) {
    case 8:
      launch(std::integral_constant<int, 8>{});
      break;
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    case 2:
      launch(std::integral_constant<int, 2>{});
      break;
    default:
      launch(std::integral_constant<int, 1>{});
      break;
  }
}

template <typename scalar_t>
void call_rms_norm_mxfp4_quant_kernel(
    torch::Tensor& out,
    std::optional<torch::Tensor>& residual,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor& scales,
    float epsilon,
    int group_size,
    bool is_scale_transposed) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int hidden_size = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_size;
  const int num_groups = hidden_size / group_size;
  const int num_vec = hidden_size / static_cast<int>(16 / sizeof(sycl_t));

  const int64_t scale_stride_token =
      is_scale_transposed ? 1 : static_cast<int64_t>(num_groups);
  const int64_t scale_stride_group = is_scale_transposed ? num_tokens : 1LL;

  auto* out_ptr = out.data_ptr<uint8_t>();
  auto* input_ptr = input.data_ptr<scalar_t>();
  auto* weight_ptr = weight.data_ptr<scalar_t>();
  auto* scales_ptr = scales.data_ptr<float>();

  if (try_rms_norm_mx_quant<scalar_t, uint8_t>(
          out_ptr,
          residual,
          input,
          weight,
          scales_ptr,
          epsilon,
          is_scale_transposed)) {
    return;
  }

  auto& queue = vllm::xpu::vllmGetQueue();
  const int eu_count = static_cast<int>(
      queue.get_device().get_info<sycl::info::device::max_compute_units>());
  const int64_t block_size =
      fused_norm_quant_block_size(num_tokens, eu_count, num_vec, 4, true);

  auto launch = [&](auto has_residual_tag) {
    constexpr bool has_residual = decltype(has_residual_tag)::value;
    sycl_t* residual_ptr = nullptr;
    if constexpr (has_residual) {
      residual_ptr = (sycl_t*)residual->data_ptr<scalar_t>();
    }
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> row_smem(
          sycl::range<1>(hidden_size), cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(num_tokens * block_size, block_size),
          rms_norm_mxfp4_quant_kernel<sycl_t, has_residual>(
              out_ptr,
              residual_ptr,
              (const sycl_t*)input_ptr,
              (const sycl_t*)weight_ptr,
              scales_ptr,
              epsilon,
              hidden_size,
              group_size,
              scale_stride_token,
              scale_stride_group,
              row_smem));
    });
  };

  if (residual.has_value()) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

}  // namespace vllm

void rms_norm_dynamic_per_token_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor& scales,
    double const epsilon,
    std::optional<torch::Tensor> scale_ub,
    std::optional<torch::Tensor> residual) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(
      weight.dtype() == input.dtype(),
      "weight and input must have the same dtype");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      out.dtype() == torch::kFloat8_e4m3fn || out.dtype() == torch::kInt8,
      "output must be float8_e4m3fn or int8");
  if (scale_ub.has_value()) {
    TORCH_CHECK(
        out.dtype() == torch::kFloat8_e4m3fn,
        "scale_ub is only supported for FP8 output");
  }
  if (residual.has_value()) {
    TORCH_CHECK(
        residual->scalar_type() == input.scalar_type(),
        "residual and input must have the same dtype");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
  }

  if (out.dtype() == torch::kFloat8_e4m3fn) {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "rms_norm_dynamic_per_token_quant", [&] {
          vllm::call_rms_norm_dynamic_per_token_quant_kernel<
              scalar_t,
              at::Float8_e4m3fn>(
              out,
              residual,
              input,
              weight,
              scale_ub,
              scales,
              static_cast<float>(epsilon));
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "rms_norm_dynamic_per_token_quant_int8", [&] {
          vllm::call_rms_norm_dynamic_per_token_quant_kernel<scalar_t, int8_t>(
              out,
              residual,
              input,
              weight,
              scale_ub,
              scales,
              static_cast<float>(epsilon));
        });
  }
}

void rms_norm_per_block_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor& scales,
    double const epsilon,
    std::optional<torch::Tensor> scale_ub,
    std::optional<torch::Tensor> residual,
    int64_t group_size,
    bool is_scale_transposed,
    bool scale_ue8m0) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(
      input.stride(-1) == 1, "input must be contiguous in the last dimension");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(
      weight.dtype() == input.dtype(),
      "weight and input must have the same dtype");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      out.dtype() == torch::kFloat8_e4m3fn || out.dtype() == torch::kInt8,
      "output must be float8_e4m3fn or int8");
  TORCH_CHECK(
      input.size(-1) % group_size == 0,
      "hidden_size must be divisible by group_size");
  if (residual.has_value()) {
    TORCH_CHECK(
        residual->scalar_type() == input.scalar_type(),
        "residual and input must have the same dtype");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
  }
  if (scale_ue8m0) {
    TORCH_CHECK(
        out.dtype() == torch::kFloat8_e4m3fn,
        "scale_ue8m0 (MX FP8) is only supported for float8_e4m3fn output");
  }

  if (out.dtype() == torch::kFloat8_e4m3fn) {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "rms_norm_per_block_quant", [&] {
          vllm::
              call_rms_norm_per_block_quant_kernel<scalar_t, at::Float8_e4m3fn>(
                  out,
                  residual,
                  input,
                  weight,
                  scales,
                  static_cast<float>(epsilon),
                  static_cast<int>(group_size),
                  is_scale_transposed,
                  scale_ue8m0);
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "rms_norm_per_block_quant_int8", [&] {
          vllm::call_rms_norm_per_block_quant_kernel<scalar_t, int8_t>(
              out,
              residual,
              input,
              weight,
              scales,
              static_cast<float>(epsilon),
              static_cast<int>(group_size),
              is_scale_transposed,
              /*scale_ue8m0=*/false);
        });
  }
}
void rms_norm_static_fp8_quant(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& scale,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(
      weight.dtype() == input.dtype(),
      "weight and input must have the same dtype");

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_static_fp8_quant", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "rms_norm_static_fp8_quant_fp8", [&] {
              vllm::call_rms_norm_static_fp8_quant_kernel<scalar_t, fp8_t>(
                  out, input, weight, scale, static_cast<float>(epsilon));
            });
      });
}

void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& scale,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(
      residual.scalar_type() == input.scalar_type(),
      "residual and input must have the same dtype");
  TORCH_CHECK(
      weight.scalar_type() == input.scalar_type(),
      "weight and input must have the same dtype");

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "fused_add_rms_norm_static_fp8_quant", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "fused_add_rms_norm_static_fp8_quant_fp8", [&] {
              vllm::call_fused_add_rms_norm_static_fp8_quant_kernel<
                  scalar_t,
                  fp8_t>(
                  out,
                  input,
                  residual,
                  weight,
                  scale,
                  static_cast<float>(epsilon));
            });
      });
}

void rms_norm_mxfp4_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor& scales,
    double const epsilon,
    std::optional<torch::Tensor> residual,
    int64_t group_size) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(
      weight.dtype() == input.dtype(),
      "weight and input must have the same dtype");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      out.scalar_type() == at::ScalarType::Byte,
      "output must be uint8 (packed FP4)");
  TORCH_CHECK(
      group_size == 32, "MXFP4 requires group_size == 32, got ", group_size);
  TORCH_CHECK(
      input.size(-1) % group_size == 0,
      "hidden_size must be divisible by group_size");
  TORCH_CHECK(
      input.size(-1) % 2 == 0, "hidden_size must be even for FP4 packing");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2-D");
  if (residual.has_value()) {
    TORCH_CHECK(
        residual->scalar_type() == input.scalar_type(),
        "residual and input must have the same dtype");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
  }

  const bool is_scale_transposed = scales.stride(0) < scales.stride(1);

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_mxfp4_quant", [&] {
        vllm::call_rms_norm_mxfp4_quant_kernel<scalar_t>(
            out,
            residual,
            input,
            weight,
            scales,
            static_cast<float>(epsilon),
            static_cast<int>(group_size),
            is_scale_transposed);
      });
}
