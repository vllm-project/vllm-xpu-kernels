// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

/**
 * Int8 × int8 → float16 matmul via oneDNN (symmetric or asymmetric).
 *
 * result : [m, n]  float16
 * mat1   : [m, k]  int8  (quantized activations)
 * m1_sc  : [m, 1]  fp16/fp32 per-token activation scales, or [1] per-tensor
 * m1_zp  : [m, 1]  int32 per-token activation zero points, or [1] per-tensor
 * mat2   : [k, n]  int8  in NT format (i.e. the non-contiguous transpose
 *                  view of a [n, k] row-major tensor, strides [1, k])
 * m2_sc  : [1, n]  fp16/fp32 per-channel weight scales,
 *                  or [k/group_size, n] per-group scales,
 *                  or [1] per-tensor
 * m2_zp  : [1, n]  int32 per-channel weight zero points,
 *                  or [k/group_size, n] per-group zero points,
 *                  or [1] per-tensor
 * group_size : group size for weight quantization (-1 for per-channel)
 * bias   : optional [n] or [m, n]
 */
static inline void dnnl_matmul_w8a8_int8(
    torch::Tensor& result,
    const torch::Tensor& mat1,
    const torch::Tensor& m1_sc,
    const torch::Tensor& m1_zp,
    const torch::Tensor& mat2,
    const torch::Tensor& m2_sc,
    const torch::Tensor& m2_zp,
    int64_t group_size,
    const std::optional<torch::Tensor>& bias) {
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();
  const int k = *(src_sz.end() - 1);

  auto jd = joint_dtypes_t::int8;
  bias_type_t b_type = get_bias_type(bias, m, n);

  // Compute leading dimensions
  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false,
        "Unsupported input dimension for int8-int8 matmul: ",
        mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  // mat2 is expected in NT format: [k, n] with strides [1, k]
  // (the transpose-view of a [n, k] row-major weight tensor)
  TORCH_CHECK(
      mat2.strides()[mat2.dim() - 2] == 1,
      "int8 weight must be in NT format (transposed view of [n, k])");
  int64_t ldb = mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[leading_dim];

  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  auto f_attr = [&](primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // Activation scales: per-token [m, 1] (numel == m) or per-tensor [1]
    if (m1_sc.numel() == m) {
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */ (1 << 0) + (1 << 1),
          {1, k},
          get_onednn_dtype(m1_sc));
      // Activation zero points: per-token
      pattr.set_zero_points(
          DNNL_ARG_SRC,
          /* mask */ (1 << 0) + (1 << 1),
          {1, k},
          memory::data_type::s32);
    } else {
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */ 0,
          {},
          get_onednn_dtype(m1_sc));
      // Activation zero points: per-tensor
      pattr.set_zero_points(
          DNNL_ARG_SRC,
          /* mask */ 0,
          {},
          memory::data_type::s32);
    }

    // Weight scales: per-tensor, per-channel, or per-group
    if (m2_sc.numel() == 1) {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ 0,
          {},
          get_onednn_dtype(m2_sc));
      // Weight zero points: per-tensor
      pattr.set_zero_points(
          DNNL_ARG_WEIGHTS,
          /* mask */ 0,
          {},
          memory::data_type::s32);
    } else if (group_size == -1 || m2_sc.numel() == n) {
      // per-channel: one scale per output channel
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 1),
          {},
          get_onednn_dtype(m2_sc));
      // Weight zero points: per-channel
      pattr.set_zero_points(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 1),
          {},
          memory::data_type::s32);
    } else {
      // per-group: [k/group_size, n] scales
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 0) + (1 << 1),
          {group_size, 1},
          get_onednn_dtype(m2_sc));
      // Weight zero points: per-group
      pattr.set_zero_points(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 0) + (1 << 1),
          {group_size, 1},
          memory::data_type::s32);
    }
  };

  // Encode scale and zero point counts into the cache key
  int64_t sc_group_size = (static_cast<int64_t>(m1_sc.numel()) << 32) |
                          static_cast<int64_t>(m2_sc.numel());
  int64_t zp_group_size = (static_cast<int64_t>(m1_zp.numel()) << 32) |
                          static_cast<int64_t>(m2_zp.numel());

  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd,
      trans_type_t::nt,
      b_type,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      dev_id,
      f_attr,
      sc_group_size,
      zp_group_size);

  int arg_off = 0;
  // Set activation scales
  matmul_ext.set_attribute(
      arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc.data_ptr(), [&]() {
        return make_onednn_memory(
            get_onednn_md(m1_sc), engine, m1_sc.data_ptr());
      });
  // Set activation zero points
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
      m1_zp.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(m1_zp), engine, m1_zp.data_ptr());
      });
  // Set weight scales
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      m2_sc.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(m2_sc), engine, m2_sc.data_ptr());
      });
  // Set weight zero points
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
      m2_zp.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(m2_zp), engine, m2_zp.data_ptr());
      });

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);
  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (get_shape(b_type) != bias_shape_t::none) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }

  int scratchpad_size = matmul_ext.get_scratchpad_size();
  torch::Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto& strm = GpuStreamManager::Instance().get_stream();
  matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}

}  // namespace oneDNN
