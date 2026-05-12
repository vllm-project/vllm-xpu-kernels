#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

static inline void dnnl_matmul_w4a8_int4(
    torch::Tensor& result,       // dst, [m, n]
    const torch::Tensor& mat1,   // quantized src, [b, m, k]
    const torch::Tensor& m1_sc,  // src m2_sc, [b * m, 1] or [b]
    const torch::Tensor& m1_zp,  // src zero point, [b * m, 1] or [1]
    const torch::Tensor& mat2,   // quantized weight, [k/8, n] transpose
    const torch::Tensor& m2_sc,  // [k/group_size, n]
    const torch::Tensor& m2_zp,  // [k/group_size, n/8]
    int64_t group_size,          // group size for weight quantization
    const std::optional<torch::Tensor>& bias) {
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();  // presume channel last format
  const int k = *(src_sz.end() - 1);

  // get joint dtypes
  joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  if (in_dtype == at::ScalarType::Char) {
    jd = joint_dtypes_t::s8_int4;
  } else if (in_dtype == at::ScalarType::Byte) {
    jd = joint_dtypes_t::u8_int4;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for int4-int8 matmul: ", in_dtype);
  }

  // get bias type
  bias_type_t b_type = get_bias_type(bias, m, n);

  // get lda ldb and ldc
  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false,
        "Unsupported input dimension for int4-int8 matmul: ",
        mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] * 8;
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (m1_sc.numel() == m) {
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */ (1 << 0) + (1 << 1),
          {1, k},
          get_onednn_dtype(m1_sc));
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
      pattr.set_zero_points(
          DNNL_ARG_SRC,
          /* mask */ 0,
          {},
          memory::data_type::s32);
    }

    pattr.set_scales(
        DNNL_ARG_WEIGHTS,
        /* mask */ (1 << 0) + (1 << 1),
        {group_size, 1},
        get_onednn_dtype(m2_sc));
    if (m2_zp.numel() == 1) {
      pattr.set_zero_points(
          DNNL_ARG_WEIGHTS,
          /* mask */ 0,
          {},
          memory::data_type::s8);
    } else {
      pattr.set_zero_points(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 0) + (1 << 1),
          {group_size, 1},
          memory::data_type::u4);
    }
  };

  // ************************************************************
  // get device, engine, stream
  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  // Encode the number of elements of m1_sc and m2_sc into a single int64_t:
  // upper 32 bits: m1_sc.numel(), lower 32 bits: m2_sc.numel()
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
  // set m2_sc and zero point for matmul args
  matmul_ext.set_attribute(
      arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc.data_ptr(), [&]() {
        return make_onednn_memory(
            get_onednn_md(m1_sc), engine, m1_sc.data_ptr());
      });
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
      m1_zp.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(m1_zp), engine, m1_zp.data_ptr());
      });

  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      m2_sc.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(m2_sc), engine, m2_sc.data_ptr());
      });

  if (m2_zp.numel() == 1) {
    // set zp_md for symmetric quantization
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
        m2_zp.data_ptr(),
        [&]() {
          return make_onednn_memory(
              get_onednn_md(m2_zp), engine, m2_zp.data_ptr());
        });
  } else {
    // set zp_md for asymmetric quantization
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
        m2_zp.data_ptr(),
        [&]() {
          auto num_groups = k / group_size;
          dnnl::memory zp_B_u4_m(
              {{num_groups, n}, memory::data_type::u4, {n, 1}},
              engine,
              m2_zp.data_ptr());
          return zp_B_u4_m;
        });
  }

  // set general args
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
