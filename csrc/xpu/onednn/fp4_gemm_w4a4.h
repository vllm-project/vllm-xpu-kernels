#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

static inline void dnnl_matmul_w4a4_fp4(
    torch::Tensor& result,      // dst, [b, m, n]
    const torch::Tensor& mat1,  // src, [b, m, k]
    const torch::Tensor& mat2,  // quantized weight, [k, n] transpose
    bool is_nt,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& m1_sc,
    const torch::Tensor& m2_sc) {
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();  // presume channel last format
  const int k = (*(src_sz.end() - 1)) * 2;

  // get joint dtypes
  joint_dtypes_t jd;
  auto out_dtype = result.scalar_type();
  auto m1_sc_dtype = m1_sc.scalar_type();
  auto m2_sc_dtype = m2_sc.scalar_type();
  if (m1_sc_dtype == at::ScalarType::Float8_e8m0fnu) {
    TORCH_CHECK(
        m2_sc_dtype == at::ScalarType::Float8_e8m0fnu,
        "Mismatched scale data types in mxfp4 matmul: ",
        m1_sc_dtype,
        " vs ",
        m2_sc_dtype);
    jd = out_dtype == at::ScalarType::BFloat16 ? joint_dtypes_t::mxfp4_bf16
                                               : joint_dtypes_t::mxfp4_f16;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported scale type for fp4 matmul: ", m1_sc_dtype);
  }

  // get bias type
  bias_type_t b_type = get_bias_type(bias, m, n);

  trans_type_t tt = trans_type_t::nn;
  if (is_nt) {
    // transpose mat2
    tt = trans_type_t::nt;
  }

  // get lda ldb and ldc
  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false, "Unsupported input dimension for fp4 matmul: ", mat1.dim());
  }
  int64_t lda = 2 * mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
                    ? mat2.strides()[mat2.dim() - 2]
                    : 2 * (mat2.strides()[mat2.dim() - 1]);
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (m1_sc_dtype == at::ScalarType::Float8_e8m0fnu) {
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */ (1 << 0) + (1 << 1),
          {1, 32},
          get_onednn_dtype(m1_sc));
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 0) + (1 << 1),
          {32, 1},
          get_onednn_dtype(m2_sc));
    } else {
      if (m1_sc.numel() == 1) {
        pattr.set_scales(
            DNNL_ARG_SRC,
            /* mask */ 0,
            {},
            get_onednn_dtype(m1_sc));
        /* per tensor quant */
      } else {
        pattr.set_scales(
            DNNL_ARG_SRC,
            /* mask */ (1 << 0) + (1 << 1),
            {1, k},
            get_onednn_dtype(m1_sc));
        /* per token quant */
      }

      if (m2_sc.numel() == 1) {
        pattr.set_scales(
            DNNL_ARG_WEIGHTS,
            /* mask */ 0,
            {},
            get_onednn_dtype(m2_sc));
        /* per tensor quant */
      } else {
        pattr.set_scales(
            DNNL_ARG_WEIGHTS,
            /* mask */ (1 << 1),
            {},
            get_onednn_dtype(m2_sc));
        /* per channel quant */
      }
    }
  };

  int arg_off = 0;

  // ************************************************************
  // get device, engine, stream
  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  int m1_sc_group_size = m1_sc.numel();
  int m2_sc_group_size = m2_sc.numel();
  int sc_group_size = (m1_sc_group_size << 8) | m2_sc_group_size;
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd, tt, b_type, m, n, k, lda, ldb, ldc, dev_id, f_attr, sc_group_size);

  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      m2_sc.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(m2_sc), engine, m2_sc.data_ptr());
      });
  matmul_ext.set_attribute(
      arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc.data_ptr(), [&]() {
        return make_onednn_memory(
            get_onednn_md(m1_sc), engine, m1_sc.data_ptr());
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
  auto qfp4_matmul_event =
      matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}
}  // namespace oneDNN
