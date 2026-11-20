#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

// Weight-only MXFP4 GEMM: 16-bit src x f4_e2m1 weights with one e8m0 scale per
// 32 elements along K (OCP MX), decompressed by oneDNN.
static inline void dnnl_matmul_w4a16_mxfp4(
    torch::Tensor& result,      // dst, [m, n]
    const torch::Tensor& mat1,  // src, [m, k]
    const torch::Tensor& mat2,  // weight, [k/2, n] view of [n, k/2] (nt)
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& scale) {  // [k/32, n] e8m0
  const int m = mat1.size(0);
  const int n = result.size(1);
  const int k = mat1.size(1);

  joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  if (in_dtype == at::ScalarType::Half) {
    jd = joint_dtypes_t::f16_mxfp4;
  } else if (in_dtype == at::ScalarType::BFloat16) {
    jd = joint_dtypes_t::bf16_mxfp4;
  } else {
    TORCH_CHECK(false, "Unsupported dtype for mxfp4 w4a16 matmul: ", in_dtype);
  }

  bias_type_t b_type = get_bias_type(bias, m, n);
  int64_t lda = mat1.strides()[0];
  int64_t ldb = 2 * mat2.strides()[1];
  int64_t ldc = result.strides()[0];

  auto f_attr = [&](primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    pattr.set_scales(
        DNNL_ARG_WEIGHTS,
        /* mask */ (1 << 0) + (1 << 1),
        {32, 1},
        get_onednn_dtype(scale));
    pattr.set_fpmath_mode(
        in_dtype == at::ScalarType::BFloat16 ? dnnl::fpmath_mode::bf16
                                             : dnnl::fpmath_mode::f16,
        true);
  };

  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
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
      /* scale_group_size */ 32,
      /* zp_group_size */ 0);

  int arg_off = 0;
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      scale.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(scale), engine, scale.data_ptr());
      });

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(6);
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
