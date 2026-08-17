#pragma once

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>

#include <dnnl.hpp>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

// INT8 W8A16 GEMM: bf16/f16 activations x int8 (s8) weights, symmetric
// group-quantized (no zero point).
//
//   result = (mat1 @ mat2^T) * scale + bias
//
//   mat1  : src, [b, m, k]  bf16/f16
//   mat2  : quantized weight, [k, n] s8 (1 byte per weight, no packing),
//           passed as the transpose of a contiguous [n, k] tensor so that k
//           is the contiguous (stride-1) dimension (trans_type_t::nt).
//   m2_sc : weight scale, [k/group_size, n] bf16 (group quant along k,
//           per-channel along n).
//
// This mirrors dnnl_matmul_w4a16_int4 but with s8 weights (no packing, no
// zero point).
static inline void dnnl_matmul_w8a16_int8(
    torch::Tensor& result,      // dst, [b, m, n]
    const torch::Tensor& mat1,  // src, [b, m, k]
    const torch::Tensor& mat2,  // quantized weight, [k, n] transpose
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& m2_sc,  // [k/group_size, n]
    const int64_t group_size) {
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();  // presume channel last format
  const int k = *(src_sz.end() - 1);

  // get joint dtypes
  joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  if (in_dtype == at::ScalarType::Half) {
    jd = joint_dtypes_t::f16_int8;
  } else if (in_dtype == at::ScalarType::BFloat16) {
    jd = joint_dtypes_t::bf16_int8;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for int8 matmul: ", mat1.scalar_type());
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
        false, "Unsupported input dimension for int8 matmul: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
                    ? mat2.strides()[mat2.dim() - 2]
                    : mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    // group quant along k (one scale per group_size), per-channel along n.
    pattr.set_scales(
        DNNL_ARG_WEIGHTS,
        /* mask */ (1 << 0) + (1 << 1),
        {group_size, 1},
        get_onednn_dtype(m2_sc));
    // oneDNN requires fpmath_mode with apply_to_int=true for integral (s8)
    // weights carrying a K-dimension scale mask (group quant). Without it,
    // matmul_pd.hpp::attr_scales_ok rejects the K-mask as "unsupported scales
    // configuration". Mirrors the proven int4 kernel.
    pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    if (in_dtype == at::ScalarType::BFloat16) {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
    }
  };

  // ************************************************************
  // get device, engine, stream
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
      group_size);

  int arg_off = 0;
  // set scale for matmul args
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      m2_sc.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(m2_sc), engine, m2_sc.data_ptr());
      });

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
