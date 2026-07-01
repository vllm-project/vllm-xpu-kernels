#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

static inline void dnnl_matmul_w8a8_fp8(
    torch::Tensor& result,      // dst, [b, m, n]
    const torch::Tensor& mat1,  // src, [b, m, k]
    const torch::Tensor& mat2,  // quantized weight, [k, n] transpose
    bool is_nt,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& m1_sc,
    const torch::Tensor& m2_sc) {
  TORCH_CHECK(
      mat2.dim() == 2 || mat2.dim() == 3,
      "fp8 matmul expects weight dim to be 2 or 3, got ",
      mat2.dim(),
      ".");

  const bool is_batched = mat2.dim() == 3;

  int64_t batch = 1;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;

  if (is_batched) {
    TORCH_CHECK(
        mat1.dim() == 3,
        "Batched fp8 matmul expects src shape [B, M, K], got ",
        mat1.sizes(),
        ".");
    TORCH_CHECK(
        result.dim() == 3,
        "Batched fp8 matmul expects dst shape [B, M, N], got ",
        result.sizes(),
        ".");
    TORCH_CHECK(
        mat1.size(0) == mat2.size(0),
        "Batched fp8 matmul expects input and weight batches to match, got ",
        mat1.size(0),
        " and ",
        mat2.size(0),
        ".");

    batch = mat1.size(0);
    m = mat1.size(1);
    k = mat1.size(2);
    n = mat2.size(-1);
    TORCH_CHECK(
        result.size(0) == batch && result.size(1) == m && result.size(2) == n,
        "Batched fp8 matmul got unexpected dst shape ",
        result.sizes(),
        ", expected [",
        batch,
        ", ",
        m,
        ", ",
        n,
        "].");
  } else {
    auto src_sz = mat1.sizes();
    auto o_sz = result.sizes();
    m = std::reduce(
        src_sz.begin(),
        src_sz.end() - 1,
        int64_t{1},
        std::multiplies<int64_t>());
    n = o_sz.back();
    k = *(src_sz.end() - 1);
  }

  joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  auto out_dtype = result.scalar_type();
  if (in_dtype == at::ScalarType::Float8_e5m2) {
    jd = out_dtype == at::ScalarType::BFloat16 ? joint_dtypes_t::f8_e5m2_bf16
                                               : joint_dtypes_t::f8_e5m2_f16;
  } else if (in_dtype == at::ScalarType::Float8_e4m3fn) {
    jd = out_dtype == at::ScalarType::BFloat16 ? joint_dtypes_t::f8_e4m3_bf16
                                               : joint_dtypes_t::f8_e4m3_f16;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for fp8 matmul: ", in_dtype);
  }

  bias_type_t b_type = get_bias_type(bias, batch * m, n);
  if (is_batched) {
    TORCH_CHECK(
        get_shape(b_type) == bias_shape_t::none ||
            get_shape(b_type) == bias_shape_t::scalar ||
            get_shape(b_type) == bias_shape_t::n,
        "Batched fp8 matmul only supports empty, scalar, or [N] bias.");
  }

  trans_type_t tt = is_nt ? trans_type_t::nt : trans_type_t::nn;

  torch::Tensor src_sc = m1_sc;
  torch::Tensor wei_sc = m2_sc;

  bool src_scalar = src_sc.numel() == 1;
  bool src_block_quant = false;
  int64_t src_group_k = k;

  bool wei_scalar = wei_sc.numel() == 1;
  bool wei_block_quant = false;
  int64_t wei_group_k = k;
  int64_t wei_group_n = n;

  if (is_batched) {
    if (!src_scalar) {
      if (src_sc.dim() == 2) {
        TORCH_CHECK(
            src_sc.size(0) == batch && src_sc.size(1) == m,
            "Batched src per-token scale expects shape [B, M], got ",
            src_sc.sizes(),
            ".");
      } else if (src_sc.dim() == 3) {
        TORCH_CHECK(
            src_sc.size(0) == batch && src_sc.size(1) == m,
            "Batched src grouped scale expects shape [B, M, gK], got ",
            src_sc.sizes(),
            ".");
        const int64_t src_scale_k = src_sc.size(2);
        TORCH_CHECK(
            k % src_scale_k == 0,
            "Batched src grouped scale count must divide K, got K=",
            k,
            " and grouped dim ",
            src_scale_k,
            ".");
        src_block_quant = true;
        src_group_k = k / src_scale_k;
      } else {
        TORCH_CHECK(
            false,
            "Unsupported batched src scale rank for fp8 matmul: ",
            src_sc.dim());
      }
    }

    if (!wei_scalar) {
      if (wei_sc.dim() == 1) {
        TORCH_CHECK(
            batch == 1,
            "Batched weight per-channel scale must include batch dimension, "
            "got ",
            wei_sc.sizes(),
            ".");
        TORCH_CHECK(
            wei_sc.size(0) == n,
            "Batched weight per-channel scale expects N elements, got ",
            wei_sc.sizes(),
            ".");
        wei_sc = wei_sc.reshape({1, n});
      } else if (wei_sc.dim() == 2) {
        TORCH_CHECK(
            wei_sc.size(0) == batch && wei_sc.size(1) == n,
            "Batched weight per-channel scale expects shape [B, N], got ",
            wei_sc.sizes(),
            ".");
      } else if (wei_sc.dim() == 3) {
        TORCH_CHECK(
            wei_sc.size(0) == batch,
            "Batched weight block scale expects shape [B, gK, gN], got ",
            wei_sc.sizes(),
            ".");
        TORCH_CHECK(
            wei_sc.size(1) >= 1 && wei_sc.size(2) >= 1,
            "Invalid batched weight block scale shape: ",
            wei_sc.sizes(),
            ".");
        TORCH_CHECK(
            k % wei_sc.size(1) == 0 && n % wei_sc.size(2) == 0,
            "Batched weight block scale dims must divide [K, N], got K=",
            k,
            ", N=",
            n,
            ", scale shape ",
            wei_sc.sizes(),
            ".");
        wei_block_quant = true;
        wei_group_k = k / wei_sc.size(1);
        wei_group_n = n / wei_sc.size(2);
      } else {
        TORCH_CHECK(
            false,
            "Unsupported batched weight scale rank for fp8 matmul: ",
            wei_sc.dim());
      }
    }
  } else {
    src_block_quant = (src_sc.dim() == 2) && (src_sc.size(1) > 1);
    if (src_block_quant) {
      TORCH_CHECK(
          wei_sc.dim() == 2,
          "Block quant weight scales expect 2D shape [gK, gN], got ",
          wei_sc.sizes(),
          ".");
      TORCH_CHECK(
          src_sc.size(1) == wei_sc.size(0),
          "Mismatch group size in input and weight.",
          src_sc.size(1),
          " vs ",
          wei_sc.size(0));
      src_group_k = k / wei_sc.size(0);
      wei_block_quant = true;
      wei_group_k = k / wei_sc.size(0);
      wei_group_n = n / wei_sc.size(1);
    }
  }

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (src_scalar) {
      pattr.set_scales(
          DNNL_ARG_SRC, /* mask */ 0, {}, get_onednn_dtype(src_sc));
    } else if (src_block_quant) {
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */
          is_batched ? (1 << 0) + (1 << 1) + (1 << 2) : (1 << 0) + (1 << 1),
          {1, src_group_k},
          get_onednn_dtype(src_sc));
    } else {
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */ (1 << 0) + (1 << 1),
          is_batched ? memory::dims{} : memory::dims{1, k},
          get_onednn_dtype(src_sc));
    }

    if (wei_scalar) {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS, /* mask */ 0, {}, get_onednn_dtype(wei_sc));
    } else if (wei_block_quant) {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */
          is_batched ? (1 << 0) + (1 << 1) + (1 << 2) : (1 << 0) + (1 << 1),
          {wei_group_k, wei_group_n},
          get_onednn_dtype(wei_sc));
    } else {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ is_batched ? (1 << 0) + (1 << 2) : (1 << 1),
          {},
          get_onednn_dtype(wei_sc));
    }
  };

  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  auto hash_dims = [](const torch::Tensor& tensor) {
    int hash = tensor.dim();
    for (const auto size : tensor.sizes()) {
      hash = hash * 131 + static_cast<int>(size);
    }
    return hash;
  };

  int sc_group_size = 0;
  if (is_batched) {
    sc_group_size = (hash_dims(src_sc) << 8) ^ hash_dims(wei_sc);
  } else {
    sc_group_size = (src_sc.numel() << 8) | wei_sc.numel();
  }
  if (src_sc.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
    sc_group_size |= (1 << 30);
  }

  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false, "Unsupported input dimension for fp8 matmul: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
                    ? mat2.strides()[mat2.dim() - 2]
                    : mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[leading_dim];

  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd,
      tt,
      b_type,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      lda,
      ldb,
      ldc,
      dev_id,
      f_attr,
      sc_group_size,
      1,
      is_batched ? static_cast<int>(batch) : -1);
  int arg_off = 0;
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      wei_sc.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(wei_sc), engine, wei_sc.data_ptr());
      });
  matmul_ext.set_attribute(
      arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_sc.data_ptr(), [&]() {
        return make_onednn_memory(
            get_onednn_md(src_sc), engine, src_sc.data_ptr());
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
  auto qfp8_matmul_event =
      matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}
}  // namespace oneDNN
