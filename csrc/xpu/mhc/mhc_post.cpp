#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <cstdint>
#include "utils.h"

namespace vllm {

using bf16 = sycl::ext::oneapi::bfloat16;

static constexpr int HC = 4;

template <int N>
static inline sycl::vec<bf16, N> vload_bf16(const bf16* p) {
  return *reinterpret_cast<const sycl::vec<bf16, N>*>(p);
}
template <int N>
static inline void vstore_bf16(bf16* p, const sycl::vec<bf16, N>& v) {
  *reinterpret_cast<sycl::vec<bf16, N>*>(p) = v;
}

class MhcPostKernel;

static sycl::event launch_mhc_post(
    sycl::queue& q,
    const float* __restrict a,
    const bf16* __restrict b,
    const float* __restrict c,
    const bf16* __restrict d,
    bf16* __restrict x,
    int N,
    int H) {
  static constexpr int VEC_POST = 8;
  static constexpr int WG_SIZE = 256;

  sycl::range<1> global(static_cast<size_t>(N) * WG_SIZE);
  sycl::range<1> local(WG_SIZE);

  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> coeff_s(HC * HC + HC, h);

    h.parallel_for<MhcPostKernel>(
        sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> it)
            [[sycl::reqd_sub_group_size(16), intel::kernel_args_restrict]] {
              const int tok = it.get_group(0);
              const int lid = it.get_local_id(0);

              if (lid == 0) {
                const float* a_ptr = a + tok * HC * HC;
                const float* c_ptr = c + tok * HC;
#pragma unroll
                for (int i = 0; i < HC * HC; ++i)
                  coeff_s[i] = a_ptr[i];
#pragma unroll
                for (int i = 0; i < HC; ++i)
                  coeff_s[HC * HC + i] = c_ptr[i];
              }
              sycl::group_barrier(it.get_group());

              float a_reg[HC][HC];
              float c_reg[HC];
#pragma unroll
              for (int i = 0; i < HC; ++i) {
                c_reg[i] = coeff_s[HC * HC + i];
#pragma unroll
                for (int j = 0; j < HC; ++j)
                  a_reg[i][j] = coeff_s[i * HC + j];
              }

              for (int k = lid * VEC_POST; k < H; k += WG_SIZE * VEC_POST) {
                if (k + VEC_POST <= H) {
                  auto dv = vload_bf16<VEC_POST>(d + tok * H + k);

                  float acc[HC][VEC_POST];
#pragma unroll
                  for (int o = 0; o < HC; ++o)
#pragma unroll
                    for (int v = 0; v < VEC_POST; ++v)
                      acc[o][v] = c_reg[o] * float(dv[v]);

#pragma unroll
                  for (int i = 0; i < HC; ++i) {
                    auto bv = vload_bf16<VEC_POST>(b + (tok * HC + i) * H + k);
#pragma unroll
                    for (int o = 0; o < HC; ++o) {
                      float a_val = a_reg[i][o];
#pragma unroll
                      for (int v = 0; v < VEC_POST; ++v)
                        acc[o][v] += a_val * float(bv[v]);
                    }
                  }

#pragma unroll
                  for (int o = 0; o < HC; ++o) {
                    sycl::vec<bf16, VEC_POST> ov;
#pragma unroll
                    for (int v = 0; v < VEC_POST; ++v)
                      ov[v] = bf16(acc[o][v]);
                    vstore_bf16<VEC_POST>(x + (tok * HC + o) * H + k, ov);
                  }
                } else {
                  for (int kk = k; kk < H; ++kk) {
                    float d_val = float(d[tok * H + kk]);
                    float acc[HC];
#pragma unroll
                    for (int o = 0; o < HC; ++o)
                      acc[o] = c_reg[o] * d_val;

#pragma unroll
                    for (int i = 0; i < HC; ++i) {
                      float b_val = float(b[(tok * HC + i) * H + kk]);
#pragma unroll
                      for (int o = 0; o < HC; ++o)
                        acc[o] += a_reg[i][o] * b_val;
                    }
#pragma unroll
                    for (int o = 0; o < HC; ++o)
                      x[(tok * HC + o) * H + kk] = bf16(acc[o]);
                  }
                }
              }
            });
  });
}

}  // namespace vllm

using namespace vllm;

at::Tensor mhc_post(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix) {
  TORCH_CHECK(residual.is_xpu(), "mhc_post: tensors must be on XPU");
  TORCH_CHECK(
      residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(
      post_layer_mix.scalar_type() == at::kFloat,
      "post_layer_mix must be float32");
  TORCH_CHECK(
      comb_res_mix.scalar_type() == at::kFloat, "comb_res_mix must be float32");

  auto residual_c = residual.contiguous();
  auto x_c = x.contiguous();
  auto a_c = comb_res_mix.contiguous();
  auto c_c = post_layer_mix.squeeze(-1).contiguous();

  const int64_t H = residual_c.size(-1);
  const int64_t N = residual_c.size(0);

  auto out = at::empty_like(residual_c);
  if (N == 0) return out;

  auto& q = vllm::xpu::vllmGetQueue();
  launch_mhc_post(
      q,
      a_c.data_ptr<float>(),
      reinterpret_cast<const bf16*>(residual_c.data_ptr()),
      c_c.data_ptr<float>(),
      reinterpret_cast<const bf16*>(x_c.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(H));

  return out;
}
