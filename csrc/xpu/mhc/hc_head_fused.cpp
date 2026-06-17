// DeepSeek V4 HC head fused kernel — optimized single-kernel implementation.

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "utils.h"

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
static constexpr int HC = 4;

using vllm::xpu::aligned_vec;

static inline float sigmoid(float x) {
    return 1.f / (1.f + sycl::native::exp(-x));
}

class HcHeadFusedOptKernel;

sycl::event launch_hc_head_fused_opt(
    sycl::queue& q,
    const bf16* hs_flat,
    const float* fn,
    const float* hc_scale,
    const float* hc_base,
    bf16* out,
    int N,
    int H,
    float rms_eps,
    float hc_eps) {
    static constexpr int BLOCK_M = 2;
    static constexpr int VEC_DOT = 4;
    static constexpr int VEC_OUT = 8;
    static constexpr int SG_SIZE = 16;
    static constexpr int WG_SIZE = 256;
    static constexpr int NUM_SG = WG_SIZE / SG_SIZE;
    static constexpr int NUM_RED = HC + 1;
    const int hc_h = HC * H;

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    syclex::properties kernel_props{
        syclex::sub_group_size<SG_SIZE>,
        intelex::grf_size<128>};

    const size_t num_wgs = static_cast<size_t>((N + BLOCK_M - 1) / BLOCK_M);
    sycl::range<1> global(num_wgs * WG_SIZE);
    sycl::range<1> local(WG_SIZE);

    return q.submit([&](sycl::handler& h) {
        using vec_bf16_dot_t = aligned_vec<bf16, VEC_DOT>;
        using vec_f32_dot_t = aligned_vec<float, VEC_DOT>;
        using vec_bf16_out_t = aligned_vec<bf16, VEC_OUT>;

        sycl::local_accessor<float, 1> red_scratch(NUM_RED * NUM_SG * BLOCK_M, h);
        sycl::local_accessor<float, 1> pre_s(HC * BLOCK_M, h);

        h.parallel_for<HcHeadFusedOptKernel>(
            sycl::nd_range<1>(global, local), kernel_props,
            [=](sycl::nd_item<1> it) {
                const int wg_id = it.get_group(0);
                const int tid = it.get_local_id(0);
                auto sg = it.get_sub_group();
                const int sg_id = sg.get_group_id()[0];
                const int sg_lane = sg.get_local_id()[0];
                const int tok_base = wg_id * BLOCK_M;

                float local_mix[BLOCK_M][HC];
                float local_sq[BLOCK_M];
                #pragma unroll
                for (int t = 0; t < BLOCK_M; ++t) {
                    local_sq[t] = 0.f;
                    #pragma unroll
                    for (int j = 0; j < HC; ++j)
                        local_mix[t][j] = 0.f;
                }

                for (int k = tid * VEC_DOT; k < hc_h; k += WG_SIZE * VEC_DOT) {
                    vec_f32_dot_t fn_tile[HC];
                    #pragma unroll
                    for (int j = 0; j < HC; ++j)
                        fn_tile[j] = *reinterpret_cast<const vec_f32_dot_t*>(fn + j * hc_h + k);

                    #pragma unroll
                    for (int t = 0; t < BLOCK_M; ++t) {
                        int tok = tok_base + t;
                        if (tok >= N)
                            break;
                        auto xv = *reinterpret_cast<const vec_bf16_dot_t*>(hs_flat + tok * hc_h + k);
                        float xf[VEC_DOT];
                        #pragma unroll
                        for (int v = 0; v < VEC_DOT; ++v)
                            xf[v] = float(xv.val[v]);
                        #pragma unroll
                        for (int v = 0; v < VEC_DOT; ++v)
                            local_sq[t] += xf[v] * xf[v];
                        #pragma unroll
                        for (int j = 0; j < HC; ++j) {
                            float acc = 0.f;
                            #pragma unroll
                            for (int v = 0; v < VEC_DOT; ++v)
                                acc += xf[v] * fn_tile[j].val[v];
                            local_mix[t][j] += acc;
                        }
                    }
                }

                #pragma unroll
                for (int t = 0; t < BLOCK_M; ++t) {
                    int tok = tok_base + t;
                    if (tok >= N)
                        break;
                    float* tok_scratch = &red_scratch[t * NUM_RED * NUM_SG];
                    float sg_sq = sycl::reduce_over_group(sg, local_sq[t], sycl::plus<float>());
                    if (sg_lane == 0)
                        tok_scratch[sg_id * NUM_RED] = sg_sq;
                    #pragma unroll
                    for (int j = 0; j < HC; ++j) {
                        float sg_mix = sycl::reduce_over_group(sg, local_mix[t][j], sycl::plus<float>());
                        if (sg_lane == 0)
                            tok_scratch[sg_id * NUM_RED + j + 1] = sg_mix;
                    }
                    sycl::group_barrier(it.get_group());

                    if (sg_id == 0 && sg_lane < NUM_RED) {
                        float s = 0.f;
                        #pragma unroll
                        for (int si = 0; si < NUM_SG; ++si)
                            s += tok_scratch[si * NUM_RED + sg_lane];
                        tok_scratch[sg_lane] = s;
                    }
                    sycl::group_barrier(it.get_group());

                    float rrms = sycl::native::rsqrt(tok_scratch[0] / static_cast<float>(hc_h) + rms_eps);
                    if (tid < HC) {
                        float mix_val = tok_scratch[tid + 1] * rrms;
                        pre_s[t * HC + tid] = sigmoid(sycl::fma(mix_val, hc_scale[0], hc_base[tid])) + hc_eps;
                    }
                }
                sycl::group_barrier(it.get_group());

                #pragma unroll
                for (int t = 0; t < BLOCK_M; ++t) {
                    int tok = tok_base + t;
                    if (tok >= N)
                        break;

                    float pre_reg[HC];
                    #pragma unroll
                    for (int m = 0; m < HC; ++m)
                        pre_reg[m] = pre_s[t * HC + m];

                    for (int k = tid * VEC_OUT; k < H; k += WG_SIZE * VEC_OUT) {
                        float acc[VEC_OUT];
                        #pragma unroll
                        for (int v = 0; v < VEC_OUT; ++v)
                            acc[v] = 0.f;
                        #pragma unroll
                        for (int m = 0; m < HC; ++m) {
                            auto rv = *reinterpret_cast<const vec_bf16_out_t*>(hs_flat + (tok * HC + m) * H + k);
                            #pragma unroll
                            for (int v = 0; v < VEC_OUT; ++v)
                                acc[v] += pre_reg[m] * float(rv.val[v]);
                        }
                        vec_bf16_out_t ov;
                        #pragma unroll
                        for (int v = 0; v < VEC_OUT; ++v)
                            ov.val[v] = static_cast<bf16>(acc[v]);
                        *reinterpret_cast<vec_bf16_out_t*>(out + tok * H + k) = ov;
                    }
                }
            });
    });
}

}  // namespace

void hc_head_fused(
    const at::Tensor& hs_flat,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    at::Tensor& out,
    double rms_eps,
    double hc_eps) {
    TORCH_CHECK(hs_flat.is_xpu() && fn.is_xpu() && hc_scale.is_xpu() && hc_base.is_xpu() && out.is_xpu(),
                "hc_head_fused: tensors must be on XPU");
    TORCH_CHECK(hs_flat.scalar_type() == at::kBFloat16, "hs_flat must be bfloat16");
    TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");
    TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
    TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");
    TORCH_CHECK(out.is_contiguous(), "hc_head_fused: out must be contiguous");

    const int64_t N = hs_flat.size(0);
    const int64_t H = hs_flat.size(2);
    TORCH_CHECK(H % 8 == 0, "hc_head_fused: hidden_size must be a multiple of 8 for vectorized loads/stores");

    TORCH_CHECK(hs_flat.dim() == 3, "hc_head_fused: hs_flat must be [N, HC, H]");
    TORCH_CHECK(hs_flat.size(1) == HC, "hc_head_fused: hs_flat dim1 must be HC=4");
    TORCH_CHECK(fn.size(0) == HC && fn.size(1) == HC * H, "hc_head_fused: fn shape mismatch");
    TORCH_CHECK(hc_scale.numel() == 1, "hc_head_fused: hc_scale must have 1 element");
    TORCH_CHECK(hc_base.numel() == HC, "hc_head_fused: hc_base size mismatch");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == N && out.size(1) == H,
                "hc_head_fused: out shape mismatch");

    if (N == 0)
        return;

    auto hs_c = hs_flat.contiguous();
    auto fn_c = fn.contiguous();
    auto& q = vllm::xpu::vllmGetQueue();
    launch_hc_head_fused_opt(
        q,
        reinterpret_cast<const bf16*>(hs_c.data_ptr()),
        fn_c.data_ptr<float>(),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        reinterpret_cast<bf16*>(out.data_ptr()),
        static_cast<int>(N),
        static_cast<int>(H),
        static_cast<float>(rms_eps),
        static_cast<float>(hc_eps));
}
