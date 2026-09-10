#include <torch/all.h>

#include <cstdlib>

#include "collective/moe_dtype_policy.hpp"
#include "csrc/utils.h"

namespace gpu::cutlass_kernel {

namespace grouped_gemm {
template <class moe_policy>
void kernel_functor(
    sycl::queue& stream,
    void* ptr_A,
    void* ptr_A_scale,
    void* ptr_B,
    void* ptr_B_scale,
    void* ptr_bias,
    void* ptr_D,
    void* rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t groups);

// Work-group tiles for the high-occupancy ("prefill", avg_tokens > 32) path.
// Ids match XE3_GG_FORCE_TILE for profiling overrides.
enum class PrefillTile : int {
  k256x256 = 0,
  k256x128 = 1,
  k128x256 = 2,
  k128x128 = 3,
};

// Dtype families that share a tile-selection rule on the prefill path.
enum class PrefillFamily { kBF16, kMXFP8, kMXFP4 };

// Selects the prefill work-group tile. Only two tiles are ever optimal:
// 256x256 (best arithmetic intensity) and 128x256 (packs the 32 Xe cores more
// fully when 256x256's last wave is short). The choice follows one rule per
// dtype family:
//
// BF16, MXFP8, and MXFP4 keep their base tile while it keeps the cores busy,
// then fall back to
//      128x256 once the final wave drops below a utilization threshold. MXFP4
//      is 4-bit and compute-bound, so it tolerates a shorter tail wave (>=0.85)
//      than the more wave-quantization-sensitive BF16 (>=0.90).
inline PrefillTile pick_prefill_tile(
    PrefillFamily family, int64_t M_total, int64_t N, int64_t groups) {
  // Manual override for tuning/profiling.
  if (const char* env = std::getenv("XE3_GG_FORCE_TILE")) {
    return static_cast<PrefillTile>(std::atoi(env));
  }

  constexpr int kCores = 32;
  const double kMinUtil = (family == PrefillFamily::kMXFP4) ? 0.85 : 0.90;
  const int64_t M_g = M_total / (groups > 0 ? groups : 1);
  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  int64_t tiles_256 =
      groups * ceil_div(M_g > 0 ? M_g : 1, 256) * ceil_div(N, 256);
  int64_t waves_256 = ceil_div(tiles_256, kCores);
  double util_256 =
      static_cast<double>(tiles_256) / static_cast<double>(waves_256 * kCores);

  if (util_256 >= kMinUtil) {
    return PrefillTile::k256x256;
  }
  return PrefillTile::k128x256;
}
}  // namespace grouped_gemm

/* gemm2(group_A, w2, output, offset) */

at::Tensor grouped_gemm_func(
    at::Tensor& ptr_A,
    const c10::optional<at::Tensor>& ptr_A_scale,
    at::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor& ptr_D,
    at::Tensor& rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t groups) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  auto A_dtype = ptr_A.dtype();
  auto avg_tokens_cnt = ptr_A.size(0) / groups;

  // Every stride is rebuilt from {N, K, groups} via make_cute_packed_stride()
  // and the tensors' own strides are never read, so a mismatched layout is
  // silently wrong instead of an error. LayoutA/C/D are RowMajor for all
  // policies; LayoutB is RowMajor (dense along N) except for the 4-bit weight
  // policies (mxfp4, w4a8) which are ColumnMajor (dense along K). Both scale
  // surfaces are MN-major.
  const bool a_is_fp4 = A_dtype == at::kFloat4_e2m1fn_x2;
  const bool b_is_fp4 = ptr_B.dtype() == at::kFloat4_e2m1fn_x2;
  const int64_t A_K = a_is_fp4 ? K / 2 : K;

  TORCH_CHECK(groups > 0 && N > 0 && K > 0, "N, K and num_experts must be > 0");
  TORCH_CHECK(!a_is_fp4 || K % 2 == 0, "K must be even for 4-bit ptr_A");

  // The XE 2D block copies need each row of B and D to start on a 128-bit
  // boundary, and cutlass reports a violation by aborting the process rather
  // than returning, so reject it here instead.
  const int64_t b_bits = ptr_B.dtype().itemsize() * 8 / (b_is_fp4 ? 2 : 1);
  const int64_t d_bits = ptr_D.dtype().itemsize() * 8;
  TORCH_CHECK(
      (N * b_bits) % 128 == 0 && (N * d_bits) % 128 == 0,
      "N must be a multiple of 128 bits of the weight and output element type, "
      "so N must be a multiple of 32 for 4-bit, 16 for 8-bit and 8 for 16-bit "
      "weights");

  TORCH_CHECK(
      ptr_A.dim() == 2 && ptr_A.size(1) == A_K && ptr_A.is_contiguous(),
      "ptr_A must be a contiguous 2D [total_M, K] tensor, with K halved when "
      "it is 4-bit packed");

  TORCH_CHECK(
      ptr_D.dim() == 2 && ptr_D.size(0) == ptr_A.size(0) &&
          ptr_D.size(1) == N && ptr_D.is_contiguous(),
      "ptr_D must be a contiguous 2D [total_M, N] tensor");

  TORCH_CHECK(
      ptr_B.dim() == 3 && ptr_B.size(0) == groups,
      "ptr_B must be 3D with one leading entry per expert");
  if (b_is_fp4) {
    TORCH_CHECK(
        ptr_B.size(1) == N && ptr_B.size(2) == K / 2 && ptr_B.is_contiguous(),
        "4-bit ptr_B must be a contiguous [num_experts, N, K/2] tensor");
  } else {
    // Element (e, n, k) must sit at e * N * K + n + k * N. Callers spell that
    // either as a contiguous (E, K, N) or as an (E, N, K) view of it.
    TORCH_CHECK(
        (ptr_B.size(1) == K && ptr_B.size(2) == N && ptr_B.is_contiguous()) ||
            (ptr_B.size(1) == N && ptr_B.size(2) == K &&
             ptr_B.stride(0) == N * K && (N == 1 || ptr_B.stride(1) == 1) &&
             (K == 1 || ptr_B.stride(2) == N)),
        "ptr_B must be dense along N: either a contiguous [num_experts, K, N] "
        "tensor or its [num_experts, N, K] transposed view");
  }

  TORCH_CHECK(
      rows_per_expert.dim() == 1 && rows_per_expert.numel() == groups &&
          rows_per_expert.scalar_type() == at::kInt,
      "rows_per_expert must be a 1D int32 tensor with one entry per expert");

  // cutlass_grouped_gemm_xe3() has already expanded the caller's
  // [num_experts, N] bias into one fp32 row per token (ElementC).
  if (ptr_bias) {
    TORCH_CHECK(
        ptr_bias->dim() == 2 && ptr_bias->size(0) == ptr_A.size(0) &&
            ptr_bias->size(1) == N && ptr_bias->is_contiguous() &&
            ptr_bias->scalar_type() == at::kFloat,
        "ptr_bias must expand to a contiguous fp32 [total_M, N] tensor, so the "
        "caller's bias must be [num_experts, N]");
  }

  if (ptr_A_scale && ptr_A_scale->dtype() == at::kFloat8_e8m0fnu) {
    const int64_t scale_k = (K + 31) / 32;
    TORCH_CHECK(
        ptr_B_scale && ptr_B_scale->dtype() == at::kFloat8_e8m0fnu,
        "block-scaled grouped GEMM requires an e8m0 ptr_B_scale");
    // reorder_mxfp_scales() hands over a plain [padded_M, scale_k] buffer that
    // it filled MN-major, so only its extent is meaningful here. Each expert's
    // row count is rounded up to a multiple of 4.
    TORCH_CHECK(
        ptr_A_scale->dim() == 2 && ptr_A_scale->size(1) == scale_k &&
            ptr_A_scale->is_contiguous(),
        "ptr_A_scale must be a contiguous [padded_M, ceil(K/32)] tensor");
    TORCH_CHECK(
        ptr_A_scale->size(0) >= ptr_A.size(0) &&
            ptr_A_scale->size(0) <= ptr_A.size(0) + 3 * groups,
        "ptr_A_scale must have total_M rows with each expert padded up to a "
        "multiple of 4");
    // Element (e, n, k) must sit at e * N * scale_k + n + k * N.
    TORCH_CHECK(
        ptr_B_scale->dim() == 3 && ptr_B_scale->size(0) == groups &&
            ((ptr_B_scale->size(1) == scale_k && ptr_B_scale->size(2) == N &&
              ptr_B_scale->is_contiguous()) ||
             (ptr_B_scale->size(1) == N && ptr_B_scale->size(2) == scale_k &&
              ptr_B_scale->stride(0) == N * scale_k &&
              (N == 1 || ptr_B_scale->stride(1) == 1) &&
              (scale_k == 1 || ptr_B_scale->stride(2) == N))),
        "ptr_B_scale must be dense along N: either a contiguous "
        "[num_experts, ceil(K/32), N] tensor or its [num_experts, N, "
        "ceil(K/32)] transposed view");
  } else if (ptr_A_scale && ptr_A_scale->dtype() == at::kFloat) {
    TORCH_CHECK(
        ptr_B_scale && ptr_B_scale->dtype() == at::kFloat,
        "fp8 grouped GEMM requires a float32 ptr_B_scale");
    if (ptr_A_scale->numel() == 1) {
      TORCH_CHECK(
          ptr_B_scale->numel() == groups,
          "per-tensor fp8 ptr_B_scale must hold one scale per expert");
    } else {
      const int64_t scale_k = (K + 127) / 128;
      const int64_t scale_n = (N + 127) / 128;
      TORCH_CHECK(
          ptr_A_scale->numel() == ptr_A.size(0) * scale_k,
          "block fp8 ptr_A_scale must hold total_M * ceil(K/128) scales");
      TORCH_CHECK(
          ptr_B_scale->dim() == 3 && ptr_B_scale->size(0) == groups &&
              ptr_B_scale->size(1) == scale_n &&
              ptr_B_scale->size(2) == scale_k && ptr_B_scale->is_contiguous(),
          "block fp8 ptr_B_scale must be a contiguous [num_experts, "
          "ceil(N/128), ceil(K/128)] tensor");
    }
  }

#define CALL_KERNEL_WITH_POLICY(POLICY)                \
  grouped_gemm::kernel_functor<POLICY>(                \
      dpcpp_queue,                                     \
      ptr_A.data_ptr(),                                \
      ptr_A_scale ? ptr_A_scale->data_ptr() : nullptr, \
      ptr_B.data_ptr(),                                \
      ptr_B_scale ? ptr_B_scale->data_ptr() : nullptr, \
      ptr_bias ? ptr_bias->data_ptr() : nullptr,       \
      ptr_D.data_ptr(),                                \
      rows_per_expert.data_ptr(),                      \
      N,                                               \
      K,                                               \
      groups)

// Dispatches the high-occupancy path to the wave-quantization-selected tile.
// FAMILY is a dtype prefix providing FAMILY##_policy (256x256) plus the
// _256x128 / _128x256 / _128x128 tile variants. FAM is the PrefillFamily tag.
#define DISPATCH_PREFILL_TILE(FAMILY, FAM)                           \
  switch (grouped_gemm::pick_prefill_tile(                           \
      grouped_gemm::PrefillFamily::FAM, ptr_A.size(0), N, groups)) { \
    case grouped_gemm::PrefillTile::k256x128: {                      \
      using moe_policy = grouped_gemm::FAMILY##_256x128_policy;      \
      CALL_KERNEL_WITH_POLICY(moe_policy);                           \
      break;                                                         \
    }                                                                \
    case grouped_gemm::PrefillTile::k128x256: {                      \
      using moe_policy = grouped_gemm::FAMILY##_128x256_policy;      \
      CALL_KERNEL_WITH_POLICY(moe_policy);                           \
      break;                                                         \
    }                                                                \
    case grouped_gemm::PrefillTile::k128x128: {                      \
      using moe_policy = grouped_gemm::FAMILY##_128x128_policy;      \
      CALL_KERNEL_WITH_POLICY(moe_policy);                           \
      break;                                                         \
    }                                                                \
    default: {                                                       \
      using moe_policy = grouped_gemm::FAMILY##_policy;              \
      CALL_KERNEL_WITH_POLICY(moe_policy);                           \
    }                                                                \
  }

  if (A_dtype == at::kBFloat16) {
    if (avg_tokens_cnt > 32) {
      DISPATCH_PREFILL_TILE(moe_bf16, kBF16);
    } else if (avg_tokens_cnt > 4) {
      using moe_policy = grouped_gemm::moe_bf16_mid_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    } else {
      if (K >= 1024) {
        using moe_policy = grouped_gemm::moe_bf16_decode_k64_policy;
        CALL_KERNEL_WITH_POLICY(moe_policy);
      } else {
        using moe_policy = grouped_gemm::moe_bf16_decode_policy;
        CALL_KERNEL_WITH_POLICY(moe_policy);
      }
    }
  } else if (A_dtype == at::kHalf) {
    if (avg_tokens_cnt > 32) {
      using moe_policy = grouped_gemm::moe_fp16_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    } else if (avg_tokens_cnt > 4) {
      using moe_policy = grouped_gemm::moe_fp16_mid_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    } else {
      using moe_policy = grouped_gemm::moe_fp16_decode_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    }
  } else if (
      A_dtype == at::kFloat8_e4m3fn && ptr_A_scale &&
      ptr_A_scale->dtype() == at::kFloat8_e8m0fnu) {
    // After cutlass PR #570 the optimized block-scaled MXFP mainloop
    // handles unaligned M directly (the 2D scale loader rounds the
    // surface width up to 4-byte alignment internally), so we no longer
    // need the scalar scale-load fallback.
    if (ptr_B.dtype() == at::kFloat4_e2m1fn_x2) {
      // W4A8: MXFP8 activation (A=e4m3) x MXFP4 weight (B=e2m1). Same e8m0
      // block scales as the symmetric recipes; only the weight is 4-bit.
      if (avg_tokens_cnt > 32) {
        using moe_policy = grouped_gemm::moe_w4a8_policy;
        CALL_KERNEL_WITH_POLICY(moe_policy);
      } else {
        using moe_policy = grouped_gemm::moe_w4a8_mid_policy;
        CALL_KERNEL_WITH_POLICY(moe_policy);
      }
    } else if (avg_tokens_cnt > 32) {
      DISPATCH_PREFILL_TILE(moe_mxfp8, kMXFP8);
    } else if (avg_tokens_cnt > 4) {
      using moe_policy = grouped_gemm::moe_mxfp8_mid_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    } else {
      using moe_policy = grouped_gemm::moe_mxfp8_decode_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    }
  } else if (
      A_dtype == at::kFloat4_e2m1fn_x2 && ptr_A_scale &&
      ptr_A_scale->dtype() == at::kFloat8_e8m0fnu) {
    if (avg_tokens_cnt > 32) {
      if (K <= 1024 && N >= 1024) {
        if (avg_tokens_cnt <= 2048) {
          using moe_policy = grouped_gemm::moe_mxfp4_256x128_policy;
          CALL_KERNEL_WITH_POLICY(moe_policy);
        } else {
          using moe_policy = grouped_gemm::moe_mxfp4_downproj_wide_policy;
          CALL_KERNEL_WITH_POLICY(moe_policy);
        }
      } else {
        DISPATCH_PREFILL_TILE(moe_mxfp4, kMXFP4);
      }
    } else if (avg_tokens_cnt > 4) {
      using moe_policy = grouped_gemm::moe_mxfp4_mid_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    } else {
      using moe_policy = grouped_gemm::moe_mxfp4_decode_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    }
  } else if (
      A_dtype == at::kFloat8_e4m3fn && ptr_A_scale &&
      ptr_A_scale->dtype() == at::kFloat) {
    if (ptr_A_scale->numel() == 1) {
      // Per-tensor FP8: A scale is a single scalar [1], B scale is one scalar
      // per expert [E].
      if (avg_tokens_cnt > 32) {
        using moe_policy = grouped_gemm::moe_fp8pertensor_policy;
        CALL_KERNEL_WITH_POLICY(moe_policy);
      } else if (avg_tokens_cnt > 4) {
        using moe_policy = grouped_gemm::moe_fp8pertensor_mid_policy;
        CALL_KERNEL_WITH_POLICY(moe_policy);
      } else {
        if (K <= 512) {
          using moe_policy =
              grouped_gemm::moe_fp8pertensor_decode_shortk_policy;
          CALL_KERNEL_WITH_POLICY(moe_policy);
        } else if (N <= 1024) {
          using moe_policy =
              grouped_gemm::moe_fp8pertensor_decode_narrow_policy;
          CALL_KERNEL_WITH_POLICY(moe_policy);
        } else if (K <= 1024) {
          using moe_policy = grouped_gemm::moe_fp8pertensor_decode_lowk_policy;
          CALL_KERNEL_WITH_POLICY(moe_policy);
        } else {
          using moe_policy = grouped_gemm::moe_fp8pertensor_decode_policy;
          CALL_KERNEL_WITH_POLICY(moe_policy);
        }
      }
    } else if (avg_tokens_cnt > 32) {
      using moe_policy = grouped_gemm::moe_fp8block_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    } else if (avg_tokens_cnt > 4) {
      using moe_policy = grouped_gemm::moe_fp8block_mid_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    } else {
      using moe_policy = grouped_gemm::moe_fp8block_decode_policy;
      CALL_KERNEL_WITH_POLICY(moe_policy);
    }
  } else {
    TORCH_CHECK(
        false,
        "grouped_gemm_func only supports BF16/FP16/MXFP8/MXFP4/FP8(block) "
        "dtypes, but got: ",
        A_dtype);
  }
  return ptr_D;
}

}  // namespace gpu::cutlass_kernel
