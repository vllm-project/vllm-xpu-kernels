// SPDX-License-Identifier: Apache-2.0
//
// fp8_gemv_w8a16  : out = fp16(x @ (w * scale)^T), x [M,K] fp16 (M <= 8),
//                   w [N,K] row-major fp8-e4m3, scale fp32 [1].
// fp8_gemv2_w8a16 : two weights sharing one input, one launch
//                   (GDN in_proj_qkvz + in_proj_ba).
//
// Single kernel launch per call, no allocation besides the outputs. See
// fp8_gemv.hpp for the kernel design.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include <ATen/DeviceGuard.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>

#include <optional>
#include <tuple>
#include <utility>

#include "utils.h"
#include "fp8_gemv.hpp"
#include "../fp8_gemv_interface.h"

namespace vllm::fp8_gemv {

namespace syclex = sycl::ext::oneapi::experimental;

// ------------------------------------------------------------ kernel -----
// Work-group = RG row blocks (16 rows each) x KS K-split sub-groups.
template <int MP, int KS, int RG, int NSEG, typename Scalar>
struct Fp8GemvKernel {
  const Scalar* a;
  int a_pitch_bytes;
  int M;
  int K;
  Seg<Scalar> seg0;
  Seg<Scalar> seg1;

  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto* part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RG * KS * MP * kSg]>(item.get_group());
    const int g = int(item.get_group(0));
    const bool second = NSEG == 2 && g >= seg1.wg_begin;
    const Seg<Scalar>& seg = second ? seg1 : seg0;
    const u32x8 none[1][kChunkSteps] = {};
    const int wg_row0 = (g - seg.wg_begin) * RG * kRowBlock;
    const GlobalA<MP, Scalar> ga{a, K, M, a_pitch_bytes};
    if constexpr (NSEG == 2 && RG >= 2) {
      // A segment that fits in half a work-group (e.g. the 32-row GDN
      // in_proj_ba next to in_proj_qkvz) re-maps the sub-groups to twice the
      // K split, so its lone work-group finishes early instead of forming
      // the kernel's tail.
      if (seg.N - wg_row0 <= (RG / 2) * kRowBlock) {
        gemv_body<MP, KS * 2, RG / 2>(
            item, seg, wg_row0, ga, part, M, K, none, false);
        return;
      }
    }
    gemv_body<MP, KS, RG>(item, seg, wg_row0, ga, part, M, K, none, false);
#endif
  }
};

template <int MP, int KS, int RG, int NSEG, typename Scalar>
static void launch_gemv(
    sycl::queue& q,
    const Scalar* a,
    int a_pitch_bytes,
    int M,
    int K,
    const Seg<Scalar>& s0,
    const Seg<Scalar>& s1,
    int total_wgs) {
  // enqueue-functions launch: ~4 us less host time than queue::parallel_for
  syclex::nd_launch(
      q,
      sycl::nd_range<1>(size_t(total_wgs) * RG * KS * kSg, RG * KS * kSg),
      Fp8GemvKernel<MP, KS, RG, NSEG, Scalar>{a, a_pitch_bytes, M, K, s0, s1});
}

template <int KS, int RG, int NSEG, typename Scalar>
static void launch_mp(
    sycl::queue& q,
    const Scalar* a,
    int a_pitch_bytes,
    int M,
    int K,
    const Seg<Scalar>& s0,
    const Seg<Scalar>& s1,
    int total_wgs) {
  if (M == 1)
    launch_gemv<1, KS, RG, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1, total_wgs);
  else if (M == 2)
    launch_gemv<2, KS, RG, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1, total_wgs);
  else if (M <= 4)
    launch_gemv<4, KS, RG, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1, total_wgs);
  else
    launch_gemv<8, KS, RG, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1, total_wgs);
}

// Work decomposition, picked per call from (M, N) (measured on B70, K=2048):
//  * tiny N (<= 512, e.g. the 32-row GDN in_proj_ba): K split 16 ways so a
//    few row blocks still spread over enough sub-groups;
//  * M >= 5 and N >= 4096: KS = 2, RG = 4 -- fewer, longer sub-groups and
//    a packed (128 B / store) epilogue; ~2% faster than the default there;
//  * otherwise KS = 4, RG = 1 (4 sub-groups per 16 rows).
template <int KS, int RG, int NSEG, typename Scalar>
static void run_cfg(
    sycl::queue& q,
    const Scalar* a,
    int a_pitch_bytes,
    int M,
    int K,
    Seg<Scalar> s0,
    Seg<Scalar> s1) {
  constexpr int rows = RG * kRowBlock;
  const int w0 = (s0.N + rows - 1) / rows;
  s0.wg_begin = 0;
  s1.wg_begin = w0;
  const int total = w0 + (NSEG == 2 ? (s1.N + rows - 1) / rows : 0);
  launch_mp<KS, RG, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1, total);
}

template <int NSEG, typename Scalar>
static void run_gemv(
    sycl::queue& q,
    const Scalar* a,
    int a_pitch_bytes,
    int M,
    int K,
    Seg<Scalar> s0,
    Seg<Scalar> s1) {
  // Two weights: schedule the smaller one first so its work-groups do not
  // form a tail behind the large one.
  if (NSEG == 2 && s1.N < s0.N) std::swap(s0, s1);
  const int64_t n_total = int64_t(s0.N) + (NSEG == 2 ? s1.N : 0);
  if (n_total <= 512)
    run_cfg<16, 1, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1);
  else if (M >= 5 && n_total >= 4096)
    run_cfg<2, 4, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1);
  else
    run_cfg<4, 1, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1);
}

// ------------------------------------------------------ host checks -----
static inline bool aligned64(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 63u) == 0;
}

// Activation [.., K] fp16 viewable as [M, K] with a 2D-block-IO legal pitch.
static bool
act_ok(const torch::Tensor& x, int64_t& M, int64_t& K, int64_t& pitch) {
  if (!x.is_xpu() ||
      !(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) ||
      x.dim() < 2)
    return false;
  K = x.size(-1);
  if (K <= 0 || x.stride(-1) != 1) return false;
  M = x.numel() / K;
  if (x.dim() == 2) {
    pitch = x.stride(0);
  } else {
    if (!x.is_contiguous()) return false;
    pitch = K;
  }
  // 2D block IO: base 64 B aligned, pitch multiple of 16 B, >= width.
  return M >= 1 && M <= kMaxM && aligned64(x.data_ptr()) &&
         (pitch * 2) % 16 == 0 && pitch >= K && pitch * 2 < (int64_t(1) << 24);
}

static bool
weight_ok(const torch::Tensor& w, const torch::Tensor& x, int64_t K) {
  return w.device() == x.device() && w.dim() == 2 &&
         w.scalar_type() == at::ScalarType::Float8_e4m3fn && w.size(1) == K &&
         w.size(0) >= 1 && w.size(0) < (int64_t(1) << 24) && w.stride(1) == 1 &&
         w.stride(0) == K && aligned64(w.data_ptr()) && K % kChunkK == 0 &&
         K < (int64_t(1) << 24);
}

static bool
scale_ok(const std::optional<torch::Tensor>& s, const torch::Tensor& x) {
  return s.has_value() && s->device() == x.device() &&
         s->scalar_type() == at::kFloat && s->numel() == 1;
}

bool fp8_gemv_w8a16_supported(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const std::optional<torch::Tensor>& scale,
    const std::optional<torch::Tensor>& bias) {
  int64_t M, K, pitch;
  if (bias.has_value() || !act_ok(x, M, K, pitch)) return false;
  // Dispatch policy: the kernels are correct for M <= 8, but only M == 1 is
  // tuned/validated to beat oneDNN on every target shape; M > 1 keeps oneDNN.
  if (M != 1) return false;
  if (!weight_ok(w, x, K) || !scale_ok(scale, x)) return false;
  // Architecture is immutable per device; cache the last query.
  static thread_local int last_dev = -1;
  static thread_local bool last_ok = false;
  const int dev = x.get_device();
  if (dev != last_dev) {
    last_ok = vllm::xpu::is_bmg(dev);
    last_dev = dev;
  }
  return last_ok;
}

// Same predicate for the NT view used by fp8_gemm_w8a16(A, B = W^T [K, N]).
bool fp8_gemv_w8a16_supported_nt(
    const torch::Tensor& a,
    const torch::Tensor& b_kn,
    const std::optional<torch::Tensor>& scale,
    const std::optional<torch::Tensor>& bias) {
  if (b_kn.dim() != 2) return false;
  return fp8_gemv_w8a16_supported(a, b_kn.t(), scale, bias);
}

static void
check_common(const torch::Tensor& x, int64_t& M, int64_t& K, int64_t& pitch) {
  TORCH_CHECK(x.is_xpu(), "fp8_gemv: x must be an XPU tensor");
  TORCH_CHECK(
      (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16),
      "fp8_gemv: x must be float16 or bfloat16");
  TORCH_CHECK(x.dim() >= 2, "fp8_gemv: x must be at least 2-D");
  TORCH_CHECK(x.stride(-1) == 1, "fp8_gemv: x last dim must be contiguous");
  TORCH_CHECK(
      act_ok(x, M, K, pitch),
      "fp8_gemv: x must be viewable as [M<=8, K] with 64 B aligned base and a "
      "16 B multiple row pitch (got shape ",
      x.sizes(),
      ", strides ",
      x.strides(),
      ")");
}

static void check_weight(
    const torch::Tensor& w,
    const torch::Tensor& s,
    const torch::Tensor& x,
    int64_t K,
    const char* name) {
  TORCH_CHECK(w.device() == x.device(), "fp8_gemv: ", name, " device mismatch");
  TORCH_CHECK(
      w.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "fp8_gemv: ",
      name,
      " must be float8_e4m3fn");
  TORCH_CHECK(
      w.dim() == 2 && w.size(1) == K,
      "fp8_gemv: ",
      name,
      " must be [N, K=",
      K,
      "], got ",
      w.sizes());
  TORCH_CHECK(
      w.stride(1) == 1 && w.stride(0) == K,
      "fp8_gemv: ",
      name,
      " must be row-major contiguous [N, K]");
  TORCH_CHECK(
      aligned64(w.data_ptr()), "fp8_gemv: ", name, " must be 64 B aligned");
  TORCH_CHECK(
      K % kChunkK == 0 && K < (int64_t(1) << 24),
      "fp8_gemv: K must be a "
      "multiple of 64, got ",
      K);
  TORCH_CHECK(w.size(0) < (int64_t(1) << 24), "fp8_gemv: N too large");
  TORCH_CHECK(
      s.device() == x.device() && s.scalar_type() == at::kFloat &&
          s.numel() == 1,
      "fp8_gemv: scale must be a 1-element float32 tensor on the same device");
}

static std::vector<int64_t> out_shape(const torch::Tensor& x, int64_t N) {
  auto sz = x.sizes().vec();
  sz.back() = N;
  return sz;
}

// ------------------------------------------------------ host entries -----
template <typename Scalar>
torch::Tensor fp8_gemv_w8a16_impl(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& scale) {
  int64_t M, K, pitch;
  check_common(x, M, K, pitch);
  check_weight(w, scale, x, K, "w");
  const at::DeviceGuard guard(x.device());
  const int64_t N = w.size(0);
  auto out = at::empty(out_shape(x, N), x.options());
  auto& q = vllm::xpu::vllmGetQueue();
  Seg<Scalar> s0{
      reinterpret_cast<const uint8_t*>(w.data_ptr()),
      scale.data_ptr<float>(),
      reinterpret_cast<Scalar*>(out.data_ptr()),
      int(N),
      int(N),
      0};
  run_gemv<1>(
      q,
      reinterpret_cast<const Scalar*>(x.data_ptr()),
      int(pitch * 2),
      int(M),
      int(K),
      s0,
      s0);
  return out;
}

torch::Tensor fp8_gemv_w8a16(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& scale) {
  if (x.scalar_type() == at::kHalf)
    return fp8_gemv_w8a16_impl<sycl::half>(x, w, scale);
  return fp8_gemv_w8a16_impl<sycl::ext::oneapi::bfloat16>(x, w, scale);
}

// Bridge entry for the existing fp8_gemm_w8a16(A, B=[K,N] NT view, ...) op:
// full checks first, returns nullopt (caller keeps its oneDNN path) when
// unsupported.
std::optional<torch::Tensor> try_fp8_gemv_w8a16(
    const torch::Tensor& a,
    const torch::Tensor& b_kn,
    const std::optional<torch::Tensor>& scale,
    const std::optional<torch::Tensor>& bias) {
  if (!fp8_gemv_w8a16_supported_nt(a, b_kn, scale, bias)) return std::nullopt;
  return fp8_gemv_w8a16(a, b_kn.t(), *scale);
}

}  // namespace vllm::fp8_gemv
