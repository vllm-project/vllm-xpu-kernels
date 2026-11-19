// Int4 MoE GEMV for decode-sized grouped GEMMs (a few rows per call).
//
// Same layout as the int4 path of cutlass_grouped_gemm_interface:
//   A        [rows, K]      fp16/bf16, rows grouped by expert
//   B        [E, N, K / 2]  int8, two signed 4-bit values per byte, element
//                           2j in the low nibble (see implement_zp)
//   B_scale  [E, N, K / G]  same dtype as A, symmetric per-group scale
//   rows_per_expert [E]     int32
//   D        [rows, N]      same dtype as A
//
// One work-group handles one expert that has rows and a block of N. Each
// sub-group computes NR output columns for up to MC rows at a time; its lanes
// split K into chunks of KCV values.

#include <sycl/sycl.hpp>

#include <algorithm>

#include "csrc/utils.h"
#include "xpu/ops.h"

namespace vllm {

constexpr int kSubGroupSize = 16;

template <typename T, int NR, int SGS_PER_WG, int KCV, int MC>
struct MoeInt4GemvKernel {
  const T* A;
  const int8_t* B;
  const T* S;
  const int* rows;
  T* D;
  int N, K, E, G, nblocks;
  sycl::local_accessor<int, 1> slm;

  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void
  operator()(sycl::nd_item<1> it) const {
    auto sg = it.get_sub_group();
    const int wg = it.get_group(0);
    const int slot = wg / nblocks;
    const int nb = wg % nblocks;

    // Find the slot-th expert with rows and its first row: each work-item
    // scans a slice of rows_per_expert.
    {
      const int wi = it.get_local_id(0);
      const int ws = it.get_local_range(0);
      const int chunk = (E + ws - 1) / ws;
      const int i0 = sycl::min(E, wi * chunk);
      const int i1 = sycl::min(E, i0 + chunk);
      int cnt = 0, rsum = 0;
      for (int i = i0; i < i1; ++i) {
        const int r = rows[i];
        cnt += r > 0;
        rsum += r;
      }
      const int ecnt = sycl::exclusive_scan_over_group(
          it.get_group(), cnt, sycl::plus<int>());
      const int erows = sycl::exclusive_scan_over_group(
          it.get_group(), rsum, sycl::plus<int>());
      if (wi == 0) slm[0] = -1;
      sycl::group_barrier(it.get_group());
      if (slot >= ecnt && slot < ecnt + cnt) {
        int seen = ecnt, pre = erows;
        for (int i = i0; i < i1; ++i) {
          const int r = rows[i];
          if (r > 0) {
            if (seen == slot) {
              slm[0] = i;
              slm[1] = pre;
              break;
            }
            ++seen;
          }
          pre += r;
        }
      }
    }
    sycl::group_barrier(it.get_group());
    const int e = slm[0];
    const int pre = slm[1];
    if (e < 0) return;

    const int m_e = rows[e];
    const int sgid = sg.get_group_linear_id();
    const int lane = sg.get_local_linear_id();
    const int n0 = (nb * SGS_PER_WG + sgid) * NR;
    if (n0 >= N) return;
    const int ngroups = K / G;
    const int8_t* Be = B + (int64_t)e * N * (K / 2);
    const T* Se = S + (int64_t)e * N * ngroups;
    using WV = sycl::vec<uint32_t, KCV / 8>;
    using AV = sycl::vec<T, 8>;

    for (int m0 = 0; m0 < m_e; m0 += MC) {
      const int mc = sycl::min(MC, m_e - m0);
      float acc[NR][MC];
#pragma unroll
      for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int m = 0; m < MC; ++m)
          acc[r][m] = 0.f;

      for (int kb = lane * KCV; kb < K; kb += kSubGroupSize * KCV) {
        float a[MC][KCV];
#pragma unroll
        for (int m = 0; m < MC; ++m) {
          if (m < mc) {
            const AV* ap = reinterpret_cast<const AV*>(
                A + (int64_t)(pre + m0 + m) * K + kb);
#pragma unroll
            for (int v = 0; v < KCV / 8; ++v) {
              const AV x = ap[v];
#pragma unroll
              for (int j = 0; j < 8; ++j)
                a[m][v * 8 + j] = static_cast<float>(x[j]);
            }
          }
        }
        const int g = kb / G;
#pragma unroll
        for (int r = 0; r < NR; ++r) {
          const int n = n0 + r;
          if (n >= N) break;
          const WV w =
              *reinterpret_cast<const WV*>(Be + (int64_t)n * (K / 2) + kb / 2);
          const float s = static_cast<float>(Se[(int64_t)n * ngroups + g]);
          float d[MC];
#pragma unroll
          for (int m = 0; m < MC; ++m)
            d[m] = 0.f;
#pragma unroll
          for (int v = 0; v < KCV / 8; ++v) {
            const uint32_t word = w[v];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              int q = (word >> (4 * j)) & 0xF;
              q = q >= 8 ? q - 16 : q;
#pragma unroll
              for (int m = 0; m < MC; ++m)
                d[m] += (float)q * a[m][v * 8 + j];
            }
          }
#pragma unroll
          for (int m = 0; m < MC; ++m)
            acc[r][m] += d[m] * s;
        }
      }

#pragma unroll
      for (int r = 0; r < NR; ++r) {
        const int n = n0 + r;
#pragma unroll
        for (int m = 0; m < MC; ++m) {
          const float v =
              sycl::reduce_over_group(sg, acc[r][m], sycl::plus<float>());
          if (lane == 0 && m < mc && n < N)
            D[(int64_t)(pre + m0 + m) * N + n] = static_cast<T>(v);
        }
      }
    }
  }
};

template <typename scalar_t>
void call_moe_int4_gemv(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& B_scale,
    const torch::Tensor& rows_per_expert,
    torch::Tensor& D,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    int64_t group_size) {
  using T = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  constexpr int NR = 4, SGS = 8, KCV = 32, MC = 2;
  const int nblocks = (int)((N + NR * SGS - 1) / (NR * SGS));
  // At most one active expert per input row.
  const int64_t slots = std::min<int64_t>(A.size(0), num_experts);
  if (slots == 0) return;
  const sycl::range<1> local(SGS * kSubGroupSize);
  const sycl::range<1> global(slots * nblocks * SGS * kSubGroupSize);

  const T* a = reinterpret_cast<const T*>(A.data_ptr());
  const int8_t* b = reinterpret_cast<const int8_t*>(B.data_ptr());
  const T* s = reinterpret_cast<const T*>(B_scale.data_ptr());
  const int* r = rows_per_expert.data_ptr<int>();
  T* d = reinterpret_cast<T*>(D.data_ptr());

  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int, 1> slm(sycl::range<1>(2), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(global, local),
        MoeInt4GemvKernel<T, NR, SGS, KCV, MC>{
            a,
            b,
            s,
            r,
            d,
            (int)N,
            (int)K,
            (int)num_experts,
            (int)group_size,
            nblocks,
            slm});
  });
}

}  // namespace vllm

void moe_int4_gemv(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& B_scale,
    const torch::Tensor& rows_per_expert,
    torch::Tensor& D,
    int64_t N,
    int64_t K,
    int64_t num_experts) {
  TORCH_CHECK(
      A.is_contiguous() && B.is_contiguous() && B_scale.is_contiguous() &&
          D.is_contiguous(),
      "moe_int4_gemv: inputs must be contiguous");
  TORCH_CHECK(
      A.scalar_type() == at::kHalf || A.scalar_type() == at::kBFloat16,
      "moe_int4_gemv: A must be fp16 or bf16");
  TORCH_CHECK(
      B_scale.scalar_type() == A.scalar_type() &&
          D.scalar_type() == A.scalar_type(),
      "moe_int4_gemv: B_scale and D must have the dtype of A");
  TORCH_CHECK(B.scalar_type() == at::kChar, "moe_int4_gemv: B must be int8");
  TORCH_CHECK(
      rows_per_expert.scalar_type() == at::kInt &&
          rows_per_expert.numel() == num_experts,
      "moe_int4_gemv: rows_per_expert must be int32 [num_experts]");
  TORCH_CHECK(
      A.dim() == 2 && A.size(1) == K && D.dim() == 2 &&
          D.size(0) == A.size(0) && D.size(1) == N,
      "moe_int4_gemv: A must be [rows, K] and D [rows, N]");
  TORCH_CHECK(
      B.dim() == 3 && B.size(0) == num_experts && B.size(1) == N &&
          B.size(2) == K / 2,
      "moe_int4_gemv: B must be [num_experts, N, K / 2]");
  TORCH_CHECK(
      B_scale.dim() == 3 && B_scale.size(0) == num_experts &&
          B_scale.size(1) == N && K % B_scale.size(2) == 0,
      "moe_int4_gemv: B_scale must be [num_experts, N, K / group_size]");
  const int64_t group_size = K / B_scale.size(2);
  TORCH_CHECK(
      K % 512 == 0 && group_size % 32 == 0,
      "moe_int4_gemv: needs K % 512 == 0 and group_size % 32 == 0");

  if (A.scalar_type() == at::kHalf) {
    vllm::call_moe_int4_gemv<c10::Half>(
        A, B, B_scale, rows_per_expert, D, N, K, num_experts, group_size);
  } else {
    vllm::call_moe_int4_gemv<c10::BFloat16>(
        A, B, B_scale, rows_per_expert, D, N, K, num_experts, group_size);
  }
}
