#include <torch/all.h>
#include "gemm_xe2.h"
#include "gemm_xe2_impl.hpp"

// Tile selection heuristic for bf16 GEMM on Intel Xe2 (BMG).
//
// Two tile_n regimes:
//   tile_n=256 (16 N-tiles for N=4096): best when M is small (fewer total WGs)
//     or very large (compute-bound, larger tile = fewer waves).
//   tile_n=128 (32 N-tiles for N=4096): best for mid-range M where more WGs
//     improve EU occupancy.
//
// tile_m=384 (tile_n=128): for M in (256, 384] — single M-wave, no waste.
static int select_tile_m(int m) {
  if (m <= 8) return 8;
  if (m <= 16) return 16;
  if (m <= 32) return 32;
  if (m <= 64) return 64;
  if (m <= 128) return 128;
  if (m <= 256) return 256;
  if (m <= 384) return 384;  // 384x128, single M-wave
  // For M > 384: tile_n=128 variants give better occupancy up to ~1024,
  // then tile_n=256 wins for compute-bound large M.
  if (m > 1024) return 2560;  // sentinel for 256x256
  // For 384 < M <= 1024: 256x128 wins only when M is exact multiple of 256,
  // otherwise 128x128 is better due to lower per-WG overhead (16 vs 32 SGs).
  if (m % 256 == 0) return 2561;  // sentinel for 256x128
  return 1280;                    // sentinel for 128x128
}

torch::Tensor cutlass_gemm_xe2(
    sycl::queue& queue, const torch::Tensor& A, const torch::Tensor& B) {
  CHECK_DEVICE(A);
  CHECK_DEVICE(B);
  TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
  TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bf16");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(
      A.size(1) == B.size(0), "A [M,K] @ B [K,N]: K dimensions must match");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  int tm = select_tile_m(M);

  bool b_row_major = B.stride(1) == 1 && B.stride(0) == N;
  bool b_col_major = B.stride(0) == 1 && B.stride(1) == K;
  TORCH_CHECK(
      b_row_major || b_col_major,
      "B must be either row-major (contiguous) or column-major");

  // Dispatch table. Sentinels: 1280=128x128, 2561=256x128, 2560=256x256
  if (b_col_major) {
    switch (tm) {
      case 8:
        return run_gemm<Config_8_RC>(queue, A, B, M, N, K);
      case 16:
        return run_gemm<Config_16_RC>(queue, A, B, M, N, K);
      case 32:
        return run_gemm<Config_32_RC>(queue, A, B, M, N, K);
      case 64:
        return run_gemm<Config_64_RC>(queue, A, B, M, N, K);
      case 128:
        return run_gemm<Config_128_RC>(queue, A, B, M, N, K);
      case 256:
        return run_gemm<Config_256_RC>(queue, A, B, M, N, K);
      case 384:
        return run_gemm<Config_384_RC>(queue, A, B, M, N, K);
      case 1280:
        return run_gemm<Config_128n_RC>(queue, A, B, M, N, K);
      case 2561:
        return run_gemm<Config_256n_RC>(queue, A, B, M, N, K);
      case 2560:
        return run_gemm<Config_256_RC>(queue, A, B, M, N, K);
    }
  } else {
    switch (tm) {
      case 8:
        return run_gemm<Config_8_RR>(queue, A, B, M, N, K);
      case 16:
        return run_gemm<Config_16_RR>(queue, A, B, M, N, K);
      case 32:
        return run_gemm<Config_32_RR>(queue, A, B, M, N, K);
      case 64:
        return run_gemm<Config_64_RR>(queue, A, B, M, N, K);
      case 128:
        return run_gemm<Config_128_RR>(queue, A, B, M, N, K);
      case 256:
        return run_gemm<Config_256_RR>(queue, A, B, M, N, K);
      case 384:
        return run_gemm<Config_384_RR>(queue, A, B, M, N, K);
      case 1280:
        return run_gemm<Config_128n_RR>(queue, A, B, M, N, K);
      case 2561:
        return run_gemm<Config_256n_RR>(queue, A, B, M, N, K);
      case 2560:
        return run_gemm<Config_256_RR>(queue, A, B, M, N, K);
    }
  }
  TORCH_CHECK(false, "Invalid tile selection");
}
