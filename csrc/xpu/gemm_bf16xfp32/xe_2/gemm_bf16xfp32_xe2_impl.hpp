/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * BF16xFP32 GEMM implementation using DualGemm infrastructure from sycl-tla.
 *
 * Emulates FP32-precision GEMM using two BF16 DPAS operations:
 *   result = X * W_high + (X * W_low) * scale
 *
 * where W_high and W_low are BF16 decompositions of an FP32 weight matrix W.
 **************************************************************************************************/
#pragma once

#include <torch/all.h>
#include "csrc/utils.h"

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

// DualGemm infrastructure from applications/
#include "dual_gemm/kernel/xe_dual_gemm.hpp"
#include "dual_gemm/collective/xe_dual_gemm_mma.hpp"
#include "dual_gemm/collective/xe_dual_gemm_epilogue.hpp"

// Local epilogue
#include "bf16xfp32_epilogue.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace gemm_bf16xfp32 {
using namespace cute;

// Default policy: good for medium/large M (prefill).
struct policy_default {
  using TileShape = Shape<_128, _64, _64>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
      Layout<TileShape>,
      Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>>::TiledMMA;
  static constexpr int PipelineStages = 2;
};

// Small M policy: for decode (M=1~32)
struct policy_small_m {
  using TileShape = Shape<_32, _64, _64>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
      Layout<TileShape>,
      Layout<Shape<_2, _2, _1>, Stride<_2, _1, _0>>>::TiledMMA;
  static constexpr int PipelineStages = 2;
};

// Medium M policy: for small batch prefill (M=17~64)
struct policy_medium_m {
  using TileShape = Shape<_64, _64, _64>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
      Layout<TileShape>,
      Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>>::TiledMMA;
  static constexpr int PipelineStages = 2;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel type assembly
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Policy>
struct DualGemmKernelType {
  using ElementA = bfloat16_t;
  using ElementB = bfloat16_t;
  using ElementAccumulator = float;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
  using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
  using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

  using TileShape = typename Policy::TileShape;
  using TiledMma = typename Policy::TiledMma;
  static constexpr int PipelineStages = Policy::PipelineStages;

  using GmemTiledCopyA = XE_2D_U16x16x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  // Individual epilogues (pass-through, do NOT write output)
  using EpilogueOp = cutlass::epilogue::fusion::LinCombEltAct<
      cutlass::epilogue::thread::Identity,
      ElementOutput,
      float,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;

  static constexpr bool WriteEpilogueOutput = false;

  using CollectiveEpilogue = cutlass::epilogue::collective::DualGemmEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      StrideC,
      ElementOutput,
      StrideD,
      FusionCallBacks,
      XE_2D_U32x8x16_LD_N,
      XE_2D_U32x8x16_ST_N,
      WriteEpilogueOutput>;

  // BF16xFP32 combining epilogue: D = acc_high + acc_low * scale
  using CollectiveBF16xFP32Epilogue =
      cutlass::epilogue::collective::BF16xFP32Epilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,  // No source C
          StrideC,
          ElementOutput,
          StrideD,
          void,  // No load op
          XE_2D_U32x8x16_ST_N>;

  // Mainloop: shared A, dual B (W_high, W_low)
  using CollectiveMainloop = cutlass::gemm::collective::DualGemmMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementA,
      StrideA,
      ElementB,
      StrideB,
      TiledMma,
      GmemTiledCopyA,
      GmemTiledCopyB>;

  // Full kernel
  using GemmKernel = cutlass::gemm::kernel::DualGemm<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      CollectiveEpilogue,
      CollectiveBF16xFP32Epilogue>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// SYCL kernel name tags
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Policy>
class GemmBF16xFP32KernelName;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Launch function
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Policy>
void launch_gemm_bf16xfp32(
    sycl::queue& queue,
    const bfloat16_t* A,       // [M, K]
    const bfloat16_t* B_high,  // [K, N]
    const bfloat16_t* B_low,   // [K, N]
    float* D,                  // [M, N] (or [splits, M, N] when splits > 1)
    int M,
    int N,
    int K,
    float scale,
    int splits = 1) {
  using Kernel = typename DualGemmKernelType<Policy>::GemmKernel;
  using StrideA = typename DualGemmKernelType<Policy>::StrideA;
  using StrideB = typename DualGemmKernelType<Policy>::StrideB;
  using StrideD = typename DualGemmKernelType<Policy>::StrideD;

  const int Kp = K / splits;

  // A [M, K] as (M, Kp, L): row stride = K, L step advances Kp along K.
  StrideA stride_A;
  cute::get<0>(stride_A) = static_cast<int64_t>(K);
  cute::get<2>(stride_A) = static_cast<int64_t>(Kp);

  // B [K, N] as (N, Kp, L): N contiguous, K stride = N, L step = Kp * N.
  StrideB stride_B;
  cute::get<1>(stride_B) = static_cast<int64_t>(N);
  cute::get<2>(stride_B) = static_cast<int64_t>(Kp) * N;

  // D [L, M, N] as (M, N, L): N contiguous, M stride = N, L step = M * N.
  StrideD stride_D;
  cute::get<0>(stride_D) = static_cast<int64_t>(N);
  cute::get<2>(stride_D) = static_cast<int64_t>(M) * N;

  using EpilogueArguments = typename Kernel::EpilogueArguments0;
  EpilogueArguments epilogue_args0{
      {1.0f, 0.0f}, nullptr, stride_D, nullptr, stride_D};
  EpilogueArguments epilogue_args1{
      {1.0f, 0.0f}, nullptr, stride_D, nullptr, stride_D};

  using BF16xFP32EpilogueArguments =
      typename Kernel::DualGemmElemActEpilogueArguments;
  BF16xFP32EpilogueArguments bf16xfp32_args{D, stride_D, scale};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  const auto mode = (splits > 1) ? cutlass::gemm::GemmUniversalMode::kBatched
                                 : cutlass::gemm::GemmUniversalMode::kGemm;

  typename Kernel::Arguments arguments{
      mode,
      {M, N, Kp, splits},
      {A, stride_A, B_high, stride_B, B_low, stride_B},
      epilogue_args0,
      epilogue_args1,
      bf16xfp32_args,
      hw_info};

  TORCH_CHECK(
      Kernel::can_implement(arguments),
      "gemm_bf16xfp32: problem size not supported by kernel. "
      "M=",
      M,
      " N=",
      N,
      " K=",
      K,
      " splits=",
      splits);

  size_t workspace_size = Kernel::get_workspace_size(arguments);
  TORCH_CHECK(
      workspace_size == 0, "gemm_bf16xfp32: unexpected workspace requirement");

  typename Kernel::Params params =
      Kernel::to_underlying_arguments(arguments, nullptr);

  dim3 block = Kernel::get_block_shape();
  dim3 grid = Kernel::get_grid_shape(params);

  sycl::range<3> local(1, 1, block.x);
  sycl::range<3> global(grid.z, grid.y, grid.x);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{
      syclex::sub_group_size<Kernel::DispatchPolicy::SubgroupSize>,
      intelex::grf_size<256>};

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GemmBF16xFP32KernelName<Policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          // SharedStorageSize is 0 for DualGemm
          char* smem_buf = nullptr;
          Kernel kernel_op;
          kernel_op(params, smem_buf);
        });
  });
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Split-K partial reduction
///////////////////////////////////////////////////////////////////////////////////////////////////

class ReduceSplitsKernelName;

inline void launch_reduce_splits(
    sycl::queue& queue,
    const float* partials,  // [splits, M, N]
    float* D,               // [M, N]
    int splits,
    int M,
    int N) {
  const int64_t mn = static_cast<int64_t>(M) * N;
  constexpr int kWg = 256;
  const int64_t n_groups = (mn + kWg - 1) / kWg;

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<ReduceSplitsKernelName>(
        sycl::nd_range<1>{
            sycl::range<1>(static_cast<size_t>(n_groups) * kWg),
            sycl::range<1>(kWg)},
        [=](sycl::nd_item<1> item) {
          const int64_t idx = static_cast<int64_t>(item.get_global_linear_id());
          if (idx >= mn) {
            return;
          }
          float acc = 0.0f;
          const float* p = partials + idx;
          for (int s = 0; s < splits; ++s) {
            acc += p[static_cast<int64_t>(s) * mn];
          }
          D[idx] = acc;
        });
  });
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Top-level dispatch (selects tile policy based on M)
///////////////////////////////////////////////////////////////////////////////////////////////////

inline at::Tensor gemm_bf16xfp32_xe2_impl(
    at::Tensor& A,       // [M, K] bf16
    at::Tensor& B_high,  // [K, N] bf16 (transposed weight)
    at::Tensor& B_low,   // [K, N] bf16 (transposed weight)
    double scale) {
  TORCH_CHECK(A.dim() == 2, "A must be 2D [M, K]");
  TORCH_CHECK(B_high.dim() == 2, "B_high must be 2D [K, N]");
  TORCH_CHECK(B_low.dim() == 2, "B_low must be 2D [K, N]");
  TORCH_CHECK(A.dtype() == at::kBFloat16, "A must be BF16");
  TORCH_CHECK(B_high.dtype() == at::kBFloat16, "B_high must be BF16");
  TORCH_CHECK(B_low.dtype() == at::kBFloat16, "B_low must be BF16");
  TORCH_CHECK(A.is_xpu(), "A must be on XPU");
  TORCH_CHECK(B_high.is_xpu(), "B_high must be on XPU");
  TORCH_CHECK(B_low.is_xpu(), "B_low must be on XPU");
  TORCH_CHECK(
      A.device() == B_high.device() && A.device() == B_low.device(),
      "A, B_high, and B_low must be on the same XPU device");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_high.is_contiguous(), "B_high must be contiguous");
  TORCH_CHECK(B_low.is_contiguous(), "B_low must be contiguous");

  int M = A.size(0);
  int K = A.size(1);
  int N = B_high.size(1);

  TORCH_CHECK(B_high.size(0) == K, "B_high.size(0) must match A.size(1) (K)");
  TORCH_CHECK(B_low.size(0) == K, "B_low.size(0) must match A.size(1) (K)");
  TORCH_CHECK(
      B_low.size(1) == N, "B_low.size(1) must match B_high.size(1) (N)");
  TORCH_CHECK(N % 4 == 0, "N must be divisible by 4");

  // Allocate output [M, N] fp32
  auto D = at::empty({M, N}, A.options().dtype(at::kFloat));

  auto& queue = at::xpu::getCurrentXPUStream(A.device().index()).queue();

  const bfloat16_t* ptr_A = reinterpret_cast<const bfloat16_t*>(A.data_ptr());
  const bfloat16_t* ptr_B_high =
      reinterpret_cast<const bfloat16_t*>(B_high.data_ptr());
  const bfloat16_t* ptr_B_low =
      reinterpret_cast<const bfloat16_t*>(B_low.data_ptr());
  float* ptr_D = reinterpret_cast<float*>(D.data_ptr());

  float scale_f = static_cast<float>(scale);

  constexpr int kEusPerXeCore = 8;  // Xe2 (BMG): 8 XVEs per Xe-core
  int n_eus = static_cast<int>(
      queue.get_device().get_info<sycl::info::device::max_compute_units>());
  int n_cores = n_eus / kEusPerXeCore;
  if (n_cores < 1) {
    n_cores = 1;
  }

  const int fill_threshold = (n_cores + 1) / 2;
  const int grid_n = (N + 63) / 64;
  auto num_wgs = [&](int blk_m) { return ((M + blk_m - 1) / blk_m) * grid_n; };

  if (num_wgs(128) >= fill_threshold) {
    launch_gemm_bf16xfp32<policy_default>(
        queue, ptr_A, ptr_B_high, ptr_B_low, ptr_D, M, N, K, scale_f);
  } else if (num_wgs(64) >= fill_threshold) {
    launch_gemm_bf16xfp32<policy_medium_m>(
        queue, ptr_A, ptr_B_high, ptr_B_low, ptr_D, M, N, K, scale_f);
  } else {
    const int base_wgs = num_wgs(32);
    const int max_wgs = 2 * n_cores;
    int splits = 1;
    int fill_wgs = base_wgs;  // best occupancy so far (splits == 1)
    for (int s = 2; base_wgs * s <= max_wgs; ++s) {
      if (K % s != 0 || (K / s) % 64 != 0) {
        continue;
      }
      const int wgs = base_wgs * s;
      if (wgs >= n_cores) {  // first split that fully occupies -> take it
        splits = s;
        fill_wgs = wgs;
        break;
      }
      if (wgs > fill_wgs) {  // otherwise keep the most-filling split
        splits = s;
        fill_wgs = wgs;
      }
    }

    if (splits > 1) {
      auto D_ws = at::empty({splits, M, N}, A.options().dtype(at::kFloat));
      float* ptr_D_ws = reinterpret_cast<float*>(D_ws.data_ptr());
      launch_gemm_bf16xfp32<policy_small_m>(
          queue,
          ptr_A,
          ptr_B_high,
          ptr_B_low,
          ptr_D_ws,
          M,
          N,
          K,
          scale_f,
          splits);
      launch_reduce_splits(queue, ptr_D_ws, ptr_D, splits, M, N);
    } else {
      launch_gemm_bf16xfp32<policy_small_m>(
          queue, ptr_A, ptr_B_high, ptr_B_low, ptr_D, M, N, K, scale_f);
    }
  }

  return D;
}

}  // namespace gemm_bf16xfp32
