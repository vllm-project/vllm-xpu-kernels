#pragma once

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "activation_utils.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace FusedMOE {

using namespace cute;

template <typename TB>
CUTE_DEVICE TB apply_scale(TB& x, float& y) {
  static_assert(
      is_any_of_v<TB, bfloat16_t, half_t>, "Only BF16 & FP16 are supported");
  uint16_t z = sycl::bit_cast<uint16_t>(x);
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
  if constexpr (is_same_v<TB, half_t>) {
    asm("{\n"
        ".decl Z_FP16 v_type=G type=HF num_elts=16 alias=<%0,0>\n"
        ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        "mul (M1, 16) Z_FP16(0,0)<1> Z_FP16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
        "}\n"
        : "+rw"(z)
        : "rw"(y));
  } else {
    asm("{\n"
        ".decl Z_BF16 v_type=G type=BF num_elts=16 alias=<%0,0>\n"
        ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        "mul (M1, 16) Z_BF16(0,0)<1> Z_BF16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
        "}\n"
        : "+rw"(z)
        : "rw"(y));
  }
#endif
  return sycl::bit_cast<TB>(z);
}

template <
    bool has_clamping,
    ActivationType activation_type,
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyC,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void gemm_up(
    ATensor const& A,   // (M,K)
    BTensor const& B1,  // (N,K)
    BTensor const& B2,  // (N,K)
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,   // (M,N)
    DTensor& C2,  // (M,N)
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma,
    float gemm1_clamp_limit) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B1.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b1 = get_block_2d_copy_B<GmemTiledCopyB>(mma, B1);
  auto copy_b2 = get_block_2d_copy_B<GmemTiledCopyB>(mma, B2);
  auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b1 = copy_b1.get_slice(local_id);
  auto thr_copy_b2 = copy_b2.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB1 = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tCrB2 = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB1 = thr_copy_b1.partition_sg_fragment_D(gB(_, _, 0));
  auto tBrB2 = thr_copy_b2.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB1 = thr_copy_b1.partition_S(gB);
  Tensor tBgB2 = thr_copy_b2.partition_S(gB);

  /* Partition C */
  auto tCrC1 = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC2 = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b1 = make_block_2d_prefetch(copy_b1);
  auto prefetch_b2 = make_block_2d_prefetch(copy_b2);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B1 = prefetch_b1.get_slice(local_id);
  auto thr_prefetch_B2 = prefetch_b2.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB1 = thr_prefetch_B1.partition_S(gB);
  auto pBgB2 = thr_prefetch_B2.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  clear(tCrC1);
  clear(tCrC2);

  using ElementB = typename BTensor::element_type;
  static constexpr bool is_B_fp8_type =
      std::is_same_v<ElementB, cutlass::float_e5m2_t> ||
      std::is_same_v<ElementB, cutlass::float_e4m3_t>;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b1, pBgB1(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b2, pBgB2(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    reorder(tArA, tCrA);

    copy(copy_b1, tBgB1(_, _, _, k_tile), tBrB1);
    reorder(tBrB1, tCrB1);
    cute::gemm(mma, tCrA, tCrB1, tCrC1);

    copy(copy_b2, tBgB2(_, _, _, k_tile), tBrB2);
    reorder(tBrB2, tCrB2);
    cute::gemm(mma, tCrA, tCrB2, tCrC2);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b1, pBgB1(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b2, pBgB2(_, _, _, k_tile_prefetch));
    }

    barrier_wait(barrier_scope);
  }

  if constexpr (is_B_fp8_type) {
    float B_scale = Scales[0];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC1.size(); ++i) {
      tCrC1(i) *= B_scale;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC2.size(); ++i) {
      tCrC2(i) *= B_scale;
    }
  }

  if (Bias != nullptr) {
    static constexpr auto ATOM_M =
        get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N =
        get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

    auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;

    static constexpr auto tile_m = get<0>(wg_tile);
    static constexpr auto tile_n = get<1>(wg_tile);

    // 32 * 64
    static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
    static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;

    int sg_local_id = cutlass::get_sub_group_local_id();
    static constexpr int sg_local_range = 16;
    int n_tile_start = wg_n * tile_n;
    int n_sg_start = sg_local_n_coord * SG_N;

    auto gemm_n = get<0>(B1.shape());

    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float1 = Bias[n_tile_start + n_sg_start + sg_local_n];
      float b_float2 = Bias[gemm_n + n_tile_start + n_sg_start + sg_local_n];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC1(sn * SG_M + sm) += b_float1;
        tCrC2(sn * SG_M + sm) += b_float2;
      }
    }
  }

  if constexpr (has_clamping) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC1.size(); ++i) {
      tCrC1(i) = sycl::fmin(tCrC1(i), gemm1_clamp_limit);
      tCrC2(i) = sycl::fmax(tCrC2(i), -gemm1_clamp_limit);
      tCrC2(i) = sycl::fmin(tCrC2(i), gemm1_clamp_limit);
    }
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < tCrC1.size(); ++i) {
    if constexpr (activation_type == ActivationType::SILU) {
      tCrC1(i) = silu_kernel(tCrC1(i)) * tCrC2(i);
    } else if constexpr (activation_type == ActivationType::GELU) {
      tCrC1(i) = gelu_kernel(tCrC1(i)) * tCrC2(i);
    } else if constexpr (activation_type == ActivationType::GELU_TANH) {
      tCrC1(i) = gelu_tanh_kernel(tCrC1(i)) * tCrC2(i);
    } else if constexpr (activation_type == ActivationType::SWIGLUOAI) {
      tCrC1(i) = swigluoai_and_mul(tCrC1(i), tCrC2(i), 1.702, 7.0);
    } else if constexpr (activation_type == ActivationType::SWIGLUSTEP) {
      tCrC1(i) = swiglustep_and_mul(tCrC1(i), tCrC2(i), 7.0);
    }
  }

  if constexpr (activation_type == ActivationType::RELU2_NO_MUL) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC1.size(); ++i) {
      tCrC1(i) = relu2_no_mul_kernel(tCrC1(i));
    }
  }

  reorder(tCrC1, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);

  if constexpr (activation_type == ActivationType::RELU2_NO_MUL) {
    auto copy_c2 = get_block_2d_copy_D<GmemTiledCopyC>(mma, C2);
    auto thr_copy_c2 = copy_c2.get_slice(local_id);
    auto tCgC2 = thr_copy_c2.partition_D(gC);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC2.size(); ++i) {
      tCrC2(i) = relu2_no_mul_kernel(tCrC2(i));
    }
    reorder(tCrC2, tCrC_out);
    copy(copy_c2, tCrC_out, tCgC2);
  }
}

template <
    bool has_clamping,
    ActivationType activation_type,
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyC,
    int GroupSize,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void gemm_up_4bits(
    ATensor const& A,   // (M,K)
    BTensor const& B1,  // (N,K)
    BTensor const& B2,  // (N,K)
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,   // (M,N)
    DTensor& C2,  // (M,N)
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma,
    float gemm1_clamp_limit) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  static constexpr int group_size = GroupSize;
  static constexpr int sg_local_range = 16;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B1.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b1 = get_block_2d_copy_B<GmemTiledCopyB>(mma, B1);
  auto copy_b2 = get_block_2d_copy_B<GmemTiledCopyB>(mma, B2);
  auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b1 = copy_b1.get_slice(local_id);
  auto thr_copy_b2 = copy_b2.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB1 = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tCrB2 = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB1 = thr_copy_b1.partition_sg_fragment_D(gB(_, _, 0));
  auto tBrB2 = thr_copy_b2.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB1 = thr_copy_b1.partition_S(gB);
  Tensor tBgB2 = thr_copy_b2.partition_S(gB);

  /* Partition C */
  auto tCrC1 = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC2 = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b1 = make_block_2d_prefetch(copy_b1);
  auto prefetch_b2 = make_block_2d_prefetch(copy_b2);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B1 = prefetch_b1.get_slice(local_id);
  auto thr_prefetch_B2 = prefetch_b2.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB1 = thr_prefetch_B1.partition_S(gB);
  auto pBgB2 = thr_prefetch_B2.partition_S(gB);

  const int prefetch_dist = 6;

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K =
      get<3>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto tile_k = get<2>(wg_tile);

  static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;
  static constexpr auto SG_K = tile_k / ATOM_K;  // BLK_K / ATOM_K;

  static constexpr auto thr_N = get<1>(tCrB1.shape());
  static constexpr auto channel_num = get<0>(get<0>(tCrB1.shape()));
  auto n_tile_start = wg_n * tile_n;

  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int sg_local_id = cutlass::get_sub_group_local_id();
  int n_sg_start = sg_local_n_coord * SG_N;
  int group_num = get<1>(A.shape()) / group_size;
  int x_idx = sg_local_id / channel_num;

  using scaleStoreType = conditional_t<is_same_v<TA, half_t>, half_t, float>;
  scaleStoreType scales1[thr_N * channel_num];
  scaleStoreType scales2[thr_N * channel_num];

  clear(tCrC1);
  clear(tCrC2);

  auto gemm_n = get<0>(B1.shape());
  const ElementS* Scales1 = Scales;
  const ElementS* Scales2 = Scales + gemm_n * group_num;

  auto prefetch_scales = [&](const ElementS* Scales_base, int group_idx) {
    auto next_scales_tensor = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<const ElementS*>(
                Scales_base + (n_tile_start + n_sg_start) * group_num +
                group_idx)),
        make_layout(
            make_shape(Int<SG_N>{}, Int<1>{}),
            make_stride(group_num, Int<1>{})));
    auto prefetch_scales = make_block_2d_prefetch<1>(
        make_shape(Int<SG_N>{}, Int<1>{}), next_scales_tensor);
    auto thr_prefetch_scales = prefetch_scales.get_slice(sg_local_id);
    auto pSgS = thr_prefetch_scales.partition_S(
        make_identity_tensor(make_shape(Int<SG_N>{}, Int<1>{})));
    prefetch(prefetch_scales, pSgS(_, 0, 0));
  };

  auto load_scales = [&](const ElementS* Scales_base,
                         scaleStoreType* scales,
                         int group_idx) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < thr_N; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < channel_num; ++c) {
        int real_idx = x_idx + c * (sg_local_range / channel_num);
        int sg_local_n = n * sg_local_range + real_idx;
        scaleStoreType scale;
        if constexpr (std::is_same_v<TB, int4_t>) {
          scale = Scales_base
              [(n_tile_start + n_sg_start + sg_local_n) * group_num +
               group_idx];
        } else if constexpr (std::is_same_v<TB, float_e2m1_t>) {
          uint32_t scale_u32 =
              Scales_base
                  [(n_tile_start + n_sg_start + sg_local_n) * group_num +
                   group_idx]
              << 23;
          scale =
              static_cast<scaleStoreType>(reinterpret_cast<float&>(scale_u32));
        }

        scales[n * channel_num + c] = scale;
      }
    }
  };

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b1, pBgB1(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b2, pBgB2(_, _, _, k_tile_prefetch));

    if (k_tile_prefetch * group_size < shape<1>(A)) {
      prefetch_scales(Scales1, k_tile_prefetch);
      prefetch_scales(Scales2, k_tile_prefetch);
    }
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    if (k_tile * tile_k % group_size == 0) {
      int group_idx = (k_tile * tile_k) / group_size;

      load_scales(Scales1, scales1, group_idx);
      load_scales(Scales2, scales2, group_idx);

      if ((group_idx + prefetch_dist) * group_size < shape<1>(A)) {
        prefetch_scales(Scales1, group_idx + prefetch_dist);
        prefetch_scales(Scales2, group_idx + prefetch_dist);
      }
    }

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    reorder(tArA, tCrA);

    copy(copy_b1, tBgB1(_, _, _, k_tile), tBrB1);
    reorder(tBrB1, tCrB1);
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < thr_N; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < channel_num; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tCrB1.size() / thr_N / channel_num; ++i) {
          if constexpr (std::is_same_v<TA, half_t>) {
            tCrB1(cute::tuple(c, _), n, _)[i] *= scales1[n * channel_num + c];
          } else {
            tCrB1(cute::tuple(c, _), n, _)[i] = apply_scale(
                tCrB1(cute::tuple(c, _), n, _)[i],
                scales1[n * channel_num + c]);
          }
        }
      }
    }
    cute::gemm(mma, tCrA, tCrB1, tCrC1);

    copy(copy_b2, tBgB2(_, _, _, k_tile), tBrB2);
    reorder(tBrB2, tCrB2);
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < thr_N; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < channel_num; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tCrB1.size() / thr_N / channel_num; ++i) {
          if constexpr (std::is_same_v<TA, half_t>) {
            tCrB2(cute::tuple(c, _), n, _)[i] *= scales2[n * channel_num + c];
          } else {
            tCrB2(cute::tuple(c, _), n, _)[i] = apply_scale(
                tCrB2(cute::tuple(c, _), n, _)[i],
                scales2[n * channel_num + c]);
          }
        }
      }
    }
    cute::gemm(mma, tCrA, tCrB2, tCrC2);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b1, pBgB1(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b2, pBgB2(_, _, _, k_tile_prefetch));
    }

    barrier_wait(barrier_scope);
  }

  if (Bias != nullptr) {
    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float1 = Bias[n_tile_start + n_sg_start + sg_local_n];
      float b_float2 = Bias[gemm_n + n_tile_start + n_sg_start + sg_local_n];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC1(sn * SG_M + sm) += b_float1;
        tCrC2(sn * SG_M + sm) += b_float2;
      }
    }
  }

  if constexpr (has_clamping) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC1.size(); ++i) {
      tCrC1(i) = sycl::fmin(tCrC1(i), gemm1_clamp_limit);
      tCrC2(i) = sycl::fmax(tCrC2(i), -gemm1_clamp_limit);
      tCrC2(i) = sycl::fmin(tCrC2(i), gemm1_clamp_limit);
    }
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < tCrC1.size(); ++i) {
    if constexpr (activation_type == ActivationType::SILU) {
      tCrC1(i) = silu_kernel(tCrC1(i)) * tCrC2(i);
    } else if constexpr (activation_type == ActivationType::GELU) {
      tCrC1(i) = gelu_kernel(tCrC1(i)) * tCrC2(i);
    } else if constexpr (activation_type == ActivationType::GELU_TANH) {
      tCrC1(i) = gelu_tanh_kernel(tCrC1(i)) * tCrC2(i);
    } else if constexpr (activation_type == ActivationType::SWIGLUOAI) {
      tCrC1(i) = swigluoai_and_mul(tCrC1(i), tCrC2(i), 1.702, 7.0);
    } else if constexpr (activation_type == ActivationType::SWIGLUSTEP) {
      tCrC1(i) = swiglustep_and_mul(tCrC1(i), tCrC2(i), 7.0);
    }
  }

  if constexpr (activation_type == ActivationType::RELU2_NO_MUL) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC1.size(); ++i) {
      tCrC1(i) = relu2_no_mul_kernel(tCrC1(i));
    }
  }

  reorder(tCrC1, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);

  if constexpr (activation_type == ActivationType::RELU2_NO_MUL) {
    auto copy_c2 = get_block_2d_copy_D<GmemTiledCopyC>(mma, C2);
    auto thr_copy_c2 = copy_c2.get_slice(local_id);
    auto tCgC2 = thr_copy_c2.partition_D(gC);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC2.size(); ++i) {
      tCrC2(i) = relu2_no_mul_kernel(tCrC2(i));
    }
    reorder(tCrC2, tCrC_out);
    copy(copy_c2, tCrC_out, tCgC2);
  }
}

}  // namespace FusedMOE
