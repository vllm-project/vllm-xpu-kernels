// SPDX-License-Identifier: Apache-2.0
//
// Cache-hinted block 2D store for the D output of the Xe3 MoE grouped GEMM.
//
// SYCL-TLA's XE_STORE_2D always issues lsc_store_block2d with its built-in
// cache control. Some configs (see `StoreCache` in runner/moe_types.hpp) write
// D with an explicit L1/L2/L3 policy instead, so this header adds a store atom
// that carries the policy, plus get_moe_block_2d_copy_D() which builds the
// same TiledCopy as make_block_2d_copy_D() and only swaps the store message.
#pragma once

#include <cute/tensor.hpp>

namespace gpu::cutlass_kernel::grouped_gemm {

// L1/L2/L3 cache control of the store, spelled like the LSC message suffix
// (WB = write-back, UC = uncached).
enum class XeStoreCachePolicy {
  kWB_UC_UC,
};

#define VLLM_XE_MOE_STORE_2D_ASM(CACHE_SUFFIX)             \
  asm("lsc_store_block2d.ugm" CACHE_SUFFIX                 \
      " (M1, 1) flat[%1+(0,0)] %0:d%2.%3x%4nn" ::"rw"(sv), \
      "rw.u"(payload),                                     \
      "P"(Bits),                                           \
      "P"(Width),                                          \
      "P"(Height))

template <int Bits, int Height, int Width, XeStoreCachePolicy CachePolicy>
struct XeMoEStore2D : cute::XE_Copy_Op_2D_Base<Bits, Height, Width> {
  static_assert(Height <= 8, "Height exceeds hardware limits");
  static constexpr XeStoreCachePolicy StoreCachePolicy = CachePolicy;

  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int* payload, const T* src) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace cute::intel;
    auto& sv = *reinterpret_cast<
        const storage_vector_t<T, Width * Height * Bits / sg_size>*>(src);
    if constexpr (CachePolicy == XeStoreCachePolicy::kWB_UC_UC) {
      VLLM_XE_MOE_STORE_2D_ASM(".wb.uc.uc");
    } else {
      static_assert(
          CachePolicy == XeStoreCachePolicy::kWB_UC_UC,
          "Unsupported Xe store cache policy");
    }
#else
    CUTE_INVALID_CONTROL_PATH(
        "Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }
};

#undef VLLM_XE_MOE_STORE_2D_ASM

template <XeStoreCachePolicy CachePolicy, int Bits, int Height, int Width>
CUTE_HOST_DEVICE constexpr auto
with_store_cache(cute::XE_STORE_2D<Bits, Height, Width> const&) {
  return XeMoEStore2D<Bits, Height, Width, CachePolicy>{};
}

}  // namespace gpu::cutlass_kernel::grouped_gemm

namespace cute {

template <
    int B,
    int H,
    int W,
    gpu::cutlass_kernel::grouped_gemm::XeStoreCachePolicy P>
struct is_xe_block_2d_atom<
    gpu::cutlass_kernel::grouped_gemm::XeMoEStore2D<B, H, W, P>>
    : std::true_type {};

// Same as the XE_STORE_2D traits; only the store op differs.
template <
    class XMode,
    class YMode,
    typename ValType,
    typename TiledStrides,
    int CopyBits,
    int Height,
    int Width,
    gpu::cutlass_kernel::grouped_gemm::XeStoreCachePolicy CachePolicy>
struct Copy_Traits<
    gpu::cutlass_kernel::grouped_gemm::
        XeMoEStore2D<CopyBits, Height, Width, CachePolicy>,
    XMode,
    YMode,
    ValType,
    TiledStrides>
    : Xe2DTraitsBase<
          gpu::cutlass_kernel::grouped_gemm::
              XeMoEStore2D<CopyBits, Height, Width, CachePolicy>,
          XMode,
          YMode,
          ValType,
          TiledStrides> {
  // (src-thr, src-val) -> (x, y)
  using SrcLayout = XeInterleavedLayout<
      Layout<Shape<Int<Width>, Int<Height>>>,
      CopyBits,
      sizeof_bits_v<ValType>>;

  using RefLayout = SrcLayout;
  using DstLayout = decltype(replace<0>(
      RefLayout{}, Layout<Shape<intel::_SGSize>, Stride<_0>>{}));

  using Op = gpu::cutlass_kernel::grouped_gemm::
      XeMoEStore2D<CopyBits, Height, Width, CachePolicy>;
  using Super = Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides>;
  using Traits = typename Super::Traits;

  using Super::Super;

  template <class SEngine, class SLayout, class DEngine, class DLayout>
  CUTE_DEVICE friend constexpr void copy_unpack(
      Traits const& traits,
      Tensor<SEngine, SLayout> const& src,
      Tensor<DEngine, DLayout>& dst) {
    using SType = typename SEngine::value_type;
    using SrcLayout = typename Traits::SrcLayout;
    using DstLayout = typename Traits::DstLayout;
    constexpr auto SBits = sizeof_bits_v<SType>;

    static_assert(
        is_counting_layout_v<DLayout>,
        "Destination tensor must be a coordinate tensor.");
    static_assert(is_rmem_v<SEngine>, "Source tensor must be in registers.");
    static_assert(
        size(SLayout{}) * SBits == size<1>(SrcLayout{}),
        "Source tensor size does not match copy atom size.");
    static_assert(
        size(DLayout{}) * SBits == size<1>(DstLayout{}),
        "Destination tensor size does not match copy atom size.");

    traits.template update_payload<SBits>(dst.data().coord_);
    Op::copy(
        traits.payload,
        recast_ptr<int_byte_t<bits_to_bytes(Super::ValBits)>>(&*src.data()));
  }
};

}  // namespace cute

namespace gpu::cutlass_kernel::grouped_gemm {

// get_block_2d_copy_D<void>() with an optional store cache policy: StoreCache
// is either void (SYCL-TLA default store) or cute::C<XeStoreCachePolicy::...>.
template <class StoreCache, class TiledMMA, class GEngine, class GLayout>
CUTE_HOST_DEVICE auto get_moe_block_2d_copy_D(
    TiledMMA const& mma, cute::Tensor<GEngine, GLayout> const& gmem) {
  if constexpr (std::is_void_v<StoreCache>) {
    return cute::make_block_2d_copy_D(mma, gmem);
  } else {
    using ValType = typename GEngine::value_type;
    using MMAType = typename TiledMMA::ValTypeD;
    auto cD = cute::make_identity_tensor(cute::select<0, 1>(mma.tile_mnk()));
    auto op = cute::block_2d_selector<ValType, MMAType, true>(
        mma.get_slice(0).atom_partition_C(cD).layout(), gmem.stride());
    return cute::make_block_2d_copy_CD<ValType>(
               with_store_cache<StoreCache::value>(op), mma, gmem.stride())
        .with(gmem);
  }
}

}  // namespace gpu::cutlass_kernel::grouped_gemm
