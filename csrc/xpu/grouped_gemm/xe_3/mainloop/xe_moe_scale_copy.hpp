// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cutlass/gemm/collective/xe_common_blockscaled_mxfp.hpp"

namespace cute {

// Each block is GRF-padded: two 1x32-byte blocks need 128 bytes, not 64.
// Keep one-row loads in one block to preserve the packed scale fragment.
template <int Height, int Width>
struct MoEGreedyScaleLoad
    : MXFP_SCALE_LOAD_2D<8, Height, Width, Height == 1 ? Width : 32> {};

template <
    int Height,
    int Width,
    class XMode,
    class YMode,
    class ValType,
    class TiledStrides>
struct Copy_Traits<
    MoEGreedyScaleLoad<Height, Width>,
    XMode,
    YMode,
    ValType,
    TiledStrides>
    : Xe2DTraitsBase<
          MoEGreedyScaleLoad<Height, Width>,
          XMode,
          YMode,
          ValType,
          TiledStrides> {
  using Op = MoEGreedyScaleLoad<Height, Width>;
  using Super = Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides>;
  using Traits = typename Super::Traits;
  static constexpr int BlockWidth = Op::AtomWidth / Op::BlockCount;
  static_assert(
      Op::BlockCount == 1 || (Height * BlockWidth) % 64 == 0,
      "Multi-block scale loads must not introduce inter-block GRF padding");
  using Standard = Copy_Traits<
      XE_LOAD_2D<8, Height, Width, BlockWidth>,
      XMode,
      YMode,
      ValType,
      TiledStrides>;
  using DstLayout = typename Standard::DstLayout;
  using SrcLayout = typename Standard::SrcLayout;
  using RefLayout = typename Standard::RefLayout;
  using AtomShape = Shape<Int<Height>, Int<Width>>;
  using BlockShape = Shape<Int<Height>, Int<BlockWidth>>;
  using Super::Super;

  uint64_t scale_ptr;
  int logical_width;

  template <class Engine, class Layout>
  CUTE_DEVICE Copy_Traits(Tensor<Engine, Layout> const& tensor) {
    static_assert(sizeof_bits_v<ValType> == 8);
    scale_ptr = reinterpret_cast<uint64_t>(raw_pointer_cast(tensor.data()));
    logical_width = shape<XMode::value>(tensor);
    // The padded4 ABI can place an expert's scales between cache-line
    // boundaries. Prefetch the nearby aligned surface, but gather the exact
    // bytes below whenever that surface cannot be used for a real load.
    this->base_ptr = scale_ptr & ~uint64_t(63);
    this->width = (logical_width + 3) & ~3;
    this->height = shape<YMode::value>(tensor);
    this->pitch = stride<YMode::value>(tensor);
    this->tiled_strides = replace<XMode::value>(
        replace<YMode::value>(tensor.stride(), _0{}), _0{});
    this->device_init();
  }

  template <class SE, class SL, class DE, class DL>
  CUTE_DEVICE friend constexpr void copy_unpack(
      Traits const& traits, Tensor<SE, SL> const& src, Tensor<DE, DL>& dst) {
    static_assert(is_counting_layout_v<SL> && is_rmem_v<DE>);
    if (traits.scale_ptr == traits.base_ptr) {
      traits.template update_payload<8>(src.data().coord_);
      Op::copy(
          traits.payload, recast_ptr<uint8_t>(raw_pointer_cast(dst.data())));
      return;
    }
    const auto coord = src.data().coord_;
    const int x = get<XMode::value>(coord);
    const int y = get<YMode::value>(coord);
    const auto* data = reinterpret_cast<uint8_t const*>(traits.scale_ptr) +
                       inner_product(coord, traits.tiled_strides);
    auto bytes = recast<uint8_t>(dst);
    const int lane = int(sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                             .get_local_linear_id()) %
                     16;
    CUTE_UNROLL
    for (int i = 0; i < size(bytes); ++i) {
      const int element = DstLayout{}(make_coord(lane, i * 8)) / 8;
      const int sx = x + element % Width;
      const int sy = y + element / Width;
      bytes(i) = sx >= 0 && sx < traits.logical_width && sy >= 0 &&
                         sy < int(traits.height)
                     ? data[int64_t(sy) * traits.pitch + sx]
                     : 0;
    }
  }
};

}  // namespace cute
