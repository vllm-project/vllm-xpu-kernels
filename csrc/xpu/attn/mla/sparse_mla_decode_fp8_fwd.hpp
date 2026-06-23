#pragma once

#include "utils.hpp"
#include "flash.h"

#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cute/algorithm/subgroup_algorithms.hpp>

#include <cute/util/compat/device.hpp>
#include <cute/util/compat/dims.hpp>
#include <cute/util/compat/launch_policy.hpp>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <limits>

#include "flash_attention_v2/collective/copy_block_slm.hpp"

using namespace cute;

namespace FLASH_NAMESPACE {

#ifndef FLASH_MLA_PREFILL_V_SPLIT
#define FLASH_MLA_PREFILL_V_SPLIT 4
#endif


template<int B_H_>
struct KernelTraits {
    using ElementQ = cutlass::bfloat16_t;
    using ElementKV = cutlass::bfloat16_t;
    using ElementO = cutlass::bfloat16_t;

    using StrideQ = cute::tuple<int, _1, int>;
    using StrideKV = cute::tuple<int, _1, int>;
    using StrideO = cute::tuple<int, _1, int>;

    static constexpr int B_H = B_H_;      // h_q block size
    static constexpr int SUBGROUP_SIZE = intel::sg_size;
    static constexpr int NUM_SUBGROUPS = B_H > 16 ? (B_H > 32 ? 8 : 4) : 4;
    static constexpr int NUM_THREADS = NUM_SUBGROUPS * SUBGROUP_SIZE;
    static constexpr int B_TOPK = 64;    // topk_length block size

    // static constexpr int D_QK = 576;
    static constexpr int D_PE = 64;
    static constexpr int D_V = 512;
    static constexpr int V_SPLIT = FLASH_MLA_PREFILL_V_SPLIT;
    static_assert(V_SPLIT >= 1, "V_SPLIT must be >= 1");
    static_assert(D_V % V_SPLIT == 0, "D_V must be divisible by V_SPLIT");
    static constexpr int D_V_PER_SPLIT = D_V / V_SPLIT;
    static constexpr int HEAD_DIM_TILE_SIZE = 32;

    static constexpr int stages = 64 / B_TOPK;
    static_assert(stages == 1, "only support single stage for now");

    // 576 / 32 = 18
    // Q head packing size = B_H
    using TileShapeQK = Shape<Int<B_H>, Int<B_TOPK>, Int<HEAD_DIM_TILE_SIZE>>;
    using SubgroupLayoutQK = conditional_t<(B_H > 16),
        Layout<Shape<Int<NUM_SUBGROUPS>, _1, _1>>,
        Layout<Shape<_1, Int<NUM_SUBGROUPS>, _1>>
    >;

    using TileShapePV = Shape<Int<B_H>, Int<HEAD_DIM_TILE_SIZE>, Int<B_TOPK>>;
    using SubgroupLayoutPV = conditional_t<(B_H > 16),
        Layout<Shape<Int<NUM_SUBGROUPS>, _1, _1>>,
        Layout<Shape<_1, _1, Int<NUM_SUBGROUPS>>>
    >;

    // D_V / 64 = 8 tiles for v_dim
    using TileShapeOut = Shape<Int<B_H>, Int<D_V_PER_SPLIT>>;

    using SmemTileLayoutK = Layout<Shape<Int<B_TOPK>, Int<HEAD_DIM_TILE_SIZE>>, Stride<Int<HEAD_DIM_TILE_SIZE>, _1>>;
    using SmemTileLayoutV = Layout<Shape<Int<HEAD_DIM_TILE_SIZE>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>;

    constexpr static int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
    // bf16 dpas m8n16k16
    // (8, 128, 64) / ((8, 16, 16) * (1, 16, 1)) = (1, 1, 4) iterations per subgroup
    constexpr static int MAX_M_DPAS = 8;
    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, MAX_M_DPAS), float, bfloat16_t>;
    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;
};

template<int D_QK, bool HAVE_TOPK_LENGTH>
class SparseDecodeGatherDequantKernel {
public:
    using Params = SparseAttnDecodeParams;

    static constexpr int NUM_THREADS = 128;
    static constexpr int SUBGROUP_SIZE = intel::sg_size;
    static constexpr int NUM_SUBGROUPS = NUM_THREADS / SUBGROUP_SIZE;
    static constexpr int B_TOPK = 64;
    static constexpr int FP8_VALUES_PER_PACK = 8;
    static constexpr int BF16_VALUES_PER_PACK = 4;
    using PackedElement = uint64_t;
    static_assert(D_QK == 512, "packed fp8 sparse decode currently supports logical D_QK=512");
    static_assert(D_QK % SUBGROUP_SIZE == 0, "D_QK must be divisible by SUBGROUP_SIZE");
    static_assert(D_QK == SPARSE_MLA_FP8_NOPE_BYTES + SPARSE_MLA_FP8_ROPE_DIM,
                  "logical D_QK must match packed fp8 NoPE + RoPE dimensions");
    static_assert(SPARSE_MLA_FP8_NOPE_BYTES % FP8_VALUES_PER_PACK == 0,
                  "NoPE fp8 bytes must be divisible by the packed fp8 width");
    static_assert(SPARSE_MLA_FP8_ROPE_DIM % BF16_VALUES_PER_PACK == 0,
                  "RoPE bf16 values must be divisible by the packed bf16 width");
    static_assert(SPARSE_MLA_FP8_NOPE_BYTES / 64 == SPARSE_MLA_FP8_SCALE_BYTES_PER_TOKEN - 1,
                  "only the first seven scale bytes are valid for 448 NoPE values");
    static_assert(sizeof(PackedElement) == FP8_VALUES_PER_PACK,
                  "PackedElement must cover one fp8 lane chunk");
    static_assert(sizeof(PackedElement) == sizeof(cutlass::bfloat16_t) * BF16_VALUES_PER_PACK,
                  "PackedElement must cover one bf16 lane chunk");
    static constexpr int NUM_VALS_PER_THREAD = D_QK / SUBGROUP_SIZE;

    CUTLASS_DEVICE
    static float e8m0_to_float(uint8_t scale_byte) {
        return sycl::native::exp2(static_cast<float>(static_cast<int>(scale_byte) - 127));
    }

    CUTLASS_DEVICE
    static uint16_t fp8_e4m3_scaled_to_bf16_bits(uint8_t fp8_byte, uint8_t scale_byte) {
        const float scale = e8m0_to_float(scale_byte);
        const auto fp8_val = cutlass::float_e4m3_t::bitcast(fp8_byte);
        return cutlass::bfloat16_t(static_cast<float>(fp8_val) * scale).storage;
    }

    CUTLASS_DEVICE
    static void store_dequantized_token_scalar(cutlass::bfloat16_t* gathered_row,
                                               const uint8_t* token_data,
                                               const uint8_t* token_scales,
                                               bool valid_token,
                                               int lane_id) {
        CUTE_UNROLL
        for (int n = 0; n < NUM_VALS_PER_THREAD; ++n) {
            int dim_idx = n * SUBGROUP_SIZE + lane_id;
            cutlass::bfloat16_t kv_val = cutlass::bfloat16_t(0.0f);
            if (valid_token && dim_idx < SPARSE_MLA_FP8_NOPE_BYTES) {
                int scale_idx = dim_idx / 64;
                float scale = e8m0_to_float(token_scales[scale_idx]);
                auto fp8_val = *reinterpret_cast<const cutlass::float_e4m3_t*>(token_data + dim_idx);
                kv_val = cutlass::bfloat16_t(static_cast<float>(fp8_val) * scale);
            } else if (valid_token) {
                const auto* rope_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(token_data + SPARSE_MLA_FP8_NOPE_BYTES);
                kv_val = rope_ptr[dim_idx - SPARSE_MLA_FP8_NOPE_BYTES];
            }
            gathered_row[dim_idx] = kv_val;
        }
    }

    CUTLASS_DEVICE
    static void store_dequantized_token_packed(cutlass::bfloat16_t* gathered_row,
                                               const uint8_t* token_data,
                                               const uint8_t* token_scales,
                                               bool valid_token,
                                               int lane_id) {
        for (int d_base = lane_id * FP8_VALUES_PER_PACK;
             d_base < SPARSE_MLA_FP8_NOPE_BYTES;
             d_base += SUBGROUP_SIZE * FP8_VALUES_PER_PACK) {
            PackedElement packed_lo = 0;
            PackedElement packed_hi = 0;
            if (valid_token) {
                const PackedElement packed_fp8 = *reinterpret_cast<const PackedElement*>(token_data + d_base);
                const uint8_t scale_byte = token_scales[d_base / 64];
                CUTE_UNROLL
                for (int vec_offset = 0; vec_offset < FP8_VALUES_PER_PACK; ++vec_offset) {
                    const uint8_t fp8_byte = static_cast<uint8_t>(packed_fp8 >> (8 * vec_offset));
                    const uint16_t bf16_bits = fp8_e4m3_scaled_to_bf16_bits(fp8_byte, scale_byte);
                    if (vec_offset < BF16_VALUES_PER_PACK) {
                        packed_lo |= PackedElement(bf16_bits) << (16 * vec_offset);
                    } else {
                        packed_hi |= PackedElement(bf16_bits) << (16 * (vec_offset - BF16_VALUES_PER_PACK));
                    }
                }
            }
            *reinterpret_cast<PackedElement*>(gathered_row + d_base) = packed_lo;
            *reinterpret_cast<PackedElement*>(gathered_row + d_base + BF16_VALUES_PER_PACK) = packed_hi;
        }

        for (int rope_base = lane_id * BF16_VALUES_PER_PACK;
             rope_base < SPARSE_MLA_FP8_ROPE_DIM;
             rope_base += SUBGROUP_SIZE * BF16_VALUES_PER_PACK) {
            const int dim_idx = SPARSE_MLA_FP8_NOPE_BYTES + rope_base;
            const PackedElement value = valid_token
                ? *reinterpret_cast<const PackedElement*>(token_data + SPARSE_MLA_FP8_NOPE_BYTES
                                                          + rope_base * sizeof(cutlass::bfloat16_t))
                : PackedElement(0);
            *reinterpret_cast<PackedElement*>(gathered_row + dim_idx) = value;
        }
    }

    CUTLASS_DEVICE
    static void store_dequantized_token(cutlass::bfloat16_t* gathered_row,
                                        const uint8_t* active_kv,
                                        int token_idx,
                                        int active_page_block_size,
                                        int active_stride_kv_block,
                                        bool valid_token,
                                        bool can_pack,
                                        int lane_id) {
        const uint8_t* token_data = nullptr;
        const uint8_t* token_scales = nullptr;
        if (valid_token) {
            int block_idx = token_idx / active_page_block_size;
            int rel_idx = token_idx - block_idx * active_page_block_size;
            token_data = active_kv + block_idx * active_stride_kv_block + rel_idx * SPARSE_MLA_FP8_DATA_BYTES_PER_TOKEN;
            token_scales = active_kv + block_idx * active_stride_kv_block
                + active_page_block_size * SPARSE_MLA_FP8_DATA_BYTES_PER_TOKEN
                + rel_idx * SPARSE_MLA_FP8_SCALE_BYTES_PER_TOKEN;
        }

        if (can_pack) {
            store_dequantized_token_packed(gathered_row, token_data, token_scales, valid_token, lane_id);
        } else {
            store_dequantized_token_scalar(gathered_row, token_data, token_scales, valid_token, lane_id);
        }
    }

    CUTLASS_DEVICE
    void operator()(const Params& params, char* smem_buf) const {
        const int thr_id = int(ThreadIdxX());
        const int sg_id = thr_id / SUBGROUP_SIZE;
        const int lane_id = thr_id % SUBGROUP_SIZE;
        const int seq_linear_idx = int(BlockIdxX());
        const int batch_idx = seq_linear_idx / params.s_q;
        const int seq_idx = seq_linear_idx - batch_idx * params.s_q;
        const int topk_block_idx = int(BlockIdxY());
        const int topk_base = topk_block_idx * B_TOPK;

        auto* gathered_k = params.gathered_k + batch_idx * params.stride_gathered_k_b + seq_idx * params.stride_gathered_k_s_q;
        auto* gathered_valid_mask = params.gathered_valid_mask + batch_idx * params.stride_gathered_mask_b + seq_idx * params.stride_gathered_mask_s_q;
        const int* main_indices = params.indices + batch_idx * params.stride_indices_b + seq_idx * params.stride_indices_s_q;
        const int* extra_indices = params.extra_indices == nullptr ? nullptr : params.extra_indices + batch_idx * params.stride_extra_indices_b + seq_idx * params.stride_extra_indices_s_q;

        auto resolve_topk_length = [&](const int* topk_length_ptr, int topk, int stride_b) {
            if constexpr (HAVE_TOPK_LENGTH) {
                if (topk_length_ptr != nullptr) {
                    return *(topk_length_ptr + batch_idx * stride_b);
                }
            }
            return topk;
        };

        const int main_topk_length = resolve_topk_length(params.topk_length, params.topk,
                                                         params.stride_topk_length_b);
        const int extra_topk_length = resolve_topk_length(params.extra_topk_length, params.extra_topk,
                                                          params.stride_extra_topk_length_b);

        for (int local_topk_idx = sg_id; local_topk_idx < B_TOPK; local_topk_idx += NUM_SUBGROUPS) {
            int topk_idx = topk_base + local_topk_idx;
            if (topk_idx >= params.gathered_topk) {
                continue;
            }

            bool is_extra = topk_idx >= params.topk;
            int range_topk_idx = is_extra ? topk_idx - params.topk : topk_idx;
            int active_topk = is_extra ? params.extra_topk : params.topk;
            int active_topk_length = is_extra ? extra_topk_length : main_topk_length;
            const int* active_indices = is_extra ? extra_indices : main_indices;
            const uint8_t* active_kv = is_extra ? params.extra_kv : params.kv;
            int active_num_blocks = is_extra ? params.extra_num_blocks : params.num_blocks;
            int active_page_block_size = is_extra ? params.extra_page_block_size : params.page_block_size;
            int active_stride_kv_block = is_extra ? params.stride_extra_kv_block : params.stride_kv_block;
            const bool can_pack = params.stride_gathered_k_topk % BF16_VALUES_PER_PACK == 0
                && active_stride_kv_block % sizeof(PackedElement) == 0
                && (reinterpret_cast<uintptr_t>(active_kv) & (sizeof(PackedElement) - 1)) == 0
                && (reinterpret_cast<uintptr_t>(gathered_k) & (sizeof(PackedElement) - 1)) == 0;

            bool valid_token = false;
            int token_idx = -1;
            if (active_indices != nullptr && active_kv != nullptr && range_topk_idx < active_topk && range_topk_idx < active_topk_length) {
                token_idx = active_indices[range_topk_idx];
                valid_token = token_idx >= 0 && token_idx < active_num_blocks * active_page_block_size;
            }

            cutlass::bfloat16_t* gathered_row = gathered_k + topk_idx * params.stride_gathered_k_topk;
            store_dequantized_token(gathered_row, active_kv, token_idx, active_page_block_size,
                                    active_stride_kv_block, valid_token, can_pack, lane_id);

            if (lane_id == 0) {
                gathered_valid_mask[topk_idx] = static_cast<int>(valid_token);
            }
        }
    }
};

template<int D_QK, bool HAS_ATTN_SINK, typename Traits>
class DenseDecodeFwdKernel {
public:
    using ElementQ = typename Traits::ElementQ;
    using ElementKV = typename Traits::ElementKV;
    using ElementO = typename Traits::ElementO;
    using TiledMMAQK = typename Traits::TiledMMAQK;
    using TiledMMAPV = typename Traits::TiledMMAPV;
    using TileShapeQK = typename Traits::TileShapeQK;
    using TileShapePV = typename Traits::TileShapePV;
    using TileShapeOut = typename Traits::TileShapeOut;
    using ElementS = typename TiledMMAQK::ValTypeD;
    using ElementA = typename TiledMMAPV::ValTypeD;

    using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));
    using SingleFragA = decltype(TiledMMAPV{}.get_slice(0).partition_sg_fragment_C(make_identity_tensor(select<0,1>(TiledMMAPV{}.tile_mnk()))));
    static_assert(Traits::D_V_PER_SPLIT % get<1>(TileShapePV{}) == 0,
                  "D_V_PER_SPLIT must be divisible by TileShapePV N dimension");
    using FragA = expand_sg_fragment_t<SingleFragA, 1, Traits::D_V_PER_SPLIT / get<1>(TileShapePV{})>;
    using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));

    static auto reduce_sg_v_helper() {
        constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
        constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
        return Int<(v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg) : v_total_sg>{};
    }

    using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
    using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
    using ReduceSGV = decltype(reduce_sg_v_helper());
    using ReduceSGLayout = decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));
    using SGTileShapeO = decltype(shape_div(take<0,2>(SGTileShapeA{}), shape(ReduceSGLayout{})));
    using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
        make_layout(select<1,0>(SGTileShapeO{}), Stride<E<1>, E<0>>{})
    ));
    using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));
    using SGPerWG = decltype(product(take<1,4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));
    using GmemLayoutO = Layout<Shape<int, Int<Traits::D_V_PER_SPLIT>>, Stride<Int<Traits::D_V>, _1>>;
    using TensorO2D = decltype(make_tensor(make_gmem_ptr((ElementO*)nullptr), GmemLayoutO{}));

    static auto default_tiled_copy_O_helper() {
        if constexpr (ReduceK{} == _1{})
            return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
        else
            return make_block_2d_copy_D_subtiled(TiledMMAPV{}, ReduceFragA{}.tv_layout(), ReduceSGLayout{}, TensorO2D{});
    }

    using TiledCopyO = decltype(default_tiled_copy_O_helper());

    struct SharedStorageNonReduceK {};
    using AlignedSGTileA_Q = C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) * intel::sg_size>;

    struct SharedStorageReduceK {
        cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
        cute::array<ElementA, AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data, a_max_data;
    };

    using SharedStorage = conditional_t<(ReduceK{} > 1), SharedStorageReduceK, SharedStorageNonReduceK>;
    static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);
    using Params = SparseAttnDecodeParams;

    CUTLASS_DEVICE
    void operator()(const Params& params, char* smem_buf) const {
        using namespace sycl::ext::oneapi::this_work_item;

        SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

        const ElementQ* q = reinterpret_cast<const ElementQ*>(params.q);
        const int* gathered_valid_mask = params.gathered_valid_mask;
        ElementO* out = params.out;
        float* lse = params.lse;

        const int thr_id = int(ThreadIdxX());
        const int sg_id = thr_id / Traits::SUBGROUP_SIZE;
        const int tid_in_sg = thr_id % Traits::SUBGROUP_SIZE;
        const int v_split_idx = int(BlockIdxY());
        const int cur_v_start_idx = v_split_idx * Traits::D_V_PER_SPLIT;

        int num_head_blocks = ceil_div(params.h_q, Traits::B_H);
        const int wg_id = int(BlockIdxX());
        int q_tile_idx = wg_id / num_head_blocks;
        int batch_idx = q_tile_idx / params.s_q;
        int seq_idx = q_tile_idx - batch_idx * params.s_q;
        int head_bid = wg_id % num_head_blocks;
        int cur_head_start_idx = head_bid * Traits::B_H;

        auto *q_ptr = q + batch_idx * params.stride_q_b + seq_idx * params.stride_q_s_q;
        auto q_layout = make_layout(make_shape(params.h_q, D_QK), make_stride(params.stride_q_h_q, _1{}));
        Tensor Q = make_tensor(make_gmem_ptr(q_ptr), q_layout);

        auto *out_ptr = out + batch_idx * params.stride_o_b + seq_idx * params.stride_o_s_q;
        auto o_layout = make_layout(make_shape(params.h_q, Traits::D_V), make_stride(params.stride_o_h_q, _1{}));
        Tensor O = make_tensor(make_gmem_ptr(out_ptr), o_layout);

        const auto* gathered_k_ptr = params.gathered_k + batch_idx * params.stride_gathered_k_b + seq_idx * params.stride_gathered_k_s_q;
        const auto* gathered_v_ptr = gathered_k_ptr + cur_v_start_idx;
        auto gathered_k_layout = make_layout(make_shape(params.gathered_topk, D_QK), make_stride(params.stride_gathered_k_topk, _1{}));
        auto gathered_v_layout = make_layout(make_shape(Traits::D_V_PER_SPLIT, params.gathered_topk), make_stride(_1{}, params.stride_gathered_k_topk));
        Tensor K = make_tensor(make_gmem_ptr(const_cast<ElementKV*>(gathered_k_ptr)), gathered_k_layout);
        Tensor V = make_tensor(make_gmem_ptr(const_cast<ElementKV*>(gathered_v_ptr)), gathered_v_layout);

        Tensor proxyQ = make_identity_tensor(Q.shape());
        Tensor proxyK = make_identity_tensor(K.shape());
        Tensor proxyP = make_identity_tensor(select<0,1>(TileShapeQK{}));
        Tensor proxyV = make_identity_tensor(V.shape());
        Tensor proxyO = make_identity_tensor(O.shape());

        constexpr int V_TILE_PER_SPLIT = Traits::D_V_PER_SPLIT / get<1>(TileShapePV{});
        auto tile_shape_v = make_shape(Int<Traits::D_V_PER_SPLIT>{}, get<2>(TileShapePV{}));

        Tensor gQ = local_tile(proxyQ, TileShapeQK{}, make_coord(head_bid,_,_), Step<_1, X, _1>{});
        Tensor gK = local_tile(proxyK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
        Tensor gV = local_tile(proxyV, tile_shape_v, make_coord(0, _));
        Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});
        Tensor gO = local_tile(proxyO, TileShapeOut{}, make_coord(head_bid, v_split_idx));

        TiledMMAQK mma_qk{};
        TiledMMAPV mma_pv{};

        auto tiled_copy_Q = make_block_2d_copy_A(mma_qk, Q);
        TiledCopyO tiled_copy_O{O};
        auto thr_copy_q = tiled_copy_Q.get_slice(thr_id);
        auto thr_copy_o = tiled_copy_O.get_slice(thr_id);
        auto tiled_copy_k = make_block_2d_copy_B(mma_qk, K);
        auto tiled_copy_v = make_block_2d_copy_B(mma_pv, V);
        auto thr_copy_k = tiled_copy_k.get_slice(thr_id);
        auto thr_copy_v = tiled_copy_v.get_slice(thr_id);

        auto thr_mma_qk = mma_qk.get_slice(thr_id);
        auto thr_mma_pv = mma_pv.get_slice(thr_id);

        auto tQgQ = thr_copy_q.partition_S(gQ);
        auto tKgK = thr_copy_k.partition_S(gK);
        auto tVgV = thr_copy_v.partition_S(gV_split);

        auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));
        Tensor coord_Q = make_identity_tensor(select<0,2>(TileShapeQK{}));
        auto tQcQ = thr_mma_qk.partition_A(coord_Q);
        auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));

        auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
        auto tOgO = thr_copy_o.partition_D(gO);
        auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_,_,0,0));
        auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_,_,0,0));
        auto tSrS = thr_mma_qk.partition_sg_fragment_C(proxyP);

        auto qk_gemm_one_tile = [&](int block_idx, int tile_idx) {
            copy(tiled_copy_Q, tQgQ(_,_,_,tile_idx), tQrQ);
            reorder(tQrQ, tSrQ);
            copy(tiled_copy_k, tKgK(_,_,_,block_idx,tile_idx), tKrK);
            reorder(tKrK, tSrK);
            cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        };

        const int* gathered_valid_mask_ptr = gathered_valid_mask + batch_idx * params.stride_gathered_mask_b + seq_idx * params.stride_gathered_mask_s_q;
        auto mask_rS = [&](int block_idx) {
            Tensor coord_S = make_identity_tensor(select<0,1>(TileShapeQK{}));
            Tensor tCgC = thr_mma_qk.partition_C(coord_S);
            CUTE_UNROLL
            for (int i = 0; i < tSrS.size(); ++i) {
                int col_idx = get<1>(tCgC(i));
                int topk_idx = block_idx * Traits::B_TOPK + col_idx;
                bool is_valid = topk_idx < params.gathered_topk && gathered_valid_mask_ptr[topk_idx];
                if (!is_valid) {
                    tSrS(i) = cutlass::platform::numeric_limits<ElementS>::lowest();
                }
            }
        };

        FragA tArA;
        FragARow tA_max, tA_sum;
        cute::clear(tArA);
        cute::fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
        cute::fill(tA_sum, 0.0f);

        auto online_softmax_and_rescale_o = [&](int block_idx) {
            auto tA_bmax = reduce<1>(tSrS, sycl::maximum{});
            auto tA_prev_max = tA_max;
            CUTE_UNROLL
            for (int i = 0; i < tA_max.size(); i++) {
                tA_max(i) = sycl::max(tA_max(i), params.sm_scale_div_log2 * tA_bmax(i));
            }

            CUTE_UNROLL
            for (int i = 0; i < tSrS.size(); i++) {
                tSrS(i) = sycl::native::exp2(params.sm_scale_div_log2 * tSrS(i) - broadcast<0>(tA_max, tSrS, i));
            }

            FragARow rescale;
            if (block_idx > 0) {
                CUTE_UNROLL
                for (int i = 0; i < tA_max.size(); i++) {
                    rescale(i) = sycl::native::exp2(tA_prev_max(i) - tA_max(i));
                    tA_sum(i) *= rescale(i);
                }
            }

            auto tA_bsum = reduce<1>(tSrS, sycl::plus<void>{});
            CUTE_UNROLL
            for (int i = 0; i < tA_sum.size(); i++) {
                tA_sum(i) += tA_bsum(i);
            }
            return rescale;
        };

        auto tArP = thr_mma_pv.partition_sg_fragment_A(proxyP);
        auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_,_,0,0));
        auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_,_,0,0));

        auto pv_gemm_one_tile = [&](int block_idx, int local_v_tile_idx) {
            copy(tiled_copy_v, tVgV(_,_,_,local_v_tile_idx,block_idx), tVrV);
            reorder(tVrV, tArV);
            cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,local_v_tile_idx));
        };

        auto reduce_L = [&]() {
            if constexpr (ReduceK{} == _1{}) {
                return std::make_tuple(tArA, tA_max, tA_sum, true);
            } else {
                auto thr_vak = group<1,3>(mma_pv.get_thr_layout_vmnk()).get_flat_coord(assert_uniform(thr_id));
                auto a_tile = get<1>(thr_vak);
                auto k_blk = get<2>(thr_vak);

                auto shape_A = append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{}/ReduceK{});
                auto shape_A_row = make_shape(get<0>(SGTileShapeO{}), shape(ReduceSGLayout{}), ReduceK{}, SGPerWG{}/ReduceK{});
                auto sA_layout = group<2,4>(flat_divide(make_ordered_layout(shape_A, Step<_1,_0,_2,_3>{}), SGTileShapeO{}));
                auto sA_row_stride = make_stride(_1{}, make_stride(get<0>(shape_A_row), _0{}),
                                                 AlignedSGTileA_Q{}, AlignedSGTileA_Q{} * ReduceK{});
                auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);
                auto basis2 = make_basis_like(SGTileShapeO{});
                auto sA_coords = make_layout(append(SGTileShapeO{}, shape(ReduceSGLayout{})),
                                             append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

                auto sA = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_data), sA_layout);
                auto sA_max = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_max_data), sA_row_layout);
                auto sA_sum = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_sum_data), sA_row_layout);

                copy_block_r2s(tA_max, sA_max(_,_,k_blk,a_tile));
                barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
                copy_block_r2s(tA_sum, sA_sum(_,_,k_blk,a_tile));
                copy_block_r2s(tArA, sA(_,_,_,k_blk,a_tile), sA_coords);

                bool active = (k_blk < size(ReduceSGLayout{})) || (ReduceK{} == size(ReduceSGLayout{}));
                barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
                barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

                ReduceFragA rA;
                ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

                if (active) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int kr = 0; kr < ReduceK{}; kr++) {
                        copy_block_s2r(sA_max(_,k_blk,kr,a_tile), rA_kmax[kr]);
                    }

                    rA_max = rA_kmax[0];
                    for (int kr = 1; kr < ReduceK{}; kr++) {
                        cute::transform(rA_max, rA_kmax[kr], rA_max, cute::max_fn{});
                    }

                    for (int kr = 0; kr < ReduceK{}; kr++) {
                        cute::transform(rA_max, rA_kmax[kr], rA_kmax[kr], [](auto gmax, auto kmax) {
                            return sycl::native::exp2(kmax - gmax);
                        });
                    }
                }

                barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

                if (active) {
                    clear(rA_sum);
                    CUTLASS_PRAGMA_UNROLL
                    for (int kr = 0; kr < ReduceK{}; kr++) {
                        ReduceFragARow rA_sum_read;
                        copy_block_s2r(sA_sum(_,k_blk,kr,a_tile), rA_sum_read);
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < rA_sum_read.size(); i++) {
                            rA_sum(i) += rA_sum_read(i) * rA_kmax[kr](i);
                        }
                    }

                    clear(rA);
                    CUTLASS_PRAGMA_UNROLL
                    for (int kr = 0; kr < ReduceK{}; kr++) {
                        ReduceFragA rA_read;
                        copy_block_s2r(sA(_,_,k_blk,kr,a_tile), sA_coords(_,_,0), rA_read);
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < rA_read.size(); i++) {
                            rA(i) += rA_read(i) * broadcast<0>(rA_kmax[kr], rA, i);
                        }
                    }
                }
                return std::make_tuple(rA, rA_max, rA_sum, active);
            }
        };

        auto final_rescale_and_store_o = [&]() {
            auto [rA, rA_max, rA_sum, active] = reduce_L();
            if (!active) return;

            constexpr int valid_tid_per_sg = Traits::B_H / get<0>(typename Traits::SubgroupLayoutQK{}.shape());
            static_assert(valid_tid_per_sg <= Traits::SUBGROUP_SIZE, "valid_tid_per_sg must be less than or equal to SUBGROUP_SIZE");
            if (v_split_idx == 0 && tid_in_sg < valid_tid_per_sg && sg_id * valid_tid_per_sg < Traits::B_H) {
                int local_head_idx = sg_id * valid_tid_per_sg + tid_in_sg;
                int head_idx = cur_head_start_idx + local_head_idx;
                if (head_idx < params.h_q) {
                    int lse_idx = batch_idx * params.stride_lse_b + seq_idx * params.stride_lse_s_q + head_idx;

                    float row_max = -INFINITY;
                    if (rA_max(0) != params.sm_scale_div_log2 * cutlass::platform::numeric_limits<ElementA>::lowest()) {
                        row_max = rA_max(0) * LOG_E_2;
                    }

                    float row_lse = INFINITY;
                    if (rA_sum(0) > 0.f) {
                        row_lse = row_max + sycl::native::log2(rA_sum(0)) * LOG_E_2;
                    }
                    lse[lse_idx] = row_lse;
                }
            }

            if constexpr (HAS_ATTN_SINK) {
                if constexpr (ReduceK{} > 1) {
                    auto thr_vak = group<1,3>(mma_pv.get_thr_layout_vmnk()).get_flat_coord(assert_uniform(thr_id));
                    int reduce_head_offset = (get<2>(thr_vak) / ReduceSGV{}) * get<0>(SGTileShapeO{});

                    CUTE_UNROLL
                    for (int i = 0; i < rA.size(); ++i) {
                        auto global_exp_sum = broadcast<0>(rA_sum, rA, i);
                        ElementA final_rescale = global_exp_sum != 0 ? ElementA(1) / global_exp_sum : ElementA(0);
                        int local_head_idx = get<0>(rA.tv_layout()(0, i));
                        int head_idx = cur_head_start_idx + reduce_head_offset + local_head_idx;
                        if (head_idx < params.h_q) {
                            auto global_max = broadcast<0>(rA_max, rA, i);
                            float attn_sink_val = params.attn_sink[head_idx];
                            ElementA sink_exp_sum = sycl::native::exp2(
                                static_cast<ElementA>(attn_sink_val * FLASH_NAMESPACE::LOG_2_E) - global_max);
                            ElementA global_exp_sum_with_sink = global_exp_sum + sink_exp_sum;
                            final_rescale = global_exp_sum_with_sink != 0 ? ElementA(1) / global_exp_sum_with_sink : ElementA(0);
                        }
                        rA(i) *= final_rescale;
                    }
                } else {
                    if (tid_in_sg < valid_tid_per_sg && sg_id * valid_tid_per_sg < Traits::B_H) {
                        int local_head_idx = sg_id * valid_tid_per_sg + tid_in_sg;
                        int head_idx = cur_head_start_idx + local_head_idx;
                        if (head_idx < params.h_q) {
                            float attn_sink_val = params.attn_sink[head_idx];
                            CUTE_UNROLL
                            for (int i = 0; i < rA_sum.size(); ++i) {
                                rA_sum(i) += sycl::native::exp2(
                                    static_cast<ElementA>(attn_sink_val * FLASH_NAMESPACE::LOG_2_E) - rA_max(i));
                            }
                        }
                    }

                    CUTE_UNROLL
                    for (int i = 0; i < rA_sum.size(); ++i) {
                        if (rA_sum(i) != 0) {
                            rA_sum(i) = ElementA(1) / rA_sum(i);
                        } else {
                            rA_sum(i) = 0;
                        }
                    }

                    CUTE_UNROLL
                    for (int i = 0; i < rA.size(); ++i) {
                        rA(i) *= broadcast<0>(rA_sum, rA, i);
                    }
                }
            } else {
                CUTE_UNROLL
                for (int i = 0; i < rA_sum.size(); ++i) {
                    if (rA_sum(i) != 0) {
                        rA_sum(i) = ElementA(1) / rA_sum(i);
                    } else {
                        rA_sum(i) = 0;
                    }
                }

                CUTE_UNROLL
                for (int i = 0; i < rA.size(); ++i) {
                    rA(i) *= broadcast<0>(rA_sum, rA, i);
                }
            }

            reorder(rA, tOrO);
            copy(tiled_copy_O, tOrO, tOgO);
        };

        const int num_topk_blocks = ceil_div(params.gathered_topk, Traits::B_TOPK);
        CUTE_NO_UNROLL
        for (int topk_idx = 0; topk_idx < num_topk_blocks; ++topk_idx) {
            clear(tSrS);
            CUTE_UNROLL
            for (int i = 0; i < (D_QK / get<2>(TileShapeQK{})); ++i) {
                qk_gemm_one_tile(topk_idx, i);
            }

            mask_rS(topk_idx);
            auto rescale = online_softmax_and_rescale_o(topk_idx);
            reorder(tSrS, tArP);

            if (topk_idx > 0) {
                CUTE_UNROLL
                for (int i = 0; i < tArA.size(); i++) {
                    tArA(i) *= broadcast<0>(rescale, tArA, i);
                }
            }

            CUTE_UNROLL
            for (int i = 0; i < V_TILE_PER_SPLIT; ++i) {
                pv_gemm_one_tile(topk_idx, i);
            }
        }

        final_rescale_and_store_o();
    }
};

template<int D_QK, bool HAVE_TOPK_LENGTH, bool HAS_ATTN_SINK, int B_H>
void launch_sparse_mla_decode_fp8_fwd_kernel_policy(const XPUSparseDecodeAttnFwdParams& params) {
    using Traits = KernelTraits<B_H>;

    TORCH_CHECK(params.h_kv == 1, "h_kv must be 1");
    TORCH_CHECK(params.h_q > 0, "h_q must be > 0");
    TORCH_CHECK(params.d_qk == D_QK, "Invalid d_qk for this kernel instantiation");
    TORCH_CHECK(params.d_v == Traits::D_V, "d_v must match KernelTraits::D_V");
    TORCH_CHECK(params.gathered_topk == params.topk + params.extra_topk,
                "gathered_topk must equal topk + extra_topk");

    using GatherDequantKernel = SparseDecodeGatherDequantKernel<D_QK, HAVE_TOPK_LENGTH>;
    using DenseKernel = DenseDecodeFwdKernel<D_QK, HAS_ATTN_SINK, Traits>;

    if (params.gathered_topk > 0) {
        dim3 block(GatherDequantKernel::NUM_THREADS, 1, 1);
        dim3 grid(params.b * params.s_q, ceil_div(params.gathered_topk, GatherDequantKernel::B_TOPK), 1);

        const auto sycl_block = compat::dim3(block.x, block.y, block.z);
        const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

        compat::experimental::launch_properties launch_props {
            sycl::ext::oneapi::experimental::work_group_scratch_size(0),
        };
        compat::experimental::kernel_properties kernel_props{
            sycl::ext::oneapi::experimental::sub_group_size<GatherDequantKernel::SUBGROUP_SIZE>,
            // Gather kernel uses the XE2 GRF mode for more parallel subgroups.
            sycl::ext::intel::experimental::grf_size<128>
        };
        compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
        auto event = compat::experimental::launch<cutlass::device_kernel<GatherDequantKernel>, GatherDequantKernel>(policy, params.queue, static_cast<SparseAttnDecodeParams>(params));
    }

    {
        dim3 block(Traits::NUM_THREADS, 1, 1);
        dim3 grid(ceil_div(params.h_q, Traits::B_H) * params.s_q * params.b, Traits::V_SPLIT, 1);

        const auto sycl_block = compat::dim3(block.x, block.y, block.z);
        const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);
        const int smem_size = DenseKernel::SharedStorageSize;

        compat::experimental::launch_properties launch_props {
          sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
        };
        compat::experimental::kernel_properties kernel_props{
          sycl::ext::oneapi::experimental::sub_group_size<Traits::SUBGROUP_SIZE>,
            sycl::ext::intel::experimental::grf_size<256>
        };
        compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
        auto event = compat::experimental::launch<cutlass::device_kernel<DenseKernel>, DenseKernel>(policy, params.queue, static_cast<SparseAttnDecodeParams>(params));
    }
}

inline int sparse_mla_decode_select_b_h(const XPUSparseDecodeAttnFwdParams& params) {
    TORCH_CHECK(params.h_q > 0, "h_q must be > 0");
    // TODO: currently use simple rule to decide B_H, in fucture need to consider
    // smart heruistics to balance occupancy and per-WG workload
    if (params.h_q <= 8) return 8;
    if (params.h_q <= 16) return 16;
    if (params.h_q <= 32) return 32;
    return 64;
}

template<int D_QK, bool HAS_TOPK_LENGTH, bool HAS_ATTN_SINK>
void launch_sparse_mla_decode_fp8_fwd_kernel(const XPUSparseDecodeAttnFwdParams& params) {
    switch (sparse_mla_decode_select_b_h(params)) {
        case 8:
            launch_sparse_mla_decode_fp8_fwd_kernel_policy<D_QK, HAS_TOPK_LENGTH, HAS_ATTN_SINK, 8>(params);
            break;
        case 16:
            launch_sparse_mla_decode_fp8_fwd_kernel_policy<D_QK, HAS_TOPK_LENGTH, HAS_ATTN_SINK, 16>(params);
            break;
        case 32:
            launch_sparse_mla_decode_fp8_fwd_kernel_policy<D_QK, HAS_TOPK_LENGTH, HAS_ATTN_SINK, 32>(params);
            break;
        default:
            launch_sparse_mla_decode_fp8_fwd_kernel_policy<D_QK, HAS_TOPK_LENGTH, HAS_ATTN_SINK, 64>(params);
            break;
    }
}

} // namespace FLASH_NAMESPACE
