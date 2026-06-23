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

#include <cstdint>

#include "flash_attention_v2/collective/copy_block_slm.hpp"

using namespace cute;

namespace FLASH_NAMESPACE {

template<int B_H_>
struct KernelTraits {
    using ElementQ = cutlass::bfloat16_t;
    using ElementKV = cutlass::bfloat16_t;
    using ElementO = cutlass::bfloat16_t;

    using StrideQ = cute::tuple<int, _1, int>;
    using StrideKV = cute::tuple<int, _1, int>;
    using StrideO = cute::tuple<int, _1, int>;

    static constexpr int B_H = B_H_;      // h_q block size
    // TODO: need tuned
    // when B_H <= 16, try larger B_TOPK
    static constexpr int B_TOPK = B_H > 16 ? 64 : 256;    // topk_length block size

    static constexpr int D_PE = 64;
    static constexpr int D_V = 512;
    // TODO: need tuned
    static constexpr int V_SPLIT = 4;
    static_assert(V_SPLIT >= 1, "V_SPLIT must be >= 1");
    static_assert(D_V % V_SPLIT == 0, "D_V must be divisible by V_SPLIT");
    static constexpr int D_V_PER_SPLIT = D_V / V_SPLIT;
    // TODO: need tuned
    static constexpr int HEAD_DIM_TILE_SIZE = 64;
    static constexpr int stages = 1;

    static constexpr int SUBGROUP_SIZE = intel::sg_size;
    static constexpr int NUM_SUBGROUPS = B_H > 16 ? (B_H > 32 ? 8 : 4) : 4;
    static constexpr int NUM_THREADS = NUM_SUBGROUPS * SUBGROUP_SIZE;

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

    using TileShapeOut = Shape<Int<B_H>, Int<D_V_PER_SPLIT>>;

    constexpr static int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
    constexpr static int MAX_M_DPAS = 8;
    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, MAX_M_DPAS), float, bfloat16_t>;
    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;
};

template<int D_QK, bool HAVE_TOPK_LENGTH, bool HAS_ATTN_SINK, typename Traits>
class DensePrefillFwdKernel {
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

    // Split k-reduced tiles between participating subgroups.
    using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

    // (8,1,4)
    using SingleFragA = decltype(TiledMMAPV{}.get_slice(0).partition_sg_fragment_C(make_identity_tensor(select<0,1>(TiledMMAPV{}.tile_mnk()))));
    // （8,1,4,8)
    static_assert(Traits::D_V_PER_SPLIT % get<1>(TileShapePV{}) == 0,
                  "D_V_PER_SPLIT must be divisible by TileShapePV N dimension");
    using FragA = expand_sg_fragment_t<SingleFragA, 1, Traits::D_V_PER_SPLIT / get<1>(TileShapePV{})>;     // (atom val,q',v',VV)
    // storing updated global row-max and row-sum
    // (1) - thr_id 0 ~ B_H-1 holds max of each row
    using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
    // static_assert(is_same_v<decltype(FragA{}.shape()), float>, "dtype mismatch");

    static auto reduce_sg_v_helper() {
        constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
        constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
        return Int<(v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg) : v_total_sg>{};
    }

    // Split-local accumulator tile over V dimension, e.g. (8, 128) when V_SPLIT=4.
    using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
    // (8)
    using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
    // (2)
    using ReduceSGV = decltype(reduce_sg_v_helper());
    // (8, 2)
    using ReduceSGLayout = decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));
    // (1, 256)
    using SGTileShapeO = decltype(shape_div(take<0,2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

    // static_assert(is_same_v<decltype(SGTileShapeO{}), float>, "dtype mismatch");

    using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
        make_layout(select<1,0>(SGTileShapeO{}),
                    Stride<E<1>, E<0>>{})
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

    // SLM padding to avoid bank conflicts on Intel XPU.
    // Power-of-2 D_QK strides (e.g., 512) cause all SIMD16 lanes to hit the same SLM bank,
    // resulting in serialized access. Padding by 64 elements makes D_QK=512 use a 576-element
    // row stride (1152 bytes), which is empirically conflict-free (same as D_QK=576 kernel).
    // static constexpr int SLM_STRIDE = ((D_QK & (D_QK - 1)) == 0) ? D_QK + 64 : D_QK;

    struct SharedStorageNonReduceK {};

    // Shared memory storage
    // Note sum/max tiles are padded to 16 elements, due to limitations in CuTe block load infrastructure.
    using AlignedSGTileA_Q = C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) * intel::sg_size>;

    struct SharedStorageReduceK {
        cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
        cute::array<ElementA,   AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data, a_max_data;
    };

    using SharedStorage = conditional_t<(ReduceK{} > 1), SharedStorageReduceK, SharedStorageNonReduceK>;
    static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                        : sizeof(SharedStorage);
    using Params = SparseAttnFwdParams;

    CUTLASS_DEVICE
    void operator()(const Params& params, char* smem_buf) const {
        using namespace sycl::ext::oneapi::this_work_item;

        SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

        const ElementQ* q = params.q;
        const ElementKV* gathered_k = params.gathered_k;
        const int* topk_length = params.topk_length;
        ElementO* out = params.out;
        float* max_logits = params.max_logits;
        float* lse = params.lse;

        const int thr_id = int(ThreadIdxX());
        const int sg_id = thr_id / Traits::SUBGROUP_SIZE;
        const int tid_in_sg = thr_id % Traits::SUBGROUP_SIZE;

        const int v_split_idx = int(BlockIdxY());
        const int cur_v_start_idx = v_split_idx * Traits::D_V_PER_SPLIT;

        int num_head_blocks = ceil_div(params.h_q, Traits::B_H);
        const int wg_id = int(BlockIdxX());
        int seq_idx = wg_id / num_head_blocks;
        int head_bid = wg_id % num_head_blocks;

        // using GmemLayoutQ = Layout<Shape<int, Int<D_QK>>, Stride<int, _1>>;
        // using GmemLayoutO = Layout<Shape<int, Int<Traits::D_V_PER_SPLIT>>, Stride<int, _1>>;

        auto *q_ptr = q + seq_idx * params.stride_q_s_q;
        auto q_layout = make_layout(make_shape(params.h_q, D_QK), make_stride(D_QK, _1{}));
        Tensor Q = make_tensor(make_gmem_ptr(q_ptr), q_layout);

        auto * out_ptr = out + seq_idx * params.h_q * params.d_v;
        auto o_layout = make_layout(make_shape(params.h_q, Traits::D_V), make_stride(Traits::D_V, _1{}));
        Tensor O = make_tensor(make_gmem_ptr(out_ptr), o_layout);

        Tensor proxyQ = make_identity_tensor(Q.shape());
        // S = Q@K^T
        // P = Softmax(S)
        // (64,64)
        Tensor proxyP = make_identity_tensor(select<0,1>(TileShapeQK{})); // (h,k)
        // O = P@V
        Tensor proxyO = make_identity_tensor(O.shape());

        auto* gathered_k_ptr = gathered_k + seq_idx * params.stride_gathered_k_s_q;
        auto* gathered_v_ptr = gathered_k + seq_idx * params.stride_gathered_k_s_q + cur_v_start_idx;
        auto gathered_k_layout = make_layout(
            make_shape(params.topk, D_QK),
            make_stride(params.stride_gathered_k_topk, _1{}));
        auto gathered_v_layout = make_layout(
            make_shape(Traits::D_V_PER_SPLIT, params.topk),
            make_stride(_1{}, params.stride_gathered_k_topk));
        Tensor K = make_tensor(make_gmem_ptr(const_cast<ElementKV*>(gathered_k_ptr)), gathered_k_layout);
        Tensor V = make_tensor(make_gmem_ptr(const_cast<ElementKV*>(gathered_v_ptr)), gathered_v_layout);
        Tensor proxyK = make_identity_tensor(K.shape());
        Tensor proxyV = make_identity_tensor(V.shape());
        constexpr int V_TILE_PER_SPLIT = Traits::D_V_PER_SPLIT / get<1>(TileShapePV{});
        auto tile_shape_v = make_shape(Int<Traits::D_V_PER_SPLIT>{}, get<2>(TileShapePV{}));

        // (8,64,9)
        Tensor gQ = local_tile(proxyQ, TileShapeQK{}, make_coord(head_bid,_,_), Step<_1, X, _1>{}); // (h,d,D)
        Tensor gK = local_tile(proxyK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (k,d,K,D)
        Tensor gV = local_tile(proxyV, tile_shape_v, make_coord(0, _)); // (v,k,K)
        Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{}); // (v,k,VV,K)
        Tensor gO = local_tile(proxyO, TileShapeOut{}, make_coord(head_bid,v_split_idx)); // (h,d)

        TiledMMAQK mma_qk{};
        TiledMMAPV mma_pv{};

        // (V,M,N,K)
        //  V -> num threads per MMA atom
        //  M -> repetition of M dimension in MMA operation
        //  N -> repetition of N dimension in MMA operation
        //  K -> repetition of reduction dimension in MMA operation
        // V*M*N*K == total threads participating in the MMA operation
        //
        // (16,1,1,16)
        // static_assert(is_same_v<decltype(typename Traits::TiledMMAQK::ThrLayoutVMNK{}), float>, "dtype mismatch");

        // create tiled copy for loading from gmem or storing to gmem
        auto tiled_copy_Q = make_block_2d_copy_A(mma_qk, Q);
        // auto tiled_copy_O = make_block_2d_copy_D(mma_pv, O);
        // using TensorO2D = decltype(O(append<rank_v<decltype(O)>>(make_coord(_,_),0)));
        // using TensorO2D = decltype(O);
        // static_assert(is_same_v<decltype(TensorO2D{}.shape()), float>, "dtype mismatch");
        // auto tiled_copy_O = (ReduceK{} > _1{}) ? make_block_2d_copy_D_subtiled(mma_pv, ReduceFragA{}.tv_layout(), ReduceSGLayout{}, TensorO2D{})
        //                                     : make_block_2d_copy_D(mma_pv, O);
        TiledCopyO tiled_copy_O{O};
        // get copy slice for each work item
        auto thr_copy_q = tiled_copy_Q.get_slice(thr_id);
        auto thr_copy_o = tiled_copy_O.get_slice(thr_id);
        auto tiled_copy_k = make_block_2d_copy_B(mma_qk, K);
        auto tiled_copy_v = make_block_2d_copy_B(mma_pv, V);
        auto thr_copy_k = tiled_copy_k.get_slice(thr_id);
        auto thr_copy_v = tiled_copy_v.get_slice(thr_id);

        auto thr_mma_qk = mma_qk.get_slice(thr_id);
        auto thr_mma_pv = mma_pv.get_slice(thr_id);

        // (((8,2),2),1,1,9)
        auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
        auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
        auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

        /* Create register fragments for MMA and copies */
        auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));
        auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));

        auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
        auto tOgO = thr_copy_o.partition_D(gO);

        auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_,_,0,0));
        auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_,_,0,0));

        auto tSrS = thr_mma_qk.partition_sg_fragment_C(proxyP);

        // TODO: use inline asm for loading gmem?
        const int real_topk_length = HAVE_TOPK_LENGTH ? *(topk_length + seq_idx) : params.topk;

        const int* gathered_valid_mask = params.gathered_valid_mask + seq_idx * params.stride_gathered_mask_s_q;

        // copy one tile of Q and K to call mma
        auto qk_gemm_one_tile = [&](int block_idx, int tile_idx) {
            copy(tiled_copy_Q, tQgQ(_,_,_,tile_idx), tQrQ);
            copy(tiled_copy_k, tKgK(_,_,_,block_idx,tile_idx), tKrK);
            reorder(tQrQ, tSrQ);
            reorder(tKrK, tSrK);
            cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        };

        auto mask_rS = [&](int block_idx) {
            // for those invalid tokens, we need mask value to -INF for later softmax calculation
            Tensor coord_S = make_identity_tensor(select<0,1>(TileShapeQK{}));
            Tensor tCgC = thr_mma_qk.partition_C(coord_S);

            CUTE_UNROLL
            for (int i = 0; i < tSrS.size(); ++i) {
                // [B_H, B_TOPK]
                int row_idx = get<0>(tCgC(i));
                int col_idx = get<1>(tCgC(i));
#if 0
                if (row_idx == 0 && col_idx < 10 && seq_idx == 1 && head_bid == 0) {
                    cute::print("row_idx: %d, col_idx: %d, tSrS(0): %f\n", row_idx, col_idx, tSrS(i) * params.sm_scale);
                }
#endif
                // if the column index corresponds to an invalid token, mask it out
                int topk_idx = block_idx * Traits::B_TOPK + col_idx;
                bool is_valid = topk_idx < real_topk_length && topk_idx < params.topk && gathered_valid_mask[topk_idx];
                if (!is_valid) {
                    // note that cannot use -INFINITY due to numerical issue when applying exp2(prev_max-cur_max) later, use the lowest value instead
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
            // rescale accumulated row-sum
            if (block_idx > 0) {
                CUTE_UNROLL
                for (int i = 0; i < tA_max.size(); i++) {
                    rescale(i) = sycl::native::exp2(tA_prev_max(i) - tA_max(i));
                    tA_sum(i) *= rescale(i);
                }
            }

            // update global row-sum
            auto tA_bsum = reduce<1>(tSrS, sycl::plus<void>{});
            for (int i = 0; i < tA_sum.size(); i++) {
                tA_sum(i) += tA_bsum(i);
            }

#if 0
            if (ThreadIdxX() == 0 && seq_idx == 1 && head_bid == 0) {
                cute::print("block_idx: %d, tA_sum(0): %f, tA_max: %f\n", block_idx, tA_sum(0), tA_max(0) * params.sm_scale);
            }
#endif
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
            // refer to https://github.com/intel/sycl-tla/blob/main/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp#L188
            if constexpr (ReduceK{} == _1{}) {
                return std::make_tuple(tArA, tA_max, tA_sum, true);
            } else {
                /* Identify A tile ID and k block for this subgroup. */
                auto thr_vak = group<1,3>(mma_pv.get_thr_layout_vmnk()).get_flat_coord(assert_uniform(thr_id));
                auto a_tile = get<1>(thr_vak);
                auto k_blk = get<2>(thr_vak);

                /* Set up SLM tensors and partition A tiles among participating subgroups */
                auto shape_A     = append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{}/ReduceK{});
                auto shape_A_row = make_shape(get<0>(SGTileShapeO{}), shape(ReduceSGLayout{}), ReduceK{}, SGPerWG{}/ReduceK{});

                /* Physical layouts, with subtile modes broken out */
                auto sA_layout = group<2,4>(flat_divide(make_ordered_layout(shape_A, Step<_1,_0,_2,_3>{}), SGTileShapeO{}));
                auto sA_row_stride = make_stride(_1{}, make_stride(get<0>(shape_A_row), _0{}),
                                                AlignedSGTileA_Q{}, AlignedSGTileA_Q{} * ReduceK{});
                auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

                /* Coordinate layouts, with subtile modes broken out */
                auto basis2 = make_basis_like(SGTileShapeO{});
                auto sA_coords = make_layout(append(SGTileShapeO{}, shape(ReduceSGLayout{})),
                                            append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

                auto sA     = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_data),     sA_layout);      // (q,v,rblk_dst,rblk_src,a_tile)
                auto sA_max = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_max_data), sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)
                auto sA_sum = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_sum_data), sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)

                /* Write my contributions to SLM. */
                copy_block_r2s(tA_max, sA_max(_,_,k_blk,a_tile));
                barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
                copy_block_r2s(tA_sum, sA_sum(_,_,k_blk,a_tile));
                copy_block_r2s(tArA, sA(_,_,_,k_blk,a_tile), sA_coords);

                bool active = (k_blk      < size(ReduceSGLayout{}))
                            || (ReduceK{} == size(ReduceSGLayout{}));    // help compiler out

                /* Wait for maxima to be available, signal other data available */
                barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
                barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

                ReduceFragA rA;
                ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

                if (active) {
                    /* Read A_max back from SLM and reduce. */
                    CUTLASS_PRAGMA_UNROLL
                    for (int kr = 0; kr < ReduceK{}; kr++) {
                        copy_block_s2r(sA_max(_,k_blk,kr,a_tile), rA_kmax[kr]);
                    }

                    rA_max = rA_kmax[0];
                    for (int kr = 1; kr < ReduceK{}; kr++)
                        cute::transform(rA_max, rA_kmax[kr], rA_max, cute::max_fn{});

                    /* Calculate scale factors for aligning per-block maxima. */
                    for (int kr = 0; kr < ReduceK{}; kr++) {
                        cute::transform(rA_max, rA_kmax[kr], rA_kmax[kr], [](auto gmax, auto kmax) {
                            return sycl::native::exp2(kmax - gmax);
                        });
                    }
                }

                /* Wait for A/A_sum data to be available */
                barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

                if (active) {
                    /* Read A/A_sum back from SLM, align scaling to new maxima, and reduce. */
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
            // reduce global row-sum and row-max
            auto [rA, rA_max, rA_sum, active] = reduce_L();

            if (!active) return;

            // start idx of current head block
            int cur_head_start_idx = head_bid * Traits::B_H;
            // store max logits and lse
            constexpr int valid_tid_per_sg = Traits::B_H / get<0>(typename Traits::SubgroupLayoutQK{}.shape());
            static_assert(valid_tid_per_sg <= Traits::SUBGROUP_SIZE, "valid_tid_per_sg must be less than or equal to SUBGROUP_SIZE");
            if (v_split_idx == 0 && tid_in_sg < valid_tid_per_sg && sg_id * valid_tid_per_sg < Traits::B_H) {
                int local_head_idx = sg_id * valid_tid_per_sg + tid_in_sg;
                int head_idx = cur_head_start_idx + local_head_idx;

                if (head_idx < params.h_q) {
                    int out_idx = seq_idx * params.h_q + head_idx;

                    float row_max = -INFINITY;
                    // this condition is for the case when no valid tokens are selected
                    if (rA_max(0) != params.sm_scale_div_log2 * cutlass::platform::numeric_limits<ElementA>::lowest()) {
                        row_max = rA_max(0) * LOG_E_2;
                    }

                    float row_lse = INFINITY;
                    // this condition is for the case when no valid tokens are selected
                    if (rA_sum(0) > 0.f) {
                        row_lse = row_max + sycl::native::log2(rA_sum(0)) * LOG_E_2;
                    }

                    max_logits[out_idx] = row_max;
                    lse[out_idx] = row_lse;
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
                            // to avoid divide-by-zero
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
                        // to avoid divide-by-zero
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

        const int num_topk_blocks = HAVE_TOPK_LENGTH ? ceil_div(real_topk_length, Traits::B_TOPK) : ceil_div(params.topk, Traits::B_TOPK);

        // mainloop
        CUTE_NO_UNROLL
        for (int topk_idx = 0; topk_idx < num_topk_blocks; ++topk_idx) {
            clear(tSrS);
            CUTE_UNROLL
            for (int i = 0; i < (D_QK / get<2>(TileShapeQK{})); ++i) {
                qk_gemm_one_tile(topk_idx, i);
            }

            mask_rS(topk_idx);
            auto rescale = online_softmax_and_rescale_o(topk_idx);

            // fp32 -> bf16
            reorder(tSrS, tArP);

            // rescale acc
            if (topk_idx > 0) {
                CUTE_UNROLL
                for (int i = 0; i < tArA.size(); i++) {
                    tArA(i) *= broadcast<0>(rescale, tArA, i);
                }
            }
            // Process only split-local V tiles for this workgroup.
            CUTE_UNROLL
            for (int i = 0; i < V_TILE_PER_SPLIT; ++i) {
                pv_gemm_one_tile(topk_idx, i);
            }
        }

        // epilogue
        final_rescale_and_store_o();
    }
};

template<int D_QK, bool HAVE_TOPK_LENGTH>
class SparsePrefillGatherKernel {
public:
    using ElementKV = cutlass::bfloat16_t;
    using Params = SparseAttnFwdParams;

    static constexpr int NUM_THREADS = 128;
    static constexpr int SUBGROUP_SIZE = intel::sg_size;
    static constexpr int NUM_SUBGROUPS = NUM_THREADS / SUBGROUP_SIZE;
    static constexpr int B_TOPK = 64;
    static constexpr int ELEMENTS_PER_LANE = 4;
    using PackedElementKV = uint64_t;
    static_assert(NUM_THREADS % SUBGROUP_SIZE == 0, "NUM_THREADS must be divisible by SUBGROUP_SIZE");
    static_assert(sizeof(PackedElementKV) == sizeof(ElementKV) * ELEMENTS_PER_LANE,
                  "PackedElementKV must cover exactly one lane chunk");
    static_assert(D_QK % (SUBGROUP_SIZE * ELEMENTS_PER_LANE) == 0,
                  "D_QK must be divisible by one subgroup span");

    CUTLASS_DEVICE
    void operator()(const Params& params, char* smem_buf) const {
        const int thr_id = int(ThreadIdxX());
        const int sg_id = thr_id / SUBGROUP_SIZE;
        const int lane_id = thr_id % SUBGROUP_SIZE;
        const int seq_idx = int(BlockIdxX());
        const int topk_block_idx = int(BlockIdxY());
        const int topk_base = topk_block_idx * B_TOPK;
        const int real_topk_length = HAVE_TOPK_LENGTH ? params.topk_length[seq_idx] : params.topk;

        const int* indices = params.indices + seq_idx * params.stride_indices_s_q;
        ElementKV* gathered_k = params.gathered_k + seq_idx * params.stride_gathered_k_s_q;
        int* gathered_valid_mask = params.gathered_valid_mask + seq_idx * params.stride_gathered_mask_s_q;
        const bool can_pack = params.stride_kv_s_kv % ELEMENTS_PER_LANE == 0
            && params.stride_gathered_k_topk % ELEMENTS_PER_LANE == 0
            && (reinterpret_cast<uintptr_t>(params.kv) & (sizeof(PackedElementKV) - 1)) == 0
            && (reinterpret_cast<uintptr_t>(gathered_k) & (sizeof(PackedElementKV) - 1)) == 0;

        for (int local_topk_idx = sg_id; local_topk_idx < B_TOPK; local_topk_idx += NUM_SUBGROUPS) {
            const int topk_idx = topk_base + local_topk_idx;
            if (topk_idx >= params.topk) {
                continue;
            }

            const bool in_range = topk_idx < real_topk_length;
            const int token_idx = in_range ? indices[topk_idx] : -1;
            const bool valid_token = token_idx >= 0 && token_idx < params.s_kv;

            const int gathered_row_offset = topk_idx * params.stride_gathered_k_topk;
            const int kv_row_offset = valid_token ? token_idx * params.stride_kv_s_kv : 0;
            for (int d_base = lane_id * ELEMENTS_PER_LANE; d_base < D_QK; d_base += SUBGROUP_SIZE * ELEMENTS_PER_LANE) {
                const int gathered_offset = gathered_row_offset + d_base;
                if (can_pack) {
                    const PackedElementKV value = valid_token
                        ? *reinterpret_cast<const PackedElementKV*>(params.kv + kv_row_offset + d_base)
                        : PackedElementKV(0);
                    *reinterpret_cast<PackedElementKV*>(gathered_k + gathered_offset) = value;
                } else {
                    CUTE_UNROLL
                    for (int vec_offset = 0; vec_offset < ELEMENTS_PER_LANE; ++vec_offset) {
                        const int d_idx = d_base + vec_offset;
                        const int scalar_gathered_offset = gathered_row_offset + d_idx;
                        const ElementKV value = valid_token ? params.kv[kv_row_offset + d_idx] : ElementKV(0);
                        gathered_k[scalar_gathered_offset] = value;
                    }
                }
            }
        }

        for (int local_topk_idx = thr_id; local_topk_idx < B_TOPK; local_topk_idx += NUM_THREADS) {
            const int topk_idx = topk_base + local_topk_idx;
            if (topk_idx < params.topk) {
                const bool in_range = topk_idx < real_topk_length;
                const int token_idx = in_range ? indices[topk_idx] : -1;
                gathered_valid_mask[topk_idx] = (int)(token_idx >= 0 && token_idx < params.s_kv);
            }
        }
    }
};

template<int D_QK, bool HAVE_TOPK_LENGTH>
void launch_sparse_mla_prefill_gather_kernel(const XPUSparseAttnFwdParams& params) {
    using Kernel = SparsePrefillGatherKernel<D_QK, HAVE_TOPK_LENGTH>;

    dim3 block(Kernel::NUM_THREADS, 1, 1);
    dim3 grid(params.s_q, ceil_div(params.topk, Kernel::B_TOPK), 1);

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    using namespace compat::experimental;

    int smem_size = 0;
    compat::experimental::launch_properties launch_props {
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<intel::sg_size>,
        // Gather kernel uses the XE2 GRF mode for more parallel subgroups.
      sycl::ext::intel::experimental::grf_size<128>
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel>(policy, params.queue, static_cast<SparseAttnFwdParams>(params));
}

template<int D_QK, bool HAVE_TOPK_LENGTH, bool HAS_ATTN_SINK, int B_H>
void launch_mla_prefill_fwd_kernel_policy(const XPUSparseAttnFwdParams& params) {
    using Traits = KernelTraits<B_H>;

    TORCH_CHECK(params.h_kv == 1, "h_kv must be 1");
    TORCH_CHECK(params.h_q > 1, "h_q must be > 1");
    TORCH_CHECK(params.d_qk == D_QK, "Invalid d_qk for this kernel instantiation");
    TORCH_CHECK(params.d_v == Traits::D_V, "d_v must match KernelTraits::D_V");

    using Kernel = DensePrefillFwdKernel<D_QK, HAVE_TOPK_LENGTH, HAS_ATTN_SINK, Traits>;

    dim3 block(Traits::NUM_THREADS, 1, 1);
    dim3 grid(ceil_div(params.h_q, Traits::B_H) * params.s_q, Traits::V_SPLIT, 1);

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    const int smem_size = Kernel::SharedStorageSize;

    compat::experimental::launch_properties launch_props {
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<Traits::SUBGROUP_SIZE>,
      sycl::ext::intel::experimental::grf_size<256>
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel>(policy, params.queue, static_cast<SparseAttnFwdParams>(params));
}

inline int sparse_mla_prefill_select_b_h(const XPUSparseAttnFwdParams& params) {
    TORCH_CHECK(params.h_q > 1, "h_q must be > 1");
    if (params.h_q < 8) return 8;
    if (params.h_q < 16) return 16;
    if (params.h_q < 32) return 32;
    return 64;
}

template<int D_QK, bool HAVE_TOPK_LENGTH, bool HAS_ATTN_SINK>
void launch_mla_prefill_fwd_kernel(const XPUSparseAttnFwdParams& params) {
    switch (sparse_mla_prefill_select_b_h(params)) {
        case 8:
            launch_mla_prefill_fwd_kernel_policy<D_QK, HAVE_TOPK_LENGTH, HAS_ATTN_SINK, 8>(params);
            break;
        case 16:
            launch_mla_prefill_fwd_kernel_policy<D_QK, HAVE_TOPK_LENGTH, HAS_ATTN_SINK, 16>(params);
            break;
        case 32:
            launch_mla_prefill_fwd_kernel_policy<D_QK, HAVE_TOPK_LENGTH, HAS_ATTN_SINK, 32>(params);
            break;
        default:
            launch_mla_prefill_fwd_kernel_policy<D_QK, HAVE_TOPK_LENGTH, HAS_ATTN_SINK, 64>(params);
            break;
    }
}

template<int D_QK, bool HAVE_TOPK_LENGTH, bool HAS_ATTN_SINK>
void launch_sparse_mla_prefill_fwd_kernel(const XPUSparseAttnFwdParams& params) {
    launch_sparse_mla_prefill_gather_kernel<D_QK, HAVE_TOPK_LENGTH>(params);
    launch_mla_prefill_fwd_kernel<D_QK, HAVE_TOPK_LENGTH, HAS_ATTN_SINK>(params);
}

} // namespace FLASH_NAMESPACE
