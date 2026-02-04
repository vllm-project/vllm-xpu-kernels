#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "collective/chunk_prefill_scheduler.hpp"
#include "collective/chunk_prefill_epilogue.hpp"
#include "kernel/paged_decode_kernel.hpp"

#include "fmha_utils.hpp"

using namespace cute;

using decode_policy_q8_h64 = decode_policy_qpacked_head<_8, _64>;
using decode_policy_q8_h96 = decode_policy_qpacked_head<_8, _96>;
using decode_policy_q8_h128 = decode_policy_qpacked_head<_8, _128>;
using decode_policy_q8_h192 = decode_policy_qpacked_head<_8, _192>;
using decode_policy_q8_h256 = decode_policy_qpacked_head<_8, _256>;
using decode_policy_q16_h64 = decode_policy_qpacked_head<_16, _64>;
using decode_policy_q16_h96 = decode_policy_qpacked_head<_16, _96>;
using decode_policy_q16_h128 = decode_policy_qpacked_head<_16, _128>;
using decode_policy_q16_h192 = decode_policy_qpacked_head<_16, _192>;
using decode_policy_q16_h256 = decode_policy_qpacked_head<_16, _256>;

struct paged_decode_args_t {
  void* query;
  void* key;
  void* value;
  void* out;
  void* tem_out;
  void* exp_sums;
  void* max_logits;
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  float sm_scale;
  void* sm_sink;
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int head_size;
  int max_blocks_per_seq;
  int block_size;
  int window_size_left = -1;
  int window_size_right = -1;
  bool is_varlen = false;
  bool is_paged = false;
  bool is_causal = false;
  bool is_local = false;
  bool is_sink = false;
  int num_kv_splits = 1;
};

template <class FMHAKernel, class ReductionSplitKernel, bool isVarLen>
struct DecodeKernelLauncher {
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;
  using ElementLSE = typename FMHAKernel::ElementLSE;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::DecodeProblemShape<isVarLen>;
  using ProblemShapeTypeInit = cutlass::fmha::kernel::DecodeProblemShape<false>;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideO stride_Oaccum;
  StrideO stride_exp_sums;
  StrideO stride_max_logits;

  int num_kv_splits;

  ProblemShapeType initialize(const paged_decode_args_t& args) {
    ProblemShapeType shape;
    ProblemShapeTypeInit shape_init;
    auto batch = shape.batch = shape_init.batch = args.batch_size;
    auto num_heads_q = shape.num_heads_q = shape_init.num_heads_q =
        args.num_heads_q;
    auto num_heads_kv = shape.num_heads_kv = shape_init.num_heads_kv =
        args.num_heads_k;
    auto head_size_qk = shape.head_size_qk = shape_init.head_size_qk =
        args.head_size;
    auto head_size_vo = shape.head_size_vo = shape_init.head_size_vo =
        args.head_size;

    if constexpr (isVarLen) {
      batch = shape_init.batch = 1;
      shape_init.seq_len_qo = args.total_seqlen_q;
      shape_init.seq_len_kv = args.total_seqlen_k;

      shape.seq_len_qo =
          cutlass::fmha::collective::VariableLength{args.max_queries};
      shape.seq_len_qo.cumulative_length =
          reinterpret_cast<int*>(args.cu_seqlens_q);
      shape.seq_len_kv =
          cutlass::fmha::collective::VariableLength{args.max_keys};
      shape.seq_len_kv.cumulative_length =
          reinterpret_cast<int*>(args.cu_seqlens_k);
    } else {
      shape.seq_len_qo = shape_init.seq_len_qo = args.max_queries;
      shape.seq_len_kv = shape_init.seq_len_kv = args.max_keys;
    }

    auto seq_len_qo = shape_init.seq_len_qo;
    auto seq_len_kv = shape_init.seq_len_kv;

    num_kv_splits = args.num_kv_splits;

    stride_Q = cutlass::make_cute_packed_stride(
        StrideQ{},
        cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch));
    stride_K = cutlass::make_cute_packed_stride(
        StrideK{},
        cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch));
    stride_V = cutlass::make_cute_packed_stride(
        StrideV{},
        cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch));
    stride_O = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch));
    stride_Oaccum = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(
            seq_len_qo, head_size_vo, num_heads_q * num_kv_splits, batch));

    stride_exp_sums = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch));

    stride_max_logits = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch));

    return shape;
  }

  cutlass::Status
  run(sycl::queue& queue,
      const paged_decode_args_t& args,
      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(args);

    typename FMHAKernel::Arguments arguments{
        {
            shape,
            reinterpret_cast<ElementQ*>(args.query),
            stride_Q,
            reinterpret_cast<ElementK*>(args.key),
            stride_K,
            reinterpret_cast<ElementV*>(args.value),
            stride_V,
            reinterpret_cast<ElementO*>(args.tem_out),
            stride_Oaccum,
            reinterpret_cast<ElementLSE*>(args.exp_sums),
            stride_exp_sums,
            reinterpret_cast<ElementLSE*>(args.max_logits),
            stride_max_logits,
            reinterpret_cast<ElementQ*>(args.sm_sink),
        },
        {args.sm_scale,
         static_cast<int*>(args.block_table),
         args.block_size,
         args.max_blocks_per_seq,
         args.total_seqlen_k},
        {},
        hw_info,
        args.num_kv_splits};

    typename ReductionSplitKernel::Arguments reduce_arg{
        {shape,
         reinterpret_cast<ElementO*>(args.out),
         stride_O,
         reinterpret_cast<ElementO*>(args.tem_out),
         stride_Oaccum,
         reinterpret_cast<ElementLSE*>(args.exp_sums),
         stride_exp_sums,
         reinterpret_cast<ElementLSE*>(args.max_logits),
         stride_max_logits},
        hw_info,
        args.num_kv_splits};

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    size_t reduce_workspace_size =
        ReductionSplitKernel::get_workspace_size(reduce_arg);
    cutlass::device_memory::allocation<uint8_t> workspace(
        workspace_size + reduce_workspace_size);

    if (!FMHAKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << args.batch_size << 'x'
                << args.num_heads_q << 'x' << args.max_queries << 'x'
                << args.max_keys << 'x' << args.head_size << 'x'
                << args.head_size << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    FMHAKernel::initialize_workspace(arguments, workspace.get());

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto params =
        FMHAKernel::to_underlying_arguments(arguments, workspace.get());
    auto reduce_params = ReductionSplitKernel::to_underlying_arguments(
        reduce_arg, workspace.get() + workspace_size);

    ReductionSplitKernel::initialize_workspace(
        reduce_arg, workspace.get() + workspace_size);
    run(queue, params, reduce_params, args.num_kv_splits > 1);

    return cutlass::Status::kSuccess;
  }

  static void
  run(sycl::queue& queue,
      typename FMHAKernel::Params params,
      typename ReductionSplitKernel::Params reduce_params,
      bool need_reduce) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group
    // scratch memory extension
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};
    auto event =
        compat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(
            policy, queue, params);
    EventManager::getInstance().addEvent(event);

    // event.wait();

    if (need_reduce) {
      dim3 const reduce_grid =
          ReductionSplitKernel::get_grid_shape(reduce_params);
      int reduce_smem_size = ReductionSplitKernel::SharedStorageSize;
      const auto reduce_sycl_block = compat::dim3(block.x, block.y, block.z);
      const auto reduce_sycl_grid =
          compat::dim3(reduce_grid.x, reduce_grid.y, reduce_grid.z);
      compat::experimental::launch_properties launch_props_reduce{
          syclex::work_group_scratch_size(reduce_smem_size),
      };
      compat::experimental::launch_policy reduce_policy{
          reduce_sycl_grid, reduce_sycl_block, launch_props_reduce, kernel_props};

      // wait for FA kernel finished
      // maybe no need wait here if launched with in-order queue

      auto reduce_event = compat::experimental::launch<
          cutlass::device_kernel<ReductionSplitKernel>>(
          reduce_policy, queue, reduce_params);

      // reduce_event.wait();

      EventManager::getInstance().addEvent(reduce_event);
    }
  }
};

template <
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_, /* void -> default */
    int PipelineStages,
    bool Causal = false,
    bool Local = false,
    bool Sink = false,
    typename ElementQ = bfloat16_t,
    typename ElementK = bfloat16_t,
    typename ElementV = bfloat16_t,
    typename ElementO = float,
    typename MMAOperation_ = void, /* void -> default */
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename StrideOaccum = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void, /* void -> default block 2D */
    typename GmemTiledCopyK = void,
    typename GmemTiledCopyV = void,
    typename GmemTiledCopyO = void>
struct PagedDecodeConfig {
  static constexpr int SGTileQ =
      get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<
      is_void_v<MMAOperation_>,
      XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>,
      MMAOperation_>;
  using SubgroupLayoutPV = cute::conditional_t<
      is_void_v<SubgroupLayoutPV_>,
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
      SubgroupLayoutPV_>;

  template <class Scheduler>
  static void run(sycl::queue& queue, const paged_decode_args_t& args) {
    constexpr bool VarLen = true;
    constexpr bool Paged = true;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    using ProblemShapeType = cutlass::fmha::kernel::DecodeProblemShape<VarLen>;

    using TiledMMAQK = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapeQK>,
        SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapePV>,
        SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(
          make_gmem_ptr(&val),
          make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideOaccum{}));
    using TensorLSE = decltype(make_dummy_tensor(float{}, StrideO{}));

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::DecodeFwdMainloop<
        MainloopDispatchPolicy,
        Paged,
        Causal,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV>;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::DecodeFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        TensorLSE,
        void,
        Sink>;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdSplitKVKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;

    using ReduceSplitKernel = cutlass::fmha::kernel::ReduceSplitK<
        ProblemShapeType,
        cutlass::fmha::kernel::XeReduceSplitKTileScheduler,
        FMHAKernel>;

    DecodeKernelLauncher<FMHAKernel, ReduceSplitKernel, VarLen> launcher;

    launcher.run(queue, args, hw_info);
  }

  static void
  kernel_dispatch(sycl::queue& queue, const paged_decode_args_t& args) {
    return run<cutlass::fmha::kernel::DecodeTileScheduler>(queue, args);
  }
};

// Template function for explicit instantiation
template <typename decode_policy, bool Causal, bool Local, bool Sink>
void decode_policy_dispatch_impl(
    sycl::queue& queue, CutlassDType cuType, const paged_decode_args_t& args) {
  const int PipelineStages = 1;
  if (cuType == CutlassDType::half) {
    return PagedDecodeConfig<
        typename decode_policy::ShapeQK,
        typename decode_policy::ShapePV,
        typename decode_policy::ShapeOut,
        typename decode_policy::SubgroupLayoutQK,
        void,
        PipelineStages,
        Causal,
        Local,
        Sink,
        half_t,
        half_t,
        half_t,
        half_t>::kernel_dispatch(queue, args);
  } else {
    return PagedDecodeConfig<
        typename decode_policy::ShapeQK,
        typename decode_policy::ShapePV,
        typename decode_policy::ShapeOut,
        typename decode_policy::SubgroupLayoutQK,
        void,
        PipelineStages,
        Causal,
        Local,
        Sink,
        bfloat16_t,
        bfloat16_t,
        bfloat16_t,
        bfloat16_t>::kernel_dispatch(queue, args);
  }
}
