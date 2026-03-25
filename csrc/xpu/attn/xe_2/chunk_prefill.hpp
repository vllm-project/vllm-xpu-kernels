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
#include "kernel/chunk_prefill_kernel.hpp"

#include "fmha_utils.hpp"

using namespace cute;

// Forward declaration - implementation moved to chunk_prefill_dispatch.hpp
// This reduces the amount of template code parsed in each translation unit
template <typename chunk_policy, bool Paged, bool Causal, bool Local, bool Sink>
void policy_dispatch_impl(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args);

struct chunk_prefill_args_t {
  void* query;
  void* key;
  void* value;
  void* out;
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  void* k_scale;
  void* v_scale;
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
};

// ============================================================================
// Type traits to reduce template instantiation overhead
// ============================================================================

// Helper to select element types based on runtime CutlassDType
template <CutlassDType QType, CutlassDType KType>
struct ElementTypeSelector {
  static_assert(
      QType == CutlassDType::half || QType == CutlassDType::bfloat16,
      "Q must be half or bfloat16");
  using ElementQ =
      cute::conditional_t<QType == CutlassDType::half, half_t, bfloat16_t>;
  using ElementO = ElementQ;

  static_assert(
      KType == CutlassDType::half || KType == CutlassDType::bfloat16 ||
          KType == CutlassDType::float8_e4m3 ||
          KType == CutlassDType::float8_e5m2,
      "K must be half, bfloat16, float8_e4m3, or float8_e5m2");
  using ElementK = cute::conditional_t<
      KType == CutlassDType::half,
      half_t,
      cute::conditional_t<
          KType == CutlassDType::bfloat16,
          bfloat16_t,
          cute::conditional_t<
              KType == CutlassDType::float8_e4m3,
              float_e4m3_t,
              float_e5m2_t>>>;
  using ElementV = ElementK;
};

// Runtime type dispatch helper - reduces template explosion by using function
// pointers
using FMHARunFunc =
    void (*)(sycl::queue& queue, const chunk_prefill_args_t& args);

namespace detail {

// Helper to reduce template nesting: pre-compute stride shapes
inline auto make_stride_shapes(
    int seq_len_qo,
    int seq_len_kv,
    int head_size_qk,
    int head_size_vo,
    int num_heads_q,
    int num_heads_kv,
    int batch) {
  return std::make_tuple(
      cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch),
      cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch),
      cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch),
      cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch));
}

// Non-templated stride computation to reduce compile time
// The actual strides are computed inside the kernel launcher template

}  // namespace detail

template <class FMHAKernel, bool isVarLen>
struct KernelLauncher {
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;
  using ProblemShapeTypeInit = cutlass::fmha::kernel::FMHAProblemShape<false>;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  ProblemShapeType initialize(const chunk_prefill_args_t& args) {
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

    return shape;
  }

  cutlass::Status
  run(sycl::queue& queue,
      const chunk_prefill_args_t& args,
      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(args);

    typename FMHAKernel::Arguments arguments{
        {shape,
         reinterpret_cast<ElementQ*>(args.query),
         stride_Q,
         reinterpret_cast<ElementK*>(args.key),
         stride_K,
         reinterpret_cast<ElementV*>(args.value),
         stride_V,
         reinterpret_cast<ElementO*>(args.out),
         stride_O,
         reinterpret_cast<ElementQ*>(args.sm_sink)},
        {args.sm_scale,
         args.k_scale,
         args.v_scale,
         static_cast<int*>(args.block_table),
         args.block_size,
         args.max_blocks_per_seq,
         args.total_seqlen_k,
         args.window_size_left,
         args.window_size_right},
        {},
        hw_info};

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Initialize the workspace
    FMHAKernel::initialize_workspace(arguments, workspace.get());

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto params =
        FMHAKernel::to_underlying_arguments(arguments, workspace.get());

    run_kernel(queue, params);

    return cutlass::Status::kSuccess;
  }

  static void
  run_kernel(sycl::queue& queue, typename FMHAKernel::Params params) {
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
  }
};

// ============================================================================
// Simplified FMHAConfig with reduced template parameters
// ============================================================================

template <
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    int PipelineStages,
    bool Paged,
    bool Causal,
    bool Local,
    bool Sink,
    typename ElementQ,
    typename ElementK,
    typename ElementV,
    typename ElementO,
    typename MMAOperation_ = void,
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void,
    typename GmemTiledCopyK = void,
    typename GmemTiledCopyV = void,
    typename GmemTiledCopyO = void>
struct FMHAConfigImpl {
  static constexpr int SGTileQ =
      get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<
      is_void_v<MMAOperation_>,
      XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>,
      MMAOperation_>;
  using SubgroupLayoutPV =
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  template <class Scheduler>
  static void run(sycl::queue& queue, const chunk_prefill_args_t& args) {
    constexpr bool VarLen = true;
    cutlass::KernelHardwareInfo hw_info;

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<VarLen>;

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
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        Local,
        Paged,
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
    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
        Sink,
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO>;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;

    KernelLauncher<FMHAKernel, VarLen> launcher;
    launcher.run(queue, args, hw_info);
  }

  static void
  kernel_dispatch(sycl::queue& queue, const chunk_prefill_args_t& args) {
    return run<cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(
        queue, args);
  }
};

// Type alias for cleaner instantiation
template <typename chunk_policy, bool Paged, bool Causal, bool Local, bool Sink>
using FMHAConfig = FMHAConfigImpl<
    typename chunk_policy::ShapeQK,
    typename chunk_policy::ShapePV,
    typename chunk_policy::ShapeOut,
    typename chunk_policy::SubgroupLayoutQK,
    2,  // PipelineStages
    Paged,
    Causal,
    Local,
    Sink>;

// ============================================================================
// Type dispatch helper - reduces code duplication in policy_dispatch_impl
// ============================================================================

template <typename chunk_policy, bool Paged, bool Causal, bool Local, bool Sink>
struct FMHADispatcher {
  template <
      typename ElementQ,
      typename ElementK,
      typename ElementV,
      typename ElementO>
  static void dispatch(sycl::queue& queue, const chunk_prefill_args_t& args) {
    using Config = FMHAConfigImpl<
        typename chunk_policy::ShapeQK,
        typename chunk_policy::ShapePV,
        typename chunk_policy::ShapeOut,
        typename chunk_policy::SubgroupLayoutQK,
        2,  // PipelineStages
        Paged,
        Causal,
        Local,
        Sink,
        ElementQ,
        ElementK,
        ElementV,
        ElementO>;
    Config::kernel_dispatch(queue, args);
  }
};

// ============================================================================
// policy_dispatch_impl - implementation moved to reduce header bloat
// Each template instantiation is now smaller and compiles faster
// ============================================================================

template <typename chunk_policy, bool Paged, bool Causal, bool Local, bool Sink>
void policy_dispatch_impl(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args) {
  using Dispatcher = FMHADispatcher<chunk_policy, Paged, Causal, Local, Sink>;

  if (cuQKType.q_type == CutlassDType::half) {
    if (cuQKType.k_type == CutlassDType::half) {
      Dispatcher::template dispatch<half_t, half_t, half_t, half_t>(
          queue, args);
    } else if (cuQKType.k_type == CutlassDType::float8_e4m3) {
      Dispatcher::template dispatch<half_t, float_e4m3_t, float_e4m3_t, half_t>(
          queue, args);
    } else if (cuQKType.k_type == CutlassDType::float8_e5m2) {
      Dispatcher::template dispatch<half_t, float_e5m2_t, float_e5m2_t, half_t>(
          queue, args);
    }
  } else {  // bfloat16
    if (cuQKType.k_type == CutlassDType::bfloat16) {
      Dispatcher::
          template dispatch<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>(
              queue, args);
    } else if (cuQKType.k_type == CutlassDType::float8_e4m3) {
      Dispatcher::
          template dispatch<bfloat16_t, float_e4m3_t, float_e4m3_t, bfloat16_t>(
              queue, args);
    } else if (cuQKType.k_type == CutlassDType::float8_e5m2) {
      Dispatcher::
          template dispatch<bfloat16_t, float_e5m2_t, float_e5m2_t, bfloat16_t>(
              queue, args);
    }
  }
}
