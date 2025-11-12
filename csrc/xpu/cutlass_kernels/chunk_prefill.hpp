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

#include "./collective/chunk_prefill_scheduler.hpp"
#include "./collective/chunk_prefill_epilogue.hpp"
#include "./collective/chunk_prefill_softmax.hpp"
#include "chunk_prefill_kernel.hpp"

#include "fmha_utils.hpp"

using namespace cute;

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
  bool is_causal = false;
  bool is_local = false;
  bool is_sink = false;
};

template <class FMHAChunkPrefillKernel>
struct KernelLauncher {
  using StrideQ = typename FMHAChunkPrefillKernel::StrideQ;
  using StrideK = typename FMHAChunkPrefillKernel::StrideK;
  using StrideV = typename FMHAChunkPrefillKernel::StrideV;
  using StrideO = typename FMHAChunkPrefillKernel::StrideO;

  using ElementQ = typename FMHAChunkPrefillKernel::ElementQ;
  using ElementK = typename FMHAChunkPrefillKernel::ElementK;
  using ElementV = typename FMHAChunkPrefillKernel::ElementV;
  using ElementAcc = typename FMHAChunkPrefillKernel::ElementAccumulator;

  using CollectiveEpilogue =
      typename FMHAChunkPrefillKernel::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;
  using ElementSink = typename CollectiveEpilogue::ElementSink;

  using ProblemShapeType = typename FMHAChunkPrefillKernel::ProblemShape;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K_cache;
  StrideV stride_V_cache;
  StrideO stride_O;
  uint64_t seed = 0;

  ProblemShapeType initialize(const chunk_prefill_args_t& args) {
    auto problem_shape = cute::make_tuple(
        1, args.num_heads_q, args.num_heads_k, args.total_seqlen_q,
        args.total_seqlen_k, args.head_size, args.head_size);
    auto problem_shape_out = cute::make_tuple(
        args.batch_size, args.num_heads_q, args.num_heads_k,
        cutlass::fmha::collective::VariableLength{args.max_queries},  // cu_q
        cutlass::fmha::collective::VariableLength{
            args.max_keys},  // cu_kv_cache
        args.head_size, args.head_size);
    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv_cache,
          head_size_qk, head_size_vo] = problem_shape;
    auto group_q_size = num_heads_q / num_heads_kv;
    auto group_q_num = num_heads_q / group_q_size;

    stride_Q = cutlass::make_cute_packed_stride(
        StrideQ{},
        cute::make_shape(seq_len_qo, num_heads_q * head_size_qk, batch));
    stride_K_cache = cutlass::make_cute_packed_stride(
        StrideK{},
        cute::make_shape(seq_len_kv_cache, num_heads_kv * head_size_qk, batch));
    stride_V_cache = cutlass::make_cute_packed_stride(
        StrideV{},
        cute::make_shape(head_size_vo * num_heads_kv, seq_len_kv_cache, batch));

    stride_O = cutlass::make_cute_packed_stride(
        StrideO{}, cute::make_shape(seq_len_qo * group_q_size,
                                    group_q_num * head_size_vo, batch));

    get<3>(problem_shape_out).cumulative_length =
        reinterpret_cast<int*>(args.cu_seqlens_q);
    get<4>(problem_shape_out).cumulative_length =
        reinterpret_cast<int*>(args.cu_seqlens_k);

    return problem_shape_out;
  }

  cutlass::Status run(sycl::queue& queue, const chunk_prefill_args_t& args,
                      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = initialize(args);

    typename FMHAChunkPrefillKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {reinterpret_cast<ElementQ*>(args.query), stride_Q,
         reinterpret_cast<ElementK*>(args.key), stride_K_cache,
         reinterpret_cast<ElementV*>(args.value), stride_V_cache,
         static_cast<int*>(args.block_table), args.block_size,
         args.max_blocks_per_seq, args.total_seqlen_k, args.window_size_left,
         args.window_size_right},
        {args.sm_scale},
        {reinterpret_cast<ElementOutput*>(args.out), stride_O,
         reinterpret_cast<ElementSink*>(args.sm_sink)},
        hw_info};

    // Define device-global scratch memory
    size_t workspace_size =
        FMHAChunkPrefillKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FMHAChunkPrefillKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    FMHAChunkPrefillKernel::initialize_workspace(arguments, workspace.get());

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto params = FMHAChunkPrefillKernel::to_underlying_arguments(
        arguments, workspace.get());

    // Run the Flash Attention implementation.
    run(queue, params);

    return cutlass::Status::kSuccess;
  }

  static void run(sycl::queue& queue,
                  typename FMHAChunkPrefillKernel::Params params) {
    dim3 const block = FMHAChunkPrefillKernel::get_block_shape();
    dim3 const grid = FMHAChunkPrefillKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAChunkPrefillKernel::SharedStorageSize;

    const auto sycl_block = syclcompat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = syclcompat::dim3(grid.x, grid.y, grid.z);

    syclcompat::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    syclcompat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<
            FMHAChunkPrefillKernel::DispatchPolicy::SubgroupSize>};
    syclcompat::experimental::launch_policy policy{sycl_grid, sycl_block,
                                                   launch_props, kernel_props};
    auto event = syclcompat::experimental::launch<
        cutlass::device_kernel<FMHAChunkPrefillKernel>>(policy, queue, params);

    // EventManager::getInstance().addEvent(event);
  }
};

template <typename TileShapeQK, typename TileShapePV, typename TileShapeOutput,
          typename SubgroupLayout, int PipelineStages,
          typename ElementInputQ = bfloat16_t,
          typename MMAOperation = XE_8x16x16_F32BF16BF16F32_TT,
          typename GmemTiledCopyQ = XE_2D_U16x8x32_LD_N,
          typename GmemTiledCopyK = XE_2D_U16x16x16_LD_T,
          typename GmemTiledCopyV = XE_2D_U16x16x32_LD_V,
          typename ElementAccumulator = float,
          typename ElementComputeEpilogue = float,
          typename GmemTiledCopyStore = XE_2D_U16x8x16_ST_N>
struct FMHAKernel {
  template <class Scheduler, bool Causal, bool Local, bool Sink>
  static void run(sycl::queue& queue, const chunk_prefill_args_t& args) {
    cutlass::KernelHardwareInfo hw_info;

    static constexpr bool PagedKV = true;
    using LayoutQ = cutlass::layout::RowMajor;
    using LayoutK = cutlass::layout::ColumnMajor;
    using LayoutV = cutlass::layout::RowMajor;
    using LayoutO = cutlass::layout::RowMajor;

    using ElementInputKV = ElementInputQ;
    using ElementOutput = ElementInputQ;
    using ElementSink = ElementInputQ;

    using GEMMDispatchPolicy =
        cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
    using CollectiveEpilogue =
        cutlass::flash_attention::collective::FlashChunkPrefillEpilogue<
            Sink, EpilogueDispatchPolicy, MMAOperation, TileShapeOutput,
            SubgroupLayout, ElementComputeEpilogue, ElementOutput,
            cutlass::gemm::TagToStrideC_t<LayoutO>, ElementOutput,
            GmemTiledCopyStore, ElementSink>;
    using CollectiveSoftmaxEpilogue =
        cutlass::flash_attention::collective::FlashChunkPrefillSoftmaxEpilogue<
            Causal, Local, EpilogueDispatchPolicy, ElementAccumulator>;

    using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>;
    using namespace cutlass::fmha::collective;
    using ProblemShapeVarlen =
        cute::tuple<int, int, int, VariableLength, VariableLength, int, int>;
    using ProblemShapeType = ProblemShapeVarlen;

    // Mainloop
    using CollectiveMainloop =
        cutlass::flash_attention::collective::FlashChunkPrefillMma<
            GEMMDispatchPolicy, ProblemShapeType, ElementInputQ,
            cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInputKV,
            cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInputKV,
            cutlass::gemm::TagToStrideB_t<LayoutV>, MMAOperation, TileShapeQK,
            TileShapePV, SubgroupLayout,
            GmemTiledCopyQ,  // Q
            GmemTiledCopyK,  // K
            GmemTiledCopyV,  // V,
            Causal, Local, PagedKV>;

    using FMHAChunkPrefillKernel =
        cutlass::flash_attention::kernel::FMHAPrefillChunk<
            ProblemShapeType, CollectiveMainloop, CollectiveSoftmaxEpilogue,
            CollectiveEpilogue, Scheduler>;

    KernelLauncher<FMHAChunkPrefillKernel> launcher;

    launcher.run(queue, args, hw_info);
  }

  template <bool... Bs>
  static void kernel_dispatch(sycl::queue& queue,
                              const chunk_prefill_args_t& args) {
    return run<cutlass::flash_attention::IndividualScheduler, Bs...>(queue,
                                                                     args);
  }

  template <bool... Bs, typename... Ts>
  static void kernel_dispatch(sycl::queue& queue,
                              const chunk_prefill_args_t& args, bool b,
                              Ts... ts) {
    if (b) {
      kernel_dispatch<Bs..., true>(queue, args, ts...);
    } else {
      kernel_dispatch<Bs..., false>(queue, args, ts...);
    }
  }
};

template <typename chunk_policy>
void policy_dispatch(sycl::queue& queue, CutlassType cuType,
                     const chunk_prefill_args_t& args) {
  const int PipelineStages = 2;
  if (cuType == CutlassType::half) {
    FMHAKernel<typename chunk_policy::ShapeQK, typename chunk_policy::ShapePV,
               typename chunk_policy::ShapeOutPut,
               typename chunk_policy::SubgroupLayout, PipelineStages,
               cutlass::half_t,
               XE_8x16x16_F32F16F16F32_TT>::kernel_dispatch(queue, args,
                                                            args.is_causal,
                                                            args.is_local,
                                                            args.is_sink);
  } else {
    FMHAKernel<typename chunk_policy::ShapeQK, typename chunk_policy::ShapePV,
               typename chunk_policy::ShapeOutPut,
               typename chunk_policy::SubgroupLayout,
               PipelineStages>::kernel_dispatch(queue, args, args.is_causal,
                                                args.is_local, args.is_sink);
  }
}

void cutlass_chunk_prefill_impl(
    sycl::queue& queue,
    const at::Tensor& query,      // [seq_q, heads, head_size]
    const at::Tensor& key_cache,  // [num_block, block_size, heads, head_size]
    const at::Tensor& value_cache, at::Tensor& out,
    const at::Tensor& block_table, const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, int max_seqlen_q, int max_seqlen_k,
    double sm_scale, std::optional<const at::Tensor>& sm_sink_,
    int window_size_left, int window_size_right, bool is_causal, bool is_local,
    bool is_sink) {
  int num_block = key_cache.size(0);
  int block_size = key_cache.size(1);
  int num_heads_q = query.size(1);
  int num_heads_kv = key_cache.size(2);
  int head_size = query.size(2);
  int batch_size = cu_seqlens_q.numel() - 1;
  int max_blocks_per_seq = block_table.size(1);
  int total_seqlen_q = query.size(0);
  int total_seqlen_k = num_block * block_size;

  if (is_local) {
    window_size_left = window_size_left == -1 ? max_seqlen_k : window_size_left;
    window_size_right =
        window_size_right == -1 ? max_seqlen_k : window_size_right;
  }

  chunk_prefill_args_t args = {query.data_ptr(),
                               key_cache.data_ptr(),
                               value_cache.data_ptr(),
                               out.data_ptr(),
                               block_table.data_ptr(),
                               cu_seqlens_q.data_ptr(),
                               cu_seqlens_k.data_ptr(),
                               max_seqlen_q,
                               max_seqlen_k,
                               total_seqlen_q,
                               total_seqlen_k,
                               static_cast<float>(sm_scale),
                               is_sink ? sm_sink_.value().data_ptr() : nullptr,
                               batch_size,
                               num_heads_q,
                               num_heads_kv,
                               head_size,
                               max_blocks_per_seq,
                               block_size,
                               window_size_left,
                               window_size_right,
                               is_causal,
                               is_local,
                               is_sink};
  CutlassType cuType = aten_to_Cutlass_dtype(query);

  static constexpr int max_head_size = 256;
  TORCH_CHECK(head_size <= max_head_size,
              "FMHA forward only supports head dimension at most " +
                  std::to_string(max_head_size));

  if (args.head_size == HEAD_SIZE_LIMIT_0) {
    policy_dispatch<chunk_policy_head64>(queue, cuType, args);
  } else if (args.head_size == HEAD_SIZE_LIMIT_1) {
    policy_dispatch<chunk_policy_head128>(queue, cuType, args);
  } else if (args.head_size == HEAD_SIZE_LIMIT_2) {
    policy_dispatch<chunk_policy_head192>(queue, cuType, args);
  } else if (args.head_size == HEAD_SIZE_LIMIT_3) {
    policy_dispatch<chunk_policy_head256>(queue, cuType, args);
  } else {
    TORCH_CHECK(false, "Unsupported head size for fmha");
  }
}
