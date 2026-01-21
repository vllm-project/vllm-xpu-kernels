
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
#include "chunk_prefill.hpp"
#include "fmha_utils.hpp"

using namespace cute;

template <typename chunk_policy, bool... Bs>
void policy_dispatch_impl(
    sycl::queue& queue, CutlassType cuType, const chunk_prefill_args_t& args) {
  policy_dispatch<chunk_policy, Bs...>(queue, cuType, args);
}

template <typename chunk_policy, bool... Bs, typename... Ts>
void policy_dispatch_impl(
    sycl::queue& queue,
    CutlassType cuType,
    const chunk_prefill_args_t& args,
    bool b,
    Ts... ts) {
  if (b) {
    policy_dispatch_impl<chunk_policy, Bs..., true>(queue, cuType, args, ts...);
  } else {
    policy_dispatch_impl<chunk_policy, Bs..., false>(
        queue, cuType, args, ts...);
  }
}

void cutlass_chunk_prefill_impl(
    sycl::queue& queue,
    const at::Tensor& query,      // [seq_q, heads, head_size]
    const at::Tensor& key_cache,  // [num_block, block_size, heads, head_size]
    const at::Tensor& value_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    double sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink);
