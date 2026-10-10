#include "csrc/utils.h"
#include "moe_shared_fused_interface.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/moe_shared_fused_decode_xe2.h"
#endif

// Routed rows (num_tokens * top_k) up to which the 8-row GEMM tile is used
// by default. benchmark/benchmark_moe_shared_fused_decode.py on BMG (H=2048,
// I=256, E=256, top_k=8) shows tile 8 at or slightly ahead of tile 16 for
// every decode M (1..8, i.e. up to 64 rows); larger M keeps tile 16.
constexpr int64_t kMoeSharedFusedTileM8MaxRows = 64;

void moe_shared_fused_decode_interface(
    torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& shared_w13,
    const torch::Tensor& shared_w13_scale,
    const torch::Tensor& shared_w2,
    const torch::Tensor& shared_w2_scale,
    const torch::Tensor& shared_gate,
    torch::Tensor& ws,
    int64_t tile_m) {
  if (vllm::xpu::is_xe2_arch() || vllm::xpu::is_xe3_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Xe3 reuses the Xe2 implementation, as the grouped GEMM does.
    if (tile_m == 0) {
      tile_m = x.size(0) * topk_ids.size(1) <= kMoeSharedFusedTileM8MaxRows
                   ? kMoeSharedFusedTileM8
                   : kMoeSharedFusedTileM16;
    }
    moe_shared_fused_decode_xe2(
        output,
        x,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w2,
        w2_scale,
        shared_w13,
        shared_w13_scale,
        shared_w2,
        shared_w2_scale,
        shared_gate,
        ws,
        tile_m);
    return;
#else
    TORCH_CHECK(false, "XE2 kernels are not enabled in this build.");
#endif
  }
  TORCH_CHECK(false, "moe_shared_fused_decode: unsupported device arch.");
}
