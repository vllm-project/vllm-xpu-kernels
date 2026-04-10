#include "pytorch_shim.h"

#include "core/registration.h"
#include "xpu/attn/attn_interface.h"
#include "utils.h"
#include <torch/all.h>

namespace FLASH_NAMESPACE {

inline int get_num_splits(
    const sycl::queue& queue,
    const int& batch_size,
    const int& num_heads_kv,
    const int& max_seqlen_k,
    const int& block_size) {
  auto device = queue.get_device();
  int num_xe_cores =
      device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
      device
          .get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();

  int cur_parallel = batch_size * num_heads_kv;
  int kv_blocks = (max_seqlen_k + block_size - 1) / block_size;

  // Minimum KV blocks to benefit from splitting, aligned with the kernel's
  // kMinBlocksForSplit.  Below this the kernel falls back to single-split
  // regardless, and splitting only adds ReduceSplitK overhead.
  int min_kv_blocks = (block_size <= 64) ? 32 : 128;
  if (kv_blocks < min_kv_blocks) return 1;

  // GPU well-utilized or saturated: splitting only adds ReduceSplitK
  // overhead without improving FMHA throughput.
  if (cur_parallel >= num_xe_cores) return 1;

  // Under-utilized (cur_parallel < num_xe_cores): split to fill the GPU.
  int target_splits;
  if (num_heads_kv >= 4) {
    // Many KV heads: each split adds kv_heads WGs.  Scale inversely
    // with block_size — p64 gets ~20 splits, p128 gets ~10.
    target_splits = std::max(4, num_xe_cores * 64 / block_size);
  } else {
    // Few KV heads: target at least num_xe_cores splits for parallelism;
    // allow more for very long sequences (~10 blocks/split at p64).
    int blocks_per_split = (block_size <= 64) ? 10 : 8;
    target_splits = std::max(num_xe_cores, kv_blocks / blocks_per_split);
  }

  // Each split must process at least 3 KV blocks to amortize overhead.
  int max_splits_blocks = std::max(1, kv_blocks / 3);
  // Hard cap: beyond 40 splits, diminishing returns.
  int num_splits = std::min({target_splits, max_splits_blocks, 40});
  return std::max(1, num_splits);
}

std::vector<at::Tensor> mha_varlen_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor>& out_,
    const at::Tensor& cu_seqlens_q,  // b+1
    const at::Tensor& cu_seqlens_k,  // b+1
    std::optional<at::Tensor>& seqused_k,
    std::optional<const at::Tensor>& leftpad_k_,  // batch_size
    std::optional<at::Tensor>&
        block_table_,  // batch_size x max_num_blocks_per_seq
    std::optional<at::Tensor>& alibi_slopes_,  // num_heads or b x num_heads
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    std::optional<const at::Tensor>& k_scale,
    std::optional<const at::Tensor>& v_scale,
    float softmax_scale,
    std::optional<const at::Tensor>& softmax_sink_,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_,
    std::optional<int> num_splits) {
  auto q_type = q.scalar_type();
  auto k_type = k.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
      "VLLM Kernel XPU only supports fp16 and bf16 type");

  TORCH_CHECK(
      v.scalar_type() == k_type, "key and value must have the same dtype");
  if (k_type != at::ScalarType::Float8_e5m2 &&
      k_type != at::ScalarType::Float8_e4m3fn) {
    TORCH_CHECK(
        k.scalar_type() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(
        v.scalar_type() == q_type, "query and value must have the same dtype");
  }

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  TORCH_CHECK(
      q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(q.dim() == 3, "query must be in ragged format");
  CHECK_CONTIGUOUS(q);

  at::Tensor block_table;
  bool is_paged = block_table_.has_value();
  if (is_paged) {
    block_table = *block_table_;
    CHECK_DEVICE(block_table);
    TORCH_CHECK(
        block_table.dtype() == torch::kInt32,
        "page_table must have dtype torch.int32");
    TORCH_CHECK(
        block_table.stride(-1) == 1,
        "page_table must have contiguous last dimension");
  }

  CHECK_DEVICE(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_q);
  TORCH_CHECK(
      cu_seqlens_q.dtype() == torch::kInt32,
      "cu_seqlens_q must have dtype torch.int32");

  CHECK_DEVICE(cu_seqlens_k);
  CHECK_CONTIGUOUS(cu_seqlens_k);
  TORCH_CHECK(
      cu_seqlens_k.dtype() == torch::kInt32,
      "cu_seqlens_k must have dtype torch.int32");

  auto& queue = vllm::xpu::vllmGetQueue(q.device().index());

  at::Tensor out;
  if (out_.has_value()) {
    out = *out_;
  }

  bool is_varlen = true;
  bool is_local = (window_size_left != -1) | (window_size_right != -1);
  bool is_sink = softmax_sink_.has_value();

  if (max_seqlen_q > 1 || !is_paged) {
    if (!out_.has_value()) {
      out = torch::empty_like(q);
    }
    at::Tensor seqlens_k = is_paged ? *seqused_k : cu_seqlens_k;

    cutlass_chunk_prefill_interface(
        queue,
        q,
        k,
        v,
        out,
        block_table,
        cu_seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        k_scale,
        v_scale,
        softmax_scale,
        softmax_sink_,
        window_size_left,
        window_size_right,
        is_varlen,
        is_paged,
        is_causal,
        is_local,
        is_sink);
  } else {
    // Normalize -1 (unbounded) to max_seqlen_k for kernel masking logic
    // In decode phase the window_size_right doesn't have effect
    int eff_window_left =
        window_size_left == -1 ? max_seqlen_k : window_size_left;
    int eff_window_right =
        window_size_right == -1 ? max_seqlen_k : window_size_right;
    int effective_seqlen_k =
        is_local ? std::min(max_seqlen_k, eff_window_left + 1) : max_seqlen_k;

    int num_tokens = q.size(0);
    int batch_size = static_cast<int>(cu_seqlens_q.size(0)) - 1;
    int num_heads_q = q.size(1);
    int v_head_dim = v.size(-1);
    int num_heads_kv = k.size(2);
    int block_size = k.size(1);

    // Output shape uses V's head_dim (may differ from Q/K for MLA)
    if (!out_.has_value()) {
      out = torch::empty(
          {num_tokens, num_heads_q, v_head_dim},
          q.options().device(q.device()));
    }

    int num_kv_splits = num_splits.value_or(get_num_splits(
        queue, batch_size, num_heads_kv, effective_seqlen_k, block_size));

    at::Tensor tmp_out =
        num_kv_splits == 1
            ? out
            : at::empty(
                  {num_tokens, num_heads_q * num_kv_splits, v_head_dim},
                  q.options().device(q.device()));
    at::Tensor max_logits = at::empty(
        {num_tokens, num_heads_q, num_kv_splits},
        q.options().dtype(at::kFloat).device(q.device()));
    at::Tensor exp_sums = at::empty(
        {num_tokens, num_heads_q, num_kv_splits},
        q.options().dtype(at::kFloat).device(q.device()));

    at::Tensor seqlens_k = is_paged ? *seqused_k : cu_seqlens_k;

    // For paged decode (single query per sequence), causal masking is a
    // no-op: seqused_k already constrains KV to only the valid past tokens,
    // so there are no "future" tokens to mask. Passing is_causal=true
    // triggers a seq_len formula that adds +q_sg_tile extra KV positions,
    // causing invalid cache entries to pollute the attention output.
    cutlass_paged_decode_interface(
        queue,
        q,
        k,
        v,
        out,
        tmp_out,
        exp_sums,
        max_logits,
        block_table,
        cu_seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        k_scale,
        v_scale,
        softmax_scale,
        softmax_sink_,
        eff_window_left,
        eff_window_right,
        is_varlen,
        is_paged,
        false,  // is_causal: always false for decode; see comment above
        is_local,
        is_sink,
        num_kv_splits);
  }

  if (return_softmax) {
    // FIXME: current do not support store softmax_lse out
    auto softmax_lse = torch::empty_like(out);
    return {out, softmax_lse};
  } else {
    at::Tensor softmax_lse;
    return {out, softmax_lse};
  }
}
}  // namespace FLASH_NAMESPACE

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "varlen_fwd(Tensor q, Tensor k, Tensor v, Tensor!? out, Tensor "
      "cu_seqlens_q, "
      "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? "
      "block_table, Tensor? alibi_slopes, "
      "int max_seqlen_q, int max_seqlen_k, float p_dropout, Tensor? k_scale, "
      "Tensor? v_scale, "
      "float softmax_scale, Tensor? softmax_sink, bool zero_tensors, "
      "bool is_causal, int window_size_left, int window_size_right, float "
      "softcap, bool return_softmax, "
      "Generator? gen, int? num_splits) -> Tensor[]");
  ops.impl(
      "varlen_fwd",
      torch::kXPU,
      make_pytorch_shim(&FLASH_NAMESPACE::mha_varlen_fwd));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
