#include "pytorch_shim.h"

#include "core/registration.h"
#include "xpu/attn/attn_interface.h"
#include "utils.h"
#include <torch/all.h>

namespace FLASH_NAMESPACE {

inline int get_num_splits(
    const int& batch_size,
    const int& num_heads_kv,
    const int& max_seqlen_k,
    const int& block_size) {
  constexpr int parallel_ = 20;
  constexpr int parallel_2 = 40;

  int cur_parallel_d = batch_size * num_heads_kv;

  int num_splits = (parallel_ + cur_parallel_d - 1) / cur_parallel_d;

  if (cur_parallel_d * num_splits > parallel_ && num_splits > 1) {
    num_splits = std::ceil(parallel_2 / static_cast<float>(cur_parallel_d)) - 1;
  }

  int max_splits = (max_seqlen_k + block_size - 1) / block_size;
  max_splits = std::min(max_splits, parallel_);
  return std::min(num_splits, max_splits);
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
  bool is_fp8kv = false;
  if (k_type == at::ScalarType::Float8_e5m2 ||
      k_type == at::ScalarType::Float8_e4m3fn) {
    is_fp8kv = true;
  } else {
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
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(v);

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
  } else {
    out = torch::empty_like(q);
  }

  bool is_varlen = true;
  bool is_local = (window_size_left != -1) | (window_size_right != -1);
  bool is_sink = softmax_sink_.has_value();

  if (max_seqlen_q > 1 || is_local || !is_paged || is_fp8kv) {
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
    int num_tokens = q.size(0);
    int batch_size = static_cast<int>(cu_seqlens_q.size(0)) - 1;
    int num_heads_q = q.size(1);
    int head_dim = q.size(2);
    int num_heads_kv = k.size(2);
    int block_size = k.size(1);

    int num_kv_splits = num_splits.value_or(
        get_num_splits(batch_size, num_heads_kv, max_seqlen_k, block_size));

    at::Tensor tmp_out =
        num_kv_splits == 1
            ? out
            : at::empty(
                  {num_tokens, num_heads_q * num_kv_splits, head_dim},
                  q.options().device(q.device()));
    at::Tensor max_logits = at::empty(
        {num_tokens, num_heads_q, num_kv_splits},
        q.options().dtype(at::kFloat).device(q.device()));
    at::Tensor exp_sums = at::empty(
        {num_tokens, num_heads_q, num_kv_splits},
        q.options().dtype(at::kFloat).device(q.device()));

    at::Tensor seqlens_k = is_paged ? *seqused_k : cu_seqlens_k;

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
        softmax_scale,
        softmax_sink_,
        window_size_left,
        window_size_right,
        is_varlen,
        is_paged,
        is_causal,
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
