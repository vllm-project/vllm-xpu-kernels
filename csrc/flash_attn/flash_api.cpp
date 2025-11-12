#include "pytorch_shim.h"

#include "core/registration.h"
#include "xpu/cutlass_kernels/chunk_prefill.hpp"
#include "utils.h"
#include <torch/all.h>

namespace FLASH_NAMESPACE {

std::vector<at::Tensor> mha_varlen_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    std::optional<at::Tensor>& out_,
    const at::Tensor& cu_seqlens_q,  // b+1
    const at::Tensor& cu_seqlens_k,  // b+1
    std::optional<at::Tensor>& seqused_k,
    std::optional<const at::Tensor>& leftpad_k_,  // batch_size
    at::Tensor& block_table_,  // batch_size x max_num_blocks_per_seq
    std::optional<at::Tensor>& alibi_slopes_,  // num_heads or b x num_heads
    int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale,
    std::optional<const at::Tensor>& softmax_sink_, const bool zero_tensors,
    bool is_causal, int window_size_left, int window_size_right,
    const float softcap, const bool return_softmax,
    std::optional<at::Generator> gen_) {
  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
      "VLLM Kernel XPU only supports fp16 and bf16 type");

  TORCH_CHECK(k.scalar_type() == q_type,
              "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type,
              "query and value must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  CHECK_DEVICE(block_table_);
  TORCH_CHECK(block_table_.dtype() == torch::kInt32,
              "page_table must have dtype torch.int32");
  TORCH_CHECK(block_table_.stride(-1) == 1,
              "page_table must have contiguous last dimension");

  TORCH_CHECK(q.dim() == 3, "query must be in ragged format");
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_q);
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32,
              "cu_seqlens_q must have dtype torch.int32");

  CHECK_DEVICE(cu_seqlens_k);
  CHECK_CONTIGUOUS(cu_seqlens_k);
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32,
              "cu_seqlens_k must have dtype torch.int32");

  auto& queue = vllm::xpu::vllmGetQueue(q.device().index());

  at::Tensor out;
  if (out_.has_value()) {
    out = *out_;
  } else {
    out = torch::empty_like(q);
  }

  bool is_local = (window_size_left != -1) | (window_size_right != -1);
  bool is_sink = softmax_sink_.has_value();

  cutlass_chunk_prefill_impl(queue, q, k, v, out, block_table_, cu_seqlens_q,
                             cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                             softmax_scale, softmax_sink_, window_size_left,
                             window_size_right, is_causal, is_local, is_sink);

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
      "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor "
      "block_table, Tensor? alibi_slopes, "
      "int max_seqlen_q, int max_seqlen_k, float p_dropout, float "
      "softmax_scale, Tensor? softmax_sink, bool zero_tensors, "
      "bool is_causal, int window_size_left, int window_size_right, float "
      "softcap, bool return_softmax, "
      "Generator? gen) -> Tensor[]");
  ops.impl("varlen_fwd", torch::kXPU,
           make_pytorch_shim(&FLASH_NAMESPACE::mha_varlen_fwd));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)