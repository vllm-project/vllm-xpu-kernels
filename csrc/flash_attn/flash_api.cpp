#include "pytorch_shim.h"

#include "core/registration.h"
#include "xpu/cutlass_kernels/chunk_prefill.hpp"
#include "utils.h"
#include <torch/all.h>

namespace FLASH_NAMESPACE {

std::vector<at::Tensor> mha_varlen_fwd(
    const at::Tensor& q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& k,  // total_k x num_heads_k x head_size, total_k :=
                          // \sum_{i=0}^{b} s_i or num_blocks x page_block_size
                          // x num_heads_k x head_size if there's a block_table.
    const at::Tensor& v,  // total_k x num_heads_k x head_size, total_k :=
                          // \sum_{i=0}^{b} s_i or num_blocks x page_block_size
                          // x num_heads_k x head_size if there's a block_table.
    std::optional<at::Tensor>& out_,  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q,  // b+1
    const at::Tensor& cu_seqlens_k,  // b+1
    std::optional<at::Tensor>& seqused_k,  // b. If given, only this many elements of each batch
                    // element's keys are used.
    std::optional<const at::Tensor>& leftpad_k_,  // batch_size
    at::Tensor& block_table_,  // batch_size x max_num_blocks_per_seq
    std::optional<at::Tensor>& alibi_slopes_,  // num_heads or b x num_heads
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax, std::optional<at::Generator> gen_) {
  auto& queue = vllm::xpu::vllmGetQueue();

  at::Tensor out;
  if(out_.has_value()) {
    out = *out_;
  }
  else {
    out = torch::zeros_like(q);
  }

  cutlass_chunk_prefill_impl(
      queue,
      q,
      k,
      v,
      out,
      block_table_,
      cu_seqlens_q,
      cu_seqlens_k,
      max_seqlen_q,
      max_seqlen_k,
      softmax_scale,
      is_causal);

  if(return_softmax) {
    // FIXME: current do not support store softmax_lse out
    auto softmax_lse = torch::empty_like(out);
    return {out, softmax_lse};
  }
  else {
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
      "softmax_scale, bool zero_tensors, "
      "bool is_causal, int window_size_left, int window_size_right, float "
      "softcap, bool return_softmax, "
      "Generator? gen) -> Tensor[]");
  ops.impl("varlen_fwd", torch::kXPU,
           make_pytorch_shim(&FLASH_NAMESPACE::mha_varlen_fwd));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)