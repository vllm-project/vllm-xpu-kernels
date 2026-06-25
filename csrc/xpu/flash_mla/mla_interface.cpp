#include "xpu/flash_mla/mla_interface.h"

#ifdef VLLM_SPARSE_MLA_ENABLED

namespace vllm::xpu::attn::mla {

std::vector<at::Tensor> flash_mla_sparse_attn_prefill_impl(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& output,
    bool return_softmax_lse);

std::tuple<at::Tensor, std::optional<at::Tensor>>
flash_mla_sparse_attn_decode_impl(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& out,
    std::optional<at::Tensor>& tile_scheduler_metadata,
    std::optional<at::Tensor>& num_splits,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int d_v,
    float sm_scale,
    bool return_softmax_lse);

}  // namespace vllm::xpu::attn::mla

std::vector<at::Tensor> flash_mla_sparse_attn_prefill_interface(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& output,
    bool return_softmax_lse) {
  return vllm::xpu::attn::mla::flash_mla_sparse_attn_prefill_impl(
      q,
      kv,
      indices,
      sm_scale,
      d_v,
      attn_sink,
      topk_length,
      output,
      return_softmax_lse);
}

std::tuple<at::Tensor, std::optional<at::Tensor>>
flash_mla_sparse_attn_decode_interface(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& out,
    std::optional<at::Tensor>& tile_scheduler_metadata,
    std::optional<at::Tensor>& num_splits,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int d_v,
    float sm_scale,
    bool return_softmax_lse) {
  return vllm::xpu::attn::mla::flash_mla_sparse_attn_decode_impl(
      q,
      kv,
      indices,
      topk_length,
      attn_sink,
      out,
      tile_scheduler_metadata,
      num_splits,
      extra_kv,
      extra_indices,
      extra_topk_length,
      d_v,
      sm_scale,
      return_softmax_lse);
}

#endif
