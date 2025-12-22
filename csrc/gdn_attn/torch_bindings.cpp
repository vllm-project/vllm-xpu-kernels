#include "core/registration.h"
#include "ops.h"

#include <torch/library.h>
#include <torch/version.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, xpu_ops) {
  at::Tag stride_tag = at::Tag::needs_fixed_stride_order;

  xpu_ops.def(
      "gdn_attention(Tensor! core_attn_out, Tensor! z, Tensor projected_states_qkvz, Tensor projected_states_ba,"
      "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim, int tp_size) -> ()");
  xpu_ops.impl("gdn_attention", torch::kXPU, &gdn_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
