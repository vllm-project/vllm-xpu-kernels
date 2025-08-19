#include "core/registration.h"
#include "moe_ops.h"

#include <torch/library.h>
#include <torch/version.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Grouped TopK
  m.def(
      "grouped_topk(Tensor hidden_states, Tensor gating_output, int n_topk, "
      "bool renormalize, int n_expert_group, int n_topk_group, str "
      "scoring_func, Tensor? bias=None) -> (Tensor, Tensor)");
  m.impl("grouped_topk", torch::kXPU, &grouped_topk);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
