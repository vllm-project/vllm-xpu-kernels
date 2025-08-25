#include "core/registration.h"
#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor input, Tensor! output) -> ()");
  m.impl("moe_sum", torch::kXPU, &moe_sum);

  // Grouped TopK
  m.def(
      "grouped_topk(Tensor hidden_states, Tensor gating_output, int n_topk, "
      "bool renormalize, int n_expert_group, int n_topk_group, str "
      "scoring_func, Tensor? bias=None) -> (Tensor, Tensor)");
  m.impl("grouped_topk", torch::kXPU, &grouped_topk);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
