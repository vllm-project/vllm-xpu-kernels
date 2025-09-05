#include "core/registration.h"
#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor input, Tensor! output) -> ()");
  m.impl("moe_sum", torch::kXPU, &moe_sum);

  // Apply grouped topk routing to select experts.
  m.def(
      "grouped_topk(Tensor scores, Tensor scores_with_bias, int n_group, int "
      "topk_group, int topk, bool renormalize, float "
      "routed_scaling_factor) -> (Tensor, Tensor)");
  m.impl("grouped_topk", torch::kXPU, &grouped_topk);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
