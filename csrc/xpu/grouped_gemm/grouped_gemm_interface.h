#include <torch/all.h>

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A,
    const c10::optional<at::Tensor>& ptr_A_scale,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts);

void cutlass_batched_gemm_interface(
    torch::Tensor input,
    torch::Tensor weight,
    const c10::optional<at::Tensor>& bias,
    torch::Tensor output,
    torch::Tensor rows_per_expert,
    int64_t tile_m);
