#include <sycl/sycl.hpp>
#include <torch/all.h>

void chunk_kda_xe2(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,    // [1, num_tokens, heads, dim]
    const torch::Tensor& q,          // [num_tokens, heads * dim]
    const torch::Tensor& k,          // [num_tokens, heads * dim]
    const torch::Tensor& v,          // [num_tokens, heads * dim]
    const torch::Tensor& raw_gate,   // [1, >=num_tokens, heads, dim]
    const torch::Tensor& beta,       // [1, >=num_tokens, heads]
    torch::Tensor& recurrent_state,  // [slots, heads, dim, dim]
    const torch::Tensor& a_log,      // [1, 1, heads, 1]
    const torch::Tensor& dt_bias,    // [heads * dim]
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::optional<torch::Tensor>& token_indx,
    int64_t batch_size,
    int64_t num_actual_tokens,
    int64_t num_heads,
    int64_t head_dim);

// True when the chunked pipeline supports this shape at all.
bool chunk_kda_xe2_supported(int64_t head_dim);

// Bytes of scratch the chunked pipeline needs for the given problem.
int64_t chunk_kda_xe2_workspace_bytes(
    int64_t batch_size,
    int64_t num_actual_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t element_size);
