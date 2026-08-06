#include <sycl/sycl.hpp>
#include <torch/all.h>

// Returns false when the launch was abandoned because the chunk decay range
// was exceeded; the caller must then run the recurrent backend. Always true
// when `guard_decay_range` is false.
bool chunk_kda_xe2(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,    // [1, num_tokens, heads, dim]
    const torch::Tensor& q,          // [num_tokens, heads * dim]
    const torch::Tensor& k,          // [num_tokens, heads * dim]
    const torch::Tensor& v,          // [num_tokens, heads * dim]
    const torch::Tensor& raw_gate,   // [1, >=num_tokens, heads, dim]
    const torch::Tensor& raw_beta,   // [1, >=num_tokens, heads]
    torch::Tensor& recurrent_state,  // [slots, heads, dim, dim]
    const torch::Tensor& a_log,      // [1, 1, heads, 1]
    const torch::Tensor& dt_bias,    // [heads * dim]
    // < 0 selects the bounded sigmoid gate, 0 the unbounded softplus gate.
    const float lower_bound,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::optional<torch::Tensor>& token_indx,
    int64_t batch_size,
    int64_t num_actual_tokens,
    int64_t num_heads,
    int64_t head_dim,
    // When set, the pipeline synchronizes after its first stage and, if the
    // per-chunk cumulative log-decay hit the clamp, abandons the launch
    // without touching `core_attn_out` or `recurrent_state`. Callers should
    // then run the recurrent backend, which has no such range limit. Costs one
    // device synchronization per call, so it is enabled only for the bounded
    // sigmoid gate, whose per-token decay can plausibly reach the clamp.
    bool guard_decay_range,
    // Raise instead of reporting the saturation back to the caller. Used by
    // VLLM_XPU_KDA_CHUNK_STRICT=1 to make the condition visible in tests.
    bool strict_decay_range);

// True when the chunked pipeline supports this shape at all.
bool chunk_kda_xe2_supported(int64_t head_dim);

// Bytes of scratch the chunked pipeline needs for the given problem.
int64_t chunk_kda_xe2_workspace_bytes(
    int64_t batch_size,
    int64_t num_actual_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t element_size);
