#include "chunk_prefill.hpp"

using namespace cute;

template <
    typename chunk_policy,
    typename ElementQ,
    typename ElementKV,
    typename ElementO,
    bool... Bs>
void policy_dispatch_bool_func(
    sycl::queue& queue, const chunk_prefill_args_t& args) {
  policy_dispatch_impl<chunk_policy, ElementQ, ElementKV, ElementO, Bs...>(
      queue, args);
}

template <
    typename chunk_policy,
    typename ElementQ,
    typename ElementKV,
    typename ElementO,
    bool... Bs,
    typename... Ts>
void policy_dispatch_bool_func(
    sycl::queue& queue, const chunk_prefill_args_t& args, bool b, Ts... ts) {
  if (b) {
    policy_dispatch_bool_func<
        chunk_policy,
        ElementQ,
        ElementKV,
        ElementO,
        Bs...,
        true>(queue, args, ts...);
  } else {
    policy_dispatch_bool_func<
        chunk_policy,
        ElementQ,
        ElementKV,
        ElementO,
        Bs...,
        false>(queue, args, ts...);
  }
}

template <typename chunk_policy, typename ElementQ, typename ElementO>
void policy_dispatch_kv_func(
    sycl::queue& queue,
    const chunk_prefill_args_t& args,
    CutlassDType q_type,
    CutlassDType kv_type,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink) {
  if (q_type == CutlassDType::float8_e4m3) {
    TORCH_CHECK(
        kv_type == CutlassDType::float8_e4m3,
        "When Q is float8_e4m3, KV must also be float8_e4m3");
    return policy_dispatch_bool_func<
        chunk_policy,
        ElementQ,
        float_e4m3_t,
        ElementO>(queue, args, is_paged, is_causal, is_local, is_sink);
  }

  if (q_type == CutlassDType::float8_e5m2) {
    TORCH_CHECK(
        kv_type == CutlassDType::float8_e5m2,
        "When Q is float8_e5m2, KV must also be float8_e5m2");
    return policy_dispatch_bool_func<
        chunk_policy,
        ElementQ,
        float_e5m2_t,
        ElementO>(queue, args, is_paged, is_causal, is_local, is_sink);
  }

  if (q_type == CutlassDType::half) {
    if (kv_type == CutlassDType::half) {
      return policy_dispatch_bool_func<
          chunk_policy,
          ElementQ,
          half_t,
          ElementO>(queue, args, is_paged, is_causal, is_local, is_sink);
    }
    if (kv_type == CutlassDType::float8_e4m3) {
      return policy_dispatch_bool_func<
          chunk_policy,
          ElementQ,
          float_e4m3_t,
          ElementO>(queue, args, is_paged, is_causal, is_local, is_sink);
    }
    if (kv_type == CutlassDType::float8_e5m2) {
      return policy_dispatch_bool_func<
          chunk_policy,
          ElementQ,
          float_e5m2_t,
          ElementO>(queue, args, is_paged, is_causal, is_local, is_sink);
    }
    TORCH_CHECK(
        false, "When Q is half, KV must be half/float8_e4m3/float8_e5m2");
  }

  if (q_type == CutlassDType::bfloat16) {
    if (kv_type == CutlassDType::bfloat16) {
      return policy_dispatch_bool_func<
          chunk_policy,
          ElementQ,
          bfloat16_t,
          ElementO>(queue, args, is_paged, is_causal, is_local, is_sink);
    }
    if (kv_type == CutlassDType::float8_e4m3) {
      return policy_dispatch_bool_func<
          chunk_policy,
          ElementQ,
          float_e4m3_t,
          ElementO>(queue, args, is_paged, is_causal, is_local, is_sink);
    }
    if (kv_type == CutlassDType::float8_e5m2) {
      return policy_dispatch_bool_func<
          chunk_policy,
          ElementQ,
          float_e5m2_t,
          ElementO>(queue, args, is_paged, is_causal, is_local, is_sink);
    }
    TORCH_CHECK(
        false,
        "When Q is bfloat16, KV must be bfloat16/float8_e4m3/float8_e5m2");
  }

  TORCH_CHECK(false, "Unsupported query dtype for chunk prefill dispatch");
}

template <typename chunk_policy, typename ElementO>
void policy_dispatch_q_func(
    sycl::queue& queue,
    const chunk_prefill_args_t& args,
    CutlassDType q_type,
    CutlassDType kv_type,
    CutlassDType o_type,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink) {
  if (q_type == CutlassDType::half) {
    TORCH_CHECK(
        o_type == CutlassDType::half, "When Q is half, O must also be half");
    return policy_dispatch_kv_func<chunk_policy, half_t, ElementO>(
        queue, args, q_type, kv_type, is_paged, is_causal, is_local, is_sink);
  }

  if (q_type == CutlassDType::bfloat16) {
    TORCH_CHECK(
        o_type == CutlassDType::bfloat16,
        "When Q is bfloat16, O must also be bfloat16");
    return policy_dispatch_kv_func<chunk_policy, bfloat16_t, ElementO>(
        queue, args, q_type, kv_type, is_paged, is_causal, is_local, is_sink);
  }

  if (q_type == CutlassDType::float8_e4m3) {
    return policy_dispatch_kv_func<chunk_policy, float_e4m3_t, ElementO>(
        queue, args, q_type, kv_type, is_paged, is_causal, is_local, is_sink);
  }

  if (q_type == CutlassDType::float8_e5m2) {
    return policy_dispatch_kv_func<chunk_policy, float_e5m2_t, ElementO>(
        queue, args, q_type, kv_type, is_paged, is_causal, is_local, is_sink);
  }

  TORCH_CHECK(false, "Unsupported query dtype for chunk prefill dispatch");
}

template <typename chunk_policy>
void policy_dispatch_func(
    sycl::queue& queue,
    const chunk_prefill_args_t& args,
    CutlassDType q_type,
    CutlassDType kv_type,
    CutlassDType o_type,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink) {
  if (o_type == CutlassDType::half) {
    return policy_dispatch_q_func<chunk_policy, half_t>(
        queue,
        args,
        q_type,
        kv_type,
        o_type,
        is_paged,
        is_causal,
        is_local,
        is_sink);
  } else if (o_type == CutlassDType::bfloat16) {
    return policy_dispatch_q_func<chunk_policy, bfloat16_t>(
        queue,
        args,
        q_type,
        kv_type,
        o_type,
        is_paged,
        is_causal,
        is_local,
        is_sink);
  } else {
    TORCH_CHECK(false, "Unsupported output dtype for chunk prefill dispatch");
  }
}

void cutlass_chunk_prefill_impl(
    sycl::queue& queue,
    const at::Tensor& query,      // [seq_q, heads, head_size]
    const at::Tensor& key_cache,  // [num_block, block_size, heads, head_size]
    const at::Tensor& value_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& q_scale,
    std::optional<const at::Tensor>& k_scale,
    std::optional<const at::Tensor>& v_scale,
    double sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink);
