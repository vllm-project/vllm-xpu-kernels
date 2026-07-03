#include "fused_moe_impl.hpp"

template <
    bool HAS_W13_BIAS,
    bool HAS_W2_BIAS,
    bool HAS_CLAMP_LIMIT = false,
    bool IS_DECODE = false,
    typename DTYPE = bf16_t,
    int TM = (IS_DECODE ? 1 : ::TM),
    int N_TM = (IS_DECODE ? 1 : ::N_TM),
    int N_TK = ::N_TK,
    int N_TN = ::N_TN>
static void launch_fused_moe_kernel(
    sycl::queue& q,
    const DTYPE* tokens,
    const DTYPE* w13,
    const DTYPE* w2,
    const float* w13_bias,
    const float* w2_bias,
    DTYPE* output,
    int H,
    int I,
    int num_experts,
    float* intermediate,
    int32_t* row_counter,
    int K,
    const int64_t* expert_offset,
    int total_tokens,
    const int32_t* source_row,
    const float* topk_weights,
    int num_tokens,
    const int32_t* topk_ids,
    float gemm1_clamp_limit = 0.0f) {
  using Kernel = FusedMoe<
      HAS_W13_BIAS,
      HAS_W2_BIAS,
      HAS_CLAMP_LIMIT,
      IS_DECODE,
      DTYPE,
      TM,
      N_TM,
      N_TK,
      N_TN>;

  sycl::nd_range<2> nd;
  if constexpr (IS_DECODE)
    nd = Kernel::get_nd_range(0, 0, K);
  else
    nd = Kernel::get_nd_range(total_tokens, num_experts);

  q.submit([&](sycl::handler& cgh) {
    int slm_count;
    slm_count = SG_COUNT * N_TM * TM * TK;
    sycl::local_accessor<DTYPE, 1> slm_acc(slm_count, cgh);

    cgh.parallel_for(
        nd,
        Kernel{
            tokens,
            w13,
            w2,
            w13_bias,
            w2_bias,
            output,
            H,
            I,
            intermediate,
            row_counter,
            K,
            slm_acc,
            expert_offset,
            num_experts,
            total_tokens,
            source_row,
            topk_weights,
            num_tokens,
            topk_ids,
            gemm1_clamp_limit});
  });
}

static void sort_topk_ids(
    sycl::queue& q,
    const int32_t* topk_ids_ptr,
    const float* topk_weights_ptr,
    int64_t* exp_off_ptr,
    int32_t* sorted_tok_ptr,
    float* sorted_weights_ptr,
    int num_slots,
    int num_experts,
    int num_per_tok,
    int32_t* sorted_slot_ptr = nullptr) {
  static constexpr int COUNT_WG = 256;
  static constexpr int WARP_SIZE = 16;
  static constexpr int MAX_LOCAL = 32;

  auto slot_rel = torch::empty(
      num_slots,
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kXPU));
  int32_t* slot_rel_ptr = slot_rel.data_ptr<int32_t>();

  // Phase 1 — count tokens per expert; record within-expert rank
  {
    int groups = (num_slots + COUNT_WG - 1) / COUNT_WG;
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(groups * COUNT_WG), sycl::range<1>(COUNT_WG)),
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            int gid = (int)it.get_global_linear_id();
            if (gid >= num_slots) return;
            int32_t expert_id = topk_ids_ptr[gid];
            sycl::atomic_ref<
                int64_t,
                sycl::memory_order_relaxed,
                sycl::memory_scope_device,
                sycl::access::address_space::global_space>
                counter(exp_off_ptr[expert_id + 1]);
            int64_t old = counter.fetch_add(int64_t(1));
            slot_rel_ptr[gid] = (int32_t)old;
          });
    });
  }
  // Phase 2 — warp-parallel exclusive prefix scan → expert start offsets
  {
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(WARP_SIZE), sycl::range<1>(WARP_SIZE)),
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            auto sg = it.get_sub_group();
            int sg_local_id = (int)sg.get_local_linear_id();

            int elems_per_item = (num_experts + WARP_SIZE - 1) / WARP_SIZE;
            int local_start = elems_per_item * sg_local_id;
            int remained = num_experts - local_start;
            int local_elems =
                (remained > elems_per_item) ? elems_per_item : remained;

            int64_t local_storage[MAX_LOCAL];
            int64_t local_sum = 0;
            if (remained > 0) {
              for (int i = 0; i < local_elems; ++i) {
                local_storage[i] = exp_off_ptr[local_start + i + 1];
                local_sum += local_storage[i];
              }
            }
            int64_t global_sum = sycl::exclusive_scan_over_group(
                sg, local_sum, sycl::plus<int64_t>());
            if (remained > 0) {
              for (int i = 0; i < local_elems; ++i) {
                global_sum += local_storage[i];
                exp_off_ptr[local_start + i + 1] = global_sum;
              }
            }
          });
    });
  }

  // Phase 3 — scatter token indices + weights to their sorted positions
  {
    int groups = (num_slots + COUNT_WG - 1) / COUNT_WG;
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(groups * COUNT_WG), sycl::range<1>(COUNT_WG)),
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            int gid = (int)it.get_global_linear_id();
            if (gid >= num_slots) return;
            int32_t expert_id = topk_ids_ptr[gid];
            int64_t dest = (int64_t)slot_rel_ptr[gid] + exp_off_ptr[expert_id];
            sorted_tok_ptr[dest] = (int32_t)(gid / num_per_tok);
            sorted_weights_ptr[dest] = topk_weights_ptr[gid];
            if (sorted_slot_ptr) sorted_slot_ptr[dest] = gid;
          });
    });
  }
}

torch::Tensor fused_moe(
    torch::Tensor input_tokens,
    torch::Tensor w13,
    const c10::optional<torch::Tensor>& w13_bias_opt,
    torch::Tensor w2,
    const c10::optional<torch::Tensor>& w2_bias_opt,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    const c10::optional<torch::Tensor>& output_opt,
    int64_t num_experts,
    int64_t inter_size,
    int64_t hidden_size,
    double gemm1_clamp_limit = 0.0) {
  TORCH_CHECK(
      input_tokens.dtype() == torch::kBFloat16 ||
          input_tokens.dtype() == torch::kFloat16,
      "fused_moe: input_tokens must be bfloat16 or float16");

  const int H = (int)hidden_size;
  const int I = (int)inter_size;
  const int E = (int)num_experts;
  const int NT = (int)input_tokens.size(0);
  const int K = (int)topk_ids.size(1);

  torch::Tensor w13_bias_f32, w2_bias_f32;
  const float* w13_bias_ptr = nullptr;
  const float* w2_bias_ptr = nullptr;
  if (w13_bias_opt.has_value()) {
    w13_bias_f32 = w13_bias_opt.value().to(torch::kFloat32).contiguous();
    w13_bias_ptr = w13_bias_f32.data_ptr<float>();
  }
  if (w2_bias_opt.has_value()) {
    w2_bias_f32 = w2_bias_opt.value().to(torch::kFloat32).contiguous();
    w2_bias_ptr = w2_bias_f32.data_ptr<float>();
  }

  sycl::queue& q =
      at::xpu::getCurrentXPUStream(input_tokens.device().index()).queue();

  auto topk_ids_i32 = topk_ids.to(torch::kInt32).contiguous().reshape(-1);
  auto topk_wt_flat = topk_weights.to(torch::kFloat32).contiguous().reshape(-1);

  torch::Tensor out;
  if (output_opt.has_value())
    out = output_opt.value();
  else
    out = torch::zeros_like(input_tokens);

  auto intermediate = torch::zeros(
      {(int64_t)NT * K, (int64_t)H},
      torch::TensorOptions()
          .dtype(torch::kFloat32)
          .device(input_tokens.device()));
  auto row_counter = torch::zeros(
      {(int64_t)NT},
      torch::TensorOptions()
          .dtype(torch::kInt32)
          .device(input_tokens.device()));

  const bool has_w13 = (w13_bias_ptr != nullptr);
  const bool has_w2 = (w2_bias_ptr != nullptr);

  auto run = [&](auto dtype) {
    using DTYPE = decltype(dtype);
    constexpr bool is_bf16 = std::is_same_v<DTYPE, bf16_t>;
    auto raw_ptr = [&](auto& t) -> DTYPE* {
      if constexpr (is_bf16)
        return reinterpret_cast<DTYPE*>(t.template data_ptr<at::BFloat16>());
      else
        return reinterpret_cast<DTYPE*>(t.template data_ptr<at::Half>());
    };
    auto raw_cptr = [&](const auto& t) -> const DTYPE* {
      if constexpr (is_bf16)
        return reinterpret_cast<const DTYPE*>(
            t.template data_ptr<at::BFloat16>());
      else
        return reinterpret_cast<const DTYPE*>(t.template data_ptr<at::Half>());
    };

    if (NT == 1) {
      auto dispatch =
          [&](auto w13_bias_flag, auto w2_bias_flag, auto clamp_flag) {
            launch_fused_moe_kernel<
                decltype(w13_bias_flag)::value,
                decltype(w2_bias_flag)::value,
                decltype(clamp_flag)::value,
                /*IS_DECODE=*/true,
                DTYPE>(
                q,
                raw_cptr(input_tokens),
                raw_cptr(w13),
                raw_cptr(w2),
                w13_bias_ptr,
                w2_bias_ptr,
                raw_ptr(out),
                H,
                I,
                E,
                intermediate.data_ptr<float>(),
                row_counter.data_ptr<int32_t>(),
                K,
                /*expert_offset=*/nullptr,
                /*total_tokens=*/0,
                /*source_row=*/nullptr,
                /*topk_weights=*/topk_wt_flat.data_ptr<float>(),
                /*num_tokens=*/NT,
                /*topk_ids=*/topk_ids_i32.data_ptr<int32_t>(),
                /*gemm1_clamp_limit=*/(float)gemm1_clamp_limit);
          };

      auto dispatch_bias = [&](auto clamp_flag) {
        if (!has_w13 && !has_w2)
          dispatch(std::false_type{}, std::false_type{}, clamp_flag);
        else if (has_w13 && !has_w2)
          dispatch(std::true_type{}, std::false_type{}, clamp_flag);
        else if (!has_w13 && has_w2)
          dispatch(std::false_type{}, std::true_type{}, clamp_flag);
        else
          dispatch(std::true_type{}, std::true_type{}, clamp_flag);
      };
      if (gemm1_clamp_limit > 0.0)
        dispatch_bias(std::true_type{});
      else
        dispatch_bias(std::false_type{});
    } else {
      const int T = NT * K;
      auto expert_offset_t = torch::zeros(
          E + 1,
          torch::TensorOptions()
              .dtype(torch::kInt64)
              .device(input_tokens.device()));
      auto sorted_source_row = torch::empty(
          T,
          torch::TensorOptions()
              .dtype(torch::kInt32)
              .device(input_tokens.device()));
      auto sorted_weights_t = torch::empty(
          T,
          torch::TensorOptions()
              .dtype(torch::kFloat32)
              .device(input_tokens.device()));
      auto sorted_slot_ids = torch::empty(
          T,
          torch::TensorOptions()
              .dtype(torch::kInt32)
              .device(input_tokens.device()));

      sort_topk_ids(
          q,
          topk_ids_i32.data_ptr<int32_t>(),
          topk_wt_flat.data_ptr<float>(),
          expert_offset_t.data_ptr<int64_t>(),
          sorted_source_row.data_ptr<int32_t>(),
          sorted_weights_t.data_ptr<float>(),
          T,
          E,
          K,
          sorted_slot_ids.data_ptr<int32_t>());

      auto dispatch =
          [&](auto w13_bias_flag, auto w2_bias_flag, auto clamp_flag) {
            auto prefill_launch = [&](auto tm_tag) {
              constexpr int TM_val = decltype(tm_tag)::value;
              launch_fused_moe_kernel<
                  decltype(w13_bias_flag)::value,
                  decltype(w2_bias_flag)::value,
                  decltype(clamp_flag)::value,
                  /*IS_DECODE=*/false,
                  DTYPE,
                  TM_val>(
                  q,
                  raw_cptr(input_tokens),
                  raw_cptr(w13),
                  raw_cptr(w2),
                  w13_bias_ptr,
                  w2_bias_ptr,
                  raw_ptr(out),
                  H,
                  I,
                  E,
                  intermediate.data_ptr<float>(),
                  row_counter.data_ptr<int32_t>(),
                  K,
                  expert_offset_t.data_ptr<int64_t>(),
                  T,
                  sorted_source_row.data_ptr<int32_t>(),
                  sorted_weights_t.data_ptr<float>(),
                  NT,
                  sorted_slot_ids.data_ptr<int32_t>(),
                  /*gemm1_clamp_limit=*/(float)gemm1_clamp_limit);
            };
            if (K == 1)
              prefill_launch(std::integral_constant<int, 1>{});
            else if (K == 2)
              prefill_launch(std::integral_constant<int, 2>{});
            else if (K <= 4)
              prefill_launch(std::integral_constant<int, 4>{});
            else
              prefill_launch(std::integral_constant<int, ::TM>{});
          };

      auto dispatch_bias = [&](auto clamp_flag) {
        if (!has_w13 && !has_w2)
          dispatch(std::false_type{}, std::false_type{}, clamp_flag);
        else if (has_w13 && !has_w2)
          dispatch(std::true_type{}, std::false_type{}, clamp_flag);
        else if (!has_w13 && has_w2)
          dispatch(std::false_type{}, std::true_type{}, clamp_flag);
        else
          dispatch(std::true_type{}, std::true_type{}, clamp_flag);
      };
      if (gemm1_clamp_limit > 0.0)
        dispatch_bias(std::true_type{});
      else
        dispatch_bias(std::false_type{});
    }
  };

  if (input_tokens.dtype() == torch::kBFloat16)
    run(bf16_t{});
  else
    run(fp16_t{});

  return out;
}
