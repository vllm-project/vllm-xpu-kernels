#include "moe_ops.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

#include "../utils.h"
#include "../dispatch_utils.h"
#include <sycl/ext/oneapi/bfloat16.hpp>

using namespace sycl::ext::intel::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;


void moe_scatter_dynamic_quant(
    torch::Tensor& selected_experts,
    torch::Tensor& moe_weights,
    torch::Tensor& token_to_scatter_offset,
    torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start,
    torch::Tensor& hidden_states,
    torch::Tensor& experts_smooth_scale,
    torch::Tensor& scatter_tokens,
    torch::Tensor& scatter_per_token_scale,
    torch::Tensor& scatter_tokens_offset,
    int64_t shared_experts_num) {

    auto& queue = vllm::xpu::vllmGetQueue();

    int topk = selected_experts.size(1);
    int n_tokens = selected_experts.size(0);
    int n_expert = experts_token_count.size(0);
    int hd_size = hidden_states.size(1);
    int n_expert_total = n_expert + shared_experts_num;

    auto selected_experts_ptr = selected_experts.data_ptr<int32_t>();
    auto ext_tokens_cnt_ptr = experts_token_count.data_ptr<int32_t>();
    auto ext_tokens_start_ptr = experts_token_start.data_ptr<int32_t>();
    auto token_to_scatter_offset_ptr = token_to_scatter_offset.data_ptr<int32_t>();

    // Pass 1: Global memory atomic token counting
    auto e1 = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n_tokens), [=](sycl::item<1> item) {
            int token_idx = item.get_id(0);
            for (int k = 0; k < topk; ++k) {
                int expert_id = selected_experts_ptr[token_idx * topk + k];
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>
                    atomic_cnt(ext_tokens_cnt_ptr[expert_id]);
                token_to_scatter_offset_ptr[token_idx * topk + k] = atomic_cnt.fetch_add(1);
            }
        });
    });

    // Pass 2: Prefix sum for expert start indices
    auto e2 = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(e1);
        cgh.single_task([=]() {
            int32_t sum = 0;
            for(int i = 0; i < n_expert_total; ++i) {
                ext_tokens_start_ptr[i] = sum;
                sum += ext_tokens_cnt_ptr[i];
            }
        });
    });

    auto hidden_states_ptr = reinterpret_cast<bf16*>(hidden_states.data_ptr<at::BFloat16>());
    auto smooth_scale_ptr = experts_smooth_scale.data_ptr<float>();
    auto moe_weights_ptr = moe_weights.data_ptr<float>();
    auto scatter_tokens_ptr = scatter_tokens.data_ptr<int8_t>();
    auto scatter_per_token_scale_ptr = scatter_per_token_scale.data_ptr<float>();
    auto scatter_tokens_offset_ptr = scatter_tokens_offset.data_ptr<int32_t>();

    // Pass 3: Gather, quantize, and scatter
    auto launch_step3 = [&](auto unroll_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr int CHUNK = 64;
        constexpr int BS = CHUNK * UNROLL;

        int num_blocks = hd_size / BS;
        int wg_size = std::min(num_blocks, 64);

        queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(e2);
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_tokens * topk, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init(64 * sizeof(float));

                const int loc_id = item.get_local_id(1);

                if (loc_id == 0) {
                    simd<float, 64> zeros = 0.0f;
                    slm_block_store<float, 64>(0, zeros);
                }
                barrier();

                const int token_k_idx = item.get_group(0);
                const int token_idx = token_k_idx / topk;

                const int expert_id = selected_experts_ptr[token_k_idx];
                const int offset = token_to_scatter_offset_ptr[token_k_idx];
                const int expert_start = ext_tokens_start_ptr[expert_id];

                const int target_idx = expert_start + offset;
                const float weight = moe_weights_ptr[token_k_idx];

                float thread_max = 0.0f;

                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<bf16, CHUNK> hidden = block_load<bf16, CHUNK>(
                            hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK, properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>});
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK, properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>});

                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;
                        simd<float, CHUNK> smoothed_abs = sycl::ext::intel::esimd::abs(smoothed);

                        float max_val = hmax<float, float, CHUNK>(smoothed_abs);
                        thread_max = std::max(thread_max, max_val);
                    }
                }

                slm_block_store<float, 1>(loc_id * sizeof(float), thread_max);
                barrier();

                simd<float, 64> max_value_full = slm_block_load<float, 64>(0);
                float max_value_final = hmax<float, float, 64>(max_value_full);
                float this_token_scale = max_value_final / 127.0f;
                float recip_scale = this_token_scale == 0.0f ? 1.0f : (1.0f / this_token_scale);

                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<bf16, CHUNK> hidden = block_load<bf16, CHUNK>(
                            hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK, properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>});
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK, properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>});

                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;
                        simd<int8_t, CHUNK> quantized = rnde<float>(smoothed * recip_scale);

                        block_store<int8_t, CHUNK>(
                            scatter_tokens_ptr + target_idx * hd_size + hd_bid * BS + u * CHUNK, quantized, properties{cache_hint_L1<cache_hint::write_back>, cache_hint_L2<cache_hint::write_back>});
                    }
                }

                if (loc_id == 0) {
                    block_store<float, 1>(scatter_per_token_scale_ptr + target_idx, simd<float, 1>(this_token_scale), properties{cache_hint_L1<cache_hint::write_back>, cache_hint_L2<cache_hint::write_back>});
                    block_store<int32_t, 1>(scatter_tokens_offset_ptr + target_idx, simd<int32_t, 1>(token_idx), properties{cache_hint_L1<cache_hint::write_back>, cache_hint_L2<cache_hint::write_back>});
                }
            });
        });
    };

    // Dispatch based on hidden size divisibility mapping to max stable unroll bounds
    if (hd_size % 512 == 0) {
        launch_step3(std::integral_constant<int, 8>{});
    } else if (hd_size % 256 == 0) {
        launch_step3(std::integral_constant<int, 4>{});
    } else if (hd_size % 128 == 0) {
        launch_step3(std::integral_constant<int, 2>{});
    } else {
        launch_step3(std::integral_constant<int, 1>{});
    }
}
