#include <torch/all.h>
#include <ATen/xpu/XPUContext.h>

#include "utils.hpp"
#include "flash.h"

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/numeric_types.h"

#include <type_traits>

namespace vllm::xpu::attn::mla {

std::vector<at::Tensor> flash_mla_sparse_attn_prefill_impl(
    const at::Tensor &q,        // [s_q, h_q, d_qk]
    const at::Tensor &kv,       // [s_kv, h_kv, d_qk]
    const at::Tensor &indices,  // [s_q, h_kv, topk]
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink, // [h_q]
    const std::optional<at::Tensor> &topk_length, // [s_q]
    const std::optional<at::Tensor> &output, // [s_q, h_q, d_v]
    bool return_softmax_lse
) {
    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);
    bool have_topk_length = topk_length.has_value();
    TORCH_CHECK(q.dim() == 3, "q must be 3D");
    TORCH_CHECK(kv.dim() == 3, "kv must be 3D");
    TORCH_CHECK(indices.dim() == 3, "indices must be 3D");

    if (attn_sink.has_value()) {
        TORCH_CHECK(attn_sink.value().dim() == 1, "attn_sink must be 1D");
    }
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length.value().dim() == 1, "topk_length must be 1D");
    }

    TORCH_CHECK(d_qk == 576 || d_qk == 512, "d_qk must be 576 or 512 ");
    TORCH_CHECK(d_v == 512, "d_v must be 512");

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.dtype() == torch::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);
    if (attn_sink.has_value()) {
        TORCH_CHECK(attn_sink.value().dtype() == torch::kFloat32, "attn_sink must be float32");
        CHECK_DEVICE(attn_sink.value());
    }
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length.value().dtype() == torch::kInt32, "topk_length must be int32");
        CHECK_DEVICE(topk_length.value());
    }

#if !defined(VLLM_XPU_ENABLE_XE2)
    TORCH_CHECK(false, "Sparse MLA prefill requires VLLM_XPU_ENABLE_XE2 build option.");
#endif

    // Allocate results and buffers
    const c10::DeviceGuard device_guard(q.device());
    auto opts = q.options();

    at::Tensor out = output.has_value() ? output.value() : torch::empty({s_q, h_q, d_v}, opts);
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor gathered_k = torch::empty({s_q, topk, d_qk}, opts);
    at::Tensor gathered_valid_mask = torch::empty({s_q, topk}, opts.dtype(torch::kInt));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    TORCH_CHECK(hw_info.sm_count > 0, "Failed to query device multiprocessor count");

    XPUSparseAttnFwdParams params {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        (cutlass::bfloat16_t*)q.data_ptr(),
        (cutlass::bfloat16_t*)kv.data_ptr(),
        (int*)indices.data_ptr(),

        get_optional_tensor_ptr<float>(attn_sink),
        get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (cutlass::bfloat16_t*)gathered_k.data_ptr(),
        (int*)gathered_valid_mask.data_ptr(),

        int64_stride_to_int(gathered_k.stride(0)), int64_stride_to_int(gathered_k.stride(1)),
        int64_stride_to_int(gathered_valid_mask.stride(0)),

        (cutlass::bfloat16_t*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),
        return_softmax_lse,

        hw_info.sm_count,
        at::xpu::getCurrentXPUStream().queue()
    };

    auto launch_prefill = [&](auto head_dim_tag, auto topk_tag, auto sink_tag) {
        constexpr int HEAD_DIM_QK = decltype(head_dim_tag)::value;
        constexpr bool HAVE_TOPK_LENGTH = decltype(topk_tag)::value;
        constexpr bool HAS_ATTN_SINK = decltype(sink_tag)::value;
        launch_sparse_mla_prefill_fwd_kernel<HEAD_DIM_QK, HAVE_TOPK_LENGTH, HAS_ATTN_SINK>(params);
    };

    if (params.d_qk == 512) {
        if (have_topk_length) {
            if (attn_sink.has_value()) {
#if defined(VLLM_SPARSE_MLA_PREFILL_TOPKLEN_K512_SINK)
                launch_prefill(std::integral_constant<int, 512>{}, std::true_type{}, std::true_type{});
#else
                TORCH_CHECK(false,
                            "Sparse MLA prefill kernel not compiled for configuration: "
                            "prefill,512,true,true\n"
                            "Add this line to your sparse MLA config file:\n"
                            "  prefill,512,true,true");
#endif
            } else {
#if defined(VLLM_SPARSE_MLA_PREFILL_TOPKLEN_K512_NOSINK)
                launch_prefill(std::integral_constant<int, 512>{}, std::true_type{}, std::false_type{});
#else
                TORCH_CHECK(false,
                            "Sparse MLA prefill kernel not compiled for configuration: "
                            "prefill,512,true,false\n"
                            "Add this line to your sparse MLA config file:\n"
                            "  prefill,512,true,false");
#endif
            }
        } else {
            if (attn_sink.has_value()) {
#if defined(VLLM_SPARSE_MLA_PREFILL_K512_SINK)
                launch_prefill(std::integral_constant<int, 512>{}, std::false_type{}, std::true_type{});
#else
                TORCH_CHECK(false,
                            "Sparse MLA prefill kernel not compiled for configuration: "
                            "prefill,512,false,true\n"
                            "Add this line to your sparse MLA config file:\n"
                            "  prefill,512,false,true");
#endif
            } else {
#if defined(VLLM_SPARSE_MLA_PREFILL_K512_NOSINK)
                launch_prefill(std::integral_constant<int, 512>{}, std::false_type{}, std::false_type{});
#else
                TORCH_CHECK(false,
                            "Sparse MLA prefill kernel not compiled for configuration: "
                            "prefill,512,false,false\n"
                            "Add this line to your sparse MLA config file:\n"
                            "  prefill,512,false,false");
#endif
            }
        }
    } else if (params.d_qk == 576) {
        if (have_topk_length) {
            if (attn_sink.has_value()) {
#if defined(VLLM_SPARSE_MLA_PREFILL_TOPKLEN_K576_SINK)
                launch_prefill(std::integral_constant<int, 576>{}, std::true_type{}, std::true_type{});
#else
                TORCH_CHECK(false,
                            "Sparse MLA prefill kernel not compiled for configuration: "
                            "prefill,576,true,true\n"
                            "Add this line to your sparse MLA config file:\n"
                            "  prefill,576,true,true");
#endif
            } else {
#if defined(VLLM_SPARSE_MLA_PREFILL_TOPKLEN_K576_NOSINK)
                launch_prefill(std::integral_constant<int, 576>{}, std::true_type{}, std::false_type{});
#else
                TORCH_CHECK(false,
                            "Sparse MLA prefill kernel not compiled for configuration: "
                            "prefill,576,true,false\n"
                            "Add this line to your sparse MLA config file:\n"
                            "  prefill,576,true,false");
#endif
            }
        } else {
            if (attn_sink.has_value()) {
#if defined(VLLM_SPARSE_MLA_PREFILL_K576_SINK)
                launch_prefill(std::integral_constant<int, 576>{}, std::false_type{}, std::true_type{});
#else
                TORCH_CHECK(false,
                            "Sparse MLA prefill kernel not compiled for configuration: "
                            "prefill,576,false,true\n"
                            "Add this line to your sparse MLA config file:\n"
                            "  prefill,576,false,true");
#endif
            } else {
#if defined(VLLM_SPARSE_MLA_PREFILL_K576_NOSINK)
                launch_prefill(std::integral_constant<int, 576>{}, std::false_type{}, std::false_type{});
#else
                TORCH_CHECK(false,
                            "Sparse MLA prefill kernel not compiled for configuration: "
                            "prefill,576,false,false\n"
                            "Add this line to your sparse MLA config file:\n"
                            "  prefill,576,false,false");
#endif
            }
        }
    } else {
        TORCH_CHECK(false, "Unsupported sparse MLA prefill head dim: ", params.d_qk);
    }
    if (return_softmax_lse) {
        return std::vector<at::Tensor>{out, max_logits, lse};
    }
    return std::vector<at::Tensor>{out};
}

std::tuple<at::Tensor, std::optional<at::Tensor>>
flash_mla_sparse_attn_decode_impl(
    const at::Tensor &q,   // [b, s_q, h_q, d_qk]
    const at::Tensor &kv,   // [num_blocks, page_block_size, h_k, d_qk]
    const at::Tensor &indices,    // [b, s_q, topk]
    const std::optional<at::Tensor> &topk_length,   // [b]
    const std::optional<at::Tensor> &attn_sink, // [h_q]
    const std::optional<at::Tensor> &out, // [b, s_q, h_q, d_v]
    std::optional<at::Tensor> &tile_scheduler_metadata,   // num_sm_parts x (DecodingSchedMetaSize/4)
    std::optional<at::Tensor> &num_splits,                // batch_size + 1
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int d_v,
    float sm_scale,
    bool return_softmax_lse
) {
    TORCH_CHECK(q.dim() == 4, "q must be 4D [b, s_q, h_q, d_qk]");
    TORCH_CHECK(kv.dim() == 4, "kv must be 4D [num_blocks, page_block_size, h_kv, head_bytes]");
    TORCH_CHECK(indices.dim() == 3, "indices must be 3D [b, s_q, topk]");

    int b = q.size(0);
    int s_q = q.size(1);
    int h_q = q.size(2);
    int d_qk = q.size(3);
    int num_blocks = kv.size(0);
    int page_block_size = kv.size(1);
    int h_kv = kv.size(2);
    int head_bytes = kv.size(3);
    int topk = indices.size(2);
    bool have_topk_length = topk_length.has_value();

    TORCH_CHECK(indices.size(0) == b && indices.size(1) == s_q, "indices batch/query dims must match q");
    TORCH_CHECK(h_kv == 1, "sparse MLA decode currently requires h_kv == 1");
    TORCH_CHECK(d_qk == 512, "packed fp8 sparse MLA decode requires q head dim 512");
    TORCH_CHECK(d_v == 512, "d_v must be 512");
    TORCH_CHECK(head_bytes == SPARSE_MLA_FP8_HEAD_BYTES, "kv last dim must be packed head_bytes=584");
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(kv.dtype() == at::ScalarType::Float8_e4m3fn, "kv must be float8_e4m3fn packed storage");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "sparse MLA decode requires q dtype bfloat16");

    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length.value().dim() == 1, "topk_length must be 1D [b]");
        TORCH_CHECK(topk_length.value().size(0) == b, "topk_length shape must match [b]");
        TORCH_CHECK(topk_length.value().dtype() == torch::kInt32, "topk_length must be int32");
        CHECK_DEVICE(topk_length.value());
    }
    if (attn_sink.has_value()) {
        TORCH_CHECK(attn_sink.value().dim() == 1 && attn_sink.value().size(0) == h_q, "attn_sink must have shape [h_q]");
        TORCH_CHECK(attn_sink.value().dtype() == torch::kFloat32, "attn_sink must be float32");
        CHECK_DEVICE(attn_sink.value());
    }
    if (tile_scheduler_metadata.has_value()) {
        TORCH_CHECK(false, "sparse fp8 decode splitKV scheduler metadata is not supported in the no-split XPU implementation yet");
    }
    if (num_splits.has_value()) {
        TORCH_CHECK(false, "sparse fp8 decode splitKV num_splits is not supported in the no-split XPU implementation yet");
    }

    int extra_num_blocks = 0;
    int extra_page_block_size = 0;
    int extra_topk = 0;
    if (extra_kv.has_value()) {
        TORCH_CHECK(extra_indices.has_value(), "extra_indices must be provided when extra_kv is provided");
        TORCH_CHECK(extra_kv.value().dim() == 4, "extra_kv must be 4D [num_blocks, page_block_size, h_kv, head_bytes]");
        TORCH_CHECK(extra_kv.value().size(2) == 1 && extra_kv.value().size(3) == SPARSE_MLA_FP8_HEAD_BYTES,
                    "extra_kv must have h_kv=1 and head_bytes=584");
        TORCH_CHECK(extra_kv.value().dtype() == at::ScalarType::Float8_e4m3fn, "extra_kv must be float8_e4m3fn packed storage");
        CHECK_DEVICE(extra_kv.value());
        TORCH_CHECK(extra_indices.value().dim() == 3, "extra_indices must be 3D [b, s_q, extra_topk]");
        TORCH_CHECK(extra_indices.value().size(0) == b && extra_indices.value().size(1) == s_q, "extra_indices batch/query dims must match q");
        TORCH_CHECK(extra_indices.value().dtype() == torch::kInt32, "extra_indices must be int32");
        CHECK_DEVICE(extra_indices.value());
        extra_num_blocks = extra_kv.value().size(0);
        extra_page_block_size = extra_kv.value().size(1);
        extra_topk = extra_indices.value().size(2);
    } else {
        TORCH_CHECK(!extra_indices.has_value(), "extra_indices requires extra_kv");
    }
    if (extra_topk_length.has_value()) {
        TORCH_CHECK(extra_kv.has_value(), "extra_topk_length requires extra_kv");
        TORCH_CHECK(extra_topk_length.value().dim() == 1, "extra_topk_length must be 1D [b]");
        TORCH_CHECK(extra_topk_length.value().size(0) == b, "extra_topk_length shape must match [b]");
        TORCH_CHECK(extra_topk_length.value().dtype() == torch::kInt32, "extra_topk_length must be int32");
        CHECK_DEVICE(extra_topk_length.value());
    }

#if !defined(VLLM_XPU_ENABLE_XE2)
    TORCH_CHECK(false, "Sparse MLA decode requires VLLM_XPU_ENABLE_XE2 build option.");
#endif

    const c10::DeviceGuard device_guard(q.device());
    auto out_opts = q.options().dtype(torch::kBFloat16);
    at::Tensor out_tensor;
    if (out.has_value()) {
        TORCH_CHECK(out.value().sizes() == torch::IntArrayRef({b, s_q, h_q, d_v}), "out must have shape [b, s_q, h_q, d_v]");
        TORCH_CHECK(out.value().dtype() == torch::kBFloat16, "out must be bfloat16");
        CHECK_DEVICE(out.value());
        out_tensor = out.value();
    } else {
        out_tensor = torch::empty({b, s_q, h_q, d_v}, out_opts);
    }
    at::Tensor lse = torch::empty({b, s_q, h_q}, q.options().dtype(torch::kFloat));
    const int gathered_topk = topk + extra_topk;
    at::Tensor gathered_k = torch::empty({b, s_q, gathered_topk, d_qk}, out_opts);
    at::Tensor gathered_valid_mask = torch::empty({b, s_q, gathered_topk}, q.options().dtype(torch::kInt32));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    XPUSparseDecodeAttnFwdParams params;
    params.b = b;
    params.s_q = s_q;
    params.h_q = h_q;
    params.h_kv = h_kv;
    params.d_qk = d_qk;
    params.d_v = d_v;
    params.sm_scale = sm_scale;
    params.sm_scale_div_log2 = sm_scale * LOG_2_E;
    params.num_blocks = num_blocks;
    params.page_block_size = page_block_size;
    params.topk = topk;
    params.gathered_topk = gathered_topk;
    params.q = q.data_ptr();
    params.kv = reinterpret_cast<uint8_t*>(kv.data_ptr());
    params.indices = reinterpret_cast<int*>(indices.data_ptr());
    params.topk_length = get_optional_tensor_ptr<int>(topk_length);
    params.attn_sink = get_optional_tensor_ptr<float>(attn_sink);
    params.gathered_k = reinterpret_cast<cutlass::bfloat16_t*>(gathered_k.data_ptr());
    params.gathered_valid_mask = reinterpret_cast<int*>(gathered_valid_mask.data_ptr());
    params.lse = reinterpret_cast<float*>(lse.data_ptr());
    params.out = reinterpret_cast<cutlass::bfloat16_t*>(out_tensor.data_ptr());
    params.extra_num_blocks = extra_num_blocks;
    params.extra_page_block_size = extra_page_block_size;
    params.extra_topk = extra_topk;
    params.extra_kv = extra_kv.has_value() ? reinterpret_cast<uint8_t*>(extra_kv.value().data_ptr()) : nullptr;
    params.extra_indices = extra_indices.has_value() ? reinterpret_cast<int*>(extra_indices.value().data_ptr()) : nullptr;
    params.extra_topk_length = get_optional_tensor_ptr<int>(extra_topk_length);
    params.stride_q_b = int64_stride_to_int(q.stride(0));
    params.stride_q_s_q = int64_stride_to_int(q.stride(1));
    params.stride_q_h_q = int64_stride_to_int(q.stride(2));
    params.stride_kv_block = int64_stride_to_int(kv.stride(0));
    params.stride_kv_row = int64_stride_to_int(kv.stride(1));
    params.stride_kv_head = int64_stride_to_int(kv.stride(2));
    params.stride_indices_b = int64_stride_to_int(indices.stride(0));
    params.stride_indices_s_q = int64_stride_to_int(indices.stride(1));
    params.stride_topk_length_b = topk_length.has_value() ? int64_stride_to_int(topk_length.value().stride(0)) : 0;
    params.stride_topk_length_s_q = 0;
    params.stride_gathered_k_b = int64_stride_to_int(gathered_k.stride(0));
    params.stride_gathered_k_s_q = int64_stride_to_int(gathered_k.stride(1));
    params.stride_gathered_k_topk = int64_stride_to_int(gathered_k.stride(2));
    params.stride_gathered_mask_b = int64_stride_to_int(gathered_valid_mask.stride(0));
    params.stride_gathered_mask_s_q = int64_stride_to_int(gathered_valid_mask.stride(1));
    params.stride_lse_b = int64_stride_to_int(lse.stride(0));
    params.stride_lse_s_q = int64_stride_to_int(lse.stride(1));
    params.stride_o_b = int64_stride_to_int(out_tensor.stride(0));
    params.stride_o_s_q = int64_stride_to_int(out_tensor.stride(1));
    params.stride_o_h_q = int64_stride_to_int(out_tensor.stride(2));
    params.stride_extra_kv_block = extra_kv.has_value() ? int64_stride_to_int(extra_kv.value().stride(0)) : 0;
    params.stride_extra_kv_row = extra_kv.has_value() ? int64_stride_to_int(extra_kv.value().stride(1)) : 0;
    params.stride_extra_kv_head = extra_kv.has_value() ? int64_stride_to_int(extra_kv.value().stride(2)) : 0;
    params.stride_extra_indices_b = extra_indices.has_value() ? int64_stride_to_int(extra_indices.value().stride(0)) : 0;
    params.stride_extra_indices_s_q = extra_indices.has_value() ? int64_stride_to_int(extra_indices.value().stride(1)) : 0;
    params.stride_extra_topk_length_b = extra_topk_length.has_value() ? int64_stride_to_int(extra_topk_length.value().stride(0)) : 0;
    params.stride_extra_topk_length_s_q = 0;
    params.lse_accum = nullptr;
    params.o_accum = nullptr;
    params.tile_scheduler_metadata_ptr = nullptr;
    params.num_splits_ptr = nullptr;
    params.num_sm_parts = 0;
    params.num_sm = hw_info.sm_count;
    params.queue = at::xpu::getCurrentXPUStream().queue();

        auto launch_decode = [&](auto topk_tag, auto sink_tag) {
        constexpr bool HAVE_TOPK_LENGTH = decltype(topk_tag)::value;
        constexpr bool HAS_ATTN_SINK = decltype(sink_tag)::value;
        launch_sparse_mla_decode_fp8_fwd_kernel<512, HAVE_TOPK_LENGTH, HAS_ATTN_SINK>(params);
    };

    if (have_topk_length) {
        if (attn_sink.has_value()) {
    #if defined(VLLM_SPARSE_MLA_DECODE_FP8_TOPKLEN_K512_SINK)
            launch_decode(std::true_type{}, std::true_type{});
    #else
            TORCH_CHECK(false,
                "Sparse MLA decode fp8 kernel not compiled for configuration: "
                "decode_fp8,512,true,true\n"
                "Add this line to your sparse MLA config file:\n"
                "  decode_fp8,512,true,true");
    #endif
        } else {
    #if defined(VLLM_SPARSE_MLA_DECODE_FP8_TOPKLEN_K512_NOSINK)
            launch_decode(std::true_type{}, std::false_type{});
    #else
            TORCH_CHECK(false,
                "Sparse MLA decode fp8 kernel not compiled for configuration: "
                "decode_fp8,512,true,false\n"
                "Add this line to your sparse MLA config file:\n"
                "  decode_fp8,512,true,false");
    #endif
        }
    } else {
        if (attn_sink.has_value()) {
    #if defined(VLLM_SPARSE_MLA_DECODE_FP8_K512_SINK)
            launch_decode(std::false_type{}, std::true_type{});
    #else
            TORCH_CHECK(false,
                "Sparse MLA decode fp8 kernel not compiled for configuration: "
                "decode_fp8,512,false,true\n"
                "Add this line to your sparse MLA config file:\n"
                "  decode_fp8,512,false,true");
    #endif
        } else {
    #if defined(VLLM_SPARSE_MLA_DECODE_FP8_K512_NOSINK)
            launch_decode(std::false_type{}, std::false_type{});
    #else
            TORCH_CHECK(false,
                "Sparse MLA decode fp8 kernel not compiled for configuration: "
                "decode_fp8,512,false,false\n"
                "Add this line to your sparse MLA config file:\n"
                "  decode_fp8,512,false,false");
    #endif
        }
    }

    if (return_softmax_lse) {
        return {out_tensor, std::optional<at::Tensor>(lse)};
    }
    return {out_tensor, std::nullopt};
}

} // namespace vllm::xpu::attn::mla
