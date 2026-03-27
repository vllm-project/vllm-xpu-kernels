# SPDX-License-Identifier: Apache-2.0
"""Kernel-to-test mapping for model-aware UT suites.

The mapping is organized around the existing unit-test suites in ``tests/`` and
then refined down to specific pytest node ids where a suite already exposes
clear per-kernel coverage.

Design principles:
- Prefer file/test-function granularity over broad CI buckets.
- Keep a few aggregate aliases for compatibility with higher-level model maps.
- Split MoE routing variants (for example fused vs grouped top-k), since a
  model typically uses only one routing implementation.
"""

KERNEL_TO_TESTS: dict[str, list[str]] = {
    # ---------------------------------------------------------------------
    # LayerNorm suite: tests/test_layernorm.py
    # ---------------------------------------------------------------------
    "all_layernorm": [
        "tests/test_layernorm.py",
    ],
    "rms_norm": [
        "tests/test_layernorm.py::test_rms_norm",
    ],
    "rms_norm_uncontiguous": [
        "tests/test_layernorm.py::test_rms_norm_uncontigous",
    ],

    # ---------------------------------------------------------------------
    # Activation suites: tests/test_activation.py, tests/test_swigluoai...
    # ---------------------------------------------------------------------
    "all_activation": [
        "tests/test_activation.py",
        "tests/test_swigluoai_and_mul.py",
    ],
    "act_and_mul": [
        "tests/test_activation.py::test_act_and_mul",
    ],
    "gelu_family": [
        "tests/test_activation.py::test_activation",
    ],
    "swigluoai_and_mul": [
        "tests/test_swigluoai_and_mul.py::test_act_and_mul",
    ],

    # ---------------------------------------------------------------------
    # Rope / positional encoding suites
    # ---------------------------------------------------------------------
    "rotary_embedding": [
        "tests/test_rotary_embedding.py::test_rotary_embedding_opcheck",
    ],
    "all_deepseek_scaling_rope": [
        "tests/test_deepseek_scaling_rope.py",
    ],

    # ---------------------------------------------------------------------
    # Cache / KV-cache suites: tests/test_cache.py
    # ---------------------------------------------------------------------
    "all_cache": [
        "tests/test_cache.py",
    ],
    "reshape_and_cache": [
        "tests/test_cache.py::test_reshape_and_cache",
    ],
    "reshape_and_cache_flash": [
        "tests/test_cache.py::test_reshape_and_cache_flash",
    ],
    "concat_and_cache_mla": [
        "tests/test_cache.py::test_concat_and_cache_mla",
    ],
    "gather_cache_mla": [
        "tests/test_cache.py::test_gather_cache_mla",
    ],
    "swap_blocks": [
        "tests/test_cache.py::test_swap_blocks",
    ],
    "swap_blocks_mla": [
        "tests/test_cache.py::test_swap_blocks_mla",
    ],

    # ---------------------------------------------------------------------
    # Attention suites
    # ---------------------------------------------------------------------
    "all_flash_attn": [
        "tests/flash_attn/",
    ],
    "all_gdn_attn": [
        "tests/gdn_attn/",
    ],

    # ---------------------------------------------------------------------
    # LoRA suite
    # ---------------------------------------------------------------------
    "all_lora": [
        "tests/test_lora_ops.py",
    ],
    "lora_default": [
        "tests/test_lora_ops.py::test_kernels",
    ],
    "lora_hidden_size": [
        "tests/test_lora_ops.py::test_kernels_hidden_size",
    ],

    # ---------------------------------------------------------------------
    # Quantization / GEMM suites
    # ---------------------------------------------------------------------
    "all_fp8_quant": [
        "tests/test_fp8_quant.py",
    ],
    "fp8_quant_dynamic_per_tensor": [
        "tests/test_fp8_quant.py::test_dynamic_per_tensor_fp8_quant",
    ],
    "fp8_quant_dynamic_per_token": [
        "tests/test_fp8_quant.py::test_dynamic_per_token_fp8_quant",
    ],
    "fp8_quant_per_block": [
        "tests/test_fp8_quant.py::test_per_block_fp8_quant",
    ],
    "mxfp8_quant": [
        "tests/test_fp8_quant.py::test_per_block_mxfp8_quant",
    ],
    "fp8_quant_large": [
        "tests/test_fp8_quant.py::test_fp8_quant_large",
    ],
    "fp8_quant_static_group_2d": [
        "tests/test_fp8_quant.py::test_static_fp8_quant_group_2d",
    ],
    "fp8_quant_static_1d_scale": [
        "tests/test_fp8_quant.py::test_static_fp8_quant_1d_scale",
    ],
    "all_fp8_gemm_onednn": [
        "tests/test_fp8_gemm_onednn.py",
    ],
    "fp8_gemm_w8a16": [
        "tests/test_fp8_gemm_onednn.py::test_fp8_gemm_w8a16",
    ],
    "fp8_gemm_per_tensor": [
        "tests/test_fp8_gemm_onednn.py::test_fp8_gemm_per_tensor",
    ],
    "fp8_gemm_per_channel": [
        "tests/test_fp8_gemm_onednn.py::test_fp8_gemm_per_channel",
    ],
    "all_mxfp4_quant": [
        "tests/test_mxfp4_quant.py",
    ],
    "all_int4_gemm_onednn": [
        "tests/test_int4_gemm_onednn.py",
    ],
    "all_indexer_k_quant_and_cache": [
        "tests/test_indexer_k_quant_and_cache.py",
    ],

    # ---------------------------------------------------------------------
    # MoE routing top-k suites
    # ---------------------------------------------------------------------
    "all_moe_topk_fused": [
        "tests/test_topk.py",
    ],
    "moe_topk_fused_softmax": [
        "tests/test_topk.py::test_fused_topk_softmax",
    ],
    "moe_topk_fused_sigmoid": [
        "tests/test_topk.py::test_fused_topk_sigmoid",
    ],
    "all_moe_topk_grouped": [
        "tests/test_grouped_topk.py",
    ],

    # ---------------------------------------------------------------------
    # Decoder / sampling top-k suite
    # ---------------------------------------------------------------------
    "all_topk_per_row": [
        "tests/test_topk_per_row.py",
    ],

    # ---------------------------------------------------------------------
    # Fused MoE compute suites
    # ---------------------------------------------------------------------
    "all_fused_moe": [
        "tests/fused_moe/test_fused_moe.py",
        "tests/fused_moe/test_moe_prologue.py",
        "tests/fused_moe/test_grouped_gemm.py",
        "tests/fused_moe/test_remap_hidden_states.py",
    ],
    "fused_moe_main": [
        "tests/fused_moe/test_fused_moe.py::test_fused_moe",
    ],
    "fused_moe_int4": [
        "tests/fused_moe/test_fused_moe.py::test_fused_moe_int4",
    ],
    "fused_moe_mxfp4_a16": [
        "tests/fused_moe/test_fused_moe.py::test_fused_moe_mxfp4",
    ],
    "fused_moe_ep": [
        "tests/fused_moe/test_fused_moe.py::test_fused_moe_ep",
    ],
    "fused_moe_int4_ep": [
        "tests/fused_moe/test_fused_moe.py::test_fused_moe_int4_ep",
    ],
    "fused_moe_mxfp4_ep": [
        "tests/fused_moe/test_fused_moe.py::test_fused_moe_mxfp4_ep",
    ],
    "fused_moe_prologue": [
        "tests/fused_moe/test_moe_prologue.py::test_prologue",
    ],
    "all_fused_moe_grouped_gemm": [
        "tests/fused_moe/test_grouped_gemm.py",
        "tests/fused_moe/test_grouped_gemm.py::test_grouped_gemm",
    ],
    "fused_moe_grouped_gemm_xe": [
        "tests/fused_moe/test_grouped_gemm.py::test_xe_grouped_gemm",
    ],
    "fused_moe_grouped_gemm_fp8": [
        "tests/fused_moe/test_grouped_gemm.py::test_xe_grouped_gemm_fp8",
    ],
    "fused_moe_grouped_gemm_int4": [
        "tests/fused_moe/test_grouped_gemm.py::test_xe_grouped_gemm_int4",
    ],
    "fused_moe_grouped_gemm_mxfp4": [
        "tests/fused_moe/test_grouped_gemm.py::test_xe_grouped_gemm_mxfp4",
    ],
    "fused_moe_remap_hidden_states": [
        "tests/fused_moe/test_remap_hidden_states.py::test_remap_hidden_states",
    ],
    "fused_moe_init_expert_map": [
        "tests/fused_moe/test_remap_hidden_states.py::test_init_expert_map",
    ],

    # ---------------------------------------------------------------------
    # Other MoE helper suites
    # ---------------------------------------------------------------------
    "all_moe_gather": [
        "tests/test_moe_gather.py",
    ],
    "moe_gather_main": [
        "tests/test_moe_gather.py::test_moe_gather",
    ],
    "all_moe_align_block_size": [
        "tests/test_moe_align_block_size.py",
    ],
    "moe_align_block_size_main": [
        "tests/test_moe_align_block_size.py::test_moe_align_block_size",
    ],
    "moe_align_block_size_expert_map": [
        "tests/test_moe_align_block_size.py::test_moe_align_block_size_with_expert_map",
    ],
    "moe_align_block_size_deterministic": [
        "tests/test_moe_align_block_size.py::test_moe_align_block_size_deterministic",
    ],
    "batched_moe_align_block_size": [
        "tests/test_moe_align_block_size.py::test_batched_moe_align_block_size",
    ],
    "moe_align_block_size_opcheck": [
        "tests/test_moe_align_block_size.py::test_moe_align_block_size_opcheck",
    ],
    "batched_moe_align_block_size_opcheck": [
        "tests/test_moe_align_block_size.py::test_batched_moe_align_block_size_opcheck",
    ],
    "all_moe_lora_align_sum": [
        "tests/test_moe_lora_align_sum.py",
    ],
    "moe_lora_align_sum_main": [
        "tests/test_moe_lora_align_sum.py::test_moe_lora_align_block_size",
    ],
    "all_moe_sum": [
        "tests/test_moe_sum.py",
    ],
    "moe_sum_main": [
        "tests/test_moe_sum.py::test_moe_sum",
    ],

    # ---------------------------------------------------------------------
    # Utility suites
    # ---------------------------------------------------------------------
    "all_mem_alloc": [
        "tests/test_mem_alloc.py",
    ],
    "all_uva": [
        "tests/test_uva.py",
    ],
    "all_xpu_memcpy_sync": [
        "tests/test_xpu_memcpy_sync.py",
    ],
}
