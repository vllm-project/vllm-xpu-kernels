# SPDX-License-Identifier: Apache-2.0
"""Model-to-kernel mapping for model-aware UT suites.

These mappings are intentionally approximate seed profiles. They are meant to
bootstrap a model-focused suite and can be refined over time with runtime trace
information.

Profiles are additive. For example:
- default: model's normal forward path
"""

MODEL_TO_KERNELS: dict[str, dict[str, list[str]]] = {
    "llama3-bf16": {
        "default": [
            "rms_norm",
            "rotary_embedding",
            "reshape_and_cache_flash",
            "all_flash_attn",
            "act_and_mul",
        ],
    },
    "llama3-fp8": {
        "default": [
            "rms_norm",
            "rotary_embedding",
            "reshape_and_cache_flash",
            "all_flash_attn",
            "act_and_mul",
            "fp8_quant_dynamic_per_tensor",
        ]
    },
    "llama3-mxfp8": {
        "default": [
            "rms_norm",
            "rotary_embedding",
            "reshape_and_cache_flash",
            "all_flash_attn",
            "act_and_mul",
            "mxfp8_quant",
        ]
    },
    "llama4-bf16": {
        "default": [
            "rms_norm",
            "rotary_embedding",
            "reshape_and_cache_flash",
            "all_flash_attn",
            "act_and_mul",
            #llama4 use fast topk
            "fused_moe_prologue",
            "fused_moe_remap_hidden_states",
            "fused_moe_init_expert_map",
            "fused_moe_mxfp4_a16",
            "fused_moe_grouped_gemm_xe",
            "fused_moe_grouped_gemm_mxfp4",
        ],
    },
    "llama4-mxfp4": {
        "default": [
            "rms_norm",
            "rotary_embedding",
            "reshape_and_cache_flash",
            "all_flash_attn",
            "act_and_mul",
            "all_mxfp4_quant",
            #llama4 use fast topk
            "fused_moe_prologue",
            "fused_moe_remap_hidden_states",
            "fused_moe_init_expert_map",
            "fused_moe_mxfp4_a16",  # todo fix
            "fused_moe_grouped_gemm_xe",
            "fused_moe_grouped_gemm_mxfp4",  # todo fix
        ],
    }
}
