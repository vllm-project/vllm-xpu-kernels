# SPDX-License-Identifier: Apache-2.0
"""
On-demand test scope profiles for model-specific validation.

Each profile maps test file paths (relative, matching suffix) to per-function
parameter overrides. Use ``XPU_KERNEL_TEST_SCOPE=ondemand:<profile>`` to
activate a profile.

Setting a function entry to ``None`` skips that test entirely.
Setting it to ``{}`` runs the test with original (full) parameters.

Example usage:
    XPU_KERNEL_TEST_SCOPE=ondemand:llama3 pytest -v -s tests/

To add a new profile:
    1. Add a new key to ONDEMAND_PROFILES
    2. Map relevant test files → functions → parameter overrides
    3. Only include tests/shapes that the target model actually uses
"""
import torch

# ---------------------------------------------------------------------------
# Llama-family models (Llama-3-70B, CodeLlama, etc.)
#   - SiluAndMul activation, RMSNorm, Rotary Embedding, attention with 64 heads
#     and head size 128
#   - MHA: 64 heads, head_size 128
#   - Hidden: 8192, intermediate:28672 (Llama-3)
#   - FP8/BF16 quantization
# ---------------------------------------------------------------------------
LLAMA3_HEAD_SIZE = 128
LLAMA3_NUM_HEADS = 64
LLAMA3_HIDDEN_SIZE = LLAMA3_HEAD_SIZE * LLAMA3_NUM_HEADS
LLAMA3_INTERMEDIATE_SIZE = 28672
LLAMA3_NUM_KV_HEADS = 8
_LLAMA3_PROFILE = {
    "tests/test_activation.py": {
        "test_act_and_mul": {
            "activation": ["silu_and_mul"],
            "num_tokens": [1, 128, 2048],
            "d": [LLAMA3_INTERMEDIATE_SIZE],
        },
        "test_activation": None,  # Llama doesn't use standalone activations
    },
    "tests/test_layernorm.py": {
        "test_rms_norm": {
            "num_tokens": [1, 128, 2048],
            "hidden_size": [LLAMA3_HIDDEN_SIZE],
        },
    },
    "tests/test_rotary_embedding.py": {
        "test_rotary_embedding_opcheck": {
            "is_neox_style": [True],
            "max_position": [1024],
            "head_size": [LLAMA3_HEAD_SIZE],
            "seq_len": [1, 128, 1024],
        },
    },
    "tests/test_cache.py": {
        "test_reshape_and_cache_flash": {
            "num_tokens": [1, 128],
            "num_heads": [LLAMA3_NUM_HEADS],
            "head_size": [LLAMA3_HEAD_SIZE],
            "block_size": [64],
            "dtype": [torch.bfloat16],
        },
    },
    "tests/test_fp8_quant.py": {
        "test_dynamic_per_tensor_fp8_quant": {
            "num_tokens": [1, 128],
            "hidden_size": [LLAMA3_HIDDEN_SIZE],
        },
        "test_dynamic_per_token_fp8_quant": {
            "num_tokens": [1, 128],
            "hidden_size": [LLAMA3_HIDDEN_SIZE],
        },
    },
    "tests/test_fp8_gemm_onednn.py": {
        "test_fp8_gemm_per_tensor": {
            "mnk_factors":
            [(1, LLAMA3_HIDDEN_SIZE, LLAMA3_INTERMEDIATE_SIZE),
             (128, LLAMA3_HIDDEN_SIZE, LLAMA3_INTERMEDIATE_SIZE)],
        },
        "test_fp8_gemm_per_channel": {
            "mnk_factors":
            [(1, LLAMA3_HIDDEN_SIZE, LLAMA3_INTERMEDIATE_SIZE),
             (128, LLAMA3_HIDDEN_SIZE, LLAMA3_INTERMEDIATE_SIZE)],
        },
    },
    "tests/flash_attn/test_flash_attn_varlen_func.py": {
        "test_varlen_with_paged_kv": {
            "seq_lens": [[(1, 1)]],
            "num_heads": [(LLAMA3_NUM_HEADS, LLAMA3_NUM_KV_HEADS)],
            "head_size": [LLAMA3_HEAD_SIZE],
            "num_blocks": [2048],
            "window_size": [(-1, -1)],
            "is_paged": [True],
        },
        "test_decode_with_paged_kv": {
            "seq_lens": [[(1, 1)]],
            "num_heads": [(LLAMA3_NUM_HEADS, LLAMA3_NUM_KV_HEADS)],
            "head_size": [LLAMA3_HEAD_SIZE],
            "num_blocks": [2048],
            "window_size": [(-1, -1)],
        },
    },
}

LLAMA4_HEAD_SIZE = 128
LLAMA4_NUM_HEADS = 40
LLAMA4_HIDDEN_SIZE = LLAMA4_HEAD_SIZE * LLAMA4_NUM_HEADS  # 5120
LLAMA4_INTERMEDIATE_SIZE = 8192
LLAMA4_NUM_KV_HEADS = 8
LLAMA4_NUM_EXPERTS = 16
LLAMA4_TOPK = 1
_LLAMA4_PROFILE = {
    # ---- Activation: SiluAndMul (SwiGLU) ----
    "tests/test_activation.py": {
        "test_act_and_mul": {
            "activation": ["silu_and_mul"],
            "num_tokens": [1, 128, 2048],
            "d": [LLAMA4_INTERMEDIATE_SIZE],
        },
        "test_activation": None,  # Scout doesn't use standalone activations
    },
    # ---- RMSNorm ----
    "tests/test_layernorm.py": {
        "test_rms_norm": {
            "num_tokens": [1, 128, 2048],
            "hidden_size": [LLAMA4_HIDDEN_SIZE],
        },
    },
    # ---- Rotary Embedding: interleaved style (iRoPE) ----
    "tests/test_rotary_embedding.py": {
        "test_rotary_embedding_opcheck": {
            "is_neox_style": [False],  # Llama 4 uses interleaved RoPE
            "max_position": [1024],
            "head_size": [LLAMA4_HEAD_SIZE],
            "seq_len": [1, 128, 1024],
        },
    },
    # ---- KV Cache: GQA (not MLA) ----
    "tests/test_cache.py": {
        "test_reshape_and_cache_flash": {
            "num_tokens": [1, 128],
            "num_heads": [LLAMA4_NUM_KV_HEADS],
            "head_size": [LLAMA4_HEAD_SIZE],
            "block_size": [64],
            "dtype": [torch.bfloat16],
        },
    },
    # ---- TopK routing: softmax, top-1, 16 experts, use torch.topk, ignore ----
    # ---- Fused MoE: 16 experts, top-1 ----
    "tests/fused_moe/test_fused_moe.py": {
        "test_fused_moe": {
            "m,n,k": [(1, LLAMA4_INTERMEDIATE_SIZE, LLAMA4_HIDDEN_SIZE),
                      (128, LLAMA4_INTERMEDIATE_SIZE, LLAMA4_HIDDEN_SIZE)],
            "e": [LLAMA4_NUM_EXPERTS],
            "topk": [LLAMA4_TOPK],
            "dtype": [torch.bfloat16],  #FIXME: add low precision
            "has_bias": [True, False],
        },
    },
    # ---- Grouped GEMM: 16 experts, top-1 ----
    "tests/fused_moe/test_grouped_gemm.py": {
        "test_grouped_gemm": {
            "m,n,k": [(1, LLAMA4_INTERMEDIATE_SIZE, LLAMA4_HIDDEN_SIZE),
                      (128, LLAMA4_INTERMEDIATE_SIZE, LLAMA4_HIDDEN_SIZE)],
            "e": [LLAMA4_NUM_EXPERTS],
            "topk": [LLAMA4_TOPK],
            "dtype": [torch.bfloat16],  #FIXME: add low precision
            "has_bias": [True, False],
        },
    },
    # ---- MoE prologue ----
    "tests/fused_moe/test_moe_prologue.py": {
        "test_prologue": {
            "m,n,k": [(1, LLAMA4_INTERMEDIATE_SIZE, LLAMA4_HIDDEN_SIZE),
                      (128, LLAMA4_INTERMEDIATE_SIZE, LLAMA4_HIDDEN_SIZE)],
            "e": [LLAMA4_NUM_EXPERTS],
            "topk": [LLAMA4_TOPK],
        },
    },
    # ---- MoE remap hidden states ----
    "tests/fused_moe/test_remap_hidden_states.py": {
        "test_remap_hidden_states": {
            "total_experts_num": [LLAMA4_NUM_EXPERTS],
            "topk": [LLAMA4_TOPK],
        },
    },
    # ---- MoE align block size ----
    "tests/test_moe_align_block_size.py": {
        "test_moe_align_block_size": {
            "m": [1, 128, 2048],
            "num_experts": [LLAMA4_NUM_EXPERTS],
            "topk": [LLAMA4_TOPK],
            "block_size": [128],
        },
    },
    # ---- MoE gather ----
    "tests/test_moe_gather.py": {
        "test_moe_gather": {
            "input_len": [1, 128],
            "hidden_size": [LLAMA4_HIDDEN_SIZE],
            "num_experts": [LLAMA4_NUM_EXPERTS],
            "topk": [LLAMA4_TOPK],
        },
    },
    # ---- MoE sum ----
    "tests/test_moe_sum.py": {
        "test_moe_sum": {
            "m": [1, 128],
            "topk": [1],  # top-1 sum is trivial; use smallest available
            "k": [LLAMA4_HIDDEN_SIZE],
        },
    },
    # ---- Flash Attention: GQA 40q/8kv heads ----
    "tests/flash_attn/test_flash_attn_varlen_func.py": {
        "test_varlen_with_paged_kv": {
            "seq_lens": [[(1, 1328), (5, 18), (129, 463)]],
            "num_heads": [(LLAMA4_NUM_HEADS, LLAMA4_NUM_KV_HEADS)],
            "head_size": [LLAMA4_HEAD_SIZE],
            "num_blocks": [2048],
            "window_size": [(-1, -1)],
            "is_paged": [True],
        },
        "test_decode_with_paged_kv": {
            "seq_lens": [[(1, 1025), (1, 523), (1, 37)]],
            "num_heads": [(LLAMA4_NUM_HEADS, LLAMA4_NUM_KV_HEADS)],
            "head_size": [LLAMA4_HEAD_SIZE],
            "num_blocks": [2048],
            "window_size": [(-1, -1)],
        },
        "test_decode_with_paged_kv_mla": None,  # Not MLA
    },
    # ---- Merge attention states ----
    "tests/test_merge_attn_states.py": {
        "test_merge_attn_states": {
            "num_tokens": [1, 128],
            "num_query_heads": [LLAMA4_NUM_HEADS],
            "head_size": [LLAMA4_HEAD_SIZE],
            "output_dtype": [torch.bfloat16],
        },
    },
    # ---- FP8 quantization ----
    "tests/test_fp8_quant.py": {
        "test_dynamic_per_tensor_fp8_quant": {
            "num_tokens": [1, 128],
            "hidden_size": [LLAMA4_HIDDEN_SIZE],
        },
        "test_dynamic_per_token_fp8_quant": {
            "num_tokens": [1, 128],
            "hidden_size": [LLAMA4_HIDDEN_SIZE],
        },
    },
    # ---- FP8 GEMM ----
    "tests/test_fp8_gemm_onednn.py": {
        "test_fp8_gemm_per_tensor": {
            "mnk_factors": [
                (1, LLAMA4_HIDDEN_SIZE, LLAMA4_INTERMEDIATE_SIZE),
                (128, LLAMA4_HIDDEN_SIZE, LLAMA4_INTERMEDIATE_SIZE),
            ],
        },
        "test_fp8_gemm_per_channel": {
            "mnk_factors": [
                (1, LLAMA4_HIDDEN_SIZE, LLAMA4_INTERMEDIATE_SIZE),
                (128, LLAMA4_HIDDEN_SIZE, LLAMA4_INTERMEDIATE_SIZE),
            ],
        },
    },
}

# ---------------------------------------------------------------------------
# DeepSeek-V3/R1 MLA models
#   - MLA attention (Multi-head Latent Attention)
#   - MoE with grouped topk (256 experts, top-8)
#   - SiluAndMul, RMSNorm
#   - kv_lora_rank=512, qk_rope_head_dim=64, v_head_dim=128
# ---------------------------------------------------------------------------
_DEEPSEEK_PROFILE = {
    "tests/test_activation.py": {
        "test_act_and_mul": {
            "activation": ["silu_and_mul"],
            "num_tokens": [1, 128, 2048],
            "d": [13824],
            "dtype": [torch.bfloat16],
        },
    },
    "tests/test_layernorm.py": {
        "test_rms_norm": {
            "num_tokens": [1, 128, 2048],
            "hidden_size": [7168],
        },
    },
    "tests/test_rotary_embedding.py": {
        "test_rotary_embedding_opcheck": {
            "is_neox_style": [True],
            "max_position": [1024],
            "head_size": [192],
            "seq_len": [1, 128, 1024],
        },
    },
    "tests/test_cache.py": {
        "test_reshape_and_cache_flash": {
            "num_tokens": [1, 128],
            "num_heads": [8],
            "head_size": [128],
            "block_size": [64],
            "dtype": [torch.bfloat16],
        },
        "test_concat_and_cache_mla": {
            "num_tokens": [1, 128],
            "num_blocks": [4],
            "block_size": [64],
        },
    },
    "tests/test_topk.py": {
        "test_fused_topk_softmax": {
            "topk": [8],
            "n_expert": [256],
            "n_token": [1, 128, 2048],
        },
    },
    "tests/test_grouped_topk.py": {
        "test_grouped_topk": {
            "n_hidden": [256],
            "n_token": [1, 128],
            "topk": [8],
            "n_group": [8],
            "renormalize": [True],
            "scoring_func": ["softmax"],
        },
    },
    "tests/fused_moe/test_fused_moe.py": {
        "test_fused_moe": {
            "m,n,k": [(1, 5120, 7168), (128, 5120, 7168)],
            "e": [256],
            "topk": [8],
            "dtype": [torch.bfloat16],
            "has_bias": [True],
        },
    },
    "tests/fused_moe/test_grouped_gemm.py": {
        "test_grouped_gemm": {
            "m,n,k": [(1, 5120, 7168), (128, 5120, 7168)],
            "e": [256],
            "topk": [8],
            "dtype": [torch.bfloat16],
            "has_bias": [True],
        },
    },
    "tests/test_moe_align_block_size.py": {
        "test_moe_align_block_size": {
            "m": [1, 128, 2048],
            "num_experts": [256],
            "topk": [8],
            "block_size": [128],
        },
    },
    "tests/flash_attn/test_flash_attn_varlen_func.py": {
        "test_decode_with_paged_kv_mla": {
            "seq_lens": [[(1, 1025), (1, 523), (1, 37)]],
            "num_heads": [(8, 1)],
            "head_size_kv": [(192, 128)],
            "num_blocks": [2048],
        },
        "test_varlen_with_paged_kv": {
            "seq_lens": [[(1, 1328), (5, 18), (129, 463)]],
            "num_heads": [(8, 1)],
            "head_size": [128],
            "num_blocks": [2048],
            "window_size": [(-1, -1)],
            "is_paged": [True],
        },
    },
    "tests/test_merge_attn_states.py": {
        "test_merge_attn_states": {
            "num_tokens": [1, 128],
            "num_query_heads": [128],
            "head_size": [128],
            "output_dtype": [torch.bfloat16],
        },
    },
    "tests/test_fp8_quant.py": {
        "test_dynamic_per_token_fp8_quant": {
            "num_tokens": [1, 128],
            "hidden_size": [7168],
        },
        "test_per_block_fp8_quant": {
            "num_tokens_block_quant": [1, 128],
            "hidden_size_block_quant": [7168],
        },
    },
    "tests/test_swigluoai_and_mul.py": {
        "test_act_and_mul": {
            "num_tokens": [1, 128, 2048],
            "d": [13824],
            "dtype": [torch.bfloat16],
        },
    },
}

# ---------------------------------------------------------------------------
# Qwen3-30B-A3B (Qwen/Qwen3-30B-A3B-Instruct-2507)
#   - Qwen3MoE architecture, SiluAndMul (SwiGLU), RMSNorm
#   - hidden_size=2048, intermediate_size=6144, moe_intermediate_size=768
#   - 32 attention heads, 4 KV heads, head_dim=128
#   - 128 experts, top-8 routing, norm_topk_prob=True
#   - neox-style RoPE
# ---------------------------------------------------------------------------
QWEN3_30B_HEAD_DIM = 128
QWEN3_30B_NUM_HEADS = 32
QWEN3_30B_HIDDEN_SIZE = 2048
QWEN3_30B_INTERMEDIATE_SIZE = 6144        # dense FFN intermediate
QWEN3_30B_MOE_INTERMEDIATE_SIZE = 768     # per-expert FFN intermediate
QWEN3_30B_NUM_KV_HEADS = 4
QWEN3_30B_NUM_EXPERTS = 128
QWEN3_30B_TOPK = 8
_QWEN3_30B_A3B_PROFILE = {
    # ---- Activation: SiluAndMul (SwiGLU) ----
    "tests/test_activation.py": {
        "test_act_and_mul": {
            "activation": ["silu_and_mul"],
            "num_tokens": [1, 128, 2048],
            "d": [QWEN3_30B_MOE_INTERMEDIATE_SIZE],
        },
        "test_activation": None,
    },
    # ---- RMSNorm ----
    "tests/test_layernorm.py": {
        "test_rms_norm": {
            "num_tokens": [1, 128, 2048],
            "hidden_size": [QWEN3_30B_HIDDEN_SIZE],
        },
    },
    # ---- Rotary Embedding: neox style ----
    "tests/test_rotary_embedding.py": {
        "test_rotary_embedding_opcheck": {
            "is_neox_style": [True],
            "max_position": [1024],
            "head_size": [QWEN3_30B_HEAD_DIM],
            "seq_len": [1, 128, 1024],
        },
    },
    # ---- KV Cache: GQA 32q/4kv ----
    "tests/test_cache.py": {
        "test_reshape_and_cache_flash": {
            "num_tokens": [1, 128],
            "num_heads": [QWEN3_30B_NUM_KV_HEADS],
            "head_size": [QWEN3_30B_HEAD_DIM],
            "block_size": [64],
            "dtype": [torch.bfloat16],
        },
    },
    # ---- TopK routing: 128 experts, top-8 ----
    "tests/test_topk.py": {
        "test_fused_topk_softmax": {
            "topk": [QWEN3_30B_TOPK],
            "n_expert": [QWEN3_30B_NUM_EXPERTS],
            "n_token": [1, 128, 2048],
        },
    },
    # ---- Fused MoE: 128 experts, top-8 ----
    "tests/fused_moe/test_fused_moe.py": {
        "test_fused_moe": {
            "m,n,k": [
                (1, QWEN3_30B_MOE_INTERMEDIATE_SIZE, QWEN3_30B_HIDDEN_SIZE),
                (128, QWEN3_30B_MOE_INTERMEDIATE_SIZE, QWEN3_30B_HIDDEN_SIZE),
            ],
            "e": [QWEN3_30B_NUM_EXPERTS],
            "topk": [QWEN3_30B_TOPK],
            "dtype": [torch.bfloat16],
            "has_bias": [True, False],
        },
    },
    # ---- Grouped GEMM: 128 experts, top-8 ----
    "tests/fused_moe/test_grouped_gemm.py": {
        "test_grouped_gemm": {
            "m,n,k": [
                (1, QWEN3_30B_MOE_INTERMEDIATE_SIZE, QWEN3_30B_HIDDEN_SIZE),
                (128, QWEN3_30B_MOE_INTERMEDIATE_SIZE, QWEN3_30B_HIDDEN_SIZE),
            ],
            "e": [QWEN3_30B_NUM_EXPERTS],
            "topk": [QWEN3_30B_TOPK],
            "dtype": [torch.bfloat16],
            "has_bias": [True, False],
        },
    },
    # ---- MoE prologue ----
    "tests/fused_moe/test_moe_prologue.py": {
        "test_prologue": {
            "m,n,k": [
                (1, QWEN3_30B_MOE_INTERMEDIATE_SIZE, QWEN3_30B_HIDDEN_SIZE),
                (128, QWEN3_30B_MOE_INTERMEDIATE_SIZE, QWEN3_30B_HIDDEN_SIZE),
            ],
            "e": [QWEN3_30B_NUM_EXPERTS],
            "topk": [QWEN3_30B_TOPK],
        },
    },
    # ---- MoE remap hidden states ----
    "tests/fused_moe/test_remap_hidden_states.py": {
        "test_remap_hidden_states": {
            "total_experts_num": [QWEN3_30B_NUM_EXPERTS],
            "topk": [QWEN3_30B_TOPK],
            "hidden_size": [QWEN3_30B_HIDDEN_SIZE],
        },
    },
    # ---- MoE align block size ----
    "tests/test_moe_align_block_size.py": {
        "test_moe_align_block_size": {
            "m": [1, 128, 2048],
            "num_experts": [QWEN3_30B_NUM_EXPERTS],
            "topk": [QWEN3_30B_TOPK],
            "block_size": [128],
        },
    },
    # ---- MoE gather ----
    "tests/test_moe_gather.py": {
        "test_moe_gather": {
            "input_len": [1, 128],
            "hidden_size": [QWEN3_30B_HIDDEN_SIZE],
            "num_experts": [QWEN3_30B_NUM_EXPERTS],
            "topk": [QWEN3_30B_TOPK],
        },
    },
    # ---- MoE sum ----
    "tests/test_moe_sum.py": {
        "test_moe_sum": {
            "m": [1, 128],
            "topk": [QWEN3_30B_TOPK],
            "k": [QWEN3_30B_HIDDEN_SIZE],
        },
    },
    # ---- Flash Attention: GQA 32q/4kv heads ----
    "tests/flash_attn/test_flash_attn_varlen_func.py": {
        "test_varlen_with_paged_kv": {
            "seq_lens": [[(1, 1328), (5, 18), (129, 463)]],
            "num_heads": [(QWEN3_30B_NUM_HEADS, QWEN3_30B_NUM_KV_HEADS)],
            "head_size": [QWEN3_30B_HEAD_DIM],
            "num_blocks": [2048],
            "window_size": [(-1, -1)],
            "is_paged": [True],
        },
        "test_decode_with_paged_kv": {
            "seq_lens": [[(1, 1025), (1, 523), (1, 37)]],
            "num_heads": [(QWEN3_30B_NUM_HEADS, QWEN3_30B_NUM_KV_HEADS)],
            "head_size": [QWEN3_30B_HEAD_DIM],
            "num_blocks": [2048],
            "window_size": [(-1, -1)],
        },
        "test_decode_with_paged_kv_mla": None,  # Not MLA
    },
    # ---- Merge attention states ----
    "tests/test_merge_attn_states.py": {
        "test_merge_attn_states": {
            "num_tokens": [1, 128],
            "num_query_heads": [QWEN3_30B_NUM_HEADS],
            "head_size": [QWEN3_30B_HEAD_DIM],
            "output_dtype": [torch.bfloat16],
        },
    },
    # ---- FP8 quantization ----
    "tests/test_fp8_quant.py": {
        "test_dynamic_per_tensor_fp8_quant": {
            "num_tokens": [1, 128],
            "hidden_size": [QWEN3_30B_HIDDEN_SIZE],
        },
        "test_dynamic_per_token_fp8_quant": {
            "num_tokens": [1, 128],
            "hidden_size": [QWEN3_30B_HIDDEN_SIZE],
        },
    },
    # ---- FP8 GEMM ----
    "tests/test_fp8_gemm_onednn.py": {
        "test_fp8_gemm_per_tensor": {
            "mnk_factors": [
                (1, QWEN3_30B_HIDDEN_SIZE, QWEN3_30B_MOE_INTERMEDIATE_SIZE),
                (128, QWEN3_30B_HIDDEN_SIZE, QWEN3_30B_MOE_INTERMEDIATE_SIZE),
            ],
        },
        "test_fp8_gemm_per_channel": {
            "mnk_factors": [
                (1, QWEN3_30B_HIDDEN_SIZE, QWEN3_30B_MOE_INTERMEDIATE_SIZE),
                (128, QWEN3_30B_HIDDEN_SIZE, QWEN3_30B_MOE_INTERMEDIATE_SIZE),
            ],
        },
    },
}

# ---------------------------------------------------------------------------
# Qwen3-235B-A22B (Qwen/Qwen3-235B-A22B-Instruct-2507)
#   - Qwen3MoE architecture, SiluAndMul (SwiGLU), RMSNorm
#   - hidden_size=4096, intermediate_size=12288, moe_intermediate_size=1536
#   - 64 attention heads, 4 KV heads, head_dim=128
#   - 128 experts, top-8 routing, norm_topk_prob=True
#   - neox-style RoPE
# ---------------------------------------------------------------------------
QWEN3_235B_HEAD_DIM = 128
QWEN3_235B_NUM_HEADS = 64
QWEN3_235B_HIDDEN_SIZE = 4096
QWEN3_235B_INTERMEDIATE_SIZE = 12288      # dense FFN intermediate
QWEN3_235B_MOE_INTERMEDIATE_SIZE = 1536   # per-expert FFN intermediate
QWEN3_235B_NUM_KV_HEADS = 4
QWEN3_235B_NUM_EXPERTS = 128
QWEN3_235B_TOPK = 8
_QWEN3_235B_A22B_PROFILE = {
    # ---- Activation: SiluAndMul (SwiGLU) ----
    "tests/test_activation.py": {
        "test_act_and_mul": {
            "activation": ["silu_and_mul"],
            "num_tokens": [1, 128, 2048],
            "d": [QWEN3_235B_MOE_INTERMEDIATE_SIZE],
        },
        "test_activation": None,
    },
    # ---- RMSNorm ----
    "tests/test_layernorm.py": {
        "test_rms_norm": {
            "num_tokens": [1, 128, 2048],
            "hidden_size": [QWEN3_235B_HIDDEN_SIZE],
        },
    },
    # ---- Rotary Embedding: neox style ----
    "tests/test_rotary_embedding.py": {
        "test_rotary_embedding_opcheck": {
            "is_neox_style": [True],
            "max_position": [1024],
            "head_size": [QWEN3_235B_HEAD_DIM],
            "seq_len": [1, 128, 1024],
        },
    },
    # ---- KV Cache: GQA 64q/4kv ----
    "tests/test_cache.py": {
        "test_reshape_and_cache_flash": {
            "num_tokens": [1, 128],
            "num_heads": [QWEN3_235B_NUM_KV_HEADS],
            "head_size": [QWEN3_235B_HEAD_DIM],
            "block_size": [64],
            "dtype": [torch.bfloat16],
        },
    },
    # ---- TopK routing: 128 experts, top-8 ----
    "tests/test_topk.py": {
        "test_fused_topk_softmax": {
            "topk": [QWEN3_235B_TOPK],
            "n_expert": [QWEN3_235B_NUM_EXPERTS],
            "n_token": [1, 128, 2048],
        },
    },
    # ---- Fused MoE: 128 experts, top-8 ----
    "tests/fused_moe/test_fused_moe.py": {
        "test_fused_moe": {
            "m,n,k": [
                (1, QWEN3_235B_MOE_INTERMEDIATE_SIZE, QWEN3_235B_HIDDEN_SIZE),
                (128, QWEN3_235B_MOE_INTERMEDIATE_SIZE,
                 QWEN3_235B_HIDDEN_SIZE),
            ],
            "e": [QWEN3_235B_NUM_EXPERTS],
            "topk": [QWEN3_235B_TOPK],
            "dtype": [torch.bfloat16],
            "has_bias": [True, False],
        },
    },
    # ---- Grouped GEMM: 128 experts, top-8 ----
    "tests/fused_moe/test_grouped_gemm.py": {
        "test_grouped_gemm": {
            "m,n,k": [
                (1, QWEN3_235B_MOE_INTERMEDIATE_SIZE, QWEN3_235B_HIDDEN_SIZE),
                (128, QWEN3_235B_MOE_INTERMEDIATE_SIZE,
                 QWEN3_235B_HIDDEN_SIZE),
            ],
            "e": [QWEN3_235B_NUM_EXPERTS],
            "topk": [QWEN3_235B_TOPK],
            "dtype": [torch.bfloat16],
            "has_bias": [True, False],
        },
    },
    # ---- MoE prologue ----
    "tests/fused_moe/test_moe_prologue.py": {
        "test_prologue": {
            "m,n,k": [
                (1, QWEN3_235B_MOE_INTERMEDIATE_SIZE, QWEN3_235B_HIDDEN_SIZE),
                (128, QWEN3_235B_MOE_INTERMEDIATE_SIZE,
                 QWEN3_235B_HIDDEN_SIZE),
            ],
            "e": [QWEN3_235B_NUM_EXPERTS],
            "topk": [QWEN3_235B_TOPK],
        },
    },
    # ---- MoE remap hidden states ----
    "tests/fused_moe/test_remap_hidden_states.py": {
        "test_remap_hidden_states": {
            "total_experts_num": [QWEN3_235B_NUM_EXPERTS],
            "topk": [QWEN3_235B_TOPK],
            "hidden_size": [QWEN3_235B_HIDDEN_SIZE],
        },
    },
    # ---- MoE align block size ----
    "tests/test_moe_align_block_size.py": {
        "test_moe_align_block_size": {
            "m": [1, 128, 2048],
            "num_experts": [QWEN3_235B_NUM_EXPERTS],
            "topk": [QWEN3_235B_TOPK],
            "block_size": [128],
        },
    },
    # ---- MoE gather ----
    "tests/test_moe_gather.py": {
        "test_moe_gather": {
            "input_len": [1, 128],
            "hidden_size": [QWEN3_235B_HIDDEN_SIZE],
            "num_experts": [QWEN3_235B_NUM_EXPERTS],
            "topk": [QWEN3_235B_TOPK],
        },
    },
    # ---- MoE sum ----
    "tests/test_moe_sum.py": {
        "test_moe_sum": {
            "m": [1, 128],
            "topk": [QWEN3_235B_TOPK],
            "k": [QWEN3_235B_HIDDEN_SIZE],
        },
    },
    # ---- Flash Attention: GQA 64q/4kv heads ----
    "tests/flash_attn/test_flash_attn_varlen_func.py": {
        "test_varlen_with_paged_kv": {
            "seq_lens": [[(1, 1328), (5, 18), (129, 463)]],
            "num_heads": [(QWEN3_235B_NUM_HEADS, QWEN3_235B_NUM_KV_HEADS)],
            "head_size": [QWEN3_235B_HEAD_DIM],
            "num_blocks": [2048],
            "window_size": [(-1, -1)],
            "is_paged": [True],
        },
        "test_decode_with_paged_kv": {
            "seq_lens": [[(1, 1025), (1, 523), (1, 37)]],
            "num_heads": [(QWEN3_235B_NUM_HEADS, QWEN3_235B_NUM_KV_HEADS)],
            "head_size": [QWEN3_235B_HEAD_DIM],
            "num_blocks": [2048],
            "window_size": [(-1, -1)],
        },
        "test_decode_with_paged_kv_mla": None,  # Not MLA
    },
    # ---- Merge attention states ----
    "tests/test_merge_attn_states.py": {
        "test_merge_attn_states": {
            "num_tokens": [1, 128],
            "num_query_heads": [QWEN3_235B_NUM_HEADS],
            "head_size": [QWEN3_235B_HEAD_DIM],
            "output_dtype": [torch.bfloat16],
        },
    },
    # ---- FP8 quantization ----
    "tests/test_fp8_quant.py": {
        "test_dynamic_per_tensor_fp8_quant": {
            "num_tokens": [1, 128],
            "hidden_size": [QWEN3_235B_HIDDEN_SIZE],
        },
        "test_dynamic_per_token_fp8_quant": {
            "num_tokens": [1, 128],
            "hidden_size": [QWEN3_235B_HIDDEN_SIZE],
        },
    },
    # ---- FP8 GEMM ----
    "tests/test_fp8_gemm_onednn.py": {
        "test_fp8_gemm_per_tensor": {
            "mnk_factors": [
                (1, QWEN3_235B_HIDDEN_SIZE, QWEN3_235B_MOE_INTERMEDIATE_SIZE),
                (128, QWEN3_235B_HIDDEN_SIZE,
                 QWEN3_235B_MOE_INTERMEDIATE_SIZE),
            ],
        },
        "test_fp8_gemm_per_channel": {
            "mnk_factors": [
                (1, QWEN3_235B_HIDDEN_SIZE, QWEN3_235B_MOE_INTERMEDIATE_SIZE),
                (128, QWEN3_235B_HIDDEN_SIZE,
                 QWEN3_235B_MOE_INTERMEDIATE_SIZE),
            ],
        },
    },
}

# ---------------------------------------------------------------------------
# Registry of all on-demand profiles
# ---------------------------------------------------------------------------
ONDEMAND_PROFILES = {
    "llama3": _LLAMA3_PROFILE,
    "llama4": _LLAMA4_PROFILE,
    "deepseek": _DEEPSEEK_PROFILE,
    "qwen3_30b_a3b": _QWEN3_30B_A3B_PROFILE,
    "qwen3_235b_a22b": _QWEN3_235B_A22B_PROFILE,
}
