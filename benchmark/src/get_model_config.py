# SPDX-License-Identifier: Apache-2.0
import itertools

import torch

from tests.utils import get_model_config

# TODO: get "OpenGVLab/InternVL3_5-8B",
#   "deepseek-ai/DeepSeek-OCR" config failed, need to investigate
model_lists = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "openbmb/MiniCPM-V-4",
    "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen2.5-VL-32B-Instruct",
    "deepseek-ai/DeepSeek-V2-Lite", "Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3-32B",
]


def gen_cutlass_fused_moe_correctness_configs():
    mnk = [
        (1, 5120, 8192),
        (4, 5120, 8192),
        (16, 5120, 8192),
        (1024, 5120, 8192),
        (8192, 5120, 8192),
    ]
    experts = [16]
    topk = [1]
    x_dtype = [torch.float16, torch.bfloat16]
    w_dtype = [torch.float8_e5m2, torch.float8_e4m3fn, None]
    has_bias = [True, False]

    configs = list(
        itertools.product(mnk, experts, topk, x_dtype, w_dtype, has_bias))
    return configs


def gen_cutlass_fused_moe_perf_configs():
    configs = []
    x_dtype = [torch.float16, torch.bfloat16]
    w_dtype = [torch.float8_e5m2, torch.float8_e4m3fn, None]
    has_bias = [True, False]
    input_lens = [1, 4, 16, 1024, 8192]

    for model in model_lists:
        model_config = get_model_config(model, tp_size=1)
        if not model_config["is_moe"]:
            continue

        hidden_size = model_config["hidden_size"]
        intermediate_size = model_config["intermediate_size"]
        mnk = list(
            zip(input_lens, [hidden_size] * len(input_lens),
                [intermediate_size] * len(input_lens)))

        moe_top_k = model_config["moe_config"]["moe_top_k"]
        num_experts = model_config["num_groups"]
        configs += list(
            itertools.product(mnk, [num_experts],
                              [moe_top_k], x_dtype, w_dtype, has_bias))

    # Hardcoded model shapes (n, k, num_experts, topk) for various TP sizes.
    # Config tuple (m, n, k) produces:
    #   GEMM1 (gate+up): [m, 2*n, k]
    #   GEMM2 (down):    [m, k, n]
    hardcoded_model_shapes = [
        # Qwen/Qwen3-30B-A3B, Qwen3-30B-A3B-Instruct-2507,
        # Qwen3-Coder-30B-A3B-Instruct, OpenGVLab/InternVL3_5-30B-A3B
        # [128 experts, topk=8]
        (768, 2048, 128, 8),     # tp=1: [m,1536,2048], [m,2048,768]
        (384, 2048, 128, 8),     # tp=2: [m,768,2048],  [m,2048,384]
        (192, 2048, 128, 8),     # tp=4: [m,384,2048],  [m,2048,192]
        # Qwen/Qwen3-Coder-Next, Qwen3-Next-80B-A3B-Instruct,
        # Qwen3-Next-80B-A3B-Thinking
        # [512 experts, topk=10]
        (512, 2048, 512, 10),    # tp=1: [m,1024,2048], [m,2048,512]
        (256, 2048, 512, 10),    # tp=2: [m,512,2048],  [m,2048,256]
        (128, 2048, 512, 10),    # tp=4: [m,256,2048],  [m,2048,128]
        # deepseek-ai/DeepSeek-V2-Lite, moonshotai/Kimi-VL-A3B-Thinking
        # [64 experts, topk=6]
        (1408, 2048, 64, 6),     # tp=1: [m,2816,2048], [m,2048,1408]
        (704, 2048, 64, 6),      # tp=2: [m,1408,2048], [m,2048,704]
        (352, 2048, 64, 6),      # tp=4: [m,704,2048],  [m,2048,352]
        # deepseek-ai/DeepSeek-OCR
        # [64 experts, topk=6]
        (896, 1280, 64, 6),      # tp=1: [m,1792,1280], [m,1280,896]
        (448, 1280, 64, 6),      # tp=2: [m,896,1280],  [m,1280,448]
        (224, 1280, 64, 6),      # tp=4: [m,448,1280],  [m,1280,224]
        # Qwen/Qwen3.5-35B-A3B
        # [256 experts, topk=8]
        (512, 2048, 256, 8),     # tp=1: [m,1024,2048], [m,2048,512]
        (256, 2048, 256, 8),     # tp=2: [m,512,2048],  [m,2048,256]
        (128, 2048, 256, 8),     # tp=4: [m,256,2048],  [m,2048,128]
        # mistralai/Mixtral-8x7B-Instruct-v0.1
        # [8 experts, topk=2]
        (14336, 4096, 8, 2),     # tp=1: [m,28672,4096],[m,4096,14336]
        (7168, 4096, 8, 2),      # tp=2: [m,14336,4096],[m,4096,7168]
        (3584, 4096, 8, 2),      # tp=4: [m,7168,4096], [m,4096,3584]
    ]

    for n, k, num_experts, topk in hardcoded_model_shapes:
        mnk = list(zip(input_lens, [n] * len(input_lens),
                       [k] * len(input_lens)))
        configs += list(
            itertools.product(mnk, [num_experts],
                              [topk], x_dtype, w_dtype, has_bias))

    configs = set(configs)  # remove duplicates

    def sort_key(x):
        (m, n, k), moe_topk, topk_, x_dtype_, w_dtype_, bias_ = x

        return (m, n, k, moe_topk, topk_, str(x_dtype_), str(w_dtype_), bias_)

    configs = sorted(configs, key=sort_key)
    return configs


def gen_cutlass_flash_attn_varlen_correctness_configs():
    # seq_lens = [[(1, 1328), (5, 18), (129, 463)]]
    num_seqs = [3]
    query_lens = ["1,5,129"]
    kv_lens = ["1328,18,463"]

    num_heads = [(4, 4), (8, 2), (10, 2), (16, 1)]
    head_size = [64, 128, 192, 256]
    block_size = [64, 128]
    window_size = [(-1, 127), (127, -1), (64, 64), (-1, -1)]
    output_dtype = [torch.float16, torch.bfloat16]
    soft_cap = [None]
    num_blocks = [32768, 16324, 2048]
    fa_versions = [2]
    q_dtype = [None]
    is_sink = [False, True]
    is_causal = [False, True]
    is_paged = [False, True]
    kv_dtype = [torch.float8_e5m2, torch.float8_e4m3fn, None]

    configs = list(
        itertools.product(num_seqs, query_lens, kv_lens, num_heads, head_size,
                          block_size, window_size, output_dtype, soft_cap,
                          num_blocks, fa_versions, q_dtype, is_sink, is_causal,
                          is_paged, kv_dtype))

    return configs


def gen_cutlass_flash_attn_varlen_perf_configs():
    num_seqs = [4]
    query_lens = ["1024,2048,2048,8192"]
    kv_lens = ["1024,1024,2048,8192"]

    block_size = [64, 128]
    window_size = [(-1, -1)]
    output_dtype = [torch.float16, torch.bfloat16]
    soft_cap = [None]
    num_blocks = [16324]
    fa_versions = [2]
    q_dtype = [None]
    is_sink = [False, True]
    is_causal = [False, True]
    is_paged = [False, True]
    # kv_dtype = [torch.float8_e5m2, torch.float8_e4m3fn, None]     # fp8 OOM
    kv_dtype = [None]

    # Hardcoded attention shapes for models that cannot be loaded via
    # AutoConfig (e.g. diffusion models without a standard model_type).
    # Each entry is a dict with required keys:
    #   "shape": (num_attention_heads, num_key_value_heads, head_dim)
    # and optional per-entry overrides for shapes that need reduced params
    # (e.g. to fit within a given device's memory budget):
    #   "query_lens", "kv_lens", "num_blocks"
    # If an override is not provided, the default values above are used.
    hardcoded_attn_shapes = [
        # Wan-AI/Wan2.2-I2V-A14B-Diffusers (DiT video diffusion)
        # Reduced seq lens / num_blocks to fit ~20G memory (e.g. B60).
        {
            "shape": (40, 40, 128),
            "query_lens": ["256,512,512,2048"],
            "kv_lens": ["256,256,512,2048"],
            "num_blocks": [2048],
        },
    ]

    def get_configs_from_models():
        configs = []
        for model in model_lists:
            model_config = get_model_config(model, tp_size=1)
            head_size = [model_config["head_dim"]]
            num_heads = [(model_config["num_attention_heads"],
                          model_config["num_key_value_heads"])]

            configs += list(
                itertools.product(num_seqs, query_lens, kv_lens, num_heads,
                                  head_size, block_size, window_size,
                                  output_dtype, soft_cap, num_blocks,
                                  fa_versions, q_dtype, is_sink, is_causal,
                                  is_paged, kv_dtype))

        # Add hardcoded attention shapes (diffusion models, etc.)
        # Each entry may override query_lens / kv_lens / num_blocks to
        # control memory usage for that specific shape.
        for entry in hardcoded_attn_shapes:
            n_heads, n_kv_heads, h_dim = entry["shape"]
            q_lens = entry.get("query_lens", query_lens)
            k_lens = entry.get("kv_lens", kv_lens)
            n_blocks = entry.get("num_blocks", num_blocks)
            configs += list(
                itertools.product(num_seqs, q_lens, k_lens,
                                  [(n_heads, n_kv_heads)], [h_dim],
                                  block_size, window_size, output_dtype,
                                  soft_cap, n_blocks, fa_versions,
                                  q_dtype, is_sink, is_causal, is_paged,
                                  kv_dtype))

        configs = set(configs)  # remove duplicates

        def sort_key(x):
            (num_seq, query_len, kv_len, num_head, head_size, block_size,
             window_size, output_dtype_, soft_cap, num_blocks, fa_version,
             q_dtype, is_sink, is_causal, is_paged, kv_dtype) = x

            return (num_seq, query_len, kv_len, num_head, head_size,
                    block_size, window_size, str(output_dtype_),
                    soft_cap if soft_cap is not None else -1, num_blocks,
                    fa_version, str(q_dtype), is_sink, is_causal, is_paged,
                    str(kv_dtype))

        configs = sorted(configs, key=sort_key)

        # single-sequence non-paged prefill: 1025 tokens, 8 heads, head_dim=64,
        # no sink, no paged KV, causal and non-causal variants
        # Q/K/V shape=(1025, 8, 64), cu_seqlens_q/k=[0,1025], 
        # window_size=(-1,-1)
        for causal in [False, True]:
            for out_dtype in [torch.float16, torch.bfloat16]:
                configs.append((1, "1025", "1025", (8, 8), 64, 64, (-1, -1),
                                out_dtype, None, 2048, 2, None, False, causal,
                                False, None))

        # Causal-prefill shapes across common GQA ratios (MHA, GQA-4:1, GQA-8:1)
        # used to validate the XeFHMAIndividualReverseOrderTileScheduler variant
        # selected by the auto-heuristic / VLLM_XPU_FA_REVERSE_ODD_HEADS=on env
        # override. head_dim=128, fp16, causal. Each tuple is
        # (bs, q_heads, kv_heads, qlen, cached_kv_len).
        causal_prefill_shapes = [
            # MHA (32x32 heads), single-batch prefill
            (1, 32, 32,  4096,     0),
            (1, 32, 32, 16384,     0),
            # GQA 4:1 (32x8 heads), single-batch + chunked-prefill
            (1, 32,  8,  8192,     0),
            (2, 32,  8,  4096,     0),
            (1, 32,  8,  4096, 12288),
            # GQA 8:1 (64x8 heads), long-ish single-batch + chunked-prefill
            (1, 64,  8,  8192,     0),
            (1, 64,  8,  2048, 14336),
            # batched short prefill (decode-adjacent regime)
            (4, 32,  8,  2048,     0),
            (8, 16,  4,  1024,     0),
        ]
        # num_blocks=2048 fits the largest kvlen=16384 (256 blocks at bs=1).
        for bs, qh, kvh, qlen, cached in causal_prefill_shapes:
            kvlen = qlen + cached
            qlens_str = ",".join([str(qlen)] * bs)
            kvlens_str = ",".join([str(kvlen)] * bs)
            for paged in [False, True]:
                configs.append(
                    (bs, qlens_str, kvlens_str, (qh, kvh), 128, 64, (-1, -1),
                     torch.float16, None, 2048, 2, None, False, True, paged,
                     None))
        return configs

    configs = get_configs_from_models()

    return configs


def gen_cutlass_flash_attn_decode_correctness_configs():
    # seq_lens = [[(1, 1025)], [(1, 523), (1, 37), (1, 2011)], [(1, 13000)],
    #             [(1, 523), (1, 37), (1, 2011), (1, 5000)]]
    seq_lens = [
        "1,1,1025", "3,1+1+1,523+37+2011", "1,1,13000",
        "4,1+1+1+1,523+37+2011+5000"
    ]
    num_heads = [(4, 4), (8, 2), (10, 2), (16, 1)]
    head_size = [64, 128, 192, 256]
    block_size = [64, 128]
    output_dtype = [torch.float16, torch.bfloat16]
    soft_cap = [None]
    num_blocks = [32768, 2048]
    fa_versions = [2]
    q_dtype = [None]
    is_sink = [False, True]

    configs = list(
        itertools.product(seq_lens, num_heads, head_size, block_size,
                          output_dtype, soft_cap, num_blocks, fa_versions,
                          q_dtype, is_sink))
    return configs


def gen_cutlass_flash_attn_decode_perf_configs():
    seq_lens = [
        "1,1,4096", "8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
        "32," + "+".join(["1"] * 32) + "," + "+".join(["512"] * 32)
    ]
    num_heads = [(4, 4), (16, 1)]
    head_size = [64, 128, 256]
    block_size = [64, 128]
    output_dtype = [torch.float16, torch.bfloat16]
    soft_cap = [None]
    num_blocks = [2048]
    fa_versions = [2]
    q_dtype = [None]
    is_sink = [False, True]

    configs = []
    for model in model_lists:
        model_config = get_model_config(model, tp_size=1)
        head_size = [model_config["head_dim"]]
        num_heads = [(model_config["num_attention_heads"],
                      model_config["num_key_value_heads"])]

        configs += list(
            itertools.product(seq_lens, num_heads, head_size, block_size,
                              output_dtype, soft_cap, num_blocks, fa_versions,
                              q_dtype, is_sink))

    configs = set(configs)  # remove duplicates

    def sort_key(x):
        (seq_len, num_head, head_size, block_size, output_dtype_, soft_cap,
         num_blocks, fa_version, q_dtype, is_sink) = x

        return (seq_len, num_head, head_size, block_size, str(output_dtype_),
                soft_cap if soft_cap is not None else -1, num_blocks,
                fa_version, str(q_dtype), is_sink)

    configs = sorted(configs, key=sort_key)

    return configs
