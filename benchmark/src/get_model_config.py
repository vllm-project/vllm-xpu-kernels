# SPDX-License-Identifier: Apache-2.0
import itertools

import torch

from tests.utils import get_model_config

# TODO: get "OpenGVLab/InternVL3_5-8B",
#   "deepseek-ai/DeepSeek-OCR" config failed, need to investigate
model_lists = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "openbmb/MiniCPM-V-4",
    "Qwen/Qwen3-30B-A3B", "Qwen/Qwen2.5-VL-32B-Instruct",
    "deepseek-ai/DeepSeek-V2-Lite", "Qwen/Qwen3.5-35B-A3B"
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
