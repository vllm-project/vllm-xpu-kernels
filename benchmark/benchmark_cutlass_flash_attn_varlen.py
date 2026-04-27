# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

# isort: off
import gc

import torch
import triton

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)

from benchmark.src.flash_attn_interface_ import (
    flash_attn_varlen_func_CalKernelTime)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_varlen_correctness_configs as
    gen_correctness_config)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_varlen_perf_configs as gen_perf_configs)
from tests.flash_attn.test_flash_attn_varlen_func import ref_paged_attn
from tests.utils import parse_args, seed_everything
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
# isort: on

DEVICE = "xpu"


def clear_xpu_cache():
    torch.xpu.empty_cache()
    gc.collect()
    torch.xpu.synchronize()


def calculate_flops(num_query_heads, query_lens, kv_lens, head_size,
                    is_causal):
    total = 0
    for sq, sk in zip(query_lens, kv_lens):
        effective_sk = sk * 0.5 if is_causal else sk
        total += 4 * num_query_heads * sq * effective_sk * head_size
    return total


def make_varlen_with_paged_kv_input(config):
    num_seqs, query_lens, kv_lens, num_heads, head_size, \
        block_size, window_size, output_dtype, _, num_blocks, \
        _, q_dtype, is_sink, is_causal, is_paged, kv_dtype = config
    query_lens = query_lens.split(",")
    query_lens = [int(x) for x in query_lens]
    kv_lens = kv_lens.split(",")
    kv_lens = [int(x) for x in kv_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=output_dtype)
    if is_paged:
        key_cache = torch.randn(num_blocks,
                                block_size,
                                num_kv_heads,
                                head_size,
                                dtype=output_dtype)
    else:
        key_cache = torch.randn(sum(kv_lens),
                                num_query_heads,
                                head_size,
                                dtype=output_dtype)
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
    cu_kv_lens = torch.tensor([0] + kv_lens,
                              dtype=torch.int32).cumsum(dim=0,
                                                        dtype=torch.int32)
    seq_k = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    sink = None
    if is_sink:
        sink = torch.randn(num_query_heads, dtype=output_dtype)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None  #noqa: F841
    k_descale = None  #noqa: F841
    v_descale = None  #noqa: F841
    scale_shape = (num_seqs, num_kv_heads)
    is_fp8_query = q_dtype is not None
    if is_fp8_query:
        q_descale = (torch.abs(query).max() / 200).to(torch.float32)
        maybe_quantized_query = (query / q_descale).to(q_dtype)
    is_fp8kv = kv_dtype is not None
    if is_fp8kv:
        k_descale = (torch.abs(key_cache).max() / 200).to(torch.float32)
        v_descale = (torch.abs(value_cache).max() / 200).to(torch.float32)
        maybe_quantized_key_cache = (key_cache / k_descale).to(kv_dtype)
        maybe_quantized_value_cache = (value_cache / v_descale).to(kv_dtype)
    return (maybe_quantized_query, maybe_quantized_key_cache,
            maybe_quantized_value_cache, max_query_len, cu_query_lens,
            max_kv_len, cu_kv_lens, seq_k, q_descale, k_descale, v_descale,
            scale, is_causal, block_tables, window_size, sink, scale_shape,
            query, query_lens, kv_lens, is_fp8kv, is_fp8_query,
            max_num_blocks_per_seq)


def calculate_diff_varlen_paged_kv(config):
    _, _, _, _, _, _, window_size, output_dtype, _, _, _, \
        q_dtype, _, is_causal, is_paged, kv_dtype = config
    maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, \
        max_query_len, cu_query_lens, max_kv_len, cu_kv_lens, \
        seq_k, q_descale, k_descale, v_descale, scale, is_causal, \
        block_tables, window_size, sink, scale_shape, query, \
        query_lens, kv_lens, is_fp8kv, is_fp8_query, _ = \
    make_varlen_with_paged_kv_input(config)

    if is_paged:
        output = flash_attn_varlen_func(maybe_quantized_query,
                                        maybe_quantized_key_cache,
                                        maybe_quantized_value_cache,
                                        max_query_len,
                                        cu_query_lens,
                                        max_kv_len,
                                        seqused_k=seq_k,
                                        q_descale=q_descale.expand(scale_shape)
                                        if q_descale is not None else None,
                                        k_descale=k_descale.expand(scale_shape)
                                        if k_descale is not None else None,
                                        v_descale=v_descale.expand(scale_shape)
                                        if v_descale is not None else None,
                                        softmax_scale=scale,
                                        causal=is_causal,
                                        block_table=block_tables,
                                        window_size=window_size,
                                        s_aux=sink)
    else:
        output = flash_attn_varlen_func(maybe_quantized_query,
                                        maybe_quantized_key_cache,
                                        maybe_quantized_value_cache,
                                        max_query_len,
                                        cu_query_lens,
                                        max_kv_len,
                                        cu_seqlens_k=cu_kv_lens,
                                        q_descale=q_descale.expand(scale_shape)
                                        if q_descale is not None else None,
                                        k_descale=k_descale.expand(scale_shape)
                                        if k_descale is not None else None,
                                        v_descale=v_descale.expand(scale_shape)
                                        if v_descale is not None else None,
                                        softmax_scale=scale,
                                        causal=is_causal,
                                        block_table=None,
                                        window_size=window_size,
                                        s_aux=sink)

    ref_output = ref_paged_attn(query=query,
                                key_cache=maybe_quantized_key_cache,
                                value_cache=maybe_quantized_value_cache,
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                casual=is_causal,
                                is_paged=is_paged,
                                sink=sink,
                                q_descale=q_descale,
                                k_descale=k_descale,
                                v_descale=v_descale,
                                window_size_left=window_size[0],
                                window_size_right=window_size[1],
                                is_fp8kv=is_fp8kv,
                                is_fp8_query=is_fp8_query,
                                dtype=output_dtype)
    atol, rtol = 1e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    if window_size[0] != -1 or window_size[1] != -1:
        atol, rtol = 1.5e-2, 1.5e-2
    if kv_dtype is not None:
        atol, rtol = 1.5e-2, 1.5e-2
    try:
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
            f"{torch.max(torch.abs(output - ref_output))}"
        print("✅ All implementations match, ", config)
    except AssertionError as e:
        print("❌ Implementations differ, ", config, " error: ", e)


def benchmark_varlen_with_paged_kv(num_seqs,
                                   query_lens,
                                   kv_lens,
                                   num_heads,
                                   head_size,
                                   block_size,
                                   window_size,
                                   output_dtype,
                                   soft_cap,
                                   num_blocks,
                                   fa_versions,
                                   q_dtype,
                                   is_sink,
                                   is_causal,
                                   is_paged,
                                   kv_dtype,
                                   provider,
                                   iterations=20):
    maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, \
        max_query_len, cu_query_lens, max_kv_len, cu_kv_lens, \
        seq_k, q_descale, k_descale, v_descale, scale, is_causal, \
        block_tables, window_size, sink, scale_shape, _, \
        query_lens, kv_lens, _, _, max_num_blocks_per_seq = \
            make_varlen_with_paged_kv_input(config=(num_seqs,
                query_lens, kv_lens, num_heads, head_size,
                block_size, window_size, output_dtype, soft_cap,
                num_blocks, fa_versions, q_dtype, is_sink,
                is_causal, is_paged, kv_dtype))
    num_query_heads = num_heads[0]

    print(f"Running config: {num_seqs, query_lens, kv_lens, \
                              num_heads, head_size, block_size, \
                              window_size, output_dtype, soft_cap, num_blocks, \
                              fa_versions, q_dtype, is_sink, is_causal, \
                              is_paged, kv_dtype}, Provider: {provider}",
          flush=True)
    assert iterations > 5, \
    "Number of iterations should be greater than 5 to account for warmup"

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    total_latency = 0.0
    ms = 0.0

    queries = [
        torch.rand_like(maybe_quantized_query) for _ in range(iterations)
    ]

    if is_paged:
        for index in range(iterations):
            block_tables = torch.randint(0,
                                         num_blocks,
                                         (num_seqs, max_num_blocks_per_seq),
                                         dtype=torch.int32)
            if provider == "flash_kernel_time" or \
            provider == "flash_kernel_TFLOPS":
                flash_attn_varlen_func_CalKernelTime(
                    queries[index],
                    maybe_quantized_key_cache,
                    maybe_quantized_value_cache,
                    max_query_len,
                    cu_query_lens,
                    max_kv_len,
                    seqused_k=seq_k,
                    q_descale=q_descale.expand(scale_shape)
                    if q_descale is not None else None,
                    k_descale=k_descale.expand(scale_shape)
                    if k_descale is not None else None,
                    v_descale=v_descale.expand(scale_shape)
                    if v_descale is not None else None,
                    softmax_scale=scale,
                    causal=is_causal,
                    block_table=block_tables,
                    window_size=window_size,
                    s_aux=sink,
                    start_event=start,
                    end_event=end)
            else:
                start.record()
                flash_attn_varlen_func(queries[index],
                                       maybe_quantized_key_cache,
                                       maybe_quantized_value_cache,
                                       max_query_len,
                                       cu_query_lens,
                                       max_kv_len,
                                       seqused_k=seq_k,
                                       q_descale=q_descale.expand(scale_shape)
                                       if q_descale is not None else None,
                                       k_descale=k_descale.expand(scale_shape)
                                       if k_descale is not None else None,
                                       v_descale=v_descale.expand(scale_shape)
                                       if v_descale is not None else None,
                                       softmax_scale=scale,
                                       causal=is_causal,
                                       block_table=block_tables,
                                       window_size=window_size,
                                       s_aux=sink)
                end.record()
                end.synchronize()
            if index >= 5:  # skip the first 5 iterations for warmup
                total_latency += start.elapsed_time(end)
    else:
        for index in range(iterations):
            if provider == "flash_kernel_time" or \
            provider == "flash_kernel_TFLOPS":
                flash_attn_varlen_func_CalKernelTime(
                    queries[index],
                    maybe_quantized_key_cache,
                    maybe_quantized_value_cache,
                    max_query_len,
                    cu_query_lens,
                    max_kv_len,
                    cu_seqlens_k=cu_kv_lens,
                    q_descale=q_descale.expand(scale_shape)
                    if q_descale is not None else None,
                    k_descale=k_descale.expand(scale_shape)
                    if k_descale is not None else None,
                    v_descale=v_descale.expand(scale_shape)
                    if v_descale is not None else None,
                    softmax_scale=scale,
                    causal=is_causal,
                    block_table=None,
                    window_size=window_size,
                    s_aux=sink,
                    start_event=start,
                    end_event=end)
            else:
                start.record()
                flash_attn_varlen_func(queries[index],
                                       maybe_quantized_key_cache,
                                       maybe_quantized_value_cache,
                                       max_query_len,
                                       cu_query_lens,
                                       max_kv_len,
                                       cu_seqlens_k=cu_kv_lens,
                                       q_descale=q_descale.expand(scale_shape)
                                       if q_descale is not None else None,
                                       k_descale=k_descale.expand(scale_shape)
                                       if k_descale is not None else None,
                                       v_descale=v_descale.expand(scale_shape)
                                       if v_descale is not None else None,
                                       softmax_scale=scale,
                                       causal=is_causal,
                                       block_table=None,
                                       window_size=window_size,
                                       s_aux=sink)
                end.record()
                end.synchronize()
            if index >= 5:  # skip the first 5 iterations for warmup
                total_latency += start.elapsed_time(end)
    if provider == "flash_kernel_TFLOPS":
        torch.xpu.synchronize()
        ms = total_latency / (iterations - 5)
        flops = calculate_flops(num_query_heads, query_lens, kv_lens,
                                head_size, is_causal)
        tflops = flops / (ms / 1000) / 1e12
        clear_xpu_cache()
        return tflops

    torch.xpu.synchronize()
    ms = total_latency / (iterations - 5)
    clear_xpu_cache()

    return 1000 * ms


def get_benchmark_varlen_with_paged_kv(iterations=20):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "num_seqs", "query_lens", "kv_lens", "num_heads", "head_size",
                "block_size", "window_size", "output_dtype", "soft_cap",
                "num_blocks", "fa_versions", "q_dtype", "is_sink", "is_causal",
                "is_paged", "kv_dtype"
            ],
            x_vals=[tuple(c) for c in configs],
            line_arg="provider",
            line_vals=["flash", "flash_kernel_time", "flash_kernel_TFLOPS"],
            line_names=[
                "FlashAttention(us)", "FlashAttention_Kernel_Time(us)",
                "FlashAttention_TFLOPS"
            ],
            styles=[("blue", "-"), ("green", "-"), ("purple", "-")],
            ylabel="Latency (us)",
            plot_name="flash-attn-varlen",
            args={},
        ))
    def benchmark(num_seqs, query_lens, kv_lens, num_heads, head_size,
                  block_size, window_size, output_dtype, soft_cap, num_blocks,
                  fa_versions, q_dtype, is_sink, is_causal, is_paged, kv_dtype,
                  provider):
        return benchmark_varlen_with_paged_kv(
            num_seqs=num_seqs,
            query_lens=query_lens,
            kv_lens=kv_lens,
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            window_size=window_size,
            output_dtype=output_dtype,
            soft_cap=soft_cap,
            num_blocks=num_blocks,
            fa_versions=fa_versions,
            q_dtype=q_dtype,
            is_sink=is_sink,
            is_causal=is_causal,
            is_paged=is_paged,
            kv_dtype=kv_dtype,
            provider=provider,
            iterations=iterations,
        )

    return benchmark


def filter_configs(configs):
    new_configs = []
    for config in configs:
        if (config[5] == 128 and config[9] == 32768 and config[4] >= 192) or \
            (config[6][0] != -1 or config[6][1] != -1):
            print("Skipping config due to potential OOM: ", config)
            continue
        new_configs.append(config)
    return new_configs


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    seed_everything(seed)
    iterations = 20
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")

    configs = gen_correctness_config()
    configs = filter_configs(configs)

    for config in configs:
        try:
            calculate_diff_varlen_paged_kv(config)
        except Exception as e:
            print("Error in config: ", config, " error: ", e)
        clear_xpu_cache()

    configs = gen_perf_configs()
    configs = filter_configs(configs)
    benchmark = get_benchmark_varlen_with_paged_kv(iterations=iterations)
    save_path = ensure_save_path_exists(args.save_path)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=save_path)
