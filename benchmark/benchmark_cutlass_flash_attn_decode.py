# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# isort: off
import gc

import torch
import triton

from benchmark.src.flash_attn_interface_ import (
    flash_attn_varlen_func_CalKernelTime)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_decode_correctness_configs as
    gen_correctness_config)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_decode_perf_configs as gen_perf_configs)
from tests.flash_attn.test_flash_attn_varlen_func import ref_paged_attn
from tests.utils import parse_args, seed_everything
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
# isort: on

DEVICE = "xpu"


def clear_xpu_cache():
    torch.xpu.empty_cache()
    gc.collect()
    torch.xpu.synchronize()


def calculate_memory_usage(kv_len_sum, num_kv_heads, head_size, output_dtype):
    # Memory for key and value caches
    kv_cache_memory = 2 * kv_len_sum * num_kv_heads * \
    head_size * torch.tensor([], dtype=output_dtype).element_size()
    return kv_cache_memory / (1024**3)  # Convert to GB


def make_decode_with_paged_kv_input(config):
    seq_lens, num_heads, head_size, block_size, \
    output_dtype, _, num_blocks, _, q_dtype, is_sink = config
    # if num_heads == (16, 1) and head_size == 256:
    #     pytest.skip("skip test cases that may run out of SLM.")
    num_seqs = int(seq_lens.split(",")[0])
    query_lens = list(map(lambda x: int(x), seq_lens.split(",")[1].split("+")))
    kv_lens = list(map(lambda x: int(x), seq_lens.split(",")[2].split("+")))
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
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=output_dtype)
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens,
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
    if q_dtype is not None:
        # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = torch.ones(scale_shape, dtype=torch.float32)  #noqa: F841
        k_descale = torch.ones(scale_shape, dtype=torch.float32)  #noqa: F841
        v_descale = torch.ones(scale_shape, dtype=torch.float32)  #noqa: F841
    return maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, max_query_len, cu_query_lens, \
            max_kv_len, seq_k, scale, block_tables, sink, query, \
                key_cache, value_cache, query_lens, kv_lens


def calculate_diff_decode_paged_kv(config):
    _, _, _, _, _, _, _, _, q_dtype, _ = config
    maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, max_query_len, cu_query_lens, \
        max_kv_len, seq_k, scale, block_tables, sink, query, \
        key_cache, value_cache, query_lens, kv_lens = \
        make_decode_with_paged_kv_input(config)

    output = flash_attn_varlen_func(maybe_quantized_query,
                                    maybe_quantized_key_cache,
                                    maybe_quantized_value_cache,
                                    max_query_len,
                                    cu_query_lens,
                                    max_kv_len,
                                    seqused_k=seq_k,
                                    softmax_scale=scale,
                                    causal=False,
                                    block_table=block_tables,
                                    window_size=(-1, -1),
                                    s_aux=sink)

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                casual=False,
                                is_paged=True,
                                sink=sink,
                                window_size_left=-1,
                                window_size_right=-1)
    atol, rtol = 1e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    try:
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
            f"{torch.max(torch.abs(output - ref_output))}"
        print("✅ All implementations match, ", config)
    except AssertionError as e:
        print("❌ Implementations differ, ", config, " error: ", e)


def benchmark_decode_with_paged_kv(seq_lens, num_heads, head_size, block_size,
                                   output_dtype, soft_cap, num_blocks,
                                   fa_versions, q_dtype, is_sink, provider,
                                   iterations):
    maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, max_query_len, cu_query_lens, \
        max_kv_len, seq_k, scale, block_tables, sink, _, \
        _, _, _, _ = make_decode_with_paged_kv_input(
            config=(seq_lens, num_heads, head_size,
                    block_size, output_dtype, soft_cap,
                    num_blocks, fa_versions, q_dtype, is_sink))

    num_seqs = int(seq_lens.split(",")[0])
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    print(f"Running config: {seq_lens, num_heads, head_size, \
                              block_size, output_dtype, soft_cap, num_blocks, \
                              fa_versions, q_dtype, \
                              is_sink}, Provider: {provider}",
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

    if provider == "flash":
        for index in range(iterations):
            block_tables = torch.randint(0,
                                         num_blocks,
                                         (num_seqs, max_num_blocks_per_seq),
                                         dtype=torch.int32)
            start.record()
            flash_attn_varlen_func(queries[index],
                                   maybe_quantized_key_cache,
                                   maybe_quantized_value_cache,
                                   max_query_len,
                                   cu_query_lens,
                                   max_kv_len,
                                   seqused_k=seq_k,
                                   softmax_scale=scale,
                                   causal=False,
                                   block_table=block_tables,
                                   window_size=(-1, -1),
                                   s_aux=sink)
            end.record()
            end.synchronize()
            if index >= 5:  # skip the first 5 iterations for warmup
                total_latency += start.elapsed_time(end)
    else:
        for index in range(iterations):
            block_tables = torch.randint(0,
                                         num_blocks,
                                         (num_seqs, max_num_blocks_per_seq),
                                         dtype=torch.int32)
            flash_attn_varlen_func_CalKernelTime(queries[index],
                                                 maybe_quantized_key_cache,
                                                 maybe_quantized_value_cache,
                                                 max_query_len,
                                                 cu_query_lens,
                                                 max_kv_len,
                                                 seqused_k=seq_k,
                                                 softmax_scale=scale,
                                                 causal=False,
                                                 block_table=block_tables,
                                                 window_size=(-1, -1),
                                                 s_aux=sink,
                                                 start_event=start,
                                                 end_event=end)
            if index >= 5:  # skip the first 5 iterations for warmup
                total_latency += start.elapsed_time(end)
        if provider == "flash_memBandwidth":
            torch.xpu.synchronize()
            ms = total_latency / (iterations - 5)
            memory_load_GB = calculate_memory_usage(seq_k.sum().item(),
                                                    num_heads[1], head_size,
                                                    output_dtype)
            clear_xpu_cache()
            return memory_load_GB / (ms / 1000)
    torch.xpu.synchronize()
    ms = total_latency / (iterations - 5)
    clear_xpu_cache()

    return 1000 * ms


def get_benchmark_decode_with_paged_kv(iterations=20):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "seq_lens", "num_heads", "head_size", "block_size",
                "output_dtype", "soft_cap", "num_blocks", "fa_versions",
                "q_dtype", "is_sink"
            ],
            x_vals=[tuple(c) for c in configs],
            line_arg="provider",
            line_vals=["flash", "flash_kernelTime", "flash_memBandwidth"],
            line_names=[
                "FlashAttention(us)", "FlashAttention_kernelTime(us)",
                "FlashAttention_memBandwidth(GB/s)"
            ],
            styles=[("blue", "-"), ("green", "-"), ("purple", "-")],
            ylabel="Latency (us)",
            plot_name="flash-attn-decode",
            args={},
        ))
    def benchmark(seq_lens, num_heads, head_size, block_size, output_dtype,
                  soft_cap, num_blocks, fa_versions, q_dtype, is_sink,
                  provider):
        return benchmark_decode_with_paged_kv(seq_lens=seq_lens,
                                              num_heads=num_heads,
                                              head_size=head_size,
                                              block_size=block_size,
                                              output_dtype=output_dtype,
                                              soft_cap=soft_cap,
                                              num_blocks=num_blocks,
                                              fa_versions=fa_versions,
                                              q_dtype=q_dtype,
                                              is_sink=is_sink,
                                              provider=provider,
                                              iterations=iterations)

    return benchmark


def filter_configs(configs):
    new_configs = []
    for config in configs:
        if (config[1] == (16, 1) and config[2] == 256) or \
           (config[3] == 128 and config[6] == 32768 and config[2] >= 192):
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
            calculate_diff_decode_paged_kv(config)
        except Exception as e:
            print("Error in config: ", config, " error: ", e)
        clear_xpu_cache()

    configs = gen_perf_configs()
    configs = filter_configs(configs)
    benchmark = get_benchmark_decode_with_paged_kv(iterations=iterations)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
