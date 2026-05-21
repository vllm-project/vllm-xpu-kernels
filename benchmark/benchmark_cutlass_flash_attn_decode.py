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
    gen_cutlass_flash_attn_decode_correctness_configs as
    gen_correctness_config)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_decode_perf_configs as gen_perf_configs)
from tests.flash_attn.test_flash_attn_varlen_func import ref_paged_attn
from tests.utils import parse_args, seed_everything
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
from benchmark.presets import get_hardware_preset
# isort: on

DEVICE = "xpu"


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


def calculate_memory_usage(q_len_sum, kv_len_sum, num_heads, head_size,
                           query_dtype, kv_dtype, output_dtype):
    # Memory for query, key and value caches, and output
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    query_memory = q_len_sum * num_query_heads * head_size * \
        torch.tensor([], dtype=query_dtype).element_size()
    kv_cache_memory = 2 * kv_len_sum * num_kv_heads * \
        head_size * torch.tensor([], dtype=kv_dtype).element_size()
    output_memory = q_len_sum * num_query_heads * head_size * \
        torch.tensor([], dtype=output_dtype).element_size()
    # Convert to GB
    return (query_memory + kv_cache_memory + output_memory) / (1000**3)


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
    queries = [
        torch.rand_like(maybe_quantized_query) for _ in range(iterations)
    ]

    if provider == "flash":
        start_event = torch.xpu.Event(enable_timing=True)
        end_event = torch.xpu.Event(enable_timing=True)
        for index in range(5):
            block_tables = torch.randint(0,
                                         num_blocks,
                                         (num_seqs, max_num_blocks_per_seq),
                                         dtype=torch.int32)
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
        start_event.record()
        for index in range(5, iterations):
            block_tables = torch.randint(0,
                                         num_blocks,
                                         (num_seqs, max_num_blocks_per_seq),
                                         dtype=torch.int32)
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
        end_event.record()
        torch.xpu.synchronize()
        ms = start_event.elapsed_time(end_event) / (iterations - 5)
        clear_xpu_cache()
        return 1000 * ms
    else:
        start_events = [
            torch.xpu.Event(enable_timing=True)
            for _ in range(iterations - 5)
        ]
        end_events = [
            torch.xpu.Event(enable_timing=True)
            for _ in range(iterations - 5)
        ]
        for index in range(iterations):
            block_tables = torch.randint(0,
                                         num_blocks,
                                         (num_seqs, max_num_blocks_per_seq),
                                         dtype=torch.int32)
            se = start_events[index - 5] if index >= 5 else None
            ee = end_events[index - 5] if index >= 5 else None
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
                                                 start_event=se,
                                                 end_event=ee)
        torch.xpu.synchronize()
        total_latency = sum(
            start_events[i].elapsed_time(end_events[i])
            for i in range(iterations - 5)
        )
        ms = total_latency / (iterations - 5)
        if provider == "flash_memBandwidth" or provider == "flash_MBU":
            memory_load_GB = calculate_memory_usage(cu_query_lens[-1].item(),
                                                    seq_k.sum().item(),
                                                    num_heads, head_size,
                                                    queries[5].dtype,
                                                    maybe_quantized_key_cache.dtype,
                                                    output_dtype)
            measured_bw = memory_load_GB / (ms / 1000)
            if provider == "flash_MBU":
                hardware_presets = get_hardware_preset(
                    torch.xpu.get_device_name())
                if hardware_presets is None:
                    clear_xpu_cache()
                    return float("nan")
                peak_bw = hardware_presets["memory_bandwidth_GBs"]
                clear_xpu_cache()
                return (measured_bw / peak_bw) * 100
            clear_xpu_cache()
            return measured_bw
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
            line_vals=["flash", "flash_kernelTime", "flash_memBandwidth",
                       "flash_MBU"],
            line_names=[
                "FlashAttention(us)", "FlashAttention_kernelTime(us)",
                "FlashAttention_memBandwidth(GB/s)", "FlashAttention_MBU (%)"
            ],
            styles=[("blue", "-"), ("green", "-"), ("purple", "-"),
                    ("red", "-")],
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


def _mk_cfg(seq_lens, num_heads, block_size, name, head_size=128,
            dtype=torch.bfloat16, num_blocks=2048):
    # 11-tuple matching make_decode_with_paged_kv_input contract.
    return (seq_lens, num_heads, head_size, block_size, dtype, None,
            num_blocks, 2, None, False, name)


# Format: seq_lens="B,1+1+...,kv0+kv1+...", num_heads=(q, kv), block_size, name
BATCH_DECODE_CONFIGS = [
    # Uniform KV
    _mk_cfg("32," + "+".join(["1"] * 32) + "," + "+".join(["512"] * 32),
            (32, 2), 64, "32x512_uniform_(32,2)"),
    _mk_cfg("32," + "+".join(["1"] * 32) + "," + "+".join(["4096"] * 32),
            (32, 2), 64, "32x4096_uniform_(32,2)"),
    # Mixed KV (key optimization target)
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (32, 2), 64, "8xmixed_128-16384_(32,2)"),
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (32, 4), 64, "8xmixed_128-16384_(32,4)"),
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (32, 8), 64, "8xmixed_128-16384_(32,8)"),
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (40, 8), 64, "8xmixed_128-16384_(40,8)"),
    # Skewed (mostly short + one long)
    _mk_cfg("8,1+1+1+1+1+1+1+1,256+256+256+256+256+256+256+16384",
            (32, 2), 64, "8xskewed_256-16384_(32,2)"),
    # All short
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+128+256+256+512+512+1024+1024",
            (32, 2), 64, "8xshort_128-1024_(32,2)"),
    # Realistic vLLM-like
    _mk_cfg(
        "16," + "+".join(["1"] * 16) + ","
        + "+".join(["256", "512", "1024", "1024",
                    "2048", "2048", "4096", "4096",
                    "4096", "8192", "8192", "8192",
                    "16384", "16384", "16384", "16384"]),
        (32, 2), 64, "16xrealistic_mixed_(32,2)"),
    # block_size=128
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (32, 2), 128, "8xmixed_128-16384_bs128_(32,2)"),
    # MHA
    _mk_cfg("32," + "+".join(["1"] * 32) + "," + "+".join(["512"] * 32),
            (16, 16), 64, "32x512_uniform_(16,16)_MHA"),
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (16, 16), 64, "8xmixed_128-16384_(16,16)_MHA"),
]


def benchmark_batch_decode(config, iterations=200):
    """Benchmark a single batch decode config with GPU-event timing."""
    (seq_lens, num_heads, head_size, block_size, dtype, soft_cap,
     num_blocks, fa_versions, q_dtype, is_sink, name) = config

    full_config = (seq_lens, num_heads, head_size, block_size, dtype,
                   soft_cap, num_blocks, fa_versions, q_dtype, is_sink)
    (maybe_quantized_query, maybe_quantized_key_cache,
     maybe_quantized_value_cache, max_query_len, cu_query_lens,
     max_kv_len, seq_k, scale, block_tables, sink, _,
     _, _, _, _) = make_decode_with_paged_kv_input(full_config)

    num_seqs = int(seq_lens.split(",")[0])
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    queries = [torch.rand_like(maybe_quantized_query)
               for _ in range(iterations)]
    bt_list = [torch.randint(0, num_blocks,
                             (num_seqs, max_num_blocks_per_seq),
                             dtype=torch.int32)
               for _ in range(iterations)]

    def _run(i):
        flash_attn_varlen_func(
            queries[i], maybe_quantized_key_cache,
            maybe_quantized_value_cache,
            max_query_len, cu_query_lens, max_kv_len,
            seqused_k=seq_k, softmax_scale=scale,
            causal=False, block_table=bt_list[i],
            window_size=(-1, -1), s_aux=sink)

    # Warmup
    for i in range(min(10, iterations)):
        _run(i)
    torch.xpu.synchronize()

    # Timed
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    measured = iterations - 10
    start_event.record()
    for i in range(10, iterations):
        _run(i)
    end_event.record()
    torch.xpu.synchronize()

    avg_us = start_event.elapsed_time(end_event) * 1000.0 / measured

    # KV bandwidth (K + V, bf16 -> 2 bytes)
    kv_lens = list(map(int, seq_lens.split(",")[2].split("+")))
    kv_bytes = sum(kv_lens) * num_heads[1] * head_size * 2 * 2
    bw_gbs = (kv_bytes / 1e9) / (avg_us / 1e6)

    clear_xpu_cache()
    return avg_us, bw_gbs


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
    save_path = ensure_save_path_exists(args.save_path)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=save_path)

    # ================================================================
    # Batch Decode Benchmark (per-seq adaptive split-K evaluation)
    # ================================================================
    print("\n" + "=" * 80)
    print("Batch Decode Benchmark (per-seq adaptive split-K)")
    print("=" * 80)
    hdr = (f"{'config':<40} | {'batch':>5} {'kv_sum':>7} | "
           f"{'time(us)':>9} {'BW(GB/s)':>9}")
    print(hdr)
    print("-" * 80)

    for cfg in BATCH_DECODE_CONFIGS:
        name = cfg[-1]
        seq_lens = cfg[0]
        num_seqs = int(seq_lens.split(",")[0])
        kv_lens = list(map(int, seq_lens.split(",")[2].split("+")))
        kv_sum = sum(kv_lens)
        try:
            avg_us, bw_gbs = benchmark_batch_decode(cfg, iterations=200)
            print(f"{name:<40} | {num_seqs:>5} {kv_sum:>7} | "
                  f"{avg_us:>9.1f} {bw_gbs:>9.1f}")
        except Exception as e:
            print(f"{name:<40} | {num_seqs:>5} {kv_sum:>7} | "
                  f"{'ERROR':>9} {str(e)[:20]}")
        clear_xpu_cache()

    print("=" * 80)
