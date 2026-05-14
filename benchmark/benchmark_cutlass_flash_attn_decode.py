# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

# isort: off
import gc

import csv
import os

import torch

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)

from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_decode_correctness_configs as
    gen_correctness_config)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_decode_perf_configs as gen_perf_configs)
from tests.flash_attn.test_flash_attn_varlen_func import ref_paged_attn
from tests.utils import parse_args, seed_everything
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
from benchmark.presets import get_hardware_preset
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode
from benchmark.triton_utils import make_triton_softmax_buffers
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


def benchmark_decode_with_paged_kv(config, iterations=20):
    seq_lens, num_heads, head_size, block_size, output_dtype, \
        soft_cap, num_blocks, fa_versions, q_dtype, is_sink = config
    maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, max_query_len, cu_query_lens, \
        max_kv_len, seq_k, scale, block_tables, sink, _, \
        _, _, _, _ = make_decode_with_paged_kv_input(config)

    num_seqs = int(seq_lens.split(",")[0])
    num_query_heads, num_kv_heads = num_heads
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    print(f"Running config: {config}", flush=True)
    assert iterations > 5

    queries = [
        torch.rand_like(maybe_quantized_query) for _ in range(iterations)
    ]

    hardware_presets = get_hardware_preset(torch.xpu.get_device_name())

    # --- Flash Attention ---
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    for index in range(5):
        block_tables = torch.randint(0, num_blocks,
                                     (num_seqs, max_num_blocks_per_seq),
                                     dtype=torch.int32)
        flash_attn_varlen_func(queries[index],
                               maybe_quantized_key_cache,
                               maybe_quantized_value_cache,
                               max_query_len, cu_query_lens, max_kv_len,
                               seqused_k=seq_k, softmax_scale=scale,
                               causal=False, block_table=block_tables,
                               window_size=(-1, -1), s_aux=sink)
    start_event.record()
    for index in range(5, iterations):
        block_tables = torch.randint(0, num_blocks,
                                     (num_seqs, max_num_blocks_per_seq),
                                     dtype=torch.int32)
        flash_attn_varlen_func(queries[index],
                               maybe_quantized_key_cache,
                               maybe_quantized_value_cache,
                               max_query_len, cu_query_lens, max_kv_len,
                               seqused_k=seq_k, softmax_scale=scale,
                               causal=False, block_table=block_tables,
                               window_size=(-1, -1), s_aux=sink)
    end_event.record()
    torch.xpu.synchronize()
    flash_ms = start_event.elapsed_time(end_event) / (iterations - 5)
    flash_us = flash_ms * 1000

    memory_load_GB = calculate_memory_usage(seq_k.sum().item(),
                                            num_kv_heads, head_size,
                                            output_dtype)
    flash_bw = memory_load_GB / (flash_ms / 1000)
    flash_mbu = float("nan")
    if hardware_presets is not None:
        flash_mbu = (flash_bw / hardware_presets["memory_bandwidth_GBs"]) * 100

    clear_xpu_cache()

    # --- Triton Attention ---
    triton_us = float("nan")
    triton_bw = float("nan")
    triton_mbu = float("nan")
    can_run_triton = (q_dtype is None)
    if can_run_triton:
        seq_threshold_3D, num_par_softmax_segs, segm_out, segm_max, \
            segm_expsum = make_triton_softmax_buffers(
                num_query_heads, num_kv_heads, head_size)
        output = torch.empty(maybe_quantized_query.shape[0],
                             num_query_heads, head_size, dtype=output_dtype)
        start_event = torch.xpu.Event(enable_timing=True)
        end_event = torch.xpu.Event(enable_timing=True)
        for index in range(5):
            block_tables = torch.randint(0, num_blocks,
                                         (num_seqs, max_num_blocks_per_seq),
                                         dtype=torch.int32)
            unified_attention(
                q=queries[index], k=maybe_quantized_key_cache,
                v=maybe_quantized_value_cache, out=output,
                cu_seqlens_q=cu_query_lens, max_seqlen_q=max_query_len,
                seqused_k=seq_k, max_seqlen_k=max_kv_len,
                softmax_scale=scale, causal=True,
                window_size=(-1, -1), block_table=block_tables,
                softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
                seq_threshold_3D=seq_threshold_3D,
                num_par_softmax_segments=num_par_softmax_segs,
                softmax_segm_output=segm_out, softmax_segm_max=segm_max,
                softmax_segm_expsum=segm_expsum,
                sinks=sink, kv_quant_mode=KVQuantMode.NONE, use_td=True)
        start_event.record()
        for index in range(5, iterations):
            block_tables = torch.randint(0, num_blocks,
                                         (num_seqs, max_num_blocks_per_seq),
                                         dtype=torch.int32)
            unified_attention(
                q=queries[index], k=maybe_quantized_key_cache,
                v=maybe_quantized_value_cache, out=output,
                cu_seqlens_q=cu_query_lens, max_seqlen_q=max_query_len,
                seqused_k=seq_k, max_seqlen_k=max_kv_len,
                softmax_scale=scale, causal=True,
                window_size=(-1, -1), block_table=block_tables,
                softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
                seq_threshold_3D=seq_threshold_3D,
                num_par_softmax_segments=num_par_softmax_segs,
                softmax_segm_output=segm_out, softmax_segm_max=segm_max,
                softmax_segm_expsum=segm_expsum,
                sinks=sink, kv_quant_mode=KVQuantMode.NONE, use_td=True)
        end_event.record()
        torch.xpu.synchronize()
        triton_ms = start_event.elapsed_time(end_event) / (iterations - 5)
        triton_us = triton_ms * 1000
        triton_bw = memory_load_GB / (triton_ms / 1000)
        if hardware_presets is not None:
            triton_mbu = (triton_bw /
                          hardware_presets["memory_bandwidth_GBs"]) * 100
        clear_xpu_cache()

    return {
        "seq_lens": seq_lens,
        "num_heads": num_heads,
        "head_size": head_size,
        "block_size": block_size,
        "output_dtype": str(output_dtype),
        "q_dtype": str(q_dtype) if q_dtype else "",
        "is_sink": is_sink,
        "flash_latency(us)": f"{flash_us:.2f}",
        "flash_bw(GB/s)": f"{flash_bw:.2f}",
        "flash_mbu(%)": f"{flash_mbu:.2f}",
        "triton_latency(us)": f"{triton_us:.2f}",
        "triton_bw(GB/s)": f"{triton_bw:.2f}",
        "triton_mbu(%)": f"{triton_mbu:.2f}",
    }


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
    save_path = ensure_save_path_exists(args.save_path)

    results = []
    for config in configs:
        try:
            row = benchmark_decode_with_paged_kv(config, iterations=iterations)
            results.append(row)
        except Exception as e:
            print(f"Error in config: {config}, error: {e}")
        clear_xpu_cache()

    # Print table
    if results:
        headers = list(results[0].keys())
        col_widths = [max(len(h), max(len(str(r[h])) for r in results))
                      for h in headers]
        header_line = " | ".join(
            h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "-+-".join("-" * w for w in col_widths)
        print(header_line)
        print(sep_line)
        for row in results:
            print(" | ".join(
                str(row[h]).ljust(w) for h, w in zip(headers, col_widths)))

        # Save CSV
        csv_path = os.path.join(save_path, "flash-attn-decode.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")
