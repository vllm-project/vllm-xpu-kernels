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
    gen_cutlass_flash_attn_varlen_correctness_configs as
    gen_correctness_config)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_varlen_perf_configs as gen_perf_configs)
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
    _, _, _, num_heads, head_size, block_size, window_size, output_dtype, \
        _, _, _, q_dtype, _, is_causal, is_paged, kv_dtype = config
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
        print("✅ Flash implementations match, ", config)
    except AssertionError as e:
        print("❌ Flash implementations differ, ", config, " error: ", e)

    # Triton correctness check (only for paged, causal, non-FP8)
    can_run_triton = (q_dtype is None and kv_dtype is None
                      and is_paged and is_causal)
    if can_run_triton:
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        seq_threshold_3D, num_par_softmax_segs, segm_out, segm_max, \
            segm_expsum = make_triton_softmax_buffers(
                num_query_heads, num_kv_heads, head_size)
        triton_output = torch.empty(query.shape[0], num_query_heads,
                                    head_size, dtype=output_dtype)
        unified_attention(
            q=query, k=maybe_quantized_key_cache,
            v=maybe_quantized_value_cache, out=triton_output,
            cu_seqlens_q=cu_query_lens, max_seqlen_q=max_query_len,
            seqused_k=seq_k, max_seqlen_k=max_kv_len,
            softmax_scale=scale, causal=True,
            window_size=window_size, block_table=block_tables,
            softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segs,
            softmax_segm_output=segm_out, softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
            sinks=sink, kv_quant_mode=KVQuantMode.NONE, use_td=True)
        try:
            torch.testing.assert_close(triton_output, ref_output,
                                       atol=atol, rtol=rtol)
            print("✅ Triton implementations match, ", config)
        except AssertionError as e:
            print("❌ Triton implementations differ, ", config, " error: ", e)


def benchmark_varlen_with_paged_kv(config, iterations=20):
    num_seqs, query_lens_str, kv_lens_str, num_heads, head_size, \
        block_size, window_size, output_dtype, soft_cap, num_blocks, \
        fa_versions, q_dtype, is_sink, is_causal, is_paged, kv_dtype = config
    maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, \
        max_query_len, cu_query_lens, max_kv_len, cu_kv_lens, \
        seq_k, q_descale, k_descale, v_descale, scale, is_causal, \
        block_tables, window_size, sink, scale_shape, _, \
        query_lens, kv_lens, _, _, max_num_blocks_per_seq = \
            make_varlen_with_paged_kv_input(config)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]

    print(f"Running config: {config}", flush=True)
    assert iterations > 5

    queries = [
        torch.rand_like(maybe_quantized_query) for _ in range(iterations)
    ]

    hardware_presets = get_hardware_preset(torch.xpu.get_device_name())

    # --- Flash Attention ---
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    if is_paged:
        for index in range(5):
            block_tables = torch.randint(
                0, num_blocks,
                (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
            flash_attn_varlen_func(queries[index],
                                   maybe_quantized_key_cache,
                                   maybe_quantized_value_cache,
                                   max_query_len, cu_query_lens, max_kv_len,
                                   seqused_k=seq_k,
                                   q_descale=q_descale.expand(scale_shape)
                                   if q_descale is not None else None,
                                   k_descale=k_descale.expand(scale_shape)
                                   if k_descale is not None else None,
                                   v_descale=v_descale.expand(scale_shape)
                                   if v_descale is not None else None,
                                   softmax_scale=scale, causal=is_causal,
                                   block_table=block_tables,
                                   window_size=window_size, s_aux=sink)
        start_event.record()
        for index in range(5, iterations):
            block_tables = torch.randint(
                0, num_blocks,
                (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
            flash_attn_varlen_func(queries[index],
                                   maybe_quantized_key_cache,
                                   maybe_quantized_value_cache,
                                   max_query_len, cu_query_lens, max_kv_len,
                                   seqused_k=seq_k,
                                   q_descale=q_descale.expand(scale_shape)
                                   if q_descale is not None else None,
                                   k_descale=k_descale.expand(scale_shape)
                                   if k_descale is not None else None,
                                   v_descale=v_descale.expand(scale_shape)
                                   if v_descale is not None else None,
                                   softmax_scale=scale, causal=is_causal,
                                   block_table=block_tables,
                                   window_size=window_size, s_aux=sink)
        end_event.record()
    else:
        for index in range(5):
            flash_attn_varlen_func(queries[index],
                                   maybe_quantized_key_cache,
                                   maybe_quantized_value_cache,
                                   max_query_len, cu_query_lens, max_kv_len,
                                   cu_seqlens_k=cu_kv_lens,
                                   q_descale=q_descale.expand(scale_shape)
                                   if q_descale is not None else None,
                                   k_descale=k_descale.expand(scale_shape)
                                   if k_descale is not None else None,
                                   v_descale=v_descale.expand(scale_shape)
                                   if v_descale is not None else None,
                                   softmax_scale=scale, causal=is_causal,
                                   block_table=None,
                                   window_size=window_size, s_aux=sink)
        start_event.record()
        for index in range(5, iterations):
            flash_attn_varlen_func(queries[index],
                                   maybe_quantized_key_cache,
                                   maybe_quantized_value_cache,
                                   max_query_len, cu_query_lens, max_kv_len,
                                   cu_seqlens_k=cu_kv_lens,
                                   q_descale=q_descale.expand(scale_shape)
                                   if q_descale is not None else None,
                                   k_descale=k_descale.expand(scale_shape)
                                   if k_descale is not None else None,
                                   v_descale=v_descale.expand(scale_shape)
                                   if v_descale is not None else None,
                                   softmax_scale=scale, causal=is_causal,
                                   block_table=None,
                                   window_size=window_size, s_aux=sink)
        end_event.record()
    torch.xpu.synchronize()
    flash_ms = start_event.elapsed_time(end_event) / (iterations - 5)
    flash_us = flash_ms * 1000

    flops = calculate_flops(num_query_heads, query_lens, kv_lens,
                            head_size, is_causal)
    flash_tflops = flops / (flash_ms / 1000) / 1e12
    flash_mfu = float("nan")
    if hardware_presets is not None:
        flash_mfu = (flash_tflops / hardware_presets["tflops"]) * 100

    clear_xpu_cache()

    # --- Triton Attention ---
    triton_us = float("nan")
    triton_tflops = float("nan")
    triton_mfu = float("nan")
    can_run_triton = (q_dtype is None and kv_dtype is None
                      and is_paged and is_causal)
    if can_run_triton:
        seq_threshold_3D, num_par_softmax_segs, segm_out, segm_max, \
            segm_expsum = make_triton_softmax_buffers(
                num_query_heads, num_kv_heads, head_size)
        output = torch.empty(maybe_quantized_query.shape[0],
                             num_query_heads, head_size, dtype=output_dtype)
        start_event = torch.xpu.Event(enable_timing=True)
        end_event = torch.xpu.Event(enable_timing=True)
        for index in range(5):
            block_tables = torch.randint(
                0, num_blocks,
                (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
            unified_attention(
                q=queries[index], k=maybe_quantized_key_cache,
                v=maybe_quantized_value_cache, out=output,
                cu_seqlens_q=cu_query_lens, max_seqlen_q=max_query_len,
                seqused_k=seq_k, max_seqlen_k=max_kv_len,
                softmax_scale=scale, causal=True,
                window_size=window_size, block_table=block_tables,
                softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
                seq_threshold_3D=seq_threshold_3D,
                num_par_softmax_segments=num_par_softmax_segs,
                softmax_segm_output=segm_out, softmax_segm_max=segm_max,
                softmax_segm_expsum=segm_expsum,
                sinks=sink, kv_quant_mode=KVQuantMode.NONE, use_td=True)
        start_event.record()
        for index in range(5, iterations):
            block_tables = torch.randint(
                0, num_blocks,
                (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
            unified_attention(
                q=queries[index], k=maybe_quantized_key_cache,
                v=maybe_quantized_value_cache, out=output,
                cu_seqlens_q=cu_query_lens, max_seqlen_q=max_query_len,
                seqused_k=seq_k, max_seqlen_k=max_kv_len,
                softmax_scale=scale, causal=True,
                window_size=window_size, block_table=block_tables,
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
        triton_tflops = flops / (triton_ms / 1000) / 1e12
        if hardware_presets is not None:
            triton_mfu = (triton_tflops / hardware_presets["tflops"]) * 100
        clear_xpu_cache()

    return {
        "num_seqs": num_seqs,
        "query_lens": query_lens_str,
        "kv_lens": kv_lens_str,
        "num_heads": num_heads,
        "head_size": head_size,
        "block_size": block_size,
        "window_size": window_size,
        "output_dtype": str(output_dtype),
        "q_dtype": str(q_dtype) if q_dtype else "",
        "kv_dtype": str(kv_dtype) if kv_dtype else "",
        "is_causal": is_causal,
        "is_paged": is_paged,
        "flash_latency(us)": f"{flash_us:.2f}",
        "flash_tflops": f"{flash_tflops:.2f}",
        "flash_mfu(%)": f"{flash_mfu:.2f}",
        "triton_latency(us)": f"{triton_us:.2f}",
        "triton_tflops": f"{triton_tflops:.2f}",
        "triton_mfu(%)": f"{triton_mfu:.2f}",
    }


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
    seed = 4242
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
    save_path = ensure_save_path_exists(args.save_path)

    results = []
    for config in configs:
        try:
            row = benchmark_varlen_with_paged_kv(config, iterations=iterations)
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
        csv_path = os.path.join(save_path, "flash-attn-varlen.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")
