# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional

import pytest
import torch
import math

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

NUM_HEADS = [(4, 4), (8, 2), (10, 2)]
HEAD_SIZES = [64, 128, 192, 256]
BLOCK_SIZES = [64]
DTYPES = [torch.bfloat16, torch.half]
QDTYPES = [None]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]
SOFT_CAPS = [None]
SLIDING_WINDOWS = [(-1, 127), (127, -1), (127, 127), (-1, -1)]
SINK = [False, True]
CASUAL = [False, True]
PAGED = [False, True]


def ref_paged_attn(query: torch.Tensor,
                   key_cache: torch.Tensor,
                   value_cache: torch.Tensor,
                   query_lens: list[int],
                   kv_lens: list[int],
                   block_tables: torch.Tensor,
                   scale: float,
                   window_size_left: Optional[int] = None,
                   window_size_right: Optional[int] = None,
                   soft_cap: Optional[float] = None,
                   is_paged: Optional[bool] = True,
                   casual: Optional[bool] = False,
                   sink: Optional[torch.Tensor] = None) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    if is_paged:
        _, block_size, num_kv_heads, head_size = key_cache.shape
    else:
        _, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    start_idx_kv = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        if is_paged:
            num_kv_blocks = (kv_len + block_size - 1) // block_size
            block_indices = block_tables[i, :num_kv_blocks]

            k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
            k = k[:kv_len]
            v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
            v = v[:kv_len]
        else:
            k = key_cache[start_idx_kv:start_idx_kv + kv_len]
            v = value_cache[start_idx_kv:start_idx_kv + kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1],
                                        dim=1).contiguous()
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1],
                                        dim=1).contiguous()
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if window_size_right > 0 or window_size_left > 0:
            if window_size_right < 0:
                window_size_right = max(kv_lens)
            if window_size_left < 0:
                window_size_left = max(kv_lens)

            mask_right = torch.triu(empty_mask,
                                    diagonal=kv_len - query_len +
                                    window_size_right + 1).bool()
            mask_left = torch.triu(empty_mask,
                                   diagonal=kv_len - query_len -
                                   window_size_left).bool().logical_not()
            mask_local = mask_right | mask_left
            attn.masked_fill_(mask_local, float("-inf"))
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if casual:
            attn.masked_fill_(mask, float("-inf"))
        if sink is not None:
            sink_expanded = sink.view(sink.size()[0], 1,
                                      1).expand(attn.size()[0],
                                                attn.size()[1], 1)
            attn = torch.cat([attn, sink_expanded], dim=-1)
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        if sink is not None:
            attn = attn[..., :-1]
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len
        start_idx_kv += kv_len

    return torch.cat(outputs, dim=0)


#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "seq_lens": [[(1, 1328), (5, 18), (129, 463)]],
        "head_size": [64, 128],
        "num_blocks": [64],
    }
}


@pytest.mark.parametrize("seq_lens",
                         [[(1, 1328), (5, 18),
                           (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("window_size", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("fa_version", [2])
@pytest.mark.parametrize("q_dtype", QDTYPES)
@pytest.mark.parametrize("is_sink", SINK)
@pytest.mark.parametrize("is_casual", CASUAL)
@pytest.mark.parametrize("is_paged", PAGED)
@torch.inference_mode()
def test_varlen_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    window_size: tuple[int, int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    fa_version: int,
    q_dtype: Optional[torch.dtype],
    is_sink: bool,
    is_casual: bool,
    is_paged: bool,
) -> None:
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")
    # # FIXME: remove skip
    if (is_casual and seq_lens[1][0]
            == 5) and (os.getenv("SKIP_HANG_KERNEL") is not None
                       and os.getenv("SKIP_HANG_KERNEL") == "1"):
        pytest.skip("skip casual for seqlen0 to avoid runtime hang on CI.")
    if (window_size[0] != -1 or window_size[1]
            != -1) and (os.getenv("SKIP_HANG_KERNEL") is not None
                        and os.getenv("SKIP_HANG_KERNEL") == "1"):
        pytest.skip("skip local attn to avoid runtime hang on CI.")
    # if q_dtype is not None and (dtype != torch.bfloat16 or fa_version == 2):
    #     pytest.skip("Flash attention with quantized inputs is only "
    #                 "supported on version 3 with bfloat16 base type")
    torch.manual_seed(42)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    if is_paged:
        key_cache = torch.randn(num_blocks,
                                block_size,
                                num_kv_heads,
                                head_size,
                                dtype=dtype)
    else:
        key_cache = torch.randn(sum(kv_lens),
                                num_query_heads,
                                head_size,
                                dtype=dtype)
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
        sink = torch.randn(num_query_heads, dtype=dtype)

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

    if is_paged:
        output = flash_attn_varlen_func(maybe_quantized_query,
                                        maybe_quantized_key_cache,
                                        maybe_quantized_value_cache,
                                        max_query_len,
                                        cu_query_lens,
                                        max_kv_len,
                                        seqused_k=seq_k,
                                        softmax_scale=scale,
                                        causal=is_casual,
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
                                        softmax_scale=scale,
                                        causal=is_casual,
                                        block_table=None,
                                        window_size=window_size,
                                        s_aux=sink)

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                casual=is_casual,
                                is_paged=is_paged,
                                sink=sink,
                                window_size_left=window_size[0],
                                window_size_right=window_size[1])
    atol, rtol = 1e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    if window_size[0] != -1 or window_size[1] != -1:
        atol, rtol = 1.5e-2, 1.5e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
        f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.parametrize("seq_lens",
                         [[(1, 1025)], [(1, 523), (1, 37), (1, 2011)],
                          [(1, 13000)], [(1, 523), (1, 37), (1, 2011), (1, 5000)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("fa_version", [2])
@pytest.mark.parametrize("q_dtype", QDTYPES)
@pytest.mark.parametrize("is_sink", SINK)
@pytest.mark.parametrize("do_performance", [False, True])
@pytest.mark.parametrize("num_splits_kv", [2])
@torch.inference_mode()
def test_decode_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    fa_version: int,
    q_dtype: Optional[torch.dtype],
    is_sink: bool,
    do_performance: bool,
    num_splits_kv: int,
) -> None:
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")
    # # FIXME: remove skip
    # if q_dtype is not None and (dtype != torch.bfloat16 or fa_version == 2):
    #     pytest.skip("Flash attention with quantized inputs is only "
    #                 "supported on version 3 with bfloat16 base type")
    torch.manual_seed(42)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
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
        sink = torch.randn(num_query_heads, dtype=dtype)

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
    # atol, rtol = 1e-2, 1e-2
    # if q_dtype is not None:
    #     atol, rtol = 1.5e-1, 1.5e-1
    # torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
    #     f"{torch.max(torch.abs(output - ref_output))}"
    # print("Test passed!")

    if do_performance:
        # warm up
        for _ in range(10):
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
        torch.xpu.synchronize()
        iterations = 200
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.XPU],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=30,
                active=100,
                repeat=1,
            ),
            acc_events=True,
        ) as prof:
            for _ in range(iterations):
                query = torch.randn(sum(query_lens),
                                    num_query_heads,
                                    head_size,
                                    dtype=dtype)
                block_tables = torch.randint(0,
                                             num_blocks,
                                             (num_seqs, max_num_blocks_per_seq),
                                             dtype=torch.int32)
                output = flash_attn_varlen_func(query,
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
                torch.xpu.synchronize()
                prof.step()
        print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=20))
        prof.export_chrome_trace("trace.json")

        kv_blocks = (max_kv_len + block_size - 1) // block_size
        memory_load_bytes = (num_seqs * kv_blocks * block_size * num_kv_heads * head_size * 2) * 2


        # print(f"num_seqs: {num_seqs},"
        #       f" kv_blocks: {kv_blocks},"
        #       f" block_size: {block_size},"
        #       f" num_kv_heads: {num_kv_heads},"
        #       f" head_size: {head_size},"
        #       f" num_splits_kv: {num_splits_kv},"
        #       f" memory_load_bytes: {memory_load_bytes} bytes")


def get_num_splits(batch, num_h, max_k, block_size):
    parallel_ = 20
    parallel_2 = 40

    cur_parallel = batch * num_h
    num_splits = int((parallel_ + cur_parallel - 1) / cur_parallel)

    # cur_parallel = cur_parallel * num_splits
    if cur_parallel * num_splits > parallel_ and num_splits > 1:
        # num_splits = int((parallel_2 + cur_parallel - 1) / cur_parallel)
        num_splits = math.ceil(parallel_2 / cur_parallel) - 1

    max_splits = int((max_k + block_size - 1) / block_size)
    max_splits = min(max_splits, 20)
    return min(num_splits, max_splits)

if __name__ == "__main__":
    batch = 50
    block_size = 128
    head_size = 128
    # for batch in range(14, 22):
    for batch in [32]:
        for head_pair in [(32, 8)]:
            for kv_len in [1500]:
                max_value = int((kv_len + 127) / 128)
                max_value = min(max_value, 20)
                # max_value = 10
                num_splits_kv = get_num_splits(batch, head_pair[1], kv_len, block_size)
                # for num_splits_kv in range(1, max_value + 1):
                for num_splits_kv in [1, num_splits_kv]:
                    print(f"batch={batch},"
                          f" kv_len={kv_len},"
                          f" heads={head_pair},"
                          f" num_splits_kv={num_splits_kv},"
                          f" head_size={head_size},"
                          f" block_size={block_size}")
                    test_decode_with_paged_kv([(1, kv_len)] * batch,
                                              head_pair,
                                              head_size,
                                              torch.bfloat16,
                                              block_size,
                                              None,
                                              32768,
                                              2,
                                              None,
                                              False,
                                              do_performance = True,
                                              num_splits_kv = num_splits_kv)
                print(f"==================kv_len: {kv_len}======================" * 2)
            print(f"==================head_pair: {head_pair}======================")
        print(f"==================batch: {batch}======================")
