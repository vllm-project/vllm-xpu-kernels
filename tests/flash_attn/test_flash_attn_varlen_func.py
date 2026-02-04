# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional

import pytest
import torch

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

NUM_HEADS = [(4, 4), (8, 2), (10, 2), (16, 1)]
HEAD_SIZES = [64, 128, 192, 256]
BLOCK_SIZES = [128]
DTYPES = [torch.bfloat16, torch.half]
QDTYPES = [None]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]
SOFT_CAPS = [None]
SLIDING_WINDOWS = [(-1, 127), (127, -1), (64, 64), (-1, -1)]
SINK = [False, True]
CASUAL = [False, True]
PAGED = [False, True]
FP8KV = [torch.float8_e5m2, torch.float8_e4m3fn, None]


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
                   sink: Optional[torch.Tensor] = None,
                   q_descale: Optional[torch.Tensor] = None,
                   k_descale: Optional[torch.Tensor] = None,
                   v_descale: Optional[torch.Tensor] = None,
                   is_fp8kv: bool = False,
                   is_fp8_query: bool = False,
                   dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    if is_paged:
        _, block_size, num_kv_heads, head_size = key_cache.shape
    else:
        _, num_kv_heads, head_size = key_cache.shape

    if is_fp8_query:
        query = (query.to(torch.float32) * q_descale).to(dtype)

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

        if is_fp8kv:
            k = (k.to(torch.float32) * k_descale).to(dtype)
            v = (v.to(torch.float32) * v_descale).to(dtype)
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
    "test_varlen_with_paged_kv": {
        "seq_lens": [[(1, 1328), (5, 18), (129, 463)]],
        "head_size": [64, 128],
        "num_heads": [(8, 2)],
        "num_blocks": [64],
        "window_size": [(-1, -1), (127, 127)],
        "is_paged": [True]
    },
    "test_decode_with_paged_kv": {
        "seq_lens": [[(1, 1025), (1, 523), (1, 37)]],
        "num_heads": [(8, 2)],
        "head_size": [64, 128],
        "num_blocks": [64],
    }
}


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
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
@pytest.mark.parametrize("fp8_dtype", FP8KV)
@torch.inference_mode()
def not_test_varlen_with_paged_kv(
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
    fp8_dtype: Optional[torch.dtype],
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
    torch.manual_seed(4242)
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
    scale_shape = (num_seqs, num_kv_heads)
    is_fp8_query = q_dtype is not None
    if is_fp8_query:
        q_descale = (torch.abs(query).max() / 200).to(torch.float32)
        maybe_quantized_query = (query / q_descale).to(q_dtype)
    is_fp8kv = fp8_dtype is not None
    if is_fp8kv:
        k_descale = (torch.abs(key_cache).max() / 200).to(torch.float32)
        v_descale = (torch.abs(value_cache).max() / 200).to(torch.float32)
        maybe_quantized_key_cache = (key_cache / k_descale).to(fp8_dtype)
        maybe_quantized_value_cache = (value_cache / v_descale).to(fp8_dtype)

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
                                        q_descale=q_descale.expand(scale_shape)
                                        if q_descale is not None else None,
                                        k_descale=k_descale.expand(scale_shape)
                                        if k_descale is not None else None,
                                        v_descale=v_descale.expand(scale_shape)
                                        if v_descale is not None else None,
                                        softmax_scale=scale,
                                        causal=is_casual,
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
                                casual=is_casual,
                                is_paged=is_paged,
                                sink=sink,
                                q_descale=q_descale,
                                k_descale=k_descale,
                                v_descale=v_descale,
                                window_size_left=window_size[0],
                                window_size_right=window_size[1],
                                is_fp8kv=is_fp8kv,
                                is_fp8_query=is_fp8_query,
                                dtype=dtype)
    atol, rtol = 1e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    if window_size[0] != -1 or window_size[1] != -1:
        atol, rtol = 1.5e-2, 1.5e-2
    if fp8_dtype is not None:
        atol, rtol = 1.5e-2, 1.5e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
        f"{torch.max(torch.abs(output - ref_output))}"
    torch.xpu.empty_cache()


@pytest.mark.parametrize("seq_lens",
                         [[(1, 1025)], [(1, 523), (1, 37),
                                        (1, 2011)], [(1, 13000)],
                          [(1, 523), (1, 37), (1, 2011), (1, 5000)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("fa_version", [2])
@pytest.mark.parametrize("q_dtype", QDTYPES)
@pytest.mark.parametrize("is_sink", SINK)
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
    atol, rtol = 1e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
        f"{torch.max(torch.abs(output - ref_output))}"
    torch.xpu.empty_cache()
