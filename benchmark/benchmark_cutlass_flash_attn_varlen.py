# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import gc
import torch
import triton.testing

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
from tests.utils import parse_args


DEVICE = "xpu"


def clear_xpu_cache():
    torch.xpu.empty_cache()
    gc.collect()
    torch.xpu.synchronize()


import torch
import math

def naive_varlen_attention(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=False,
    softmax_scale=None,
    out=None,
):
    """
    q, k, v: [total_tokens, num_heads, head_dim]
    cu_seqlens_q/k: [batch+1]
    """

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    total_tokens, nheads, d = q.shape
    device = q.device
    dtype = q.dtype

    if out is None:
        out = torch.empty_like(q)

    batch = cu_seqlens_q.numel() - 1

    for b in range(batch):
        qs, qe = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        ks, ke = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

        q_b = q[qs:qe]      # [Lq, H, D]
        k_b = k[ks:ke]      # [Lk, H, D]
        v_b = v[ks:ke]      # [Lk, H, D]

        # transpose for matmul
        # q: [H, Lq, D], k: [H, D, Lk]
        qh = q_b.transpose(0, 1)
        kh = k_b.transpose(0, 1)
        vh = v_b.transpose(0, 1)

        # attention scores
        attn = torch.matmul(qh, kh.transpose(-2, -1)) * softmax_scale
        # [H, Lq, Lk]

        if causal:
            Lq, Lk = attn.shape[-2:]
            causal_mask = torch.triu(
                torch.ones(Lq, Lk, device=device, dtype=torch.bool),
                diagonal=1,
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)

        out_b = torch.matmul(attn, vh)  # [H, Lq, D]
        out[qs:qe] = out_b.transpose(0, 1)

    return out


def fa_varlen_attention_naive(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=False,
    softmax_scale=None,
    block_q=64,
    block_k=64,
    out=None,
):
    """
    FlashAttention-style PyTorch naive implementation (varlen).

    q, k, v: [total_tokens, num_heads, head_dim]
    cu_seqlens_q/k: [batch + 1]
    """

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    device = q.device
    dtype = q.dtype
    total_tokens, nheads, d = q.shape
    batch = cu_seqlens_q.numel() - 1

    if out is None:
        out = torch.empty_like(q)

    for b in range(batch):
        qs, qe = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        ks, ke = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

        q_b = q[qs:qe]   # [Lq, H, D]
        k_b = k[ks:ke]   # [Lk, H, D]
        v_b = v[ks:ke]   # [Lk, H, D]

        Lq = qe - qs
        Lk = ke - ks

        # transpose to [H, L, D]
        qh = q_b.transpose(0, 1)
        kh = k_b.transpose(0, 1)
        vh = v_b.transpose(0, 1)

        out_h = torch.empty((nheads, Lq, d), device=device, dtype=dtype)

        for h in range(nheads):
            qh_h = qh[h]   # [Lq, D]
            kh_h = kh[h]   # [Lk, D]
            vh_h = vh[h]   # [Lk, D]

            for q0 in range(0, Lq, block_q):
                q1 = min(q0 + block_q, Lq)
                q_blk = qh_h[q0:q1]  # [BQ, D]

                # online softmax state
                m = torch.full((q1 - q0,), -float("inf"),
                               device=device, dtype=dtype)
                l = torch.zeros((q1 - q0,),
                                device=device, dtype=dtype)
                o = torch.zeros((q1 - q0, d),
                                device=device, dtype=dtype)

                for k0 in range(0, Lk, block_k):
                    k1 = min(k0 + block_k, Lk)

                    # causal mask
                    if causal and k0 >= q1:
                        break

                    k_blk = kh_h[k0:k1]   # [BK, D]
                    v_blk = vh_h[k0:k1]   # [BK, D]

                    scores = (q_blk @ k_blk.T) * softmax_scale  # [BQ, BK]

                    if causal:
                        qi = torch.arange(q0, q1, device=device).unsqueeze(1)
                        ki = torch.arange(k0, k1, device=device).unsqueeze(0)
                        scores = scores.masked_fill(ki > qi, -float("inf"))

                    m_new = torch.maximum(m, scores.max(dim=-1).values)
                    exp_scores = torch.exp(scores - m_new[:, None])

                    l = l * torch.exp(m - m_new) + exp_scores.sum(dim=-1)
                    o = o * torch.exp(m - m_new)[:, None] + exp_scores @ v_blk
                    m = m_new

                out_h[h, q0:q1] = o / l[:, None]

        out[qs:qe] = out_h.transpose(0, 1)

    return out


def make_varlen_inputs(
    batch,
    max_seqlen,
    num_heads,
    head_dim,
    dtype,
    device=DEVICE,
):
    seqlens = torch.randint(
        low=max_seqlen // 4,
        high=max_seqlen,
        size=(batch,),
        device=device,
    )

    cu_seqlens = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)

    total_tokens = cu_seqlens[-1].item()

    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    return q, k, v, cu_seqlens, seqlens.max().item()


def calculate_diff(config):
    batch, max_seqlen, num_heads, head_dim, dtype = config
    q, k, v, cu_seqlens, max_seqlen_real = make_varlen_inputs(
        batch, max_seqlen, num_heads, head_dim, dtype
    )

    out_naive = fa_varlen_attention_naive(
        q, k, v,
        cu_seqlens, cu_seqlens,
        max_seqlen_real, max_seqlen_real,
        causal=True,
    )
    out_vllm = flash_attn_varlen_func(
        q, k, v,
        max_seqlen_real,
        cu_seqlens,
        max_seqlen_real,
        cu_seqlens,
        causal=True,
    )
    if torch.allclose(out_naive, out_vllm, atol=1e-3, rtol=0):
        print("✅ All implementations match, ", config)
    else:
        print("❌ Implementations differ, ", config)


def benchmark_flash_vs_naive(
    batch,
    max_seqlen,
    num_heads,
    head_dim,
    dtype,
    provider,
):
    q, k, v, cu_seqlens, max_seqlen_real = make_varlen_inputs(
        batch, max_seqlen, num_heads, head_dim, dtype
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "naive":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fa_varlen_attention_naive(
                q, k, v,
                cu_seqlens, cu_seqlens,
                max_seqlen_real,
                max_seqlen_real,
                causal=True,
            ),
            quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attn_varlen_func(
                q, k, v,
                max_seqlen_real,
                cu_seqlens,
                max_seqlen_real,
                cu_seqlens,
                causal=True,
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch", "max_seqlen", "num_heads", "head_dim", "dtype"],
            x_vals=[tuple(c) for c in configs],
            line_arg="provider",
            line_vals=["naive", "flash"],
            line_names=["Naive", "FlashAttention"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="Latency (us)",
            plot_name="flash-attn-varlen-vs-naive",
            args={},
        )
    )
    def benchmark(batch, max_seqlen, num_heads, head_dim, dtype, provider):
        return benchmark_flash_vs_naive(
            batch=batch,
            max_seqlen=max_seqlen,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            provider=provider,
        )

    return benchmark


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)

    batch = [1, 4, 8]
    max_seqlen = [16, 512, 1024, 2048]
    num_heads = [16]
    head_dim = [128]
    dtype = [torch.float16, torch.bfloat16]  # float32 not support, VLLM Kernel XPU only supports fp16 and bf16 type
    print("Final configuration:")
    print(f"  batch: {batch}")
    print(f"  max_seqlen: {max_seqlen}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  dtype: {dtype}")

    configs = list(
        itertools.product(batch, max_seqlen, num_heads, head_dim, dtype))

    for config in configs:
       try:
           calculate_diff(config)
       except RuntimeError as e:
           clear_xpu_cache()
           print(f"Error in config {config}: {e}")

    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)