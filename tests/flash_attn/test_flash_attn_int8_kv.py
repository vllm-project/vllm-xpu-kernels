# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 per-token-head KV cache (vLLM `int8_per_token_head`) for paged FA2.

The cache uses the block-local split layout: per block
[int8 K|V data for all (slot, head) | K scales fp32 | V scales fp32], i.e. the
same bytes per block as vLLM's inline per-token-head layout. INT8 KV is
supported by the paged decode kernel, including uniform speculative-decode
batches (which the interface runs through the decode kernel); prefill / mixed
batches are rejected (callers dequantize the referenced blocks and use the
fp16/bf16 path). The kernel output is compared with an fp32 reference
computed on the same quantized cache.
"""
import pytest
import torch

from tests.utils import format_tc
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

NUM_HEADS = [(8, 8), (16, 4), (32, 4), (24, 2)]
HEAD_SIZES = [64, 128, 192, 256, 512]
BLOCK_SIZES = [16, 32, 64, 128]
DTYPES = [torch.half, torch.bfloat16]
SLIDING_WINDOWS = [None, 128]
SINK = [False, True]
MODES = ["decode", "spec_decode"]

MINI_PYTEST_PARAMS = {
    "default": {
        "num_heads": [(16, 4)],
        "head_size": [128, 256],
        "block_size": [64],
        "window": [None],
    },
}


def quantize_per_token_head(x: torch.Tensor):
    """vLLM int8_per_token_head write semantics (fp32): scale = absmax / 127,
    round half away from zero, clamp to [-128, 127]."""
    xf = x.float()
    scale = torch.clamp(xf.abs().amax(-1) / 127.0, min=1e-6)
    y = xf * (1.0 / scale)[..., None]
    y = torch.where(y >= 0, y + 0.5, y - 0.5).clamp(-128.0, 127.0).trunc()
    return y.to(torch.int8), scale


def split_layout_views(raw, block_size, num_kv_heads, head_size):
    """K/V int8 views [nb, B, H, D] and scale views [nb, B, H] over raw
    [num_blocks, page_bytes] uint8."""
    nb = raw.shape[0]
    data = block_size * num_kv_heads * 2 * head_size
    page = raw.shape[1]
    assert page == 2 * (head_size + 4) * num_kv_heads * block_size
    d = torch.as_strided(raw.view(torch.int8),
                         (nb, block_size, num_kv_heads, 2 * head_size),
                         (page, num_kv_heads * 2 * head_size, 2 * head_size,
                          1))
    f = raw.view(torch.float32)
    sb = block_size * num_kv_heads * 4

    def scales(off):
        return torch.as_strided(f, (nb, block_size, num_kv_heads),
                                (page // 4, num_kv_heads, 1),
                                f.storage_offset() + off // 4)

    return (d[..., :head_size], d[..., head_size:], scales(data),
            scales(data + sb))


def ref_attn(q, k, v, scale, q_pos, window, sinks):
    g = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    s = torch.einsum("qhd,khd->hqk", q, k) * scale
    kpos = torch.arange(k.shape[0], device=q.device)
    ok = kpos[None, :] <= q_pos[:, None]
    if window:
        ok &= (q_pos[:, None] - kpos[None, :]) < window
    s = s.masked_fill(~ok[None], float("-inf"))
    if sinks is not None:
        m = torch.maximum(s.amax(-1), sinks[:, None].float())
        p = torch.exp(s - m[..., None])
        p = p / (p.sum(-1) + torch.exp(sinks[:, None].float() - m))[..., None]
    else:
        p = torch.softmax(s, -1)
    return torch.einsum("hqk,khd->qhd", p, v)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("window", SLIDING_WINDOWS)
@pytest.mark.parametrize("is_sink", SINK)
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@torch.inference_mode()
def test_varlen_int8_kv(mode, num_heads, head_size, block_size, window,
                        is_sink, dtype):
    torch.manual_seed(0)
    dev = "xpu"
    hq, hkv = num_heads
    # spec_decode: every sequence has the same q_len (<= seq_len).
    seq_lens = [17, 64, 200, 1 if mode == "decode" else 3, 129]
    q_lens = {
        "decode": [1] * 5,
        "spec_decode": [3] * 5,
    }[mode]
    nblk = [(L + block_size - 1) // block_size for L in seq_lens]
    num_blocks = sum(nblk) + 3
    page = 2 * (head_size + 4) * hkv * block_size
    raw = torch.full((num_blocks, page), 0x7F, dtype=torch.uint8, device=dev)
    kc, vc, ks, vs = split_layout_views(raw, block_size, hkv, head_size)
    perm = (torch.randperm(num_blocks - 1, device=dev) + 1).to(torch.int32)
    block_table = torch.zeros(len(seq_lens), max(nblk), dtype=torch.int32,
                              device=dev)
    keys, values, c = [], [], 0
    for i, (L, n) in enumerate(zip(seq_lens, nblk)):
        block_table[i, :n] = perm[c:c + n]
        c += n
        key = (torch.randn(L, hkv, head_size, device=dev) * 2).to(dtype)
        key[:, :, 7] *= 20  # outlier channel, as in real K
        val = torch.randn(L, hkv, head_size, device=dev).to(dtype)
        pos = torch.arange(L, device=dev)
        b = block_table[i, pos // block_size].long()
        o = pos % block_size
        kq, kscale = quantize_per_token_head(key)
        vq, vscale = quantize_per_token_head(val)
        kc[b, o], vc[b, o], ks[b, o], vs[b, o] = kq, vq, kscale, vscale
        keys.append(kq.float() * kscale[..., None])
        values.append(vq.float() * vscale[..., None])

    q = torch.randn(sum(q_lens), hq, head_size, device=dev).to(dtype)
    cu_q = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(),
                        dtype=torch.int32, device=dev)
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    sinks = (torch.randn(hq, device=dev) * 2).to(dtype) if is_sink else None
    scale = head_size**-0.5
    out = torch.empty_like(q)
    flash_attn_varlen_func(q,
                           kc,
                           vc,
                           max(q_lens),
                           cu_q,
                           max(seq_lens),
                           seqused_k=seqused_k,
                           softmax_scale=scale,
                           causal=True,
                           window_size=[window - 1, 0] if window else None,
                           block_table=block_table,
                           out=out,
                           k_descale=ks,
                           v_descale=vs,
                           s_aux=sinks)
    ref = []
    for i, (L, ql) in enumerate(zip(seq_lens, q_lens)):
        q0 = int(cu_q[i])
        ref.append(
            ref_attn(q[q0:q0 + ql].float(), keys[i], values[i], scale,
                     torch.arange(L - ql, L, device=dev), window, sinks))
    torch.testing.assert_close(out.float(),
                               torch.cat(ref),
                               atol=1.5e-2,
                               rtol=1.5e-2)


@pytest.mark.parametrize("q_lens", [[33], [1, 4]])
def test_varlen_int8_kv_rejects_prefill(q_lens):
    """Batches routed to chunk prefill (non-uniform or long q) are rejected."""
    dev = "xpu"
    hq, hkv, head_size, block_size = 16, 4, 128, 64
    page = 2 * (head_size + 4) * hkv * block_size
    raw = torch.zeros((4, page), dtype=torch.uint8, device=dev)
    kc, vc, ks, vs = split_layout_views(raw, block_size, hkv, head_size)
    n = len(q_lens)
    q = torch.randn(sum(q_lens), hq, head_size, device=dev, dtype=torch.half)
    cu_q = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(),
                        dtype=torch.int32,
                        device=dev)
    with pytest.raises(RuntimeError, match="decode"):
        flash_attn_varlen_func(q,
                               kc,
                               vc,
                               max(q_lens),
                               cu_q,
                               64,
                               seqused_k=torch.full((n, ),
                                                    64,
                                                    dtype=torch.int32,
                                                    device=dev),
                               causal=True,
                               block_table=torch.ones((n, 1),
                                                      dtype=torch.int32,
                                                      device=dev),
                               k_descale=ks,
                               v_descale=vs)
