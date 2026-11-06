# SPDX-License-Identifier: Apache-2.0
"""Single-card numerics of the mono decoder layers (all-reduces through a
loopback mailbox): the batched kernels (2 / 4 / 8 rows) against the one-row
kernel row by row, the speculative-decoding verify kernels (requests x T
tokens) against the one-row kernel run token by token from the same state, in
fp16 and bf16, and the bf16 kernels against the fp16 ones on the same
values."""

import itertools

import pytest
import torch

from tests.mono_decode.mono_utils import (CONVD, HD, addr, attn_tensors, cos,
                                          gdn_tensors, mailbox, requires_mono,
                                          rows)

DEVICE = "xpu:0"
SPEC_SHAPES = [(2, 2), (3, 3), (4, 2), (4, 4), (6, 3), (8, 2), (8, 4), (8, 8)]
F16, BF16 = torch.float16, torch.bfloat16
DTYPES = [F16, BF16]
_ids = itertools.count(9000)


def _register(attn: bool, tensors: list) -> tuple[int, torch.Tensor]:
    layer, mb = next(_ids), mailbox(DEVICE)
    torch.ops._xpu_C.mono_register_layer(layer, attn, addr(mb),
                                         addr(mb), tensors)
    return layer, mb


def _run(layer, h, r, positions, m0, m1=None, m2=None):
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    torch.ops._xpu_C.mono_decode_layer(layer, h, r, ho, ro, positions, m0, m1,
                                       m2)
    return ho, ro


def _check(got, ref, tol):
    c = cos(got, ref)
    assert torch.isfinite(got).all() and c > tol, f"cos {c:.7f}"


@requires_mono
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("nb", [2, 4, 8])
def test_gdn_batched_vs_single_row(nb, dtype):
    gen = torch.Generator().manual_seed(nb)
    t = gdn_tensors(gen, DEVICE, slots=2 * nb, dtype=dtype)
    conv, ssm = t[22], t[23]
    conv0, ssm0 = conv.clone(), ssm.clone()
    h, r = rows(nb, gen, DEVICE, dtype)
    layer, _mb = _register(False, t)
    ref_layer, _mb1 = _register(False, t)
    # reference: row b alone on slot nb + b (the same initial state)
    conv[nb:] = conv0[:nb]
    ssm[nb:] = ssm0[:nb]
    ref = [
        _run(ref_layer, h[b:b + 1], r[b:b + 1],
             torch.zeros(1, dtype=torch.int32, device=DEVICE),
             torch.tensor([nb + b], dtype=torch.int32, device=DEVICE))
        for b in range(nb)
    ]
    ho, ro = _run(layer, h, r, torch.zeros(1, dtype=torch.int32,
                                           device=DEVICE),
                  torch.arange(nb, dtype=torch.int32, device=DEVICE))
    torch.xpu.synchronize()
    for b in range(nb):
        _check(ho[b], ref[b][0][0], 0.9999)
        _check(ro[b], ref[b][1][0], 0.99999)
        _check(ssm[b], ssm[nb + b], 0.99999)
        assert torch.equal(conv[b, :2], conv[nb + b, :2])  # shifted window
        _check(conv[b, 2], conv[nb + b, 2], 0.99999)  # the new input


def _attn_case(gen, lc, tokens, block_size=64):
    """Request b: lc[b] cached tokens in its own blocks (shuffled block ids),
    `tokens` new tokens after them. Returns block table, context lengths,
    slots, positions, number of blocks."""
    nreq = len(lc)
    nbr = (max(lc) + tokens + block_size - 1) // block_size + 1
    perm = torch.randperm(nreq * nbr, generator=gen).to(torch.int32)
    bt = perm.view(nreq, nbr)
    slot, pos = [], []
    for b in range(nreq):
        for i in range(tokens):
            tk = lc[b] + i
            slot.append(int(bt[b, tk // block_size]) * block_size +
                        tk % block_size)
            pos.append(tk)
    return (bt.to(DEVICE), lc, torch.tensor(slot, device=DEVICE),
            torch.tensor(pos, device=DEVICE), nreq * nbr)


# cached tokens per request: unaligned lengths, different per request (the
# step's total decides static chunks or dynamic units: >= 96K tokens)
ATTN_LENGTHS = [
    [1000, 963],
    [1000, 963, 926, 889],
    [1000, 963, 926, 889, 852, 815, 778, 741],
    [9000, 8963, 8926, 8889],
    [0, 63, 64, 65, 1000, 4097, 9001, 40000],
    [40000, 63],
    [50001, 47000, 0, 63],
    [65, 0, 4097, 9001],
]


# vLLM's hybrid models raise the attention block size to match the recurrent
# state's page (Qwen3.5 / 3.6 35B-A3B: 1088 tokens)
HYBRID_BLOCK = 1088


@requires_mono
@pytest.mark.parametrize(
    "lc,block_size,dtype",
    [(lc, 64, torch.float16) for lc in ATTN_LENGTHS] +
    [(ATTN_LENGTHS[i], HYBRID_BLOCK, torch.float16) for i in (2, 4, 6)] +
    [(ATTN_LENGTHS[i], 64, torch.bfloat16) for i in (0, 2, 4, 6)] +
    [(ATTN_LENGTHS[4], HYBRID_BLOCK, torch.bfloat16)])
def test_attn_batched_vs_single_row(lc, block_size, dtype):
    nb = len(lc)
    gen = torch.Generator().manual_seed(100 + sum(lc))
    bt, lc, slot, pos, blocks = _attn_case(gen, lc, 1, block_size)
    t = attn_tensors(gen, DEVICE, blocks, block_size, dtype)
    kv = t[18]
    kv0 = kv.clone()
    h, r = rows(nb, gen, DEVICE, dtype)
    layer, _mb = _register(True, t)
    ref_layer, _mb1 = _register(True, t)
    seq = torch.tensor([c + 1 for c in lc], dtype=torch.int32, device=DEVICE)
    ref = [
        _run(ref_layer, h[b:b + 1], r[b:b + 1], pos[b:b + 1], bt[b:b + 1],
             seq[b:b + 1], slot[b:b + 1]) for b in range(nb)
    ]
    torch.xpu.synchronize()
    kv_ref = kv.clone()
    kv.copy_(kv0)
    ho, ro = _run(layer, h, r, pos, bt, seq, slot)
    torch.xpu.synchronize()
    for b in range(nb):
        _check(ho[b], ref[b][0][0], 0.9999)
        _check(ro[b], ref[b][1][0], 0.99999)
    flat, flat_ref = kv.view(-1, 2 * HD), kv_ref.view(-1, 2 * HD)
    _check(flat[slot], flat_ref[slot], 0.9999)  # the new k | v rows



@requires_mono
@pytest.mark.parametrize("dtype", DTYPES)
def test_attn_padded_rows(dtype):
    """A batch of 7 requests run by the 8-row kernel, padded like vLLM's
    graphs: the pad row has sequence length 0, slot -1, position 0 and a
    zero block-table row; it must not touch the real rows or the cache."""
    lc = [8, 70, 71, 72, 1007, 4104, 9008]
    gen = torch.Generator().manual_seed(77)
    bt, lc, slot, pos, blocks = _attn_case(gen, lc, 1, HYBRID_BLOCK)
    t = attn_tensors(gen, DEVICE, blocks, HYBRID_BLOCK, dtype)
    kv = t[18]
    kv0 = kv.clone()
    h, r = rows(8, gen, DEVICE, dtype)
    layer, _mb = _register(True, t)
    ref_layer, _mb1 = _register(True, t)
    seq = torch.tensor([c + 1 for c in lc], dtype=torch.int32, device=DEVICE)
    ref = [
        _run(ref_layer, h[b:b + 1], r[b:b + 1], pos[b:b + 1], bt[b:b + 1],
             seq[b:b + 1], slot[b:b + 1]) for b in range(7)
    ]
    torch.xpu.synchronize()
    kv_ref = kv.clone()
    kv.copy_(kv0)
    bt = torch.cat([bt, torch.zeros_like(bt[:1])])
    seq = torch.cat([seq, torch.zeros_like(seq[:1])])
    slot = torch.cat([slot, torch.full_like(slot[:1], -1)])
    pos = torch.cat([pos, torch.zeros_like(pos[:1])])
    ho, ro = _run(layer, h, r, pos, bt, seq, slot)
    torch.xpu.synchronize()
    for b in range(7):
        _check(ho[b], ref[b][0][0], 0.9999)
        _check(ro[b], ref[b][1][0], 0.99999)
    flat, flat_ref = kv.view(-1, 2 * HD), kv_ref.view(-1, 2 * HD)
    real = slot[:7]
    _check(flat[real], flat_ref[real], 0.9999)
    # Besides the new rows, only the V halves of the rest of a partial
    # 16-token tile of cached tokens (the request's own rows after the new
    # token) are zeroed; the pad row (sequence length 0, slot -1) writes
    # nothing.
    tails = [
        t for s, c in zip(real.tolist(), lc) if c % 16
        for t in range(s + 1, (s | 15) + 1)
    ]
    tails = torch.tensor(tails, dtype=torch.int64, device=DEVICE)
    assert not flat[tails, HD:].any()
    untouched = torch.ones(flat.shape[0], dtype=torch.bool, device=DEVICE)
    untouched[real] = False
    flat0 = kv0.view(-1, 2 * HD)
    assert torch.equal(flat[tails, :HD], flat0[tails, :HD])
    untouched[tails] = False
    assert torch.equal(flat[untouched], flat0[untouched])

@requires_mono
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("nb,tokens", SPEC_SHAPES)
def test_gdn_spec_vs_token_by_token(nb, tokens, dtype):
    gen = torch.Generator().manual_seed(200 + nb * 10 + tokens)
    nreq = nb // tokens
    t = gdn_tensors(gen,
                    DEVICE,
                    slots=nreq * tokens + 1,
                    conv_rows=tokens + 2,
                    dtype=dtype)
    conv, ssm = t[22], t[23]
    # num_accepted: different rollback columns per request
    acc = [1 + (3 * b + 1) % tokens for b in range(nreq)]
    idx = torch.arange(nreq * tokens, dtype=torch.int32,
                       device=DEVICE).view(nreq, tokens)
    conv0, ssm0 = conv.clone(), ssm.clone()
    h, r = rows(nb, gen, DEVICE, dtype)
    layer, _mb = _register(False, t)
    # reference: a one-row layer on its own [1, 3, 4096] / [1, ...] state
    ref_conv = torch.empty((1, 3, CONVD), dtype=conv.dtype, device=DEVICE)
    ref_ssm = torch.empty_like(ssm[:1])
    tr = t[:22] + [ref_conv, ref_ssm]
    ref_layer, _mb1 = _register(False, tr)
    zero = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    exp_h, exp_r, exp_s, exp_x = [], [], [], []
    for b in range(nreq):
        ref_conv[0] = conv0[b * tokens, acc[b] - 1:acc[b] + 2]
        ref_ssm[0] = ssm0[b * tokens + acc[b] - 1]
        for i in range(tokens):
            row = b * tokens + i
            ho, ro = _run(ref_layer, h[row:row + 1], r[row:row + 1], zero,
                          zero)
            exp_h.append(ho[0])
            exp_r.append(ro[0])
            exp_s.append(ref_ssm[0].clone())
            exp_x.append(ref_conv[0, 2].clone())
    torch.xpu.synchronize()
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    assert torch.ops._xpu_C.mono_decode_layer_spec(
        layer, tokens, h, r, ho, ro, zero, idx,
        torch.tensor(acc, dtype=torch.int32, device=DEVICE), None)
    torch.xpu.synchronize()
    for row in range(nb):
        _check(ho[row], exp_h[row], 0.9999)
        _check(ro[row], exp_r[row], 0.99999)
        _check(ssm[row], exp_s[row], 0.99999)
    for b in range(nreq):
        line = conv[b * tokens]
        hist = conv0[b * tokens, acc[b]:acc[b] + 2]
        assert torch.equal(line[:2], hist)
        _check(line[2:2 + tokens],
               torch.stack(exp_x[b * tokens:(b + 1) * tokens]), 0.99999)


def _spec_lengths(nreq, ctx):
    return [max(ctx - 37 * b, 0) for b in range(nreq)]


SPEC_IRREGULAR = [
    (8, 4, [63, 4097]),
    (8, 2, [0, 65, 1000, 9001]),
    (6, 3, [64, 40000]),
    (8, 8, [40001]),
    (4, 2, [1, 63]),
]


@requires_mono
@pytest.mark.parametrize(
    "nb,tokens,lc,block_size,dtype",
    [(nb, t, _spec_lengths(nb // t, ctx), 64, F16) for nb, t in SPEC_SHAPES
     for ctx in (1000, 9000)] + [c + (64, F16) for c in SPEC_IRREGULAR] +
    [c + (HYBRID_BLOCK, F16) for c in SPEC_IRREGULAR] +
    [(nb, t, _spec_lengths(nb // t, 1000), 64, BF16)
     for nb, t in SPEC_SHAPES] + [c + (HYBRID_BLOCK, BF16)
                                  for c in SPEC_IRREGULAR[:3]])
def test_attn_spec_vs_token_by_token(nb, tokens, lc, block_size, dtype):
    gen = torch.Generator().manual_seed(300 + nb * 10 + tokens + sum(lc))
    nreq = nb // tokens
    assert len(lc) == nreq
    bt, lc, slot, pos, blocks = _attn_case(gen, lc, tokens, block_size)
    t = attn_tensors(gen, DEVICE, blocks, block_size, dtype)
    kv = t[18]
    kv0 = kv.clone()
    h, r = rows(nb, gen, DEVICE, dtype)
    layer, _mb = _register(True, t)
    ref_layer, _mb1 = _register(True, t)
    exp = []
    for b in range(nreq):
        for i in range(tokens):
            row = b * tokens + i
            seq = torch.tensor([lc[b] + i + 1],
                               dtype=torch.int32,
                               device=DEVICE)
            exp.append(
                _run(ref_layer, h[row:row + 1], r[row:row + 1],
                     pos[row:row + 1], bt[b:b + 1], seq, slot[row:row + 1]))
    torch.xpu.synchronize()
    kv_ref = kv.clone()
    kv.copy_(kv0)
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    seq = torch.tensor([c + tokens for c in lc],
                       dtype=torch.int32,
                       device=DEVICE)
    assert torch.ops._xpu_C.mono_decode_layer_spec(layer, tokens, h, r, ho,
                                                   ro, pos, bt, seq, slot)
    torch.xpu.synchronize()
    for row in range(nb):
        _check(ho[row], exp[row][0][0], 0.9999)
        _check(ro[row], exp[row][1][0], 0.99999)
    flat, flat_ref = kv.view(-1, 2 * HD), kv_ref.view(-1, 2 * HD)
    _check(flat[slot], flat_ref[slot], 0.9999)


@requires_mono
@pytest.mark.parametrize("ctx,dtype", [(c, F16)
                                       for c in (40959, 40960, 40961, 50001,
                                                 131109)] + [(50001, BF16)])
def test_attn_long_context(ctx, dtype):
    # The one-row kernel covers 40960 cached tokens a pass. Small residual
    # rows, so res_out - (h + r) is the attention's contribution; rescaling the
    # V rows past the first pass must change it in both kernels alike.
    gen = torch.Generator().manual_seed(ctx)
    bt, lc, slot, pos, blocks = _attn_case(gen, [ctx, 63], 1)
    t = attn_tensors(gen, DEVICE, blocks, dtype=dtype)
    kv = t[18]
    h, r = rows(2, gen, DEVICE, dtype)
    h, r = h * 0.01, r * 0.01
    hr = h.float() + r.float()
    seq = torch.tensor([c + 1 for c in lc], dtype=torch.int32, device=DEVICE)
    one, _mb = _register(True, t)
    two, _mb1 = _register(True, t)

    def attn_out():
        _, ro1 = _run(one, h[:1], r[:1], pos[:1], bt[:1], seq[:1], slot[:1])
        _, ro2 = _run(two, h, r, pos, bt, seq, slot)
        torch.xpu.synchronize()
        d1, d2 = ro1[0].float() - hr[0], ro2[0].float() - hr[0]
        _check(d1, d2, 0.999)
        return d1

    before = attn_out()
    if ctx >= 50001:  # enough tokens past the first pass to matter
        bs = kv.shape[2]
        tk = torch.arange(40960, ctx)
        idx = (bt[0].cpu()[tk // bs].long() * bs + tk % bs).to(DEVICE)
        flat = kv.view(-1, 2 * HD)
        flat[idx, HD:] *= 4
        assert cos(attn_out(), before) < 0.99


@requires_mono
def test_max_context_covers_the_model():
    # Qwen3.5 / 3.6: max_position_embeddings 262144
    assert torch.ops._xpu_C.mono_max_context() >= 262144


@requires_mono
def test_unsupported_shapes():
    gen = torch.Generator().manual_seed(7)
    t = gdn_tensors(gen, DEVICE, slots=16, conv_rows=7)
    layer, _mb = _register(False, t)
    h, r = rows(10, gen, DEVICE)
    idx = torch.zeros((2, 5), dtype=torch.int32, device=DEVICE)
    acc = torch.ones(2, dtype=torch.int32, device=DEVICE)
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    # 2 requests x 5 tokens has no kernel: nothing launched
    assert not torch.ops._xpu_C.mono_decode_layer_spec(
        layer, 5, h, r, ho, ro, idx[0, :1], idx, acc, None)
    with pytest.raises(RuntimeError):  # 3 rows: no batched kernel
        torch.ops._xpu_C.mono_decode_layer(
            layer, h[:3], r[:3], ho[:3], ro[:3], idx[0, :1],
            torch.arange(3, dtype=torch.int32, device=DEVICE), None, None)


def _mixer(ro, h, r):
    """res_out - (h + r): the mixer's (attention / GDN) contribution."""
    return ro.float() - h.float() - r.float()


def _route_gap(ro, t):
    """Per row: the gap between the 8th and 9th router logit (torch, from
    res_out); a near tie may pick another 8th expert in either dtype."""
    v = ro.float()
    x = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * (
        1 + t[1].float())
    top = (x @ t[2].float().t()).topk(9, dim=-1).values
    return top[:, 7] - top[:, 8]


@requires_mono
@pytest.mark.parametrize("attn,nb,tokens", [(False, 1, 1), (False, 2, 1),
                                            (False, 8, 1), (False, 8, 4),
                                            (True, 1, 1), (True, 4, 1),
                                            (True, 8, 2)])
def test_bf16_matches_fp16(attn, nb, tokens):
    """The bf16 kernels against the fp16 ones on the same values (bf16 values,
    exact in fp16): outputs, the mixer's contribution and the new recurrent
    state / KV rows differ only by the rounding to bf16 inside the layer."""
    nreq = nb // tokens
    gen = torch.Generator().manual_seed(400 + nb * 10 + tokens + attn)
    if attn:
        bt, lc, slot, pos, blocks = _attn_case(gen,
                                               _spec_lengths(nreq, 1000),
                                               tokens)
        tb = attn_tensors(gen, DEVICE, blocks, dtype=BF16)
    else:
        tb = gdn_tensors(gen,
                         DEVICE,
                         slots=nreq * tokens,
                         conv_rows=tokens + 2,
                         dtype=BF16)
    # small residual rows: res_out is mostly the mixer's contribution (its
    # rounding to bf16 would hide a small attention output)
    hb, rb = rows(nb, gen, DEVICE, BF16)
    hb, rb = hb * 0.01, rb * 0.01
    out = {}
    for dtype in DTYPES:
        t = [x.to(dtype) if x.dtype == BF16 else x.clone() for x in tb]
        h, r = hb.to(dtype), rb.to(dtype)
        layer, _mb = _register(attn, t)
        ho, ro = torch.empty_like(h), torch.empty_like(r)
        if attn:
            seq = torch.tensor([c + tokens for c in lc],
                               dtype=torch.int32,
                               device=DEVICE)
            if tokens == 1:
                torch.ops._xpu_C.mono_decode_layer(layer, h, r, ho, ro, pos,
                                                   bt, seq, slot)
            else:
                assert torch.ops._xpu_C.mono_decode_layer_spec(
                    layer, tokens, h, r, ho, ro, pos, bt, seq, slot)
            state = t[18].view(-1, 2 * HD)[slot]
        else:
            zero = torch.zeros(1, dtype=torch.int32, device=DEVICE)
            idx = torch.arange(nreq * tokens, dtype=torch.int32,
                               device=DEVICE)
            if tokens == 1:
                torch.ops._xpu_C.mono_decode_layer(layer, h, r, ho, ro, zero,
                                                   idx, None, None)
            else:
                assert torch.ops._xpu_C.mono_decode_layer_spec(
                    layer, tokens, h, r, ho, ro, zero,
                    idx.view(nreq, tokens),
                    torch.ones(nreq, dtype=torch.int32, device=DEVICE), None)
            state = torch.cat([t[23].flatten(), t[22][:, 2:].flatten()])
        torch.xpu.synchronize()
        out[dtype] = (ho, ro, _mixer(ro, h, r), state)
    a, b = out[F16], out[BF16]
    assert b[0].dtype == BF16 and b[1].dtype == BF16
    gap = _route_gap(b[1], tb)
    for row in range(nb):
        if gap[row] > 0.02:
            _check(b[0][row], a[0][row], 0.9995)
        _check(b[1][row], a[1][row], 0.9999)
        _check(b[2][row], a[2][row], 0.9999)
    _check(b[3], a[3], 0.9999)


@requires_mono
def test_mixed_dtypes_refused():
    """Activations and KV cache / conv state in one dtype: an fp16 / bf16 mix
    is refused at registration, activations of the other dtype at the call."""
    gen = torch.Generator().manual_seed(8)
    for dtype, other in ((F16, BF16), (BF16, F16)):
        t = attn_tensors(gen, DEVICE, 4, dtype=dtype)
        t[18] = t[18].to(other)
        with pytest.raises(RuntimeError, match="mix of activations and KV"):
            _register(True, t)
        t = gdn_tensors(gen, DEVICE, slots=2, dtype=dtype)
        t[22] = t[22].to(other)
        with pytest.raises(RuntimeError, match="conv state"):
            _register(False, t)
        t = gdn_tensors(gen, DEVICE, slots=2, dtype=dtype)
        t[1] = t[1].to(other)
        with pytest.raises(RuntimeError, match="the layer's dtype"):
            _register(False, t)
        layer, _mb = _register(False, gdn_tensors(gen, DEVICE, 2,
                                                  dtype=dtype))
        h, r = rows(1, gen, DEVICE, other)
        zero = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        with pytest.raises(RuntimeError, match="the layer's dtype"):
            _run(layer, h, r, zero, zero)
