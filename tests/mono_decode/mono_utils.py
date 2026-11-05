# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the mono decode kernel tests: random decoder layers in the
layouts mono_register_layer takes (Qwen3.5 / Qwen3.6 MoE 35B-A3B shapes, one
TP2 rank), mailboxes and skip markers. The 16-bit tensors take a dtype (fp16
or bf16); the same seed gives the same values rounded to either."""

import socket
import time

import pytest
import torch

from vllm_xpu_kernels import mono_decode_interface

HID, NE, INTER = 2048, 256, 256  # INTER: per rank (512 / 2)
NKH, NVH, DK, DV = 8, 16, 128, 128  # GDN heads per rank
CONVD = 2 * NKH * DK + NVH * DV  # 4096
QKVZ = CONVD + NVH * DV  # 6144
NQ, HD = 8, 256  # attention q heads per rank, head dim (1 kv head per rank)
QKV_ROWS = NQ * 2 * HD + 2 * HD  # q | gate, k, v


def mono_available(device: int = 0) -> bool:
    return (mono_decode_interface.is_available()
            and torch.xpu.is_available()
            and torch.xpu.device_count() > device
            and mono_decode_interface.supports(torch.device("xpu", device)))


requires_mono = pytest.mark.skipif(
    not mono_available(),
    reason="needs the mono decode kernels and an Xe2 (Battlemage) GPU")
requires_two_cards = pytest.mark.skipif(
    not (mono_available(0) and mono_available(1)),
    reason="needs two Xe2 (Battlemage) GPUs")


def mailbox(device) -> torch.Tensor:
    """One layer's mailbox (zeroed): pass its address as both the own and the
    peer mailbox for a single-card loopback (the all-reduces then return twice
    this rank's partial sums)."""
    return torch.zeros(torch.ops._xpu_C.mono_mbox_bytes(),
                       dtype=torch.uint8,
                       device=device)


def _fp8(shape, gen, device, std=1.0):
    w = torch.randn(shape, generator=gen, dtype=torch.float32) * std
    return w.clamp_(-240.0, 240.0).to(torch.float8_e4m3fn).to(device)


def _w16(shape, gen, device, std=1.0, mean=0.0, dtype=torch.float16):
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * std +
            mean).to(dtype).to(device)


def _scale(k, device):
    return torch.tensor([1.0 / k**0.5], dtype=torch.float32, device=device)


def _nk(n, k, gen, device):
    """fp8 linear weight as vLLM keeps it: the [K, N] view of a row-major
    [N, K] tensor, and its per-tensor scale."""
    return _fp8((n, k), gen, device).t(), _scale(k, device)


def common_tensors(gen, device, attn: bool, dtype=torch.float16) -> list:
    out_w, out_s = _nk(HID, NQ * HD if attn else NVH * DV, gen, device)
    sw13, sw13_s = _nk(2 * INTER, HID, gen, device)
    sw2, sw2_s = _nk(HID, INTER, gen, device)
    w16 = lambda shape, std: _w16(shape, gen, device, std, dtype=dtype)
    return [
        w16((HID, ), 0.1),  # input_layernorm (Gemma: 1 + w)
        w16((HID, ), 0.1),  # post_attention_layernorm
        w16((NE, HID), 0.05),  # router
        w16((1, HID), 0.05),  # shared_expert_gate
        out_w,
        out_s,
        _fp8((NE, HID, 2 * INTER), gen, device),  # w13 (XPU MoE layout)
        torch.full((NE, ), 1.0 / HID**0.5, dtype=torch.float32, device=device),
        _fp8((NE, INTER, HID), gen, device),  # w2
        torch.full((NE, ), 1.0 / INTER**0.5, dtype=torch.float32,
                   device=device),
        sw13,
        sw13_s,
        sw2,
        sw2_s,
    ]


def gdn_tensors(gen,
                device,
                slots: int,
                conv_rows: int = 3,
                dtype=torch.float16) -> list:
    qkvz, qkvz_s = _nk(QKVZ, HID, gen, device)
    ba, ba_s = _nk(2 * NVH, HID, gen, device)
    a_log = torch.log(torch.rand(NVH, generator=gen) * 15 + 1).to(device)
    conv_state = _w16((slots, conv_rows, CONVD), gen, device, 0.5, dtype=dtype)
    ssm_state = (torch.randn((slots, NVH, DV, DK), generator=gen) *
                 0.05).to(device)
    return common_tensors(gen, device, False, dtype) + [
        qkvz,
        qkvz_s,
        ba,
        ba_s,
        _w16((CONVD, 1, 4), gen, device, 0.3, dtype=dtype),  # conv1d
        a_log,
        _w16((NVH, ), gen, device, 1.0, dtype=dtype),  # dt_bias
        _w16((DV, ), gen, device, 0.1, 1.0, dtype=dtype),  # gated norm weight
        conv_state,
        ssm_state,
    ]


def attn_tensors(gen,
                 device,
                 blocks: int,
                 block_size: int = 64,
                 dtype=torch.float16) -> list:
    qkv, qkv_s = _nk(QKV_ROWS, HID, gen, device)
    kv = _w16((blocks, 1, block_size, 2 * HD), gen, device, 1.0, dtype=dtype)
    return common_tensors(gen, device, True, dtype) + [
        qkv,
        qkv_s,
        _w16((HD, ), gen, device, 0.1, dtype=dtype),  # q_norm (Gemma: 1 + w)
        _w16((HD, ), gen, device, 0.1, dtype=dtype),  # k_norm
        kv,
    ]


def rows(n, gen, device, dtype=torch.float16):
    """n hidden rows and n residual rows."""
    return (_w16((n, HID), gen, device, 1.0, dtype=dtype),
            _w16((n, HID), gen, device, 1.0, dtype=dtype))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def addr(t: torch.Tensor) -> int:
    """Device address of t as the ops' int (int64: USM addresses can be above
    2**63)."""
    p = t.data_ptr()
    return p - (1 << 64) if p >= 1 << 63 else p


# tensors replicated on both ranks (the rest are the rank's shards / caches)
_REPLICATED = {0, 1, 2, 3}
_REPLICATED_GDN = _REPLICATED | {21}
_REPLICATED_ATTN = _REPLICATED | {16, 17}


def exchange_fd(rank, path, fd):
    """Swaps the dma-buf fds of the two ranks over a unix socket
    (SCM_RIGHTS)."""
    if rank == 0:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(1)
        conn, _ = srv.accept()
    else:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        for _ in range(600):
            try:
                conn.connect(path)
                break
            except OSError:
                time.sleep(0.05)
    with conn:
        socket.send_fds(conn, [b"f"], [fd])
        _, fds, _, _ = socket.recv_fds(conn, 1, 1)
    if rank == 0:
        srv.close()
    return fds[0]


def layer_tensors(attn,
                  shard_seed,
                  rank_dev,
                  nb,
                  tokens,
                  blocks,
                  dtype=torch.float16):
    """Replicated tensors from seed 0, sharded ones (and caches) from
    shard_seed."""
    def build(seed):
        gen = torch.Generator().manual_seed(seed)
        if attn:
            return attn_tensors(gen, rank_dev, blocks, dtype=dtype)
        return gdn_tensors(gen,
                           rank_dev,
                           slots=nb * tokens,
                           conv_rows=tokens + 2,
                           dtype=dtype)
    rep, own = build(0), build(shard_seed)
    keep = _REPLICATED_ATTN if attn else _REPLICATED_GDN
    return [rep[i] if i in keep else own[i] for i in range(len(rep))]
