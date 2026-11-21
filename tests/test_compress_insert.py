# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the fused_kv_compress_norm_rope_insert_sparse_attn
SYCL kernel.

Tests the fused gather + online-softmax compress + RMSNorm + GPT-J RoPE +
FP8/bf16 cache insert kernel against a pure-PyTorch reference, for both
DeepSeek-V4 layer shapes:
  Layer A: compress_ratio=4,   overlap=1
  Layer B: compress_ratio=128, overlap=0

Run:
    pytest tests/test_compress_insert.py -v
"""
from typing import NamedTuple

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = torch.device("xpu")

# DeepSeek-V4 compressed KV cache constants
HEAD_SIZE = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM  # 448
QUANT_BLOCK = 64
TOKEN_STRIDE = 576
SCALE_DIM = 8
REAL_SCALES = NOPE_HEAD_DIM // QUANT_BLOCK  # 7, the 8th byte is padding
RMS_EPS = 1e-6
FP8_MAX = 448.0

# Block pool for the synthetic caches. The tests validate byte-level layout,
# so a small pool is enough.
NUM_BLOCKS = 512

LAYER_CONFIGS = [
    # (compress_ratio, overlap, kv_blk, state_blk_sz)
    (4, 1, 64, 4),
    (128, 0, 2, 8),
]

# (make_non_boundary, invalidate_slot, invalidate_kv_slot)
CASES = {
    "all_valid": (False, False, False),
    "mixed_boundary": (True, False, False),
    "neg_slot": (False, True, False),
    "neg_kv_slot": (False, False, True),
}


class Inputs(NamedTuple):
    """Kernel arguments, in op call order, so ``op(*inputs)`` works."""
    state_cache: torch.Tensor
    token_to_req: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    rms_norm_weight: torch.Tensor
    rms_norm_eps: float
    cos_sin_cache: torch.Tensor
    k_cache: torch.Tensor
    kv_slot_mapping: torch.Tensor
    kv_blk: int
    compress_ratio: int
    overlap: int
    rope_head_dim: int
    token_stride: int
    scale_dim: int
    kv_block_stride: int


def make_inputs(config, num_tokens, case="all_valid", seed=42, device=DEVICE):
    compress_ratio, overlap, kv_blk, state_blk_sz = config
    make_non_boundary, invalidate_slot, invalidate_kv_slot = CASES[case]
    state_width = (1 + overlap) * HEAD_SIZE
    max_position = (num_tokens + 1) * compress_ratio - 1
    gen = torch.Generator(device="cpu").manual_seed(seed)

    state_cache = (torch.randn(NUM_BLOCKS, state_blk_sz, 2 * state_width,
                               generator=gen) * 0.1).to(device)
    block_table = torch.randint(0, NUM_BLOCKS,
                                (1, max_position // state_blk_sz + 8),
                                dtype=torch.int32, generator=gen).to(device)
    k_cache = torch.zeros(NUM_BLOCKS, kv_blk, TOKEN_STRIDE + SCALE_DIM,
                          dtype=torch.uint8, device=device)

    positions = ((torch.arange(num_tokens) + 1) * compress_ratio - 1).to(device)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    kv_slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    rms_norm_weight = torch.ones(HEAD_SIZE, dtype=torch.float32, device=device)

    # Perturb odd tokens so the kernel's early-exit paths are exercised.
    odd = torch.arange(1, num_tokens, 2, device=device)
    if make_non_boundary:
        positions[odd] += 1
    if invalidate_slot:
        slot_mapping[odd] = -1
    if invalidate_kv_slot:
        kv_slot_mapping[odd] = -1

    # +2 rows of slack for the non-boundary case bumping a position by 1.
    half = ROPE_HEAD_DIM // 2
    freq = torch.arange(half, dtype=torch.float32) / half
    base = torch.arange(max_position + 2, dtype=torch.float32).unsqueeze(1)
    cos_sin_cache = torch.cat([
        torch.cos(base * (10000 ** (-freq))),
        torch.sin(base * (10000 ** (-freq))),
    ], dim=1).to(device)

    return Inputs(
        state_cache, token_to_req, positions, slot_mapping, block_table,
        rms_norm_weight, RMS_EPS, cos_sin_cache, k_cache, kv_slot_mapping,
        kv_blk, compress_ratio, overlap, ROPE_HEAD_DIM,
        TOKEN_STRIDE, SCALE_DIM, int(k_cache.stride(0)),
    )


def slot_offsets(kv_slot: int, inp: Inputs):
    """Byte offsets of a slot's token payload and of its scale bytes."""
    base = (kv_slot // inp.kv_blk) * inp.kv_block_stride
    kv_pos = kv_slot % inp.kv_blk
    return (base + kv_pos * inp.token_stride,
            base + inp.kv_blk * inp.token_stride + kv_pos * inp.scale_dim)


def written_slots(inp: Inputs):
    """Slots the kernel is expected to write, in order and deduplicated."""
    slots = []
    for t in range(inp.positions.numel()):
        kv_slot = int(inp.kv_slot_mapping[t].item())
        if int(inp.slot_mapping[t].item()) < 0 or kv_slot < 0:
            continue
        if (int(inp.positions[t].item()) + 1) % inp.compress_ratio != 0:
            continue
        if kv_slot not in slots:
            slots.append(kv_slot)
    return slots


def quantize_nope(nope: torch.Tensor):
    """Quantize the NOPE half to FP8 bytes plus one ue8m0 scale per block."""
    fp8_bytes = torch.empty(NOPE_HEAD_DIM, dtype=torch.uint8)
    scales = torch.empty(REAL_SCALES, dtype=torch.uint8)

    for b in range(REAL_SCALES):
        blk = nope[b * QUANT_BLOCK:(b + 1) * QUANT_BLOCK]
        absmax = torch.clamp(blk.abs().max(), min=1e-4)
        exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
        inv_scale = 2.0 ** (-float(exponent.item()))

        scales[b] = torch.clamp(exponent + 127.0, 0.0, 255.0).to(torch.uint8)
        q = torch.clamp(blk * inv_scale, -FP8_MAX, FP8_MAX)
        fp8_bytes[b * QUANT_BLOCK:(b + 1) * QUANT_BLOCK] = (
            q.to(torch.float8_e4m3fn).view(torch.uint8))

    return fp8_bytes, scales


def reference_kv_compress_insert(inp: Inputs) -> None:
    """Pure PyTorch reference; writes raw bytes into ``inp.k_cache``."""
    state_width = inp.state_cache.shape[-1] // 2
    state_block_size = inp.state_cache.shape[1]
    n_gather = (1 + inp.overlap) * inp.compress_ratio
    half_rope = inp.rope_head_dim // 2
    raw = inp.k_cache.reshape(-1)

    for t in range(inp.positions.numel()):
        position = int(inp.positions[t].item())
        kv_slot = int(inp.kv_slot_mapping[t].item())
        if int(inp.slot_mapping[t].item()) < 0 or kv_slot < 0:
            continue
        if (position + 1) % inp.compress_ratio != 0:
            continue

        # Gather the compression window and softmax-weight it
        req = int(inp.token_to_req[t].item())
        kv_rows, score_rows = [], []
        for gi in range(n_gather):
            gp = position - n_gather + 1 + gi
            if gp < 0:
                kv_rows.append(torch.zeros(HEAD_SIZE))
                score_rows.append(torch.full((HEAD_SIZE,), float("-inf")))
                continue
            sb = int(inp.block_table[req, gp // state_block_size].item())
            row = inp.state_cache[sb, gp % state_block_size].float()
            off = HEAD_SIZE if gi >= inp.compress_ratio else 0
            kv_rows.append(row[off:off + HEAD_SIZE])
            score_rows.append(
                row[state_width + off:state_width + off + HEAD_SIZE])

        weights = torch.softmax(torch.stack(score_rows), dim=0)
        compressed = (torch.stack(kv_rows) * weights).sum(dim=0)

        # RMSNorm
        variance = compressed.pow(2).mean()
        normed = compressed * torch.rsqrt(variance + inp.rms_norm_eps)
        normed = normed * inp.rms_norm_weight.float()

        # The quant path is defined on bf16-rounded pre-rope values
        pre_rope = normed.to(torch.bfloat16).float()

        # GPT-J RoPE on the tail pairs
        post_rope = normed.clone()
        if inp.rope_head_dim > 0:
            cp = (position // inp.compress_ratio) * inp.compress_ratio
            cos = inp.cos_sin_cache[cp, :half_rope].float()
            sin = inp.cos_sin_cache[cp, half_rope:].float()
            pairs = post_rope[NOPE_HEAD_DIM:].reshape(half_rope, 2)
            rotated = torch.empty_like(pairs)
            rotated[:, 0] = pairs[:, 0] * cos - pairs[:, 1] * sin
            rotated[:, 1] = pairs[:, 1] * cos + pairs[:, 0] * sin
            post_rope[NOPE_HEAD_DIM:] = rotated.reshape(inp.rope_head_dim)

        fp8_nope, scales = quantize_nope(pre_rope[:NOPE_HEAD_DIM])
        rope_bytes = (
            post_rope[NOPE_HEAD_DIM:].to(torch.bfloat16).view(torch.uint8))

        token_off, scale_off = slot_offsets(kv_slot, inp)
        raw[token_off:token_off + NOPE_HEAD_DIM] = fp8_nope
        raw[token_off + NOPE_HEAD_DIM:token_off + inp.token_stride] = rope_bytes
        raw[scale_off:scale_off + REAL_SCALES] = scales
        raw[scale_off + REAL_SCALES] = 0


@pytest.mark.parametrize("config", LAYER_CONFIGS, ids=["cr4", "cr128"])
@pytest.mark.parametrize("num_tokens", [1, 5, 16, 33])
@pytest.mark.parametrize("case", list(CASES))
@torch.inference_mode()
def test_correctness(config, num_tokens, case):
    """Test written cache bytes and untouched bytes against the reference."""
    inp = make_inputs(config, num_tokens, case)
    # Same seed, so the CPU inputs are bit-identical, with a zeroed cache.
    ref = make_inputs(config, num_tokens, case, device="cpu")

    torch.ops._xpu_C.fused_kv_compress_norm_rope_insert_sparse_attn(*inp)
    reference_kv_compress_insert(ref)

    out_raw = inp.k_cache.cpu().reshape(-1)
    ref_raw = ref.k_cache.reshape(-1)

    untouched = torch.ones_like(out_raw, dtype=torch.bool)
    for kv_slot in written_slots(ref):
        token_off, scale_off = slot_offsets(kv_slot, inp)
        untouched[token_off:token_off + TOKEN_STRIDE] = False
        untouched[scale_off:scale_off + SCALE_DIM] = False

        # FP8 NOPE payload and ue8m0 scales must match byte for byte
        assert torch.equal(
            out_raw[token_off:token_off + NOPE_HEAD_DIM],
            ref_raw[token_off:token_off + NOPE_HEAD_DIM],
        ), f"FP8 NOPE mismatch at slot {kv_slot}"
        assert torch.equal(
            out_raw[scale_off:scale_off + SCALE_DIM],
            ref_raw[scale_off:scale_off + SCALE_DIM],
        ), f"Scale mismatch at slot {kv_slot}"

        # Rope tail: the SYCL online softmax accumulates in a different order
        # than the reference, so compare bf16 values instead of raw bytes
        rope = slice(token_off + NOPE_HEAD_DIM, token_off + TOKEN_STRIDE)
        torch.testing.assert_close(
            out_raw[rope].view(torch.bfloat16).float(),
            ref_raw[rope].view(torch.bfloat16).float(),
            rtol=0, atol=1e-2,
        )

    assert torch.equal(out_raw[untouched], ref_raw[untouched]), (
        "Kernel wrote outside the expected slots")


_ROPE_ERR = r"rope_head_dim must be a multiple of 4 and in \[0, 512\]"
_RATIO_ERR = r"unsupported \(compress_ratio, overlap\)"
_BAD_STRIDE = LAYER_CONFIGS[0][2] * (TOKEN_STRIDE + SCALE_DIM) - 1

# vLLM only ever builds (4, 1) and (128, 0): overlap is derived as
# compress_ratio == 4, and the compressor asserts the ratio is 4 or 128. The
# kernel relies on that to bound its block-id prefetch by n_gather <= 128, so
# any other pair must be rejected rather than silently under-prefetched.
CONTRACT_VIOLATIONS = {
    "kv_block_stride": (dict(kv_block_stride=_BAD_STRIDE),
                        "kv_block_stride too small"),
    "rope_head_dim_unaligned": (dict(rope_head_dim=31), _ROPE_ERR),
    "rope_head_dim_too_large": (dict(rope_head_dim=516), _ROPE_ERR),
    "token_stride": (dict(token_stride=TOKEN_STRIDE - 1),
                     r"token_stride too small for fp8\+rope payload"),
    "scale_dim": (dict(scale_dim=6),
                  "scale_dim too small for NOPE quant blocks"),
    # Right ratio, wrong overlap: the pairing most likely to be misconfigured.
    "ratio_4_0": (dict(compress_ratio=4, overlap=0), _RATIO_ERR),
    "ratio_128_1": (dict(compress_ratio=128, overlap=1), _RATIO_ERR),
    # Ratio outside {4, 128} entirely.
    "ratio_8_0": (dict(compress_ratio=8, overlap=0), _RATIO_ERR),
}


@pytest.mark.parametrize("violation", list(CONTRACT_VIOLATIONS))
@torch.inference_mode()
def test_contract_violations_raise(violation):
    """Test that out-of-contract arguments are rejected on the host."""
    overrides, err_match = CONTRACT_VIOLATIONS[violation]
    inp = make_inputs(LAYER_CONFIGS[0], num_tokens=1)

    with pytest.raises(RuntimeError, match=err_match):
        torch.ops._xpu_C.fused_kv_compress_norm_rope_insert_sparse_attn(
            *inp._replace(**overrides))
