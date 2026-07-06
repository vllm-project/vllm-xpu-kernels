# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for deepseek_fused_indexer_q_rope_mxfp4 SYCL kernel.

Tests the fused GPT-J RoPE + per-block MXFP4 quantize + weight-fold kernel
against a pure-PyTorch reference matching the standalone benchmark's ref_kernel.

Run:
    pytest tests/test_deepseek_fused_indexer_q_rope_mxfp4.py -v
"""
import math

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = torch.device("xpu")

# DeepSeek-V4 indexer Q constants
NUM_HEADS = 64
HEAD_DIM = 128
NOPE_DIM = 64
ROPE_DIM = 64
HALF_ROPE = 32
MXFP4_BLOCK_SIZE = 32
NUM_BLOCKS = 4
FP4_MAX = 6.0


def encode_fp4(x: float) -> int:
    """Staircase FP4 E2M1 encoder."""
    sign = 0x8 if x < 0 else 0x0
    a = abs(x)
    code = (int(a > 0.25) + int(a > 0.75) + int(a > 1.25) +
            int(a > 1.75) + int(a > 2.5) + int(a > 3.5) + int(a > 5.0))
    return code | sign


def reference_fused_indexer_q_rope_mxfp4(
    q, positions, cos_sin_cache, index_weights, softmax_scale, head_scale
):
    """Pure PyTorch reference matching standalone ref_kernel semantics."""
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    combined_scale = softmax_scale * head_scale

    packed_out = torch.zeros(num_tokens, num_heads, HEAD_DIM // 2,
                             dtype=torch.uint8, device=q.device)
    scales_out = torch.zeros(num_tokens, num_heads, NUM_BLOCKS,
                             dtype=torch.uint8, device=q.device)
    weights_out = torch.zeros(num_tokens, num_heads,
                              dtype=torch.float32, device=q.device)

    for t in range(num_tokens):
        pos = positions[t].item()
        cos = cos_sin_cache[pos, :HALF_ROPE].float()
        sin = cos_sin_cache[pos, HALF_ROPE:].float()

        for h in range(num_heads):
            q_f = q[t, h].float()

            # NoPE passthrough + GPT-J RoPE with bf16 roundtrip
            fp = torch.zeros(HEAD_DIM, dtype=torch.float32)
            fp[:NOPE_DIM] = q_f[:NOPE_DIM]
            for p in range(HALF_ROPE):
                xe = q_f[NOPE_DIM + 2 * p]
                xo = q_f[NOPE_DIM + 2 * p + 1]
                fp[NOPE_DIM + 2 * p] = (xe * cos[p] - xo * sin[p]).to(
                    torch.bfloat16).float()
                fp[NOPE_DIM + 2 * p + 1] = (xo * cos[p] + xe * sin[p]).to(
                    torch.bfloat16).float()

            # Per-block MXFP4 quantization
            for b in range(NUM_BLOCKS):
                base = b * MXFP4_BLOCK_SIZE
                block = fp[base:base + MXFP4_BLOCK_SIZE]

                # amax with minimum floor.
                # Matches fast_inv_scale subnormal guard.
                amax = block.abs().max().item()
                if amax == 0:
                    amax = FP4_MAX * (2.0 ** -126)

                # ue8m0: biased exponent of scale = 2^ceil(log2(amax/6))
                lr = math.ceil(
                    math.log2(max(amax / FP4_MAX, 2.0**-126))
                )
                lr = min(max(lr, -127), 127)
                ue8m0 = int(lr + 127)
                scales_out[t, h, b] = ue8m0

                inv_scale = 1.0 / (2.0 ** lr)
                for p in range(16):
                    vl = max(-FP4_MAX, min(fp[base + p * 2].item() * inv_scale,
                                           FP4_MAX))
                    vh = max(
                        -FP4_MAX,
                        min(fp[base + p * 2 + 1].item() * inv_scale, FP4_MAX),
                    )
                    packed_out[t, h, base // 2 + p] = (
                        ((encode_fp4(vh) & 0xF) << 4) | (encode_fp4(vl) & 0xF))

            # Weight fold
            w = index_weights[t, h].float().item()
            weights_out[t, h] = w * combined_scale

    return packed_out, scales_out, weights_out


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
@pytest.mark.parametrize("seed", [42])
def test_correctness(num_tokens, seed):
    """Test packed FP4, scales, and folded weights against reference."""
    torch.manual_seed(seed)
    max_pos = 4096
    softmax_scale = 0.08838834764831845
    head_scale = 0.125

    q = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.randint(0, max_pos, (num_tokens,),
                              dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE) * 0.5
    index_weights = torch.rand(num_tokens, NUM_HEADS,
                               dtype=torch.bfloat16, device=DEVICE)

    # Reference
    pk_ref, sc_ref, w_ref = reference_fused_indexer_q_rope_mxfp4(
        q, positions, cos_sin_cache, index_weights, softmax_scale, head_scale)

    # SYCL kernel
    packed_out = torch.empty(num_tokens, NUM_HEADS, HEAD_DIM // 2,
                             dtype=torch.uint8, device=DEVICE)
    scales_out = torch.empty(num_tokens, NUM_HEADS, NUM_BLOCKS,
                             dtype=torch.uint8, device=DEVICE)
    weights_out = torch.empty(num_tokens, NUM_HEADS,
                              dtype=torch.float32, device=DEVICE)
    torch.ops._xpu_C.deepseek_fused_indexer_q_rope_mxfp4(
        q, positions, cos_sin_cache, index_weights,
        softmax_scale, head_scale, packed_out, scales_out, weights_out)

    # Check scales: must match exactly
    sc_diff = (scales_out.int() - sc_ref.int()).abs()
    assert sc_diff.max().item() == 0, (
        f"Scale mismatch: {sc_diff.sum().item()} bytes differ")

    # Check packed FP4: must match exactly
    pk_diff = (packed_out.int() - pk_ref.int()).abs()
    assert pk_diff.max().item() == 0, (
        f"Packed FP4 mismatch: {pk_diff.sum().item()} bytes differ")

    # Check weights: should match within float precision
    w_diff = (weights_out - w_ref).abs()
    assert w_diff.max().item() < 1e-5, (
        f"Weight diff: max={w_diff.max().item():.8f}")


@pytest.mark.parametrize("dist", ["uniform_50", "tiny", "zeros", "mixed"])
def test_edge_cases(dist):
    """Test with extreme input distributions."""
    torch.manual_seed(7)
    num_tokens = 4
    max_pos = 4096
    softmax_scale = 0.08838834764831845
    head_scale = 0.125

    if dist == "uniform_50":
        q = (torch.rand(num_tokens, NUM_HEADS, HEAD_DIM, device=DEVICE) * 100
             - 50).to(torch.bfloat16)
    elif dist == "tiny":
        q = (torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, device=DEVICE)
             * 1e-30).to(torch.bfloat16)
    elif dist == "zeros":
        q = torch.zeros(num_tokens, NUM_HEADS, HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    elif dist == "mixed":
        # Different magnitudes per block within each head
        q = torch.zeros(num_tokens, NUM_HEADS, HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
        mags = [0.01, 1.0, 30.0, 200.0]
        for b in range(4):
            base = b * MXFP4_BLOCK_SIZE
            q[:, :, base:base + MXFP4_BLOCK_SIZE] = (
                torch.randn(num_tokens, NUM_HEADS, MXFP4_BLOCK_SIZE,
                            device=DEVICE) * mags[b]).to(torch.bfloat16)

    positions = torch.randint(0, max_pos, (num_tokens,),
                              dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE) * 0.5
    index_weights = torch.rand(num_tokens, NUM_HEADS,
                               dtype=torch.bfloat16, device=DEVICE)

    pk_ref, sc_ref, w_ref = reference_fused_indexer_q_rope_mxfp4(
        q, positions, cos_sin_cache, index_weights, softmax_scale, head_scale)

    packed_out = torch.empty(num_tokens, NUM_HEADS, HEAD_DIM // 2,
                             dtype=torch.uint8, device=DEVICE)
    scales_out = torch.empty(num_tokens, NUM_HEADS, NUM_BLOCKS,
                             dtype=torch.uint8, device=DEVICE)
    weights_out = torch.empty(num_tokens, NUM_HEADS,
                              dtype=torch.float32, device=DEVICE)
    torch.ops._xpu_C.deepseek_fused_indexer_q_rope_mxfp4(
        q, positions, cos_sin_cache, index_weights,
        softmax_scale, head_scale, packed_out, scales_out, weights_out)

    sc_diff = (scales_out.int() - sc_ref.int()).abs()
    pk_diff = (packed_out.int() - pk_ref.int()).abs()
    w_diff = (weights_out - w_ref).abs()
    assert sc_diff.max().item() == 0, f"[{dist}] Scale mismatch"
    assert pk_diff.max().item() == 0, f"[{dist}] Packed FP4 mismatch"
    assert w_diff.max().item() < 1e-5, f"[{dist}] Weight diff too large"
