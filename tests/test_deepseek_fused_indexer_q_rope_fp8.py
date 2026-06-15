# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for deepseek_fused_indexer_q_rope_fp8 SYCL kernel.

Tests the fused GPT-J RoPE + per-head FP8 quantize + weight-fold kernel
against a pure-PyTorch reference matching the Triton implementation.

Run:
    pytest tests/test_deepseek_fused_indexer_q_rope_fp8.py -v
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


def reference_fused_indexer_q_rope_fp8(
    q, positions, cos_sin_cache, index_weights, softmax_scale, head_scale
):
    """Pure PyTorch reference matching Triton.

    Matches `_fused_indexer_q_rope_quant_kernel` semantics.
    """
    num_tokens = q.shape[0]
    num_heads = q.shape[1]

    q_fp8 = torch.zeros(num_tokens, num_heads, HEAD_DIM,
                        dtype=torch.uint8, device=q.device)
    weights_out = torch.zeros(num_tokens, num_heads,
                              dtype=torch.float32, device=q.device)

    for t in range(num_tokens):
        pos = positions[t].item()
        cos = cos_sin_cache[pos, :HALF_ROPE].float()
        sin = cos_sin_cache[pos, HALF_ROPE:].float()

        for h in range(num_heads):
            q_f = q[t, h].float()
            nope = q_f[:NOPE_DIM]
            rope = q_f[NOPE_DIM:]

            # GPT-J RoPE
            x_even = rope[0::2]
            x_odd = rope[1::2]
            r_even = x_even * cos - x_odd * sin
            r_odd = x_odd * cos + x_even * sin

            # bf16 roundtrip (matches Triton/SYCL numerics)
            r_even = r_even.to(torch.bfloat16).float()
            r_odd = r_odd.to(torch.bfloat16).float()

            # Reconstruct full head
            rope_out = torch.stack([r_even, r_odd], dim=-1).flatten()
            vals = torch.cat([nope, rope_out])

            # Per-head scale: 2^ceil(log2(max(amax, 1e-4) / 448))
            amax = vals.abs().max().clamp(min=1e-4).item()
            scale = 2.0 ** math.ceil(math.log2(amax / 448.0))
            inv_scale = 1.0 / scale

            # Quantize to fp8_e4m3fn via torch
            scaled = (vals * inv_scale).clamp(-448.0, 448.0)
            fp8_vals = scaled.to(torch.float8_e4m3fn)
            q_fp8[t, h] = fp8_vals.view(torch.uint8)

            # Weight fold
            w = index_weights[t, h].float().item()
            weights_out[t, h] = w * scale * softmax_scale * head_scale

    return q_fp8, weights_out


@pytest.mark.parametrize("num_tokens", [1, 64, 128])
@pytest.mark.parametrize("seed", [0, 42])
def test_correctness(num_tokens, seed):
    """Test FP8 quantized Q and folded weights against reference."""
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
    q_ref, w_ref = reference_fused_indexer_q_rope_fp8(
        q, positions, cos_sin_cache, index_weights, softmax_scale, head_scale)

    # SYCL kernel
    q_fp8 = torch.empty(num_tokens, NUM_HEADS, HEAD_DIM,
                        dtype=torch.uint8, device=DEVICE)
    weights_out = torch.empty(num_tokens, NUM_HEADS,
                              dtype=torch.float32, device=DEVICE)
    torch.ops._xpu_C.deepseek_fused_indexer_q_rope_fp8(
        q, positions, cos_sin_cache, index_weights,
        softmax_scale, head_scale, q_fp8, weights_out)

    # Check FP8 bytes: allow ±1 (rounding mode difference)
    q_dev = q_fp8.to(torch.int16)
    q_r = q_ref.to(torch.int16)
    byte_diff = (q_dev - q_r).abs()
    assert byte_diff.max().item() <= 1, (
        f"FP8 byte diff > 1: max={byte_diff.max().item()}")

    # Check weights: should match within float precision
    w_diff = (weights_out - w_ref).abs()
    w_rel = w_diff / (w_ref.abs() + 1e-8)
    assert w_rel.max().item() < 1e-3, (
        f"Weight rel diff: max={w_rel.max().item():.6f}")


def test_weight_fold_semantics():
    """Verify weight_out = weight * scale * softmax_scale * head_scale."""
    torch.manual_seed(123)
    num_tokens = 8
    max_pos = 4096
    softmax_scale = 0.1
    head_scale = 0.2

    # Use small Q so scale is predictable
    q = torch.ones(num_tokens, NUM_HEADS, HEAD_DIM,
                   dtype=torch.bfloat16, device=DEVICE) * 0.5
    positions = torch.zeros(num_tokens, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.zeros(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)
    cos_sin_cache[:, :HALF_ROPE] = 1.0  # cos=1, sin=0 → identity RoPE
    index_weights = torch.ones(num_tokens, NUM_HEADS,
                               dtype=torch.bfloat16, device=DEVICE)

    q_fp8 = torch.empty(num_tokens, NUM_HEADS, HEAD_DIM,
                        dtype=torch.uint8, device=DEVICE)
    weights_out = torch.empty(num_tokens, NUM_HEADS,
                              dtype=torch.float32, device=DEVICE)
    torch.ops._xpu_C.deepseek_fused_indexer_q_rope_fp8(
        q, positions, cos_sin_cache, index_weights,
        softmax_scale, head_scale, q_fp8, weights_out)

    # All values are 0.5, so amax=0.5.
    # log2(0.5 / 448) ~= -9.807, ceil = -9, so scale = 2^-9 = 1/512.
    # scale = 2^(-9) = 1/512
    expected_scale = 2.0 ** math.ceil(math.log2(0.5 / 448.0))
    expected_w = 1.0 * expected_scale * softmax_scale * head_scale

    # All weights should be the same
    assert torch.allclose(weights_out,
                          torch.full_like(weights_out, expected_w),
                          rtol=1e-3, atol=1e-6), (
        f"Expected {expected_w}, got {weights_out[0, 0].item()}")
