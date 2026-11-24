# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for deepseek_inv_rope_fp8_quant SYCL kernel.

Tests the fused inverse GPT-J RoPE + block-scaled FP8 E4M3 quantization kernel
against a pure-PyTorch reference, in both output formats: per-128 float32
scales (use_mxfp8=False) and per-32 raw E8M0 scales (use_mxfp8=True).

Run:
    pytest --noconftest tests/test_deepseek_inv_rope_fp8_quant.py -v
"""
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = torch.device("xpu")

# DeepSeek-V4 default dimensions
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
HALF_ROPE = ROPE_DIM // 2
QUANT_GROUP_SIZE = 128
CHUNKS_PER_HEAD = HEAD_DIM // QUANT_GROUP_SIZE  # 4
MXFP8_GROUP_SIZE = 32
MX_BLOCKS_PER_HEAD = HEAD_DIM // MXFP8_GROUP_SIZE  # 16
FP8_MAX = 448.0  # max value of float8_e4m3fn


def _ref_ue8m0_quant_block(x: torch.Tensor):
    """Per-128-block UE8M0 quantization (power-of-2 scales).

    x: [T, QUANT_GROUP_SIZE] float32
    Returns: (x_fp8 [T, QUANT_GROUP_SIZE] float8_e4m3fn, scale [T] float32)
    """
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale_raw = absmax / FP8_MAX
    # ceil to power of 2
    scale = torch.exp2(torch.ceil(torch.log2(scale_raw)))
    x_scaled = (x / scale).clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale.squeeze(-1)


def reference_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = NOPE_DIM,
    rope_dim: int = ROPE_DIM,
    quant_group_size: int = QUANT_GROUP_SIZE,
):
    """Full reference: inverse RoPE in fp32 + UE8M0 FP8 quant.

    Returns:
        fp8_out:   [G, T, hpg*D] float8_e4m3fn
        scale_out: [G, T, hpg*blocks] float32
    """
    T, H, D = o.shape
    half_rope = rope_dim // 2
    d = heads_per_group * D
    chunks_per_head = D // quant_group_size
    S = heads_per_group * chunks_per_head

    # Reshape [T, H, D] -> [T, G, hpg, D]
    o_4d = o.view(T, n_groups, heads_per_group, D)

    # Lookup cos/sin in fp32
    cos_sin = cos_sin_cache[positions]  # [T, rope_dim] fp32
    cos = cos_sin[:, :half_rope]  # [T, half_rope]
    sin = cos_sin[:, half_rope:]  # [T, half_rope]

    # Apply inverse RoPE in fp32
    o_f32 = o_4d.float()
    rope_region = o_f32[..., nope_dim:]  # [T, G, hpg, rope_dim]
    x_even = rope_region[..., ::2]
    x_odd = rope_region[..., 1::2]
    cos_b = cos[:, None, None, :]
    sin_b = sin[:, None, None, :]
    new_even = x_even * cos_b + x_odd * sin_b
    new_odd = x_odd * cos_b - x_even * sin_b
    out_rope = torch.stack([new_even, new_odd], dim=-1).flatten(-2)
    rotated = torch.cat([o_f32[..., :nope_dim], out_rope], dim=-1)

    # Per-128-block FP8 quantization
    # rotated: [T, G, hpg, D] -> quantize per chunk
    fp8_buf = torch.empty(n_groups, T, d, dtype=torch.float8_e4m3fn,
                          device=o.device)
    scale_buf = torch.empty(n_groups, T, S, dtype=torch.float32,
                            device=o.device)

    for g in range(n_groups):
        for h in range(heads_per_group):
            for c in range(chunks_per_head):
                offset = c * quant_group_size
                block = rotated[:, g, h, offset:offset + quant_group_size]
                x_fp8, scale = _ref_ue8m0_quant_block(block)

                out_offset = h * D + offset
                fp8_buf[g, :, out_offset:out_offset + quant_group_size] = x_fp8

                scale_idx = h * chunks_per_head + c
                scale_buf[g, :, scale_idx] = scale

    return fp8_buf, scale_buf


def assert_dequant_close(ref_fp8, ref_scale, fused_fp8, fused_scale,
                         tight_atol=0.1, loose_atol=1.0,
                         tight_pass_rate=0.98,
                         quant_group_size=QUANT_GROUP_SIZE):
    """Compare via dequantization (fp8 * scale).

    FP8 quantization with power-of-2 scales has inherent boundary effects:
    - FMA precision differences in RoPE can shift absmax by 1 bf16 ULP
    - This can cause scale to differ by 2x at power-of-2 boundaries
    - Result: ~0.1% of elements may have dequant diff up to 1 fp8 ULP * scale

    We verify:
    1. ALL elements within loose_atol (handles 2x scale boundary)
    2. >tight_pass_rate elements within tight_atol (validates bulk correctness)
    """
    # Reshape scales for broadcasting
    G, T, S = ref_scale.shape
    D = ref_fp8.shape[2]
    chunks_per_head = HEAD_DIM // quant_group_size
    heads_per_group = S // chunks_per_head

    ref_deq = torch.zeros(G, T, D, dtype=torch.float32, device=ref_fp8.device)
    fused_deq = torch.zeros(G, T, D, dtype=torch.float32, device=ref_fp8.device)

    for h in range(heads_per_group):
        for c in range(chunks_per_head):
            offset = h * HEAD_DIM + c * quant_group_size
            scale_idx = h * chunks_per_head + c
            ref_s = ref_scale[:, :, scale_idx:scale_idx + 1]
            fused_s = fused_scale[:, :, scale_idx:scale_idx + 1]
            ref_deq[:, :, offset:offset + quant_group_size] = (
                ref_fp8[:, :, offset:offset + quant_group_size].float() * ref_s)
            fused_deq[:, :, offset:offset + quant_group_size] = (
                fused_fp8[:, :, offset:offset + quant_group_size].float()
                * fused_s)

    diff = (fused_deq - ref_deq).abs()
    max_diff = diff.max().item()
    num_elements = diff.numel()
    tight_pass = (diff <= tight_atol).sum().item()
    tight_rate = tight_pass / num_elements

    # Hard bound: no element should exceed loose_atol
    assert max_diff <= loose_atol, (
        f"Max dequant diff {max_diff:.4f} exceeds loose bound {loose_atol}. "
        f"This indicates a real bug, not just quantization boundary effects.")

    # Soft bound: vast majority should be within tight_atol
    assert tight_rate >= tight_pass_rate, (
        f"Only {tight_rate*100:.2f}% elements within tight_atol={tight_atol}, "
        f"need {tight_pass_rate*100:.1f}%. "
        f"Failing: {num_elements - tight_pass}/{num_elements}")


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 32, 128])
@pytest.mark.parametrize(
    "num_heads,n_groups",
    [(64, 8), (32, 4)],
    ids=["H64_G8", "H32_G4"],
)
@pytest.mark.parametrize("seed", [0, 42])
def test_correctness(num_tokens, num_heads, n_groups, seed):
    """Compare SYCL kernel against reference for FP8 values and scales."""
    torch.manual_seed(seed)
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    o = torch.randn(num_tokens, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.randint(0, max_pos, (num_tokens,),
                              dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    # Reference
    ref_fp8, ref_scale = reference_inv_rope_fp8_quant(
        o, positions, cos_sin_cache, n_groups, heads_per_group)

    # SYCL kernel
    fused_fp8, fused_scale = torch.ops._xpu_C.deepseek_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM)

    # Check shapes
    d = heads_per_group * HEAD_DIM
    S = heads_per_group * CHUNKS_PER_HEAD
    assert fused_fp8.shape == (n_groups, num_tokens, d)
    assert fused_scale.shape == (n_groups, num_tokens, S)
    assert ref_fp8.shape == fused_fp8.shape
    assert ref_scale.shape == fused_scale.shape

    # Scales: both use UE8M0 (power-of-2), should be very close.
    # Allow up to factor-of-2 difference due to FMA rounding at boundaries.
    scale_ratio = fused_scale / ref_scale.clamp(min=1e-30)
    assert scale_ratio.max() <= 2.0 and scale_ratio.min() >= 0.5, (
        f"Scale ratio out of [0.5, 2]: min={scale_ratio.min():.4f} "
        f"max={scale_ratio.max():.4f}")

    # Compare via dequant
    assert_dequant_close(ref_fp8, ref_scale, fused_fp8, fused_scale)


@pytest.mark.parametrize("num_tokens", [1, 16, 64, 8192])
def test_output_layout(num_tokens):
    """Verify output tensors have correct shape and dtype."""
    num_heads, n_groups = 64, 8
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    o = torch.randn(num_tokens, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.randint(0, max_pos, (num_tokens,),
                              dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    fp8_out, scale_out = torch.ops._xpu_C.deepseek_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM)

    d = heads_per_group * HEAD_DIM
    S = heads_per_group * CHUNKS_PER_HEAD

    assert fp8_out.shape == (n_groups, num_tokens, d)
    assert fp8_out.dtype == torch.float8_e4m3fn
    assert fp8_out.is_contiguous()

    assert scale_out.shape == (n_groups, num_tokens, S)
    assert scale_out.dtype == torch.float32
    assert scale_out.is_contiguous()


@pytest.mark.parametrize("num_tokens", [1, 4, 32, 128])
@pytest.mark.parametrize(
    "num_heads,n_groups",
    [(64, 8), (32, 4)],
    ids=["H64_G8", "H32_G4"],
)
@pytest.mark.parametrize("seed", [0, 42])
def test_correctness_mxfp8(num_tokens, num_heads, n_groups, seed):
    """MXFP8 path: 32-element blocks with raw E8M0 scales."""
    torch.manual_seed(seed)
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    o = torch.randn(num_tokens, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.randint(0, max_pos, (num_tokens,),
                              dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    ref_fp8, ref_scale = reference_inv_rope_fp8_quant(
        o, positions, cos_sin_cache, n_groups, heads_per_group,
        quant_group_size=MXFP8_GROUP_SIZE)

    fused_fp8, fused_scale = torch.ops._xpu_C.deepseek_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM, True)

    d = heads_per_group * HEAD_DIM
    S = heads_per_group * MX_BLOCKS_PER_HEAD
    assert fused_fp8.shape == (n_groups, num_tokens, d)
    assert fused_scale.shape == (n_groups, num_tokens, S)
    assert fused_scale.dtype == torch.float8_e8m0fnu

    # E8M0 stores the biased exponent only, so .float() is the exact 2**e.
    fused_scale_f32 = fused_scale.float()

    scale_ratio = fused_scale_f32 / ref_scale.clamp(min=1e-30)
    assert scale_ratio.max() <= 2.0 and scale_ratio.min() >= 0.5, (
        f"Scale ratio out of [0.5, 2]: min={scale_ratio.min():.4f} "
        f"max={scale_ratio.max():.4f}")

    assert_dequant_close(ref_fp8, ref_scale, fused_fp8, fused_scale_f32,
                         quant_group_size=MXFP8_GROUP_SIZE)


@pytest.mark.parametrize("num_tokens", [1, 16, 64, 8192])
def test_output_layout_mxfp8(num_tokens):
    """MXFP8 path: verify scale tensor shape and dtype."""
    num_heads, n_groups = 64, 8
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    o = torch.randn(num_tokens, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.randint(0, max_pos, (num_tokens,),
                              dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    fp8_out, scale_out = torch.ops._xpu_C.deepseek_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM, True)

    d = heads_per_group * HEAD_DIM
    S = heads_per_group * MX_BLOCKS_PER_HEAD

    assert fp8_out.shape == (n_groups, num_tokens, d)
    assert fp8_out.dtype == torch.float8_e4m3fn
    assert fp8_out.is_contiguous()

    assert scale_out.shape == (n_groups, num_tokens, S)
    assert scale_out.dtype == torch.float8_e8m0fnu
    assert scale_out.is_contiguous()


def test_mxfp8_scale_block_order():
    """Pin the block -> scale-byte mapping with 16 distinct scales per head.

    The kernel assembles a head's 16 e8m0 bytes across two lane halves, so a
    wrong interleave is a pure permutation. randn inputs only ever produce 2-4
    distinct scales per head, which would hide most such permutations.
    """
    num_heads, n_groups, num_tokens = 8, 1, 4
    heads_per_group = num_heads // n_groups

    # cos=1, sin=0 makes the inverse rope an identity, so the magnitude ladder
    # reaches the rope blocks unchanged and every scale stays exact.
    cos_sin_cache = torch.zeros(16, ROPE_DIM, dtype=torch.float32,
                                device=DEVICE)
    cos_sin_cache[:, :ROPE_DIM // 2] = 1.0
    positions = torch.zeros(num_tokens, dtype=torch.int64, device=DEVICE)

    amp = torch.tensor([2.0 ** (b - 8) for b in range(MX_BLOCKS_PER_HEAD)],
                       dtype=torch.float32, device=DEVICE)
    o = amp.repeat_interleave(MXFP8_GROUP_SIZE).expand(
        num_tokens, num_heads, HEAD_DIM).to(torch.bfloat16).contiguous()

    _, scale_out = torch.ops._xpu_C.deepseek_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM, True)

    got = scale_out.float().view(n_groups, num_tokens, heads_per_group,
                                 MX_BLOCKS_PER_HEAD)
    # 2^(b-8) / 448 ceils to 2^(b-16).
    expect = torch.tensor([2.0 ** (b - 16) for b in range(MX_BLOCKS_PER_HEAD)],
                          dtype=torch.float32, device=DEVICE)
    assert torch.equal(got, expect.expand_as(got)), (
        f"block -> scale mapping is wrong\n"
        f"  expected {expect.tolist()}\n"
        f"  got      {got[0, 0, 0].tolist()}")


def test_mxfp8_block_absmax_covers_every_lane():
    """Every one of a block's 32 elements must reach the absmax reduction.

    The kernel reduces all 16 blocks with a single butterfly, where lane l ends
    up owning block l. A butterfly that keeps the wrong half still produces a
    plausible absmax, and randn inputs hide it: max-of-16 vs max-of-32 usually
    ceils to the same power of two. Here each block has one dominant element,
    placed so that block b's peak lives in lane b, so dropping any lane shifts
    that block's scale by 2**-8.
    """
    num_heads, n_groups, num_tokens = 8, 1, 4
    heads_per_group = num_heads // n_groups

    cos_sin_cache = torch.zeros(16, ROPE_DIM, dtype=torch.float32,
                                device=DEVICE)
    cos_sin_cache[:, :ROPE_DIM // 2] = 1.0
    positions = torch.zeros(num_tokens, dtype=torch.int64, device=DEVICE)

    head = torch.empty(HEAD_DIM, dtype=torch.float32, device=DEVICE)
    for b in range(MX_BLOCKS_PER_HEAD):
        base = b * MXFP8_GROUP_SIZE
        head[base:base + MXFP8_GROUP_SIZE] = 2.0**(b - 16)
        # Lane l holds elements 2l and 2l+1, so 2b walks the peak one lane per
        # block; the +b%2 alternates the even/odd slot within that lane.
        head[base + 2 * b + (b % 2)] = -(2.0**(b - 8))

    o = head.expand(num_tokens, num_heads,
                    HEAD_DIM).to(torch.bfloat16).contiguous()

    _, scale_out = torch.ops._xpu_C.deepseek_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM, True)

    got = scale_out.float().view(n_groups, num_tokens, heads_per_group,
                                 MX_BLOCKS_PER_HEAD)
    # 2^(b-8) / 448 ceils to 2^(b-16); losing the peak would give 2^(b-24).
    expect = torch.tensor([2.0**(b - 16) for b in range(MX_BLOCKS_PER_HEAD)],
                          dtype=torch.float32, device=DEVICE)
    assert torch.equal(got, expect.expand_as(got)), (
        f"a block absmax missed some of its 32 elements\n"
        f"  expected {expect.tolist()}\n"
        f"  got      {got[0, 0, 0].tolist()}")


def test_zero_tokens():
    """Edge case: zero tokens should return empty tensors."""
    num_heads, n_groups = 64, 8
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    o = torch.empty(0, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.empty(0, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    fp8_out, scale_out = torch.ops._xpu_C.deepseek_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM)

    d = heads_per_group * HEAD_DIM
    S = heads_per_group * CHUNKS_PER_HEAD
    assert fp8_out.shape == (n_groups, 0, d)
    assert scale_out.shape == (n_groups, 0, S)


def test_scales_are_power_of_two():
    """Verify all scales are exact powers of 2 (UE8M0 property)."""
    num_tokens, num_heads, n_groups = 32, 64, 8
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    torch.manual_seed(123)
    o = torch.randn(num_tokens, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.randint(0, max_pos, (num_tokens,),
                              dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    _, scale_out = torch.ops._xpu_C.deepseek_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM)

    # For power-of-2 floats, log2 should be exactly integer
    log2_scales = torch.log2(scale_out)
    is_integer = (log2_scales == log2_scales.round())
    assert is_integer.all(), (
        f"Not all scales are power-of-2. "
        f"Non-pow2 count: {(~is_integer).sum().item()}")


def test_fp8_values_in_range():
    """Verify all FP8 values are representable (within [-FP8_MAX, FP8_MAX])."""
    num_tokens, num_heads, n_groups = 64, 64, 8
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    torch.manual_seed(7)
    o = torch.randn(num_tokens, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.randint(0, max_pos, (num_tokens,),
                              dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    fp8_out, _ = torch.ops._xpu_C.deepseek_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM)

    # fp8_e4m3fn values should be in [-448, 448]
    fp8_float = fp8_out.float()
    assert fp8_float.abs().max() <= FP8_MAX, (
        f"FP8 value out of range: max={fp8_float.abs().max().item()}")
