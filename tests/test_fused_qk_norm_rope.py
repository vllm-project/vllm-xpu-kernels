# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401 - registers torch.ops._C ops
from tests.utils import opcheck

DTYPES = [torch.half, torch.bfloat16]
IS_NEOX = [True, False]
EPS_VALUES = [1e-5, 1e-6]
HEAD_DIMS = [64, 128]
NUM_TOKENS = [1, 4, 32]
HEAD_CONFIGS = [(16, 4), (32, 8)]  # (num_heads_q, num_heads_kv)
ROTARY_RATIOS = [1.0, 0.5]
SEEDS = [13]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [4],
        "head_dim": [128],
        "head_config": [(16, 4)],
        "rotary_ratio": [1.0],
    },
}


def _rms_norm(x: torch.Tensor, weight: torch.Tensor,
              eps: float) -> torch.Tensor:
    """Reference RMSNorm: x is [..., head_dim], weight is [head_dim]."""
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      is_neox: bool) -> torch.Tensor:
    """Apply RoPE to x. x shape: [num_tokens, num_heads, head_dim].
    cos/sin shape: [num_tokens, rotary_dim/2]."""
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim].float()
    x_pass = x[..., rotary_dim:]

    if is_neox:
        # Neox style: first half and second half
        half = rotary_dim // 2
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        cos_val = cos.unsqueeze(1)  # [tokens, 1, rotary_dim/2]
        sin_val = sin.unsqueeze(1)
        o1 = x1 * cos_val - x2 * sin_val
        o2 = x2 * cos_val + x1 * sin_val
        x_rot = torch.cat([o1, o2], dim=-1)
    else:
        # Interleaved style: pairs of (x0, x1)
        x_rot = x_rot.view(*x_rot.shape[:-1], -1, 2)
        x0 = x_rot[..., 0]
        x1 = x_rot[..., 1]
        cos_val = cos.unsqueeze(1)
        sin_val = sin.unsqueeze(1)
        o0 = x0 * cos_val - x1 * sin_val
        o1 = x0 * sin_val + x1 * cos_val
        x_rot = torch.stack([o0, o1], dim=-1).flatten(-2)

    return torch.cat([x_rot.to(x.dtype), x_pass], dim=-1)


def _reference_fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    # RMSNorm on Q heads
    q_by_head = q.view(-1, num_heads_q, head_dim)
    q_by_head = _rms_norm(q_by_head, q_weight, eps)

    # RMSNorm on K heads
    k_by_head = k.view(-1, num_heads_kv, head_dim)
    k_by_head = _rms_norm(k_by_head, k_weight, eps)

    # Get cos/sin for the given positions
    rotary_dim = cos_sin_cache.shape[1]
    embed_dim = rotary_dim // 2
    cos_sin = cos_sin_cache[positions]  # [num_tokens, rotary_dim]
    cos = cos_sin[:, :embed_dim].float()
    sin = cos_sin[:, embed_dim:].float()

    # Apply RoPE
    q_by_head = _apply_rotary_emb(q_by_head, cos, sin, is_neox)
    k_by_head = _apply_rotary_emb(k_by_head, cos, sin, is_neox)

    q = q_by_head.view(-1, q_size)
    k = k_by_head.view(-1, kv_size)

    return torch.cat([q, k, v], dim=-1)


@pytest.mark.parametrize("device", XPU_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("head_config", HEAD_CONFIGS)
@pytest.mark.parametrize("rotary_ratio", ROTARY_RATIOS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_fused_qk_norm_rope(
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    eps: float,
    head_dim: int,
    num_tokens: int,
    head_config: tuple,
    rotary_ratio: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    num_heads_q, num_heads_kv = head_config
    rotary_dim = int(head_dim * rotary_ratio)
    max_position = 4096

    total_dim = (num_heads_q + 2 * num_heads_kv) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_fused = qkv_base.clone()
    positions = torch.randint(0,
                              max_position, (num_tokens, ),
                              dtype=torch.long,
                              device=device)

    q_weight = torch.empty(head_dim, dtype=dtype, device=device)
    q_weight.normal_(mean=1.0, std=0.1)
    k_weight = torch.empty(head_dim, dtype=dtype, device=device)
    k_weight.normal_(mean=1.0, std=0.1)

    # Build cos_sin_cache: [max_position, rotary_dim]
    # Layout: [cos_0..cos_{rotary_dim/2-1}, sin_0..sin_{rotary_dim/2-1}]
    inv_freq = 1.0 / (10000.0**(
        torch.arange(0, rotary_dim // 2, dtype=torch.float32, device=device) /
        (rotary_dim // 2)))
    t = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    # Reference (before in-place op)
    ref_result = _reference_fused_qk_norm_rope(
        qkv_base,
        num_heads_q,
        num_heads_kv,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        positions,
    )

    # Run fused kernel (in-place on qkv_fused)
    torch.ops._C.fused_qk_norm_rope(
        qkv_fused,
        num_heads_q,
        num_heads_kv,
        num_heads_kv,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        positions,
    )

    if dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (1e-2, 1e-2)

    torch.testing.assert_close(
        qkv_fused,
        ref_result,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("device", XPU_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@torch.inference_mode()
def test_fused_qk_norm_rope_opcheck(
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
) -> None:
    """Validate the op schema and registration with opcheck."""
    torch.manual_seed(42)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    num_heads_q, num_heads_kv = 16, 4
    head_dim = 128
    num_tokens = 4
    eps = 1e-5
    max_position = 4096
    rotary_dim = head_dim

    total_dim = (num_heads_q + 2 * num_heads_kv) * head_dim
    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_weight = torch.ones(head_dim, dtype=dtype, device=device)
    k_weight = torch.ones(head_dim, dtype=dtype, device=device)

    inv_freq = 1.0 / (10000.0**(
        torch.arange(0, rotary_dim // 2, dtype=torch.float32, device=device) /
        (rotary_dim // 2)))
    t = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    opcheck(
        torch.ops._C.fused_qk_norm_rope,
        (
            qkv,
            num_heads_q,
            num_heads_kv,
            num_heads_kv,
            head_dim,
            eps,
            q_weight,
            k_weight,
            cos_sin_cache,
            is_neox,
            positions,
        ),
    )
