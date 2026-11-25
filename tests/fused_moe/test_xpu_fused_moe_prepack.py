# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
import types

import pytest
import torch

_stubbed_c = False
for _ext in ("_C", "_moe_C", "_xpu_C"):
    _mod_name = f"vllm_xpu_kernels.{_ext}"
    if _mod_name not in sys.modules and importlib.util.find_spec(
            _mod_name) is None:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)
        if _ext == "_C":
            _stubbed_c = True

if _stubbed_c:
    for _op_name in (
        "silu_and_mul",
        "gelu_and_mul",
        "gelu_tanh_and_mul",
        "swigluoai_and_mul",
        "relu2_no_mul",
        "swiglustep_and_mul",
        "situ_and_mul",
    ):
        if not hasattr(torch.ops._C, _op_name):
            setattr(torch.ops._C, _op_name, lambda *args, **kwargs: None)

import vllm_xpu_kernels.fused_moe_interface as fmi  # noqa: E402
from vllm_xpu_kernels.fused_moe_interface import (  # noqa: E402
    REF_FUSED_MOE_ENV, XpuFusedMoe, implement_zp)


@pytest.mark.parametrize("uses_xe2", [True, False])
def test_init_does_not_mutate_and_prepack_converts_bf16(monkeypatch, uses_xe2):
    monkeypatch.setattr(fmi, "_uses_xe2_grouped_gemm", lambda _: uses_xe2)

    num_experts, inter_size, hidden_size = 2, 16, 32
    w13 = torch.nn.Parameter(
        torch.randn(num_experts, 2 * inter_size, hidden_size,
                    dtype=torch.bfloat16),
        requires_grad=False,
    )
    w2 = torch.nn.Parameter(
        torch.randn(num_experts, hidden_size, inter_size,
                    dtype=torch.bfloat16),
        requires_grad=False,
    )
    w13_orig = w13.detach().clone()
    w2_orig = w2.detach().clone()

    moe = XpuFusedMoe(
        w13=w13,
        w13_scales=None,
        w13_bias=None,
        w2=w2,
        w2_scales=None,
        w2_bias=None,
        n_experts_per_token=1,
        activation="silu",
        num_experts=num_experts,
    )

    # __init__ must not mutate weights or mark w13.xpu_fused_moe.
    assert not hasattr(w13, "xpu_fused_moe")
    assert w13.shape == (num_experts, 2 * inter_size, hidden_size)
    assert w2.shape == (num_experts, hidden_size, inter_size)
    assert torch.equal(w13, w13_orig)
    assert torch.equal(w2, w2_orig)

    moe.prepack_weights()

    assert getattr(w13, "xpu_fused_moe", False) is True
    assert moe.w13 is w13
    assert moe.w2 is w2
    assert w13.is_contiguous() and w2.is_contiguous()
    assert w13.shape == (num_experts, hidden_size, 2 * inter_size)
    assert w2.shape == (num_experts, inter_size, hidden_size)
    assert torch.equal(w13, w13_orig.transpose(-1, -2).contiguous())
    assert torch.equal(w2, w2_orig.transpose(-1, -2).contiguous())
    assert moe.inter_size == inter_size

    # Calling prepack_weights() again is a no-op because w13.xpu_fused_moe is
    # already set.
    moe.prepack_weights()
    assert w13.shape == (num_experts, hidden_size, 2 * inter_size)
    assert w2.shape == (num_experts, inter_size, hidden_size)
    assert torch.equal(w13, w13_orig.transpose(-1, -2).contiguous())
    assert torch.equal(w2, w2_orig.transpose(-1, -2).contiguous())
    assert moe.inter_size == inter_size


def test_init_accepts_non_contiguous_and_prepacked_weights(monkeypatch):
    monkeypatch.setattr(fmi, "_uses_xe2_grouped_gemm", lambda _: True)

    num_experts, inter_size, hidden_size = 2, 16, 32
    # Non-contiguous checkpoint tensor: __init__ must not assert contiguity.
    w13_nc = torch.randn(num_experts, hidden_size, 2 * inter_size,
                         dtype=torch.bfloat16).transpose(-1, -2)
    w2_nc = torch.randn(num_experts, inter_size, hidden_size,
                        dtype=torch.bfloat16).transpose(-1, -2)
    assert not w13_nc.is_contiguous() and not w2_nc.is_contiguous()

    moe_nc = XpuFusedMoe(
        w13=w13_nc,
        w13_scales=None,
        w13_bias=None,
        w2=w2_nc,
        w2_scales=None,
        w2_bias=None,
        n_experts_per_token=1,
        activation="silu",
        num_experts=num_experts,
    )
    assert not w13_nc.is_contiguous()
    moe_nc.prepack_weights()
    assert w13_nc.is_contiguous() and w2_nc.is_contiguous()
    assert moe_nc.inter_size == inter_size

    # Already-prepacked weights ([E, K, 2*inter]): __init__ computes inter_size
    # without needing prepack_weights().
    moe_reloaded = XpuFusedMoe(
        w13=w13_nc,
        w13_scales=None,
        w13_bias=None,
        w2=w2_nc,
        w2_scales=None,
        w2_bias=None,
        n_experts_per_token=1,
        activation="silu",
        num_experts=num_experts,
    )
    assert moe_reloaded.inter_size == inter_size


@pytest.mark.parametrize("uses_xe2", [True, False])
def test_prepack_block_fp8_scales(monkeypatch, uses_xe2):
    monkeypatch.setattr(fmi, "_uses_xe2_grouped_gemm", lambda _: uses_xe2)

    num_experts, inter_size, hidden_size = 2, 128, 256
    w13 = torch.randn(num_experts, 2 * inter_size,
                      hidden_size).to(torch.float8_e4m3fn)
    w2 = torch.randn(num_experts, hidden_size,
                     inter_size).to(torch.float8_e4m3fn)
    w13_scales = torch.rand(num_experts, 2 * inter_size // 128,
                            hidden_size // 128,
                            dtype=torch.float32)
    w2_scales = torch.rand(num_experts, hidden_size // 128,
                           inter_size // 128,
                           dtype=torch.float32)
    w13_scales_orig = w13_scales.clone()
    w2_scales_orig = w2_scales.clone()

    moe = XpuFusedMoe(
        w13=w13,
        w13_scales=w13_scales,
        w13_bias=None,
        w2=w2,
        w2_scales=w2_scales,
        w2_bias=None,
        n_experts_per_token=1,
        activation="silu",
        num_experts=num_experts,
    )

    assert torch.equal(w13_scales, w13_scales_orig)
    assert torch.equal(w2_scales, w2_scales_orig)

    moe.prepack_weights()

    assert moe.inter_size == inter_size
    assert w13.shape == (num_experts, hidden_size, 2 * inter_size)
    assert w2.shape == (num_experts, inter_size, hidden_size)
    if uses_xe2:
        assert torch.equal(w13_scales,
                           w13_scales_orig.transpose(-1, -2).contiguous())
        assert torch.equal(w2_scales,
                           w2_scales_orig.transpose(-1, -2).contiguous())
    else:
        assert torch.equal(w13_scales, w13_scales_orig)
        assert torch.equal(w2_scales, w2_scales_orig)


def test_prepack_int4_applies_implement_zp(monkeypatch):
    monkeypatch.setattr(fmi, "_uses_xe2_grouped_gemm", lambda _: True)

    num_experts, inter_size, hidden_size = 2, 16, 64
    w13 = torch.randint(0, 256, (num_experts, 2 * inter_size, hidden_size // 2),
                        dtype=torch.uint8)
    w2 = torch.randint(0, 256, (num_experts, hidden_size, inter_size // 2),
                       dtype=torch.uint8)
    w13_scales = torch.rand(num_experts, 2 * inter_size, hidden_size // 32,
                            dtype=torch.bfloat16)
    w2_scales = torch.rand(num_experts, hidden_size, inter_size // 32,
                           dtype=torch.bfloat16)
    w13_orig = w13.clone()
    w2_orig = w2.clone()

    moe = XpuFusedMoe(
        w13=w13,
        w13_scales=w13_scales,
        w13_bias=None,
        w2=w2,
        w2_scales=w2_scales,
        w2_bias=None,
        n_experts_per_token=1,
        activation="silu",
        num_experts=num_experts,
    )

    # __init__ must not rewrite uint8 -> int8.
    assert w13.dtype == torch.uint8
    assert w2.dtype == torch.uint8
    assert torch.equal(w13, w13_orig)
    assert moe.inter_size == inter_size

    moe.prepack_weights()

    assert w13.dtype == torch.int8
    assert w2.dtype == torch.int8
    assert w13.is_contiguous() and w2.is_contiguous()
    assert moe.inter_size == inter_size
    for i in range(num_experts):
        assert torch.equal(w13[i], implement_zp(w13_orig[i]).to(torch.int8))
        assert torch.equal(w2[i], implement_zp(w2_orig[i]).to(torch.int8))


def test_prepack_ref_path_is_noop(monkeypatch):
    monkeypatch.setenv(REF_FUSED_MOE_ENV, "1")
    monkeypatch.setattr(fmi, "_uses_xe2_grouped_gemm", lambda _: True)

    num_experts, inter_size, hidden_size = 2, 16, 32
    w13 = torch.randn(num_experts, 2 * inter_size, hidden_size,
                      dtype=torch.bfloat16)
    w2 = torch.randn(num_experts, hidden_size, inter_size,
                     dtype=torch.bfloat16)
    w13_orig = w13.clone()
    w2_orig = w2.clone()

    moe = XpuFusedMoe(
        w13=w13,
        w13_scales=None,
        w13_bias=None,
        w2=w2,
        w2_scales=None,
        w2_bias=None,
        n_experts_per_token=1,
        activation="silu",
        num_experts=num_experts,
    )
    assert moe._use_ref is True
    assert moe.inter_size == inter_size

    moe.prepack_weights()

    assert w13.shape == (num_experts, 2 * inter_size, hidden_size)
    assert w2.shape == (num_experts, hidden_size, inter_size)
    assert torch.equal(w13, w13_orig)
    assert torch.equal(w2, w2_orig)
    assert moe.inter_size == inter_size
