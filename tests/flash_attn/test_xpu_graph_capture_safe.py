# SPDX-License-Identifier: Apache-2.0
"""Fail-closed-by-default + capture auto-on for FA2 / flash_attn_varlen_func."""
from __future__ import annotations

import logging

import pytest
import torch

import vllm_xpu_kernels.flash_attn_interface as fai
from vllm_xpu_kernels.flash_attn_interface import (
    build_decode_split_plan,
    flash_attn_varlen_func,
)

_ALLOW_ENV = "VLLM_XPU_ATTN_ALLOW_FALLBACK"


def _truthy_env(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)


def _clear_allow_env(monkeypatch) -> None:
    monkeypatch.delenv(_ALLOW_ENV, raising=False)


@pytest.mark.parametrize("val,expect", [
    ("1", True),
    ("ON", True),
    ("true", True),
    ("Yes", True),
    ("0", False),
    ("", False),
])
def test_env_allow_fallback(monkeypatch, val, expect):
    if val == "":
        _clear_allow_env(monkeypatch)
    else:
        _truthy_env(monkeypatch, _ALLOW_ENV, val)
    assert fai._env_allow_fallback() is expect


def test_should_fail_closed_by_default(monkeypatch):
    _clear_allow_env(monkeypatch)
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: False)
    assert fai._should_fail_closed() is True


def test_should_fail_closed_when_capturing_even_if_allow(monkeypatch):
    _truthy_env(monkeypatch, _ALLOW_ENV, "1")
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: True)
    assert fai._should_fail_closed() is True


def test_should_not_fail_closed_when_allow_and_eager(monkeypatch):
    _truthy_env(monkeypatch, _ALLOW_ENV, "1")
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: False)
    assert fai._should_fail_closed() is False


def test_build_decode_split_plan_accepts_cpu_and_list():
    splits, work = build_decode_split_plan(
        [128, 256],
        kv_tile=64,
        num_kv_splits=8,
        num_xe_cores=20,
        num_heads_kv=2,
    )
    assert splits.dtype == torch.int32
    assert work.ndim == 2 and work.size(1) == 4
    splits2, _ = build_decode_split_plan(
        torch.tensor([128, 256], dtype=torch.int32, device="cpu"),
        kv_tile=64,
        num_kv_splits=8,
        num_xe_cores=20,
        num_heads_kv=2,
    )
    assert torch.equal(splits, splits2)


def test_build_decode_split_plan_rejects_device_tensor():
    if not torch.xpu.is_available():
        # Still validate the check using a meta/cpu-fake path: construct
        # a non-CPU tensor if CUDA/XPU unavailable by skipping.
        pytest.skip("Need a non-CPU device to exercise the D2H guard")
    kv = torch.tensor([128, 256], dtype=torch.int32, device="xpu")
    with pytest.raises(RuntimeError, match="host-side kv_lens"):
        build_decode_split_plan(
            kv,
            kv_tile=64,
            num_kv_splits=8,
            num_xe_cores=20,
            num_heads_kv=2,
        )


def _minimal_varlen_args(device: torch.device, *, with_out: bool = False):
    headdim = 64
    nq, nkv = 8, 2
    block = 64
    batch = 1
    kv_len = 128
    q = torch.randn(batch, nq, headdim, dtype=torch.float16, device=device)
    k = torch.randn(2, block, nkv, headdim, dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused = torch.tensor([kv_len], dtype=torch.int32, device=device)
    block_table = torch.zeros(batch, 2, dtype=torch.int32, device=device)
    block_table[0, 0] = 0
    block_table[0, 1] = 1
    args = dict(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=1,
        cu_seqlens_q=cu_q,
        max_seqlen_k=kv_len,
        seqused_k=seqused,
        block_table=block_table,
        softmax_scale=headdim**-0.5,
        causal=True,
    )
    if with_out:
        args["out"] = torch.empty_like(q)
    return args


def _require_fa2_xpu():
    if not fai.FA2_AVAILABLE:
        pytest.skip("FA2 extension not available")
    if not torch.xpu.is_available():
        pytest.skip("XPU required for FA2 dispatch path")


def test_default_raises_on_not_compiled(monkeypatch):
    _require_fa2_xpu()
    _clear_allow_env(monkeypatch)
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: False)

    def boom(*_a, **_k):
        raise RuntimeError("Paged decode kernel not compiled for this config")

    monkeypatch.setattr(torch.ops._vllm_fa2_C, "varlen_fwd", boom)
    args = _minimal_varlen_args(torch.device("xpu"))
    with pytest.raises(RuntimeError, match="Fail-closed|refusing"):
        flash_attn_varlen_func(**args)


def test_capturing_raises_even_with_allow_fallback(monkeypatch):
    _require_fa2_xpu()
    _truthy_env(monkeypatch, _ALLOW_ENV, "1")
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: True)

    def boom(*_a, **_k):
        raise RuntimeError("Paged decode kernel not compiled for this config")

    monkeypatch.setattr(torch.ops._vllm_fa2_C, "varlen_fwd", boom)
    # Capture requires a preallocated out before the op is invoked.
    args = _minimal_varlen_args(torch.device("xpu"), with_out=True)
    with pytest.raises(RuntimeError, match="Fail-closed|refusing|capture"):
        flash_attn_varlen_func(**args)


def test_capturing_requires_preallocated_out(monkeypatch):
    _require_fa2_xpu()
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: True)
    args = _minimal_varlen_args(torch.device("xpu"), with_out=False)
    with pytest.raises(RuntimeError, match="preallocated `out`"):
        flash_attn_varlen_func(**args)


def test_device_host_kv_lens_split_plan_raises(monkeypatch):
    _require_fa2_xpu()
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: True)

    def boom(*_a, **_k):
        raise RuntimeError("should not reach varlen_fwd")

    monkeypatch.setattr(torch.ops._vllm_fa2_C, "varlen_fwd", boom)
    device = torch.device("xpu")
    args = _minimal_varlen_args(device, with_out=True)
    args.pop("seqused_k")
    args["host_kv_lens"] = torch.tensor([128], dtype=torch.int32, device=device)
    args["num_splits_kv"] = 4
    with pytest.raises(RuntimeError, match="host-side kv_lens"):
        flash_attn_varlen_func(**args)


def test_spec_decode_skipped_when_capturing(monkeypatch):
    _require_fa2_xpu()
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: True)
    called = {"spec": False, "fwd": False}

    def fake_spec(*_a, **_k):
        called["spec"] = True
        return torch.empty(0)

    def fake_fwd(*_a, **_k):
        called["fwd"] = True
        q = _a[0]
        return (torch.empty_like(q), None)

    monkeypatch.setattr(fai, "_spec_decode_varlen_fwd", fake_spec)
    monkeypatch.setattr(torch.ops._vllm_fa2_C, "varlen_fwd", fake_fwd)

    device = torch.device("xpu")
    headdim = 64
    batch, q_len, nq, nkv = 2, 4, 8, 2
    block = 64
    q = torch.randn(
        batch * q_len, nq, headdim, dtype=torch.float16, device=device)
    k = torch.randn(
        8, block, nkv, headdim, dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    seqused = torch.tensor([128, 128], dtype=torch.int32, device=device)
    block_table = torch.zeros(batch, 4, dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=q_len,
        cu_seqlens_q=cu_q,
        max_seqlen_k=128,
        seqused_k=seqused,
        block_table=block_table,
        softmax_scale=headdim**-0.5,
        causal=True,
        out=out,
    )
    assert called["spec"] is False
    assert called["fwd"] is True


def test_fallback_allowed_when_opt_in_eager(monkeypatch, caplog):
    _require_fa2_xpu()
    _truthy_env(monkeypatch, _ALLOW_ENV, "1")
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: False)

    def boom(*_a, **_k):
        raise RuntimeError("Paged decode kernel not compiled for this config")

    monkeypatch.setattr(torch.ops._vllm_fa2_C, "varlen_fwd", boom)
    args = _minimal_varlen_args(torch.device("xpu"))
    with caplog.at_level(logging.WARNING, logger=fai.__name__):
        out = flash_attn_varlen_func(**args)
    assert out is not None
    assert any("falling back" in r.message for r in caplog.records)


@pytest.mark.skipif(
    not hasattr(torch.xpu, "XPUGraph") or not torch.xpu.is_available(),
    reason="torch.xpu.XPUGraph not available")
@pytest.mark.xfail(
    reason="FA2 uses work_group_scratch_memory; SYCL Graph / XPUGraph "
    "support is incomplete on current stacks (see HAL capture notes).",
    strict=False,
)
def test_xpugraph_capture_replay_smoke():
    """Capture→replay on a compiled paged-decode shape when FA2 is present."""
    if not fai.FA2_AVAILABLE:
        pytest.skip("FA2 extension not available")

    device = torch.device("xpu")
    headdim = 128
    nq, nkv = 8, 1
    block = 64
    batch = 1
    kv_len = 128
    q = torch.randn(batch, nq, headdim, dtype=torch.float16, device=device)
    k = torch.randn(
        2, block, nkv, headdim, dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused = torch.tensor([kv_len], dtype=torch.int32, device=device)
    block_table = torch.zeros(batch, 2, dtype=torch.int32, device=device)
    block_table[0, 0] = 0
    block_table[0, 1] = 1
    out = torch.empty_like(q)
    scale = headdim**-0.5

    def run():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=kv_len,
            seqused_k=seqused,
            block_table=block_table,
            softmax_scale=scale,
            causal=True,
            out=out,
        )

    # Warmup eager path; skip if this shape is not compiled (fail-closed).
    try:
        for _ in range(3):
            run()
            torch.xpu.synchronize()
    except RuntimeError as e:
        if "not compiled" in str(e) or "Fail-closed" in str(e):
            pytest.skip(f"shape not compiled in this build: {e}")
        raise

    g = torch.xpu.XPUGraph()
    capture_ctx = getattr(torch.xpu, "graph", None)
    if capture_ctx is None:
        pytest.skip("torch.xpu.graph context manager not available")
    with capture_ctx(g):
        run()
    for _ in range(5):
        g.replay()
        torch.xpu.synchronize()
    assert torch.isfinite(out).all()
