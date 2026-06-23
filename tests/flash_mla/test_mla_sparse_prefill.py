# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_xpu_kernels.flash_mla_interface import flash_mla_sparse_fwd


# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "has_attn_sink": [True],
        "has_topk_length": [True],
        "s_q": [2],
        "h_q": [8],
        "topk": [88],
    },
}


def _merge_two_lse(
    lse0: torch.Tensor, lse1: torch.Tensor | None, s_q: int, h_q: int
) -> torch.Tensor:
    if lse1 is None:
        return lse0
    return torch.logsumexp(
        torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
        dim=0,
    )

# adjusted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L19
def _reference_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]

    idx = indices.clone().squeeze(1)
    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0) >= topk_length.unsqueeze(1)
        idx[mask] = -1
    invalid_mask = (idx < 0) | (idx >= s_kv)
    idx[invalid_mask] = 0

    q_f = q.float()
    gathered = kv.index_select(0, idx.flatten()).reshape(s_q, topk, d_qk).float()
    logits = q_f @ gathered.transpose(1, 2)
    logits *= sm_scale
    logits[invalid_mask.unsqueeze(1).expand_as(logits)] = float("-inf")

    lse = torch.logsumexp(logits, dim=-1)
    max_logits = logits.max(dim=-1).values

    lse_for_out = _merge_two_lse(lse, attn_sink, s_q, h_q)
    lse_for_out = lse_for_out.clone()
    lse_for_out[lse_for_out == float("-inf")] = float("inf")
    probs = torch.exp(logits - lse_for_out.unsqueeze(-1))
    out = probs @ gathered[..., :d_v]

    lse[lse == float("-inf")] = float("inf")
    return out.to(kv.dtype), max_logits, lse


@pytest.mark.parametrize("has_attn_sink", [False, True])
@pytest.mark.parametrize("has_topk_length", [False, True])
@pytest.mark.parametrize("d_qk", [512, 576])
@pytest.mark.parametrize("s_q", [512, 576])
@pytest.mark.parametrize("h_q", [8, 22, 64, 128])
@pytest.mark.parametrize("topk", [1, 128])
def test_mla_sparse_prefill_fwd(d_qk: int, has_topk_length: bool, has_attn_sink: bool, h_q: int, s_q: int, topk: int):
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    device = "xpu"
    dtype = torch.bfloat16
    torch.manual_seed(123)

    s_kv = 2048
    h_kv = 1
    d_v = 512

    q = torch.randn((s_q, h_q, d_qk), device=device, dtype=dtype)
    kv = torch.randn((s_kv, h_kv, d_qk), device=device, dtype=dtype)
    indices = torch.randint(0, s_kv, (s_q, h_kv, topk), device=device, dtype=torch.int32)
    indices[0, 0, -1] = s_kv + 7

    topk_length = None
    if has_topk_length:
        topk_length = torch.randint(1, topk + 1, (s_q,), device=device, dtype=torch.int32)

    attn_sink = None
    if has_attn_sink:
        attn_sink = torch.randn((h_q,), device=device, dtype=torch.float32)

    sm_scale = d_qk ** -0.5
    ref_out, ref_max, ref_lse = _reference_sparse_prefill(
        q, kv, indices, sm_scale, d_v, topk_length=topk_length, attn_sink=attn_sink
    )

    out, max_logits, lse = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        return_softmax_lse=True,
    )

    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)

    torch.testing.assert_close(max_logits, ref_max, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
