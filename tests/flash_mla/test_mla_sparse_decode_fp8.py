# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_xpu_kernels.flash_mla_interface import flash_mla_with_kvcache

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "has_attn_sink": [True],
        "has_extra": [True],
        "h_q": [8],
        "s_q": [1],
        "topk": [66],
        "extra_topk": [28],
    },
}

# DeepSeek V4 specific constants
NOPE_DIM = 448
ROPE_DIM = 64
D_QK = NOPE_DIM + ROPE_DIM
D_V = 512
DATA_BYTES_PER_TOKEN = NOPE_DIM + ROPE_DIM * 2
SCALE_BYTES_PER_TOKEN = 8
HEAD_BYTES = DATA_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN


def _pack_sparse_fp8_kv_deepseek_v4(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, block_size, num_heads, d_qk = kv.shape
    assert num_heads == 1
    assert d_qk == D_QK

    packed_storage = torch.empty(
        (num_blocks, block_size * HEAD_BYTES), device=kv.device, dtype=torch.uint8
    )
    dequant = torch.empty_like(kv, dtype=torch.float32)

    nope = kv[:, :, 0, :NOPE_DIM].float()
    rope = kv[:, :, 0, NOPE_DIM:].contiguous()
    scale_bytes = torch.full(
        (num_blocks, block_size, SCALE_BYTES_PER_TOKEN),
        127,
        device=kv.device,
        dtype=torch.uint8,
    )
    scale_bytes[..., 7] = 0

    nope_fp8 = nope.to(torch.float8_e4m3fn)
    dequant[:, :, 0, :NOPE_DIM] = nope_fp8.float()
    dequant[:, :, 0, NOPE_DIM:] = rope.float()

    data_region = packed_storage[:, : block_size * DATA_BYTES_PER_TOKEN].view(
        num_blocks, block_size, DATA_BYTES_PER_TOKEN
    )
    scale_region = packed_storage[:, block_size * DATA_BYTES_PER_TOKEN :].view(
        num_blocks, block_size, SCALE_BYTES_PER_TOKEN
    )

    data_region[..., :NOPE_DIM] = nope_fp8.contiguous().view(torch.uint8)
    data_region[..., NOPE_DIM:] = rope.view(torch.uint8).view(
        num_blocks, block_size, ROPE_DIM * 2
    )
    scale_region.copy_(scale_bytes)

    packed = packed_storage.as_strided(
        (num_blocks, block_size, num_heads, HEAD_BYTES),
        (block_size * HEAD_BYTES, DATA_BYTES_PER_TOKEN, HEAD_BYTES, 1),
    ).view(torch.float8_e4m3fn)
    return packed, dequant

# adjusted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L55
def _ref_sparse_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_kv: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, s_q, h_q, _ = q.shape
    q = q.float().view(b * s_q, h_q, D_QK)

    def _normalize_topk_length(active_tl: torch.Tensor) -> torch.Tensor:
        if active_tl.dim() == 1:
            return active_tl.view(b, 1, 1)
        return active_tl.unsqueeze(-1)

    def _process_kv_scope(
        active_kv: torch.Tensor,
        active_indices: torch.Tensor,
        active_tl: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk = active_indices.size(-1)
        active_flat = active_kv.view(-1, D_QK)

        # Keep the official invalid semantics and also guard out-of-range
        # positive indices used by this test.
        invalid_mask = (active_indices < 0) | (active_indices >= active_flat.size(0))
        if active_tl is not None:
            normalized_tl = _normalize_topk_length(active_tl)
            invalid_mask |= torch.arange(0, topk, device=active_indices.device).view(
                1, 1, topk
            ).expand(b, s_q, topk) >= normalized_tl

        gather_idx = torch.clamp(active_indices, min=0, max=active_flat.size(0) - 1)
        gathered = active_flat.index_select(0, gather_idx.reshape(-1)).view(
            b, s_q, topk, D_QK
        )
        return gathered, invalid_mask

    gathered_kv, invalid_mask = _process_kv_scope(kv, indices, topk_length)
    if extra_kv is not None and extra_indices is not None:
        extra_gathered_kv, extra_invalid_mask = _process_kv_scope(
            extra_kv, extra_indices, extra_topk_length
        )
        gathered_kv = torch.cat([gathered_kv, extra_gathered_kv], dim=2)
        invalid_mask = torch.cat([invalid_mask, extra_invalid_mask], dim=2)

    gathered_kv = gathered_kv.view(b * s_q, -1, D_QK).float()
    gathered_kv[gathered_kv != gathered_kv] = 0.0

    attn_weight = q @ gathered_kv.transpose(-1, -2)
    attn_weight *= sm_scale
    attn_weight[
        invalid_mask.view(b * s_q, 1, -1).expand(b * s_q, h_q, invalid_mask.size(-1))
    ] = float("-inf")

    lse = attn_weight.logsumexp(dim=-1).view(b, s_q, h_q)
    probs = torch.exp(attn_weight - lse.view(b * s_q, h_q, 1))
    output = probs @ gathered_kv[..., :D_V]
    output = output.view(b, s_q, h_q, D_V)

    if attn_sink is not None:
        output *= (1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))).unsqueeze(-1)

    lonely_q_mask = lse == float("-inf")
    output[lonely_q_mask.unsqueeze(-1).expand(b, s_q, h_q, D_V)] = 0.0
    lse = lse.masked_fill(lonely_q_mask, float("inf"))
    return output.to(torch.bfloat16), lse


@pytest.mark.parametrize("has_attn_sink", [False, True])
@pytest.mark.parametrize("has_extra", [False, True])
@pytest.mark.parametrize("h_q", [8, 22, 64, 128])
@pytest.mark.parametrize("s_q", [15, 33, 88])
@pytest.mark.parametrize("topk", [1, 66, 288])
@pytest.mark.parametrize("extra_topk", [1000, 2048])
def test_mla_sparse_decode_fp8_fwd(has_attn_sink: bool, has_extra: bool, h_q: int, s_q: int, topk: int, extra_topk: int):
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    device = "xpu"
    torch.manual_seed(8)

    b = s_q
    s_q = 1
    num_blocks = 64
    block_size = 128
    extra_num_blocks = 128
    extra_block_size = 128
    sm_scale = D_QK ** -0.5

    q_ref = torch.randn((b, s_q, h_q, D_QK), device=device, dtype=torch.float32) * 0.5
    q = q_ref.to(torch.bfloat16)
    q_for_ref = q.float()

    logical_kv = torch.randn((num_blocks, block_size, 1, D_QK), device=device, dtype=torch.bfloat16) * 0.5
    packed_kv, dequant_kv = _pack_sparse_fp8_kv_deepseek_v4(logical_kv)

    indices = torch.randint(0, num_blocks * block_size, (b, s_q, topk), device=device, dtype=torch.int32)
    indices[0, 0, -1] = num_blocks * block_size + 9
    topk_length = torch.randint(1, topk + 1, (b,), device=device, dtype=torch.int32)

    attn_sink = None
    if has_attn_sink:
        attn_sink = torch.randn((h_q,), device=device, dtype=torch.float32) * 0.25

    packed_extra_kv = None
    dequant_extra_kv = None
    extra_indices = None
    extra_topk_length = None
    if has_extra:
        logical_extra_kv = torch.randn(
            (extra_num_blocks, extra_block_size, 1, D_QK),
            device=device,
            dtype=torch.bfloat16,
        ) * 0.5
        packed_extra_kv, dequant_extra_kv = _pack_sparse_fp8_kv_deepseek_v4(logical_extra_kv)
        extra_indices = torch.randint(
            0,
            extra_num_blocks * extra_block_size,
            (b, s_q, extra_topk),
            device=device,
            dtype=torch.int32,
        )
        extra_topk_length = torch.randint(1, extra_topk + 1, (b,), device=device, dtype=torch.int32)

    out, lse = flash_mla_with_kvcache(
        q=q,
        k_cache=packed_kv,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=D_V,
        tile_scheduler_metadata=None,
        out=None,
        num_splits=None,
        softmax_scale=sm_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=packed_extra_kv,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        return_softmax_lse=True,
    )

    ref_out, ref_lse = _ref_sparse_decode(
        q_for_ref,
        dequant_kv,
        indices,
        sm_scale,
        topk_length,
        attn_sink,
        dequant_extra_kv,
        extra_indices,
        extra_topk_length,
    )

    assert out.shape == (b, s_q, h_q, D_V)
    assert lse is not None
    assert lse.shape == (b, s_q, h_q)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
