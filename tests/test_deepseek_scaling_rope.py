# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.register_ops import deepseek_scaling_rope

DEVICE = torch.device("xpu")

#override pytest parameters when 'USE_SIMULATOR' eq 1
SIMULATOR_PYTEST_PARAMS = {
    "default": {
        "batch": [1],
    },
}


def _rotate_neox(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    res = x.flatten(-2)
    return res


class TestTorchMethod:

    def ref_deepseek_scaling_rope(
        self,
        positions,
        query,
        key,
        cos_sin_cache,
        rotary_dim,
        head_size,
        is_neox_style,
        offsets=None,
    ):
        query_rot = query[..., :rotary_dim]
        key_rot = key[..., :rotary_dim]
        if rotary_dim < head_size:
            query_pass = query[..., rotary_dim:]
            key_pass = key[..., rotary_dim:]

        cos_sin_cache = cos_sin_cache.to(positions.device)
        cos_sin = cos_sin_cache[torch.
                                add(positions, offsets
                                    ) if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        rotate_fn = _rotate_neox if is_neox_style else _rotate_gptj

        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if rotary_dim < head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key

    @pytest.mark.parametrize("seed", [123, 356, 478])
    @pytest.mark.parametrize("dtype",
                             [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("batch", [1, 2, 16, 32])
    @pytest.mark.parametrize("q_num_head,k_num_head", [(16, 1), (32, 1)])
    @pytest.mark.parametrize("rotary_dim", [64, 128])
    @pytest.mark.parametrize("q_head_pad,k_head_pad", [(0, 0), (128, 512)])
    @pytest.mark.parametrize("is_neox", [True, False])
    def test_deepseek_scaling_rope(
        self,
        seed,
        dtype,
        batch,
        q_num_head,
        k_num_head,
        rotary_dim,
        q_head_pad,
        k_head_pad,
        is_neox,
    ):
        torch.manual_seed(seed)
        head_size = rotary_dim
        # if rotary_dim < head_size, reference code wrong behavior
        # and not going to fix the original code
        positions = torch.randint(0, batch * 10000, (batch, ), device=DEVICE)
        cos_sin_cache = torch.randn(batch * 10000, rotary_dim,
                                    device=DEVICE).to(dtype)
        q_head_size_pad = q_head_pad + head_size
        k_head_size_pad = k_head_pad + head_size
        query_pad = torch.randn(batch,
                                q_num_head,
                                q_head_size_pad,
                                device=DEVICE).to(dtype)
        key_pad = torch.randn(batch,
                              k_num_head,
                              k_head_size_pad,
                              device=DEVICE).to(dtype)
        query = query_pad[..., :head_size]
        key = key_pad[..., :head_size]
        ref_query, ref_key = self.ref_deepseek_scaling_rope(
            positions, query, key, cos_sin_cache, rotary_dim, head_size,
            is_neox)
        query_out, key_out = deepseek_scaling_rope(positions, query, key, None,
                                                   cos_sin_cache, rotary_dim,
                                                   is_neox)
        torch.testing.assert_close(ref_query, query_out, atol=5e-3, rtol=1e-3)
        torch.testing.assert_close(ref_key, key_out, atol=5e-3, rtol=1e-3)
