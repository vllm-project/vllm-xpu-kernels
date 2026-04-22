# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICES = ["xpu:0"]


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, :block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim:] = sf.view(
        num_blocks,
        block_size,
    ).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor,
    dims: tuple,
    use_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def _generate_cp_test_data(seq_len: int, seq_len_kv: int, device: str):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
    ke = torch.zeros(seq_len, dtype=torch.int32, device=device)
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def _pytorch_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    seq_len_kv = kv.shape[0]
    k = kv.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    mask_lo = (
        torch.arange(
            0,
            seq_len_kv,
            device=q.device,
        )[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(
            0,
            seq_len_kv,
            device=q.device,
        )[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits


def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, _, dim = q.size()
    num_blocks, block_size, _, _ = kv_cache.size()

    kv_cache = kv_cache.view(num_blocks, -1)
    kv_cache_value = kv_cache[:, :block_size * dim].view(
        num_blocks, block_size, 1, dim)
    kv_cache_scale = kv_cache[:, block_size * dim:].view(
        num_blocks,
        block_size,
        1,
        4,
    ).view(torch.float32)

    q = q.float()
    kv_cache_value = kv_cache_value.view(fp8_dtype).float() * kv_cache_scale

    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(
            context_len - next_n,
            context_len,
            device=q.device,
        )
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(
            0, 1).contiguous()
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache_value[block_idx]
            k_offsets = torch.arange(block_rk * block_size,
                                     (block_rk + 1) * block_size,
                                     device=q.device)
            mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <=
                                                         q_offsets[:, None])
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n:(i + 1) * next_n,
                block_rk * block_size:(block_rk + 1) * block_size,
            ] = torch.where(
                k_offsets[None, :] <= q_offsets[:, None],
                s,
                float("-inf"),
            )
    return logits


@pytest.mark.parametrize("seq_len_qkv", [(512, 1024), (24, 24)])
@pytest.mark.parametrize("disable_cp", [True, False])
@pytest.mark.parametrize("device", DEVICES)
def test_fp8_mqa_logits_xpu(seq_len_qkv, disable_cp, device):
    torch.manual_seed(0)
    random.seed(0)
    num_heads, head_dim = 64, 128

    seq_len, seq_len_kv = seq_len_qkv

    q = torch.randn(seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv,
                     head_dim,
                     device=device,
                     dtype=torch.bfloat16)
    weights = torch.randn(seq_len,
                          num_heads,
                          device=device,
                          dtype=torch.float32)

    if disable_cp:
        ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
        ke = torch.arange(seq_len, dtype=torch.int32, device=device) + (
            seq_len_kv - seq_len)
    else:
        ks, ke = _generate_cp_test_data(seq_len, seq_len_kv, device=device)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0, ), False)

    ref_logits = _pytorch_mqa_logits(
        q=q_fp8,
        kv=kv_fp8,
        scale=kv_scales,
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
    )
    logits = torch.ops._xpu_C.fp8_mqa_logits(q_fp8, kv_fp8, kv_scales, weights,
                                             ks, ke)

    ref_neginf_mask = ref_logits == float("-inf")
    neginf_mask = logits == float("-inf")
    assert torch.equal(neginf_mask, ref_neginf_mask)

    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
    logits = logits.masked_fill(neginf_mask, 0)
    diff = calc_diff(logits, ref_logits)
    assert diff < 1e-3, f"{diff=}"


@pytest.mark.parametrize("bs_nextn", [(4, 1), (4, 2), (2, 2)])
@pytest.mark.parametrize("device", DEVICES)
def test_fp8_paged_mqa_logits_xpu(bs_nextn, device):
    torch.manual_seed(0)
    random.seed(0)

    batch_size, next_n = bs_nextn
    max_model_len = 4096

    for heads, index_dim in [(64, 128)]:
        for avg_kv in (2048, ):
            num_blocks, blocksize = max_model_len * 2, 64

            q = torch.randn((batch_size, next_n, heads, index_dim),
                            device=device,
                            dtype=torch.bfloat16)
            kv_cache = torch.randn((num_blocks, blocksize, 1, index_dim),
                                   device=device,
                                   dtype=torch.bfloat16)
            weights = torch.randn((batch_size * next_n, heads),
                                  device=device,
                                  dtype=torch.float32)

            context_lens = torch.randint(int(0.8 * avg_kv),
                                         int(1.2 * avg_kv),
                                         (batch_size, ),
                                         device=device,
                                         dtype=torch.int32)
            max_blocks = (
                (context_lens.max().item() + blocksize - 1) // blocksize
            )
            block_tables = torch.zeros((batch_size, max_blocks),
                                       device=device,
                                       dtype=torch.int32)

            counter = 0
            block_idx_pool = list(range(num_blocks))
            random.shuffle(block_idx_pool)
            for i in range(batch_size):
                ctx_len = int(context_lens[i].item())
                for j in range((ctx_len + blocksize - 1) // blocksize):
                    block_tables[i][j] = block_idx_pool[counter]
                    counter += 1

            q_fp8 = q.to(torch.float8_e4m3fn)
            kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)
            schedule_metadata = None

            ref_logits = fp8_paged_mqa_logits_torch(
                q_fp8,
                kv_cache_fp8,
                weights,
                context_lens,
                block_tables,
                max_model_len,
            )

            logits = torch.ops._xpu_C.fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                context_lens,
                block_tables,
                schedule_metadata,
                max_model_len,
            )

            ref_neginf_mask = ref_logits == float("-inf")
            neginf_mask = logits == float("-inf")
            assert torch.equal(neginf_mask, ref_neginf_mask)

            positions = torch.arange(
                max_model_len, device=device).unsqueeze(0).expand(
                    batch_size * next_n, -1)
            row_indices = torch.arange(
                batch_size * next_n, device=device) // next_n
            next_n_offset = torch.arange(
                batch_size * next_n, device=device) % next_n
            mask = positions <= (
                context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)

            logits = logits.masked_fill(~mask, 0)
            ref_logits = ref_logits.masked_fill(~mask, 0)
            diff = calc_diff(logits, ref_logits)
            assert diff < 1e-3, f"{diff=}"
