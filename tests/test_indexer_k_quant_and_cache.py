# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401
from tests import register_ops as ops

torch.set_printoptions(threshold=torch.inf, precision=8)

eps = 1e-4

DEVICE = "xpu"
NUM_TOKENS = [1, 8, 17, 64]
HEAD_DIMS = [128, 256, 512]
QUANT_BLOCK_SIZES = [128]
BLOCK_SIZES = [16]
SCALE_FMTS = ["ue8m0", "fp8e4m3"]
# TODO: will add back torch.bfloat16, torch.float16
# after fp8_e4m3 acc is verified
DTYPES = [torch.float32]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1],
        "head_dim": [128],
        "quant_block_size": [128],
        "block_size": [16],
        "scale_fmt": ["ue8m0"],
        "dtype": [torch.float32],
    },
}


def _pytorch_group_quant(
    x: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_ue8m0 is None:
        use_ue8m0 = False
    if dtype is None:
        dtype = torch.float8_e4m3fn

    assert x.shape[-1] % group_size == 0
    assert x.stride(-1) == 1

    if out_q is None:
        x_q = torch.empty_like(x, dtype=dtype)
    else:
        assert out_q.shape == x.shape
        x_q = out_q

    original_shape = x.shape
    num_groups = original_shape[-1] // group_size
    group_shape = original_shape[:-1] + (num_groups, group_size)
    x_grouped = x.view(group_shape)

    abs_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=False)
    abs_max = torch.maximum(abs_max,
                            torch.tensor(eps, device=x.device, dtype=x.dtype))

    FP8_MAX = torch.finfo(dtype).max
    FP8_MIN = torch.finfo(dtype).min
    scale_raw = abs_max / FP8_MAX

    if use_ue8m0:
        scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))
    else:
        scales = scale_raw

    scales_expanded = scales.unsqueeze(-1)
    x_scaled = x_grouped / scales_expanded
    x_clamped = torch.clamp(x_scaled, FP8_MIN, FP8_MAX)
    x_quantized = x_clamped.to(dtype)
    x_q.copy_(x_quantized.view(original_shape))

    if column_major_scales:
        scales_shape = (num_groups, ) + original_shape[:-1]
        x_s = scales.permute(-1, *range(len(original_shape) - 1))
        x_s = x_s.contiguous().view(scales_shape)
    else:
        x_s = scales.contiguous()

    return x_q, x_s.float()


def ref_indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
) -> None:
    head_dim = k.shape[-1]
    num_groups = head_dim // quant_block_size
    block_size = kv_cache.shape[1]
    cache_stride = kv_cache.shape[2]

    k_fp8, k_scale = _pytorch_group_quant(
        k,
        group_size=quant_block_size,
        column_major_scales=False,
        use_ue8m0=(scale_fmt == "ue8m0"),
    )

    k_fp8_bytes = k_fp8.view(torch.uint8)
    kv_cache_flat_bytes = kv_cache.view(-1)
    kv_cache_flat_float = kv_cache_flat_bytes.view(torch.float32)

    for i, slot_idx in enumerate(slot_mapping.flatten().tolist()):
        if slot_idx < 0:
            continue
        block_idx = slot_idx // block_size
        block_offset = slot_idx % block_size

        # fp8 写入: block内fp8区域用head_dim步进
        fp8_start = block_idx * block_size * cache_stride + \
        block_offset * head_dim
        kv_cache_flat_bytes[fp8_start:fp8_start + head_dim] = k_fp8_bytes[i]

        # scale 写入: fp8区域之后，用float索引
        for g in range(num_groups):
            scale_float_idx = (
                block_idx * block_size * cache_stride + block_size * head_dim +
                (block_offset * head_dim + g * quant_block_size) * 4 //
                quant_block_size)
            kv_cache_flat_float[scale_float_idx // 4] = k_scale[i, g]

    kv_cache.copy_(kv_cache_flat_bytes.view(kv_cache.shape))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("quant_block_size", QUANT_BLOCK_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("scale_fmt", SCALE_FMTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_indexer_k_quant_and_cache(num_tokens, head_dim, quant_block_size,
                                   block_size, scale_fmt, dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    assert head_dim % quant_block_size == 0, \
        f"head_dim {head_dim} must be divisible " \
        f"by quant_block_size {quant_block_size}"

    num_groups = head_dim // quant_block_size
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache_stride = head_dim + num_groups * 4

    torch.manual_seed(42)
    k = torch.randn((num_tokens, head_dim), device=DEVICE, dtype=dtype)

    kv_cache_ref = torch.zeros((num_blocks, block_size, cache_stride),
                               dtype=torch.uint8,
                               device=DEVICE)
    kv_cache_xpu = torch.zeros((num_blocks, block_size, cache_stride),
                               dtype=torch.uint8,
                               device=DEVICE)

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)

    ref_indexer_k_quant_and_cache(k, kv_cache_ref, slot_mapping,
                                  quant_block_size, scale_fmt)
    ops.indexer_k_quant_and_cache(k, kv_cache_xpu, slot_mapping,
                                  quant_block_size, scale_fmt)

    for block_idx in range(num_blocks):
        block_ref = kv_cache_ref[block_idx].view(-1)
        block_xpu = kv_cache_xpu[block_idx].view(-1)

        fp8_end = block_size * head_dim
        ref_fp8 = block_ref[:fp8_end]
        out_fp8 = block_xpu[:fp8_end]
        ref_scale = block_ref[fp8_end:].view(torch.float32)
        out_scale = block_xpu[fp8_end:].view(torch.float32)

        assert torch.equal(
            ref_fp8,
            out_fp8), (f"[block={block_idx}] FP8 mismatch: max diff="
                       f"{(ref_fp8.view(torch.float8_e4m3fn).float() - \
            out_fp8.view(torch.float8_e4m3fn).float()).abs().max()}")
        assert torch.allclose(
            ref_scale, out_scale,
            atol=1e-5), (f"[block={block_idx}] Scale mismatch: "
                         f"ref={ref_scale}, out={out_scale}")


if __name__ == "__main__":
    test_indexer_k_quant_and_cache(
        num_tokens=17,
        head_dim=256,
        quant_block_size=128,
        block_size=16,
        scale_fmt="ue8m0",
        dtype=torch.float32,
    )
