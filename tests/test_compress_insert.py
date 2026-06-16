# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the SYCL xpu_compress_insert_fp8mix kernel.

Covers both layer configurations from DeepSeek-V4-FP8-tiny:
  Layer A: compress_ratio=4,   overlap=1
  Layer B: compress_ratio=128, overlap=0
"""
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401 - registers torch.ops._xpu_C ops
from tests.utils import opcheck

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU is required for xpu_compress_insert tests",
)

HEAD_SIZE = 512
ROPE_HEAD_DIM = 64
RMS_EPS = 1e-6
NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM
TOKEN_STRIDE = 576
SCALE_DIM = 8

_xpu_count = torch.xpu.device_count() if hasattr(torch, "xpu") else 1
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if _xpu_count == 1 else 2)
]

DEFAULT_OPCHECK_TEST_UTILS: tuple[str, ...] = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
)

LAYER_CONFIGS = [
    # (compress_ratio, overlap, kv_cache_block_size, state_blk_sz, label)
    (4,   1, 64, 4, "A_cr4"),
    (128, 0,  2, 8, "B_cr128"),
]

NUM_TOKENS_MINI = [1, 16]
FP8MIX_NUM_TOKENS = [1, 5, 16, 33]

FP8MIX_CASES = [
    # (label, make_non_boundary, invalidate_slot_mapping, invalidate_kv_slot)
    ("all_valid_boundary", False, False, False),
    ("mixed_boundary", True, False, False),
    ("neg_slot", False, True, False),
    ("neg_kv_slot", False, False, True),
    ("mixed_boundary_neg_slot", True, True, False),
    ("mixed_boundary_neg_kv_slot", True, False, True),
]

MINI_PYTEST_PARAMS = {
    "default": {
        "config": LAYER_CONFIGS[:1],
        "num_tokens": NUM_TOKENS_MINI,
    },
}


def _quantize_nope_to_fp8mix(
    pre_rope_nope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize 448-dim NOPE to FP8 bytes and 7 ue8m0 scales (+1 pad later)."""
    assert pre_rope_nope.numel() == NOPE_HEAD_DIM
    fp8_bytes = torch.empty(NOPE_HEAD_DIM, dtype=torch.uint8)
    scales = torch.empty(NOPE_HEAD_DIM // 64, dtype=torch.uint8)

    for b in range(NOPE_HEAD_DIM // 64):
        s = b * 64
        e = s + 64
        blk = pre_rope_nope[s:e]
        absmax = torch.clamp(blk.abs().max(), min=1e-4)
        raw_scale = absmax / 448.0
        exponent = torch.ceil(torch.log2(raw_scale))
        inv_scale = float(2.0 ** (-float(exponent.item())))

        encoded = torch.clamp(exponent + 127.0, 0.0, 255.0)
        scales[b] = encoded.to(torch.uint8)

        q = torch.clamp(blk * inv_scale, -448.0, 448.0)
        fp8_blk = q.to(torch.float8_e4m3fn).view(torch.uint8)
        fp8_bytes[s:e] = fp8_blk

    return fp8_bytes, scales


def _ref_c4_kv_insert_fp8mix(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache_u8: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
    overlap: int,
    rope_head_dim: int,
    token_stride: int = TOKEN_STRIDE,
    scale_dim: int = SCALE_DIM,
) -> None:
    state_width = state_cache.shape[-1] // 2
    head_size = state_width // (1 + overlap)
    state_block_size = state_cache.shape[1]
    n_gather = (1 + overlap) * compress_ratio
    nope_head_dim = head_size - rope_head_dim
    half_rope = rope_head_dim // 2
    kv_block_stride = int(k_cache_u8.stride(0))

    # Flatten as raw bytes to match kernel addressing.
    raw = k_cache_u8.reshape(-1)

    for token_idx in range(positions.numel()):
        if int(slot_mapping[token_idx].item()) < 0:
            continue
        position = int(positions[token_idx].item())
        if (position + 1) % compress_ratio != 0:
            continue

        req_idx = int(token_to_req_indices[token_idx].item())
        start = position - n_gather + 1
        kv_rows, score_rows = [], []
        for gi in range(n_gather):
            gp = start + gi
            if gp < 0:
                kv_rows.append(torch.zeros(head_size, dtype=torch.float32,
                                           device=state_cache.device))
                score_rows.append(torch.full((head_size,), float("-inf"),
                                             dtype=torch.float32,
                                             device=state_cache.device))
                continue
            lb = gp // state_block_size
            sb = int(block_table[req_idx, lb].item())
            pb = gp % state_block_size
            row = state_cache[sb, pb].float()
            head_off = head_size if gi >= compress_ratio else 0
            kv_rows.append(row[head_off: head_off + head_size])
            score_off = state_width + head_off
            score_rows.append(row[score_off: score_off + head_size])

        kv = torch.stack(kv_rows, dim=0)
        weights = torch.softmax(torch.stack(score_rows, dim=0), dim=0)
        compressed = (kv * weights).sum(dim=0)

        variance = compressed.pow(2).mean()
        normed = compressed * torch.rsqrt(variance + rms_norm_eps)
        normed = normed * rms_norm_weight.float()

        # Triton parity: bf16 roundtrip on pre-rope values before FP8 quant.
        pre_rope = normed.to(torch.bfloat16).to(torch.float32)
        post_rope = normed.clone()

        if rope_head_dim > 0:
            cp = (position // compress_ratio) * compress_ratio
            cos = cos_sin_cache[cp, :half_rope].float()
            sin = cos_sin_cache[cp, half_rope:].float()
            rope = post_rope[nope_head_dim:].reshape(half_rope, 2)
            rotated = torch.empty_like(rope)
            rotated[:, 0] = rope[:, 0] * cos - rope[:, 1] * sin
            rotated[:, 1] = rope[:, 1] * cos + rope[:, 0] * sin
            post_rope = post_rope.clone()
            post_rope[nope_head_dim:] = rotated.reshape(rope_head_dim)

        kv_slot = int(kv_slot_mapping[token_idx].item())
        if kv_slot < 0:
            continue
        kv_block = kv_slot // kv_cache_block_size
        kv_pos = kv_slot % kv_cache_block_size

        fp8_nope, scales = _quantize_nope_to_fp8mix(pre_rope[:nope_head_dim])
        rope_bytes = (
            post_rope[nope_head_dim:]
            .to(torch.bfloat16)
            .view(torch.uint8)
        )

        block_base = kv_block * kv_block_stride
        token_off = block_base + kv_pos * token_stride
        scale_off = block_base + kv_cache_block_size * token_stride + kv_pos * scale_dim

        raw[token_off: token_off + nope_head_dim] = fp8_nope
        raw[token_off + nope_head_dim: token_off + token_stride] = rope_bytes
        raw[scale_off: scale_off + 7] = scales
        raw[scale_off + 7] = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(compress_ratio, overlap, kv_cache_block_size, state_blk_sz,
                 num_tokens, device, seed=42):
    state_width = (1 + overlap) * HEAD_SIZE
    num_blocks = 16384
    gen = torch.Generator(device="cpu").manual_seed(seed)

    state_cache = (torch.randn(num_blocks, state_blk_sz, 2 * state_width,
                               generator=gen, device="cpu") * 0.1).to(device)
    k_cache = torch.zeros(num_blocks, kv_cache_block_size, HEAD_SIZE,
                          dtype=torch.bfloat16, device=device)
    block_table = torch.randint(0, num_blocks, (1, 4096),
                                dtype=torch.int32, generator=gen,
                                device="cpu").to(device)

    # All tokens on boundary positions so the kernel does real work.
    positions = ((torch.arange(num_tokens) + 1) * compress_ratio - 1).to(device)
    token_to_req_indices = torch.zeros(num_tokens, dtype=torch.int32,
                                       device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    kv_slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    rms_norm_weight = torch.ones(HEAD_SIZE, dtype=torch.float32, device=device)

    max_pos = int(positions.max().item()) + compress_ratio + 1024
    half = ROPE_HEAD_DIM // 2
    freq = torch.arange(half, dtype=torch.float32) / half
    base = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
    cos_sin_cache = torch.cat([
        torch.cos(base * (10000 ** (-freq))),
        torch.sin(base * (10000 ** (-freq))),
    ], dim=1).to(device)

    return dict(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=RMS_EPS,
        cos_sin_cache=cos_sin_cache,
        k_cache=k_cache,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rope_head_dim=ROPE_HEAD_DIM,
    )


def _make_inputs_small_fp8mix(
    compress_ratio,
    overlap,
    kv_cache_block_size,
    state_blk_sz,
    num_tokens,
    device,
    seed=42,
):
    """Memory-light inputs for fp8mix correctness tests.

    fp8mix tests validate byte-level layout and do not require huge cache pools.
    """
    state_width = (1 + overlap) * HEAD_SIZE
    num_blocks = 512
    max_position = int(((num_tokens + 1) * compress_ratio - 1))
    bt_cols = max(1024, max_position // state_blk_sz + 8)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    state_cache = (torch.randn(num_blocks, state_blk_sz, 2 * state_width,
                               generator=gen, device="cpu") * 0.1).to(device)
    k_cache = torch.zeros(num_blocks, kv_cache_block_size, HEAD_SIZE,
                          dtype=torch.bfloat16, device=device)
    block_table = torch.randint(0, num_blocks, (1, bt_cols),
                                dtype=torch.int32, generator=gen,
                                device="cpu").to(device)

    positions = ((torch.arange(num_tokens) + 1) * compress_ratio - 1).to(device)
    token_to_req_indices = torch.zeros(num_tokens, dtype=torch.int32,
                                       device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    kv_slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    rms_norm_weight = torch.ones(HEAD_SIZE, dtype=torch.float32, device=device)

    max_pos = int(positions.max().item()) + compress_ratio + 256
    half = ROPE_HEAD_DIM // 2
    freq = torch.arange(half, dtype=torch.float32) / half
    base = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
    cos_sin_cache = torch.cat([
        torch.cos(base * (10000 ** (-freq))),
        torch.sin(base * (10000 ** (-freq))),
    ], dim=1).to(device)

    return dict(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=RMS_EPS,
        cos_sin_cache=cos_sin_cache,
        k_cache=k_cache,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rope_head_dim=ROPE_HEAD_DIM,
    )


def _apply_fp8mix_case(
    xpu_in: dict,
    compress_ratio: int,
    make_non_boundary: bool,
    invalidate_slot_mapping: bool,
    invalidate_kv_slot: bool,
) -> None:
    """Mutate generated inputs to cover boundary and invalid-mapping cases."""
    num_tokens = xpu_in["positions"].numel()
    if num_tokens == 0:
        return

    if make_non_boundary and num_tokens > 1:
        odd_idx = torch.arange(1, num_tokens, 2, device=xpu_in["positions"].device)
        xpu_in["positions"][odd_idx] += 1

    if invalidate_slot_mapping and num_tokens > 1:
        odd_idx = torch.arange(1, num_tokens, 2, device=xpu_in["slot_mapping"].device)
        xpu_in["slot_mapping"][odd_idx] = -1

    if invalidate_kv_slot and num_tokens > 1:
        odd_idx = torch.arange(1, num_tokens, 2, device=xpu_in["kv_slot_mapping"].device)
        xpu_in["kv_slot_mapping"][odd_idx] = -1


def _collect_fp8mix_written_slots(
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> list[int]:
    written: list[int] = []
    for i in range(positions.numel()):
        if int(slot_mapping[i].item()) < 0:
            continue
        if (int(positions[i].item()) + 1) % compress_ratio != 0:
            continue
        kv_slot = int(kv_slot_mapping[i].item())
        if kv_slot < 0:
            continue
        written.append(kv_slot)
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for s in written:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config", LAYER_CONFIGS,
                         ids=[c[4] for c in LAYER_CONFIGS])
@pytest.mark.parametrize("num_tokens", FP8MIX_NUM_TOKENS)
@pytest.mark.parametrize("case", FP8MIX_CASES,
                         ids=[c[0] for c in FP8MIX_CASES])
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_c4_kv_insert_fp8mix_layout_and_bytes(config, num_tokens, case, device):
    compress_ratio, overlap, kv_blk, state_blk_sz, _ = config
    _, make_non_boundary, invalidate_slot_mapping, invalidate_kv_slot = case
    torch.xpu.set_device(device)
    device_obj = torch.device(device)

    xpu_in = _make_inputs_small_fp8mix(
        compress_ratio, overlap, kv_blk, state_blk_sz, num_tokens, device_obj
    )
    _apply_fp8mix_case(
        xpu_in,
        compress_ratio=compress_ratio,
        make_non_boundary=make_non_boundary,
        invalidate_slot_mapping=invalidate_slot_mapping,
        invalidate_kv_slot=invalidate_kv_slot,
    )
    cpu_in = {k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v)
              for k, v in xpu_in.items()}

    # Layout is [num_blocks, kv_blk, token_stride + scale_dim], and op writes
    # raw block bytes with token area + tail scales area.
    k_cache_u8_xpu = torch.zeros(
        xpu_in["k_cache"].shape[0],
        kv_blk,
        TOKEN_STRIDE + SCALE_DIM,
        dtype=torch.uint8,
        device=device_obj,
    )
    k_cache_u8_cpu = torch.zeros(
        cpu_in["k_cache"].shape[0],
        kv_blk,
        TOKEN_STRIDE + SCALE_DIM,
        dtype=torch.uint8,
        device="cpu",
    )

    torch.ops._xpu_C.xpu_compress_insert_fp8mix(
        xpu_in["state_cache"],
        xpu_in["token_to_req_indices"],
        xpu_in["positions"],
        xpu_in["slot_mapping"],
        xpu_in["block_table"],
        xpu_in["rms_norm_weight"],
        xpu_in["rms_norm_eps"],
        xpu_in["cos_sin_cache"],
        k_cache_u8_xpu,
        xpu_in["kv_slot_mapping"],
        kv_blk,
        compress_ratio,
        overlap,
        ROPE_HEAD_DIM,
        TOKEN_STRIDE,
        SCALE_DIM,
        int(k_cache_u8_xpu.stride(0)),
    )

    # Keep opcheck lightweight: only one representative case validates
    # registration/schema/faketensor contracts.
    opcheck(
        torch.ops._xpu_C.xpu_compress_insert_fp8mix,
        (
            xpu_in["state_cache"],
            xpu_in["token_to_req_indices"],
            xpu_in["positions"],
            xpu_in["slot_mapping"],
            xpu_in["block_table"],
            xpu_in["rms_norm_weight"],
            xpu_in["rms_norm_eps"],
            xpu_in["cos_sin_cache"],
            k_cache_u8_xpu,
            xpu_in["kv_slot_mapping"],
            kv_blk,
            compress_ratio,
            overlap,
            ROPE_HEAD_DIM,
            TOKEN_STRIDE,
            SCALE_DIM,
            int(k_cache_u8_xpu.stride(0)),
        ),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
        cond=(
            compress_ratio == LAYER_CONFIGS[0][0]
            and overlap == LAYER_CONFIGS[0][1]
            and num_tokens == FP8MIX_NUM_TOKENS[0]
            and case[0] == FP8MIX_CASES[0][0]
            and device == XPU_DEVICES[0]
        ),
    )

    _ref_c4_kv_insert_fp8mix(
        state_cache=cpu_in["state_cache"],
        token_to_req_indices=cpu_in["token_to_req_indices"],
        positions=cpu_in["positions"],
        slot_mapping=cpu_in["slot_mapping"],
        block_table=cpu_in["block_table"],
        rms_norm_weight=cpu_in["rms_norm_weight"],
        rms_norm_eps=cpu_in["rms_norm_eps"],
        cos_sin_cache=cpu_in["cos_sin_cache"],
        k_cache_u8=k_cache_u8_cpu,
        kv_slot_mapping=cpu_in["kv_slot_mapping"],
        kv_cache_block_size=kv_blk,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rope_head_dim=ROPE_HEAD_DIM,
        token_stride=TOKEN_STRIDE,
        scale_dim=SCALE_DIM,
    )

    xpu_raw = k_cache_u8_xpu.cpu().reshape(-1)
    cpu_raw = k_cache_u8_cpu.reshape(-1)
    written_slots = _collect_fp8mix_written_slots(
        positions=xpu_in["positions"].cpu(),
        slot_mapping=xpu_in["slot_mapping"].cpu(),
        kv_slot_mapping=xpu_in["kv_slot_mapping"].cpu(),
        compress_ratio=compress_ratio,
    )

    # Bytes for slots that should be written must match reference.
    for kv_slot in written_slots:
        kv_block = kv_slot // kv_blk
        kv_pos = kv_slot % kv_blk
        blk_stride = int(k_cache_u8_xpu.stride(0))
        base = kv_block * blk_stride
        token_off = base + kv_pos * TOKEN_STRIDE
        scale_off = base + kv_blk * TOKEN_STRIDE + kv_pos * SCALE_DIM

        # 448B fp8 NOPE payload
        assert torch.equal(
            xpu_raw[token_off: token_off + NOPE_HEAD_DIM],
            cpu_raw[token_off: token_off + NOPE_HEAD_DIM],
        )
        # 64-dim rope bf16 payload = 128B.
        # Allow tiny numerical drift between SYCL online-softmax and Python
        # reference accumulation order while still checking bf16-level values.
        xpu_rope_bf16 = (
            xpu_raw[token_off + NOPE_HEAD_DIM: token_off + TOKEN_STRIDE]
            .view(torch.bfloat16)
            .float()
        )
        cpu_rope_bf16 = (
            cpu_raw[token_off + NOPE_HEAD_DIM: token_off + TOKEN_STRIDE]
            .view(torch.bfloat16)
            .float()
        )
        torch.testing.assert_close(xpu_rope_bf16, cpu_rope_bf16, rtol=0, atol=1e-2)
        # 8B scales (7 real + 1 pad)
        assert torch.equal(
            xpu_raw[scale_off: scale_off + SCALE_DIM],
            cpu_raw[scale_off: scale_off + SCALE_DIM],
        )

    # Non-written bytes should stay untouched (zero-initialized).
    if len(written_slots) < num_tokens:
        untouched = torch.ones_like(xpu_raw, dtype=torch.bool)
        blk_stride = int(k_cache_u8_xpu.stride(0))
        for kv_slot in written_slots:
            kv_block = kv_slot // kv_blk
            kv_pos = kv_slot % kv_blk
            base = kv_block * blk_stride
            token_off = base + kv_pos * TOKEN_STRIDE
            scale_off = base + kv_blk * TOKEN_STRIDE + kv_pos * SCALE_DIM
            untouched[token_off: token_off + TOKEN_STRIDE] = False
            untouched[scale_off: scale_off + SCALE_DIM] = False

        assert torch.equal(xpu_raw[untouched], cpu_raw[untouched])


@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_c4_kv_insert_fp8mix_invalid_stride_raises(device):
    config = LAYER_CONFIGS[0]
    compress_ratio, overlap, kv_blk, state_blk_sz, _ = config
    torch.xpu.set_device(device)
    device_obj = torch.device(device)
    xpu_in = _make_inputs_small_fp8mix(
        compress_ratio,
        overlap,
        kv_blk,
        state_blk_sz,
        num_tokens=1,
        device=device_obj,
    )

    k_cache_u8_xpu = torch.zeros(
        xpu_in["k_cache"].shape[0],
        kv_blk,
        TOKEN_STRIDE + SCALE_DIM,
        dtype=torch.uint8,
        device=device_obj,
    )
    # Deliberately violate kv_block_stride lower bound.
    bad_stride = kv_blk * (TOKEN_STRIDE + SCALE_DIM) - 1

    with pytest.raises(RuntimeError, match="kv_block_stride too small"):
        torch.ops._xpu_C.xpu_compress_insert_fp8mix(
            xpu_in["state_cache"],
            xpu_in["token_to_req_indices"],
            xpu_in["positions"],
            xpu_in["slot_mapping"],
            xpu_in["block_table"],
            xpu_in["rms_norm_weight"],
            xpu_in["rms_norm_eps"],
            xpu_in["cos_sin_cache"],
            k_cache_u8_xpu,
            xpu_in["kv_slot_mapping"],
            kv_blk,
            compress_ratio,
            overlap,
            ROPE_HEAD_DIM,
            TOKEN_STRIDE,
            SCALE_DIM,
            bad_stride,
        )


@pytest.mark.parametrize(
    "arg_name,bad_value,err_match",
    [
        ("rope_head_dim", 31, r"rope_head_dim must be even and in \[0, 512\]"),
        ("token_stride", TOKEN_STRIDE - 1, r"token_stride too small for fp8\+rope payload"),
        ("scale_dim", 6, "scale_dim too small for NOPE quant blocks"),
    ],
)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_c4_kv_insert_fp8mix_invalid_contract_args_raise(
    arg_name, bad_value, err_match, device
):
    config = LAYER_CONFIGS[0]
    compress_ratio, overlap, kv_blk, state_blk_sz, _ = config
    torch.xpu.set_device(device)
    device_obj = torch.device(device)
    xpu_in = _make_inputs_small_fp8mix(
        compress_ratio,
        overlap,
        kv_blk,
        state_blk_sz,
        num_tokens=1,
        device=device_obj,
    )

    k_cache_u8_xpu = torch.zeros(
        xpu_in["k_cache"].shape[0],
        kv_blk,
        TOKEN_STRIDE + SCALE_DIM,
        dtype=torch.uint8,
        device=device_obj,
    )

    rope_head_dim = ROPE_HEAD_DIM
    token_stride = TOKEN_STRIDE
    scale_dim = SCALE_DIM
    if arg_name == "rope_head_dim":
        rope_head_dim = bad_value
    elif arg_name == "token_stride":
        token_stride = bad_value
    elif arg_name == "scale_dim":
        scale_dim = bad_value

    with pytest.raises(RuntimeError, match=err_match):
        torch.ops._xpu_C.xpu_compress_insert_fp8mix(
            xpu_in["state_cache"],
            xpu_in["token_to_req_indices"],
            xpu_in["positions"],
            xpu_in["slot_mapping"],
            xpu_in["block_table"],
            xpu_in["rms_norm_weight"],
            xpu_in["rms_norm_eps"],
            xpu_in["cos_sin_cache"],
            k_cache_u8_xpu,
            xpu_in["kv_slot_mapping"],
            kv_blk,
            compress_ratio,
            overlap,
            rope_head_dim,
            token_stride,
            scale_dim,
            int(k_cache_u8_xpu.stride(0)),
        )
