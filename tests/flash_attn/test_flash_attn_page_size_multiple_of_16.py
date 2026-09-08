# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged KV page-size routing tests for the Xe2 FMHA kernels.

The kernels' real constraint on a paged KV page size is that the page size
divide the KV tile width exactly: the mainloop walks ``page_size /
TileShapeQK[1]`` sub-tiles per page through the page-table indirection, so the
division must be exact. Two tile widths exist -- the standard
``chunk_policy_head*`` policies use TileShapeQK[1] = 32, the
``chunk_policy_head*_b16`` policies use 16 -- but four sites enumerated the
accepted sizes as "16, 32, or any positive multiple of 64":

  * ``csrc/xpu/attn/xe_2/fmha_xe2.cpp``           chunk-prefill validation
    plus the ``use_b16_policy`` tile selector
  * ``csrc/xpu/attn/xe_2/paged_decode_utils.hpp`` ``dispatch_by_page_size``
  * ``csrc/flash_attn/flash_api.cpp``             ``get_num_splits``
  * ``vllm_xpu_kernels/flash_attn_interface.py``  ``_kv_tile_from_block_size``,
    the Python mirror of ``get_num_splits`` used to build the decode split plan

All four now accept any positive multiple of 16, keeping every size the old
rule accepted on the tile it already had. Two classes are newly admitted:
multiples of 16 that are not multiples of 32 (48, 80, 240, 880, 1360, ...), and
odd multiples of 32 (96, 160, 224, 1440, ...). Which tile the second class
gets is a throughput choice -- both widths divide it exactly -- so the tests
here assert correctness and leave that choice open.

Three failure modes are being guarded here:

  1. relaxing the validation without moving the ``use_b16_policy`` selector
     would dispatch 880 to the 32-wide standard policy, where 880 / 32 = 27.5
     is not an integer: silently wrong attention output rather than an error;
  2. relaxing too far would let a page size that is not a multiple of 16
     through to a kernel that cannot represent it;
  3. leaving the Python mirror behind would size the decode split plan for a
     wider tile than the kernel runs, silently skipping most of the KV history.

Page size 880 = 16 x 55 appears throughout because it is a real workload's
value: AR video diffusion stores one video frame per KV page, so the page size
is the frame sequence length. It is a multiple of 16 but not of 32, which makes
it the case that distinguishes a correct tile selector from one that was only
relaxed at the validation.

Run with:
    source /opt/intel/oneapi/setvars.sh
    XPU_KERNEL_TEST_SCOPE=full \\
        pytest tests/flash_attn/test_flash_attn_page_size_multiple_of_16.py -v
"""

import contextlib
from typing import NamedTuple, Optional

import pytest
import torch

# The dense oracle is imported from the existing varlen module rather than
# duplicated here. This PR's tests live in their own file so that a small
# kernel change stays reviewable next to its tests; the alternative is to
# inline these cases into tests/flash_attn/test_flash_attn_varlen_func.py
# beside the other paged tests (~98 KB), which is equally valid but buries the
# additions in a large diff. Only `ref_paged_attn` is reused, and it is called
# with its upstream keyword names throughout (including the `casual` spelling).
from tests.flash_attn.test_flash_attn_varlen_func import ref_paged_attn
from vllm_xpu_kernels import flash_attn_interface as _fa_interface
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

# DreamZero (AR video diffusion) stores one video frame per KV page, so
# block_size == frame seqlen == 880 = 16 x 55. Per rank at TP=4: 10 heads,
# head_dim 128, bf16, non-causal, batch 1, always prefill-shaped.
DREAMZERO_PAGE = 880
DREAMZERO_HEADS = 10
DREAMZERO_HEAD_DIM = 128
# The rolling history reaches 21 frames, i.e. 21 x 880 = 18,480 KV tokens.
DREAMZERO_MAX_HISTORY_FRAMES = 21
DREAMZERO_MAX_KV = DREAMZERO_MAX_HISTORY_FRAMES * DREAMZERO_PAGE

# Page sizes the pre-relaxation rule already accepted -- these must keep
# working and keep their tile (no-regression gate).
OLD_RULE_PAGE_SIZES = [16, 32, 64, 128, 512, 2048]
# Multiples of 16 that are NOT multiples of 32: rejected before the
# relaxation. Only the 16-wide tile divides these exactly.
NEWLY_SUPPORTED_B16_PAGE_SIZES = [48, 80, 240, 880, 1360]
# Odd multiples of 32: also rejected before the relaxation. BOTH tile widths
# divide these exactly, so either is correct and the choice is a perf one.
NEWLY_SUPPORTED_STD_PAGE_SIZES = [96, 160, 224, 1440]
# Not multiples of 16: rejected before and after.
REJECTED_PAGE_SIZES = [1, 15, 17, 24, 33, 100]

_DEV = "xpu"

# Written into every KV element the kernel must not read (padded block-table
# entries, and the unused tail of a partial last page). An over-read puts a
# huge logit into the softmax, so a boundary bug shows up as a hard mismatch
# against the oracle instead of a few LSBs of noise.
_POISON = 1.0e4

# bf16 tolerance. atol matches the upstream paged cases; rtol is relaxed from
# upstream's 1e-2 to 2e-2 because these cases run head_dim 128 with KV lengths
# up to 18,480, an order of magnitude more accumulation than the upstream paged
# tests (kv <= 2048), and bf16 has 8 mantissa bits. This is a deviation from
# upstream's convention, not parity with it.
_ATOL = 2e-2
_RTOL = 2e-2


class KernelNotCompiled(RuntimeError):
    """The installed wheel has no kernel for the config under test.

    Raised in place of ``flash_attn_varlen_func``'s silent PyTorch fallback.
    """


def _xpu_device_count() -> int:
    try:
        return torch.xpu.device_count()
    except Exception:  # torch built without XPU support
        return 0


# Applied to each device test individually instead of as a module-level
# `pytestmark`, because the two contract tests make no kernel call and must
# still run on a machine without an XPU -- a module-level skipif cannot be
# opted out of per test.
requires_xpu = pytest.mark.skipif(
    _xpu_device_count() == 0,
    reason="no XPU device visible; run "
    "`source /opt/intel/oneapi/setvars.sh` first, otherwise libumf.so.1 is "
    "missing, SYCL reports 0 platforms and torch.xpu.device_count() == 0")

# `flash_attn_varlen_func` catches any RuntimeError containing "not compiled"
# and silently substitutes `_fallback_varlen_attn`, a PyTorch reference. Every
# oracle comparison in this module would then pass without the SYCL kernel ever
# running -- reference against reference -- which is exactly what a wheel built
# before the head128 chunk-prefill configs landed does. The notice it emits is
# deduplicated per config by `_warned_missing_configs`, so it is silent after
# the first occurrence and cannot be relied on as a per-test signal.
#
# Fail closed: require the fallback entry point to exist at import time, so a
# rename breaks loudly here rather than quietly disarming every assertion
# below, and replace it per call (see `_no_silent_fallback`).
_FALLBACK_ATTR = "_fallback_varlen_attn"
assert hasattr(_fa_interface, _FALLBACK_ATTR), (
    f"vllm_xpu_kernels.flash_attn_interface has no {_FALLBACK_ATTR}; this "
    "module detects the silent PyTorch fallback by intercepting it, and "
    "without that interception every oracle comparison here is vacuous. "
    "Upstream has renamed it -- update _FALLBACK_ATTR rather than deleting "
    "this assertion.")


@contextlib.contextmanager
def _no_silent_fallback():
    """Turn the silent PyTorch fallback into ``KernelNotCompiled``.

    Positive detection, per call: it does not depend on the deduplicated
    ``_warned_missing_configs`` notice, so the tenth test is protected exactly
    as well as the first.
    """
    original = getattr(_fa_interface, _FALLBACK_ATTR)

    def _refuse(*args, **kwargs):
        raise KernelNotCompiled(
            "the installed wheel has no SYCL kernel for this config and "
            "flash_attn_varlen_func fell back to the PyTorch reference; "
            "rebuild vllm-xpu-kernels rather than trusting this comparison")

    setattr(_fa_interface, _FALLBACK_ATTR, _refuse)
    try:
        yield
    finally:
        setattr(_fa_interface, _FALLBACK_ATTR, original)


class PagedCase(NamedTuple):
    """Everything both the kernel and the oracle need for one paged call."""

    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    cu_query_lens: torch.Tensor
    seq_k: torch.Tensor
    query_lens: list[int]
    kv_lens: list[int]
    max_query_len: int
    max_kv_len: int
    block_size: int
    scale: float
    dtype: torch.dtype


def _build_paged_case(
    block_size: int,
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int] = (DREAMZERO_HEADS, DREAMZERO_HEADS),
    head_size: int = DREAMZERO_HEAD_DIM,
    dtype: torch.dtype = torch.bfloat16,
    padded_stride: bool = False,
    num_layers: int = 2,
    seed: int = 4242,
) -> PagedCase:
    """Build one paged-KV test case with poisoned out-of-range KV.

    * physical blocks are handed out in reverse order, so the block table is
      never the identity mapping;
    * one dedicated physical block is filled entirely with ``_POISON`` and
      every padded (unused) block-table entry points at it;
    * the unused tail of a partial last page is poisoned as well;
    * ``padded_stride=True`` builds the K/V pools as views
      ``combined[:, 0, 0, :, :, :]`` / ``combined[:, 0, 1, :, :, :]`` of a
      ``(num_blocks, num_layers, 2, block_size, heads, head_dim)`` tensor, so
      the physical page stride is larger than the logical page -- the uniform
      cross-layer KV layout used by the offloading connector.

    Every tensor is created with an explicit ``device=``: an earlier revision
    called ``torch.set_default_device("xpu")`` here and never restored it,
    which leaked into the rest of the pytest session.
    """
    torch.xpu.set_device(f"{_DEV}:0")
    torch.manual_seed(seed)

    query_lens = [q for q, _ in seq_lens]
    kv_lens = [k for _, k in seq_lens]
    num_seqs = len(seq_lens)
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device=_DEV)

    # Physical-block budget: pages actually used plus one shared poison page.
    pages_per_seq = [(k + block_size - 1) // block_size for k in kv_lens]
    total_used_pages = sum(pages_per_seq)
    poison_block = total_used_pages
    num_blocks = total_used_pages + 1

    if padded_stride:
        combined = torch.randn(num_blocks,
                               num_layers,
                               2,
                               block_size,
                               num_kv_heads,
                               head_size,
                               dtype=dtype,
                               device=_DEV)
        key_cache = combined[:, 0, 0, :, :, :]
        value_cache = combined[:, 0, 1, :, :, :]
        assert not key_cache.is_contiguous()
        assert key_cache.stride(0) == num_layers * 2 * block_size * \
            num_kv_heads * head_size
    else:
        key_cache = torch.randn(num_blocks,
                                block_size,
                                num_kv_heads,
                                head_size,
                                dtype=dtype,
                                device=_DEV)
        value_cache = torch.randn_like(key_cache)

    key_cache[poison_block].fill_(_POISON)
    value_cache[poison_block].fill_(_POISON)

    max_pages = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.full((num_seqs, max_pages),
                              poison_block,
                              dtype=torch.int32,
                              device=_DEV)
    # Reversed assignment: seq 0 gets the highest physical block indices.
    scatter = list(range(total_used_pages))[::-1]
    cursor = 0
    for i, npages in enumerate(pages_per_seq):
        phys = scatter[cursor:cursor + npages]
        block_tables[i, :npages] = torch.tensor(phys,
                                                dtype=torch.int32,
                                                device=_DEV)
        last_valid = kv_lens[i] - (npages - 1) * block_size
        if last_valid < block_size:
            key_cache[phys[-1], last_valid:].fill_(_POISON)
            value_cache[phys[-1], last_valid:].fill_(_POISON)
        cursor += npages

    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32,
                                 device=_DEV).cumsum(dim=0,
                                                     dtype=torch.int32)
    seq_k = torch.tensor(kv_lens, dtype=torch.int32, device=_DEV)

    return PagedCase(query=query,
                     key_cache=key_cache,
                     value_cache=value_cache,
                     block_tables=block_tables,
                     cu_query_lens=cu_query_lens,
                     seq_k=seq_k,
                     query_lens=query_lens,
                     kv_lens=kv_lens,
                     max_query_len=max_query_len,
                     max_kv_len=max_kv_len,
                     block_size=block_size,
                     scale=scale,
                     dtype=dtype)


def _run_kernel(case: PagedCase, is_mix_batch: bool = False) -> torch.Tensor:
    """Call the kernel, refusing the silent PyTorch fallback.

    Raises ``KernelNotCompiled`` if the wheel lacks a kernel for this config.
    """
    with _no_silent_fallback():
        output = flash_attn_varlen_func(case.query,
                                        case.key_cache,
                                        case.value_cache,
                                        case.max_query_len,
                                        case.cu_query_lens,
                                        case.max_kv_len,
                                        seqused_k=case.seq_k,
                                        softmax_scale=case.scale,
                                        causal=False,
                                        block_table=case.block_tables,
                                        window_size=(-1, -1),
                                        is_mix_batch=is_mix_batch,
                                        s_aux=None)
    # The entry point returns either a tensor or an (out, lse) pair, depending
    # on whether softmax_lse was requested; take the output either way.
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def _reference(case: PagedCase) -> torch.Tensor:
    # window_size_left/right = -1 is ref_paged_attn's "disabled" sentinel; its
    # signature defaults them to None but the body compares them numerically,
    # so passing None raises TypeError.
    return ref_paged_attn(query=case.query.contiguous(),
                          key_cache=case.key_cache.contiguous(),
                          value_cache=case.value_cache.contiguous(),
                          query_lens=case.query_lens,
                          kv_lens=case.kv_lens,
                          block_tables=case.block_tables,
                          scale=case.scale,
                          window_size_left=-1,
                          window_size_right=-1,
                          soft_cap=None,
                          is_paged=True,
                          casual=False,
                          sink=None,
                          q_descale=None,
                          k_descale=None,
                          v_descale=None,
                          is_fp8kv=False,
                          is_fp8_query=False,
                          dtype=case.dtype)


def _assert_matches_reference(case: PagedCase,
                              is_mix_batch: bool = False) -> None:
    output = _run_kernel(case, is_mix_batch=is_mix_batch)
    ref_output = _reference(case)
    torch.testing.assert_close(
        output,
        ref_output,
        atol=_ATOL,
        rtol=_RTOL,
        msg=lambda m: f"block_size={case.block_size} "
        f"is_mix_batch={is_mix_batch}\n{m}")


def _sweep_seq_lens(block_size: int) -> list[tuple[int, int]]:
    """Two prefill sequences, scaled to the page size under test.

    seq 0 is exactly one full page; seq 1 spans two full pages plus a half
    page. So seq 0's block-table row has a poisoned padded tail and seq 1 has
    a poisoned partial last page, for every page size in the sweep.
    """
    return [(block_size, block_size),
            (min(block_size, 128), 2 * block_size + block_size // 2)]


def _dreamzero_seq_lens(action_tail: int) -> list[tuple[int, int]]:
    """A DreamZero batch shape, one video frame per page.

    action_tail=0  -> page-aligned:  q in {880, 1760}, kv in {880, 1760,
                      3520, 7920} (up to 9 frames of rolling history).
    action_tail=25 -> the last sequence carries a 25-token action tail, so it
                      is not page-aligned: q=1785, kv=7945.

    Covers query lengths {880, 1760, 1785} and KV lengths {880, 1760, 3520,
    7920, 7945}. It does NOT reach the deepest production history of 21 frames
    / 18,480 tokens -- that is a separate test
    (``test_dreamzero_full_history_880_page``) so that its much larger oracle
    allocation can be skipped independently.
    """
    base = [(880, 880), (1760, 1760), (1760, 3520)]
    if action_tail == 0:
        return base + [(1760, 7920)]
    return base + [(1760 + action_tail, 7920 + action_tail)]


# ---------------------------------------------------------------------------
# The routing contract, expressed as invariants rather than as one rule.
# ---------------------------------------------------------------------------
# Two page-size rules have shipped -- the pre-relaxation enumeration ("16, 32,
# or any positive multiple of 64") and the current multiple-of-16 one -- and
# the tile assigned to some classes is still under discussion. So the contract
# tests below assert the properties EVERY correct routing must have, rather
# than pinning the tile a given page size happens to receive. That keeps this
# file from needing an edit in lockstep with the selector.
CHUNK_PREFILL_TILES = (16, 32)  # chunk_policy_head*_b16 / chunk_policy_head*
DECODE_TILES = (16, 32, 64)     # dispatch_by_page_size tile templates


def _tile_before_relaxation(page_size: int) -> Optional[int]:
    """The enumerated rule that predates the multiple-of-16 relaxation.

    Returns None for anything it rejected. Used as the reference for "no
    previously-working page size changed tile", which is the property that
    makes the relaxation safe rather than merely passing.
    """
    if page_size == 16:
        return 16
    if page_size == 32:
        return 32
    if page_size > 0 and (page_size % 64) == 0:
        return 64
    return None


def _dividing_tiles(page_size: int, tiles) -> list[int]:
    """Every tile width in ``tiles`` that divides ``page_size`` exactly.

    The mainloop walks ``page_size / TileShapeQK[1]`` sub-tiles per page, so an
    inexact division is silently wrong output rather than an error. A tile is a
    legal choice for a page size if and only if it appears here.
    """
    if page_size <= 0:
        return []
    return [t for t in tiles if page_size % t == 0]


def _chunk_prefill_tile_half_fixed(block_size: int) -> Optional[int]:
    """The trap: validation relaxed but the OLD ``block_size == 16`` selector
    kept. Accepts 880 and then hands it to the 32-wide policy, whose sub-tile
    division does not come out even."""
    if not (block_size > 0 and (block_size % 16) == 0):
        return None
    return 16 if block_size == 16 else 32


@pytest.mark.parametrize("block_size", OLD_RULE_PAGE_SIZES)
@requires_xpu
@torch.inference_mode()
def test_previously_supported_page_sizes_unchanged(block_size: int) -> None:
    """No-regression gate: every page size the old rule accepted still works.

    Passes identically on a stock and a patched build -- the new divisibility
    routing selects the same tile for all of these sizes.
    """
    case = _build_paged_case(block_size,
                             _sweep_seq_lens(block_size),
                             num_heads=(8, 2))
    _assert_matches_reference(case)
    torch.xpu.empty_cache()


@pytest.mark.parametrize("block_size", NEWLY_SUPPORTED_B16_PAGE_SIZES)
@requires_xpu
@torch.inference_mode()
def test_newly_supported_b16_page_sizes(block_size: int) -> None:
    """Multiples of 16 that are not multiples of 32 -> the 16-wide tile.

    Each of these raised "chunk_prefill: unsupported block_size=<n>" before
    the change; each must now route to the 16-wide *_b16 policy and match the
    dense oracle. A mismatch here (rather than an exception) is the signature
    of the validation having been relaxed without moving the tile selector.
    """
    case = _build_paged_case(block_size,
                             _sweep_seq_lens(block_size),
                             num_heads=(8, 2))
    _assert_matches_reference(case)
    torch.xpu.empty_cache()


@pytest.mark.parametrize("block_size", NEWLY_SUPPORTED_STD_PAGE_SIZES)
@requires_xpu
@torch.inference_mode()
def test_newly_supported_std_page_sizes(block_size: int) -> None:
    """Odd multiples of 32 -- the second newly-admitted class.

    The old rule rejected these even though BOTH tile widths divide them
    exactly (96 / 32 = 3 and 96 / 16 = 6). Which one the selector picks is a
    throughput question, not a correctness one, so this test only asserts the
    output matches the oracle -- it passes whichever tile is chosen. It is here
    because this class takes a different route from the b16 class above and so
    needs its own coverage.
    """
    case = _build_paged_case(block_size,
                             _sweep_seq_lens(block_size),
                             num_heads=(8, 2))
    _assert_matches_reference(case)
    torch.xpu.empty_cache()


@pytest.mark.parametrize("padded_stride", [False, True])
@pytest.mark.parametrize("action_tail", [0, 25])
@requires_xpu
@torch.inference_mode()
def test_dreamzero_880_page(action_tail: int, padded_stride: bool) -> None:
    """The production shape: page 880, 10 heads, head_dim 128, bf16.

    Non-causal and all prefill-shaped (a full frame of queries per denoise
    step, never a single decode token). With is_mix_batch left at its default
    and max_seqlen_q > 1, ``is_prefill_only = (!mix_batch && max_seqlen_q > 1)
    | !is_paged`` is true, so this is the pure chunk-prefill route -- and
    vllm-omni #6962 does not pass is_mix_batch, so that default is what
    production gets. tiles_per_page = 880 / 16 = 55, i.e. the mainloop walks 55
    sub-tiles per physical page; on the b16 path that had only ever been
    exercised at tiles_per_page == 1 (page_size == 16).

    KV depth here is up to 9 frames; the 21-frame maximum is
    ``test_dreamzero_full_history_880_page``.
    """
    case = _build_paged_case(DREAMZERO_PAGE,
                             _dreamzero_seq_lens(action_tail),
                             num_heads=(DREAMZERO_HEADS, DREAMZERO_HEADS),
                             head_size=DREAMZERO_HEAD_DIM,
                             padded_stride=padded_stride)
    _assert_matches_reference(case)
    torch.xpu.empty_cache()


@pytest.mark.parametrize("action_tail", [0, 25])
@requires_xpu
@torch.inference_mode()
def test_dreamzero_full_history_880_page(action_tail: int) -> None:
    """The deepest production shape: one frame of query, 21 frames of history.

    18,480 KV tokens is 1,155 sub-tiles per sequence at the 16-wide tile and 21
    page-table entries -- the deepest page-table walk and the largest
    max_seqlen_k the workload ever produces, so it is where an index-width or
    accumulator problem on the newly-exercised b16 path would surface. Kept
    separate from test_dreamzero_880_page because the oracle materialises a
    dense (query_len, heads, kv_len) score tensor, which at 880 x 10 x 18,480
    is the largest allocation in this module.
    """
    seq_lens = [(DREAMZERO_PAGE, DREAMZERO_MAX_KV + action_tail)]
    case = _build_paged_case(DREAMZERO_PAGE,
                             seq_lens,
                             num_heads=(DREAMZERO_HEADS, DREAMZERO_HEADS),
                             head_size=DREAMZERO_HEAD_DIM)
    _assert_matches_reference(case)
    torch.xpu.empty_cache()


@pytest.mark.parametrize("decode_only", [False, True])
@requires_xpu
@torch.inference_mode()
def test_mix_batch_880_page(decode_only: bool) -> None:
    """Page 880 through the paged-decode kernel as well as chunk prefill.

    With is_mix_batch=True, flash_api.cpp runs the chunk-prefill kernel (masked
    to the prefill rows) and then the paged-decode kernel over the batch. The
    decode kernel has its own page-size check, which raised "Unsupported page
    size for fmha: 880" before ``dispatch_by_page_size`` was changed -- so this
    test is coverage a chunk-prefill-only fix could not have had.

    decode_only=True makes every sequence a single query token, which takes the
    max_seqlen_q == 1 branch and is therefore also the only case here that
    exercises get_num_splits() and the Python split-plan mirror at page 880.

    Skips rather than fails when the wheel has no non-causal head-128 page-16
    decode tuple: ``paged_decode_default.conf`` ships
    ``8,128,64,false,false,false`` but not the page-16 equivalent, so the route
    this test newly reaches has no compiled kernel until those rows are added
    (``apply/patch_b1_paged_decode_conf.py``) and the wheel rebuilt. That is a
    build-coverage gap, not a kernel defect, and it must not read as either a
    pass or a correctness failure.
    """
    if decode_only:
        seq_lens = [(1, 880), (1, 1760), (1, 3520), (1, 7920)]
    else:
        seq_lens = [(880, 880), (1, 3520), (1760, 1760), (1, 7920)]
    case = _build_paged_case(DREAMZERO_PAGE,
                             seq_lens,
                             num_heads=(DREAMZERO_HEADS, DREAMZERO_HEADS),
                             head_size=DREAMZERO_HEAD_DIM)
    try:
        _assert_matches_reference(case, is_mix_batch=True)
    except KernelNotCompiled as exc:
        pytest.skip(
            "no compiled paged-decode kernel for qgroup/head128/page16 "
            "non-causal; add the rows from "
            "apply/patch_b1_paged_decode_conf.py to "
            "csrc/xpu/attn/kernel_configs/paged_decode_default.conf and "
            f"rebuild the wheel. ({exc})")
    torch.xpu.empty_cache()


@pytest.mark.parametrize("block_size", REJECTED_PAGE_SIZES)
@requires_xpu
@torch.inference_mode()
def test_page_size_not_multiple_of_16_still_rejected(block_size: int) -> None:
    """The relaxation must stop at multiples of 16.

    A page size that does not divide the 16-wide tile exactly cannot be walked
    by the mainloop, so it has to raise rather than produce numbers. The
    assertion deliberately matches only "block_size" and the offending value,
    not the list of supported sizes, so this test passes unchanged on a stock
    build (whose message still reads "16, 32, or any positive multiple of 64").
    """
    kv_len = 4 * block_size
    case = _build_paged_case(block_size, [(min(16, kv_len), kv_len)],
                             num_heads=(8, 2))
    with pytest.raises(RuntimeError) as excinfo:
        _run_kernel(case)
    message = str(excinfo.value)
    assert not isinstance(excinfo.value, KernelNotCompiled), (
        "expected a page-size rejection, got the missing-kernel fallback: "
        f"{message}")
    assert str(block_size) in message, message
    assert "block_size" in message, message
    torch.xpu.empty_cache()


def test_tile_selection_is_exact_division() -> None:
    """The routing contract, checkable without a device.

    Makes no kernel call. Asserts the properties any correct page-size routing
    must have, rather than the tile a particular rule assigns:

      1. every admitted page size has at least one tile width that divides it
         exactly, at both dispatch sites. An inexact division is silently wrong
         output rather than an error, so this is the premise the whole
         relaxation rests on;
      2. nothing ever needs a tile wider than 64. The 8-subgroup K-split _128
         decode policy is disabled upstream over a faulty cross-subgroup SLM
         reduction, so it has to stay unreachable;
      3. the tile the pre-relaxation rule assigned still divides exactly, so
         leaving previously-working sizes where they were is always legal;
      4. relaxing the validation while keeping the old ``block_size == 16``
         selector picks a tile that does NOT divide -- the one mistake that
         produces wrong numbers instead of an exception.

    The device tests above are what exercise the kernel; this pins the contract
    they are checking.
    """
    swept = sorted(
        set(range(16, 2049, 16)) | set(OLD_RULE_PAGE_SIZES)
        | set(NEWLY_SUPPORTED_B16_PAGE_SIZES)
        | set(NEWLY_SUPPORTED_STD_PAGE_SIZES))

    for page_size in swept:
        prefill = _dividing_tiles(page_size, CHUNK_PREFILL_TILES)
        decode = _dividing_tiles(page_size, DECODE_TILES)
        # (1) at least one legal tile at each site.
        assert prefill, f"page {page_size}: no chunk-prefill tile divides it"
        assert decode, f"page {page_size}: no decode tile divides it"
        # (2) 16 divides every multiple of 16, so a wider-than-64 tile is never
        # required to make a page size representable.
        assert 16 in prefill and 16 in decode, page_size
        # (3) the pre-relaxation assignment remains a legal choice.
        old = _tile_before_relaxation(page_size)
        if old is not None:
            assert page_size % old == 0, (page_size, old)
            assert old in decode, (page_size, old)

    # Anything that is not a positive multiple of 16 has no legal tile at all,
    # which is why the validation stops there.
    for page_size in REJECTED_PAGE_SIZES + [0, -16, 2047]:
        assert not _dividing_tiles(page_size, CHUNK_PREFILL_TILES), page_size
        assert not _dividing_tiles(page_size, DECODE_TILES), page_size
        assert _tile_before_relaxation(page_size) is None, page_size

    # (4) the trap. For the b16 class the half-fixed selector picks 32, which
    # does not divide: 880 / 32 = 27.5. Only this class witnesses the bug --
    # the odd multiples of 32 divide 32 exactly, so the half-fixed selector is
    # accidentally correct for them, which is why 880 and not 96 exposes it.
    for page_size in NEWLY_SUPPORTED_B16_PAGE_SIZES:
        trap = _chunk_prefill_tile_half_fixed(page_size)
        assert trap == 32, page_size
        assert page_size % trap != 0, (
            f"page {page_size} divides the 32-wide tile exactly, so it is not "
            "a witness for the selector bug")
        assert 16 in _dividing_tiles(page_size, CHUNK_PREFILL_TILES), page_size
    for page_size in NEWLY_SUPPORTED_STD_PAGE_SIZES:
        assert _chunk_prefill_tile_half_fixed(page_size) == 32, page_size
        assert page_size % 32 == 0, page_size


def test_python_split_plan_mirror_is_consistent() -> None:
    """The host split-plan mirror must agree with the tile the kernel runs.

    ``_kv_tile_from_block_size`` feeds ``build_decode_split_plan``, whose
    docstring requires kv_tile to "equal the kernel's get<1>(TileShapeQK{})".
    If the host computes the tile count with a wider tile than the kernel runs,
    the emitted work list covers only part of the tile index space and the tail
    of the KV history is never attended -- silently, with no error raised.

    Asserted as invariants rather than against a specific rule, so it holds
    whichever routing is in force: the mirror's tile must divide the block size
    exactly, must be one the dispatch can select, and must not move a page size
    the pre-relaxation rule already handled. Host-only, no device needed.
    """
    mirror = getattr(_fa_interface, "_kv_tile_from_block_size", None)
    if mirror is None:
        pytest.skip("flash_attn_interface has no _kv_tile_from_block_size; "
                    "upstream has renamed or removed the split-plan mirror")

    for page_size in sorted(
            set(range(16, 2049, 16)) | set(OLD_RULE_PAGE_SIZES)
            | set(NEWLY_SUPPORTED_B16_PAGE_SIZES)
            | set(NEWLY_SUPPORTED_STD_PAGE_SIZES)):
        tile = mirror(page_size)
        assert tile in DECODE_TILES, (page_size, tile)
        assert page_size % tile == 0, (
            f"mirror returns kv_tile={tile} for block_size={page_size}, which "
            f"it does not divide: the split plan would cover "
            f"{page_size // tile} whole tiles where the kernel walks "
            f"{page_size / tile}")
        old = _tile_before_relaxation(page_size)
        if old is not None:
            assert tile == old, (
                f"block_size={page_size} was handled before the relaxation "
                f"with kv_tile={old}; the mirror now returns {tile}")
