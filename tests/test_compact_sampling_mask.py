# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from tests.ops.compact_sampling_mask_op import (compact_sampling_mask_torch,
                                                compact_sampling_mask_xpu,
                                                unpack_support)
from tests.utils import seed_everything

DEVICE = "xpu"

BATCH_SIZE = [1, 4, 32]
VOCAB_SIZE = [128, 1024, 4096]
MAX_NUM_KEPT = [8, 64, 512]
INACTIVE_FRACTION = [0.0, 0.5, 1.0]
FINITE_FRACTION = [0.0, 0.2, 0.5, 1.0]

# CI/mini scope parameter overrides
MINI_PYTEST_PARAMS = {
    "default": {
        "batch_size": [1, 4],
        "vocab_size": [128, 1024],
        "max_num_kept": [8, 64],
        "inactive_fraction": [0.0, 0.5],
        "finite_fraction": [0.2, 1.0],
    },
}


def _make_logits(batch_size, vocab_size, finite_fraction, device):
    logits = torch.randn(batch_size,
                         vocab_size,
                         dtype=torch.float32,
                         device=device)
    if finite_fraction < 1.0:
        # Mask out (1 - finite_fraction) of the entries with -inf, mimicking
        # rows where only a subset of the vocab survived upstream filtering
        # (e.g. top_k/top_p).
        drop_mask = torch.rand(batch_size, vocab_size,
                               device=device) >= finite_fraction
        logits = logits.masked_fill(drop_mask, float("-inf"))
    return logits


def _make_num_sampled_tokens(batch_size, inactive_fraction, device):
    num_sampled_tokens = torch.ones(batch_size,
                                    dtype=torch.int32,
                                    device=device)
    num_inactive = int(round(batch_size * inactive_fraction))
    if num_inactive > 0:
        num_sampled_tokens[:num_inactive] = 0
    return num_sampled_tokens


def _check_against_reference(logits, num_sampled_tokens, max_num_kept):
    vocab_size = logits.shape[1]
    batch_size = logits.shape[0]

    token_ids, packed_mask, counts = compact_sampling_mask_xpu(
        logits, num_sampled_tokens, max_num_kept)

    ref_token_ids, ref_packed_mask, ref_counts = compact_sampling_mask_torch(
        logits, num_sampled_tokens, max_num_kept)

    torch.testing.assert_close(counts, ref_counts, rtol=0, atol=0)
    torch.testing.assert_close(packed_mask, ref_packed_mask, rtol=0, atol=0)

    # token_ids beyond a row's count are unspecified (never read downstream),
    # so only compare the valid compact prefix for rows that fit.
    counts_np = counts.cpu().numpy()
    width = token_ids.shape[1]
    for row in range(batch_size):
        n = int(counts_np[row])
        if n <= width:
            torch.testing.assert_close(token_ids[row, :n],
                                       ref_token_ids[row, :n],
                                       rtol=0,
                                       atol=0)

    # Cross-check against the downstream consumer's fallback logic
    # (compact token_ids vs. unpacked bitmask) for full self-consistency.
    supports = unpack_support(token_ids, packed_mask, counts, vocab_size)
    ref_supports = unpack_support(ref_token_ids, ref_packed_mask, ref_counts,
                                  vocab_size)
    for row in range(batch_size):
        np.testing.assert_array_equal(supports[row], ref_supports[row])


@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZE)
@pytest.mark.parametrize("max_num_kept", MAX_NUM_KEPT)
@pytest.mark.parametrize("inactive_fraction", INACTIVE_FRACTION)
@pytest.mark.parametrize("finite_fraction", FINITE_FRACTION)
def test_compact_sampling_mask(batch_size, vocab_size, max_num_kept,
                               inactive_fraction, finite_fraction):

    seed_everything(42)

    max_num_kept = min(max_num_kept, vocab_size)

    logits = _make_logits(batch_size, vocab_size, finite_fraction, DEVICE)
    num_sampled_tokens = _make_num_sampled_tokens(batch_size,
                                                  inactive_fraction, DEVICE)

    _check_against_reference(logits, num_sampled_tokens, max_num_kept)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("vocab_size", [1, 7, 33, 50, 513, 1001])
@pytest.mark.parametrize("max_num_kept", [1, 8, 32])
@pytest.mark.parametrize("finite_fraction", [0.3, 1.0])
def test_compact_sampling_mask_unaligned_vocab_size(batch_size, vocab_size,
                                                    max_num_kept,
                                                    finite_fraction):
    """vocab_size not a multiple of the wave size / byte width exercises the
    scalar-fallback load/store and tail-byte packing paths."""
    seed_everything(42)

    max_num_kept = min(max_num_kept, vocab_size)

    logits = _make_logits(batch_size, vocab_size, finite_fraction, DEVICE)
    num_sampled_tokens = _make_num_sampled_tokens(batch_size, 0.0, DEVICE)

    _check_against_reference(logits, num_sampled_tokens, max_num_kept)


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("vocab_size", [256, 2048])
def test_compact_sampling_mask_all_inactive(batch_size, vocab_size):
    """Rows with no sampled token must report count=0 and an all-zero
    bitmask/token_ids, regardless of the (irrelevant) logits contents."""
    seed_everything(42)

    logits = torch.randn(batch_size,
                         vocab_size,
                         dtype=torch.float32,
                         device=DEVICE)
    num_sampled_tokens = torch.zeros(batch_size,
                                     dtype=torch.int32,
                                     device=DEVICE)

    token_ids, packed_mask, counts = compact_sampling_mask_xpu(
        logits, num_sampled_tokens, max_num_kept=32)

    torch.testing.assert_close(counts,
                               torch.zeros_like(counts),
                               rtol=0,
                               atol=0)
    torch.testing.assert_close(packed_mask,
                               torch.zeros_like(packed_mask),
                               rtol=0,
                               atol=0)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("vocab_size", [1024])
@pytest.mark.parametrize("infinite_value", ["inf", "nan"])
def test_compact_sampling_mask_non_finite_values(batch_size, vocab_size,
                                                 infinite_value):
    """+inf/nan logits must never be counted as finite/kept."""
    seed_everything(42)

    logits = torch.randn(batch_size,
                         vocab_size,
                         dtype=torch.float32,
                         device=DEVICE)
    logits[:, ::2] = float(infinite_value)
    num_sampled_tokens = torch.ones(batch_size,
                                    dtype=torch.int32,
                                    device=DEVICE)

    _check_against_reference(logits, num_sampled_tokens, max_num_kept=64)
