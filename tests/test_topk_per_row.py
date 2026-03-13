# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from tests.register_ops import topk_per_row_decode, topk_per_row_prefill

# This file is same as https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_top_k_per_row.py with
# modification of testing XPU platform. Here just for quick testing, in future this could be removed and
# tested in upstream repo instead

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_rows": [8],
        "top_k": [128],
        "batch_size": [2],
        "next_n": [2],
        "vocab_size": [2000],
    },
}

# Test parameters
NUM_ROWS = [1, 32, 2050, 8239]
TOP_K_VALUES = [2048, 3000]
BATCH_SIZE = [1, 2, 512]
NEXT_N = [1, 8]
DATA_GENERATION = ["random", "10LSBits"]
# > 200 * 1000 to test split work, 
# only use for large vocab test
VOCAB_SIZE = [300000]


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    data_generation: str,
) -> torch.Tensor:
    """Create random logits tensor for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Generate logits with some structure to make testing more meaningful
    if data_generation == "random":
        logits = torch.randn(row_starts.shape[0],
                             max(row_ends),
                             dtype=dtype,
                             device="xpu")
    elif data_generation == "10LSBits":
        top_22_bits_mask = 0xFFFFFC00
        last_10_bits_mask = 0x000003FF
        fixed_top_22_bits = 0x3F900000
        # Generate random bits for the last 10 bits
        random_bottom_bits = torch.randint(
            0,
            2**10,
            (row_starts.shape[0], max(row_ends)),
            dtype=torch.int32,
            device="xpu",
        )
        # Combine: fixed top 22 bits with random last 10 bits
        logits_bits = (fixed_top_22_bits & top_22_bits_mask) | (
            random_bottom_bits & last_10_bits_mask)
        logits = logits_bits.view(dtype)

    for i, end in enumerate(row_ends):
        logits[i, end:] = float("-inf")
    return logits


def create_row_boundaries(
        seq_len: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create row start and end indices for testing."""
    row_starts = torch.zeros(seq_len, dtype=torch.int32, device="xpu")
    row_ends = torch.arange(1, seq_len + 1, device="xpu", dtype=torch.int32)
    return row_starts, row_ends


def compare_top_k_results(
    logits: torch.Tensor,
    xpu_indices: torch.Tensor,
    torch_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare results from XPU top_k_per_row with torch.topk.
    Both results should be sorted and contain the same top-k elements.
    """
    num_rows = xpu_indices.shape[0]

    for row_idx in range(num_rows):
        # Get valid elements using row boundaries
        row_start = row_starts[row_idx].item()
        row_end = row_ends[row_idx].item()
        row_length = row_end - row_start
        num_valid = min(top_k, row_length)
        xpu_row_indices = xpu_indices[row_idx][:num_valid].cpu()
        torch_row_indices = torch_indices[row_idx][:num_valid].cpu()

        # Compare the sets of indices first
        xpu_set = set(xpu_row_indices.tolist())
        torch_set = set(torch_row_indices.tolist())
        if xpu_set == torch_set:
            continue

        # Any difference in elements, compare the values
        logits_row = logits[row_idx]
        xpu_row_values = [logits_row[i] for i in xpu_row_indices]
        torch_row_values = [logits_row[i] for i in torch_row_indices]

        xpu_only_values, torch_only_values = [], []
        for idx in xpu_set - torch_set:
            xpu_pos = (xpu_row_indices == idx).nonzero(as_tuple=True)[0]
            xpu_only_values.append(xpu_row_values[xpu_pos[0]])

        for idx in torch_set - xpu_set:
            torch_pos = (torch_row_indices == idx).nonzero(as_tuple=True)[0]
            torch_only_values.append(torch_row_values[torch_pos[0]])

        if len(xpu_only_values) != len(torch_only_values):
            return False
        if not torch.allclose(
                torch.tensor(xpu_only_values, device="xpu"),
                torch.tensor(torch_only_values, device="xpu"),
                rtol=tolerance,
                atol=tolerance,
        ):
            return False

    return True


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@torch.inference_mode()
def test_top_k_per_row(
    num_rows: int,
    top_k: int,
) -> None:
    """
    Test top_k_per_row.
    """
    torch.set_default_device("xpu")
    device = "xpu"
    torch.xpu.memory.empty_cache()

    # Create test data
    vocab_size = 20000
    row_starts, row_ends = create_row_boundaries(num_rows, vocab_size)
    logits = create_random_logits(row_starts, row_ends, torch.float32, 42,
                                  "random")

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)

    # Run XPU implementation
    topk_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        top_k,
    )

    # Run reference implementation
    torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends,
        top_k), "XPU top_k_per_row_prefill results don't match torch.topk"


def _run_top_k_per_row_decode_test(
    top_k: int,
    batch_size: int,
    next_n: int,
    vocab_size: int,
    data_generation: str,
) -> None:
    """
    Helper function to run top_k_per_row_decode test with given parameters.
    """
    torch.set_default_device("xpu")
    device = "xpu"

    # Create test data
    num_rows = batch_size * next_n
    seq_lens = torch.randint(vocab_size, (batch_size, ),
                             dtype=torch.int32,
                             device=device)
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    row_indices = torch.arange(num_rows, device=device) // next_n
    next_n_offset = torch.arange(num_rows, device=device) % next_n
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1
    logits = create_random_logits(row_starts, row_ends, torch.float32, 42,
                                  data_generation)

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)

    # Run XPU implementation
    topk_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        top_k,
    )

    torch.xpu.synchronize()

    # Run reference implementation
    torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    torch.xpu.synchronize()

    # Compare results
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends,
        top_k), "XPU top_k_per_row_decode results don't match torch.topk"


@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("next_n", NEXT_N)
@pytest.mark.parametrize("data_generation", DATA_GENERATION)
@torch.inference_mode()
def test_top_k_per_row_decode(
    top_k: int,
    batch_size: int,
    next_n: int,
    data_generation: str,
) -> None:
    """
    Test top_k_per_row with seq_lens tensor.
    """
    torch.xpu.memory.empty_cache()

    vocab_size = 20000
    _run_top_k_per_row_decode_test(top_k, batch_size, next_n, vocab_size,
                                   data_generation)


@pytest.mark.parametrize("vocab_size", VOCAB_SIZE)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@torch.inference_mode()
def test_top_k_per_row_decode_large_vocab_size(
    vocab_size: int,
    top_k: int,
) -> None:
    """
    Test top_k_per_row_decode with large vocabulary size.
    """
    torch.xpu.memory.empty_cache()

    batch_size = 8
    next_n = 2
    data_generation = "random"
    _run_top_k_per_row_decode_test(top_k, batch_size, next_n, vocab_size,
                                   data_generation)
