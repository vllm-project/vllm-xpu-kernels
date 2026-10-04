# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from tests.register_ops import topk_per_row_decode, topk_per_row_prefill

DEVICE = "cpu"
KERNEL_DEVICE = "xpu"


def _to_kernel(x):
    return None if x is None else x.to(KERNEL_DEVICE)


def _assert_exact_rows(logits, output, starts, ends, top_k):
    values = logits.cpu()
    selected = output.cpu().long()
    for row in range(values.shape[0]):
        begin = int(starts[row])
        end = int(ends[row])
        length = end - begin
        count = min(top_k, length)
        indices = selected[row, :count]
        assert torch.all((indices >= 0) & (indices < length)), row
        assert indices.unique().numel() == count, row
        assert torch.all(selected[row, count:] == -1), row
        actual = values[row, begin:end][indices].sort(descending=True).values
        expected = values[row, begin:end].topk(count).values
        torch.testing.assert_close(actual,
                                   expected,
                                   rtol=0,
                                   atol=0,
                                   msg=f"row {row}")


BOUNDARY_CASES = [
    ("prefill", 4096, 512),
    ("prefill", 4097, 512),
    ("prefill", 8192, 1024),
    ("prefill", 8193, 1024),
    ("prefill", 10240, 2048),
    ("prefill", 10240, 2049),
    ("prefill", 10241, 2048),
    ("decode", 4096, 512),
    ("decode", 4097, 512),
    ("decode", 8192, 1024),
    ("decode", 8193, 1024),
    ("decode", 10240, 2048),
    ("decode", 10240, 2049),
    ("decode", 10241, 2048),
]
MEDIUM_CASES = [
    ("decode", 4097, 1, "random", 1),
    ("prefill", 10240, 512, "random", 1),
    ("decode", 8320, 512, "clustered", 1),
    ("prefill", 8193, 2048, "equal", 1),
    ("prefill", 8320, 512, "random", 2),
    ("decode", 8320, 512, "sample-high", 1),
    ("decode", 10240, 512, "random", 2),
    ("decode", 10240, 2048, "random", 1),
    ("decode", 8193, 2048, "equal", 1),
]
SPECIAL_CASES = [
    (mode, 8193, 512, "infinity", 1, "infinity")
    for mode in ("prefill", "decode")
] + [(mode, 8320, top_k, "random", 1, scenario)
     for mode in ("prefill", "decode")
     for top_k, scenario in ((512, "empty-rows"), (0, "empty-k"))]


@pytest.mark.parametrize(
    "mode,width,top_k,distribution,stride,scenario",
    [(*case, "random", 1, "boundary") for case in BOUNDARY_CASES] +
    [(*case, "medium") for case in MEDIUM_CASES] + SPECIAL_CASES)
@torch.inference_mode()
def test_topk_dispatch_boundaries_and_dirty_padding(
        mode, width, top_k, distribution, stride, scenario):
    """Check dispatch, invalid padding, and medium-path distributions."""
    torch.manual_seed(87 if scenario == "boundary" else 123)
    rows = {
        "boundary": 7,
        "medium": 32,
        "infinity": 4,
        "empty-k": 4,
        "empty-rows": 0,
    }[scenario]
    starts = torch.zeros(rows, dtype=torch.int32)
    if scenario == "boundary":
        if mode == "prefill":
            starts[1::2] = 7
        lengths = torch.tensor([0, 1, top_k - 1, top_k, top_k + 1, width // 2,
                                width - 7],
                               dtype=torch.int32)
        ends = starts + lengths
        next_n = 1
    elif scenario == "infinity" or scenario == "empty-k":
        ends = torch.full((rows, ), width, dtype=torch.int32)
        next_n = 1
    elif scenario == "empty-rows":
        ends = torch.empty(0, dtype=torch.int32)
        next_n = 1
    elif mode == "prefill":
        ends = torch.linspace(0, width, rows).to(torch.int32)
        starts[rows // 2:] = 7
        next_n = 1
    else:
        next_n = 8
        seq_lens = torch.tensor([8, top_k + 8, width - 1, width],
                                dtype=torch.int32)
        positions = torch.arange(rows)
        ends = (seq_lens[positions // next_n] - next_n
                + positions % next_n + 1)

    storage = torch.randn(rows, width * stride + 13)
    logits = storage[:, :width * stride:stride]
    if distribution == "clustered":
        logits.mul_(1e-6).add_(1)
    elif distribution == "equal":
        logits.fill_(1)
    elif distribution == "sample-high":
        logits[:, :128] = 1e6
    elif distribution == "infinity":
        logits.fill_(-float("inf"))
        logits[0, :top_k + 8] = torch.arange(top_k + 8)
        logits[1, :top_k + 8] = 1
        logits[1, 0] = float("inf")
        logits[2, :top_k + 8] = -1
        logits[2, top_k] = -float("inf")
        logits[3, :top_k + 8] = torch.arange(top_k + 8) * 2**-23 + 1
    if scenario in ("boundary", "medium"):
        # Invalid columns can hold larger values or NaNs.
        for row in range(rows):
            logits[row, int(ends[row]):] = (1e20 if row % 2 == 0
                                            else float("nan"))
    xpu_logits = storage.to(KERNEL_DEVICE)[:, :width * stride:stride]
    assert xpu_logits.stride(0) == width * stride + 13
    assert xpu_logits.stride(1) == stride
    output = torch.full((rows, top_k),
                        -777,
                        dtype=torch.int32,
                        device=KERNEL_DEVICE)
    replay_lengths = ([width] * rows, [0, 1, top_k, top_k + 1],
                      [top_k + 1, width, 0, top_k - 1])
    for replay in range(3 if scenario == "infinity" else 2):
        if scenario == "infinity":
            ends = torch.tensor(replay_lengths[replay], dtype=torch.int32)
        output.fill_(-777)
        if mode == "prefill":
            torch.ops._C.top_k_per_row_prefill(
                xpu_logits, starts.to(KERNEL_DEVICE), ends.to(KERNEL_DEVICE),
                output, rows, xpu_logits.stride(0), xpu_logits.stride(1),
                top_k)
        else:
            if next_n == 1:
                seq_lens = ends
            torch.ops._C.top_k_per_row_decode(
                xpu_logits, next_n, seq_lens.to(KERNEL_DEVICE), output, rows,
                xpu_logits.stride(0), xpu_logits.stride(1), top_k)
        _assert_exact_rows(logits, output, starts, ends, top_k)


@torch.inference_mode()
def test_medium_topk_decode_mtp_reused_output():
    """Adapt vLLM's multi-token length expansion and changed-length replay."""
    next_n, width, top_k = 4, 10240, 512
    rows = 3 * next_n
    torch.manual_seed(14)
    logits = torch.randn(rows, width)
    xpu_logits = logits.to(KERNEL_DEVICE)
    starts = torch.zeros(rows, dtype=torch.int32)
    output = torch.full((rows, top_k),
                        -777,
                        dtype=torch.int32,
                        device=KERNEL_DEVICE)
    for lengths in (
        ([next_n, top_k + next_n, width],
         [width, next_n, top_k + next_n])):
        seq_lens = torch.tensor(lengths, dtype=torch.int32)
        row_ids = torch.arange(rows)
        ends = (seq_lens[row_ids // next_n] - next_n + row_ids % next_n + 1)
        output.fill_(-777)
        torch.ops._C.top_k_per_row_decode(
            xpu_logits, next_n, seq_lens.to(KERNEL_DEVICE), output, rows,
            width, 1, top_k)
        _assert_exact_rows(logits, output, starts, ends, top_k)


# This file is same as
# https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_top_k_per_row.py
# with modification of testing XPU platform. Here just for quick testing, in
# future this could be removed and tested in upstream repo instead

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
