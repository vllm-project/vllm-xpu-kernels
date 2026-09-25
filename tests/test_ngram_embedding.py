# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the LongCat n-gram embedding id kernel.

Mirrors the CUDA kernel in
https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/ngram_embedding_kernels.cu
and its caller ``LongcatNgramModelState._compute_oe_ids`` in
vllm/model_executor/models/longcat_flash_ngram.py.
"""

import pytest
import torch

from tests.register_ops import ngram_compute_n_gram_ids
from tests.utils import opcheck

DEVICE = "xpu"

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "ne_n": [3],
        "ne_k": [2],
        "req_lens": [[1, 17, 0, 300]],
    },
}

NE_N = [2, 3, 5]
NE_K = [1, 4]
# Covers single token (decode), > work-group size (256) ragged prefill,
# zero-length requests and many small requests.
REQ_LENS = [
    [1],
    [7],
    [300],
    [1, 17, 0, 512],
    [1] * 64,
    [0, 0],
]

VOCAB_SIZE = 131072
NGRAM_VOCAB_SIZE_RATIO = 78


def make_ngram_tables(ne_n: int, ne_k: int, vocab: int, m: int):
    """Same construction as ``NgramEmbedding._init_ngram_embeddings``."""
    num_embedders = ne_k * (ne_n - 1)
    ne_weights = torch.zeros(ne_n - 1, ne_k, ne_n, dtype=torch.int32)
    ne_mods = torch.zeros(ne_n - 1, ne_k, dtype=torch.int32)
    for i in range(ne_n - 1):
        for j in range(ne_k):
            mod = int(m + 2 * (i * ne_k + j) + 1)
            ne_mods[i, j] = mod
            for delta in range(ne_n):
                ne_weights[i, j, delta] = pow(vocab, delta, mod)
    offsets = [0]
    for i in range(num_embedders):
        offsets.append(offsets[-1] + int(m + i * 2 + 1))
    exclusive_sizes = torch.tensor(offsets, dtype=torch.int32)
    return ne_weights, ne_mods, exclusive_sizes


def ref_compute_n_gram_ids(ne_n, ne_k, ne_weights, ne_mods,
                           exclusive_ne_embedder_size_sums,
                           exclusive_req_len_sums, ne_token_table, row_indices,
                           column_starts) -> torch.Tensor:
    """Vectorized int64 CPU reference of ``ComputeNGramIdsKernel``."""
    w = ne_weights.cpu().long()
    mods = ne_mods.cpu().long()
    excl = exclusive_ne_embedder_size_sums.cpu().long()
    qsl = exclusive_req_len_sums.cpu().long()
    table = ne_token_table.cpu().long()
    rows = row_indices.cpu().long()
    cols = column_starts.cpu().long()

    num_reqs = qsl.numel() - 1
    num_tokens = int(qsl[-1].item())
    num_configs = (ne_n - 1) * ne_k
    out = torch.empty(num_tokens, num_configs, dtype=torch.int64)
    if num_tokens == 0:
        return out.int()

    max_context_len = table.shape[1]
    flat = table.flatten()
    req_lens = qsl[1:] - qsl[:-1]
    tok_req = torch.repeat_interleave(torch.arange(num_reqs), req_lens)
    offset = torch.arange(num_tokens) - qsl[:-1][tok_req]
    base = rows[tok_req] * max_context_len
    cur = base + cols[tok_req] + offset

    for n in range(ne_n - 1):
        for k in range(ne_k):
            mod = mods[n, k]
            acc = torch.zeros(num_tokens, dtype=torch.int64)
            alive = torch.ones(num_tokens, dtype=torch.bool)
            for j in range(n + 2):
                idx = cur - j
                alive &= idx >= base
                tok = flat[idx.clamp(min=0)]
                alive &= tok >= 0
                term = (tok.clamp(min=0) * w[n, k, j]) % mod
                acc += torch.where(alive, term, 0)
            out[:, n * ne_k + k] = acc % mod + excl[n * ne_k + k]
    return out.int()


def run_kernel(ne_n, ne_k, ne_weights, ne_mods, exclusive_sizes, qsl, table,
               row_indices, column_starts) -> torch.Tensor:
    num_tokens = int(qsl[-1].item())
    n_gram_ids = torch.full((num_tokens, (ne_n - 1) * ne_k),
                            -1,
                            dtype=torch.int32,
                            device=DEVICE)
    ngram_compute_n_gram_ids(ne_n, ne_k, ne_weights, ne_mods, exclusive_sizes,
                             qsl, table, row_indices, column_starts,
                             n_gram_ids)
    return n_gram_ids


def make_qsl(req_lens: list[int]) -> torch.Tensor:
    lens = torch.tensor(req_lens, dtype=torch.int32)
    return torch.cat([torch.zeros(1, dtype=torch.int32),
                      torch.cumsum(lens, 0, dtype=torch.int32)])


@pytest.mark.parametrize("ne_n", NE_N)
@pytest.mark.parametrize("ne_k", NE_K)
@pytest.mark.parametrize("req_lens", REQ_LENS)
@torch.inference_mode()
def test_ngram_compute_n_gram_ids(ne_n: int, ne_k: int,
                                  req_lens: list[int]) -> None:
    """Scattered rows / column starts in a larger persistent token table."""
    torch.manual_seed(0)
    num_reqs = len(req_lens)
    max_running_reqs = num_reqs + 5
    max_context_len = max(req_lens) + 64

    ne_weights, ne_mods, exclusive_sizes = make_ngram_tables(
        ne_n, ne_k, VOCAB_SIZE, NGRAM_VOCAB_SIZE_RATIO * VOCAB_SIZE)

    table = torch.randint(0,
                          VOCAB_SIZE, (max_running_reqs, max_context_len),
                          dtype=torch.int32)
    # Negative entries mark ignored tokens (EOS boundary / padding).
    ignored = torch.rand(table.shape) < 0.05
    table[ignored] = -table[ignored] - 1

    row_indices = torch.randperm(max_running_reqs)[:num_reqs].long()
    column_starts = torch.tensor([
        int(torch.randint(0, max_context_len - L + 1, (1, )).item())
        for L in req_lens
    ],
                                 dtype=torch.int32)
    # Force some requests to start at column 0 to hit the row-start boundary.
    column_starts[::2] = 0
    qsl = make_qsl(req_lens)

    args = [ne_weights, ne_mods, exclusive_sizes, qsl, table, row_indices,
            column_starts]
    expected = ref_compute_n_gram_ids(ne_n, ne_k, *args)
    actual = run_kernel(ne_n, ne_k, *[a.to(DEVICE) for a in args])

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("ne_n", [2, 4])
@pytest.mark.parametrize("ne_k", [3])
@torch.inference_mode()
def test_ngram_compute_n_gram_ids_random_weights(ne_n: int,
                                                 ne_k: int) -> None:
    """Large weights / tokens near int32 max stress the uint64 modular sum."""
    torch.manual_seed(1)
    req_lens = [5, 260, 1]
    num_reqs = len(req_lens)
    max_context_len = 300
    int32_max = torch.iinfo(torch.int32).max

    ne_mods = torch.randint(int32_max // 2,
                            int32_max, (ne_n - 1, ne_k),
                            dtype=torch.int32)
    ne_weights = (torch.randint(0, int32_max, (ne_n - 1, ne_k, ne_n),
                                dtype=torch.int64) %
                  ne_mods.long().unsqueeze(-1)).int()
    exclusive_sizes = torch.randint(0,
                                    1 << 20, ((ne_n - 1) * ne_k + 1, ),
                                    dtype=torch.int32)
    table = torch.randint(int32_max - 1000,
                          int32_max, (num_reqs, max_context_len),
                          dtype=torch.int32)
    row_indices = torch.arange(num_reqs, dtype=torch.int64)
    column_starts = torch.tensor([3, 0, 299], dtype=torch.int32)
    qsl = make_qsl(req_lens)

    args = [ne_weights, ne_mods, exclusive_sizes, qsl, table, row_indices,
            column_starts]
    expected = ref_compute_n_gram_ids(ne_n, ne_k, *args)
    actual = run_kernel(ne_n, ne_k, *[a.to(DEVICE) for a in args])

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("ne_n", [3, 5])
@pytest.mark.parametrize("ne_k", [4])
@torch.inference_mode()
def test_ngram_compute_n_gram_ids_model_flow(ne_n: int, ne_k: int) -> None:
    """Prefill + decode steps laid out like ``_compute_oe_ids`` in vLLM."""
    torch.manual_seed(2)
    eos_id = 2
    ctx_len = ne_n - 1
    num_reqs = 3
    ne_weights, ne_mods, exclusive_sizes = make_ngram_tables(
        ne_n, ne_k, VOCAB_SIZE, NGRAM_VOCAB_SIZE_RATIO * VOCAB_SIZE)
    ne_weights = ne_weights.to(DEVICE)
    ne_mods = ne_mods.to(DEVICE)
    exclusive_sizes = exclusive_sizes.to(DEVICE)
    # Fresh requests have an all-ignored context.
    token_context = torch.full((num_reqs, ctx_len),
                               -1,
                               dtype=torch.int32,
                               device=DEVICE)

    for step_lens in ([9, 300, 1], [1, 1, 1], [1, 4, 1]):
        qsl = make_qsl(step_lens).to(DEVICE)
        num_tokens = int(qsl[-1].item())
        cur = torch.randint(0, VOCAB_SIZE, (num_tokens, ),
                            dtype=torch.int32,
                            device=DEVICE)
        cur[::7] = eos_id
        cur_neg = torch.where(cur == eos_id, -cur, cur)
        req_lens = qsl[1:] - qsl[:-1]
        width = ctx_len + int(req_lens.max().item())

        table = torch.full((num_reqs, width),
                           -1,
                           dtype=torch.int32,
                           device=DEVICE)
        table[:, :ctx_len] = token_context
        tok_req = torch.repeat_interleave(
            torch.arange(num_reqs, device=DEVICE), req_lens.long())
        col = ctx_len + (torch.arange(num_tokens, device=DEVICE) -
                         qsl[:-1].long()[tok_req])
        table[tok_req, col] = cur_neg
        column_starts = torch.full((num_reqs, ),
                                   ctx_len,
                                   dtype=torch.int32,
                                   device=DEVICE)
        row_indices = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)

        args = [ne_weights, ne_mods, exclusive_sizes, qsl, table,
                row_indices, column_starts]
        expected = ref_compute_n_gram_ids(ne_n, ne_k, *args)
        actual = run_kernel(ne_n, ne_k, *args)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

        gather = req_lens.long().unsqueeze(1) + torch.arange(
            ctx_len, device=DEVICE).unsqueeze(0)
        rows = torch.arange(num_reqs, device=DEVICE).unsqueeze(1)
        token_context = table[rows, gather]


@torch.inference_mode()
def test_ngram_compute_n_gram_ids_empty_batch() -> None:
    ne_n, ne_k = 3, 2
    ne_weights, ne_mods, exclusive_sizes = make_ngram_tables(
        ne_n, ne_k, VOCAB_SIZE, NGRAM_VOCAB_SIZE_RATIO * VOCAB_SIZE)
    qsl = torch.zeros(1, dtype=torch.int32)
    table = torch.zeros((1, 8), dtype=torch.int32)
    row_indices = torch.zeros(0, dtype=torch.int64)
    column_starts = torch.zeros(0, dtype=torch.int32)
    args = [ne_weights, ne_mods, exclusive_sizes, qsl, table, row_indices,
            column_starts]
    actual = run_kernel(ne_n, ne_k, *[a.to(DEVICE) for a in args])
    assert actual.shape == (0, (ne_n - 1) * ne_k)


@torch.inference_mode()
def test_ngram_compute_n_gram_ids_opcheck() -> None:
    ne_n, ne_k = 3, 2
    ne_weights, ne_mods, exclusive_sizes = make_ngram_tables(
        ne_n, ne_k, VOCAB_SIZE, NGRAM_VOCAB_SIZE_RATIO * VOCAB_SIZE)
    req_lens = [4, 1]
    qsl = make_qsl(req_lens)
    table = torch.randint(0, VOCAB_SIZE, (2, 8), dtype=torch.int32)
    row_indices = torch.arange(2, dtype=torch.int64)
    column_starts = torch.tensor([2, 2], dtype=torch.int32)
    n_gram_ids = torch.empty((sum(req_lens), (ne_n - 1) * ne_k),
                             dtype=torch.int32)
    args = [
        t.to(DEVICE) for t in (ne_weights, ne_mods, exclusive_sizes, qsl,
                               table, row_indices, column_starts, n_gram_ids)
    ]
    opcheck(torch.ops._C.ngram_compute_n_gram_ids, (ne_n, ne_k, *args))
