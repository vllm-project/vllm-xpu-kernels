# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.topk_topp_sampler_op import TopKTopPSampler
from tests.utils import seed_everything

DEVICE = "xpu"

BATCH_SIZE = [1, 32, 1024]
VOCAB_SIZE = [1024, 2048, 4096]
K = [1, 32, 128, 1024, None]
P = [0.1, 0.2, 0.4, 0.8, 1.0, None]
LOGPROBS_MODE = ["raw_logits", "processed_logits", "processed_logprobs"]

# CI/mini scope parameter overrides
MINI_PYTEST_PARAMS = {
    "default": {
        "batch_size": [1, 32],
        "vocab_size": [1024],
        "k": [1, 128, None],
        "p": [0.5, None],
        "logprobs_mode": ["raw_logits"],
    },
}


def _run_xpu_sampler(logits, top_k, top_p, seed_offsets):
    random_sampled = torch.empty(
        logits.shape[0], dtype=torch.int64, device=logits.device)
    torch.ops._xpu_C.topk_topp_sampler(
        random_sampled,
        None,
        logits,
        top_k,
        top_p,
        "raw_logits",
        seed_offsets,
        1.0,
    )
    return random_sampled


@pytest.mark.parametrize(
    ("top_k_value", "top_p_value"),
    [(None, None), (32, None), (None, 0.8), (32, 0.8)],
)
def test_per_row_seed_offsets_match_individual_calls(
        top_k_value, top_p_value):
    batch_size = 8
    vocab_size = 1024
    generator = torch.Generator(device="cpu").manual_seed(42)
    logits = torch.randn(
        batch_size, vocab_size, generator=generator).to(DEVICE)
    top_k = (
        torch.full(
            (batch_size, ), top_k_value, dtype=torch.int64, device=DEVICE)
        if top_k_value is not None else None
    )
    top_p = (
        torch.full(
            (batch_size, ), top_p_value, dtype=torch.float32, device=DEVICE)
        if top_p_value is not None else None
    )
    seed_offsets = torch.tensor(
        [[100 + row, row * vocab_size] for row in range(batch_size)],
        dtype=torch.int64,
        device=DEVICE,
    )

    batched = _run_xpu_sampler(
        logits.clone(), top_k, top_p, seed_offsets)
    individual = torch.stack([
        _run_xpu_sampler(
            logits[row:row + 1].clone(),
            top_k[row:row + 1] if top_k is not None else None,
            top_p[row:row + 1] if top_p is not None else None,
            seed_offsets[row].cpu(),
        )[0]
        for row in range(batch_size)
    ])
    torch.testing.assert_close(batched, individual, rtol=0, atol=0)

    permutation = torch.tensor(
        [5, 2, 7, 0, 3, 6, 1, 4], dtype=torch.int64, device=DEVICE)
    permuted = _run_xpu_sampler(
        logits[permutation].clone(),
        top_k[permutation] if top_k is not None else None,
        top_p[permutation] if top_p is not None else None,
        seed_offsets[permutation],
    )
    torch.testing.assert_close(
        permuted, batched[permutation], rtol=0, atol=0)


def test_per_row_seed_offsets_validate_shape_and_device():
    batch_size = 2
    logits = torch.randn(
        batch_size, 128, dtype=torch.float32, device=DEVICE)

    with pytest.raises(
            RuntimeError, match="same device as logits"):
        _run_xpu_sampler(
            logits.clone(),
            None,
            None,
            torch.zeros(batch_size, 2, dtype=torch.int64),
        )

    with pytest.raises(
            RuntimeError, match=r"shape \[batch_size, 2\]"):
        _run_xpu_sampler(
            logits.clone(),
            None,
            None,
            torch.zeros(
                batch_size, 3, dtype=torch.int64, device=DEVICE),
        )

    with pytest.raises(RuntimeError, match="rank 1 or 2"):
        _run_xpu_sampler(
            logits.clone(),
            None,
            None,
            torch.zeros(
                batch_size, 2, 1, dtype=torch.int64, device=DEVICE),
        )


@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZE)
@pytest.mark.parametrize("k", K)
@pytest.mark.parametrize("p", P)
@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODE)
def test_topk_topp(batch_size, vocab_size, k, p, logprobs_mode):

    seed_everything(42)

    generators = {}

    logits = torch.randn(batch_size,
                         vocab_size,
                         dtype=torch.float,
                         device=DEVICE)
    ref_logits = logits.clone()

    top_k = None
    top_p = None
    if k is not None:
        if k != vocab_size:
            top_k = torch.randint(1, k + 1, (batch_size, ), device=DEVICE)
        else:
            top_k = torch.full((batch_size, ),
                               vocab_size,
                               dtype=torch.long,
                               device=DEVICE)
    if p is not None:
        if p != 1.0:
            top_p = 1.0 - torch.rand(
                batch_size, dtype=torch.float, device=DEVICE)
        else:
            top_p = torch.ones([batch_size], dtype=torch.float, device=DEVICE)

    topk_topp_sampler = TopKTopPSampler(logprobs_mode=logprobs_mode)

    random_sampled, logits_to_return = topk_topp_sampler.forward_xpu(
        logits=logits,
        generators=generators,
        k=top_k,
        p=top_p,
    )

    ref_random_sampled, ref_logits_to_return =\
        topk_topp_sampler.forward_native(
        logits=ref_logits,
        generators=generators,
        k=top_k,
        p=top_p,
    )

    torch.testing.assert_close(random_sampled,
                               ref_random_sampled,
                               rtol=0,
                               atol=0)
    if logits_to_return is not None:
        if top_p is None:
            torch.testing.assert_close(logits_to_return,
                                       ref_logits_to_return,
                                       rtol=1e-5,
                                       atol=1e-5)
        else:
            # Top-p involved: allow small differences
            # Either < 1% of kept values OR < 5 values absolute
            xpu_kept = (logits_to_return != float("-inf")).sum(dim=-1)
            ref_kept = (ref_logits_to_return != float("-inf")).sum(dim=-1)

            max_diff = (ref_kept - xpu_kept).abs().max().item()
            max_kept = ref_kept.max().item()
            if max_kept > 0 and max_diff > 3:
                diff_pct = max_diff / max_kept * 100
                assert diff_pct < 0.5, (
                    f"Top-p mask difference too large: {diff_pct:.2f}% "
                    f"(max diff {max_diff} values out of {max_kept})")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("vocab_size", [1024])
@pytest.mark.parametrize("k", K)
@pytest.mark.parametrize("p", P)
@pytest.mark.parametrize("logprobs_mode", ["raw_logits"])
@pytest.mark.parametrize("infinite_value", ["-inf", "inf", "nan"])
@pytest.mark.parametrize("is_full_infinite", [True, False])
def test_topk_topp_infinite(
    batch_size, vocab_size, k, p, logprobs_mode,
    infinite_value, is_full_infinite):

    seed_everything(42)

    generators = {}

    logits = torch.randn(batch_size,
                         vocab_size,
                         dtype=torch.float,
                         device=DEVICE)

    top_k = None
    top_p = None
    if k is not None:
        if k != vocab_size:
            top_k = torch.randint(1, k + 1, (batch_size, ), device=DEVICE)
        else:
            top_k = torch.full((batch_size, ),
                               vocab_size,
                               dtype=torch.long,
                               device=DEVICE)
    if p is not None:
        if p != 1.0:
            top_p = 1.0 - torch.rand(
                batch_size, dtype=torch.float, device=DEVICE)
        else:
            top_p = torch.ones([batch_size], dtype=torch.float, device=DEVICE)

    topk_topp_sampler = TopKTopPSampler(logprobs_mode=logprobs_mode)

    if not is_full_infinite:
        logits[0][0] = float(infinite_value)
    else:
        logits.fill_(float(infinite_value))

    random_sampled, logits_to_return = topk_topp_sampler.forward_xpu(
        logits=logits,
        generators=generators,
        k=top_k,
        p=top_p,
    )

    torch.xpu.synchronize()


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("vocab_size", [33, 50, 513])
@pytest.mark.parametrize("k", [1, 8, None])
@pytest.mark.parametrize("p", [0.5, None])
@pytest.mark.parametrize("logprobs_mode", ["raw_logits"])
def test_topk_topp_unaligned_vocab_size(
    batch_size, vocab_size, k, p, logprobs_mode):

    if k is None and p is None:
        pytest.skip("Nothing to sample against; covered by other tests.")

    seed_everything(42)

    generators = {}

    logits = torch.randn(batch_size,
                         vocab_size,
                         dtype=torch.float,
                         device=DEVICE)
    ref_logits = logits.clone()

    top_k = None
    top_p = None
    if k is not None:
        top_k = torch.randint(1, k + 1, (batch_size, ), device=DEVICE)
    if p is not None:
        top_p = 1.0 - torch.rand(
            batch_size, dtype=torch.float, device=DEVICE)

    topk_topp_sampler = TopKTopPSampler(logprobs_mode=logprobs_mode)

    random_sampled, logits_to_return = topk_topp_sampler.forward_xpu(
        logits=logits,
        generators=generators,
        k=top_k,
        p=top_p,
    )

    ref_random_sampled, ref_logits_to_return =\
        topk_topp_sampler.forward_native(
        logits=ref_logits,
        generators=generators,
        k=top_k,
        p=top_p,
    )

    torch.testing.assert_close(random_sampled,
                               ref_random_sampled,
                               rtol=0,
                               atol=0)
    if logits_to_return is not None:
        if top_p is None:
            torch.testing.assert_close(logits_to_return,
                                       ref_logits_to_return,
                                       rtol=1e-5,
                                       atol=1e-5)
        else:
            # Top-p involved: allow small differences
            # Either < 1% of kept values OR < 5 values absolute
            xpu_kept = (logits_to_return != float("-inf")).sum(dim=-1)
            ref_kept = (ref_logits_to_return != float("-inf")).sum(dim=-1)

            max_diff = (ref_kept - xpu_kept).abs().max().item()
            max_kept = ref_kept.max().item()
            if max_kept > 0 and max_diff > 3:
                diff_pct = max_diff / max_kept * 100
                assert diff_pct < 0.5, (
                    f"Top-p mask difference too large: {diff_pct:.2f}% "
                    f"(max diff {max_diff} values out of {max_kept})")
