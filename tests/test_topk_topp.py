# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn

from typing import Literal

import vllm_xpu_kernels._xpu_C  # noqa: F401
from tests.utils import seed_everything

DEVICE = "xpu"

BATCH_SIZE = [1, 32, 1024]
VOCAL_SIZE = [1024, 2048, 4096]
K = [1, 32, 128, 1024, None]
P = [0.1, 0.2, 0.4, 0.8, 1.0, None]
LOGPROBS_MODE = ["raw_logits", "processed_logits", "processed_logprobs"]

BATCH_SIZE = [2048]
VOCAL_SIZE = [131072]
K = [32]
P = [None]
LOGPROBS_MODE = ["raw_logits"]


LogprobsMode = Literal[
    "raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"
]

def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        generator = torch.xpu.default_generators[q.device.index]
        state = generator.get_state()
        seed, _ = state.view(torch.int64)
        # replace for UT pass
        # q.exponential_()
        offset = 0
        seeds = torch.tensor([seed, offset], dtype=torch.int64, device=torch.device("cpu"))
        torch.ops._xpu_C.exponential_2d_(q, seeds, 1.0)
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)
    # return probs.argmax(dim=-1).view(-1)

def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.
    Note however that it involves a GPU->CPU sync which can be detrimental for
    async scheduling performance.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    return logits.masked_fill_(logits < top_k_mask, -float("inf"))

def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    return apply_top_k_top_p_pytorch(logits, k, p)

class TopKTopPSampler(nn.Module):
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None:
        super().__init__()
        self.logprobs_mode = logprobs_mode

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return

    def forward_xpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        seeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        XPU implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        if (k is None and p is None) or generators:
            return self.forward_native(logits, generators, k, p)
        batch_size = logits.shape[0]
        random_sampled = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
        processed_logprobs = None
        if self.logprobs_mode == "processed_logits" or self.logprobs_mode == "processed_logprobs":
            processed_logprobs = torch.empty_like(logits)

        processed_logprobs = torch.empty_like(logits)

        # logits[:] = logits.softmax(dim=-1, dtype=torch.float32)
        # if(k is not None and p is not None):
        #     torch.ops._xpu_C.top_k_mask_logits(logits, top_k)

        logits = logits.softmax(dim=-1, dtype=torch.float32)

        if seeds is None:
            generator = torch.xpu.default_generators[logits.device.index]
            state = generator.get_state()
            seed, _ = state.view(torch.int64)
            offset = 0
            seeds = torch.tensor([seed, offset], dtype=torch.int64, device=torch.device("cpu"))

        torch.ops._xpu_C.topk_topp_sampler(
            random_sampled,
            processed_logprobs,
            logits,
            k,
            p,
            self.logprobs_mode,
            seeds,
            1.0)
        return random_sampled, processed_logprobs



@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("vocal_size", VOCAL_SIZE)
@pytest.mark.parametrize("k", K)
@pytest.mark.parametrize("p", P)
@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODE)
def test_topk_topp(batch_size, vocal_size, k, p, logprobs_mode):

    seed_everything(0)

    generators={}

    if generators:
        print("True")
    else:
        print("False")

    logits = torch.randn(
        batch_size, vocal_size, dtype=torch.float, device=DEVICE
    )
    ref_logits = logits.clone()

    top_k = None
    top_p = None
    if k is not None:
        top_k = torch.randint(1, k + 1, (batch_size,), device=DEVICE)
        print("top_k", top_k.shape, top_k.dtype)
    if p is not None:
        top_p = 1.0 - torch.rand(
            batch_size, dtype=torch.float, device=DEVICE
        )

    topk_topp_sampler = TopKTopPSampler(logprobs_mode=logprobs_mode)

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.XPU,
            ],
            record_shapes=True,
        ) as prof:
        random_sampled, processed_logprobs = topk_topp_sampler.forward_xpu(
            logits=logits,
            generators=generators,
            k=top_k,
            p=top_p,
        )
    print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=10), flush=True)

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.XPU,
            ],
            record_shapes=True,
        ) as prof_ref:
        ref_random_sampled, ref_processed_logprobs = topk_topp_sampler.forward_native(
            logits=ref_logits,
            generators=generators,
            k=top_k,
            p=top_p,
        )
    print(prof_ref.key_averages().table(sort_by="self_xpu_time_total", row_limit=10), flush=True)

    # print("ref_random_sampled", ref_random_sampled.shape, ref_random_sampled.dtype, ref_random_sampled)
    # print("random_sampled", random_sampled[468])
    # print("ref_random_sampled", ref_random_sampled[468])

    # print("logits", logits)
    # print("ref_logits", ref_logits)
    # print("processed_logprobs", processed_logprobs)

    # mask1 = logits != float('-inf')
    # mask2 = ref_logits != float('-inf')

    # torch.testing.assert_close(mask1.sum(-1), top_k, rtol=0, atol=0)
    torch.testing.assert_close(random_sampled, ref_random_sampled, rtol=0, atol=0)
    if processed_logprobs is not None:
        # print("ref_processed_logprobs", ref_processed_logprobs.shape, ref_processed_logprobs.dtype)
        # torch.testing.assert_close(processed_logprobs, ref_processed_logprobs, rtol=1e-2, atol=1e-2)
        pass
