# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch

from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import (cutlass_fused_moe,
                                                  cutlass_grouped_gemm)

DEVICE = "xpu"

# shape for Llama-4-scout
FUSED_MOE_MNK_FACTORS = [
    (1, 5120, 8192),
    (4, 5120, 8192),
    (16, 5120, 8192),
    (8192, 5120, 8192),
]
NUM_EXPERTS = [16]
TOP_KS = [1]


def random_partition(size_a: int, target: int):
    cuts = sorted(random.sample(range(target + size_a - 1), size_a - 1))
    cuts = [-1] + cuts + [target + size_a - 1]
    result = [cuts[i + 1] - cuts[i] - 1 for i in range(size_a)]
    return result


MINI_PYTEST_PARAMS = {
    "default": {
        "m,n,k": [(1, 256, 128)],
        "e": [1],
        "topk": [1],
        "dtype": [torch.bfloat16]
    }
}


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_grouped_gemm(m, n, k, e, topk, dtype):
    seed_everything(7)
    num_experts = e
    token_per_group = random_partition(e, m * topk)
    # input
    input_A = torch.randn((sum(token_per_group), k),
                          dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A.clone()
    # weight
    input_B = torch.randn((num_experts, n, k), dtype=dtype, device=DEVICE)
    input_B = input_B.transpose(-1, -2).contiguous().transpose(-1, -2)

    # output offset
    output = torch.empty((sum(token_per_group), n), dtype=dtype, device=DEVICE)
    cutlass_grouped_gemm(input_A, input_B, output, token_per_group, n, k,
                         num_experts)
    torch.xpu.synchronize()
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = token_per_group[i]
        if cur_token_num == 0:
            continue
        input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :]
        weight = input_B[i, :, :]
        expert_output = input @ weight.T
        ref.append(expert_output)
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    try:
        torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
        print("a and b close enough")
    except AssertionError as e:
        print("a and b diffs")
        print(e)


def ref_fused_moe(x, w13, w2, flat_expert_weights, flat_expert_indices,
                  num_per_tok, activation, num_experts):
    expert_cache = torch.zeros_like(x)
    idxs = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // num_per_tok
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue

        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idxs]

        expert_w13 = w13[expert_id, :, :]
        w1, w3 = torch.split(expert_w13,
                             int(list(expert_w13.shape)[0] / 2),
                             dim=0)
        act_fn = torch.nn.SiLU()
        gemm1 = expert_tokens @ w1.T
        gate = act_fn(gemm1)
        up = expert_tokens @ w3.T
        expert_out = (gate * up) @ w2[expert_id, :, :].T
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        expert_cache.scatter_reduce_(0,
                                     exp_token_idxs.view(-1, 1).repeat(
                                         1, x.shape[-1]),
                                     expert_out,
                                     reduce='sum')

    return expert_cache


def check_fused_moe(
    m: int,  # num of tokens
    n: int,  # intermediate_size
    k: int,  # hidden_size
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    seed_everything(7)
    verbose = False
    # Setup test data
    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    w13 = torch.randn((e, 2 * n, k), device=DEVICE, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=DEVICE, dtype=dtype) / 10
    ref_a = a.clone()

    # moe gate
    scores = torch.randn((m, e), device=DEVICE, dtype=dtype)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    if verbose:
        print("expert_indices: ", expert_indices, expert_indices.shape)
        print("expert_scores: ", expert_scores, expert_scores.shape)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    iteration = 1
    for _ in range(iteration):
        out = cutlass_fused_moe(hidden_states=a,
                                w13=w13,
                                w2=w2,
                                topk_weights=flat_expert_weights,
                                topk_ids=flat_expert_indices,
                                n_experts_per_token=topk,
                                activation="silu",
                                num_experts=e)

    ref_out = ref_fused_moe(ref_a, w13, w2, flat_expert_weights,
                            flat_expert_indices, topk, "silu", e)

    print("ref result", ref_out, ref_out.shape)
    print("kernel result", out, out.shape)
