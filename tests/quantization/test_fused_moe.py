import pytest
import torch
from math import ceil
from typing import Callable, Optional, Union
from vllm_xpu_kernels.fused_moe_interface import cutlass_fused_moe, cutlass_grouped_gemm

NUM_EXPERTS = [8, 64, 192]
EP_SIZE = [1, 4]
TOP_KS = [2, 6]

FUSED_MOE_MNK_FACTORS = [
    (1, 128, 128),
    (1, 2048, 128),
    (33, 2048, 128),
    (222, 1024, 1024),
    (32768, 128, 128),
    (32768, 2048, 511),
    (40000, 1024, 1024),
]

FUSED_MOE_WN16_MNK_FACTORS = [
    (1, 128, 128),
    (1, 1024, 1024),
    (32, 2048, 128),
    (32, 1024, 1024),
    (222, 2048, 1024),
]

DEVICE = "xpu"

def test_grouped_gemm(num_experts, n, k, token_per_group):
    # input
    input_A = torch.randn((sum(token_per_group), k), dtype=torch.bfloat16, device="xpu")

    # weight
    input_B = torch.randn((num_experts, n, k), dtype=torch.bfloat16, device="xpu")
    input_B = input_B.transpose(-1, -2).contiguous().transpose(-1, -2)

    # output offset
    output = torch.empty((sum(token_per_group), n), dtype=torch.float32, device="xpu")
    offset = torch.tensor(token_per_group, dtype=torch.int64, device="cpu" )

    cutlass_grouped_gemm(input_A, input_B, output, offset, n, k, num_experts)

    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = token_per_group[i]
        if cur_token_num == 0:
            continue
        input = input_A[pre_token_sum:pre_token_sum + cur_token_num, :]
        weight = input_B[i, :, :]
        expert_output = input @ weight.T
        ref.append(expert_output)
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0).float()

    print(torch.allclose(output, ref, rtol=1, atol=1))
    max_diff = (output - ref).abs().max()
    print("Max absolute difference:", max_diff)

# @pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
# @pytest.mark.parametrize("e", NUM_EXPERTS)
# @pytest.mark.parametrize("topk", TOP_KS)
# @pytest.mark.parametrize("ep_size", EP_SIZE)
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_moe(
    m: int, # num of tokens
    n: int, # intermediate_size
    k: int, # hidden_size
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
  ):
    # todo: seed

    # Setup test data
    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    w13 = torch.randn((e, 2 * n, k), device=DEVICE, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=DEVICE, dtype=dtype) / 10

    # moe gate
    scores = torch.randn((m, e), device=DEVICE, dtype=dtype)
    expert_indices, expert_scores = torch.topk(scores, k=topk, dim=-1, sorted=False)
    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    cutlass_fused_moe(hidden_states=a,
                      w13=w13,
                      w2=w2,
                      topk_weights=flat_expert_weights,
                      topk_ids=flat_expert_indices,
                      n_experts_per_token=topk,
                      inplace=True,
                      activation="silu",
                      num_experts=e)

    # print("result", a, a.shape)

if __name__ == "__main__":
    test_fused_moe(
        m = 33,
        n = 2048,
        k = 128,
        e = 16,
        topk = 2,
        ep_size = 1,
        dtype = torch.bfloat16
    )
    # test_grouped_gemm(num_experts=16, n=5120, k=8192, token_per_group=[1,2,6,8,12,0,1,5,1,2,6,8,12,0,1,5])
