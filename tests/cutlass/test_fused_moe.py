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


def calculate_device_mem(m, k, n, e, topk, dtype):
    total = 0
    x = m*k
    w13 = e*2*n*k
    w2 = e*k*n
    topk_w = topk*m
    topk_id = topk*m
    expert_cache = x
    gemm1_out = m*2*n
    gemm2_out = m*k
    total += x + w13 + w2 + topk_w + topk_id + expert_cache + gemm1_out + gemm2_out
    byte_per_data = 4
    if dtype == torch.bfloat16:
        byte_per_data = 2
    total_bytes_G = total * byte_per_data / 1000 / 1000 / 1000
    print("fused moe should take device memory: ", total_bytes_G, "G")


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

def ref_fused_moe(x,
                  w13,
                  w2,
                  flat_expert_weights,
                  flat_expert_indices,
                  num_per_tok,
                  activation,
                  num_experts):

    expert_cache = torch.zeros_like(x).float()
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
        w1, w3 = torch.split(expert_w13, int(list(expert_w13.shape)[0]/2), dim=0)
        act_fn = torch.nn.SiLU()
        gemm1 = expert_tokens @ w1.T
        gate = act_fn(gemm1)
        up = expert_tokens @ w3.T
        expert_out = (gate * up) @ w2[expert_id, :, :].T
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        expert_cache.scatter_reduce_(
            0,
            exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
            expert_out.float(),
            reduce='sum'
        )

    return expert_cache


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
    verbose = False
    # Setup test data
    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    w13 = torch.randn((e, 2 * n, k), device=DEVICE, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=DEVICE, dtype=dtype) / 10

    # moe gate
    scores = torch.randn((m, e), device=DEVICE, dtype=dtype)
    expert_scores, expert_indices = torch.topk(scores, k=topk, dim=-1, sorted=False)

    if verbose:
        print("expert_indices: ", expert_indices, expert_indices.shape)
        print("expert_scores: ", expert_scores, expert_scores.shape)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    out = cutlass_fused_moe(hidden_states=a,
                            w13=w13,
                            w2=w2,
                            topk_weights=flat_expert_weights,
                            topk_ids=flat_expert_indices,
                            n_experts_per_token=topk,
                            activation="silu",
                            num_experts=e)

    ref_out = ref_fused_moe(a,
                  w13,
                  w2,
                  flat_expert_weights,
                  flat_expert_indices,
                  topk,
                  "silu",
                  e)

    print("ref result", ref_out, ref_out.shape)
    print("kernel result", out, out.shape)
    print(torch.allclose(out, ref_out, rtol=1, atol=1))
    max_diff = (out - ref_out).abs().max()
    print("Max absolute difference:", max_diff)



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
