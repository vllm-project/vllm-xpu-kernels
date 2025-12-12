import torch
import random
import torch.profiler
from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm
from vllm_xpu_kernels.fused_moe_interface import cutlass_xe_grouped_gemm

def random_partition(size_a: int, target: int, randomp: bool):
    if randomp:
        cuts = sorted(random.sample(range(target + size_a - 1), size_a - 1))
        cuts = [-1] + cuts + [target + size_a - 1]
        result = [cuts[i + 1] - cuts[i] - 1 for i in range(size_a)]
        return result
    else:
        return [target//size_a] * size_a

def test_moe_gemm(n_experts, intermediate_size, hidden_size, tokens, topk, dtype, gemm1=True):

    iterations = 10

    total_m = tokens * topk

    input_a1 = torch.randn(total_m, hidden_size, dtype=dtype, device="xpu") / 10
    input_a2 = input_a1.clone()
    w13 = torch.randn(n_experts, 2*intermediate_size, hidden_size, dtype=dtype, device="xpu") / 10
    w2 = torch.randn(n_experts, hidden_size, intermediate_size, dtype=dtype, device="xpu") / 10
    w13 = w13.transpose(1, 2).contiguous().transpose(1, 2)
    w2 = w2.transpose(1, 2).contiguous().transpose(1, 2)

    w13_r = torch.randn(n_experts, hidden_size, 2*intermediate_size, dtype=dtype, device="xpu") / 10
    w2_r = torch.randn(n_experts, intermediate_size, hidden_size, dtype=dtype, device="xpu") / 10

    token_per_group = random_partition(n_experts, total_m, True)
    token_per_group_r = torch.tensor(token_per_group, device="xpu", dtype=torch.int32)
    bias = None
    print("distribution:" , token_per_group)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU  # ✅ 使用 XPU
        ],
        # record_shapes=True,
        with_stack=False,
        # profile_memory=True,
        with_flops=True,
    ) as prof:

        ref_o1 = torch.empty((total_m, 2*intermediate_size), dtype=dtype, device="xpu")
        ref_o3 = torch.empty((total_m, hidden_size), dtype=dtype, device="xpu")

        if gemm1:
            for _ in range(iterations):
                cutlass_xe_grouped_gemm(input_a2, w13_r, None, bias, ref_o1, token_per_group_r, 2*intermediate_size, hidden_size, n_experts, False, False)
        else:
            ref_o2 = (ref_o1[:, :intermediate_size])
            for _ in range(iterations):
                cutlass_xe_grouped_gemm(ref_o2, w2_r, None,  bias, ref_o3, token_per_group_r, hidden_size, intermediate_size, n_experts, False, False)


        cutlass_o1 = torch.empty((total_m, 2*intermediate_size), dtype=dtype, device="xpu")
        cutlass_o3 = torch.empty((total_m, hidden_size), dtype=dtype, device="xpu")
        if gemm1:
            for _ in range(iterations):
                cutlass_grouped_gemm(input_a1, w13, bias, cutlass_o1, token_per_group, 2*intermediate_size, hidden_size, n_experts)
        else:
            cutlass_o2 = (cutlass_o1[:, :intermediate_size])
            for _ in range(iterations):
                cutlass_grouped_gemm(cutlass_o2, w2, bias, cutlass_o3, token_per_group, hidden_size, intermediate_size, n_experts)


    print(prof.key_averages().table(
        sort_by="self_xpu_time_total",  # 以XPU耗时排序
        row_limit=-1
    ))

if __name__ == "__main__":
    # prefill
    # test_moe_gemm(n_experts=16, intermediate_size=8192, hidden_size=5120, topk=1, dtype=torch.bfloat16, tokens=8192, gemm1=True)
    # decode
    test_moe_gemm(n_experts=16, intermediate_size=8192, hidden_size=5120, topk=1, dtype=torch.bfloat16, tokens=8, gemm1=True)
