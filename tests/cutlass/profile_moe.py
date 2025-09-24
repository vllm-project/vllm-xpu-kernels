import torch
import torch.profiler
import intel_extension_for_pytorch
from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm

def linear_silu_mul(x):
    half = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (half,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return torch.ops.torch_ipex.silu_and_mul(x, out)

def test_moe_gemm(n_experts, intermediate_size, hidden_size, tokens, topk, dtype):

    total_m = tokens * topk

    input_a1 = torch.randn(total_m, hidden_size, dtype=dtype, device="xpu") / 10
    input_a2 = input_a1.clone()
    w13 = torch.randn(n_experts, 2*intermediate_size, hidden_size, dtype=dtype, device="xpu") / 10
    w2 = torch.randn(n_experts, hidden_size, intermediate_size, dtype=dtype, device="xpu") / 10
    w13 = w13.transpose(1, 2).contiguous()
    w2 = w2.transpose(1, 2).contiguous()

    offset = [int(total_m//n_experts)] * n_experts
    rows_for_experts = torch.tensor(offset, device="xpu", dtype=torch.int32)
    rows_for_experts_ipex = rows_for_experts.to(torch.int32).to("cpu")

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

        with torch.profiler.record_function("#####ipex_moegemm_1"):
            ipex_o1 = torch.xpu.moe_gemm(input_a1, w13, rows_for_experts, rows_for_experts_ipex, n_experts)
            torch.xpu.synchronize()

        ipex_o2 = linear_silu_mul(ipex_o1)
        torch.xpu.synchronize()

        with torch.profiler.record_function("#####ipex_moegemm_2"):
            ipex_o3 = torch.xpu.moe_gemm(ipex_o2, w2, rows_for_experts, rows_for_experts_ipex, n_experts)
            torch.xpu.synchronize()


        w13 = w13.transpose(1, 2)
        w2 = w2.transpose(1, 2)

        cutlass_o1 = torch.empty((total_m, 2*intermediate_size), dtype=dtype, device="xpu")
        cutlass_o3 = torch.empty((total_m, hidden_size), dtype=dtype, device="xpu")
        with torch.profiler.record_function("@@@@@cutlass_moegemm_1"):
            cutlass_grouped_gemm(input_a2, w13, cutlass_o1, offset, 2*intermediate_size, hidden_size, n_experts)
            torch.xpu.synchronize()
        cutlass_o2 = linear_silu_mul(cutlass_o1)
        torch.xpu.synchronize()

        with torch.profiler.record_function("@@@@@cutlass_moegemm_2"):
            cutlass_grouped_gemm(cutlass_o2, w2, cutlass_o3, offset, hidden_size, intermediate_size, n_experts)
            torch.xpu.synchronize()


    print(prof.key_averages().table(
        sort_by="self_xpu_time_total",  # 以XPU耗时排序
        row_limit=-1
    ))

    print("ipex out: \n", ipex_o3, ipex_o3.shape)
    print("cutlass out: \n", cutlass_o3, cutlass_o3.shape)
    torch.testing.assert_close(ipex_o3.to(float), cutlass_o3.to(float), rtol=0, atol=2e-2)

if __name__ == "__main__":
    test_moe_gemm(n_experts=16, intermediate_size=8192, hidden_size=5120, topk=1, dtype=torch.bfloat16, tokens=16*512)
