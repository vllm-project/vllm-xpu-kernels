import pytest
import torch
import torch.profiler
from math import ceil

from vllm_xpu_kernels.scaled_mm_interface import cutlass_basic_gemm, cutlass_scaled_mm

DTYPES = [torch.bfloat16]

# @pytest.mark.parametrize("dtype", DTYPES)
def test_base_gemm(dtype):
    # import ipdb
    # ipdb.set_trace()
    torch.set_default_device("xpu")

    m = 1024
    n = 2048
    k = 4096

    inputA = torch.randn(m, k, dtype=dtype)
    inputB = torch.randn(n, k, dtype=dtype)
    inputC = torch.randn(m, n, dtype=dtype)
    res = torch.empty_like(inputC)

    alpha = 1.0
    beta = 0.1
    ref_D = alpha * (inputA @ inputB.t()) + beta * inputC
    cutlass_B = inputB.transpose(1, 0).contiguous().transpose(1, 0)
    print("cutlassB ", cutlass_B.shape, cutlass_B.stride())
    cutlass_basic_gemm(inputA, cutlass_B, inputC, res, alpha, beta)
    print(res)
    print(ref_D)


def ref_scaled_mm(inputA_fp8, inputB_fp8, scaleA, scaleB, block_size):
    m, k = inputA_fp8.shape
    n = inputB_fp8.shape[0]
    a_scale = scaleA
    b_scale = scaleB
    block_shape_n = block_size[0]
    block_shape_k = block_size[1]
    scale_n = b_scale.shape[0]
    scale_k = b_scale.shape[1]

    a_scale = a_scale.unsqueeze(-1).repeat(1, 1, block_shape_k)
    a_scale = a_scale.reshape(m, scale_k * block_shape_k)
    a_scale = a_scale[:, :k]

    b_scale = (
        b_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    b_scale = b_scale[:n, :k]
    a_full = a_scale.half()
    b_full = b_scale.half()

    A32 = inputA_fp8.to(torch.float16)
    B32 = inputB_fp8.to(torch.float16)

    res = (A32 * a_full) @ (B32 * b_full).T
    return res


def test_scaled_mm():
    torch.set_default_device("xpu")

    m = 8192
    k = 7168
    n = 1536
    block_n, block_k = (16, 16)

    dtype = torch.float8_e4m3fn

    inputA = torch.randn(m, k, dtype=torch.float32).to(dtype)
    inputB = torch.randn(n, k, dtype=torch.float32).to(dtype)
    inputC = torch.randn(m, n, dtype=torch.float32)
    res = torch.empty_like(inputC)

    scale_k = ceil(k / block_k)
    scaleA = torch.rand((m, scale_k), dtype=torch.float16)
    scale_n = ceil(n / block_n)
    scaleB = torch.rand((scale_n, scale_k), dtype=torch.float16)

    out = cutlass_scaled_mm(inputA, inputB, scaleA, scaleB, [block_n, block_k])
    ref = ref_scaled_mm(inputA, inputB, scaleA, scaleB, [block_n, block_k])
    print(out)
    print(ref)

    try:
        torch.testing.assert_close(out.float(), ref.float(), rtol=5e-1, atol=1.5e-1)
        print("a and b are close enough")
    except AssertionError as e:
        print("a and b are different")
        print(e)

def profile_scaled_mm():
    torch.set_default_device("xpu")

    m = 8192
    k = 7168
    n = 1536
    block_n, block_k = (16, 16)

    dtype = torch.float8_e4m3fn

    inputA = torch.randn(m, k, dtype=torch.float32).to(dtype)
    inputB = torch.randn(n, k, dtype=torch.float32).to(dtype)
    inputC = torch.randn(m, n, dtype=torch.float32)
    res = torch.empty_like(inputC)

    scale_k = ceil(k / block_k)
    scaleA = torch.rand((m, scale_k), dtype=torch.float16)
    scale_n = ceil(n / block_n)
    scaleB = torch.rand((scale_n, scale_k), dtype=torch.float16)

    for _ in range(5):
        out = cutlass_scaled_mm(inputA, inputB, scaleA, scaleB, [block_n, block_k])

    with torch.profiler.profile(
        activities=[
            # torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU
        ],
        # record_shapes=True,
        with_stack=False,
        # profile_memory=True,
        with_flops=True,
     ) as prof:
        out = cutlass_scaled_mm(inputA, inputB, scaleA, scaleB, [block_n, block_k])

    print(prof.key_averages().table(
         sort_by="self_xpu_time_total",
         row_limit=-1
     ))

    flops = 0
    flops += 2*m*n*k #mm
    flops += m*k + n*k #dequante
    flops /= 1e9
    mem = 0
    mem += m*k*1 + n*k*1 + m*k*2 + n*k*2 + m*n*4
    print(f"FLOPs: {flops:.3f} G")
    print(f"Memory: {mem / 1e6:.3f} MB")


if __name__ == '__main__':
    # test_base_gemm(DTYPES[0])
    # test_scaled_mm()
    profile_scaled_mm()
