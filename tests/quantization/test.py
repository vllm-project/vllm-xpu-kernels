import pytest
import torch

from vllm_xpu_kernels.scaled_mm_interface import cutlass_basic_gemm, cutlass_scaled_mm

DTYPES = [torch.bfloat16]

# @pytest.mark.parametrize("dtype", DTYPES)
def test_base_gemm(dtype):
    # import ipdb
    # ipdb.set_trace()
    torch.set_default_device("xpu")

    m = 4096
    n = 4096
    k = 4096

    inputA = torch.randn(m, k, dtype=dtype)
    inputB = torch.randn(k, n, dtype=dtype)
    inputC = torch.randn(m, n, dtype=dtype)
    res = torch.empty_like(inputC)

    alpha = 1.0
    beta = 0.1
    ref_D = alpha * (inputA @ inputB) + beta * inputC
    cutlass_basic_gemm(inputA, inputB, inputC, res, alpha, beta)
    print(res)
    print(ref_D)

if __name__ == '__main__':
    test_base_gemm(DTYPES[0])
