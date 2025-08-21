import torch
from torch.nn.modules.utils import _pair
from torch import nn, Tensor
from . import _vllm_fp8_C

def cutlass_basic_gemm(inputA, inputB, inputC, res, alpha=1.0, beta=0.0):
    print("init cutlass_basic_gemm")
    return torch.ops._vllm_fp8_C.cutlass_gemm(inputA, inputB, inputC, res, alpha, beta)


def cutlass_scaled_mm(
    inputA,
    inputB,
    scaleA,
    scaleB,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
):
    m = inputA.shape[0]
    n = inputB.shape[0]
    k = inputA.shape[1]

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

    inputA = inputA.contiguous()
    inputB = inputB.transpose(1, 0).contiguous().transpose(1, 0)
    a_scale = a_scale.contiguous()
    b_scale = b_scale.transpose(1, 0).contiguous().transpose(1, 0)

    a_scale = a_scale.half()
    b_scale = b_scale.half()

    res_float = torch.empty([m, n], dtype=torch.float, device=inputA.device)

    torch.ops._vllm_fp8_C.cutlass_scaled_mm(
        inputA,
        inputB,
        a_scale,
        b_scale,
        res_float,
        1.0,
        0.0,
    )
    res = res_float.to(dtype=output_dtype)
    return res
