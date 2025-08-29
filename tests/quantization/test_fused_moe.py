import pytest
import torch
from math import ceil
from typing import Callable, Optional, Union
from vllm_xpu_kernels.fused_moe_interface import cutlass_fused_moe

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

# @pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
# @pytest.mark.parametrize("e", NUM_EXPERTS)
# @pytest.mark.parametrize("topk", TOP_KS)
# @pytest.mark.parametrize("ep_size", EP_SIZE)
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
  ):
    # todo: seed

    #
    # Setup test data
    #

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device=DEVICE, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=DEVICE, dtype=dtype) / 10

    score = torch.randn((m, e), device=DEVICE, dtype=dtype)
    cutlass_fused_moe()

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
