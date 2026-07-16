# SPDX-License-Identifier: Apache-2.0
"""Correctness test for the BF16xFP32 emulated-FP32 GEMM.

The kernel emulates an FP32-precision GEMM (BF16 activation x FP32 weight)
by decomposing the FP32 weight into two BF16 matrices and running a DualGemm:

    out = A @ B_high.T + (A @ B_low.T) * scale

This mirrors the way the Hunyuan-V3 MoE router (GateLinear) is computed on
XPU today.  The result is compared against the reference ``F.linear`` path
that vLLM currently uses on XPU:

        x_fp32 = x_bf16.float()
        out    = F.linear(x_fp32, W_fp32)      # FP32 x FP32 GEMM

Run:
    pytest -q tests/test_gemm_bf16xfp32.py
"""

import pytest
import torch

from tests.register_ops import gemm_bf16xfp32

DEVICE = "xpu"
SCALE = 1.0 / 256.0

NUM_TOKENS = [1, 4, 8, 16, 32, 64, 256, 1024, 4096, 8192]
NUM_EXPERTS = [128, 192, 256]
HIDDEN_SIZES = [3072, 4096, 6144]


def _split_fp32_weight(weight: torch.Tensor, scale: float = SCALE):
    """FP32 weight [N, K] -> (W_high, W_low) BF16 decomposition.

    The kernel computes ``Y[m,n] = sum_k A[m,k] * B[k,n]`` (i.e. F.linear),
    so the components are returned transposed to ``[K, N]`` (N-contiguous),
    matching the layout the kernel expects.

    Reconstruction: weight ~= fp32(W_high.T) + fp32(W_low.T) * scale
    """
    assert weight.dtype == torch.float32
    W_high = weight.to(torch.bfloat16)
    residual = weight - W_high.float()
    W_low = (residual / scale).to(torch.bfloat16)
    return W_high.t().contiguous(), W_low.t().contiguous()


def _make_inputs(m, n, k, seed=1234):
    torch.manual_seed(seed)
    x_bf16 = (torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE) / 8.0)
    w_fp32 = (torch.randn(n, k, dtype=torch.float32, device=DEVICE) / 8.0)
    return x_bf16, w_fp32


def _ref_flinear(x_bf16, w_fp32):
    return torch.nn.functional.linear(x_bf16.float(), w_fp32)


def _ref_bf16_naive(x_bf16, w_fp32):
    w_high = w_fp32.to(torch.bfloat16)
    return torch.nn.functional.linear(x_bf16, w_high).float()


@pytest.mark.parametrize("k", HIDDEN_SIZES)
@pytest.mark.parametrize("n", NUM_EXPERTS)
@pytest.mark.parametrize("m", NUM_TOKENS)
def test_gemm_bf16xfp32_correctness(m, n, k):
    x_bf16, w_fp32 = _make_inputs(m, n, k)

    ref = _ref_flinear(x_bf16, w_fp32)

    w_high, w_low = _split_fp32_weight(w_fp32, SCALE)
    out = gemm_bf16xfp32(x_bf16, w_high, w_low, SCALE)
    torch.xpu.synchronize()

    assert out.dtype == torch.float32
    assert out.shape == ref.shape

    abs_err = (out - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-6)
    max_abs = abs_err.max().item()
    mean_rel = rel_err.mean().item()

    naive = _ref_bf16_naive(x_bf16, w_fp32)
    naive_max_rel = ((naive - ref).abs() / (ref.abs() + 1e-6)).max().item()
    dual_max_rel = rel_err.max().item()

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)

    assert max_abs < 1e-4, f"max abs err {max_abs:.2e} exceeds 1e-4"
    assert mean_rel < 1e-4, f"mean rel err {mean_rel:.2e} exceeds 1e-4"

    assert dual_max_rel <= naive_max_rel + 1e-3, (
        f"dual-bf16 rel err {dual_max_rel:.4e} worse than "
        f"naive bf16 {naive_max_rel:.4e}")

if __name__ == "__main__":
    import itertools
    for m, n, k in itertools.product(NUM_TOKENS, NUM_EXPERTS, HIDDEN_SIZES):
        test_gemm_bf16xfp32_correctness(m, n, k)
    print("\nAll gemm_bf16xfp32 correctness tests passed.")
