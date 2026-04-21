import pytest
import torch

import tests.register_ops as ops  # noqa: F401

torch.manual_seed(42)

def baseline_moe_swiglu(scatter_tokens, smooth_scale, experts_token_count, experts_token_start, dst_dtype):
    """Pure PyTorch vectorized baseline for MoE SwiGLU + Dynamic Quantization."""
    num_experts = experts_token_count.shape[0]
    hidden_size = scatter_tokens.shape[1] // 2
    device = scatter_tokens.device

    expert_ids = torch.arange(num_experts, device=device).repeat_interleave(experts_token_count)
    token_scales = smooth_scale[expert_ids].float()

    x1 = scatter_tokens[:, :hidden_size].float()
    x2 = scatter_tokens[:, hidden_size:].float()

    x1_silu = torch.nn.functional.silu(x1)
    swiglu = x1_silu * x2
    scaled = swiglu * token_scales

    unquantized_math = scaled

    quant_max = 448.0 if dst_dtype == torch.float8_e4m3fn else 127.0
    max_vals = scaled.abs().max(dim=-1).values
    scale = max_vals / quant_max
    scale = torch.where(scale == 0, torch.tensor(1.0, device=device), scale)

    if dst_dtype == torch.int8:
        quant_tokens = torch.round(scaled / scale.unsqueeze(-1)).to(torch.int8)
    else:
        quant_tokens = (scaled / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        
    per_token_scale = scale

    return quant_tokens, per_token_scale, unquantized_math

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_scattered": [64, 256],
        "hidden_size": [64, 128, 256],
        "num_experts": [64],
    },
}

@pytest.mark.parametrize("num_scattered", [64, 256, 1024, 4096])
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 1024, 4096])
@pytest.mark.parametrize("num_experts", [64, 256])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.float8_e4m3fn])
def test_moe_swiglu_dynamic_quant(num_scattered, hidden_size, num_experts, src_dtype, dst_dtype):
    device = "xpu"

    experts_token_count = torch.zeros(num_experts, dtype=torch.int32, device=device)
    experts_token_start = torch.zeros(num_experts, dtype=torch.int32, device=device)

    tokens_per_expert = num_scattered // num_experts
    experts_token_count[:] = tokens_per_expert
    experts_token_count[-1] = num_scattered - (tokens_per_expert * (num_experts - 1))

    start = 0
    max_token_num = 0
    for i in range(num_experts):
        experts_token_start[i] = start
        start += experts_token_count[i].item()
        if experts_token_count[i].item() > max_token_num:
            max_token_num = experts_token_count[i].item()

    scatter_tokens = torch.randn((num_scattered, hidden_size * 2), dtype=src_dtype, device=device)
    smooth_scale = torch.rand((num_experts, hidden_size), dtype=torch.float32, device=device)

    out_quant_tokens = torch.zeros((num_scattered, hidden_size), dtype=dst_dtype, device=device)
    out_per_scale = torch.zeros(num_scattered, dtype=torch.float32, device=device)

    # 1. Base Accuracy Reference
    ref_quant_tokens, ref_per_scale, ref_unquantized_tokens = baseline_moe_swiglu(
        scatter_tokens, smooth_scale, experts_token_count, experts_token_start, dst_dtype
    )

    out_quant_tokens.zero_()
    out_per_scale.zero_()

    # 2. XPU Kernel Output
    torch.ops._moe_C.moe_swiglu_dynamic_quant(
        scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
        out_quant_tokens, out_per_scale, num_experts, max_token_num
    )
    torch.xpu.synchronize()

    # 3. Accuracy Verifications
    assert out_quant_tokens.shape == ref_quant_tokens.shape, "Output quant shape mismatch!"
    assert out_per_scale.shape == ref_per_scale.shape, "Output scale shape mismatch!"

    torch.testing.assert_close(out_per_scale, ref_per_scale, atol=1e-5, rtol=1e-5)

    if dst_dtype == torch.int8:
        diff = (out_quant_tokens.int() - ref_quant_tokens.int()).abs()
        assert diff.max().item() <= 1, "Quantized values diverge completely!"
        
    custom_dequantized = out_quant_tokens.float() * out_per_scale.unsqueeze(1)
    ref_dequantized = ref_quant_tokens.float() * ref_per_scale.unsqueeze(1)
    
    float_diff = (custom_dequantized - ref_dequantized).abs()
        
    if dst_dtype == torch.int8:
        # INT8 has perfect correlation
        assert float_diff.max().item() < 0.5, f"INT8 dequantized math outputs diverge! Error: {float_diff.max().item()}"
    else:
        # FP8 has massive steps between values at the top of its range (step size between 448 and 416 is 32)
        # An absolute ULP rounding disagreement can manifest as a ~16.0 difference before applying the scale
        max_expected_error = 16.0 * out_per_scale.max().item() + 1e-3
        assert float_diff.max().item() <= max_expected_error, f"FP8 dequantized math outputs diverge too broadly. Max error: {float_diff.max().item()} > Allowed: {max_expected_error}"