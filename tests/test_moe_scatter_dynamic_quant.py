import pytest
import torch

import tests.register_ops as ops  # noqa: F401

torch.manual_seed(42)

def baseline_moe_scatter(selected_experts, moe_weights, hidden_states, experts_smooth_scale, total_experts, dst_dtype):
    """Pure PyTorch vectorized baseline for MoE Scatter + Dynamic Quantization."""
    num_tokens, topk = selected_experts.shape
    hidden_size = hidden_states.shape[1]
    device = hidden_states.device

    out_tokens_count = torch.zeros(total_experts, dtype=torch.int32, device=device)
    out_token_to_scatter_offset = torch.zeros_like(selected_experts, dtype=torch.int32)

    # 1. Count and offsets
    for t in range(num_tokens):
        for k in range(topk):
            exp = selected_experts[t, k].item()
            out_token_to_scatter_offset[t, k] = out_tokens_count[exp].item()
            out_tokens_count[exp] += 1

    # 2. Prefix sum
    out_tokens_start = torch.zeros_like(out_tokens_count)
    out_tokens_start[1:] = torch.cumsum(out_tokens_count[:-1], dim=0)

    # 3. Target mapping
    flat_experts = selected_experts.flatten().long()
    flat_offsets = out_token_to_scatter_offset.flatten()
    target_idx = out_tokens_start[flat_experts] + flat_offsets

    # 4. Gather & compute
    h_expanded = hidden_states.float().repeat_interleave(topk, dim=0)
    w_expanded = moe_weights.float().flatten().unsqueeze(-1)
    s_extracted = experts_smooth_scale.float()[flat_experts]

    smoothed = h_expanded * s_extracted * w_expanded

    # 5. Quantize
    quant_max = 448.0 if dst_dtype == torch.float8_e4m3fn else 127.0
    max_vals = smoothed.abs().max(dim=-1).values
    scale = max_vals / quant_max
    scale = torch.where(scale == 0, torch.tensor(1.0, device=device), scale)
    
    if dst_dtype == torch.int8:
        quantized = torch.round(smoothed / scale.unsqueeze(-1)).to(torch.int8)
    else:
        quantized = (smoothed / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

    # 6. Scatter outputs
    total_scattered = num_tokens * topk
    scatter_tokens = torch.zeros((total_scattered, hidden_size), dtype=dst_dtype, device=device)
    scatter_tokens[target_idx] = quantized

    per_token_scale = torch.zeros(total_scattered, dtype=torch.float32, device=device)
    per_token_scale[target_idx] = scale

    tokens_offset = torch.zeros(total_scattered, dtype=torch.int32, device=device)
    flat_t = torch.arange(num_tokens, device=device).repeat_interleave(topk).int()
    tokens_offset[target_idx] = flat_t

    unquantized_math = torch.zeros((total_scattered, hidden_size), dtype=torch.float32, device=device)
    unquantized_math[target_idx] = smoothed

    return (out_token_to_scatter_offset, out_tokens_count, out_tokens_start,
            scatter_tokens, per_token_scale, tokens_offset, unquantized_math)


#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [256],
        "hidden_size": [64],
        "topk": [5],
        "num_experts": [64],
    },
}

@pytest.mark.parametrize("num_tokens", [64, 256])
@pytest.mark.parametrize("hidden_size", [64, 256, 1024])
@pytest.mark.parametrize("topk", [5, 8])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("shared_experts_num", [1])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.float8_e4m3fn])
def test_moe_scatter_dynamic_quant(num_tokens, hidden_size, topk, num_experts, shared_experts_num, src_dtype, dst_dtype):
    device = "xpu"
    
    total_experts = num_experts + shared_experts_num
    total_topk = topk + shared_experts_num

    # Base routing for normal experts
    routed_experts = torch.rand((num_tokens, num_experts), device=device).topk(topk, dim=-1).indices.to(torch.int32)
    
    # Append shared experts (they always process every token, usually placed at the end of expert list)
    if shared_experts_num > 0:
        shared_ids = torch.arange(num_experts, total_experts, dtype=torch.int32, device=device).unsqueeze(0).expand(num_tokens, shared_experts_num)
        selected_experts = torch.cat([routed_experts, shared_ids], dim=-1)
    else:
        selected_experts = routed_experts

    moe_weights = torch.rand((num_tokens, total_topk), dtype=torch.float32, device=device)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=src_dtype, device=device)
    experts_smooth_scale = torch.rand((total_experts, hidden_size), dtype=torch.float32, device=device)

    out_t_offset = torch.zeros_like(selected_experts)
    out_t_count = torch.zeros(total_experts, dtype=torch.int32, device=device)
    out_t_start = torch.zeros(total_experts, dtype=torch.int32, device=device)
    out_scatter_tokens = torch.zeros((num_tokens * total_topk, hidden_size), dtype=dst_dtype, device=device)
    out_per_scale = torch.zeros(num_tokens * total_topk, dtype=torch.float32, device=device)
    out_tokens_offset = torch.zeros(num_tokens * total_topk, dtype=torch.int32, device=device)

    # 1. Base Accuracy Reference
    (ref_t_offset, ref_t_count, ref_t_start, ref_scatter_tokens,
     ref_per_scale, ref_tokens_offset, ref_unquantized_tokens) = baseline_moe_scatter(
         selected_experts, moe_weights, hidden_states, experts_smooth_scale, total_experts, dst_dtype
    )

    # 2. XPU Kernel Output
    torch.ops._moe_C.moe_scatter_dynamic_quant(
        selected_experts, moe_weights, out_t_offset, out_t_count, out_t_start,
        hidden_states, experts_smooth_scale, out_scatter_tokens, out_per_scale,
        out_tokens_offset, shared_experts_num
    )
    torch.xpu.synchronize()

    # 3. Accuracy Verifications
    torch.testing.assert_close(out_t_count, ref_t_count)
    torch.testing.assert_close(out_t_start, ref_t_start)

    expert_ids = torch.zeros(num_tokens * total_topk, dtype=torch.int64, device=device)
    for i in range(total_experts):
        start, count = ref_t_start[i].item(), ref_t_count[i].item()
        if count > 0:
            expert_ids[start:start+count] = i

    sort_key_custom = (expert_ids * num_tokens) + out_tokens_offset
    sort_key_ref = (expert_ids * num_tokens) + ref_tokens_offset

    sort_idx_custom = torch.argsort(sort_key_custom)
    sort_idx_ref = torch.argsort(sort_key_ref)

    torch.testing.assert_close(out_tokens_offset[sort_idx_custom], ref_tokens_offset[sort_idx_ref])
    torch.testing.assert_close(out_per_scale[sort_idx_custom], ref_per_scale[sort_idx_ref], atol=1e-5, rtol=1e-5)

    if dst_dtype == torch.int8:
        diff = (out_scatter_tokens[sort_idx_custom].int() - ref_scatter_tokens[sort_idx_ref].int()).abs()
        assert diff.max().item() <= 1, "Quantized values diverge completely!"

    custom_dequantized = out_scatter_tokens[sort_idx_custom].float() * out_per_scale[sort_idx_custom].unsqueeze(1)
    ref_dequantized = ref_scatter_tokens[sort_idx_ref].float() * ref_per_scale[sort_idx_ref].unsqueeze(1)
    
    float_diff = (custom_dequantized - ref_dequantized).abs()
        
    if dst_dtype == torch.int8:
        assert float_diff.max().item() < 0.5, f"INT8 dequantized math outputs diverge! Error: {float_diff.max().item()}"
    else:
        max_expected_error = 16.0 * out_per_scale.max().item() + 1e-3
        assert float_diff.max().item() <= max_expected_error, f"FP8 dequantized math outputs diverge too broadly. Max error: {float_diff.max().item()} > Allowed: {max_expected_error}"
