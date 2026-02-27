# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import gc
import torch
import torch.nn.functional as F
import triton
import triton.testing
from vllm.platforms import current_platform
from vllm_xpu_kernels.fused_moe_interface import compute_num_tokens_per_block, ceilDiv, implement_zp
from tests.utils import parse_args


DEVICE = "xpu"
FP8_DTYPE = current_platform.fp8_dtype()


def clear_xpu_cache():
    torch.xpu.empty_cache()
    gc.collect()
    torch.xpu.synchronize()


# Test token imblance by sampling topk ids from a Zipf distribution, which is common in MoE models
def gen_zipf_topk_ids(num_tokens, num_experts, topk, alpha=1.2, device=DEVICE):
    probs = torch.arange(1, num_experts + 1, device=device).float().pow(-alpha)
    probs /= probs.sum()
    ids = torch.multinomial(probs, num_tokens * topk, replacement=True)
    return ids.view(num_tokens, topk)


def gen_topk_weights(num_tokens, topk, device=DEVICE, dtype=torch.float16):
    w = torch.rand(num_tokens, topk, device=device, dtype=dtype)
    return w / w.sum(dim=-1, keepdim=True)


def naive_moe(
    hidden_states,   # [T, H]
    w13, w13_bias,
    w2, w2_bias,
    topk_ids, topk_weights,
    activation="silu",
):
    T, H = hidden_states.shape
    topk = topk_ids.shape[1]
    I = w13.shape[1] // 2

    output = torch.zeros_like(hidden_states)

    for t in range(T):
        x = hidden_states[t]
        out = torch.zeros(H, device=x.device, dtype=x.dtype)

        for i in range(topk):
            e = int(topk_ids[t, i])
            w = topk_weights[t, i]

            y = F.linear(x, w13[e], w13_bias[e] if w13_bias is not None else None)
            g, u = y.split(I, dim=0)

            if activation == "silu":
                h = F.silu(g) * u
            elif activation == "gelu":
                h = F.gelu(g) * u
            else:
                raise NotImplementedError

            z = F.linear(h, w2[e], w2_bias[e] if w2_bias is not None else None)
            out += w * z

        output[t] = out

    torch.xpu.synchronize()
    return output


def xpu_fused_moe(hidden_states,
                  w13,
                  w13_scales,
                  w13_bias,
                  w2,
                  w2_scales,
                  w2_bias,
                  topk_weights,
                  topk_ids,
                  n_experts_per_token,
                  activation,
                  num_experts,
                  ep_rank=0,
                  ep_size=1,
                  output=None,
                  is_fp8=False,
                  is_int4=False,
                  is_mxfp4=False):
    '''
    hidden_states: [num_rows, hidden_size]
    w13: [num_experts, 2*inter_size, hidden_size]
    w13_scales:
        None for bf16/fp16
        or [num_experts] for fp8
        or [num_experts, 2*inter_size, hidden_size // group_size] for 4bits
    w13_bias: [num_experts, 2*inter_size] or None
    w2: [num_experts, hidden_size, inter_size]
    w2_scales:
        None for bf16/fp16
        or [num_experts] for fp8
        or [num_experts, hidden_size, inter_size // group_size] for 4bits
    w2_bias: [num_experts, hidden_size] or None
    topk_weights: [num_rows, topk]
    topk_ids: [num_rows, topk]
    n_experts_per_token: int
    activation: str
    num_experts: int
    is_int4: bool
    is_mxfp4: bool
    '''
    if output is None:
        output = torch.empty_like(hidden_states)
    else:
        assert output.shape == hidden_states.shape, \
            "output shape must be the same as hidden_states shape"
    inter_size = list(w13.shape)[-2] // 2

    assert w13.is_contiguous() and w2.is_contiguous()

    # 4bits support [E, N, K]
    # other types [E, K, N]
    if not is_int4 and not is_mxfp4:
        if not hasattr(w13, 'xpu_fused_moe'):
            w13.data = w13.transpose(-1, -2).contiguous()
            w2.data = w2.transpose(-1, -2).contiguous()
            w13.xpu_fused_moe = True
            w13.inter_size = inter_size
        else:
            inter_size = w13.inter_size

    if is_int4 and not hasattr(w13, 'xpu_fused_moe'):
        w13_tmp = torch.empty_like(w13)
        w2_tmp = torch.empty_like(w2)
        for i in range(num_experts):
            w13_tmp[i] = implement_zp(w13[i])
            w2_tmp[i] = implement_zp(w2[i])
        w13_tmp = w13_tmp.contiguous()
        w2_tmp = w2_tmp.contiguous()
        w13.data = w13_tmp
        w2.data = w2_tmp
        w13.xpu_fused_moe = True

    # TODO: will all integrated in Cpp func. Temporary expose before gemm fusion
    num_rows, hidden_size = list(hidden_states.shape)
    num_experts_per_node = num_experts
    experts_per_token = n_experts_per_token
    num_moe_inputs = n_experts_per_token * num_rows
    permuted_elems = num_moe_inputs * hidden_size
    # interbuf_elems = num_moe_inputs * inter_size
    permuted_row_to_unpermuted_row_size = num_moe_inputs * 4
    permuted_token_selected_experts_size = num_moe_inputs * 4
    src_to_dest_map_size = experts_per_token * num_rows * 4
    expert_first_token_offset_size = (num_experts_per_node + 1) * 8
    num_tokens_per_block = compute_num_tokens_per_block(
        num_rows, num_experts_per_node)
    num_blocks_per_seq = ceilDiv(num_rows, num_tokens_per_block)
    blocked_expert_counts_size = num_experts_per_node * num_blocks_per_seq * 4
    blocked_expert_counts_cumsum_size = blocked_expert_counts_size
    blocked_row_to_unpermuted_row_size = num_experts_per_node * num_rows * 4
    permuted_data_size = permuted_elems * 2
    permuted_token_final_scales_size = num_moe_inputs * 4

    ws_map = {}
    map_offset = 0

    def config_ws(name, size):
        nonlocal map_offset
        if size % 256 != 0:
            size += 256 - size % 256
        ws_map[name] = (size, map_offset)
        map_offset += size

    config_ws("permuted_row_to_unpermuted_row",
              permuted_row_to_unpermuted_row_size)
    config_ws("permuted_token_selected_experts",
              permuted_token_selected_experts_size)
    config_ws("unpermuted_row_to_permuted_row", src_to_dest_map_size)
    config_ws("blocked_expert_counts", blocked_expert_counts_size)
    config_ws("blocked_expert_counts_cumsum",
              blocked_expert_counts_cumsum_size)
    config_ws("blocked_row_to_unpermuted_row",
              blocked_row_to_unpermuted_row_size)
    config_ws("expert_first_token_offset", expert_first_token_offset_size)
    config_ws("permuted_token_final_scales", permuted_token_final_scales_size)
    config_ws("overlapped_gemm1_gemm2_inputs", permuted_data_size)

    workspace = torch.zeros(map_offset,
                            dtype=torch.uint8,
                            device=hidden_states.device)
    if topk_ids.dtype == torch.int32:
        topk_ids = topk_ids.to(torch.int64)
    torch.ops._moe_C.fused_moe_prologue(
        input=hidden_states,
        input_scales=None,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        workspace=workspace,
        hidden_size=hidden_size,
        inter_size=inter_size,
        block_k=1,
        ep_rank=ep_rank,
        ep_size=ep_size,
        num_experts_on_rank=num_experts_per_node)

    expert_first_token_offset = workspace[
        ws_map["expert_first_token_offset"][1]:
        ws_map["expert_first_token_offset"][1] +
        expert_first_token_offset_size].view(torch.int64)
    permuted_row_to_unpermuted_row = workspace[
        ws_map["permuted_row_to_unpermuted_row"][1]:
        ws_map["permuted_row_to_unpermuted_row"][1] +
        permuted_row_to_unpermuted_row_size].view(torch.int32)
    unpermuted_row_to_permuted_row = workspace[
        ws_map["unpermuted_row_to_permuted_row"][1]:
        ws_map["unpermuted_row_to_permuted_row"][1] +
        src_to_dest_map_size].view(torch.int32)
    gemm1_input = workspace[ws_map["overlapped_gemm1_gemm2_inputs"][1]:
                            ws_map["overlapped_gemm1_gemm2_inputs"][1] +
                            permuted_data_size].view(hidden_states.dtype).view(
                                num_moe_inputs, hidden_size)
    gemm1_output = torch.empty((num_moe_inputs, 2 * inter_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    if not is_fp8 and not is_int4 and not is_mxfp4:
        gemm1_scales = None
        gemm2_scales = None
    else:
        gemm1_scales = w13_scales
        gemm2_scales = w2_scales

    ########### gemm1 ##################
    input_B = w13

    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=gemm1_input,
        ptr_B=input_B,
        ptr_scales=gemm1_scales,
        ptr_bias=w13_bias,
        ptr_D=gemm1_output,
        expert_first_token_offset=expert_first_token_offset,
        N=2 * inter_size,
        K=hidden_size,
        num_experts=num_experts_per_node,
        is_B_int4=is_int4,
        is_B_mxfp4=is_mxfp4)

    # act
    act_output = torch.empty((num_moe_inputs, inter_size),
                             dtype=gemm1_output.dtype,
                             device=gemm1_output.device)
    if activation == "silu":
        torch.ops._C.silu_and_mul(act_output, gemm1_output)
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(act_output, gemm1_output)
    elif activation == "swigluoai":
        torch.ops._C.swigluoai_and_mul(act_output, gemm1_output, 1.702, 7.0)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

    ########### gemm2 ##################
    input_A = act_output.contiguous()
    input_B = w2
    gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_scales=gemm2_scales,
        ptr_bias=w2_bias,
        ptr_D=gemm2_output,
        expert_first_token_offset=expert_first_token_offset,
        N=hidden_size,
        K=inter_size,
        num_experts=num_experts_per_node,
        is_B_int4=is_int4,
        is_B_mxfp4=is_mxfp4)

    torch.ops._moe_C.moe_gather(output, gemm2_output, topk_weights,
                                permuted_row_to_unpermuted_row,
                                unpermuted_row_to_permuted_row,
                                expert_first_token_offset,
                                num_experts_per_node)

    torch.xpu.synchronize()
    return output


def fused_moe_xpu(
    hidden_states,
    w13, w13_scales, w13_bias,
    w2, w2_scales, w2_bias,
    topk_ids, topk_weights,
    activation,
    num_experts,
):
    return xpu_fused_moe(
        hidden_states=hidden_states,
        w13=w13,
        w13_scales=w13_scales,
        w13_bias=w13_bias,
        w2=w2,
        w2_scales=w2_scales,
        w2_bias=w2_bias,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        n_experts_per_token=topk_ids.shape[1],
        activation=activation,
        num_experts=num_experts,
        is_fp8=False,
        is_int4=False,
        is_mxfp4=False,
    )


def calculate_diff(config):
    m, num_experts, topk, hidden_size, dtype = config
    activation = "silu"
    torch.manual_seed(0)
    inter_size = 4 * hidden_size  # realistic MoE ratio

    hidden_states = torch.randn(
        (m, hidden_size), device=DEVICE, dtype=dtype
    )

    topk_ids = gen_zipf_topk_ids(
        m, num_experts, topk, device=DEVICE
    )
    topk_weights = gen_topk_weights(
        m, topk, device=DEVICE, dtype=torch.float32
    )

    w13 = torch.randn(
        num_experts, 2 * inter_size, hidden_size,
        device=DEVICE, dtype=dtype
    )
    w2 = torch.randn(
        num_experts, hidden_size, inter_size,
        device=DEVICE, dtype=dtype
    )

    w13_bias = torch.randn(
        num_experts, 2 * inter_size,
        device=DEVICE, dtype=dtype
    )
    w2_bias = torch.randn(
        num_experts, hidden_size,
        device=DEVICE, dtype=dtype
    )
    output_naive = naive_moe(
        hidden_states,
        w13, w13_bias,
        w2, w2_bias,
        topk_ids, topk_weights,
        activation,
    )
    output_vllm = fused_moe_xpu(
        hidden_states,
        w13, None, w13_bias,
        w2, None, w2_bias,
        topk_ids, topk_weights,
        activation,
        num_experts,
    )

    if torch.allclose(output_naive, output_vllm, atol=2e-2, rtol=0):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ, ", config)


def get_benchmark(configs, activation="silu"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "num_experts", "topk", "hidden_size", "dtype"],
            x_vals=[tuple(c) for c in configs],
            line_arg="provider",
            line_vals=["naive", "fused"],
            line_names=["Naive", "Cutlass Fused"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="Latency (us)",
            plot_name="moe-cutlass-vs-naive",
            args={},
        )
    )
    def benchmark(m, num_experts, topk, hidden_size, dtype, provider):
        torch.manual_seed(0)

        inter_size = 4 * hidden_size  # realistic MoE ratio

        hidden_states = torch.randn(
            (m, hidden_size), device=DEVICE, dtype=dtype
        )

        topk_ids = gen_zipf_topk_ids(
            m, num_experts, topk, device=DEVICE
        )
        topk_weights = gen_topk_weights(
            m, topk, device=DEVICE, dtype=dtype
        )

        w13 = torch.randn(
            num_experts, 2 * inter_size, hidden_size,
            device=DEVICE, dtype=dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, inter_size,
            device=DEVICE, dtype=dtype
        )

        w13_bias = torch.randn(
            num_experts, 2 * inter_size,
            device=DEVICE, dtype=dtype
        )
        w2_bias = torch.randn(
            num_experts, hidden_size,
            device=DEVICE, dtype=dtype
        )

        quantiles = [0.5, 0.2, 0.8]

        if provider == "naive":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: naive_moe(
                    hidden_states,
                    w13, w13_bias,
                    w2, w2_bias,
                    topk_ids, topk_weights,
                    activation,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_moe_xpu(
                    hidden_states,
                    w13, None, w13_bias,
                    w2, None, w2_bias,
                    topk_ids, topk_weights,
                    activation,
                    num_experts,
                ),
                quantiles=quantiles,
            )
        clear_xpu_cache()
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)

    m = [32, 64, 256, 512, 1024]
    experts = [8, 16, 32, 64]
    topk = [1, 2, 4]
    hidden_size = [4096, 8192]
    dtype = [torch.float16, torch.bfloat16]     # float32 not supported by cutlass kernels: permuted_data_size = permuted_elems * 2
    print("Final configuration:")
    print(f"  m: {m}")
    print(f"  num_experts: {experts}")
    print(f"  topk: {topk}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  dtype: {dtype}")

    configs = list(
        itertools.product(m, experts, topk, hidden_size, dtype))

    for config in configs:
        try:
            calculate_diff(config)
        except RuntimeError as e:
            clear_xpu_cache()
            print(f"Error in config {config}: {e}")

    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
