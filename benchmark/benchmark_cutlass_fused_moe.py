# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import gc
import torch
import torch.nn.functional as F
import triton
import triton.testing
from vllm.platforms import current_platform
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe
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

    if torch.allclose(output_naive, output_vllm, atol=1e-3, rtol=0):
        print("✅ All implementations match, ", config)
    else:
        print("❌ Implementations differ, ", config)


def get_benchmark():
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
        activation = "silu"

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
    hidden_size = [4096]    # 8192 causes OOM
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
