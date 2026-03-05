# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import gc
import torch
import torch.nn.functional as F
import triton
import triton.testing
from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe
from tests.utils import parse_args
from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.fused_moe.test_fused_moe import ref_fused_moe


DEVICE = "xpu"
def clear_xpu_cache():
    torch.xpu.empty_cache()
    gc.collect()
    torch.xpu.synchronize()


def make_fused_moe_input(config):
    mnk, e, topk, dtype, w_dtype, has_bias = config
    m, n, k = mnk
    input_len = m
    hidden_size = k
    intermediate_size = n
    num_experts = e

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = torch.randn((num_experts, 2 * intermediate_size, hidden_size),
                      device=DEVICE,
                      dtype=dtype) / 16
    w2 = torch.randn((num_experts, hidden_size, intermediate_size),
                     device=DEVICE,
                     dtype=dtype) / 16
    ref_a = a.clone()

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=DEVICE,
            dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    if w_dtype is not None:
        w13_fp8 = torch.empty_like(w13, dtype=w_dtype)
        w2_fp8 = torch.empty_like(w2, dtype=w_dtype)

        # scale
        random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
        w13_scales = torch.pow(2.0, random_exponents.float())
        random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
        w2_scales = torch.pow(2.0, random_exponents.float())

        for i in range(num_experts):
            w13_fp8[i], _ = scaled_fp8_quant(w13[i],
                                             w13_scales[i].to(torch.float32),
                                             False,
                                             False,
                                             fp8_dtype=w_dtype)
            w2_fp8[i], _ = scaled_fp8_quant(w2[i],
                                            w2_scales[i].to(torch.float32),
                                            False,
                                            False,
                                            fp8_dtype=w_dtype)
        w13 = w13_fp8
        w2 = w2_fp8

        ref_w13 = torch.empty_like(w13_fp8, dtype=dtype)
        ref_w2 = torch.empty_like(w2_fp8, dtype=dtype)
        for i in range(num_experts):
            ref_w13[i] = w13_fp8[i].to(dtype) * w13_scales[i]
            ref_w2[i] = w2_fp8[i].to(dtype) * w2_scales[i]
    else:
        w13_scales = None
        w2_scales = None
        ref_w13 = w13
        ref_w2 = w2

    return (ref_a, ref_w13, w13_bias, ref_w2, w2_bias, flat_expert_weights,
            flat_expert_indices, a, w13, w13_scales, w2, w2_scales,
            expert_scores, expert_indices)


def calculate_diff(config):
    _, e, topk, dtype, w_dtype, _ = config
    ref_a, ref_w13, w13_bias, ref_w2, w2_bias, flat_expert_weights, \
        flat_expert_indices, a, w13, w13_scales, w2, w2_scales, \
            expert_scores, expert_indices = make_fused_moe_input(config)

    ref_out = ref_fused_moe(ref_a, ref_w13, w13_bias, ref_w2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e)

    ref_a, ref_w13, w13_bias, ref_w2, w2_bias, flat_expert_weights, \
        flat_expert_indices, a, w13, w13_scales, w2, w2_scales, \
            expert_scores, expert_indices = make_fused_moe_input(config)
    output = xpu_fused_moe(hidden_states=a,
                           w13=w13,
                           w13_scales=w13_scales,
                           w13_bias=w13_bias,
                           w2=w2,
                           w2_scales=w2_scales,
                           w2_bias=w2_bias,
                           topk_weights=expert_scores,
                           topk_ids=expert_indices,
                           n_experts_per_token=topk,
                           activation="silu",
                           num_experts=e,
                           is_fp8=(w_dtype is not None))
    if dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 2e-2
        atol = 2e-2

    try:
        torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)
        print("✅ All implementations match, ", config)
    except AssertionError as e:
        print("❌ Implementations differ, ", config, " error: ", e)


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "num_experts", "topk", "dtype", "w_dtype", "has_bias"],
            x_vals=[(*tuple(c)[0], *tuple(c)[1:]) for c in configs],
            line_arg="provider",
            line_vals=["native", "vllm"],
            line_names=["native", "vllm"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="Latency (us)",
            plot_name="moe-cutlass-vs-native",
            args={},
        )
    )
    def benchmark(m, n, k, num_experts, topk, dtype, w_dtype, has_bias, provider):
        quantiles = [0.5, 0.2, 0.8]

        print(f"Running config: {(m, n, k, num_experts, topk, dtype, w_dtype, has_bias)}, Provider: {provider}", flush=True)
        if provider == "native":
            ref_a, ref_w13, w13_bias, ref_w2, w2_bias, flat_expert_weights, \
            flat_expert_indices, a, w13, w13_scales, w2, w2_scales, \
                expert_scores, expert_indices = make_fused_moe_input(config=((m, n, k), num_experts, topk, dtype, w_dtype, has_bias))
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: ref_fused_moe(ref_a, ref_w13, w13_bias, ref_w2, w2_bias,
                flat_expert_weights, flat_expert_indices, topk,
                "silu", num_experts),
                quantiles=quantiles,
            )
        else:
            ref_a, ref_w13, w13_bias, ref_w2, w2_bias, flat_expert_weights, \
            flat_expert_indices, a, w13, w13_scales, w2, w2_scales, \
                expert_scores, expert_indices = make_fused_moe_input(config=((m, n, k), num_experts, topk, dtype, w_dtype, has_bias))
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: xpu_fused_moe(hidden_states=a,
                w13=w13,
                w13_scales=w13_scales,
                w13_bias=w13_bias,
                w2=w2,
                w2_scales=w2_scales,
                w2_bias=w2_bias,
                topk_weights=expert_scores,
                topk_ids=expert_indices,
                n_experts_per_token=topk,
                activation="silu",
                num_experts=num_experts,
                is_fp8=(w_dtype is not None)),
                quantiles=quantiles,
            )
        clear_xpu_cache()
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    seed_everything(seed)

    mnk =[
        (1, 5120, 8192),
        (4, 5120, 8192),
        (16, 5120, 8192),
        (8192, 5120, 8192),
    ]
    experts = [16]
    topk = [1]
    dtype = [torch.float16, torch.bfloat16]
    w_dtype = [torch.float8_e5m2, torch.float8_e4m3fn, None]
    has_bias = [True, False]
    print("Final configuration:")
    print(f"  m,n,k: {mnk}")
    print(f"  experts: {experts}")
    print(f"  topk: {topk}")
    print(f"  dtype: {dtype}")
    print(f"  w_dtype: {w_dtype}")
    print(f"  has_bias: {has_bias}")

    configs = list(
        itertools.product(mnk, experts, topk, dtype, w_dtype, has_bias))

    for config in configs:
        try:
            calculate_diff(config)
        except Exception as e:
            print("Error in config: ", config, " error: ", e)
        clear_xpu_cache()

    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
