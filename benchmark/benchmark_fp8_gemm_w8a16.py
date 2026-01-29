# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch
import triton

from tests.utils import parse_args, STR_DTYPE_TO_TORCH_DTYPE
from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.register_ops import fp8_gemm_w8a16


def fp8_gemm_naive(input, weight, trans_wei):
    if trans_wei:
        output_ref = torch.matmul(input, weight.t())
    else:
        output_ref = torch.matmul(input, weight)
    return output_ref

def fp8_gemm_vllm(input, weight_fp8, trans_wei, scale_wei, is_mbk):
    if is_mbk:
        input = input.transpose(0, 1)
    output_fp8 = fp8_gemm_w8a16(
        input,
        weight_fp8.transpose(0, 1) if trans_wei else weight_fp8,
        scale_wei,
        torch.Tensor(),
    )
    output_fp8 = output_fp8.transpose(0, 1) if is_mbk else output_fp8
    return output_fp8

def data_preparation(dtype, fp8_dtype, trans_wei, batch, m, n, k):
    input = torch.randn([batch, m, k], dtype=dtype,
                        device=torch.device("xpu")) / 10.0
    if trans_wei:
        weight = torch.ones([n, k], dtype=dtype).xpu()
    else:
        weight = torch.ones([k, n], dtype=dtype).xpu()
    scale_wei = (torch.ones(batch) * 4).xpu()
    scale_shape = None
    weight_fp8, _ = scaled_fp8_quant(
        weight,
        scale=scale_wei,
        use_per_token_if_dynamic=False,
        output=None,
        fp8_dtype=fp8_dtype,
        scale_ub=scale_shape,
    )
    return input, weight, weight_fp8, scale_wei


def calculate_diff(config):
    dtype, fp8_dtype, trans_wei, is_mbk, batch, (m, n, k) = config

    input, weight, weight_fp8, scale_wei = data_preparation(
        STR_DTYPE_TO_TORCH_DTYPE[dtype],
        STR_DTYPE_TO_TORCH_DTYPE[fp8_dtype],
        trans_wei, batch, m, n, k
    )

    # onednn fp8 gemm
    output_naive = fp8_gemm_naive(input.clone(), weight, trans_wei)
    output_vllm = fp8_gemm_vllm(input.clone(), weight_fp8, trans_wei, scale_wei, is_mbk)

    #print(f"Naive output={output_naive}")
    #print(f"vLLM output={output_vllm}")

    if torch.allclose(output_naive, output_vllm, atol=5e-2, rtol=5e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["dtype", "fp8_dtype", "trans_wei", "is_mbk", "batch", "m_n_k"],
            x_vals=[tuple(i) for i in configs],
            line_arg="provider",
            line_vals=["native", "vllm"],
            line_names=["Native", "vLLM"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="fp8-gemm-perf",
            args={},
        )
    )
    def benchmark(dtype, fp8_dtype, trans_wei, is_mbk, batch, m_n_k, provider):
        m, n, k = m_n_k
        input, weight, weight_fp8, scale_wei = data_preparation(
            STR_DTYPE_TO_TORCH_DTYPE[dtype],
            STR_DTYPE_TO_TORCH_DTYPE[fp8_dtype],
            trans_wei, batch, m, n, k
        )

        quantiles = [0.5, 0.2, 0.8]

        if provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fp8_gemm_naive(
                    input.clone(),
                    weight, trans_wei
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fp8_gemm_vllm(
                    input.clone(),
                    weight_fp8, trans_wei, scale_wei,
                    is_mbk
                ),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)

    dtype = ["half", "bfloat16"]
    fp8_dtype = ["fp8_e4m3", "fp8_e5m2"]
    trans_wei = [True, False]
    is_mbk = [True, False]
    batch_size = [1]
    MNK_FACTORS = [
        (1, 4096, 1),
        (1, 32, 1024),
        (4, 16, 1024),
        (8, 32, 1024),
        (8, 512, 1024),
    ]
    print("Final configuration:")
    print(f"  dtype: {dtype}")
    print(f"  fp8_dtype: {fp8_dtype}")
    print(f"  transpose_weight: {trans_wei}")
    print(f"  is_mbk: {is_mbk}")
    print(f"  batch_size: {batch_size}")
    print(f"  mnk_factors: {MNK_FACTORS}")

    configs = list(
        itertools.product(dtype, fp8_dtype, trans_wei, is_mbk,
                          batch_size, MNK_FACTORS))
    # fp8_numerical_sanity_check
    for config in configs:
        calculate_diff(config)

    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
