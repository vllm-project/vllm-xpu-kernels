# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

# isort: off
import argparse
import gc
import itertools

import torch
import triton
import triton.testing

from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.ops.mx_utils import (
    FP4_EBITS,
    FP4_MBITS,
    _floatx_unpacked_to_f32,
    from_blocked_format,
    to_mxfp,
    unpack_uint4,
)
from tests.register_ops import fp4_gemm, fp8_gemm, fp8_gemm_w8a16
from tests.utils import seed_everything
# isort: on

ALL_BENCHMARKS = [
    "bf16",
    "fp8",
    "fp8_w8a16",
    "fp8_per_channel",
    "mxfp8",
    "mxfp4",
    "model_shapes",
]

DEVICE = "xpu"


def clear_xpu_cache():
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    gc.collect()


def calculate_flops(m, n, k):
    return 2 * m * n * k


def calculate_memory_bytes(m, n, k, x_dtype, w_dtype=None):
    """Calculate total memory read/written in bytes."""
    x_elem = torch.tensor([], dtype=x_dtype).element_size()
    w_elem = torch.tensor(
        [], dtype=x_dtype if w_dtype is None else w_dtype
    ).element_size()
    out_elem = torch.tensor([], dtype=x_dtype).element_size()
    # input + weight + output
    return m * k * x_elem + k * n * w_elem + m * n * out_elem


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

# Generic MNK shapes for benchmarking
GENERIC_MNK = [
    (512, 1024, 2048),
    (1024, 2048, 2048),
    (4096, 4096, 4096),
    (6144, 12288, 4096),
]

FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
OUT_DTYPES = [torch.float16, torch.bfloat16]

# KPI model weight shapes: {model/TP: [(K, N), ...]}
# All shapes are pre-resolved with TP/EP division applied.
KPI_WEIGHT_SHAPES = {
    # llama-3-70b TP=1
    "llama-3-70b/TP1": [
        (8192, 10240),  # qkv
        (8192, 8192),  # out
        (8192, 57344),  # gate_up
        (28672, 8192),  # down
    ],
    # llama-3-70b TP=2
    "llama-3-70b/TP2": [
        (8192, 5120),  # qkv
        (4096, 8192),  # out
        (8192, 28672),  # gate_up
        (14336, 8192),  # down
    ],
    # Llama 4 Maverick/Scout TP=8
    "llama-4-maverick/TP8": [
        (5120, 896),  # qkv
        (640, 5120),  # out
        (5120, 4096),  # gate_up
        (2048, 5120),  # down
    ],
    # LLama3 8B TP=1
    "llama-3-8b/TP1": [
        (4096, 6144),  # qkv
        (4096, 4096),  # out
        (4096, 28672),  # gate_up
        (14336, 4096),  # down
    ],
    # gpt-oss-120b TP=2 EP=1
    "gpt-oss-120b/TP2": [
        (2880, 2560),  # qkv
        (2048, 2880),  # out
        (2880, 5760),  # gate_up
        (2880, 2880),  # down
    ],
    # gpt-oss-120b TP=4 EP=1
    "gpt-oss-120b/TP4": [
        (2880, 1280),  # qkv
        (1024, 2880),  # out
        (2880, 5760),  # gate_up
        (2880, 2880),  # down
    ],
    # gpt-oss-120b TP=8 EP=1
    "gpt-oss-120b/TP8": [
        (2880, 640),  # qkv
        (512, 2880),  # out
        (2880, 5760),  # gate_up
        (2880, 2880),  # down
    ],
    # Qwen 235B TP=2 EP=1
    "qwen-235b/TP2": [
        (4096, 4608),  # qkv
        (4096, 4096),  # out
        (4096, 3072),  # gate_up
        (1536, 4096),  # down
    ],
    # Qwen 235B TP=4 EP=1
    "qwen-235b/TP4": [
        (4096, 2304),  # qkv
        (2048, 4096),  # out
        (4096, 3072),  # gate_up
        (1536, 4096),  # down
    ],
    # Qwen 235B TP=8 EP=1
    "qwen-235b/TP8": [
        (4096, 1152),  # qkv
        (1024, 4096),  # out
        (4096, 3072),  # gate_up
        (1536, 4096),  # down
    ],
    # DeepSeek R1(671B) TP=4 EP=1
    "deepseek-r1/TP4": [
        (7168, 1536),  # qkv
        (1536, 6144),  # qkv
        (7168, 576),  # qkv
        (512, 8192),  # qkv
        (4096, 7168),  # out
        (7168, 9216),  # gate_up
        (7168, 4096),  # gate_up (EP=1)
        (7168, 36864),  # gate_up (shared)
        (4608, 7168),  # down
        (2048, 7168),  # down (EP=1)
        (18432, 7168),  # down (shared)
    ],
    # DeepSeek R1(671B) TP=8 EP=1
    "deepseek-r1/TP8": [
        (7168, 1536),  # qkv
        (1536, 3072),  # qkv
        (7168, 576),  # qkv
        (512, 4096),  # qkv
        (2048, 7168),  # out
        (7168, 4608),  # gate_up
        (7168, 4096),  # gate_up (EP=1)
        (7168, 36864),  # gate_up (shared)
        (2304, 7168),  # down
        (2048, 7168),  # down (EP=1)
        (18432, 7168),  # down (shared)
    ],
    # DeepSeek R1(671B) TP=16 EP=1
    "deepseek-r1/TP16": [
        (7168, 1536),  # qkv
        (1536, 1536),  # qkv
        (7168, 576),  # qkv
        (512, 2048),  # qkv
        (1024, 7168),  # out
        (7168, 2304),  # gate_up
        (7168, 4096),  # gate_up (EP=1)
        (7168, 36864),  # gate_up (shared)
        (1152, 7168),  # down
        (2048, 7168),  # down (EP=1)
        (18432, 7168),  # down (shared)
    ],
}


def gen_fp8_gemm_perf_configs():
    """Generate configs for fp8_gemm (w8a8) per-tensor benchmark."""
    configs = []
    for mnk, out_dtype, fp8_dtype in itertools.product(
        GENERIC_MNK, OUT_DTYPES, FP8_DTYPES
    ):
        configs.append((mnk, out_dtype, fp8_dtype))
    return configs


def gen_fp8_gemm_w8a16_perf_configs():
    """Generate configs for fp8_gemm_w8a16 benchmark."""
    configs = []
    for mnk, out_dtype, fp8_dtype in itertools.product(
        GENERIC_MNK, OUT_DTYPES, FP8_DTYPES
    ):
        configs.append((mnk, out_dtype, fp8_dtype))
    return configs


def gen_fp8_gemm_per_channel_perf_configs():
    """Generate configs for fp8_gemm per-channel benchmark."""
    configs = []
    for mnk, out_dtype, fp8_dtype in itertools.product(
        GENERIC_MNK, OUT_DTYPES, [torch.float8_e4m3fn]
    ):
        configs.append((mnk, out_dtype, fp8_dtype))
    return configs


def gen_mxfp8_gemm_perf_configs():
    """Generate configs for mxfp8 gemm benchmark."""
    configs = []
    for mnk, out_dtype in itertools.product(GENERIC_MNK, OUT_DTYPES):
        configs.append((mnk, out_dtype))
    return configs


def gen_bf16_gemm_perf_configs():
    """Generate configs for bf16 gemm benchmark."""
    configs = []
    for mnk, dtype in itertools.product(GENERIC_MNK, OUT_DTYPES):
        configs.append((mnk, dtype))
    return configs


def gen_mxfp4_gemm_perf_configs():
    """Generate configs for mxfp4 gemm benchmark."""
    configs = []
    for mnk, out_dtype in itertools.product(GENERIC_MNK, OUT_DTYPES):
        configs.append((mnk, out_dtype))
    return configs


def gen_weight_shape_configs(dtype_kind="fp8"):
    """Generate configs from KPI model weight shapes.

    Args:
        dtype_kind: "bf16" for bf16 configs, "fp8" for fp8 per-tensor configs,
                    "fp8_w8a16" for w8a16, "fp8_per_channel" for per-channel.
    """
    configs = []
    m_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        384,
        512,
        640,
        768,
        896,
        1024,
        4096,
    ]
    for model_name, shapes in KPI_WEIGHT_SHAPES.items():
        for (k, n), m in itertools.product(shapes, m_sizes):
            if dtype_kind == "bf16":
                for out_dtype in OUT_DTYPES:
                    configs.append(((m, n, k), out_dtype))
            elif dtype_kind == "fp8" or dtype_kind == "fp8_w8a16":
                for out_dtype, fp8_dtype in itertools.product(
                    OUT_DTYPES, FP8_DTYPES
                ):
                    configs.append(((m, n, k), out_dtype, fp8_dtype))
            elif dtype_kind == "fp8_per_channel":
                for out_dtype in OUT_DTYPES:
                    configs.append(((m, n, k), out_dtype, torch.float8_e4m3fn))
    return configs


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def check_fp8_gemm_per_tensor(config):
    mnk, out_dtype, fp8_dtype = config
    m, n, k = mnk

    input = torch.randn([m, k], dtype=out_dtype, device=DEVICE) / 10.0
    weight = torch.randn([n, k], dtype=out_dtype, device=DEVICE) / 10.0

    scale_src = torch.tensor(4.0, device=DEVICE)
    scale_wei = torch.tensor(4.0, device=DEVICE)

    input_fp8, _ = scaled_fp8_quant(input, scale_src, fp8_dtype=fp8_dtype)
    weight_fp8, _ = scaled_fp8_quant(weight, scale_wei, fp8_dtype=fp8_dtype)

    input_hp = input_fp8.to(out_dtype) * scale_src.to(out_dtype)
    weight_hp = weight_fp8.to(out_dtype) * scale_wei.to(out_dtype)
    output_ref = torch.matmul(input_hp, weight_hp.t())

    output = fp8_gemm(
        input_fp8,
        weight_fp8.transpose(0, 1),
        out_dtype,
        scale_src,
        scale_wei,
    )

    try:
        torch.testing.assert_close(output, output_ref, atol=6e-2, rtol=6e-2)
        print("✅ fp8_gemm per-tensor passed:", config)
    except AssertionError as e:
        print("❌ fp8_gemm per-tensor failed:", config, "error:", e)


def check_fp8_gemm_w8a16(config):
    mnk, out_dtype, fp8_dtype = config
    m, n, k = mnk

    input = torch.randn([m, k], dtype=out_dtype, device=DEVICE) / 10.0
    weight = torch.ones([n, k], dtype=out_dtype, device=DEVICE)
    scale_wei = torch.tensor(4.0, device=DEVICE)

    weight_fp8, _ = scaled_fp8_quant(
        weight, scale_wei, fp8_dtype=fp8_dtype, group_shape=(-1, 1)
    )
    weight_hp = weight_fp8.to(out_dtype) * scale_wei.to(out_dtype)
    output_ref = torch.matmul(input, weight_hp.t())

    output = fp8_gemm_w8a16(
        input,
        weight_fp8.transpose(0, 1),
        scale_wei,
    )

    try:
        torch.testing.assert_close(output, output_ref, atol=5e-2, rtol=5e-2)
        print("✅ fp8_gemm_w8a16 passed:", config)
    except AssertionError as e:
        print("❌ fp8_gemm_w8a16 failed:", config, "error:", e)


def check_fp8_gemm_per_channel(config):
    mnk, out_dtype, fp8_dtype = config
    m, n, k = mnk

    input = torch.randn([m, k], dtype=out_dtype, device=DEVICE) / 10.0
    weight = torch.randn([n, k], dtype=out_dtype, device=DEVICE) / 10.0

    input_fp8, scale_src = scaled_fp8_quant(
        input, use_per_token_if_dynamic=True, fp8_dtype=fp8_dtype
    )
    weight_fp8, scale_wei = scaled_fp8_quant(
        weight, use_per_token_if_dynamic=True, fp8_dtype=fp8_dtype
    )

    input_hp = input_fp8.to(out_dtype) * scale_src.to(out_dtype)
    weight_hp = weight_fp8.to(out_dtype) * scale_wei.to(out_dtype)
    output_ref = torch.matmul(input_hp, weight_hp.t())

    output = fp8_gemm(
        input_fp8,
        weight_fp8.transpose(0, 1),
        out_dtype,
        scale_src,
        scale_wei,
    )

    try:
        torch.testing.assert_close(output, output_ref, atol=6e-2, rtol=6e-2)
        print("✅ fp8_gemm per-channel passed:", config)
    except AssertionError as e:
        print("❌ fp8_gemm per-channel failed:", config, "error:", e)


def check_mxfp8_gemm(config):
    mnk, out_dtype = config
    m, n, k = mnk

    inputs = torch.randn((m, k), dtype=out_dtype, device=DEVICE) * 0.01
    weights = torch.randn((n, k), dtype=out_dtype, device=DEVICE) * 0.01

    if out_dtype == torch.half:
        inputs = inputs.to(torch.float32)
        weights = weights.to(torch.float32)

    inputs_hp, inputs_lp, inputs_scale = _convert_to_mxfp8(inputs)
    weights_hp, weights_lp, weights_scale = _convert_to_mxfp8(weights)

    output = fp8_gemm(
        inputs_lp,
        weights_lp.transpose(0, 1),
        out_dtype,
        inputs_scale,
        weights_scale,
    )
    output_ref = torch.matmul(
        inputs_hp.to(out_dtype), weights_hp.to(out_dtype).t()
    )

    try:
        torch.testing.assert_close(output, output_ref, atol=5e-2, rtol=5e-2)
        print("✅ mxfp8_gemm passed:", config)
    except AssertionError as e:
        print("❌ mxfp8_gemm failed:", config, "error:", e)


def check_mxfp4_gemm(config):
    mnk, out_dtype = config
    m, n, k = mnk

    inputs = torch.randn((m, k), dtype=out_dtype, device=DEVICE) * 0.01
    weights = torch.randn((n, k), dtype=out_dtype, device=DEVICE) * 0.01

    if out_dtype == torch.half:
        inputs = inputs.to(torch.float32)
        weights = weights.to(torch.float32)

    inputs_hp, inputs_lp, inputs_scale = _convert_to_mxfp4(inputs)
    weights_hp, weights_lp, weights_scale = _convert_to_mxfp4(weights)

    output = fp4_gemm(
        inputs_lp,
        weights_lp.transpose(0, 1),
        inputs_scale,
        weights_scale,
        out_dtype,
    )
    output_ref = torch.matmul(
        inputs_hp.to(out_dtype), weights_hp.to(out_dtype).t()
    )

    try:
        torch.testing.assert_close(
            output.to(torch.float),
            output_ref.to(torch.float),
            atol=5e-2,
            rtol=5e-2,
        )
        print("✅ mxfp4_gemm passed:", config)
    except AssertionError as e:
        print("❌ mxfp4_gemm failed:", config, "error:", e)


# ---------------------------------------------------------------------------
# Benchmark: bf16 gemm (F.linear baseline)
# ---------------------------------------------------------------------------


def get_bf16_gemm_benchmark(configs, iterations):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "dtype"],
            x_vals=[(*c[0], c[1]) for c in configs],
            line_arg="provider",
            line_vals=["latency_us", "tflops", "bandwidth_GBs"],
            line_names=["Latency (us)", "TFLOPS", "Bandwidth (GB/s)"],
            styles=[("blue", "-"), ("green", "--"), ("orange", ":")],
            ylabel="value",
            plot_name="bf16_gemm",
            args={},
        )
    )
    def benchmark(m, n, k, dtype, provider, iterations=iterations):
        total_latency = 0.0
        assert iterations > 5

        input = torch.randn([m, k], dtype=dtype, device=DEVICE) / 10.0
        weight = torch.randn([n, k], dtype=dtype, device=DEVICE) / 10.0

        start_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        end_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        for index in range(iterations):
            if index >= 5:
                start_event[index - 5].record()
            torch.nn.functional.linear(input, weight)
            if index >= 5:
                end_event[index - 5].record()
        torch.xpu.synchronize()
        total_latency = sum(
            start_event[i].elapsed_time(end_event[i])
            for i in range(iterations - 5)
        )
        ms = total_latency / (iterations - 5)
        clear_xpu_cache()

        if provider == "tflops":
            flops = calculate_flops(m, n, k)
            return flops / (ms / 1000) / 1e12
        elif provider == "bandwidth_GBs":
            mem_bytes = calculate_memory_bytes(m, n, k, dtype)
            return mem_bytes / (ms / 1000) / 1e9
        else:
            return 1000 * ms

    return benchmark


# ---------------------------------------------------------------------------
# Benchmark: fp8_gemm per-tensor (w8a8)
# ---------------------------------------------------------------------------


def get_fp8_gemm_benchmark(configs, iterations):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "out_dtype", "fp8_dtype"],
            x_vals=[(*c[0], c[1], c[2]) for c in configs],
            line_arg="provider",
            line_vals=["latency_us", "tflops", "bandwidth_GBs"],
            line_names=["Latency (us)", "TFLOPS", "Bandwidth (GB/s)"],
            styles=[("blue", "-"), ("green", "--"), ("orange", ":")],
            ylabel="value",
            plot_name="fp8_gemm_per_tensor",
            args={},
        )
    )
    def benchmark(
        m, n, k, out_dtype, fp8_dtype, provider, iterations=iterations
    ):
        total_latency = 0.0
        assert iterations > 5

        input = torch.randn([m, k], dtype=out_dtype, device=DEVICE) / 10.0
        weight = torch.randn([n, k], dtype=out_dtype, device=DEVICE) / 10.0

        scale_src = torch.tensor(4.0, device=DEVICE)
        scale_wei = torch.tensor(4.0, device=DEVICE)

        input_fp8, _ = scaled_fp8_quant(input, scale_src, fp8_dtype=fp8_dtype)
        weight_fp8, _ = scaled_fp8_quant(weight, scale_wei, fp8_dtype=fp8_dtype)
        weight_fp8_t = weight_fp8.transpose(0, 1)

        start_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        end_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        for index in range(iterations):
            if index >= 5:
                start_event[index - 5].record()
            fp8_gemm(
                input_fp8,
                weight_fp8_t,
                out_dtype,
                scale_src,
                scale_wei,
            )
            if index >= 5:
                end_event[index - 5].record()

        torch.xpu.synchronize()
        total_latency = sum(
            start_event[i].elapsed_time(end_event[i])
            for i in range(iterations - 5)
        )
        ms = total_latency / (iterations - 5)
        clear_xpu_cache()

        if provider == "tflops":
            flops = calculate_flops(m, n, k)
            return flops / (ms / 1000) / 1e12
        elif provider == "bandwidth_GBs":
            mem_bytes = calculate_memory_bytes(m, n, k, out_dtype, fp8_dtype)
            return mem_bytes / (ms / 1000) / 1e9
        else:  # latency_us
            return 1000 * ms

    return benchmark


# ---------------------------------------------------------------------------
# Benchmark: fp8_gemm_w8a16
# ---------------------------------------------------------------------------


def get_fp8_gemm_w8a16_benchmark(configs, iterations):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "out_dtype", "fp8_dtype"],
            x_vals=[(*c[0], c[1], c[2]) for c in configs],
            line_arg="provider",
            line_vals=["latency_us", "tflops", "bandwidth_GBs"],
            line_names=["Latency (us)", "TFLOPS", "Bandwidth (GB/s)"],
            styles=[("blue", "-"), ("green", "--"), ("orange", ":")],
            ylabel="value",
            plot_name="fp8_gemm_w8a16",
            args={},
        )
    )
    def benchmark(
        m, n, k, out_dtype, fp8_dtype, provider, iterations=iterations
    ):
        total_latency = 0.0
        assert iterations > 5

        input = torch.randn([m, k], dtype=out_dtype, device=DEVICE) / 10.0
        weight = torch.ones([n, k], dtype=out_dtype, device=DEVICE)
        scale_wei = torch.tensor(4.0, device=DEVICE)

        weight_fp8, _ = scaled_fp8_quant(
            weight, scale_wei, fp8_dtype=fp8_dtype, group_shape=(-1, 1)
        )
        weight_fp8_t = weight_fp8.transpose(0, 1)

        start_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        end_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        for index in range(iterations):
            if index >= 5:
                start_event[index - 5].record()
            fp8_gemm_w8a16(input, weight_fp8_t, scale_wei, torch.Tensor())
            if index >= 5:
                end_event[index - 5].record()

        torch.xpu.synchronize()
        total_latency = sum(
            start_event[i].elapsed_time(end_event[i])
            for i in range(iterations - 5)
        )
        ms = total_latency / (iterations - 5)
        clear_xpu_cache()

        if provider == "tflops":
            flops = calculate_flops(m, n, k)
            return flops / (ms / 1000) / 1e12
        elif provider == "bandwidth_GBs":
            mem_bytes = calculate_memory_bytes(m, n, k, out_dtype, fp8_dtype)
            return mem_bytes / (ms / 1000) / 1e9
        else:
            return 1000 * ms

    return benchmark


# ---------------------------------------------------------------------------
# Benchmark: fp8_gemm per-channel
# ---------------------------------------------------------------------------


def get_fp8_gemm_per_channel_benchmark(configs, iterations):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "out_dtype", "fp8_dtype"],
            x_vals=[(*c[0], c[1], c[2]) for c in configs],
            line_arg="provider",
            line_vals=["latency_us", "tflops", "bandwidth_GBs"],
            line_names=["Latency (us)", "TFLOPS", "Bandwidth (GB/s)"],
            styles=[("blue", "-"), ("green", "--"), ("orange", ":")],
            ylabel="value",
            plot_name="fp8_gemm_per_channel",
            args={},
        )
    )
    def benchmark(
        m, n, k, out_dtype, fp8_dtype, provider, iterations=iterations
    ):
        total_latency = 0.0
        assert iterations > 5

        input = torch.randn([m, k], dtype=out_dtype, device=DEVICE) / 10.0
        weight = torch.randn([n, k], dtype=out_dtype, device=DEVICE) / 10.0

        input_fp8, scale_src = scaled_fp8_quant(
            input, use_per_token_if_dynamic=True, fp8_dtype=fp8_dtype
        )
        weight_fp8, scale_wei = scaled_fp8_quant(
            weight, use_per_token_if_dynamic=True, fp8_dtype=fp8_dtype
        )
        weight_fp8_t = weight_fp8.transpose(0, 1)

        start_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        end_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        for index in range(iterations):
            if index >= 5:
                start_event[index - 5].record()
            fp8_gemm(
                input_fp8,
                weight_fp8_t,
                out_dtype,
                scale_src,
                scale_wei,
            )
            if index >= 5:
                end_event[index - 5].record()

        torch.xpu.synchronize()
        total_latency = sum(
            start_event[i].elapsed_time(end_event[i])
            for i in range(iterations - 5)
        )
        ms = total_latency / (iterations - 5)
        clear_xpu_cache()

        if provider == "tflops":
            flops = calculate_flops(m, n, k)
            return flops / (ms / 1000) / 1e12
        elif provider == "bandwidth_GBs":
            mem_bytes = calculate_memory_bytes(m, n, k, out_dtype, fp8_dtype)
            return mem_bytes / (ms / 1000) / 1e9
        else:
            return 1000 * ms

    return benchmark


# ---------------------------------------------------------------------------
# Benchmark: mxfp8 gemm
# ---------------------------------------------------------------------------


def get_mxfp8_gemm_benchmark(configs, iterations):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "out_dtype"],
            x_vals=[(*c[0], c[1]) for c in configs],
            line_arg="provider",
            line_vals=["latency_us", "tflops", "bandwidth_GBs"],
            line_names=["Latency (us)", "TFLOPS", "Bandwidth (GB/s)"],
            styles=[("blue", "-"), ("green", "--"), ("orange", ":")],
            ylabel="value",
            plot_name="mxfp8_gemm",
            args={},
        )
    )
    def benchmark(m, n, k, out_dtype, provider, iterations=iterations):
        total_latency = 0.0
        assert iterations > 5

        inputs = torch.randn((m, k), dtype=out_dtype, device=DEVICE) * 0.01
        weights = torch.randn((n, k), dtype=out_dtype, device=DEVICE) * 0.01

        if out_dtype == torch.half:
            inputs = inputs.to(torch.float32)
            weights = weights.to(torch.float32)

        _, inputs_lp, inputs_scale = _convert_to_mxfp8(inputs)
        _, weights_lp, weights_scale = _convert_to_mxfp8(weights)

        start_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        end_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        for index in range(iterations):
            if index >= 5:
                start_event[index - 5].record()
            fp8_gemm(
                inputs_lp,
                weights_lp.transpose(0, 1),
                out_dtype,
                inputs_scale,
                weights_scale,
            )
            if index >= 5:
                end_event[index - 5].record()

        torch.xpu.synchronize()
        total_latency = sum(
            start_event[i].elapsed_time(end_event[i])
            for i in range(iterations - 5)
        )
        ms = total_latency / (iterations - 5)
        clear_xpu_cache()

        if provider == "tflops":
            flops = calculate_flops(m, n, k)
            return flops / (ms / 1000) / 1e12
        elif provider == "bandwidth_GBs":
            mem_bytes = calculate_memory_bytes(
                m, n, k, out_dtype, torch.float8_e4m3fn
            )
            return mem_bytes / (ms / 1000) / 1e9
        else:
            return 1000 * ms

    return benchmark


def _convert_to_mxfp8(t):
    t_scale, t_lp = to_mxfp(t, format="mxfp8")
    t_hp = from_blocked_format(t_lp, t_scale, blocksize=32)
    return t_hp, t_lp, t_scale


def _convert_to_mxfp4(t):
    t_scale, t_lp = to_mxfp(t, format="mxfp4")
    t_hp = from_blocked_format(
        _floatx_unpacked_to_f32(unpack_uint4(t_lp), FP4_EBITS, FP4_MBITS),
        t_scale,
        blocksize=32,
    )
    return t_hp, t_lp, t_scale


# ---------------------------------------------------------------------------
# Benchmark: mxfp4 gemm
# ---------------------------------------------------------------------------


def get_mxfp4_gemm_benchmark(configs, iterations):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "out_dtype"],
            x_vals=[(*c[0], c[1]) for c in configs],
            line_arg="provider",
            line_vals=["latency_us", "tflops", "bandwidth_GBs"],
            line_names=["Latency (us)", "TFLOPS", "Bandwidth (GB/s)"],
            styles=[("blue", "-"), ("green", "--"), ("orange", ":")],
            ylabel="value",
            plot_name="mxfp4_gemm",
            args={},
        )
    )
    def benchmark(m, n, k, out_dtype, provider, iterations=iterations):
        total_latency = 0.0
        assert iterations > 5

        inputs = torch.randn((m, k), dtype=out_dtype, device=DEVICE) * 0.01
        weights = torch.randn((n, k), dtype=out_dtype, device=DEVICE) * 0.01

        if out_dtype == torch.half:
            inputs = inputs.to(torch.float32)
            weights = weights.to(torch.float32)

        _, inputs_lp, inputs_scale = _convert_to_mxfp4(inputs)
        _, weights_lp, weights_scale = _convert_to_mxfp4(weights)

        start_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        end_event = [
            torch.xpu.Event(enable_timing=True) for i in range(iterations - 5)
        ]
        for index in range(iterations):
            if index >= 5:
                start_event[index - 5].record()
            fp4_gemm(
                inputs_lp,
                weights_lp.transpose(0, 1),
                inputs_scale,
                weights_scale,
                out_dtype,
            )
            if index >= 5:
                end_event[index - 5].record()

        torch.xpu.synchronize()
        total_latency = sum(
            start_event[i].elapsed_time(end_event[i])
            for i in range(iterations - 5)
        )
        ms = total_latency / (iterations - 5)
        clear_xpu_cache()

        if provider == "tflops":
            flops = calculate_flops(m, n, k)
            return flops / (ms / 1000) / 1e12
        elif provider == "bandwidth_GBs":
            # fp4 weight is packed as uint8 (2 elements per byte)
            w_bytes = k * n // 2
            x_bytes = m * k // 2
            out_bytes = m * n * torch.tensor([], dtype=out_dtype).element_size()
            mem_bytes = x_bytes + w_bytes + out_bytes
            return mem_bytes / (ms / 1000) / 1e9
        else:
            return 1000 * ms

    return benchmark


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def gemm_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["bf16"],
        choices=ALL_BENCHMARKS + ["all"],
        help=(
            "Benchmarks to run. Default: bf16. "
            "Use 'all' to run everything. "
            f"Choices: {ALL_BENCHMARKS}"
        ),
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/gemm/",
        help="Path to save benchmark results",
    )
    args = parser.parse_args()
    if "all" in args.benchmarks:
        args.benchmarks = ALL_BENCHMARKS
    return args


if __name__ == "__main__":
    args = gemm_parse_args()
    seed_everything(1234)
    iterations = 200
    enabled = set(args.benchmarks)

    use_model_shapes = "model_shapes" in enabled

    bench_registry = {
        "bf16": (
            "bf16 gemm (F.linear)",
            gen_bf16_gemm_perf_configs,
            get_bf16_gemm_benchmark,
            None,
        ),
        "fp8": (
            "fp8_gemm per-tensor (w8a8)",
            gen_fp8_gemm_perf_configs,
            get_fp8_gemm_benchmark,
            check_fp8_gemm_per_tensor,
        ),
        "fp8_w8a16": (
            "fp8_gemm_w8a16",
            gen_fp8_gemm_w8a16_perf_configs,
            get_fp8_gemm_w8a16_benchmark,
            check_fp8_gemm_w8a16,
        ),
        "fp8_per_channel": (
            "fp8_gemm per-channel",
            gen_fp8_gemm_per_channel_perf_configs,
            get_fp8_gemm_per_channel_benchmark,
            check_fp8_gemm_per_channel,
        ),
        "mxfp8": (
            "mxfp8 gemm",
            gen_mxfp8_gemm_perf_configs,
            get_mxfp8_gemm_benchmark,
            check_mxfp8_gemm,
        ),
        "mxfp4": (
            "mxfp4 gemm",
            gen_mxfp4_gemm_perf_configs,
            get_mxfp4_gemm_benchmark,
            check_mxfp4_gemm,
        ),
    }

    for name, (
        label,
        gen_configs,
        get_bench,
        check_fn,
    ) in bench_registry.items():
        if name not in enabled:
            continue

        if use_model_shapes and name in (
            "bf16",
            "fp8",
            "fp8_w8a16",
            "fp8_per_channel",
        ):
            configs = gen_weight_shape_configs(name)
            suffix = " with model weight shapes"
        else:
            configs = gen_configs()
            suffix = ""

        if not configs:
            continue

        # Correctness
        if check_fn is not None:
            print()
            print("=" * 60)
            print(f"Correctness: {label}")
            print("=" * 60)
            for cfg in configs[:6]:
                try:
                    check_fn(cfg)
                except Exception as e:
                    print("Error:", cfg, e)
                clear_xpu_cache()

        # Performance
        print()
        print("=" * 60)
        print(f"Performance: {label}{suffix}")
        print("=" * 60)
        bench = get_bench(configs, iterations)
        bench.run(print_data=True, save_path=args.save_path)
