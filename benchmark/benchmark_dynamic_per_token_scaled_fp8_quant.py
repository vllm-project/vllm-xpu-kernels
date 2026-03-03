# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Callable
from unittest.mock import patch

import torch
import pandas as pd

from tests.utils import parse_args, opcheck
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE


DEVICE = "xpu"


def with_triton_mode(fn):
    """Temporarily force the Triton fallback path"""

    def wrapped(*args, **kwargs):
        with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
            return fn(*args, **kwargs)

    return wrapped


# TODO(luka): use standalone_compile utility
def with_dyn_arg(fn: Callable, arg_index: int, dim_index: int):
    def inner(*args):
        torch._dynamo.mark_dynamic(args[arg_index], dim_index)
        return fn(*args)

    return inner


def bench_compile(fn: Callable):
    # recompile for different shapes
    fwd = torch.compile(fn, fullgraph=True, dynamic=False)

    # First dim is explicitly dynamic to simulate vLLM usage
    return with_dyn_arg(fwd, 0, 0)


torch._dynamo.config.recompile_limit = 8888


def calculate_diff(config):
    group_shape, column_major_scale, batch_size, hidden_size, dtype = config
    """Calculate the difference between Inductor and CUDA implementations."""
    device = torch.device(DEVICE)
    x = torch.randn((batch_size, hidden_size), dtype=dtype, device=device)

    quant_fp8 = QuantFP8(False, group_shape, column_major_scales=column_major_scale)

    torch_out, torch_scale = bench_compile(quant_fp8.forward_native)(x)
    torch_eager_out, torch_eager_scale = quant_fp8.forward_native(x)
    cuda_out, cuda_scale = quant_fp8.forward_cuda(x)

    try:
        torch.testing.assert_close(
            cuda_out.to(torch.float32),
            torch_out.to(torch.float32),
            rtol=1e-3,
            atol=1e-5,
        )
        torch.testing.assert_close(cuda_scale, torch_scale, rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(
            cuda_out.to(torch.float32),
            torch_eager_out.to(torch.float32),
            rtol=1e-3,
            atol=1e-5,
        )
        torch.testing.assert_close(cuda_scale, torch_eager_scale, rtol=1e-3, atol=1e-5)
        print("✅ All implementations match, ", config)
    except AssertionError as e:
        print("❌ Implementations differ, ", config)
        print(e)


def compute_geomean_speedups(
    df: pd.DataFrame,
    baseline_col: str,
    speedup_cols: list[str],
    groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute geometric mean speedups over a baseline column.

    Args:
        df: Input dataframe
        baseline_col: Column to use as baseline
        speedup_cols: Columns to compute speedups for
        groupby_cols: Columns to group by. If None, compute over entire df.

    Returns:
        pd.DataFrame with geometric mean speedups
    """
    from scipy.stats import gmean

    def geo_speedup(group: pd.DataFrame) -> pd.Series:
        ratios = {
            col: (group[baseline_col] / group[col]).values for col in speedup_cols
        }
        return pd.Series({col: gmean(vals) for col, vals in ratios.items()})

    if groupby_cols is None:
        result = geo_speedup(df).to_frame().T
    else:
        result = (
            df.groupby(groupby_cols)
            .apply(geo_speedup, include_groups=False)
            .reset_index()
        )

    return result


@default_vllm_config()
def benchmark_quantization(
    batch_size,
    hidden_size,
    provider,
    group_shape: GroupShape,
    col_major: bool,
    dtype: torch.dtype,
):
    device = torch.device(DEVICE)

    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    quant_fp8 = QuantFP8(False, group_shape, column_major_scales=col_major)

    if provider == "torch":
        x_clone = x.clone()
        fn = lambda: bench_compile(quant_fp8.forward_native)(x_clone)
    elif provider == DEVICE:
        x_clone = x.clone()
        fn = lambda: quant_fp8.forward_cuda(x_clone)
    elif provider == "triton":
        if not group_shape.is_per_group():
            # Triton only supported for per-group
            return 0, 0, 0

        x_clone = x.clone()
        fn = lambda: with_triton_mode(quant_fp8.forward_cuda)(x_clone)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


configs = []


def get_benchmark():
    benchmark = triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["hidden_size", "batch_size", "col_major", "group_shape", "dtype"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["torch", DEVICE, "triton"],
            line_names=["Torch (Compiled)", DEVICE.upper(), "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("black", "-")],
            ylabel="us",
            plot_name="QuantFP8 performance",
            args={},
        )
    )(benchmark_quantization)
    return benchmark


if __name__ == "__main__":

    print("No XPU support!!!")
    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)

    hidden_sizes = [896, 1024, 2048, 4096, 7168]
    batch_sizes = [1, 16, 128, 512, 1024]
    group_shapes = [GroupShape.PER_TOKEN]
    column_major_scales = [True, False]
    dtypes = ["half", "bfloat16", "float"]
    print("Final configuration:")
    print("hidden_sizes:", hidden_sizes)
    print("batch_sizes:", batch_sizes)
    print("group_shapes:", group_shapes)
    print("column_major_scales:", column_major_scales)
    print("dtypes:", dtypes)

    configs = list(
        itertools.product(group_shapes, column_major_scales, batch_sizes, hidden_sizes, dtypes))

    for config in configs:
        calculate_diff(config)

    benchmark = get_benchmark()
    # Run performance benchmark
    df = benchmark.run(print_data=True, return_df=True)

    # Print geomean speedups
    geo_table_grouped = compute_geomean_speedups(
        df,
        baseline_col="Torch (Compiled)",
        speedup_cols=[DEVICE.upper(), "Triton"],
        groupby_cols=["col_major", "group_shape"],
    )

    print("Speedup over Torch (Compiled)")
    print(geo_table_grouped.to_string(index=False))
