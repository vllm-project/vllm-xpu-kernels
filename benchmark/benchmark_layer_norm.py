# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import itertools
from typing import Optional

import torch
import torch.nn.functional as F
import triton
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

import tests.register_ops as vllm_ops
from tests.utils import parse_args


def layernorm_naive(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    """torch.nn.functional.layer_norm: the real fused-ATen fallback this
    kernel replaces (not the CustomOp's unfused fp32 correctness reference)."""
    if residual is not None:
        x = x + residual
        residual = x
    out = F.layer_norm(x, (x.shape[-1], ), weight, bias, eps)
    if residual is None:
        return out
    return out, residual


@torch.compile
def layernorm_compile(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)

    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    x = (x - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        x = x * weight.float()
    if bias is not None:
        x = x + bias.float()
    x = x.to(orig_dtype)
    if residual is None:
        return x
    return x, residual


def _run_vllm_norm(fused_op, plain_op, x, weight, bias, residual, eps):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        fused_op(x, residual, weight, bias, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        plain_op(out, x, weight, bias, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def layernorm_vllm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    return _run_vllm_norm(vllm_ops.fused_add_layer_norm, vllm_ops.layer_norm,
                          x, weight, bias, residual, eps)


def nemotron_layernorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    """Same +1-folded-weight trick as GemmaRMSNorm: layer_norm(x) with no
    weight/bias applied by F.layer_norm itself, then (1 + weight) and bias
    applied manually -- F.layer_norm has no native "(1 + weight)" fold.
    The fold is done in fp32 (matching the kernel's internal accumulators)
    since doing it directly in bf16 loses enough precision to fail the
    allclose check against the real kernel at hidden_size=4096."""
    orig_dtype = x.dtype
    if residual is not None:
        x = x + residual
        residual = x
    out = F.layer_norm(x.float(), (x.shape[-1], ), None, None, eps)
    out = out * (1.0 + weight.float())
    if bias is not None:
        out = out + bias.float()
    out = out.to(orig_dtype)
    if residual is None:
        return out
    return out, residual


def nemotron_layernorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    return _run_vllm_norm(vllm_ops.fused_add_nemotron_layer_norm,
                          vllm_ops.nemotron_layer_norm, x, weight, bias,
                          residual, eps)


@torch.compile
def nemotron_layernorm_compile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)

    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    x = (x - mean) * torch.rsqrt(var + eps)
    x = x * (1.0 + weight.float())
    if bias is not None:
        x = x + bias.float()
    x = x.to(orig_dtype)
    if residual is None:
        return x
    return x, residual


def calculate_diff(batch_size, seq_len, hidden_size, use_residual=True,
                   nemotron=False):
    dtype = torch.bfloat16
    x = torch.randn(batch_size,
                    seq_len,
                    hidden_size,
                    dtype=dtype,
                    device="xpu")
    weight = torch.randn(hidden_size, dtype=dtype, device="xpu")
    bias = torch.randn(hidden_size, dtype=dtype, device="xpu")
    residual = torch.randn_like(x) if use_residual else None

    if nemotron:
        naive_fn, vllm_fn = nemotron_layernorm_naive, nemotron_layernorm_vllm
    else:
        naive_fn, vllm_fn = layernorm_naive, layernorm_vllm

    output_naive = naive_fn(
        x.clone(), weight, bias,
        residual.clone() if residual is not None else None)
    output_vllm = vllm_fn(
        x.clone(), weight, bias,
        residual.clone() if residual is not None else None)

    if use_residual:
        output_naive = output_naive[0]
        output_vllm = output_vllm[0]

    print(f"Naive output={output_naive}")
    print(f"vLLM output={output_vllm}")

    if torch.allclose(output_naive, output_vllm, atol=1e-2, rtol=1e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


def get_benchmark(use_residual, dtype, nemotron=False):
    if nemotron:
        naive_fn, vllm_fn = nemotron_layernorm_naive, nemotron_layernorm_vllm
        compile_fn = nemotron_layernorm_compile
    else:
        naive_fn, vllm_fn = layernorm_naive, layernorm_vllm
        compile_fn = layernorm_compile

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["head_num", "batch_size", "seq_len"],
            x_vals=[tuple(_) for _ in configs],
            line_arg="provider",
            line_vals=["native", "vllm", "t.compile"],
            line_names=["Native", "vLLM", "t.compile"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="us",
            plot_name=(f"{'nemotron-' if nemotron else ''}layernorm-perf-"
                       f"{'with' if use_residual else 'without'}-residual"),
            args={},
        ))
    def benchmark(head_num, batch_size, seq_len, provider):
        hidden_size = head_num * 128  # assuming head_dim = 128

        x = torch.randn(batch_size,
                        seq_len,
                        hidden_size,
                        dtype=dtype,
                        device="xpu")
        weight = torch.randn(hidden_size, dtype=dtype, device="xpu")
        bias = torch.randn(hidden_size, dtype=dtype, device="xpu")
        residual = torch.randn_like(x) if use_residual else None

        quantiles = [0.5, 0.2, 0.8]

        # Clone once, outside the timed region -- see benchmark_rmsnorm.py's
        # identical note: vLLM's fused_add_* variants mutate in place, so
        # do_bench's repeated calls reuse the same buffers across iterations,
        # matching production (no per-call clone) and avoiding an extra
        # memory-bound copy inside the timed loop.
        x_bench = x.clone()
        residual_bench = residual.clone() if residual is not None else None

        if provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: naive_fn(x_bench, weight, bias, residual_bench),
                quantiles=quantiles,
            )
        elif provider == "t.compile":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: compile_fn(x_bench, weight, bias, residual_bench),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: vllm_fn(x_bench, weight, bias, residual_bench),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()
    # parse_args()'s shared --save-path default ("./configs/rmsnorm/") is
    # named for its original RMSNorm benchmark; give this file its own
    # default instead of intermixing results in that directory.
    if args.save_path == "./configs/rmsnorm/":
        args.save_path = "./configs/layernorm/"

    print("Final configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Data type: {args.dtype}")
    print(f"  Use residual: {args.use_residual}")

    batch_size_range = [2**i for i in range(0, 7, 2)]
    seq_length_range = [2**i for i in range(6, 10, 1)]
    head_num_range = args.head_num_range
    configs = list(
        itertools.product(head_num_range, batch_size_range, seq_length_range))

    for nemotron in (False, True):
        label = "NemotronLayerNorm1P" if nemotron else "LayerNorm"
        print(f"\n=== {label} ===")

        # Run correctness test
        calculate_diff(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            use_residual=args.use_residual,
            nemotron=nemotron,
        )

        # Get the benchmark function with proper use_residual setting
        benchmark = get_benchmark(args.use_residual, args.dtype,
                                  nemotron=nemotron)
        # Run performance benchmark
        benchmark.run(print_data=True, save_path=args.save_path)
