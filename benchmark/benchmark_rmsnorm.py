# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from typing import Optional, Union

import torch
import triton
from torch import nn

from tests import register_ops as vllm_ops
from tests.utils import check_ipex_availability, parse_args

HAS_IPEX = check_ipex_availability()

if HAS_IPEX:
    import intel_extension_for_pytorch as ipex


class HuggingFaceRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


@torch.compile
def rmsnorm_compile(x: torch.Tensor,
                    weight: torch.Tensor,
                    residual: Optional[torch.Tensor] = None,
                    eps: float = 1e-6):
    """PyTorch-native implementation equivalent to forward()."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)

    x_var = x
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)

    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype)
    x = x * weight
    if residual is None:
        return x
    else:
        return x, residual


def rmsnorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_ipex(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    """IPEX implementation using ipex.llm.functional.rms_norm"""
    if not HAS_IPEX:
        raise RuntimeError("IPEX is not available")

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])

    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])
        if hasattr(ipex.llm.functional, 'fused_add_rms_norm'):
            output, residual_out = ipex.llm.functional.fused_add_rms_norm(
                x, residual, weight, eps)
            output = (output.view(orig_shape), residual_out.view(orig_shape))
        else:
            x = x + residual
            output = ipex.llm.functional.rms_norm(x, weight, eps)
            output = (output.view(orig_shape), x.view(orig_shape))
    else:
        output = ipex.llm.functional.rms_norm(x, weight, eps)
        output = output.view(orig_shape)

    return output


def calculate_diff(batch_size, seq_len, hidden_size, use_residual=True):
    dtype = torch.bfloat16
    x = torch.randn(batch_size,
                    seq_len,
                    hidden_size,
                    dtype=dtype,
                    device="xpu")
    weight = torch.ones(hidden_size, dtype=dtype, device="xpu")
    residual = torch.randn_like(x) if use_residual else None

    output_naive = rmsnorm_naive(
        x.clone(), weight,
        residual.clone() if residual is not None else None)
    output_vllm = rmsnorm_vllm(
        x.clone(), weight,
        residual.clone() if residual is not None else None)

    if use_residual:
        output_naive = output_naive[0]
        output_vllm = output_vllm[0]

    print(f"Naive output={output_naive}")
    print(f"vLLM output={output_vllm}")

    if HAS_IPEX:
        try:
            output_ipex = rmsnorm_ipex(
                x.clone(), weight,
                residual.clone() if residual is not None else None)
            if use_residual:
                output_ipex = output_ipex[0]
            print(f"IPEX output={output_ipex}")

            if torch.allclose(output_naive, output_ipex, atol=1e-2, rtol=1e-2):
                print("✅ IPEX implementation matches naive")
            else:
                print("❌ IPEX implementation differs from naive")
        except Exception as e:
            print(f"❌ IPEX implementation failed: {e}")

    if torch.allclose(output_naive, output_vllm, atol=1e-2, rtol=1e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


def get_benchmark(use_residual, dtype):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["head_num", "batch_size", "seq_len"],
            x_vals=[tuple(_) for _ in configs],
            line_arg="provider",
            line_vals=["huggingface", "vllm", "t.compile", "ipex"]
            if HAS_IPEX else ["huggingface", "vllm", "t.compile"],
            line_names=["HuggingFace", "vLLM", "t.compile", "IPEX"]
            if HAS_IPEX else ["HuggingFace", "vLLM", "t.compile"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"),
                    ("red", "-")] if HAS_IPEX else [("blue", "-"),
                                                    ("green", "-"),
                                                    ("orange", "-")],
            ylabel="us",
            plot_name=
            f"rmsnorm-perf-{'with' if use_residual else 'without'}-residual",
            args={},
        ))
    def benchmark(head_num, batch_size, seq_len, provider):
        hidden_size = head_num * 128  # assuming head_dim = 128

        x = torch.randn(batch_size,
                        seq_len,
                        hidden_size,
                        dtype=dtype,
                        device="xpu")
        weight = torch.ones(hidden_size, dtype=dtype, device="xpu")
        residual = torch.randn_like(x) if use_residual else None

        quantiles = [0.5, 0.2, 0.8]

        if provider == "huggingface":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_naive(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "t.compile":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_compile(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "ipex" and HAS_IPEX:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_ipex(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_vllm(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()

    print("Final configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Intermediate size: {args.intermediate_size}")
    print(f"  Number of groups: {args.num_groups}")
    print(f"  Data type: {args.dtype}")
    print(f"  Use residual: {args.use_residual}")

    batch_size_range = [2**i for i in range(0, 7, 2)]
    seq_length_range = [2**i for i in range(6, 10, 1)]
    head_num_range = args.head_num_range
    configs = list(
        itertools.product(head_num_range, batch_size_range, seq_length_range))

    if HAS_IPEX:
        print("✅ IPEX is available")
        print(f"IPEX version: {ipex.__version__}")
    else:
        print("⚠️  IPEX is not available, skipping IPEX benchmarks")

    # Run correctness test
    calculate_diff(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        use_residual=args.use_residual,
    )

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark(args.use_residual, args.dtype)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
