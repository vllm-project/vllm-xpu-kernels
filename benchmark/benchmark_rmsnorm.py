# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from typing import Optional, Union

import torch
import triton
from torch import nn

from tests import register_ops as vllm_ops
from tests.utils import parse_args


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


def rmsnorm_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    native_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    native_norm.weight = nn.Parameter(weight)
    native_norm = native_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = native_norm(x, residual)

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


def calculate_diff(config):
    head_num, batch_size, seq_len, use_residual, dtype = config
    hidden_size = head_num * 128  # assuming head_dim = 128
    x = torch.randn(batch_size,
                    seq_len,
                    hidden_size,
                    dtype=dtype,
                    device="xpu")
    weight = torch.ones(hidden_size, dtype=dtype, device="xpu")
    residual = torch.randn_like(x) if use_residual else None

    output_native = rmsnorm_native(
        x.clone(), weight,
        residual.clone() if residual is not None else None)
    output_vllm = rmsnorm_vllm(
        x.clone(), weight,
        residual.clone() if residual is not None else None)

    if use_residual:
        output_native = output_native[0]
        output_vllm = output_vllm[0]

    if torch.allclose(output_native, output_vllm, atol=1e-2, rtol=1e-2):
        print("✅ All implementations match, ", config)
    else:
        print("❌ Implementations differ, ", config)


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["head_num", "batch_size", "seq_len", "use_residual", "dtype"],
            x_vals=[tuple(_) for _ in configs],
            line_arg="provider",
            line_vals=["huggingface", "vllm", "t.compile"],
            line_names=["HuggingFace", "vLLM", "t.compile"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="us",
            plot_name=
            f"rmsnorm-perf",
            args={},
        ))
    def benchmark(head_num, batch_size, seq_len, use_residual, dtype, provider):
        hidden_size = head_num * 128  # assuming head_dim = 128

        x = torch.randn(batch_size,
                        seq_len,
                        hidden_size,
                        dtype=dtype,
                        device="xpu")
        weight = torch.ones(hidden_size, dtype=dtype, device="xpu")
        residual = torch.randn_like(x) if use_residual else None

        quantiles = [0.5, 0.2, 0.8]
        print(f"Running config: {(head_num, batch_size, seq_len, use_residual, dtype)}, Provider: {provider}", flush=True)

        if provider == "huggingface":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_native(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "t.compile":
            x_clone = x.clone()
            residual_clone = residual.clone() if residual is not None else None
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_compile(
                    x_clone,
                    weight,
                    residual_clone,
                ),
                quantiles=quantiles,
            )
        else:
            x_clone = x.clone()
            residual_clone = residual.clone() if residual is not None else None
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_vllm(
                    x_clone,
                    weight,
                    residual_clone,
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
    use_residual = [True, False]
    dtype = [torch.float16, torch.bfloat16]
    configs = list(
        itertools.product(head_num_range, batch_size_range, seq_length_range, use_residual, dtype))


    # Run correctness test
    for config in configs:
        calculate_diff(
            config
        )

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
