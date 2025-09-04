# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from argparse import ArgumentParser

import torch

from tests.ops.layernorm_op import RMSNorm
from tests.utils import STR_DTYPE_TO_TORCH_DTYPE


@torch.inference_mode()
def main(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int = 0,
    num_warmup_iters: int = 5,
    num_iters: int = 100,
) -> None:
    torch.set_default_device("xpu")

    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    def run_xpu_benchmark(num_iters: int) -> float:
        torch.xpu.synchronize()

        start_time = time.perf_counter()

        for _ in range(num_iters):
            layer(x, residual)
        torch.xpu.synchronize()

        end_time = time.perf_counter()

        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark = run_xpu_benchmark
    run_benchmark(num_iters=num_warmup_iters)

    # Benchmark.
    latency = run_benchmark(num_iters=num_iters)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the layernorm kernel.")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--add-residual", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup-iters", type=int, default=5)
    parser.add_argument("--num-iters",
                        type=int,
                        default=100,
                        help="Number of benchmark iterations. ")

    args = parser.parse_args()
    print(args)

    main(
        num_tokens=args.num_tokens,
        hidden_size=args.hidden_size,
        add_residual=args.add_residual,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        num_warmup_iters=args.num_warmup_iters,
        num_iters=args.num_iters,
    )
