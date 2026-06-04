# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import itertools
from argparse import ArgumentParser

import torch
import triton
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests.register_ops import topk_per_row_decode, topk_per_row_prefill

# Prefill configs
prefill_num_rows_range = [1,1024]
prefill_vocab_size_range = [65536 ,131072]
prefill_topk_range = [1, 1024, 2048]
prefill_configs = list(
    itertools.product(
        prefill_num_rows_range,
        prefill_vocab_size_range,
        prefill_topk_range,
    ))

# Decode configs
decode_batch_size_range = [1,1024]
decode_next_n_range = [1, 8]
decode_vocab_size_range = [65536, 131072]
decode_topk_range = [1, 1024, 2048]
decode_configs = list(
    itertools.product(
        decode_batch_size_range,
        decode_next_n_range,
        decode_vocab_size_range,
        decode_topk_range,
    ))


def native_topk_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    num_rows: int,
    top_k: int,
) -> torch.Tensor:
    indices = logits.topk(min(top_k, logits.shape[1]), dim=-1)[1]
    return indices.to(torch.int32)


def native_topk_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    num_rows: int,
    top_k: int,
) -> torch.Tensor:
    indices = logits.topk(min(top_k, logits.shape[1]), dim=-1)[1]
    return indices.to(torch.int32)


def get_benchmark(mode: str):
    if mode == "prefill":
        plot_name = "topk_per_row_prefill-perf"

        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=[
                    "num_rows",
                    "vocab_size",
                    "top_k",
                ],
                x_vals=[tuple(_) for _ in prefill_configs],
                line_arg="provider",
                line_vals=["vllm", "native"],
                line_names=["vllm", "native"],
                styles=[("blue", "-"), ("green", "-")],
                ylabel="us",
                plot_name=plot_name,
                args={},
            ))
        def benchmark(
            num_rows: int,
            vocab_size: int,
            top_k: int,
            provider: str = "vllm",
        ):
            logits = torch.randn((num_rows, vocab_size),
                                 dtype=torch.float32,
                                 device="xpu")
            row_starts = torch.zeros(num_rows,
                                     dtype=torch.int32,
                                     device="xpu")
            row_ends = torch.full((num_rows, ),
                                  vocab_size,
                                  dtype=torch.int32,
                                  device="xpu")
            indices = torch.empty((num_rows, top_k),
                                  dtype=torch.int32,
                                  device="xpu")

            quantiles = [0.5, 0.2, 0.8]

            if provider == "vllm":
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: topk_per_row_prefill(
                        logits, row_starts, row_ends, indices, num_rows, top_k),
                    quantiles=quantiles,
                )
            elif provider == "native":
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: native_topk_per_row_prefill(
                        logits, row_starts, row_ends, num_rows, top_k),
                    quantiles=quantiles,
                )

            return 1000 * ms, 1000 * max_ms, 1000 * min_ms

        return benchmark

    elif mode == "decode":
        plot_name = "topk_per_row_decode-perf"

        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=[
                    "batch_size",
                    "next_n",
                    "vocab_size",
                    "top_k",
                ],
                x_vals=[tuple(_) for _ in decode_configs],
                line_arg="provider",
                line_vals=["vllm", "native"],
                line_names=["vllm", "native"],
                styles=[("blue", "-"), ("green", "-")],
                ylabel="us",
                plot_name=plot_name,
                args={},
            ))
        def benchmark(
            batch_size: int,
            next_n: int,
            vocab_size: int,
            top_k: int,
            provider: str = "vllm",
        ):
            num_rows = batch_size * next_n
            seq_lens = torch.full((batch_size, ),
                                  vocab_size,
                                  dtype=torch.int32,
                                  device="xpu")
            logits = torch.randn((num_rows, vocab_size),
                                 dtype=torch.float32,
                                 device="xpu")
            indices = torch.empty((num_rows, top_k),
                                  dtype=torch.int32,
                                  device="xpu")

            quantiles = [0.5, 0.2, 0.8]

            if provider == "vllm":
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: topk_per_row_decode(
                        logits, next_n, seq_lens, indices, num_rows, top_k),
                    quantiles=quantiles,
                )
            elif provider == "native":
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: native_topk_per_row_decode(
                        logits, next_n, seq_lens, num_rows, top_k),
                    quantiles=quantiles,
                )

            return 1000 * ms, 1000 * max_ms, 1000 * min_ms

        return benchmark

    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Benchmark the topk_per_row kernel.")
    parser.add_argument(
        "--mode",
        choices=["prefill", "decode"],
        default="prefill",
        help="Mode to benchmark (prefill or decode)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/topk_per_row/",
        help="Path to save topk_per_row benchmark results",
    )

    args = parser.parse_args()

    benchmark = get_benchmark(args.mode)
    save_path = f"{args.save_path.rstrip('/')}/{args.mode}"
    benchmark.run(print_data=True, save_path=save_path)
