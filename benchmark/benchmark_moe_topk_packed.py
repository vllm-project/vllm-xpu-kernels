# SPDX-License-Identifier: Apache-2.0
"""Device-time benchmark for 512-expert MoE softmax top-k.

Run on each branch with the same command and build to compare dispatch paths.
"""

import argparse
import time

import torch
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)
import tests.register_ops  # noqa: E402, F401


def device_time_us(run, warmup, rep, graph_repeats):
    for _ in range(warmup):
        run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(graph_repeats):
            run()
    for _ in range(warmup):
        graph.replay()
    torch.xpu.synchronize()
    timings = []
    for _ in range(rep):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        timings.append((start, end))
    torch.xpu.synchronize()
    samples = sorted(
        start.elapsed_time(end) * 1000 / graph_repeats
        for start, end in timings)
    return samples[rep // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype",
                        choices=["float16", "bfloat16"],
                        default="bfloat16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--graph-repeats", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(2026)
    dtype = getattr(torch, args.dtype)
    print("dtype,tokens,experts,topk,renormalize,p50_us", flush=True)
    for n_token in (1, 64, 512, 513, 2048, 2049, 4096):
        logits = torch.randn(n_token, 512, device="xpu", dtype=dtype)
        for topk in (3, 10):
            weights = torch.empty(n_token,
                                  topk,
                                  device="xpu",
                                  dtype=torch.float32)
            ids = torch.empty(n_token, topk, device="xpu", dtype=torch.int32)
            source = torch.empty_like(ids)

            def run(weights=weights, ids=ids, source=source, logits=logits):
                torch.ops._moe_C.topk_softmax(weights, ids, source, logits,
                                              True, None, None)

            median = device_time_us(run, args.warmup, args.rep,
                                    args.graph_repeats)
            print(f"{args.dtype},{n_token},512,{topk},True,{median:.3f}",
                  flush=True)
        torch.xpu.synchronize()
        time.sleep(2)


if __name__ == "__main__":
    main()
