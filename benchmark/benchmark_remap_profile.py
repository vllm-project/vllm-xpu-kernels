# SPDX-License-Identifier: Apache-2.0
"""Profile the existing native top-k/dispatch benchmark without duplicating it.

Usage:
    python benchmark_remap_profile.py \
        /path/to/benchmark_xpu_moe_topk_dispatch.py \
        --tokens 32768 --stages dispatch_remap
"""

import importlib.util
import json
import sys
from collections import defaultdict

import torch

source = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location("moe_benchmark", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_benchmark = module.benchmark


def profile_benchmark(function, args):
    result = original_benchmark(function, args)
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
    ]) as profiler:
        for _ in range(5):
            function()
        torch.xpu.synchronize()
    path = f"remap-copy-profile-{args.tokens}.json"
    profiler.export_chrome_trace(path)
    with open(path) as stream:
        events = json.load(stream)["traceEvents"]
    durations = defaultdict(list)
    for event in events:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            durations[event["name"]].append(event["dur"])
    for name, values in durations.items():
        print(f"KERNEL mean_us={sum(values) / len(values):.3f} "
              f"events={len(values)} {name}", flush=True)
    print(f"Trace: {path}", flush=True)
    return result


module.benchmark = profile_benchmark
module.main()
