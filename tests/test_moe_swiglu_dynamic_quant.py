import os
import csv
import time
import pytest
import torch
from collections import defaultdict

import tests.register_ops as ops  # noqa: F401

torch.manual_seed(42)

CSV_FILENAME = "test_moe_swiglu_quant.csv"
_CACHE_FLUSH_TENSOR = None


def _flush_cache(device: str):
    global _CACHE_FLUSH_TENSOR
    if _CACHE_FLUSH_TENSOR is None or _CACHE_FLUSH_TENSOR.device.type != device:
        _CACHE_FLUSH_TENSOR = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int32, device=device)
    _CACHE_FLUSH_TENSOR.zero_()


@pytest.fixture(scope="session", autouse=True)
def setup_csv():
    categories = ["dtype", "num_scattered", "hidden_size", "num_experts"]

    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(categories + ["status", "custom_time_us", "custom_bw_gbps", "custom_tflops"])

    yield

    if not os.path.exists(CSV_FILENAME):
        return

    global_bw, global_tf, global_c_lat = [], [], []
    global_pass = global_total = 0
    grouped = defaultdict(lambda: defaultdict(lambda: {"bw": [], "tf": [], "c_lat": [], "passes": 0, "total": 0}))

    with open(CSV_FILENAME, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row["status"]
            global_total += 1
            is_pass = (status == "PASS")

            if is_pass:
                global_pass += 1
                try:
                    bw, tf = float(row["custom_bw_gbps"]), float(row["custom_tflops"])
                    c_lat = float(row["custom_time_us"])
                    global_bw.append(bw)
                    global_tf.append(tf)
                    global_c_lat.append(c_lat)
                except ValueError:
                    is_pass = False

            for cat in categories:
                val = row[cat]
                grouped[cat][val]["total"] += 1
                if is_pass:
                    grouped[cat][val]["passes"] += 1
                    grouped[cat][val]["bw"].append(bw)
                    grouped[cat][val]["tf"].append(tf)
                    grouped[cat][val]["c_lat"].append(c_lat)

    if global_total == 0:
        return

    g_pass_pct = (global_pass / global_total) * 100
    g_mu_bw = sum(global_bw) / len(global_bw) if global_bw else 0
    g_min_bw = min(global_bw) if global_bw else 0
    g_max_bw = max(global_bw) if global_bw else 0

    g_mu_tf = sum(global_tf) / len(global_tf) if global_tf else 0
    g_min_tf = min(global_tf) if global_tf else 0
    g_max_tf = max(global_tf) if global_tf else 0

    g_mu_c_lat = sum(global_c_lat) / len(global_c_lat) if global_c_lat else 0

    print("\n\n" + "=" * 120)
    print(f"{'PARAMETER':<23} {'VALUE':<12} {'PASS%':<7} {'LATENCY (us)':<15} | {'BW (GB/s) [Mean / Min / Max]':<28} | {'TFLOPS [Mean / Min / Max]'}")
    print("-" * 120)
    print(f"{'Global Average':<23} {'-':<12} {g_pass_pct:<6.1f}% {g_mu_c_lat:<15.1f} | {g_mu_bw:<6.1f} / {g_min_bw:<6.1f} / {g_max_bw:<6.1f}       | {g_mu_tf:<6.3f} / {g_min_tf:<6.3f} / {g_max_tf:<6.3f}")
    print("-" * 120)

    for cat in categories:
        print(f"{cat.upper().replace('_', ' ')}")
        for val, data in grouped[cat].items():
            pass_pct = (data["passes"] / data["total"]) * 100
            valid = len(data["c_lat"]) > 0

            grp_c_lat = sum(data["c_lat"]) / len(data["c_lat"]) if valid else 0.0
            lat_str = f"{grp_c_lat:<15.1f}" if valid else "N/A"

            mean_bw = sum(data['bw']) / len(data['bw']) if valid else 0
            min_bw = min(data['bw']) if valid else 0
            max_bw = max(data['bw']) if valid else 0

            mean_tf = sum(data['tf']) / len(data['tf']) if valid else 0
            min_tf = min(data['tf']) if valid else 0
            max_tf = max(data['tf']) if valid else 0

            print(f"{'':<23} {val:<12} {pass_pct:<6.1f}% {lat_str:<15} | "
                  f"{mean_bw:<6.1f} / {min_bw:<6.1f} / {max_bw:<6.1f}       | "
                  f"{mean_tf:<6.3f} / {min_tf:<6.3f} / {max_tf:<6.3f}")
    print("=" * 120 + "\n")


def baseline_moe_swiglu(scatter_tokens, smooth_scale, experts_token_count, experts_token_start):
    """Pure PyTorch vectorized baseline for MoE SwiGLU + Dynamic Quantization."""
    num_experts = experts_token_count.shape[0]
    hidden_size = scatter_tokens.shape[1] // 2
    device = scatter_tokens.device

    expert_ids = torch.arange(num_experts, device=device).repeat_interleave(experts_token_count)
    token_scales = smooth_scale[expert_ids].float()

    x1 = scatter_tokens[:, :hidden_size].float()
    x2 = scatter_tokens[:, hidden_size:].float()

    x1_silu = torch.nn.functional.silu(x1)
    swiglu = x1_silu * x2
    scaled = swiglu * token_scales

    unquantized_math = scaled

    max_vals = scaled.abs().max(dim=-1).values
    scale = max_vals / 127.0
    scale = torch.where(scale == 0, torch.tensor(1.0, device=device), scale)

    quant_tokens = torch.round(scaled / scale.unsqueeze(-1)).to(torch.int8)
    per_token_scale = scale

    return quant_tokens, per_token_scale, unquantized_math

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_scattered": [40, 80, 256],
        "hidden_size": [64, 128, 256],
        "num_experts": [64],
    },
}

@pytest.mark.parametrize("num_scattered", [40, 80, 256, 1024, 4096])
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 1024, 2048, 4096, 7168])
@pytest.mark.parametrize("num_experts", [64, 256])
def test_moe_swiglu_dynamic_quant(num_scattered, hidden_size, num_experts):
    device = "xpu"
    dtype = torch.bfloat16

    experts_token_count = torch.zeros(num_experts, dtype=torch.int32, device=device)
    experts_token_start = torch.zeros(num_experts, dtype=torch.int32, device=device)

    tokens_per_expert = num_scattered // num_experts
    experts_token_count[:] = tokens_per_expert
    experts_token_count[-1] = num_scattered - (tokens_per_expert * (num_experts - 1))

    start = 0
    max_token_num = 0
    for i in range(num_experts):
        experts_token_start[i] = start
        start += experts_token_count[i].item()
        if experts_token_count[i].item() > max_token_num:
            max_token_num = experts_token_count[i].item()

    scatter_tokens = torch.randn((num_scattered, hidden_size * 2), dtype=dtype, device=device)
    smooth_scale = torch.rand((num_experts, hidden_size), dtype=torch.float32, device=device)

    out_quant_tokens = torch.zeros((num_scattered, hidden_size), dtype=torch.int8, device=device)
    out_per_scale = torch.zeros(num_scattered, dtype=torch.float32, device=device)

    status = "FAIL"
    custom_time_s = custom_bw = custom_tflops = 0.0
    accuracy_check_total_time = prep_time = bench_time = bench_iters = 0.0

    try:
        accuracy_check_start_time = time.perf_counter()

        # 1. Base Accuracy Reference
        ref_quant_tokens, ref_per_scale, ref_unquantized_tokens = baseline_moe_swiglu(
            scatter_tokens, smooth_scale, experts_token_count, experts_token_start
        )

        out_quant_tokens.zero_()
        out_per_scale.zero_()

        # 2. XPU Kernel Output
        torch.ops._moe_C.moe_swiglu_dynamic_quant(
            scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
            out_quant_tokens, out_per_scale, num_experts, max_token_num
        )
        torch.xpu.synchronize()

        # 3. Accuracy Verifications
        custom_dequantized = out_quant_tokens.float() * out_per_scale.unsqueeze(1)
        float_diff = (custom_dequantized - ref_unquantized_tokens).abs()
        max_float_error = float_diff.max().item()

        torch.testing.assert_close(out_per_scale, ref_per_scale, atol=1e-5, rtol=1e-5)

        diff = (out_quant_tokens.int() - ref_quant_tokens.int()).abs()
        assert diff.max().item() <= 1, "Quantized values diverge completely!"
        assert max_float_error < 0.5, f"Dequantized mathematical outputs diverge! Max float error: {max_float_error}"

        accuracy_check_total_time = time.perf_counter() - accuracy_check_start_time

        # 4. Benchmarks
        def bench_fn(warmup=2, profile=4, min_test_iters=25, max_test_iters=float("inf"), max_test_time=0.3, sleep_duration=0.05):
            bench_prep_start_time = time.perf_counter()
            # Warmup
            for _ in range(warmup):
                torch.ops._moe_C.moe_swiglu_dynamic_quant(
                    scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
                    out_quant_tokens, out_per_scale, num_experts, max_token_num
                )
            torch.xpu.synchronize()

            # Profile
            start_time = time.perf_counter()
            for _ in range(profile):
                torch.ops._moe_C.moe_swiglu_dynamic_quant(
                    scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
                    out_quant_tokens, out_per_scale, num_experts, max_token_num
                )
            torch.xpu.synchronize()

            profile_latency = (time.perf_counter() - start_time) / profile

            prefer_iters = int(max_test_time / profile_latency) if profile_latency > 0 else min_test_iters
            prefer_iters = max(min_test_iters, min(prefer_iters, int(max_test_iters) if max_test_iters != float("inf") else prefer_iters))

            time.sleep(sleep_duration)

            # Warmup
            for _ in range(warmup):
                torch.ops._moe_C.moe_swiglu_dynamic_quant(
                    scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
                    out_quant_tokens, out_per_scale, num_experts, max_token_num
                )
            torch.xpu.synchronize()

            bench_prep_total_time = time.perf_counter() - bench_prep_start_time

            # Benchmark
            start_time = time.perf_counter()
            for _ in range(prefer_iters):
                torch.ops._moe_C.moe_swiglu_dynamic_quant(
                    scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
                    out_quant_tokens, out_per_scale, num_experts, max_token_num
                )
            torch.xpu.synchronize()
            total_time = time.perf_counter() - start_time

            return total_time / prefer_iters, bench_prep_total_time, total_time, prefer_iters

        custom_time_s, prep_time, bench_time, bench_iters = bench_fn()

        # 5. Metrics
        # Bytes = Read Scatter 2*H (bf16), Read Scale H (f32), Write Quant H (int8), Write per_token 1 (f32)
        total_bytes = (num_scattered * hidden_size * 2 * 2) + \
                      (num_scattered * hidden_size * 4) + \
                      (num_scattered * hidden_size * 1) + \
                      (num_scattered * 4)

        # Flops approx: silu + mul + scale + quant (rating at roughly 6 operations per element)
        total_flops = num_scattered * hidden_size * 6

        custom_bw = (total_bytes / 1e9) / custom_time_s
        custom_tflops = (total_flops / 1e12) / custom_time_s
        status = "PASS"

    except Exception as e:
        status = "FAIL"
        raise e

    finally:
        print(f"\n[SwiGLU Quant] N={num_scattered} Hid={hidden_size} Exp={num_experts} | "
              f"{status} | Custom: {custom_time_s*1e6:.2f} us ({custom_bw:.2f} GB/s) | "
              f"Accuracy Time: {accuracy_check_total_time:.4f} s | Prep Time: {prep_time:.4f} s | "
              f"Bench Time: {bench_time:.4f} s | Bench Iters: {int(bench_iters)}")

        with open(CSV_FILENAME, "a", newline="") as f:
            csv.writer(f).writerow([
                "bfloat16", num_scattered, hidden_size, num_experts, status,
                custom_time_s * 1e6, custom_bw, custom_tflops
            ])
