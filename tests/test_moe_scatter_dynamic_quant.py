import os
import csv
import time
import pytest
import torch
import statistics
from collections import defaultdict

import tests.register_ops as ops  # noqa: F401

torch.manual_seed(42)

CSV_FILENAME = "test_moe_scatter_quant.csv"
_CACHE_FLUSH_TENSOR = None


def _flush_cache(device: str):
    global _CACHE_FLUSH_TENSOR
    if _CACHE_FLUSH_TENSOR is None or _CACHE_FLUSH_TENSOR.device.type != device:
        _CACHE_FLUSH_TENSOR = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int32, device=device)
    _CACHE_FLUSH_TENSOR.zero_()


@pytest.fixture(scope="session", autouse=True)
def setup_csv():
    categories = ["dtype", "num_tokens", "hidden_size", "topk", "num_experts"]

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


def baseline_moe_scatter(selected_experts, moe_weights, hidden_states, experts_smooth_scale, total_experts):
    """Pure PyTorch vectorized baseline for MoE Scatter + Dynamic Quantization."""
    num_tokens, topk = selected_experts.shape
    hidden_size = hidden_states.shape[1]
    device = hidden_states.device

    out_tokens_count = torch.zeros(total_experts, dtype=torch.int32, device=device)
    out_token_to_scatter_offset = torch.zeros_like(selected_experts, dtype=torch.int32)

    # 1. Count and offsets
    for t in range(num_tokens):
        for k in range(topk):
            exp = selected_experts[t, k].item()
            out_token_to_scatter_offset[t, k] = out_tokens_count[exp].item()
            out_tokens_count[exp] += 1

    # 2. Prefix sum
    out_tokens_start = torch.zeros_like(out_tokens_count)
    out_tokens_start[1:] = torch.cumsum(out_tokens_count[:-1], dim=0)

    # 3. Target mapping
    flat_experts = selected_experts.flatten().long()
    flat_offsets = out_token_to_scatter_offset.flatten()
    target_idx = out_tokens_start[flat_experts] + flat_offsets

    # 4. Gather & compute
    h_expanded = hidden_states.float().repeat_interleave(topk, dim=0)
    w_expanded = moe_weights.float().flatten().unsqueeze(-1)
    s_extracted = experts_smooth_scale.float()[flat_experts]

    smoothed = h_expanded * s_extracted * w_expanded

    # 5. Quantize
    max_vals = smoothed.abs().max(dim=-1).values
    scale = max_vals / 127.0
    scale = torch.where(scale == 0, torch.tensor(1.0, device=device), scale)
    quantized = torch.round(smoothed / scale.unsqueeze(-1)).to(torch.int8)

    # 6. Scatter outputs
    total_scattered = num_tokens * topk
    scatter_tokens = torch.zeros((total_scattered, hidden_size), dtype=torch.int8, device=device)
    scatter_tokens[target_idx] = quantized

    per_token_scale = torch.zeros(total_scattered, dtype=torch.float32, device=device)
    per_token_scale[target_idx] = scale

    tokens_offset = torch.zeros(total_scattered, dtype=torch.int32, device=device)
    flat_t = torch.arange(num_tokens, device=device).repeat_interleave(topk).int()
    tokens_offset[target_idx] = flat_t

    unquantized_math = torch.zeros((total_scattered, hidden_size), dtype=torch.float32, device=device)
    unquantized_math[target_idx] = smoothed

    return (out_token_to_scatter_offset, out_tokens_count, out_tokens_start,
            scatter_tokens, per_token_scale, tokens_offset, unquantized_math)


#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [40, 80],
        "hidden_size": [64, 256],
        "topk": [5],
        "num_experts": [64],
    },
}

@pytest.mark.parametrize("num_tokens", [40, 80, 256, 4096])
@pytest.mark.parametrize("hidden_size", [64, 256, 1024, 7168])
@pytest.mark.parametrize("topk", [5, 8])
@pytest.mark.parametrize("num_experts", [64, 256])
def test_moe_scatter_dynamic_quant(num_tokens, hidden_size, topk, num_experts):
    device = "xpu"
    dtype = torch.bfloat16
    shared_experts_num = 0

    selected_experts = torch.rand((num_tokens, num_experts), device=device).topk(topk, dim=-1).indices.to(torch.int32)
    moe_weights = torch.rand((num_tokens, topk), dtype=torch.float32, device=device)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device)
    experts_smooth_scale = torch.rand((num_experts, hidden_size), dtype=torch.float32, device=device)

    out_t_offset = torch.zeros_like(selected_experts)
    out_t_count = torch.zeros(num_experts, dtype=torch.int32, device=device)
    out_t_start = torch.zeros(num_experts, dtype=torch.int32, device=device)
    out_scatter_tokens = torch.zeros((num_tokens * topk, hidden_size), dtype=torch.int8, device=device)
    out_per_scale = torch.zeros(num_tokens * topk, dtype=torch.float32, device=device)
    out_tokens_offset = torch.zeros(num_tokens * topk, dtype=torch.int32, device=device)

    status = "FAIL"
    custom_time_s = custom_bw = custom_tflops = 0.0
    accuracy_check_total_time = prep_time = bench_time = bench_iters = 0.0

    try:
        accuracy_check_start_time = time.perf_counter()

        # 1. Base Accuracy Reference
        (ref_t_offset, ref_t_count, ref_t_start, ref_scatter_tokens,
         ref_per_scale, ref_tokens_offset, ref_unquantized_tokens) = baseline_moe_scatter(
             selected_experts, moe_weights, hidden_states, experts_smooth_scale, num_experts
        )

        # 2. XPU Kernel Output
        torch.ops._moe_C.moe_scatter_dynamic_quant(
            selected_experts, moe_weights, out_t_offset, out_t_count, out_t_start,
            hidden_states, experts_smooth_scale, out_scatter_tokens, out_per_scale,
            out_tokens_offset, shared_experts_num
        )
        torch.xpu.synchronize()

        # 3. Accuracy Verifications
        torch.testing.assert_close(out_t_count, ref_t_count)
        torch.testing.assert_close(out_t_start, ref_t_start)

        expert_ids = torch.zeros(num_tokens * topk, dtype=torch.int64, device=device)
        for i in range(num_experts):
            start, count = ref_t_start[i].item(), ref_t_count[i].item()
            if count > 0:
                expert_ids[start:start+count] = i

        sort_key_custom = (expert_ids * num_tokens) + out_tokens_offset
        sort_key_ref = (expert_ids * num_tokens) + ref_tokens_offset

        sort_idx_custom = torch.argsort(sort_key_custom)
        sort_idx_ref = torch.argsort(sort_key_ref)

        torch.testing.assert_close(out_tokens_offset[sort_idx_custom], ref_tokens_offset[sort_idx_ref])
        torch.testing.assert_close(out_per_scale[sort_idx_custom], ref_per_scale[sort_idx_ref], atol=1e-5, rtol=1e-5)

        diff = (out_scatter_tokens[sort_idx_custom].int() - ref_scatter_tokens[sort_idx_ref].int()).abs()
        assert diff.max().item() <= 1, "Quantized values diverge completely!"

        custom_dequantized = out_scatter_tokens.float() * out_per_scale.unsqueeze(1)
        flat_experts = selected_experts.long()

        ref_indices = ref_t_start[flat_experts] + ref_t_offset
        custom_indices = out_t_start[flat_experts] + out_t_offset

        float_diff = (custom_dequantized[custom_indices] - ref_unquantized_tokens[ref_indices]).abs()
        max_float_error = float_diff.max().item()
        assert max_float_error < 0.5, f"Dequantized mathematical outputs diverge! Max float error: {max_float_error}"

        accuracy_check_total_time = time.perf_counter() - accuracy_check_start_time

        # 4. Performance Benchmarks
        def bench_fn(warmup=2, profile=4, min_test_iters=25, max_test_iters=float("inf"), max_test_time=0.5, sleep_duration=0.05):
            bench_prep_start_time = time.perf_counter()
            # Warmup
            for _ in range(warmup):
                out_t_count.zero_()
                torch.ops._moe_C.moe_scatter_dynamic_quant(
                    selected_experts, moe_weights, out_t_offset, out_t_count, out_t_start,
                    hidden_states, experts_smooth_scale, out_scatter_tokens, out_per_scale,
                    out_tokens_offset, shared_experts_num
                )
            torch.xpu.synchronize()

            # Profile
            start_time = time.perf_counter()
            for _ in range(profile):
                out_t_count.zero_()
                torch.ops._moe_C.moe_scatter_dynamic_quant(
                    selected_experts, moe_weights, out_t_offset, out_t_count, out_t_start,
                    hidden_states, experts_smooth_scale, out_scatter_tokens, out_per_scale,
                    out_tokens_offset, shared_experts_num
                )
            torch.xpu.synchronize()
            profile_latency = (time.perf_counter() - start_time) / profile

            prefer_iters = int(max_test_time / profile_latency) if profile_latency > 0 else min_test_iters
            prefer_iters = max(min_test_iters, min(prefer_iters, int(max_test_iters) if max_test_iters != float("inf") else prefer_iters))

            time.sleep(sleep_duration)

            # Warmup
            for _ in range(warmup):
                out_t_count.zero_()
                torch.ops._moe_C.moe_scatter_dynamic_quant(
                    selected_experts, moe_weights, out_t_offset, out_t_count, out_t_start,
                    hidden_states, experts_smooth_scale, out_scatter_tokens, out_per_scale,
                    out_tokens_offset, shared_experts_num
                )
            torch.xpu.synchronize()

            bench_prep_total_time = time.perf_counter() - bench_prep_start_time

            # Benchmark
            start_time = time.perf_counter()
            for _ in range(prefer_iters):
                out_t_count.zero_()
                torch.ops._moe_C.moe_scatter_dynamic_quant(
                    selected_experts, moe_weights, out_t_offset, out_t_count, out_t_start,
                    hidden_states, experts_smooth_scale, out_scatter_tokens, out_per_scale,
                    out_tokens_offset, shared_experts_num
                )
            torch.xpu.synchronize()
            total_time = time.perf_counter() - start_time

            return total_time / prefer_iters, bench_prep_total_time, total_time, prefer_iters

        custom_time_s, prep_time, bench_time, bench_iters = bench_fn()

        # 5. Calculate Metrics
        total_bytes = (num_tokens * hidden_size * 2) + \
                      (num_experts * hidden_size * 4) + \
                      (num_tokens * topk * 4) + \
                      (num_tokens * topk * hidden_size * 1)
        total_flops = num_tokens * topk * hidden_size * 2

        custom_bw = (total_bytes / 1e9) / custom_time_s
        custom_tflops = (total_flops / 1e12) / custom_time_s
        status = "PASS"

    except Exception as e:
        status = "FAIL"
        raise e
    finally:
        print(f"\n[Scatter Quant] Tok={num_tokens} Hid={hidden_size} TopK={topk} Exp={num_experts} | "
              f"{status} | Custom: {custom_time_s*1e6:.2f} us ({custom_bw:.2f} GB/s) | "
              f"Accuracy Time: {accuracy_check_total_time:.4f} s | Prep Time: {prep_time:.4f} s | "
              f"Bench Time: {bench_time:.4f} s | Bench Iters: {int(bench_iters)}")

        with open(CSV_FILENAME, "a", newline="") as f:
            csv.writer(f).writerow([
                "bfloat16", num_tokens, hidden_size, topk, num_experts, status,
                custom_time_s * 1e6, custom_bw, custom_tflops
            ])
