"""
## Benchmark Modes

The test suite supports two modes:

**Benchmark Mode (default):**
- `BENCHMARK_MODE=1` or not set
- Performs 1 warmup run + 5 benchmark runs
- Results are averaged for accurate measurements
- Use for performance analysis

**Fast Mode:**
- `BENCHMARK_MODE=0`
- Single run, no warmup
- Faster execution for quick validation
- Results include cold-start overhead

Example comparison:
```bash
# Fast mode: 104ms (includes JIT compilation overhead)
BENCHMARK_MODE=0 pytest test_wan22_kernels_ops_bf16.py::TestConv1DKernels::test_conv1d_audio_encoder_large -v -s

# Benchmark mode: 1.134ms (accurate, warmed-up performance)
BENCHMARK_MODE=1 pytest test_wan22_kernels_ops_bf16.py::TestConv1DKernels::test_conv1d_audio_encoder_large -v -s
```

Usage:
```bash
# Run all tests with accurate benchmarking (default, BENCHMARK_MODE=1)
python -m pytest test_wan22_kernels_ops_bf16.py -v -s

# Run all tests in fast mode (quick validation)
BENCHMARK_MODE=0 python -m pytest test_wan22_kernels_ops_bf16.py -v -s

# View generated report
cat wan22_test_report.md
```

Note: The report `wan22_test_report.md` is automatically generated after tests complete.
"""

import os

import numpy as np
import pytest
import torch

# Test device configuration
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"

# Configuration from profiling runs (WAN 2.2 A14B)
BATCH_SIZE = 1
SEQ_LEN = 75600  # 21 × 45 × 80 patches
HIDDEN_DIM = 5120
NUM_HEADS = 40
HEAD_DIM = 128
FFN_DIM = 13824
TEXT_SEQ_LEN = 512
NUM_FRAMES_T2V = 81
NUM_FRAMES_S2V = 80
LATENT_HEIGHT = 90
LATENT_WIDTH = 160

WARMUP = 1
RUNS = 5

# Benchmark mode: if False, run only once (no warmup, no averaging)
# Set via environment variable: BENCHMARK_MODE=0 to disable
BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "1") != "0"

# Store test results for report generation
TEST_RESULTS = []


def calculate_flops(op_type, shape_info):
    """Calculate FLOPs for different operation types."""
    if op_type == "conv1d":
        batch = shape_info["batch"]
        in_ch = shape_info["in_channels"]
        out_ch = shape_info["out_channels"]
        kernel = shape_info["kernel_size"]
        out_len = shape_info["output_length"]
        return 2 * batch * out_ch * out_len * in_ch * kernel

    elif op_type == "conv2d":
        batch = shape_info["batch"]
        in_ch = shape_info["in_channels"]
        out_ch = shape_info["out_channels"]
        k_h = shape_info["kernel_h"]
        k_w = shape_info["kernel_w"]
        out_h = shape_info["output_h"]
        out_w = shape_info["output_w"]
        return 2 * batch * out_ch * out_h * out_w * in_ch * k_h * k_w

    elif op_type == "conv3d":
        batch = shape_info["batch"]
        in_ch = shape_info["in_channels"]
        out_ch = shape_info["out_channels"]
        k_d = shape_info["kernel_d"]
        k_h = shape_info["kernel_h"]
        k_w = shape_info["kernel_w"]
        out_d = shape_info["output_d"]
        out_h = shape_info["output_h"]
        out_w = shape_info["output_w"]
        return 2 * batch * out_ch * out_d * out_h * out_w * in_ch * k_d * k_h * k_w

    elif op_type == "linear":
        batch = shape_info["batch"]
        seq_len = shape_info["seq_len"]
        in_features = shape_info["in_features"]
        out_features = shape_info["out_features"]
        return 2 * batch * seq_len * in_features * out_features

    elif op_type == "flash_attn":
        batch = shape_info["batch"]
        num_heads = shape_info["num_heads"]
        seq_len = shape_info["seq_len"]
        head_dim = shape_info["head_dim"]
        return 4 * batch * num_heads * seq_len * seq_len * head_dim

    elif op_type == "elementwise":
        return shape_info.get("elements", 0) * 2  # Minimal compute

    else:
        return 0


def calculate_memory_bytes(op_type, shape_info):
    """Calculate total memory bytes accessed."""
    dtype_bytes = 2  # bfloat16 = 2 bytes

    if op_type in ["conv1d", "conv2d", "conv3d"]:
        input_bytes = shape_info["input_size"] * dtype_bytes
        weight_bytes = shape_info["weight_size"] * dtype_bytes
        output_bytes = shape_info["output_size"] * dtype_bytes
        return input_bytes + weight_bytes + output_bytes

    elif op_type == "linear":
        batch = shape_info["batch"]
        seq_len = shape_info["seq_len"]
        in_features = shape_info["in_features"]
        out_features = shape_info["out_features"]
        input_bytes = batch * seq_len * in_features * dtype_bytes
        weight_bytes = in_features * out_features * dtype_bytes
        output_bytes = batch * seq_len * out_features * dtype_bytes
        return input_bytes + weight_bytes + output_bytes

    elif op_type == "flash_attn":
        batch = shape_info["batch"]
        num_heads = shape_info["num_heads"]
        seq_len = shape_info["seq_len"]
        head_dim = shape_info["head_dim"]
        # Q, K, V inputs + output
        return 4 * batch * num_heads * seq_len * head_dim * dtype_bytes

    elif op_type == "elementwise":
        return shape_info.get("elements", 0) * dtype_bytes * 2  # Read + write

    else:
        return 0


def benchmark_with_metrics(name, prepare_fn, compute_fn, op_type, shape_info, profiling_info=None, benchmark=None):
    """
    Benchmark with CPU->XPU workflow, measuring kernel time only.

    Args:
        name: Test name
        prepare_fn: Function that creates tensors/models on CPU, returns (tensors_dict, operation)
        compute_fn: Function that performs computation, takes (tensors_dict, operation)
        op_type: Operation type for FLOP calculation
        shape_info: Shape information for metrics
        profiling_info: Optional profiling notes
        benchmark: If True, do warmup + 5 runs. If False, run once only. If None, use BENCHMARK_MODE.

    Returns:
        Dict with metrics or None if CPU
    """
    if DEVICE == "cpu":
        return None

    if benchmark is None:
        benchmark = BENCHMARK_MODE

    # Step 1: Prepare on CPU
    tensors_dict, operation = prepare_fn()

    # Step 2: Transfer to XPU (not measured)
    tensors_xpu = {}
    for k, v in tensors_dict.items():
        if isinstance(v, torch.Tensor):
            tensors_xpu[k] = v.to("xpu")
        else:
            tensors_xpu[k] = v

    # Transfer operation(s) to XPU
    if operation is None:
        pass  # No operation to transfer
    elif isinstance(operation, (tuple, list)):
        # Handle multiple operations (e.g., k_proj, v_proj)
        operation = type(operation)(op.to("xpu") if hasattr(op, "to") else op for op in operation)
    elif hasattr(operation, "to"):
        operation = operation.to("xpu")

    # Step 3: Warmup (if benchmark mode)
    if benchmark:
        for _ in range(WARMUP):
            compute_fn(tensors_xpu, operation)
            torch.xpu.synchronize()

    # Step 4: Benchmark kernel only
    latencies = []
    num_runs = RUNS if benchmark else 1
    for _ in range(num_runs):
        torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)

        start.record()
        compute_fn(tensors_xpu, operation)
        end.record()

        torch.xpu.synchronize()
        latencies.append(start.elapsed_time(end))

    # Step 5: Calculate metrics
    avg_ms = np.mean(latencies)
    std_ms = np.std(latencies)

    total_flops = calculate_flops(op_type, shape_info)
    total_memory_bytes = calculate_memory_bytes(op_type, shape_info)

    tflops = (total_flops / (avg_ms / 1000)) / 1e12 if avg_ms > 0 else 0
    memory_bw_gbs = (total_memory_bytes / (avg_ms / 1000)) / 1e9 if avg_ms > 0 else 0

    result = {
        "name": name,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "tflops": tflops,
        "memory_bw_gbs": memory_bw_gbs,
        "shape_info": shape_info,
        "profiling_info": profiling_info,
    }

    TEST_RESULTS.append(result)
    return result


# ============================================================================
# PART 1: KERNEL-LEVEL TESTS (from test_wan22_kernels.py)
# ============================================================================

# ============================================================================
# 1. Conv1D Operations (S2V Audio Encoder)
# ============================================================================


class TestConv1DKernels:
    """Test Conv1D operations (audio encoding in S2V)."""

    def test_conv1d_audio_encoder_large(self):
        """Test Conv1D in audio encoder (large input: 80K samples)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            audio = torch.randn(1, 1, 80056, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv1d(1, 512, kernel_size=10, device="cpu", dtype=torch.bfloat16)
            return {"audio": audio}, conv

        def compute(tensors, operation):
            return operation(tensors["audio"])

        shape_info = {
            "batch": 1,
            "in_channels": 1,
            "out_channels": 512,
            "kernel_size": 10,
            "output_length": 80047,  # (80056 - 10 + 1)
            "input_size": 1 * 1 * 80056,
            "weight_size": 512 * 1 * 10,
            "output_size": 1 * 512 * 80047,
        }

        result = benchmark_with_metrics("Conv1D Audio 80K", prepare, compute, "conv1d", shape_info)

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_conv1d_audio_projector(self):
        """Test Conv1D in audio feature projector (S2V: 4 calls, 92ms)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            audio_proj = torch.randn(4, 1280, 156, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv1d(1280, 2560, kernel_size=3, device="cpu", dtype=torch.bfloat16)
            return {"audio_proj": audio_proj}, conv

        def compute(tensors, operation):
            return operation(tensors["audio_proj"])

        shape_info = {
            "batch": 4,
            "in_channels": 1280,
            "out_channels": 2560,
            "kernel_size": 3,
            "output_length": 154,  # (156 - 3 + 1)
            "input_size": 4 * 1280 * 156,
            "weight_size": 2560 * 1280 * 3,
            "output_size": 4 * 2560 * 154,
        }

        result = benchmark_with_metrics(
            "Conv1D Audio Projector",
            prepare,
            compute,
            "conv1d",
            shape_info,
            profiling_info="S2V: 4 calls, 92.66ms total",
        )

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_conv1d_audio_encoder_mid(self):
        """Test Conv1D in audio encoder (mid-size: 8K samples)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            audio_features = torch.randn(1, 512, 8004, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv1d(512, 512, kernel_size=3, device="cpu", dtype=torch.bfloat16)
            return {"audio_features": audio_features}, conv

        def compute(tensors, operation):
            return operation(tensors["audio_features"])

        shape_info = {
            "batch": 1,
            "in_channels": 512,
            "out_channels": 512,
            "kernel_size": 3,
            "output_length": 8002,  # (8004 - 3 + 1)
            "input_size": 1 * 512 * 8004,
            "weight_size": 512 * 512 * 3,
            "output_size": 1 * 512 * 8002,
        }

        result = benchmark_with_metrics("Conv1D Audio Encoder Mid", prepare, compute, "conv1d", shape_info)

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )


# ============================================================================
# 2. Conv2D Operations (VAE Upsampling)
# ============================================================================


class TestConv2DKernels:
    """Test Conv2D operations (VAE decoder upsampling)."""

    def test_conv2d_vae_256x256(self):
        """Test Conv2D at 256x256 (near pixel space)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(4, 192, 256, 256, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv2d(192, 96, kernel_size=3, padding=1, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, conv

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {
            "batch": 4,
            "in_channels": 192,
            "out_channels": 96,
            "kernel_h": 3,
            "kernel_w": 3,
            "output_h": 256,
            "output_w": 256,
            "input_size": 4 * 192 * 256 * 256,
            "weight_size": 96 * 192 * 3 * 3,
            "output_size": 4 * 96 * 256 * 256,
        }

        result = benchmark_with_metrics("Conv2D 256x256", prepare, compute, "conv2d", shape_info)

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_conv2d_vae_latent_32x32(self):
        """Test Conv2D at 32x32 latent space."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(1, 384, 32, 32, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv2d(384, 1152, kernel_size=1, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, conv

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {
            "batch": 1,
            "in_channels": 384,
            "out_channels": 1152,
            "kernel_h": 1,
            "kernel_w": 1,
            "output_h": 32,
            "output_w": 32,
            "input_size": 1 * 384 * 32 * 32,
            "weight_size": 1152 * 384 * 1 * 1,
            "output_size": 1 * 1152 * 32 * 32,
        }

        result = benchmark_with_metrics("Conv2D VAE 32x32", prepare, compute, "conv2d", shape_info)

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_conv2d_vae_upsample_64x64(self):
        """Test Conv2D at 64x64 resolution."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(2, 384, 64, 64, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv2d(384, 192, kernel_size=3, padding=1, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, conv

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {
            "batch": 2,
            "in_channels": 384,
            "out_channels": 192,
            "kernel_h": 3,
            "kernel_w": 3,
            "output_h": 64,
            "output_w": 64,
            "input_size": 2 * 384 * 64 * 64,
            "weight_size": 192 * 384 * 3 * 3,
            "output_size": 2 * 192 * 64 * 64,
        }

        result = benchmark_with_metrics("Conv2D VAE 64x64", prepare, compute, "conv2d", shape_info)

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_conv2d_vae_upsample_128x128(self):
        """Test Conv2D at 128x128 resolution."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(4, 384, 128, 128, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv2d(384, 192, kernel_size=3, padding=1, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, conv

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {
            "batch": 4,
            "in_channels": 384,
            "out_channels": 192,
            "kernel_h": 3,
            "kernel_w": 3,
            "output_h": 128,
            "output_w": 128,
            "input_size": 4 * 384 * 128 * 128,
            "weight_size": 192 * 384 * 3 * 3,
            "output_size": 4 * 192 * 128 * 128,
        }

        result = benchmark_with_metrics("Conv2D VAE 128x128", prepare, compute, "conv2d", shape_info)

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )


# ============================================================================
# 3. Conv3D Operations (VAE - CRITICAL: 40%+ XPU time)
# ============================================================================


class TestConv3DKernels:
    """Test Conv3D operations (VAE video encoder/decoder)."""

    def test_conv3d_vae_shape1_384_3_34_34(self):
        """Test most frequent Conv3D shape: [1, 384, 3, 34, 34]."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(1, 384, 3, 34, 34, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv3d(384, 384, kernel_size=3, padding=1, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, conv

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {
            "batch": 1,
            "in_channels": 384,
            "out_channels": 384,
            "kernel_d": 3,
            "kernel_h": 3,
            "kernel_w": 3,
            "output_d": 3,
            "output_h": 34,
            "output_w": 34,
            "input_size": 1 * 384 * 3 * 34 * 34,
            "weight_size": 384 * 384 * 3 * 3 * 3,
            "output_size": 1 * 384 * 3 * 34 * 34,
        }

        result = benchmark_with_metrics(
            "Conv3D [1,384,3,34,34]",
            prepare,
            compute,
            "conv3d",
            shape_info,
            profiling_info="T2V: 7,560 calls | S2V: 13,320 calls",
        )

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_conv3d_vae_shape2_96_6_258_258(self):
        """Test Conv3D shape: [1, 96, 6, 258, 258]."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(1, 96, 6, 258, 258, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv3d(96, 96, kernel_size=3, padding=1, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, conv

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {
            "batch": 1,
            "in_channels": 96,
            "out_channels": 96,
            "kernel_d": 3,
            "kernel_h": 3,
            "kernel_w": 3,
            "output_d": 6,
            "output_h": 258,
            "output_w": 258,
            "input_size": 1 * 96 * 6 * 258 * 258,
            "weight_size": 96 * 96 * 3 * 3 * 3,
            "output_size": 1 * 96 * 6 * 258 * 258,
        }

        result = benchmark_with_metrics(
            "Conv3D [1,96,6,258,258]",
            prepare,
            compute,
            "conv3d",
            shape_info,
            profiling_info="T2V: 4,320 calls | S2V: 6,912 calls",
        )

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_conv3d_vae_decoder_to_rgb(self):
        """Test Conv3D VAE decoder final layer (latent to RGB)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            latent = torch.randn(1, 96, 6, 258, 258, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv3d(96, 3, kernel_size=3, padding=1, device="cpu", dtype=torch.bfloat16)
            return {"latent": latent}, conv

        def compute(tensors, operation):
            return operation(tensors["latent"])

        shape_info = {
            "batch": 1,
            "in_channels": 96,
            "out_channels": 3,
            "kernel_d": 3,
            "kernel_h": 3,
            "kernel_w": 3,
            "output_d": 6,
            "output_h": 258,
            "output_w": 258,
            "input_size": 1 * 96 * 6 * 258 * 258,
            "weight_size": 3 * 96 * 3 * 3 * 3,
            "output_size": 1 * 3 * 6 * 258 * 258,
        }

        result = benchmark_with_metrics("Conv3D to RGB", prepare, compute, "conv3d", shape_info)

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_conv3d_vae_shape3_192_6_130_130(self):
        """Test Conv3D shape: [1, 192, 6, 130, 130]."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(1, 192, 6, 130, 130, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv3d(192, 192, kernel_size=3, padding=1, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, conv

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {
            "batch": 1,
            "in_channels": 192,
            "out_channels": 192,
            "kernel_d": 3,
            "kernel_h": 3,
            "kernel_w": 3,
            "output_d": 6,
            "output_h": 130,
            "output_w": 130,
            "input_size": 1 * 192 * 6 * 130 * 130,
            "weight_size": 192 * 192 * 3 * 3 * 3,
            "output_size": 1 * 192 * 6 * 130 * 130,
        }

        result = benchmark_with_metrics(
            "Conv3D [1,192,6,130,130]",
            prepare,
            compute,
            "conv3d",
            shape_info,
            profiling_info="T2V: 4,320 calls | S2V: 6,264 calls",
        )

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_conv3d_vae_shape4_384_4_66_66(self):
        """Test Conv3D shape: [1, 384, 4, 66, 66]."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(1, 384, 4, 66, 66, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv3d(384, 384, kernel_size=3, padding=1, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, conv

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {
            "batch": 1,
            "in_channels": 384,
            "out_channels": 384,
            "kernel_d": 3,
            "kernel_h": 3,
            "kernel_w": 3,
            "output_d": 4,
            "output_h": 66,
            "output_w": 66,
            "input_size": 1 * 384 * 4 * 66 * 66,
            "weight_size": 384 * 384 * 3 * 3 * 3,
            "output_size": 1 * 384 * 4 * 66 * 66,
        }

        result = benchmark_with_metrics(
            "Conv3D [1,384,4,66,66]",
            prepare,
            compute,
            "conv3d",
            shape_info,
            profiling_info="T2V: 3,600 calls | S2V: 5,544 calls",
        )

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_conv3d_vae_encoder_full_resolution(self):
        """Test Conv3D VAE encoder at full 720p resolution."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            video = torch.randn(1, 3, 20, 720, 1280, device="cpu", dtype=torch.bfloat16)
            conv = torch.nn.Conv3d(3, 128, kernel_size=3, padding=1, device="cpu", dtype=torch.bfloat16)
            return {"video": video}, conv

        def compute(tensors, operation):
            return operation(tensors["video"])

        shape_info = {
            "batch": 1,
            "in_channels": 3,
            "out_channels": 128,
            "kernel_d": 3,
            "kernel_h": 3,
            "kernel_w": 3,
            "output_d": 20,
            "output_h": 720,
            "output_w": 1280,
            "input_size": 1 * 3 * 20 * 720 * 1280,
            "weight_size": 128 * 3 * 3 * 3 * 3,
            "output_size": 1 * 128 * 20 * 720 * 1280,
        }

        result = benchmark_with_metrics("Conv3D VAE Encoder 720p", prepare, compute, "conv3d", shape_info)

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )


# ============================================================================
# 4. Flash Attention Kernels (12-17% XPU time)
# ============================================================================


class TestFlashAttentionKernels:
    """Test Flash Attention using vllm_xpu_kernels.flash_attn_varlen_func."""

    def test_flash_attention_varlen_self(self):
        """Test Flash Attention for self-attention (75,600 x 75,600)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        try:
            from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

            def prepare():
                q = torch.randn(BATCH_SIZE * SEQ_LEN, NUM_HEADS, HEAD_DIM, device="cpu", dtype=torch.bfloat16)
                k = torch.randn(BATCH_SIZE * SEQ_LEN, NUM_HEADS, HEAD_DIM, device="cpu", dtype=torch.bfloat16)
                v = torch.randn(BATCH_SIZE * SEQ_LEN, NUM_HEADS, HEAD_DIM, device="cpu", dtype=torch.bfloat16)
                cu_seqlens = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device="cpu")
                return {"q": q, "k": k, "v": v, "cu_seqlens": cu_seqlens}, None

            def compute(tensors, operation):
                softmax_scale = 1.0 / (HEAD_DIM**0.5)
                return flash_attn_varlen_func(
                    tensors["q"],
                    tensors["k"],
                    tensors["v"],
                    cu_seqlens_q=tensors["cu_seqlens"],
                    cu_seqlens_k=tensors["cu_seqlens"],
                    max_seqlen_q=SEQ_LEN,
                    max_seqlen_k=SEQ_LEN,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=False,
                )

            shape_info = {"batch": BATCH_SIZE, "num_heads": NUM_HEADS, "seq_len": SEQ_LEN, "head_dim": HEAD_DIM}

            result = benchmark_with_metrics(
                "Flash Attention Varlen Self",
                prepare,
                compute,
                "flash_attn",
                shape_info,
                profiling_info="WAN self-attention (non-causal, seq=75600)",
            )

            if result:
                print(
                    f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                    f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
                )

        except Exception as e:
            pytest.skip(f"flash_attn_varlen_func not available: {e}")

    def test_flash_attention_varlen_cross(self):
        """Test Flash Attention for cross-attention (75,600 x 512)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        try:
            from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

            def prepare():
                q = torch.randn(BATCH_SIZE * SEQ_LEN, NUM_HEADS, HEAD_DIM, device="cpu", dtype=torch.bfloat16)
                k = torch.randn(BATCH_SIZE * TEXT_SEQ_LEN, NUM_HEADS, HEAD_DIM, device="cpu", dtype=torch.bfloat16)
                v = torch.randn(BATCH_SIZE * TEXT_SEQ_LEN, NUM_HEADS, HEAD_DIM, device="cpu", dtype=torch.bfloat16)
                cu_seqlens_q = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device="cpu")
                cu_seqlens_kv = torch.tensor([0, TEXT_SEQ_LEN], dtype=torch.int32, device="cpu")
                return {"q": q, "k": k, "v": v, "cu_seqlens_q": cu_seqlens_q, "cu_seqlens_kv": cu_seqlens_kv}, None

            def compute(tensors, operation):
                softmax_scale = 1.0 / (HEAD_DIM**0.5)
                return flash_attn_varlen_func(
                    tensors["q"],
                    tensors["k"],
                    tensors["v"],
                    cu_seqlens_q=tensors["cu_seqlens_q"],
                    cu_seqlens_k=tensors["cu_seqlens_kv"],
                    max_seqlen_q=SEQ_LEN,
                    max_seqlen_k=TEXT_SEQ_LEN,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=False,
                )

            shape_info = {"batch": BATCH_SIZE, "num_heads": NUM_HEADS, "seq_len": SEQ_LEN, "head_dim": HEAD_DIM}

            result = benchmark_with_metrics(
                "Flash Attention Varlen Cross",
                prepare,
                compute,
                "flash_attn",
                shape_info,
                profiling_info="T2V: 16 calls | S2V: 24 calls",
            )

            if result:
                print(
                    f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                    f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
                )

        except Exception as e:
            pytest.skip(f"flash_attn_varlen_func not available: {e}")


# ============================================================================
# 5. FFN Operations (using torch.nn.Linear)
# ============================================================================


class TestFFNOps:
    """Test FFN operations using torch.nn.Linear."""

    def test_ffn_upproject(self):
        """Test FFN upproject (5120 -> 13824)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device="cpu", dtype=torch.bfloat16)
            linear = torch.nn.Linear(HIDDEN_DIM, FFN_DIM, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, linear

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {"batch": BATCH_SIZE, "seq_len": SEQ_LEN, "in_features": HIDDEN_DIM, "out_features": FFN_DIM}

        result = benchmark_with_metrics(
            "FFN Upproject", prepare, compute, "linear", shape_info, profiling_info="T2V: 16 calls | S2V: 24 calls"
        )

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_ffn_downproject(self):
        """Test FFN downproject (13824 -> 5120)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(BATCH_SIZE, SEQ_LEN, FFN_DIM, device="cpu", dtype=torch.bfloat16)
            linear = torch.nn.Linear(FFN_DIM, HIDDEN_DIM, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, linear

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {"batch": BATCH_SIZE, "seq_len": SEQ_LEN, "in_features": FFN_DIM, "out_features": HIDDEN_DIM}

        result = benchmark_with_metrics(
            "FFN Downproject", prepare, compute, "linear", shape_info, profiling_info="T2V: 16 calls | S2V: 24 calls"
        )

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_qkv_projection_linear(self):
        """Test QKV projection using torch.nn.Linear (5120 -> 15360)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device="cpu", dtype=torch.bfloat16)
            linear = torch.nn.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, linear

        def compute(tensors, operation):
            return operation(tensors["x"])

        shape_info = {
            "batch": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "in_features": HIDDEN_DIM,
            "out_features": 3 * HIDDEN_DIM,
        }

        result = benchmark_with_metrics(
            "QKV Projection (Linear)",
            prepare,
            compute,
            "linear",
            shape_info,
            profiling_info="T2V: 16 calls | S2V: 24 calls",
        )

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )


# ============================================================================
# 6. Activation Functions
# ============================================================================


class TestActivationOps:
    """Test activation functions."""

    def test_silu_activation(self):
        """Test SiLU activation (3.34% XPU time in S2V, 0.3% in T2V)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, None

        def compute(tensors, operation):
            return torch.nn.functional.silu(tensors["x"])

        shape_info = {"elements": BATCH_SIZE * SEQ_LEN * HIDDEN_DIM}

        result = benchmark_with_metrics(
            "SiLU", prepare, compute, "elementwise", shape_info, profiling_info="S2V: 28,850 calls"
        )

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )

    def test_gelu_activation(self):
        """Test GELU activation (used in FFN)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        def prepare():
            x = torch.randn(BATCH_SIZE, SEQ_LEN, FFN_DIM, device="cpu", dtype=torch.bfloat16)
            return {"x": x}, None

        def compute(tensors, operation):
            return torch.nn.functional.gelu(tensors["x"], approximate="tanh")

        shape_info = {"elements": BATCH_SIZE * SEQ_LEN * FFN_DIM}

        result = benchmark_with_metrics("GELU", prepare, compute, "elementwise", shape_info)

        if result:
            print(
                f"\n{result['name']}: {result['avg_ms']:.3f}ms, "
                f"{result['tflops']:.2f} TFLOPs, {result['memory_bw_gbs']:.2f} GB/s"
            )


# ============================================================================
# Report Generation
# ============================================================================


def generate_markdown_report(output_file="wan22_test_report.md"):
    """Generate Markdown report from test results."""
    if not TEST_RESULTS:
        return

    md = f"""# WAN 2.2 Operations Test Report

## Summary
| Metric | Value |
|--------|-------|
| Total Tests | {len(TEST_RESULTS)} |
| Total TFLOPs | {sum(r["tflops"] for r in TEST_RESULTS):.2f} |
| Avg Memory BW | {np.mean([r["memory_bw_gbs"] for r in TEST_RESULTS]):.2f} GB/s |

## Detailed Results

| Test Name | Latency (ms) | TFLOPs | Memory BW (GB/s) | Shape | Notes |
|-----------|--------------|--------|------------------|-------|-------|
"""

    for result in TEST_RESULTS:
        profiling = result.get("profiling_info", "")
        shape_info = result.get("shape_info", {})

        shape_str = ""
        if "batch" in shape_info and "seq_len" in shape_info:
            shape_str = f"B={shape_info.get('batch')}, S={shape_info.get('seq_len')}"
            if "in_features" in shape_info:
                shape_str += f", I={shape_info.get('in_features')}, O={shape_info.get('out_features')}"
        elif "in_channels" in shape_info and "out_channels" in shape_info:
            shape_str = (
                f"B={shape_info.get('batch')}, IC={shape_info.get('in_channels')}, OC={shape_info.get('out_channels')}"
            )
            if "kernel_size" in shape_info:
                shape_str += f", K={shape_info.get('kernel_size')}"
        elif "num_heads" in shape_info:
            shape_str = (
                f"B={shape_info.get('batch')}, H={shape_info.get('num_heads')}, "
                f"S={shape_info.get('seq_len')}, D={shape_info.get('head_dim')}"
            )
        elif "elements" in shape_info:
            shape_str = f"Elements={shape_info.get('elements')}"

        md += f"| {result['name']} | {result['avg_ms']:.3f} | {result['tflops']:.2f} | "
        md += f"{result['memory_bw_gbs']:.2f} | {shape_str} | {profiling} |\n"

    with open(output_file, "w") as f:
        f.write(md)

    print(f"\nMarkdown report generated: {output_file}")


def pytest_sessionfinish(session, exitstatus):
    """Generate report after all tests complete."""
    if TEST_RESULTS:
        generate_markdown_report()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
    if TEST_RESULTS:
        generate_markdown_report()
