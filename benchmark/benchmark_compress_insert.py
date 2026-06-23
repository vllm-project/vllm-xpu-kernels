# SPDX-License-Identifier: Apache-2.0
"""Benchmark for xpu_compress_insert_fp8mix kernel (SYCL and Triton).

Mirrors real input shapes from DeepSeek-V4:
  Layer A: compress_ratio=4,   overlap=1, kv_blk=64
  Layer B: compress_ratio=128, overlap=0, kv_blk=2

Usage:
    ZE_AFFINITY_MASK=0 python benchmark_compress_insert.py
    ZE_AFFINITY_MASK=0 python benchmark_compress_insert.py --layer B --num-tokens 8192
    ZE_AFFINITY_MASK=0 python benchmark_compress_insert.py --iters 100
"""
import argparse
import gc
import time

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401 - registers torch.ops._xpu_C ops

HEAD_SIZE = 512
ROPE_HEAD_DIM = 64
RMS_EPS = 1e-6
TOKEN_STRIDE = 576
SCALE_DIM = 8

LAYER_CONFIGS = {
    "A": dict(compress_ratio=4,   overlap=1, kv_cache_block_size=64,
              state_blk_sz=4, label="A_cr4"),
    "B": dict(compress_ratio=128, overlap=0, kv_cache_block_size=2,
              state_blk_sz=8, label="B_cr128"),
}

NUM_TOKENS_DEFAULT = [1, 4, 16, 64, 128, 256, 1024, 2048, 4096]


def make_inputs(cfg, num_tokens, device, seed=0, num_blocks=152522):
    cr = cfg["compress_ratio"]
    overlap = cfg["overlap"]
    kv_blk = cfg["kv_cache_block_size"]
    state_blk_sz = cfg["state_blk_sz"]
    state_width = (1 + overlap) * HEAD_SIZE
    gen = torch.Generator(device="cpu").manual_seed(seed)

    state_cache = (torch.randn(num_blocks, state_blk_sz, 2 * state_width,
                               generator=gen) * 0.1).to(device)
    k_cache_fp8mix = torch.zeros(
        num_blocks,
        kv_blk,
        TOKEN_STRIDE + SCALE_DIM,
        dtype=torch.uint8,
        device=device,
    )
    bt_cols = 200000 if cr == 128 else 16384
    block_table = torch.randint(0, num_blocks, (1, bt_cols),
                                dtype=torch.int32, generator=gen).to(device)

    # All tokens on boundary so kernel does real work every call.
    positions = ((torch.arange(num_tokens) + 1) * cr - 1).to(device)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    kv_slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    # Match production: DeepSeek-V4 compressor RMSNorm weight follows the model
    # dtype (bf16), so use bf16 here to exercise the in-kernel upcast path.
    rms_norm_weight = torch.ones(HEAD_SIZE, dtype=torch.bfloat16, device=device)

    max_pos = int(positions.max().item()) + cr + 1024
    half = ROPE_HEAD_DIM // 2
    freq = torch.arange(half, dtype=torch.float32) / half
    base = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
    cos_sin_cache = torch.cat([
        torch.cos(base * (10000 ** (-freq))),
        torch.sin(base * (10000 ** (-freq))),
    ], dim=1).to(device)

    common = dict(
        state_cache=state_cache,
        token_to_req=token_to_req,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        rms_norm_weight=rms_norm_weight,
        rms_eps=RMS_EPS,
        cos_sin_cache=cos_sin_cache,
        kv_slot_mapping=kv_slot_mapping,
        kv_blk=kv_blk,
        cr=cr,
        overlap=overlap,
        rope_head_dim=ROPE_HEAD_DIM,
    )

    fp8mix_inputs = (
        common["state_cache"], common["token_to_req"], common["positions"],
        common["slot_mapping"], common["block_table"],
        common["rms_norm_weight"], common["rms_eps"], common["cos_sin_cache"],
        k_cache_fp8mix, common["kv_slot_mapping"], common["kv_blk"],
        common["cr"], common["overlap"], common["rope_head_dim"],
        TOKEN_STRIDE, SCALE_DIM, int(k_cache_fp8mix.stride(0)),
    )

    # Keep tuple shape for callers that do: _, fp8mix_inputs = make_inputs(...)
    return None, fp8mix_inputs


def estimate_bytes(cfg, num_tokens, _path="fp8mix"):
    n_gather = (1 + cfg["overlap"]) * cfg["compress_ratio"]
    read_bytes = num_tokens * n_gather * HEAD_SIZE * 4 * 2
    write_bytes = num_tokens * (TOKEN_STRIDE + SCALE_DIM)
    return read_bytes + write_bytes


def sync_device(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def release_device_cache(device):
    # Avoid long sweeps accumulating allocator pressure and triggering DEVICE_LOST.
    gc.collect()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def benchmark_path(cfg, num_tokens, path, fp8mix_inputs, warmup=5, iters=50):
    device = fp8mix_inputs[0].device
    (
        state_cache,
        token_to_req,
        positions,
        slot_mapping,
        block_table,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        _k_cache_fp8mix,
        kv_slot_mapping,
        kv_blk,
        compress_ratio,
        overlap,
        rope_head_dim,
        token_stride,
        scale_dim,
        _kv_block_stride,
    ) = fp8mix_inputs

    # Allocate benchmark cache tensor here to align with compare script behavior.
    k_cache_fp8mix = torch.zeros(
        state_cache.size(0),
        kv_blk,
        token_stride + scale_dim,
        dtype=torch.uint8,
        device=device,
    )

    def run():
        torch.ops._xpu_C.xpu_compress_insert_fp8mix(
            state_cache,
            token_to_req,
            positions,
            slot_mapping,
            block_table,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            k_cache_fp8mix,
            kv_slot_mapping,
            kv_blk,
            compress_ratio,
            overlap,
            rope_head_dim,
            token_stride,
            scale_dim,
            int(k_cache_fp8mix.stride(0)),
        )

    for _ in range(warmup):
        run()
    sync_device(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    sync_device(device)
    t1 = time.perf_counter()

    lat_s = (t1 - t0) / iters
    nbytes = estimate_bytes(cfg, num_tokens, path)
    gbps = nbytes / lat_s / 1e9
    return lat_s * 1e6, gbps


def benchmark_triton_path(cfg, num_tokens, fp8mix_inputs,
                          warmup=5, iters=50):
    device = fp8mix_inputs[0].device
    if device.type not in ("cuda", "xpu"):
        raise ValueError("Triton path currently supports cuda/xpu devices only")

    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        _fused_kv_compress_norm_rope_insert_sparse_attn as fused_kernel,
    )

    (
        state_cache,
        token_to_req,
        positions,
        slot_mapping,
        block_table,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        k_cache_fp8mix,
        kv_slot_mapping,
        kv_blk,
        compress_ratio,
        overlap,
        rope_head_dim,
        token_stride,
        scale_dim,
        kv_block_stride,
    ) = fp8mix_inputs
    state_width = (1 + overlap) * HEAD_SIZE

    def run():
        fused_kernel[(int(positions.numel()),)](
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            token_to_req,
            positions,
            slot_mapping,
            block_table,
            block_table.stride(0),
            cfg["state_blk_sz"],
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            cos_sin_cache.stride(0),
            k_cache_fp8mix,
            kv_slot_mapping,
            kv_blk,
            HEAD_SIZE=HEAD_SIZE,
            TRITON_BLOCK_SIZE=1 << (HEAD_SIZE - 1).bit_length(),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=compress_ratio,
            OVERLAP=overlap,
            ROPE_HEAD_DIM=rope_head_dim,
            FP8_MAX=448.0,
            QUANT_BLOCK=64,
            TOKEN_STRIDE=token_stride,
            SCALE_DIM=scale_dim,
            KV_BLOCK_STRIDE=kv_block_stride,
            num_warps=4,
        )

    for _ in range(warmup):
        run()
    sync_device(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    sync_device(device)
    t1 = time.perf_counter()

    lat_s = (t1 - t0) / iters
    nbytes = estimate_bytes(cfg, num_tokens, "fp8mix")
    gbps = nbytes / lat_s / 1e9
    return lat_s * 1e6, gbps


def compare_sycl_triton_correctness(cfg, num_tokens,
                                    sycl_device="xpu", triton_device="cuda",
                                    num_blocks=152522,
                                    fp8_ulp_tol=1,
                                    fp8_max_hard_diff_bytes=0,
                                    rope_atol=1e-2,
                                    rope_max_diff_elems=0,
                                    ):
    sycl_dev = torch.device(sycl_device)
    triton_dev = torch.device(triton_device)
    if triton_dev.type not in ("cuda", "xpu"):
        raise ValueError("--triton-device must be cuda/xpu for correctness comparison")

    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        _fused_kv_compress_norm_rope_insert_sparse_attn as fused_kernel,
    )

    _, fp8mix_inputs = make_inputs(cfg, num_tokens, sycl_dev, seed=123, num_blocks=num_blocks)
    (state_cache_x,
     token_to_req_x,
     positions_x,
     slot_mapping_x,
     block_table_x,
     rms_w_x,
     rms_eps_x,
     cos_sin_x,
     _k_cache_x,
     kv_slot_x,
     kv_blk,
     cr,
     overlap,
     rope_dim,
     token_stride,
     scale_dim,
     _kv_stride_x) = fp8mix_inputs

    k_cache_x = None
    k_cache_c = None
    try:
        k_cache_x = torch.zeros(
            state_cache_x.size(0),
            kv_blk,
            token_stride + scale_dim,
            dtype=torch.uint8,
            device=sycl_dev,
        )
        torch.ops._xpu_C.xpu_compress_insert_fp8mix(
            state_cache_x,
            token_to_req_x,
            positions_x,
            slot_mapping_x,
            block_table_x,
            rms_w_x,
            rms_eps_x,
            cos_sin_x,
            k_cache_x,
            kv_slot_x,
            kv_blk,
            cr,
            overlap,
            rope_dim,
            token_stride,
            scale_dim,
            int(k_cache_x.stride(0)),
        )
        sync_device(sycl_dev)

        # If both sides run on XPU, reuse the same input tensors to avoid
        # xpu->cpu->xpu duplication that may trigger OOM/device_lost.
        if sycl_dev.type == "xpu" and triton_dev.type == "xpu":
            state_cache_c = state_cache_x
            token_to_req_c = token_to_req_x
            positions_c = positions_x
            slot_mapping_c = slot_mapping_x
            block_table_c = block_table_x
            rms_w_c = rms_w_x
            cos_sin_c = cos_sin_x
            kv_slot_c = kv_slot_x
            k_cache_c = torch.zeros_like(k_cache_x)
        else:
            state_cache_c = state_cache_x.cpu().to(triton_dev)
            token_to_req_c = token_to_req_x.cpu().to(triton_dev)
            positions_c = positions_x.cpu().to(triton_dev)
            slot_mapping_c = slot_mapping_x.cpu().to(triton_dev)
            block_table_c = block_table_x.cpu().to(triton_dev)
            rms_w_c = rms_w_x.cpu().to(triton_dev)
            cos_sin_c = cos_sin_x.cpu().to(triton_dev)
            kv_slot_c = kv_slot_x.cpu().to(triton_dev)
            k_cache_c = torch.zeros_like(k_cache_x.cpu()).to(triton_dev)

        fused_kernel[(int(positions_c.numel()),)](
            state_cache_c,
            state_cache_c.stride(0),
            state_cache_c.stride(1),
            token_to_req_c,
            positions_c,
            slot_mapping_c,
            block_table_c,
            block_table_c.stride(0),
            cfg["state_blk_sz"],
            rms_w_c,
            RMS_EPS,
            cos_sin_c,
            cos_sin_c.stride(0),
            k_cache_c,
            kv_slot_c,
            kv_blk,
            HEAD_SIZE=HEAD_SIZE,
            TRITON_BLOCK_SIZE=1 << (HEAD_SIZE - 1).bit_length(),
            STATE_WIDTH=(1 + overlap) * HEAD_SIZE,
            COMPRESS_RATIO=cr,
            OVERLAP=overlap,
            ROPE_HEAD_DIM=rope_dim,
            FP8_MAX=448.0,
            QUANT_BLOCK=64,
            TOKEN_STRIDE=token_stride,
            SCALE_DIM=scale_dim,
            KV_BLOCK_STRIDE=int(k_cache_c.stride(0)),
            num_warps=4,
        )
        sync_device(triton_dev)

        xcpu = k_cache_x.cpu()
        ccpu = k_cache_c.cpu()

        # Compare only slots that this launch writes, instead of the full cache.
        slot_ids = kv_slot_x.detach().to(torch.int64).cpu()
        unique_slots = torch.unique(slot_ids)
        active_blocks = unique_slots // kv_blk
        active_tokens = unique_slots % kv_blk

        # Cache is block-packed in memory:
        #   [token payload area: kv_blk * token_stride][scale area: kv_blk * scale_dim]
        # so per-token payload/scale must be gathered by flat offsets, not [:, token, :].
        x_block_view = xcpu.view(xcpu.size(0), -1)
        c_block_view = ccpu.view(ccpu.size(0), -1)
        x_blocks = x_block_view[active_blocks]
        c_blocks = c_block_view[active_blocks]

        token_offsets = active_tokens * token_stride
        token_cols = token_offsets[:, None] + torch.arange(token_stride, dtype=torch.int64)
        x_token = x_blocks.gather(1, token_cols)
        c_token = c_blocks.gather(1, token_cols)

        scale_base = kv_blk * token_stride
        scale_offsets = scale_base + active_tokens * scale_dim
        scale_cols = scale_offsets[:, None] + torch.arange(scale_dim, dtype=torch.int64)
        x_scale = x_blocks.gather(1, scale_cols)
        c_scale = c_blocks.gather(1, scale_cols)

        nope_head_dim = HEAD_SIZE - rope_dim

        # Exact bytes for fp8 payload.
        x_nope = x_token[:, :nope_head_dim]
        c_nope = c_token[:, :nope_head_dim]
        nope_neq = (x_nope != c_nope)
        fp8_diff_bytes = int(nope_neq.sum().item())
        fp8_total_bytes = int(x_nope.numel())
        # Treat <= fp8_ulp_tol byte delta as acceptable fp8 quantization jitter.
        fp8_absdiff = (x_nope.to(torch.int16) - c_nope.to(torch.int16)).abs()
        fp8_hard_neq = fp8_absdiff > int(fp8_ulp_tol)
        fp8_hard_diff_bytes = int(fp8_hard_neq.sum().item())

        # bf16-value comparison (not raw-byte) for rope payload.
        rope_bytes = token_stride - nope_head_dim
        x_rope_b = x_token[:, nope_head_dim:token_stride].contiguous()
        c_rope_b = c_token[:, nope_head_dim:token_stride].contiguous()
        x_rope = x_rope_b.reshape(-1, rope_bytes).view(torch.bfloat16).float()
        c_rope = c_rope_b.reshape(-1, rope_bytes).view(torch.bfloat16).float()
        rope_close = torch.isclose(x_rope, c_rope, rtol=0, atol=float(rope_atol))
        rope_diff_elems = int((~rope_close).sum().item())
        rope_total_elems = int(rope_close.numel())

        # Exact bytes for scales.
        scale_neq = (x_scale != c_scale)
        scale_diff_bytes = int(scale_neq.sum().item())
        scale_total_bytes = int(x_scale.numel())

        detail = None
        if fp8_hard_diff_bytes > 0:
            first = torch.nonzero(fp8_hard_neq, as_tuple=False)[0]
            row = int(first[0].item())
            idx = int(first[1].item())
            blk = int(active_blocks[row].item())
            token_in_block = int(active_tokens[row].item())
            detail = {
                "block": blk,
                "token_in_block": token_in_block,
                "region": "fp8_nope",
                "region_idx": idx,
                "sycl": int(x_nope[row, idx].item()),
                "triton": int(c_nope[row, idx].item()),
                "absdiff": int(fp8_absdiff[row, idx].item()),
            }
        elif rope_diff_elems > 0:
            first = torch.nonzero(~rope_close, as_tuple=False)[0]
            row = int(first[0].item())
            idx = int(first[1].item())
            blk = int(active_blocks[row].item())
            token_in_block = int(active_tokens[row].item())
            detail = {
                "block": blk,
                "token_in_block": token_in_block,
                "region": "rope_bf16_value",
                "region_idx": idx,
                "sycl": float(x_rope[row, idx].item()),
                "triton": float(c_rope[row, idx].item()),
            }
        elif scale_diff_bytes > 0:
            first = torch.nonzero(scale_neq, as_tuple=False)[0]
            row = int(first[0].item())
            idx = int(first[1].item())
            blk = int(active_blocks[row].item())
            token_in_block = int(active_tokens[row].item())
            detail = {
                "block": blk,
                "token_in_block": token_in_block,
                "region": "scale_u8",
                "region_idx": idx,
                "sycl": int(x_scale[row, idx].item()),
                "triton": int(c_scale[row, idx].item()),
            }

        stats = {
            "fp8_diff_bytes": fp8_diff_bytes,
            "fp8_total_bytes": fp8_total_bytes,
            "fp8_hard_diff_bytes": fp8_hard_diff_bytes,
            "fp8_ulp_tol": int(fp8_ulp_tol),
            "rope_atol": float(rope_atol),
            "rope_max_diff_elems": int(rope_max_diff_elems),
            "rope_diff_elems": rope_diff_elems,
            "rope_total_elems": rope_total_elems,
            "scale_diff_bytes": scale_diff_bytes,
            "scale_total_bytes": scale_total_bytes,
        }
        is_ok = (
            fp8_hard_diff_bytes <= int(fp8_max_hard_diff_bytes)
            and rope_diff_elems <= int(rope_max_diff_elems)
            and scale_diff_bytes == 0
        )

        return is_ok, stats, detail
    finally:
        del k_cache_x
        del k_cache_c
        release_device_cache(sycl_dev)
        release_device_cache(triton_dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["A", "B", "both"], default="both")
    ap.add_argument("--path", choices=["fp8mix"], default="fp8mix")
    ap.add_argument("--impl", choices=["sycl", "triton", "both"], default="both",
                    help="kernel implementation to benchmark")
    ap.add_argument("--sycl-device", default="xpu")
    ap.add_argument("--triton-device", default="xpu")
    ap.add_argument("--num-blocks", type=int, default=8192,
                    help="number of state/kv blocks for synthetic input")
    ap.add_argument("--check-correctness", action="store_true",
                    help="run standalone SYCL vs Triton cache-byte correctness check")
    ap.add_argument("--correctness-num-blocks", type=int, default=2048,
                    help="num_blocks used only for correctness comparison to reduce memory pressure")
    ap.add_argument("--correctness-max-tokens", type=int, default=4096,
                    help="skip correctness when num_tokens exceeds this threshold")
    ap.add_argument("--fp8-ulp-tol", type=int, default=1,
                    help="treat fp8 byte abs diff <= this value as acceptable (default: 1)")
    ap.add_argument("--fp8-max-hard-diff-bytes", type=int, default=0,
                    help="max allowed fp8 bytes beyond --fp8-ulp-tol (default: 0)")
    ap.add_argument("--rope-atol", type=float, default=1e-2,
                    help="absolute tolerance for rope bf16 value compare (default: 1e-2)")
    ap.add_argument("--rope-max-diff-elems", type=int, default=1,
                    help="max allowed rope elements beyond --rope-atol (default: 1)")
    ap.add_argument("--num-tokens", type=int, default=None,
                    help="single num_tokens value; default: sweep")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    layers = (["A", "B"] if args.layer == "both"
              else [args.layer])
    nt_list = ([args.num_tokens] if args.num_tokens
               else NUM_TOKENS_DEFAULT)

    if args.check_correctness:
        for lname in layers:
            cfg = LAYER_CONFIGS[lname]
            for nt in nt_list:
                if nt <= args.correctness_max_tokens:
                    # Correctness path uses slot_mapping/kv_slot_mapping = arange(num_tokens),
                    # so num_blocks must be large enough to cover max(kv_slot // kv_blk).
                    min_blocks_for_slots = (nt + cfg["kv_cache_block_size"] - 1) // cfg["kv_cache_block_size"]
                    corr_num_blocks = max(args.correctness_num_blocks, min_blocks_for_slots)
                    ok, stats, detail = compare_sycl_triton_correctness(
                        cfg,
                        nt,
                        sycl_device=args.sycl_device,
                        triton_device=args.triton_device,
                        num_blocks=corr_num_blocks,
                        fp8_ulp_tol=args.fp8_ulp_tol,
                        fp8_max_hard_diff_bytes=args.fp8_max_hard_diff_bytes,
                        rope_atol=args.rope_atol,
                        rope_max_diff_elems=args.rope_max_diff_elems,
                    )
                    status = "PASS" if ok else "FAIL"
                    print(
                        f"correctness[{cfg['label']} nt={nt}]: {status} "
                        f"fp8_diff={stats['fp8_diff_bytes']}/{stats['fp8_total_bytes']} "
                        f"fp8_hard_diff={stats['fp8_hard_diff_bytes']} "
                        f"(ulp_tol={stats['fp8_ulp_tol']}) "
                        f"rope_diff={stats['rope_diff_elems']}/{stats['rope_total_elems']} "
                        f"(atol={stats['rope_atol']:.6g}, max={stats['rope_max_diff_elems']}) "
                        f"scale_diff={stats['scale_diff_bytes']}/{stats['scale_total_bytes']}"
                    )
                    if detail is not None:
                        print(
                            "  first_mismatch: "
                            f"block={detail['block']} "
                            f"token_in_block={detail['token_in_block']} "
                            f"region={detail['region']} idx={detail['region_idx']} "
                            f"sycl={detail['sycl']} triton={detail['triton']}"
                        )
                else:
                    print(
                        f"correctness[{cfg['label']} nt={nt}]: SKIP "
                        f"(>{args.correctness_max_tokens} tokens)"
                    )
                release_device_cache(torch.device(args.sycl_device))
                release_device_cache(torch.device(args.triton_device))
            print()
        return

    if args.impl == "both":
        print(f"{'layer':<10} {'num_tok':>8} {'sycl_us':>10} {'triton_us':>10} {'sycl_GB/s':>10} {'triton_GB/s':>12} {'triton/sycl':>12}")
        print("-" * 96)
    else:
        print(f"{'layer':<10} {'impl':<8} {'path':<8} {'num_tok':>8} {'lat_us':>10} {'GB/s':>10} {'us/tok':>10}")
        print("-" * 66)

    for lname in layers:
        cfg = LAYER_CONFIGS[lname]
        for nt in nt_list:
            if args.impl == "both":
                _, sycl_inputs = make_inputs(
                    cfg,
                    nt,
                    torch.device(args.sycl_device),
                    num_blocks=args.num_blocks,
                )
                sycl_us, sycl_gbps = benchmark_path(
                    cfg, nt, "fp8mix", sycl_inputs,
                    warmup=args.warmup, iters=args.iters,
                )
                _, triton_inputs = make_inputs(
                    cfg,
                    nt,
                    torch.device(args.triton_device),
                    num_blocks=args.num_blocks,
                )
                triton_us, triton_gbps = benchmark_triton_path(
                    cfg, nt, triton_inputs,
                    warmup=args.warmup, iters=args.iters,
                )
                ratio = triton_us / sycl_us if sycl_us > 0 else float("inf")
                print(
                    f"{cfg['label']:<10} {nt:>8} {sycl_us:>10.2f} {triton_us:>10.2f} "
                    f"{sycl_gbps:>10.1f} {triton_gbps:>12.1f} {ratio:>12.2f}"
                )
                release_device_cache(torch.device(args.sycl_device))
                release_device_cache(torch.device(args.triton_device))
            else:
                if args.impl == "triton":
                    _, triton_inputs = make_inputs(
                        cfg,
                        nt,
                        torch.device(args.triton_device),
                        num_blocks=args.num_blocks,
                    )
                    lat_us, gbps = benchmark_triton_path(
                        cfg, nt, triton_inputs,
                        warmup=args.warmup, iters=args.iters,
                    )
                else:
                    _, sycl_inputs = make_inputs(
                        cfg,
                        nt,
                        torch.device(args.sycl_device),
                        num_blocks=args.num_blocks,
                    )
                    lat_us, gbps = benchmark_path(
                        cfg, nt, args.path, sycl_inputs,
                        warmup=args.warmup, iters=args.iters,
                    )
                print(
                    f"{cfg['label']:<10} {args.impl:<8} {args.path:<8} {nt:>8} {lat_us:>10.2f} "
                    f"{gbps:>10.1f} {lat_us/nt:>10.2f}"
                )
                if args.impl == "triton":
                    release_device_cache(torch.device(args.triton_device))
                else:
                    release_device_cache(torch.device(args.sycl_device))
        print()


if __name__ == "__main__":
    main()
