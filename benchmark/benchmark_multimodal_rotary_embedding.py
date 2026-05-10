# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for Multi-Modal Rotary Embedding (M-RoPE) SYCL kernel.

M-RoPE (used by e.g. Qwen2-VL) partitions the rotation dimensions into
sections so that each section can encode a different positional axis
(temporal / height / width for video/image tokens).

Usage:
    python benchmark/benchmark_multimodal_rotary_embedding.py
"""

import argparse

import torch
import triton
import triton.language as tl

from tests import register_ops as ops

# ─── Model-inspired M-RoPE configurations ───────────────────────────────────
# Format: (head_size, rot_dim, mrope_section, description)
# mrope_section values are in embed_dim (= rot_dim/2) units and must sum to
# rot_dim/2.
MROPE_CONFIGS = {
    "Qwen2-VL-7B": {
        "head_size": 128,
        "rot_dim": 128,
        "mrope_section": [16, 24, 24],  # temporal / height / width
        "num_heads": 28,
        "num_kv_heads": 4,
    },
    "Qwen2-VL-72B": {
        "head_size": 128,
        "rot_dim": 128,
        "mrope_section": [16, 24, 24],
        "num_heads": 64,
        "num_kv_heads": 8,
    },
    "Qwen2.5-VL-7B": {
        "head_size": 128,
        "rot_dim": 128,
        "mrope_section": [16, 24, 24],
        "num_heads": 28,
        "num_kv_heads": 4,
    },
    "Qwen3-VL-4B": {
        "head_size": 128,
        "rot_dim": 128,
        "mrope_section": [24, 20, 20],  # temporal / height / width
        "num_heads": 32,
        "num_kv_heads": 8,
    },
    "Custom-Small": {
        "head_size": 64,
        "rot_dim": 64,
        "mrope_section": [8, 12, 12],
        "num_heads": 8,
        "num_kv_heads": 2,
    },
}


# ─── Helper: build cos/sin cache ────────────────────────────────────────────
def compute_cos_sin_cache(max_position: int,
                          rot_dim: int,
                          base: float = 10000.0,
                          dtype: torch.dtype = torch.bfloat16,
                          device: str = "xpu") -> torch.Tensor:
    inv_freq = 1.0 / (base**(torch.arange(
        0, rot_dim, 2, dtype=torch.float, device="cpu") / rot_dim))
    t = torch.arange(max_position, dtype=torch.float, device="cpu")
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_pos, rot_dim//2]
    cache = torch.cat((freqs.cos(), freqs.sin()),
                      dim=-1)  # [max_pos, rot_dim]
    return cache.to(dtype=dtype, device=device)


# ─── vLLM PyTorch-native M-RoPE (from vllm MRotaryEmbedding.forward_native) ─
def mrope_vllm_torch(
    positions: torch.Tensor,       # [num_sections, num_tokens]
    query: torch.Tensor,           # [num_tokens, num_heads, head_size]
    key: torch.Tensor,             # [num_tokens, num_kv_heads, head_size]
    cos_sin_cache: torch.Tensor,   # [max_position, rot_dim]
    mrope_section: list[int],
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's PyTorch-native M-RoPE (MRotaryEmbedding.forward_native).

    Works on flat [num_tokens, hidden] layout internally, same as vLLM.
    """
    num_tokens = query.shape[0]
    head_size = query.shape[2]
    rot_dim = cos_sin_cache.shape[1]

    # Flatten to [num_tokens, num_heads * head_size] as vLLM does
    q_flat = query.reshape(num_tokens, -1).clone()
    k_flat = key.reshape(num_tokens, -1).clone()

    # cos_sin_cache[positions] → [num_sections, num_tokens, rot_dim]
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)  # rot_dim//2

    # Merge sections: for each section s, take cos[s, :, lo:hi]
    cos = torch.cat(
        [m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
        dim=-1,
    )  # [num_tokens, rot_dim//2]
    sin = torch.cat(
        [m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
        dim=-1,
    )  # [num_tokens, rot_dim//2]

    # Reshape to [num_tokens, num_heads, head_size]
    q_out = q_flat.view(num_tokens, -1, head_size)
    k_out = k_flat.view(num_tokens, -1, head_size)
    q_rot = q_out[..., :rot_dim]
    q_pass = q_out[..., rot_dim:]
    k_rot = k_out[..., :rot_dim]
    k_pass = k_out[..., rot_dim:]

    # Apply rotation (ApplyRotaryEmb.forward_static logic)
    cos_e = cos.unsqueeze(-2).to(q_rot.dtype)  # [num_tokens, 1, rot_dim//2]
    sin_e = sin.unsqueeze(-2).to(q_rot.dtype)

    if is_neox:
        q1, q2 = torch.chunk(q_rot, 2, dim=-1)
        q_rot = torch.cat((q1 * cos_e - q2 * sin_e,
                           q2 * cos_e + q1 * sin_e), dim=-1)
        k1, k2 = torch.chunk(k_rot, 2, dim=-1)
        k_rot = torch.cat((k1 * cos_e - k2 * sin_e,
                           k2 * cos_e + k1 * sin_e), dim=-1)
    else:
        q1 = q_rot[..., ::2]
        q2 = q_rot[..., 1::2]
        q_rot = torch.stack((q1 * cos_e - q2 * sin_e,
                             q2 * cos_e + q1 * sin_e), dim=-1).flatten(-2)
        k1 = k_rot[..., ::2]
        k2 = k_rot[..., 1::2]
        k_rot = torch.stack((k1 * cos_e - k2 * sin_e,
                             k2 * cos_e + k1 * sin_e), dim=-1).flatten(-2)

    q_out = torch.cat((q_rot, q_pass), dim=-1)
    k_out = torch.cat((k_rot, k_pass), dim=-1)

    return q_out, k_out


# ─── torch.compile'd vLLM torch M-RoPE ───────────────────────────────────────
mrope_vllm_compiled = torch.compile(mrope_vllm_torch)


# ─── vLLM Triton M-RoPE (from vllm triton_mrope) ────────────────────────────
@triton.jit
def _triton_mrope_forward(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    half_rd = rd // 2
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (
            cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (
            cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # Left half of head
    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd
        + tl.arange(0, pad_hd // 2)[None, :])
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd
        + tl.arange(0, pad_hd // 2)[None, :])
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2)

    q_tile_1 = tl.load(
        q_ptr + first_half_q_offsets, mask=first_q_mask, other=0
    ).to(sin_row.dtype)
    k_tile_1 = tl.load(
        k_ptr + first_half_k_offsets, mask=first_k_mask, other=0
    ).to(sin_row.dtype)

    # Right half of head
    second_half_q_offsets = first_half_q_offsets + (rd // 2)
    second_half_k_offsets = first_half_k_offsets + (rd // 2)

    q_tile_2 = tl.load(
        q_ptr + second_half_q_offsets, mask=first_q_mask, other=0
    ).to(sin_row.dtype)
    k_tile_2 = tl.load(
        k_ptr + second_half_k_offsets, mask=first_k_mask, other=0
    ).to(sin_row.dtype)

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=first_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=first_k_mask)


def mrope_vllm_triton(
    positions: torch.Tensor,       # [num_sections, num_tokens]
    query: torch.Tensor,           # [num_tokens, num_heads, head_size]
    key: torch.Tensor,             # [num_tokens, num_kv_heads, head_size]
    cos_sin_cache: torch.Tensor,   # [max_position, rot_dim]
    mrope_section: list[int],
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's Triton fused M-RoPE kernel.

    Only supports NeoX-style (is_neox=True) and exactly 3 sections.
    """
    num_tokens = query.shape[0]
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    head_size = query.shape[2]
    rot_dim = cos_sin_cache.shape[1]

    # Flatten to [num_tokens, num_heads * head_size]
    q_flat = query.reshape(num_tokens, -1).clone().contiguous()
    k_flat = key.reshape(num_tokens, -1).clone().contiguous()

    # cos_sin_cache[positions] → [3, num_tokens, rot_dim]
    cos_sin = cos_sin_cache[positions]  # [3, num_tokens, rot_dim]
    cos, sin = cos_sin.chunk(2, dim=-1)  # each [3, num_tokens, rot_dim//2]
    cos = cos.contiguous()
    sin = sin.contiguous()

    pad_hd = triton.next_power_of_2(head_size)
    pad_n_qh = triton.next_power_of_2(num_heads)
    pad_n_kh = triton.next_power_of_2(num_kv_heads)

    _triton_mrope_forward[(num_tokens,)](
        q_flat,
        k_flat,
        cos,
        sin,
        num_tokens,
        num_heads,
        num_kv_heads,
        head_size,
        rot_dim,
        pad_n_qh,
        pad_n_kh,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        False,  # is_interleaved (NeoX = not interleaved)
    )

    return (
        q_flat.view(num_tokens, num_heads, head_size),
        k_flat.view(num_tokens, num_kv_heads, head_size),
    )


# ─── XPU kernel wrapper ─────────────────────────────────────────────────────
def mrope_xpu_kernel(positions: torch.Tensor, query: torch.Tensor,
                     key: torch.Tensor, head_size: int,
                     cos_sin_cache: torch.Tensor, is_neox: bool,
                     mrope_section: list[int]) -> None:
    """In-place M-RoPE via the SYCL kernel."""
    ops.multimodal_rotary_embedding(positions, query, key, head_size,
                                    cos_sin_cache, is_neox, mrope_section)


# ─── Benchmark ───────────────────────────────────────────────────────────────
def get_benchmark(config: dict, dtype: torch.dtype, is_neox: bool,
                  max_position: int, num_tokens_list: list[int]):
    head_size = config["head_size"]
    rot_dim = config["rot_dim"]
    mrope_section = config["mrope_section"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    num_sections = len(mrope_section)

    if is_neox:
        providers = ["vllm_torch", "vllm_t.compile",
                      "vllm_triton", "xpu_kernel"]
        provider_names = ["torch eager",
                          "torch compile", "Triton", "XPU Kernel"]
    else:
        providers = ["vllm_torch", "vllm_t.compile",
                      "xpu_kernel"]
        provider_names = ["torch eager",
                          "torch compile", "XPU Kernel"]
    styles = [("red", "-."), ("crimson", "-."),
              ("purple", ":"), ("green", "-")]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=num_tokens_list,
            line_arg="provider",
            line_vals=providers,
            line_names=provider_names,
            styles=styles,
            ylabel="us",
            plot_name=(
                f"mrope-{num_heads}h-{num_kv_heads}kv-"
                f"hs{head_size}-rd{rot_dim}-"
                f"sec{'_'.join(map(str, mrope_section))}-"
                f"{'neox' if is_neox else 'gptj'}"),
            args={},
        ))
    def benchmark(num_tokens, provider):
        cos_sin_cache = compute_cos_sin_cache(max_position, rot_dim,
                                              dtype=dtype)
        positions = torch.stack([
            torch.randint(0, max_position, (num_tokens, ), device="xpu")
            for _ in range(num_sections)
        ])
        query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype,
                             device="xpu")
        key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype,
                           device="xpu")

        quantiles = [0.5, 0.2, 0.8]

        if provider == "vllm_torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: mrope_vllm_torch(
                    positions, query.clone(), key.clone(), cos_sin_cache,
                    mrope_section, is_neox),
                quantiles=quantiles,
            )
        elif provider == "vllm_t.compile":
            # Warmup compile
            mrope_vllm_compiled(positions, query.clone(), key.clone(),
                                cos_sin_cache, mrope_section, is_neox)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: mrope_vllm_compiled(
                    positions, query.clone(), key.clone(), cos_sin_cache,
                    mrope_section, is_neox),
                quantiles=quantiles,
            )
        elif provider == "vllm_triton":
            # vLLM triton kernel only supports NeoX + 3 sections
            assert is_neox and len(mrope_section) == 3, \
                "vLLM triton mrope only supports NeoX + 3 sections"
            # Warmup triton JIT
            mrope_vllm_triton(positions, query.clone(), key.clone(),
                              cos_sin_cache, mrope_section, is_neox)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: mrope_vllm_triton(
                    positions, query.clone(), key.clone(), cos_sin_cache,
                    mrope_section, is_neox),
                quantiles=quantiles,
            )
        else:  # xpu_kernel
            def run_kernel():
                q = query.clone()
                k = key.clone()
                mrope_xpu_kernel(positions, q, k, head_size, cos_sin_cache,
                                 is_neox, mrope_section)
            ms, min_ms, max_ms = triton.testing.do_bench(
                run_kernel,
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def parse_benchmark_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Multi-Modal Rotary Embedding (M-RoPE)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(MROPE_CONFIGS.keys()),
        help="Run benchmark for a specific model config. "
        "If not set, benchmarks all configs.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=None,
        help="List of num_tokens values to benchmark. "
        "Default: powers of 2 from 1 to 8192.",
    )
    parser.add_argument(
        "--max-position",
        type=int,
        default=8192,
        help="Maximum position for cos/sin cache (default: 8192)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type (default: bfloat16)",
    )
    parser.add_argument(
        "--is-neox",
        action="store_true",
        default=True,
        help="Use NeoX-style rotation (default: True)",
    )
    parser.add_argument(
        "--no-neox",
        action="store_true",
        help="Use GPT-J style rotation instead of NeoX",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/mrope/",
        help="Path to save benchmark results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_benchmark_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    is_neox = not args.no_neox

    if args.num_tokens is not None:
        num_tokens_list = args.num_tokens
    else:
        num_tokens_list = [2**i for i in range(0, 14)]  # 1 to 8192

    configs_to_run = {}
    if args.config:
        configs_to_run[args.config] = MROPE_CONFIGS[args.config]
    else:
        configs_to_run = MROPE_CONFIGS

    print("=" * 70)
    print("Multi-Modal Rotary Embedding (M-RoPE) Benchmark")
    print("=" * 70)
    print(f"  dtype:        {args.dtype}")
    print(f"  is_neox:      {is_neox}")
    print(f"  max_position: {args.max_position}")
    print(f"  num_tokens:   {num_tokens_list}")
    print(f"  configs:      {list(configs_to_run.keys())}")
    print()

    # ── Performance benchmark ──
    for name, cfg in configs_to_run.items():
        print(f"--- Benchmarking: {name} ---")
        print(f"    heads={cfg['num_heads']}, kv_heads={cfg['num_kv_heads']}, "
              f"head_size={cfg['head_size']}, rot_dim={cfg['rot_dim']}, "
              f"sections={cfg['mrope_section']}")

        bench = get_benchmark(cfg, dtype, is_neox, args.max_position,
                              num_tokens_list)
        bench.run(print_data=True, save_path=args.save_path)
        print()
