# SPDX-License-Identifier: Apache-2.0
"""Load the optional MiniCPM5 resident decoder library only when requested."""

import hashlib
from pathlib import Path
from threading import Lock

import torch

ABI_VERSION = 3
OPS = (
    "abi_version",
    "supported",
    "control_words",
    "health_offset",
    "mailbox_bytes",
    "validate",
    "decode",
)
_LOCK = Lock()


def load():
    with _LOCK:
        ops = torch.ops.minicpm5_mono
        paths = list(Path(__file__).parent.glob("_minicpm5_mono_C*.so"))
        if len(paths) != 1:
            raise ImportError(
                "Install a Xe2 wheel containing the MiniCPM5 mono extension "
                "(MINICPM5_MONO_KERNELS_ENABLED)"
            )
        path = str(paths[0].resolve())
        loaded = {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
        if any(hasattr(ops, name) for name in OPS) and path not in loaded:
            raise RuntimeError("MiniCPM5 mono ops came from another library")
        if not all(hasattr(ops, name) for name in OPS):
            torch.ops.load_library(path)
        if (
            not all(hasattr(ops, name) for name in OPS)
            or ops.abi_version() != ABI_VERSION
        ):
            raise RuntimeError("MiniCPM5 mono library ABI mismatch")
        return ops


def library_info():
    """Report the exact loaded native artifact without querying any device."""
    ops = load()
    path = next(Path(__file__).parent.glob("_minicpm5_mono_C*.so")).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "abi": ops.abi_version(),
        "max_context_len": (
            ops.max_context_len() if hasattr(ops, "max_context_len") else 32768
        ),
        "rotary_cache_version": (
            ops.rotary_cache_version()
            if hasattr(ops, "rotary_cache_version")
            else 0
        ),
        "supported_rows": (
            list(ops.supported_rows())
            if hasattr(ops, "supported_rows")
            else [1, 2, 4, 8]
        ),
        "activation_dtypes": (
            list(ops.activation_dtypes())
            if hasattr(ops, "activation_dtypes")
            else ["bfloat16"]
        ),
        "stage_timing": (
            ops.stage_timing_enabled()
            if hasattr(ops, "stage_timing_enabled")
            else False
        ),
        "padding_mask_version": (
            ops.padding_mask_version()
            if hasattr(ops, "padding_mask_version")
            else 0
        ),
        "padding_mask_rows": (
            list(ops.padding_mask_rows())
            if hasattr(ops, "padding_mask_rows")
            else []
        ),
        "control_words": ops.control_words(),
        "attention_overlap": (
            ops.attention_overlap_enabled()
            if hasattr(ops, "attention_overlap_enabled")
            else None
        ),
        "attention_partial_groups": {
            str(tp): ops.attention_partial_groups(tp)
            if hasattr(ops, "attention_partial_groups")
            else 32
            for tp in (1, 2)
        },
        "attention_variant": (
            ops.attention_variant()
            if hasattr(ops, "attention_variant")
            else "scalar-qk-per-head-fp32-softmax-pv"
        ),
        "router_variant": (
            ops.router_variant()
            if hasattr(ops, "router_variant")
            else "scalar-bf16-weight-fp32-accumulate"
        ),
        "router_logits_dtype": (
            ops.router_logits_dtype()
            if hasattr(ops, "router_logits_dtype")
            else "activation"
        ),
        "mlp_phase_variant": (
            ops.mlp_phase_variant()
            if hasattr(ops, "mlp_phase_variant")
            else "shared-activation-global-phase14"
        ),
        "projection_variant": (
            ops.projection_variant()
            if hasattr(ops, "projection_variant")
            else "scalar-fp32-rows1-2-4-8"
        ),
        "expert_union_variant": (
            ops.expert_union_variant()
            if hasattr(ops, "expert_union_variant")
            else "repeated-global-expert-scans-for-matrix-path"
        ),
        "fp8_decode_variant": (
            ops.fp8_decode_variant()
            if hasattr(ops, "fp8_decode_variant")
            else "exact-bf16-clz-subnormals"
        ),
        "runtime_variant": (
            ops.runtime_variant()
            if hasattr(ops, "runtime_variant")
            else "per-control-health-serial-page-validation"
        ),
        "handoff_variant": (
            ops.handoff_variant() if hasattr(ops, "handoff_variant") else None
        ),
    }


def workspace(tp, device, dtype=torch.bfloat16):
    """One scratch for all layers on one stream, reserved before capture."""
    if tp not in (1, 2):
        raise ValueError("MiniCPM5 mono supports TP1/TP2")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("MiniCPM5 mono supports BF16/FP16 activations")
    q, kv, inter = 32 // tp, 2 // tp, 512 // tp
    ops = load()
    partial_groups = (
        ops.attention_partial_groups(tp)
        if hasattr(ops, "attention_partial_groups")
        else 32
    )
    layouts = [
        ((8, 2048), dtype),
        ((8, 2 * q * 128 + 2 * kv * 128), dtype),
        ((8, q * 128), dtype),
        ((8, partial_groups, q, 130), torch.float32),
        ((8, 17 * inter), dtype),
        ((8, 160), torch.float32),
        ((8, 16), torch.float32),
        ((8, 16), torch.int32),
        ((8, 16, 2 * inter, 8), torch.float32),
        ((8, 17, 2048), dtype),
    ]
    return [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in layouts
    ]
