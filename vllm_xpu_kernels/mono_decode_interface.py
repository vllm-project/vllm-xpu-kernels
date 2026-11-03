# SPDX-License-Identifier: Apache-2.0
"""Decode-only mono decoder layers for Qwen3.5 / Qwen3.6 MoE on Xe2 at TP2.

Wraps the ``torch.ops._xpu_C.mono_*`` ops (contracts in
``csrc/xpu/mono_decode/mono_decode_interface.h``): each decoder layer of a
decode step of 1, 2, 4 or 8 rows, or of a speculative-decoding verify step,
runs as one persistent kernel (GatedDeltaNet or full attention, then the MoE
with the shared expert), with both TP all-reduces inside the kernel through a
mailbox in the peer GPU's memory (Level Zero IPC). The MTP drafter's prologue
(two norms and the fc) and the vocabulary-parallel top-k candidate exchange
are ops of the same kind.

The kernels are built for the 35B-A3B shapes: hidden 2048, 256 experts of
intermediate 512 (top-8) plus one gated shared expert, fp16 activations,
fp8-e4m3 per-tensor weights, an fp16 KV cache with blocks of a multiple of 64
tokens and an fp32 recurrent state. They read the caller's weights and caches
in place, launch on the current XPU stream and allocate nothing per call.

Every wait for the peer GPU gives up after about 12 s and records the failure
in the device's health words (``health``), so a stalled peer fails both ranks
instead of hanging them; the caller checks the words and stops every rank.
"""

import torch

# The ops of the mono decode ABI.
OPS = (
    "mono_decode_supported",
    "mono_max_context",
    "mono_health",
    "mono_ipc_create",
    "mono_ipc_export",
    "mono_ipc_open",
    "mono_ipc_ptr",
    "mono_mbox_bytes",
    "mono_register_layer",
    "mono_decode_layer",
    "mono_decode_layer_spec",
    "mono_register_mtp",
    "mono_mtp_prologue",
    "mono_lmhead_mbox_bytes",
    "mono_shard_topk_max_k",
    "mono_shard_topk",
)

# Health words (``health``): FAILED is sticky; TIMEOUTS counts the waits that
# gave up.
HEALTH_FAILED, HEALTH_TIMEOUTS = 0, 1

# Decode rows of the decode kernels; rows x tokens of the verify kernels.
DECODE_ROWS = (1, 2, 4, 8)
VERIFY_SHAPES = ((2, 2), (3, 3), (4, 2), (4, 4), (6, 3), (8, 2), (8, 4),
                 (8, 8))
# Rows of the MTP drafter's prologue kernel.
MTP_PROLOGUE_ROWS = (1, 2, 3, 4, 6, 8)


def is_available() -> bool:
    """Whether this build provides the mono decode ops (built with
    ``MONO_DECODE_KERNELS_ENABLED`` and Xe2 kernels)."""
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        return False
    return all(hasattr(torch.ops._xpu_C, name) for name in OPS)


def supports(device: torch.device | None = None) -> bool:
    """Whether the ops run on ``device`` (defaults to the current XPU): an Xe2
    (Battlemage) GPU with at least 32 Xe cores (the layer kernels keep 32
    work-groups resident)."""
    if device is not None and device.type != "xpu":
        return False
    index = -1 if device is None or device.index is None else device.index
    return (is_available() and torch.xpu.is_available()
            and torch.ops._xpu_C.mono_decode_supported(index))


def max_context() -> int:
    """Longest context (cached tokens of a request) the attention kernels
    serve. The kernels read sequence lengths from device memory and do not
    check them: callers must not exceed this."""
    return int(torch.ops._xpu_C.mono_max_context())


def health(device: int = -1) -> torch.Tensor:
    """The device's int32 [8] health words (device memory, never freed)."""
    return torch.ops._xpu_C.mono_health(device)
