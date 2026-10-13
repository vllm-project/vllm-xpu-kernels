# SPDX-License-Identifier: Apache-2.0
"""Fused routed + shared-expert MoE forward for small-M decode on Xe2/Xe3.

Wraps ``torch.ops._xpu_C.moe_shared_fused_decode_interface``: routed experts
and the single sigmoid-gated shared expert run in one op (six kernels, no
per-call allocation).

FP16 or BF16 activations with fp8-e4m3 per-tensor (per-expert) weights and fp32
scalar scales are supported, with SiLU activation and no bias. Padding expert
IDs (-1) contribute zero to routed experts; the shared expert still runs.
Valid expert IDs must be distinct within each token's top-k routes.
The output is the per-rank partial sum (tensor-parallel reduction is left
to the caller).
"""

import torch

# Largest num_tokens * top_k handled by the kernel's routing map.
_MAX_ROUTED_ROWS = 512
# Xe block-2D descriptors use 24-bit byte widths/pitches.
_MAX_BLOCK_2D_BYTES = (1 << 24) - 1


def is_available() -> bool:
    """Whether this build provides the fused decode op."""
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._xpu_C, "moe_shared_fused_decode_interface")


def supports(
    hidden_states_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    """Whether the op supports this configuration.

    Requires FP16/BF16 activations, fp8-e4m3 per-tensor weights,
    positive dimensions, num_tokens * top_k <= 512, and H/I multiples of
    32. Each compact routed row must start at a 64-byte-aligned address.
    """
    return (
        is_available()
        and hidden_states_dtype in (torch.float16, torch.bfloat16)
        and weight_dtype == torch.float8_e4m3fn
        and num_tokens > 0
        and top_k > 0
        and num_tokens * top_k <= _MAX_ROUTED_ROWS
        and hidden_size > 0
        and hidden_size % 32 == 0
        and hidden_size <= _MAX_BLOCK_2D_BYTES // 2
        and intermediate_size > 0
        and intermediate_size % 32 == 0
        and intermediate_size <= _MAX_BLOCK_2D_BYTES // 4
    )


def moe_shared_fused_workspace_bytes(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    w13_n: int,
    w2_k: int,
) -> int:
    """Workspace size in bytes; must match the carve-out in the C++ op."""
    if num_tokens <= 0 or top_k <= 0:
        raise ValueError("num_tokens and top_k must be positive")
    rows = num_tokens * top_k
    if rows > _MAX_ROUTED_ROWS:
        raise ValueError("num_tokens * top_k must not exceed 512")
    if (hidden_size <= 0 or w2_k <= 0 or w13_n != 2 * w2_k
            or hidden_size % 32 or w2_k % 32
            or hidden_size > _MAX_BLOCK_2D_BYTES // 2
            or w2_k > _MAX_BLOCK_2D_BYTES // 4):
        raise ValueError(
            "H and I must be positive multiples of 32 within the block-2D "
            "limits, and w13_n must equal 2 * I")
    # A hot expert can span multiple tiles; reserve one tile per routed row.
    umt = rows

    def align(b: int) -> int:
        return (b + 255) // 256 * 256

    o_wperm = align(4 * (3 * umt + 1 + rows))
    o_gate = align(o_wperm + 4 * rows)
    o_ap = align(o_gate + 4 * num_tokens)
    return o_ap + 2 * (
        rows * hidden_size
        + (rows + num_tokens) * w13_n
        + (rows + num_tokens) * w2_k
        + rows * hidden_size
        + num_tokens * hidden_size
    )


class XpuMoESharedFusedDecode:
    """Holds the weights of one MoE block and a per-M persistent workspace.

    Weight layouts:
      w13 [E, H, 2I] fp8-e4m3, w13_scales [E] fp32
      w2  [E, I, H]  fp8-e4m3, w2_scales  [E] fp32
      shared_w13 [H, 2I] fp8-e4m3, shared_w13_scale [1] fp32
      shared_w2  [I, H]  fp8-e4m3, shared_w2_scale  [1] fp32
      shared_gate [H] or [1, H] with the activation dtype

    Matrices and hidden states require 64-byte-aligned base addresses;
    shared_gate requires 32-byte alignment. Contiguous storage-offset views
    do not necessarily meet those native requirements.

    tile_m selects the GEMM M-tile (8 or 16); 0 lets the op pick by M.

    Each instance is bound to its weights' device. Calls use the current XPU
    stream; calls with the same M share workspace and must be ordered. Use
    separate instances for overlapping calls on different streams.
    """

    def __init__(
        self,
        w13: torch.Tensor,
        w13_scales: torch.Tensor,
        w2: torch.Tensor,
        w2_scales: torch.Tensor,
        shared_w13: torch.Tensor,
        shared_w13_scale: torch.Tensor,
        shared_w2: torch.Tensor,
        shared_w2_scale: torch.Tensor,
        shared_gate: torch.Tensor,
        top_k: int,
        tile_m: int = 0,
    ):
        if tile_m not in (0, 8, 16):
            raise ValueError("tile_m must be 0 (auto), 8 or 16")
        self.device = w13.device
        if self.device.type != "xpu":
            raise ValueError("w13 must be an XPU tensor")
        for name, tensor in (
            ("w13_scales", w13_scales),
            ("w2", w2),
            ("w2_scales", w2_scales),
            ("shared_w13", shared_w13),
            ("shared_w13_scale", shared_w13_scale),
            ("shared_w2", shared_w2),
            ("shared_w2_scale", shared_w2_scale),
            ("shared_gate", shared_gate),
        ):
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} must be on the same XPU device as w13")
        if w13.dim() != 3 or w2.dim() != 3:
            raise ValueError("routed weights must be rank three")
        num_experts, hidden, w13_n = w13.shape
        if not (0 < num_experts <= 512 and 0 < top_k <= num_experts):
            raise ValueError("require 1 <= top_k <= num_experts <= 512")
        if w2.shape[0] != num_experts or w2.shape[2] != hidden:
            raise ValueError(
                "routed weight shapes must be [E, H, 2I], [E, I, H]")
        # Validate dimensions before copying weights or allocating scratch.
        moe_shared_fused_workspace_bytes(1, top_k, hidden, w13_n, w2.shape[1])
        self.w13 = w13.contiguous()
        self.w13_scales = w13_scales.contiguous()
        self.w2 = w2.contiguous()
        self.w2_scales = w2_scales.contiguous()
        self.shared_w13 = shared_w13.contiguous()
        self.shared_w13_scale = shared_w13_scale.reshape(1).contiguous()
        self.shared_w2 = shared_w2.contiguous()
        self.shared_w2_scale = shared_w2_scale.reshape(1).contiguous()
        self.shared_gate = shared_gate.reshape(-1).contiguous()
        self.top_k = top_k
        self.num_experts = w13.shape[0]
        self.tile_m = tile_m
        self._ws: dict[int, torch.Tensor] = {}

    def workspace(self, num_tokens: int) -> torch.Tensor:
        """Return persistent scratch on this instance's weights' device."""
        ws = self._ws.get(num_tokens)
        if ws is None:
            nbytes = moe_shared_fused_workspace_bytes(
                num_tokens,
                self.top_k,
                self.w13.shape[1],
                self.w13.shape[2],
                self.w2.shape[1],
            )
            ws = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            self._ws[num_tokens] = ws
        return ws

    def forward(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Writes routed + gated shared (per-rank partial sum) into output."""
        if hidden_states.device != self.device:
            raise ValueError(
                "hidden_states must be on the same XPU device as w13")
        ws = self.workspace(hidden_states.shape[0])
        torch.ops._xpu_C.moe_shared_fused_decode_interface(
            output,
            hidden_states,
            topk_ids,
            topk_weights,
            self.w13,
            self.w13_scales,
            self.w2,
            self.w2_scales,
            self.shared_w13,
            self.shared_w13_scale,
            self.shared_w2,
            self.shared_w2_scale,
            self.shared_gate,
            ws,
            self.tile_m,
        )
        return output
