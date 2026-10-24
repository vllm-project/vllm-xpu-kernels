# SPDX-License-Identifier: Apache-2.0
"""Fused routed + shared-expert MoE forward for small-M decode on Xe2/Xe3.

Wraps ``torch.ops._xpu_C.moe_shared_fused_decode_interface``: routed experts
and the single sigmoid-gated shared expert run in one op (six kernels, no
per-call allocation).

FP16 or BF16 activations with fp8-e4m3 per-tensor (per-expert) weights and fp32
scalar scales are supported, with SiLU activation and no bias. The output is
the per-rank partial sum (tensor-parallel reduction is left to the caller).
"""

import torch

# Largest num_tokens * top_k handled by the kernel's routing map.
_MAX_ROUTED_ROWS = 512


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
    num_tokens * top_k <= 512, hidden_size % 16 == 0 and
    intermediate_size % 8 == 0.
    """
    return (
        is_available()
        and hidden_states_dtype in (torch.float16, torch.bfloat16)
        and weight_dtype == torch.float8_e4m3fn
        and num_tokens * top_k <= _MAX_ROUTED_ROWS
        and hidden_size % 16 == 0
        and intermediate_size % 8 == 0
    )


def moe_shared_fused_workspace_bytes(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    hidden_size: int,
    w13_n: int,
    w2_k: int,
) -> int:
    """Workspace size in bytes; must match the carve-out in the C++ op."""
    rows = num_tokens * top_k
    umt = min(rows, num_experts)

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

    tile_m selects the GEMM M-tile (8 or 16); 0 lets the op pick by M.
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
        assert tile_m in (0, 8, 16), "tile_m must be 0 (auto), 8 or 16"
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

    def workspace(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        ws = self._ws.get(num_tokens)
        if ws is None:
            nbytes = moe_shared_fused_workspace_bytes(
                num_tokens,
                self.top_k,
                self.num_experts,
                self.w13.shape[1],
                self.w13.shape[2],
                self.w2.shape[1],
            )
            ws = torch.empty(nbytes, dtype=torch.uint8, device=device)
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
        ws = self.workspace(hidden_states.shape[0], hidden_states.device)
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
