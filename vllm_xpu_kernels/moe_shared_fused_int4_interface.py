# SPDX-License-Identifier: Apache-2.0
"""Fused routed + shared-expert MoE forward for small-M decode on Xe2/Xe3,
INT4-AWQ recipe.

Wraps ``torch.ops._xpu_C.moe_shared_fused_decode_int4_interface``: routed
experts and the single sigmoid-gated shared expert run in one op (six
kernels, no per-call allocation).

FP16 or BF16 activations with int8-packed 4-bit per-expert weights and
per-group scales (same dtype as the activations) are supported, with SiLU
activation and no bias. Per-group scales apply in-register in the shared
int4 mainloop.
The output is the per-rank partial sum (tensor-parallel reduction is left
to the caller).
"""

import torch

from vllm_xpu_kernels.moe_shared_fused_interface import (
    router_supports as _router_supports,
)

# Largest num_tokens * top_k handled by the kernel's routing map.
_MAX_ROUTED_ROWS = 512
# Group sizes the fused int4 mainloop instantiates.
_VALID_GROUP_SIZES = (32, 64, 128, 256)


def is_available() -> bool:
    """Whether this build provides the int4 fused decode op."""
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._xpu_C, "moe_shared_fused_decode_int4_interface")


def supports(
    hidden_states_dtype: torch.dtype,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
) -> bool:
    """Whether the int4 op supports this configuration.

    Requires FP16/BF16 activations, num_tokens * top_k <= 512,
    hidden_size % 256 == 0, intermediate_size % 8 == 0 and a group
    size of 32, 64, 128 or 256. Weight/scales dtypes and shapes are
    checked at the C++ entry.
    """
    return (
        is_available()
        and hidden_states_dtype in (torch.float16, torch.bfloat16)
        and num_tokens * top_k <= _MAX_ROUTED_ROWS
        and hidden_size % 256 == 0
        and intermediate_size % 8 == 0
        and group_size in _VALID_GROUP_SIZES
    )


def moe_shared_fused_int4_workspace_bytes(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    hidden_size: int,
    w13_n: int,
    w2_k: int,
) -> int:
    """Workspace size in bytes; must match the carve-out in the C++ op.

    Same activation-buffer geometry as the fp8 op (w13_n/w2_k are the
    logical, unpacked dims).
    """
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


class XpuMoESharedFusedDecodeInt4:
    """Holds the weights of one MoE block and a per-M persistent workspace.

    Weight layouts (all contiguous, [E, N, K] storage order, K packed):
      w13 [E, 2I, H/2] int8-packed 4-bit, w13_scales [E, 2I, H/G]
      w2  [E, H, I/2]  int8-packed 4-bit, w2_scales  [E, H, I/G]
    Shared experts are optional (None = routed-only; the caller then runs
    the dense shared MLP separately). When present:
      shared_w13 [2I, H/2] int8-packed, shared_w13_scale [1, 2I, H/G]
      shared_w2  [H, I/2]  int8-packed, shared_w2_scale  [1, H, I/G]
      shared_gate [H] with the activation dtype

    tile_m selects the GEMM M-tile (8 or 16); 0 lets the op pick by M.

    With router_weight ([E, H], activation dtype) set, forward_routed() also
    computes the softmax top-k routing via the shared router op, for up to
    16 tokens per call.
    """

    def __init__(
        self,
        w13: torch.Tensor,
        w13_scales: torch.Tensor,
        w2: torch.Tensor,
        w2_scales: torch.Tensor,
        shared_w13: torch.Tensor | None,
        shared_w13_scale: torch.Tensor | None,
        shared_w2: torch.Tensor | None,
        shared_w2_scale: torch.Tensor | None,
        shared_gate: torch.Tensor | None,
        top_k: int,
        tile_m: int = 0,
        router_weight: torch.Tensor | None = None,
        renormalize: bool = True,
    ):
        assert tile_m in (0, 8, 16), "tile_m must be 0 (auto), 8 or 16"
        self.w13 = w13.contiguous()
        self.w13_scales = w13_scales.contiguous()
        self.w2 = w2.contiguous()
        self.w2_scales = w2_scales.contiguous()
        self.shared_w13 = (
            shared_w13.contiguous() if shared_w13 is not None else None)
        self.shared_w13_scale = (
            shared_w13_scale.contiguous()
            if shared_w13_scale is not None
            else None)
        self.shared_w2 = (
            shared_w2.contiguous() if shared_w2 is not None else None)
        self.shared_w2_scale = (
            shared_w2_scale.contiguous()
            if shared_w2_scale is not None
            else None)
        self.shared_gate = (
            shared_gate.reshape(-1).contiguous()
            if shared_gate is not None
            else None)
        self.top_k = top_k
        self.num_experts = w13.shape[0]
        self.tile_m = tile_m
        self._ws: dict[int, torch.Tensor] = {}
        self.router_weight = (
            router_weight.contiguous() if router_weight is not None else None
        )
        self.renormalize = renormalize
        self._routing: dict[int, tuple[torch.Tensor, ...]] = {}

    def workspace(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        ws = self._ws.get(num_tokens)
        if ws is None:
            # [E, N, K] storage, K packed: hidden = N2, w13_n = N1.
            nbytes = moe_shared_fused_int4_workspace_bytes(
                num_tokens,
                self.top_k,
                self.num_experts,
                self.w2.shape[1],
                self.w13.shape[1],
                self.w2.shape[2] * 2,
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
        import vllm_xpu_kernels._xpu_C  # noqa: F401  (fused grid op)
        ws = self.workspace(hidden_states.shape[0], hidden_states.device)
        torch.ops._xpu_C.moe_shared_fused_decode_int4_interface(
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

    def route(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax top-k routing via the shared router op (1 launch).

        Returns (topk_weights fp32, topk_ids int32), [M, top_k] each;
        buffers are reused across calls with the same M. Only shapes the
        shared _moe_C router supports are accepted.
        """
        assert self.router_weight is not None, "no router_weight given"
        import vllm_xpu_kernels._moe_C  # noqa: F401  (router op)
        assert _router_supports(
            self.router_weight.dtype,
            self.num_experts,
            self.top_k,
            hidden_states.shape[1],
        ), "router op does not support this shape"
        logits, topk_weights, topk_ids = self._routing_buffers(
            hidden_states.shape[0], hidden_states.device
        )
        torch.ops._moe_C.router_gemv_topk_softmax(
            hidden_states,
            self.router_weight,
            logits,
            topk_weights,
            topk_ids,
            self.renormalize,
        )
        return topk_weights, topk_ids

    def _routing_buffers(
        self, m: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        bufs = self._routing.get(m)
        if bufs is None:
            bufs = (
                torch.empty(m,
                            self.num_experts,
                            dtype=torch.float32,
                            device=device),
                torch.empty(m, self.top_k, dtype=torch.float32, device=device),
                torch.empty(m, self.top_k, dtype=torch.int32, device=device),
            )
            self._routing[m] = bufs
        return bufs

    def forward_routed(
        self, output: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """route() + forward(); returns (output, topk_ids)."""
        topk_weights, topk_ids = self.route(hidden_states)
        out = self.forward(output, hidden_states, topk_weights, topk_ids)
        return out, topk_ids
