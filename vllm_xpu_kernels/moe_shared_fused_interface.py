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
# Limits of torch.ops._moe_C.router_gemv_topk_softmax (and of the fused
# residual-add + RMSNorm variant, which also needs K <= 6144).
_ROUTER_MAX_TOKENS = 16
_ROUTER_NORM_MAX_HIDDEN = 6144
_ROUTER_MAX_EXPERTS = 512
_ROUTER_MAX_TOP_K = 16


def is_available() -> bool:
    """Whether this build provides the fused decode op."""
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._xpu_C, "moe_shared_fused_decode_interface")


def router_is_available() -> bool:
    """Whether this build provides the fused decode router op."""
    try:
        import vllm_xpu_kernels._moe_C  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._moe_C, "router_gemv_topk_softmax")


def router_norm_supports(hidden_size: int) -> bool:
    """Whether the residual-add + RMSNorm router variant supports hidden_size
    (in addition to router_supports())."""
    return (
        router_is_available()
        and hasattr(torch.ops._moe_C, "router_resadd_norm_gemv_topk_softmax")
        and hidden_size <= _ROUTER_NORM_MAX_HIDDEN
    )


def router_supports(
    router_weight_dtype: torch.dtype,
    num_experts: int,
    top_k: int,
    hidden_size: int,
) -> bool:
    """Whether the fused FP16/BF16 router GEMV + softmax top-k supports
    this configuration (for up to 16 tokens per call)."""
    return (
        router_is_available()
        and router_weight_dtype in (torch.float16, torch.bfloat16)
        and num_experts <= _ROUTER_MAX_EXPERTS
        and 1 <= top_k <= min(_ROUTER_MAX_TOP_K, num_experts)
        and hidden_size % 128 == 0
    )


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

    With router_weight ([E, H], activation dtype) set, forward_routed() also
    computes the softmax top-k routing (renormalized if renormalize) in the op,
    for up to 16 tokens per call.
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
        router_weight: torch.Tensor | None = None,
        renormalize: bool = True,
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
        self.router_weight = (
            router_weight.contiguous() if router_weight is not None else None
        )
        self.renormalize = renormalize
        self._routing: dict[int, tuple[torch.Tensor, ...]] = {}

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

    def route(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax top-k routing of hidden_states with router_weight.

        Returns (topk_weights fp32, topk_ids int32), [M, top_k] each; the
        buffers are reused across calls with the same M.
        """
        assert self.router_weight is not None, "no router_weight given"
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

    def forward_resadd_norm_routed(
        self,
        output: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        normed_out: torch.Tensor,
        residual_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MoE block including its input residual-add + Gemma RMSNorm.

        residual_out = x + residual; normed_out = rms_norm(residual_out) *
        (1 + norm_weight); then routing and experts on normed_out. The norm
        and the routing run in one kernel. Returns (output, topk_ids).
        """
        assert self.router_weight is not None, "no router_weight given"
        logits, topk_weights, topk_ids = self._routing_buffers(
            x.shape[0], x.device)
        torch.ops._moe_C.router_resadd_norm_gemv_topk_softmax(
            x,
            residual,
            norm_weight,
            eps,
            self.router_weight,
            logits,
            topk_weights,
            topk_ids,
            normed_out,
            residual_out,
            self.renormalize,
        )
        out = self.forward(output, normed_out, topk_weights, topk_ids)
        return out, topk_ids
