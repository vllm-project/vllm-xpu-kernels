# SPDX-License-Identifier: Apache-2.0
"""Fused routed + shared-expert MoE forward for small-M decode on Xe2/Xe3,
INT4-AWQ recipe.

Wraps ``torch.ops._xpu_C.moe_shared_fused_decode_int4_interface``: routed
experts and the single sigmoid-gated shared expert run in one op (six
kernels, no per-call allocation).

FP16 or BF16 activations with int8-packed 4-bit per-expert weights and
per-group scales (same dtype as the activations) are supported, with SiLU
activation and no bias. Per-group scales apply in-register in the shared
int4 mainloop. Padding expert IDs (-1) contribute zero to routed experts;
valid IDs must be distinct within each token's top-k routes.
The output is the per-rank partial sum (tensor-parallel reduction is left
to the caller).
"""

import torch

from vllm_xpu_kernels.moe_shared_fused_interface import (
    router_supports as _router_supports,
)

# Xe block-2D descriptors use 24-bit byte widths/pitches.
_MAX_BLOCK_2D_BYTES = (1 << 24) - 1
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
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    *,
    device: torch.device | None = None,
) -> bool:
    """Whether the int4 op supports this configuration.

    Defaults to the current XPU. Requires a device supported by the native
    Xe2/Xe3 dispatcher, FP16/BF16 activations, an int4 mainloop group size,
    positive dimensions, H a multiple of 256 and packed I/2 a multiple of
    4. The caller selects eligible token counts before invoking the op.
    Weight/scales dtypes and shapes are checked at the C++ entry.
    """
    if device is not None and device.type != "xpu":
        return False
    device_index = (-1 if device is None or device.index is None
                    else device.index)
    return (is_available()
            and torch.xpu.is_available()
            and hidden_states_dtype in (torch.float16, torch.bfloat16)
            and 0 < top_k <= 512
            and hidden_size > 0 and hidden_size % 256 == 0
            and hidden_size <= _MAX_BLOCK_2D_BYTES // 2
            and intermediate_size > 0 and intermediate_size % 8 == 0
            and intermediate_size <= _MAX_BLOCK_2D_BYTES // 4
            and group_size in _VALID_GROUP_SIZES
            and (torch.ops._xpu_C.is_xe2_arch(device_index)
                 or torch.ops._xpu_C.is_xe3_arch(device_index)))


def moe_shared_fused_int4_workspace_bytes(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    w13_n: int,
    w2_k: int,
) -> int:
    """Workspace size in bytes; must match the carve-out in the C++ op.

    Same activation-buffer geometry as the fp8 op (w13_n/w2_k are the
    logical, unpacked dims).
    """
    if not 0 < top_k <= 512:
        raise ValueError("top_k must be in [1, 512]")
    rows = num_tokens * top_k
    if (hidden_size <= 0 or w2_k <= 0 or w13_n != 2 * w2_k
            or hidden_size % 32 or w2_k % 8
            or hidden_size > _MAX_BLOCK_2D_BYTES // 2
            or w2_k > _MAX_BLOCK_2D_BYTES // 4):
        raise ValueError(
            "H and I must be positive multiples of 32 and 8 within the "
            "block-2D limits, and w13_n must equal 2 * I")
    # A hot expert can span multiple tiles; reserve one tile per routed row.
    umt = rows

    def align_workspace_offset(num_bytes: int) -> int:
        # Match the C++ op's align256 scratch-region boundaries.
        alignment_bytes = 256
        return ((num_bytes + alignment_bytes - 1) // alignment_bytes
                * alignment_bytes)

    int32_bytes = float32_bytes = 4
    activation_bytes = 2  # FP16/BF16 intermediates.
    # mt_expert, mt_row_base, mt_rows [umt], total_mt [1], perm_pos [rows].
    o_wperm = align_workspace_offset(int32_bytes * (3 * umt + 1 + rows))
    # Permuted routing weights [rows], then sigmoid shared gates [num_tokens].
    o_gate = align_workspace_offset(o_wperm + float32_bytes * rows)
    o_ap = align_workspace_offset(o_gate + float32_bytes * num_tokens)
    # a_perm [R,H], d1 [R+M,2I], act [R+M,I], d2 [R,H], sh2 [M,H].
    return o_ap + activation_bytes * (
        rows * hidden_size + (rows + num_tokens) * w13_n +
        (rows + num_tokens) * w2_k + rows * hidden_size +
        num_tokens * hidden_size)


class XpuMoESharedFusedDecodeInt4:
    """Holds the weights of one MoE block and a per-M persistent workspace.

    Weight layouts (all contiguous, [E, N, K] storage order, K packed):
      w13 [E, 2I, H/2] int8-packed 4-bit, w13_scales [E, 2I, H/G]
      w2  [E, H, I/2]  int8-packed 4-bit, w2_scales  [E, H, I/G]
    Shared experts are optional (None = routed-only; the caller then runs
    the dense shared MLP separately). When present:
      shared_w13 [2I, H/2] int8-packed, shared_w13_scale [1, 2I, H/G]
      shared_w2  [H, I/2]  int8-packed, shared_w2_scale  [1, H, I/G]
      shared_gate [H] or [1, H] with the activation dtype

    Matrices require 64-byte-aligned base addresses; shared_gate requires
    32-byte alignment. Contiguous storage-offset views do not necessarily
    meet those native requirements.

    tile_m selects the GEMM M-tile (8 or 16); 0 lets the op pick by M.

    Each instance is bound to its weights' device. Calls use the current XPU
    stream; calls with the same M share workspace and must be ordered. Use
    separate instances for overlapping calls on different streams.

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
        if tile_m not in (0, 8, 16):
            raise ValueError("tile_m must be 0 (auto), 8 or 16")
        self.device = w13.device
        if self.device.type != "xpu":
            raise ValueError("w13 must be an XPU tensor")
        shared = (shared_w13, shared_w13_scale, shared_w2, shared_w2_scale,
                  shared_gate)
        if any(t is None for t in shared) and not all(
                t is None for t in shared):
            raise ValueError(
                "shared experts are all-or-none: pass all five tensors "
                "or None for routed-only")
        self.has_shared = shared_w13 is not None
        tensors = [w13_scales, w2, w2_scales, router_weight]
        names = ["w13_scales", "w2", "w2_scales", "router_weight"]
        if self.has_shared:
            tensors.extend(shared)
            names.extend(("shared_w13", "shared_w13_scale", "shared_w2",
                          "shared_w2_scale", "shared_gate"))
        for name, tensor in zip(names, tensors):
            if tensor is not None and tensor.device != self.device:
                raise ValueError(
                    f"{name} must be on the same XPU device as w13")
        if w13.dim() != 3 or w2.dim() != 3:
            raise ValueError("routed weights must be rank three")
        num_experts, w13_n, w13_k_packed = w13.shape
        w2_n, w2_k_packed = w2.shape[1], w2.shape[2]
        if not (0 < num_experts <= 512 and 0 < top_k <= num_experts):
            raise ValueError("require 1 <= top_k <= num_experts <= 512")
        if (w2.shape[0] != num_experts or w13_k_packed * 2 != w2_n
                or w13_n != 4 * w2_k_packed):
            raise ValueError(
                "routed weight shapes must be w13 [E, 2I, H/2], "
                "w2 [E, H, I/2] packed")
        if (w13.dtype != torch.int8 or w2.dtype != torch.int8):
            raise ValueError("routed weights must be int8-packed 4-bit")
        for name, tensor in (("w13_scales", w13_scales),
                             ("w2_scales", w2_scales)):
            if (tensor.dtype not in (torch.float16, torch.bfloat16)
                    or tensor.dim() != 3
                    or tensor.shape[0] != num_experts):
                raise ValueError(
                    f"{name} must be [E, N, K/G] in fp16 or bf16")
        if (w13_scales.shape[1] != w13_n
                or w2_scales.shape[1] != w2_n):
            raise ValueError("routed scales must match their weight N dims")
        group_size = w13_k_packed * 2 // w13_scales.shape[2]
        if (w13_k_packed * 2 % w13_scales.shape[2] != 0
                or group_size not in _VALID_GROUP_SIZES
                or w2_k_packed * 2 // w2_scales.shape[2] != group_size):
            raise ValueError(
                "scales K/G must divide K with one group size of 32, 64, "
                "128 or 256 for w13 and w2")
        if self.has_shared:
            if (shared_w13.dim() != 2 or shared_w13.shape != (w13_n,
                                                              w13_k_packed)
                    or shared_w2.shape != (w2_n, w2_k_packed)):
                raise ValueError(
                    "shared weights must be [2I, H/2] and [H, I/2] packed")
            if (shared_w13_scale.dim() != 3 or shared_w2_scale.dim() != 3
                    or shared_w13_scale.shape[0] != 1
                    or shared_w2_scale.shape[0] != 1
                    or shared_w13_scale.shape[1:] != (w13_n,
                                                     w13_k_packed * 2
                                                     // group_size)
                    or shared_w2_scale.shape[1:] != (w2_n,
                                                    w2_k_packed * 2
                                                    // group_size)):
                raise ValueError(
                    "shared scales must be [1, N, K/G] matching the "
                    "routed group size")
            if (shared_w13.dtype != torch.int8
                    or shared_w2.dtype != torch.int8):
                raise ValueError(
                    "shared weights must be int8-packed 4-bit")
            if (shared_w13_scale.dtype not in (torch.float16, torch.bfloat16)
                    or shared_w2_scale.dtype
                    not in (torch.float16, torch.bfloat16)):
                raise ValueError(
                    "shared scales must be fp16 or bf16")
            if (shared_gate.dim() not in (1, 2)
                    or shared_gate.numel() != w2_n
                    or shared_gate.dtype
                    not in (torch.float16, torch.bfloat16)):
                raise ValueError(
                    "shared_gate must be [H] or [1, H] in fp16 or bf16")
        if any(not t.is_contiguous() for t in (w13, w13_scales, w2, w2_scales)
               + (() if not self.has_shared else
                  (shared_w13, shared_w13_scale, shared_w2, shared_w2_scale,
                   shared_gate))):
            raise ValueError("prepared weights and scales must be contiguous")
        # GEMM matrices use 64-byte-aligned block-2D loads. The shared gate
        # dot-product loads alignas(32) vectors of 16 FP16/BF16 elements.
        if any(t.data_ptr() % 64 for t in (w13, w2)
               + (() if not self.has_shared else (shared_w13, shared_w2))):
            raise ValueError("prepared weights require aligned base addresses")
        if self.has_shared and shared_gate.data_ptr() % 32:
            raise ValueError(
                "shared_gate requires a 32-byte-aligned base address")
        moe_shared_fused_int4_workspace_bytes(1, top_k, w2_n, w13_n,
                                              w2_k_packed * 2)
        self.w13 = w13
        self.w13_scales = w13_scales
        self.w2 = w2
        self.w2_scales = w2_scales
        self.shared_w13 = shared_w13
        self.shared_w13_scale = shared_w13_scale
        self.shared_w2 = shared_w2
        self.shared_w2_scale = shared_w2_scale
        self.shared_gate = shared_gate
        self.top_k = top_k
        self.num_experts = w13.shape[0]
        self.tile_m = tile_m
        self._ws: dict[int, torch.Tensor] = {}
        self.router_weight = router_weight
        self.renormalize = renormalize
        self._routing: dict[int, tuple[torch.Tensor, ...]] = {}

    def workspace(self, num_tokens: int) -> torch.Tensor:
        """Return persistent scratch on this instance's weights' device."""
        ws = self._ws.get(num_tokens)
        if ws is None:
            # [E, N, K] storage, K packed: hidden = N2, w13_n = N1.
            nbytes = moe_shared_fused_int4_workspace_bytes(
                num_tokens,
                self.top_k,
                self.w2.shape[1],
                self.w13.shape[1],
                self.w2.shape[2] * 2,
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
        import vllm_xpu_kernels._xpu_C  # noqa: F401  (fused grid op)
        if hidden_states.device != self.device:
            raise ValueError(
                "hidden_states must be on the same XPU device as w13")
        ws = self.workspace(hidden_states.shape[0])
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
        if self.router_weight is None:
            raise ValueError("no router_weight given")
        import vllm_xpu_kernels._moe_C  # noqa: F401  (router op)
        if not _router_supports(
            self.router_weight.dtype,
            self.num_experts,
            self.top_k,
            hidden_states.shape[1],
        ):
            raise ValueError("router op does not support this shape")
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
