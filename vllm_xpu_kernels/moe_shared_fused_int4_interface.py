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

# Symmetric-AutoRound/GPTQ repack adapter. Converts the checkpoint layout
# that vLLM's moe_wna16 GPTQ branch consumes into the grid's int8-packed
# signed nibbles + per-group scales. AWQ checkpoints need no adapter (the
# grid already indexes their standard form).
_AUTOROUND_GROUP_SIZES = (32, 64, 128, 256)


def _autoround_zeros_symmetric(qzeros_2d: torch.Tensor, ng: int, n: int) -> None:
    if qzeros_2d.shape != (ng, n // 8):
        raise ValueError("autoround qzeros must be [K/G, N/8]")
    # Symmetric quant stores 7 in every nibble; the stock loader adds 1 to
    # align with AWQ (effective zero 8), which the XOR below folds in.
    zu = qzeros_2d.to(torch.int64) & 0xFFFFFFFF
    for k in range(8):
        if not bool((((zu >> (4 * k)) & 0xF) == 7).all()):
            raise ValueError(
                "autoround qzeros must be symmetric (all nibbles 7)")


def _autoround_order(g_idx: torch.Tensor, k: int,
                     group_size: int) -> torch.Tensor | None:
    if g_idx.dim() != 1 or g_idx.shape[0] != k:
        raise ValueError("autoround g_idx must be [K]")
    want = torch.arange(k, dtype=g_idx.dtype, device=g_idx.device) // group_size
    if torch.equal(g_idx, want):
        return None  # Identity groups: K already ordered (Ornith shape).
    order = torch.argsort(g_idx, stable=True)
    if not torch.equal(g_idx[order], want):
        raise ValueError("autoround g_idx must partition K into full groups")
    return order


def autoround_to_grid(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one symmetric-AutoRound/GPTQ linear to grid int4 tensors.

    Accepts a single [K/8, N] int32 linear or E-stacked [E, K/8, N];
    qzeros/g_idx may be shared ([K/G, N/8] / [K]) or stacked like qweight.
    Returns (w [N, K/2] (or [E, N, K/2]) int8-packed signed nibbles with
    the low nibble holding the even-K element, scales [N, K/G] (or
    [E, N, K/G]) transposed but otherwise untouched, same dtype/device).
    Raises ValueError on asymmetric zeros, group sizes outside the grid's
    (32, 64, 128, 256) set, or unexpected shapes/dtypes; the caller sees
    the stock loader's layout otherwise.
    """
    batched = qweight.dim() == 3
    if not batched and qweight.dim() != 2:
        raise ValueError("autoround qweight must be [K/8, N] or [E, K/8, N]")
    if qweight.dtype != torch.int32:
        raise ValueError("autoround qweight must be int32-packed")
    e = qweight.shape[0] if batched else None
    k_packed, n = qweight.shape[-2], qweight.shape[-1]
    k = k_packed * 8
    if n % 8:
        raise ValueError("autoround N must be a multiple of 8")
    # The group size is inferred from the scales, as in the stock loader.
    ng_hint = scales.shape[-2] if scales.dim() in (2, 3) else 0
    group_size = k // ng_hint if ng_hint and not k % ng_hint else 0
    if group_size not in _AUTOROUND_GROUP_SIZES or k % group_size:
        raise ValueError("autoround scales must imply a grid group size "
                         "of 32, 64, 128 or 256")
    ng = k // group_size
    flat_qw = qweight.reshape(-1, k_packed, n) if batched else qweight.unsqueeze(0)
    if scales.dim() == 2:
        if scales.shape != (ng, n):
            raise ValueError("autoround scales must be [K/G, N]")
        if scales.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("autoround scales must be fp16 or bf16")
        flat_sc = scales.unsqueeze(0).expand(flat_qw.shape[0], ng, n)
    elif scales.dim() == 3 and (e is None or scales.shape[0] == flat_qw.shape[0]):
        if scales.shape[-2:] != (ng, n):
            raise ValueError("autoround scales must be [K/G, N] per expert")
        if scales.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("autoround scales must be fp16 or bf16")
        flat_sc = scales
    else:
        raise ValueError("autoround scales must be [K/G, N] or [E, K/G, N]")
    if qzeros.dim() == 2:
        _autoround_zeros_symmetric(qzeros, ng, n)
    elif qzeros.dim() == 3 and qzeros.shape[0] == flat_qw.shape[0]:
        for i in range(qzeros.shape[0]):
            _autoround_zeros_symmetric(qzeros[i], ng, n)
    else:
        raise ValueError("autoround qzeros must be [K/G, N/8] or [E, K/G, N/8]")
    order = _autoround_order(g_idx, k, group_size)
    # Stock GPTQ branch: qweight.T viewed as bytes is already the AWQ
    # standard (low nibble = even k); XOR 0x88 folds the -8 zero point
    # into plain 4-bit two's-complement signed nibbles for the grid.
    u8 = flat_qw.transpose(-1, -2).contiguous().view(torch.uint8)
    u8 = u8.reshape(flat_qw.shape[0], n, k // 2)
    if order is not None:
        lo = (u8 & 0xF)
        hi = ((u8 >> 4) & 0xF)
        u = torch.stack([lo, hi], dim=-1).reshape(flat_qw.shape[0], n, k)
        u = u[:, :, order]
        u8 = (u[:, :, 0::2] | (u[:, :, 1::2] << 4)).to(torch.uint8)
    w = (u8 ^ 0x88).to(torch.int8)
    s = flat_sc.transpose(-1, -2).contiguous()
    if not batched:
        w, s = w[0], s[0]
    return w, s


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


    @classmethod
    def from_autoround(
        cls,
        gate_qweight: torch.Tensor,
        gate_scales: torch.Tensor,
        gate_qzeros: torch.Tensor,
        gate_g_idx: torch.Tensor,
        up_qweight: torch.Tensor,
        up_scales: torch.Tensor,
        up_qzeros: torch.Tensor,
        up_g_idx: torch.Tensor,
        down_qweight: torch.Tensor,
        down_scales: torch.Tensor,
        down_qzeros: torch.Tensor,
        down_g_idx: torch.Tensor,
        shared_gate_qweight: torch.Tensor | None = None,
        shared_gate_scales: torch.Tensor | None = None,
        shared_gate_qzeros: torch.Tensor | None = None,
        shared_gate_g_idx: torch.Tensor | None = None,
        shared_up_qweight: torch.Tensor | None = None,
        shared_up_scales: torch.Tensor | None = None,
        shared_up_qzeros: torch.Tensor | None = None,
        shared_up_g_idx: torch.Tensor | None = None,
        shared_down_qweight: torch.Tensor | None = None,
        shared_down_scales: torch.Tensor | None = None,
        shared_down_qzeros: torch.Tensor | None = None,
        shared_down_g_idx: torch.Tensor | None = None,
        shared_gate: torch.Tensor | None = None,
        router_weight: torch.Tensor | None = None,
        top_k: int = 8,
        tile_m: int = 0,
        renormalize: bool = True,
        device: torch.device | None = None,
    ) -> "XpuMoESharedFusedDecodeInt4":
        # Build a routed-path instance from E-stacked AutoRound tensors.
        # gate/up_qweight [E, H/8, I], down_qweight [E, I/8, H], scales
        # [E, K/G, N] (or shared 2D); each linear carries its own
        # qzeros/g_idx, shared or stacked (see autoround_to_grid). Shared
        # experts convert the same way when all twelve shared quantized
        # tensors plus the dense shared_gate are given, or all None for
        # routed-only. Converted weights move to `device` (default:
        # router_weight's device, else the converted device, which must be
        # XPU); alignment/dtype/shape checks stay in __init__.
        w_gate, s_gate = autoround_to_grid(gate_qweight, gate_scales,
                                           gate_qzeros, gate_g_idx)
        w_up, s_up = autoround_to_grid(up_qweight, up_scales,
                                       up_qzeros, up_g_idx)
        w_down, s_down = autoround_to_grid(down_qweight, down_scales,
                                           down_qzeros, down_g_idx)
        w13 = torch.cat([w_gate, w_up], dim=-2).contiguous()
        s13 = torch.cat([s_gate, s_up], dim=-2).contiguous()
        shared_q = (shared_gate_qweight, shared_gate_scales,
                    shared_gate_qzeros, shared_gate_g_idx,
                    shared_up_qweight, shared_up_scales,
                    shared_up_qzeros, shared_up_g_idx,
                    shared_down_qweight, shared_down_scales,
                    shared_down_qzeros, shared_down_g_idx)
        if all(t is None for t in shared_q) and shared_gate is None:
            sw13 = ss13 = sw2 = ss2 = gate = None
        elif any(t is None for t in (*shared_q, shared_gate)):
            raise ValueError(
                "shared experts are all-or-none: pass all twelve quantized "
                "tensors plus the dense shared_gate, or None for routed-only")
        else:
            sw_gate, ss_gate = autoround_to_grid(
                shared_gate_qweight, shared_gate_scales,
                shared_gate_qzeros, shared_gate_g_idx)
            sw_up, ss_up = autoround_to_grid(
                shared_up_qweight, shared_up_scales,
                shared_up_qzeros, shared_up_g_idx)
            sw_down, ss_down = autoround_to_grid(
                shared_down_qweight, shared_down_scales,
                shared_down_qzeros, shared_down_g_idx)
            if sw_gate.dim() != 2:
                raise ValueError("shared quantized tensors must be 2D")
            sw13 = torch.cat([sw_gate, sw_up], dim=-2).contiguous()
            ss13 = torch.cat([ss_gate, ss_up], dim=-2).unsqueeze(0)
            ss13 = ss13.contiguous()
            sw2 = sw_down.contiguous()
            ss2 = ss_down.unsqueeze(0).contiguous()
            gate = shared_gate
        if device is None:
            device = (router_weight.device if router_weight is not None
                      else w13.device)
        if device.type != "xpu":
            raise ValueError("from_autoround target device must be XPU")
        return cls(w13.to(device), s13.to(device), w_down.to(device),
                   s_down.to(device),
                   None if sw13 is None else sw13.to(device),
                   None if ss13 is None else ss13.to(device),
                   None if sw2 is None else sw2.to(device),
                   None if ss2 is None else ss2.to(device),
                   None if gate is None else gate.to(device),
                   top_k, tile_m=tile_m,
                   router_weight=router_weight, renormalize=renormalize)

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
