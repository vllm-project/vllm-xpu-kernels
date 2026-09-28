# SPDX-License-Identifier: Apache-2.0
"""Optional routed-expert adapter used by the guarded XpuFusedMoe path.

The adapter accepts the existing model call-site arguments and writes into
its supplied output buffer. Shared-expert arguments to the native kernel
are always None, preserving the separate model-level shared-expert path.

The supported geometry is H=2048, I_local=256, E=256 and top_k=8, with FP16
activations, FP8 E4M3 per-expert scalar weights and SiLU. The caller owns
shape and batch-size eligibility checks. This iteration validates M=1.
"""

import os

import torch

_SO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "libgrouped_gemm_xe_2.so",
)
if os.path.exists(_SO):
    torch.ops.load_library(_SO)

_DEFAULT_POLICY_ID = 0  # w8a16_policy_m_16; measured indistinguishable from
                        # policy 1 (w8a16_policy_m_8) across M=1..8 in the
                        # impl3 sweep -- fixed internally, not caller-visible.


class XpuFusedMoeV4:
    def __init__(
        self,
        w13,
        w13_scales,
        w13_bias,
        w2,
        w2_scales,
        w2_bias,
        n_experts_per_token,
        activation,
        num_experts,
        ep_rank=0,
        ep_size=1,
        expert_map=None,
        gemm1_clamp_limit=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    ):
        # Same contiguity/support contract as XpuFusedMoe.__init__.
        assert w13.is_contiguous() and w2.is_contiguous()
        assert w13_bias is None and w2_bias is None, "bias not supported"
        assert (w13.dtype == torch.float8_e4m3fn
                and w2.dtype == torch.float8_e4m3fn), (
                    "specialized op supports fp8-e4m3 per-tensor weights only")
        assert (
            w13_scales is not None
            and w13_scales.dtype == torch.float32
            and w13_scales.dim() == 1
        ), "fp8 per-tensor(per-expert) scales required"
        assert activation == "silu", "specialized op hardcodes SiLU (Qwen3.5)"
        assert ep_size == 1 and expert_map is None, "EP not supported"
        assert gemm1_clamp_limit is None, "clamp not supported"

        self.w13 = w13
        self.w13_scales = w13_scales
        self.w2 = w2
        self.w2_scales = w2_scales
        self.n_experts_per_token = n_experts_per_token
        self.activation = activation
        self.num_experts = num_experts
        self.inter_size = w13.shape[-1] // 2
        self.recipe = "fp8"

        self._ws_cache = {}

    def _get_workspace(self, m, device):
        """Persistent per-batch-size workspace, internal implementation
        detail (not part of the public contract -- XpuFusedMoe callers
        never see or manage this)."""
        key = (m, str(device))
        ws = self._ws_cache.get(key)
        if ws is None:
            R = m * self.n_experts_per_token
            E = self.num_experts
            umt = min(R, E)
            K1, N1 = self.w13.shape[1], self.w13.shape[2]
            K2, N2 = self.w2.shape[1], self.w2.shape[2]
            align = lambda b: (b + 255) // 256 * 256
            o_int = 0
            o_wperm = align(o_int + 4 * (3 * umt + 1 + R))
            o_gate = align(o_wperm + 4 * R)
            o_ap = align(o_gate + 4 * m)
            total = (o_ap + 2 * (R * K1 + (R + m) * N1 + (R + m) * K2
                                 + R * N2 + m * N2))
            ws = torch.empty(total, dtype=torch.uint8, device=device)
            self._ws_cache[key] = ws
        return ws

    def apply(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
        a1q_scale=None,
    ):
        assert expert_map is None and a1q_scale is None, (
            "specialized op supports the plain per-tensor-fp8 path only"
        )
        ws = self._get_workspace(output.shape[0], output.device)
        return torch.ops.moe_qwen35_sp.fused_forward(
            output, hidden_states, topk_ids, topk_weights,
            self.w13, self.w13_scales, self.w2, self.w2_scales,
            None, None, None, None, None,  # shared_* -- always disabled
            ws, _DEFAULT_POLICY_ID,
        )
