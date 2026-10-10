# Fused MoE decode kernel

| Operator | Semantics and supported inputs | Calling contract |
| --- | --- | --- |
| `_xpu_C.moe_shared_fused_decode_interface` | Routed experts plus one sigmoid-gated shared expert; FP16/BF16 activations, FP8 E4M3 per-tensor weights, FP32 scales, SiLU; Xe2/Xe3 | Output and workspace are mutable. Layout, dtype and shape checks precede submission on the current XPU stream. The caller performs routing and tensor-parallel reduction. |

[`XpuMoESharedFusedDecode`](vllm_xpu_kernels/moe_shared_fused_interface.py)
holds weights and a workspace for each token count. Calls sharing a workspace
must be ordered; overlapping calls on different streams require separate
instances. `is_available()` reports whether the native operator was built.
