# MiniCPM5 native operators

The `_minicpm5_mono_C` extension registers the `minicpm5_mono` operator
namespace. Xe2 builds include it by default. Set
`MINICPM5_MONO_KERNELS_ENABLED=OFF` to omit it. Running the operators requires
a supported Battlemage device.

| Operators | Purpose |
| --- | --- |
| `validate`, `decode` | Validate and run a resident target layer with FP8 weights and FP16 or BF16 activations in TP1 or TP2. |
| `draft_decode` | Run a seven-query FP16 draft layer in TP1 or TP2. |
| `candidate_topk`, `candidate_greedy`, `candidate_greedy_bias` | Exchange vocabulary candidates between TP2 peers while preserving global token IDs. |
| `abi_version`, `supported`, `control_words`, `health_offset`, `mailbox_bytes` | Check native capabilities and allocate per-layer control and peer storage. |
| `router_logits_dtype` | Report the precision of the target router logits before sigmoid. |

The Python loader checks the extension ABI on first use. The standard-decode
tuning targets 1–8 live requests per step at TP1 or TP2. The checkpoint-free
tests in [tests/minicpm5](tests/minicpm5) exercise target, draft, and peer math.
