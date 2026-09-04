# Kernel Configuration Files

This directory contains `.conf` files that control which attention kernel
variants are compiled.

| File | Kernels | Use Case |
|------|---------|----------|
| `chunk_prefill_full.conf` | 240 | Chunk prefill — all combinations |
| `chunk_prefill_default.conf` | 49 | Chunk prefill — Llama, Qwen, DeepSeek MLA, Falcon, Gemma, Phi, GLM |
| `paged_decode_full.conf` | 384 | Paged decode — all combinations |
| `paged_decode_default.conf` | 32 | Paged decode — Llama, Qwen, DeepSeek MLA, Falcon, Starcoder2, Phi, VLM2Vec |

For config file format, usage examples, model-specific guidance, and
troubleshooting, see **[KERNEL_CONFIGURATION.md](../../../../KERNEL_CONFIGURATION.md)**
at the repository root.

Note: a chunk prefill config line that omits the trailing `b16` field expands
to two kernels (the standard and `_b16` policy for that head size). Adding an
explicit `b16` value selects a single policy, so each such line produces one
kernel. `chunk_prefill_default.conf` is fully explicit and its 49 lines produce
49 kernels. Paged decode config lines map one-to-one to kernels.
