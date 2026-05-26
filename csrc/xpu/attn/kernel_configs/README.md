# Kernel Configuration Files

This directory contains `.conf` files that control which attention kernel
variants are compiled.

| File | Kernels | Use Case |
|------|---------|----------|
| `chunk_prefill_full.conf` | 216 | Chunk prefill — all combinations |
| `chunk_prefill_default.conf` | 9 | Chunk prefill — Llama, Qwen, DeepSeek MLA |
| `paged_decode_full.conf` | 384 | Paged decode — all combinations |
| `paged_decode_default.conf` | 11 | Paged decode — Llama, Qwen, DeepSeek MLA |

For config file format, usage examples, model-specific guidance, and
troubleshooting, see **[KERNEL_CONFIGURATION.md](../../../../KERNEL_CONFIGURATION.md)**
at the repository root.
