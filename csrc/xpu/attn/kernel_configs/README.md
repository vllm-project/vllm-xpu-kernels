# Kernel Configuration Files

This directory contains `.conf` files that control which attention kernel
variants are compiled.

| File | Kernels | Use Case |
|------|---------|----------|
| `chunk_prefill_full.conf` | 180 | Chunk prefill — all combinations |
| `chunk_prefill_default.conf` | 49 | Chunk prefill — Llama, Qwen, DeepSeek MLA, Falcon, Gemma, Phi, GLM |
| `paged_decode_full.conf` | 384 | Paged decode — all combinations |
| `paged_decode_default.conf` | 32 | Paged decode — Llama, Qwen, DeepSeek MLA, Falcon, Starcoder2, Phi, VLM2Vec |

For config file format, usage examples, model-specific guidance, and
troubleshooting, see **[KERNEL_CONFIGURATION.md](../../../../KERNEL_CONFIGURATION.md)**
at the repository root.

Note: every chunk prefill config line has seven required fields and maps to
exactly one kernel, so the 49 lines in `chunk_prefill_default.conf` produce 49
kernels. The trailing `b16` field picks the tile policy (`chunk_policy_head<N>`
or `chunk_policy_head<N>_b16`). Paged decode config lines map one-to-one to
kernels.

The `all` keyword in `chunk_prefill_full.conf` expands to every valid
combination except `b16` with `paged=false`: `fmha_xe2.cpp` only selects a
`_b16` policy when the attention is paged with block size 16, so those kernels
could never be dispatched.
