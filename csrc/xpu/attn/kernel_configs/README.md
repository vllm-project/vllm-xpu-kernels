# Kernel Configuration Files

This directory contains `.conf` files that control which attention kernel
variants are compiled.

| File | Kernels | Use Case |
|------|---------|----------|
| `chunk_prefill_full.conf` | 240 | Chunk prefill — all combinations |
| `chunk_prefill_default.conf` | 70 | Chunk prefill — Llama, Qwen, DeepSeek MLA, Falcon, Gemma, Phi, GLM |
| `paged_decode_full.conf` | 384 | Paged decode — all combinations |
| `paged_decode_default.conf` | 32 | Paged decode — Llama, Qwen, DeepSeek MLA, Falcon, Starcoder2, Phi, VLM2Vec |

For config file format, usage examples, model-specific guidance, and
troubleshooting, see **[KERNEL_CONFIGURATION.md](../../../../KERNEL_CONFIGURATION.md)**
at the repository root.

Note: each chunk prefill config line expands to two kernels (the standard and
`_b16` policy for that head size), so the 35 lines in
`chunk_prefill_default.conf` produce 70 kernels. Paged decode config lines map
one-to-one to kernels.
