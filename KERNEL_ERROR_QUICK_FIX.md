# Quick Kernel Configuration Troubleshooting

If you encounter an error like:

```
❌ Chunk prefill kernel tuple not compiled for this configuration.
```

**This page gives you instant solutions.**

---

## 30-Second Fix

```bash
# Recompile with all kernels enabled
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf pip install .
```

This solves ~95% of kernel errors. It takes ~30 minutes to build but supports all models.

---

## If You Need Faster Builds

For faster compilation, match your model's configuration:

### Llama / Qwen / Mistral

```bash
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf pip install .
# ~5 min build time
```

### DeepSeek

```bash
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_deepseek.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_deepseek.conf pip install .
# ~3 min build time
```

### Mixed / Custom Models
See [KERNEL_CONFIGURATION.md](KERNEL_CONFIGURATION.md) for custom configuration.

---

## What's Going On?

vLLM-XPU selectively compiles attention kernels based on configuration. This reduces build time and binary size, but means you need to recompile if your model uses a kernel combination that wasn't built.

**Kernel parameters that matter:**
- **head_size**: 64, 96, 128, 192, 256, 512 (depends on your model)
- **causal**: Usually `true` for language models
- **paged**: `true` for KV cache paging (decode optimization)
- **local**: `true` if model uses sliding window attention
- **sink**: `true` if model uses sink tokens
- **lse**: Log-Sum-Exp merge (for distributed attention)

---

## Still Having Issues?

Read the full guide: [KERNEL_CONFIGURATION.md](KERNEL_CONFIGURATION.md)

---

## Build Configuration Variables

```bash
# Available config files (in csrc/xpu/attn/kernel_configs/)
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf|chunk_prefill_default.conf|chunk_prefill_custom.conf
VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf|paged_decode_default.conf|paged_decode_custom.conf

# Example: Build with defaults
pip install .

# Example: Build with custom configs
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_custom.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_custom.conf pip install .
```
