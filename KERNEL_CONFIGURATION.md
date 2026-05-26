# vLLM-XPU Kernel Configuration Guide

When running vLLM with XPU kernels, you may encounter errors like:

```
❌ Chunk prefill kernel tuple not compiled for this configuration.
```

This guide explains how to configure which kernels are compiled and how to fix missing kernel configurations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Kernel Types](#kernel-types)
3. [Configuration Presets](#configuration-presets)
4. [Bool Combinations](#bool-combinations)
5. [Custom Configuration](#custom-configuration)
6. [Build & Install](#build--install)

---

## Quick Start

### If you see a kernel missing error:

**Option A: Use the full preset (recommended for first-time setup)**

```bash
# Recompile with all kernels enabled
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf pip install .
```

**Option B: Customize to your models**
1. Identify your model's head_size and required bool combinations (see section below)
2. Create a custom config file in `csrc/xpu/attn/kernel_configs/`
3. Rebuild with that config

---

## Kernel Types

vLLM-XPU has two main kernel categories:

### 1. **Chunk Prefill** (Prompt Processing)
- Used when processing prompt tokens
- Configuration: `VLLM_CHUNK_PREFILL_CONFIG`
- Bool parameters: `paged`, `causal`, `local`, `sink`, `lse`

### 2. **Paged Decode** (Token Generation)
- Used when generating tokens one-by-one
- Configuration: `VLLM_PAGED_DECODE_CONFIG`
- Bool parameters: `causal`, `local`, `sink`

---

## Configuration Presets

Configuration files are located in: `csrc/xpu/attn/kernel_configs/`

### Chunk Prefill Presets

| Preset | File | Kernels | Use Case |
|--------|------|---------|----------|
| `full` | `chunk_prefill_full.conf` | 216 | All combinations |
| `default` | `chunk_prefill_default.conf` | 9 | Llama, Qwen, DeepSeek MLA (default) |

### Paged Decode Presets

| Preset | File | Kernels | Use Case |
|--------|------|---------|----------|
| `full` | `paged_decode_full.conf` | 384 | All combinations |
| `default` | `paged_decode_default.conf` | 11 | Llama, Qwen, DeepSeek MLA (default) |

### Recommended Combinations

| Model Family | Head Sizes | Chunk Prefill Config | Paged Decode Config |
|--------------|-----------|--------|----------|
| Llama 2/3, Qwen, Mistral | 128 | `default` | `default` |
| DeepSeek (MLA) | 192 | `default` | `default` |
| Mixed / other models | Multiple | `full` | `full` |

---

## Bool Combinations

### Chunk Prefill: 5 Bool Parameters

```
(paged, causal, local, sink, lse)
```

**Constraints:**
- `lse` (Log-Sum-Exp) requires: `paged=false`, `local=false`, `sink=false`
  - Used for distributed attention merging (chunked prefill states)
  - Cannot combine with paging, sliding window (local), or sink tokens

| paged | causal | local | sink | lse | Valid? | Use Case |
|-------|--------|-------|------|-----|--------|----------|
| true | true | false | false | false | ✅ | Paged KV cache (decode-like) |
| false | true | false | false | false | ✅ | Initial prompt, no paging |
| false | true | false | false | true | ✅ | Chunked prefill state merge |
| false | true | true | false | false | ✅ | Sliding window attention |
| false | true | true | true | false | ✅ | Sliding window + sink tokens |
| false | true | false | true | false | ✅ | Sink token optimization |
| true | true | true | \* | \* | ❌ | LSE incompatible |
| *|* | *|* | true | ❌ | LSE requires constraints |

### Paged Decode: 3 Bool Parameters

```
(causal, local, sink)
```

**Typical combinations:**
- `(true, false, false)` - Standard causal (most common)
- `(true, true, false)` - Sliding window
- `(true, true, true)` - Sliding window + sink tokens

---

## Custom Configuration

### Step 1: Identify Your Requirements

Determine what `head_size` and bool combinations you need:

```python
# Pseudo-code to identify requirements
for model in models:
    head_size = model.hidden_size // model.num_heads
    causal = True  # Most models use causal attention
    local = model.has_sliding_window  # e.g., Mistral, Qwen
    sink = model.has_sink_tokens  # e.g., some attention variants
    paged = True  # Usually enable paged decode
    lse = model.uses_distributed_attention  # e.g., multi-GPU prefill merge
```

### Step 2: Create a Config File

Create `csrc/xpu/attn/kernel_configs/chunk_prefill_custom.conf`:

```conf
# Example: Llama + DeepSeek + Mistral
# Format: head_size,paged,causal,local,sink,lse

# Llama / Qwen (head_size=128)
128,true,true,false,false,false
128,false,true,false,false,false
128,false,true,false,false,true

# Mistral with sliding window (head_size=128)
128,false,true,true,false,false

# DeepSeek MLA (head_size=192)
192,true,true,false,false,false
192,false,true,false,false,false
192,false,true,false,false,true
```

For paged_decode, create `csrc/xpu/attn/kernel_configs/paged_decode_custom.conf`:

```conf
# Format: head_size,causal,local,sink

# Standard causal (most models)
128,true,false,false
192,true,false,false

# With sliding window
128,true,true,false
192,true,true,false
```

### Step 3: Rebuild with Custom Config

```bash
# Chunk prefill with custom config
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_custom.conf pip install .

# Or both at once
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_custom.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_custom.conf pip install .
```

---

## Build & Install

### Basic Installation

```bash
# Default (builds all kernel variants — full config)
pip install .

# With explicit full config (same as above)
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf pip install .

# Optimized build (Llama/Qwen/DeepSeek only, ~97% fewer kernels)
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf pip install .

# With custom configs
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_custom.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_custom.conf pip install .
```

### What Preset Should I Use?

**Full build (default — all combinations, all models):**

```bash
pip install .
# Or explicitly:
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf pip install .
```

- Compiles all kernels (216 for chunk prefill, 384 for paged decode)
- Build time: ~30 minutes
- Supports all models without recompilation

**Optimized build (Llama/Qwen/DeepSeek, ~97% fewer kernels):**

```bash
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf pip install .
```

- Compiles only 6 kernels per kernel type
- Build time: ~2 minutes
- If a missing-kernel error occurs, rebuild with `full` or a custom config

---

## Troubleshooting

### Error: "Chunk prefill kernel not compiled for this configuration"

**Problem:** The head_size for your model is not in the config.

**Solution:**

```bash
# Option 1: Use full config
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf pip install .

# Option 2: Add your head_size to a config file and rebuild
# Edit csrc/xpu/attn/kernel_configs/chunk_prefill_custom.conf and add:
#   your_head_size,true,true,false,false,false
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_custom.conf pip install .
```

### Error: "Chunk prefill kernel tuple not compiled for this configuration"

**Problem:** The bool combination (paged/causal/local/sink/lse) for your model is not compiled.

**Solution:**

```bash
# Option 1: Use full config
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf pip install .

# Option 2: Add the bool combination to your config file
# Edit csrc/xpu/attn/kernel_configs/chunk_prefill_custom.conf and add:
#   head_size,your_paged,your_causal,your_local,your_sink,your_lse
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_custom.conf pip install .
```

### How do I check which configs are compiled?

Check the CMake output during build:

```bash
# Look for lines like:
# -- Generated chunk_prefill kernel sources: 6 files 
#    (config: .../chunk_prefill_default.conf)
# -- Generated paged_decode kernel sources: 384 files 
#    (config: .../paged_decode_full.conf)
```

Or inspect the generated header:

```bash
cat build/temp_template/csrc/xpu/attn/xe_2/chunk_prefill_enabled_policies_gen.hpp
```

---

## Performance Notes

### Build Time vs Runtime Flexibility

| Config | Build Time | Kernels | Flexibility |
|--------|-----------|---------|-------------|
| `default` | ~2 min | ~6 | Llama, Qwen, DeepSeek MLA |
| `full` | ~30 min | ~216+ | All models |

### Binary Size Impact

- Each unique (head_size, bool-combination) → ~500KB-2MB compiled kernel
- `default`: ~3MB | `full`: ~100MB+

### Recommendation

- **Development / CI**: Use `full` for broad compatibility
- **Production (known models)**: Use `default` or a custom config for faster builds

---

## Further Reading

- For implementation details: See `csrc/xpu/attn/xe_2/chunk_prefill_configure.cmake`
- Config file format: See `csrc/xpu/attn/kernel_configs/chunk_prefill_full.conf`
- Runtime error paths: See `csrc/xpu/attn/xe_2/chunk_prefill_utils.hpp`
